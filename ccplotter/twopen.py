"""ccplotter/twopen.py -- opt-in TWO-PEN job (backend plumbing only).

Takes two ALREADY-GENERATED stroke layers (SVG in mm page coordinates: a "blue"
layer and a "black" layer) and produces
  * two COMPLETE gcode files, one per pen profile, each through the same
    placement -> linesort -> gwrite(elegoo_cc) -> microlift -> smoothing ->
    reach-validate chain a normal job uses, with its own preamble/postamble;
  * one two-colour preview SVG (blue under black);
  * one run record holding both gcodes, each stamped with its own back-link.

Layer GENERATION from a photo is not this module's job. Nothing here touches
_composite, so existing modes' output is unchanged.

Safety contract
  * z_up / z_down / z_micro / z_touch / speeds / XY offset / reach clamp come from
    EACH PEN PROFILE, never from the request body (a normal job takes z_down from the
    body; that is deliberately NOT done here).
  * The vpype global config (gwrite.elegoo_cc.default_values) is process-wide
    mutable state. Each pass sets it, and it is RESTORED afterwards (finally), so a
    normal job run after a two-pen job is unaffected.
  * No pen-change / pause codes are emitted: each file is a full, normal-lifecycle
    print. The pass header is an ASCII comment only.
  * Refusals (TwoPenRefusal -> HTTP 4xx): unknown pen id, same pen twice, unreadable
    or empty layer, reach failure, uncalibrated z_touch.
"""
import copy
import re
import shlex
from pathlib import Path
from uuid import uuid4

import vpype as vp
import vpype_cli

from ccplotter.atomic_io import atomic_target
from ccplotter.cal import (load_cal, load_pen_profiles, resolve_pen_offset,
                           resolve_pen_profile, resolve_z_micro, resolve_z_touch,
                           UnknownPenProfile, ZTouchNotCalibrated)
from ccplotter.config import CONFIG_PATH, PAPER_SIZES, PREVIEW_COLOR, WORK_DIR
from ccplotter.gcode_post import (_apply_motion_smoothing, _fix_gwrite_source_comment,
                                  _gcode_lifecycle_values, _gcode_pendown_bbox,
                                  _microlift_gcode)
from ccplotter.reach import (active_reach_profile, pen_reach_profile,
                             validate_gcode_xy_motion)
from ccplotter.svg_io import _paper_spec
from ccplotter.composite import preview_pen_width_mm

BLUE_PREVIEW_COLOR = "#1f4fd8"
BLACK_PREVIEW_COLOR = PREVIEW_COLOR          # same ink colour a normal preview uses
TWOPEN_DIRNAME = "twopen"

PASS_HEADERS = {
    1: "PASS 1 OF 2: BLUE -- after this file, swap to the black pen, re-check z, do not move the paper",
    2: "PASS 2 OF 2: BLACK -- final pass; the paper must not have moved since pass 1, re-check z for this pen",
}


class TwoPenRefusal(Exception):
    """A refusal the route turns into a 4xx (never a 500)."""
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------- inputs
def _parse_paper(spec):
    spec = str(spec or "b5").lower()
    try:
        if spec in PAPER_SIZES:
            return PAPER_SIZES[spec]
        w, h = spec.split("x")
        w, h = float(w), float(h)
    except Exception:
        raise TwoPenRefusal("Invalid paper size")
    if w <= 0 or h <= 0:
        raise TwoPenRefusal("Invalid paper size")
    return w, h


def _materialise_layer(value, dest: Path, label: str) -> Path:
    """`value` is an SVG file path or inline SVG content. Returns a file path."""
    if not isinstance(value, str) or not value.strip():
        raise TwoPenRefusal(f"{label}_svg is required (SVG path or content)")
    v = value.strip()
    if v.startswith("<"):
        check_svg_mm(v, label)
        dest.write_text(v, encoding="utf-8")
        return dest
    p = Path(v)
    if p.suffix.lower() != ".svg" or not p.is_file():
        raise TwoPenRefusal(f"{label}_svg is not an existing .svg file")
    check_svg_mm(p.read_text(encoding="utf-8", errors="replace")[:4096], label)
    return p


def resolve_pens(body):
    """(blue_id, blue_profile, black_id, black_profile) or TwoPenRefusal."""
    bid, kid = body.get("blue_pen_id"), body.get("black_pen_id")
    if not bid or not kid:
        raise TwoPenRefusal("blue_pen_id and black_pen_id are both required")
    if bid == kid:
        raise TwoPenRefusal("blue_pen_id and black_pen_id must differ (two different pens)")
    try:
        return bid, resolve_pen_profile(bid), kid, resolve_pen_profile(kid)
    except UnknownPenProfile as e:
        raise TwoPenRefusal(str(e))


def _need(profile, key, pen_id):
    v = (profile or {}).get(key)
    if v is None:
        raise TwoPenRefusal(f"pen profile '{pen_id}' has no {key}; refusing to guess it for a two-pen job")
    return float(v)


# --------------------------------------------------------------------------- placement
def place_layer(layer_svg: Path, out_svg: Path, paper_w, paper_h, scale=1.0, rot=0.0,
                dx=0.0, dy=0.0):
    """Apply the SAME explicit transform to a layer (no auto-fit: both layers must
    register), pin the page, write the placed SVG. Returns the stroke count."""
    page_orient = "-l " if paper_w > paper_h else ""
    with atomic_target(out_svg) as tmp:
        vpype_cli.execute(f"read -l 1 {shlex.quote(str(layer_svg))} scale -l 1 {scale} {scale} "
                          f"rotate -l 1 -- {rot} translate -l 1 -- {dx}mm {dy}mm "
                          f"pagesize {page_orient}{_paper_spec(paper_w, paper_h)} "
                          f"write {shlex.quote(str(tmp))}")
    doc = vp.read_multilayer_svg(str(out_svg), quantization=0.1)
    return sum(len(lc) for lc in doc.layers.values())


# --------------------------------------------------------------------------- one pass
TIME_LABEL = "model (b), not stopwatch-validated"


def _time_est(gcode_path):
    """Model (b): per-move length / modal feed over the EMITTED gcode, Z moves included, no
    acceleration or lookahead (a floor, not a forecast). Reuses the calibration sheets' helper."""
    from ccplotter.calibration_fragment import time_model_b
    sec = float(time_model_b(gcode_path))
    return {"model": "b", "seconds": round(sec, 1), "minutes": round(sec / 60.0, 2), "label": TIME_LABEL}


# --------------------------------------------------------------------------- mm check
_ROOT = re.compile(r"<svg\b[^>]*>", re.I | re.S)


def check_svg_mm(text: str, label: str):
    """Refuse (400) any layer whose width/height/viewBox are not millimetres: width and height
    must carry an mm unit, and the viewBox must be 1 user unit = 1 mm (or = 1 CSS px, as vpype
    writes it, i.e. consistent with the mm width). Anything else would silently mis-scale."""
    m = _ROOT.search(text)
    if not m:
        raise TwoPenRefusal(f"{label}_svg is not an SVG document")
    root = m.group(0)
    def attr(n):
        a = re.search(r"\b" + n + r"\s*=\s*[\"']([^\"']*)[\"']", root, re.I)
        return a.group(1).strip() if a else None
    w, h, vb = attr("width"), attr("height"), attr("viewBox")
    mw = re.fullmatch(r"([0-9.]+)\s*mm", w or "", re.I)
    mh = re.fullmatch(r"([0-9.]+)\s*mm", h or "", re.I)
    if not (mw and mh):
        raise TwoPenRefusal(f"{label}_svg width/height must be in mm (got width={w!r} height={h!r}); "
                            "layers must be mm page coordinates")
    nums = re.findall(r"-?[0-9.]+(?:e-?[0-9]+)?", vb or "")
    if len(nums) != 4:
        raise TwoPenRefusal(f"{label}_svg needs a viewBox in mm user units (got {vb!r})")
    vw, vh = float(nums[2]), float(nums[3])
    ok = False
    for unit in (1.0, 96.0 / 25.4):      # user unit = 1 mm, or = 1 CSS px
        if abs(vw - float(mw.group(1)) * unit) <= 0.01 * vw and abs(vh - float(mh.group(1)) * unit) <= 0.01 * vh:
            ok = True
    if not ok:
        raise TwoPenRefusal(f"{label}_svg viewBox {vb!r} is not consistent with width/height in mm "
                            f"({w}, {h}); layers must be mm page coordinates")
def _insert_header_comment(path: Path, texts):
    data = path.read_bytes()
    eol = b"\r\n" if b"\r\n" in data[:4096] else b"\n"
    lines = data.splitlines(keepends=True)
    at = next((i for i, ln in enumerate(lines[:20]) if ln.startswith(b"; Source:")), 0) + 1
    for k, text in enumerate([texts] if isinstance(texts, str) else texts):
        lines.insert(at + k, b"; " + text.encode("ascii") + eol)     # raises on non-ASCII
    path.write_bytes(b"".join(lines))


def build_header(pass_no, rid, partner_name, placement, pen_id, z_touch):
    """ASCII header: exactly ONE line starts with '; PASS'; the rest are PAIR/PLACEMENT/PEN."""
    pl = placement
    return [PASS_HEADERS[pass_no],
            f"PAIR {rid} partner_file={partner_name}",
            "PLACEMENT shared offset_x={:.3f} offset_y={:.3f} scale={:g} rot={:g} dx={:g} dy={:g} "
            "paper={}x{}mm paper_rot={:g}".format(pl["offset_x"], pl["offset_y"], pl["scale"], pl["rot"],
                                                  pl["dx"], pl["dy"], pl["paper_w"], pl["paper_h"], pl["paper_rot"]),
            f"PEN this_pass pen_id={pen_id} z_touch_abs={z_touch:.2f}"]


def _set_gwrite_defaults(values: dict):
    vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update(values)


def generate_pass(placed_svg: Path, out_gcode: Path, pen_id, pen_profile, *, paper_w, paper_h,
                  paper_rot, offset_x, offset_y, pass_no, header_fn=None, microlift=True, end_policy="release",
                  quiet_fans=True, motion_smoothing="off", tip_window=None):
    """One complete gcode file for one pen. Mirrors _composite's gcode block, with every
    pen-dependent value taken from `pen_profile`. Returns a stats dict. Raises
    TwoPenRefusal on reach / calibration failure. The global vpype gwrite config is
    snapshotted and restored (see module docstring)."""
    motion_smoothing = str(motion_smoothing).lower()
    if motion_smoothing not in ("off", "zoned"):
        raise TwoPenRefusal("motion_smoothing must be 'off' or 'zoned'")
    z_up = _need(pen_profile, "z_up", pen_id)
    z_down = _need(pen_profile, "z_down", pen_id)
    draw_speed = float(pen_profile.get("draw_speed", 3000))
    travel_speed = float(pen_profile.get("travel_speed", 5000))
    plunge_speed = float(pen_profile.get("plunge_speed", 500))
    lift_speed = float(pen_profile.get("lift_speed", 2000))

    # Snapshot the WHOLE vpype config (it is process-wide); restored in `finally`.
    saved = copy.deepcopy(vp.config_manager.config)
    try:
        vp.config_manager.load_config_file(str(CONFIG_PATH))
        cal = load_cal()
        try:
            z_touch, z_src = resolve_z_touch(cal, pen_profile)
        except ZTouchNotCalibrated as e:
            raise TwoPenRefusal(str(e), 422)
        reach = pen_reach_profile(active_reach_profile(cal), pen_profile)
        pen_off_x, pen_off_y = resolve_pen_offset(cal, pen_profile)
        comp_off_x, comp_off_y = float(offset_x) - pen_off_x, float(offset_y) - pen_off_y
        park_x, park_y = reach["safe_park_point"]
        z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), z_up)
        _set_gwrite_defaults({
            "z_up": z_up, "z_down": z_down, "z_up_abs": z_touch + z_up,
            "z_down_abs": z_touch + z_down, "z_touch_abs": z_touch, "z_travel_abs": z_travel_abs,
            "draw_speed": draw_speed, "travel_speed": travel_speed,
            "plunge_speed": plunge_speed, "lift_speed": lift_speed, "z_auto": "1",
            "origin_x": park_x, "origin_y": park_y,
            "motion_smoothing_restore": "M204 S5000\n" if motion_smoothing == "zoned" else "",
            **_gcode_lifecycle_values(end_policy, quiet_fans)})
        rot = (f" rotate -o {paper_w / 2}mm {paper_h / 2}mm -- {paper_rot}"
               if abs(float(paper_rot)) > 1e-6 else "")
        with atomic_target(out_gcode) as tmp_output:
            vpype_cli.execute(f"read {shlex.quote(str(placed_svg))} linesort{rot} "
                              f"translate -- {comp_off_x}mm {-comp_off_y}mm "
                              f"gwrite --profile elegoo_cc {shlex.quote(str(tmp_output))}")
            _fix_gwrite_source_comment(tmp_output, out_gcode)

        z_micro, z_micro_src = resolve_z_micro(cal, pen_profile)
        z_up_abs, z_short_abs = z_touch + z_up, z_touch + z_micro
        mega_mm = float(cal.get("mega_hop_mm", 70.0))
        micro_hop_mm = float(pen_profile.get("micro_hop_mm", cal.get("micro_hop_mm", 8.0)))
        do_micro = microlift and z_up_abs > z_short_abs + 0.05
        do_mega = mega_mm > 0 and z_travel_abs > z_up_abs + 0.05
        microlift_stats = None
        if do_micro or do_mega:
            microlift_stats = _microlift_gcode(
                out_gcode, z_up_abs, z_short_abs=(z_short_abs if do_micro else None),
                short_mm=micro_hop_mm, z_mega_abs=(z_travel_abs if do_mega else None),
                long_mm=mega_mm)
        _apply_motion_smoothing(out_gcode, motion_smoothing)
        _insert_header_comment(out_gcode, header_fn(z_touch) if header_fn else PASS_HEADERS[pass_no])
        try:
            bb = _gcode_pendown_bbox(out_gcode, z_touch + z_down)
            motion = validate_gcode_xy_motion(out_gcode, reach)
        except Exception as e:
            raise TwoPenRefusal(f"reach safety check failed for pen '{pen_id}': {e}", 422)
        tip_bbox = None
        if tip_window is not None:
            # The PEN-TIP window on the paper (plate frame). tip = nozzle + composed pen offset.
            if not bb:
                raise TwoPenRefusal(f"pen '{pen_id}': no pen-down motion to check against the tip window", 422)
            tip_bbox = [bb[0] + pen_off_x, bb[1] + pen_off_y, bb[2] + pen_off_x, bb[3] + pen_off_y]
            (wx0, wx1), (wy0, wy1) = tip_window["x"], tip_window["y"]
            if tip_bbox[0] < wx0 or tip_bbox[2] > wx1 or tip_bbox[1] < wy0 or tip_bbox[3] > wy1:
                raise TwoPenRefusal(
                    f"pen '{pen_id}': pen-down tip bbox X[{tip_bbox[0]:.2f},{tip_bbox[2]:.2f}] "
                    f"Y[{tip_bbox[1]:.2f},{tip_bbox[3]:.2f}] leaves the paper window "
                    f"X[{wx0},{wx1}] Y[{wy0},{wy1}]", 422)
        return {"pen_id": pen_id, "z_touch_abs": z_touch, "z_touch_source": z_src,
                "z_up": z_up, "z_down": z_down, "z_down_abs": z_touch + z_down,
                "z_micro": z_micro, "z_micro_source": z_micro_src,
                "speeds": {"draw": draw_speed, "travel": travel_speed,
                           "plunge": plunge_speed, "lift": lift_speed},
                "pen_offset_mm": [pen_off_x, pen_off_y],
                "nozzle_offset_mm": [comp_off_x, comp_off_y], "reach_profile": reach["id"],
                "pendown_bbox": [round(v, 1) for v in bb] if bb else None,
                "tip_bbox": [round(v, 3) for v in tip_bbox] if tip_bbox else None,
                "tip_window": tip_window,
                "xy_moves_checked": motion["xy_moves"], "reach_validated": True,
                "microlift": microlift_stats,
                "time_est": _time_est(out_gcode)}
    finally:
        vp.config_manager.config.clear()
        vp.config_manager.config.update(saved)


# --------------------------------------------------------------------------- preview
def render_twopen_preview(blue_svg: Path, black_svg: Path, out_svg: Path, blue_pen_mm, black_pen_mm):
    """Two-colour true-width preview, blue UNDER black (layer 1 is written first)."""
    with atomic_target(out_svg) as tmp:
        vpype_cli.execute(
            f"read -l 1 {shlex.quote(str(blue_svg))} read -l 2 {shlex.quote(str(black_svg))} "
            f"linesort color -l 1 {BLUE_PREVIEW_COLOR} color -l 2 {BLACK_PREVIEW_COLOR} "
            f"penwidth -l 1 {blue_pen_mm}mm penwidth -l 2 {black_pen_mm}mm "
            f"write {shlex.quote(str(tmp))}")


# --------------------------------------------------------------------------- job
def build_twopen(body: dict, rid: str = None):
    """Run the whole job. Returns a dict (files, stats). Raises TwoPenRefusal.
    Files land in WORK_DIR/twopen/. The run record is added by the route."""
    bid, bprof, kid, kprof = resolve_pens(body)
    paper_w, paper_h = _parse_paper(body.get("paper", "b5"))
    rid = rid or uuid4().hex[:8]
    outdir = WORK_DIR / TWOPEN_DIRNAME
    scratch = outdir / f"{rid}_work"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        scale = float(body.get("scale", 1.0))
        rot = float(body.get("rot", 0.0))
        dx, dy = float(body.get("dx", 0.0)), float(body.get("dy", 0.0))
        paper_rot = float(body.get("paper_rot", 0.0))
        cal = load_cal()
        reach_default = active_reach_profile(cal)   # shared, UN-clamped: both pens register
        offset_x = float(body.get("offset_x", reach_default["bbox"][0]))
        offset_y = float(body.get("offset_y", reach_default["bbox"][1]))
    except TwoPenRefusal:
        raise
    except Exception as e:
        raise TwoPenRefusal(f"bad placement parameter: {e}")
    if scale <= 0:
        raise TwoPenRefusal("scale must be positive")

    layers = {}
    for key in ("blue", "black"):
        src = _materialise_layer(body.get(f"{key}_svg"), scratch / f"{key}_in.svg", key)
        placed = scratch / f"{key}_placed.svg"
        try:
            n = place_layer(src, placed, paper_w, paper_h, scale, rot, dx, dy)
        except Exception as e:
            raise TwoPenRefusal(f"{key}_svg could not be read/placed: {e}")
        if n == 0:
            raise TwoPenRefusal(f"{key} layer has no strokes")
        layers[key] = (placed, n)

    tw = body.get("tip_window")
    if tw is not None:
        try:
            tw = {"x": [float(v) for v in tw["x"]], "y": [float(v) for v in tw["y"]]}
        except Exception:
            raise TwoPenRefusal("tip_window must be {x:[min,max], y:[min,max]}")
    common = dict(tip_window=tw, paper_w=paper_w, paper_h=paper_h, paper_rot=paper_rot,
                  offset_x=offset_x, offset_y=offset_y,
                  microlift=bool(body.get("microlift", True)),
                  end_policy=str(body.get("end_policy", "release")),
                  quiet_fans=bool(body.get("quiet_fans", True)),
                  motion_smoothing=body.get("motion_smoothing", "off"))
    plan = [(1, "blue", bid, bprof), (2, "black", kid, kprof)]
    names = {c: f"{rid}_{n}_{c}_{_safe(pid)}.gcode" for n, c, pid, _ in plan}
    partner = {"blue": names["black"], "black": names["blue"]}
    placement = {"offset_x": offset_x, "offset_y": offset_y, "scale": scale, "rot": rot, "dx": dx,
                 "dy": dy, "paper_w": paper_w, "paper_h": paper_h, "paper_rot": paper_rot}
    files, passes = {}, []
    for pass_no, color, pid, prof in plan:
        out = outdir / names[color]
        files[color] = out
        hf = (lambda zt, _n=pass_no, _c=color, _p=pid: build_header(_n, rid, partner[_c], placement, _p, zt))
        st = generate_pass(layers[color][0], out, pid, prof, pass_no=pass_no, header_fn=hf, **common)
        st.update({"pass": pass_no, "color": color, "strokes": layers[color][1],
                   "file": f"{TWOPEN_DIRNAME}/{out.name}"})
        passes.append(st)

    preview = outdir / f"{rid}_twopen.preview.svg"
    render_twopen_preview(layers["blue"][0], layers["black"][0], preview,
                          preview_pen_width_mm(bprof, bprof), preview_pen_width_mm(kprof, kprof))
    return {"rid": rid, "files": files, "preview": preview, "passes": passes,
            "paper_w": paper_w, "paper_h": paper_h, "blue_profile": bprof,
            "black_profile": kprof, "placement": placement, "blue_id": bid, "black_id": kid,
            "layers": {k: str(v[0]) for k, v in layers.items()}}


def _safe(pen_id):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(pen_id))
