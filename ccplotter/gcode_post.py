import math
import re
import shlex
from pathlib import Path

import vpype as vp
import vpype_cli

from ccplotter.atomic_io import atomic_target, atomic_write_text
from ccplotter.config import (FAN_OFF_GCODE, PEN_MIN_X, PEN_MIN_Y, PEN_MAX_X,
                               PEN_MAX_Y, PREVIEW_COLOR, CONFIG_PATH, WORK_DIR, PX2MM)
from ccplotter.cal import resolve_pen_offset, load_cal, active_pen_profile, resolve_z_touch
from ccplotter.reach import active_reach_profile, validate_gcode_xy_motion, pen_reach_profile
from ccplotter.svg_io import _write_paths_svg


def _fix_gwrite_source_comment(tmp_path: Path, final_path: Path) -> None:
    """The elegoo_cc `gwrite` profile's document_start template (vpype.toml)
    embeds the destination path passed to `gwrite` as a '; Source: {filename}'
    comment. When atomic_io.atomic_target hands `gwrite` a TEMP path instead
    of the real target (see composite.py / _calibration_gcode below), that
    comment names the temp file instead of the real one, which would differ
    in bytes from writing straight to `final_path`. Patch the comment back to the real destination's string form (exactly
    what the direct-write path always embedded) before the atomic replace,
    on the raw bytes -- no text-mode round trip, so nothing else in the file
    can be touched by this."""
    data = tmp_path.read_bytes()
    data = data.replace(str(tmp_path).encode("ascii"), str(final_path).encode("ascii"))
    tmp_path.write_bytes(data)


def _gcode_lifecycle_values(end_policy: str = "release", quiet_fans: bool = True) -> dict:
    """The end-of-job postamble AND the one-time preamble fan-off live INSIDE
    the shared vpype.toml elegoo_cc profile (document_start/document_end), not
    in a Python-side append after gwrite. Every writer that calls
    `gwrite --profile elegoo_cc` therefore gets a complete ending, and none can
    forget a separate append step. This helper computes the three template
    values every writer must feed into default_values before calling gwrite.

    end_policy: "release" (default, the stock firmware behaviour -- M400 then
    M84) or "hold" (motors stay energized; M84 is commented out, not omitted, so
    the file still documents the choice).
    quiet_fans: one-time fan-off block after initial positioning, before any
    drawing motion. Fan commands are not repeated through the file: a file of
    bare M-codes with no motion halts this firmware, and this block has real
    motion on both sides."""
    end_stepper = "M84" if str(end_policy).lower() != "hold" else "; motors held (end_policy=hold)"
    preamble_fans = "\n".join(FAN_OFF_GCODE) if quiet_fans else "; quiet_fans off"
    return {"end_policy": str(end_policy).lower(), "end_stepper": end_stepper,
            "preamble_fans": preamble_fans}

def _microlift_gcode(gcode_path: Path, z_up_abs: float, z_short_abs: float = None,
                     short_mm: float = 8.0, z_mega_abs: float = None, long_mm: float = 70.0):
    """Z post-processor: rewrite each per-hop lift height by TRAVEL DISTANCE.
      - hop < short_mm  -> z_short_abs  (MICRO: less lead-screw wear)        [if set]
      - hop > long_mm   -> z_mega_abs   (MEGA: extra clearance, anti-drag on
                                          the longest traverses)             [if set]
      - otherwise         keep the full z_up_abs lift.
    Z-move COUNT unchanged. Only touches exact lift/travel/plunge triples; doc
    start/end (already at full travel clearance) untouched."""
    import re
    rx_xy = re.compile(r"^G1 X([-\d.]+) Y([-\d.]+)")
    rx_z = re.compile(r"^(G1 Z)([-\d.]+)( F[-\d.]+)")
    lines = gcode_path.read_text().splitlines()
    cur, n_short, n_mega, n_lifts, saved, added = None, 0, 0, 0, 0.0, 0.0
    for i in range(len(lines)):
        mz = rx_z.match(lines[i])
        if not (mz and abs(float(mz.group(2)) - z_up_abs) < 0.01 and i + 2 < len(lines)):
            mxy = rx_xy.match(lines[i])
            if mxy:
                cur = (float(mxy.group(1)), float(mxy.group(2)))
            continue
        mt = rx_xy.match(lines[i + 1])
        md = rx_z.match(lines[i + 2])
        if not (mt and md and float(md.group(2)) < z_up_abs - 0.01):
            continue                                   # not a lift/travel/plunge stroke triple
        n_lifts += 1
        tx, ty = float(mt.group(1)), float(mt.group(2))
        hop = math.hypot(tx - cur[0], ty - cur[1]) if cur is not None else 0.0
        if z_short_abs is not None and cur is not None and hop < short_mm:
            lines[i] = f"{mz.group(1)}{z_short_abs:.2f}{mz.group(3)}"
            n_short += 1
            saved += 2.0 * (z_up_abs - z_short_abs)    # up + down travel saved
        elif z_mega_abs is not None and cur is not None and hop > long_mm:
            lines[i] = f"{mz.group(1)}{z_mega_abs:.2f}{mz.group(3)}"
            n_mega += 1
            added += 2.0 * (z_mega_abs - z_up_abs)     # extra up + down on long hops
    if n_short or n_mega:
        atomic_write_text(gcode_path, "\n".join(lines) + "\n")
    return {"lifts": n_lifts, "short_lifts": n_short, "mega_lifts": n_mega,
            "z_travel_saved_mm": round(saved, 1), "mega_added_mm": round(added, 1),
            "short_lift_mm": (round(z_short_abs, 2) if z_short_abs is not None else None),
            "mega_lift_mm": (round(z_mega_abs, 2) if z_mega_abs is not None else None)}

def _gcode_pendown_bbox(gcode_path: Path, z_draw_abs: float, tol: float = 0.05):
    """Machine-coord bbox of PEN-DOWN moves in a written gcode (the real drawing
    extent). Used to flag a drawing that falls outside the pen-reachable area
    when the paper is placed freely on the build plate. Returns
    (min_x, min_y, max_x, max_y) or None."""
    import re
    pen_down = False; cx = cy = None
    xs, ys = [], []
    for line in gcode_path.read_text().splitlines():
        mz = re.search(r"Z(-?[\d.]+)", line)
        if mz is not None:
            pen_down = float(mz.group(1)) <= z_draw_abs + tol
        mx = re.search(r"X(-?[\d.]+)", line); my = re.search(r"Y(-?[\d.]+)", line)
        if mx: cx = float(mx.group(1))
        if my: cy = float(my.group(1))
        if pen_down and (mx or my) and cx is not None and cy is not None:
            xs.append(cx); ys.append(cy)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))

def _fan_probe_gcode(out_path: Path) -> Path:
    """Empirical M-code probe. Pure M-codes + dwells, NO motion. Plot with the
    case fans ON and note which step (1-5) kills which fan; that table decides
    whether the gcode path or the websocket path is the fix."""
    # DISABLED in the UI. Kept only for reference. Uses FIRMWARE-VALID indices
    # (M106 no-P=hotend, P2=part cooling, P3=chamber); P0/P1 are invalid and halt
    # the Klipper firmware. Do NOT sideload bare M-code files to this printer; use
    # the end-of-plot postamble instead.
    L = ["; ccPlotter FAN PROBE — DISABLED. Reference only.",
         "; No motion. Each step waits ~10s. Note which step # stops which fan.",
         "G4 S10                 ; step 0: baseline (fans still ON)",
         "M106 S0                ; step 1: hotend fan off",
         "G4 S10",
         "M106 P2 S0             ; step 2: part-cooling fan off",
         "G4 S10",
         "M106 P3 S0             ; step 3: chamber/case fan off",
         "G4 S10",
         "M107                   ; step 4: M107 (fan off)",
         "G4 S10",
         "; DONE — report which step # killed which fan (or 'none of them')."]
    atomic_write_text(out_path, "\n".join(L) + "\n")
    return out_path


def _inject_fan_off(gcode_path: Path, every: int = 500):
    """Emit fan-off in the PREAMBLE (before first motion) and re-assert it every
    `every` lines. Not called by the app: repeating fan commands through the
    body of a plot is avoided on this firmware (the supported paths are the
    one-time preamble block and the end-of-plot postamble in vpype.toml)."""
    try:
        lines = Path(gcode_path).read_text().splitlines()
    except Exception:
        return
    out, injected, since = [], False, 0
    for ln in lines:
        if not injected and (ln.startswith("G0") or ln.startswith("G1")):
            out.append("; fan-off preamble"); out.extend(FAN_OFF_GCODE); injected = True; since = 0
        out.append(ln); since += 1
        if injected and since >= every:
            out.extend(FAN_OFF_GCODE); since = 0
    if not injected:
        out = ["; fan-off preamble"] + list(FAN_OFF_GCODE) + out
    try:
        atomic_write_text(Path(gcode_path), "\n".join(out) + "\n")
    except Exception as e:
        print(f"  [fans] injection failed: {e}")


def validate_calibration_gcode_safety(gcode_path: Path, z_down_abs: float) -> dict:
    """Safety self-checks for GENERATED CALIBRATION GCODE, so a bad calibration
    file can't reach the printer via the /api/calibration_plot route. Checks,
    each a RuntimeError on failure (same convention as
    validate_gcode_xy_motion -- the caller wraps this in try/except -> jsonify
    error, 500):
      - ASCII only (a non-ASCII byte, e.g. an em-dash, crashes the printer's
        firmware gcode parser).
      - no G92 (absolute machine coords only).
      - no M106 P0 / M106 P1 (invalid fan indices on this Klipper firmware;
        they HALT the machine).
      - every commanded Z >= z_down_abs (the deepest legal pen-down height for
        THIS plot's resolved z_touch_abs) -- a stray Z below that would drive
        the pen into the bed.
    XY-reach is validated separately by validate_gcode_xy_motion (unchanged)."""
    try:
        text = gcode_path.read_bytes().decode("ascii")
    except UnicodeDecodeError as e:
        raise RuntimeError(f"Calibration gcode is not pure ASCII (firmware parser hazard): {e}") from e
    lines = text.splitlines()
    if any(l.strip().startswith("G92") for l in lines):
        raise RuntimeError("Calibration gcode contains G92 (absolute machine coords only).")
    if any(l.strip().startswith("M106 P0") or l.strip().startswith("M106 P1") for l in lines):
        raise RuntimeError("Calibration gcode contains an invalid fan index (M106 P0/P1) -- HALTS this firmware.")
    zs = []
    for l in lines:
        if l.startswith("G1 Z"):
            try:
                zs.append(float(l.split("Z", 1)[1].split()[0]))
            except (IndexError, ValueError):
                pass
    bad = sorted(z for z in zs if z < z_down_abs - 0.01)
    if bad:
        raise RuntimeError(
            f"Calibration gcode commands Z below z_down_abs={z_down_abs:.2f}mm "
            f"(pen-into-bed risk): {bad}")
    return {"ascii_ok": True, "no_g92": True, "fan_index_ok": True,
            "z_levels_mm": sorted(set(round(z, 3) for z in zs)),
            "z_down_abs": round(z_down_abs, 3), "z_floor_ok": True}


def _calibration_gcode(out_gcode: Path, pen_mm=0.5, z_up=3.5, z_down=-1.0,
                       draw_speed=3000, travel_speed=8000, plunge_speed=500, lift_speed=2000,
                       offset_x=None, offset_y=None, end_policy="release", quiet_fans=True,
                       profile=None):
    """Parametric CALIBRATION PLOT (no image input). Emits, at true physical mm:
      - four 10 mm squares POCKET-FILLED at stepover 0.7/0.8/0.9/1.0 x pen
      - line ladders (8 parallel lines) at spacing 0.3/0.5/0.7/1.0 mm
      - a 20 mm ruled segment with 5 mm ticks
    Measure with calipers -> confirms mm->gcode scale, sets pen_cal (real deposited
    line width) and pocket_stepover. Same gwrite/elegoo_cc path as real plots.

    profile: the pen profile whose z_touch_abs is authoritative for this plot
    (falls back to calibration.json's global z_touch_abs via resolve_z_touch()
    if the profile has none). Defaults to the currently-active pen profile so a
    caller who doesn't pass one still gets a per-pen-correct plot; pass one
    explicitly (as the route below does) to make "which pen this plot is for"
    an explicit, auditable choice rather than an implicit global read."""
    from shapely.geometry import Polygon
    ppmm = 1.0 / PX2MM
    polys = []                                           # mm, layout space

    # Resolve the pen profile up front: the ladder spacings below are derived
    # from it, and it is also what resolve_z_touch() consumes further down.
    # (Resolving it here, not only at the Z step, keeps a profile-less caller
    # from getting the EnerGel's spacings on another pen.)
    if profile is None:
        profile = active_pen_profile()

    # Spacings (mm), fine -> coarse L->R. The plot answers "at what line/pass
    # spacing does it go solid?", so the six bands must BRACKET the pen's real
    # deposit width -- a band set entirely below it reads solid all the way
    # across and measures nothing.
    #
    # The set [0.1, 0.15, 0.2, 0.25, 0.3, 0.4] is 0.25mm (the EnerGel's
    # measured deposit) x [0.4, 0.6, 0.8, 1.0, 1.2, 1.6], so the spacings are
    # pen-relative: a fixed absolute set would not bracket another pen. For a
    # ~0.4mm ballpoint deposit, every band of the EnerGel set would sit at or
    # below the line width, all six would plot solid, and the test would be
    # unreadable. Rounding keeps 0.25*0.6 at 0.15, not 0.15000000000000002.
    # fill_mm is the estimate being tested -- a rough one is fine, the
    # 0.4x-1.6x bracket is wide enough to straddle a bad guess.
    _RATIOS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.6]
    _fill_est = float((profile or {}).get("fill_mm") or 0.25)
    if not (0.02 <= _fill_est <= 2.0):                   # nonsense guard -> EnerGel-era default
        _fill_est = 0.25
    SPACINGS = [round(_fill_est * r, 3) for r in _RATIOS]
    pitch, side = 11.0, 8.0

    def sq_pocket_abs(x0, y0, s):                        # pocket at absolute ring spacing s
        poly = Polygon([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)])
        rings = [list(poly.exterior.coords)]
        d = s / 2.0
        while True:
            ins = poly.buffer(-d)
            if ins.is_empty:
                break
            for g in ([ins] if ins.geom_type == 'Polygon' else list(ins.geoms)):
                if not g.is_empty:
                    rings.append(list(g.exterior.coords))
            d += s
        return rings

    for i, s in enumerate(SPACINGS):                     # 1) pocketed squares (fill test)
        polys += sq_pocket_abs(i * pitch, 0.0, s)
    for i, s in enumerate(SPACINGS):                     # 2) line ladders (deposited-width test)
        bx = i * pitch
        n = max(6, min(40, int(round(3.0 / s))))
        for k in range(n):
            yy = 11.0 + k * s
            polys.append([(bx, yy), (bx + side, yy)])
    ry = 17.0                                            # 3) 20mm ruler + 5mm ticks
    polys.append([(0.0, ry), (20.0, ry)])
    for t in range(0, 21, 5):
        polys.append([(float(t), ry), (float(t), ry + 3.0)])

    W = int(round(70 * ppmm)); Hh = int(round(24 * ppmm))
    polys_px = [[(x * ppmm, y * ppmm) for x, y in p] for p in polys]
    # Pattern filename keyed off out_gcode's stem (not a fixed name) so a pack
    # generated for one pen can't have its pattern SVG silently overwritten by
    # a later generation for a different pen before the user imports it as a
    # drawing (see ccplotter/routes/pen.py api_calibration_import).
    svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_paths_svg(svg, W, Hh, float(pen_mm) * ppmm, polylines=polys_px)

    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()
    # Per-pen reach clamp: same conservative intersection as composite.py -- a
    # profile with no reach_clamp gets this active_reach_profile() object back
    # unchanged.
    reach = pen_reach_profile(active_reach_profile(cal), profile)
    z_touch, z_touch_source = resolve_z_touch(cal, profile)   # profile resolved at fn top
    print(f"  [z] calibration plot z_touch_abs={z_touch:.2f}mm (source={z_touch_source})")
    pen_off_x, pen_off_y = resolve_pen_offset(cal, profile)
    offset_x = reach["bbox"][0] + 20.0 if offset_x is None else float(offset_x)
    offset_y = reach["bbox"][1] + 20.0 if offset_y is None else float(offset_y)
    comp_off_x, comp_off_y = offset_x - pen_off_x, offset_y - pen_off_y
    # The calibration writer's start/end movement is physical nozzle motion too.
    # Use the profile's deliberately inboard point rather than rectangular clamping.
    park_x, park_y = reach["safe_park_point"]
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))
    vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({
        "z_up": float(z_up), "z_down": float(z_down), "z_up_abs": z_touch + float(z_up),
        "z_down_abs": z_touch + float(z_down), "z_touch_abs": z_touch, "z_travel_abs": z_travel_abs,
        "draw_speed": float(draw_speed), "travel_speed": float(travel_speed),
        "plunge_speed": float(plunge_speed), "lift_speed": float(lift_speed),
        "z_auto": "1",
        "origin_x": park_x, "origin_y": park_y, **_gcode_lifecycle_values(end_policy, quiet_fans)})
    with atomic_target(out_gcode) as tmp_gcode:
        vpype_cli.execute(f'read {shlex.quote(str(svg))} pagesize {W}x{Hh}px linesort '
                          f'translate -- {comp_off_x}mm {-comp_off_y}mm '
                          f'gwrite --profile elegoo_cc {shlex.quote(str(tmp_gcode))}')
        _fix_gwrite_source_comment(tmp_gcode, out_gcode)
    validate_gcode_xy_motion(out_gcode, reach)
    z_down_abs = z_touch + float(z_down)
    safety = validate_calibration_gcode_safety(out_gcode, z_down_abs)
    # Postamble + preamble fan-off come from the elegoo_cc profile itself now
    # (vpype.toml) -- nothing to append here. This writer cannot produce an
    # incomplete ending short of not calling gwrite at all.
    # Preview SVG (accent-colored) for the main preview + positioning plate.
    prev = WORK_DIR / "calibration_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(svg))} color {PREVIEW_COLOR} '
                          f'write {shlex.quote(str(tmp_prev))}')
    # Pen-target machine bbox of the drawn pattern (content ~63x20mm at the offset).
    cw, chh = 63.0, 20.0
    place = {"x": offset_x, "y": offset_y, "w": cw, "h": chh}   # machine pen-target mm
    return {"pen_mm": pen_mm, "pattern_mm": [70, 24], "offset_mm": [offset_x, offset_y],
            "polylines": len(polys), "preview": prev.name, "place": place,
            "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
            "z_up_abs": round(z_touch + float(z_up), 3), "z_down_abs": round(z_down_abs, 3),
            "safety_selfcheck": safety,
            # Report the spacings ACTUALLY drawn (derived from fill_mm), so the UI
            # never labels bands with spacings the plot did not use.
            "spacings_mm": list(SPACINGS),
            "fill_mm_estimate": _fill_est,
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
            "svg_wh_px": [W, Hh], "svg_path": str(svg)}

_G1_XY_RE = re.compile(r"^G1 X(-?\d+(?:\.\d+)?) Y(-?\d+(?:\.\d+)?) F(-?\d+(?:\.\d+)?)$")
_G1_Z_RE = re.compile(r"^G1 Z-?\d+(?:\.\d+)? F-?\d+(?:\.\d+)?$")


def _apply_motion_smoothing(gcode_path: Path, mode: str = "off", *, dense_accel: int = 750) -> dict:
    """Opt-in geometric M204 zoning for gwrite's pen-stroke blocks.

    Each gwrite stroke is ``Z-up, XY travel, Z-down, draw...``.  The mean
    segment length is measured from that travel point plus the draw points;
    strokes below 1 mm use the conservative dense acceleration.  Commands are
    emitted only when the commanded acceleration state changes, never per draw
    segment.  ``off`` deliberately does not read or rewrite the file, which
    preserves byte identity for existing jobs.
    """
    mode = str(mode).lower()
    if mode == "off":
        return {"mode": "off", "insertions": 0, "line_count": None,
                "insertion_rate": 0.0, "dense_strokes": 0, "long_strokes": 0}
    if mode != "zoned":
        raise ValueError("motion_smoothing must be 'off' or 'zoned'")

    lines = gcode_path.read_text(encoding="ascii").splitlines()
    insert_before: dict[int, list[str]] = {}
    current_accel = None
    dense_strokes = long_strokes = 0
    i = 0
    while i + 3 < len(lines):
        # This exact structural pattern comes from vpype's segment_first
        # template.  It intentionally excludes the document-start approach,
        # which has no following plunge move.
        if not (_G1_Z_RE.fullmatch(lines[i]) and
                (start_match := _G1_XY_RE.fullmatch(lines[i + 1])) and
                _G1_Z_RE.fullmatch(lines[i + 2])):
            i += 1
            continue
        points = [(float(start_match.group(1)), float(start_match.group(2)))]
        j = i + 3
        while j < len(lines):
            draw_match = _G1_XY_RE.fullmatch(lines[j])
            if not draw_match:
                break
            points.append((float(draw_match.group(1)), float(draw_match.group(2))))
            j += 1
        if len(points) < 2:
            i += 1
            continue
        path_len = sum(math.hypot(bx - ax, by - ay)
                       for (ax, ay), (bx, by) in zip(points, points[1:]))
        is_dense = (path_len / (len(points) - 1)) < 1.0
        # Travel (and sparse draw) are S3000.  Set it before the lift/travel
        # block, then switch only dense strokes after the plunge and before ink.
        if current_accel != 3000:
            insert_before.setdefault(i, []).append("M204 S3000")
            current_accel = 3000
        if is_dense:
            dense_strokes += 1
            if current_accel != dense_accel:
                insert_before.setdefault(i + 3, []).append(f"M204 S{dense_accel}")
                current_accel = dense_accel
        else:
            long_strokes += 1
        i = j

    if not insert_before:
        return {"mode": mode, "insertions": 0, "line_count": len(lines),
                "insertion_rate": 0.0, "dense_strokes": dense_strokes,
                "long_strokes": long_strokes, "dense_accel": dense_accel}
    out = []
    for idx, line in enumerate(lines):
        out.extend(insert_before.get(idx, ()))
        out.append(line)
    atomic_write_text(gcode_path, "\n".join(out) + "\n", encoding="ascii", newline="\n")
    insertions = sum(len(v) for v in insert_before.values())
    return {"mode": mode, "insertions": insertions, "line_count": len(lines),
            "insertion_rate": insertions / len(lines), "dense_strokes": dense_strokes,
            "long_strokes": long_strokes, "dense_accel": dense_accel}
