"""ccplotter/twopen_keytint.py -- "Black Line + Blue Tone" (key + tint), backend generator.

Photo in -> TWO mm page-coordinate stroke layers (blue = mid-tone tint, black = darks + structure) that
the existing two-pen pipeline (ccplotter/twopen.py, via POST /api/twopen_keytint) turns into two complete
gcode files and a two-colour preview. Pass order is FIXED by twopen.py: pass 1 = BLUE, pass 2 = BLACK
(blue is laid first, black overprints it).

THE RECIPE (every rule is global, no per-image logic)
  base     the app's own bwphoto job (XDoG front end at the default native side -- "C1024F" for a photo whose
           short side is >= 1024 px -- then CLAHE, then the Trace V3 mask). The job is RUN, not re-implemented:
           its post-CLAHE grey raster, its trace options and its page placement are captured, so both layers use
           exactly the geometry a plain bwphoto job on the same body would use (resolution, pass pitch, fit).
  M0       the plain job's ink mask (mask._binarize_mask on that raster).
  black    BLACK = core | E  where  core = M0 & (smoothed grey < T_DARK_FRAC * T)  and  E = M0 minus its morphological
           opening at the pen width (thin detail). Dark regions are filled solid, no ring/pitch textures.
  strip    STRIP_IN_BLACK (tentative until the "M10-REG v1" sheet B is read): the sub-pen strip of M0 that lies
           within one pen width of BLACK is drawn BLACK, and blue fragments below the dust floor that touch black
           move into black (directional; black never moves to blue).
  blue     mid-tone tint = (3x3-blurred grey < T + a * (255 - T)) minus the black footprint, a = blue_amount
           (one global request parameter, range 0..0.5; ~0.375 suits the photos tried).
  trace    each mask goes through the UNCHANGED trace_v3._trace_svg with the plain job's options, then the same
           vpype chain composite.py runs (scale/translate by the plain job's placement, linemerge, pagesize) and
           the same low-lift router, so a layer is a normal routed layer.

WHY THE BLACK PASS IS NOT BYTE-IDENTICAL TO A PLAIN bwphoto JOB: the plain job draws all of M0 in one ink; here the
lighter, thick part of M0 (grey between T_DARK_FRAC*T and T, away from thin detail) is handed to BLUE, and the strip
rule adds sub-pen ring pixels to black. That is the recipe, not drift. The machinery IS identical: tracing M0 itself
through layer_from_mask() reproduces the plain job's routed strokes (tests/test_twopen_keytint.py positive control).

All pen widths here are NOMINAL (the BIC laid width is unmeasured until the "M10-REG v1" sheet is read).
"""
import shutil
import shlex
import threading
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import vpype as vp
import vpype_cli

# ---------------------------------------------------------------- constants (global, documented)
BLUE_AMOUNT_DEFAULT = 0.375          # observed good range 0.375-0.5 (defaults, not per-image presets)
BLUE_AMOUNT_MIN, BLUE_AMOUNT_MAX = 0.0, 0.5   # the supported range (a = 0 .. 0.5); requests are clamped to it
ROLE_WIDTH_MM = 0.12                 # nominal BIC pen width: opening radius for E, strip/clearance radius
T_DARK_FRAC = 0.8                    # core = smoothed grey < 0.8 * Otsu T
GREY_SMOOTH_SIGMA_MM = 0.3           # Gaussian sigma of the grey used for core
STRIP_IN_BLACK = True                # strip in BLACK (tentative until "M10-REG v1" sheet B is read)
DEFAULT_BLUE_PEN = "bic_cristal_blue"
DEFAULT_BLACK_PEN = "bic_cristal_black"
DEFAULT_PAPER = "257x182"            # JIS B5 Kokuyo landscape (same default as /api/twotone)
DEFAULT_MARGIN_MM = 10.0
PREVIEW_PPM = 12.0                   # PNG preview resolution, px per mm
PREVIEW_SS = 2                       # PNG supersampling
PREVIEW_PAPER_RGB = (250, 248, 242)  # paper colour in the PNG preview
PREVIEW_BLUE_RGB, PREVIEW_BLACK_RGB = (31, 79, 216), (17, 17, 17)   # UNMEASURED ink colours
BLUE_CONNECTOR_MM = 0.0                # blue is NOT routed with the body value until the dense-gated chain lands
# ---- OPT-IN "blue margin under black" (margin + connector + fold; body flag blue_trap, default OFF).
# All thresholds are in mm / multiples of the BLUE pen's fill_mm (never pixels), so they scale with resolution and pen.
TRAP_MARGIN_PEN = 1.0                # margin: blue grows into black by d = 1 pen width
TRAP_HOLE_RADIUS_PEN = 1.5           # holes of the blue mask smaller than pi*(1.5 W)^2 are closed (floor px, strict <)
TRAP_FOLD_RADIUS_PEN = 1.5           # fold ("F"): black components smaller than pi*(1.5 W)^2 are REASSIGNED to blue, never deleted
TRAP_BLUE_CONNECTOR_MM = 0.5         # blue connectors at 0.5 mm, mask-checked against the blue + black union (stage_e_mask)
CONSTANTS = {
    "blue_amount_default": BLUE_AMOUNT_DEFAULT, "blue_amount_range": [BLUE_AMOUNT_MIN, BLUE_AMOUNT_MAX],
    "role_width_mm": ROLE_WIDTH_MM, "t_dark_frac": T_DARK_FRAC, "grey_smooth_sigma_mm": GREY_SMOOTH_SIGMA_MM,
    "strip_in_black": STRIP_IN_BLACK, "pass_order": ["blue", "black"],
    # The connector rule, visible here and echoed per layer in info["connectors"].
    "connector_rule": "blue connectors: chain only (D1695), 0.0 until chain lands, unless blue_trap is on, then 0.5 mm mask-checked against the blue+black union (D1721); black connectors: body connector_distance as plain",
    "blue_connector_distance_mm": BLUE_CONNECTOR_MM,
    "provenance": "exp4 D0 + blue ladder, exp5 Q1-k (tentative), D1425-D1480, D1524/D1532; widths nominal",
    # opt-in hatched blue (default OFF): see ccplotter/twopen_hatch.py (set below, after the module-level constants)
    # opt-in blue margin under black (default OFF); thresholds in pen widths of the BLUE profile's fill_mm
    "blue_trap": {"option": "body.blue_trap (bool, default false)", "margin_pen_widths": TRAP_MARGIN_PEN,
                  "hole_close_radius_pen_widths": TRAP_HOLE_RADIUS_PEN, "fold_radius_pen_widths": TRAP_FOLD_RADIUS_PEN,
                  "blue_connector_distance_mm": TRAP_BLUE_CONNECTOR_MM, "blue_connector_mask_checked_against": "blue+black union",
                  "blue_walk_cap": "ROUTE_WALK_MAX_SEGS_TWOPEN", "d_numbers": "D1704-D1712 (X, F), D1713-D1720 (T1), Kai 2026-10-21",
                  "artifacts": "artifacts/twopen-trap-2026-10-21/"},
}

from ccplotter.twopen_hatch import CONSTANTS_HATCHED as _CONSTANTS_HATCHED        # noqa: E402  (module has no import-time dependency on this one)
CONSTANTS["hatched_blue"] = _CONSTANTS_HATCHED


class KeytintError(Exception):
    """Raised for a refusable request/generation problem; the route maps it to a TwoPenRefusal (4xx)."""
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def clamp_blue_amount(value):
    """Request value -> float in [BLUE_AMOUNT_MIN, BLUE_AMOUNT_MAX]; None -> the default; non-numeric -> KeytintError."""
    if value is None or value == "":
        return BLUE_AMOUNT_DEFAULT
    if isinstance(value, bool):
        raise KeytintError("blue_amount must be a number")
    try:
        a = float(value)
    except (TypeError, ValueError):
        raise KeytintError("blue_amount must be a number")
    if a != a:
        raise KeytintError("blue_amount must be a number")
    return min(BLUE_AMOUNT_MAX, max(BLUE_AMOUNT_MIN, a))


# ---------------------------------------------------------------- step 1: run the plain bwphoto job, capture its facts
_HOOK_LOCK = threading.Lock()


def capture_plain_bwphoto(image_path, workdir: Path, *, paper_w, paper_h, margin, pen_id, paper_spec, body_extra=None,
                          item=None):
    """Run a plain bwphoto PREVIEW job on `image_path` through the real /api/process route (in-process test client,
    no socket) and capture, from THAT job: the post-CLAHE grey raster entering the trace and its options (a temporary,
    thread-filtered wrapper on render._trace_svg), the page placement (scale, rotation centre, translate -- recomputed
    with composite.py's single-item fit rule and cross-checked against the response's item_geometry), and its routed SVG.
    `body_extra` = the rest of a normal Generate body (offsets, paper_keepout, registered_paper, paper_rot, settings ...);
    it reaches the plain job unchanged, so the plain job and both layers share ONE placement and ONE keep-out.
    `item` = optional {scale, rot, dx, dy, crop}; the drawing is copied to a fresh id every time (no cache can skip the trace).
    Must NOT be called while holding routes.core._PROCESS_LOCK (the route takes it). Returns a dict; raises KeytintError."""
    import ccplotter.render as rnd
    import ccplotter.routes.core  # noqa: F401  (registers /api/process on the shared app)
    from ccplotter.composite import _rotated_extent_mm
    from ccplotter.config import PX2MM, WORK_DIR
    from ccplotter.routes import app
    from ccplotter.svg_io import _svg_bbox_mm
    image_path = Path(image_path)
    it = {"scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}
    it.update({k: v for k, v in (item or {}).items() if k in it})
    if it["rot"] is None:
        it["rot"] = 0
    did = uuid4().hex[:8]
    src = WORK_DIR / f"src_{did}{image_path.suffix.lower()}"
    shutil.copyfile(image_path, src)
    owner = threading.get_ident()
    calls = []
    body = dict(body_extra or {})
    body.update({"style": "bwphoto", "job_type": "preview", "paper": paper_spec, "margin": margin, "pen_id": pen_id,
                 "debug": True,    # debug skips the variant cache (vcache) so the trace always runs and the hook always fires
                 "items": [{"id": did, **it}]})
    try:
        with _HOOK_LOCK:
            orig = rnd._trace_svg

            def _wrap(img, out_path, opts, *a, **k):
                r = orig(img, out_path, opts, *a, **k)
                if threading.get_ident() == owner:
                    calls.append((np.array(img, copy=True), dict(opts)))
                return r
            rnd._trace_svg = _wrap
            try:
                resp = app.test_client().post("/api/process", json=body)
            finally:
                rnd._trace_svg = orig
        if resp.status_code == 429:
            raise KeytintError("busy: another job is being processed", 429)
        j = resp.get_json(silent=True) or {}
        if resp.status_code != 200:
            raise KeytintError(f"plain bwphoto job failed ({resp.status_code}): {j.get('error', 'no detail')}",
                               resp.status_code if 400 <= resp.status_code < 500 else 500)
        if not calls:
            raise KeytintError("the bwphoto job produced no trace call to capture", 500)
        img, opts = calls[-1]
        renders = sorted(WORK_DIR.glob(f"src_{did}_render_*.svg"), key=lambda p: p.stat().st_mtime)
        if not renders:
            raise KeytintError("the bwphoto job left no render svg", 500)
        render_svg = renders[-1]
        # composite.py single-item placement (cols = rows = 1): fit = min(cell / rotated size) * 0.9, s = fit * scale,
        # translate = cell centre - source centre + (dx, dy); scale and rotate act about the source bbox centre.
        nb = _svg_bbox_mm(render_svg)
        nw, nh = max(nb[2] - nb[0], 1e-3), max(nb[3] - nb[1], 1e-3)
        ncx, ncy = (nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2
        cell_w, cell_h = paper_w - 2 * margin, paper_h - 2 * margin
        ccx, ccy = margin + cell_w / 2, margin + cell_h / 2
        rot = float(it["rot"])
        rw, rh = _rotated_extent_mm(nw, nh, rot)
        fit = min(cell_w / rw, cell_h / rh) * 0.9
        s = fit * float(it["scale"])
        tx, ty = ccx - ncx + float(it["dx"]), ccy - ncy + float(it["dy"])
        ig = (j.get("item_geometry") or {}).get(did)
        if not ig:
            raise KeytintError("the bwphoto job returned no item_geometry", 500)
        if (abs(ig["base_fit_scale"] - fit) > 1e-7 or abs(ig["layout_center_mm"][0] - ccx) > 1e-5
                or abs(ig["source_center_mm"][0] - ncx) > 1e-5 or abs(ig["source_center_mm"][1] - ncy) > 1e-5):
            raise KeytintError("placement recomputation disagrees with composite.py's item_geometry "
                               f"(fit {fit:.9f} vs {ig['base_fit_scale']}); refusing to guess", 500)
        routed_copy = Path(workdir) / "plain_bwphoto.routed.svg"
        shutil.copyfile(WORK_DIR / "layout.routed.svg", routed_copy)
        return {"gray": img, "opts": opts, "trace_calls": len(calls), "fit": s, "tx": tx, "ty": ty, "rot": rot,
                "c0_px": (float(ncx / PX2MM), float(ncy / PX2MM)), "mm_per_px": s * PX2MM, "paper_w": paper_w, "paper_h": paper_h,
                "margin": margin, "plain_routed_svg": routed_copy, "item_geometry": ig,
                "item_bboxes": j.get("item_bboxes") or {},     # UI handles only (main-flow mode); never fed to the separation
                "effective_res": opts.get("resolution"),
                "paper_keepout": (body_extra or {}).get("paper_keepout"),
                # the plain path's own expression (routes/core.py), so both layers route like the plain job
                "connector_distance": float((body_extra or {}).get("connector_distance",
                                                                   (body_extra or {}).get("join_nearby", 0.0))),
                "response_summary": {k: j.get(k) for k in ("stroke_count", "draw_moves", "pen_lifts", "draw_length_mm",
                                                           "estimated_plot_time_min", "warnings") if k in j}}
    finally:
        for p in list(WORK_DIR.glob(f"src_{did}*")):
            try:
                p.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------- step 2: the separation (pure arrays)
def _disc(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def min_mark_px(opts):
    """trace_v3._trace_svg's dust floor in working-grid px (trace_v3.py:2300-2316, same formula)."""
    fill_px = max(1.2, float(opts.get("fill_px") or 1.2))
    ov = opts.get("_mm_per_px_override")
    mmpx = float(ov) if ov is not None else float(opts.get("fill_mm", 0.15)) / fill_px
    return max(2, int(round(float(opts.get("min_mark_mm2", 0.035)) / (mmpx ** 2))))


def _merge_small(black, blue, min_px):
    """DIRECTIONAL: 4-connected blue fragments below the dust floor that touch black move INTO black."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(blue.astype(np.uint8), connectivity=4)
    small = np.zeros(n, bool)
    small[1:] = st[1:, cv2.CC_STAT_AREA] < min_px
    sm = small[lab] & blue
    touch = cv2.dilate(black.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ln, llab = cv2.connectedComponents(sm.astype(np.uint8), connectivity=4)
    tcomp = np.zeros(ln, bool)
    np.logical_or.at(tcomp, llab[sm & touch], True)
    tcomp[0] = False
    move = tcomp[llab] & sm
    return black | move, blue & ~move, {"blue_small_components": int(ln - 1), "blue_moved_to_black_px": int(move.sum())}


def separate(gray, opts, mm_per_px, blue_amount, strip_in_black=None):
    """Grey raster (the plain job's post-CLAHE array) -> dict of boolean masks + readouts.
    Keys: M0, black, blue, plus E, core, T, t_dark, blue_threshold, radius_px, info."""
    import ccplotter.mask as mk
    strip = STRIP_IN_BLACK if strip_in_black is None else bool(strip_in_black)
    a = clamp_blue_amount(blue_amount)
    M0 = mk._binarize_mask(gray, opts)
    b3 = cv2.GaussianBlur(gray, (3, 3), 0)
    T = float(cv2.threshold(b3, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0])
    Gs = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), GREY_SMOOTH_SIGMA_MM / mm_per_px)
    t_dark = T_DARK_FRAC * T
    r = max(1, int(round(ROLE_WIDTH_MM / mm_per_px)))
    opened = cv2.morphologyEx(M0.astype(np.uint8), cv2.MORPH_OPEN, _disc(r)) > 0
    E = M0 & ~opened
    core = M0 & (Gs < t_dark)
    BLACK = core | E
    bd = cv2.dilate(BLACK.astype(np.uint8), _disc(r)) > 0
    t_blue = T + a * (255.0 - T)
    blue_all = b3 < t_blue
    info = {"T": T, "t_dark": t_dark, "blue_threshold": t_blue, "blue_amount": a, "radius_px": r,
            "mm_per_px": mm_per_px, "strip_in_black": strip, "min_mark_px": min_mark_px(opts)}
    if strip:
        blk = BLACK | (M0 & bd)                       # the sub-pen ring of M0 around BLACK is black
        black, blue, minfo = _merge_small(blk, blue_all & ~blk, info["min_mark_px"])
        info.update(minfo)
    else:                                              # strip left blank
        black, blue = BLACK, blue_all & ~bd
    px2 = mm_per_px ** 2
    info.update({"m0_mm2": float(M0.sum() * px2), "black_mm2": float(black.sum() * px2),
                 "blue_mm2": float(blue.sum() * px2), "E_mm2": float(E.sum() * px2), "core_mm2": float(core.sum() * px2)})
    return {"M0": M0, "black": black, "blue": blue, "E": E, "core": core, "T": T, "t_dark": t_dark,
            "blue_threshold": t_blue, "radius_px": r, "info": info}


# ---------------------------------------------------------------- opt-in: blue margin under black (T1 + X + F)
def trap_fold(black, area_mm2, mm_per_px):
    """F: 8-connected black components with area < area_mm2 are split off. Returns (black_kept, folded_mask, folded_areas_mm2).
    The caller REASSIGNS the folded mask to blue (never deletes it)."""
    px2 = mm_per_px ** 2
    n, lab, st, _ = cv2.connectedComponentsWithStats(black.astype(np.uint8), connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = st[1:, cv2.CC_STAT_AREA] * px2 < area_mm2
    sm = small[lab] & black
    return black & ~sm, sm, st[1:, cv2.CC_STAT_AREA][small[1:]] * px2


def trap_blue_mask(blue_a, black_a, mm_per_px, pen_mm, leak=False):
    """T1 blue mask: blue_A | (dilate(blue_A, d) & black_A), then holes < pi*(hole_r*W)^2 mm2 closed (floor px).
    Returns (T, T_noclose). leak=True drops the black restriction (POSITIVE CONTROL only: must fail the invariants)."""
    from scipy import ndimage as ndi
    from skimage.morphology import remove_small_holes
    px2 = mm_per_px ** 2
    d = TRAP_MARGIN_PEN * pen_mm
    f_px = int(np.floor(np.pi * (TRAP_HOLE_RADIUS_PEN * pen_mm) ** 2 / px2))
    edt = ndi.distance_transform_edt(~blue_a) * mm_per_px
    dil = edt <= d + 1e-12
    tn = blue_a | (dil if leak else (dil & black_a))
    return remove_small_holes(tn, max_size=f_px), tn


def trap_invariants(blue_a, black_a, t, t_noclose, spk, black_f):
    """The standing checks as numbers (used at run time and by the tests, with positive controls):
    contains_blue_A; added_paper_px (addition outside black_A and outside the closed holes) must be 0; closed holes ringed by
    pre-closing blue; fold conserves the blue+black union."""
    from scipy import ndimage as ndi
    closed = t & ~t_noclose
    new = t & ~blue_a
    paper_px = int((new & ~black_a & ~closed).sum())
    ring = ndi.binary_dilation(closed, structure=ndi.generate_binary_structure(2, 1)) & ~closed
    ring_bad = int((ring & ~t_noclose).sum())
    tf = t | spk
    u_before, u_after = black_a | t, black_f | tf
    out = {"contains_blue_A": bool((blue_a & ~t).sum() == 0), "added_px": int(new.sum()),
           "added_in_black_px": int((new & black_a).sum()), "added_closed_holes_px": int((new & closed).sum()),
           "added_paper_px": paper_px, "closed_ring_not_blue_px": ring_bad,
           "union_before_px": int(u_before.sum()), "union_after_px": int(u_after.sum()),
           "union_delta_px": int(u_after.sum()) - int(u_before.sum())}
    out["ok"] = bool(out["contains_blue_A"] and paper_px == 0 and ring_bad == 0 and out["union_delta_px"] == 0)
    return out


def apply_blue_trap(sep, mm_per_px, pen_mm):
    """sep = separate() result. Returns dict(blue, black, info): blue = T1 blue | folded specks, black = A black minus the
    folded specks. Raises KeytintError if an invariant fails (a bug, never a user error)."""
    black_a, blue_a = sep["black"], sep["blue"]
    px2 = mm_per_px ** 2
    f_mm2 = float(np.pi * (TRAP_FOLD_RADIUS_PEN * pen_mm) ** 2)
    t, tn = trap_blue_mask(blue_a, black_a, mm_per_px, pen_mm)
    black_f, spk, spk_areas = trap_fold(black_a, f_mm2, mm_per_px)
    inv = trap_invariants(blue_a, black_a, t, tn, spk, black_f)
    if not inv["ok"]:
        raise KeytintError(f"blue_trap invariant failed: {inv}", 500)
    blue = t | spk
    info = {"enabled": True, "pen_mm": pen_mm, "margin_mm": TRAP_MARGIN_PEN * pen_mm,
            "hole_close_mm2": float(np.pi * (TRAP_HOLE_RADIUS_PEN * pen_mm) ** 2), "fold_mm2": f_mm2,
            "fold_components": int(len(spk_areas)), "fold_mm2_total": float(spk.sum() * px2),
            "blue_mm2_before": float(blue_a.sum() * px2), "blue_mm2_after": float(blue.sum() * px2),
            "black_mm2_before": float(black_a.sum() * px2), "black_mm2_after": float(black_f.sum() * px2),
            "invariants": inv, "blue_connector_distance_mm": TRAP_BLUE_CONNECTOR_MM,
            "constants": CONSTANTS["blue_trap"]}
    return {"blue": blue, "black": black_f, "info": info}


# ---------------------------------------------------------------- step 3: mask -> placed, routed layer
def _page_orient(paper_w, paper_h):
    return "-l " if paper_w > paper_h else ""


def layer_from_mask(mask, opts, geom, workdir: Path, tag, stats_out=None, max_walk_segs=None, check_mask=None, stay_down=False):
    """mask (bool, working grid) -> routed layer SVG in page mm, by the SAME chain composite.py runs for a bwphoto item:
    trace_v3._trace_svg (mask handed in bilevel; _binarize_mask's bilevel path reproduces the mask exactly) -> render
    svg (linesort) -> scale/rotate/translate by the plain job's placement -> linemerge(vec_merge) -> pagesize ->
    low-lift router (geom["connector_distance"], default 0). Returns the routed svg Path, or None when the mask draws nothing.
    check_mask (opt-in blue_trap only; None = the unchanged path): a bool mask on the same grid; connectors are then
    mask-checked against THAT mask's final trace (the blue + black union) via stage_e_mask."""
    import ccplotter.trace_v3 as tv3
    from ccplotter.composite import _apply_global_clip, _paper_keepout_safe_region
    from ccplotter.stats_warnings import _route_low_lift_doc, ROUTE_WALK_MAX_SEGS_PLAIN
    from ccplotter.svg_io import _paper_spec
    if not mask.any():
        return None
    workdir = Path(workdir)
    img = np.where(mask, 0, 255).astype(np.uint8)
    raw = workdir / f"{tag}_trace.svg"
    o1 = dict(opts)
    if stay_down:
        o1["_stage_e_mask"] = True          # persist THIS layer's own final trace mask (stay-down checks against it)
    tv3._trace_svg(img, raw, o1)
    rawsd = None
    if stay_down:
        from ccplotter.stage_e_mask import load_sidecar as _ls, write_sidecar as _ws
        _own = _ls(raw)
        if _own is None:
            raise KeytintError("stay_down_over_ink: the layer mask sidecar was not written; refusing to route unchecked connectors", 500)
        rawsd = workdir / f"{tag}_sd_trace.svg"       # own-mask sidecar kept apart: blue_trap overwrites raw's with the union
        _ws(rawsd, _own[0], _own[1])
    if check_mask is not None:
        from ccplotter.stage_e_mask import load_sidecar, write_sidecar
        o2 = dict(opts)
        o2["_stage_e_mask"] = True
        rawc = workdir / f"{tag}_chk_trace.svg"
        tv3._trace_svg(np.where(check_mask, 0, 255).astype(np.uint8), rawc, o2)
        tm, fpx = load_sidecar(rawc)
        if tm is None:
            raise KeytintError("blue_trap: the union mask sidecar was not written; refusing to route unchecked connectors", 500)
        write_sidecar(raw, tm, fpx)
    q = lambda p: shlex.quote(str(p).replace("\\", "/"))
    render = workdir / f"{tag}_render.svg"
    vpype_cli.execute(f"read {q(raw)} linesort penwidth {float(opts.get('fill_mm', 0.15))}mm write {q(render)}")
    merge_tol = float(opts.get("vec_merge", 0.8))
    spec = _paper_spec(geom["paper_w"], geom["paper_h"])
    clean, routed, conn = workdir / f"{tag}.clean.svg", workdir / f"{tag}.routed.svg", workdir / f"{tag}.connectors.svg"
    # scale/rotate act about the PLAIN job's source bbox centre (vpype's default is the layer's OWN bbox centre, which
    # differs between the blue and black layers), so both layers get exactly the plain job's transform.
    ox, oy = geom["c0_px"]
    vpype_cli.execute(f"read -l 1 {q(render)} scale -l 1 -o {ox!r}px {oy!r}px {geom['fit']} {geom['fit']} "
                      f"rotate -l 1 -o {ox!r}px {oy!r}px -- {float(geom.get('rot', 0.0))} "
                      f"translate -l 1 -- {geom['tx']}mm {geom['ty']}mm linemerge --tolerance {merge_tol}mm -l 1 "
                      f"pagesize {_page_orient(geom['paper_w'], geom['paper_h'])}{spec} write {q(clean)}")
    dist = float(geom.get("connector_distance", 0.0) or 0.0)
    ctx = None
    if check_mask is not None:
        from ccplotter.config import PX2MM
        ox_, oy_ = geom["c0_px"]
        ctx = {1: {"svg": raw, "s": float(geom["fit"]), "rot_deg": float(geom.get("rot", 0.0)), "c0_px": (ox_, oy_),
                   "t_px": (geom["tx"] / PX2MM, geom["ty"] / PX2MM)}}
    sd_ctx = None
    if stay_down:
        from ccplotter.config import PX2MM as _PX
        sd_ctx = {1: {"svg": rawsd, "s": float(geom["fit"]), "rot_deg": float(geom.get("rot", 0.0)), "c0_px": tuple(geom["c0_px"]),
                      "t_px": (geom["tx"] / _PX, geom["ty"] / _PX)}}
    rs = _route_low_lift_doc(clean, routed, dist, None, connectors_svg=conn, mask_ctx=ctx,
                            max_walk_segs=max_walk_segs or ROUTE_WALK_MAX_SEGS_PLAIN,
                            **({"stay_down_ctx": sd_ctx} if sd_ctx is not None else {}))
    if stats_out is not None:       # router stats are free: record them (info readout)
        stats_out.update({"connector_distance_mm": dist, "connector_count": rs.get("connector_count"),
                          "connector_draw_length_mm": rs.get("connector_draw_length_mm")})
        if rs.get("stay_down_over_ink"):
            stats_out["stay_down_over_ink"] = rs["stay_down_over_ink"]
    # Paper keep-out (the Kokuyo hole strip): the SAME server-side clip composite.py applies after routing, with the
    # same region (pulled back by half the resolved fill width), so a layer can never enter the strip.
    try:
        region = _paper_keepout_safe_region(geom["paper_w"], geom["paper_h"], geom.get("paper_keepout"),
                                            0.5 * max(0.0, float(opts.get("fill_mm", 0.15))))
    except ValueError as e:
        raise KeytintError(str(e), 400)
    if region is not None:
        _apply_global_clip(routed, region)
    return routed


def load_strokes_mm(svg_path):
    """All strokes of a (routed) layer SVG as (n,2) float arrays in page mm."""
    from ccplotter.config import PX2MM
    doc = vp.read_multilayer_svg(str(svg_path), quantization=0.1)
    out = []
    for lc in doc.layers.values():
        for line in lc:
            a = np.asarray(line)
            out.append(np.c_[a.real, a.imag] * PX2MM)
    return out


def layer_geom(prep):
    """The placement dict layer_from_mask() needs, from a capture_plain_bwphoto() result."""
    g = {k: prep[k] for k in ("fit", "tx", "ty", "rot", "c0_px", "paper_w", "paper_h", "paper_keepout")}
    g["connector_distance"] = float(prep.get("connector_distance", 0.0) or 0.0)
    return g


def write_layer_svg(strokes, paper_w, paper_h, path, color="#000000"):
    """Strokes (page mm) -> an SVG twopen.check_svg_mm accepts: width/height in mm, viewBox 1 unit = 1 mm."""
    body = "".join('<polyline points="' + " ".join(f"{x:.4f},{y:.4f}" for x, y in s) + '"/>' for s in strokes if len(s) >= 2)
    Path(path).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{paper_w:g}mm" height="{paper_h:g}mm" '
        f'viewBox="0 0 {paper_w:g} {paper_h:g}" fill="none" stroke="{color}" stroke-width="0.12">{body}</svg>', encoding="ascii")
    return Path(path)


def stroke_stats(strokes):
    """strokes / pen_lifts (one per stroke, the app's convention) / draw_mm for a stroke list."""
    draw = float(sum(np.hypot(*np.diff(s, axis=0).T).sum() for s in strokes if len(s) > 1))
    return {"strokes": len(strokes), "pen_lifts": len(strokes), "draw_mm": round(draw, 1)}


# ---------------------------------------------------------------- preview PNG + debug PNGs
def render_png(layers, paper_w, paper_h, out_png, ppm=PREVIEW_PPM, ss=PREVIEW_SS):
    """Two-colour true-width raster preview: blue under black. layers = [(strokes_mm, rgb, width_mm), ...] in draw order."""
    W, H = int(round(paper_w * ppm)), int(round(paper_h * ppm))
    img = np.empty((H, W, 3), np.float32)
    img[:] = np.array(PREVIEW_PAPER_RGB, np.float32)
    for strokes, rgb, width in layers:
        m = np.zeros((H * ss, W * ss), np.uint8)
        th = max(1, int(round(width * ppm * ss)))
        for s in strokes:
            if len(s) < 2:
                continue
            q = (s * ppm * ss * 16).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(m, [q], False, 255, th, cv2.LINE_8, shift=4)
        a = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - a) + rgb[c] * a
    out = cv2.cvtColor((np.clip(img, 0, 255) + .5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_png), out)
    return Path(out_png)


def write_debug_pngs(sep, layers_mm, geom, widths, outdir: Path):
    """S12 inspectability: the masks (working grid) and each layer rendered alone at pen width. Returns {name: path}."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("M0", "black", "blue"):
        p = outdir / f"mask_{name}.png"
        cv2.imwrite(str(p), sep[name].astype(np.uint8) * 255)
        files[f"mask_{name}"] = str(p)
    for color, rgb in (("blue", PREVIEW_BLUE_RGB), ("black", PREVIEW_BLACK_RGB)):
        p = outdir / f"layer_{color}.png"
        render_png([(layers_mm[color], rgb, widths[color])], geom["paper_w"], geom["paper_h"], p)
        files[f"layer_{color}"] = str(p)
    return files


# ---------------------------------------------------------------- orchestration (called by the route)
def generate_layers(prep, blue_amount, jobdir: Path, widths, strip_in_black=None, debug=False, blue_trap=False,
                    blue_fill_mm=None, stay_down_over_ink=False, hatched_blue=False):
    """prep = capture_plain_bwphoto() result. Writes blue/black layer SVGs + preview PNG (+ debug PNGs) into jobdir.
    blue_trap (default False = byte-identical to before): the opt-in blue margin under black (T1 + X + F); its thresholds
    are multiples of blue_fill_mm (the active BLUE profile's fill_mm; falls back to the plain job's fill_mm).
    hatched_blue (default False = byte-identical to before): the opt-in "Hatched blue (lighter, fewer pen lifts)" (arm B3s, ccplotter/twopen_hatch.py).
    PRECEDENCE: ON replaces the whole blue layer by the hatch; the black layer is blue_trap's F black (computed here even if blue_trap is
    False; blue_trap's T1 blue mask and 0.5 mm blue connector are then unused); stay_down_over_ink still applies to the black layer.
    Returns dict(layers{color: svg path}, strokes{color: list}, stats, info, preview_png, debug_files)."""
    from ccplotter.stats_warnings import ROUTE_WALK_MAX_SEGS_TWOPEN
    jobdir = Path(jobdir)
    jobdir.mkdir(parents=True, exist_ok=True)
    geom = layer_geom(prep)
    sep = separate(prep["gray"], prep["opts"], prep["mm_per_px"], blue_amount, strip_in_black)
    svgs, strokes, conn_info = {}, {}, {}
    body_cd = float(prep.get("connector_distance", 0.0) or 0.0)
    trap_info, route_masks, check_mask = None, {"blue": sep["blue"], "black": sep["black"]}, None
    pen_mm = float(blue_fill_mm if blue_fill_mm else prep["opts"].get("fill_mm", ROLE_WIDTH_MM))
    if hatched_blue and not sep["info"]["strip_in_black"]:
        raise KeytintError("hatched_blue is defined with the strip-in-black rule (the recipe default)", 400)
    if blue_trap or hatched_blue:
        tr = apply_blue_trap(sep, prep["mm_per_px"], pen_mm)
        trap_info, route_masks = tr["info"], {"blue": tr["blue"], "black": tr["black"]}
        check_mask = tr["blue"] | tr["black"]              # X: blue connectors may only lie inside the blue + black union
        if hatched_blue and not blue_trap:                 # the F black only; the T1 blue is not used
            trap_info = {"enabled": False, "black_fold_implied_by_hatched_blue": True, "pen_mm": pen_mm,
                         "fold_mm2": tr["info"]["fold_mm2"], "fold_components": tr["info"]["fold_components"],
                         "fold_mm2_total": tr["info"]["fold_mm2_total"], "black_mm2_before": tr["info"]["black_mm2_before"],
                         "black_mm2_after": tr["info"]["black_mm2_after"], "invariants": tr["info"]["invariants"]}
    for color, rgb in (("blue", "#1f4fd8"), ("black", "#111111")):
        cd = BLUE_CONNECTOR_MM if color == "blue" else body_cd         # the CONSTANTS["connector_rule"]
        if blue_trap and color == "blue":
            cd = TRAP_BLUE_CONNECTOR_MM
        conn_info[color] = {"connector_distance_mm": cd, "connector_count": 0, "connector_draw_length_mm": 0.0}
        capped = color == "black" or (blue_trap and color == "blue")    # key layer; the trap blue layer too (both can produce tens of thousands of strokes)
        if hatched_blue and color == "blue":
            from ccplotter.twopen_hatch import hatched_blue_layer
            routed = hatched_blue_layer(prep, sep, geom, prep["opts"], jobdir, color, pen_mm, stats_out=conn_info[color])
        else:
            routed = layer_from_mask(route_masks[color], prep["opts"], dict(geom, connector_distance=cd), jobdir, color, stats_out=conn_info[color],
                                     max_walk_segs=ROUTE_WALK_MAX_SEGS_TWOPEN if (capped or stay_down_over_ink) else None,
                                     check_mask=check_mask if (blue_trap and color == "blue" and not hatched_blue) else None,
                                     stay_down=bool(stay_down_over_ink))
        strokes[color] = load_strokes_mm(routed) if routed else []
        svgs[color] = (write_layer_svg(strokes[color], prep["paper_w"], prep["paper_h"], jobdir / f"{color}_mm.svg", rgb)
                       if strokes[color] else None)
    stats = {c: stroke_stats(strokes[c]) for c in strokes}
    preview = render_png([(strokes["blue"], PREVIEW_BLUE_RGB, widths["blue"]),
                          (strokes["black"], PREVIEW_BLACK_RGB, widths["black"])],
                         prep["paper_w"], prep["paper_h"], jobdir / "preview_keytint.png")
    dbg = write_debug_pngs(sep, strokes, geom, widths, jobdir / "debug") if debug else {}
    info = dict(sep["info"])
    info.update({"constants": CONSTANTS, "placement": {"fit": prep["fit"], "tx": prep["tx"], "ty": prep["ty"]},
                 "trace_calls_captured": prep["trace_calls"], "effective_res": prep["effective_res"],
                 "plain_job": prep["response_summary"], "pass_order": ["blue", "black"],
                 "body_connector_distance_mm": body_cd, "connectors": conn_info})
    info["blue_trap"] = trap_info if trap_info else {"enabled": False}
    if hatched_blue:        # recorded only when on: option, constants, the hatch audit; the crack figures are NOT computed by the job
        info["hatched_blue"] = {"enabled": True, "pen_mm": pen_mm, "layer": conn_info["blue"].get("hatched_blue"),
                                "constants": CONSTANTS["hatched_blue"],
                                "crack_figures": "not computed by the job; offline prototype figures (uncovered-by-either and uncovered-by-black) are in constants.prototype_measurements"}
    if stay_down_over_ink:      # recorded only when on: option + per-layer router audit (candidates / fused / rejected / mm)
        info["stay_down_over_ink"] = {"enabled": True, "rule": "pen-up travel -> pen-down connector of any length iff the straight "
                                      "connector stays inside that layer's own ink mask (stage_e_mask predicate)",
                                      "layers": {c: conn_info[c].get("stay_down_over_ink") for c in conn_info},
                                      "walk_cap": ROUTE_WALK_MAX_SEGS_TWOPEN}
        if hatched_blue:        # the hatch linker (pen-down only inside blue u black) is the blue layer's own stay-down; the router is not run on it
            info["stay_down_over_ink"]["layers"]["blue"] = {"not_applicable": "hatched_blue links its own runs, mask-checked against blue u black"}
    return {"layers": svgs, "strokes": strokes, "stats": stats, "info": info, "preview_png": preview,
            "debug_files": dbg, "sep": sep}
