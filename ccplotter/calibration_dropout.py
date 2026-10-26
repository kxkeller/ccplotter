"""ccplotter/calibration_dropout.py -- POSITION/DIRECTION dropout map sheet.

What it measures: on a hatch job a large share of commanded pen-down length can carry no
ink, much of it in coherent gaps >=1 mm long -- a DROPOUT (pen not making contact or not
depositing), not a uniformly thinner line. This sheet finds out whether dropout is a
function of POSITION on the plate (pen height, bed or paper flatness) or is roughly
uniform (ink flow / starvation), and secondarily whether direction matters and whether a
second pass recovers a dropout. This module only builds and validates the g-code; the
analysis is done on a scan of the plotted sheet.

How to read it: scan the sheet and measure the inked fraction of each isolated line.
Plot continuity against line Y (PRIMARY band) and against direction (DIRECTION band); a
trend with Y points at plate/paper height, no trend points at ink supply. Compare REPEAT
lines against their single-pass neighbours to see whether a second pass fills gaps.

HARD-CODED MACHINE AND PEN (limitation). The sheet is tied to the author's measured
machine: it refuses to run unless the active reach profile is EXPECT_REACH_ID
("plotter_2003_v32") with bbox EXPECT_BBOX, and it IGNORES the `profile` argument the
dispatcher passes. It always re-resolves EXPECT_PEN_ID ("uright_gel", never
"uright_gel_halfpitch") out of pen_profiles.json and hard-asserts
label / pen_mm / pen_offset / z_touch_abs / z_down / draw_speed against the values the
sheet was designed for. A silent pen substitution would put the wrong pen on the legend,
so it is refused. To use another machine or pen, edit the EXPECT_* constants.

NO SWEEP. Z and draw speed stay FIXED at the pinned profile's stock values
(z_down=-0.6, draw_speed=3000); the question is position, not pressure or speed. There is
exactly one gwrite cell and no _splice_blocks multi-block splice, so none of that
splice's failure modes apply.

PAPER / KEEP-OUT. JIS B5 257x182 mm landscape with a 10 mm hole-keepout height budget,
the same numbers as calibration_penwidth.py / calibration_singlepass.py. (config.py's
PAPER_SIZES['b5'] is the ISO 250x176 size and is not used.) The keepout is budgeted as a
HEIGHT reduction only; this module does not decide which physical edge (front or back)
the punched-hole strip sits on -- that follows the paper's placement on the plate. The
app's live paper_keepout default (assets/templates/index.html) is edge 'front'.

NO FIDUCIALS. Registration for scan-side analysis uses the g-code mm to photo px mapping
(origin_mm and MM_PER_PX from the raster's meta.json plus a correction affine) instead of
optical fiducials: a fiducial that is partially inked or voided biases its own centroid.
The sheet carries only a plain footprint outline (`calibration_plots.rect_outline`).

DESIGN, THREE BANDS:
  PRIMARY   -- long isolated horizontal lines, evenly spaced over most of the reachable
               height. The POSITION map: continuity of each line as a function of its Y,
               decorrelated from plot time and from within-stroke start position.
  DIRECTION -- isolated vertical lines, same idea, to test direction dependence.
  REPEAT    -- a few lines drawn TWICE (identical two-pass) beside otherwise identical
               single-pass lines, to test whether a second pass recovers a dropout.

CONFOUND CONTROLS BUILT INTO THE GEOMETRY:
  * PRIMARY lines are emitted in a FIXED SEEDED PERMUTATION of plot order
    (DRAW_ORDER_SEED), decoupling Y-position from time since plot start; otherwise an
    ink-starvation-over-time effect would look like a Y gradient. The permutation is
    recorded in the returned info dict (`primary_draw_order`).
  * Alternate direction: PRIMARY line index k (by Y-order, not draw order) is drawn
    left-to-right if k is even, right-to-left if odd; DIRECTION line k alternates
    top-to-bottom / bottom-to-top the same way. This decorrelates position from
    arc-length since pen-down, because a fresh pen-down starts DRY (about 40% ink in the
    first 0.15 mm vs 60% past 3.5 mm), which would alias with a position effect at
    whichever end every line starts.
  * ONE deliberately widened gap at a fixed PRIMARY index (WIDE_GAP_AFTER_IDX) breaks the
    pitch's periodicity, so an indexing script cannot lose count if an end line drops
    out entirely.
  * REPEAT-band lines are geometrically separate from PRIMARY and are excluded from the
    primary Y-gradient analysis.

NO LINESORT: the emitted polyline order IS the plot order; the permutation above is
applied when the list is built, not left to routing.

Reached only through the generic POST /api/calibration_sheet dispatcher (id="dropout_map"
in calibration_registry.py).
"""
import math
import random
import re
import shlex
from pathlib import Path

import vpype as vp
import vpype_cli

from ccplotter.atomic_io import atomic_target
from ccplotter.config import CONFIG_PATH, PREVIEW_COLOR, PX2MM, WORK_DIR
from ccplotter.cal import resolve_pen_offset, load_cal, load_pen_profiles, resolve_z_touch
from ccplotter.gcode_post import (_gcode_lifecycle_values, validate_calibration_gcode_safety,
                                  _fix_gwrite_source_comment)
from ccplotter.reach import active_reach_profile, validate_gcode_xy_motion, pen_reach_profile
from ccplotter.svg_io import _write_paths_svg
from ccplotter.calibration_plots import rect_outline

# ---------------------------------------------------------------------------
# Expected runtime envelope -- re-verified at generation time, not assumed (same
# convention as calibration_singlepass.py's EXPECT_* guards: an environment without
# reach_profiles falls back to a different default reach profile and would plot
# against the wrong envelope). Tied to the author's measured machine geometry.
# ---------------------------------------------------------------------------
EXPECT_REACH_ID = "plotter_2003_v32"
EXPECT_BBOX = (3.0, 42.5, 255.0, 254.5)
EXPECT_PEN_ID = "uright_gel"
EXPECT_PEN_LABEL = "uRight Gel refill"
EXPECT_PEN_MM = 0.255
EXPECT_PEN_OFFSET = (0.0, 0.0)
EXPECT_Z_TOUCH_ABS = 11.25
EXPECT_Z_DOWN_STOCK = -0.60
EXPECT_DRAW_SPEED_STOCK = 3000

# ---------------------------------------------------------------------------
# Paper / keep-out (mm). Same convention and numbers as
# calibration_penwidth.py / calibration_singlepass.py -- see module docstring.
# ---------------------------------------------------------------------------
PAPER_W_MM, PAPER_H_MM = 257.0, 182.0
HOLE_KEEPOUT_MM = 10.0
SHEET_MARGIN_MM = 5.0
_PAD_MM = 3.0
_USABLE_W = PAPER_W_MM - 2 * SHEET_MARGIN_MM
_USABLE_H = PAPER_H_MM - HOLE_KEEPOUT_MM - 2 * SHEET_MARGIN_MM

# ---------------------------------------------------------------------------
# PRIMARY band -- the position map.
# ---------------------------------------------------------------------------
PRIMARY_LEN_MM = 200.0
PRIMARY_PITCH_MM = 5.0
PRIMARY_N = 15
WIDE_GAP_AFTER_IDX = 7          # breaks pitch periodicity at a fixed index
WIDE_GAP_EXTRA_MM = 4.0
DRAW_ORDER_SEED = 20260909      # fixed seed so the plot order is reproducible

# ---------------------------------------------------------------------------
# DIRECTION band -- direction dependence.
# ---------------------------------------------------------------------------
DIRECTION_LEN_MM = 40.0
DIRECTION_PITCH_MM = 5.0
DIRECTION_N = 16

# ---------------------------------------------------------------------------
# REPEAT band -- single vs double pass, paired.
# ---------------------------------------------------------------------------
REPEAT_LEN_MM = 70.0
REPEAT_PITCH_MM = 4.0
N_REPEAT_PAIRS = 3

BAND_GAP_MM = 8.0

PENDOWN_MAX_MM = round(PRIMARY_LEN_MM + 5.0, 1)   # longest legitimate stroke
                                                    # is one PRIMARY line;
                                                    # this is a length guard
                                                    # ON TOP OF the geometry
                                                    # (axis-alignment) guard
                                                    # below, not instead of it.
AXIS_TOL_MM = 0.02


def _primary_ys():
    ys = []
    y = 0.0
    for k in range(PRIMARY_N):
        ys.append(y)
        y += PRIMARY_PITCH_MM
        if k == WIDE_GAP_AFTER_IDX:
            y += WIDE_GAP_EXTRA_MM
    return ys


PRIMARY_YS = _primary_ys()
PRIMARY_H = PRIMARY_YS[-1]
PRIMARY_WIDE_GAP_MM = PRIMARY_PITCH_MM + WIDE_GAP_EXTRA_MM

DIRECTION_XS = [k * DIRECTION_PITCH_MM for k in range(DIRECTION_N)]
DIRECTION_SPAN_MM = DIRECTION_XS[-1]

REPEAT_H = (2 * N_REPEAT_PAIRS - 1) * REPEAT_PITCH_MM

DIRECTION_Y0 = PRIMARY_H + BAND_GAP_MM
REPEAT_Y0 = DIRECTION_Y0 + DIRECTION_LEN_MM + BAND_GAP_MM

FOOTPRINT_W = max(PRIMARY_LEN_MM, DIRECTION_SPAN_MM, REPEAT_LEN_MM)
FOOTPRINT_H = REPEAT_Y0 + REPEAT_H

assert FOOTPRINT_W + 2 * _PAD_MM <= _USABLE_W, (
    f"dropout-map sheet footprint width {FOOTPRINT_W + 2 * _PAD_MM:.1f}mm exceeds "
    f"the JIS B5 landscape usable width {_USABLE_W:.1f}mm "
    f"({PAPER_W_MM}mm paper - 2x{SHEET_MARGIN_MM}mm margin)")
assert FOOTPRINT_H + 2 * _PAD_MM <= _USABLE_H, (
    f"dropout-map sheet footprint height {FOOTPRINT_H + 2 * _PAD_MM:.1f}mm exceeds "
    f"the JIS B5 landscape usable height {_USABLE_H:.1f}mm "
    f"({PAPER_H_MM}mm paper - {HOLE_KEEPOUT_MM}mm hole keep-out - "
    f"2x{SHEET_MARGIN_MM}mm margin)")


def build_primary_band():
    """Returns (polys, meta). meta records Y, direction and draw-order index
    per line -- everything the scan analysis needs, so it does not have to
    re-derive geometry from the gcode alone."""
    lines = []
    for k, y in enumerate(PRIMARY_YS):
        forward = (k % 2 == 0)
        pts = [(0.0, y), (PRIMARY_LEN_MM, y)] if forward else [(PRIMARY_LEN_MM, y), (0.0, y)]
        lines.append({"idx": k, "y_mm": round(y, 3), "direction": "LR" if forward else "RL",
                      "pts": pts})
    order = list(range(PRIMARY_N))
    random.Random(DRAW_ORDER_SEED).shuffle(order)
    polys = [lines[i]["pts"] for i in order]
    meta = [{"idx": lines[i]["idx"], "y_mm": lines[i]["y_mm"], "direction": lines[i]["direction"],
             "draw_order_position": pos} for pos, i in enumerate(order)]
    meta_by_idx = sorted(meta, key=lambda m: m["idx"])
    return polys, meta_by_idx, order


def build_direction_band():
    lines = []
    for k, x in enumerate(DIRECTION_XS):
        forward = (k % 2 == 0)
        y0, y1 = DIRECTION_Y0, DIRECTION_Y0 + DIRECTION_LEN_MM
        pts = [(x, y0), (x, y1)] if forward else [(x, y1), (x, y0)]
        lines.append({"idx": k, "x_mm": round(x, 3), "direction": "TB" if forward else "BT",
                      "pts": pts})
    polys = [ln["pts"] for ln in lines]
    meta = [{"idx": ln["idx"], "x_mm": ln["x_mm"], "direction": ln["direction"]} for ln in lines]
    return polys, meta


def build_repeat_band():
    polys, meta = [], []
    for p in range(N_REPEAT_PAIRS):
        for role, row in (("single", 2 * p), ("double", 2 * p + 1)):
            y = REPEAT_Y0 + row * REPEAT_PITCH_MM
            forward = (row % 2 == 0)
            pts = [(0.0, y), (REPEAT_LEN_MM, y)] if forward else [(REPEAT_LEN_MM, y), (0.0, y)]
            n_passes = 1 if role == "single" else 2
            for pass_i in range(n_passes):
                polys.append(list(pts))
            meta.append({"pair": p, "role": role, "y_mm": round(y, 3),
                        "direction": "LR" if forward else "RL", "n_passes": n_passes})
    return polys, meta


def build_all():
    """Full sheet in local mm coords (y grows down). Returns (polys, info)."""
    p_polys, p_meta, p_order = build_primary_band()
    d_polys, d_meta = build_direction_band()
    r_polys, r_meta = build_repeat_band()
    frame = rect_outline(-2.0, -2.0, FOOTPRINT_W + 4.0, FOOTPRINT_H + 4.0)
    all_polys = list(p_polys) + list(d_polys) + list(r_polys) + list(frame)
    n_pendowns_expected = len(all_polys)
    info = {
        "primary": {"n": PRIMARY_N, "len_mm": PRIMARY_LEN_MM, "pitch_mm": PRIMARY_PITCH_MM,
                    "wide_gap_after_idx": WIDE_GAP_AFTER_IDX,
                    "wide_gap_mm": PRIMARY_WIDE_GAP_MM,
                    "draw_order_seed": DRAW_ORDER_SEED,
                    "draw_order": p_order, "lines": p_meta},
        "direction": {"n": DIRECTION_N, "len_mm": DIRECTION_LEN_MM,
                      "pitch_mm": DIRECTION_PITCH_MM, "lines": d_meta},
        "repeat": {"n_pairs": N_REPEAT_PAIRS, "len_mm": REPEAT_LEN_MM,
                   "pitch_mm": REPEAT_PITCH_MM, "lines": r_meta},
        "n_pendowns_expected": n_pendowns_expected,
        "band_origins_mm": {"primary_y0": 0.0, "direction_y0": DIRECTION_Y0,
                            "repeat_y0": REPEAT_Y0},
    }
    return all_polys, info


# ---------------------------------------------------------------------------
# Safety self-checks. Geometry-based (axis-alignment), not just a length
# threshold -- a length-only guard sized for ~200mm legitimate strokes cannot
# catch a diagonal drag shorter than that.
# ---------------------------------------------------------------------------
def _assert_no_pendown_travel(path, z_down_abs):
    x = y = None
    down = False
    worst_len = 0.0
    worst_diag = 0.0
    n_len_violations = 0
    n_diag_violations = 0
    n_examined = 0
    with Path(path).open() as fh:
        for line in fh:
            if not line.startswith("G1"):
                continue
            mz = re.search(r"Z(-?[\d.]+)", line)
            if mz:
                down = abs(round(float(mz.group(1)), 2) - round(z_down_abs, 2)) < 1e-6
                continue
            mx = re.search(r"X(-?[\d.]+)", line)
            my = re.search(r"Y(-?[\d.]+)", line)
            nx = float(mx.group(1)) if mx else x
            ny = float(my.group(1)) if my else y
            if down and x is not None and nx is not None:
                dx, dy = abs(nx - x), abs(ny - y)
                d = math.hypot(dx, dy)
                n_examined += 1
                worst_len = max(worst_len, d)
                if d > PENDOWN_MAX_MM:
                    n_len_violations += 1
                if d > 1e-9:
                    off_axis = min(dx, dy)
                    worst_diag = max(worst_diag, off_axis)
                    if off_axis > AXIS_TOL_MM:
                        n_diag_violations += 1
            x, y = nx, ny
    if n_len_violations or n_diag_violations:
        raise RuntimeError(
            f"Dropout-map sheet has {n_len_violations} pen-DOWN move(s) over "
            f"{PENDOWN_MAX_MM}mm (worst {worst_len:.1f}mm) and {n_diag_violations} "
            f"non-axis-aligned pen-DOWN move(s) (worst off-axis {worst_diag:.3f}mm > "
            f"{AXIS_TOL_MM}mm tolerance). Every drawn line in this sheet is a pure "
            f"horizontal or vertical stroke; a diagonal or over-length pen-down move "
            f"means the pen dragged between lines. Do not plot this file.")
    return {
        "pendown_travel_violations": n_len_violations,
        "pendown_diagonal_violations": n_diag_violations,
        "max_pendown_mm": round(worst_len, 2),
        "max_pendown_offaxis_mm": round(worst_diag, 4),
        "pendown_segments_examined": n_examined,
    }


def _count_pendown_segments(path, z_down_abs):
    n_down_cycles = 0
    down = False
    with Path(path).open() as fh:
        for line in fh:
            if not line.startswith("G1"):
                continue
            mz = re.search(r"Z(-?[\d.]+)", line)
            if not mz:
                continue
            is_down = abs(round(float(mz.group(1)), 2) - round(z_down_abs, 2)) < 1e-6
            if is_down and not down:
                n_down_cycles += 1
            down = is_down
    return n_down_cycles


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------
def dropout_gcode(out_gcode: Path, pen_mm=None, z_up=3.5,
                  travel_speed=8000, plunge_speed=500, lift_speed=2000,
                  offset_x=None, offset_y=None, end_policy="release",
                  quiet_fans=True, profile=None):
    """Emit the full dropout-map sheet as ONE g-code file, ONE gwrite call
    (no sweep, no splice). `profile` is accepted for signature parity with
    the generic dispatcher but is IGNORED: the sheet always re-resolves
    EXPECT_PEN_ID directly (see module docstring)."""
    del profile
    ppmm = 1.0 / PX2MM
    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()

    reach0 = active_reach_profile(cal)
    if reach0["id"] != EXPECT_REACH_ID or tuple(round(v, 1) for v in reach0["bbox"]) != EXPECT_BBOX:
        raise RuntimeError(
            f"Runtime reach envelope mismatch: got id={reach0['id']!r} bbox={reach0['bbox']}, "
            f"expected id={EXPECT_REACH_ID!r} bbox={EXPECT_BBOX}. Refusing to generate against "
            f"an unexpected envelope.")

    profiles = load_pen_profiles()
    if EXPECT_PEN_ID not in profiles.get("profiles", {}):
        raise RuntimeError(f"pen profile {EXPECT_PEN_ID!r} not found in pen_profiles.json")
    pen = profiles["profiles"][EXPECT_PEN_ID]

    label = pen.get("label")
    if label != EXPECT_PEN_LABEL:
        raise RuntimeError(f"pen {EXPECT_PEN_ID!r} label mismatch: got {label!r}, "
                           f"expected {EXPECT_PEN_LABEL!r}. Refusing (this is exactly the "
                           f"repo-root pen_profiles.json CWD shadow / stale-runtime-copy "
                           f"failure mode).")
    if abs(float(pen.get("pen_mm", 0.0)) - EXPECT_PEN_MM) > 1e-6:
        raise RuntimeError(f"pen {EXPECT_PEN_ID!r} pen_mm mismatch: got {pen.get('pen_mm')}, "
                           f"expected {EXPECT_PEN_MM}")
    got_offset = (float(pen.get("pen_offset_x", 0.0)), float(pen.get("pen_offset_y", 0.0)))
    if any(abs(a - b) > 1e-6 for a, b in zip(got_offset, EXPECT_PEN_OFFSET)):
        raise RuntimeError(f"pen {EXPECT_PEN_ID!r} pen_offset mismatch: got {got_offset}, "
                           f"expected {EXPECT_PEN_OFFSET}")

    z_touch, z_touch_source = resolve_z_touch(cal, pen)
    if abs(z_touch - EXPECT_Z_TOUCH_ABS) > 1e-6:
        raise RuntimeError(f"z_touch_abs mismatch for {EXPECT_PEN_ID}: got {z_touch}, "
                           f"expected {EXPECT_Z_TOUCH_ABS}")
    z_down = float(pen.get("z_down", 0.0))
    if abs(z_down - EXPECT_Z_DOWN_STOCK) > 1e-6:
        raise RuntimeError(f"stock z_down mismatch for {EXPECT_PEN_ID}: got {z_down}, "
                           f"expected {EXPECT_Z_DOWN_STOCK}")
    draw_speed = float(pen.get("draw_speed", 0.0))
    if abs(draw_speed - EXPECT_DRAW_SPEED_STOCK) > 1e-6:
        raise RuntimeError(f"stock draw_speed mismatch for {EXPECT_PEN_ID}: got {draw_speed}, "
                           f"expected {EXPECT_DRAW_SPEED_STOCK}")

    reach = pen_reach_profile(reach0, pen)
    if pen_mm is None:
        pen_mm = EXPECT_PEN_MM

    pen_off_x, pen_off_y = resolve_pen_offset(cal, pen)
    PLACEMENT_MARGIN_MM = 20.0
    pad = _PAD_MM
    canvas_w_mm, canvas_h_mm = FOOTPRINT_W + 2 * pad, FOOTPRINT_H + 2 * pad

    offset_x = reach["bbox"][0] + PLACEMENT_MARGIN_MM if offset_x is None else float(offset_x)
    if offset_y is None:
        # Same vertical_flip handling as calibration_penwidth.py's offset_y default:
        # elegoo_cc's gwrite profile mirrors Y about max(canvas_w_mm, canvas_h_mm) for
        # a landscape canvas, not about the canvas height.
        flip_axis_mm = max(canvas_w_mm, canvas_h_mm)
        near_edge_target = reach["bbox"][1] + PLACEMENT_MARGIN_MM
        comp_off_y = near_edge_target - flip_axis_mm + pad + FOOTPRINT_H
        offset_y = comp_off_y + pen_off_y
    else:
        offset_y = float(offset_y)
    comp_off_x, comp_off_y = offset_x - pen_off_x, offset_y - pen_off_y
    park_x, park_y = reach["safe_park_point"]
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))

    W = int(round(canvas_w_mm * ppmm))
    H = int(round(canvas_h_mm * ppmm))

    all_polys, info = build_all()
    placed = [[((x + pad) * ppmm, (y + pad) * ppmm) for x, y in p] for p in all_polys]
    svg = WORK_DIR / f".{out_gcode.stem}.svg"
    _write_paths_svg(svg, W, H, float(pen_mm) * ppmm, polylines=placed)

    z_down_abs = z_touch + z_down
    vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({
        "z_up": float(z_up), "z_up_abs": z_touch + float(z_up),
        "z_down": z_down, "z_down_abs": z_down_abs,
        "z_touch_abs": z_touch, "z_travel_abs": z_travel_abs,
        "draw_speed": draw_speed, "travel_speed": float(travel_speed),
        "plunge_speed": float(plunge_speed), "lift_speed": float(lift_speed),
        "z_auto": "1", "origin_x": park_x, "origin_y": park_y,
        **_gcode_lifecycle_values(end_policy, quiet_fans)})

    with atomic_target(out_gcode) as tmp_gcode:
        vpype_cli.execute(f'read {shlex.quote(str(svg))} pagesize {W}x{H}px '
                          f'translate -- {comp_off_x}mm {-comp_off_y}mm '
                          f'gwrite --profile elegoo_cc {shlex.quote(str(tmp_gcode))}')
        _fix_gwrite_source_comment(tmp_gcode, out_gcode)
    svg.unlink(missing_ok=True)

    validate_gcode_xy_motion(out_gcode, reach)
    travel_check = _assert_no_pendown_travel(out_gcode, z_down_abs)
    safety = validate_calibration_gcode_safety(out_gcode, z_down_abs)
    n_pendown_total = _count_pendown_segments(out_gcode, z_down_abs)
    n_pendown_expected_total = info["n_pendowns_expected"]

    prev_svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_paths_svg(prev_svg, W, H, float(pen_mm) * ppmm, polylines=placed)
    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(prev_svg))} color {PREVIEW_COLOR} '
                          f'write {shlex.quote(str(tmp_prev))}')

    return {"kind": "dropout_map", "pen_mm": pen_mm, "pen_id": EXPECT_PEN_ID,
            "pen_label": EXPECT_PEN_LABEL,
            "footprint_mm": [round(canvas_w_mm, 1), round(canvas_h_mm, 1)],
            "usable_mm": [round(_USABLE_W, 1), round(_USABLE_H, 1)],
            "offset_mm": [offset_x, offset_y],
            "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
            "z_down": z_down, "z_down_abs": round(z_down_abs, 3), "draw_speed": draw_speed,
            "reach_id": reach0["id"], "reach_bbox": reach0["bbox"],
            "n_pendown_expected_total": n_pendown_expected_total,
            "n_pendown_actual_total": n_pendown_total,
            "pendown_count_matches": n_pendown_total == n_pendown_expected_total,
            "deepest_z_down_abs": round(z_down_abs, 3),
            "safety_selfcheck": safety, "travel_selfcheck": travel_check,
            "preview": prev.name, "svg_path": str(prev_svg), "svg_wh_px": [W, H],
            "pattern_mm": [round(canvas_w_mm, 1), round(canvas_h_mm, 1)],
            "place": {"x": offset_x, "y": offset_y, "w": canvas_w_mm, "h": canvas_h_mm},
            "z_up_abs": round(z_touch + float(z_up), 3), "z_down_abs": round(z_down_abs, 3),
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
            "bands": info}
