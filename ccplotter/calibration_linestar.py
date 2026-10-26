"""ccplotter/calibration_linestar.py -- 16-direction isolated-pass "line star" sheet (uright_gel).

What it measures: a gel pen can lay a WIDER line at 0 deg (~0.29 mm) than on diagonals.
Four candidate mechanisms could explain that, and this sheet separates them from ONE
plotted, ONE scanned sheet:
  * tip / mount TILT -- a difference between opposite TRAVEL SENSES of the same line
    (e.g. 0 deg vs 180 deg);
  * AXIS ASYMMETRY -- 0 deg vs 90 deg, independent of diagonal effects;
  * DIAGONAL ANISOTROPY -- the 45/135 family vs the 0/90 family;
  * SCANNER SCALE ANISOTROPY -- a fake effect from a non-square scan, not the pen.

How to read it: measure the laid width of each isolated pass (central 60 % of its length,
see MEASURE_CENTRAL_FRACTION) with a separate measurement script (not part of this
repository) and group by direction. The returned `layout` dict (direction_id, repeat_idx,
grid_row/col, order_idx, design and machine-frame angle, start/end mm) is the key to map
scan strokes back to directions. The expected signatures for each mechanism are in
PREDICTION and are written into the g-code header (`_PREDICTION_LINES`, pure ASCII
comment lines) and the returned response dict, before anything is plotted.

DESIGN:
  * Isolated single passes only, >=3 mm clearance between any two elements.
  * 16 directions = 8 angles (0,22.5,...,157.5 deg) x both travel senses -- equivalently,
    16 full-circle directions spaced 22.5 deg apart (0,22.5,...,337.5 deg): direction_id d
    has design angle d*22.5 deg, and d, d+8 (mod 16) are the same LINE drawn in opposite
    travel senses.
  * Each pass >=15 mm long.
  * >=3 repeats per direction, positioned by a Latin-square layout (see
    `_build_latin_square`) so plate position and plot order are never confounded with
    angle; PASS EMISSION ORDER is separately randomized with a fixed seed (`SEED`).
  * Stock speed/Z from the profile; angle is the only variable, no sweep.
  * On-sheet scale fiducials at 0/90/45 deg with tick marks at a known 10 mm pitch, so a
    scan's per-axis scale self-calibrates independently of the pen-width question.
  * A small optional coverage twin (four 8x8 mm hatched patches at 0/45/90/135 deg,
    0.24 mm pitch) rides along since the whole sheet plots in a few minutes (model (a); see
    `linestar_gcode`'s returned `plot_time_estimate_min`).
  * No per-element printed ids: a tiny ASCII label would either collide with the 3 mm
    clearance budget or force the grid past the reach envelope. Grid position is
    deliberately NOT visible in reading order; that is the confound this sheet breaks.
  * An asymmetric 3-stroke "F" glyph (see `_f_glyph`) sits in a fixed corner so a
    flipped or rotated scan is detectable before any measurement is trusted.

HARD-CODED MACHINE AND PEN (limitation). The sheet is tied to the author's measured
machine: it refuses to run unless the active reach profile is EXPECT_REACH_ID
("plotter_2003_v32") with bbox EXPECT_BBOX. The pen is PINNED: the layout and pass
geometry are generic, but the PREDICTION numbers come from uright_gel's own wedge
reading, so the generator IGNORES the dispatcher's active pen and always re-resolves
EXPECT_PEN_ID ("uright_gel") out of pen_profiles.json. If `profile` is passed and names
a different pen, it refuses loudly rather than silently emitting uright_gel's g-code
under another pen's name and Z. `profile=None` always generates the pinned sheet.
label / pen_mm / pen_offset / z_down / draw_speed are hard-asserted against fixed
constants.

Z_TOUCH_ABS IS A SANITY RANGE, NOT A PINNED LITERAL: it changes whenever the pen is
recalibrated, so a literal would go stale. It is asserted FINITE and inside
Z_TOUCH_SANITY_RANGE_MM (5-20 mm), and the resolved value and its source
(`resolve_z_touch`) are returned in the info dict and written to the g-code header
exactly as read.

CLEARANCE RULE, SCOPED. The >=3 mm clearance rule is enforced between every ISOLATED
GROUP -- each direction pass, each fiducial (baseline plus its own ticks, as one group),
the F glyph (its 3 strokes, as one group), and each coverage-twin PATCH (its whole hatch,
as one group) -- via `_check_clearance`, computed in design (pre-gcode) space with
`shapely` LineString/MultiLineString distances. It is NOT enforced between the individual
0.24 mm-pitch hatch lines WITHIN one coverage-twin patch: that spacing is the coverage
measurement itself (the same "own-patch hatch is not a cross-element drag" distinction
`calibration_fillmode.py`'s `_pendown_check` draws). The frame outline and the single
on-sheet legend line are page furniture, not measurement elements, and are excluded from
the clearance check.

ANGLE READ-BACK (mirror-transform auto-detection). Every pass element is a straight
2-point polyline, so its machine-frame direction is read directly out of the emitted
g-code (`_parse_strokes`), never inferred from the `translate`/`gwrite` pipeline. That
pipeline (one `translate` + `gwrite --profile elegoo_cc`, as in every sibling generic
sheet) has exactly one geometric degree of freedom beyond translation: whether Y is
mirrored. `_detect_mirror` finds that sign EMPIRICALLY from the emitted stroke at
direction_id=4 (design angle 90 deg, a pure "+y" design vector with no x-component), then
every OTHER pass element's machine-frame angle is asserted within 0.5 deg of the
transform applied to its own design angle; a cross-element drag or an unexpected extra
rotation fails this.

Reached only through the generic POST /api/calibration_sheet dispatcher (id="linestar"
in calibration_registry.py).
"""
from __future__ import annotations

import math
import random
import re
import shlex
from pathlib import Path

import vpype as vp
import vpype_cli
from shapely.geometry import LineString, MultiLineString

from ccplotter.atomic_io import atomic_target
from ccplotter.config import CONFIG_PATH, PREVIEW_COLOR, PX2MM, WORK_DIR
from ccplotter.cal import resolve_pen_offset, load_cal, load_pen_profiles, resolve_z_touch
from ccplotter.gcode_post import (_gcode_lifecycle_values, validate_calibration_gcode_safety,
                                  _fix_gwrite_source_comment)
from ccplotter.reach import active_reach_profile, validate_gcode_xy_motion, pen_reach_profile
from ccplotter.svg_io import _write_paths_svg

# ---------------------------------------------------------------------------
# Expected runtime envelope -- re-verified at generation time, not assumed (same
# convention as every sibling generic sheet: an environment without reach_profiles
# falls back to a different default reach profile and would plot against the wrong
# envelope). Tied to the author's measured machine geometry.
# ---------------------------------------------------------------------------
EXPECT_REACH_ID = "plotter_2003_v32"
EXPECT_BBOX = (3.0, 42.5, 255.0, 254.5)
EXPECT_PEN_ID = "uright_gel"
EXPECT_PEN_LABEL = "uRight Gel refill"
EXPECT_PEN_MM = 0.255
EXPECT_PEN_OFFSET = (0.0, 0.0)
EXPECT_Z_DOWN_STOCK = -0.60
EXPECT_DRAW_SPEED_STOCK = 3000
# z_touch_abs is intentionally NOT a pinned literal -- see module docstring.
Z_TOUCH_SANITY_RANGE_MM = (5.0, 20.0)

# ---------------------------------------------------------------------------
# Paper / keep-out (mm). Same convention as every sibling generic sheet:
# JIS B5 257x182mm LANDSCAPE, not config.PAPER_SIZES.
# ---------------------------------------------------------------------------
PAPER_W_MM, PAPER_H_MM = 257.0, 182.0
HOLE_KEEPOUT_MM = 10.0
SHEET_MARGIN_MM = 5.0
_PAD_MM = 3.0
_USABLE_W = PAPER_W_MM - 2 * SHEET_MARGIN_MM
_USABLE_H = PAPER_H_MM - HOLE_KEEPOUT_MM - 2 * SHEET_MARGIN_MM

# ---------------------------------------------------------------------------
# Sheet design. Changing it invalidates comparison with earlier scans.
# ---------------------------------------------------------------------------
N_DIRECTIONS = 16          # 8 angles x 2 senses, 22.5deg apart, full circle
N_REPEATS = 3
PASS_LEN_MM = 15.0
MEASURE_CENTRAL_FRACTION = 0.6     # consumed by the measurement script
CLEARANCE_MM = 3.0

GRID_COLS, GRID_ROWS = 8, 6         # 48 cells, exact fit for 16 dirs x 3 reps
CELL_PITCH_MM = 19.0
GRID_W = GRID_COLS * CELL_PITCH_MM   # 152.0
GRID_H = GRID_ROWS * CELL_PITCH_MM   # 114.0

BAND_GAP_MM = 6.0            # >> CLEARANCE_MM, headroom for the shapely check
FID_TICK_PITCH_MM = 10.0
FID0_LEN_MM = 100.0
FID90_LEN_MM = 100.0
FID45_LEN_MM = 80.0          # conservative "longest that fits the corner band"
                              # (not exhaustively maximized -- the footprint
                              # assert below is the real safety net).
TICK_H_MM = 4.0

TWIN_SIZE_MM = 8.0
TWIN_PITCH_MM = 0.24
TWIN_GAP_MM = 4.0
TWIN_ANGLES = (0.0, 45.0, 90.0, 135.0)

GLYPH_H_MM = 8.0
GLYPH_W_MM = 6.0

SEED = 20260914   # fixed seed so the pass order is reproducible

# Plot-time model (a) = draw time at draw_speed + a fixed per-pen-down-cycle lift
# budget (z_up up/down at plunge_speed/lift_speed) -- NO travel-move time, because
# registry sheets have no microlift; this is a floor, not a forecast.
#
# The diagonal-anisotropy figure below (~40 %) is a LOWER BOUND, not a point estimate.
# An area/peak-width estimator has a positive bias that GROWS as true width shrinks
# (the thinner 45/135 reading is biased MORE than the wider 0/90 reading), so any RAW
# gap between the two families understates the true gap -- read every number here as
# "at least this much anisotropy", never as the whole effect. A measurement should
# only declare "anisotropic" when the 45/135-vs-0/90 gap exceeds 2x the bias band
# measured on the actual capture (from that photo's own twin-patch edge); below that,
# a real effect as large as ~15 % could sit inside the noise floor.
PREDICTION = {
    "isotropic": "all 16 direction medians within +/-10% of each other",
    "diagonal_anisotropy": "45/135-family narrower than 0/90-family by AT "
                            "LEAST ~40% (uright_gel wedge-read-2026-09-11 "
                            "hint; a LOWER BOUND, see module docstring/D378)",
    "axis_asymmetry": "0deg median != 90deg median",
    "tilt": "opposite travel senses of the same line (e.g. 0deg vs 180deg) differ",
}
_PREDICTION_LINES = [
    "LINESTAR PREDICTION (pre-registered 2026-09-14, uright_gel, D375/D378):",
    f"isotropic -> {PREDICTION['isotropic']}",
    f"diagonal anisotropy (LOWER BOUND) -> {PREDICTION['diagonal_anisotropy']}",
    f"axis asymmetry -> {PREDICTION['axis_asymmetry']}",
    f"tilt -> {PREDICTION['tilt']}",
    "D378: report raw AND deconvolved widths; declare 'anisotropic' only",
    "if the 45/135 vs 0/90 gap exceeds 2x that capture's own measured bias",
    "band -- below that, a real effect up to ~15% may be hidden. See",
    "measure_linestar.py classify_anisotropy().",
    "Per-element direction_id/repeat_idx/machine angle is in the response",
    "'layout' field, not printed on the sheet -- see",
    "artifacts/linestar-sheet-2026-09-14/measure_linestar.py",
]


# ---------------------------------------------------------------------------
# Geometry builders (design space, mm, y grows down like image space -- same
# convention every sibling module uses before the vpype translate/mirror).
# ---------------------------------------------------------------------------
def _direction_angle_deg(direction_id: int) -> float:
    return (direction_id % N_DIRECTIONS) * (360.0 / N_DIRECTIONS)


def _build_latin_square():
    """48 slots (16 directions x 3 repeats) onto an 8x6 grid, 3 row-pair
    BLOCKS of 16 cells each. Within block k, direction d gets
    col=(d%8+3k)%8, row_in_block=((d//8)+k)%2 -- a Latin square in both col
    (offset 3, coprime with 8) and row (parity cycling with k), so each
    direction lands in 3 distinct columns and 3 distinct absolute rows
    across its 3 repeats, and every one of the 48 grid cells is used exactly
    once. Verified by assertion below."""
    slots = []
    for k in range(3):
        for d in range(N_DIRECTIONS):
            col = (d % 8 + 3 * k) % 8
            row_in_block = ((d // 8) + k) % 2
            row = 2 * k + row_in_block
            slots.append({"direction_id": d, "repeat_idx": k,
                          "block_idx": k, "grid_col": col, "grid_row": row})
    assert len(slots) == 48
    cells = {(s["grid_col"], s["grid_row"]) for s in slots}
    assert len(cells) == 48, "Latin square collision: not every grid cell used exactly once"
    for d in range(N_DIRECTIONS):
        rows = {s["grid_row"] for s in slots if s["direction_id"] == d}
        cols = {s["grid_col"] for s in slots if s["direction_id"] == d}
        assert len(rows) == 3, f"direction {d}: repeats share a row ({rows})"
        assert len(cols) == 3, f"direction {d}: repeats share a column ({cols})"
    return slots


def _pass_pts(cx, cy, angle_deg, length=PASS_LEN_MM):
    a = math.radians(angle_deg)
    dx, dy = math.cos(a) * length, math.sin(a) * length
    return [(cx - dx / 2.0, cy - dy / 2.0), (cx + dx / 2.0, cy + dy / 2.0)]


def _build_grid_elements():
    """Returns (elements, pass_records) for the 48 direction passes, in the
    Latin-square GRID positions. Emission ORDER is assigned separately (see
    `_order_passes`) -- grid position and emission order are deliberately
    independent axes, which is the whole point of a Latin square design."""
    slots = _build_latin_square()
    elements, records = [], []
    for s in slots:
        cx = s["grid_col"] * CELL_PITCH_MM + CELL_PITCH_MM / 2.0
        cy = s["grid_row"] * CELL_PITCH_MM + CELL_PITCH_MM / 2.0
        angle = _direction_angle_deg(s["direction_id"])
        pts = _pass_pts(cx, cy, angle)
        elements.append({"pts": pts, "kind": "pass", "direction_id": s["direction_id"],
                          "repeat_idx": s["repeat_idx"], "block_idx": s["block_idx"],
                          "grid_col": s["grid_col"], "grid_row": s["grid_row"],
                          "design_angle_deg": angle})
        records.append(dict(elements[-1]))
    return elements, records


def _order_passes(elements, seed=SEED):
    """Randomizes PASS EMISSION ORDER with the fixed seed --
    non-pass elements (fiducials/glyph/twin patches) keep a fixed
    deterministic order, since only the pass-vs-pass comparison is the thing
    plate position/plot order must not confound. Returns a NEW list in the
    randomized order, each element additionally carries `order_idx`."""
    idxs = list(range(len(elements)))
    random.Random(seed).shuffle(idxs)
    out = []
    for order_idx, i in enumerate(idxs):
        e = dict(elements[i])
        e["order_idx"] = order_idx
        out.append(e)
    return out


def _ticks_along(x0, y0, length, orientation, pitch=FID_TICK_PITCH_MM, tick_h=TICK_H_MM):
    """Perpendicular tick marks at an exact mm PITCH along a baseline (not
    n-evenly-spaced-over-a-width like calibration_plots.ticks -- this sheet
    needs known ABSOLUTE spacing for a scan's per-axis scale self-
    calibration). `orientation` in {"h","v","diag"}."""
    out = []
    n = int(round(length / pitch))
    for k in range(n + 1):
        d = k * pitch
        if d > length + 1e-9:
            break
        if orientation == "h":
            out.append([(x0 + d, y0), (x0 + d, y0 + tick_h)])
        elif orientation == "v":
            out.append([(x0, y0 + d), (x0 + tick_h, y0 + d)])
        else:  # diag, 45deg baseline; ticks perpendicular (slope -1), offset to +x/+y side
            bx, by = x0 + d / math.sqrt(2.0), y0 + d / math.sqrt(2.0)
            t = tick_h / math.sqrt(2.0)
            out.append([(bx, by), (bx + t, by - t)])
    return out


def _fiducial_0(x0, y0, length=FID0_LEN_MM):
    baseline = [(x0, y0), (x0 + length, y0)]
    ticks = _ticks_along(x0, y0, length, "h")
    return [baseline] + ticks, {"kind": "fiducial", "axis_deg": 0.0,
                                "length_mm": length, "origin_mm": [x0, y0],
                                "tick_pitch_mm": FID_TICK_PITCH_MM}


def _fiducial_90(x0, y0, length=FID90_LEN_MM):
    baseline = [(x0, y0), (x0, y0 + length)]
    ticks = _ticks_along(x0, y0, length, "v")
    return [baseline] + ticks, {"kind": "fiducial", "axis_deg": 90.0,
                                "length_mm": length, "origin_mm": [x0, y0],
                                "tick_pitch_mm": FID_TICK_PITCH_MM}


def _fiducial_45(x0, y0, length=FID45_LEN_MM):
    end = (x0 + length / math.sqrt(2.0), y0 + length / math.sqrt(2.0))
    baseline = [(x0, y0), end]
    ticks = _ticks_along(x0, y0, length, "diag")
    return [baseline] + ticks, {"kind": "fiducial", "axis_deg": 45.0,
                                "length_mm": length, "origin_mm": [x0, y0],
                                "tick_pitch_mm": FID_TICK_PITCH_MM}


def _f_glyph(x0, y0):
    """Asymmetric orientation marker: 3 strokes forming an 'F' (vertical
    spine + a longer top bar + a shorter mid bar) -- deliberately not
    symmetric under any flip/rotation, so a mirrored/rotated scan is
    detectable on sight before any measurement is trusted."""
    spine = [(x0, y0), (x0, y0 + GLYPH_H_MM)]
    top = [(x0, y0), (x0 + GLYPH_W_MM, y0)]
    mid = [(x0, y0 + GLYPH_H_MM * 0.5), (x0 + GLYPH_W_MM * 0.55, y0 + GLYPH_H_MM * 0.5)]
    return [spine, top, mid]


def _hatch_axis(size, pitch, vertical=False):
    out, t, k = [], 0.0, 0
    while t <= size + 1e-9:
        if not vertical:
            pts = [(0.0, t), (size, t)] if k % 2 == 0 else [(size, t), (0.0, t)]
        else:
            pts = [(t, 0.0), (t, size)] if k % 2 == 0 else [(t, size), (t, 0.0)]
        out.append(pts)
        t += pitch
        k += 1
    return out


def _hatch_diag(size, pitch, mirror=False):
    out = []
    dc = pitch * math.sqrt(2.0)
    c = -size + dc / 2.0
    k = 0
    while c < size:
        x_lo, x_hi = max(0.0, -c), min(size, size - c)
        if x_hi - x_lo > 1e-9:
            length = (x_hi - x_lo) * math.sqrt(2.0)
            if length >= 1.0:
                if not mirror:
                    p0, p1 = (x_lo, x_lo + c), (x_hi, x_hi + c)
                else:
                    p0, p1 = (size - x_lo, x_lo + c), (size - x_hi, x_hi + c)
                pts = [p0, p1] if k % 2 == 0 else [p1, p0]
                out.append(pts)
                k += 1
        c += dc
    return out


def _twin_patch(x0, y0, angle_deg, size=TWIN_SIZE_MM, pitch=TWIN_PITCH_MM):
    if angle_deg == 0.0:
        raw = _hatch_axis(size, pitch, vertical=False)
    elif angle_deg == 90.0:
        raw = _hatch_axis(size, pitch, vertical=True)
    elif angle_deg == 45.0:
        raw = _hatch_diag(size, pitch, mirror=False)
    else:  # 135
        raw = _hatch_diag(size, pitch, mirror=True)
    return [[(x + x0, y + y0) for x, y in pts] for pts in raw]


def _poly_length_mm(pts):
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1))


def _poly_bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(xs) - min(xs), max(ys) - min(ys)


def _build_elements():
    """Assembles every design-space element (grid passes, fiducials, F
    glyph, coverage twin) and a `groups` dict for the clearance check.
    Returns (elements, pass_records, fid_records, groups, footprint)."""
    grid_elems, pass_records = _build_grid_elements()
    grid_elems = _order_passes(grid_elems)

    # F glyph: fixed corner above the grid (negative local y), >=BAND_GAP
    # clear of the grid's own top edge.
    glyph_x0, glyph_y0 = 0.0, -(GLYPH_H_MM + BAND_GAP_MM)
    glyph_strokes = _f_glyph(glyph_x0, glyph_y0)

    # 0deg fiducial: below the grid.
    fid0_x0, fid0_y0 = 0.0, GRID_H + BAND_GAP_MM
    fid0_strokes, fid0_rec = _fiducial_0(fid0_x0, fid0_y0)

    # Coverage twin: same Y band as the 0deg fiducial, to its right.
    twin_x0 = fid0_x0 + FID0_LEN_MM + BAND_GAP_MM
    twin_y0 = fid0_y0
    twin_groups = {}
    twin_records = []
    tx = twin_x0
    for ang in TWIN_ANGLES:
        strokes = _twin_patch(tx, twin_y0, ang)
        twin_groups[f"twin_{ang:g}"] = [s for s in strokes]
        twin_records.append({"angle_deg": ang, "origin_mm": [tx, twin_y0],
                             "size_mm": TWIN_SIZE_MM, "pitch_mm": TWIN_PITCH_MM,
                             "n_lines": len(strokes)})
        tx += TWIN_SIZE_MM + TWIN_GAP_MM

    # 90deg fiducial: right of the grid.
    fid90_x0, fid90_y0 = GRID_W + BAND_GAP_MM, 0.0
    fid90_strokes, fid90_rec = _fiducial_90(fid90_x0, fid90_y0)

    # 45deg fiducial: spare corner beyond the 90deg fiducial's tick strip.
    fid45_x0 = fid90_x0 + TICK_H_MM + BAND_GAP_MM
    fid45_y0 = 0.0
    fid45_strokes, fid45_rec = _fiducial_45(fid45_x0, fid45_y0)

    elements = []
    elements += grid_elems
    for pts in glyph_strokes:
        elements.append({"pts": pts, "kind": "glyph"})
    for pts in fid0_strokes:
        elements.append({"pts": pts, "kind": "fiducial", "fid_axis_deg": 0.0})
    for ang, strokes in zip(TWIN_ANGLES, [twin_groups[f"twin_{a:g}"] for a in TWIN_ANGLES]):
        for pts in strokes:
            elements.append({"pts": pts, "kind": "twin_fill", "twin_angle_deg": ang})
    for pts in fid90_strokes:
        elements.append({"pts": pts, "kind": "fiducial", "fid_axis_deg": 90.0})
    for pts in fid45_strokes:
        elements.append({"pts": pts, "kind": "fiducial", "fid_axis_deg": 45.0})

    xs = [x for e in elements for x, _y in e["pts"]]
    ys = [y for e in elements for _x, y in e["pts"]]
    minx, miny = min(xs), min(ys)
    elements = [{**e, "pts": [(x - minx, y - miny) for x, y in e["pts"]]} for e in elements]
    footprint_w = max(xs) - minx
    footprint_h = max(ys) - miny

    if footprint_w + 2 * _PAD_MM > _USABLE_W:
        raise RuntimeError(f"linestar: footprint width {footprint_w + 2 * _PAD_MM:.1f}mm "
                           f"exceeds the JIS B5 landscape usable width {_USABLE_W:.1f}mm")
    if footprint_h + 2 * _PAD_MM > _USABLE_H:
        raise RuntimeError(f"linestar: footprint height {footprint_h + 2 * _PAD_MM:.1f}mm "
                           f"exceeds the JIS B5 landscape usable height {_USABLE_H:.1f}mm")

    # Clearance groups (design space, shifted like `elements` above).
    def _shift(pts_list):
        return [[(x - minx, y - miny) for x, y in pts] for pts in pts_list]

    groups = {}
    for i, s in enumerate(pass_records):
        groups[f"pass_{s['direction_id']}_{s['repeat_idx']}"] = _shift(
            [_pass_pts(s['grid_col'] * CELL_PITCH_MM + CELL_PITCH_MM / 2.0,
                       s['grid_row'] * CELL_PITCH_MM + CELL_PITCH_MM / 2.0,
                       s['design_angle_deg'])])
    groups["glyph"] = _shift(glyph_strokes)
    groups["fid_0"] = _shift(fid0_strokes)
    groups["fid_90"] = _shift(fid90_strokes)
    groups["fid_45"] = _shift(fid45_strokes)
    for ang in TWIN_ANGLES:
        groups[f"twin_{ang:g}"] = _shift(twin_groups[f"twin_{ang:g}"])

    fid_records = [fid0_rec, fid90_rec, fid45_rec]
    return elements, fid_records, twin_records, groups, footprint_w, footprint_h


def _group_geom(pts_list):
    lines = [LineString(p) for p in pts_list if len(p) >= 2]
    return lines[0] if len(lines) == 1 else MultiLineString(lines)


def _check_clearance(groups, min_mm=CLEARANCE_MM):
    """Pairwise minimum distance between every isolated GROUP (see module
    docstring on scope: within-patch coverage-twin hatch is excluded by
    construction -- it is never its own group here), computed with shapely
    in design (pre-gcode, isometric) space."""
    names = list(groups.keys())
    geoms = [_group_geom(groups[n]) for n in names]
    violations = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = geoms[i].distance(geoms[j])
            if d < min_mm - 1e-9:
                violations.append([names[i], names[j], round(d, 3)])
    if violations:
        raise RuntimeError(
            f"linestar: {len(violations)} element-group pair(s) violate the {min_mm}mm "
            f"clearance rule: {violations[:8]}. Do not plot this file.")
    return {"n_groups": len(names), "min_clearance_mm": min_mm,
            "n_pairs_checked": len(names) * (len(names) - 1) // 2}


# ---------------------------------------------------------------------------
# Post-emission validation: isometry-invariant length+bbox match (catches
# cross-element drags of any kind), full-360deg angle read-back with
# empirical mirror-detection for pass elements, exactly-3-per-direction, and
# a non-zero examined-segment guard so the checks cannot pass vacuously --
# same idiom as every sibling generic sheet.
# ---------------------------------------------------------------------------
def _parse_strokes(path, z_down_abs, z_tol=1e-2):
    strokes = []
    cur = None
    x = y = None
    down = False
    with Path(path).open() as fh:
        for line in fh:
            if not line.startswith("G1"):
                continue
            mz = re.search(r"Z(-?[\d.]+)", line)
            if mz:
                was_down = down
                down = abs(round(float(mz.group(1)), 3) - round(z_down_abs, 3)) < z_tol
                if down and not was_down:
                    cur = {"pts": []}
                    if x is not None:
                        cur["pts"].append((x, y))
                    strokes.append(cur)
                elif was_down and not down:
                    cur = None
                continue
            mx = re.search(r"X(-?[\d.]+)", line)
            my = re.search(r"Y(-?[\d.]+)", line)
            nx = float(mx.group(1)) if mx else x
            ny = float(my.group(1)) if my else y
            if down and cur is not None:
                cur["pts"].append((nx, ny))
            x, y = nx, ny
    return [s for s in strokes if len(s["pts"]) >= 2]


def _detect_mirror(elements, strokes, reference_direction_id=4):
    """Empirically determines whether the pipeline mirrors Y, from the
    emitted stroke of the direction_id=4 pass (design angle 90deg, a pure
    +y design vector with zero x-component -- unambiguous either way).
    Read from the emitted g-code, not inferred from the translate/gwrite call."""
    idx = next(i for i, e in enumerate(elements)
               if e["kind"] == "pass" and e["direction_id"] == reference_direction_id)
    (x0, y0), (x1, y1) = elements[idx]["pts"][0], elements[idx]["pts"][-1]
    dy_design = y1 - y0
    (mx0, my0), (mx1, my1) = strokes[idx]["pts"][0], strokes[idx]["pts"][-1]
    dy_machine = my1 - my0
    mirror = abs(dy_machine - (-dy_design)) < abs(dy_machine - dy_design)
    return mirror, idx


def _pendown_check(out_gcode, z_down_abs, elements, n_pendowns_expected):
    strokes = _parse_strokes(out_gcode, z_down_abs)
    n_pendown_segments = sum(len(s["pts"]) - 1 for s in strokes)
    if n_pendown_segments <= 0:
        raise RuntimeError(
            "linestar: pendown_segments_examined is ZERO -- the guard never ran "
            "(a vacuous pass). Do not plot this file.")
    if len(strokes) != n_pendowns_expected:
        raise RuntimeError(
            f"linestar: stroke count mismatch: emitted {len(strokes)} pen-down cycles, "
            f"expected {n_pendowns_expected} (one per design element). Do not plot this file.")

    length_mismatches, bbox_mismatches = [], []
    for i, (elem, stroke) in enumerate(zip(elements, strokes)):
        expected_len = _poly_length_mm(elem["pts"])
        actual_len = _poly_length_mm(stroke["pts"])
        if abs(actual_len - expected_len) > max(0.05, 0.01 * expected_len):
            length_mismatches.append({"elem_idx": i, "kind": elem["kind"],
                                       "expected_mm": round(expected_len, 3),
                                       "actual_mm": round(actual_len, 3)})
        ew, eh = _poly_bbox(elem["pts"])
        aw, ah = _poly_bbox(stroke["pts"])
        tol = max(0.05, 0.01 * max(ew, eh))
        if abs(aw - ew) > tol or abs(ah - eh) > tol:
            bbox_mismatches.append({"elem_idx": i, "kind": elem["kind"]})
    if length_mismatches:
        raise RuntimeError(
            f"linestar: {len(length_mismatches)} element(s) have machine-length != "
            f"design-length beyond tolerance: {length_mismatches[:5]}. Do not plot this file.")
    if bbox_mismatches:
        raise RuntimeError(
            f"linestar: {len(bbox_mismatches)} element(s) have a machine bbox that does not "
            f"match their design bbox -- cross-element drag guard firing: "
            f"{bbox_mismatches[:5]}. Do not plot this file.")

    mirror, ref_idx = _detect_mirror(elements, strokes)

    pass_recs = []
    per_dir_count = {}
    max_dev = 0.0
    for i, (elem, stroke) in enumerate(zip(elements, strokes)):
        if elem["kind"] != "pass":
            continue
        design_a = elem["design_angle_deg"]
        expected_machine_a = (-design_a) % 360.0 if mirror else design_a % 360.0
        (sx0, sy0), (sx1, sy1) = stroke["pts"][0], stroke["pts"][-1]
        actual_a = math.degrees(math.atan2(sy1 - sy0, sx1 - sx0)) % 360.0
        dev = min(abs(actual_a - expected_machine_a),
                  360.0 - abs(actual_a - expected_machine_a))
        max_dev = max(max_dev, dev)
        d = elem["direction_id"]
        per_dir_count[d] = per_dir_count.get(d, 0) + 1
        pass_recs.append({
            "direction_id": d, "repeat_idx": elem["repeat_idx"], "block_idx": elem["block_idx"],
            "grid_col": elem["grid_col"], "grid_row": elem["grid_row"],
            "order_idx": elem["order_idx"], "design_angle_deg": design_a,
            "machine_angle_deg": round(actual_a, 3),
            "expected_machine_angle_deg": round(expected_machine_a, 3),
            "angle_deviation_deg": round(dev, 4),
            "start_mm": [round(sx0, 3), round(sy0, 3)], "end_mm": [round(sx1, 3), round(sy1, 3)],
        })
    if max_dev >= 0.5:
        raise RuntimeError(
            f"linestar: a pass element's machine-frame angle deviates {max_dev:.2f}deg from "
            f"its design angle (after Y-mirror={mirror}) -- unexpected geometry. "
            f"Do not plot this file.")
    bad_counts = {d: n for d, n in per_dir_count.items() if n != N_REPEATS}
    if bad_counts:
        raise RuntimeError(
            f"linestar: direction(s) with != {N_REPEATS} emitted pass strokes: "
            f"{bad_counts}. Do not plot this file.")

    return {
        "pendown_segments_examined": n_pendown_segments,
        "n_strokes": len(strokes),
        "n_pendowns_expected": n_pendowns_expected,
        "mirror_y_detected": mirror,
        "mirror_reference_direction_id": 4,
        "max_pass_angle_deviation_deg": round(max_dev, 4),
        "n_directions_with_correct_repeat_count": N_DIRECTIONS - len(bad_counts),
        "pass_records": pass_recs,
    }, strokes


# ---------------------------------------------------------------------------
# Emission -- single Z / single draw speed (angle is the only variable), one
# gwrite call, same pattern as every sibling generic sheet.
# ---------------------------------------------------------------------------
def _pen_id_from_profile(profile: dict, profiles_doc: dict):
    matches = [pid for pid, p in profiles_doc.get("profiles", {}).items() if p == profile]
    return matches[0] if len(matches) == 1 else None


def _prepend_ascii_header(path, lines):
    text = Path(path).read_text(encoding="ascii")
    header = "".join(f"; {ln}\n" for ln in lines)
    Path(path).write_text(header + text, encoding="ascii")


def linestar_gcode(out_gcode: Path, pen_mm=None, z_up=3.5,
                   travel_speed=8000, plunge_speed=500, lift_speed=2000,
                   offset_x=None, offset_y=None, end_policy="release",
                   quiet_fans=True, profile=None):
    """Emit the 16-direction line-star sheet as ONE g-code file, ONE gwrite
    call. `profile`, if given, must name EXPECT_PEN_ID -- see module
    docstring "PEN IS PINNED, NOT RESOLVED"."""
    ppmm = 1.0 / PX2MM
    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()

    profiles = load_pen_profiles()
    if EXPECT_PEN_ID not in profiles.get("profiles", {}):
        raise RuntimeError(f"pen profile {EXPECT_PEN_ID!r} not found in pen_profiles.json")
    pen = profiles["profiles"][EXPECT_PEN_ID]

    if profile is not None and profile != pen:
        req_id = _pen_id_from_profile(profile, profiles)
        req_id_str = repr(req_id) if req_id else "unknown (blob matches no current pen)"
        raise RuntimeError(
            f"linestar: refusing to generate -- this sheet is PINNED to {EXPECT_PEN_ID!r} "
            f"({EXPECT_PEN_LABEL!r}); the active/requested pen is {req_id_str} "
            f"({profile.get('label')!r}). This sheet's Latin-square/direction design is "
            f"generic but its PREDICTION numbers are calibrated from uright_gel's own "
            f"wedge evidence and are not a valid measurement of a different pen. "
            f"Mount uright_gel and re-request, or POST with pen_id='uright_gel' explicitly.")

    reach0 = active_reach_profile(cal)
    if reach0["id"] != EXPECT_REACH_ID or tuple(round(v, 1) for v in reach0["bbox"]) != EXPECT_BBOX:
        raise RuntimeError(
            f"Runtime reach envelope mismatch: got id={reach0['id']!r} bbox={reach0['bbox']}, "
            f"expected id={EXPECT_REACH_ID!r} bbox={EXPECT_BBOX}. Refusing to generate against "
            f"an unexpected envelope.")

    label = pen.get("label")
    if label != EXPECT_PEN_LABEL:
        raise RuntimeError(f"pen {EXPECT_PEN_ID!r} label mismatch: got {label!r}, "
                           f"expected {EXPECT_PEN_LABEL!r}.")
    if abs(float(pen.get("pen_mm", 0.0)) - EXPECT_PEN_MM) > 1e-6:
        raise RuntimeError(f"pen {EXPECT_PEN_ID!r} pen_mm mismatch: got {pen.get('pen_mm')}, "
                           f"expected {EXPECT_PEN_MM}")
    got_offset = (float(pen.get("pen_offset_x", 0.0)), float(pen.get("pen_offset_y", 0.0)))
    if any(abs(a - b) > 1e-6 for a, b in zip(got_offset, EXPECT_PEN_OFFSET)):
        raise RuntimeError(f"pen {EXPECT_PEN_ID!r} pen_offset mismatch: got {got_offset}, "
                           f"expected {EXPECT_PEN_OFFSET}")

    z_touch, z_touch_source = resolve_z_touch(cal, pen)
    lo, hi = Z_TOUCH_SANITY_RANGE_MM
    if not (lo <= z_touch <= hi):
        raise RuntimeError(f"z_touch_abs for {EXPECT_PEN_ID} resolved to {z_touch}, outside "
                           f"the sanity range [{lo},{hi}]mm -- refusing.")
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

    (elements, fid_records, twin_records, groups,
     footprint_w, footprint_h) = _build_elements()
    clearance_check = _check_clearance(groups)
    n_pendowns_expected = len(elements)

    PLACEMENT_MARGIN_MM = 20.0
    pad = _PAD_MM
    canvas_w_mm, canvas_h_mm = footprint_w + 2 * pad, footprint_h + 2 * pad

    offset_x = reach["bbox"][0] + PLACEMENT_MARGIN_MM if offset_x is None else float(offset_x)
    if offset_y is None:
        flip_axis_mm = max(canvas_w_mm, canvas_h_mm)
        near_edge_target = reach["bbox"][1] + PLACEMENT_MARGIN_MM
        comp_off_y = near_edge_target - flip_axis_mm + pad + footprint_h
        offset_y = comp_off_y + pen_off_y
    else:
        offset_y = float(offset_y)
    comp_off_x, comp_off_y = offset_x - pen_off_x, offset_y - pen_off_y
    park_x, park_y = reach["safe_park_point"]
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))

    W = int(round(canvas_w_mm * ppmm))
    H = int(round(canvas_h_mm * ppmm))

    placed = [[((x + pad) * ppmm, (y + pad) * ppmm) for x, y in e["pts"]] for e in elements]
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
        _prepend_ascii_header(tmp_gcode, _PREDICTION_LINES)
    svg.unlink(missing_ok=True)

    validate_gcode_xy_motion(out_gcode, reach)
    safety = validate_calibration_gcode_safety(out_gcode, z_down_abs)
    pendown_check, strokes = _pendown_check(out_gcode, z_down_abs, elements, n_pendowns_expected)

    # Plot-time estimate, model (a): draw time at draw_speed + a fixed
    # per-pen-down-cycle lift budget, NO travel-move time (registry sheets
    # have no microlift, so this is a floor, not a forecast).
    total_draw_mm = sum(_poly_length_mm(s["pts"]) for s in strokes)
    plot_time_s = total_draw_mm / max(1.0, draw_speed / 60.0)
    lift_budget_s = pendown_check["n_strokes"] * (
        (abs(z_up) / max(1.0, plunge_speed / 60.0)) + (abs(z_up) / max(1.0, lift_speed / 60.0)))
    plot_time_s += lift_budget_s

    prev_svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_paths_svg(prev_svg, W, H, float(pen_mm) * ppmm, polylines=placed)
    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(prev_svg))} color {PREVIEW_COLOR} '
                          f'write {shlex.quote(str(tmp_prev))}')

    layout = {
        "n_directions": N_DIRECTIONS, "n_repeats": N_REPEATS, "pass_len_mm": PASS_LEN_MM,
        "measure_central_fraction": MEASURE_CENTRAL_FRACTION, "seed": SEED,
        "clearance_mm": CLEARANCE_MM, "grid": {"cols": GRID_COLS, "rows": GRID_ROWS,
                                                "cell_pitch_mm": CELL_PITCH_MM},
        "passes": pendown_check["pass_records"],
        "fiducials": fid_records,
        "twin_patches": twin_records,
        "prediction": PREDICTION,
        "mirror_y_detected": pendown_check["mirror_y_detected"],
    }

    return {"kind": "linestar", "pen_mm": pen_mm, "pen_id": EXPECT_PEN_ID,
            "pen_label": EXPECT_PEN_LABEL,
            "footprint_mm": [round(canvas_w_mm, 1), round(canvas_h_mm, 1)],
            "usable_mm": [round(_USABLE_W, 1), round(_USABLE_H, 1)],
            "offset_mm": [offset_x, offset_y],
            "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
            "z_down": z_down, "z_down_abs": round(z_down_abs, 3), "draw_speed": draw_speed,
            "reach_id": reach0["id"], "reach_bbox": reach0["bbox"],
            "n_pendown_expected_total": n_pendowns_expected,
            "n_pendown_actual_total": pendown_check["n_strokes"],
            "pendown_count_matches": pendown_check["n_strokes"] == n_pendowns_expected,
            "pendown_segments_examined": pendown_check["pendown_segments_examined"],
            "deepest_z_down_abs": round(z_down_abs, 3),
            "safety_selfcheck": safety, "pendown_check": {k: v for k, v in pendown_check.items()
                                                           if k != "pass_records"},
            "clearance_check": clearance_check,
            "prediction": PREDICTION,
            "plot_time_estimate_s": round(plot_time_s, 1),
            "plot_time_estimate_min": round(plot_time_s / 60.0, 2),
            "plot_time_model_label": "model (a); registry sheets have no microlift, so this "
                                     "is a realistic floor, not a forecast (D290/D291/D336). "
                                     "No travel-move time counted.",
            "preview": prev.name, "svg_path": str(prev_svg), "svg_wh_px": [W, H],
            "pattern_mm": [round(canvas_w_mm, 1), round(canvas_h_mm, 1)],
            "place": {"x": offset_x, "y": offset_y, "w": canvas_w_mm, "h": canvas_h_mm},
            "z_up_abs": round(z_touch + float(z_up), 3),
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
            "layout": layout}
