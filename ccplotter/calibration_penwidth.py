"""ccplotter/calibration_penwidth.py -- LAID WIDTH DISTRIBUTION calibration sheet.

What it measures: the distribution (not just the median) of the line width the pen
actually lays, as a function of contact pressure (Z), draw speed and stroke angle.
`laid_mm` (0.255 for uright_gel) is a median over ~29,000 width measurements; the median
alone cannot set a fill-pitch margin, because pitch = median width leaves about half the
passes UNDER median, and a true-width render at 100 px/mm shows white striping where a
sub-median stretch fails to reach the neighbouring pass. Percentiles let the margin be
chosen from data instead of a guessed safety factor.

How to read it: the sheet is isolated single strokes, never overlapping fills, so each
stroke's width can be read independently. Scan it at 600 dpi; each stroke gives one width
sample per scanned pixel COLUMN along its length (sum ink vertically, keep positive
columns). A 25 mm stroke at ~23.6 px/mm contributes on the order of 500 raw samples before
edge trimming, and the full factorial yields on the order of 10^5. Take percentiles per
(z_down, draw_speed, angle) cell.

FACTORS (full factorial, one sheet):
  z_down      Z_DOWN_LEVELS   contact pressure, bracketing the active profile's own
                              z_down (one of the levels IS the profile's stock value, as
                              in calibration_deposition.Z_DOWN_LEVELS).
  draw_speed  DRAW_SPEEDS     bracketing the profile's own draw_speed.
  angle       0 / 45 deg      axis-aligned vs diagonal -- a ballpoint/gel tip's effective
                              deposit is direction-dependent, and a real plotted image
                              draws at every angle, not just axis-aligned.

  z_down x draw_speed cells are each their own complete gwrite output (default_values
  cannot vary within one gwrite call), spliced into ONE file by
  calibration_deposition's `_splice_blocks` (structural preamble/postamble anchoring, not
  longest-common-prefix diffing; see that function's docstring). Both angle groups for a
  given (z_down, draw_speed) sit in the SAME cell and the same gwrite call, with different
  stroke geometry.

  NO LINESORT: routing would bridge the very gaps this sheet measures.

SCAN-SIDE FIDUCIALS. Each cell carries one small SOLID square fiducial (dense hatch, not
an outline) at a fixed, known pitch (CELL_W+CELL_GAP_X, CELL_H+CELL_GAP_Y). A scan
analysis can detect these blobs, self-calibrate scale from the grid's known pitch, and
fit a least-squares affine transform tolerant of missing blobs.

PAPER / KEEP-OUT. The paper is JIS B5 257x182 mm LANDSCAPE, NOT `PAPER_SIZES["b5"]` in
config.py, which is ISO B5 176x250. The sheet's footprint is asserted to fit inside that
paper minus a HOLE_KEEPOUT_MM strip (matches composite.py's `kokuyo_b5_26h` 10 mm
default) plus a plain margin on every side. This generator does not use composite.py's
keepout machinery (it is a standalone true-mm g-code file); it budgets for the same
physical strip so the footprint constant is honest about what paper it fits.

Safety: Z comes from the active pen profile (raises if uncalibrated), the reach clamp is
honoured, XY motion is validated against the reach profile, and a post-condition rejects
any pen-down move long enough to be a cell-to-cell drag. This module has no EXPECT_REACH_ID
pin; it works against whatever reach profile is active.

Reached through POST /api/calibration_penwidth (ccplotter/routes/pen.py).
"""
import math
import shlex
from pathlib import Path

import vpype as vp
import vpype_cli

from ccplotter.atomic_io import atomic_target
from ccplotter.config import CONFIG_PATH, PREVIEW_COLOR, PX2MM, WORK_DIR
from ccplotter.cal import resolve_pen_offset, load_cal, active_pen_profile, resolve_z_touch
from ccplotter.gcode_post import (_gcode_lifecycle_values, validate_calibration_gcode_safety,
                                  _fix_gwrite_source_comment)
from ccplotter.reach import active_reach_profile, validate_gcode_xy_motion, pen_reach_profile
from ccplotter.svg_io import _write_paths_svg
from ccplotter.calibration_deposition import _splice_blocks
from ccplotter.calibration_plots import ticks, rect_outline, hatch_h

# ---------------------------------------------------------------------------
# Paper / keep-out (mm). The paper is JIS B5 182x257 -- NOT the "b5" entry in
# config.PAPER_SIZES (that is ISO B5, 176x250). Landscape: 257 wide, 182 tall.
# ---------------------------------------------------------------------------
PAPER_W_MM, PAPER_H_MM = 257.0, 182.0
# Kokuyo B5 26-hole punch strip: same 10mm default as composite.py's
# `_paper_keepout_safe_region(..., keepout={"profile": "kokuyo_b5_26h"})` --
# one shared number, so the budget cannot drift from the real plot pipeline.
HOLE_KEEPOUT_MM = 10.0
SHEET_MARGIN_MM = 5.0   # plain margin on every remaining edge

# ---------------------------------------------------------------------------
# Sweep levels. z_down/draw_speed each bracket the ACTIVE PROFILE's own
# stock value (one level IS that value), same convention as
# calibration_deposition.Z_DOWN_LEVELS / DRAW_SPEEDS -- the sheet contains
# its own baseline, not only excursions from it.
# ---------------------------------------------------------------------------
Z_DOWN_LEVELS = (-0.3, -0.6, -0.9, -1.2)
DRAW_SPEEDS = (1500, 2500, 3500, 4500)
ANGLES_DEG = (0, 45)

GRID_COLS = len(Z_DOWN_LEVELS)     # 4 -- one column per pressure
GRID_ROWS = len(DRAW_SPEEDS)       # 4 -- one row per speed

# ---------------------------------------------------------------------------
# Per-cell isolated-stroke geometry. Both angle groups are sized so their
# geometric bounding-box WIDTH matches (WIDTH_TARGET) -- this is what lets
# them sit side by side in a compact, predictable cell footprint instead of
# stacking (stacking a 45deg group under a 0deg group blows the cell height
# budget: a 45deg stroke's own length adds directly to vertical extent on
# top of the replicate stacking, unlike a 0deg stroke).
#
# Perpendicular gap between adjacent replicates of the SAME angle group:
# offsetting each replicate purely in Y by PITCH_MM gives a true perpendicular
# separation of PITCH_MM*cos(angle) -- for 45deg that is ~0.71*PITCH_MM, still
# ~7x the pen's nominal width at the sheet's default PITCH_MM, and lines
# offset by a pure Y-translation from a shared start X are strictly parallel
# (never intersect) regardless of angle, so this needs no per-angle special
# case.
# ---------------------------------------------------------------------------
STROKE_LEN_MM = 25.0       # physical stroke length, SAME for both angle
                           # groups (a stroke is a stroke regardless of which
                           # way it points) -- at ~23.6 px/mm (600dpi) this
                           # contributes ~500 raw width samples/stroke before
                           # edge trimming (per-pixel-column method)
N_STROKES_0 = 13           # replicates, axis-aligned group
N_STROKES_45 = 5           # replicates, diagonal group (its own LENGTH adds
                           # directly to vertical extent on top of replicate
                           # stacking, unlike the 0deg group, so fewer
                           # replicates keeps the cell footprint in the JIS
                           # B5 landscape height budget; see module docstring)
PITCH_MM = 2.0             # replicate-to-replicate Y offset, both groups --
                           # true perpendicular separation is PITCH_MM at
                           # 0deg and PITCH_MM*cos(45) ~= 1.41mm at 45deg,
                           # both >5x the pen's nominal ~0.25-0.5mm width
GROUP_GAP_X_MM = 4.0       # gap between the 0deg block and the 45deg block

LEN_0 = STROKE_LEN_MM
LEN_45 = STROKE_LEN_MM

TICK_H = 2.0
TICK_GAP_MM = 1.5
FIDUCIAL_MM = 3.0          # solid scan fiducial, side length
FIDUCIAL_GAP_MM = 1.0
FIDUCIAL_HATCH_MM = 0.3    # dense enough to read as solid ink, not a hatch


def stroke_group(x0, y0, angle_deg, n, length, pitch):
    """`n` ISOLATED single-stroke polylines (2-point, own plunge/draw/lift
    each) at `angle_deg` from x0,y0, each `length` mm, offset from the last
    by `pitch` mm along +Y only. Parallel-line geometry: any two of these are
    a pure translation apart (never intersect), so this is safe for any
    angle in (0, 90) without a special case."""
    rad = math.radians(angle_deg)
    dx, dy = length * math.cos(rad), length * math.sin(rad)
    return [[(x0, y0 + k * pitch), (x0 + dx, y0 + dy + k * pitch)] for k in range(n)]


def fiducial(x0, y0, size=FIDUCIAL_MM):
    """Solid scan-registration blob: outline + dense hatch so it reads as one
    solid connected component under threshold, for blob detection (self-
    calibrated px/mm from the grid's known pitch, least-squares affine
    fit). NOT a measurement target itself."""
    return rect_outline(x0, y0, size, size) + hatch_h(x0, y0, size, size, FIDUCIAL_HATCH_MM)


def build_penwidth_cell(col_idx, row_idx):
    """One (z_down, draw_speed) cell in local mm coords: a 0deg isolated-
    stroke group and a 45deg isolated-stroke group side by side, a fiducial
    blob, and col/row identification ticks (LEFT = pressure column index
    1..N, RIGHT = speed row index 1..N -- same idiom
    calibration_deposition.build_deposition_cell uses, so a loose sheet
    fragment is still identifiable without the legend).

    Returns (polys, spec); spec is bookkeeping for the response, not
    gcode."""
    polys = []
    group0 = stroke_group(0.0, 0.0, 0, N_STROKES_0, LEN_0, PITCH_MM)
    h0 = (N_STROKES_0 - 1) * PITCH_MM
    polys += group0

    x45 = LEN_0 + GROUP_GAP_X_MM
    group45 = stroke_group(x45, 0.0, 45, N_STROKES_45, LEN_45, PITCH_MM)
    h45 = (N_STROKES_45 - 1) * PITCH_MM + LEN_45 * math.sin(math.radians(45))
    w45 = LEN_45 * math.cos(math.radians(45))
    polys += group45

    content_h = max(h0, h45)
    content_w = x45 + w45

    y_tick = content_h + TICK_GAP_MM
    polys += [[(0.0, y_tick), (10.0, y_tick)]]
    polys += ticks(0.0, 10.0, y_tick, col_idx + 1, tick_h=TICK_H)
    rx = content_w - 10.0
    polys += [[(rx, y_tick), (rx + 10.0, y_tick)]]
    polys += ticks(rx, 10.0, y_tick, row_idx + 1, tick_h=TICK_H)

    fid_y = y_tick + TICK_H * 1.5 + FIDUCIAL_GAP_MM
    polys += fiducial(0.0, fid_y)

    cell_h = fid_y + FIDUCIAL_MM
    spec = {"len_0_mm": LEN_0, "n_0": N_STROKES_0, "len_45_mm": round(LEN_45, 3),
            "n_45": N_STROKES_45, "pitch_mm": PITCH_MM,
            "content_w_mm": round(content_w, 3), "cell_h_mm": round(cell_h, 3),
            "fiducial_xy_mm": [0.0, round(fid_y, 3)], "fiducial_size_mm": FIDUCIAL_MM,
            "col_ticks": col_idx + 1, "row_ticks": row_idx + 1}
    return polys, content_w, cell_h, spec


# Cell geometry is uniform across the grid (same replicate counts/lengths at
# every z_down/draw_speed), so compute it once for layout purposes.
_PROBE_POLYS, CELL_W, CELL_H, _PROBE_SPEC = build_penwidth_cell(0, 0)
CELL_GAP_X = 5.0
CELL_GAP_Y = 6.0

FOOTPRINT_W = GRID_COLS * CELL_W + (GRID_COLS - 1) * CELL_GAP_X
FOOTPRINT_H = GRID_ROWS * CELL_H + (GRID_ROWS - 1) * CELL_GAP_Y

_PAD_MM = 3.0
_USABLE_W = PAPER_W_MM - 2 * SHEET_MARGIN_MM
_USABLE_H = PAPER_H_MM - HOLE_KEEPOUT_MM - 2 * SHEET_MARGIN_MM
assert FOOTPRINT_W + 2 * _PAD_MM <= _USABLE_W, (
    f"penwidth sheet footprint width {FOOTPRINT_W + 2 * _PAD_MM:.1f}mm exceeds "
    f"the JIS B5 landscape usable width {_USABLE_W:.1f}mm "
    f"({PAPER_W_MM}mm paper - 2x{SHEET_MARGIN_MM}mm margin)")
assert FOOTPRINT_H + 2 * _PAD_MM <= _USABLE_H, (
    f"penwidth sheet footprint height {FOOTPRINT_H + 2 * _PAD_MM:.1f}mm exceeds "
    f"the JIS B5 landscape usable height {_USABLE_H:.1f}mm "
    f"({PAPER_H_MM}mm paper - {HOLE_KEEPOUT_MM}mm hole keep-out - "
    f"2x{SHEET_MARGIN_MM}mm margin)")


# A cell's longest legitimate pen-down move is one isolated stroke (LEN_45,
# the longer of the two groups). PENDOWN_MAX_MM sits between that and the
# much larger cell-to-cell pitch (CELL_W+CELL_GAP_X), so a cell-to-cell drag
# is caught, at a threshold sized for THIS sheet's own geometry
# (calibration_deposition.PENDOWN_MAX_MM=15.0 is sized for its 8mm patches
# and would false-flag this sheet's legitimate ~35mm diagonal strokes).
PENDOWN_MAX_MM = round(LEN_45 + 5.0, 1)


def _assert_no_pendown_travel(path, draw_z_levels):
    """Fail loudly if any pen-down move exceeds PENDOWN_MAX_MM -- same check as
    calibration_deposition._assert_no_pendown_travel (only the threshold
    differs), which catches an un-anchored splice dragging the pen between
    cells."""
    import re as _re
    x = y = None
    down = False
    worst = 0.0
    n = 0
    n_examined = 0
    with Path(path).open() as fh:
        for line in fh:
            if not line.startswith("G1"):
                continue
            mz = _re.search(r"Z(-?[\d.]+)", line)
            if mz:
                down = round(float(mz.group(1)), 2) in draw_z_levels
                continue
            mx = _re.search(r"X(-?[\d.]+)", line)
            my = _re.search(r"Y(-?[\d.]+)", line)
            nx = float(mx.group(1)) if mx else x
            ny = float(my.group(1)) if my else y
            if down and x is not None and nx is not None:
                d = math.hypot(nx - x, ny - y)
                n_examined += 1
                worst = max(worst, d)
                if d > PENDOWN_MAX_MM:
                    n += 1
            x, y = nx, ny
    if n:
        raise RuntimeError(
            f"Penwidth sheet has {n} pen-DOWN move(s) longer than {PENDOWN_MAX_MM}mm "
            f"(worst {worst:.1f}mm). The pen would drag ink between cells/strokes -- "
            "the same splice defect calibration_deposition.py's post-condition catches; "
            "do not plot this file.")
    return {
        "pendown_travel_violations": 0,
        "max_pendown_mm": round(worst, 2),
        "pendown_segments_examined": n_examined,
    }


def penwidth_gcode(out_gcode: Path, pen_mm=None, z_up=3.5,
                   travel_speed=8000, plunge_speed=500, lift_speed=2000,
                   offset_x=None, offset_y=None, end_policy="release",
                   quiet_fans=True, profile=None,
                   z_down_levels=Z_DOWN_LEVELS, draw_speeds=DRAW_SPEEDS):
    """Emit the full penwidth-distribution sheet as ONE g-code file.

    Z comes from the active pen profile via resolve_z_touch() -- PROFILE
    WINS, raises ZTouchNotCalibrated rather than defaulting (same convention
    as every generator in this package). The reach clamp (e.g. the angled
    Preppy mount's documented y_max) is honoured via pen_reach_profile(),
    exactly like calibration_deposition.deposition_gcode -- a profile with no
    reach_clamp gets the unclamped machine envelope back unchanged."""
    ppmm = 1.0 / PX2MM
    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()
    if profile is None:
        profile = active_pen_profile()
    reach = pen_reach_profile(active_reach_profile(cal), profile)
    if pen_mm is None:
        pen_mm = float(profile.get("pen_mm", 0.5))

    z_touch, z_touch_source = resolve_z_touch(cal, profile)
    pen_off_x, pen_off_y = resolve_pen_offset(cal, profile)
    PLACEMENT_MARGIN_MM = 20.0
    pad = _PAD_MM
    canvas_w_mm, canvas_h_mm = FOOTPRINT_W + 2 * pad, FOOTPRINT_H + 2 * pad
    offset_x = reach["bbox"][0] + PLACEMENT_MARGIN_MM if offset_x is None else float(offset_x)
    if offset_y is None:
        # The elegoo_cc gwrite profile's `vertical_flip` mirrors Y about
        # max(canvas_w_mm, canvas_h_mm) for a LANDSCAPE canvas (wider than tall),
        # NOT about the canvas height. A default-placement formula of
        # `bbox[1] + MARGIN + pen_off_y` silently assumes the flip axis is
        # canvas_h_mm; that error equals the canvas W-H gap, which is ~47mm for
        # this sheet and would put the content ~47mm further from the plotter
        # front than intended (validate_gcode_xy_motion below catches the
        # resulting bbox escape).
        #
        # So solve directly for the machine offset that lands the NEAR (front)
        # edge of the content at reach.bbox[1]+PLACEMENT_MARGIN_MM, using the
        # ACTUAL flip axis. The content's own machine-Y SPAN is invariant to this
        # (it only shifts by a fixed additive amount), so this is exact, not an
        # approximation.
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

    texts, labels, cells = [], [], []
    for r, speed in enumerate(draw_speeds):
        for c, z_down in enumerate(z_down_levels):
            polys, _cw, _ch, spec = build_penwidth_cell(c, r)
            dx = pad + c * (CELL_W + CELL_GAP_X)
            dy = pad + r * (CELL_H + CELL_GAP_Y)
            placed = [[((x + dx) * ppmm, (y + dy) * ppmm) for x, y in p] for p in polys]
            # ONE temp svg/gcode path is reused by every cell: see
            # calibration_deposition.deposition_gcode for why a per-cell temp
            # name would corrupt _splice_blocks' boundary detection (the
            # {filename} interpolated into document_start).
            svg = WORK_DIR / f".{out_gcode.stem}_cell.svg"
            _write_paths_svg(svg, W, H, float(pen_mm) * ppmm, polylines=placed)

            z_down_abs = z_touch + float(z_down)
            vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({
                "z_up": float(z_up), "z_up_abs": z_touch + float(z_up),
                "z_down": float(z_down), "z_down_abs": z_down_abs,
                "z_touch_abs": z_touch, "z_travel_abs": z_travel_abs,
                "draw_speed": float(speed), "travel_speed": float(travel_speed),
                "plunge_speed": float(plunge_speed), "lift_speed": float(lift_speed),
                "z_auto": "1", "origin_x": park_x, "origin_y": park_y,
                **_gcode_lifecycle_values(end_policy, quiet_fans)})

            tmp = WORK_DIR / f".{out_gcode.stem}_cell.gcode"
            vpype_cli.execute(f'read {shlex.quote(str(svg))} pagesize {W}x{H}px '
                              f'translate -- {comp_off_x}mm {-comp_off_y}mm '
                              f'gwrite --profile elegoo_cc {shlex.quote(str(tmp))}')
            texts.append(tmp.read_text(encoding="utf-8"))
            labels.append(f"; CELL col={c+1} row={r+1} z_down={z_down} "
                          f"z_down_abs={z_down_abs:.2f} draw_speed={speed}")
            cells.append({"col": c + 1, "row": r + 1, "z_down": z_down,
                          "z_down_abs": round(z_down_abs, 3), "draw_speed": speed,
                          "origin_mm": [round(dx, 2), round(dy, 2)], **spec})
    tmp.unlink(missing_ok=True)
    svg.unlink(missing_ok=True)

    combined = _splice_blocks(texts, labels, z_touch + float(z_up))
    with atomic_target(out_gcode) as tmp_gcode:
        Path(tmp_gcode).write_text(combined, encoding="utf-8")
        _fix_gwrite_source_comment(tmp_gcode, out_gcode)

    validate_gcode_xy_motion(out_gcode, reach)
    deepest = z_touch + min(float(z) for z in z_down_levels)
    _assert_no_pendown_travel(out_gcode, {round(z_touch + float(z), 2) for z in z_down_levels})
    safety = validate_calibration_gcode_safety(out_gcode, deepest)

    all_polys = []
    for r in range(len(draw_speeds)):
        for c in range(len(z_down_levels)):
            polys, _cw, _ch, _spec = build_penwidth_cell(c, r)
            dx = pad + c * (CELL_W + CELL_GAP_X)
            dy = pad + r * (CELL_H + CELL_GAP_Y)
            all_polys += [[((x + dx) * ppmm, (y + dy) * ppmm) for x, y in p] for p in polys]
    prev_svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_paths_svg(prev_svg, W, H, float(pen_mm) * ppmm, polylines=all_polys)
    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(prev_svg))} color {PREVIEW_COLOR} '
                          f'write {shlex.quote(str(tmp_prev))}')

    n_strokes = len(cells) * (N_STROKES_0 + N_STROKES_45)
    stroke_len_total_mm = len(cells) * (N_STROKES_0 * LEN_0 + N_STROKES_45 * LEN_45)

    return {"kind": "penwidth", "pen_mm": pen_mm,
            "footprint_mm": [round(FOOTPRINT_W + 2 * pad, 1), round(FOOTPRINT_H + 2 * pad, 1)],
            "offset_mm": [offset_x, offset_y],
            "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
            "z_down_levels": list(z_down_levels), "draw_speeds": list(draw_speeds),
            "angles_deg": list(ANGLES_DEG),
            "n_cells": len(cells), "n_strokes": n_strokes,
            "n_strokes_0deg_per_cell": N_STROKES_0, "n_strokes_45deg_per_cell": N_STROKES_45,
            "stroke_len_0deg_mm": LEN_0, "stroke_len_45deg_mm": round(LEN_45, 3),
            "stroke_len_total_mm": round(stroke_len_total_mm, 1),
            "deepest_z_down_abs": round(deepest, 3),
            "safety_selfcheck": safety, "preview": prev.name,
            "svg_path": str(prev_svg), "svg_wh_px": [W, H],
            "pattern_mm": [round(FOOTPRINT_W + 2 * pad, 1), round(FOOTPRINT_H + 2 * pad, 1)],
            "place": {"x": offset_x, "y": offset_y,
                      "w": FOOTPRINT_W + 2 * pad, "h": FOOTPRINT_H + 2 * pad},
            "z_up_abs": round(z_touch + float(z_up), 3),
            "z_down_abs": round(deepest, 3),
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
            "cells": cells}
