"""ccplotter/calibration_singlepass.py -- single-pass landing test sheet.

What it measures: whether an isolated single pen pass leaves a mark, as a function of
draw speed and pen pressure (z_down). Each of 13 cells has a dense CONTROL patch that must
ink (so a blank cell is a real result, not a dead pen), plus two blocks of 5 replicates:
  * ISOLATED  -- single 8 mm strokes drawn as 1 pass (left) and 2 passes (right), each
                 its own pen-down;
  * CONTINUED -- the same 1-pass / 2-pass strokes, but each preceded by a 4 mm dense
                 ANCHOR patch so the test stroke starts with the pen already inking.
Comparing the two blocks separates "the pen does not start" from "the pen does not keep
going".

How to read it: sweep the grid. Columns are the four pressures (Z_DOWN_LEVELS), rows are
the three speeds (DRAW_SPEEDS_MAIN); the extra 13th cell in the fifth column is a
NEGATIVE CONTROL at the stock Z and a 3000 mm/min speed. Count which strokes are blank in
each cell. Cell-identity ticks at the bottom (LEFT = column index, RIGHT = row index) keep
a loose fragment identifiable.

HARD-CODED MACHINE AND PEN (limitation). The sheet is tied to the author's measured machine
and pen. `singlepass_gcode` has five hard `RuntimeError` shadow guards: EXPECT_REACH_ID
("plotter_2003_v32"), EXPECT_BBOX, EXPECT_PEN_ID ("uright_gel"), EXPECT_Z_TOUCH_ABS (11.25)
and EXPECT_Z_DOWN_STOCK. It resolves that one named pen profile directly out of
pen_profiles.json rather than the currently ACTIVE profile that other calibration_* modules
use, because a sheet generated silently against a different pen's Z or reach would be worse
than useless, so it refuses instead. Using another pen means a deliberate edit of
EXPECT_PEN_ID and the constants, not a parameter.

The sweep is emitted as separate gwrite blocks per (z_down, draw_speed) cell, spliced by
calibration_deposition's `_splice_blocks`; do not plot it from the canvas, which would
flatten every cell to one Z and speed.

Reached only through the registry-dispatched POST /api/calibration_sheet (id="singlepass").
"""
import math
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
from ccplotter.calibration_deposition import _splice_blocks, serpentine
from ccplotter.calibration_plots import ticks

# ---------------------------------------------------------------------------
# Expected runtime envelope -- re-verified at generation time, not assumed. A
# mismatch here means this sheet is not safe to generate against whatever
# pen/reach IS currently active/configured (for example an environment without
# reach_profiles falls back to a different default reach profile), so it refuses
# rather than silently drawing at the wrong Z or against the wrong reach
# envelope. Tied to the author's measured machine geometry.
# ---------------------------------------------------------------------------
EXPECT_REACH_ID = "plotter_2003_v32"
EXPECT_BBOX = (3.0, 42.5, 255.0, 254.5)
EXPECT_PEN_ID = "uright_gel"
EXPECT_Z_TOUCH_ABS = 11.25
EXPECT_Z_DOWN_STOCK = -0.60

# ---------------------------------------------------------------------------
# Paper / keep-out (mm). JIS B5 182x257 landscape, same formula as
# calibration_penwidth.py.
# ---------------------------------------------------------------------------
PAPER_W_MM, PAPER_H_MM = 257.0, 182.0
HOLE_KEEPOUT_MM = 10.0
SHEET_MARGIN_MM = 5.0

# ---------------------------------------------------------------------------
# Sweep levels. z_down brackets the stock -0.60. Speeds are a 500/1000/1500 main
# sweep, plus ONE negative-control cell at the stock 3000.
# ---------------------------------------------------------------------------
Z_DOWN_LEVELS = (-0.20, -0.60, -1.00, -1.40)
DRAW_SPEEDS_MAIN = (500, 1000, 1500)
NEG_CONTROL_SPEED = 3000
NEG_CONTROL_Z_DOWN = -0.60   # must equal EXPECT_Z_DOWN_STOCK -- asserted below
assert NEG_CONTROL_Z_DOWN == EXPECT_Z_DOWN_STOCK

GRID_COLS = len(Z_DOWN_LEVELS) + 1   # +1 reserves the 5th column for the
                                       # negative-control cell so its tick
                                       # pair never collides with a real
                                       # grid coordinate
GRID_ROWS = len(DRAW_SPEEDS_MAIN)     # 3

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Cell geometry (mm, local cell space, y grows down).
# ---------------------------------------------------------------------------
STROKE_LEN = 8.0
N_REPLICATES = 5

ANCHOR_W = 1.0
ANCHOR_PITCH = 0.2
ANCHOR_CONT_LEN = 4.0

MARK_LEN = 1.0
MARK_GAP = 4.0
BLOCK_X0 = MARK_LEN + MARK_GAP
COL_GAP = 4.0
ROW_PITCH = 3.0
BLOCK_PAD = 1.0

CONTROL_SIZE = 4.0
CONTROL_PITCH = 0.2
CONTROL_GAP = 3.0
BLOCK_GAP = 6.0

TICK_H = 1.8
TICKS_GAP = 2.0

UNIT_W = {"isolated": STROKE_LEN,
          "continued": ANCHOR_CONT_LEN + STROKE_LEN}
BLOCK_CONTENT_W = {k: BLOCK_X0 + 2 * w + COL_GAP for k, w in UNIT_W.items()}
CELL_W = max(BLOCK_CONTENT_W.values())
BLOCK_H = (N_REPLICATES - 1) * ROW_PITCH + 2 * BLOCK_PAD

_Y_ISOLATED0 = CONTROL_SIZE + CONTROL_GAP
_Y_CONTINUED0 = _Y_ISOLATED0 + BLOCK_H + BLOCK_GAP
_Y_TICKS0 = _Y_CONTINUED0 + BLOCK_H + TICKS_GAP
CELL_H = _Y_TICKS0 + TICK_H * 1.5 + 1.0

CELL_GAP_X = 6.0
CELL_GAP_Y = 6.0
FOOTPRINT_W = GRID_COLS * CELL_W + (GRID_COLS - 1) * CELL_GAP_X
FOOTPRINT_H = GRID_ROWS * CELL_H + (GRID_ROWS - 1) * CELL_GAP_Y

_PAD_MM = 3.0
_USABLE_W = PAPER_W_MM - 2 * SHEET_MARGIN_MM
_USABLE_H = PAPER_H_MM - HOLE_KEEPOUT_MM - 2 * SHEET_MARGIN_MM
assert FOOTPRINT_W + 2 * _PAD_MM <= _USABLE_W, (
    f"singlepass v2 sheet footprint width {FOOTPRINT_W + 2 * _PAD_MM:.1f}mm exceeds "
    f"the JIS B5 landscape usable width {_USABLE_W:.1f}mm")
assert FOOTPRINT_H + 2 * _PAD_MM <= _USABLE_H, (
    f"singlepass v2 sheet footprint height {FOOTPRINT_H + 2 * _PAD_MM:.1f}mm exceeds "
    f"the JIS B5 landscape usable height {_USABLE_H:.1f}mm")

PENDOWN_MAX_MM = 12.0


# ---------------------------------------------------------------------------
# Explicit cell list -- NOT a nested double loop over two full axes, because
# the negative-control cell is intentionally outside the regular grid.
# Each entry: (col_idx, row_idx, z_down, draw_speed, is_negative_control, label).
# ---------------------------------------------------------------------------
def build_cell_specs():
    specs = []
    for row_idx, speed in enumerate(DRAW_SPEEDS_MAIN):
        for col_idx, z_down in enumerate(Z_DOWN_LEVELS):
            specs.append({"col_idx": col_idx, "row_idx": row_idx,
                          "z_down": z_down, "draw_speed": speed,
                          "is_negative_control": False})
    specs.append({"col_idx": GRID_COLS - 1, "row_idx": GRID_ROWS - 1,
                  "z_down": NEG_CONTROL_Z_DOWN, "draw_speed": NEG_CONTROL_SPEED,
                  "is_negative_control": True})
    return specs


BUILD_CELLS = build_cell_specs()
assert len(BUILD_CELLS) == 13
assert sum(1 for s in BUILD_CELLS if s["is_negative_control"]) == 1


# ---------------------------------------------------------------------------
# Geometry builders (ISOLATED / CONTINUED)
# ---------------------------------------------------------------------------
def _reliable(polys, times=3):
    out = []
    for _ in range(times):
        out.extend(polys)
    return out


def dense_patch(x0, y0, length, width, pitch=ANCHOR_PITCH):
    return serpentine(x0, y0, length, width, pitch)


def row_marker(y):
    return _reliable([[(0.0, y - MARK_LEN / 2), (0.0, y + MARK_LEN / 2)]])


def isolated_unit(x0, y, passes):
    return [[(x0, y), (x0 + STROKE_LEN, y)] for _ in range(passes)]


def continued_unit(x0, y, passes):
    anchor = dense_patch(x0, y, ANCHOR_CONT_LEN, ANCHOR_W)
    test_y = y + ANCHOR_W / 2.0
    test_x0 = x0 + ANCHOR_CONT_LEN
    test = [[(test_x0, test_y), (test_x0 + STROKE_LEN, test_y)] for _ in range(passes)]
    return anchor + test


_UNIT_BUILDERS = {"isolated": isolated_unit, "continued": continued_unit}


def build_block(kind, y0):
    builder = _UNIT_BUILDERS[kind]
    uw = UNIT_W[kind]
    polys = []
    groups = []
    for r in range(N_REPLICATES):
        y = y0 + BLOCK_PAD + r * ROW_PITCH
        if kind == "isolated":
            polys += row_marker(y)
        x_p1 = BLOCK_X0
        x_p2 = BLOCK_X0 + uw + COL_GAP
        u1 = builder(x_p1, y, 1)
        u2 = builder(x_p2, y, 2)
        polys += u1
        polys += u2
        n_anchor = 0 if kind == "isolated" else 1
        groups.append({"kind": kind, "replicate": r + 1, "pass_count": 1,
                       "n_anchor_pendowns": n_anchor, "n_test_pendowns": 1})
        groups.append({"kind": kind, "replicate": r + 1, "pass_count": 2,
                       "n_anchor_pendowns": n_anchor, "n_test_pendowns": 2})
    return polys, groups


def build_cell(col_idx, row_idx):
    """One cell: control patch, ISOLATED + CONTINUED blocks, cell-identity
    ticks."""
    polys = list(dense_patch(0.0, 0.0, CONTROL_SIZE, CONTROL_SIZE, CONTROL_PITCH))
    all_groups = []
    for kind, y0 in (("isolated", _Y_ISOLATED0), ("continued", _Y_CONTINUED0)):
        bp, bg = build_block(kind, y0)
        polys += bp
        all_groups += bg

    polys += _reliable([[(0.0, _Y_TICKS0), (10.0, _Y_TICKS0)]], times=2)
    polys += _reliable(ticks(0.0, 10.0, _Y_TICKS0, col_idx + 1, tick_h=TICK_H), times=2)
    rx = CELL_W - 10.0
    polys += _reliable([[(rx, _Y_TICKS0), (rx + 10.0, _Y_TICKS0)]], times=2)
    polys += _reliable(ticks(rx, 10.0, _Y_TICKS0, row_idx + 1, tick_h=TICK_H), times=2)

    spec = {"groups": all_groups, "col_ticks": col_idx + 1, "row_ticks": row_idx + 1,
            "control_xy_mm": [0.0, 0.0], "control_size_mm": CONTROL_SIZE}
    return polys, spec


def _spotcheck_group_order(polys):
    """Confirm the anchor precedes its test stroke in emission order (no
    linesort anywhere in this file)."""
    bp, _ = build_block("continued", 0.0)
    anchor_len = len(dense_patch(BLOCK_X0, BLOCK_PAD, ANCHOR_CONT_LEN, ANCHOR_W))
    idx_anchor = 0
    idx_test = anchor_len
    assert len(bp[idx_anchor]) > 2, "expected anchor polyline (many points) at this index"
    assert len(bp[idx_test]) == 2, "expected 2-point test stroke immediately after the anchor"
    return {"anchor_before_test": True, "anchor_index": idx_anchor, "test_index": idx_test}


# ---------------------------------------------------------------------------
# Safety self-checks (own PENDOWN_MAX_MM threshold -- deposition's/
# penwidth's are module constants baked into THEIR functions and cannot be
# reused directly).
# ---------------------------------------------------------------------------
def _assert_no_pendown_travel(path, draw_z_levels):
    x = y = None
    down = False
    worst = 0.0
    n = 0
    n_examined = 0
    with Path(path).open() as fh:
        for line in fh:
            if not line.startswith("G1"):
                continue
            mz = re.search(r"Z(-?[\d.]+)", line)
            if mz:
                down = round(float(mz.group(1)), 2) in draw_z_levels
                continue
            mx = re.search(r"X(-?[\d.]+)", line)
            my = re.search(r"Y(-?[\d.]+)", line)
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
            f"Singlepass v2 sheet has {n} pen-DOWN move(s) longer than {PENDOWN_MAX_MM}mm "
            f"(worst {worst:.1f}mm). Do not plot this file.")
    return {
        "pendown_travel_violations": 0,
        "max_pendown_mm": round(worst, 2),
        "pendown_segments_examined": n_examined,
    }


def _count_pendown_segments(path, draw_z_levels):
    """Counts pen-down CYCLES (Z-down to next Z-up), not XY lines seen while
    down."""
    n_down_cycles = 0
    down = False
    with Path(path).open() as fh:
        for line in fh:
            if not line.startswith("G1"):
                continue
            mz = re.search(r"Z(-?[\d.]+)", line)
            if not mz:
                continue
            z = round(float(mz.group(1)), 2)
            is_down = z in draw_z_levels
            if is_down and not down:
                n_down_cycles += 1
            down = is_down
    return n_down_cycles


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------
def singlepass_gcode(out_gcode: Path, pen_mm=None, z_up=3.5,
                     travel_speed=8000, plunge_speed=500, lift_speed=2000,
                     offset_x=None, offset_y=None, end_policy="release",
                     quiet_fans=True, profile=None, cell_specs=BUILD_CELLS):
    """Emit the full single-pass landing sheet as ONE g-code file.

    Unlike most other calibration_* generators in this package, this one
    does NOT fall back to the currently-ACTIVE pen profile when `profile` is
    omitted -- it resolves EXPECT_PEN_ID directly and hard-asserts the
    resolved reach envelope / pen / z_touch_abs / stock z_down against the
    EXPECT_* constants (see module docstring)."""
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
    if profile is None:
        profile = profiles["profiles"][EXPECT_PEN_ID]
    z_touch, z_touch_source = resolve_z_touch(cal, profile)
    if abs(z_touch - EXPECT_Z_TOUCH_ABS) > 1e-6:
        raise RuntimeError(f"z_touch_abs mismatch for {EXPECT_PEN_ID}: got {z_touch}, "
                           f"expected {EXPECT_Z_TOUCH_ABS}")
    if abs(float(profile.get("z_down", 0.0)) - EXPECT_Z_DOWN_STOCK) > 1e-6:
        raise RuntimeError(f"stock z_down mismatch for {EXPECT_PEN_ID}: got "
                           f"{profile.get('z_down')}, expected {EXPECT_Z_DOWN_STOCK}")

    reach = pen_reach_profile(reach0, profile)
    if pen_mm is None:
        pen_mm = float(profile.get("pen_mm", 0.255))

    pen_off_x, pen_off_y = resolve_pen_offset(cal, profile)
    PLACEMENT_MARGIN_MM = 20.0
    pad = _PAD_MM
    canvas_w_mm, canvas_h_mm = FOOTPRINT_W + 2 * pad, FOOTPRINT_H + 2 * pad

    offset_x = reach["bbox"][0] + PLACEMENT_MARGIN_MM if offset_x is None else float(offset_x)
    if offset_y is None:
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
    z_down_levels_all = sorted(set(Z_DOWN_LEVELS) | {NEG_CONTROL_Z_DOWN})
    for spec in cell_specs:
        c, r = spec["col_idx"], spec["row_idx"]
        z_down, speed = spec["z_down"], spec["draw_speed"]
        polys, cspec = build_cell(c, r)
        dx = pad + c * (CELL_W + CELL_GAP_X)
        dy = pad + r * (CELL_H + CELL_GAP_Y)
        placed = [[((x + dx) * ppmm, (y + dy) * ppmm) for x, y in p] for p in polys]
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
        nc_tag = " NEGATIVE_CONTROL" if spec["is_negative_control"] else ""
        labels.append(f"; CELL col={c+1} row={r+1} z_down={z_down} "
                      f"z_down_abs={z_down_abs:.2f} draw_speed={speed}{nc_tag}")
        n_groups = len(cspec["groups"])
        n_pendowns_expected = len(polys)
        cells.append({"col": c + 1, "row": r + 1, "z_down": z_down,
                      "z_down_abs": round(z_down_abs, 3), "draw_speed": speed,
                      "is_negative_control": spec["is_negative_control"],
                      "origin_mm": [round(dx, 2), round(dy, 2)],
                      "n_groups": n_groups, "n_pendowns_expected": n_pendowns_expected,
                      **cspec})
    tmp.unlink(missing_ok=True)
    svg.unlink(missing_ok=True)

    combined = _splice_blocks(texts, labels, z_touch + float(z_up))
    with atomic_target(out_gcode) as tmp_gcode:
        Path(tmp_gcode).write_text(combined, encoding="utf-8")
        _fix_gwrite_source_comment(tmp_gcode, out_gcode)

    validate_gcode_xy_motion(out_gcode, reach)
    deepest = z_touch + min(float(z) for z in z_down_levels_all)
    draw_z_set = {round(z_touch + float(z), 2) for z in z_down_levels_all}
    travel_check = _assert_no_pendown_travel(out_gcode, draw_z_set)
    safety = validate_calibration_gcode_safety(out_gcode, deepest)
    n_pendown_total = _count_pendown_segments(out_gcode, draw_z_set)
    n_pendown_expected_total = sum(c["n_pendowns_expected"] for c in cells)

    order_check = _spotcheck_group_order(build_cell(0, 0)[0])

    all_polys = []
    for spec in cell_specs:
        c, r = spec["col_idx"], spec["row_idx"]
        polys, _ = build_cell(c, r)
        dx = pad + c * (CELL_W + CELL_GAP_X)
        dy = pad + r * (CELL_H + CELL_GAP_Y)
        all_polys += [[((x + dx) * ppmm, (y + dy) * ppmm) for x, y in p] for p in polys]
    prev_svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_paths_svg(prev_svg, W, H, float(pen_mm) * ppmm, polylines=all_polys)
    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(prev_svg))} color {PREVIEW_COLOR} '
                          f'write {shlex.quote(str(tmp_prev))}')

    return {"kind": "singlepass_v2", "pen_mm": pen_mm, "pen_id": EXPECT_PEN_ID,
            "footprint_mm": [round(FOOTPRINT_W + 2 * pad, 1), round(FOOTPRINT_H + 2 * pad, 1)],
            "usable_mm": [round(_USABLE_W, 1), round(_USABLE_H, 1)],
            "offset_mm": [offset_x, offset_y],
            "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
            "z_down_levels": list(Z_DOWN_LEVELS), "draw_speeds_main": list(DRAW_SPEEDS_MAIN),
            "neg_control_speed": NEG_CONTROL_SPEED, "neg_control_z_down": NEG_CONTROL_Z_DOWN,
            "n_cells": len(cells), "n_groups_total": sum(c["n_groups"] for c in cells),
            "n_pendown_expected_total": n_pendown_expected_total,
            "n_pendown_actual_total": n_pendown_total,
            "pendown_count_matches": n_pendown_total == n_pendown_expected_total,
            "deepest_z_down_abs": round(deepest, 3),
            "safety_selfcheck": safety, "travel_selfcheck": travel_check,
            "order_selfcheck": order_check,
            "preview": prev.name, "svg_path": str(prev_svg), "svg_wh_px": [W, H],
            "pattern_mm": [round(FOOTPRINT_W + 2 * pad, 1), round(FOOTPRINT_H + 2 * pad, 1)],
            "place": {"x": offset_x, "y": offset_y, "w": canvas_w_mm, "h": canvas_h_mm},
            "z_up_abs": round(z_touch + float(z_up), 3), "z_down_abs": round(deepest, 3),
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
            "cells": cells}
