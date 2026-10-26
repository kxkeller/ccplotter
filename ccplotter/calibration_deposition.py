"""ccplotter/calibration_deposition.py -- DEPOSITION calibration sheet.

What it measures: whether ink actually reaches the paper on passes that were commanded.
A fill can show white gaps even when the commanded pass pitch is much finer than the
pen's laid line width; in that case spacing is not the cause. Typical figures for a gel
pen: commanded pitch 0.117 mm, laid line width 0.256 mm (profile value 0.500), so
geometric coverage is ~2.2x redundant, yet ~28 % of a preview-SOLID area stays white.
This sheet sweeps the physical suspects that the other charts hold fixed.

How it differs from the other charts: the wedge (`calibration_plots.build_wedge_polylines`)
answers "what spacing gives what tone" and the linewidth block answers "how wide is the
line". Both hold Z and speed FIXED at the profile's values, so neither can see a
deposition failure. This sheet holds geometry near-constant and sweeps Z and speed.

FACTORS (full factorial, one sheet, ~160 x 127 mm):

  z_down     5 levels   contact pressure (Z relative to the profile's z_touch_abs).
                        The prime suspect: a draw Z only 0.6 mm below touch may not
                        load the pen enough.
  draw_speed 4 levels   a gel pen starved at speed lays less ink.
  pitch      3 levels   0.5x / 1.0x / 1.5x fill_mm. The CONTROL: if the 1.5x column
                        cracks no worse than the 0.5x column, spacing is not the cause
                        and `fill_mm` tuning will not help.
  lift       2 levels   continuous serpentine vs the same coverage chopped into
                        CHOP_MM strokes, which forces a pen-up/pen-down between each.
                        Isolates lift recovery: strokes start dry (a measured sheet
                        showed 40 % ink in the first 0.15 mm vs 60 % past 3.5 mm).

  = 5 x 4 x 3 x 2 = 120 fill patches, plus one isolated-singles strip per (z, speed)
  cell (20 strips, 60 lines) for reading laid WIDTH under a loupe at each
  pressure/speed.

How to read it: compare patches along the pressure columns, speed rows and pitch
columns; the cell where white gaps vanish gives the Z and speed to use. Tick marks
(LEFT group = pressure column index 1..5, RIGHT group = speed row index 1..4) identify a
patch cut off the sheet.

HOW THE SWEEP IS EMITTED. `draw_speed` and `z_down` are gwrite `default_values`, so they
cannot vary within a single gwrite run. Each (z, speed) cell is therefore its own
complete gwrite output, and the 20 outputs are spliced into ONE file by
`_splice_blocks`. This is safe for the same reason `calibration_plots._splice_pressure_blocks`
is: elegoo_cc's `document_start` interpolates z_travel_abs / origin / travel_speed /
lift_speed / fan lifecycle, and `document_end` the same -- NEITHER references draw_speed
or z_down. Those two appear only in `segment` (`F{draw_speed}`) and `segment_first`
(`Z{z_down_abs}`), i.e. in the BODY. So all 20 outputs share a byte-identical preamble
and postamble, and every drawing move carries its own explicit F. `_splice_blocks`
verifies that rather than assuming it, and raises if it fails.

NO LINESORT, as in the other charts: routing would bridge the gaps being measured.
Every cell keeps its own emission order.

Safety: Z comes from the active pen profile (raises if not calibrated), the whole file
is checked against the DEEPEST z_down of the sweep, XY motion is checked against the
reach profile, and a post-condition rejects any pen-down move long enough to be a
cell-to-cell drag. Unlike several sibling sheets this module has no EXPECT_REACH_ID
pin; it uses whatever reach profile is active.

The sheet is registered in calibration_registry under id "deposition".
"""
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
from ccplotter.calibration_plots import ticks, rect_outline

# ---------------------------------------------------------------------------
# Cell geometry (mm, layout space, y grows down -- same convention as
# calibration_plots).
# ---------------------------------------------------------------------------
PATCH = 8.0          # fill patch side. 8mm at the tightest pitch is ~91 passes:
                     # plenty of structure to read, and small enough that the
                     # full factorial fits one B5 sheet.
PGAP = 2.5           # gap between patches inside a cell
SINGLES_GAP = 3.0    # gap above the isolated-singles strip
SINGLES_LEN = 8.0    # length of each isolated line
SINGLES_N = 3
SINGLES_PITCH = 2.0  # far wider than any pen: these must NOT touch
TICK_H = 1.8
CHOP_MM = 3.0        # chopped-stroke length. Not shorter: 8mm/3mm = 3 strokes
                     # per pass, ~13k extra pen-downs over the sheet already.

PITCH_MULTS = (0.5, 1.0, 1.5)
N_PITCH = len(PITCH_MULTS)

CELL_W = N_PITCH * PATCH + (N_PITCH - 1) * PGAP            # 29.0
_ROW1_Y = PATCH + PGAP                                      # 10.5
_SINGLES_Y = 2 * PATCH + PGAP + SINGLES_GAP                 # 21.5
_TICKS_Y = _SINGLES_Y + (SINGLES_N - 1) * SINGLES_PITCH + 2.0
CELL_H = _TICKS_Y + TICK_H * 1.5 + 1.0

CELL_GAP_X = 4.0
CELL_GAP_Y = 5.0

# Sweep levels. z_down is RELATIVE to the profile's z_touch_abs, exactly like
# every other generator here: more negative = harder press. The profile's own
# value (-0.6 for uright_gel) is one of the levels so the sheet contains its own
# baseline rather than only excursions from it.
Z_DOWN_LEVELS = (-0.2, -0.6, -1.0, -1.4, -1.8)
DRAW_SPEEDS = (1000, 2000, 3000, 4500)

GRID_COLS = len(Z_DOWN_LEVELS)      # 5 -- one column per pressure
GRID_ROWS = len(DRAW_SPEEDS)        # 4 -- one row per speed

FOOTPRINT_W = GRID_COLS * CELL_W + (GRID_COLS - 1) * CELL_GAP_X
FOOTPRINT_H = GRID_ROWS * CELL_H + (GRID_ROWS - 1) * CELL_GAP_Y


def serpentine(x0, y0, w, h, pitch):
    """ONE continuous boustrophedon polyline filling the patch at `pitch`.

    One polyline, so the plotter draws it with a single pen-down (the 'continuous' arm). Compare hatch_h() in calibration_plots,
    which returns one polyline PER LINE and would lift between every pass."""
    pts = []
    y = y0
    left = True
    while y <= y0 + h + 1e-9:
        pts.append((x0 if left else x0 + w, y))
        pts.append((x0 + w if left else x0, y))
        y += pitch
        if y <= y0 + h + 1e-9:
            pts.append((x0 + w if left else x0, y))
        left = not left
    return [pts] if len(pts) > 1 else []


def chopped(x0, y0, w, h, pitch, chop=CHOP_MM):
    """The SAME coverage as serpentine(), cut into separate polylines of
    `chop` mm laid end to end. Identical ink geometry, many more pen-downs --
    so any difference between this patch and its serpentine twin is lift
    recovery, not spacing."""
    out = []
    y = y0
    left = True
    while y <= y0 + h + 1e-9:
        n = max(1, int(round(w / chop)))
        edges = [x0 + w * k / n for k in range(n + 1)]
        spans = list(zip(edges[:-1], edges[1:]))
        if not left:
            spans = [(b, a) for a, b in reversed(spans)]
        for a, b in spans:
            out.append([(a, y), (b, y)])
        y += pitch
        left = not left
    return out


def singles(x0, y0):
    """Isolated lines, far enough apart that they cannot merge. Gives the
    pen's laid WIDTH at this cell's pressure and speed, directly, under a
    loupe or a 1200dpi scan."""
    return [[(x0, y0 + k * SINGLES_PITCH), (x0 + SINGLES_LEN, y0 + k * SINGLES_PITCH)]
            for k in range(SINGLES_N)]


def build_deposition_cell(fill_mm, col_idx, row_idx):
    """One (z_down, draw_speed) cell in local mm coords. Returns (polys, spec).

    Row 0 = continuous serpentine at the three pitches.
    Row 1 = the same three pitches, chopped.
    Then the singles strip, then tick marks: LEFT group = pressure column
    index (1..5), RIGHT group = speed row index (1..4), so a loose patch cut
    off the sheet is still identifiable without the legend."""
    polys, patches = [], []
    for i, mult in enumerate(PITCH_MULTS):
        pitch = fill_mm * mult
        x = i * (PATCH + PGAP)
        polys += serpentine(x, 0.0, PATCH, PATCH, pitch)
        patches.append({"row": "continuous", "pitch_mult": mult,
                        "pitch_mm": round(pitch, 4),
                        "rect_mm": [x, 0.0, PATCH, PATCH]})
        polys += chopped(x, _ROW1_Y, PATCH, PATCH, pitch)
        patches.append({"row": "chopped", "pitch_mult": mult,
                        "pitch_mm": round(pitch, 4), "chop_mm": CHOP_MM,
                        "rect_mm": [x, _ROW1_Y, PATCH, PATCH]})
    polys += singles(0.0, _SINGLES_Y)
    polys += [[(0.0, _TICKS_Y), (10.0, _TICKS_Y)]]
    polys += ticks(0.0, 10.0, _TICKS_Y, col_idx + 1, tick_h=TICK_H)
    rx = CELL_W - 10.0
    polys += [[(rx, _TICKS_Y), (rx + 10.0, _TICKS_Y)]]
    polys += ticks(rx, 10.0, _TICKS_Y, row_idx + 1, tick_h=TICK_H)
    return polys, {"patches": patches, "singles_n": SINGLES_N,
                   "col_ticks": col_idx + 1, "row_ticks": row_idx + 1}


def _splice_blocks(texts, labels, z_up_abs):
    """N-way splice of complete gwrite outputs into one file.

    The preamble is ANCHORED, not diffed, which is why this function differs from
    calibration_plots._splice_pressure_blocks.

    A longest-common-leading-run diff is wrong here: `segment_first` opens every path
    with `G1 Z{z_up_abs}`, and z_up_abs is identical across all cells by design, so the
    common-prefix scan would run PAST the end of document_start and swallow each body's
    opening LIFT. Emitted once at the top instead of once per body, every cell would
    begin with its travel move while the pen was still down at the previous cell's
    z_down, dragging a pen-down diagonal across each cell boundary.

    So the boundaries are located structurally instead:
      preamble ends at the FIRST `G1 Z{z_up_abs}` line -- document_start's own
        Z move is to z_travel_abs, a different height, so this cannot collide;
      postamble starts at the FIRST `M400` -- gwrite bodies never emit M400,
        only document_end does.
    Both are then verified byte-identical across every block.

    Known limitation: calibration_plots._splice_pressure_blocks uses the diffing approach
    and has the same flaw; it is not on a default path because pressure_compare
    defaults to False."""
    if len(texts) != len(labels):
        raise ValueError("texts/labels length mismatch")
    lift_line_prefix = f"G1 Z{z_up_abs:.2f}"
    split = [t.splitlines() for t in texts]

    def boundaries(lines):
        pre = next((i for i, l in enumerate(lines) if l.startswith(lift_line_prefix)), None)
        post = next((i for i, l in enumerate(lines) if l.strip() == "M400"), None)
        if pre is None or post is None or post <= pre:
            raise RuntimeError(
                "Deposition splice failed: could not locate the preamble/postamble "
                f"boundaries (first '{lift_line_prefix}' at {pre}, first 'M400' at {post}). "
                "The elegoo_cc gwrite templates must have changed.")
        return pre, post

    pre0, post0 = boundaries(split[0])
    preamble, postamble = split[0][:pre0], split[0][post0:]
    out = list(preamble)
    for lines, label in zip(split, labels):
        pre, post = boundaries(lines)
        if lines[:pre] != preamble or lines[post:] != postamble:
            raise RuntimeError(
                "Deposition splice failed: blocks disagree on preamble or postamble. "
                "Only draw_speed and z_down may vary across cells -- something that "
                "document_start/document_end DO interpolate was varied.")
        out.append(label)
        out += lines[pre:post]
    out += postamble
    return "\n".join(out) + "\n"



PENDOWN_MAX_MM = 15.0   # a cell is 29mm wide; its longest legitimate pen-down
                        # move is one 8mm patch pass. 15mm leaves headroom
                        # without admitting a cell-to-cell drag (>=30mm).


def _assert_no_pendown_travel(path, draw_z_levels):
    """Fail loudly if any pen-down move exceeds PENDOWN_MAX_MM."""
    import math
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
            f"Deposition sheet has {n} pen-DOWN move(s) longer than {PENDOWN_MAX_MM}mm "
            f"(worst {worst:.1f}mm). The pen would drag ink between cells. This is the "
            "splice defect; do not plot this file.")
    return {
        "pendown_travel_violations": 0,
        "max_pendown_mm": round(worst, 2),
        "pendown_segments_examined": n_examined,
    }


def deposition_gcode(out_gcode: Path, pen_mm=None, z_up=3.5,
                     travel_speed=8000, plunge_speed=500, lift_speed=2000,
                     offset_x=None, offset_y=None, end_policy="release",
                     quiet_fans=True, profile=None,
                     z_down_levels=Z_DOWN_LEVELS, draw_speeds=DRAW_SPEEDS):
    """Emit the full deposition sheet as ONE g-code file.

    Z comes from the active pen profile via resolve_z_touch() -- PROFILE WINS,
    raises ZTouchNotCalibrated rather than defaulting, exactly like every other
    generator in this package. The safety self-check runs against the DEEPEST
    z_down of the sweep, so the whole file is validated against its hardest
    press, not its gentlest."""
    ppmm = 1.0 / PX2MM
    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()
    if profile is None:
        profile = active_pen_profile()
    reach = pen_reach_profile(active_reach_profile(cal), profile)
    fill_mm = float(profile.get("fill_mm", 0.25))
    if pen_mm is None:
        pen_mm = float(profile.get("pen_mm", 0.5))

    z_touch, z_touch_source = resolve_z_touch(cal, profile)
    pen_off_x, pen_off_y = resolve_pen_offset(cal, profile)
    PLACEMENT_MARGIN_MM = 20.0
    offset_x = reach["bbox"][0] + PLACEMENT_MARGIN_MM if offset_x is None else float(offset_x)
    offset_y = (reach["bbox"][1] + PLACEMENT_MARGIN_MM + pen_off_y) if offset_y is None else float(offset_y)
    comp_off_x, comp_off_y = offset_x - pen_off_x, offset_y - pen_off_y
    park_x, park_y = reach["safe_park_point"]
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))

    pad = 3.0
    W = int(round((FOOTPRINT_W + 2 * pad) * ppmm))
    H = int(round((FOOTPRINT_H + 2 * pad) * ppmm))

    texts, labels, cells = [], [], []
    for r, speed in enumerate(draw_speeds):
        for c, z_down in enumerate(z_down_levels):
            polys, spec = build_deposition_cell(fill_mm, c, r)
            dx = pad + c * (CELL_W + CELL_GAP_X)
            dy = pad + r * (CELL_H + CELL_GAP_Y)
            placed = [[((x + dx) * ppmm, (y + dy) * ppmm) for x, y in p] for p in polys]
            # ONE temp svg/gcode path is reused by every cell: elegoo_cc's
            # document_start interpolates {filename}, so a per-cell temp name
            # would differ in the preamble and the splice would repeat the whole
            # preamble (and its cross-plate travel sweep) 20 times.
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
    # POST-CONDITION: no pen-down move may cross between cells. A cell's patches are
    # 8mm wide, so any pen-down move longer than PENDOWN_MAX_MM means the pen dragged
    # from one cell to the next -- the failure the anchored splice prevents. Checked
    # rather than trusted because such a drag is not obvious in the g-code text.
    _assert_no_pendown_travel(out_gcode, {round(z_touch + float(z), 2) for z in z_down_levels})
    safety = validate_calibration_gcode_safety(out_gcode, deepest)

    all_polys = []
    for r in range(len(draw_speeds)):
        for c in range(len(z_down_levels)):
            polys, _ = build_deposition_cell(fill_mm, c, r)
            dx = pad + c * (CELL_W + CELL_GAP_X)
            dy = pad + r * (CELL_H + CELL_GAP_Y)
            all_polys += [[((x + dx) * ppmm, (y + dy) * ppmm) for x, y in p] for p in polys]
    # The pattern SVG is KEPT (stable name, not a temp): /api/calibration_import
    # adds it to the canvas so the sheet's position on the plate can be checked
    # against the paper before a long plot. It is a POSITION PREVIEW ONLY: the sheet
    # must be plotted from its own g-code, not re-emitted through the composite path.
    prev_svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_paths_svg(prev_svg, W, H, float(pen_mm) * ppmm, polylines=all_polys)
    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(prev_svg))} color {PREVIEW_COLOR} '
                          f'write {shlex.quote(str(tmp_prev))}')

    return {"kind": "deposition", "pen_mm": pen_mm, "fill_mm": fill_mm,
            "footprint_mm": [round(FOOTPRINT_W + 2 * pad, 1), round(FOOTPRINT_H + 2 * pad, 1)],
            "offset_mm": [offset_x, offset_y],
            "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
            "z_down_levels": list(z_down_levels), "draw_speeds": list(draw_speeds),
            "pitch_mults": list(PITCH_MULTS), "chop_mm": CHOP_MM,
            "n_cells": len(cells), "n_patches": len(cells) * N_PITCH * 2,
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
