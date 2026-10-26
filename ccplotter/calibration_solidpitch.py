"""ccplotter/calibration_solidpitch.py -- solid-fill pitch ladder sheet (uright_gel).

What it measures: whether tightening the fill pitch toward the pen's solid core width makes
a solid area read fully dark BEFORE overdraw starts removing ink (a double pass of gel
lifting or smearing wet ink instead of adding to it). Standard uright_gel fill pitch is
~0.229 mm (= laid_mm 0.255 mm); half-pitch plots (~0.118 mm) look solid on paper. The pen's
SOLID CORE width (the dark centre of its laid stroke, independent of the fainter laid-width
tails) is ~0.18 mm. Five pitches -- 0.229 (the default), 0.20, 0.18 (~core), 0.15, 0.118
(half pitch) -- are crossed with two fill GEOMETRIES (contour rings vs zigzag boustrophedon
rows) at equal pitch, so darkness-vs-pitch and geometry-vs-geometry are both readable off
one sheet.

How to read it: photograph or scan the sheet, measure the mean darkness of each square, and
normalise with the white-reference square. Plot darkness against pitch for each geometry.
Check the controls first (see INSTRUMENT-FAILURE RULE): if the 0.85 mm control does not
read open, the capture chain failed to resolve pitch and no ladder conclusion may be drawn.

HARD-CODED MACHINE AND PEN (limitation). The sheet is tied to the author's measured machine
and pen: it refuses to run unless the active reach profile is EXPECT_REACH_ID
("plotter_2003_v32") with bbox EXPECT_BBOX. The pen is PINNED: the pitch ladder and
predictions are calibrated numbers specific to uright_gel's own wedge and core-width
reading, so the generator IGNORES the dispatcher's active pen and always re-resolves
EXPECT_PEN_ID ("uright_gel") directly out of `pen_profiles.json`. If `profile` is passed and
names a different pen, it refuses loudly rather than silently emitting uright_gel's g-code
under another pen's name and Z. `profile=None` always generates the pinned sheet.
label / pen_mm / pen_offset / z_down / draw_speed are hard-asserted against fixed constants.
z_touch_abs is NOT a pinned literal, because it changes whenever the pen is recalibrated;
it is asserted FINITE and inside a plausible (5,20) mm range, and the resolved value and its
source are returned in the info dict and written to the g-code header exactly as read.

GEOMETRY IS BUILT DIRECTLY, NOT TRACED. Every square is emitted straight from closed-form
ring / row formulas (`_contour_rings`, `_zigzag_rows`) -- no `trace_v3._trace_svg` call, no
Otsu / hole-floor / dt-band machinery. The sheet needs an EXACT pitch per square; going
through the trace pipeline would reintroduce the resolution and rounding uncertainty a
solid-square patch is meant to remove.

SIZE / REALIZED PITCH, EXACT BY CONSTRUCTION. For pitch p, let m = ceil(15.0 mm / p) (the
smallest integer >= 15/p, with a small epsilon against float division noise; see
`_ring_count` / `_square_geom`). The realized square side is S = m * p (>= 15.0 mm, never
padded down). CONTOUR rings sit at inset k*p from every edge, k = 0 .. K-1, K = ceil(m/2)
(the k where the ring degenerates to zero is excluded) -- closed, axis-aligned squares,
outermost ring's own outer edge exactly S. ZIGZAG rows sit at y = i*p, i = 0 .. m (m+1 rows,
edge rows landing exactly on y=0 and y=S), one continuous boustrophedon polyline per square.
Both modes share the same realized outer footprint S at a given pitch, so a
contour-vs-zigzag comparison at equal pitch is not confounded by size. A leftover BARE
CENTRE smaller than one pitch is inherent to the contour construction; it is recorded in
`layout` per square (`contour_center_gap_mm`), not padded away.

ITEMS: 16 filled squares plus a white reference (17 real grid items):
  * 5 pitches x 2 geometries = 10 squares;
  * a NULL PAIR: 0.18 mm contour at two DIFFERENT grid positions, so any darkness spread
    between them upper-bounds scan noise / vignetting rather than a pitch effect;
  * a DEPOSITION CONTROL: 0.118 mm zigzag drawn TWICE at the SAME position as two
    back-to-back pen-down cycles over identical geometry -- the direct test of "does overdraw
    remove ink instead of adding it";
  * a POSITIVE CONTROL LADDER of three over-wide pitches (0.35, 0.50, 0.85 mm), CONTOUR only
    (they exist to prove the instrument can see a pitch effect, not to re-run the geometry
    comparison), built through the same `_contour_rings` / `_square_geom` path as every other
    square;
  * a thin-border-only white-paper reference square (empty interior) for scan white-level
    normalization.

POSITIVE CONTROL: GAP ARITHMETIC. A null result across the five tight pitches is
uninterpretable unless something on the sheet is EXPECTED to look banded. For ring pitch p
and round pen laid width w (`EXPECT_PEN_MM`, 0.255 mm): the STRAIGHT gap between two adjacent
parallel ring edges is `straight_gap = p - w`. The CORNER of a square ring is where a round
pen under-fills worst: the gap along the diagonal at a ring's corner is
`corner_gap = p*sqrt(2) - (w/2)*sqrt(2) - w/2` (`_control_gap_mm`), roughly DOUBLE the
straight gap, because two ring edges meeting at a right angle leave a wedge a round pen
cannot reach.

Reading a gap as a prediction uses an on-paper threshold stated in GAP terms, not pitch: a
gap below ~0.25 mm does not survive on paper; a gap above ~0.50 mm reads as open, legible
detail; between the two is a grey-blur regime. These are empirical thresholds (what a gap of
a given size survives as through ink, paper and capture), not geometric identities. Applied
to `corner_gap` (the worst case within a ring):
  0.229 mm: straight -0.026 mm (overlap), corner ~0.02 mm, ~0.03 % bare -> SOLID (the
            default; a sanity floor, not a control)
  0.35 mm:  straight 0.095 mm, corner ~0.187 mm (both < 0.25 mm), ~27.3 % bare -> SOLID
            (the straight gap runs the full perimeter, so it is the bulk of the bare area)
  0.50 mm:  straight 0.245 mm, corner ~0.399 mm -- both in the blur regime -> TRANSITIONAL:
            a visible ring pattern PLUS a separate ~1 mm bare centre
            (`contour_center_gap_mm`), two distinct facts
  0.85 mm:  straight 0.595 mm, corner ~0.894 mm -- both above 0.50 mm, ~70 % bare
            (the ring-to-ring centreline annulus frame, (p-w)/p = 0.700) -> OPEN, the one
            pitch this sheet expects to read as legible, separated rings, unambiguously
            NOT solid. A whole-square frame (including the centre region) gives ~0.68; both
            clear the instrument-failure threshold by more than 2x, so no prediction depends
            on the difference.
Exact `n_rings` / `side_mm` / `contour_center_gap_mm` per control pitch are in `_square_geom`
and the emitted `layout.squares` records, read back from the run's geometry.

INSTRUMENT-FAILURE RULE: 0.85 mm is the one pitch predicted OPEN without qualification. If
it does NOT read visibly open / separated on a scan, the measurement chain (scan / photo /
exposure / resolution) -- not the pen or the fill pitch -- has failed to resolve pitch, and
NO conclusion about the 0.229-0.118 mm ladder (rises-to-plateau, overdraw-falls,
contour-vs-zigzag) may be drawn from that run. Evaluate this BEFORE any plateau verdict.
0.35 and 0.50 mm sanity-check the transition but do not gate instrument validity.

PLACEMENT: RANDOMIZED, RECORDED, NOT CONFOUNDED WITH PLOT ORDER (the idiom of
`calibration_linestar.py`, simplified to a flat grid since there is no angle axis to protect).
The 17 real grid items are assigned to 17 of the GRID_COLS x GRID_ROWS = 18 cells with one
`random.Random(SEED_PLACEMENT)` permutation, and a SEPARATE `random.Random(SEED_ORDER)`
permutation decides EMISSION order, so plate position and print order are independent draws.
The deposition control's two passes stay ADJACENT in emission order (one dict entry expands
to two identical-geometry elements back to back): the question is literally what a second
pass immediately following the first does to the same ink.

18 CELLS, 17 ITEMS, ONE EMPTY. GRID_ROWS is 2 and 17 items do not divide evenly, so
GRID_COLS=9 (18 cells) is the smallest rectangle that holds them. `_assign_grid` shuffles the
full 18-cell list and zips it with the 17 specs, so the unused cell is also randomized and
recorded in the returned `layout` dict (`grid.unused_cells`), so it is not mistaken for a
missing square.

PITCH REALIZED FROM THE EMITTED GCODE, NOT ASSUMED. For contour squares the check reads back
adjacent rings' half-side difference; for zigzag squares, adjacent rows' perpendicular
spacing. Both are asserted within `PITCH_REALIZATION_TOL_MM` (0.01 mm, one order of magnitude
above `gwrite`'s own %.3f coordinate rounding; see `vpype.toml`'s `segment` / `segment_first`
format strings) of the commanded pitch. A mismatch means a translate / gwrite rounding or
geometry bug, not the physical question the sheet asks.

PLOT-TIME BUDGET (model (a), same as `calibration_linestar.py` / `calibration_fillmode.py`:
draw time at `draw_speed` + a fixed per-pen-down-cycle lift budget, NO travel-move time;
registry sheets have no microlift, so this is a floor, not a forecast). See
`plot_time_estimate_min` in the returned info dict for the figure computed against the run's
realized geometry; the design target is about 20 minutes with squares at the >=15 mm floor.

PREDICTION (written in the g-code header, see `_PREDICTION_LINES`):
  - Darkness rises as pitch tightens toward the pen's ~0.18 mm solid core, plateauing near
    pitch ~ core (0.17-0.18 mm).
  - Below ~0.15 mm pitch, darkness may FALL as overdraw removes / redistributes gel ink; the
    deposition control (0.118 mm zigzag, drawn twice) is predicted to read LIGHTER than the
    single-pass 0.118 mm zigzag square if that mechanism is present.
  - Contour and zigzag are compared at EQUAL pitch; no directional prediction is made between
    them (that comparison was run at a single pitch in `calibration_fillmode.py`; this sheet
    covers the pitch axis).

The measurement script that scores these squares is a separate tool and is not part of this
repository.

Reached only through the generic POST /api/calibration_sheet dispatcher (id="solidpitch" in
`calibration_registry.py`).
"""
from __future__ import annotations

import math
import random
import re
import shlex
import textwrap
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
# convention as every sibling generic sheet). Tied to the author's measured
# machine geometry.
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
PITCHES_MM = (0.229, 0.20, 0.18, 0.15, 0.118)
GEOMETRIES = ("contour", "zigzag")
MIN_SIDE_MM = 15.0
NULL_PITCH_MM = 0.18
NULL_MODE = "contour"
DEPOSITION_PITCH_MM = 0.118
DEPOSITION_MODE = "zigzag"
# Positive-control pitches -- see module docstring "POSITIVE CONTROL: GAP
# ARITHMETIC". CONTOUR only, deliberately over-wide so darkness banding is
# expected, not a pitch-ladder question.
CONTROL_PITCHES_MM = (0.35, 0.50, 0.85)
CONTROL_MODE = "contour"
# Predicted reading per control pitch (module docstring); 0.85mm is the ONE
# unqualified prediction -- see "INSTRUMENT-FAILURE RULE".
CONTROL_PREDICTED_READING = {0.35: "SOLID", 0.50: "TRANSITIONAL", 0.85: "OPEN"}
# Predicted bare-area fraction of each control square's own p x p ring cell
# (round-pen buffer union against a single ring, shapely): 0.0004, 0.2736, 0.4911,
# 0.7004 for 0.229/0.35/0.50/0.85mm, from the ring-to-ring annulus frame, which is
# the same construction at every pitch. A whole-square frame (includes the centre
# region, excludes the outer ring's outboard half-width) gives ~0.68 at 0.85mm;
# both clear the 0.6 gate threshold below by >2x, so no prediction depends on the
# difference.
CONTROL_PREDICTED_BARE_FRAC = {0.35: 0.273, 0.50: 0.491, 0.85: 0.70}
# Instrument-failure gate: 0.85mm is predicted ~70% bare, i.e. roughly 30% as dark
# as a fully solid square. A gate at 0.6x the ladder's own darkest reading gives
# ~2x margin against that prediction -- if 0.85mm's square is not at least this
# much lighter than the ladder AND distinguishable from the null pair's own noise
# floor, the scan/photo instrument -- not the pen -- failed to resolve pitch.
CONTROL_OPEN_MAX_DARKNESS_FRAC = 0.6
CLEARANCE_MM = 4.0
PITCH_REALIZATION_TOL_MM = 0.01

SEED_PLACEMENT = 20260915
SEED_ORDER = 20260916

GRID_COLS, GRID_ROWS = 9, 2      # 18 cells for 17 real items (16 filled + 1
                                  # white-ref) -- ONE cell deliberately left
                                  # empty, see docstring "18 CELLS, 17 ITEMS"
CELL_PITCH_MM = 20.0             # >= max realized side (~15.3mm, 0.85mm
                                  # control) + CLEARANCE_MM
GRID_W = GRID_COLS * CELL_PITCH_MM
GRID_H = GRID_ROWS * CELL_PITCH_MM

BAND_GAP_MM = 6.0
FID_TICK_PITCH_MM = 10.0
FID0_LEN_MM = 100.0
FID90_LEN_MM = 60.0
TICK_H_MM = 4.0

GLYPH_H_MM = 8.0
GLYPH_W_MM = 6.0

WHITE_REF_SIDE_MM = 15.0

PREDICTION = {
    "rises_to_plateau": "darkness rises as pitch tightens toward the pen's "
                         "~0.18mm solid core, plateauing near pitch ~ core "
                         "(0.17-0.18mm)",
    "overdraw_falls": "below ~0.15mm pitch, darkness may FALL as overdraw "
                       "removes/redistributes gel ink; the deposition "
                       "control (0.118mm zigzag x2 passes) is predicted to "
                       "read LIGHTER than the single-pass 0.118mm zigzag "
                       "square if this mechanism is present",
    "contour_vs_zigzag": "compared at EQUAL pitch; no directional prediction "
                          "is pre-registered between geometries here",
    "positive_control": "GAP ARITHMETIC (D451, replaces a rejected pitch-only "
                         "S7 read): straight_gap = pitch - laid(0.255mm); "
                         "corner_gap = pitch*sqrt(2) - (laid/2)*sqrt(2) - laid/2 "
                         "(a round pen under-fills a square ring's corner "
                         "roughly 2x the straight gap, worst case within a "
                         "ring). DIRECTION S7's OWN thresholds are stated in "
                         "gap terms, not pitch: a gap below ~0.25mm does not "
                         "survive on paper, a gap above ~0.50mm reads as open "
                         "legible detail, between is a grey-blur regime -- "
                         "applied to corner_gap (worst case): 0.35mm -> SOLID "
                         "(corner_gap ~0.187mm < 0.25mm, ~27.3% bare); 0.50mm "
                         "-> TRANSITIONAL (corner_gap ~0.399mm in the blur "
                         "regime, ~49.1% bare, PLUS a separate ~1mm bare "
                         "centre -- two facts, not one, both making it "
                         "lighter than 0.35mm while still surviving as a "
                         "visible ring pattern); 0.85mm -> OPEN (corner_gap "
                         "~0.894mm > 0.50mm, ~0.70 (70%) bare (canonical, "
                         "D463), unambiguously "
                         "separated rings) -- the ONE "
                         "unqualified prediction on this sheet. "
                         "INSTRUMENT-FAILURE RULE: if 0.85mm does not read "
                         "open, the scan/photo instrument -- not the pen -- "
                         "failed to resolve pitch this run, and no ladder "
                         "conclusion (rises-to-plateau, overdraw-falls, "
                         "contour-vs-zigzag, INCLUDING a plateau report) may "
                         "be drawn from it. Evaluated BEFORE any plateau "
                         "verdict; measure_solidpitch.py WITHHOLDS the "
                         "plateau verdict, not just annotates it, when this "
                         "gate reports instrument_failed.",
}
# The prediction text is written into the g-code header as comment lines. Every
# entry is wrapped through the SAME helper (not just the longest one), so no
# single entry gets a tacit exemption from the line-length rule.
#
# WORKING RULE, NOT A FIRMWARE SPEC: g-code files already sent to the printer top
# out at 112-121 chars per line, and the firmware parser has crashed on a single
# bad character (a non-ASCII em-dash), so `_PREDICTION_LINE_WRAP_WIDTH` below
# stays under that observed-safe range. It is NOT a documented firmware
# line-length limit (none is known); it is a margin against an unknown one.
_PREDICTION_LINE_WRAP_WIDTH = 110

# Each entry is ONE logical prediction statement; `_wrap_prediction_lines`
# below wraps every one through the same width.
_RAW_PREDICTION_ENTRIES = [
    "SOLIDPITCH PREDICTION (pre-registered 2026-09-15, uright_gel, D413/D414; "
    "positive-control REVISED 2026-09-15 D451):",
    f"rises-to-plateau -> {PREDICTION['rises_to_plateau']}",
    f"overdraw-falls -> {PREDICTION['overdraw_falls']}",
    f"contour-vs-zigzag -> {PREDICTION['contour_vs_zigzag']}",
    f"positive-control -> {PREDICTION['positive_control']}",
    "Null pair (0.18mm contour x2 positions) upper-bounds scan noise/vignetting; "
    "deposition control isolates overdraw from pitch.",
    "Per-square id/pitch/mode/grid position/order is in the response 'layout' "
    "field and the sibling *_layout.json file, not printed on the sheet -- see "
    "artifacts/solidpitch-sheet-2026-09-15/measure_solidpitch.py",
]


def _wrap_prediction_lines(entries, width=_PREDICTION_LINE_WRAP_WIDTH):
    """Wraps every entry through the SAME `textwrap.wrap` call --
    continuation lines get a 2-space indent so they are visually
    distinguishable from a new logical entry in the raw gcode comments."""
    out = []
    for entry in entries:
        wrapped = textwrap.wrap(entry, width=width) or [""]
        out.append(wrapped[0])
        out.extend(f"  {ln}" for ln in wrapped[1:])
    return out


_PREDICTION_LINES = _wrap_prediction_lines(_RAW_PREDICTION_ENTRIES)


# ---------------------------------------------------------------------------
# Geometry builders (design space, mm; local square coords with (0,0) at the
# top-left corner, y grows down -- same convention every sibling module uses
# before the vpype translate/mirror).
# ---------------------------------------------------------------------------
def _square_m(pitch_mm: float) -> int:
    """Smallest integer m such that m*pitch >= MIN_SIDE_MM (a small epsilon
    guards against float-division noise landing exactly on an integer,
    e.g. 15.0/0.15 == 100.0 but could evaluate as 99.999999999994)."""
    return int(math.ceil(round(MIN_SIDE_MM / pitch_mm, 9) - 1e-9))


def _square_geom(pitch_mm: float):
    """Returns (side_mm, n_rows, n_rings, center_gap_mm) for one pitch --
    shared by both geometries so contour and zigzag realize the exact same
    outer footprint at equal pitch (module docstring)."""
    m = _square_m(pitch_mm)
    side = m * pitch_mm
    n_rows = m + 1
    n_rings = math.ceil(m / 2)
    center_gap = side - 2 * (n_rings - 1) * pitch_mm - pitch_mm if n_rings > 0 else side
    # ring k=0..n_rings-1 sits at inset k*pitch; the innermost drawn ring has
    # half-side (side/2 - (n_rings-1)*pitch); the leftover bare centre is
    # twice that, floored at 0 (a ring landing exactly on the centre has
    # zero leftover, not negative).
    center_gap = max(0.0, side - 2 * (n_rings - 1) * pitch_mm)
    return side, n_rows, n_rings, center_gap


def _control_gap_mm(pitch_mm: float, pen_mm: float = EXPECT_PEN_MM):
    """Gap arithmetic for a round pen tracing concentric square rings
    (module docstring "POSITIVE CONTROL: GAP ARITHMETIC"). `straight_gap` is the
    uncovered distance between two adjacent parallel ring edges;
    `corner_gap` is the uncovered distance along the diagonal at a ring's
    right-angle corner, where a round pen under-fills worst -- roughly
    DOUBLE the straight gap. Returns (straight_gap_mm, corner_gap_mm)."""
    straight_gap = pitch_mm - pen_mm
    corner_gap = (pitch_mm * math.sqrt(2.0) - (pen_mm / 2.0) * math.sqrt(2.0)
                  - pen_mm / 2.0)
    return straight_gap, corner_gap


def _contour_rings(pitch_mm: float, side_mm: float, n_rings: int):
    """Closed, axis-aligned concentric rings, outer ring's own outer edge at
    exactly `side_mm`. Returns a list of closed 5-point polylines, outermost
    first."""
    rings = []
    for k in range(n_rings):
        lo = k * pitch_mm
        hi = side_mm - k * pitch_mm
        rings.append([(lo, lo), (hi, lo), (hi, hi), (lo, hi), (lo, lo)])
    return rings


def _zigzag_rows(pitch_mm: float, side_mm: float, n_rows: int):
    """ONE continuous boustrophedon polyline: rows at y = i*pitch,
    i = 0..n_rows-1, alternating direction, no separate pen-lift between
    rows (module docstring)."""
    pts = []
    for i in range(n_rows):
        y = i * pitch_mm
        if i % 2 == 0:
            pts.append((0.0, y))
            pts.append((side_mm, y))
        else:
            pts.append((side_mm, y))
            pts.append((0.0, y))
    return [pts]


def _poly_length_mm(pts):
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1))


def _poly_bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(xs) - min(xs), max(ys) - min(ys)


def _rect_outline(x0, y0, w, h):
    return [[(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h), (x0, y0)]]


def _ticks_along(x0, y0, length, orientation, pitch=FID_TICK_PITCH_MM, tick_h=TICK_H_MM):
    """Perpendicular tick marks at an exact mm PITCH along a baseline, for a
    scan's per-axis scale self-calibration. `orientation` in {"h","v"}."""
    out = []
    n = int(round(length / pitch))
    for k in range(n + 1):
        d = k * pitch
        if d > length + 1e-9:
            break
        if orientation == "h":
            out.append([(x0 + d, y0), (x0 + d, y0 + tick_h)])
        else:
            out.append([(x0, y0 + d), (x0 + tick_h, y0 + d)])
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


def _glyph(x0, y0):
    """Asymmetric orientation marker: 3 strokes forming an 'F' -- deliberately
    not symmetric under any flip/rotation, same idiom as
    calibration_linestar.py's `_f_glyph` (independently implemented here,
    module self-containment convention)."""
    spine = [(x0, y0), (x0, y0 + GLYPH_H_MM)]
    top = [(x0, y0), (x0 + GLYPH_W_MM, y0)]
    mid = [(x0, y0 + GLYPH_H_MM * 0.5), (x0 + GLYPH_W_MM * 0.55, y0 + GLYPH_H_MM * 0.5)]
    return [spine, top, mid]


def _build_square_specs():
    """The 16 filled squares (13 ladder/null/deposition + 3 positive-control)
    + 1 white-ref item = 17 real grid items, as design-time specs
    (before grid placement). Each dict: id, role, pitch_mm, mode, n_passes
    (1 or 2 for the deposition control)."""
    specs = []
    for p in PITCHES_MM:
        for mode in GEOMETRIES:
            specs.append({"id": f"pitch_{p:g}_{mode}", "role": "pitch",
                          "pitch_mm": p, "mode": mode, "n_passes": 1})
    specs.append({"id": "null_A", "role": "null", "pitch_mm": NULL_PITCH_MM,
                  "mode": NULL_MODE, "n_passes": 1})
    specs.append({"id": "null_B", "role": "null", "pitch_mm": NULL_PITCH_MM,
                  "mode": NULL_MODE, "n_passes": 1})
    specs.append({"id": "deposition", "role": "deposition",
                  "pitch_mm": DEPOSITION_PITCH_MM, "mode": DEPOSITION_MODE,
                  "n_passes": 2})
    # Positive-control LADDER (module docstring "POSITIVE CONTROL: GAP
    # ARITHMETIC") -- deliberately over-wide, CONTOUR-only, no crossing with
    # zigzag: these exist to prove the instrument CAN see a pitch effect, not to
    # re-run the geometry comparison. Each spec also carries its predicted
    # reading/bare-fraction and gap arithmetic so the layout JSON is
    # self-describing without recomputing from constants.
    for p in CONTROL_PITCHES_MM:
        straight_gap, corner_gap = _control_gap_mm(p)
        specs.append({"id": f"control_{p:g}_{CONTROL_MODE}", "role": "control",
                      "pitch_mm": p, "mode": CONTROL_MODE, "n_passes": 1,
                      "predicted_reading": CONTROL_PREDICTED_READING[p],
                      "predicted_bare_frac": CONTROL_PREDICTED_BARE_FRAC[p],
                      "straight_gap_mm": round(straight_gap, 4),
                      "corner_gap_mm": round(corner_gap, 4)})
    specs.append({"id": "white_ref", "role": "white_ref",
                  "pitch_mm": None, "mode": None, "n_passes": 1})
    assert len(specs) == 17, f"expected 17 real grid items, got {len(specs)}"
    return specs


def _assign_grid(specs, seed=SEED_PLACEMENT):
    """Randomizes GRID-CELL assignment with a fixed seed -- plate position is
    never confounded with pitch/geometry. GRID_COLS x GRID_ROWS may exceed
    len(specs) by a small amount (18 cells for 17 real items) -- `zip` below
    only consumes as many shuffled cells as there are specs, so any excess cells
    are simply never assigned; the caller (`_build_elements`) records which
    cell(s) were left unused."""
    cells = [(c, r) for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    assert len(cells) >= len(specs), (
        f"solidpitch: grid has {len(cells)} cells, fewer than the "
        f"{len(specs)} real grid items -- widen GRID_COLS/GRID_ROWS.")
    rng = random.Random(seed)
    shuffled = list(cells)
    rng.shuffle(shuffled)
    used = shuffled[:len(specs)]
    unused = shuffled[len(specs):]
    out = []
    for spec, (col, row) in zip(specs, used):
        s = dict(spec)
        s["grid_col"], s["grid_row"] = col, row
        out.append(s)
    return out, [[c, r] for c, r in unused]


def _assign_order(specs, seed=SEED_ORDER):
    """Randomizes EMISSION order with a SEPARATE fixed seed -- independent of
    grid-cell placement (module docstring). Returns a NEW list in emission order,
    each spec carrying `order_idx`."""
    idxs = list(range(len(specs)))
    random.Random(seed).shuffle(idxs)
    out = []
    for order_idx, i in enumerate(idxs):
        s = dict(specs[i])
        s["order_idx"] = order_idx
        out.append(s)
    return out


def _build_elements():
    """Assembles every design-space element (16 filled squares' rings/rows,
    the white-ref outline, fiducials, glyph) and a `groups` dict for the
    clearance check. Returns (elements, square_records, fid_records, groups,
    footprint_w, footprint_h, unused_cells)."""
    specs = _build_square_specs()
    specs, unused_cells = _assign_grid(specs)
    specs = _assign_order(specs)
    specs.sort(key=lambda s: s["order_idx"])

    elements = []
    square_records = []
    groups = {}
    for spec in specs:
        cx = spec["grid_col"] * CELL_PITCH_MM
        cy = spec["grid_row"] * CELL_PITCH_MM

        if spec["role"] == "white_ref":
            side = WHITE_REF_SIDE_MM
            ox = cx + (CELL_PITCH_MM - side) / 2.0
            oy = cy + (CELL_PITCH_MM - side) / 2.0
            polys = _rect_outline(ox, oy, side, side)
            group_polys = []
            for pts in polys:
                shifted = [(x, y) for x, y in pts]
                elements.append({"pts": shifted, "kind": "white_ref",
                                  "square_id": spec["id"], "order_idx": spec["order_idx"]})
                group_polys.append(shifted)
            groups[spec["id"]] = group_polys
            square_records.append({
                "id": spec["id"], "role": spec["role"], "pitch_mm": None, "mode": None,
                "grid_col": spec["grid_col"], "grid_row": spec["grid_row"],
                "order_idx": spec["order_idx"], "side_mm": side,
                "origin_mm": [round(ox, 3), round(oy, 3)], "n_pendowns": 1,
            })
            continue

        pitch = spec["pitch_mm"]
        side, n_rows, n_rings, center_gap = _square_geom(pitch)
        ox = cx + (CELL_PITCH_MM - side) / 2.0
        oy = cy + (CELL_PITCH_MM - side) / 2.0

        if spec["mode"] == "contour":
            polys = _contour_rings(pitch, side, n_rings)
            n_pendowns_per_pass = n_rings
        else:
            polys = _zigzag_rows(pitch, side, n_rows)
            n_pendowns_per_pass = 1

        group_polys = []
        elem_start = len(elements)
        for pass_idx in range(spec["n_passes"]):
            for pts in polys:
                shifted = [(x + ox, y + oy) for x, y in pts]
                elements.append({"pts": shifted, "kind": "square_fill",
                                  "square_id": spec["id"], "role": spec["role"],
                                  "pitch_mm": pitch, "mode": spec["mode"],
                                  "pass_idx": pass_idx, "order_idx": spec["order_idx"]})
                group_polys.append(shifted)
        elem_end = len(elements)
        groups[spec["id"]] = group_polys

        rec = {
            "id": spec["id"], "role": spec["role"], "pitch_mm": pitch, "mode": spec["mode"],
            "grid_col": spec["grid_col"], "grid_row": spec["grid_row"],
            "order_idx": spec["order_idx"], "side_mm": round(side, 4),
            "n_passes": spec["n_passes"], "n_pendowns_per_pass": n_pendowns_per_pass,
            "n_pendowns": n_pendowns_per_pass * spec["n_passes"],
            "n_rows_or_rings": n_rings if spec["mode"] == "contour" else n_rows,
            "contour_center_gap_mm": round(center_gap, 4) if spec["mode"] == "contour" else None,
            "origin_mm": [round(ox, 3), round(oy, 3)],
            "elem_range": [elem_start, elem_end],
        }
        if spec["role"] == "control":
            # Gap arithmetic + predicted reading, carried through from the spec
            # (module docstring "POSITIVE CONTROL: GAP ARITHMETIC") -- read back
            # here, not recomputed, so the layout JSON matches the prediction
            # written to the g-code header.
            rec["predicted_reading"] = spec["predicted_reading"]
            rec["predicted_bare_frac"] = spec["predicted_bare_frac"]
            rec["straight_gap_mm"] = spec["straight_gap_mm"]
            rec["corner_gap_mm"] = spec["corner_gap_mm"]
        square_records.append(rec)

    # F glyph: fixed corner above the grid.
    glyph_x0, glyph_y0 = 0.0, -(GLYPH_H_MM + BAND_GAP_MM)
    glyph_strokes = _glyph(glyph_x0, glyph_y0)
    for pts in glyph_strokes:
        elements.append({"pts": pts, "kind": "glyph"})
    groups["glyph"] = glyph_strokes

    # 0deg fiducial: below the grid.
    fid0_x0, fid0_y0 = 0.0, GRID_H + BAND_GAP_MM
    fid0_strokes, fid0_rec = _fiducial_0(fid0_x0, fid0_y0)
    for pts in fid0_strokes:
        elements.append({"pts": pts, "kind": "fiducial", "fid_axis_deg": 0.0})
    groups["fid_0"] = fid0_strokes

    # 90deg fiducial: right of the grid.
    fid90_x0, fid90_y0 = GRID_W + BAND_GAP_MM, 0.0
    fid90_strokes, fid90_rec = _fiducial_90(fid90_x0, fid90_y0)
    for pts in fid90_strokes:
        elements.append({"pts": pts, "kind": "fiducial", "fid_axis_deg": 90.0})
    groups["fid_90"] = fid90_strokes

    xs = [x for e in elements for x, _y in e["pts"]]
    ys = [y for e in elements for _x, y in e["pts"]]
    minx, miny = min(xs), min(ys)
    elements = [{**e, "pts": [(x - minx, y - miny) for x, y in e["pts"]]} for e in elements]
    footprint_w = max(xs) - minx
    footprint_h = max(ys) - miny

    if footprint_w + 2 * _PAD_MM > _USABLE_W:
        raise RuntimeError(f"solidpitch: footprint width {footprint_w + 2 * _PAD_MM:.1f}mm "
                           f"exceeds the JIS B5 landscape usable width {_USABLE_W:.1f}mm")
    if footprint_h + 2 * _PAD_MM > _USABLE_H:
        raise RuntimeError(f"solidpitch: footprint height {footprint_h + 2 * _PAD_MM:.1f}mm "
                           f"exceeds the JIS B5 landscape usable height {_USABLE_H:.1f}mm")

    def _shift(pts_list):
        return [[(x - minx, y - miny) for x, y in pts] for pts in pts_list]

    shifted_groups = {name: _shift(pts_list) for name, pts_list in groups.items()}

    for rec in square_records:
        rec["origin_mm"] = [round(rec["origin_mm"][0] - minx, 3),
                             round(rec["origin_mm"][1] - miny, 3)]

    # The fiducial records' origin_mm are shifted here by the same offset as
    # square_records above. The layout JSON must report each fiducial's (minx,
    # miny) where it is ACTUALLY plotted: a measurement script trusting stale
    # coordinates would search for the fiducial in the wrong place on a real scan.
    for rec in (fid0_rec, fid90_rec):
        rec["origin_mm"] = [round(rec["origin_mm"][0] - minx, 3),
                             round(rec["origin_mm"][1] - miny, 3)]

    fid_records = [fid0_rec, fid90_rec]
    return (elements, square_records, fid_records, shifted_groups, footprint_w,
            footprint_h, unused_cells)


def _group_geom(pts_list):
    lines = [LineString(p) for p in pts_list if len(p) >= 2]
    return lines[0] if len(lines) == 1 else MultiLineString(lines)


def _check_clearance(groups, min_mm=CLEARANCE_MM):
    """Pairwise minimum distance between every isolated GROUP (each square's
    own rings/rows count as ONE group; the >=4mm clearance rule is enforced
    BETWEEN groups, not within a square's own pitch -- same scoping idiom as
    `calibration_linestar.py`'s `_check_clearance`)."""
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
            f"solidpitch: {len(violations)} element-group pair(s) violate the {min_mm}mm "
            f"clearance rule: {violations[:8]}. Do not plot this file.")
    return {"n_groups": len(names), "min_clearance_mm": min_mm,
            "n_pairs_checked": len(names) * (len(names) - 1) // 2}


# ---------------------------------------------------------------------------
# Post-emission validation: isometry-invariant length+bbox match per element
# (catches cross-element drags), realized-pitch readback per square, and a
# non-zero examined-segment guard so the checks cannot pass vacuously -- same
# idiom as every sibling generic sheet.
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


def _pendown_check(out_gcode, z_down_abs, elements, n_pendowns_expected):
    strokes = _parse_strokes(out_gcode, z_down_abs)
    n_pendown_segments = sum(len(s["pts"]) - 1 for s in strokes)
    if n_pendown_segments <= 0:
        raise RuntimeError(
            "solidpitch: pendown_segments_examined is ZERO -- the guard never ran "
            "(a vacuous pass). Do not plot this file.")
    if len(strokes) != n_pendowns_expected:
        raise RuntimeError(
            f"solidpitch: stroke count mismatch: emitted {len(strokes)} pen-down cycles, "
            f"expected {n_pendowns_expected} (one per design element). Do not plot this file.")

    length_mismatches, bbox_mismatches = [], []
    for i, (elem, stroke) in enumerate(zip(elements, strokes)):
        expected_len = _poly_length_mm(elem["pts"])
        actual_len = _poly_length_mm(stroke["pts"])
        if abs(actual_len - expected_len) > max(0.05, 0.01 * expected_len):
            length_mismatches.append({"elem_idx": i, "kind": elem["kind"],
                                       "square_id": elem.get("square_id"),
                                       "expected_mm": round(expected_len, 3),
                                       "actual_mm": round(actual_len, 3)})
        ew, eh = _poly_bbox(elem["pts"])
        aw, ah = _poly_bbox(stroke["pts"])
        tol = max(0.05, 0.01 * max(ew, eh))
        if abs(aw - ew) > tol or abs(ah - eh) > tol:
            bbox_mismatches.append({"elem_idx": i, "kind": elem["kind"],
                                     "square_id": elem.get("square_id")})
    if length_mismatches:
        raise RuntimeError(
            f"solidpitch: {len(length_mismatches)} element(s) have machine-length != "
            f"design-length beyond tolerance: {length_mismatches[:5]}. Do not plot this file.")
    if bbox_mismatches:
        raise RuntimeError(
            f"solidpitch: {len(bbox_mismatches)} element(s) have a machine bbox that does not "
            f"match their design bbox -- cross-element drag guard firing: "
            f"{bbox_mismatches[:5]}. Do not plot this file.")

    return {
        "pendown_segments_examined": n_pendown_segments,
        "n_strokes": len(strokes),
        "n_pendowns_expected": n_pendowns_expected,
    }, strokes


def _pitch_realization(square_records, strokes):
    """Reads REALIZED pitch back out of the emitted gcode strokes -- adjacent
    ring half-side difference for contour, adjacent row perpendicular
    spacing for zigzag (same idea as `calibration_fillmode.py`'s
    `_pitch_realization_zigzag`). Asserts within PITCH_REALIZATION_TOL_MM."""
    out = {}
    worst_dev = 0.0
    for rec in square_records:
        if rec["role"] == "white_ref":
            continue
        a, b = rec["elem_range"]
        n_per_pass = rec["n_pendowns_per_pass"]
        pass0_strokes = strokes[a:a + n_per_pass]
        if rec["mode"] == "contour":
            if len(pass0_strokes) < 2:
                continue
            sides = []
            for s in pass0_strokes:
                w, h = _poly_bbox(s["pts"])
                sides.append((w + h) / 2.0)
            gaps = [abs(sides[i] - sides[i + 1]) / 2.0 for i in range(len(sides) - 1)]
        else:
            if len(pass0_strokes) != 1 or len(pass0_strokes[0]["pts"]) < 4:
                continue
            pts = pass0_strokes[0]["pts"]
            ys = sorted({round(p[1], 4) for p in pts})
            gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        if not gaps:
            continue
        realized = sum(gaps) / len(gaps)
        dev = abs(realized - rec["pitch_mm"])
        worst_dev = max(worst_dev, dev)
        out[rec["id"]] = {"commanded_pitch_mm": rec["pitch_mm"],
                          "realized_pitch_mm": round(realized, 5),
                          "deviation_mm": round(dev, 5),
                          "n_gaps_measured": len(gaps)}
    if worst_dev > PITCH_REALIZATION_TOL_MM:
        raise RuntimeError(
            f"solidpitch: worst realized-pitch deviation {worst_dev:.4f}mm exceeds "
            f"{PITCH_REALIZATION_TOL_MM}mm tolerance -- gwrite rounding or a geometry bug, "
            f"not the physical question this sheet asks. Do not plot this file.")
    return out


# ---------------------------------------------------------------------------
# Emission -- single Z / single draw speed (pitch/geometry are the only
# variables), one gwrite call, same pattern as every sibling generic sheet.
# ---------------------------------------------------------------------------
def _pen_id_from_profile(profile: dict, profiles_doc: dict):
    matches = [pid for pid, p in profiles_doc.get("profiles", {}).items() if p == profile]
    return matches[0] if len(matches) == 1 else None


def _prepend_ascii_header(path, lines):
    text = Path(path).read_text(encoding="ascii")
    header = "".join(f"; {ln}\n" for ln in lines)
    Path(path).write_text(header + text, encoding="ascii")


def solidpitch_gcode(out_gcode: Path, pen_mm=None, z_up=3.5,
                     travel_speed=8000, plunge_speed=500, lift_speed=2000,
                     offset_x=None, offset_y=None, end_policy="release",
                     quiet_fans=True, profile=None):
    """Emit the solid-pitch ladder sheet as ONE g-code file, ONE gwrite
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
            f"solidpitch: refusing to generate -- this sheet is PINNED to {EXPECT_PEN_ID!r} "
            f"({EXPECT_PEN_LABEL!r}); the active/requested pen is {req_id_str} "
            f"({profile.get('label')!r}). This sheet's pitch ladder and predictions are "
            f"calibrated from uright_gel's own wedge/core-width evidence and are "
            f"not a valid measurement of a different pen. Mount uright_gel and re-request, "
            f"or POST with pen_id='uright_gel' explicitly.")

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

    (elements, square_records, fid_records, groups,
     footprint_w, footprint_h, unused_cells) = _build_elements()
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
        # Known limitation: when this module is run directly (dev scripts, not the
        # app's own route), `_fix_gwrite_source_comment` below embeds the absolute
        # path of the working directory as a "; Source: ..." comment line, which
        # can be longer (~132 chars) than this module's wrapped prediction lines.
        # App-generated files carry the short work-dir-relative path instead. The
        # comment is produced by `_fix_gwrite_source_comment` in `gcode_post.py`, a
        # shared writer used by every sibling calibration module.
        _fix_gwrite_source_comment(tmp_gcode, out_gcode)
        _prepend_ascii_header(tmp_gcode, _PREDICTION_LINES)
    svg.unlink(missing_ok=True)

    validate_gcode_xy_motion(out_gcode, reach)
    safety = validate_calibration_gcode_safety(out_gcode, z_down_abs)
    pendown_check, strokes = _pendown_check(out_gcode, z_down_abs, elements, n_pendowns_expected)
    pitch_realization = _pitch_realization(square_records, strokes)

    # Plot-time estimate, model (a) -- see module docstring.
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
        "pitches_mm": list(PITCHES_MM), "geometries": list(GEOMETRIES),
        "control_pitches_mm": list(CONTROL_PITCHES_MM), "control_mode": CONTROL_MODE,
        "control_predicted_reading": CONTROL_PREDICTED_READING,
        "control_predicted_bare_frac": CONTROL_PREDICTED_BARE_FRAC,
        "control_open_max_darkness_frac": CONTROL_OPEN_MAX_DARKNESS_FRAC,
        "min_side_mm": MIN_SIDE_MM, "null_pitch_mm": NULL_PITCH_MM,
        "deposition_pitch_mm": DEPOSITION_PITCH_MM,
        "clearance_mm": CLEARANCE_MM,
        "pitch_realization_tol_mm": PITCH_REALIZATION_TOL_MM,
        "seed_placement": SEED_PLACEMENT, "seed_order": SEED_ORDER,
        "grid": {"cols": GRID_COLS, "rows": GRID_ROWS, "cell_pitch_mm": CELL_PITCH_MM,
                 "unused_cells": unused_cells},
        "squares": square_records,
        "fiducials": fid_records,
        "prediction": PREDICTION,
        "pitch_realization": pitch_realization,
    }

    import json as _json
    layout_path = WORK_DIR / f"{out_gcode.stem}_layout.json"
    layout_path.write_text(_json.dumps(layout, indent=2), encoding="utf-8")

    return {"kind": "solidpitch", "pen_mm": pen_mm, "pen_id": EXPECT_PEN_ID,
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
            "safety_selfcheck": safety, "pendown_check": pendown_check,
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
            "layout": layout, "layout_path": str(layout_path)}
