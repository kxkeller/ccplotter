"""ccplotter/calibration_plots.py -- gray step-wedge and companion calibration plots.

What the sheets measure and how to read them:
  * Gray step-wedge (`_wedge_gcode`): 16 horizontal-hatch patches at stepped spacings
    spanning WEDGE_LO_MULT..WEDGE_HI_MULT x the pen's fill_mm, a continuous spacing ramp
    over the same range, and 3 dark-end cross-hatch patches. It measures TONE versus line
    spacing: find the spacing where the patch reads solid and where it stops reading as
    tone and becomes "white space and lines".
  * Combo / min-ink strip / fill swatches: the min-ink strip measures how short a stroke
    can be before the pen stops marking (a ball-rotation threshold, independent of pen
    spacing); the swatches show three fill spacings as solid blocks.
  * Line-width block: pairs of lines at stepped separations plus isolated single lines.
    The tightest pair that still shows white between its two lines is the deposited line
    width; the isolated singles give the width directly under a loupe or a scan.
  * Registration marks: marks plotted on the bare build plate so paper can be aligned
    to them afterwards.

All of these resolve Z the same way `_calibration_gcode` does -- resolve_z_touch(cal,
profile), PROFILE WINS, raising ZTouchNotCalibrated rather than ever defaulting to 0.0
(see ccplotter/cal.py) -- and every file is checked by validate_gcode_xy_motion and
validate_calibration_gcode_safety (ASCII-only, no G92, valid fan indices, every Z at or
above the floor). Unlike several sibling modules this one has no EXPECT_REACH_ID pin: it
uses the active reach profile (with the pen's reach clamp applied).

The module is separate from gcode_post.py because the wedge is a distinct GEOMETRY
(layout constants, hatch/ramp builders, patch bookkeeping) on top of the same low-level
gwrite plumbing gcode_post.py exposes (_gcode_lifecycle_values,
validate_calibration_gcode_safety).

The wedge spacing is anchored to the ACTIVE PEN's fill_mm (see _resolve_fill_mm and
build_wedge_polylines), and an optional second, harder-pressure pass of the whole chart
can be stacked on the same sheet (see _wedge_gcode and _splice_pressure_blocks).
"""
import math
import re
import shlex
from pathlib import Path

import vpype as vp
import vpype_cli

from ccplotter.atomic_io import atomic_target, atomic_write_text
from ccplotter.config import CONFIG_PATH, PREVIEW_COLOR, PX2MM, WORK_DIR
from ccplotter.cal import resolve_pen_offset, load_cal, active_pen_profile, resolve_z_touch
from ccplotter.gcode_post import (_gcode_lifecycle_values, validate_calibration_gcode_safety,
                                   _fix_gwrite_source_comment, _gcode_pendown_bbox)
from ccplotter.reach import active_reach_profile, validate_gcode_xy_motion, point_in_polygon, pen_reach_profile
from ccplotter.svg_io import _write_paths_svg

# ---------------------------------------------------------------------------
# Layout geometry (mm, layout space -- y grows "down" like image space, same
# convention _calibration_gcode uses before the vpype translate flips it).
# The SPACING VALUES drawn into this geometry are not module constants; they are
# derived per-pen in build_wedge_polylines, because a fixed ladder is wrong for
# any pen whose deposit width is not ~0.25mm (a BIC Cristal measures ~0.32mm) and
# most of a fixed range plots either fully solid or fully separated lines, which
# tells you nothing.
# ---------------------------------------------------------------------------
PATCH = 12.0            # patch side, mm
GAP = 3.0                # gap between patches, mm
LABEL_H = 4.0             # tick-mark strip height under each patch, mm
TICK_H = 2.0             # individual tick length, mm
ROW_H = PATCH + LABEL_H

# Pen-relative spacing range. Tone reads as TONE only down to roughly 50% coverage,
# i.e. spacing <= ~2x the pen's deposited width -- coarser than that and it reads as
# "just white space and lines", on every pen tested. So 2.0x fill_mm is the top
# (coarsest/lightest) of the USABLE range, not an arbitrary ceiling. Neither pen
# tested reached a fully solid patch even near 1x geometric coverage (a BIC patch at
# 0.78x overlap still showed white), so the true solid point must be BRACKETED, not
# assumed at 1x: the bottom of the range goes below 1x fill_mm, hence 0.5x.
WEDGE_LO_MULT = 0.5
WEDGE_HI_MULT = 2.0

# Step count / grid shape. 8 columns x 2 rows = 16 patches. The grid's width
# (117mm) stays under the ramp strip's fixed 120mm width, which is what sets
# FOOTPRINT_W (see the assert below), so the footprint is 120 x 73mm regardless of
# the column count. 16 patches are packed into the 0.5x-2.0x band, where all the
# usable variation lives.
GRID_COLS = 8
GRID_ROWS = 2
WEDGE_STEP_COUNT = GRID_COLS * GRID_ROWS   # 16

GRID_W = GRID_COLS * PATCH + (GRID_COLS - 1) * GAP     # 117.0
GRID_H = GRID_ROWS * ROW_H + (GRID_ROWS - 1) * GAP      # 35.0

RAMP_W, RAMP_H = 120.0, 12.0
RAMP_Y0 = GRID_H + GAP                                    # 38.0

CROSS_Y0 = RAMP_Y0 + RAMP_H + LABEL_H + GAP                # 57.0
CROSS_LABELS = [13, 14, 15]
CROSS_W = len(CROSS_LABELS) * PATCH + (len(CROSS_LABELS) - 1) * GAP   # 42.0

FOOTPRINT_W = max(GRID_W, RAMP_W, CROSS_W)                  # 120.0 (unchanged)
FOOTPRINT_H = CROSS_Y0 + PATCH + LABEL_H                     # 73.0 (unchanged)
assert FOOTPRINT_W <= 150.0 and FOOTPRINT_H <= 80.0, "footprint exceeds cap"

# Two-pressure-block stacking: vertical gap between the normal-pressure chart
# (top) and the double-pressure chart (bottom) on the same sheet, wide enough to
# clear the block-identifier tick marks (see block_id_marks) without touching
# either chart's own content.
BLOCK_GAP = 15.0

# Default pressure multiplier for the second block. Applied to z_down (a
# NEGATIVE offset from z_touch_abs), so 2.0x makes the plunge twice as deep
# below paper contact, not twice the absolute Z.
PRESSURE_MULT_DEFAULT = 2.0

# Pressure comparison default: OFF. On the pens tested, pressure made little
# difference to tone, so the two-block pressure-comparison chart only doubles plot
# time and ink for no information; the default is a single normal-pressure block.
# The capability is kept as an explicit opt-in (pressure_compare=True) for a pen
# that might show pressure sensitivity.
PRESSURE_COMPARE_DEFAULT = False


def _resolve_fill_mm(profile) -> float:
    """Same convention as gcode_post._calibration_gcode's _fill_est: read the
    ACTIVE PEN profile's measured fill_mm (the real solid-fill deposit
    spacing from a calibration session), guard against nonsense (an empty/
    corrupt/pre-fill_mm profile), and fall back to the EnerGel-era default
    0.25mm -- never silently to 0.0 or an unset value."""
    fill_est = float((profile or {}).get("fill_mm") or 0.25)
    if not (0.02 <= fill_est <= 2.0):                # nonsense guard -> EnerGel-era default
        fill_est = 0.25
    return fill_est


def _geometric_spacings(lo: float, hi: float, n: int) -> list:
    """n spacings (mm), geometrically stepped from lo to hi inclusive (equal
    RATIO between consecutive steps, not equal mm difference). Coverage/tone
    response is a ratio effect (the usable threshold is ~2x pen width, a
    RELATIVE figure, not an absolute mm one)."""
    if n <= 1:
        return [round(lo, 3)]
    ratio = (hi / lo) ** (1.0 / (n - 1))
    return [round(lo * (ratio ** k), 3) for k in range(n)]


def hatch_h(x0, y0, w, h, spacing):
    """Horizontal parallel lines filling (x0,y0,w,h) at absolute spacing."""
    lines, y, n = [], y0, 0
    while y <= y0 + h + 1e-9:
        lines.append([(x0, y), (x0 + w, y)])
        y += spacing
        n += 1
    return lines


def hatch_v(x0, y0, w, h, spacing):
    """Vertical parallel lines filling (x0,y0,w,h) at absolute spacing."""
    lines, x = [], x0
    while x <= x0 + w + 1e-9:
        lines.append([(x, y0), (x, y0 + h)])
        x += spacing
    return lines


def ramp_lines(x0, y0, w, h, s_lo, s_hi):
    """Vertical lines whose LOCAL spacing grows smoothly (Euler-stepped
    linear interpolation) from s_lo at x=0 to s_hi at x=w."""
    lines, x = [], 0.0
    while x <= w + 1e-9:
        lines.append([(x0 + x, y0), (x0 + x, y0 + h)])
        frac = x / w
        s = s_lo + (s_hi - s_lo) * frac
        x += s
    return lines


def rect_outline(x0, y0, w, h):
    return [[(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h), (x0, y0)]]


def ticks(x0, w, y0, n, tick_h=TICK_H):
    """n evenly spaced vertical tick marks under a patch/strip of width w,
    starting tick_h below y0 (the bottom edge of the thing being labeled)."""
    out = []
    if n <= 0:
        return out
    pitch = w / (n + 1)
    for k in range(1, n + 1):
        tx = x0 + k * pitch
        out.append([(tx, y0), (tx, y0 + tick_h)])
    return out


def block_id_marks(x0, y0, n):
    """Extends the SAME tick-mark idiom `ticks()` uses for patch numbering
    (a short baseline + n evenly spaced vertical ticks) to mark WHICH
    pressure block a chart is, rather than inventing a new marking style.
    1 tick = NORMAL pressure (plots first, top of sheet). 2 ticks = DOUBLE
    pressure (plots second, bottom of sheet). Drawn well clear of the grid
    (see BLOCK_GAP)."""
    strip_w = 8.0
    out = [[(x0, y0), (x0 + strip_w, y0)]]
    out += ticks(x0, strip_w, y0, n, tick_h=TICK_H * 1.5)
    return out


def build_wedge_polylines(fill_mm=0.25):
    """One gray step-wedge chart at the given pen's fill_mm (mm): 16
    stepped-spacing patches (8x2 grid, horizontal hatch) spanning
    WEDGE_LO_MULT..WEDGE_HI_MULT x fill_mm, a continuous spacing ramp over
    the same range, and a 3-patch dark-end cross-hatch extension (cross-hatching
    measurably darkens tone below the single-family solid point). Returns
    (polys, patch_rects); patch_rects is bookkeeping for the UI/README, not
    gcode."""
    lo, hi = WEDGE_LO_MULT * fill_mm, WEDGE_HI_MULT * fill_mm
    step_spacings = _geometric_spacings(lo, hi, WEDGE_STEP_COUNT)
    ramp_s_lo, ramp_s_hi = lo, hi
    # Checkpoints along the ramp: the two ends (lo/hi, same as the stepped
    # grid's bracket), the midpoint, and 1.0x fill_mm itself, the point that is
    # NOT reliably solid.
    ramp_checkpoints = sorted(set(round(m * fill_mm, 3) for m in (0.5, 0.75, 1.0, 1.5, 2.0)))
    # Cross-hatch spacings as ratios to fill_mm (1.2x single family, 1.68x and
    # 2.4x second family), so they scale with any pen's fill_mm.
    cross_spacing_1 = round(1.2 * fill_mm, 3)
    cross_patches = [
        {"label": 13, "second_family": None},                       # single-family baseline
        {"label": 14, "second_family": round(1.68 * fill_mm, 3)},    # + 90deg family
        {"label": 15, "second_family": round(2.4 * fill_mm, 3)},     # + 90deg family, coarser
    ]

    polys = []
    patch_rects = []   # bookkeeping for preview/README/UI, not gcode

    # 1) WEDGE_STEP_COUNT stepped-spacing patches, GRID_COLS x GRID_ROWS grid.
    for i, s in enumerate(step_spacings):
        col, row = i % GRID_COLS, i // GRID_COLS
        x0 = col * (PATCH + GAP)
        y0 = row * (ROW_H + GAP)
        polys += rect_outline(x0, y0, PATCH, PATCH)
        polys += hatch_h(x0, y0, PATCH, PATCH, s)
        polys += ticks(x0, PATCH, y0 + PATCH + 1.0, i + 1)
        patch_rects.append({"index": i + 1, "spacing_mm": s, "x0": x0, "y0": y0,
                             "w": PATCH, "h": PATCH, "kind": "single-family"})

    # 2) continuous ramp strip, lo -> hi smoothly across 120mm.
    polys += rect_outline(0.0, RAMP_Y0, RAMP_W, RAMP_H)
    polys += ramp_lines(0.0, RAMP_Y0, RAMP_W, RAMP_H, ramp_s_lo, ramp_s_hi)
    for target in ramp_checkpoints:
        xf = RAMP_W * (target - ramp_s_lo) / (ramp_s_hi - ramp_s_lo)
        xf = max(0.0, min(RAMP_W, xf))
        polys.append([(xf, RAMP_Y0 + RAMP_H + 1.0), (xf, RAMP_Y0 + RAMP_H + 1.0 + TICK_H)])
    patch_rects.append({"index": "ramp", "spacing_mm": [ramp_s_lo, ramp_s_hi],
                         "x0": 0.0, "y0": RAMP_Y0, "w": RAMP_W, "h": RAMP_H,
                         "kind": "continuous-ramp", "checkpoints_mm": ramp_checkpoints})

    # 3) dark-end cross-hatch extension: 3 patches.
    for j, spec in enumerate(cross_patches):
        x0 = j * (PATCH + GAP)
        y0 = CROSS_Y0
        polys += rect_outline(x0, y0, PATCH, PATCH)
        polys += hatch_h(x0, y0, PATCH, PATCH, cross_spacing_1)
        kind = "single-family"
        if spec["second_family"] is not None:
            polys += hatch_v(x0, y0, PATCH, PATCH, spec["second_family"])
            kind = f"crosshatch+{spec['second_family']}mm"
        polys += ticks(x0, PATCH, y0 + PATCH + 1.0, spec["label"])
        patch_rects.append({"index": spec["label"], "spacing_mm": cross_spacing_1,
                             "second_family_mm": spec["second_family"], "x0": x0, "y0": y0,
                             "w": PATCH, "h": PATCH, "kind": kind})

    # 4) corner registration mark (outside the grid, orientation only).
    polys.append([(-2.0, -2.0), (-2.0, 2.0)])
    polys.append([(-2.0, -2.0), (2.0, -2.0)])

    return polys, patch_rects


# ---------------------------------------------------------------------------
# Combo sheet: the min-ink strip and a fill-spacing swatch on the SAME sheet as
# the gray wedge, driven by whichever pen is active (not hardcoded to the BIC).
# See _combo_gcode below.
# ---------------------------------------------------------------------------

# Min-ink strip geometry: PEN-INDEPENDENT stroke lengths (a physical
# ball-rotation threshold, not a pen-relative spacing like fill_mm/the wedge)
# -- (length_mm, x_pitch_mm) per row, reference row first. The reference row uses
# a 6.0mm pitch (5.0mm strokes would overlap consecutive starts at the usual 4mm
# pitch); the 11 test rows use a 4mm pitch (max 1.5mm stroke, no overlap risk).
MININK_ROWS = [(5.0, 6.0)] + [(L, 4.0) for L in
                               (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5)]
MININK_N_COLS = 12
MININK_ROW_PITCH_Y = 6.0


def build_minink_polylines():
    """12 rows x 12 ISOLATED 2-point horizontal strokes (own plunge/draw/lift
    each -- see _combo_gcode: every polyline in this sheet becomes its own
    gwrite segment_first/segment cycle at the sheet's single z_up_abs/
    z_down_abs, so isolation falls out of "one polyline per stroke" with no
    extra plumbing). Returns (polys, row_bands); row_bands is legend
    bookkeeping (which Y-band is which length), not gcode."""
    polys, row_bands = [], []
    for i, (length_mm, pitch_x) in enumerate(MININK_ROWS):
        y = i * MININK_ROW_PITCH_Y
        for k in range(MININK_N_COLS):
            x0 = k * pitch_x
            polys.append([(x0, y), (x0 + length_mm, y)])
        row_bands.append({"row": i, "length_mm": length_mm, "y_mm": y,
                           "x_pitch_mm": pitch_x, "n_strokes": MININK_N_COLS,
                           "kind": "reference" if i == 0 else "test"})
    return polys, row_bands


# Fill-spacing swatch: 3 solid ~15x15mm blocks, pitches = these multiples of
# fill_mm (loosest -> tightest, left -> right). At BIC fill_mm=0.19 this is
# ~0.19/0.15/0.13mm.
FILL_SWATCH_MULTS = [1.0, 0.8, 0.66]
SWATCH_W = SWATCH_H = 15.0
SWATCH_GAP_X = 8.0


def build_fill_swatch_polylines(fill_mm):
    """3 blocks of ISOLATED parallel horizontal fill lines (calibration_plots.
    hatch_h -- same idiom the wedge patches already use, own plunge/draw/lift
    per line, never routed through the real hatch/tone generator). Returns
    (polys, swatch_specs); swatch_specs is legend bookkeeping (which X-band is
    which pitch), not gcode."""
    polys, specs = [], []
    x0 = 0.0
    for mult in FILL_SWATCH_MULTS:
        pitch = round(mult * fill_mm, 3)
        lines = hatch_h(x0, 0.0, SWATCH_W, SWATCH_H, pitch)
        polys += lines
        specs.append({"mult": mult, "pitch_mm": pitch, "x0_mm": x0, "x1_mm": x0 + SWATCH_W,
                       "n_lines": len(lines)})
        x0 += SWATCH_W + SWATCH_GAP_X
    return polys, specs


# Vertical/horizontal clear gaps between the combo sheet's sections.
# -- Line-width block --------------------------------------------------------
# Measures the pen's DEPOSITED LINE WIDTH, which nothing else on this sheet
# does. The wedge measures tone, the swatches measure fill SPACING, the min-ink
# strip measures stroke LENGTH -- none of them isolate width.
#
# Why it is needed: the preview renders every stroke at `fill_mm`, but fill_mm is
# the loosest pass PITCH that fills solid, not a line width. If the real line is
# wider, every preview is optimistic and fine detail that looks crisp on screen
# fuses on paper.
#
# The range is ANCHORED TO pen_mm, NOT fill_mm. Anchoring the instrument's range
# to fill_mm would couple it to the very constant this sheet exists to check.
# pen_mm (the ball diameter) is a physical fact not in dispute.
#
# SATURATION CUTS BOTH WAYS, and the COARSE end is the likelier failure:
#   - every pair reads as TWO  -> width is below the fine end; re-run lower.
#   - every pair MERGES        -> width exceeds the coarse end; re-run higher.
# A 1.0mm ball plausibly lays 0.4-0.5mm. The range therefore runs to 1.2x
# pen_mm (1.2mm at BIC) so an unexpectedly wide line still lands inside the
# ladder instead of returning a null sheet.
LINEW_LO_MULT, LINEW_HI_MULT = 0.15, 1.2      # x pen_mm
LINEW_STEPS = 12
LINEW_LEN = 12.0          # mm, per line
LINEW_PAIR_GAP = 4.0      # mm between adjacent PAIRS (>> any separation tested)
LINEW_SOLO_GAP = 6.0      # mm isolating the single reference lines
LINEW_SOLO_REPLICATES = 3
# Wrap the pair ladders instead of running all 12 across one row. Unwrapped the
# block is 194mm wide, which overflows the reach envelope wherever it lands.
# Six per row keeps it well inside B5 and the ticks still number 1-12 in reading
# order.
LINEW_COLS = 6


def build_linewidth_polylines(pen_mm):
    """Line-width block. Returns (polys, spec); spec is legend bookkeeping for
    the README/UI, never gcode.

    Every polyline is a 2-point stroke, so each becomes its own plunge/draw/lift
    cycle in the emitter -- the same isolation the min-ink strip relies on.

    READ ORDER: the ISOLATED SINGLES are the PRIMARY measurement, with
    replicates; the pair ladder is an independent cross-check. Singles are
    primary because pair-merge is another eyeball threshold read, and it is
    biased UPWARD -- two adjacent wet ballpoint lines can bleed together at a
    separation slightly greater than the true width. A single line has no such
    term and can be measured directly from a 1200dpi scan (0.021mm/px, ample for
    a 0.3mm line).

    If the two reads DISAGREE, that disagreement is the finding (it quantifies
    bleed) and must be reported, not reconciled away.
    """
    seps = _geometric_spacings(LINEW_LO_MULT * pen_mm, LINEW_HI_MULT * pen_mm,
                               LINEW_STEPS)
    polys = []
    spec = {"anchor": "pen_mm", "pen_mm": pen_mm, "separations_mm": seps,
            "line_len_mm": LINEW_LEN, "primary": "isolated_singles",
            "cross_check": "pair_ladder", "singles": [], "orientations": []}

    # 1) PRIMARY: isolated single lines, replicated, at production draw speed
    #    (the sheet's single draw_speed applies to every stroke here).
    y = 0.0
    for ang in (0, 90):
        for r in range(LINEW_SOLO_REPLICATES):
            x = r * (LINEW_LEN + LINEW_SOLO_GAP)
            if ang == 0:
                polys.append([(x, y), (x + LINEW_LEN, y)])
            else:
                polys.append([(x, y), (x, y + LINEW_LEN)])
            spec["singles"].append({"angle_deg": ang, "replicate": r + 1,
                                    "x_mm": x, "y_mm": y})
        y += (LINEW_LEN if ang == 90 else 0) + LINEW_SOLO_GAP
    y += LINEW_SOLO_GAP

    # 2) CROSS-CHECK: pair ladders, wrapped LINEW_COLS per row.
    for orient in (0, 90):
        bands = []
        row_h = LINEW_LEN + LINEW_SOLO_GAP
        for i, s in enumerate(seps):
            col, row = i % LINEW_COLS, i // LINEW_COLS
            ry = y + row * row_h
            if orient == 0:
                cx = col * (LINEW_LEN + LINEW_PAIR_GAP)
                polys.append([(cx, ry), (cx + LINEW_LEN, ry)])
                polys.append([(cx, ry + s), (cx + LINEW_LEN, ry + s)])
                polys += ticks(cx, LINEW_LEN, ry + s + 1.0, i + 1)
                bands.append({"index": i + 1, "separation_mm": s, "x_mm": cx, "y_mm": ry})
            else:
                cx = col * (LINEW_PAIR_GAP + LINEW_HI_MULT * pen_mm + 1.0)
                polys.append([(cx, ry), (cx, ry + LINEW_LEN)])
                polys.append([(cx + s, ry), (cx + s, ry + LINEW_LEN)])
                polys += ticks(cx, max(s, 1.0), ry + LINEW_LEN + 1.0, i + 1)
                bands.append({"index": i + 1, "separation_mm": s, "x_mm": cx, "y_mm": ry})
        spec["orientations"].append({"angle_deg": orient, "y_mm": y,
                                     "cols": LINEW_COLS, "pairs": bands})
        y += ((len(seps) + LINEW_COLS - 1) // LINEW_COLS) * row_h + LINEW_SOLO_GAP

    return polys, spec


COMBO_GAP_V = 10.0
COMBO_GAP_H = 10.0


def _combo_gcode(out_gcode: Path, pen_mm=0.5, z_up=3.5, z_down=-1.0,
                  draw_speed=3000, travel_speed=8000, plunge_speed=500, lift_speed=2000,
                  offset_x=None, offset_y=None, end_policy="release", quiet_fans=True,
                  profile=None):
    """All-in-one calibration sheet: gray wedge (top) + min-ink strip and
    fill-spacing swatches side-by-side (below) on ONE true-size gcode file,
    driven by whichever pen profile is active (or explicitly passed) -- not
    hardcoded to any one pen. Structurally modeled on _wedge_gcode: same
    signature shape, same Z resolution (resolve_z_touch via the active
    profile, PROFILE WINS, raises ZTouchNotCalibrated -- never a silent
    Z=0.0), same fill_mm resolution (_resolve_fill_mm), same single
    z_up/z_down for the WHOLE sheet (every polyline below becomes its own
    isolated plunge/draw/lift at that one z_up_abs/z_down_abs via gwrite's
    z_auto segment_first/segment templates -- see vpype.toml -- so the
    min-ink strokes get real single-stroke isolation with no extra
    machinery).

    Layout (mm, layout space, y grows down): wedge at (0,0), footprint
    ~120x73mm (build_wedge_polylines). COMBO_GAP_V clear gap below
    it, then the min-ink strip (~71x66mm) and fill swatches (~61x15mm,
    top-aligned with the strip) side by side, separated by COMBO_GAP_H.
    Placing them side by side rather than stacking all three keeps the total
    footprint (~142x149mm incl. padding) inside the confirmed reach envelope
    (validated below via validate_gcode_xy_motion), instead of needing the
    full ~73+66+15mm+gaps of a vertical stack.

    Placement: identical formula/margin convention to _wedge_gcode (an
    inboard PLACEMENT_MARGIN_MM, with pen_off_y folded into the Y default for
    tall content) -- the route wraps this the same way _gen_with_fallback
    wraps _wedge_gcode/_calibration_gcode, so a caller-requested offset that
    doesn't fit falls back to this known-good default rather than failing
    outright."""
    ppmm = 1.0 / PX2MM

    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()
    if profile is None:
        profile = active_pen_profile()
    # Per-pen reach clamp: same conservative-by-construction intersection as
    # composite.py -- a profile with no reach_clamp gets this
    # active_reach_profile() object back unchanged.
    reach = pen_reach_profile(active_reach_profile(cal), profile)
    fill_mm = _resolve_fill_mm(profile)

    def _bbox(polys):
        xs = [p[0] for poly in polys for p in poly]
        ys = [p[1] for poly in polys for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    def _shift(polys, dx, dy):
        return [[(x + dx, y + dy) for x, y in poly] for poly in polys]

    # 1) WEDGE, verbatim geometry, normalized to start at (0,0).
    wedge_raw, patch_rects = build_wedge_polylines(fill_mm)
    wx0, wy0, wx1, wy1 = _bbox(wedge_raw)
    wedge_polys = _shift(wedge_raw, -wx0, -wy0)
    wedge_w, wedge_h = wx1 - wx0, wy1 - wy0

    # 2) MIN-INK strip, normalized, placed below the wedge.
    minink_raw, row_bands = build_minink_polylines()
    mx0, my0, mx1, my1 = _bbox(minink_raw)
    minink_y0 = wedge_h + COMBO_GAP_V
    minink_polys = _shift(minink_raw, -mx0, minink_y0 - my0)
    minink_w, minink_h = mx1 - mx0, my1 - my0

    # 3) FILL SWATCHES, normalized, placed beside the min-ink strip (same Y
    # band -- top-aligned), not stacked below it (see docstring: this is what
    # keeps the combined footprint inside reach).
    swatch_raw, swatch_specs = build_fill_swatch_polylines(fill_mm)
    sx0, sy0, sx1, sy1 = _bbox(swatch_raw)
    swatch_x0 = minink_w + COMBO_GAP_H
    swatch_polys = _shift(swatch_raw, swatch_x0 - sx0, minink_y0 - sy0)

    all_polys = wedge_polys + minink_polys + swatch_polys
    xs = [p[0] for poly in all_polys for p in poly]
    ys = [p[1] for poly in all_polys for p in poly]
    layout_w = max(xs) - min(xs)
    layout_h = max(ys) - min(ys)
    min_x, min_y = min(xs), min(ys)
    pad = 3.0
    norm_polys = [[(x - min_x + pad, y - min_y + pad) for x, y in poly] for poly in all_polys]
    W = int(round((layout_w + 2 * pad) * ppmm))
    H = int(round((layout_h + 2 * pad) * ppmm))
    polys_px = [[(x * ppmm, y * ppmm) for x, y in p] for p in norm_polys]

    svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_paths_svg(svg, W, H, float(pen_mm) * ppmm, polylines=polys_px)

    z_touch, z_touch_source = resolve_z_touch(cal, profile)
    print(f"  [z] combo sheet z_touch_abs={z_touch:.2f}mm (source={z_touch_source})")
    pen_off_x, pen_off_y = resolve_pen_offset(cal, profile)
    # Same inboard-placement convention as _wedge_gcode's pressure_compare
    # path (folds pen_off_y into the Y default) -- this sheet is similarly
    # tall (~149mm incl. padding), so the same correction applies
    # unconditionally.
    PLACEMENT_MARGIN_MM = 20.0
    offset_x = reach["bbox"][0] + PLACEMENT_MARGIN_MM if offset_x is None else float(offset_x)
    offset_y = (reach["bbox"][1] + PLACEMENT_MARGIN_MM + pen_off_y) if offset_y is None else float(offset_y)
    comp_off_x, comp_off_y = offset_x - pen_off_x, offset_y - pen_off_y
    park_x, park_y = reach["safe_park_point"]
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))
    z_down_abs = z_touch + float(z_down)

    vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({
        "z_up": float(z_up), "z_up_abs": z_touch + float(z_up), "z_down": float(z_down),
        "z_down_abs": z_down_abs, "z_touch_abs": z_touch, "z_travel_abs": z_travel_abs,
        "draw_speed": float(draw_speed), "travel_speed": float(travel_speed),
        "plunge_speed": float(plunge_speed), "lift_speed": float(lift_speed),
        "z_auto": "1", "origin_x": park_x, "origin_y": park_y,
        **_gcode_lifecycle_values(end_policy, quiet_fans)})

    with atomic_target(out_gcode) as tmp_gcode:
        vpype_cli.execute(f'read {shlex.quote(str(svg))} pagesize {W}x{H}px linesort '
                           f'translate -- {comp_off_x}mm {-comp_off_y}mm '
                           f'gwrite --profile elegoo_cc {shlex.quote(str(tmp_gcode))}')
        _fix_gwrite_source_comment(tmp_gcode, out_gcode)
    validate_gcode_xy_motion(out_gcode, reach)
    safety = validate_calibration_gcode_safety(out_gcode, z_down_abs)

    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(svg))} color {PREVIEW_COLOR} '
                           f'write {shlex.quote(str(tmp_prev))}')

    place = {"x": offset_x, "y": offset_y, "w": layout_w + 2 * pad, "h": layout_h + 2 * pad}
    return {"pen_mm": pen_mm, "fill_mm": fill_mm,
            "pattern_mm": [layout_w + 2 * pad, layout_h + 2 * pad],
            "offset_mm": [offset_x, offset_y],
            "polylines": len(all_polys), "polylines_wedge": len(wedge_polys),
            "polylines_minink": len(minink_polys), "polylines_swatch": len(swatch_polys),
            "preview": prev.name, "place": place,
            "patch_rects": patch_rects, "row_bands": row_bands, "swatch_specs": swatch_specs,
            "sections": {
                "wedge": {"x0_mm": pad, "y0_mm": pad, "w_mm": wedge_w, "h_mm": wedge_h},
                "minink": {"x0_mm": pad, "y0_mm": pad + minink_y0, "w_mm": minink_w, "h_mm": minink_h},
                "swatch": {"x0_mm": pad + swatch_x0, "y0_mm": pad + minink_y0,
                           "w_mm": sx1 - sx0, "h_mm": sy1 - sy0},
            },
            "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
            "z_up_abs": round(z_touch + float(z_up), 3),
            "z_down_abs": round(z_down_abs, 3),
            "safety_selfcheck": safety,
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
            "svg_wh_px": [W, H], "svg_path": str(svg)}



# Individual calibration blocks, each plottable on its own sheet. The all-in-one
# _combo_gcode and its route are kept alongside them, unchanged.
CAL_BLOCKS = {
    "wedge":     "Gray step wedge (tone ladder)",
    "swatch":    "Fill-spacing swatches",
    "minink":    "Min-ink strip (shortest stroke that inks)",
    "linewidth": "Line width (how wide a line the pen lays)",
}


def _build_cal_block(kind, fill_mm, pen_mm):
    """Dispatch to one block builder. Returns (polys, spec)."""
    if kind == "wedge":
        polys, spec = build_wedge_polylines(fill_mm)
        return polys, {"patch_rects": spec}
    if kind == "swatch":
        polys, spec = build_fill_swatch_polylines(fill_mm)
        return polys, {"swatch_specs": spec}
    if kind == "minink":
        polys, spec = build_minink_polylines()
        return polys, {"row_bands": spec}
    if kind == "linewidth":
        # Anchored to the ball diameter, not fill_mm -- see the line-width block comment.
        polys, spec = build_linewidth_polylines(float(pen_mm))
        return polys, {"linewidth_spec": spec}
    raise ValueError(f"unknown calibration block {kind!r} (expected one of {sorted(CAL_BLOCKS)})")


def _calblock_gcode(out_gcode: Path, kind="linewidth", pen_mm=0.5, z_up=3.5, z_down=-1.0,
                     draw_speed=3000, travel_speed=8000, plunge_speed=500,
                     lift_speed=2000, offset_x=None, offset_y=None,
                     end_policy="release", quiet_fans=True, profile=None):
    """ONE calibration block, on its own sheet. Not part of the all-in-one:
    the block is 156x48mm and the AIO stack is already ~150mm tall, so adding it
    would need 208mm against ~172mm of usable Kokuyo B5 height.

    Structurally a sibling of _combo_gcode/_wedge_gcode, deliberately duplicated
    rather than refactored into them: those two already coexist as parallel
    implementations by design, and this must not be able to perturb a sheet
    already relied on for tone/spacing/min-ink readings.

    Reading it: the tightest PAIR that still shows white between its two lines
    is the pen's deposited line width. Two orientations because a ballpoint's
    effective width is direction-dependent (see config.py).
    Isolated singles at the bottom are the control, and give the width directly
    under a loupe or a scan."""
    ppmm = 1.0 / PX2MM

    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()
    if profile is None:
        profile = active_pen_profile()
    # Per-pen reach clamp: same conservative-by-construction intersection as
    # composite.py -- a profile with no reach_clamp gets this
    # active_reach_profile() object back unchanged.
    reach = pen_reach_profile(active_reach_profile(cal), profile)
    fill_mm = _resolve_fill_mm(profile)

    # Anchored to the pen's ball diameter, not fill_mm -- see the line-width block comment.
    raw, block_spec = _build_cal_block(kind, fill_mm, pen_mm)
    xs = [q[0] for poly in raw for q in poly]
    ys = [q[1] for poly in raw for q in poly]
    layout_w, layout_h = max(xs) - min(xs), max(ys) - min(ys)
    min_x, min_y = min(xs), min(ys)
    pad = 3.0
    norm = [[(x - min_x + pad, y - min_y + pad) for x, y in poly] for poly in raw]
    W = int(round((layout_w + 2 * pad) * ppmm))
    H = int(round((layout_h + 2 * pad) * ppmm))
    polys_px = [[(x * ppmm, y * ppmm) for x, y in q] for q in norm]

    svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_paths_svg(svg, W, H, float(pen_mm) * ppmm, polylines=polys_px)

    z_touch, z_touch_source = resolve_z_touch(cal, profile)
    print(f"  [z] {kind} block z_touch_abs={z_touch:.2f}mm (source={z_touch_source})")
    pen_off_x, pen_off_y = resolve_pen_offset(cal, profile)
    PLACEMENT_MARGIN_MM = 20.0
    offset_x = reach["bbox"][0] + PLACEMENT_MARGIN_MM if offset_x is None else float(offset_x)
    offset_y = (reach["bbox"][1] + PLACEMENT_MARGIN_MM + pen_off_y) if offset_y is None else float(offset_y)
    comp_off_x, comp_off_y = offset_x - pen_off_x, offset_y - pen_off_y
    park_x, park_y = reach["safe_park_point"]
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))
    z_down_abs = z_touch + float(z_down)

    vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({
        "z_up": float(z_up), "z_up_abs": z_touch + float(z_up), "z_down": float(z_down),
        "z_down_abs": z_down_abs, "z_touch_abs": z_touch, "z_travel_abs": z_travel_abs,
        "draw_speed": float(draw_speed), "travel_speed": float(travel_speed),
        "plunge_speed": float(plunge_speed), "lift_speed": float(lift_speed),
        "z_auto": "1", "origin_x": park_x, "origin_y": park_y,
        **_gcode_lifecycle_values(end_policy, quiet_fans)})

    # NO linesort: like the calibration charts generally, this sheet must not be
    # reordered or routed. Pen-down connectors between adjacent pairs would
    # bridge the very gaps being measured and destroy the reading.
    with atomic_target(out_gcode) as tmp_gcode:
        vpype_cli.execute(f'read {shlex.quote(str(svg))} pagesize {W}x{H}px '
                          f'translate -- {comp_off_x}mm {-comp_off_y}mm '
                          f'gwrite --profile elegoo_cc {shlex.quote(str(tmp_gcode))}')
        _fix_gwrite_source_comment(tmp_gcode, out_gcode)
    validate_gcode_xy_motion(out_gcode, reach)
    safety = validate_calibration_gcode_safety(out_gcode, z_down_abs)

    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(svg))} color {PREVIEW_COLOR} '
                          f'write {shlex.quote(str(tmp_prev))}')

    return {"pen_mm": pen_mm, "fill_mm": fill_mm,
            "pattern_mm": [layout_w + 2 * pad, layout_h + 2 * pad],
            "offset_mm": [offset_x, offset_y], "polylines": len(raw),
            "preview": prev.name,
            "place": {"x": offset_x, "y": offset_y,
                      "w": layout_w + 2 * pad, "h": layout_h + 2 * pad},
            "kind": kind, "block_label": CAL_BLOCKS.get(kind, kind), **block_spec,
            "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
            "z_up_abs": round(z_touch + float(z_up), 3),
            "z_down_abs": round(z_down_abs, 3),
            "safety_selfcheck": safety,
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
            "svg_wh_px": [W, H], "svg_path": str(svg)}


# ---------------------------------------------------------------------------
# Registration-marks plot, plotted on the BARE BUILD PLATE (no paper) so the plate
# can be marked at the pen marks and paper aligned to them afterward. This is NOT a
# paper-relative/true_size DRAWING import like the combo/wedge/min-ink charts
# above -- there is no "paper" for those marks to be true-size ON. It models the
# DIRECT/ABSOLUTE gcode path instead (same convention _calibration_gcode and the
# index.html calibration loader use): offset_x/offset_y ARE the intended ABSOLUTE
# MACHINE position of the paper's own front-left corner (index.html computes
# `offset_x = BED - (offsetX + paperW)`, `offset_y = offsetY` and sends that to
# every other calibration route), so the marks land wherever the paper is
# CURRENTLY PLACED on the plate. Unlike every other generator in this file,
# offset_x/offset_y are therefore REQUIRED here (no reach-bbox-corner default):
# silently relocating the whole plot would mean marking the WRONG spot. Only an
# optional hole-circle reach failure is handled leniently; either required edge
# line must fit.
# ---------------------------------------------------------------------------

# Centerline gap: deliberately close to the sheet edge for easy alignment,
# yet outside it even with the standard 0.5mm pen.
REG_REFERENCE_EDGE_OFFSET_MM = 0.5
REG_REFERENCE_X0_MM = 5.0
REG_REFERENCE_X1_MM = 252.0
REG_HOLE_REFERENCE_RADIUS_MM = 2.0
REG_HOLE_REFERENCE_CENTERS_X_MM = (28.75, 228.25)
# The punched-hole edge is at the paper frame's bottom (paper_y=paper_h, plotter
# front), not its top. The per-pen `reach_clamp` in ccplotter/reach.py handles the
# angled Preppy's unreachable-datum problem without moving the paper. "From
# bottom", not "from top".
REG_HOLE_REFERENCE_CENTER_FROM_BOTTOM_MM = 6.25


def _reg_mark_groups(paper_w, paper_h, **_legacy_options):
    """Bare-plate reference geometry in paper coordinates (mm, y grows down).

    The two required horizontal strokes sit 0.5mm *outside* the paper's top and
    bottom edges, making the paper edge itself the alignment target rather than
    covering the reference with paper. Two independent, optional 2mm-radius
    circles sit at Kokuyo B5's first/last useful hole centers, nominally 6.25mm
    from the paper frame's BOTTOM edge (the punched edge maps to the plotter
    front; measured hole centers are 4.0-4.5mm from the edge), so they can be
    seen through 5.5mm punches. Legacy request options are accepted but
    intentionally do not alter this fixed physical reference geometry.
    """
    paper_w, paper_h = float(paper_w), float(paper_h)
    if paper_w < REG_REFERENCE_X1_MM:
        raise ValueError("registration reference needs paper width >= 252mm")
    y_hole = paper_h - REG_HOLE_REFERENCE_CENTER_FROM_BOTTOM_MM
    circle_steps = 16
    groups = [
        {"name": "top_edge_reference", "mandatory": True,
         "polys": [[(REG_REFERENCE_X0_MM, -REG_REFERENCE_EDGE_OFFSET_MM),
                    (REG_REFERENCE_X1_MM, -REG_REFERENCE_EDGE_OFFSET_MM)]]},
        {"name": "bottom_edge_reference", "mandatory": True,
         "polys": [[(REG_REFERENCE_X0_MM, paper_h + REG_REFERENCE_EDGE_OFFSET_MM),
                    (REG_REFERENCE_X1_MM, paper_h + REG_REFERENCE_EDGE_OFFSET_MM)]]},
    ]
    for index, cx in enumerate(REG_HOLE_REFERENCE_CENTERS_X_MM, start=1):
        poly = []
        for step in range(circle_steps + 1):
            angle = 2.0 * 3.141592653589793 * step / circle_steps
            poly.append((cx + REG_HOLE_REFERENCE_RADIUS_MM * math.cos(angle),
                         y_hole + REG_HOLE_REFERENCE_RADIUS_MM * math.sin(angle)))
        groups.append({"name": f"hole_reference_{index}", "mandatory": False, "polys": [poly]})
    return groups


def _parse_ordered_strokes(text: str, z_up_abs: float, tol: float = 0.005):
    """Parse isolated gwrite strokes IN EMISSION ORDER, returning every
    pen-down XY point for each input polyline. No linesort means output/input
    order remains 1:1, including the multi-segment hole-reference circles."""
    lines = text.splitlines()
    zup_prefix = f"G1 Z{z_up_abs:.2f}"
    out = []
    i = 0
    while i < len(lines):
        zdown_match = re.match(r"G1 Z(-?[\d.]+)", lines[i + 2]) if i + 2 < len(lines) else None
        if (lines[i].startswith(zup_prefix) and i + 3 < len(lines) and zdown_match
                and float(zdown_match.group(1)) < float(z_up_abs) - tol):
            travel = lines[i + 1]
            mx0 = re.search(r"X(-?[\d.]+)", travel); my0 = re.search(r"Y(-?[\d.]+)", travel)
            if mx0 and my0:
                points = [(float(mx0.group(1)), float(my0.group(1)))]
                j = i + 3
                # A final stroke is followed by the higher travel lift rather
                # than another per-stroke z_up. Stop at *any* Z command so
                # postamble park moves cannot be misread as circle geometry.
                while j < len(lines) and not lines[j].startswith("G1 Z"):
                    mx = re.search(r"X(-?[\d.]+)", lines[j]); my = re.search(r"Y(-?[\d.]+)", lines[j])
                    if mx and my:
                        points.append((float(mx.group(1)), float(my.group(1))))
                    j += 1
                out.append(points)
                i = j
            else:
                i += 1
        else:
            i += 1
    return out


def _registration_marks_gcode(out_gcode: Path, paper_w, paper_h, offset_x, offset_y,
                               pen_mm=0.5, z_up=3.5, z_down=-0.6,
                               draw_speed=3000, travel_speed=8000, plunge_speed=500, lift_speed=2000,
                               end_policy="release", quiet_fans=True, profile=None):
    """Build-plate registration marks: two mandatory horizontal edge lines
    0.5mm outside the paper plus two optional circles at Kokuyo B5 hole centers,
    ALL in absolute machine coordinates matching exactly where the app would
    place a paper.w x paper_h sheet at (offset_x, offset_y) (same offset
    semantics as _calibration_gcode/_composite -- the PEN-target machine
    offset of the paper's front-left corner). Z comes from the active pen
    profile's resolve_z_touch(), same as every other generator here (plots
    on the bare plate at that pen's normal calibrated touch height; the
    missing paper thickness is accounted for by eye on the first mark).

    Two-pass generation (no linesort, ever -- this is the one thing this
    generator does differently from every sibling in this file): pass 1
    emits ALL candidate marks and is parsed back (_parse_ordered_strokes)
    to learn each mark's ACTUAL machine coordinates without re-deriving the
    vertical_flip/translate math by hand; any mark with a point outside the
    active reach polygon is recorded in `skipped`, one entry per offending
    group, with the coordinate. Pass 2 re-emits only the surviving marks to
    `out_gcode` and is what validate_gcode_xy_motion/
    validate_calibration_gcode_safety actually check. Unlike
    api_calibration_combo, this route must never silently relocate the paper
    reference, so offset_x/offset_y are used exactly as given.

    Edge-reference reachability: the unusable punched-hole strip may sit in a
    band the angled Preppy attachment can never reach (pen Y capped at 163.2mm
    on the active profile), so requiring BOTH off-sheet edge-reference lines to
    be reachable would block that configuration. Rule: **at least one**
    edge-reference line must be reachable (the remaining one plus the known
    paper dimensions are sufficient to register the sheet in Y); if literally
    zero of ALL candidate marks (edges + hole circles) are reachable, this
    refuses -- there would be nothing left to mark. Any skipped mark (edge
    reference or hole circle) is NEVER silently dropped: it is named in the
    returned `skipped` list AND in an ASCII `; SKIPPED (unreachable with active
    pen): <name>` comment block at the head of the emitted G-code itself, so
    the missing datum is visible before the plate is marked."""
    ppmm = 1.0 / PX2MM
    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()
    if profile is None:
        profile = active_pen_profile()
    # Per-pen reach clamp: same conservative-by-construction intersection as
    # composite.py -- a profile with no reach_clamp gets this
    # active_reach_profile() object back unchanged.
    reach = pen_reach_profile(active_reach_profile(cal), profile)
    z_touch, z_touch_source = resolve_z_touch(cal, profile)
    print(f"  [z] registration marks z_touch_abs={z_touch:.2f}mm (source={z_touch_source})")
    pen_off_x, pen_off_y = resolve_pen_offset(cal, profile)
    offset_x, offset_y = float(offset_x), float(offset_y)
    comp_off_x, comp_off_y = offset_x - pen_off_x, offset_y - pen_off_y
    park_x, park_y = reach["safe_park_point"]
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))
    z_up_abs = z_touch + float(z_up)
    z_down_abs = z_touch + float(z_down)

    vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({
        "z_up": float(z_up), "z_up_abs": z_up_abs, "z_down": float(z_down),
        "z_down_abs": z_down_abs, "z_touch_abs": z_touch, "z_travel_abs": z_travel_abs,
        "draw_speed": float(draw_speed), "travel_speed": float(travel_speed),
        "plunge_speed": float(plunge_speed), "lift_speed": float(lift_speed),
        "z_auto": "1", "origin_x": park_x, "origin_y": park_y,
        **_gcode_lifecycle_values(end_policy, quiet_fans)})

    all_groups = _reg_mark_groups(paper_w, paper_h)

    def _emit(groups, svg_path, gcode_path):
        polys = [poly for g in groups for poly in g["polys"]]
        # The references deliberately extend beyond the sheet. Shift them into
        # a taller SVG canvas so vpype pagesize cannot crop them. The matching
        # translate compensation below keeps y=0..paper_h
        # at exactly the pre-existing physical paper placement.
        y_pad_mm = REG_REFERENCE_EDGE_OFFSET_MM
        W = int(round(float(paper_w) * ppmm)) + 1
        H = int(round((float(paper_h) + 2.0 * y_pad_mm) * ppmm)) + 1
        polys_px = [[(x * ppmm, (y + y_pad_mm) * ppmm) for x, y in p] for p in polys]
        _write_paths_svg(svg_path, W, H, float(pen_mm) * ppmm, polylines=polys_px)
        # NO linesort -- gcode emission order must equal `polys` order 1:1 so
        # _parse_ordered_strokes can assign strokes back to their group.
        vpype_cli.execute(f'read {shlex.quote(str(svg_path))} pagesize {W}x{H}px '
                           f'translate -- {comp_off_x}mm {-comp_off_y - y_pad_mm}mm '
                           f'gwrite --profile elegoo_cc {shlex.quote(str(gcode_path))}')
        return W, H

    # Pass 1: everything, to learn ACTUAL machine coordinates per group.
    prov_svg = WORK_DIR / f"{out_gcode.stem}_prov_pattern.svg"
    prov_gcode = WORK_DIR / f"{out_gcode.stem}_prov.gcode.tmp"
    _emit(all_groups, prov_svg, prov_gcode)
    prov_text = prov_gcode.read_bytes().decode("ascii")
    strokes = _parse_ordered_strokes(prov_text, z_up_abs)
    poly_counts = [len(g["polys"]) for g in all_groups]
    if len(strokes) != sum(poly_counts):
        raise RuntimeError(
            f"Registration-marks provisional parse mismatch: expected {sum(poly_counts)} "
            f"strokes (one per input polyline), parsed {len(strokes)} -- gwrite output order "
            f"assumption (no linesort) may not hold.")

    reach_poly = reach["polygon"]
    corners_machine, skipped, kept_groups = {}, [], []
    si = 0
    for g, n in zip(all_groups, poly_counts):
        group_strokes = strokes[si:si + n]
        si += n
        bad = None
        for stroke_points in group_strokes:
            for px, py in stroke_points:
                if not point_in_polygon((px, py), reach_poly):
                    bad = (px, py)
                    break
            if bad:
                break
        if g.get("mandatory"):
            corners_machine[g["name"]] = {"first_point_xy": list(group_strokes[0][0]),
                                            "reachable": bad is None}
        if bad is None:
            kept_groups.append(g)
        else:
            skipped.append({"name": g["name"],
                             "reason": f"point X{bad[0]:.3f} Y{bad[1]:.3f} is outside reach "
                                       f"profile '{reach['label']}'"})

    skipped_names = {s["name"] for s in skipped}
    mandatory_names = [g["name"] for g in all_groups if g.get("mandatory")]
    reachable_mandatory = [n for n in mandatory_names if n not in skipped_names]
    if not reachable_mandatory:
        for p in (prov_svg, prov_gcode):
            try:
                p.unlink()
            except OSError:
                pass
        raise ValueError("at least one off-sheet edge-reference line must fit the active pen "
                         f"reach; unreachable: {', '.join(mandatory_names)}")
    if not kept_groups:
        for p in (prov_svg, prov_gcode):
            try:
                p.unlink()
            except OSError:
                pass
        raise ValueError("no registration marks fit the active pen reach")

    # Pass 2: only the reachable marks -- this is the FINAL, real output.
    svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    tmp_gcode = WORK_DIR / f"{out_gcode.stem}.gcode.tmp"
    W, H = _emit(kept_groups, svg, tmp_gcode)
    with atomic_target(out_gcode) as final_tmp:
        final_tmp.write_bytes(tmp_gcode.read_bytes())
        _fix_gwrite_source_comment(final_tmp, out_gcode)
        # Skipped-datum disclosure: a mark that is dropped for being unreachable
        # must never be silently missing -- name it in an ASCII comment block
        # ahead of the first motion command so the missing datum is visible
        # before marking the plate. Inserted before the first "G21" (the
        # document_start template's first real command, vpype.toml) -- pure
        # comment lines, never touches the motion stream.
        if skipped:
            data = final_tmp.read_bytes()
            skip_block = "".join(
                f"; SKIPPED (unreachable with active pen): {s['name']} ({s['reason']})\n"
                for s in skipped).encode("ascii")
            marker = b"G21\n"
            idx = data.find(marker)
            data = (data[:idx] + skip_block + data[idx:]) if idx != -1 else (skip_block + data)
            final_tmp.write_bytes(data)
    for p in (prov_svg, prov_gcode, tmp_gcode):
        try:
            p.unlink()
        except OSError:
            pass

    validate_gcode_xy_motion(out_gcode, reach)
    safety = validate_calibration_gcode_safety(out_gcode, z_down_abs)

    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(svg))} color {PREVIEW_COLOR} '
                           f'write {shlex.quote(str(tmp_prev))}')

    machine_bbox = _gcode_pendown_bbox(out_gcode, z_down_abs) if kept_groups else None
    return {"paper_mm": [float(paper_w), float(paper_h)], "offset_mm": [offset_x, offset_y],
            "reference_edge_offset_mm": REG_REFERENCE_EDGE_OFFSET_MM,
            "reference_x_mm": [REG_REFERENCE_X0_MM, REG_REFERENCE_X1_MM],
            "hole_reference_radius_mm": REG_HOLE_REFERENCE_RADIUS_MM,
            "hole_reference_centers_paper_mm": [[x, float(paper_h) - REG_HOLE_REFERENCE_CENTER_FROM_BOTTOM_MM]
                                                for x in REG_HOLE_REFERENCE_CENTERS_X_MM],
            "polylines": sum(len(g["polys"]) for g in kept_groups),
            "corners_machine": corners_machine, "skipped": skipped,
            "machine_bbox": (list(machine_bbox) if machine_bbox else None),
            "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
            "z_up_abs": round(z_up_abs, 3), "z_down_abs": round(z_down_abs, 3),
            "safety_selfcheck": safety,
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
            "svg_wh_px": [W, H], "preview": prev.name, "svg_path": str(svg)}


def _splice_pressure_blocks(text_a: str, text_b: str, label_a: str, label_b: str) -> str:
    """Combine two full elegoo_cc gwrite outputs (each independently a
    complete, valid file: preamble + body + postamble) into ONE file with a
    single shared preamble/postamble and both bodies back to back, block A
    FIRST. Relies on this profile's document_start/document_end templates
    (vpype.toml) referencing only z_up/z_travel/origin/speed/fan-lifecycle
    values -- NEVER z_down (see vpype_gcode's gwrite: document_start/
    document_end are filled from `default_values` and never see per-segment
    z_down/z_down_abs, which only appear in `segment_first`). So as long as
    both blocks were generated with identical z_up, speeds, offsets, origin,
    end_policy and quiet_fans (true here -- only z_down/z_down_abs differ
    between the normal and double-pressure calls), text_a and text_b share a
    byte-identical preamble AND postamble. Finding the longest common leading
    and trailing RUN OF LINES locates those shared blocks without depending
    on vpype's internal template strings, and raises loudly if that
    assumption doesn't hold instead of silently emitting a malformed file."""
    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()
    n_pref = 0
    while (n_pref < len(lines_a) and n_pref < len(lines_b)
           and lines_a[n_pref] == lines_b[n_pref]):
        n_pref += 1
    n_suf = 0
    while (n_suf < len(lines_a) - n_pref and n_suf < len(lines_b) - n_pref
           and lines_a[len(lines_a) - 1 - n_suf] == lines_b[len(lines_b) - 1 - n_suf]):
        n_suf += 1
    if n_pref == 0 or n_suf == 0:
        raise RuntimeError(
            "Pressure-block splice failed: the two blocks' gwrite output shares no "
            "common preamble/postamble lines -- one of the assumed-identical "
            "parameters (z_up, speeds, offsets, origin, end_policy/quiet_fans) must "
            "have differed between the normal- and double-pressure calls.")
    preamble = lines_a[:n_pref]
    body_a = lines_a[n_pref: len(lines_a) - n_suf]
    body_b = lines_b[n_pref: len(lines_b) - n_suf]
    postamble = lines_a[len(lines_a) - n_suf:]
    combined = preamble + [label_a] + body_a + [label_b] + body_b + postamble
    return "\n".join(combined) + "\n"


def _wedge_gcode(out_gcode: Path, pen_mm=0.25, z_up=3.5, z_down=-0.6,
                  draw_speed=3000, travel_speed=8000, plunge_speed=500, lift_speed=2000,
                  offset_x=None, offset_y=None, end_policy="release", quiet_fans=True,
                  profile=None, pressure_mult=PRESSURE_MULT_DEFAULT,
                  pressure_compare=PRESSURE_COMPARE_DEFAULT):
    """Gray step-wedge calibration plot. Default: a SINGLE block at the pen's
    normal z_down (pressure made little difference to tone on the pens tested).
    Pass `pressure_compare=True` to opt into the two-block layout -- an upper
    block at normal z_down and a lower block at `pressure_mult` x z_down
    (default 2.0x) -- for a pen that might show pressure sensitivity.

    Spacings come from build_wedge_polylines(fill_mm) -- pen-relative, see
    that function's docstring and the WEDGE_LO_MULT/WEDGE_HI_MULT comments
    above. Z heights come from the ACTIVE PEN PROFILE via resolve_z_touch(),
    exactly like _calibration_gcode. In pressure_compare mode the
    normal-pressure block plots FIRST (so the machine can be watched before
    the harder pass begins, and aborted if the tip digs) and is clearly
    marked from the double-pressure block via block_id_marks (1 tick vs 2,
    extending the same tick-numbering idiom the chart already uses) plus an
    ASCII gcode comment ahead of each block's body. In single-block mode
    (default) there is only one block/one z_down, so no block-id ticks are
    drawn -- there is nothing to disambiguate. Routed through
    validate_gcode_xy_motion + validate_calibration_gcode_safety (Z floor =
    the single block's z_down_abs, or the MINIMUM z_down_abs of the two
    blocks -- the deeper/harder one -- in pressure_compare mode; see the call
    below) so it cannot emit an unsafe file (ASCII-only, no G92, no bad fan
    index, every Z >= that floor)."""
    ppmm = 1.0 / PX2MM

    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()
    if profile is None:
        profile = active_pen_profile()
    # Per-pen reach clamp: same conservative-by-construction intersection as
    # composite.py -- a profile with no reach_clamp gets this
    # active_reach_profile() object back unchanged.
    reach = pen_reach_profile(active_reach_profile(cal), profile)
    fill_mm = _resolve_fill_mm(profile)

    # --- geometry: one chart built once. Single-block mode (default) draws
    # it once, unmarked; pressure_compare stacks two copies vertically
    # (normal pressure on top, double pressure below), each with a block-id
    # tick mark. ---
    base_polys, patch_rects = build_wedge_polylines(fill_mm)
    xs = [p[0] for poly in base_polys for p in poly]
    ys = [p[1] for poly in base_polys for p in poly]
    chart_min_x, chart_min_y = min(xs), min(ys)
    chart_h = max(ys) - chart_min_y
    polys_a = [[(x - chart_min_x, y - chart_min_y) for x, y in poly] for poly in base_polys]

    if pressure_compare:
        y_shift = chart_h + BLOCK_GAP
        polys_b = [[(x, y + y_shift) for x, y in poly] for poly in polys_a]
        polys_a = polys_a + block_id_marks(0.0, -7.0, 1)              # 1 tick: NORMAL
        polys_b = polys_b + block_id_marks(0.0, y_shift - 7.0, 2)      # 2 ticks: DOUBLE
    else:
        polys_b = []

    all_polys = polys_a + polys_b
    xs = [p[0] for poly in all_polys for p in poly]
    ys = [p[1] for poly in all_polys for p in poly]
    layout_w = max(xs) - min(xs)
    layout_h = max(ys) - min(ys)
    min_x, min_y = min(xs), min(ys)
    pad = 3.0

    def _norm(polys):
        return [[(x - min_x + pad, y - min_y + pad) for x, y in poly] for poly in polys]

    polys_a_s, polys_b_s = _norm(polys_a), _norm(polys_b)
    W = int(round((layout_w + 2 * pad) * ppmm))
    H = int(round((layout_h + 2 * pad) * ppmm))
    polys_a_px = [[(x * ppmm, y * ppmm) for x, y in p] for p in polys_a_s]
    polys_b_px = [[(x * ppmm, y * ppmm) for x, y in p] for p in polys_b_s]

    # Per-block pattern SVG(s) sharing the SAME canvas size (W,H) so the
    # translate/offset math applied to each gwrite call below is identical --
    # in pressure_compare mode that identical-offset property is what makes
    # both blocks' preamble/postamble byte-identical for
    # _splice_pressure_blocks.
    svg_a = WORK_DIR / f"{out_gcode.stem}_pattern_normal.svg"
    _write_paths_svg(svg_a, W, H, float(pen_mm) * ppmm, polylines=polys_a_px)
    if pressure_compare:
        svg_b = WORK_DIR / f"{out_gcode.stem}_pattern_double.svg"
        _write_paths_svg(svg_b, W, H, float(pen_mm) * ppmm, polylines=polys_b_px)
    # Combined pattern SVG, same naming convention (`{stem}_pattern.svg`)
    # _calibration_gcode uses -- this is what /api/calibration_import loads
    # to position the full plot (single block, or both blocks in
    # pressure_compare mode) as one drawing item.
    svg_full = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_paths_svg(svg_full, W, H, float(pen_mm) * ppmm, polylines=polys_a_px + polys_b_px)

    z_touch, z_touch_source = resolve_z_touch(cal, profile)
    print(f"  [z] gray wedge z_touch_abs={z_touch:.2f}mm (source={z_touch_source})")
    pen_off_x, pen_off_y = resolve_pen_offset(cal, profile)
    # Default inboard placement margin. _calibration_gcode uses the RAW
    # `bbox + 20.0` as offset_x/offset_y and only then subtracts pen_off_*
    # to get comp_off_x/comp_off_y (the value that actually places content in
    # machine coords) -- fine there because that plot is short (~24mm) and
    # pen_off_x is tiny (~-1mm), so the realized margin barely differs from
    # the nominal 20mm. In pressure_compare mode this wedge is TWO stacked
    # pressure blocks (~170mm+ tall) and pen_off_y is NOT tiny (-45mm on this
    # machine), so reusing that formula unmodified leaves an effectively
    # ~65mm realized margin (20 - pen_off_y) that eats most of the confirmed
    # reach's 212mm Y span before any content is even drawn -- caught this
    # via validate_gcode_xy_motion raising on the two-block plot, not
    # assumed. Fix: fold the pen offset into the DEFAULT here (not into the
    # shared comp_off_x/comp_off_y formula below, which stays identical to
    # every other generator) so the realized margin (comp_off_x/comp_off_y)
    # is actually the nominal PLACEMENT_MARGIN_MM, same intent as the shared
    # convention, just computed correctly for content this tall. Applied
    # unconditionally (single-block content is only ~half as tall, so this
    # still lands well within reach -- verified below) rather than branching
    # on pressure_compare, keeping ONE placement formula for both modes.
    # Only applies to offset_y (Y is the tight axis here); X keeps the
    # original formula since pen_off_x's effect on it is negligible.
    PLACEMENT_MARGIN_MM = 20.0
    offset_x = reach["bbox"][0] + PLACEMENT_MARGIN_MM if offset_x is None else float(offset_x)
    offset_y = (reach["bbox"][1] + PLACEMENT_MARGIN_MM + pen_off_y) if offset_y is None else float(offset_y)
    comp_off_x, comp_off_y = offset_x - pen_off_x, offset_y - pen_off_y
    park_x, park_y = reach["safe_park_point"]
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))

    z_down_norm = float(z_down)
    z_down_abs_norm = z_touch + z_down_norm

    base_defaults = {
        "z_up": float(z_up), "z_up_abs": z_touch + float(z_up), "z_touch_abs": z_touch,
        "z_travel_abs": z_travel_abs,
        "draw_speed": float(draw_speed), "travel_speed": float(travel_speed),
        "plunge_speed": float(plunge_speed), "lift_speed": float(lift_speed),
        "z_auto": "1", "origin_x": park_x, "origin_y": park_y,
        **_gcode_lifecycle_values(end_policy, quiet_fans)}

    tmp_a = WORK_DIR / f"{out_gcode.stem}_block_normal.gcode.tmp"

    vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({
        **base_defaults, "z_down": z_down_norm, "z_down_abs": z_down_abs_norm})
    vpype_cli.execute(f'read {shlex.quote(str(svg_a))} pagesize {W}x{H}px linesort '
                       f'translate -- {comp_off_x}mm {-comp_off_y}mm '
                       f'gwrite --profile elegoo_cc {shlex.quote(str(tmp_a))}')

    if pressure_compare:
        z_down_double = float(z_down) * float(pressure_mult)
        z_down_abs_double = z_touch + z_down_double
        tmp_b = WORK_DIR / f"{out_gcode.stem}_block_double.gcode.tmp"
        vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({
            **base_defaults, "z_down": z_down_double, "z_down_abs": z_down_abs_double})
        vpype_cli.execute(f'read {shlex.quote(str(svg_b))} pagesize {W}x{H}px linesort '
                           f'translate -- {comp_off_x}mm {-comp_off_y}mm '
                           f'gwrite --profile elegoo_cc {shlex.quote(str(tmp_b))}')

        text_a = tmp_a.read_bytes().decode("ascii")
        text_b = tmp_b.read_bytes().decode("ascii")
        label_a = (f"; PRESSURE BLOCK 1/2: NORMAL  z_down={z_down_norm:.2f}mm "
                   f"z_down_abs={z_down_abs_norm:.2f}mm -- plots FIRST (top of sheet)")
        label_b = (f"; PRESSURE BLOCK 2/2: DOUBLE  z_down={z_down_double:.2f}mm "
                   f"z_down_abs={z_down_abs_double:.2f}mm ({pressure_mult:.2f}x normal) -- "
                   f"plots SECOND (bottom of sheet)")
        combined_text = _splice_pressure_blocks(text_a, text_b, label_a, label_b)
        atomic_write_text(out_gcode, combined_text, encoding="ascii", newline="\n")
        try:
            tmp_a.unlink()
            tmp_b.unlink()
        except OSError:
            pass
        # Z floor is the MINIMUM z_down_abs across BOTH blocks, i.e. the
        # deeper (double-pressure) one -- that block plunges further below
        # paper contact than the normal block, so it is the one that
        # actually sets the pen-into-bed hazard boundary for this file.
        # Using only the normal block's z_down_abs here would pass
        # vacuously (every double-pressure Z would read as "below floor"
        # and still slip through unchecked).
        z_down_abs_floor = min(z_down_abs_norm, z_down_abs_double)
    else:
        text_a = tmp_a.read_bytes().decode("ascii")
        atomic_write_text(out_gcode, text_a, encoding="ascii", newline="\n")
        try:
            tmp_a.unlink()
        except OSError:
            pass
        z_down_abs_double = None
        # Single block: the ONLY z_down_abs in the file IS the floor. No
        # stale/vacuous "double" value is consulted here.
        z_down_abs_floor = z_down_abs_norm

    validate_gcode_xy_motion(out_gcode, reach)
    safety = validate_calibration_gcode_safety(out_gcode, z_down_abs_floor)

    prev = WORK_DIR / "wedge_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(svg_full))} color {PREVIEW_COLOR} '
                           f'write {shlex.quote(str(tmp_prev))}')

    place = {"x": offset_x, "y": offset_y, "w": layout_w + 2 * pad, "h": layout_h + 2 * pad}
    if pressure_compare:
        patch_rects_all = ([{**pr, "block": "normal"} for pr in patch_rects]
                            + [{**pr, "block": "double"} for pr in patch_rects])
    else:
        patch_rects_all = [{**pr, "block": "normal"} for pr in patch_rects]
    result = {"pen_mm": pen_mm, "fill_mm": fill_mm,
              "pattern_mm": [layout_w + 2 * pad, layout_h + 2 * pad],
              "offset_mm": [offset_x, offset_y],
              "polylines": len(polys_a) + len(polys_b), "preview": prev.name,
              "place": place, "patch_rects": patch_rects_all,
              "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
              "z_up_abs": round(z_touch + float(z_up), 3),
              # Backward-compatible single value (what index.html reads today):
              # the normal block's draw Z, i.e. the plot's primary/first Z.
              "z_down_abs": round(z_down_abs_norm, 3),
              "z_down_abs_normal": round(z_down_abs_norm, 3),
              "z_down_abs_double": round(z_down_abs_double, 3) if z_down_abs_double is not None else None,
              "z_down_abs_floor": round(z_down_abs_floor, 3),
              "pressure_mult": float(pressure_mult),
              "pressure_compare": bool(pressure_compare),
              "safety_selfcheck": safety,
              "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
              "svg_wh_px": [W, H],
              "svg_path": str(svg_full), "svg_path_normal": str(svg_a),
              "svg_path_double": str(svg_b) if pressure_compare else None}
    return result
