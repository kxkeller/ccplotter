"""ccplotter/calibration_fragment.py -- fragment dose-response sheet.

What it measures: the BIC "dimples" (less-inked centres inside contour fills that the
render predicts SOLID) are a deposition effect, not geometry. Ring cores are only a minor
contributor, and neither the connector router (Stage E) nor the pocket_link join is the
main lever. The hypothesis this sheet tests is FRAGMENTATION: short pieces, many pen-down
starts and direction reversals keep a ballpoint from rolling long enough to transfer ink,
so the inside of a dense fill under-inks. The sheet DOSES that one variable and nothing
else:

  every patch is a SOLID patch filled at the BIC pitch (profile fill_mm, 0.12),
  every row abuts, so commanded coverage is identical across arms and only the
  fragment structure differs. A dose-response (not a yes/no) is the readout.

HARD-CODED MACHINE AND PEN (limitation). The sheet is tied to the author's measured
machine: EXPECT_REACH_ID ("plotter_2003_v32"), EXPECT_BBOX and the Z / speed constants
are imported from calibration_pocket and the generator refuses any other reach profile.
The pen is PINNED to bic_cristal_blue (fill_mm must be exactly 0.12; z_touch is READ
from the profile, never hardcoded). Placement, order and jitter are seeded.

THE ARMS (patch = 6.0 mm wide x 49 rows; rows at pitch p, row 0 at -24p, local
frame centre-origin, rotated by theta = 0 / 90 deg by replicate):
  N   null        ONE continuous boustrophedon polyline (1 pen-down, 0 lifts).
  R   rows        each row its own stroke, alternate direction (49 pen-downs):
                  the f=0 anchor of the density ladder.
  A1  L<len>      PIECE-LENGTH LADDER: every row cut into abutting pieces of
                  0.3 / 0.5 / 1 / 2 / 4 mm, each its own pen-down. Cut phases
                  are staggered per row (golden-ratio) so seams never align into
                  a column; end pieces are partial, so the realised mean piece
                  length is reported per patch, not assumed.
  A3  F<pct>      OPEN-START DENSITY at FIXED piece length 0.5 mm: a fraction
                  f (1/8, 1/4, 1/2) of rows is cut into 0.5 mm pieces, the rest
                  stay whole rows. f=0 is R, f=1 is A1/L0.5. Start density varies
                  with piece length held constant -- the ladder varies both.
  A4  V<len>      REVERSALS-ONLY: one continuous polyline, NO lifts, in vertical
                  strips of width w (0.3 .. 4 mm); inside a strip the path
                  serpentines through all rows, so the direction reverses every w
                  mm of travel; strips chain with zero-length joins. Same row
                  union as every other arm; reversal count ~ piece count of the
                  A1 ladder at L=w, pen-down count 1.
  A5  P<len>      BRIDGED PAIRS (nearest honest version of "Stage E ON"): the A1
                  piece at L in row 2m and its twin in row 2m+1 are fused by a
                  one-pitch pen-down connector into a U, so pen-downs are HALVED
                  and replaced by tight reversals -- what route_low_lift does to
                  short hops. Compare with A1 at the same L. NOT the real Stage E
                  (that acts on a traced mask).
  A2  D*          POCKET_LINK EQUIVALENT on a 6 mm disc (nested rings at
                  rho_k = r-(k+.5)p, the production contour rule): D0 = separate
                  closed ring strokes (link 0), D3 = rings chained by pen-down
                  connectors of length <= 3p (link 3, asserted), DN = shape-
                  matched continuous zigzag disc (null for the disc family).
                  Ring k+1 starts 2p of arc further round than ring k in BOTH
                  arms, so the only difference is the join.
  A6  S3          today's contour fill, square: nested square rings chained like
                  link 3 (production default pocket_link 3.0).
  H<m>            POSITIVE CONTROL: the null with a commanded bare centre (m
                  rows either side of centre removed, x cut half-width (m+1)p).
                  Bare centre ~2.0 / 6.9 / 14.7 / 25 % of the centre window at
                  laid 0.20 (computed from a true-width raster, not asserted
                  from the formula; per_arm_stats.csv has 0.10-0.24).

DIFFERENCES FROM PRODUCTION:
  * synthetic closed-form geometry, not traced (calibration_pocket convention).
  * A2/A6 are an EQUIVALENT of pocket_link: the connectors are radial/diagonal
    segments of one pitch to a few pitch, not the production nearest-point
    greedy chain on traced polygons; D3 is an equivalent, not the real knob.
  * A5 is an equivalent of Stage E (connector_mask_check), not the real knob.
  * MICROLIFT IS APPLIED (as production does): hops < 8 mm lift only to
    z_touch + z_micro, hops > mega_hop_mm get the mega lift. Other registry sheets
    historically had none; without it this sheet would test a slower, higher-lift
    pen than the one that plots real jobs. Stats report how many lifts were
    short.
  * linesimplify/linesort are NOT applied (exact straight rows; reordering would
    destroy the dose).

HOW TO READ IT (per patch, from a 600 dpi scan): centre-window ink density vs a band
inset 0.75-1.25 mm from the patch edge; windows are defined in WINDOWS below and
repeated in the layout JSON. A fall in centre/band density as pieces shorten or
reversals increase is the fragmentation effect; the H<m> patches calibrate how a known
bare centre reads on the same scale.
"""
from __future__ import annotations

import json
import math
import random
import shlex
from fractions import Fraction
from pathlib import Path

import numpy as np
import vpype as vp
import vpype_cli
from shapely.geometry import box

from ccplotter.atomic_io import atomic_target
from ccplotter.config import CONFIG_PATH, PREVIEW_COLOR, PX2MM, WORK_DIR
from ccplotter.cal import (resolve_pen_offset, load_cal, load_pen_profiles, resolve_z_touch,
                           resolve_z_micro)
from ccplotter.gcode_post import (_gcode_lifecycle_values, validate_calibration_gcode_safety,
                                  _microlift_gcode)
from ccplotter.reach import active_reach_profile, validate_gcode_xy_motion, pen_reach_profile
from ccplotter.calibration_deposition import _splice_blocks
from ccplotter import calibration_pocket as _pk
from ccplotter.calibration_pocket import (
    EXPECT_REACH_ID, EXPECT_BBOX, EXPECT_Z_DOWN_STOCK, EXPECT_DRAW_SPEED_STOCK,
    Z_TOUCH_SANITY_RANGE_MM, TIP_X_MIN, TIP_X_MAX, TIP_Y_MIN, TIP_Y_MAX,
    X0_TIP_DEFAULT, Y_FRONT_TIP_DEFAULT, SHEET_W, SHEET_H, FID_SIDE_MM, FID_PITCH_MAX_FRAC_OF_PEN,
    FID0_LEN_MM, FID90_LEN_MM, GRID_X0, FAST, MIN_STROKE_MM,
    _poly_length_mm, _contour_rho_list, _circle_n, _square_rows, _square_rings, _ticks, _glyph,
    _write_svg_hp, _prepend_ascii_header, _parse_strokes_f, _check_clearance, check_strokes,
)

_PAD_MM = 3.0

PEN_ID = "bic_cristal_blue"
PEN_LABEL = "BIC Cristal 1.0 Blue"
PEN_MM = 1.0
PITCH_MM = 0.12                      # locked (BIC fill_mm); row count below depends on it

# ---------------------------------------------------------------------------
# Sheet design constants. Changing them changes the dose structure and invalidates
# comparison with earlier scans.
# ---------------------------------------------------------------------------
PATCH_W_MM = 6.0
N_ROWS = 49                          # odd: strips/pairs chain with zero connectors
PATCH_H_MM = N_ROWS * PITCH_MM       # 5.88
DISC_R_MM = 3.0
PHI = 0.6180339887498949             # golden-ratio phase step for piece seams
MIN_PIECE_MM = 0.05
LADDER_MM = (0.3, 0.5, 1.0, 2.0, 4.0)
STRIP_MM = (0.3, 0.5, 1.0, 2.0, 4.0)
PAIR_MM = (0.5, 1.0)
DENSITY_FRACS = (Fraction(1, 8), Fraction(1, 4), Fraction(1, 2))
DENSITY_PIECE_MM = 0.5
HOLE_ROWS = (1, 2, 3, 4)             # rows removed either side of centre
N_REPS = 4                           # ladder/null/arms: theta 0,90,0,90
N_REPS_HOLE = 2
RING_SHIFT_P = 2.0                   # arc shift per ring, in pitches
LINK_MAX_P = 3.0                     # connector length cap (pocket_link default 3.0 x fill)

# Readout windows (patch-local mm, centre origin). Chebyshev for square patches,
# Euclidean for disc patches. Band is inset 0.75-1.25 mm from the 3.0 mm half-edge.
WINDOWS = {"centre_half_mm": 1.0, "band_inner_mm": 1.75, "band_outer_mm": 2.25,
           "square_metric": "chebyshev", "disc_metric": "euclidean"}
LAID_PREVIEW_MM = (0.16, 0.20, 0.24)

LABEL_CAP_MM = 2.0                   # printed label height
LABEL_FONT = "futural"
LABEL_GAP_MM = 1.0                   # label bottom to patch top

GRID_COLS, GRID_ROWS = 15, 7
CELL_W_MM, CELL_H_MM = 13.5, 15.5
GRID_Y0 = 11.0
JITTER_MM = 0.6

# Pitch-transfer probe (generator option, OFF by default): N and L03 at the
# production ring pitch's low end. Nominal 0.085 mm; production ring pitch is about
# 0.066-0.109 mm.
PROBE_PITCH_MM = 0.085
PROBE_N_REPS = 4

SEED_PLACEMENT = 20261003
SEED_ORDER = 20261004
SEED_JITTER = 20261005

PREDICTION_HEADER = [
    "FRAGMENT SHEET (D906): solid patches at BIC pitch, dose of fragmentation only.",
    "Hypothesis, decision rule, controls PRE-REGISTERED in artifacts/fragment-sheet-2026-10-03/PREREG.md.",
    "Read ONLY with the pre-registered windows; capture per docs/CAPTURE_RULE.md (D686).",
]


# ---------------------------------------------------------------------------
# Patch builders -- local frame, centre origin, y down (layout convention).
# Each returns a list of polylines (pen-down strokes in emission order).
# ---------------------------------------------------------------------------
def _nrows(p: float = PITCH_MM) -> int:
    """Row count for pitch p. The locked 0.12 mm pitch keeps N_ROWS = 49 EXACTLY (the
    base sheet is unchanged); any other pitch (the 0.085 probe) gets the ODD
    count nearest to PATCH_H_MM / p, so strips/pairs would still chain with zero
    connectors (69 rows at 0.085 -> 5.865 mm tall)."""
    if abs(p - PITCH_MM) < 1e-9:
        return N_ROWS
    n = int(round(PATCH_H_MM / p))
    return n if n % 2 == 1 else n + 1


def _ys(p: float = PITCH_MM):
    n = _nrows(p)
    return [(i - (n - 1) / 2.0) * p for i in range(n)]


def _dir_row(y, x0, x1, i):
    return [(x0, y), (x1, y)] if i % 2 == 0 else [(x1, y), (x0, y)]


def build_null(p: float = PITCH_MM):
    """ONE continuous boustrophedon polyline."""
    h = PATCH_W_MM / 2.0
    poly = []
    for i, y in enumerate(_ys(p)):
        poly += _dir_row(y, -h, h, i)
    return [poly]


def build_rows(p: float = PITCH_MM):
    h = PATCH_W_MM / 2.0
    return [_dir_row(y, -h, h, i) for i, y in enumerate(_ys(p))]


def _segments(x0: float, x1: float, L: float, phase: float):
    """Abutting pieces of nominal length L with the first cut at x0+phase
    (phase in [0, L)); a cut within MIN_PIECE_MM of an end is dropped."""
    cuts, c = [], x0 + (phase if phase > 1e-9 else L)
    while c < x1 - 1e-9:
        if c > x0 + MIN_PIECE_MM and c < x1 - MIN_PIECE_MM:
            cuts.append(c)
        c += L
    edges = [x0] + cuts + [x1]
    return [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]


def _phase(i: int, L: float) -> float:
    return ((i * PHI) % 1.0) * L


def build_ladder(L: float, frac: Fraction = Fraction(1), p: float = PITCH_MM):
    """Rows cut into pieces of nominal length L for the rows selected by `frac`
    (all rows when frac == 1); the other rows stay whole. Row i alternates
    direction; pieces are emitted in travel order, each oriented along it."""
    h = PATCH_W_MM / 2.0
    n, d = frac.numerator, frac.denominator
    out = []
    for i, y in enumerate(_ys(p)):
        fragmented = ((i + 1) * n) // d > (i * n) // d
        if not fragmented:
            out.append(_dir_row(y, -h, h, i))
            continue
        segs = _segments(-h, h, L, _phase(i, L))
        if i % 2 == 1:
            segs = [(b, a) for a, b in reversed(segs)]
        for a, b in segs:
            out.append([(a, y), (b, y)])
    return out


def build_pairs(L: float, p: float = PITCH_MM):
    """Bridged pairs: piece in row 2m and its twin in row 2m+1 fused into a U by
    a one-pitch pen-down connector. A leftover odd row stays a plain ladder row."""
    h = PATCH_W_MM / 2.0
    ys = _ys(p)
    nr = len(ys)
    out = []
    for m in range(nr // 2):
        ya, yb = ys[2 * m], ys[2 * m + 1]
        segs = _segments(-h, h, L, _phase(m, L))
        rev = (m % 2 == 1)
        if rev:
            segs = list(reversed(segs))
        for a, b in segs:
            if not rev:
                out.append([(a, ya), (b, ya), (b, yb), (a, yb)])
            else:
                out.append([(b, ya), (a, ya), (a, yb), (b, yb)])
    i = nr - 1                                         # leftover (last) row
    segs = _segments(-h, h, L, _phase(i, L))
    for a, b in segs:
        out.append([(a, ys[i]), (b, ys[i])])
    return out


def build_strips(w: float, p: float = PITCH_MM):
    """Reversals-only: ONE polyline, no lifts. Vertical strips of width w; inside
    a strip the path serpentines through every row (down for even strips, up for
    odd), so the direction reverses every w mm of travel. Odd row count makes
    each strip end where the next begins (zero-length join)."""
    h = PATCH_W_MM / 2.0
    ys = _ys(p)
    edges = [-h]
    c = -h + w
    while c < h - 1e-9:
        if c < h - MIN_PIECE_MM:
            edges.append(c)
        c += w
    edges.append(h)
    poly = []
    for j in range(len(edges) - 1):
        a, b = edges[j], edges[j + 1]
        order = range(len(ys)) if j % 2 == 0 else range(len(ys) - 1, -1, -1)
        for i in order:
            seg = [(a, ys[i]), (b, ys[i])] if i % 2 == 0 else [(b, ys[i]), (a, ys[i])]
            for q in seg:
                if not poly or abs(poly[-1][0] - q[0]) > 1e-9 or abs(poly[-1][1] - q[1]) > 1e-9:
                    poly.append(q)
    return [poly]


def build_hole(m: int, p: float = PITCH_MM):
    """Positive control: the null with a commanded bare centre. Rows with
    |y| <= m*p are cut over |x| < (m+1)*p; a cut row becomes two strokes."""
    h = PATCH_W_MM / 2.0
    xc = (m + 1) * p
    out, cur = [], []
    for i, y in enumerate(_ys(p)):
        if abs(y) <= m * p + 1e-9:
            if i % 2 == 0:                              # travel L -> R
                first, second = (-h, -xc), (xc, h)
            else:                                       # travel R -> L
                first, second = (h, xc), (-xc, -h)
            cur += [(first[0], y), (first[1], y)]
            out.append(cur)
            cur = [(second[0], y), (second[1], y)]
        else:
            cur += _dir_row(y, -h, h, i)
    out.append(cur)
    return out


def _disc_ring(rho: float, start_rad: float):
    n = _circle_n(rho)
    pts = [(rho * math.cos(start_rad + 2 * math.pi * i / n),
            rho * math.sin(start_rad + 2 * math.pi * i / n)) for i in range(n)]
    pts.append(pts[0])
    return pts


def _ring_starts(rhos, p):
    starts, th = [], -math.pi / 2.0
    for k, rho in enumerate(rhos):
        starts.append(th)
        nxt = rhos[k + 1] if k + 1 < len(rhos) else rho
        th += min(RING_SHIFT_P * p / max(nxt, 1e-6), math.pi / 3.0)
    return starts


def build_disc_rings(link: bool, p: float = PITCH_MM):
    """Nested disc rings (production contour rule). link=False: separate closed
    strokes; link=True: chained with pen-down connectors <= LINK_MAX_P * p."""
    rhos = _contour_rho_list(DISC_R_MM, p)
    starts = _ring_starts(rhos, p)
    rings = [_disc_ring(r, s) for r, s in zip(rhos, starts)]
    if not link:
        return rings
    chain = list(rings[0])
    for k in range(1, len(rings)):
        a, b = chain[-1], rings[k][0]
        if math.hypot(b[0] - a[0], b[1] - a[1]) > LINK_MAX_P * p + 1e-9:
            raise RuntimeError("fragment: ring connector exceeds the link cap")
        chain += rings[k]
    return [chain]


def build_disc_null(p: float = PITCH_MM):
    """Shape-matched null for the disc family: one continuous zigzag disc, rows at
    pitch clipped to the stroke-centre disc (r - p/2), same inset as the rings."""
    rho0 = DISC_R_MM - p / 2.0
    poly = []
    for i, y in enumerate(_ys(p)):
        if abs(y) > rho0:
            continue
        hl = math.sqrt(max(rho0 * rho0 - y * y, 0.0))
        poly += _dir_row(y, -hl, hl, i)
    return [poly]


def build_square_rings_link3(p: float = PITCH_MM):
    """Today's contour fill on a square: nested square rings chained (link 3)."""
    chain, k = [], 0
    while True:
        hs = PATCH_W_MM / 2.0 - (k + 0.5) * p
        if hs < 0.02:
            break
        ring = [(-hs, -hs), (hs, -hs), (hs, hs), (-hs, hs), (-hs, -hs)]
        if chain:
            a, b = chain[-1], ring[0]
            if math.hypot(b[0] - a[0], b[1] - a[1]) > LINK_MAX_P * p + 1e-9:
                raise RuntimeError("fragment: square connector exceeds the link cap")
        chain += ring
        k += 1
    return [chain]


# ---------------------------------------------------------------------------
# Statistics and raster readout (used by the generator, the preview script and
# the tests; one definition).
# ---------------------------------------------------------------------------
REV_CONNECTOR_MAX_P = 1.5            # segments <= 1.5 pitches are turnaround connectors


def _unit(a, b):
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    return None if L < 1e-12 else ((b[0] - a[0]) / L, (b[1] - a[1]) / L)


def _main_dirs(s, p=PITCH_MM):
    """Unit directions of the 'main' segments of a polyline (connector steps of
    <= REV_CONNECTOR_MAX_P pitches are skipped, so a boustrophedon turnaround --
    two 90 degree corners and a one-pitch step -- counts as ONE reversal)."""
    out = []
    for i in range(len(s) - 1):
        dx, dy = s[i + 1][0] - s[i][0], s[i + 1][1] - s[i][1]
        L = math.hypot(dx, dy)
        if L > REV_CONNECTOR_MAX_P * p + 1e-9:
            out.append((dx / L, dy / L))
    return out


def patch_stats(strokes, p: float = PITCH_MM):
    """Per-patch fragment statistics from a list of pen-down polylines.
    reversals_in_stroke : anti-parallel consecutive main segments inside a stroke.
    reversals_between   : anti-parallel last-main-segment -> next stroke's first
                          main segment across each lift (a lifted reversal).
    reversals_total     : their sum -- the quantity the A4 arm doses without lifts."""
    n = len(strokes)
    draw = sum(_poly_length_mm(s) for s in strokes)
    rev_in, firsts, lasts = 0, [], []
    for s in strokes:
        d = _main_dirs(s, p)
        for a, b in zip(d, d[1:]):
            if a[0] * b[0] + a[1] * b[1] < -0.5:
                rev_in += 1
        firsts.append(_unit(s[0], s[1]))              # ANY length: a 0.1 mm piece still
        lasts.append(_unit(s[-2], s[-1]))             # has a direction across the lift
    rev_between = 0
    for la, fb in zip(lasts, firsts[1:]):
        if la is not None and fb is not None and la[0] * fb[0] + la[1] * fb[1] < -0.5:
            rev_between += 1
    return {"pieces": n, "pendown_starts": n, "lifts": max(n - 1, 0),
            "draw_mm": round(draw, 3), "mean_piece_mm": round(draw / n, 4) if n else 0.0,
            "reversals_in_stroke": rev_in, "reversals_between_strokes": rev_between,
            "reversals_total": rev_in + rev_between}


def stamp_strokes(img, strokes, laid_mm, ppmm, x0_mm=0.0, y0_mm=0.0):
    """Draw polylines onto a uint8 image at their TRUE laid width: every segment is a
    filled rectangle of width laid_mm, every vertex a filled circle (round caps and
    joins), sub-pixel coordinates (shift 4). cv2.polylines(thickness=t) is NOT used:
    it draws 2*(t//2)+1 pixels, so t=5 and t=6 render identically and a 'laid 0.24'
    raster would silently equal laid 0.28.
    Pixel (i, j) covers x0_mm + [i, i+1)/ppmm."""
    import cv2
    sh = 4
    k = float(1 << sh)
    r_mm = laid_mm / 2.0

    def px(q):
        return ((q[0] - x0_mm) * ppmm * k, (q[1] - y0_mm) * ppmm * k)

    rr = int(round(r_mm * ppmm * k - 0.5 * k))     # cv2 fills edge-inclusive: -half a pixel
    for s in strokes:
        pts = [px(q) for q in s]
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy)
            if L < 1e-9:
                continue
            nx, ny = -dy / L * rr, dx / L * rr
            quad = np.array([[ax + nx, ay + ny], [bx + nx, by + ny],
                             [bx - nx, by - ny], [ax - nx, ay - ny]]).round().astype(np.int32)
            cv2.fillPoly(img, [quad.reshape(-1, 1, 2)], 255, cv2.LINE_8, sh)
        for (ax, ay) in pts:
            cv2.circle(img, (int(round(ax)), int(round(ay))), rr, 255, -1, cv2.LINE_8, sh)
    return img


def raster_patch(strokes, laid_mm, ppmm=25, half_mm=4.0, ss=4):
    """Float [0,1] ink-coverage image of one patch (local frame, centre origin) with
    strokes at the given laid width, rendered at ss x supersampling and area-averaged
    down to ppmm px per mm (square pixels). ss=4 makes 0.12/0.16/0.20/0.24 mm exact
    integer pixel widths (12/16/20/24 px at 100 px/mm)."""
    import cv2
    n = int(round(2 * half_mm * ppmm))
    big = np.zeros((n * ss, n * ss), np.uint8)
    stamp_strokes(big, strokes, laid_mm, ppmm * ss, x0_mm=-half_mm, y0_mm=-half_mm)
    small = cv2.resize(big, (n, n), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def window_density(img, shape, ppmm=25, half_mm=4.0, win=WINDOWS):
    """(centre_density, band_density, centre_px, band_px) of an ink image."""
    n = img.shape[0]
    c = (np.arange(n) + 0.5) / ppmm - half_mm
    X, Y = np.meshgrid(c, c)
    r = np.maximum(np.abs(X), np.abs(Y)) if shape == "square" else np.hypot(X, Y)
    cm = r <= win["centre_half_mm"]
    bm = (r >= win["band_inner_mm"]) & (r <= win["band_outer_mm"])
    return float(img[cm].mean()), float(img[bm].mean()), int(cm.sum()), int(bm.sum())


# ---------------------------------------------------------------------------
# Item table, placement
# ---------------------------------------------------------------------------
def _fmt_len(x):
    return f"{int(round(x * 10)):02d}"


def _item_table(probe: bool = False):
    """Each item: id, label (<= 6 ASCII chars), arm, shape, builder spec, rep.
    The probe items are appended LAST so every base item keeps its cell and
    jitter (placement and jitter are seeded draws consumed in item order)."""
    items = []

    def add(arm, code, shape, spec, n_reps, params):
        for rep in range(n_reps):
            items.append({"arm": arm, "code": code, "shape": shape, "spec": spec,
                          "rep": rep, "theta_deg": 0.0 if rep % 2 == 0 else 90.0,
                          "params": params,
                          "label": f"{code}-{rep + 1}", "id": f"{code}-{rep + 1}"})

    add("A0_null", "N", "square", ("null",), N_REPS, {"lifts": 0})
    add("A0_rows", "R", "square", ("rows",), N_REPS, {"f": 0.0})
    for m in HOLE_ROWS:
        add("A0_hole", f"H{m}", "square", ("hole", m), N_REPS_HOLE,
            {"rows_removed_each_side": m, "x_cut_half_mm": round((m + 1) * PITCH_MM, 4)})
    for L in LADDER_MM:
        add("A1_ladder", f"L{_fmt_len(L)}", "square", ("ladder", L), N_REPS, {"piece_mm": L})
    for f in DENSITY_FRACS:
        add("A3_density", f"F{int(round(float(f) * 100)):02d}", "square", ("density", f), N_REPS,
            {"f": float(f), "piece_mm": DENSITY_PIECE_MM})
    for w in STRIP_MM:
        add("A4_reversals", f"V{_fmt_len(w)}", "square", ("strips", w), N_REPS, {"strip_mm": w})
    for L in PAIR_MM:
        add("A5_pairs", f"P{_fmt_len(L)}", "square", ("pairs", L), N_REPS, {"piece_mm": L})
    add("A2_disc_null", "DN", "disc", ("disc_null",), N_REPS, {})
    add("A2_disc_link0", "D0", "disc", ("disc_rings", False), N_REPS, {"pocket_link_equiv": 0})
    add("A2_disc_link3", "D3", "disc", ("disc_rings", True), N_REPS, {"pocket_link_equiv": 3})
    add("A6_sq_link3", "S3", "square", ("sq_link3",), N_REPS, {"pocket_link_equiv": 3})
    if probe:
        n0 = len(items)
        add("A7_probe_null", "NP", "square", ("null",), PROBE_N_REPS, {"lifts": 0})
        add("A7_probe_L03", "L03P", "square", ("ladder", 0.3), PROBE_N_REPS, {"piece_mm": 0.3})
        for it in items[n0:]:
            it["pitch_mm"] = PROBE_PITCH_MM
            it["params"] = dict(it["params"], pitch_mm=PROBE_PITCH_MM,
                                n_rows=_nrows(PROBE_PITCH_MM))
    return items


def build_patch(item, p: float = PITCH_MM):
    p = item.get("pitch_mm", p)
    s = item["spec"]
    k = s[0]
    if k == "null":
        return build_null(p)
    if k == "rows":
        return build_rows(p)
    if k == "hole":
        return build_hole(s[1], p)
    if k == "ladder":
        return build_ladder(s[1], Fraction(1), p)
    if k == "density":
        return build_ladder(DENSITY_PIECE_MM, s[1], p)
    if k == "strips":
        return build_strips(s[1], p)
    if k == "pairs":
        return build_pairs(s[1], p)
    if k == "disc_null":
        return build_disc_null(p)
    if k == "disc_rings":
        return build_disc_rings(s[1], p)
    if k == "sq_link3":
        return build_square_rings_link3(p)
    raise ValueError(k)


def hole_commanded(ppmm=25):
    """Commanded delta = 1 - centre/band density of each hole control, by laid width
    (the quantity the readout measures; the reader's gain_H regresses on it)."""
    out = {}
    for laid in LAID_PREVIEW_MM:
        d = {}
        for m in HOLE_ROWS:
            c, b, _, _ = window_density(raster_patch(build_hole(m), laid, ppmm=ppmm), "square", ppmm=ppmm)
            d[f"H{m}"] = round(1.0 - c / b, 5)
        out[f"{laid:.2f}"] = d
    return out


def _assign_cells(items):
    n_cells = GRID_COLS * GRID_ROWS
    if len(items) > n_cells:
        raise RuntimeError(f"fragment: {len(items)} items > {n_cells} cells")
    rng = random.Random(SEED_PLACEMENT)
    cells = list(range(n_cells))
    rng.shuffle(cells)
    jr = random.Random(SEED_JITTER)
    for it, cell in zip(items, cells):
        c, r = cell % GRID_COLS, cell // GRID_COLS
        it["cell"] = [c, r]
        it["cx"] = round(GRID_X0 + (c + 0.5) * CELL_W_MM + jr.uniform(-JITTER_MM, JITTER_MM), 4)
        it["cy"] = round(GRID_Y0 + (r + 0.5) * CELL_H_MM + jr.uniform(-JITTER_MM, JITTER_MM), 4)
    return sorted(set(range(n_cells)) - set(cells[:len(items)]))


def _emission_order(n):
    order = list(range(n))
    random.Random(SEED_ORDER).shuffle(order)
    return order


def _label_scale():
    from vpype import text as _vt
    b = _vt.text_line("H", LABEL_FONT, 1.0).bounds()
    return LABEL_CAP_MM / (b[3] - b[1])


def label_strokes(text: str, cx: float, y_bottom: float):
    """Hershey 'futural' label, ASCII only, cap height LABEL_CAP_MM (scale fixed
    from the glyph 'H', so all labels share one size), baseline at y_bottom (layout
    y down: the label sits ABOVE y_bottom), horizontally centred on cx."""
    from vpype import text as _vt
    if not text.isascii():
        raise ValueError("label must be ASCII")
    scale = _label_scale()
    base = _vt.text_line("H", LABEL_FONT, 1.0).bounds()[3]     # baseline of 'H'
    lc = _vt.text_line(text, LABEL_FONT, 1.0)
    b = lc.bounds()
    xm = (b[0] + b[2]) / 2.0
    return [[((z.real - xm) * scale + cx, (z.imag - base) * scale + y_bottom) for z in ln]
            for ln in lc.lines]


def _rot_pts(pts, theta_deg, cx, cy):
    return _pk._rot(pts, theta_deg, cx, cy)


def build_layout(pitch_mm: float = PITCH_MM, probe: bool = False):
    if abs(pitch_mm - PITCH_MM) > 1e-9:
        raise RuntimeError(f"fragment: pitch {pitch_mm} != locked BIC fill_mm {PITCH_MM}")
    items = _item_table(probe)
    unused = _assign_cells(items)
    elements, records, geoms = [], [], {}

    def add(pls, kind, gid):
        a = len(elements)
        for pl in pls:
            if _poly_length_mm(pl) >= MIN_STROKE_MM - 1e-12:
                elements.append({"pts": [(float(x), float(y)) for x, y in pl],
                                 "speed": FAST, "kind": kind, "gid": gid})
        return [a, len(elements)]

    fixed = {}
    # The six registration fiducials are emitted FIRST (before the glyph, the
    # rulers, the prime and every patch), so they are laid by the freshest pen state
    # and the drift covariate (emission_index) never contains them.
    fid_pos = [(2.0, 2.0), (SHEET_W / 2 - 2.0, 2.0), (SHEET_W - 6.0, 2.0),
               (2.0, 148.0), (SHEET_W / 2 - 2.0, 148.0), (SHEET_W - 6.0, 148.0)]
    fid_records = []
    for i, (fx, fy) in enumerate(fid_pos):
        fp = min(pitch_mm, FID_PITCH_MAX_FRAC_OF_PEN * PEN_MM)
        pls = _square_rings(FID_SIDE_MM, fp, fx + FID_SIDE_MM / 2, fy + FID_SIDE_MM / 2)
        fixed[f"fid{i}"] = [[(fx, fy), (fx + FID_SIDE_MM, fy + FID_SIDE_MM)]]
        add(pls, "fiducial", f"fid{i}")
        fid_records.append({"id": f"fid{i}", "origin_mm": [fx, fy], "side_mm": FID_SIDE_MM,
                            "center_mm": [fx + FID_SIDE_MM / 2, fy + FID_SIDE_MM / 2],
                            "pitch_mm": fp, "n_rings": len(pls)})
    pls = _glyph(12.0, 1.0) + [[(24.0 + 3.0 * i, 1.0), (24.0 + 3.0 * i, 8.0)] for i in range(3)]
    fixed["glyph"] = pls
    add(pls, "glyph", "glyph")
    r0 = [[(GRID_X0, 158.0), (GRID_X0 + FID0_LEN_MM, 158.0)]] + _ticks(GRID_X0, 158.0, FID0_LEN_MM, "h", -1)
    r90 = [[(234.0, 22.0), (234.0, 22.0 + FID90_LEN_MM)]] + _ticks(234.0, 22.0, FID90_LEN_MM, "v", -1)
    fixed["ruler0"], fixed["ruler90"] = r0, r90
    add(r0, "ruler", "ruler0")
    add(r90, "ruler", "ruler90")
    prime = _square_rows(_pk.PRIME_SIDE_MM, _pk.PRIME_PITCH_MM, 227.0, 133.0)
    fixed["prime"] = prime
    add(prime, "prime", "prime")

    for emis, idx in enumerate(_emission_order(len(items))):
        it = items[idx]
        ip = it.get("pitch_mm", pitch_mm)
        cx, cy, th = it["cx"], it["cy"], it["theta_deg"]
        local = build_patch(it, pitch_mm)
        stats = patch_stats(local, ip)
        placed = [_rot_pts(pl, th, cx, cy) for pl in local]
        # label sits above the PATCH (rotated patch extents differ by < 0.2 mm)
        half = max(PATCH_W_MM, PATCH_H_MM) / 2.0
        lab = label_strokes(it["label"], cx, cy - half - LABEL_GAP_MM)
        rng_patch = add(placed, "patch", it["id"])
        rng_lab = add(lab, "label", it["id"] + "_label")
        allpts = [q for e in elements[rng_patch[0]:rng_lab[1]] for q in e["pts"]]
        bx0, by0 = min(q[0] for q in allpts), min(q[1] for q in allpts)
        bx1, by1 = max(q[0] for q in allpts), max(q[1] for q in allpts)
        labpts = [q for e in elements[rng_lab[0]:rng_lab[1]] for q in e["pts"]]
        lab_bbox = [round(min(q[0] for q in labpts), 3), round(min(q[1] for q in labpts), 3),
                    round(max(q[0] for q in labpts), 3), round(max(q[1] for q in labpts), 3)]
        geoms[it["id"]] = box(bx0, by0, bx1, by1)
        records.append({"id": it["id"], "label": it["label"], "arm": it["arm"], "code": it["code"],
                        "shape": it["shape"], "spec": [str(x) for x in it["spec"]],
                        "params": it["params"], "rep": it["rep"], "theta_deg": th,
                        "pitch_mm": ip, "n_rows": _nrows(ip),
                        "emission_index": emis, "pen_down_start_index": rng_patch[0],
                        "label_bbox_mm": lab_bbox,
                        "cell": it["cell"], "center_mm": [cx, cy],
                        "bbox_mm": [round(bx0, 3), round(by0, 3), round(bx1, 3), round(by1, 3)],
                        "elem_range_patch": rng_patch, "elem_range_label": rng_lab,
                        "n_label_strokes": rng_lab[1] - rng_lab[0], **stats})

    meta = {"unused_cells": unused, "fiducials": fid_records, "n_elements": len(elements),
            "probe": bool(probe),
            "unused_cell_centres_mm": [[round(GRID_X0 + ((c % GRID_COLS) + 0.5) * CELL_W_MM, 3),
                                        round(GRID_Y0 + ((c // GRID_COLS) + 0.5) * CELL_H_MM, 3)]
                                       for c in unused]}
    return elements, records, fixed, geoms, meta


def arm_table(records):
    """Aggregate per-patch stats to one row per arm/code (means; n)."""
    by = {}
    for r in records:
        by.setdefault((r["arm"], r["code"]), []).append(r)
    rows = []
    for (arm, code), rs in sorted(by.items()):
        def mean(k):
            return round(sum(x[k] for x in rs) / len(rs), 3)
        rows.append({"arm": arm, "code": code, "n": len(rs), "pieces": mean("pieces"),
                     "pendown_starts": mean("pendown_starts"), "lifts": mean("lifts"),
                     "reversals_in_stroke": mean("reversals_in_stroke"),
                     "reversals_between_strokes": mean("reversals_between_strokes"),
                     "reversals_total": mean("reversals_total"), "draw_mm": mean("draw_mm"),
                     "mean_piece_mm": mean("mean_piece_mm")})
    return rows


# ---------------------------------------------------------------------------
# Time models (labelled; NEITHER is stopwatch-validated).
# ---------------------------------------------------------------------------
def time_model_a(elements, plunge_speed=500, lift_speed=2000, z_up=3.5):
    t = sum(_poly_length_mm(e["pts"]) / max(1.0, e["speed"] / 60.0) for e in elements)
    return t + len(elements) * (abs(z_up) / (plunge_speed / 60.0) + abs(z_up) / (lift_speed / 60.0))


def time_model_b(gcode_path):
    """Per-move length / modal feed over the EMITTED gcode, Z included, no
    acceleration, no dwell, no firmware lookahead. A floor, not a forecast."""
    import re
    pos = {"X": None, "Y": None, "Z": None}
    f = 3000.0
    t = 0.0
    for line in Path(gcode_path).read_text(encoding="ascii").splitlines():
        if not line.startswith(("G0", "G1")):
            continue
        mf = re.search(r"F(-?[\d.]+)", line)
        if mf:
            f = float(mf.group(1))
        d2 = 0.0
        for ax in "XYZ":
            m = re.search(ax + r"(-?[\d.]+)", line)
            if m:
                v = float(m.group(1))
                if pos[ax] is not None:
                    d2 += (v - pos[ax]) ** 2
                pos[ax] = v
        t += math.sqrt(d2) / max(1.0, f / 60.0)
    return t


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------
def fragment_gcode(out_gcode: Path, pen_mm=None, z_up=3.5, travel_speed=8000, plunge_speed=500,
                   lift_speed=2000, offset_x=None, offset_y=None, end_policy="release",
                   quiet_fans=True, profile=None, microlift=True, probe=False):
    ppmm = 1.0 / PX2MM
    offset_x_given = offset_x is not None
    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()
    profiles = load_pen_profiles()
    if PEN_ID not in profiles.get("profiles", {}):
        raise RuntimeError(f"pen profile {PEN_ID!r} not found in pen_profiles.json")
    pen = profiles["profiles"][PEN_ID]
    if profile is not None and profile != pen:
        raise RuntimeError(
            f"fragment: refusing to generate -- this sheet is PINNED to {PEN_ID!r} "
            f"({PEN_LABEL!r}); the requested/active profile is {profile.get('label')!r}. "
            f"Mount {PEN_ID} and re-request, or POST with pen_id={PEN_ID!r} explicitly.")

    reach0 = active_reach_profile(cal)
    if reach0["id"] != EXPECT_REACH_ID or tuple(round(v, 1) for v in reach0["bbox"]) != EXPECT_BBOX:
        raise RuntimeError(f"Runtime reach envelope mismatch: got id={reach0['id']!r} "
                           f"bbox={reach0['bbox']}, expected {EXPECT_REACH_ID!r} {EXPECT_BBOX}.")
    if pen.get("label") != PEN_LABEL:
        raise RuntimeError(f"pen {PEN_ID!r} label mismatch: {pen.get('label')!r} != {PEN_LABEL!r}")
    if abs(float(pen.get("pen_mm", 0.0)) - PEN_MM) > 1e-6:
        raise RuntimeError(f"pen {PEN_ID!r} pen_mm mismatch: {pen.get('pen_mm')} != {PEN_MM}")
    got_off = (float(pen.get("pen_offset_x", 0.0)), float(pen.get("pen_offset_y", 0.0)))
    if any(abs(a) > 1e-6 for a in got_off):
        raise RuntimeError(f"pen {PEN_ID!r} pen_offset mismatch: {got_off} != (0, 0)")
    z_touch, z_touch_source = resolve_z_touch(cal, pen)
    lo, hi = Z_TOUCH_SANITY_RANGE_MM
    if not (lo <= z_touch <= hi):
        raise RuntimeError(f"z_touch_abs for {PEN_ID} resolved to {z_touch}, outside [{lo},{hi}] mm.")
    z_down = float(pen.get("z_down", 0.0))
    if abs(z_down - EXPECT_Z_DOWN_STOCK) > 1e-6:
        raise RuntimeError(f"stock z_down mismatch for {PEN_ID}: {z_down} != {EXPECT_Z_DOWN_STOCK}")
    draw_speed = float(pen.get("draw_speed", 0.0))
    if abs(draw_speed - EXPECT_DRAW_SPEED_STOCK) > 1e-6 or draw_speed != FAST:
        raise RuntimeError(f"stock draw_speed mismatch for {PEN_ID}: {draw_speed} != {FAST}")
    pitch_mm = float(pen.get("fill_mm", 0.0))
    if abs(pitch_mm - PITCH_MM) > 1e-9:
        raise RuntimeError(f"fill_mm for {PEN_ID} = {pitch_mm}; this sheet is built for the locked "
                           f"{PITCH_MM} mm BIC pitch (row count, windows) -- refusing "
                           f"(the fill slider is set; Reset it for this sheet).")

    reach = pen_reach_profile(reach0, pen)
    if pen_mm is None:
        pen_mm = PEN_MM
    pen_off_x, pen_off_y = resolve_pen_offset(cal, pen)

    elements, records, fixed_groups, geoms, meta = build_layout(pitch_mm, probe=bool(probe))
    clearance_check = _check_clearance(geoms, fixed_groups)

    pad = _PAD_MM
    U = 25.4 / 96.0
    S = ppmm * U
    W = int(round((SHEET_W + 2 * pad) * ppmm))
    H = int(round((SHEET_H + 2 * pad) * ppmm))
    canvas_w, canvas_h = SHEET_W + 2 * pad, SHEET_H + 2 * pad
    flip_axis = max(W, H) * U
    offset_x = (X0_TIP_DEFAULT - pad * S) if offset_x is None else float(offset_x)
    default_placement = (not offset_x_given) and offset_y is None
    if offset_y is None:
        offset_y = Y_FRONT_TIP_DEFAULT + (pad + SHEET_H) * S - flip_axis
    else:
        offset_y = float(offset_y)
    comp_off_x, comp_off_y = offset_x - pen_off_x, offset_y - pen_off_y

    def expect_xy(x, y):
        return ((pad + x) * S + comp_off_x, flip_axis + comp_off_y - (pad + y) * S)

    def tip_xy(x, y):
        cx, cy = expect_xy(x, y)
        return (cx + pen_off_x, cy + pen_off_y)

    allx = [q[0] for e in elements for q in e["pts"]]
    ally = [q[1] for e in elements for q in e["pts"]]
    tips = [tip_xy(min(allx), min(ally)), tip_xy(max(allx), max(ally))]
    tx = [t[0] for t in tips]
    ty = [t[1] for t in tips]
    if min(tx) < TIP_X_MIN or max(tx) > TIP_X_MAX or min(ty) < TIP_Y_MIN or max(ty) > TIP_Y_MAX:
        raise RuntimeError(
            f"fragment: tip window X[{min(tx):.1f},{max(tx):.1f}] Y[{min(ty):.1f},{max(ty):.1f}] "
            f"outside the paper window X[{TIP_X_MIN},{TIP_X_MAX}] Y[{TIP_Y_MIN},{TIP_Y_MAX}] "
            f"(B5 landscape, holes at the FRONT; frame moved per D1487).")
    park_x, park_y = reach["safe_park_point"]
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))
    z_down_abs = z_touch + z_down
    z_up_abs = z_touch + float(z_up)

    placed_all = [[((x + pad) * ppmm, (y + pad) * ppmm) for x, y in e["pts"]] for e in elements]

    blk_svg = WORK_DIR / f".{out_gcode.stem}_blk.svg"
    blk_gc = WORK_DIR / f".{out_gcode.stem}_blk.gcode"
    _write_svg_hp(blk_svg, W, H, float(pen_mm) * ppmm, placed_all)
    vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({
        "z_up": float(z_up), "z_up_abs": z_up_abs,
        "z_down": z_down, "z_down_abs": z_down_abs,
        "z_touch_abs": z_touch, "z_travel_abs": z_travel_abs,
        "draw_speed": float(FAST), "travel_speed": float(travel_speed),
        "plunge_speed": float(plunge_speed), "lift_speed": float(lift_speed),
        "z_auto": "1", "origin_x": park_x, "origin_y": park_y,
        **_gcode_lifecycle_values(end_policy, quiet_fans)})
    vpype_cli.execute(f'read {shlex.quote(str(blk_svg))} pagesize {W}x{H}px '
                      f'translate -- {comp_off_x}mm {-comp_off_y}mm '
                      f'gwrite --profile elegoo_cc {shlex.quote(str(blk_gc))}')
    texts = [blk_gc.read_text(encoding="utf-8")]
    labels = [f"; BLOCK 1/1 draw_speed=F{FAST} elements=0..{len(elements) - 1}"]
    blk_svg.unlink(missing_ok=True)
    combined = _splice_blocks(texts, labels, z_up_abs)
    combined = combined.replace(str(blk_gc), str(out_gcode))
    blk_gc.unlink(missing_ok=True)

    header = list(PREDICTION_HEADER) + ([
        f"PROBE VARIANT (D965): + N and L03 at pitch {PROBE_PITCH_MM} mm, n={PROBE_N_REPS} each "
        f"({_nrows(PROBE_PITCH_MM)} rows); secondary pre-registration pending the owner's choice."
    ] if probe else []) + [
        f"pen_id={PEN_ID} z_touch_abs={z_touch:.3f} ({z_touch_source}) z_down_abs={z_down_abs:.3f}",
        f"pitch_mm={pitch_mm} (profile fill_mm) draw F{FAST} strokes={len(elements)} "
        f"microlift={'on' if microlift else 'off'}",
        f"seeds placement={SEED_PLACEMENT} order={SEED_ORDER} jitter={SEED_JITTER}",
        "RECORD z_touch before plotting; BIC=18.95 / uRight=11.45 differ by 7.50 mm (DIRECTION S10).",
    ]
    with atomic_target(out_gcode) as tmp_gcode:
        Path(tmp_gcode).write_text(combined, encoding="utf-8")
        _prepend_ascii_header(tmp_gcode, header)

    # Production-fidelity micro/mega lifts (composite.py convention), applied to the
    # emitted file BEFORE the guards so the guards read what the printer will read.
    microlift_stats = None
    z_micro, z_micro_source = resolve_z_micro(cal, pen)
    z_short_abs = z_touch + z_micro
    mega_mm = float(cal.get("mega_hop_mm", 70.0))
    micro_hop_mm = float(pen.get("micro_hop_mm", cal.get("micro_hop_mm", 8.0)))
    do_micro = bool(microlift) and z_up_abs > z_short_abs + 0.05
    do_mega = bool(microlift) and mega_mm > 0 and z_travel_abs > z_up_abs + 0.05
    if do_micro or do_mega:
        microlift_stats = _microlift_gcode(
            out_gcode, z_up_abs,
            z_short_abs=(z_short_abs if do_micro else None), short_mm=micro_hop_mm,
            z_mega_abs=(z_travel_abs if do_mega else None), long_mm=mega_mm)
        microlift_stats["micro_hop_mm"] = micro_hop_mm if do_micro else None
        microlift_stats["z_micro_source"] = z_micro_source

    validate_gcode_xy_motion(out_gcode, reach)
    safety = validate_calibration_gcode_safety(out_gcode, z_down_abs)
    strokes = _parse_strokes_f(out_gcode, z_down_abs)
    stroke_check = check_strokes(strokes, elements, expect_xy)
    selftest_caught = _pk.selftest()

    total_mm = sum(_poly_length_mm(s["pts"]) for s in strokes)
    t_a = time_model_a(elements, plunge_speed, lift_speed, z_up)
    t_b = time_model_b(out_gcode)

    prev_svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_svg_hp(prev_svg, W, H, float(pen_mm) * ppmm, placed_all)
    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(prev_svg))} color {PREVIEW_COLOR} '
                          f'write {shlex.quote(str(tmp_prev))}')

    layout = {
        "sheet": "fragment", "pen_id": PEN_ID, "pitch_mm": pitch_mm,
        "probe": bool(probe), "probe_pitch_mm": PROBE_PITCH_MM if probe else None,
        "probe_n_rows": _nrows(PROBE_PITCH_MM) if probe else None,
        "emission_order_note": "fiducials, glyph, rulers, prime first; then patches in a seeded "
                               "shuffle of ALL items (a larger item set re-shuffles: probe and base "
                               "emission orders differ); emission_index is the patch order.",
        "patch_w_mm": PATCH_W_MM, "n_rows": N_ROWS, "patch_h_mm": round(PATCH_H_MM, 4),
        "disc_r_mm": DISC_R_MM, "windows": WINDOWS, "laid_preview_mm": list(LAID_PREVIEW_MM),
        "ladder_mm": list(LADDER_MM), "strip_mm": list(STRIP_MM), "pair_mm": list(PAIR_MM),
        "density_fracs": [str(f) for f in DENSITY_FRACS], "density_piece_mm": DENSITY_PIECE_MM,
        "hole_rows": list(HOLE_ROWS), "n_reps": N_REPS, "n_reps_hole": N_REPS_HOLE,
        "seeds": {"placement": SEED_PLACEMENT, "order": SEED_ORDER, "jitter": SEED_JITTER},
        "grid": {"cols": GRID_COLS, "rows": GRID_ROWS, "cell_w_mm": CELL_W_MM,
                 "cell_h_mm": CELL_H_MM, "x0": GRID_X0, "y0": GRID_Y0, "jitter_mm": JITTER_MM,
                 "unused_cells": meta["unused_cells"],
                 "unused_cell_centres_mm": meta["unused_cell_centres_mm"]},
        "sheet_wh_mm": [SHEET_W, SHEET_H], "pad_mm": pad,
        "strokes_bbox_mm": [round(min(allx), 3), round(min(ally), 3),
                            round(max(allx), 3), round(max(ally), 3)],
        "hole_commanded_delta": hole_commanded(),
        "mapping": {"offset_x": offset_x, "offset_y": offset_y, "flip_axis_mm": flip_axis,
                    "comp_off_x": comp_off_x, "comp_off_y": comp_off_y,
                    "pen_off_x": pen_off_x, "pen_off_y": pen_off_y,
                    "cmd_x": "pad + x + comp_off_x", "cmd_y": "flip + comp_off_y - (pad + y)",
                    "tip_x": "cmd_x + pen_off_x", "tip_y": "cmd_y + pen_off_y"},
        "tip_window_mm": {"x": [min(tx), max(tx)], "y": [min(ty), max(ty)]},
        "paper_window_mm": {"x": [TIP_X_MIN, TIP_X_MAX], "y": [TIP_Y_MIN, TIP_Y_MAX]},
        "z_touch_abs": z_touch, "z_touch_source": z_touch_source, "z_down_abs": z_down_abs,
        "z_up_abs": z_up_abs, "z_short_abs": z_short_abs, "microlift": microlift_stats,
        "items": records, "fiducials": meta["fiducials"],
        "rulers": {"h": {"origin_mm": [GRID_X0, 158.0], "len_mm": FID0_LEN_MM},
                   "v": {"origin_mm": [234.0, 22.0], "len_mm": FID90_LEN_MM},
                   "tick_pitch_mm": _pk.FID_TICK_PITCH_MM, "tick_h_mm": _pk.FID_TICK_H_MM,
                   "tick_dir": -1},
        "arm_table": arm_table(records),
        "stroke_check": stroke_check, "selftest_caught": selftest_caught,
        "clearance_check": clearance_check,
        "time_model_a_s": round(t_a, 1), "time_model_b_s": round(t_b, 1),
    }
    layout_path = WORK_DIR / f"{out_gcode.stem}_layout.json"
    layout_path.write_text(json.dumps(layout, indent=1), encoding="utf-8")

    return {"kind": "fragment", "pen_mm": pen_mm, "pen_id": PEN_ID, "pen_label": PEN_LABEL,
            "footprint_mm": [round(canvas_w, 1), round(canvas_h, 1)],
            "offset_mm": [offset_x, offset_y],
            "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
            "z_down": z_down, "z_down_abs": round(z_down_abs, 3), "draw_speed": draw_speed,
            "reach_id": reach0["id"], "reach_bbox": reach0["bbox"],
            "n_pendown_expected_total": len(elements),
            "n_pendown_actual_total": stroke_check["n_strokes"],
            "pendown_count_matches": stroke_check["n_strokes"] == len(elements),
            "deepest_z_down_abs": round(z_down_abs, 3),
            "safety_selfcheck": safety, "stroke_check": stroke_check,
            "clearance_check": clearance_check, "microlift": microlift_stats,
            "tip_window_mm": layout["tip_window_mm"], "default_placement": default_placement,
            "plot_time_estimate_s": round(t_b, 1),
            "plot_time_estimate_min": round(t_b / 60.0, 2),
            "plot_time_model_label": "model (b): per-move length / modal feed over the emitted gcode, "
                                     "Z included, microlift applied, no acceleration/dwell; a floor, "
                                     "NOT stopwatch-validated.",
            "plot_time_model_a_min": round(t_a / 60.0, 2),
            "preview": prev.name, "svg_path": str(prev_svg), "svg_wh_px": [W, H],
            "pattern_mm": [round(canvas_w, 1), round(canvas_h, 1)],
            "place": {"x": offset_x, "y": offset_y, "w": canvas_w, "h": canvas_h},
            "z_up_abs": round(z_up_abs, 3),
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
            "layout_path": str(layout_path), "total_draw_mm": round(total_mm, 1)}


def fragment_bic(out_gcode, **kw):
    return fragment_gcode(out_gcode, **kw)


def fragment_bic_probe(out_gcode, **kw):
    """Probe variant: the base sheet + N/L03 at pitch 0.085 (8 more patches). This is the
    generator the single `fragment_bic` registry entry uses; `fragment_bic` (base, 92
    patches) stays available for comparison and tests."""
    kw["probe"] = True
    return fragment_gcode(out_gcode, **kw)
