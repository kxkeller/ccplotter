"""ccplotter/calibration_pocket.py -- pocket test sheet.

What it measures: contour-filled regions that the render predicts SOLID can show light
"dimples" (~22 per cm2 in scans of BIC contour-fill sheets). They co-locate with
contour-fill ring cores (1.5-2x enrichment over a shifted null), and the commanded
geometry essentially does not contain them at any plausible laid width, so they are
DEPOSITION effects. Two mechanisms are possible and unseparated:
  (i)  ball starvation on tiny-radius SHORT closed core paths (a ballpoint
       transfers ink by rolling under drag; a ~2 mm core loop at radius
       ~0.33 mm < the 0.5 mm ball radius gives it ~40 ms of rolling);
  (ii) ring-convergence overdraw: the last k rings of a contour fill
       stack on each other inside one laid width, and re-passes lift ink.
This sheet is built so the two make DIFFERENT predictions:
  * SHAPE   : disc pockets (core path length ~ radius) vs capsule pockets
              (core path length ~ constant 2*SPINE_LEN at every radius).
  * SPEED   : the same contour pockets at F1500 and F3000.
  * ARM     : contour / spine / zigzag / hybrid at the SAME pitch.
  * PEN     : a Platinum Preppy twin on identical geometry (not a ball tip).

How to read it: scan the sheet, find light blobs inside each pocket's stroke-centre
region, and compare their rate across shapes, speeds and arms against the NEGATIVE
control squares (no commanded hole, no ring core, no convergence). The POSITIVE
discs have known bare cores and measure the detector's recall. The extra speed-arm
pockets (r >= 0.6 mm, 9 per arm, giving n = 24 per arm for the F3000 / F1500 contour
arms) fill otherwise unused grid cells.

HARD-CODED MACHINE (limitation). The sheet is tied to the author's measured machine: it
refuses to run unless the active reach profile is EXPECT_REACH_ID ("plotter_2003_v32") with
bbox EXPECT_BBOX, and the paper window and placement constants below are in that machine's
pen-TIP/plate frame. Several other sheets (calibration_fragment, calibration_photofe,
calibration_twopen) import these constants from here.

GEOMETRY IS BUILT DIRECTLY, NOT TRACED. Same rationale as calibration_solidpitch: every
stroke is emitted from closed-form formulas so the commanded geometry is exact and the arms
differ ONLY in the property under test. The sheet reproduces the production contour rule
`dt > (k+0.5)*pitch` analytically (ring k sits at inscribed radius rho_k = r-(k+0.5)*pitch,
emitted outer->inner, each ring its own pen-down closed stroke, vertices from the same
0.019 mm sagitta tolerance as trace_v3's approxPolyDP(0.4px)). `spine` and `hybrid` are
generated HERE, in this module only -- there is no trace_v3 fill mode behind them.

THE ARMS (pocket = capsule of half-width r and straight length 2*SPINE_LEN, or a
disc of radius r; r = INSCRIBED radius; stroke-centre region = pocket inset by
pitch/2, half-width rho0 = r - pitch/2):
  contour : rings rho_k = r-(k+.5)p while > 0, outer->inner.
  zigzag  : one continuous boustrophedon polyline, rows parallel to the long
            axis at pitch p starting at -rho0, clipped to the stroke-centre
            region (an extra last row at +rho0 if the remainder exceeds p/2).
  spine   : medial-axis stroke (offset 0) plus parallel open strokes at the
            ring-grid offsets +-(rho0 - m*p), emitted CENTRE-OUT, one pen-down
            each, no turnarounds, no closed cores. (capsule only; a disc's
            medial axis is a point.)
  hybrid  : contour rings while rho_k >= R_HYBRID_MM (a global inscribed-radius
            threshold in mm) and spine-style open rows for the core, on
            the continuing ring grid. For r < R_HYBRID_MM + p/2 it IS the
            spine arm (no ring is emitted).
  Every arm covers the same stroke-centre region at the same pitch; the equal-
  coverage readback (geometric bare fraction at three laid widths) is computed by an
  offline analysis, not asserted from the formulas.

DIFFERENCES FROM PRODUCTION (read before claiming anything transfers):
  * production rings are pixel-quantized (dt on the mask grid, 0.0474 mm/px) and
    are re-ordered/linked downstream; here they are analytic and ordered
    outer->inner. Closed-ring START angle is fixed per replicate.
  * the speed arm uses F1500 vs F3000 per-point F words (vpype `segment`); the
    plunge/lift feeds are unchanged.
  * order-within-pocket differs between arms by construction (rings outer->inner,
    spine centre-out); it is part of what each arm IS.

CONTROLS ON THE SHEET (so a null result cannot be an instrument failure):
  * NEGATIVE: 12 solid 5x5 mm squares drawn by zigzag at the pen's pitch -- the
    population the shifted-window null is drawn from (no commanded hole, no ring
    core, no convergence).
  * POSITIVE: 9 discs (r 1.5 mm, contour) deliberately stopped early to leave a
    bare core of 0.25 / 0.40 / 0.60 mm nominal diameter x 3 replicates -- known
    light blobs at known places, to measure detector recall.
  * a PRIME patch (6x6 mm, pitch 0.5) emitted first so the first measured stroke
    is not the pen's first stroke on paper. Not measured.
  * 6 SOLID FIDUCIAL SQUARES (4 mm, pen pitch, 6 positions, 3 noncollinear needed)
    sized to survive a dark threshold of 90, plus 0/90 deg tick rulers (10 mm).

PEN IS PINNED, NOT RESOLVED. `pocket_bic` -> bic_cristal_blue, `pocket_preppy` ->
platinum_preppy. A different requested/active profile raises. label / pen_mm /
pen_offset / z_down / draw_speed are asserted against constants; z_touch_abs and
fill_mm are READ from the profile and recorded (z_touch is the sheet's pinned Z,
asserted finite and inside a sanity range). Same convention as solidpitch.

PLACEMENT. Default placement is in the pen-TIP frame (paper = plate coordinates,
B5 JIS landscape 257x182, punch holes at the FRONT): layout x=0 -> tip X 10.0,
layout y=H (front) -> tip Y 14.0. The default is checked against the paper window
(8 mm side margins, hole keep-out 10 mm + 2 mm at the front, 3 mm margin at the back)
and the pen's reach; an override (offset_x/offset_y, same semantics as solidpitch) is
validated the same way and falls back through the dispatcher's own note if it does not fit.

Reached only through the generic /api/calibration_sheet dispatcher (registered in
calibration_registry.py).
"""
from __future__ import annotations

import json
import math
import random
import re
import shlex
from pathlib import Path

import vpype as vp
import vpype_cli
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

from ccplotter.atomic_io import atomic_target
from ccplotter.config import CONFIG_PATH, PREVIEW_COLOR, PX2MM, WORK_DIR
from ccplotter.cal import resolve_pen_offset, load_cal, load_pen_profiles, resolve_z_touch
from ccplotter.gcode_post import _gcode_lifecycle_values, validate_calibration_gcode_safety
from ccplotter.reach import active_reach_profile, validate_gcode_xy_motion, pen_reach_profile
from ccplotter.calibration_deposition import _splice_blocks

# ---------------------------------------------------------------------------
# Expected runtime envelope -- re-verified at generation time. Tied to the
# author's measured machine geometry.
# ---------------------------------------------------------------------------
EXPECT_REACH_ID = "plotter_2003_v32"
EXPECT_BBOX = (3.0, 42.5, 255.0, 254.5)
EXPECT_Z_DOWN_STOCK = -0.60
EXPECT_DRAW_SPEED_STOCK = 3000
Z_TOUCH_SANITY_RANGE_MM = (5.0, 25.0)

PEN_SPECS = {
    "bic_cristal_blue": {
        "label": "BIC Cristal 1.0 Blue", "pen_mm": 1.0, "offset": (0.0, 0.0),
        "short": "bic", "id_bars": 1,
    },
    "platinum_preppy": {
        "label": "Platinum Preppy (45deg angled mount, 2026-08-01)", "pen_mm": 0.25,
        "offset": (4.4, -6.2), "short": "preppy", "id_bars": 2,
    },
}

# ---------------------------------------------------------------------------
# Paper window (mm, pen-TIP frame == plate frame). JIS B5 257x182 landscape,
# punch holes at the FRONT, front long edge 1.5 mm from the plate front. Front
# keep-out is the 10 mm hole strip + 2 mm; the back is a plain 3 mm margin with no
# keep-out.
# ---------------------------------------------------------------------------
PAPER_W_MM, PAPER_H_MM = 257.0, 182.0
PAPER_FRONT_Y_MM = 1.5
HOLE_KEEPOUT_MM = 10.0
TIP_X_MIN, TIP_X_MAX = 8.0, 249.0           # >= 8 mm from either side edge
TIP_Y_MIN = PAPER_FRONT_Y_MM + HOLE_KEEPOUT_MM + 2.0     # 13.5
TIP_Y_MAX = PAPER_FRONT_Y_MM + PAPER_H_MM - 3.0          # 180.5
X0_TIP_DEFAULT = 10.0                       # layout x=0 -> tip X
Y_FRONT_TIP_DEFAULT = 14.0                  # layout y=H -> tip Y
_PAD_MM = 3.0
CLEARANCE_MM = 2.0                          # min gap between item outlines (ink
                                            # halo is laid/2 ~ 0.1-0.3 mm each side)

# ---------------------------------------------------------------------------
# Sheet design constants. Changing them changes the geometry under test and
# invalidates comparison with earlier scans.
# ---------------------------------------------------------------------------
RUNGS_MM = (0.30, 0.45, 0.60, 0.80, 1.00, 1.40, 2.00)   # inscribed radius r
N_REPS = 3
ORIENT_DEG = (0.0, 45.0, 90.0)                           # replicate k -> angle
SPINE_LEN_MM = 1.5                                       # capsule half-straight
R_HYBRID_MM = 0.45                                       # hybrid ring->spine rule
SAGITTA_MM = 0.019                                       # 0.4 px * 0.0474 mm/px
FAST, SLOW = 3000, 1500
BIC_PITCH_REF_MM = 0.12                                  # for the Preppy P012 arm
MIN_STROKE_MM = 0.05

# arm id -> (shape, mode, speed, pitch_override_mm or None, only_if_pitch_differs)
ARMS = (
    ("cap_contour_F3000", "capsule", "contour", FAST, None, False),
    ("cap_spine_F3000", "capsule", "spine", FAST, None, False),
    ("cap_zigzag_F3000", "capsule", "zigzag", FAST, None, False),
    ("cap_hybrid_F3000", "capsule", "hybrid", FAST, None, False),
    ("cap_contour_F1500", "capsule", "contour", SLOW, None, False),
    ("disc_contour_F3000", "disc", "contour", FAST, None, False),
    ("disc_contour_F1500", "disc", "contour", SLOW, None, False),
    ("disc_zigzag_F3000", "disc", "zigzag", FAST, None, False),
    ("cap_contour_P012_F3000", "capsule", "contour", FAST, BIC_PITCH_REF_MM, True),
)
# Extra speed-arm pockets, r >= 0.6 mm, 9 per arm (rungs 0.60/0.80/1.00/1.40 x2,
# rung 2.00 x1) -> n = 15 + 9 = 24 per arm.
SPEED_EXTRA_ARMS = ("cap_contour_F3000", "cap_contour_F1500")
SPEED_EXTRA_PER_RUNG = ((0.60, 2), (0.80, 2), (1.00, 2), (1.40, 2), (2.00, 1))
POS_HOLE_DIAM_MM = (0.25, 0.40, 0.60)
POS_DISC_R_MM = 1.5
N_NEG = 12
NEG_SIDE_MM = 5.0
PRIME_SIDE_MM, PRIME_PITCH_MM = 6.0, 0.5
FID_SIDE_MM = 4.0
FID_PITCH_MAX_FRAC_OF_PEN = 0.6   # fiducials are drawn at <= 0.6 x pen_mm so they stay SOLID; BIC keeps 0.12
FID_TICK_PITCH_MM, FID_TICK_H_MM = 10.0, 3.0
FID0_LEN_MM, FID90_LEN_MM = 200.0, 100.0
GLYPH_H_MM, GLYPH_W_MM = 8.0, 6.0

SHEET_W, SHEET_H = 237.0, 161.0
GRID_COLS, GRID_ROWS = 19, 12
CELL_MM = 11.0
GRID_X0, GRID_Y0 = 9.0, 11.0
JITTER_MM = 0.7

SEED_PLACEMENT = 20261002
SEED_ORDER = 20261003
SEED_JITTER = 20261004

PREDICTION_HEADER = [
    "POCKET SHEET (D813): dark pockets r=0.30-2.00mm x4 arms + speed arm + disc/capsule.",
    "Predictions are PRE-REGISTERED in artifacts/pocket-sheet-2026-10-02/PREREG.md (hashed).",
    "Read ONLY with the pre-registered detector; capture per docs/CAPTURE_RULE.md (D686).",
]


# ---------------------------------------------------------------------------
# Geometry (local frame: spine along +x, centre at origin, y down in layout).
# ---------------------------------------------------------------------------
def _circle_n(rho: float) -> int:
    """Vertices on a full circle of radius rho within SAGITTA_MM (what
    approxPolyDP(0.4px) leaves). Minimum 3."""
    if rho <= SAGITTA_MM:
        return 3
    return max(3, int(math.ceil(math.pi / math.acos(1.0 - SAGITTA_MM / rho))))


def _rot(pts, theta_deg, cx, cy):
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]


def _stadium_ring(rho: float, h: float):
    """Closed stadium (half-width rho, straight half-length h), start at the
    middle of one flat side. h == 0 -> circle starting at angle -90 deg."""
    n = _circle_n(rho)
    if h <= 1e-9:
        pts = [(rho * math.cos(-math.pi / 2 + 2 * math.pi * i / n),
                rho * math.sin(-math.pi / 2 + 2 * math.pi * i / n)) for i in range(n)]
        pts.append(pts[0])
        return pts
    half = max(2, n // 2)
    pts = [(0.0, -rho), (h, -rho)]
    for i in range(1, half):
        a = -math.pi / 2 + math.pi * i / half
        pts.append((h + rho * math.cos(a), rho * math.sin(a)))
    pts += [(h, rho), (-h, rho)]
    for i in range(1, half):
        a = math.pi / 2 + math.pi * i / half
        pts.append((-h + rho * math.cos(a), rho * math.sin(a)))
    pts += [(-h, -rho), (0.0, -rho)]
    return pts


def _contour_rho_list(r: float, p: float):
    out, k = [], 0
    while True:
        rho = r - (k + 0.5) * p
        if rho <= 1e-9:
            break
        out.append(rho)
        k += 1
    return out


def _row(o: float, half_len: float):
    return [(-half_len, o), (half_len, o)]


def _chord_half(h: float, rho0: float, o: float) -> float:
    return h + math.sqrt(max(rho0 * rho0 - o * o, 0.0))


def _parallel_offsets(start: float, p: float):
    """Positive row offsets on the ring grid (start, start-p, ...) while > p/2,
    plus the centre row (0.0) when the smallest positive offset leaves more than
    p/2 to the axis (or there is none). Returns the full signed, sorted-by-|o|
    list with the centre row first."""
    pos, o = [], start
    while o > p / 2 + 1e-9:
        pos.append(o)
        o -= p
    need_centre = (not pos) or (min(pos) > p / 2 + 1e-9)
    out = [0.0] if need_centre else []
    for o in sorted(pos):
        out += [o, -o]
    return out


def build_arm(shape: str, mode: str, r: float, p: float, theta_deg: float,
              cx: float, cy: float):
    """-> list of closed/open polylines in layout mm (emission order)."""
    h = SPINE_LEN_MM if shape == "capsule" else 0.0
    rho0 = r - p / 2.0
    local = []
    if mode == "contour":
        for rho in _contour_rho_list(r, p):
            local.append(_stadium_ring(rho, h))
    elif mode == "zigzag":
        ys, y = [], -rho0
        while y <= rho0 + 1e-9:
            ys.append(y)
            y += p
        if rho0 - ys[-1] > p / 2 + 1e-9:
            ys.append(rho0)
        poly, flip = [], False
        for y in ys:
            hl = _chord_half(h, rho0, y)
            if hl < 0.03:
                continue
            a, b = (-hl, y), (hl, y)
            poly += [b, a] if flip else [a, b]
            flip = not flip
        local.append(poly)
    elif mode in ("spine", "hybrid"):
        if shape != "capsule":
            raise ValueError("spine/hybrid arms are capsule-only")
        if mode == "spine":
            for o in _parallel_offsets(rho0, p):
                local.append(_row(o, _chord_half(h, rho0, o)))
        else:
            rhos = [rho for rho in _contour_rho_list(r, p) if rho >= R_HYBRID_MM - 1e-9]
            for rho in rhos:
                local.append(_stadium_ring(rho, h))
            if rhos:
                core = rhos[-1] - p
                for o in _parallel_offsets(core, p):
                    local.append(_row(o, _chord_half(h, max(core, 0.0), o)))
            else:
                for o in _parallel_offsets(rho0, p):
                    local.append(_row(o, _chord_half(h, rho0, o)))
    else:
        raise ValueError(mode)
    out = []
    for pl in local:
        if len(pl) >= 2 and _poly_length_mm(pl) >= MIN_STROKE_MM:
            out.append(_rot(pl, theta_deg, cx, cy))
    return out


def build_pos_hole(diam_mm: float, p: float, theta_deg: float, cx: float, cy: float):
    """Disc r=POS_DISC_R_MM, contour rings stopped so a bare core of ~diam
    remains. Returns (polylines, nominal_bare_radius_mm)."""
    b = diam_mm / 2.0
    rhos = [rho for rho in _contour_rho_list(POS_DISC_R_MM, p) if rho >= b + p / 2 - 1e-9]
    out = [_rot(_stadium_ring(rho, 0.0), theta_deg, cx, cy) for rho in rhos]
    bare = (rhos[-1] - p / 2.0) if rhos else POS_DISC_R_MM
    return out, bare


def _square_rows(side: float, p: float, cx: float, cy: float):
    """Solid square by one continuous zigzag, rows at pitch p inset p/2."""
    rho0 = side / 2.0 - p / 2.0
    ys, y = [], -rho0
    while y <= rho0 + 1e-9:
        ys.append(y)
        y += p
    if rho0 - ys[-1] > p / 2 + 1e-9:
        ys.append(rho0)
    poly, flip = [], False
    for y in ys:
        a, b = (cx - rho0, cy + y), (cx + rho0, cy + y)
        poly += [b, a] if flip else [a, b]
        flip = not flip
    return [poly]


def _square_rings(side: float, p: float, cx: float, cy: float):
    out, k = [], 0
    while True:
        hs = side / 2.0 - (k + 0.5) * p
        if hs < 0.02:
            break
        out.append([(cx - hs, cy - hs), (cx + hs, cy - hs), (cx + hs, cy + hs),
                    (cx - hs, cy + hs), (cx - hs, cy - hs)])
        k += 1
    return out


def _poly_length_mm(pts):
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1))


def _poly_bbox(pts):
    xs = [q[0] for q in pts]
    ys = [q[1] for q in pts]
    return max(xs) - min(xs), max(ys) - min(ys)


def _ticks(x0, y0, length, orientation, tick_dir):
    out, n = [], int(round(length / FID_TICK_PITCH_MM))
    for k in range(n + 1):
        d = k * FID_TICK_PITCH_MM
        if orientation == "h":
            out.append([(x0 + d, y0), (x0 + d, y0 + tick_dir * FID_TICK_H_MM)])
        else:
            out.append([(x0, y0 + d), (x0 + tick_dir * FID_TICK_H_MM, y0 + d)])
    return out


def _glyph(x0, y0):
    spine = [(x0, y0), (x0, y0 + GLYPH_H_MM)]
    top = [(x0, y0), (x0 + GLYPH_W_MM, y0)]
    mid = [(x0, y0 + GLYPH_H_MM * 0.5), (x0 + GLYPH_W_MM * 0.55, y0 + GLYPH_H_MM * 0.5)]
    return [spine, top, mid]


# ---------------------------------------------------------------------------
# Item table + placement. Identical for BOTH pens (the P012 arm slots are simply
# left blank on the BIC), so the Preppy twin is the same geometry in the same
# cells.
# ---------------------------------------------------------------------------
def _item_table():
    items = []
    for arm_id, shape, mode, speed, p_ovr, only_diff in ARMS:
        for rung in RUNGS_MM:
            for rep in range(N_REPS):
                items.append({"kind": "pocket", "arm": arm_id, "shape": shape,
                              "mode": mode, "speed": speed, "p_override": p_ovr,
                              "only_if_pitch_differs": only_diff,
                              "r_mm": rung, "rep": rep, "theta_deg": ORIENT_DEG[rep],
                              "id": f"{arm_id}__r{rung:.2f}__k{rep}"})
    for dd in POS_HOLE_DIAM_MM:
        for rep in range(N_REPS):
            items.append({"kind": "pos_hole", "arm": "pos_hole", "diam_mm": dd, "rep": rep,
                          "speed": FAST, "theta_deg": ORIENT_DEG[rep],
                          "id": f"pos_hole__d{dd:.2f}__k{rep}"})
    for i in range(N_NEG):
        items.append({"kind": "neg_square", "arm": "neg_square", "rep": i, "speed": FAST,
                      "theta_deg": 0.0, "id": f"neg_square__{i:02d}"})
# The 18 otherwise unused cells go to the speed-arm pair (cap_contour F3000 /
# F1500), 9 + 9, over r >= 0.6 mm. They are APPENDED last so every earlier item
# keeps its cell and jitter (the jitter stream is consumed in item order); the
# extras land exactly in the otherwise unused cells. Extra replicates are rep 3/4
# per rung; orientations cycle 0/45/90 in sequence order (3 each per arm).
    for arm_id, shape, mode, speed, p_ovr, only_diff in ARMS:
        if arm_id not in SPEED_EXTRA_ARMS:
            continue
        seq = 0
        for rung, n_extra in SPEED_EXTRA_PER_RUNG:
            for j in range(n_extra):
                rep = N_REPS + j
                th = ORIENT_DEG[seq % len(ORIENT_DEG)]
                seq += 1
                items.append({"kind": "pocket", "arm": arm_id, "shape": shape,
                              "mode": mode, "speed": speed, "p_override": p_ovr,
                              "only_if_pitch_differs": only_diff,
                              "r_mm": rung, "rep": rep, "theta_deg": th,
                              "id": f"{arm_id}__r{rung:.2f}__k{rep}"})
    return items


def _assign_cells(items):
    n_cells = GRID_COLS * GRID_ROWS
    if len(items) > n_cells:
        raise RuntimeError(f"pocket: {len(items)} items > {n_cells} cells")
    rng = random.Random(SEED_PLACEMENT)
    cells = list(range(n_cells))
    rng.shuffle(cells)
    jr = random.Random(SEED_JITTER)
    for it, cell in zip(items, cells):
        c, r = cell % GRID_COLS, cell // GRID_COLS
        it["cell"] = [c, r]
        it["cx"] = round(GRID_X0 + (c + 0.5) * CELL_MM + jr.uniform(-JITTER_MM, JITTER_MM), 4)
        it["cy"] = round(GRID_Y0 + (r + 0.5) * CELL_MM + jr.uniform(-JITTER_MM, JITTER_MM), 4)
    unused = sorted(set(range(n_cells)) - set(cells[:len(items)]))
    return [[u % GRID_COLS, u // GRID_COLS] for u in unused]


def _emission_order(n_items):
    order = list(range(n_items))
    random.Random(SEED_ORDER).shuffle(order)
    return order


def _outline_geom(it, p):
    cx, cy = it["cx"], it["cy"]
    if it["kind"] == "pocket":
        if it["shape"] == "disc":
            return Point(cx, cy).buffer(it["r_mm"], 32)
        a, b = _rot([(-SPINE_LEN_MM, 0), (SPINE_LEN_MM, 0)], it["theta_deg"], cx, cy)
        return LineString([a, b]).buffer(it["r_mm"], 32)
    if it["kind"] == "pos_hole":
        return Point(cx, cy).buffer(POS_DISC_R_MM, 32)
    return box(cx - NEG_SIDE_MM / 2, cy - NEG_SIDE_MM / 2, cx + NEG_SIDE_MM / 2, cy + NEG_SIDE_MM / 2)


def build_layout_for_pen(pen_id: str, pitch_mm: float):
    spec = PEN_SPECS[pen_id]
    items = _item_table()
    unused = _assign_cells(items)
    include_p012 = abs(pitch_mm - BIC_PITCH_REF_MM) > 1e-9
    elements, records, geoms = [], [], {}

    def add(pls, speed, kind, gid):
        a = len(elements)
        for pl in pls:
            if _poly_length_mm(pl) >= MIN_STROKE_MM - 1e-12:
                elements.append({"pts": [(float(x), float(y)) for x, y in pl],
                                 "speed": speed, "kind": kind, "gid": gid})
        return [a, len(elements)]

    fixed_groups = {}
    # -- header strokes: glyph, pen-id bars, rulers, prime patch --
    pls = _glyph(12.0, 1.0)
    pls += [[(24.0 + 3.0 * i, 1.0), (24.0 + 3.0 * i, 8.0)] for i in range(spec["id_bars"])]
    fixed_groups["glyph"] = pls
    add(pls, FAST, "glyph", "glyph")
    r0 = [[(GRID_X0, 158.0), (GRID_X0 + FID0_LEN_MM, 158.0)]] + _ticks(GRID_X0, 158.0, FID0_LEN_MM, "h", -1)
    r90 = [[(234.0, 22.0), (234.0, 22.0 + FID90_LEN_MM)]] + _ticks(234.0, 22.0, FID90_LEN_MM, "v", -1)
    fixed_groups["ruler0"], fixed_groups["ruler90"] = r0, r90
    add(r0, FAST, "ruler", "ruler0")
    add(r90, FAST, "ruler", "ruler90")
    prime = _square_rows(PRIME_SIDE_MM, PRIME_PITCH_MM, 227.0, 133.0)
    fixed_groups["prime"] = prime
    add(prime, FAST, "prime", "prime")

    # -- pockets/controls in seeded random order --
    p012_drop = 0
    for idx in _emission_order(len(items)):
        it = items[idx]
        cx, cy, th = it["cx"], it["cy"], it["theta_deg"]
        if it["kind"] == "pocket":
            if it["only_if_pitch_differs"] and not include_p012:
                p012_drop += 1
                continue
            p = it["p_override"] if it["p_override"] else pitch_mm
            pls = build_arm(it["shape"], it["mode"], it["r_mm"], p, th, cx, cy)
            rec_extra = {"pitch_mm": p, "r_mm": it["r_mm"], "shape": it["shape"],
                         "mode": it["mode"], "rep": it["rep"], "theta_deg": th,
                         "n_rings_or_rows": len(pls)}
        elif it["kind"] == "pos_hole":
            pls, bare = build_pos_hole(it["diam_mm"], pitch_mm, th, cx, cy)
            rec_extra = {"pitch_mm": pitch_mm, "diam_mm_commanded": it["diam_mm"],
                         "nominal_bare_radius_mm": round(bare, 4), "rep": it["rep"],
                         "theta_deg": th, "n_rings_or_rows": len(pls)}
        else:
            pls = _square_rows(NEG_SIDE_MM, pitch_mm, cx, cy)
            rec_extra = {"pitch_mm": pitch_mm, "side_mm": NEG_SIDE_MM, "rep": it["rep"],
                         "n_rings_or_rows": len(pls)}
        rng = add(pls, it["speed"], it["kind"], it["id"])
        geoms[it["id"]] = _outline_geom(it, pitch_mm)
        records.append({"id": it["id"], "kind": it["kind"], "arm": it["arm"],
                        "speed": it["speed"], "cell": it["cell"],
                        "center_mm": [it["cx"], it["cy"]], "elem_range": rng,
                        "path_len_mm": round(sum(_poly_length_mm(e["pts"])
                                                 for e in elements[rng[0]:rng[1]]), 3),
                        **rec_extra})

    # -- six solid fiducial squares, last --
    fid_pos = [(2.0, 2.0), (SHEET_W / 2 - 2.0, 2.0), (SHEET_W - 6.0, 2.0),
               (2.0, 148.0), (SHEET_W / 2 - 2.0, 148.0), (SHEET_W - 6.0, 148.0)]
    fid_records = []
    for i, (fx, fy) in enumerate(fid_pos):
        fid_pitch = min(pitch_mm, FID_PITCH_MAX_FRAC_OF_PEN * spec['pen_mm'])
        pls = _square_rings(FID_SIDE_MM, fid_pitch, fx + FID_SIDE_MM / 2, fy + FID_SIDE_MM / 2)
        fixed_groups[f"fid{i}"] = [[(fx, fy), (fx + FID_SIDE_MM, fy + FID_SIDE_MM)]]
        add(pls, FAST, "fiducial", f"fid{i}")
        fid_records.append({"id": f"fid{i}", "origin_mm": [fx, fy], "side_mm": FID_SIDE_MM,
                            "center_mm": [fx + FID_SIDE_MM / 2, fy + FID_SIDE_MM / 2],
                            "pitch_mm": fid_pitch, "n_rings": len(pls)})
    meta = {"unused_cells": unused, "p012_included": include_p012, "p012_slots_blank": p012_drop,
            "fiducials": fid_records, "n_elements": len(elements)}
    return elements, records, fixed_groups, geoms, meta


def _check_clearance(geoms, fixed_groups, min_mm=CLEARANCE_MM):
    allg = dict(geoms)
    for k, pls in fixed_groups.items():
        allg[k] = unary_union([LineString(pl) for pl in pls if len(pl) >= 2]).buffer(0.05)
    names = list(allg)
    viol, n_pairs = [], 0
    for i in range(len(names)):
        gi = allg[names[i]]
        for j in range(i + 1, len(names)):
            gj = allg[names[j]]
            if gi.bounds[2] + min_mm < gj.bounds[0] or gj.bounds[2] + min_mm < gi.bounds[0] \
                    or gi.bounds[3] + min_mm < gj.bounds[1] or gj.bounds[3] + min_mm < gi.bounds[1]:
                continue
            n_pairs += 1
            d = gi.distance(gj)
            if d < min_mm - 1e-9:
                viol.append([names[i], names[j], round(d, 3)])
    if viol:
        raise RuntimeError(f"pocket: {len(viol)} item pair(s) violate the {min_mm}mm clearance "
                           f"rule: {viol[:6]}. Do not plot this file.")
    return {"n_groups": len(names), "min_clearance_mm": min_mm, "n_pairs_checked": n_pairs}


# ---------------------------------------------------------------------------
# Post-emission validation: per-stroke position (catches a wrong flip/offset
# mapping), per-stroke F, stroke count. Parsed from the EMITTED file.
# ---------------------------------------------------------------------------
def _parse_strokes_f(path, z_down_abs, z_tol=1e-2):
    strokes, cur, x, y, down = [], None, None, None, False
    with Path(path).open() as fh:
        for line in fh:
            if not line.startswith("G1"):
                continue
            mz = re.search(r"Z(-?[\d.]+)", line)
            if mz:
                was = down
                down = abs(round(float(mz.group(1)), 3) - round(z_down_abs, 3)) < z_tol
                if down and not was:
                    cur = {"pts": [], "f": []}
                    if x is not None:
                        cur["pts"].append((x, y))
                    strokes.append(cur)
                elif was and not down:
                    cur = None
                continue
            mx = re.search(r"X(-?[\d.]+)", line)
            my = re.search(r"Y(-?[\d.]+)", line)
            mf = re.search(r"F(-?[\d.]+)", line)
            nx = float(mx.group(1)) if mx else x
            ny = float(my.group(1)) if my else y
            if down and cur is not None:
                cur["pts"].append((nx, ny))
                cur["f"].append(float(mf.group(1)) if mf else None)
            x, y = nx, ny
    return [s for s in strokes if len(s["pts"]) >= 2]


def check_strokes(strokes, elements, expect_xy, pos_tol=0.0075):
    """Every emitted stroke must match its design element in vertex count,
    vertex position (mapped through `expect_xy`) and per-point F. Raises on any
    mismatch; returns summary. Positive-controlled by selftest()."""
    if not strokes:
        raise RuntimeError("pocket: ZERO pen-down strokes parsed -- the guard never ran.")
    if len(strokes) != len(elements):
        raise RuntimeError(f"pocket: stroke count mismatch: emitted {len(strokes)}, "
                           f"expected {len(elements)}. Do not plot this file.")
    worst, bad_f, bad_n = 0.0, [], []
    for i, (s, e) in enumerate(zip(strokes, elements)):
        if len(s["pts"]) != len(e["pts"]):
            bad_n.append([i, len(s["pts"]), len(e["pts"])])
            continue
        for (sx, sy), (ex, ey) in zip(s["pts"], e["pts"]):
            tx, ty = expect_xy(ex, ey)
            worst = max(worst, abs(sx - tx), abs(sy - ty))
        fs = s["f"]
        if any(abs(f - e["speed"]) > 1e-6 for f in fs if f is not None) or not fs:
            bad_f.append([i, e["gid"], e["speed"], sorted(set(fs))[:3]])
    if bad_n:
        raise RuntimeError(f"pocket: {len(bad_n)} stroke(s) differ in vertex count from design "
                           f"(idx, emitted, design): {bad_n[:5]}. Do not plot this file.")
    if worst > pos_tol:
        raise RuntimeError(f"pocket: emitted vs design position deviation {worst:.4f} mm > "
                           f"{pos_tol} mm -- the flip/offset mapping is wrong. Do not plot.")
    if bad_f:
        raise RuntimeError(f"pocket: {len(bad_f)} stroke(s) carry the wrong F: {bad_f[:5]}. "
                           f"Do not plot this file.")
    return {"n_strokes": len(strokes), "worst_position_dev_mm": round(worst, 5),
            "n_speed_F1500": sum(1 for e in elements if e["speed"] == SLOW),
            "n_speed_F3000": sum(1 for e in elements if e["speed"] == FAST)}


def selftest():
    """Negative controls for check_strokes: a shifted mapping, a wrong F and a
    dropped stroke must each be CAUGHT. Returns the list of caught cases."""
    els = [{"pts": [(1.0, 1.0), (2.0, 1.0)], "speed": FAST, "kind": "x", "gid": "a"},
           {"pts": [(1.0, 2.0), (2.0, 2.0)], "speed": SLOW, "kind": "x", "gid": "b"}]
    good = [{"pts": [(11.0, 21.0), (12.0, 21.0)], "f": [FAST]},
            {"pts": [(11.0, 22.0), (12.0, 22.0)], "f": [SLOW]}]
    xy = lambda x, y: (x + 10.0, y + 20.0)
    check_strokes(good, els, xy)                       # must pass
    caught = []
    for name, bad, f in (
        ("shifted_mapping", [{"pts": [(11.05, 21.0), (12.05, 21.0)], "f": [FAST]}, good[1]], xy),
        ("wrong_F", [good[0], {"pts": good[1]["pts"], "f": [FAST]}], xy),
        ("dropped_stroke", [good[0]], xy),
        ("vertex_count", [{"pts": [(11.0, 21.0), (12.0, 21.0), (12.5, 21.0)], "f": [FAST, FAST]}, good[1]], xy),
    ):
        try:
            check_strokes(bad, els, f)
        except RuntimeError:
            caught.append(name)
    # clearance guard positive control: two outlines 1.0 mm apart MUST raise, and
    # two 5 mm apart must pass (so "0 pairs checked" cannot hide a dead guard).
    try:
        _check_clearance({"a": box(0, 0, 3, 3), "b": box(4, 0, 7, 3)}, {})
    except RuntimeError:
        caught.append("clearance_violation")
    _check_clearance({"a": box(0, 0, 3, 3), "b": box(8, 0, 11, 3)}, {})
    if len(caught) != 5:
        raise RuntimeError(f"pocket selftest: only caught {caught}")
    return caught


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------
def _write_svg_hp(path: Path, w: int, h: int, stroke_px: float, polylines):
    """Like svg_io._write_paths_svg but 4-decimal px coordinates (the 2-decimal
    rounding there is 0.0026 mm, ~9% of a 0.03 mm innermost ring)."""
    import svgwrite
    with atomic_target(path) as tmp:
        dwg = svgwrite.Drawing(str(tmp), size=(f"{w}px", f"{h}px"), viewBox=f"0 0 {w} {h}")
        g = dwg.g(stroke="black", stroke_width=round(float(stroke_px), 3), fill="none")
        for pts in polylines:
            g.add(dwg.polyline([(round(float(x), 4), round(float(y), 4)) for x, y in pts]))
        dwg.add(g)
        dwg.save()
    return path


def _prepend_ascii_header(path, lines):
    text = Path(path).read_text(encoding="ascii")
    Path(path).write_text("".join(f"; {ln}\n" for ln in lines) + text, encoding="ascii")


def _blocks(elements):
    out, cur = [], None
    for i, e in enumerate(elements):
        if cur is None or cur["speed"] != e["speed"]:
            cur = {"speed": e["speed"], "idx": []}
            out.append(cur)
        cur["idx"].append(i)
    return out


def pocket_gcode(pen_id, out_gcode: Path, pen_mm=None, z_up=3.5,
                 travel_speed=8000, plunge_speed=500, lift_speed=2000,
                 offset_x=None, offset_y=None, end_policy="release",
                 quiet_fans=True, profile=None):
    spec = PEN_SPECS[pen_id]
    ppmm = 1.0 / PX2MM
    offset_x_given = offset_x is not None
    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()
    profiles = load_pen_profiles()
    if pen_id not in profiles.get("profiles", {}):
        raise RuntimeError(f"pen profile {pen_id!r} not found in pen_profiles.json")
    pen = profiles["profiles"][pen_id]
    if profile is not None and profile != pen:
        raise RuntimeError(
            f"pocket: refusing to generate -- this sheet is PINNED to {pen_id!r} "
            f"({spec['label']!r}); the requested/active profile is {profile.get('label')!r}. "
            f"Mount {pen_id} and re-request, or POST with pen_id={pen_id!r} explicitly.")

    reach0 = active_reach_profile(cal)
    if reach0["id"] != EXPECT_REACH_ID or tuple(round(v, 1) for v in reach0["bbox"]) != EXPECT_BBOX:
        raise RuntimeError(f"Runtime reach envelope mismatch: got id={reach0['id']!r} "
                           f"bbox={reach0['bbox']}, expected {EXPECT_REACH_ID!r} {EXPECT_BBOX}.")
    if pen.get("label") != spec["label"]:
        raise RuntimeError(f"pen {pen_id!r} label mismatch: {pen.get('label')!r} != {spec['label']!r}")
    if abs(float(pen.get("pen_mm", 0.0)) - spec["pen_mm"]) > 1e-6:
        raise RuntimeError(f"pen {pen_id!r} pen_mm mismatch: {pen.get('pen_mm')} != {spec['pen_mm']}")
    got_off = (float(pen.get("pen_offset_x", 0.0)), float(pen.get("pen_offset_y", 0.0)))
    if any(abs(a - b) > 1e-6 for a, b in zip(got_off, spec["offset"])):
        raise RuntimeError(f"pen {pen_id!r} pen_offset mismatch: {got_off} != {spec['offset']}")
    z_touch, z_touch_source = resolve_z_touch(cal, pen)
    lo, hi = Z_TOUCH_SANITY_RANGE_MM
    if not (lo <= z_touch <= hi):
        raise RuntimeError(f"z_touch_abs for {pen_id} resolved to {z_touch}, outside [{lo},{hi}] mm.")
    z_down = float(pen.get("z_down", 0.0))
    if abs(z_down - EXPECT_Z_DOWN_STOCK) > 1e-6:
        raise RuntimeError(f"stock z_down mismatch for {pen_id}: {z_down} != {EXPECT_Z_DOWN_STOCK}")
    draw_speed = float(pen.get("draw_speed", 0.0))
    if abs(draw_speed - EXPECT_DRAW_SPEED_STOCK) > 1e-6 or draw_speed != FAST:
        raise RuntimeError(f"stock draw_speed mismatch for {pen_id}: {draw_speed} != {FAST}")
    pitch_mm = float(pen.get("fill_mm", 0.0))
    if not (0.05 <= pitch_mm <= 0.6):
        raise RuntimeError(f"fill_mm for {pen_id} = {pitch_mm} outside the sane pitch range "
                           f"(the fill slider is set; Reset it for this sheet).")

    reach = pen_reach_profile(reach0, pen)
    if pen_mm is None:
        pen_mm = spec["pen_mm"]
    pen_off_x, pen_off_y = resolve_pen_offset(cal, pen)

    elements, records, fixed_groups, geoms, meta = build_layout_for_pen(pen_id, pitch_mm)
    clearance_check = _check_clearance(geoms, fixed_groups)

    pad = _PAD_MM
    # Exact vpype geometry: a px is 25.4/96 mm; the SVG is written at ppmm=1/PX2MM px
    # per layout mm (the production convention), so a layout mm maps to S real mm.
    # Page dims are INTEGER px, so the vertical_flip axis (the larger page side) is
    # the rounded page, not canvas_mm -- 0.11 mm of Y error if the two are confused.
    U = 25.4 / 96.0
    S = ppmm * U
    W = int(round((SHEET_W + 2 * pad) * ppmm))
    H = int(round((SHEET_H + 2 * pad) * ppmm))
    canvas_w, canvas_h = SHEET_W + 2 * pad, SHEET_H + 2 * pad
    flip_axis = max(W, H) * U
    offset_x = (X0_TIP_DEFAULT - pad * S) if offset_x is None else float(offset_x)
    default_placement = (not offset_x_given) and offset_y is None
    if offset_y is None:
        # tip Y of layout y=H (front edge) = Y_FRONT_TIP_DEFAULT
        offset_y = Y_FRONT_TIP_DEFAULT + (pad + SHEET_H) * S - flip_axis
    else:
        offset_y = float(offset_y)
    comp_off_x, comp_off_y = offset_x - pen_off_x, offset_y - pen_off_y

    def expect_xy(x, y):       # layout mm -> commanded (nozzle) machine mm
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
            f"pocket: tip window X[{min(tx):.1f},{max(tx):.1f}] Y[{min(ty):.1f},{max(ty):.1f}] "
            f"outside the paper window X[{TIP_X_MIN},{TIP_X_MAX}] Y[{TIP_Y_MIN},{TIP_Y_MAX}] "
            f"(B5 landscape, holes at the FRONT; frame moved per D1487).")
    park_x, park_y = reach["safe_park_point"]
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))
    z_down_abs = z_touch + z_down

    placed_all = [[((x + pad) * ppmm, (y + pad) * ppmm) for x, y in e["pts"]] for e in elements]

    blocks = _blocks(elements)
    texts, labels = [], []
    blk_svg = WORK_DIR / f".{out_gcode.stem}_blk.svg"
    blk_gc = WORK_DIR / f".{out_gcode.stem}_blk.gcode"   # ONE shared name: {filename}
    for bi, blk in enumerate(blocks):
        _write_svg_hp(blk_svg, W, H, float(pen_mm) * ppmm, [placed_all[i] for i in blk["idx"]])
        vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({
            "z_up": float(z_up), "z_up_abs": z_touch + float(z_up),
            "z_down": z_down, "z_down_abs": z_down_abs,
            "z_touch_abs": z_touch, "z_travel_abs": z_travel_abs,
            "draw_speed": float(blk["speed"]), "travel_speed": float(travel_speed),
            "plunge_speed": float(plunge_speed), "lift_speed": float(lift_speed),
            "z_auto": "1", "origin_x": park_x, "origin_y": park_y,
            **_gcode_lifecycle_values(end_policy, quiet_fans)})
        vpype_cli.execute(f'read {shlex.quote(str(blk_svg))} pagesize {W}x{H}px '
                          f'translate -- {comp_off_x}mm {-comp_off_y}mm '
                          f'gwrite --profile elegoo_cc {shlex.quote(str(blk_gc))}')
        texts.append(blk_gc.read_text(encoding="utf-8"))
        labels.append(f"; BLOCK {bi + 1}/{len(blocks)} draw_speed=F{blk['speed']} "
                      f"elements={blk['idx'][0]}..{blk['idx'][-1]}")
    blk_svg.unlink(missing_ok=True)
    combined = _splice_blocks(texts, labels, z_touch + float(z_up))
    combined = combined.replace(str(blk_gc), str(out_gcode))
    blk_gc.unlink(missing_ok=True)

    header = list(PREDICTION_HEADER) + [
        f"pen_id={pen_id} z_touch_abs={z_touch:.3f} ({z_touch_source}) z_down_abs={z_down_abs:.3f}",
        f"pitch_mm={pitch_mm} (profile fill_mm) draw F{FAST} / speed-arm F{SLOW} "
        f"blocks={len(blocks)} strokes={len(elements)}",
        f"seeds placement={SEED_PLACEMENT} order={SEED_ORDER} jitter={SEED_JITTER}",
        "RECORD z_touch before plotting; BIC=18.95 / uRight=11.45 differ by 7.50 mm (DIRECTION S10).",
    ]
    with atomic_target(out_gcode) as tmp_gcode:
        Path(tmp_gcode).write_text(combined, encoding="utf-8")
        _prepend_ascii_header(tmp_gcode, header)

    validate_gcode_xy_motion(out_gcode, reach)
    safety = validate_calibration_gcode_safety(out_gcode, z_down_abs)
    strokes = _parse_strokes_f(out_gcode, z_down_abs)
    stroke_check = check_strokes(strokes, elements, expect_xy)
    selftest_caught = selftest()

    total_mm = sum(_poly_length_mm(s["pts"]) for s in strokes)
    plot_time_s = sum(_poly_length_mm(e["pts"]) / max(1.0, e["speed"] / 60.0) for e in elements)
    plot_time_s += len(elements) * (abs(z_up) / max(1.0, plunge_speed / 60.0)
                                    + abs(z_up) / max(1.0, lift_speed / 60.0))

    prev_svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_svg_hp(prev_svg, W, H, float(pen_mm) * ppmm, placed_all)
    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(prev_svg))} color {PREVIEW_COLOR} '
                          f'write {shlex.quote(str(tmp_prev))}')

    layout = {
        "sheet": "pocket", "pen_id": pen_id, "pitch_mm": pitch_mm,
        "rungs_mm": list(RUNGS_MM), "n_reps": N_REPS, "orient_deg": list(ORIENT_DEG),
        "spine_len_mm": SPINE_LEN_MM, "r_hybrid_mm": R_HYBRID_MM, "sagitta_mm": SAGITTA_MM,
        "speeds": {"fast": FAST, "slow": SLOW}, "arms": [a[0] for a in ARMS],
        "seeds": {"placement": SEED_PLACEMENT, "order": SEED_ORDER, "jitter": SEED_JITTER},
        "grid": {"cols": GRID_COLS, "rows": GRID_ROWS, "cell_mm": CELL_MM,
                 "x0": GRID_X0, "y0": GRID_Y0, "jitter_mm": JITTER_MM,
                 "unused_cells": meta["unused_cells"]},
        "sheet_wh_mm": [SHEET_W, SHEET_H], "pad_mm": pad,
        "mapping": {"offset_x": offset_x, "offset_y": offset_y, "flip_axis_mm": flip_axis,
                    "comp_off_x": comp_off_x, "comp_off_y": comp_off_y,
                    "pen_off_x": pen_off_x, "pen_off_y": pen_off_y,
                    "cmd_x": "pad + x + comp_off_x", "cmd_y": "flip + comp_off_y - (pad + y)",
                    "tip_x": "cmd_x + pen_off_x", "tip_y": "cmd_y + pen_off_y"},
        "tip_window_mm": {"x": [min(tx), max(tx)], "y": [min(ty), max(ty)]},
        "paper_window_mm": {"x": [TIP_X_MIN, TIP_X_MAX], "y": [TIP_Y_MIN, TIP_Y_MAX]},
        "z_touch_abs": z_touch, "z_touch_source": z_touch_source, "z_down_abs": z_down_abs,
        "items": records, "fiducials": meta["fiducials"],
        "rulers": {"h": {"origin_mm": [GRID_X0, 158.0], "len_mm": FID0_LEN_MM},
                   "v": {"origin_mm": [234.0, 22.0], "len_mm": FID90_LEN_MM},
                   "tick_pitch_mm": FID_TICK_PITCH_MM},
        "p012_included": meta["p012_included"], "p012_slots_blank": meta["p012_slots_blank"],
        "blocks": [{"speed": b["speed"], "first": b["idx"][0], "last": b["idx"][-1]} for b in blocks],
        "stroke_check": stroke_check, "selftest_caught": selftest_caught,
        "clearance_check": clearance_check,
    }
    layout_path = WORK_DIR / f"{out_gcode.stem}_layout.json"
    layout_path.write_text(json.dumps(layout, indent=1), encoding="utf-8")

    return {"kind": "pocket", "pen_mm": pen_mm, "pen_id": pen_id, "pen_label": spec["label"],
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
            "clearance_check": clearance_check, "n_blocks": len(blocks),
            "tip_window_mm": layout["tip_window_mm"], "default_placement": default_placement,
            "plot_time_estimate_s": round(plot_time_s, 1),
            "plot_time_estimate_min": round(plot_time_s / 60.0, 2),
            "plot_time_model_label": "model (a): path length / feed + per-stroke plunge+lift; no "
                                     "travel time, no microlift; a floor, not a forecast.",
            "preview": prev.name, "svg_path": str(prev_svg), "svg_wh_px": [W, H],
            "pattern_mm": [round(canvas_w, 1), round(canvas_h, 1)],
            "place": {"x": offset_x, "y": offset_y, "w": canvas_w, "h": canvas_h},
            "z_up_abs": round(z_touch + float(z_up), 3),
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
            "layout_path": str(layout_path), "total_draw_mm": round(total_mm, 1)}


def pocket_bic(out_gcode, **kw):
    return pocket_gcode("bic_cristal_blue", out_gcode, **kw)


def pocket_preppy(out_gcode, **kw):
    return pocket_gcode("platinum_preppy", out_gcode, **kw)
