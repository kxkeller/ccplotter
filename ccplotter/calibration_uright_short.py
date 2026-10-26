"""ccplotter/calibration_uright_short.py -- one SHORT uRight gel calibration sheet.

What it measures: uRight plots can leave blank areas on the paper although the g-code
matches the preview (to 0.0007 mm), so the blanks are deposition / contact effects. Ranked
candidates:
  (1) start-skipping after pen lifts,
  (2) regional contact loss across the plate,
  (3) the laid width is narrower than assumed. The uRight laid width has not been
      measured: 0.051 / 0.092 are pitch targets, 0.255 may be a scan blur floor,
      core_mm 0.18 is an in-sample proxy.
This ONE sheet separates the three on one piece of paper:

  1. LAID-WIDTH WEDGE. Parallel-pass patches on an ABSOLUTE pitch ladder (0.08 .. 0.50 mm,
     13 steps; never scaled off profile fields) at 0 and 45 deg of pass direction. Each
     patch is ONE continuous serpentine (one pen-down, zero lifts), so start-skipping
     cannot contaminate the width reading. 45 deg patches are the same 6 x 6 mm serpentine
     rotated about its centre. CAVEAT: "zero lifts" means one start per patch. Each wedge
     patch still begins with ONE post-travel pen-down, so its first row can be
     start-skipped; exclude the first row from the transition read. Read any pair
     comparison with its sample size in mind: n = 4 per arm, permutation floor p = 1/70,
     no power study. Plus three isolated 20 mm single lines at 0/45/90 deg. Laid width is
     UNMEASURED: where the wedge goes from striped to solid on paper IS the measurement.
  2. START-SKIP PAIR. N (one continuous serpentine) vs L03 (the same path cut into 0.3 mm
     pieces, each its own pen-down) at pitch 0.13 and 0.25 mm, 4 replicates per arm per
     pitch (16 patches, 4.5 mm square, PAIR_SIDE_MM, to keep the plot short). Commanded ink
     is IDENTICAL: L03 cuts the very polyline N draws (rows with golden-ratio staggered cut
     phases, as in calibration_fragment, the one-pitch edge connector riding on the last
     piece of each row), so total length is equal to float precision. Placement is a seeded
     shuffle of the cells. Lighter L03 patches than N point at start-skipping.
  3. CONTACT GRID. 6 x 4 identical marks over the full usable page: a 6 mm line at 0 deg, a
     6 mm line at 90 deg and a 3 x 3 mm continuous serpentine at pitch 0.13 mm. Regional
     contact loss shows as marks that thin out in one region.

Emission order is ONE seeded shuffle across ALL measured items (wedge, pair, grid, lines),
so arm / pitch / position is not confounded with position in time (pen dryness).
Fiducials, glyph, rulers and the prime go FIRST (as calibration_fragment). Every item
carries a printed ASCII label. Micro/mega lifts are applied as production does.

HARD-CODED MACHINE AND PEN (limitation). The sheet is tied to the author's measured machine:
EXPECT_REACH_ID ("plotter_2003_v32") and EXPECT_BBOX are imported from calibration_pocket
and the generator refuses any other reach profile. The pen is PINNED to "uright_gel".
"uright_gel_halfpitch" / "uright_gel_tight3x" share its z_touch; if one of them is the
requested profile the sheet is still generated from the BASE uright_gel profile (its z and
feeds), after asserting the variant's Z fields equal the base's. z_touch, z_down, z_up,
z_micro and every feedrate are READ from the live profile. All geometry is absolute mm and
independent of the profile's fill_mm / laid_mm / pen_mm.

Gel wet-pooling limits: no pitch < 0.08 mm, no patch larger than 6 x 6 mm (the 45 deg
patches are 6 x 6 mm squares rotated: bbox 8.5 mm, area unchanged).
"""
from __future__ import annotations

import json
import math
import random
import shlex
from pathlib import Path

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
    EXPECT_REACH_ID, EXPECT_BBOX, Z_TOUCH_SANITY_RANGE_MM, TIP_X_MIN, TIP_X_MAX, TIP_Y_MIN,
    TIP_Y_MAX, X0_TIP_DEFAULT, Y_FRONT_TIP_DEFAULT, SHEET_W, SHEET_H, FID_SIDE_MM, FID0_LEN_MM,
    FID90_LEN_MM, GRID_X0, MIN_STROKE_MM, _poly_length_mm, _square_rings, _square_rows, _ticks,
    _glyph, _write_svg_hp, _prepend_ascii_header, _parse_strokes_f, _check_clearance,
    check_strokes,
)
from ccplotter.calibration_fragment import (label_strokes, time_model_a, time_model_b,
                                            _segments, _phase)

_PAD_MM = 3.0

PIN_ID = "uright_gel"
FAMILY_IDS = ("uright_gel", "uright_gel_halfpitch", "uright_gel_tight3x")
SHEET_ID = "uright_short"

# ---- absolute geometry (mm). NOTHING below is derived from fill_mm / laid_mm / pen_mm.
WEDGE_PITCHES_MM = (0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.23, 0.26, 0.30, 0.35, 0.40, 0.50)
WEDGE_ANGLES_DEG = (0.0, 45.0)
MIN_PITCH_MM = 0.08                  # gel wet-pooling floor
PATCH_SIDE_MM = 6.0                  # wedge / pair patch side; also the hard maximum
MAX_PATCH_SIDE_MM = 6.0
LINE_LEN_MM = 20.0
LINE_ANGLES_DEG = (0.0, 45.0, 90.0)
PAIR_PITCHES_MM = (0.13, 0.25)
PAIR_REPS = 4
PAIR_SIDE_MM = 4.5                   # pair patch side (smaller than PATCH_SIDE_MM to keep the plot short;
                                     # pen-downs scale with its square).
PIECE_MM = 0.3
GRID_COLS, GRID_ROWS = 6, 4
MARK_LINE_MM = 6.0
MARK_PATCH_MM = 3.0
MARK_PITCH_MM = 0.13
FID_PITCH_MM = 0.08                  # absolute; = the sheet's tightest pitch so the fiducials stay solid for a thin pen
LABEL_CAP_MM = 2.0                   # == calibration_fragment.LABEL_CAP_MM (asserted)
LABEL_GAP_MM = 1.0

SEED_PLACEMENT = 20261012
SEED_ORDER = 20261013

# layout-frame placement (mm; y down; layout 237 x 161, same frame as fragment_bic)
MARK_X0, MARK_DX = 9.0, 39.0
MARK_Y = (12.0, 52.0, 92.0, 132.0)
MARK_W, MARK_H = 13.0, 9.0
WEDGE_X0, WEDGE_DX = 9.0, 12.0
WEDGE_ROW_TOP = (24.0, 38.0)         # label-top of the 0 deg / 45 deg rows
WEDGE_HALF = PATCH_SIDE_MM * math.sqrt(2.0) / 2.0   # 4.243: label offset for ALL wedge patches
PAIR_X0, PAIR_DX = 9.0, 12.0
PAIR_ROW_TOP = (66.0, 77.0)
LINES_CX = (40.0, 90.0, 140.0)
LINES_CY = 118.0
LEGEND_CX = 197.0
LEGEND_Y = (28.0, 33.0, 38.0, 43.0)  # baselines

assert all(p >= MIN_PITCH_MM - 1e-12 for p in WEDGE_PITCHES_MM + PAIR_PITCHES_MM + (MARK_PITCH_MM,))
assert PATCH_SIDE_MM <= MAX_PATCH_SIDE_MM and PAIR_SIDE_MM <= MAX_PATCH_SIDE_MM
assert MARK_PATCH_MM <= MAX_PATCH_SIDE_MM

PREDICTION_HEADER = [
    "URIGHT SHORT SHEET (D1087/D1088): laid-width wedge + start-skip pair + contact grid.",
    "Separates (1) start-skipping, (2) regional contact loss, (3) narrower-than-assumed laid width.",
    "LAID WIDTH UNMEASURED: pitches are commanded spacings, not widths. Gel: no pitch < 0.08, no patch > 6x6 mm.",
]


# ---------------------------------------------------------------------------
# Builders (local frame, centre origin, y down). Each returns pen-down polylines.
# ---------------------------------------------------------------------------
def n_rows(side: float, p: float) -> int:
    return int(round(side / p))


def build_serpentine(w: float, h: float, p: float):
    """ONE continuous boustrophedon polyline, width w, rows at pitch p, n = round(h/p)
    rows centred on the origin (patch height n*p ~ h). Returns [polyline]."""
    n = n_rows(h, p)
    half = w / 2.0
    poly = []
    for i in range(n):
        y = (i - (n - 1) / 2.0) * p
        poly += [(-half, y), (half, y)] if i % 2 == 0 else [(half, y), (-half, y)]
    return [poly]


def build_l03(w: float, h: float, p: float, L: float = PIECE_MM):
    """The SAME path as build_serpentine, cut into pieces of nominal length L, each its
    own pen-down. Each row is cut at golden-ratio staggered phases (as in calibration_fragment);
    the one-pitch edge connector rides on the LAST piece of its row, so the union of
    pieces is exactly the serpentine polyline (equal total length)."""
    n = n_rows(h, p)
    half = w / 2.0
    out = []
    for i in range(n):
        y = (i - (n - 1) / 2.0) * p
        segs = _segments(-half, half, L, _phase(i, L))
        if i % 2 == 1:
            segs = [(b, a) for a, b in reversed(segs)]
        for k, (a, b) in enumerate(segs):
            piece = [(a, y), (b, y)]
            if k == len(segs) - 1 and i < n - 1:
                piece.append((b, (i + 1 - (n - 1) / 2.0) * p))
            out.append(piece)
    return out


def build_line(length: float):
    return [[(-length / 2.0, 0.0), (length / 2.0, 0.0)]]


def build_mark():
    """Contact-grid mark in mark-local coords: origin = top-left of the mark box (label
    box on top). Returns (strokes, label_cx, label_baseline_y)."""
    ytop = MARK_H - MARK_LINE_MM            # shapes box top (label above it)
    hline = [(0.0, ytop + MARK_LINE_MM / 2.0), (MARK_LINE_MM, ytop + MARK_LINE_MM / 2.0)]
    vline = [(MARK_LINE_MM + 2.0, ytop), (MARK_LINE_MM + 2.0, ytop + MARK_LINE_MM)]
    px = MARK_LINE_MM + 4.0
    py = ytop + MARK_LINE_MM / 2.0
    patch = build_serpentine(MARK_PATCH_MM, MARK_PATCH_MM, MARK_PITCH_MM)[0]
    patch = [(px + MARK_PATCH_MM / 2.0 + x, py + y) for x, y in patch]
    return [hline, vline, patch], MARK_W / 2.0, LABEL_CAP_MM


def _rot(pts, theta, cx, cy):
    return _pk._rot(pts, theta, cx, cy)


def _tr(pts, dx, dy):
    return [(x + dx, y + dy) for x, y in pts]


# ---------------------------------------------------------------------------
# Item table
# ---------------------------------------------------------------------------
def _hundredths(p):
    return f"{int(round(p * 100)):02d}"


def build_items():
    """All measured items with their layout positions. Placement: wedge/lines/marks are
    fixed lattice positions; the pair cells are a SEEDED shuffle."""
    items = []
    # wedge
    for ri, ang in enumerate(WEDGE_ANGLES_DEG):
        for ci, p in enumerate(WEDGE_PITCHES_MM):
            cx = WEDGE_X0 + WEDGE_DX * ci + WEDGE_DX / 2.0
            cy = WEDGE_ROW_TOP[ri] + LABEL_CAP_MM + LABEL_GAP_MM + WEDGE_HALF
            items.append({"kind": "wedge", "id": f"W{_hundredths(p)}-{int(ang)}",
                          "label": f"{_hundredths(p)}-{int(ang)}", "pitch_mm": p,
                          "theta_deg": ang, "cx": cx, "cy": cy, "half": WEDGE_HALF,
                          "arm": "wedge", "side_mm": PATCH_SIDE_MM})
    # isolated lines
    for k, ang in enumerate(LINE_ANGLES_DEG):
        items.append({"kind": "line", "id": f"LN-{int(ang)}", "label": f"L20-{int(ang)}",
                      "theta_deg": ang, "cx": LINES_CX[k], "cy": LINES_CY,
                      "half": LINE_LEN_MM / 2.0, "arm": "line", "length_mm": LINE_LEN_MM})
    # start-skip pair
    pair = []
    for p in PAIR_PITCHES_MM:
        for arm in ("N", "L"):
            for rep in range(PAIR_REPS):
                pair.append({"kind": "pair", "arm": arm, "pitch_mm": p, "rep": rep,
                             "theta_deg": 0.0 if rep % 2 == 0 else 90.0,
                             "half": PAIR_SIDE_MM / 2.0, "side_mm": PAIR_SIDE_MM,
                             "id": f"{arm}{_hundredths(p)}-{rep + 1}",
                             "label": f"{arm}{_hundredths(p)}-{rep + 1}"})
    cells = [(c, r) for r in range(2) for c in range(8)]
    random.Random(SEED_PLACEMENT).shuffle(cells)
    for it, (c, r) in zip(pair, cells):
        it["cell"] = [c, r]
        it["cx"] = PAIR_X0 + PAIR_DX * c + PAIR_DX / 2.0
        it["cy"] = PAIR_ROW_TOP[r] + LABEL_CAP_MM + LABEL_GAP_MM + PAIR_SIDE_MM / 2.0
        items.append(it)
    # contact marks
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            items.append({"kind": "mark", "id": f"C{r + 1}{c + 1}", "label": f"C{r + 1}{c + 1}",
                          "arm": "mark", "grid": [c, r],
                          "ox": MARK_X0 + MARK_DX * c, "oy": MARK_Y[r]})
    return items


def build_patch(it):
    """Local-frame strokes of one measured item (patch or line); marks handled apart."""
    k = it["kind"]
    if k == "wedge":
        return build_serpentine(PATCH_SIDE_MM, PATCH_SIDE_MM, it["pitch_mm"])
    if k == "line":
        return build_line(LINE_LEN_MM)
    if k == "pair":
        if it["arm"] == "N":
            return build_serpentine(PAIR_SIDE_MM, PAIR_SIDE_MM, it["pitch_mm"])
        return build_l03(PAIR_SIDE_MM, PAIR_SIDE_MM, it["pitch_mm"])
    raise ValueError(k)


def ink_length(strokes):
    return sum(_poly_length_mm(s) for s in strokes)


def legend_lines():
    return ["NN-A = PITCH NN/100 MM, ANGLE A",
            "LAID WIDTH UNMEASURED",
            "N = ONE STROKE, L = 0.3 MM PIECES",
            "CNM = GRID MARK, L20 = 20 MM LINE"]


def build_layout(draw_speed: float):
    """-> elements (emission order), records (per item), fixed groups, geoms, meta."""
    items = build_items()
    elements, records, geoms = [], [], {}

    def add(pls, kind, gid):
        a = len(elements)
        for pl in pls:
            if _poly_length_mm(pl) < MIN_STROKE_MM - 1e-12:
                raise RuntimeError(f"uright_short: stroke shorter than {MIN_STROKE_MM} mm in {gid}")
            elements.append({"pts": [(float(x), float(y)) for x, y in pl],
                             "speed": draw_speed, "kind": kind, "gid": gid})
        return [a, len(elements)]

    fixed = {}
    fid_pos = [(2.0, 2.0), (SHEET_W / 2 - 2.0, 2.0), (SHEET_W - 6.0, 2.0),
               (2.0, 148.0), (SHEET_W / 2 - 2.0, 148.0), (SHEET_W - 6.0, 148.0)]
    fid_records = []
    for i, (fx, fy) in enumerate(fid_pos):
        pls = _square_rings(FID_SIDE_MM, FID_PITCH_MM, fx + FID_SIDE_MM / 2, fy + FID_SIDE_MM / 2)
        fixed[f"fid{i}"] = [[(fx, fy), (fx + FID_SIDE_MM, fy + FID_SIDE_MM)]]
        add(pls, "fiducial", f"fid{i}")
        fid_records.append({"id": f"fid{i}", "origin_mm": [fx, fy], "side_mm": FID_SIDE_MM,
                            "center_mm": [fx + FID_SIDE_MM / 2, fy + FID_SIDE_MM / 2],
                            "pitch_mm": FID_PITCH_MM, "n_rings": len(pls)})
    pls = _glyph(12.0, 1.0) + [[(24.0 + 3.0 * i, 1.0), (24.0 + 3.0 * i, 8.0)] for i in range(5)]
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
    leg = []
    for text, yb in zip(legend_lines(), LEGEND_Y):
        leg += label_strokes(text, LEGEND_CX, yb)
    fixed["legend"] = leg
    add(leg, "legend", "legend")
    n_fixed = len(elements)

    order = list(range(len(items)))
    random.Random(SEED_ORDER).shuffle(order)
    for emis, idx in enumerate(order):
        it = items[idx]
        if it["kind"] == "mark":
            local, lcx, lby = build_mark()
            placed = [_tr(s, it["ox"], it["oy"]) for s in local]
            lab = label_strokes(it["label"], it["ox"] + lcx, it["oy"] + lby)
            cx, cy = it["ox"] + MARK_W / 2.0, it["oy"] + MARK_H / 2.0
        else:
            local = build_patch(it)
            cx, cy = it["cx"], it["cy"]
            placed = [_rot(s, it["theta_deg"], cx, cy) for s in local]
            lab = label_strokes(it["label"], cx, cy - it["half"] - LABEL_GAP_MM)
        rng_patch = add(placed, it["kind"], it["id"])
        rng_lab = add(lab, "label", it["id"] + "_label")
        allpts = [q for e in elements[rng_patch[0]:rng_lab[1]] for q in e["pts"]]
        bx0, by0 = min(q[0] for q in allpts), min(q[1] for q in allpts)
        bx1, by1 = max(q[0] for q in allpts), max(q[1] for q in allpts)
        geoms[it["id"]] = box(bx0, by0, bx1, by1)
        rec = {k: v for k, v in it.items() if k not in ("half",)}
        rec.update({
            "emission_index": emis, "pen_down_start_index": rng_patch[0],
            "n_pen_downs": rng_patch[1] - rng_patch[0],
            "n_label_strokes": rng_lab[1] - rng_lab[0],
            "ink_mm": round(ink_length(placed), 4),
            "bbox_mm": [round(bx0, 3), round(by0, 3), round(bx1, 3), round(by1, 3)],
            "elem_range_patch": rng_patch, "elem_range_label": rng_lab,
            "center_mm": [round(cx, 4), round(cy, 4)]})
        if it["kind"] in ("wedge", "pair"):
            rec["n_rows"] = n_rows(it["side_mm"], it["pitch_mm"])
            rec["patch_h_mm"] = round(rec["n_rows"] * it["pitch_mm"], 4)
        records.append(rec)
    meta = {"fiducials": fid_records, "n_fixed_elements": n_fixed, "n_elements": len(elements),
            "legend": legend_lines()}
    return elements, records, fixed, geoms, meta


# ---------------------------------------------------------------------------
# Pen resolution (pinned)
# ---------------------------------------------------------------------------
_SAME_AS_BASE = ("z_touch_abs", "z_down", "z_up", "z_micro", "draw_speed", "travel_speed",
                 "plunge_speed", "lift_speed", "pen_offset_x", "pen_offset_y")


def resolve_pinned_pen(profile=None, pen_id=None):
    """-> (base_pen_dict, requested_id, note). Refuses any pen outside the uRight family;
    a family variant is generated from the BASE profile after its Z/feed fields are
    asserted equal to the base's."""
    profiles = load_pen_profiles().get("profiles", {})
    if PIN_ID not in profiles:
        raise RuntimeError(f"uright_short: pen profile {PIN_ID!r} not found in pen_profiles.json")
    base = profiles[PIN_ID]
    req = pen_id
    if profile is not None:
        matches = [k for k, v in profiles.items() if v == profile]
        if not matches:
            raise RuntimeError("uright_short: the profile passed in matches no pen currently in "
                               "pen_profiles.json by value -- refusing (stale-blob hazard).")
        blob = PIN_ID if PIN_ID in matches else matches[0]
        if req is not None and req != blob:
            raise RuntimeError(f"uright_short: pen_id={req!r} conflicts with the profile passed "
                               f"in ({blob!r}) -- refusing.")
        req = blob
    req = req or PIN_ID
    if req not in FAMILY_IDS:
        raise RuntimeError(
            f"uright_short: refusing to generate -- this sheet is PINNED to {PIN_ID!r} (uRight gel; "
            f"its variants {FAMILY_IDS[1:]} are generated from the base profile). The requested "
            f"pen is {req!r}. Mount the uRight and re-request, or POST with pen_id={PIN_ID!r}.")
    note = ""
    if req != PIN_ID:
        var = profiles[req]
        for f in _SAME_AS_BASE:
            if var.get(f) != base.get(f):
                raise RuntimeError(f"uright_short: requested {req!r} differs from {PIN_ID!r} in "
                                   f"{f!r} ({var.get(f)!r} vs {base.get(f)!r}) -- refusing.")
        note = (f"requested profile {req!r} shares {PIN_ID!r}'s z/feeds; the sheet is generated "
                f"from the BASE id {PIN_ID!r} (its fill_mm/laid_mm are not used).")
    for f in ("z_touch_abs", "z_down", "z_up", "draw_speed", "travel_speed", "plunge_speed",
              "lift_speed"):
        if f not in base:
            raise RuntimeError(f"uright_short: profile {PIN_ID!r} lacks required field {f!r}")
    return base, req, note


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------
def uright_short_gcode(out_gcode: Path, pen_mm=None, z_up=None, travel_speed=None,
                       plunge_speed=None, lift_speed=None, offset_x=None, offset_y=None,
                       end_policy="release", quiet_fans=True, profile=None, pen_id=None,
                       microlift=True):
    out_gcode = Path(out_gcode)
    ppmm = 1.0 / PX2MM
    offset_x_given = offset_x is not None
    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()
    pen, req_id, pin_note = resolve_pinned_pen(profile, pen_id)

    reach0 = active_reach_profile(cal)
    if reach0["id"] != EXPECT_REACH_ID or tuple(round(v, 1) for v in reach0["bbox"]) != EXPECT_BBOX:
        raise RuntimeError(f"Runtime reach envelope mismatch: got id={reach0['id']!r} "
                           f"bbox={reach0['bbox']}, expected {EXPECT_REACH_ID!r} {EXPECT_BBOX}.")
    got_off = (float(pen.get("pen_offset_x", 0.0)), float(pen.get("pen_offset_y", 0.0)))
    if any(abs(a) > 1e-6 for a in got_off):
        raise RuntimeError(f"pen {PIN_ID!r} pen_offset field {got_off} != (0, 0)")
    z_touch, z_touch_source = resolve_z_touch(cal, pen)
    lo, hi = Z_TOUCH_SANITY_RANGE_MM
    if not (lo <= z_touch <= hi):
        raise RuntimeError(f"z_touch_abs for {PIN_ID} resolved to {z_touch}, outside [{lo},{hi}] mm.")
    z_down = float(pen["z_down"])
    if not (-2.0 <= z_down <= 0.0):
        raise RuntimeError(f"z_down {z_down} for {PIN_ID} outside [-2, 0] mm")
    draw_speed = float(pen["draw_speed"])
    z_up = float(pen["z_up"]) if z_up is None else float(z_up)
    travel_speed = float(pen["travel_speed"]) if travel_speed is None else float(travel_speed)
    plunge_speed = float(pen["plunge_speed"]) if plunge_speed is None else float(plunge_speed)
    lift_speed = float(pen["lift_speed"]) if lift_speed is None else float(lift_speed)
    if pen_mm is None:
        pen_mm = float(pen.get("pen_mm", 0.255))     # preview stroke width only

    reach = pen_reach_profile(reach0, pen)
    pen_off_x, pen_off_y = resolve_pen_offset(cal, pen)

    elements, records, fixed_groups, geoms, meta = build_layout(draw_speed)
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
            f"uright_short: tip window X[{min(tx):.1f},{max(tx):.1f}] Y[{min(ty):.1f},{max(ty):.1f}] "
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
        "draw_speed": float(draw_speed), "travel_speed": float(travel_speed),
        "plunge_speed": float(plunge_speed), "lift_speed": float(lift_speed),
        "z_auto": "1", "origin_x": park_x, "origin_y": park_y,
        **_gcode_lifecycle_values(end_policy, quiet_fans)})
    vpype_cli.execute(f'read {shlex.quote(str(blk_svg))} pagesize {W}x{H}px '
                      f'translate -- {comp_off_x}mm {-comp_off_y}mm '
                      f'gwrite --profile elegoo_cc {shlex.quote(str(blk_gc))}')
    texts = [blk_gc.read_text(encoding="utf-8")]
    labels = [f"; BLOCK 1/1 draw_speed=F{int(draw_speed)} elements=0..{len(elements) - 1}"]
    blk_svg.unlink(missing_ok=True)
    combined = _splice_blocks(texts, labels, z_up_abs)
    combined = combined.replace(str(blk_gc), str(out_gcode))
    blk_gc.unlink(missing_ok=True)

    header = list(PREDICTION_HEADER) + [
        f"pen_id={PIN_ID} (requested {req_id}) z_touch_abs={z_touch:.3f} ({z_touch_source}) "
        f"z_down_abs={z_down_abs:.3f}",
        f"draw F{int(draw_speed)} travel F{int(travel_speed)} strokes={len(elements)} "
        f"microlift={'on' if microlift else 'off'} (all from the live profile)",
        f"seeds placement={SEED_PLACEMENT} order={SEED_ORDER}",
    ] + ([pin_note] if pin_note else []) + [
        "RECORD z_touch before plotting; BIC=18.95 / uRight=11.45 differ by 7.50 mm (DIRECTION S10).",
    ]
    header = [h.encode("ascii", "replace").decode("ascii") for h in header]
    with atomic_target(out_gcode) as tmp_gcode:
        Path(tmp_gcode).write_text(combined, encoding="utf-8")
        _prepend_ascii_header(tmp_gcode, header)

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

    pair_recs = [r for r in records if r["kind"] == "pair"]
    pair_ink = {}
    for r in pair_recs:
        pair_ink.setdefault(f"{r['pitch_mm']}", {}).setdefault(r["arm"], []).append(r["ink_mm"])
    layout = {
        "sheet": SHEET_ID, "pen_id": PIN_ID, "requested_pen_id": req_id, "pin_note": pin_note,
        "laid_width": "UNMEASURED (pitches are commanded spacings, not widths)",
        "wedge_pitches_mm": list(WEDGE_PITCHES_MM), "wedge_angles_deg": list(WEDGE_ANGLES_DEG),
        "pair_pitches_mm": list(PAIR_PITCHES_MM), "pair_reps": PAIR_REPS, "piece_mm": PIECE_MM,
        "patch_side_mm": PATCH_SIDE_MM, "pair_side_mm": PAIR_SIDE_MM, "line_len_mm": LINE_LEN_MM,
        "grid": {"cols": GRID_COLS, "rows": GRID_ROWS, "mark_x0": MARK_X0, "mark_dx": MARK_DX,
                 "mark_y": list(MARK_Y), "mark_line_mm": MARK_LINE_MM,
                 "mark_patch_mm": MARK_PATCH_MM, "mark_pitch_mm": MARK_PITCH_MM},
        "seeds": {"placement": SEED_PLACEMENT, "order": SEED_ORDER},
        "emission_order_note": "fiducials, glyph, rulers, prime, legend first; then ALL measured "
                               "items in one seeded shuffle; emission_index is the item order.",
        "sheet_wh_mm": [SHEET_W, SHEET_H], "pad_mm": pad,
        "strokes_bbox_mm": [round(min(allx), 3), round(min(ally), 3),
                            round(max(allx), 3), round(max(ally), 3)],
        "mapping": {"offset_x": offset_x, "offset_y": offset_y, "flip_axis_mm": flip_axis,
                    "comp_off_x": comp_off_x, "comp_off_y": comp_off_y,
                    "pen_off_x": pen_off_x, "pen_off_y": pen_off_y, "S": S,
                    "cmd_x": "pad + x + comp_off_x", "cmd_y": "flip + comp_off_y - (pad + y)"},
        "tip_window_mm": {"x": [min(tx), max(tx)], "y": [min(ty), max(ty)]},
        "paper_window_mm": {"x": [TIP_X_MIN, TIP_X_MAX], "y": [TIP_Y_MIN, TIP_Y_MAX]},
        "z_touch_abs": z_touch, "z_touch_source": z_touch_source, "z_down_abs": z_down_abs,
        "z_up_abs": z_up_abs, "z_short_abs": z_short_abs, "microlift": microlift_stats,
        "feeds": {"draw": draw_speed, "travel": travel_speed, "plunge": plunge_speed,
                  "lift": lift_speed, "z_up": z_up},
        "items": records, "fiducials": meta["fiducials"], "legend": meta["legend"],
        "pair_ink_mm": pair_ink,
        "rulers": {"h": {"origin_mm": [GRID_X0, 158.0], "len_mm": FID0_LEN_MM},
                   "v": {"origin_mm": [234.0, 22.0], "len_mm": FID90_LEN_MM},
                   "tick_pitch_mm": _pk.FID_TICK_PITCH_MM, "tick_h_mm": _pk.FID_TICK_H_MM,
                   "tick_dir": -1},
        "stroke_check": stroke_check, "selftest_caught": selftest_caught,
        "clearance_check": clearance_check,
        "n_pendowns": len(elements), "total_draw_mm": round(total_mm, 1),
        "time_model_a_s": round(t_a, 1), "time_model_b_s": round(t_b, 1),
    }
    layout_path = WORK_DIR / f"{out_gcode.stem}_layout.json"
    layout_path.write_text(json.dumps(layout, indent=1), encoding="utf-8")

    return {"kind": SHEET_ID, "pen_mm": pen_mm, "pen_id": PIN_ID, "requested_pen_id": req_id,
            "pin_note": pin_note, "pen_label": pen.get("label"),
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
            "plot_time_model_a_label": "model (a): per-element length/feed + fixed plunge/lift at "
                                       "full z_up per pen-down; Z-blind, high; NOT validated.",
            "preview": prev.name, "svg_path": str(prev_svg), "svg_wh_px": [W, H],
            "pattern_mm": [round(canvas_w, 1), round(canvas_h, 1)],
            "place": {"x": offset_x, "y": offset_y, "w": canvas_w, "h": canvas_h},
            "z_up_abs": round(z_up_abs, 3),
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
            "layout_path": str(layout_path), "total_draw_mm": round(total_mm, 1)}


def uright_short(out_gcode, **kw):
    return uright_short_gcode(out_gcode, **kw)
