"""ccplotter/calibration_inkroute.py -- ink-routed-connectors paper test sheet (Kai, 2026-10-29: "1-sure ... if it looks as good
then take it if there's less z moves and time"; reviewer D1902d / D1895).

QUESTION (Kai): the opt-in request key `ink_routed_connectors` (ccplotter/ink_routing.py) replaces some pen lifts by a short
pen-down path that stays inside the ink; offline it removes ~23 % of the lifts of the heli bwphoto. Does it LOOK as good ON PAPER,
and is it actually faster with the stopwatch (fewer Z moves, less time)? Offline numbers cannot say: the look at pen width on real
paper and the real clock are paper-only facts.

WHAT IS ON THE SHEET. ONE 100 x 70 mm window of the heli bwphoto full-page job (production defaults, BIC 0.12, stay-down over ink
ON), drawn TWICE side by side:
  A OFF  (left)   today's production output (request key absent)                     -- plots FIRST
  B ON   (right)  the same job with `ink_routed_connectors: true`                    -- plots SECOND, after ALL of A
Both halves are the EMITTED pen-down strokes of the real /api/process gcode (in emitted order), clipped to the window (not
re-traced), so the connectors, the stroke order and the simplification are exactly what the app produced. The window contains the
registered G3b hot spot of the A/B audit (page mm 47-55 x 140-146).

HALVES DO NOT INTERLEAVE (Kai stopwatches each half): all fixed items (fiducials, glyph, title, rulers, prime, labels, corner marks)
come first, then every stroke of half A, an ASCII comment marker, then every stroke of half B. The two marker lines in the gcode
('; ===== HALF A ...' / '; ===== HALF B ...') are the stopwatch cut points.

PRECOMPUTED ASSET (D1023): the DATA file assets/calibration/inkroute_asset.json.xz (PyInstaller datas entry in desktop/ccplotter.spec),
NOT embedded here. Its sha256 and payload sha256 are constants below and CHECKED at load; a mismatch refuses to generate. Built by
artifacts/inkroute-paper-sheet-2026-10-29/scripts/build_asset.py.

DISCLOSED DIFFERENCES FROM A FULL-PAGE PLOT: only the window is drawn; strokes crossing a window edge are cut there and each cut
piece is its own pen-down (border pieces, counted in the ledger; the window lift counts are pieces, NOT the page-wide lift counts);
pieces < 0.05 mm are dropped; microlift is applied as production does; linesimplify/linesort are NOT re-run. The window is NOT a
page-wide prediction: the page-wide lift reduction is larger or smaller than the window's.

PEN IS PINNED to bic_cristal_blue (fill_mm 0.12; z_touch READ from the profile at generation time and printed in the gcode header,
never hardcoded). Printed per half: 'A OFF' / 'B ON', the pen-down (lift) count and the MODEL time for that half. Model time is
model (b) (per-move length / feed over the emitted gcode, Z included, microlift applied, no acceleration): a floor, NOT
stopwatch-validated. Half A always plots before half B, so slow ink warm-up and pen drift are not randomised away: say so when
reading the stopwatch.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import re
import shlex
import statistics
import sys
from pathlib import Path

import vpype as vp
import vpype_cli
from shapely.geometry import box

from ccplotter.atomic_io import atomic_target
from ccplotter.config import CONFIG_PATH, PREVIEW_COLOR, PX2MM, WORK_DIR
from ccplotter.cal import (resolve_pen_offset, load_cal, load_pen_profiles, resolve_z_touch, resolve_z_micro)
from ccplotter.gcode_post import (_gcode_lifecycle_values, validate_calibration_gcode_safety, _microlift_gcode)
from ccplotter.reach import active_reach_profile, validate_gcode_xy_motion, pen_reach_profile
from ccplotter.calibration_deposition import _splice_blocks
from ccplotter import calibration_pocket as _pk
from ccplotter.calibration_fragment import (label_strokes, time_model_a, time_model_b)
from ccplotter.calibration_pocket import (
    EXPECT_REACH_ID, EXPECT_BBOX, EXPECT_Z_DOWN_STOCK, EXPECT_DRAW_SPEED_STOCK,
    Z_TOUCH_SANITY_RANGE_MM, TIP_X_MIN, TIP_X_MAX, TIP_Y_MIN, TIP_Y_MAX,
    X0_TIP_DEFAULT, Y_FRONT_TIP_DEFAULT, SHEET_W, SHEET_H, FID_SIDE_MM, FID_PITCH_MAX_FRAC_OF_PEN,
    FID0_LEN_MM, FID90_LEN_MM, GRID_X0, FAST, MIN_STROKE_MM,
    _poly_length_mm, _square_rows, _square_rings, _ticks, _glyph,
    _write_svg_hp, _prepend_ascii_header, _parse_strokes_f, _check_clearance, check_strokes,
)

_PAD_MM = 3.0
PEN_ID = "bic_cristal_blue"
PEN_LABEL = "BIC Cristal 1.0 Blue"
PEN_MM = 1.0
PITCH_MM = 0.12

ASSET_REL = Path("assets") / "calibration" / "inkroute_asset.json.xz"
ASSET_FILE_SHA256 = "068448c9632afef58114525da4a3a322dff3ee981f3fe390a5dea72e7b03ce6d"      # written by artifacts/inkroute-paper-sheet-2026-10-29/scripts/build_asset.py
ASSET_PAYLOAD_SHA256 = "ca8ee7bee7d25cf0e2bfd6f43480cf4150884efcb680f6915ca40e022d853117"    # written by build_asset.py

ARMS = ("off", "on")
ARM_LABEL = {"off": "A OFF", "on": "B ON"}
ARM_HALF = {"off": "A", "on": "B"}
EMISSION_ORDER = ("off", "on")          # A first, then B: NEVER interleaved (stopwatch cut points)
MARKER_PREFIX = {"off": "; ===== HALF A", "on": "; ===== HALF B"}

# ---- geometry of the sheet (237 x 161 mm; tip X = 10 + x, tip Y = 169 - y) ----
COL_X0, COL_PITCH = 14.0, 108.0
WIN_Y0 = 42.0
LABEL_BASE_DY = 3.0                    # label baseline this far above the window top (cap height 2.0)
STAT_BASE_DY = 6.0                     # stats-line baseline this far below the window bottom
MARK_OFF_MM, MARK_LEN_MM = 0.4, 1.0
CLEAR_MM = 0.2
TITLE_TEXT = "INKROUTE BIC nominal 0.12"
TITLE_CX, TITLE_BASE_Y = 78.0, 6.5
PRIME_CX, PRIME_CY = 216.0, 4.5
SHORT_PIECE_MM = 0.5
TIME_LABEL_ITERS = 4


class AssetError(RuntimeError):
    pass


def asset_path() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / ASSET_REL.name
    return Path(__file__).resolve().parent.parent / ASSET_REL


_ASSET_CACHE = {}


def load_asset(path: Path | None = None, file_sha: str | None = None, payload_sha: str | None = None):
    """Verified (meta, cells). cells = {arm: [(pts[(x, y) mm rel. to the window origin], border_flag), ...]} in EMITTED order.
    Refuses to return anything if either sha does not match."""
    key = (str(path) if path else None, file_sha, payload_sha)
    if key in _ASSET_CACHE:
        return _ASSET_CACHE[key]
    p = Path(path) if path else asset_path()
    if not p.is_file():
        raise AssetError(f"inkroute asset missing: {p}")
    xz = p.read_bytes()
    if hashlib.sha256(xz).hexdigest() != (file_sha or ASSET_FILE_SHA256):
        raise AssetError("inkroute asset file sha mismatch: refusing to generate. Rebuild with build_asset.py; do not plot.")
    try:
        raw = lzma.decompress(xz)
    except lzma.LZMAError as e:
        raise AssetError(f"inkroute asset not decompressible: {e}")
    if hashlib.sha256(raw).hexdigest() != (payload_sha or ASSET_PAYLOAD_SHA256):
        raise AssetError("inkroute asset payload sha mismatch: refusing to generate.")
    d = json.loads(raw.decode("utf-8"))
    meta = d["meta"]
    q = meta["quant_um"] / 1000.0
    cells = {}
    for arm in meta["arms"]:
        pcs = d["cells"][arm]
        cells[arm] = [([(x * q, y * q) for x, y in pc["p"]], bool(pc["b"])) for pc in pcs]
        c = meta["counts"][arm]
        nv = sum(len(pc["p"]) for pc in pcs)
        if (len(pcs), nv, sum(pc["b"] for pc in pcs)) != (c["n_pieces"], c["n_verts"], c["n_border"]):
            raise AssetError(f"inkroute asset: arm {arm} counts do not match meta")
    _ASSET_CACHE[key] = (meta, cells)
    return meta, cells


def cell_origin(c: int):
    return COL_X0 + c * COL_PITCH, WIN_Y0


def _corner_marks(x0, y0, x1, y1):
    o, L = MARK_OFF_MM, MARK_LEN_MM
    return [[(x0 - o, y0 - o + L), (x0 - o, y0 - o), (x0 - o + L, y0 - o)],
            [(x1 + o - L, y0 - o), (x1 + o, y0 - o), (x1 + o, y0 - o + L)],
            [(x0 - o, y1 + o - L), (x0 - o, y1 + o), (x0 - o + L, y1 + o)],
            [(x1 + o - L, y1 + o), (x1 + o, y1 + o), (x1 + o, y1 + o - L)]]


def fmt_min(seconds: float) -> str:
    return f"{seconds / 60.0:.1f}"


def stat_text(n_lifts: int, t_min_str: str) -> str:
    return f"{n_lifts} LIFTS {t_min_str} MIN MODEL"


def build_layout(meta=None, cells=None, times_min=None):
    """times_min: {arm: 'xx.x'} the printed model minutes (strings). Default placeholder '0.0' (the generator iterates to the measured value)."""
    if meta is None:
        meta, cells = load_asset()
    times_min = times_min or {}
    w_mm, h_mm = meta["win_w_mm"], meta["win_h_mm"]
    elements, records, geoms, fixed = [], [], {}, {}

    def add(pls, kind, gid):
        a = len(elements)
        for pl in pls:
            if len(pl) >= 2 and _poly_length_mm(pl) >= MIN_STROKE_MM - 1e-12:
                elements.append({"pts": [(float(x), float(y)) for x, y in pl], "speed": FAST, "kind": kind, "gid": gid})
        return [a, len(elements)]

    fid_pos = [(2.0, 2.0), (SHEET_W / 2 - 2.0, 2.0), (SHEET_W - 6.0, 2.0),
               (2.0, 148.0), (SHEET_W / 2 - 2.0, 148.0), (SHEET_W - 6.0, 148.0)]
    fid_records = []
    fp = min(PITCH_MM, FID_PITCH_MAX_FRAC_OF_PEN * PEN_MM)
    for i, (fx, fy) in enumerate(fid_pos):
        pls = _square_rings(FID_SIDE_MM, fp, fx + FID_SIDE_MM / 2, fy + FID_SIDE_MM / 2)
        fixed[f"fid{i}"] = [[(fx, fy), (fx + FID_SIDE_MM, fy + FID_SIDE_MM)]]
        add(pls, "fiducial", f"fid{i}")
        fid_records.append({"id": f"fid{i}", "origin_mm": [fx, fy], "side_mm": FID_SIDE_MM,
                            "center_mm": [fx + FID_SIDE_MM / 2, fy + FID_SIDE_MM / 2], "pitch_mm": fp, "n_rings": len(pls)})
    pls = _glyph(12.0, 1.0) + [[(24.0 + 3.0 * i, 1.0), (24.0 + 3.0 * i, 8.0)] for i in range(2)]      # two bars: not the rowangle sheet's three
    fixed["glyph"] = pls
    add(pls, "glyph", "glyph")
    title = label_strokes(TITLE_TEXT, TITLE_CX, TITLE_BASE_Y)
    fixed["title"] = title
    add(title, "title", "title")
    r0 = [[(GRID_X0, 158.0), (GRID_X0 + FID0_LEN_MM, 158.0)]] + _ticks(GRID_X0, 158.0, FID0_LEN_MM, "h", -1)
    r90 = [[(234.0, 22.0), (234.0, 22.0 + FID90_LEN_MM)]] + _ticks(234.0, 22.0, FID90_LEN_MM, "v", -1)
    fixed["ruler0"], fixed["ruler90"] = r0, r90
    add(r0, "ruler", "ruler0")
    add(r90, "ruler", "ruler90")
    prime = _square_rows(_pk.PRIME_SIDE_MM, _pk.PRIME_PITCH_MM, PRIME_CX, PRIME_CY)
    fixed["prime"] = prime
    add(prime, "prime", "prime")

    # labels (both halves) and corner marks (both halves) BEFORE any half data; the corner marks are emitted LAST so the pen position
    # entering half A is independent of the printed numbers (the per-half model time then does not depend on the label text).
    prep = {}
    for arm in EMISSION_ORDER:
        c = ARMS.index(arm)
        gid = f"patch|{arm}"
        ox, oy = cell_origin(c)
        x1, y1 = ox + w_mm, oy + h_mm
        n_lifts = len(cells[arm])
        kept_n = sum(1 for pts, _b in cells[arm] if len(pts) >= 2 and _poly_length_mm(pts) >= MIN_STROKE_MM - 1e-12)
        lab_text = ARM_LABEL[arm]
        stat = stat_text(kept_n, times_min.get(arm, "0.0"))
        lab = label_strokes(lab_text, ox + w_mm / 2.0, oy - LABEL_BASE_DY)
        lab_stat = label_strokes(stat, ox + w_mm / 2.0, y1 + STAT_BASE_DY)
        for ls, txt in ((lab, lab_text), (lab_stat, stat)):
            lx = [p[0] for pl in ls for p in pl]
            if min(lx) < ox - 0.5 or max(lx) > x1 + 0.5:
                raise RuntimeError(f"inkroute: label {txt!r} is wider than the {w_mm} mm window")
        rng_lab = add(lab, "label", gid + "|label")
        rng_stat = add(lab_stat, "label", gid + "|stat")
        prep[arm] = {"gid": gid, "ox": ox, "oy": oy, "x1": x1, "y1": y1, "lab": lab, "lab_stat": lab_stat, "stat_text": stat,
                     "rng_lab": rng_lab, "rng_stat": rng_stat, "lab_text": lab_text, "n_lifts": kept_n, "c": c}
    for arm in EMISSION_ORDER:
        p = prep[arm]
        p["marks"] = _corner_marks(p["ox"], p["oy"], p["x1"], p["y1"])
        p["rng_marks"] = add(p["marks"], "mark", p["gid"] + "|marks")
    n_fixed = len(elements)
    for emis, arm in enumerate(EMISSION_ORDER):
        p = prep[arm]
        gid, ox, oy, x1, y1 = p["gid"], p["ox"], p["oy"], p["x1"], p["y1"]
        pieces = cells[arm]
        data_pls = [[(ox + px, oy + py) for px, py in pts] for pts, _b in pieces]
        data_pls = [[(min(max(x, ox), x1), min(max(y, oy), y1)) for x, y in pl] for pl in data_pls]
        rng_data = add(data_pls, "data", gid)
        kept_flags = [bool(b) for (pts, b), pl in zip(pieces, data_pls)
                      if len(pl) >= 2 and _poly_length_mm(pl) >= MIN_STROKE_MM - 1e-12]
        if len(kept_flags) != rng_data[1] - rng_data[0]:
            raise RuntimeError("inkroute: border-flag / element misalignment")
        geoms[gid] = box(ox, oy, x1, y1)
        fixed[gid + "|label"] = p["lab"]
        fixed[gid + "|stat"] = p["lab_stat"]
        for mi, m in enumerate(p["marks"]):
            fixed[f"{gid}|mark{mi}"] = [m]
        records.append({"id": gid, "arm": arm, "half": ARM_HALF[arm], "col": p["c"], "label": p["lab_text"], "stat_label": p["stat_text"],
                        "window_mm": [round(ox, 3), round(oy, 3), round(x1, 3), round(y1, 3)], "emission_index": emis,
                        "elem_range_label": p["rng_lab"], "elem_range_stat": p["rng_stat"], "elem_range_marks": p["rng_marks"],
                        "elem_range_data": rng_data, "border_flags": kept_flags})
    return elements, records, fixed, geoms, {"n_elements": len(elements), "n_fixed": n_fixed, "fiducials": fid_records}


def clearance_check(geoms, fixed):
    return _check_clearance(geoms, fixed, min_mm=CLEAR_MM)


LEDGER_POPULATION = ("population = every pen-down stroke the sheet draws inside that half's window (100 x 70 mm), counted from "
                     "the EMITTED gcode after clipping and after dropping pieces < 0.05 mm; 'border' pieces touch the window "
                     "edge (clip artifacts of the cut), 'interior' pieces do not. These counts are WINDOW pieces, not the page-wide "
                     "lift counts of the full job.")


def ledger(records, strokes):
    per = []
    for rec in records:
        a, b = rec["elem_range_data"]
        sub = strokes[a:b]
        lens = [_poly_length_mm(s["pts"]) for s in sub]
        flags = rec["border_flags"]
        if len(lens) != len(flags):
            raise RuntimeError("inkroute: ledger misaligned")
        per.append({"id": rec["id"], "arm": rec["arm"], "half": rec["half"], "pen_downs": len(sub), "draw_mm": round(sum(lens), 2),
                    "vertices": sum(len(s["pts"]) for s in sub), "border_pieces": int(sum(flags)),
                    "lt_0p5mm": sum(1 for x in lens if x < SHORT_PIECE_MM),
                    "median_piece_mm": round(statistics.median(lens), 3) if lens else 0.0})
    return {"population": LEDGER_POPULATION, "per_patch": per}


_MOVE_RE = {a: re.compile(a + r"(-?[\d.]+)") for a in "XYZF"}


def half_times(gcode_text: str):
    """Model (b) seconds per half from the EMITTED gcode text (same arithmetic as calibration_fragment.time_model_b: per-move length /
    modal feed, Z included, no acceleration), cut at the two marker comment lines. A = marker A .. marker B; B = marker B .. first M400
    (the postamble). Returns {'off': s, 'on': s, 'total': s}. Raises if a marker is missing, duplicated or out of order."""
    import math
    lines = gcode_text.splitlines()
    idx = {}
    for arm, pre in MARKER_PREFIX.items():
        hits = [i for i, l in enumerate(lines) if l.startswith(pre)]
        if len(hits) != 1:
            raise RuntimeError(f"inkroute: marker {pre!r} found {len(hits)} times (expected 1)")
        idx[arm] = hits[0]
    m400 = next((i for i, l in enumerate(lines) if l.strip() == "M400"), None)
    if not (idx["off"] < idx["on"] < (m400 if m400 is not None else 10 ** 9)):
        raise RuntimeError("inkroute: half markers are not in the order A, B, postamble")
    pos = {"X": None, "Y": None, "Z": None}
    f = 3000.0
    acc = {"pre": 0.0, "off": 0.0, "on": 0.0, "post": 0.0}
    for i, line in enumerate(lines):
        if not line.startswith(("G0", "G1")):
            continue
        mf = _MOVE_RE["F"].search(line)
        if mf:
            f = float(mf.group(1))
        d2 = 0.0
        for ax in "XYZ":
            m = _MOVE_RE[ax].search(line)
            if m:
                v = float(m.group(1))
                if pos[ax] is not None:
                    d2 += (v - pos[ax]) ** 2
                pos[ax] = v
        seg = "pre" if i < idx["off"] else ("off" if i < idx["on"] else ("on" if (m400 is None or i < m400) else "post"))
        acc[seg] += math.sqrt(d2) / max(1.0, f / 60.0)
    return {"off": acc["off"], "on": acc["on"], "total": sum(acc.values())}


def halves_in_order(gcode_text: str, lifts):
    """True iff the emitted file has marker A before marker B and EXACTLY lifts['off'] pen-downs between A and B and lifts['on'] after B
    (so half A is wholly before half B, nothing interleaves). Pen-down = a Z move down to the file's deepest Z."""
    lines = gcode_text.splitlines()
    ia = next(i for i, l in enumerate(lines) if l.startswith(MARKER_PREFIX["off"]))
    ib = next(i for i, l in enumerate(lines) if l.startswith(MARKER_PREFIX["on"]))
    if not ia < ib:
        return False
    zs = [float(m.group(1)) for l in lines if l.startswith(("G0", "G1")) and (m := _MOVE_RE["Z"].search(l))]
    zdown = min(zs) + 0.01

    def downs(a, b):
        n, was = 0, False
        for l in lines[a:b]:
            if not l.startswith(("G0", "G1")):
                continue
            m = _MOVE_RE["Z"].search(l)
            if m and not _MOVE_RE["X"].search(l):
                d = float(m.group(1)) <= zdown
                if d and not was:
                    n += 1
                was = d
        return n
    m400 = next((i for i, l in enumerate(lines) if l.strip() == "M400"), len(lines))
    return downs(ia, ib) == lifts["off"] and downs(ib, m400) == lifts["on"]


def _emit(out_gcode, meta, cells, times_min, *, pen, cal, reach0, reach, pen_off_x, pen_off_y, z_touch, z_touch_source, z_down,
          z_up, travel_speed, plunge_speed, lift_speed, offset_x, offset_y, end_policy, quiet_fans, microlift, pen_mm):
    """One full generation pass. Returns everything the caller needs; writes out_gcode."""
    ppmm = 1.0 / PX2MM
    offset_x_given = offset_x is not None
    elements, records, fixed_groups, geoms, lmeta = build_layout(meta, cells, times_min)
    clr = clearance_check(geoms, fixed_groups)
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
            f"inkroute: tip window X[{min(tx):.1f},{max(tx):.1f}] Y[{min(ty):.1f},{max(ty):.1f}] outside the paper window "
            f"X[{TIP_X_MIN},{TIP_X_MAX}] Y[{TIP_Y_MIN},{TIP_Y_MAX}] (B5 landscape, holes at the FRONT; frame moved per D1487).")
    park_x, park_y = reach["safe_park_point"]
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))
    z_down_abs = z_touch + z_down
    z_up_abs = z_touch + float(z_up)

    # THREE blocks, spliced in order: [fixed items + labels + corner marks] [half A strokes] [half B strokes]
    rA, rB = records[0]["elem_range_data"], records[1]["elem_range_data"]
    n_fixed = lmeta["n_fixed"]
    if not (rA[0] == n_fixed and rA[1] == rB[0] and rB[1] == len(elements)):
        raise RuntimeError("inkroute: element ranges do not form fixed | A | B")
    spans = [(0, n_fixed), tuple(rA), tuple(rB)]
    placed_all = [[((x + pad) * ppmm, (y + pad) * ppmm) for x, y in e["pts"]] for e in elements]
    vp.config_manager.load_config_file(str(CONFIG_PATH))
    vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({
        "z_up": float(z_up), "z_up_abs": z_up_abs, "z_down": z_down, "z_down_abs": z_down_abs,
        "z_touch_abs": z_touch, "z_travel_abs": z_travel_abs,
        "draw_speed": float(FAST), "travel_speed": float(travel_speed),
        "plunge_speed": float(plunge_speed), "lift_speed": float(lift_speed),
        "z_auto": "1", "origin_x": park_x, "origin_y": park_y,
        **_gcode_lifecycle_values(end_policy, quiet_fans)})
    # ONE scratch name reused for every block: gwrite stamps its source path into the preamble and _splice_blocks requires every
    # block's preamble to be byte-identical (the path is replaced by out_gcode after the splice).
    blk_svg = WORK_DIR / f".{out_gcode.stem}_blk.svg"
    blk_gc = WORK_DIR / f".{out_gcode.stem}_blk.gcode"
    texts = []
    for a, b in spans:
        _write_svg_hp(blk_svg, W, H, float(pen_mm) * ppmm, placed_all[a:b])
        vpype_cli.execute(f'read {shlex.quote(str(blk_svg))} pagesize {W}x{H}px '
                          f'translate -- {comp_off_x}mm {-comp_off_y}mm '
                          f'gwrite --profile elegoo_cc {shlex.quote(str(blk_gc))}')
        texts.append(blk_gc.read_text(encoding="utf-8"))
    blk_svg.unlink(missing_ok=True)
    labels = [f"; BLOCK 1/3 fixed items: fiducials, glyph, title, rulers, prime, labels, corner marks (elements 0..{n_fixed - 1})",
              f"{MARKER_PREFIX['off']} (A OFF, ink_routed_connectors absent) BEGINS: elements {rA[0]}..{rA[1] - 1}, "
              f"{rA[1] - rA[0]} pen-downs -- START THE STOPWATCH FOR HALF A WHEN THE PEN REACHES THIS POINT =====",
              f"{MARKER_PREFIX['on']} (B ON, ink_routed_connectors true) BEGINS: elements {rB[0]}..{rB[1] - 1}, "
              f"{rB[1] - rB[0]} pen-downs -- half A is finished; STOP HALF A / START HALF B HERE ====="]
    combined = _splice_blocks(texts, labels, z_up_abs)
    combined = combined.replace(str(blk_gc), str(out_gcode))
    blk_gc.unlink(missing_ok=True)
    return {"elements": elements, "records": records, "fixed": fixed_groups, "geoms": geoms, "lmeta": lmeta, "clr": clr,
            "W": W, "H": H, "canvas_w": canvas_w, "canvas_h": canvas_h, "offset_x": offset_x, "offset_y": offset_y,
            "flip_axis": flip_axis, "comp_off_x": comp_off_x, "comp_off_y": comp_off_y, "expect_xy": expect_xy,
            "tx": tx, "ty": ty, "allx": allx, "ally": ally, "z_down_abs": z_down_abs, "z_up_abs": z_up_abs,
            "z_travel_abs": z_travel_abs, "pad": pad, "default_placement": default_placement, "placed_all": placed_all,
            "combined": combined}


def inkroute_gcode(out_gcode: Path, pen_mm=None, z_up=3.5, travel_speed=8000, plunge_speed=500,
                   lift_speed=2000, offset_x=None, offset_y=None, end_policy="release",
                   quiet_fans=True, profile=None, microlift=True):
    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()
    profiles = load_pen_profiles()
    if PEN_ID not in profiles.get("profiles", {}):
        raise RuntimeError(f"pen profile {PEN_ID!r} not found in pen_profiles.json")
    pen = profiles["profiles"][PEN_ID]
    if profile is not None and profile != pen:
        raise RuntimeError(
            f"inkroute: refusing to generate -- this sheet is PINNED to {PEN_ID!r} ({PEN_LABEL!r}); the requested/active "
            f"profile is {profile.get('label')!r}. Mount {PEN_ID} and re-request, or POST with pen_id={PEN_ID!r} explicitly.")
    reach0 = active_reach_profile(cal)
    if reach0["id"] != EXPECT_REACH_ID or tuple(round(v, 1) for v in reach0["bbox"]) != EXPECT_BBOX:
        raise RuntimeError(f"Runtime reach envelope mismatch: got id={reach0['id']!r} bbox={reach0['bbox']}, "
                           f"expected {EXPECT_REACH_ID!r} {EXPECT_BBOX}.")
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
        raise RuntimeError(f"fill_mm for {PEN_ID} = {pitch_mm}; the strokes in this sheet were made at the locked "
                           f"{PITCH_MM} mm BIC pitch -- refusing (the fill slider is set; Reset it for this sheet).")

    reach = pen_reach_profile(reach0, pen)
    if pen_mm is None:
        pen_mm = PEN_MM
    pen_off_x, pen_off_y = resolve_pen_offset(cal, pen)
    meta, cells = load_asset()                      # sha-verified or AssetError

    kw = dict(pen=pen, cal=cal, reach0=reach0, reach=reach, pen_off_x=pen_off_x, pen_off_y=pen_off_y, z_touch=z_touch,
              z_touch_source=z_touch_source, z_down=z_down, z_up=z_up, travel_speed=travel_speed, plunge_speed=plunge_speed,
              lift_speed=lift_speed, offset_x=offset_x, offset_y=offset_y, end_policy=end_policy, quiet_fans=quiet_fans,
              microlift=microlift, pen_mm=pen_mm)

    # The printed MODEL minutes are measured from the emitted gcode. They are independent of the label text (labels precede the corner
    # marks, which are the last items before half A), so the loop below converges at once; it is a loop only to PROVE that.
    times_min = {a: "0.0" for a in ARMS}
    last = None
    for _ in range(TIME_LABEL_ITERS):
        st = _emit(out_gcode, meta, cells, times_min, **kw)
        _finalize(out_gcode, st, meta, pen, pitch_mm, z_touch, z_touch_source, microlift, z_up, cal)
        ht = half_times(Path(out_gcode).read_text(encoding="ascii"))
        new = {a: fmt_min(ht[a]) for a in ARMS}
        if new == times_min:
            last = (st, ht)
            break
        times_min = new
    if last is None:
        raise RuntimeError(f"inkroute: printed model minutes did not converge in {TIME_LABEL_ITERS} passes: {times_min}")
    st, ht = last
    return _report(out_gcode, st, ht, meta, pen, pitch_mm, z_touch, z_touch_source, z_down, draw_speed, reach0, pen_mm,
                   plunge_speed, lift_speed, z_up, end_policy, quiet_fans, microlift, cal, times_min)


def _finalize(out_gcode, st, meta, pen, pitch_mm, z_touch, z_touch_source, microlift, z_up, cal):
    """Header, microlift (as production), written atomically to out_gcode. Returns nothing; validation happens in _report."""
    elements = st["elements"]
    z_down_abs, z_up_abs, z_travel_abs = st["z_down_abs"], st["z_up_abs"], st["z_travel_abs"]
    header = [
        "INK-ROUTED CONNECTORS SHEET: ONE heli bwphoto window (100 x 70 mm) drawn TWICE: HALF A = OFF (production today), HALF B = ON",
        "(request key ink_routed_connectors true). Real full-page /api/process gcode strokes clipped to the window at page scale 1.",
        "WHAT TO DO: time each half with a stopwatch (two marker lines below: '; ===== HALF A' and '; ===== HALF B'); compare the look",
        "of A and B at plot scale; RE-READ the live z_touch at plot time and compare it with the value printed on the next line.",
        "ALL of half A plots before ALL of half B (no interleaving). Label each half you time: model time printed on the sheet is a floor.",
        f"asset file sha256={ASSET_FILE_SHA256} payload={ASSET_PAYLOAD_SHA256}",
        f"pen_id={PEN_ID} z_touch_abs={z_touch:.3f} ({z_touch_source}) z_down_abs={z_down_abs:.3f}",
        f"pitch_mm={pitch_mm} (profile fill_mm) draw F{FAST} strokes={len(elements)} microlift={'on' if microlift else 'off'}",
        f"RECORD the resolved z_touch before plotting (this sheet: {z_touch:.3f}); the active pen profile drifts (DIRECTION S10).",
        "Label this sheet: nominal 0.12, laid unknown. EXPERIMENTAL: ink_routed_connectors is opt-in and OFF by default.",
    ]
    with atomic_target(out_gcode) as tmp_gcode:
        Path(tmp_gcode).write_text(st["combined"], encoding="utf-8")
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
            out_gcode, z_up_abs, z_short_abs=(z_short_abs if do_micro else None), short_mm=micro_hop_mm,
            z_mega_abs=(z_travel_abs if do_mega else None), long_mm=mega_mm)
        microlift_stats["micro_hop_mm"] = micro_hop_mm if do_micro else None
        microlift_stats["z_micro_source"] = z_micro_source
    st["microlift_stats"] = microlift_stats
    st["z_short_abs"] = z_short_abs
    return None


def _report(out_gcode, st, ht, meta, pen, pitch_mm, z_touch, z_touch_source, z_down, draw_speed, reach0, pen_mm, plunge_speed,
            lift_speed, z_up, end_policy, quiet_fans, microlift, cal, times_min):
    elements, records = st["elements"], st["records"]
    z_down_abs, z_up_abs = st["z_down_abs"], st["z_up_abs"]
    reach = pen_reach_profile(reach0, pen)
    validate_gcode_xy_motion(out_gcode, reach)
    safety = validate_calibration_gcode_safety(out_gcode, z_down_abs)
    strokes = _parse_strokes_f(out_gcode, z_down_abs)
    stroke_check = check_strokes(strokes, elements, st["expect_xy"])
    selftest_caught = _pk.selftest()
    led = ledger(records, strokes)
    total_mm = sum(_poly_length_mm(s["pts"]) for s in strokes)
    t_a = time_model_a(elements, plunge_speed, lift_speed, z_up)
    t_b = time_model_b(out_gcode)
    if abs(t_b - ht["total"]) > 1e-6:
        raise RuntimeError(f"inkroute: half_times total {ht['total']} != time_model_b {t_b}")
    lifts = {p["arm"]: p["pen_downs"] for p in led["per_patch"]}
    gtext = Path(out_gcode).read_text(encoding="ascii")
    if not halves_in_order(gtext, lifts):
        raise RuntimeError("inkroute: halves are not ordered A then B in the emitted gcode")
    for rec in records:                      # the printed numbers must be the measured ones
        want = stat_text(lifts[rec["arm"]], fmt_min(ht[rec["arm"]]))
        if rec["stat_label"] != want:
            raise RuntimeError(f"inkroute: printed stats {rec['stat_label']!r} != measured {want!r}")
    W, H, pad = st["W"], st["H"], st["pad"]
    _po = resolve_pen_offset(cal, pen)
    prev_svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_svg_hp(prev_svg, W, H, float(pen_mm) * (1.0 / PX2MM), st["placed_all"])
    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(prev_svg))} color {PREVIEW_COLOR} write {shlex.quote(str(tmp_prev))}')

    halves = {r["arm"]: {"half": r["half"], "label": r["label"], "pen_downs": lifts[r["arm"]],
                         "model_b_s": round(ht[r["arm"]], 1), "model_b_min": round(ht[r["arm"]] / 60.0, 2)} for r in records}
    layout = {
        "sheet": "inkroute", "pen_id": PEN_ID, "pitch_mm": pitch_mm,
        "question": "Does ink_routed_connectors (B ON) look as good as today's output (A OFF) on paper, and is it faster by the stopwatch?",
        "label_rule": "nominal 0.12, laid unknown",
        "stopwatch": "time each half between its marker line and the next; half A plots entirely before half B; model minutes are a floor (model b), NOT stopwatch-validated",
        "asset": {"file_sha256": ASSET_FILE_SHA256, "payload_sha256": ASSET_PAYLOAD_SHA256, "quant_um": meta["quant_um"],
                  "source": meta["source"], "window_page_mm": meta["window_page_mm"]},
        "arms": list(ARMS), "arm_labels": ARM_LABEL, "emission_order": list(EMISSION_ORDER), "halves": halves,
        "printed_minutes": times_min,
        "win_w_mm": meta["win_w_mm"], "win_h_mm": meta["win_h_mm"],
        "grid": {"col_x0": COL_X0, "col_pitch": COL_PITCH, "win_y0": WIN_Y0, "label_base_dy": LABEL_BASE_DY, "stat_base_dy": STAT_BASE_DY},
        "sheet_wh_mm": [SHEET_W, SHEET_H], "pad_mm": pad,
        "strokes_bbox_mm": [round(min(st["allx"]), 3), round(min(st["ally"]), 3), round(max(st["allx"]), 3), round(max(st["ally"]), 3)],
        "mapping": {"offset_x": st["offset_x"], "offset_y": st["offset_y"], "flip_axis_mm": st["flip_axis"], "comp_off_x": st["comp_off_x"],
                    "comp_off_y": st["comp_off_y"], "pen_off_x": _po[0], "pen_off_y": _po[1],
                    "cmd_x": "pad + x + comp_off_x", "cmd_y": "flip + comp_off_y - (pad + y)", "tip_x": "cmd_x + pen_off_x",
                    "tip_y": "cmd_y + pen_off_y"},
        "tip_window_mm": {"x": [min(st["tx"]), max(st["tx"])], "y": [min(st["ty"]), max(st["ty"])]},
        "paper_window_mm": {"x": [TIP_X_MIN, TIP_X_MAX], "y": [TIP_Y_MIN, TIP_Y_MAX]},
        "z_touch_abs": z_touch, "z_touch_source": z_touch_source, "z_down_abs": z_down_abs, "z_up_abs": z_up_abs,
        "z_short_abs": st["z_short_abs"], "microlift": st["microlift_stats"],
        "cells": [{k: v for k, v in r.items() if k != "border_flags"} for r in records],
        "fiducials": st["lmeta"]["fiducials"],
        "ledger": led, "stroke_check": stroke_check, "selftest_caught": selftest_caught, "clearance_check": st["clr"],
        "time_model_a_s": round(t_a, 1), "time_model_b_s": round(t_b, 1),
    }
    layout_path = WORK_DIR / f"{out_gcode.stem}_layout.json"
    layout_path.write_text(json.dumps(layout, indent=1), encoding="utf-8")

    return {"kind": "inkroute", "pen_mm": pen_mm, "pen_id": PEN_ID, "pen_label": PEN_LABEL,
            "footprint_mm": [round(st["canvas_w"], 1), round(st["canvas_h"], 1)], "offset_mm": [st["offset_x"], st["offset_y"]],
            "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source, "z_down": z_down,
            "z_down_abs": round(z_down_abs, 3), "draw_speed": draw_speed, "reach_id": reach0["id"],
            "reach_bbox": reach0["bbox"], "n_pendown_expected_total": len(elements),
            "n_pendown_actual_total": stroke_check["n_strokes"],
            "pendown_count_matches": stroke_check["n_strokes"] == len(elements),
            "deepest_z_down_abs": round(z_down_abs, 3), "safety_selfcheck": safety, "stroke_check": stroke_check,
            "clearance_check": st["clr"], "microlift": st["microlift_stats"], "tip_window_mm": layout["tip_window_mm"],
            "default_placement": st["default_placement"], "plot_time_estimate_s": round(t_b, 1),
            "plot_time_estimate_min": round(t_b / 60.0, 2),
            "plot_time_model_label": "model (b): per-move length / modal feed over the emitted gcode, Z included, microlift "
                                     "applied, no acceleration/dwell; a floor, NOT stopwatch-validated.",
            "plot_time_model_a_min": round(t_a / 60.0, 2),
            "plot_time_model_a_label": "model (a): draw length / feed + n_pen_downs x (plunge + lift) at nominal Z travel, "
                                       "Z-blind to microlift; ~2x high; NOT stopwatch-validated.",
            "halves": halves,
            "preview": prev.name, "svg_path": str(prev_svg), "svg_wh_px": [W, H],
            "pattern_mm": [round(st["canvas_w"], 1), round(st["canvas_h"], 1)],
            "place": {"x": st["offset_x"], "y": st["offset_y"], "w": st["canvas_w"], "h": st["canvas_h"]},
            "z_up_abs": round(z_up_abs, 3),
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans), "layout_path": str(layout_path),
            "total_draw_mm": round(total_mm, 1), "asset_sha256": ASSET_FILE_SHA256}


def inkroute_bic(out_gcode, **kw):
    return inkroute_gcode(out_gcode, **kw)
