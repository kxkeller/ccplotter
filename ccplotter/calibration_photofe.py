"""ccplotter/calibration_photofe.py -- photo front-end physical validation sheet.

What it measures: whether a finer photo front-end setting (S1.2 and finer) gives source-edge
recall that is VISIBLY USEFUL ON PAPER, and where the finer settings cross from useful
structure into texture / interpolation noise. Recall figures alone are not evidence; the
judgement is made by eye on the plotted sheet.

How to read it: 8 rows (photo regions) x 6 columns (arms). Compare the six cells of a row
side by side; each row is the same photo region traced with a different front-end setting.
Look for the column where extra detail stops looking like real structure (lettering, edges,
mechanics) and starts looking like grain or interpolation noise.

WHAT IS ON THE SHEET. Each cell is a 35 x 13 mm WINDOW of the REAL tracev3 output of a
FULL-PAGE plot of that photo (257 x 176 page, margin 10, scale 1, rot 0), clipped to the
window. The crop is NOT re-traced: the strokes are exactly the strokes the full-page plot
would draw in that window, with the same fill, hole floor, routing and stroke order. The
sheet is at page scale 1, so a window is the same physical size as on a full-page plot. mm
per SOURCE pixel differs by photo (0.048 .. 0.208 mm/px) and is recorded per row in the
layout json.

ARMS (columns): A = default bw_xdog_raster path (sigma ~0.2 mm); S1.4 / S1.2 / S1.0 = front
end at side 1024 with sigma_src 1.4 / 1.2 / 1.0 px; C1024F = side 1024, sigma 0.8 formula;
NATcap = side min(1024, source short side), sigma 1.2 px (on photos whose short side >= 1024
it is byte-identical to S1.2 and serves as a repeatability control; only heli / a350 /
giraffe (short side 627 / 600 / 780) differ, where side 1024 would be pure interpolation).

PRECOMPUTED ASSET: the clipped strokes are embedded in ccplotter/calibration_photofe_data.py,
with sha256 values recorded there and CHECKED at load. Re-running the front end at
generation time is not possible: it needs the source photos, the heavy non-deterministic
pipeline and a trace context the window does not have, and vendoring crops would change the
trace. The asset is an embedded .py module (not a data file) because the PyInstaller build
does not bundle ccplotter's non-py files. ~1 MB.

DIFFERENCES FROM A FULL-PAGE PLOT:
  * only the 48 windows are drawn (what is outside each window is absent); strokes crossing a
    window edge are CUT there and each cut piece is its own pen-down (flagged border in the
    layout; these are clip artifacts, not the arm's own fragmentation -- the ledger reports
    them separately);
  * pieces shorter than 0.05 mm after clipping are dropped (MIN_STROKE_MM);
  * microlift is applied as production does; linesimplify/linesort are NOT re-run (the strokes
    are already routed).

HARD-CODED MACHINE AND PEN (limitation). The sheet is tied to the author's measured machine:
EXPECT_REACH_ID ("plotter_2003_v32"), EXPECT_BBOX and the Z / speed constants are imported
from calibration_pocket and the generator refuses any other reach profile. The pen is PINNED
to bic_cristal_blue (fill_mm 0.12; z_touch is READ from the profile). The sheet is labelled
"nominal 0.12, laid unknown". Emission order: fiducials, glyph, title, rulers, prime FIRST;
then row by row, with the arm order ROTATED by row so a drifting pen does not align with one
arm. Time figures are model estimates, not stopwatch-validated.
"""
from __future__ import annotations

import base64
import hashlib
import json
import lzma
import shlex
import statistics
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
from ccplotter import calibration_photofe_data as _data
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
LAID_PREVIEW_MM = (0.12, 0.16, 0.20, 0.24)

# ---- geometry of the sheet (237 x 161 mm; tip X = 10 + x, tip Y = 169 - y) ----
N_COLS = 6
COL_X0, COL_PITCH = 9.0, 37.0          # windows are 35 mm wide, 2.0 mm apart
ROW_Y0, ROW_PITCH = 10.5, 17.0
LABEL_BASE_DY = 2.0                    # label baseline below the row top (cap height 2.0)
WIN_DY = 3.1                           # window top below the row top
MARK_OFF_MM, MARK_LEN_MM = 0.4, 1.0    # corner L-marks outside each window
CLEAR_MM = 0.2                         # min gap between any two distinct groups
TITLE_TEXT = "PHOTOFE BIC nominal 0.12"
TITLE_CX, TITLE_BASE_Y = 78.0, 6.5
PRIME_CX, PRIME_CY = 216.0, 4.5
BORDER_EPS_MM = 0.0                    # (border flag comes from the asset)
SHORT_PIECE_MM = 0.5


# ---------------------------------------------------------------------------
# Asset: load, verify (sha), decode. Pure functions; the cache is module-level.
# ---------------------------------------------------------------------------
class AssetError(RuntimeError):
    pass


def asset_shas(b64_text=None, xz_sha=None, raw_sha=None):
    """(xz_sha_ok, payload_sha_ok, xz_bytes, payload_bytes). Parameters let the tests hand in a corrupted copy."""
    b64_text = _data.ASSET_XZ_B64 if b64_text is None else b64_text
    xz = base64.b64decode("".join(b64_text.split()))
    xz_ok = hashlib.sha256(xz).hexdigest() == (_data.ASSET_XZ_SHA256 if xz_sha is None else xz_sha)
    try:
        raw = lzma.decompress(xz)
    except lzma.LZMAError:
        return xz_ok, False, xz, b""
    raw_ok = hashlib.sha256(raw).hexdigest() == (_data.PAYLOAD_SHA256 if raw_sha is None else raw_sha)
    return xz_ok, raw_ok, xz, raw


def _read_varint(buf, i):
    z = 0
    shift = 0
    while True:
        b = buf[i]
        i += 1
        z |= (b & 0x7F) << shift
        if b < 0x80:
            return z, i
        shift += 7


def decode_payload(raw: bytes, meta: dict):
    """-> {(row_index, arm): [(pts[(x_mm, y_mm) rel. to window origin], border_flag), ...]}.
    Verifies per-cell piece/vertex counts against the meta and that the payload is fully consumed."""
    if raw[:4] != b"PFE1":
        raise AssetError("photofe asset: bad magic")
    q = meta["quant_um"] / 1000.0
    i, out = 4, {}
    for ri, row in enumerate(meta["rows"]):
        for arm in meta["arms"]:
            n_pieces, i = _read_varint(raw, i)
            cell, nv_tot, nb = [], 0, 0
            for _ in range(n_pieces):
                h, i = _read_varint(raw, i)
                nv, border = h >> 1, h & 1
                x = y = 0
                pts = []
                for k in range(nv):
                    zx, i = _read_varint(raw, i)
                    zy, i = _read_varint(raw, i)
                    x += (zx >> 1) ^ -(zx & 1)
                    y += (zy >> 1) ^ -(zy & 1)
                    pts.append((x * q, y * q))
                cell.append((pts, bool(border)))
                nv_tot += nv
                nb += border
            c = row["cells"][arm]
            if (len(cell), nv_tot, nb) != (c["n_pieces"], c["n_verts"], c["n_border"]):
                raise AssetError(f"photofe asset: cell {row['id']}/{arm} counts "
                                 f"{(len(cell), nv_tot, nb)} != meta "
                                 f"{(c['n_pieces'], c['n_verts'], c['n_border'])}")
            out[(ri, arm)] = cell
    if i != len(raw):
        raise AssetError(f"photofe asset: {len(raw) - i} trailing bytes")
    return out


_ASSET_CACHE = {}


def load_asset():
    """Verified (meta, cells). Refuses to return anything if either sha does not match."""
    if "v" not in _ASSET_CACHE:
        xz_ok, raw_ok, _xz, raw = asset_shas()
        if not (xz_ok and raw_ok):
            raise AssetError(f"photofe asset sha mismatch (xz ok={xz_ok}, payload ok={raw_ok}): refusing to "
                             f"generate. Regenerate with build_asset.py; do not plot.")
        meta = json.loads(_data.META_JSON)
        _ASSET_CACHE["v"] = (meta, decode_payload(raw, meta))
    return _ASSET_CACHE["v"]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def cell_origin(r: int, c: int):
    """Sheet-frame (x, y) of the window's top-left corner for row r, column c (column == arm index)."""
    return COL_X0 + c * COL_PITCH, ROW_Y0 + r * ROW_PITCH + WIN_DY


def _corner_marks(x0, y0, x1, y1):
    o, L = MARK_OFF_MM, MARK_LEN_MM
    return [[(x0 - o, y0 - o + L), (x0 - o, y0 - o), (x0 - o + L, y0 - o)],
            [(x1 + o - L, y0 - o), (x1 + o, y0 - o), (x1 + o, y0 - o + L)],
            [(x0 - o, y1 + o - L), (x0 - o, y1 + o), (x0 - o + L, y1 + o)],
            [(x1 + o - L, y1 + o), (x1 + o, y1 + o), (x1 + o, y1 + o - L)]]


def _cell_label(arm, row_id):
    return f"{arm} {row_id}"


def _emission_arm_order(r, n):
    """Arm order within row r: rotated by r (a latin-square-ish rotation) so pen drift does not align with one arm."""
    return [(j + r) % n for j in range(n)]


def build_layout(meta=None, cells=None):
    if meta is None:
        meta, cells = load_asset()
    arms = meta["arms"]
    w_mm, h_mm = meta["win_w_mm"], meta["win_h_mm"]
    elements, records, geoms, fixed = [], [], {}, {}

    def add(pls, kind, gid):
        a = len(elements)
        for pl in pls:
            if len(pl) >= 2 and _poly_length_mm(pl) >= MIN_STROKE_MM - 1e-12:
                elements.append({"pts": [(float(x), float(y)) for x, y in pl],
                                 "speed": FAST, "kind": kind, "gid": gid})
        return [a, len(elements)]

    # fiducials FIRST (freshest pen state), then glyph, title, rulers, prime -- same order as calibration_fragment
    fid_pos = [(2.0, 2.0), (SHEET_W / 2 - 2.0, 2.0), (SHEET_W - 6.0, 2.0),
               (2.0, 148.0), (SHEET_W / 2 - 2.0, 148.0), (SHEET_W - 6.0, 148.0)]
    fid_records = []
    fp = min(PITCH_MM, FID_PITCH_MAX_FRAC_OF_PEN * PEN_MM)
    for i, (fx, fy) in enumerate(fid_pos):
        pls = _square_rings(FID_SIDE_MM, fp, fx + FID_SIDE_MM / 2, fy + FID_SIDE_MM / 2)
        fixed[f"fid{i}"] = [[(fx, fy), (fx + FID_SIDE_MM, fy + FID_SIDE_MM)]]
        add(pls, "fiducial", f"fid{i}")
        fid_records.append({"id": f"fid{i}", "origin_mm": [fx, fy], "side_mm": FID_SIDE_MM,
                            "center_mm": [fx + FID_SIDE_MM / 2, fy + FID_SIDE_MM / 2],
                            "pitch_mm": fp, "n_rings": len(pls)})
    pls = _glyph(12.0, 1.0) + [[(24.0 + 3.0 * i, 1.0), (24.0 + 3.0 * i, 8.0)] for i in range(3)]
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

    emis = 0
    for r, row in enumerate(meta["rows"]):
        for c in _emission_arm_order(r, len(arms)):
            arm = arms[c]
            gid = f"{row['id']}|{arm}"
            ox, oy = cell_origin(r, c)
            x1, y1 = ox + w_mm, oy + h_mm
            lab_text = _cell_label(arm, row["id"])
            lab = label_strokes(lab_text, ox + w_mm / 2.0, oy - (WIN_DY - LABEL_BASE_DY))
            lx = [p[0] for pl in lab for p in pl]
            if min(lx) < ox - 0.5 or max(lx) > x1 + 0.5:
                raise RuntimeError(f"photofe: label {lab_text!r} is {max(lx) - min(lx):.1f} mm wide, wider than "
                                   f"the {w_mm} mm cell")
            marks = _corner_marks(ox, oy, x1, y1)
            rng_lab = add(lab, "label", gid + "|label")
            rng_mark = add(marks, "mark", gid + "|marks")
            pieces = cells[(r, arm)]
            data_pls = [[(ox + px, oy + py) for px, py in pts] for pts, _b in pieces]
            # a clipped point may sit a rounding hair outside the window; clamp to it (never moves a point > 0.001 mm)
            data_pls = [[(min(max(x, ox), x1), min(max(y, oy), y1)) for x, y in pl] for pl in data_pls]
            rng_data = add(data_pls, "data", gid)
            # border flags in ELEMENT order (add() may drop a degenerate piece: keep the alignment honest)
            kept_flags, k = [], 0
            for (pts, b), pl in zip(pieces, data_pls):
                if len(pl) >= 2 and _poly_length_mm(pl) >= MIN_STROKE_MM - 1e-12:
                    kept_flags.append(bool(b))
            if len(kept_flags) != rng_data[1] - rng_data[0]:
                raise RuntimeError("photofe: border-flag / element misalignment")
            geoms[gid] = box(ox, oy, x1, y1)
            fixed[gid + "|label"] = lab
            for mi, m in enumerate(marks):
                fixed[f"{gid}|mark{mi}"] = [m]
            records.append({"id": gid, "row_id": row["id"], "row": r, "col": c, "arm": arm, "photo": row["photo"],
                            "label": lab_text, "window_mm": [round(ox, 3), round(oy, 3), round(x1, 3), round(y1, 3)],
                            "emission_index": emis, "elem_range_label": rng_lab, "elem_range_marks": rng_mark,
                            "elem_range_data": rng_data, "border_flags": kept_flags,
                            "job": row["cells"][arm]["job"],
                            "routed_svg_sha256": row["cells"][arm]["routed_svg_sha256"]})
            emis += 1

    meta_out = {"n_elements": len(elements), "fiducials": fid_records}
    return elements, records, fixed, geoms, meta_out


def clearance_check(geoms, fixed):
    """Every distinct group (windows, labels, corner marks, fiducials, glyph, title, rulers, prime) at least
    CLEAR_MM from every other. Marks of one cell are separate groups at MARK_OFF_MM from their own window."""
    return _check_clearance(geoms, fixed, min_mm=CLEAR_MM)


# ---------------------------------------------------------------------------
# Ledger (per arm): names the POPULATION measured, plus a per-cell fragmentation ledger
# ---------------------------------------------------------------------------
LEDGER_POPULATION = ("population = every pen-down stroke the sheet draws inside that arm's 8 windows (8 regions, "
                     "35 x 13 mm each), counted from the EMITTED gcode, after clipping and after dropping pieces "
                     "< 0.05 mm. 'border' pieces touch a window edge and are clip artifacts of the cut, not the "
                     "arm's own fragmentation; 'interior' pieces do not touch an edge.")


def ledger(records, strokes):
    """Per-arm and per-cell fragmentation ledger from the parsed gcode strokes (aligned 1:1 with elements)."""
    by_arm, by_cell = {}, []
    for rec in records:
        a, b = rec["elem_range_data"]
        sub = strokes[a:b]
        lens = [_poly_length_mm(s["pts"]) for s in sub]
        flags = rec["border_flags"]
        if len(lens) != len(flags):
            raise RuntimeError("photofe: ledger misaligned")
        row = {"id": rec["id"], "arm": rec["arm"], "row_id": rec["row_id"], "pen_downs": len(sub),
               "draw_mm": round(sum(lens), 2), "vertices": sum(len(s["pts"]) for s in sub),
               "lt_0p5mm": sum(1 for x in lens if x < SHORT_PIECE_MM),
               "lt_0p5mm_border": sum(1 for x, f in zip(lens, flags) if x < SHORT_PIECE_MM and f),
               "lt_0p5mm_interior": sum(1 for x, f in zip(lens, flags) if x < SHORT_PIECE_MM and not f),
               "border_pieces": int(sum(flags)),
               "median_piece_mm": round(statistics.median(lens), 3) if lens else 0.0}
        by_cell.append(row)
        g = by_arm.setdefault(rec["arm"], {"arm": rec["arm"], "cells": 0, "pen_downs": 0, "draw_mm": 0.0,
                                           "vertices": 0, "lt_0p5mm": 0, "lt_0p5mm_border": 0,
                                           "lt_0p5mm_interior": 0, "border_pieces": 0, "_lens": []})
        g["cells"] += 1
        for k in ("pen_downs", "draw_mm", "vertices", "lt_0p5mm", "lt_0p5mm_border", "lt_0p5mm_interior",
                  "border_pieces"):
            g[k] += row[k]
        g["_lens"] += lens
    arms = []
    for a in by_arm.values():
        lens = a.pop("_lens")
        a["draw_mm"] = round(a["draw_mm"], 1)
        a["median_piece_mm"] = round(statistics.median(lens), 3)
        a["mean_piece_mm"] = round(sum(lens) / len(lens), 3)
        a["pen_downs_per_m"] = round(a["pen_downs"] / (a["draw_mm"] / 1000.0), 1)
        arms.append(a)
    return {"population": LEDGER_POPULATION, "per_arm": arms, "per_cell": by_cell}


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------
def photofe_gcode(out_gcode: Path, pen_mm=None, z_up=3.5, travel_speed=8000, plunge_speed=500,
                  lift_speed=2000, offset_x=None, offset_y=None, end_policy="release",
                  quiet_fans=True, profile=None, microlift=True):
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
            f"photofe: refusing to generate -- this sheet is PINNED to {PEN_ID!r} "
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
        raise RuntimeError(f"fill_mm for {PEN_ID} = {pitch_mm}; the traced strokes in this sheet were made at the "
                           f"locked {PITCH_MM} mm BIC pitch -- refusing "
                           f"(the fill slider is set; Reset it for this sheet).")

    reach = pen_reach_profile(reach0, pen)
    if pen_mm is None:
        pen_mm = PEN_MM
    pen_off_x, pen_off_y = resolve_pen_offset(cal, pen)

    meta, cells = load_asset()                      # sha-verified or AssetError
    elements, records, fixed_groups, geoms, lmeta = build_layout(meta, cells)
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
            f"photofe: tip window X[{min(tx):.1f},{max(tx):.1f}] Y[{min(ty):.1f},{max(ty):.1f}] "
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

    header = [
        "PHOTO FRONT-END SHEET: 8 regions x 6 arms (A, S1.4, S1.2, S1.0, C1024F, NATcap), real full-page trace clipped",
        "to 35 x 13 mm windows at page scale 1; the crops are NOT re-traced. Question: is S1.2's extra recall useful?",
        f"asset sha256 xz={_data.ASSET_XZ_SHA256} payload={_data.PAYLOAD_SHA256}",
        f"pen_id={PEN_ID} z_touch_abs={z_touch:.3f} ({z_touch_source}) z_down_abs={z_down_abs:.3f}",
        f"pitch_mm={pitch_mm} (profile fill_mm) draw F{FAST} strokes={len(elements)} "
        f"microlift={'on' if microlift else 'off'}",
        "RECORD z_touch before plotting; BIC=18.95 / uRight=11.45 differ by 7.50 mm (DIRECTION S10).",
        "Label this sheet: nominal 0.12, laid unknown.",
    ]
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
    led = ledger(records, strokes)

    total_mm = sum(_poly_length_mm(s["pts"]) for s in strokes)
    t_a = time_model_a(elements, plunge_speed, lift_speed, z_up)
    t_b = time_model_b(out_gcode)

    prev_svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_svg_hp(prev_svg, W, H, float(pen_mm) * ppmm, placed_all)
    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(prev_svg))} color {PREVIEW_COLOR} '
                          f'write {shlex.quote(str(tmp_prev))}')

    rows_out = [{k: r[k] for k in ("id", "photo", "name", "class", "why", "centre_mm", "win_mm", "src_px",
                                   "short_px", "mm_per_src_px", "nat_side_px", "sigma_mm")}
                | {"cells": {a: {k: v for k, v in c.items()} for a, c in r["cells"].items()}}
                for r in meta["rows"]]
    layout = {
        "sheet": "photofe", "pen_id": PEN_ID, "pitch_mm": pitch_mm,
        "question": "Is S1.2's extra source-edge recall visibly useful on paper, and where do finer settings "
                    "cross from structure into texture/interpolation noise?",
        "label_rule": "nominal 0.12, laid unknown",
        "asset": {"xz_sha256": _data.ASSET_XZ_SHA256, "payload_sha256": _data.PAYLOAD_SHA256,
                  "quant_um": meta["quant_um"], "source": meta["source"]},
        "arms": meta["arms"], "rows": rows_out, "n_rows": len(rows_out), "n_cols": N_COLS,
        "win_w_mm": meta["win_w_mm"], "win_h_mm": meta["win_h_mm"],
        "grid": {"col_x0": COL_X0, "col_pitch": COL_PITCH, "row_y0": ROW_Y0, "row_pitch": ROW_PITCH,
                 "win_dy": WIN_DY, "label_base_dy": LABEL_BASE_DY},
        "emission_order_note": "fiducials, glyph, title, rulers, prime first; then row by row (top to bottom), "
                               "arm order rotated by the row index; emission_index is the cell order.",
        "laid_preview_mm": list(LAID_PREVIEW_MM),
        "sheet_wh_mm": [SHEET_W, SHEET_H], "pad_mm": pad,
        "strokes_bbox_mm": [round(min(allx), 3), round(min(ally), 3), round(max(allx), 3), round(max(ally), 3)],
        "mapping": {"offset_x": offset_x, "offset_y": offset_y, "flip_axis_mm": flip_axis,
                    "comp_off_x": comp_off_x, "comp_off_y": comp_off_y,
                    "pen_off_x": pen_off_x, "pen_off_y": pen_off_y,
                    "cmd_x": "pad + x + comp_off_x", "cmd_y": "flip + comp_off_y - (pad + y)",
                    "tip_x": "cmd_x + pen_off_x", "tip_y": "cmd_y + pen_off_y"},
        "tip_window_mm": {"x": [min(tx), max(tx)], "y": [min(ty), max(ty)]},
        "paper_window_mm": {"x": [TIP_X_MIN, TIP_X_MAX], "y": [TIP_Y_MIN, TIP_Y_MAX]},
        "z_touch_abs": z_touch, "z_touch_source": z_touch_source, "z_down_abs": z_down_abs,
        "z_up_abs": z_up_abs, "z_short_abs": z_short_abs, "microlift": microlift_stats,
        "cells": [{k: v for k, v in r.items() if k != "border_flags"} for r in records],
        "fiducials": lmeta["fiducials"],
        "rulers": {"h": {"origin_mm": [GRID_X0, 158.0], "len_mm": FID0_LEN_MM},
                   "v": {"origin_mm": [234.0, 22.0], "len_mm": FID90_LEN_MM},
                   "tick_pitch_mm": _pk.FID_TICK_PITCH_MM, "tick_h_mm": _pk.FID_TICK_H_MM, "tick_dir": -1},
        "ledger": led,
        "stroke_check": stroke_check, "selftest_caught": selftest_caught, "clearance_check": clr,
        "time_model_a_s": round(t_a, 1), "time_model_b_s": round(t_b, 1),
    }
    layout_path = WORK_DIR / f"{out_gcode.stem}_layout.json"
    layout_path.write_text(json.dumps(layout, indent=1), encoding="utf-8")

    return {"kind": "photofe", "pen_mm": pen_mm, "pen_id": PEN_ID, "pen_label": PEN_LABEL,
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
            "clearance_check": clr, "microlift": microlift_stats,
            "tip_window_mm": layout["tip_window_mm"], "default_placement": default_placement,
            "plot_time_estimate_s": round(t_b, 1),
            "plot_time_estimate_min": round(t_b / 60.0, 2),
            "plot_time_model_label": "model (b): per-move length / modal feed over the emitted gcode, "
                                     "Z included, microlift applied, no acceleration/dwell; a floor, "
                                     "NOT stopwatch-validated.",
            "plot_time_model_a_min": round(t_a / 60.0, 2),
            "plot_time_model_a_label": "model (a): draw length / feed + n_pen_downs x (plunge + lift) at nominal "
                                       "Z travel, Z-blind to microlift; ~2x high; NOT stopwatch-validated.",
            "preview": prev.name, "svg_path": str(prev_svg), "svg_wh_px": [W, H],
            "pattern_mm": [round(canvas_w, 1), round(canvas_h, 1)],
            "place": {"x": offset_x, "y": offset_y, "w": canvas_w, "h": canvas_h},
            "z_up_abs": round(z_up_abs, 3),
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
            "layout_path": str(layout_path), "total_draw_mm": round(total_mm, 1),
            "asset_sha256": _data.ASSET_XZ_SHA256}


def photofe_bic(out_gcode, **kw):
    return photofe_gcode(out_gcode, **kw)
