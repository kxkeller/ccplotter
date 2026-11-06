"""ccplotter/calibration_rowangle.py -- row-direction paper test sheet (Kai, 2026-10-26: "probs a paper thing").

QUESTION (Kai): the rows of the opt-in outline_rows solid fill run at one global angle and look mechanical. Do other
row looks read better ON PAPER? Offline (artifacts/rowfill-angle-2026-10-26/) coverage did not depend on direction;
the look is Kai's call and the pen's real anisotropy / dimples are paper-only effects.

WHAT IS ON THE SHEET. ONE window of the helicl (cleaned heli) full-page job, drawn FOUR times side by side, left to right:
  1 ZERO    row_angle_rule "zero" (explicit; the app default is a135 "/" since 2026-11-04) -- 0 deg of the image frame (the control)
  2 A45     every region at 45 deg
  3 ACROSS  each region's rows along its SHORT axis (near-round regions: 45 deg fallback, eigenvalue ratio < 1.3)
  4 SET     each region 0/45/90/135 deg by a deterministic hash of its centroid + pixel count (EXPERIMENTAL mixed look)
Each patch is a 34 x 40 mm WINDOW of the REAL tracev3 output (core_fill=outline_rows, BIC 0.12, stay-down over ink ON = the
production default, resolution pinned) of the full-page job, clipped to the window (not re-traced), turned 180 deg so the
helicopter reads upright. All four arms share one mask and one placement; only the row angle differs.

PRECOMPUTED ASSET (D1023): shipped as the DATA file assets/calibration/rowangle_asset.json.xz (PyInstaller datas entry in
desktop/ccplotter.spec), NOT embedded in this module. Its sha256 and its payload sha256 are constants below and CHECKED at
load; a mismatch refuses to generate. Built by artifacts/rowfill-paper-sheet-2026-10-26/scripts/build_asset.py.

DISCLOSED DIFFERENCES FROM A FULL-PAGE PLOT: only the window is drawn; strokes crossing a window edge are cut there and each
cut piece is its own pen-down (border pieces, counted separately in the ledger); pieces < 0.05 mm are dropped; microlift is
applied as production does; linesimplify/linesort are NOT re-run (strokes are already routed).

PEN IS PINNED to bic_cristal_blue (fill_mm 0.12; z_touch READ from the profile at generation time and recorded). Label the
sheet "nominal 0.12, laid unknown". Emission order: fiducials, glyph, title, rulers, prime first; then the four patches in a
recorded interleaved order (not left to right) so pen drift does not align with one arm. Time figures are
labelled and NOT stopwatch-validated.
"""
from __future__ import annotations

import hashlib
import json
import lzma
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

ASSET_REL = Path("assets") / "calibration" / "rowangle_asset.json.xz"
ASSET_FILE_SHA256 = "d71705bfa8227beda0e816206f489566403c7c96e19c1fcd6ac8dac13068c0f3"      # written by artifacts/rowfill-paper-sheet-2026-10-26/scripts/build_asset.py
ASSET_PAYLOAD_SHA256 = "5b24960eaee4bb812b9d16505644b58ca7f3c02fa9e714d1c605f206230d86b7"    # written by build_asset.py

ARMS = ("zero", "a45", "across", "set")
ARM_LABEL = {"zero": "1 ZERO", "a45": "2 A45", "across": "3 ACROSS", "set": "4 SET"}
EMISSION_ORDER = ("across", "zero", "set", "a45")          # recorded; NOT left to right

# ---- geometry of the sheet (237 x 161 mm; tip X = 10 + x, tip Y = 169 - y) ----
COL_X0, COL_PITCH = 14.0, 55.0
WIN_Y0 = 44.0
LABEL_BASE_DY = 3.0                    # label baseline this far above the window top (cap height 2.0)
MARK_OFF_MM, MARK_LEN_MM = 0.4, 1.0
CLEAR_MM = 0.2
TITLE_TEXT = "ROWANGLE BIC nominal 0.12"
TITLE_CX, TITLE_BASE_Y = 78.0, 6.5
PRIME_CX, PRIME_CY = 216.0, 4.5
SHORT_PIECE_MM = 0.5


class AssetError(RuntimeError):
    pass


def asset_path() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / ASSET_REL.name
    return Path(__file__).resolve().parent.parent / ASSET_REL


_ASSET_CACHE = {}


def load_asset(path: Path | None = None, file_sha: str | None = None, payload_sha: str | None = None):
    """Verified (meta, cells). cells = {arm: [(pts[(x, y) mm rel. to the window origin], border_flag), ...]}.
    Refuses to return anything if either sha does not match."""
    key = (str(path) if path else None, file_sha, payload_sha)
    if key in _ASSET_CACHE:
        return _ASSET_CACHE[key]
    p = Path(path) if path else asset_path()
    if not p.is_file():
        raise AssetError(f"rowangle asset missing: {p}")
    xz = p.read_bytes()
    if hashlib.sha256(xz).hexdigest() != (file_sha or ASSET_FILE_SHA256):
        raise AssetError("rowangle asset file sha mismatch: refusing to generate. Rebuild with build_asset.py; do not plot.")
    try:
        raw = lzma.decompress(xz)
    except lzma.LZMAError as e:
        raise AssetError(f"rowangle asset not decompressible: {e}")
    if hashlib.sha256(raw).hexdigest() != (payload_sha or ASSET_PAYLOAD_SHA256):
        raise AssetError("rowangle asset payload sha mismatch: refusing to generate.")
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
            raise AssetError(f"rowangle asset: arm {arm} counts do not match meta")
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


def build_layout(meta=None, cells=None):
    if meta is None:
        meta, cells = load_asset()
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

    for emis, arm in enumerate(EMISSION_ORDER):
        c = ARMS.index(arm)
        gid = f"patch|{arm}"
        ox, oy = cell_origin(c)
        x1, y1 = ox + w_mm, oy + h_mm
        lab_text = ARM_LABEL[arm]
        lab = label_strokes(lab_text, ox + w_mm / 2.0, oy - LABEL_BASE_DY)
        lx = [p[0] for pl in lab for p in pl]
        if min(lx) < ox - 0.5 or max(lx) > x1 + 0.5:
            raise RuntimeError(f"rowangle: label {lab_text!r} is wider than the {w_mm} mm window")
        marks = _corner_marks(ox, oy, x1, y1)
        rng_lab = add(lab, "label", gid + "|label")
        rng_mark = add(marks, "mark", gid + "|marks")
        pieces = cells[arm]
        data_pls = [[(ox + px, oy + py) for px, py in pts] for pts, _b in pieces]
        data_pls = [[(min(max(x, ox), x1), min(max(y, oy), y1)) for x, y in pl] for pl in data_pls]
        rng_data = add(data_pls, "data", gid)
        kept_flags = [bool(b) for (pts, b), pl in zip(pieces, data_pls)
                      if len(pl) >= 2 and _poly_length_mm(pl) >= MIN_STROKE_MM - 1e-12]
        if len(kept_flags) != rng_data[1] - rng_data[0]:
            raise RuntimeError("rowangle: border-flag / element misalignment")
        geoms[gid] = box(ox, oy, x1, y1)
        fixed[gid + "|label"] = lab
        for mi, m in enumerate(marks):
            fixed[f"{gid}|mark{mi}"] = [m]
        records.append({"id": gid, "arm": arm, "col": c, "label": lab_text,
                        "window_mm": [round(ox, 3), round(oy, 3), round(x1, 3), round(y1, 3)], "emission_index": emis,
                        "elem_range_label": rng_lab, "elem_range_marks": rng_mark, "elem_range_data": rng_data,
                        "border_flags": kept_flags})
    return elements, records, fixed, geoms, {"n_elements": len(elements), "fiducials": fid_records}


def clearance_check(geoms, fixed):
    return _check_clearance(geoms, fixed, min_mm=CLEAR_MM)


LEDGER_POPULATION = ("population = every pen-down stroke the sheet draws inside that patch's window (34 x 40 mm), counted from "
                     "the EMITTED gcode after clipping and after dropping pieces < 0.05 mm; 'border' pieces touch the window "
                     "edge (clip artifacts of the cut), 'interior' pieces do not.")


def ledger(records, strokes):
    per = []
    for rec in records:
        a, b = rec["elem_range_data"]
        sub = strokes[a:b]
        lens = [_poly_length_mm(s["pts"]) for s in sub]
        flags = rec["border_flags"]
        if len(lens) != len(flags):
            raise RuntimeError("rowangle: ledger misaligned")
        per.append({"id": rec["id"], "arm": rec["arm"], "pen_downs": len(sub), "draw_mm": round(sum(lens), 2),
                    "vertices": sum(len(s["pts"]) for s in sub), "border_pieces": int(sum(flags)),
                    "lt_0p5mm": sum(1 for x in lens if x < SHORT_PIECE_MM),
                    "median_piece_mm": round(statistics.median(lens), 3) if lens else 0.0})
    return {"population": LEDGER_POPULATION, "per_patch": per}


def rowangle_gcode(out_gcode: Path, pen_mm=None, z_up=3.5, travel_speed=8000, plunge_speed=500,
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
            f"rowangle: refusing to generate -- this sheet is PINNED to {PEN_ID!r} ({PEN_LABEL!r}); the requested/active "
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
        raise RuntimeError(f"fill_mm for {PEN_ID} = {pitch_mm}; the traced strokes in this sheet were made at the locked "
                           f"{PITCH_MM} mm BIC pitch -- refusing (the fill slider is set; Reset it for this sheet).")

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
            f"rowangle: tip window X[{min(tx):.1f},{max(tx):.1f}] Y[{min(ty):.1f},{max(ty):.1f}] outside the paper window "
            f"X[{TIP_X_MIN},{TIP_X_MAX}] Y[{TIP_Y_MIN},{TIP_Y_MAX}] (B5 landscape, holes at the FRONT; frame moved per D1487).")
    park_x, park_y = reach["safe_park_point"]
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))
    z_down_abs = z_touch + z_down
    z_up_abs = z_touch + float(z_up)

    placed_all = [[((x + pad) * ppmm, (y + pad) * ppmm) for x, y in e["pts"]] for e in elements]
    blk_svg = WORK_DIR / f".{out_gcode.stem}_blk.svg"
    blk_gc = WORK_DIR / f".{out_gcode.stem}_blk.gcode"
    _write_svg_hp(blk_svg, W, H, float(pen_mm) * ppmm, placed_all)
    vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({
        "z_up": float(z_up), "z_up_abs": z_up_abs, "z_down": z_down, "z_down_abs": z_down_abs,
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
        "ROW ANGLE SHEET: one heli window x 4 row looks (1 ZERO, 2 A45, 3 ACROSS, 4 SET), real full-page outline_rows trace clipped",
        "to 34 x 40 mm windows at page scale 1; not re-traced. Question: which row look reads best on paper?",
        f"asset file sha256={ASSET_FILE_SHA256} payload={ASSET_PAYLOAD_SHA256}",
        f"pen_id={PEN_ID} z_touch_abs={z_touch:.3f} ({z_touch_source}) z_down_abs={z_down_abs:.3f}",
        f"pitch_mm={pitch_mm} (profile fill_mm) draw F{FAST} strokes={len(elements)} microlift={'on' if microlift else 'off'}",
        f"RECORD the resolved z_touch before plotting (this sheet: {z_touch:.3f}); the active pen profile drifts (DIRECTION S10).",
        "Label this sheet: nominal 0.12, laid unknown. EXPERIMENTAL row looks; ZERO is today's default.",
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
            out_gcode, z_up_abs, z_short_abs=(z_short_abs if do_micro else None), short_mm=micro_hop_mm,
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
        vpype_cli.execute(f'read {shlex.quote(str(prev_svg))} color {PREVIEW_COLOR} write {shlex.quote(str(tmp_prev))}')

    layout = {
        "sheet": "rowangle", "pen_id": PEN_ID, "pitch_mm": pitch_mm,
        "question": "Which row look (0 deg today, constant 45, ACROSS the short side, SET mixed) reads best on paper?",
        "label_rule": "nominal 0.12, laid unknown",
        "asset": {"file_sha256": ASSET_FILE_SHA256, "payload_sha256": ASSET_PAYLOAD_SHA256, "quant_um": meta["quant_um"],
                  "source": meta["source"], "window_page_mm": meta["window_page_mm"], "rotated_180": meta["rotated_180"]},
        "arms": list(ARMS), "arm_labels": ARM_LABEL, "emission_order": list(EMISSION_ORDER),
        "win_w_mm": meta["win_w_mm"], "win_h_mm": meta["win_h_mm"],
        "grid": {"col_x0": COL_X0, "col_pitch": COL_PITCH, "win_y0": WIN_Y0, "label_base_dy": LABEL_BASE_DY},
        "sheet_wh_mm": [SHEET_W, SHEET_H], "pad_mm": pad,
        "strokes_bbox_mm": [round(min(allx), 3), round(min(ally), 3), round(max(allx), 3), round(max(ally), 3)],
        "mapping": {"offset_x": offset_x, "offset_y": offset_y, "flip_axis_mm": flip_axis, "comp_off_x": comp_off_x,
                    "comp_off_y": comp_off_y, "pen_off_x": pen_off_x, "pen_off_y": pen_off_y,
                    "cmd_x": "pad + x + comp_off_x", "cmd_y": "flip + comp_off_y - (pad + y)", "tip_x": "cmd_x + pen_off_x",
                    "tip_y": "cmd_y + pen_off_y"},
        "tip_window_mm": {"x": [min(tx), max(tx)], "y": [min(ty), max(ty)]},
        "paper_window_mm": {"x": [TIP_X_MIN, TIP_X_MAX], "y": [TIP_Y_MIN, TIP_Y_MAX]},
        "z_touch_abs": z_touch, "z_touch_source": z_touch_source, "z_down_abs": z_down_abs, "z_up_abs": z_up_abs,
        "z_short_abs": z_short_abs, "microlift": microlift_stats,
        "cells": [{k: v for k, v in r.items() if k != "border_flags"} for r in records],
        "fiducials": lmeta["fiducials"],
        "ledger": led, "stroke_check": stroke_check, "selftest_caught": selftest_caught, "clearance_check": clr,
        "time_model_a_s": round(t_a, 1), "time_model_b_s": round(t_b, 1),
    }
    layout_path = WORK_DIR / f"{out_gcode.stem}_layout.json"
    layout_path.write_text(json.dumps(layout, indent=1), encoding="utf-8")

    return {"kind": "rowangle", "pen_mm": pen_mm, "pen_id": PEN_ID, "pen_label": PEN_LABEL,
            "footprint_mm": [round(canvas_w, 1), round(canvas_h, 1)], "offset_mm": [offset_x, offset_y],
            "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source, "z_down": z_down,
            "z_down_abs": round(z_down_abs, 3), "draw_speed": draw_speed, "reach_id": reach0["id"],
            "reach_bbox": reach0["bbox"], "n_pendown_expected_total": len(elements),
            "n_pendown_actual_total": stroke_check["n_strokes"],
            "pendown_count_matches": stroke_check["n_strokes"] == len(elements),
            "deepest_z_down_abs": round(z_down_abs, 3), "safety_selfcheck": safety, "stroke_check": stroke_check,
            "clearance_check": clr, "microlift": microlift_stats, "tip_window_mm": layout["tip_window_mm"],
            "default_placement": default_placement, "plot_time_estimate_s": round(t_b, 1),
            "plot_time_estimate_min": round(t_b / 60.0, 2),
            "plot_time_model_label": "model (b): per-move length / modal feed over the emitted gcode, Z included, microlift "
                                     "applied, no acceleration/dwell; a floor, NOT stopwatch-validated.",
            "plot_time_model_a_min": round(t_a / 60.0, 2),
            "plot_time_model_a_label": "model (a): draw length / feed + n_pen_downs x (plunge + lift) at nominal Z travel, "
                                       "Z-blind to microlift; ~2x high; NOT stopwatch-validated.",
            "preview": prev.name, "svg_path": str(prev_svg), "svg_wh_px": [W, H],
            "pattern_mm": [round(canvas_w, 1), round(canvas_h, 1)],
            "place": {"x": offset_x, "y": offset_y, "w": canvas_w, "h": canvas_h}, "z_up_abs": round(z_up_abs, 3),
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans), "layout_path": str(layout_path),
            "total_draw_mm": round(total_mm, 1), "asset_sha256": ASSET_FILE_SHA256}


def rowangle_bic(out_gcode, **kw):
    return rowangle_gcode(out_gcode, **kw)
