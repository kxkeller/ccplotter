"""Tests for the photo front-end validation sheet (ccplotter/calibration_photofe.py).

Every guard is paired with a NEGATIVE control run in the same test (a tampered input the guard must reject):
a check that has never been observed to fail is not known to work.

Checked: emitted gcode pure ASCII / no G92 / valid fans / Z floor / reach box plotter_2003_v32 / real print
structure; every cell carries a printed label and the label strokes reach the gcode (pen-down count == design count);
the embedded stroke asset's sha256 (both the compressed blob and the decoded payload) and its decode self-consistency;
the cell geometry (windows, labels, marks, clearances); the layout reproduces the asset (no re-trace, window origin +
asset point == emitted design point); the ledger is derived from the emitted gcode and aligned with the asset counts;
NATcap == S1.2 where the source short side >= 1024 (and differs where it is not); pen pinned, z_touch READ from the
profile, fill_mm locked, determinism, registry entry.

Run: python tests\\test_calibration_photofe.py   (never with the repo root as CWD; the module chdirs to a private env)
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENV_SRC = REPO / "tests" / "fixtures" / "rot_fix_2026_09_01" / "_env"

if not ENV_SRC.is_dir():
    print(f"SKIP: missing runtime config fixture {ENV_SRC}")
    raise SystemExit(1)

_ENV = Path(tempfile.mkdtemp(prefix="ccplotter_pfe_env_"))
for _f in ("vpype.toml", "pen_profiles.json", "calibration.json"):
    shutil.copy2(ENV_SRC / _f, _ENV / _f)
os.chdir(_ENV)
os.environ["CCPLOTTER_HOME"] = str(_ENV)
_WORK = Path(tempfile.mkdtemp(prefix="ccplotter_pfe_work_"))
os.environ["CCPLOTTER_WORK_DIR"] = str(_WORK)
sys.path.insert(0, str(REPO))

from ccplotter import calibration_photofe as pf  # noqa: E402
from ccplotter import calibration_photofe_data as pfd  # noqa: E402
from ccplotter.calibration_registry import SHEETS  # noqa: E402

REACH_BBOX = (3.0, 42.5, 255.0, 254.5)       # plotter_2003_v32
BAD_FAN = re.compile(r"M106\s+P[01](\s|$)")
_cache = {}


# ----------------------------------------------------------------- helpers ---
def _gen(tag="base"):
    if tag not in _cache:
        out = _WORK / f"_pfe_{tag}.gcode"
        info = SHEETS["photofe_bic"].generator(out)
        _cache[tag] = (out, info)
    return _cache[tag]


def _check_gcode_text(raw: bytes, deepest_z: float):
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as e:
        raise AssertionError(f"non-ASCII byte at {e.start}") from None
    lines = text.split("\n")
    assert not [i for i, l in enumerate(lines) if l.startswith("G92")], "G92 present"
    assert not [i for i, l in enumerate(lines) if BAD_FAN.search(l)], "M106 P0/P1 present"
    zs, xs, ys, n = [], [], [], 0
    for l in lines:
        if not l.startswith(("G0", "G1")):
            continue
        n += 1
        if (m := re.search(r"Z(-?[\d.]+)", l)):
            zs.append(float(m.group(1)))
        if (m := re.search(r"X(-?[\d.]+)", l)):
            xs.append(float(m.group(1)))
        if (m := re.search(r"Y(-?[\d.]+)", l)):
            ys.append(float(m.group(1)))
    assert n > 0 and xs and ys and zs, "no print structure"
    assert min(zs) >= deepest_z - 1e-6, f"Z {min(zs)} below floor {deepest_z}"
    bx0, by0, bx1, by1 = REACH_BBOX
    assert bx0 <= min(xs) and max(xs) <= bx1 and by0 <= min(ys) and max(ys) <= by1, \
        f"XY [{min(xs)},{max(xs)}]x[{min(ys)},{max(ys)}] outside reach {REACH_BBOX}"
    return {"n_moves": n, "x": (min(xs), max(xs)), "y": (min(ys), max(ys)), "zmin": min(zs)}


def _raises(fn, *a, exc=(AssertionError,), **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    return False


# ------------------------------------------------------------------- tests ---
def test_gcode_safety_and_reach():
    out, info = _gen()
    raw = out.read_bytes()
    r = _check_gcode_text(raw, info["deepest_z_down_abs"])
    assert r["n_moves"] > 50000, f"suspiciously few moves for ~17.7k strokes: {r['n_moves']}"
    z = info["deepest_z_down_abs"]
    assert _raises(_check_gcode_text, raw + "; — em dash\n".encode("utf-8"), z), "non-ASCII not caught"
    assert _raises(_check_gcode_text, raw + b"G92 X0 Y0\n", z), "G92 not caught"
    assert _raises(_check_gcode_text, raw + b"M106 P1 S255\n", z), "M106 P1 not caught"
    assert _raises(_check_gcode_text, raw + b"M106 P0 S255\n", z), "M106 P0 not caught"
    assert _raises(_check_gcode_text, raw + f"G1 Z{z - 0.5:.2f} F500\n".encode(), z), "low Z not caught"
    assert _raises(_check_gcode_text, raw + b"G1 X300.0 Y100.0 F5000\n", z), "out-of-reach X not caught"
    assert _raises(_check_gcode_text, b"; comments only\n", z), "M-code-only file not caught"
    # fan codes actually present in the file are all valid ones
    fans = set(re.findall(r"^M106[^\n]*", raw.decode("ascii"), re.M))
    assert all(re.match(r"M106(\s+P[23])?(\s|$)", f) for f in fans), fans
    # tip window inside the paper window. Punch holes are at the FRONT, so the window is Y [13.5, 180.5] (front 10 mm strip + 2 mm; plain 3 mm back margin), not [4.5, 171.5] (which assumes holes at the back)
    tw = info["tip_window_mm"]
    assert 8.0 <= tw["x"][0] and tw["x"][1] <= 249.0 and 13.5 <= tw["y"][0] and tw["y"][1] <= 180.5, tw
    # the safety suite of the generator itself ran and the negative controls fired
    assert len(json.loads(Path(info["layout_path"]).read_text())["selftest_caught"]) == 5


def test_asset_sha_and_decode_integrity():
    xz_ok, raw_ok, xz, raw = pf.asset_shas()
    assert xz_ok and raw_ok, (xz_ok, raw_ok)
    meta, cells = pf.load_asset()
    assert len(meta["rows"]) == 8 and meta["arms"] == ["A", "S1.4", "S1.2", "S1.0", "C1024F", "NATcap"]
    assert len(cells) == 48 and all(len(v) > 50 for v in cells.values())
    # --- negative controls ---
    b64 = pfd.ASSET_XZ_B64
    flat = "".join(b64.split())
    mid = len(flat) // 2
    bad = flat[:mid] + ("B" if flat[mid] != "B" else "C") + flat[mid + 1:]
    bx, br, _, _ = pf.asset_shas(b64_text=bad)
    assert not (bx and br), "a corrupted blob passed the sha check"
    bx, br, _, _ = pf.asset_shas(xz_sha="0" * 64)
    assert not bx, "wrong recorded xz sha accepted"
    bx, br, _, _ = pf.asset_shas(raw_sha="0" * 64)
    assert not br, "wrong recorded payload sha accepted"
    assert _raises(pf.decode_payload, raw[:-5], meta, exc=(Exception,)), "truncated payload decoded"
    assert _raises(pf.decode_payload, raw + b"\x00", meta, exc=(pf.AssetError,)), "trailing bytes accepted"
    assert _raises(pf.decode_payload, b"XXXX" + raw[4:], meta, exc=(pf.AssetError,)), "bad magic accepted"
    # a refusing generator: poison the module's constant and the sheet must NOT generate
    saved, saved_cache = pfd.ASSET_XZ_SHA256, dict(pf._ASSET_CACHE)
    try:
        pf._ASSET_CACHE.clear()
        pfd.ASSET_XZ_SHA256 = "0" * 64
        assert _raises(SHEETS["photofe_bic"].generator, _WORK / "_pfe_poison.gcode", exc=(pf.AssetError,)), \
            "generator ran with a mismatching asset sha"
    finally:
        pfd.ASSET_XZ_SHA256 = saved
        pf._ASSET_CACHE.clear()
        pf._ASSET_CACHE.update(saved_cache)
    assert not (_WORK / "_pfe_poison.gcode").exists()
    # the clipped strokes lie inside the 35 x 13 mm window (+ rounding)
    for (ri, arm), pcs in cells.items():
        for pts, _b in pcs:
            assert all(-1e-9 <= x <= 35.0 + 1e-9 and -1e-9 <= y <= 13.0 + 1e-9 for x, y in pts), (ri, arm)


def test_every_cell_has_a_label_and_it_reaches_the_gcode():
    out, info = _gen()
    layout = json.loads(Path(info["layout_path"]).read_text(encoding="utf-8"))
    cells = layout["cells"]
    assert len(cells) == 48 and len({c["id"] for c in cells}) == 48
    assert len({(c["row_id"], c["arm"]) for c in cells}) == 48

    def check(cells, n_expected):
        for c in cells:
            a, b = c["elem_range_label"]
            assert c["label"].isascii() and c["label"] == f"{c['arm']} {c['row_id']}", c["label"]
            assert b - a >= 4, f"{c['id']}: label has {b - a} strokes"
            m0, m1 = c["elem_range_marks"]
            assert m1 - m0 == 4, f"{c['id']}: {m1 - m0} corner marks"
        assert info["n_pendown_actual_total"] == n_expected, "label strokes did not reach the gcode"

    n = info["n_pendown_expected_total"]
    check(cells, n)
    broken = [dict(c) for c in cells]
    broken[5]["elem_range_label"] = [broken[5]["elem_range_label"][0]] * 2
    assert _raises(check, broken, n), "missing label not caught"
    wrong = [dict(c) for c in cells]
    wrong[9]["label"] = "A nowhere"
    assert _raises(check, wrong, n), "wrong label not caught"
    assert _raises(check, cells, n - 1), "dropped gcode strokes not caught"
    assert info["pendown_count_matches"] is True
    # check_strokes itself rejects a dropped stroke (its own positive control)
    from ccplotter.calibration_pocket import check_strokes
    els, *_ = pf.build_layout()
    assert len(els) == n
    assert _raises(check_strokes, [{"pts": e["pts"], "f": [3000.0]} for e in els[:-1]], els, lambda x, y: (x, y),
                   exc=(RuntimeError,)), "dropped design stroke not caught"
    # the fiducials are emitted first
    kinds = [e["kind"] for e in els]
    assert kinds[:6] == ["fiducial"] * 6 or kinds[0] == "fiducial"
    assert kinds.index("data") > max(i for i, k in enumerate(kinds) if k == "fiducial")


def test_layout_reproduces_the_asset_without_retrace():
    meta, cells = pf.load_asset()
    els, recs, fixed, geoms, _m = pf.build_layout(meta, cells)
    for rec in recs:
        ri, arm = rec["row"], rec["arm"]
        ox, oy, x1, y1 = rec["window_mm"]
        a, b = rec["elem_range_data"]
        design = els[a:b]
        pcs = cells[(ri, arm)]
        assert len(design) == len(pcs), (rec["id"], len(design), len(pcs))
        for e, (pts, _flag) in zip(design, pcs):
            assert len(e["pts"]) == len(pts)
            for (ex, ey), (px, py) in zip(e["pts"], pts):
                assert abs(ex - (ox + px)) < 1e-3 and abs(ey - (oy + py)) < 1e-3, rec["id"]
                assert ox - 1e-9 <= ex <= x1 + 1e-9 and oy - 1e-9 <= ey <= y1 + 1e-9
    # windows are 35 x 13 and equally sized for every arm (scale 1: same physical size as a full-page plot)
    for rec in recs:
        ox, oy, x1, y1 = rec["window_mm"]
        assert abs((x1 - ox) - 35.0) < 1e-6 and abs((y1 - oy) - 13.0) < 1e-6
    # negative control: shift one asset point and the comparison must fail
    tampered = {k: [(list(p), f) for p, f in v] for k, v in cells.items()}
    tampered[(0, "A")][0][0][0] = (tampered[(0, "A")][0][0][0][0] + 0.5, tampered[(0, "A")][0][0][0][1])
    els2, recs2, *_ = pf.build_layout(meta, tampered)
    r0 = next(r for r in recs2 if r["id"] == "heli-letter|A")
    ox, oy, *_ = r0["window_mm"]
    e0 = els2[r0["elem_range_data"][0]]
    assert abs(e0["pts"][0][0] - (ox + cells[(0, "A")][0][0][0][0])) > 0.4, "tamper control is dead"


def test_geometry_clearance_and_its_negative_control():
    meta, cells = pf.load_asset()
    els, recs, fixed, geoms, _m = pf.build_layout(meta, cells)
    clr = pf.clearance_check(geoms, fixed)
    assert clr["n_pairs_checked"] > 0 and clr["min_clearance_mm"] == pf.CLEAR_MM
    # arms occupy columns in order; rows in order; 8 x 6
    assert {(r["row"], r["col"]) for r in recs} == {(r, c) for r in range(8) for c in range(6)}
    for r in recs:
        assert meta["arms"][r["col"]] == r["arm"]
    # emission order rotates by row: the first arm drawn differs between rows 0..5
    first = {}
    for r in sorted(recs, key=lambda r: r["emission_index"]):
        first.setdefault(r["row"], r["arm"])
    assert len(set(list(first.values())[:6])) == 6, first
    # negative control: a label dropped on top of a window must be caught
    from shapely.geometry import box
    bad_fixed = dict(fixed)
    bad_fixed["intruder"] = [[(recs[0]["window_mm"][0] + 5, recs[0]["window_mm"][1] + 5),
                              (recs[0]["window_mm"][0] + 9, recs[0]["window_mm"][1] + 5)]]
    assert _raises(pf.clearance_check, geoms, bad_fixed, exc=(RuntimeError,)), "intruding stroke not caught"
    # all strokes inside the sheet and the sheet inside the paper window
    xs = [q[0] for e in els for q in e["pts"]]
    ys = [q[1] for e in els for q in e["pts"]]
    assert min(xs) >= 0 and max(xs) <= 237.0 and min(ys) >= 0 and max(ys) <= 161.0


def test_nat_cap_repeatability_control_and_where_it_differs():
    meta, cells = pf.load_asset()
    same, differ = [], []
    for ri, row in enumerate(meta["rows"]):
        a, b = cells[(ri, "S1.2")], cells[(ri, "NATcap")]
        eq = (len(a) == len(b)) and all(pa == pb for (pa, _), (pb, _) in zip(a, b))
        (same if eq else differ).append(row["photo"])
        if row["short_px"] >= 1024:
            assert eq, f"{row['id']}: short side {row['short_px']} >= 1024 but NATcap != S1.2"
            assert row["nat_side_px"] == 1024
        else:
            assert not eq, f"{row['id']}: NATcap should differ from S1.2 (short side {row['short_px']})"
            assert row["nat_side_px"] == row["short_px"]
    assert sorted(set(same)) == ["belltower", "bike", "tram"] and sorted(set(differ)) == ["a350", "giraffe", "heli"], (same, differ)
    # negative control: the equality predicate does reject a perturbed copy
    a = cells[(1, "S1.2")]
    b = [(list(p), f) for p, f in cells[(1, "NATcap")]]
    b[3][0][0] = (b[3][0][0][0] + 0.01, b[3][0][0][1])
    assert not all(pa == pb for (pa, _), (pb, _) in zip(a, b)), "perturbation not detected"


def test_ledger_is_from_emitted_gcode_and_matches_the_asset():
    out, info = _gen()
    layout = json.loads(Path(info["layout_path"]).read_text(encoding="utf-8"))
    led = layout["ledger"]
    meta, cells = pf.load_asset()
    assert "population" in led and "border" in led["population"] and "EMITTED gcode" in led["population"]
    assert [a["arm"] for a in led["per_arm"]] == meta["arms"] or {a["arm"] for a in led["per_arm"]} == set(meta["arms"])
    tot = 0
    for a in led["per_arm"]:
        assert a["cells"] == 8 and a["pen_downs"] > 1000
        assert a["lt_0p5mm"] == a["lt_0p5mm_border"] + a["lt_0p5mm_interior"]
        tot += a["pen_downs"]
    n_data = sum(1 for r in layout["cells"] for _ in range(*r["elem_range_data"]))
    assert tot == n_data, (tot, n_data)
    # every cell's pen-down count equals the asset's piece count for that cell
    byid = {c["id"]: c for c in led["per_cell"]}
    for ri, row in enumerate(meta["rows"]):
        for arm in meta["arms"]:
            assert byid[f"{row['id']}|{arm}"]["pen_downs"] == row["cells"][arm]["n_pieces"], (row["id"], arm)
    # NEGATIVE CONTROL: the ledger refuses a misaligned stroke list (a dropped stroke in the middle)
    els, recs, *_ = pf.build_layout(meta, cells)
    from ccplotter.calibration_pocket import _parse_strokes_f
    strokes = _parse_strokes_f(out, info["z_down_abs"])
    assert len(strokes) == len(els)
    led2 = pf.ledger(recs, strokes)
    assert led2["per_arm"][0]["pen_downs"] == next(a for a in led["per_arm"] if a["arm"] == led2["per_arm"][0]["arm"])["pen_downs"]
    del strokes[recs[2]["elem_range_data"][0] + 1]
    shifted = pf.ledger(recs[:2], strokes)              # early cells still fine ...
    assert shifted["per_cell"][0]["pen_downs"] == recs[0]["elem_range_data"][1] - recs[0]["elem_range_data"][0]
    assert _raises(pf.ledger, recs, strokes, exc=(RuntimeError, IndexError)) or \
        sum(c["pen_downs"] for c in pf.ledger(recs, strokes)["per_cell"]) != n_data, "misalignment undetected"


def test_pen_pinned_zread_from_profile_and_fill_locked():
    out, info = _gen()
    base_z = info["z_down_abs"]
    assert abs(info["z_touch_abs"] - 18.75) < 1e-9 and abs(base_z - (18.75 - 0.6)) < 1e-9, info["z_touch_abs"]
    orig = (_ENV / "pen_profiles.json").read_bytes()
    try:
        p2 = json.loads(orig)
        p2["profiles"]["bic_cristal_blue"]["z_touch_abs"] = 17.5
        (_ENV / "pen_profiles.json").write_text(json.dumps(p2), encoding="utf-8")
        out2 = _WORK / "_pfe_z.gcode"
        info2 = SHEETS["photofe_bic"].generator(out2)
        assert abs(info2["z_down_abs"] - 16.9) < 1e-9, info2["z_down_abs"]
        assert min(float(m) for m in re.findall(r"Z(-?[\d.]+)", out2.read_text())) >= 16.9 - 1e-6
        assert "Z18.15" not in out2.read_text(), "old (fixture) z_down leaked into the file"
        assert out2.read_bytes() != out.read_bytes(), "determinism control: a different z_touch changed nothing"
        p3 = json.loads(orig)
        p3["profiles"]["bic_cristal_blue"]["fill_mm"] = 0.15
        (_ENV / "pen_profiles.json").write_text(json.dumps(p3), encoding="utf-8")
        assert _raises(SHEETS["photofe_bic"].generator, _WORK / "_pfe_fill.gcode", exc=(RuntimeError,)), \
            "fill_mm != 0.12 was not refused"
        (_ENV / "pen_profiles.json").write_bytes(orig)
        other = {"label": "Some Other Pen", "pen_mm": 0.5}
        assert _raises(SHEETS["photofe_bic"].generator, _WORK / "_pfe_pin.gcode", profile=other,
                       exc=(RuntimeError,)), "pinned pen was not enforced"
    finally:
        (_ENV / "pen_profiles.json").write_bytes(orig)


def test_microlift_applied_and_deterministic():
    out, info = _gen()
    ml = info["microlift"]
    assert ml and ml["short_lifts"] > 0.9 * ml["lifts"], ml
    z_short = round(info["z_touch_abs"] + 0.9, 2)
    zs = {round(float(m), 2) for m in re.findall(r"^G1 Z(-?[\d.]+)", out.read_text(), re.M)}
    assert z_short in zs, f"micro level {z_short} missing from {sorted(zs)}"
    out3 = _WORK / "_pfe_base2.gcode"
    SHEETS["photofe_bic"].generator(out3)
    ta = out.read_text().replace(out.name, "X")
    tb = out3.read_text().replace(out3.name, "X")
    assert ta == tb, "generator is not deterministic"
    assert ta != ta.replace("G1 Z", "G1 Y", 1), "(sanity) the comparison text is non-trivial"


def test_registered_generic_and_plot_warning():
    s = SHEETS["photofe_bic"]
    assert s.ui == "generic" and s.generator is not None and not getattr(s, "archived", False)
    assert "plot the linked g-code" in s.plot_warning.lower() and "z_touch" in s.plot_warning
    assert "nominal 0.12, laid unknown" in s.blurb
    assert "pen_id=bic_cristal_blue" in s.plot_warning


def _run_all():
    tests = [test_gcode_safety_and_reach, test_asset_sha_and_decode_integrity,
             test_every_cell_has_a_label_and_it_reaches_the_gcode, test_layout_reproduces_the_asset_without_retrace,
             test_geometry_clearance_and_its_negative_control, test_nat_cap_repeatability_control_and_where_it_differs,
             test_ledger_is_from_emitted_gcode_and_matches_the_asset,
             test_microlift_applied_and_deterministic, test_pen_pinned_zread_from_profile_and_fill_locked,
             test_registered_generic_and_plot_warning]
    passed, failed = [], []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed.append(t.__name__)
        except Exception as e:
            import traceback
            print(f"FAIL {t.__name__}: {str(e).encode('ascii', 'replace').decode('ascii')}")
            traceback.print_exc()
            failed.append(t.__name__)
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
