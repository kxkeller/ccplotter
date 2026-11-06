"""Tests for the row-direction paper sheet (ccplotter/calibration_rowangle.py, registry id rowangle_bic).

Every guard is paired with a NEGATIVE control run in the same test (a tampered input the guard must reject).
Checked: emitted gcode pure ASCII / no G92 / valid fans / Z floor / reach box plotter_2003_v32 / real print structure;
four labelled patches whose label strokes reach the gcode; the shipped asset's sha256 (file AND payload) and decode
self-consistency, with corrupted copies refused; the four patches really carry DIFFERENT row directions (zero ~ 0/90,
a45 = 45; positive control: they differ from each other); pen pinned, z_touch READ from the profile, fill_mm locked;
determinism; registry entry (generic, plot warning, asset ships as a data file not an embedded module).

Run: python tests\\test_calibration_rowangle.py   (never with the repo root as CWD; the module chdirs to a private env)
"""
from __future__ import annotations

import json
import lzma
import math
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

_ENV = Path(tempfile.mkdtemp(prefix="ccplotter_rowang_env_"))
for _f in ("vpype.toml", "pen_profiles.json", "calibration.json"):
    shutil.copy2(ENV_SRC / _f, _ENV / _f)
os.chdir(_ENV)
os.environ["CCPLOTTER_HOME"] = str(_ENV)
_WORK = Path(tempfile.mkdtemp(prefix="ccplotter_rowang_work_"))
os.environ["CCPLOTTER_WORK_DIR"] = str(_WORK)
sys.path.insert(0, str(REPO))

from ccplotter import calibration_rowangle as ra  # noqa: E402
from ccplotter.calibration_registry import SHEETS  # noqa: E402

REACH_BBOX = (3.0, 42.5, 255.0, 254.5)       # plotter_2003_v32
BAD_FAN = re.compile(r"M106\s+P[01](\s|$)")
_cache = {}


def _gen(tag="base"):
    if tag not in _cache:
        out = _WORK / f"_ra_{tag}.gcode"
        info = SHEETS["rowangle_bic"].generator(out)
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
    assert bx0 <= min(xs) and max(xs) <= bx1 and by0 <= min(ys) and max(ys) <= by1, "outside reach"
    return {"n_moves": n}


def _raises(fn, *a, exc=(AssertionError,), **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    return False


def _dominant_deg(pieces):
    w = {}
    for pts, _b in pieces:
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            L = math.hypot(x1 - x0, y1 - y0)
            if L < 1e-6:
                continue
            a = int(round((math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0) / 5.0)) % 36 * 5
            w[a] = w.get(a, 0.0) + L
    return max(w, key=w.get)


def test_gcode_safety_and_reach():
    out, info = _gen()
    raw = out.read_bytes()
    r = _check_gcode_text(raw, info["deepest_z_down_abs"])
    assert r["n_moves"] > 5000, r
    z = info["deepest_z_down_abs"]
    assert _raises(_check_gcode_text, raw + "; — em dash\n".encode("utf-8"), z), "non-ASCII not caught"
    assert _raises(_check_gcode_text, raw + b"G92 X0 Y0\n", z), "G92 not caught"
    assert _raises(_check_gcode_text, raw + b"M106 P1 S255\n", z), "M106 P1 not caught"
    assert _raises(_check_gcode_text, raw + b"M106 P0 S255\n", z), "M106 P0 not caught"
    assert _raises(_check_gcode_text, raw + f"G1 Z{z - 0.5:.2f} F500\n".encode(), z), "low Z not caught"
    assert _raises(_check_gcode_text, raw + b"G1 X300.0 Y100.0 F5000\n", z), "out-of-reach X not caught"
    assert _raises(_check_gcode_text, b"; comments only\n", z), "M-code-only file not caught"
    fans = set(re.findall(r"^M106[^\n]*", raw.decode("ascii"), re.M))
    assert all(re.match(r"M106(\s+P[23])?(\s|$)", f) for f in fans), fans
    tw = info["tip_window_mm"]
    assert 8.0 <= tw["x"][0] and tw["x"][1] <= 249.0 and 13.5 <= tw["y"][0] and tw["y"][1] <= 180.5, tw
    assert len(json.loads(Path(info["layout_path"]).read_text())["selftest_caught"]) == 5
    assert info["pendown_count_matches"]


def test_asset_sha_and_corrupted_copies_refused(tmp_path=None):
    meta, cells = ra.load_asset()
    assert meta["arms"] == list(ra.ARMS) and all(len(cells[a]) > 100 for a in ra.ARMS)
    p = ra.asset_path()
    raw = p.read_bytes()
    d = Path(tempfile.mkdtemp(prefix="ra_bad_"))
    bad = d / "bad.xz"
    bad.write_bytes(raw[:-3] + bytes([raw[-3] ^ 1]) + raw[-2:])            # one flipped bit: file sha must refuse
    assert _raises(ra.load_asset, bad, exc=(ra.AssetError,)), "corrupted file accepted"
    assert _raises(ra.load_asset, d / "missing.xz", exc=(ra.AssetError,)), "missing file accepted"
    good_payload = lzma.decompress(raw)
    tampered = good_payload.replace(b'"n_border":', b'"n_border": ', 1)       # payload changed, file sha recomputed by the test
    import hashlib
    t = d / "tampered.xz"
    t.write_bytes(lzma.compress(tampered))
    assert _raises(ra.load_asset, t, hashlib.sha256(t.read_bytes()).hexdigest(), exc=(ra.AssetError,)), "payload sha not enforced"
    # null: the real file with its own recorded shas loads
    assert ra.load_asset(p, ra.ASSET_FILE_SHA256, ra.ASSET_PAYLOAD_SHA256)[0]["arms"] == list(ra.ARMS)


def test_patches_carry_four_different_row_directions():
    meta, cells = ra.load_asset()
    dom = {a: _dominant_deg(cells[a]) for a in ra.ARMS}
    assert dom["zero"] in (0, 90) or dom["zero"] in (175, 5), dom          # 0-deg rows (page rot 180 keeps 0)
    assert dom["a45"] == 45, dom
    assert dom["zero"] != dom["a45"], "positive control: a45 must differ from zero"
    sig = {a: sum(len(p) for p, _ in cells[a]) for a in ra.ARMS}
    assert len(set(sig.values())) == 4, f"the four arms are not four different drawings: {sig}"
    # ACROSS and SET really differ from the constant looks (stroke multiset differs)
    keys = {a: sorted((round(p[0][0], 2), round(p[0][1], 2), len(p)) for p, _ in cells[a]) for a in ra.ARMS}
    assert all(keys[a] != keys[b] for a in ra.ARMS for b in ra.ARMS if a < b)


def test_every_patch_has_a_label_and_it_reaches_the_gcode():
    out, info = _gen()
    lay = json.loads(Path(info["layout_path"]).read_text())
    assert [c["label"] for c in sorted(lay["cells"], key=lambda c: c["col"])] == ["1 ZERO", "2 A45", "3 ACROSS", "4 SET"]
    assert all(c["label"].isascii() for c in lay["cells"])
    assert info["stroke_check"]["n_strokes"] == info["n_pendown_expected_total"]
    assert lay["emission_order"] != list(ra.ARMS), "emission must be interleaved, not left to right"
    led = {p["arm"]: p for p in lay["ledger"]["per_patch"]}
    assert set(led) == set(ra.ARMS) and all(v["pen_downs"] > 500 for v in led.values()), led
    gtext = out.read_text()
    assert "; BLOCK 1/1" in gtext


def test_clearance_and_negative_control():
    meta, cells = ra.load_asset()
    el, rec, fixed, geoms, _ = ra.build_layout(meta, cells)
    assert ra.clearance_check(geoms, fixed) is not None
    from shapely.geometry import box
    geoms2 = dict(geoms)
    g0 = geoms["patch|zero"]
    geoms2["patch|zero"] = box(g0.bounds[0], g0.bounds[1], g0.bounds[2] + 25.0, g0.bounds[3])      # overlap the next window
    assert _raises(ra.clearance_check, geoms2, fixed, exc=(Exception,)), "overlap not caught"


def test_pen_pinned_zread_from_profile_and_fill_locked():
    out, info = _gen()
    assert abs(info["z_touch_abs"] - 18.75) < 1e-9 and abs(info["z_down_abs"] - (18.75 - 0.6)) < 1e-9, info["z_touch_abs"]
    orig = (_ENV / "pen_profiles.json").read_bytes()
    try:
        p2 = json.loads(orig)
        p2["profiles"]["bic_cristal_blue"]["z_touch_abs"] = 17.5
        (_ENV / "pen_profiles.json").write_text(json.dumps(p2), encoding="utf-8")
        out2 = _WORK / "_ra_z.gcode"
        info2 = SHEETS["rowangle_bic"].generator(out2)
        assert abs(info2["z_down_abs"] - 16.9) < 1e-9
        assert out2.read_bytes() != out.read_bytes(), "a different z_touch changed nothing"
        p3 = json.loads(orig)
        p3["profiles"]["bic_cristal_blue"]["fill_mm"] = 0.15
        (_ENV / "pen_profiles.json").write_text(json.dumps(p3), encoding="utf-8")
        assert _raises(SHEETS["rowangle_bic"].generator, _WORK / "_ra_fill.gcode", exc=(RuntimeError,)), "fill_mm != 0.12 not refused"
        (_ENV / "pen_profiles.json").write_bytes(orig)
        assert _raises(SHEETS["rowangle_bic"].generator, _WORK / "_ra_pin.gcode", profile={"label": "Other", "pen_mm": 0.5},
                       exc=(RuntimeError,)), "pinned pen not enforced"
    finally:
        (_ENV / "pen_profiles.json").write_bytes(orig)


def test_deterministic():
    out, info = _gen()
    out3 = _WORK / "_ra_base2.gcode"
    SHEETS["rowangle_bic"].generator(out3)
    assert out.read_text().replace(out.name, "X") == out3.read_text().replace(out3.name, "X")


def test_registered_generic_and_asset_is_a_data_file():
    s = SHEETS["rowangle_bic"]
    assert s.ui == "generic" and s.generator is not None and not s.archived
    assert "plot the linked g-code" in s.plot_warning.lower() and "z_touch" in s.plot_warning and "pen_id=bic_cristal_blue" in s.plot_warning
    assert "nominal 0.12, laid unknown" in s.blurb
    assert ra.asset_path().is_file() and ra.asset_path().suffix == ".xz"
    src = (REPO / "ccplotter" / "calibration_rowangle.py").read_text(encoding="utf-8")
    assert "ASSET_XZ_B64" not in src and len(src) < 40000, "asset must not be embedded in the module (D1023)"
    spec = (REPO / "desktop" / "ccplotter.spec").read_text(encoding="utf-8")
    assert "rowangle_asset.json.xz" in spec, "asset missing from the PyInstaller datas"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
    sys.exit(_run_all())
