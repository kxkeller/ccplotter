"""Tests for ccplotter/calibration_uright_short.py (the uRight short sheet).

Each guard is paired with a negative control where one is cheap (a check that cannot fail proves nothing).
Run: python tests\\test_calibration_uright_short.py   (one process per file)
"""
from __future__ import annotations

import hashlib
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

_ENV = Path(tempfile.mkdtemp(prefix="ccplotter_urs_env_"))
for _f in ("vpype.toml", "pen_profiles.json", "calibration.json"):
    shutil.copy2(ENV_SRC / _f, _ENV / _f)
os.chdir(_ENV)
os.environ["CCPLOTTER_HOME"] = str(_ENV)
_WORK = Path(tempfile.mkdtemp(prefix="ccplotter_urs_work_"))
os.environ["CCPLOTTER_WORK_DIR"] = str(_WORK)
sys.path.insert(0, str(REPO))

from ccplotter import calibration_uright_short as cu  # noqa: E402
from ccplotter.calibration_registry import SHEETS, generic_sheets  # noqa: E402

REACH_BBOX = (3.0, 42.5, 255.0, 254.5)
BAD_FAN = re.compile(r"M106\s+P[01](\s|$)")
OKFAN = re.compile(r"M106(\s+P[23])?(\s|$)")
_cache = {}


def _gen(tag="base"):
    if tag not in _cache:
        out = _WORK / f"_urs_{tag}.gcode"
        _cache[tag] = (out, SHEETS["uright_short"].generator(out))
    return _cache[tag]


def _profiles():
    return json.loads((_ENV / "pen_profiles.json").read_text(encoding="utf-8"))


def _fixture_pen():
    return _profiles()["profiles"]["uright_gel"]


def _refuses(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except RuntimeError:
        return True
    return False


def test_registered_generic_and_listed():
    assert "uright_short" in SHEETS and SHEETS["uright_short"] in generic_sheets()
    s = SHEETS["uright_short"]
    assert "UNMEASURED" in s.blurb and "uright_gel_halfpitch" in s.plot_warning


def test_safety_set_and_reach():
    out, info = _gen()
    text = out.read_bytes().decode("ascii")          # raises on non-ASCII
    lines = text.split("\n")
    assert not [l for l in lines if l.startswith("G92")]
    assert not [l for l in lines if BAD_FAN.search(l)]
    assert all(OKFAN.search(l) for l in lines if l.startswith("M106"))
    xs, ys, zs = [], [], []
    for l in lines:
        if l.startswith(("G0", "G1")):
            for a, acc in (("X", xs), ("Y", ys), ("Z", zs)):
                m = re.search(a + r"(-?[\d.]+)", l)
                if m:
                    acc.append(float(m.group(1)))
    assert xs and ys and zs
    bx0, by0, bx1, by1 = REACH_BBOX
    assert bx0 <= min(xs) and max(xs) <= bx1 and by0 <= min(ys) and max(ys) <= by1
    assert min(zs) >= info["deepest_z_down_abs"] - 1e-6
    # negative control: the reach predicate fails on an escaped coordinate
    assert not (bx0 <= min(xs) - 1000)
    assert BAD_FAN.search("M106 P1") and not BAD_FAN.search("M106 P2")


def test_pen_pinned_and_family_variants():
    pen = _fixture_pen()
    assert _refuses(cu.uright_short_gcode, _WORK / "_pin.gcode", profile={"label": "Other", "pen_mm": 0.5})
    other = _profiles()["profiles"]["bic_cristal_blue"]
    assert _refuses(cu.uright_short_gcode, _WORK / "_pin2.gcode", profile=other)
    assert _refuses(cu.uright_short_gcode, _WORK / "_pin3.gcode", pen_id="bic_cristal_blue")
    orig = (_ENV / "pen_profiles.json").read_bytes()
    try:
        p2 = json.loads(orig)
        v = dict(pen)
        v["label"] = "variant"
        v["fill_mm"] = 0.13
        v["laid_mm"] = 0.051
        p2["profiles"]["uright_gel_halfpitch"] = v
        (_ENV / "pen_profiles.json").write_text(json.dumps(p2), encoding="utf-8")
        info = cu.uright_short_gcode(_WORK / "_var.gcode", profile=v)
        assert info["pen_id"] == "uright_gel" and info["requested_pen_id"] == "uright_gel_halfpitch"
        assert info["pin_note"]
        v2 = dict(v)
        v2["z_touch_abs"] = v["z_touch_abs"] + 1.0
        p2["profiles"]["uright_gel_tight3x"] = v2
        (_ENV / "pen_profiles.json").write_text(json.dumps(p2), encoding="utf-8")
        assert _refuses(cu.uright_short_gcode, _WORK / "_var2.gcode", profile=v2)
    finally:
        (_ENV / "pen_profiles.json").write_bytes(orig)


def test_z_touch_read_from_profile():
    out, info = _gen()
    pen = _fixture_pen()
    assert abs(info["z_touch_abs"] - pen["z_touch_abs"]) < 1e-9
    orig = (_ENV / "pen_profiles.json").read_bytes()
    try:
        p2 = json.loads(orig)
        p2["profiles"]["uright_gel"]["z_touch_abs"] = 12.0
        (_ENV / "pen_profiles.json").write_text(json.dumps(p2), encoding="utf-8")
        o2 = _WORK / "_urs_z.gcode"
        i2 = SHEETS["uright_short"].generator(o2)
        assert abs(i2["z_down_abs"] - (12.0 + pen["z_down"])) < 1e-9
        t2 = o2.read_text()
        zs = [float(m) for m in re.findall(r"Z(-?[\d.]+)", t2)]
        assert min(zs) >= 12.0 + pen["z_down"] - 1e-6
        assert f"Z{pen['z_touch_abs'] + pen['z_down']:.2f}" not in t2
    finally:
        (_ENV / "pen_profiles.json").write_bytes(orig)


def test_pendown_count_matches_design():
    out, info = _gen()
    els, recs, fixed, geoms, meta = cu.build_layout(3000.0)
    assert info["n_pendown_expected_total"] == len(els) == info["n_pendown_actual_total"]
    assert info["pendown_count_matches"]
    n, prev = 0, False                                 # independent count from the file
    for l in out.read_text().split("\n"):
        if l.startswith("G1") and "Z" in l:
            down = abs(float(re.search(r"Z(-?[\d.]+)", l).group(1)) - info["z_down_abs"]) < 1e-2
            n += int(down and not prev)
            prev = down
    assert n == len(els), (n, len(els))


def test_wedge_patches_one_stroke_and_geometry_absolute():
    els, recs, fixed, geoms, meta = cu.build_layout(3000.0)
    wedge = [r for r in recs if r["kind"] == "wedge"]
    assert len(wedge) == 26
    assert sorted({r["pitch_mm"] for r in wedge}) == sorted(cu.WEDGE_PITCHES_MM)
    for r in wedge:
        assert r["n_pen_downs"] == 1, r["id"]
        assert r["pitch_mm"] >= 0.08 - 1e-12
    assert any(r["n_pen_downs"] > 1 for r in recs if r["kind"] == "pair" and r["arm"] == "L")  # neg control
    base = _gen()[0]
    orig = (_ENV / "pen_profiles.json").read_bytes()
    try:
        p2 = json.loads(orig)
        p2["profiles"]["uright_gel"]["fill_mm"] = 0.5
        p2["profiles"]["uright_gel"]["laid_mm"] = 0.9
        (_ENV / "pen_profiles.json").write_text(json.dumps(p2), encoding="utf-8")
        o2 = _WORK / "_urs_fill.gcode"
        SHEETS["uright_short"].generator(o2)

        def body(p):
            return "\n".join(l for l in p.read_text().split("\n") if not l.startswith(";"))
        assert body(o2) == body(base), "geometry depends on fill_mm/laid_mm"
    finally:
        (_ENV / "pen_profiles.json").write_bytes(orig)


def test_patch_size_limits():
    for it in cu.build_items():
        if it["kind"] in ("wedge", "pair"):
            st = cu.build_patch(it)
            xs = [q[0] for s in st for q in s]
            ys = [q[1] for s in st for q in s]
            assert max(xs) - min(xs) <= 6.0 + 1e-9 and max(ys) - min(ys) <= 6.0 + 1e-9, it["id"]
    assert cu.MIN_PITCH_MM == 0.08


def test_pair_equal_ink_and_arms():
    els, recs, fixed, geoms, meta = cu.build_layout(3000.0)
    pair = [r for r in recs if r["kind"] == "pair"]
    assert len(pair) == 16
    for p in cu.PAIR_PITCHES_MM:
        for rep in range(cu.PAIR_REPS):
            n = next(r for r in pair if r["arm"] == "N" and r["pitch_mm"] == p and r["rep"] == rep)
            l = next(r for r in pair if r["arm"] == "L" and r["pitch_mm"] == p and r["rep"] == rep)
            assert abs(n["ink_mm"] - l["ink_mm"]) < 1e-6, (n["id"], n["ink_mm"], l["ink_mm"])
            assert n["n_pen_downs"] == 1 and l["n_pen_downs"] > 100
    l = next(r for r in pair if r["arm"] == "L" and r["pitch_mm"] == 0.13)
    assert 0.2 < l["ink_mm"] / l["n_pen_downs"] < 0.42
    n13 = next(r for r in pair if r["arm"] == "N" and r["pitch_mm"] == 0.13)
    n25 = next(r for r in pair if r["arm"] == "N" and r["pitch_mm"] == 0.25)
    assert abs(n13["ink_mm"] - n25["ink_mm"]) > 10                    # negative control
    for arm in ("N", "L"):                                           # not confounded with position
        assert {r["cell"][1] for r in pair if r["arm"] == arm} == {0, 1}


def test_contact_grid_identical_marks():
    els, recs, fixed, geoms, meta = cu.build_layout(3000.0)
    marks = [r for r in recs if r["kind"] == "mark"]
    assert len(marks) == 24 and all(r["n_pen_downs"] == 3 for r in marks)
    assert len({round(r["ink_mm"], 6) for r in marks}) == 1
    xs = [r["center_mm"][0] for r in marks]
    ys = [r["center_mm"][1] for r in marks]
    assert max(xs) - min(xs) > 150 and max(ys) - min(ys) > 100


def test_fiducials_first_and_labels():
    els, recs, fixed, geoms, meta = cu.build_layout(3000.0)
    kinds = [e["kind"] for e in els]
    first_item = next(i for i, k in enumerate(kinds) if k in ("wedge", "pair", "mark", "line"))
    assert kinds[:6].count("fiducial") == 6 or kinds[0] == "fiducial"
    assert all(k in ("fiducial", "glyph", "ruler", "prime", "legend") for k in kinds[:first_item])
    assert all(k != "fiducial" for k in kinds[first_item:])
    for r in recs:
        assert r["n_label_strokes"] > 0, r["id"]
    assert len(recs) == 26 + 3 + 16 + 24


def test_clearance_guard_has_teeth():
    from ccplotter.calibration_pocket import _check_clearance
    from shapely.geometry import box
    out, info = _gen()
    # n_pairs_checked counts only pairs whose boxes are within 2 mm; it is 0 when everything is
    # well separated, so assert the group count and recompute the minimum gap independently.
    assert info["clearance_check"]["n_groups"] > 50
    els, recs, fixed, geoms, meta = cu.build_layout(3000.0)
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    g = list(geoms.values())
    for k, pls in fixed.items():  # fixed groups too (legend, rulers, prime, fiducials)
        g.append(unary_union([LineString(pl) for pl in pls if len(pl) >= 2]).buffer(0.05))
    gap = min(g[i].distance(g[j]) for i in range(len(g)) for j in range(i + 1, len(g)))
    assert gap >= 2.0 - 1e-9, gap
    assert _refuses(_check_clearance, {"a": box(0, 0, 3, 3), "b": box(4, 0, 7, 3)}, {})


def test_microlift_and_feeds_from_profile():
    out, info = _gen()
    ml = info["microlift"]
    assert ml and ml["short_lifts"] > 0.9 * ml["lifts"], ml
    assert f"F{int(_fixture_pen()['draw_speed'])}" in out.read_text()


def test_deterministic():
    o2 = _WORK / "_urs_again.gcode"
    SHEETS["uright_short"].generator(o2)
    def body(p):                      # comment lines carry the output file name
        return chr(10).join(l for l in p.read_text().split(chr(10)) if not l.startswith(";"))
    assert hashlib.sha256(body(o2).encode()).hexdigest() == hashlib.sha256(body(_gen()[0]).encode()).hexdigest()


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("FAIL", t.__name__, str(e).encode("ascii", "replace").decode())
            bad += 1
    print(f"\n{len(tests) - bad} passed, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
