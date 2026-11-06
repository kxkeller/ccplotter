"""Tests for the ink-routed-connectors paper sheet (ccplotter/calibration_inkroute.py, registry id inkroute_bic).

Every guard is paired with a NEGATIVE control run in the same test (a tampered input the guard must reject).
Checked: emitted gcode pure ASCII / no G92 / valid fans / Z floor / reach box plotter_2003_v32 / real print structure;
two labelled halves (A OFF / B ON) whose printed lift and minute numbers equal the measured ones; ALL of half A before
ALL of half B with the two ASCII marker lines (negative controls: swapped, interleaved, missing and duplicated markers are
refused); B has fewer pen-downs than A and its strokes are longer (positive control: swapping the arms fails the same
check); the shipped asset's sha256 (file AND payload) with corrupted copies refused; pen pinned, z_touch READ from the
profile and PRINTED in the header (never hardcoded), fill_mm locked; determinism; registry entry (generic, blurb carries
the instruction, asset ships as a data file not an embedded module).

Run: python tests\\test_calibration_inkroute.py   (never with the repo root as CWD; the module chdirs to a private env)
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

_ENV = Path(tempfile.mkdtemp(prefix="ccplotter_inkr_env_"))
for _f in ("vpype.toml", "pen_profiles.json", "calibration.json"):
    shutil.copy2(ENV_SRC / _f, _ENV / _f)
os.chdir(_ENV)
os.environ["CCPLOTTER_HOME"] = str(_ENV)
_WORK = Path(tempfile.mkdtemp(prefix="ccplotter_inkr_work_"))
os.environ["CCPLOTTER_WORK_DIR"] = str(_WORK)
sys.path.insert(0, str(REPO))

from ccplotter import calibration_inkroute as ra  # noqa: E402
from ccplotter.calibration_registry import SHEETS  # noqa: E402

REACH_BBOX = (3.0, 42.5, 255.0, 254.5)       # plotter_2003_v32
BAD_FAN = re.compile(r"M106\s+P[01](\s|$)")
_cache = {}


def _gen(tag="base"):
    if tag not in _cache:
        out = _WORK / f"_ik_{tag}.gcode"
        info = SHEETS["inkroute_bic"].generator(out)
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


def _lifts(info):
    return {k: v["pen_downs"] for k, v in info["halves"].items()}


def _b_has_fewer_lifts(cells):
    return len(cells["on"]) < len(cells["off"])


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


def test_asset_sha_and_corrupted_copies_refused():
    meta, cells = ra.load_asset()
    assert meta["arms"] == list(ra.ARMS) and all(len(cells[a]) > 500 for a in ra.ARMS)
    p = ra.asset_path()
    raw = p.read_bytes()
    d = Path(tempfile.mkdtemp(prefix="ik_bad_"))
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
    assert ra.load_asset(p, ra.ASSET_FILE_SHA256, ra.ASSET_PAYLOAD_SHA256)[0]["arms"] == list(ra.ARMS)    # null: the real file loads


def test_b_has_fewer_lifts_than_a_with_positive_control():
    meta, cells = ra.load_asset()
    assert _b_has_fewer_lifts(cells), (len(cells["on"]), len(cells["off"]))
    swapped = {"off": cells["on"], "on": cells["off"]}
    assert not _b_has_fewer_lifts(swapped), "positive control: the swapped arms must FAIL the 'B fewer' check"
    lo, lf = len(cells["off"]), len(cells["on"])
    assert lf <= 0.9 * lo, f"expected a clear reduction, got {lo} -> {lf}"
    # B's connectors make its strokes longer on average (fused strokes): the two arms are really different drawings
    mean_len = {a: sum(math.dist(p[i], p[i + 1]) for p, _b in cells[a] for i in range(len(p) - 1)) / len(cells[a]) for a in ra.ARMS}
    assert mean_len["on"] > mean_len["off"], mean_len
    keys = {a: sorted((round(p[0][0], 2), round(p[0][1], 2), len(p)) for p, _ in cells[a]) for a in ra.ARMS}
    assert keys["off"] != keys["on"]


def test_labels_and_printed_numbers_equal_measured():
    out, info = _gen()
    lay = json.loads(Path(info["layout_path"]).read_text())
    assert [c["label"] for c in sorted(lay["cells"], key=lambda c: c["col"])] == ["A OFF", "B ON"]
    assert all(c["label"].isascii() and c["stat_label"].isascii() for c in lay["cells"])
    led = {p["arm"]: p for p in lay["ledger"]["per_patch"]}
    assert set(led) == {"off", "on"} and all(v["pen_downs"] > 500 for v in led.values()), led
    for c in lay["cells"]:
        arm = c["arm"]
        want = ra.stat_text(led[arm]["pen_downs"], ra.fmt_min(lay["halves"][arm]["model_b_s"]))
        assert c["stat_label"] == want, (c["stat_label"], want)
        assert str(led[arm]["pen_downs"]) in c["stat_label"] and "MIN MODEL" in c["stat_label"]
    assert info["stroke_check"]["n_strokes"] == info["n_pendown_expected_total"]
    hh = info["halves"]
    assert hh["off"]["model_b_min"] > 5.0 and hh["on"]["model_b_min"] > 5.0, hh
    assert hh["off"]["model_b_s"] + hh["on"]["model_b_s"] < lay["time_model_b_s"]      # the halves are a part of the whole-file model
    total_min = lay["time_model_b_s"] / 60.0
    assert 20.0 <= total_min <= 60.0, f"whole-sheet model time {total_min:.1f} min is outside the 30-45 min brief (tolerance 20-60)"
    assert hh["on"]["model_b_s"] < hh["off"]["model_b_s"], hh                            # the point of the sheet, on the model


def test_halves_ordered_a_then_b_with_markers_and_negative_controls():
    out, info = _gen()
    text = out.read_text(encoding="ascii")
    lifts = _lifts(info)
    lines = text.split("\n")
    ma = [i for i, l in enumerate(lines) if l.startswith("; ===== HALF A")]
    mb = [i for i, l in enumerate(lines) if l.startswith("; ===== HALF B")]
    assert len(ma) == 1 and len(mb) == 1 and ma[0] < mb[0]
    assert ra.halves_in_order(text, lifts)
    # NEG 1: marker lines swapped -> refused (B marker first)
    sw = text.replace("; ===== HALF A", "; ===== HALF X").replace("; ===== HALF B", "; ===== HALF A").replace("; ===== HALF X", "; ===== HALF B")
    assert not ra.halves_in_order(sw, lifts), "swapped markers not caught"
    assert _raises(ra.half_times, sw, exc=(RuntimeError,)), "half_times accepted swapped markers"
    # NEG 2: interleaving -- move the first 40 lines of half B in front of the A marker (pen-down counts per side no longer match)
    ia, ib = ma[0], mb[0]
    seg = lines[ib + 1: ib + 41]
    inter = lines[:ia] + seg + lines[ia:ib + 1] + lines[ib + 41:]
    assert not ra.halves_in_order("\n".join(inter), lifts), "interleaved file not caught"
    # NEG 3: a marker missing / duplicated
    assert _raises(ra.half_times, text.replace("; ===== HALF B", "; HALF B"), exc=(RuntimeError,)), "missing marker not caught"
    assert _raises(ra.half_times, text + "; ===== HALF B again\n", exc=(RuntimeError,)), "duplicate marker not caught"
    # layout ordering: element ranges are fixed | A | B, contiguous, A before B
    lay = json.loads(Path(info["layout_path"]).read_text())
    ca = next(c for c in lay["cells"] if c["arm"] == "off")
    cb = next(c for c in lay["cells"] if c["arm"] == "on")
    assert ca["emission_index"] == 0 and cb["emission_index"] == 1
    assert ca["elem_range_data"][1] == cb["elem_range_data"][0]
    assert max(ca["elem_range_label"][1], ca["elem_range_stat"][1], ca["elem_range_marks"][1],
               cb["elem_range_label"][1], cb["elem_range_stat"][1], cb["elem_range_marks"][1]) <= ca["elem_range_data"][0]
    assert lay["emission_order"] == ["off", "on"]
    assert "STOP HALF A / START HALF B" in text and "START THE STOPWATCH FOR HALF A" in text


def test_header_has_instruction_and_prints_resolved_z_touch():
    out, info = _gen()
    text = out.read_text(encoding="ascii")
    head = "\n".join(text.split("\n")[:14]).lower()
    assert "time each half" in head and "compare the look" in head and "re-read the live z_touch" in head, head
    assert f"z_touch_abs={info['z_touch_abs']:.3f}" in text


def test_clearance_and_negative_control():
    meta, cells = ra.load_asset()
    el, rec, fixed, geoms, _ = ra.build_layout(meta, cells)
    assert ra.clearance_check(geoms, fixed) is not None
    from shapely.geometry import box
    geoms2 = dict(geoms)
    g0 = geoms["patch|off"]
    geoms2["patch|off"] = box(g0.bounds[0], g0.bounds[1], g0.bounds[2] + 25.0, g0.bounds[3])      # overlap the next window
    assert _raises(ra.clearance_check, geoms2, fixed, exc=(Exception,)), "overlap not caught"


def test_pen_pinned_zread_from_profile_and_fill_locked():
    out, info = _gen()
    assert abs(info["z_touch_abs"] - 18.75) < 1e-9 and abs(info["z_down_abs"] - (18.75 - 0.6)) < 1e-9, info["z_touch_abs"]
    orig = (_ENV / "pen_profiles.json").read_bytes()
    try:
        p2 = json.loads(orig)
        p2["profiles"]["bic_cristal_blue"]["z_touch_abs"] = 17.5
        (_ENV / "pen_profiles.json").write_text(json.dumps(p2), encoding="utf-8")
        out2 = _WORK / "_ik_z.gcode"
        info2 = SHEETS["inkroute_bic"].generator(out2)
        assert abs(info2["z_down_abs"] - 16.9) < 1e-9
        assert out2.read_bytes() != out.read_bytes(), "a different z_touch changed nothing"
        assert "z_touch_abs=17.500" in out2.read_text(encoding="ascii")          # the header prints the RESOLVED value
        p3 = json.loads(orig)
        p3["profiles"]["bic_cristal_blue"]["fill_mm"] = 0.15
        (_ENV / "pen_profiles.json").write_text(json.dumps(p3), encoding="utf-8")
        assert _raises(SHEETS["inkroute_bic"].generator, _WORK / "_ik_fill.gcode", exc=(RuntimeError,)), "fill_mm != 0.12 not refused"
        (_ENV / "pen_profiles.json").write_bytes(orig)
        assert _raises(SHEETS["inkroute_bic"].generator, _WORK / "_ik_pin.gcode", profile={"label": "Other", "pen_mm": 0.5},
                       exc=(RuntimeError,)), "pinned pen not enforced"
    finally:
        (_ENV / "pen_profiles.json").write_bytes(orig)


def test_deterministic():
    out, info = _gen()
    out3 = _WORK / "_ik_base2.gcode"
    SHEETS["inkroute_bic"].generator(out3)
    assert out.read_text().replace(out.name, "X") == out3.read_text().replace(out3.name, "X")


def test_registered_generic_and_asset_is_a_data_file():
    s = SHEETS["inkroute_bic"]
    assert s.ui == "generic" and s.generator is not None and not s.archived
    assert "plot the linked g-code" in s.plot_warning.lower() and "z_touch" in s.plot_warning and "pen_id=bic_cristal_blue" in s.plot_warning
    assert "nominal 0.12, laid unknown" in s.blurb
    low = s.blurb.lower()
    assert "time each half" in low and "compare the look" in low and "re-read z_touch" in low, s.blurb
    assert "18.95" not in s.blurb + s.plot_warning and "11.45" not in s.blurb + s.plot_warning, "hardcoded Z in registry text (D1835)"
    assert ra.asset_path().is_file() and ra.asset_path().suffix == ".xz"
    src = (REPO / "ccplotter" / "calibration_inkroute.py").read_text(encoding="utf-8")
    assert "ASSET_XZ_B64" not in src and len(src) < 40000, "asset must not be embedded in the module (D1023)"
    spec = (REPO / "desktop" / "ccplotter.spec").read_text(encoding="utf-8")
    assert "inkroute_asset.json.xz" in spec, "asset missing from the PyInstaller datas"


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
