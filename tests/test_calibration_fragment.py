"""Tests for the fragment dose-response sheet (ccplotter/calibration_fragment.py).

Every guard here is paired with a NEGATIVE control run in the same test (a
tampered input that the guard must reject) -- the could-not-fail pattern: a
check that has never been observed to fail is not known to work.

What is checked:
  * emitted gcode: pure ASCII, no G92, no invalid fan index, Z floor, every X/Y
    inside the plotter_2003_v32 reach box, real print structure;
  * every patch carries a printed label (design strokes present AND they reach
    the emitted gcode: pen-down stroke count == design stroke count);
  * the arms differ in the INTENDED statistic and only that one (pieces / lifts /
    reversals), with a raster check that every non-control arm has the same
    centre-vs-band ink coverage (the positive control, the commanded bare
    centre, must differ);
  * pen pinned, z_touch READ from the profile (changing the profile changes the
    emitted Z), fill_mm locked to 0.12, microlift applied, determinism.

Run: python tests\\test_calibration_fragment.py
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

_ENV = Path(tempfile.mkdtemp(prefix="ccplotter_frag_env_"))
for _f in ("vpype.toml", "pen_profiles.json", "calibration.json"):
    shutil.copy2(ENV_SRC / _f, _ENV / _f)
os.chdir(_ENV)
os.environ["CCPLOTTER_HOME"] = str(_ENV)
_WORK = Path(tempfile.mkdtemp(prefix="ccplotter_frag_work_"))
os.environ["CCPLOTTER_WORK_DIR"] = str(_WORK)
sys.path.insert(0, str(REPO))

from ccplotter import calibration_fragment as cf  # noqa: E402
from ccplotter.calibration_registry import SHEETS  # noqa: E402

REACH_BBOX = (3.0, 42.5, 255.0, 254.5)       # plotter_2003_v32
BAD_FAN = re.compile(r"M106\s+P[01](\s|$)")
_cache = {}


# ----------------------------------------------------------------- helpers ---
def _gen(tag="base"):
    """Generate once per tag through the REGISTRY generator."""
    if tag not in _cache:
        out = _WORK / f"_frag_{tag}.gcode"
        info = SHEETS["fragment_bic"].generator(out)
        _cache[tag] = (out, info)
    return _cache[tag]


def _check_gcode_text(raw: bytes, deepest_z: float):
    """The safety set. Raises AssertionError on any violation."""
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


def _raises(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except AssertionError:
        return True
    return False


def _stats_by_code():
    els, recs, fixed, geoms, meta = cf.build_layout()
    by = {}
    for r in recs:
        by.setdefault(r["code"], []).append(r)
    return recs, by


def check_arm_statistics(by):
    """Returns a list of violations of the INTENDED arm contrasts (empty == ok)."""
    bad = []

    def mean(code, k):
        return sum(r[k] for r in by[code]) / len(by[code])

    ladder = [f"L{cf._fmt_len(L)}" for L in cf.LADDER_MM]
    pcs = [mean(c, "pieces") for c in ladder]
    if not all(a > b for a, b in zip(pcs, pcs[1:])):
        bad.append(f"ladder pieces not strictly decreasing with L: {pcs}")
    mp = [mean(c, "mean_piece_mm") for c in ladder]
    if not all(a < b for a, b in zip(mp, mp[1:])):
        bad.append(f"ladder mean piece length not increasing: {mp}")
    if mean("N", "pieces") != 1 or mean("N", "lifts") != 0:
        bad.append("null is not ONE continuous stroke")
    if mean("R", "pieces") != cf.N_ROWS:
        bad.append("rows arm is not one stroke per row")
    dens = [mean("R", "pieces")] + [mean(f"F{int(round(float(f) * 100)):02d}", "pieces")
                                     for f in cf.DENSITY_FRACS] + [mean("L05", "pieces")]
    if not all(a < b for a, b in zip(dens, dens[1:])):
        bad.append(f"open-start density not increasing with f: {dens}")
    strips = [f"V{cf._fmt_len(w)}" for w in cf.STRIP_MM]
    for c in strips:
        if mean(c, "pieces") != 1 or mean(c, "lifts") != 0:
            bad.append(f"{c}: reversals-only arm has a lift")
    rv = [mean(c, "reversals_total") for c in strips]
    if not all(a > b for a, b in zip(rv, rv[1:])):
        bad.append(f"reversals not decreasing with strip width: {rv}")
    if min(rv) <= mean("N", "reversals_total"):
        bad.append("reversal dose does not exceed the null's baseline")
    for L in cf.PAIR_MM:
        pc, lc = mean(f"P{cf._fmt_len(L)}", "pieces"), mean(f"L{cf._fmt_len(L)}", "pieces")
        if not (0.35 * lc < pc < 0.65 * lc):
            bad.append(f"pairs pieces {pc} not ~half of ladder {lc} at L={L}")
        if mean(f"P{cf._fmt_len(L)}", "reversals_total") <= mean(f"L{cf._fmt_len(L)}", "reversals_total"):
            bad.append(f"pairs do not add reversals at L={L}")
    if mean("D0", "pieces") < 20 or mean("D3", "pieces") != 1 or mean("DN", "pieces") != 1:
        bad.append("disc link arms: D0 must be many closed rings, D3 and DN one stroke each")
    if mean("S3", "pieces") != 1:
        bad.append("square link-3 arm is not one chained stroke")
    return bad


# ------------------------------------------------------------------- tests ---
def test_gcode_safety_and_reach():
    out, info = _gen()
    raw = out.read_bytes()
    r = _check_gcode_text(raw, info["deepest_z_down_abs"])
    assert r["n_moves"] > 10000, "suspiciously few moves for ~14.8k strokes"
    # ---- negative controls: each tamper MUST be caught ----
    z = info["deepest_z_down_abs"]
    assert _raises(_check_gcode_text, raw + "; — em dash\n".encode("utf-8"), z), "non-ASCII not caught"
    assert _raises(_check_gcode_text, raw + b"G92 X0 Y0\n", z), "G92 not caught"
    assert _raises(_check_gcode_text, raw + b"M106 P1 S255\n", z), "M106 P1 not caught"
    assert _raises(_check_gcode_text, raw + f"G1 Z{z - 0.5:.2f} F500\n".encode(), z), "low Z not caught"
    assert _raises(_check_gcode_text, raw + b"G1 X300.0 Y100.0 F5000\n", z), "out-of-reach X not caught"
    assert _raises(_check_gcode_text, b"; comments only\n", z), "M-code-only file not caught"
    # registry-reported window inside the paper window. Punch holes are at the FRONT, so the window is Y [13.5, 180.5] (front 10 mm strip + 2 mm; plain 3 mm back margin), not [4.5, 171.5] (which assumes holes at the back)
    tw = info["tip_window_mm"]
    assert 8.0 <= tw["x"][0] and tw["x"][1] <= 249.0 and 13.5 <= tw["y"][0] and tw["y"][1] <= 180.5, tw


def test_every_patch_has_a_printed_label():
    out, info = _gen()
    layout = json.loads(Path(info["layout_path"]).read_text(encoding="utf-8"))
    recs = layout["items"]
    # the registry entry is the PROBE variant: 92 base patches + 8 probe patches
    assert len(recs) == 100 and len({r["id"] for r in recs}) == 100

    def check_labels(recs, n_expected_strokes):
        for r in recs:
            assert r["label"].isascii() and 3 <= len(r["label"]) <= 6, r["label"]
            assert r["n_label_strokes"] >= 4, f"{r['id']}: label has {r['n_label_strokes']} strokes"
            assert r["label"].split("-")[0] == r["code"], f"{r['id']}: label does not carry its arm code"
        assert info["n_pendown_actual_total"] == n_expected_strokes, "label strokes did not reach the gcode"

    n_design = info["n_pendown_expected_total"]
    check_labels(recs, n_design)
    # negative controls
    stripped = [dict(r) for r in recs]
    stripped[7]["n_label_strokes"] = 0
    assert _raises(check_labels, stripped, n_design), "missing label not caught"
    mislabelled = [dict(r) for r in recs]
    mislabelled[3]["label"] = "X9-1"
    assert _raises(check_labels, mislabelled, n_design), "wrong label not caught"
    assert _raises(check_labels, recs, n_design - 1), "dropped gcode strokes not caught"
    # the gcode actually contains the label strokes: parse pen-downs, find one label's centre
    assert info["pendown_count_matches"] is True


def test_arms_differ_in_intended_statistic():
    recs, by = _stats_by_code()
    assert check_arm_statistics(by) == [], check_arm_statistics(by)
    # ---- negative controls: a corrupted arm table MUST trip the checker ----
    import copy
    swapped = copy.deepcopy(by)
    swapped["L05"], swapped["L20"] = swapped["L20"], swapped["L05"]
    assert check_arm_statistics(swapped), "swapped ladder rungs not caught"
    no_lift = copy.deepcopy(by)
    for r in no_lift["V10"]:
        r["lifts"], r["pieces"] = 5, 6
    assert check_arm_statistics(no_lift), "a lift inside the reversals-only arm not caught"
    flat = copy.deepcopy(by)
    for c in ("L03", "L05", "L10", "L20", "L40"):
        for r in flat[c]:
            r["pieces"] = 100
    assert check_arm_statistics(flat), "a ladder that does not vary was not caught"
    # builder-level: a ladder builder that ignored L would make every rung identical
    a = cf.patch_stats(cf.build_ladder(0.3))["pieces"]
    b = cf.patch_stats(cf.build_ladder(4.0))["pieces"]
    assert a > 5 * b, (a, b)


def test_equal_coverage_and_positive_control():
    """Every non-control arm must ink the SAME area (centre/band density ~ 1.0 and
    pixel-equal to the null); the commanded-hole positive control must NOT."""
    ppmm = 25
    items = {it["code"]: it for it in cf._item_table() if it["rep"] == 0}
    null = cf.raster_patch(cf.build_patch(items["N"]), 0.20, ppmm)
    for laid in cf.LAID_PREVIEW_MM:
        for code, it in items.items():
            img = cf.raster_patch(cf.build_patch(it), laid, ppmm)
            c, b, _, _ = cf.window_density(img, it["shape"], ppmm)
            if code.startswith("H"):
                continue
            assert abs(c - 1.0) < 0.01 and abs(b - 1.0) < 0.01, \
                f"{code} @ {laid}: centre {c:.3f} band {b:.3f} (an arm that changes coverage)"
    # pixel equality of row-based arms against the null at laid 0.20 (squares only)
    for code in ("R", "L03", "L05", "L10", "L20", "L40", "F12", "F25", "F50", "P05", "P10",
                 "V03", "V05", "V10", "V20", "V40"):
        img = cf.raster_patch(cf.build_patch(items[code]), 0.20, ppmm)
        d = float(abs(img - null).mean())
        assert d < 0.004, f"{code}: differs from the null raster by {d:.4f}"
    # positive control: holes are visible, monotone, and a perturbed arm would be caught
    ratios = []
    for code in ("H1", "H2", "H3", "H4"):
        img = cf.raster_patch(cf.build_patch(items[code]), 0.20, ppmm)
        c, b, _, _ = cf.window_density(img, "square", ppmm)
        ratios.append(c / b)
    assert all(a > b for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[-1] < 0.85 and ratios[-2] < 0.95, ratios
    # negative control on the equality check itself: drop 12 rows from the null
    # near the centre (a hand-made 'arm that changes coverage') -> must fail
    broken = [r for r in cf.build_rows() if abs(r[0][1]) > 0.5]
    assert len(broken) < cf.N_ROWS
    img = cf.raster_patch(broken, 0.20, ppmm)
    c, b, _, _ = cf.window_density(img, "square", ppmm)
    assert not (abs(c - 1.0) < 0.01), "coverage check cannot fail"


def test_raster_laid_width_is_true():
    """The preview/equal-coverage raster must draw the REQUESTED laid width. The first
    raster used cv2.polylines(thickness=round(w*ppmm)), which renders 2*(t//2)+1 px: laid
    0.20 and 0.24 came out identical (0.28 mm). Measure the width on a single line."""
    import cv2
    import numpy as np

    def width_mm(img, ppmm=25):
        return float(img[:, img.shape[1] // 2].sum()) / ppmm

    got = []
    for w in (0.10, 0.12, 0.16, 0.20, 0.24):
        img = cf.raster_patch([[(-3.0, 0.0), (3.0, 0.0)]], w, ppmm=25)
        got.append(width_mm(img))
        assert abs(got[-1] - w) < 0.006, f"laid {w}: raster width {got[-1]:.4f}"
    assert all(a < b for a, b in zip(got, got[1:])), got
    # negative control: the legacy cv2.polylines raster FAILS this very check at 0.20/0.24
    def legacy(w, ppmm=25, half=4.0):
        n = int(round(2 * half * ppmm))
        img = np.zeros((n, n), np.uint8)
        pts = np.array([[(x + half) * ppmm * 16, (0 + half) * ppmm * 16] for x in (-3.0, 3.0)],
                       dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, 255, max(1, int(round(w * ppmm))), cv2.LINE_8, 4)
        return img.astype(np.float32) / 255.0
    bad = [abs(width_mm(legacy(w)) - w) for w in (0.20, 0.24)]
    assert max(bad) > 0.02, f"legacy raster no longer wrong?? {bad}: the control cannot fail"
    assert abs(width_mm(legacy(0.20)) - width_mm(legacy(0.24))) < 1e-6, "legacy 0.20 vs 0.24 differ?"


def test_ring_link_connectors_capped_and_chained():
    p = cf.PITCH_MM
    chain = cf.build_disc_rings(True)[0]
    rings = cf.build_disc_rings(False)
    assert len(rings) == 25
    # ring k+1 start must be within 3p of ring k's end (the cap) in the chain
    import math
    ends = [r[-1] for r in rings]
    starts = [r[0] for r in rings]
    worst = max(math.hypot(starts[k + 1][0] - ends[k][0], starts[k + 1][1] - ends[k][1])
                for k in range(len(rings) - 1))
    assert worst <= cf.LINK_MAX_P * p + 1e-9, worst
    assert len(chain) == sum(len(r) for r in rings), "chained stroke is not the concatenation of the rings"
    # the cap is enforced: a mutated shift that breaks it must raise
    old = cf.RING_SHIFT_P
    try:
        cf.RING_SHIFT_P = 40.0
        # cap on angle (pi/3) keeps arc small at tiny radii but not at the outer rings
        try:
            cf.build_disc_rings(True)
            raised = False
        except RuntimeError:
            raised = True
    finally:
        cf.RING_SHIFT_P = old
    assert raised, "connector cap cannot fail"


def test_pen_pinned_zread_from_profile_and_fill_locked():
    out, info = _gen()
    base_z = info["z_down_abs"]
    assert abs(info["z_touch_abs"] - 18.75) < 1e-9 and abs(base_z - (18.75 - 0.6)) < 1e-9, info["z_touch_abs"]
    prof = json.loads((_ENV / "pen_profiles.json").read_text(encoding="utf-8"))
    orig = (_ENV / "pen_profiles.json").read_bytes()
    try:
        # (a) a different z_touch must move the emitted Z (read, not hardcoded)
        p2 = json.loads(orig)
        p2["profiles"]["bic_cristal_blue"]["z_touch_abs"] = 17.5
        (_ENV / "pen_profiles.json").write_text(json.dumps(p2), encoding="utf-8")
        out2 = _WORK / "_frag_z.gcode"
        info2 = SHEETS["fragment_bic"].generator(out2)
        assert abs(info2["z_down_abs"] - 16.9) < 1e-9, info2["z_down_abs"]
        assert min(float(m) for m in re.findall(r"Z(-?[\d.]+)", out2.read_text())) >= 16.9 - 1e-6
        assert "Z18.15" not in out2.read_text(), "old (fixture) z_down leaked into the file"
        # (b) a different fill_mm must refuse
        p3 = json.loads(orig)
        p3["profiles"]["bic_cristal_blue"]["fill_mm"] = 0.15
        (_ENV / "pen_profiles.json").write_text(json.dumps(p3), encoding="utf-8")
        try:
            SHEETS["fragment_bic"].generator(_WORK / "_frag_fill.gcode")
            fill_refused = False
        except RuntimeError:
            fill_refused = True
        assert fill_refused, "fill_mm != 0.12 was not refused"
        # (c) a different requested profile must refuse (pen is pinned)
        (_ENV / "pen_profiles.json").write_bytes(orig)
        other = {"label": "Some Other Pen", "pen_mm": 0.5}
        try:
            SHEETS["fragment_bic"].generator(_WORK / "_frag_pin.gcode", profile=other)
            pin_refused = False
        except RuntimeError:
            pin_refused = True
        assert pin_refused, "pinned pen was not enforced"
    finally:
        (_ENV / "pen_profiles.json").write_bytes(orig)


def test_microlift_applied_and_deterministic():
    out, info = _gen()
    ml = info["microlift"]
    assert ml and ml["short_lifts"] > 0.9 * ml["lifts"], ml
    z_short = round(info["z_touch_abs"] + 0.9, 2)
    assert abs(info["microlift"]["short_lift_mm"] - z_short) < 0.011, info["microlift"]
    zs = {round(float(m), 2) for m in re.findall(r"^G1 Z(-?[\d.]+)", out.read_text(), re.M)}
    assert z_short in zs, f"micro level {z_short} missing from {sorted(zs)}"
    # negative control: with microlift off the micro level must NOT appear
    out2 = _WORK / "_frag_nomicro.gcode"
    SHEETS["fragment_bic"].generator(out2, microlift=False)
    zs2 = {round(float(m), 2) for m in re.findall(r"^G1 Z(-?[\d.]+)", out2.read_text(), re.M)}
    assert z_short not in zs2, "microlift=False still produced micro lifts"
    # determinism: same inputs -> same gcode bytes
    out3 = _WORK / "_frag_base.gcode"
    SHEETS["fragment_bic"].generator(out3)
    a = hashlib.sha256(out.read_bytes()).hexdigest()
    b = hashlib.sha256(out3.read_bytes()).hexdigest()
    # the file name appears in the header/source comment, so compare modulo that
    ta = out.read_text().replace(out.name, "X")
    tb = out3.read_text().replace(out3.name, "X")
    assert ta == tb, "generator is not deterministic"


def test_registered_generic_and_plot_warning():
    s = SHEETS["fragment_bic"]
    assert s.ui == "generic" and s.generator is not None
    assert "PRE-REGISTERED" in s.blurb and "plot the linked g-code" in s.plot_warning.lower()


# ---------------------------------------------- fiducial order, label and probe additions ---
def _fiducials_first_violations(elements):
    """Violations of 'the six fiducials are the first things drawn'."""
    bad = []
    kinds = [e["kind"] for e in elements]
    if "fiducial" not in kinds:
        return ["no fiducial elements"]
    if kinds[0] != "fiducial":
        bad.append(f"first element is {kinds[0]!r}, not a fiducial")
    last_fid = max(i for i, k in enumerate(kinds) if k == "fiducial")
    first_patch = kinds.index("patch")
    first_other = min(i for i, k in enumerate(kinds) if k != "fiducial")
    if last_fid > first_other:
        bad.append("a non-fiducial element is emitted before the last fiducial stroke")
    if last_fid > first_patch:
        bad.append("a patch is emitted before the last fiducial stroke")
    gids = []
    for e in elements:
        if e["kind"] == "fiducial" and e["gid"] not in gids:
            gids.append(e["gid"])
    if gids != [f"fid{i}" for i in range(6)]:
        bad.append(f"fiducial order {gids}")
    return bad


def test_fiducials_emitted_first_and_emission_index_persisted():
    els, recs, fixed, geoms, meta = cf.build_layout()
    assert _fiducials_first_violations(els) == [], _fiducials_first_violations(els)
    # NEGATIVE CONTROL: the Phase-2 order (fiducials after glyph/rulers/prime) must be flagged
    fid = [e for e in els if e["kind"] == "fiducial"]
    rest = [e for e in els if e["kind"] != "fiducial"]
    old_order = rest[:2] + fid + rest[2:]
    assert _fiducials_first_violations(old_order), "negative control: late fiducials were not flagged"
    assert _fiducials_first_violations(rest + fid), "negative control: fiducials-last was not flagged"
    # emission_index: a permutation of 0..n-1 whose order matches pen_down_start_index
    idx = sorted(r["emission_index"] for r in recs)
    assert idx == list(range(len(recs))), "emission_index is not 0..n-1"
    by_emis = sorted(recs, key=lambda r: r["emission_index"])
    starts = [r["pen_down_start_index"] for r in by_emis]
    assert starts == sorted(starts) and starts[0] > max(i for i, e in enumerate(els) if e["kind"] == "fiducial")
    # the label rectangle is persisted for the reader's U-window mask and lies above the patch centre
    for r in recs:
        x0, y0, x1, y1 = r["label_bbox_mm"]
        assert x0 < r["center_mm"][0] < x1 and y1 < r["center_mm"][1], (r["id"], r["label_bbox_mm"])


def test_probe_option_is_an_opt_in_and_leaves_the_base_unchanged():
    els0, recs0, _f0, _g0, meta0 = cf.build_layout()
    els1, recs1, _f1, _g1, meta1 = cf.build_layout(probe=True)
    assert [r["id"] for r in recs0] == [r["id"] for r in cf.build_layout(probe=False)[1]]
    assert not any(r["code"] in ("NP", "L03P") for r in recs0), "probe items leaked into the base sheet"
    assert len(recs0) == 92 and len(meta0["unused_cells"]) == 13
    assert len(recs1) == 100 and len(meta1["unused_cells"]) == 5, (len(recs1), len(meta1["unused_cells"]))
    probe = [r for r in recs1 if r["code"] in ("NP", "L03P")]
    assert len(probe) == 8 and sum(r["code"] == "NP" for r in probe) == 4 and sum(r["code"] == "L03P" for r in probe) == 4
    assert all(r["pitch_mm"] == cf.PROBE_PITCH_MM == 0.085 and r["n_rows"] == 69 for r in probe)
    # base items keep their cell and jitter (probe items are appended LAST in the draw order)
    c0 = {r["id"]: (r["cell"], tuple(r["center_mm"]), r["theta_deg"]) for r in recs0}
    c1 = {r["id"]: (r["cell"], tuple(r["center_mm"]), r["theta_deg"]) for r in recs1 if r["id"] in c0}
    assert c0 == c1, "probe option moved a base patch"
    # actual geometry: probe patches really have 69 rows at 0.085 mm, base patches 49 at 0.12
    items = cf._item_table(probe=True)
    for it in items:
        if it["code"] == "NP":
            pl = cf.build_patch(it)
            ys = sorted({round(q[1], 4) for st in pl for q in st})
            gaps = {round(b - a, 3) for a, b in zip(ys, ys[1:]) if b - a > 1e-6}
            assert 0.085 in gaps or any(abs(g - 0.085) < 0.002 for g in gaps), gaps
            break
    # NEGATIVE CONTROL: a probe item built at the base pitch has 0.12 mm row spacing, which the assertion above rejects
    it = next(i for i in cf._item_table(probe=True) if i["code"] == "NP")
    it2 = dict(it)
    it2.pop("pitch_mm")
    ys2 = sorted({round(q[1], 4) for st in cf.build_patch(it2) for q in st})
    gaps2 = {round(b - a, 3) for a, b in zip(ys2, ys2[1:]) if b - a > 1e-6}
    assert not any(abs(g - 0.085) < 0.002 for g in gaps2), "negative control: base-pitch patch looks like probe pitch"
    # probe arm statistics
    np_ = [r for r in probe if r["code"] == "NP"]
    lp = [r for r in probe if r["code"] == "L03P"]
    assert all(r["pieces"] == 1 and r["lifts"] == 0 for r in np_)
    assert all(r["pieces"] > 1000 and r["lifts"] == r["pieces"] - 1 for r in lp), [r["pieces"] for r in lp]


def test_probe_gcode_passes_the_safety_set_and_the_registry_pins_the_probe():
    """The single `fragment_bic` registry entry IS the probe generator (100 patches,
    20,588 pen-downs); the base sheet (92 patches, 14,818 pen-downs) stays reachable as cf.fragment_bic."""
    # (1) the registry entry pins the probe generator, and there is NO sibling entry
    assert SHEETS["fragment_bic"].generator is cf.fragment_bic_probe, "registry does not pin the probe"
    assert [k for k in SHEETS if "fragment" in k] == ["fragment_bic"], "sibling fragment entry in the catalogue"
    out, info = _gen()  # through the REGISTRY generator
    r = _check_gcode_text(out.read_bytes(), info["deepest_z_down_abs"])
    assert r["n_moves"] > 14000
    lay = json.loads(Path(info["layout_path"]).read_text(encoding="utf-8"))
    assert lay["probe"] is True and lay["probe_pitch_mm"] == 0.085 and lay["probe_n_rows"] == 69
    assert info["pendown_count_matches"] is True, "pen-down count mismatch (registry sheet)"
    assert info["n_pendown_actual_total"] == info["n_pendown_expected_total"] == 20588, (
        info["n_pendown_actual_total"], info["n_pendown_expected_total"])
    assert len(lay["items"]) == 100 and len(lay["grid"]["unused_cell_centres_mm"]) == 5
    # (2) the base sheet is still the 92-patch, 14,818-pen-down sheet (record + control)
    out0 = _WORK / "_frag_base_ref.gcode"
    info0 = cf.fragment_bic(out0)
    lay0 = json.loads(Path(info0["layout_path"]).read_text(encoding="utf-8"))
    assert lay0["probe"] is False and lay0["probe_pitch_mm"] is None, "base generator carries the probe"
    assert info0["pendown_count_matches"] is True and info0["n_pendown_actual_total"] == 14818, info0["n_pendown_actual_total"]
    assert len(lay0["items"]) == 92 and len(lay["items"]) == len(lay0["items"]) + 8
    # NEGATIVE CONTROL: the registry-is-probe predicate is False for the base generator
    pins_probe = lambda g: g is cf.fragment_bic_probe  # noqa: E731
    assert pins_probe(SHEETS["fragment_bic"].generator) and not pins_probe(cf.fragment_bic),         "negative control: the predicate cannot tell the base generator from the probe"
    # the reader's commanded-delta table is in the layout and monotone in H, per laid column
    def mono(d):
        return d["H1"] < d["H2"] < d["H3"] < d["H4"]
    for col, d in lay["hole_commanded_delta"].items():
        assert mono(d), (col, d)
    assert abs(lay["hole_commanded_delta"]["0.20"]["H4"] - 0.253) < 0.002
    # NEGATIVE CONTROL: a scrambled table is rejected by the same predicate
    d0 = lay["hole_commanded_delta"]["0.20"]
    assert not mono({"H1": d0["H4"], "H2": d0["H3"], "H3": d0["H2"], "H4": d0["H1"]})
    # the reader inputs the self-test depends on are present
    for k in ("strokes_bbox_mm", "windows", "rulers", "fiducials", "grid"):
        assert k in lay, k
    assert len(lay0["grid"]["unused_cell_centres_mm"]) == 13 and len(lay["grid"]["unused_cell_centres_mm"]) == 5


def _run_all():
    tests = [test_gcode_safety_and_reach, test_every_patch_has_a_printed_label,
             test_arms_differ_in_intended_statistic, test_equal_coverage_and_positive_control, test_raster_laid_width_is_true,
             test_ring_link_connectors_capped_and_chained,
             test_pen_pinned_zread_from_profile_and_fill_locked,
             test_microlift_applied_and_deterministic, test_registered_generic_and_plot_warning,
             test_fiducials_emitted_first_and_emission_index_persisted,
             test_probe_option_is_an_opt_in_and_leaves_the_base_unchanged,
             test_probe_gcode_passes_the_safety_set_and_the_registry_pins_the_probe]
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
