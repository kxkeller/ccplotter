"""Structure-aware binary (M11 mode B) -- request key bw_structure_binary, DEFAULT ON since 2026-11-04 (OFF only on strict falsy), bwphoto only.

Could-not-fail doctrine: every acceptance has a rejection next to it.
  * algorithm: a sharp thin line survives; a blurred weak-edge blob in the candidate ink is dropped (positive control: the SAME blob drawn
    sharp is kept); a long persistent source contour with no ink nearby is added, a short one is not.
  * route (/api/process, Flask test client, never :5001): key absent == explicit true (ON, audit.structure_binary present); explicit
    false / 0 / "0" / "false" / "False" == the pinned pre-flip gcode with no record; junk (None, "", "no", ...) == ON;
    tracev3 / illustration are byte-identical whatever the key says.
  * UI: static checks of assets/templates/index.html (checkbox checked, tooltip) + the real body expression executed in node
    (explicit boolean in bwphoto, key absent elsewhere).
No os.chdir / env assignment at import time (D1935): the runtime home, work dir and cwd are set by a module fixture and undone after.
Run: pytest tests/test_structure_binary.py  (one file; combined pytest runs have known pre-existing cross-file contamination).
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")
HTML = (REPO / "assets" / "templates" / "index.html").read_text(encoding="utf-8")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from ccplotter import structure_binary as sb  # noqa: E402  (pure numpy/cv2 module: importing it touches no runtime state)


# --------------------------------------------------------------------------- algorithm
def _scene():
    """Source luminance + candidate ink K on a 237 x 160 mm page at 0.1 mm/px (2370 x 1600)."""
    H, W = 1600, 2370
    L = np.full((H, W), 235, np.uint8)
    L[300:1300, 200:204] = 20                               # sharp thin vertical line, 100 mm long, 0.4 mm wide
    blob = np.zeros((H, W), np.uint8); cv2.circle(blob, (1200, 800), 8, 255, -1)       # 1.6 mm diameter -> ~2 mm2
    L_blur = L.copy(); L_blur[blob > 0] = 20; L_blur = cv2.GaussianBlur(L_blur, (0, 0), 14)    # a diffuse smudge (sigma 1.4 mm)
    L_sharp = L.copy(); L_sharp[blob > 0] = 20                                                   # the same blob, hard-edged
    return L, L_blur, L_sharp, blob > 0


def _K(L, thresh=128):
    return L < thresh


def _bgr(L):
    return cv2.cvtColor(L, cv2.COLOR_GRAY2BGR)


def test_sharp_line_kept_blurred_blob_dropped_with_sharp_blob_as_positive_control():
    L, Lb, Ls, blob = _scene()
    # source and candidate ink for the blurred case: ink threshold on the blurred source gives a (smaller) blob; make K the blob disc itself
    K_line = np.zeros(L.shape, bool); K_line[300:1300, 198:206] = True
    Kb = K_line | blob
    Rb, cb = sb.structure_binary_raster(Kb, _bgr(Lb), 0.1)
    inked_b = Rb == 0
    assert inked_b[300:1300, 198:206].mean() > 0.9, "sharp thin line must survive"
    assert inked_b[790:810, 1190:1210].sum() == 0 and cb["components_dropped"] >= 1, "blurred weak-edge blob must be dropped"
    Rs, cs = sb.structure_binary_raster(Kb, _bgr(Ls), 0.1)
    assert (Rs == 0)[790:810, 1190:1210].sum() > 0.5 * blob.sum() and cs["components_dropped"] == 0, "positive control: the same blob drawn SHARP is kept"


def test_long_persistent_source_contour_with_no_ink_is_added_short_one_is_not():
    H, W = 1600, 2370
    L = np.full((H, W), 235, np.uint8)
    L[:, 1000:] = 150                                       # a soft but persistent tonal step, 160 mm long (>= 8 mm), well above the gradient floor
    L = cv2.GaussianBlur(L, (0, 0), 6)
    Kempty = np.zeros((H, W), bool)
    R, c = sb.structure_binary_raster(Kempty, _bgr(L), 0.1)
    assert c["ink_px_added"] > 0 and (R == 0)[:, 990:1010].any(), "long persistent contour must be added where K has no ink"
    # null: a uniform image has no contour
    R0, c0 = sb.structure_binary_raster(Kempty, _bgr(np.full((H, W), 200, np.uint8)), 0.1)
    assert c0["ink_px_added"] == 0 and not (R0 == 0).any()
    # short contour: a 1.2 mm square patch (perimeter 4.8 mm < the 8 mm floor) adds nothing; the SAME edge profile on a 40 mm square (positive control) does
    def patch(n):
        Lp = np.full((H, W), 235, np.uint8); Lp[800:800 + n, 1000:1000 + n] = 60
        return cv2.GaussianBlur(Lp, (0, 0), 1.5)
    _, c_small = sb.structure_binary_raster(Kempty, _bgr(patch(12)), 0.1)
    _, c_big = sb.structure_binary_raster(Kempty, _bgr(patch(400)), 0.1)
    assert c_small["ink_px_added"] == 0, "contour shorter than 8 mm must not be added"
    assert c_big["ink_px_added"] > 0, "positive control: a long contour with the same edge profile is added"


def test_deterministic_and_output_is_strict_binary():
    L, Lb, Ls, blob = _scene()
    K = _K(Ls)
    a, _ = sb.structure_binary_raster(K, _bgr(Ls), 0.1)
    b, _ = sb.structure_binary_raster(K, _bgr(Ls), 0.1)
    assert a.dtype == np.uint8 and set(np.unique(a).tolist()) <= {0, 255} and np.array_equal(a, b)


def test_params_are_the_pinned_global_set():
    assert sb.PARAMS["diffuse_area_mm2"] == 3.0 and sb.PARAMS["diffuse_E_max"] == 0.30 and sb.PARAMS["diffuse_sharp_max"] == 0.35
    assert sb.PARAMS["line_min_len_mm"] == 8.0 and sb.PARAMS["line_width_mm"] == 0.30 and sb.PARAMS["line_add_dist_mm"] == 2.5
    assert sb.PARAMS["analysis_mm_per_px"] == 0.09 and sb.PARAMS["coh_min"] == 0.55 and sb.PARAMS["min_component_mm2"] == 0.035


def test_mm_per_px_matches_trace_v3_derivation():
    img = np.zeros((1000, 2000), np.uint8)
    o = {"fill_mm": 0.2, "paper_w_mm": 257.0, "margin_mm": 10.0}      # fill_px = 0.2 / 0.1185 = 1.69 (above the 1.2 floor)
    assert sb.mm_per_px_of(img, o) == pytest.approx(237.0 / 2000)
    assert sb.mm_per_px_of(img, {**o, "fill_px": 2.0}) == pytest.approx(0.1)
    assert sb.mm_per_px_of(img, {**o, "_mm_per_px_override": 0.077}) == 0.077
    assert sb.mm_per_px_of(img, {**o, "fill_px": 0.5}) == pytest.approx(0.2 / 1.2), "1.2 px floor, as trace_v3"


# --------------------------------------------------------------------------- route
@pytest.fixture(scope="module")
def route_env():
    """Isolated runtime home + work dir + cwd for the Flask-route tests; undone at module teardown (never at import)."""
    assert PHOTO.exists(), "tracked fixture photo missing"
    mp = pytest.MonkeyPatch()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _test_home
    home = _test_home.use_temp_home("ccplotter_test_home_sb_")           # sets CCPLOTTER_HOME (assignment)
    mp.setenv("CCPLOTTER_HOME", str(home))
    mp.setenv("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_sb_"))
    mp.chdir(REPO)
    from ccplotter.routes.core import app
    from ccplotter.config import WORK_DIR
    yield app, Path(WORK_DIR)
    mp.undo()
    shutil.rmtree(home, ignore_errors=True)


PHOTO = REPO / "assets" / "Model Images" / "rfr-bigboy-03.jpg"      # git-tracked benchmark photo (steam locomotive); ON drops 85 components on it


def _photo(path):
    """The tracked benchmark photo, 1100 px long side. A synthetic scene did not exercise the mode (nothing dropped through the XDoG front end)."""
    im = cv2.imread(str(PHOTO))
    s = 1100 / max(im.shape[:2])
    cv2.imwrite(str(path), cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))


def _run(env, did, extra, style="bwphoto", crop=None):
    app, WORK_DIR = env
    _photo(WORK_DIR / f"src_{did}.png")
    resp = app.test_client().post("/api/process", json={
        "job_type": "gcode", "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0.0, "dy": 0.0, "crop": crop}],
        "style": style, "resolution": 1500, "threshold_mode": "otsu", "connector_distance": 0.5,
        "paper": "120x120", "margin": 10.0, "debug": True, **extra})
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    g = (WORK_DIR / body["gcode_file"]).read_text(encoding="ascii")
    lines = [l for l in g.splitlines() if not l.startswith(";")]
    infos = sorted(WORK_DIR.glob(f"*{did}*.svg.info.json"), key=lambda p: p.stat().st_mtime)
    audit = json.loads(infos[-1].read_text()).get("audit", {}) if infos else {}
    return body, hashlib.sha256("\n".join(lines).encode()).hexdigest(), audit


OFF_VARIANTS = (False, 0, "0", "false", "False")           # the strict-falsy set (routes/core.py); everything else is ON
JUNK_VARIANTS = (None, "", "no", "off", "FALSE", "null", 2, [], {})


def test_route_absent_equals_explicit_true_and_records_provenance(route_env):
    _, h_abs, a_abs = _run(route_env, "sb_abs", {})
    _, h_on, a_on = _run(route_env, "sb_exp_on", {"bw_structure_binary": True})
    assert h_abs == h_on, "absent must be ON (identical gcode to explicit true)"
    rec = a_abs.get("structure_binary")
    assert rec and rec["version"] == sb.VERSION and rec["params"] == sb.PARAMS and rec["mm_per_px"] > 0
    assert 0.0 < rec["R_ink_frac"] <= 1.0 and "R_sha16" in rec and rec["components_dropped"] > 0 and rec["R_ink_frac"] < rec["K_ink_frac"]
    assert rec["ink_px_dropped"] > 0
    assert a_on.get("structure_binary", {}).get("R_sha16") == rec["R_sha16"]


# gcode sha (non-comment lines) of this fixture run with the mode OFF, captured at HEAD cd919f7f BEFORE the default flip (key absent == OFF then).
# Explicit false after the flip must reproduce it exactly (artifacts/structure-binary-default-2026-11-04/RESULT.md).
PRE_FLIP_PIN = "b40148b467cb5f276e94c0ac6ab273d8ff61307e24ca217c134db074e31a0fed"


def test_route_explicit_false_equals_pre_flip_output_and_differs_from_on(route_env):
    _, h_off, a_off = _run(route_env, "sb_exp_off", {"bw_structure_binary": False})
    assert "structure_binary" not in a_off
    assert h_off == PRE_FLIP_PIN, "explicit false must reproduce the pre-flip (mode OFF) gcode"
    _, h_on, _ = _run(route_env, "sb_cmp_on", {})
    assert h_on != h_off, "positive control: ON must change the generated job (else deleting the mechanism would pass)"
    _, h_on2, _ = _run(route_env, "sb_cmp_on2", {"bw_structure_binary": True})
    assert h_on2 == h_on, "idempotent determinism of the ON path"


def test_route_strict_falsy_variants_are_off_and_junk_is_on(route_env):
    for i, v in enumerate(OFF_VARIANTS):
        _, h, a = _run(route_env, f"sb_off{i}", {"bw_structure_binary": v})
        assert h == PRE_FLIP_PIN and "structure_binary" not in a, repr(v)
    _, h_on, _ = _run(route_env, "sb_junk_ref", {})
    for i, v in enumerate(JUNK_VARIANTS):
        _, h, a = _run(route_env, f"sb_junk{i}", {"bw_structure_binary": v})
        assert h == h_on and "structure_binary" in a, repr(v)


def test_route_on_with_item_crop(route_env):
    crop = [0.1, 0.1, 0.9, 0.9]
    _, h0, a0 = _run(route_env, "sb_crop_off", {"bw_structure_binary": False}, crop=crop)
    _, h1, a1 = _run(route_env, "sb_crop_on", {}, crop=crop)   # _run asserts HTTP 200; absent == ON
    rec = a1.get("structure_binary")
    assert rec and rec["components_dropped"] > 0 and "structure_binary" not in a0
    assert h1 != h0, "ON with a crop must change the job"
    _, h_full, _ = _run(route_env, "sb_nocrop_on", {})
    assert h_full != h1, "positive control: the crop is honoured (cropped ON differs from uncropped ON)"


def test_debug_dumps_the_structure_binary_png(route_env):
    _, WORK_DIR = route_env
    _run(route_env, "sb_dbg", {"bw_structure_binary": True})            # debug=True in _run
    pngs = list(WORK_DIR.glob("*_structure_binary.png"))
    assert pngs and set(np.unique(cv2.imread(str(pngs[0]), 0)).tolist()) <= {0, 255}


def test_tracev3_and_illustration_are_byte_identical_whatever_the_key_says(route_env):
    for style in ("tracev3", "illustration"):
        _, h0, a0 = _run(route_env, f"sb_{style}_abs", {}, style=style)
        assert "structure_binary" not in a0, style
        for i, v in enumerate((True, False, "true", "0")):
            _, h1, a1 = _run(route_env, f"sb_{style}_{i}", {"bw_structure_binary": v}, style=style)
            assert h1 == h0 and "structure_binary" not in a1, (style, v)


# --------------------------------------------------------------------------- UI
def test_checkbox_present_checked_hidden_with_label_and_tooltip():
    m = re.search(r'<input type="checkbox" id="structure-binary"([^>]*)>', HTML)
    assert m and re.search(r"\bchecked\b", m.group(1)), "ON by default: the checkbox is checked"
    assert "Structure-aware binary (experimental)" not in HTML and "Structure-aware binary" in HTML
    lab = re.search(r'<label[^>]*title="([^"]*)"[^>]*>\s*<input type="checkbox" id="structure-binary"', HTML)
    assert lab and lab.group(1).startswith("On by default:") and "Uncheck to keep every speck" in lab.group(1) and "B/W photo mode only" in lab.group(1)
    assert "off by default" not in lab.group(1).lower()
    assert re.search(r'<div class="field" id="structure-binary-field" style="display:none">', HTML)
    assert "structure-binary" not in "".join(re.findall(r"localStorage\.(?:setItem|getItem)\('([^']+)'", HTML)), "not persisted"
    assert re.search(r"\.\.\.\(structureBinaryStyleOk\(\) \? \{ bw_structure_binary: structureBinaryActive\(\) \} : \{\}\)", HTML), "explicit boolean in bwphoto"
    assert len(re.findall(r"bw_structure_binary\s*:", HTML)) == 1, "the key must be built in exactly one place"


def _sent(style, checked):
    """Run the real body-building expression in node; returns the value of bw_structure_binary, or "ABSENT" when the key is not sent."""
    fns = "".join(re.search(r"function %s\(\).*?\n" % n, HTML).group(0) for n in ("structureBinaryStyleOk", "structureBinaryActive"))
    expr = re.search(r"(\.\.\.\(structureBinaryStyleOk\(\).*?\{\}\))", HTML).group(1)
    js = (fns + "\nconst V = " + json.dumps({"img-style": {"value": style}, "structure-binary": {"checked": checked}}) +
          ";\nfunction ge(id) { return V[id]; }\nconst body = {" + expr + "};\nprocess.stdout.write(JSON.stringify(body));")
    out = json.loads(subprocess.run([NODE, "-e", js], capture_output=True, text=True, check=True).stdout)
    return out.get("bw_structure_binary", "ABSENT")


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_explicit_boolean_sent_in_bwphoto_only():
    assert _sent("bwphoto", True) is True, "checked -> explicit true"
    assert _sent("bwphoto", False) is False, "unchecked -> explicit false (absent would mean ON)"
    for st in ("tracev3", "illustration", "twopenkt", "plotready", "tsp", "sketchfield", "photo512"):
        assert _sent(st, True) == "ABSENT" and _sent(st, False) == "ABSENT", st
