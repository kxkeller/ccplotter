"""Two-pen "Black Line + Blue Tone" (key + tint): POST /api/twopen_keytint, ccplotter/twopen_keytint.py.

Same conventions as tests/test_twopen.py: explicit temp runtime home BEFORE any ccplotter import, temp WORK_DIR,
Flask test client (no server, never :5001). Every null is paired with a positive control:
  * pens differ in EVERY z field, and the checker is shown to FAIL on crossed profiles;
  * the black-pass machinery is checked against a plain bwphoto job (strokes identical, max |dxy| ~ 0) AND the
    checker is shown to fail on a different mask;
  * "no tint beyond the plain ink mask" at blue_amount 0 is paired with a large tint at the default;
  * blue ink is monotone in blue_amount, shown on three images' worth of rungs (mask area) and on traced draw mm.
Each plain bwphoto job takes ~25 s on the pagoda fixture; the module runs three of them.
"""
import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home; pen/calibration never resolve from CWD
_HOME = _test_home.use_temp_home("ccplotter_test_home_keytint_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_keytint_"))
sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import pytest

import ccplotter.routes.runs  # noqa: F401
from ccplotter import twopen_keytint as kt
from ccplotter.cal import load_pen_profiles, save_pen_profiles
from ccplotter.config import FAN_OFF_GCODE, WORK_DIR
from ccplotter.runs import _runs_load

IMG = REPO / "tests" / "fixtures" / "twotone" / "pagoda.jpg"
PAPER, MARGIN = "150x110", 5.0
BLUE, BLACK = "bic_cristal_blue", "bic_cristal_black"
BLUE_Z = {"z_touch_abs": 18.75, "z_up": 3.2, "z_down": -0.9, "z_micro": 0.8, "draw_speed": 2500, "plunge_speed": 450}
BLACK_Z = {"z_touch_abs": 17.35, "z_up": 3.7, "z_down": -0.4, "z_micro": 0.65, "draw_speed": 3500, "plunge_speed": 550}


def _client():
    from ccplotter.routes.core import app
    return app.test_client()


@pytest.fixture(scope="module", autouse=True)
def _distinct_profiles():
    """Make the two default pens differ in EVERY z field (temp home only), so a crossed profile fails."""
    d = load_pen_profiles()
    d["profiles"][BLUE].update(BLUE_Z)
    d["profiles"][BLACK].update(BLACK_Z)
    save_pen_profiles(d)
    return d


def _post(**extra):
    body = {"image": str(IMG), "paper": PAPER, "margin": MARGIN}
    body.update(extra)
    return _client().post("/api/twopen_keytint", json=body)


@pytest.fixture(scope="module")
def job():
    r = _post(debug=True)
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    files = {g["color"]: WORK_DIR / g["file"] for g in j["gcode_files"]}
    return {"resp": j, "files": files, "text": {k: v.read_text(encoding="utf-8") for k, v in files.items()},
            "rec": next(x for x in _runs_load() if x["id"] == j["run_id"])}


@pytest.fixture(scope="module")
def prep(tmp_path_factory):
    wd = tmp_path_factory.mktemp("kt_prep")
    return kt.capture_plain_bwphoto(IMG, wd, paper_w=150.0, paper_h=110.0, margin=MARGIN, pen_id=BLACK,
                                    paper_spec=PAPER)


def _g1_z(text):
    return [float(m.group(1)) for m in re.finditer(r"^G1 Z(-?\d+(?:\.\d+)?)", text, re.M)]


def _check_z(text, prof):
    zt = prof["z_touch_abs"]
    zs = _g1_z(text)
    down, short = round(zt + prof["z_down"], 2), round(zt + prof["z_micro"], 2)
    assert down in zs and short in zs, (down, short, sorted(set(zs)))
    # (z_up_abs itself is legitimate in a real job: microlift uses it for medium hops, so it is not asserted absent)
    assert re.search(rf"^G1 Z{down:.2f} F{prof['plunge_speed']:.0f}$", text, re.M)
    assert re.search(rf"^G1 X[\d.]+ Y[\d.]+ F{prof['draw_speed']:.0f}$", text, re.M)
    assert min(zs) >= down - 1e-6


# ------------------------------------------------------------------ route: files, preview, safety
def test_route_200_two_downloadable_files_and_a_preview(job):
    j = job["resp"]
    assert j["ok"] and [g["color"] for g in j["gcode_files"]] == ["blue", "black"]      # pass 1 blue, pass 2 black
    assert [g["pass"] for g in j["gcode_files"]] == [1, 2]
    for p in job["files"].values():
        assert p.is_file() and p.stat().st_size > 500
    # downloadable through the app's /work route, the way the UI fetches them
    for g in j["gcode_files"]:
        r = _client().get(f"/work/{g['file']}")
        assert r.status_code == 200 and len(r.data) > 500
    svg = (WORK_DIR / j["preview_file"]).read_text(encoding="utf-8").lower()
    assert "#1f4fd8" in svg and "#16181d" in svg and svg.index("#1f4fd8") < svg.index("#16181d")   # blue under black
    png = cv2.imread(str(WORK_DIR / j["preview_png"]))
    assert png is not None and png.shape[1] == int(round(150 * kt.PREVIEW_PPM))
    px = png.reshape(-1, 3).astype(int)
    is_blue = ((px[:, 0] > 150) & (px[:, 2] < 120)).sum()           # BGR: blue ink has high B, low R
    is_dark = (px.sum(1) < 200).sum()
    assert is_blue > 100 and is_dark > 100, "the PNG preview must contain both inks"
    r = _client().get(f"/work/{j['preview_png']}")
    assert r.status_code == 200 and r.data[:4] == b"\x89PNG"


BAD_FAN = re.compile(r"\bM106\s+P[01]\b", re.I)
BAD_PAUSE = re.compile(r"^\s*(M0|M1|M600|M25|M226)\b", re.I | re.M)


def test_ascii_no_g92_valid_fans_no_pause_on_both(job):
    for color, p in job["files"].items():
        text = p.read_bytes().decode("ascii")                        # raises on any non-ASCII byte
        code = "\n".join(l.split(";")[0] for l in text.splitlines())
        assert not re.search(r"\bG92\b", code), color
        assert not BAD_FAN.search(text), color
        assert not BAD_PAUSE.search(text), color
        for ln in text.splitlines():
            if ln.startswith("M106"):
                assert ln.split(";")[0].strip() in {"M106 S0", "M106 P2 S0", "M106 P3 S0", *FAN_OFF_GCODE} or \
                    re.fullmatch(r"M106( P[23])?( S\d+)?", ln.split(";")[0].strip()), ln
        ls = [l.strip() for l in text.splitlines()]
        for must in ("G21", "G90", "M117 Plotting...", "M400", "M107", "M84", "M117 Plot complete"):
            assert must in ls, (color, must)
        first_draw = next(i for i, l in enumerate(ls) if l.startswith("G1 X") and " F" in l)
        post = max(i for i, l in enumerate(ls) if l == "M107")
        assert [l for l in ls[first_draw:post] if l.startswith(("M106", "M107"))] == []   # fan-off is postamble only
    # the scanner can fail: a poisoned sample trips it
    assert BAD_FAN.search("M106 P0 S0") and BAD_PAUSE.search("M600\n") and re.search(r"\bG92\b", "G92 X0")


# ------------------------------------------------------------------ per-pass pen ids and live z
def test_headers_carry_each_passs_pen_id_and_live_z_touch(job):
    prof = load_pen_profiles()["profiles"]
    for color, pid, z in (("blue", BLUE, 18.75), ("black", BLACK, 17.35)):
        text = job["text"][color]
        assert f"; PEN this_pass pen_id={pid} z_touch_abs={z:.2f}" in text, color
        assert prof[pid]["z_touch_abs"] == z
        _check_z(text, prof[pid])
    assert "; PASS 1 OF 2: BLUE" in job["text"]["blue"] and "; PASS 2 OF 2: BLACK" in job["text"]["black"]
    assert job["files"]["blue"].name.endswith(f"_1_blue_{BLUE}.gcode") and job["files"]["black"].name.endswith(f"_2_black_{BLACK}.gcode")


def test_crossed_profile_would_fail_positive_control(job):
    prof = load_pen_profiles()["profiles"]
    with pytest.raises(AssertionError):
        _check_z(job["text"]["blue"], prof[BLACK])
    with pytest.raises(AssertionError):
        _check_z(job["text"]["black"], prof[BLUE])
    assert "z_touch_abs=17.35" not in job["text"]["blue"] and "z_touch_abs=18.75" not in job["text"]["black"]


@pytest.fixture(scope="module")
def job_live_z():
    """Second route job: black z_touch changed in the profile store, image sent as image_b64, debug OFF, blue_amount 0.5."""
    import base64
    d = load_pen_profiles()
    old = d["profiles"][BLACK]["z_touch_abs"]
    d["profiles"][BLACK]["z_touch_abs"] = 16.10
    save_pen_profiles(d)
    try:
        r = _client().post("/api/twopen_keytint", json={"image_b64": base64.b64encode(IMG.read_bytes()).decode(),
                                                        "paper": PAPER, "margin": MARGIN, "blue_amount": 0.5})
    finally:
        d["profiles"][BLACK]["z_touch_abs"] = old
        save_pen_profiles(d)
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def test_z_is_read_live_per_request_not_cached(job, job_live_z):
    """Change the black pen's z_touch in the profile store, post again: the NEW z lands (nothing is a constant)."""
    g = {x["color"]: x for x in job_live_z["gcode_files"]}
    assert g["black"]["z_touch_abs"] == 16.10 and g["blue"]["z_touch_abs"] == 18.75
    assert "z_touch_abs=16.10" in (WORK_DIR / g["black"]["file"]).read_text(encoding="utf-8")
    assert "z_touch_abs=17.35" in job["text"]["black"]                      # the earlier job kept its own
    # more tint at 0.5 than at the default (route-level monotone check, same image)
    assert job_live_z["keytint"]["layer_stats"]["blue"]["draw_mm"] > job["resp"]["keytint"]["layer_stats"]["blue"]["draw_mm"]


# ------------------------------------------------------------------ black pass vs a plain bwphoto job
def _strokes(svg):
    return kt.load_strokes_mm(svg)


def _max_dxy(a, b):
    assert len(a) == len(b), (len(a), len(b))
    return max((float(np.abs(x - y).max()) if x.shape == y.shape else 1e9) for x, y in zip(a, b))


def test_captured_job_equals_an_independent_plain_bwphoto_job(prep):
    """The debug-mode capture run is a normal bwphoto job: a separate, non-debug request through the same route gives
    the same routed strokes."""
    c = _client()
    with open(IMG, "rb") as f:
        did = c.post("/api/add_drawing", data={"file": (f, IMG.name)}, content_type="multipart/form-data").get_json()["id"]
    r = c.post("/api/process", json={"style": "bwphoto", "job_type": "preview", "paper": PAPER, "margin": MARGIN,
                                     "pen_id": BLACK, "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}]})
    assert r.status_code == 200, r.get_json()
    independent = _strokes(WORK_DIR / "layout.routed.svg")
    assert _max_dxy(independent, _strokes(prep["plain_routed_svg"])) == 0.0


def test_machinery_reproduces_plain_job_strokes_positive_control(prep, tmp_path):
    """Tracing the plain job's OWN mask (M0) through layer_from_mask() gives the plain job's routed strokes exactly."""
    sep = kt.separate(prep["gray"], prep["opts"], prep["mm_per_px"], kt.BLUE_AMOUNT_DEFAULT)
    geom = kt.layer_geom(prep)
    routed = kt.layer_from_mask(sep["M0"], prep["opts"], geom, tmp_path, "m0")
    plain = _strokes(prep["plain_routed_svg"])
    assert len(plain) > 500
    assert _max_dxy(_strokes(routed), plain) <= 1e-6
    # the comparison can fail: a different mask does not match
    other = kt.layer_from_mask(sep["black"], prep["opts"], geom, tmp_path, "blk")
    assert len(_strokes(other)) != len(plain) or _max_dxy(_strokes(other), plain) > 1e-3


def test_black_pass_is_not_the_plain_job_by_design_and_says_why(prep, job):
    """The recipe's black layer is core|E|strip, not all of M0 (the mid-tone share goes to blue). Quantify it."""
    sep = kt.separate(prep["gray"], prep["opts"], prep["mm_per_px"], kt.BLUE_AMOUNT_DEFAULT)
    assert not np.array_equal(sep["black"], sep["M0"])
    plain_n = len(_strokes(prep["plain_routed_svg"]))
    assert job["resp"]["keytint"]["layer_stats"]["black"]["strokes"] != plain_n
    assert job["resp"]["keytint"]["separation"]["plain_job"]["stroke_count"] == plain_n
    # route-level: the black gcode pass carries exactly the black layer's strokes
    black_pass = next(p for p in job["resp"]["passes"] if p["color"] == "black")
    assert black_pass["strokes"] == job["resp"]["keytint"]["layer_stats"]["black"]["strokes"]


def test_strip_constant_is_named_and_switches_the_black_mask(prep):
    assert kt.STRIP_IN_BLACK is True
    on = kt.separate(prep["gray"], prep["opts"], prep["mm_per_px"], 0.375, strip_in_black=True)
    off = kt.separate(prep["gray"], prep["opts"], prep["mm_per_px"], 0.375, strip_in_black=False)
    assert on["info"]["black_mm2"] > off["info"]["black_mm2"]                  # the strip adds black ...
    assert on["info"]["blue_mm2"] > off["info"]["blue_mm2"]  # ... and blue now runs up to black's edge (no clearance ring is left undrawn)


# ------------------------------------------------------------------ blue_amount: null control + monotone
def test_blue_amount_zero_adds_no_tint_beyond_the_plain_ink_mask_null_control(prep):
    """At a=0 the blue threshold IS the Otsu cut, so blue lies inside the plain ink mask M0 (exp4 rung a=0 is the
    mid-tone SHARE of M0, not an empty layer). The null is therefore 'no tint outside M0'; the positive control is
    the default amount, which tints a large area outside M0."""
    def outside(a):
        s = kt.separate(prep["gray"], prep["opts"], prep["mm_per_px"], a)
        m0d = cv2.dilate(s["M0"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0     # 1 px tolerance for the 3x3 blur
        return float((s["blue"] & ~m0d).sum() * prep["mm_per_px"] ** 2), s["info"]["blue_mm2"]
    out0, blue0 = outside(0.0)
    out_d, blue_d = outside(kt.BLUE_AMOUNT_DEFAULT)
    assert out0 < 0.02 * max(blue_d, 1.0), (out0, blue_d)
    assert out_d > 20 * max(out0, 1e-6) and out_d > 10.0, (out0, out_d)
    assert blue0 < blue_d


def test_blue_ink_is_monotone_in_blue_amount(prep, tmp_path):
    areas = [kt.separate(prep["gray"], prep["opts"], prep["mm_per_px"], a)["info"]["blue_mm2"] for a in (0.0, 0.125, 0.25, 0.375, 0.5)]
    assert all(x < y for x, y in zip(areas, areas[1:])), areas
    geom = kt.layer_geom(prep)
    draw = []
    for a in (0.0, 0.25, 0.5):
        s = kt.separate(prep["gray"], prep["opts"], prep["mm_per_px"], a)
        draw.append(kt.stroke_stats(kt.load_strokes_mm(kt.layer_from_mask(s["blue"], prep["opts"], geom, tmp_path, f"b{a}")))["draw_mm"])
    assert draw[0] < draw[1] < draw[2], draw


def test_blue_amount_clamped_default_and_refusals(job):
    assert job["resp"]["keytint"]["blue_amount"] == kt.BLUE_AMOUNT_DEFAULT == 0.375
    assert kt.clamp_blue_amount(9) == kt.BLUE_AMOUNT_MAX == 0.5 and kt.clamp_blue_amount(-3) == 0.0
    assert kt.clamp_blue_amount(None) == 0.375 and kt.clamp_blue_amount("0.25") == 0.25
    for bad in ("abc", True):
        with pytest.raises(kt.KeytintError):
            kt.clamp_blue_amount(bad)
    assert _post(blue_amount="abc").status_code == 400


# ------------------------------------------------------------------ run record, provenance, debug outputs, refusals
def test_run_record_and_provenance_per_pass(job):
    rec, rid, j = job["rec"], job["resp"]["run_id"], job["resp"]
    assert rec["type"] == "twopen_keytint" and {"gcode_blue", "gcode_black", "preview", "preview_png"} <= set(rec["files"])
    assert rec["keytint"]["blue_amount"] == 0.375 and rec["keytint"]["pass_order"] == ["blue", "black"]
    assert [p["pen_id"] for p in rec["passes"]] == [BLUE, BLACK]
    prov = json.loads((_HOME / rec["provenance_file"]).read_text(encoding="utf-8")) if (_HOME / rec["provenance_file"]).exists() \
        else json.loads((Path(os.environ["CCPLOTTER_HOME"]) / rec["provenance_file"]).read_text(encoding="utf-8"))
    res = prov["resolved"]
    assert res["keytint"]["separation"]["constants"]["strip_in_black"] is True
    assert [p["pen_id"] for p in res["passes"]] == [BLUE, BLACK]
    assert {p["z_touch_abs"] for p in res["passes"]} == {18.75, 17.35}
    for key in ("gcode_blue", "gcode_black"):
        assert "provenance=" in (WORK_DIR / rec["files"][key]).read_text(encoding="utf-8")


def test_debug_pngs_written_only_when_debug_on(job, job_live_z):
    dbg = job["resp"]["keytint"]["debug_files"]
    assert {"mask_M0", "mask_black", "mask_blue", "layer_blue", "layer_black"} <= set(dbg)
    for p in dbg.values():
        assert Path(p).is_file() and cv2.imread(p) is not None
    m = cv2.imread(dbg["mask_blue"], 0)
    assert 0 < (m > 0).mean() < 0.5
    assert job_live_z["keytint"]["debug_files"] == {}                      # that job ran with debug off


def test_each_pass_uses_its_own_pen_and_defaults(job):
    g = {x["color"]: x for x in job["resp"]["gcode_files"]}
    assert g["blue"]["pen_id"] == BLUE and g["black"]["pen_id"] == BLACK
    st = job["resp"]["keytint"]["layer_stats"]
    for c in ("blue", "black"):
        assert st[c]["strokes"] > 50 and st[c]["draw_mm"] > 100 and st[c]["pen_lifts"] == st[c]["strokes"]
        assert g[c]["time_est"]["model"] == "b" and "not stopwatch-validated" in g[c]["time_est"]["label"]


@pytest.mark.parametrize("patch", [
    {"blue_pen_id": "no_such_pen"}, {"black_pen_id": "no_such_pen"}, {"black_pen_id": BLUE},
    {"image": "C:/definitely/not/here.jpg"}, {"image": None}, {"paper": "banana"}, {"margin": 90},
    {"offset_x": 900.0}])
def test_refusals_are_4xx_never_500(patch):
    before = len(_runs_load())
    body = {"image": str(IMG), "paper": PAPER, "margin": MARGIN}
    body.update(patch)
    if body.get("image") is None:
        body.pop("image")
    r = _client().post("/api/twopen_keytint", json=body)
    assert 400 <= r.status_code < 500, (patch, r.status_code, r.get_json())
    assert r.get_json().get("error")
    assert len(_runs_load()) == before


def test_existing_twopen_route_default_behaviour_unchanged():
    """/api/twopen still takes two ready SVG layers and returns the original response shape (no keytint keys)."""
    svg = lambda body: (f'<svg xmlns="http://www.w3.org/2000/svg" width="120mm" height="120mm" viewBox="0 0 120 120" '
                        f'fill="none" stroke="black" stroke-width="0.3">{body}</svg>')
    blue = svg("".join(f'<line x1="25" y1="{y}" x2="95" y2="{y}"/>' for y in range(30, 91, 4)))
    black = svg('<rect x="25" y="30" width="70" height="60"/>')
    r = _client().post("/api/twopen", json={"blue_svg": blue, "black_svg": black, "blue_pen_id": BLUE,
                                            "black_pen_id": "energel_05", "paper": "120x120"})
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert "keytint" not in j and "preview_png" not in j and [g["color"] for g in j["gcode_files"]] == ["blue", "black"]
    rec = next(x for x in _runs_load() if x["id"] == j["run_id"])
    assert rec["type"] == "twopen" and "keytint" not in rec and "preview_png" not in rec["files"]


# ------------------------------------------------------------------ UI hookup (assets/templates/index.html)
INDEX = REPO / "assets" / "templates" / "index.html"


def _index_scripts():
    import re as _re
    return _re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", INDEX.read_text(encoding="utf-8"), _re.S)


def _node_check(js, tmp_path):
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    f = tmp_path / "x.js"
    f.write_text(js, encoding="utf-8")
    return subprocess.run([node, "--check", str(f)], capture_output=True, text=True)


def test_ui_box_ids_and_handler_body(tmp_path):
    html = INDEX.read_text(encoding="utf-8")
    for i in ("keytint-box", "keytint-blue-amount", "keytint-btn", "keytint-result"):
        assert f'id="{i}"' in html, i
    assert 'value="0.375" min="0" max="0.5" step="0.025"' in html
    assert "Generate Black Line + Blue Tone (two files)" in html and 'onclick="genKeyTint()"' in html
    assert "Blue amount 0 still draws the mid-tone share in blue; it is not no blue" in html
    js = "\n".join(_index_scripts())
    a = js.index("async function genKeyTint()")
    handler = js[a:js.index("async function genDepositionSheet()")]
    assert "buildBody('preview')" in handler                      # reuses the normal body builder, no copy
    assert "style: 'bwphoto'" in handler and "blue_amount" in handler and "/api/twopen_keytint" in handler
    assert "state.drawings.length" in handler and "calSheet" in handler and "toast(" in handler
    assert "Check the active pen profile before plotting EACH pass" in handler
    assert "<img" in handler and "j.preview_png" in handler
    # no apostrophe may sit inside a word of the new code (it would end a single-quoted JS string early)
    import re as _re
    assert not _re.search(r"\w'\w", handler), _re.search(r"\w'\w", handler).group(0)
    assert _re.search(r"\w'\w", "it's") and "Blue amount 0 still" in html      # control: the pattern can fire


def test_ui_script_passes_node_check_and_the_check_can_fail(tmp_path):
    scripts = _index_scripts()
    assert scripts
    for k, js in enumerate(scripts):
        r = _node_check(js, tmp_path)
        assert r.returncode == 0, r.stderr[:500]
    bad = _node_check("const s = 'it's broken';", tmp_path)          # positive control: an apostrophe in a string
    assert bad.returncode != 0


# ------------------------------------------------------------------ keep-out / placement parity with a plain bwphoto job
KEEPOUT = {"profile": "kokuyo_b5_26h", "edge": "front", "mm": 10.0}
KO_LIMIT = 110.0 - 10.0 - 0.5 * 0.12          # allowed paper-frame y (front strip), pulled back half a fill width


def _canvas_body(did, **extra):
    b = {"style": "bwphoto", "paper": PAPER, "margin": MARGIN, "registered_paper": True, "paper_keepout": KEEPOUT,
         "paper_rot": 0, "offset_x": 60.0, "offset_y": 70.0, "microlift": True, "end_policy": "release", "quiet_fans": True,
         "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 9.0, "crop": None, "fill_page": False,
                    "true_size": False, "registration_marks": False}]}
    b.update(extra)
    return b


def _add_drawing():
    with open(IMG, "rb") as f:
        return _client().post("/api/add_drawing", data={"file": (f, IMG.name)}, content_type="multipart/form-data").get_json()["id"]


def _ymax(strokes):
    return max(float(s[:, 1].max()) for s in strokes)


@pytest.fixture(scope="module")
def ko():
    """Canvas-mode keytint job with the hole-strip keep-out and a dy that pushes the picture INTO the strip, plus the plain
    bwphoto jobs (same body) with and without the keep-out."""
    did = _add_drawing()
    r = _client().post("/api/twopen_keytint", json=_canvas_body(did, blue_amount=0.375, debug=False))
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    plain_ko = _client().post("/api/process", json={**_canvas_body(did), "job_type": "preview", "pen_id": BLACK})
    assert plain_ko.status_code == 200, plain_ko.get_json()
    ko_strokes = _strokes(WORK_DIR / "layout.routed.svg")
    nk = _canvas_body(did)
    nk.pop("paper_keepout")
    plain_free = _client().post("/api/process", json={**nk, "job_type": "preview", "pen_id": BLACK})
    assert plain_free.status_code == 200, plain_free.get_json()
    return {"resp": j, "plain_ko": ko_strokes, "plain_free": _strokes(WORK_DIR / "layout.routed.svg"), "did": did}


def test_keepout_layers_inside_the_plain_jobs_allowed_window(ko):
    assert _ymax(ko["plain_free"]) > KO_LIMIT + 1.0, "control: without the keep-out the picture reaches the strip"
    assert _ymax(ko["plain_ko"]) <= KO_LIMIT + 1e-3                    # the plain job clips server-side (composite.py)
    for color in ("blue", "black"):
        strokes = _strokes(ko["resp"]["keytint"]["layers"][color])
        assert strokes and _ymax(strokes) <= KO_LIMIT + 1e-3, (color, _ymax(strokes))
    # and the two layers sit where the plain job sits (same placement): ink bboxes agree to the stroke-width scale
    plain = np.vstack(ko["plain_ko"])
    both = np.vstack(_strokes(ko["resp"]["keytint"]["layers"]["black"]) + _strokes(ko["resp"]["keytint"]["layers"]["blue"]))
    assert np.abs(plain.min(0) - both.min(0)).max() < 1.0 and np.abs(plain.max(0) - both.max(0)).max() < 1.0


def test_plain_job_reach_warning_is_propagated_and_clip_is_silent(ko):
    from ccplotter.routes.twopen import _plain_reach_warnings
    synthetic = [{"level": "danger", "msg": "This layout currently falls outside the pen's reachable area, so it"},
                 {"level": "warning", "msg": "Pen-physics ceiling"}]
    assert _plain_reach_warnings(synthetic) == [synthetic[0]["msg"]] and _plain_reach_warnings(None) == []
    kt_w = ko["resp"]["keytint"]["warnings"]
    plain_w = ko["resp"]["keytint"]["separation"]["plain_job"].get("warnings", [])
    assert all(m in kt_w for m in _plain_reach_warnings(plain_w))       # keytint carries every plain reach warning
    # a keep-out clip produced no warning in the plain job either (silent, matching /api/process)
    assert not any("keep" in str(w).lower() and "clip" in str(w).lower() for w in plain_w)


def test_reach_violation_refused_by_keytint_as_by_process(ko):
    did = ko["did"]
    bad = _canvas_body(did, offset_x=900.0)
    p = _client().post("/api/process", json={**bad, "job_type": "gcode", "pen_id": BLACK})
    assert p.status_code >= 400 and "reach" in (p.get_json().get("error", "")).lower()
    k = _client().post("/api/twopen_keytint", json=bad)
    assert 400 <= k.status_code < 500 and "reach" in k.get_json()["error"].lower(), k.get_json()
    # positive control: the same body with a legal offset is accepted by both
    assert ko["resp"]["ok"]


def test_canvas_mode_refuses_unsupported_items():
    did = _add_drawing()
    for patch in ({"fill_page": True}, {"true_size": True}):
        b = _canvas_body(did)
        b["items"][0].update(patch)
        assert _client().post("/api/twopen_keytint", json=b).status_code == 400
    b = _canvas_body(did)
    b["items"] = b["items"] * 2
    assert _client().post("/api/twopen_keytint", json=b).status_code == 400
