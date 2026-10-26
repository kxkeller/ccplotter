"""Main-flow mode "Black Line + Blue Tone (2 pens)": the mode-selector wiring around POST /api/twopen_keytint.

Same conventions as tests/test_twopen_keytint.py: explicit temp runtime home BEFORE any ccplotter import, temp WORK_DIR,
Flask test client (no server, never :5001). Covers: the mode route produces two files (pass 1 blue, pass 2 black) with each
pen's own Z, the two-file Save names, the preview carries both colours, a preview does NOT archive a run while a Generate
does, ASCII / no G92 / valid fans / no pen-change code on both saved files, the hole-strip keep-out parity, the folder-save
helper (all-or-nothing, no traversal), and the UI hookup (option, hidden old box, node --check with a failing control).
Every null has a positive control.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home; pen/calibration never resolve from CWD
_HOME = _test_home.use_temp_home("ccplotter_test_home_twopenmode_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_twopenmode_"))
sys.path.insert(0, str(REPO))

import numpy as np
import pytest

import ccplotter.routes.runs  # noqa: F401
from ccplotter.cal import load_pen_profiles, save_pen_profiles
from ccplotter.config import WORK_DIR
from ccplotter.pair_save import copy_files
from ccplotter.runs import _runs_load

IMG = REPO / "tests" / "fixtures" / "twotone" / "pagoda.jpg"
PAPER, MARGIN = "150x110", 5.0
BLUE, BLACK = "bic_cristal_blue", "bic_cristal_black"
BLUE_Z = {"z_touch_abs": 18.75, "z_up": 3.2, "z_down": -0.9, "z_micro": 0.8, "draw_speed": 2500, "plunge_speed": 450}
BLACK_Z = {"z_touch_abs": 17.35, "z_up": 3.7, "z_down": -0.4, "z_micro": 0.65, "draw_speed": 3500, "plunge_speed": 550}
KEEPOUT = {"profile": "kokuyo_b5_26h", "edge": "front", "mm": 10.0}
KO_LIMIT = 110.0 - 10.0 - 0.5 * 0.12          # allowed paper-frame y (front strip), pulled back half a fill width
INDEX = REPO / "assets" / "templates" / "index.html"
CRLF, LF = chr(13) + chr(10), chr(10)


def _client():
    from ccplotter.routes.core import app
    return app.test_client()


@pytest.fixture(scope="module", autouse=True)
def _distinct_profiles():
    """The two default pens differ in EVERY z field (temp home only), so a crossed profile fails."""
    d = load_pen_profiles()
    d["profiles"][BLUE].update(BLUE_Z)
    d["profiles"][BLACK].update(BLACK_Z)
    save_pen_profiles(d)
    return d


def _add_drawing():
    with open(IMG, "rb") as f:
        return _client().post("/api/add_drawing", data={"file": (f, IMG.name)}, content_type="multipart/form-data").get_json()["id"]


def _body(did, job_type, main_flow=True, **extra):
    """What the UI posts in the mode: a normal Generate body (buildBody) with style bwphoto, main_flow and blue_amount."""
    b = {"style": "bwphoto", "job_type": job_type, "paper": PAPER, "margin": MARGIN, "registered_paper": False,
         "paper_rot": 0, "offset_x": 60.0, "offset_y": 70.0, "microlift": True, "end_policy": "release", "quiet_fans": True,
         "blue_amount": 0.375, "pen_id": BLUE,        # the active pen is IGNORED by the mode (the pens are fixed)
         "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None, "fill_page": False,
                    "true_size": False, "registration_marks": False}]}
    if main_flow:
        b["main_flow"] = True
    b.update(extra)
    return b


@pytest.fixture(scope="module")
def gen():
    did = _add_drawing()
    before = len(_runs_load())
    r = _client().post("/api/twopen_keytint", json=_body(did, "gcode"))
    assert r.status_code == 200, r.get_json()
    return {"did": did, "resp": r.get_json(), "runs_before": before}


def _g1_z(text):
    return [float(m.group(1)) for m in re.finditer(r"^G1 Z(-?\d+(?:\.\d+)?)", text, re.M)]


def _check_z(text, prof):
    zt = prof["z_touch_abs"]
    zs = _g1_z(text)
    down, short = round(zt + prof["z_down"], 2), round(zt + prof["z_micro"], 2)
    assert down in zs and short in zs, (down, short, sorted(set(zs)))
    assert re.search(rf"^G1 Z{down:.2f} F{prof['plunge_speed']:.0f}$", text, re.M)
    assert re.search(rf"^G1 X[\d.]+ Y[\d.]+ F{prof['draw_speed']:.0f}$", text, re.M)
    assert min(zs) >= down - 1e-6


def test_ui_item_geometry_item_settings_match_the_sent_canvas_item(gen):
    """Fit to Paper / Fill Page / auto-fit: the UI's captureFitBounds accepts item_geometry only when
    sameFitItem(sent, meta.item_settings) holds, which compares the item id. The plain job runs under an internal id, so the
    block must carry the CANVAS id there (else Fit says 'Process this drawing before fitting')."""
    sent = _body(gen["did"], "gcode")["items"][0]
    meta = gen["resp"]["ui"]["item_geometry"][gen["did"]]
    st = meta["item_settings"]
    assert str(st["id"]) == str(gen["did"])
    for k in ("scale", "rot", "dx", "dy"):
        assert abs(float(st.get(k) or 0) - float(sent.get(k) or 0)) <= 1e-7, k
    assert st.get("crop") == sent.get("crop")
    # positive control: the comparison is not vacuous -- a foreign id is detected
    assert str(st["id"]) != "c0ffee00"


# ------------------------------------------------------------------ the mode route: two files, names, preview
def test_mode_returns_process_shaped_ui_block_with_two_ordered_passes(gen):
    ui = gen["resp"]["ui"]
    for k in ("paper", "preview_file", "stroke_count", "draw_length_mm", "estimated_plot_time_min", "item_geometry",
              "item_bboxes", "warnings", "keytint_ui"):
        assert k in ui, k
    ps = ui["keytint_ui"]["passes"]
    assert [p["color"] for p in ps] == ["blue", "black"] and [p["pass"] for p in ps] == [1, 2]
    assert [p["pen_id"] for p in ps] == [BLUE, BLACK]                          # fixed pens, the body pen_id is ignored
    assert [p["save_suffix"] for p in ps] == ["_1_blue.gcode", "_2_black.gcode"]
    assert all("model (b)" in p["time_label"] and "not stopwatch-validated" in p["time_label"] for p in ps)
    # item geometry is keyed by the CANVAS drawing id (the UI's fit/handles read it by that id)
    assert gen["did"] in ui["item_geometry"] and ui["item_geometry"][gen["did"]]["base_fit_scale"] > 0
    assert ui["stroke_count"] == sum(p["strokes"] for p in ps) and ui["stroke_count"] > 0
    # D1631a: Z moves follow the /api/process definition (2 per stroke: down + up), NOT the stroke count
    assert ui["estimated_z_moves"] == 2 * sum(p["pen_lifts"] for p in ps) != ui["stroke_count"]
    # D1631b: the run archive folder is reported so a failed Save can name where the two files already are
    rd = Path(ui["keytint_ui"]["run_dir"])
    assert rd.name == gen["resp"]["run_id"] and (rd / "gcode_blue.gcode").is_file() and (rd / "gcode_black.gcode").is_file()


def test_save_writes_both_files_with_plot_order_in_the_names(gen, tmp_path):
    ps = gen["resp"]["ui"]["keytint_ui"]["passes"]
    base = "tramLandscapeLeft1019"
    items = [[p["run_file"], base + p["save_suffix"]] for p in ps]
    r = copy_files(WORK_DIR, tmp_path, items)
    assert r["ok"] and sorted(Path(x).name for x in r["paths"]) == [f"{base}_1_blue.gcode", f"{base}_2_black.gcode"]
    for p in ps:
        assert (tmp_path / (base + p["save_suffix"])).read_bytes() == (WORK_DIR / p["run_file"]).read_bytes()


def test_each_saved_file_is_standalone_with_its_own_pens_z_and_clean_code(gen, tmp_path):
    ps = gen["resp"]["ui"]["keytint_ui"]["passes"]
    copy_files(WORK_DIR, tmp_path, [[p["run_file"], "x" + p["save_suffix"]] for p in ps])
    profs = {"blue": BLUE_Z, "black": BLACK_Z}
    texts = {}
    for p in ps:
        raw = (tmp_path / ("x" + p["save_suffix"])).read_bytes()
        text = raw.decode("ascii").replace(CRLF, LF)      # raises on any non-ASCII byte; the files are CRLF
        texts[p["color"]] = text
        code = "\n".join(l.split(";")[0] for l in text.splitlines())
        assert not re.search(r"\bG92\b", code), p["color"]
        assert not re.search(r"\bM106\s+P[01]\b", text, re.I), p["color"]               # only M106 / P2 / P3 exist
        assert not re.search(r"^\s*(M0|M1|M600|M25|M226)\b", text, re.I | re.M), p["color"]   # no pen-change / pause code
        assert re.search(r"^; PASS", text, re.M) and f"pen_id={p['pen_id']}" in text
        _check_z(text, profs[p["color"]])                       # THAT pass's pen profile, live
    # positive control: the checker rejects crossed profiles
    with pytest.raises(AssertionError):
        _check_z(texts["blue"], BLACK_Z)
    # fans: only the preamble / postamble lines (M106 S0, P2 S0, P3 S0) exist; NONE injected mid-stream, and the fan-off
    # postamble closes each file
    for color, text in texts.items():
        lines = [l.split(";")[0].strip() for l in text.splitlines()]
        code = [l for l in lines if l]
        fan = [i for i, l in enumerate(code) if l.startswith("M106")]
        assert fan and all(code[i] in ("M106 S0", "M106 P2 S0", "M106 P3 S0") for i in fan), color
        assert all(i < 12 or i >= len(code) - 12 for i in fan), color            # preamble or postamble only
        assert "M106 P3 S0" in code[-12:] and code[-1].startswith("M117"), color


def test_preview_contains_both_pen_colours_blue_under_black(gen):
    ui = gen["resp"]["ui"]
    svg = (WORK_DIR / ui["preview_file"]).read_text(encoding="utf-8").lower()
    assert "#1f4fd8" in svg and "#16181d" in svg and svg.index("#1f4fd8") < svg.index("#16181d")
    assert _client().get(f"/work/{ui['preview_file']}").status_code == 200
    # control: a one-colour preview text would fail the same assertion
    assert not ("#1f4fd8" in "<svg>#16181d</svg>")


# ------------------------------------------------------------------ preview vs generate: archive semantics
def test_preview_job_archives_nothing_while_generate_archives_one_run(gen):
    runs_after_gen = _runs_load()
    assert any(x["id"] == gen["resp"]["run_id"] for x in runs_after_gen)                   # Generate recorded its run
    assert (WORK_DIR / gen["resp"]["ui"]["keytint_ui"]["passes"][0]["run_file"]).is_file()
    n = len(runs_after_gen)
    r = _client().post("/api/twopen_keytint", json=_body(gen["did"], "preview"))
    assert r.status_code == 200, r.get_json()
    ui = r.get_json()["ui"]
    assert ui["preview_only"] is True and ui["keytint_ui"]["run_id"] is None
    assert all(p["run_file"] is None for p in ui["keytint_ui"]["passes"])                  # nothing to Save from a preview
    assert len(_runs_load()) == n                                                           # no run record for a preview
    # positive control: the OLD canvas call (no main_flow) with job_type preview still archives its run
    r2 = _client().post("/api/twopen_keytint", json=_body(gen["did"], "preview", main_flow=False))
    assert r2.status_code == 200 and "ui" not in r2.get_json()
    assert len(_runs_load()) == n + 1


# ------------------------------------------------------------------ keep-out parity
def _strokes(svg):
    from ccplotter import twopen_keytint as kt
    return kt.load_strokes_mm(svg)


def test_reach_refusal_names_the_pen_axis_and_overshoot_in_mm(gen):
    # The 422 on an out-of-reach placement carries the reach numbers (which edge, by how many mm), per pass pen
    r = _client().post("/api/twopen_keytint", json=_body(gen["did"], "preview", offset_x=900.0))
    assert 400 <= r.status_code < 500, r.status_code
    msg = r.get_json()["error"]
    assert "reach" in msg.lower() and re.search(r"\d+(\.\d+)?\s*mm", msg) and BLUE in msg, msg
    assert re.search(r"past the (left|right|front|back)", msg), msg
    # positive control: a legal placement is accepted
    assert gen["resp"]["ok"]


def test_keepout_parity_mode_layers_equal_the_old_route_and_stay_out_of_the_strip():
    did = _add_drawing()
    ko = dict(registered_paper=True, paper_keepout=KEEPOUT)
    items = [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 9.0, "crop": None, "fill_page": False,
              "true_size": False, "registration_marks": False}]
    a = _client().post("/api/twopen_keytint", json=_body(did, "preview", items=items, **ko))
    b = _client().post("/api/twopen_keytint", json=_body(did, "preview", main_flow=False, items=items, **ko))
    free = _client().post("/api/twopen_keytint", json=_body(did, "preview", main_flow=False, items=items))
    for r in (a, b, free):
        assert r.status_code == 200, r.get_json()
    la, lb, lf = (r.get_json()["keytint"]["layers"] for r in (a, b, free))
    for color in ("blue", "black"):
        sa, sb = _strokes(la[color]), _strokes(lb[color])
        assert len(sa) == len(sb) and sa
        assert max(float(np.abs(x - y).max()) for x, y in zip(sa, sb)) < 1e-6          # the mode changes no geometry
        assert max(float(s[:, 1].max()) for s in sa) <= KO_LIMIT + 1e-3, color            # clipped out of the hole strip
    # positive control: without the keep-out the picture does reach the strip
    assert max(float(s[:, 1].max()) for c in ("blue", "black") for s in _strokes(lf[c])) > KO_LIMIT + 1.0


# ------------------------------------------------------------------ folder-save helper
def test_pair_save_is_all_or_nothing_and_never_escapes(tmp_path):
    work = tmp_path / "work"; (work / "runs" / "r1").mkdir(parents=True)
    (work / "runs" / "r1" / "a.gcode").write_text("A", encoding="ascii")
    (work / "runs" / "r1" / "b.gcode").write_text("B", encoding="ascii")
    out = tmp_path / "out"; out.mkdir()
    good = [["runs/r1/a.gcode", "n_1_blue.gcode"], ["runs/r1/b.gcode", "n_2_black.gcode"]]
    r = copy_files(work, out, good)
    assert r["ok"] and (out / "n_1_blue.gcode").read_text() == "A" and (out / "n_2_black.gcode").read_text() == "B"
    # second call: both exist -> refused, nothing changed; overwrite=True writes
    (out / "n_1_blue.gcode").write_text("OLD", encoding="ascii")
    r = copy_files(work, out, good)
    assert not r["ok"] and sorted(r["exists"]) == ["n_1_blue.gcode", "n_2_black.gcode"]
    assert (out / "n_1_blue.gcode").read_text() == "OLD"
    assert copy_files(work, out, good, overwrite=True)["ok"] and (out / "n_1_blue.gcode").read_text() == "A"
    # a missing second source writes NOTHING (all-or-nothing), traversal and bad names are refused
    out2 = tmp_path / "out2"; out2.mkdir()
    r = copy_files(work, out2, [good[0], ["runs/r1/missing.gcode", "z.gcode"]])
    assert not r["ok"] and not list(out2.iterdir())
    for bad_src in ("../secret.gcode", "/abs.gcode", "C:/x.gcode", "runs//a.gcode"):
        assert not copy_files(work, out2, [[bad_src, "q.gcode"]])["ok"], bad_src
    for bad_name in ("../q.gcode", "sub/q.gcode", "q.txt", ""):
        assert not copy_files(work, out2, [["runs/r1/a.gcode", bad_name]])["ok"], bad_name
    assert not copy_files(work, tmp_path / "nope", good)["ok"] and not list(out2.iterdir())


# ------------------------------------------------------------------ UI hookup (assets/templates/index.html)
def _scripts():
    return re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", INDEX.read_text(encoding="utf-8"), re.S)


def _node_check(js, tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    f = tmp_path / "x.js"
    f.write_text(js, encoding="utf-8")
    return subprocess.run([node, "--check", str(f)], capture_output=True, text=True)


def test_ui_mode_option_controls_and_hidden_old_box():
    html = INDEX.read_text(encoding="utf-8")
    assert '<option value="twopenkt">Black Line + Blue Tone (2 pens)' in html
    assert re.search(r'id="twopenkt-blue-amount"[^>]*min="0" max="0.5" step="0.025" value="0.375"', html)
    assert 'id="style-twopenkt"' in html and "  twopenkt: '" in html
    box = re.search(r'<div id="keytint-box" style="([^"]*)"', html).group(1)
    assert box.startswith("display:none")                                  # old box hidden, code and ids kept
    js = "\n".join(_scripts())
    assert "async function genKeyTint()" in js
    pf = js[js.index("async function processFile("):js.index("function processPreview()")]
    assert "payload.style === 'twopenkt'" in pf and "/api/twopen_keytint" in pf and "main_flow: true" in pf
    assert "style: 'bwphoto'" in pf and "dj.ui" in pf
    sv = js[js.index("function keyTintBaseName()"):js.index("async function saveToDisk(")]
    assert "api.pick_folder()" in sv and "api.save_files_to_folder(" in sv and "p.run_file" in sv
    assert "if (state.ktUi) { saveKeyTintFiles(); return; }" in js
    # D1631b: EVERY failure toast of the folder save names where the two files already are (k.run_dir, full path)
    fn = sv[sv.index("async function saveKeyTintFiles()"):]
    assert "k.run_dir" in fn and "gcode_blue.gcode" in fn and "gcode_black.gcode" in fn
    failing = [l for l in fn.splitlines() if "'error'" in l]
    assert len(failing) >= 4 and all("where" in l for l in failing), failing
    # The pens note says the overlay / Fit use the two BIC profiles, not the ACTIVE pen
    assert "the plate reach overlay, Fit to Paper and auto-fit use the two BIC profiles (the ACTIVE pen is ignored)" in js.replace("In this mode the plate", "the plate")
    # no apostrophe inside a word in the new single-quoted JS (it ends the string early; build 125aa28d)
    for chunk in (pf, sv, js[js.index("function updateTwoPenKtPens()"):js.index("function updateThresholdUI()")]):
        assert not re.search(r"\w'\w", chunk), re.search(r"\w'\w", chunk).group(0)
    assert re.search(r"\w'\w", "it's")                                      # control: the pattern can fire


def test_ui_script_passes_node_check_and_the_check_can_fail(tmp_path):
    scripts = _scripts()
    assert scripts
    for js in scripts:
        r = _node_check(js, tmp_path)
        assert r.returncode == 0, r.stderr[:500]
    big = max(scripts, key=len)
    assert _node_check(big.replace("function saveKeyTintFiles() {", "function saveKeyTintFiles() { const s = 'it's x';", 1)
                       .replace("async function saveKeyTintFiles() {", "async function saveKeyTintFiles() { const s = 'it's x';", 1),
                       tmp_path).returncode != 0                          # positive control: an injected error is caught


def test_desktop_bridge_exposes_the_folder_methods():
    src = (REPO / "desktop" / "ccplotter_app.py").read_text(encoding="utf-8")
    assert "def pick_folder(self)" in src and "def save_files_to_folder(self, folder, items, overwrite=False)" in src
    assert "from ccplotter.pair_save import copy_files" in src
