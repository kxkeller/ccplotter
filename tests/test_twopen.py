"""Opt-in two-pen job backend (POST /api/twopen, ccplotter/twopen.py).

Two stroke layers in -> two COMPLETE gcode files (one per pen profile), one two-colour
preview, one run record with both gcodes back-linked. Same conventions as
tests/test_run_provenance.py: explicit temp runtime home BEFORE any ccplotter import,
temp WORK_DIR, Flask test client (no server, never :5001).

Positive controls are paired with every null: the Z test uses two pens whose z values
DIFFER (so "each file matches its own profile" cannot pass if both used one profile),
the config-restore test shows the config really IS mutated mid-pass, and the "normal job
unchanged" test compares real /api/process output before and after a two-pen job.
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
_HOME = _test_home.use_temp_home("ccplotter_test_home_twopen_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_twopen_"))
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
from PIL import Image
import vpype as vp

import ccplotter.routes.runs  # noqa: F401
from ccplotter import twopen as tp
from ccplotter.cal import load_pen_profiles, save_pen_profiles, resolve_z_touch
from ccplotter.config import FAN_OFF_GCODE, WORK_DIR
from ccplotter.runs import (BACKLINK_PREFIX, _runs_load, strip_run_stamps)

BLUE, BLACK = "bic_cristal_blue", "energel_05"      # z_touch 18.75 vs 12.05 in the seed


def _client():
    from ccplotter.routes.core import app
    return app.test_client()


@pytest.fixture(scope="module", autouse=True)
def _distinct_profiles():
    """Make the two pens differ in EVERY z field, in the TEMP home only."""
    d = load_pen_profiles()
    d["profiles"][BLUE].update({"z_up": 3.2, "z_down": -0.9, "z_micro": 0.8,
                                "draw_speed": 2500, "plunge_speed": 450})
    d["profiles"][BLACK].update({"z_up": 3.7, "z_down": -0.5, "z_micro": 0.5,
                                 "draw_speed": 3500, "plunge_speed": 550})
    save_pen_profiles(d)
    return d


def _svg(kind, w=120, h=120):
    """Hatch lines (blue) / outline box + diagonal (black), mm page coords."""
    if kind == "blue":
        body = "".join(f'<line x1="25" y1="{y}" x2="95" y2="{y}"/>' for y in range(30, 91, 4))
    else:
        body = ('<rect x="25" y="30" width="70" height="60"/>'
                '<line x1="25" y1="30" x2="95" y2="90"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" '
            f'viewBox="0 0 {w} {h}" fill="none" stroke="black" stroke-width="0.3">{body}</svg>')


def _body(**extra):
    b = {"blue_svg": _svg("blue"), "black_svg": _svg("black"),
         "blue_pen_id": BLUE, "black_pen_id": BLACK, "paper": "120x120"}
    b.update(extra)
    return b


def _post(body):
    return _client().post("/api/twopen", json=body)


@pytest.fixture(scope="module")
def job():
    r = _post(_body())
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    files = {g["color"]: WORK_DIR / g["file"] for g in j["gcode_files"]}
    return {"resp": j, "files": files, "text": {k: v.read_text(encoding="utf-8") for k, v in files.items()},
            "rec": next(x for x in _runs_load() if x["id"] == j["run_id"])}


def _lines(text):
    return [ln.strip() for ln in text.splitlines()]


def _g1_z(text):
    return [float(m.group(1)) for m in re.finditer(r"^G1 Z(-?\d+(?:\.\d+)?)", text, re.M)]


# ------------------------------------------------------------------ files + lifecycle
def test_two_files_named_and_ordered(job):
    j = job["resp"]
    assert [g["color"] for g in j["gcode_files"]] == ["blue", "black"]
    nb, nk = job["files"]["blue"].name, job["files"]["black"].name
    assert nb == f"{j['run_id']}_1_blue_{BLUE}.gcode"
    assert nk == f"{j['run_id']}_2_black_{BLACK}.gcode"
    assert all(p.is_file() and p.stat().st_size > 500 for p in job["files"].values())


def test_each_file_is_a_complete_normal_lifecycle(job):
    for color, text in job["text"].items():
        ls = _lines(text)
        assert ls[0].startswith("; Pen plotter"), color
        for must in ("G21", "G90", "T0", "M117 Plotting...", "M400", "M107", "M84", "M117 Plot complete"):
            assert must in ls, (color, must)
        assert text.count("M117 Plot complete") == 1          # exactly one lifecycle per file
        assert text.count("M117 Plotting...") == 1
        # draws something: plunges to its own z_down
        assert any(l.startswith("G1 X") for l in ls)
        # fan-off only in preamble (quiet_fans) + the single postamble
        fan_lines = [l for l in ls if l.startswith("M106")]
        assert set(fan_lines) <= {"M106 S0", "M106 P2 S0", "M106 P3 S0", *FAN_OFF_GCODE}
        # end of file: postamble last
        assert ls[-1] == "M117 Plot complete" or ls[-2:] == ["M84", "M117 Plot complete"] or "M117 Plot complete" in ls[-3:]


def test_header_comments_present_and_ordered(job):
    assert PASS_1 in job["text"]["blue"] and PASS_2 in job["text"]["black"]
    assert PASS_2 not in job["text"]["blue"] and PASS_1 not in job["text"]["black"]


PASS_1 = "; PASS 1 OF 2: BLUE -- after this file, swap to the black pen, re-check z, do not move the paper"
PASS_2 = "; PASS 2 OF 2: BLACK"


# ------------------------------------------------------------------ z values per profile
def test_z_values_come_from_each_profile_not_the_body(job):
    """The request body carries z_down=-5 / z_up=9 as a trap: they must be ignored."""
    r = _post(_body(z_down=-5.0, z_up=9.0, draw_speed=9999))
    assert r.status_code == 200, r.get_json()
    prof = load_pen_profiles()["profiles"]
    for text, pid in ((job["text"]["blue"], BLUE), (job["text"]["black"], BLACK)):
        _check_z(text, prof[pid])
    j2 = r.get_json()
    for g, pid in zip(j2["gcode_files"], (BLUE, BLACK)):
        text = (WORK_DIR / g["file"]).read_text(encoding="utf-8")
        _check_z(text, prof[pid])
        assert "G1 Z-" not in text and "F9999" not in text
        assert 9.0 + prof[pid]["z_touch_abs"] not in _g1_z(text)


def _check_z(text, prof):
    zt = prof["z_touch_abs"]
    zs = _g1_z(text)
    down, short = round(zt + prof["z_down"], 2), round(zt + prof["z_micro"], 2)
    # microlift rewrites every z_up hop to THIS pen's contact + z_micro (short) / travel (long)
    assert down in zs and short in zs, (down, short, sorted(set(zs)))
    assert round(zt + prof["z_up"], 2) not in zs
    # the plunge line uses the pen's own plunge speed; draw uses its draw speed
    assert re.search(rf"^G1 Z{down:.2f} F{prof['plunge_speed']:.0f}$", text, re.M)
    assert re.search(rf"^G1 X[\d.]+ Y[\d.]+ F{prof['draw_speed']:.0f}$", text, re.M)
    # nothing plunges lower than this pen's own contact depth
    assert min(zs) >= down - 1e-6


def test_the_two_files_really_differ_in_z(job):
    """Positive control: if both passes had used one profile this would fail."""
    zb, zk = set(_g1_z(job["text"]["blue"])), set(_g1_z(job["text"]["black"]))
    assert round(18.75 - 0.9, 2) in zb and round(12.05 - 0.5, 2) in zk
    assert round(18.75 - 0.9, 2) not in zk and round(12.05 - 0.5, 2) not in zb


def test_pen_xy_offset_is_per_pen():
    """Same layer through two pens with different pen_offset_y lands at different nozzle Y."""
    d = load_pen_profiles()
    assert d["profiles"]["platinum_preppy"]["pen_offset_y"] == -8.0
    r = _post(_body(blue_pen_id="platinum_preppy"))
    assert r.status_code == 200, r.get_json()
    t1 = (WORK_DIR / r.get_json()["gcode_files"][0]["file"]).read_text(encoding="utf-8")
    t2 = (WORK_DIR / r.get_json()["gcode_files"][1]["file"]).read_text(encoding="utf-8")
    prof = load_pen_profiles()["profiles"]

    def ys(t, pid):
        f = "%.0f" % prof[pid]["draw_speed"]
        return [float(m.group(1)) for m in re.finditer(r"^G1 X[\d.]+ Y(-?[\d.]+) F" + f + "$", t, re.M)]
    # t1 = preppy (blue pass), t2 = energel (black pass): same page geometry, nozzle driven 8mm apart
    d = min(ys(t1, "platinum_preppy")) - min(ys(t2, BLACK))
    assert abs(abs(d) - 8.0) < 0.3, d


# ------------------------------------------------------------------ safety scan
BAD_FAN = re.compile(r"\bM106\s+P[01]\b", re.I)
BAD_PAUSE = re.compile(r"^\s*(M0|M1|M600|M25|M226)\b", re.I | re.M)


def test_ascii_no_g92_fan_indices_no_pause_on_both(job):
    for color, p in job["files"].items():
        raw = p.read_bytes()
        raw.decode("ascii")                                        # raises on any non-ASCII byte
        text = raw.decode("ascii")
        code = "\n".join(l.split(";")[0] for l in text.splitlines())   # comments may MENTION G92
        assert not re.search(r"\bG92\b", code), color
        assert not BAD_FAN.search(text), color
        assert not BAD_PAUSE.search(text), color
        for ln in text.splitlines():
            if ln.startswith("M106"):
                assert ln.split(";")[0].strip() in {"M106 S0", "M106 P2 S0", "M106 P3 S0"} or \
                    re.fullmatch(r"M106( P[23])?( S\d+)?", ln.split(";")[0].strip()), ln
    # the scanner can fail: a poisoned sample trips it
    assert BAD_FAN.search("M106 P0 S0") and BAD_PAUSE.search("M600\n") and re.search(r"\bG92\b", "G92 X0")


def test_fan_off_is_postamble_only_after_motion(job):
    """No M106/M107 between the first draw move and the postamble M400."""
    for text in job["text"].values():
        ls = _lines(text)
        first_draw = next(i for i, l in enumerate(ls) if l.startswith("G1 X") and "F3" in l or l.startswith("G1 X") and "F2500" in l)
        post = max(i for i, l in enumerate(ls) if l == "M107")
        mid = [l for l in ls[first_draw:post] if l.startswith(("M106", "M107"))]
        assert mid == [], mid


# ------------------------------------------------------------------ preview
def test_preview_has_two_colours_blue_under_black(job):
    svg = (WORK_DIR / job["resp"]["preview_file"]).read_text(encoding="utf-8").lower()
    assert tp.BLUE_PREVIEW_COLOR in svg and tp.BLACK_PREVIEW_COLOR in svg
    assert svg.index(tp.BLUE_PREVIEW_COLOR) < svg.index(tp.BLACK_PREVIEW_COLOR)   # blue written first = under
    # a normal preview stays single-colour (PREVIEW_COLOR default untouched)
    from ccplotter.config import PREVIEW_COLOR
    assert PREVIEW_COLOR == "#16181d"


# ------------------------------------------------------------------ run record
def test_run_record_lists_both_files_with_backlinks(job):
    rec, rid = job["rec"], job["resp"]["run_id"]
    assert rec["type"] == "twopen" and rec["id"] == rid
    assert {"gcode_blue", "gcode_black", "preview"} <= set(rec["files"])
    assert rec["provenance_file"] == f"run_provenance/{rid}.json"
    for key in ("gcode_blue", "gcode_black"):
        archived = (WORK_DIR / rec["files"][key]).read_text(encoding="utf-8")
        assert f"{BACKLINK_PREFIX}id={rid} provenance={rec['provenance_file']}" in archived, key
        assert archived.count(BACKLINK_PREFIX) == 1
    # the sideload files carry it too, and the archive is byte-equal to them
    for color, key in (("blue", "gcode_blue"), ("black", "gcode_black")):
        assert job["text"][color] == (WORK_DIR / rec["files"][key]).read_text(encoding="utf-8")
        assert BACKLINK_PREFIX in job["text"][color]
    assert rec["gcode_backlink"] is True


# ------------------------------------------------------------------ refusals
@pytest.mark.parametrize("patch", [
    {"blue_pen_id": "no_such_pen"},
    {"black_pen_id": "no_such_pen"},
    {"black_pen_id": BLUE},                      # same pen twice
    {"blue_pen_id": None},
    {"blue_svg": "C:/definitely/not/here.svg"},
    {"black_svg": '<svg xmlns="http://www.w3.org/2000/svg" width="120mm" height="120mm"></svg>'},   # empty
    {"offset_x": 900.0},                          # reach failure
    {"offset_y": -400.0},                         # reach failure
    {"paper": "banana"},
])
def test_refusals_are_4xx_never_500(patch):
    before = len(_runs_load())
    r = _post(_body(**patch))
    assert 400 <= r.status_code < 500, (patch, r.status_code, r.get_json())
    assert r.get_json().get("error")
    assert len(_runs_load()) == before            # a refusal leaves no run record


def test_reach_failure_is_attributed_to_reach():
    r = _post(_body(offset_x=900.0))
    assert r.status_code == 422 and "reach" in r.get_json()["error"].lower()


def test_non_json_body_is_400():
    r = _client().post("/api/twopen", data="nope", content_type="text/plain")
    assert r.status_code == 400


# ------------------------------------------------------------------ vpype global config
def _gw_defaults():
    """The whole vpype config, as plain JSON (so any leaked key anywhere shows up)."""
    return json.loads(json.dumps(vp.config_manager.config, default=str, sort_keys=True))


def _gw_values():
    return vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"]


def test_global_vpype_config_restored_after_a_twopen_job():
    before = _gw_defaults()
    r = _post(_body())
    assert r.status_code == 200, r.get_json()
    assert _gw_defaults() == before
    # ... and also restored when a pass FAILS part-way (reach refusal in pass 1)
    r2 = _post(_body(offset_x=900.0))
    assert r2.status_code == 422
    assert _gw_defaults() == before


def test_config_really_is_mutated_mid_pass_positive_control(tmp_path):
    """If generate_pass never touched the config this test file's restore check would be
    vacuous. Observe the mutation from inside a pass."""
    seen = {}
    real = tp._microlift_gcode

    def spy(path, z_up_abs, **kw):
        seen["z_down_abs"] = vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"]["z_down_abs"]
        return real(path, z_up_abs, **kw)
    tp._microlift_gcode = spy
    try:
        before = _gw_defaults()
        assert _post(_body()).status_code == 200
    finally:
        tp._microlift_gcode = real
    assert seen["z_down_abs"] == pytest.approx(12.05 - 0.5)
    assert before.get("gwrite", {}).get("elegoo_cc", {}).get("default_values", {}).get("z_down_abs") != seen["z_down_abs"]
    assert _gw_defaults() == before


def _fixed_gray(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return (127 + 90 * np.sin(xx / 11.0) * np.cos(yy / 17.0)).clip(0, 255).astype(np.uint8)


def test_normal_job_after_twopen_is_identical_to_one_before():
    did = "twopen_normal_src"
    base = _fixed_gray(500, 700)
    img = np.stack([base, np.roll(base, 5, axis=0), np.roll(base, -5, axis=1)], axis=-1)
    Image.fromarray(img, mode="RGB").save(WORK_DIR / f"src_{did}.jpg", quality=92)
    body = {"job_type": "gcode", "style": "tracev3", "debug": True, "paper": "120x120", "margin": 10.0,
            "resolution": 600, "pen_id": BLACK,
            "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}]}

    def run_normal():
        r = _client().post("/api/process", json=json.loads(json.dumps(body)))
        assert r.status_code == 200, r.get_json()
        return strip_run_stamps((WORK_DIR / "layout.gcode").read_bytes())

    before = run_normal()
    assert _post(_body(blue_pen_id=BLUE, black_pen_id="platinum_preppy", motion_smoothing="zoned",
                       quiet_fans=False, end_policy="hold")).status_code == 200
    after = run_normal()
    assert len(before) > 1000
    assert after == before
    # control: a twopen-flavoured gcode is NOT what a normal job emits (files differ), so
    # equality above is not a trivially-constant comparison
    assert before != job_bytes_stub()


def job_bytes_stub():
    r = _post(_body())
    return strip_run_stamps((WORK_DIR / r.get_json()["gcode_files"][0]["file"]).read_bytes())


# ------------------------------------------------------------------ additional behaviours
def test_exactly_one_pass_header_line_per_file(job):
    for color, text in job["text"].items():
        assert len([l for l in text.splitlines() if l.startswith("; PASS")]) == 1, color


def test_pass_header_carries_pair_partner_placement_pen(job):
    rid = job["resp"]["run_id"]
    prof = load_pen_profiles()["profiles"]
    names = {c: job["files"][c].name for c in job["files"]}
    for color, other, pid in (("blue", "black", BLUE), ("black", "blue", BLACK)):
        t = job["text"][color]
        assert f"; PAIR {rid} partner_file={names[other]}" in t
        m = re.search(r"^; PLACEMENT shared (.*)$", t, re.M)
        assert m, color
        for key in ("offset_x=", "offset_y=", "scale=1", "rot=0", "dx=0", "dy=0", "paper=120.0x120.0mm", "paper_rot=0"):
            assert key in m.group(1), (key, m.group(1))
        assert f"; PEN this_pass pen_id={pid} z_touch_abs={prof[pid]['z_touch_abs']:.2f}" in t
    # placement line is identical in both files (shared); pen lines differ
    pl = [re.search(r"^; PLACEMENT.*$", job["text"][c], re.M).group(0) for c in ("blue", "black")]
    assert pl[0] == pl[1]
    pn = [re.search(r"^; PEN .*$", job["text"][c], re.M).group(0) for c in ("blue", "black")]
    assert pn[0] != pn[1]


def test_nondefault_placement_is_echoed_in_header():
    r = _post(_body(scale=0.8, rot=90, dx=3, dy=-2, paper_rot=0))
    assert r.status_code == 200, r.get_json()
    t = (WORK_DIR / r.get_json()["gcode_files"][0]["file"]).read_text(encoding="utf-8")
    m = re.search(r"^; PLACEMENT shared (.*)$", t, re.M).group(1)
    assert "scale=0.8" in m and "rot=90" in m and "dx=3" in m and "dy=-2" in m


def test_provenance_resolved_has_both_placement_and_pen_offsets(job):
    from ccplotter.runs import load_run_provenance
    res = load_run_provenance(job["resp"]["run_id"])["resolved"]
    prof = load_pen_profiles()["profiles"]
    assert res["placement"]["scale"] == 1.0 and res["placement"]["paper_w"] == 120
    offs = res["placement_offsets_mm"]
    assert set(offs) == {"blue", "black", "shared_pen_target"}
    assert offs["blue"] and offs["black"] and len(offs["shared_pen_target"]) == 2
    from ccplotter.cal import load_cal, resolve_pen_offset      # composed: global mount + per-pen
    for c, pid in (("blue", BLUE), ("black", BLACK)):
        assert res["pen_offsets_mm"][c] == list(resolve_pen_offset(load_cal(), prof[pid]))
    for c in ("blue", "black"):      # nozzle offset = shared target - pen offset, per pen
        po = res["pen_offsets_mm"][c]
        assert offs[c] == pytest.approx([offs["shared_pen_target"][0] - po[0], offs["shared_pen_target"][1] - po[1]])


def test_preppy_pair_records_distinct_offsets():
    r = _post(_body(blue_pen_id="platinum_preppy"))
    assert r.status_code == 200, r.get_json()
    from ccplotter.runs import load_run_provenance
    res = load_run_provenance(r.get_json()["run_id"])["resolved"]
    assert res["pen_offsets_mm"]["blue"][1] - res["pen_offsets_mm"]["black"][1] == pytest.approx(-8.0)
    assert res["placement_offsets_mm"]["blue"] != res["placement_offsets_mm"]["black"]


def test_time_estimate_model_b_labelled_in_response_and_record(job):
    from ccplotter.calibration_fragment import time_model_b
    for g in job["resp"]["gcode_files"]:
        t = g["time_est"]
        assert t["model"] == "b" and t["label"] == "model (b), not stopwatch-validated"
        assert t["minutes"] > 0
        assert t["seconds"] == pytest.approx(time_model_b(WORK_DIR / g["file"]), abs=0.1)
    assert all(p["time_est"]["label"].startswith("model (b)") for p in job["rec"]["passes"])


_LINE = '<line x1="1" y1="1" x2="9" y2="9"/>'
_NS = 'xmlns="http://www.w3.org/2000/svg"'


@pytest.mark.parametrize("bad", [
    f'<svg {_NS} width="120" height="120" viewBox="0 0 120 120">{_LINE}</svg>',
    f'<svg {_NS} width="12cm" height="12cm" viewBox="0 0 120 120">{_LINE}</svg>',
    f'<svg {_NS} width="120mm" height="120mm">{_LINE}</svg>',
    f'<svg {_NS} width="120mm" height="120mm" viewBox="0 0 1000 1000">{_LINE}</svg>',
])
def test_non_mm_layers_refused_400(bad):
    for key in ("blue_svg", "black_svg"):
        r = _post(_body(**{key: bad}))
        assert r.status_code == 400, (key, r.status_code, r.get_json())
        assert "mm" in r.get_json()["error"].lower()


def test_mm_check_accepts_vpype_px_viewbox():
    px = 120 * 96 / 25.4
    ok = f'<svg {_NS} width="120mm" height="120mm" viewBox="0 0 {px:.3f} {px:.3f}">{_LINE}</svg>'
    tp.check_svg_mm(ok, "blue")        # does not raise
    with pytest.raises(tp.TwoPenRefusal):
        tp.check_svg_mm(ok.replace("120mm", "120px", 1), "blue")


# preview width helper: byte-identical to the expression inlined in _composite
def _old_inline(render_opts, pen_profile):
    pen_mm = float((render_opts or {}).get("laid_mm")
                   or (render_opts or {}).get("fill_mm", 0.15))
    _core = (pen_profile or {}).get("core_mm")
    return float(_core) if _core else pen_mm


@pytest.mark.parametrize("prof", [
    {"core_mm": 0.18, "laid_mm": 0.3, "fill_mm": 0.25},     # core set
    {"core_mm": 0.18, "fill_mm": 0.25},
    {"laid_mm": 0.3, "fill_mm": 0.25},                      # laid
    {"laid_mm": 0.3},
    {"fill_mm": 0.25},                                      # fill only
    {"core_mm": None, "fill_mm": 0.25},
    {"core_mm": 0, "laid_mm": 0.0, "fill_mm": 0.12},
    {},                                                     # none -> 0.15
    None,
])
def test_preview_pen_width_helper_matches_old_inline_expression(prof):
    from ccplotter.composite import preview_pen_width_mm
    for ro in (prof, {}, {"fill_mm": 0.4}, {"laid_mm": 0.33, "fill_mm": 0.4}, None):
        assert preview_pen_width_mm(ro, prof) == _old_inline(ro, prof)
    assert preview_pen_width_mm({}, {}) == 0.15
    assert preview_pen_width_mm(None, {"core_mm": 0.2}) == 0.2


def test_twopen_preview_uses_core_laid_fill_chain():
    d = load_pen_profiles()
    d["profiles"][BLUE]["core_mm"] = 0.21
    d["profiles"][BLACK].pop("core_mm", None)
    d["profiles"][BLACK]["laid_mm"] = 0.37
    save_pen_profiles(d)
    try:
        r = _post(_body())
        assert r.status_code == 200, r.get_json()
        svg = (WORK_DIR / r.get_json()["preview_file"]).read_text(encoding="utf-8")
        widths = [float(x) for x in re.findall(r'stroke-width="([\d.]+)"', svg)]
        mm = lambda v: v * 25.4 / 96.0
        assert any(abs(mm(w) - 0.21) < 0.01 for w in widths), widths
        assert any(abs(mm(w) - 0.37) < 0.01 for w in widths), widths
    finally:
        d["profiles"][BLUE].pop("core_mm", None)
        d["profiles"][BLACK].pop("laid_mm", None)
        save_pen_profiles(d)
