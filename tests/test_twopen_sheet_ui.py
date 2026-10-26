"""M10-REG two-pen sheets reachable from the UI: served index.html control, catalogue route,
both variants return two files, safety scans, tip window, provenance with BOTH shas, canvas-gcode refusal."""
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home
_HOME = _test_home.use_temp_home("ccplotter_test_home_twopenui_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_twopenui_"))
sys.path.insert(0, str(REPO))

import pytest

import ccplotter.routes.pen  # noqa: F401
import ccplotter.routes.runs  # noqa: F401
from ccplotter.config import WORK_DIR
from ccplotter.routes.core import app
from ccplotter.runs import load_run_provenance, strip_run_backlink


def _client():
    return app.test_client()


@pytest.fixture(scope="module")
def sheets():
    from ccplotter.cal import load_pen_profiles, save_pen_profiles
    d = load_pen_profiles()
    if "bic_cristal_blue_b" not in d["profiles"]:
        cp = dict(d["profiles"]["bic_cristal_blue"])
        cp["label"] = "TEST COPY (temp home only)"
        d["profiles"]["bic_cristal_blue_b"] = cp
        save_pen_profiles(d)
    out = {}
    for v in ("A", "B"):
        r = _client().post("/api/twopen_calsheet", json={"variant": v})
        assert r.status_code == 200, r.get_json()
        out[v] = r.get_json()
    return out


def test_served_index_has_the_control():
    html = _client().get("/").get_data(as_text=True)
    for needle in ("twopen-sheet-variant", "genTwoPenSheet", "/api/twopen_calsheet",
                   "check the active pen profile".lower()):
        assert needle.lower() in html.lower(), needle
    assert "swap the refill without moving the paper" in html


def test_catalogue_presets_pens(sheets):
    j = _client().get("/api/twopen_calsheet_info").get_json()["sheets"]
    ids = {s["variant"]: [p["pen_id"] for p in s["pens"]] for s in j}
    assert ids == {"A": ["bic_cristal_blue", "bic_cristal_blue_b"], "B": ["bic_cristal_blue", "bic_cristal_black"]}


@pytest.mark.parametrize("v", ["A", "B"])
def test_two_files_names_and_safety(sheets, v):
    j = sheets[v]
    assert len(j["download"]) == 2 and len(j["gcode_files"]) == 2
    names = [d["download_name"] for d in j["download"]]
    assert names[0] == f"m10reg_{v}_pass1_bic_cristal_blue.gcode"
    assert names[1] == f"m10reg_{v}_pass2_{'bic_cristal_blue_b' if v == 'A' else 'bic_cristal_black'}.gcode"
    assert j["pass_colors"][0] != j["pass_colors"][1]
    for d in j["download"]:
        raw = _client().get(d["url"]).get_data()
        text = raw.decode("ascii")                               # pure ASCII or this raises
        code = "\n".join(l.split(";")[0] for l in text.splitlines())
        assert not re.search(r"\bG92\b", code)
        assert not re.search(r"^\s*(M0|M1|M600|M25|M226|PAUSE)\b", code, re.I | re.M)
        fans = [l.strip() for l in code.splitlines() if l.strip().startswith("M106")]
        assert fans and all(re.fullmatch(r"M106( P[23])?( S\d+)?", f) for f in fans), fans
        assert not re.search(r"\bM106\s+P[01]\b", code)
        lines = code.splitlines()
        fan_idx = [i for i, l in enumerate(lines) if l.strip().startswith("M106")]
        draw_idx = [i for i, l in enumerate(lines) if l.startswith(("G1 X", "G1 Y"))]
        # No fan code inside the plot body. The quiet_fans fan-off block (same as every other sheet) sits in the
        # preamble (before the first move) and the postamble (after the last); nothing in between.
        assert all(i < draw_idx[0] or i > draw_idx[-1] for i in fan_idx), "fan code inside the plot body"
        assert any(i > draw_idx[-1] for i in fan_idx), "fan-off missing from the postamble"
    for p in j["passes"]:
        tx0, ty0, tx1, ty1 = p["tip_bbox"]
        assert 13.5 <= ty0 and ty1 <= 180.5, p["tip_bbox"]
        assert ty0 - 1.5 >= 12.0


@pytest.mark.parametrize("v", ["A", "B"])
def test_provenance_has_both_shas(sheets, v):
    j = sheets[v]
    prov = load_run_provenance(j["run_id"])
    assert prov and prov["kind"] == "twopen_calsheet" and prov["sheet_variant"] == v
    for n, d in ((1, j["download"][0]), (2, j["download"][1])):
        rec = prov["gcode_sha256"][f"pass{n}"]
        data = _client().get(d["url"]).get_data()
        assert rec["sha256"] == hashlib.sha256(data).hexdigest()
        assert rec["sha256_without_backlink"] == hashlib.sha256(strip_run_backlink(data)).hexdigest()
    assert prov["gcode_sha256"]["pass1"]["sha256"] != prov["gcode_sha256"]["pass2"]["sha256"]
    assert j["gcode_sha256"] == prov["gcode_sha256"]


@pytest.mark.parametrize("v", ["A", "B"])
def test_canvas_preview_blocks_gcode(sheets, v):
    c = _client()
    imp = c.post("/api/calibration_import", json={"svg_name": sheets[v]["canvas_preview_svg"], "label": "x"}).get_json()
    assert imp["is_cal_sheet"] is True
    body = {"paper": "120x120", "margin": 10.0, "pen_id": "bic_cristal_blue",
            "items": [{"id": imp["id"], "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}]}
    r = c.post("/api/process", json={**body, "job_type": "gcode"})
    assert r.status_code == 400 and "registry file" in r.get_json()["error"]


def test_preview_has_two_distinct_colours(sheets):
    svg = (WORK_DIR / sheets["A"]["canvas_preview_svg"]).read_text(encoding="utf-8").lower()
    assert "#1f4fd8" in svg and "#e8590c" in svg
