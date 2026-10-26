"""Calibration-sheet provenance and canvas refusal.

Every /api/calibration_sheet generation leaves a run_provenance sidecar
(profile, sheet type, params, output path, gcode sha256) and one app log line; the gcode
bytes are untouched. A registry sheet imported onto the canvas as a POSITION
PREVIEW is refused for canvas gcode (preview still allowed), because canvas processing
would rescale or reorder the sheet's exact geometry; an ordinary drawing is not refused.
Run: python -m pytest tests/test_calsheet_provenance_and_canvas_refusal.py  (one file).
"""
import hashlib
import logging
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home
_HOME = _test_home.use_temp_home("ccplotter_test_home_calsheet_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_calsheet_"))
sys.path.insert(0, str(REPO))

import pytest

import ccplotter.routes.pen  # noqa: F401
from ccplotter.routes.core import app
from ccplotter.config import WORK_DIR
from ccplotter.runs import load_run_provenance
from ccplotter.calibration_registry import generic_sheets


def _first_sheet_id():
    last = None
    for s in generic_sheets():
        last = s.id
        r = app.test_client().post("/api/calibration_sheet", json={"id": s.id})
        if r.status_code == 200:
            return s.id, r
    raise AssertionError(f"no generic sheet generated on the test profile (last {last})")


@pytest.fixture(scope="module")
def sheet(request):
    cap = []

    class H(logging.Handler):
        def emit(self, rec):
            cap.append(rec.getMessage())
    h = H(); lg = logging.getLogger("ccplotter.calsheet")
    lg.setLevel(logging.INFO); lg.addHandler(h)
    sid, r = _first_sheet_id()
    lg.removeHandler(h)
    j = r.get_json()
    return {"id": sid, "json": j, "log": cap,
            "gcode": (WORK_DIR / j["block"]["file"]).read_bytes()}


def test_provenance_sidecar_has_sha_and_profile(sheet):
    prov = sheet["json"]["provenance"]
    assert prov and prov["id"].startswith("cal-")
    rec = load_run_provenance(prov["id"])
    assert rec["kind"] == "calibration_sheet"
    assert rec["sheet_type"] == sheet["id"]
    assert rec["pen_id"] == sheet["json"]["pen_id"] and rec["pen_profile"]
    assert rec["output_path"].endswith(sheet["json"]["block"]["file"])
    assert rec["gcode_sha256"] == hashlib.sha256(sheet["gcode"]).hexdigest()
    assert rec["gcode_sha256"] == prov["sha256"] and rec["created_at"]


def test_provenance_is_under_temp_home_and_gcode_unstamped(sheet):
    prov = sheet["json"]["provenance"]
    assert (_HOME / prov["provenance_file"]).is_file()
    assert b"ccplotter-run:" not in sheet["gcode"]      # bytes not back-linked
    # negative control: a different sha must not equal the recorded one
    assert hashlib.sha256(sheet["gcode"] + b" ").hexdigest() != prov["sha256"]


def test_app_log_line_carries_profile_and_sha(sheet):
    sha = sheet["json"]["provenance"]["sha256"]
    line = [m for m in sheet["log"] if sha in m]
    assert line and f"pen_id={sheet['json']['pen_id']}" in line[0]


def test_canvas_gcode_of_sheet_preview_refused(sheet):
    c = app.test_client()
    imp = c.post("/api/calibration_import",
                 json={"svg_name": sheet["json"]["block"]["svg_name"], "label": "x"}).get_json()
    assert imp["is_cal_sheet"] is True
    body = {"paper": "120x120", "margin": 10.0, "pen_id": sheet["json"]["pen_id"],
            "items": [{"id": imp["id"], "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}]}
    r = c.post("/api/process", json={**body, "job_type": "gcode"})
    assert r.status_code == 400 and "registry file" in r.get_json()["error"]
    # preview stays allowed (the refusal is not a blanket block)
    r2 = c.post("/api/process", json={**body, "job_type": "preview"})
    assert r2.status_code != 400 or "registry file" not in (r2.get_json() or {}).get("error", "")


def test_non_sheet_calibration_import_not_flagged(sheet):
    # positive control: an svg that did NOT come from the dispatcher is not marked
    p = WORK_DIR / "plain_cal_test.svg"
    p.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
                 '<path d="M0 0 L10 10" stroke="black" fill="none"/></svg>')
    imp = app.test_client().post("/api/calibration_import",
                                 json={"svg_name": p.name, "label": "x"}).get_json()
    assert imp["is_cal_sheet"] is False
    assert not (WORK_DIR / f"calsheetmark_{imp['id']}.flag").exists()
