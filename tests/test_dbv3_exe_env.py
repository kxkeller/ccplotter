"""CCPLOTTER_DBV3_EXE selects the DrawingBotV3 executable; unset means a clear 404.

Each case runs in a fresh subprocess because the path is read when the routes
module is imported. The launch route is never called with an executable set, so
nothing is ever started.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CHILD = r'''
import json, os, sys
sys.path.insert(0, os.environ["REPO"])
sys.path.insert(0, os.path.join(os.environ["REPO"], "tests"))
import _test_home; _test_home.use_temp_home()
os.chdir(os.environ["REPO"])
import app
c = app.app.test_client()
out = {"status": c.get("/api/dbv3_status").get_json()}
if os.environ.get("CALL_LAUNCH") == "1":
    r = c.post("/api/dbv3_launch", json={})
    out["launch_status"] = r.status_code
    out["launch_body"] = r.get_json()
    r = c.post("/api/dbv3_send_image", data={}, content_type="multipart/form-data")
    out["send_status"] = r.status_code
print("RESULT" + json.dumps(out), flush=True)
os._exit(0)
'''


def _run(exe=None, call_launch=False):
    env = dict(os.environ, REPO=str(REPO), CALL_LAUNCH="1" if call_launch else "0")
    env.pop("CCPLOTTER_DBV3_EXE", None)
    if exe:
        env["CCPLOTTER_DBV3_EXE"] = exe
    r = subprocess.run([sys.executable, "-I", "-c", CHILD], env=env, cwd=str(REPO),
                       capture_output=True, text=True, timeout=300)
    lines = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    assert lines, f"child failed: rc={r.returncode}\n{r.stdout[-800:]}\n{r.stderr[-1500:]}"
    return json.loads(lines[-1][6:])


def test_unset_means_not_configured_and_404():
    out = _run(call_launch=True)
    assert out["status"]["exe_configured"] is False
    assert out["launch_status"] == 404 and "CCPLOTTER_DBV3_EXE" in out["launch_body"]["error"]
    assert out["send_status"] == 404


def test_set_to_a_missing_file_is_404():
    out = _run(exe=str(REPO / "no-such-DrawingBotV3.exe"), call_launch=True)
    assert out["status"]["exe_configured"] is False
    assert out["launch_status"] == 404


def test_set_to_an_existing_file_is_configured():
    out = _run(exe=sys.executable)        # any existing file; the launch route is not called
    assert out["status"]["exe_configured"] is True
