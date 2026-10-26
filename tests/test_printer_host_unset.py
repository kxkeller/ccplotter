"""With CCPLOTTER_PRINTER_HOST unset the app makes no outbound connection attempt.

The child process patches every way the app could reach the network (the
websocket factory, socket.connect/connect_ex/create_connection, getaddrinfo, the
requests session) BEFORE importing the app, then: imports app, calls
start_printer(), loads the index page, uploads an image and runs a preview job
through the Flask test client, and finally calls every printer route. The parent
asserts that nothing was attempted and that each printer route refuses
immediately with a message naming CCPLOTTER_PRINTER_HOST.

Positive control: the same child with CCPLOTTER_PRINTER_HOST set must be seen
dialing (websocket factory called), so the detector can fail.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CHILD = r'''
import json, os, socket, sys, threading, time
sys.path.insert(0, os.environ["REPO"])
sys.path.insert(0, os.path.join(os.environ["REPO"], "tests"))
import _test_home; _test_home.use_temp_home()
os.chdir(os.environ["REPO"])
calls = {"factory": 0, "socket": 0, "dns": 0, "requests": 0, "printer_threads": 0}
import websocket
class FakeWS:
    def __init__(self, *a, **k): calls["factory"] += 1
    def run_forever(self, *a, **k): raise RuntimeError("fake: no network")
    def close(self): pass
websocket.WebSocketApp = FakeWS
def _block(name):
    def f(self, *a, **k):
        calls["socket"] += 1
        raise OSError("blocked by test: " + name)
    return f
socket.socket.connect = _block("connect")
socket.socket.connect_ex = _block("connect_ex")
def _cc(*a, **k):
    calls["socket"] += 1
    raise OSError("blocked by test: create_connection")
socket.create_connection = _cc
_gai = socket.getaddrinfo
def _g(host, *a, **k):
    if str(host) not in ("127.0.0.1", "localhost", "::1", ""):
        calls["dns"] += 1
    return _gai(host, *a, **k)
socket.getaddrinfo = _g
import requests.sessions
def _r(self, *a, **k):
    calls["requests"] += 1
    raise OSError("blocked by test: requests")
requests.sessions.Session.request = _r
_ts = threading.Thread.start
def _s(self, *a, **k):
    t = repr(getattr(self, "_target", None))
    if "Printer" in t or "printer" in t:
        calls["printer_threads"] += 1
    return _ts(self, *a, **k)
threading.Thread.start = _s

import app
started = app.start_printer()
out = {"started": started, "routes": {}}
if os.environ.get("EXERCISE") == "1":
    c = app.app.test_client()
    assert c.get("/").status_code == 200
    png = os.path.join(os.environ["REPO"], "tests", "golden_tests", "fixtures",
                       "synth-tallframe-height-limited", "source.png")
    with open(png, "rb") as f:
        r = c.post("/api/add_drawing", data={"file": (f, "source.png")}, content_type="multipart/form-data")
    assert r.status_code == 200, r.data[:300]
    req = json.load(open(os.path.join(os.environ["REPO"], "tests", "golden_tests", "fixtures",
                                      "synth-tallframe-height-limited", "request.json")))
    req["items"][0]["id"] = r.get_json()["id"]
    r = c.post("/api/process", json=req)
    out["preview_status"] = r.status_code
    out["preview_job_type"] = (r.get_json() or {}).get("job_type")
    for path, body in (("/api/send", {"gcode_file": "layout.gcode"}),
                       ("/api/gcode", {"cmd": "M84"}),
                       ("/api/fans_off", {}),
                       ("/api/calibrate_z", {}),
                       ("/api/reconnect", {})):
        t0 = time.time()
        r = c.post(path, json=body)
        out["routes"][path] = {"status": r.status_code, "secs": round(time.time() - t0, 2),
                               "body": r.get_json()}
    r = c.get("/api/status")
    out["status_code"] = r.status_code
out["calls"] = calls
print("RESULT" + json.dumps(out), flush=True)
os._exit(0)
'''


def _run(host=None, exercise=True):
    env = dict(os.environ, REPO=str(REPO), EXERCISE="1" if exercise else "0")
    env.pop("CCPLOTTER_PRINTER_HOST", None)
    env.pop("CCPLOTTER_NO_PRINTER", None)
    if host:
        env["CCPLOTTER_PRINTER_HOST"] = host
    r = subprocess.run([sys.executable, "-I", "-c", CHILD], env=env, cwd=str(REPO),
                       capture_output=True, text=True, timeout=600)
    lines = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    assert lines, f"child failed: rc={r.returncode}\n{r.stdout[-800:]}\n{r.stderr[-1500:]}"
    return json.loads(lines[-1][6:])


def test_unset_host_makes_no_outbound_attempt_through_startup_and_preview():
    out = _run()
    assert out["started"] is False
    assert out["preview_status"] == 200 and out["preview_job_type"] == "gcode", out
    assert out["calls"] == {"factory": 0, "socket": 0, "dns": 0, "requests": 0,
                            "printer_threads": 0}, out


def test_unset_host_every_printer_route_refuses_immediately_and_clearly():
    out = _run()
    assert set(out["routes"]) == {"/api/send", "/api/gcode", "/api/fans_off",
                                  "/api/calibrate_z", "/api/reconnect"}
    for path, r in out["routes"].items():
        assert r["status"] == 503, (path, r)
        assert r["secs"] < 3, (path, r)
        assert "CCPLOTTER_PRINTER_HOST" in json.dumps(r["body"]), (path, r)
    assert out["status_code"] == 200          # /api/status stays a local read
    assert out["calls"]["factory"] == 0 and out["calls"]["socket"] == 0, out


def test_positive_control_set_host_is_dialed():
    out = _run(host="192.0.2.1", exercise=False)   # RFC 5737 documentation address
    assert out["started"] is True
    # Thread.start is counted synchronously inside start_printer(), so this is not racy.
    assert out["calls"]["factory"] >= 1 or out["calls"]["printer_threads"] >= 1, out
