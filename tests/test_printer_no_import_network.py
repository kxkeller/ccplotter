"""Importing the routes / app module must not dial the printer.

Each case runs in a FRESH subprocess with the websocket factory, socket
connect and Thread.start patched BEFORE the import, and reports what was
touched as JSON. Positive controls prove the detector sees a real dial:
explicit connect() and start_printer() DO call the factory, and a variant with
connect() neutered is seen as silent (so the control can fail).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

CHILD = r'''
import json, os, socket, sys, threading, time
sys.path.insert(0, os.environ["REPO"])
sys.path.insert(0, os.path.join(os.environ["REPO"], "tests"))
import _test_home; _test_home.use_temp_home()
os.chdir(os.environ["REPO"])
calls = {"factory": 0, "socket": 0, "threads": 0, "tnames": []}
import websocket
class FakeWS:
    def __init__(self, *a, **k): calls["factory"] += 1
    def run_forever(self, *a, **k): raise RuntimeError("fake: no network")
    def close(self): pass
websocket.WebSocketApp = FakeWS
_oc = socket.socket.connect
def _c(self, *a, **k): calls["socket"] += 1; raise OSError("blocked by test")
socket.socket.connect = _c
socket.create_connection = lambda *a, **k: (calls.__setitem__("socket", calls["socket"] + 1), (_ for _ in ()).throw(OSError("blocked")))
_ts = threading.Thread.start
def _s(self, *a, **k): calls["threads"] += 1; calls["tnames"].append(repr(getattr(self, "_target", None)));  return _ts(self, *a, **k)
threading.Thread.start = _s
mode = os.environ["MODE"]
if mode == "import_routes":
    import ccplotter.routes.core
elif mode == "import_app":
    import app
else:
    import ccplotter.routes.core
    from ccplotter import printer as P
    base = dict(calls)
    if mode == "connect":
        P.printer.connect()
    elif mode == "start":
        import app
        assert app.start_printer() is True
        assert app.start_printer() is False   # idempotent
    elif mode == "start_optout":
        assert P.start_printer() is False
        time.sleep(0.5)
    elif mode == "connect_neutered":
        P.printer.connect = lambda: None
        P.printer.connect()
    deadline = time.time() + (0.5 if mode in ("start_optout", "connect_neutered") else 3)
    while time.time() < deadline and calls["factory"] == base["factory"]:
        time.sleep(0.05)
print("RESULT" + json.dumps(calls), flush=True)
os._exit(0)
'''


def _run(mode):
    # A host must be configured for start_printer()/connect() to dial at all; the import
    # cases then prove that even a configured host is not dialed at import time.
    # 192.0.2.1 is the RFC 5737 documentation address; the child's socket/factory are patched anyway.
    env = dict(os.environ, REPO=str(REPO), MODE=mode, CCPLOTTER_PRINTER_HOST="192.0.2.1")
    r = subprocess.run([sys.executable, "-I", "-c", CHILD], env=env, cwd=str(REPO),
                       capture_output=True, text=True, timeout=180)
    lines = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    assert lines, f"child failed ({mode}): rc={r.returncode}\n{r.stdout[-800:]}\n{r.stderr[-1500:]}"
    return json.loads(lines[-1][6:])


@pytest.mark.parametrize("mode", ["import_routes", "import_app"])
def test_import_opens_no_socket_no_factory(mode):
    c = _run(mode)
    assert c["factory"] == 0, c
    assert c["socket"] == 0, c


def test_import_routes_spawns_no_printer_thread():
    # Thread.start is patched before import; no Printer thread may be spawned.
    c = _run("import_routes")
    # Unrelated Popen reader threads (version.py's git probe) are allowed; any
    # thread whose target comes from the Printer is not.
    bad = [t for t in c["tnames"] if "Printer" in t or "printer" in t or "connect" in t]
    assert bad == [], c


def test_explicit_connect_calls_factory():
    c = _run("connect")
    assert c["factory"] >= 1, c


def test_startup_hook_calls_factory():
    c = _run("start")
    assert c["factory"] >= 1, c


def test_positive_control_detects_broken_lazy_start():
    # connect() neutered -> the same detector must see ZERO factory calls,
    # proving the controls above can fail.
    assert _run("connect_neutered")["factory"] == 0


def test_opt_out_env_blocks_startup_hook():
    # The replay harness sets CCPLOTTER_NO_PRINTER=1 on its app.py child.
    os.environ["CCPLOTTER_NO_PRINTER"] = "1"
    try:
        assert _run("start_optout")["factory"] == 0
    finally:
        del os.environ["CCPLOTTER_NO_PRINTER"]
