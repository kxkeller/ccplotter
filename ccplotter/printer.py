import json
import logging
import os
import time
import threading
from pathlib import Path

import websocket

from ccplotter.config import PRINTER_IP

# ── Printer API ───────────────────────────────────────────────────────────────

class Printer:
    def __init__(self, ip: str):
        self.ip = ip
        self._ws = None
        self._last_snap = {}
        self._lock = threading.Lock()
        self.connected = False

    def connect(self):
        if not self.ip:
            return   # no printer host configured (CCPLOTTER_PRINTER_HOST unset): never dial
        def _run():
            while True:
                try:
                    self._ws = websocket.WebSocketApp(
                        f"ws://{self.ip}/websocket",
                        on_message=self._on_message,
                        on_error=self._on_error,
                        on_close=self._on_close
                    )
                    self._ws.on_open = self._on_open
                    self.connected = True
                    self._ws.run_forever(ping_timeout=10)
                except Exception: pass
                self.connected = False
                time.sleep(5)
        threading.Thread(target=_run, daemon=True).start()

    def disconnect(self):
        if self._ws: self._ws.close()

    def _on_open(self, ws):
        # Request status every 2s
        def _poll():
            while self.connected:
                try: ws.send(self._make_pkt(0))
                except: break
                time.sleep(2)
        threading.Thread(target=_poll, daemon=True).start()

    def _on_message(self, ws, raw):
        try:
            d = json.loads(raw)
            if "Status" in d:
                with self._lock: self._last_snap = d["Status"]
        except Exception: pass

    def _on_error(self, ws, e): pass
    def _on_close(self, ws, code, msg): pass

    def _make_pkt(self, cmd: int, data: dict | None = None) -> str:
        return json.dumps({"Id": "", "Data": {"Cmd": cmd, "Data": data or {}, "RequestID": "", "MainboardID": "", "TimeStamp": int(time.time()*1000), "From": 1}})

    def get_snapshot(self) -> dict:
        with self._lock: return self._last_snap.copy()

    def get_z(self) -> float | None:
        c = self.get_snapshot().get("CurrenCoord", "")
        if not c: return None
        parts = c.split(",")
        return float(parts[2].strip()) if len(parts) >= 3 else None

    def refresh(self):
        if self._ws: self._ws.send(self._make_pkt(0))

    def start_print(self, filename: str) -> bool:
        # Cmd 1: Start print
        if not self._ws: return False
        self._ws.send(self._make_pkt(1, {"Name": filename}))
        return True

    def upload_file(self, local_path: Path) -> tuple[bool, str]:
        if not self.ip:
            return False, "No printer host configured (set CCPLOTTER_PRINTER_HOST)."
        import requests
        try:
            with open(local_path, 'rb') as f:
                r = requests.post(f"http://{self.ip}/upload", files={'file': (local_path.name, f)}, timeout=30)
            return r.status_code == 200, r.text
        except Exception as e:
            return False, str(e)

# No network at import. Constructing Printer opens nothing; the status
# websocket is dialed only by start_printer() (app.py __main__ / desktop launch)
# or the explicit /api/reconnect route calling printer.connect().
printer = Printer(PRINTER_IP)
_log = logging.getLogger("ccplotter.printer")
_START_LOCK = threading.Lock()
_started = False


def start_printer() -> bool:
    """App startup hook: connect the status websocket once. Idempotent.
    Does nothing (no connection) when CCPLOTTER_PRINTER_HOST is unset or
    CCPLOTTER_NO_PRINTER=1.
    Returns True if this call started it."""
    global _started
    if os.environ.get("CCPLOTTER_NO_PRINTER") == "1":
        _log.info("printer: NOT started (CCPLOTTER_NO_PRINTER=1)")
        return False   # replay harness / offline dev servers: never dial
    if not PRINTER_IP:
        _log.info("printer: NOT started (CCPLOTTER_PRINTER_HOST not set)")
        return False   # no printer host configured: never dial
    with _START_LOCK:
        if _started:
            return False
        _started = True
    printer.connect()
    _log.info("printer: status websocket started")
    return True
