#!/usr/bin/env python3
"""
replay_harness.py -- byte-identity replay harness for the app. It runs a fixed set
of recorded requests through the dev server and diffs the results against a pinned
manifest, so a mechanical refactor (moving code between modules) can be proven
behaviour-identical by a green `verify` run.

Modes
-----
  record  - starts a dev server on :5001, runs every fixture TWICE (a
            determinism guard -- if the pair isn't identical, refuses to
            write a manifest and exits nonzero), then writes the manifest.
  verify  - starts a dev server on :5001, runs every fixture ONCE, and diffs
            the result against the pinned manifest field-by-field. Any
            mismatch is a nonzero exit with a readable per-field diff.

PENDING fixtures (see the PENDING_FIXTURES dict below): a fixture can be
known-failing on the committed tree for a reason outside this harness's
remit, with no decision yet on what to do about it. `record` writes an
explicit {"status": "pending-kai-ruling", ...} manifest entry for these
instead of comparison fields (single run, not double-run compared);
`verify` reports them loudly every run -- never green, never silently
absent -- until the entry is retired, re-fit, or promoted to a NEGATIVE
fixture.

NEGATIVE fixtures (see the NEGATIVE_FIXTURES dict below): a fixture whose
archived POST is supposed to trip a hard safety block, with that refusal
asserted as a permanent tripwire (currently: golden-tram-hongkong-lightrail,
which trips the reach safety check). `record` writes
{"status": "negative", "expected_http_status": ..., "expected_error_
substring": ...} instead of comparison fields; `verify` treats "the block
fires with a matching status + stable error substring" as green (labeled
NEGATIVE) and "the block is absent or fires with a different error" as a
loud FAIL -- if an out-of-reach job ever silently starts generating gcode
again, the harness goes red.

Pen context: every plotting fixture is pinned to a specific
pen_id (see FIXTURE_PEN_ID below), injected into its request body
regardless of which pen is currently ACTIVE in pen_profiles.json. Without
the pin, switching the active pen (e.g. for physical calibration) would
silently change fill_mm and therefore every traced fixture's output, with
no code change to attribute it to. check_pen_context()
preflights that the pinned pen_id still exists before running anything, and
refuses to run (naming the fixture + pen) rather than silently falling back
to the active one.

Manifest location: tests/golden_tests/replay_manifest.json (override with
--manifest PATH). One JSON file, keyed by fixture label, plus the git sha it
was recorded at.
Fixture paths below are relative to the repository root.

Fixture set:
  - tests/golden_tests/expected/alaska-737-jun14        (settings.json + source.png)
  - tests/golden_tests/expected/tram-hongkong-lightrail  (settings.json + source.png)
  - tests/golden_tests/fixtures/loco-472-trestle         (settings.json + source.png)
  - tests/golden_tests/fixtures/dotstorm-proxy-rivets-ballast
    (request.json; source is loco-472-trestle/source.png -- see that
    fixture's source_pointer.md, byte-identical source confirmed by sha256)
  - tests/golden_tests/fixtures/loco-lineart-isolation-trial
    (request.json = dotstorm-proxy-rivets-ballast's request + fidelity_isolation
    ="trial" + debug=false; same loco source. A flag-on variant that guards
    the isolation-gate candidate path.)
  - tests/golden_tests/fixtures/plane            (replay_request.json + source.png)
  - tests/golden_tests/fixtures/ship             (replay_request.json + source.png)
  - /api/calibration_plot (no image input; exercises the gcode-emission path
    directly)
  - /api/fan_probe (no image input; pure M-code text, no vpype SVG involved)

What gets captured per fixture
--------------------------------------------------
  - request:      {path, sha256} of the on-disk request/settings JSON template
                   (None for the two no-image endpoints; their body is {}).
  - http_status:  the raw HTTP status code.
  - gcode_sha256: full byte SHA-256 of the written .gcode file.
  - preview_svg_sha256_stripped: SHA-256 of the preview SVG with vpype's
    <dc:date>...</dc:date> wall-clock metadata stripped first (grep-confirmed
    the only source of embedded timestamps in preview.svg -- see
    tests/golden_tests/expected/README.md).
  - audit: the debug audit dicts (trace_info incl. _TRACE_AUDIT/_LINEART_AUDIT
    contents, debug_metrics, sketch_audit when present) as their own
    canonicalized (sorted-key, float-rounded) block, both raw (for readable
    diffs) and as audit_json_sha256 (quick pass/fail).
  - warnings: response['warnings'] verbatim (level/msg/tier per item).
  - response_metadata: every other top-level response field EXCEPT the
    volatile ones (elapsed_s, job_id, cached -- see ignored_fields in the
    manifest) and except item_bboxes AND item_geometry, both of which are
    keyed by the upload id (random per upload) and are renormalized to a
    plain list in item/request order before comparison so a fresh upload id
    never causes a false mismatch. Both dicts are built inside the SAME
    `for idx, item in enumerate(items)` loop in ccplotter/composite.py
    (item_geometry[str(item["id"])] = {...} and layer_of[L] = str(item["id"])
    with L = idx + 1, which item_bboxes iterates via layer_of.items()) --
    plain dict insertion order in increasing idx, so both are guaranteed to
    be in request-item order.

Determinism guard: `record` runs each fixture twice against the SAME server
lifecycle and refuses to write a manifest if the pair differs in any of the
above fields (a mismatch means the pipeline itself is nondeterministic, not a
false positive from re-uploading with a new id -- item_bboxes and item_geometry are already
order-normalized).

Server lifecycle: this script owns starting and stopping ONE dev server for
the whole run (not one per fixture). It guards against a stacked server:
before starting,
it checks :5001, and if something is already listening there it identifies
the PID via `netstat -ano` and kills it (dev-server-only guard; the frozen
ccPlotter.exe binds a DYNAMIC port via desktop/ccplotter_app.py's
free_port(), never :5001, so this can never touch a running ccPlotter.exe).
If the port cannot be freed, the harness refuses to run. The server is always
stopped in a finally block, even on failure/exception.

Usage:
    python scripts/replay_harness.py record [--manifest PATH]
    python scripts/replay_harness.py verify [--manifest PATH]
"""
import argparse
import atexit
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.ssim_fixture import compute_ssim
# Default stays 5001. The override exists because a port can be held by
# something this script CANNOT free (for example a zombie python process
# that `taskkill` reports as not running and Stop-Process refuses to kill):
# ensure_port_free() correctly refuses to run in that case, and the harness
# is unusable until the machine is rebooted. Overriding the port preserves the
# stacked-server guard (it still refuses if the CHOSEN port cannot be freed)
# while routing around a port that is wedged rather than merely occupied.
PORT = int(os.environ.get("CCPLOTTER_HARNESS_PORT", "5001"))
BASE_URL = f"http://127.0.0.1:{PORT}"
DEFAULT_MANIFEST = ROOT / "tests" / "golden_tests" / "replay_manifest.json"
SERVER_LOG = ROOT / "work" / "_replay_harness_server.log"

# Hermetic calibration env for the harness's OWN dev server.
# pen_profiles.json and calibration.json resolve from the runtime home (see
# CCPLOTTER_HOME below), never from the CWD; vpype.toml is still CWD-relative.
# Reading whatever pen_profiles.json/calibration.json sit in the repo root
# (gitignored, mutable, machine-local state) would make the harness assert
# "byte-identical to whatever calibration happens to be lying around" instead
# of "byte-identical to the calibration the fixtures were recorded under".
# HARNESS_ENV_DIR pins the exact energel_05 profile + global calibration
# fields in play when the golden manifest was recorded (its recorded_at); see
# the provenance notes inside its two JSON files. It is tracked in git so the
# harness can never silently depend on ambient repo-root or %LOCALAPPDATA%
# state.
HARNESS_ENV_DIR = ROOT / "scripts" / "harness_env"
# The harness process ITSELF imports ccplotter (e.g. run_process_fixture ->
# ccplotter.mask -> ccplotter.config), and config resolves pen_profiles.json /
# calibration.json from the runtime home, never the CWD. Pin the explicit opt-in
# here (assignment, not setdefault: an ambient value must not leak into a hermetic
# run) so the in-process import and the spawned dev server see the SAME env.
os.environ["CCPLOTTER_HOME"] = str(HARNESS_ENV_DIR)
# The harness's own server process resolves WORK_DIR relative to ITS cwd
# (HARNESS_ENV_DIR), so every generated file this script reads back lives
# under HARNESS_ENV_DIR/"work", not ROOT/"work" -- see the comment in
# start_server() for why WORK_DIR is deliberately left unoverridden.
WORK_DIR_FOR_HARNESS = HARNESS_ENV_DIR / "work"

IGNORED_FIELDS = {"elapsed_s", "job_id", "cached"}
DC_DATE_RE = re.compile(rb"<dc:date>.*?</dc:date>", re.DOTALL)

# PENDING fixtures: known-failing on the committed tree for a reason that is
# NOT this harness's concern to resolve (e.g. a deliberate safety change),
# with NO decision yet on what to do about it. NEVER recorded/verified as a
# normal pass/fail pair; `record` writes an explicit {"status":
# "pending-kai-ruling", ...} placeholder instead of comparison fields, and
# `verify` reports them loudly every run -- never green, never silently
# absent -- until the fixture is retired, re-fit, or promoted to a real
# (expected-failure) NEGATIVE fixture (see NEGATIVE_FIXTURES below).
PENDING_FIXTURES = {}

# NEGATIVE fixtures: a fixture's archived POST is SUPPOSED
# to trip a hard safety block, and that block is asserted as a permanent
# tripwire -- not compared byte-for-byte (the gcode/preview/audit fields
# don't exist when generation is refused), but the REFUSAL ITSELF is the
# thing under test. `record` writes {"status": "negative", ...} with the
# expected http_status and a STABLE substring of the error (not the full
# message byte-for-byte -- deliberately robust to incidental wording/
# coordinate drift elsewhere in the message; only the identifying phrase for
# THIS safety mechanism must survive). `verify` treats "the block fires with
# a matching status + substring" as green (labeled NEGATIVE); the block being
# ABSENT (unexpected success) or firing with a DIFFERENT error is a loud
# FAIL -- this is the actual safety property: if an out-of-reach job ever
# silently starts generating gcode again, the harness goes red.
#
# golden-tram-hongkong-lightrail is registered because its archived
# placement draws up to Y257.5 in an X band, which the plotter_2003_v32
# envelope (a rectangle capping Y at 254.5) rejects. That is an accepted
# trade for the envelope's full B5-plus width. The reach.py hard-block
# validator itself is independent of which fixture sits outside the active
# envelope; to move the tripwire, edit the dict below.
NEGATIVE_FIXTURES = {
    "golden-tram-hongkong-lightrail": {
        "reason": "Archived settings.json places the drawing so its bbox "
                  "reaches Y257.5 at X~84.7; plotter_2002's stepped polygon "
                  "allowed Y up to 264.5 in that X band (drawing fit), but "
                  "the Kai-ratified plotter_2003_v32 rectangle caps Y at "
                  "254.5 everywhere -- an accepted trade for full B5-plus "
                  "width (Kai, 2026-07-16: \"we likely are losing reach in "
                  "some places, but more than make up for it...\").",
        "gate": "docs/history/REVIEW_LOG.md 2026-07-16 (evening, V3.2 manifest drift "
               "attribution); Kai ruling on the V3.2 envelope trade "
               "(2026-07-16).",
        "expected_http_status": 500,
        # Stable identifying substring only -- NOT the full message (line
        # numbers / coordinates are incidental and may shift with unrelated
        # geometry changes). This phrase is composite.py's own literal
        # wrapper text around the reach.py raise; if it ever stops appearing
        # on this fixture, the safety block is gone and verify must go red.
        "expected_error_substring": "reach safety check failed",
    },
}

# Pen context. Every fixture below was ORIGINALLY recorded with no explicit
# pen selection in its request JSON -- i.e. against whatever pen happened to
# be ACTIVE in pen_profiles.json at capture time, which for every fixture
# here was "energel_05". Switching the live active pen afterwards would change
# stroke_count/fill_px/pen_mm with no code change to attribute it to.
# ccplotter/routes/core.py and routes/pen.py accept an explicit pen_id
# request field (ccplotter/cal.py resolve_pen_profile()) that pins the
# resolved pen profile regardless of what's currently active; omitting it
# uses the ambient active pen (additive, no semantics change for any other
# caller). FIXTURE_PEN_ID below is that pin,
# injected into the POST body by run_process_fixture /
# run_calibration_plot_fixture for every fixture that plots (fan_probe is
# pure M-code and never resolves a pen, so it has none). This makes every
# fixture reproducible byte-for-byte no matter which pen is selected in
# the live app, and -- per check_pen_context() below -- the harness refuses
# to run at all (naming the fixture and the missing pen_id) if that pinned
# profile is ever renamed/deleted rather than silently drifting onto
# whichever pen happens to be active.
FIXTURE_PEN_ID = "energel_05"

FIXTURES = [
    {
        "label": "golden-alaska-737-jun14",
        "kind": "process",
        "request_path": "tests/golden_tests/expected/alaska-737-jun14/settings.json",
        "sources": ["tests/golden_tests/expected/alaska-737-jun14/source.png"],
        "ssim_reference": "tests/golden_tests/expected/alaska-737-jun14/preview.png",
        "ssim_width": 1500,
        "ssim_min": 0.999,
        "pen_id": FIXTURE_PEN_ID,
    },
    {
        "label": "golden-tram-hongkong-lightrail",
        "kind": "process",
        "request_path": "tests/golden_tests/expected/tram-hongkong-lightrail/settings.json",
        "sources": ["tests/golden_tests/expected/tram-hongkong-lightrail/source.png"],
        "pen_id": FIXTURE_PEN_ID,
    },
    {
        "label": "loco-472-trestle",
        "kind": "process",
        "request_path": "tests/golden_tests/fixtures/loco-472-trestle/settings.json",
        "sources": ["tests/golden_tests/fixtures/loco-472-trestle/source.png"],
        "pen_id": FIXTURE_PEN_ID,
    },
    {
        "label": "dotstorm-proxy-rivets-ballast",
        "kind": "process",
        "request_path": "tests/golden_tests/fixtures/dotstorm-proxy-rivets-ballast/request.json",
        # Byte-identical to loco-472-trestle/source.png -- see this fixture's
        # own source_pointer.md (sha256-verified, not inferred).
        "sources": ["tests/golden_tests/fixtures/loco-472-trestle/source.png"],
        "pen_id": FIXTURE_PEN_ID,
    },
    {
        # Flag-on variant fixture (fidelity_isolation=trial) on the loco
        # line-art request. It gives the isolation-gate candidate path
        # (_component_fidelity_detail's opt-in pre-filter removal) its own
        # byte-identity guard. Same source as dotstorm-proxy-rivets-ballast --
        # see this fixture's own source_pointer.md.
        "label": "loco-lineart-isolation-trial",
        "kind": "process",
        "request_path": "tests/golden_tests/fixtures/loco-lineart-isolation-trial/request.json",
        "sources": ["tests/golden_tests/fixtures/loco-472-trestle/source.png"],
        "pen_id": FIXTURE_PEN_ID,
    },
    {
        # Height-limited golden fixture. The other process fixtures above
        # are width-limited -- none exercises the min(cellW/nw, cellH/nh)*0.9
        # height branch of composite.py:651 / _preflight's mm/px fit.
        # Synthetic portrait raster (800x1400px, pure line art, no filled
        # regions) on a wide-ish 220x160mm sheet: usable_h/nh=0.1 <
        # usable_w/nw=0.25, a 2.5x margin between branches. See
        # tests/test_height_fixture_preflight.py.
        "label": "synth-tallframe-height-limited",
        "kind": "process",
        "request_path": "tests/golden_tests/fixtures/synth-tallframe-height-limited/request.json",
        "sources": ["tests/golden_tests/fixtures/synth-tallframe-height-limited/source.png"],
        "pen_id": FIXTURE_PEN_ID,
    },
    {
        "label": "plane",
        "kind": "process",
        "request_path": "tests/golden_tests/fixtures/plane/replay_request.json",
        "sources": ["tests/golden_tests/fixtures/plane/source.png"],
        "pen_id": FIXTURE_PEN_ID,
    },
    {
        "label": "ship",
        "kind": "process",
        "request_path": "tests/golden_tests/fixtures/ship/replay_request.json",
        "sources": ["tests/golden_tests/fixtures/ship/source.png"],
        "pen_id": FIXTURE_PEN_ID,
    },
    {
        # bwphoto recipe at the default bw_xdog_side=native. The request
        # carries no bw_xdog_side key, so the default itself is under test. See the fixture README.
        "label": "bwphoto-native-default",
        "kind": "process",
        "request_path": "tests/golden_tests/fixtures/bwphoto-native-default/request.json",
        "sources": ["tests/fixtures/twotone/dahlia.jpg"],
        "pen_id": FIXTURE_PEN_ID,
    },
    {
        "label": "calibration_plot",
        "kind": "calibration_plot",
        "request_body": {},
        "pen_id": FIXTURE_PEN_ID,
    },
    {
        "label": "fan_probe",
        "kind": "fan_probe",
        "request_body": {},
        # No pen_id: /api/fan_probe is pure M-code, never resolves a pen
        # profile (see its route docstring in ccplotter/routes/pen.py).
    },
]


# ── helpers ──────────────────────────────────────────────────────────────

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# /api/process stamps one per-run "; ccplotter-run: id=... provenance=..." comment
# into the gcode header (ccplotter/runs.py BACKLINK_PREFIX; a test pins the two equal).
# A per-run id is nondeterministic by design, so the gcode hash EXCLUDES exactly that
# line; every other byte is still compared.
RUN_BACKLINK_PREFIX = b"; ccplotter-run: "


def strip_run_backlink(b: bytes) -> bytes:
    return b"".join(ln for ln in b.splitlines(keepends=True)
                    if not ln.startswith(RUN_BACKLINK_PREFIX))


def strip_dc_date(svg_bytes: bytes) -> bytes:
    return DC_DATE_RE.sub(b"<dc:date/>", svg_bytes)


def round_floats(obj, nd=6):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, nd) for v in obj]
    return obj


def canonical_json(obj) -> str:
    return json.dumps(round_floats(obj), sort_keys=True, ensure_ascii=True,
                       separators=(",", ":"), default=str)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        cwd=str(ROOT)).decode().strip()
    except Exception:
        return "unknown"


# ── server lifecycle ────────────────────────────────────────────────────

def _is_listening(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def _find_pid_on_port(port: int):
    try:
        out = subprocess.check_output(["netstat", "-ano"], text=True, errors="ignore")
    except Exception:
        return None
    needle = f":{port} "
    for line in out.splitlines():
        if "LISTENING" in line and needle in line:
            parts = line.split()
            if parts:
                return parts[-1]
    return None


# -- exclusive harness lock ----------------------------------------------
# Two concurrent verifies (two workers, or a worker and a person) would RACE: the second
# one's ensure_port_free() would taskkill the first one's live server mid-run, killing
# its first fixture with a connection reset. So the
# harness takes an exclusive lock file BEFORE touching the port and holds it until its
# server is stopped. A held lock, or a port that is already bound, is a loud refusal,
# never a kill. Killing a stale listener is opt-in only:
# CCPLOTTER_HARNESS_KILL_STALE=1 (for a verified-dead leftover server).

class HarnessBusy(RuntimeError):
    """Another harness run holds the lock, or the port is bound by someone else."""


def harness_lock_path(port: int = None) -> Path:
    return ROOT / "work" / f"_replay_harness_{PORT if port is None else port}.lock"


def _pid_alive(pid: int) -> bool:
    try:
        out = subprocess.check_output(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"],
                                      text=True, errors="ignore",
                                      stderr=subprocess.DEVNULL)
    except Exception:
        return True      # cannot tell: assume alive (refuse rather than race)
    return str(int(pid)) in out.split()


def _read_lock(path: Path):
    """(pid or None, 'start time' string) from a lock file; tolerant of garbage."""
    try:
        parts = Path(path).read_text().split(None, 1)
        return int(parts[0]), (parts[1].strip() if len(parts) > 1 else "?")
    except Exception:
        return None, "?"


def acquire_harness_lock(path: Path = None) -> Path:
    """Create the lock file exclusively (O_EXCL). A lock whose recorded PID is dead is
    reclaimed once; a live holder -> HarnessBusy with its PID and start time."""
    path = Path(path) if path is not None else harness_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (0, 1):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder, when = _read_lock(path)
            if attempt == 0 and holder is not None and not _pid_alive(holder):
                print(f"[harness] stale lock {path} (holder PID {holder} is gone) -- reclaiming")
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            raise HarnessBusy(
                f"REFUSING: another replay harness run holds {path} (PID {holder}, started "
                f"{when}). Two concurrent verifies race for :{PORT} and kill each other's "
                f"server. Wait for it to finish, or delete the lock only if that PID is "
                f"verifiably dead.")
        os.write(fd, f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}".encode())
        os.close(fd)
        atexit.register(release_harness_lock, path, os.getpid())
        return path
    raise HarnessBusy(f"REFUSING: could not take {path}")


def release_harness_lock(path: Path, owner_pid: int = None) -> None:
    """Remove the lock iff WE own it (never delete another run's lock)."""
    holder, _ = _read_lock(path)
    if holder is not None and holder == (owner_pid if owner_pid is not None else os.getpid()):
        try:
            Path(path).unlink()
        except OSError:
            pass


def ensure_port_free(port: int, timeout: float = 10.0):
    """Stacked-server guard. Kills a stale dev-server listener on
    `port` if present. Never touches ccPlotter.exe: the frozen app binds a
    dynamic port (desktop/ccplotter_app.py free_port()), never :5001."""
    if not _is_listening(port):
        return
    pid = _find_pid_on_port(port)
    if os.environ.get("CCPLOTTER_HARNESS_KILL_STALE") != "1":
        raise HarnessBusy(
            f"REFUSING: :{port} is already bound (PID {pid or '?'}). The harness no longer "
            f"kills a listener it did not start -- it may be another run's live "
            f"server. If it is a verified-dead leftover, free it yourself "
            f"(taskkill /F /PID <pid>) or set CCPLOTTER_HARNESS_KILL_STALE=1.")
    if pid:
        print(f"[harness] stale listener on :{port} (PID {pid}) -- killing it")
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
    else:
        print(f"[harness] something is listening on :{port} but PID lookup failed")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _is_listening(port):
            return
        time.sleep(0.5)
    raise RuntimeError(f"refuse to run: could not free port {port} for the dev server")


def start_server():
    lock = acquire_harness_lock()
    try:
        proc = _start_server_locked()
    except BaseException:
        release_harness_lock(lock)
        raise
    proc._harness_lock = lock
    return proc


def _start_server_locked():
    ensure_port_free(PORT)
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    # vpype.toml is CONFIG_PATH in ccplotter/config.py -- also CWD-relative,
    # also loaded server-side (composite.py's vp.config_manager.load_config_file).
    # It supplies only the gwrite writer TEMPLATE (unit/vertical_flip/segment
    # strings + a placeholder default_values block composite.py immediately
    # overwrites with the resolved calibration) -- ROOT's copy is the single
    # source of truth for that template, so it's copied fresh into
    # HARNESS_ENV_DIR on every run rather than duplicated/tracked there, which
    # would risk silently drifting out of sync with ROOT/vpype.toml.
    shutil.copy2(ROOT / "vpype.toml", HARNESS_ENV_DIR / "vpype.toml")
    logf = open(SERVER_LOG, "wb")
    # Deliberately do NOT override WORK_DIR (ccplotter/config.py) via
    # CCPLOTTER_WORK_DIR here: it defaults to a CWD-relative Path("work"),
    # and vpype's gwrite writer embeds that path VERBATIM as a "; Source:
    # {filename}" gcode header comment (ccplotter/../vpype.toml
    # document_start). Pointing WORK_DIR at an absolute ROOT/work path was
    # tried first and made every gcode_sha256 mismatch even though the Z
    # geometry was byte-identical -- only that header comment's string
    # (relative "work\\layout.gcode" vs absolute "C:\\...\\work\\layout.gcode")
    # had changed. Leaving WORK_DIR at its relative default reproduces the
    # exact "work\\..." comment the golden manifest was recorded with (also
    # a relative WORK_DIR, cwd=ROOT at record time) -- it just now resolves
    # under HARNESS_ENV_DIR instead of ROOT, so every ROOT/"work" read below
    # is HARNESS_ENV_DIR/"work" instead (see WORK_DIR_FOR_HARNESS).
    # pen_profiles.json / calibration.json resolve from the RUNTIME HOME,
    # never the CWD. The harness's hermetic env is selected by the EXPLICIT
    # CCPLOTTER_HOME opt-in (cwd == home, so the CWD-shadow guard has nothing to
    # compare). Without this the server would read the live %LOCALAPPDATA% files.
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "app.py")], cwd=str(HARNESS_ENV_DIR),
        env=dict(os.environ, CCPLOTTER_HOME=str(HARNESS_ENV_DIR),
                 CCPLOTTER_DEV_HOST="127.0.0.1",
                 CCPLOTTER_NO_PRINTER="1"),  # harness server never dials the printer
        stdout=logf, stderr=subprocess.STDOUT)
    proc._logf = logf  # keep the handle alive / closable
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if proc.poll() is not None:
            logf.close()
            raise RuntimeError(f"dev server exited early (see {SERVER_LOG}); "
                                f"exit code {proc.returncode}")
        try:
            r = requests.get(BASE_URL + "/api/version", timeout=1)
            if r.status_code == 200:
                print(f"[harness] dev server up on :{PORT} (pid {proc.pid})")
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    stop_server(proc)
    raise RuntimeError(f"dev server did not respond on :{PORT} within 30s (see {SERVER_LOG})")


def stop_server(proc):
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        logf = getattr(proc, "_logf", None)
        if logf:
            try:
                logf.close()
            except Exception:
                pass
    # Confirm we actually freed the port (never leave a server running, even
    # on assertion failure -- caller invokes this from finally).
    deadline = time.time() + 5.0
    while time.time() < deadline and _is_listening(PORT):
        time.sleep(0.25)
    if _is_listening(PORT):
        print(f"[harness] WARNING: :{PORT} still answering after stop_server()")
    lock = getattr(proc, "_harness_lock", None)
    if lock:
        release_harness_lock(lock)        # only after the port is confirmed handled
    else:
        print(f"[harness] dev server stopped, :{PORT} free")


# ── fixture runners ─────────────────────────────────────────────────────

def upload(path: Path) -> str:
    with open(path, "rb") as f:
        r = requests.post(BASE_URL + "/api/add_drawing",
                           files={"file": (path.name, f)})
    r.raise_for_status()
    return r.json()["id"]


def _split_response(resp: dict):
    """Normalize + split a /api/process-shaped response into
    (audit, warnings, response_metadata)."""
    resp = dict(resp)
    if isinstance(resp.get("item_bboxes"), dict):
        # Keyed by the (random-per-upload) drawing id -- renormalize to a
        # plain order-preserved list so a fresh upload id never causes a
        # false mismatch (dict insertion order == items[] request order).
        resp["item_bboxes"] = list(resp["item_bboxes"].values())
    if isinstance(resp.get("item_geometry"), dict):
        # Same keyed-by-upload-id shape as item_bboxes, same fix: without the
        # normalization a fresh upload id would produce a false
        # DETERMINISM FAILURE in `record`, not real pipeline nondeterminism.
        # ccplotter/composite.py builds item_geometry[str(item["id"])] = {...}
        # inside the same idx-ordered loop that builds layer_of (which
        # item_bboxes is keyed through), so dict insertion order == items[]
        # request order here too; see this file's module docstring.
        #
        # SECOND layer of the same id-randomness: composite.py also ECHOES
        # the raw upload id back inside each entry's own
        # item_settings.id (L167) -- unlike item_bboxes, which never embeds
        # the id as a VALUE, only as a (now-discarded) dict key. Converting
        # the outer dict to a list is not enough; item_settings.id must be
        # blanked too, or a fresh upload id still produces a spurious
        # DETERMINISM FAILURE / false verify mismatch on this one field.
        items_list = list(resp["item_geometry"].values())
        for entry in items_list:
            if isinstance(entry, dict) and isinstance(entry.get("item_settings"), dict):
                if "id" in entry["item_settings"]:
                    entry["item_settings"]["id"] = "<upload-id, order-normalized>"
        resp["item_geometry"] = items_list
    audit_keys = ("trace_info", "debug_metrics", "sketch_audit")
    audit = {k: resp[k] for k in audit_keys if k in resp}
    warnings = resp.get("warnings", [])
    metadata = {k: v for k, v in resp.items()
                if k not in audit_keys and k != "warnings" and k not in IGNORED_FIELDS}
    return round_floats(audit), round_floats(warnings), round_floats(metadata)


def run_process_fixture(fx: dict) -> dict:
    req_path = fx["request_path"]
    raw = (ROOT / req_path).read_bytes()
    body = json.loads(raw)
    # Pen-pinning: inject the fixture's recorded pen context so
    # the result is reproducible no matter which pen is currently ACTIVE in
    # pen_profiles.json. The recorded manifest's `request.sha256` is over
    # the ORIGINAL on-disk file (unchanged -- see result["request"] below),
    # not this mutated in-memory body, so this injection doesn't perturb
    # that hash. See FIXTURE_PEN_ID / check_pen_context() above.
    if fx.get("pen_id"):
        body["pen_id"] = fx["pen_id"]
    sources = fx["sources"]
    items = body.get("items", [])
    if len(items) != len(sources):
        raise RuntimeError(f"{fx['label']}: {len(items)} items but {len(sources)} sources configured")
    # Native pixel dimensions of each source file, read directly off disk
    # BEFORE upload/any pipeline resize. The manifest's source_*_mm is the
    # traced SVG's ink bbox in mm (composite.py:658-661), not the raw
    # native [W,H] px that _preflight itself reads (trace_v3.py:1253, cv2.imread
    # shape). So this adds a separate top-level `source_native_px_wh` key on
    # the fixture's recorded result, in request-item order matching
    # `sources`, parallel to the existing `item_geometry`/`item_bboxes`
    # order-normalization. Not read by fixture_diff() (only http_status/
    # gcode_sha256/preview_svg_sha256_stripped/audit_json_sha256 plus
    # response_metadata/warnings/audit are compared), so it cannot change
    # verify's pass/fail.
    result_native_px_wh = []
    for src in sources:
        with Image.open(ROOT / src) as _im:
            result_native_px_wh.append(list(_im.size))
    for item, src in zip(items, sources):
        item["id"] = upload(ROOT / src)

    r = requests.post(BASE_URL + "/api/process", json=body, timeout=300)
    result = {
        "kind": "process",
        "request": {"path": req_path, "sha256": sha256_bytes(raw)},
        "http_status": r.status_code,
        "source_native_px_wh": result_native_px_wh,
    }
    try:
        resp = r.json()
    except Exception:
        resp = {}

    if r.status_code == 200 and resp.get("job_type") == "gcode":
        gcode_file = resp.get("gcode_file", "layout.gcode")
        preview_file = resp.get("preview_file", "layout.preview.svg")
        gpath, ppath = WORK_DIR_FOR_HARNESS / gcode_file, WORK_DIR_FOR_HARNESS / preview_file
        result["gcode_file"] = gcode_file
        result["gcode_sha256"] = sha256_bytes(strip_run_backlink(gpath.read_bytes()))
        result["preview_file"] = preview_file
        result["preview_svg_sha256_stripped"] = sha256_bytes(strip_dc_date(ppath.read_bytes()))
        if fx.get("ssim_reference"):
            with tempfile.TemporaryDirectory(prefix="ccplotter_ssim_") as td:
                rendered = Path(td) / "preview.png"
                from ccplotter.mask import _inkscape_exe
                inkscape = _inkscape_exe()
                if not inkscape:
                    raise RuntimeError("Inkscape is required for the SSIM fixture")
                reference = ROOT / fx["ssim_reference"]
                with Image.open(reference) as image:
                    reference_size = image.size
                if reference_size[0] != int(fx["ssim_width"]):
                    raise RuntimeError(
                        f"{fx['label']}: SSIM reference width {reference_size[0]} does not "
                        f"match configured width {fx['ssim_width']}"
                    )
                subprocess.run([inkscape, str(ppath), f"--export-filename={rendered}",
                                f"--export-width={reference_size[0]}",
                                "--export-background=white"],
                               check=True, capture_output=True, timeout=120)
                value = compute_ssim(reference, rendered)
            result["ssim"] = {"reference": fx["ssim_reference"], "value": round(value, 6),
                              "minimum": float(fx["ssim_min"])}
            if value < float(fx["ssim_min"]):
                raise RuntimeError(f"{fx['label']}: SSIM {value:.6f} below floor {fx['ssim_min']:.6f}")
    else:
        result["gcode_file"] = None
        result["gcode_sha256"] = None
        result["preview_file"] = None
        result["preview_svg_sha256_stripped"] = None
        result["ssim"] = None

    audit, warnings, metadata = _split_response(resp)
    result["audit"] = audit
    result["audit_json_sha256"] = sha256_bytes(canonical_json(audit).encode())
    result["warnings"] = warnings
    result["response_metadata"] = metadata
    return result


def run_calibration_plot_fixture(fx: dict) -> dict:
    body = dict(fx.get("request_body", {}))
    if fx.get("pen_id"):
        body["pen_id"] = fx["pen_id"]
    r = requests.post(BASE_URL + "/api/calibration_plot", json=body, timeout=60)
    result = {
        "kind": "calibration_plot",
        "request": {"path": None, "sha256": sha256_bytes(canonical_json(body).encode())},
        "http_status": r.status_code,
    }
    try:
        resp = r.json()
    except Exception:
        resp = {}
    if isinstance(resp.get("preview_url"), str):
        # api_calibration_plot appends a random `?t=<uuid hex>` cache-busting
        # stamp (uuid4().hex[:6]) -- volatile by design, not a real content
        # change. Strip the query string before comparing/storing.
        resp["preview_url"] = resp["preview_url"].split("?")[0]
    if r.status_code == 200 and resp.get("ok"):
        gfile = resp.get("file")
        result["gcode_file"] = gfile
        result["gcode_sha256"] = sha256_bytes((WORK_DIR_FOR_HARNESS / gfile).read_bytes())
        pfile = resp.get("preview")
        if pfile:
            result["preview_file"] = pfile
            result["preview_svg_sha256_stripped"] = sha256_bytes(
                strip_dc_date((WORK_DIR_FOR_HARNESS / pfile).read_bytes()))
        else:
            result["preview_file"] = None
            result["preview_svg_sha256_stripped"] = None
    else:
        result["gcode_file"] = None
        result["gcode_sha256"] = None
        result["preview_file"] = None
        result["preview_svg_sha256_stripped"] = None
    _, _, metadata = _split_response(resp)
    result["audit"] = {}
    result["audit_json_sha256"] = sha256_bytes(canonical_json({}).encode())
    result["warnings"] = []
    result["response_metadata"] = metadata
    return result


def run_fan_probe_fixture(fx: dict) -> dict:
    body = fx.get("request_body", {})
    r = requests.post(BASE_URL + "/api/fan_probe", json=body, timeout=30)
    result = {
        "kind": "fan_probe",
        "request": {"path": None, "sha256": sha256_bytes(canonical_json(body).encode())},
        "http_status": r.status_code,
    }
    try:
        resp = r.json()
    except Exception:
        resp = {}
    if r.status_code == 200 and resp.get("ok"):
        gfile = resp.get("file")
        result["gcode_file"] = gfile
        result["gcode_sha256"] = sha256_bytes((WORK_DIR_FOR_HARNESS / gfile).read_bytes())
    else:
        result["gcode_file"] = None
        result["gcode_sha256"] = None
    result["preview_file"] = None
    result["preview_svg_sha256_stripped"] = None
    _, _, metadata = _split_response(resp)
    result["audit"] = {}
    result["audit_json_sha256"] = sha256_bytes(canonical_json({}).encode())
    result["warnings"] = []
    result["response_metadata"] = metadata
    return result


FIXTURE_RUNNERS = {
    "process": run_process_fixture,
    "calibration_plot": run_calibration_plot_fixture,
    "fan_probe": run_fan_probe_fixture,
}


def run_fixture(fx: dict) -> dict:
    return FIXTURE_RUNNERS[fx["kind"]](fx)


def check_pen_context():
    """Preflight: before running ANY fixture,
    confirm every fixture's declared pen_id still exists in the live
    pen_profiles.json the dev server is about to read. Fails LOUDLY, naming
    every offending fixture and its missing pen_id, and refuses to run at
    all -- never silently falls back to whichever pen happens to be
    ACTIVE (that silent fallback would change stroke_count/fill_px/pen_mm
    with no code change to attribute it to)."""
    r = requests.get(BASE_URL + "/api/pen_profiles", timeout=10)
    r.raise_for_status()
    known = set((r.json() or {}).get("profiles", {}))
    missing = [(fx["label"], fx["pen_id"]) for fx in FIXTURES
               if fx.get("pen_id") and fx["pen_id"] not in known]
    if missing:
        lines = "\n".join(f"    {label}: pen_id={pid!r}" for label, pid in missing)
        raise RuntimeError(
            "refuse to run: the following fixture(s) declare a pen_id that "
            "no longer exists in pen_profiles.json (renamed or deleted?):\n"
            f"{lines}\n"
            "Restore that profile (id must match exactly) or deliberately "
            "update the fixture's pen_id and re-record -- this harness will "
            "never silently fall back to whichever pen is currently ACTIVE."
        )


# ── diffing ──────────────────────────────────────────────────────────────

def _diff_values(path: str, a, b, out: list):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: MISSING in first -> {b[k]!r}")
            elif k not in b:
                out.append(f"{path}.{k}: {a[k]!r} -> MISSING in second")
            else:
                _diff_values(f"{path}.{k}", a[k], b[k], out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} -> {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff_values(f"{path}[{i}]", x, y, out)
    else:
        if a != b:
            out.append(f"{path}: {a!r} -> {b!r}")


def _ssim_diff(label: str, recorded, current, *, require_exact_value: bool) -> list:
    """Compare SSIM's stable contract, and (during verify) its score floor.

    A record run still compares the complete payload exactly as its
    determinism guard.  A verify run deliberately does not require a freshly
    rasterized SSIM score to equal the old measurement: it requires the same
    reference/floor contract and a score at or above that floor.
    """
    if require_exact_value:
        if recorded != current:
            return [f"{label}.ssim: {recorded!r} -> {current!r}"]
        return []

    if recorded is None or current is None:
        if recorded != current:
            return [f"{label}.ssim contract mismatch: {recorded!r} -> {current!r}"]
        return []
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return [f"{label}.ssim contract mismatch: expected mappings, got "
                f"{recorded!r} -> {current!r}"]

    contract_fields = ("reference", "minimum")
    contract_diffs = [
        f"{field}: {recorded.get(field)!r} -> {current.get(field)!r}"
        for field in contract_fields
        if recorded.get(field) != current.get(field)
    ]
    if contract_diffs:
        return [f"{label}.ssim contract mismatch: " + "; ".join(contract_diffs)]

    try:
        value = float(current["value"])
        floor = float(recorded["minimum"])
    except (KeyError, TypeError, ValueError):
        return [f"{label}.ssim contract mismatch: missing or non-numeric "
                f"value/minimum in {current!r}"]
    if value < floor:
        return [f"{label}.ssim below floor: measured {value:.6f} < required {floor:.6f}"]
    return []


def fixture_diff(label: str, a: dict, b: dict, *, require_exact_ssim_value: bool = True) -> list:
    out = []
    for key in ("http_status", "gcode_sha256", "preview_svg_sha256_stripped", "audit_json_sha256"):
        if a.get(key) != b.get(key):
            out.append(f"{label}.{key}: {a.get(key)!r} -> {b.get(key)!r}")
    out.extend(_ssim_diff(label, a.get("ssim"), b.get("ssim"),
                          require_exact_value=require_exact_ssim_value))
    # response_metadata.coverage_audit is advisory-only paper-coverage prediction
    # (ccplotter/coverage_audit.py): it reads the already-generated preview and
    # adds no geometry, so it carries no information about a regression in the
    # actual plot. It also legitimately changes with pen calibration (laid_mm),
    # independent of any code change under test -- comparing it byte-for-byte
    # drowns the harness's real signal in advisory noise (7 of 9
    # fixtures flagged MISSING -> None on this key alone). Excluded here, not
    # from the recorded manifest -- record still captures it for humans reading
    # the fixture file, verify just does not gate on it.
    _meta_a = dict(a.get("response_metadata") or {})
    _meta_b = dict(b.get("response_metadata") or {})
    _meta_a.pop("coverage_audit", None)
    _meta_b.pop("coverage_audit", None)
    # response_metadata.realised_pitch is a READOUT added after the manifest
    # was recorded: it multiplies numbers the pipeline already holds and changes no stroke or
    # gcode byte (gcode_sha256 / preview / audit above still gate). Excluded here, not from the
    # manifest and not re-recorded, exactly like coverage_audit; absent from older manifests.
    # COMPLETE LIST of response_metadata keys excluded from the comparison: coverage_audit and
    # realised_pitch. Both pops are removed at the next re-record, when the manifest itself
    # will carry them.
    # will carry them.
    _meta_a.pop("realised_pitch", None)
    _meta_b.pop("realised_pitch", None)
    _diff_values(f"{label}.response_metadata", _meta_a, _meta_b, out)
    _diff_values(f"{label}.warnings", a.get("warnings", []), b.get("warnings", []), out)
    _diff_values(f"{label}.audit", a.get("audit", {}), b.get("audit", {}), out)
    return out


# ── commands ─────────────────────────────────────────────────────────────

def _negative_observed_error(result: dict):
    return (result.get("response_metadata") or {}).get("error")


def _negative_matches(result: dict, spec: dict):
    """True if `result` (a run_fixture output) matches a NEGATIVE fixture's
    expected outcome: the expected HTTP status AND the stable identifying
    substring present in the observed error. Deliberately NOT a byte-exact
    comparison -- incidental wording/coordinates in the message may drift
    without the safety property itself changing."""
    if result.get("http_status") != spec["expected_http_status"]:
        return False
    err = _negative_observed_error(result) or ""
    return spec["expected_error_substring"] in err


def cmd_record(manifest_path: Path, only: str = None) -> int:
    """`only` = LABEL: record just that fixture (same run-twice determinism check) and merge
    its entry into the EXISTING manifest. Every other entry, and the top-level recorded_at /
    app_sha / ignored_fields, are left exactly as they were; a .bak-before-only-<label> copy of
    the manifest is written first. Without `only`, behaviour is unchanged (re-record all)."""
    to_run = FIXTURES
    existing = None
    if only is not None:
        to_run = [fx for fx in FIXTURES if fx["label"] == only]
        if not to_run:
            print(f"[record] --only {only!r}: no such fixture in FIXTURES")
            return 2
        if not manifest_path.exists():
            print(f"[record] --only needs an existing manifest at {manifest_path}")
            return 2
        existing_text = manifest_path.read_text()
        existing = json.loads(existing_text)
        # Re-serialise with the EXISTING file's indent so untouched entries stay byte-identical.
        _lines = existing_text.splitlines()
        existing_indent = len(_lines[1]) - len(_lines[1].lstrip(" ")) if len(_lines) > 1 else 1
        if only in existing["fixtures"]:
            print(f"[record] --only {only!r}: already in the manifest; refusing to overwrite it")
            return 2
    proc = start_server()
    try:
        check_pen_context()
        fixtures_out = {}
        for fx in to_run:
            label = fx["label"]
            if label in PENDING_FIXTURES:
                pend = PENDING_FIXTURES[label]
                print(f"[record] {label}: PENDING fixture -- running once to confirm "
                      f"the known-failure shape, NOT double-run compared, NOT recorded green.")
                r1 = run_fixture(fx)
                if r1.get("http_status") == 200:
                    print(f"[record] {label}: WARNING -- this PENDING fixture now returns "
                          f"HTTP 200 (was expected to fail). Recording as pending anyway "
                          f"per the gate; review it, it may mean the "
                          f"situation changed.")
                fixtures_out[label] = {
                    "status": "pending-kai-ruling",
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "last_observed_http_status": r1.get("http_status"),
                    "last_observed_error": (r1.get("response_metadata") or {}).get("error"),
                    **pend,
                }
                print(f"[record] {label}: recorded as PENDING (status=pending-kai-ruling)")
                continue
            if label in NEGATIVE_FIXTURES:
                neg = NEGATIVE_FIXTURES[label]
                print(f"[record] {label}: NEGATIVE fixture -- running once to confirm "
                      f"the expected-error shape (status={neg['expected_http_status']}, "
                      f"substring={neg['expected_error_substring']!r}), NOT double-run "
                      f"compared byte-for-byte.")
                r1 = run_fixture(fx)
                if not _negative_matches(r1, neg):
                    print(f"[record] {label}: REFUSING to record -- the expected safety "
                          f"block did NOT fire as specified. observed http_status="
                          f"{r1.get('http_status')!r}, observed error="
                          f"{_negative_observed_error(r1)!r}. Fix the NEGATIVE_FIXTURES "
                          f"spec or investigate why the block changed before recording.")
                    return 2
                fixtures_out[label] = {
                    "status": "negative",
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "expected_http_status": neg["expected_http_status"],
                    "expected_error_substring": neg["expected_error_substring"],
                    "last_observed_error": _negative_observed_error(r1),
                    "reason": neg.get("reason"), "gate": neg.get("gate"),
                    "attribution_audit": neg.get("attribution_audit"),
                }
                print(f"[record] {label}: recorded as NEGATIVE (status=negative, "
                      f"block confirmed firing as expected)")
                continue
            print(f"[record] {label}: run 1/2 ...")
            r1 = run_fixture(fx)
            print(f"[record] {label}: run 2/2 ...")
            r2 = run_fixture(fx)
            diffs = fixture_diff(label, r1, r2)
            if diffs:
                print(f"[record] DETERMINISM FAILURE for {label} -- refusing to write a manifest:")
                for d in diffs:
                    print(f"    {d}")
                return 2
            fixtures_out[label] = r1
            print(f"[record] {label}: OK (byte-identical across 2 runs)")
        manifest = {
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "app_sha": git_sha(),
            "fixtures": fixtures_out,
            "ignored_fields": sorted(IGNORED_FIELDS),
        }
        if existing is not None:
            bak = manifest_path.with_name(manifest_path.name + f".bak-before-only-{only}")
            bak.write_bytes(manifest_path.read_bytes())
            existing["fixtures"].update(fixtures_out)
            manifest = existing
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=(existing_indent if existing is not None else 1),
                                            sort_keys=True))
        print(f"[record] manifest written: {manifest_path} ({len(fixtures_out)} fixtures, "
              f"app_sha={manifest['app_sha']})")
        return 0
    finally:
        stop_server(proc)


def cmd_verify(manifest_path: Path) -> int:
    if not manifest_path.exists():
        print(f"[verify] no manifest at {manifest_path} -- run `record` first")
        return 2
    manifest = json.loads(manifest_path.read_text())
    proc = start_server()
    failures = 0
    pending = 0
    negatives_ok = 0
    try:
        check_pen_context()
        for fx in FIXTURES:
            label = fx["label"]
            pinned = manifest["fixtures"].get(label)
            if pinned is None:
                print(f"[verify] {label}: NOT IN MANIFEST -- new fixture since last record, skipping")
                continue
            if pinned.get("status") == "pending-kai-ruling" or label in PENDING_FIXTURES:
                pending += 1
                print(f"[verify] {label}: *** PENDING (ruling required -- {pinned.get('gate', 'no gate recorded')}) ***")
                print(f"    reason: {pinned.get('reason', '(no reason recorded)')}")
                print(f"    NOT compared byte-for-byte, NOT counted as pass or fail. "
                      f"See {pinned.get('attribution_audit', 'manifest-remediation/')}.")
                continue
            if pinned.get("status") == "negative" or label in NEGATIVE_FIXTURES:
                print(f"[verify] {label}: running (NEGATIVE fixture -- expecting the "
                      f"safety block to fire) ...")
                cur = run_fixture(fx)
                if _negative_matches(cur, pinned):
                    negatives_ok += 1
                    print(f"[verify] {label}: OK (NEGATIVE -- safety block fired as "
                          f"expected, http_status={pinned.get('expected_http_status')}, "
                          f"substring={pinned.get('expected_error_substring')!r} present)")
                else:
                    failures += 1
                    print(f"[verify] *** SAFETY TRIPWIRE FAILURE for {label} *** -- the "
                          f"expected block did NOT fire as specified:")
                    print(f"    expected: http_status={pinned.get('expected_http_status')}, "
                          f"error contains {pinned.get('expected_error_substring')!r}")
                    print(f"    observed: http_status={cur.get('http_status')!r}, "
                          f"error={_negative_observed_error(cur)!r}")
                    print(f"    This means an out-of-reach job may be generating gcode "
                          f"again -- investigate immediately, do not silence this fixture.")
                continue
            print(f"[verify] {label}: running ...")
            cur = run_fixture(fx)
            diffs = fixture_diff(label, pinned, cur, require_exact_ssim_value=False)
            if diffs:
                failures += 1
                print(f"[verify] MISMATCH for {label}:")
                for d in diffs:
                    print(f"    {d}")
            else:
                print(f"[verify] {label}: OK")
    finally:
        stop_server(proc)
    compared = len(manifest['fixtures']) - pending - negatives_ok
    if pending:
        print(f"[verify] *** {pending} fixture(s) PENDING a ruling -- see lines above, "
              f"never treat as green *** ")
    if negatives_ok:
        print(f"[verify] {negatives_ok} fixture(s) NEGATIVE-OK (safety block fired as expected)")
    if failures:
        print(f"[verify] FAILED: {failures}/{compared} compared fixture(s) mismatched or "
              f"failed a safety tripwire ({pending} additional PENDING, {negatives_ok} "
              f"NEGATIVE-OK; recorded at app_sha={manifest.get('app_sha')})")
        return 1
    print(f"[verify] PASSED: all {compared} compared fixtures byte-identical to manifest, "
          f"{negatives_ok} NEGATIVE-OK ({pending} additional PENDING; "
          f"recorded at app_sha={manifest.get('app_sha')})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["record", "verify"])
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--only", default=None, metavar="LABEL",
                    help="record: run and merge only this NEW fixture into the existing manifest")
    args = ap.parse_args()
    manifest_path = Path(args.manifest)
    try:
        if args.mode == "record":
            sys.exit(cmd_record(manifest_path, args.only))
        else:
            sys.exit(cmd_verify(manifest_path))
    except HarnessBusy as e:
        print(f"[harness] {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
