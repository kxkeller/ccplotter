"""Per-run request-body persistence (ccplotter/runs.py build_run_provenance).

Reproducing a plotted job requires its exact settings. Every gcode run leaves
<runtime home>/run_provenance/<run id>.json:
the request exactly as received, the post-recipe body, the resolved pen profile, the
calibration sha256 and the app commit/dirty. These tests pin the round trip that matters
and keep it honest:

  1. Run a job through the real route, read the stored body back, POST it again and get
     the IDENTICAL routed-geometry sha. NEGATIVE CONTROL: a tampered body gives a
     DIFFERENT sha (otherwise "identical" proves nothing -- the could-not-fail pattern).
  2. The stored request_body is the body AS RECEIVED: a bwphoto request does not gain
     the recipe keys in request_body, but effective_body has them.
  3. Pen profile / pen_id / calibration sha / app identity are the live resolved values.
  4. The sidecar is written under the temp runtime home, never the real
     %LOCALAPPDATA%\\ccPlotter (snapshot of that dir before/after).
  5. Old runs (no provenance) still load; a run with provenance links it.

Same conventions as tests/test_bw_photo.py: explicit temp home BEFORE any ccplotter
import, temp WORK_DIR, runs under pytest or directly.
"""
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home; pen/calibration never resolve from CWD
_HOME = _test_home.use_temp_home("ccplotter_test_home_runprov_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_runprov_"))
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
from PIL import Image

import ccplotter.routes.runs  # noqa: F401 (registers /api/runs on the shared app; before any request)
from ccplotter import runtime_home as rh
from ccplotter.config import BW_PHOTO_RECIPE, CALIBRATION_PATH, RUNTIME_HOME, WORK_DIR
import ccplotter.runs as runs_mod
from ccplotter.runs import (BACKLINK_PREFIX, PROVENANCE_DIRNAME, _runs_add, _runs_load,
                            gcode_backlink_line, load_run_provenance,
                            stamp_gcode_backlink, strip_run_backlink)


def _real_runtime_listing():
    """(relative path, size, mtime_ns) of everything under the REAL default home, except
    *.log (a live desktop app / other server appends to app.log while tests run -- a
    confounder, not our write; every non-log file and the run_provenance dir are covered)."""
    real = rh.default_home()
    if not real.is_dir():
        return None
    out = []
    for p in sorted(real.rglob("*")):
        if p.suffix == ".log":
            continue
        try:
            st = p.stat()
            out.append((str(p.relative_to(real)), st.st_size, st.st_mtime_ns))
        except OSError:
            pass
    return out


def _fixed_gray(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return (127 + 90 * np.sin(xx / 11.0) * np.cos(yy / 17.0)).clip(0, 255).astype(np.uint8)


def _make_photo(path: Path, w=900, h=700):
    base = _fixed_gray(h, w)
    img = np.stack([base, np.roll(base, 5, axis=0), np.roll(base, -5, axis=1)], axis=-1)
    Image.fromarray(img, mode="RGB").save(path, quality=92)


def _points_of(text):
    pts = []
    for ps in re.findall(r'points="([^"]+)"', text):
        for tok in ps.split():
            if "," in tok:
                x, y = tok.split(",")
                pts.append((round(float(x), 3), round(float(y), 3)))
    return sorted(pts)


def _geom_sha(resp_json):
    routed = resp_json.get("debug_files", {}).get("routed_svg")
    assert routed, "no debug_files.routed_svg in response"
    pts = _points_of((WORK_DIR / routed).read_text(encoding="utf-8"))
    assert pts, "routed svg has no points -- the sha would be a constant"
    return hashlib.sha256(repr(pts).encode()).hexdigest(), len(pts)


def _post(body):
    from ccplotter.routes.core import app
    return app.test_client().post("/api/process", json=body)


def _body(did, **extra):
    from ccplotter.cal import load_pen_profiles
    b = {"job_type": "gcode", "style": "tracev3", "debug": True, "paper": "120x120",
         "margin": 10.0, "resolution": 600, "pen_id": load_pen_profiles().get("active"),
         "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}]}
    b.update(extra)
    return b


@pytest.fixture(scope="module")
def run_one():
    """One real gcode run; returns (sent body, response json, run record, provenance,
    real-home listing before, after)."""
    did = "runprov_src"
    _make_photo(WORK_DIR / f"src_{did}.jpg")
    before = _real_runtime_listing()
    sent = _body(did)
    sent_copy = json.loads(json.dumps(sent))
    resp = _post(sent)
    assert resp.status_code == 200, resp.get_json()
    j = resp.get_json()
    after = _real_runtime_listing()
    recs = [r for r in _runs_load() if r.get("type") == "gcode"]
    assert recs, "route did not register a gcode run"
    rec = recs[0]
    return {"sent": sent_copy, "resp": j, "rec": rec,
            "gcode": (WORK_DIR / "layout.gcode").read_bytes(),
            "bundle_gcode": __import__("ccplotter.bundle", fromlist=["x"])._LAST_BUNDLE["gcode"],
            "prov": load_run_provenance(rec["id"]), "before": before, "after": after}


def test_home_is_the_temp_home_not_the_real_one():
    assert RUNTIME_HOME == _HOME.resolve()
    assert RUNTIME_HOME != rh.default_home()
    assert CALIBRATION_PATH.parent == RUNTIME_HOME


def test_sidecar_is_written_under_the_runtime_home_and_linked(run_one):
    rec = run_one["rec"]
    assert rec["provenance_file"] == f"{PROVENANCE_DIRNAME}/{rec['id']}.json"
    f = RUNTIME_HOME / rec["provenance_file"]
    assert f.is_file() and f.stat().st_size < 64 * 1024     # small: no images
    assert not any(v in f.read_text(encoding="utf-8").lower()
                   for v in ("base64,", "\x89png"))


def test_nothing_written_to_the_real_localappdata(run_one):
    assert run_one["before"] == run_one["after"], \
        "the real %LOCALAPPDATA%\\ccPlotter changed while a test ran"


def test_request_body_is_as_received(run_one):
    assert run_one["prov"]["request_body"] == run_one["sent"]


def test_resolved_inputs_are_recorded(run_one):
    from ccplotter.cal import resolve_pen_profile
    p = run_one["prov"]
    assert p["pen_id"] == run_one["sent"]["pen_id"]
    assert p["pen_profile"] == resolve_pen_profile(p["pen_id"])
    assert p["pen_profile"] and "fill_mm" in p["pen_profile"]
    assert p["calibration_sha256"] == hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest()
    assert p["runtime_home"] == str(RUNTIME_HOME)
    assert p["app"]["commit"] and "dirty" in p["app"] and p["schema"] == 1
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", p["created_at"])


def test_round_trip_regenerates_identical_routed_geometry_and_tamper_differs(run_one):
    """THE round trip: stored body -> regenerate -> identical routed-geometry sha.
    debug=True bypasses the variant cache, so this is a real second generation."""
    sha0, n0 = _geom_sha(run_one["resp"])
    stored = load_run_provenance(run_one["rec"]["id"])["request_body"]
    assert stored["debug"] is True                         # vcache really bypassed
    r1 = _post(stored)
    assert r1.status_code == 200, r1.get_json()
    assert not r1.get_json().get("cached")
    sha1, n1 = _geom_sha(r1.get_json())
    assert (sha1, n1) == (sha0, n0)
    # The regenerated GCODE is byte-identical once the per-run back-link line is
    # removed (and the raw bytes differ, because the two runs have different ids).
    g1 = (WORK_DIR / "layout.gcode").read_bytes()
    assert strip_run_backlink(g1) == strip_run_backlink(run_one["gcode"])
    assert g1 != run_one["gcode"], "two runs carried the same run id"
    # negative control: the SAME replay path with one field tampered must differ
    tampered = json.loads(json.dumps(stored))
    tampered["margin"] = 30.0
    r2 = _post(tampered)
    assert r2.status_code == 200, r2.get_json()
    sha2, _ = _geom_sha(r2.get_json())
    assert sha2 != sha0, "tampered body gave the same geometry: the check cannot fail"


def test_bwphoto_request_body_excludes_recipe_effective_body_includes_it(run_one):
    did = "runprov_bw_src"
    _make_photo(WORK_DIR / f"src_{did}.jpg", 700, 500)
    sent = _body(did, style="bwphoto", resolution=3000)
    sent.pop("resolution")
    sent["clahe"] = False                                   # the recipe must override this
    resp = _post(json.loads(json.dumps(sent)))
    assert resp.status_code == 200, resp.get_json()
    rec = [r for r in _runs_load() if r.get("type") == "gcode"][0]
    p = load_run_provenance(rec["id"])
    assert p["request_body"] == sent                         # as received (clahe False)
    assert p["request_body"]["style"] == "bwphoto"
    assert p["effective_body"]["clahe"] is BW_PHOTO_RECIPE["clahe"]
    assert p["effective_body"]["resolution"] == BW_PHOTO_RECIPE["resolution"]


def test_old_runs_without_provenance_still_load():
    rid = _runs_add({"type": "ai", "note": "legacy-shaped record"})
    rec = next(r for r in _runs_load() if r["id"] == rid)
    assert "provenance_file" not in rec
    assert load_run_provenance(rid) is None
    from ccplotter.routes.core import app
    listing = app.test_client().get("/api/runs").get_json()
    assert any(r["id"] == rid for r in listing)


def test_provenance_failure_never_costs_the_run_record(monkeypatch):
    import ccplotter.runs as runs_mod
    monkeypatch.setattr(runs_mod, "_provenance_write",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    rid = runs_mod._runs_add({"type": "gcode", "note": "x"}, provenance={"schema": 1})
    rec = next(r for r in _runs_load() if r["id"] == rid)
    assert "provenance_file" not in rec


# -- resolved quantities --------------------------------------------------------

def test_resolved_quantities_are_recorded(run_one):
    r = run_one["prov"]["resolved"]
    ti = run_one["rec"]["trace_info"]
    assert r["effective_res"] == ti["effective_res"] and r["effective_res"]
    assert r["requested_res"] == ti["requested_res"]
    assert r["fill_px"] == ti["audit"]["fill_px"] and r["fill_px"] > 0
    assert r["mm_per_px"] == ti["audit"]["mm_per_px"] and r["mm_per_px"] > 0
    assert isinstance(r["reach_profile"], str) and r["reach_profile"]


def test_resolved_from_stats_is_none_safe():
    assert runs_mod.resolved_from_stats({}, "plotter_x") == {
        "requested_res": None, "effective_res": None, "fill_px": None, "mm_per_px": None,
        "fill_mm_resolved": None, "reach_profile": "plotter_x"}


# -- untracked_count -------------------------------------------------------------

def test_untracked_count_recorded_and_matches_git(run_one):
    import subprocess
    app = run_one["prov"]["app"]
    assert "untracked_count" in app
    if app["source"] == "source":
        por = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(REPO))
        live = sum(1 for ln in por.splitlines() if ln.startswith(b"??"))
        # computed once per process; the tree may legitimately change during a run
        assert isinstance(app["untracked_count"], int) and app["untracked_count"] >= 0
        assert abs(app["untracked_count"] - live) <= 25


# -- pruning never removes a sidecar whose run is still in runs.json -------------

def test_prune_never_removes_a_live_runs_sidecar(monkeypatch):
    d = RUNTIME_HOME / PROVENANCE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runs_mod, "PROVENANCE_KEEP", 1)
    live_ids = []
    for i in range(3):
        live_ids.append(_runs_add({"type": "gcode", "note": f"live{i}"},
                                  provenance={"schema": 1, "n": i}))
    # orphans (no run in runs.json) and the live sidecars, all backdated: the LIVE ones are
    # the OLDEST files, so a naive keep-newest-N prune would delete them.
    for i in range(3):
        (d / f"orphan{i}.json").write_text("{}")
    old = time.time() - 10_000
    for lid in live_ids:
        os.utime(d / f"{lid}.json", (old, old))
    for i in range(3):
        os.utime(d / f"orphan{i}.json", (old + 100 + i, old + 100 + i))
    rid = _runs_add({"type": "gcode", "note": "trigger"}, provenance={"schema": 1})
    present = {f.stem for f in d.glob("*.json")}
    for lid in live_ids + [rid]:
        assert lid in present, f"live run {lid} lost its provenance sidecar"
    # positive control: pruning really is active (orphans beyond KEEP were removed)
    assert not ({"orphan0", "orphan1", "orphan2"} <= present)
    assert sum(1 for s in present if s.startswith("orphan")) <= 1


# -- gcode header back-link ------------------------------------------------------

def test_gcode_backlink_in_layout_archive_and_bundle(run_one):
    rec, g = run_one["rec"], run_one["gcode"]
    want = gcode_backlink_line(rec["id"], rec["provenance_file"]).encode("ascii")
    lines = g.splitlines()
    assert lines.count(want) == 1, "layout.gcode lacks exactly one back-link line"
    assert [ln for ln in lines if ln.startswith(BACKLINK_PREFIX.encode())] == [want]
    assert lines[1].startswith(b"; Source:") and lines[2] == want       # placed after Source
    g.decode("ascii")                                                   # pure ASCII file
    assert not [ln for ln in lines if ln.strip().upper().startswith(b"G92")]   # no G92 command
    assert b"G92" not in want
    assert want.startswith(b"; ")                                       # plain comment
    assert rec["gcode_backlink"] is True
    arch = (WORK_DIR / rec["files"]["gcode"]).read_bytes()
    assert arch == g, "archive copy differs from the sideloaded file"
    assert run_one["bundle_gcode"] == g, "download bundle gcode differs from the sideloaded file"
    assert b"\r\n" in g[:300] and g.count(b"\r\n") >= g.count(b"\n") - 1   # EOL style kept


def test_stamp_is_idempotent_and_strip_restores_original(tmp_path):
    src = b"; Pen plotter\r\n; Source: x\r\n; PRE\r\nG1 X1 Y1\r\n"
    f = tmp_path / "a.gcode"
    f.write_bytes(src)
    stamp_gcode_backlink(f, "aaaaaaaa", "run_provenance/aaaaaaaa.json")
    stamp_gcode_backlink(f, "bbbbbbbb", "run_provenance/bbbbbbbb.json")   # replaces, not adds
    out = f.read_bytes()
    assert out.count(BACKLINK_PREFIX.encode()) == 1 and b"bbbbbbbb" in out
    assert strip_run_backlink(out) == src
    assert strip_run_backlink(src) == src                 # no-op on an unstamped file
    with pytest.raises(UnicodeEncodeError):
        gcode_backlink_line("id", "run_provenance/\u2014.json")           # em-dash refused


def test_harness_strip_prefix_matches_the_app_prefix():
    text = (REPO / "scripts" / "replay_harness.py").read_text(encoding="utf-8")
    assert f'RUN_BACKLINK_PREFIX = b"{BACKLINK_PREFIX}"' in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# --- rows/hybrid provenance gap: the RESOLVED core_r_mm is recorded -----------------------

def _prov_for(did, **extra):
    _make_photo(WORK_DIR / f"src_{did}.jpg")
    resp = _post(_body(did, **extra))
    assert resp.status_code == 200, resp.get_json()
    rec = [r for r in _runs_load() if r.get("type") == "gcode"][0]
    return load_run_provenance(rec["id"]), (WORK_DIR / "layout.gcode").read_bytes()


def test_rows_and_hybrid_record_resolved_core_r_mm_and_gcode_is_unchanged(run_one):
    from ccplotter.config import CORE_R_MM_DEFAULT, ROWS_CORE_R_MM_DEFAULT
    # positive: rows with no core_r_mm sent -> backend default recorded in effective_body only
    p, g_rows = _prov_for("runprov_rows", core_fill="rows")
    assert "core_r_mm" not in p["request_body"]
    assert p["effective_body"]["core_r_mm"] == ROWS_CORE_R_MM_DEFAULT
    # an explicit value is recorded as resolved
    p2, g_rows2 = _prov_for("runprov_rows", core_fill="rows", core_r_mm=1.7)
    assert p2["effective_body"]["core_r_mm"] == 1.7
    # hybrid default
    p3, _ = _prov_for("runprov_hyb", core_fill="hybrid")
    assert p3["effective_body"]["core_r_mm"] == CORE_R_MM_DEFAULT
    # the recorded value is the one that drives generation: sending it explicitly gives the
    # same gcode as leaving it to the default (provenance is read-only; bytes unchanged)
    _, g_rows_explicit = _prov_for("runprov_rows", core_fill="rows", core_r_mm=ROWS_CORE_R_MM_DEFAULT)
    strip = lambda b: b"\n".join(l for l in b.split(b"\n") if not l.startswith(b"; ccplotter-run:"))
    assert strip(g_rows) == strip(g_rows_explicit)
    # negative control: a different core_r_mm is recorded differently (the check can fail)
    assert p2["effective_body"]["core_r_mm"] != p["effective_body"]["core_r_mm"]
    # default (rings) jobs gain no core_r_mm key
    assert "core_r_mm" not in run_one["prov"]["effective_body"]
