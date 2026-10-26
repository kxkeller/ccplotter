"""Per-job `realised_pitch` (READOUT ONLY).

The slider sets a TARGET pass spacing; the resolution floor/cap and the preliminary analytic scale can
make the real spacing on the page differ (a BIC bwphoto job plotted ~0.55 x fill, and a uRight job could not
move at all). The app reports the REAL spacing: fill_px_final x (FINAL placed scale x PX2MM), per
tracev3-engine item. Guards, through the real Flask route:

  1. present for tracev3, bwphoto and illustration (Trace V3+); None/absent for non-tracev3 styles;
  2. POSITIVE CONTROL against an independent measurement: a solid block traces to concentric rings; the
     ring-to-ring spacing measured on the routed SVG (final placed mm) matches realised_pitch_mm within 10%,
     at two placement scales (tracev3 holds spacing in mm by design, so both must read about the same);
  3. NEGATIVE CONTROL: the same measurement REJECTS a number that is 1.5x off (the check can fail);
     a back-solve from fill_mm/fill_px would fail in the BW dead zone, which is why the readout is
     taken from the final placement instead;
  4. provenance: resolved.realised_pitch is on the run sidecar for a gcode job;
  5. readout only: the gcode of a tracev3 job is unchanged by the readout (stripped sha is stable run to run
     and the response keys other than realised_pitch are untouched; the replay harness covers the
     wider parity check).
"""
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home BEFORE any ccplotter import
_test_home.use_temp_home("ccplotter_test_home_rpitch_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_rpitch_"))
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
from PIL import Image

import ccplotter.routes.runs  # noqa: F401
from ccplotter.config import WORK_DIR
from ccplotter.runs import _runs_load, load_run_provenance

PX2MM = 25.4 / 96.0
TOL = 0.10


def _make_block(path: Path, w=1100, h=800):
    img = np.full((h, w), 255, np.uint8)
    img[200:600, 300:800] = 20
    Image.fromarray(img).convert("RGB").save(path, quality=95)
    return path


def _run(did, style, scale=1.0, job_type="preview", extra=None):
    from ccplotter.cal import load_pen_profiles
    from ccplotter.routes.core import app
    _make_block(WORK_DIR / f"src_{did}.jpg")
    body = {"job_type": job_type, "paper": "120x120", "margin": 10.0, "debug": True, "style": style,
            "pen_id": load_pen_profiles().get("active"),
            "items": [{"id": did, "scale": scale, "rot": 0, "dx": 0, "dy": 0, "crop": None}]}
    body.update(extra or {})
    resp = app.test_client().post("/api/process", json=body)
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


def _measured_ring_pitch_mm(j):
    """Independent measurement: concentric rings of a solid rectangle. The left vertical sides of successive
    rings sit at x = x0, x0+p, x0+2p ...; p in routed-SVG user units (96 dpi px) x PX2MM = mm on the page."""
    rs = (WORK_DIR / j["debug_files"]["routed_svg"]).read_text(encoding="utf-8")
    xs, ys = [], []
    for ps in re.findall(r'points="([^"]+)"', rs):
        for tok in ps.split():
            if "," in tok:
                x, y = tok.split(",")
                xs.append(float(x)); ys.append(float(y))
    xs, ys = np.array(xs), np.array(ys)
    xmid = 0.5 * (xs.min() + xs.max())
    left = np.unique(np.round(xs[xs < xmid], 3))
    d = np.diff(left)
    d = d[d > 0.05]
    assert len(d) > 5, "not enough rings to measure"
    return float(np.median(d)) * PX2MM


def _within(a, b, tol=TOL):
    return abs(a - b) / b <= tol


# ------------------------------------------------------------------ 1. presence / absence

@pytest.mark.parametrize("style", ["tracev3", "bwphoto", "illustration"])
def test_realised_pitch_present_for_tracev3_engine_styles(style):
    j = _run(f"rp_{style}", style)
    rp = j.get("realised_pitch")
    assert rp and len(rp["items"]) == 1, rp
    it = rp["items"][0]
    assert it["id"] == f"rp_{style}" and it["realised_pitch_mm"] > 0
    assert rp["min_mm"] == rp["max_mm"] == it["realised_pitch_mm"]
    # it is exactly fill_px_final x final_scale x PX2MM (the arithmetic is shown, not hidden)
    assert it["realised_pitch_mm"] == pytest.approx(it["fill_px_final"] * it["final_scale"] * PX2MM, rel=2e-3)
    assert it["fill_px_final"] >= 1.2 - 1e-9          # the trace's representability floor
    # the final scale it used is the one the response echoes for the placement
    ig = list(j["item_geometry"].values())[0]["scale_aware_equivalence_prelim_approx"]
    assert it["final_scale"] == pytest.approx(ig["final_s"], rel=1e-6)


def test_realised_pitch_absent_for_non_tracev3_style():
    j = _run("rp_plotready", "plotready")
    assert not j.get("realised_pitch")


# ------------------------------------------------------------------ 2+3. positive / negative control

@pytest.mark.parametrize("style,scale", [("tracev3", 1.0), ("tracev3", 0.5), ("illustration", 1.0)])
def test_realised_pitch_matches_measured_ring_pitch(style, scale):
    j = _run(f"rp_m_{style}_{int(scale * 100)}", style, scale=scale)
    real = j["realised_pitch"]["items"][0]["realised_pitch_mm"]
    meas = _measured_ring_pitch_mm(j)
    assert _within(real, meas), (style, scale, real, meas)
    # NEGATIVE CONTROL: the identical check must reject a number 1.5x off (so it can fail)
    assert not _within(real * 1.5, meas)
    assert not _within(real / 1.5, meas)


# ------------------------------------------------------------------ 4. provenance

def test_provenance_carries_realised_pitch():
    j = _run("rp_prov", "illustration", job_type="gcode")
    rec = next(r for r in _runs_load() if r.get("type") == "gcode")
    prov = load_run_provenance(rec["id"])
    rp = prov["resolved"].get("realised_pitch")
    assert rp and rp["items"][0]["realised_pitch_mm"] == j["realised_pitch"]["items"][0]["realised_pitch_mm"]


def test_provenance_omits_realised_pitch_for_non_tracev3():
    _run("rp_prov2", "plotready", job_type="gcode")
    rec = next(r for r in _runs_load() if r.get("type") == "gcode")
    assert "realised_pitch" not in load_run_provenance(rec["id"])["resolved"]


# ------------------------------------------------------------------ 5. readout only

def test_readout_does_not_change_the_gcode():
    import hashlib

    def sha(j):
        raw = (WORK_DIR / j["gcode_file"]).read_bytes()
        keep = [ln for ln in raw.splitlines(keepends=True) if not ln.startswith((b"; ccplotter-run: ", b"; Source:"))]
        return hashlib.sha256(b"".join(keep)).hexdigest()
    a = _run("rp_g1", "tracev3", job_type="gcode")
    b = _run("rp_g1", "tracev3", job_type="gcode")
    assert sha(a) == sha(b)
