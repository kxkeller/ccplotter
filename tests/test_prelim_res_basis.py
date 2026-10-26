"""Opt-in `prelim_res_basis` (private request key, scale_aware_v2 only, no UI).

The defect it addresses: composite.py's v2 resolution boost takes
base_res = the REQUESTED resolution (3000 for the bwphoto recipe), but the preliminary trace runs at
min(requested, MODE_MAX_RES["tracev3"] = 2200). The boost ratio, the fill_px it scales and the
res_basis_boost are therefore under-counted by requested/2200 (1.364x at 3000). Values:

  absent / "requested"  default behaviour, byte for byte
  "prelim_effective"    base_res = the preliminary trace's effective resolution (render.resolve_trace_res)
  "final_solve"         also solves the 1.2 px floor in the FINAL frame

Fixture. A bwphoto job with a BIC pen (fill_mm 0.12, scale 1.16, 257x182 paper, margin 10) has a
preliminary ink bbox of 582.03 x 356.35 mm on the 2200-px grid, i.e. an ink bbox that is the whole frame,
aspect 1.633. A synthetic image of that aspect whose ink fills the frame reproduces the same placement
arithmetic without the photo, so the measured realised pitches are reproduced here THROUGH the
committed realised_pitch readout (not through the fill_mm/fill_px back-solve):
    default          0.099 mm
    final_solve      0.090 mm
    prelim_effective 0.135 mm   (same as requesting resolution 2200)
Controls are paired with every claim: the unfixed arm reproduces the defect value (the fixture can show it),
the non-boosting / 2200-base arms are unchanged (the flag cannot move what it must not), and the byte-identity
check carries a positive control (flag-on differs). The traces are the cost (a 5480 px grid is ~70 s), so
identical requests are cached within the process.
"""
import functools
import hashlib
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home BEFORE any ccplotter import
_test_home.use_temp_home("ccplotter_test_home_prb_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_prb_"))
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
from PIL import Image

import ccplotter.routes.runs  # noqa: F401
from ccplotter import config as cfg
from ccplotter.config import WORK_DIR, MODE_MAX_RES
from ccplotter.render import resolve_trace_res

PEN = "bic_prb_test"
SRC_W, SRC_H = 1100, 673  # aspect 1.6345 ~ the reference job's 582.03 / 356.35 = 1.6333; ink fills the frame
SMALL = "160x120"                 # inside the temp home's reach profile (257x182 is not), still boosts at fill 0.12


def _install_pen(fill_mm=0.12):
    from ccplotter.cal import load_pen_profiles, save_pen_profiles
    d = load_pen_profiles()
    d["profiles"][PEN] = {"label": "BIC prb test", "pen_mm": 1.0, "fill_mm": fill_mm, "fill_mm_calibrated": fill_mm,
                          "z_touch_abs": 18.95, "z_up": 3.5, "z_down": -0.6, "z_micro": 0.9, "draw_speed": 3000,
                          "travel_speed": 5000, "plunge_speed": 500, "lift_speed": 2000,
                          "pen_offset_x": 0.0, "pen_offset_y": 0.0}
    save_pen_profiles(d)


def _make_frame_block(did):
    img = np.full((SRC_H, SRC_W), 255, np.uint8)
    img[3:SRC_H - 3, 3:SRC_W - 3] = 20
    Image.fromarray(img).convert("RGB").save(WORK_DIR / f"src_{did}.jpg", quality=95)


def _run(did, style="tracev3", prb=None, resolution=3000, fill_mm=0.12, scale=1.16, job_type="preview",
         body_extra=None, paper="257x182"):
    from ccplotter.routes.core import app
    _install_pen(fill_mm)
    _make_frame_block(did)
    body = {"job_type": job_type, "paper": paper, "margin": 10.0, "style": style, "pen_id": PEN,
            "items": [{"id": did, "scale": scale, "rot": 0, "dx": 0, "dy": 0, "crop": None}]}
    if style == "tracev3":
        body["resolution"] = resolution
    if prb is not None:
        body["prelim_res_basis"] = prb
    body.update(body_extra or {})
    return app.test_client().post("/api/process", json=body)


def _stripped_sha(j):
    raw = (WORK_DIR / j["gcode_file"]).read_bytes()
    keep = [ln for ln in raw.splitlines(keepends=True) if not ln.startswith((b"; ccplotter-run: ", b"; Source:"))]
    return hashlib.sha256(b"".join(keep)).hexdigest()


@functools.lru_cache(maxsize=None)
def _go(style="tracev3", prb=None, resolution=3000, fill_mm=0.12):
    """One cached preview run at the reference placement (257x182, margin 10, scale 1.16): (realised_pitch_mm, json)."""
    resp = _run(f"prb_{style}_{prb}_{resolution}_{fill_mm}", style=style, prb=prb, resolution=resolution, fill_mm=fill_mm)
    assert resp.status_code == 200, resp.get_json()
    j = resp.get_json()
    return j["realised_pitch"]["items"][0]["realised_pitch_mm"], j


def _geom(j):
    return list(j["item_geometry"].values())[0]["scale_aware_equivalence_prelim_approx"]


def _boost(j):
    return _geom(j)["v2_res_boost"]


# ------------------------------------------------------------------ the single source of the 2200

def test_resolve_trace_res_is_the_single_source_not_a_literal():
    # the effective resolution of the preliminary tracev3 trace IS the mode cap, whatever MODE_MAX_RES says
    for req in (1000, 2200, 3000, 4000):
        r, cap, eff = resolve_trace_res({"mode": "tracev3", "resolution": req})
        assert (r, cap, eff) == (req, MODE_MAX_RES["tracev3"], min(req, MODE_MAX_RES["tracev3"]))
    # a v2 cap override (the FINAL pass) lifts it, bounded by SCALE_AWARE_V2_NATIVE_CAP
    assert resolve_trace_res({"mode": "tracev3", "resolution": 9000, "_tracev3_res_cap_override": 99999})[2] \
        == cfg.SCALE_AWARE_V2_NATIVE_CAP
    # other modes keep their own cap
    assert resolve_trace_res({"mode": "outline", "resolution": 3000})[2] == MODE_MAX_RES["outline"]


def test_flag_records_the_basis_it_used():
    _, j = _go(prb="prelim_effective")
    blk = _geom(j)["prelim_res_basis"]
    assert blk["mode"] == "prelim_effective" and blk["requested_res"] == 3000
    assert blk["prelim_effective_res"] == MODE_MAX_RES["tracev3"]       # from the helper, not a literal


# ------------------------------------------------------------------ the three realised pitches (readout)

def test_flag_off_reproduces_the_defect_value():
    p, j = _go()
    assert p == pytest.approx(0.099, rel=0.015), p            # diagnosis K_bw_0783addb
    assert _boost(j)["base_res"] == 3000                      # the REQUESTED resolution: the defect
    assert "prelim_res_basis" not in _geom(j) and "prelim_res_basis" not in j


def test_final_solve_reproduces_cfb_0_090():
    p, j = _go(prb="final_solve")
    assert p == pytest.approx(0.090, rel=0.015), p            # diagnosis CFB_bw_fill0.12
    assert j["prelim_res_basis"] == "final_solve"
    ri = j["realised_pitch"]["items"][0]
    assert ri["fill_px_final"] > 1.2 and not ri["floor_binds"]   # the floor is solved in the final frame
    b = _boost(j)
    assert b["base_res"] == MODE_MAX_RES["tracev3"] and b["boosted_res"] == cfg.SCALE_AWARE_V2_NATIVE_CAP


def test_prelim_effective_reproduces_c_bw_res2200_0_135_and_equals_requesting_2200():
    p, j = _go(prb="prelim_effective")
    assert p == pytest.approx(0.135, rel=0.015), p  # same as requesting resolution 2200 (fill .12)
    # CONTROL (the diagnosis' own CF-A): requesting 2200 with the flag OFF is the same drawing -> same pitch
    p2200, _ = _go(resolution=2200)
    assert p == pytest.approx(p2200, rel=1e-3), (p, p2200)
    # and the three arms are genuinely different (the fixture can tell them apart)
    assert _go(prb="final_solve")[0] < _go()[0] < p


def test_non_floor_case_both_flags_coincide_and_are_coarser_than_today():
    # fill 0.30: the 1.2 px floor does not bind (diagnosis K_bw_fill0.30 / CFB_bw_fill0.30), so the two flag values
    # must agree, and both undo the 1.364x over-density of today (0.55 x fill -> 0.75 x fill)
    off, pe, fs = _go(fill_mm=0.30)[0], _go(prb="prelim_effective", fill_mm=0.30)[0], _go(prb="final_solve", fill_mm=0.30)[0]
    assert pe == pytest.approx(fs, rel=1e-3)
    assert pe / off == pytest.approx(3000 / 2200, rel=0.03), (off, pe)
    assert pe == pytest.approx(0.75 * 0.30, rel=0.03)         # lands on the designed K x fill_mm


def test_no_boost_case_is_untouched_by_either_flag():
    # fill 0.60: fill_px >= 3 in the prelim frame, the boost never fires (diagnosis C_bw_fill0.60_noboost)
    off = _go(fill_mm=0.60)
    fp = off[1]["realised_pitch"]["items"][0]["fill_px_final"]
    assert fp >= 3.0 - 1e-9                                   # the control: it really is a no-boost case
    for prb in ("prelim_effective", "final_solve"):
        p, j = _go(prb=prb, fill_mm=0.60)
        assert p == off[0] and j["realised_pitch"]["items"][0]["fill_px_final"] == fp


# ------------------------------------------------------------------ the illustration clamp

def test_illustration_clamp_unchanged_at_base_2200():
    # base 2200 (the recipe's pin): the corrected basis equals the requested one -> the flag moves nothing.
    # For the illustration style final_solve == prelim_effective (the clamp IS its final-frame solve, 0.9 x fill_mm).
    off, jo = _go(style="illustration")
    assert off == pytest.approx(0.108, rel=0.01)  # the illustration clamp pitch for the reference job
    for prb in ("prelim_effective", "final_solve"):
        p, j = _go(style="illustration", prb=prb)
        assert p == off, (prb, p, off)
        assert _boost(j)["boosted_res"] == _boost(jo)["boosted_res"]
        assert _boost(j)["actual_boost"] == _boost(jo)["actual_boost"]


def test_illustration_clamp_reads_the_corrected_basis_when_base_is_not_2200(monkeypatch):
    # POSITIVE CONTROL for "the corrected basis feeds the clamp": break the recipe's pin (base 3000).
    # flag off -> the clamp mis-sizes by 3000/2200; prelim_effective -> it equals the pinned-2200 answer.
    pinned, _ = _go(style="illustration")
    monkeypatch.setitem(cfg.ILLUSTRATION_RECIPE, "resolution", 3000)
    resp = _run("prb_ilc1", style="illustration")
    assert resp.status_code == 200
    broken = resp.get_json()["realised_pitch"]["items"][0]["realised_pitch_mm"]
    resp = _run("prb_ilc2", style="illustration", prb="prelim_effective")
    assert resp.status_code == 200
    jf = resp.get_json()
    fixed = jf["realised_pitch"]["items"][0]["realised_pitch_mm"]
    assert fixed == pytest.approx(pinned, rel=1e-3), (fixed, pinned)
    assert abs(broken - pinned) / pinned > 0.05, (broken, pinned)   # the control can fail: the unfixed clamp is off
    b = _boost(jf)
    assert b["base_res"] == MODE_MAX_RES["tracev3"] and b["requested_res"] == 3000


# ------------------------------------------------------------------ flag-off byte identity / scope

def test_flag_off_is_byte_identical_and_the_key_never_reaches_a_trace(monkeypatch):
    import ccplotter.composite as comp
    seen = []
    orig = comp._source_svg

    def spy(did, render_opts, debug_prefix=None, pen_id=None):
        seen.append(dict(render_opts))
        return orig(did, render_opts, debug_prefix=debug_prefix, pen_id=pen_id)
    monkeypatch.setattr(comp, "_source_svg", spy)
    runs = {}
    for k, prb in (("none", None), ("requested", "requested"), ("fs", "final_solve"), ("pe", "prelim_effective")):
        r = _run("prb_id", prb=prb, job_type="gcode", paper=SMALL, scale=1.0)
        assert r.status_code == 200, r.get_json()
        runs[k] = (r.get_json(), _stripped_sha(r.get_json()))
    assert runs["none"][1] == runs["requested"][1]
    assert seen and all("prelim_res_basis" not in o for o in seen)   # not in render_opts/item_opts: no cache key moves
    # POSITIVE CONTROL: the flag-on arm is a different drawing than flag-off (the hash can tell).  At this placement the
    # 1.2 px floor does not bind, so final_solve and prelim_effective coincide (both solve the same 3 px target).
    assert runs["fs"][1] != runs["none"][1] and runs["pe"][1] != runs["none"][1]
    assert runs["fs"][1] == runs["pe"][1]
    a, b = runs["none"][0], runs["requested"][0]
    assert "realised_pitch" in a and "prelim_res_basis" not in a and "prelim_res_basis" not in b
    assert runs["fs"][0]["prelim_res_basis"] == "final_solve"


def test_flag_is_ignored_outside_scale_aware_v2_and_validated():
    shas = []
    for prb in (None, "final_solve"):
        r = _run("prb_sc", prb=prb, job_type="gcode", paper=SMALL, scale=1.0, body_extra={"scale_aware_v2": False})
        assert r.status_code == 200, r.get_json()
        assert "prelim_res_basis" not in r.get_json()
        shas.append(_stripped_sha(r.get_json()))
    assert shas[0] == shas[1]
    bad = _run("prb_bad", prb="bogus")
    assert bad.status_code == 400 and "prelim_res_basis" in bad.get_json()["error"]
