"""Tests for the B/W photo mode (opt-in style "bwphoto").

The mode ships a locked B/W config -- XDoG p02_e55 at shorter-side 512 (or native), then
tracev3 with CLAHE on and tone off -- as a real, opt-in style (ccplotter/bw_photo.py,
config.BW_PHOTO_RECIPE, routes/core.py, render.py). This file is the standing guard:

  1. xdog() / bw_xdog_raster() give a pinned output hash on small FIXED inputs
     (a change to the verbatim front end must be deliberate).
  2. On the six real benchmark photos the raster is byte-equal to the locked
     reference rasters (skipped, loudly, if those images are absent).
  3. The recipe dict is pinned.
  4. Through the real Flask route: style="bwphoto" runs the front end, keeps the
     raster as debug_xdog.png, cleans up its temp file, and its geometry DIFFERS
     from plain tracev3 at matched settings (so deleting the mechanism fails).
  5. KEY ABSENT == OLD BEHAVIOUR: a plain style="tracev3" request never calls
     the front end, and a caller cannot smuggle `front_end` into it (identical
     geometry with and without the smuggled key).
  6. A hostile bwphoto request (clahe off, tone on, tiny resolution, native_res)
     is forced back to the recipe: identical geometry to the clean request.

Same convention as tests/test_photo512_recipe.py: runs under pytest or directly;
route tests write their source into config.WORK_DIR so they work whether this
module or another test module imported ccplotter.config first.
"""
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home; pen/calibration never resolve from CWD
_test_home.use_temp_home("ccplotter_test_home_bwphoto_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_bwphoto_"))
sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import pytest
from PIL import Image

import ccplotter.bw_photo as bw_photo
from ccplotter.bw_photo import bw_xdog_raster, xdog
from ccplotter.config import (BW_PHOTO_FRONT_END, BW_PHOTO_RECIPE, WORK_DIR)

M09_RASTERS = REPO / "tests" / "fixtures" / "benchmark" / "m09_rasters"
SRC = REPO / "tests" / "fixtures" / "benchmark" / "sources"
SIX = {"ice": "07_ice1_railway_source.jpg", "belltower": "02_belltower_source.jpg",
       "tram": "04_hktram_source.jpg", "bike": "01_bike_source.jpg",
       "a350": "06_a350_source.jpg", "giraffe": "03_giraffe_source.png"}

# Pinned outputs on fixed synthetic inputs (generated below, no files needed).
PIN_XDOG_64x96 = "2778f34cdc5802d49d549fda67b8ece5d4cdd29f27a0e1aac92645e9be9043f7"
PIN_RASTER_120x180 = "89651832ee279db457f1ac0b23137f724de8d1563460f6dc6afd41c074cdb63b"   # explicit side=512
PIN_RASTER_NATIVE_120x180 = "1aa94f6a5f20a74bef141593f4f8c2991395bffa8b99c0240ec5f7bf66f55bf2"  # the request default (no bw_xdog_side) is native, so the pinned raster is the native-side output


def _fixed_gray(h, w, fx, fy):
    yy, xx = np.mgrid[0:h, 0:w]
    return (128 + 100 * np.sin(xx / fx) * np.cos(yy / fy)).astype(np.uint8)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1-3. The front end itself
# ---------------------------------------------------------------------------

def test_xdog_pinned_hash_on_fixed_input():
    out = xdog(_fixed_gray(96, 64, 5.0, 7.0), 0.8)
    assert out.dtype == np.uint8 and out.shape == (96, 64)
    assert _sha(out) == PIN_XDOG_64x96


def test_bw_xdog_raster_pinned_hash_on_fixed_png(tmp_path):
    p = tmp_path / "fixed.png"
    cv2.imencode(".png", _fixed_gray(120, 180, 11.0, 17.0))[1].tofile(str(p))
    out = bw_xdog_raster(p, 512)  # explicit side=512, the locked reference side
    assert out.shape == (512, 768), out.shape          # shorter side 512, aspect kept
    assert _sha(out) == PIN_RASTER_120x180
    assert bw_xdog_raster(p, 512).tobytes() == out.tobytes()  # deterministic
    assert bw_xdog_raster(p).tobytes() == out.tobytes()       # raster-function default stays the locked 512 cell


def test_bwphoto_request_default_side_is_native_pinned(tmp_path):
    # PIN_RASTER_NATIVE_120x180 = fresh baseline for the new bwphoto request default.
    assert bw_photo.BW_XDOG_DEFAULT_SIDE == "native"  # the request default is native
    assert bw_photo.parse_bw_xdog_side(None) == "native"
    p = tmp_path / "fixed.png"
    cv2.imencode(".png", _fixed_gray(120, 180, 11.0, 17.0))[1].tofile(str(p))
    out = bw_xdog_raster(p, bw_photo.parse_bw_xdog_side(None))
    assert out.shape == (120, 180), out.shape          # native = short side 120 (< 1024 cap)
    assert _sha(out) == PIN_RASTER_NATIVE_120x180


def test_front_end_parameters_are_the_locked_cell():
    assert (bw_photo.BW_XDOG_SIDE, bw_photo.BW_XDOG_P, bw_photo.BW_XDOG_K,
            bw_photo.BW_XDOG_EPS, bw_photo.BW_XDOG_PHI) == (512, 20.0, 1.6, 0.55, 2.0)
    # BW_XDOG_SIDE stays the locked reference-cell / raster-function default; the bwphoto REQUEST default is separate.
    assert bw_photo.BW_XDOG_DEFAULT_SIDE == "native"  # the request default is native


def test_recipe_dict_pinned():
    assert BW_PHOTO_FRONT_END == "bw_xdog"
    assert BW_PHOTO_RECIPE == {"clahe": True, "tone": "off", "resolution": 3000, "native_res": False}


@pytest.mark.parametrize("name", sorted(SIX))
def test_raster_byte_equal_to_m09_locked_raster(name):
    src = SRC / SIX[name]
    locked = M09_RASTERS / f"{name}__phi2_s512.png"
    if not (src.exists() and locked.exists()):
        pytest.skip(f"fixture image not redistributable; place {SIX[name]} at {SRC} and "
                    f"{locked.name} at {M09_RASTERS} to run")
    want = cv2.imdecode(np.fromfile(str(locked), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    got = bw_xdog_raster(src, 512)  # explicit side=512 == the locked reference raster
    assert got.shape == want.shape and np.array_equal(got, want)
    # and the PNG bytes the app writes are the locked reference file's bytes
    ok, buf = cv2.imencode(".png", got)
    assert ok and buf.tobytes() == locked.read_bytes()


# ---------------------------------------------------------------------------
# 4-6. Through the real route
# ---------------------------------------------------------------------------

def _make_photo(path: Path, w: int, h: int):
    base = _fixed_gray(h, w, 11.0, 17.0)
    img = np.stack([base, np.roll(base, 5, axis=0), np.roll(base, -5, axis=1)], axis=-1)
    Image.fromarray(img, mode="RGB").save(path, quality=92)
    return path


def _process(did, extra):
    from ccplotter.routes.core import app
    return app.test_client().post("/api/process", json={
        "job_type": "preview", "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}],
        "paper": "120x120", "margin": 10.0, "debug": True, **extra})


def _points_of(text):
    pts = []
    for ps in re.findall(r'points="([^"]+)"', text):
        for tok in ps.split():
            if "," in tok:
                x, y = tok.split(",")
                pts.append((round(float(x), 3), round(float(y), 3)))
    return sorted(pts)


def _run(did, extra, w=1500, h=1000):
    """One real /api/process call; returns (response json, routed points)."""
    _make_photo(WORK_DIR / f"src_{did}.jpg", w, h)
    resp = _process(did, extra)
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    routed = body.get("debug_files", {}).get("routed_svg")
    assert routed, "no debug_files.routed_svg in response"
    return body, _points_of((WORK_DIR / routed).read_text(encoding="utf-8"))


MATCHED = {"clahe": BW_PHOTO_RECIPE["clahe"], "tone": BW_PHOTO_RECIPE["tone"],
           "resolution": BW_PHOTO_RECIPE["resolution"], "native_res": BW_PHOTO_RECIPE["native_res"]}


def test_route_runs_front_end_keeps_debug_raster_and_cleans_up(monkeypatch):
    did = "bwphoto_route"
    calls, temps = [], []
    orig = bw_photo.prepare_bw_xdog_source

    def spy(raw_path, work_dir, debug_prefix=None, **kw):
        p = orig(raw_path, work_dir, debug_prefix, **kw)
        calls.append(debug_prefix)
        temps.append(p)
        return p

    monkeypatch.setattr(bw_photo, "prepare_bw_xdog_source", spy)
    body, pts = _run(did, {"style": "bwphoto"})
    assert calls, "front end never ran for style='bwphoto'"
    assert not [t for t in temps if t.exists()], "temp raster leaked"
    assert not list(WORK_DIR.glob("bwphoto_src_*.png"))
    # the debug raster is the locked raster of THIS source (positive: equals recomputation)
    assert body["debug_files"].get("xdog") == "debug_xdog.png", body["debug_files"]
    kept = cv2.imdecode(np.fromfile(str(WORK_DIR / "debug_xdog.png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    # no bw_xdog_side -> native (1500x1000 source -> short side 1000)
    assert np.array_equal(kept, bw_xdog_raster(WORK_DIR / f"src_{did}.jpg", "native"))
    assert min(kept.shape) == 1000
    # info.json states the front end (DIRECTION s8: visible in the IR)
    info = list(WORK_DIR.glob(f"src_{did}_render_*.svg.info.json"))
    assert info and '"front_end"' in info[0].read_text()
    assert pts, "no strokes routed"


def test_route_explicit_512_debug_raster_is_the_old_default():
    """With no bw_xdog_side the request default is native; an explicit 512 stays byte-identical to the 512-side raster."""
    _run("bwphoto_exp512", {"style": "bwphoto", "bw_xdog_side": 512})
    kept = cv2.imdecode(np.fromfile(str(WORK_DIR / "debug_xdog.png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(kept, bw_xdog_raster(WORK_DIR / "src_bwphoto_exp512.jpg", 512)) and min(kept.shape) == 512


def test_bwphoto_geometry_differs_from_matched_plain_tracev3():
    """LOAD-BEARING: at matched clahe/tone/resolution/native_res the ONLY remaining
    difference is the XDoG front end -- identical geometry would mean it is not firing."""
    _, pts_bw = _run("bwphoto_vs_a", {"style": "bwphoto"})
    _, pts_plain = _run("bwphoto_vs_b", {"style": "tracev3", **MATCHED})
    assert pts_bw and pts_plain and pts_bw != pts_plain


def test_key_absent_is_old_behaviour_and_front_end_cannot_be_smuggled(monkeypatch):
    calls = []
    orig = bw_photo.prepare_bw_xdog_source
    monkeypatch.setattr(bw_photo, "prepare_bw_xdog_source",
                        lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    _, pts_plain = _run("smuggle_plain", {"style": "tracev3"})
    _, pts_smug = _run("smuggle_key", {"style": "tracev3", "front_end": "bw_xdog"})
    assert not calls, "front end ran for a plain tracev3 request"
    assert pts_plain == pts_smug, "front_end smuggled under style='tracev3' changed the output"
    # positive control for the pair above: the same call under bwphoto DOES fire it
    _run("smuggle_ctrl", {"style": "bwphoto"})
    assert calls, "control failed: front end did not fire under style='bwphoto'"


def test_hostile_bwphoto_body_is_forced_back_to_the_recipe():
    _, pts_clean = _run("hostile_clean", {"style": "bwphoto"})
    _, pts_host = _run("hostile_body", {"style": "bwphoto", "clahe": False, "tone": "balanced",
                                        "resolution": 300, "native_res": True,
                                        "source_side_cap": 256, "front_end": "none"})
    assert pts_clean and pts_clean == pts_host


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
