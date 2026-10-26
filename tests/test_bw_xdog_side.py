"""bwphoto `bw_xdog_side` (native DEFAULT | 512 | 768 | 1024 | full).

  1. explicit 512 / "512" is byte-identical to the 512-side path: raster bytes,
     routed geometry, info.json "side" 512, and no new key in render_opts
  2. 1024 changes the raster shape and info.json records side 1024 (+ interpolation flag)
  3. native on a small source == its short side (and caps at 1024 on a big one)
  4. an invalid value is rejected (400); ignored outside bwphoto
  5. "full" uses the source's true short side, with no cap and no upsampling
Pins in tests/test_bw_photo.py are the flag-off oracle; this file does not touch them.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home
_test_home.use_temp_home("ccplotter_test_home_bwside_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_bwside_"))
sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import pytest

import ccplotter.bw_photo as bwp
from ccplotter.bw_photo import bw_xdog_raster, parse_bw_xdog_side, resolve_bw_xdog_side
from ccplotter.config import WORK_DIR
from test_bw_photo import _fixed_gray, _make_photo, _points_of, _process


def _png(path, h, w):
    cv2.imencode(".png", _fixed_gray(h, w, 11.0, 17.0))[1].tofile(str(path))
    return path


def test_parse_accepts_documented_values_and_rejects_the_rest():
    # no bw_xdog_side means native (the default)
    assert parse_bw_xdog_side(None) == "native" and parse_bw_xdog_side("") == "native"
    assert parse_bw_xdog_side(512) == 512 and parse_bw_xdog_side("512") == 512
    assert parse_bw_xdog_side("768") == 768 and parse_bw_xdog_side(1024) == 1024
    assert parse_bw_xdog_side("Native") == "native"
    for bad in ("2048", "abc", 0, -512, 1000, True, "1e3x", "512.5"):
        with pytest.raises(ValueError):
            parse_bw_xdog_side(bad)


def test_raster_flag_off_is_byte_identical(tmp_path):
    p = _png(tmp_path / "a.png", 120, 180)
    base = bw_xdog_raster(p)
    assert bw_xdog_raster(p, 512).tobytes() == base.tobytes()
    assert base.shape == (512, 768)


def test_raster_1024_changes_shape_and_native_small_source_is_its_short_side(tmp_path):
    p = _png(tmp_path / "a.png", 120, 180)
    r1024 = bw_xdog_raster(p, 1024)
    assert r1024.shape == (1024, 1536)
    rn = bw_xdog_raster(p, "native")
    assert rn.shape == (120, 180)                      # short side 120 = native
    big = _png(tmp_path / "big.png", 1500, 2000)
    assert bw_xdog_raster(big, "native").shape == (1024, 1365)   # capped at 1024
    # control: native on the small source is NOT the 512 raster
    assert rn.shape != bw_xdog_raster(p).shape


def test_resolve_reports_side_and_interpolation(tmp_path):
    small = _png(tmp_path / "s.png", 700, 900)
    assert resolve_bw_xdog_side(small, 512) == (512, None, None)
    assert resolve_bw_xdog_side(small, 1024) == (1024, 700, True)       # upsampled from 700
    assert resolve_bw_xdog_side(small, "native") == (700, 700, False)
    big = _png(tmp_path / "b.png", 1500, 2000)
    assert resolve_bw_xdog_side(big, 1024) == (1024, 1500, False)
    assert resolve_bw_xdog_side(big, "native") == (1024, 1500, False)


# ---- through the route ---------------------------------------------------------------------
def _go(did, extra, w=1500, h=1000):
    _make_photo(WORK_DIR / f"src_{did}.jpg", w, h)
    r = _process(did, {"style": "bwphoto", **extra})
    return r, did


def _info(did):
    f = sorted(WORK_DIR.glob(f"src_{did}_render_*.svg.info.json"))
    assert f, "no info.json"
    return json.loads(f[-1].read_text())


def _geom(r):
    body = r.get_json()
    return _points_of((WORK_DIR / body["debug_files"]["routed_svg"]).read_text(encoding="utf-8"))


def _xdog_png(r):
    body = r.get_json()
    return cv2.imdecode(np.fromfile(str(WORK_DIR / body["debug_files"]["xdog"]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def test_route_flag_off_identical_and_info_side_512():
    # debug files reuse one filename per job, so read each run's geometry BEFORE the next run
    # the absent-key run is native (not 512); explicit 512 / "512" are the 512-side raster and equal each other.
    r0, d0 = _go("bwside_off0", {}); g0 = _geom(r0)
    r1, d1 = _go("bwside_off1", {"bw_xdog_side": 512}); g1 = _geom(r1)
    r2, d2 = _go("bwside_off2", {"bw_xdog_side": "512"}); g2 = _geom(r2)
    assert r0.status_code == r1.status_code == r2.status_code == 200
    assert g1 and g1 == g2
    assert g0 != g1, "default (native) produced the 512 geometry -- default not flipped"
    fe0 = _info(d0)["front_end"]
    assert fe0["side_used"] == 1000 and fe0["side_requested"] == "native"
    for d in (d1, d2):
        fe = _info(d)["front_end"]
        assert fe["side"] == 512 and "side_used" not in fe and "interpolated" not in fe


def test_route_1024_changes_raster_and_records_side():
    # baseline for "1024 differs" is the explicit 512
    r0, _ = _go("bwside_a", {"bw_xdog_side": 512}); g0 = _geom(r0)
    r1, d1 = _go("bwside_b", {"bw_xdog_side": 1024})
    assert r1.status_code == 200, r1.get_json()
    assert min(_xdog_png(r1).shape) == 1024          # debug raster is the 1024 one
    fe = _info(d1)["front_end"]
    assert fe["side"] == 1024 and fe["side_used"] == 1024 and fe["side_requested"] == 1024
    assert fe["src_short_side"] == 1000 and fe["interpolated"] is True    # 1000 px source upsampled to 1024
    assert fe["sigma_rescaled"] is False
    assert g0 and _geom(r1) != g0, "1024 produced the 512 geometry -- flag not wired"


def test_route_native_small_source_equals_short_side():
    r, d = _go("bwside_nat", {"bw_xdog_side": "native"}, w=900, h=600)
    assert r.status_code == 200, r.get_json()
    fe = _info(d)["front_end"]
    assert fe["side_used"] == 600 and fe["src_short_side"] == 600 and fe["interpolated"] is False
    assert min(_xdog_png(r).shape) == 600


@pytest.mark.parametrize("bad", ["2048", "huge", 300, True])
def test_route_rejects_invalid_value(bad):
    r, _ = _go("bwside_bad", {"bw_xdog_side": bad})
    assert r.status_code == 400 and "bw_xdog_side" in r.get_json()["error"]


def test_key_ignored_outside_bwphoto():
    _make_photo(WORK_DIR / "src_bwside_plain.jpg", 1500, 1000)
    a = _process("bwside_plain", {"style": "tracev3"}); ga = _geom(a)
    b = _process("bwside_plain", {"style": "tracev3", "bw_xdog_side": 1024}); gb = _geom(b)
    assert a.status_code == b.status_code == 200
    assert ga and ga == gb


# ---- "full" (opt-in): the source's TRUE short side, no cap, no upsampling ----------
def test_parse_accepts_full():
    assert parse_bw_xdog_side("full") == "full" and parse_bw_xdog_side(" Full ") == "full"
    assert parse_bw_xdog_side(None) == "native"            # default unchanged


def test_resolve_full_vs_native_cap(tmp_path):
    big = _png(tmp_path / "b.png", 1405, 2000)
    assert resolve_bw_xdog_side(big, "full") == (1405, 1405, False)
    assert resolve_bw_xdog_side(big, "native") == (1024, 1405, False)   # positive control: native cap still exists
    small = _png(tmp_path / "s.png", 600, 900)
    assert resolve_bw_xdog_side(small, "full") == (600, 600, False)


def test_raster_full_shape_equals_source(tmp_path):
    big = _png(tmp_path / "b.png", 1405, 2000)
    assert bw_xdog_raster(big, "full").shape == (1405, 2000)
    assert bw_xdog_raster(big, "native").shape == (1024, 1458)          # control: differs from full
    assert bw_xdog_raster(big, "full").tobytes() == bw_xdog_raster(big, "full").tobytes()


def test_route_full_records_true_short_side():
    r, d = _go("bwside_full", {"bw_xdog_side": "full"}, w=1500, h=1200)
    assert r.status_code == 200, r.get_json()
    fe = _info(d)["front_end"]
    assert fe["side_used"] == 1200 and fe["side_requested"] == "full"
    assert fe["src_short_side"] == 1200 and fe["interpolated"] is False
    assert min(_xdog_png(r).shape) == 1200
