"""Two-pen key layer honours the body's connector_distance like the plain job; blue stays 0.0
(CONSTANTS["connector_rule"]) until a dense-gated chain exists. Absent -> 0.0 on every entry path.

Null control: with no connector_distance in the request (and with an explicit 0.0) both layers' routed strokes equal what
the existing routing produced. That behaviour IS `_route_low_lift_doc(clean, routed, 0.0, None)` on the same clean svg, so the reference
is computed in-test by exactly that call (a saved sha would go stale with any unrelated vpype/trace change).
Positive control: black at 0.5 differs from black at 0.0 (fewer strokes); blue at 0.5 is byte-identical to blue at 0.0.
Temp runtime home + work dir, Flask test client, never :5001. Three plain bwphoto jobs (~25 s each).
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home
_HOME = _test_home.use_temp_home("ccplotter_test_home_ktconn_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_ktconn_"))
sys.path.insert(0, str(REPO))

import numpy as np
import pytest

from ccplotter import twopen_keytint as kt

IMG = REPO / "tests" / "fixtures" / "twotone" / "pagoda.jpg"
PAPER, MARGIN, BLACK = "150x110", 5.0, "bic_cristal_black"


import base64


def _prep(tmp_path_factory, name, **extra):
    wd = tmp_path_factory.mktemp(name)
    return kt.capture_plain_bwphoto(IMG, wd, paper_w=150.0, paper_h=110.0, margin=MARGIN, pen_id=BLACK,
                                    paper_spec=PAPER, body_extra=extra or None)


def _gen(prep, tmp_path_factory, name):
    jd = tmp_path_factory.mktemp(name)
    return jd, kt.generate_layers(prep, 0.5, jd, {"blue": 0.5, "black": 0.5})


@pytest.fixture(scope="module")
def prep0(tmp_path_factory):
    return _prep(tmp_path_factory, "ktc0")


@pytest.fixture(scope="module")
def prep05(tmp_path_factory):
    return _prep(tmp_path_factory, "ktc05", connector_distance=0.5)


@pytest.fixture(scope="module")
def run0(prep0, tmp_path_factory):
    return _gen(prep0, tmp_path_factory, "ktc_def")


@pytest.fixture(scope="module")
def run05(prep05, tmp_path_factory):
    return _gen(prep05, tmp_path_factory, "ktc_05")


def _same(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


def _bytes(run, color):
    jd, g = run
    return Path(g["layers"][color]).read_bytes() if g["layers"][color] else b""


def test_rule_is_visible_in_constants_and_info(run0, run05):
    assert "blue connectors: chain only (D1695), 0.0 until chain lands" in kt.CONSTANTS["connector_rule"]
    assert kt.CONSTANTS["blue_connector_distance_mm"] == 0.0
    c0, c5 = run0[1]["info"]["connectors"], run05[1]["info"]["connectors"]
    assert c0["black"]["connector_distance_mm"] == 0.0 and c0["black"]["connector_count"] == 0
    assert c5["black"]["connector_distance_mm"] == 0.5 and c5["black"]["connector_count"] > 0
    assert c5["black"]["connector_draw_length_mm"] > 0
    assert c5["blue"]["connector_distance_mm"] == 0.0 and c5["blue"]["connector_count"] == 0     # blue excluded


def test_capture_reads_the_plain_paths_value(prep0, prep05):
    assert prep0["connector_distance"] == 0.0 and prep05["connector_distance"] == 0.5
    assert kt.layer_geom(prep05)["connector_distance"] == 0.5


def test_null_default_and_explicit_zero_equal_head_routing(prep0, run0, tmp_path_factory):
    from ccplotter.stats_warnings import _route_low_lift_doc
    jd, g = run0
    run_zero = _gen(dict(prep0, connector_distance=0.0), tmp_path_factory, "ktc_zero")
    checked = 0
    for color in ("blue", "black"):
        if not g["strokes"][color]:
            continue
        ref_svg = jd / f"{color}.head_ref.routed.svg"
        _route_low_lift_doc(jd / f"{color}.clean.svg", ref_svg, 0.0, None)  # the unchanged hardcoded call
        ref = kt.load_strokes_mm(ref_svg)
        assert _same(g["strokes"][color], ref), color
        assert _same(run_zero[1]["strokes"][color], ref), color
        assert _bytes(run0, color) == _bytes(run_zero, color), color            # byte-identical layer svg
        checked += 1
    assert checked >= 1


def test_positive_control_black_at_half_mm_differs_from_zero(run0, run05):
    n0, n5 = len(run0[1]["strokes"]["black"]), len(run05[1]["strokes"]["black"])
    assert n5 < n0, (n0, n5)                   # fused strokes -> fewer pen lifts; fails if the value is ignored
    assert _bytes(run0, "black") != _bytes(run05, "black")


def test_blue_at_half_mm_is_byte_identical_to_zero(run0, run05):
    assert _same(run0[1]["strokes"]["blue"], run05[1]["strokes"]["blue"])
    assert _bytes(run0, "blue") == _bytes(run05, "blue") and len(_bytes(run0, "blue")) > 0


def test_b64_and_image_paths_carry_connector_distance_to_black(prep0, run0, tmp_path_factory):
    from ccplotter.routes.twopen import _keytint_prepare
    b64 = base64.b64encode(IMG.read_bytes()).decode()
    kp = _keytint_prepare({"image_b64": b64, "paper": PAPER, "margin": MARGIN, "connector_distance": 0.5}, "ktc_b64")
    assert kp["prep"]["connector_distance"] == 0.5
    _, g = _gen(kp["prep"], tmp_path_factory, "ktc_b64g")
    assert len(g["strokes"]["black"]) < len(run0[1]["strokes"]["black"])      # positive: black changed
    assert _same(g["strokes"]["blue"], run0[1]["strokes"]["blue"])            # blue still 0.0
    # (the `image` path shares the same _cd -> body_extra code as image_b64; only the b64 path is run end-to-end here;
    # absent -> 0.0 is covered by prep0/run0 above, which carry no connector_distance)
    with pytest.raises(Exception):
        _keytint_prepare({"image": str(IMG), "paper": PAPER, "margin": MARGIN, "connector_distance": "x"}, "ktc_bad")
