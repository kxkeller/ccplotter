"""Opt-in "blue margin under black" (trap rule T1 + X + F), body flag blue_trap, default OFF.
Port parity: port_parity/*_parity.json.

OFF (default / explicit false): both layers equal the existing routing, byte-identical layer svg (null control, as
test_twopen_keytint_connector.py: the existing call is computed in-test). ON: blue connectors lie only inside blue U black (positive
control: synthetic connectors across white are rejected by the same checker); the fold conserves the union ink (positive control:
deleting the specks fails); blue mask contains A blue and the addition lies in black U closed holes (positive control: leak
fails); thresholds scale with pen width in mm, not pixels (same physical geometry on two grids folds identically).
Temp runtime home + work dir, Flask test client, never :5001.
"""
import math
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home
_HOME = _test_home.use_temp_home("ccplotter_test_home_kttrap_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_kttrap_"))
sys.path.insert(0, str(REPO))

import base64
import shutil

import cv2
import numpy as np
import pytest

from ccplotter import twopen_keytint as kt

IMG = REPO / "tests" / "fixtures" / "twotone" / "pagoda.jpg"
PAPER, MARGIN, BLACK = "150x110", 5.0, "bic_cristal_black"
W = 0.12


def _same(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


# ------------------------------------------------------------------ pure-mask controls (synthetic, fast)
MPP = 0.02       # 50 px/mm


def _synth():
    """black blob with a pen-scale hole-ish notch, blue strip touching it, a far-away speck of black, and white paper."""
    H, Wd = 400, 600
    black = np.zeros((H, Wd), bool)
    blue = np.zeros((H, Wd), bool)
    black[100:300, 100:300] = True                  # 4 x 4 mm black region
    blue[100:300, 300:320] = True                   # blue strip abutting the black (right side)
    black[150:154, 450:454] = True                  # speck: 0.08 x 0.08 mm = 0.0064 mm2 < F (0.1018)
    return blue, black


def test_trap_mask_invariants_and_scaling_constants():
    blue, black = _synth()
    t, tn = kt.trap_blue_mask(blue, black, MPP, W)
    spk = kt.trap_fold(black, np.pi * (1.5 * W) ** 2, MPP)[1]
    bf = black & ~spk
    inv = kt.trap_invariants(blue, black, t, tn, spk, bf)
    assert inv["ok"] and inv["contains_blue_A"] and inv["added_paper_px"] == 0 and inv["union_delta_px"] == 0
    assert inv["added_in_black_px"] > 0                    # the margin really grew into black (not a vacuous pass)
    # margin is d = 1 pen width = 6 px at 0.02 mm/px, only inside black
    assert t[200, 294:300].all() and not t[200, 293]            # exactly 6 px = 0.12 mm into black, no further
    assert not t[200, 320:].any()                                # nothing grows onto paper beyond the strip


def test_positive_control_leak_fails_the_addition_subset_check():
    blue, black = _synth()
    t, tn = kt.trap_blue_mask(blue, black, MPP, W, leak=True)       # dilation allowed onto paper
    spk = np.zeros_like(black)
    inv = kt.trap_invariants(blue, black, t, tn, spk, black)
    assert inv["added_paper_px"] > 0 and not inv["ok"]


def test_positive_control_deleting_the_fold_fails_conservation():
    blue, black = _synth()
    t, tn = kt.trap_blue_mask(blue, black, MPP, W)
    bf, spk, areas = kt.trap_fold(black, np.pi * (1.5 * W) ** 2, MPP)
    assert len(areas) == 1 and spk.sum() == 16                       # exactly the speck
    assert kt.trap_invariants(blue, black, t, tn, spk, bf)["union_delta_px"] == 0        # reassigned: conserved
    # DELETING control: the speck vanishes instead of moving to blue -> union loses its ink -> must be detected
    deleting = kt.trap_invariants(blue, black, t, tn, np.zeros_like(spk), bf)
    assert deleting["union_delta_px"] == -16 and not deleting["ok"]


def test_apply_blue_trap_reassigns_speck_to_blue_never_deletes():
    blue, black = _synth()
    tr = kt.apply_blue_trap({"black": black, "blue": blue}, MPP, W)
    assert tr["blue"][150:154, 450:454].all() and not tr["black"][150:154, 450:454].any()
    assert tr["black"][100:300, 100:300].all()                       # the large black region is untouched
    assert tr["info"]["fold_components"] == 1 and tr["info"]["invariants"]["ok"]
    assert (blue & ~tr["blue"]).sum() == 0


def test_thresholds_are_mm_not_pixels():
    """The same physical speck (area 0.0064 mm2, and a 2 x F-sized one) folds identically on a 2x finer grid."""
    for mpp in (0.02, 0.01):
        s = int(round(1 / mpp))                  # px per mm
        black = np.zeros((int(8 * s), int(12 * s)), bool)
        a = int(round(0.08 * s)); b = int(round(0.60 * s))       # 0.08 mm square (folds) ; 0.60 mm square = 0.36 mm2 (> F 0.1018, stays)
        black[2 * s:2 * s + a, 2 * s:2 * s + a] = True
        black[5 * s:5 * s + b, 5 * s:5 * s + b] = True
        _, spk, areas = kt.trap_fold(black, np.pi * (1.5 * W) ** 2, mpp)
        assert len(areas) == 1 and spk[2 * s:2 * s + a, 2 * s:2 * s + a].all() and not spk[5 * s:5 * s + b, 5 * s:5 * s + b].any(), mpp
    # and the pen width is the lever: a 0.24 mm pen has a 4x larger F, so the 0.60 mm square (0.36 mm2 < 0.407) folds too
    _, spk2, areas2 = kt.trap_fold(black, np.pi * (1.5 * 0.24) ** 2, mpp)
    assert len(areas2) == 2


def test_constants_are_documented_with_d_numbers():
    c = kt.CONSTANTS["blue_trap"]
    assert c["margin_pen_widths"] == 1.0 and c["fold_radius_pen_widths"] == 1.5 and c["hole_close_radius_pen_widths"] == 1.5
    assert c["blue_connector_distance_mm"] == 0.5 and "D1713" in c["d_numbers"] and "D1704" in c["d_numbers"]
    doc = (REPO / "docs" / "TUNING.md").read_text(encoding="utf-8")
    for k in ("TRAP_MARGIN_PEN", "TRAP_HOLE_RADIUS_PEN", "TRAP_FOLD_RADIUS_PEN", "TRAP_BLUE_CONNECTOR_MM"):
        assert k in doc, k


# ------------------------------------------------------------------ end to end on the pagoda fixture
@pytest.fixture(scope="module")
def prep(tmp_path_factory):
    wd = tmp_path_factory.mktemp("kttrap_prep")
    return kt.capture_plain_bwphoto(IMG, wd, paper_w=150.0, paper_h=110.0, margin=MARGIN, pen_id=BLACK,
                                    paper_spec=PAPER, body_extra={"connector_distance": 0.5})


def _gen(prep, tmp_path_factory, name, **kw):
    jd = tmp_path_factory.mktemp(name)
    return jd, kt.generate_layers(prep, 0.375, jd, {"blue": 0.12, "black": 0.12}, **kw)


@pytest.fixture(scope="module")
def run_default(prep, tmp_path_factory):
    return _gen(prep, tmp_path_factory, "kttrap_def")


@pytest.fixture(scope="module")
def run_off(prep, tmp_path_factory):
    return _gen(prep, tmp_path_factory, "kttrap_off", blue_trap=False, blue_fill_mm=0.12)


@pytest.fixture(scope="module")
def run_on(prep, tmp_path_factory):
    return _gen(prep, tmp_path_factory, "kttrap_on", blue_trap=True, blue_fill_mm=0.12)


def _bytes(run, color):
    jd, g = run
    return Path(g["layers"][color]).read_bytes() if g["layers"][color] else b""


def test_off_equals_head_routing_and_default(prep, run_default, run_off):
    """The existing behaviour: blue routed at 0.0, black at the body's 0.5, each from sep[color], caps as before. Computed in-test."""
    from ccplotter.stats_warnings import _route_low_lift_doc, ROUTE_WALK_MAX_SEGS_TWOPEN
    jd, g = run_default
    for color, cd, cap in (("blue", 0.0, None), ("black", 0.5, ROUTE_WALK_MAX_SEGS_TWOPEN)):
        if not g["strokes"][color]:
            continue
        ref_svg = jd / f"{color}.head_ref.routed.svg"
        kw = {} if cap is None else {"max_walk_segs": cap}
        _route_low_lift_doc(jd / f"{color}.clean.svg", ref_svg, cd, None, **kw)
        assert _same(g["strokes"][color], kt.load_strokes_mm(ref_svg)), color
        assert _same(run_off[1]["strokes"][color], g["strokes"][color]), color
        assert _bytes(run_off, color) == _bytes(run_default, color), color
    assert g["info"]["blue_trap"] == {"enabled": False} and run_off[1]["info"]["blue_trap"] == {"enabled": False}
    assert g["info"]["connectors"]["blue"]["connector_distance_mm"] == 0.0


def test_on_changes_the_layers_and_records_everything(run_default, run_on):
    g_off, g_on = run_default[1], run_on[1]
    assert not _same(g_on["strokes"]["blue"], g_off["strokes"]["blue"])        # positive: the option does something
    assert len(g_on["strokes"]["blue"]) + len(g_on["strokes"]["black"]) < len(g_off["strokes"]["blue"]) + len(g_off["strokes"]["black"])
    t = g_on["info"]["blue_trap"]
    assert t["enabled"] and t["pen_mm"] == 0.12 and abs(t["margin_mm"] - 0.12) < 1e-12
    assert t["invariants"]["ok"] and t["invariants"]["union_delta_px"] == 0
    assert t["constants"] == kt.CONSTANTS["blue_trap"]
    assert g_on["info"]["connectors"]["blue"]["connector_distance_mm"] == 0.5
    assert g_on["info"]["connectors"]["black"]["connector_distance_mm"] == 0.5     # body value, as OFF


def test_on_blue_connectors_only_inside_union_and_checker_rejects_white(prep, run_on):
    """Wiring check of X. Connectors of the ON blue layer are fed back through the SAME mask checker built from the
    union trace sidecar -> all accepted; synthetic connectors across white in that mask -> all rejected (positive control)."""
    from ccplotter.config import PX2MM
    from ccplotter.stage_e_mask import LayerMaskCheck, load_sidecar
    jd, g = run_on
    assert g["info"]["connectors"]["blue"]["connector_count"] > 0, "no blue connectors on the fixture: the test would be vacuous"
    tmask, fpx = load_sidecar(jd / "blue_chk_trace.svg")
    ox, oy = prep["c0_px"]
    chk = LayerMaskCheck(tmask, fpx, (ox, oy), float(prep["fit"]), float(prep.get("rot", 0.0)), (prep["tx"] / PX2MM, prep["ty"] / PX2MM))
    conns = kt.load_strokes_mm(jd / "blue.connectors.svg")
    assert len(conns) == g["info"]["connectors"]["blue"]["connector_count"]
    A = np.array([(s[0, 0] + 1j * s[0, 1]) / PX2MM for s in conns])
    B = np.array([(s[-1, 0] + 1j * s[-1, 1]) / PX2MM for s in conns])
    assert chk.ok(A, B).all()
    # positive control: endpoints of real mask pixels with white between them
    rng = np.random.default_rng(3)
    edge = np.argwhere(tmask & ~cv2.erode(tmask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    Lpx = 0.45 / (float(prep["fit"]) * PX2MM)
    rot = complex(math.cos(math.radians(prep.get("rot", 0.0))), math.sin(math.radians(prep.get("rot", 0.0))))
    c0, t = complex(ox, oy), complex(prep["tx"] / PX2MM, prep["ty"] / PX2MM)
    fwd = lambda p: c0 + rot * float(prep["fit"]) * (complex(p[0], p[1]) - c0) + t
    PA, PB = [], []
    for _ in range(40000):
        y, x = edge[rng.integers(len(edge))]
        th = rng.uniform(0, 2 * math.pi)
        x2, y2 = x + Lpx * math.cos(th), y + Lpx * math.sin(th)
        xi, yi = int(round(x2)), int(round(y2))
        if not (0 <= xi < tmask.shape[1] and 0 <= yi < tmask.shape[0]) or not tmask[yi, xi]:
            continue
        ts = np.linspace(0, 1, 40)
        sx, sy = np.rint(x + (x2 - x) * ts).astype(int), np.rint(y + (y2 - y) * ts).astype(int)
        if (~tmask[sy, sx]).sum() >= 3:
            PA.append(fwd((x, y)))
            PB.append(fwd((x2, y2)))
        if len(PA) >= 100:
            break
    assert len(PA) >= 10, "could not build across-white controls"
    assert not chk.ok(np.array(PA), np.array(PB)).any()


def test_blue_trap_flag_parse_all_entry_paths_and_not_forwarded_to_plain_job(prep, tmp_path_factory):
    from ccplotter.config import WORK_DIR
    from ccplotter.routes.twopen import _keytint_prepare, TwoPenRefusal
    b64 = base64.b64encode(IMG.read_bytes()).decode()
    base = {"image_b64": b64, "paper": PAPER, "margin": MARGIN}
    assert _keytint_prepare(dict(base), "kttrap_b0")["blue_trap"] is False
    assert _keytint_prepare(dict(base, blue_trap=True), "kttrap_b1")["blue_trap"] is True
    assert _keytint_prepare(dict(base, blue_trap=False), "kttrap_b2")["blue_trap"] is False
    with pytest.raises(TwoPenRefusal):
        _keytint_prepare(dict(base, blue_trap="yes"), "kttrap_b3")
    assert _keytint_prepare({"image": str(IMG), "paper": PAPER, "margin": MARGIN, "blue_trap": True}, "kttrap_i1")["blue_trap"] is True
    # canvas path: needs an uploaded src_<id>.jpg in WORK_DIR; blue_trap must be read AND dropped from the plain job's body
    did = "kttrap01"
    src = Path(WORK_DIR) / f"src_{did}.jpg"
    shutil.copyfile(IMG, src)
    try:
        kp = _keytint_prepare({"items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0}], "paper": PAPER, "margin": MARGIN,
                               "style": "bwphoto", "job_type": "preview", "blue_trap": True, "connector_distance": 0.5}, "kttrap_c1")
        assert kp["blue_trap"] is True and kp["prep"]["connector_distance"] == 0.5
    finally:
        src.unlink(missing_ok=True)
