"""Opt-in "Hatched blue (lighter, fewer pen lifts)", body flag hatched_blue, default OFF.
Port parity: tests/fixtures/hatchblue_port/*_parity.json
(tram 20,550 / heli 5,247 lifts, pointwise max diff 0.0 mm against the prototype's saved strokes).

OFF (default / explicit false): both layers byte-identical to the existing routing (null control, the existing call computed in-test). ON: blue is a hatch, fewer lifts, lower
blue ink than blue_trap's solid fill (positive control: the solid fill really is more); the hatch stays inside its clipped region (positive control: clip 0 fails the
containment count); the linker's connectors lie inside blue u black (positive control: synthetic connectors across white rejected); every sampled run obeys the
ordered-dither rule recomputed here from a literal table; ON + stay_down_over_ink runs and the black stay-down joins stay inside black's own ink.
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
_HOME = _test_home.use_temp_home("ccplotter_test_home_kthatch_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_kthatch_"))
sys.path.insert(0, str(REPO))

import base64
import json

import cv2
import numpy as np
import pytest

from ccplotter import twopen_keytint as kt
from ccplotter import twopen_hatch as th
from ccplotter.config import PX2MM
from ccplotter.stage_e_mask import LayerMaskCheck, load_sidecar

IMG = REPO / "tests" / "fixtures" / "twotone" / "pagoda.jpg"
PAPER, MARGIN, BLACK = "150x110", 5.0, "bic_cristal_black"
W = 0.12
PAR = REPO / "tests" / "fixtures" / "hatchblue_port"


def _same(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


@pytest.fixture(scope="module")
def prep(tmp_path_factory):
    wd = tmp_path_factory.mktemp("kthatch_prep")
    return kt.capture_plain_bwphoto(IMG, wd, paper_w=150.0, paper_h=110.0, margin=MARGIN, pen_id=BLACK,
                                    paper_spec=PAPER, body_extra={"connector_distance": 0.5})


def _gen(prep, tmp_path_factory, name, **kw):
    jd = tmp_path_factory.mktemp(name)
    return jd, kt.generate_layers(prep, 0.375, jd, {"blue": 0.12, "black": 0.12}, **kw)


@pytest.fixture(scope="module")
def run_default(prep, tmp_path_factory):
    return _gen(prep, tmp_path_factory, "kthatch_def")


@pytest.fixture(scope="module")
def run_off(prep, tmp_path_factory):
    return _gen(prep, tmp_path_factory, "kthatch_off", hatched_blue=False, blue_fill_mm=0.12)


@pytest.fixture(scope="module")
def run_trap(prep, tmp_path_factory):
    return _gen(prep, tmp_path_factory, "kthatch_trap", blue_trap=True, blue_fill_mm=0.12)


@pytest.fixture(scope="module")
def run_on(prep, tmp_path_factory):
    return _gen(prep, tmp_path_factory, "kthatch_on", hatched_blue=True, blue_fill_mm=0.12)


@pytest.fixture(scope="module")
def run_on_trap(prep, tmp_path_factory):
    return _gen(prep, tmp_path_factory, "kthatch_ontrap", hatched_blue=True, blue_trap=True, blue_fill_mm=0.12)


@pytest.fixture(scope="module")
def run_on_sd(prep, tmp_path_factory):
    return _gen(prep, tmp_path_factory, "kthatch_onsd", hatched_blue=True, stay_down_over_ink=True, blue_fill_mm=0.12)


def _bytes(run, color):
    jd, g = run
    return Path(g["layers"][color]).read_bytes() if g["layers"][color] else b""


# ------------------------------------------------------------------ constants
def test_constants_documented_with_derivation_and_in_tuning():
    c = kt.CONSTANTS["hatched_blue"]
    assert c["sigma_pen_widths"] == 1.0 and c["link_radius_pen_widths"] == 8.0 and c["link_candidates_cost_guard"] == 40
    assert c["clip_pen_widths"] == 0.5 and "0.11 mm" in c["clip_mm_rule"] and c["min_run_pen_widths"] == 1.0 and c["dither_levels"] == 8
    assert c["line_pitch_pen_widths"] == [1.0, 8.0] and "1,000,000" in c["walk_cap_not_used"]
    assert abs(th.CLIP_PEN * W + 1.0 / th.HATCH_PPM - 0.11) < 1e-12
    assert "B1c" in c["excluded"] and "H1" in c["excluded"]
    doc = (REPO / "docs" / "TUNING.md").read_text(encoding="utf-8")
    for k in ("SIGMA_PEN", "CLIP_PEN", "LINK_RADIUS_PEN", "LINK_CANDIDATES", "WALK_CAP_UNUSED", "hatched_blue", "0.07", "0.27", "Hatched blue (lighter, fewer pen lifts)"):
        assert k in doc, k


# ------------------------------------------------------------------ OFF identity
def test_off_equals_head_routing_and_default(run_default, run_off):
    from ccplotter.stats_warnings import _route_low_lift_doc, ROUTE_WALK_MAX_SEGS_TWOPEN
    jd, g = run_default
    for color, cd, cap in (("blue", 0.0, None), ("black", 0.5, ROUTE_WALK_MAX_SEGS_TWOPEN)):
        ref_svg = jd / f"{color}.head_ref.routed.svg"
        _route_low_lift_doc(jd / f"{color}.clean.svg", ref_svg, cd, None, **({} if cap is None else {"max_walk_segs": cap}))
        assert _same(g["strokes"][color], kt.load_strokes_mm(ref_svg)), color
        assert _same(run_off[1]["strokes"][color], g["strokes"][color]), color
        assert _bytes(run_off, color) == _bytes(run_default, color), color
    assert "hatched_blue" not in g["info"] and "hatched_blue" not in run_off[1]["info"]
    assert g["info"]["blue_trap"] == {"enabled": False}


# ------------------------------------------------------------------ ON
def test_on_changes_blue_uses_trap_black_and_records_everything(run_default, run_trap, run_on):
    g0, gt, g1 = run_default[1], run_trap[1], run_on[1]
    assert not _same(g1["strokes"]["blue"], g0["strokes"]["blue"])
    assert _same(g1["strokes"]["black"], gt["strokes"]["black"]), "black must be blue_trap's F black"
    assert len(g1["strokes"]["blue"]) < len(gt["strokes"]["blue"])        # fewer pen lifts than the solid trap fill
    hb = g1["info"]["hatched_blue"]
    assert hb["enabled"] and hb["pen_mm"] == 0.12 and hb["constants"] == kt.CONSTANTS["hatched_blue"]
    assert hb["layer"]["hatch_violations"] == 0 and hb["layer"]["runs"] > 0 and "crack_figures" in hb
    bt = g1["info"]["blue_trap"]
    assert bt["enabled"] is False and bt["black_fold_implied_by_hatched_blue"] and bt["invariants"]["ok"]
    assert g1["info"]["connectors"]["black"]["connector_distance_mm"] == 0.5        # body value, as OFF
    assert g1["info"]["connectors"]["blue"]["connector_distance_mm"] == pytest.approx(8 * 0.12)


def test_precedence_hatched_blue_ignores_trap_blue(run_on, run_on_trap):
    """Both flags on: identical strokes to hatched alone (blue_trap contributes only the black fold, which hatched already implies)."""
    for color in ("blue", "black"):
        assert _same(run_on[1]["strokes"][color], run_on_trap[1]["strokes"][color]), color
    assert run_on_trap[1]["info"]["blue_trap"]["enabled"] is True and "hatched_blue" in run_on_trap[1]["info"]


def _raster(strokes, paper_w, paper_h, ppm=20.0, width=W):
    m = np.zeros((int(paper_h * ppm), int(paper_w * ppm)), np.uint8)
    for s in strokes:
        if len(s) >= 2:
            cv2.polylines(m, [(s * ppm * 16).astype(np.int32).reshape(-1, 1, 2)], False, 255, max(1, int(round(width * ppm))), cv2.LINE_8, shift=4)
    return m > 0


def _visible_blue(run, ppm=20.0):
    """blue ink (mm2) NOT hidden under the black ink: the hatch also runs under black (c = 1 there), which is invisible; the
    'about half the blue coverage' of the tooltip is the visible blue."""
    bl = _raster(run[1]["strokes"]["blue"], 150.0, 110.0)
    bk = _raster(run[1]["strokes"]["black"], 150.0, 110.0)
    return float((bl & ~bk).sum()) / ppm ** 2


def test_hatch_blue_visible_ink_is_lower_than_solid_fill_with_positive_control(run_trap, run_on, run_default):
    solid, hatch = _visible_blue(run_trap), _visible_blue(run_on)
    assert 0 < hatch < solid, (hatch, solid)
    # positive control: the measure can fail -- the plain solid blue layer is also more than the hatch, and a layer is not lower than itself
    assert _visible_blue(run_default) > hatch
    assert not (_visible_blue(run_trap) < solid)


def test_connectors_only_inside_union_and_checker_rejects_white(prep, run_on):
    jd, g = run_on
    assert g["info"]["connectors"]["blue"]["connector_count"] > 0, "vacuous: no hatch connectors"
    tmask, fpx = load_sidecar(jd / "blue_chk_trace.svg")
    ox, oy = prep["c0_px"]
    chk = LayerMaskCheck(tmask, fpx, (ox, oy), float(prep["fit"]), float(prep.get("rot", 0.0)), (prep["tx"] / PX2MM, prep["ty"] / PX2MM), tol_px=0.25)
    conns = kt.load_strokes_mm(jd / "blue.connectors.svg")
    assert len(conns) == g["info"]["connectors"]["blue"]["connector_count"]
    assert max(np.hypot(*(s[-1] - s[0])) for s in conns) <= 8 * W + 1e-3
    A = np.array([(s[0, 0] + 1j * s[0, 1]) / PX2MM for s in conns])
    B = np.array([(s[-1, 0] + 1j * s[-1, 1]) / PX2MM for s in conns])
    assert chk.ok(A, B).all()
    rng = np.random.default_rng(3)
    edge = np.argwhere(tmask & ~cv2.erode(tmask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    Lpx = 0.45 / (float(prep["fit"]) * PX2MM)
    rot = complex(math.cos(math.radians(prep.get("rot", 0.0))), math.sin(math.radians(prep.get("rot", 0.0))))
    c0, t = complex(ox, oy), complex(prep["tx"] / PX2MM, prep["ty"] / PX2MM)
    fwd = lambda p: c0 + rot * float(prep["fit"]) * (complex(p[0], p[1]) - c0) + t
    PA, PB = [], []
    for _ in range(40000):
        y, x = edge[rng.integers(len(edge))]
        a = rng.uniform(0, 2 * math.pi)
        x2, y2 = x + Lpx * math.cos(a), y + Lpx * math.sin(a)
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


def test_hatch_clip_containment_and_dither_rule(prep):
    """The runs stay >= clip inside U (positive control: an unclipped generation fails the same count) and obey the ordered-dither rule
    recomputed from a LITERAL rank table (independent of _bitrev)."""
    sep = kt.separate(prep["gray"], prep["opts"], prep["mm_per_px"], 0.375)
    c, U, _ = th.target_fields(sep, prep["gray"], prep["opts"], prep["mm_per_px"], W)
    geom = kt.layer_geom(prep)
    runs, dt, Up, clip = th.gen_hatch(c, U, geom, 150.0, 110.0, W)
    assert len(runs) > 100 and abs(clip - 0.11) < 1e-12
    assert th.hatch_violations(runs, dt, clip) == 0
    loose, _, _, _ = th.gen_hatch(c, U, geom, 150.0, 110.0, W, clip_mm=0.0)
    assert len(loose) > len(runs) and th.hatch_violations(loose, dt, clip) > 0        # positive control
    RANK = [0, 4, 2, 6, 1, 5, 3, 7]
    Mw = th._warp_matrix(geom, th.HATCH_PPM)
    Cp = cv2.warpAffine(c.astype(np.float32), Mw, (int(150 * th.HATCH_PPM), int(110 * th.HATCH_PPM)), flags=cv2.INTER_LINEAR)
    nv = np.array([-math.sin(math.radians(45)), math.cos(math.radians(45))])
    def qat(p):          # max level in the 3x3 raster neighbourhood: the generator samples every 0.025 mm and rounds to a pixel, the midpoint may round to a neighbour
        ix, iy = int(round(p[0] * th.HATCH_PPM)), int(round(p[1] * th.HATCH_PPM))
        return max(math.ceil(8 * float(Cp[iy + dy, ix + dx]) - 1e-6) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    for r in runs[::7]:
        mid = (r[0] + r[1]) / 2
        k = int(round(mid @ nv / W - 0.5))
        assert qat(mid) > RANK[k % 8], (k, qat(mid))
    # negative control of the check itself: a WRONG rank table (reversed) must be violated by real runs
    WRONG = RANK[::-1]
    violated = sum(int(not (qat((r[0] + r[1]) / 2) > WRONG[int(round(((r[0] + r[1]) / 2) @ nv / W - 0.5)) % 8])) for r in runs[::7])
    assert violated > 0


def test_recorded_prototype_parity_artifacts():
    """The committed parity files (tests/fixtures/hatchblue_port/): ON == the prototype pointwise on tram and heli. Static read."""
    for key, total in (("tram", 20550), ("heli", 5247)):
        j = json.loads((PAR / f"{key}_parity.json").read_text(encoding="utf-8"))
        assert j["blue_pointwise_identical"] and j["black_pointwise_identical"] and j["total_lifts_equal_expected"]
        assert j["app"]["total_lifts"] == total and j["blue_max_abs_diff_mm"] == 0.0 and j["black_max_abs_diff_mm"] == 0.0


# ------------------------------------------------------------------ composition with stay_down_over_ink
def test_hatched_plus_stay_down_black_joins_inside_own_ink_and_blue_unchanged(prep, run_on, run_on_sd):
    jd, g = run_on_sd
    assert _same(g["strokes"]["blue"], run_on[1]["strokes"]["blue"]), "stay_down must not touch the hatched blue layer"
    n0, n1 = len(run_on[1]["strokes"]["black"]), len(g["strokes"]["black"])
    assert n1 < n0, "vacuous: stay_down fused nothing on black"
    rec = g["info"]["stay_down_over_ink"]
    assert rec["layers"]["black"]["totals"]["fused"] == n0 - n1 and "not_applicable" in rec["layers"]["blue"]
    ox, oy = prep["c0_px"]
    tmask, fpx = load_sidecar(jd / "black_sd_trace.svg")
    chk = LayerMaskCheck(tmask, fpx, (ox, oy), float(prep["fit"]), float(prep.get("rot", 0.0)), (prep["tx"] / PX2MM, prep["ty"] / PX2MM), tol_px=0.25)
    conns = kt.load_strokes_mm(jd / "black.connectors.svg")
    long_ = [s for s in conns if np.hypot(*(s[-1] - s[0])) > 0.6]
    assert long_
    A = np.array([(s[0, 0] + 1j * s[0, 1]) / PX2MM for s in long_])
    B = np.array([(s[-1, 0] + 1j * s[-1, 1]) / PX2MM for s in long_])
    assert chk.ok(A, B).all()
    # blue hatch connectors stay inside blue u black (the linker's own mask)
    umask, ufpx = load_sidecar(jd / "blue_chk_trace.svg")
    uchk = LayerMaskCheck(umask, ufpx, (ox, oy), float(prep["fit"]), float(prep.get("rot", 0.0)), (prep["tx"] / PX2MM, prep["ty"] / PX2MM), tol_px=0.25)
    bc = kt.load_strokes_mm(jd / "blue.connectors.svg")
    assert uchk.ok(np.array([(s[0, 0] + 1j * s[0, 1]) / PX2MM for s in bc]), np.array([(s[-1, 0] + 1j * s[-1, 1]) / PX2MM for s in bc])).all()


# ------------------------------------------------------------------ route flag
def test_flag_parse_all_entry_paths_and_not_forwarded_to_plain_job(prep):
    import shutil
    from ccplotter.config import WORK_DIR
    from ccplotter.routes.twopen import _keytint_prepare, TwoPenRefusal
    b64 = base64.b64encode(IMG.read_bytes()).decode()
    base = {"image_b64": b64, "paper": PAPER, "margin": MARGIN}
    assert _keytint_prepare(dict(base), "kthatch_b0")["hatched_blue"] is False
    assert _keytint_prepare(dict(base, hatched_blue=True), "kthatch_b1")["hatched_blue"] is True
    assert _keytint_prepare(dict(base, hatched_blue=False), "kthatch_b2")["hatched_blue"] is False
    with pytest.raises(TwoPenRefusal):
        _keytint_prepare(dict(base, hatched_blue="yes"), "kthatch_b3")
    assert _keytint_prepare({"image": str(IMG), "paper": PAPER, "margin": MARGIN, "hatched_blue": True}, "kthatch_i1")["hatched_blue"] is True
    did = "kthatch01"
    src = Path(WORK_DIR) / f"src_{did}.jpg"
    shutil.copyfile(IMG, src)
    try:
        kp = _keytint_prepare({"items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0}], "paper": PAPER, "margin": MARGIN,
                               "style": "bwphoto", "job_type": "preview", "hatched_blue": True, "connector_distance": 0.5}, "kthatch_c1")
        assert kp["hatched_blue"] is True and kp["prep"]["connector_distance"] == 0.5
    finally:
        src.unlink(missing_ok=True)
