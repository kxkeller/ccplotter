"""Tests for three opt-in fixes, all default OFF:

  core_fill="hybrid"  (+ core_r_mm)   ccplotter/core_fill.py, trace_v3 contour fill
  edge_inset_mm                       same module, sub-pixel first ring
  bw_clahe / bw_cut                   bwphoto-only mask overrides (mask.py, routes/core.py)

Every positive control here is paired with a null: the hybrid test also proves
today's rings DO violate the rule on the same image, the inset test also shows
the baseline ring sits elsewhere, the bw tests also show the keys are inert
outside style="bwphoto" and when set to their defaults.
"""
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home; pen/calibration never resolve from CWD
_test_home.use_temp_home("ccplotter_test_home_corefill_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_corefill_"))
sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import pytest
from PIL import Image

from ccplotter.config import BW_CLAHE_SOFT_CLIP, BW_PHOTO_RECIPE, CORE_R_MM_DEFAULT, WORK_DIR

MM = 0.0474                      # true mm per working px used by these synthetics
FILL_MM = 0.12                   # BIC-like pitch
FILL_PX = FILL_MM / MM


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pc(mm_per_px):
    """`_place_cell` (composite's only producer of the mm/px scale) pinned to an exact mm/px:
    true_size=True -> fit 1.0 -> mm/px = scale * PX2MM, so scale = mm_per_px / PX2MM."""
    from ccplotter.config import PX2MM
    return [1.0, 1.0, 0.0, mm_per_px / PX2MM, True]


def _trace(img_black_on_white, extra, tmp_path, name="t"):
    """Run the real _trace_svg on a bilevel image; return list of polylines."""
    from ccplotter.trace_v3 import _trace_svg
    out = tmp_path / f"{name}.svg"
    opts = {"fill_px": FILL_PX, "fill_mm": FILL_MM, "threshold_mode": "otsu", "pocket_link": 0,
            "_place_cell": _pc(MM), **extra}
    _trace_svg(img_black_on_white, out, opts)
    polys = []
    for ps in re.findall(r'points="([^"]+)"', out.read_text(encoding="utf-8")):
        polys.append([tuple(float(v) for v in t.split(",")) for t in ps.split()])
    return polys


def _synth_discs_and_capsules():
    """Discs (top row) and horizontal capsules (stacked below), radius 0.3..2.0 mm,
    black on white (bilevel -> no pre-blur, exact shapes)."""
    img = np.full((1400, 1500), 255, np.uint8)
    radii = [0.3, 0.5, 0.8, 1.2, 2.0]
    x = 80
    for r in radii:
        rp = r / MM
        cv2.circle(img, (x + int(round(rp)), 120), int(round(rp)), 0, -1)
        x += int(2 * rp) + 50
    y = 300
    for r in radii:
        rp = r / MM
        y += int(round(rp)) + 30
        cv2.line(img, (100, y), (500, y), 0, max(1, int(round(2 * rp))), cv2.LINE_8)
        cv2.circle(img, (100, y), int(round(rp)), 0, -1)
        cv2.circle(img, (500, y), int(round(rp)), 0, -1)
        y += int(round(rp))
    return img


def _is_closed(p):
    return len(p) >= 3 and abs(p[0][0] - p[-1][0]) < 0.6 and abs(p[0][1] - p[-1][1]) < 0.6


def _inscribed_px(p, canvas_shape, ss=4):
    """Inscribed-circle radius (px) of a closed polyline, by rasterising the filled
    polygon at 4x and taking max of its distance transform."""
    pts = np.round(np.array(p, np.float64) * ss).astype(np.int32)
    x0, y0 = pts.min(axis=0) - 4
    m = np.zeros((pts[:, 1].max() - y0 + 8, pts[:, 0].max() - x0 + 8), np.uint8)
    cv2.fillPoly(m, [pts - np.array([x0, y0])], 255)
    return float(cv2.distanceTransform(m, cv2.DIST_L2, 5).max()) / ss


# ---------------------------------------------------------------------------
# core_fill = hybrid
# ---------------------------------------------------------------------------

def test_hybrid_no_ring_below_core_r_and_cores_are_straight(tmp_path):
    img = _synth_discs_and_capsules()
    core_r_px = CORE_R_MM_DEFAULT / MM
    tol = 1.6                                           # px: 0.4px approxPolyDP + pixel quantisation of dt

    base = _trace(img, {}, tmp_path, "base")
    hyb = _trace(img, {"core_fill": "hybrid", "core_r_mm": CORE_R_MM_DEFAULT}, tmp_path, "hyb")

    # NULL / positive control for the metric: today's rings DO include sub-threshold closed rings
    base_small = [p for p in base if _is_closed(p) and _inscribed_px(p, img.shape) < core_r_px - tol]
    assert base_small, "control failed: baseline has no closed ring under core_r -- test could not fail"

    closed = [p for p in hyb if _is_closed(p)]
    open_ = [p for p in hyb if not _is_closed(p)]
    assert closed and open_
    bad = [(round(_inscribed_px(p, img.shape) * MM, 3)) for p in closed
           if _inscribed_px(p, img.shape) < core_r_px - tol]
    assert not bad, f"closed rings with inscribed radius < core_r_mm: {bad} mm"

    # every core pass is made of straight rows: long segments share one direction per
    # core (within 2 deg) and successive rows sit one pitch apart (+-1 px)
    for p in open_:
        pts = np.array(p)
        segs = pts[1:] - pts[:-1]
        ln = np.hypot(segs[:, 0], segs[:, 1])
        long_ = segs[ln > 3.2 * FILL_PX]
        if len(long_) < 2:
            continue
        ang = np.degrees(np.arctan2(long_[:, 1], long_[:, 0])) % 180.0
        d = np.abs(((ang - ang[0]) + 90.0) % 180.0 - 90.0)
        assert d.max() < 2.0, f"rows not parallel: {d.max():.2f} deg"
    # pitch between consecutive long rows within one chain
    pitches = []
    for p in open_:
        pts = np.array(p)
        segs = pts[1:] - pts[:-1]
        ln = np.hypot(segs[:, 0], segs[:, 1])
        idx = [i for i in range(len(segs)) if ln[i] > 3.2 * FILL_PX]
        for a, b in zip(idx, idx[1:]):
            if b == a + 2:                              # row, connector, row
                u = segs[a] / ln[a]
                n = np.array([-u[1], u[0]])
                pitches.append(abs(float(np.dot(pts[b] - pts[a], n))))
    assert pitches, "no multi-row cores in the synthetic -- test could not check pitch"
    assert max(pitches) <= FILL_PX + 0.05, max(pitches)     # never wider than the ring pitch
    assert abs(np.median(pitches) - FILL_PX) < 0.5, (np.median(pitches), FILL_PX)

    # ring k that IS emitted is bit-for-bit today's ring: hybrid rings are a subset of baseline rings
    base_set = {tuple(map(tuple, p)) for p in base}
    kept = [p for p in closed if tuple(map(tuple, p)) in base_set]
    assert kept, "hybrid rings no longer coincide with today's rings"


def test_hybrid_round_cores_get_straight_passes_not_points_or_loops(tmp_path):
    """r 0.3mm disc: whole shape is below core_r -> no ring at all, only straight rows."""
    img = np.full((200, 200), 255, np.uint8)
    cv2.circle(img, (100, 100), int(round(0.3 / MM)), 0, -1)
    hyb = _trace(img, {"core_fill": "hybrid"}, tmp_path)
    assert hyb and not [p for p in hyb if _is_closed(p)], "closed loop emitted for a sub-threshold disc"
    pts = np.array([q for p in hyb for q in p])
    assert len(pts) >= 4
    segs = [np.array(p[i + 1]) - np.array(p[i]) for p in hyb for i in range(len(p) - 1)]
    assert max(np.hypot(*s) for s in segs) > 0.5 * (2 * 0.3 / MM), "no chord-length straight pass"


def test_hybrid_core_mask_debug_and_audit_keys_only_when_active(tmp_path):
    from ccplotter.trace_v3 import _trace_svg
    import ccplotter.trace_v3 as tv3
    img = _synth_discs_and_capsules()
    opts = {"fill_px": FILL_PX, "fill_mm": FILL_MM, "threshold_mode": "otsu", "_place_cell": _pc(MM)}
    _trace_svg(img, tmp_path / "off.svg", dict(opts), debug_prefix=str(tmp_path / "dbg_off"))
    off_audit = dict(tv3._TRACE_AUDIT)
    assert not (tmp_path / "dbg_off_core_mask.png").exists()
    assert "core_fill" not in off_audit and "edge_inset_mm" not in off_audit and "flag_mm_per_px" not in off_audit
    _trace_svg(img, tmp_path / "on.svg", {**opts, "core_fill": "hybrid"}, debug_prefix=str(tmp_path / "dbg_on"))
    au = dict(tv3._TRACE_AUDIT)
    cm = cv2.imread(str(tmp_path / "dbg_on_core_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert cm is not None and cm.shape == img.shape and cm.max() == 255
    assert au["core_fill"] == "hybrid" and au["n_core_regions"] >= 5 and au["n_core_passes"] >= 5
    assert au["flag_mm_per_px_source"] == "placed_fit_from_mask_bbox" and abs(au["flag_mm_per_px"] - MM) < 1e-6


def test_hybrid_uses_true_scale_not_the_backsolve(tmp_path):
    """Same image, same fill_px, different TRUE mm/px: the core threshold in px moves."""
    img = np.full((400, 400), 255, np.uint8)
    cv2.circle(img, (200, 200), 14, 0, -1)               # radius 14 px
    opts = {"core_fill": "hybrid", "core_r_mm": 0.45}
    a = _trace(img, {**opts, "_place_cell": _pc(0.0474)}, tmp_path, "a")   # core_r = 9.5 px -> rings exist
    b = _trace(img, {**opts, "_place_cell": _pc(0.0900)}, tmp_path, "b")   # core_r = 5.0 px -> more rings
    ca = len([p for p in a if _is_closed(p)])
    cb = len([p for p in b if _is_closed(p)])
    assert cb > ca, (ca, cb)


def test_place_cell_derives_true_scale_from_the_masks_own_bbox(tmp_path):
    """`_place_cell` = [cellW, cellH, rot, scale, true_size] -> mm/px = the composite final-fit
    arithmetic (0.9 * min(cell / rotated ink bbox) * scale * PX2MM) on the trace's own mask bbox."""
    import ccplotter.trace_v3 as tv3
    from ccplotter.config import PX2MM
    img = np.full((600, 800), 255, np.uint8)
    img[100:500, 200:600] = 0                              # 400 x 400 px square
    base = {"fill_px": FILL_PX, "fill_mm": FILL_MM, "threshold_mode": "otsu", "pocket_link": 0,
            "core_fill": "hybrid"}
    from ccplotter.trace_v3 import _trace_svg
    _trace_svg(img, tmp_path / "a.svg", {**base, "_place_cell": [237.0, 156.0, 0.0, 1.0, False]})
    a = dict(tv3._TRACE_AUDIT)
    want = 0.9 * min(237.0 / (400 * PX2MM), 156.0 / (400 * PX2MM)) * PX2MM
    assert a["flag_mm_per_px_source"] == "placed_fit_from_mask_bbox"
    assert abs(a["flag_mm_per_px"] - want) < 1e-5, (a["flag_mm_per_px"], want)
    # user scale and a 90 deg rotation of a non-square mask move it the way the final fit does
    img2 = np.full((600, 800), 255, np.uint8)
    img2[100:500, 100:700] = 0                             # 600 wide x 400 tall
    _trace_svg(img2, tmp_path / "b.svg", {**base, "_place_cell": [237.0, 156.0, 90.0, 0.5, False]})
    b = dict(tv3._TRACE_AUDIT)
    want_b = 0.9 * min(237.0 / (400 * PX2MM), 156.0 / (600 * PX2MM)) * 0.5 * PX2MM   # rot 90 swaps w/h
    assert abs(b["flag_mm_per_px"] - want_b) < 1e-5, (b["flag_mm_per_px"], want_b)
    # with no placement at all, the labelled back-solve is the fallback
    _trace_svg(img, tmp_path / "d.svg", dict(base))
    assert dict(tv3._TRACE_AUDIT)["flag_mm_per_px_source"] == "backsolved_fill_mm_over_fill_px"


def test_curved_core_fallback_is_counted_once_per_blob_not_once_per_level(tmp_path):
    """One thin curved stroke (a half-annulus, width 18 px < 2 x core_r) falls back to rings.
    The same blob is re-tested at every deeper level (levels counter >= 2) but is ONE blob
    (unique == 1). Positive control: a straight capsule of the same width is row-filled, counts 0."""
    import ccplotter.trace_v3 as tv3
    from ccplotter.trace_v3 import _trace_svg
    img = np.full((500, 500), 255, np.uint8)
    cv2.ellipse(img, (250, 250), (150, 150), 0, 200, 340, 0, 18)
    opts = {"fill_px": FILL_PX, "fill_mm": FILL_MM, "threshold_mode": "otsu", "pocket_link": 0,
            "_place_cell": _pc(MM), "core_fill": "hybrid"}
    _trace_svg(img, tmp_path / "c.svg", dict(opts))
    au = dict(tv3._TRACE_AUDIT)
    assert au["n_core_curved_ring_levels"] >= 2, au
    assert au["n_core_curved_unique"] == 1, au
    cap = np.full((500, 500), 255, np.uint8)
    cv2.line(cap, (60, 250), (440, 250), 0, 18)
    _trace_svg(cap, tmp_path / "s.svg", dict(opts))
    au2 = dict(tv3._TRACE_AUDIT)
    assert au2["n_core_curved_unique"] == 0 and au2["n_core_regions"] >= 1, au2


# ---------------------------------------------------------------------------
# edge_inset_mm
# ---------------------------------------------------------------------------

def test_edge_inset_places_outermost_ring_to_0p01mm_on_a_square(tmp_path):
    x0, y0, side = 100, 100, 400                         # ink pixels x in [100,500), edge at x = 99.5 (pixel centres)
    img = np.full((700, 700), 255, np.uint8)
    img[y0:y0 + side, x0:x0 + side] = 0
    true_left_edge = x0 - 0.5
    for inset_mm in (0.06, 0.10, 0.17):
        polys = _trace(img, {"edge_inset_mm": inset_mm}, tmp_path, f"in{int(inset_mm*100)}")
        rings = [p for p in polys if _is_closed(p)]
        rings.sort(key=lambda p: min(q[0] for q in p))   # outermost first
        outer = rings[0]
        xs = [q[0] for q in outer]; ys = [q[1] for q in outer]
        d_left = (min(xs) - true_left_edge) * MM
        d_right = ((x0 + side - 0.5) - max(xs)) * MM
        d_top = (min(ys) - (y0 - 0.5)) * MM
        d_bot = ((y0 + side - 0.5) - max(ys)) * MM
        for d in (d_left, d_right, d_top, d_bot):
            assert abs(d - inset_mm) < 0.01, (inset_mm, d_left, d_right, d_top, d_bot)
        # pitch unchanged: the next ring is one fill pitch further in
        second = rings[1]
        d2 = (min(q[0] for q in second) - true_left_edge) * MM
        assert abs((d2 - d_left) - FILL_MM) < 0.01, (d2 - d_left, FILL_MM)
    # NULL: the baseline's outermost ring is NOT at those insets (it is pixel-quantised at ~0.5 footprint)
    base = [p for p in _trace(img, {}, tmp_path, "base") if _is_closed(p)]
    base.sort(key=lambda p: min(q[0] for q in p))
    d_base = (min(q[0] for q in base[0]) - true_left_edge) * MM
    assert all(abs(d_base - v) > 0.0 for v in (0.06, 0.10, 0.17))
    assert abs(d_base - 0.5 * FILL_MM) < 0.03            # ~half a footprint, to a pixel


def test_edge_inset_does_not_drop_thin_components(tmp_path):
    """A stroke thinner than 2*inset must fall back to the skeleton, never vanish."""
    img = np.full((200, 400), 255, np.uint8)
    cv2.line(img, (30, 100), (370, 100), 0, 4, cv2.LINE_8)   # ~0.19mm wide
    polys = _trace(img, {"edge_inset_mm": 0.20}, tmp_path)
    assert polys and max(np.hypot(p[-1][0] - p[0][0], p[-1][1] - p[0][1]) for p in polys) > 200


# ---------------------------------------------------------------------------
# flags off == absent  (the explicit defaults change nothing)
# ---------------------------------------------------------------------------

def test_flag_defaults_are_inert_in_trace(tmp_path):
    img = _synth_discs_and_capsules()
    a = _trace(img, {}, tmp_path, "a")
    b = _trace(img, {"core_fill": "rings", "edge_inset_mm": 0.0}, tmp_path, "b")
    assert a == b


# ---------------------------------------------------------------------------
# bw_clahe / bw_cut : mask direction on the benchmark + route plumbing
# ---------------------------------------------------------------------------

SRC = REPO / "tests" / "fixtures" / "benchmark" / "sources"
SIX = {"ice": "07_ice1_railway_source.jpg", "belltower": "02_belltower_source.jpg",
       "tram": "04_hktram_source.jpg", "bike": "01_bike_source.jpg",
       "a350": "06_a350_source.jpg", "giraffe": "03_giraffe_source.png"}


def _s1_ink(name, tmp_path, clahe, clip, cut):
    from ccplotter.bw_photo import bw_xdog_raster
    from ccplotter.mask import _binarize_mask, _load_gray
    src = SRC / SIX[name]
    if not src.exists():
        pytest.skip(f"fixture image not redistributable; place {SIX[name]} at {SRC} to run")
    ras = bw_xdog_raster(src)
    p = tmp_path / f"{name}_ras.png"
    cv2.imwrite(str(p), ras)
    g = _load_gray(p, 3000, clahe, "auto", None, None, clahe_clip=clip)
    o = {"threshold_mode": "otsu"}
    if cut == "mid":
        o["bw_cut"] = "mid"
    return float(_binarize_mask(g, o).mean())


@pytest.mark.parametrize("name", sorted(SIX))
def test_bw_cut_mid_inks_less_than_otsu_on_the_benchmark(name, tmp_path):
    """Otsu lands at ~145-169 grey on these XDoG rasters (stage-loss RESULT), a fixed cut at
    128 is LOWER, so strictly less of the image is ink."""
    otsu = _s1_ink(name, tmp_path, True, 2.5, "otsu")
    mid = _s1_ink(name, tmp_path, True, 2.5, "mid")
    assert mid < otsu - 0.005, (name, otsu, mid)


@pytest.mark.parametrize("tmode", ["sensitivity", "otsu"])
def test_bw_cut_mid_applies_under_the_threshold_modes_the_route_actually_sends(tmode):
    """Regression: the A/B run found bw_cut=mid a silent no-op because /api/process sends
    threshold_mode='sensitivity' (not 'otsu'). Positive control: mid must change the mask under
    BOTH. Negative control: an explicit manual mode still wins over bw_cut."""
    from ccplotter.mask import _binarize_mask
    rng = np.random.default_rng(3)
    g = np.clip(rng.normal(150, 40, (200, 200)), 0, 255).astype(np.uint8)
    base = _binarize_mask(g, {"threshold_mode": tmode})
    mid = _binarize_mask(g, {"threshold_mode": tmode, "bw_cut": "mid"})
    assert (base != mid).any(), "bw_cut=mid changed nothing"
    man = _binarize_mask(g, {"threshold_mode": "manual", "manual_threshold": 100, "bw_cut": "mid"})
    ref = _binarize_mask(g, {"threshold_mode": "manual", "manual_threshold": 100})
    assert (man == ref).all()


@pytest.mark.parametrize("name", sorted(SIX))
def test_bw_clahe_soft_sits_between_on_and_off(name, tmp_path):
    on = _s1_ink(name, tmp_path, True, 2.5, "otsu")
    soft = _s1_ink(name, tmp_path, True, BW_CLAHE_SOFT_CLIP, "otsu")
    off = _s1_ink(name, tmp_path, False, 2.5, "otsu")
    assert abs(soft - on) > 1e-5, f"soft identical to on for {name}: clip not applied"
    lo, hi = min(on, off) - 1e-6, max(on, off) + 1e-6
    assert lo <= soft <= hi, (name, on, soft, off)
    assert abs(off - on) > 1e-4


def test_load_gray_default_clip_is_unchanged_2p5(tmp_path):
    from ccplotter.mask import _load_gray
    g = np.fromfunction(lambda y, x: 128 + 90 * np.sin(x / 9.0) * np.cos(y / 13.0), (150, 220)).astype(np.uint8)
    p = tmp_path / "g.png"
    cv2.imwrite(str(p), g)
    a = _load_gray(p, 400, True, "auto", None, None)
    b = _load_gray(p, 400, True, "auto", None, None, clahe_clip=2.5)
    c = _load_gray(p, 400, True, "auto", None, None, clahe_clip=1.25)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def _make_photo(path: Path, w: int, h: int):
    yy, xx = np.mgrid[0:h, 0:w]
    base = (128 + 100 * np.sin(xx / 11.0) * np.cos(yy / 17.0)).astype(np.uint8)
    img = np.stack([base, np.roll(base, 5, axis=0), np.roll(base, -5, axis=1)], axis=-1)
    Image.fromarray(img, mode="RGB").save(path, quality=92)


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


def _run(did, extra, w=900, h=600):
    _make_photo(WORK_DIR / f"src_{did}.jpg", w, h)
    resp = _process(did, extra)
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    routed = body["debug_files"]["routed_svg"]
    renders = sorted(p.name for p in WORK_DIR.glob(f"src_{did}_render_*.svg"))
    return body, _points_of((WORK_DIR / routed).read_text(encoding="utf-8")), renders


def test_route_bw_keys_default_equals_absent_and_cache_key_unmoved():
    _, pts_a, rn_a = _run("bwk_absent", {"style": "bwphoto"})
    _, pts_b, rn_b = _run("bwk_default", {"style": "bwphoto", "bw_clahe": "on", "bw_cut": "otsu",
                                          "core_fill": "rings", "edge_inset_mm": 0})
    assert pts_a and pts_a == pts_b
    # render cache key: hash part of the render file name is a function of render_opts only,
    # and the default keys must not enter it
    assert re.sub(r"src_[a-z_]+_render_", "", rn_a[0]) == re.sub(r"src_[a-z_]+_render_", "", rn_b[0])


def test_route_bw_keys_are_honoured_under_bwphoto_and_change_geometry():
    _, base, _ = _run("bwk_base", {"style": "bwphoto"})
    for k, v in (("bw_clahe", "soft"), ("bw_clahe", "off"), ("bw_cut", "mid")):
        _, pts, _ = _run(f"bwk_{k}_{v}", {"style": "bwphoto", k: v})
        assert pts and pts != base, f"{k}={v} did not change the geometry under bwphoto"


def test_route_bw_keys_are_ignored_outside_bwphoto():
    """A plain tracev3 request must not honour bw_* (cannot be smuggled in)."""
    _, a, ra = _run("bwk_plain_a", {"style": "tracev3"})
    _, b, rb = _run("bwk_plain_b", {"style": "tracev3", "bw_clahe": "off", "bw_cut": "mid"})
    assert a == b
    assert re.sub(r"src_[a-z_]+_render_", "", ra[0]) == re.sub(r"src_[a-z_]+_render_", "", rb[0])


def test_route_hostile_bwphoto_still_forced_to_recipe_while_bw_keys_apply():
    """The forced-recipe logic ignores everything else: hostile clahe=False/tone/res under
    bwphoto with bw_clahe=soft gives exactly the clean bw_clahe=soft request."""
    _, clean, _ = _run("bwk_hclean", {"style": "bwphoto", "bw_clahe": "soft"})
    _, host, _ = _run("bwk_hhost", {"style": "bwphoto", "bw_clahe": "soft", "clahe": False, "tone": "balanced",
                                    "resolution": 300, "native_res": True})
    assert clean and clean == host
    assert BW_PHOTO_RECIPE == {"clahe": True, "tone": "off", "resolution": 3000, "native_res": False}


def test_route_hybrid_and_inset_reach_the_trace_and_core_mask_listed():
    body, pts, _ = _run("hyb_route", {"style": "tracev3", "core_fill": "hybrid", "edge_inset_mm": 0.08,
                                      "scale_aware": True, "resolution": 1500})
    assert body["debug_files"].get("core_mask") == "debug_core_mask.png", body["debug_files"]
    base_body, base_pts, _ = _run("hyb_route_b", {"style": "tracev3", "scale_aware": True, "resolution": 1500})
    assert "core_mask" not in base_body["debug_files"]
    assert pts and pts != base_pts


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
