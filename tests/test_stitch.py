"""Tests for the opt-in `stitch_mm` routing flag (ccplotter/stitch.py).

Layout of the evidence (could-not-fail doctrine: every null has a positive control that can
fail, and the mask check is never judged by the dt predicate it implements):

  * unit: the predicate against an INDEPENDENT oracle (cv2.erode of the mask by a disc, a
    different algorithm from the distance transform), a positive control (gap-free pair
    joins) next to its negative control (the same pair across a white gap does not).
  * unit: acyclic, zero-length, closed-ring, hairpin-preference and determinism properties.
  * integration through the real _trace_svg on a fragmented synthetic: lifts drop, every
    connector lies on black pixels of the SOURCE image, key absent == key 0 byte-for-byte.
  * route: key parsing, out-of-range ignored, cache key untouched when absent.
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
_test_home.use_temp_home("ccplotter_test_home_stitch_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_stitch_"))
sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import pytest
from PIL import Image

from ccplotter.config import PX2MM, STITCH_MM_MAX, WORK_DIR
from ccplotter.stitch import stitch_open_strokes

MM = 0.0474                      # mm per working px used by these synthetics
FILL_MM = 0.12
FILL_PX = FILL_MM / MM
ERODE = 0.5 * FILL_PX


def _pc(mm_per_px):
    return [1.0, 1.0, 0.0, mm_per_px / PX2MM, True]


def _dt(mask):
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)


def _bar_mask(h=60, w=300, y0=20, y1=40):
    m = np.zeros((h, w), bool)
    m[y0:y1, :] = True
    return m


def _hseg(x0, x1, y):
    return [(float(x0), float(y)), (float(x1), float(y))]


def _oracle_inside(mask, A, B, r):
    """Independent check: the segment A-B, sampled at 0.25 px, lies on pixels that survive a
    cv2.erode of the mask by a disc of radius r (nearest-pixel lookup)."""
    k = int(np.floor(r))
    er = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
    n = max(2, int(np.ceil(np.hypot(B[0] - A[0], B[1] - A[1]) / 0.25)))
    for t in np.linspace(0, 1, n):
        x = int(round(A[0] + (B[0] - A[0]) * t))
        y = int(round(A[1] + (B[1] - A[1]) * t))
        if not er[y, x]:
            return False
    return True


# ---------------------------------------------------------------------------
# unit: predicate, positive and negative control
# ---------------------------------------------------------------------------

def test_gap_free_pair_is_joined_and_connector_passes_the_independent_oracle():
    m = _bar_mask()
    dt = _dt(m)
    a, b = _hseg(10, 100, 30), _hseg(103, 200, 30)
    new, mem, conns, rec = stitch_open_strokes([a, b], dt, stitch_px=6.0, erode_px=ERODE, tangent_px=2.5)
    assert len(new) == 1 and mem == [[0, 1]] and len(conns) == 1
    x0, y0, x1, y1 = conns[0][:4]
    assert _oracle_inside(m, (x0, y0), (x1, y1), ERODE - 0.5)
    assert new[0][0] == a[0] and new[0][-1] == b[-1]
    assert rec["n_joins"] == 1 and rec["rejected_offmask"] == 0


def test_same_pair_across_a_white_gap_is_NOT_joined_and_counted_off_mask():
    """Negative control paired with the positive one above: identical geometry but a 4 px white
    gap cut through the bar between the two ends."""
    m = _bar_mask()
    gap = m.copy()
    gap[:, 100:105] = False
    a, b = _hseg(10, 99, 30), _hseg(106, 200, 30)
    new_ok, _, c_ok, _ = stitch_open_strokes([a, b], _dt(m), 8.0, ERODE, 2.5)
    new_gap, _, c_gap, rec = stitch_open_strokes([a, b], _dt(gap), 8.0, ERODE, 2.5)
    assert len(new_ok) == 1 and len(c_ok) == 1            # the control can fail
    assert len(new_gap) == 2 and c_gap == []
    assert rec["rejected_offmask"] == 1 and rec["n_joins"] == 0
    # and the oracle agrees the straight line is NOT inside the eroded gapped mask
    assert not _oracle_inside(gap, (99, 30), (106, 30), ERODE - 0.5)


def test_connector_hugging_the_edge_is_rejected_but_one_pen_inside_passes():
    m = _bar_mask(y0=20, y1=40)
    dt = _dt(m)
    # a run along the very first row of ink (dt = 1 < erode) must not be bridged ...
    edge_a, edge_b = _hseg(10, 100, 20), _hseg(104, 200, 20)
    # ... while the same run through the middle is
    mid_a, mid_b = _hseg(10, 100, 30), _hseg(104, 200, 30)
    _, _, c_edge, r_edge = stitch_open_strokes([edge_a, edge_b], dt, 6.0, ERODE, 2.5)
    _, _, c_mid, _ = stitch_open_strokes([mid_a, mid_b], dt, 6.0, ERODE, 2.5)
    assert c_edge == [] and r_edge["rejected_offmask"] == 1 and r_edge["rejected_offmask_thin_endpoint"] == 1
    assert len(c_mid) == 1


def test_candidates_beyond_stitch_distance_are_ignored():
    m = _bar_mask()
    a, b = _hseg(10, 100, 30), _hseg(120, 200, 30)
    new, _, conns, rec = stitch_open_strokes([a, b], _dt(m), 6.0, ERODE, 2.5)
    assert len(new) == 2 and conns == [] and rec["n_candidate_pairs"] == 0
    new, _, conns, _ = stitch_open_strokes([a, b], _dt(m), 25.0, ERODE, 2.5)
    assert len(new) == 1 and len(conns) == 1


# ---------------------------------------------------------------------------
# unit: structural properties
# ---------------------------------------------------------------------------

def test_no_loop_is_closed_and_a_stroke_is_never_joined_to_itself():
    m = _bar_mask(h=80, w=200, y0=10, y1=70)
    # a "C" made of two strokes whose two ends nearly meet on both sides -> only ONE join
    a = [(50.0, 20.0), (50.0, 40.0)]
    b = [(52.0, 40.0), (52.0, 20.0)]
    new, mem, conns, rec = stitch_open_strokes([a, b], _dt(m), 5.0, ERODE, 2.5)
    assert len(new) == 1 and rec["n_joins"] == 1 and rec["rejected_cycle"] == 1
    # a single open stroke whose own ends are within reach is left alone
    s = [(50.0, 20.0), (50.0, 60.0), (52.0, 20.0)]
    new, _, conns, rec = stitch_open_strokes([s], _dt(m), 5.0, ERODE, 2.5)
    assert new == [s] and conns == []


def test_closed_rings_are_untouched_and_keep_their_identity():
    m = _bar_mask(h=100, w=200, y0=5, y1=95)
    ring = [(50.0, 30.0), (90.0, 30.0), (90.0, 60.0), (50.0, 60.0), (50.0, 30.0)]
    a, b = _hseg(120, 140, 50), _hseg(142, 170, 50)
    new, mem, _, _ = stitch_open_strokes([ring, a, b], _dt(m), 5.0, ERODE, 2.5)
    assert new[0] is ring and mem[0] == [0] and len(new) == 2 and mem[1] == [1, 2]


def test_coincident_endpoints_join_with_no_connector_and_no_ink():
    m = _bar_mask()
    a, b = _hseg(10, 100, 30), _hseg(100.0, 200, 30)
    new, _, conns, rec = stitch_open_strokes([a, b], _dt(m), 4.0, ERODE, 2.5)
    assert len(new) == 1 and conns == [] and rec["n_joins_zero_len"] == 1 and rec["connector_len_px"] == 0.0
    assert new[0] == [(10.0, 30.0), (100.0, 30.0), (200.0, 30.0)]


def test_non_hairpin_partner_is_preferred_over_a_closer_hairpin():
    """The running stroke's end has two partners in reach: a CLOSER stroke heading back along
    it (hairpin, 2.2 px) and a farther one that continues the line (5 px). The greedy takes the
    continuation and reports no hairpin."""
    m = np.ones((120, 300), bool)
    dt = _dt(m)
    run = [(20.0, 60.0), (100.0, 60.0)]
    hair = [(98.0, 61.0), (30.0, 61.0)]           # 2.2 px from the run's end, heading straight back
    cont = [(105.0, 60.0), (200.0, 60.0)]         # 5 px away, continues the line; 7 px from hair
    new, mem, conns, rec = stitch_open_strokes([run, hair, cont], dt, 6.0, ERODE, 2.5)
    chains = {tuple(sorted(x)) for x in mem}
    assert (0, 2) in chains, mem
    assert rec["hairpin_joins"] == 0


def test_hairpin_join_is_taken_when_it_is_the_only_option_and_is_counted():
    m = np.ones((120, 300), bool)
    a = [(20.0, 60.0), (100.0, 60.0)]
    b = [(103.0, 62.0), (30.0, 62.0)]
    new, mem, conns, rec = stitch_open_strokes([a, b], _dt(m), 6.0, ERODE, 2.5)
    assert len(new) == 1 and rec["hairpin_joins"] == 1 and conns[0][5] > 135.0


def test_deterministic():
    rng = np.random.default_rng(3)
    m = np.ones((200, 400), bool)
    pieces = []
    for _ in range(80):
        x, y = rng.uniform(20, 380), rng.uniform(20, 180)
        ang = rng.uniform(0, 2 * np.pi)
        pieces.append([(x, y), (x + 15 * np.cos(ang), y + 15 * np.sin(ang))])
    r1 = stitch_open_strokes(pieces, _dt(m), 12.0, ERODE, 2.5)
    r2 = stitch_open_strokes(pieces, _dt(m), 12.0, ERODE, 2.5)
    assert r1[0] == r2[0] and r1[1] == r2[1] and r1[2] == r2[2]
    # every original piece appears in exactly one chain (nothing lost, nothing duplicated)
    flat = sorted(i for mem in r1[1] for i in mem)
    assert flat == list(range(len(pieces)))


def test_random_connectors_all_pass_the_independent_oracle():
    """Property test: scatter short fragments over an irregular mask, then re-verify EVERY
    emitted connector with cv2.erode (not dt)."""
    rng = np.random.default_rng(11)
    m = np.zeros((300, 500), bool)
    for _ in range(9):
        cv2.circle(m.view(np.uint8), (int(rng.integers(50, 450)), int(rng.integers(50, 250))),
                   int(rng.integers(20, 60)), 1, -1)
    m = m.astype(bool)
    dt = _dt(m)
    ys, xs = np.nonzero(dt > ERODE)
    pieces = []
    for _ in range(300):
        i = int(rng.integers(len(xs)))
        x, y = float(xs[i]), float(ys[i])
        ang = rng.uniform(0, 2 * np.pi)
        pieces.append([(x, y), (x + 4 * np.cos(ang), y + 4 * np.sin(ang))])
    new, mem, conns, rec = stitch_open_strokes(pieces, dt, 15.0, ERODE, 2.5)
    assert len(conns) > 10, "control: the scatter must produce joins or this test proves nothing"
    assert rec["rejected_offmask"] > 0, "control: some candidates must have crossed outside"
    bad = [c for c in conns if not _oracle_inside(m, c[:2], c[2:4], ERODE - 1.0)]
    assert bad == []


# ---------------------------------------------------------------------------
# integration through the real _trace_svg
# ---------------------------------------------------------------------------

def _trace(img, extra, tmp_path, name="t", debug=False):
    from ccplotter.trace_v3 import _trace_svg
    out = tmp_path / f"{name}.svg"
    opts = {"fill_px": FILL_PX, "fill_mm": FILL_MM, "threshold_mode": "otsu", "pocket_link": 0,
            "_place_cell": _pc(MM), **extra}
    prefix = str(tmp_path / f"{name}_dbg") if debug else None
    _trace_svg(img, out, opts, debug_prefix=prefix)
    text = out.read_text(encoding="utf-8")
    polys = [[tuple(float(v) for v in t.split(",")) for t in ps.split()]
             for ps in re.findall(r'points="([^"]+)"', text)]
    return polys, text


def _fragmented():
    """Rows of 1 mm x 0.6 mm bars separated by 0.25 mm white gaps (positive: lifts to remove
    inside each bar's notch-free body is nil, so the gap is what must NOT be crossed) plus
    notched 3x-tall bars whose rings/rows fragment around the notch."""
    img = np.full((600, 1400), 255, np.uint8)
    t = int(round(0.6 / MM))
    x = 40
    for _ in range(12):
        L = int(round(1.0 / MM))
        cv2.rectangle(img, (x, 100), (x + L, 100 + t), 0, -1)
        x += L + int(round(0.25 / MM))
    x = 40
    for _ in range(6):
        L = int(round(1.0 / MM))
        cv2.rectangle(img, (x, 250), (x + L, 250 + 3 * t), 0, -1)
        cv2.rectangle(img, (x + L // 2 - 1, 250), (x + L // 2 + 1, 250 + 3 * t // 2), 255, -1)
        x += L + int(round(0.2 / MM))
    return img


def test_key_absent_zero_and_audit_are_byte_identical(tmp_path):
    img = _fragmented()
    a, ta = _trace(img, {"core_fill": "hybrid"}, tmp_path, "a")
    b, tb = _trace(img, {"core_fill": "hybrid", "stitch_mm": 0.0}, tmp_path, "b")
    re_date = re.compile(r"<dc:date>[^<]*</dc:date>")
    assert a == b and re_date.sub("", ta) == re_date.sub("", tb)
    from ccplotter import trace_v3
    assert "stitch" not in trace_v3._TRACE_AUDIT


def test_stitch_drops_lifts_and_every_connector_lies_on_source_black(tmp_path):
    img = _fragmented()
    base, _ = _trace(img, {"core_fill": "hybrid"}, tmp_path, "base")
    st, _ = _trace(img, {"core_fill": "hybrid", "stitch_mm": 1.0}, tmp_path, "st", debug=True)
    import json
    from ccplotter import trace_v3
    rec = trace_v3._TRACE_AUDIT["stitch"]
    assert len(st) < len(base) and rec["lifts_after"] == len(st) and rec["lifts_before"] == len(base)
    assert rec["connectors_added"] > 0 and rec["connector_len_mm"] > 0
    dbg = json.load(open(tmp_path / "st_dbg_stitch.json"))
    assert len(dbg["connectors_px"]) == rec["connectors_added"]
    black = img < 128
    for x0, y0, x1, y1 in [c[:4] for c in dbg["connectors_px"]]:
        for t in np.linspace(0, 1, 40):
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            assert black[int(round(y)), int(round(x))], f"connector sample ({x:.1f},{y:.1f}) is on white"
    # the connector audit is an independent raster measure and is reported beside its baseline
    assert {"connector_offmask_pct", "original_strokes_offmask_pct",
            "original_edge_segments_offmask_pct"} <= set(rec["audit"])


def test_stitch_never_joins_fragments_across_the_white_gaps(tmp_path):
    """Negative control at integration level: the 12 plain bars sit 0.25 mm apart (well within
    stitch_mm=1.0). No connector may bridge from one bar to the next."""
    img = _fragmented()
    _, _ = _trace(img, {"core_fill": "hybrid", "stitch_mm": 1.0}, tmp_path, "gap", debug=True)
    import json
    dbg = json.load(open(tmp_path / "gap_dbg_stitch.json"))
    t = int(round(0.6 / MM))
    for x0, y0, x1, y1 in [c[:4] for c in dbg["connectors_px"]]:
        if 95 <= y0 <= 100 + t + 5 and 95 <= y1 <= 100 + t + 5:       # the plain-bar row
            # both ends must be in the SAME bar: no white between them along the connector
            for tt in np.linspace(0, 1, 60):
                x, y = x0 + (x1 - x0) * tt, y0 + (y1 - y0) * tt
                assert img[int(round(y)), int(round(x))] < 128
    assert dbg["rejected_offmask"] > 0, "control: the bars' gaps must have produced rejections"


def test_stitch_skips_with_a_stated_reason_under_tone_or_zigzag(tmp_path):
    img = _fragmented()
    from ccplotter import trace_v3
    base, _ = _trace(img, {"fill_mode": "zigzag"}, tmp_path, "zz0")
    z, _ = _trace(img, {"fill_mode": "zigzag", "stitch_mm": 1.0}, tmp_path, "zz1")
    assert z == base
    assert trace_v3._TRACE_AUDIT["stitch"]["skipped"] == "zigzag"


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------

def _make_photo(path: Path, w: int, h: int):
    yy, xx = np.mgrid[0:h, 0:w]
    base = (128 + 100 * np.sin(xx / 11.0) * np.cos(yy / 17.0)).astype(np.uint8)
    img = np.stack([base, np.roll(base, 5, axis=0), np.roll(base, -5, axis=1)], axis=-1)
    Image.fromarray(img, mode="RGB").save(path, quality=92)


def _run(did, extra, w=900, h=600):
    from ccplotter.routes.core import app
    _make_photo(WORK_DIR / f"src_{did}.jpg", w, h)
    resp = app.test_client().post("/api/process", json={
        "job_type": "preview", "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}],
        "paper": "120x120", "margin": 10.0, "debug": True, **extra})
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    text = (WORK_DIR / body["debug_files"]["routed_svg"]).read_text(encoding="utf-8")
    pts = sorted(tuple(t.split(",")) for ps in re.findall(r'points="([^"]+)"', text) for t in ps.split() if "," in t)
    renders = sorted(p.name for p in WORK_DIR.glob(f"src_{did}_render_*.svg"))
    return body, pts, renders


def test_route_absent_zero_and_out_of_range_equal_and_cache_key_unmoved():
    base = {"style": "tracev3", "scale_aware": True, "resolution": 1500}
    _, a, ra = _run("stk_absent", base)
    for i, v in enumerate((0, "0", None, -1, STITCH_MM_MAX + 1, "junk")):
        _, b, rb = _run(f"stk_inert{i}", {**base, "stitch_mm": v})
        assert a and a == b, v
        assert re.sub(r"src_[a-z_0-9]+_render_", "", ra[0]) == re.sub(r"src_[a-z_0-9]+_render_", "", rb[0]), v


def test_route_stitch_reaches_the_trace_and_lists_the_record():
    base = {"style": "tracev3", "scale_aware": True, "resolution": 1500, "core_fill": "hybrid"}
    body0, pts0, r0 = _run("stk_r0", base)
    body1, pts1, r1 = _run("stk_r1", {**base, "stitch_mm": 1.0})
    assert "stitch" not in body0["debug_files"]
    assert body1["debug_files"].get("stitch") == "debug_stitch.json", body1["debug_files"]
    assert pts1 and pts1 != pts0
    assert re.sub(r"src_[a-z_0-9]+_render_", "", r0[0]) != re.sub(r"src_[a-z_0-9]+_render_", "", r1[0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
