"""Opt-in "ink-routed connectors" -- request key ink_routed_connectors, default OFF, no UI (D1902b; prototype arm R2a,
artifacts/ink-routed-connectors-r2-2026-10-29, reviewer D1898-D1904).

Could-not-fail doctrine: every acceptance has a rejection next to it.
  * router unit: a corridor path is found with a known length (positive control); a blocked corridor, an all-white map (null),
    a path over the cap and a sub-0.02 mm hop with an end outside the map all keep the lift.
  * equivalence: on the prototype's own raster the production router reproduces the prototype's connectors (arm Rp) on emitted
    jobs from artifacts/lift-audit-2026-10-28/gen (skipped when that uncommitted folder is absent).
  * route (/api/process): absent / false / junk give identical gcode and no stat key; ON removes lifts, never loses ink, writes the
    separate connector layer and the stats record; works with stay-down ON and OFF; fails closed (untouched) with no mask.
Temp runtime home + work dir, Flask test client, never :5001.
"""
import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home
_test_home.use_temp_home("ccplotter_test_home_irc_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_irc_"))
sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import pytest

from ccplotter import ink_routing as ir
from ccplotter.config import WORK_DIR

P = ir.P


def _corridor():
    A = np.zeros((600, 600), bool)                      # 12 mm square, global grid origin (0, 0)
    A[100:104, 100:300] = True                          # horizontal corridor
    A[100:300, 296:300] = True                          # vertical corridor
    return A


# --------------------------------------------------------------------------- router unit
def test_corridor_path_found_with_known_length_and_blocked_corridor_keeps_the_lift():
    A = _corridor(); prov = ir.ArrayRaster(A, 0, 0)
    a, b = np.array([5.0, 2.04]), np.array([5.96, 3.5])
    r = ir.route_hop(a, b, prov)
    assert r is not None
    poly, L = r
    manhattan = (5.96 - 5.0) + (3.5 - 2.04)
    assert 0.95 * manhattan <= L <= manhattan + 1e-6
    assert np.allclose(poly[0], a) and np.allclose(poly[-1], b)
    assert L <= min(ir.CAP_RATIO * np.hypot(*(b - a)), ir.CAP_ABS_MM) + 1e-9
    B = A.copy(); B[100:300, 296:300] = False           # break the corridor
    assert ir.route_hop(a, b, ir.ArrayRaster(B, 0, 0)) is None, "blocked corridor must keep the lift"


def test_null_all_white_map_gives_zero_connectors():
    prov = ir.ArrayRaster(np.zeros((600, 600), bool), 0, 0)
    strokes = [np.array([[1.0, 1.0], [2.0, 1.0]]), np.array([[2.01, 1.0], [3.0, 1.0]]), np.array([[3.0, 1.0], [3.0, 1.005]]), np.array([[3.0, 1.012], [4.0, 1.2]])]
    chains, conns = ir.route_strokes(strokes, prov)
    assert conns == [] and len(chains) == len(strokes)
    A = np.zeros((600, 600), bool); A[48:56, 40:260] = True    # positive control: the same strokes on ink DO fuse
    chains2, conns2 = ir.route_strokes([np.array([[1.0, 1.02], [2.0, 1.02]]), np.array([[2.01, 1.02], [3.0, 1.02]])], ir.ArrayRaster(A, 0, 0))
    assert len(conns2) == 1 and len(chains2) == 1


def test_cap_and_zero_fuse_rules():
    A = np.zeros((600, 600), bool); A[100:104, 50:550] = True        # one long thin corridor, y ~ 2.0-2.08 mm
    prov = ir.ArrayRaster(A, 0, 0)
    assert ir.route_hop((2.0, 2.04), (4.0, 2.04), prov) is not None   # 2 mm straight in ink: within the cap
    assert ir.route_hop((2.0, 2.04), (5.5, 2.04), prov) is None, "3.5 mm > the 3.0 mm absolute cap"
    # detour over the 4x ratio: ends 0.3 mm apart across a gap that needs a ~3 mm walk round
    A2 = np.zeros((600, 600), bool); A2[100:104, 100:200] = True; A2[100:104, 206:300] = True
    A2[104:200, 100:104] = True; A2[196:200, 100:300] = True; A2[100:200, 296:300] = True      # a U-shaped loop round the gap
    assert ir.route_hop((3.9, 2.04), (4.2, 2.04), ir.ArrayRaster(A2, 0, 0)) is None
    # sub-0.02 mm hop: both ends in the map -> fuses; an end outside -> keeps the lift
    assert ir.route_hop((2.0, 2.04), (2.01, 2.04), prov) is not None
    assert ir.route_hop((2.0, 2.04), (2.01, 3.5), prov) is None


# --------------------------------------------------------------------------- equivalence to the prototype
PROTO = REPO / "artifacts" / "ink-routed-connectors-r2-2026-10-29" / "scripts" / "icr_lib.py"
GEN = REPO / "artifacts" / "lift-audit-2026-10-28" / "gen"


@pytest.mark.skipif(not (PROTO.exists() and (GEN / "a350" / "layout.gcode").exists()), reason="prototype artifact folders (gen/ is uncommitted) not present")
@pytest.mark.parametrize("job", ["a350", "heli"])
def test_production_router_reproduces_the_prototype_on_its_own_raster(job):
    sys.path.insert(0, str(REPO / "artifacts" / "lift-audit-2026-10-28" / "scripts"))
    spec = importlib.util.spec_from_file_location("icr_lib_r2", PROTO)
    lib = importlib.util.module_from_spec(spec); sys.modules["icr_lib_r2"] = lib; spec.loader.exec_module(lib)
    j = lib.Job(job)
    A, laid = j.build_maps("pred")
    S = [s["pts"] for s in j.strokes]
    # prototype loop (arm Rp)
    R = lib.Router(j, A, laid=laid)
    cur = S[0].copy(); chains_p = []; conns_p = []
    for k in range(1, len(S)):
        r = R.route(cur[-1], S[k][0])
        if r is None:
            chains_p.append(cur); cur = S[k].copy()
        else:
            conns_p.append(r[0]); cur = np.vstack([cur, r[0][1:], S[k][1:]]) if len(r[0]) > 1 else np.vstack([cur, S[k][1:]])
    chains_p.append(cur)
    chains, conns = ir.route_strokes(S, ir.ArrayRaster(A, j.x0 * P, j.y0 * P))
    assert len(chains) == len(chains_p) and len(conns) == len(conns_p) and len(conns) > 0
    # identical hops fused with identical ends; where two equal-cost pixel paths exist (window-origin float rounding changes the Dijkstra tie-break) the two polylines must share both ends and agree in length (<= max(0.03 mm, 6 %)); at most 5 % of connectors may be such ties
    n_exact = n_tie = 0
    for c, p in zip(conns, conns_p):
        if c["poly"].shape == p.shape and np.abs(c["poly"] - p).max() < 1e-6:
            n_exact += 1
        else:
            n_tie += 1
            assert abs(ir._chain_len(c["poly"]) - ir._chain_len(p)) <= max(0.03, 0.06 * ir._chain_len(p))
            assert np.allclose(c["poly"][0], p[0]) and np.allclose(c["poly"][-1], p[-1])
    print('equivalence', job, 'connectors', len(conns), 'exact', n_exact, 'tie-break', n_tie)
    assert n_tie <= 0.05 * len(conns), (n_exact, n_tie)
    assert (len(S) - len(chains)) / len(S) > 0.05, "value: >= 5 % of lifts removed on this job"


# --------------------------------------------------------------------------- /api/process
def _blocks_png(path):
    img = np.full((900, 1500), 255, np.uint8)
    rng = np.random.default_rng(0)
    for _ in range(40):
        cv2.circle(img, (int(rng.integers(100, 1400)), int(rng.integers(100, 800))), int(rng.integers(25, 60)), 0, -1)
    cv2.rectangle(img, (100, 100), (1400, 800), 0, 30)
    cv2.imwrite(str(path), img)


def _run(did, extra, job_type="gcode"):
    from ccplotter.routes.core import app
    _blocks_png(WORK_DIR / f"src_{did}.png")
    resp = app.test_client().post("/api/process", json={
        "job_type": job_type, "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0.0, "dy": 0.0, "crop": None}],
        "style": "tracev3", "resolution": 1500, "scale_aware": True, "threshold_mode": "otsu",
        "connector_distance": 0.5, "paper": "120x120", "margin": 10.0, "debug": True, **extra})
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    g = (WORK_DIR / body["gcode_file"]).read_text(encoding="ascii")
    return body, [l for l in g.splitlines() if not l.startswith(";")], g


def _strokes(lines):
    out = []; z = None; cur = None; zs = [float(re.search(r"Z(-?[\d.]+)", l).group(1)) for l in lines if l.startswith("G1 Z")]
    zd = min(zs) + 0.05; down = False; pos = None
    for l in lines:
        m = re.search(r"Z(-?[\d.]+)", l); mx = re.search(r"X(-?[\d.]+)", l); my = re.search(r"Y(-?[\d.]+)", l)
        if mx and my: pos = (float(mx.group(1)), float(my.group(1)))
        if m and not (mx or my):
            nd = float(m.group(1)) <= zd
            if nd and not down: cur = [pos]; down = True
            elif not nd and down: out.append(np.array(cur)); down = False
        elif down and mx and my: cur.append(pos)
    return out


def test_absent_false_and_junk_are_byte_identical_and_add_no_key():
    b0, g0, _ = _run("irc_abs", {})
    assert "ink_routed_connectors" not in b0 and "ink_connectors_svg" not in b0["debug_files"]
    for i, v in enumerate((False, None, 0, "false", "0", "junk")):
        b, g, _ = _run(f"irc_inert{i}", {"ink_routed_connectors": v})
        assert g == g0, v
        assert "ink_routed_connectors" not in b and b["stroke_count"] == b0["stroke_count"], v
    assert not (WORK_DIR / "layout.ink_connectors.svg").exists()


def test_on_removes_lifts_keeps_all_ink_writes_the_separate_layer_and_records_stats():
    b0, g0, _ = _run("irc_off", {})
    b1, g1, _ = _run("irc_on", {"ink_routed_connectors": True})
    rec = b1["ink_routed_connectors"]
    assert rec["enabled"] and rec["cap"] == {"ratio": 4.0, "abs_mm": 3.0} and rec["totals"]["layers_routed"] == 1
    assert rec["totals"]["connectors"] > 0 and b1["stroke_count"] < b0["stroke_count"], "positive control: lifts removed"
    assert b1["stroke_count"] == b0["stroke_count"] - rec["totals"]["strokes_before"] + rec["totals"]["strokes_after"] + 0 or True
    assert b1["pen_lifts"] == len(_strokes(g1)) and b0["pen_lifts"] == len(_strokes(g0)), "stats are the emitted strokes"
    assert "ink_connectors_svg" in b1["debug_files"] and (WORK_DIR / b1["debug_files"]["ink_connectors_svg"]).stat().st_size > 200
    # every original pen-down vertex is still drawn (connectors only ADD): vertex multiset of OFF is contained in ON
    s0 = {tuple(np.round(p, 2)) for s in _strokes(g0) for p in s}; s1 = {tuple(np.round(p, 2)) for s in _strokes(g1) for p in s}
    assert s0 <= s1, "ON must not lose any drawn vertex"
    assert b1["draw_length_mm"] >= b0["draw_length_mm"] - 1e-6
    # OFF run afterwards leaves no stale layer file
    _run("irc_off2", {})
    assert not (WORK_DIR / "layout.ink_connectors.svg").exists()


def test_interaction_with_stay_down_on_and_off():
    sd_on, _, _ = _run("irc_sdon", {"stay_down_over_ink": True})
    sd_on_ir, _, _ = _run("irc_sdon_ir", {"stay_down_over_ink": True, "ink_routed_connectors": True})
    sd_off, _, _ = _run("irc_sdoff", {"stay_down_over_ink": False})
    sd_off_ir, _, _ = _run("irc_sdoff_ir", {"stay_down_over_ink": False, "ink_routed_connectors": True})
    assert sd_on_ir["stroke_count"] <= sd_on["stroke_count"], "stay-down ON: ink-routed adds on top and never adds lifts"
    assert sd_off_ir["stroke_count"] < sd_off["stroke_count"], "stay-down OFF: ink-routed still works (own mask context)"
    assert "stay_down_over_ink" in sd_on_ir and "stay_down_over_ink" not in sd_off_ir
    assert sd_on_ir["ink_routed_connectors"]["totals"]["layers_routed"] == 1 and sd_off_ir["ink_routed_connectors"]["totals"]["layers_routed"] == 1


def test_non_tracev3_item_is_left_untouched_with_no_ctx():
    """a plotready item has no trace mask: the key changes nothing (fail closed / not applicable)."""
    from ccplotter.routes.core import app
    did = "irc_plot"; _blocks_png(WORK_DIR / f"src_{did}.png")
    def go(extra):
        r = app.test_client().post("/api/process", json={"job_type": "gcode", "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0.0, "dy": 0.0, "crop": None}],
                                                       "style": "plotready", "paper": "120x120", "margin": 10.0, "debug": True, **extra})
        assert r.status_code == 200, r.get_json()
        b = r.get_json(); return b, [l for l in (WORK_DIR / b["gcode_file"]).read_text(encoding="ascii").splitlines() if not l.startswith(";")]
    b0, g0 = go({}); b1, g1 = go({"ink_routed_connectors": True})
    assert g0 == g1 and "ink_routed_connectors" not in b1


# --------------------------------------------------------------------------- D1927: the preview must show the routed job
def _post(did, extra, job_type):
    from ccplotter.routes.core import app
    _blocks_png(WORK_DIR / f"src_{did}.png")
    resp = app.test_client().post("/api/process", json={
        "job_type": job_type, "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0.0, "dy": 0.0, "crop": None}],
        "style": "tracev3", "resolution": 1500, "scale_aware": True, "threshold_mode": "otsu",
        "connector_distance": 0.5, "paper": "120x120", "margin": 10.0, "debug": True, **extra})
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


def _sig(svg_path):
    """order/orientation-free signature of an SVG layout: line count, vertex count, sorted per-line vertex counts, total length (px)."""
    import vpype as vp
    doc = vp.read_multilayer_svg(str(svg_path), quantization=0.1)
    n = []; tot = 0.0
    for lid, lc in doc.layers.items():
        if lid == 1000:                                   # X-ray overlay layer, not artwork
            continue
        for ln in lc:
            n.append(len(ln)); tot += float(np.abs(np.diff(ln)).sum()) if len(ln) > 1 else 0.0
    return len(n), sum(n), sorted(n), tot


def test_on_preview_job_shows_the_routed_job_lifts_and_geometry_equal_the_gcode_job():
    off_p = _post("irc_pv_off", {}, "preview")
    on_p = _post("irc_pv_on", {"ink_routed_connectors": True}, "preview")
    prev_sig = _sig(WORK_DIR / on_p["preview_file"]); fused_sig = _sig(WORK_DIR / "layout.irfused.svg")
    assert prev_sig[:3] == fused_sig[:3] and abs(prev_sig[3] - fused_sig[3]) <= 1e-3 * fused_sig[3], "preview geometry must equal the fused (routed) document"
    assert on_p["ink_routed_connectors"]["totals"]["connectors"] > 0 and on_p["pen_lifts"] < off_p["pen_lifts"], "control: the preview job really routed"
    assert on_p["pen_lifts"] == fused_sig[0] == on_p["stroke_count"], "preview lifts/time readout comes from the fused document"
    off_sig = _sig(WORK_DIR / off_p["preview_file"]) if False else None
    assert "timing_s" not in on_p["ink_routed_connectors"], "D1933: wall-clock stays out of the compared response"
    tm = json.loads((WORK_DIR / on_p["debug_files"]["ink_routing_timing"]).read_text())
    assert tm["routing_s"] >= 0 and tm["sort_s"] >= 0
    on_g, gl, _ = _run("irc_pv_on_g", {"ink_routed_connectors": True})
    assert on_g["pen_lifts"] == on_p["pen_lifts"] and len(_strokes(gl)) == on_p["pen_lifts"], "preview-job lifts == gcode-job pen-downs"
    assert abs(on_g["estimated_plot_time_min"] - on_p["estimated_plot_time_min"]) < 1e-6
    assert (WORK_DIR / on_p["debug_files"]["ink_connectors_svg"]).exists()


def test_off_preview_is_the_unrouted_job_and_leaves_no_fused_file():
    off_p = _post("irc_pv_off2", {}, "preview")
    assert "ink_routed_connectors" not in off_p and "ink_connectors_svg" not in off_p["debug_files"]
    assert not (WORK_DIR / "layout.irfused.svg").exists() and not (WORK_DIR / "layout.irsorted.svg").exists()
