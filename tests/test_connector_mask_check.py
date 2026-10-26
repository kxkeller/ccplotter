"""Tests for the opt-in `connector_mask_check` Stage E flag (ccplotter/stage_e_mask.py).

Could-not-fail doctrine: every acceptance has a rejection next to it and every "off" has a
positive control proving the router WOULD have fused.

  * unit: a fuse across a white gap is rejected (the unchecked router fuses the same pair -- the
    control); a fuse inside ink is accepted; a connector straddling a thin white notch is rejected;
    zero-length joins are exempt; the target backfill skips rejected gaps and counts them.
  * transform: the page->trace mapping is checked against REAL vpype scale/rotate/translate (the
    commands composite issues) for several rotations, with a wrong-transform negative control.
  * off-state: connector_check=None is the original router; checked-but-nothing-to-reject equals it
    byte for byte; the route key absent / false / junk gives the same routed geometry and the same
    trace-cache key.
  * integration through the real route: the record is listed, counts are consistent, off-mask mm
    falls, rejected connectors are listed under debug.
"""
import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home; pen/calibration never resolve from CWD
_test_home.use_temp_home("ccplotter_test_home_cmc_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_cmc_"))
sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import pytest

from ccplotter.config import PX2MM, WORK_DIR
from ccplotter.stage_e_mask import LayerMaskCheck, NoMaskCheck, build_layer_check
from ccplotter.stats_warnings import _route_low_lift

FILL_PX = 1.2                      # thr = 0.6 px


def _c(*pts):
    return np.array([complex(x, y) for x, y in pts])


def _identity_check(mask, fill_px=FILL_PX):
    return LayerMaskCheck(mask, fill_px, (0.0, 0.0), 1.0, 0.0, (0.0, 0.0))


def _bar(h=40, w=200, y0=10, y1=30):
    m = np.zeros((h, w), bool)
    m[y0:y1, :] = True
    return m


def _two_strokes(x_end_a, x_start_b, y=20.0):
    return [_c((10, y), (x_end_a, y)), _c((x_start_b, y), (60, y))]


# --------------------------------------------------------------------------- unit

def test_fuse_across_a_white_gap_is_rejected_and_the_unchecked_router_fused_it():
    mask = _bar()
    mask[:, 30:33] = False                                    # 3 px white gap (0.8 mm) across the bar
    lines = _two_strokes(29.0, 33.0)                          # endpoints inside ink, gap = 4 px
    conn_px = 2.0 / PX2MM                                     # 2 mm = 7.6 px: well over the 4 px gap
    plain, c_plain, _, _ = _route_low_lift(lines, conn_px)
    assert len(plain) == 1 and len(c_plain) == 1, "control: without the check the router bridges the white gap"
    st = {}
    out, conns, _, _ = _route_low_lift(lines, conn_px, connector_check=_identity_check(mask), check_stats=st)
    assert len(out) == 2 and conns == [], "with the check the fuse must become a pen lift"
    assert (st["fuses_attempted"], st["fuses_accepted"], st["fuses_rejected"]) == (1, 0, 1)
    assert st["unchecked_offmask_mm"] > 0.05 and st["accepted_offmask_mm"] == 0.0
    assert len(st["rejected_segments"]) == 1


def test_fuse_inside_ink_is_accepted_and_matches_the_unchecked_router():
    mask = _bar()
    lines = _two_strokes(29.0, 32.0)
    conn_px = 2.0 / PX2MM
    plain = _route_low_lift(lines, conn_px)
    st = {}
    chk = _route_low_lift(lines, conn_px, connector_check=_identity_check(mask), check_stats=st)
    assert len(chk[0]) == 1 and len(chk[1]) == 1
    assert all(np.array_equal(a, b) for a, b in zip(plain[0], chk[0])) and plain[2:] == chk[2:]
    assert (st["fuses_attempted"], st["fuses_accepted"], st["fuses_rejected"]) == (1, 1, 0)
    assert st["accepted_offmask_mm"] == 0.0 and st["accepted_spill_mm"] == 0.0


def test_connector_near_the_edge_is_rejected_one_pen_inside_passes():
    """Mask edge: first ink row is y=10. A connector at y=9.5 has dt 0.5 px (< thr 0.6) and is rejected; at y=12 it passes."""
    mask = _bar(y0=10, y1=30)
    near = [_c((10, 9.5), (29, 9.5)), _c((32, 9.5), (60, 9.5))]           # bilinear dt = 0.5 < 0.6 - 0.05
    deep = [_c((10, 12.0), (29, 12.0)), _c((32, 12.0), (60, 12.0))]
    conn_px = 2.0 / PX2MM
    o1 = _route_low_lift(near, conn_px, connector_check=_identity_check(mask))
    o2 = _route_low_lift(deep, conn_px, connector_check=_identity_check(mask))
    assert len(o1[0]) == 2, "bilinear dt below the half-pitch threshold must be refused"
    assert len(o2[0]) == 1


def test_zero_length_joins_are_exempt_even_at_a_thin_feature():
    mask = np.zeros((40, 200), bool)
    mask[19:21, :] = True                                     # 2 px tall: dt max 1.0 but thin
    chk = _identity_check(mask, fill_px=4.0)                  # thr 2.0 > any dt in this mask
    lines = [_c((10, 19.5), (30, 19.5)), _c((30, 19.5), (60, 19.5))]
    out, conns, _, _ = _route_low_lift(lines, 2.0 / PX2MM, connector_check=chk)
    assert len(out) == 1 and len(conns) == 1                  # coincident endpoints: nothing to test
    lines2 = [_c((10, 19.5), (30, 19.5)), _c((32, 19.5), (60, 19.5))]
    out2, _, _, _ = _route_low_lift(lines2, 2.0 / PX2MM, connector_check=chk)
    assert len(out2) == 2, "control: a real connector in the same thin feature IS refused"


def test_target_backfill_skips_rejected_gaps_and_counts_them():
    mask = _bar(w=400)
    mask[:, 100:104] = False
    lines = [_c((10, 20), (50, 20)), _c((53, 20), (98, 20)), _c((106, 20), (150, 20)), _c((154, 20), (200, 20))]
    conn_px = 0.4 / PX2MM                                     # 1.5 px: fuses the 3 px? no -> only via the target
    hard_cap = 3.0 / PX2MM
    plain, cp, _, _ = _route_low_lift(lines, conn_px, target_strokes=1, hard_cap_px=hard_cap)
    st = {}
    out, conns, _, _ = _route_low_lift(lines, conn_px, target_strokes=1, hard_cap_px=hard_cap,
                                       connector_check=_identity_check(mask), check_stats=st)
    assert len(plain) == 1, "control: the unchecked backfill reaches the target"
    assert len(out) == 2, "the 8 px white gap must stay a lift, the others fuse"
    assert st["fuses_attempted"] == 3 and st["fuses_accepted"] == 2 and st["fuses_rejected"] == 1
    assert st["strokes_unchecked"] == 1 and st["strokes_checked"] == 2


def test_no_mask_check_is_fail_closed_but_exempts_zero_length():
    lines = [_c((0, 0), (10, 0)), _c((10, 0), (20, 0)), _c((22, 0), (30, 0))]
    out, conns, _, _ = _route_low_lift(lines, 2.0 / PX2MM, connector_check=NoMaskCheck("test"))
    assert len(out) == 2 and len(conns) == 1                  # the zero-length join only


def test_off_state_is_the_original_router_byte_for_byte():
    rng = np.random.default_rng(7)
    lines = [np.cumsum(rng.normal(0, 2, 6) + 1j * rng.normal(0, 2, 6)) + complex(rng.uniform(0, 120), rng.uniform(0, 80))
             for _ in range(300)]
    a = _route_low_lift(lines, 1.2 / PX2MM, 40, 5.0 / PX2MM)
    b = _route_low_lift(lines, 1.2 / PX2MM, 40, 5.0 / PX2MM, connector_check=None, check_stats=None)
    assert all(np.array_equal(x, y) for x, y in zip(a[0], b[0]))
    assert all(np.array_equal(x, y) for x, y in zip(a[1], b[1])) and a[2:] == b[2:]
    # checked, over an all-ink mask: nothing to reject -> identical to the unchecked result
    allink = np.ones((400, 600), bool)
    st = {}
    c = _route_low_lift(lines, 1.2 / PX2MM, 40, 5.0 / PX2MM, connector_check=_identity_check(allink), check_stats=st)
    assert all(np.array_equal(x, y) for x, y in zip(a[0], c[0])) and a[2:] == c[2:]
    assert st["fuses_rejected"] == 0 and st["fuses_accepted"] == len(a[1])


# --------------------------------------------------------------------------- transform vs real vpype

def _vpype_place(svg, s, rot, tx_mm, ty_mm, out):
    import vpype_cli
    vpype_cli.execute(f"read -l 1 {Path(svg).as_posix()} scale -l 1 {s} {s} rotate -l 1 -- {rot} translate -l 1 -- {tx_mm}mm {ty_mm}mm pagesize 600x600mm write {Path(out).as_posix()}")


@pytest.mark.parametrize("rot", [0.0, 90.0, 180.0, 270.0, 33.0])
def test_to_trace_inverts_the_real_vpype_placement(tmp_path, rot):
    import vpype as vp
    pts = [(20.0, 15.0), (180.0, 22.0), (95.5, 60.25), (20.0, 15.0)]
    svg = tmp_path / "in.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="200px" height="80px" viewBox="0 0 200 80">'
                   '<g stroke="black" fill="none"><polyline points="20,15 180,22 95.5,60.25"/>'
                   '<polyline points="30,70 150,10"/></g></svg>')
    doc = vp.read_multilayer_svg(str(svg), quantization=0.1)
    b = doc.bounds()
    c0 = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    s, tx_mm, ty_mm = 0.83, 141.5, 112.25
    out = tmp_path / "out.svg"
    _vpype_place(svg, s, rot, tx_mm, ty_mm, out)
    placed = vp.read_multilayer_svg(str(out), quantization=0.1)
    page = np.concatenate([np.asarray(l) for l in placed.layers[1]])
    src = np.concatenate([np.asarray(l) for l in doc.layers[1]])
    m = np.ones((80, 200), bool)
    good = LayerMaskCheck(m, FILL_PX, c0, s, rot, (tx_mm / PX2MM, ty_mm / PX2MM))
    back = good.to_trace(page)
    assert np.abs(back[:, 0] + 1j * back[:, 1] - src).max() < 0.02, "inverse must recover the source vertices"
    if rot not in (0.0,):
        wrong = LayerMaskCheck(m, FILL_PX, c0, s, 0.0, (tx_mm / PX2MM, ty_mm / PX2MM))     # rotation ignored
        bw = wrong.to_trace(page)
        assert np.abs(bw[:, 0] + 1j * bw[:, 1] - src).max() > 1.0, "control: a wrong placement must NOT recover"


def test_self_check_passes_for_the_true_placement_and_fails_for_a_wrong_one():
    mask = np.zeros((80, 200), bool)
    mask[10:70, 20:180] = True
    v = _c((30, 20), (170, 60), (100, 40))
    ok = build_layer_check_from(mask, v, dx=0.0)
    assert ok.available and ok.self_check_frac > 0.98
    bad = build_layer_check_from(mask, v + 500.0, dx=0.0)       # vertices nowhere near the mask
    assert not bad.available and "placement_self_check_failed" in bad.reason


def build_layer_check_from(mask, vertices, dx):
    import ccplotter.stage_e_mask as sm
    p = Path(tempfile.mkdtemp(prefix="cmc_sc_")) / "t.svg"
    sm.write_sidecar(p, mask, FILL_PX)
    ctx = {"svg": p, "s": 1.0, "rot_deg": 0.0, "c0_px": (0.0, 0.0), "t_px": (dx, 0.0)}
    return sm.build_layer_check(ctx, vertices)


# --------------------------------------------------------------------------- route integration

def _fragmented_png(path):
    """Rows of thin horizontal lines broken by 5 px white gaps (~0.33 mm at the placed scale): the
    skeleton strokes end facing each other across white, which is exactly what Stage E fuses."""
    img = np.full((600, 1500), 255, np.uint8)
    for row, y in enumerate((120, 260, 400)):
        x = 40 + 17 * row
        for _ in range(12):
            cv2.rectangle(img, (x, y), (x + 100, y + 1), 0, -1)
            x += 100 + 5
    cv2.imwrite(str(path), img)


def _run(did, extra, rot=0):
    from ccplotter.routes.core import app
    _fragmented_png(WORK_DIR / f"src_{did}.png")
    resp = app.test_client().post("/api/process", json={
        "job_type": "preview", "items": [{"id": did, "scale": 1, "rot": rot, "dx": 3.0, "dy": -2.0, "crop": None}],
        "style": "tracev3", "resolution": 1500, "scale_aware": True, "threshold_mode": "otsu",
        "connector_distance": 0.5, "paper": "120x120", "margin": 10.0, "debug": True, **extra})
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    text = (WORK_DIR / body["debug_files"]["routed_svg"]).read_text(encoding="utf-8")
    pts = sorted(tuple(t.split(",")) for ps in re.findall(r'points="([^"]+)"', text) for t in ps.split() if "," in t)
    renders = sorted(p.name for p in WORK_DIR.glob(f"src_{did}_render_*.svg"))
    return body, pts, renders


def test_route_absent_false_and_junk_give_identical_output_and_cache_key():
    body0, a, ra = _run("cmc_absent", {})
    assert "connector_mask_check" not in body0 and "connector_mask_check" not in body0["debug_files"]
    for i, v in enumerate((False, 0, None, "false", "0", "junk")):
        b_, b, rb = _run(f"cmc_inert{i}", {"connector_mask_check": v})
        assert a and a == b, v
        assert re.sub(r"src_[a-z_0-9]+_render_", "", ra[0]) == re.sub(r"src_[a-z_0-9]+_render_", "", rb[0]), v
        assert "connector_mask_check" not in b_


@pytest.mark.parametrize("rot", [0, 90])
def test_route_on_plumbs_the_mask_and_records_a_consistent_layer(rot):
    """Through the real route: the trace mask reaches Stage E (sidecar beside the render), the
    reconstructed placement reproduces the layer's own ink, the record is listed and saved. (Whether a
    given synthetic trace has Stage E hazards is picture-dependent; the effect is tested at doc level
    below.)"""
    base, pts0, _ = _run(f"cmc_off{rot}", {}, rot)
    on, pts1, r1 = _run(f"cmc_on{rot}", {"connector_mask_check": True}, rot)
    rec = on["connector_mask_check"]
    t = rec["totals"]
    assert t["layers_checked"] == 1 and t["layers_failed_closed"] == 0, rec
    lay = list(rec["layers"].values())[0]
    assert lay["self_check_frac"] > 0.98, "placement reconstruction must reproduce the layer's own ink"
    assert t["fuses_attempted"] == t["fuses_accepted"] + t["fuses_rejected"]
    assert t["strokes_unchecked"] == base["strokes_after"], "the 'unchecked' replay must equal the real off-run"
    assert t["accepted_offmask_mm"] <= t["unchecked_offmask_mm"] + 1e-9
    assert on["debug_files"].get("connector_mask_check") == "debug_connector_mask_check.json"
    saved = json.loads((WORK_DIR / "debug_connector_mask_check.json").read_text())
    assert saved["totals"] == rec["totals"]
    assert (WORK_DIR / (r1[0] + ".mask.npz")).exists()
    assert "connector_mask_check" not in base and "connector_mask_check" not in base["debug_files"]


def _write_doc(path, strokes, w=300, h=100):
    import vpype as vp
    doc = vp.Document()
    lc = vp.LineCollection()
    for s in strokes:
        lc.append(s)
    doc.add(lc, 1)
    doc.page_size = (w, h)
    with open(path, "w") as f:
        vp.write_svg(f, doc, page_size=(w, h), color_mode="none")


def test_route_doc_rejects_white_gap_fuse_and_keeps_in_ink_fuse_with_unchecked_control():
    from ccplotter.stats_warnings import _route_low_lift_doc
    d = Path(tempfile.mkdtemp(prefix="cmc_doc_"))
    m = np.zeros((100, 300), bool)
    m[46:54, 10:101] = True; m[46:54, 106:197] = True      # y=50: two bars, 6 px white gap
    m[76:84, 10:197] = True                                 # y=80: one continuous bar
    svg = d / "layer.svg"
    strokes = [_c((10, 50), (100, 50)), _c((106, 50), (196, 50)), _c((10, 80), (100, 80)), _c((103, 80), (196, 80))]
    _write_doc(svg, strokes)
    from ccplotter.stage_e_mask import write_sidecar
    write_sidecar(svg, m, 2.4)
    ctx = {1: {"svg": svg, "s": 1.0, "rot_deg": 0.0, "c0_px": (0.0, 0.0), "t_px": (0.0, 0.0)}}
    conn = 2.0                                              # mm; 2.0 mm = 7.56 px > 6 px gap
    off = _route_low_lift_doc(svg, d / "off.svg", conn, None)
    on = _route_low_lift_doc(svg, d / "on.svg", conn, None, mask_ctx=ctx)
    assert "connector_mask_check" not in off
    t = on["connector_mask_check"]["totals"]
    # control: unchecked fuses both pairs (off-run strokes 4 -> 2) and ~5 px (the white pixels) of that is off the mask
    assert off["strokes_after"] == 2 and t["strokes_unchecked"] == 2
    assert t["fuses_planned_unchecked"] == 2 and t["fuses_attempted"] == 2
    assert t["fuses_accepted"] == 1 and t["fuses_rejected"] == 1
    assert on["strokes_after"] == 3
    assert t["unchecked_offmask_mm"] > 4.0 * PX2MM and t["accepted_offmask_mm"] == 0.0
    rej = on["connector_mask_check"]["_rejected_segments_page_px"]
    assert len(rej) == 1


def test_fail_closed_layer_raises_a_visible_warning_and_is_absent_otherwise():
    """A layer with no usable mask is fail-closed (no bridges) and must say so in
    build_warnings; a healthy run, an off run and an empty layer must not."""
    from ccplotter.stage_e_mask import write_sidecar
    from ccplotter.stats_warnings import _route_low_lift_doc, build_warnings
    d = Path(tempfile.mkdtemp(prefix="cmc_warn_"))
    svg = d / "layer.svg"
    _write_doc(svg, [_c((10, 50), (100, 50)), _c((106, 50), (196, 50))])
    # build_warnings needs the two base counters; everything else it reads is optional.
    _st = lambda st: {"stroke_count": st["strokes_after"], "estimated_z_moves": st["z_moves_after"], **st}
    msgs = lambda st: [w["msg"] for w in build_warnings(_st(st)) if "mask check unavailable" in w["msg"]]

    # (1) layer 1 has NO ctx at all -> fail closed -> warning naming layer 1
    res = _route_low_lift_doc(svg, d / "a.svg", 2.0, None, mask_ctx={})
    assert res["connector_mask_check"]["totals"]["layers_failed_closed"] == 1
    assert res["strokes_after"] == res["strokes_before"], "fail-closed: no bridge may be made"
    got = msgs(res)
    assert len(got) == 1 and got[0].startswith("mask check unavailable for layer 1 -- lifts not reduced there"), got
    assert all(ord(ch) < 128 for ch in got[0])
    assert [w for w in build_warnings(_st(res)) if "mask check unavailable" in w["msg"]][0]["tier"] == "primary"

    # (2) ctx present but the sidecar file is missing -> same outcome via the sidecar path
    ctx = {1: {"svg": svg, "s": 1.0, "rot_deg": 0.0, "c0_px": (0.0, 0.0), "t_px": (0.0, 0.0)}}
    res = _route_low_lift_doc(svg, d / "b.svg", 2.0, None, mask_ctx=ctx)
    assert len(msgs(res)) == 1 and "no_trace_mask_sidecar" in msgs(res)[0]

    # (3) CONTROL: a healthy layer (own-ink mask + sidecar) gives no such warning
    m = np.zeros((100, 300), bool)
    m[46:54, 10:101] = True; m[46:54, 106:197] = True
    write_sidecar(svg, m, 2.4)
    res = _route_low_lift_doc(svg, d / "c.svg", 2.0, None, mask_ctx=ctx)
    assert res["connector_mask_check"]["totals"]["layers_failed_closed"] == 0 and msgs(res) == []

    # (4) CONTROL: the off run (no record) has none either, and an empty layer is not a failure
    assert msgs(_route_low_lift_doc(svg, d / "d.svg", 2.0, None)) == []
    empty = d / "empty.svg"
    _write_doc(empty, [])
    assert msgs(_route_low_lift_doc(empty, d / "e.svg", 2.0, None, mask_ctx={})) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
