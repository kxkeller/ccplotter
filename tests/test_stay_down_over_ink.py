"""Opt-in "Stay down over ink (no stray lines)" -- body flag stay_down_over_ink, default OFF.
Rule: after routing, a pen-up travel becomes a pen-down connector, of ANY
length, iff the straight connector stays inside the layer's OWN ink mask (ccplotter/stage_e_mask.py predicate).

Could-not-fail doctrine: every acceptance has a rejection next to it, every OFF has a positive control proving ON would
have changed something.
  * router unit: connector across white rejected (the unchecked router fuses the same pair -- control); connector fully
    inside ink accepted, at a length far beyond the distance setting; OFF (check None) == original, and an
    all-rejecting check == OFF byte for byte.
  * doc level: ON reduces lifts on a fixture with in-ink gaps; a layer with no mask fails closed and warns.
  * route (/api/process): absent / false / junk == identical routed geometry; ON joins strokes that OFF leaves.
  * two-pen (generate_layers): OFF == default; ON reduces lifts; every stay-down connector re-checks inside the
    layer's own ink; synthetic connectors across white are rejected by the same checker; flag parsed on every
    entry path and never forwarded to the plain job.
Temp runtime home + work dir, Flask test client, never :5001.
"""
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home
_test_home.use_temp_home("ccplotter_test_home_sdoi_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_sdoi_"))
sys.path.insert(0, str(REPO))

import base64
import shutil

import cv2
import numpy as np
import pytest

from ccplotter.config import PX2MM, WORK_DIR
from ccplotter.stage_e_mask import LayerMaskCheck, NoMaskCheck, write_sidecar
from ccplotter.stats_warnings import _route_low_lift, _route_low_lift_doc

FILL_PX = 1.2


def _c(*pts):
    return np.array([complex(x, y) for x, y in pts])


def _chk(mask):
    return LayerMaskCheck(mask, FILL_PX, (0.0, 0.0), 1.0, 0.0, (0.0, 0.0))


def _same_lines(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


# --------------------------------------------------------------------------- router unit
def test_connector_across_white_is_rejected_and_the_unchecked_router_would_join_it():
    mask = np.zeros((60, 400), bool)
    mask[10:30, 0:100] = True
    mask[10:30, 160:260] = True                         # 60 px white gap between two ink bars
    lines = [_c((10, 20), (99, 20)), _c((160, 20), (250, 20))]
    big = 400.0                                         # px: unlimited distance
    ctl, c_ctl, _, _ = _route_low_lift(lines, big)
    assert len(ctl) == 1 and len(c_ctl) == 1, "control: a huge unchecked distance bridges the white gap"
    st = {}
    out, conns, _, _ = _route_low_lift(lines, 0.0, stay_down_check=_chk(mask), stay_down_stats=st)
    assert len(out) == 2 and conns == []
    assert (st["candidates"], st["fused"], st["rejected"]) == (1, 0, 1)


def test_long_connector_fully_inside_ink_is_accepted_beyond_the_distance_setting():
    mask = np.zeros((60, 400), bool)
    mask[10:30, :] = True                               # one continuous 400 px bar
    lines = [_c((10, 20), (60, 20)), _c((300, 20), (390, 20))]      # 240 px (= 63 mm) travel, all in ink
    off, c_off, _, _ = _route_low_lift(lines, 0.5 / PX2MM)
    assert len(off) == 2 and c_off == [], "control: the 0.5 mm router leaves this long travel as a pen lift"
    st = {}
    on, c_on, clen, longest = _route_low_lift(lines, 0.5 / PX2MM, stay_down_check=_chk(mask), stay_down_stats=st)
    assert len(on) == 1 and len(c_on) == 1 and abs(longest - 240.0) < 1e-6
    assert (st["lifts_before"], st["lifts_after"], st["fused"]) == (2, 1, 1)
    assert st["longest_fused_mm"] > 60.0


def test_off_is_the_original_router_and_an_all_rejecting_check_equals_off():
    mask = np.zeros((60, 400), bool)
    mask[10:30, :] = True
    rng = np.random.default_rng(1)
    lines = [_c((x, 20), (x + 20, 20)) for x in rng.permutation(np.arange(10, 360, 40))]
    base = _route_low_lift(lines, 0.5 / PX2MM)
    none = _route_low_lift(lines, 0.5 / PX2MM, stay_down_check=None, stay_down_stats=None)
    assert _same_lines(base[0], none[0]) and len(base[1]) == len(none[1])
    blank = np.zeros_like(mask)                         # nothing is ink -> nothing may be joined
    rej = _route_low_lift(lines, 0.5 / PX2MM, stay_down_check=_chk(blank))
    assert _same_lines(base[0], rej[0]) and len(base[1]) == len(rej[1])
    on = _route_low_lift(lines, 0.5 / PX2MM, stay_down_check=_chk(mask))
    assert len(on[0]) < len(base[0]), "positive control: ON must join something here"
    assert len(on[0]) == 1


# --------------------------------------------------------------------------- doc level
def _write_doc(path, strokes, w=200, h=100):
    import vpype as vp
    doc = vp.Document()
    lc = vp.LineCollection()
    for s in strokes:
        lc.append(s)
    doc.add(lc, 1)
    doc.page_size = (w, h)
    with open(path, "w") as f:
        vp.write_svg(f, doc, page_size=(w, h), color_mode="none")


def _doc_fixture(tmp):
    """y=10: two bars with a 60 px white gap (reject).  y=90: one continuous bar cut in two, 60 px (~16 mm) in-ink travel (accept).
    The rows are 80 px apart, so the nearest-endpoint walk takes both 60 px gaps."""
    m = np.zeros((100, 200), bool)
    m[6:14, 10:61] = True
    m[6:14, 120:171] = True
    m[86:94, 10:171] = True
    svg = tmp / "layer.svg"
    strokes = [_c((10, 10), (60, 10)), _c((120, 10), (170, 10)), _c((10, 90), (60, 90)), _c((120, 90), (170, 90))]
    _write_doc(svg, strokes)
    write_sidecar(svg, m, 2.4)
    ctx = {1: {"svg": svg, "s": 1.0, "rot_deg": 0.0, "c0_px": (0.0, 0.0), "t_px": (0.0, 0.0)}}
    return svg, ctx


def test_doc_reduces_lifts_rejects_white_and_records(tmp_path):
    svg, ctx = _doc_fixture(tmp_path)
    off = _route_low_lift_doc(svg, tmp_path / "off.svg", 0.5, None)
    on = _route_low_lift_doc(svg, tmp_path / "on.svg", 0.5, None, stay_down_ctx=ctx)
    assert "stay_down_over_ink" not in off
    assert off["strokes_after"] == 4, "control: nothing is within 0.5 mm"
    rec = on["stay_down_over_ink"]
    t = rec["totals"]
    # the y=90 pair (in ink) joins; the y=10 pair (white between) and the cross-row travels do not
    assert on["strokes_after"] == 3 and off["strokes_after"] == 4
    assert t["fused"] >= 1 and t["rejected"] >= 1 and t["candidates"] == t["fused"] + t["rejected"]
    assert t["layers_checked"] == 1 and t["layers_failed_closed"] == 0 and rec["enabled"] is True
    assert on["longest_connector_mm"] > 10.0, "the in-ink connector is far longer than the 0.5 mm distance setting"
    # the same pair of white-separated bars is never bridged
    import vpype as vp
    doc = vp.read_multilayer_svg(str(tmp_path / "on.svg"), quantization=0.1)
    for ln in doc.layers[1]:
        a = np.asarray(ln)
        for p, q in zip(a[:-1], a[1:]):
            if abs(p.imag - 10) < 1 and abs(q.imag - 10) < 1:
                assert not (p.real < 90 < q.real or q.real < 90 < p.real), "a connector crossed the white gap"
    # independent of mask_ctx: both on is allowed and the short fuses keep their own check
    both = _route_low_lift_doc(svg, tmp_path / "both.svg", 0.5, None, mask_ctx=ctx, stay_down_ctx=ctx)
    assert "connector_mask_check" in both and "stay_down_over_ink" in both


def test_doc_layer_without_mask_fails_closed_and_warns(tmp_path):
    from ccplotter.stats_warnings import build_warnings
    svg, ctx = _doc_fixture(tmp_path)
    res = _route_low_lift_doc(svg, tmp_path / "a.svg", 0.5, None, stay_down_ctx={})
    assert res["strokes_after"] == res["strokes_before"], "fail closed: nothing may be joined"
    assert res["stay_down_over_ink"]["totals"]["layers_failed_closed"] == 1
    st = {"stroke_count": res["strokes_after"], "estimated_z_moves": res["z_moves_after"], **res}
    got = [w["msg"] for w in build_warnings(st) if "stay-down mask unavailable" in w["msg"]]
    assert len(got) == 1 and all(ord(ch) < 128 for ch in got[0])
    ok = _route_low_lift_doc(svg, tmp_path / "b.svg", 0.5, None, stay_down_ctx=ctx)
    st = {"stroke_count": ok["strokes_after"], "estimated_z_moves": ok["z_moves_after"], **ok}
    assert not [w for w in build_warnings(st) if "stay-down mask unavailable" in w["msg"]]


# --------------------------------------------------------------------------- /api/process
def _blocks_png(path):
    """40 solid discs inside a thick frame: separate strokes whose travels are sometimes wholly in ink (disc touching the
    frame or another disc) and sometimes across bare paper."""
    img = np.full((900, 1500), 255, np.uint8)
    rng = np.random.default_rng(0)
    for _ in range(40):
        cv2.circle(img, (int(rng.integers(100, 1400)), int(rng.integers(100, 800))), int(rng.integers(25, 60)), 0, -1)
    cv2.rectangle(img, (100, 100), (1400, 800), 0, 30)
    cv2.imwrite(str(path), img)


def _run(did, extra):
    from ccplotter.routes.core import app
    _blocks_png(WORK_DIR / f"src_{did}.png")
    resp = app.test_client().post("/api/process", json={
        "job_type": "preview", "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0.0, "dy": 0.0, "crop": None}],
        "style": "tracev3", "resolution": 1500, "scale_aware": True, "threshold_mode": "otsu",
        "connector_distance": 0.0, "paper": "120x120", "margin": 10.0, "debug": True, **extra})
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    text = (WORK_DIR / body["debug_files"]["routed_svg"]).read_text(encoding="utf-8")
    pts = sorted(tuple(t.split(",")) for ps in re.findall(r'points="([^"]+)"', text) for t in ps.split() if "," in t)
    return body, pts


def test_route_absent_false_and_junk_are_identical_and_on_joins_strokes():
    b0, a = _run("sdoi_absent", {})
    assert "stay_down_over_ink" not in b0
    for i, v in enumerate((False, 0, None, "false", "0", "junk")):
        b, p = _run(f"sdoi_inert{i}", {"stay_down_over_ink": v})
        assert a and a == p, v
        assert "stay_down_over_ink" not in b and b["strokes_after"] == b0["strokes_after"], v
    on, p_on = _run("sdoi_on", {"stay_down_over_ink": True})
    rec = on["stay_down_over_ink"]
    assert rec["totals"]["layers_checked"] == 1 and rec["totals"]["layers_failed_closed"] == 0, rec
    assert rec["totals"]["fused"] > 0 and on["strokes_after"] < b0["strokes_after"], (on["strokes_after"], b0["strokes_after"])
    assert rec["totals"]["rejected"] > 0, "positive control: travels across bare paper exist here and must stay pen lifts"
    assert on["strokes_after"] > 1
    assert on["z_moves_after"] < b0["z_moves_after"]
    assert on["strokes_after"] == b0["strokes_after"] - rec["totals"]["fused"]
    assert rec["totals"]["fused"] + rec["totals"]["rejected"] == rec["totals"]["candidates"]


# --------------------------------------------------------------------------- two-pen
IMG = REPO / "tests" / "fixtures" / "twotone" / "pagoda.jpg"
PAPER, MARGIN, BLACK = "150x110", 5.0, "bic_cristal_black"


@pytest.fixture(scope="module")
def prep(tmp_path_factory):
    from ccplotter import twopen_keytint as kt
    wd = tmp_path_factory.mktemp("sdoi_prep")
    return kt.capture_plain_bwphoto(IMG, wd, paper_w=150.0, paper_h=110.0, margin=MARGIN, pen_id=BLACK,
                                    paper_spec=PAPER, body_extra={"connector_distance": 0.5})


def _gen(prep, tmp_path_factory, name, **kw):
    from ccplotter import twopen_keytint as kt
    jd = tmp_path_factory.mktemp(name)
    return jd, kt.generate_layers(prep, 0.375, jd, {"blue": 0.12, "black": 0.12}, **kw)


@pytest.fixture(scope="module")
def run_default(prep, tmp_path_factory):
    return _gen(prep, tmp_path_factory, "sdoi_def")


@pytest.fixture(scope="module")
def run_off(prep, tmp_path_factory):
    return _gen(prep, tmp_path_factory, "sdoi_off", stay_down_over_ink=False)


@pytest.fixture(scope="module")
def run_on(prep, tmp_path_factory):
    return _gen(prep, tmp_path_factory, "sdoi_on", stay_down_over_ink=True)


def test_twopen_off_equals_default_byte_for_byte(run_default, run_off):
    for color in ("blue", "black"):
        a, b = run_default[1]["layers"][color], run_off[1]["layers"][color]
        assert (a and b) and Path(a).read_bytes() == Path(b).read_bytes(), color
    assert "stay_down_over_ink" not in run_default[1]["info"] and "stay_down_over_ink" not in run_off[1]["info"]


def test_twopen_on_reduces_lifts_records_and_every_connector_is_inside_own_ink(prep, run_default, run_on):
    from ccplotter import twopen_keytint as kt
    from ccplotter.stage_e_mask import load_sidecar
    g0, g1 = run_default[1], run_on[1]
    n0 = {c: len(g0["strokes"][c]) for c in ("blue", "black")}
    n1 = {c: len(g1["strokes"][c]) for c in ("blue", "black")}
    assert sum(n1.values()) < sum(n0.values()), (n0, n1)
    rec = g1["info"]["stay_down_over_ink"]
    assert rec["enabled"] and set(rec["layers"]) == {"blue", "black"}
    jd = run_on[0]
    ox, oy = prep["c0_px"]
    for color in ("blue", "black"):
        lay = rec["layers"][color]
        assert lay["totals"]["layers_checked"] == 1 and lay["totals"]["fused"] > 0, (color, lay)
        assert n1[color] == n0[color] - lay["totals"]["fused"], color
        tmask, fpx = load_sidecar(jd / f"{color}_sd_trace.svg")
        chk = LayerMaskCheck(tmask, fpx, (ox, oy), float(prep["fit"]), float(prep.get("rot", 0.0)),
                             (prep["tx"] / PX2MM, prep["ty"] / PX2MM))
        conns = kt.load_strokes_mm(jd / f"{color}.connectors.svg")
        # the connectors file also holds the 0.5 mm unchecked ones; keep the long (stay-down only) ones
        long_ = [s for s in conns if np.hypot(*(s[-1] - s[0])) > 0.6]
        assert long_, f"no long connectors on {color}: the test would be vacuous"
        A = np.array([(s[0, 0] + 1j * s[0, 1]) / PX2MM for s in long_])
        B = np.array([(s[-1, 0] + 1j * s[-1, 1]) / PX2MM for s in long_])
        assert chk.ok(A, B).all(), color
        # positive control: a travel from one ink pixel to a far ink pixel across white must be rejected
        ys, xs = np.nonzero(tmask)
        i0 = 0
        far = np.argmax(np.hypot(xs - xs[i0], ys - ys[i0]))
        path = np.linspace(0, 1, 60)
        sx = np.rint(xs[i0] + (xs[far] - xs[i0]) * path).astype(int)
        sy = np.rint(ys[i0] + (ys[far] - ys[i0]) * path).astype(int)
        if (~tmask[sy, sx]).sum() >= 3:         # only a real across-white pair is a control
            rot = complex(np.cos(np.radians(prep.get("rot", 0.0))), np.sin(np.radians(prep.get("rot", 0.0))))
            c0, t = complex(ox, oy), complex(prep["tx"] / PX2MM, prep["ty"] / PX2MM)
            fwd = lambda x, y: c0 + rot * float(prep["fit"]) * (complex(x, y) - c0) + t
            assert not chk.ok(np.array([fwd(xs[i0], ys[i0])]), np.array([fwd(xs[far], ys[far])])).any()


@pytest.fixture(scope="module")
def run_both(prep, tmp_path_factory):
    return _gen(prep, tmp_path_factory, "sdoi_both", blue_trap=True, blue_fill_mm=0.12, stay_down_over_ink=True)


@pytest.fixture(scope="module")
def run_trap(prep, tmp_path_factory):
    return _gen(prep, tmp_path_factory, "sdoi_trap", blue_trap=True, blue_fill_mm=0.12)


def test_combination_blue_trap_and_stay_down(prep, run_trap, run_both):
    """blue_trap + stay_down together. Blue's 0.5 mm connectors lie in the blue+black UNION (trap check), blue's
    stay-down joins lie in blue's OWN ink; black's stay-down join count equals what stay-down adds on top of the blue_trap-only run
    (blue_trap changes the black mask -- specks fold to blue -- so the like-for-like baseline is trap-only, not the no-trap run)."""
    from ccplotter import twopen_keytint as kt
    from ccplotter.stage_e_mask import load_sidecar
    jd, g = run_both
    ox, oy = prep["c0_px"]
    mk = lambda m, f: LayerMaskCheck(m, f, (ox, oy), float(prep["fit"]), float(prep.get("rot", 0.0)), (prep["tx"] / PX2MM, prep["ty"] / PX2MM))
    conns = kt.load_strokes_mm(jd / "blue.connectors.svg")
    ln = np.array([np.hypot(*(s[-1] - s[0])) for s in conns])
    toA = lambda ss: (np.array([(s[0, 0] + 1j * s[0, 1]) / PX2MM for s in ss]), np.array([(s[-1, 0] + 1j * s[-1, 1]) / PX2MM for s in ss]))
    short = [s for s, l in zip(conns, ln) if l <= 0.5 + 1e-3]
    long_ = [s for s, l in zip(conns, ln) if l > 0.5 + 1e-3]
    assert short and long_, "vacuous: need both kinds of blue connector"
    union, ufpx = load_sidecar(jd / "blue_chk_trace.svg")
    own, ofpx = load_sidecar(jd / "blue_sd_trace.svg")
    assert not np.array_equal(union, own), "vacuous: union mask must differ from blue's own mask"
    assert mk(union, ufpx).ok(*toA(short)).all(), "a 0.5 mm blue connector left the blue+black union"
    # The connectors file is read back through SVG rounding (2 decimals + 0.1 px quantisation) while the router decided on the
    # unrounded endpoints (measured: 1 of 709 sits ~0.03 px under the threshold on read-back), so the re-check gets 0.25 px slack.
    own_chk = LayerMaskCheck(own, ofpx, (ox, oy), float(prep["fit"]), float(prep.get("rot", 0.0)), (prep["tx"] / PX2MM, prep["ty"] / PX2MM), tol_px=0.25)
    assert own_chk.ok(*toA(long_)).all(), "a blue stay-down join left blue's own ink"
    # positive control: the own-mask check really rejects (travels across paper / outside own ink exist)
    bl = g["info"]["stay_down_over_ink"]["layers"]
    assert bl["blue"]["totals"]["rejected"] > 0
    assert bl["blue"]["totals"]["fused"] == len(long_) and bl["blue"]["totals"]["layers_failed_closed"] == 0
    k0, k1 = len(run_trap[1]["strokes"]["black"]), len(g["strokes"]["black"])
    assert k1 < k0 and bl["black"]["totals"]["fused"] == k0 - k1
    assert len(run_trap[1]["strokes"]["blue"]) - len(g["strokes"]["blue"]) == bl["blue"]["totals"]["fused"]
    assert g["info"]["blue_trap"]["enabled"] is True


def test_stay_down_flag_parse_all_entry_paths_and_not_forwarded_to_plain_job(prep):
    from ccplotter.routes.twopen import _keytint_prepare, TwoPenRefusal
    b64 = base64.b64encode(IMG.read_bytes()).decode()
    base = {"image_b64": b64, "paper": PAPER, "margin": MARGIN}
    assert _keytint_prepare(dict(base), "sdoi_b0")["stay_down_over_ink"] is False
    assert _keytint_prepare(dict(base, stay_down_over_ink=True), "sdoi_b1")["stay_down_over_ink"] is True
    assert _keytint_prepare(dict(base, stay_down_over_ink=False), "sdoi_b2")["stay_down_over_ink"] is False
    with pytest.raises(TwoPenRefusal):
        _keytint_prepare(dict(base, stay_down_over_ink="yes"), "sdoi_b3")
    assert _keytint_prepare({"image": str(IMG), "paper": PAPER, "margin": MARGIN, "stay_down_over_ink": True}, "sdoi_i1")["stay_down_over_ink"] is True
    did = "sdoi01"
    src = Path(WORK_DIR) / f"src_{did}.jpg"
    shutil.copyfile(IMG, src)
    try:
        kp = _keytint_prepare({"items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0}], "paper": PAPER, "margin": MARGIN,
                               "style": "bwphoto", "job_type": "preview", "stay_down_over_ink": True, "connector_distance": 0.5}, "sdoi_c1")
        assert kp["stay_down_over_ink"] is True and kp["prep"]["connector_distance"] == 0.5
        assert "stay_down_over_ink" not in kp["prep"]["response_summary"]
    finally:
        src.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
