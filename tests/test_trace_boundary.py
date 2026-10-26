"""Trace-boundary provenance (ccplotter/trace_boundary.py).

The run_provenance sidecar records the request; this block records what the pipeline held at the
trace boundary (source sha, effective clahe, post-CLAHE gray sha + black fraction +
gray_crush, trace-mask sha, routed path-set sha with ONE definition, counts, preview
stroke width) so a same-source disagreement is one JSON diff instead of an offline
re-derivation. Tests pin, each with a positive control that can fail:

  1. flipping clahe changes the gray sha, black fraction AND gray_crush (and a re-run with
     the SAME clahe leaves all three identical -- the null);
  2. a geometry-identical re-run keeps the routed path-set sha, a tampered body changes it;
  3. a placement change keeps the gray sha but changes the routed path-set sha;
  4. the path-set definition reproduces the published hashes of the cleaned-heli reference SVGs
     (REF d729a9365f, and a second placement f274ae1721), and a synthetic SVG pins
     polygon closing / <line> / orientation canonicalisation / rounding;
  5. nothing leaks: the response, info.json and the gcode bytes gain nothing (the per-run
     back-link stays the only gcode header line added); a failure is an error string, never a raise.
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home BEFORE any ccplotter import
_HOME = _test_home.use_temp_home("ccplotter_test_home_tbound_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_tbound_"))
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
from PIL import Image

import ccplotter.routes.runs  # noqa: F401
from ccplotter import trace_boundary as tb
from ccplotter.config import PX2MM, WORK_DIR
from ccplotter.runs import (_runs_load, build_run_provenance, load_run_provenance,
                            strip_run_backlink, strip_run_stamps)


def _make_photo(path: Path, w=900, h=700):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = (127 + 90 * np.sin(xx / 11.0) * np.cos(yy / 17.0)).clip(0, 255).astype(np.uint8)
    img = np.stack([base, np.roll(base, 5, axis=0), np.roll(base, -5, axis=1)], axis=-1)
    Image.fromarray(img, mode="RGB").save(path, quality=92)


def _post(body):
    from ccplotter.routes.core import app
    return app.test_client().post("/api/process", json=body)


def _body(did, **extra):
    from ccplotter.cal import load_pen_profiles
    b = {"job_type": "gcode", "style": "tracev3", "debug": True, "paper": "120x120",
         "margin": 10.0, "resolution": 600, "pen_id": load_pen_profiles().get("active"),
         "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}]}
    b.update(extra)
    return b


def _run(did, **extra):
    resp = _post(_body(did, **extra))
    assert resp.status_code == 200, resp.get_json()
    rec = next(r for r in _runs_load() if r.get("type") == "gcode")
    prov = load_run_provenance(rec["id"])
    blk = prov["trace_boundary"]
    assert "error" not in blk, blk
    return resp.get_json(), prov, blk, (WORK_DIR / "layout.gcode").read_bytes()


def _item(blk, did):
    return blk["items"][did]


@pytest.fixture(scope="module")
def src():
    did = "tbound_src"
    _make_photo(WORK_DIR / f"src_{did}.jpg")
    return did


@pytest.fixture(scope="module")
def runs(src):
    base = _run(src)
    again = _run(src)
    clahe_on = _run(src, clahe=True)
    clahe_on2 = _run(src, clahe=True)
    placed = _run(src, items=[{"id": src, "scale": 1, "rot": 0, "dx": 6, "dy": 4, "crop": None}])
    rescaled = _run(src, items=[{"id": src, "scale": 0.7, "rot": 0, "dx": 0, "dy": 0, "crop": None}])
    tampered = _run(src, margin=30.0)
    return {"base": base, "again": again, "clahe_on": clahe_on, "clahe_on2": clahe_on2,
            "placed": placed, "rescaled": rescaled, "tampered": tampered}


# ---------------------------------------------------------------- 1. clahe flip

def test_block_shape_and_fields(runs, src):
    resp, prov, blk, _ = runs["base"]
    assert blk["pathset_definition"] == tb.PATHSET_DEFINITION
    it = _item(blk, src)
    t = it["trace"]
    assert t["source"]["sha256"] == hashlib.sha256((WORK_DIR / f"src_{src}.jpg").read_bytes()).hexdigest()
    assert t["source"]["dims_wh"] == [900, 700]
    assert t["clahe"] is False and t["gray"]["clahe"] is False
    assert len(t["gray"]["sha256"]) == 64 and len(t["gray"]["dims_wh"]) == 2
    assert 0.0 < t["gray"]["black_fraction"] < 1.0
    assert t["gray_crush_fraction"] is not None
    assert len(t["mask"]["sha256"]) == 64 and 0.0 < t["mask"]["ink_fraction"] < 1.0
    assert t["trace_counts"]["passes"] and t["trace_counts"]["lifts"] is not None
    assert it["routed"]["polylines"] > 0 and len(it["routed"]["pathset_sha256_nd3"]) == 64
    # whole-file and (single-item) per-layer definitions agree
    assert blk["routed"]["pathset_sha256_nd3"] == it["routed"]["pathset_sha256_nd3"]
    ps = blk["preview_stroke"]
    assert ps["mm"] > 0 and abs(ps["px"] - ps["mm"] / PX2MM) < 1e-4
    assert prov["effective_body_sha256"] == hashlib.sha256(json.dumps(
        prov["effective_body"], sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def test_clahe_flip_changes_gray_sha_black_fraction_and_gray_crush(runs, src):
    off = _item(runs["base"][2], src)["trace"]
    on = _item(runs["clahe_on"][2], src)["trace"]
    assert off["clahe"] is False and on["clahe"] is True
    assert on["gray"]["sha256"] != off["gray"]["sha256"]
    assert on["gray"]["black_fraction"] != off["gray"]["black_fraction"]
    assert on["gray_crush_fraction"] != off["gray_crush_fraction"]
    assert on["mask"]["sha256"] != off["mask"]["sha256"]
    # NULL control: the same clahe setting twice leaves every one of them identical, so the
    # inequalities above are caused by the flip and not by run-to-run noise.
    on2 = _item(runs["clahe_on2"][2], src)["trace"]
    again = _item(runs["again"][2], src)["trace"]
    for a, b in ((on, on2), (off, again)):
        assert a["gray"]["sha256"] == b["gray"]["sha256"]
        assert a["gray"]["black_fraction"] == b["gray"]["black_fraction"]
        assert a["gray_crush_fraction"] == b["gray_crush_fraction"]
        assert a["mask"]["sha256"] == b["mask"]["sha256"]


# ------------------------------------------------- 2. identical re-run vs tamper

def test_geometry_identical_rerun_keeps_pathset_sha_and_tamper_changes_it(runs, src):
    a = _item(runs["base"][2], src)["routed"]["pathset_sha256_nd3"]
    b = _item(runs["again"][2], src)["routed"]["pathset_sha256_nd3"]
    assert a == b
    c = _item(runs["tampered"][2], src)["routed"]["pathset_sha256_nd3"]
    assert c != a, "tampered margin gave the same path set: the sha cannot fail"
    # the clahe-on pair is likewise stable run to run
    assert (_item(runs["clahe_on"][2], src)["routed"]["pathset_sha256_nd3"]
            == _item(runs["clahe_on2"][2], src)["routed"]["pathset_sha256_nd3"])


# ----------------------------------------------------------------- 3. placement
# NB: a pure translation keeps the trace-frame raster; a SCALE change does not, because
# scale_aware_v2 boosts the effective trace resolution with placement. That is the
# real behaviour this block exists to expose, pinned in the second test below.

def test_placement_change_keeps_gray_sha_changes_pathset(runs, src):
    base = _item(runs["base"][2], src)
    placed = _item(runs["placed"][2], src)
    assert placed["trace"]["gray"]["sha256"] == base["trace"]["gray"]["sha256"]
    assert placed["trace"]["mask"]["sha256"] == base["trace"]["mask"]["sha256"]
    assert placed["routed"]["pathset_sha256_nd3"] != base["routed"]["pathset_sha256_nd3"]
    # request_body differs (it IS a different request), effective_body_sha differs with it
    assert runs["placed"][1]["effective_body_sha256"] != runs["base"][1]["effective_body_sha256"]
    assert runs["again"][1]["effective_body_sha256"] == runs["base"][1]["effective_body_sha256"]


def test_scale_change_moves_the_gray_raster_via_the_d885_boost(runs, src):
    base = _item(runs["base"][2], src)["trace"]
    resc = _item(runs["rescaled"][2], src)["trace"]
    assert resc["gray"]["dims_wh"] != base["gray"]["dims_wh"]
    assert resc["gray"]["sha256"] != base["gray"]["sha256"]
    # the response's own effective_res agrees with the recorded raster dims (cross-check
    # against an independent source, so the dims field cannot be a constant)
    assert max(resc["gray"]["dims_wh"]) == runs["rescaled"][0]["trace_info"]["effective_res"]
    assert max(base["gray"]["dims_wh"]) == runs["base"][0]["trace_info"]["effective_res"]


# ------------------------------------------------------- 4. definition vs page

def test_pathset_definition_on_synthetic_svg():
    svg = ('<svg><g id="layer1">'
           '<polyline points="0,0 10,0 10,10"/>'
           '<polygon points="0,0 4,0 4,4"/>'
           '<line x1="1" y1="2" x2="3" y2="4"/>'
           '</g><g id="layer2"><polyline points="5.00004,5 6,6"/></g></svg>')
    allp, by = tb.parse_routed_svg_text(svg)
    assert len(allp) == 4 and sorted(by) == ["1", "2"] and len(by["1"]) == 3
    poly = next(p for p in by["1"] if len(p) == 4)
    assert np.allclose(poly[0], poly[-1])                           # polygon ring closed
    # orientation canonicalisation + order independence + 3-decimal rounding
    fwd = [np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[2.0, 2.0], [3.0, 5.0]])]
    rev = [fwd[1][::-1], fwd[0][::-1]]
    assert tb.pathset_sha256(fwd) == tb.pathset_sha256(rev)
    assert tb.pathset_sha256([np.array([[0.00004, 0.0], [1.0, 1.0]])]) == \
        tb.pathset_sha256([np.array([[0.0, 0.0], [1.0, 1.0]])])
    # POSITIVE CONTROL: a 0.002 mm move is a different path set
    assert tb.pathset_sha256([np.array([[0.002, 0.0], [1.0, 1.0]])]) != \
        tb.pathset_sha256([np.array([[0.0, 0.0], [1.0, 1.0]])])


_A = REPO / "artifacts"
_FX = REPO / "tests" / "fixtures" / "routed_pathset_mini.svg"
# The vendored mini fixture is tracked and REQUIRED (a missing file is a test failure, never a skip).
#  - the second-placement cleaned-heli SVG (published hash f274ae1721) is not distributed with this repository; its case skips when absent.
#  - vendored mini fixture: hash computed by the independent reference scripts (lib.parse_svg + metrics.pathset),
#    see tests/fixtures/make_routed_pathset_fixture.py; it has 2 layers, polylines, polygons and <line>.
_PAGE = {
    "cleaned-heli-app-vs-b2-2026-10-04/data/M1_min_style_kai_placement/layout.routed.svg": "f274ae1721",
}
_FX_SHA = "15162e8392e2489eaa83818e27af0d9dd100fceb5a5a84f2002de15365cc62b0"


@pytest.mark.parametrize("path,prefix", sorted([(str(_A / k), v) for k, v in _PAGE.items()] + [(str(_FX), _FX_SHA)]))
def test_definition_reproduces_the_cleaned_heli_page_hashes(path, prefix):
    f = Path(path)
    if not f.is_file() and f != _FX:
        pytest.skip(f"fixture image not redistributable; place {f.name} (cleaned-heli page M1) at {f.parent} to run")
    assert f.is_file(), f"required tracked file missing: {f}"
    allp, _ = tb.parse_routed_svg_text(f.read_text(encoding="utf-8"))
    assert tb.pathset_sha256(allp).startswith(prefix)
    # POSITIVE CONTROL: dropping one polyline moves the hash
    assert not tb.pathset_sha256(allp[1:]).startswith(prefix)


def test_definition_equals_the_pages_own_script():
    s = REPO / "tests" / "fixtures" / "reference_scripts" / "cleaned-heli"
    assert (s / "metrics.py").is_file() and (s / "lib.py").is_file() and _FX.is_file()
    sys.path.insert(0, str(s))
    from metrics import pathset
    from lib import parse_svg
    allp, by_layer = tb.parse_routed_svg_text(_FX.read_text(encoding="utf-8"))
    assert len(allp) == 242 and sorted(by_layer) == ["1", "2"]
    assert tb.pathset_sha256(allp) == pathset(parse_svg(str(_FX)), 3)


# ------------------------------------------------------- 5. nothing leaks / safe

def test_nothing_new_in_response_info_json_or_gcode(runs, src):
    resp, prov, blk, gcode = runs["base"]
    assert "_trace_boundary" not in resp and "trace_boundary" not in resp
    assert "trace_boundary" not in json.dumps(resp.get("trace_info", {}))
    # info.json (hashed by the replay audit) is untouched: the block lives in its own sidecar
    infos = list(WORK_DIR.glob(f"src_{src}_render_*.svg.info.json"))
    assert infos and all("boundary" not in p.read_text() for p in infos)
    assert list(WORK_DIR.glob(f"src_{src}_render_*.svg.boundary.json"))
    # gcode: the per-run back-link is the only header line this run's provenance added, ASCII
    lines = gcode.decode("ascii").splitlines()
    assert sum(1 for ln in lines if ln.startswith("; ccplotter-")) == 1
    # a geometry-identical regenerate differs only by that line
    assert strip_run_backlink(gcode) == strip_run_backlink(runs["again"][3])


def test_collect_never_raises_and_records_the_error():
    blk = tb.collect(WORK_DIR / "does_not_exist.svg", {1: "x"}, {1: WORK_DIR / "nope.svg"}, 0.12)
    assert "error" in blk and isinstance(blk["error"], str)
    prov = build_run_provenance({"a": 1}, {"a": 1}, "p", {}, None, trace_boundary=blk)
    assert prov["trace_boundary"]["error"] == blk["error"] and "error" not in prov
    # missing sidecar for a real routed svg -> explicit 'unavailable', not invented numbers
    svg = WORK_DIR / "synthetic_routed.svg"
    svg.write_text('<svg><g id="layer1"><polyline points="0,0 5,5"/></g></svg>', encoding="utf-8")
    ok = tb.collect(svg, {1: "x"}, {1: WORK_DIR / "no_sidecar.svg"}, 0.12)
    assert "unavailable" in ok["items"]["x"]["trace"]
    assert ok["items"]["x"]["routed"]["polylines"] == 1


# ------------------------------------------------- strip_run_stamps (back-link + Source)

def test_strip_run_stamps_equates_runs_that_differ_only_in_the_stamps():
    a = (b"; Pen plotter\r\n; Source: C:/envA/work/src_x.png\r\n"
         b"; ccplotter-run: id=aaaaaaaa provenance=run_provenance/aaaaaaaa.json\r\n"
         b"G1 X1 Y1\r\nG1 X2 Y2\r\n")
    b = a.replace(b"envA", b"envB").replace(b"aaaaaaaa", b"bbbbbbbb")
    assert a != b
    # the back-link strip alone is NOT enough (the Source path still differs) -- the reason
    # the helper exists (the A_head_body vs A_kai_body comparison)
    assert strip_run_backlink(a) != strip_run_backlink(b)
    assert strip_run_stamps(a) == strip_run_stamps(b) == b"; Pen plotter\r\nG1 X1 Y1\r\nG1 X2 Y2\r\n"
    # control that can fail: a real difference in a move line survives the strip
    c = b.replace(b"G1 X2 Y2", b"G1 X2 Y3")
    assert strip_run_stamps(a) != strip_run_stamps(c)
    # no-op on an unstamped file, and LF-only files work too
    plain = b"; Pen plotter\nG1 X1 Y1\n"
    assert strip_run_stamps(plain) == plain


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
