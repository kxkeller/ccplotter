"""Tests for the opt-in core_fill="outline_rows": ONE outline ring per boundary, then
back-and-forth rows over everything inside it (no inner rings, no size rule).

Each claim is paired with a null that can fail:
  * 1 closed ring per boundary under outline_rows   <- the SAME shapes under `rings` give many
  * rows reach the ring (no seam)                    <- first-row-to-ring distance is measured
  * default/rings/rows untouched                     <- outline_rows output differs (identity can fail)
  * provenance records the new mode                  <- a rows run records no outline fields
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
_test_home.use_temp_home("ccplotter_test_home_outlinerows_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_outlinerows_"))
sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import pytest
from PIL import Image

import ccplotter.routes.runs  # noqa: F401 (registers /api/runs on the shared app)
from ccplotter.config import WORK_DIR
from ccplotter.runs import _runs_load, load_run_provenance

MM = 0.0474
FILL_MM = 0.12
FILL_PX = FILL_MM / MM
PINNED_DISC_SHA = "febc04a0e8471e8e27a6e6789d3e2a0d88e2096c20bb10d12e0ed1ee5819d9a2"


def _pc(mm_per_px):
    from ccplotter.config import PX2MM
    return [1.0, 1.0, 0.0, mm_per_px / PX2MM, True]


def _trace(img, extra, tmp_path, name="t", with_audit=False):
    from ccplotter.trace_v3 import _trace_svg
    out = tmp_path / f"{name}.svg"
    opts = {"fill_px": FILL_PX, "fill_mm": FILL_MM, "threshold_mode": "otsu", "pocket_link": 0,
            "_place_cell": _pc(MM), **extra}
    _trace_svg(img, out, opts, debug_prefix=str(tmp_path / f"dbg_{name}"))
    text = out.read_text(encoding="utf-8")
    polys = [[tuple(float(v) for v in t.split(",")) for t in ps.split()]
             for ps in re.findall(r'points="([^"]+)"', text)]
    if with_audit:
        from ccplotter.trace_v3 import _TRACE_AUDIT
        return polys, dict(_TRACE_AUDIT)
    return polys


def _is_closed(p):
    return len(p) >= 3 and abs(p[0][0] - p[-1][0]) < 0.6 and abs(p[0][1] - p[-1][1]) < 0.6


def _disc():
    img = np.full((500, 500), 255, np.uint8)
    cv2.circle(img, (250, 250), int(4.0 / MM), 0, -1)       # r ~ 84 px = 4 mm
    return img


def _disc_with_hole():
    img = _disc()
    cv2.circle(img, (250, 250), int(1.5 / MM), 255, -1)     # hole r 1.5 mm
    return img


def _split(polys):
    closed = [p for p in polys if _is_closed(p)]
    openp = [p for p in polys if not _is_closed(p)]
    return closed, openp


# --------------------------------------------------------------------------- positive control

def test_filled_disc_one_ring_rest_rows_and_rings_mode_gives_many(tmp_path):
    img = _disc()
    o_closed, o_open = _split(_trace(img, {"core_fill": "outline_rows"}, tmp_path, "o"))
    r_closed, r_open = _split(_trace(img, {"core_fill": "rings"}, tmp_path, "r"))
    assert len(o_closed) == 1, len(o_closed)
    assert len(o_open) >= 1 and sum(len(p) for p in o_open) >= 60   # interior = open serpentine rows
    # null: the SAME disc under rings has many nested rings, so the 1-ring assertion can fail
    assert len(r_closed) > 10 and len(r_closed) > len(o_closed)


def test_disc_with_hole_one_ring_per_boundary_and_rings_mode_gives_more(tmp_path):
    img = _disc_with_hole()
    o_closed, o_open = _split(_trace(img, {"core_fill": "outline_rows"}, tmp_path, "o"))
    r_closed, _ = _split(_trace(img, {"core_fill": "rings"}, tmp_path, "r"))
    assert len(o_closed) == 2, len(o_closed)     # outer boundary + hole boundary, 1 each
    assert len(o_open) >= 1 and sum(len(p) for p in o_open) >= 40
    assert len(r_closed) > 2 and len(r_closed) > len(o_closed)
    # the two rings are the two boundaries: one around the centre far out, one close to the hole
    cx = cy = 250.0
    radii = sorted(float(np.median(np.hypot(np.array(p)[:, 0] - cx, np.array(p)[:, 1] - cy))) for p in o_closed)
    assert abs(radii[0] - 1.5 / MM) < 2.5 * FILL_PX and abs(radii[1] - 4.0 / MM) < 2.5 * FILL_PX


# --------------------------------------------------------------------------- no seam

def test_rows_reach_the_ring_without_a_seam(tmp_path):
    """First row line sits one pitch inside the ring (ring at dt=0.5p, rows start at dt=1.5p): the
    open-row end points are ~FILL_PX from the ring polyline; a seam would be >> that."""
    from shapely.geometry import LineString, Point
    img = _disc()
    closed, openp = _split(_trace(img, {"core_fill": "outline_rows"}, tmp_path, "o"))
    ring = LineString(closed[0])
    ends = [pt for p in openp for pt in p]   # every vertex = a row end
    d = np.array([ring.distance(Point(e)) for e in ends])
    assert len(d) >= 10
    assert d.min() > 0.5 * FILL_PX                         # rows do not duplicate the ring
    assert d.max() < 1.6 * FILL_PX, (d.max() / FILL_PX)    # but every row end is within ~one pitch of it
    # null: with a doubled pitch the same measure exceeds the limit, so the bound can fail
    assert (d.max() * 2.0) > 1.6 * FILL_PX


def test_rows_run_at_one_global_angle_and_stay_in_mask(tmp_path):
    img = _disc_with_hole()
    mask = (img < 128).astype(np.uint8)
    polys, au = _trace(img, {"core_fill": "outline_rows"}, tmp_path, "o", with_audit=True)
    closed, openp = _split(polys)
    seg = np.concatenate([np.diff(np.array(p), axis=0) for p in openp])
    ln = np.hypot(seg[:, 0], seg[:, 1])
    long_ = seg[ln > 3 * FILL_PX]
    assert len(long_) >= 20
    ang = np.degrees(np.arctan2(long_[:, 1], long_[:, 0])) % 180.0
    assert (np.minimum(ang, 180.0 - ang) < 1.0).mean() > 0.9
    ok = cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0
    canvas = np.zeros(mask.shape, np.uint8)
    for p in polys:
        cv2.polylines(canvas, [np.round(np.array(p)).astype(np.int32).reshape(-1, 1, 2)], _is_closed(p), 255, 1)
    assert int(((canvas > 0) & ~ok).sum()) == 0
    assert au["core_fill"] == "outline_rows" and au["outline_rings"] == 1 and au["n_outline_rings"] == 2


# --------------------------------------------------------------------------- other modes untouched

def test_other_modes_unchanged_and_audit_keys_do_not_leak(tmp_path):
    img = _disc()
    base = _trace(img, {}, tmp_path, "base")
    assert base == _trace(img, {"core_fill": "rings"}, tmp_path, "rings")
    o = _trace(img, {"core_fill": "outline_rows"}, tmp_path, "o")
    assert o != base                                       # null: the identity check can fail
    _, au_rows = _trace(img, {"core_fill": "rows"}, tmp_path, "rows", with_audit=True)
    assert au_rows["core_fill"] == "rows" and "outline_rings" not in au_rows
    # rows is repeatable and still differs from outline_rows (distinct modes)
    assert _trace(img, {"core_fill": "rows"}, tmp_path, "rows2") != o


def test_core_fill_outline_rows_not_confused_with_core_r(tmp_path):
    """core_r_mm is irrelevant to outline_rows: same output with and without it."""
    img = _disc()
    a = _trace(img, {"core_fill": "outline_rows"}, tmp_path, "a")
    b = _trace(img, {"core_fill": "outline_rows", "core_r_mm": 2.5}, tmp_path, "b")
    assert a == b


# --------------------------------------------------------------------------- provenance (route)

def _make_photo(path: Path, w=900, h=700):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = (127 + 90 * np.sin(xx / 11.0) * np.cos(yy / 17.0)).clip(0, 255).astype(np.uint8)
    img = np.stack([base, np.roll(base, 5, axis=0), np.roll(base, -5, axis=1)], axis=-1)
    Image.fromarray(img, mode="RGB").save(path, quality=92)


def _prov_for(did, **extra):
    from ccplotter.routes.core import app
    from ccplotter.cal import load_pen_profiles
    _make_photo(WORK_DIR / f"src_{did}.jpg")
    b = {"job_type": "gcode", "style": "tracev3", "debug": True, "paper": "120x120",
         "margin": 10.0, "resolution": 600, "pen_id": load_pen_profiles().get("active"),
         "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}]}
    b.update(extra)
    resp = app.test_client().post("/api/process", json=b)
    assert resp.status_code == 200, resp.get_json()
    rec = [r for r in _runs_load() if r.get("type") == "gcode"][0]
    return load_run_provenance(rec["id"]), (WORK_DIR / "layout.gcode").read_bytes()


def test_provenance_records_outline_rows_mode():
    p, g_o = _prov_for("orows_prov", core_fill="outline_rows")
    assert p["request_body"]["core_fill"] == "outline_rows"
    assert p["effective_body"]["core_fill"] == "outline_rows"
    assert p["effective_body"]["core_outline_rings"] == 1
    assert "core_r_mm" not in p["effective_body"]          # not a rows-size-rule mode
    # null: a rows run and a rings run carry no outline fields (the check can fail)
    p2, g_r = _prov_for("orows_prov", core_fill="rows")
    assert p2["effective_body"]["core_fill"] == "rows" and "core_outline_rings" not in p2["effective_body"]
    p3, g_d = _prov_for("orows_prov")
    assert "core_outline_rings" not in p3["effective_body"]
    strip = lambda b: b"\n".join(l for l in b.split(b"\n") if not l.startswith(b"; ccplotter-run:"))
    assert strip(g_o) != strip(g_d)                        # the mode really changes the gcode
    g_o.decode("ascii")


# --------------------------------------------------------------------------- inset is refused, never recorded

def test_inset_ignored_by_trace_and_never_recorded_and_inset0_identity(tmp_path):
    import hashlib
    img = _disc()
    base, au0 = _trace(img, {"core_fill": "outline_rows"}, tmp_path, "i0", with_audit=True)
    zero = _trace(img, {"core_fill": "outline_rows", "edge_inset_mm": 0.0}, tmp_path, "i00")
    assert base == zero                                           # inset 0 == key absent
    assert hashlib.sha256(repr(base).encode()).hexdigest() == PINNED_DISC_SHA
    ins, au = _trace(img, {"core_fill": "outline_rows", "edge_inset_mm": 0.3}, tmp_path, "i3", with_audit=True)
    assert ins == base and "edge_inset_mm" not in au and "edge_inset_px" not in au
    # positive control: under plain `rings` the same inset DOES move the output and is recorded
    r0 = _trace(img, {"core_fill": "rings"}, tmp_path, "r0")
    r3, aur = _trace(img, {"core_fill": "rings", "edge_inset_mm": 0.3}, tmp_path, "r3", with_audit=True)
    assert r3 != r0 and aur.get("edge_inset_mm") == 0.3


def test_route_refuses_outline_rows_with_inset_and_writes_no_gcode():
    from ccplotter.routes.core import app
    from ccplotter.cal import load_pen_profiles
    did = "orows_inset"
    _make_photo(WORK_DIR / f"src_{did}.jpg")
    def body(**extra):
        b = {"job_type": "gcode", "style": "tracev3", "debug": True, "paper": "120x120", "margin": 10.0,
             "resolution": 600, "pen_id": load_pen_profiles().get("active"),
             "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}]}
        b.update(extra); return b
    n0 = len([r for r in _runs_load() if r.get("type") == "gcode"])
    g = WORK_DIR / "layout.gcode"
    m0 = g.stat().st_mtime_ns if g.exists() else None
    r = app.test_client().post("/api/process", json=body(core_fill="outline_rows", edge_inset_mm=0.2))
    assert r.status_code == 400
    err = r.get_json()["error"]
    assert "outline_rows" in err and "edge_inset_mm" in err
    assert len([x for x in _runs_load() if x.get("type") == "gcode"]) == n0
    assert (g.stat().st_mtime_ns if g.exists() else None) == m0
    # null: the same request without the inset succeeds, and rows + inset is NOT refused
    assert app.test_client().post("/api/process", json=body(core_fill="outline_rows")).status_code == 200
    assert app.test_client().post("/api/process", json=body(core_fill="rows", edge_inset_mm=0.2)).status_code == 200


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
