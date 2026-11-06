"""Tests for the EXPERIMENTAL opt-in `row_angle_rule` (2026-10-26; zero|a45|across|set) on core_fill rows / outline_rows.

Every claim is paired with a null that can fail:
  * absent == "zero" (byte-identical, no audit field)      <- a non-zero rule DOES differ (positive control)
  * a45 rows run at 45 deg                                  <- the zero rule's rows run at 0 deg (measured, same bars)
  * across follows the SHAPE (horizontal bar -> ~90, vertical bar -> ~0) <- the swapped shape swaps the angle
  * near-round region -> 45 deg fallback, counted           <- an elongated region is not a fallback
  * set is deterministic, only {0,45,90,135}, histogram sums to the region count
  * route refuses unknown rules / rules without rows modes  <- the same body with a valid rule succeeds
"""
import math
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home; pen/calibration never resolve from CWD
_test_home.use_temp_home("ccplotter_test_home_rowangle_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_rowangle_"))
sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import pytest
from PIL import Image

import ccplotter.routes.runs  # noqa: F401
from ccplotter.config import WORK_DIR
from ccplotter.core_fill import row_angle_for_region, ROW_ANGLE_RULES

MM = 0.0474
FILL_MM = 0.12
FILL_PX = FILL_MM / MM


def _pc(mm_per_px):
    from ccplotter.config import PX2MM
    return [1.0, 1.0, 0.0, mm_per_px / PX2MM, True]


def _trace(img, extra, tmp_path, name="t"):
    from ccplotter.trace_v3 import _trace_svg
    out = tmp_path / f"{name}.svg"
    opts = {"fill_px": FILL_PX, "fill_mm": FILL_MM, "threshold_mode": "otsu", "pocket_link": 0, "_place_cell": _pc(MM), **extra}
    _trace_svg(img, out, opts, debug_prefix=str(tmp_path / f"dbg_{name}"))
    from ccplotter.trace_v3 import _TRACE_AUDIT      # re-imported AFTER the call: the module rebinds it every run
    text = out.read_text(encoding="utf-8")
    polys = [[tuple(float(v) for v in t.split(",")) for t in ps.split()] for ps in re.findall(r'points="([^"]+)"', text)]
    return polys, dict(_TRACE_AUDIT)


def _is_closed(p):
    return len(p) >= 3 and abs(p[0][0] - p[-1][0]) < 0.6 and abs(p[0][1] - p[-1][1]) < 0.6


def _dominant_deg(polys):
    """length-weighted dominant direction (deg, mod 180, 5-deg bins) of the OPEN polylines (the rows)."""
    w = {}
    for p in polys:
        if _is_closed(p):
            continue
        for (x0, y0), (x1, y1) in zip(p, p[1:]):
            L = math.hypot(x1 - x0, y1 - y0)
            if L < 1e-6:
                continue
            a = int(round((math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0) / 5.0)) % 36 * 5
            w[a] = w.get(a, 0.0) + L
    return max(w, key=w.get)


def _bars(horizontal=True):
    img = np.full((500, 500), 255, np.uint8)
    if horizontal:
        cv2.rectangle(img, (60, 200), (440, 300), 0, -1)       # 380 x 100 px
    else:
        cv2.rectangle(img, (200, 60), (300, 440), 0, -1)
    return img


def _disc():
    img = np.full((500, 500), 255, np.uint8)
    cv2.circle(img, (250, 250), 90, 0, -1)
    return img


def _blobs():
    img = np.full((500, 500), 255, np.uint8)
    cv2.rectangle(img, (30, 40), (200, 100), 0, -1)
    cv2.rectangle(img, (260, 30), (320, 220), 0, -1)
    cv2.circle(img, (120, 330), 60, 0, -1)
    cv2.ellipse(img, (350, 380), (100, 35), 30, 0, 360, 0, -1)
    cv2.rectangle(img, (40, 440), (150, 480), 0, -1)
    return img


# --------------------------------------------------------------------------- unit: the rule function
def test_rule_function_each_rule():
    ys, xs = np.nonzero(np.ones((20, 200), bool))             # wide region, long axis horizontal
    xs, ys = xs.astype(float), ys.astype(float)
    assert row_angle_for_region("zero", xs, ys, 0.0) == (0.0, False)
    assert row_angle_for_region("a45", xs, ys, 0.0) == (45.0, False)
    assert row_angle_for_region("a135", xs, ys, 0.0) == (135.0, False)     # 2026-11-04: "/" on the preview at item rot 0
    a, fb = row_angle_for_region("across", xs, ys, 0.0)
    assert not fb and abs((a % 180.0) - 90.0) < 1.0           # short side of a horizontal bar is vertical
    a2, fb2 = row_angle_for_region("across", ys, xs, 0.0)     # transposed: tall bar -> rows horizontal
    assert not fb2 and min(a2 % 180.0, 180.0 - a2 % 180.0) < 1.0
    sq_y, sq_x = np.nonzero(np.ones((60, 60), bool))
    assert row_angle_for_region("across", sq_x.astype(float), sq_y.astype(float), 0.0) == (45.0, True)   # near-round fallback
    s1 = row_angle_for_region("set", xs, ys, 0.0); s2 = row_angle_for_region("set", xs.copy(), ys.copy(), 0.0)
    assert s1 == s2 and s1[0] in (0.0, 45.0, 90.0, 135.0)
    rng = np.random.default_rng(5)
    seen = {row_angle_for_region("set", xs + float(rng.uniform(0, 500)), ys + float(rng.uniform(0, 500)), 0.0)[0] for _ in range(80)}
    assert seen == {0.0, 45.0, 90.0, 135.0}                   # the hash really spreads over all four
    assert ROW_ANGLE_RULES == ("zero", "a45", "a135", "across", "set")
    from ccplotter.core_fill import ROW_ANGLE_DEFAULT
    assert ROW_ANGLE_DEFAULT == "a135"


# --------------------------------------------------------------------------- trace level
@pytest.mark.parametrize("cf", ["outline_rows", "rows"])
def test_zero_and_absent_identical_no_audit_field(tmp_path, cf):
    img = _blobs()
    base, au0 = _trace(img, {"core_fill": cf}, tmp_path, "b")
    zero, au1 = _trace(img, {"core_fill": cf, "row_angle_rule": "zero"}, tmp_path, "z")
    assert base == zero and len(base) > 8
    for au in (au0, au1):
        assert "row_angle_rule" not in au and "row_angle_hist" not in au and "row_angle_fallback_regions" not in au
    # positive control: a non-zero rule must change the output and add the audit fields
    alt, au2 = _trace(img, {"core_fill": cf, "row_angle_rule": "a45"}, tmp_path, "a")
    assert alt != base
    assert au2["row_angle_rule"] == "a45" and au2["row_angle_hist"]["45"] >= 1


def test_a45_changes_row_direction_from_0_to_45(tmp_path):
    img = _bars(True)
    z, _ = _trace(img, {"core_fill": "outline_rows"}, tmp_path, "z")
    a, au = _trace(img, {"core_fill": "outline_rows", "row_angle_rule": "a45"}, tmp_path, "a")
    assert _dominant_deg(z) == 0 and _dominant_deg(a) == 45
    assert sum(au["row_angle_hist"].values()) >= 1 and au["row_angle_hist"]["45"] == sum(au["row_angle_hist"].values())


def test_a135_rows_run_at_135_deg_in_the_trace_frame(tmp_path):
    """a135 == -45 deg, y down: dominant direction 135 (null: a45 on the SAME bars is 45; zero is 0)."""
    img = _bars(True)
    z, _ = _trace(img, {"core_fill": "outline_rows"}, tmp_path, "z")
    a45, _ = _trace(img, {"core_fill": "outline_rows", "row_angle_rule": "a45"}, tmp_path, "a")
    a, au = _trace(img, {"core_fill": "outline_rows", "row_angle_rule": "a135"}, tmp_path, "b")
    assert (_dominant_deg(z), _dominant_deg(a45), _dominant_deg(a)) == (0, 45, 135)
    assert au["row_angle_rule"] == "a135" and au["row_angle_hist"]["135"] == sum(au["row_angle_hist"].values()) >= 1
    assert au["row_angle_fallback_regions"] == 0
    # sense on the image: rows of a135 go from bottom-left to top-right (x increases while y DECREASES)
    p = max((q for q in a if not _is_closed(q)), key=len)
    assert (p[-1][0] - p[0][0]) * (p[-1][1] - p[0][1]) < 0 or (p[1][0] - p[0][0]) * (p[1][1] - p[0][1]) < 0


def test_across_follows_the_shape(tmp_path):
    h, auh = _trace(_bars(True), {"core_fill": "outline_rows", "row_angle_rule": "across"}, tmp_path, "h")
    v, auv = _trace(_bars(False), {"core_fill": "outline_rows", "row_angle_rule": "across"}, tmp_path, "v")
    assert _dominant_deg(h) == 90 and _dominant_deg(v) == 0     # swapped shape swaps the direction (null: not constant)
    assert auh["row_angle_fallback_regions"] == 0 and auh["row_angle_hist"]["90"] >= 1
    assert auv["row_angle_hist"]["0"] >= 1


def test_across_near_round_falls_back_to_45_and_is_counted(tmp_path):
    d, au = _trace(_disc(), {"core_fill": "outline_rows", "row_angle_rule": "across"}, tmp_path, "d")
    assert _dominant_deg(d) == 45 and au["row_angle_fallback_regions"] >= 1 and au["row_angle_hist"]["45"] >= 1
    # null: an elongated shape is not a fallback
    _, au2 = _trace(_bars(True), {"core_fill": "outline_rows", "row_angle_rule": "across"}, tmp_path, "e")
    assert au2["row_angle_fallback_regions"] == 0


def test_set_deterministic_and_only_four_angles(tmp_path):
    img = _blobs()
    a, au = _trace(img, {"core_fill": "outline_rows", "row_angle_rule": "set"}, tmp_path, "s1")
    b, _ = _trace(img, {"core_fill": "outline_rows", "row_angle_rule": "set"}, tmp_path, "s2")
    assert a == b and sum(au["row_angle_hist"].values()) >= 4
    assert set(au["row_angle_hist"]) == {"0", "45", "90", "135"} and au["row_angle_fallback_regions"] == 0
    z, _ = _trace(img, {"core_fill": "outline_rows"}, tmp_path, "z")
    assert a != z


# --------------------------------------------------------------------------- route
def _make_photo(path: Path, w=900, h=700):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = (127 + 90 * np.sin(xx / 11.0) * np.cos(yy / 17.0)).clip(0, 255).astype(np.uint8)
    img = np.stack([base, np.roll(base, 5, axis=0), np.roll(base, -5, axis=1)], axis=-1)
    Image.fromarray(img, mode="RGB").save(path, quality=92)


def test_route_validation_and_zero_identity():
    from ccplotter.routes.core import app
    from ccplotter.cal import load_pen_profiles
    did = "rowangle_route"
    _make_photo(WORK_DIR / f"src_{did}.jpg")

    def body(**extra):
        b = {"job_type": "gcode", "style": "tracev3", "debug": True, "paper": "120x120", "margin": 10.0, "resolution": 600,
             "pen_id": load_pen_profiles().get("active"), "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}]}
        b.update(extra); return b

    strip = lambda b: b"\n".join(l for l in b.split(b"\n") if not l.startswith(b"; ccplotter-run:"))
    c = app.test_client()
    assert c.post("/api/process", json=body(core_fill="outline_rows", row_angle_rule="bogus")).status_code == 400
    assert c.post("/api/process", json=body(row_angle_rule="a45")).status_code == 400            # rings: rule is meaningless
    gc = lambda: strip((WORK_DIR / "layout.gcode").read_bytes())
    # 2026-11-04 default flip: ABSENT key + outline_rows == explicit a135; explicit "zero" is the old 0-deg output
    assert c.post("/api/process", json=body(core_fill="outline_rows")).status_code == 200
    g_base = gc()
    assert c.post("/api/process", json=body(core_fill="outline_rows", row_angle_rule="a135")).status_code == 200
    assert gc() == g_base                                                                          # absent == a135, byte for byte
    assert c.post("/api/process", json=body(core_fill="outline_rows", row_angle_rule="")).status_code == 200
    assert gc() == g_base                                                                          # empty == absent
    assert c.post("/api/process", json=body(core_fill="outline_rows", row_angle_rule="zero")).status_code == 200
    g_zero = gc()
    assert g_zero != g_base                                                                        # positive control: the flip moved the gcode
    r = c.post("/api/process", json=body(core_fill="outline_rows", row_angle_rule="a45"))
    assert r.status_code == 200
    assert gc() not in (g_base, g_zero)                                                            # a45 is a third, different output
    # rings / unset core_fill: no default, no change (rings job unaffected by the flip)
    assert c.post("/api/process", json=body()).status_code == 200
    g_rings = gc()
    assert c.post("/api/process", json=body(row_angle_rule="zero")).status_code == 200
    assert gc() == g_rings
    (WORK_DIR / "layout.gcode").read_bytes().decode("ascii")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
