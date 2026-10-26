"""Tests for the opt-in core_fill="rows": rings as usual, then back-and-forth
rows in the core of every LARGE region whatever its shape (curved cores included).

Every claim has a positive control paired with a null:
  * large curved cores get rows            <- vs hybrid, which sends the SAME curved core back to rings
  * small regions are identical to default <- the size rule is shown to bite on a mixed image
  * default byte-identity                  <- negative control: rows DOES change a big-disc image
  * mask containment                       <- negative control: the checker flags a deliberately outside stroke
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
_test_home.use_temp_home("ccplotter_test_home_corerows_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_corerows_"))
sys.path.insert(0, str(REPO))

import cv2
import numpy as np

from ccplotter.config import ROWS_CORE_R_MM_DEFAULT

MM = 0.0474
FILL_MM = 0.12
FILL_PX = FILL_MM / MM
CORE_R_PX = ROWS_CORE_R_MM_DEFAULT / MM


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


def _img_bigdisc():
    """One big disc (r 4 mm) + one small disc (r 0.5 mm) + a curved thick arc (width 1.7 mm)."""
    img = np.full((600, 900), 255, np.uint8)
    cv2.circle(img, (150, 150), int(4.0 / MM), 0, -1)
    cv2.circle(img, (700, 100), int(0.5 / MM), 0, -1)
    return img


def _img_curved():
    """A thick curved band (a 160-degree arc, width 1.9 mm, radius 6 mm): inscribed radius ~0.95 mm
    (< 1.0 so the hybrid's own size rule would not even start) and strongly curved: a hybrid core
    fails the rect-fill test, a straight-row fill of it would chop it."""
    img = np.full((700, 700), 255, np.uint8)
    cv2.ellipse(img, (350, 450), (int(6.0 / MM),) * 2, 0, 200, 340, 0, int(1.9 / MM))
    return img


def _img_big_curved():
    """A big curved fat band: width 6 mm (inscribed radius 3 mm), arc radius 10 mm."""
    img = np.full((900, 900), 255, np.uint8)
    cv2.ellipse(img, (450, 620), (int(10.0 / MM),) * 2, 0, 215, 325, 0, int(6.0 / MM))
    return img


def _mask_of(img):
    from ccplotter.trace_v3 import _trace_svg  # noqa: F401 (import side effects only)
    return (img < 128).astype(np.uint8)


def _outside_px(polys, mask, pen_px=1):
    """Number of stroke pixels (1-px lines) outside the mask dilated by 1.5 px."""
    ok = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1) > 0
    canvas = np.zeros(mask.shape, np.uint8)
    for p in polys:
        cv2.polylines(canvas, [np.round(np.array(p)).astype(np.int32).reshape(-1, 1, 2)], _is_closed(p), 255, pen_px)
    return int(((canvas > 0) & ~ok).sum())


# --------------------------------------------------------------------------- byte identity

def test_default_and_rings_value_are_byte_identical_and_hybrid_untouched(tmp_path):
    img = _img_bigdisc()
    base = _trace(img, {}, tmp_path, "base")
    again = _trace(img, {"core_fill": "rings"}, tmp_path, "again")
    assert base == again and len(base) > 20
    # negative control: rows on the SAME image does change the output (the identity check can fail)
    rows = _trace(img, {"core_fill": "rows"}, tmp_path, "rows")
    assert rows != base


def test_rows_with_no_large_region_equals_default_exactly(tmp_path):
    """Only small specks/dots (all origins below the size rule): output == default, polyline for polyline."""
    img = np.full((500, 900), 255, np.uint8)
    x = 60
    for r in (0.2, 0.35, 0.5, 0.7, 0.9):
        cv2.circle(img, (x + int(r / MM), 100), int(r / MM), 0, -1)
        cv2.line(img, (x, 300), (x + 150, 300), 0, max(1, int(2 * r / MM) // 2))
        x += int(2 * r / MM) + 60
    base = _trace(img, {}, tmp_path, "base")
    rows = _trace(img, {"core_fill": "rows"}, tmp_path, "rows")
    assert base == rows and len(base) > 10


def test_small_regions_identical_to_default_in_mixed_image(tmp_path):
    """Big disc + small disc: every default polyline lying fully inside the small disc's bbox
    must appear unchanged in the rows output (the small disc keeps its rings)."""
    img = _img_bigdisc()
    base = _trace(img, {}, tmp_path, "base")
    rows = _trace(img, {"core_fill": "rows"}, tmp_path, "rows")
    box = (600, 0, 800, 200)   # small disc at (700,100), r ~10px

    def inside(p):
        return all(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in p)
    small_base = [p for p in base if inside(p)]
    small_rows = [p for p in rows if inside(p)]
    assert small_base, "control failed: no default rings found in the small-disc window"
    assert small_base == small_rows
    # and the big disc DID change (size rule bites): rows has fewer closed rings there
    big_closed = lambda ps: sum(1 for p in ps if _is_closed(p) and p[0][0] < 400)
    assert big_closed(rows) < big_closed(base)


# --------------------------------------------------------------------------- shape / containment

def test_large_curved_core_gets_rows_hybrid_does_not(tmp_path):
    img = _img_big_curved()
    polys_h, au_h = _trace(img, {"core_fill": "hybrid", "core_r_mm": 1.0}, tmp_path, "hyb", with_audit=True)
    polys_r, au_r = _trace(img, {"core_fill": "rows", "core_r_mm": 1.0}, tmp_path, "rows", with_audit=True)
    assert au_r["core_fill"] == "rows" and au_r["rows_angle_deg"] == 0.0
    assert au_r["n_core_regions"] >= 1 and au_r["n_core_rows"] >= 5, au_r
    assert au_r["n_core_curved_ring_levels"] == 0          # no rect-fill test in rows mode
    # positive control: hybrid sends this curved core back to rings (that is the defect this mode fixes)
    assert au_h["n_core_curved_ring_levels"] > 0 and au_h["n_core_rows"] < au_r["n_core_rows"]
    # the core is real straight rows: open polylines with long, mostly-horizontal segments
    open_ = [p for p in polys_r if not _is_closed(p)]
    assert open_
    seg = np.concatenate([np.diff(np.array(p), axis=0) for p in open_])
    ln = np.hypot(seg[:, 0], seg[:, 1])
    long_ = seg[ln > 3 * FILL_PX]
    assert len(long_) >= 20
    ang = np.degrees(np.arctan2(long_[:, 1], long_[:, 0])) % 180.0
    assert (np.minimum(ang, 180.0 - ang) < 1.0).mean() > 0.9      # ONE global angle (0 deg)


def test_rows_stay_inside_mask_and_checker_can_fail(tmp_path):
    for img in (_img_bigdisc(), _img_big_curved()):
        mask = _mask_of(img)
        polys = _trace(img, {"core_fill": "rows"}, tmp_path, "rows")
        assert _outside_px(polys, mask) == 0
        # negative control: a stroke a few px outside the mask is flagged
        ys, xs = np.nonzero(mask)
        bad = polys + [[(float(xs.min()) - 6, float(ys.min()) - 6), (float(xs.min()) - 6, float(ys.min()) + 30)]]
        assert _outside_px(bad, mask) > 0


def test_rows_cover_core_no_pinhole(tmp_path):
    """Plot-scale check on the disc: stamp strokes at the pitch width; the big disc's centre is inked
    (rows reach the middle) and uncovered share of the interior is small."""
    img = _img_bigdisc()
    mask = _mask_of(img)
    polys = _trace(img, {"core_fill": "rows"}, tmp_path, "rows")
    canvas = np.zeros(mask.shape, np.uint8)
    for p in polys:
        cv2.polylines(canvas, [np.round(np.array(p)).astype(np.int32).reshape(-1, 1, 2)], _is_closed(p), 255,
                      max(1, int(round(FILL_PX))))
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    inner = dt >= 2 * FILL_PX
    assert inner.sum() > 1000
    assert (inner & (canvas == 0)).sum() / inner.sum() < 0.03
    assert canvas[150, 150] > 0


def test_core_rows_angle_kwarg_default_unchanged():
    """core_rows(angle_deg=None) is the principal-axis behaviour hybrid uses; angle_deg=0 is axis-aligned."""
    from shapely.geometry import Polygon
    from ccplotter.core_fill import core_rows
    poly = Polygon([(0, 0), (40, 20), (38, 26), (-2, 6)])         # a tilted bar
    from shapely.geometry import Point
    gy, gx = np.mgrid[-4:32, -4:44]
    keep = np.array([poly.contains(Point(x, y)) for x, y in zip(gx.ravel(), gy.ravel())])
    xs, ys = gx.ravel()[keep].astype(float), gy.ravel()[keep].astype(float)
    a, _ = core_rows(poly, ys, xs, 3.0)
    b, _ = core_rows(poly, ys, xs, 3.0, angle_deg=0.0)
    ang = lambda pl: abs(np.degrees(np.arctan2(pl[0][1][1] - pl[0][0][1], pl[0][1][0] - pl[0][0][0])))
    assert abs(b[0][1][1] - b[0][0][1]) < 1e-9          # horizontal rows
    assert ang(a) > 5.0                                 # principal axis follows the tilt
