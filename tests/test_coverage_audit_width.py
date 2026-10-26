"""coverage_audit's rasterised stroke width must match the commanded pen width
(laid_mm) as closely as the pixel grid allows.

cv2.polylines(..., thickness=th, lineType=cv2.LINE_8) does not draw `th` px
wide -- it draws the smallest ODD px count >= th+1. A naive coverage_audit
that picked th=int(laid_mm*ppmm) and reported raster_width_mm as th/ppmm
would treat th itself as the drawn width, getting both the pick and the
report wrong. This test rasterises a real horizontal stroke exactly the way
audit() does and measures its ACTUAL pixel width, so it cannot pass by
agreeing with the module's own (possibly wrong) internal arithmetic.

Same "scripted assertions, plain exit code" convention as
tests/test_rot_fit_fix.py / tests/test_preflight_scaling.py (no pytest
dependency in this repo). Run directly:
    python tests\\test_coverage_audit_width.py
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ccplotter import coverage_audit as ca  # noqa: E402


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def _measure_drawn_width_px(th, ppmm=40.0, length_mm=5.0):
    """Render a horizontal stroke exactly as coverage_audit.audit() does
    (same cv2 call, same LINE_8), and measure its real pixel width via a
    cross-section away from the line's endpoints (caps can flare width)."""
    W = int(length_mm * ppmm) + 40
    H = 60
    canvas = np.zeros((H, W), np.uint8)
    y = H // 2
    pts = np.array([[20, y], [W - 20, y]], dtype=np.int32)
    cv2.polylines(canvas, [pts], False, 255, thickness=th, lineType=cv2.LINE_8)
    col = canvas[:, W // 2]
    rows = np.nonzero(col > 0)[0]
    assert len(rows) > 0, "nothing drawn"
    return int(rows.max() - rows.min() + 1)


def _old_buggy_th(laid_mm, ppmm):
    """The naive commanded-thickness formula: th = max(1, int(laid_mm * ppmm)).
    Reimplemented here, not imported, as the reference the test compares
    against (the module no longer uses it)."""
    return max(1, int(laid_mm * ppmm))


# A case chosen to cleanly separate old vs new: laid_mm=0.225, ppmm=40 puts
# the target EXACTLY at 9.0 px, an odd integer that IS achievable -- but the
# old formula picks th=int(9.0)=9 (odd), which cv2 draws as 11px (odd th ->
# th+2), a 2px / 22% overshoot, when th=8 (even -> th+1=9) hits it exactly.
LAID_MM = 0.225
PPMM = 40.0
TARGET_PX = LAID_MM * PPMM  # 9.0


def _grid_strokes(n=111, pitch_mm=0.09, length_mm=20.0):
    """A simple horizontal-line fill grid, sized (20mm square, tight 0.09mm
    pitch) to clear audit()'s `fill.sum() >= 100` minimum-fill-area floor at
    ppmm=32 for BOTH the thin (0.08mm) and thick (0.16mm) laid_mm used by the
    positive-control test -- used only to give audit() enough ink to compute
    a fill region and a geometry_coverage."""
    return [[(0.0, i * pitch_mm), (length_mm, i * pitch_mm)] for i in range(n)]


def test_old_formula_overshoots_by_2px():
    """Demonstrates the defect: the naive th selection draws a stroke 2px
    (22%) wider than the achievable, exact target. Documents the defect; not a
    test of current code (current code no longer computes th this way)."""
    th_old = _old_buggy_th(LAID_MM, PPMM)
    drawn_old = _measure_drawn_width_px(th_old, ppmm=PPMM)
    assert th_old == 9, th_old
    assert drawn_old == 11, drawn_old
    assert approx(abs(drawn_old - TARGET_PX), 2.0), (drawn_old, TARGET_PX)


def test_new_th_selection_hits_target_width():
    """The fixed selection (_target_th_and_width) must choose a thickness
    whose ACTUALLY DRAWN width is the nearest achievable to the target -- in
    this case, exact. THIS IS THE TEST THAT FAILS AGAINST THE OLD CODE: the
    old th=int(laid_mm*ppmm)=9 draws 11px here (see
    test_old_formula_overshoots_by_2px), not the 9px this asserts."""
    th, drawn_px, raster_width_mm, err_mm = ca._target_th_and_width(LAID_MM, PPMM)
    assert th == 8, th
    measured = _measure_drawn_width_px(th, ppmm=PPMM)
    # the function's own bookkeeping must match what cv2 actually draws
    assert measured == drawn_px, (measured, drawn_px)
    assert measured == 9, measured
    assert abs(measured - TARGET_PX) <= 1.0, (measured, TARGET_PX)
    # raster_width_mm/err_mm must describe the REAL drawn width, not th/ppmm
    assert approx(raster_width_mm, 9 / PPMM), raster_width_mm
    assert approx(err_mm, 9 / PPMM - LAID_MM), err_mm


def test_target_th_and_width_never_off_by_more_than_1px():
    """General bound: for any laid_mm/ppmm, the nearest-achievable-odd-width
    selection cannot be more than 1.0px away from the target (odd integers
    are 2px apart, so the worst case is exactly the midpoint)."""
    cases = [
        (0.255, 32.0), (0.12, 32.0), (0.4, 24.0), (0.5, 16.0), (0.275, 40.0),
        (0.18, 32.0), (0.3, 32.0), (1.0, 32.0), (0.225, 40.0), (0.16, 32.0),
    ]
    for laid_mm, ppmm in cases:
        th, drawn_px, raster_width_mm, err_mm = ca._target_th_and_width(laid_mm, ppmm)
        target_px = laid_mm * ppmm
        assert abs(drawn_px - target_px) <= 1.0 + 1e-9, (laid_mm, ppmm, drawn_px, target_px)
        measured = _measure_drawn_width_px(th, ppmm=ppmm)
        assert measured == drawn_px, (laid_mm, ppmm, measured, drawn_px)


def test_drawn_width_probe_matches_measured_table():
    """Regression pin for the cv2 behaviour this whole fix depends on -- if a
    future OpenCV changes this, this test (not just the probe script) should
    catch it."""
    expected = {1: 1, 2: 3, 3: 5, 4: 5, 5: 7, 6: 7, 7: 9, 8: 9, 9: 11, 10: 11}
    for th, want in expected.items():
        got = ca._drawn_width_for_th(th)
        assert got == want, (th, got, want)


def test_audit_geometry_coverage_responds_to_deliberate_width_change():
    """Positive control: doubling the pen's laid_mm (hence the rasterised
    stroke width) on the SAME strokes must measurably raise geometry_coverage
    -- if this metric did not respond to width at all, a bug in width
    selection would be invisible to it."""
    strokes = _grid_strokes()
    thin = ca.audit(strokes, laid_mm=0.08, ppmm=32.0)
    thick = ca.audit(strokes, laid_mm=0.16, ppmm=32.0)
    assert thick["geometry_coverage"] > thin["geometry_coverage"] + 0.02, \
        (thin["geometry_coverage"], thick["geometry_coverage"])


def test_audit_geometry_coverage_null_control_same_width_same_result():
    """Null control: re-running audit() on identical inputs must not drift --
    isolates the positive-control result to the width parameter, not noise."""
    strokes = _grid_strokes()
    a = ca.audit(strokes, laid_mm=0.12, ppmm=32.0)
    b = ca.audit(strokes, laid_mm=0.12, ppmm=32.0)
    assert a["geometry_coverage"] == b["geometry_coverage"], (a, b)
    assert a["raster_width_mm"] == b["raster_width_mm"], (a, b)


_TESTS = [
    test_old_formula_overshoots_by_2px,
    test_new_th_selection_hits_target_width,
    test_target_th_and_width_never_off_by_more_than_1px,
    test_drawn_width_probe_matches_measured_table,
    test_audit_geometry_coverage_responds_to_deliberate_width_change,
    test_audit_geometry_coverage_null_control_same_width_same_result,
]


def _run_all():
    failed = 0
    for t in _TESTS:
        name = t.__name__
        try:
            t()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
            traceback.print_exc()
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {e}")
            traceback.print_exc()
    total = len(_TESTS)
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
