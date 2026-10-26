"""Tests for the _preflight mm/px scaling.

_preflight must report mm per source pixel using the same fit as composite.py
(the 0.9 fit factor and the binding axis), not a width-only formula.

This file follows the repo's "scripted assertions, plain exit code" convention
(tests/golden_tests/replay_harness.py is an end-to-end HTTP replay harness
against a live dev server -- not suited to a pure-function unit check): run
directly with
    .venv\\Scripts\\python.exe tests\\test_preflight_scaling.py
It also uses plain `test_*` functions + `assert`, so it is picked up by
pytest for free if pytest is ever added to this project.

IMPORTANT: sets CCPLOTTER_WORK_DIR to an isolated temp directory BEFORE
importing anything from ccplotter, so this never touches the real work/
directory or a live dev server (none is started; ccplotter.routes.app is
exercised only via Flask's in-process test_client(), which binds no port).

SKIP discipline: the two BENCH_TRAM-gated tests raise SkipTest on a missing
fixture rather than printing SKIP and returning (which _run_all() would count
as an ordinary PASS, so a green "N/N passed" could mean fewer tests ran).
_run_all() tracks skips as a THIRD, distinct outcome, prints them LOUDLY in
the summary line (never folded into the passed count), and returns non-zero
if there are ANY -- a green run can only mean every test asserted something.
Same convention as tests/test_gapaudit_coverage_scaling.py.
"""
import os
import sys
import tempfile
from pathlib import Path


try:
    import pytest as _pytest
    _SkipBase = _pytest.skip.Exception     # so a raised SkipTest is a pytest skip, not a failure
except ImportError:                        # direct run without pytest installed
    _SkipBase = Exception


class SkipTest(_SkipBase):
    """Raised by a test when an optional fixture is missing. Tracked as a
    distinct SKIPPED outcome by _run_all() -- never silently counted as a
    PASS."""


REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home; pen/calibration never resolve from CWD
_test_home.use_temp_home("ccplotter_test_home_preflight_")
_TMP_WORK = tempfile.mkdtemp(prefix="ccplotter_test_work_")
os.environ["CCPLOTTER_WORK_DIR"] = _TMP_WORK
sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image

from ccplotter.trace_v3 import _preflight


def _make_gray(path: Path, w: int, h: int):
    """A synthetic grayscale raster with SOME ink structure (a centered
    rectangle) so _preflight's Otsu/skeletonize/distance-transform machinery
    has real strokes to measure -- mm/px correctness does not depend on
    content, but a blank image would make finest_stroke_mm/finest_gap_mm
    trivially zero for every case, which would mask a broken pipeline."""
    img = np.full((h, w), 255, dtype=np.uint8)
    x0, y0 = w // 4, h // 4
    x1, y1 = w - w // 4, h - h // 4
    img[y0:y1, x0:x1] = 0
    img[y0 + max(1, h // 20):y1 - max(1, h // 20),
        x0 + max(1, w // 20):x1 - max(1, w // 20)] = 255
    Image.fromarray(img).save(path)
    return path


WORK = Path(_TMP_WORK)


def test_height_limited_branch_uses_height_not_width():
    """nw=100 (usable_w/nw=0.8) vs nh=200 (usable_h/nh=0.4) on a 100x100mm/
    margin10 page -- height is the binding dimension. A width-only formula
    (drawable_mm/nw, drawable_mm=usable_w only) would give 0.8*... with
    no 0.9 factor at all; the correct formula must pick the SMALLER (height)
    ratio and apply 0.9."""
    src = _make_gray(WORK / "src_test_h.png", w=100, h=200)
    r = _preflight(src, paper_w_mm=100.0, paper_h_mm=100.0, margin_mm=10.0)
    usable_w, usable_h = 80.0, 80.0
    expected = round(min(usable_w / 100, usable_h / 200) * 0.9, 4)
    assert expected == round(0.4 * 0.9, 4) == 0.36
    assert r["mm_per_px"] == expected, r


def test_width_limited_branch_uses_width_not_height():
    """Same page, transposed image (nw=200, nh=100) -- now width binds."""
    src = _make_gray(WORK / "src_test_w.png", w=200, h=100)
    r = _preflight(src, paper_w_mm=100.0, paper_h_mm=100.0, margin_mm=10.0)
    usable_w, usable_h = 80.0, 80.0
    expected = round(min(usable_w / 200, usable_h / 100) * 0.9, 4)
    assert expected == round(0.4 * 0.9, 4) == 0.36
    assert r["mm_per_px"] == expected, r


def test_non_square_paper_selects_correct_branch_both_ways():
    """paper 140x100mm, margin 10 -> usable 120x80. A wide image (200x100)
    is width-limited (120/200=0.6 < 80/100=0.8); a tall image (100x200,
    same pixel counts swapped) is height-limited (120/100=1.2 > 80/200=0.4).
    This is the case the old width-only formula got wrong for one of the
    two orientations while accidentally looking plausible for the other."""
    wide = _make_gray(WORK / "src_test_wide.png", w=200, h=100)
    r_wide = _preflight(wide, paper_w_mm=140.0, paper_h_mm=100.0, margin_mm=10.0)
    assert r_wide["mm_per_px"] == round(min(120 / 200, 80 / 100) * 0.9, 4) == round(0.6 * 0.9, 4)

    tall = _make_gray(WORK / "src_test_tall.png", w=100, h=200)
    r_tall = _preflight(tall, paper_w_mm=140.0, paper_h_mm=100.0, margin_mm=10.0)
    assert r_tall["mm_per_px"] == round(min(120 / 100, 80 / 200) * 0.9, 4) == round(0.4 * 0.9, 4)


def test_09_fit_factor_present():
    """Square image, square page, zero margin -- both dims give the same
    ratio, so the ONLY thing separating the correct mm/px from the naive
    (no-0.9) usable/nw is the 0.9 factor itself."""
    src = _make_gray(WORK / "src_test_sq.png", w=100, h=100)
    r = _preflight(src, paper_w_mm=100.0, paper_h_mm=100.0, margin_mm=0.0)
    naive = 100.0 / 100        # pre-fix width-only, no-0.9 model
    assert r["mm_per_px"] == round(naive * 0.9, 4) == 0.9
    assert r["mm_per_px"] < naive, "0.9 fit factor must shrink mm/px vs. the naive model"


def test_paper_h_mm_defaults_to_square():
    """Omitting paper_h_mm must behave exactly like passing paper_h_mm ==
    paper_w_mm (square page) -- the documented default for old-style callers
    that only ever knew about paper_w_mm."""
    src = _make_gray(WORK / "src_test_default.png", w=100, h=300)
    r_omitted = _preflight(src, paper_w_mm=120.0, margin_mm=10.0)
    r_explicit_square = _preflight(src, paper_w_mm=120.0, paper_h_mm=120.0, margin_mm=10.0)
    assert r_omitted["mm_per_px"] == r_explicit_square["mm_per_px"]
    # and it must NOT equal what a wildly different height would give (sanity
    # that the default is actually being exercised, not coincidentally equal)
    r_explicit_tall = _preflight(src, paper_w_mm=120.0, paper_h_mm=400.0, margin_mm=10.0)
    assert r_omitted["mm_per_px"] != r_explicit_tall["mm_per_px"]


def test_api_preflight_route_forwards_paper_height():
    """routes/pen.py api_preflight must not discard the paper's height
    component. Drives the real Flask route in-process (test_client(), no
    port bound) with a custom rectangular paper string ('80x40', the same
    'WxH' shape the front end sends for a custom paper -- index.html
    ~3377/3380) on a strongly height-limited image, and checks the response
    matches calling _preflight directly with BOTH dimensions -- which is
    only possible if the route parsed and forwarded paper_h_mm."""
    from ccplotter.routes.pen import app

    did = "routetest1"
    src = _make_gray(WORK / f"src_{did}.png", w=60, h=600)  # very tall -> strongly height-limited
    client = app.test_client()
    resp = client.post("/api/preflight", json={"id": did, "paper": "80x40", "margin": 10})
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body["ok"] is True

    direct = _preflight(src, paper_w_mm=80.0, paper_h_mm=40.0, margin_mm=10.0,
                        deposit_mm=body["deposit_mm"])
    assert body["mm_per_px"] == direct["mm_per_px"], (body["mm_per_px"], direct["mm_per_px"])

    # Guard against the old bug reappearing: the width-only, no-0.9 value
    # would have been (80-20)/60 = 1.0 -- nothing like the correct,
    # height-bound, 0.9-fit answer. Assert the response is NOT that.
    old_wrong = (80.0 - 2 * 10.0) / 60
    assert body["mm_per_px"] != round(old_wrong, 4)
    assert body["mm_per_px"] < old_wrong


BENCH_TRAM = REPO / "tests" / "fixtures" / "benchmark" / "anime_hires" / "tram_s1024.png"


def test_suggest_paper_not_inert_on_height_branch():
    """On a height-limited source, a suggestion that scales paper WIDTH only
    does nothing to mmpx when height is the binding dimension. tram @
    257x176 is height-limited (worst case 2.26x) and lands in
    a non-clean tier -- the suggested sheet must be non-square (both new
    keys populated and different) and must actually raise
    gaps_below_visible_pct when applied, not just grow the wrong axis."""
    if not BENCH_TRAM.exists():
        raise SkipTest(f"benchmark fixture not present: {BENCH_TRAM}")
    r = _preflight(BENCH_TRAM, paper_w_mm=257.0, paper_h_mm=176.0, margin_mm=10.0, deposit_mm=0.15)
    assert r["tier"] != "clean", r["tier"]
    assert r["suggest_paper_w_mm"] is not None and r["suggest_paper_h_mm"] is not None
    assert r["suggest_paper_w_mm"] != r["suggest_paper_h_mm"], \
        "257x176 is non-square -- the scaled-sheet suggestion must stay non-square too"
    assert r["suggest_paper_mm"] == r["suggest_paper_w_mm"]

    # Guard against a width-only suggestion:
    # round(257 * 0.3/median_gap_mm) would be a WIDTH-only number applied to a
    # height-limited source. Re-running _preflight AT the new suggested sheet
    # must actually improve the visible-gap read, proving the number is not
    # arithmetically inert.
    r2 = _preflight(BENCH_TRAM, paper_w_mm=float(r["suggest_paper_w_mm"]),
                    paper_h_mm=float(r["suggest_paper_h_mm"]), margin_mm=10.0, deposit_mm=0.15)
    assert r2["gaps_below_visible_pct"] < r["gaps_below_visible_pct"], \
        (r["gaps_below_visible_pct"], r2["gaps_below_visible_pct"])
    assert r2["mm_per_px"] > r["mm_per_px"]


def test_suggest_paper_single_scalar_consistent_on_square_paper():
    """tram @ 180x180 is ALSO height-limited but the input
    paper is square -- suggest_paper_mm (the legacy single-scalar key) must
    still equal both suggest_paper_w_mm and suggest_paper_h_mm exactly, so
    existing square-paper wording/behavior is unchanged."""
    if not BENCH_TRAM.exists():
        raise SkipTest(f"benchmark fixture not present: {BENCH_TRAM}")
    r = _preflight(BENCH_TRAM, paper_w_mm=180.0, paper_h_mm=180.0, margin_mm=10.0, deposit_mm=0.15)
    assert r["tier"] != "clean", r["tier"]
    assert r["suggest_paper_w_mm"] == r["suggest_paper_h_mm"] == r["suggest_paper_mm"]


def _run_all():
    """SKIP is a THIRD outcome, distinct from PASS
    and FAIL, tracked separately and printed loudly in the summary -- never
    silently folded into a passing count. Any skip makes the run's exit code
    non-zero, so a green "N/N passed" line can only ever mean every test
    actually ran and actually asserted something (same convention as
    tests/test_gapaudit_coverage_scaling.py's _run_all())."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    skipped = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except SkipTest as e:
            skipped.append((t.__name__, str(e)))
            print(f"  SKIP  {t.__name__}: {e}")
        except AssertionError as e:
            failures.append(t.__name__)
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures.append(t.__name__)
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    n_passed = len(tests) - len(failures) - len(skipped)
    print()
    print(f"{n_passed}/{len(tests)} passed, {len(failures)}/{len(tests)} FAILED, "
          f"{len(skipped)}/{len(tests)} SKIPPED")
    if skipped:
        print(f"*** {len(skipped)} SKIPPED -- NOT verified, do not read as passing: "
              f"{', '.join(n for n, _ in skipped)} ***")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
    if failures or skipped:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
