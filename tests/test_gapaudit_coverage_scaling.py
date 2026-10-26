"""Tests for the mm/px scaling in trace_v3._gap_audit and _coverage_debug.

These two functions must report the same mm/px as _preflight for the same
source and paper (see tests/test_preflight_scaling.py); a width-only formula
without the composite fit factor overstates mm/px on height-limited frames.

Scope: ccplotter/trace_v3.py _gap_audit and _coverage_debug, plus the two
ccplotter/routes/pen.py route handlers that call them (api_gap_audit,
api_coverage_debug). _preflight itself is covered by test_preflight_scaling.py.

Known limitation (not fixed here): _coverage_debug references the bare name
`active_pen_profile`, which is never imported into ccplotter/trace_v3.py's
module namespace. Calling _coverage_debug (directly, or via
/api/coverage_debug) raises `NameError: name 'active_pen_profile' is not
defined` the moment it reaches that line. The mm/px checks below are run around
it with a scoped, TEST-ONLY monkeypatch of `ccplotter.trace_v3.active_pen_profile`
(never a file edit). The test
`test_coverage_debug_active_pen_profile_nameerror_is_preexisting` pins the
bug's current existence (delete/update it when the bug is fixed).

SKIP discipline: a missing benchmark fixture must never silently count as a
PASS. Every test that depends on an optional fixture raises `SkipTest` (below)
instead of printing and returning; `_run_all()` tracks skips as a THIRD,
distinct outcome, prints them LOUDLY in the final summary line (never folded
into "N/N passed"), and returns a non-zero exit code if there are ANY skips
-- a green run means every test actually ran and asserted something.
`test_gap_audit_preflight_agreement_all_six_both_papers` requires ALL 12
source/paper cells or raises SkipTest naming exactly which are missing,
rather than passing after checking only some of them.

Run directly (same convention as test_preflight_scaling.py, no pytest dep):
    .venv\\Scripts\\python.exe tests\\test_gapaudit_coverage_scaling.py
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
_test_home.use_temp_home("ccplotter_test_home_gapaudit_")
_TMP_WORK = tempfile.mkdtemp(prefix="ccplotter_test_work_gapaudit_")
os.environ["CCPLOTTER_WORK_DIR"] = _TMP_WORK
sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image

from ccplotter.trace_v3 import _gap_audit, _coverage_debug, _preflight
import ccplotter.trace_v3 as _tv3

WORK = Path(_TMP_WORK)


def _make_gray(path: Path, w: int, h: int):
    """Same synthetic-source convention as test_preflight_scaling.py's
    _make_gray -- a centered ink rectangle so Otsu/skeletonize/distance-
    transform have real strokes and gaps to measure."""
    img = np.full((h, w), 255, dtype=np.uint8)
    x0, y0 = w // 4, h // 4
    x1, y1 = w - w // 4, h - h // 4
    img[y0:y1, x0:x1] = 0
    img[y0 + max(1, h // 20):y1 - max(1, h // 20),
        x0 + max(1, w // 20):x1 - max(1, w // 20)] = 255
    Image.fromarray(img).save(path)
    return path


# ---------------------------------------------------------------------------
# _gap_audit
# ---------------------------------------------------------------------------

def test_gap_audit_height_limited_branch_uses_height_not_width():
    """nw=100 (usable_w/nw=0.8) vs nh=200 (usable_h/nh=0.4) on a 100x100mm/
    margin10 page -- height binds. Mirrors test_preflight_scaling.py's
    equivalent test for _preflight."""
    src = _make_gray(WORK / "src_ga_h.png", w=100, h=200)
    r = _gap_audit(src, resolution=50, paper_w_mm=100.0, paper_h_mm=100.0, margin_mm=10.0)
    usable_w, usable_h = 80.0, 80.0
    expected_native = round(min(usable_w / 100, usable_h / 200) * 0.9, 4)
    assert expected_native == round(0.4 * 0.9, 4) == 0.36
    assert r["mm_per_px_native"] == expected_native, r
    # trace stage: resolution=50 -> trace_h = int(200*50/100) = 100
    expected_trace = round(min(usable_w / 50, usable_h / 100) * 0.9, 4)
    assert r["mm_per_px_trace"] == expected_trace, r


def test_gap_audit_width_limited_branch_uses_width_not_height():
    src = _make_gray(WORK / "src_ga_w.png", w=200, h=100)
    r = _gap_audit(src, resolution=50, paper_w_mm=100.0, paper_h_mm=100.0, margin_mm=10.0)
    usable_w, usable_h = 80.0, 80.0
    expected_native = round(min(usable_w / 200, usable_h / 100) * 0.9, 4)
    assert expected_native == round(0.4 * 0.9, 4) == 0.36
    assert r["mm_per_px_native"] == expected_native, r


def test_gap_audit_09_fit_factor_present():
    src = _make_gray(WORK / "src_ga_sq.png", w=100, h=100)
    r = _gap_audit(src, resolution=100, paper_w_mm=100.0, paper_h_mm=100.0, margin_mm=0.0)
    naive = 100.0 / 100        # pre-fix width-only, no-0.9 model
    assert r["mm_per_px_native"] == round(naive * 0.9, 4) == 0.9
    assert r["mm_per_px_native"] < naive


def test_gap_audit_paper_h_mm_defaults_to_square():
    src = _make_gray(WORK / "src_ga_default.png", w=100, h=300)
    r_omitted = _gap_audit(src, resolution=50, paper_w_mm=120.0, margin_mm=10.0)
    r_explicit_square = _gap_audit(src, resolution=50, paper_w_mm=120.0, paper_h_mm=120.0, margin_mm=10.0)
    assert r_omitted["mm_per_px_native"] == r_explicit_square["mm_per_px_native"]
    r_explicit_tall = _gap_audit(src, resolution=50, paper_w_mm=120.0, paper_h_mm=400.0, margin_mm=10.0)
    assert r_omitted["mm_per_px_native"] != r_explicit_tall["mm_per_px_native"]


def test_api_gap_audit_route_forwards_paper_height():
    """routes/pen.py api_gap_audit must not discard the paper's height
    component -- same check as test_preflight_scaling.py's
    test_api_preflight_route_forwards_paper_height, driven against the real
    Flask route in-process (test_client(), no port bound)."""
    from ccplotter.routes.pen import app

    did = "gaudittest1"
    src = _make_gray(WORK / f"src_{did}.png", w=60, h=600)  # strongly height-limited
    client = app.test_client()
    resp = client.post("/api/gap_audit", json={"id": did, "paper": "80x40", "margin": 10, "resolution": 40})
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body["ok"] is True

    direct = _gap_audit(src, resolution=40, paper_w_mm=80.0, paper_h_mm=40.0, margin_mm=10.0,
                        deposit_mm=body["meta"]["deposit_mm"])
    assert body["meta"]["mm_per_px_native"] == direct["mm_per_px_native"], \
        (body["meta"]["mm_per_px_native"], direct["mm_per_px_native"])

    # Guard against the old bug: width-only, no-0.9 would have been (80-20)/60 = 1.0.
    old_wrong = (80.0 - 2 * 10.0) / 60
    assert body["meta"]["mm_per_px_native"] != round(old_wrong, 4)
    assert body["meta"]["mm_per_px_native"] < old_wrong


BENCH_TRAM = REPO / "tests" / "fixtures" / "benchmark" / "anime_hires" / "tram_s1024.png"


def test_gap_audit_preflight_agreement():
    """The central invariant: _preflight and _gap_audit must report the SAME
    mm/px for the same source and paper. tram @ 257x176 is the worst-case
    height-limited cell: a width-only formula overstates mm/px there by about
    2.26x, so any disagreement between the two shows up most strongly here."""
    if not BENCH_TRAM.exists():
        raise SkipTest(f"fixture image not redistributable; place tram_s1024.png at {BENCH_TRAM.parent} to run")
    pf = _preflight(BENCH_TRAM, paper_w_mm=257.0, paper_h_mm=176.0, margin_mm=10.0, deposit_mm=0.15)
    ga = _gap_audit(BENCH_TRAM, resolution=400, paper_w_mm=257.0, paper_h_mm=176.0, margin_mm=10.0,
                    deposit_mm=0.15)
    assert pf["mm_per_px"] == ga["mm_per_px_native"], (pf["mm_per_px"], ga["mm_per_px_native"])


def test_gap_audit_preflight_agreement_all_six_both_papers():
    """Same agreement check, all six benchmark sources, both papers (12 cells).

    It requires ALL 12 cells; if even one source is missing it raises
    SkipTest NAMING every missing source, so a partial fixture set is
    never silently reported as a pass."""
    src_dir = REPO / "tests" / "fixtures" / "benchmark" / "anime_hires"
    sources = ["ice", "belltower", "tram", "bike", "a350", "giraffe"]
    papers = [(180.0, 180.0), (257.0, 176.0)]
    missing = [name for name in sources if not (src_dir / f"{name}_s1024.png").exists()]
    if missing:
        raise SkipTest(f"fixture image not redistributable; place {[n + '_s1024.png' for n in missing]} "
                       f"at {src_dir} to run -- 0 of 12 cells verified")
    n_verified = 0
    for name in sources:
        src = src_dir / f"{name}_s1024.png"
        for pw, ph in papers:
            pf = _preflight(src, paper_w_mm=pw, paper_h_mm=ph, margin_mm=10.0, deposit_mm=0.15)
            ga = _gap_audit(src, resolution=400, paper_w_mm=pw, paper_h_mm=ph, margin_mm=10.0,
                            deposit_mm=0.15)
            assert pf["mm_per_px"] == ga["mm_per_px_native"], \
                (name, pw, ph, pf["mm_per_px"], ga["mm_per_px_native"])
            n_verified += 1
    assert n_verified == len(sources) * len(papers) == 12, n_verified


# ---------------------------------------------------------------------------
# _coverage_debug
# ---------------------------------------------------------------------------

def test_coverage_debug_active_pen_profile_nameerror_is_preexisting():
    """Pins a known, unrelated limitation: _coverage_debug currently raises
    NameError the moment it's actually called (active_pen_profile is
    referenced but never imported into ccplotter/trace_v3.py). If that import
    is fixed, this test should be updated/removed in the fixing change, not
    silently made to pass here."""
    assert "active_pen_profile" not in dir(_tv3), \
        "active_pen_profile now resolves in trace_v3's namespace -- the " \
        "pre-existing bug this test pins appears to be fixed; update/remove " \
        "this test in that fixing commit, not here."
    src = BENCH_TRAM
    if not src.exists():
        raise SkipTest(f"fixture image not redistributable; place tram_s1024.png at {src.parent} to run")
    try:
        _coverage_debug(src, base_mode="lineart", resolution=200, paper_w_mm=180.0, margin_mm=10.0)
        raise AssertionError("expected NameError (active_pen_profile) -- did not raise")
    except NameError as e:
        assert "active_pen_profile" in str(e)


def _with_active_pen_profile_patched(fn):
    """Scoped, TEST-ONLY monkeypatch (never a file edit) working around the
    pre-existing NameError above so the mm/px fix itself can be validated
    end-to-end. Restores trace_v3's namespace afterward."""
    from ccplotter.cal import active_pen_profile as real_fn
    had_attr = hasattr(_tv3, "active_pen_profile")
    prev = getattr(_tv3, "active_pen_profile", None)
    _tv3.active_pen_profile = real_fn
    try:
        return fn()
    finally:
        if had_attr:
            _tv3.active_pen_profile = prev
        else:
            delattr(_tv3, "active_pen_profile")


def test_coverage_debug_height_limited_mm_per_px():
    """With the pre-existing NameError worked around (test-only patch, see
    above), _coverage_debug's mm_per_px must use min(usable_w/W, usable_h/H)*0.9
    on ITS OWN working-resolution W,H (from _load_gray), not width-only."""
    src = BENCH_TRAM
    if not src.exists():
        raise SkipTest(f"fixture image not redistributable; place tram_s1024.png at {src.parent} to run")

    def _run():
        from ccplotter.mask import _load_gray
        img = _load_gray(src, 300)
        H, W = img.shape[:2]
        r = _coverage_debug(src, base_mode="lineart", resolution=300,
                            paper_w_mm=257.0, paper_h_mm=176.0, margin_mm=10.0, pen_mm=0.5, tone="off")
        expected = round(max(1e-6, min((257.0 - 20) / W, (176.0 - 20) / H) * 0.9), 4)
        assert r["stats"]["mm_per_px"] == expected, (r["stats"]["mm_per_px"], expected, W, H)
        # Guard against the old bug: width-only would give (257-20)/W.
        old_wrong = round(max(1e-6, (257.0 - 20) / W), 4)
        assert r["stats"]["mm_per_px"] != old_wrong
        assert r["stats"]["mm_per_px"] < old_wrong
    _with_active_pen_profile_patched(_run)


def test_coverage_debug_paper_h_mm_defaults_to_square():
    def _run():
        src = _make_gray(WORK / "src_cov_default.png", w=100, h=300)
        r_omitted = _coverage_debug(src, base_mode="lineart", resolution=100, paper_w_mm=120.0, margin_mm=10.0)
        r_sq = _coverage_debug(src, base_mode="lineart", resolution=100, paper_w_mm=120.0,
                               paper_h_mm=120.0, margin_mm=10.0)
        assert r_omitted["stats"]["mm_per_px"] == r_sq["stats"]["mm_per_px"]
        r_tall = _coverage_debug(src, base_mode="lineart", resolution=100, paper_w_mm=120.0,
                                 paper_h_mm=400.0, margin_mm=10.0)
        assert r_omitted["stats"]["mm_per_px"] != r_tall["stats"]["mm_per_px"]
    _with_active_pen_profile_patched(_run)


def test_api_coverage_debug_route_forwards_paper_height():
    """routes/pen.py api_coverage_debug must not discard the paper's height
    component. Driven against the real Flask route in-process, with the
    pre-existing active_pen_profile NameError worked around (test-only
    patch) so the route logic under test -- height forwarding -- is what's
    actually exercised."""
    def _run():
        from ccplotter.routes.pen import app
        did = "covtest1"
        src = _make_gray(WORK / f"src_{did}.png", w=60, h=600)  # strongly height-limited
        client = app.test_client()
        resp = client.post("/api/coverage_debug", json={
            "id": did, "base": "lineart", "paper": "80x40", "margin": 10, "resolution": 60})
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        assert body["ok"] is True

        from ccplotter.mask import _load_gray
        img = _load_gray(src, 60)
        H, W = img.shape[:2]
        expected = round(max(1e-6, min((80.0 - 20) / W, (40.0 - 20) / H) * 0.9), 4)
        assert body["stats"]["mm_per_px"] == expected, (body["stats"]["mm_per_px"], expected)
        old_wrong = round(max(1e-6, (80.0 - 20) / W), 4)
        assert body["stats"]["mm_per_px"] != old_wrong
        assert body["stats"]["mm_per_px"] < old_wrong
    _with_active_pen_profile_patched(_run)


def _run_all():
    """SKIP is a THIRD outcome, distinct from PASS and FAIL, tracked
    separately and printed loudly in the summary -- never silently folded
    into a passing count. Any skip makes the run's exit code non-zero, so a
    green "N/N passed" line can only ever mean every test actually ran and
    actually asserted something."""
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
