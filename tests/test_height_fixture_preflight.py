"""Height-limited golden fixture: `_preflight` is checked against a RECORDED layout
number (the fixture's `base_fit_scale` plus the source's native pixel dimensions),
not against a re-typed copy of composite.py:651's own
`min(cellW/nw, cellH/nh) * 0.9` formula.

Fixture: tests/golden_tests/fixtures/synth-tallframe-height-limited/
  (source.png, request.json) + its entry in
  tests/golden_tests/replay_manifest.json (label
  "synth-tallframe-height-limited").

The other golden fixtures are width-limited, so none of them exercises the
HEIGHT branch of composite.py:651 / _preflight's mm/px fit
(ccplotter/trace_v3.py:1259). This fixture is a synthetic 800x1400px
portrait raster (pure line art, no filled regions) on a 220x160mm sheet
(margin 10 -> usable 200x140mm):
    usable_w/nw = 200/800 = 0.25   (width branch)
    usable_h/nh = 140/1400 = 0.1   (height branch, BINDING -- 2.5x margin)

Why the comparison is NOT a re-typed formula
---------------------------------------------
`base_fit_scale` (recorded in the manifest) is composite.py's own real
`fit = min(cellW/nw_svg, cellH/nh_svg) * 0.9` output, at the pipeline's
working/trace resolution -- not the raw native canvas. Converting it to a
"mm per NATIVE source pixel" (what `_preflight` reports) uses only two
things that are NOT composite.py's fit formula:
  - PX2MM (ccplotter/config.py): the fixed vpype-units constant, imported
    directly, not re-derived.
  - The pipeline's OWN recorded ratio between its working trace resolution
    and the source's native long side (trace_info.effective_res /
    trace_info.native_px), also read straight from the recorded audit, not
    computed here.
See `_recorded_layout_mm_per_native_px` below for the full derivation and
why it is EXACT (not approximate) along this fixture's HEIGHT axis
specifically -- the axis that actually binds here.

SKIP discipline (same convention as tests/test_gapaudit_coverage_scaling.py):
a missing fixture/manifest entry never counts as a pass. Run directly:
    .venv\\Scripts\\python.exe tests\\test_height_fixture_preflight.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path


class SkipTest(Exception):
    """A required fixture/manifest entry is missing. Tracked as a distinct
    SKIPPED outcome by _run_all() -- never silently counted as a PASS."""


REPO = Path(__file__).resolve().parent.parent
# os.chdir(REPO) also resolves a repo-root pen_profiles.json/calibration.json
# if one exists: such a file shadows %LOCALAPPDATA%/ccPlotter's real one
# whenever CWD is the repo root. This file is safe from that only because
# every assertion here reads `mm_per_px`, which _preflight derives from paper/
# margin/native-pixel geometry and never touches the active pen profile at
# all -- unlike _gap_audit/_coverage_debug, nothing here would silently
# measure the wrong pen's fill_mm even if CWD-resolution picked up a
# different profile.
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home; pen/calibration never resolve from CWD
_test_home.use_temp_home("ccplotter_test_home_heightfixture_")
_TMP_WORK = tempfile.mkdtemp(prefix="ccplotter_test_work_heightfixture_")
os.environ["CCPLOTTER_WORK_DIR"] = _TMP_WORK
sys.path.insert(0, str(REPO))

from ccplotter.config import PX2MM
from ccplotter.trace_v3 import _preflight

FIXTURE_LABEL = "synth-tallframe-height-limited"
FIXTURE_DIR = REPO / "tests/golden_tests/fixtures/synth-tallframe-height-limited"
SOURCE_PNG = FIXTURE_DIR / "source.png"
REQUEST_JSON = FIXTURE_DIR / "request.json"
MANIFEST_PATH = REPO / "tests/golden_tests/replay_manifest.json"

# Agreement tolerance: 0.5%, the floor for real photographic sources
# (composite.py fits the SVG INK
# BBOX, not the source canvas -- a residual under any raster-based model).
# Measured residual on this fixture: 0.23% -- comfortably
# inside, with margin to spare.
TOLERANCE = 0.005


def _load_fixture_entry():
    if not SOURCE_PNG.exists() or not REQUEST_JSON.exists():
        raise SkipTest(f"fixture files missing under {FIXTURE_DIR} -- "
                        f"run artifacts/height-fixture-2026-08-16/gen_fixture.py")
    if not MANIFEST_PATH.exists():
        raise SkipTest(f"no manifest at {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text())
    entry = manifest.get("fixtures", {}).get(FIXTURE_LABEL)
    if entry is None:
        raise SkipTest(f"{FIXTURE_LABEL!r} not recorded in {MANIFEST_PATH} -- see "
                        f"artifacts/height-fixture-2026-08-16/RESULT.md for how it was captured")
    if "source_native_px_wh" not in entry:
        raise SkipTest(f"{FIXTURE_LABEL!r} manifest entry has no source_native_px_wh -- "
                        f"stale entry, re-record via the harness (scripts/replay_harness.py, "
                        f"run_process_fixture now captures this field)")
    return entry


def _paper_and_margin():
    req = json.loads(REQUEST_JSON.read_text())
    paper_w, paper_h = (float(x) for x in req["paper"].split("x"))
    return paper_w, paper_h, float(req["margin"])


def _recorded_layout_mm_per_native_px(entry) -> float:
    """The GROUND-TRUTH mm-per-native-source-pixel implied by composite.py's
    OWN recorded fit for this fixture -- built entirely from RECORDED
    numbers (base_fit_scale, trace_info.effective_res, trace_info.native_px)
    plus the fixed PX2MM constant, never from re-deriving
    `min(cellW/nw, cellH/nh) * 0.9` ourselves.

    mm_per_trace_px = base_fit_scale * PX2MM
        (definitional: composite.py's SVG canvas is in CSS px, 1 px =
        PX2MM mm -- ccplotter/config.py, ccplotter/svg_io.py's
        _svg_bbox_mm reads the same SVG units composite.py places into).

    mm_per_native_px = mm_per_trace_px * (effective_res / native_px)
        (the pipeline's own recorded working-resolution-to-native ratio).
        This is EXACT (not merely approximate) along this fixture's HEIGHT
        axis: trace_info.native_px is recorded as max(native_w, native_h)
        (ccplotter/render.py), and this fixture's native HEIGHT (1400px) IS
        that max side -- mask._load_gray's aspect-preserving resize sets
        the long axis to precisely `effective_res` px with no intermediate
        rounding (a (1400, 800) source resized to effective_res=3000 gives
        exactly (3000, 1714, 3), never 2999/3001). Height is this
        fixture's BINDING axis (see test_height_fixture_is_genuinely_
        height_limited below), so this exactness is exactly where it
        matters."""
    item = entry["response_metadata"]["item_geometry"][0]
    ti = entry["audit"]["trace_info"]
    ratio = ti["effective_res"] / ti["native_px"]
    return item["base_fit_scale"] * PX2MM * ratio


def test_height_fixture_is_genuinely_height_limited():
    """Proof, on the RAW native-pixel basis _preflight itself uses, that
    this fixture is height-limited -- not a coin-flip: usable_h/nh must be
    strictly less than usable_w/nw, by a comfortable margin (>2x)."""
    entry = _load_fixture_entry()
    nw, nh = entry["source_native_px_wh"][0]
    paper_w, paper_h, margin = _paper_and_margin()
    usable_w, usable_h = paper_w - 2 * margin, paper_h - 2 * margin
    w_ratio, h_ratio = usable_w / nw, usable_h / nh
    assert h_ratio < w_ratio, (h_ratio, w_ratio)
    assert w_ratio / h_ratio > 2.0, \
        f"height binds by only {w_ratio / h_ratio:.3f}x -- too close to a coin-flip"


def test_preflight_matches_recorded_layout_fit():
    """THE test criterion 2 exists for: _preflight's mm/px, run against the
    fixture's real source file, must agree with the RECORDED layout number
    (base_fit_scale, as captured by the real composite.py pipeline through
    the harness) within TOLERANCE -- not against a re-typed copy of
    _preflight's own formula."""
    entry = _load_fixture_entry()
    ground_truth = _recorded_layout_mm_per_native_px(entry)
    paper_w, paper_h, margin = _paper_and_margin()
    r = _preflight(SOURCE_PNG, paper_w_mm=paper_w, paper_h_mm=paper_h, margin_mm=margin)
    residual = abs(r["mm_per_px"] - ground_truth) / ground_truth
    assert residual < TOLERANCE, \
        (f"_preflight={r['mm_per_px']}, recorded-layout={ground_truth}, "
         f"residual={residual:.4%} >= tolerance {TOLERANCE:.2%}")


def test_old_width_only_no_09_model_fails_this_fixture():
    """The old width-only formula
    (`mmpx = (paper_w - 2*margin) / nw`, no 0.9 fit factor) must measurably
    FAIL against the recorded layout number on this fixture: it overstates
    mm/px by 2.78x (177% off), not a marginal miss."""
    entry = _load_fixture_entry()
    ground_truth = _recorded_layout_mm_per_native_px(entry)
    nw, _nh = entry["source_native_px_wh"][0]
    paper_w, _paper_h, margin = _paper_and_margin()
    usable_w = paper_w - 2 * margin
    old_mmpx = usable_w / nw
    residual = abs(old_mmpx - ground_truth) / ground_truth
    assert residual > 1.0, \
        (f"old width-only/no-0.9 model ({old_mmpx}) should be wildly off the recorded "
         f"layout number ({ground_truth}), was only {residual:.4%} off")


def test_agreement_check_detects_a_perturbed_fit_constant():
    """Drift-detection sensitivity: simulates what a change to composite.py:651
    would do to `base_fit_scale`, WITHOUT editing composite.py, and confirms
    the SAME comparison test_preflight_matches_recorded_layout_fit uses would
    then FAIL. This test pins the ARITHMETIC sensitivity permanently.

    What this test does NOT cover -- three distinct cases, not equivalent:
      1. `_preflight`'s own formula changes -> this test fails immediately
         (live `_preflight` output diverges from the frozen recorded
         ground truth). Genuinely closed.
      2. `composite.py:651` changes AND this fixture is RE-RECORDED
         (`base_fit_scale` in the manifest updates) -> this test fails
         (ground truth moves, `_preflight` does not).
      3. `composite.py:651` changes WITHOUT a re-record -> this test still
         PASSES, because it compares live `_preflight` against a FROZEN
         manifest number that no longer reflects composite's real
         behaviour. That gap is covered by a SEPARATE surface, not this
         one: `scripts/replay_harness.py verify` goes red immediately on
         every process fixture (`base_fit_scale` lives inside
         `response_metadata.item_geometry`, a field `fixture_diff()`
         compares against the frozen manifest -- no re-record needed for
         the HARNESS to notice, only for THIS test's ground truth to move
         with it). Two independent surfaces, not one test that catches
         everything."""
    entry = _load_fixture_entry()
    real_item = entry["response_metadata"]["item_geometry"][0]
    real_base_fit = real_item["base_fit_scale"]
    ti = entry["audit"]["trace_info"]
    ratio = ti["effective_res"] / ti["native_px"]
    nw, nh = entry["source_native_px_wh"][0]
    paper_w, paper_h, margin = _paper_and_margin()
    usable_w, usable_h = paper_w - 2 * margin, paper_h - 2 * margin

    r = _preflight(SOURCE_PNG, paper_w_mm=paper_w, paper_h_mm=paper_h, margin_mm=margin)

    # (a) the 0.9 fit factor perturbed to 1.0 (~11.1% swing)
    perturbed_fit_a = real_base_fit * (1.0 / 0.9)
    perturbed_gt_a = perturbed_fit_a * PX2MM * ratio
    residual_a = abs(r["mm_per_px"] - perturbed_gt_a) / perturbed_gt_a
    assert residual_a > TOLERANCE, \
        "comparison did not notice a perturbed-0.9 composite.py:651"

    # (b) min(cellW/nw,cellH/nh) replaced by the WIDTH branch only (the
    # old composite bug's shape) -- rescale the real height-branch fit by
    # the branch ratio measured on this fixture's own native px.
    branch_ratio = (usable_w / nw) / (usable_h / nh)
    perturbed_fit_b = real_base_fit * branch_ratio
    perturbed_gt_b = perturbed_fit_b * PX2MM * ratio
    residual_b = abs(r["mm_per_px"] - perturbed_gt_b) / perturbed_gt_b
    assert residual_b > TOLERANCE, \
        "comparison did not notice a width-branch-only composite.py:651"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures, skipped = [], []
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
