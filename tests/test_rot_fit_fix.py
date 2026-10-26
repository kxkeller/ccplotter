"""Tests for the rot/analytic-fit ordering.

Invariant: ccplotter/composite.py computes the per-item cell-fit scale
(`fit = min(cellW/nw, cellH/nh) * 0.9`, at TWO sites -- the scale_aware_v2
preliminary resolution-boost decision, and the final placement scale) from the
item's ROTATED footprint. `item["rot"]` is also applied as a vpype layer
transform on already-scaled geometry, so a fit computed from the UNROTATED
source bbox overflows the cell for 90/270 rotations. `_rotated_extent_mm()`
computes the axis-aligned bbox of the ROTATED footprint (exact swap-or-no-op
for cardinal rotations, no trigonometry, so rot 0/180 stays byte-identical to
the unrotated path), and both fit sites use it.

Same "scripted assertions, plain exit code" convention as
tests/test_preflight_scaling.py (no pytest dependency in this repo; picked
up by pytest for free if ever added). Run directly with:
    python tests\\test_rot_fit_fix.py

SKIP discipline (same as tests/test_gapaudit_coverage_scaling.py):
a missing fixture raises SkipTest, never silently counts as a PASS. _run_all()
tracks skips as a THIRD, distinct outcome and returns non-zero if there are ANY.

Scope of test_b (the UI-shaped fixture): it replays a real captured
"Fill Page (crop)" POST body verbatim (only `items[0].id` is
substituted, since drawing ids are server-assigned uuid4 hex and cannot be
literally reused across a fresh upload). It confirms the fit/reporting logic
works on the real request shape.

It does NOT assert "the plot fills the page": the final drawn footprint is
~248 x 69.68mm on a 257 x 182mm page, because the crop step intersects the placed
item against `_fill_page_safe_region()` (paper INTERSECT reach polygon) and the
placed item overflows that region on every side, so the pen-reachable band, not
the fit, binds the result. At the captured request's placement only ~77.8mm of the
182mm sheet is pen-reachable, so drawing ~69.68mm of it is correct; the app must
say so loudly (see test_fillpage_warns_when_it_cannot_reach_the_page below).
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


class SkipTest(Exception):
    """Raised by a test when an optional fixture is missing. Tracked as a
    distinct SKIPPED outcome by _run_all() -- never silently counted as a
    PASS."""


REPO = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO / "tests" / "fixtures" / "rot_fix_2026_09_01"

# Self-contained runtime config (vpype.toml/pen_profiles.json/calibration.json)
# -- byte-verified copies of the SAME runtime config that produced the real
# captured "Fill Page (crop)" request this file replays. Copied into an
# isolated tempdir (never
# the live %LOCALAPPDATA% or repo-root config) so this test is deterministic
# and never touches real runtime state.
_TEST_ENV = Path(tempfile.mkdtemp(prefix="ccplotter_test_env_rotfix_"))
for _f in ("vpype.toml", "pen_profiles.json", "calibration.json"):
    shutil.copy2(FIXTURE_DIR / "_env" / _f, _TEST_ENV / _f)
os.chdir(_TEST_ENV)
os.environ["CCPLOTTER_HOME"] = str(_TEST_ENV)  # explicit runtime home (cwd == home)
_TMP_WORK = tempfile.mkdtemp(prefix="ccplotter_test_work_rotfix_")
os.environ["CCPLOTTER_WORK_DIR"] = _TMP_WORK
sys.path.insert(0, str(REPO))

import app as app_module  # noqa: E402  (registers all routes as import side effect)
from ccplotter.svg_io import _svg_bbox_mm  # noqa: E402  (same fn composite.py uses)
from ccplotter.config import WORK_DIR  # noqa: E402
from ccplotter.composite import _rotated_extent_mm  # noqa: E402

client = app_module.app.test_client()


def _upload(path: Path) -> str:
    with open(path, "rb") as fh:
        r = client.post("/api/add_drawing", data={"file": (fh, path.name)},
                         content_type="multipart/form-data")
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()["id"]


def test_rotated_extent_helper_cardinal_exact_no_trig():
    """rot in {0, 180} must be an EXACT no-op (byte-identical floats, not
    merely close), so the existing rot-0/180 output stays byte-identical. rot in {90, 270} must be an EXACT
    swap. Both take the no-trigonometry path in _rotated_extent_mm."""
    w, h = 120.0, 40.0
    assert _rotated_extent_mm(w, h, 0.0) == (w, h)
    assert _rotated_extent_mm(w, h, 180.0) == (w, h)
    assert _rotated_extent_mm(w, h, -180.0) == (w, h)
    assert _rotated_extent_mm(w, h, 360.0) == (w, h)
    assert _rotated_extent_mm(w, h, 90.0) == (h, w)
    assert _rotated_extent_mm(w, h, 270.0) == (h, w)
    assert _rotated_extent_mm(w, h, -90.0) == (h, w)
    # non-cardinal: closed-form 4-corner rotate-and-bound, sanity-checked
    # against the trig formula directly (this IS the trig path, so agreement
    # with the formula it's defined by is the only meaningful check).
    import math
    rw, rh = _rotated_extent_mm(w, h, 45.0)
    rad = math.radians(45.0)
    assert abs(rw - (abs(w * math.cos(rad)) + abs(h * math.sin(rad)))) < 1e-9
    assert abs(rh - (abs(w * math.sin(rad)) + abs(h * math.cos(rad)))) < 1e-9


def test_synthetic_rot90_bbox_fits_its_cell():
    """Bare-API repro: a deliberately elongated (3:1) test rectangle placed into a
    LANDSCAPE cell (cellW=190mm != cellH=90mm) at rot=0 and rot=90,
    scale_aware_v2=True (the boost decision is ALSO gated on rot, per the
    prelim-site fix). job_type=preview -- no reach-safety gate, no fill_page
    crop, so the true drawn bbox is unclipped and directly comparable to the
    cell it was fit against.

    With a fit computed from the unrotated bbox, the TRUE drawn bbox at rot=90
    would be 149.4mm tall against a 90mm cell (66% over) and the app's own
    reported actual_placed_height_mm (34.35mm) would not move at all between
    rot=0 and rot=90 -- silently wrong, no backstop. The true bbox must fit
    the cell and the reported size must match it.
    """
    test_png = FIXTURE_DIR / "test_rect_3to1.png"
    if not test_png.exists():
        raise SkipTest(f"missing fixture {test_png}")
    did = _upload(test_png)
    PAPER_W, PAPER_H, MARGIN = 200.0, 100.0, 5.0
    cellW, cellH = PAPER_W - 2 * MARGIN, PAPER_H - 2 * MARGIN

    def run(rot):
        body = {
            "items": [{"id": did, "scale": 1.0, "rot": rot, "dx": 0, "dy": 0}],
            "paper": f"{PAPER_W:g}x{PAPER_H:g}", "margin": MARGIN,
            "job_type": "preview", "pen_id": "uright_gel", "style": "tracev3",
            "fill_mm": 0.25, "scale_aware_v2": True,
        }
        r = client.post("/api/process", json=body)
        assert r.status_code == 200, r.get_data(as_text=True)
        res = r.get_json()
        geom = res["item_geometry"][did]
        eq = geom.get("scale_aware_equivalence_prelim_approx") or {}
        preview_path = WORK_DIR / res["preview_file"]
        true_bbox = _svg_bbox_mm(preview_path)
        assert true_bbox, f"no drawn geometry at rot={rot}"
        true_w, true_h = round(true_bbox[2] - true_bbox[0], 3), round(true_bbox[3] - true_bbox[1], 3)
        return eq.get("actual_placed_width_mm"), eq.get("actual_placed_height_mm"), true_w, true_h

    w0, h0, tw0, th0 = run(0.0)
    w90, h90, tw90, th90 = run(90.0)

    # Reported actual_placed_* now matches the independently-measured TRUE
    # drawn bbox at BOTH rotations (the response contract stops lying).
    assert abs(w0 - tw0) < 0.1 and abs(h0 - th0) < 0.1, (w0, h0, tw0, th0)
    assert abs(w90 - tw90) < 0.1 and abs(h90 - th90) < 0.1, (w90, h90, tw90, th90)

    # The actual invariant: BOTH rotations' TRUE bbox must fit inside the cell
    # (an unrotated-bbox fit made rot=90 overflow cellH=90mm by 66%, drawing 149.4mm tall
    # while the app reported an unchanged, rot-invariant 34.35mm).
    assert tw0 <= cellW + 0.5, f"rot=0 true width {tw0}mm exceeds cellW {cellW}mm"
    assert th0 <= cellH + 0.5, f"rot=0 true height {th0}mm exceeds cellH {cellH}mm"
    assert tw90 <= cellW + 0.5, f"rot=90 true width {tw90}mm exceeds cellW {cellW}mm"
    assert th90 <= cellH + 0.5, f"rot=90 true height {th90}mm exceeds cellH {cellH}mm"

    # rot=90 must be a DIFFERENT placement from rot=0, not rot-invariant
    # (an unrotated-bbox fit left actual_placed_height_mm at 34.35mm regardless
    # of rot). NOTE: this 3:1 source is more elongated than the 190x90
    # landscape cell (aspect 2.11:1), so rot=0 is WIDTH-bound (fills 90% of
    # cellW, leaving headroom on H) while rot=90 is HEIGHT-bound (fills 90%
    # of cellH, leaving headroom on W) -- the two placed sizes are correctly
    # NOT a simple axis swap of each other; only "clearly different, and
    # each individually correct for its own binding axis" is asserted.
    assert abs(h90 - 0.9 * cellH) < 1.0, f"rot=90 should be height-bound at 0.9*cellH={0.9*cellH}, got h90={h90}"
    assert abs(w0 - 0.9 * cellW) < 1.0, f"rot=0 should be width-bound at 0.9*cellW={0.9*cellW}, got w0={w0}"
    assert (w0, h0) != (w90, h90), "rot=90 placement identical to rot=0 -- fit did not react to rotation"


def test_ui_captured_fillpage_fixture_replayed_verbatim():
    """Real-request fixture: a captured POST body from a real browser click of
    "Fill Page (crop)" on a portrait source against a landscape page, replayed
    verbatim (only items[0].id substituted -- see module docstring).

    Asserts the fit BASIS and the response contract. It deliberately does NOT
    assert "fills the page": the final footprint is limited by the
    pen-reachable band, not the fit (see the module docstring and
    test_fillpage_warns_when_it_cannot_reach_the_page below).
    """
    body_path = FIXTURE_DIR / "captured_fillpage_body.json"
    src_path = FIXTURE_DIR / "citybus_source.png"
    if not (body_path.exists() and src_path.exists()):
        raise SkipTest(f"missing fixture {body_path} or {src_path}")

    did = _upload(src_path)
    body = json.loads(body_path.read_text())
    original_id = body["items"][0]["id"]
    body["items"][0]["id"] = did
    assert body["items"][0]["rot"] == 90 and body["items"][0]["scale"] == 1.85, (
        "captured fixture body changed shape -- re-verify this test's assumptions")

    r = client.post("/api/process", json=body)
    assert r.status_code == 200, r.get_data(as_text=True)
    res = r.get_json()
    geom = res["item_geometry"][did]
    eq = geom.get("scale_aware_equivalence_prelim_approx") or {}

    placed_bbox = geom.get("fill_page_placed_bbox_mm")
    assert placed_bbox, "no fill_page_placed_bbox_mm in response"
    placed_w = placed_bbox[2] - placed_bbox[0]
    placed_h = placed_bbox[3] - placed_bbox[1]

    # actual_placed_width/height_mm (condition 4: the response contract)
    # must equal the item's actual placed (ROTATED) footprint -- internal
    # consistency between the two numbers the fix touches.
    assert abs(eq.get("actual_placed_width_mm", -1) - placed_w) < 0.5
    assert abs(eq.get("actual_placed_height_mm", -1) - placed_h) < 0.5

    # An unrotated-bbox fit gives 269.73mm wide x 177.90mm tall for this
    # request. The rotation-aware fit is materially larger (~1.46x) -- pin that
    # growth so a regression back to the too-small, wrong-basis fit is caught.
    PRE_FIX_PLACED_W, PRE_FIX_PLACED_H = 269.73, 177.902
    assert placed_w > PRE_FIX_PLACED_W * 1.2, (
        f"placed width {placed_w:.1f}mm is not materially larger than the "
        f"known pre-fix buggy value {PRE_FIX_PLACED_W}mm -- fit may have regressed")
    assert placed_h > PRE_FIX_PLACED_H * 1.2, (
        f"placed height {placed_h:.1f}mm is not materially larger than the "
        f"known pre-fix buggy value {PRE_FIX_PLACED_H}mm -- fit may have regressed")

    # Final drawn width is genuinely correct -- the long axis DOES fill.
    preview_path = WORK_DIR / res["preview_file"]
    true_bbox = _svg_bbox_mm(preview_path)
    assert true_bbox, "no drawn geometry in the final preview"
    final_w = round(true_bbox[2] - true_bbox[0], 1)
    assert 240.0 <= final_w <= 256.0, f"final drawn width {final_w}mm moved unexpectedly"
    # DELIBERATELY NOT ASSERTED HERE: the final drawn HEIGHT (~69.68mm). It is
    # limited by the pen-reachable band at this placement, not by the fit, so
    # pinning a range here would cement an incidental number. The reach limit
    # is asserted by the Fill-Page reach-warning test below.


def _independent_safe_region(paper_w, paper_h, offset_x, offset_y, paper_rot=0.0):
    """Recompute paper INTERSECT reach INTERSECT 2mm-inset WITHOUT calling
    _fill_page_safe_region -- so the contract test below is not asserting that
    function against itself. Plain rectangle math against the reach bbox; the
    active reach polygon is rectangular (plotter_2003_v32), so a bbox
    intersection is exact here, and the test asserts agreement with the
    production function before relying on either."""
    from ccplotter.cal import load_cal, active_pen_profile, resolve_pen_offset
    from ccplotter.reach import active_reach_profile, pen_reach_profile
    cal = load_cal(); prof = active_pen_profile()
    ox, oy = resolve_pen_offset(cal, prof)
    reach = pen_reach_profile(active_reach_profile(cal), prof)
    poly = reach["polygon"]
    assert paper_rot == 0.0, "independent recompute only implemented for paper_rot=0"
    cx, cy = offset_x - ox, offset_y - oy
    # paper-frame reach (same inverse transform composite.py documents)
    ys = [p[1] + cy for p in poly]
    xs = [p[0] - cx for p in poly]
    lo_y, hi_y = max(0.0, min(ys)), min(paper_h, max(ys))
    lo_x, hi_x = max(0.0, min(xs)), min(paper_w, max(xs))
    if hi_y <= lo_y or hi_x <= lo_x:
        return 0.0, 0.0
    return max(0.0, (hi_x - lo_x) - 4.0), max(0.0, (hi_y - lo_y) - 4.0)  # 2mm each side


def test_fillpage_fills_the_REACHABLE_region():
    """CONTRACT test: Fill Page must fill the pen-reachable
    band, which is NOT the same as the page.

    The reachable band depends on where the paper sits. At the captured
    request's placement only ~77.8mm of the 182mm sheet is pen-reachable, and
    drawing ~69.68mm of it is CORRECT behaviour. Doing that silently is what
    the Fill-Page reach warning below prevents.
    """
    body_path = FIXTURE_DIR / "captured_fillpage_body.json"
    src_path = FIXTURE_DIR / "citybus_source.png"
    if not (body_path.exists() and src_path.exists()):
        raise SkipTest(f"missing fixture {body_path} or {src_path}")
    did = _upload(src_path)
    body = json.loads(body_path.read_text())
    body["items"][0]["id"] = did
    pw, ph = (float(v) for v in str(body["paper"]).lower().split("x"))
    ind_w, ind_h = _independent_safe_region(pw, ph, float(body.get("offset_x", 0.0)),
                                            float(body.get("offset_y", 0.0)),
                                            float(body.get("paper_rot", 0.0)))
    assert ind_h > 0, "independent recompute produced an empty safe region"

    r = client.post("/api/process", json=body)
    assert r.status_code == 200, r.get_data(as_text=True)
    res = r.get_json()
    true_bbox = _svg_bbox_mm(WORK_DIR / res["preview_file"])
    assert true_bbox, "no drawn geometry in the final preview"
    final_h = round(true_bbox[3] - true_bbox[1], 2)
    assert final_h >= 0.85 * ind_h, (
        f"drawn height {final_h}mm is well under the {ind_h:.1f}mm reachable band "
        f"-- Fill Page is not filling what the pen CAN reach")
    assert final_h <= ind_h + 2.0, (
        f"drawn height {final_h}mm EXCEEDS the {ind_h:.1f}mm reachable band -- "
        f"ink would be commanded where the pen cannot go")


def test_fillpage_warns_when_it_cannot_reach_the_page():
    """Fill-Page reach warning (ccplotter/composite.py fill_page_reach_stats +
    ccplotter/stats_warnings.py build_warnings).

    When the pen-reachable band is materially smaller than the sheet, Fill
    Page draws a fraction of the page. At the captured request's placement
    that is ~78mm of a 182mm sheet -- 43% -- and the app must say so,
    loudly, with numbers, under a control labelled "Fill the page".

    Preflight must say "this will not work at this size" loudly, with numbers,
    before ink hits paper. This is the same obligation applied to
    reachability instead of pen pitch.
    """
    body_path = FIXTURE_DIR / "captured_fillpage_body.json"
    src_path = FIXTURE_DIR / "citybus_source.png"
    if not (body_path.exists() and src_path.exists()):
        raise SkipTest(f"missing fixture {body_path} or {src_path}")
    did = _upload(src_path)
    body = json.loads(body_path.read_text())
    body["items"][0]["id"] = did
    pw, ph = (float(v) for v in str(body["paper"]).lower().split("x"))
    _, ind_h = _independent_safe_region(pw, ph, float(body.get("offset_x", 0.0)),
                                        float(body.get("offset_y", 0.0)),
                                        float(body.get("paper_rot", 0.0)))
    ratio = ind_h / ph
    assert ratio < 0.9, (
        f"precondition gone: reachable/paper is {ratio:.2f}; this fixture no "
        f"longer exercises the silent-underfill case")
    r = client.post("/api/process", json=body)
    assert r.status_code == 200, r.get_data(as_text=True)
    # MARKER, not a keyword sweep. Looking for the substring "reach" anywhere
    # in the warnings would pass on an unrelated line-density warning
    # containing the word "reaches" -- a check that cannot fail.
    # The shipped warning MUST carry this exact marker so the test cannot pass
    # on an incidental word match.
    MARKER = "pen-reachable"
    warns = [str(w.get("msg", "")) for w in (r.get_json().get("warnings") or [])]
    hit = [w for w in warns if MARKER in w.lower()]
    assert hit, (
        f"Fill Page delivered {ratio*100:.0f}% of the page height with NO warning "
        f"containing the required marker {MARKER!r}. Warnings were: {warns!r}")


def _safe(msg):
    """Windows consoles here are cp1252; app warning text can contain U+2192
    and friends. Printing a failure message must never itself raise -- that
    turns a legible XFAIL into an unrelated UnicodeEncodeError traceback."""
    return str(msg).encode("ascii", "replace").decode("ascii")


def _run_all():
    tests = [
        test_rotated_extent_helper_cardinal_exact_no_trig,
        test_synthetic_rot90_bbox_fits_its_cell,
        test_ui_captured_fillpage_fixture_replayed_verbatim,
        test_fillpage_fills_the_REACHABLE_region,
        test_fillpage_warns_when_it_cannot_reach_the_page,
    ]
    # Tests expected to FAIL because they name an open, undiagnosed
    # defect. An XFAIL is not a pass and must never be reported as one; an
    # XPASS (it started passing) is a loud event -- it means the defect was
    # fixed, and the test must be revisited.
    xfail_tests = []
    passed, failed, skipped, xfailed, xpassed = [], [], [], [], []
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"PASS {name}")
            passed.append(name)
        except SkipTest as e:
            print(f"SKIP {name}: {_safe(e)}")
            skipped.append(name)
        except Exception as e:
            print(f"FAIL {name}: {_safe(e)}")
            failed.append(name)
    for t in xfail_tests:
        name = t.__name__
        try:
            t()
            print(f"XPASS {name}  <-- the open defect appears FIXED; re-gate it")
            xpassed.append(name)
        except SkipTest as e:
            print(f"SKIP {name}: {_safe(e)}")
            skipped.append(name)
        except Exception as e:
            print(f"XFAIL {name} (expected, open defect): {_safe(e)}")
            xfailed.append(name)
    print(f"\n{len(passed)} passed, {len(failed)} failed, {len(skipped)} SKIPPED "
          f"(never counted as pass), {len(xfailed)} xfailed (open defect, expected), "
          f"{len(xpassed)} XPASSED")
    if skipped:
        print(f"SKIPPED: {skipped}")
    if xfailed:
        print(f"OPEN DEFECT still failing, as expected: {xfailed}")
    if xpassed:
        print(f"ATTENTION -- previously-failing defect now passes: {xpassed}")
    return 0 if (not failed and not skipped) else 1


if __name__ == "__main__":
    sys.exit(_run_all())
