"""Tests for the Photo-512 mode.

Ships a plain greyscale photograph recipe (source pre-downsampled to shorter side
512px, CLAHE off, no pre-simplification, tone off, engine tracev3, everything else
stock) as a real, opt-in style value ("photo512").

The recipe lives in ONE place: ccplotter/config.py's PHOTO512_RECIPE /
PHOTO512_SOURCE_SIDE_PX. This file pins that dict verbatim and proves, end to
end through the real Flask route (test_client(), no port bound):

  1. (LOAD-BEARING) that the source-side pre-downsample mechanism is
     actually load-bearing: style="photo512" produces DIFFERENT geometry
     than an otherwise-matched plain style="tracev3" request. Deleting the
     render_opts["source_side_cap"] wiring silently degrades photo512 to
     plain tracev3@cap while every other test still passes; this test is the
     one that catches it.
  2. tests are named for what they actually assert: the "reaches 512" test
     observes the side argument via a spy on the real call site, and the
     "cleans up after render" test observes render.py's own cleanup code
     (not a helper call plus a manual unlink).
  3. the temp pre-shrink file is removed even when the code inside the
     try block raises (not just on the success path).
  4. a caller cannot smuggle `source_side_cap` into a plain
     style="tracev3" request and trigger the pre-shrink path outside
     style="photo512".
  5. `native_res=True` sent alongside style="photo512" is forced back
     to False -- proven by identical output with/without that override,
     not merely by a config-dict read.
  6. selecting style="photo512" forces clahe/tone/resolution even when the
     caller's own request body tries to set them to something else.

Same "scripted assertions, plain exit code" convention as
tests/test_preflight_scaling.py (no pytest dependency; run directly):
    .venv\\Scripts\\python.exe tests\\test_photo512_recipe.py

SKIP discipline (same convention as the other files here): a missing optional
fixture raises SkipTest, never silently counts as a PASS; _run_all() tracks
skips as a distinct outcome and a green run can only mean every test actually
ran and asserted something.
"""
import os
import re
import sys
import tempfile
from pathlib import Path


class SkipTest(Exception):
    """Raised by a test when an optional fixture is missing."""


REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home; pen/calibration never resolve from CWD
_test_home.use_temp_home("ccplotter_test_home_photo512_")
_TMP_WORK = tempfile.mkdtemp(prefix="ccplotter_test_work_photo512_")
os.environ["CCPLOTTER_WORK_DIR"] = _TMP_WORK
sys.path.insert(0, str(REPO))

import numpy as np
import cv2
from PIL import Image

from ccplotter.config import PHOTO512_RECIPE, PHOTO512_SOURCE_SIDE_PX, MODE_MAX_RES, WORK_DIR
from ccplotter.mask import _prepare_shorter_side_gray
import ccplotter.render as _render_mod

WORK = Path(_TMP_WORK)


def _make_photo(path: Path, w: int, h: int):
    """A synthetic 'photograph' with real tonal texture (not flat/bilevel) --
    a soft gradient plus a checker patch, so both the recipe test and the
    tracer have real content to work with. Saved as color JPEG (like the six
    real benchmark sources), not pre-flattened, so the pre-downsample step's
    own greyscale conversion is genuinely exercised."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = (128 + 100 * np.sin(xx / 11.0) * np.cos(yy / 17.0)).astype(np.uint8)
    img = np.stack([base, np.roll(base, 5, axis=0), np.roll(base, -5, axis=1)], axis=-1)
    Image.fromarray(img, mode="RGB").save(path, quality=92)
    return path


# ---------------------------------------------------------------------------
# 1. The recipe dict itself -- pinned verbatim
# ---------------------------------------------------------------------------

def test_recipe_dict_pinned():
    assert PHOTO512_SOURCE_SIDE_PX == 512, PHOTO512_SOURCE_SIDE_PX
    assert PHOTO512_RECIPE == {
        "clahe": False,
        "tone": "off",
        # The recipe's own literal value, not
        # MODE_MAX_RES["tracev3"] -- see config.py's comment for why the two
        # are not interchangeable (composite.py's scale_aware_v2 boost reads
        # this as a multiplicand, not just a value that gets capped).
        "resolution": 3000,
        "source_side_cap": 512,
        "native_res": False,
    }, PHOTO512_RECIPE


# ---------------------------------------------------------------------------
# 2. _prepare_shorter_side_gray -- matches the experiment's own resize
#    formula (00_make_inputs.py::load_gray_side) exactly
# ---------------------------------------------------------------------------

def test_prepare_shorter_side_gray_matches_experiment_formula():
    src = _make_photo(WORK / "src_formulatest.jpg", w=900, h=300)
    out_path = _prepare_shorter_side_gray(src, 512)
    try:
        got = cv2.imdecode(np.fromfile(str(out_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        # hand-computed per 00_make_inputs.py::load_gray_side: s = side/min(h,w)
        expect_src = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        h, w = expect_src.shape
        s = 512 / min(h, w)
        expect = cv2.resize(expect_src, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        assert got.shape == expect.shape, (got.shape, expect.shape)
        assert min(got.shape) == 512, got.shape  # shorter side lands exactly on 512
        assert np.array_equal(got, expect), "pixel mismatch vs hand-computed reference resize"
    finally:
        try: out_path.unlink()
        except Exception: pass


# ---------------------------------------------------------------------------
# 3. temp file survives an exception inside the try block
# ---------------------------------------------------------------------------

def test_render_cleans_up_temp_file_on_exception():
    """Unit-level pin: render.py's try/finally around
    _load_gray must remove the photo512 temp source even when _load_gray
    itself raises -- NOT only on the success path. Calls
    render_raster_to_svg directly (not through the HTTP route) so the
    exception is observed directly, monkeypatching ccplotter.render's OWN
    bound name for _load_gray (the name render.py's code actually resolves
    at call time, per its `from ccplotter.mask import (_load_gray, ...)`)."""
    src = _make_photo(WORK / "src_exceptiontest.jpg", w=800, h=600)
    out_path = WORK_DIR / "exceptiontest_out.svg"
    before = set(WORK_DIR.glob("photo512_src_*.png"))

    class _Boom(Exception):
        pass

    orig_load_gray = _render_mod._load_gray

    def _raiser(*a, **kw):
        raise _Boom("synthetic failure inside the try block")

    _render_mod._load_gray = _raiser
    try:
        raised = False
        try:
            _render_mod.render_raster_to_svg(src, out_path, {
                "mode": "tracev3", "resolution": 3000, "clahe": False, "tone": "off",
                "source_side_cap": 512, "fill_mm": 0.25, "paper_w_mm": 180.0, "margin_mm": 10.0,
            })
        except _Boom:
            raised = True
        assert raised, "expected the synthetic _load_gray failure to propagate"
    finally:
        _render_mod._load_gray = orig_load_gray

    after = set(WORK_DIR.glob("photo512_src_*.png"))
    assert before == after, f"temp file leaked on the exception path: {after - before}"


# ---------------------------------------------------------------------------
# 4. End-to-end through the real route
# ---------------------------------------------------------------------------

def _process(did, body_extra):
    from ccplotter.routes.core import app
    client = app.test_client()
    resp = client.post("/api/process", json={
        "job_type": "preview", "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}],
        "paper": "120x120", "margin": 10.0, "debug": True,
        **body_extra,
    })
    return resp


def _info_json_for(did):
    hits = list(WORK_DIR.glob(f"src_{did}_render_*.svg.info.json"))
    if not hits:
        raise SkipTest(f"no .info.json sidecar found for {did} -- render.py's trace_info write path changed")
    import json
    return json.loads(hits[0].read_text())


def _points_of(svg_text):
    pts = []
    for ps in re.findall(r'points="([^"]+)"', svg_text):
        for tok in ps.split():
            if ',' in tok:
                x, y = tok.split(',')
                pts.append((round(float(x), 3), round(float(y), 3)))
    return sorted(pts)


def _process_and_get_points(did, body_extra):
    """Runs one real /api/process call and returns the routed SVG's parsed
    polyline points immediately (before the next call can overwrite
    WORK_DIR's fixed-name layout.routed.svg)."""
    resp = _process(did, body_extra)
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    routed_name = body.get("debug_files", {}).get("routed_svg")
    if not routed_name:
        raise SkipTest("no debug_files.routed_svg in response -- debug plumbing changed")
    text = (WORK_DIR / routed_name).read_text(encoding="utf-8")
    return body, _points_of(text)


def test_photo512_forces_recipe_over_caller_overrides():
    did = "photo512forcetest"
    _make_photo(WORK / f"src_{did}.jpg", w=1600, h=1000)
    # Deliberately WRONG values the recipe must override: clahe on, tone
    # dense, a resolution far below the tracev3 cap.
    resp = _process(did, {"style": "photo512", "clahe": True, "tone": "dense", "resolution": 300})
    assert resp.status_code == 200, resp.get_json()
    info = _info_json_for(did)
    assert info["mode"] == "tracev3", info
    # requested_res in the sidecar reflects what render.py actually saw for
    # "resolution" -- must NOT be the caller's 300 (proves the override
    # fired), and can only ever be >= the recipe's own 3000 (scale_aware_v2
    # can boost it further upward, never below its own base).
    assert info["requested_res"] != 300, info
    assert info["requested_res"] >= PHOTO512_RECIPE["resolution"], info
    body = resp.get_json()
    assert body.get("stroke_count", 0) > 0, body


def test_photo512_forces_native_res_false_even_when_caller_overrides():
    """native_res=True sent alongside style="photo512" must be forced
    back to False. There is no sidecar field to read this from directly, so
    prove it by FORCE: a photo512 request WITH native_res=True in the body
    must produce IDENTICAL geometry to the same request WITHOUT it -- if the
    override were not firing, native_res=True would make render.py's
    tracev3 branch load the ORIGINAL full-res source via
    _load_native_gray(raw_path) in parallel with the 512-pre-shrunk `img`
    (render.py's tracev3 branch, `ng = _load_native_gray(raw_path) if
    opts.get("native_res") else None`), which changes the mask/trace path."""
    did_a = "photo512nativeoff"
    did_b = "photo512nativeon"
    _make_photo(WORK / f"src_{did_a}.jpg", w=1400, h=900)
    _make_photo(WORK / f"src_{did_b}.jpg", w=1400, h=900)  # identical content, distinct id
    _, pts_a = _process_and_get_points(did_a, {"style": "photo512"})
    _, pts_b = _process_and_get_points(did_b, {"style": "photo512", "native_res": True})
    assert pts_a == pts_b, "native_res=True from the caller changed photo512 output -- D7 override not firing"


def test_photo512_route_shrinks_source_to_512():
    """Observes the `side` argument render.py passes to
    _prepare_shorter_side_gray, via a spy that wraps the real implementation
    (still calls through, so the route completes normally), so the test
    cannot pass on file cleanup alone. NOTE: scale_aware_v2 (default-on for
    tracev3) runs a cheap PRELIMINARY trace pass before the final one
    (composite.py's _source_svg called once to learn nw/nh, once for the real
    render), so TWO calls is the correct, expected count here -- both must
    still be 512."""
    did = "photo512sidecheck"
    _make_photo(WORK / f"src_{did}.jpg", w=2000, h=1400)
    calls = []
    orig = _render_mod._prepare_shorter_side_gray

    def _spy(raw_path, side):
        calls.append(side)
        return orig(raw_path, side)

    _render_mod._prepare_shorter_side_gray = _spy
    try:
        resp = _process(did, {"style": "photo512"})
        assert resp.status_code == 200, resp.get_json()
    finally:
        _render_mod._prepare_shorter_side_gray = orig
    assert calls, "the pre-shrink step never fired at all"
    assert all(c == 512 for c in calls), calls


def test_route_cleans_up_temp_file_after_success():
    """Captures the REAL temp path render.py creates via a spy, then asserts
    render.py's own try/finally removed it after the route call returns (the
    test never unlinks it itself, so it exercises render.py's cleanup code)."""
    did = "photo512cleanuproute"
    _make_photo(WORK / f"src_{did}.jpg", w=1800, h=1200)
    captured = []
    orig = _render_mod._prepare_shorter_side_gray

    def _spy(raw_path, side):
        p = orig(raw_path, side)
        captured.append(p)
        return p

    _render_mod._prepare_shorter_side_gray = _spy
    try:
        resp = _process(did, {"style": "photo512"})
        assert resp.status_code == 200, resp.get_json()
    finally:
        _render_mod._prepare_shorter_side_gray = orig
    # scale_aware_v2's preliminary + final pass (see test above) each create
    # and clean up their OWN temp file -- both must be gone.
    assert captured, "the pre-shrink step never fired at all"
    leftover = [p for p in captured if p.exists()]
    assert not leftover, f"render.py did not clean up its own temp file(s): {leftover}"


def test_non_photo512_tracev3_request_never_touches_photo512_path():
    """Plain style='tracev3' (no photo512) must be completely unaffected --
    no source_side_cap side effect, no temp file, ever."""
    before = set(WORK_DIR.glob("photo512_src_*.png"))
    did = "plaintracev3test"
    _make_photo(WORK / f"src_{did}.jpg", w=1200, h=800)
    resp = _process(did, {"style": "tracev3"})
    assert resp.status_code == 200, resp.get_json()
    after = set(WORK_DIR.glob("photo512_src_*.png"))
    assert before == after
    info = _info_json_for(did)
    assert info["mode"] == "tracev3"
    # NOTE: requested_res is NOT asserted to a fixed number here -- tracev3's
    # scale_aware_v2 default recomputes an item-specific analytic
    # resolution from the placed size, so it legitimately varies with image/
    # paper geometry even for a plain, non-photo512 request. The load-bearing
    # check for THIS test is the temp-file absence above: the source-side
    # pre-downsample path never fires for style="tracev3".


def test_source_side_cap_cannot_be_smuggled_via_plain_tracev3_request():
    """A caller sending style="tracev3" (NOT "photo512") plus a raw
    source_side_cap=512 in the request body must be ignored -- proven by
    identical output to the same request without that key, AND by no temp
    file ever appearing. render_opts["source_side_cap"] must not be read
    unconditionally from the body, or this would trigger the pre-shrink path
    under any style."""
    did_plain = "smuggletest_plain"
    did_smuggled = "smuggletest_smuggled"
    _make_photo(WORK / f"src_{did_plain}.jpg", w=1300, h=900)
    _make_photo(WORK / f"src_{did_smuggled}.jpg", w=1300, h=900)  # identical content
    before = set(WORK_DIR.glob("photo512_src_*.png"))
    _, pts_plain = _process_and_get_points(did_plain, {"style": "tracev3"})
    _, pts_smuggled = _process_and_get_points(did_smuggled, {"style": "tracev3", "source_side_cap": 512})
    after = set(WORK_DIR.glob("photo512_src_*.png"))
    assert before == after, "source_side_cap was honored under style='tracev3' -- D6 regression"
    assert pts_plain == pts_smuggled, "source_side_cap changed tracev3 output when it should have been ignored"


# ---------------------------------------------------------------------------
# 5. LOAD-BEARING -- the mechanism actually changes the geometry, and a
#    future edit that silently removes it (e.g. deleting the
#    render_opts["source_side_cap"] line) will fail THIS test even though
#    every other test above can still pass.
# ---------------------------------------------------------------------------

def test_photo512_geometry_differs_from_equivalent_plain_tracev3():
    did_photo = "d1_photo512"
    did_plain = "d1_plaintracev3_matched"
    _make_photo(WORK / f"src_{did_photo}.jpg", w=1800, h=1200)
    _make_photo(WORK / f"src_{did_plain}.jpg", w=1800, h=1200)  # identical content
    _, pts_photo512 = _process_and_get_points(did_photo, {"style": "photo512"})
    # Every OTHER field in PHOTO512_RECIPE matched BY HAND -- the only axis
    # left different is the one under test: source_side_cap is not settable
    # by a caller under style="tracev3", so this plain request traces
    # the FULL source at the same resolution/clahe/tone/native_res photo512
    # itself forces, with no 512 pre-shrink.
    _, pts_plain = _process_and_get_points(did_plain, {
        "style": "tracev3", "clahe": PHOTO512_RECIPE["clahe"], "tone": PHOTO512_RECIPE["tone"],
        "resolution": PHOTO512_RECIPE["resolution"], "native_res": PHOTO512_RECIPE["native_res"],
    })
    assert pts_photo512 != pts_plain, (
        "photo512 produced IDENTICAL geometry to plain tracev3 at matched settings -- "
        "the source-side pre-downsample mechanism is not firing (or was silently "
        "deleted from render_opts construction)."
    )


def _run_all():
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
