import json
import shlex
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import vpype_cli

import ccplotter.lineart as _lineart_mod
import ccplotter.trace_v3 as _trace_v3_mod
import ccplotter.pfm_native as _pfm_native_mod

from ccplotter.atomic_io import atomic_open, atomic_target, atomic_write_text
from ccplotter.config import WORK_DIR, MODE_MAX_RES, SCALE_AWARE_V2_NATIVE_CAP
from ccplotter.mask import (_load_gray, _stroke_px_for, _skeleton_to_polylines,
                             trace_to_svg, _load_native_gray, _prepare_shorter_side_gray)
from ccplotter.svg_io import _write_paths_svg
from ccplotter.sketch import (_generate_stipple_points, _solve_tsp, _split_route_on_jumps,
                                _centerline_svg, _hatch_geom, _lsd_svg, _sketch_svg)
from ccplotter.trace_v3 import _lineart_quality_check, _trace_svg
from ccplotter.lineart import _lineart_svg
from ccplotter.hybrid import _hybrid_svg


def resolve_trace_res(opts: dict):
    """The resolution policy's (requested_res, mode_cap, effective_res) for a render request, BEFORE the
    plotready/lineart/sketchv2 native-floor step in render_raster_to_svg (which never applies to tracev3).

    SINGLE SOURCE for "what working resolution does a trace of these opts run at". render_raster_to_svg
    calls it; composite.py's opt-in `prelim_res_basis` calls it to learn the PRELIMINARY trace's effective
    resolution (the tracev3 cap, MODE_MAX_RES["tracev3"]) instead of hard-coding it.

    Scale-adaptive composite (opt-in): composite.py may ask for a resolution above the normal per-mode cap
    for THIS item's final trace pass only, when its analytic fill_px would otherwise quantize too thin. The
    override is clamped to SCALE_AWARE_V2_NATIVE_CAP, which is separate from the general native_res
    NATIVE_CAP (see config.py). Without the override key, mode_cap is the plain per-mode cap."""
    mode = opts.get("mode", "outline")
    mode_cap = MODE_MAX_RES.get(mode, 1000)
    if mode == "tracev3" and opts.get("_tracev3_res_cap_override"):
        mode_cap = max(mode_cap, min(int(opts["_tracev3_res_cap_override"]), SCALE_AWARE_V2_NATIVE_CAP))
    requested_res = int(opts.get("resolution", 1000))
    return requested_res, mode_cap, min(requested_res, mode_cap)


def render_raster_to_svg(raw_path: Path, out_path: Path, opts: dict, debug_prefix: str = None, pen_id: str = None) -> Path:
    # pen_id (provenance only): the RESOLVED pen id this job is plotting with,
    # used for audit-dict stamping (below) and never read by any generation
    # code path in this function. It is a separate kwarg rather than an opts key
    # so it stays out of the cache-key hash (see composite.py's _source_svg).
    import time, cv2, numpy as np
    from skimage.morphology import skeletonize
    t_start = time.time()
    mode = opts.get("mode", "outline")

    # Clear any stale debug intermediates (e.g. from a different mode) so the UI
    # never links a leftover edges.png while showing a lineart run, etc.
    if debug_prefix:
        for old in WORK_DIR.glob("debug_*.png"):
            try: old.unlink()
            except: pass
        for old in WORK_DIR.glob("debug_*.svg"):
            try: old.unlink()
            except: pass

    # 1. Resolution policy — explicit and never silently destructive.
    #    Line-art tracers (plotready/lineart) must NEVER downscale the source
    #    below native: the generic Resolution default (1000) was quietly
    #    tracing a 38%-size blur of 2368px AI sources, destroying small text
    #    before the tracer ran. Bilevel (already-B/W) sources additionally
    #    trace at 2x native — upsampled strokes are thicker, skeletons come
    #    out smoother, and tiny glyphs survive. The full decision is written
    #    to <render>.info.json so the UI can SHOW what happened.
    # (mode_cap / v2 cap-override policy: see resolve_trace_res above.)
    requested_res, mode_cap, effective_res = resolve_trace_res(opts)
    native_px, src_bilevel = None, False
    if mode in ("plotready", "lineart", "sketchv2"):
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(raw_path) as _im:
                native_px = max(_im.size)
                ga = np.asarray(_im.convert("L"))
            src_bilevel = float(np.mean((ga < 40) | (ga > 215))) > 0.9
            floor = native_px * 2 if src_bilevel else native_px
            effective_res = min(max(effective_res, floor), mode_cap)
        except Exception:
            pass
    trace_info = {"mode": mode, "requested_res": requested_res, "mode_cap": mode_cap,
                  "native_px": native_px, "effective_res": effective_res,
                  "bilevel_source": src_bilevel,
                  "downscaled": bool(native_px and effective_res < native_px),
                  "upsampled": bool(native_px and effective_res > native_px)}
    if opts.get("front_end") == "bw_xdog":
        # B/W photo mode only: state the fixed front end in info.json so the
        # abstraction is visible. Absent for every other mode.
        from ccplotter import bw_photo as _bwp
        trace_info["front_end"] = {"name": "bw_xdog", "side": _bwp.BW_XDOG_SIDE, "p": _bwp.BW_XDOG_P,
                                   "k": _bwp.BW_XDOG_K, "eps": _bwp.BW_XDOG_EPS, "phi": _bwp.BW_XDOG_PHI}
        if opts.get("bw_xdog_side") not in (None, "", 512):   # an explicit 512 is the default side: no extra keys
            # Opt-in higher-detail side: "side" states the side ACTUALLY used,
            # plus request / source / interpolation facts. sigma formula NOT rescaled (0.8 px of the used raster).
            _su, _short, _interp = _bwp.resolve_bw_xdog_side(raw_path, opts["bw_xdog_side"])
            trace_info["front_end"].update({
                "side": _su, "side_used": _su, "side_requested": opts["bw_xdog_side"],
                "src_short_side": _short, "interpolated": _interp,
                "sigma_px": _bwp.BW_XDOG_SIGMA_PX, "sigma_rescaled": False})
        # Opt-in mask overrides, recorded ONLY when active (absent otherwise).
        if opts.get("bw_clahe_clip") or opts.get("bw_cut") or opts.get("clahe") is False:
            trace_info["bw_overrides"] = {
                "clahe": bool(opts.get("clahe", False)),
                "clahe_clip": float(opts.get("bw_clahe_clip") or (2.5 if opts.get("clahe") else 0.0)),
                "cut": opts.get("bw_cut") or "otsu"}
    try:
        atomic_write_text(Path(str(out_path) + ".info.json"), json.dumps(trace_info))
    except Exception:
        pass
    print(f"  [pipe] mode={mode} requested_res={requested_res} effective_res={effective_res} "
          f"native={native_px} bilevel={src_bilevel}")

    ch, gamma, crop = opts.get("channel", "auto"), float(opts.get("gamma", 1.0)), opts.get("crop")

    # 2. Load and Preprocess (with early resize)
    # Photo-512 mode (config.PHOTO512_RECIPE) only: pre-shrink the SOURCE to a
    # fixed shorter side before _load_gray's own resize/channel-select runs --
    # see _prepare_shorter_side_gray's docstring. `source_side_cap` is absent
    # (falsy) for every other request, so this block is a no-op and
    # _load_gray receives the original raw_path, so every mode other than
    # Photo-512 is unaffected.
    _p512_src_cap = opts.get("source_side_cap")
    _p512_tmp = None
    _load_gray_src = raw_path
    if _p512_src_cap:
        _p512_tmp = _prepare_shorter_side_gray(raw_path, int(_p512_src_cap))
        _load_gray_src = _p512_tmp
    elif opts.get("front_end") == "bw_xdog":
        # B/W photo mode (ccplotter/bw_photo.py, config.BW_PHOTO_RECIPE) only:
        # replace the source with the locked XDoG p02_e55 @ 512 raster before
        # _load_gray, exactly as the m09 locked-cell jobs fed it. The raster
        # is also kept as <debug_prefix>_xdog.png when debug output is on.
        # `front_end` is None/absent for every other request -- no-op.
        from ccplotter.bw_photo import prepare_bw_xdog_source
        if opts.get("bw_xdog_side") not in (None, "", 512):
            _p512_tmp = prepare_bw_xdog_source(raw_path, WORK_DIR, debug_prefix, side=opts["bw_xdog_side"])
        else:
            _p512_tmp = prepare_bw_xdog_source(raw_path, WORK_DIR, debug_prefix)
        _load_gray_src = _p512_tmp
    try:
        # The temp pre-shrink file must be removed even if _load_gray raises:
        # WORK_DIR has no glob-based purge for photo512_src_*.png, so a leaked
        # one would stay forever.
        # bw_clahe_clip: bwphoto "soft" opt-in only (routes/core.py); absent
        # for every other request -> the default clip of 2.5.
        _clip_kw = {"clahe_clip": float(opts["bw_clahe_clip"])} if opts.get("bw_clahe_clip") else {}
        img = _load_gray(_load_gray_src, effective_res, opts.get("clahe", False), ch, crop, "blue" if opts.get("isolate_blue") else None, **_clip_kw)
    finally:
        if _p512_tmp is not None:
            try: _p512_tmp.unlink()
            except Exception: pass
    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_gray.png", img)

    temp_img = WORK_DIR / f"temp_{uuid4().hex[:4]}.png"
    from PIL import Image; Image.fromarray(img).save(temp_img)
    import shlex

    if mode == "tsp":
        true_spacing = max(2.0, float(opts.get("spacing", 1.0)) + 1.0)
        t0 = time.time()
        pts = _generate_stipple_points(img, true_spacing, gamma, float(opts.get("contrast", 1.0)))
        if not pts: raise RuntimeError("TSP: no dots generated (image too light).")

        # Max TSP point cap
        max_pts = 15000
        if len(pts) > max_pts:
            print(f"  [warn] TSP capped from {len(pts)} to {max_pts} points")
            pts = pts[:max_pts]

        route = _solve_tsp(pts)
        print(f"  [perf] tsp: solved {len(pts)} pts in {time.time()-t0:.2f}s")

        strokes = _split_route_on_jumps(route, max(true_spacing * 1.5, 4.0))
        temp_route = WORK_DIR / f"temp_{uuid4().hex[:4]}.svg"
        _write_paths_svg(temp_route, img.shape[1], img.shape[0], _stroke_px_for(opts, img.shape[1]), polylines=strokes)
        tsp_min = max(0.0, float(opts.get("tsp_min_len", 1.5)))
        with atomic_target(out_path) as tmp_out:
            vpype_cli.execute(f"read {shlex.quote(str(temp_route))} filter --min-length {tsp_min}mm linesimplify --tolerance 0.1mm linesort penwidth {float(opts.get('fill_mm', 0.15))}mm write {shlex.quote(str(tmp_out))}")
        if temp_route.exists(): temp_route.unlink()

    elif mode == "flow":
        t0 = time.time()
        with atomic_target(out_path) as tmp_out:
            vpype_cli.execute(f"flow_img --max_size {effective_res} {shlex.quote(str(temp_img))} linesort penwidth {float(opts.get('fill_mm', 0.15))}mm write {shlex.quote(str(tmp_out))}")
        print(f"  [perf] flow: {time.time()-t0:.2f}s")

    elif mode == "centerline":
        t0 = time.time()
        temp_cl = WORK_DIR / f"temp_{uuid4().hex[:4]}.svg"
        _centerline_svg(img, temp_cl, opts)
        with atomic_target(out_path) as tmp_out:
            vpype_cli.execute(f"read {shlex.quote(str(temp_cl))} linemerge --tolerance 0.3mm filter --min-length 1.0mm linesimplify --tolerance 0.1mm linesort penwidth {float(opts.get('fill_mm', 0.15))}mm write {shlex.quote(str(tmp_out))}")
        if temp_cl.exists(): temp_cl.unlink()
        print(f"  [perf] centerline: {time.time()-t0:.2f}s")

    elif mode == "hybrid":
        t0 = time.time()
        temp_cl = WORK_DIR / f"temp_{uuid4().hex[:4]}.svg"
        _hybrid_svg(img, temp_cl, opts, debug_prefix=debug_prefix, raw_path=raw_path)
        if debug_prefix:
            import shutil; shutil.copy(temp_cl, f"{debug_prefix}_raw_paths.svg")
        merge_tol = max(0.0, float(opts.get("vec_merge", 0.8)))
        min_len = max(0.0, float(opts.get("vec_min_len", 1.0)))
        # Small-detail mode: relax the cleanup min-length floor so tiny detail
        # strokes survive vpype filtering (they're the whole point of the lane).
        _sd = str(opts.get("small_detail", "off")).lower()
        if _sd == "balanced": min_len = min(min_len, 0.6)
        elif _sd == "preserve": min_len = min(min_len, 0.3)
        simplify_tol = max(0.02, float(opts.get("outline_simplify", 0.15)))
        with atomic_target(out_path) as tmp_out:
            vpype_cli.execute(f"read {shlex.quote(str(temp_cl))} linemerge --tolerance {merge_tol}mm filter --min-length {min_len}mm linesimplify --tolerance {simplify_tol}mm linesort penwidth {float(opts.get('fill_mm', 0.15))}mm write {shlex.quote(str(tmp_out))}")
        if temp_cl.exists(): temp_cl.unlink()
        print(f"  [perf] hybrid: {time.time()-t0:.2f}s")

    elif mode == "plotready":
        # Plot-Ready Line Art: reliable centerline tracing of ALREADY-CLEAN line
        # art. No Canny/LSD/Hybrid — just threshold → skeleton → graph-trace, with
        # detail-preserving defaults and a quality check. "If it's already good
        # line art, don't reinterpret it — just trace it cleanly."
        t0 = time.time()
        q = _lineart_quality_check(img, effective_res)
        try:
            with atomic_open(WORK_DIR / "debug_plotready_quality.json", "w") as f:
                json.dump(q, f)
        except Exception: pass
        temp_cl = WORK_DIR / f"temp_{uuid4().hex[:4]}.svg"
        try:
            _lineart_svg(img, temp_cl, opts, debug_prefix=debug_prefix)   # saves gray/mask/skeleton debug
        except RuntimeError as e:
            # Too-dense skeleton usually means the input ISN'T clean line art.
            if not q["looks_like_line_art"]:
                raise RuntimeError(f"This doesn't look like clean line art (shading {q['gray_pct']:.0f}%, "
                                   f"coverage {q['foreground_pct']:.0f}%). Clean it first (AI Clean to Line Art) "
                                   f"or use Hybrid mode.")
            raise RuntimeError(f"Line art too dense to trace at {effective_res}px. Lower Resolution or raise Noise Cull. ({e})")
        if debug_prefix:
            import shutil; shutil.copy(temp_cl, f"{debug_prefix}_raw_paths.svg")
        # Minimal, detail-preserving cleanup (no aggressive short-stroke removal).
        merge_tol = max(0.0, float(opts.get("vec_merge", 0.3)))
        min_len = max(0.0, float(opts.get("vec_min_len", 0.3)))
        simplify_tol = max(0.02, float(opts.get("outline_simplify", 0.06)))
        with atomic_target(out_path) as tmp_out:
            vpype_cli.execute(f"read {shlex.quote(str(temp_cl))} linemerge --tolerance {merge_tol}mm filter --min-length {min_len}mm linesimplify --tolerance {simplify_tol}mm linesort penwidth {float(opts.get('fill_mm', 0.15))}mm write {shlex.quote(str(tmp_out))}")
        if temp_cl.exists(): temp_cl.unlink()
        print(f"  [plotready] thick={q['stroke_thickness_px']}px fg={q['foreground_pct']}% gray={q['gray_pct']}% "
              f"comps={q['components']} clean={q['looks_like_line_art']} {time.time()-t0:.2f}s")

    elif mode == "lineart":
        t0 = time.time()
        temp_cl = WORK_DIR / f"temp_{uuid4().hex[:4]}.svg"
        _lineart_svg(img, temp_cl, opts, debug_prefix=debug_prefix)
        if debug_prefix:
            import shutil; shutil.copy(temp_cl, f"{debug_prefix}_raw_paths.svg")
        merge_tol = max(0.0, float(opts.get("vec_merge", 0.2)))
        min_len = max(0.0, float(opts.get("vec_min_len", 0.4)))
        simplify_tol = max(0.02, float(opts.get("outline_simplify", 0.08)))
        with atomic_target(out_path) as tmp_out:
            vpype_cli.execute(f"read {shlex.quote(str(temp_cl))} linemerge --tolerance {merge_tol}mm filter --min-length {min_len}mm linesimplify --tolerance {simplify_tol}mm linesort penwidth {float(opts.get('fill_mm', 0.15))}mm write {shlex.quote(str(tmp_out))}")
        if temp_cl.exists(): temp_cl.unlink()
        print(f"  [perf] lineart: {time.time()-t0:.2f}s")

    elif mode == "sketchv2":
        # Sketch V2 — width-aware faithful tracer (SEPARATE path; clean mode untouched).
        # Keep ALL detail: NO `filter --min-length` (would delete hatch strokes) and NO
        # gap-bridge `linemerge` (would weld close hatch lines). Just simplify + sort.
        t0 = time.time()
        temp_cl = WORK_DIR / f"temp_{uuid4().hex[:4]}.svg"
        _sketch_svg(img, temp_cl, opts, debug_prefix=debug_prefix)
        if debug_prefix:
            import shutil; shutil.copy(temp_cl, f"{debug_prefix}_raw_paths.svg")
        simplify_tol = max(0.02, float(opts.get("outline_simplify", 0.08)))
        with atomic_target(out_path) as tmp_out:
            vpype_cli.execute(f"read {shlex.quote(str(temp_cl))} linesimplify --tolerance {simplify_tol}mm linesort penwidth {float(opts.get('fill_mm', 0.15))}mm write {shlex.quote(str(tmp_out))}")
        if temp_cl.exists(): temp_cl.unlink()
        print(f"  [perf] sketchv2: {time.time()-t0:.2f}s")

    elif mode == "sketchfield":
        # Sketch Field -- native "Sketch Lines" PFM (opt-in, additive-only;
        # see ccplotter/pfm_native.py).
        # Consumes the ALREADY-LOADED continuous (pre-threshold) working-res
        # `img` from step 2 above -- the exact array every other raster mode
        # reads before its OWN binarization. No filter/min-length or linemerge cleanup here
        # (same "keep ALL detail" rationale as sketchv2 below): the darkness-
        # erasure loop already produced the intended chain shapes.
        t0 = time.time()
        temp_cl = WORK_DIR / f"temp_{uuid4().hex[:4]}.svg"
        _pfm_native_mod._sketch_field_svg(img, temp_cl, opts, raw_path, debug_prefix=debug_prefix)
        if debug_prefix:
            import shutil; shutil.copy(temp_cl, f"{debug_prefix}_raw_paths.svg")
        simplify_tol = max(0.02, float(opts.get("outline_simplify", 0.1)))
        # No silent 0.15 fallback here: fill_mm must agree with pfm_native's
        # erase width (same "fill_mm" key, always supplied from the pen profile
        # by routes/core.py). Raise instead of drifting, as
        # pfm_native._sketch_field_svg does.
        if "fill_mm" not in opts or opts.get("fill_mm") is None:
            raise ValueError("sketchfield: fill_mm missing from render_opts "
                              "(expected from the active pen profile)")
        with atomic_target(out_path) as tmp_out:
            vpype_cli.execute(f"read {shlex.quote(str(temp_cl))} linesimplify --tolerance {simplify_tol}mm linesort penwidth {float(opts['fill_mm'])}mm write {shlex.quote(str(tmp_out))}")
        if temp_cl.exists(): temp_cl.unlink()
        print(f"  [perf] sketchfield: {time.time()-t0:.2f}s")

    elif mode == "tracev3":
        # Trace V3 — source-faithful ink COVERAGE via DT level-set passes (pen =
        # endmill, mask = the spec). fill_px = the active pen's real solid-fill
        # spacing, plumbed from the pen profile at the call site.
        t0 = time.time()
        temp_cl = WORK_DIR / f"temp_{uuid4().hex[:4]}.svg"
        # Native-res thresholding is OPT-IN (native_res, default off). On UPSCALE
        # sources it adds staircase + fragmentation for no real detail gain (the
        # source has none beyond native); the default path thresholds the smooth
        # interpolated working-res image, which gives cleaner strokes.
        ng = _load_native_gray(raw_path) if opts.get("native_res") else None
        # Trace-boundary provenance (metadata only; both calls never raise): hash the
        # post-CLAHE gray raster as it ENTERS the trace, reset the last-mask record.
        from ccplotter import trace_boundary as _tbd
        _tbd.reset_mask_record()
        _tb_gray = _tbd.gray_record(img, opts.get("clahe", False), opts.get("bw_clahe_clip"))
        _trace_svg(img, temp_cl, opts, debug_prefix=debug_prefix, native_gray=ng)
        with atomic_target(out_path) as tmp_out:
            vpype_cli.execute(f"read {shlex.quote(str(temp_cl))} linesort penwidth {float(opts.get('fill_mm', 0.15))}mm write {shlex.quote(str(tmp_out))}")
        # connector_mask_check: trace_v3 wrote the final-mask sidecar beside the temp SVG; hand it
        # to the real render SVG. Exists only under the private `_stage_e_mask` key (absent: no-op).
        _sc = Path(str(temp_cl) + ".mask.npz")
        if _sc.exists(): _sc.replace(Path(str(out_path) + ".mask.npz"))
        if temp_cl.exists(): temp_cl.unlink()
        _tbd.write_render_sidecar(out_path, opts, raw_path, _tb_gray, _trace_v3_mod._TRACE_AUDIT)
        print(f"  [perf] tracev3: {time.time()-t0:.2f}s")

    elif mode in ("outline", "sketch"):
        t0 = time.time()
        smooth = cv2.bilateralFilter(img, 9, 75, 75)
        if opts.get("auto_edges", False):
            hi, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            edges = cv2.Canny(smooth, int(0.5 * hi), int(hi))
        else:
            edges = cv2.Canny(smooth, int(opts.get("canny_low", 80)), int(opts.get("canny_high", 160)))

        close_iters = int(opts.get("outline_close", 1))
        if close_iters > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
        if debug_prefix:
            cv2.imwrite(f"{debug_prefix}_edges.png", edges)

        # 3. SKELETON COMPLEXITY GUARD
        skel = skeletonize(edges > 0)
        skel_pixels = int(skel.sum())
        if skel_pixels > 120000:
            raise RuntimeError(f"Skeleton too complex ({skel_pixels}px). Try lower resolution or higher Noise Cull.")
        if debug_prefix:
            cv2.imwrite(f"{debug_prefix}_skeleton.png", (skel.astype(np.uint8) * 255))

        polys = _skeleton_to_polylines(skel, prune_len=int(opts.get("prune_len", 12)))

        if mode == "sketch":
            px_spacing = max(4.0, float(opts.get("spacing", 1.0)) * (effective_res / 200.0))
            hlines = _hatch_geom(img, spacing=px_spacing, levels=int(opts.get("levels", 2)), contrast=float(opts.get("contrast", 1.0)))
            polys.extend(hlines)

        temp_cl = WORK_DIR / f"temp_{uuid4().hex[:4]}.svg"
        _write_paths_svg(temp_cl, img.shape[1], img.shape[0], _stroke_px_for(opts, img.shape[1]), polylines=polys)
        if debug_prefix:
            import shutil; shutil.copy(temp_cl, f"{debug_prefix}_raw_paths.svg")
        merge_tol = max(0.0, float(opts.get("vec_merge", 0.5)))
        min_len = max(0.0, float(opts.get("vec_min_len", 1.5)))
        simplify_tol = max(0.05, float(opts.get("outline_simplify", 0.3)))
        with atomic_target(out_path) as tmp_out:
            vpype_cli.execute(f"read {shlex.quote(str(temp_cl))} linemerge --tolerance {merge_tol}mm filter --min-length {min_len}mm linesimplify --tolerance {simplify_tol}mm linesort penwidth {float(opts.get('fill_mm', 0.15))}mm write {shlex.quote(str(tmp_out))}")
        if temp_cl.exists(): temp_cl.unlink()
        print(f"  [perf] {mode}: {time.time()-t0:.2f}s, skel_px {skel_pixels}, polys {len(polys)}")

    elif mode == "schematic":
        t0 = time.time()
        temp_lsd = WORK_DIR / f"temp_{uuid4().hex[:4]}.svg"
        _lsd_svg(img, temp_lsd, opts, min_len=int(opts.get("lsd_min", 15)))
        with atomic_target(out_path) as tmp_out:
            vpype_cli.execute(f"read {shlex.quote(str(temp_lsd))} linesimplify --tolerance 0.1mm linesort penwidth {float(opts.get('fill_mm', 0.15))}mm write {shlex.quote(str(tmp_out))}")
        if temp_lsd.exists(): temp_lsd.unlink()
        print(f"  [perf] schematic: {time.time()-t0:.2f}s")

    else:
        raw_svg = WORK_DIR / f"raw_inkscape_{uuid4().hex[:4]}.svg"
        trace_to_svg(temp_img, raw_svg, opts.get("speckle", 6))
        with atomic_target(out_path) as tmp_out:
            vpype_cli.execute(f"read {shlex.quote(str(raw_svg))} linesimplify --tolerance 0.1mm linesort penwidth {float(opts.get('fill_mm', 0.15))}mm write {shlex.quote(str(tmp_out))}")
        if raw_svg.exists(): raw_svg.unlink()

    if temp_img.exists(): temp_img.unlink()
    # Merge line-art audit counts (closed loops kept, fills hollowed) into the
    # info.json written above — populated only now that tracing has run.
    if mode in ("plotready", "lineart") and _lineart_mod._LINEART_AUDIT:
        try:
            ip = Path(str(out_path) + ".info.json")
            info = json.loads(ip.read_text()) if ip.exists() else {}
            info["audit"] = dict(_lineart_mod._LINEART_AUDIT)
            atomic_write_text(ip, json.dumps(info))
        except Exception: pass
    # Sketch Field (native PFM) seed/params/generation stats -> info.json,
    # surfaced as trace_info.audit exactly like the other modes.
    if mode == "sketchfield" and _pfm_native_mod._PFM_AUDIT:
        try:
            ip = Path(str(out_path) + ".info.json")
            info = json.loads(ip.read_text()) if ip.exists() else {}
            info["audit"] = dict(_pfm_native_mod._PFM_AUDIT)
            atomic_write_text(ip, json.dumps(info))
        except Exception: pass
    # Trace V3 (tracev3) dot-cull / lifts / DT stats -> info.json, surfaced as trace_info.
    if mode == "tracev3" and _trace_v3_mod._TRACE_AUDIT:
        try:
            ip = Path(str(out_path) + ".info.json")
            info = json.loads(ip.read_text()) if ip.exists() else {}
            info["audit"] = dict(_trace_v3_mod._TRACE_AUDIT)
            # Provenance only: stamp the RESOLVED pen_id and RESOLVED fill_mm this
            # trace actually ran with, alongside the fill_px/mm_per_px/levels/
            # passes fields, so a mismatch between the ambient active pen and the
            # intended pen is visible in the artifact. Neither field is read by
            # any generation code.
            info["audit"]["pen_id_resolved"] = pen_id
            info["audit"]["fill_mm_resolved"] = opts.get("fill_mm")
            atomic_write_text(ip, json.dumps(info))
        except Exception: pass
    print(f"  [perf] total {mode}: {time.time()-t_start:.2f}s")
    return out_path
