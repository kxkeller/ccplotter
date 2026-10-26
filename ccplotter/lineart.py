from pathlib import Path

from ccplotter.mask import (_binarize_lineart, _gray_crush_metric, _micro_detail_mask,
                             _outline_polys, _component_fidelity_detail,
                             _skeleton_to_polylines, _smooth_polys, _stroke_px_for)
from ccplotter.svg_io import _write_paths_svg

_LINEART_AUDIT = {}   # last _lineart_svg run: strokes/closed_loops/hollowed_px/skel_px

def _lineart_svg(img, out_path: Path, opts: dict, debug_prefix: str = None) -> Path:
    """Specialized engine for clean, high-contrast line art or AI-generated
    sketches. Treats the image as already-drawn ink, avoiding Canny edges.

    Threshold modes (opts['threshold_mode']): 'otsu' (default, global auto),
    'adaptive' (local Gaussian — recovers detail on uneven lighting), 'manual'
    (fixed cutoff via opts['manual_threshold'])."""
    import cv2, numpy as np, time
    from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, binary_closing, disk
    t0 = time.time()
    _img_raw = img          # pre-blur original, for the tiny-text super-res crop

    # BILEVEL FAST PATH: AI-cleaned sources are already pure black/white line
    # art. The photo-oriented preprocessing actively destroys their small
    # detail: the denoise
    # blur drags 2-3px glyph interiors below a high manual threshold, welding
    # character strokes into blobs whose skeletons are unreadable polygon
    # squiggles. For bilevel input: no blur, midpoint threshold, no closing —
    # the ink is the ink, just trace it.
    tmode = str(opts.get("threshold_mode", "otsu")).lower()
    sensitivity_mode = (tmode == "sensitivity")
    bilevel = False                                # set in the legacy branch; gap-close reads it

    # INK-SENSITIVITY path (faithful line-art binarization): no blur, hysteresis
    # + line-shape test so faint anti-aliased contours survive. Used by the
    # Faithful Line Art preset and the Ink Sensitivity control; the AI-clean
    # manual flow does not use it.
    if sensitivity_mode:
        mask = _binarize_lineart(img, str(opts.get("ink_sensitivity", "sensitive")))
    else:
        # BILEVEL FAST PATH: AI-cleaned sources are already pure black/white line
        # art. The photo-oriented preprocessing actively destroys their small
        # detail: the denoise
        # blur drags 2-3px glyph interiors below a high manual threshold, welding
        # character strokes into blobs whose skeletons are unreadable polygon
        # squiggles. For bilevel input: no blur, midpoint threshold, no closing.
        bilevel = float(np.mean((img < 40) | (img > 215))) > 0.9
        if not bilevel:
            img = cv2.GaussianBlur(img, (3, 3), 0)   # light denoise (photos/soft grays only)
        if bilevel:
            _, mask = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
        elif tmode == "manual":
            t = int(opts.get("manual_threshold", 128))
            _, mask = cv2.threshold(img, t, 255, cv2.THRESH_BINARY_INV)
        elif tmode == "adaptive":
            block = int(opts.get("adaptive_block", 25))
            if block % 2 == 0: block += 1
            block = max(3, block)
            C = int(opts.get("adaptive_c", 7))
            mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, C)
        else:  # otsu
            _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = mask > 0
    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_mask_raw.png", (mask.astype(np.uint8) * 255))

    # Remove tiny garbage but keep small details like windows/text
    min_area = int(opts.get("lineart_min_area", 20))
    if min_area > 0:
        mask = remove_small_objects(mask, min_size=min_area)

    # BUBBLE FIX: fill tiny white holes INSIDE strokes (anti-aliasing speckle in
    # AI-generated line art). Without this the skeleton loops around each hole,
    # producing "bubbles" along outlines. -1 = auto. The auto threshold MUST
    # stay small: AI text/glyphs are outlined shapes with white interiors only
    # 10-100px² — fill those and the skeleton collapses each character into an
    # unreadable squiggle blob. AA pinholes are ≤ a few px even upscaled, so 8px²·scale²
    # still catches them while every legible counter survives.
    hole_px = int(opts.get("hole_fill", -1))
    if hole_px < 0:
        res_scale = max(img.shape) / 1000.0
        hole_px = round(8 * res_scale * res_scale)
    if hole_px > 0:
        mask = remove_small_holes(mask, area_threshold=hole_px)

    # Close tiny gaps in generated lines (skip for bilevel/sensitivity: ink is
    # solid, and closing welds adjacent glyph strokes / parallel faint lines)
    close_px = 0 if (bilevel or sensitivity_mode) else int(opts.get("lineart_gap_close", 1))
    if close_px > 0:
        mask = binary_closing(mask, footprint=disk(close_px))
    # FILLED-SHAPE HOLLOWING (preserve mechanical detail): a solid filled region
    # — landing-gear wheels, dark windows, logos — skeletonizes to a single dot
    # and VANISHES. Erase only the interior deeper than ~a pen radius, leaving an
    # outline band that skeletonizes into the real shape (e.g. wheel circles).
    # Thin strokes have no deep interior, so linework is untouched.
    n_hollowed = 0
    if int(opts.get("hollow_fills", 1)):
        res_sc0 = max(img.shape) / 1000.0
        fill_limit = max(3.0, 2.5 * res_sc0)       # ~one pen-stroke half-width
        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        deep = dt > fill_limit
        n_hollowed = int(deep.sum())
        if n_hollowed:
            mask = mask & (dt <= fill_limit)
    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_mask_clean.png", (mask.astype(np.uint8) * 255))

    # Otsu shadow-crush metric (same metric as tracev3; see the
    # _gray_crush_metric docstring). Uses
    # _img_raw (pre-blur, matches debug_gray.png) against the final cleaned
    # mask (post-hollowing, matches debug_mask_clean.png).
    _pw_mm0 = float(opts.get("paper_w_mm", 180.0)); _mg_mm0 = float(opts.get("margin_mm", 10.0))
    _mm_per_px0 = max(1e-6, (_pw_mm0 - 2 * _mg_mm0) / max(img.shape))
    _gray_crush_fraction = _gray_crush_metric(_img_raw, mask, _mm_per_px0)

    # MICRO-DETAIL PROTECTION (preserve_text): protect tiny text/label/rivet/texture
    # marks so their fragile spurs survive pruning. DETAIL-FIRST: default 'balanced'
    # protects every small mark (rich detail). Speckle removal is a SEPARATE opt-in
    # control (noise_filter, default 'off') so cleanup never silently deletes detail.
    # preserve_text 'off' → protect_mask=None → the plain clean skeleton path.
    protect_mask, n_protect = None, 0
    _pt = str(opts.get("preserve_text", "balanced")).lower()
    if _pt not in ("off", "false", "0", ""):
        _lvl = "aggressive" if _pt == "aggressive" else "balanced"   # legacy 'on'/'true' → balanced
        protect_mask, n_protect = _micro_detail_mask(
            mask, level=_lvl, noise_filter=str(opts.get("noise_filter", "off")),
            debug_prefix=debug_prefix,
            r_factor=float(opts.get("md_radius_factor", 2.0)),
            max_support=int(opts.get("md_max_support", 14)))

    # TEXT/GLYPH OUTLINE PASS: skeletonization destroys letterforms — a glyph's
    # medial axis is not its shape, so bowls/junctions in A/P/R/E collapse
    # ("A"->"λ", "P"->"r"), and it can't be fixed by resolution. The binary mask
    # is legible, so OUTLINE-trace the small text/detail components (exactly what
    # _micro_detail_mask flags) and skeletonize ONLY the remaining linework.
    # Outlines are appended un-smoothed (crisp corners). Gated on preserve_text:
    # 'off' leaves protect_mask None → skel_mask = mask (no outline pass).
    # NOTE: outline-tracing text makes real anti-aliased placard/label text look
    # like hollow "bubble" letters. On real content the SKELETON centerline reads
    # better and is faithful to the source's thin strokes, so text_outline
    # defaults OFF (skeleton path). The outline pass is available
    # (text_outline='on') for crisp tiny synthetic text.
    text_outline_polys = []
    skel_mask = mask
    _text_outline_on = str(opts.get("text_outline", "off")).lower() in ("on", "true", "1")
    if protect_mask is not None and _text_outline_on and protect_mask.any():
        text_outline_polys = _outline_polys(mask & protect_mask)
        skel_mask = mask & ~protect_mask

    # SMALL-DETAIL FIDELITY (see _component_fidelity_detail). Not a region
    # classifier: per SMALL ink component, simulate skeleton vs local super-res
    # contour, keep whichever actually reproduces that component's own ink shape
    # better once rendered at pen width. 'off' leaves skel_mask as built above.
    tiny_polys, tiny_min_h = [], 0.0
    _fidelity_on = str(opts.get("small_detail_fidelity", "on")).lower() not in ("off", "false", "0", "")
    _fidelity_checked = _fidelity_switched = 0
    if _fidelity_on:
        _fill_mm = float(opts.get("fill_mm", 0.15))
        _pw_mm = float(opts.get("paper_w_mm", 180.0)); _mg_mm = float(opts.get("margin_mm", 10.0))
        _fill_px = max(1.5, _fill_mm / max(1e-6, (_pw_mm - 2 * _mg_mm) / img.shape[1]))
        _cache = opts.get("_fidelity_cache")   # optional dict, threaded by the caller for reuse
        _dbg_scores = [] if debug_prefix else None
        # Opt-in: 'strict' (default) = isolation pre-filter active. 'trial' =
        # pre-filter removed, every area-gate-passing component enters the
        # fidelity trial directly. See the isolation= docstring on
        # _component_fidelity_detail.
        _isolation_mode = str(opts.get("fidelity_isolation", "strict")).lower()
        if _isolation_mode not in ("strict", "trial"):
            _isolation_mode = "strict"
        excl_mask, tiny_polys, _fidelity_checked, _fidelity_switched, tiny_min_h = _component_fidelity_detail(
            mask, _img_raw, _fill_px, cache=_cache, debug_scores=_dbg_scores, isolation=_isolation_mode)
        if debug_prefix and _dbg_scores is not None:
            import json as _json
            with open(f"{debug_prefix}_fidelity_scores.json", "w") as _f:
                _json.dump(_dbg_scores, _f)
        if excl_mask.any():
            skel_mask = skel_mask & ~excl_mask

    skel = skeletonize(skel_mask)

    # SKELETON COMPLEXITY GUARD — scales with working resolution (a dense but
    # legitimate drawing at 2000px has ~4x the skeleton pixels of 1000px).
    skel_pixels = int(skel.sum())
    res_sc = max(img.shape) / 1000.0
    skel_cap = int(120000 * max(1.0, res_sc * res_sc))
    if skel_pixels > skel_cap:
        raise RuntimeError(f"Line art too complex ({skel_pixels}px > {skel_cap}). Try higher Noise Cull or lower Resolution.")
    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_skeleton.png", (skel.astype(np.uint8) * 255))

    prune_len = int(opts.get("prune_len", 8))
    # With the text-outline pass, text pixels are already gone from `skel`;
    # protect_mask is only still needed for spur protection if we DIDN'T outline.
    _skel_protect = None if text_outline_polys else protect_mask
    polys = _skeleton_to_polylines(skel, prune_len=prune_len, protect_mask=_skel_protect)

    # Anti-jitter: window scales with working resolution (~2px of smoothing per
    # 1000px of image) so the wobble fix is resolution-independent.
    sm_k = int(opts.get("smooth_px", 0)) or max(3, (int(2 * res_sc) | 1))
    polys = _smooth_polys(polys, k=min(sm_k, 9))
    # Glyph outlines + tiny-text contours keep crisp corners — append AFTER smoothing.
    _n_skel_polys = len(polys)
    polys = polys + text_outline_polys + tiny_polys

    # Audit counts (closed loops = mechanical details like wheels/windows kept).
    closed = sum(1 for p in polys if len(p) >= 4 and
                 (abs(p[0][0] - p[-1][0]) + abs(p[0][1] - p[-1][1])) <= 2.0)
    global _LINEART_AUDIT
    _LINEART_AUDIT = {"strokes": len(polys), "closed_loops": closed,
                      "hollowed_px": n_hollowed, "skel_px": skel_pixels,
                      "text_outlines": len(text_outline_polys),
                      # Rendering decisions are always reported: this on/off and the
                      # counts below are what info.json/the UI stamp.
                      "small_detail_fidelity": "on" if _fidelity_on else "off",
                      "small_detail_checked": _fidelity_checked,
                      "small_detail_switched": _fidelity_switched,
                      "small_detail_min_switched_px": round(tiny_min_h, 1),
                      # Which isolation mode ran
                      # ('strict' when fidelity is off too, since the key is only
                      # consulted inside the fidelity branch).
                      "fidelity_isolation": _isolation_mode if _fidelity_on else "strict",
                      "gray_crush_fraction": round(_gray_crush_fraction, 4)}

    # FIDELITY DEBUG: which components switched to the contour candidate, shown
    # directly as a mask overlay -- the decision is per-component, not per-region.
    if debug_prefix and _fidelity_on:
        try:
            ov = cv2.cvtColor((mask.astype(np.uint8) * 60), cv2.COLOR_GRAY2BGR)
            ov[excl_mask] = (0, 0, 255)   # red = this ink used the contour candidate
            cv2.imwrite(f"{debug_prefix}_fidelity_switched.png", ov)
        except Exception as _e:
            print(f"  [fidelity] debug save failed: {_e}")

    _write_paths_svg(out_path, img.shape[1], img.shape[0], _stroke_px_for(opts, img.shape[1]), polylines=polys)
    print(f"    [perf] _lineart_svg: {time.time()-t0:.2f}s, skel_px {skel_pixels}, thresh={tmode}, "
          f"closed_loops={closed}, hollowed={n_hollowed}px, protected_microdetail={n_protect}, "
          f"text_outlines={len(text_outline_polys)}, small_detail_fidelity={'on' if _fidelity_on else 'off'}("
          f"{_fidelity_checked} checked, {_fidelity_switched} switched, "
          f"min_switched {tiny_min_h:.0f}px)")
    return out_path
