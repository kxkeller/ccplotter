from pathlib import Path

from ccplotter.config import WORK_DIR
from ccplotter.mask import (_binarize_mask, _skeleton_to_polylines, _smooth_polys,
                             _stroke_px_for, _gray_crush_metric, _load_gray,
                             _load_native_gray)
from ccplotter.svg_io import _write_paths_svg
from ccplotter.sketch import _sketch_svg
from ccplotter.lineart import _lineart_svg

def _lineart_quality_check(img, effective_res):
    """Inspect a bitmap and report whether it looks like clean line art — used by
    Plot-Ready Line Art to warn (not block) when the input isn't actually clean.
    General/deterministic; no subject knowledge."""
    import cv2, numpy as np
    bg = float(np.percentile(img, 85))                       # background brightness
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mb = mask > 0
    cov = float(mb.mean()) * 100.0                           # foreground coverage %
    gray = float(((img > 60) & (img < 200)).mean()) * 100.0  # mid-gray (shading) %
    if mb.any():
        dt = cv2.distanceTransform(mb.astype(np.uint8), cv2.DIST_L2, 3)
        thick = float(2.0 * dt[mb].mean())                   # ~mean stroke thickness px
    else:
        thick = 0.0
    n, _, _, _ = cv2.connectedComponentsWithStats(mb.astype(np.uint8), 8)
    comps = int(n - 1)
    thin_t = max(1.5, effective_res / 1600.0)
    thick_t = max(8.0, effective_res / 220.0)
    clean = (bg >= 190) and (cov < 35) and (gray < 18)
    warnings = []
    if gray >= 18:
        warnings.append(("warning", f"This image has lots of gray/shading ({gray:.0f}%). Clean it to line art first (AI Clean to Line Art) for best results."))
    if cov >= 40:
        warnings.append(("warning", f"High foreground coverage ({cov:.0f}%) — this may be a filled/shaded image, not line art."))
    if 0 < thick < thin_t:
        warnings.append(("okay", "Very thin / broken strokes — raise resolution or gap repair if details break up."))
    if thick > thick_t:
        warnings.append(("okay", "Very thick strokes — centerline tracing may simplify some detail."))
    if clean and not warnings:
        warnings.append(("okay", "Looks like clean line art — tracing should be faithful."))
    return {
        "background_brightness": round(bg, 1), "foreground_pct": round(cov, 2),
        "gray_pct": round(gray, 2), "stroke_thickness_px": round(thick, 1),
        "components": comps, "looks_like_line_art": clean, "warnings": warnings,
    }

_TRACE_AUDIT = {}   # last _trace_svg run


def _mask_gap_close(mask, mm_per_px, d_px_requested, debug_prefix=None):
    """Close mask holes narrower than a kernel diameter given directly in
    TRACE-FRAME PIXELS, without moving the outer/page-frame boundary
    (opt-in `mask_gap_close_px`). The operator is:

        out | (close(out, ellipse(d)) & fill_holes(out))

    `fill_holes(out)` restricts every addition to pixels already strictly
    enclosed by `out` (i.e. inside a hole), so pixels connected to the
    frame/background can never be added regardless of kernel size -- the
    outer-edge-unchanged property holds by construction, not by luck.

    The knob is a PIXEL diameter, not an mm width, because the `mm_per_px`
    available at this function's call site in _trace_svg is `fill_mm/fill_px`.
    For a pen with a measured `laid_mm` (halfpitch, tight3x, ...) that value
    differs from the TRUE working-grid mm/px by a factor of fill_mm/laid_mm
    (about 1.96x for halfpitch, 2.7x for tight3x), because composite derives
    fill_px from laid_mm while the back-solve divides by fill_mm.
    `mm_per_px` is therefore used below ONLY to label `realized_mm_nominal`
    (an approximate, trace-frame-nominal figure, NOT the physical width on
    paper for laid_mm pens) and for the area/histogram diagnostics. It never
    sizes the kernel, so the scale mismatch cannot move drawn geometry, only
    the audit's mm labels.

    Returns (new_mask, audit_dict).
    """
    import cv2, numpy as np
    import scipy.ndimage as ndi
    # ODD kernel only -- an EVEN-sized ellipse's cv2 anchor defaults to
    # size//2 (not centered), which makes closed area non-monotonic in the
    # diameter (a larger request can close fewer pixels). Round UP to the
    # nearest odd (never down -- the requested diameter is a floor, not a
    # target to shrink) so the kernel is always centered and at least as large
    # as requested.
    d = max(1, int(d_px_requested))
    if d % 2 == 0:
        d += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    filled = ndi.binary_fill_holes(mask)
    new_mask = mask | (closed & filled)

    added = new_mask & ~mask
    n_comp, lbl, _stats, _cent = cv2.connectedComponentsWithStats(added.astype(np.uint8), 8)
    holes_closed = max(0, n_comp - 1)
    area_px = int(added.sum())
    widths_px = []
    if holes_closed:
        # Diagnostic width estimate per closed component: 2x the component's
        # own max distance-transform value, i.e. the diameter of the largest
        # disk inscribed in the closed hole. Not exact for non-round holes,
        # but monotonic with true width and cheap -- used only for the audit
        # histogram, not for any downstream geometry decision.
        dt_added = cv2.distanceTransform(added.astype(np.uint8), cv2.DIST_L2, 5)
        for i in range(1, n_comp):
            widths_px.append(float(2.0 * dt_added[lbl == i].max()))
    hist_bins_px = [0, 1, 2, 3, 4, 5, 7, 9, 13, 21, 1000000]
    if widths_px:
        hist_counts = np.histogram(widths_px, bins=hist_bins_px)[0]
    else:
        hist_counts = np.zeros(len(hist_bins_px) - 1, dtype=int)

    audit = {
        "mask_gap_close_requested_px": int(d_px_requested),
        "mask_gap_close_kernel_px": int(d),
        "mask_gap_close_holes_closed": int(holes_closed),
        "mask_gap_close_area_px": area_px,
        "mask_gap_close_width_hist_px_bins": hist_bins_px,
        "mask_gap_close_width_hist_px_counts": [int(c) for c in hist_counts],
        # TRACE-FRAME NOMINAL only -- mm_per_px here is fill_mm/fill_px, not
        # the true working-grid mm/px for laid_mm pens (see this function's
        # docstring). Do not read this as a physical width.
        "mask_gap_close_realized_mm_nominal": round(d * mm_per_px, 4),
    }
    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_gap_close_added.png", (added.astype(np.uint8) * 255))
    return new_mask, audit


def _tracev3_mask(native_gray, opts, W, H):
    """Native-res mask (opt-in `native_res`): binarize at NATIVE res (gaps are real
    pixels there), then resize the BINARY mask to the working grid preserving gaps:
      * downscale: INTER_AREA majority vote (a target pixel is ink only if >50%
        ink) — a gap survives when it owns the majority of its target cell.
      * upscale (small source): INTER_LINEAR on the float mask, then the same
        0.5 cut (see the comment below); the level-set approxPolyDP smooths the
        geometry.
    Thresholding-then-resizing keeps gaps that resizing-then-thresholding would
    blur away: Otsu never sees a gray gap to misjudge."""
    import cv2, numpy as np
    m = _binarize_mask(native_gray, opts)
    nh, nw = m.shape[:2]
    if (nw, nh) == (W, H):
        return m
    # threshold at 0.5 = keep ink only where the resized cell is majority ink, so a
    # gap (minority ink) survives whenever it owns its cell. AREA for downscale
    # (true majority vote), LINEAR for upscale (smooth edges vs blocky NEAREST;
    # sub-1.5-native-px gaps that vanish are sub-deposit anyway = physically lost).
    interp = cv2.INTER_AREA if nw * nh > W * H else cv2.INTER_LINEAR
    return cv2.resize(m.astype(np.float32), (W, H), interpolation=interp) > 0.5


def _gap_guarantee(mask, fill_px):
    """Flag-gated: widen physics-marginal gaps to one deposit width.
    A gap in [0.5, 1.0]*deposit survives in the mask but bleeds shut on paper (the
    pen lays ~one deposit wide). Carve a deposit-wide channel along each such gap's
    centerline — eroding the flanking ink — so the gap opens to ~one deposit. Trades
    hairline-thinner surrounding ink for a gap that actually prints (pipes/valves
    stay visible). Pure geometry: a WIDTH test on the background distance transform,
    no semantics. Returns (new_mask, marginal_ridge_px)."""
    import cv2, numpy as np
    from skimage.morphology import skeletonize
    bg = (~mask).astype(np.uint8)
    bg_dt = cv2.distanceTransform(bg, cv2.DIST_L2, 5)      # half-gap-width at bg px
    ridge = skeletonize(bg > 0)
    lo, hi = 0.25 * fill_px, 0.5 * fill_px                  # half-widths of [0.5,1.0]*fill gaps
    marginal = ridge & (bg_dt >= lo) & (bg_dt < hi)
    n = int(marginal.sum())
    if not n:
        return mask, 0
    r = max(1, int(round(fill_px / 2)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    carve = cv2.dilate(marginal.astype(np.uint8), k) > 0
    return (mask & ~carve), n


def _link_pockets(polys, link_dist, pocket_close="force", tags=None):
    """Chain nearby passes (nested pocket rings sit ~fill apart) into continuous
    paths, drawing a short pen-DOWN connector instead of a lift. Greedy nearest-
    vertex via cKDTree (fast); each ring is rotated to start at the join point so
    the path is near-continuous. Connectors <= link_dist stay on/near ink, so
    coverage is unchanged while pen-lifts (and gcode/plot cost) drop sharply.

    pocket_close (opt-in): 'force' (default) closes every seed and every
    joined-in fragment's own end->start seam unconditionally, even when the
    fragment was never actually a closed ring (an OPEN skeleton/pinch-fallback
    dash still gets a straight line drawn from its far end back to its near end).
    For an already-closed ring (a real level-set pocket contour) that seam is
    just the ring's own natural edge -- always fine, always drawn, in EITHER
    mode. For a genuinely OPEN fragment it is FALSE GEOMETRY when the seam is
    long (it can span tens of mm on a dense source). 'guarded' only
    closes an open fragment's own seam if the seam's two endpoints are within
    link_dist of each other (the SAME proximity constant already used for join
    decisions -- no new constant invented). When a guarded fragment's seam is
    too long, no ink is lost: the fragment still draws in full, just as its own
    separate output chain past the join point instead of a falsely-bridged
    loop -- trading one extra pen lift for correctness, never trading
    correctness for a lift.

    tags (diagnostic-only, opt-in via _trace_svg's provenance_tag flag,
    stroke-provenance instrumentation): optional list, same length
    as `polys`, giving each input polyline's producer-of-origin. When
    provided, this function ALSO returns a parallel `out_tags` list (one
    entry per output chain -- the deduped, order-preserving set of origin
    tags among every input ring that got merged into that chain, since force
    mode can and does merge rings from different producers). When `tags` is
    None (every call site except the diagnostic one), only `out` is returned
    and no bookkeeping runs; the geometry is the same either way."""
    import numpy as np
    if len(polys) < 2:
        return polys if tags is None else (polys, [[t] for t in tags])
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return polys if tags is None else (polys, [[t] for t in tags])
    guarded = str(pocket_close).lower() == "guarded"
    rs = [np.asarray(p, float) for p in polys]
    def _open(r):
        return r[:-1] if len(r) > 1 and bool((r[0] == r[-1]).all()) else r
    def _was_closed(r):
        return len(r) > 1 and bool((r[0] == r[-1]).all())
    def _seam_ok(ro):
        # ro is the OPENED point array. The seam is the edge (ro[-1], ro[0])
        # that closes it back into a loop. Always fine in force mode, and
        # always fine for input that was ALREADY a closed ring (that seam is
        # real ring topology, not a fabricated bridge). Guarded-gated only
        # for genuinely open input.
        if not guarded:
            return True
        d = float(np.hypot(ro[-1][0] - ro[0][0], ro[-1][1] - ro[0][1]))
        return d <= link_dist
    allpts, ring_of, idx_of = [], [], []
    for ri, r in enumerate(rs):
        ro = _open(r)
        for vi, v in enumerate(ro):
            allpts.append(v); ring_of.append(ri); idx_of.append(vi)
    allpts = np.array(allpts); ring_of = np.array(ring_of); idx_of = np.array(idx_of)
    tree = cKDTree(allpts)
    used = np.zeros(len(rs), bool)
    out = []
    _track = tags is not None  # diagnostic-only ring-provenance bookkeeping
    out_rings = [] if _track else None
    while not used.all():
        seed = int(np.argmax(~used))
        r = _open(rs[seed]); used[seed] = True
        chain_rings = [seed] if _track else None
        close_seed = True if _was_closed(rs[seed]) else _seam_ok(r)
        chain = [tuple(v) for v in r] + ([tuple(r[0])] if close_seed else [])  # close the seed ring (guard-gated)
        cur = np.array(chain[-1])
        while True:
            dd, ii = tree.query(cur, k=32)
            nx, nv = -1, 0
            for d, g in zip(np.atleast_1d(dd), np.atleast_1d(ii)):
                if not used[ring_of[g]] and d <= link_dist:
                    nx, nv = int(ring_of[g]), int(idx_of[g]); break
            if nx < 0:
                break
            used[nx] = True
            r = _open(rs[nx])
            if _track:
                chain_rings.append(nx)
            if _was_closed(rs[nx]) or _seam_ok(r):
                # legit ring (or open fragment whose own seam is within
                # link_dist) -- draw the full rotated+closed fragment,
                # exactly as force mode does.
                rot = np.vstack([r[nv:], r[:nv], r[nv:nv + 1]])  # start at join point, close
                chain.extend(tuple(v) for v in rot)
                cur = rot[-1]
            else:
                # OPEN fragment whose own end->start seam exceeds link_dist --
                # never bridge it with fake ink. Draw only the join-side run
                # (r[nv:], real consecutive-index edges, joined onto the
                # current chain), and emit the far side (r[:nv+1], also all
                # real consecutive-index edges) as its own separate output
                # chain -- one extra pen lift, zero false geometry.
                fwd = r[nv:]
                chain.extend(tuple(v) for v in fwd)
                cur = fwd[-1]
                if nv > 0:
                    out.append([tuple(v) for v in r[:nv + 1]])
                    if _track:
                        out_rings.append([nx])
        out.append(chain)
        if _track:
            out_rings.append(chain_rings)
    if _track:
        out_tags = []
        for ring_ids in out_rings:
            seen = []
            for rid in ring_ids:
                t = tags[rid]
                if t not in seen:
                    seen.append(t)
            out_tags.append(seen)
        return out, out_tags
    return out


def _level_set_fill(region, spacing_px, fallback=True):
    """Tone helper — width-proportional level-set passes over ONE mask region at
    an arbitrary pass SPACING. This is the Trace V3 pass-generator logic (pass
    k = contour(dt > (k+0.5)*spacing)); parameterizing spacing is the ONLY lever
    tone uses — pure geometry, zero classification. spacing=fill_px reproduces the
    solid V3 fill; larger spacing leaves ink-free gaps between passes -> a lighter
    tone. Thin regions with no full pass (max_dt <= 0.5*spacing) fall back to a
    skeleton centerline so nothing is dropped."""
    import cv2, numpy as np
    from skimage.morphology import skeletonize
    region = region.astype(np.uint8)
    if not region.any():
        return []
    dt = cv2.distanceTransform(region, cv2.DIST_L2, 5)
    passes = []
    k, maxk = 0, int(dt.max() / spacing_px) + 2
    while k < maxk:
        band = (dt > (k + 0.5) * spacing_px).astype(np.uint8)
        if not band.any():
            break
        cnts, _ = cv2.findContours(band, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
        for c in cnts:
            if len(c) >= 2:
                cc = cv2.approxPolyDP(c, 0.4, True).reshape(-1, 2)
                if len(cc) >= 2:
                    p = [(float(x), float(y)) for x, y in cc]
                    p.append(p[0])
                    passes.append(p)
        k += 1
    if fallback:
        # zero-pass components (too thin for even the innermost pass) -> centerline
        n, lbl, _stats, _ = cv2.connectedComponentsWithStats(region, 8)
        if n > 1:
            maxdt = np.zeros(n, np.float32)
            np.maximum.at(maxdt, lbl.ravel(), dt.ravel())
            zero = [i for i in range(1, n) if maxdt[i] <= 0.5 * spacing_px]
            if zero:
                zmask = np.isin(lbl, zero)
                passes += _skeleton_to_polylines(skeletonize(zmask),
                                                 prune_len=max(2, int(spacing_px)))
    return passes


def _outline_only(region):
    """Tone helper — external boundary contours only (the faintest tone treatment for
    the lightest ink level: trace the region edge, leave the interior blank)."""
    import cv2
    cnts, _ = cv2.findContours(region.astype("uint8"), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_TC89_L1)
    out = []
    for c in cnts:
        if len(c) >= 2:
            cc = cv2.approxPolyDP(c, 0.6, True).reshape(-1, 2)
            if len(cc) >= 3:
                p = [(float(x), float(y)) for x, y in cc]
                p.append(p[0])
                out.append(p)
    return out


# ---------------------------------------------------------------------------
# Tone (grayscale) generator. It runs only in the tone-ON path (`if tone_on:` in
# _trace_svg); with tone off the plain contour-fill path runs. Tone is PARALLEL
# HATCH IN A PEN-RELATIVE SPACING WINDOW, the mechanism a plotted gray wedge
# showed to read as tone on paper. Cross-hatch (for a true-black dark end) is not
# implemented.
#
# TONE_LEVELS_N_DEFAULT / TONE_WINDOW_*_MULT / TONE_SOLID_MULT are anchored to a
# plotted gray wedge: 16 patches at fill_mm=0.19, hatch spacing geometric from
# 0.5x to 2.0x fill_mm (p1=0.50x .. p16=2.0x). Reading it on paper:
#   - patches 1-3 (<=~0.60x fill_mm): fully SOLID, indistinguishable from each
#     other -- the true-black / solid floor.
#   - patch 4 (0.66x): first distinguishable step (darkest non-solid tone).
#   - gradation is real but subtle through ~patch 13-14 (1.52x-1.66x); the usable
#     range is capped there (light ceiling).
#   - patches 15-16 (1.8x-2.0x): "lines on white" dead zone, excluded.
# About 7 distinguishable levels fit across the graded window, so 7 is the
# default; it is a working default, not a validated optimum. These are spacing
# calibrations for the current parallel-hatch mechanism and would need to be
# re-measured if the mechanism changes. N is a PARAMETER (opts["tone_levels_n"]),
# not hardcoded into the generator -- TONE_LEVELS_N_DEFAULT is only the fallback
# when the caller does not override it.
TONE_LEVELS_N_DEFAULT = 7

# Hatch-spacing window (see the wedge-reading comment block above).
# TONE_SOLID_MULT: spacing at or below this multiple of fill_px is
# indistinguishable from true-black solid fill (wedge patches 1-3, <=~0.60x).
# The darkest quantile band in `_multitone_passes` is always filled fully
# solid via `_level_set_fill` (not hatched at all, i.e. denser than even this
# floor) -- this constant documents/traces that choice against the measured
# solid threshold rather than driving a spacing computation directly.
# TONE_WINDOW_LO_MULT / TONE_WINDOW_HI_MULT: the graded-hatch band, spanning
# patch 4 (first distinguishable step, 0.66x) through the capped light
# ceiling (patch 13-14, ~1.5x-1.6x -- 1.6 chosen as the round anchor). The
# lightest hatch level is always clamped to the HI end -- never looser --
# which is what enforces "clamp anything lighter to no-ink": there is no band
# beyond the lightest, and content lighter than that was already excluded
# from `ink_mask` upstream (the binarize step), so nothing in this generator
# can ever place ink past the ceiling. Spacing looser than TONE_WINDOW_HI_MULT
# is the wedge's own "lines on white" dead zone (patches 15-16) and is
# excluded by construction (never reached).
TONE_SOLID_MULT = 0.6
TONE_WINDOW_LO_MULT = 0.66
TONE_WINDOW_HI_MULT = 1.6

# Single fixed page angle, GLOBAL (no sub-region partition). A 45-degree hatch is
# the angle the gray wedge was plotted at.
TONE_HATCH_ANGLE_DEG = 45.0

# Transfer-curve family for the tone knob (off / faithful / balanced / dense).
# gamma reshapes how a level's RANK (1..N-1, darkest windowed level to lightest)
# is redistributed across the window: gamma<1 front-loads more ranks toward the
# TIGHT (darker-reading) end of the window, so "dense" adds ink relative to
# "faithful". The curve is independent of the wedge calibration; only N and the
# window anchors above come from it.
_TONE_KNOB_GAMMA = {
    "faithful": 1.00,
    "balanced": 0.75,
    "dense":    0.55,
}


def _hatch_fill(region, spacing_px, angle_deg=TONE_HATCH_ANGLE_DEG,
                merge_gap_px=None, min_run_px=None):
    """Tone generator primitive -- straight PARALLEL HATCH lines at
    a single fixed page angle, confined to `region`, at pass spacing
    `spacing_px`. Replaces the concentric-ring `_level_set_fill` as the
    *tone-band* fill (dark stays on `_level_set_fill`, see `_multitone_passes`):
    the plotted gray wedge showed parallel hatching, not concentric rings, to be
    the physical tone channel.

    Mechanism: rotate `region` onto an expanded canvas (cv2.warpAffine, nearest
    neighbor, so nothing clips) so hatch lines become plain horizontal scanlines
    in the rotated frame; walk rows every `spacing_px`; run-length-encode each
    row's ink/background runs; MERGE runs separated by a gap <= merge_gap_px
    (bridges a stray raster dropout inside one hatch stroke); DROP surviving
    runs shorter than min_run_px. The gap-merge and short-run drop stop a single
    intended stroke from fragmenting into dashes. Tone quantizes to N discrete
    bands and fills each band as a plain region, so there is no per-pixel
    dithering or duty-cycle accumulator. Surviving runs are emitted as straight
    2-point polylines, mapped back through the inverse rotation into the
    original image frame.

    Defaults: merge_gap_px = spacing_px (bridge a dropout up to one pass-
    spacing wide); min_run_px = 2*spacing_px. Both scale PEN-RELATIVE with this
    band's own spacing rather than using a fixed absolute mm value.
    """
    import cv2, numpy as np
    region = np.asarray(region).astype(np.uint8)
    if not region.any() or spacing_px <= 0:
        return []
    if merge_gap_px is None:
        merge_gap_px = spacing_px
    if min_run_px is None:
        min_run_px = 2.0 * spacing_px
    H, W = region.shape[:2]
    diag = int(np.ceil(np.hypot(H, W))) + 4
    cx, cy = W / 2.0, H / 2.0
    ncx, ncy = diag / 2.0, diag / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M[0, 2] += ncx - cx
    M[1, 2] += ncy - cy
    rot = cv2.warpAffine(region, M, (diag, diag), flags=cv2.INTER_NEAREST, borderValue=0)
    Minv = cv2.invertAffineTransform(M)
    passes = []
    step = max(1, int(round(spacing_px)))
    for y in range(0, diag, step):
        row = rot[y] > 0
        if not row.any():
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        idx = np.flatnonzero(d)
        starts, ends = idx[0::2], idx[1::2]        # [start, end) True runs
        merged = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            if merged and (s - merged[-1][1]) <= merge_gap_px:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        for s, e in merged:
            if (e - s) < min_run_px:
                continue
            x0 = Minv[0, 0] * s + Minv[0, 1] * y + Minv[0, 2]
            y0 = Minv[1, 0] * s + Minv[1, 1] * y + Minv[1, 2]
            x1 = Minv[0, 0] * (e - 1) + Minv[0, 1] * y + Minv[0, 2]
            y1 = Minv[1, 0] * (e - 1) + Minv[1, 1] * y + Minv[1, 2]
            passes.append([(float(x0), float(y0)), (float(x1), float(y1))])
    return passes


def _zigzag_fill(mask, fill_px, angle_deg=0.0, merge_gap_px=0.0, min_run_px=None,
                  supersample=1, connect_guarded=False, connect_max_gap_px=0.0,
                  connect_pen_px=None, connect_boustrophedon=False,
                  connect_proximity=False, connect_split_merge=False,
                  connect_record_connectors=False, connect_endlink=False,
                  connect_endlink_max_drow=1):
    """SOLID-fill generator (opt-in `fill_mode="zigzag"`): parallel scanlines, one
    open 2-point polyline per ink run, no bridging by default. It uses the same
    rotate / walk-rows / run-length-encode / inverse-affine skeleton as
    `_hatch_fill` (the tone-mode primitive) with four differences that a SOLID
    fill needs (each would be a defect for solid fill, though right for tone):

      1. Float/sub-pixel scanline placement. `_hatch_fill` quantizes with
         `step = max(1, int(round(spacing_px)))`, which gives a pitch error
         of +16% at fill_px=2.58. Here every scanline
         position `y_f = k * fill_px` is an exact float; ink presence is
         looked up at the nearest raster row (optionally on a
         `supersample`-upscaled raster for finer cross-axis precision), but
         the OUTPUT polyline is mapped through the inverse affine using
         `y_f` itself -- scan-axis pitch is then exact by construction,
         independent of `supersample`.
      2. `merge_gap_px` defaults to 0 (not `_hatch_fill`'s `spacing_px`) --
         bridging up to one pitch draws straight over small holes the mask's
         own hole floor (trace_v3.py `_trace_svg`, `hole_frac`) deliberately
         preserved.
      3. `min_run_px` is caller-supplied (~= 1 pen width), not
         `2*spacing_px` -- drops are counted in the returned audit, not
         silently discarded.
      4. Every surviving run is its own OPEN 2-point polyline unless one of
         the connect_* options below links runs; the default is
         `fill_connect="none"`.

    mask: 2D bool/uint8 array, ink=nonzero -- the SAME array the contour
      branch (`_trace_svg`'s tone-off `else:`) receives, post dot-cull and
      post hole-fill.
    fill_px: scanline pitch in mask-pixel units, same units as `_hatch_fill`'s
      `spacing_px` / `_trace_svg`'s own `fill_px`.
    angle_deg: IMAGE-frame rotation for the scan direction (same convention
      as `_hatch_fill`/`cv2.getRotationMatrix2D`). Callers wanting a
      MACHINE-frame angle must supply the already-corrected image-frame
      value -- that back-computation lives in composite.py, not here.
    supersample: int >= 1. Upsamples the rotated mask by this factor before
      row-walking, improving CROSS-axis (run-boundary/hole-edge) precision
      only -- scan-axis pitch does not depend on it (see point 1). Memory
      cost is O(S^2) on the (H+W)-diagonal canvas (roughly 400 MB at S=2 on
      a 5000 px job) -- the default is 1 (no extra memory) unless a caller
      opts in explicitly.

    connect_guarded (opt-in `fill_connect="guarded"`): the guarded connector.
      Two runs on the SAME scanline are adjacent-in-raster by construction
      (`merged` below is built by walking the row left to right), so "join
      run i to run i+1" is a structural question about the gap between
      them, not a search. Unguarded joining (`merge_gap_px > 0`)
      bridges ANY gap up to that width regardless of what is
      under it -- the hazard the default "none" path avoids. Guarded joining instead bridges a gap of any width up to
      `connect_max_gap_px` PROVIDED every pixel of it, in the same rotated
      raster the runs were detected from, lies within `dilate(rot,
      pen_radius)` -- i.e. the connector's own painted footprint (pen width,
      same convention as `_zigzag_offmask_ink_pct`/`_stroke_px_for`) would
      not add ink anywhere the mask does not already call for. The dilation
      is computed ONCE per call, not per candidate. `connect_pen_px`
      defaults to `min_run_px` (both already approximate one pen width at
      the zigzag call site) when not given. `connect_max_gap_px` is a
      separate, always-enforced ceiling: an in-mask gap still needs a bound,
      because a white slit thinner than the pen but long and PARALLEL to the
      scan direction is filled by the dilation at any length, and joining
      across a large solid region is not what this feature is for. Runs joined this way are merged into ONE segment (a
      longer 2-point straight run spanning both, since collinearity is
      guaranteed by same-scanline construction) before `min_run_px` is
      tested, so a joined pair can survive a length floor that would have
      dropped either half alone -- `n_dropped_short` still reflects drops
      AFTER joining, so it can be lower than with `connect_guarded=False`
      on the same input. Every join and
      every refusal (by mask or by ceiling) is counted in the returned
      audit (`n_joins`, `n_join_refused_offmask`, `n_join_refused_ceiling`).
      Default `False`: no joins are made.

    connect_boustrophedon (opt-in `fill_connect="boustrophedon"`,
      extends `connect_guarded` -- requires it True): turns the per-run
      output (one independent 2-point pass per surviving run, `connect_
      guarded`'s own within-row joins notwithstanding) into an ACTUAL
      boustrophedon -- one continuous chain across MANY scanlines, direction
      alternating row to row (row 1 L->R, row 2 R->L, ...), exactly like FDM
      slicer infill. The row-to-row join reuses the SAME `rot_dil` dilated
      raster and the SAME always-enforced `connect_max_gap_px` ceiling
      already built for the intra-row guarded predicate (`_seg_in_mask`
      below generalizes the intra-row `rot_dil[y_i, prev_e:s].all()` test
      from a same-row horizontal slice to an arbitrary line segment on that
      identical array -- not a second predicate). A row-to-row join is
      accepted only if the straight segment between the previous chain's
      open end and the next run's entry point lies entirely within
      `rot_dil` AND its length is within the ceiling; otherwise the pen
      lifts and a new chain starts, exactly like `connect_guarded`'s own
      row-internal refusal. When False (default), the plain per-row emit loop
      below runs.

    Returns (passes, audit) where audit carries n_scanlines/n_runs/
    n_dropped_short/dropped_len_px_total -- consumed by `_trace_svg`'s
    zigzag-only, flag-gated `_TRACE_AUDIT` keys (never added when
    fill_mode != "zigzag", so the default path's audit JSON is untouched)."""
    import cv2, numpy as np
    mask = np.asarray(mask)
    mask_u8 = (mask > 0).astype(np.uint8)
    if min_run_px is None:
        min_run_px = fill_px
    if not mask_u8.any() or fill_px <= 0:
        return [], {"n_scanlines": 0, "n_runs": 0, "n_dropped_short": 0,
                     "dropped_len_px_total": 0.0, "runs_per_scanline": []}

    H, W = mask_u8.shape[:2]
    S = max(1, int(supersample))
    diag = int(np.ceil(np.hypot(H, W))) + 4
    diagS = diag * S

    mask_S = mask_u8
    if S > 1:
        mask_S = cv2.resize(mask_u8, (W * S, H * S), interpolation=cv2.INTER_NEAREST)

    cxS, cyS = (W * S) / 2.0, (H * S) / 2.0
    ncx, ncy = diagS / 2.0, diagS / 2.0
    M = cv2.getRotationMatrix2D((cxS, cyS), angle_deg, 1.0)
    M[0, 2] += ncx - cxS
    M[1, 2] += ncy - cyS
    rot = cv2.warpAffine(mask_S, M, (diagS, diagS), flags=cv2.INTER_NEAREST, borderValue=0)
    Minv = cv2.invertAffineTransform(M)

    spacing_super = fill_px * S
    gap_super = merge_gap_px * S
    n_scan = int(diagS / spacing_super) + 1

    # Guarded connector (opt-in) -- see this function's own docstring. `rot_dil` is the SAME rotated raster runs are detected from,
    # dilated once by the pen radius (same ellipse-kernel convention as
    # `_zigzag_offmask_ink_pct`), so the row-wise gap test below is a direct
    # readout: "does the pen's own footprint already cover this gap" is
    # exactly `rot_dil[y_i, prev_e:s].all()`. Computed unconditionally when
    # requested, once, never per-candidate.
    n_joins = 0
    n_join_refused_offmask = 0
    n_join_refused_ceiling = 0
    rot_dil = None
    ceil_super = 0.0
    if connect_guarded:
        _pen_px_c = float(connect_pen_px) if connect_pen_px is not None else float(min_run_px)
        _pen_super = max(1.0, _pen_px_c * S)
        _r = max(1, int(round(_pen_super / 2.0)))
        _k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _r + 1, 2 * _r + 1))
        rot_dil = cv2.dilate(rot, _k) > 0
        ceil_super = float(connect_max_gap_px) * S

    passes = []
    runs_per_scanline = []
    n_dropped = 0
    dropped_len_total = 0.0

    # Boustrophedon row-to-row joins (opt-in fill_connect="boustrophedon" --
    # see this function's own docstring). All state below is inert (never
    # read) when connect_boustrophedon is False.
    #
    # PER-COLUMN, not per-row. Alternating direction PER ROW across ALL of
    # that row's runs sends the traversal to the far side of the row on every
    # transition whenever a row has more than one run (multiple disjoint
    # span-columns on the same scanline, the norm on a real drawing), and the
    # guarded intra-row test correctly refuses almost every such far-side hop.
    # The goal is one continuous path per contiguous span-column, so the open
    # CHAIN is tracked PER COLUMN (by x-overlap with the previous row's run),
    # not per physical row.
    # `_open_cols`: list of {"s","e","chain","last_test"} -- `s`/`e` are the
    # most recently absorbed run's super-pixel column span (for the NEXT
    # row's overlap test), `chain` is the column's accumulated OUTPUT
    # points, `last_test` is the column's open end in TEST coords (y_i
    # grid) for the join predicate.
    n_row_joins = 0
    n_row_join_refused_offmask = 0
    n_row_join_refused_ceiling = 0
    _open_cols = []

    # Diagnosis-only "traversed components" floor: `n_mask_components_4conn`
    # alone over-counts, because it includes components whose runs are ALL
    # dropped by `min_run_px`; those contribute zero chains and so cannot be a
    # chain floor. Computed BEFORE the scan loop
    # (gated on connect_boustrophedon, same discipline as everywhere else
    # in this block) so the per-row loop below can look up each SURVIVING
    # run's own component label as it goes.
    _mask_comp_labels = None
    if connect_boustrophedon:
        _n_mask_components_raw, _mask_comp_labels = cv2.connectedComponents(
            mask_u8, connectivity=4)
    _traversed_mask_labels = set()

    # Diagnosis-only census ledger. Every append to `passes` (a
    # chain ENDING) and every entry created in `new_cols` (a chain STARTING)
    # is attributed to exactly one mutually-exclusive, exhaustive reason
    # below. Counters only -- no change to `passes`/`_open_cols` control
    # flow, no change to which branch runs or what it appends; this block
    # only classifies decisions the surrounding code already makes. Inert
    # (never incremented) when connect_boustrophedon is False.
    census_end_split = 0            # old column overlapped >=2 runs below
    census_end_merge = 0            # old column's one overlapping run is
                                     # itself claimed by >=2 open columns
    census_end_no_overlap = 0       # old column overlapped 0 runs below
    census_end_no_overlap_near_miss = 0  # sub-bucket of the above: SOME run
                                     # on that row existed within
                                     # connect_max_gap_px (super-px) of the
                                     # column's open end but did not
                                     # x-overlap it -- i.e. a run was
                                     # "close" but this per-column predicate
                                     # is x-overlap, not proximity, so it
                                     # does not count as a candidate at all
                                     # (distinct from a refused join, which
                                     # requires x-overlap first).
    census_end_refused_offmask = 0  # == n_row_join_refused_offmask, mirrored
                                     # here so the full ledger sums standalone
    census_end_refused_ceiling = 0  # == n_row_join_refused_ceiling, mirrored
    census_end_row_empty = 0        # row had zero surviving runs -- full flush
    census_end_eoi = 0              # end-of-image flush of remaining columns

    census_start_after_refusal = 0  # 1:1 match found, join refused -> new col
    census_start_fresh_no_col = 0   # run overlapped 0 open columns
    census_start_fresh_ambiguous = 0  # run overlapped >=2 open columns (merge,
                                     # run side)
    census_start_fresh_after_split = 0  # run's one overlapping column itself
                                     # overlapped >=2 runs (split, run side)

    census_cols_carried_over_blank_row = 0  # open columns survive a fully
                                     # blank scanline (`row.any()` false,
                                     # `continue` before the open-column
                                     # logic runs at all) -- these are NOT a
                                     # separate end/start event by
                                     # themselves, counted for exhaustiveness
                                     # bookkeeping only.

    # Component-count floor (b): union-find over SURVIVING runs only (post
    # min_run_px), edges = x-overlap between a row's runs and the
    # immediately preceding NON-EMPTY-ROW's runs -- ignores the join
    # predicate (mask/ceiling) entirely, so this is the count the
    # traversal would see if union alone (no refusal) decided chains. Not
    # read by any control-flow branch above -- diagnosis only.
    _dsu_parent = []

    def _dsu_find(i):
        while _dsu_parent[i] != i:
            _dsu_parent[i] = _dsu_parent[_dsu_parent[i]]
            i = _dsu_parent[i]
        return i

    def _dsu_union(a, b):
        ra, rb = _dsu_find(a), _dsu_find(b)
        if ra != rb:
            _dsu_parent[ra] = rb

    _dsu_prev_row = []  # list of (s, e, dsu_id) for the previous NON-EMPTY row

    def _seg_in_mask(p0, p1):
        # Generalizes the intra-row guarded test
        # (`rot_dil[y_i, prev_e:s].all()`) from a same-row horizontal slice
        # to an arbitrary line segment on the SAME `rot_dil` array -- reused
        # predicate, not a second one.
        x0, y0 = p0
        x1, y1 = p1
        xmin = max(0, int(np.floor(min(x0, x1))) - 1)
        xmax = min(rot_dil.shape[1] - 1, int(np.ceil(max(x0, x1))) + 1)
        ymin = max(0, int(np.floor(min(y0, y1))) - 1)
        ymax = min(rot_dil.shape[0] - 1, int(np.ceil(max(y0, y1))) + 1)
        if xmax < xmin or ymax < ymin:
            return True
        w, h = xmax - xmin + 1, ymax - ymin + 1
        canvas = np.zeros((h, w), np.uint8)
        cv2.line(canvas, (int(round(x0 - xmin)), int(round(y0 - ymin))),
                 (int(round(x1 - xmin)), int(round(y1 - ymin))), 1, 1,
                 lineType=cv2.LINE_8)
        seg = canvas.astype(bool)
        return bool(np.all(rot_dil[ymin:ymax + 1, xmin:xmax + 1][seg]))

    def _map_super_pt(pt):
        xs, ys = pt
        xo = Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2]
        yo = Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]
        return (float(xo / S), float(yo / S))

    # Two more continuation modes, each its own opt-in switch, extending
    # `connect_boustrophedon` -- inert (never read/incremented) unless it is True,
    # and unless their own flag is also True. NO NEW JOIN PREDICATE: every
    # connector either mode adds goes through the SAME `_seg_in_mask` (on
    # the SAME `rot_dil`) + the SAME `ceil_super` ceiling the existing
    # intra-row/row-to-row joins already use -- `_pick_safe` below is a
    # dispatcher over that one predicate, not a second one.
    #
    # connect_proximity: an open column with ZERO x-overlapping runs on the
    # next row (the `census_end_no_overlap` case) may continue into the
    # NEAREST unclaimed run whose connector passes the safety test, instead
    # of always flushing. This addresses the `near_miss` case in the census.
    #
    # connect_split_merge: on a split (one open column x-overlaps >=2 runs),
    # the column continues into ONE branch (nearest safe) instead of
    # flushing both; the other branch(es) start fresh columns as before
    # (unchanged code path). On a merge (one run x-overlapped by >=2 open
    # columns), ONE incoming column (nearest safe) continues into it; the
    # rest end as `census_end_merge` (unchanged meaning: "this column's
    # only candidate run went to someone else").
    #
    # Each run/column is claimed by AT MOST one connector per row -- ties
    # broken deterministically by `(distance, x-position)` ascending
    # (`_pick_safe`'s own sort key). Phase order (1:1 -> split -> merge ->
    # proximity, each phase working only on what the PRIOR phase left
    # unconsumed) is a deterministic choice, not a globally optimal
    # matching -- a split picking branch R2 can strand a different column
    # that also wanted R2 (that column falls through to its own normal
    # classification). It is a greedy dispatcher, not a solver.
    n_split_continued = 0
    n_merge_continued = 0
    n_proximity_joins = 0
    n_proximity_refused_offmask = 0
    n_proximity_no_candidate = 0
    # The connector ledger below is diagnosis-only. Building it on every
    # boustrophedon call would cost a large dict list that no caller reads, so
    # it is populated only when either new switch is on, OR an explicit
    # `connect_record_connectors=True` kwarg is passed (default False) for a
    # caller that wants the ledger on its own, e.g. to calibrate `row_join`'s
    # baseline overdraw rate with both switches off. `_record_connector` is a
    # no-op (single boolean check) when neither condition holds.
    # `_connect_endlink_eff`: computed here (not just at the endlink block
    # itself, further down) -- forced off unless connect_boustrophedon
    # actually fired AND connect_guarded built `rot_dil` (the raster
    # `_seg_in_mask` tests against), at the FUNCTION level, not only at
    # `_trace_svg`'s one caller, so a direct call with
    # `connect_endlink=True, connect_boustrophedon=False` (or
    # `connect_guarded=False`) degrades to a no-op instead of crashing on
    # `rot_dil=None`.
    _connect_endlink_eff = bool(connect_endlink) and connect_boustrophedon and rot_dil is not None
    # `_record_ledger` is NOT forced on by `_connect_endlink_eff`: an
    # endlink-only caller that wants the connector ledger (e.g. for its own
    # overdraw-risk measurement) must pass `connect_record_connectors=True`
    # explicitly, like any other caller.
    _record_ledger = bool(connect_proximity or connect_split_merge
                           or connect_record_connectors)
    _connectors = []  # diagnosis-only ledger, every connector of every kind
                       # (including the plain 1:1 row-to-row join,
                       # tagged "row_join" -- the calibrating baseline for
                       # the overdraw-risk measurement). NOT merged into
                       # `_TRACE_AUDIT` (would be huge); an analysis harness can
                       # capture it by wrapping `_zigzag_fill`. Pure diagnostics -- never read by any
                       # control-flow branch, never affects `passes`.

    def _record_connector(p_from_super, p_to_super, kind):
        if not _record_ledger:
            return
        _connectors.append({"p0": _map_super_pt(p_from_super),
                             "p1": _map_super_pt(p_to_super), "kind": kind})

    def _pick_safe(lx, ly, candidates):
        """candidates: list of (ri, s, e) in row_runs' super-px frame.
        Returns ((ri, entry_col, exit_col, entry_test, dist), any_in_ceiling)
        -- `any_in_ceiling` distinguishes "nothing was even close enough to
        try" from "something was close but failed the mask test", for the
        caller's own counters. Sorted by (distance, entry x) ascending --
        the SAME distance formula the existing row-to-row join already
        computes (`hypot(dx, dy)`), tried nearest-first, first PASSING
        candidate wins (not just literally nearest -- nearest THAT PASSES)."""
        scored = []
        for ri, s, e in candidates:
            if abs(s - lx) <= abs((e - 1) - lx):
                entry_col, exit_col = s, e - 1
            else:
                entry_col, exit_col = e - 1, s
            entry_test = (float(entry_col), float(y_i))
            dist = float(np.hypot(entry_test[0] - lx, entry_test[1] - ly))
            scored.append((dist, entry_col, ri, exit_col, entry_test))
        scored.sort(key=lambda t: (t[0], t[1]))
        any_in_ceiling = False
        for dist, entry_col, ri, exit_col, entry_test in scored:
            if dist <= ceil_super:
                any_in_ceiling = True
                if _seg_in_mask((lx, ly), entry_test):
                    return (ri, entry_col, exit_col, entry_test, dist), any_in_ceiling
        return None, any_in_ceiling

    for k in range(n_scan):
        y_f = k * spacing_super  # exact float row position, SUPERSAMPLED frame
        y_i = int(round(y_f))
        if y_i < 0 or y_i >= diagS:
            runs_per_scanline.append(0)
            continue
        row = rot[y_i] > 0
        if not row.any():
            runs_per_scanline.append(0)
            census_cols_carried_over_blank_row += len(_open_cols)
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        idx = np.flatnonzero(d)
        starts, ends = idx[0::2], idx[1::2]
        merged = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            gap = s - merged[-1][1] if merged else None
            if merged and gap <= gap_super:
                merged[-1] = (merged[-1][0], e)
                continue
            if merged and connect_guarded and gap <= ceil_super:
                # Candidate join: every pixel strictly between the facing
                # endpoints, on THIS row of the dilated raster, must be
                # in-mask (pen radius already accounted for by rot_dil).
                # An empty slice (gap == 0, runs already touching) is
                # vacuously True -- correct, there is no connector to test.
                prev_e = merged[-1][1]
                if rot_dil[y_i, prev_e:s].all():
                    merged[-1] = (merged[-1][0], e)
                    n_joins += 1
                    continue
                else:
                    n_join_refused_offmask += 1
            elif merged and connect_guarded:
                n_join_refused_ceiling += 1
            merged.append((s, e))

        # Runs surviving the length floor, LEFT-TO-RIGHT order, same filter
        # both paths below use.
        row_runs = []
        for s, e in merged:
            run_len_orig = (e - s) / S
            if run_len_orig < min_run_px:
                n_dropped += 1
                dropped_len_total += run_len_orig
                continue
            row_runs.append((s, e))
        runs_per_scanline.append(len(row_runs))

        if not connect_boustrophedon:
            # Default path -- one independent 2-point pass per surviving run.
            for s, e in row_runs:
                x0 = Minv[0, 0] * s + Minv[0, 1] * y_f + Minv[0, 2]
                y0 = Minv[1, 0] * s + Minv[1, 1] * y_f + Minv[1, 2]
                x1 = Minv[0, 0] * (e - 1) + Minv[0, 1] * y_f + Minv[0, 2]
                y1 = Minv[1, 0] * (e - 1) + Minv[1, 1] * y_f + Minv[1, 2]
                passes.append([(float(x0 / S), float(y0 / S)), (float(x1 / S), float(y1 / S))])
            continue

        # Diagnosis-only component-floor DSU update (see block comment
        # above the loop) -- runs for every row that reaches this point,
        # including a `row_runs`-empty one (the loop below then adds
        # nothing and `_dsu_prev_row` correctly becomes `[]`). Ignores the
        # join predicate entirely by design; not read by any control-flow
        # branch below.
        _dsu_new_row = []
        for _s, _e in row_runs:
            _my_id = len(_dsu_parent)
            _dsu_parent.append(_my_id)
            for _ps, _pe, _pid in _dsu_prev_row:
                if not (_e <= _ps or _s >= _pe):
                    _dsu_union(_my_id, _pid)
            _dsu_new_row.append((_s, _e, _my_id))
        _dsu_prev_row = _dsu_new_row

        # Diagnosis-only "traversed components" lookup -- for each
        # SURVIVING run this row, map its midpoint back through the SAME
        # `Minv` the real output uses (general affine, not assumed
        # translation-only) into original mask-pixel coordinates and read
        # off which `cv2.connectedComponents` label it belongs to. A mask
        # component that never contributes a surviving run (e.g. every run
        # in it was dropped by `min_run_px`) never appears here -- this gives
        # the floor of chains achievable if every TRAVERSED component were
        # one pass, distinct from `n_mask_components_4conn`, which counts
        # untraversed components too.
        if _mask_comp_labels is not None:
            _H_orig, _W_orig = mask_u8.shape[:2]
            for _s, _e in row_runs:
                _mx = (_s + _e - 1) / 2.0
                _ox, _oy = _map_super_pt((_mx, float(y_i)))
                _oxi, _oyi = int(round(_ox)), int(round(_oy))
                if 0 <= _oyi < _H_orig and 0 <= _oxi < _W_orig:
                    _lbl = int(_mask_comp_labels[_oyi, _oxi])
                    if _lbl != 0:
                        _traversed_mask_labels.add(_lbl)

        # Boustrophedon path (fill_connect="boustrophedon" only), PER
        # SPAN-COLUMN -- see the state comment above this loop. A run
        # continues an open column only if it x-overlaps that column's most
        # recent run AND that overlap is 1:1 (no split/merge this row,
        # which would make "the" continuation ambiguous) -- ambiguous cases
        # are treated conservatively: flush the old column(s), start fresh
        # column(s), never guess.
        if not row_runs:
            # A row can reach here with zero surviving runs only when every
            # raster run present was filtered by min_run_px (rows with no
            # ink at all never reach this point -- see the `row.any()`
            # check above). Nothing continues any open column across a
            # fully-empty scan position -- flush everything.
            census_end_row_empty += len(_open_cols)
            for _oc in _open_cols:
                passes.append([_map_super_pt(p) for p in _oc["chain"]])
            _open_cols = []
            continue

        overlap_of_col = [[] for _ in _open_cols]
        overlap_of_run = [[] for _ in row_runs]
        for ci, _oc in enumerate(_open_cols):
            for ri, (s, e) in enumerate(row_runs):
                if not (e <= _oc["s"] or s >= _oc["e"]):
                    overlap_of_col[ci].append(ri)
                    overlap_of_run[ri].append(ci)

        new_cols = []
        consumed_cols = set()
        consumed_runs = set()
        for ci, _oc in enumerate(_open_cols):
            if len(overlap_of_col[ci]) != 1:
                continue  # 0 (nothing below) or >=2 (split) -- flush below
            ri = overlap_of_col[ci][0]
            if len(overlap_of_run[ri]) != 1:
                continue  # this run overlaps >=2 open columns (merge) -- flush below
            s, e = row_runs[ri]
            lx, ly = _oc["last_test"]
            # Entry = whichever end of the new run is nearer the column's
            # open end -- this is what makes the traversal self-alternate
            # (L->R then R->L then L->R...) without a separate parity
            # counter: after ending a run at its right edge, the nearest
            # entry into the row below is naturally that row's right edge.
            if abs(s - lx) <= abs((e - 1) - lx):
                entry_col, exit_col = s, e - 1
            else:
                entry_col, exit_col = e - 1, s
            entry_test = (float(entry_col), float(y_i))
            exit_test = (float(exit_col), float(y_i))
            entry_out = (float(entry_col), float(y_f))
            exit_out = (float(exit_col), float(y_f))
            dx = entry_test[0] - lx
            dy = entry_test[1] - ly
            dist = float(np.hypot(dx, dy))
            if dist <= ceil_super and _seg_in_mask((lx, ly), entry_test):
                n_row_joins += 1
                _record_connector(_oc["chain"][-1], entry_out, "row_join")
                _oc["chain"].append(entry_out)
                _oc["chain"].append(exit_out)
                _oc["s"], _oc["e"] = s, e
                _oc["last_test"] = exit_test
                new_cols.append(_oc)
            else:
                if dist > ceil_super:
                    n_row_join_refused_ceiling += 1
                    census_end_refused_ceiling += 1
                else:
                    n_row_join_refused_offmask += 1
                    census_end_refused_offmask += 1
                passes.append([_map_super_pt(p) for p in _oc["chain"]])
                new_cols.append({"s": s, "e": e, "chain": [entry_out, exit_out],
                                  "last_test": exit_test})
                census_start_after_refusal += 1
            consumed_cols.add(ci)
            consumed_runs.add(ri)

        # Phase 2 (connect_split_merge only): a split column (x-overlaps
        # >=2 runs) continues into ONE safe branch instead of always
        # flushing. The other branch(es) are simply left unconsumed --
        # phase 4/the fresh-start loop picks them up, so no separate code path
        # is needed for "the other branch."
        if connect_split_merge:
            for ci, _oc in enumerate(_open_cols):
                if ci in consumed_cols or len(overlap_of_col[ci]) < 2:
                    continue
                cand = [(ri, row_runs[ri][0], row_runs[ri][1])
                        for ri in overlap_of_col[ci] if ri not in consumed_runs]
                if not cand:
                    continue
                lx, ly = _oc["last_test"]
                picked, _ = _pick_safe(lx, ly, cand)
                if picked is None:
                    continue
                ri, entry_col, exit_col, entry_test, dist = picked
                s, e = row_runs[ri]
                entry_out = (float(entry_col), float(y_f))
                exit_out = (float(exit_col), float(y_f))
                n_split_continued += 1
                _record_connector(_oc["chain"][-1], entry_out, "split")
                _oc["chain"].append(entry_out)
                _oc["chain"].append(exit_out)
                _oc["s"], _oc["e"] = s, e
                _oc["last_test"] = (float(exit_col), float(y_i))
                new_cols.append(_oc)
                consumed_cols.add(ci)
                consumed_runs.add(ri)

        # Phase 3 (connect_split_merge only): a merge run (x-overlapped by
        # >=2 open columns, none of which found a clean 1:1 or a split
        # continuation above) is claimed by ONE safe incoming column. The
        # rest of that run's candidate columns are left unconsumed -- they
        # fall through to the existing flush loop and classify as
        # `census_end_merge` ("this column's only candidate run went to
        # someone else").
        if connect_split_merge:
            for ri, (s, e) in enumerate(row_runs):
                if ri in consumed_runs or len(overlap_of_run[ri]) < 2:
                    continue
                cand_cis = [ci for ci in overlap_of_run[ri] if ci not in consumed_cols]
                if not cand_cis:
                    continue
                cand = [(ci, _open_cols[ci]["last_test"]) for ci in cand_cis]
                scored = []
                for ci, (lx, ly) in cand:
                    if abs(s - lx) <= abs((e - 1) - lx):
                        entry_col, exit_col = s, e - 1
                    else:
                        entry_col, exit_col = e - 1, s
                    entry_test = (float(entry_col), float(y_i))
                    dist = float(np.hypot(entry_test[0] - lx, entry_test[1] - ly))
                    scored.append((dist, lx, ci, entry_col, exit_col, entry_test, (lx, ly)))
                scored.sort(key=lambda t: (t[0], t[1]))
                chosen = None
                for dist, lx, ci, entry_col, exit_col, entry_test, oc_pt in scored:
                    if dist <= ceil_super and _seg_in_mask(oc_pt, entry_test):
                        chosen = (ci, entry_col, exit_col, entry_test)
                        break
                if chosen is None:
                    continue
                ci, entry_col, exit_col, entry_test = chosen
                _oc = _open_cols[ci]
                entry_out = (float(entry_col), float(y_f))
                exit_out = (float(exit_col), float(y_f))
                n_merge_continued += 1
                _record_connector(_oc["chain"][-1], entry_out, "merge")
                _oc["chain"].append(entry_out)
                _oc["chain"].append(exit_out)
                _oc["s"], _oc["e"] = s, e
                _oc["last_test"] = (float(exit_col), float(y_i))
                new_cols.append(_oc)
                consumed_cols.add(ci)
                consumed_runs.add(ri)

        # Phase 4 (connect_proximity only): an ORPHAN column (ZERO
        # x-overlapping runs -- phases 2/3 never touch these, both require
        # >=2 overlaps) may continue into the NEAREST unclaimed run whose
        # connector passes the safety test. Deterministic column order
        # (lowest `s` first) so shared candidates resolve the same way
        # every run.
        if connect_proximity:
            orphan_cis = [ci for ci, _oc in enumerate(_open_cols)
                          if ci not in consumed_cols and len(overlap_of_col[ci]) == 0]
            orphan_cis.sort(key=lambda ci: _open_cols[ci]["s"])
            for ci in orphan_cis:
                _oc = _open_cols[ci]
                free_runs = [(ri, row_runs[ri][0], row_runs[ri][1])
                             for ri in range(len(row_runs)) if ri not in consumed_runs]
                if not free_runs:
                    n_proximity_no_candidate += 1
                    continue
                lx, ly = _oc["last_test"]
                picked, any_in_ceiling = _pick_safe(lx, ly, free_runs)
                if picked is None:
                    if any_in_ceiling:
                        n_proximity_refused_offmask += 1
                    else:
                        n_proximity_no_candidate += 1
                    continue
                ri, entry_col, exit_col, entry_test, dist = picked
                s, e = row_runs[ri]
                entry_out = (float(entry_col), float(y_f))
                exit_out = (float(exit_col), float(y_f))
                n_proximity_joins += 1
                _record_connector(_oc["chain"][-1], entry_out, "proximity")
                _oc["chain"].append(entry_out)
                _oc["chain"].append(exit_out)
                _oc["s"], _oc["e"] = s, e
                _oc["last_test"] = (float(exit_col), float(y_i))
                new_cols.append(_oc)
                consumed_cols.add(ci)
                consumed_runs.add(ri)

        # Old columns that found no clean 1:1 continuation this row -- their
        # chain is done, flush it (never guess across a split/merge).
        # Diagnosis-only classification (does not change which columns are
        # flushed or how -- only WHY): `overlap_of_col[ci]`
        # is 0 (no run below at all) or >=2 (split); a column reaching here
        # with exactly 1 overlap was skipped earlier in this function only
        # because that one candidate run is itself claimed by >=2 open
        # columns (merge, column side) -- the three cases are exhaustive by
        # construction of the `consumed_cols` test above.
        for ci, _oc in enumerate(_open_cols):
            if ci in consumed_cols:
                continue
            _n_ov = len(overlap_of_col[ci])
            if _n_ov >= 2:
                census_end_split += 1
            elif _n_ov == 0:
                census_end_no_overlap += 1
                # near_miss: use the SAME Euclidean distance the real join
                # predicate computes (`dist = hypot(dx, dy)` above, dy = this
                # row's y_i minus the column's `last_test` y); an x-distance-
                # only test would ignore the row-to-row dy term the real
                # predicate always includes. Still an UPPER BOUND on true candidates:
                # this only re-derives the distance/ceiling half of the
                # join test, never `_seg_in_mask` (the mask half) -- some
                # near_miss-flagged cases would still fail the mask test if
                # they ever reached it.
                _lx, _ly = _oc["last_test"]
                _near = False
                for _rs, _re in row_runs:
                    _ddx = 0.0 if (_rs <= _lx <= _re) else min(abs(_lx - _rs), abs(_lx - _re))
                    _ddy = float(y_i) - _ly
                    _d_near = float(np.hypot(_ddx, _ddy))
                    if _d_near <= ceil_super:
                        _near = True
                        break
                if _near:
                    census_end_no_overlap_near_miss += 1
            else:
                census_end_merge += 1
            passes.append([_map_super_pt(p) for p in _oc["chain"]])

        # Runs that found no clean 1:1 continuation -- start a fresh column
        # (default L->R entry, matches the non-boustrophedon convention).
        # Diagnosis-only classification, same exhaustiveness argument as
        # above but from the RUN side: `overlap_of_run[ri]` 0 (no open
        # column overlaps this run at all -- includes every first-row run),
        # >=2 (this run is the ambiguous target of a merge), or exactly 1
        # but not consumed (that one column itself had a split, i.e.
        # overlapped >=2 runs -- the run-side mirror of `census_end_split`).
        for ri, (s, e) in enumerate(row_runs):
            if ri in consumed_runs:
                continue
            _n_ov_run = len(overlap_of_run[ri])
            if _n_ov_run == 0:
                census_start_fresh_no_col += 1
            elif _n_ov_run >= 2:
                census_start_fresh_ambiguous += 1
            else:
                census_start_fresh_after_split += 1
            entry_out = (float(s), float(y_f))
            exit_out = (float(e - 1), float(y_f))
            new_cols.append({"s": s, "e": e, "chain": [entry_out, exit_out],
                              "last_test": (float(e - 1), float(y_i))})

        _open_cols = new_cols

    if connect_boustrophedon:
        census_end_eoi += len(_open_cols)
        for _oc in _open_cols:
            passes.append([_map_super_pt(p) for p in _oc["chain"]])

    # Chain-END linking pass (opt-in `connect_endlink`; `_trace_svg` forces it
    # off unless `connect_boustrophedon`, and it is kwargs-only, so no request
    # key reaches it; if one is ever added it must be part of the cache key).
    # GOAL: link a finished chain's free END to another finished chain's free
    # END across ROWS. Phases 1-4 above only continue a column into a run
    # x-overlapping (or, with proximity, nearest on) the VERY NEXT row; this
    # pass targets the ends that failed all of that (`census_end_no_overlap`
    # and `census_end_merge`, both counted BEFORE this block runs and
    # unaffected by it).
    #
    # SAVINGS CEILING: each accepted link reduces the chain count by exactly 1,
    # so the hard ceiling is `n_chains_pre_endlink - n_mask_components_traversed`.
    # Counting END EVENTS (no_overlap + merge) overstates what is reachable:
    # each join consumes 2 ends. This pass does not filter candidates by end
    # category; it considers ALL chain ends generically, though most accepted
    # joins are expected to come from no_overlap/merge ends (the closest
    # cross-row gaps). The Δrow bound, the kind-specific ceiling (both
    # <= `ceil_super`) and greedy (non-optimal) matching keep the realized
    # yield well under the hard ceiling; it grows with
    # `connect_endlink_max_drow` (0 = same-row only, and most no_overlap/merge
    # ends are cross-row by definition).
    #
    # NO NEW JOIN PREDICATE: acceptance is `_seg_in_mask` on the SAME
    # `rot_dil`, at the SAME `ceil_super`-derived (shrunk by Δrow) ceiling
    # every other connector in this function already uses -- this block
    # only supplies new CANDIDATE PAIRS (chain ends, row-bucketed by a Δrow
    # window) for that one predicate to accept or refuse.
    n_endlink_joins = 0
    n_endlink_refused_offmask = 0
    n_endlink_refused_ceiling = 0
    n_endlink_refused_drow = 0
    n_endlink_refused_cycle = 0
    n_endlink_no_partner = 0
    n_chains_pre_endlink = len(passes)
    if _connect_endlink_eff:
        def _to_super(pt):
            # Inverse of `_map_super_pt`: OUTPUT (mask-pixel) frame -> SUPER
            # (rot/diagS, test) frame, via the SAME forward `M` this
            # function built `rot` from (exact up to float rounding --
            # `Minv` is `cv2.invertAffineTransform(M)`).
            xo, yo = pt
            xs, ys = xo * S, yo * S
            xr = M[0, 0] * xs + M[0, 1] * ys + M[0, 2]
            yr = M[1, 0] * xs + M[1, 1] * ys + M[1, 2]
            return (float(xr), float(yr))

        ends = []  # (chain_idx, slot(0=head,1=tail), super_pt, row_k)
        for _ci, _p in enumerate(passes):
            if _p is None or len(_p) < 2:
                continue
            for _slot, _pt in ((0, _p[0]), (1, _p[-1])):
                _sx, _sy = _to_super(_pt)
                _k = int(round(_sy / spacing_super))
                ends.append((_ci, _slot, (_sx, _sy), _k))

        _ends_by_k = {}
        for _idx, (_ci, _slot, _pt, _k) in enumerate(ends):
            _ends_by_k.setdefault(_k, []).append(_idx)

        # Kind-specific ceiling, <= the existing `ceil_super` and shrinking
        # with the row-bound so the Δrow sweep is not confounded by a fixed
        # Euclidean reach.
        _endlink_ceil = (min(ceil_super, (connect_endlink_max_drow + 1) * spacing_super)
                          if ceil_super > 0 else
                          (connect_endlink_max_drow + 1) * spacing_super)

        _uf_parent = list(range(len(passes)))

        def _uf_find(i):
            while _uf_parent[i] != i:
                _uf_parent[i] = _uf_parent[_uf_parent[i]]
                i = _uf_parent[i]
            return i

        # `_head_id`/`_tail_id`: for the chain currently stored at
        # `passes[root]` (root = a UNION-FIND representative, not
        # necessarily the original chain index once merges start), which
        # ORIGINAL (chain_idx, slot) end is exposed at its front/back right
        # at this moment. Updated on every accepted join so later joins onto an
        # already-merged chain still orient correctly.
        _head_id = {}
        _tail_id = {}
        for _ci, _p in enumerate(passes):
            if _p is None or len(_p) < 2:
                continue
            _head_id[_ci] = (_ci, 0)
            _tail_id[_ci] = (_ci, 1)

        # Candidate pairs, ROW-BUCKETED (not all-pairs -- O(n^2) over
        # tens of thousands of ends is too slow): for each end, only
        # search rows within `connect_endlink_max_drow` of its own row.
        # `_j > _i` (index order into `ends`) ensures every unordered pair
        # of ends is generated exactly once.
        _cand_pairs = []
        for _i, (_ci_a, _slot_a, _pt_a, _k_a) in enumerate(ends):
            for _k_b in range(_k_a - connect_endlink_max_drow,
                               _k_a + connect_endlink_max_drow + 1):
                for _j in _ends_by_k.get(_k_b, []):
                    if _j <= _i:
                        continue
                    _ci_b, _slot_b, _pt_b, _k_b2 = ends[_j]
                    if _ci_b == _ci_a:
                        continue
                    _dist = float(np.hypot(_pt_a[0] - _pt_b[0], _pt_a[1] - _pt_b[1]))
                    if _dist > _endlink_ceil:
                        # Only a real "refusal" if the pair WOULD have been
                        # eligible under the general (non-Δrow-shrunk)
                        # ceiling every other connector in this function
                        # uses -- i.e. "refused specifically by the
                        # kind-specific shrink", not merely "far apart".
                        # Millions of in-window-but-implausibly-far pairs on
                        # a real job are non-candidates, not refusals.
                        if _dist <= ceil_super:
                            n_endlink_refused_ceiling += 1
                        continue
                    _cand_pairs.append((_dist, min(_ci_a, _ci_b), max(_ci_a, _ci_b), _i, _j))
        _cand_pairs.sort(key=lambda t: (t[0], t[1], t[2]))

        # Either-end greedy linking: sort all in-window, in-ceiling
        # candidate pairs by (distance, min chain id, max chain id)
        # ascending (same tie-break style `_pick_safe` already uses), accept
        # the first that passes acyclicity (union-find: refuse if both ends'
        # chains already share a root) and the mask predicate. Reversal:
        # whichever side's consumed end is currently its HEAD is reversed so
        # the join always concatenates tail-of-A + head-of-B -- this is a
        # geometric relabelling only (a reversed open polyline is still a
        # valid pen path; the boustrophedon L/R alternation was a
        # convenience for choosing entry columns during the scan, not an
        # invariant later stages depend on). No duplicate points: `A[-1]`
        # and `B[0]` are two DIFFERENT chains' own distinct endpoints by
        # construction, so concatenation never repeats a point (a
        # zero-length connector would require coincident, not merely nearby,
        # endpoints -- possible only if `_dist` rounds to 0.0; such links are
        # counted, not filtered).
        _consumed_end = set()
        for _dist, _lo, _hi, _i, _j in _cand_pairs:
            _ci_a, _slot_a, _pt_a, _k_a = ends[_i]
            _ci_b, _slot_b, _pt_b, _k_b = ends[_j]
            _end_a, _end_b = (_ci_a, _slot_a), (_ci_b, _slot_b)
            if _end_a in _consumed_end or _end_b in _consumed_end:
                continue
            _ra, _rb = _uf_find(_ci_a), _uf_find(_ci_b)
            if _ra == _rb:
                n_endlink_refused_cycle += 1
                continue
            if not _seg_in_mask(_pt_a, _pt_b):
                n_endlink_refused_offmask += 1
                continue
            _chain_ra, _chain_rb = passes[_ra], passes[_rb]
            if _head_id[_ra] == _end_a:
                _chain_ra = list(reversed(_chain_ra))
                _head_id[_ra], _tail_id[_ra] = _tail_id[_ra], _head_id[_ra]
            if _tail_id[_rb] == _end_b:
                _chain_rb = list(reversed(_chain_rb))
                _head_id[_rb], _tail_id[_rb] = _tail_id[_rb], _head_id[_rb]
            assert _tail_id[_ra] == _end_a, "endlink: A not oriented tail-first"
            assert _head_id[_rb] == _end_b, "endlink: B not oriented head-first"
            _new_head, _new_tail = _head_id[_ra], _tail_id[_rb]
            passes[_ra] = _chain_ra + _chain_rb
            passes[_rb] = None
            _uf_parent[_rb] = _ra
            _head_id[_ra], _tail_id[_ra] = _new_head, _new_tail
            del _head_id[_rb]
            del _tail_id[_rb]
            _consumed_end.add(_end_a)
            _consumed_end.add(_end_b)
            n_endlink_joins += 1
            # Tag the connector kind by |Δrow|: a diagonal spanning n>=2 rows
            # draws across n-1 freshly-inked rows, which is near-certain
            # overdraw for a gel pen, so a risk analysis can stratify
            # endlink_d1/d2/d3 instead of blending them into one "endlink"
            # bucket.
            _record_connector(_pt_a, _pt_b, f"endlink_d{abs(_k_a - _k_b)}")

        # `n_endlink_no_partner` / `n_endlink_refused_drow`: among ends that
        # never joined and never even generated an in-window candidate pair,
        # probe EVERY row from one beyond the Δrow window out to the row
        # distance the FULL `ceil_super` could still reach (distance only,
        # never `_seg_in_mask`) purely to attribute "a same-frame candidate
        # existed but only past the Δrow bound" vs "nothing was ever close,
        # even ignoring the bound" -- same upper-bound discipline as
        # `census_end_no_overlap_near_miss` above. Diagnostic only, never
        # affects `passes`.
        _ends_in_any_pair = set()
        for _dist, _lo, _hi, _i, _j in _cand_pairs:
            _ends_in_any_pair.add((ends[_i][0], ends[_i][1]))
            _ends_in_any_pair.add((ends[_j][0], ends[_j][1]))
        _wide_probe_rows = int(np.ceil(ceil_super / spacing_super)) + 1 if spacing_super > 0 else 1
        for _idx, (_ci, _slot, _pt, _k) in enumerate(ends):
            _end = (_ci, _slot)
            if _end in _consumed_end or _end in _ends_in_any_pair:
                continue
            _found_wider = False
            _wide_ks = list(range(_k - _wide_probe_rows, _k - connect_endlink_max_drow)) + \
                list(range(_k + connect_endlink_max_drow + 1, _k + _wide_probe_rows + 1))
            for _k_b in _wide_ks:
                for _j in _ends_by_k.get(_k_b, []):
                    _ci_b, _slot_b, _pt_b, _k_b2 = ends[_j]
                    if _ci_b == _ci:
                        continue
                    _dwide = float(np.hypot(_pt[0] - _pt_b[0], _pt[1] - _pt_b[1]))
                    if _dwide <= ceil_super:
                        _found_wider = True
                        break
                if _found_wider:
                    break
            if _found_wider:
                n_endlink_refused_drow += 1
            else:
                n_endlink_no_partner += 1

        passes = [p for p in passes if p is not None]

    # Diagnosis-only summary -- component-count floors and a chain-length
    # census. GATED on `connect_boustrophedon` because
    # `cv2.connectedComponents` is not free; gating avoids adding compute cost
    # to `fill_connect in ("none", "guarded")` calls that never asked for this
    # diagnosis. Only merged into
    # `_TRACE_AUDIT` when `fill_connect == "boustrophedon"` fired, same gate
    # `_trace_svg` already uses for the row-join keys above.
    if connect_boustrophedon:
        # `_mask_comp_labels` was already computed BEFORE the scan loop
        # (gated the same way) so the per-row traversal lookup above could
        # use it -- reused here, not recomputed.
        _n_mask_components = int(_mask_comp_labels.max())
        _n_mask_components_traversed = len(_traversed_mask_labels)
        _dsu_roots = set()
        for _i in range(len(_dsu_parent)):
            _dsu_roots.add(_dsu_find(_i))
        _n_dsu_components = len(_dsu_roots)
        _chain_lens_px = []
        _chain_npts = []
        for _p in passes:
            _chain_npts.append(len(_p))
            _L = 0.0
            for _a, _b in zip(_p, _p[1:]):
                _L += float(np.hypot(_b[0] - _a[0], _b[1] - _a[1]))
            _chain_lens_px.append(_L)
        _n_2pt_chains = sum(1 for _n in _chain_npts if _n <= 2)
    else:
        _n_mask_components = None
        _n_mask_components_traversed = None
        _n_dsu_components = None
        _n_2pt_chains = None
        _chain_lens_px = []
        _chain_npts = []
    census = {
        "census_end_split": census_end_split,
        "census_end_merge": census_end_merge,
        "census_end_no_overlap": census_end_no_overlap,
        "census_end_no_overlap_near_miss": census_end_no_overlap_near_miss,
        "census_end_refused_offmask": census_end_refused_offmask,
        "census_end_refused_ceiling": census_end_refused_ceiling,
        "census_end_row_empty": census_end_row_empty,
        "census_end_eoi": census_end_eoi,
        "census_end_total": (census_end_split + census_end_merge + census_end_no_overlap
                              + census_end_refused_offmask + census_end_refused_ceiling
                              + census_end_row_empty + census_end_eoi),
        "census_start_after_refusal": census_start_after_refusal,
        "census_start_fresh_no_col": census_start_fresh_no_col,
        "census_start_fresh_ambiguous": census_start_fresh_ambiguous,
        "census_start_fresh_after_split": census_start_fresh_after_split,
        "census_start_total": (census_start_after_refusal + census_start_fresh_no_col
                                + census_start_fresh_ambiguous + census_start_fresh_after_split),
        "census_cols_carried_over_blank_row": census_cols_carried_over_blank_row,
        "n_mask_components_4conn": _n_mask_components,
        "n_mask_components_traversed": _n_mask_components_traversed,
        "n_dsu_components_ignoring_predicate": _n_dsu_components,
        "chain_n2pt_count": _n_2pt_chains,
        "chain_len_px_min": (min(_chain_lens_px) if _chain_lens_px else None),
        "chain_len_px_median": (float(np.median(_chain_lens_px)) if _chain_lens_px else None),
        "chain_len_px_mean": (float(np.mean(_chain_lens_px)) if _chain_lens_px else None),
        "chain_len_px_max": (max(_chain_lens_px) if _chain_lens_px else None),
        "chain_len_px_all": _chain_lens_px,
        "chain_npts_all": _chain_npts,
    }

    audit = {
        "n_scanlines": n_scan,
        "n_runs": sum(runs_per_scanline),
        "n_dropped_short": n_dropped,
        "dropped_len_px_total": float(dropped_len_total),
        "runs_per_scanline": runs_per_scanline,
        "supersample": S,
        "fill_px": float(fill_px),
        "merge_gap_px": float(merge_gap_px),
        "min_run_px": float(min_run_px),
        "angle_deg": float(angle_deg),
        # Guarded-connector counters -- only meaningful when
        # connect_guarded was True; 0/0/0 (not omitted) when False, so a
        # caller reading this dict never needs to branch on whether the
        # feature ran. `_trace_svg` still gates whether these keys reach
        # `_TRACE_AUDIT` on `fill_connect != "none"`.
        "connect_guarded": bool(connect_guarded),
        "connect_max_gap_px": float(connect_max_gap_px),
        "n_joins": n_joins,
        "n_join_refused_offmask": n_join_refused_offmask,
        "n_join_refused_ceiling": n_join_refused_ceiling,
        # Boustrophedon row-to-row counters -- 0/0/0 (not
        # omitted) when connect_boustrophedon is False, same convention as
        # the guarded counters above. `n_chains` is `len(passes)` -- the
        # actual pen-lift count this feature exists to reduce, distinct
        # from `n_runs` (raw surviving scanline runs).
        "connect_boustrophedon": bool(connect_boustrophedon),
        "n_row_joins": n_row_joins,
        "n_row_join_refused_offmask": n_row_join_refused_offmask,
        "n_row_join_refused_ceiling": n_row_join_refused_ceiling,
        "n_chains": len(passes),
        # Proximity/split-merge continuation -- 0/0/0 when their own flag is off,
        # same convention as every other connector family above. NO NEW
        # JOIN PREDICATE -- both modes dispatch through `_pick_safe`, which
        # is `_seg_in_mask` + `ceil_super`, the SAME predicate every other
        # connector in this function already uses.
        "connect_proximity": bool(connect_proximity),
        "connect_split_merge": bool(connect_split_merge),
        "n_split_continued": n_split_continued,
        "n_merge_continued": n_merge_continued,
        "n_proximity_joins": n_proximity_joins,
        "n_proximity_refused_offmask": n_proximity_refused_offmask,
        "n_proximity_no_candidate": n_proximity_no_candidate,
        "n_joins_total": (n_row_joins + n_split_continued + n_merge_continued
                          + n_proximity_joins),
        # Chain-END linking -- 0/0/0/... when connect_endlink is off, same convention as every
        # other connector family above. `n_chains_pre_endlink` is the chain
        # count BEFORE this pass ran (== census_end_total, the sum of the
        # per-row flush classification above, always -- endlink runs AFTER
        # that classification and never revisits it); `n_chains` (below,
        # `len(passes)`) is the count AFTER, so
        # `n_chains == n_chains_pre_endlink - n_endlink_joins` always holds
        # when connect_endlink is on. See the block comment above this
        # pass's own code for the savings ceiling.
        "connect_endlink": bool(_connect_endlink_eff),
        "connect_endlink_max_drow": int(connect_endlink_max_drow),
        "n_chains_pre_endlink": n_chains_pre_endlink,
        "n_endlink_joins": n_endlink_joins,
        "n_endlink_refused_offmask": n_endlink_refused_offmask,
        "n_endlink_refused_ceiling": n_endlink_refused_ceiling,
        "n_endlink_refused_drow": n_endlink_refused_drow,
        "n_endlink_refused_cycle": n_endlink_refused_cycle,
        "n_endlink_no_partner": n_endlink_no_partner,
        # Diagnosis-only connector ledger -- NOT merged into `_TRACE_AUDIT`
        # (would be large); an analysis harness can capture it by wrapping
        # `_zigzag_fill`. Pure diagnostics.
        "connectors": _connectors,
        # Diagnosis-only census -- see block comment above `census = {...}`. Present
        # in this function's own return dict unconditionally (cheap,
        # doesn't affect `passes`); `_trace_svg` gates whether it reaches
        # `_TRACE_AUDIT` on `fill_connect == "boustrophedon"`, same as the
        # row-join keys just above.
        **census,
    }
    return passes, audit


def _zigzag_offmask_ink_pct(passes, mask, pen_px):
    """Pen-down geometry assertion, TRACE-STAGE SCOPE ONLY. Rasterizes
    `_zigzag_fill`'s own raw `passes` output at true pen width (same
    convention as `_trace_svg`'s own `_speck_ink_pct` block: `cv2.polylines`
    at `_stroke_px_for(opts, W)`, `cv2.LINE_8`) and reports the fraction of
    THAT ink lying outside `dilate(mask, pen_radius)`. It runs on `passes`
    BEFORE `_link_pockets` (bypassed for zigzag anyway, see `_trace_svg`'s
    `link_mult` gate), before Stage D's `linemerge`, before Stage E's
    connector router, before the Fill Page clip, and before `gwrite` -- i.e.
    it checks `_zigzag_fill`'s OWN output against the mask and nothing else.
    It cannot observe, and is not a substitute for, any downstream stage
    that could fabricate a connector; that needs a structural check on the
    emitted gcode (threshold-free: cross-scanline-hop count at pass pitch)."""
    import cv2, numpy as np
    H, W = np.asarray(mask).shape[:2]
    if not passes:
        return 0.0
    canvas = np.zeros((H, W), np.uint8)
    for p in passes:
        if len(p) < 2:
            continue
        pts = np.rint(np.asarray(p, np.float32)).astype(np.int32)
        cv2.polylines(canvas, [pts], False, 255, max(1, int(round(pen_px))),
                      lineType=cv2.LINE_8)
    r = max(1, int(round(pen_px / 2.0)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dil_mask = cv2.dilate(np.asarray(mask).astype(np.uint8), k) > 0
    ink = canvas > 0
    total = int(np.count_nonzero(ink))
    off = int(np.count_nonzero(ink & ~dil_mask))
    return 100.0 * off / max(1, total)


# Fallback pen narrow-direction laid width (mm), used ONLY when the caller
# does not supply `fill_pen_narrow_mm` in opts. This is the uRight gel's own
# wedge reading (~0.17mm), NOT a measurement of any other pen -- see the loud caveat this constant's
# only caller (`_zigzag_fill_per_region`) attaches to its audit output
# whenever the fallback fires (`coverage_constraint_source` key).
_FALLBACK_PEN_NARROW_MM = 0.1704


def _zigzag_fill_per_region(mask, fill_px, fixed_angle_deg=0.0, merge_gap_px=0.0,
                             min_run_px=None, frame_offset_deg=0.0,
                             fill_mm=None, pen_narrow_mm=None,
                             min_region_dim_mult=2.0, max_regions_evaluated=400,
                             candidate_step_deg=45.0):
    """Per-region scan angle (opt-in `fill_angle_mode="per_region"`). Per-connected-
    component variant of `_zigzag_fill`: instead of one fixed scan angle for
    the whole mask, each component (`cv2.connectedComponentsWithStats`,
    8-connectivity -- the SAME call `_trace_svg` already makes for dot-cull/
    routing, reused via `_label_moments` for the free principal-axis PCA
    `_route_predicate_batch` already computes) picks its OWN scan angle by
    ACTUALLY generating `_zigzag_fill` on that component's cropped bounding
    box at a small set of candidate angles and scoring the real resulting
    runs -- not a proxy. Component sizes are typically very unequal (on one
    measured job a single component held 62.6% of the ink and the top three
    85.2%), so in practice this mostly chooses the best angle for a handful of
    large regions.

    SCORING FUNCTION (stated once, used everywhere): for a candidate angle,
    generate `_zigzag_fill` on the region crop and score
        score = n_runs + n_dropped_short
    (lower is better; ties broken by higher mean surviving run length).
    Rationale: at fixed pitch the region's total ink length is ~constant
    regardless of angle, so minimizing (surviving runs + dropped runs) is
    equivalent to maximizing mean run length, and it also counts a dropped
    run as a cost -- a dropped run is ink `_zigzag_fill` decided not to draw
    AND a would-be pen-down `_zigzag_fill` didn't even get credit for, so
    ignoring it would reward an angle that quietly throws ink away over one
    that draws a few more short strokes. The goal is the fewest short strokes
    and the most long ones.

    CANDIDATES: {0, candidate_step_deg, 2*candidate_step_deg, ...} spanning
    [0, 180) (default step 45deg -> {0,45,90,135}), UNION the region's own
    principal-axis angle (from `_label_moments`) and its perpendicular,
    each rounded to the nearest degree and reduced mod 180. This is
    cheaper than scanning every 15deg: the per-candidate cost is a full
    `_zigzag_fill` call, and the dominant component's crop can be effectively
    the whole canvas, so evaluating 12 candidates there would cost ~3x this
    function's 4-6-candidate budget.

    TINY-REGION FALLBACK: components whose bbox max dimension is under
    `min_region_dim_mult * min_run_px` cannot fit more than ~2 candidate-
    angle run lengths regardless of angle, so evaluating them wastes time
    for no expected benefit -- they get `fixed_angle_deg` unevaluated. Also
    applies past `max_regions_evaluated` (by descending area) as a hard
    cost cap; both counts and the ink share left on the fixed angle are in
    the returned audit (`n_regions_fallback_tiny`, `n_regions_fallback_cap`,
    `fallback_ink_pct`).

    COVERAGE CONSTRAINT (pen-dependent, so not assumed): this
    function does not know the true anisotropic laid width without a real
    wedge reading for the ACTIVE pen. `pen_narrow_mm` is the caller-supplied,
    wedge-measured narrow-direction laid width in mm; if omitted, this
    function falls back to `_FALLBACK_PEN_NARROW_MM` (the uRight gel's own
    wedge number) and reports that substitution loudly in the audit
    (`coverage_constraint_source`) -- it is NOT a measurement of whatever
    pen is actually active. `overlap = pen_narrow_mm / fill_mm` (fill_mm is
    the REQUIRED real spacing in mm, not px -- pass it explicitly). If
    `overlap < 1.0` the full angle search is UNSAFE (it could select a
    diagonal that opens real gaps on a wide-spaced pen): every region's
    candidate set is restricted to the two MACHINE-frame axis angles {0, 90}
    (converted to image frame via `frame_offset_deg`, image_angle =
    machine_angle - frame_offset_deg, mirroring composite.py's own fixed-
    angle conversion) instead of the full 4-6-candidate set. This branch,
    once decided, applies to the WHOLE job (a job has one active pen), not
    per region.

    mask: 2D bool/uint8 array, ink=nonzero -- same convention as
      `_zigzag_fill`.
    fill_px: scanline pitch, px units (same as `_zigzag_fill`).
    fixed_angle_deg: IMAGE-frame angle used for the tiny-region/capped
      fallback and as one of the always-included candidates.
    frame_offset_deg: `item.rot + paper_rot`, the SAME quantity
      composite.py subtracts from a machine-frame `fill_angle_deg` before
      it reaches `_zigzag_fill` (composite.py `_fill_frame_offset_deg`) --
      needed here only to express the {0,90} machine-frame restriction in
      image-frame terms; the empirical scoring itself is angle-frame-
      agnostic (it just runs `_zigzag_fill` and counts what comes out).

    Returns (passes, audit) where audit merges the same keys `_zigzag_fill`
    returns (aggregated across regions: n_scanlines, n_runs,
    n_dropped_short, dropped_len_px_total) plus per-region detail
    (`regions`: list of {label, area_px, bbox, principal_axis_deg,
    chosen_angle_deg, chosen_angle_machine_deg, candidate_scores,
    evaluated: bool}), cost accounting, and the coverage-constraint
    decision."""
    import time
    import cv2, numpy as np
    t_start = time.time()
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if min_run_px is None:
        min_run_px = fill_px
    if not mask_u8.any() or fill_px <= 0:
        return [], {"n_scanlines": 0, "n_runs": 0, "n_dropped_short": 0,
                     "dropped_len_px_total": 0.0, "regions": [],
                     "n_regions": 0, "n_regions_evaluated": 0,
                     "n_regions_fallback_tiny": 0, "n_regions_fallback_cap": 0,
                     "fallback_ink_pct": 0.0, "eval_seconds": 0.0,
                     "coverage_constraint": {}}

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    moments = _label_moments(lbl, n)
    principal_axis = moments["principal_axis_deg"]  # image-frame, [-90, 90)

    # --- coverage constraint (job-level; decided once) ---
    _narrow_mm = float(pen_narrow_mm) if pen_narrow_mm is not None else _FALLBACK_PEN_NARROW_MM
    _fill_mm = float(fill_mm) if fill_mm is not None else None
    coverage_constraint = {
        "pen_narrow_mm": _narrow_mm,
        "fill_mm": _fill_mm,
        "source": "caller" if pen_narrow_mm is not None else "fallback_uright_wedge",
    }
    restrict_axes_only = True  # safe default if fill_mm is unknown
    if _fill_mm is not None and _fill_mm > 0:
        overlap = _narrow_mm / _fill_mm
        coverage_constraint["overlap"] = round(overlap, 4)
        restrict_axes_only = overlap < 1.0
    coverage_constraint["restrict_axes_only"] = restrict_axes_only

    if restrict_axes_only:
        # MACHINE-frame {0, 90} -> image-frame candidates.
        img0 = (0.0 - frame_offset_deg) % 180.0
        img90 = (90.0 - frame_offset_deg) % 180.0
        base_candidates = sorted({round(img0, 3), round(img90, 3)})
    else:
        base_candidates = sorted({(k * candidate_step_deg) % 180.0
                                   for k in range(int(round(180.0 / candidate_step_deg)))})

    min_region_dim_px = min_region_dim_mult * min_run_px
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([], dtype=np.int64)
    total_ink = int(areas.sum()) if areas.size else 0
    order = (np.argsort(-areas) + 1).tolist() if areas.size else []

    passes_all = []
    agg = {"n_scanlines": 0, "n_runs": 0, "n_dropped_short": 0, "dropped_len_px_total": 0.0}
    regions_audit = []
    n_eval, n_fallback_tiny, n_fallback_cap, fallback_ink = 0, 0, 0, 0

    for rank, i in enumerate(order):
        x0, y0, w, h, area = stats[i]
        too_tiny = max(w, h) < min_region_dim_px
        over_cap = rank >= max_regions_evaluated
        if too_tiny or over_cap:
            angle = fixed_angle_deg
            pad = int(np.ceil(fill_px)) + 2
            xs0, ys0 = max(0, x0 - pad), max(0, y0 - pad)
            xs1, ys1 = min(mask_u8.shape[1], x0 + w + pad), min(mask_u8.shape[0], y0 + h + pad)
            crop = (lbl[ys0:ys1, xs0:xs1] == i)
            p, a = _zigzag_fill(crop, fill_px, angle_deg=angle,
                                 merge_gap_px=merge_gap_px, min_run_px=min_run_px)
            for seg in p:
                passes_all.append([(seg[0][0] + xs0, seg[0][1] + ys0),
                                    (seg[1][0] + xs0, seg[1][1] + ys0)])
            for k in ("n_scanlines", "n_runs", "n_dropped_short"):
                agg[k] += a.get(k, 0)
            agg["dropped_len_px_total"] += a.get("dropped_len_px_total", 0.0)
            if too_tiny:
                n_fallback_tiny += 1
            else:
                n_fallback_cap += 1
            fallback_ink += int(area)
            regions_audit.append({
                "label": int(i), "area_px": int(area), "bbox": [int(x0), int(y0), int(w), int(h)],
                "principal_axis_deg": round(float(principal_axis[i]), 2),
                "chosen_angle_deg": angle, "chosen_angle_machine_deg":
                    round((angle + frame_offset_deg) % 180.0, 2),
                "evaluated": False, "reason": "tiny" if too_tiny else "cap",
            })
            continue

        n_eval += 1
        pax = float(principal_axis[i]) % 180.0
        if restrict_axes_only:
            # Coverage constraint binds job-wide: do NOT widen the search
            # with the principal axis/fixed angle when the pen's narrow
            # direction can't cover a diagonal -- base_candidates is already
            # exactly the machine-frame {0, 90} pair.
            candidates = sorted(set(base_candidates))
        else:
            candidates = sorted(set(base_candidates) |
                                 {round(pax, 2), round((pax + 90.0) % 180.0, 2),
                                  float(fixed_angle_deg) % 180.0})
        pad = int(np.ceil(fill_px)) + 2
        xs0, ys0 = max(0, x0 - pad), max(0, y0 - pad)
        xs1, ys1 = min(mask_u8.shape[1], x0 + w + pad), min(mask_u8.shape[0], y0 + h + pad)
        crop = (lbl[ys0:ys1, xs0:xs1] == i)

        best = None
        scores = {}
        for cand in candidates:
            p, a = _zigzag_fill(crop, fill_px, angle_deg=cand,
                                 merge_gap_px=merge_gap_px, min_run_px=min_run_px)
            n_runs, n_drop = a.get("n_runs", 0), a.get("n_dropped_short", 0)
            score = n_runs + n_drop
            mean_len = 0.0
            if p:
                mean_len = float(np.mean([np.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]) for s in p]))
            scores[round(cand, 2)] = {"score": score, "n_runs": n_runs,
                                       "n_dropped_short": n_drop, "mean_run_len_px": round(mean_len, 3)}
            key = (score, -mean_len)
            if best is None or key < best[0]:
                best = (key, cand, p, a)

        _, chosen, p, a = best
        for seg in p:
            passes_all.append([(seg[0][0] + xs0, seg[0][1] + ys0),
                                (seg[1][0] + xs0, seg[1][1] + ys0)])
        for k in ("n_scanlines", "n_runs", "n_dropped_short"):
            agg[k] += a.get(k, 0)
        agg["dropped_len_px_total"] += a.get("dropped_len_px_total", 0.0)
        regions_audit.append({
            "label": int(i), "area_px": int(area), "bbox": [int(x0), int(y0), int(w), int(h)],
            "principal_axis_deg": round(pax, 2),
            "chosen_angle_deg": round(chosen, 2),
            "chosen_angle_machine_deg": round((chosen + frame_offset_deg) % 180.0, 2),
            "candidate_scores": scores, "evaluated": True,
        })

    audit = dict(agg)
    audit.update({
        "regions": regions_audit,
        "n_regions": int(n - 1),
        "n_regions_evaluated": n_eval,
        "n_regions_fallback_tiny": n_fallback_tiny,
        "n_regions_fallback_cap": n_fallback_cap,
        "fallback_ink_pct": round(100.0 * fallback_ink / max(1, total_ink), 3),
        "eval_seconds": round(time.time() - t_start, 3),
        "coverage_constraint": coverage_constraint,
        "candidate_step_deg": candidate_step_deg,
        "fixed_angle_deg": fixed_angle_deg,
        "fill_px": float(fill_px),
        "merge_gap_px": float(merge_gap_px),
        "min_run_px": float(min_run_px),
    })
    return passes_all, audit


def _multitone_passes(img, ink_mask, fill_px, min_mark_px, tone_mode,
                      n_levels=TONE_LEVELS_N_DEFAULT):
    """Tone generator. Quantizes ink into `n_levels` luminance bands (multiotsu;
    N is a PARAMETER, see the TONE_LEVELS_N_DEFAULT comment above) and fills:
      - the darkest band SOLID, via `_level_set_fill` at spacing=fill_px --
        the exact mechanism the tone-off path uses for its own solid fill (no
        faked darker gray; true-black cross-hatch is not implemented).
      - every lighter band with PARALLEL HATCH (`_hatch_fill`) at a spacing
        continuously mapped from that band's rank into the wedge-measured
        [TONE_WINDOW_LO_MULT, TONE_WINDOW_HI_MULT] x fill_px window.

    Degenerate input (line art / too few distinct ink values for n_levels
    bands) collapses to ONE solid band, the same fill as tone-off. See
    _trace_svg's hole_frac peek, which restores the tone-off hole floor when
    this branch is going to fire, so tone=on line art matches tone=off (the
    same solid tone, not just "one solid tone"). Returns (passes, audit_dict)."""
    import cv2, numpy as np
    from skimage.morphology import remove_small_objects
    gamma = _TONE_KNOB_GAMMA.get(tone_mode, _TONE_KNOB_GAMMA["balanced"])
    n_levels = max(1, int(n_levels))
    vals = img[ink_mask]
    uniq = np.unique(vals)
    # Line art / near-bilevel ink, or n_levels<=1: one tone. Solid-fill it
    # exactly like OFF (same _level_set_fill call OFF's own inline loop mirrors).
    if n_levels <= 1 or uniq.size < max(2, n_levels):
        passes = _level_set_fill(ink_mask, fill_px)
        return passes, {"tone_mode": tone_mode, "tone_thresholds": None,
                        "tone_levels_n": n_levels,
                        "tone_levels": {"solid": {"ink_pct": round(float(ink_mask.mean()) * 100, 2),
                                                 "spacing_mult": 1.0, "passes": len(passes)}},
                        "tone_degenerate": True}
    try:
        from skimage.filters import threshold_multiotsu
        # float64 + explicit nbins is NOT cosmetic. skimage IGNORES `nbins` for
        # integer input and uses one bin per integer value (256), and the
        # multi-Otsu search is combinatorial in bins: C(256,6) ~ 3.7e11 at
        # classes=7, i.e. an effective hang. `vals` is uint8, so passing nbins
        # alone does nothing -- the CAST is what makes it take effect (the
        # even nbins=16 on uint8 input still hangs).
        #
        # Measured timings (two separate runs, not the same measurement):
        #   real ink values, uint8 input:
        #       classes=4 -> 0.05s, classes=5 -> 2.1s, classes=7 -> 30+ min
        #       (killed); with the float64 cast, 0.013s at classes=7.
        #   400k synthetic beta-distributed samples:
        #       classes=5 1.80s -> 0.0035s; classes=7 30+ min -> 0.0151s.
        #
        # Without the cast, tone mode at its default
        # (TONE_LEVELS_N_DEFAULT = 7) would hang.
        #
        # nbins=32 gives 32 levels of resolution for the band EDGES, ample for
        # <=8 bands: the resulting thresholds land within 3-6 (on the 0-255
        # scale) of an equal-population quantile split.
        th = threshold_multiotsu(vals.astype(np.float64), classes=n_levels, nbins=32)
        thresholds = [float(t) for t in th]
    except Exception:
        # fall back to a single median split (still tone-only)
        thresholds = [float(np.median(vals))] * (n_levels - 1)
    bounds = [-1.0] + thresholds + [256.0]
    bands = []
    for i in range(n_levels):
        lo, hi = bounds[i], bounds[i + 1]
        m = ink_mask & (img >= lo) & (img < hi)
        # de-sliver by the same absolute-area floor used for dot-cull, so
        # multiotsu boundary noise doesn't spawn dust loops (geometry only).
        bands.append(remove_small_objects(m, min_size=min_mark_px))

    passes, levels = [], {}
    for i, m in enumerate(bands):
        if i == 0:
            name, sp_mult = "dark", 1.0
            p = _level_set_fill(m, fill_px)
        else:
            name = f"L{i}"
            frac = (i / max(1, n_levels - 1)) ** gamma
            sp_mult = TONE_WINDOW_LO_MULT + frac * (TONE_WINDOW_HI_MULT - TONE_WINDOW_LO_MULT)
            sp_mult = min(TONE_WINDOW_HI_MULT, max(TONE_WINDOW_LO_MULT, sp_mult))
            p = _hatch_fill(m, fill_px * sp_mult)
        passes += p
        levels[name] = {"ink_pct": round(float(m.mean()) * 100, 2),
                        "spacing_mult": round(sp_mult, 3), "passes": len(p)}
    return passes, {"tone_mode": tone_mode,
                    "tone_thresholds": [round(t, 1) for t in thresholds],
                    "tone_levels_n": n_levels,
                    "tone_levels": levels, "tone_degenerate": False}


def _route_predicate_pass(comp_mask) -> bool:
    """Routing geometric predicate — separates
    'elongated thin stroke' (grain hatch, letter stroke, appendage tip) from
    'boundary rim of a compact blob' (rivet, gravel dot) on a single isolated
    connected component. Purely geometric/dimensionless (no absolute-pixel
    constant): PCA elongation ratio, skeleton endpoint count, skeleton branch
    density.
    comp_mask: bool HxW array, single 8-connected component already isolated.

    This single-component entry point crops to the component's own tight
    bounding box before skeletonizing (skeletonize() cost scales with array
    size). It is kept for standalone callers; the hot path in _trace_svg's
    centerline_v2 branch uses the batched `_route_predicate_batch` below
    instead of calling this per component."""
    import cv2, numpy as np
    from skimage.morphology import skeletonize
    ys, xs = np.nonzero(comp_mask)
    area = ys.size
    if area < 2:
        return False
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    if cov.ndim == 0 or np.isnan(cov).any():
        elong = 1.0
    else:
        eigvals = np.linalg.eigvalsh(cov.reshape(2, 2) if cov.size == 4 else np.array([[cov, 0], [0, 0]]))
        eigvals = np.clip(eigvals, 1e-6, None)
        elong = float(np.sqrt(eigvals[-1] / eigvals[0]))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = 2
    H, W = comp_mask.shape
    crop = comp_mask[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
    skel = skeletonize(crop)
    skel_len = int(skel.sum())
    if skel_len >= 1:
        sk = skel.astype(np.uint8)
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        conv = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        neigh = conv[sk == 1] - 10
        endpoints = int((neigh == 1).sum())
        branches = int((neigh >= 3).sum())
    else:
        endpoints, branches = 0, 0
    branch_per_skelpx = branches / max(1, skel_len)
    return (elong >= 2.5) and (endpoints <= 2) and (branch_per_skelpx <= 0.05)


def _label_moments(lbl, n):
    """Per-label 0th/1st/2nd raw moments and principal-axis eigen-decomposition,
    shared by `_route_predicate_batch` (elongation) and the per-region angle
    selector (principal axis), so both use the same bincount/covariance math.

    lbl: int32 label map. n: total label count (0..n-1, label 0 = background).
    Returns a dict of length-n float64 arrays: count, mean_x, mean_y, var_x,
    var_y, cov_xy, lam_hi, lam_lo, elong (sqrt(lam_hi/lam_lo)), and
    principal_axis_deg (0.5*atan2(2*cov_xy, var_x - var_y), in degrees,
    range [-90, 90) -- the image-frame angle of the major axis, equivalent
    mod 180 to any angle+180)."""
    import numpy as np
    H, W = lbl.shape
    lbl_flat = lbl.ravel()
    yy, xx = np.indices((H, W))
    xx = xx.ravel().astype(np.float64)
    yy = yy.ravel().astype(np.float64)
    count = np.bincount(lbl_flat, minlength=n).astype(np.float64)
    sum_x = np.bincount(lbl_flat, weights=xx, minlength=n)
    sum_y = np.bincount(lbl_flat, weights=yy, minlength=n)
    sum_xx = np.bincount(lbl_flat, weights=xx * xx, minlength=n)
    sum_yy = np.bincount(lbl_flat, weights=yy * yy, minlength=n)
    sum_xy = np.bincount(lbl_flat, weights=xx * yy, minlength=n)
    safe_count = np.maximum(count, 1.0)
    mean_x = sum_x / safe_count
    mean_y = sum_y / safe_count
    var_x = sum_xx / safe_count - mean_x ** 2
    var_y = sum_yy / safe_count - mean_y ** 2
    cov_xy = sum_xy / safe_count - mean_x * mean_y
    trace = var_x + var_y
    det = var_x * var_y - cov_xy ** 2
    disc = np.sqrt(np.clip(trace ** 2 / 4.0 - det, 0.0, None))
    lam_hi = np.clip(trace / 2.0 + disc, 1e-6, None)
    lam_lo = np.clip(trace / 2.0 - disc, 1e-6, None)
    elong_all = np.sqrt(lam_hi / lam_lo)
    principal_axis_deg = np.degrees(0.5 * np.arctan2(2.0 * cov_xy, var_x - var_y))
    return {
        "count": count, "mean_x": mean_x, "mean_y": mean_y,
        "var_x": var_x, "var_y": var_y, "cov_xy": cov_xy,
        "lam_hi": lam_hi, "lam_lo": lam_lo, "elong": elong_all,
        "principal_axis_deg": principal_axis_deg,
    }


def _route_predicate_batch(lbl, n, stats, candidate_ids):
    """Vectorized batch form of `_route_predicate_pass` for many candidate
    components against ONE label map. Same label-wise np.bincount trick as
    `_gray_crush_metric` / the fallback-routing block: PCA elongation for
    EVERY label is computed in one pass over the whole image (sum/sumsq/sumxy
    per label via bincount, eigenvalues of the resulting 2x2 covariance in
    closed form) instead of a per-component np.cov call. Elongation is a
    RATIO of the two eigenvalues, so it is invariant to the ddof=1 vs ddof=0
    scaling difference between np.cov and this bincount-moments form (both
    eigenvalues scale by the same constant, which cancels in the ratio) —
    numerically identical decisions to the original per-component np.cov path.

    skeletonize() has no label-wise vectorized equivalent, so it still runs
    once per candidate component — but on that component's CROPPED BOUNDING
    BOX (from connectedComponentsWithStats' `stats`), not the full H*W
    canvas. Calling `_route_predicate_pass(lbl == i)` for each of thousands of
    components would allocate and skeletonize a full-size boolean array every
    time.

    lbl: int32 label map (from cv2.connectedComponentsWithStats).
    n: total label count (0..n-1, label 0 = background).
    stats: the (n,5) stats array from the same connectedComponentsWithStats
      call (columns: x, y, w, h, area).
    candidate_ids: iterable of label ids to evaluate (need not be all of
      0..n-1 — e.g. only zero_labels or only T_new's components).
    Returns: dict {label_id: bool_pass}, matching `_route_predicate_pass`'s
    boolean semantics for each of those ids."""
    import cv2, numpy as np
    from skimage.morphology import skeletonize
    H, W = lbl.shape
    moments = _label_moments(lbl, n)
    count, elong_all = moments["count"], moments["elong"]

    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    result = {}
    for i in candidate_ids:
        if count[i] < 2:
            result[i] = False
            continue
        elong = float(elong_all[i])
        x0, y0, w, h = int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])
        pad = 2
        xs0, ys0 = max(0, x0 - pad), max(0, y0 - pad)
        xs1, ys1 = min(W, x0 + w + pad), min(H, y0 + h + pad)
        crop = (lbl[ys0:ys1, xs0:xs1] == i)
        skel = skeletonize(crop)
        skel_len = int(skel.sum())
        if skel_len >= 1:
            sk = skel.astype(np.uint8)
            conv = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
            neigh = conv[sk == 1] - 10
            endpoints = int((neigh == 1).sum())
            branches = int((neigh >= 3).sum())
        else:
            endpoints, branches = 0, 0
        branch_per_skelpx = branches / max(1, skel_len)
        result[i] = (elong >= 2.5) and (endpoints <= 2) and (branch_per_skelpx <= 0.05)
    return result


def _trace_svg(img, out_path: Path, opts: dict, debug_prefix: str = None,
               native_gray=None) -> Path:
    """TRACE MODE (Sketch V3): source-faithful ink COVERAGE, pure geometry.
    Pen = endmill, binary mask = the whole spec. Width-proportional passes via
    distance-transform LEVEL SETS: pass k = contour(dt > (k+0.5)*fill) so every
    stroke gets round(width/fill) parallel passes spaced at the pen's REAL solid-
    fill width (from the active pen profile). NO thin/thick split, no classifier.
    Rider: components too thin for even one level (max_dt <= 0.5*fill) fall back to
    skeleton centerlines so nothing is dropped; count is logged/audited."""
    import cv2, numpy as np, time
    from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
    t0 = time.time()
    H, W = img.shape[:2]
    # real solid-fill spacing in px: from the active pen profile's measured fill_mm
    # and the true trace scale (mm/px at this resolution + paper).
    fill_px = opts.get("fill_px")
    if fill_px is None:
        fill_mm = float(opts.get("fill_mm", 0.15))
        pw = float(opts.get("paper_w_mm", 180.0)); mg = float(opts.get("margin_mm", 10.0))
        fill_px = fill_mm / max(1e-6, (pw - 2 * mg) / W)
    _fill_px_pre_clamp = float(fill_px)
    fill_px = max(1.2, _fill_px_pre_clamp)
    # Surface the 1.2px representability floor binding, regardless of which
    # upstream path produced fill_px. Reporting only: does not change fill_px,
    # passes, or the written SVG.
    _fill_px_clamp_binds = bool(_fill_px_pre_clamp < 1.2)

    # Opt-in ink-recovery flag. DEFAULT "v1" keeps the speck-drop cull below.
    # Speck-drop is the ONLY ink-destroying stage in the tone=off routing chain: skeleton
    # fallback and band-continuity routing are strictly additive (they only
    # ever append to `passes`, never remove from it), and pocket-link only
    # reorders/chains existing geometry (it can add a short closing seam,
    # never drop a polyline) -- speck-drop is the sole stage in this file
    # that does `passes = [p for p in passes if ...]`, i.e. actually discards
    # geometry. The discarded closed loops are not redundant inner rings:
    # their exclusive footprint (pixels the retained passes cover before but
    # not after the drop) is recorded in the `speck_ink_pct` audit field, and
    # they barely overlap band-continuity's own added ink, so they are real,
    # isolated fill fragments from the base dt-band/skeleton stage, not
    # routing debris.
    # fill_routing="no_speck_drop" skips ONLY the speck-drop cull below,
    # leaving the dt-band generator, skeleton fallback, band-continuity
    # routing, and pocket-link as they are -- the minimal change that keeps
    # the discarded fragments in `passes`.
    _fill_routing = str(opts.get("fill_routing", "v1")).lower()

    # Fill mode (opt-in). Default "contour" takes the dt-band level-set branch
    # below; "zigzag" is gated INSIDE the tone_on == False branch only (tone
    # mode never uses it). Defaults initialized here (not just
    # inside the branch) so the flag-gated _TRACE_AUDIT emission below never
    # NameErrors when tone_on is True or fill_mode != "zigzag".
    _fill_mode = str(opts.get("fill_mode", "contour")).lower()
    _zigzag_audit = {}
    _zigzag_offmask_pct = None
    _fill_angle_mode = "fixed"
    _fill_connect = "none"

    # Diagnostic-only stroke-provenance tagging (opt-in, same style as
    # fill_routing/native_res/pre_blur). OFF by default (no `_prov`
    # bookkeeping is built when unset, and every site that touches it is gated
    # on `_provenance`). ON: tags each polyline
    # with its producer-of-origin at creation, tracks that tag through the
    # speck-drop cull and _link_pockets's proximity merge (which can and does
    # combine rings from different producers into one output chain -- see
    # _link_pockets's own `tags` docstring), and writes a sidecar JSON next to
    # `out_path` (path override via opts["provenance_out"]). Does not change
    # `passes`, `_TRACE_AUDIT`, or the written SVG in any way.
    _provenance = str(opts.get("provenance_tag", "")).lower() in ("1", "true", "on")
    _prov = None

    # native_res (opt-in): threshold at the source's native resolution (gaps are
    # real pixels there), then resize the binary mask to the working grid
    # preserving gaps. Without native_gray, the working-res image is binarized.
    if native_gray is not None:
        mask = _tracev3_mask(native_gray, opts, W, H)
    else:
        mask = _binarize_mask(img, opts)
    # The fill_mm/fill_px back-solve below recovers the true working-grid mm/px
    # only because fill_px was ITSELF derived from fill_mm (the ordinary case).
    # When composite.py's fill_spacing_mm branch supplies fill_px from a
    # REQUESTED mm pitch instead, it also supplies the real mm/px directly
    # (`_mm_per_px_override`) so this assumption keeps holding; without the
    # override the formula below is used.
    _mm_per_px_override = opts.get("_mm_per_px_override")
    if _mm_per_px_override is not None:
        mm_per_px = float(_mm_per_px_override)
    else:
        mm_per_px = float(opts.get("fill_mm", 0.15)) / fill_px
    # Cull specks by ABSOLUTE ink area (mm^2). A pen-relative floor (~1.2px^2)
    # would be a no-op at 0.15mm, and each dust dot costs a full lift cycle.
    # Small != noise (a 0.5mm rivet ~0.2mm^2 = 4x the floor survives) — area only.
    # Default 0.035: a higher floor (0.05) removed real texture (0.65% of ink on
    # a textured reference drawing), while 0.035 removed only 0.40% (true dust).
    min_mark_mm2 = float(opts.get("min_mark_mm2", 0.035))
    min_mark_px = max(2, int(round(min_mark_mm2 / (mm_per_px ** 2))))
    _n0, _lbl0, _st0, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    _ink0 = int(mask.sum())
    _dropped_comps = int((_st0[1:, cv2.CC_STAT_AREA] < min_mark_px).sum()) if _n0 > 1 else 0
    mask = remove_small_objects(mask, min_size=min_mark_px)
    _dropped_ink_pct = (_ink0 - int(mask.sum())) / max(1, _ink0) * 100.0
    # Hole fill floor: holes smaller than hole_frac x the pen footprint
    # (0.25 * pi * (fill_px/2)^2, at least 4 px) are filled. Tone mode loosens
    # this (hole_keep, default 0.08x) so small white detail — pipes behind
    # wheels, gaps in latticework — survives as paper instead of being filled shut.
    #
    # Peek at whether tone will actually multitone THIS input before picking the
    # hole floor. Degenerate/line-art input collapses to ONE solid band, the same
    # fill as tone off (see _multitone_passes), so it must also get the tone-off
    # floor (0.25) rather than the loosened 0.08; otherwise tone=on line art
    # would differ from tone=off on hole handling alone even though both draw
    # "one solid tone". Only genuinely multi-tone input gets the loosened floor.
    tone_mode = str(opts.get("tone", "off")).lower()
    tone_on = tone_mode not in ("", "off", "none")
    tone_levels_n = int(opts.get("tone_levels_n", TONE_LEVELS_N_DEFAULT))
    _tone_will_multitone = False
    if tone_on:
        _vals_peek = img[mask]
        _uniq_peek = np.unique(_vals_peek) if _vals_peek.size else np.array([])
        _tone_will_multitone = tone_levels_n > 1 and _uniq_peek.size >= max(2, tone_levels_n)
    _pen_footprint = np.pi * (fill_px / 2) ** 2
    hole_frac = float(opts.get("hole_keep", 0.08)) if _tone_will_multitone else 0.25
    hole_q = max(4, int(hole_frac * _pen_footprint))
    mask = remove_small_holes(mask, hole_q)

    # Mask gap-close (opt-in `mask_gap_close_px`). Closes mask holes narrower
    # than the requested KERNEL DIAMETER IN TRACE-FRAME PX (not mm -- see the
    # scale note below) BEFORE ring generation sees them, right after the hole
    # fill above; see _mask_gap_close's docstring for the operator. The default
    # 0/None (<=1) leaves `mask` untouched. Scoped to the contour fill path
    # ONLY, so it does not change zigzag output. The gate follows the
    # `(not tone_on) and _fill_mode == ...` idiom used throughout (e.g. the
    # zigzag-only audit block below): tone is a separate generator
    # (_multitone_passes) that fill_mode never reaches, so it is excluded the
    # same way. `tone_on` is decided above (see `_tone_will_multitone`) from the
    # PRE-hole-fill mask, so this block cannot change that decision.
    #
    # Scale note: `mm_per_px` passed to _mask_gap_close below is this
    # function's own back-solve (fill_mm/fill_px), which for a pen with a
    # measured `laid_mm` is off from the TRUE working-grid mm/px by
    # (fill_mm/laid_mm) -- 1.96x for halfpitch, 2.7x for tight3x -- because
    # composite derives fill_px from laid_mm while this back-solve divides by
    # fill_mm. composite corrects this for fill_spacing_mm via
    # `_mm_per_px_override`; that correction is not applied here. Since the knob
    # is a raw pixel count, the mismatch cannot move drawn geometry; it only
    # makes `mask_gap_close_realized_mm_nominal` in the audit a trace-frame
    # label, not a physical width. Any mm/pen-width-denominated exposure of this
    # knob (e.g. a UI control) must go through the true mm/px rather than infer
    # it from this key.
    _gap_close_px = int(opts.get("mask_gap_close_px", 0) or 0)
    _gap_close_active = _gap_close_px > 1 and (not tone_on) and _fill_mode != "zigzag"
    _gap_close_audit = {}
    if _gap_close_active:
        mask, _gap_close_audit = _mask_gap_close(mask, mm_per_px, _gap_close_px, debug_prefix)

    # gap_guarantee (flag-gated) — widen physics-marginal gaps so they survive the pen.
    _gg_marginal = 0
    if str(opts.get("gap_guarantee", "")).lower() in ("1", "true", "on"):
        mask, _gg_marginal = _gap_guarantee(mask, fill_px)

    # Otsu shadow-crush metric — cheap, on the mask we're about to trace; see the
    # _gray_crush_metric docstring.
    _gray_crush_fraction = _gray_crush_metric(img, mask, mm_per_px)

    tone_audit = {}
    _route_audit = {}
    # Opt-in contour variants (ccplotter/core_fill.py): all inert unless
    # core_fill="hybrid" / edge_inset_mm>0 reaches the elif below.
    _hybrid_core_passes = []
    _core_audit = {}
    _zero_thr = 0.5 * fill_px
    _zigzag_n_chainable_pre_speck = None
    if tone_on:
        # Tonal fill (parallel-hatch generator). The `else:` branch below is the
        # plain Trace V3 pass generator; this `if tone_on:` gate is the ONLY way
        # to reach the tone generator, so tone=off never runs it.
        passes, tone_audit = _multitone_passes(img, mask, fill_px, min_mark_px,
                                               tone_mode, n_levels=tone_levels_n)
        n_pass_polys, n_levels = len(passes), len(tone_audit.get("tone_levels", {}))
        zero_labels, fallback_polys = [], []
        if _provenance:
            _prov = ["tone_multitone"] * len(passes)
    else:
        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

        if _fill_mode == "zigzag":
            # `dt` above is KEPT unconditionally -- the zero-pass skeleton fallback
            # just below needs it, and zigzag has the SAME too-thin-for-one-
            # scanline problem contour does.
            _pen_px_zz = max(1.0, float(_stroke_px_for(opts, W)))
            _min_run_px_zz = float(opts.get("fill_zigzag_min_run_px", _pen_px_zz))
            _zz_angle_deg = float(opts.get("fill_angle_deg", 0.0) or 0.0)
            # Per-region angle selection (opt-in fill_angle_mode="per_region").
            # Default "fixed" calls _zigzag_fill with one angle.
            _fill_angle_mode = str(opts.get("fill_angle_mode", "fixed")).lower()
            # Guarded connectors (opt-in `fill_connect="guarded"`). Default
            # "none" calls _zigzag_fill/_zigzag_fill_per_region below with
            # connect_guarded=False (that function's own default). Scoped to
            # fixed-angle zigzag only --
            # `_zigzag_fill_per_region` runs its own per-component
            # `_zigzag_fill` calls on cropped, re-angled sub-rasters with no
            # shared scanline frame across components, so "guarded" there is
            # not implemented; refuse loudly rather than
            # silently ignore the request.
            # Boustrophedon (opt-in `fill_connect="boustrophedon"`, extends
            # guarded). Same per_region restriction as guarded, same reason
            # (no shared scanline frame across per-region crops).
            _fill_connect = str(opts.get("fill_connect", "none")).lower()
            if _fill_connect not in ("none", "guarded", "boustrophedon"):
                raise ValueError(f"fill_connect: unknown value {_fill_connect!r} "
                                  "(expected 'none', 'guarded', or 'boustrophedon')")
            if _fill_connect in ("guarded", "boustrophedon") and _fill_angle_mode == "per_region":
                raise ValueError(f"fill_connect={_fill_connect!r} is not implemented for "
                                  "fill_angle_mode='per_region' (no shared scanline "
                                  "frame across per-region crops) -- unset one of them")
            _connect_guarded = _fill_connect in ("guarded", "boustrophedon")
            _connect_boustrophedon = _fill_connect == "boustrophedon"
            # Proximity / split-merge continuation -- two independently-switchable
            # extensions to the boustrophedon tracker, each OFF by default.
            # Kwargs-only, no request/template plumbing: reachable via
            # `render_opts["fill_zigzag_proximity"/"fill_zigzag_split_merge"]`
            # like every other zigzag opt. Meaningless (and forced off) unless
            # `_connect_boustrophedon` is True.
            #
            # These two keys, plus `fill_zigzag_endlink` /
            # `fill_zigzag_endlink_max_drow`, are read straight off `opts`.
            # Nothing in `ccplotter/routes/core.py` or any template sets them,
            # so they are not reachable from an HTTP request. If any is ever
            # plumbed to a request path, it MUST also be added to
            # `routes/core.py`'s cache-key gate (the one `fill_connect`'s
            # `"guarded"`/`"boustrophedon"` values already go through);
            # otherwise a render cached under `fill_connect` alone would be
            # served for a request that asked for the option on (or off).
            _connect_proximity = _connect_boustrophedon and bool(
                opts.get("fill_zigzag_proximity", False))
            _connect_split_merge = _connect_boustrophedon and bool(
                opts.get("fill_zigzag_split_merge", False))
            _connect_endlink = _connect_boustrophedon and bool(
                opts.get("fill_zigzag_endlink", False))
            _connect_endlink_max_drow = int(opts.get("fill_zigzag_endlink_max_drow", 1))
            # Diagnostic-only ledger opt-in -- lets a harness request the
            # connector ledger (for an overdraw-risk measurement) with BOTH
            # switches off, e.g. to calibrate `row_join`'s own baseline rate.
            # Same not-reachable-from-a-request status, and the same cache-key
            # caveat, as the switches above.
            _connect_record_connectors = bool(
                opts.get("fill_zigzag_record_connectors", False))
            # Gap ceiling (mm, not px). The in-mask test is self-limiting for
            # isotropic holes at ~pen-diameter scale, but a white slit thinner
            # than the pen and parallel to the scan direction is filled by the
            # dilation at ANY length, so an explicit ceiling is the only thing
            # bounding that case. Default 2.0mm: close to double the largest
            # interior-hole width the project's gap detector classifies at
            # typical pen scale (`detect_gaps`, min_width_mm=0.25), while
            # staying far below a full scanline length -- a join beyond this
            # is not "closing a hole", it is stitching two features that
            # happen to share a raster row.
            _connect_max_gap_mm = float(opts.get("fill_connect_max_gap_mm", 2.0))
            _connect_max_gap_px = _connect_max_gap_mm / mm_per_px

            # Perimeter + infill. In this file `passes` IS the whole drawing, and
            # contour mode's OUTERMOST dt-band ring (k=0 below) serves as every
            # shape's outline; replacing `passes` wholesale with zigzag runs
            # would throw the outlines away. Draws
            # `fill_zigzag_perimeter_count` (default 1) dt-band level-set
            # rings FIRST -- the exact same extraction code as the `else`
            # branch just below (k=0..N-1, `dt > (k+0.5)*fill_px`), so a
            # perimeter ring is pixel-for-pixel what contour mode would have
            # drawn at that level -- then zigzags ONLY the interior strictly
            # inside the last perimeter band, so infill meets the perimeter
            # without overdrawing it (the interior boundary at k=N-1 IS the
            # perimeter ring's own inner edge -- no separate inset constant).
            # perimeter_count=0 gives wholesale zigzag (interior_mask = mask,
            # no perimeter passes); it is available but not the default.
            _perimeter_count = max(0, int(opts.get("fill_zigzag_perimeter_count", 1)))
            _perimeter_passes = []
            _interior_mask = mask
            if _perimeter_count > 0:
                _peri_band = None
                for _pk in range(_perimeter_count):
                    _peri_band = (dt > (_pk + 0.5) * fill_px).astype(np.uint8)
                    if not _peri_band.any():
                        break
                    _cnts, _ = cv2.findContours(_peri_band, cv2.RETR_LIST,
                                                 cv2.CHAIN_APPROX_TC89_L1)
                    for _c in _cnts:
                        if len(_c) >= 2:
                            _cc = cv2.approxPolyDP(_c, 0.4, True).reshape(-1, 2)
                            if len(_cc) >= 2:
                                _p = [(float(x), float(y)) for x, y in _cc]
                                _p.append(_p[0])  # closed ring
                                _perimeter_passes.append(_p)
                _interior_mask = (_peri_band > 0) if _peri_band is not None else np.zeros_like(mask)

            if _fill_angle_mode == "per_region":
                interior_passes, _zigzag_audit = _zigzag_fill_per_region(
                    _interior_mask, fill_px, fixed_angle_deg=_zz_angle_deg, merge_gap_px=0.0,
                    min_run_px=_min_run_px_zz,
                    frame_offset_deg=float(opts.get("_fill_frame_offset_deg", 0.0) or 0.0),
                    fill_mm=opts.get("fill_mm"),
                    pen_narrow_mm=opts.get("fill_pen_narrow_mm"))
            else:
                interior_passes, _zigzag_audit = _zigzag_fill(
                    _interior_mask, fill_px, angle_deg=_zz_angle_deg, merge_gap_px=0.0,
                    min_run_px=_min_run_px_zz, connect_guarded=_connect_guarded,
                    connect_max_gap_px=(_connect_max_gap_px if _connect_guarded else 0.0),
                    connect_pen_px=_pen_px_zz,
                    connect_boustrophedon=_connect_boustrophedon,
                    connect_proximity=_connect_proximity,
                    connect_split_merge=_connect_split_merge,
                    connect_record_connectors=_connect_record_connectors,
                    connect_endlink=_connect_endlink,
                    connect_endlink_max_drow=_connect_endlink_max_drow)
            passes = _perimeter_passes + interior_passes
            n_pass_polys, n_levels = len(passes), _zigzag_audit.get("n_scanlines", 0)
            if _provenance:
                _prov = (["zigzag_perimeter"] * len(_perimeter_passes) +
                         ["zigzag_infill"] * len(interior_passes))
            # Pen-down geometry assertion, TRACE-STAGE SCOPE ONLY (see
            # _zigzag_offmask_ink_pct's docstring) -- computed on the COMBINED
            # perimeter+infill `passes` against the FULL mask (not just the
            # interior), before the pocket-link stage (applied only to the
            # chainable slice, see link_mult below) and before speck-drop (which
            # is inert on open runs anyway), and before Stage D/E/clip/gwrite.
            # It cannot observe anything those downstream stages could
            # fabricate.
            _zigzag_offmask_pct = _zigzag_offmask_ink_pct(passes, mask, _pen_px_zz)
        elif (opts.get("core_fill") in ("hybrid", "rows", "outline_rows")
              or float(opts.get("edge_inset_mm") or 0.0) > 0.0):
            # Opt-in hybrid core fill and/or edge inset. mm flags need the TRUE
            # mm/px: composite.py supplies the placement cell as `_place_cell`
            # (only when a flag is on) and the scale is derived below from this
            # trace's own mask bbox. The fill_mm/fill_px back-solve is the
            # labelled fallback for callers without a placement (it is off by
            # 1/0.75 or worse for pens whose laid width differs from fill_mm,
            # which is why it is not used when the true scale exists).
            from ccplotter.core_fill import contour_passes, ROWS_ANGLE_DEG, OUTLINE_ROWS_RINGS
            _tm = None
            _flag_src = "backsolved_fill_mm_over_fill_px"
            _pc = opts.get("_place_cell")
            if _pc:
                # same arithmetic as composite's final fit (fit = 0.9 * min(cell / rotated
                # ink bbox) * scale), with the ink bbox taken from this trace's own final
                # mask (rings sit inside it; the difference is a pixel or two in thousands)
                _ys, _xs = np.nonzero(mask)
                if len(_xs):
                    from ccplotter.config import PX2MM as _PX2MM
                    from ccplotter.composite import _rotated_extent_mm as _rot_ext
                    _bw = float(_xs.max() - _xs.min() + 1) * _PX2MM
                    _bh = float(_ys.max() - _ys.min() + 1) * _PX2MM
                    _cw, _ch, _rot, _scl, _true_size = _pc
                    _rw, _rh = _rot_ext(_bw, _bh, _rot)
                    _fit = 1.0 if _true_size else min(_cw / _rw, _ch / _rh) * 0.9
                    _tm = _fit * float(_scl) * _PX2MM
                    _flag_src = "placed_fit_from_mask_bbox"
            _flag_mm_px = float(_tm) if _tm else mm_per_px
            _hyb = opts.get("core_fill") == "hybrid"
            # core_fill="rows": default rings, then back-and-forth rows in the
            # cores of LARGE regions whatever their shape (core_fill.contour_passes rows_cfg)
            _rows_mode = opts.get("core_fill") == "rows"
            # core_fill="outline_rows": ONE outline ring, then rows over the whole
            # interior (core_fill.contour_passes_outline_rows); no core_r_mm, no size rule
            _orows = opts.get("core_fill") == "outline_rows"
            # outline_rows cannot honour an inset (the route refuses it); never raise _zero_thr
            # or record edge_inset_* for a ring that is not drawn
            _inset_mm = 0.0 if _orows else float(opts.get("edge_inset_mm") or 0.0)
            _core_r_mm = 0.0 if _orows else float(opts.get("core_r_mm") or (1.0 if _rows_mode else 0.45))
            _core_r_px = (_core_r_mm / _flag_mm_px) if (_hyb or _rows_mode) else 0.0
            _level0 = (_inset_mm / _flag_mm_px + 0.5) if _inset_mm > 0 else 0.5 * fill_px
            if _inset_mm > 0:
                _zero_thr = _level0   # thinner than the inset ring can sit -> skeleton, never dropped
            _rings, _hybrid_core_passes, _core_mask, _cau = contour_passes(
                dt, fill_px, _level0, _core_r_px, _inset_mm > 0, int(dt.max() / fill_px) + 2,
                rows_cfg=({"angle_deg": ROWS_ANGLE_DEG, **({"outline_rows": True} if _orows else {})}
                          if (_rows_mode or _orows) else None))
            passes = _rings
            n_pass_polys, n_levels = len(_rings) + len(_hybrid_core_passes), _cau["n_levels"]
            if debug_prefix and (_hyb or _rows_mode or _orows):
                cv2.imwrite(f"{debug_prefix}_core_mask.png", _core_mask)
            _core_audit = {"flag_mm_per_px": round(_flag_mm_px, 6), "flag_mm_per_px_source": _flag_src,
                           **({"core_fill": ("outline_rows" if _orows else ("rows" if _rows_mode else "hybrid")),
                               "core_r_mm": (None if _orows else _core_r_mm),
                               "core_r_px": (None if _orows else round(_core_r_px, 2)),
                               "n_core_regions": _cau["n_core_regions"], "n_core_rows": _cau["n_core_rows"],
                               "n_core_axis_lines": _cau["n_core_lines"],
                               "n_core_skipped": _cau["n_core_skipped"],
                               "n_core_curved_ring_levels": _cau["n_core_curved_ring"],
                               "n_core_curved_unique": _cau["n_core_curved_unique"],
                               "n_core_passes": len(_hybrid_core_passes),
                               **({"rows_angle_deg": ROWS_ANGLE_DEG,
                                   "n_core_small_ring_levels": _cau["n_core_small_ring"]}
                                  if (_rows_mode or _orows) else {}),
                               **({"outline_rings": OUTLINE_ROWS_RINGS, "n_outline_rings": _cau["n_rings"]}
                                  if _orows else {})}
                              if (_hyb or _rows_mode or _orows) else {}),
                           **({"edge_inset_mm": _inset_mm, "edge_inset_px": round(_inset_mm / _flag_mm_px, 3)}
                              if _inset_mm > 0 else {})}
            if _provenance:
                _prov = ["primary"] * len(passes)
        else:
            # width-proportional level-set passes (raster; fast, no shapely)
            passes = []
            k, maxk = 0, int(dt.max() / fill_px) + 2
            while k < maxk:
                band = (dt > (k + 0.5) * fill_px).astype(np.uint8)
                if not band.any():
                    break
                cnts, _ = cv2.findContours(band, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
                for c in cnts:
                    if len(c) >= 2:
                        cc = cv2.approxPolyDP(c, 0.4, True).reshape(-1, 2)
                        if len(cc) >= 2:
                            p = [(float(x), float(y)) for x, y in cc]
                            p.append(p[0])                          # closed ring
                            passes.append(p)
                k += 1
            n_pass_polys, n_levels = len(passes), k
            if _provenance:
                _prov = ["primary"] * len(passes)

        # RIDER: skeleton fallback for zero-pass components (max_dt <= 0.5*fill)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        maxdt = np.zeros(n, np.float32)
        np.maximum.at(maxdt, lbl.ravel(), dt.ravel())
        zero_labels = [i for i in range(1, n) if maxdt[i] <= _zero_thr]
        fallback_polys = []
        _texture_routing = str(opts.get("texture_routing", "off")).lower()
        _routing_v2 = _texture_routing == "centerline_v2"
        _routing_bc = _texture_routing == "band_continuity_v1"
        _route_audit = {}
        if not _routing_v2:
            if zero_labels:
                zmask = np.isin(lbl, zero_labels)
                fallback_polys = _skeleton_to_polylines(skeletonize(zmask), prune_len=max(2, int(fill_px)))
                if _provenance:
                    _prov += ["fallback_skeleton_zero_pass"] * len(fallback_polys)
        else:
            # texture_routing="centerline_v2" (opt-in): split zero_labels by the
            # geometric predicate. PASSED (elongated open stroke) -> smoothed
            # centerline path. REJECTED (compact/closed-loop, e.g.
            # rivets/gravel) -> the raw, unsmoothed skeleton fallback used when
            # texture_routing is off.
            _zl_verdict = _route_predicate_batch(lbl, n, stats, zero_labels)
            zl_pass = [i for i in zero_labels if _zl_verdict[i]]
            zl_reject = [i for i in zero_labels if not _zl_verdict[i]]
            if zl_reject:
                zmask_reject = np.isin(lbl, zl_reject)
                _add = _skeleton_to_polylines(
                    skeletonize(zmask_reject), prune_len=max(2, int(fill_px)))
                fallback_polys += _add
                if _provenance:
                    _prov += ["fallback_centerline_v2_zero_reject"] * len(_add)

            # T_new: per-pixel dt<=0.5*fill_px, EXCLUDING pixels already claimed
            # by a zero_labels component (candidate v1's overclaim population).
            # Predicate-passing T_new components (thin appendages hanging off a
            # thick parent) are claimed via the same smoothed-centerline path;
            # predicate-rejected T_new stays unclaimed.
            T = mask.astype(bool) & (dt <= 0.5 * fill_px)
            zmask_all = np.isin(lbl, zero_labels) if zero_labels else np.zeros_like(mask, dtype=bool)
            T_new = T & (~zmask_all)
            nT, lblT, statsT, _ = cv2.connectedComponentsWithStats(T_new.astype(np.uint8), 8)
            _tnew_verdict = _route_predicate_batch(lblT, nT, statsT, range(1, nT))
            tnew_pass = [i for i in range(1, nT) if _tnew_verdict[i]]
            tnew_reject_n = sum(1 for i in range(1, nT) if not _tnew_verdict[i])

            centerline_polys = []
            _cl_prov = [] if _provenance else None
            if zl_pass:
                zmask_pass = np.isin(lbl, zl_pass)
                _add = _skeleton_to_polylines(
                    skeletonize(zmask_pass), prune_len=max(2, int(fill_px)))
                centerline_polys += _add
                if _provenance:
                    _cl_prov += ["fallback_centerline_v2_zero_pass"] * len(_add)
            if tnew_pass:
                tmask_pass = np.isin(lblT, tnew_pass)
                _add = _skeleton_to_polylines(
                    skeletonize(tmask_pass), prune_len=max(2, int(fill_px)))
                centerline_polys += _add
                if _provenance:
                    _cl_prov += ["fallback_centerline_v2_tnew"] * len(_add)
            centerline_polys = _smooth_polys(centerline_polys)  # 1:1, order-preserving
            fallback_polys += centerline_polys
            if _provenance:
                _prov += _cl_prov

            _route_audit = {
                "route_zero_pass": len(zl_pass), "route_zero_reject": len(zl_reject),
                "route_tnew_components": nT - 1,
                "route_tnew_pass": len(tnew_pass), "route_tnew_reject": tnew_reject_n,
            }

        if _routing_bc:
            # texture_routing="band_continuity_v1" (opt-in). Disjoint scope from
            # centerline_v2: operates ONLY within non-zero_pass components
            # (those whose max_dt already clears 0.5*fill_px and so already get a
            # k=0 band contour above), not on zero_labels/T_new (centerline_v2's
            # population). The `if not _routing_v2:` branch just above still runs
            # the plain zero_labels skeleton fallback, since _routing_v2 is False
            # whenever _routing_bc is True (same opts key, mutually exclusive
            # string values).
            #
            # Unit = "pinch run": a maximal connected run of a component's
            # own skeleton pixels whose LOCAL dt sits <=0.5*fill_px despite
            # the component's GLOBAL max_dt clearing that floor — the sub-region the k=0 raster band drops out on
            # (serifs, letter counters, plank/rivet contact throats).
            #
            # No-double-draw: `band` (dt>0.5*fill_px,
            # already contoured above, untouched) and `T = dt<=0.5*fill_px`
            # are a strict-inequality split of the SAME dt field -> disjoint
            # by construction. This block only ever draws within a firing
            # run's own dilated footprint intersected with the GLOBAL ink
            # mask (equivalent to intersecting with the component's own mask,
            # since distinct 8-connected components never touch by
            # definition) — never touches `band` pixels, so no ink region
            # can receive both a contour-band pass and a centerline pass.
            #
            # Firing predicate: fire a pinch run iff it does NOT abut a skeleton
            # branch point (degree>=3 elsewhere in the component's own full
            # skeleton -- the signature of a bridge inside a networked/branchy
            # mesh, e.g. rivet/gravel texture contact, rather than a simple
            # mid-stroke pinch) AND its length is >=2px (excludes single-pixel
            # raster-noise corner nicks). Junction-touching runs are left
            # alone (their band fragment survives or is dropped by the speck
            # filter). This is a partial signal; the real check on it is
            # whether it creates a storm of dots on textured sources.
            non_zero_ids = [i for i in range(1, n)
                            if maxdt[i] > 0.5 * fill_px and int(stats[i, cv2.CC_STAT_AREA]) >= 4]
            _pinch_total = 0
            _pinch_fired = 0
            _pinch_rej_junction = 0
            _pinch_rej_short = 0
            _bc_kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
            bc_footprint = np.zeros((H, W), dtype=bool)
            for i in non_zero_ids:
                x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                              int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
                pad = 2
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
                comp_mask = (lbl[y0:y1, x0:x1] == i)
                skel = skeletonize(comp_mask)
                sk = skel.astype(np.uint8)
                if sk.sum() == 0:
                    continue
                conv = cv2.filter2D(sk, -1, _bc_kernel, borderType=cv2.BORDER_CONSTANT)
                degree = np.where(sk == 1, conv - 10, 0)
                thin = skel & (dt[y0:y1, x0:x1] <= 0.5 * fill_px)
                if not thin.any():
                    continue
                nrun, rlbl, rstats, _ = cv2.connectedComponentsWithStats(thin.astype(np.uint8), 8)
                if nrun <= 1:
                    continue
                # Vectorized junction-adjacency: a per-run
                # cv2.dilate(run_mask, 3x3) over the WHOLE crop is
                # O(n_runs * crop_px) -- far too slow on a whole-page fused-ink
                # component (millions of px, thousands of runs). Instead: find every high-degree
                # ("junction") skeleton pixel ONCE per component (cheap,
                # global boolean), then look up which run label(s) sit in
                # its immediate 8-neighborhood via direct coordinate
                # indexing (touches only as many pixels as there are
                # junction pixels x 8, not runs x crop size) -- same
                # adjacency relation (run pixel within 1px of a degree>=3
                # skeleton pixel outside the run), same predicate result.
                jy, jx = np.nonzero(sk.astype(bool) & (degree >= 3))
                touching_runs = set()
                if jy.size:
                    ch, cw = thin.shape
                    # exclude a run's OWN pixels from counting as its own
                    # "external" junction (equivalent to a per-run
                    # `border = dil & sk & ~run_mask` test: an internal branch
                    # point inside a run must not make that same run
                    # self-report as junction-touching).
                    rlbl_self = rlbl[jy, jx]
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny = np.clip(jy + dy, 0, ch - 1)
                            nx = np.clip(jx + dx, 0, cw - 1)
                            nl = rlbl[ny, nx]
                            valid = (nl > 0) & (nl != rlbl_self)
                            nz = nl[valid]
                            if nz.size:
                                touching_runs.update(np.unique(nz).tolist())
                fire_ids = []
                for j in range(1, nrun):
                    _pinch_total += 1
                    rlen = int(rstats[j, cv2.CC_STAT_AREA])
                    if j in touching_runs:
                        _pinch_rej_junction += 1
                        continue
                    if rlen < 2:
                        _pinch_rej_short += 1
                        continue
                    _pinch_fired += 1
                    fire_ids.append(j)
                # np.isin over the crop ONCE (not per firing run): doing
                # `fire_mask |= (rlbl == j)` inside the loop would hit the same
                # O(n_runs * crop_px) cost as a per-run junction dilate.
                if fire_ids:
                    fire_mask = np.isin(rlbl, fire_ids)
                    bc_footprint[y0:y1, x0:x1] |= fire_mask
            bc_polys = []
            if bc_footprint.any():
                # Dilate each firing run's own footprint by ~1 fill_px so the
                # re-skeletonize below recovers well-formed endpoints that
                # abut the surviving band contour, then restrict to actual ink (global mask
                # intersection == per-component intersection here, since
                # distinct components never touch).
                _bc_r = max(1, int(round(fill_px)))
                _bc_dilate_kernel = np.ones((2 * _bc_r + 1, 2 * _bc_r + 1), np.uint8)
                dil_footprint = cv2.dilate(bc_footprint.astype(np.uint8), _bc_dilate_kernel).astype(bool)
                dil_footprint &= mask.astype(bool)
                bc_polys = _skeleton_to_polylines(skeletonize(dil_footprint), prune_len=max(2, int(fill_px)))
                bc_polys = _smooth_polys(bc_polys)
            fallback_polys += bc_polys
            if _provenance:
                _prov += ["fallback_band_continuity"] * len(bc_polys)
            _route_audit = {
                **_route_audit,
                "pinch_runs_total": _pinch_total,
                "pinch_runs_fired": _pinch_fired,
                "pinch_runs_rejected": _pinch_total - _pinch_fired,
                "pinch_runs_rejected_junction": _pinch_rej_junction,
                "pinch_runs_rejected_short": _pinch_rej_short,
            }
        if (not tone_on) and _fill_mode == "zigzag":
            # Reorder so the CHAINABLE populations (perimeter
            # rings + skeleton-fallback dashes -- both contour-identical
            # geometry: the same dt-band level-set rings / same skeleton
            # centerlines contour mode draws) sit CONTIGUOUS at the front of
            # `passes`, with the scanline INFILL population held as a
            # separate contiguous tail. This is what lets the link_mult
            # block below run `_link_pockets` on the chainable slice only,
            # by simple index split, without ever handing it the infill
            # runs (unsafe -- `_link_pockets` never tests a connector against
            # the mask). Provenance tags (when requested) are reordered to
            # match, from the SAME pre-reorder `_prov` list built above
            # (zigzag_perimeter + zigzag_infill + fallback_* tags, in that
            # order).
            passes = _perimeter_passes + fallback_polys + interior_passes
            if _provenance and _prov is not None:
                _n_peri, _n_inf, _n_fb = (len(_perimeter_passes), len(interior_passes),
                                           len(fallback_polys))
                _prov_peri = _prov[:_n_peri]
                _prov_inf = _prov[_n_peri:_n_peri + _n_inf]
                _prov_fb = _prov[_n_peri + _n_inf:_n_peri + _n_inf + _n_fb]
                _prov = _prov_peri + _prov_fb + _prov_inf
            _zigzag_n_chainable_pre_speck = len(_perimeter_passes) + len(fallback_polys)
        else:
            passes += fallback_polys

    # Drop tiny CLOSED-speck polylines (dust loops shorter than ~3 pen
    # widths). Open strokes (real thin lines / skeleton rescues) are kept.
    #
    # fill_routing="no_speck_drop" (opt-in, see the flag comment above this
    # function's `fill_px` setup): skip this cull entirely. DEFAULT ("v1")
    # runs the cull. _pre_speck/_speck_polys/n_speck_dropped/_speck_ink_pct
    # are computed in both branches so the audit dict and the perf log line
    # have the same shape; only their VALUES differ when the cull is skipped
    # (0 dropped).
    _speck_len = 3.0 * fill_px
    def _is_speck(p):
        return (len(p) >= 3 and abs(p[0][0] - p[-1][0]) < 0.6 and abs(p[0][1] - p[-1][1]) < 0.6
                and cv2.arcLength(np.array(p, np.float32), True) < _speck_len)
    _pre_speck = len(passes)
    if _fill_routing == "no_speck_drop":
        _speck_polys = []
    else:
        _speck_keep = [not _is_speck(p) for p in passes]  # same predicate, same order
        _speck_polys = [p for p, keep in zip(passes, _speck_keep) if not keep]
        if _provenance and _prov is not None:
            _prov = [t for t, keep in zip(_prov, _speck_keep) if keep]
        passes = [p for p, keep in zip(passes, _speck_keep) if keep]
    n_speck_dropped = len(_speck_polys)
    # Bookkeeping: how many of the chainable (perimeter+fallback)
    # prefix survived the speck cull, so the link_mult split below still
    # knows exactly where the untouched infill tail begins in `passes`
    # (order-preserving cull, same predicate/order as everywhere else in
    # this function).
    if _zigzag_n_chainable_pre_speck is not None:
        if _fill_routing == "no_speck_drop":
            _zigzag_n_chainable = _zigzag_n_chainable_pre_speck
        else:
            _zigzag_n_chainable = sum(1 for keep in _speck_keep[:_zigzag_n_chainable_pre_speck] if keep)
    else:
        _zigzag_n_chainable = None
    # Counts alone conceal whether a few filtered loops carry meaningful ink.
    # Measure the removed *visible* footprint at the same pen width the SVG
    # preview uses. This is instrumentation only (it does not change `passes`),
    # and the metric is the exclusive pre/post
    # raster difference so overlap with retained paths is not over-counted.
    _speck_ink_pct = 0.0
    if _speck_polys:
        _pen_px = max(1, int(round(_stroke_px_for(opts, W))))
        _all_ink = np.zeros((H, W), np.uint8)
        _kept_ink = np.zeros((H, W), np.uint8)
        for _canvas, _polys in ((_all_ink, passes + _speck_polys),
                                (_kept_ink, passes)):
            for _poly in _polys:
                if len(_poly) < 2:
                    continue
                _pts = np.rint(np.asarray(_poly, np.float32)).astype(np.int32)
                _closed = (len(_poly) >= 3 and
                           abs(_poly[0][0] - _poly[-1][0]) < 0.6 and
                           abs(_poly[0][1] - _poly[-1][1]) < 0.6)
                cv2.polylines(_canvas, [_pts], _closed, 255, _pen_px,
                              lineType=cv2.LINE_8)
        _all_ink_n = int(np.count_nonzero(_all_ink))
        _speck_ink_pct = (100.0 * int(np.count_nonzero((_all_ink > 0) &
                                                        (_kept_ink == 0))) /
                          max(1, _all_ink_n))

    # RIDER (speed): link nested rings into continuous paths -> far fewer pen-lifts
    # (faster gcode + kinder plot), coverage unchanged. pocket_link = link distance
    # in multiples of fill (0 disables).
    lifts_before = len(passes) + len(_hybrid_core_passes)
    link_mult = float(opts.get("pocket_link", 3.0))
    _is_zigzag_link = (not tone_on) and _fill_mode == "zigzag"
    _zigzag_lifts_chainable_before_link = None
    _zigzag_lifts_chainable_after_link = None
    if _is_zigzag_link:
        # `_link_pockets` must not touch the zigzag scanline INFILL
        # population: it never tests a connector against the mask, which is
        # unsafe across the interior's own holes. It is NOT unsafe
        # for the perimeter rings or the skeleton-fallback dashes: both are
        # contour-identical geometry (the same dt-band level-set rings /
        # same skeleton centerlines contour mode draws at those same
        # levels), and contour mode chains them via this same call.
        # `passes` was reordered above (chainable prefix, infill tail) so
        # this is a simple index split, never a mask-aware decision on the
        # infill runs themselves -- they are sliced OUT before
        # `_link_pockets` is ever called, not filtered by it.
        _n_chain = _zigzag_n_chainable if _zigzag_n_chainable is not None else 0
        _chainable = passes[:_n_chain]
        _infill_tail = passes[_n_chain:]
        _zigzag_lifts_chainable_before_link = len(_chainable)
        assert len(_chainable) + len(_infill_tail) == len(passes)
    # pocket_close (opt-in): force (default) | guarded (never bridges an open
    # fragment's own end->start seam with fake ink beyond link_dist -- see
    # _link_pockets docstring).
    pocket_close = str(opts.get("pocket_close", "force"))
    _prov_pre_link_census = None
    if _provenance and _prov is not None:
        _prov_pre_link_census = {}
        for _t in _prov:
            _prov_pre_link_census[_t] = _prov_pre_link_census.get(_t, 0) + 1
    if _is_zigzag_link:
        if link_mult > 0 and len(_chainable) > 2:
            if _provenance and _prov is not None:
                _chain_tags = _prov[:_n_chain]
                _infill_tags = _prov[_n_chain:]
                _chainable, _chain_tags = _link_pockets(
                    _chainable, link_mult * fill_px, pocket_close=pocket_close,
                    tags=_chain_tags)
                _prov = _chain_tags + [[t] for t in _infill_tags]
            else:
                _chainable = _link_pockets(_chainable, link_mult * fill_px,
                                            pocket_close=pocket_close)
        elif _provenance and _prov is not None:
            _prov = [[t] for t in _prov]
        passes = _chainable + _infill_tail
        _zigzag_lifts_chainable_after_link = len(_chainable)
        # The infill tail is untouched by construction (sliced out above,
        # never passed to `_link_pockets`) -- assert it, don't just claim it.
        assert len(_infill_tail) == (len(passes) - len(_chainable))
    elif link_mult > 0 and len(passes) > 2:
        if _provenance and _prov is not None:
            passes, _prov = _link_pockets(passes, link_mult * fill_px,
                                           pocket_close=pocket_close, tags=_prov)
        else:
            passes = _link_pockets(passes, link_mult * fill_px, pocket_close=pocket_close)
    elif _provenance and _prov is not None:
        # link_pockets never ran (too few passes / disabled) -- normalize to
        # the same list-of-lists shape _link_pockets(tags=...) returns, so
        # the sidecar format is uniform regardless of which path fired.
        _prov = [[t] for t in _prov]

    if _hybrid_core_passes:
        # core_fill=hybrid: straight core rows are appended AFTER the ring pocket-link
        # and are never handed to _link_pockets (it never tests a connector against
        # the mask -- same reason zigzag infill is sliced out of it). Chaining inside
        # a core is already mask-checked in core_fill.core_rows.
        passes = passes + _hybrid_core_passes
        if _provenance and _prov is not None:
            _prov = _prov + [["hybrid_core"] for _ in _hybrid_core_passes]

    # Opt-in stitch_mm: join short OPEN strokes into longer continuous runs with
    # pen-down connectors that lie entirely inside the ink mask eroded by half a fill pitch
    # (ccplotter/stitch.py). Contour fill only (zigzag / tone are untouched; skipped is recorded).
    # Absent / 0: this block does not run and `_stitch_audit` stays {}.
    _stitch_audit = {}
    if float(opts.get("stitch_mm") or 0.0) > 0.0:
        if tone_on or _fill_mode == "zigzag":
            _stitch_audit = {"stitch": {"stitch_mm": float(opts.get("stitch_mm")),
                                        "skipped": "tone" if tone_on else "zigzag"}}
        else:
            from ccplotter.stitch import stitch_passes
            passes, _st_members, _st_rec = stitch_passes(
                passes, len(_hybrid_core_passes), mask, dt, opts, fill_px, mm_per_px,
                debug_prefix=debug_prefix)
            if _provenance and _prov is not None:
                _prov = [list(dict.fromkeys(t for i in mem for t in _prov[i])) for mem in _st_members]
            _stitch_audit = {"stitch": _st_rec}

    _write_paths_svg(out_path, W, H, _stroke_px_for(opts, W), polylines=passes)

    # Trace-boundary provenance: remember the FINAL mask's sha/dims/ink fraction in
    # memory (metadata only, never raises, not in _TRACE_AUDIT / info.json / the response).
    from ccplotter import trace_boundary as _tbd
    _tbd.record_mask(mask)

    # Opt-in connector_mask_check: persist the FINAL trace mask beside the SVG so Stage E
    # (composite -> stats_warnings._route_low_lift) can test its fuse connectors against the real
    # ink mask. Private key `_stage_e_mask` is set by composite only under that flag; absent: this
    # does not run and no file is written.
    if opts.get("_stage_e_mask"):
        from ccplotter.stage_e_mask import write_sidecar
        write_sidecar(out_path, mask, fill_px)

    if _provenance and _prov is not None:
        # Diagnostic sidecar only -- written AFTER the real SVG, never read by
        # any production code path, never alters `passes` or the SVG above.
        import json as _json
        _prov_out = opts.get("provenance_out")
        if not _prov_out:
            _prov_out = str(Path(out_path).with_name(Path(out_path).stem + "_provenance.json"))
        _census = {}
        for _origins in _prov:
            for _t in _origins:
                _census[_t] = _census.get(_t, 0) + 1
        _sidecar = {
            "source_svg": str(out_path),
            "n_chains": len(passes),
            "chains": [{"i": _i, "origins": _prov[_i], "n_points": len(passes[_i])}
                       for _i in range(len(passes))],
            "origin_census": _census,
            # census BEFORE _link_pockets merges rings (1:1 with each raw
            # polyline's own single producer, no merging/dedup yet) -- the
            # apples-to-apples comparison against _TRACE_AUDIT's own
            # zero_pass/fallback_strokes counts.
            "origin_census_pre_link": _prov_pre_link_census,
        }
        try:
            with open(_prov_out, "w") as _pf:
                _json.dump(_sidecar, _pf)
        except Exception as _e:
            print(f"    [trace] provenance_tag: sidecar write failed ({_e}) -- diagnostic only, not fatal")

    global _TRACE_AUDIT
    _lpm = len(passes) / max(0.001, sum(cv2.arcLength(np.array(p, np.float32), False)
                                        for p in passes) * mm_per_px / 1000.0)
    _TRACE_AUDIT = {"passes": n_pass_polys, "levels": n_levels,
                    "zero_pass_components": len(zero_labels),
                    "fallback_strokes": len(fallback_polys),
                    "fill_px": round(fill_px, 2), "mm_per_px": round(mm_per_px, 4),
                    # Emit fill_px_clamp_binds only when it is True. Emitting it
                    # unconditionally (even as a constant False) would change
                    # audit_json_sha256 for every fixture; audit keys are
                    # added only when their option fires (see the fill_routing
                    # NOTE just below).
                    **({"fill_px_clamp_binds": True} if _fill_px_clamp_binds else {}),
                    "min_mark_mm2": min_mark_mm2, "dropped_components": _dropped_comps,
                    "dropped_ink_pct": round(_dropped_ink_pct, 3),
                    "speck_polys_dropped": n_speck_dropped,
                    "speck_ink_pct": round(_speck_ink_pct, 3),
                    "lifts_before_link": lifts_before, "lifts": len(passes),
                    "lifts_per_meter": round(_lpm, 0), "total_polys": len(passes),
                    "tone": tone_mode, "gap_guarantee_ridge_px": _gg_marginal,
                    "gray_crush_fraction": round(_gray_crush_fraction, 4),
                    "texture_routing": str(opts.get("texture_routing", "off")),
                    # NOTE: fill_routing is deliberately NOT added to this audit
                    # dict -- adding it would change audit_json_sha256 for
                    # every fixture even though it is purely additive metadata.
                    # The default path's audit JSON is therefore unchanged.
                    **tone_audit, **_route_audit, **_gap_close_audit, **_core_audit, **_stitch_audit,
                    # Zigzag-only audit keys, emitted ONLY when fill_mode ==
                    # "zigzag" actually fired -- same rule as above: emitting
                    # them unconditionally (even a constant "contour"/None)
                    # would change audit_json_sha256 for every fixture.
                    **({"fill_mode": "zigzag",
                        "zigzag_offmask_ink_pct": round(_zigzag_offmask_pct, 3),
                        "zigzag_n_scanlines": _zigzag_audit.get("n_scanlines"),
                        "zigzag_n_runs": _zigzag_audit.get("n_runs"),
                        "zigzag_n_dropped_short": _zigzag_audit.get("n_dropped_short"),
                        "zigzag_dropped_len_px_total":
                            round(_zigzag_audit.get("dropped_len_px_total", 0.0), 2),
                        "zigzag_fill_angle_deg_image_frame": _zigzag_audit.get("angle_deg"),
                        # Perimeter + infill -- always emitted
                        # alongside the other zigzag-only keys (same gate),
                        # not separately flagged: perimeter_count is part of
                        # zigzag's own geometry, not an independent
                        # opt-in like fill_connect/fill_angle_mode below.
                        "zigzag_perimeter_count": _perimeter_count,
                        "zigzag_n_perimeter_passes": len(_perimeter_passes),
                        "zigzag_n_infill_passes": len(interior_passes),
                        # Shows the perimeter+fallback/infill separation held:
                        # "before"/"after" are lift counts of the CHAINABLE
                        # (perimeter+fallback) slice only, straddling its own
                        # `_link_pockets` call; infill's own lift count is
                        # `zigzag_n_infill_passes` above, which the link step
                        # never changes (infill is sliced out of that call).
                        "zigzag_n_chainable_lifts_before_link":
                            _zigzag_lifts_chainable_before_link,
                        "zigzag_n_chainable_lifts_after_link":
                            _zigzag_lifts_chainable_after_link,
                       } if ((not tone_on) and _fill_mode == "zigzag" and
                             _zigzag_offmask_pct is not None) else {}),
                    # Guarded-connector audit keys, emitted ONLY when
                    # fill_connect in ("guarded", "boustrophedon") actually
                    # fired -- same rule as the zigzag block above (with
                    # fill_connect="none" the audit JSON gains no keys).
                    **({"fill_connect": _fill_connect,
                        "zigzag_connect_max_gap_mm": _connect_max_gap_mm,
                        "zigzag_n_joins": _zigzag_audit.get("n_joins"),
                        "zigzag_n_join_refused_offmask":
                            _zigzag_audit.get("n_join_refused_offmask"),
                        "zigzag_n_join_refused_ceiling":
                            _zigzag_audit.get("n_join_refused_ceiling"),
                       } if ((not tone_on) and _fill_mode == "zigzag" and
                             _fill_connect in ("guarded", "boustrophedon") and
                             _zigzag_offmask_pct is not None) else {}),
                    # Boustrophedon-only audit keys, emitted ONLY when
                    # fill_connect == "boustrophedon" actually fired -- same
                    # rule as above.
                    **({"zigzag_n_row_joins": _zigzag_audit.get("n_row_joins"),
                        "zigzag_n_row_join_refused_offmask":
                            _zigzag_audit.get("n_row_join_refused_offmask"),
                        "zigzag_n_row_join_refused_ceiling":
                            _zigzag_audit.get("n_row_join_refused_ceiling"),
                        "zigzag_n_chains": _zigzag_audit.get("n_chains"),
                       } if ((not tone_on) and _fill_mode == "zigzag" and
                             _fill_connect == "boustrophedon" and
                             _zigzag_offmask_pct is not None) else {}),
                    # Proximity / split-merge continuation audit keys. Gated more
                    # tightly than the boustrophedon block above, on either
                    # switch actually being True (not merely on
                    # fill_connect=="boustrophedon"), so with BOTH switches at
                    # their OFF default `_TRACE_AUDIT` gains no keys beyond the
                    # boustrophedon ones.
                    **({"zigzag_connect_proximity": _zigzag_audit.get("connect_proximity"),
                        "zigzag_connect_split_merge": _zigzag_audit.get("connect_split_merge"),
                        "zigzag_n_split_continued": _zigzag_audit.get("n_split_continued"),
                        "zigzag_n_merge_continued": _zigzag_audit.get("n_merge_continued"),
                        "zigzag_n_proximity_joins": _zigzag_audit.get("n_proximity_joins"),
                        "zigzag_n_proximity_refused_offmask":
                            _zigzag_audit.get("n_proximity_refused_offmask"),
                        "zigzag_n_proximity_no_candidate":
                            _zigzag_audit.get("n_proximity_no_candidate"),
                        "zigzag_n_joins_total": _zigzag_audit.get("n_joins_total"),
                       } if ((not tone_on) and _fill_mode == "zigzag" and
                             _fill_connect == "boustrophedon" and
                             (_connect_proximity or _connect_split_merge) and
                             _zigzag_offmask_pct is not None) else {}),
                    # Chain-END linking audit keys. Gated as tightly as the
                    # proximity/split-merge block above: on `_connect_endlink`
                    # actually being True, not merely on
                    # fill_connect=="boustrophedon" -- with the switch at its
                    # OFF default, `_TRACE_AUDIT` gains no keys for it.
                    **({"zigzag_connect_endlink": _zigzag_audit.get("connect_endlink"),
                        "zigzag_connect_endlink_max_drow":
                            _zigzag_audit.get("connect_endlink_max_drow"),
                        "zigzag_n_chains_pre_endlink":
                            _zigzag_audit.get("n_chains_pre_endlink"),
                        "zigzag_n_endlink_joins": _zigzag_audit.get("n_endlink_joins"),
                        "zigzag_n_endlink_refused_offmask":
                            _zigzag_audit.get("n_endlink_refused_offmask"),
                        "zigzag_n_endlink_refused_ceiling":
                            _zigzag_audit.get("n_endlink_refused_ceiling"),
                        "zigzag_n_endlink_refused_drow":
                            _zigzag_audit.get("n_endlink_refused_drow"),
                        "zigzag_n_endlink_refused_cycle":
                            _zigzag_audit.get("n_endlink_refused_cycle"),
                        "zigzag_n_endlink_no_partner":
                            _zigzag_audit.get("n_endlink_no_partner"),
                       } if ((not tone_on) and _fill_mode == "zigzag" and
                             _fill_connect == "boustrophedon" and
                             _connect_endlink and
                             _zigzag_offmask_pct is not None) else {}),
                    # Diagnosis-only census keys. Same gate as the boustrophedon
                    # block above: emitted ONLY when fill_connect ==
                    # "boustrophedon" actually fired, so the default / none /
                    # guarded audit JSON (and its sha256) has no census keys.
                    **({"zigzag_census_end_split": _zigzag_audit.get("census_end_split"),
                        "zigzag_census_end_merge": _zigzag_audit.get("census_end_merge"),
                        "zigzag_census_end_no_overlap": _zigzag_audit.get("census_end_no_overlap"),
                        "zigzag_census_end_no_overlap_near_miss":
                            _zigzag_audit.get("census_end_no_overlap_near_miss"),
                        "zigzag_census_end_refused_offmask":
                            _zigzag_audit.get("census_end_refused_offmask"),
                        "zigzag_census_end_refused_ceiling":
                            _zigzag_audit.get("census_end_refused_ceiling"),
                        "zigzag_census_end_row_empty": _zigzag_audit.get("census_end_row_empty"),
                        "zigzag_census_end_eoi": _zigzag_audit.get("census_end_eoi"),
                        "zigzag_census_end_total": _zigzag_audit.get("census_end_total"),
                        "zigzag_census_start_after_refusal":
                            _zigzag_audit.get("census_start_after_refusal"),
                        "zigzag_census_start_fresh_no_col":
                            _zigzag_audit.get("census_start_fresh_no_col"),
                        "zigzag_census_start_fresh_ambiguous":
                            _zigzag_audit.get("census_start_fresh_ambiguous"),
                        "zigzag_census_start_fresh_after_split":
                            _zigzag_audit.get("census_start_fresh_after_split"),
                        "zigzag_census_start_total": _zigzag_audit.get("census_start_total"),
                        "zigzag_census_cols_carried_over_blank_row":
                            _zigzag_audit.get("census_cols_carried_over_blank_row"),
                        "zigzag_n_mask_components_4conn":
                            _zigzag_audit.get("n_mask_components_4conn"),
                        "zigzag_n_mask_components_traversed":
                            _zigzag_audit.get("n_mask_components_traversed"),
                        "zigzag_n_dsu_components_ignoring_predicate":
                            _zigzag_audit.get("n_dsu_components_ignoring_predicate"),
                        "zigzag_chain_n2pt_count": _zigzag_audit.get("chain_n2pt_count"),
                        "zigzag_chain_len_px_min": _zigzag_audit.get("chain_len_px_min"),
                        "zigzag_chain_len_px_median": _zigzag_audit.get("chain_len_px_median"),
                        "zigzag_chain_len_px_mean": _zigzag_audit.get("chain_len_px_mean"),
                        "zigzag_chain_len_px_max": _zigzag_audit.get("chain_len_px_max"),
                       } if ((not tone_on) and _fill_mode == "zigzag" and
                             _fill_connect == "boustrophedon" and
                             _zigzag_offmask_pct is not None) else {}),
                    # Per-region angle selection audit keys, emitted ONLY when
                    # fill_angle_mode == "per_region" actually fired -- same
                    # rule as the zigzag block above (flag-off audit JSON gains
                    # no keys).
                    **({"fill_angle_mode": "per_region",
                        "zigzag_n_regions": _zigzag_audit.get("n_regions"),
                        "zigzag_n_regions_evaluated": _zigzag_audit.get("n_regions_evaluated"),
                        "zigzag_n_regions_fallback_tiny": _zigzag_audit.get("n_regions_fallback_tiny"),
                        "zigzag_n_regions_fallback_cap": _zigzag_audit.get("n_regions_fallback_cap"),
                        "zigzag_fallback_ink_pct": _zigzag_audit.get("fallback_ink_pct"),
                        "zigzag_per_region_eval_seconds": _zigzag_audit.get("eval_seconds"),
                        "zigzag_coverage_constraint": _zigzag_audit.get("coverage_constraint"),
                        "zigzag_regions": _zigzag_audit.get("regions"),
                       } if ((not tone_on) and _fill_mode == "zigzag" and
                             _fill_angle_mode == "per_region" and
                             _zigzag_offmask_pct is not None) else {})}
    if _dropped_ink_pct > 0.5:
        print(f"    [trace] WARNING: dot-cull dropped {_dropped_ink_pct:.1f}% of ink "
              f"(>0.5% — that 'noise' may be texture; lower min_mark_mm2).")
    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_trace_mask.png", (mask.astype(np.uint8) * 255))
    print(f"    [perf] _trace_svg: {time.time()-t0:.2f}s levels={n_levels} passes={n_pass_polys} "
          f"dotcull={_dropped_comps}comps/{_dropped_ink_pct:.2f}%ink +{n_speck_dropped}specks "
          f"zero_pass={len(zero_labels)}(+{len(fallback_polys)}skel) "
          f"lifts {lifts_before}->{len(passes)} fill_px={fill_px:.2f}")
    return out_path


def _orientation_map(gray):
    """Local dominant orientation via structure tensor → BGR (hue=angle,
    saturation=coherence). Debug/analysis only."""
    import cv2, numpy as np
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), 4)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), 4)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), 4)
    ang = 0.5 * np.arctan2(2 * Jxy, (Jxx - Jyy))
    coher = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-6)
    hue = (((ang + np.pi / 2) / np.pi) * 179).astype(np.uint8)
    sat = np.clip(coher * 255, 0, 255).astype(np.uint8)
    val = np.full_like(hue, 220)
    return cv2.cvtColor(cv2.merge([hue, sat, val]), cv2.COLOR_HSV2BGR)


def _gap_widths_mm(mask, mm_per_px, dens_sigma_px):
    """White-gap width distribution INSIDE ink-dense regions (gap forensics).
    A hatch gap is a run of background bounded by ink. Distance-transform the
    BACKGROUND (dist to nearest ink), skeletonize the background to its ridge
    (one pixel per gap centerline), and read 2*dt on that ridge = full gap width.
    Restrict to ink-dense regions (blurred ink fraction > 0.5) so open paper isn't
    counted as a 'gap'. Returns (widths_mm array, ridge_len_px). Pure measurement."""
    import cv2, numpy as np
    from skimage.morphology import skeletonize
    ink = (mask > 0).astype(np.uint8)
    bg = 1 - ink
    dt = cv2.distanceTransform(bg, cv2.DIST_L2, 5)
    dens = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), max(2.0, dens_sigma_px))
    ridge = skeletonize(bg > 0) & (dens > 0.5)
    w_px = 2.0 * dt[ridge]
    w_mm = w_px * mm_per_px
    # gaps only: exclude the tail of wide open areas (> 3mm isn't a hatch gap)
    w_mm = w_mm[(w_mm > 0) & (w_mm < 3.0)]
    return w_mm, int(ridge.sum())


def _gap_audit(src_path, resolution=2600, paper_w_mm=180.0, margin_mm=10.0,
               deposit_mm=0.15, paper_h_mm=None):
    """GAP FORENSICS (measurement only; changes no tracing behaviour). Tracks
    white-hatch-gap survival through the three stages where gaps die:
      (a) source NATIVE res, plain Otsu (best case — what the file actually holds)
      (b) resampled to trace res, then _binarize_mask with the default pre-blur
          (isolates the RESAMPLE + pre-threshold BLUR loss)
      (c) native-res thresholding (_tracev3_mask), then the binary mask resized
          to trace res (the native_res path)
    Reports a gap-width histogram (mm at plot scale) + % of gap ridge >= deposit_mm
    surviving each stage, and a side-by-side crop of the densest region's mask at
    each stage. The stage with the big survival drop is the killer.

    mm/px model (same as _preflight): the layout's own per-item fit is
    composite's `fit = min(cellW/nw, cellH/nh) * 0.9`, so mm/px at any raster
    stage is `min(usable_w/w, usable_h/h) * 0.9` on that stage's own pixel
    dimensions. A width-only model without the 0.9 factor would undercount
    mm/px and make gap survival look better than it is. paper_h_mm defaults to
    paper_w_mm (square page) and is the last positional slot, so a width-only
    call still behaves sensibly and a caller passing deposit_mm positionally
    can never land in paper_h_mm's slot."""
    import cv2, numpy as np, time
    t0 = time.time()
    raw = cv2.imread(str(Path(src_path)), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        from PIL import Image
        raw = cv2.cvtColor(np.array(Image.open(src_path).convert("RGB")), cv2.COLOR_RGB2GRAY)
    nh, nw = raw.shape[:2]
    paper_h_mm = paper_w_mm if paper_h_mm is None else paper_h_mm
    usable_w_mm = float(paper_w_mm) - 2 * float(margin_mm)
    usable_h_mm = float(paper_h_mm) - 2 * float(margin_mm)
    drawable_mm = usable_w_mm      # usable WIDTH, mm
    trace_w = int(resolution)
    trace_h = int(nh * trace_w / nw)
    mm_px_native = min(usable_w_mm / nw, usable_h_mm / nh) * 0.9
    mm_px_trace = min(usable_w_mm / trace_w, usable_h_mm / trace_h) * 0.9

    def _otsu(g):
        _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return m > 0

    # (a) native — the ceiling: what the file actually holds
    mask_a = _otsu(raw)
    native_ink = float(mask_a.mean()) * 100
    # (b) resampled to trace res, then binarized with the default (3,3) pre-blur
    if trace_w < nw:
        g_b = cv2.resize(raw, (trace_w, int(nh * trace_w / nw)), interpolation=cv2.INTER_AREA)
    else:
        g_b = cv2.resize(raw, (trace_w, int(nh * trace_w / nw)), interpolation=cv2.INTER_CUBIC)
    mask_b = _binarize_mask(g_b, {"threshold_mode": "otsu", "pre_blur": 3})
    # (c) native_res path — threshold at native, resize the binary mask
    mask_c = _tracev3_mask(raw, {"threshold_mode": "otsu"}, trace_w, int(nh * trace_w / nw))

    sig_mm = 0.6                              # "dense" measured over ~0.6mm neighborhoods
    stages = []
    for name, mask, mmpx in (("a_native", mask_a, mm_px_native),
                             ("b_old_resample_blur", mask_b, mm_px_trace),
                             ("c_new_nativeres", mask_c, mm_px_trace)):
        w_mm, ridge = _gap_widths_mm(mask, mmpx, sig_mm / mmpx)
        surviving_mm = float((w_mm >= deposit_mm).sum()) * mmpx
        total_mm = float(w_mm.size) * mmpx
        ink_pct = round(float(mask.mean()) * 100, 1)
        stages.append({
            "stage": name, "mm_per_px": round(mmpx, 4), "ink_pct": ink_pct,
            # ROBUST faithfulness metric: ink% delta vs native. 0 = gaps preserved;
            # POSITIVE = gaps filled with ink (the failure mode). The ridge metric
            # below is scale-sensitive; this one is not.
            "ink_vs_native_pts": round(ink_pct - native_ink, 1),
            "gap_ridge_mm": round(total_mm, 1),
            "gaps_ge_deposit_mm": round(surviving_mm, 1),
            "pct_ge_deposit": round(float((w_mm >= deposit_mm).mean()) * 100, 1) if w_mm.size else 0.0,
            "median_gap_mm": round(float(np.median(w_mm)), 3) if w_mm.size else 0.0,
            "_wmm": w_mm})
    base_surv = max(1e-6, stages[0]["gaps_ge_deposit_mm"])
    for s in stages:
        s["survival_vs_native_pct"] = round(s["gaps_ge_deposit_mm"] / base_surv * 100, 1)

    # ---- histogram image (mm bins, 3 stages overlaid) ----
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 3.0]
    Hh, Ww = 360, 760
    hist = np.full((Hh, Ww, 3), 22, np.uint8)
    cols = {"a_native": (120, 120, 120), "b_old_resample_blur": (90, 90, 240), "c_new_nativeres": (110, 220, 110)}
    nb = len(bins) - 1
    bw = Ww // (nb + 1)
    counts = {s["stage"]: np.histogram(s["_wmm"], bins=bins)[0].astype(float) for s in stages}
    mx = max(1.0, max(c.max() for c in counts.values()))
    for bi in range(nb):
        for si, s in enumerate(stages):
            c = counts[s["stage"]][bi] / mx
            x0 = bi * bw + si * (bw // 3) + 8
            y1 = Hh - 24
            y0 = int(y1 - c * (Hh - 50))
            cv2.rectangle(hist, (x0, y0), (x0 + bw // 3 - 2, y1), cols[s["stage"]], -1)
        cv2.putText(hist, f"{bins[bi+1]:.2f}", (bi * bw + 6, Hh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)
    # deposit line
    dep_bin = next((i for i, b in enumerate(bins) if b >= deposit_mm), nb)
    xdep = dep_bin * bw + 4
    cv2.line(hist, (xdep, 8), (xdep, Hh - 24), (80, 220, 120), 1)
    cv2.putText(hist, f"{deposit_mm}mm deposit", (xdep + 4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 220, 120), 1)
    for si, (nm, col) in enumerate([("native", cols["a_native"]), ("old (resamp+blur)", cols["b_old_resample_blur"]), ("new (nativeres)", cols["c_new_nativeres"])]):
        cv2.rectangle(hist, (Ww - 190, 30 + si * 20), (Ww - 178, 42 + si * 20), col, -1)
        cv2.putText(hist, nm, (Ww - 172, 41 + si * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    # ---- most-HATCHED-region crop (native coords -> per-stage). Target the region
    # with the most gap RIDGES (dense hatch = shadow/gravel), not the most ink
    # (that's just solid slabs like the logo, where gaps aren't the story). ----
    from skimage.morphology import skeletonize as _sk
    win_mm = 14.0
    _bg_a = 1 - mask_a.astype(np.uint8)
    _dens_a = cv2.GaussianBlur(mask_a.astype(np.float32), (0, 0), max(4.0, 0.6 / mm_px_native))
    _ridge_a = (_sk(_bg_a > 0) & (_dens_a > 0.5)).astype(np.float32)
    _hatch = cv2.GaussianBlur(_ridge_a, (0, 0), max(6.0, win_mm / mm_px_native / 3))
    cy, cx = np.unravel_index(int(np.argmax(_hatch)), _hatch.shape)
    def _crop(mask, mmpx, ccx, ccy):
        half = int(win_mm / mmpx / 2)
        h, w = mask.shape[:2]
        x0, y0 = max(0, ccx - half), max(0, ccy - half)
        x1, y1 = min(w, ccx + half), min(h, ccy + half)
        c = (mask[y0:y1, x0:x1].astype(np.uint8)) * 255
        return cv2.resize(c, (380, 380), interpolation=cv2.INTER_NEAREST)
    # map the native crop centre into each stage's pixel space
    crops = []
    for name, mask, mmpx, sx, sy in (
            ("NATIVE (ceiling)", mask_a, mm_px_native, cx, cy),
            ("OLD resamp+blur", mask_b, mm_px_trace, int(cx * trace_w / nw), int(cy * trace_w / nw)),
            ("NEW nativeres", mask_c, mm_px_trace, int(cx * trace_w / nw), int(cy * trace_w / nw))):
        cc = cv2.cvtColor(_crop(mask, mmpx, sx, sy), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(cc, (0, 0), (380, 22), (28, 28, 28), -1)
        cv2.putText(cc, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 230, 90), 1)
        crops.append(cc)
    crop_row = np.hstack(crops)

    def sv(name, im):
        cv2.imwrite(str(WORK_DIR / name), im); return name
    files = {"histogram": sv("gapaudit_hist.png", hist),
             "crops": sv("gapaudit_crops.png", crop_row)}
    for s in stages:
        s.pop("_wmm", None)
    print(f"    [gapaudit] native {nw}x{nh} ({mm_px_native:.4f}mm/px)  "
          f"ink vs native (0=faithful): old {stages[1]['ink_vs_native_pts']:+}pts  "
          f"new {stages[2]['ink_vs_native_pts']:+}pts  {time.time()-t0:.1f}s")
    return {"stages": stages,
            "native_px": [nw, nh], "trace_px": trace_w,
            "mm_per_px_native": round(mm_px_native, 4), "mm_per_px_trace": round(mm_px_trace, 4),
            "deposit_mm": deposit_mm, "drawable_mm": round(drawable_mm, 1),
            "drawable_h_mm": round(usable_h_mm, 1),
            "files": files,
            "paths": {k: str((WORK_DIR / v).resolve()) for k, v in files.items()}}


def _preflight(src_path, paper_w_mm=180.0, margin_mm=10.0, deposit_mm=0.15, paper_h_mm=None):
    """Preflight — SOURCE-SUITABILITY check (runs on load, before plotting).
    The pipeline transmits the drawing; when the drawing is finer than the pen at
    the chosen size, the honest job is to SAY SO with numbers. Measures, at native
    res and current paper/pen: finest surviving stroke, finest surviving gap, and
    the % of the drawing's texture below the printable pitch. The visible-gap ceiling
    is ~2x deposit (a gap only ~1x deposit reads as a hairline against dense ink
    on paper), so 'below pitch' is measured against 2x deposit, with the
    bare 1x line reported too. Cheap, deterministic, no plotting.

    mm/px model: mirrors the layout's own per-item fit,
    composite's `fit = min(cellW/nw, cellH/nh) * 0.9` — the drawing must fit the
    usable box in BOTH dimensions, and the layout always shrinks it 10% below the tight
    fit. paper_h_mm defaults to paper_w_mm (square page) so an old-style call with only
    paper_w_mm still behaves sensibly; pass it explicitly for a non-square sheet.
    paper_h_mm is keyword-only in practice (last positional slot) so a caller passing
    margin_mm/deposit_mm positionally can never silently land in paper_h_mm's slot.

    Known-unmodelled factors: this mirrors composite.py's SINGLE-ITEM fit only. It does not account for multi-item grids
    (composite.py `_grid`/`cellW`/`cellH` divide the cell for n>1 items), `true_size`
    (fit forced to 1.0), `fill_page` (cover-crop, bypasses `fit` entirely), or
    `offset_x`/`offset_y` placement. It also uses the RAW native-pixel raster as `nw,nh`
    where composite.py's own fit uses the traced SVG's ink bbox in mm, which was about
    0.1% smaller than the native canvas on real renders; this preflight is therefore
    very slightly conservative (over-warns), not optimistic, on that residual."""
    import cv2, numpy as np
    from skimage.morphology import skeletonize
    raw = cv2.imread(str(Path(src_path)), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        from PIL import Image
        raw = cv2.cvtColor(np.array(Image.open(src_path).convert("RGB")), cv2.COLOR_RGB2GRAY)
    nh, nw = raw.shape[:2]
    paper_h_mm = paper_w_mm if paper_h_mm is None else paper_h_mm
    usable_w_mm = float(paper_w_mm) - 2 * float(margin_mm)
    usable_h_mm = float(paper_h_mm) - 2 * float(margin_mm)
    drawable_mm = usable_w_mm      # usable WIDTH, mm (the `drawable_mm` key in the result)
    mmpx = min(usable_w_mm / nw, usable_h_mm / nh) * 0.9
    dep_px = deposit_mm / mmpx
    visible_px = 2.0 * dep_px                      # a gap needs ~2x deposit to READ
    _, m = cv2.threshold(raw, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = m > 0
    # stroke widths (2*ink DT on the ink skeleton = full stroke width)
    ink_dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    stroke_px = 2.0 * ink_dt[skeletonize(ink)]
    stroke_px = stroke_px[stroke_px > 0]
    # gap widths inside ink-dense regions only (hatch/shadow texture, not open paper)
    bg_dt = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5)
    dens = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), max(2.0, 0.6 / mmpx)) > 0.5
    texture_pct = round(float(dens.mean()) * 100, 1)      # % of page that is dense hatch
    gap_px = 2.0 * bg_dt[skeletonize(~ink) & dens]
    gap_px = gap_px[(gap_px > 0) & (gap_px < 3.0 / mmpx)]

    def pct_below(arr, thr):
        return round(float((arr < thr).mean()) * 100, 0) if arr.size else 0.0
    finest_stroke_mm = round(float(np.percentile(stroke_px, 5)) * mmpx, 3) if stroke_px.size else 0.0
    finest_gap_mm = round(float(np.percentile(gap_px, 5)) * mmpx, 3) if gap_px.size else 0.0
    gaps_sub_visible = pct_below(gap_px, visible_px)      # of hatch gaps, % that won't READ
    gaps_sub_bare = pct_below(gap_px, dep_px)             # of hatch gaps, % that won't PRINT at all
    strokes_sub_bare = pct_below(stroke_px, dep_px)       # strokes thinner than the pen
    median_gap_mm = round(float(np.median(gap_px)) * mmpx, 3) if gap_px.size else 0.0
    # THE headline number: fraction of the WHOLE drawing that is sub-pitch texture.
    # (weight the sub-pitch gap fraction by how much of the page is actually hatch —
    # otherwise a sparse clean drawing with a few edge gaps cries wolf.)
    problem_pct = round(texture_pct * gaps_sub_visible / 100.0, 1)
    # Paper size that would lift the median gap to the 2x-deposit visible ceiling.
    # Scaling paper_w_mm alone would be inert on a height-limited source: growing
    # width does not change mmpx when height is the binding dimension.
    # `scale` is the factor mmpx must grow by. Applying `scale` to BOTH paper_w_mm
    # and paper_h_mm -- a proportionally scaled sheet, not a wider one -- is PROVEN
    # sufficient for mmpx on EITHER branch: usable_w' = scale*paper_w - 2*margin =
    # scale*usable_w + 2*margin*(scale-1) > scale*usable_w for scale>1, margin>0
    # (same for usable_h), so min(usable_w'/nw, usable_h'/nh) > scale*min(usable_w/nw,
    # usable_h/nh) always -- i.e. new_mmpx > scale*mmpx regardless of which
    # dimension binds. That is where the proof stops:
    # median_gap_mm is NOT linear in mmpx -- the gap POPULATION itself shifts with
    # mmpx, via dens's sigma max(2.0, 0.6/mmpx) and the gap_px < 3.0/mmpx cutoff,
    # the same mechanism that moves texture_pct by -0.1 to -3.9pt. The step from
    # mmpx to median_gap_mm is EMPIRICAL (it held on the test sources, taking
    # gaps_below_visible from 79-89% to 0.0%); it is not guaranteed in general.
    paper_for_visible = suggest_paper_w_mm = suggest_paper_h_mm = None
    if median_gap_mm > 0 and median_gap_mm < 2 * deposit_mm:
        import math
        scale = (2 * deposit_mm) / median_gap_mm
        # Round UP to the next 10mm (not to nearest): round-to-nearest can round
        # BOTH dimensions down at once (e.g. an exact target of 281.4x192.7mm
        # becoming 280x190 lands the median gap at 0.299mm against a 0.300mm
        # target). A 0.3% miss is not a 0.3% degradation: the ceiling is a
        # threshold on a tight hatch pitch, so the share of sub-visible gaps
        # can stay above 80% at the rounded-down size and drop to 0% at the
        # rounded-up one. Rounding up keeps the mmpx bound intact.
        suggest_paper_w_mm = int(math.ceil(float(paper_w_mm) * scale / 10.0) * 10)
        suggest_paper_h_mm = int(math.ceil(float(paper_h_mm) * scale / 10.0) * 10)
        # Single-scalar key (for the options-list consumer):
        # the WIDTH of the scaled sheet above, paired with suggest_paper_h_mm --
        # meaningful on both branches because the whole sheet grows together, not
        # because width happens to be binding.
        paper_for_visible = suggest_paper_w_mm

    # verdict tiers by the share of the whole drawing that is sub-pitch texture
    if problem_pct < 5:
        tier, msg = "clean", (f"Prints cleanly at {int(paper_w_mm)} mm / {deposit_mm} mm pen — "
                              f"detail is above printable pitch.")
    elif problem_pct < 18:
        tier, msg = "caution", (f"~{problem_pct}% of this drawing is hatch/texture below printable "
                                f"pitch at {int(paper_w_mm)} mm / {deposit_mm} mm pen — those zones "
                                f"may read partly solid.")
    else:
        tier, msg = "over", (f"~{problem_pct}% of this drawing is hatch/texture below printable pitch "
                             f"at {int(paper_w_mm)} mm / {deposit_mm} mm pen — hatched/shadow zones "
                             f"will plot as solid fill.")
    # Recommend among the three responses (display logic only, no pipeline
    # change): plot bigger if the math says so, tone=Balanced as the immediate
    # stopgap (about +50% pen-lifts / plot time), regenerate the source as the
    # durable fix.
    opts_txt = []
    if tier != "clean":
        if paper_for_visible and paper_for_visible > paper_w_mm:
            if suggest_paper_w_mm == suggest_paper_h_mm:
                opts_txt.append(f"plot at ~{paper_for_visible} mm to lift it above pitch")
            else:
                opts_txt.append(f"plot at ~{suggest_paper_w_mm}x{suggest_paper_h_mm} mm "
                                f"to lift it above pitch")
        opts_txt.append("try tone=Balanced now (Trace V3 tonal fill) — lighter shadows on this "
                        "exact source, no regeneration needed, costs ~50% more plot time")
        opts_txt.append("regenerate the source at 2-4x resolution / coarser hatching (durable fix)")
        opts_txt.append("or accept solid fill in those zones")
    return {"native_px": [nw, nh], "mm_per_px": round(mmpx, 4),
            "deposit_mm": deposit_mm, "drawable_mm": round(drawable_mm, 1),
            "drawable_h_mm": round(usable_h_mm, 1),
            "finest_stroke_mm": finest_stroke_mm, "finest_gap_mm": finest_gap_mm,
            "median_gap_mm": median_gap_mm, "texture_pct": texture_pct,
            "problem_pct": problem_pct,
            "gaps_below_visible_pct": gaps_sub_visible, "gaps_below_bare_pct": gaps_sub_bare,
            "strokes_below_pen_pct": strokes_sub_bare,
            "suggest_paper_mm": paper_for_visible,
            "suggest_paper_w_mm": suggest_paper_w_mm, "suggest_paper_h_mm": suggest_paper_h_mm,
            "tier": tier, "verdict": msg, "options": opts_txt}


def _coverage_debug(src_path, base_mode="sketchv2", resolution=2600,
                    paper_w_mm=180.0, margin_mm=10.0, pen_mm=0.5, tone="off",
                    paper_h_mm=None):
    """SOURCE-FAITHFUL COVERAGE — DEBUG RENDERER (measurement only, NO stroke
    generation). Compares SOURCE ink to what the current trace actually plots at a
    real pen width. Writes: source / ink mask / simulated plot / density-error map /
    binary underfill-overfill overlay, + stats. base_mode: 'lineart' | 'sketchv2' |
    'tracev3'. tone (tracev3 only): 'off'|'faithful'|'balanced'|'dense'.

    The headline metric is DENSITY error, not binary coverage. Binary
    under/overfill can't score TONE: a mid-gray region should get PARTIAL ink, but
    a binary compare flags that as underfilled. So we compare blurred plotted-ink
    density against the blurred SOURCE inverted-grayscale (tonal demand), sigma ~2
    pen widths. mean|density error| is the scalar; for tracev3 tone we also render
    the OFF (solid V3) baseline and report the delta (accept = tone below baseline).
    A separate BINARY underfill is kept for the DARK level (dark ink must stay
    solid — accept <3%).

    mm/px model (same as _preflight): `mm_per_px` here only sizes the SIMULATED
    pen stroke width (pen_px) drawn into this debug renderer's own raster — it
    reaches no production geometry or G-code. It follows the layout's own fit,
    composite's `min(usable_w/nw, usable_h/nh) * 0.9`, on this stage's own
    working-resolution pixel dimensions. paper_h_mm defaults to paper_w_mm
    (square page) and is the last positional slot."""
    import cv2, numpy as np, re, time
    img = _load_gray(Path(src_path), int(resolution))
    H, W = img.shape[:2]
    paper_h_mm = paper_w_mm if paper_h_mm is None else paper_h_mm
    usable_w_mm = float(paper_w_mm) - 2 * float(margin_mm)
    usable_h_mm = float(paper_h_mm) - 2 * float(margin_mm)
    mm_per_px = max(1e-6, min(usable_w_mm / W, usable_h_mm / H) * 0.9)
    # Simulate at the pen's REAL deposited line width (measured fill_mm), not the
    # nominal tip size — otherwise coverage is optimistic (the EnerGel lays ~0.15mm).
    deposit_mm = float(active_pen_profile().get("fill_mm", pen_mm))
    pen_px = max(1.0, deposit_mm / mm_per_px)
    sigma = max(4.0, pen_px * 2.0)

    _, sm = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    src_ink = (sm > 0).astype(np.float32)
    src_d = cv2.GaussianBlur(src_ink, (0, 0), sigma)                 # binary ink density
    # TONAL demand: inverted grayscale (dark source = more ink wanted),
    # but only where there's real ink (gate by the ink mask so paper texture/JPEG
    # noise near white doesn't invent demand). 0..1, same blur as the plot sim.
    tonal = (255.0 - img.astype(np.float32)) / 255.0 * src_ink
    tgt_d = cv2.GaussianBlur(tonal, (0, 0), sigma)

    base_opts = {"threshold_mode": "otsu", "lineart_min_area": 4, "hole_fill": -1,
                 "lineart_gap_close": 0, "hollow_fills": 1, "prune_len": 3,
                 "preserve_text": "balanced", "small_detail_fidelity": "on",
                 "sketch_gain": 0.12, "sketch_max_passes": 2, "sketch_spacing_px": 0,
                 "sketch_min_area": 6, "outline_simplify": 0.08}
    tk = max(1, int(round(pen_px)))

    def _run(tone_val):
        """Render the base at a given tone and simulate the plotted ink. Returns
        (out_ink float mask, polys, trace_s, audit dict)."""
        opts = dict(base_opts)
        tmp = WORK_DIR / f"_covtrace_{base_mode}_{tone_val}.svg"
        t0 = time.time()
        if base_mode == "tracev3":
            opts = {**opts, "fill_mm": float(active_pen_profile().get("fill_mm", 0.15)),
                    "paper_w_mm": float(paper_w_mm), "margin_mm": float(margin_mm),
                    "tone": tone_val}
            _ng = _load_native_gray(Path(src_path)) if opts.get("native_res") else None
            _trace_svg(img.copy(), tmp, opts, native_gray=_ng)
        elif base_mode == "sketchv2":
            _sketch_svg(img.copy(), tmp, opts)
        else:
            _lineart_svg(img.copy(), tmp, opts)
        ts = time.time() - t0
        aud = dict(_TRACE_AUDIT) if base_mode == "tracev3" else {}
        pl = []
        for ps in re.findall(r'points="([^"]+)"', tmp.read_text()):
            co = [tuple(map(float, t.split(','))) for t in ps.split() if ',' in t]
            if len(co) >= 2:
                pl.append(co)
        if tmp.exists():
            tmp.unlink()
        o = np.zeros((H, W), np.uint8)
        for p in pl:
            pts = np.array(p, np.int32)
            if len(pts) >= 2:
                closed = len(p) >= 3 and abs(p[0][0] - p[-1][0]) < 0.6 and abs(p[0][1] - p[-1][1]) < 0.6
                cv2.polylines(o, [pts], closed, 255, tk)
        return (o > 0).astype(np.float32), pl, ts, aud

    tone = str(tone).lower()
    tone_on = base_mode == "tracev3" and tone not in ("", "off", "none")
    out_ink, polys, trace_s, audit = _run(tone)
    out_d = cv2.GaussianBlur(out_ink, (0, 0), sigma)

    # DENSITY error (signed): + = source wants more ink here, - = over-inked.
    derr = tgt_d - out_d
    mean_derr = float(np.abs(derr).mean())
    # DARK-level binary underfill: where the source is near-solid (tonal demand high),
    # plotted ink must be solid too. Accept < 3%.
    dark_zone = tgt_d > 0.80
    dark_under = float((dark_zone & (out_d < tgt_d - 0.15)).mean()) * 100.0
    # binary under/overfill (for the dark-ink view and the stats)
    diff = src_d - out_d
    under = np.clip(diff, 0, None)
    over = np.clip(-diff, 0, None)

    length_mm = sum(cv2.arcLength(np.array(p, np.float32), False) for p in polys) * mm_per_px
    n_lifts = len(polys)
    ink_cov = float((out_ink * src_ink).sum()) / max(1.0, float(src_ink.sum())) * 100.0
    est_min = length_mm / 3000.0 + n_lifts * 0.25 / 60.0

    # OFF baseline (V3 solid) for the acceptance delta — only when a tone is active.
    baseline = None
    if tone_on:
        b_ink, b_polys, _, _ = _run("off")
        b_d = cv2.GaussianBlur(b_ink, (0, 0), sigma)
        b_mean_derr = float(np.abs(tgt_d - b_d).mean())
        b_len = sum(cv2.arcLength(np.array(p, np.float32), False) for p in b_polys) * mm_per_px
        b_est = b_len / 3000.0 + len(b_polys) * 0.25 / 60.0
        baseline = {"mean_density_err": round(b_mean_derr, 4), "lifts": len(b_polys),
                    "est_min": round(b_est, 1)}

    dw = 760
    def dz(im):
        c = cv2.resize(im, (dw, int(dw * H / W)))
        return c if c.ndim == 3 else cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
    # binary overlay (cell 4)
    overlay = np.zeros((H, W, 3), np.uint8)
    overlay[..., 2] = np.clip(under * 400, 0, 255).astype(np.uint8)   # red = underfill
    overlay[..., 0] = np.clip(over * 400, 0, 255).astype(np.uint8)    # blue = overfill
    blend = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.35, overlay, 1.0, 0)
    # density-error map (cell 5): red = under (too light), blue = over (too dark)
    dmap = np.zeros((H, W, 3), np.uint8)
    dmap[..., 2] = np.clip(np.clip(derr, 0, None) * 510, 0, 255).astype(np.uint8)
    dmap[..., 0] = np.clip(np.clip(-derr, 0, None) * 510, 0, 255).astype(np.uint8)
    dblend = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.35, dmap, 1.0, 0)

    def sv(name, im):
        cv2.imwrite(str(WORK_DIR / name), im); return name
    b = f"coverage_{base_mode}"
    _src_bgr, _ink_bgr = dz(img), dz((src_ink * 255).astype(np.uint8))
    _sim_bgr, _ovl_bgr, _den_bgr = dz((out_ink * 255).astype(np.uint8)), dz(blend), dz(dblend)
    files = {"source": sv(f"{b}_source.png", _src_bgr),
             "ink": sv(f"{b}_ink.png", _ink_bgr),
             "sim": sv(f"{b}_sim.png", _sim_bgr),
             "overlay": sv(f"{b}_overlay.png", _ovl_bgr),
             "density": sv(f"{b}_density.png", _den_bgr)}
    def _cell(im, title):
        im = im.copy()
        cv2.rectangle(im, (0, 0), (im.shape[1], 28), (28, 28, 28), -1)
        cv2.putText(im, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 230, 90), 2)
        return im
    _tone_tag = f" [{tone}]" if tone_on else ""
    _cells = [_cell(_src_bgr, "1. SOURCE"), _cell(_ink_bgr, "2. SOURCE INK MASK"),
              _cell(_sim_bgr, f"3. SIMULATED PLOT @ {deposit_mm}mm{_tone_tag}"),
              _cell(_ovl_bgr, "4. BINARY UNDER (red) / OVER (blue)"),
              _cell(_den_bgr, "5. DENSITY ERROR: too-light (red) / too-dark (blue)")]
    _ch, _cw = _cells[0].shape[:2]
    _leg = np.full((_ch, _cw, 3), 22, np.uint8)
    _lvl = audit.get("tone_levels") or {}
    _lvl_lines = [f" {k}: {v['ink_pct']}% ink, {v['passes']} passes"
                  for k, v in _lvl.items()] if _lvl else []
    _delta = ""
    if baseline:
        _d = mean_derr - baseline["mean_density_err"]
        _delta = f" (V3 off {baseline['mean_density_err']}, {'-' if _d<0 else '+'}{abs(_d):.4f})"
    _lines = [f"BASE: {base_mode}{_tone_tag}", "",
              f"MEAN DENSITY ERR: {mean_derr:.4f}{_delta}",
              f"  {'PASS' if (not baseline or mean_derr <= baseline['mean_density_err']) else 'WORSE than V3'} vs V3 baseline",
              f"DARK-LEVEL UNDERFILL: {dark_under:.1f}%  {'PASS' if dark_under < 3 else 'FAIL (>3%)'}",
              "", "LEGEND (cell 5 = density)",
              " red  = plotted too LIGHT (wants ink)",
              " blue = plotted too DARK (over-inked)",
              " neutral = tone matched", ""] + _lvl_lines + ["",
              f"source ink {round(float(src_ink.mean())*100,1)}%  inked {round(ink_cov,1)}%",
              f"binary under {round(float((under>0.15).mean())*100,1)}%  over {round(float((over>0.15).mean())*100,1)}%",
              f"pen {pen_mm}mm nom / {deposit_mm}mm real = {pen_px:.1f}px",
              f"polylines {len(polys)}  lifts {n_lifts}  ~{est_min:.0f}min  res {int(resolution)}"]
    for _i, _t in enumerate(_lines):
        cv2.putText(_leg, _t, (12, 30 + _i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 225, 225), 1)
    _cells.append(_leg)
    sheet = np.vstack([np.hstack(_cells[0:3]), np.hstack(_cells[3:6])])
    files["sheet"] = sv(f"{b}_sheet.png", sheet)
    stats = {"base": base_mode, "tone": tone if tone_on else "off",
             "source_ink_pct": round(float(src_ink.mean()) * 100, 1),
             "output_ink_pct": round(float(out_ink.mean()) * 100, 1),
             "ink_covered_pct": round(ink_cov, 1),
             "mean_density_err": round(mean_derr, 4),
             "dark_underfill_pct": round(dark_under, 1),
             "dark_underfill_ok": bool(dark_under < 3),
             "underfill_pct": round(float((under > 0.15).mean()) * 100, 1),
             "overfill_pct": round(float((over > 0.15).mean()) * 100, 1),
             "pen_px": round(pen_px, 1), "pen_mm": float(pen_mm), "deposit_mm": round(deposit_mm, 3),
             "mm_per_px": round(mm_per_px, 4), "res": int(resolution),
             "polys": len(polys), "lifts": n_lifts, "draw_m": round(length_mm / 1000, 1),
             "est_min": round(est_min, 1), "trace_s": round(trace_s, 1),
             "tone_levels": _lvl, "tone_thresholds": audit.get("tone_thresholds")}
    if baseline:
        stats["baseline_off"] = baseline
        stats["density_err_vs_v3"] = round(mean_derr - baseline["mean_density_err"], 4)
        stats["density_ok"] = bool(mean_derr <= baseline["mean_density_err"])
    return {"files": files, "stats": stats,
            "paths": {k: str((WORK_DIR / v).resolve()) for k, v in files.items()}}


