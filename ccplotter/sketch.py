from pathlib import Path

from ccplotter.mask import _stroke_px_for, _skeleton_to_polylines, _binarize_lineart
from ccplotter.svg_io import _write_paths_svg, _read_polylines_svg
from ccplotter.lineart import _lineart_svg


def _generate_stipple_points(img, spacing=4.0, gamma=1.0, contrast=1.0, relax_iters=3):
    """Weighted (blue-noise) stippling for DrawBot-style TSP shading.

    Dot DENSITY tracks image darkness — dark regions pack points tightly, light
    regions leave them sparse — so the connecting TSP line reads as real tone
    instead of a flat scribble. Points are placed by darkness-weighted rejection
    sampling, then evened out with a few Lloyd relaxation passes (assign a cloud
    of darkness-weighted 'mass' samples to the nearest point, move each point to
    the centroid of its mass).

    The tone map is auto contrast-stretched (percentile) and pushed through a
    power curve so mid-grays collapse toward white and only true darks stay"""
    import numpy as np
    from scipy.spatial import KDTree
    h, w = img.shape
    img_f = img.astype(np.float32)
    p_low, p_high = np.percentile(img_f, (2, 98))
    img_f = np.clip((img_f - p_low) / (p_high - p_low + 1e-6), 0, 1)
    # Power curve: γ stretches mid-tones, contrast pushes ends apart.
    density = np.power(1.0 - img_f, gamma)
    density = np.clip(density * contrast, 0, 1)

    pts = []
    # 1. Rejection sampling (fast initial set)
    y, x = np.mgrid[0:h, 0:w]
    mask = np.random.rand(h, w) < (density * (0.5 / spacing))
    pts = np.column_stack((x[mask], y[mask])).astype(np.float32)
    if len(pts) < 10: return []

    # 2. Weighted Lloyd Relaxation
    # We sample 'mass' from the darkness map. Each point moves toward the
    # center of mass of its Voronoi region.
    for _ in range(relax_iters):
        tree = KDTree(pts)
        # Sample points to represent mass
        sy, sx = np.mgrid[0:h:int(spacing), 0:w:int(spacing)]
        samples = np.column_stack((sx.ravel(), sy.ravel()))
        weights = density[sy.ravel(), sx.ravel()]

        # Assign samples to nearest point
        _, idx = tree.query(samples)
        new_pts = np.zeros_like(pts)
        mass = np.zeros(len(pts))
        np.add.at(new_pts, idx, samples * weights[:, None])
        np.add.at(mass, idx, weights)

        valid = mass > 1e-6
        pts[valid] = new_pts[valid] / mass[valid][:, None]

    return pts.tolist()

def _solve_tsp(pts):
    """Nearest-neighbor + 2-opt local search. Effective for up to ~15k points."""
    import numpy as np
    P = np.asarray(pts, dtype=np.float32)
    n = len(P)
    if n < 3: return pts

    # 1. Nearest Neighbor
    order = [0]
    unused = set(range(1, n))
    curr = 0
    while unused:
        idx = list(unused)
        dists = np.sum((P[idx] - P[curr])**2, axis=1)
        best = idx[np.argmin(dists)]
        order.append(best)
        unused.remove(best)
        curr = best

    # 2-opt below indexes `order` with numpy arrays (order[js]); a Python list
    # can't be indexed by an array, so make it an ndarray.
    order = np.asarray(order, dtype=np.intp)

    # 2. 2-opt (Local search)
    # NumPy instead of a Python inner loop. E[k] holds the squared length of the
    # edge between order[k] and order[k+1]; reversing a segment only changes its
    # two boundary edges, so we recompute just E[i:j+1] after each accepted swap.
    window = 300
    max_iters = 12

    def _seg_edges(coords):
        d = coords[:-1] - coords[1:]
        return np.einsum("ij,ij->i", d, d)

    E = _seg_edges(P[order])  # squared edge lengths, length n-1

    for _ in range(max_iters):
        improved = False
        i = 0
        while i < n - 2:
            jmax = min(i + window, n - 1)
            js = np.arange(i + 2, jmax)
            if js.size == 0:
                i += 1
                continue

            p_i = P[order[i]]
            p_i1 = P[order[i + 1]]
            new_a = np.einsum("ij,ij->i", P[order[js]] - p_i, P[order[js]] - p_i)
            new_b = np.einsum("ij,ij->i", P[order[js + 1]] - p_i1, P[order[js + 1]] - p_i1)
            delta = (new_a + new_b) - (E[i] + E[js])

            k = int(np.argmin(delta))
            if delta[k] < -1e-9:
                j = int(js[k])
                order[i + 1:j + 1] = order[i + 1:j + 1][::-1]
                seg = P[order[i:j + 2]]
                E[i:j + 1] = np.einsum("ij,ij->i", seg[:-1] - seg[1:], seg[:-1] - seg[1:])
                improved = True
            i += 1
        if not improved:
            break

    return P[order].tolist()

def _split_route_on_jumps(route, max_jump):
    """Break the single TSP tour into separate strokes wherever it would have to
    leap across empty space. Each long hop becomes a PEN LIFT instead of a line
    drawn across the white sky. Returns a list of polylines (each >= 2 points)."""
    import numpy as np
    P = np.asarray(route, dtype=np.float64)
    if len(P) < 2:
        return [P.tolist()] if len(P) else []
    seg_len = np.linalg.norm(np.diff(P, axis=0), axis=1)
    # Split points: index i+1 starts a new stroke when the hop into it is too far.
    cuts = np.where(seg_len > max_jump)[0] + 1
    strokes = []
    for chunk in np.split(P, cuts):
        if len(chunk) >= 2:
            strokes.append(chunk.tolist())
    return strokes

def _lsd_svg(img, out_path: Path, opts: dict, min_len: int = 15) -> Path:
    """Native geometric line detection using OpenCV LSD. Much cleaner than
    tracing for blueprints/schematics."""
    import cv2, time
    t0 = time.time()
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(img)
    polys = []
    if lines is not None:
        if len(lines) > 20000:
            print(f"    [warn] LSD detected too many segments ({len(lines)}), capping")
            lines = lines[:20000]
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            if ((x1 - x2)**2 + (y1 - y2)**2)**0.5 >= min_len:
                polys.append([(float(x1), float(y1)), (float(x2), float(y2))])
    _write_paths_svg(out_path, img.shape[1], img.shape[0], _stroke_px_for(opts, img.shape[1]), polylines=polys)
    print(f"    [perf] _lsd_svg: {len(polys)} lines, {time.time()-t0:.2f}s")
    return out_path

def _centerline_svg(img, out_path: Path, opts: dict) -> Path:
    """Improved single-stroke centerline trace: Adaptive binarization preserves
    fine detail and avoids 'polygon artifacts' in shaded areas."""
    import cv2, time
    from skimage.morphology import skeletonize
    t0 = time.time()
    # Adaptive threshold is much more robust for textured paper/shading
    binimg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9)
    skel = skeletonize(binimg > 0)

    skel_pixels = int(skel.sum())
    if skel_pixels > 150000:
        raise RuntimeError(f"Centerline too complex ({skel_pixels}px). Try higher resolution or cleaner source.")

    polys = _skeleton_to_polylines(skel, prune_len=8)
    _write_paths_svg(out_path, img.shape[1], img.shape[0], _stroke_px_for(opts, img.shape[1]), polylines=polys)
    print(f"    [perf] _centerline_svg: {len(polys)} paths, {time.time()-t0:.2f}s")
    return out_path

def _hatch_geom(img, spacing=3.0, levels=2, angle=45, contrast=1.0):
    """Generate tonal hatching lines. Darker areas get more overlapping lines
    at different angles."""
    import cv2, numpy as np, time
    t0 = time.time()
    h, w = img.shape
    darkness = 1.0 - (img.astype(np.float32) / 255.0)
    darkness = np.power(darkness, contrast)

    all_lines = []
    max_hatch_lines = 50000
    for i in range(levels):
        if len(all_lines) >= max_hatch_lines: break
        # Rotating angle for cross-hatching
        curr_angle = (angle + i * 45) % 180
        # Threshold for this level
        thresh = (i + 1) / (levels + 1)
        mask = (darkness > thresh).astype(np.uint8) * 255
        if cv2.countNonZero(mask) == 0: continue

        # Fast hatching via rotation and scanning
        M = cv2.getRotationMatrix2D((w/2, h/2), curr_angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += (nw / 2) - w / 2
        M[1, 2] += (nh / 2) - h / 2
        rot_mask = cv2.warpAffine(mask, M, (nw, nh))

        invM = cv2.invertAffineTransform(M)
        step = max(1, int(spacing))
        for y in range(0, nh, step):
            if len(all_lines) >= max_hatch_lines: break
            row = rot_mask[y, :]
            runs = np.where(np.diff(row.astype(np.int16)) != 0)[0]
            if row[0] > 0: runs = np.insert(runs, 0, 0)
            if row[-1] > 0: runs = np.append(runs, nw - 1)
            for j in range(0, len(runs) - 1, 2):
                x0, x1 = runs[j], runs[j+1]
                if x1 - x0 < 2: continue
                p0 = np.dot(invM, [x0, y, 1])
                p1 = np.dot(invM, [x1, y, 1])
                all_lines.append([(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))])

    print(f"  [perf] _hatch_geom: {len(all_lines)} lines, {time.time()-t0:.2f}s")
    if len(all_lines) >= max_hatch_lines:
        print("  [warn] hatching capped at 50,000 lines")
    return all_lines

_SKETCH_AUDIT = {}   # last _sketch_svg run (Sketch V2)

def _sketch_svg(img, out_path: Path, opts: dict, debug_prefix: str = None) -> Path:
    """Sketch V2 = the EXACT line-art trace (base geometry, identical fidelity) PLUS
    conservative tonal weight. The geometry is NOT re-derived — it reuses _lineart_svg,
    so the subject stays as faithful/recognizable as line art and dense hatching stays
    line art's many marks. On top, genuinely-thick strokes get parallel offset passes
    (Line Weight Gain). Solid-region FILL (making darks darker) is a later phase; here a
    dark solid is just line art's (hollowed) outline, undeformed. Clean mode untouched.

    Pen width is a TEMPORARY pixel estimate (final plot scale unknown here); logged."""
    import cv2, numpy as np, time
    from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
    t0 = time.time()
    H, W = img.shape[:2]
    global _SKETCH_AUDIT

    # 1. BASE GEOMETRY = the exact line-art trace (reuse the clean tracer UNCHANGED, so
    #    Sketch is "line art + weight", never a separate/worse drawing): same fidelity,
    #    same hatching, same hollowed solids.
    base_svg = Path(str(out_path) + ".lineartbase.svg")
    _lineart_svg(img, base_svg, opts, debug_prefix=debug_prefix)
    base_polys = _read_polylines_svg(base_svg)
    try: base_svg.unlink()
    except Exception: pass

    gain = min(1.0, max(0.0, float(opts.get("sketch_gain", 0.12))))
    max_passes = max(1, int(opts.get("sketch_max_passes", 2)))
    # gain 0 (or no geometry) → Sketch IS line art, byte-for-byte.
    if gain <= 0 or not base_polys:
        _write_paths_svg(out_path, W, H, _stroke_px_for(opts, W), polylines=base_polys)
        _SKETCH_AUDIT = {"gain": round(gain, 2), "max_passes": max_passes,
                         "base_strokes": len(base_polys), "thickened": 0,
                         "centerline_only": len(base_polys), "offset_runs": 0,
                         "total_polys": len(base_polys)}
        print(f"  [sketchv2] gain={gain:.2f} -> line-art geometry only "
              f"({len(base_polys)} strokes) [perf {time.time()-t0:.2f}s]")
        return out_path

    # 2. Source ink width map for tonal-weight decisions: un-hollowed mask (duplicated
    #    binarization so _lineart_svg stays untouched; this mask is ONLY for widths).
    tmode = str(opts.get("threshold_mode", "otsu")).lower()
    if tmode == "sensitivity":
        mask = _binarize_lineart(img, str(opts.get("ink_sensitivity", "sensitive")))
    else:
        bilevel = float(np.mean((img < 40) | (img > 215))) > 0.9
        src = img if bilevel else cv2.GaussianBlur(img, (3, 3), 0)
        if bilevel:
            _, mask = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY_INV)
        elif tmode == "manual":
            _, mask = cv2.threshold(src, int(opts.get("manual_threshold", 128)), 255, cv2.THRESH_BINARY_INV)
        elif tmode == "adaptive":
            b = int(opts.get("adaptive_block", 25)); b = max(3, b + 1 - (b % 2))
            mask = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, b, int(opts.get("adaptive_c", 7)))
        else:
            _, mask = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = mask > 0
    min_area = int(opts.get("sketch_min_area", 6))
    if min_area > 0:
        mask = remove_small_objects(mask, min_size=min_area)
    mask = remove_small_holes(mask, area_threshold=4)
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    sk_dt = dt[skeletonize(mask)]
    pen_px = float(opts.get("sketch_spacing_px", 0)) or (max(1.0, 2.0 * float(np.percentile(sk_dt, 25))) if sk_dt.size else 1.0)
    # Offset pitch ~0.68x pen width: slightly tighter so the parallel rails overlap a touch
    # more and the thickened stroke reads as ONE thicker line, not several neighboring rails.
    spacing = max(0.55, pen_px * 0.68)

    # 3. LINE-WEIGHT BUILD-UP on the line-art base. Each base stroke is kept as-is, then
    #    THICKENED (parallel passes beside it) by how "major" it is:
    #      - long structural strokes (silhouette / major edges) -> bolder,
    #      - strokes inside DARK/DENSE regions -> bolder (tonal weight build-up),
    #      - short, isolated fine detail (windows / text / panel lines) -> stays thin.
    #    Passes BUILD weight (parallel, NOT clamped to source width) so the difference
    #    from line art is visible. gain x max_passes controls how dramatic.
    # 3. LINE-WEIGHT BUILD-UP — COMPONENT-level, contour-biased, with a hard TEXTURE gate.
    #    Weight is NOT decided per stroke fragment (that made fragments of the SAME object —
    #    a "472" label, a wheel rim, a side rod — jump thick->thin against each other). Instead
    #    we (a) HARD-protect strokes inside a dense field of parallel siblings (leaf springs,
    #    spokes, fins, rivets, bolts, pipes, hatch — fine repeated detail that must stay thin),
    #    then (b) GROUP the remaining strokes into connected COMPONENTS (fragments whose
    #    endpoints nearly touch), and (c) assign ONE uniform pass count per component. So a
    #    contour/glyph/rim carries a single consistent weight with no thin->huge->thin jumps.
    #    Component importance = total length x isolation; only the top `gain` fraction thickens,
    #    ramping smoothly 0->max_passes (one offset line per level, alternating sides).
    import math
    lengths = []
    for p in base_polys:
        a = np.asarray(p, float)
        lengths.append(float(np.hypot(*(a[1:] - a[:-1]).T).sum()) if len(a) >= 2 else 0.0)
    Nc = len(base_polys)
    # local LINE density (counts nearby sibling strokes) from rasterized centerlines
    lines_img = np.zeros((H, W), np.float32)
    for p in base_polys:
        a = np.asarray(p, float)
        if len(a) >= 2:
            cv2.polylines(lines_img, [np.round(a).astype(np.int32)], False, 1.0, 1)
    sigma = max(4.0, 4.0 * pen_px)
    line_dens = cv2.GaussianBlur(lines_img, (0, 0), sigma)
    # ABSOLUTE calibration: a single isolated 1px line blurs to this peak density. Dividing
    # the local density (minus self) by it gives ~the number of NEARBY SIBLING lines. This
    # is context-independent, so dense fields are caught even when they fill the whole crop
    # (a median-relative gate failed there — the median was already texture-level).
    peak_iso = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    SIB_GATE = 1.6                                          # >= ~1.6 packed neighbor lines -> protected. Biased
                                                            # HARDER toward protecting repeated parallel detail
                                                            # (leaf springs, spokes, rivet rows, fins) — their
                                                            # meaning is spacing+repetition, not heaviness.

    # 3a. per-stroke sibling count + hard texture/short-detail protection (never thickens)
    siblings = np.zeros(Nc, dtype=float)
    protected = np.zeros(Nc, dtype=bool)                    # in a dense parallel field -> stay thin
    too_short = np.zeros(Nc, dtype=bool)                    # fine detail / dust -> stay thin
    n_protected = 0
    for i, (poly, seglen) in enumerate(zip(base_polys, lengths)):
        a = np.asarray(poly, float)
        if len(a) < 2 or seglen < max(8.0, 3.0 * pen_px):
            too_short[i] = True; continue
        xi = np.clip(np.round(a[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(a[:, 1]).astype(int), 0, H - 1)
        siblings[i] = max(0.0, float(line_dens[yi, xi].mean()) - peak_iso) / peak_iso
        if siblings[i] >= SIB_GATE:
            protected[i] = True; n_protected += 1

    eligible = [i for i in range(Nc) if not protected[i] and not too_short[i]]

    # 3b. GROUP eligible fragments into connected components by endpoint proximity (union-find).
    #     Fragments whose endpoints nearly touch belong to the same physical stroke/object, so
    #     they must share one weight. join_d scales with pen width (a real corner gap is small).
    parent = list(range(Nc))
    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb: parent[rb] = ra
    join_d = max(3.0, 2.2 * pen_px)                         # a touch looser: fragments of one long contour
                                                            # merge into ONE component -> smoother, less chunky
    cell = max(1.0, join_d)
    grid = {}                                               # cell -> [(idx, ex, ey)] endpoint hash
    for i in eligible:
        a = np.asarray(base_polys[i], float)
        for ex, ey in (a[0], a[-1]):
            grid.setdefault((int(ex // cell), int(ey // cell)), []).append((i, ex, ey))
    jd2 = join_d * join_d
    for i in eligible:
        a = np.asarray(base_polys[i], float)
        for ex, ey in (a[0], a[-1]):
            kx, ky = int(ex // cell), int(ey // cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for (j, jx, jy) in grid.get((kx + dx, ky + dy), ()):
                        if j != i and (ex - jx) ** 2 + (ey - jy) ** 2 <= jd2:
                            _union(i, j)
    comps = {}
    for i in eligible:
        comps.setdefault(_find(i), []).append(i)

    # 3c. SUBJECT-FIRST targeting (spend the budget on MAJOR forms, not background marks).
    #     Design intent: LESS total thickening, BETTER targeted. A component only competes for
    #     weight if it is among the LONGER forms (>= 60th pct of component length, and above an
    #     absolute pen-scaled floor) — silhouette, side rods, wheel rims, frame edges. Everything
    #     shorter (ballast pebbles, stray hatch marks, background clutter, glyphs) stays thin.
    #     Among the major forms, importance = length x isolation; only the top `gain` thickens.
    comp_roots = list(comps.keys())
    Ncomp = len(comp_roots)
    comp_len = np.array([sum(lengths[i] for i in comps[r]) for r in comp_roots], dtype=float)
    comp_sib = np.array([float(np.mean([siblings[i] for i in comps[r]])) for r in comp_roots], dtype=float)
    comp_bbox = np.zeros((Ncomp, 4), dtype=float)              # x0,y0,x1,y1 (for label clustering)
    for n, r in enumerate(comp_roots):
        pts = np.concatenate([np.asarray(base_polys[i], float) for i in comps[r]], axis=0)
        comp_bbox[n] = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())

    len_floor = max(float(np.percentile(comp_len, 60)) if comp_len.size else 0.0, 8.0 * pen_px)
    major = comp_len >= len_floor                             # subject-scale forms only compete for weight
    Llo = float(np.percentile(comp_len[major], 40)) if major.any() else 0.0
    Lhi = max(float(np.percentile(comp_len[major], 90)) if major.any() else 1.0, Llo + 1.0)
    f_len = np.clip((comp_len - Llo) / (Lhi - Llo), 0.0, 1.0)
    f_iso = np.clip(1.0 - comp_sib / SIB_GATE, 0.0, 1.0)
    comp_imp = np.where(major, f_len * f_iso, 0.0)
    comp_rank = np.zeros(Ncomp, dtype=float)
    cidx = np.where(comp_imp > 0.0)[0]
    if cidx.size:
        order = cidx[np.argsort(comp_imp[cidx], kind="stable")]
        comp_rank[order] = np.arange(cidx.size) / max(1, cidx.size - 1)
    comp_passes = np.zeros(Ncomp, dtype=int)
    for n in range(Ncomp):
        if comp_imp[n] <= 0.0:
            continue
        g = float(np.clip((comp_rank[n] - (1.0 - gain)) / max(gain, 1e-3), 0.0, 1.0))
        comp_passes[n] = int(round(max_passes * g))

    # 3c-2. LABEL / TEXT grouping. Cluster nearby SMALL components (numerals like "472", plate
    #     text) into one visual unit, then force every member to ONE uniform weight (the group
    #     minimum — thin after the subject-first gate) so no glyph or stroke-within-a-glyph is
    #     bolder than its neighbors. Grid-hashed by bbox center to stay near-linear.
    small = np.where(comp_len < len_floor)[0]
    n_label_groups = 0
    if small.size >= 3:
        lp = list(range(Ncomp))
        def _lfind(x):
            while lp[x] != x: lp[x] = lp[lp[x]]; x = lp[x]
            return x
        gap = max(6.0, 4.0 * pen_px)                          # ~1.5 character gaps between glyphs
        cellL = max(1.0, gap)
        ghash = {}
        for n in small:
            cx = 0.5 * (comp_bbox[n, 0] + comp_bbox[n, 2]); cy = 0.5 * (comp_bbox[n, 1] + comp_bbox[n, 3])
            ghash.setdefault((int(cx // cellL), int(cy // cellL)), []).append(n)
        for n in small:
            ax0, ay0, ax1, ay1 = comp_bbox[n]
            cx = 0.5 * (ax0 + ax1); cy = 0.5 * (ay0 + ay1)
            kx, ky = int(cx // cellL), int(cy // cellL)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for m in ghash.get((kx + dx, ky + dy), ()):
                        if m <= n: continue
                        bx0, by0, bx1, by1 = comp_bbox[m]
                        gx = max(0.0, max(ax0, bx0) - min(ax1, bx1))
                        gy = max(0.0, max(ay0, by0) - min(ay1, by1))
                        if gx <= gap and gy <= gap:
                            ra, rb = _lfind(int(n)), _lfind(int(m))
                            if ra != rb: lp[rb] = ra
        groups = {}
        for n in small:
            groups.setdefault(_lfind(int(n)), []).append(int(n))
        for members in groups.values():
            if len(members) < 3: continue                     # a label = several small marks clustered together
            n_label_groups += 1
            w = int(min(comp_passes[m] for m in members))     # uniform across the whole label, biased thin
            for m in members: comp_passes[m] = w

    passes_by_root = {r: int(comp_passes[n]) for n, r in enumerate(comp_roots)}

    # 3d. emit: every base stroke as-is; offset passes added uniformly per component.
    out_polys, n_thin, n_thick, n_off = [], 0, 0, 0
    pass_hist = {1: 0, 2: 0, 3: 0, 4: 0}
    for i, poly in enumerate(base_polys):
        out_polys.append(poly)                                  # line-art geometry, as-is
        passes = passes_by_root.get(_find(i), 0) if (not protected[i] and not too_short[i]) else 0
        if passes <= 0:
            n_thin += 1; continue
        n_thick += 1; pass_hist[min(passes, 4)] = pass_hist.get(min(passes, 4), 0) + 1
        P = np.asarray(poly, float)
        T = np.zeros_like(P); T[1:-1] = P[2:] - P[:-2]; T[0] = P[1] - P[0]; T[-1] = P[-1] - P[-2]
        Ln = np.hypot(T[:, 0], T[:, 1]); Ln[Ln == 0] = 1.0; T /= Ln[:, None]
        Nrm = np.stack([-T[:, 1], T[:, 0]], 1)                  # per-vertex unit normal
        for k in range(1, passes + 1):                          # one line per level, alternating sides
            side = -1.0 if (k % 2 == 1) else 1.0
            off = P + (side * ((k + 1) // 2) * spacing) * Nrm
            off[:, 0] = np.clip(off[:, 0], 0, W - 1); off[:, 1] = np.clip(off[:, 1], 0, H - 1)
            out_polys.append([(float(ox), float(oy)) for ox, oy in off])
            n_off += 1

    _write_paths_svg(out_path, W, H, _stroke_px_for(opts, W), polylines=out_polys)
    print(f"  [sketchv2] base={len(base_polys)} components={len(comp_roots)} major={int(major.sum())} labels={n_label_groups} "
          f"| gain={gain:.2f} maxp={max_passes} pen_px={pen_px:.1f} "
          f"| thickened={n_thick} protected(texture)={n_protected} thin={n_thin} | "
          f"1p={pass_hist.get(1,0)} 2p={pass_hist.get(2,0)} 3p={pass_hist.get(3,0)} 4p+={pass_hist.get(4,0)} "
          f"[perf {time.time()-t0:.2f}s]")
    _SKETCH_AUDIT = {"gain": round(gain, 2), "max_passes": max_passes,
                     "base_strokes": len(base_polys), "components": len(comp_roots),
                     "major_forms": int(major.sum()), "label_groups": n_label_groups, "thickened": n_thick,
                     "protected_texture": n_protected, "centerline_only": n_thin,
                     "pass1": pass_hist.get(1, 0), "pass2": pass_hist.get(2, 0),
                     "pass3": pass_hist.get(3, 0), "pass4plus": pass_hist.get(4, 0),
                     "offset_runs": n_off, "pen_px": round(pen_px, 2), "total_polys": len(out_polys)}
    return out_path
