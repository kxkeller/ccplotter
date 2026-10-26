"""Opt-in contour-fill variants for Trace V3.

Two independent flags, both OFF by default (trace_v3._trace_svg only calls in
here when `core_fill == "hybrid"` or `edge_inset_mm > 0`; with both absent the
default ring loop runs and its output is byte-identical):

  core_fill="hybrid"  concentric rings stop once the remaining interior band's
                      INSCRIBED RADIUS drops below `core_r_mm`; what is left (a
                      "core") is filled with straight parallel passes at the
                      same pitch, boustrophedon-chained. No tiny closing loops.
  edge_inset_mm=d     the first ring sits d mm inside the shape edge instead of
                      0.5*footprint. Ring pitch is unchanged. Rings are traced
                      SUB-PIXEL (skimage marching squares on the distance
                      transform) so the requested inset is realised to well
                      under a pixel instead of being quantised to the pixel grid.

Rule, per connected component C of band_k = {dt > T_k}, T_k = L0 + k*fill_px
(L0 = 0.5*fill_px by default; inset_px + 0.5 when edge_inset_mm is set -- the 0.5 is
the pixel-centre -> pixel-edge offset of the Euclidean distance transform):
  * ring k of C is emitted iff its inscribed radius  maxdt(C) - T_k  >= core_r_px
    (the same rho_k >= R_HYBRID_MM rule as calibration_pocket.py's `hybrid` arm);
  * otherwise C is a CORE: ring k and every deeper ring of C are replaced by
    straight rows inside C's own boundary contour {dt = T_k}, spaced fill_px,
    starting AT the contour (where ring k would have been), parallel to C's
    principal axis, plus an extra last row on the far boundary if the remainder
    exceeds fill_px/2. A core narrower than one pitch gets a single centre row.
Curved / branching thin cores (area < CORE_MIN_RECT_FILL x the area of their
principal-axis bounding box) are NOT row-filled -- rows along one straight axis
would chop a bent stroke into hundreds of short chords. Those components keep
the default rings (long thin loops, not the tiny closing loops the hybrid removes);
the count is reported as n_core_curved_ring_levels.

core_fill="rows" (opt-in) -- same ring loop as the default until a region's
remaining inscribed radius drops below `core_r_mm` (default 1.0), then every remaining
core component C that is LARGE gets back-and-forth rows whatever its shape (no rect-fill
test, curved cores included). C is large when max(dt over C) >= core_r (a solid blob) or when
max(dt over C) >= core_r/2 and area(C) >= pi*core_r^2 (a wide curved/long core); thin line
work and small dots are never large and keep the untouched default ring path (same cv2 band,
same T = (k+0.5)*fill_px). See ROWS_* constants and `contour_passes(rows_cfg=...)`. Rows run along ONE fixed global
angle (ROWS_ANGLE_DEG, image/working frame), not per-region.

Why not reuse trace_v3._zigzag_fill: it anchors scanlines to a canvas grid
(arbitrary phase, up to one pitch of uncovered margin at each core edge, and a
per-call diagonal-canvas cost x thousands of cores); the core rows must start on
the core's own boundary so they continue the ring grid.
"""
import numpy as np


def _contours_subpixel(dt_crop, comp_crop, level, x0, y0):
    """Closed sub-pixel contours {dt = level} around the pixels of `comp_crop`
    (True where dt > level for this component). Returns [ (N,2) float arrays of
    (x, y) in full-image pixel-index coordinates ]. The field is dt on the
    component dilated by 1px (neighbours there have their TRUE dt <= level, so
    the interpolation is exact) and 0 elsewhere; zero padding closes contours."""
    import cv2
    from skimage.measure import find_contours
    k3 = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(comp_crop.astype(np.uint8), k3) > 0
    field = np.where(dil, dt_crop, 0.0).astype(np.float32)
    field = np.pad(field, 1, mode="constant", constant_values=0.0)
    out = []
    for c in find_contours(field, level):
        if len(c) < 4:
            continue
        xy = np.stack([c[:, 1] + (x0 - 1), c[:, 0] + (y0 - 1)], axis=1)
        out.append(xy)
    return out


def _simplify_closed(xy, eps):
    import cv2
    pts = xy[:-1] if (len(xy) > 2 and np.allclose(xy[0], xy[-1])) else xy
    if len(pts) < 3:
        return None
    ap = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), float(eps), True).reshape(-1, 2)
    if len(ap) < 2:
        return None
    return ap


def _polygon_from_contours(contours):
    """Even-odd polygon (holes handled) from closed (x, y) contours."""
    from shapely.geometry import Polygon
    poly = None
    for c in contours:
        if len(c) < 3:
            continue
        p = Polygon(c)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty:
            continue
        poly = p if poly is None else poly.symmetric_difference(p)
    return poly


CONNECTOR_MAX_PITCHES = 3.0   # row-end offset a chained turnaround may span (pen lifts vs straightness)
CONNECTOR_SLACK_PX = 0.5      # a connector may stray this far outside the (pixel-staircase) core outline
CORE_MIN_RECT_FILL = 0.6      # a core is row-filled only if area >= 0.6 x its principal-axis bounding box;
                              # curved / branching thin strokes fail and keep the default rings (see contour_passes)


ROWS_ANGLE_DEG = 0.0          # core_fill="rows": ONE global row direction (image/working-frame x axis)
ROWS_HALFWIDTH_FRAC = 0.5     # core_fill="rows" size rule, second clause: half-width floor, x core_r
OUTLINE_ROWS_RINGS = 1        # core_fill="outline_rows": outline rings before the rows (one global rule)


def core_rows(poly, ys, xs, pitch, min_chord=0.25, min_rect_fill=0.0, angle_deg=None, link_multi=False):
    """Straight parallel rows over `poly` (shapely), at `pitch`, parallel to the
    principal axis of the pixel set (xs, ys). Returns (list of polylines
    [(x, y), ...], n_rows_with_ink). Rows are sorted across the core and
    boustrophedon-chained only where the connector stays inside the polygon.
    Returns (None, -1) when the core is not straight enough for one global axis
    (area / principal-axis bounding box < min_rect_fill)."""
    from shapely.geometry import LineString
    from shapely.ops import linemerge
    if poly is None or poly.is_empty:
        return [], 0
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    if angle_deg is not None:
        ang = np.radians(float(angle_deg))
    elif len(xs) >= 2:
        c = np.cov(np.stack([xs - cx, ys - cy]))
        ang = 0.5 * np.arctan2(2.0 * c[0, 1], c[0, 0] - c[1, 1])
    else:
        ang = 0.0
    ca, sa = np.cos(ang), np.sin(ang)

    def fwd(x, y):   # image frame -> frame whose x axis is the principal axis
        dx, dy = x - cx, y - cy
        return dx * ca + dy * sa, -dx * sa + dy * ca

    def back(u, v):
        return cx + u * ca - v * sa, cy + u * sa + v * ca

    from shapely.ops import transform
    rp = transform(lambda x, y, z=None: fwd(np.asarray(x), np.asarray(y)), poly)
    minx, miny, maxx, maxy = rp.bounds
    ext = maxy - miny
    if ext >= pitch and min_rect_fill > 0 and rp.area < min_rect_fill * (maxx - minx) * ext:
        return None, -1
    if ext < pitch:
        vs = [0.5 * (miny + maxy)]
    else:
        m = int(np.floor(ext / pitch + 1e-9))
        vs = [miny + i * pitch for i in range(m + 1)]
        if ext - m * pitch > 0.5 * pitch:
            vs.append(maxy)
    eps = 1e-6 * max(1.0, ext)
    pad = (maxx - minx) + 1.0
    rows = []   # per row: list of (u0, u1) segments, left->right
    for v in vs:
        vv = min(max(v, miny + eps), maxy - eps) if ext > 2 * eps else v
        inter = rp.intersection(LineString([(minx - pad, vv), (maxx + pad, vv)]))
        segs = []
        if inter.is_empty:
            rows.append((vv, segs)); continue
        geoms = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
        for g in geoms:
            if g.geom_type == "MultiLineString":
                parts = list(g.geoms)
            elif g.geom_type == "LineString":
                parts = [g]
            else:
                continue
            for q in parts:
                us = [p[0] for p in q.coords]
                u0, u1 = min(us), max(us)
                if u1 - u0 >= min_chord:
                    segs.append((u0, u1))
        segs.sort()
        rows.append((vv, segs))
    chains = []          # each: list of (u, v) in the principal frame
    cur, flip, n_ink = None, False, 0
    cover = rp.buffer(CONNECTOR_SLACK_PX)
    for vv, segs in ([] if link_multi else rows):
        if not segs:
            continue
        n_ink += 1
        if len(segs) != 1:
            if cur: chains.append(cur)
            cur = None
            for (u0, u1) in segs:
                chains.append([(u0, vv), (u1, vv)])
            flip = False
            continue
        u0, u1 = segs[0]
        a, b = ((u1, vv), (u0, vv)) if flip else ((u0, vv), (u1, vv))
        if cur is not None:
            conn = LineString([cur[-1], a])
            # turnaround connector: must lie INSIDE the core (so it adds no ink outside it)
            # and span <= CONNECTOR_MAX_PITCHES pitches along the row axis; a longer
            # diagonal is not a turnaround -- start a new polyline (one pen lift) instead.
            if abs(cur[-1][0] - a[0]) <= CONNECTOR_MAX_PITCHES * pitch and cover.contains(conn):
                cur += [a, b]; flip = not flip
                continue
            chains.append(cur)
        cur = [a, b]
        flip = not flip
    if cur: chains.append(cur)
    if link_multi:      # core_fill="rows": rows with several segments stay chained (see _chain_rows_multi)
        n_ink = sum(1 for _, sg in rows if sg)
        chains = _chain_rows_multi(rows, cover, pitch)
    out = []
    for ch in chains:
        pts = [back(u, v) for (u, v) in ch]
        out.append([(float(x), float(y)) for x, y in pts])
    return out, n_ink


def _chain_rows_multi(rows, cover, pitch):
    """Boustrophedon chaining that also handles rows split into several segments (curved / branching
    cores). Each segment of a row is appended to the still-open chain of the previous ink row whose
    end is nearest to one of the segment's ends (the chain then runs through the segment starting from
    that end), provided the connector lies inside `cover` (the core outline + CONNECTOR_SLACK_PX) and
    spans <= CONNECTOR_MAX_PITCHES pitches along the row axis. Otherwise a new chain (one pen lift)
    starts. A chain not continued by the next row is closed. Deterministic (row order, segment order)."""
    from shapely.geometry import LineString
    done, active = [], []
    for vv, segs in rows:
        if not segs:
            done += active; active = []
            continue
        free, new_active = list(active), []
        for (u0, u1) in segs:
            best = None
            for ch in free:
                eu = ch[-1][0]
                for a, b in (((u0, vv), (u1, vv)), ((u1, vv), (u0, vv))):
                    d = abs(eu - a[0])
                    if d > CONNECTOR_MAX_PITCHES * pitch or (best is not None and d >= best[0]):
                        continue
                    if cover.contains(LineString([ch[-1], a])):
                        best = (d, ch, a, b)
            if best is not None:
                _, ch, a, b = best
                free.remove(ch)
                ch += [a, b]
                new_active.append(ch)
            else:
                new_active.append([(u0, vv), (u1, vv)])
        done += free
        active = new_active
    return done + active


def _axis_line(xs, ys):
    """Fallback straight pass through a pixel set along its principal axis."""
    if len(xs) < 2:
        return None
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    c = np.cov(np.stack([xs - cx, ys - cy]))
    ang = 0.5 * np.arctan2(2.0 * c[0, 1], c[0, 0] - c[1, 1])
    d = np.array([np.cos(ang), np.sin(ang)])
    t = (xs - cx) * d[0] + (ys - cy) * d[1]
    a = (cx + t.min() * d[0], cy + t.min() * d[1])
    b = (cx + t.max() * d[0], cy + t.max() * d[1])
    if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < 1e-6:
        return None
    return [(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))]


def contour_passes_outline_rows(dt, fill_px, max_levels, rows_cfg):
    """core_fill="outline_rows" (opt-in): OUTLINE_ROWS_RINGS ring(s) at
    T_k = (k+0.5)*fill_px (one closed ring per boundary, holes included), then EVERY connected
    component of {dt > (K+0.5)*fill_px} is filled with back-and-forth rows at pitch fill_px along
    the one global angle, starting AT that component's own contour (so the first row sits exactly
    one pitch inside the ring: no seam), chained by core_rows(link_multi=True). No size rule, no
    shape test, no deeper rings: no small closed rings can appear in the interior. A core that
    holds no row of >= min_chord gets one axis stroke, else is skipped (counted).
    Returns (ring_passes, core_passes, core_mask_u8, audit) like contour_passes."""
    import cv2
    H, W = dt.shape
    K = OUTLINE_ROWS_RINGS
    rings, cores = [], []
    core_mask = np.zeros((H, W), np.uint8)
    au = {"n_core_regions": 0, "n_core_rows": 0, "n_core_lines": 0,
          "n_core_skipped": 0, "n_core_curved_ring": 0, "n_core_curved_unique": 0,
          "n_rings": 0, "n_levels": 0, "n_core_small_ring": 0, "n_core_failed": 0}
    k = 0
    while k <= K and k < max_levels:
        T = (k + 0.5) * fill_px
        band = (dt > T) & (core_mask == 0)
        if not band.any():
            break
        au["n_levels"] = k + 1
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), 8)
        ok = np.ones(n, bool)
        ok[0] = False
        is_core = np.zeros(n, bool)
        if k == K:
            is_core[1:] = True
            ok[:] = False
        for i in np.nonzero(is_core)[0]:
            x, y, w, h = (int(v) for v in stats[i, :4])
            x0, y0 = max(0, x - 2), max(0, y - 2)
            x1, y1 = min(W, x + w + 2), min(H, y + h + 2)
            sub = lbl[y0:y1, x0:x1] == i
            ys, xs = np.nonzero(sub)
            xs = xs.astype(np.float64) + x0
            ys = ys.astype(np.float64) + y0
            try:
                cs = _contours_subpixel(dt[y0:y1, x0:x1], sub, T, x0, y0)
                poly = _polygon_from_contours(cs)
                polys, n_ink = core_rows(poly, ys, xs, fill_px, min_rect_fill=0.0,
                                         angle_deg=rows_cfg["angle_deg"], link_multi=True)
            except Exception:
                polys, n_ink = [], 0
                au["n_core_failed"] += 1
            core_mask[y0:y1, x0:x1][sub] = 255
            au["n_core_regions"] += 1
            if n_ink == 0:
                ln = _axis_line(xs, ys)
                if ln is None:
                    au["n_core_skipped"] += 1
                    continue
                polys = [ln]
                au["n_core_lines"] += 1
            else:
                au["n_core_rows"] += n_ink
            cores.extend(polys)
        if ok.any():
            bp = ok[lbl].astype(np.uint8)
            cnts, _ = cv2.findContours(bp, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
            for c in cnts:
                if len(c) >= 2:
                    cc = cv2.approxPolyDP(c, 0.4, True).reshape(-1, 2)
                    if len(cc) >= 2:
                        p = [(float(a), float(b)) for a, b in cc]
                        p.append(p[0])
                        rings.append(p)
                        au["n_rings"] += 1
        k += 1
    return rings, cores, core_mask, au


def contour_passes(dt, fill_px, level0, core_r_px, subpixel, max_levels, rows_cfg=None):
    """Level-set rings with the hybrid core rule and/or sub-pixel placement.
    core_r_px <= 0 -> no cores (plain rings, used by edge_inset_mm alone).
    rows_cfg (core_fill="rows" only; None = hybrid / inset behaviour): dict
    {"angle_deg": ...}. Failing components that are LARGE (own max dt >= core_r, or
    >= core_r/2 and area >= pi*core_r^2) get rows whatever their shape; the rest stay in the ring path.
    Returns (ring_passes, core_passes, core_mask_u8, audit)."""
    if rows_cfg is not None and rows_cfg.get("outline_rows"):
        return contour_passes_outline_rows(dt, fill_px, max_levels, rows_cfg)
    import cv2
    from scipy import ndimage
    H, W = dt.shape
    rings, cores = [], []
    core_mask = np.zeros((H, W), np.uint8)
    au = {"n_core_regions": 0, "n_core_rows": 0, "n_core_lines": 0,
          "n_core_skipped": 0, "n_core_curved_ring": 0, "n_core_curved_unique": 0,
          "n_rings": 0, "n_levels": 0, "n_core_small_ring": 0}
    large_org = None      # truthy only in rows mode
    if rows_cfg is not None and core_r_px > 0:
        large_org = True
    # a curved blob that falls back to rings is re-tested (and re-counted in
    # n_core_curved_ring) at every deeper level; curved_seen lets us count each blob once
    curved_seen = np.zeros((H, W), bool)
    simp_eps = 0.15 if subpixel else 0.4
    k = 0
    while k < max_levels:
        # rows mode (no sub-pixel): T written exactly as the default loop's (k + 0.5) * fill_px so
        # the ring bands are bit-identical to the default
        T = (k + 0.5) * fill_px if (large_org is not None and not subpixel) else level0 + k * fill_px
        # deeper levels never re-enter a region already claimed by a core
        band = (dt > T) & (core_mask == 0) if core_r_px > 0 else (dt > T)
        if not band.any():
            break
        au["n_levels"] = k + 1
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), 8)
        mx = np.zeros(n, np.float32)
        if n > 1:
            mx[1:] = ndimage.maximum(dt, lbl, index=np.arange(1, n))
        ok = (mx - T) >= core_r_px if core_r_px > 0 else np.ones(n, bool)
        ok[0] = False
        is_core = np.zeros(n, bool)
        if core_r_px > 0:
            is_core[1:] = ~ok[1:]
        if large_org is not None and is_core.any():
            # rows mode size rule, per failing component C (its OWN measures):
            #   half = max(dt) over C  = inscribed radius of the original shape where C sits
            #   LARGE iff half >= core_r                               (a solid blob)
            #          or (half >= ROWS_HALFWIDTH_FRAC*core_r and area(C) >= pi*core_r^2)
            #                                                          (a wide-enough curved/long core)
            # Everything else is not a core: it keeps the default ring path (re-tested at deeper
            # levels, exactly as the default loop runs it). Thin line work (half < core_r/2) is
            # therefore never row-filled however long it is.
            idx = np.nonzero(is_core)[0]
            half = mx[idx]
            area = stats[idx, cv2.CC_STAT_AREA].astype(np.float64)
            large = (half >= core_r_px) | ((half >= ROWS_HALFWIDTH_FRAC * core_r_px) & (area >= np.pi * core_r_px ** 2))
            ok[idx[~large]] = True
            au["n_core_small_ring"] += int((~large).sum())
            is_core[idx[~large]] = False
        # ---- cores for failing components (decided BEFORE the rings so that a curved /
        # branching thin core can fall back to the default rings instead) ----
        for i in np.nonzero(is_core)[0]:
            x, y, w, h = (int(v) for v in stats[i, :4])
            x0, y0 = max(0, x - 2), max(0, y - 2)
            x1, y1 = min(W, x + w + 2), min(H, y + h + 2)
            sub = lbl[y0:y1, x0:x1] == i
            ys, xs = np.nonzero(sub)
            xs = xs.astype(np.float64) + x0
            ys = ys.astype(np.float64) + y0
            polys = []
            try:
                cs = _contours_subpixel(dt[y0:y1, x0:x1], sub, T, x0, y0)
                poly = _polygon_from_contours(cs)
                if rows_cfg is None:
                    polys, n_ink = core_rows(poly, ys, xs, fill_px, min_rect_fill=CORE_MIN_RECT_FILL)
                else:
                    polys, n_ink = core_rows(poly, ys, xs, fill_px, min_rect_fill=0.0,
                                             angle_deg=rows_cfg["angle_deg"], link_multi=True)
            except Exception:
                polys, n_ink = [], 0
            if polys is None:       # curved / branching thin core: rows along one axis would chop it
                ok[i] = True
                au["n_core_curved_ring"] += 1
                if not curved_seen[y0:y1, x0:x1][sub].any():
                    au["n_core_curved_unique"] += 1
                curved_seen[y0:y1, x0:x1][sub] = True
                continue
            core_mask[y0:y1, x0:x1][sub] = 255
            au["n_core_regions"] += 1
            if n_ink == 0:
                ln = _axis_line(xs, ys)
                if ln is None:
                    au["n_core_skipped"] += 1
                    continue
                polys = [ln]; au["n_core_lines"] += 1
            else:
                au["n_core_rows"] += n_ink
            cores.extend(polys)
        # ---- rings for passing components ----
        if ok.any():
            if not subpixel:
                bp = ok[lbl].astype(np.uint8)
                cnts, _ = cv2.findContours(bp, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
                for c in cnts:
                    if len(c) >= 2:
                        cc = cv2.approxPolyDP(c, 0.4, True).reshape(-1, 2)
                        if len(cc) >= 2:
                            p = [(float(x), float(y)) for x, y in cc]
                            p.append(p[0])
                            rings.append(p); au["n_rings"] += 1
            else:
                for i in np.nonzero(ok)[0]:
                    x, y, w, h = (int(v) for v in stats[i, :4])
                    x0, y0 = max(0, x - 2), max(0, y - 2)
                    x1, y1 = min(W, x + w + 2), min(H, y + h + 2)
                    sub = lbl[y0:y1, x0:x1] == i
                    for xy in _contours_subpixel(dt[y0:y1, x0:x1], sub, T, x0, y0):
                        ap = _simplify_closed(xy, simp_eps)
                        if ap is None:
                            continue
                        p = [(float(a), float(b)) for a, b in ap]
                        p.append(p[0])
                        rings.append(p); au["n_rings"] += 1
        k += 1
    return rings, cores, core_mask, au
