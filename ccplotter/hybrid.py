from pathlib import Path

from ccplotter.config import WORK_DIR
from ccplotter.mask import _skeleton_to_polylines, _load_gray, _stroke_px_for
from ccplotter.svg_io import _write_paths_svg

# ── Hybrid mode (adaptive fusion) ───────────────────────────────────────────
# Deterministic CV: generate candidate line layers from several detectors
# (geometric LSD, multi-scale Canny edges, ink/threshold centerlines), score
# them by length × gradient support, fuse via an occupancy grid (dedup), stitch
# visually-continuous fragments, then hand off to the normal vpype cleanup +
# low-lift routing. Not subject-specific — adaptive weighting decides which
# detector to trust per image.

HYBRID_PRESETS = {
    # geom_min_px: LSD min segment length · edge_variants: # Canny passes
    # min_len_px/min_support: candidate filter · dedup_tol/overlap: dedup grid
    # stitch_gap_px: visual stitch reach · *_bias: per-source score multiplier
    "balanced": {"geom_min_px": 18, "edge_variants": 3, "ink_modes": ["otsu", "adaptive"],
                 "min_len_px": 10, "min_support": 0.06, "dedup_tol": 3.0, "dedup_overlap": 0.6,
                 "stitch_gap_px": 6.0, "max_candidates": 3000, "max_accepted": 2000,
                 "geom_bias": 1.0, "edge_bias": 1.0, "ink_bias": 1.0},
    "sparse":   {"geom_min_px": 30, "edge_variants": 2, "ink_modes": ["otsu"],
                 "min_len_px": 18, "min_support": 0.10, "dedup_tol": 4.0, "dedup_overlap": 0.5,
                 "stitch_gap_px": 8.0, "max_candidates": 1500, "max_accepted": 900,
                 "geom_bias": 1.2, "edge_bias": 0.8, "ink_bias": 0.9},
    "detail":   {"geom_min_px": 12, "edge_variants": 3, "ink_modes": ["otsu", "adaptive"],
                 "min_len_px": 6, "min_support": 0.04, "dedup_tol": 2.5, "dedup_overlap": 0.7,
                 "stitch_gap_px": 5.0, "max_candidates": 5000, "max_accepted": 3000,
                 "geom_bias": 1.0, "edge_bias": 1.1, "ink_bias": 1.0},
    "bold":     {"geom_min_px": 28, "edge_variants": 2, "ink_modes": ["otsu"],
                 "min_len_px": 22, "min_support": 0.10, "dedup_tol": 4.0, "dedup_overlap": 0.5,
                 "stitch_gap_px": 12.0, "max_candidates": 2000, "max_accepted": 1200,
                 "geom_bias": 1.3, "edge_bias": 1.0, "ink_bias": 0.7},
}

def _poly_len_px(pts):
    import numpy as np
    p = np.asarray(pts, float)
    return float(np.hypot(*(p[1:] - p[:-1]).T).sum()) if len(p) >= 2 else 0.0

def _resample_pts(pts, step=2.0):
    """Even ~step-px samples along a polyline (for support + occupancy tests)."""
    import numpy as np
    p = np.asarray(pts, float)
    if len(p) < 2: return p
    seg = np.hypot(*(p[1:] - p[:-1]).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-6: return p[:1]
    d = np.linspace(0, total, max(2, int(total / step) + 1))
    return np.column_stack([np.interp(d, cum, p[:, 0]), np.interp(d, cum, p[:, 1])])

def _grad_mag(img):
    """Normalised gradient magnitude — used to score how well a line is backed
    by real image contrast (line support)."""
    import cv2, numpy as np
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    return m / (float(m.max()) + 1e-6)

def _line_support(pts, gmag):
    import numpy as np
    s = _resample_pts(pts, 2.0)
    if len(s) == 0: return 0.0
    h, w = gmag.shape
    xs = np.clip(s[:, 0].astype(int), 0, w - 1)
    ys = np.clip(s[:, 1].astype(int), 0, h - 1)
    return float(gmag[ys, xs].mean())

def _hybrid_geom_candidates(img, min_px):
    """Layer A: long straight/structural segments via LSD."""
    import cv2, numpy as np
    out = []
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(img)
    if lines is None: return out
    if len(lines) > 20000: lines = lines[:20000]
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 >= min_px:
            out.append({"pts": np.array([[x1, y1], [x2, y2]], float), "source": "geom"})
    return out

def _hybrid_edge_candidates(img, variants, max_skel=120000):
    """Layer B: curved photo contours via multi-scale Canny → skeleton polylines."""
    import cv2, numpy as np
    from skimage.morphology import skeletonize
    out = []
    sm = cv2.bilateralFilter(img, 7, 60, 60)
    for lo, hi in variants:
        edges = cv2.Canny(sm, int(lo), int(hi))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        sk = skeletonize(edges > 0)
        if int(sk.sum()) > max_skel: continue
        for poly in _skeleton_to_polylines(sk, prune_len=8):
            out.append({"pts": np.array(poly, float), "source": "edge"})
    return out

def _hybrid_ink_candidates(img, modes, opts, max_skel=120000, min_area=None, prune_len=None):
    """Layer C: true ink/threshold strokes via threshold → skeleton polylines.
    Skips a threshold variant whose mask is too dense (likely shading/texture).
    `min_area`/`prune_len` let small-detail mode keep tinier marks."""
    import cv2, numpy as np
    from skimage.morphology import skeletonize, remove_small_objects
    out = []
    if min_area is None: min_area = int(opts.get("lineart_min_area", 40))
    if prune_len is None: prune_len = int(opts.get("prune_len", 8))
    g = cv2.GaussianBlur(img, (3, 3), 0)
    for m in modes:
        if m == "otsu":
            _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif m == "adaptive":
            mask = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 7)
        elif m == "manual":
            _, mask = cv2.threshold(g, int(opts.get("manual_threshold", 128)), 255, cv2.THRESH_BINARY_INV)
        else:
            continue
        mb = mask > 0
        cov = float(mb.mean())
        if cov > 0.35:            # too much ink → shaded/noisy, drop this candidate
            continue
        if min_area > 0:
            mb = remove_small_objects(mb, min_size=int(min_area))
        sk = skeletonize(mb)
        if int(sk.sum()) > max_skel: continue
        for poly in _skeleton_to_polylines(sk, prune_len=int(prune_len)):
            out.append({"pts": np.array(poly, float), "source": "ink", "coverage": cov})
    return out

def _hybrid_auto_weights(geom, edges, ink):
    """Estimate per-source reliability from candidate statistics (general, not
    subject-specific). Lots of tiny Canny fragments → downweight edges; dense
    ink mask → downweight ink; many long LSD lines → upweight geometry."""
    import numpy as np
    gw = ew = iw = 1.0
    if len(edges) > 4000: ew *= 0.5
    elif len(edges) > 2000: ew *= 0.7
    if len(geom) >= 20: gw *= 1.2
    if ink:
        cov = float(np.mean([c.get("coverage", 0.0) for c in ink]))
        if cov > 0.20: iw *= 0.5
        elif cov < 0.05: iw *= 1.3
    return gw, ew, iw

def _hybrid_fuse(cands, gmag, params):
    """Score, filter, and deduplicate candidate lines via an occupancy grid.
    Highest-scoring (longest, best-supported) lines are laid down first; later
    lines that mostly overlap already-claimed cells are rejected as duplicates.
    Returns (accepted, rejected, diag) where diag breaks down WHY lines were
    dropped (short / low-support / duplicate) and by source — key for diagnosing
    small-detail loss (e.g. ink strokes rejected as duplicates)."""
    import numpy as np
    from collections import defaultdict
    w = params["weights"]
    for c in cands:
        c["length"] = _poly_len_px(c["pts"])
        c["support"] = _line_support(c["pts"], gmag)
        c["score"] = c["length"] * (0.4 + c["support"]) * w.get(c["source"], 1.0)

    reasons = {"short": 0, "low_support": 0, "duplicate": 0}
    acc_src, rej_src = defaultdict(int), defaultdict(int)
    kept = []
    for c in cands:
        if c["length"] < params["min_len_px"]:
            reasons["short"] += 1; rej_src[c["source"]] += 1; continue
        if c["support"] < params["min_support"] and c["source"] != "ink":
            reasons["low_support"] += 1; rej_src[c["source"]] += 1; continue
        kept.append(c)
    kept.sort(key=lambda c: -c["score"])
    kept = kept[:params["max_candidates"]]

    tol, ov = params["dedup_tol"], params["dedup_overlap"]
    relax = bool(params.get("ink_dedup_relax"))   # small-detail: keep dense ink clusters
    occ, accepted, rejected = set(), [], []
    def cells(pts):
        s = _resample_pts(pts, max(1.0, tol * 0.6))
        return {(int(x // tol), int(y // tol)) for x, y in s}
    for c in kept:
        cs = cells(c["pts"])
        if not cs: continue
        ovc = min(0.95, ov + 0.25) if (relax and c["source"] == "ink") else ov
        frac = sum(1 for k in cs if k in occ) / len(cs)
        if frac > ovc:
            reasons["duplicate"] += 1; rej_src[c["source"]] += 1; rejected.append(c)
        else:
            accepted.append(c); acc_src[c["source"]] += 1
            for (cx, cy) in cs:                 # dilate by 1 cell (~tol) for tolerance
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        occ.add((cx + dx, cy + dy))
            if len(accepted) >= params["max_accepted"]:
                break
    diag = {"reasons": reasons, "accepted_by_source": dict(acc_src), "rejected_by_source": dict(rej_src)}
    return accepted, rejected, diag

def _hybrid_stitch(lines, gap_px):
    """Visual stitching (distinct from low-lift routing): join fragment endpoints
    that are close AND tangent-compatible, so a broken contour becomes one line.
    Greedy; bounded by caller (skipped above ~3000 lines)."""
    import numpy as np
    segs = [np.asarray(l, float) for l in lines if len(l) >= 2]
    n = len(segs)
    if n <= 1: return segs
    used = [False] * n
    def tangent(seg, end):
        k = min(4, len(seg) - 1)
        v = (seg[k] - seg[0]) if end == 0 else (seg[-1 - k] - seg[-1])
        nv = np.linalg.norm(v); return v / nv if nv > 0 else v
    result = []
    for i in range(n):
        if used[i]: continue
        used[i] = True
        chain = [p for p in segs[i]]
        for _pass in range(2):                  # grow tail, flip, grow other end
            grew = True
            while grew:
                grew = False
                tail = np.asarray(chain[-1])
                ct = np.asarray(chain[-1]) - np.asarray(chain[-2])
                nct = np.linalg.norm(ct)
                if nct == 0: break
                ct /= nct
                best, bestd = None, gap_px
                for j in range(n):
                    if used[j]: continue
                    for end in (0, 1):
                        ep = segs[j][0] if end == 0 else segs[j][-1]
                        d = float(np.linalg.norm(ep - tail))
                        if d <= bestd and float(np.dot(ct, tangent(segs[j], end))) >= 0.5:
                            bestd, best = d, (j, end)
                if best is not None:
                    j, end = best; used[j] = True
                    seg = segs[j] if end == 0 else segs[j][::-1]
                    chain.extend([p for p in seg])
                    grew = True
            chain = chain[::-1]
        result.append(np.asarray(chain))
    return result

def _hybrid_objective(lines, gmag):
    """Transparency metric: supported line length minus a stroke-count penalty."""
    import numpy as np
    if not lines: return 0.0
    total = sum(_poly_len_px(l) for l in lines)
    sup = float(np.mean([_line_support(l, gmag) for l in lines]))
    return total * sup - len(lines) * 5.0

# Per-level params for the small-detail lane. Detail is scored by local
# contrast/darkness/cluster coherence — NOT length — and has its own caps so it
# can't explode the whole plot. General: text, logos, labels, icons, fine marks.
# The lane works by finding small-detail ROIs and reprocessing each crop locally
# at higher effective resolution, so skeletonization doesn't shred tiny strokes.
HYBRID_DETAIL_PARAMS = {
    "balanced": {"min_area": 6, "prune": 4, "bh_k": 13, "ad_block": 15, "ad_c": 6,
                 "detail_min_len_px": 4.0, "score_floor": 0.35, "dedup_overlap": 0.85,
                 "max_detail_candidates": 1500, "max_detail_accepted": 600,
                 "cluster_radius": 14.0, "dense_neigh": 40,
                 # ROI detection + local reprocessing
                 "roi_bh_k": 13, "min_comp_area": 4, "max_comp_area": 1200, "max_comp_dim": 60,
                 "merge_dilate": 17, "min_region_ink": 40, "min_roi_dim": 14,
                 "max_roi_area_frac": 0.16, "max_rois": 8, "max_total_area_frac": 0.45, "roi_pad": 8,
                 "hires_max": 2200, "roi_upscale": 2.0, "max_crop_px": 1_600_000, "max_detail_per_roi": 220},
    "preserve": {"min_area": 3, "prune": 3, "bh_k": 15, "ad_block": 13, "ad_c": 5,
                 "detail_min_len_px": 2.5, "score_floor": 0.25, "dedup_overlap": 0.92,
                 "max_detail_candidates": 3000, "max_detail_accepted": 1200,
                 "cluster_radius": 16.0, "dense_neigh": 60,
                 "roi_bh_k": 15, "min_comp_area": 3, "max_comp_area": 2000, "max_comp_dim": 80,
                 "merge_dilate": 21, "min_region_ink": 25, "min_roi_dim": 12,
                 "max_roi_area_frac": 0.22, "max_rois": 14, "max_total_area_frac": 0.6, "roi_pad": 10,
                 "hires_max": 2600, "roi_upscale": 3.0, "max_crop_px": 2_500_000, "max_detail_per_roi": 400},
}

def _merge_rois(rois, w, h, max_total_area_frac):
    """Merge overlapping ROI boxes; enforce a total-area cap (drop lowest-score
    boxes once the cap is exceeded)."""
    def overlap(a, b):
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
    boxes = [dict(r) for r in rois]
    merged = True
    while merged:
        merged = False
        out = []
        for r in boxes:
            hit = None
            for o in out:
                if overlap(r["bbox"], o["bbox"]):
                    hit = o; break
            if hit:
                a, b = r["bbox"], hit["bbox"]
                hit["bbox"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                hit["score"] += r["score"]; merged = True
            else:
                out.append(r)
        boxes = out
    boxes.sort(key=lambda r: -r["score"])
    cap, total, kept = max_total_area_frac * w * h, 0.0, []
    for r in boxes:
        x0, y0, x1, y1 = r["bbox"]; area = (x1 - x0) * (y1 - y0)
        if total + area > cap and kept:
            continue
        total += area; kept.append(r)
    return kept

def _find_small_detail_rois(img, dp):
    """Find compact clusters of small dark strokes (general high-frequency
    detail). Black-hat → small connected components → dilate-merge into regions.
    Returns padded, merged, capped ROI boxes in img coords. No OCR/semantics."""
    import cv2, numpy as np
    h, w = img.shape
    g = cv2.GaussianBlur(img, (3, 3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dp["roi_bh_k"], dp["roi_bh_k"]))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)
    _, m = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    small = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if dp["min_comp_area"] <= a <= dp["max_comp_area"] and cw <= dp["max_comp_dim"] and ch <= dp["max_comp_dim"]:
            small[lab == i] = 255
    if not small.any():
        return []
    # merge nearby small components into regions
    d = dp["merge_dilate"]
    dil = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats((dil > 0).astype(np.uint8), 8)
    rois = []
    for i in range(1, n2):
        x0, y0 = stats2[i, cv2.CC_STAT_LEFT], stats2[i, cv2.CC_STAT_TOP]
        rw, rh = stats2[i, cv2.CC_STAT_WIDTH], stats2[i, cv2.CC_STAT_HEIGHT]
        x1, y1 = x0 + rw, y0 + rh
        ink = int((small[y0:y1, x0:x1] > 0).sum())
        if ink < dp["min_region_ink"]: continue
        if rw * rh > dp["max_roi_area_frac"] * w * h: continue       # too big → texture/structure
        if rw < dp["min_roi_dim"] and rh < dp["min_roi_dim"]: continue
        rois.append({"bbox": [int(x0), int(y0), int(x1), int(y1)], "score": ink, "reason": "dense_dark_components"})
    rois.sort(key=lambda r: -r["score"])
    rois = rois[:dp["max_rois"]]
    pad = dp["roi_pad"]
    for r in rois:
        x0, y0, x1, y1 = r["bbox"]
        r["bbox"] = [max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)]
    return _merge_rois(rois, w, h, dp["max_total_area_frac"])

def _hybrid_detail_roi_candidates(hires, scale, rois, dp):
    """Reprocess each detail ROI locally at higher resolution: crop from the
    high-res gray, upscale, local-contrast normalise, threshold (adaptive +
    black-hat), skeletonize, then map vectors back to img coords.
    `scale` = hires_dim / img_dim. Returns candidate dicts in img coords."""
    import cv2, numpy as np
    from skimage.morphology import skeletonize, remove_small_objects
    out = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for r in rois:
        x0, y0, x1, y1 = r["bbox"]                       # img coords
        hx0, hy0, hx1, hy1 = int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale)
        crop = hires[hy0:hy1, hx0:hx1]
        if crop.size == 0 or min(crop.shape) < 3: continue
        ch, cw = crop.shape
        up = dp["roi_upscale"]
        if ch * cw * up * up > dp["max_crop_px"]:
            up = max(1.0, (dp["max_crop_px"] / (ch * cw)) ** 0.5)
        crop_u = cv2.resize(crop, (max(1, int(cw * up)), max(1, int(ch * up))), interpolation=cv2.INTER_CUBIC)
        crop_u = clahe.apply(crop_u)
        cg = cv2.GaussianBlur(crop_u, (3, 3), 0)
        blk = dp["ad_block"] | 1
        m_ad = cv2.adaptiveThreshold(cg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, dp["ad_c"])
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dp["bh_k"], dp["bh_k"]))
        _, m_bh = cv2.threshold(cv2.morphologyEx(cg, cv2.MORPH_BLACKHAT, kk), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = remove_small_objects(((m_ad > 0) | (m_bh > 0)), min_size=int(dp["min_area"] * up))
        if not mask.any(): continue
        sk = skeletonize(mask)
        f = up * scale                                   # crop_u px → img px divisor
        roi_polys = _skeleton_to_polylines(sk, prune_len=dp["prune"])
        for poly in roi_polys[:dp["max_detail_per_roi"]]:
            arr = np.array(poly, float)
            arr[:, 0] = x0 + arr[:, 0] / f
            arr[:, 1] = y0 + arr[:, 1] / f
            out.append({"pts": arr, "source": "detail"})
    return out

def _hybrid_accept_detail(cands, img, gmag, agree, dp, dedup_tol):
    """Score and accept small-detail candidates. Scored by local contrast +
    darkness + cluster coherence (+survives-multi-threshold), NOT length. Dedup
    is relaxed so dense clusters survive; own caps prevent explosion."""
    import numpy as np
    from collections import defaultdict
    h, w = img.shape
    inv = 1.0 - img.astype(np.float32) / 255.0
    if not cands:
        return [], [], 0, False
    # cluster neighbour counts on a coarse grid
    cell = dp["cluster_radius"]
    grid = defaultdict(int); mids = []
    for c in cands:
        m = c["pts"][len(c["pts"]) // 2]; mids.append(m)
        grid[(int(m[0] // cell), int(m[1] // cell))] += 1
    for c, m in zip(cands, mids):
        gx, gy = int(m[0] // cell), int(m[1] // cell)
        neigh = sum(grid.get((gx + dx, gy + dy), 0) for dx in (-1, 0, 1) for dy in (-1, 0, 1)) - 1
        s = _resample_pts(c["pts"], 2.0)
        xs = np.clip(s[:, 0].astype(int), 0, w - 1); ys = np.clip(s[:, 1].astype(int), 0, h - 1)
        contrast = float(gmag[ys, xs].mean())
        darkness = float(inv[ys, xs].mean())
        survives = (float(agree[ys, xs].mean()) - 1.0) if agree is not None else 0.0   # +ve if seen by >1 threshold
        c["length"] = _poly_len_px(c["pts"]); c["neigh"] = neigh
        cluster = min(max(neigh, 0), 6) / 6.0
        iso_pen = 0.5 if (neigh == 0 and c["length"] < dp["detail_min_len_px"] * 2) else 0.0
        dense_pen = 0.4 if neigh > dp["dense_neigh"] else 0.0
        c["score"] = contrast + darkness + 0.3 * survives + 0.25 * cluster - iso_pen - dense_pen

    kept, rejected = [], []
    for c in cands:
        (kept if (c["score"] >= dp["score_floor"] and c["length"] >= dp["detail_min_len_px"]) else rejected).append(c)
    kept.sort(key=lambda c: -c["score"])
    overflow = len(kept) > dp["max_detail_candidates"]
    if overflow:
        rejected.extend(kept[dp["max_detail_candidates"]:]); kept = kept[:dp["max_detail_candidates"]]

    tol, ov = dedup_tol, dp["dedup_overlap"]
    occ, accepted = set(), []
    def cells(pts):
        s = _resample_pts(pts, max(1.0, tol * 0.6))
        return {(int(x // tol), int(y // tol)) for x, y in s}
    for c in kept:
        cs = cells(c["pts"])
        if not cs: continue
        if sum(1 for k in cs if k in occ) / len(cs) > ov:
            rejected.append(c)
        else:
            accepted.append(c); occ |= cs
            if len(accepted) >= dp["max_detail_accepted"]: break

    # cluster count metric: connected components of occupied coarse cells
    cellset = {(int(c["pts"][len(c["pts"]) // 2][0] // cell), int(c["pts"][len(c["pts"]) // 2][1] // cell)) for c in accepted}
    seen, clusters = set(), 0
    for cc in cellset:
        if cc in seen: continue
        clusters += 1; stack = [cc]
        while stack:
            x, y = stack.pop()
            if (x, y) in seen or (x, y) not in cellset: continue
            seen.add((x, y))
            stack += [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    return accepted, rejected, clusters, overflow

def _write_colored_svg(out_path, w, h, groups):
    """groups: list of (color, stroke_width, [polylines]). Debug visualisation."""
    import svgwrite
    dwg = svgwrite.Drawing(str(out_path), size=(f"{w}px", f"{h}px"), viewBox=f"0 0 {w} {h}")
    for color, sw, polys in groups:
        g = dwg.g(stroke=color, stroke_width=sw, fill="none")
        for pts in polys:
            if len(pts) >= 2:
                g.add(dwg.polyline([(round(float(x), 2), round(float(y), 2)) for x, y in pts]))
        dwg.add(g)
    dwg.save()
    return out_path

def _hybrid_svg(img, out_path: Path, opts: dict, debug_prefix: str = None, raw_path: Path = None) -> Path:
    import cv2, numpy as np, time, json
    t0 = time.time()
    preset = str(opts.get("hybrid_preset", "balanced")).lower()
    P = dict(HYBRID_PRESETS.get(preset, HYBRID_PRESETS["balanced"]))

    sd = str(opts.get("small_detail", "off")).lower()

    # ── MAJOR STRUCTURE LANE (length-weighted; unaffected by small-detail mode) ──
    gmag = _grad_mag(img)
    geom = _hybrid_geom_candidates(img, P["geom_min_px"])

    sm = cv2.bilateralFilter(img, 7, 60, 60)
    hi, _ = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi = max(40.0, float(hi))
    variants = [(0.5 * hi, hi), (0.33 * hi, 0.7 * hi), (0.7 * hi, 1.3 * hi)]
    edges = _hybrid_edge_candidates(img, variants[:P["edge_variants"]])
    ink = _hybrid_ink_candidates(img, P["ink_modes"], opts)

    def _wf(v):
        try: return float(v)
        except (TypeError, ValueError): return 1.0
    if str(opts.get("geometry_weight", "auto")) == "auto":
        gw, ew, iw = _hybrid_auto_weights(geom, edges, ink)
    else:
        gw, ew, iw = _wf(opts.get("geometry_weight")), _wf(opts.get("edge_weight")), _wf(opts.get("ink_weight"))
    gw *= P["geom_bias"]; ew *= P["edge_bias"]; iw *= P["ink_bias"]
    P["weights"] = {"geom": gw, "edge": ew, "ink": iw}

    cands = geom + edges + ink
    if len(cands) > P["max_candidates"] * 3:        # hard pre-cap by raw length
        cands.sort(key=lambda c: -_poly_len_px(c["pts"]))
        cands = cands[:P["max_candidates"] * 3]
    accepted, rejected, diag = _hybrid_fuse(cands, gmag, P)
    acc_lines = [c["pts"] for c in accepted]
    stitched = _hybrid_stitch(acc_lines, P["stitch_gap_px"]) if len(acc_lines) <= 3000 else acc_lines

    # ── SMALL-DETAIL LANE: local high-res ROI pass (separate, own caps) ──
    # Find compact small-stroke regions, reprocess each crop locally at higher
    # resolution (from the raw image when available), score by contrast/cluster
    # (not length), and merge as a separate layer. NOT subjected to the major
    # occupancy dedup, so a detail cluster is never dropped for overlapping a
    # long nearby line. Major outlines are untouched.
    det_acc, det_rej, det_clusters, det_overflow, det_all, rois = [], [], 0, False, [], []
    reason_if_empty = "disabled"
    sd_block = {"mode": sd, "roi_count": 0, "roi_area_pct": 0, "candidates": 0, "accepted": 0,
                "rejected": 0, "strokes_added": 0, "estimated_z_moves_added": 0,
                "median_detail_length_px": 0, "min_detail_length_px": 0, "overflow": False,
                "reason_if_empty": "disabled"}
    if sd in HYBRID_DETAIL_PARAMS:
        dp = HYBRID_DETAIL_PARAMS[sd]
        rois = _find_small_detail_rois(img, dp)
        # Debug option force_detail_roi: when detection finds no ROI, force ONE ROI
        # around the strongest local-detail region so the detail lane can be
        # exercised end-to-end.
        if opts.get("force_detail_roi") and not rois:
            h, w = img.shape
            bh = cv2.morphologyEx(cv2.GaussianBlur(img, (3, 3), 0), cv2.MORPH_BLACKHAT,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dp["roi_bh_k"], dp["roi_bh_k"])))
            _, _, _, maxloc = cv2.minMaxLoc(cv2.GaussianBlur(bh.astype(np.float32), (0, 0), 9))
            cx, cy = maxloc; s = max(40, min(h, w) // 6)
            rois = [{"bbox": [max(0, int(cx - s)), max(0, int(cy - s)), min(w, int(cx + s)), min(h, int(cy + s))],
                     "score": 1, "reason": "forced"}]
            print(f"  [detail] force_detail_roi -> 1 forced ROI at ({cx},{cy})")
        if rois and raw_path is not None:
            hires = _load_gray(raw_path, dp["hires_max"], opts.get("clahe", False),
                               opts.get("channel", "auto"), opts.get("crop"),
                               "blue" if opts.get("isolate_blue") else None)
            scale = hires.shape[1] / float(img.shape[1])
            det_all = _hybrid_detail_roi_candidates(hires, scale, rois, dp)
        det_acc, det_rej, det_clusters, det_overflow = _hybrid_accept_detail(det_all, img, gmag, None, dp, P["dedup_tol"])
        det_lens = sorted(c["length"] for c in det_acc)
        roi_area = sum((r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]) for r in rois)
        # Why is the detail layer empty (if it is)? Drives the UI message.
        if not rois:
            reason_if_empty = "no_rois"
        elif not det_acc:
            reason_if_empty = "all_rejected"
        else:
            reason_if_empty = ""
        sd_block = {
            "mode": sd, "roi_count": len(rois),
            "roi_area_pct": round(100.0 * roi_area / float(img.shape[0] * img.shape[1]), 1),
            "candidates": len(det_all), "accepted": len(det_acc), "rejected": len(det_rej),
            "clusters": det_clusters, "strokes_added": len(det_acc),
            "estimated_z_moves_added": 2 * len(det_acc),
            "median_detail_length_px": round(float(np.median(det_lens)), 1) if det_lens else 0,
            "min_detail_length_px": round(det_lens[0], 1) if det_lens else 0,
            "overflow": det_overflow, "reason_if_empty": reason_if_empty,
        }
    print(f"  [detail] mode={sd} rois={len(rois)} candidates={len(det_all)} "
          f"accepted={len(det_acc)} rejected={len(det_rej)} reason={reason_if_empty or 'ok'}")

    # Final geometry = stitched major outlines + accepted detail strokes (kept raw).
    polys = [[(float(x), float(y)) for x, y in l] for l in stitched]
    detail_polys = [[(float(x), float(y)) for x, y in c["pts"]] for c in det_acc]
    _write_paths_svg(out_path, img.shape[1], img.shape[0], _stroke_px_for(opts, img.shape[1]), polylines=polys + detail_polys)

    acc_lens = sorted(c["length"] for c in accepted)
    def _stat(fn, default=0.0):
        return round(float(fn(acc_lens)), 1) if acc_lens else default
    metrics = {
        "preset": preset, "small_detail": sd_block,
        "candidate_counts": {"geom": len(geom), "edge": len(edges), "ink": len(ink),
                             "detail": len(det_all), "total": len(geom)+len(edges)+len(ink)+len(det_all)},
        "accepted_counts": dict(diag["accepted_by_source"], detail=len(det_acc), total=len(accepted)+len(det_acc)),
        "rejected_counts": dict(diag["rejected_by_source"], total=len(rejected), detail=len(det_rej), **diag["reasons"]),
        "line_length_stats_px": {
            "min": _stat(min), "median": _stat(np.median), "mean": _stat(np.mean), "max": _stat(max),
        },
        "final_strokes_pre_vpype": len(stitched) + len(det_acc),
        "weights": {"geom": round(gw, 2), "edge": round(ew, 2), "ink": round(iw, 2)},
        "objective_score": round(_hybrid_objective(stitched, gmag), 1),
        # flat keys read by the footer summary
        "geometry_candidates": len(geom), "edge_candidates": len(edges), "ink_candidates": len(ink),
        "accepted_lines": len(accepted) + len(det_acc), "rejected_lines": len(rejected) + len(det_rej),
        "duplicates_removed": diag["reasons"]["duplicate"],
    }
    try:
        with open(WORK_DIR / "debug_hybrid_stats.json", "w") as f:
            json.dump(metrics, f)
    except Exception: pass

    if debug_prefix:
        W, H = img.shape[1], img.shape[0]
        by = lambda src, pool: [[(float(x), float(y)) for x, y in c["pts"]] for c in pool if c["source"] == src]
        allc = geom + edges + ink
        # Very dark, high-contrast colors so debug lines read clearly on the
        # light-grey inspector background.
        C_GEOM, C_EDGE, C_INK, C_REJ, C_FUSED, C_DET = "#172554", "#14532d", "#7c2d12", "#7f1d1d", "#3b0764", "#831843"
        _write_colored_svg(f"{debug_prefix}_hybrid_geom.svg", W, H, [(C_GEOM, 1.4, by("geom", allc))])
        _write_colored_svg(f"{debug_prefix}_hybrid_edges.svg", W, H, [(C_EDGE, 1.4, by("edge", allc))])
        _write_colored_svg(f"{debug_prefix}_hybrid_ink.svg", W, H, [(C_INK, 1.4, by("ink", allc))])
        _write_colored_svg(f"{debug_prefix}_hybrid_candidates.svg", W, H,
                           [(C_GEOM, 1.2, by("geom", allc)), (C_EDGE, 1.2, by("edge", allc)), (C_INK, 1.2, by("ink", allc))])
        _write_colored_svg(f"{debug_prefix}_hybrid_rejected.svg", W, H,
                           [(C_REJ, 1.2, [[(float(x), float(y)) for x, y in c["pts"]] for c in rejected])])
        _write_colored_svg(f"{debug_prefix}_hybrid_fused.svg", W, H, [(C_FUSED, 1.4, polys), (C_DET, 1.4, detail_polys)])
        # ── Small-detail ROI lane debug layers ──
        # Write each layer ONLY when it has content, so the inspector never shows
        # a blank card (a missing file → an explicit message instead).
        _det = lambda pool: [[(float(x), float(y)) for x, y in c["pts"]] for c in pool]
        if rois:
            roi_rects = [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
                         for (x0, y0, x1, y1) in (r["bbox"] for r in rois)]
            _write_colored_svg(f"{debug_prefix}_detail_roi_boxes.svg", W, H, [("#0891b2", 1.6, roi_rects)])  # cyan boxes
            try:
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for (x0, y0, x1, y1) in (r["bbox"] for r in rois):
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (200, 145, 8), 2)   # cyan-ish (BGR)
                cv2.imwrite(f"{debug_prefix}_detail_rois.png", vis)
            except Exception: pass
        if det_all:
            _write_colored_svg(f"{debug_prefix}_detail_candidates.svg", W, H, [(C_DET, 1.2, _det(det_all))])
        if detail_polys:
            _write_colored_svg(f"{debug_prefix}_detail_accepted.svg", W, H, [(C_DET, 1.4, detail_polys)])
            _write_colored_svg(f"{debug_prefix}_detail_merged.svg", W, H, [(C_FUSED, 1.2, polys), (C_DET, 1.4, detail_polys)])
        if det_rej:
            _write_colored_svg(f"{debug_prefix}_detail_rejected.svg", W, H, [("#7f1d1d", 1.0, _det(det_rej))])

    print(f"  [hybrid] preset={preset} sd={sd} geom={len(geom)} edge={len(edges)} ink={len(ink)} "
          f"-> major={len(stitched)} detail={len(det_acc)}/{len(det_all)} (clusters={det_clusters}) "
          f"w=({gw:.2f},{ew:.2f},{iw:.2f}) score={metrics['objective_score']} {time.time()-t0:.2f}s")
    return out_path

