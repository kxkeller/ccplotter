"""Opt-in mask-checked stitching of short open strokes (`stitch_mm`).

Why: locally fragmented open-stroke ink (short pieces, dense lifts and turnarounds) is where
dimples in the plotted line cluster. Joining fragments into longer continuous runs removes
those lifts. Everything here is OFF unless the request carries `stitch_mm > 0`; trace_v3 only
reaches this module under that flag, so with the key absent the output is unchanged.

Rule
----
Every OPEN stroke (first and last point > CLOSED_TOL_PX apart; closed rings have no ends and
are never touched) exposes two endpoints. A pair of endpoints of DIFFERENT strokes is a
join candidate when
  * they are within `stitch_px` of each other, and
  * the straight pen-down connector between them lies entirely inside the ink mask ERODED by
    half a fill pitch: every sample (0.5 px apart, bilinear) has dt >= erode_px, where dt is
    the same Euclidean distance transform the level-set rings are cut from and
    erode_px = 0.5 * fill_px -- the outermost ring sits on that very contour (dt > 0.5 *
    fill_px), so a connector never reaches closer to the source edge than the first ring
    does, and never inks paper the source leaves white. Connectors are STRAIGHT; there is
    no routed / geodesic fallback.
  * joining does not close a loop (union-find; a stroke is never joined to itself).
Coincident endpoints (< ZERO_LEN_PX apart) join with no connector at all -- they add no ink.
Candidates are taken greedily, shortest first, in TWO classes: first every join whose
junction turn is below HAIRPIN_DEG, then (only for endpoints still free) the hairpins. A
junction's turn is the largest of: heading-in -> connector, connector -> heading-out (or
heading-in -> heading-out when the connector has no length). A hairpin is a turn > 135 deg,
i.e. the pen would nearly reverse on itself. Hairpin joins are permitted (they still trade a
lift for a turnaround) but are deferred, counted and reported.

Thin features: a stroke narrower than one fill pitch has no eroded interior (its centreline
has dt < erode_px), so a connector from it cannot pass the strict test. That is the point of
the test (a connector off a hairline would ink a full pen width of paper the source leaves
white), but it is surfaced, not hidden: `rejected_offmask_thin_endpoint` counts refused
candidates whose own endpoint is already below the threshold.

The pass runs twice when core_fill=hybrid: phase "core" over the core rows alone (so cores
are linked into runs BEFORE they meet anything else), then phase "all" over every open
stroke including the already-linked core runs.
"""
import math

import numpy as np

HAIRPIN_DEG = 135.0
CLOSED_TOL_PX = 0.6      # same predicate as trace_v3._is_speck
SAMPLE_STEP_PX = 0.5
ZERO_LEN_PX = 0.05
_CHUNK = 20000           # candidate pairs sampled per vectorised block


def placed_mm_per_px(opts, mask, fallback_mm_per_px):
    """True mm per working px of the PLACED drawing, same arithmetic as the hybrid / inset
    branch of trace_v3._trace_svg (composite's final fit from `_place_cell` and this trace's
    own mask bbox). Returns (mm_per_px, source). Falls back to trace_v3's own
    fill_mm/fill_px back-solve (labelled; off by 1/0.75 or worse for pens whose laid width differs
    from fill_mm)
    when the caller supplied no placement."""
    pc = opts.get("_place_cell")
    if pc:
        ys, xs = np.nonzero(mask)
        if len(xs):
            from ccplotter.config import PX2MM
            from ccplotter.composite import _rotated_extent_mm
            bw = float(xs.max() - xs.min() + 1) * PX2MM
            bh = float(ys.max() - ys.min() + 1) * PX2MM
            cw, ch, rot, scl, true_size = pc
            rw, rh = _rotated_extent_mm(bw, bh, rot)
            fit = 1.0 if true_size else min(cw / rw, ch / rh) * 0.9
            return float(fit) * float(scl) * PX2MM, "placed_fit_from_mask_bbox"
    return float(fallback_mm_per_px), "backsolved_fill_mm_over_fill_px"


def _unit_out(P, tail, look):
    """Outward unit heading at the head (tail=False) or tail (tail=True) of polyline P: the
    vector from the first point at least `look` px away (else the far end) to the endpoint."""
    n = len(P)
    if tail:
        e = P[-1]
        rng = range(n - 2, -1, -1)
    else:
        e = P[0]
        rng = range(1, n)
    ref = P[0] if tail else P[-1]
    for k in rng:
        if math.hypot(P[k][0] - e[0], P[k][1] - e[1]) >= look:
            ref = P[k]
            break
    v = np.array([e[0] - ref[0], e[1] - ref[1]], float)
    nv = float(np.hypot(*v))
    return v / nv if nv > 1e-12 else np.zeros(2)


def _angle_deg(a, b):
    """Row-wise angle between vector arrays a, b (M,2); a zero vector counts as 0 deg."""
    na = np.hypot(a[:, 0], a[:, 1])
    nb = np.hypot(b[:, 0], b[:, 1])
    den = na * nb
    cos = np.where(den > 1e-12, (a * b).sum(1) / np.where(den > 1e-12, den, 1.0), 1.0)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _dt_at(dt, xy):
    from scipy.ndimage import map_coordinates
    return map_coordinates(dt, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")


def _connector_in_mask(dt, A, B, thr, step=SAMPLE_STEP_PX):
    """Vectorised: bool per pair, True iff every sample on the segment A[i]->B[i]
    (both ends included, `step` px apart) has bilinear dt >= thr."""
    d = np.hypot(*(B - A).T)
    ns = np.maximum(2, np.ceil(d / step).astype(np.int64) + 1)
    starts = np.cumsum(ns) - ns
    rep = np.repeat(np.arange(len(d)), ns)
    off = np.arange(int(ns.sum())) - np.repeat(starts, ns)
    t = off / np.repeat(ns - 1, ns)
    pts = A[rep] + (B[rep] - A[rep]) * t[:, None]
    ok = _dt_at(dt, pts) >= thr - 1e-6
    return np.logical_and.reduceat(ok, starts)


def stitch_open_strokes(pieces, dt, stitch_px, erode_px, tangent_px, hairpin_deg=HAIRPIN_DEG):
    """Greedy mask-checked joining of open strokes. `pieces` = list of [(x, y), ...];
    `dt` = distance transform of the ink mask (px). Returns
    (new_pieces, members, connectors, rec):
      new_pieces  list of polylines; strokes that were not joined are the ORIGINAL objects
      members     per new piece, the original indices merged into it
      connectors  list of (x0, y0, x1, y1, len_px, turn_deg) actually added (length > 0)
      rec         counters (see keys)"""
    from scipy.spatial import cKDTree
    n = len(pieces)
    arrs = [np.asarray(p, float) for p in pieces]
    open_ids = [i for i, a in enumerate(arrs)
                if len(a) >= 2 and math.hypot(a[0][0] - a[-1][0], a[0][1] - a[-1][1]) > CLOSED_TOL_PX]
    m = len(open_ids)
    rec = {"n_pieces": n, "n_open": m, "n_closed_or_degenerate": n - m,
           "n_candidate_pairs": 0, "n_same_piece_pairs": 0, "n_joins": 0, "n_joins_zero_len": 0,
           "connector_len_px": 0.0, "connector_max_px": 0.0,
           "rejected_offmask": 0, "rejected_offmask_thin_endpoint": 0, "rejected_cycle": 0,
           "hairpin_joins": 0, "n_after": n}
    if m < 2:
        return list(pieces), [[i] for i in range(n)], [], rec
    E = np.empty((2 * m, 2))
    H = np.zeros((2 * m, 2))
    for k, i in enumerate(open_ids):
        a = arrs[i]
        E[2 * k], E[2 * k + 1] = a[0], a[-1]
        H[2 * k] = _unit_out(a, False, tangent_px)
        H[2 * k + 1] = _unit_out(a, True, tangent_px)
    tree = cKDTree(E)
    pr = tree.query_pairs(float(stitch_px) + 1e-9, output_type="ndarray")
    if len(pr):
        same = (pr[:, 0] // 2) == (pr[:, 1] // 2)
        rec["n_same_piece_pairs"] = int(same.sum())
        pr = pr[~same]
    rec["n_candidate_pairs"] = int(len(pr))
    if not len(pr):
        return list(pieces), [[i] for i in range(n)], [], rec
    pa, pb = pr[:, 0], pr[:, 1]
    A, B = E[pa], E[pb]
    C = B - A
    d = np.hypot(C[:, 0], C[:, 1])
    ha, hb = H[pa], H[pb]
    zero = d <= ZERO_LEN_PX
    t1 = _angle_deg(ha, C)
    t2 = _angle_deg(C, -hb)
    tz = _angle_deg(ha, -hb)
    turn = np.where(zero, tz, np.maximum(t1, t2))
    hair = turn > hairpin_deg
    inmask = np.ones(len(pr), bool)
    nz = np.nonzero(~zero)[0]
    for s in range(0, len(nz), _CHUNK):
        idx = nz[s:s + _CHUNK]
        inmask[idx] = _connector_in_mask(dt, A[idx], B[idx], erode_px)
    thin = (_dt_at(dt, A) < erode_px - 1e-6) | (_dt_at(dt, B) < erode_px - 1e-6)
    order = np.lexsort((pb, pa, d, hair.astype(np.int8)))   # class, then length, then ids

    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used = np.zeros(2 * m, bool)
    nbr = np.full(2 * m, -1, np.int64)
    connectors = []
    for c in order:
        a, b = int(pa[c]), int(pb[c])
        if used[a] or used[b]:
            continue
        if not inmask[c]:
            rec["rejected_offmask"] += 1
            if thin[c]:
                rec["rejected_offmask_thin_endpoint"] += 1
            continue
        ra, rb = find(a // 2), find(b // 2)
        if ra == rb:
            rec["rejected_cycle"] += 1
            continue
        parent[ra] = rb
        used[a] = used[b] = True
        nbr[a], nbr[b] = b, a
        rec["n_joins"] += 1
        if hair[c]:
            rec["hairpin_joins"] += 1
        if zero[c]:
            rec["n_joins_zero_len"] += 1
        else:
            L = float(d[c])
            rec["connector_len_px"] += L
            rec["connector_max_px"] = max(rec["connector_max_px"], L)
            connectors.append((float(A[c][0]), float(A[c][1]), float(B[c][0]), float(B[c][1]),
                               L, float(turn[c])))
    if rec["n_joins"] == 0:
        return list(pieces), [[i] for i in range(n)], connectors, rec

    k_of = {i: k for k, i in enumerate(open_ids)}
    done = np.zeros(m, bool)
    new_pieces, members = [], []
    for i in range(n):
        k = k_of.get(i)
        if k is None:
            new_pieces.append(pieces[i]); members.append([i])
            continue
        if done[k]:
            continue
        if nbr[2 * k] < 0 and nbr[2 * k + 1] < 0:
            done[k] = True
            new_pieces.append(pieces[i]); members.append([i])
            continue
        cur, out_e = k, 0                      # walk out through the head until a free end
        while nbr[2 * cur + out_e] >= 0:
            p = int(nbr[2 * cur + out_e])
            cur, out_e = p // 2, 1 - (p % 2)
        ent = out_e                            # chain starts at that free end
        chain, mem = [], []
        while True:
            done[cur] = True
            a = arrs[open_ids[cur]]
            seq = a if ent == 0 else a[::-1]
            pts = [(float(x), float(y)) for x, y in seq]
            if chain and abs(chain[-1][0] - pts[0][0]) < 1e-9 and abs(chain[-1][1] - pts[0][1]) < 1e-9:
                pts = pts[1:]
            chain.extend(pts)
            mem.append(open_ids[cur])
            p = int(nbr[2 * cur + (1 - ent)])
            if p < 0:
                break
            cur, ent = p // 2, p % 2
        new_pieces.append(chain); members.append(mem)
    rec["n_after"] = len(new_pieces)
    return new_pieces, members, connectors, rec


def _raster_off_pct(polys, mask, thick):
    """% of the footprint pixels (polylines drawn at `thick` px, LINE_8) outside `mask`."""
    import cv2
    H, W = mask.shape[:2]
    canvas = np.zeros((H, W), np.uint8)
    for p in polys:
        if len(p) < 2:
            continue
        cv2.polylines(canvas, [np.rint(np.asarray(p, np.float32)).astype(np.int32)], False, 255,
                      thick, lineType=cv2.LINE_8)
    ink = canvas > 0
    tot = int(np.count_nonzero(ink))
    off = int(np.count_nonzero(ink & ~mask.astype(bool)))
    return tot, off


def stitch_passes(passes, n_core, mask, dt, opts, fill_px, fallback_mm_per_px, debug_prefix=None):
    """Orchestrator called by trace_v3._trace_svg (only when stitch_mm > 0).
    `passes` is the final stroke list; its LAST `n_core` entries are the hybrid core rows.
    Returns (new_passes, members, record). `members[j]` = indices into the INPUT list merged
    into output j (for the provenance sidecar)."""
    import json
    stitch_mm = float(opts.get("stitch_mm") or 0.0)
    mm_px, src = placed_mm_per_px(opts, mask, fallback_mm_per_px)
    stitch_px = stitch_mm / mm_px
    erode_px = 0.5 * float(fill_px)
    tangent_px = max(1.0, float(fill_px))
    n_in = len(passes)
    rec = {"stitch_mm": stitch_mm, "stitch_px": round(stitch_px, 3), "mm_per_px": round(mm_px, 6),
           "mm_per_px_source": src, "erode_px": round(erode_px, 3),
           "erode_mm": round(erode_px * mm_px, 4), "hairpin_deg": HAIRPIN_DEG,
           "lifts_before": n_in}
    cur = list(passes)
    cur_members = [[i] for i in range(n_in)]
    all_conn = []
    phases = {}
    if n_core:
        head, core = cur[:n_in - n_core], cur[n_in - n_core:]
        new_core, mem_core, conn, prec = stitch_open_strokes(core, dt, stitch_px, erode_px, tangent_px)
        phases["core"] = prec
        all_conn += [(*c, "core") for c in conn]
        base = n_in - n_core
        cur = head + new_core
        cur_members = cur_members[:base] + [[base + j for j in m] for m in mem_core]
    new, mem, conn, prec = stitch_open_strokes(cur, dt, stitch_px, erode_px, tangent_px)
    phases["all"] = prec
    all_conn += [(*c, "all") for c in conn]
    # members relative to the INPUT list
    final_members = [[orig for j in m for orig in cur_members[j]] for m in mem]
    n_conn = len(all_conn)
    len_px = sum(c[4] for c in all_conn)
    rec.update({
        "lifts_after": len(new),
        "connectors_added": n_conn,
        "connector_len_mm": round(len_px * mm_px, 3),
        "connector_max_mm": round(max([c[4] for c in all_conn], default=0.0) * mm_px, 4),
        "joins_zero_len": sum(p["n_joins_zero_len"] for p in phases.values()),
        "rejected_offmask": sum(p["rejected_offmask"] for p in phases.values()),
        "rejected_offmask_thin_endpoint": sum(p["rejected_offmask_thin_endpoint"] for p in phases.values()),
        "rejected_cycle": sum(p["rejected_cycle"] for p in phases.values()),
        "hairpin_joins": sum(p["hairpin_joins"] for p in phases.values()),
        "phases": {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                   for k, v in phases.items()},
    })
    # independent raster audit at pen width (NOT the dt predicate): connector footprint outside the
    # mask, against the same measure for the ORIGINAL strokes (pixel quantisation sets a floor).
    thick = max(1, int(round(float(fill_px))))
    mask_b = np.asarray(mask).astype(bool)
    ctot, coff = _raster_off_pct([[(c[0], c[1]), (c[2], c[3])] for c in all_conn], mask_b, thick)
    btot, boff = _raster_off_pct(passes, mask_b, thick)
    # edge-hugging baseline: only the original segments whose midpoint sits within 1.5 * erode_px of
    # the boundary (rings / outer rows) -- a connector can only live there, so compare like with like.
    edge_segs = []
    for p in passes:
        a = np.asarray(p, float)
        if len(a) < 2:
            continue
        mid = (a[:-1] + a[1:]) / 2.0
        near = _dt_at(dt, mid) <= 1.5 * erode_px
        for i in np.nonzero(near)[0]:
            edge_segs.append([tuple(a[i]), tuple(a[i + 1])])
    etot, eoff = _raster_off_pct(edge_segs, mask_b, thick)
    rec["audit"] = {"raster_thickness_px": thick,
                    "connector_footprint_px": ctot, "connector_offmask_px": coff,
                    "connector_offmask_pct": round(100.0 * coff / max(1, ctot), 4),
                    "original_strokes_offmask_pct": round(100.0 * boff / max(1, btot), 4),
                    "original_edge_segments_offmask_pct": round(100.0 * eoff / max(1, etot), 4)}
    if debug_prefix:
        try:
            full = dict(rec)
            full["connectors_px"] = [[round(v, 2) for v in c[:6]] + [c[6]] for c in all_conn]
            with open(f"{debug_prefix}_stitch.json", "w") as f:
                json.dump(full, f)
        except Exception as e:      # diagnostic only, never fatal
            print(f"    [trace] stitch: debug record write failed ({e})")
    return new, final_members, rec
