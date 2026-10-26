"""ccplotter/twopen_hatch.py -- OPT-IN "Hatched blue (lighter, fewer pen lifts)" for the two-pen key+tint job (body flag hatched_blue, default OFF).

The blue layer is NOT filled solid: it is long 45-degree hatch lines whose density follows a pen-scale blue target, so the blue is lighter (about half the
blue coverage of the solid fill), looks different (paper shows between lines) and needs far fewer pen lifts. The black layer is the
blue_trap black (specks reassigned to blue, never deleted); see generate_layers.

THE MAPPING (parameter-free, every constant below is a multiple of the BLUE pen's fill_mm W or a raster-grid constant; nothing is fitted
per image; there is no per-image threshold bisection):
  b3      3x3-blurred grey (the same grey separate() thresholds)
  b3s     b3 smoothed by a Gaussian of sigma 1 W                       (SIGMA_PEN, the pen-scale target)
  blue_all  b3s < blue_threshold (the request's blue_amount threshold), then the dust rule _merge_small against the unsmoothed black
  c       clip((blue_threshold - b3s) / (blue_threshold - T), 0, 1), c = 1 on the (unfolded) black mask   -> ink density wanted, 0..1
  U       (b3s < blue_threshold) | black | blue_all                     -> where hatch may be drawn (blue u black)
  lines   a 45-degree family, pitch W, offset v=(k+0.5)W; line k is drawn at a place iff ceil(8c) > bitrev8(k mod 8)
          (an 8-level ordered-dither over line index: c=1 -> every line, c=1/8 -> every eighth line)
  clip    hatch is only placed where the distance to the outside of U is >= CLIP = 0.5 W + 1 raster px (0.11 mm at W 0.12)
  runs    maximal pieces of a line, kept iff longer than MIN_RUN = W
  link    explicit switchback linker: from the exit point choose among unused run endpoints within LINK_RADIUS = 8 W (the largest pitch the
          dither can produce), the first candidate (forward line, same line, backward line; then nearer) whose straight connector passes
          the app's stage_e_mask predicate against the final trace mask of U; at most LINK_CANDIDATES = 40 are tested per exit (a COST guard,
          not a fitted value); no candidate -> pen lift. Chains are then ordered by the app's nearest-endpoint walk (_nn_walk_order).
"""
import math
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- constants (fixed, documented, derived)
SIGMA_PEN = 1.0              # Gaussian sigma of the blue target, in blue pen widths W (sigma 0 would reproduce the traced blue)
LEVELS = 8                   # ordered-dither levels over line index (bit reversal of k mod 8) -> sparsest drawn line pitch = 8 W
PITCH_PEN = 1.0              # densest line pitch = 1 W (full coverage at pen width)
ANGLE_DEG = 45.0             # hatch direction
CLIP_PEN = 0.5               # clip = 0.5 W + 1 raster px: pen centre must be half a pen width + 1 px inside U, so ink stays inside U
MIN_RUN_PEN = 1.0            # runs shorter than 1 W are dropped (a dot, not a line)
LINK_RADIUS_PEN = 8.0        # switchback link radius = 8 W = the largest pitch the 8-level dither produces
LINK_CANDIDATES = 40         # COST GUARD: candidates mask-tested per exit point. Not fitted to quality; caps CPU on dense areas.
HATCH_PPM = 20.0             # hatch raster grid, px per mm (0.05 mm/px): coverage / clip / U are evaluated on this grid in page space
RUN_STEP_MM = 0.025          # sample step along a line when finding runs (half a raster px)
WALK_CAP_UNUSED = 1_000_000  # walk cap of the superseded router-linked variant; the explicit linker has no cap. Recorded, not live.

CONSTANTS_HATCHED = {
    "option": "body.hatched_blue (bool, default false)", "label": "Hatched blue (lighter, fewer pen lifts)",
    "sigma_pen_widths": SIGMA_PEN, "dither_levels": LEVELS, "line_pitch_pen_widths": [PITCH_PEN, float(LEVELS)], "angle_deg": ANGLE_DEG,
    "clip_mm_rule": "0.5 W + 1 raster px (0.11 mm at W 0.12)", "clip_pen_widths": CLIP_PEN, "min_run_pen_widths": MIN_RUN_PEN,
    "link_radius_pen_widths": LINK_RADIUS_PEN, "link_preference": "forward line, same line, backward line; then nearer",
    "link_candidates_cost_guard": LINK_CANDIDATES, "hatch_ppm": HATCH_PPM, "run_step_mm": RUN_STEP_MM,
    "walk_cap_not_used": "1,000,000 applied to the superseded router-linked arm B3 only; B3s uses the explicit linker",
    "W": "the BLUE profile's fill_mm (like blue_trap), fallback the plain job's fill_mm",
    "black": "blue_trap F black (specks < pi*(1.5 W)^2 reassigned to blue); implied by this option",
    "precedence": "hatched_blue replaces the blue layer entirely (blue_trap's T1 blue mask and 0.5 mm blue connector are not used); "
                  "black = F black; stay_down_over_ink applies to the black layer (own mask); the hatch linker is already pen-down only inside blue u black",
    "excluded": "per-image threshold bisection (B1c/B3c) and the H1/H1v gain arms are never part of this option (D1736, D1762, D1763)",
    "prototype_measurements": {
        "note": "offline prototype numbers (artifacts/twopen-blue-redesign-2026-10-21/, twopen-hatchblue-confirm-2026-10-21/); the job does NOT compute them. Models unvalidated, nothing plotted.",
        "lifts_tram_R_to_B3s": [37345, 20550], "lifts_heli_R_to_B3s": [10401, 5247],
        "model_c_hours_tram_R_to_B3s": [9.82, 7.47], "model_c_hours_heli_R_to_B3s": [3.22, 2.14],
        "uncovered_by_either_pct_tram_R_to_B3s": [1.36, 0.18], "uncovered_by_either_pct_heli_R_to_B3s": [0.07, 0.27],
        "uncovered_by_black_pct_tram_heli_unchanged": [1.55, 2.33],
        "blue_coverage_of_target_tram_heli_R_vs_B3s": [[0.92, 0.97], [0.55, 0.51]],
        "tone_vs_R_Lstar_B3s_mid_bands": {"tram": "+2.2..+2.6", "heli": "+5..+8"}},
    "d_numbers": "D1736-D1743, D1761-D1768", "artifacts": "artifacts/twopen-blue-redesign-2026-10-21/ ; artifacts/twopen-hatchblue-confirm-2026-10-21/",
}


def _bitrev(L):
    bits = int(round(math.log2(L)))
    return np.array([int(format(i, f"0{bits}b")[::-1], 2) for i in range(L)])


def _warp_matrix(geom, ppm):
    """grid px -> hatch-raster px (page mm * ppm), the same transform the plain job applies (rig.geometry 'Mw')."""
    from ccplotter.config import PX2MM
    fit, tx, ty, rot = geom["fit"], geom["tx"], geom["ty"], geom.get("rot", 0.0)
    c0 = np.array(geom["c0_px"], float)
    th = math.radians(rot)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t = np.array([tx, ty])
    Mw = np.zeros((2, 3))
    Mw[:, :2] = R * (fit * PX2MM) * ppm
    Mw[:, 2] = (PX2MM * c0 + t - R @ (fit * PX2MM * c0)) * ppm
    return Mw


def target_fields(sep, gray, opts, mm_per_px, pen_mm):
    """sep = separate() result. Returns (c, U, blue_all): coverage target (float32, working grid), hatch region, blue_all mask."""
    from ccplotter.twopen_keytint import _disc, _merge_small
    T, t_blue, r = float(sep["T"]), float(sep["blue_threshold"]), int(sep["radius_px"])
    b3 = cv2.GaussianBlur(gray, (3, 3), 0)
    BLACK = sep["core"] | sep["E"]
    bd = cv2.dilate(BLACK.astype(np.uint8), _disc(r)) > 0
    blk = BLACK | (sep["M0"] & bd)
    b3s = cv2.GaussianBlur(b3.astype(np.float32), (0, 0), SIGMA_PEN * pen_mm / mm_per_px)
    blue_all = b3s < t_blue
    _black2, blue2, _ = _merge_small(blk, blue_all & ~blk, sep["info"]["min_mark_px"])
    blackA = sep["black"]
    b3x = b3s.astype(np.float32)
    c = np.clip((t_blue - b3x) / (t_blue - T), 0.0, 1.0).astype(np.float32)
    c[blackA] = 1.0
    U = (b3x < t_blue) | blackA | blue2
    return c, U, blue2


def gen_hatch(c, U, geom, paper_w, paper_h, pen_mm, clip_mm=None):
    """Hatch runs in page mm: list of (2,2) arrays [[x0,y0],[x1,y1]]. Also returns (dt, Up) for the containment check."""
    from scipy import ndimage as ndi
    W = float(pen_mm)
    ppm = HATCH_PPM
    HP, WP = int(paper_h * ppm), int(paper_w * ppm)
    Mw = _warp_matrix(geom, ppm)
    warp = lambda a: cv2.warpAffine(a, Mw, (WP, HP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    clip = (CLIP_PEN * W + 1.0 / ppm) if clip_mm is None else clip_mm
    Up = warp(U.astype(np.float32)) > 0.5
    dt = ndi.distance_transform_edt(Up) / ppm
    Ue = dt >= clip
    Cp = warp(c.astype(np.float32))
    Q = np.clip(np.ceil(LEVELS * Cp - 1e-6), 0, LEVELS).astype(np.uint16)
    Qe = Q * Ue.astype(np.uint16)
    RANK = _bitrev(LEVELS)
    ang = np.radians(ANGLE_DEG)
    tx_, ty_ = np.cos(ang), np.sin(ang)
    nx_, ny_ = -ty_, tx_
    corners = np.array([[0, 0], [paper_w, 0], [0, paper_h], [paper_w, paper_h]])
    us = corners @ np.array([tx_, ty_])
    vs = corners @ np.array([nx_, ny_])
    u = np.arange(np.floor(us.min()), np.ceil(us.max()) + RUN_STEP_MM, RUN_STEP_MM)
    k0, k1 = int(np.floor(vs.min() / (PITCH_PEN * W))), int(np.ceil(vs.max() / (PITCH_PEN * W)))
    S = []
    for ks in range(k0, k1 + 1, 64):
        kk = np.arange(ks, min(ks + 64, k1 + 1))
        v = (kk + 0.5) * W
        X = u[None, :] * tx_ + v[:, None] * nx_
        Y = u[None, :] * ty_ + v[:, None] * ny_
        ix = np.rint(X * ppm).astype(np.int64)
        iy = np.rint(Y * ppm).astype(np.int64)
        ok = (ix >= 0) & (ix < WP) & (iy >= 0) & (iy < HP)
        q = np.zeros(X.shape, np.uint16)
        q[ok] = Qe[iy[ok], ix[ok]]
        keep = q > RANK[kk % LEVELS][:, None]
        kp = np.pad(keep, ((0, 0), (1, 1))).astype(np.int8)
        d = np.diff(kp, axis=1)
        rs, cs = np.nonzero(d == 1)
        re, ce = np.nonzero(d == -1)
        ce = ce - 1
        Ln = u[ce] - u[cs]
        sel = Ln >= MIN_RUN_PEN * W - 1e-9
        rr = rs[sel]
        a_ = u[cs[sel]]
        b_ = u[ce[sel]]
        vv = v[rr]
        P0 = np.stack([a_ * tx_ + vv * nx_, a_ * ty_ + vv * ny_], 1)
        P1 = np.stack([b_ * tx_ + vv * nx_, b_ * ty_ + vv * ny_], 1)
        S.extend(np.stack([P0, P1], 1))
    return S, dt, Up, clip


def hatch_violations(S, dt, clip_mm, tol_mm=0.05):
    """Number of 0.025 mm samples along the runs whose distance to the outside of U is < clip - tol (must be 0)."""
    if not S:
        return 0
    ppm = HATCH_PPM
    HP, WP = dt.shape
    A = np.array(S)
    n = np.maximum(2, np.ceil(np.hypot(*(A[:, 1] - A[:, 0]).T) / RUN_STEP_MM).astype(int) + 1)
    bad = 0
    for m in np.unique(n):
        sel = A[n == m]
        t = np.linspace(0, 1, m)[None, :, None]
        P = (sel[:, 0:1, :] * (1 - t) + sel[:, 1:2, :] * t).reshape(-1, 2)
        ix = np.clip(np.rint(P[:, 0] * ppm).astype(int), 0, WP - 1)
        iy = np.clip(np.rint(P[:, 1] * ppm).astype(int), 0, HP - 1)
        bad += int((dt[iy, ix] < clip_mm - tol_mm).sum())
    return bad


def link_runs(runs, chk, pen_mm, cd=None, ncand=LINK_CANDIDATES):
    """Explicit switchback linker. runs (N,2,2) page mm; chk = LayerMaskCheck of the final trace mask of U.
    Returns (chains, connectors, stats)."""
    from scipy.spatial import cKDTree
    from ccplotter.config import PX2MM
    W = float(pen_mm)
    cd = LINK_RADIUS_PEN * W if cd is None else cd
    NV = np.array([-np.sin(np.radians(ANGLE_DEG)), np.cos(np.radians(ANGLE_DEG))])
    N = len(runs)
    P0, P1 = runs[:, 0], runs[:, 1]
    k = np.rint(P0 @ NV / (PITCH_PEN * W) - 0.5).astype(np.int64)
    E = np.concatenate([P0, P1])
    tree = cKDTree(E)
    used = np.zeros(N, bool)
    chains, conn = [], []
    toz = lambda p: p[:, 0] / PX2MM + 1j * p[:, 1] / PX2MM
    rem = N
    last = E[0]
    stats = dict(candidate_tests=0, accepted=0, ends_no_candidate=0, ends_all_rejected=0)
    while rem > 0:
        kq = 16
        while True:
            _, idx = tree.query(last, k=min(kq, 2 * N))
            idx = np.atleast_1d(idx)
            un = [i for i in idx if not used[i % N]]
            if un or kq >= 2 * N:
                break
            kq *= 4
        if not un:
            un = [i for i in range(2 * N) if not used[i % N]][:1]
        sid = int(un[0])
        r = sid % N
        enter = sid // N
        pts = [E[sid], E[(1 - enter) * N + r]]
        used[r] = True
        rem -= 1
        cur_k = k[r]
        ex = pts[-1]
        while True:
            ids = tree.query_ball_point(ex, cd)
            cands = []
            for i in ids:
                rr = i % N
                if used[rr]:
                    continue
                dk = k[rr] - cur_k
                cands.append((0 if dk > 0 else (1 if dk == 0 else 2), abs(dk), float(np.hypot(*(E[i] - ex))), i))
            if not cands:
                stats["ends_no_candidate"] += 1
                break
            cands.sort()
            cands = cands[:ncand]
            A = toz(np.repeat(ex[None, :], len(cands), 0))
            B = toz(np.array([E[c[3]] for c in cands]))
            ok = chk.ok(A, B)
            stats["candidate_tests"] += len(cands)
            sel = np.nonzero(ok)[0]
            if not len(sel):
                stats["ends_all_rejected"] += 1
                break
            i = cands[int(sel[0])][3]
            rr = i % N
            en = i // N
            conn.append(np.array([ex, E[i]]))
            pts.append(E[i])
            pts.append(E[(1 - en) * N + rr])
            used[rr] = True
            rem -= 1
            cur_k = k[rr]
            ex = pts[-1]
            stats["accepted"] += 1
        chains.append(np.array(pts))
        last = ex
    return chains, conn, stats


def hatched_blue_layer(prep, sep, geom, opts, workdir: Path, tag, pen_mm, stats_out=None):
    """The hatched blue layer: returns the routed (linked + keep-out clipped) layer SVG Path, or None if nothing is drawn.
    Writes <tag>_chk_trace.svg (+ sidecar, the final trace mask of U the linker checks against) and <tag>.connectors.svg."""
    import ccplotter.trace_v3 as tv3
    from ccplotter import twopen_keytint as kt
    from ccplotter.composite import _apply_global_clip, _paper_keepout_safe_region
    from ccplotter.config import PX2MM
    from ccplotter.stage_e_mask import LayerMaskCheck, load_sidecar
    from ccplotter.stats_warnings import _nn_walk_order
    workdir = Path(workdir)
    paper_w, paper_h = float(geom["paper_w"]), float(geom["paper_h"])
    mm_per_px = float(prep["mm_per_px"])
    c, U, _blue_all = target_fields(sep, prep["gray"], opts, mm_per_px, pen_mm)
    runs, dt, Up, clip = gen_hatch(c, U, geom, paper_w, paper_h, pen_mm)
    if not runs:
        return None
    viol = hatch_violations(runs, dt, clip)
    if viol:
        raise kt.KeytintError(f"hatched_blue: {viol} hatch samples fall outside the clipped region (bug)", 500)
    # the final trace mask of U (stage_e_mask), exactly as the layer-H / U checker
    o2 = dict(opts)
    o2["_stage_e_mask"] = True
    rawc = workdir / f"{tag}_chk_trace.svg"
    tv3._trace_svg(np.where(U, 0, 255).astype(np.uint8), rawc, o2)
    tm, fpx = load_sidecar(rawc)
    if tm is None:
        raise kt.KeytintError("hatched_blue: the union mask sidecar was not written; refusing to link unchecked", 500)
    chk = LayerMaskCheck(tm, fpx, geom["c0_px"], float(geom["fit"]), float(geom.get("rot", 0.0)), (geom["tx"] / PX2MM, geom["ty"] / PX2MM))
    chains, conn, lst = link_runs(np.array(runs), chk, pen_mm)
    cz = lambda p_: p_[:, 0] + 1j * p_[:, 1]
    starts = cz(np.array([ch[0] for ch in chains]))
    ends = cz(np.array([ch[-1] for ch in chains]))
    order = _nn_walk_order(starts, ends, ends[0])
    S = [chains[i][::-1] if f else chains[i] for (i, f) in order]
    svg, csvg = workdir / f"{tag}.routed.svg", workdir / f"{tag}.connectors.svg"
    kt.write_layer_svg(S, paper_w, paper_h, svg)
    kt.write_layer_svg(conn, paper_w, paper_h, csvg)
    try:
        region = _paper_keepout_safe_region(paper_w, paper_h, geom.get("paper_keepout"), 0.5 * max(0.0, float(opts.get("fill_mm", 0.15))))
    except ValueError as e:
        raise kt.KeytintError(str(e), 400)
    if region is not None:
        _apply_global_clip(svg, region)
    if stats_out is not None:
        conns = kt.load_strokes_mm(csvg)
        clen = float(sum(np.hypot(*np.diff(s, axis=0).T).sum() for s in conns if len(s) > 1))
        stats_out.update({"connector_distance_mm": LINK_RADIUS_PEN * pen_mm, "connector_count": len(conns), "connector_draw_length_mm": round(clen, 1),
                          "hatched_blue": {"enabled": True, "pen_mm": pen_mm, "runs": len(runs), "chains": len(chains), "clip_mm": clip,
                                           "link": lst, "u_mm2": float(U.sum() * mm_per_px ** 2), "c_mean_over_u": float(c[U].mean()),
                                           "hatch_violations": viol, "constants": CONSTANTS_HATCHED}})
    return svg
