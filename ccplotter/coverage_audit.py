r"""ccplotter/coverage_audit.py -- what the paper will actually look like.

THE PROBLEM THIS SOLVES. The preview can show solid dark fills while the plot
comes back cracked. This module reports two numbers for a job before it is
plotted: how much of the intended fill is white in the commanded GEOMETRY (a
mask-hole problem, present in the preview too if you zoom in), and how much MORE
white the paper is expected to show on top of that, from deposition loss
(overdraw removing ink on tight passes).

Under-generation (mask holes in the commanded geometry) is usually the dominant
term; deposition/overdraw loss is a smaller band (3-7 points on the one job it
was checked against) and is method-dependent, because the binarization chosen
to read a scan swings the paper-white figure by several points. Treat the
deposition term as a band, never a point value.

The deposition sweep (120 patches on real paper, pen `uright_gel`, the author's
measurement) shows coverage falling as commanded pitch tightens below the pen's
laid width --

    pitch 0.0875mm (2.9x overlap) -> 82.2% coverage
    pitch 0.1750mm (1.5x overlap) -> 92.0%
    pitch 0.2625mm (just touching) -> 96.9%

Pressure (86.6-93.2% across z_down -0.2..-1.8) and speed (87.5-95.2% across
1000..4500) were both flat: neither is the mechanism. `_SWEEP` below encodes
this curve.

WHAT THIS MODULE DOES. Reports both numbers for a job, before it is plotted:

    geometry_coverage   what fraction of intended-fill the strokes cover, at
                        the pen's real laid width. Anything below 1.0 is
                        already visible in the preview, if you zoom in.
    predicted_coverage  geometry_coverage x the deposition factor for the
                        commanded pitch, from the sweep above.
    predicted_white     what the paper should come back as.

So "the preview looked solid" stops being a surprise: the numbers say what the
sheet will do while there is still time to change it.

WIRED IN, ADVISORY ONLY. `ccplotter/routes/core.py` calls `audit()` on the
preview geometry for every /api/process response whose pen profile carries a
measured `laid_mm`, and attaches the result as `res["coverage_audit"]`. It reads
the already-generated preview SVG, adds no geometry, changes no stroke, and
touches no default. A pen with no measured `laid_mm` gets
`coverage_audit: None`. Any failure degrades to `{"error": ...}` rather than
failing the preview.

DEPOSITION CURVE IS PEN-SCOPED. `_SWEEP` was measured on exactly one pen,
`uright_gel`. A different pen's measured `laid_mm` licenses the GEOMETRY target
(geometry_coverage, geometry_white, overlap_ratio) -- those are pure raster
geometry and hold for any pen. It does NOT license the deposition curve:
overdraw-driven ink loss is a property of that pen's ink and tip, not a
geometric fact, and no other pen has been swept. `audit()` takes an optional
`pen_id` so a caller can say which pen it is auditing; the returned
`deposition_basis` field always says which pen the curve came from, and when
`pen_id` disagrees with it, the notes say plainly that the deposition figure is
extrapolated from another pen's ink, not measured for this one. It still
predicts (best available estimate), it just cannot claim to be measurement for
an unswept pen.

RASTER WIDTH. cv2.polylines with LINE_8 draws the smallest odd px count
>= th+1, never th (th=5 draws 7px, th=7 draws 9px, th=8 draws 9px). Reporting
`th` as the drawn width would inflate `ink`/`geometry_coverage`/
`predicted_paper_white` and make `raster_width_mm`/`raster_width_err_mm`
describe a width that was never rasterised. `_target_th_and_width` instead picks
the cv2 thickness whose REAL drawn width is nearest laid_mm and reports that
real width.
"""
from __future__ import annotations

import math

# Deposition factor vs overlap ratio (laid width / commanded pitch), from a
# deposition sweep: 120 patches, 5 pressures x 4 speeds x 3 pitches, uright_gel,
# measured off a scan of the sheet.
#
# Measured points, as (overlap, coverage):
#   0.97 -> 0.969     passes just touch (pitch 0.2625 vs laid 0.255)
#   1.46 -> 0.920     pitch 0.1750
#   2.91 -> 0.822     pitch 0.0875
# Below 1.0 the pen leaves geometric gaps and coverage is bounded by the
# spacing itself, so the curve is only defined at/above just-touching; callers
# get a linear geometric bound there instead.
#
# Known discontinuity at overlap == 0.97: deposition_factor() returns exactly
# 1.0 at and below 0.97 (the geometric "no overlap yet" bound) but the
# measured point AT 0.97 is 0.969 -- a real ~3.1pp step in the data. Smoothing
# it would mean inventing a measured point between 0.97 and 1.46, which
# nothing licenses; left as-is and documented rather than interpolated away.
_SWEEP_PEN = "uright_gel"
_SWEEP_DATE = "2026-08-23"
DEPOSITION_BASIS = f"{_SWEEP_PEN} {_SWEEP_DATE} sweep"
_SWEEP = [(0.97, 0.969), (1.46, 0.920), (2.91, 0.822)]


def deposition_factor(overlap: float) -> float:
    """Fraction of commanded ink that reaches paper, at this overlap ratio.

    Linear interpolation between the three measured points; flat beyond the
    ends rather than extrapolating, because nothing past 2.91x was measured and
    a fitted curve there would be invention. Returns 1.0 at or below
    just-touching -- there is no overdraw penalty when passes do not overlap.

    Overlap in [0.95, 1.0) returns a factor at or near 1.0 with no penalty
    baked in here -- callers must not read that as margin. laid_mm is a
    MEDIAN pen width (n=29,261); sub-median stretches in that band will gap.
    `_notes()` is responsible for saying so; see the module docstring.
    """
    if overlap <= _SWEEP[0][0]:
        return 1.0
    if overlap >= _SWEEP[-1][0]:
        return _SWEEP[-1][1]
    for (x0, y0), (x1, y1) in zip(_SWEEP, _SWEEP[1:]):
        if x0 <= overlap <= x1:
            t = (overlap - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return 1.0


def _drawn_width_for_th(th):
    """What cv2.polylines(..., thickness=th, lineType=cv2.LINE_8) ACTUALLY
    draws, in px -- measured, not assumed.

    cv2's thick-line renderer does NOT draw `th` px wide. Probed over
    th=1..16, LINE_8: th=1 draws 1px (a special-cased hairline); for th>=2 the
    drawn width is always the smallest ODD number >= th+1 -- i.e. even th
    draws th+1, odd th (>=3) draws th+2. LINE_AA follows the same odd-only
    rule, offset +2px vs LINE_8 at every th. This module uses LINE_8's drawn
    width, so `th` itself is never the right number to report as "how wide
    the stroke is."
    """
    if th <= 1:
        return 1
    return th + 1 if th % 2 == 0 else th + 2


def _th_for_drawn_width(w):
    """Smallest cv2 thickness that makes _drawn_width_for_th return `w`
    (w must be an odd px count >= 1; see _drawn_width_for_th)."""
    if w <= 1:
        return 1
    return w - 1  # even th -> _drawn_width_for_th(w - 1) == w


def _target_th_and_width(laid_mm, ppmm):
    """Pick the cv2 thickness whose ACTUAL drawn width is nearest laid_mm at
    this ppmm, and report the width really drawn -- not the commanded `th`.

    Drawn widths are always odd px (see _drawn_width_for_th), so most target
    widths are unreachable exactly; this picks the nearest achievable odd
    width and lets the caller see the real residual, honestly, rather than
    hiding it behind a `raster_width_mm` computed from `th` as if `th` were
    what got drawn.

    Uses round(), not int()/truncation, to find the nearest odd width:
    truncation would systematically bias the pick toward the narrower of the
    two candidate widths regardless of which is actually closer to
    target_px, which is not "nearest achievable," it is "always short."

    Returns (th, drawn_px, raster_width_mm, raster_width_err_mm).
    """
    target_px = laid_mm * ppmm
    n = round((target_px - 1) / 2)
    w = max(1, int(2 * n + 1))
    th = _th_for_drawn_width(w)
    raster_width_mm = w / ppmm
    raster_width_err_mm = raster_width_mm - laid_mm
    return th, w, raster_width_mm, raster_width_err_mm


def _raster_width_rel_err(laid_mm, ppmm):
    """Relative error of the width cv2 will ACTUALLY draw, at this ppmm, for
    the nearest-achievable thickness (see _target_th_and_width).

    Not sign-constrained: the nearest achievable odd width can land on either
    side of laid_mm, so this can be positive or negative.
    """
    _, _, raster_width_mm, err_mm = _target_th_and_width(laid_mm, ppmm)
    return err_mm / laid_mm


def _pick_ppmm(laid_mm, ppmm_hint, width_mm, height_mm,
               px_ceiling=240_000_000, max_mult=3.0, target_rel_err=0.02,
               step=0.05):
    """Raise ppmm above the caller's hint to shrink the nearest-achievable
    quantisation error on `laid_mm` (see _target_th_and_width), without
    silently going below the hint and without tripping the raster-size
    ceiling.

    Cost guard: W*H grows with ppmm**2 for a fixed bbox, so this only
    searches up to `ppmm_hint * max_mult` and stops raising ppmm the moment
    the resulting raster would exceed `px_ceiling` -- W*H is monotonic
    increasing in ppmm for a fixed bbox, so the first violation ends the
    search. It never returns less than `ppmm_hint`: if every candidate above
    the hint is worse or over budget, the hint itself is returned unchanged.

    Returns (ppmm_used, rel_err_at_that_ppmm).
    """
    best_ppmm = ppmm_hint
    best_err = _raster_width_rel_err(laid_mm, ppmm_hint)
    if abs(best_err) <= target_rel_err:
        return best_ppmm, best_err

    cap = ppmm_hint * max_mult
    n_steps = int((cap - ppmm_hint) / step)
    for i in range(1, n_steps + 1):
        cand = ppmm_hint + i * step
        w = int(width_mm * cand) + 8
        h = int(height_mm * cand) + 8
        if w * h > px_ceiling:
            break  # monotonic in ppmm for fixed bbox -- no point continuing
        err = _raster_width_rel_err(laid_mm, cand)
        if abs(err) < abs(best_err):
            best_ppmm, best_err = cand, err
            if abs(best_err) <= target_rel_err:
                break
    return best_ppmm, best_err


def audit(strokes_mm, laid_mm, ppmm=32.0, fill_density=0.55, pen_id=None, fill_mode=None):
    """Measure a job's fill coverage and predict what the paper will do.

    strokes_mm : iterable of polylines, [(x_mm, y_mm), ...], in paper mm
    laid_mm    : the pen's MEASURED laid line width (profile `laid_mm`), NOT
                 `fill_mm` (pass spacing) and NOT `pen_mm` (a nominal barrel
                 figure that was wrong by 2x on the one pen ever measured).
    ppmm       : FLOOR/hint for raster resolution, not a fixed value. 32 px/mm
                 puts ~8 px across the pen, enough to resolve sub-pen gaps --
                 but cv2 can only draw odd whole-pixel widths (see below) so
                 quantising the pen width to the nearest one can leave several
                 percent of error at that
                 resolution. `audit()` may raise ppmm above this hint to
                 shrink that error toward ~2%, subject to the raster-size
                 ceiling and a fixed search cap (`_pick_ppmm`); it never goes
                 below the hint. The ppmm actually used is returned as `ppmm`.
    fill_density : local ink density above which a region counts as
                 intended-fill. Typical value: 0.55 in a 2.5mm window.
    pen_id     : optional, which pen this job is actually being drawn with.
                 Only used to annotate the deposition figure's provenance --
                 it changes no arithmetic. Omitted (None) leaves the output
                 unchanged; routes/core.py does not pass it.
    fill_mode  : optional. `_median_pitch` ("nearest approach between points of
                 DIFFERENT strokes") is unreliable on ordinary contour fixtures
                 (it reads about 1.56x low there); zigzag's dense scanline
                 geometry is a harder case for it, not an easier one, and no
                 replacement estimator exists yet. When
                 `fill_mode == "zigzag"`, `commanded_pitch_mm`/`overlap_ratio`/
                 `deposition_factor`/`predicted_paper_*` are still computed and
                 returned unchanged but the overlap/deposition NOTES that tell
                 the user to act on them ("widening the pass pitch... recovers
                 it") are suppressed, and
                 `pitch_estimate_unreliable: true` is added to the output so a
                 reader can tell why those notes are missing. `geometry_coverage`
                 does not depend on `_median_pitch` at all and is unaffected.
                 Omitted (None) leaves the output unchanged -- no
                 `pitch_estimate_unreliable` key is added at all unless
                 `fill_mode == "zigzag"` (additive metadata, same convention as
                 every other opt-in key in this codebase).

    Returns a dict; every field is measured or interpolated from measurement,
    none is assumed. `deposition_basis` names the pen the deposition curve was
    swept on (uright_gel only, see module docstring) -- geometry fields are
    pen-agnostic, the deposition-derived ones (deposition_factor,
    predicted_paper_coverage, predicted_paper_white) are not, and `notes` says
    so explicitly when `pen_id` names a different pen.
    """
    import cv2
    import numpy as np

    pts = [p for s in strokes_mm for p in s]
    if not pts:
        return {"error": "no strokes"}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    width_mm, height_mm = (x1 - x0), (y1 - y0)

    # cv2.polylines with thickness=th draws an odd number of pixels that is
    # always >= th+1, not th (see _drawn_width_for_th), so
    # `_target_th_and_width` picks the cv2 thickness whose REAL drawn width is
    # nearest laid_mm and returns that real drawn width for reporting.
    # Nearest-achievable (not floor) is deliberate: flooring `th` would not buy
    # a safety margin, since the drawn width is not `th`, and cv2's narrowest
    # line is already 1px. Ppmm is raised (never lowered) above the caller's
    # hint to shrink whatever quantisation residual remains, bounded by the
    # raster-size ceiling.
    ppmm, _ = _pick_ppmm(laid_mm, ppmm, width_mm, height_mm)

    W = int(width_mm * ppmm) + 8
    H = int(height_mm * ppmm) + 8
    if W < 8 or H < 8 or W * H > 240_000_000:
        return {"error": f"raster {W}x{H} out of range"}

    canvas = np.zeros((H, W), np.uint8)
    th, _drawn_px, raster_width_mm, raster_width_err_mm = _target_th_and_width(laid_mm, ppmm)
    for s in strokes_mm:
        if len(s) < 2:
            continue
        p = np.array([[(px - x0) * ppmm + 4, (py - y0) * ppmm + 4] for px, py in s],
                     dtype=np.int32)
        cv2.polylines(canvas, [p], False, 255, th, lineType=cv2.LINE_8)
    ink = canvas > 0

    win = int(2.5 * ppmm) | 1
    dens = cv2.blur(ink.astype(np.float32), (win, win))
    fill = dens > fill_density
    if fill.sum() < 100:
        return {"fill_area_frac": float(fill.mean()), "note": "no fill regions in this job"}

    geom_cov = float(ink[fill].mean())

    # commanded pitch: nearest approach between points of DIFFERENT strokes
    pitch = _median_pitch(strokes_mm)
    overlap = (laid_mm / pitch) if pitch and pitch > 0 else None
    dep = deposition_factor(overlap) if overlap else None
    pred = geom_cov * dep if dep else None

    out = {
        "laid_mm": laid_mm,
        "ppmm": round(ppmm, 3),
        "raster_width_mm": round(raster_width_mm, 4),
        "raster_width_err_mm": round(raster_width_err_mm, 4),
        "fill_area_frac": round(float(fill.mean()), 4),
        "geometry_coverage": round(geom_cov, 4),
        "geometry_white": round(1 - geom_cov, 4),
        "commanded_pitch_mm": round(pitch, 4) if pitch else None,
        "overlap_ratio": round(overlap, 3) if overlap else None,
        "deposition_factor": round(dep, 4) if dep else None,
        "deposition_basis": DEPOSITION_BASIS,
        "predicted_paper_coverage": round(pred, 4) if pred else None,
        "predicted_paper_white": round(1 - pred, 4) if pred else None,
    }
    # Values above are kept even when the pitch estimator behind them
    # (`_median_pitch`) is known to be unreliable for this geometry -- only the
    # NOTES (which tell the user to act on overlap_ratio) and the flag are
    # conditioned on fill_mode.
    _pitch_unreliable = (fill_mode == "zigzag")
    if _pitch_unreliable:
        out["pitch_estimate_unreliable"] = True
    out["notes"] = _notes(out, pen_id, suppress_overlap_notes=_pitch_unreliable)
    return out


def _median_pitch(strokes_mm, sample=40000):
    """Median distance from a drawn point to the nearest point of a DIFFERENT
    stroke -- the real spacing between neighbouring passes."""
    import numpy as np
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    dens, sid = [], []
    for i, s in enumerate(strokes_mm):
        if len(s) < 2:
            continue
        a = np.asarray(s, dtype=float)
        seg = np.linalg.norm(np.diff(a, axis=0), axis=1)
        L = float(seg.sum())
        if L <= 0:
            continue
        n = max(2, int(L / 0.05))
        cum = np.concatenate(([0], np.cumsum(seg)))
        t = np.linspace(0, L, n)
        dens.append(np.stack([np.interp(t, cum, a[:, 0]), np.interp(t, cum, a[:, 1])], 1))
        sid.append(np.full(n, i))
    if not dens:
        return None
    D = np.vstack(dens)
    I = np.concatenate(sid)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(D), size=min(sample, len(D)), replace=False)
    dd, ii = cKDTree(D).query(D[idx], k=40)
    out = []
    for rd, ri, src in zip(dd, ii, I[idx]):
        m = I[ri] != src
        if m.any():
            out.append(rd[m][0])
    return float(np.median(out)) if out else None


def _notes(a, pen_id=None, suppress_overlap_notes=False):
    n = []
    gw = a.get("geometry_white")
    if gw is not None and gw > 0.05:
        n.append(f"{gw*100:.0f}% of intended-fill is white IN THE GEOMETRY. This is holes "
                 f"in the source mask, not a fill defect, and it is present in the preview "
                 f"too -- it just reads as solid at preview zoom.")
    ov = a.get("overlap_ratio")
    # These three branches all tell the user to act on `ov`, which comes from
    # `_median_pitch`, an estimator that is unreliable for zigzag fills.
    # `suppress_overlap_notes` (True only when the caller passed
    # fill_mode="zigzag") withholds the ADVICE without touching
    # `ov`/`deposition_factor`/`predicted_paper_*` themselves, which stay in the
    # returned dict.
    if suppress_overlap_notes:
        pass
    elif ov and ov > 1.2:
        n.append(f"passes overlap {ov:.2f}x. Measured on paper, overdraw REMOVES ink: "
                 f"expect only {a['deposition_factor']*100:.0f}% of commanded ink to land. "
                 f"Widening the pass pitch toward the pen's {a['laid_mm']}mm width recovers it.")
    elif ov and ov < 0.95:
        n.append(f"passes are {ov:.2f}x the pen width apart -- they do not touch, so the "
                 f"fill cannot read solid however well the ink deposits.")
    elif ov and 0.95 <= ov < 1.0:
        n.append(f"passes are {ov:.2f}x the pen width apart -- at or just below just-touching, "
                 f"with NO margin. The deposition curve reports near-full coverage here, but "
                 f"{a['laid_mm']}mm is a MEDIAN pen width (n=29,261): sub-median stretches of "
                 f"the same pen will gap at this pitch even though the median would not.")
    if pen_id is not None and pen_id != _SWEEP_PEN and a.get("deposition_factor") is not None:
        n.append(f"deposition figure is EXTRAPOLATED from {_SWEEP_PEN}'s ink ({DEPOSITION_BASIS}), "
                 f"not measured for {pen_id}. It is the best available estimate, not a "
                 f"measurement of this pen.")
    pw = a.get("predicted_paper_white")
    if pw is not None:
        n.append(f"predicted paper: {pw*100:.0f}% white inside fill regions.")
    return n
