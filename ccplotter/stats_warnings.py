import logging
import math

import numpy as np
import vpype as vp

from ccplotter.atomic_io import atomic_open
from ccplotter.config import (PX2MM, GRAY_CRUSH_TRIGGER, SCALE_AWARE_V2_NATIVE_CAP,
                              SCALE_AWARE_V2_FILL_PX_FLOOR)


# ── Plotter stats & safety warnings (computed from vector data, no gcode needed) ──
def _polyline_len_mm(line):
    """Length in mm of a vpype line (1-D complex np array, px units)."""
    if line is None or len(line) < 2:
        return 0.0
    return float(np.abs(np.diff(line)).sum()) * PX2MM

def estimate_plot_time(draw_len_mm, travel_len_mm, n_lifts, draw_speed, travel_speed, plunge_speed, lift_speed, z_up, z_down):
    """Rough plot time in seconds. Speeds are mm/min."""
    zdist = abs(float(z_up) - float(z_down))
    draw_t   = (draw_len_mm   / max(float(draw_speed), 1.0))   * 60.0
    travel_t = (travel_len_mm / max(float(travel_speed), 1.0)) * 60.0
    z_t      = n_lifts * ((zdist / max(float(plunge_speed), 1.0)) + (zdist / max(float(lift_speed), 1.0))) * 60.0
    return draw_t + travel_t + z_t

def compute_vector_stats(doc, draw_speed=3000, travel_speed=8000, plunge_speed=500, lift_speed=2000, z_up=3.5, z_down=-1.0):
    """Plotter stats straight from a laid-out vpype Document (paper-space px).
    No G-code required — gives the user safety info at preview time."""
    lines = [ln for lc in doc.layers.values() for ln in lc]
    stroke_count = len(lines)
    lens = [_polyline_len_mm(ln) for ln in lines]
    draw_len = sum(lens)
    travel_len, prev_end = 0.0, None
    for ln in lines:
        if len(ln) == 0:
            continue
        if prev_end is not None:
            travel_len += abs(ln[0] - prev_end) * PX2MM
        prev_end = ln[-1]
    n_lifts = stroke_count                 # ~1 pen lift per stroke
    z_moves = 2 * stroke_count             # down + up per stroke
    b = doc.bounds()
    bbox_mm = [round(b[0]*PX2MM, 1), round(b[1]*PX2MM, 1), round(b[2]*PX2MM, 1), round(b[3]*PX2MM, 1)] if b else [0, 0, 0, 0]
    t = estimate_plot_time(draw_len, travel_len, n_lifts, draw_speed, travel_speed, plunge_speed, lift_speed, z_up, z_down)
    # Per-stroke length stats (mm) — diagnoses tiny strokes / plottability.
    slens = sorted(lens)
    length_stats = {
        "min_mm": round(slens[0], 2) if slens else 0.0,
        "median_mm": round(float(np.median(slens)), 2) if slens else 0.0,
        "mean_mm": round(float(np.mean(slens)), 2) if slens else 0.0,
        "max_mm": round(slens[-1], 2) if slens else 0.0,
    }
    return {
        "stroke_count": stroke_count,
        "estimated_pen_lifts": n_lifts,
        "estimated_z_moves": z_moves,
        "draw_length_mm": round(draw_len, 1),
        "travel_length_mm": round(travel_len, 1),
        "estimated_plot_time_s": round(t, 1),
        "estimated_plot_time_min": round(t / 60.0, 1),
        "bbox_mm": bbox_mm,
        "line_length_stats_mm": length_stats,
        "min_stroke_mm": length_stats["min_mm"],
    }

def illustration_pitch_warning(pitch_mm, fill_mm):
    """The mandatory pitch-vs-fill warning text (exact wording is part of the contract)."""
    return (f"Trace V3+: passes {pitch_mm:.3f} mm apart at this placement; "
            f"your pen's fill is {fill_mm:.3f} mm; reduce scale/paper or use Trace (V3)")


def _illustration_pitch_hits(stats):
    """[(item_id, message)] for every item the pitch-vs-fill check fires on (see illustration_pitch_warnings)."""
    hits = []
    for _iid, _ig in (stats.get("item_geometry") or {}).items():
        _equiv = (_ig or {}).get("scale_aware_equivalence_prelim_approx") or {}
        _p, _f = _equiv.get("illustration_pitch_mm"), _equiv.get("illustration_fill_mm")
        if _p is not None and _f is not None and _p > _f + 1e-9:
            hits.append((str(_iid), illustration_pitch_warning(_p, _f)))
    return hits


def illustration_pitch_warnings(stats):
    """Messages for every item whose Line Art / Illustration pass pitch exceeds the
    pen's fill_mm. Reads only the illustration-only fields composite.py adds."""
    return [m for _iid, m in _illustration_pitch_hits(stats)]


# Realised pass pitch vs the pen. The readout (stats["realised_pitch"], composite.py
# _realised_pitch_record) is rounded to 4 decimals, so its quantum is 0.0001 mm (half-quantum
# 0.00005). The warning fires only when the pitch exceeds the reference by MORE than this, so
# rounding noise at equality (pitch == fill) stays silent; 0.0001 mm is under 0.1% of any pen
# fill in the seed profiles (0.12 mm and up).
REALISED_PITCH_TOL_MM = 1e-4


def realised_pitch_warning(pitch_mm, ref_mm, laid=False, item_id=None, many=False):
    """The realised-pitch warning text. `laid` selects the measured-line-width wording; `item_id` is shown only when
    the job has several tracev3-engine items (so a single-item job's text is the plain sentence)."""
    what = (f"this pen's measured line width {ref_mm:.3f} mm" if laid else f"this pen's fill {ref_mm:.3f} mm")
    msg = (f"Passes {pitch_mm:.3f} mm apart on this page, wider than {what} -- expect white "
           f"slivers between passes; plot smaller or use a coarser pen.")
    return f"Item {item_id}: {msg}" if many and item_id is not None else msg


def realised_pitch_warnings(stats, laid_mm=None, skip_ids=()):
    """Messages for every tracev3-engine item whose REALISED pass pitch (stats["realised_pitch"],
    from the final placed scale) exceeds the pen's fill_mm -- or `laid_mm` when the pen profile carries a
    measured laid width (that is what the pitch is aimed at on such pens). Keyed ONLY on the
    realised_pitch block: no block, no message. `skip_ids` = item ids already covered by the stronger
    pitch-vs-fill danger (deduplication)."""
    rp = stats.get("realised_pitch")
    items = (rp or {}).get("items") if isinstance(rp, dict) else None
    if not items:
        return []
    laid = laid_mm is not None and float(laid_mm) > 0
    msgs = []
    for it in items:
        if str(it.get("id")) in skip_ids:
            continue
        pitch = it.get("realised_pitch_mm")
        ref = float(laid_mm) if laid else it.get("fill_mm_target")
        if pitch is None or ref is None:
            continue
        if float(pitch) > float(ref) + REALISED_PITCH_TOL_MM:
            msgs.append(realised_pitch_warning(float(pitch), float(ref), laid, it.get("id"), len(items) > 1))
    return msgs


def build_warnings(stats, laid_mm=None):
    """`laid_mm` (optional, default None = fill_mm comparison): the active pen profile's measured laid
    width, for the realised-pitch check. Not in stats; the caller passes it.

    Map stats to {level, msg, tier} warnings. Levels: good|okay|warning|danger.
    Tiers: 'primary' (always shown — physics/reachability/quality that gates a
    decision) vs 'secondary' (advisory volume — collapsed by default in the UI;
    still shipped in the API for anyone who wants it). Stroke-count and Z-move
    advisories are secondary; the pen-physics ceiling (tiny detail vs pen width)
    stays primary and loud."""
    out = []
    def add(level, msg, tier="primary"):
        out.append({"level": level, "msg": msg, "tier": tier})

    # ── Trace resolution transparency (hidden downscaling destroyed text once;
    #    never let it happen silently again) ──
    ti = stats.get("trace_info") or {}
    if ti.get("downscaled"):
        add("danger", f"This source is being DOWNSCALED before tracing ({ti['native_px']}px → {ti['effective_res']}px) — "
                       f"tiny text/details may be destroyed. Raise Resolution (mode cap {ti['mode_cap']}px).")

    # ── connector_mask_check: a layer that failed closed got NO lift-saving bridges, silently
    #    otherwise. Present only when the opt-in ran, so an off run's warnings are untouched. ──
    _mc = ((stats.get("connector_mask_check") or {}).get("totals") or {}).get("layers_failed_closed_detail") or {}
    for _lid in sorted(_mc, key=lambda k: (len(str(k)), str(k))):
        add("warning", f"mask check unavailable for layer {_lid} -- lifts not reduced there ({_mc[_lid]})")

    # ── stay_down_over_ink (opt-in): same rule for a layer that failed closed -- nothing was joined there.
    _sd = ((stats.get("stay_down_over_ink") or {}).get("totals") or {}).get("layers_failed_closed_detail") or {}
    for _lid in sorted(_sd, key=lambda k: (len(str(k)), str(k))):
        add("warning", f"stay-down mask unavailable for layer {_lid} -- lifts not reduced there ({_sd[_lid]})")

    # ── scale_aware_v2 SCALE_AWARE_V2_NATIVE_CAP saturation cliff. v2's
    #    resolution-floor boost keeps the tight k-scaled spacing precise at large
    #    placements by raising the working resolution for that item's trace pass,
    #    bounded by SCALE_AWARE_V2_NATIVE_CAP. A finer fill_mm needs
    #    proportionally more resolution for the same real spacing, so the cliff
    #    falls at smaller placements as fill_mm shrinks. Measured (plane fixture,
    #    B5, base resolution 2200): at fill_mm=0.25 the boost first activates
    #    ~141mm with an interpolated cliff ~427mm; at fill_mm=0.19 the cliff is
    #    ~248-250mm, almost exactly B5's own 250mm long edge, i.e. an ordinary
    #    full-bleed B5 plot. The ceiling (5480, config.py) clears that case with
    #    headroom while staying below the boost formula's own hard bound (2.5x
    #    base_res, from SCALE_AWARE_V2_FILL_PX_FLOOR / the 1.2px fill_px_real
    #    clamp), so genuinely larger placements (>=~260mm on this geometry) still
    #    saturate and still warn. The trigger below is the general,
    #    geometry-independent signature of that failure (boosted_res pinned at
    #    SCALE_AWARE_V2_NATIVE_CAP while actual_boost < needed_boost), not a
    #    hardcoded mm value -- the cliff's mm location shifts with fill_mm,
    #    paper size, and requested resolution, this condition does not. ──
    for _iid, _ig in (stats.get("item_geometry") or {}).items():
        _equiv = (_ig or {}).get("scale_aware_equivalence_prelim_approx") or {}
        _boost = _equiv.get("v2_res_boost")
        if not isinstance(_boost, dict):
            continue
        _needed, _actual, _boosted = _boost.get("needed_boost"), _boost.get("actual_boost"), _boost.get("boosted_res")
        if _needed is None or _actual is None or _boosted is None:
            continue
        if _boosted >= SCALE_AWARE_V2_NATIVE_CAP and _actual < _needed - 1e-6:
            _fpx = _equiv.get("fill_px_real")
            # Line Art / Illustration ONLY: the style's pitch clamp holds the physical
            # pass spacing even when the resolution cap binds, so this "can no longer
            # hold its own spacing target" danger would be false (and its "plot smaller"
            # advice misleading). While the pitch is held (<= the pen's fill_mm) say what
            # IS lost -- raster detail per pass -- at okay level. When the pitch exceeds
            # fill the danger stays (and the pitch-vs-fill warning fires below). Keyed on the illustration-only fields, which no other style
            # carries, so every other style's warnings are byte-identical.
            _ip, _if = _equiv.get("illustration_pitch_mm"), _equiv.get("illustration_fill_mm")
            if _ip is not None and _if is not None and _ip <= _if + 1e-9:
                add("okay",
                    f"Trace V3+: spacing held at {_ip:.3f} mm (pen fill {_if:.3f}). "
                    f"Trace detail is capped at this size ({float(_fpx):.2f} px per pass; "
                    f"{SCALE_AWARE_V2_FILL_PX_FLOOR:g} is the design target).")
                continue
            add("danger",
                f"This placement has hit the line-density resolution boost's ceiling "
                f"(needed {_needed}x, capped at {_actual}x) and can no longer hold its "
                f"own spacing target (fill_px real ~{_fpx}px, below the 3.0px floor). "
                f"Where this begins depends on your pen's fill_mm and the placement "
                f"size -- it is NOT a fixed mm figure (a finer pen's calibration "
                f"reaches this ceiling at a smaller placement than a coarser one's). "
                f"On the fixture used to size this ceiling, crack% moved the WRONG "
                f"direction past a comparable cap while wall-clock stayed near-worst-"
                f"case -- the density fix silently loses effectiveness right where it "
                f"costs the most. Plot smaller, or expect degraded line density here.")

    # ── Line Art / Illustration pitch vs the pen's fill. The style's pass
    #    spacing is fixed by the trace frame, so at a large placement it can exceed the
    #    width the pen lays (white slivers between passes). Fields exist only for the
    #    illustration style (composite.py), so no other style's warnings can change. ──
    _d1014 = _illustration_pitch_hits(stats)
    for _iid, _m in _d1014:
        add("danger", _m)

    # ── Realised pass pitch wider than the pen's fill / laid width. Keyed only on
    #    stats["realised_pitch"] (tracev3-engine items); an item already covered by the
    #    pitch-vs-fill danger above is skipped, so one item never gets two pitch messages (the danger is the
    #    stronger one). WARNING tier. ──
    for _m in realised_pitch_warnings(stats, laid_mm=laid_mm, skip_ids={i for i, _ in _d1014}):
        add("warning", _m)

    # ── Preview placement out of reach (WARNING, not a refusal) ──
    # Crowded pages on registered paper should keep rendering a preview rather
    # than erroring out. The preview does not refuse, so the warning has to live
    # here instead -- loudly, because the job WILL be refused at G-code
    # generation if the layout is left like this.
    _prw = stats.get("preview_reach_warning")
    if _prw:
        add("danger",
            f"This layout currently falls outside the pen's reachable area, so it "
            f"will be REFUSED when you generate G-code. The preview is still shown "
            f"so you can keep arranging -- use Fit to Paper, or move the content "
            f"inward, before generating. Details: {_prw}")

    # ── Fill Page reach shortfall. _fill_page_safe_region (composite.py) computes
    #    paper-INTERSECT-reach correctly; without this warning the shortfall is
    #    silent: when only 77.8mm of a 182mm sheet is pen-reachable, Fill Page
    #    (crop) draws ~43% of the page height with nothing saying so. The 0.9
    #    threshold is measured against a table of placements on the author's
    #    machine: reachable height ranged 17.5%-97.8% of page height across
    #    realistic offset_y, and 0.9 sits above every row except the best two, so
    #    it fires whenever there is a real shortfall and stays silent only near
    #    full reach. Fires ONLY when fill_page was actually requested
    #    (fill_page_reach present). ──
    _fpr = stats.get("fill_page_reach")
    if isinstance(_fpr, dict):
        _ratio = float(_fpr.get("height_ratio", 1.0) or 0.0)
        if _ratio < 0.9:
            _safe_h, _paper_h = _fpr.get("safe_h_mm", 0.0), _fpr.get("paper_h_mm", 0.0)
            add("danger",
                f"Only {_safe_h:.0f}mm of this {_paper_h:.0f}mm sheet is pen-reachable at this "
                f"paper placement ({_ratio*100:.0f}%) -- Fill Page will draw that band and leave "
                f"the rest blank. Moving the paper toward the front of the machine increases it.")

    # ── Reachability: drawing falling outside the pen-reachable area ──
    mb = stats.get("machine_bbox")
    if isinstance(mb, dict) and mb.get("off_reach_mm", 0) > 0.2:
        bb, r = mb["bbox"], mb.get("reach_bbox", mb.get("reach", []))
        profile = mb.get("reach_label", "active reach profile")
        add("danger", f"Drawing extends {mb['off_reach_mm']}mm OUTSIDE the pen-reachable area "
                       f"(drawing X[{bb[0]}-{bb[2]}] Y[{bb[1]}-{bb[3]}] vs {profile} bounding extent {r}). "
                       f"It will plot off the paper / hit the travel limits — move the drawing inside the green reachable boundary before plotting.")

    # ── Pen-physics ceiling (KEEP LOUD, primary, singular): one line covering
    #    both tiny-detail-vs-pen-width (from the Small-Detail Fidelity audit) and
    #    shortest-stroke-vs-pen-width, so the stack doesn't repeat the same
    #    physical limit twice. ──
    _audit = (ti.get("audit") or {})
    _gpx = float(_audit.get("small_detail_min_switched_px") or 0)
    _ceiling_bits = []
    if _gpx > 0 and _audit.get("small_detail_switched"):
        _eff = float(ti.get("effective_res") or 0)
        # bbox_mm (drawing extent in mm) is present at preview time; machine_bbox
        # only on gcode builds — use whichever is available.
        _bb = stats.get("bbox_mm") or (stats.get("machine_bbox") or {}).get("bbox")
        _pen = float(stats.get("pen_width_mm", 0) or 0.5) or 0.5
        if _eff and _bb:
            _glyph_mm = _gpx * (float(_bb[2]) - float(_bb[0])) / _eff
            _ratio = _glyph_mm / _pen
            if _ratio < 2.0:
                _ceiling_bits.append(f"tiny detail ~{_glyph_mm:.1f}mm (~{_ratio:.1f}× the {_pen}mm pen)")
    pen = float(stats.get("pen_width_mm", 0) or 0)
    min_mm = float(stats.get("min_stroke_mm", 0) or 0)
    if pen > 0 and stats.get("stroke_count", 0) > 0 and min_mm < 0.5:
        _ceiling_bits.append(f"smallest strokes ~{min_mm}mm")
    if _ceiling_bits:
        add("warning", f"Pen-physics ceiling: {'; '.join(_ceiling_bits)} — below the ~0.5mm / 2× pen-width line, "
                        f"readability may be physically limited. Plot larger, use a finer pen, or accept symbolic marks.")

    # ── Sketch Field (native PFM) line-art input disclosure: a loud numeric
    #    warning, not a classifier-gated confirm. PFM input is opt-in
    #    stylization, and the stroke-width heuristic reused here is
    #    prototype-grade (a single global scalar), so it discloses and never
    #    blocks. It reuses the grayscale prototype's classifier formula verbatim
    #    via pfm_native._measure_stroke_width_px (see that module's docstring). ──
    #    pfm_native._measure_stroke_width_px -- see that module's docstring). ──
    _swc = _audit.get("stroke_width_check")
    if isinstance(_swc, dict) and _swc.get("verdict") == "line_art":
        add("warning",
            f"Sketch Field input measures as LINE ART, not tonal gray-mass "
            f"(mean stroke width ~{_swc.get('stroke_width_px')}px, under the "
            f"{_swc.get('line_art_ceiling_px')}px line-art ceiling; ink covers "
            f"~{_swc.get('area_fraction', 0)*100:.0f}% of the image). Sketch Field "
            f"will RESTYLE this as hatched sketch strokes rather than preserve it "
            f"as clean lines -- expected behavior for an opt-in stylization mode, "
            f"not a bug. Use tracev3/Line Art if you want the original strokes kept.")

    # ── Otsu shadow-crush. gray_crush_fraction lives in trace_info.audit for
    #    BOTH tracev3 (_TRACE_AUDIT) and plotready/lineart (_LINEART_AUDIT) --
    #    same info.json merge point -- so this one check covers all three modes.
    #    Primary tier; fires only at or above GRAY_CRUSH_TRIGGER (1%), and below
    #    that nothing is shown at all. It supersedes the whole-image gray_pct
    #    advisory for these styles; that check stays where it is (plotready
    #    secondary) and is not wired into tracev3, since that would change
    #    tracev3's default output.
    _gcf = float(_audit.get("gray_crush_fraction") or 0)
    if _gcf >= GRAY_CRUSH_TRIGGER:
        add("warning", f"Shading/hatching is being crushed into solid blobs by the ink threshold "
                        f"(~{_gcf*100:.0f}% of the ink comes from gray-wash source regions collapsed "
                        f"into filled areas, not lines). Consider AI Clean to Line Art for a cleaner result.")

    sc, zm = stats["stroke_count"], stats["estimated_z_moves"]
    if sc >= 2500:
        add("warning", f"Very dense: {sc} strokes. Prefer Connect Strokes / Target Z-moves (keeps detail); Min Stroke Length / Noise Cull also reduce strokes but DELETE small detail (text, ticks).", "secondary")
    elif sc >= 1200:
        add("warning", f"Dense: {sc} strokes. Raise Connect Strokes first (keeps detail); Smoothness / Min Stroke Length reduce strokes by deleting detail.", "secondary")
    elif sc >= 500:
        add("okay", f"{sc} strokes — okay.", "secondary")
    if zm >= 4000:
        add("warning", f"Estimated {zm:,} Z-moves — very hard on the Z stepper. Raise Connect Strokes / set Target Z-moves before sending (detail-preserving); Min Stroke Length helps but deletes small detail.", "secondary")
    elif zm >= 2000:
        add("warning", f"Estimated {zm:,} Z-moves — high. Raise Connect Strokes or set Target Z-moves (keeps detail).", "secondary")
    elif zm >= 1000:
        add("okay", f"Estimated {zm:,} Z-moves — moderate.", "secondary")

    # NOTE: there is deliberately no plot-time threshold warning here. It would
    # change the `warnings` array for EVERY request whose estimated_plot_time_min
    # crosses the threshold, not just DrawingBotV3 imports (a 2573-shape file
    # estimates 58.1 min at default settings), and would break byte-identity of
    # existing modes' output: a pinned case at 61.3 min would gain a fifth warning
    # where its manifest has exactly four. Adding one needs an explicit default
    # change with the replay manifest re-recorded. estimated_plot_time_min is still
    # computed and always returned/shown (stat-time, run history).

    # ── Low-lift routing summary + aggressiveness warnings ──
    cc = stats.get("connector_count", 0)
    if cc:
        before, after = stats.get("strokes_before"), stats.get("strokes_after")
        zb, za = stats.get("z_moves_before"), stats.get("z_moves_after")
        cdist, longest = stats.get("connector_distance_mm", 0), stats.get("longest_connector_mm", 0)
        cdraw = stats.get("connector_draw_length_mm", 0)
        add("okay", f"Low-lift routing: {before}→{after} strokes, {zb:,}→{za:,} Z-moves · {cc} connectors, +{cdraw}mm drawn.", "secondary")
        if longest > max(cdist, 0.01) + 0.1:
            add("warning", f"Aggressive routing: connectors up to {longest}mm drawn to hit the Z-move target — expect visible joining marks across the artwork. Raise the target or it draws longer bridges.")
        elif cdist >= 1.5:
            add("warning", f"Long connectors enabled ({cdist}mm): bridges up to {longest}mm may show. Lower Connect Strokes if line purity matters more than Z-moves.")

    # ── Physical plottability: shortest strokes vs pen tip width (blob/vanish
    #    note only — the below-0.5mm case is already folded into the ceiling
    #    warning above) ──
    if pen > 0 and stats.get("stroke_count", 0) > 0 and 0.5 <= min_mm < 2 * pen:
        add("okay", f"Some strokes (~{min_mm}mm) are shorter than 2× pen width ({pen}mm) — fine detail may blob or vanish on paper.", "secondary")

    # ── Plot-Ready Line Art quality warnings ──
    lq = stats.get("lineart_quality")
    if isinstance(lq, dict):
        for level, msg in lq.get("warnings", []):
            tier = "secondary" if "gray/shading" in msg else "primary"
            add(level, msg, tier)

    # ── Small-detail lane notes ──
    sdb = (stats.get("hybrid") or {}).get("small_detail")
    if isinstance(sdb, dict) and sdb.get("mode") in ("balanced", "preserve"):
        acc = sdb.get("accepted", 0)
        if sdb.get("overflow"):
            add("warning", "Small-detail preservation produced many tiny strokes (capped). Use Balanced or raise Noise Cull.")
        if acc:
            add("okay", f"Small-detail ROI lane: {sdb.get('roi_count',0)} regions ({sdb.get('roi_area_pct',0)}% of image) → kept {acc} strokes (~{sdb.get('estimated_z_moves_added',0)} extra Z-moves).", "secondary")
        elif sdb.get("roi_count", 0) == 0:
            add("okay", f"Small Detail = {sdb['mode']}: no small-detail regions found.", "secondary")
        else:
            add("okay", f"Small Detail = {sdb['mode']}: {sdb.get('roi_count',0)} regions found but no strokes passed local support tests.", "secondary")
    return out

def _target_fuse(fuse, gaps, max_conn_px, target_strokes, hard_cap_px, ok=None):
    """The target_strokes backfill of `_route_low_lift` (same rule), optionally skipping gaps whose
    connector failed the mask check (`ok`). Returns (fuse, examined) where `examined` counts the
    extra gaps the backfill tried (accepted or, with `ok`, rejected)."""
    import numpy as np
    examined = 0
    if target_strokes is not None and target_strokes >= 1:
        cap = hard_cap_px if hard_cap_px else max_conn_px * 4.0
        strokes_now = int((~fuse).sum()) + 1
        for gi in np.argsort(gaps):
            if strokes_now <= target_strokes:
                break
            if fuse[gi]:
                continue
            if gaps[gi] > cap:             # ascending -> nothing smaller left to fuse
                break
            examined += 1
            if ok is not None and not ok[gi]:
                continue
            fuse[gi] = True; strokes_now -= 1
    return fuse, examined

def _mask_checked_fuse(oriented, gaps, fuse0, max_conn_px, target_strokes, hard_cap_px, check, check_stats):
    """connector_mask_check: choose the fuse set when every fused connector must pass `check`.
    Candidates = every gap the unchecked router could use (<= max_conn_px, or <= the target cap)."""
    import numpy as np
    limit = max_conn_px + 1e-6
    if target_strokes is not None and target_strokes >= 1:
        limit = max(limit, hard_cap_px if hard_cap_px else max_conn_px * 4.0)
    cand = np.nonzero(gaps <= limit)[0]
    ends = np.array([o[-1] for o in oriented[:-1]])
    nxts = np.array([o[0] for o in oriented[1:]])
    ok = np.zeros(len(gaps), bool)
    if len(cand):
        ok[cand] = check.ok(ends[cand], nxts[cand])
    plain, _ = _target_fuse(fuse0.copy(), gaps, max_conn_px, target_strokes, hard_cap_px, None)
    fuse, examined = _target_fuse(fuse0 & ok, gaps, max_conn_px, target_strokes, hard_cap_px, ok)
    if check_stats is not None:
        n = len(oriented)
        att = int(fuse0.sum()) + examined
        check_stats.update({
            "fuses_planned_unchecked": int(plain.sum()), "fuses_attempted": att,
            "fuses_accepted": int(fuse.sum()), "fuses_rejected": att - int(fuse.sum()),
            "fuses_added_by_target_backfill": int((fuse & ~plain).sum()),
            "strokes_unchecked": n - int(plain.sum()), "strokes_checked": n - int(fuse.sum())})
        px = np.abs(ends - nxts)
        pi, fi = np.nonzero(plain)[0], np.nonzero(fuse)[0]
        ri = np.nonzero(plain & ~fuse)[0]
        m = check.measure(ends[pi], nxts[pi]); a = check.measure(ends[fi], nxts[fi])
        check_stats["unchecked_connector_mm"] = float(px[pi].sum()) * PX2MM
        check_stats["accepted_connector_mm"] = float(px[fi].sum()) * PX2MM
        check_stats["rejected_connector_mm"] = float(px[ri].sum()) * PX2MM
        if m is not None:
            check_stats["unchecked_offmask_mm"], check_stats["unchecked_spill_mm"] = m[1], m[2]
            check_stats["accepted_offmask_mm"], check_stats["accepted_spill_mm"] = a[1], a[2]
        check_stats["rejected_segments"] = [(float(ends[i].real), float(ends[i].imag),
                                             float(nxts[i].real), float(nxts[i].imag)) for i in ri]
    return fuse

# Cost caps for the nearest-endpoint walk (a parameter of _route_low_lift / _route_low_lift_doc). The walk is
# spatially indexed (cKDTree candidates, exact comparison decides): measured 1.0 s at 31k and 3.2 s at 100k strokes,
# roughly n log n. Above the cap the router is SKIPPED with a logged WARNING and a route_skipped_guard flag (never
# silent).
# The PLAIN path stays at 20000: raising it would change default single-pen output for layers of 20k-100k strokes.
# Only the two-pen KEY layer passes ROUTE_WALK_MAX_SEGS_TWOPEN.
ROUTE_WALK_MAX_SEGS_PLAIN = 20000
ROUTE_WALK_MAX_SEGS_TWOPEN = 100000


def _nn_walk_order(starts, ends, first_end):
    """Greedy nearest-endpoint walk, equal to the reference O(n^2) loop on every tested input (incl. ties);
    deterministic where the np.abs(complex) reference was not bit-stable (1-ulp differences possible):
    ds=|starts-cur|, de=|ends-cur| (used=inf), i_s/i_e = lowest-index argmin of each,
    start wins when ds[i_s] <= de[i_e]. A KD-tree only proposes candidates (all points
    within a slightly inflated radius of the tree's nearest); the choice is made by the
    comparison on those candidates (same lowest-index/start-wins rules), so ties match."""
    from scipy.spatial import cKDTree
    n = len(starts)
    if starts.dtype.kind != "c" or ends.dtype.kind != "c":
        return _nn_walk_order_bruteforce(starts, ends, first_end)
    pts = np.concatenate([starts, ends])
    xy = np.column_stack([pts.real, pts.imag])
    sx, sy, ex, ey = starts.real.copy(), starts.imag.copy(), ends.real.copy(), ends.imag.copy()
    used = np.zeros(n, bool)
    order = [(0, False)]; used[0] = True
    cur = first_end
    tree = None; tree_ids = None; built_live = 0; dead = 0
    for _ in range(n - 1):
        if tree is None or dead * 2 > built_live:
            alive = np.nonzero(~np.concatenate([used, used]))[0]
            tree_ids = alive; tree = cKDTree(xy[alive]); built_live = len(alive); dead = 0
        cx, cy = float(cur.real), float(cur.imag)
        q = (cx, cy)
        k = 4
        while True:
            kk = min(k, len(tree_ids))
            dd, ii = tree.query(q, k=kk)
            dd = np.atleast_1d(dd); ii = np.atleast_1d(ii)
            ok = ~used[tree_ids[ii] % n]
            if ok.any() or kk >= len(tree_ids):
                break
            k *= 4
        d0 = float(dd[np.argmax(ok)])
        r = d0 * (1 + 1e-9) + 1e-9
        cand = tree_ids[np.asarray(tree.query_ball_point(q, r), dtype=np.intp)]
        cand = cand[~used[cand % n]]
        cs = np.sort(cand[cand < n])
        ce = np.sort(cand[cand >= n] - n)
        # Distances by sqrt(dx*dx+dy*dy) on real arrays: per-element IEEE ops, so identical coordinates give
        # identical distances regardless of array length/alignment. (np.abs on complex arrays is NOT bit-stable
        # across alignment/lane position in numpy 1.26 -- a start==end tie can flip -- which would make the
        # walk nondeterministic between calls; found when test_twopen_keytint_connector went flaky.)
        ds = np.sqrt((sx[cs] - cx) ** 2 + (sy[cs] - cy) ** 2) if len(cs) else np.empty(0)
        de = np.sqrt((ex[ce] - cx) ** 2 + (ey[ce] - cy) ** 2) if len(ce) else np.empty(0)
        bs = float(ds.min()) if len(ds) else np.inf
        be = float(de.min()) if len(de) else np.inf
        if bs <= be:
            i, flip = int(cs[int(np.argmin(ds))]), False
        else:
            i, flip = int(ce[int(np.argmin(de))]), True
        used[i] = True; dead += 2
        order.append((i, flip))
        cur = starts[i] if flip else ends[i]
    return order


def _nn_walk_order_bruteforce(starts, ends, first_end):
    n = len(starts)
    used = np.zeros(n, bool)
    order = [(0, False)]; used[0] = True
    cur_end = first_end
    for _ in range(n - 1):
        ds = np.abs(starts - cur_end); ds[used] = np.inf
        de = np.abs(ends - cur_end);   de[used] = np.inf
        i_s, i_e = int(np.argmin(ds)), int(np.argmin(de))
        i, flip = (i_s, False) if ds[i_s] <= de[i_e] else (i_e, True)
        used[i] = True
        order.append((i, flip))
        cur_end = starts[i] if flip else ends[i]
    return order


def _route_low_lift(lines, max_conn_px, target_strokes=None, hard_cap_px=None, connector_check=None, check_stats=None, skip_info=None,
                     max_walk_segs=ROUTE_WALK_MAX_SEGS_PLAIN, stay_down_check=None, stay_down_stats=None):
    """Greedy low-lift endpoint router (Stage E, after cleanup / before output).

    Orders strokes by a nearest-endpoint walk (travel order), then FUSES
    consecutive strokes into one polyline — drawing the gap as a PEN-DOWN
    connector segment — whenever the gap is short enough. Each fuse removes one
    pen lift (two Z-moves) at the cost of a little extra XY drawing. Smallest
    gaps are fused first; if `target_strokes` is given, fusing continues past
    `max_conn_px` (bounded by `hard_cap_px`) until the target stroke count is
    met. Returns (new_lines, connectors, connector_len_px, longest_px) where
    connectors is the list of 2-pt gap segments drawn (len = connector count).

    `connector_check` (opt-in, connector_mask_check; default None = the router exactly as
    before): an object from ccplotter.stage_e_mask with .ok(A, B) -> bool per candidate. A fuse is
    allowed only where it passes; a rejected fuse stays a pen lift. The stroke ORDER is unchanged
    (the nearest-endpoint walk ignores the mask), so on/off differ only in which gaps are fused.
    `check_stats` (dict, optional) is filled with the per-layer audit.

    `stay_down_check` (opt-in "Stay down over ink", default None = the router exactly as before): an
    object from ccplotter.stage_e_mask with .ok(A, B). AFTER the normal fuse set is chosen (whatever
    max_conn_px / connector_check / target said), every gap still left as a pen lift is fused as a
    pen-down connector of ANY length iff the straight connector passes the check (it stays inside
    the layer's own ink mask). Order is unchanged. `stay_down_stats` (dict, optional) gets the audit."""
    import numpy as np
    segs = [np.asarray(l) for l in lines if l is not None and len(l) >= 1]
    if len(segs) <= 1:
        return segs, [], 0.0, 0.0
    if len(segs) > max_walk_segs:          # cost cap (see constants); never silent
        logging.getLogger(__name__).warning(
            "low-lift router SKIPPED: %d strokes > cost cap %d; strokes left unjoined",
            len(segs), max_walk_segs)
        if skip_info is not None:
            skip_info["route_skipped_guard"] = True
            skip_info["strokes"] = len(segs)
        return segs, [], 0.0, 0.0
    n = len(segs)
    starts = np.array([s[0] for s in segs])
    ends = np.array([s[-1] for s in segs])

    # ── 1. Greedy nearest-endpoint ordering (with per-stroke flip) ──
    order = _nn_walk_order(starts, ends, segs[0][-1])
    oriented = [segs[i][::-1] if f else segs[i] for (i, f) in order]

    # ── 2. gaps between consecutive strokes; choose which to fuse ──
    gaps = np.array([abs(oriented[k + 1][0] - oriented[k][-1]) for k in range(n - 1)])
    fuse = gaps <= max_conn_px + 1e-6      # epsilon: fuse gaps sitting exactly on the limit
    if connector_check is not None:
        fuse = _mask_checked_fuse(oriented, gaps, fuse, max_conn_px, target_strokes, hard_cap_px,
                                  connector_check, check_stats)
    elif target_strokes is not None and target_strokes >= 1:
        cap = hard_cap_px if hard_cap_px else max_conn_px * 4.0
        strokes_now = int((~fuse).sum()) + 1
        for gi in np.argsort(gaps):
            if strokes_now <= target_strokes:
                break
            if fuse[gi]:
                continue
            if gaps[gi] > cap:             # ascending → nothing smaller left to fuse
                break
            fuse[gi] = True; strokes_now -= 1

    if stay_down_check is not None:
        sd_ends = np.array([o[-1] for o in oriented[:-1]])
        sd_nxts = np.array([o[0] for o in oriented[1:]])
        cand = np.nonzero(~fuse)[0]
        sd_ok = np.zeros(len(gaps), bool)
        if len(cand):
            sd_ok[cand] = stay_down_check.ok(sd_ends[cand], sd_nxts[cand])
        added = ~fuse & sd_ok
        if stay_down_stats is not None:
            stay_down_stats.update({"lifts_before": int((~fuse).sum()) + 1, "candidates": int(len(cand)),
                                    "fused": int(added.sum()), "rejected": int(len(cand) - added.sum()),
                                    "fused_mm": float(gaps[added].sum()) * PX2MM,
                                    "longest_fused_mm": (float(gaps[added].max()) * PX2MM if added.any() else 0.0),
                                    "lifts_after": int((~(fuse | added)).sum()) + 1})
        fuse = fuse | added

    # ── 3. build fused polylines (concatenation = implicit pen-down connector) ──
    new_lines, connectors, conn_len, longest = [], [], 0.0, 0.0
    chain = [oriented[0]]
    for k in range(n - 1):
        if fuse[k]:
            connectors.append(np.array([oriented[k][-1], oriented[k + 1][0]]))
            chain.append(oriented[k + 1])
            g = float(gaps[k]); conn_len += g; longest = max(longest, g)
        else:
            new_lines.append(np.concatenate(chain)); chain = [oriented[k + 1]]
    new_lines.append(np.concatenate(chain))
    return new_lines, connectors, conn_len, longest

def _route_low_lift_doc(in_svg, out_svg, conn_dist_mm, target_z_moves, connectors_svg=None, bypass_layers=None, mask_ctx=None,
                       max_walk_segs=ROUTE_WALK_MAX_SEGS_PLAIN, stay_down_ctx=None):
    """Read the cleaned layout SVG, run the low-lift router per layer (never
    bridging across separate drawings), write the routed multilayer SVG, and
    return before/after routing stats. If connectors_svg is given, the bare
    connector segments are also written there (X-ray overlay for the preview —
    the segments are already fused into out_svg, this is display-only).

    bypass_layers (calibration-verbatim): a set of layer ids
    that must pass through UNROUTED -- zero fusing, zero pen-down connectors,
    regardless of conn_dist_mm/target_z_moves. Used for calibration (true_size)
    items, whose stroke topology and Z-move count IS the measurement. Default
    None/empty -> every layer is routed.

    mask_ctx (connector_mask_check, default None = nothing below
    changes): dict layer-id -> placement context (see composite). When given, every
    non-bypass layer's fuses are tested against that layer's trace mask
    (ccplotter.stage_e_mask) and the returned stats gain a `connector_mask_check`
    record; a layer without a usable mask is fail-closed (no fuses) and named.

    stay_down_ctx (opt-in "Stay down over ink", default None = nothing below changes): dict
    layer-id -> placement context (same shape as mask_ctx, but its sidecar is the layer's OWN ink
    mask). Independent of mask_ctx: the connector-distance fuses keep whatever checking they had;
    on top, every remaining pen lift whose straight connector lies wholly inside the layer's own
    ink becomes a pen-down connector of any length. Stats gain a `stay_down_over_ink` record."""
    bypass_layers = bypass_layers or set()
    doc = vp.read_multilayer_svg(str(in_svg), quantization=0.1)
    conn_px = max(0.0, float(conn_dist_mm)) / PX2MM
    before = sum(len(lc) for lc in doc.layers.values())
    target_strokes = None
    if target_z_moves and float(target_z_moves) > 0:
        target_strokes = max(1, int(math.ceil(float(target_z_moves) / 2.0)))
    # Target may push connectors past the distance setting, but never unboundedly.
    hard_cap_px = max(conn_px * 4.0, (float(conn_dist_mm) + 4.0) / PX2MM)

    out = vp.Document()
    out.page_size = doc.page_size
    all_connectors, conn_len_px, longest_px = [], 0.0, 0.0
    mc_tot, mc_layers, mc_rejected = None, {}, []
    skipped = {}
    if mask_ctx is not None:
        from ccplotter.stage_e_mask import empty_totals, build_layer_check
        mc_tot = empty_totals()
    sd_layers, sd_tot = {}, None
    if stay_down_ctx is not None:
        sd_tot = {"layers_checked": 0, "layers_failed_closed": 0, "candidates": 0, "fused": 0, "rejected": 0, "fused_mm": 0.0, "layers_failed_closed_detail": {}}
    for lid, lc in doc.layers.items():
        lines = list(lc)
        if lid in bypass_layers:
            # Verbatim pass-through: no reordering, no fusing, no connectors.
            nlc = vp.LineCollection()
            for nl in lines:
                nlc.append(nl)
            out.add(nlc, lid)
            continue
        tgt = None
        if target_strokes and before > 0:     # split the target across layers by share
            tgt = max(1, int(round(target_strokes * len(lines) / before)))
        _si = {}
        _sd_kw = {}
        if stay_down_ctx is not None:
            from ccplotter.stage_e_mask import NoMaskCheck, build_layer_check as _bl
            if lid in stay_down_ctx and lines:
                _sdchk = _bl(stay_down_ctx[lid], np.concatenate([np.asarray(l) for l in lines]))
            else:
                _sdchk = NoMaskCheck("no_trace_mask_for_layer")
            _sds = {}
            _sd_kw = {"stay_down_check": _sdchk, "stay_down_stats": _sds}
        if mask_ctx is None:
            new_lines, conns, cl, lg = _route_low_lift(lines, conn_px, tgt, hard_cap_px, skip_info=_si, max_walk_segs=max_walk_segs, **_sd_kw)
        else:
            if lid in mask_ctx and lines:
                chk = build_layer_check(mask_ctx[lid], np.concatenate([np.asarray(l) for l in lines]))
            else:
                from ccplotter.stage_e_mask import NoMaskCheck
                chk = NoMaskCheck("no_trace_mask_for_layer")
            cs = {}
            new_lines, conns, cl, lg = _route_low_lift(lines, conn_px, tgt, hard_cap_px,
                                                       connector_check=chk, check_stats=cs, skip_info=_si, max_walk_segs=max_walk_segs, **_sd_kw)
            mc_rejected.extend((lid,) + r for r in cs.pop("rejected_segments", []))
            info = chk.info()
            if not chk.available:
                if lines:      # an empty layer has nothing to bridge: not a failure worth a warning
                    mc_tot["layers_failed_closed"] += 1
                    mc_tot["layers_failed_closed_detail"][str(lid)] = chk.reason
            else:
                mc_tot["layers_checked"] += 1
            for k, v in cs.items():
                if isinstance(v, (int, float)):
                    mc_tot[k] += v
            mc_layers[str(lid)] = {**info, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in cs.items()}}
        if stay_down_ctx is not None:
            _e = {**_sdchk.info(), **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in _sds.items()}}
            sd_layers[str(lid)] = _e
            if _sdchk.available:
                sd_tot["layers_checked"] += 1
            elif lines:
                sd_tot["layers_failed_closed"] += 1
                sd_tot["layers_failed_closed_detail"][str(lid)] = _sdchk.reason
            for k in ("candidates", "fused", "rejected", "fused_mm"):
                sd_tot[k] += _sds.get(k, 0)
        if _si:
            skipped[str(lid)] = _si.get("strokes")
        nlc = vp.LineCollection()
        for nl in new_lines:
            nlc.append(nl)
        out.add(nlc, lid)
        all_connectors.extend(conns); conn_len_px += cl; longest_px = max(longest_px, lg)
    after = sum(len(lc) for lc in out.layers.values())
    conn_count = len(all_connectors)
    with atomic_open(out_svg, "w") as f:
        vp.write_svg(f, out, page_size=out.page_size, color_mode="none")
    if connectors_svg is not None:
        cdoc = vp.Document()
        cdoc.page_size = doc.page_size
        clc = vp.LineCollection()
        for seg in all_connectors:
            clc.append(seg)
        cdoc.add(clc, 1)
        with atomic_open(connectors_svg, "w") as f:
            vp.write_svg(f, cdoc, page_size=cdoc.page_size, color_mode="none")
    res = {
        "strokes_before": before, "strokes_after": after,
        "z_moves_before": 2 * before, "z_moves_after": 2 * after,
        "connector_count": conn_count,
        "connector_draw_length_mm": round(conn_len_px * PX2MM, 1),
        "longest_connector_mm": round(longest_px * PX2MM, 2),
        "connector_distance_mm": round(float(conn_dist_mm), 2),
        "target_z_moves": int(target_z_moves) if target_z_moves else 0,
    }
    if skipped:
        res["route_skipped_guard"] = True
        res["route_skipped_layers"] = skipped
    if sd_tot is not None:
        from ccplotter.stage_e_mask import DT_TOL_PX as _DT
        res["stay_down_over_ink"] = {
            "enabled": True,
            "rule": ("after routing, a pen-up travel becomes a pen-down connector, of any length, iff the straight "
                     "connector stays inside the layer's own ink mask (stage_e_mask predicate: every 0.5 trace px sample "
                     "has dt >= 0.5*fill_px - dt_tol_px); a layer with no usable mask fails closed (no stay-down)"),
            "dt_tol_px": _DT,
            "totals": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sd_tot.items()},
            "layers": sd_layers,
        }
    if mc_tot is not None:
        from ccplotter.stage_e_mask import DT_TOL_PX
        res["connector_mask_check"] = {
            "predicate": ("stitch.py: every sample (0.5 trace px) of the straight connector has bilinear "
                          "dt >= 0.5*fill_px - dt_tol_px on the final trace mask; zero-length joins exempt"),
            "dt_tol_px": DT_TOL_PX,
            "scope": "Stage E only; Stage D (vpype linemerge) joins are unchanged and unchecked",
            "totals": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in mc_tot.items()},
            "layers": mc_layers,
            "_rejected_segments_page_px": mc_rejected,
        }
    return res
