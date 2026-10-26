"""ccplotter/pfm_native.py -- Sketch Field (native "Sketch Lines" PFM), v1.

CLEAN-ROOM NOTICE: no DrawingBotV3 (DBV3) Java source was read, fetched, or
consulted -- this module was written from (a) the prose of DBV3's public
documentation (docs.drawingbotv3.com/en/latest/pfms.html, a public docs page,
not the GPL repository's src/ tree) and (b) the classical, published "greedy
darkest-line search with brightness erasure" idea that page describes (an
algorithm, not protected expression; DBV3's Java implementation of it was never
read). Parameter names and ranges follow the documented parameter model
(Line Density, Drawing Delta Angle, and so on).

Algorithm: maintain a mutable
"darkness" map D = 255 - (working-res, pre-threshold grayscale). Repeatedly:
seed a chain at one of the darkest remaining points; grow a chain of short
line segments ("squiggle"), each picked by scoring K random-angle/
random-length candidates by the SUMMED darkness they cross (coarse
np.linspace traversal, the numpy equivalent of a Bresenham walk) and keeping
the darkest; after drawing each segment, ERASE (subtract) ink along its path
in D -- this is the tone-reproduction mechanism: a region keeps attracting
lines until it has been "paid for" in drawn ink. The whole run stops once a
configured fraction of the image's total darkness has been paid off
(`sketch_field_density`) or the hard segment cap is hit
(`sketch_field_max_lines`).

Pen-width awareness: the erase width used for the tone loop is the active pen
profile's MEASURED real deposit width -- fill_mm = 0.25mm in the built-in
config._PEN_PROFILE_SEED, from an isolated-line calibration -- converted to working-res px via the same paper_w_mm/margin_mm -> px scale
every other mode's `_stroke_px_for` (ccplotter/mask.py) uses. DBV3 itself
only approximates this in raw pixels; this port "pays for" exactly the ink
the physical pen will deposit, not a guessed brush radius.

Shared code: this module does NOT touch ccplotter/mask.py or ccplotter/render.py
loader internals. render.py already loads the continuous (pre-threshold)
working-res grayscale via `_load_gray` once, before the per-mode dispatch
(the same array every other raster mode consumes before ITS OWN
binarization) -- `_sketch_field_svg` below just receives that array as its
first argument. No shared mask/render code was modified to expose it.
"""
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from ccplotter.svg_io import _write_paths_svg

# Last-run audit (mirrors the _SKETCH_AUDIT / _TRACE_AUDIT module-global
# pattern already used by ccplotter/sketch.py and ccplotter/trace_v3.py).
# render.py merges this dict into the render SVG's <out>.info.json "audit"
# block on a sketchfield run, which composite.py already surfaces as
# stats["trace_info"]["audit"], which core.py's _runs_add(...,
# "trace_info": stats.get("trace_info"), ...) call at the gcode-build site
# archives verbatim. The resolved seed and full parameter dict are therefore
# recorded in the gcode run record with no separate provenance mechanism.
_PFM_AUDIT = {}

# Global stroke-width line-art heuristic: Otsu binarize -> distance transform ->
# 2*mean(half-width) at foreground pixels. It is a DISCLOSURE number reported in
# the audit (never a gate).
STROKE_WIDTH_LINEART_MAX_PX = 8.0


def _measure_stroke_width_px(img_u8: np.ndarray) -> dict:
    """Global (whole-image, not sub-region) stroke-width measurement -- same
    caveats as any single-scalar classifier: it is not a
    calibrated distribution+gap threshold. Disclosure only."""
    _, binary = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary01 = (binary > 0).astype(np.uint8)
    area_fraction = float(binary01.mean())
    dt = cv2.distanceTransform(binary01, cv2.DIST_L2, 5)
    fg = dt[binary01 > 0]
    stroke_width_px = float(2.0 * fg.mean()) if fg.size else 0.0
    return {
        "area_fraction": round(area_fraction, 4),
        "stroke_width_px": round(stroke_width_px, 2),
        "line_art_ceiling_px": STROKE_WIDTH_LINEART_MAX_PX,
        "verdict": "line_art" if stroke_width_px < STROKE_WIDTH_LINEART_MAX_PX else "gray_mass",
    }


def _resolve_seed(raw_path: Path, opts: dict):
    """The default seed is the SHA-256 of the source file's
    BYTES (content hash) -- NEVER `_source_fingerprint` (vcache.py:30-44,
    mtime+size). A re-saved or re-copied byte-identical source must resolve
    to the identical seed and therefore identical polylines/gcode. An
    explicit `sketch_field_seed` in the request overrides the derived value
    (UI 'reroll' path); the resolved seed is always recorded either way."""
    data = Path(raw_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    content_seed = int(sha[:16], 16)  # 64 bits of the content hash
    explicit = opts.get("sketch_field_seed", None)
    if explicit is not None and str(explicit) != "":
        try:
            # Mask to 64 unsigned bits so a
            # negative seed (e.g. -5) can never reach np.random.PCG64, which
            # raises ValueError on negative input and would 500 the request.
            # A masked seed is still fully deterministic (same input -> same
            # output every time), just not the literal signed value typed in.
            return int(explicit) & ((1 << 64) - 1), sha, True
        except (TypeError, ValueError):
            pass
    return content_seed, sha, False


def _mm_to_px(mm, mm_per_px: float) -> float:
    return max(0.0, float(mm)) / max(1e-9, mm_per_px)


def _sample_segment_score(D: np.ndarray, x0, y0, x1, y1, H, W):
    """Coarse Bresenham-equivalent traversal via np.linspace + nearest-pixel
    lookup; returns (summed darkness, sample count)."""
    n = max(2, int(round(math.hypot(x1 - x0, y1 - y0))))
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    xi = np.clip(xs.astype(np.int32), 0, W - 1)
    yi = np.clip(ys.astype(np.int32), 0, H - 1)
    vals = D[yi, xi]
    return float(vals.sum()), n


def _erase_segment(D: np.ndarray, x0, y0, x1, y1, H, W, erase_px, erase_amount):
    """Subtract `erase_amount` from D along the drawn segment, at the pen's
    measured deposit width (erase_px). ROI-limited (ink "paid" is local) so
    long runs stay fast on a ~1200px working image."""
    pad = int(math.ceil(erase_px / 2.0)) + 2
    xmin = max(0, int(min(x0, x1)) - pad); xmax = min(W, int(max(x0, x1)) + pad + 1)
    ymin = max(0, int(min(y0, y1)) - pad); ymax = min(H, int(max(y0, y1)) + pad + 1)
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    roi = D[ymin:ymax, xmin:xmax]
    mask = np.zeros(roi.shape, dtype=np.uint8)
    p0 = (int(round(x0 - xmin)), int(round(y0 - ymin)))
    p1 = (int(round(x1 - xmin)), int(round(y1 - ymin)))
    thick = max(1, int(round(erase_px)))
    cv2.line(mask, p0, p1, 255, thickness=thick, lineType=cv2.LINE_8)
    idx = mask > 0
    if not np.any(idx):
        return 0.0
    before = float(roi[idx].sum())
    roi[idx] = np.maximum(0.0, roi[idx] - erase_amount)
    after = float(roi[idx].sum())
    return before - after


def _generate_sketch_lines(img_u8: np.ndarray, mm_per_px: float, opts: dict,
                            rng: np.random.Generator, fill_mm: float):
    """Core greedy darkest-seeking chain search. ALL randomness flows through
    the single `rng` (one numpy.random.Generator(PCG64(seed)) instance,
    caller-owned). Returns (polylines_px, audit_dict)."""
    H, W = img_u8.shape
    D = (255.0 - img_u8.astype(np.float64))
    total_darkness = float(D.sum())
    audit = {"H": H, "W": W, "total_darkness": round(total_darkness, 1)}
    if total_darkness <= 1.0:
        audit["stopped_reason"] = "blank_source"
        return [], audit

    # Defaults favour short-to-medium strokes with a tight angle cone and high
    # density. Long segments with a wide cone wander out of dark regions into
    # empty space, leaving sparse coverage (max_lines/density exhausted while
    # most of the image is untouched). A tighter Drawing Delta Angle chains
    # segments into coherent squiggles instead of a near-random new heading per
    # step; a high Line Density gives photo tone accuracy; the darkness floors
    # stop a chain before it crosses into bright/empty areas.
    density = float(np.clip(opts.get("sketch_field_density", 0.75), 0.01, 0.98))
    min_len_px = max(1.0, _mm_to_px(opts.get("sketch_field_min_len_mm", 1.0), mm_per_px))
    max_len_px = max(min_len_px + 1.0, _mm_to_px(opts.get("sketch_field_max_len_mm", 4.0), mm_per_px))
    angle_tests = max(1, int(opts.get("sketch_field_angle_tests", 12)))
    # angle_delta_deg IS the "angle policy" knob: 360 = every new
    # segment direction is unconstrained/random (DBV3 docs' "Unlimited"
    # cone); a smaller cone chains segments into smoother continuous
    # squiggles (DBV3 docs' `Drawing Delta Angle`).
    angle_delta_deg = float(np.clip(opts.get("sketch_field_angle_delta_deg", 25.0), 1.0, 360.0))
    squiggle_max_px = max(min_len_px, _mm_to_px(opts.get("sketch_field_squiggle_max_mm", 8.0), mm_per_px))
    max_lines = max(1, int(opts.get("sketch_field_max_lines", 20000)))
    erase_px = max(1.0, _mm_to_px(fill_mm, mm_per_px))
    erase_amount = float(np.clip(opts.get("sketch_field_erase_amount", 45.0), 1.0, 255.0))
    seed_floor = float(np.clip(opts.get("sketch_field_seed_floor", 40.0), 0.0, 255.0))
    avg_darkness_floor = float(np.clip(opts.get("sketch_field_min_avg_darkness", 18.0), 0.0, 255.0))
    seed_candidates_n = max(8, int(opts.get("sketch_field_seed_candidates", 40)))
    max_reseed_attempts = max(200, max_lines * 4)   # safety valve, pathological (near-blank) images
    max_stall_iters = max(200, max_lines)            # safety valve, chains that never grow

    delta_rad = math.radians(angle_delta_deg)

    polylines = []
    total_lines = 0
    paid_darkness = 0.0
    reseed_attempts = 0
    stall_iters = 0
    stop_reason = "max_lines_reached"
    stop_all = False

    while not stop_all and total_lines < max_lines:
        cand_x = rng.integers(0, W, size=seed_candidates_n)
        cand_y = rng.integers(0, H, size=seed_candidates_n)
        cand_vals = D[cand_y, cand_x]
        best_i = int(np.argmax(cand_vals))
        if cand_vals[best_i] < seed_floor:
            reseed_attempts += 1
            if reseed_attempts > max_reseed_attempts:
                stop_reason = "no_dark_seed_left"
                break
            continue

        cx, cy = float(cand_x[best_i]), float(cand_y[best_i])
        chain = [(cx, cy)]
        squiggle_len_px = 0.0
        prev_heading = None

        while total_lines < max_lines:
            if prev_heading is None:
                lo, hi = 0.0, 2.0 * math.pi
            else:
                lo, hi = prev_heading - delta_rad, prev_heading + delta_rad
            angles = rng.uniform(lo, hi, size=angle_tests)
            lens = rng.uniform(min_len_px, max_len_px, size=angle_tests)
            best_score, best_avg, best_end, best_ang = -1.0, -1.0, None, None
            for ang, ln in zip(angles, lens):
                ex = float(np.clip(cx + ln * math.cos(ang), 0, W - 1))
                ey = float(np.clip(cy + ln * math.sin(ang), 0, H - 1))
                score, n = _sample_segment_score(D, cx, cy, ex, ey, H, W)
                avg = score / n
                if score > best_score:
                    best_score, best_avg, best_end, best_ang = score, avg, (ex, ey), ang
            if best_end is None or best_avg < avg_darkness_floor:
                break  # chain exhausted -- reseed elsewhere
            ex, ey = best_end
            paid = _erase_segment(D, cx, cy, ex, ey, H, W, erase_px, erase_amount)
            paid_darkness += paid
            chain.append((ex, ey))
            squiggle_len_px += math.hypot(ex - cx, ey - cy)
            cx, cy = ex, ey
            prev_heading = best_ang
            total_lines += 1
            if paid_darkness >= total_darkness * density:
                stop_reason = "density_reached"
                stop_all = True
                break
            if squiggle_len_px >= squiggle_max_px:
                break  # chain complete -- reseed elsewhere

        if len(chain) >= 2:
            polylines.append(chain)
            stall_iters = 0
        else:
            stall_iters += 1
            if stall_iters > max_stall_iters:
                stop_reason = "stalled_no_candidates"
                break

    audit.update({
        "lines_drawn": total_lines,
        "chains": len(polylines),
        "paid_darkness": round(paid_darkness, 1),
        "paid_fraction": round(paid_darkness / total_darkness, 4) if total_darkness else 0.0,
        "density_target": density,
        "stopped_reason": stop_reason,
        "erase_px": round(erase_px, 3),
        "erase_amount": erase_amount,
        "min_len_px": round(min_len_px, 2), "max_len_px": round(max_len_px, 2),
        "squiggle_max_px": round(squiggle_max_px, 2),
        "angle_tests": angle_tests, "angle_delta_deg": angle_delta_deg,
    })
    return polylines, audit


def _sketch_field_svg(img_u8: np.ndarray, out_path: Path, opts: dict, raw_path: Path,
                       debug_prefix: str = None) -> Path:
    """Entry point matching the existing per-mode render.py contract (see
    `_sketch_svg` in ccplotter/sketch.py for the sibling signature). `img_u8`
    is the already-loaded, working-res, PRE-THRESHOLD grayscale render.py
    hands to every mode before its own binarization."""
    global _PFM_AUDIT
    H, W = img_u8.shape

    # Placed-size basis: composite.py computes the TRUE placed mm_per_px
    # analytically (this item's fit on its grid cell, item scale, true_size)
    # and threads it through as "mm_per_px_real" -- preferred whenever present.
    # The declared-paper formula (paper_w_mm/margin_mm, ignoring fit/scale/grid)
    # is only a fallback for callers that bypass composite.py (a direct
    # _generate_sketch_lines harness or any other bare-module caller); every
    # /api/process request supplies the real value.
    if opts.get("mm_per_px_real") is not None:
        mm_per_px = max(1e-9, float(opts["mm_per_px_real"]))
    else:
        pw_mm = float(opts.get("paper_w_mm", 180.0))
        mg_mm = float(opts.get("margin_mm", 10.0))
        mm_per_px = max(1e-6, (pw_mm - 2 * mg_mm)) / max(1.0, float(W))
    # Measured real pen deposit width (see module docstring); same key
    # ("fill_mm") every other mode reads for its own pen-width-aware pass
    # spacing. No silent default here -- routes/core.py ALWAYS supplies fill_mm
    # from the active pen profile before render_opts reaches this module, so a
    # missing key means something upstream is broken, not "use a guess".
    # Raising here (rather than falling back to a value that could disagree
    # with render.py's vpype penwidth) prevents the erase width and the
    # drawn stroke width from silently diverging -- the exact shape of the
    # hairline-preview scar.
    if "fill_mm" not in opts or opts.get("fill_mm") is None:
        raise ValueError("sketchfield: fill_mm missing from render_opts "
                          "(expected from the active pen profile)")
    fill_mm = float(opts["fill_mm"])

    seed, source_sha256, seed_explicit = _resolve_seed(raw_path, opts)
    rng = np.random.Generator(np.random.PCG64(seed))

    stroke_width_check = _measure_stroke_width_px(img_u8)  # disclosure metric (always computed)

    polylines_px, gen_audit = _generate_sketch_lines(img_u8, mm_per_px, opts, rng, fill_mm)

    stroke_px = max(0.1, fill_mm / mm_per_px)
    _write_paths_svg(out_path, W, H, stroke_px, polylines=polylines_px)

    params = {
        "mode": "sketchfield",
        "seed": int(seed), "seed_source": "explicit" if seed_explicit else "content_sha256",
        "source_sha256": source_sha256,
        "sketch_field_density": float(np.clip(opts.get("sketch_field_density", 0.75), 0.01, 0.98)),
        "sketch_field_min_len_mm": float(opts.get("sketch_field_min_len_mm", 1.0)),
        "sketch_field_max_len_mm": float(opts.get("sketch_field_max_len_mm", 4.0)),
        "sketch_field_angle_tests": int(opts.get("sketch_field_angle_tests", 12)),
        "sketch_field_angle_delta_deg": float(opts.get("sketch_field_angle_delta_deg", 25.0)),
        "sketch_field_squiggle_max_mm": float(opts.get("sketch_field_squiggle_max_mm", 8.0)),
        "sketch_field_max_lines": int(opts.get("sketch_field_max_lines", 20000)),
        "sketch_field_erase_amount": float(opts.get("sketch_field_erase_amount", 45.0)),
        "fill_mm_erase_width": fill_mm,
        "mm_per_px": round(mm_per_px, 6),
        "mm_per_px_basis": "placed_real" if opts.get("mm_per_px_real") is not None else "declared_paper_legacy",
    }
    _PFM_AUDIT = {**params, **gen_audit, "stroke_width_check": stroke_width_check}
    return out_path
