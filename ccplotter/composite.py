import cmath
import json
import math
import shlex
from pathlib import Path

import vpype as vp
import vpype_cli
from shapely.geometry import LineString as _ShapelyLineString
from shapely.geometry import Polygon as _ShapelyPolygon
from shapely.geometry import box as _shapely_box

from ccplotter.atomic_io import atomic_open, atomic_target, atomic_write_text
from ccplotter.config import (WORK_DIR, RENDER_PIPELINE_VER, PEN_MIN_X, PEN_MIN_Y,
                               PEN_MAX_X, PEN_MAX_Y, CONFIG_PATH, PX2MM,
                               PREVIEW_COLOR, XRAY_LAYER, CONNECTOR_COLOR,
                               SCALE_AWARE_V2_K, SCALE_AWARE_V2_FILL_PX_FLOOR, ILLUSTRATION_PITCH_CLAMP_FRAC,
                               NATIVE_CAP, SCALE_AWARE_V2_NATIVE_CAP, MODE_MAX_RES,
                               SIMPLIFY_FILL_MULT, FILL_SPACING_MM_FLOOR_REF_MM)
from ccplotter.cal import (load_cal, active_pen_profile, resolve_z_touch,
                           load_pen_profiles, resolve_pen_offset, resolve_z_micro)
from ccplotter.reach import active_reach_profile, validate_gcode_xy_motion, pen_reach_profile
from ccplotter.svg_io import _pdf_to_svg, _dxf_to_svg, _paper_spec, _svg_bbox_mm
from ccplotter.render import render_raster_to_svg, resolve_trace_res
from ccplotter.stats_warnings import compute_vector_stats, _route_low_lift_doc, ROUTE_WALK_MAX_SEGS_TWOPEN
from ccplotter.gcode_post import (_apply_motion_smoothing, _gcode_lifecycle_values,
                                  _microlift_gcode, _gcode_pendown_bbox,
                                  _fix_gwrite_source_comment)


def preview_pen_width_mm(render_opts, pen_profile):
    """Preview stroke width chain: core_mm (profile, when measured), else laid_mm, else
    fill_mm, else 0.15. ONE definition shared by _composite and the two-pen preview."""
    pen_mm = float((render_opts or {}).get("laid_mm")
                   or (render_opts or {}).get("fill_mm", 0.15))
    _core = (pen_profile or {}).get("core_mm")
    return float(_core) if _core else pen_mm


def _render_routed_preview(routed_svg: Path, out_svg: Path, pen_mm: float):
    """True-width routed preview command; shared contract for eager output."""
    with atomic_target(out_svg) as tmp_preview:
        vpype_cli.execute(f'read {shlex.quote(str(routed_svg))} linesort penwidth {pen_mm}mm color {PREVIEW_COLOR} write {shlex.quote(str(tmp_preview))}')


def _render_clean_preview(clean_svg: Path, out_svg: Path, pen_mm: float):
    """Former eager Clean command, retained byte-for-byte for lazy rendering."""
    with atomic_target(out_svg) as tmp_clean_preview:
        vpype_cli.execute(f'read {shlex.quote(str(clean_svg))} linesort penwidth {pen_mm}mm color {PREVIEW_COLOR} write {shlex.quote(str(tmp_clean_preview))}')


def _render_xray_preview(routed_svg: Path, connectors_svg: Path, out_svg: Path, pen_mm: float):
    """Former eager X-ray command, retained byte-for-byte for lazy rendering."""
    with atomic_target(out_svg) as tmp_xray:
        vpype_cli.execute(f'read {shlex.quote(str(routed_svg))} linesort penwidth {pen_mm}mm color {PREVIEW_COLOR}'
                          f' read -l {XRAY_LAYER} {shlex.quote(str(connectors_svg))}'
                          f' color -l {XRAY_LAYER} {CONNECTOR_COLOR} penwidth -l {XRAY_LAYER} 0.45mm'
                          f' write {shlex.quote(str(tmp_xray))}')


def _source_svg(did: str, render_opts: dict, debug_prefix: str = None, pen_id: str = None) -> Path:
    raw = [m for m in WORK_DIR.glob(f"src_{did}.*") if not (m.name.endswith("_traced.svg") or "_render_" in m.name or m.name.endswith("_render.svg"))]
    if not raw: raise FileNotFoundError(f"drawing {did} not found")
    src = raw[0]
    suf = src.suffix.lower()
    if suf == ".svg": return src
    if suf in (".pdf", ".dxf"):
        # Lossless vector conversion — independent of raster render opts, cache once.
        out = WORK_DIR / f"src_{did}_vector.svg"
        if not out.exists():
            (_pdf_to_svg if suf == ".pdf" else _dxf_to_svg)(src, out)
        return out
    import hashlib
    clean_opts = {k: v for k, v in render_opts.items() if v is not None}
    clean_opts["_pipeline_ver"] = RENDER_PIPELINE_VER   # bust cache on engine code changes
    _tex = WORK_DIR / f"src_{did}_texture.svg"
    if _tex.exists():
        clean_opts["_texture_mtime"] = int(_tex.stat().st_mtime)  # regenerated sidecar busts cache
    opt_hash = hashlib.md5(json.dumps(clean_opts, sort_keys=True).encode()).hexdigest()[:8]
    out = WORK_DIR / f"src_{did}_render_{opt_hash}.svg"
    # Purge stale renders for this drawing (every slider tweak makes a new hash);
    # keep only the current one so WORK_DIR doesn't bloat, but still cache-hit it.
    for old in WORK_DIR.glob(f"src_{did}_render_*.svg"):
        if old.name != out.name:
            try: old.unlink()
            except: pass
            try: Path(str(old) + ".info.json").unlink()
            except: pass
            try: Path(str(old) + ".mask.npz").unlink()      # connector-mask sidecar (absent unless connector_mask_check)
            except: pass
            try: Path(str(old) + ".boundary.json").unlink() # trace-boundary sidecar
            except: pass
    # Re-render on cache miss; also when debug is requested but its intermediates
    # are missing (so routing-only changes still hit the cache).
    need = not out.exists()
    if debug_prefix and not Path(f"{debug_prefix}_gray.png").exists():
        need = True
    if need:
        # pen_id: NOT part of clean_opts/opt_hash above (provenance-only,
        # audit-dict metadata) -- passed as a separate kwarg so it cannot
        # touch the cache key or generation chain.
        render_raster_to_svg(src, out, render_opts, debug_prefix=debug_prefix, pen_id=pen_id)
        # Experimental vector texture sidecar (AI backfill layer): merged AFTER
        # tracing, ONLY when explicitly enabled. Default OFF so an old leftover
        # src_<id>_texture.svg can never silently contaminate a clean run.
        tex = WORK_DIR / f"src_{did}_texture.svg"
        tex_exists = tex.exists()
        tex_enabled = bool(render_opts.get("texture_layer", False))
        tex_merged = 0
        if tex_exists and tex_enabled:
            try:
                tex_merged = _merge_texture_layer(out, tex)
            except Exception as e:
                print(f"  [texture] merge failed: {e}")
        elif tex_exists:
            print(f"  [texture] sidecar found for {did} but disabled -- NOT merged")
        ip = Path(str(out) + ".info.json")
        try:
            info = json.loads(ip.read_text()) if ip.exists() else {}
            info["texture_sidecar_exists"] = tex_exists
            info["texture_enabled"] = tex_enabled
            info["texture_strokes"] = tex_merged
            atomic_write_text(ip, json.dumps(info))
        except Exception: pass
    return out

def _merge_texture_layer(render_svg: Path, texture_svg: Path) -> int:
    """Append a source's vector texture sidecar (pebble outlines, grass dashes,
    cloud contours, in source-image px) onto its traced render SVG, which may
    be at 2x source resolution. Returns the number of texture strokes merged."""
    doc = vp.read_multilayer_svg(str(render_svg), quantization=0.4)
    lc, w, h = vp.read_svg(str(texture_svg), quantization=0.4)
    if len(lc) == 0:
        return 0
    pw = doc.page_size[0] if doc.page_size else None
    f = (pw / w) if (pw and w) else 1.0
    lc.scale(f)
    lid = max(doc.layers.keys(), default=1)
    doc.add(lc, lid)
    with atomic_open(render_svg, "w") as fh:
        vp.write_svg(fh, doc, page_size=doc.page_size, color_mode="none")
    return len(lc)

def _grid(n: int) -> tuple[int, int]:
    if n <= 1: return 1, 1
    if n <= 2: return 2, 1
    if n <= 4: return 2, 2
    if n <= 6: return 3, 2
    cols = math.ceil(math.sqrt(n))
    return cols, math.ceil(n / cols)

# ── Fill Page (cover-crop) ──
# The clip runs after routing so that it cuts the already-linked strokes.
# `item.crop` (raster-stage, see mask.py:_load_gray) is NEVER touched by this
# feature -- Fill Page is a separate, vector-stage, post-routing clip keyed off
# the `item.fill_page` boolean request field. ──

def _rotate_cw_deg(pt, center, deg):
    """Same convention as vpype's `rotate` CLI (clockwise-positive about
    `center`, LineCollection.rotate: `line *= complex(cos(a), sin(a))`).
    Used to invert composite.py's own forward paper->machine transform, so
    Fill Page's safe-region math reuses the exact sign/rotation convention
    the gcode-gen transform below (rotate then translate) already uses,
    rather than re-deriving it."""
    z = complex(pt[0] - center[0], pt[1] - center[1]) * cmath.exp(1j * math.radians(deg))
    return (z.real + center[0], z.imag + center[1])


def _rotated_extent_mm(w, h, rot_deg):
    """Axis-aligned bounding box (width, height) of a w x h rectangle after
    rotating it by rot_deg about its own center. Used to fit the per-item
    cell-fit scale against the ROTATED footprint instead of the raw
    (pre-rotation) source bbox, which would mis-size any rotation that is
    not a multiple of 180 degrees.

    Cardinal rotations (any exact multiple of 90 degrees) take a closed-form
    swap-or-no-op path with NO trigonometry: sin(radians(180)) is ~1.2e-16,
    not exactly 0, so a pure trig formula would perturb rot in {0, 180} in
    the low-order bits and could flip a downstream rounded/rasterized pixel
    -- changing the output for the common, unrotated case. Non-cardinal angles use the closed-form 4-corner
    rotate-and-bound (`|w*cos| + |h*sin|`, `|w*sin| + |h*cos|`).
    """
    r = float(rot_deg) % 360.0
    if abs(r % 90.0) < 1e-9:
        swap = (round(r / 90.0) % 2) == 1
        return (h, w) if swap else (w, h)
    rad = math.radians(r)
    rw = abs(w * math.cos(rad)) + abs(h * math.sin(rad))
    rh = abs(w * math.sin(rad)) + abs(h * math.cos(rad))
    return (rw, rh)

def _reach_polygon_paper_frame(reach_polygon, paper_w, paper_h, paper_rot, comp_off_x, comp_off_y):
    """Inverse of the forward paper->machine transform `_composite` applies
    at gcode-gen time (`rotate -o pw/2 ph/2 -- paper_rot` then `translate --
    comp_off_x -comp_off_y`): map the reach polygon (machine mm) back into
    paper-frame mm, the frame `clean_svg`/`routed_svg` actually live in."""
    center = (paper_w / 2.0, paper_h / 2.0)
    out = []
    for x, y in reach_polygon:
        px, py = x - comp_off_x, y + comp_off_y
        out.append(_rotate_cw_deg((px, py), center, -paper_rot))
    return out

def _fill_page_safe_region(paper_w, paper_h, reach_paper_poly):
    """paper INTERSECT reach (both in paper-frame mm), buffered 2mm inward: the same
    'paper INTERSECT reach INTERSECT 2mm-inset' region the existing Fit button's client-side
    `rectSafeInReach`/`fitItemInConcaveReach` already validate placements
    against, computed here authoritatively in shapely."""
    paper_poly = _shapely_box(0.0, 0.0, paper_w, paper_h)
    reach_poly = _ShapelyPolygon(reach_paper_poly)
    if not reach_poly.is_valid:
        reach_poly = reach_poly.buffer(0)
    inter = paper_poly.intersection(reach_poly)
    if inter.is_empty:
        return None
    safe = inter.buffer(-2.0, join_style=2)  # 2mm inward, mitred (square) corners
    if safe.is_empty:
        return None
    if safe.geom_type == "MultiPolygon":
        safe = max(safe.geoms, key=lambda g: g.area)
    return safe if safe.geom_type == "Polygon" else None

def _paper_keepout_safe_region(paper_w, paper_h, keepout, ink_radius_mm=0.0):
    """Return the drawable paper-frame region after an explicit physical-paper
    keepout. Currently this is the Kokuyo B5 26-hole strip: official punch
    geometry is 5.5mm diameter, 9.5mm pitch, centers nominally 6.25mm (measured 4.0-4.5mm) from the punched
    edge, so a 10mm strip covers the 9mm hole envelope with 1mm margin.
    Geometry is centerlines, so the drawable boundary is pulled back by the
    resolved deposited-stroke radius: ink cannot enter the advertised strip.

    Two mutually exclusive placements:
      - `edge='front'`: the paper frame's BOTTOM edge (paper_y in
        [paper_h-mm, paper_h]) maps to the plotter front and is excluded
        (holes at the plotter front).
      - `edge='back'`: the paper frame's TOP edge (paper_y in [0, mm]) maps to
        the plotter back and is excluded (holes at the plotter back).

    Both edges are supported; the caller's default is 'front'. The per-pen
    `reach_clamp` in ccplotter/reach.py handles an angled pen mount's
    unreachable datum without moving the paper."""
    if not isinstance(keepout, dict) or keepout.get("profile") != "kokuyo_b5_26h":
        return None
    edge = str(keepout.get("edge", "front"))
    if edge not in ("front", "back"):
        raise ValueError("unsupported paper keepout edge (must be 'front' or 'back')")
    mm = float(keepout.get("mm", 10.0))
    if not (9.0 <= mm < float(paper_h)):
        raise ValueError(f"Kokuyo B5 {edge} keepout must be at least 9mm and smaller than the paper height")
    ink_radius_mm = max(0.0, float(ink_radius_mm))
    if mm + ink_radius_mm >= float(paper_h):
        raise ValueError(f"Kokuyo B5 {edge} keepout plus ink radius must fit on the paper")
    if edge == "front":
        return _shapely_box(0.0, 0.0, float(paper_w), float(paper_h) - mm - ink_radius_mm)
    return _shapely_box(0.0, mm + ink_radius_mm, float(paper_w), float(paper_h))

def _clip_layer_to_region(lc, region_polygon_mm, sliver_floor_mm=0.01):
    """Clip every line in a vpype LineCollection (px, PX2MM/mm) to a shapely
    polygon (mm). Each line is intersected independently (a clean geometric
    cut, never a re-link/heal) and any resulting piece under
    `sliver_floor_mm` is dropped (clip-created sub-floor slivers)."""
    import numpy as np
    from shapely.prepared import prep
    new_lc = vp.LineCollection()
    kept = dropped = shattered = 0
    prepared = prep(region_polygon_mm)
    for line in lc:
        if len(line) < 2:
            dropped += 1
            continue
        pts_mm = [(p.real * PX2MM, p.imag * PX2MM) for p in line]
        geom = _ShapelyLineString(pts_mm)
        # FAST PATH — and the whole reason this is not a bare `.intersection()`.
        # GEOS *nodes* its overlay inputs: a self-intersecting polyline (every
        # hatch chain, every closed contour, every router-fused stroke) comes
        # back from .intersection() split at each self-crossing EVEN WHEN IT
        # LIES ENTIRELY INSIDE THE REGION. That silently multiplied stroke
        # count ~6x (and pen lifts with it) for every clipped job. A line the
        # region covers is not clipped at all, so it passes through verbatim.
        if prepared.covers(geom):
            new_lc.append(line)
            kept += 1
            continue
        if not prepared.intersects(geom):
            dropped += 1
            continue
        # Genuine straddler: clip segment-by-segment (a 2-point segment is
        # always simple, so it cannot be noded apart) and re-stitch the
        # consecutive survivors, so the line only breaks where it actually
        # leaves the region.
        runs, cur = [], []
        for a, b in zip(pts_mm, pts_mm[1:]):
            piece = _ShapelyLineString([a, b]).intersection(region_polygon_mm)
            if piece.is_empty:
                sub = []
            elif piece.geom_type == "LineString":
                sub = [piece]
            elif piece.geom_type in ("MultiLineString", "GeometryCollection"):
                sub = [g for g in piece.geoms if g.geom_type == "LineString"]
            else:
                sub = []
            for g in sub:
                coords = list(g.coords)
                if len(coords) < 2:
                    continue
                if cur and abs(complex(*cur[-1]) - complex(*coords[0])) <= 1e-9:
                    cur.extend(coords[1:])
                else:
                    if len(cur) >= 2:
                        runs.append(cur)
                    cur = list(coords)
            if not sub and len(cur) >= 2:
                runs.append(cur); cur = []
            elif not sub:
                cur = []
        if len(cur) >= 2:
            runs.append(cur)
        wrote_any = False
        for coords in runs:
            arr = np.array([complex(x / PX2MM, y / PX2MM) for x, y in coords])
            if float(_ShapelyLineString(coords).length) < sliver_floor_mm:
                continue
            new_lc.append(arr)
            kept += 1
            wrote_any = True
        if wrote_any:
            shattered += 1
        else:
            dropped += 1
    return new_lc, {"kept": kept, "dropped": dropped, "clipped_lines": shattered}

def _apply_fill_page_clips(svg_path, clips):
    """Clip specific layers of an already-written, already-ROUTED multilayer
    SVG to each item's Fill Page crop polygon. Must run AFTER
    `_route_low_lift_doc` (Stage E), never before it: the router links strokes
    and the clip must cut the already-linked result."""
    if not clips:
        return {}
    doc = vp.read_multilayer_svg(str(svg_path), quantization=0.1)
    report = {}
    for lid, region in clips.items():
        if lid not in doc.layers or region is None or region.is_empty:
            continue
        new_lc, r = _clip_layer_to_region(doc.layers[lid], region)
        doc.layers[lid] = new_lc
        report[str(lid)] = r
    with atomic_open(svg_path, "w") as f:
        vp.write_svg(f, doc, page_size=doc.page_size, color_mode="none")
    return report

def _apply_global_clip(svg_path, region):
    """Clip every routed layer to one paper-level safe region after routing.
    This is deliberately post-router: no connector can bridge a physical paper
    keepout, and preview/G-code consume the same clipped SVG."""
    if region is None or region.is_empty:
        return {}
    doc = vp.read_multilayer_svg(str(svg_path), quantization=0.1)
    report = {}
    for lid, lc in list(doc.layers.items()):
        new_lc, r = _clip_layer_to_region(lc, region)
        doc.layers[lid] = new_lc
        report[str(lid)] = r
    with atomic_open(svg_path, "w") as f:
        vp.write_svg(f, doc, page_size=doc.page_size, color_mode="none")
    return report

def _realised_pitch_record(item_id, item_opts, analytic_equiv, svg, final_scale):
    """Per-item REALISED pass pitch: READOUT ONLY -- reads finished numbers,
    changes no stroke, no gcode byte, no cache key.

        realised_pitch_mm = fill_px_final * mm_per_px_final

    fill_px_final = the fill pitch (working-grid px) the trace ACTUALLY ran at, i.e. after the
    1.2 px representability floor in trace_v3:
      * analytic path (scale_aware / scale_aware_v2 / fill_spacing_mm): max(1.2, the fill_px
        composite.py handed to trace_v3 as item_opts["fill_px"]);
      * legacy path (no fill_px override): the trace's own audit fill_px, read from the
        item's render info.json (2-decimal rounded);
    mm_per_px_final = final_scale * PX2MM, where final_scale is `s` at the placement site in
    _composite (s = fit * item.scale, fit from the FINAL traced SVG's ink bbox -- the same `s`
    the vpype `scale -l L s s` transform applies and that is echoed as final_s). One working px
    of the traced SVG becomes PX2MM * s mm on the page.

    It deliberately does NOT use (a) the PRELIMINARY analytic scale (s0 / mm_per_px_real, derived
    from a first trace pass at the unboosted resolution; measured up to ~1.36x off for BIC
    bwphoto) nor (b) trace_v3's fill_mm/fill_px back-solve (wrong
    whenever fill_px was not derived from fill_mm). Returns None for anything that is not a
    raster item traced by the tracev3 engine (vector imports, calibration true_size items are
    filtered by the caller)."""
    try:
        if item_opts.get("mode") != "tracev3" or "_render_" not in Path(svg).name:
            return None
        audit_fill_px = None
        try:
            _ip = Path(str(svg) + ".info.json")
            if _ip.exists():
                _a = (json.loads(_ip.read_text()).get("audit") or {})
                if _a.get("fill_px") is not None:
                    audit_fill_px = float(_a["fill_px"])
        except Exception:
            audit_fill_px = None
        pre_clamp = None
        if analytic_equiv is not None and item_opts.get("fill_px") is not None:
            pre_clamp = float(item_opts["fill_px"])
            fill_px_final, src = max(1.2, pre_clamp), "composite item_opts.fill_px (analytic path), floored at 1.2"
        elif audit_fill_px is not None:
            fill_px_final, src = audit_fill_px, "trace audit fill_px (render info.json)"
        else:
            return None
        mm_per_px_final = float(final_scale) * PX2MM
        pitch = fill_px_final * mm_per_px_final
        return {
            "id": str(item_id),
            "realised_pitch_mm": round(pitch, 4),
            "fill_px_final": round(fill_px_final, 4),
            "mm_per_px_final": round(mm_per_px_final, 6),
            "final_scale": round(float(final_scale), 9),
            "fill_px_source": src,
            "fill_px_audit": audit_fill_px,
            "floor_binds": bool(pre_clamp is not None and pre_clamp < 1.2),
            "fill_mm_target": (round(float(item_opts["fill_mm"]), 6) if item_opts.get("fill_mm") is not None else None),
            "scale_source": "composite._composite placement s = fit*item.scale (final traced SVG bbox); x PX2MM",
        }
    except Exception:
        return None


def _composite(items, paper_w, paper_h, margin, output_path, z_up, z_down, draw_speed, travel_speed, plunge_speed, lift_speed, offset_x=None, offset_y=None, render_opts=None, paper_rot=0.0, generate_gcode=True, connector_distance=0.0, target_z_moves=None, microlift=True, debug=False, end_policy="release", quiet_fans=True, motion_smoothing="off", pen_profile=None, pen_id=None, paper_keepout=None, preview_reach_check=False, connector_mask_check=False, prelim_res_basis=None, stay_down_over_ink=False):
    # pen_profile: the resolved pen profile this job is plotting with.
    # Defaults to None -> active_pen_profile(). Callers that already resolved a
    # specific profile (e.g. routes/core.py honouring an explicit pen_id
    # override) pass it through here instead of letting this function read
    # the ambient ACTIVE profile a second time -- see ccplotter/cal.py
    # resolve_pen_profile() and scripts/replay_harness.py.
    if pen_profile is None:
        pen_profile = active_pen_profile()
    # pen_id (provenance-only): the RESOLVED pen id string this job actually
    # plotted with, threaded through to _source_svg/render_raster_to_svg
    # purely so it can be stamped into the tracev3 audit dict alongside
    # fill_px/mm_per_px/levels/passes. NEVER added to render_opts/item_opts
    # (which feed _source_svg's cache-key hash) -- passed as a separate
    # kwarg all the way down so this cannot touch the generation chain or
    # cache keying. Defaults to the ambient active profile id, matching
    # pen_profile's own fallback above.
    if pen_id is None:
        pen_id = load_pen_profiles().get("active")
    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()
    z_touch, z_touch_source = resolve_z_touch(cal, pen_profile)
    print(f"  [z] z_touch_abs={z_touch:.2f}mm (source={z_touch_source})")
    reach = active_reach_profile(cal)
    # Per-pen reach clamp: some mounts physically foul the printer well inside
    # the machine's own nozzle envelope (on the author's machine a 45-degree pen
    # mount collides above nozzle Y 201.5; a bare refill has no such limit).
    # pen_reach_profile() intersects the global polygon with pen_profile's
    # optional `reach_clamp` and is CONSERVATIVE BY CONSTRUCTION: a profile
    # with no `reach_clamp` gets this exact `reach` object back unchanged, so
    # that pen's offsets, park point, preview, and the final
    # validate_gcode_xy_motion call below are unaffected. See ccplotter/reach.py
    # for the clamp/attribution details.
    reach = pen_reach_profile(reach, pen_profile)
    # Pen↔nozzle XY offset: the pen tip sits at nozzle + (pen_off_x, pen_off_y) in
    # machine coords. G-code commands the NOZZLE, so to land the PEN at the intended
    # position we drive the nozzle to (intended - pen_off). offset_x/offset_y are the
    # intended PEN-target machine offsets from the UI; comp_off_* is the nozzle offset.
    offset_x = reach["bbox"][0] if offset_x is None else float(offset_x)
    offset_y = reach["bbox"][1] if offset_y is None else float(offset_y)
    pen_off_x, pen_off_y = resolve_pen_offset(cal, pen_profile)
    comp_off_x, comp_off_y = float(offset_x) - pen_off_x, float(offset_y) - pen_off_y
    # Park/preamble origin is CLAMPED to the reachable area. The paper may be
    # placed anywhere on the build plate (free placement), so its raw front-left
    # corner can be off-reach; clamping the park point means the approach + park
    # moves never drive the head into the front/edge limits. (Drawing geometry
    # still uses the true offset; off-reach drawing is caught by build_warnings.)
    park_x, park_y = reach["safe_park_point"]
    # Dedicated TRAVEL clearance for the big cross-plate approach/park moves (doc
    # start/end), separate from the per-stroke z_up so we can lift high there
    # without adding Z-wear to every stroke. z_travel = mm above pen-touch (cal-
    # tunable); kept ≥ the per-stroke full lift so the initial sweep clears warps.
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))
    motion_smoothing = str(motion_smoothing).lower()
    if motion_smoothing not in ("off", "zoned"):
        raise ValueError("motion_smoothing must be 'off' or 'zoned'")
    vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({"z_up": float(z_up), "z_down": float(z_down), "z_up_abs": z_touch + float(z_up), "z_down_abs": z_touch + float(z_down), "z_touch_abs": z_touch, "z_travel_abs": z_travel_abs, "draw_speed": float(draw_speed), "travel_speed": float(travel_speed), "plunge_speed": float(plunge_speed), "lift_speed": float(lift_speed), "z_auto": "1", "origin_x": park_x, "origin_y": park_y, "motion_smoothing_restore": "M204 S5000\n" if motion_smoothing == "zoned" else "", **_gcode_lifecycle_values(end_policy, quiet_fans)})
    spec, n = _paper_spec(paper_w, paper_h), len(items)
    # vpype `pagesize` NORMALIZES a "WxH" spec to PORTRAIT (smaller dim = width),
    # so a landscape sheet (w>h) silently became a portrait page — the wide artwork
    # was then placed into a too-narrow page and clipped. Force orientation with -l.
    page_orient = '-l ' if paper_w > paper_h else ''
    cols, rows = _grid(n)
    pageW, pageH = paper_w - 2 * margin, paper_h - 2 * margin
    cellW, cellH = pageW / cols, pageH / rows
    reads, xforms, layer_of, item_geometry = [], [], {}, {}
    realised_pitch_items = []   # per-item realised pass pitch (readout only, see _realised_pitch_record)
    item_svgs = {}   # layer -> render SVG
    first_svg = None
    scale_aware = bool((render_opts or {}).get("scale_aware", False))
    # Scale-adaptation v2 (opt-in, independent flag). Does NOT require
    # scale_aware=True; setting scale_aware_v2 alone is sufficient to enter the
    # analytic-fill_px branch below with the v2 target instead of v1's
    # spec-exact one. scale_aware=True alone (v1) and neither flag (legacy)
    # behave as before.
    scale_aware_v2 = bool((render_opts or {}).get("scale_aware_v2", False))
    # prelim_res_basis (opt-in, PRIVATE/experimental, scale_aware_v2 ONLY, no UI): the
    # v2 resolution boost below takes base_res = the REQUESTED resolution, but the preliminary trace
    # ran at min(requested, MODE_MAX_RES["tracev3"]) (render.resolve_trace_res), so actual_boost and
    # the res_basis_boost it feeds are under-counted by requested/effective (1.364x for 3000/2200).
    # None/"requested" (the default) keeps the code path unchanged. "prelim_effective": base_res = the preliminary trace's effective resolution (the
    # 1.2 px floor is still applied in the preliminary frame). "final_solve": additionally solves the
    # floor in the FINAL frame (target pitch -> boost so fill_px reaches the 3 px floor, else the cap).
    # Passed as a kwarg, NOT put in render_opts/item_opts: the preliminary trace's cache key (and every
    # other trace key) stays identical to a flag-off run.
    if prelim_res_basis in (None, "", "requested"):
        prelim_res_basis = None
    elif prelim_res_basis not in ("prelim_effective", "final_solve"):
        raise ValueError(f"unknown prelim_res_basis {prelim_res_basis!r}")
    # mm-spacing (opt-in). Independent of scale_aware/scale_aware_v2/laid_mm:
    # when set, fill_spacing_mm names the REQUESTED real pass pitch directly
    # (mm), not a multiplier or a profile-measured target that silently
    # overrides it. (scale_aware_v2's laid_mm path ignores the fill_mm argument
    # entirely once a measured laid_mm exists on the pen profile, so a fill_mm
    # sweep would leave fill_px unchanged.) Absent (None, the default) this flag
    # changes nothing -- every predicate below is
    # `... or fill_spacing_mm is not None`, so with it unset the entry
    # conditions and all downstream branches are unchanged.
    fill_spacing_mm = (render_opts or {}).get("fill_spacing_mm")
    if fill_spacing_mm is not None:
        fill_spacing_mm = float(fill_spacing_mm)
    # Fill Page (cover-crop): computed ONCE per request, in paper-frame mm,
    # reused by every item that requests it. Untouched (None/empty) when no item
    # requests fill_page.
    fill_page_safe_region = None
    # Reach shortfall: _fill_page_safe_region computes paper INTERSECT reach
    # correctly, but without this the user is not told HOW MUCH of the page that
    # leaves (e.g. 77.8mm of a 182mm sheet, so "Fill Page" would draw ~43% of the
    # page height silently). This dict carries the facts build_warnings needs to
    # say so loudly; None/absent when no item requests fill_page.
    fill_page_reach_stats = None
    if any(bool(it.get("fill_page")) for it in items):
        reach_paper_poly = _reach_polygon_paper_frame(reach["polygon"], paper_w, paper_h, paper_rot, comp_off_x, comp_off_y)
        fill_page_safe_region = _fill_page_safe_region(paper_w, paper_h, reach_paper_poly)
        if fill_page_safe_region is not None:
            sx0, sy0, sx1, sy1 = fill_page_safe_region.bounds
            safe_w, safe_h = sx1 - sx0, sy1 - sy0
            fill_page_reach_stats = {
                "paper_w_mm": round(float(paper_w), 3), "paper_h_mm": round(float(paper_h), 3),
                "safe_w_mm": round(safe_w, 3), "safe_h_mm": round(safe_h, 3),
                "height_ratio": round(safe_h / float(paper_h), 6) if paper_h else 0.0,
            }
        else:
            fill_page_reach_stats = {
                "paper_w_mm": round(float(paper_w), 3), "paper_h_mm": round(float(paper_h), 3),
                "safe_w_mm": 0.0, "safe_h_mm": 0.0, "height_ratio": 0.0,
            }
    fill_page_clips = {}
    # SVG geometry describes deposited-stroke centerlines. Pull the boundary
    # inward by half the resolved solid-fill width so the physical ink edge
    # cannot cross into the advertised no-ink strip around the Kokuyo holes.
    paper_keepout_ink_radius_mm = 0.5 * max(
        0.0, float((render_opts or {}).get("fill_mm", pen_profile.get("fill_mm", 0.15)))
    )
    paper_keepout_region = _paper_keepout_safe_region(
        paper_w, paper_h, paper_keepout, paper_keepout_ink_radius_mm
    )
    # Calibration-verbatim layers. A calibration plot's geometry (patch
    # spacings, ladder rungs, crosshatch pitch) IS the measurement -- true_size
    # pins its SCALE, but the density-adaptation and low-lift-routing stages
    # could still rewrite its stroke topology after placement. Any item flagged
    # true_size is tracked here by layer id and used below to (a) skip the
    # scale_aware_v2/scale_aware analytic fill_px branch entirely for that item
    # (it only matters for tracev3 raster sources anyway -- calibration imports
    # are always already-vector .svg, so _source_svg never consults
    # item_opts["fill_px"] for them -- but the branch would compute analytics
    # that misfire the resolution-boost saturation warning against a target that
    # was never rescaled), (b) keep that item's layer OUT of the linemerge -l
    # selector (below, Stage D) so its strokes are never fused/jogged at all,
    # and (c) keep it out of the low-lift connector router (below, Stage E) so
    # it gets zero pen-down connectors. Empty when no item requests true_size
    # (the common case), in which case Stage D/E run on every layer.
    verbatim_layers = set()
    # connector_mask_check (opt-in): per-layer placement context for the Stage E mask test.
    # None when off -> nothing below this line and nothing at Stage E changes.
    _mask_ctx = {} if connector_mask_check else None
    # stay_down_over_ink (opt-in, default OFF): the same placement context, but its Stage E use is the
    # "stay down over ink" pass (any-length pen-down connectors inside the layer's own ink). None when off.
    _sd_ctx = {} if stay_down_over_ink else None
    for idx, item in enumerate(items):
        L = idx + 1
        item_opts = (render_opts or {}).copy()
        if "crop" in item: item_opts["crop"] = item["crop"]
        is_verbatim_item = bool(item.get("true_size"))
        if is_verbatim_item:
            verbatim_layers.add(L)
        # Zigzag fill_mode. Reuses this SAME verbatim_layers mechanism the
        # true_size calibration items use (comment above: "(b) keep that item's
        # layer OUT of the linemerge -l selector ... (c) keep it out of the
        # low-lift connector router") rather than inventing a second one.
        # fill_connect="none" promises no fabricated pen-down connectors;
        # without this, Stage D's unconditional `linemerge --tolerance
        # {vec_merge}mm` (default 0.8mm, larger than any realistic fill pitch)
        # silently reconnects adjacent zigzag run endpoints, and Stage E's
        # connector router is a second, independent path to the same hazard.
        # Deliberately NOT folded into `is_verbatim_item` -- that flag also
        # disables the scale_aware/scale_aware_v2 analytic-fit branch and forces
        # fit=1.0 (true_size-specific), neither of which a zigzag item should lose.
        _is_zigzag_item = (item_opts.get("mode") == "tracev3" and
                            str(item_opts.get("fill_mode", "contour")).lower() == "zigzag")
        if _is_zigzag_item:
            verbatim_layers.add(L)
        # fill_angle_deg is defined in MACHINE frame, but _zigzag_fill runs in
        # IMAGE frame -- item["rot"] (read below as `r`, applied as a vpype layer
        # rotation) and paper_rot (the whole-page rotation applied once at
        # gcode-gen) both get applied to this item's geometry AFTER tracing.
        # Back-compute the image-frame angle to send to trace_v3 so a
        # machine-frame request survives both rotations. Sign matches
        # `_rotate_cw_deg`'s clockwise-positive convention -- the same one
        # item["rot"]/paper_rot both use downstream.
        if _is_zigzag_item:
            _fill_angle_machine = float(item_opts.get("fill_angle_deg", 0.0) or 0.0)
            _fill_frame_offset = float(item.get("rot", 0.0)) + float(paper_rot)
            item_opts["fill_angle_deg"] = _fill_angle_machine - _fill_frame_offset
            # Per-region mode evaluates candidate angles per component and must
            # apply the SAME machine<->image offset this fixed-angle path just
            # used, per candidate, to check the pen-coverage constraint (see
            # trace_v3's `_zigzag_fill_per_region` docstring). Added ONLY when the
            # mode is on, so flag-off item_opts/opt_hash is untouched.
            if str(item_opts.get("fill_angle_mode", "fixed")).lower() == "per_region":
                item_opts["_fill_frame_offset_deg"] = _fill_frame_offset
        # connector_mask_check (opt-in): ask trace_v3 to persist this item's final trace mask
        # beside its SVG. Private key, set only under the flag, so the off-run opt_hash is untouched.
        if ((connector_mask_check or stay_down_over_ink) and not is_verbatim_item and not _is_zigzag_item
                and item_opts.get("mode") == "tracev3"):
            item_opts["_stage_e_mask"] = True
        # Debug intermediates only for the first (primary) drawing.
        dbg = str(WORK_DIR / "debug") if (debug and idx == 0) else None
        col, row = idx % cols, idx // cols
        ccx0, ccy0 = margin + col * cellW + cellW / 2, margin + row * cellH + cellH / 2
        if (not is_verbatim_item and item_opts.get("mode") == "tracev3"
                and (item_opts.get("core_fill") in ("hybrid", "rows", "outline_rows") or item_opts.get("edge_inset_mm")
                     or item_opts.get("stitch_mm"))):
            # Opt-in core_fill / edge_inset_mm are mm-denominated, but trace_v3's own
            # mm_per_px is the fill_mm/fill_px back-solve (wrong for pens with a
            # measured laid_mm) and the analytic placed-scale below is a PRELIMINARY-pass
            # estimate that is off by up to ~1.4x for bwphoto (measured: row pitch in the routed gcode matched the placed scale, not the
            # estimate). So hand trace_v3 the placement INPUTS (the same ones the
            # final fit below uses) via a SEPARATE private key that nothing else
            # reads; trace_v3 combines them with its own final mask bbox, which is
            # exactly what the final fit sees. Only set when a flag is active, so
            # flag-off opt_hash / audit / bytes are untouched.
            item_opts["_place_cell"] = [round(float(cellW), 6), round(float(cellH), 6),
                                        float(item.get("rot", 0.0)), float(item.get("scale", 1.0)),
                                        bool(item.get("true_size"))]
        analytic_equiv = None
        if not is_verbatim_item and (scale_aware or scale_aware_v2 or fill_spacing_mm is not None) and item_opts.get("mode") == "tracev3":
            # Scale-aware fill_px (opt-in). Chicken/egg:
            # the real placed scale `s` needs the traced SVG's own ink bbox
            # (nw/nh below), but fill_px must be known BEFORE tracing. Resolve
            # with a cheap PRELIMINARY pass using the legacy (declared-paper)
            # fill_px purely to learn nw/nh at this working resolution, derive
            # the analytic placed scale from the SAME fit formula composite.py
            # uses below, convert to a real fill_px, and re-render with that
            # fill_px override for the FINAL pass (below, unchanged code path).
            # Perf cost: one extra trace pass per tracev3 item when this flag is
            # on; the cache key differs (fill_px is now part of render_opts),
            # so differently-sized placements of the same source stop sharing a
            # cached trace once this flag is on.
            prelim_svg = _source_svg(str(item["id"]), item_opts, debug_prefix=None, pen_id=pen_id)
            pnb = _svg_bbox_mm(prelim_svg)
            # prelim_res_basis: the resolution the PRELIMINARY trace actually ran at, from render.py's
            # resolution policy (single source; no literal). Read here, before any boost below edits
            # item_opts["resolution"] / ["_tracev3_res_cap_override"]. Not evaluated when the flag is off.
            _prelim_req_res, _, _prelim_eff_res = resolve_trace_res(item_opts) if prelim_res_basis else (None, None, None)
            if pnb:
                pnw, pnh = max(pnb[2] - pnb[0], 1e-3), max(pnb[3] - pnb[1], 1e-3)
                # Fit against the ROTATED footprint, not the raw source bbox --
                # item["rot"] is read here, not only at the final placement
                # site below. pnw/pnh themselves stay source-frame (used for
                # reporting/mm_per_px derivation elsewhere unchanged).
                prot_w, prot_h = _rotated_extent_mm(pnw, pnh, item.get("rot", 0.0))
                fit0 = min(cellW / prot_w, cellH / prot_h) * 0.9
                s0 = fit0 * float(item.get("scale", 1.0))
                fill_mm = float(item_opts.get("fill_mm", 0.15))
                mm_per_px_real = max(1e-9, s0 * PX2MM)
                v2_res_boost = None
                spacing_res_boost = None
                # mm-spacing: takes priority over scale_aware_v2/legacy when
                # set, because it names the pitch directly instead of deriving
                # it from fill_mm (via K) or a profile's laid_mm. The 1.2px
                # floor is the same real limit trace_v3.py already enforces
                # (a mask pixel is a mask pixel); only the SOURCE of
                # fill_px_real changes here, not that floor.
                mm_spacing_clamp_binds = False
                fill_spacing_mm_precision_risk = None
                # mm-spacing PRECISION floor, distinct from the 1.2px
                # representability floor above. Without a resolution boost on
                # this path, the production ladder (0.14-0.25mm requested)
                # produced fill_px_real in the 1.44-2.58px range, and trace_v3.py's
                # dt-band level-set generator places pass contours via
                # cv2.findContours on an INTEGER-PIXEL raster mask -- below ~3
                # working px per level, contour placement quantizes to the pixel
                # grid and realized pitch stops tracking the request, including
                # inverting it (0.20mm requested landed TIGHTER than 0.17mm
                # requested, measured from gcode). This is the SAME floor
                # scale_aware_v2 already gates on (SCALE_AWARE_V2_FILL_PX_FLOOR,
                # see "Working-resolution floor" in config.py): re-running the same
                # ladder at a placement/resolution where fill_px_real clears 3px
                # gives monotonic realized pitch at +1..+6% error.
                #
                # The boost is sized against FILL_SPACING_MM_FLOOR_REF_MM, NOT
                # against THIS call's own fill_spacing_mm value -- deliberately,
                # so the working resolution (hence mm_per_px_real, hence the
                # traced mask and dropped_components) does not vary with the
                # requested spacing itself. A PER-REQUEST boost (mirroring
                # scale_aware_v2's own formula, keyed on this call's
                # fill_px_real_raw) would couple the mask to the requested
                # spacing: different spacings would silently produce different
                # masks. Sizing against one fixed reference means every request
                # from FILL_SPACING_MM_FLOOR_REF_MM up to whatever coarser value
                # is asked for shares one working resolution for a given
                # placement, so tone stays independent of spacing
                # (dropped_components is identical across the ladder).
                if fill_spacing_mm is not None:
                    fill_px_real_raw = fill_spacing_mm / mm_per_px_real
                    fill_px_real = max(1.2, fill_px_real_raw)
                    mm_spacing_clamp_binds = bool(fill_px_real_raw < 1.2)
                    _ref_px_raw = FILL_SPACING_MM_FLOOR_REF_MM / mm_per_px_real
                    if _ref_px_raw < SCALE_AWARE_V2_FILL_PX_FLOOR:
                        base_res = int(item_opts.get("resolution", 1000))
                        needed_boost = SCALE_AWARE_V2_FILL_PX_FLOOR / max(_ref_px_raw, 1e-6)
                        boosted_res = min(int(round(base_res * needed_boost)), SCALE_AWARE_V2_NATIVE_CAP)
                        actual_boost = boosted_res / max(base_res, 1)
                        if actual_boost > 1.0:
                            item_opts["resolution"] = boosted_res
                            item_opts["_tracev3_res_cap_override"] = boosted_res
                            fill_px_real = fill_px_real * actual_boost
                        spacing_res_boost = {
                            "base_res": base_res, "needed_boost": round(needed_boost, 4),
                            "boosted_res": boosted_res, "actual_boost": round(actual_boost, 4),
                            "reference_mm": FILL_SPACING_MM_FLOOR_REF_MM,
                            "fill_px_real_pre_boost": round(fill_px_real_raw, 6),
                        }
                    # Honest per-request readout: even after the fixed
                    # reference-sized boost above, THIS request can still land
                    # under the precision floor if fill_spacing_mm is finer
                    # than FILL_SPACING_MM_FLOOR_REF_MM, or if
                    # SCALE_AWARE_V2_NATIVE_CAP prevented the full boost from
                    # firing. Reporting-only -- does not change fill_px_real.
                    fill_spacing_mm_precision_risk = bool(fill_px_real < SCALE_AWARE_V2_FILL_PX_FLOOR)
                elif scale_aware_v2:
                    # v2 target: real pass spacing = k * fill_mm, where k is a
                    # CHOSEN constant bounded by paper evidence, not derived to
                    # reproduce any one legacy density -- see
                    # config.SCALE_AWARE_V2_K's basis comment.
                    #
                    # MEASURED-PEN TARGET. When the profile carries a MEASURED laid
                    # line width, target that directly instead of
                    # SCALE_AWARE_V2_K x fill_mm.
                    #
                    # Why: K is 0.75, so `fill_mm` never commands the pitch it
                    # names, and the residual quantisation moves it again.
                    # Passes at exactly the laid width just touch: no geometric
                    # gap, no overdraw. In a deposition sweep (120 patches on
                    # paper, one pen), coverage was 96.9% at just-touching, 92.0%
                    # at 1.46x and 82.2% at 2.91x overlap -- overdrawing wet gel
                    # lifts ink rather than adding it.
                    #
                    # GATED ON `laid_mm`, which only exists on a pen someone has
                    # actually measured. Every profile without it takes the
                    # original K x fill_mm path, so no unmeasured pen's output
                    # changes.
                    _laid = (pen_profile or {}).get("laid_mm")
                    if _laid:
                        fill_mm_target = float(_laid)
                    else:
                        fill_mm_target = SCALE_AWARE_V2_K * fill_mm
                    fill_px_real = max(1.2, fill_mm_target / mm_per_px_real)
                    if (prelim_res_basis == "final_solve"
                            and not (bool((render_opts or {}).get("illustration_pitch_clamp"))
                                     and bool((render_opts or {}).get("illustration")))):
                        # Solve the pitch floor in the FINAL frame: do not pre-floor at 1.2 px in the
                        # prelim frame (that pins a BIC bwphoto at 1.2 x 2.49 = 2.99 px = 0.135 mm).
                        # The boost below then targets the 3 px floor from the TRUE target fill_px;
                        # the trace still floors at 1.2 px if the cap prevents reaching it. Skipped
                        # for the illustration style: its pitch clamp below IS the final-frame solve,
                        # with its own contracted target (0.9 x fill_mm, not K x fill_mm), so for
                        # that style final_solve == prelim_effective
                        fill_px_real = fill_mm_target / mm_per_px_real
                    # Working-res floor: raise resolution for THIS item's final
                    # pass only, bounded by SCALE_AWARE_V2_NATIVE_CAP (separate from
                    # the general NATIVE_CAP memory guard; see config.py for why:
                    # NATIVE_CAP also gates native_res threshold-mode source
                    # loading, an unrelated path this boost must not inflate), when
                    # the tighter v2 target would otherwise quantize into too few
                    # working px (the distance-transform level-set bands in
                    # trace_v3.py lose pixel-space precision). Legacy/v1 never touch
                    # item_opts["resolution"].
                    if fill_px_real < SCALE_AWARE_V2_FILL_PX_FLOOR:
                        base_res = int(item_opts.get("resolution", 1000))
                        if prelim_res_basis:
                            base_res = _prelim_eff_res   # the grid the prelim bbox (mm_per_px_real) is on
                        needed_boost = SCALE_AWARE_V2_FILL_PX_FLOOR / max(fill_px_real, 1e-6)
                        boosted_res = min(int(round(base_res * needed_boost)), SCALE_AWARE_V2_NATIVE_CAP)
                        actual_boost = boosted_res / max(base_res, 1)
                        if actual_boost > 1.0:
                            item_opts["resolution"] = boosted_res
                            item_opts["_tracev3_res_cap_override"] = boosted_res
                            fill_px_real = fill_px_real * actual_boost
                        v2_res_boost = {
                            "base_res": base_res, "needed_boost": round(needed_boost, 4),
                            "boosted_res": boosted_res, "actual_boost": round(actual_boost, 4),
                            "fill_px_real_pre_boost": round(fill_mm_target / mm_per_px_real, 6),
                        }
                        if prelim_res_basis:
                            v2_res_boost["prelim_res_basis"] = prelim_res_basis
                            v2_res_boost["requested_res"] = int(_prelim_req_res)
                else:
                    fill_px_real = max(1.2, fill_mm / mm_per_px_real)
                # Line Art / Illustration pitch clamp (opt-in, STYLE-SCOPED:
                # reads render_opts["illustration"] AND ["illustration_pitch_clamp"],
                # both of which only the illustration style ever sets -- every other
                # style skips this block).
                # The v2 formula above floors fill_px_real at 1.2 in the PRELIM frame,
                # and the resolution boost then leaves the physical pitch
                # (fill_px_real * mm_per_px) at 1.2 * m0 -- which at a large placement
                # exceeds the pen's fill_mm (cracks). Re-solve in the FINAL frame
                # instead: target pitch = ILLUSTRATION_PITCH_CLAMP_FRAC * fill_mm,
                # boost chosen so fill_px reaches the 3px floor if the native cap
                # allows, else the cap (and fill_px lands below the floor; the existing
                # boost-ceiling warning then says so). Fires only when the unclamped
                # pitch would exceed fill_mm, so a placement that is already fine
                # (e.g. the validated REF placement) is left exactly as it was.
                illus_clamp_rec = None
                if (bool((render_opts or {}).get("illustration_pitch_clamp"))
                        and bool((render_opts or {}).get("illustration"))
                        and scale_aware_v2 and fill_spacing_mm is None):
                    _b0 = (float(v2_res_boost["actual_boost"])
                           if (v2_res_boost and v2_res_boost.get("actual_boost", 1.0) > 1.0) else 1.0)
                    _pitch_unclamped = fill_px_real * mm_per_px_real / _b0
                    if _pitch_unclamped > fill_mm + 1e-9:
                        # Under prelim_res_basis the clamp reads the corrected basis too -- with the
                        # flag off it is right only because ILLUSTRATION_RECIPE pins resolution 2200 (config.py).
                        _base_res = (int(v2_res_boost["base_res"]) if v2_res_boost
                                     else (_prelim_eff_res if prelim_res_basis
                                           else int(item_opts.get("resolution", 1000))))
                        _target = ILLUSTRATION_PITCH_CLAMP_FRAC * fill_mm
                        _need = max(1.0, SCALE_AWARE_V2_FILL_PX_FLOOR * mm_per_px_real / _target)
                        _res = min(int(round(_base_res * _need)), SCALE_AWARE_V2_NATIVE_CAP)
                        _bc = max(1.0, _res / max(_base_res, 1))
                        if _bc > 1.0:
                            item_opts["resolution"] = _res
                            item_opts["_tracev3_res_cap_override"] = _res
                        fill_px_real = _target * _bc / mm_per_px_real
                        v2_res_boost = {
                            "base_res": _base_res, "needed_boost": round(_need, 4),
                            "boosted_res": int(_res if _bc > 1.0 else _base_res),
                            "actual_boost": round(_bc, 4),
                            "fill_px_real_pre_boost": round(_target / mm_per_px_real, 6),
                        }
                        if prelim_res_basis:
                            v2_res_boost["prelim_res_basis"] = prelim_res_basis
                            v2_res_boost["requested_res"] = int(_prelim_req_res)
                        illus_clamp_rec = {
                            "applied": True, "target_pitch_mm": round(_target, 6),
                            "unclamped_pitch_mm": round(_pitch_unclamped, 6),
                            "frac_of_fill_mm": ILLUSTRATION_PITCH_CLAMP_FRAC,
                            "fill_px_final": round(fill_px_real, 6),
                        }
                item_opts["fill_px"] = fill_px_real
                # Reporting-basis correction. fit0/s0/mm_per_px_real above were
                # derived from pnw/pnh -- the PRELIMINARY trace's bbox at the
                # UNBOOSTED working resolution. When the v2 resolution-floor boost
                # fires (immediately above), the FINAL trace pass (below) runs at a
                # higher working resolution, whose raw px (hence raw mm, PX2MM is a
                # fixed constant) bbox is proportionally larger for the identical
                # logical content -- this is universal app behavior (see
                # svg_io._svg_bbox_mm), not v2-specific. Left uncorrected,
                # analytic_* would sit next to final_*/actual_placed_* (computed
                # below off the FINAL, boosted-resolution SVG) at a DIFFERENT
                # resolution basis, diverging by exactly the boost ratio and
                # reading as fit-quality error when it is a resolution-basis
                # mismatch. So analytic_base_fit/analytic_s/mm_per_px_real/
                # analytic_placed_* are re-derived at the SAME (final,
                # post-boost-decision) resolution basis the FINAL pass will use,
                # via cheap arithmetic against the already-known boost ratio
                # (v2_res_boost["actual_boost"]) -- NOT a new trace pass, and this
                # block runs strictly AFTER item_opts/fill_px were finalized above,
                # so it cannot affect routing or gcode. The raw prelim numbers are
                # preserved verbatim under the *_prelim_raw keys so the two bases
                # are never conflated.
                # mm-spacing v2's own fixed-reference boost (spacing_res_boost)
                # fires on the SAME resolution knob v2_res_boost does, so it
                # needs the same basis correction below -- extend the check
                # rather than duplicate the re-derivation block.
                _active_boost = v2_res_boost if (v2_res_boost and v2_res_boost.get("actual_boost", 1.0) > 1.0) \
                    else (spacing_res_boost if (spacing_res_boost and spacing_res_boost.get("actual_boost", 1.0) > 1.0)
                          else None)
                res_basis_boost = float(_active_boost["actual_boost"]) if _active_boost else 1.0
                pnw_fb, pnh_fb = pnw * res_basis_boost, pnh * res_basis_boost
                # Same rotated-footprint fit as prot_w/prot_h above,
                # re-applied at the boosted-resolution basis (rotation extent
                # scales linearly with a uniform boost, but re-deriving via
                # the helper keeps this block self-contained/obviously
                # correct rather than relying on that algebraic fact).
                prot_w_fb, prot_h_fb = _rotated_extent_mm(pnw_fb, pnh_fb, item.get("rot", 0.0))
                fit0_fb = min(cellW / prot_w_fb, cellH / prot_h_fb) * 0.9
                s0_fb = fit0_fb * float(item.get("scale", 1.0))
                mm_per_px_real_fb = max(1e-9, s0_fb * PX2MM)
                if fill_spacing_mm is not None:
                    # trace_v3.py back-solves mm_per_px as fill_mm/fill_px,
                    # which only recovers the true working-grid mm/px because
                    # fill_px was ITSELF derived from fill_mm there. Once
                    # fill_px comes from fill_spacing_mm instead, that back-
                    # solve is off by (fill_mm/fill_spacing_mm) and would move
                    # the speck-cull floor, hole floor, and gap/length audits
                    # with spacing alone -- defeating the whole point of this
                    # flag (spacing and tone must stay independent). Pass the
                    # real mm/px through explicitly, at the SAME resolution
                    # basis the final trace pass actually runs at (mm_per_px_real_fb,
                    # not the unboosted mm_per_px_real -- matters whenever
                    # spacing_res_boost fired above); trace_v3.py only reads
                    # this key when fill_spacing_mm was set upstream, so no
                    # other path is affected.
                    item_opts["_mm_per_px_override"] = mm_per_px_real_fb
                analytic_equiv = {
                    "prelim_bbox_mm": [round(v, 6) for v in pnb],
                    "prelim_width_mm": round(pnw, 6), "prelim_height_mm": round(pnh, 6),
                    "scale_aware_v2": scale_aware_v2,
                    "v2_res_boost": v2_res_boost,
                    # These five keys are emitted ONLY when fill_spacing_mm was
                    # actually requested: unconditional emission would change
                    # audit_json_sha256 for fixtures that run scale_aware_v2 (hence
                    # enter this branch) without ever setting fill_spacing_mm. Same
                    # precedent as trace_v3.py's own fill_routing note (search
                    # "deliberately NOT added to this audit dict"): purely additive
                    # metadata is still a real output change to a byte-identity
                    # contract.
                    **({
                        "fill_spacing_mm_requested": fill_spacing_mm,
                        "fill_spacing_mm_clamp_binds": mm_spacing_clamp_binds,
                        # Named "target_post_boost", not "realized": the value is
                        # computed purely from this call's own inputs (fill_px_real,
                        # itself derived from fill_spacing_mm, times
                        # mm_per_px_real_fb), so it cannot disagree with the request
                        # by construction. It equals the request unless the 1.2px
                        # representability clamp or the fixed-reference resolution
                        # boost moved fill_px_real, in which case it is the TARGET
                        # those adjustments are aiming for -- not a readout of what
                        # the gcode ended up containing. Nothing here measures
                        # emitted geometry.
                        "fill_spacing_mm_target_post_boost": (
                            round(fill_px_real * mm_per_px_real_fb, 6)
                            if fill_spacing_mm is not None else None),
                        "fill_spacing_mm_precision_risk": fill_spacing_mm_precision_risk,
                        "spacing_res_boost": spacing_res_boost,
                    } if fill_spacing_mm is not None else {}),
                    "analytic_base_fit": round(fit0_fb, 9), "analytic_s": round(s0_fb, 9),
                    "mm_per_px_real": round(mm_per_px_real_fb, 9),
                    "fill_px_real": round(fill_px_real, 6),
                    # Line Art / Illustration only (no key for any other style, so
                    # their audit/response bytes do not move). The PHYSICAL pass pitch this
                    # placement commands, for the stats_warnings pitch-vs-fill_mm check.
                    **({
                        # REALIZED pitch: trace_v3 floors fill_px at 1.2 px, so a
                        # placement so large that the target is < 1.2 final px really lays 1.2 px.
                        "illustration_pitch_mm": round(max(fill_px_real, 1.2) * mm_per_px_real_fb, 6),
                        "illustration_pitch_floor_binds": bool(fill_px_real < 1.2),
                        "illustration_fill_mm": round(fill_mm, 6),
                        "illustration_pitch_clamp": illus_clamp_rec,
                    } if bool((render_opts or {}).get("illustration")) else {}),
                    # Rotated-footprint placed size (matches actual_placed_*
                    # below, which is also now rotation-aware), so
                    # divergence_width/height_mm compares like with like.
                    "analytic_placed_width_mm": round(prot_w_fb * s0_fb, 6),
                    "analytic_placed_height_mm": round(prot_h_fb * s0_fb, 6),
                    # Raw prelim-pass numbers, UNBOOSTED resolution basis --
                    # kept for transparency/debuggability, NOT directly
                    # comparable to final_*/actual_placed_* when a boost
                    # fired (analytic_resolution_basis below states which
                    # case applies).
                    "analytic_base_fit_prelim_raw": round(fit0, 9),
                    "analytic_s_prelim_raw": round(s0, 9),
                    "mm_per_px_real_prelim_raw": round(mm_per_px_real, 9),
                    "analytic_resolution_basis": ("final (boost fired, corrected via actual_boost)"
                                                   if res_basis_boost > 1.0
                                                   else "prelim==final (no boost fired)"),
                }
                if prelim_res_basis and scale_aware_v2 and fill_spacing_mm is None:
                    # present ONLY under the opt-in flag (flag-off item_geometry is untouched)
                    analytic_equiv["prelim_res_basis"] = {
                        "mode": prelim_res_basis, "requested_res": int(_prelim_req_res),
                        "prelim_effective_res": int(_prelim_eff_res),
                        "res_basis_boost_applied": round(res_basis_boost, 6),
                    }
        elif item_opts.get("mode") == "sketchfield":
            # Sketch Field placed-size basis. pfm_native._sketch_field_svg
            # would otherwise compute mm_per_px from the DECLARED paper
            # (paper_w_mm/margin_mm), ignoring this item's actual placed size on
            # its grid cell (about 0.9x off in the common single-item width-bound
            # case, worse off-aspect). erase_px IS the tone mechanism
            # (pfm_native.py module docstring), so a wrong basis is not cosmetic.
            # Unconditional (no scale_aware/scale_aware_v2 flag gate).
            # Cheap analytic version, no preliminary generation pass needed
            # (unlike tracev3's ink-shape case): Sketch Field's raster canvas
            # is always the source's own aspect ratio at the resolution
            # policy render.py's render_raster_to_svg step 1 applies
            # (MODE_MAX_RES cap + aspect-preserving max-side resize) --
            # reimplemented here (not imported) to avoid a
            # render.py<->composite.py import cycle; the two are kept in
            # lockstep by sharing the MODE_MAX_RES constant.
            # narrow, well-commented 5-line duplication, not a new pipeline.
            try:
                _raw_candidates = [m for m in WORK_DIR.glob(f"src_{item['id']}.*")
                                    if not (m.name.endswith("_traced.svg") or "_render_" in m.name
                                            or m.name.endswith("_render.svg"))]
                if _raw_candidates:
                    from PIL import Image as _PILImg
                    with _PILImg.open(_raw_candidates[0]) as _im:
                        _iw, _ih = _im.size
                    _req_res = int(item_opts.get("resolution", 900))
                    _cap = MODE_MAX_RES.get("sketchfield", 1200)
                    _eff_res = min(_req_res, _cap)
                    _sres = _eff_res / max(1, max(_iw, _ih))
                    _W, _H = max(1, round(_iw * _sres)), max(1, round(_ih * _sres))
                    _nw, _nh = _W * PX2MM, _H * PX2MM
                    # Fit against the ROTATED footprint: item["rot"] is applied
                    # later as a vpype layer transform, so the cell-fit must use
                    # the rotated extent. At rot in {0, 180} _rotated_extent_mm
                    # takes an exact no-op path, so the result there is unchanged.
                    _rnw, _rnh = _rotated_extent_mm(_nw, _nh, item.get("rot", 0.0))
                    _fit0 = 1.0 if bool(item.get("true_size")) else min(cellW / _rnw, cellH / _rnh) * 0.9
                    _s0 = _fit0 * float(item.get("scale", 1.0))
                    item_opts["mm_per_px_real"] = max(1e-9, _s0 * PX2MM)
            except Exception:
                pass  # fall back to pfm_native's legacy declared-paper basis
        svg = _source_svg(str(item["id"]), item_opts, debug_prefix=dbg, pen_id=pen_id)
        if idx == 0: first_svg = svg
        item_svgs[L] = svg          # layer -> render SVG (trace-boundary sidecar lookup)
        reads.append(f'read -l {L} {shlex.quote(str(svg))}')
        nb = _svg_bbox_mm(svg)
        if not nb: continue
        nw, nh = max(nb[2] - nb[0], 1e-3), max(nb[3] - nb[1], 1e-3)
        ncx, ncy = (nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2
        ccx, ccy = ccx0, ccy0
        # true_size (calibration-as-drawing): a calibration plot's mm ladders
        # only mean what they say at literal 1:1 scale -- the normal per-item
        # "auto-fit to its grid cell" (fit = min(cellW/nw, cellH/nh) * 0.9) is
        # exactly the kind of silent rescale that would invalidate it. Gated
        # behind an item field, so callers that never send true_size compute
        # the same `fit` as always.
        # `r` is read BEFORE `fit` is computed, and `fit` is computed against
        # the ROTATED footprint (rw, rh), not the raw source (nw, nh): the
        # rotation is applied afterward as a vpype layer transform to
        # already-scaled geometry, so fitting the unrotated shape would fit the
        # wrong shape against the cell for any rot that is not a multiple of 180
        # (a 90-degree swap in the worst case). At rot in {0, 180},
        # _rotated_extent_mm returns (nw, nh) via an exact no-trig path, so
        # `fit`/`s` are unchanged there.
        r = float(item.get("rot", 0.0))
        rw, rh = _rotated_extent_mm(nw, nh, r)
        fit = 1.0 if bool(item.get("true_size")) else min(cellW / rw, cellH / rh) * 0.9
        s = fit * float(item.get("scale", 1.0))
        tx, ty = ccx - ncx + float(item.get("dx", 0.0)), ccy - ncy + float(item.get("dy", 0.0))
        if (_mask_ctx is not None or _sd_ctx is not None) and item_opts.get("_stage_e_mask"):
            # Exact placement applied below by vpype: scale s and rotate r about the layer's bbox
            # centre (c0), then translate (tx, ty). Page units are vpype px (mm / PX2MM).
            _ctx1 = {"svg": svg, "s": s, "rot_deg": r,
                     "c0_px": ((nb[0] + nb[2]) / 2 / PX2MM, (nb[1] + nb[3]) / 2 / PX2MM),
                     "t_px": (tx / PX2MM, ty / PX2MM)}
            if _mask_ctx is not None:
                _mask_ctx[L] = _ctx1
            if _sd_ctx is not None:
                _sd_ctx[L] = _ctx1
        # Authoritative fit metadata: this is the source vector bbox BEFORE the
        # user scale/rotation and before cleanup/routing can add artifacts.  The
        # browser must never reverse-engineer these dimensions from a rotated
        # axis-aligned output bbox (that was the rotated-image crop bug).
        item_geometry[str(item["id"])] = {
            "source_bbox_mm": [round(v, 6) for v in nb],
            "source_width_mm": round(nw, 6), "source_height_mm": round(nh, 6),
            "source_center_mm": [round(ncx, 6), round(ncy, 6)],
            "base_fit_scale": round(fit, 9),
            "layout_center_mm": [round(ccx, 6), round(ccy, 6)],
            # Echo the exact item fields used to create this metadata so the UI
            # can reject a response that does not match its request snapshot.
            "item_settings": {"id": item.get("id"), "scale": float(item.get("scale", 1.0)),
                              "rot": float(item.get("rot", 0.0)), "dx": float(item.get("dx", 0.0)),
                              "dy": float(item.get("dy", 0.0)), "crop": item.get("crop")},
        }
        # New key added ONLY when actually requested, so a request with no
        # "fill_page" key keeps an unchanged item_settings dict.
        # Realised pass pitch: `s` just above is the FINAL placement scale (fit from this
        # item's final traced SVG bbox, the value `scale -l L s s` applies below); the helper
        # multiplies the fill px the trace really ran at by s * PX2MM. Readout only.
        if not is_verbatim_item:
            _rp = _realised_pitch_record(item["id"], item_opts, analytic_equiv, svg, s)
            if _rp is not None:
                realised_pitch_items.append(_rp)
        if bool(item.get("fill_page")):
            item_geometry[str(item["id"])]["item_settings"]["fill_page"] = True
        if bool(item.get("true_size")):
            item_geometry[str(item["id"])]["item_settings"]["true_size"] = True
        # Fill Page (cover-crop): the placed axis-aligned bbox at this item's
        # OWN scale/rot/dx/dy (unmodified — Fill Page reuses the normal
        # placement math above verbatim, it does not override scale/rotate/
        # translate), intersected with the request-wide safe region. rot is
        # restricted to 0/90 so the placed bbox stays axis-aligned; the
        # /api/process route rejects any other rot when fill_page is set.
        if bool(item.get("fill_page")) and fill_page_safe_region is not None:
            swap = (round(r / 90.0) % 2) == 1
            placed_w, placed_h = (nh if swap else nw) * s, (nw if swap else nh) * s
            center_x, center_y = ncx + tx, ncy + ty
            placed_bbox_poly = _shapely_box(center_x - placed_w / 2, center_y - placed_h / 2,
                                             center_x + placed_w / 2, center_y + placed_h / 2)
            crop_poly = placed_bbox_poly.intersection(fill_page_safe_region)
            if not crop_poly.is_empty:
                fill_page_clips[L] = crop_poly
                item_geometry[str(item["id"])]["fill_page_placed_bbox_mm"] = [round(v, 3) for v in placed_bbox_poly.bounds]
                item_geometry[str(item["id"])]["fill_page_crop_bbox_mm"] = [round(v, 3) for v in crop_poly.bounds]
            else:
                item_geometry[str(item["id"])]["fill_page_placed_bbox_mm"] = [round(v, 3) for v in placed_bbox_poly.bounds]
                item_geometry[str(item["id"])]["fill_page_crop_bbox_mm"] = None
        if analytic_equiv is not None:
            analytic_equiv["final_width_mm"] = round(nw, 6)
            analytic_equiv["final_height_mm"] = round(nh, 6)
            analytic_equiv["final_fit"] = round(fit, 9)
            analytic_equiv["final_s"] = round(s, 9)
            # actual_placed_width/height_mm report the ROTATED footprint
            # (rw/rh, not nw/nh) -- the response contract's own promise
            # (browser must never reverse-engineer placed size from a
            # rotated output bbox) only holds if this is the TRUE placed
            # size, and for rot outside {0, 180} that is rw*s/rh*s, not the
            # pre-rotation nw*s/nh*s.
            analytic_equiv["actual_placed_width_mm"] = round(rw * s, 6)
            analytic_equiv["actual_placed_height_mm"] = round(rh * s, 6)
            analytic_equiv["divergence_width_mm"] = round(abs(analytic_equiv["analytic_placed_width_mm"] - rw * s), 6)
            analytic_equiv["divergence_height_mm"] = round(abs(analytic_equiv["analytic_placed_height_mm"] - rh * s), 6)
            # Named -prelim_approx (not "scale_aware_equivalence"): the
            # "analytic_*" numbers in here come from the FIRST (preliminary,
            # legacy-fill_px) trace pass, not the real/final one, and can diverge
            # from the final placed size on the non-fit-binding axis by more than
            # float rounding (up to ~1mm in a heavily-downscaled multi-item case).
            # The "final_*"/"actual_placed_*" numbers ARE the real, final-pass
            # truth (what actually gets plotted) and are safe to read as such;
            # only the "analytic_*"/"prelim_*" numbers are the approximation.
            # "analytic_*" is re-derived at the FINAL (post-boost-decision)
            # resolution basis when the v2 resolution-floor boost fires (see the
            # comment above the analytic_equiv construction), so it stays directly
            # comparable to "final_*"/"divergence_*" without a resolution-ratio
            # correction. The unboosted-basis prelim numbers are kept verbatim
            # under "*_prelim_raw" / "analytic_resolution_basis".
            item_geometry[str(item["id"])]["scale_aware_equivalence_prelim_approx"] = analytic_equiv
        xforms.extend([f'scale -l {L} {s} {s}', f'rotate -l {L} -- {r}', f'translate -l {L} -- {tx}mm {ty}mm'])
        layer_of[L] = str(item["id"])

    # ── PROCESS PIPELINE ──
    # D. Vector cleanup: read + place each layer, merge touching strokes (vec_merge),
    #    pin to the page, write the cleaned (but un-routed) geometry to disk.
    # E. Low-lift routing (Python): fuse nearby endpoints with pen-down connectors.
    # G/H. Build preview + (optional) G-code from the ROUTED geometry.
    merge_tol = float(render_opts.get("vec_merge", 0.8))
    rot = f' rotate -o {paper_w / 2}mm {paper_h / 2}mm -- {paper_rot}' if abs(float(paper_rot)) > 1e-6 else ''

    # Stage D linemerge: restrict to non-verbatim layers via vpype's own `-l`
    # layer selector (linemerge is @layer_processor -- layers outside the
    # selector pass through completely untouched, confirmed by reading
    # vpype_cli's linemerge callback). `noncal_layers` == every layer id
    # when no item requested true_size (the common case), so the emitted
    # command string is the plain one whenever verbatim_layers is empty.
    noncal_layers = sorted(L for L in layer_of if L not in verbatim_layers)
    merge_cmd = (f' linemerge --tolerance {merge_tol}mm -l {",".join(str(x) for x in noncal_layers)}'
                 if noncal_layers else '')

    clean_svg = output_path.with_suffix(".clean.svg")
    with atomic_target(clean_svg) as tmp_clean_svg:
        vpype_cli.execute(' '.join(reads + xforms) + merge_cmd +
                          f' pagesize {page_orient}{spec} write {shlex.quote(str(tmp_clean_svg))}')

    routed_svg = output_path.with_suffix(".routed.svg")
    connectors_svg = output_path.with_suffix(".connectors.svg")
    route_stats = _route_low_lift_doc(clean_svg, routed_svg, connector_distance, target_z_moves, connectors_svg=connectors_svg, bypass_layers=verbatim_layers, **({"mask_ctx": _mask_ctx} if _mask_ctx is not None else {}),
                                     # Stay down over ink (opt-in): the walk cap rises to the two-pen one, else tram-size plain
                                     # jobs (>20000 strokes) skip the router and nothing could be joined.
                                     **({"stay_down_ctx": _sd_ctx, "max_walk_segs": ROUTE_WALK_MAX_SEGS_TWOPEN} if _sd_ctx is not None else {}))
    if _mask_ctx is not None and "connector_mask_check" in route_stats:
        # Connector-mask per-job record. The (large) rejected-connector list is written only under debug.
        _rej = route_stats["connector_mask_check"].pop("_rejected_segments_page_px", [])
        atomic_write_text(WORK_DIR / "debug_connector_mask_check.json", json.dumps(route_stats["connector_mask_check"], indent=1))
        if debug:
            atomic_write_text(WORK_DIR / "debug_connector_mask_rejected.json", json.dumps(
                {"units": "page px (vpype, mm/PX2MM); entries [layer_id, x0, y0, x1, y1]", "segments": _rej}))
        else:
            try: (WORK_DIR / "debug_connector_mask_rejected.json").unlink()
            except OSError: pass
    else:
        # off run: drop stale connector-mask records from an earlier on-run so the vcache never snapshots them under an off key
        for _stale in ("debug_connector_mask_check.json", "debug_connector_mask_rejected.json"):
            try: (WORK_DIR / _stale).unlink()
            except OSError: pass

    # Fill Page (cover-crop): the clip runs AFTER routing (Stage E), never
    # before it, because the router links strokes and the clip must cut the
    # linked result. Only touches routed_svg (what
    # preview/gcode read below) and the display-only connectors_svg overlay;
    # clean_svg / clean_preview stay the unclipped pre-routing snapshot.
    fill_page_clip_report = {}
    if fill_page_clips:
        fill_page_clip_report["routed"] = _apply_fill_page_clips(routed_svg, fill_page_clips)
        fill_page_clip_report["connectors"] = _apply_fill_page_clips(connectors_svg, fill_page_clips)

    paper_keepout_clip_report = {}
    if paper_keepout_region is not None:
        paper_keepout_clip_report["routed"] = _apply_global_clip(routed_svg, paper_keepout_region)
        paper_keepout_clip_report["connectors"] = _apply_global_clip(connectors_svg, paper_keepout_region)

    # Plain routed preview = artwork only. The connector X-ray (coral overlay,
    # display-only — connector geometry is already fused into the routed paths)
    # is a SEPARATE preview so router marks are never confused with artwork.
    #
    # TRUTHFUL RENDERING: the preview must show ink at the pen's physical
    # width. vpype's SVG writer falls back to a hardcoded 1px stroke whenever no
    # `penwidth` property is set on the pipeline, which would render ink at an
    # arbitrary hairline decoupled from fill_mm. `penwidth {pen_mm}mm` makes the
    # preview equal the ink-sim's own substrate -- one instrument, not two that
    # can drift apart.
    # Registered-paper previews must prove the same final machine placement as
    # generation before spending time rendering three preview variants. Emit to
    # a private temporary file solely for the existing full-path reach
    # validator, then remove it. Keep routed ordering: ordering is irrelevant
    # to this placement proof and `linesort` is deliberately deferred to final
    # G-code generation.
    #
    # WARNS, DOES NOT REFUSE. This check only runs on REGISTERED paper, and
    # while composing a crowded page it would otherwise fire on every preview and
    # leave an error instead of a picture, even though the items had been added.
    #
    # Downgrading it is safe because the machine never sees this file: it is a
    # throwaway emitted solely to test placement and deleted immediately. The
    # HARD refusal at final G-code generation (validate_gcode_xy_motion on
    # output_path, below, inside `if generate_gcode:`) is unchanged, so nothing
    # out of reach can ever reach the printer.
    preview_reach_warning = None
    if preview_reach_check and not generate_gcode:
        preflight_gcode = output_path.with_name(f".{output_path.stem}.preview-reach.gcode")
        try:
            with atomic_target(preflight_gcode) as tmp_preflight:
                vpype_cli.execute(f'read {shlex.quote(str(routed_svg))}{rot} translate -- {comp_off_x}mm {-comp_off_y}mm gwrite --profile elegoo_cc {shlex.quote(str(tmp_preflight))}')
            validate_gcode_xy_motion(preflight_gcode, reach)
        except Exception as e:
            preview_reach_warning = str(e)
            print(f"  [preview-reach] out of reach (preview continues): {e}")
        finally:
            try:
                preflight_gcode.unlink()
            except FileNotFoundError:
                pass

    # `laid_mm` first: fill_mm is the sideways STEP between passes, not the
    # width of the mark the tip makes.
    pen_mm = float((render_opts or {}).get("laid_mm")
                   or (render_opts or {}).get("fill_mm", 0.15))
    # PREVIEW STROKE WIDTH: the pen's SOLID CORE, when it has been measured.
    #
    # WHY. Drawing every stroke as a solid band at the pen's full width would
    # promise a solid black the pen never lays: the physical mark is PEAKED -- a
    # dark core inside light fringes. Comparisons of screen against paper would
    # then read as "the plotter lost my ink". On one measured sheet (5mm tiles)
    # the preview/plot net gap was 10.01pp of commanded ink at total width but
    # only 2.07pp at core width -- an honest-width preview closes ~79% of it.
    #
    # PREVIEW-ONLY. `pen_mm` is ALSO the geometry basis for the G-code simplify
    # tolerance below (`simplify_mm = simplify_mult * pen_mm`), so reassigning
    # `pen_mm` to the core width would silently move EMITTED GEOMETRY (dormant
    # only while SIMPLIFY_FILL_MULT is 0.0 and `simplify_mult` is not overridden
    # per request). A separate variable makes "the preview shows core_mm"
    # literally true in the code.
    #
    # GATED ON A MEASURED FIELD. `core_mm` exists only on a pen someone has
    # actually measured; every profile without it renders with the full width.
    # Keying off a nominal figure would silently change every pen's preview on
    # no evidence.
    #
    # NO SEED PROFILE CARRIES core_mm. Populating it is a calibration act that
    # needs a gating audit and a full-page true-pen-width A/B. The ~0.15mm
    # figure seen in the one A/B so far is not established: it rests on n=3
    # strokes, and which side of it the truth falls on is undetermined.
    preview_pen_mm = preview_pen_width_mm(render_opts, pen_profile)
    preview_path = output_path.with_suffix(".preview.svg")
    _render_routed_preview(routed_svg, preview_path, preview_pen_mm)
    microlift_stats = None
    motion_smoothing_stats = None
    drawing_bbox_machine = None
    # Point-count reduction on the FINAL G-code path (disabled by default; see
    # config.SIMPLIFY_FILL_MULT). The emitted contours carry every point at
    # raster tolerance (`approxPolyDP(c, 0.4, True)` -- 0.4 PIXELS), so a real
    # job ships ~1.13M segments averaging 0.30mm. On one measured job 82% of plot
    # time is draw motion and 58% of draw segments are shorter than the 0.25mm of
    # runway needed to reach the commanded feedrate, so the machine spends the
    # plot accelerating and decelerating. Dropping points the pen cannot resolve
    # would cut time and the direction-change rate that drives noise and
    # lead-screw wear, but it also opens cracks in fills (see
    # config.SIMPLIFY_FILL_MULT for the measurements).
    #
    # Tolerance is PEN-RELATIVE (SIMPLIFY_FILL_MULT x fill_mm) -- see config.py
    # for why pass pitch, not pen width, is the governing scale. `simplify_mult`
    # overrides per request; 0 disables and restores the exact command string
    # this path emits without simplification.
    #
    # Placed before `linesort` (simplify geometry, then order it) and therefore
    # UPSTREAM of the microlift rewrite, motion smoothing, and -- critically --
    # `validate_gcode_xy_motion` at the end of this block, which re-checks every
    # emitted XY endpoint and chord against the reach envelope. Simplification
    # can move a point by up to the tolerance, so it must never bypass that
    # check; running it here means it cannot.
    simplify_mult = float((render_opts or {}).get("simplify_mult", SIMPLIFY_FILL_MULT))
    simplify_mm = round(simplify_mult * pen_mm, 4) if simplify_mult > 0 else 0.0
    simplify_cmd = f' linesimplify --tolerance {simplify_mm}mm' if simplify_mm > 0 else ''
    if generate_gcode:
        with atomic_target(output_path) as tmp_output:
            vpype_cli.execute(f'read {shlex.quote(str(routed_svg))}{simplify_cmd} linesort{rot} translate -- {comp_off_x}mm {-comp_off_y}mm gwrite --profile elegoo_cc {shlex.quote(str(tmp_output))}')
            _fix_gwrite_source_comment(tmp_output, output_path)
        # Micro-lift: short travels get a low hop instead of the full z_up —
        # same Z count, far less Z travel = far less lead-screw wear. Micro-lift
        # height is z_touch + z_micro (configurable; default 0.6mm above contact).
        # Only runs when the full lift is meaningfully above the micro height.
        # Per-pen z_micro WINS over the global (cal.resolve_z_micro). Source is
        # reported in the stats so a plot can be audited for which file actually
        # set its lift height.
        z_micro, z_micro_source = resolve_z_micro(cal, pen_profile)
        z_up_abs = z_touch + float(z_up)
        z_short_abs = z_touch + z_micro
        # MEGA-hop: the LONGEST travels lift extra-high to avoid dragging across bed
        # warp (inverse of micro-lift). Reuses the high travel clearance z_travel_abs
        # so the initial cross-page move AND every long mid-plot traverse get the same
        # clearance. Tunable: mega_hop_mm (distance above which a hop is "long"); 0 off.
        mega_mm = float(cal.get("mega_hop_mm", 70.0))
        # How SHORT a hop must be to qualify for the low micro lift (per-pen value
        # wins, then global, then 8.0mm). Height is only half the lever -- a 6-8mm
        # traverse held ~1mm off the paper is where drag actually shows, while the
        # sub-1mm hops inside dense hatching are harmless. Lowering this buys back
        # clearance on the risky hops WITHOUT paying the Z-time cost on every
        # short hop.
        micro_hop_mm = float((pen_profile or {}).get("micro_hop_mm",
                             cal.get("micro_hop_mm", 8.0)))
        do_micro = microlift and z_up_abs > z_short_abs + 0.05
        do_mega = mega_mm > 0 and z_travel_abs > z_up_abs + 0.05
        if do_micro or do_mega:
            try:
                microlift_stats = _microlift_gcode(
                    output_path, z_up_abs,
                    z_short_abs=(z_short_abs if do_micro else None), short_mm=micro_hop_mm,
                    z_mega_abs=(z_travel_abs if do_mega else None), long_mm=mega_mm)
                microlift_stats["micro_hop_mm"] = micro_hop_mm if do_micro else None
                microlift_stats["short_lift_mm"] = round(z_micro, 2) if do_micro else None
                microlift_stats["short_lift_source"] = z_micro_source if do_micro else None
                print(f"  [microlift] z_micro={z_micro} ({z_micro_source}); "
                      f"{microlift_stats['short_lifts']}/{microlift_stats['lifts']} short, "
                      f"{microlift_stats['mega_lifts']} mega (>{mega_mm:.0f}mm @ Z{z_travel_abs:.2f}); "
                      f"-{microlift_stats['z_travel_saved_mm']}mm +{microlift_stats['mega_added_mm']}mm Z")
            except Exception as e:
                print(f"  [microlift] failed (gcode left at full lifts): {e}")
        # Zone the final stroke blocks, after microlift has completed its own
        # structural rewrite and before the normal reach-safety validation.
        motion_smoothing_stats = _apply_motion_smoothing(output_path, motion_smoothing)
        # Every commanded XY endpoint and chord (including travel) must remain
        # inside the active, potentially concave nozzle envelope.  A bounding
        # box is insufficient: a chord can cross the stepped profile's notch.
        try:
            bb = _gcode_pendown_bbox(output_path, z_touch + float(z_down))
            motion = validate_gcode_xy_motion(output_path, reach)
            if bb:
                drawing_bbox_machine = {"bbox": [round(v, 1) for v in bb],
                                        "reach_profile": reach["id"], "reach_label": reach["label"],
                                        "reach_polygon": [[round(v, 3) for v in p] for p in reach["polygon"]],
                                        "reach_bbox": [round(v, 3) for v in reach["bbox"]],
                                        "xy_moves_checked": motion["xy_moves"], "reach_validated": True}
        except Exception as e:
            raise RuntimeError(f"reach safety check failed: {e}") from e

        # Postamble + preamble fan-off are baked into the elegoo_cc gwrite profile
        # itself (vpype.toml) via the default_values set above -- nothing to
        # append here. Fan commands are not repeated through the body of the file
        # (a file of bare M-codes with no motion halts this firmware); the
        # one-time block is flanked by real motion.

    item_bboxes, stats = {}, compute_vector_stats(vp.Document(), draw_speed, travel_speed, plunge_speed, lift_speed, z_up, z_down)
    try:
        prev = vp.read_multilayer_svg(str(preview_path), quantization=0.1)
        # Drop the X-ray overlay before stats — its segments are already inside
        # the fused routed paths; counting them would double the connector draw.
        prev.layers.pop(XRAY_LAYER, None)
        for L, did in layer_of.items():
            if L in prev.layers:
                b = prev.layers[L].bounds()
                if b: item_bboxes[did] = [round(b[0] * PX2MM, 2), round(b[1] * PX2MM, 2), round(b[2] * PX2MM, 2), round(b[3] * PX2MM, 2)]
        stats = compute_vector_stats(prev, draw_speed, travel_speed, plunge_speed, lift_speed, z_up, z_down)
    except Exception as e: print(f"stats/bbox compute failed: {e}")
    stats.update(route_stats)
    stats["item_geometry"] = item_geometry
    if realised_pitch_items:
        # Absent for every job with no tracev3-engine raster item (other styles, vector
        # imports, calibration sheets).
        _rp_vals = [r["realised_pitch_mm"] for r in realised_pitch_items]
        stats["realised_pitch"] = {"items": realised_pitch_items,
                                   "min_mm": min(_rp_vals), "max_mm": max(_rp_vals),
                                   "basis": "fill_px_final x (final placed scale x PX2MM); "
                                            "not the preliminary scale, not the fill_mm/fill_px back-solve"}
    if prelim_res_basis and scale_aware_v2 and fill_spacing_mm is None:
        stats["prelim_res_basis"] = prelim_res_basis   # opt-in flag only; absent when off
    if fill_page_reach_stats is not None:
        stats["fill_page_reach"] = fill_page_reach_stats
    if fill_page_clip_report:
        stats["fill_page_clip"] = fill_page_clip_report
    if paper_keepout_clip_report:
        stats["paper_keepout_clip"] = paper_keepout_clip_report
        stats["paper_keepout_ink_radius_mm"] = round(paper_keepout_ink_radius_mm, 4)
    # Carried so the UI can warn loudly without the pipeline having refused --
    # see the preview_reach_warning block above.
    if preview_reach_warning:
        stats["preview_reach_warning"] = preview_reach_warning
    stats["preview_clean_available"] = True
    stats["preview_xray_available"] = bool(route_stats.get("connector_count", 0) > 0)
    # Trace-boundary provenance: private key, popped by the route before the response
    # is built (so it reaches only the run-provenance sidecar). collect() never raises.
    from ccplotter import trace_boundary as _tbd
    stats["_trace_boundary"] = _tbd.collect(routed_svg, layer_of, item_svgs, preview_pen_mm,
                                            laid_pen_mm=pen_mm)
    stats["_lazy_preview_sources"] = {"clean": clean_svg, "routed": routed_svg,
                                       "connectors": connectors_svg, "pen_mm": preview_pen_mm}
    if microlift_stats:
        stats["microlift"] = microlift_stats
    if motion_smoothing == "zoned" and motion_smoothing_stats:
        stats["motion_smoothing"] = motion_smoothing_stats
    if drawing_bbox_machine:
        stats["machine_bbox"] = drawing_bbox_machine

    # Trace-resolution transparency for the primary drawing: written by
    # render_raster_to_svg next to the (possibly cached) render SVG.
    try:
        tinfo = Path(str(first_svg) + ".info.json")
        if tinfo.exists():
            stats["trace_info"] = json.loads(tinfo.read_text())
    except Exception:
        pass

    # ── Hybrid mode metrics (candidate counts, weights, score) ──
    if (render_opts or {}).get("mode") == "hybrid":
        try:
            with open(WORK_DIR / "debug_hybrid_stats.json") as f:
                stats["hybrid"] = json.load(f)
        except Exception: pass

    # ── Plot-Ready Line Art quality check ──
    if (render_opts or {}).get("mode") == "plotready":
        try:
            with open(WORK_DIR / "debug_plotready_quality.json") as f:
                stats["lineart_quality"] = json.load(f)
        except Exception: pass

    # ── Debug outputs: point the UI at whatever intermediates exist on disk ──
    debug_files = {}
    if debug:
        for key, fn in (("gray", "debug_gray.png"), ("xdog", "debug_xdog.png"), ("mask_raw", "debug_mask_raw.png"),
                        ("mask_clean", "debug_mask_clean.png"), ("edges", "debug_edges.png"),
                        ("skeleton", "debug_skeleton.png"), ("raw_paths", "debug_raw_paths.svg"),
                        ("hybrid_geom", "debug_hybrid_geom.svg"), ("hybrid_edges", "debug_hybrid_edges.svg"),
                        ("hybrid_ink", "debug_hybrid_ink.svg"), ("hybrid_candidates", "debug_hybrid_candidates.svg"),
                        ("hybrid_rejected", "debug_hybrid_rejected.svg"), ("hybrid_fused", "debug_hybrid_fused.svg"),
                        ("detail_rois", "debug_detail_rois.png"), ("detail_roi_boxes", "debug_detail_roi_boxes.svg"),
                        ("detail_candidates", "debug_detail_candidates.svg"), ("detail_accepted", "debug_detail_accepted.svg"),
                        ("detail_rejected", "debug_detail_rejected.svg"), ("detail_merged", "debug_detail_merged.svg")):
            if (WORK_DIR / fn).exists(): debug_files[key] = fn
        # core_fill=hybrid inspectable intermediate: listed only
        # when the flag is active, so a stale file never leaks into other runs.
        if (render_opts or {}).get("core_fill") in ("hybrid", "rows", "outline_rows") and (WORK_DIR / "debug_core_mask.png").exists():
            debug_files["core_mask"] = "debug_core_mask.png"
        # stitch_mm per-job record: same listed-only-when-active rule.
        if (render_opts or {}).get("stitch_mm") and (WORK_DIR / "debug_stitch.json").exists():
            debug_files["stitch"] = "debug_stitch.json"
        # connector_mask_check per-job record: same listed-only-when-active rule.
        if connector_mask_check and (WORK_DIR / "debug_connector_mask_check.json").exists():
            debug_files["connector_mask_check"] = "debug_connector_mask_check.json"
            if (WORK_DIR / "debug_connector_mask_rejected.json").exists():
                debug_files["connector_mask_rejected"] = "debug_connector_mask_rejected.json"
        for key, p in (("clean_svg", clean_svg), ("routed_svg", routed_svg), ("preview_svg", preview_path), ("connectors_svg", connectors_svg)):
            if p.exists(): debug_files[key] = p.name
    stats["debug_files"] = debug_files
    # Stamp the lifecycle policy actually used into info.json (the
    # gcode header comment carries the same values -- see vpype.toml document_start).
    stats["end_policy"] = str(end_policy).lower()
    stats["quiet_fans"] = bool(quiet_fans)
    return preview_path, item_bboxes, stats
