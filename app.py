import math
import os
import sys
import json
import time
import shlex
import threading
import io
import zipfile
from pathlib import Path
from uuid import uuid4
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import websocket
import cv2
import numpy as np
import vpype as vp
import vpype_cli

# ── Version / build identity ───────────────────────────────────────────────────
# Defined in ccplotter/version.py and re-exported here so existing references
# (internal, desktop/ccplotter_app.py, scripts) resolve against `app`.
from ccplotter.version import _load_version_info, _version_string, VERSION_INFO

# ── Configuration ─────────────────────────────────────────────────────────────
# Defined in ccplotter/config.py and re-exported here.
from ccplotter.config import *  # noqa: F401,F403

# ── Run archive: self-contained snapshots of expensive runs (AI cleans, gcode
# builds) so the user can compare/restore without re-running minutes of AI.
# Defined in ccplotter/runs.py and re-exported here. ──
from ccplotter.runs import RUNS_DIR, RUNS_INDEX, RUNS_KEEP, _RUNS_LOCK, _runs_load, _runs_add

# ── Last-generation bundle: in-memory snapshot of the most recent COMPLETED
# (job_type='gcode') /api/process call, captured at generation time so a later
# /download_bundle always matches what was actually served/plotted -- not
# whatever happens to be sitting in work/layout.* by the time of download
# (those get overwritten by the next run).
# Defined in ccplotter/bundle.py. _snapshot_bundle and _BUNDLE_LOCK are
# re-exported as bare names (a function and a Lock, never rebound).
# _LAST_BUNDLE is NOT: _snapshot_bundle rebinds it with `global` inside
# ccplotter.bundle, so `from ccplotter.bundle import _LAST_BUNDLE` here would
# freeze app.py's copy at import time (None forever, and download_bundle would
# always 404). Read it through the module instead:
# `import ccplotter.bundle as _bundle_mod`. ──
import ccplotter.bundle as _bundle_mod
from ccplotter.bundle import _snapshot_bundle, _BUNDLE_LOCK

# ── Current-image variant cache: while the SAME source set is loaded, flipping
# between Line Art / Sketch (and tweaking settings) would re-run the whole
# pipeline every time even though the trace SVG is already disk-cached. This
# caches the FULL result (preview/clean/xray SVGs + gcode + JSON) per
# (source-fingerprint, settings), so switching BACK to a variant is a file
# restore, not a recompute. It is session/current-image scoped (not durable
# like Recent Runs) and is discarded when the loaded source set changes
# (add/delete/replace/crop/transform all change the key or the fingerprint).
# Defined in ccplotter/vcache.py. _VCACHE_FP, _VCACHE_SEQ and _VCACHE_CURRENT are
# rebound via `global` inside _vcache_get/_vcache_put, so they are NOT
# re-exported as bare names; the direct readers in app.py (api_variants,
# api_variants_use) use `_vcache_mod._VCACHE_FP` / `_vcache_mod._VCACHE_CURRENT`
# to see the live value. _VCACHE (dict) and _VCACHE_LOCK (Lock) are never
# rebound, so bare re-exports share the same object. VCACHE_DIR, VCACHE_KEEP
# and _VCACHE_OUTPUTS are constants. ──
import ccplotter.vcache as _vcache_mod
from ccplotter.vcache import (VCACHE_DIR, VCACHE_KEEP, _VCACHE, _VCACHE_LOCK,
                               _VCACHE_OUTPUTS, _source_fingerprint, _variant_key,
                               _variant_label, _vcache_get, _vcache_put)

def _inject_white_bg(svg_path):
    """Insert a soft 'paper' background rect covering the viewBox as the first drawable
    element. Sketch previews need dark-ink-on-light-paper so stroke WIDTH and hatch
    DENSITY read as real tone; a transparent preview on the dark canvas inverts tone."""
    import re
    try:
        t = open(svg_path, encoding="utf-8").read()
    except Exception:
        return
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', t)
    if not m:
        return
    rect = f'<rect x="0" y="0" width="{m.group(1)}" height="{m.group(2)}" fill="{SKETCH_PAPER}"/>'
    if "<defs/>" in t:
        t = t.replace("<defs/>", "<defs/>" + rect, 1)
    else:
        t = re.sub(r'(<svg[^>]*>)', r'\1' + rect, t, count=1)
    open(svg_path, "w", encoding="utf-8").write(t)

# _gcode_lifecycle_values / _microlift_gcode / _gcode_pendown_bbox /
# _fan_probe_gcode / _inject_fan_off / _calibration_gcode are defined in
# ccplotter/gcode_post.py. They are plain functions with no module-level
# mutable state, so bare re-exports are safe.
from ccplotter.gcode_post import (_gcode_lifecycle_values, _microlift_gcode,
                                   _gcode_pendown_bbox, _fan_probe_gcode,
                                   _inject_fan_off, _calibration_gcode)

# Defined in ccplotter/printer.py. `printer = Printer(PRINTER_IP)` is assigned
# once and never rebound; the object is mutated in place (self._ws,
# self.connected, self._last_snap), so a bare re-export shares the same
# instance.
from ccplotter.printer import Printer, printer, start_printer

# load_cal / save_cal / load_pen_profiles / save_pen_profiles / active_pen_profile
# are defined in ccplotter/cal.py.
from ccplotter.cal import load_cal, save_cal, load_pen_profiles, save_pen_profiles, active_pen_profile

# _dxf_to_svg / _pdf_to_svg / _paper_spec / _svg_bbox_mm / _write_paths_svg /
# _read_polylines_svg are defined in ccplotter/svg_io.py. They are plain
# functions with no rebound globals, so bare re-exports are safe.
from ccplotter.svg_io import (_dxf_to_svg, _pdf_to_svg, _paper_spec, _svg_bbox_mm,
                               _write_paths_svg, _read_polylines_svg)

# ── Compositing / gcode assembly: defined in ccplotter/composite.py.
# _source_svg, _merge_texture_layer, _grid and _composite (layout + gcode
# emission) live together because they call each other (_source_svg calls
# _merge_texture_layer; _composite calls _grid and _source_svg). The module has
# no `global` statements and depends only on ccplotter.config, .cal, .svg_io,
# .render, .stats_warnings and .gcode_post.
#
# _texture_backfill is NOT in composite.py: it depends only on _write_paths_svg
# (svg_io) and _stroke_px_for (mask), and its sole caller is _ai_clean_worker,
# so it lives in ccplotter/ai/clean.py. composite.py never imports it; the two
# are linked only by a file-naming convention (src_<id>_texture.svg on disk)
# that _merge_texture_layer reads.
from ccplotter.composite import _source_svg, _merge_texture_layer, _grid, _composite


# _polyline_len_mm / estimate_plot_time / compute_vector_stats / build_warnings /
# _route_low_lift / _route_low_lift_doc are defined in ccplotter/stats_warnings.py.
# They are plain functions with no module-level mutable state, so bare
# re-exports are safe.
from ccplotter.stats_warnings import (_polyline_len_mm, estimate_plot_time,
                                       compute_vector_stats, build_warnings,
                                       _route_low_lift, _route_low_lift_doc)
# _inkscape_exe / trace_to_svg / _load_gray / _stroke_px_for / _JOIN_COS /
# _skeleton_to_polylines / _binarize_lineart / _micro_detail_mask /
# _outline_polys / _superres_contour_polys / _component_fidelity_detail /
# _binarize_mask / _load_native_gray / _smooth_polys / _gray_crush_metric are
# defined in ccplotter/mask.py: stateless mask/geometry helpers with no `global`
# statements, so bare re-exports are safe.
#
# Placement is by call graph. _superres_contour_polys sits here because its only
# caller is _component_fidelity_detail; putting it in trace_v3.py would make
# trace_v3 and mask import each other. _skeleton_to_polylines (and _JOIN_COS)
# is a shared geometry helper used by _component_fidelity_detail and every
# trace engine (sketch/trace_v3/hybrid/render), all of which import from mask,
# never the reverse. _smooth_polys and _gray_crush_metric are called by both
# _trace_svg (trace_v3.py) and _lineart_svg (lineart.py); keeping them in
# either module would force an import edge between those two.
from ccplotter.mask import (_inkscape_exe, trace_to_svg, _load_gray, _stroke_px_for,
                             _JOIN_COS, _skeleton_to_polylines, _binarize_lineart,
                             _micro_detail_mask, _outline_polys, _superres_contour_polys,
                             _component_fidelity_detail, _binarize_mask, _load_native_gray,
                             _smooth_polys, _gray_crush_metric)

# ── Line-art engine: ccplotter/lineart.py.
# sketch.py imports _lineart_svg (line art is the base geometry of the sketch
# engine), so lineart.py must not depend on sketch.py.
#
# _LINEART_AUDIT is rebound via `global _LINEART_AUDIT` inside _lineart_svg, so
# it is NOT re-exported as a bare name. Readers (render_raster_to_svg) use
# `_lineart_mod._LINEART_AUDIT` to see the live value.
import ccplotter.lineart as _lineart_mod
from ccplotter.lineart import _lineart_svg

# ── Sketch V2 engine: ccplotter/sketch.py (imports _lineart_svg from lineart.py).
#
# _SKETCH_AUDIT is rebound via `global _SKETCH_AUDIT` inside _sketch_svg (in the
# gain<=0 early return and in the normal path), so it is NOT re-exported as a
# bare name. Readers (api_process) use `_sketch_mod._SKETCH_AUDIT` to see the
# live value.
import ccplotter.sketch as _sketch_mod
from ccplotter.sketch import (_generate_stipple_points, _solve_tsp, _split_route_on_jumps,
                               _lsd_svg, _centerline_svg, _hatch_geom, _sketch_svg)

# ── Trace V3 engine: ccplotter/trace_v3.py.
# _lineart_quality_check is defined here; it is called only by
# render_raster_to_svg and the AI-clean family, not by sketch/lineart/hybrid.
# _route_predicate_batch is the vectorized form of _route_predicate_pass and is
# called only from _trace_svg, so it lives alongside it.
#
# _TRACE_AUDIT is rebound via `global _TRACE_AUDIT` inside _trace_svg, so it is
# NOT re-exported as a bare name. External readers (render_raster_to_svg) use
# `_trace_v3_mod._TRACE_AUDIT`. _coverage_debug reads it as a plain module
# global because it lives in the same module as the rebind site.
import ccplotter.trace_v3 as _trace_v3_mod
from ccplotter.trace_v3 import (_lineart_quality_check, _tracev3_mask, _gap_guarantee,
                                 _link_pockets, _level_set_fill, _outline_only,
                                 _multitone_passes, _route_predicate_pass,
                                 _route_predicate_batch, _trace_svg, _orientation_map,
                                 _gap_widths_mm, _gap_audit, _preflight, _coverage_debug)

# ── Hybrid engine: ccplotter/hybrid.py.
# Includes HYBRID_PRESETS and HYBRID_DETAIL_PARAMS, which are used only by this
# engine and so live with it rather than in config.py. No `global` statements,
# so every symbol is a safe bare re-export.
#
# hybrid.py is an independent leaf on config/mask/svg_io: its functions never
# call _sketch_svg/_lineart_svg/_trace_svg. Only the render dispatcher,
# render_raster_to_svg, calls _hybrid_svg.
from ccplotter.hybrid import (HYBRID_PRESETS, HYBRID_DETAIL_PARAMS, _poly_len_px,
                               _resample_pts, _grad_mag, _line_support,
                               _hybrid_geom_candidates, _hybrid_edge_candidates,
                               _hybrid_ink_candidates, _hybrid_auto_weights,
                               _hybrid_fuse, _hybrid_stitch, _hybrid_objective,
                               _merge_rois, _find_small_detail_rois,
                               _hybrid_detail_roi_candidates, _hybrid_accept_detail,
                               _write_colored_svg, _hybrid_svg)

# ── Render dispatcher: ccplotter/render.py.
# Depends on mask/svg_io/sketch/trace_v3/lineart/hybrid. render_raster_to_svg
# has no `global` statements. It reads the lineart and trace_v3 audit globals
# through its own `import ccplotter.lineart as _lineart_mod` /
# `import ccplotter.trace_v3 as _trace_v3_mod`, so it always sees the live
# values.
from ccplotter.render import render_raster_to_svg

# ── Local AI cleanup (ComfyUI bridge, runs on the user's GPU) ────────────────
# Architecture: AI converts ANY raster into a clean B&W stroke rendering
# (style = mode), then the deterministic backend (Plot-Ready trace → routing →
# gcode) does the rest. The AI result is saved as a NEW source — the original
# upload is never overwritten. Requires a running ComfyUI (see the ComfyUI documentation);
# everything degrades gracefully with a clear message when it's not up.
#
# Defined in ccplotter/ai/comfy.py: _ai_cfg, _comfy_url, _comfy_available,
# _comfy_checkpoints, _comfy_controlnets, _comfy_has_node, _comfy_kontext_unet,
# _ai_kontext_workflow(_multiref), _comfy_upload, _comfy_free,
# _ai_custom_workflow, _ai_workflow, _comfy_run, _comfy_run_cancellable(_multi),
# class _AICancelled.
#
# AI_PROMPTS lives in ai/comfy.py (not ai/clean.py) because _ai_workflow
# (`pr = prompts or AI_PROMPTS[style]`) is its only code dependency, and
# ai/clean.py already imports from ai/comfy.py.
#
# No `global` statements in this group, so bare re-exports are safe.
from ccplotter.ai.comfy import (AI_PROMPTS, _ai_cfg, _comfy_url, _comfy_available,
                                 _comfy_checkpoints, _comfy_controlnets, _comfy_has_node,
                                 _comfy_kontext_unet, _ai_kontext_workflow,
                                 _ai_kontext_workflow_multiref, _comfy_upload, _comfy_free,
                                 _ai_custom_workflow, _ai_workflow, _comfy_run,
                                 _AICancelled, _comfy_run_cancellable,
                                 _comfy_run_cancellable_multi)

# ── OpenAI line-art provider (cloud; replaces local ComfyUI when configured) ──
# Defined in ccplotter/ai/openai_provider.py; no `global` statements.
from ccplotter.ai.openai_provider import _openai_key, _use_openai, _openai_lineart_bytes

# ── AI Clean for Plotter (async job workflow, ComfyUI/OpenAI cleanup) ───────
# Defined in ccplotter/ai/clean.py: AI_KONTEXT_PROMPTS, AI_STYLE_PROMPTS,
# AI_CLEAN_PRESETS, AI_CLEAN_BACKGROUND, AI_CLEAN_FIDELITY, AI_CLEAN_SPEEDS,
# _AI_LOCK, _AI_JOBS, _ai_postclean, _ai_lineart_cleanup, _texture_backfill,
# _ai_verdict, _ai_clean_worker. _texture_backfill's only caller is
# _ai_clean_worker, and its dependencies are svg_io/mask, never composite.py.
#
# _AI_LOCK (a Lock) and _AI_JOBS (a dict mutated by item assignment) are never
# rebound, so bare re-exports share the same objects. There are no `global`
# statements in this group.
from ccplotter.ai.clean import (AI_KONTEXT_PROMPTS, AI_STYLE_PROMPTS, AI_CLEAN_PRESETS,
                                 AI_CLEAN_BACKGROUND, AI_CLEAN_FIDELITY, AI_CLEAN_SPEEDS,
                                 _AI_LOCK, _AI_JOBS, _ai_postclean, _ai_lineart_cleanup,
                                 _texture_backfill, _ai_verdict, _ai_clean_worker)


# ── Flask app ─────────────────────────────────────────────────────────────────
# The Flask() instance lives in ccplotter/routes/__init__.py, which avoids the
# duplicate-module hazard of the same file being imported as both `__main__`
# and `app`. app.py re-exports the name `app` (desktop/ccplotter_app.py does
# `import app as ccapp; ccapp.app...`).
#
# Importing each ccplotter.routes submodule below, for its side effect only,
# fires its @app.route(...) decorators and registers the handlers on the shared
# `app`. The groups are core (index, /api/calibrate_z, /api/reconnect,
# /api/version, /api/status, /api/calibration, /api/add_drawing, /api/add_text,
# /api/clear_drawings, /api/delete_drawing, /api/process under _PROCESS_LOCK,
# /work/<f>, /download_bundle, /api/send, /api/gcode, /api/fans_off), pen, runs,
# ai_clean, studio, dbv3 and bench. Flask's url_map does not depend on
# registration order.
from ccplotter.routes import app  # noqa: F401 (re-exported; see module comment)
from ccplotter.routes import core  # noqa: F401 (registration side effect only)
from ccplotter.routes import pen  # noqa: F401 (registration side effect only)
from ccplotter.routes import runs as _routes_runs  # noqa: F401 (registration side effect only; aliased -- distinct from the ccplotter.runs domain module already imported above)
from ccplotter.routes import ai_clean  # noqa: F401 (registration side effect only)
from ccplotter.routes import studio as _routes_studio  # noqa: F401 (registration side effect only; aliased -- distinct from the ccplotter.studio domain module already imported above)
from ccplotter.routes import dbv3 as _routes_dbv3  # noqa: F401 (registration side effect only; full-integration launch/poll routes, see ccplotter/routes/dbv3.py)
from ccplotter.routes import bench as _routes_bench  # noqa: F401 (registration side effect only; aliased -- distinct from the ccplotter.bench domain module already imported above; last route group; app.py defines no @app.route(...) handlers of its own)


def resolve_dev_host():
    """Dev-server bind address: CCPLOTTER_DEV_HOST if set, else loopback only."""
    import os
    return os.environ.get("CCPLOTTER_DEV_HOST") or "127.0.0.1"


if __name__ == "__main__":
    # STACKED-SERVER GUARD: SO_REUSEADDR on Windows lets multiple instances bind
    # :5001 simultaneously — stale processes then serve old code (recurring,
    # very confusing). Refuse to start if something already answers on 5001.
    import os
    import socket
    # The port defaults to 5001 and can be overridden with CCPLOTTER_DEV_PORT
    # (or CCPLOTTER_HARNESS_PORT, so one variable drives both the app and the
    # replay harness). A port can be wedged, held by a process that cannot be
    # killed, so the override lets the app run elsewhere without weakening the
    # stacked-server check, which applies to whichever port is chosen.
    _port = int(os.environ.get("CCPLOTTER_DEV_PORT")
                or os.environ.get("CCPLOTTER_HARNESS_PORT")
                or "5001")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    if probe.connect_ex(("127.0.0.1", _port)) == 0:
        probe.close()
        raise SystemExit(f"ERROR: another server is already running on :{_port} - kill it first "
                         f"(netstat -ano | findstr :{_port}), then restart.")
    probe.close()
    # The printer status websocket is dialed here, never at import.
    start_printer()
    # Bind to loopback by default; CCPLOTTER_DEV_HOST overrides (see resolve_dev_host).
    app.run(host=resolve_dev_host(), port=_port, debug=False, threaded=True)
