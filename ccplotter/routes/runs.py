"""ccplotter/routes/runs.py -- variant-cache switcher + run-archive Flask
routes: /api/variants, /api/variants/use, /api/trace_compare, /api/runs,
/api/runs/<rid> (DELETE), /api/runs/<rid>/use_source.

NOT to be confused with the domain module ccplotter/runs.py (the run-archive
storage layer -- RUNS_DIR/_runs_load/_runs_add); this is the distinct
ccplotter.routes.runs submodule (route HANDLERS only), a different
fully-qualified import path with no collision.

Shared state: _vcache_mod._VCACHE_FP / _vcache_mod._VCACHE_CURRENT are read
through the live module attribute, because ccplotter.vcache rebinds both with
`global` (a name imported by value would go stale). _VCACHE (a dict mutated in
place) and _VCACHE_LOCK (a Lock) are never rebound and are imported by name.
"""
import json
import time
from uuid import uuid4

import numpy as np
import vpype as vp
from flask import jsonify, request

from ccplotter.atomic_io import atomic_copy, atomic_write_text
from ccplotter.config import PX2MM, RASTER_EXTS, WORK_DIR
from ccplotter.render import render_raster_to_svg
from ccplotter.runs import RUNS_DIR, RUNS_INDEX, _RUNS_LOCK, _runs_load
import ccplotter.vcache as _vcache_mod
from ccplotter.vcache import _VCACHE, _VCACHE_LOCK, _vcache_get

from ccplotter.routes import app

# ── Current-image variant cache API ───────────────────────────────────────────

@app.route("/api/variants")
def api_variants():
    """List the cached variants for the currently-loaded source set (newest
    first), marking the one currently shown. Lets the UI render a temporary
    variant switcher for fast A/B without recompute."""
    with _VCACHE_LOCK:
        out = []
        for k, ent in sorted(_VCACHE.items(), key=lambda kv: -kv[1]["order"]):
            out.append({"key": k, "label": ent.get("label") or "variant",
                        "at": ent.get("at", ""), "active": (k == _vcache_mod._VCACHE_CURRENT),
                        "mode": ent["res"].get("style") or ent["res"].get("mode"),
                        "strokes": ent["res"].get("stroke_count"),
                        "z_moves": ent["res"].get("z_moves"),
                        "has_gcode": "layout.gcode" in ent.get("files", {})})
        return jsonify({"fingerprint": _vcache_mod._VCACHE_FP, "current": _vcache_mod._VCACHE_CURRENT, "variants": out})

@app.route("/api/variants/use", methods=["POST"])
def api_variants_use():
    """Restore a cached variant by key (instant, no recompute) and return its
    stored result — same shape as /api/process so the UI applies it identically."""
    key = (request.json or {}).get("key", "")
    with _VCACHE_LOCK:
        ent = _VCACHE.get(key)
        fp = _vcache_mod._VCACHE_FP
    if not ent:
        return jsonify({"error": "variant not in cache (it may have been evicted or the image changed)"}), 404
    res = _vcache_get(fp, key)                      # restores files + sets current
    if res is None:
        return jsonify({"error": "variant snapshot missing — reprocess to rebuild it"}), 410
    res["cached"] = True
    res["job_id"] = uuid4().hex[:8]
    return jsonify(res)

# ── Run archive API ──────────────────────────────────────────────────────────

@app.route("/api/trace_compare", methods=["POST"])
def api_trace_compare():
    """Render a clean line-art source at all four Ink Sensitivity levels and
    return mask + clean-vector previews + stats for each, so the user can pick
    the best fidelity/noise tradeoff. Audit tool — no routing, no Z optimization."""
    body = request.json or {}
    did = str(body.get("id", ""))
    raw = [m for m in WORK_DIR.glob(f"src_{did}.*") if "_render_" not in m.name and not m.name.endswith("_vector.svg")]
    if not raw:
        return jsonify({"error": f"drawing {did} not found"}), 404
    src = raw[0]
    if src.suffix.lower() not in RASTER_EXTS:
        return jsonify({"error": "Trace Compare is for raster line-art sources."}), 400
    cand = []
    for s in ("strict", "normal", "sensitive", "very_sensitive"):
        opts = {"mode": "plotready", "threshold_mode": "sensitivity", "ink_sensitivity": s,
                "lineart_min_area": 0, "hole_fill": 0, "lineart_gap_close": 0,
                "outline_simplify": 0.05, "resolution": int(body.get("resolution", 3000))}
        out = WORK_DIR / f"_cmp_{s}.svg"
        dbg = str(WORK_DIR / f"_cmp_{s}")
        try:
            render_raster_to_svg(src, out, opts, debug_prefix=dbg)
            doc = vp.read_multilayer_svg(str(out), quantization=0.1)
            lines = [l for lc in doc.layers.values() for l in lc]
            nstroke = len(lines)
            draw_mm = sum(lc.length() for lc in doc.layers.values()) * PX2MM
            tiny = sum(1 for l in lines if len(l) and float(np.abs(np.diff(l)).sum()) * PX2MM < 1.5)
            cand.append({"sensitivity": s, "ok": True,
                         "mask_url": f"/work/_cmp_{s}_mask_raw.png?t={int(time.time())}",
                         "preview_url": f"/work/_cmp_{s}.svg?t={int(time.time())}",
                         "strokes": nstroke, "tiny_strokes": tiny,
                         "draw_length_mm": round(draw_mm, 0),
                         "est_z_moves": nstroke * 2})
        except Exception as e:
            cand.append({"sensitivity": s, "ok": False, "error": str(e)[:200]})
    return jsonify({"id": did, "candidates": cand})

@app.route("/api/runs")
def api_runs():
    return jsonify(_runs_load())

@app.route("/api/runs/<rid>", methods=["DELETE"])
def api_runs_delete(rid):
    import shutil
    with _RUNS_LOCK:
        runs = [r for r in _runs_load() if r.get("id") != rid]
        atomic_write_text(RUNS_INDEX, json.dumps(runs, indent=1))
    shutil.rmtree(RUNS_DIR / rid, ignore_errors=True)
    return jsonify({"ok": True})

@app.route("/api/runs/<rid>/use_source", methods=["POST"])
def api_runs_use_source(rid):
    """Restore an archived AI-cleaned source as a fresh drawing source —
    NO AI re-run. Copies the snapshot back into work/ under a new id."""
    import shutil
    rec = next((r for r in _runs_load() if r.get("id") == rid), None)
    if not rec: return jsonify({"error": "unknown run"}), 404
    rel = (rec.get("files") or {}).get("cleaned")
    if not rel: return jsonify({"error": "this run has no cleaned source snapshot"}), 400
    src = WORK_DIR / rel
    if not src.exists(): return jsonify({"error": "snapshot file missing on disk"}), 404
    did = uuid4().hex[:8]
    dst = WORK_DIR / f"src_{did}{src.suffix}"
    atomic_copy(src, dst)
    return jsonify({"id": did, "src_name": dst.name, "run": rec})
