"""ccplotter/routes/studio.py -- Source Art Studio Flask routes (Modularize
routes: /model_images/<fn>,
/api/studio/images, /api/studio/pairs, /api/studio/style_card,
/api/studio/generate, /api/studio/status/<jid>, /api/studio/cancel/<jid>,
/api/studio/use.

Shared state: _STUDIO_JOBS is a dict mutated only by item assignment
(`_STUDIO_JOBS[jid] = job` in api_studio_generate) and never rebound, so
importing it by name is safe. _AI_LOCK (shared with ai_clean.py/bench.py) is
likewise never rebound.
"""
import json
import os
import time
import threading
from uuid import uuid4

from flask import jsonify, request, send_from_directory

from ccplotter.atomic_io import atomic_copy, atomic_write_text
from ccplotter.config import (MODEL_IMAGES_DIR, RASTER_EXTS, STUDIO_PAIRS,
                               STUDIO_STYLECARD, WORK_DIR)
from ccplotter.ai.clean import AI_CLEAN_SPEEDS, _AI_LOCK
from ccplotter.studio import (STUDIO_STYLE_CARD_DEFAULT, STUDIO_CANDIDATE_STYLES,
                               _studio_style_card, _studio_slug, _STUDIO_JOBS, _studio_worker)

from ccplotter.routes import app

@app.route("/model_images/<path:fn>")
def serve_model_image(fn):
    return send_from_directory(MODEL_IMAGES_DIR, fn)

@app.route("/api/studio/images")
def api_studio_images():
    """List everything in Model Images so the user can pair originals to targets."""
    if not MODEL_IMAGES_DIR.exists():
        return jsonify({"images": []})
    imgs = []
    for p in sorted(MODEL_IMAGES_DIR.iterdir()):
        if p.suffix.lower() in RASTER_EXTS:
            imgs.append({"name": p.name, "url": f"/model_images/{p.name}",
                         "guess": "target" if "chatgpt" in p.name.lower() else "original",
                         "size_kb": round(p.stat().st_size / 1024)})
    return jsonify({"images": imgs})

@app.route("/api/studio/pairs", methods=["GET", "POST"])
def api_studio_pairs():
    if request.method == "POST":
        atomic_write_text(STUDIO_PAIRS, json.dumps(request.json or [], indent=1))
        return jsonify({"ok": True})
    try:
        return jsonify(json.loads(STUDIO_PAIRS.read_text()))
    except Exception:
        return jsonify([])

@app.route("/api/studio/style_card", methods=["GET", "POST"])
def api_studio_style_card():
    if request.method == "POST":
        atomic_write_text(STUDIO_STYLECARD, str((request.json or {}).get("text", "")), encoding="utf-8")
        return jsonify({"ok": True})
    return jsonify({"text": _studio_style_card(), "default": STUDIO_STYLE_CARD_DEFAULT})


@app.route("/api/studio/generate", methods=["POST"])
def api_studio_generate():
    body = request.json or {}
    styles = [s for s in body.get("styles", list(STUDIO_CANDIDATE_STYLES)) if s in STUDIO_CANDIDATE_STYLES]
    if not styles:
        return jsonify({"error": "no valid styles"}), 400
    speed = body.get("speed", "fast")
    if speed not in AI_CLEAN_SPEEDS:
        return jsonify({"error": f"unknown speed '{speed}'"}), 400
    # source: a Model Images filename, or a layout drawing id
    img_name = body.get("image"); did = body.get("drawing_id")
    if img_name:
        src = MODEL_IMAGES_DIR / img_name
        if not src.is_file(): return jsonify({"error": f"image '{img_name}' not found"}), 404
        source_name = img_name
    elif did:
        raw = [m for m in WORK_DIR.glob(f"src_{did}.*") if "_render_" not in m.name and not m.name.endswith("_vector.svg")]
        if not raw: return jsonify({"error": f"drawing {did} not found"}), 404
        src, source_name = raw[0], raw[0].name
    else:
        return jsonify({"error": "provide image or drawing_id"}), 400
    # Reference-guided experiment: resolve the paired ChatGPT target for this
    # source (from the pairs manifest). Only when use_reference + a pair exists.
    use_reference = bool(body.get("use_reference", False))
    target_path = target_name = None
    if use_reference:
        explicit = body.get("target")
        if explicit and (MODEL_IMAGES_DIR / explicit).is_file():
            target_name = explicit
        else:
            try:
                pairs = json.loads(STUDIO_PAIRS.read_text())
                pr = next((p for p in pairs if p.get("original") == source_name), None)
                target_name = pr.get("target") if pr else None
            except Exception:
                target_name = None
        if target_name and (MODEL_IMAGES_DIR / target_name).is_file():
            target_path = str(MODEL_IMAGES_DIR / target_name)
        else:
            use_reference = False                       # no pair → skip silently (still 4 candidates)
    if not _AI_LOCK.acquire(blocking=False):
        return jsonify({"error": "An AI job is already running — wait or cancel it."}), 429
    jid = uuid4().hex[:8]
    job = {"id": jid, "status": "running", "phase": "queued", "t0": time.time(), "cancel": False,
           "src_path": str(src), "source_name": source_name, "styles": styles, "speed": speed,
           "case": _studio_slug(source_name), "results": [],
           "crisp": bool(body.get("crisp", True)), "recover": bool(body.get("recover", True)),
           "use_reference": use_reference, "target_path": target_path, "target_name": target_name,
           "seed": int(body.get("seed", 0)) or int.from_bytes(os.urandom(4), "little")}
    _STUDIO_JOBS[jid] = job
    threading.Thread(target=_studio_worker, args=(job,), daemon=True).start()
    return jsonify({"job_id": jid, "case": job["case"], "reference": target_name if use_reference else None})

@app.route("/api/studio/status/<jid>")
def api_studio_status(jid):
    job = _STUDIO_JOBS.get(jid)
    if not job: return jsonify({"error": "unknown job"}), 404
    return jsonify({"job_id": jid, "status": job["status"], "phase": job["phase"],
                    "elapsed_s": round(time.time() - job["t0"], 1),
                    "results": job.get("results", []), "error": job.get("error"),
                    "ref_error": job.get("ref_error"), "reference": job.get("target_name"),
                    "case": job["case"], "source_name": job.get("source_name")})

@app.route("/api/studio/cancel/<jid>", methods=["POST"])
def api_studio_cancel(jid):
    job = _STUDIO_JOBS.get(jid)
    if job: job["cancel"] = True
    return jsonify({"ok": True})

@app.route("/api/studio/use", methods=["POST"])
def api_studio_use():
    """Push a chosen candidate into the trace pipeline as a new drawing source."""
    import shutil
    body = request.json or {}
    rel = body.get("file", "")
    src = WORK_DIR / rel
    if not rel or not src.is_file():
        return jsonify({"error": "candidate file not found"}), 404
    did = uuid4().hex[:8]
    dst = WORK_DIR / f"src_{did}.png"
    atomic_copy(src, dst)
    return jsonify({"id": did, "src_name": dst.name})
