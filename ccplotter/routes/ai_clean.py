"""ccplotter/routes/ai_clean.py -- AI Clean for Plotter Flask routes
Routes: /api/ai_status, /api/ai_clean, /api/ai_clean/start,
/api/ai_clean/status/<job_id>, /api/ai_clean/cancel/<job_id>.

The AI backends (ComfyUI, OpenAI) are optional; these routes report a clear
error or status when a server or key is unavailable instead of failing.

Shared state: _AI_LOCK (a Lock) and _AI_JOBS (a dict mutated only by item
assignment) are never rebound in ccplotter.ai.clean, so importing them by name
here is safe (a name that is rebound with `global` would go stale in this
module).
"""
import json
import os
import threading
import time
from uuid import uuid4

from flask import jsonify, request

from ccplotter.atomic_io import atomic_target, atomic_write_bytes, atomic_write_text
from ccplotter.config import (AI_CONFIG_PATH, AI_DEFAULTS, AI_OPENAI_MODELS,
                               RASTER_EXTS, WORK_DIR)
from ccplotter.ai.comfy import (AI_PROMPTS, _ai_cfg, _comfy_available, _comfy_checkpoints,
                                 _comfy_controlnets, _comfy_has_node, _comfy_kontext_unet,
                                 _ai_kontext_workflow, _comfy_upload, _ai_custom_workflow,
                                 _ai_workflow, _comfy_run, _comfy_url)
from ccplotter.ai.openai_provider import _openai_key, _use_openai
from ccplotter.ai.clean import (AI_KONTEXT_PROMPTS, AI_CLEAN_PRESETS, AI_CLEAN_FIDELITY,
                                 AI_CLEAN_SPEEDS, AI_CLEAN_BACKGROUND, _AI_LOCK, _AI_JOBS,
                                 _ai_clean_worker)

from ccplotter.routes import app

@app.route("/api/ai_status")
def api_ai_status():
    cfgd = _ai_cfg()
    use_oa = _use_openai()                       # short-circuits the comfy probe when on OpenAI
    ok = use_oa or _comfy_available()
    if not AI_CONFIG_PATH.exists():
        try: atomic_write_text(AI_CONFIG_PATH, json.dumps(AI_DEFAULTS, indent=2))   # discoverable config
        except Exception: pass
    return jsonify({"available": ok, "provider": cfgd.get("provider", "openai"),
                    "openai_ready": use_oa, "openai_model": cfgd.get("openai_model", "gpt-image-1"),
                    "url": _comfy_url(),
                    "checkpoints": [] if use_oa else (_comfy_checkpoints() if ok else []),
                    "checkpoint_configured": cfgd.get("checkpoint", "")})

@app.route("/api/ai_clean", methods=["POST"])
def api_ai_clean():
    """AI front-end: convert a raster source into a clean B&W stroke rendering
    (style: lineart | sketch) via local ComfyUI. Saves the result as a NEW
    source (original preserved) and returns its drawing id."""
    body = request.json or {}
    did, style = str(body.get("id", "")), str(body.get("style", "lineart"))
    if style not in AI_PROMPTS: return jsonify({"error": "style must be 'lineart' or 'sketch'"}), 400
    raw = [m for m in WORK_DIR.glob(f"src_{did}.*") if "_render_" not in m.name and not m.name.endswith("_vector.svg")]
    if not raw: return jsonify({"error": f"drawing {did} not found"}), 404
    src = raw[0]
    if src.suffix.lower() not in RASTER_EXTS:
        return jsonify({"error": "AI cleanup is for raster images — this source is already vector."}), 400
    if not _comfy_available():
        return jsonify({"error": f"ComfyUI is not running at {_comfy_url()}. Start it, then retry."}), 503
    if not _AI_LOCK.acquire(blocking=False):
        return jsonify({"error": "An AI job is already running — wait for it to finish."}), 429
    t0 = time.time()
    try:
        cfgd = _ai_cfg()
        cfgd["lineart_variant"] = str(body.get("lineart_variant", cfgd.get("lineart_variant", "fine")))
        use_kontext = cfgd["lineart_variant"] == "kontext"
        kontext_unet = _comfy_kontext_unet() if use_kontext else ""
        if use_kontext and not kontext_unet:
            return jsonify({"error": "FLUX Kontext model not installed in ComfyUI (models/diffusion_models)."}), 503
        has_lineart_pre = _comfy_has_node("LineArtPreprocessor")
        needs_ckpt = not use_kontext and not (style == "lineart" and has_lineart_pre)
        ckpt = cfgd.get("checkpoint") or ""
        if not ckpt and needs_ckpt:
            cks = _comfy_checkpoints()
            if not cks:
                return jsonify({"error": "ComfyUI has no checkpoints installed. Put a .safetensors model in ComfyUI/models/checkpoints."}), 503
            ckpt = cks[0]
        # Prepare input at an SD-friendly size (dims multiple of 8).
        # Kontext rescales internally (FluxKontextImageScale) — give it more pixels.
        from PIL import Image
        im = Image.open(src).convert("RGB")
        max_side = 1536 if use_kontext else float(cfgd["max_side"])
        s = min(1.0, max_side / max(im.size))
        w = max(64, int(im.width * s) // 8 * 8)
        h = max(64, int(im.height * s) // 8 * 8)
        tmp = WORK_DIR / f"ai_in_{uuid4().hex[:6]}.png"
        im.save(tmp)
        try: image_name = _comfy_upload(tmp)
        finally: tmp.unlink(missing_ok=True)
        seed = int(body.get("seed", 0)) or int.from_bytes(os.urandom(4), "little")
        # ControlNet pick: prefer lineart (pairs with the annotator), else canny.
        cn = cfgd.get("controlnet")
        if cn is None or cn == "":
            cns = _comfy_controlnets()
            cn = (next((c for c in cns if "lineart" in c.lower()), None) if has_lineart_pre else None) \
                 or next((c for c in cns if "canny" in c.lower()), None)
        elif cn == "off":
            cn = None
        if use_kontext:
            wf = _ai_kontext_workflow(image_name, AI_KONTEXT_PROMPTS[style], seed, kontext_unet, cfgd)
            run_timeout = max(float(cfgd["timeout_s"]), 1800.0)   # first run loads ~17GB of weights
        else:
            wf = _ai_custom_workflow(style, image_name, seed) or _ai_workflow(
                image_name, style, ckpt, w, h, seed, cfgd, controlnet=cn, has_lineart_pre=has_lineart_pre)
            run_timeout = float(cfgd["timeout_s"])
        iminfo = _comfy_run(wf, run_timeout)
        import requests
        r = requests.get(f"{_comfy_url()}/view", params={"filename": iminfo["filename"],
                         "subfolder": iminfo.get("subfolder", ""), "type": iminfo.get("type", "output")}, timeout=120)
        r.raise_for_status()
        newdid = uuid4().hex[:8]
        out_path = WORK_DIR / f"src_{newdid}.png"
        atomic_write_bytes(out_path, r.content)   # NEW source; original untouched
        if style == "lineart" and has_lineart_pre and not use_kontext:
            # Annotator lines are soft gray — stretch so they trace as solid ink.
            # Darkest 1% → black, anything near-white stays white.
            import cv2, numpy as np
            g = cv2.imread(str(out_path), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                lo = float(np.percentile(g, 1))
                hi = 250.0
                stretched = np.clip((g.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)
                with atomic_target(out_path) as tmp_out:
                    cv2.imwrite(str(tmp_out), stretched)
        elapsed = round(time.time() - t0, 1)
        engine = ("kontext" if use_kontext else
                  "annotator" if (style == "lineart" and has_lineart_pre) else
                  f"diffusion ckpt={ckpt} cn={cn or 'off'}")
        print(f"  [ai] style={style} engine={engine} seed={seed} {w}x{h} elapsed={elapsed}s -> src_{newdid}.png")
        return jsonify({"id": newdid, "name": f"AI {style}", "is_raster": True,
                        "src_name": f"src_{newdid}.png", "elapsed_s": elapsed, "engine": engine,
                        "seed": seed, "checkpoint": ckpt, "controlnet": cn or "off", "source_id": did})
    except Exception as e:
        return jsonify({"error": f"AI cleanup failed: {e}"}), 500
    finally:
        _AI_LOCK.release()

@app.route("/api/ai_clean/start", methods=["POST"])
def api_ai_clean_start():
    """Start an async AI Clean for Plotter job. Returns a job_id immediately;
    the browser polls /api/ai_clean/status/<job_id>. The cleaned result becomes
    a NEW source file — the original is never modified or auto-replaced."""
    body = request.json or {}
    did = str(body.get("id", ""))
    preset = str(body.get("preset", "sparse"))
    fidelity = str(body.get("fidelity", "balanced"))
    speed = str(body.get("speed", "fast"))
    if preset not in AI_CLEAN_PRESETS: return jsonify({"error": f"unknown preset '{preset}'"}), 400
    if fidelity not in AI_CLEAN_FIDELITY: return jsonify({"error": f"unknown fidelity '{fidelity}'"}), 400
    if speed not in AI_CLEAN_SPEEDS: return jsonify({"error": f"unknown speed '{speed}'"}), 400
    backfill = str(body.get("backfill", "off"))
    if backfill not in ("off", "light", "full"): return jsonify({"error": f"unknown backfill '{backfill}'"}), 400
    bg_detail = str(body.get("background_detail", "balanced"))
    if bg_detail not in AI_CLEAN_BACKGROUND: return jsonify({"error": f"unknown background_detail '{bg_detail}'"}), 400
    quality = str(body.get("quality", "medium"))           # OpenAI image quality (Standard/High toggle)
    if quality not in ("low", "medium", "high"): quality = "medium"
    model = str(body.get("model", ""))                      # OpenAI image model (UI selector)
    if model not in AI_OPENAI_MODELS: model = _ai_cfg().get("openai_model", "chatgpt-image-latest")
    raw = [m for m in WORK_DIR.glob(f"src_{did}.*") if "_render_" not in m.name and not m.name.endswith("_vector.svg")]
    if not raw: return jsonify({"error": f"drawing {did} not found"}), 404
    src = raw[0]
    if src.suffix.lower() not in RASTER_EXTS:
        return jsonify({"error": "AI cleanup is for raster images — this source is already vector."}), 400
    if _ai_cfg().get("provider", "openai") == "openai":
        if not _openai_key():
            return jsonify({"error": "OpenAI provider selected but no API key found. Set OPENAI_API_KEY (or openai_key.txt), then retry."}), 503
    elif not _comfy_available():
        return jsonify({"error": f"ComfyUI is not running at {_comfy_url()}. Start it, then retry."}), 503
    sb = body.get("subject_box")
    if sb:
        try:
            x0, y0, x1, y1 = (float(v) for v in sb)
            ok_box = 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0 and (x1 - x0) >= 0.05 and (y1 - y0) >= 0.05
        except Exception:
            ok_box = False
        if not ok_box:
            return jsonify({"error": "subject_box must be [x0,y0,x1,y1], normalized 0-1, at least 5% in each dimension"}), 400
        sb = [x0, y0, x1, y1]
    if not _AI_LOCK.acquire(blocking=False):
        return jsonify({"error": "An AI job is already running — wait for it or cancel it first."}), 429
    job_id = uuid4().hex[:8]
    job = {"id": job_id, "status": "running", "phase": "queued", "t0": time.time(),
           "style": str(body.get("style", "lineart")),   # OpenAI style prompt selector
           "openai_quality": quality, "openai_model": model,
           "preset": preset, "fidelity": fidelity, "speed": speed, "cancel": False,
           "backfill": backfill, "background_detail": bg_detail,
           "src_path": str(src), "source_id": did, "subject_box": sb or None,
           "seed": int(body.get("seed", 0)) or int.from_bytes(os.urandom(4), "little")}
    _AI_JOBS[job_id] = job
    threading.Thread(target=_ai_clean_worker, args=(job,), daemon=True).start()
    return jsonify({"job_id": job_id})

@app.route("/api/ai_clean/status/<job_id>")
def api_ai_clean_status(job_id):
    job = _AI_JOBS.get(job_id)
    if not job: return jsonify({"error": "unknown job"}), 404
    out = {"job_id": job_id, "status": job["status"], "phase": job["phase"],
           "elapsed_s": round(time.time() - job["t0"], 1),
           "preset_label": AI_CLEAN_PRESETS[job["preset"]]["label"],
           "fidelity_label": AI_CLEAN_FIDELITY[job["fidelity"]]["label"],
           "speed_label": AI_CLEAN_SPEEDS[job["speed"]]["label"],
           "engine": job.get("engine", "")}
    if job["status"] == "done":
        out["result"] = job["result"]
        out["elapsed_s"] = job["result"]["elapsed_s"]
    elif job["status"] == "error":
        out["error"] = job.get("error", "unknown error")
    return jsonify(out)

@app.route("/api/ai_clean/cancel/<job_id>", methods=["POST"])
def api_ai_clean_cancel(job_id):
    job = _AI_JOBS.get(job_id)
    if not job: return jsonify({"error": "unknown job"}), 404
    if job["status"] != "running":
        return jsonify({"ok": False, "status": job["status"]})
    job["cancel"] = True
    # Poke ComfyUI right away so a running sampler stops (single-job design,
    # so interrupting "whatever runs" is always our job).
    import requests
    try: requests.post(f"{_comfy_url()}/interrupt", timeout=5)
    except Exception: pass
    return jsonify({"ok": True})
