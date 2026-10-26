import json
import time
from pathlib import Path

from ccplotter.config import AI_CONFIG_PATH, AI_DEFAULTS

# ── Local AI cleanup (ComfyUI bridge, runs on the user's GPU) ────────────────
# Architecture: AI converts ANY raster into a clean B&W stroke rendering
# (style = mode), then the deterministic backend (Plot-Ready trace → routing →
# gcode) does the rest. The AI result is saved as a NEW source — the original
# upload is never overwritten. Requires a running ComfyUI (see the ComfyUI documentation);
# everything degrades gracefully with a clear message when it's not up.

# AI_PROMPTS lives here rather than in ai/clean.py. Its only code dependency is
# this module's _ai_workflow (`pr = prompts or AI_PROMPTS[style]`); the route
# that checks style membership just reads the re-exported name. Defining it in
# clean.py would make comfy.py import from clean.py, which is backwards from
# the dependency direction (ai/clean.py imports ai/comfy.py, never the reverse)
# and would create an import cycle.
AI_PROMPTS = {
    "lineart": {
        "pos": "lineart, line art, clean coloring book page, thin black ink outlines only, "
               "flat pure white background, monochrome outline drawing, no shading, no fills, high detail",
        "neg": "photo, photorealistic, color, shading, gradients, gray tones, solid black fill, "
               "silhouette, texture, noise, blur, watermark, signature",
    },
    "sketch": {
        "pos": "pen and ink sketch, black ink on white paper, cross-hatched shading, hand-drawn illustration, "
               "confident clean ink strokes, monochrome",
        "neg": "photo, photorealistic, color, blur, gray wash, gradients, solid black fill, watermark, signature",
    },
}

def _ai_cfg() -> dict:
    cfg = json.loads(json.dumps(AI_DEFAULTS))      # deep copy
    try:
        user = json.loads(AI_CONFIG_PATH.read_text())
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict): cfg[k].update(v)
            else: cfg[k] = v
    except Exception: pass
    return cfg

def _comfy_url() -> str: return _ai_cfg()["comfyui_url"].rstrip("/")

def _comfy_available() -> bool:
    import requests
    try: return requests.get(f"{_comfy_url()}/system_stats", timeout=3).ok
    except Exception: return False

def _comfy_checkpoints() -> list:
    import requests
    try:
        r = requests.get(f"{_comfy_url()}/object_info/CheckpointLoaderSimple", timeout=5).json()
        return list(r["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0])
    except Exception: return []

def _comfy_controlnets() -> list:
    import requests
    try:
        r = requests.get(f"{_comfy_url()}/object_info/ControlNetLoader", timeout=5).json()
        return list(r["ControlNetLoader"]["input"]["required"]["control_net_name"][0])
    except Exception: return []

def _comfy_has_node(class_type: str) -> bool:
    import requests
    try:
        r = requests.get(f"{_comfy_url()}/object_info/{class_type}", timeout=5).json()
        return class_type in r
    except Exception: return False

def _comfy_kontext_unet() -> str:
    """Return the installed FLUX Kontext diffusion-model filename, or ''."""
    import requests
    try:
        r = requests.get(f"{_comfy_url()}/object_info/UNETLoader", timeout=5).json()
        names = r["UNETLoader"]["input"]["required"]["unet_name"][0]
        return next((n for n in names if "kontext" in n.lower()), "")
    except Exception: return ""

def _ai_kontext_workflow(image_name: str, prompt: str, seed: int, unet: str, cfgd: dict) -> dict:
    return {
        "u": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "c": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors",
              "clip_name2": "t5xxl_fp8_e4m3fn_scaled.safetensors", "type": "flux", "device": "default"}},
        "v": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "s": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["2", 0]}},
        "e": {"class_type": "VAEEncode", "inputs": {"pixels": ["s", 0], "vae": ["v", 0]}},
        "p": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["c", 0], "text": prompt}},
        "r": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["p", 0], "latent": ["e", 0]}},
        "g": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["r", 0],
              "guidance": float(cfgd.get("kontext_guidance", 2.5))}},
        "z": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["p", 0]}},
        "k": {"class_type": "KSampler", "inputs": {"model": ["u", 0], "positive": ["g", 0], "negative": ["z", 0],
              "latent_image": ["e", 0], "seed": seed, "steps": int(cfgd.get("kontext_steps", 20)),
              "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "d": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["v", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["d", 0], "filename_prefix": "ccplotter_ai"}},
    }

def _ai_kontext_workflow_multiref(content_name: str, ref_name: str, prompt: str, seed: int,
                                  unet: str, cfgd: dict, method: str = "offset") -> dict:
    """Kontext NATIVE multi-reference graph (no extra models): content image +
    a style/reference image. Two chained ReferenceLatent nodes feed the
    FluxKontextMultiReferenceLatentMethod combiner; the KSampler denoises the
    CONTENT latent so composition follows the content image while the reference
    guides style/detail. Used by Source Art Studio's experimental
    reference-guided candidate. EXPERIMENTAL — Kontext is an edit model, not a
    dedicated style-transfer model."""
    return {
        "u": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "c": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors",
              "clip_name2": "t5xxl_fp8_e4m3fn_scaled.safetensors", "type": "flux", "device": "default"}},
        "v": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": content_name}},
        "2b": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        "s": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["2", 0]}},
        "sb": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["2b", 0]}},
        "e": {"class_type": "VAEEncode", "inputs": {"pixels": ["s", 0], "vae": ["v", 0]}},
        "eb": {"class_type": "VAEEncode", "inputs": {"pixels": ["sb", 0], "vae": ["v", 0]}},
        "p": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["c", 0], "text": prompt}},
        "r1": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["p", 0], "latent": ["e", 0]}},
        "r2": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["r1", 0], "latent": ["eb", 0]}},
        "m": {"class_type": "FluxKontextMultiReferenceLatentMethod",
              "inputs": {"conditioning": ["r2", 0], "reference_latents_method": method}},
        "g": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["m", 0],
              "guidance": float(cfgd.get("kontext_guidance", 2.5))}},
        "z": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["p", 0]}},
        "k": {"class_type": "KSampler", "inputs": {"model": ["u", 0], "positive": ["g", 0], "negative": ["z", 0],
              "latent_image": ["e", 0], "seed": seed, "steps": int(cfgd.get("kontext_steps", 20)),
              "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "d": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["v", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["d", 0], "filename_prefix": "ccplotter_ai"}},
    }

def _comfy_upload(path: Path) -> str:
    import requests
    with open(path, "rb") as f:
        r = requests.post(f"{_comfy_url()}/upload/image",
                          files={"image": (path.name, f, "image/png")},
                          data={"overwrite": "true"}, timeout=60)
    r.raise_for_status()
    return r.json()["name"]

def _comfy_free():
    """Ask ComfyUI to unload models + free VRAM. Needed before running the small
    LineArt annotator right after a FLUX/Kontext generation — otherwise the 11GB
    UNET is still resident and the annotator OOMs on the 16GB card."""
    import requests
    try:
        requests.post(f"{_comfy_url()}/free", json={"unload_models": True, "free_memory": True}, timeout=30)
        time.sleep(1.0)
    except Exception as e:
        print(f"  [ai] /free failed (non-fatal): {e}")

def _ai_custom_workflow(style: str, image_name: str, seed: int):
    """Power-user escape hatch: drop a ComfyUI API-format workflow as
    comfyui_workflow_<style>.json with __INPUT_IMAGE__ / __SEED__ tokens and the
    app will use it instead of the built-in img2img graph."""
    p = Path(f"comfyui_workflow_{style}.json")
    if not p.exists(): return None
    txt = p.read_text(encoding="utf-8").replace("__INPUT_IMAGE__", image_name).replace("__SEED__", str(seed))
    return json.loads(txt)

def _ai_workflow(image_name: str, style: str, ckpt: str, w: int, h: int, seed: int, cfgd: dict,
                 controlnet: str = None, has_lineart_pre: bool = False, prompts: dict = None) -> dict:
    """Built-in graph, best-available strategy per style:

    lineart + LineArtPreprocessor available → ANNOTATOR-ONLY: a small network
      purpose-built for photo→line-drawing. Deterministic, ~1s, no diffusion,
      no hallucination. This is the primary line-art engine.
    sketch (diffusion) → ControlNet-guided full-denoise restyle; guidance image
      is the lineart annotator output when available (smooth lines), else Canny.
    no ControlNet at all → plain img2img at moderate denoise (weak fallback)."""
    pr = prompts or AI_PROMPTS[style]
    wf = {
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "3": {"class_type": "ImageScale", "inputs": {"image": ["2", 0], "upscale_method": "lanczos",
                                                     "width": w, "height": h, "crop": "disabled"}},
    }

    if style == "lineart" and has_lineart_pre:
        # Pure annotator: photo → line drawing (white-on-black) → invert → save.
        # Variants trade fidelity for cleanliness: fine = all detail,
        # coarse = simplified strokes, anime = sparse clean lines (closest to a
        # stylized "ChatGPT-like" rendition).
        variant = str(cfgd.get("lineart_variant", "fine")).lower()
        if variant == "anime" and _comfy_has_node("AnimeLineArtPreprocessor"):
            wf["10"] = {"class_type": "AnimeLineArtPreprocessor", "inputs": {
                        "image": ["3", 0], "resolution": max(w, h)}}
        else:
            wf["10"] = {"class_type": "LineArtPreprocessor", "inputs": {
                        "image": ["3", 0], "coarse": "enable" if variant == "coarse" else "disable",
                        "resolution": max(w, h)}}
        wf["11"] = {"class_type": "ImageInvert", "inputs": {"image": ["10", 0]}}
        wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "ccplotter_ai"}}
        return wf

    # Diffusion paths need the checkpoint + prompts.
    wf["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}
    wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pr["pos"]}}
    wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pr["neg"]}}
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "ccplotter_ai"}}

    if controlnet:
        if has_lineart_pre:   # smooth annotator lines guide better than jagged canny
            wf["10"] = {"class_type": "LineArtPreprocessor", "inputs": {
                        "image": ["3", 0], "coarse": "disable", "resolution": max(w, h)}}
        else:
            wf["10"] = {"class_type": "Canny", "inputs": {"image": ["3", 0],
                        "low_threshold": float(cfgd.get("canny_low", 0.25)),
                        "high_threshold": float(cfgd.get("canny_high", 0.6))}}
        wf["11"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": controlnet}}
        wf["12"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
                    "positive": ["5", 0], "negative": ["6", 0], "control_net": ["11", 0],
                    "image": ["10", 0], "strength": float(cfgd.get("controlnet_strength", 0.75)),
                    "start_percent": 0.0,
                    "end_percent": float(cfgd.get("controlnet_end", 0.85))}}
        wf["13"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
        wf["7"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0],
                   "positive": ["12", 0], "negative": ["12", 1], "latent_image": ["13", 0],
                   "seed": seed, "steps": int(cfgd["steps"]), "cfg": float(cfgd["cfg"]),
                   "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}}
    else:
        wf["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
        wf["7"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0],
                   "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["4", 0],
                   "seed": seed, "steps": int(cfgd["steps"]), "cfg": float(cfgd["cfg"]),
                   "sampler_name": "euler", "scheduler": "normal",
                   "denoise": float(cfgd["denoise"].get(style, 0.7))}}
    return wf

def _comfy_run(workflow: dict, timeout_s: float) -> dict:
    """Submit a workflow, poll history until done, return the first output image
    descriptor {filename, subfolder, type}."""
    import requests
    r = requests.post(f"{_comfy_url()}/prompt", json={"prompt": workflow}, timeout=30)
    if not r.ok:
        raise RuntimeError(f"ComfyUI rejected the workflow: {r.text[:300]}")
    pid = r.json()["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        h = requests.get(f"{_comfy_url()}/history/{pid}", timeout=10).json()
        if pid in h:
            entry = h[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                raise RuntimeError(f"ComfyUI execution error: {msgs[-1][1].get('exception_message','?') if msgs else status}")
            for node in entry.get("outputs", {}).values():
                for im in node.get("images", []):
                    return im
        time.sleep(1.0)
    raise RuntimeError(f"ComfyUI timed out after {timeout_s:.0f}s")

class _AICancelled(Exception): pass

def _comfy_run_cancellable(workflow: dict, timeout_s: float, job: dict) -> dict:
    """Like _comfy_run, but checks the job's cancel flag every poll and
    interrupts ComfyUI when set."""
    import requests
    r = requests.post(f"{_comfy_url()}/prompt", json={"prompt": workflow}, timeout=30)
    if not r.ok:
        raise RuntimeError(f"ComfyUI rejected the workflow: {r.text[:300]}")
    pid = r.json()["prompt_id"]
    job["prompt_id"] = pid
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if job.get("cancel"):
            try:
                requests.post(f"{_comfy_url()}/queue", json={"delete": [pid]}, timeout=5)
                requests.post(f"{_comfy_url()}/interrupt", timeout=5)
            except Exception: pass
            raise _AICancelled()
        h = requests.get(f"{_comfy_url()}/history/{pid}", timeout=10).json()
        if pid in h:
            entry = h[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                raise RuntimeError(f"ComfyUI execution error: {msgs[-1][1].get('exception_message','?') if msgs else status}")
            for node in entry.get("outputs", {}).values():
                for im in node.get("images", []):
                    return im
        time.sleep(1.0)
    raise RuntimeError(f"ComfyUI timed out after {timeout_s:.0f}s")

def _comfy_run_cancellable_multi(workflow: dict, timeout_s: float, job: dict) -> dict:
    """Like _comfy_run_cancellable but returns the FULL outputs dict
    (node_id -> {images:[...]}) so a multi-output (batched) graph can be read."""
    import requests
    r = requests.post(f"{_comfy_url()}/prompt", json={"prompt": workflow}, timeout=30)
    if not r.ok:
        raise RuntimeError(f"ComfyUI rejected the workflow: {r.text[:300]}")
    pid = r.json()["prompt_id"]
    job["prompt_id"] = pid
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if job.get("cancel"):
            try:
                requests.post(f"{_comfy_url()}/queue", json={"delete": [pid]}, timeout=5)
                requests.post(f"{_comfy_url()}/interrupt", timeout=5)
            except Exception: pass
            raise _AICancelled()
        h = requests.get(f"{_comfy_url()}/history/{pid}", timeout=10).json()
        if pid in h:
            entry = h[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                raise RuntimeError(f"ComfyUI execution error: {msgs[-1][1].get('exception_message','?') if msgs else status}")
            outs = entry.get("outputs", {})
            if outs:
                return outs
        time.sleep(1.0)
    raise RuntimeError(f"ComfyUI timed out after {timeout_s:.0f}s")
