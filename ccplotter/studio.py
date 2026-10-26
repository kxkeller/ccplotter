import json
import time
from pathlib import Path
from uuid import uuid4

from ccplotter.config import STUDIO_DIR, STUDIO_PAIRS, STUDIO_STYLECARD
from ccplotter.ai.comfy import (_ai_cfg, _comfy_url, _comfy_available, _comfy_kontext_unet,
                                 _ai_kontext_workflow, _ai_kontext_workflow_multiref,
                                 _comfy_upload, _comfy_free, _AICancelled,
                                 _comfy_run_cancellable)
from ccplotter.ai.clean import AI_CLEAN_SPEEDS, _AI_LOCK

# ── Source Art Studio ─────────────────────────────────────────────────────────
# Reference-guided local source-art generation. Uses the user's photo/ChatGPT
# pairs as TEACHER examples (not training): a hand-editable Style Card distilled
# from them becomes the master prompt ingredient, and each new photo generates
# several candidates the user compares against the reference target.
#
# STUDIO_STYLE_CARD_DEFAULT lives here with its real caller: it is
# _studio_style_card's own fallback value (this module); the only other
# reference is the api_studio_style_card route, which imports it by name.
STUDIO_STYLE_CARD_DEFAULT = (
    "Bold, crisp, confident black ink line art on pure white paper. Uniform medium-weight lines, "
    "not pale or faint. Capture the WHOLE image faithfully — do not assume one foreground subject; "
    "draw dominant shapes, secondary structure, and meaningful fine detail. "
    "Preserve real geometry and proportions. Keep meaningful fine detail as clean lines, for "
    "example: landscapes keep ridgelines and terrain contours; city scenes keep sparse facade and "
    "window structure; vehicles keep logos, text and panel lines; bikes keep spokes, cables and "
    "mechanical detail; portraits keep key facial contours, hair and clothing folds. "
    "A clean white paper background does NOT mean deleting real structure. Keep the actual scene "
    "context (terrain, buildings, streets, rails, trees, wires and other real structures) visible "
    "as simplified thin line art; do not omit it just because it is secondary or low-contrast. "
    "Surface/scene texture (gravel, grass, foliage, water, cloud edges) should be placed and "
    "meaningful where it really exists, not random sprinkles, and only as a few representative marks. "
    "Do not invent texture, clouds, or objects that are not in the source. No black shadow blobs. "
    "No grayscale fills. No filled-solid text or logos. No duplicated or ghost structures. Simplify "
    "glass/reflections — do not draw reflected scenery inside windows.")

# Each candidate leads with a STRONG distinct directive (placed first so Kontext
# weights it), then the Style Card as supporting context. Distinct guidance +
# seed offsets make the four genuinely diverge. recover = run the general
# tiled detail-recovery pass and safe-merge it in.
# General, subject-agnostic candidate roles. No assumption of one foreground
# subject; examples span domains so the model generalizes without hardcoding.
_NO_FILL = ("Draw all text, lettering and logos as clean OUTLINES — never fill letters, logos or "
            "dark regions solid black. Do not duplicate, ghost, or invent any object or structure "
            "that is not in the source photo.")
STUDIO_CANDIDATE_STYLES = {
    "balanced":   {"label": "Balanced reference-like", "guidance": 2.8, "seed_off": 0, "recover": False,
                   "lead": "Redraw this image as clean bold black line art matching a reference "
                           "coloring-book illustration: balanced and confident, capturing the whole "
                           "image — dominant shapes, secondary structure, and meaningful fine detail "
                           "— not only the most obvious element. " + _NO_FILL},
    "fullscene":  {"label": "Full-scene detail", "guidance": 3.4, "seed_off": 7, "recover": True,
                   "lead": "Redraw the ENTIRE image as detailed black line art with rich, complete "
                           "structure across the whole frame. Capture secondary and contextual "
                           "structure that is easy to miss — e.g. distant ridgelines/terrain "
                           "contours, building facades and windows, wires/cables/spokes, foliage and "
                           "ground structure, hair and clothing folds — wherever they really appear. "
                           "Draw only real structure from the photo; never leave large real regions "
                           "blank, and never invent or duplicate objects. " + _NO_FILL},
    "structure":  {"label": "Structure-faithful", "guidance": 2.8, "seed_off": 3, "recover": False,
                   "lead": "Redraw this image prioritizing correct major shapes and proportions and "
                           "preserving meaningful fine detail accurately: text, logos and markings, "
                           "panel/architectural lines, windows, mechanical parts, key contours. Render "
                           "these crisply; lesser background may be simpler. " + _NO_FILL},
    "clean":      {"label": "Clean plotter-source", "guidance": 2.3, "seed_off": 11, "recover": False,
                   "lead": "Redraw this image as simpler, cleaner bold line art: fewer fine textures, "
                           "confident continuous strokes, the whole image still present and readable "
                           "but uncluttered — easy for a pen plotter. " + _NO_FILL},
}

def _studio_style_card():
    try:
        return STUDIO_STYLECARD.read_text(encoding="utf-8")
    except Exception:
        return STUDIO_STYLE_CARD_DEFAULT

def _studio_slug(name):
    import re
    return re.sub(r"[^a-zA-Z0-9]+", "_", Path(name).stem)[:40] or uuid4().hex[:8]

_STUDIO_JOBS = {}

def _crisp_ink(png_path: Path):
    """Crisp Ink finish: the local candidates come out paler/softer than the
    ChatGPT targets. Normalize paper to white, sharpen edges, and stretch grays
    toward black so lines read bold — without creating filled black blobs."""
    import cv2, numpy as np
    g = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return
    hi = float(np.percentile(g, 92))                       # paper level → white
    if 120.0 <= hi < 252.0:
        g = np.clip(g.astype(np.float32) * (255.0 / hi), 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(g, (0, 0), 1.2)
    g = cv2.addWeighted(g, 1.6, blur, -0.6, 0)             # unsharp → crisp edges
    lo, hiC = 60.0, 205.0                                  # contrast stretch: lines→dark, paper→white
    g = np.clip((g.astype(np.float32) - lo) * (255.0 / (hiC - lo)), 0, 255).astype(np.uint8)
    cv2.imwrite(str(png_path), g)

def _studio_protect_mask(base_gray):
    """General 'strong existing linework' mask — wherever the base candidate
    already drew confident structure, dilated. Detail-recovery ink is never
    added here, so good linework is never overwritten or duplicated (no ghosts).
    Subject-agnostic: protects whatever the base drew, not a presumed subject."""
    import cv2, numpy as np
    strong = (base_gray < 110).astype(np.uint8)
    k = max(3, int(0.006 * max(base_gray.shape)) | 1)
    return cv2.dilate(strong, np.ones((k, k), np.uint8)) > 0

def _studio_detail_recovery(im, base_png: Path, prompt, seed, unet, cfgd, run_timeout, job):
    """General detail-recovery pass (subject-agnostic, safe-merge): faint and
    secondary structure gets lost in a single global pass. Re-generate the image
    in overlapping TILES at higher effective resolution (more pixels per region
    → faint detail like windows/spokes/terrain/hair is captured), then ADD that
    ink ONLY where the base candidate is near-white (genuine gaps) and OUTSIDE
    its strong existing linework. Preserves good lines; fills missing weak
    structure only; cannot duplicate objects or overwrite artwork."""
    import cv2, numpy as np, requests
    base = cv2.imread(str(base_png), cv2.IMREAD_GRAYSCALE)
    if base is None:
        return
    BH, BW = base.shape
    protect = _studio_protect_mask(base)
    rp = ("Redraw this image region as clean thin black line art, capturing ALL real structure "
          "including faint, fine and secondary detail. Bold confident thin lines, pure white "
          "background, no shading, no fills. Draw only what is really present — do not invent or "
          "duplicate objects. " + prompt)
    W0, H0 = im.size
    grid, ov = 2, 0.12                                  # 2x2 tiles, 12% overlap
    for gy in range(grid):
        for gx in range(grid):
            fx0 = max(0.0, gx / grid - ov); fx1 = min(1.0, (gx + 1) / grid + ov)
            fy0 = max(0.0, gy / grid - ov); fy1 = min(1.0, (gy + 1) / grid + ov)
            crop = im.crop((int(W0 * fx0), int(H0 * fy0), int(W0 * fx1), int(H0 * fy1)))
            ps = min(1.0, 1280.0 / max(crop.size))      # each tile gets generous resolution
            w = max(64, int(crop.width * ps) // 8 * 8)
            h = max(64, int(crop.height * ps) // 8 * 8)
            tmp = base_png.parent / "_tile_in.png"; crop.resize((w, h)).save(tmp)
            name = _comfy_upload(tmp)
            wf = _ai_kontext_workflow(name, rp, seed, unet, cfgd)
            info = _comfy_run_cancellable(wf, run_timeout, job)
            r = requests.get(f"{_comfy_url()}/view", params={"filename": info["filename"],
                             "subfolder": info.get("subfolder", ""), "type": info.get("type", "output")}, timeout=120)
            r.raise_for_status()
            tile = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
            if tile is None:
                continue
            x0, y0, x1, y1 = int(BW * fx0), int(BH * fy0), int(BW * fx1), int(BH * fy1)
            tile = cv2.resize(tile, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
            region = base[y0:y1, x0:x1]
            prot = protect[y0:y1, x0:x1]
            # add tile ink only into genuine gaps (base near-white) outside strong
            # existing lines — fills missing detail, never overwrites/duplicates.
            add = (tile < 150) & (region > 200) & (~prot)
            region[add] = tile[add]
            base[y0:y1, x0:x1] = region
    cv2.imwrite(str(base_png), base)

def _studio_worker(job):
    import requests
    from PIL import Image
    try:
        if not _comfy_available():
            raise RuntimeError(f"ComfyUI is not running at {_comfy_url()}.")
        unet = _comfy_kontext_unet()
        if not unet:
            raise RuntimeError("FLUX Kontext is required for Source Art Studio (not installed).")
        cfgd = _ai_cfg()
        sp = AI_CLEAN_SPEEDS[job["speed"]]
        cfgd["kontext_steps"] = sp["steps"]
        run_timeout = max(float(sp["timeout_s"]), 1200.0)
        style_card = _studio_style_card()
        seed = job["seed"]
        crisp = job.get("crisp", True)
        recover_on = job.get("recover", True)
        im = Image.open(job["src_path"]).convert("RGB")
        # Use the speed's native resolution (Fast 1024 / Balanced 1216 / Final
        # 1536). A 1216 floor on Fast crashes Kontext on a 16GB card (native
        # OOM at borderline commit headroom). Reliability first; Crisp Ink bolds
        # the lines regardless of gen size, and higher res is an opt-in via speed.
        gen_side = int(sp["max_side"])
        ps = min(1.0, gen_side / max(im.size))
        w = max(64, int(im.width * ps) // 8 * 8)
        h = max(64, int(im.height * ps) // 8 * 8)
        im_s = im.resize((w, h)) if ps < 1.0 else im
        cdir = STUDIO_DIR / job["case"]
        cdir.mkdir(parents=True, exist_ok=True)
        tmp = cdir / "_input.png"; im_s.save(tmp)
        name = _comfy_upload(tmp)
        job["results"] = []
        for st in job["styles"]:
            cs = STUDIO_CANDIDATE_STYLES[st]
            job["phase"] = f"generating {cs['label']}"
            # style-specific directive FIRST (Kontext weights it), then style card
            prompt = cs["lead"] + "\n\n" + style_card
            cfgd["kontext_guidance"] = cs["guidance"]      # distinct per style
            cseed = (seed + cs["seed_off"]) & 0xFFFFFFFF   # distinct seed → genuine divergence
            wf = _ai_kontext_workflow(name, prompt, cseed, unet, cfgd)
            info = _comfy_run_cancellable(wf, run_timeout, job)
            r = requests.get(f"{_comfy_url()}/view", params={"filename": info["filename"],
                             "subfolder": info.get("subfolder", ""), "type": info.get("type", "output")}, timeout=120)
            r.raise_for_status()
            out = cdir / f"cand_{st}.png"; out.write_bytes(r.content)
            # General detail-recovery pass (for styles that request it)
            if recover_on and cs.get("recover"):
                job["phase"] = f"detail recovery ({cs['label']})"
                try:
                    _studio_detail_recovery(im_s, out, style_card, cseed, unet, cfgd, run_timeout, job)
                except _AICancelled:
                    raise
                except Exception as e:
                    print(f"  [studio] detail recovery failed: {e}")
            if crisp:
                _crisp_ink(out)                            # darken + sharpen to match target boldness
            (cdir / f"cand_{st}.prompt.txt").write_text(prompt, encoding="utf-8")
            (cdir / f"cand_{st}.settings.json").write_text(json.dumps(
                {"style": st, "label": cs["label"], "seed": cseed, "speed": job["speed"],
                 "steps": sp["steps"], "guidance": cs["guidance"], "recover": bool(recover_on and cs.get("recover")),
                 "crisp": crisp, "source": job.get("source_name")}, indent=1))
            job["results"].append({"style": st, "label": cs["label"],
                                   "url": f"/work/studio/{job['case']}/cand_{st}.png?t={int(time.time())}",
                                   "file": f"studio/{job['case']}/cand_{st}.png"})

        # EXPERIMENTAL: reference-guided candidate — content = photo, style ref =
        # paired ChatGPT target, via Kontext native multi-reference (no extra
        # models). Lets the target actively guide style/detail, not just compare.
        if job.get("use_reference") and job.get("target_path"):
            try:
                job["phase"] = "generating Balanced + ChatGPT reference"
                # Reference is only a STYLE hint — keep it small (768px) so the
                # two-image multi-ref graph fits VRAM. Full-size both images
                # crashes Kontext on a 16GB card (native OOM). Free first.
                ref_im = Image.open(job["target_path"]).convert("RGB")
                rps = min(1.0, 768.0 / max(ref_im.size))
                rw = max(64, int(ref_im.width * rps) // 8 * 8)
                rh = max(64, int(ref_im.height * rps) // 8 * 8)
                rtmp = cdir / "_refstyle.png"; (ref_im.resize((rw, rh)) if rps < 1.0 else ref_im).save(rtmp)
                ref_name = _comfy_upload(rtmp)
                _comfy_free()                              # clear VRAM fragmentation before the heavier 2-image graph
                cfgd["kontext_guidance"] = 2.8
                refprompt = ("Redraw the FIRST image as clean bold black line art. Match the LINE-ART "
                             "STYLE of the SECOND reference image — its line confidence, weight, detail "
                             "level, background treatment and overall look — but keep the exact content, "
                             "composition and subject of the FIRST image. Do not copy objects or scenery "
                             "from the second image. " + _NO_FILL + "\n\n" + style_card)
                wf = _ai_kontext_workflow_multiref(name, ref_name, refprompt, seed, unet, cfgd)
                info = _comfy_run_cancellable(wf, run_timeout, job)
                r = requests.get(f"{_comfy_url()}/view", params={"filename": info["filename"],
                                 "subfolder": info.get("subfolder", ""), "type": info.get("type", "output")}, timeout=120)
                r.raise_for_status()
                out = cdir / "cand_balanced_ref.png"; out.write_bytes(r.content)
                if crisp:
                    _crisp_ink(out)
                (cdir / "cand_balanced_ref.prompt.txt").write_text(refprompt, encoding="utf-8")
                (cdir / "cand_balanced_ref.settings.json").write_text(json.dumps(
                    {"style": "balanced_ref", "label": "Balanced + ChatGPT reference", "seed": seed,
                     "speed": job["speed"], "guidance": 2.8, "reference": job.get("target_name"),
                     "method": "kontext-multiref-offset", "crisp": crisp, "source": job.get("source_name")}, indent=1))
                job["results"].append({"style": "balanced_ref", "label": "Balanced + ChatGPT reference",
                                       "url": f"/work/studio/{job['case']}/cand_balanced_ref.png?t={int(time.time())}",
                                       "file": f"studio/{job['case']}/cand_balanced_ref.png", "experimental": True})
            except _AICancelled:
                raise
            except Exception as e:
                msg = str(e)
                if "Connection" in msg or "refused" in msg or "10054" in msg or "10061" in msg:
                    msg = ("multi-reference crashed ComfyUI (native OOM — needs more VRAM than this "
                           "16GB GPU). Restart ComfyUI. The 4 normal candidates above are unaffected.")
                print(f"  [studio] reference-guided candidate failed: {e}")
                job["ref_error"] = msg[:240]

        job["phase"], job["status"] = "done", "done"
        print(f"  [studio] job {job['id']} done: {len(job['results'])} candidates -> {cdir}")
    except _AICancelled:
        job["status"] = "cancelled"
    except Exception as e:
        msg = str(e)
        if "Connection" in msg or "refused" in msg or "10054" in msg or "10061" in msg:
            msg = ("ComfyUI crashed mid-generation (native OOM — Kontext is near this 16GB GPU's "
                   "limit). Restart ComfyUI, close other apps for more RAM headroom, and prefer Fast "
                   "speed. Any candidates already shown above are saved.")
        job["status"], job["error"] = "error", msg
        print(f"  [studio] job {job['id']} failed: {e}")
    finally:
        _AI_LOCK.release()
