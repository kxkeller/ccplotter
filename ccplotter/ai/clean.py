import json
import threading
import time
from pathlib import Path
from uuid import uuid4

from ccplotter.config import WORK_DIR
from ccplotter.runs import _runs_add
from ccplotter.svg_io import _write_paths_svg
from ccplotter.mask import _stroke_px_for
from ccplotter.trace_v3 import _lineart_quality_check
from ccplotter.ai.comfy import (_ai_cfg, _comfy_url, _comfy_checkpoints, _comfy_controlnets,
                                 _comfy_has_node, _comfy_kontext_unet, _ai_kontext_workflow,
                                 _comfy_upload, _comfy_free, _ai_workflow, _comfy_run,
                                 _AICancelled, _comfy_run_cancellable)
from ccplotter.ai.openai_provider import _use_openai, _openai_lineart_bytes

# FLUX Kontext: instruction-based image EDITING model — the closest local
# analogue to GPT-image's "redraw this as line art" judgment. Official ComfyUI
# graph (from the installed flux_kontext_dev_basic template).
AI_KONTEXT_PROMPTS = {
    "lineart": "Convert this image into clean black and white line art: smooth single-weight black ink "
               "outlines on a pure white background, coloring book style. No shading, no gray tones, no "
               "fills. Simplify busy textures (gravel, foliage, brick) into a few suggestive strokes, but "
               "keep the main subject detailed and accurate, including any text or markings.",
    "sketch": "Redraw this image as a hand-drawn pen and ink sketch on white paper: confident black ink "
              "strokes with cross-hatched shading instead of solid fills or gray tones. Keep the "
              "composition and main subject accurate.",
}

_AI_LOCK = threading.Lock()

# ── AI Clean for Plotter (async job workflow) ────────────────────────────────
# AI cleanup presets REDRAW the source image into clean line art, saved as a
# NEW source. They are NOT processing modes: tracing stays deterministic
# (Plot-Ready Line Art), and the active source only changes when the user
# explicitly picks the cleaned result in the UI. If a job fails / times out /
# is cancelled, no new source is created and nothing is switched.

# ── Active provider prompts (OpenAI gpt-image-1) ─────────────────────────────
# One tuned prompt per STYLE (image-agnostic — same text for every image; the
# image is the edit input). Rules baked into the prompts:
# stay faithful / don't invent glass interiors or
# reflections (blank or glass-textured beats fabricated); effects (mist, smoke,
# clouds, water) as flowing OUTLINE strokes not shaded blobs; light texture on
# natural/material surfaces where appropriate; balanced detail across the frame.
# Extensible: add "sketch" / artistic styles here later.
AI_STYLE_PROMPTS = {
    "lineart": (
        "Turn the provided image into clean black-and-white line art optimized for a pen plotter.\n\n"
        "Preserve the original image's main subject, composition, camera angle, proportions, perspective, "
        "and recognizable identity as faithfully as possible. Convert it into a crisp technical-style "
        "drawing using one ink color and mostly uniform pen-width lines.\n\n"
        "Use lines only: no color, no grayscale fills, no solid black areas, no soft gradients, no "
        "painterly shading, no messy noise. It should read like detailed pen-and-ink / technical "
        "illustration line art, not a sketchy filter or a shaded drawing.\n\n"
        "CRITICAL — dark areas are OUTLINED, never filled. Any dark or black region — vehicle "
        "undersides and undercarriages, wheel wells, shadowed recesses, doorways and deep openings, "
        "dark or tinted windows, dark liveries and panels — must be drawn with its boundary contour "
        "plus a few interior structural lines, and left WHITE inside. Never place a solid black or gray "
        "shape anywhere. Cast shadows on the ground, track, or walls: omit them, or suggest them with at "
        "most one or two thin contour lines — never as a filled or hatched dark mass. If a region looks "
        "dark in the photo, OUTLINE it; do not fill it. This rule overrides any urge to add depth or "
        "realism with tone.\n\n"
        "Stay faithful to what is actually in the photo — do not invent or imagine detail that isn't "
        "there. For glass, windshields, and windows, do NOT draw made-up interiors or reflected scenery; "
        "leave the glass clean white or give it light glass-like texture (a few frame and highlight "
        "contour lines) — better blank than fabricated. Do not add or duplicate mechanical fittings the "
        "photo doesn't clearly show — wipers, mirrors, antennas, lights, handles: draw only the ones "
        "actually visible, and if a shadow or reflection makes it ambiguous, leave it out rather than "
        "inventing one (for example, do not add a second windshield wiper that isn't there).\n\n"
        "Text and signage: reproduce the ACTUAL lettering, registration numbers, and logos as accurately "
        "and legibly as you can wherever it is large and clear enough to read faithfully. If a piece of "
        "text is too small or unclear to reproduce correctly, draw the sign, plate, or panel outline "
        "cleanly and leave it blank — NEVER invent fake, scrambled, or placeholder letters. Garbled text "
        "is worse than no text.\n\n"
        "Emphasize: clean outer contours; important structural forms; edges, seams, panels, windows, "
        "joints, attachments, and surface details; readable text or logos only when they can stay clear; "
        "environmental contact points and ground lines.\n\n"
        "Render effects — mist, smoke, vapor, dust, water, reflections, clouds — as simple flowing "
        "outline strokes and wispy contour lines, never as shaded blobs, and ONLY where they actually "
        "appear in the photo. Match the real sky: if it is clear or has just a few wispy clouds, keep it "
        "mostly empty white with only those few wisps — do NOT invent, add, or multiply clouds, and never "
        "fill an empty sky with manufactured cloud detail. The same applies to any large plain area: do "
        "not fabricate detail to fill it. Add texture with controlled linework only: sparse hatching, "
        "contour lines, short directional strokes, occasional stipple-like marks, flowing outlines — "
        "applied to natural elements and surface materials where they genuinely exist, enough to describe "
        "form and material without becoming noisy or dense.\n\n"
        "Simplify or remove unnecessary background clutter, but keep context that grounds the image. "
        "Balance detail across the whole frame — never one area highly detailed and the rest empty.\n\n"
        "The final image: crisp, faithful, detailed, lightly textured, readable, a single ink color, "
        "suitable for a single-pen plotter with minimal clutter — and ABSOLUTELY no solid black fills, "
        "no gray shading, and no cast-shadow masses: every dark region is outlined, white inside."
    ),
}

AI_CLEAN_PRESETS = {
    "sparse": {
        "label": "Sparse Plotter-Safe",
        # Instruction prompt for FLUX Kontext (image-editing model).
        "kontext": (
            "Convert this exact source image into sparse black-and-white pen-plotter line art. "
            "Identify the main subject (vehicle, aircraft, building, person) and preserve its real "
            "geometry, proportions, and model-specific details accurately. Preserve the perspective "
            "and major contours of the original image. Do not invent new objects or parts. "
            "Keep every major scene element: buildings and their window patterns as simplified "
            "outlines, tracks, railings, poles, wires, and people. Do not leave large parts of the "
            "scene blank. "
            "Always draw the main subject's functional details: windshield wipers, lights and light "
            "bars, mirrors, window frames with their major glass reflection lines, signage and visible "
            "text. "
            "Use clear visual hierarchy: the main subject gets the most detail; supporting structures "
            "(rails, roads, railings, overpasses, nearby buildings) get medium detail with their "
            "perspective and major repeated features; the distant background gets sparse but COMPLETE "
            "treatment — skyline outlines, simplified building faces, a few representative windows, "
            "major ridgelines. Never abandon the background halfway: apply the same consistent sparse "
            "line treatment across the entire scene. Simplify repetitive background detail into "
            "representative patterns instead of drawing every instance. "
            "Do not add texture marks: leave soft tonal fields — sky, clouds, gravel, grass, water, "
            "smooth surfaces — as clean empty white. Stylized texture is added separately later. "
            "Clean black lines on a pure white background. No grayscale shading. No filled black "
            "areas — dark surfaces drawn as outlines only, white inside. "
            "Use fewer, longer, continuous medium-weight strokes suitable for a pen plotter and "
            "downstream centerline tracing."),
        # Tag-style prompts for the SD checkpoint (img2img/ControlNet) fallback.
        "pos": ("sparse black and white pen plotter line art, clean black ink lines on pure white "
                "background, fewer longer continuous strokes, simplified background, monochrome outline "
                "drawing, medium line weight, minimal texture, no shading"),
        "neg": ("gray shading, hatching, crosshatching, stippling, pencil shading, charcoal, sketch "
                "texture, filled black regions, solid shadows, dense texture, noisy lines, extra wheels, "
                "extra objects, distorted geometry, invented parts, hallucinated details, blurry lines, "
                "broken strokes, repeated clutter, coloring book shading, photo, photorealistic, color, "
                "random scattered marks, dash noise, sprinkles, noisy sky, contour bands, "
                "abandoned half-drawn background, inconsistent background detail"),
    },
    "detailed": {
        # Permissive preset: no texture/density
        # restrictions — draw EVERYTHING, let the deterministic cleanup
        # (fill-hollowing) and the trace pipeline (noise cull, min length,
        # routing) manage what the pen actually plots.
        "label": "Detailed Line Art",
        "kontext": (
            "Redraw this exact source image as detailed black-and-white pen line art. "
            "Identify the main subject (vehicle, aircraft, building, person) and preserve its real "
            "geometry, proportions, and model-specific details accurately. Preserve the perspective "
            "and contours of the original image. Do not invent new objects or parts. "
            "Draw every visible feature: the main subject with all of its details — panel lines, "
            "windshield wipers, lights and light bars, mirrors, window frames, emblems, logos, "
            "signage and text — plus the background buildings and their windows, railings, people, "
            "track hardware, and ground texture such as gravel and ballast. "
            "Use clear visual hierarchy: the main subject most detailed, supporting structures medium, "
            "distant background sparser but COMPLETE — never abandon the background halfway; keep a "
            "consistent line treatment across the whole scene. "
            "Prioritize the front face of the main vehicle: windshield frame, headsign/destination box "
            "and its text, lights, route number, logo, front panel geometry, wipers, and door and "
            "window outlines. Keep this area clean, accurate, and uncluttered. "
            "Simplify glass and reflective surfaces: do NOT draw scenery, wires, or buildings reflected "
            "inside windshields and windows as line marks. Keep only the glass outline, frame, wipers, "
            "and at most one or two major highlight contours — glass interiors stay mostly empty white. "
            "Do not turn shadows into black blobs or dense broken texture: represent a shadow with a "
            "few clean contour boundaries, or omit it. Texture marks must correspond to real SURFACE "
            "texture (gravel, ballast, grass, foliage), never to reflections, shadows, or glass glare. "
            "Render real scene texture as STRUCTURED stylized marks confined to where it belongs: "
            "gravel and ballast as clusters of small irregular stone outlines inside the track bed or "
            "ground only; grass as short directional tufts on the ground only; clouds as a few clean "
            "flowing contour boundaries. Never scatter random small marks — every texture mark must "
            "correspond to actual content at that exact spot. Keep windshields, glass, vehicle body "
            "panels, signs, and smooth surfaces completely clean. "
            "Use clean black ink lines on a pure white background. No grayscale shading. "
            "No solid filled black areas — draw dark surfaces as outlines."),
        "pos": ("highly detailed black and white pen ink line art, clean black ink lines on pure white "
                "background, accurate detailed subject, full scene detail, monochrome drawing"),
        "neg": ("gray shading, filled black regions, solid shadows, extra objects, distorted geometry, "
                "invented parts, hallucinated details, blurry lines, photo, photorealistic, color, "
                "random scattered marks, dash noise, sprinkles, noisy sky, contour bands, "
                "abandoned half-drawn background, inconsistent background detail, "
                "reflections inside glass, reflected scenery in windows, cluttered windshield, "
                "dense marks inside windows, shadow blobs, broken shadow texture, mangled face"),
    },
}
# Background treatment dial — appended to the preset prompt. Splits the two
# outcomes the model otherwise picks at random (one side detailed / one side
# blank, mountains deleted) into an explicit user choice.
AI_CLEAN_BACKGROUND = {
    "sparse": {
        "label": "Sparse background",
        "kontext": ("Background treatment: draw the distant background as sparse simple structure — "
                    "skyline silhouettes plus a FEW representative windows and floor lines per building "
                    "(sparse, but never just empty perimeter outlines). Keep this sparse treatment EVEN "
                    "across the whole image: same density left and right, never one side detailed and "
                    "the other empty, never blank. Distant landscape is mandatory: always draw mountain "
                    "and hill silhouettes and far structure strips as single clean contour lines."),
        "pos": "sparse simplified background, skyline outlines, even background density"},
    "balanced": {
        "label": "Balanced background",
        "kontext": ("Background treatment: draw the distant background with moderate, consistent detail. "
                    "Buildings must keep readable facade structure — window grids, floor lines, and major "
                    "architectural divisions — never collapse a building to just its perimeter outline. "
                    "Keep the density EVEN across the whole image: same treatment left and right, never "
                    "one side detailed and the other empty. Distant landscape is mandatory: always draw "
                    "mountains, hills, ridgelines, and far structure strips as clean simplified contours — "
                    "never omit them."),
        "pos": "moderately detailed consistent background, even background density"},
    "detailed": {
        "label": "Detailed background",
        "kontext": ("Background treatment: draw the full background with consistent detail — every building "
                    "face with its complete window grid and architectural divisions, all structures. Keep "
                    "the density EVEN across the whole image; never abandon any region. Distant landscape "
                    "is mandatory: always draw mountains, hills, ridgelines, and far structure strips — "
                    "never omit them."),
        "pos": "fully detailed background, dense consistent background, even background density"},
}

AI_CLEAN_FIDELITY = {
    # Lower denoise = more faithful, less hallucination. Per-engine analogues:
    # kontext has no denoise, so fidelity maps to guidance; the ControlNet path
    # maps to control strength (higher strength = more faithful).
    "high":     {"label": "High Fidelity",  "denoise": 0.25, "kontext_guidance": 2.0, "cn_strength": 0.95, "cn_end": 0.95},
    "balanced": {"label": "Balanced",       "denoise": 0.35, "kontext_guidance": 2.5, "cn_strength": 0.80, "cn_end": 0.85},
    "strong":   {"label": "Strong Cleanup", "denoise": 0.45, "kontext_guidance": 3.0, "cn_strength": 0.65, "cn_end": 0.75},
}
AI_CLEAN_SPEEDS = {
    "fast":     {"label": "Fast Preview",  "max_side": 1024, "steps": 8,  "timeout_s": 300},
    "balanced": {"label": "Balanced",      "max_side": 1216, "steps": 14, "timeout_s": 600},
    "final":    {"label": "Final Quality", "max_side": 1536, "steps": 20, "timeout_s": 1200},
}

_AI_JOBS: dict = {}   # job_id -> mutable job dict; single job at a time via _AI_LOCK
# _AICancelled / _comfy_run_cancellable / _comfy_run_cancellable_multi are
# defined in ccplotter/ai/comfy.py and imported into this module's namespace
# at the top of this file.

def _ai_postclean(raw_path: Path, cleaned_path: Path, mask_path: Path):
    """Deterministic cleanup of the RAW AI OUTPUT (never the original photo):
    grayscale → normalize background to white → threshold → speck removal →
    pure B/W PNG. Returns (quality_raw, quality_cleaned, filled_pct)."""
    import cv2, numpy as np
    g = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError("could not read the AI output image")
    # Normalize background toward white (AI outputs are often slightly off-white).
    hi = float(np.percentile(g, 90))
    if 80.0 <= hi < 250.0:
        g = np.clip(g.astype(np.float32) * (255.0 / hi), 0, 255).astype(np.uint8)
    q_raw = _lineart_quality_check(g, max(g.shape))
    # Upscale the antialiased grayscale BEFORE binarizing: thresholding at AI
    # native res (~1MP) makes hard pixel staircases that look rough zoomed in
    # and feed jaggy masks to the centerline tracer. 2x cubic on the smooth
    # grayscale halves the staircase size for free.
    if max(g.shape) < 1600:
        g = cv2.resize(g, (g.shape[1] * 2, g.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    if q_raw["gray_pct"] < 8.0:
        # Clean line art: fixed 200 cutoff keeps the full antialiased body of
        # each stroke. Otsu lands mid-ramp on soft gray AI lines and
        # thins/breaks them (measured: it halved the ink of a good render).
        thr = 200.0
    else:
        # Notable gray wash: a fixed 200 cutoff converts the wash into solid
        # ink fields. Otsu separates wash from line cores; a small margin
        # above it still keeps the antialiased stroke bodies.
        otsu_t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thr = min(200.0, float(otsu_t) + 25.0)
    _, mask = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY_INV)
    # Remove only true dust — the trace pipeline has its own Detail/Noise cull,
    # so postclean must not pre-empt small legitimate marks (windows, ticks).
    min_area = max(4, int(round((max(g.shape) / 1024.0) ** 2 * 6)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    mb = keep[labels]
    # Filled-region detection: stroke pixels farther from the background than a
    # plausible pen-stroke radius are fills/shadows, not lines.
    dt = cv2.distanceTransform(mb.astype(np.uint8), cv2.DIST_L2, 3)
    fill_limit = max(5.0, max(g.shape) / 250.0)
    filled_pct = float((dt > fill_limit).mean()) * 100.0
    if filled_pct > 0.05:
        # Enforce "no filled black regions" even when the AI disobeys: erase
        # blob interiors so solid fills become drawable outline bands. Normal
        # strokes are thinner than the band radius and pass untouched.
        mb = mb & (dt <= fill_limit * 0.5)
    cleaned = np.where(mb, 0, 255).astype(np.uint8)
    cv2.imwrite(str(cleaned_path), cleaned)
    cv2.imwrite(str(mask_path), (mb.astype(np.uint8) * 255))
    q_clean = _lineart_quality_check(cleaned, max(cleaned.shape))
    return q_raw, q_clean, filled_pct

def _ai_lineart_cleanup(raw_path: Path, cleaned_path: Path, raw_annotator_path: Path,
                        job: dict = None, timeout: float = 300.0):
    """Alternative cleanup: convert the shaded raw AI output to clean line art
    with the LineArt annotator (Informative-Drawings) instead of thresholding.
    The annotator extracts single-weight CONTOUR lines, so shadows become
    boundary lines rather than black blobs, faint background lines survive, and
    drawn texture keeps its shapes. Returns (quality, raw_annotator_url) or None
    if the annotator node isn't installed."""
    import cv2, numpy as np, requests
    from PIL import Image
    if not _comfy_has_node("LineArtPreprocessor"):
        return None
    # Free the resident FLUX/Kontext UNET first — the annotator is tiny but OOMs
    # on the 16GB card if the 11GB generation model is still loaded.
    _comfy_free()
    im = Image.open(raw_path).convert("RGB")
    # Cap at 1024: the annotator's memory scales with resolution and 1536 tried
    # to allocate ~28GB. 1024 then 2x upscale in cleanup = plenty for tracing.
    ps = min(1.0, 1024.0 / max(im.size))
    w = max(64, int(im.width * ps) // 8 * 8)
    h = max(64, int(im.height * ps) // 8 * 8)
    name = _comfy_upload(raw_path)
    wf = {
        "2": {"class_type": "LoadImage", "inputs": {"image": name}},
        "3": {"class_type": "ImageScale", "inputs": {"image": ["2", 0], "upscale_method": "lanczos",
                                                     "width": w, "height": h, "crop": "disabled"}},
        "10": {"class_type": "LineArtPreprocessor", "inputs": {"image": ["3", 0],
                                                              "coarse": "disable", "resolution": max(w, h)}},
        "11": {"class_type": "ImageInvert", "inputs": {"image": ["10", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "ccplotter_la"}},
    }
    if job is not None:
        info = _comfy_run_cancellable(wf, timeout, job)
    else:
        info = _comfy_run(wf, timeout)
    r = requests.get(f"{_comfy_url()}/view", params={"filename": info["filename"],
                     "subfolder": info.get("subfolder", ""), "type": info.get("type", "output")}, timeout=120)
    r.raise_for_status()
    raw_annotator_path.write_bytes(r.content)
    # Annotator output is already line art (thin gray lines, white bg, NO fills),
    # so cleanup is light: stretch faint lines, threshold, drop dust. No
    # fill-hollowing — there are no fills to hollow.
    g = cv2.imread(str(raw_annotator_path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    if max(g.shape) < 1600:
        g = cv2.resize(g, (g.shape[1] * 2, g.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    _, mask = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY_INV)
    min_area = max(4, int(round((max(g.shape) / 1024.0) ** 2 * 6)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    cleaned = np.where(keep[labels], 0, 255).astype(np.uint8)
    cv2.imwrite(str(cleaned_path), cleaned)
    return _lineart_quality_check(cleaned, max(cleaned.shape))

def _texture_backfill(orig_path: Path, cleaned_path: Path, strength: str = "light", texture_svg: Path = None, fill_mm: float = 0.15):
    """Deterministic background texture synthesis (gravel, grass, clouds).

    Kontext is an EDIT model: it traces edges faithfully but refuses to invent
    strokes for edgeless tonal fields, no matter what the prompt authorizes —
    sky/clouds, grass, gravel, foliage come back as blank paper. So we synthesize those strokes
    from the ORIGINAL photo instead of begging the model:
      - smooth tonal regions (sky, water, gradients) → iso-brightness contour
        lines; clouds emerge as flowing closed contours,
      - textured regions (grass, gravel, foliage) → short scattered strokes
        following the local image grain, density ∝ texture × darkness.
    Texture is a SEPARATE VECTOR LAYER: marks are written as polylines to
    `texture_svg` (in cleaned-image pixel coords), never baked into the
    cleaned PNG. The trace stage merges the layer in (toggleable, counted
    separately) — pebbles stay crisp vectors instead of surviving a skeleton
    round-trip, and the outline art remains pure.
    Stroke budgets are plotter-sane (each texture stroke = 2 Z-moves).
    Returns a stats dict, or None when nothing qualified."""
    import cv2, numpy as np
    # bands LOW on purpose: sky contours must be sparse large cloud boundaries,
    # not "Jupiter bands". facade = subtle texture in small enclosed pockets
    # (distant building faces) — full only.
    # sky=False: no cloud contours at all — on plots they read as phantom
    # mountain ridgelines. Light = ground texture only.
    # max_loops = scribble density budget (cheap: a whole region is ~1 stroke).
    P = {"light": {"bands": 2, "density": 0.75, "max_strokes": 300, "max_loops": 900,
                   "min_contour": 90, "facade": False, "sky": False},
         "full":  {"bands": 3, "density": 1.1,  "max_strokes": 650, "max_loops": 1800,
                   "min_contour": 60, "facade": True, "sky": True}}.get(strength)
    if P is None:
        return None
    clean = cv2.imread(str(cleaned_path), cv2.IMREAD_GRAYSCALE)
    orig = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
    if clean is None or orig is None:
        return None
    H, W = clean.shape
    g = cv2.cvtColor(cv2.resize(orig, (W, H), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY).astype(np.float32)
    sc = max(W, H) / 1200.0                       # px scale: tunables are @1200px

    # Open canvas = far enough from any AI ink that added strokes can't mud it,
    # AND not inside a densely-drawn neighborhood (windshields, wall panels —
    # blank pockets surrounded by subject linework are intentional negative
    # space, not missing background).
    ink = clean < 128
    dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 3)
    dens = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 28.0 * sc)
    zone = (dist > 8.0 * sc).astype(np.uint8)
    # ENCLOSURE TEST: blank pockets fully surrounded by linework (windshields,
    # tram panels, building faces, signage) are intentional negative space —
    # never texture them. Only blank zones that reach the image border (sky,
    # ground, track bed) or are very large qualify.
    nz, labz, stz, _ = cv2.connectedComponentsWithStats(zone, 8)
    border_ids = set(np.unique(np.concatenate([labz[0], labz[-1], labz[:, 0], labz[:, -1]])))
    bottom_ids = set(np.unique(labz[-1]))
    top_ids = set(np.unique(labz[0]))
    keepz = np.zeros(nz, bool)
    groundz = np.zeros(nz, bool)
    for zi in range(1, nz):
        on_border = zi in border_ids
        area = stz[zi, cv2.CC_STAT_AREA]
        keepz[zi] = on_border or (area > 0.10 * W * H)
        # GROUND GATE: gravel/grass texture is only physically plausible on
        # the ground plane — blank zones connected to the image BOTTOM and
        # not to the top. Sky (touches top), building sides and any enclosed
        # pocket can never receive ground texture. This prevents
        # gravel on the sides of buildings and sprinkles in the sky.
        groundz[zi] = (zi in bottom_ids) and (zi not in top_ids)
    open_area = keepz[labz] & (dens < 0.06)
    ground_area = groundz[labz] & (dens < 0.06)

    # Texture vs smooth split from the original's local contrast.
    k = max(3, int(9 * sc) | 1)
    mu = cv2.boxFilter(g, -1, (k, k))
    std = np.sqrt(np.clip(cv2.boxFilter(g * g, -1, (k, k)) - mu * mu, 0, None))
    # Texture must be a COHERENT field, not isolated speckle: morphological
    # opening culls stray cells (window glints, cloud edges, distant façades)
    # that otherwise sprinkle strokes over highrises and sky.
    ko = max(3, int(13 * sc) | 1)
    tex_m = ((std > 16.0).astype(np.uint8))
    tex_m = cv2.morphologyEx(tex_m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
    textured = ground_area & (tex_m > 0)           # texture lives ONLY on the ground plane
    smoothr = open_area & (std <= 13.0)

    rng = np.random.default_rng(1234)
    tex_polys = []                                 # vector layer output
    n_contours = n_strokes = n_marks_total = 0

    # ── A. iso-brightness contours in smooth regions (clouds / sky / water) ──
    gb = cv2.GaussianBlur(g, (0, 0), 5.0 * sc)
    # Gradient gate: an iso-line only counts where the smoothed image actually
    # CHANGES (a cloud edge, a real tonal boundary). Without this, band levels
    # crossing near-flat sky trace wandering contours everywhere — the
    # "flying through Jupiter" failure.
    gbx = cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=3)
    gby = cv2.Sobel(gb, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gbx * gbx + gby * gby)
    vals = gb[smoothr]
    if not P.get("sky", True):
        vals = np.array([])                        # sky contours disabled for this strength
    if vals.size > 2000:
        lo, hi = float(np.percentile(vals, 4)), float(np.percentile(vals, 96))
        if hi - lo > 12.0:                         # there IS a gradient worth drawing
            for t in np.linspace(lo, hi, P["bands"] + 2)[1:-1]:
                m = ((gb < t) & smoothr).astype(np.uint8)
                cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                for c in cs:
                    pc = c[:, 0, :]                # (x, y)
                    if len(pc) < P["min_contour"] * sc:
                        continue
                    # A contour of (gb<t)&smoothr includes the region BORDER
                    # (hugging the AI's ink) — keep only true iso-line points
                    # that sit on a REAL tonal edge.
                    iso = (np.abs(gb[pc[:, 1], pc[:, 0]] - t) < 2.5) & \
                          (gmag[pc[:, 1], pc[:, 0]] > 0.35)
                    runs, cur = [], []
                    for i, ok in enumerate(iso):
                        if ok: cur.append(pc[i])
                        elif cur: runs.append(cur); cur = []
                    if cur: runs.append(cur)
                    for run in runs:
                        if len(run) < P["min_contour"] * sc:
                            continue
                        tex_polys.append([(float(px), float(py)) for px, py in run[::3]])
                        n_contours += 1

    # ── B. short grain-following strokes in textured regions (gravel/grass) ──
    if textured.any():
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        s3 = 3.0 * sc
        Jxy = cv2.GaussianBlur(gx * gy, (0, 0), s3)
        Jdd = cv2.GaussianBlur(gx * gx - gy * gy, (0, 0), s3)
        Jss = cv2.GaussianBlur(gx * gx + gy * gy, (0, 0), s3)
        theta = 0.5 * np.arctan2(2 * Jxy, Jdd)
        # coherence: ~1 = strongly directional texture (grass blades, wood
        # grain) → strokes follow it; ~0 = isotropic (gravel, ballast) →
        # random orientations read as scattered stones, not combed sprinkles.
        coh = np.sqrt(Jdd ** 2 + 4 * Jxy ** 2) / (Jss + 1e-6)
        # GRAVEL = CONTINUOUS LOOPED SCRIBBLE.
        # Discrete pebble outlines at plotter-safe density read as scattered
        # blobs. Hand illustrators draw ballast as serpentine rows of small
        # CONNECTED loops — dense tone, irregular stone shapes, and the pen
        # never lifts: a whole gravel region costs ~2 Z-moves, so density is
        # finally affordable. Grass keeps directional dashes.
        step = max(6, int(10 * sc))
        ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
        py_, px_ = ys.ravel(), xs.ravel()
        ok = textured[py_, px_]
        py_, px_ = py_[ok], px_[ok]
        if py_.size:
            dn = (255.0 - g[py_, px_]) / 255.0
            bright_ok = g[py_, px_] <= 215
            coh_pt = coh[py_, px_]
            ncc, labcc = cv2.connectedComponents(textured.astype(np.uint8), 8)
            row_h = step * 1.7

            def _serpentine_chain(marks_by_region, mark_fn):
                """Connect a region's marks into continuous pen-down paths in
                serpentine row order — Z cost ≈ 2 per region, not per mark."""
                nonlocal n_strokes, n_marks_total
                for gpts in marks_by_region.values():
                    if not gpts:
                        continue
                    gpts.sort(key=lambda p: (int(p[1] // row_h),
                                             p[0] if int(p[1] // row_h) % 2 == 0 else -p[0]))
                    path = []
                    for (x, y) in gpts:
                        path.extend(mark_fn(x, y))
                        n_marks_total += 1
                    for s0 in range(0, len(path), 1400):
                        seg = path[s0:s0 + 1400]
                        if len(seg) >= 6:
                            tex_polys.append(seg)
                            n_strokes += 1

            # — grass / directional grain: bent dashes, CHAINED per region —
            d_idx = np.where(bright_ok & (coh_pt > 0.35))[0]
            d_keep = d_idx[rng.random(d_idx.size) <
                           np.clip((0.3 + 0.5 * dn[d_idx]) * P["density"] * 0.6, 0, 0.9)]
            d_groups = {}
            for i in d_keep[:P["max_loops"]]:
                d_groups.setdefault(int(labcc[py_[i], px_[i]]), []).append((float(px_[i]), float(py_[i])))

            def _grass_mark(x, y):
                th = float(theta[int(y), int(x)]) + np.pi / 2 + rng.normal(0, 0.18)
                L = (8.0 + 8.0 * rng.random()) * sc
                dxs, dys = np.cos(th) * L / 2, np.sin(th) * L / 2
                bend = L * rng.uniform(0.10, 0.25) * rng.choice((-1, 1))
                return [(x - dxs, y - dys), (x - np.sin(th) * bend, y + np.cos(th) * bend),
                        (x + dxs, y + dys)]
            _serpentine_chain(d_groups, _grass_mark)

            # — gravel / isotropic: dense varied stone marks, chained —
            s_idx = np.where(bright_ok & (coh_pt <= 0.35))[0]
            s_keep = s_idx[rng.random(s_idx.size) <
                           np.clip((0.6 + 0.4 * dn[s_idx]) * P["density"], 0, 0.97)]
            if s_keep.size > P["max_loops"]:
                s_keep = rng.choice(s_keep, P["max_loops"], replace=False)
            s_groups = {}
            for i in s_keep:
                s_groups.setdefault(int(labcc[py_[i], px_[i]]), []).append((float(px_[i]), float(py_[i])))

            def _stone_mark(x, y):
                # mix of round loops and angular stone outlines = varied
                # clustered rubble, not a uniform chain of "o"s
                r = (3.5 + 5.0 * rng.random()) * sc
                if rng.random() < 0.6:
                    sq, a0 = rng.uniform(0.55, 0.9), rng.uniform(0, 2 * np.pi)
                    return [(x + np.cos(a) * r, y + np.sin(a) * r * sq)
                            for a in np.linspace(a0, a0 + 2 * np.pi * 1.12, 7)]
                a0 = rng.uniform(0, 2 * np.pi)
                pts = []
                for k in range(5):
                    a = a0 + 2 * np.pi * k / 4 + rng.normal(0, 0.3)
                    rr = r * rng.uniform(0.5, 1.0)
                    pts.append((x + np.cos(a) * rr, y + np.sin(a) * rr * 0.8))
                return pts
            _serpentine_chain(s_groups, _stone_mark)

    # ── C. landscape / ridgeline silhouette (deterministic "mountains") ──
    # The AI regularly forgets distant landscape (mountains, ridgelines, far
    # strips). Extract the sky-ground boundary from the PHOTO and draw it
    # wherever the AI left the area blank — distant context is never dropped.
    try:
        skyish = (std < 14.0) & (g > 120.0)
        has_ground = (~skyish).any(axis=0)
        first_not = np.argmax(~skyish, axis=0)
        ridge = np.where(has_ground, first_not, -1).astype(np.float32)
        valid = (ridge > 3) & (ridge < 0.85 * H)
        if valid.sum() > 60 * sc:
            import scipy.ndimage as ndi
            rs = ndi.median_filter(ridge, size=max(5, int(15 * sc) | 1), mode='nearest')
            run = []
            for x in range(0, W, 2):
                yv = float(rs[x])
                yi = int(min(max(yv, 0), H - 1))
                # draw only where the AI left it blank (dist = px to AI ink)
                if valid[x] and dist[yi, x] > 8.0 * sc:
                    run.append((float(x), yv))
                else:
                    if len(run) > 50 * sc:
                        tex_polys.append(run); n_contours += 1
                    run = []
            if len(run) > 50 * sc:
                tex_polys.append(run); n_contours += 1
    except Exception as e:
        print(f"  [texture] ridgeline extraction failed: {e}")

    if n_contours + n_strokes == 0 or texture_svg is None:
        return None
    _write_paths_svg(texture_svg, W, H, _stroke_px_for({"fill_mm": fill_mm}, W), polylines=tex_polys)
    return {"strength": strength, "contour_lines": n_contours,
            "texture_strokes": n_strokes,          # pen strokes (Z cost = 2x this)
            "texture_marks": n_marks_total,        # visual marks (pebbles chain into one stroke)
            "texture_svg": texture_svg.name}

def _ai_verdict(q_raw: dict, q_clean: dict, filled_pct: float):
    """Plotter-safe verdict for the cleaned output: 'likely' / 'questionable' /
    'bad' plus human-readable reasons. 'bad' notably catches the
    thresholded-photo failure mode (huge coverage / large fills)."""
    cov, comps, gray_raw = q_clean["foreground_pct"], q_clean["components"], q_raw["gray_pct"]
    reasons = []
    if cov >= 40: reasons.append(f"very high ink coverage ({cov:.0f}%) — looks filled, not line art")
    if filled_pct >= 4: reasons.append(f"AI drew large filled black regions ({filled_pct:.1f}% of image) — hollowed to outlines in the cleaned source; consider re-generating")
    if gray_raw >= 30 and cov >= 25:
        reasons.append("raw AI output still looks photographic — this is a thresholded photo, not line art")
    if reasons: return "bad", reasons
    if cov >= 25: reasons.append(f"high ink coverage ({cov:.0f}%)")
    if filled_pct >= 1.5: reasons.append(f"AI drew some filled regions ({filled_pct:.1f}% of image) — hollowed to outlines in the cleaned source")
    if gray_raw >= 20: reasons.append(f"raw AI output has notable shading ({gray_raw:.0f}%)")
    if comps > 3000: reasons.append(f"very many separate strokes ({comps}) — dense or noisy")
    if reasons: return "questionable", reasons
    return "likely", ["looks like clean plotter-safe line art"]

def _ai_clean_worker(job: dict):
    """Background thread: run one AI Clean for Plotter job end to end. On any
    failure/cancel: no new source is written, the original is untouched."""
    import requests
    try:
        cfgd = _ai_cfg()
        preset = AI_CLEAN_PRESETS[job["preset"]]
        fid = AI_CLEAN_FIDELITY[job["fidelity"]]
        sp = AI_CLEAN_SPEEDS[job["speed"]]
        bgd = AI_CLEAN_BACKGROUND[job.get("background_detail", "balanced")]
        kontext_prompt = preset["kontext"] + " " + bgd["kontext"]
        pos_prompt = preset["pos"] + ", " + bgd["pos"]
        src = Path(job["src_path"])
        seed = job["seed"]

        job["phase"] = "preparing input"
        from PIL import Image
        im = Image.open(src).convert("RGB")

        # Engine: OpenAI (cloud, no local GPU) when configured + keyed; else local
        # ComfyUI — Kontext (edit model) > checkpoint+ControlNet > annotator.
        use_openai = _use_openai()
        has_pre = False if use_openai else _comfy_has_node("LineArtPreprocessor")
        kontext_unet = "" if use_openai else _comfy_kontext_unet()
        ckpt, cn = "", None
        if use_openai:
            cfgd["openai_model"] = job.get("openai_model") or cfgd.get("openai_model", "chatgpt-image-latest")
            cfgd["openai_quality"] = job.get("openai_quality") or cfgd.get("openai_quality", "medium")
            engine = "openai " + cfgd["openai_model"]
            run_timeout = 240.0
        elif kontext_unet:
            engine = "kontext"
            cfgd["kontext_steps"] = sp["steps"]
            cfgd["kontext_guidance"] = fid["kontext_guidance"]
            run_timeout = max(float(sp["timeout_s"]), 1200.0)   # first run loads ~17GB of weights
        else:
            cks = _comfy_checkpoints()
            ckpt = cfgd.get("checkpoint") or (cks[0] if cks else "")
            if ckpt:
                cn = cfgd.get("controlnet")
                if cn is None or cn == "":
                    cns = _comfy_controlnets()
                    cn = (next((c for c in cns if "lineart" in c.lower()), None) if has_pre else None) \
                         or next((c for c in cns if "canny" in c.lower()), None)
                elif cn == "off":
                    cn = None
                cfgd["steps"] = sp["steps"]
                cfgd["denoise"]["plotter_safe"] = fid["denoise"]
                cfgd["controlnet_strength"] = fid["cn_strength"]
                cfgd["controlnet_end"] = fid["cn_end"]
                engine = f"diffusion ckpt={ckpt} cn={cn or 'off'}"
                run_timeout = float(sp["timeout_s"])
            elif has_pre:
                engine = "annotator (deterministic — this backend ignores prompt/fidelity)"
                cfgd["lineart_variant"] = "coarse"   # simplified strokes ≈ sparse
                run_timeout = float(sp["timeout_s"])
            else:
                raise RuntimeError("No usable AI engine in ComfyUI: install FLUX Kontext, a checkpoint, "
                                   "or the LineArt preprocessor.")
        job["engine"] = engine

        def _gen(pim, tag):
            """One AI pass over a PIL image with the job's engine/preset.
            Returns the PNG bytes of the generated image."""
            if use_openai:
                job["phase"] = f"OpenAI line art ({tag})"
                _oap = AI_STYLE_PROMPTS.get(job.get("style", "lineart"), AI_STYLE_PROMPTS["lineart"])
                return _openai_lineart_bytes(pim, _oap, cfgd)
            ps = min(1.0, sp["max_side"] / max(pim.size))
            w = max(64, int(pim.width * ps) // 8 * 8)
            h = max(64, int(pim.height * ps) // 8 * 8)
            tmp = WORK_DIR / f"ai_in_{job['id']}_{tag.replace(' ', '_')}.png"
            pim.save(tmp)
            job["phase"] = f"uploading to ComfyUI ({tag})"
            try: image_name = _comfy_upload(tmp)
            finally: tmp.unlink(missing_ok=True)
            if engine == "kontext":
                wf = _ai_kontext_workflow(image_name, kontext_prompt, seed, kontext_unet, cfgd)
            elif ckpt:
                wf = _ai_workflow(image_name, "plotter_safe", ckpt, w, h, seed, cfgd,
                                  controlnet=cn, has_lineart_pre=has_pre,
                                  prompts={"pos": pos_prompt, "neg": preset["neg"]})
            else:
                wf = _ai_workflow(image_name, "lineart", "", w, h, seed, cfgd, has_lineart_pre=True)
            job["phase"] = f"Comfy generation ({tag})"
            iminfo = _comfy_run_cancellable(wf, run_timeout, job)
            job["phase"] = f"downloading result ({tag})"
            r = requests.get(f"{_comfy_url()}/view", params={"filename": iminfo["filename"],
                             "subfolder": iminfo.get("subfolder", ""), "type": iminfo.get("type", "output")}, timeout=120)
            r.raise_for_status()
            return r.content

        raw_path = WORK_DIR / f"ai_raw_{job['id']}.png"
        raw_path.write_bytes(_gen(im, "full scene"))

        box = job.get("subject_box")
        if box:
            # Subject Focus: the model works at ~1MP, so small subject details
            # (wipers, light bars, text) are sub-pixel in a full-scene pass.
            # Re-run the SAME preset on a crop of the original — the subject
            # gets several times more pixels — then feather-paste the detailed
            # render back into the full-scene render before cleanup.
            import cv2, numpy as np
            x0, y0, x1, y1 = box
            mx, my = (x1 - x0) * 0.08, (y1 - y0) * 0.08   # 8% context margin
            ex0, ey0 = max(0.0, x0 - mx), max(0.0, y0 - my)
            ex1, ey1 = min(1.0, x1 + mx), min(1.0, y1 + my)
            crop = im.crop((int(ex0 * im.width), int(ey0 * im.height),
                            int(ex1 * im.width), int(ey1 * im.height)))
            subj_path = WORK_DIR / f"ai_subj_{job['id']}.png"
            subj_path.write_bytes(_gen(crop, "subject"))
            job["phase"] = "compositing subject"
            full = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
            subj = cv2.imread(str(subj_path), cv2.IMREAD_GRAYSCALE)
            if full is None or subj is None:
                raise RuntimeError("could not read AI pass outputs for compositing")
            H, W = full.shape
            bx0, by0 = int(ex0 * W), int(ey0 * H)
            bx1, by1 = min(W, int(ex1 * W)), min(H, int(ey1 * H))
            bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)
            subj = cv2.resize(subj, (bw, bh), interpolation=cv2.INTER_AREA)
            region_u8 = full[by0:by1, bx0:bx1]
            # REGISTRATION: Kontext re-frames crop passes slightly (its input
            # scaler snaps to ~1MP aspect buckets), so the subject render comes
            # back a few px off the full-scene render. Compositing unaligned
            # passes doubles every edge ("two slightly different planes").
            # Align the subject pass onto the full-scene region first.
            try:
                t_blur = cv2.GaussianBlur((region_u8 < 200).astype(np.float32), (0, 0), 3.0)
                i_blur = cv2.GaussianBlur((subj < 200).astype(np.float32), (0, 0), 3.0)
                warp = np.eye(2, 3, dtype=np.float32)
                crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
                cv2.findTransformECC(t_blur, i_blur, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
                subj = cv2.warpAffine(subj, warp, (bw, bh),
                                      flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                print(f"  [ai] subject pass registered (dx={warp[0,2]:.1f}, dy={warp[1,2]:.1f})")
            except cv2.error as e:
                print(f"  [ai] subject registration failed (compositing unaligned): {e}")
            # Pre-composite cleanup of the subject pass: kontext sometimes adds
            # a soft gray wash around its strokes; pasted at full alpha that
            # wash smudges the composite and postclean turns it into blobs.
            # Normalize background to white and clip the wash before blending — only real ink crosses over.
            hi_s = float(np.percentile(subj, 90))
            if 80.0 <= hi_s < 250.0:
                subj = np.clip(subj.astype(np.float32) * (255.0 / hi_s), 0, 255).astype(np.uint8)
            subj = np.where(subj > 185, 255, subj).astype(np.uint8)
            # Content-aware composite: the subject pass wins where it DREW —
            # and wins OUTRIGHT there (sharp alpha), otherwise the full-scene
            # strokes a few px away survive as ghost doubles. Blank subject
            # areas stay transparent so full-scene content shows through.
            f = max(8.0, 0.05 * min(bw, bh))
            yy = np.minimum(np.arange(bh), np.arange(bh)[::-1]).astype(np.float32)
            xx = np.minimum(np.arange(bw), np.arange(bw)[::-1]).astype(np.float32)
            a_edge = np.clip(np.minimum.outer(yy, xx) / f, 0.0, 1.0)
            ink = (subj < 200).astype(np.float32)
            sigma = max(4.0, 0.015 * min(bw, bh))
            dens = cv2.GaussianBlur(ink, (0, 0), sigma)
            a = a_edge * np.clip(dens * 12.0, 0.0, 1.0)
            region = region_u8.astype(np.float32)
            full[by0:by1, bx0:bx1] = (a * subj.astype(np.float32) + (1.0 - a) * region).astype(np.uint8)
            cv2.imwrite(str(raw_path), full)

        job["phase"] = "post-AI cleanup (threshold)"
        newid = uuid4().hex[:8]
        cleaned_path = WORK_DIR / f"src_{newid}.png"     # threshold B/W = a NEW source
        mask_path = WORK_DIR / f"ai_mask_{job['id']}.png"
        q_raw, q_clean, filled_pct = _ai_postclean(raw_path, cleaned_path, mask_path)

        # SECOND cleanup candidate: LineArt annotator (contours, not threshold).
        # Produced as its own independent source so the user can compare and pick.
        # Status is explicit and surfaced — never silently dropped.
        laid, la_path, q_la = None, None, None
        la_status, la_error = "unavailable", None
        la_raw_path = WORK_DIR / f"ai_lineart_raw_{job['id']}.png"
        if use_openai or not _comfy_has_node("LineArtPreprocessor"):
            la_error = "skipped (OpenAI provider)" if use_openai else "LineArtPreprocessor node not installed in ComfyUI"
            print(f"  [ai] LineArt cleanup unavailable: {la_error}")
        else:
            try:
                job["phase"] = "post-AI cleanup (LineArt annotator)"
                _laid = uuid4().hex[:8]
                _la_path = WORK_DIR / f"src_{_laid}.png"
                q = _ai_lineart_cleanup(raw_path, _la_path, la_raw_path, job=job, timeout=run_timeout)
                if q is not None and _la_path.exists():
                    laid, la_path, q_la, la_status = _laid, _la_path, q, "ready"
                    print(f"  [ai] LineArt cleanup ready -> {_la_path.name}")
                else:
                    la_status, la_error = "failed", "annotator produced no usable output"
                    print(f"  [ai] LineArt cleanup failed: {la_error}")
            except _AICancelled:
                raise
            except Exception as e:
                la_status, la_error = "failed", str(e)[:300]
                print(f"  [ai] LineArt cleanup failed: {e}")

        # Texture backfill: synthesize the background strokes (gravel/clouds/
        # grass) Kontext refuses to invent, from the original photo's tone.
        bf = job.get("backfill", "light")
        bf_stats = None
        if bf in ("light", "full"):
            job["phase"] = "texture backfill"
            # Vector sidecar layer next to the cleaned source — the trace stage
            # merges it (toggleable); cleaned PNG stays pure outline art.
            bf_stats = _texture_backfill(src, cleaned_path, strength=bf,
                                         texture_svg=WORK_DIR / f"src_{newid}_texture.svg")

        verdict, reasons = _ai_verdict(q_raw, q_clean, filled_pct)
        quality = {"raw": q_raw, "cleaned": q_clean, "filled_pct": round(filled_pct, 2),
                   "verdict": verdict, "reasons": reasons, "engine": engine, "seed": seed,
                   "preset": job["preset"], "fidelity": job["fidelity"], "speed": job["speed"],
                   "backfill": bf_stats, "lineart": q_la}
        (WORK_DIR / f"ai_quality_{job['id']}.json").write_text(json.dumps(quality, indent=2))

        elapsed = round(time.time() - job["t0"], 1)
        job["result"] = {
            "cleaned_id": newid, "cleaned_src": cleaned_path.name,
            "original_url": f"/work/{src.name}", "raw_url": f"/work/{raw_path.name}",
            "cleaned_url": f"/work/{cleaned_path.name}", "mask_url": f"/work/{mask_path.name}",
            # LineArt-annotator cleanup candidate (None if annotator unavailable)
            "lineart_id": laid, "lineart_src": (la_path.name if la_path else None),
            "lineart_url": (f"/work/{la_path.name}" if la_path else None),
            "lineart_raw_url": (f"/work/{la_raw_path.name}" if la_raw_path.exists() else None),
            "lineart_status": la_status, "lineart_error": la_error,
            "subject_url": f"/work/ai_subj_{job['id']}.png" if job.get("subject_box") else None,
            "quality": quality, "elapsed_s": elapsed, "engine": engine, "seed": seed,
        }
        job["phase"], job["status"] = "done", "done"
        # Archive the run: AI minutes are expensive — snapshot so the user can
        # compare/restore later without regenerating.
        try:
            _runs_add({"type": "ai", "engine": engine, "seed": seed, "verdict": verdict,
                       "preset": job["preset"], "fidelity": job["fidelity"], "speed": job["speed"],
                       "backfill": bf, "background_detail": job.get("background_detail"),
                       "elapsed_s": elapsed, "cleaned_id": newid, "lineart_id": laid,
                       "source_name": src.name, "subject_box": job.get("subject_box"),
                       "backfill_stats": bf_stats},
                      copy_files={"original": src, "ai_raw": raw_path,
                                  "cleaned_threshold": cleaned_path,
                                  "cleaned_lineart": (la_path if la_path else ""),
                                  "cleaned": cleaned_path,
                                  "texture": WORK_DIR / f"src_{newid}_texture.svg"})
        except Exception as e:
            print(f"  [runs] archive failed: {e}")
        print(f"  [ai] job {job['id']} done: engine={engine} seed={seed} verdict={verdict} "
              f"elapsed={elapsed}s -> {cleaned_path.name}")
    except _AICancelled:
        job["phase"], job["status"] = "cancelled", "cancelled"
        print(f"  [ai] job {job['id']} cancelled by user")
    except Exception as e:
        job["phase"], job["status"], job["error"] = "failed", "error", str(e)
        print(f"  [ai] job {job['id']} failed: {e}")
    finally:
        _AI_LOCK.release()
