import json
import time

from ccplotter.config import STUDIO_DIR, WORK_DIR, MODEL_IMAGES_DIR
from ccplotter.ai.comfy import (_ai_cfg, _comfy_url, _comfy_available, _comfy_kontext_unet,
                                 _comfy_upload, _AICancelled, _comfy_run_cancellable_multi)
from ccplotter.ai.clean import AI_CLEAN_SPEEDS, _AI_LOCK
from ccplotter.studio import (STUDIO_CANDIDATE_STYLES, _studio_style_card, _crisp_ink,
                               _studio_detail_recovery)

# ── Source Art Studio — batch benchmark / evaluation ─────────────────────────
# Loops every saved pair, generates the standard (safe, subject-agnostic)
# candidate set per pair, stores everything for a results gallery + quick human
# scoring (winner picks + 1-5 rubric + failure tags + optional note), and keeps
# a reloadable history of benchmark sessions. The experimental multi-reference
# candidate is intentionally NOT part of the benchmark (it crashes this 16GB
# GPU) — the benchmark runs only the safe candidate roles.
BENCH_DIR = STUDIO_DIR / "benchmarks"
BENCH_DIR.mkdir(exist_ok=True)
BENCH_INDEX = BENCH_DIR / "index.json"

# Quick-scoring vocabulary — clicks, not prose. All subject-agnostic.
BENCH_WIN_CATEGORIES = [
    ("best_overall",    "Best overall"),
    ("best_subject",    "Best subject fidelity"),
    ("best_background", "Best background fidelity"),
    ("best_text",       "Best text/logo/detail"),
    ("best_plotter",    "Best plotter source"),
]
BENCH_RUBRIC = [
    ("subject",    "Subject shape / composition"),
    ("detail",     "Fine detail fidelity"),
    ("text",       "Text / logo legibility"),
    ("background", "Background completeness"),
    ("crispness",  "Line crispness / confidence"),
    ("plotter",    "Plotter suitability"),
]
BENCH_FAILURE_TAGS = [
    "lost subject detail", "lost background", "text/logo unreadable",
    "solid fills / blobs", "wrong proportions", "too sparse / empty",
    "too noisy / cluttered", "ghosted / duplicated lines", "faint / pale lines",
]

_BENCH_JOBS = {}   # bid -> live progress + cancel flag while a session runs

def _bench_index_read():
    try:
        return json.loads(BENCH_INDEX.read_text())
    except Exception:
        return []

def _bench_path(bid):
    return BENCH_DIR / bid / "session.json"

def _bench_load(bid):
    try:
        return json.loads(_bench_path(bid).read_text())
    except Exception:
        return None

def _bench_save(session):
    p = _bench_path(session["id"]); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(session, indent=1))
    rated = sum(1 for pr in session.get("pairs", [])
                if (pr.get("rating") or {}).get("ratings", {}).get("best_overall"))
    meta = {"id": session["id"], "label": session.get("label", ""),
            "created": session.get("created"), "status": session.get("status"),
            "pairs": len(session.get("pairs", [])), "rated": rated, "speed": session.get("speed")}
    items = [m for m in _bench_index_read() if m.get("id") != session["id"]]
    items.insert(0, meta)
    BENCH_INDEX.write_text(json.dumps(items, indent=1))

# ── Deterministic image metrics (the 'auto-grader' — answer key = the target) ─
# Subject-agnostic CV metrics comparing each candidate to the ChatGPT target.
# Rough by design: meant to catch obvious failures (empty, noisy, blobby, wrong
# density, missing structure), NOT to be a perfect quality model. No assumption
# of a single foreground subject — "central" vs "background" are just generic
# image regions so the metric works for any scene.
_BM_CMP = (512, 512)          # common compare size
_BM_INK_T = 150               # ink threshold (line art is dark-on-white)

def _bm_load_ink(path):
    import cv2, numpy as np
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None, None
    g = cv2.resize(g, _BM_CMP, interpolation=cv2.INTER_AREA)
    ink = (g < _BM_INK_T).astype(np.uint8)
    return g, ink

def _bm_ink_iou(a, b):
    import cv2, numpy as np
    k = np.ones((3, 3), np.uint8)
    a = cv2.dilate(a, k); b = cv2.dilate(b, k)   # dilate to tolerate small misalignment
    inter = float((a & b).sum()); uni = float((a | b).sum())
    return inter / uni if uni else 0.0

def _bm_abs_metrics(gray, ink):
    """Absolute, single-image metrics."""
    import cv2, numpy as np
    H, W = ink.shape; total = float(H * W)
    y0, y1, x0, x1 = int(H * .25), int(H * .75), int(W * .25), int(W * .75)
    central = np.zeros(ink.shape, bool); central[y0:y1, x0:x1] = True
    bg = ~central
    density = float(ink.sum()) / total
    cden = float(ink[central].sum()) / max(1, int(central.sum()))
    bden = float(ink[bg].sum()) / max(1, int(bg.sum()))
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    fill_limit = max(3.0, 0.012 * min(H, W))
    blob = float((dt > fill_limit).sum()) / total                 # solid-fill interiors
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([])
    tiny_thr = max(6.0, 0.00006 * total)
    speck_count = int((areas < tiny_thr).sum()) if areas.size else 0
    tiny_px = float(areas[areas < tiny_thr].sum()) if areas.size else 0.0
    noise = tiny_px / max(1.0, float(ink.sum()))
    inkpx = gray[ink.astype(bool)]
    darkness = float((255.0 - inkpx).mean()) / 255.0 if inkpx.size else 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    crisp = float((mag > 80).sum()) / max(1.0, float(ink.sum()))  # strong edges per ink px
    return {"density": round(density, 5), "central_density": round(cden, 5),
            "background_density": round(bden, 5), "blob_pct": round(blob, 5),
            "speck_count": speck_count, "noise": round(noise, 4),
            "darkness": round(darkness, 4), "crispness": round(min(crisp, 2.0), 4)}

def _bench_metrics_for_pair(pr, session):
    """Compute deterministic metrics for every candidate of one pair against its
    ChatGPT target. Returns None if the target image is unavailable."""
    import numpy as np
    tgt = pr.get("target")
    if not tgt:
        return None
    tg, tgi = _bm_load_ink(MODEL_IMAGES_DIR / tgt)
    if tgi is None:
        return None
    target_abs = _bm_abs_metrics(tg, tgi)
    t_den = target_abs["density"]
    cands = {}; cand_inks = {}
    for c in pr.get("candidates", []):
        cg, ci = _bm_load_ink(WORK_DIR / c["file"])
        if ci is None:
            continue
        m = _bm_abs_metrics(cg, ci)
        m["density_diff"] = round(abs(m["density"] - t_den), 5)
        m["ink_iou"] = round(_bm_ink_iou(ci, tgi), 4)                  # edge/skeleton similarity to target
        m["emptiness"] = round(max(0.0, 1.0 - (m["density"] / t_den)) if t_den > 1e-6 else 0.0, 4)
        score = 100.0
        score -= 60.0 * m["emptiness"]                                # missing structure = worst
        score -= 120.0 * max(0.0, m["blob_pct"] - 0.01)               # solid fills
        score -= 200.0 * max(0.0, m["noise"] - 0.02)                  # speckle noise
        score -= 80.0 * min(1.0, m["density_diff"] / max(0.02, t_den))# density mismatch vs target
        score += 30.0 * (m["ink_iou"] - 0.3)                          # structural overlap bonus
        if m["darkness"] < 0.45:
            score -= 25.0 * (0.45 - m["darkness"]) / 0.45             # pale lines
        m["metric_score"] = round(max(0.0, min(100.0, score)), 1)
        cands[c["style"]] = m; cand_inks[c["style"]] = ci
    sims = []; keys = list(cand_inks)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sims.append(_bm_ink_iou(cand_inks[keys[i]], cand_inks[keys[j]]))
    best = max(cands, key=lambda s: cands[s]["metric_score"]) if cands else None
    return {"target": target_abs, "candidates": cands, "best_by_metric": best,
            "similarity_among_candidates": round(float(np.mean(sims)), 4) if sims else None}

# JSON keys the AI critic returns (Part 2). Scores are on a 0-10 scale.
BENCH_CRITIC_WIN_KEYS = [("best_overall", "best_overall"), ("best_subject", "best_subject"),
                         ("best_background", "best_background"), ("best_text_logo", "best_text"),
                         ("best_plotter_source", "best_plotter")]
BENCH_CRITIC_SUB = ["subject_fidelity", "background_completeness", "text_logo_preservation",
                    "fine_detail_preservation", "line_crispness", "artifact_level",
                    "plotter_suitability", "target_similarity"]

def _bench_critic_prompt(session, pr):
    """A ready-to-paste grading prompt for a vision model (ChatGPT etc.). Lists
    the exact candidate labels and the strict JSON schema to return."""
    labels = [c["label"] for c in pr.get("candidates", [])]
    cand_schema = {lab: {k: 0 for k in BENCH_CRITIC_SUB} | {"notes": "..."} for lab in labels}
    schema = {"best_overall": "<label>", "best_subject": "<label>", "best_background": "<label>",
              "best_text_logo": "<label>", "best_plotter_source": "<label>",
              "candidates": cand_schema, "main_failures": ["..."], "recommended_next_tuning": ["..."]}
    return (
        "You are grading line-art renderings for a pen plotter. The FIRST image is the original photo. "
        "The SECOND image is the TARGET (the answer key — the look I want). The remaining images are "
        "local candidate renderings, in this order: " + "; ".join(labels) + ".\n\n"
        "Grade each candidate AGAINST THE TARGET as the answer key. Do not assume a single foreground "
        "subject; judge the whole scene. Score every sub-field 0-10 (10 = best; for artifact_level, "
        "10 = NO artifacts / clean). Pick the winning candidate label for each best_* field. "
        "List the main recurring failures and concrete next tuning steps.\n\n"
        "Return ONLY valid JSON, no prose, exactly this schema (candidate keys must match the labels above):\n"
        + json.dumps(schema, indent=2))

def _bench_aggregate(session):
    """Leaderboard + summary from THREE graders — deterministic metrics, the AI
    critic (pasted JSON), and the human picks. They are allowed to disagree."""
    styles = session.get("styles", [])
    labels = session.get("style_labels", {})
    lab2style = {v: k for k, v in labels.items()}
    # ---- human picks + rubric ----
    wins = {cat: {st: 0 for st in styles} for cat, _ in BENCH_WIN_CATEGORIES}
    crit_sum = {c: 0.0 for c, _ in BENCH_RUBRIC}; crit_n = {c: 0 for c, _ in BENCH_RUBRIC}
    fail_counts = {}; rated_pairs = 0; pair_means = []
    # ---- metrics ----
    mscore_sum = {st: 0.0 for st in styles}; mscore_n = {st: 0 for st in styles}
    mbest = {st: 0 for st in styles}
    magg = {k: [] for k in ("emptiness", "blob_pct", "noise", "darkness", "ink_iou")}
    bg_deficit = []; among = []
    # ---- critic ----
    cwins = {mycat: {st: 0 for st in styles} for _, mycat in BENCH_CRITIC_WIN_KEYS}
    csub_sum = {st: {k: 0.0 for k in BENCH_CRITIC_SUB} for st in styles}
    csub_n = {st: {k: 0 for k in BENCH_CRITIC_SUB} for st in styles}
    cfail = {}; ctune = {}; critic_pairs = 0

    for pr in session.get("pairs", []):
        rt = pr.get("rating") or {}; picks = rt.get("ratings") or {}
        if any(picks.get(cat) for cat, _ in BENCH_WIN_CATEGORIES):
            rated_pairs += 1
        for cat, _ in BENCH_WIN_CATEGORIES:
            st = picks.get(cat)
            if st in wins[cat]:
                wins[cat][st] += 1
        rub = rt.get("rubric") or {}; vals = []
        for c, _ in BENCH_RUBRIC:
            v = rub.get(c)
            if isinstance(v, (int, float)) and v > 0:
                crit_sum[c] += v; crit_n[c] += 1; vals.append(v)
        if vals:
            pair_means.append({"index": pr["index"], "original": pr.get("original"),
                               "mean": round(sum(vals) / len(vals), 2), "n": len(vals)})
        for tag in (rt.get("failures") or []):
            fail_counts[tag] = fail_counts.get(tag, 0) + 1
        # metrics
        mt = pr.get("metrics") or {}; tabs = mt.get("target") or {}
        for st, m in (mt.get("candidates") or {}).items():
            if st not in styles:
                continue
            mscore_sum[st] += m.get("metric_score", 0); mscore_n[st] += 1
            for k in magg:
                if k in m:
                    magg[k].append(m[k])
            if tabs.get("background_density", 0) > 1e-6:
                bg_deficit.append(max(0.0, (tabs["background_density"] - m.get("background_density", 0)) / tabs["background_density"]))
        if mt.get("best_by_metric") in mbest:
            mbest[mt["best_by_metric"]] += 1
        if mt.get("similarity_among_candidates") is not None:
            among.append(mt["similarity_among_candidates"])
        # critic
        cr = pr.get("critic") or {}
        if cr:
            critic_pairs += 1
            for jkey, mycat in BENCH_CRITIC_WIN_KEYS:
                st = lab2style.get(cr.get(jkey))
                if st in cwins[mycat]:
                    cwins[mycat][st] += 1
            for lab, sc in (cr.get("candidates") or {}).items():
                st = lab2style.get(lab)
                if st not in styles or not isinstance(sc, dict):
                    continue
                for k in BENCH_CRITIC_SUB:
                    v = sc.get(k)
                    if isinstance(v, (int, float)):
                        csub_sum[st][k] += v; csub_n[st][k] += 1
            for f in (cr.get("main_failures") or []):
                cfail[str(f)] = cfail.get(str(f), 0) + 1
            for t in (cr.get("recommended_next_tuning") or []):
                ctune[str(t)] = ctune.get(str(t), 0) + 1

    overall_wins = {st: sum(wins[cat].get(st, 0) for cat, _ in BENCH_WIN_CATEGORIES) for st in styles}
    avg_by_criterion = {c: (round(crit_sum[c] / crit_n[c], 2) if crit_n[c] else None)
                        for c, _ in BENCH_RUBRIC}
    def _mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    metric = {"avg_score": {st: (round(mscore_sum[st] / mscore_n[st], 1) if mscore_n[st] else None) for st in styles},
              "best_counts": mbest, "avg": {k: _mean(v) for k, v in magg.items()},
              "bg_deficit": _mean(bg_deficit), "similarity_among_candidates": _mean(among)}
    critic = {"wins": cwins,
              "overall_wins": {st: sum(cwins[mycat].get(st, 0) for _, mycat in BENCH_CRITIC_WIN_KEYS) for st in styles},
              "avg_sub": {st: {k: (round(csub_sum[st][k] / csub_n[st][k], 2) if csub_n[st][k] else None) for k in BENCH_CRITIC_SUB} for st in styles},
              "failures": sorted(cfail.items(), key=lambda kv: -kv[1]),
              "tuning": sorted(ctune.items(), key=lambda kv: -kv[1]), "pairs": critic_pairs}
    # ---- derived plain-English flags (Part 3) ----
    flags = []
    def fl(cond, text, level="warn"):
        if cond:
            flags.append({"level": level, "text": text})
    av = metric["avg"]
    fl(av["emptiness"] is not None and av["emptiness"] > 0.35, "Candidates are often too empty vs the target (missing structure).")
    fl(metric["bg_deficit"] is not None and metric["bg_deficit"] > 0.40, "Background is usually missing / under-drawn vs the target.")
    fl(av["blob_pct"] is not None and av["blob_pct"] > 0.03, "Frequent solid black fills / blobs.")
    fl(av["noise"] is not None and av["noise"] > 0.03, "Frequent speckle / isolated-component noise.")
    fl(av["darkness"] is not None and av["darkness"] < 0.50, "Linework is often pale / low-contrast.")
    fl(metric["similarity_among_candidates"] is not None and metric["similarity_among_candidates"] > 0.90,
       "Candidates are nearly identical — increase divergence (seeds / guidance / prompts).")
    def _critmean(k):
        xs = [critic["avg_sub"][st][k] for st in styles if critic["avg_sub"][st][k] is not None]
        return round(sum(xs) / len(xs), 2) if xs else None
    tlp, lcr, bgc = _critmean("text_logo_preservation"), _critmean("line_crispness"), _critmean("background_completeness")
    fl(tlp is not None and tlp < 5, "AI critic: text/logos usually lost (avg %s/10)." % tlp)
    fl(lcr is not None and lcr < 5, "AI critic: linework too pale/rough (avg %s/10)." % lcr)
    fl(bgc is not None and bgc < 5, "AI critic: background often incomplete (avg %s/10)." % bgc)
    scored = [(st, metric["avg_score"][st]) for st in styles if metric["avg_score"][st] is not None]
    weak = min(scored, key=lambda x: x[1])[0] if scored else None
    strong = max(scored, key=lambda x: x[1])[0] if scored else None

    return {"wins": wins, "overall_wins": overall_wins, "avg_by_criterion": avg_by_criterion,
            "hardest_pairs": sorted(pair_means, key=lambda x: x["mean"])[:5],
            "failures": sorted(fail_counts.items(), key=lambda kv: -kv[1]),
            "rated_pairs": rated_pairs, "labels": labels,
            "metric": metric, "critic": critic, "flags": flags,
            "weakest_role": weak, "strongest_role": strong}

def _bench_batch_workflow(image_name, styles, base_seed, unet, cfgd, style_card, steps):
    """ONE Kontext graph that renders every candidate style from a single set of
    loaders + one input encode. The 11GB Kontext UNET (which can't co-reside with
    the t5xxl text encoder in 16GB VRAM) is loaded ONCE per pair instead of once
    per candidate — the big time saver. Each style still gets its own prompt
    encode, guidance, seed and SaveImage. Returns (workflow, {style: save_node})."""
    wf = {
        "u": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "c": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors",
              "clip_name2": "t5xxl_fp8_e4m3fn_scaled.safetensors", "type": "flux", "device": "default"}},
        "v": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "s": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["2", 0]}},
        "e": {"class_type": "VAEEncode", "inputs": {"pixels": ["s", 0], "vae": ["v", 0]}},
    }
    node_for = {}
    for i, st in enumerate(styles):
        cs = STUDIO_CANDIDATE_STYLES[st]
        prompt = cs["lead"] + "\n\n" + style_card
        seed = (base_seed + cs["seed_off"]) & 0xFFFFFFFF
        p, r, g, z, k, d, sv = f"p{i}", f"r{i}", f"g{i}", f"z{i}", f"k{i}", f"d{i}", f"sv{i}"
        wf[p] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["c", 0], "text": prompt}}
        wf[r] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": [p, 0], "latent": ["e", 0]}}
        wf[g] = {"class_type": "FluxGuidance", "inputs": {"conditioning": [r, 0], "guidance": float(cs["guidance"])}}
        wf[z] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [p, 0]}}
        wf[k] = {"class_type": "KSampler", "inputs": {"model": ["u", 0], "positive": [g, 0], "negative": [z, 0],
                 "latent_image": ["e", 0], "seed": seed, "steps": int(steps),
                 "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
        wf[d] = {"class_type": "VAEDecode", "inputs": {"samples": [k, 0], "vae": ["v", 0]}}
        wf[sv] = {"class_type": "SaveImage", "inputs": {"images": [d, 0], "filename_prefix": f"ccbench_{st}"}}
        node_for[st] = sv
    return wf, node_for

def _bench_gen_candidates(src_path, out_dir, styles, speed, crisp, recover_on, base_seed, job):
    """Generate the standard candidate set for ONE source into out_dir. Same
    prompts/seeds/guidance/Crisp Ink/detail-recovery as the studio path, but all
    candidates render in a SINGLE batched ComfyUI graph (one UNET load per pair).
    Never runs the experimental multi-ref candidate. Returns the candidate dicts."""
    import requests
    from PIL import Image
    unet = _comfy_kontext_unet()
    if not unet:
        raise RuntimeError("FLUX Kontext is required (not installed).")
    cfgd = _ai_cfg()
    sp = AI_CLEAN_SPEEDS[speed]
    steps = sp["steps"]
    run_timeout = max(float(sp["timeout_s"]), 1200.0)
    # one prompt now does N candidates → give it proportionally longer
    batch_timeout = max(run_timeout, 600.0 * len(styles))
    style_card = _studio_style_card()
    im = Image.open(src_path).convert("RGB")
    gen_side = int(sp["max_side"])
    ps = min(1.0, gen_side / max(im.size))
    w = max(64, int(im.width * ps) // 8 * 8); h = max(64, int(im.height * ps) // 8 * 8)
    im_s = im.resize((w, h)) if ps < 1.0 else im
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_input.png"; im_s.save(tmp)
    name = _comfy_upload(tmp)
    pp = job.get("pair_phase", "")
    job["phase"] = f"{pp} · generating {len(styles)} candidates (one model load)"
    wf, node_for = _bench_batch_workflow(name, styles, base_seed, unet, cfgd, style_card, steps)
    outputs = _comfy_run_cancellable_multi(wf, batch_timeout, job)
    cands = []
    for st in styles:
        cs = STUDIO_CANDIDATE_STYLES[st]
        cseed = (base_seed + cs["seed_off"]) & 0xFFFFFFFF
        node = outputs.get(node_for[st]) or {}
        ims = node.get("images") or []
        if not ims:
            raise RuntimeError(f"no output image for candidate '{st}'")
        info = ims[0]
        r = requests.get(f"{_comfy_url()}/view", params={"filename": info["filename"],
                         "subfolder": info.get("subfolder", ""), "type": info.get("type", "output")}, timeout=120)
        r.raise_for_status()
        out = out_dir / f"cand_{st}.png"; out.write_bytes(r.content)
        # Optional detail recovery still runs as its own (model-reloading) pass —
        # it's opt-in and OFF by default for benchmarks for exactly that reason.
        if recover_on and cs.get("recover"):
            job["phase"] = f"{pp} · detail recovery ({cs['label']})"
            cfgd2 = dict(cfgd); cfgd2["kontext_guidance"] = cs["guidance"]; cfgd2["kontext_steps"] = steps
            try:
                _studio_detail_recovery(im_s, out, style_card, cseed, unet, cfgd2, run_timeout, job)
            except _AICancelled:
                raise
            except Exception as e:
                print(f"  [bench] detail recovery failed: {e}")
        if crisp:
            _crisp_ink(out)
        (out_dir / f"cand_{st}.settings.json").write_text(json.dumps(
            {"style": st, "label": cs["label"], "seed": cseed, "speed": speed,
             "steps": steps, "guidance": cs["guidance"], "crisp": crisp,
             "recover": bool(recover_on and cs.get("recover")), "batched": True}, indent=1))
        rel = out.relative_to(WORK_DIR).as_posix()
        cands.append({"style": st, "label": cs["label"], "file": rel,
                      "url": f"/work/{rel}?t={int(time.time())}"})
    return cands

def _bench_worker(session):
    job = _BENCH_JOBS[session["id"]]
    cancelled = False
    try:
        if not _comfy_available():
            raise RuntimeError(f"ComfyUI is not running at {_comfy_url()}.")
        total = len(session["pairs"])
        for i, pr in enumerate(session["pairs"]):
            if job.get("cancel"):
                cancelled = True; break
            job["current_pair"] = i; job["done_pairs"] = i
            job["pair_phase"] = f"pair {i + 1}/{total} ({pr.get('original', '?')})"
            # resume: skip pairs that already have a complete candidate set
            if session.get("resume") and not pr.get("error") and \
               len(pr.get("candidates") or []) == len(session["styles"]):
                continue
            src = MODEL_IMAGES_DIR / (pr.get("original") or "")
            if not src.is_file():
                pr["error"] = f"original '{pr.get('original')}' not found"; pr["candidates"] = []
                _bench_save(session); continue
            try:
                out_dir = BENCH_DIR / session["id"] / pr["case"]
                pr["candidates"] = _bench_gen_candidates(
                    src, out_dir, session["styles"], session["speed"],
                    session["crisp"], session["recover"], session["seed"], job)
                pr["error"] = None
                try:                                  # auto-grade against the target
                    m = _bench_metrics_for_pair(pr, session)
                    if m:
                        pr["metrics"] = m
                except Exception as e:
                    print(f"  [bench] metrics failed: {e}")
            except _AICancelled:
                cancelled = True; _bench_save(session); break
            except Exception as e:
                msg = str(e)
                if any(s in msg for s in ("Connection", "refused", "10054", "10061")):
                    msg = ("ComfyUI crashed (native OOM — Kontext is near this 16GB GPU's limit). "
                           "Restart ComfyUI and Resume this benchmark to finish the remaining pairs.")
                pr["error"] = msg[:240]; pr["candidates"] = pr.get("candidates") or []
            _bench_save(session)
        session["status"] = "cancelled" if cancelled else "done"
        job["done_pairs"] = total
    except _AICancelled:
        session["status"] = "cancelled"
    except Exception as e:
        session["status"] = "error"; session["error"] = str(e)[:240]
    finally:
        session["finished"] = time.time()
        _bench_save(session)
        _AI_LOCK.release()
        job["live"] = False

def _bench_export_rows(session):
    """Flatten the session to one row per candidate (denormalized: pair-level
    human picks/rubric/fails repeated on each candidate row) for CSV/analysis."""
    labels = session.get("style_labels", {})
    lab2style = {v: k for k, v in labels.items()}
    rows = []
    for pr in session.get("pairs", []):
        rt = pr.get("rating") or {}; picks = rt.get("ratings") or {}; rub = rt.get("rubric") or {}
        fails = "; ".join(rt.get("failures") or []); note = (rt.get("note") or "").replace("\n", " ")
        mt = pr.get("metrics") or {}; mc = mt.get("candidates") or {}; bestm = mt.get("best_by_metric")
        cr = pr.get("critic") or {}
        cbest = {jk: lab2style.get(cr.get(jk)) for jk, _ in BENCH_CRITIC_WIN_KEYS} if cr else {}
        for c in pr.get("candidates") or []:
            st = c["style"]; m = mc.get(st) or {}
            row = {"pair": pr["index"], "original": pr.get("original"), "target": pr.get("target"),
                   "style": st, "label": c["label"], "error": pr.get("error") or ""}
            for k in ("metric_score", "density", "central_density", "background_density", "blob_pct",
                      "speck_count", "noise", "darkness", "crispness", "ink_iou", "emptiness", "density_diff"):
                row[k] = m.get(k)
            row["metric_best"] = int(st == bestm)
            row["critic_best_overall"] = int(cbest.get("best_overall") == st) if cr else ""
            for cat, _ in BENCH_WIN_CATEGORIES:
                row["human_" + cat] = int(picks.get(cat) == st)
            for rc, _ in BENCH_RUBRIC:
                row["rubric_" + rc] = rub.get(rc)
            row["failure_tags"] = fails; row["note"] = note
            rows.append(row)
    return rows

def _bench_export_md(session):
    agg = _bench_aggregate(session)
    labels = session.get("style_labels", {})
    L = lambda st: (labels.get(st) or st or "-")
    out = []
    out.append(f"# Benchmark summary - {session.get('label','')}  (`{session['id']}`)")
    out.append(f"\n{len(session.get('pairs',[]))} pairs - {agg['rated_pairs']} human-rated - "
               f"{agg['critic']['pairs']} AI-critic graded - speed {session.get('speed')}\n")
    out.append(f"**Strongest role (metric):** {L(agg.get('strongest_role'))} - "
               f"**weakest:** {L(agg.get('weakest_role'))}\n")
    if agg.get("flags"):
        out.append("## Diagnosis")
        for f in agg["flags"]:
            out.append(f"- {f['text']}")
        out.append("")
    styles = session.get("styles", [])
    out.append("## Leaderboard")
    out.append("| Role | Human wins | Critic wins | Metric avg | Metric wins |")
    out.append("|---|---|---|---|---|")
    for st in styles:
        out.append(f"| {L(st)} | {agg['overall_wins'].get(st,0)} | "
                   f"{agg['critic']['overall_wins'].get(st,0)} | "
                   f"{agg['metric']['avg_score'].get(st,'-')} | {agg['metric']['best_counts'].get(st,0)} |")
    out.append("\n## Human avg by criterion")
    for c, lbl in BENCH_RUBRIC:
        v = agg['avg_by_criterion'].get(c)
        out.append(f"- {lbl}: {v if v is not None else '-'}")
    if agg["failures"]:
        out.append("\n## Human failure tags")
        for t, n in agg["failures"]:
            out.append(f"- {t}: {n}")
    if agg["critic"]["failures"]:
        out.append("\n## AI critic failures")
        for t, n in agg["critic"]["failures"]:
            out.append(f"- {t}: {n}")
    out.append("\n## Per-pair")
    for pr in session.get("pairs", []):
        rt = pr.get("rating") or {}; picks = rt.get("ratings") or {}; rub = rt.get("rubric") or {}
        out.append(f"\n### #{pr['index']+1} {pr.get('original','?')}")
        out.append(f"- target: {pr.get('target') or '-'}")
        if pr.get("error"):
            out.append(f"- ERROR: {pr['error']}"); continue
        mc = (pr.get("metrics") or {}).get("candidates") or {}
        for c in pr.get("candidates") or []:
            ms = (mc.get(c["style"]) or {}).get("metric_score")
            out.append(f"- {c['label']}: metric_score={ms}")
        out.append(f"- human best overall: {L(picks.get('best_overall')) if picks.get('best_overall') else '-'}")
        if any(rub.get(rc) for rc, _ in BENCH_RUBRIC):
            out.append("- rubric: " + ", ".join(f"{lbl}={rub.get(rc)}" for rc, lbl in BENCH_RUBRIC if rub.get(rc)))
        if rt.get("failures"):
            out.append("- failures: " + ", ".join(rt["failures"]))
        if rt.get("note"):
            out.append(f"- note: {rt['note']}")
    return "\n".join(out) + "\n"
