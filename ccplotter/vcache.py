import json
import threading
import time
from pathlib import Path

from ccplotter.atomic_io import atomic_copy
from ccplotter.config import WORK_DIR

# ── Current-image variant cache: while the SAME source set is loaded, flipping
# between Line Art / Sketch (and tweaking settings) re-runs the whole pipeline
# every time even though the trace SVG is already disk-cached. This caches the
# FULL result (preview/clean/xray SVGs + gcode + JSON) per (source-fingerprint,
# settings) so switching BACK to a variant is an instant file-restore, not a
# recompute. It is SESSION/current-image scoped — not durable like Recent Runs —
# and is discarded the moment the loaded source set changes (add/delete/replace/
# crop/transform all change the key or the fingerprint). ──
VCACHE_DIR = WORK_DIR / "vcache"
VCACHE_DIR.mkdir(exist_ok=True)
VCACHE_KEEP = 24
_VCACHE = {}              # body_key -> {"res","files","order","label","at"}
_VCACHE_FP = None         # fingerprint of the currently-loaded source set
_VCACHE_SEQ = 0           # insertion counter for LRU-ish eviction
_VCACHE_CURRENT = None    # body_key of the result currently shown (for "active" marker)
_VCACHE_LOCK = threading.Lock()
# Stable output filenames a run produces; snapshotted/restored as a unit.
_VCACHE_OUTPUTS = ("layout.preview.svg", "layout.clean.svg", "layout.routed.svg",
                   "layout.connectors.svg", "layout.gcode",
                   "debug_connector_mask_check.json", "debug_connector_mask_rejected.json")

def _source_fingerprint(items: list) -> str:
    """Identity of the loaded source COMPOSITION: the ordered item ids plus each
    source file's on-disk (mtime, size). Replacing/re-cropping/adding/deleting a
    source changes this, which discards the whole variant cache."""
    import hashlib
    parts = []
    for it in items:
        did = str(it.get("id"))
        m = ""
        for f in WORK_DIR.glob(f"src_{did}.*"):
            if "_render_" in f.name or f.name.endswith("_traced.svg"):
                continue
            try: st = f.stat(); m = f"{f.name}:{int(st.st_mtime)}:{st.st_size}"
            except Exception: pass
            break
        parts.append(f"{did}|{m}")
    return hashlib.md5("\n".join(parts).encode()).hexdigest()[:12]

def _resolved_inputs_fingerprint(pen_id, pen_profile, cal, reach) -> str:
    """Stable fingerprint of the RESOLVED, server-side inputs a job depends on
    but the request body does not carry: the resolved pen-profile dict (z_touch,
    core/pen/fill mm, speeds, clamps...), calibration.json, and the active reach
    profile. Folded into the variant key so a stored pen/calibration/reach change
    (API write OR hand edit of the JSON) is a cache MISS, never stale gcode at an
    old Z. Canonical JSON (sort_keys) -> deterministic across processes; no id()
    or set iteration is involved."""
    import hashlib
    blob = json.dumps({"pen_id": pen_id, "pen_profile": pen_profile,
                       "cal": cal, "reach": reach},
                      sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.md5(blob.encode()).hexdigest()[:12]

def _variant_key(body: dict, resolved_fp: str = None) -> str:
    """All settings that affect output → one hash. The whole request body is
    deterministic w.r.t. output (style, resolution, thresholds, gain, routing,
    paper, item transforms/crops, job_type), so hash it canonically. job_type is
    included so a gcode build (which also produces the .gcode) is a distinct
    entry from a preview-only build. `resolved_fp` (optional) is
    _resolved_inputs_fingerprint(): the server-resolved pen profile / calibration
    / reach the body does not contain; None keeps the body-only key."""
    import hashlib
    blob = json.dumps(body, sort_keys=True, default=str)
    if resolved_fp is not None:
        blob += "|resolved:" + resolved_fp
    return hashlib.md5(blob.encode()).hexdigest()[:12]

def _variant_label(body: dict) -> str:
    """Short human label for the variant switcher, e.g. 'Line Art', 'Sketch gain 0.15'."""
    style = body.get("style", "")
    nice = {"plotready": "Line Art", "lineart": "Line Art", "sketchv2": "Sketch",
            "hybrid": "Hybrid", "tsp": "Scribble", "bwphoto": "B/W photo",
            "illustration": "Trace V3+"}.get(style, style or "?")
    parts = [nice]
    if style == "sketchv2":
        try: parts.append(f"gain {float(body.get('sketch_gain', 0.12)):.2f}")
        except Exception: pass
    extras = []
    _pt = str(body.get("preserve_text", "balanced")).lower()
    if _pt in ("off", "false", "0"): extras.append("text off")
    elif _pt == "aggressive": extras.append("text aggr")
    _nf = str(body.get("noise_filter", "off")).lower()
    if _nf in ("light", "strong"): extras.append(f"nf:{_nf}")
    cd = float(body.get("connector_distance", 0) or 0); tz = int(body.get("target_z_moves", 0) or 0)
    if cd > 0 or tz > 0: extras.append("routed")
    if body.get("job_type") == "gcode": extras.append("g-code")
    label = " ".join(parts)
    if extras: label += " (" + ", ".join(extras) + ")"
    return label

def _vcache_get(fp: str, key: str):
    """Return a cached result for (fp,key) and RESTORE its output files to the
    stable layout.* names, or None. Clears the cache if the source set changed."""
    global _VCACHE_FP, _VCACHE_CURRENT
    import shutil
    with _VCACHE_LOCK:
        if fp != _VCACHE_FP:                       # source set changed → drop everything
            _VCACHE.clear(); _VCACHE_FP = fp; _VCACHE_CURRENT = None
            return None
        ent = _VCACHE.get(key)
        if not ent:
            return None
        for stable, cached in ent["files"].items():
            cp = Path(cached)
            if not cp.is_file():                   # snapshot lost (manual cleanup) → miss
                return None
        for stable, cached in ent["files"].items():
            # Restoring a cached variant OVERWRITES the live work/<stable> file
            # (e.g. layout.routed.svg, layout.gcode). Copying in place could leave
            # a shorter cached file's bytes followed by the tail of a longer live
            # one. atomic_copy() writes to a temp file in WORK_DIR first, then
            # os.replace()s over the target, so a reader (gcode step, frontend
            # /work/ URL) only ever sees the complete old file or the complete
            # new one.
            try: atomic_copy(cached, WORK_DIR / stable)
            except Exception: return None
        _VCACHE_CURRENT = key
        return dict(ent["res"])

def _vcache_put(fp: str, key: str, res: dict, label: str = ""):
    """Snapshot the current output files and store the result under (fp,key)."""
    global _VCACHE_FP, _VCACHE_SEQ, _VCACHE_CURRENT
    import shutil
    with _VCACHE_LOCK:
        if fp != _VCACHE_FP:                       # fingerprint moved on → reset to this set
            _VCACHE.clear(); _VCACHE_FP = fp; _VCACHE_CURRENT = None
        files = {}
        for stable in _VCACHE_OUTPUTS:
            src = WORK_DIR / stable
            if src.is_file():
                dst = VCACHE_DIR / f"{key}__{stable}"
                try: atomic_copy(src, dst); files[stable] = str(dst)
                except Exception: pass
        _VCACHE_SEQ += 1
        _VCACHE[key] = {"res": dict(res), "files": files, "order": _VCACHE_SEQ,
                        "label": label, "at": time.strftime("%H:%M:%S")}
        _VCACHE_CURRENT = key
        # Evict oldest beyond the cap (and remove their snapshot files).
        if len(_VCACHE) > VCACHE_KEEP:
            for k in sorted(_VCACHE, key=lambda k: _VCACHE[k]["order"])[:len(_VCACHE) - VCACHE_KEEP]:
                for cached in _VCACHE.pop(k)["files"].values():
                    try: Path(cached).unlink()
                    except Exception: pass
