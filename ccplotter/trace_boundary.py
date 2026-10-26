"""Trace-boundary provenance.

The run-provenance sidecar records the REQUEST that made a run. When two runs of the
same source disagree, the cause is found by re-deriving the gray raster and the path
set. This module records, per traced item, the quantities at the trace boundary so
that comparison is one JSON diff:

    source sha256 + dims        the raster file handed to the renderer
    clahe                       the EFFECTIVE clahe flag (and clip when overridden)
    gray sha256 + dims          the post-CLAHE uint8 raster tracev3 thresholds
    black_fraction              fraction of gray pixels < 128 (plain, documented cut)
    gray_crush_fraction         trace_v3's calibrated shadow-crush metric (from its audit)
    mask sha256 + dims          the FINAL trace mask (after hole fill / gap close)
    mask_ink_fraction           fraction of the mask that is ink
    passes / lifts / total_polys / dropped_components   trace_v3 audit counts
    routed path-set sha         ONE definition, below
    routed polylines / lifts    counted from the routed SVG
    preview stroke mm / px      the width the preview rasteriser used

Everything is computed IN MEMORY from arrays the pipeline already holds (gray raster,
mask) or from the routed SVG it just wrote, so it works outside debug mode. It is
metadata only: nothing here is read by any generation code, nothing is added to
info.json, `_TRACE_AUDIT`, the HTTP response or the gcode (the gcode header keeps only its
run back-link line), and every entry point swallows its own errors and records an
error string instead (never raises into a job).

One-trace-per-process limit. The last-mask record (`_LAST_MASK`) is module state
with the same limit as trace_v3's `_TRACE_AUDIT`: valid because /api/process is serialised
by `_PROCESS_LOCK` and render() reads it right after its own trace.

Cache-hit note. The renderer's SVG cache means a job can reuse an earlier trace, in
which case the gray raster and mask are NOT in memory. The render step therefore
persists its record beside the cached render as `<render>.svg.boundary.json`; the
composite step reads it back. A render cached before this capture existed has no
sidecar and the block says so explicitly instead of inventing numbers.

PATH-SET DEFINITION (the one definition; tests point here)
--------------------------------------------------------------------
The routed SVG (`layout.routed.svg`, i.e. after routing and the fill-page / paper
keep-out clips, before the preview pen-width is applied) is parsed as TEXT with the
regexes in `parse_routed_svg_text`: every `<polyline points>` and `<polygon points>`
(polygon ring closed by appending its first point) and every `<line x1 y1 x2 y2>`.
Coordinates are the SVG user units (CSS px) times PX2MM (25.4/96), giving mm. Each
polyline is rounded to 3 decimals (`np.round(p, 3)`), taken in canonical orientation
`min(forward, reversed)`, the keys are sorted, and the hash is
`sha256(repr(sorted_keys).encode())`. The offline comparison script's
`pathset(polys, 3)` computes the same hash (tests/test_trace_boundary.py pins it on
committed SVGs).

Why text and not the in-memory vpype document: vpype rescales a document read back from
SVG by width/viewBox (a 1e-6 relative error when the viewBox is written to 10 decimals),
which moves coordinates by ~1e-4 mm and flips 3-decimal roundings, so a hash of the
re-read document does not equal the hash of the file's own numbers.

Note: resolved.mm_per_px is the trace_v3 back-solve, not the placed scale; the
realised pass spacing on the page is `realised_pitch` (composite._realised_pitch_record).
"""
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from ccplotter.config import PX2MM

BOUNDARY_SCHEMA = 1
BOUNDARY_SIDECAR_SUFFIX = ".boundary.json"
PATHSET_DEFINITION = (
    "routed SVG parsed as text (polyline/polygon[closed]/line), CSS px * 25.4/96 = mm, each "
    "polyline np.round(.,3), canonical orientation min(fwd,rev), keys sorted, "
    "sha256(repr(keys)); == metrics.pathset(parse_svg(f),3) of "
    "artifacts/cleaned-heli-app-vs-b2-2026-10-04/scripts")

_POLY_RE = re.compile(r'<(polyline|polygon)\s+points="([^"]+)"')
_LINE_RE = re.compile(r'<line\b([^>]*)>')
_LINE_ATTR_RE = re.compile(r'\b(x1|x2|y1|y2)="([-+\d.eE]+)"')
_LAYER_ID_RE = re.compile(r'id="layer(\d+)"')

# ---- last-trace mask record (set by trace_v3, read by render; one job at a time) ----
# `_LAST_MASK` follows the same one-trace-per-process limit as trace_v3's `_TRACE_AUDIT`
# both are module-level "last trace" state, valid because /api/process is
# serialised by `_PROCESS_LOCK` and render() reads the record immediately after its own
# trace. Two concurrent traces in one process would cross-contaminate both the audit and
# this record; the tests and the replay/parity harnesses run one job per process.
_LAST_MASK = None


def reset_mask_record():
    global _LAST_MASK
    _LAST_MASK = None


def record_mask(mask):
    """Called by trace_v3._trace_svg right after the paths are written, with the FINAL
    mask. Never raises."""
    global _LAST_MASK
    try:
        m = np.asarray(mask).astype(bool)
        _LAST_MASK = {
            "sha256": hashlib.sha256(np.packbits(m).tobytes()).hexdigest(),
            "dims_wh": [int(m.shape[1]), int(m.shape[0])],
            "ink_fraction": round(float(m.mean()), 6) if m.size else 0.0,
            "hash_note": "sha256 of np.packbits(mask.astype(bool)); dims WxH",
        }
    except Exception as e:
        _LAST_MASK = {"error": f"{type(e).__name__}: {e}"}


def last_mask_record():
    return _LAST_MASK


# ---- path set ----

def parse_routed_svg_text(text, unit=PX2MM):
    """Return (all_polylines_mm, {layer_id_str: [polylines_mm]}). See module docstring."""
    by_layer = {}
    allp = []
    for chunk in re.split(r'(?=<g[\s>])', text):
        if not chunk.startswith("<g"):
            key = None
        else:
            m = _LAYER_ID_RE.search(chunk[:400])
            key = m.group(1) if m else None
        out = []
        for tag, ps in _POLY_RE.findall(chunk):
            co = [tuple(map(float, s.split(","))) for s in ps.split() if "," in s]
            if len(co) < 2:
                continue
            a = np.array(co, np.float64) * unit
            if tag == "polygon":
                a = np.vstack([a, a[:1]])
            out.append(a)
        for tag in _LINE_RE.findall(chunk):
            a = {k: float(v) for k, v in _LINE_ATTR_RE.findall(tag)}
            if len(a) == 4:
                out.append(np.array([(a["x1"], a["y1"]), (a["x2"], a["y2"])], np.float64) * unit)
        if out:
            by_layer.setdefault(key, []).extend(out)
            allp.extend(out)
    return allp, by_layer


def pathset_sha256(polys, nd=3):
    keys = []
    for p in polys:
        f = tuple(map(tuple, np.round(p, nd).tolist()))
        keys.append(min(f, f[::-1]))
    keys.sort()
    return hashlib.sha256(repr(keys).encode()).hexdigest()


# ---- render-step capture ----

def gray_record(img, clahe, clahe_clip=None):
    """Record of the post-CLAHE gray raster entering the threshold. Never raises."""
    try:
        a = np.ascontiguousarray(img)
        rec = {"sha256": hashlib.sha256(a.tobytes()).hexdigest(),
               "dims_wh": [int(a.shape[1]), int(a.shape[0])],
               "dtype": str(a.dtype),
               "black_fraction": round(float((a < 128).mean()), 6),
               "black_fraction_note": "fraction of gray pixels < 128",
               "clahe": bool(clahe)}
        if clahe_clip is not None:
            rec["clahe_clip"] = float(clahe_clip)
        return rec
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def source_record(raw_path):
    try:
        raw_path = Path(raw_path)
        rec = {"sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
               "dims_wh": None}
        try:
            from PIL import Image
            with Image.open(raw_path) as im:
                rec["dims_wh"] = [int(im.size[0]), int(im.size[1])]
        except Exception:
            pass
        return rec
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def write_render_sidecar(out_path, opts, raw_path, gray_rec, audit):
    """Persist the render-step half of the record beside the render SVG. Never raises."""
    try:
        from ccplotter.atomic_io import atomic_write_text
        audit = audit or {}
        mk = last_mask_record()
        rec = {
            "schema": BOUNDARY_SCHEMA,
            "source": source_record(raw_path),
            "clahe": bool(opts.get("clahe", False)),
            "gray": gray_rec,
            "gray_crush_fraction": audit.get("gray_crush_fraction"),
            "mask": mk,
            "trace_counts": {k: audit.get(k) for k in
                             ("passes", "lifts", "total_polys", "dropped_components")},
        }
        atomic_write_text(Path(str(out_path) + BOUNDARY_SIDECAR_SUFFIX), json.dumps(rec))
    except Exception as e:
        try:
            Path(str(out_path) + BOUNDARY_SIDECAR_SUFFIX).write_text(
                json.dumps({"schema": BOUNDARY_SCHEMA, "error": f"{type(e).__name__}: {e}"}))
        except Exception:
            pass


# ---- composite-step collection ----

def collect(routed_svg, layer_of, item_svgs, preview_pen_mm, laid_pen_mm=None):
    """Assemble the `trace_boundary` block. layer_of: {layer int: item id}, item_svgs:
    {layer int: render SVG path}. Never raises: failure -> {"error": ...}."""
    try:
        t0 = time.time()
        text = Path(routed_svg).read_text(encoding="utf-8")
        allp, by_layer = parse_routed_svg_text(text)
        items = {}
        for L, did in layer_of.items():
            polys = by_layer.get(str(L), [])
            rec = {"layer": int(L),
                   "routed": {"polylines": len(polys),
                              "lifts": max(0, len(polys) - 1),
                              "pathset_sha256_nd3": pathset_sha256(polys) if polys else None}}
            sc = Path(str(item_svgs.get(L, "")) + BOUNDARY_SIDECAR_SUFFIX)
            try:
                if not item_svgs.get(L):
                    rec["trace"] = None
                elif sc.exists():
                    rec["trace"] = json.loads(sc.read_text())
                else:
                    rec["trace"] = {"unavailable": "no boundary sidecar (not a tracev3 "
                                    "render, or cached before boundary capture existed)"}
            except Exception as e:
                rec["trace"] = {"error": f"{type(e).__name__}: {e}"}
            items[str(did)] = rec
        pm = float(preview_pen_mm)
        return {
            "schema": BOUNDARY_SCHEMA,
            "pathset_definition": PATHSET_DEFINITION,
            "routed": {"polylines": len(allp),
                       "lifts_within_layers": sum(max(0, len(v) - 1) for v in by_layer.values()),
                       "pathset_sha256_nd3": pathset_sha256(allp) if allp else None},
            "preview_stroke": {"mm": round(pm, 6), "px": round(pm / PX2MM, 6)},
            "laid_pen_mm": laid_pen_mm,
            "items": items,
            "collect_seconds": round(time.time() - t0, 3),
        }
    except Exception as e:
        return {"schema": BOUNDARY_SCHEMA, "error": f"{type(e).__name__}: {e}"}
