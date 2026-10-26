r"""Shared helpers for the fix A/B page: routed-SVG parsing and pen-width coverage rendering.

Coverage = LINE_AA polylines drawn at SS px/mm with integer thickness, area-downsampled. SS = 50 px/mm makes the
BIC bracket widths exact integers in supersample pixels (0.12 -> 6, 0.16 -> 8, 0.20 -> 10, 0.24 -> 12).
cv2 draws round joins/caps for thick lines, i.e. the same disc-swept footprint a round pen lays.
"""
import re
from pathlib import Path

import cv2
import numpy as np

SS = 50.0                       # supersample px/mm
PX2MM = None


def px2mm():
    global PX2MM
    if PX2MM is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # repository root
        from ccplotter.config import PX2MM as v
        PX2MM = float(v)
    return PX2MM


def parse_routed(svg_path):
    """Routed SVG -> list of ndarray(n,2) in page mm."""
    k = px2mm()
    text = Path(svg_path).read_text(encoding="utf-8")
    out = []
    for tag, ps in re.findall(r'<(polyline|polygon)\s+points="([^"]+)"', text):
        co = [tuple(map(float, t.split(","))) for t in ps.split() if "," in t]
        if len(co) < 2:
            continue
        a = np.array(co, np.float64) * k
        if tag == "polygon":
            a = np.vstack([a, a[:1]])
        out.append(a)
    # The routed-SVG writer emits a two-point stroke as <line x1 x2 y1 y2/>, which the
    # points= scan above never sees. They are appended AFTER the polylines so the polyline order is
    # unchanged; an SVG with no <line> returns only the polylines.
    for tag in re.findall(r'<line\b([^>]*)>', text):
        a = {kk: float(v) for kk, v in re.findall(r'\b(x1|x2|y1|y2)="([-+\d.eE]+)"', tag)}
        if len(a) == 4:
            out.append(np.array([(a["x1"], a["y1"]), (a["x2"], a["y2"])], np.float64) * k)
    return out


def bbox(polys):
    lo = np.min([p.min(0) for p in polys], 0)
    hi = np.max([p.max(0) for p in polys], 0)
    return lo, hi


def render(polys, x0, y0, w_mm, h_mm, width_mm, out_ppm, ss=SS):
    """Coverage uint8 (0=none, 255=full) of window [x0,x0+w_mm) x [y0,y0+h_mm) at out_ppm px/mm."""
    f = ss / out_ppm
    assert abs(f - round(f)) < 1e-6, (ss, out_ppm)
    f = int(round(f))
    W, H = int(round(w_mm * out_ppm)), int(round(h_mm * out_ppm))
    band = np.zeros((H * f, W * f), np.uint8)
    thick = int(round(width_mm * ss))
    assert thick >= 1
    sh = 4
    fx = 1 << sh
    xa, xb, ya, yb = x0 - width_mm, x0 + w_mm + width_mm, y0 - width_mm, y0 + h_mm + width_mm
    for p in polys:
        if p[:, 0].max() < xa or p[:, 0].min() > xb or p[:, 1].max() < ya or p[:, 1].min() > yb:
            continue
        q = np.stack([(p[:, 0] - x0) * ss - 0.5, (p[:, 1] - y0) * ss - 0.5], 1) * fx
        cv2.polylines(band, [q.round().astype(np.int32)], False, 255, thick, cv2.LINE_AA, sh)
    return cv2.resize(band, (W, H), interpolation=cv2.INTER_AREA)


def render_tiled(polys, x0, y0, w_mm, h_mm, width_mm, out_ppm, tile_mm=60.0):
    """Same as render() but in horizontal bands (memory)."""
    W, H = int(round(w_mm * out_ppm)), int(round(h_mm * out_ppm))
    out = np.zeros((H, W), np.uint8)
    rows = int(round(tile_mm * out_ppm))
    r = 0
    while r < H:
        r1 = min(H, r + rows)
        out[r:r1] = render(polys, x0, y0 + r / out_ppm, w_mm, (r1 - r) / out_ppm, width_mm, out_ppm)
        r = r1
    return out


def polyline_length(polys):
    return float(sum(np.hypot(*(np.diff(p, axis=0)).T).sum() for p in polys))


def parse_gcode(path):
    """Persisted gcode -> list of ndarray(n,2) machine-mm polylines, one per pen-down run (pen-down = Z at
    the file's lowest G1 Z; microlift hops between runs split them). Pure read; no repo import."""
    zs = []
    lines = Path(path).read_text(encoding="ascii", errors="replace").splitlines()
    for ln in lines:
        m = re.match(r"G1\s+Z([-\d.]+)", ln)
        if m:
            zs.append(float(m.group(1)))
    zdown = min(zs)
    out, cur, z = [], [], None
    for ln in lines:
        if not ln.startswith("G1"):
            continue
        mz = re.search(r"Z([-\d.]+)", ln)
        mx = re.search(r"X([-\d.]+)", ln)
        my = re.search(r"Y([-\d.]+)", ln)
        if mz:
            z = float(mz.group(1))
            if z > zdown + 1e-6 and len(cur) >= 2:
                out.append(np.array(cur, np.float64)); cur = []
            elif z > zdown + 1e-6:
                cur = []
        if mx and my and z is not None and abs(z - zdown) < 1e-6:
            cur.append((float(mx.group(1)), float(my.group(1))))
    if len(cur) >= 2:
        out.append(np.array(cur, np.float64))
    return out, zdown
