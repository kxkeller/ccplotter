r"""Parsing + true-width stamp rasteriser.
Stamp = rasterise the CENTRELINE at 1 device px (cv2.polylines thickness=1) on a supersampled canvas, then keep every
supersample pixel within width/2 of it (exact Euclidean distance transform), so the laid width is width_mm at any angle
(checked by a positive control in the original experiment's render script). This is NOT the
cv2.polylines thickness=width approach, which rounds the width to integer px and double-counts join thickness."""
import re
from pathlib import Path
import cv2, numpy as np

PX2MM = 25.4 / 96.0


def parse_svg(path, unit=PX2MM):
    t = Path(path).read_text(encoding="utf-8")
    out = []
    for tag, ps in re.findall(r'<(polyline|polygon)\s+points="([^"]+)"', t):
        co = [tuple(map(float, s.split(","))) for s in ps.split() if "," in s]
        if len(co) < 2:
            continue
        a = np.array(co, np.float64) * unit
        if tag == "polygon":
            a = np.vstack([a, a[:1]])
        out.append(a)
    for tag in re.findall(r'<line\b([^>]*)>', t):
        a = {k: float(v) for k, v in re.findall(r'\b(x1|x2|y1|y2)="([-+\d.eE]+)"', tag)}
        if len(a) == 4:
            out.append(np.array([(a["x1"], a["y1"]), (a["x2"], a["y2"])], np.float64) * unit)
    return out


def bbox(polys):
    return np.min([p.min(0) for p in polys], 0), np.max([p.max(0) for p in polys], 0)


def stamp(polys, x0, y0, w_mm, h_mm, width_mm, out_ppm, ss=40):
    """Coverage uint8 (255 = fully inked) for window [x0,x0+w)x[y0,y0+h) at out_ppm px/mm, true stroke width."""
    f = ss / out_ppm
    assert abs(f - round(f)) < 1e-9, (ss, out_ppm)
    f = int(round(f))
    W, H = int(round(w_mm * out_ppm)), int(round(h_mm * out_ppm))
    band = np.zeros((H * f, W * f), np.uint8)
    pad = width_mm
    for p in polys:
        if p[:, 0].max() < x0 - pad or p[:, 0].min() > x0 + w_mm + pad or p[:, 1].max() < y0 - pad or p[:, 1].min() > y0 + h_mm + pad:
            continue
        q = np.stack([(p[:, 0] - x0) * ss, (p[:, 1] - y0) * ss], 1)
        cv2.polylines(band, [np.round(q).astype(np.int32)], False, 255, 1, cv2.LINE_8)
    # exact Euclidean footprint: distance to the 1-px centreline <= r (anti-aliased by 1 px), then area-downsample.
    # (A discretised-disc cv2.dilate was tried first and laid a 45-degree line 8% thin -- see stamp_control.json history.)
    r = width_mm * ss / 2.0
    dt = cv2.distanceTransform(255 - band, cv2.DIST_L2, 5)
    band = (np.clip(r + 0.5 - dt, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.resize(band, (W, H), interpolation=cv2.INTER_AREA)


def stamp_tiled(polys, x0, y0, w_mm, h_mm, width_mm, out_ppm, ss=40, tile_mm=50.0):
    W, H = int(round(w_mm * out_ppm)), int(round(h_mm * out_ppm))
    out = np.zeros((H, W), np.uint8)
    rows = int(round(tile_mm * out_ppm)); r = 0
    ov = int(round(2 * width_mm * out_ppm)) + 1       # overlap so the dilation has no seam, then trim
    while r < H:
        r1 = min(H, r + rows)
        a, b = max(0, r - ov), min(H, r1 + ov)
        t = stamp(polys, x0, y0 + a / out_ppm, w_mm, (b - a) / out_ppm, width_mm, out_ppm, ss)
        out[r:r1] = t[r - a:r1 - a]
        r = r1
    return out
