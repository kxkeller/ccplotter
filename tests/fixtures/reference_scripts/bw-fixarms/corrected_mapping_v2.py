r"""Maps two rasters (a filter image and a trace mask) into one shared page-pixel frame so
their ink can be compared pixel for pixel (retained / gained ink).

Anchoring each raster independently (its own ink bbox -> its own affine warpAffine into the
page canvas) is not enough: both land within 1-2px of the correct absolute page position, but
the rasters differ in native resolution and rounding, so independent affines leave a few
tenths of a page pixel of relative jitter between them. That is invisible in each raster's
absolute placement yet costs several points of "retained" on dense, textured content.

So each raster is cropped to its own ink first (native-pixel nonzero bounds), then the cropped
region is resized into ONE shared target box computed once (`bw_px`, `bh_px`, `px0`, `py0`,
all in page pixels) and used for BOTH rasters; they land in the identical pixel frame by
construction. Which bbox anchors the shared box does not matter: `rb` (the routed SVG's own
bbox) and `bbox_mm` (response field) agree to <0.02mm, which is asserted below.

An isotropy assertion (the ink-bbox-px / target-mm ratio must match in x and y) is kept as a
safety check on top, computed once against the shared target rather than per raster.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


class AnisotropyError(RuntimeError):
    pass


class AnchorMismatchError(RuntimeError):
    pass


def parse_polylines_mm(svg_path: Path, px2mm: float):
    text = Path(svg_path).read_text(encoding="utf-8")
    polys = []
    for ps in re.findall(r'points="([^"]+)"', text):
        co = [tuple(map(float, t.split(','))) for t in ps.split() if ',' in t]
        if len(co) >= 2:
            polys.append([(x * px2mm, y * px2mm) for x, y in co])
    # The routed-SVG writer emits a two-point stroke as <line x1= x2= y1= y2=/>, which the
    # points= scan above never sees. They are appended AFTER the polylines so the polyline
    # order is unchanged; an SVG with no <line> returns only the polylines.
    for tag in re.findall(r'<line\b([^>]*)>', text):
        a = {k: float(v) for k, v in re.findall(r'\b(x1|x2|y1|y2)="([-+\d.eE]+)"', tag)}
        if len(a) == 4:
            polys.append([(a["x1"] * px2mm, a["y1"] * px2mm), (a["x2"] * px2mm, a["y2"] * px2mm)])
    return polys


def routed_bbox_mm(polys_mm):
    xs = [x for p in polys_mm for x, y in p]
    ys = [y for p in polys_mm for x, y in p]
    return min(xs), min(ys), max(xs), max(ys)


def assert_anchor_agrees(rb, bbox_mm_response, tol_mm=0.1, label=""):
    dev = max(abs(rb[0] - bbox_mm_response[0]), abs(rb[1] - bbox_mm_response[1]),
              abs(rb[2] - bbox_mm_response[2]), abs(rb[3] - bbox_mm_response[3]))
    if dev > tol_mm:
        raise AnchorMismatchError(
            f"{label}: routed-SVG bbox vs response bbox_mm differ by {dev:.4f}mm > {tol_mm}mm -- "
            f"R1(b)'s prelim/final-pass-ambiguity assumption may not hold here, check before trusting rb.")
    return dev


def crop_ink(bin_u8, thresh=127):
    ys, xs = np.nonzero(bin_u8 > thresh)
    if xs.size == 0:
        raise RuntimeError("raster has no ink -- cannot crop")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return bin_u8[y0:y1, x0:x1], (x0, y0, x1, y1)


def shared_target_box(bbox_mm_, ppm):
    x0, y0, x1, y1 = bbox_mm_
    bw_px = max(1, int(round((x1 - x0) * ppm)))
    bh_px = max(1, int(round((y1 - y0) * ppm)))
    px0, py0 = int(round(x0 * ppm)), int(round(y0 * ppm))
    return bw_px, bh_px, px0, py0


def place_shared(raw_u8, ink_bbox, target_box, canvas_w_px, canvas_h_px, thresh=127):
    """Crop to ink_bbox, resize into the SHARED target_box (bw_px,bh_px,px0,py0)
    -- same box for every raster being compared in one job, guaranteeing
    identical pixel-frame alignment. `raw_u8` may be pre-cropped already
    (pass ink_bbox=None) or the full frame (ink_bbox given)."""
    bw_px, bh_px, px0, py0 = target_box
    if ink_bbox is not None:
        x0, y0, x1, y1 = ink_bbox
        cropped = raw_u8[y0:y1, x0:x1]
    else:
        cropped = raw_u8
    resized = cv2.resize(cropped, (bw_px, bh_px), interpolation=cv2.INTER_LINEAR)
    thr = resized > thresh
    canvas = np.zeros((canvas_h_px, canvas_w_px), bool)
    cx0, cy0 = max(0, px0), max(0, py0)
    cx1, cy1 = min(canvas_w_px, px0 + bw_px), min(canvas_h_px, py0 + bh_px)
    if cx1 > cx0 and cy1 > cy0:
        canvas[cy0:cy1, cx0:cx1] = thr[cy0 - py0:cy1 - py0, cx0 - px0:cx1 - px0]
    return canvas


def isotropy_check(ink_bbox_px, target_bbox_mm, isotropy_tol_pct=0.5, label=""):
    ix0, iy0, ix1, iy1 = ink_bbox_px
    rx0, ry0, rx1, ry1 = target_bbox_mm
    ink_w_px, ink_h_px = (ix1 - ix0), (iy1 - iy0)
    w_mm, h_mm = max(rx1 - rx0, 1e-6), max(ry1 - ry0, 1e-6)
    px_x, px_y = ink_w_px / w_mm, ink_h_px / h_mm
    avg = (px_x + px_y) / 2.0
    dev_pct = abs(px_x - px_y) / avg * 100.0
    if dev_pct > isotropy_tol_pct:
        raise AnisotropyError(f"{label}: {dev_pct:.2f}% > {isotropy_tol_pct}% (x={px_x:.4f}, y={px_y:.4f})")
    return dev_pct, px_x, px_y
