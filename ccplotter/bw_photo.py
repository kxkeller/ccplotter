"""B/W photo front end: the fixed XDoG raster, as an explicit,
deterministic preprocessing step.

Provides the OPT-IN style ("bwphoto"): XDoG with the p02_e55 parameters (p=20,
eps=0.55) at shorter-side 512 by default, then the ordinary tracev3 engine with
CLAHE on and tone off.

xdog() and the sigma formula are the reference implementation of that fixed
configuration -- do not "improve" them: the point of this module is byte parity
with it (tests/test_bw_photo.py pins a hash of the raster). The pipeline is:

    decode as single-channel greyscale (cv2.IMREAD_GRAYSCALE)
    -> resize so the SHORTER side is 512 px (INTER_AREA, no direction guard)
    -> xdog(sigma = min(h, w) / 1024 * 0.8 * (1024 / 512), p=20, k=1.6,
            eps=0.55, phi=2.0)
    -> 8-bit greyscale PNG, handed to the unchanged tracev3 source loader.

Every parameter is a module constant, global across all images (no per-image
tuning). The intermediate raster is written to the job's debug output
(debug_xdog.png) so the abstraction step is visible.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# The fixed cell. phi=2 is the SOFT cell (p02_e55); the harder variants (p=40)
# are not used.
BW_XDOG_SIDE = 512   # the fixed cell; still the bw_xdog_raster() function default
BW_XDOG_P = 20.0
BW_XDOG_K = 1.6
BW_XDOG_EPS = 0.55
BW_XDOG_PHI = 2.0
BW_XDOG_SIGMA_PX = 0.8   # documentation only: the formula below yields this at side 512


def xdog(gray, sigma, p=20.0, k=1.6, eps=0.55, phi=2.0):
    # Fixed-parameter XDoG; keep byte-identical (see module docstring).
    g = gray.astype(np.float64) / 255.0
    g1 = cv2.GaussianBlur(g, (0, 0), sigma)
    g2 = cv2.GaussianBlur(g, (0, 0), sigma * k)
    u = g1 + p * (g1 - g2)
    out = np.where(u >= eps, 1.0, 1.0 + np.tanh(phi * (u - eps)))
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# Opt-in higher-detail sides, reachable only through the bwphoto-only request key
# `bw_xdog_side` (512 is the byte-parity cell above). The sigma formula below is NOT
# rescaled: it holds sigma at 0.8 px of the new raster, so its physical size shrinks
# by 512/side.
BW_XDOG_SIDES_ALLOWED = (512, 768, 1024)
BW_XDOG_NATIVE_CAP = 1024
# Default for bwphoto REQUESTS with no bw_xdog_side is "native". An explicit 512
# reproduces the byte-parity cell exactly.
BW_XDOG_DEFAULT_SIDE = "native"


def parse_bw_xdog_side(value):
    """Request value -> 512 | 768 | 1024 | "native" | "full"; ValueError for anything else.
    Absent / empty -> BW_XDOG_DEFAULT_SIDE ("native")."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return BW_XDOG_DEFAULT_SIDE
    if isinstance(value, bool):
        raise ValueError(f"bw_xdog_side must be one of 512, 768, 1024, native, full; got {value!r}")
    s = str(value).strip().lower()
    if s in ("native", "full"):
        return s
    try:
        f = float(s)
    except ValueError:
        raise ValueError(f"bw_xdog_side must be one of 512, 768, 1024, native, full; got {value!r}")
    if f in BW_XDOG_SIDES_ALLOWED:
        return int(f)
    raise ValueError(f"bw_xdog_side must be one of 512, 768, 1024, native, full; got {value!r}")


def source_short_side(path) -> int:
    """Shorter side in px of the decoded greyscale source."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"B/W photo: could not decode source image: {path}")
    return int(min(img.shape))


def resolve_bw_xdog_side(path, side=BW_XDOG_SIDE):
    """-> (side_used, source_short_side or None, interpolated).  "native" = min(1024, short side).
    interpolated = the working raster is UPSAMPLED from the source (side > source short side)."""
    if side == BW_XDOG_SIDE:
        return BW_XDOG_SIDE, None, None
    short = source_short_side(path)
    if side == "full":   # source's true short side: no cap, no upsampling
        used = short
    else:
        used = min(BW_XDOG_NATIVE_CAP, short) if side == "native" else int(side)
    return used, short, bool(used > short)


def load_gray_side(path, side: int = BW_XDOG_SIDE):
    """Greyscale decode + shorter-side resize, as m09 00_gen_rasters.py::load_gray.
    side may also be "native" (= min(1024, source short side))."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"B/W photo: could not decode source image: {path}")
    h, w = img.shape
    if side == "native":
        side = min(BW_XDOG_NATIVE_CAP, min(h, w))
    elif side == "full":   # true short side, no cap: identity-scale (INTER_AREA at s=1.0 is deterministic)
        side = min(h, w)
    s = side / min(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def bw_xdog_raster(path, side: int = BW_XDOG_SIDE):
    """The B/W XDoG raster for a source image file (uint8, H x W)."""
    gray = load_gray_side(path, side)
    if side in ("native", "full"):
        side = min(gray.shape)
    sigma = min(gray.shape) / 1024 * 0.8 * (1024 / side)
    return xdog(gray, sigma=sigma, p=BW_XDOG_P, k=BW_XDOG_K, eps=BW_XDOG_EPS, phi=BW_XDOG_PHI)


def prepare_bw_xdog_source(raw_path, work_dir: Path, debug_prefix: str = None,
                           side=BW_XDOG_SIDE) -> Path:
    """Write the B/W raster to a temp PNG in `work_dir` and return its path.

    The caller (render.py) feeds that path to the UNCHANGED `_load_gray`, exactly
    as it loads an uploaded XDoG PNG, then unlinks it.
    When `debug_prefix` is given the same PNG is also kept as
    `<debug_prefix>_xdog.png` (debug output, inspectable per job).
    """
    from uuid import uuid4
    out = bw_xdog_raster(raw_path) if side == BW_XDOG_SIDE else bw_xdog_raster(raw_path, side)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("B/W photo: failed to encode the XDoG raster.")
    if debug_prefix:
        buf.tofile(f"{debug_prefix}_xdog.png")
    tmp = Path(work_dir) / f"bwphoto_src_{uuid4().hex[:8]}.png"
    buf.tofile(str(tmp))
    return tmp
