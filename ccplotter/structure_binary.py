r"""Structure-aware binary (EXPERIMENTAL, opt-in) -- request key `bw_structure_binary`, B/W photo (bwphoto) mode only, default OFF.

Production port of mode B of the M11 offline harness (artifacts/structure-aware-binary-p3-2026-11-01/scripts/sab_lib3.py, mode_B;
pass-3 mode_B and parameters are identical to pass 2 -- artifacts/structure-binary-optin-2026-11-01/RESULT.md).
Decision: Kai 2026-11-01 (M11 pass 3 approved as information, D1956-D1963; mode B promoted to an experimental opt-in, mode C dropped).

WHAT IT DOES (two global steps, one fixed parameter set, nothing reads the image identity):
  1. Diffuse-texture removal: connected components of the production candidate ink K that are small (< 3 mm2), have weak edges
     (mean E < 0.30) and low local sharpness (mean < 0.35) are dropped -- blurry/diffuse texture goes, sharp edges stay.
  2. Structural contours: persistent long (>= 8 mm) coherent Marr-Hildreth contours of the SOURCE luminance are drawn as 0.30 mm lines
     where K has no ink within 2.5 mm.
The result replaces the raster handed to Trace V3's entry (its bilevel fast path thresholds at 128 with no blur), so the bwphoto
CLAHE/threshold ladder cannot undo the selection. The ladder is still run, but only to obtain K. All spatial scales are physical mm,
converted with the job's own mm-per-working-px (the same value Trace V3 derives).

Measured offline (M11 pass 3, 15-image benchmark): eagle -25% pen-downs, ink 98%, sharp thin recall 0.998, crack unchanged.
NOT a default. Existing outputs are byte-identical while the key is absent (the hook is a no-op without it).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

VERSION = "sab-3-modeB (ported from sab_lib3.mode_B, M11 pass 3)"

# The ONE fixed parameter set (values identical to sab_lib3.PARAMS; keys used by mode B only -- the budget/colour keys of modes C/D are not ported).
PARAMS = {
    "analysis_mm_per_px": 0.09,       # analysis raster pitch (mm)
    "sigma_f_mm": 0.12,               # finest DoG scale
    "n_scales": 5,                    # sigma_k = sigma_f * 2^k : 0.12 0.24 0.48 0.96 1.92 mm
    "g_edge_sat": 0.10,               # E saturates at scale-normalised gradient 0.10 (ABSOLUTE, not per-image)
    "sharp_win_mm": 2.0,              # local sharpness window sigma
    "coh_int_mm": 2.0,                # structure-tensor integration sigma
    "zc_grad_min": 0.035,             # |g_2| floor for a zero crossing to count
    "zc3_grad_min": 0.035,
    "persist_dilate_mm": 0.72,        # a sigma_2 crossing must have a sigma_3 crossing within this distance
    "line_min_len_mm": 8.0,           # connected contour length floor
    "line_width_mm": 0.30,            # drawn line width
    "line_add_dist_mm": 2.5,          # only add a contour where K has no ink within this distance
    "coh_min": 0.55,                  # contour pixels must sit in coherent gradient neighbourhoods
    "diffuse_area_mm2": 3.0,
    "diffuse_E_max": 0.30,
    "diffuse_sharp_max": 0.35,
    "min_component_mm2": 0.035,       # output hygiene = production min_mark_mm2
}


# ---------------------------------------------------------------- helpers
def _gauss(a, s_px):
    return cv2.GaussianBlur(a, (0, 0), float(s_px), borderType=cv2.BORDER_REFLECT)


def _grad_mag(a):
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT) / 8.0
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT) / 8.0
    return gx, gy, np.sqrt(gx * gx + gy * gy)


def _up(a, shape_wh, interp=cv2.INTER_LINEAR):
    W, H = shape_wh
    if (a.shape[1], a.shape[0]) == (W, H):
        return a
    return cv2.resize(a, (W, H), interpolation=interp)


def _disk(r_px):
    r = max(1, int(round(r_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# ---------------------------------------------------------------- analysis (analysis resolution)
def analyze(bgr8, work_wh, mm_per_px_work, P=PARAMS):
    """bgr8: the SOURCE colour image (any size). Returns float32 maps at ANALYSIS resolution (only those mode B reads)."""
    Wk, Hk = work_wh
    ma = float(P["analysis_mm_per_px"])
    Wa = max(8, int(round(Wk * mm_per_px_work / ma)))
    Ha = max(8, int(round(Hk * mm_per_px_work / ma)))
    interp = cv2.INTER_AREA if (bgr8.shape[1] > Wa) else cv2.INTER_CUBIC
    bgr = cv2.resize(bgr8, (Wa, Ha), interpolation=interp).astype(np.float32) / 255.0
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sig = [P["sigma_f_mm"] * (2 ** k) / ma for k in range(P["n_scales"])]
    G = [_gauss(L, s) for s in sig]
    gx_l, gy_l, mag_l, g = [], [], [], []
    for k in range(P["n_scales"]):
        gx, gy, m = _grad_mag(G[k])
        gx_l.append(gx); gy_l.append(gy); mag_l.append(m); g.append(m * sig[k])
    E = np.clip(np.max(np.stack([g[1], g[2], g[3]]), axis=0) / P["g_edge_sat"], 0, 1).astype(np.float32)
    # local sharpness: ratio of the finest to the coarse unnormalised gradient along edges (sharp step ~8, blurred edge ~1)
    r = mag_l[0] / (mag_l[3] + 1e-4)
    sh_pt = np.clip(np.log2(np.maximum(r, 1e-3)) / 3.0, 0, 1).astype(np.float32)
    w_edge = (g[3] >= 0.02).astype(np.float32) * E
    sw = P["sharp_win_mm"] / ma
    sharp = np.clip(_gauss(sh_pt * w_edge, sw) / (_gauss(w_edge, sw) + 1e-3), 0, 1).astype(np.float32)
    # coherence (structure tensor of sigma_2 gradients, integrated)
    ci = P["coh_int_mm"] / ma
    jxx = _gauss(gx_l[2] * gx_l[2], ci); jyy = _gauss(gy_l[2] * gy_l[2], ci); jxy = _gauss(gx_l[2] * gy_l[2], ci)
    kappa = (((jxx - jyy) ** 2 + 4 * jxy * jxy) / ((jxx + jyy) ** 2 + 1e-12)).astype(np.float32)
    return {"Wa": Wa, "Ha": Ha, "ma": ma, "sig": sig, "L": L, "G": G, "g": g, "E": E, "sharp": sharp, "kappa": kappa}


def _length_filter(z, min_len_px):
    """keep 8-connected components (after a 1-px closing) whose thin pixel count >= min_len_px."""
    zc = cv2.dilate(z.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(zc, connectivity=8)
    cnt = np.bincount(lab[z], minlength=n)
    keep = cnt >= min_len_px
    keep[0] = False
    return keep[lab] & z


def _zc_lines(M, P=PARAMS):
    """persistent long structural contours (thin 1-px bool at analysis res)."""
    ma, sig, G, g, kappa = M["ma"], M["sig"], M["G"], M["g"], M["kappa"]

    def zc(k, gmin):
        dog = G[k] - _gauss(M["L"], sig[k] * 1.6)
        z = np.zeros(dog.shape, bool)
        sx = (dog[:, :-1] * dog[:, 1:]) < 0; sy = (dog[:-1, :] * dog[1:, :]) < 0
        z[:, :-1] |= sx; z[:-1, :] |= sy
        return z & (g[k] >= gmin)
    z2 = zc(2, P["zc_grad_min"]); z3 = zc(3, P["zc3_grad_min"])
    pr = int(round(P["persist_dilate_mm"] / ma))
    z = z2 & (cv2.dilate(z3.astype(np.uint8), _disk(pr)) > 0) & (kappa >= P["coh_min"])
    return _length_filter(z, P["line_min_len_mm"] / ma)


def _thin_to_lines(thin_a, work_wh, mm_per_px_work, P=PARAMS):
    """thin analysis-res contour -> pen-width line mask at working res."""
    if not thin_a.any():
        return np.zeros((work_wh[1], work_wh[0]), bool)
    t = cv2.resize(thin_a.astype(np.uint8), work_wh, interpolation=cv2.INTER_NEAREST)
    r = max(1.0, P["line_width_mm"] / mm_per_px_work / 2.0)
    return cv2.dilate(t, _disk(r)) > 0


def _remove_small(mask, min_area_px):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area_px
    return keep[lab]


def mode_b(K, M, mm_per_px_work, P=PARAMS):
    """K: bool ink mask at working res (production candidate ink). Returns (B bool, counts dict)."""
    H, W = K.shape
    work_wh = (W, H)
    # (1) diffuse-texture removal: small K components with weak edges and low local sharpness
    n, lab, st, _ = cv2.connectedComponentsWithStats(K.astype(np.uint8), connectivity=8)
    area_mm2 = st[:, cv2.CC_STAT_AREA] * mm_per_px_work ** 2
    Eup = _up(M["E"], work_wh); Shup = _up(M["sharp"], work_wh)
    # per-component mean of E and sharpness over the component's own pixels dilated 2 px (max-label dilation: deterministic)
    lab_d = cv2.dilate(lab.astype(np.float32), _disk(2)).astype(np.int32)
    sel = lab_d > 0
    cnt = np.bincount(lab_d[sel], minlength=n).astype(np.float64)
    meanE = np.bincount(lab_d[sel], weights=Eup[sel], minlength=n) / np.maximum(cnt, 1)
    meanSh = np.bincount(lab_d[sel], weights=Shup[sel], minlength=n) / np.maximum(cnt, 1)
    drop = (area_mm2 < P["diffuse_area_mm2"]) & (meanE < P["diffuse_E_max"]) & (meanSh < P["diffuse_sharp_max"])
    drop[0] = False
    Kb = K & ~drop[lab]
    # (2) persistent long structural contours where K has no ink nearby
    thin = _zc_lines(M, P)
    lines = _thin_to_lines(thin, work_wh, mm_per_px_work, P)
    dist_to_K = cv2.distanceTransform((~Kb).astype(np.uint8), cv2.DIST_L2, 5)
    add = lines & (dist_to_K > P["line_add_dist_mm"] / mm_per_px_work)
    B = Kb | add
    return B, {"components_dropped": int(drop.sum()), "ink_px_dropped": int((K & ~Kb).sum()), "ink_px_added": int(add.sum())}


def _hygiene(R, mm_per_px_work, P=PARAMS):
    return _remove_small(R, max(2, int(round(P["min_component_mm2"] / mm_per_px_work ** 2))))


def structure_binary_raster(K, bgr_src, mm_per_px_work, P=PARAMS):
    """K: bool candidate ink (working res); returns (R uint8 0/255 with ink = 0, counts dict)."""
    H, W = K.shape
    M = analyze(bgr_src, (W, H), mm_per_px_work, P)
    B, counts = mode_b(K, M, mm_per_px_work, P)
    R = _hygiene(B, mm_per_px_work, P)
    return np.where(R, 0, 255).astype(np.uint8), counts


# ---------------------------------------------------------------- production glue
def mm_per_px_of(img, opts):
    """The working-grid mm/px exactly as trace_v3._trace_svg derives it (fill_px floor 1.2, _mm_per_px_override)."""
    H, W = img.shape[:2]
    fill_px = opts.get("fill_px")
    if fill_px is None:
        pw = float(opts.get("paper_w_mm", 180.0)); mg = float(opts.get("margin_mm", 10.0))
        fill_px = float(opts.get("fill_mm", 0.15)) / max(1e-6, (pw - 2 * mg) / W)
    fill_px = max(1.2, float(fill_px))
    ov = opts.get("_mm_per_px_override")
    return float(ov) if ov is not None else float(opts.get("fill_mm", 0.15)) / fill_px


def load_source_bgr(raw_path, crop=None):
    """The SOURCE colour image the analysis reads (same file _load_gray reads; cv2 honours EXIF orientation as _load_gray does).
    Files > 20 MB are thumbnailed by PIL to 5200 px (the analysis raster is ~2600 px wide on a 237 mm page, so this loses nothing)."""
    from PIL import Image
    raw_path = Path(raw_path)
    if raw_path.stat().st_size > 20 * 1024 * 1024:
        with Image.open(raw_path) as pil_img:
            pil_img.thumbnail((5200, 5200))
            bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = cv2.cvtColor(np.array(Image.open(raw_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    if crop and len(crop) == 4:     # same fractional crop _load_gray applies
        h, w = bgr.shape[:2]
        ix0, iy0, ix1, iy1 = max(0, int(crop[0] * w)), max(0, int(crop[1] * h)), min(w, int(crop[2] * w)), min(h, int(crop[3] * h))
        if ix1 > ix0 and iy1 > iy0:
            bgr = bgr[iy0:iy1, ix0:ix1]
    return bgr


def apply(img, raw_path, opts, binarize):
    """Hook for render.py's tracev3 branch. img: the production working-res gray raster; binarize: trace_v3's _binarize_mask.
    Returns (R, audit): R is the final binary (0/255 uint8, same shape as img) to hand to Trace V3 in place of img."""
    K = binarize(img, opts)
    m = mm_per_px_of(img, opts)
    bgr = load_source_bgr(raw_path, opts.get("crop"))
    R, counts = structure_binary_raster(K, bgr, m)
    assert R.shape == img.shape[:2] and R.dtype == np.uint8
    audit = {"version": VERSION, "params": dict(PARAMS), "mm_per_px": m, "shape": [int(img.shape[0]), int(img.shape[1])],
             "K_ink_frac": float(K.mean()), "R_ink_frac": float((R == 0).mean()),
             "R_sha16": hashlib.sha256(np.ascontiguousarray(R).tobytes()).hexdigest()[:16], **counts}
    return R, audit
