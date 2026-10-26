"""ccplotter/twotone.py -- the fixed two-tone recipe (v16 default), an opt-in generator.

Recipe: the v16 flags inside_join + closing + solid_overlap are ON and every other
experiment flag is absent. Photo in -> two mm page-coordinate polyline layers
(blue pass first, black pass last). Only this default path exists; the v14 variant
(the three v16 flags off) is not implemented. `blue_preset` accepts only "LIGHT".

Reused app functions, unchanged: ccplotter.bw_photo.load_gray_side, ccplotter.mask._skeleton_to_polylines,
ccplotter.trace_v3._hatch_fill. The black LINES come from a Canny path (CANNY_PCT) plus a texture gate,
NOT the app's Plot-Ready / tracev3 line engine. The blue ladder has its own row walk (per-line thresholds).

Numerics are kept identical to the reference recipe so the stroke hash matches the
reference hashes pinned in tests/test_twotone.py. One quirk is kept for that parity and
noted in-line (shadow-hatch join tests the level of the pixel at the run end). All widths/colours here are UNMEASURED; nothing is written to gcode by
this module -- layers go through ccplotter.twopen (POST /api/twotone).
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects

from ccplotter.bw_photo import load_gray_side
from ccplotter.mask import _skeleton_to_polylines
from ccplotter.trace_v3 import _hatch_fill

# ------------------------------------------------------------------ constants (fixed)
C = dict(
    FIT=0.9,
    SIMPLIFY=dict(d=9, sigma_color=40, sigma_space=40), S1024=1024,
    PX_PER_MM=25, BLUR_MM=1.0, MIN_REGION_MM2=6.0,
    BLACK_FRAC=0.04, SHADOW_PITCH_PX=6, SHADOW_ANGLE=-45.0,
    CANNY_PCT=85, CANNY_LOW_FRAC=0.4, SKEL_PRUNE_MM=1.0, JOIN_MM=0.4, JOIN_ANG=35.0, MIN_EDGE_MM=2.5, SIMPLIFY_MM=0.05,
    GATE_WIN_MM=3.0, GATE_PCT=85, GATE_LEN_MM=6.0,
    GAMMA=2.2, T_LEVELS=[0.25, 2 / 6, 3 / 6, 4 / 6, 5 / 6], BAND_W=1 / 6, PHI=0.6180339887, F_PX=3, G0=(1 / 6) / 10.0,
    MIN_STROKE_MM=2.0, CONNECT_PITCHES=1.5,
)
P = C["PX_PER_MM"]
V16 = dict(INSIDE_MARGIN_MM=0.5, OVERLAP=0.9, CLOSE_MM=0.3)
STEPJ = {0: 8, 1: 8, 2: 4, 4: 2, 3: 1}      # row-index spacing between consecutive lines of the same level
DEFAULT_PAGE_MM = (257.0, 182.0)             # JIS B5 Kokuyo landscape
DEFAULT_MARGIN_MM = 10.0
PRESETS = ("LIGHT",)
BLUE_COLOR, BLACK_COLOR = "#1f4fd8", "#111111"


def live_area(page=DEFAULT_PAGE_MM, margin=DEFAULT_MARGIN_MM):
    """(live_w, live_h, (cx, cy)): the page minus the margin all round (the app's placement owns any hole strip)."""
    pw, ph = float(page[0]), float(page[1])
    x0, y0, x1, y1 = margin, margin, pw - margin, ph - margin
    return (x1 - x0, y1 - y0), ((x0 + x1) / 2, (y0 + y1) / 2)


# ------------------------------------------------------------------ helpers
def plen(p):
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def order(strokes):
    n = len(strokes)
    if n == 0:
        return []
    st = np.array([s[0] for s in strokes]); en = np.array([s[-1] for s in strokes])
    alive = np.ones(n, bool); alive[0] = False
    res = [strokes[0]]; cur = en[0]
    for _ in range(n - 1):
        idx = np.flatnonzero(alive)
        ds = ((st[idx] - cur) ** 2).sum(1); de = ((en[idx] - cur) ** 2).sum(1)
        a, b = ds.argmin(), de.argmin()
        if ds[a] <= de[b]:
            k = idx[a]; res.append(strokes[k]); cur = en[k]
        else:
            k = idx[b]; res.append(strokes[k][::-1]); cur = st[k]
        alive[k] = False
    return res


def clean(mask):
    a = int(C["MIN_REGION_MM2"] * P ** 2)
    return remove_small_holes(remove_small_objects(mask, a), a)


def merge_collinear(runs, tol=2.5):
    runs = [r for r in runs if np.hypot(*(r[-1] - r[0])) > 1e-6]
    if not runs:
        return []
    L = max(runs, key=lambda r: np.hypot(*(r[-1] - r[0])))
    d = (L[-1] - L[0]); d = d / np.hypot(*d)
    nrm = np.array([-d[1], d[0]])
    rows = {}
    for r in runs:
        rows.setdefault(round(float(r[0] @ nrm) * 2), []).append(r)
    out = []
    for rs in rows.values():
        rs = [r if (r[0] @ d) <= (r[-1] @ d) else r[::-1] for r in rs]
        rs.sort(key=lambda r: r[0] @ d)
        cur = rs[0]
        for r in rs[1:]:
            if (r[0] @ d) - (cur[-1] @ d) <= tol:
                if (r[-1] @ d) > (cur[-1] @ d):
                    cur = np.array([cur[0], r[-1]])
            else:
                out.append(cur); cur = r
        out.append(cur)
    return out


def serpentine_mask(runs, mask, pitch_mm):
    """Join hatch run ends only if the connector stays inside `mask` (1 px tolerance) and is <= 1.5 x pitch."""
    n = len(runs)
    if n == 0:
        return []
    ends = np.vstack([np.array([r[0], r[-1]]) for r in runs])
    tree = cKDTree(ends)
    used = np.zeros(n, bool)
    order_idx = sorted(range(n), key=lambda i: (round(runs[i][0][1]), runs[i][0][0]))
    H, W = mask.shape
    Z = mask.astype(np.uint8)
    lvl = {z: cv2.dilate((Z == z).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 for z in (0, 1)}
    chains = []
    for s in order_idx:
        if used[s]:
            continue
        used[s] = True
        chain = [runs[s][0], runs[s][-1]]
        cur = runs[s][-1]
        while True:
            # PARITY QUIRK -- kept ON PURPOSE, do not "fix" without regenerating the reference hashes:
            # the reference recipe tests the level of the pixel AT THE RUN END (zi). A run end that lies just outside the
            # mask therefore tests the OUTSIDE level (0) and may join across it. Harmless in practice (the 1 px
            # dilation of both levels keeps the connector short) and cleaning it changes the stroke hash, which
            # tests/test_twotone.py pins against the reference hashes. Cleaning it is a separate,
            # deliberate change with its own re-baselined hash.
            zi = int(Z[min(H - 1, max(0, int(cur[1]))), min(W - 1, max(0, int(cur[0])))])
            rad = C["CONNECT_PITCHES"] * pitch_mm * P
            cand = [c for c in tree.query_ball_point(cur, rad) if not used[c // 2]]
            best = None
            for c in sorted(cand, key=lambda c: np.hypot(*(ends[c] - cur))):
                a, b = cur, ends[c]
                m = int(np.hypot(*(b - a))) + 2
                xs = np.linspace(a[0], b[0], m).astype(int).clip(0, W - 1)
                ys = np.linspace(a[1], b[1], m).astype(int).clip(0, H - 1)
                if lvl[zi][ys, xs].all():
                    best = c; break
            if best is None:
                break
            i = best // 2
            used[i] = True
            r = runs[i] if best % 2 == 0 else runs[i][::-1]
            chain += [r[0], r[-1]]
            cur = r[-1]
        chains.append(np.array(chain))
    return chains


def _tan(a, end):
    v = (a[0] - a[min(3, len(a) - 1)]) if end == 0 else (a[-1] - a[-1 - min(3, len(a) - 1)])
    n = np.hypot(*v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def join_ends(polys, tol, maxang):
    polys = [np.asarray(p, float) for p in polys if len(p) >= 2]
    for _ in range(4):
        ends = []
        for i, p in enumerate(polys):
            ends.append((i, 0, p[0], _tan(p, 0))); ends.append((i, 1, p[-1], _tan(p, 1)))
        pts = np.array([e[2] for e in ends])
        tree = cKDTree(pts)
        pairs = sorted((np.hypot(*(pts[a] - pts[b])), a, b) for a, b in tree.query_pairs(tol))
        used, links = set(), []
        for d, a, b in pairs:
            ia, ib = ends[a], ends[b]
            if ia[0] == ib[0] or a in used or b in used:
                continue
            ang = np.degrees(np.arccos(np.clip(np.dot(ia[3], -ib[3]), -1, 1)))
            if ang <= maxang:
                links.append((a, b)); used.add(a); used.add(b)
        if not links:
            break
        nxt = {}
        for a, b in links:
            nxt[a] = b; nxt[b] = a
        seen = set(); out = []
        for i in range(len(polys)):
            if i in seen:
                continue
            cur, side = i, 0
            guard = 0
            while (2 * cur + side) in nxt and guard < 10000:
                o = nxt[2 * cur + side]
                cur, side = o // 2, 1 - (o % 2)
                guard += 1
                if cur == i:
                    break
            chain = []
            c, s = cur, side
            visited_local = 0
            while True:
                p = polys[c]
                p = p if s == 0 else p[::-1]
                chain.append(p); seen.add(c)
                exit_end = 2 * c + (1 if s == 0 else 0)
                if exit_end in nxt and visited_local < 10000:
                    o = nxt[exit_end]
                    c, s = o // 2, o % 2
                    if c in seen:
                        break
                    visited_local += 1
                else:
                    break
            out.append(np.vstack(chain))
        polys = out
    return polys


# ------------------------------------------------------------------ pipeline
def place(w_px, h_px, live, centre):
    sc = min(C["FIT"] * live[0] / w_px, C["FIT"] * live[1] / h_px)
    W, H = w_px * sc, h_px * sc
    cx, cy = centre
    return W, H, (cx - W / 2, cy - H / 2)


def prepare(path, page, margin):
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f"cannot read image {path}")
    h, w = raw.shape[:2]
    live, centre = live_area(page, margin)
    W, H, origin = place(w, h, live, centre)
    g1024 = load_gray_side(path, C["S1024"])
    # weak bilateral presimplify; no CLAHE
    s = cv2.bilateralFilter(g1024, C["SIMPLIFY"]["d"], C["SIMPLIFY"]["sigma_color"], C["SIMPLIFY"]["sigma_space"])
    Wr, Hr = int(round(W * P)), int(round(H * P))
    g = cv2.GaussianBlur(cv2.resize(s, (Wr, Hr), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), C["BLUR_MM"] * P)
    p2 = float(np.percentile(g, 2))
    TB = p2 + C["BLACK_FRAC"] * (255 - p2)
    return dict(W=W, H=H, origin=origin, simp=s, g=g, TB=TB, p2=p2)


def shadow_layer(ctx):
    """Black shadow hatch: darkest band, closing 0.3 mm, -45 deg hatch at 6 px (0.24 mm), serpentine join inside the region."""
    m = ctx["g"] < ctx["TB"]
    r = int(round(V16["CLOSE_MM"] * P))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    m = clean(m)
    if not m.any():
        return []
    runs = [np.array(r_) for r_ in _hatch_fill(m.astype(np.uint8), C["SHADOW_PITCH_PX"], angle_deg=C["SHADOW_ANGLE"], min_run_px=1)]
    keep = [r_ for r_ in merge_collinear(runs) if np.hypot(*(r_[-1] - r_[0])) / P >= C["MIN_STROKE_MM"]]
    chains = serpentine_mask(keep, m, C["SHADOW_PITCH_PX"] / P)
    return order([c / P + np.array(ctx["origin"]) for c in chains])


def black_lines(ctx):
    """PROTOTYPE Canny black lines (P85 high, 0.4 low) + skeleton + texture gate. Not the app's line engine."""
    s = ctx["simp"]
    f = s.astype(np.float32)
    mag = np.hypot(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3))
    hi = float(np.percentile(mag, C["CANNY_PCT"])); lo = C["CANNY_LOW_FRAC"] * hi
    e = cv2.Canny(s, lo, hi, L2gradient=True)
    mmpp = ctx["W"] / s.shape[1]
    w = int(round(C["GATE_WIN_MM"] / mmpp)) | 1
    dens = cv2.boxFilter((e > 0).astype(np.float32), -1, (w, w))
    sk = skeletonize(e > 0)
    polys = _skeleton_to_polylines(sk, prune_len=int(round(C["SKEL_PRUNE_MM"] / mmpp)))
    arrs = join_ends([np.array(p, float) * mmpp for p in polys if len(p) >= 2], C["JOIN_MM"], C["JOIN_ANG"])
    out = []
    for a in arrs:
        if plen(a) < C["MIN_EDGE_MM"]:
            continue
        b = np.array(LineString(a).simplify(C["SIMPLIFY_MM"]).coords)
        if len(b) >= 2 and plen(b) >= C["MIN_EDGE_MM"]:
            out.append(b + np.array(ctx["origin"]))
    lines = order(out)
    thr = float(np.percentile(dens, C["GATE_PCT"]))
    keep, dropped = [], []
    for s_ in lines:
        if plen(s_) < C["GATE_LEN_MM"]:
            q = np.floor((s_ - np.array(ctx["origin"])) / (ctx["W"] / dens.shape[1])).astype(int)
            q[:, 0] = q[:, 0].clip(0, dens.shape[1] - 1); q[:, 1] = q[:, 1].clip(0, dens.shape[0] - 1)
            if dens[q[:, 1], q[:, 0]].mean() > thr:
                dropped.append(s_)
                continue
        keep.append(s_)
    # Per-job readouts of the adaptive quantities and the texture-gate artefacts (read-only; they do not
    # change what is kept, so the stroke hash is unaffected).
    kept_mm, dropped_mm = sum(plen(x) for x in keep), sum(plen(x) for x in dropped)
    ctx["gate_map"] = dens > thr
    ctx["gate_dropped"] = dropped
    ctx["gate"] = {"dropped_strokes": len(dropped), "dropped_ink_mm": round(dropped_mm, 3),
                   "dropped_ink_fraction": round(dropped_mm / (kept_mm + dropped_mm), 6) if (kept_mm + dropped_mm) > 0 else 0.0}
    ctx["readouts"] = {"p2": round(float(ctx["p2"]), 4), "shadow_band_threshold": round(float(ctx["TB"]), 4),
                       "canny_high": round(hi, 4), "canny_low": round(lo, 4), "gate_density_threshold": round(thr, 6),
                       "gate_window_mm": C["GATE_WIN_MM"], "gate_percentile": C["GATE_PCT"], "mm_per_px": round(float(mmpp), 6)}
    return keep


def _rows(tmap, wmap, angle, step, level_of):
    H, W = tmap.shape
    diag = int(np.ceil(np.hypot(H, W))) + 4
    cx, cy = W / 2.0, H / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += diag / 2.0 - cx; M[1, 2] += diag / 2.0 - cy
    rot = cv2.warpAffine(tmap.astype(np.float32), M, (diag, diag), flags=cv2.INTER_LINEAR, borderValue=0.0)
    wrot = cv2.warpAffine(wmap.astype(np.float32), M, (diag, diag), flags=cv2.INTER_LINEAR, borderValue=1.0)
    Minv = cv2.invertAffineTransform(M)
    out = []
    TL, BW, PHI = C["T_LEVELS"], C["BAND_W"], C["PHI"]
    stepf = step * V16["OVERLAP"]   # solid_overlap: row pitch 0.9 x width so adjacent lines overlap instead of just touching
    nrows = int((diag - 2) / stepf)
    for j in range(nrows):
        yf = j * stepf
        y = int(yf); fr = yf - y
        lv = level_of(j)
        if lv is None:
            continue
        delta = ((j * PHI) % 1.0 - 0.5) * BW
        thr = TL[lv] + delta
        rv = rot[y] * (1 - fr) + rot[y + 1] * fr if fr > 0 else rot[y]
        wv = wrot[y] * (1 - fr) + wrot[y + 1] * fr if fr > 0 else wrot[y]
        row = rv >= (TL[lv] + delta * wv)
        if not row.any():
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        idx = np.flatnonzero(d)
        starts, ends = idx[0::2], idx[1::2]
        merged = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            if merged and (s - merged[-1][1]) <= 2:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        for s, e in merged:
            if (e - s) < 2:
                continue
            p0 = np.array([Minv[0, 0] * s + Minv[0, 1] * yf + Minv[0, 2], Minv[1, 0] * s + Minv[1, 1] * yf + Minv[1, 2]])
            p1 = np.array([Minv[0, 0] * (e - 1) + Minv[0, 1] * yf + Minv[0, 2], Minv[1, 0] * (e - 1) + Minv[1, 1] * yf + Minv[1, 2]])
            out.append(dict(run=np.array([p0, p1]), thr=thr, j=j, lv=lv))
    return out


def _chain(items, tmap):
    """inside_join: connect only to the NEXT row of the SAME level (same direction), only if the whole connector
    stays strictly inside the band (tone eroded 0.5 mm still >= both thresholds), radius <= 1.5 x that spacing."""
    r_in = int(round(V16["INSIDE_MARGIN_MM"] * P))
    tmin = cv2.erode(tmap.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_in + 1, 2 * r_in + 1)))
    n = len(items)
    if n == 0:
        return []
    runs = [it["run"] for it in items]
    ends = np.vstack([np.array([r[0], r[-1]]) for r in runs])
    tree = cKDTree(ends)
    used = np.zeros(n, bool)
    H, W = tmap.shape
    order_idx = sorted(range(n), key=lambda i: (round(runs[i][0][1]), runs[i][0][0]))
    chains = []
    for s in order_idx:
        if used[s]:
            continue
        used[s] = True
        chain = [runs[s][0], runs[s][-1]]
        cur = runs[s][-1]; it = items[s]; sign = 0
        while True:
            rad = C["CONNECT_PITCHES"] * (STEPJ[it["lv"]] * C["F_PX"] * (2 if it["lv"] == 3 else 1))
            cand = [c for c in tree.query_ball_point(cur, rad) if not used[c // 2]]
            cand = [c for c in cand if items[c // 2]["lv"] == it["lv"] and abs(items[c // 2]["j"] - it["j"]) == STEPJ[it["lv"]]
                    and (sign == 0 or np.sign(items[c // 2]["j"] - it["j"]) == sign)]
            best = None
            for c in sorted(cand, key=lambda c: np.hypot(*(ends[c] - cur))):
                a, b = cur, ends[c]
                m = int(np.hypot(*(b - a))) + 2
                xs = np.linspace(a[0], b[0], m).astype(int).clip(0, W - 1)
                ys = np.linspace(a[1], b[1], m).astype(int).clip(0, H - 1)
                if (tmin[ys, xs] >= max(it["thr"], items[c // 2]["thr"])).all():
                    best = c; break
            if best is None:
                break
            i = best // 2
            used[i] = True
            sign = int(np.sign(items[i]["j"] - it["j"]))
            r = runs[i] if best % 2 == 0 else runs[i][::-1]
            chain += [r[0], r[-1]]
            cur = r[-1]; it = items[i]
        chains.append(np.array(chain))
    return chains


def blue_layer(ctx):
    g = ctx["g"]
    d = 1.0 - g / 255.0
    t = d ** C["GAMMA"]
    g2 = 255.0 * (1.0 - t)
    tmap = 1.0 - g2 / 255.0                                   # = t (same float path as the reference recipe)
    r = int(round(V16["CLOSE_MM"] * P))
    tmap = cv2.morphologyEx(tmap.astype(np.float32), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    gx = cv2.Sobel(tmap.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8.0 * P
    gy = cv2.Sobel(tmap.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8.0 * P
    wmap = np.clip(np.hypot(gx, gy) / C["G0"], 0.0, 1.0)
    F = C["F_PX"]
    fam45 = _rows(tmap, wmap, 45.0, F, lambda j: 0 if j % 8 == 0 else 1 if j % 4 == 0 else 2 if j % 2 == 0 else 4)
    cross = _rows(tmap, wmap, -45.0, 2 * F, lambda j: 3)
    chains = []
    for fam in (fam45, cross):
        keep = [it for it in fam if np.hypot(*(it["run"][-1] - it["run"][0])) / P >= C["MIN_STROKE_MM"]]
        chains += _chain(keep, tmap)
    return order([c / P + np.array(ctx["origin"]) for c in chains])


# ------------------------------------------------------------------ public API
def stroke_hash(blue, lines, shadow):
    """Same hash as the reference recipe: blue | black lines | shadow hatch, rounded to 1e-4 mm."""
    h = hashlib.sha256()
    for grp in (blue, lines, shadow):
        for s in grp:
            h.update(np.round(s, 4).astype(np.float64).tobytes())
        h.update(b"|")
    return h.hexdigest()


def generate_layers(photo, page=DEFAULT_PAGE_MM, margin=DEFAULT_MARGIN_MM, blue_preset="LIGHT"):
    """photo path -> {"blue": [Nx2 mm arrays], "black": lines + shadow, "lines", "shadow", "hash", "page", "placed_mm", "origin_mm"}.
    Blue pass first, black pass last (no knockout: blue underlaps black)."""
    if str(blue_preset).upper() not in PRESETS:
        raise ValueError(f"blue_preset must be one of {PRESETS} (the frozen recipe has only LIGHT)")
    ctx = prepare(photo, page, margin)
    shadow = shadow_layer(ctx)
    lines = black_lines(ctx)
    blue = blue_layer(ctx)
    return {"blue": blue, "black": lines + shadow, "lines": lines, "shadow": shadow,
            "texture_gate": ctx["gate"], "readouts": ctx["readouts"], "gate_map": ctx["gate_map"],
            "gate_dropped": ctx["gate_dropped"],
            "hash": stroke_hash(blue, lines, shadow), "page": (float(page[0]), float(page[1])),
            "placed_mm": (ctx["W"], ctx["H"]), "origin_mm": ctx["origin"], "blue_preset": "LIGHT"}


def to_svg(strokes, page=DEFAULT_PAGE_MM, label="layer", stroke_hex="#000000"):
    """mm page-coordinate SVG (width/height in mm, viewBox 1 user unit = 1 mm), ASCII."""
    pw, ph = float(page[0]), float(page[1])
    rows = ['<polyline points="' + " ".join(f"{x:.3f},{y:.3f}" for x, y in s) + '"/>' for s in strokes if len(s) >= 2]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{pw:g}mm" height="{ph:g}mm" viewBox="0 0 {pw:g} {ph:g}">'
            f'<g id="{label}" fill="none" stroke="{stroke_hex}" stroke-width="0.12" stroke-linecap="round" '
            f'stroke-linejoin="round">' + "".join(rows) + "</g></svg>")
