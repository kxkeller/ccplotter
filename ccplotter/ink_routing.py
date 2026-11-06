"""Opt-in "ink-routed connectors" (request key `ink_routed_connectors`, default OFF; no UI).

Prototype record: artifacts/ink-routed-connectors-2026-10-28 (rounds 1) and -r2-2026-10-29 (arm R2a, the registered design). Reviewer D1891-D1904.

What it does. AFTER stay-down and after the final `linesort` (so on the order that is actually emitted), for each pen-up hop between
consecutive strokes of a layer it looks for the shortest pen-down path from the end of one stroke to the start of the next whose CENTRELINE
lies wholly in the stay-down predicate region of the layer's own trace mask (bilinear dt >= 0.5*fill_px - dt_tol_px, the stage_e_mask
predicate generalised from straight segments to paths). The path is used only if its length is <= min(4 x |hop|, 3.0 mm) (ONE global cap);
otherwise the lift stays. Adjacent strokes are then fused into one pen-down stroke.

Fixed rules (PREREG round 2, D1-D4):
  * raster 0.02 mm/px (P = 50), global grid anchored at the page origin (pixel centre (i + 0.5)/P mm); 8-connected geometric Dijkstra in a
    local window (the ellipse bbox of the allowed detour);
  * a hop shorter than 0.02 mm fuses only if BOTH ends lie in the predicate region;
  * an end connects to a pixel centre of the region within 1 px (0.02 mm) of the end (the 2x2 nearest centres), else the hop stays a lift;
  * the emitted connector = Douglas-Peucker(0.01 mm) of [exact start, pixel-centre path, exact end] (two-sided deviation <= 0.01 mm);
  * a layer without a usable mask sidecar / failing the placement self-check is left untouched and named (fail closed, like stay-down).
Reordering is NOT done (arm R only). Two-pen is not touched.
"""
import math

import cv2
import numpy as np
from skimage.graph import MCP_Geometric

from ccplotter.config import PX2MM

P = 50                      # px per mm of the routing raster
CAP_RATIO = 4.0
CAP_ABS_MM = 3.0
DP_EPS_MM = 0.01
ZERO_FUSE_MM = 0.02
SNAP_PX = 1.0


class PredicateRaster:
    """Allowed-centreline raster from a stage_e_mask.LayerMaskCheck, evaluated lazily per window.
    window(c0, c1, r0, r1) -> bool array [r0:r1, c0:c1] on the global grid (pixel centres (i + 0.5)/P mm, page mm)."""

    def __init__(self, chk):
        from ccplotter.stitch import _dt_at
        self.chk = chk; self._dt_at = _dt_at
        self.thr = chk.thr - chk.tol

    def window(self, c0, c1, r0, r1):
        if c1 <= c0 or r1 <= r0:
            return np.zeros((max(0, r1 - r0), max(0, c1 - c0)), bool)
        xs = (np.arange(c0, c1) + 0.5) / P; ys = (np.arange(r0, r1) + 0.5) / P
        gx, gy = np.meshgrid(xs, ys)
        z = (gx.ravel() + 1j * gy.ravel()) / PX2MM                 # page px (vpype units)
        tr = self.chk.to_trace(z)
        H, W = self.chk.mask.shape
        inside = (tr[:, 0] >= -1) & (tr[:, 0] <= W) & (tr[:, 1] >= -1) & (tr[:, 1] <= H)
        out = np.zeros(len(z), bool)
        if inside.any():
            out[inside] = self._dt_at(self.chk.dt, tr[inside]) >= self.thr
        return out.reshape(r1 - r0, c1 - c0)


class ArrayRaster:
    """Provider over a prebuilt bool raster whose pixel (0,0) centre is at ((ox + 0.5)/P, (oy + 0.5)/P) mm (ox, oy integer pixel offsets)."""

    def __init__(self, A, ox, oy):
        self.A = A; self.ox = int(ox); self.oy = int(oy)

    def window(self, c0, c1, r0, r1):
        out = np.zeros((max(0, r1 - r0), max(0, c1 - c0)), bool)
        a0, a1 = max(c0 - self.ox, 0), min(c1 - self.ox, self.A.shape[1]); b0, b1 = max(r0 - self.oy, 0), min(r1 - self.oy, self.A.shape[0])
        if a1 > a0 and b1 > b0:
            out[b0 + self.oy - r0:b1 + self.oy - r0, a0 + self.ox - c0:a1 + self.ox - c0] = self.A[b0:b1, a0:a1]
        return out


def _end_px(win, pt, c0, r0):
    x = pt[0] * P - c0; y = pt[1] * P - r0
    best = None
    for iy in (int(math.floor(y - 0.5)), int(math.floor(y - 0.5)) + 1):
        for ix in (int(math.floor(x - 0.5)), int(math.floor(x - 0.5)) + 1):
            if 0 <= iy < win.shape[0] and 0 <= ix < win.shape[1] and win[iy, ix]:
                dd = math.hypot(ix + 0.5 - x, iy + 0.5 - y)
                if dd <= SNAP_PX and (best is None or dd < best[0]):
                    best = (dd, (iy, ix))
    return best[1] if best else None


def _chain_len(c):
    return float(np.hypot(*np.diff(c, axis=0).T).sum()) if len(c) > 1 else 0.0


def route_hop(a, b, provider, stats=None):
    """a, b: page mm (x, y). Returns (polyline Nx2 incl. both ends, length) or None (keep the lift)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = float(np.hypot(*(b - a))); Lmax = min(CAP_RATIO * d, CAP_ABS_MM)
    if d < ZERO_FUSE_MM:
        for q in (a, b):
            ix, iy = int(math.floor(q[0] * P)), int(math.floor(q[1] * P))
            if not provider.window(ix, ix + 1, iy, iy + 1)[0, 0]:
                return None
        return (np.array([a, b]) if d > 0 else np.array([a])), d
    if d > CAP_ABS_MM:
        return None
    pad = (Lmax - d) / 2 + 0.1
    c0 = int(math.floor((min(a[0], b[0]) - pad) * P)); c1 = int(math.ceil((max(a[0], b[0]) + pad) * P))
    r0 = int(math.floor((min(a[1], b[1]) - pad) * P)); r1 = int(math.ceil((max(a[1], b[1]) + pad) * P))
    win = provider.window(c0, c1, r0, r1)
    if win.size == 0 or not win.any():
        return None
    sa = _end_px(win, a, c0, r0); sb = _end_px(win, b, c0, r0)
    if sa is None or sb is None:
        if stats is not None: stats["snap_fail"] = stats.get("snap_fail", 0) + 1
        return None
    m = MCP_Geometric(np.where(win, 1.0, np.inf), fully_connected=True)
    if stats is not None: stats["dijkstra_calls"] = stats.get("dijkstra_calls", 0) + 1
    cum, _ = m.find_costs([sa], [sb], find_all_ends=False)
    if not np.isfinite(cum[sb]):
        return None
    path = np.array(m.traceback(sb), float)
    pm = np.stack([(path[:, 1] + c0 + 0.5) / P, (path[:, 0] + r0 + 0.5) / P], 1)
    full = np.vstack([a[None], pm, b[None]])
    poly = cv2.approxPolyDP(full.astype(np.float32).reshape(-1, 1, 2), DP_EPS_MM, False).reshape(-1, 2).astype(float)
    poly[0] = a; poly[-1] = b
    L = _chain_len(poly)
    if L > Lmax + 1e-9:
        return None
    if stats is not None:
        stats["raw_vertices"] = stats.get("raw_vertices", 0) + len(pm); stats["emit_vertices"] = stats.get("emit_vertices", 0) + len(poly)
    return poly, L


def route_strokes(strokes_mm, provider, stats=None):
    """strokes_mm: list of Nx2 page-mm arrays in EMITTED order. Returns (chains, connectors) with connectors = [dict(i, poly, L, d)]."""
    if not strokes_mm:
        return [], []
    chains = []; conns = []; cur = np.asarray(strokes_mm[0], float).copy()
    for k in range(1, len(strokes_mm)):
        nxt = np.asarray(strokes_mm[k], float)
        a, b = cur[-1], nxt[0]
        r = route_hop(a, b, provider, stats)
        if r is None:
            chains.append(cur); cur = nxt.copy()
        else:
            poly, L = r
            conns.append({"i": k - 1, "poly": poly, "L": L, "d": float(np.hypot(*(b - a)))})
            cur = np.vstack([cur, poly[1:], nxt[1:]]) if len(poly) > 1 else np.vstack([cur, nxt[1:]])
    chains.append(cur)
    return chains, conns


def route_document(in_svg, out_svg, connectors_svg, ctx, build_layer_check, bypass_layers=None):
    """Read the (sorted) layout SVG, route every eligible layer, write the fused SVG + the separate connector layer SVG.
    ctx: layer id -> stage_e_mask placement context (only layers present are eligible). Returns the stats record."""
    import vpype as vp
    from ccplotter.atomic_io import atomic_open
    bypass_layers = bypass_layers or set()
    doc = vp.read_multilayer_svg(str(in_svg), quantization=0.1)
    out = vp.Document(); out.page_size = doc.page_size
    cdoc = vp.Document(); cdoc.page_size = doc.page_size; clc = vp.LineCollection()
    layers = {}; tot = {"strokes_before": 0, "strokes_after": 0, "connectors": 0, "connector_mm": 0.0, "layers_routed": 0, "layers_untouched": 0}
    for lid, lc in doc.layers.items():
        lines = list(lc); tot["strokes_before"] += len(lines)
        nlc = vp.LineCollection()
        info = None
        if lid not in bypass_layers and lid in ctx and lines:
            chk = build_layer_check(ctx[lid], np.concatenate([np.asarray(l) for l in lines]))
            if getattr(chk, "available", False):
                st = {}
                mm = [np.stack([np.real(l), np.imag(l)], 1) * PX2MM for l in lines]
                chains, conns = route_strokes(mm, PredicateRaster(chk), st)
                for ch in chains:
                    nlc.append((ch[:, 0] + 1j * ch[:, 1]) / PX2MM)
                for c in conns:
                    if len(c["poly"]) > 1:
                        clc.append((c["poly"][:, 0] + 1j * c["poly"][:, 1]) / PX2MM)
                cmm = sum(c["L"] for c in conns)
                info = {"layer": str(lid), "strokes_before": len(lines), "strokes_after": len(chains), "lifts_removed": len(lines) - len(chains),
                        "connectors": len(conns), "connector_mm": round(cmm, 1), "longest_connector_mm": round(max([c["L"] for c in conns] or [0]), 3),
                        "dijkstra_calls": st.get("dijkstra_calls", 0), "snap_fail": st.get("snap_fail", 0),
                        "vertices_raw_vs_emitted": [st.get("raw_vertices", 0), st.get("emit_vertices", 0)]}
                tot["layers_routed"] += 1; tot["connectors"] += len(conns); tot["connector_mm"] += cmm
            else:
                info = {"layer": str(lid), "untouched": getattr(chk, "reason", "no usable trace mask")}
        else:
            info = {"layer": str(lid), "untouched": "no trace mask for this layer by construction (verbatim / non-tracev3) or empty"}
        if len(nlc) == 0 and lines:
            for l in lines: nlc.append(l)
            tot["layers_untouched"] += 1
        tot["strokes_after"] += len(nlc)
        out.add(nlc, lid); layers[str(lid)] = info
    with atomic_open(out_svg, "w") as f:
        vp.write_svg(f, out, page_size=out.page_size, color_mode="none")
    cdoc.add(clc, 1)
    with atomic_open(connectors_svg, "w") as f:
        vp.write_svg(f, cdoc, page_size=cdoc.page_size, color_mode="none")
    tot["connector_mm"] = round(tot["connector_mm"], 1)
    return {"enabled": True, "arm": "R2a (route only, predicate-region map)", "cap": {"ratio": CAP_RATIO, "abs_mm": CAP_ABS_MM}, "raster_mm_per_px": 1.0 / P,
            "dp_eps_mm": DP_EPS_MM, "totals": tot, "layers": layers,
            "rule": ("after stay-down and the final linesort, a pen-up hop becomes a pen-down connector iff a path exists whose centreline lies in the layer's "
                     "stay-down predicate region and whose length <= min(4 x hop, 3.0 mm); connectors are drawn into a separate layer SVG (layout.ink_connectors.svg)")}
