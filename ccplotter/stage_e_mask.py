"""Opt-in mask check for Stage E low-lift fuses (`connector_mask_check`).

Why: `_route_low_lift` (stats_warnings.py) fuses consecutive strokes whose endpoint gap is within
the connector distance (0.5 mm by default in the UI) as a PEN-DOWN straight connector with NO test
against the ink mask, so a connector can ink paper the source leaves white. This module supplies
the test.

Predicate -- the SAME one stitch.py uses (`stitch._connector_in_mask`): the straight connector
is accepted only when every sample along it (0.5 trace px apart, bilinear) has distance-to-white
dt >= 0.5 * fill_px, evaluated in the TRACE frame on the final trace mask. Page points are mapped
back to the trace frame through the exact placement composite applied (vpype scale / rotate about
the layer bbox centre, then translate), see `LayerMaskCheck`. Two deliberate deviations, both
recorded in the audit block:
  * `DT_TOL_PX`: the page-space endpoints went through SVG 2-decimal rounding, so the threshold is
    lowered by this many trace px (dt has slope <= 1, so 0.05 px of coordinate noise moves dt by at
    most 0.05) -- without it a connector sitting exactly on an outermost-ring vertex fails on noise;
  * coincident endpoints (< stitch.ZERO_LEN_PX trace px apart) fuse with no test: they add no ink.
A layer whose mask is unavailable, or whose reconstructed placement fails the self-check (fewer than
`SELF_CHECK_MIN_FRAC` of the layer's own vertices land on a mask pixel), is FAIL-CLOSED: every
non-zero-length fuse is rejected and the layer is named in the record.

Everything here is reached only under `connector_mask_check`; with the key absent no module is
imported, no file is written, and nothing is added to any hash or audit.
"""
import math
import os
from pathlib import Path

import numpy as np

from ccplotter.config import PX2MM
from ccplotter.stitch import SAMPLE_STEP_PX, ZERO_LEN_PX, _CHUNK, _connector_in_mask, _dt_at

SIDECAR_SUFFIX = ".mask.npz"
DT_TOL_PX = 0.05
SELF_CHECK_MIN_FRAC = 0.98
SELF_CHECK_MAX_POINTS = 50000
AUDIT_STEP_MM = 0.02          # same sampling as short-piece-provenance 10_analyse.audit


def sidecar_path(svg_path):
    return Path(str(svg_path) + SIDECAR_SUFFIX)


def write_sidecar(svg_path, mask, fill_px):
    """Called by trace_v3 (only under the private `_stage_e_mask` key)."""
    p = sidecar_path(svg_path)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, mask=np.asarray(mask).astype(bool), fill_px=np.float64(fill_px))
    os.replace(tmp, p)


def load_sidecar(svg_path):
    p = sidecar_path(svg_path)
    if not p.exists():
        return None
    with np.load(p) as z:
        return np.asarray(z["mask"]).astype(bool), float(z["fill_px"])


class NoMaskCheck:
    """Fail-closed stand-in: no usable mask for this layer, so no non-zero-length fuse is allowed."""
    available = False

    def __init__(self, reason):
        self.reason = reason

    def ok(self, A, B):
        return np.abs(np.asarray(B) - np.asarray(A)) <= 1e-6

    def measure(self, A, B):
        return None

    def info(self):
        return {"available": False, "reason": self.reason}


class LayerMaskCheck:
    """Mask predicate for one layer. A, B are complex page-px endpoint arrays (vpype units)."""
    available = True

    def __init__(self, mask, fill_px, c0_px, scale, rot_deg, t_px, tol_px=DT_TOL_PX):
        import cv2
        self.mask = np.asarray(mask).astype(bool)
        self.H, self.W = self.mask.shape
        self.fill_px = float(fill_px)
        self.thr = 0.5 * self.fill_px
        self.tol = float(tol_px)
        self.dt = cv2.distanceTransform(self.mask.astype(np.uint8), cv2.DIST_L2, 5)
        self.c0 = complex(c0_px[0], c0_px[1])
        self.scale = float(scale)
        self.rot = complex(math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg)))
        self.t = complex(t_px[0], t_px[1])
        self.mm_per_trace_px = None     # set by self_check (placed mm per trace px)
        self.self_check_frac = None

    # page px -> trace px:  p' = c0 + rot * s * (p - c0) + t   =>   p = c0 + (p' - c0 - t) / (rot * s)
    def to_trace(self, z):
        z = np.asarray(z, dtype=complex)
        w = self.c0 + (z - self.c0 - self.t) / (self.rot * self.scale)
        return np.stack([w.real, w.imag], axis=-1)

    def self_check(self, vertices):
        """Fraction of the layer's own vertices that land on a mask pixel under this placement. The
        strokes ARE the ink, so a correct placement gives ~1.0; a wrong one gives ~mask coverage."""
        v = np.asarray(vertices, dtype=complex)
        if len(v) > SELF_CHECK_MAX_POINTS:
            v = v[np.linspace(0, len(v) - 1, SELF_CHECK_MAX_POINTS).astype(np.int64)]
        p = self.to_trace(v)
        ix = np.rint(p[:, 0]).astype(np.int64)
        iy = np.rint(p[:, 1]).astype(np.int64)
        inside = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
        hit = np.zeros(len(p), bool)
        hit[inside] = self.mask[iy[inside], ix[inside]]
        self.self_check_frac = float(hit.mean()) if len(hit) else 0.0
        self.mm_per_trace_px = self.scale * PX2MM
        return self.self_check_frac

    def ok(self, A, B):
        a, b = self.to_trace(A), self.to_trace(B)
        d = np.hypot(*(b - a).T)
        out = np.ones(len(d), bool)
        nz = np.nonzero(d > ZERO_LEN_PX)[0]
        for s in range(0, len(nz), _CHUNK):
            idx = nz[s:s + _CHUNK]
            out[idx] = _connector_in_mask(self.dt, a[idx], b[idx], self.thr - self.tol)
        return out

    def measure(self, A, B):
        """(total_mm, off_mm, spill_mm) of the straight connectors A[i]->B[i], sampled every
        AUDIT_STEP_MM along the page-space segment. off = sample centre on a mask==0 pixel (nearest);
        spill = bilinear dt < 0.5*fill_px (pen footprint half-width leaves the mask, strict, no tol)."""
        A = np.asarray(A, dtype=complex)
        B = np.asarray(B, dtype=complex)
        if len(A) == 0:
            return 0.0, 0.0, 0.0
        lens_mm = np.abs(B - A) * PX2MM
        ns = np.maximum(2, np.ceil(lens_mm / AUDIT_STEP_MM).astype(np.int64) + 1)
        starts = np.cumsum(ns) - ns
        rep = np.repeat(np.arange(len(A)), ns)
        off_ix = np.arange(int(ns.sum())) - np.repeat(starts, ns)
        t = off_ix / np.repeat(ns - 1, ns)
        pts = A[rep] + (B[rep] - A[rep]) * t
        p = self.to_trace(pts)
        ix = np.clip(np.rint(p[:, 0]).astype(np.int64), 0, self.W - 1)
        iy = np.clip(np.rint(p[:, 1]).astype(np.int64), 0, self.H - 1)
        off = ~self.mask[iy, ix]
        spill = _dt_at(self.dt, p) < self.thr
        cnt = np.bincount(rep, minlength=len(A))
        f_off = np.bincount(rep, weights=off, minlength=len(A)) / cnt
        f_spill = np.bincount(rep, weights=spill, minlength=len(A)) / cnt
        return float(lens_mm.sum()), float((f_off * lens_mm).sum()), float((f_spill * lens_mm).sum())

    def info(self):
        return {"available": True, "fill_px": round(self.fill_px, 4), "threshold_px": round(self.thr, 4),
                "dt_tol_px": self.tol, "sample_step_trace_px": SAMPLE_STEP_PX,
                "mask_shape": [self.H, self.W], "placed_mm_per_trace_px": self.mm_per_trace_px,
                "self_check_frac": self.self_check_frac}


def build_layer_check(ctx, layer_vertices):
    """ctx: dict from composite {svg, s, rot_deg, c0_px, t_px}; layer_vertices: complex vertices of
    that layer's Stage-E input. Returns a LayerMaskCheck, or a NoMaskCheck naming why not."""
    loaded = load_sidecar(ctx["svg"])
    if loaded is None:
        return NoMaskCheck("no_trace_mask_sidecar")
    mask, fill_px = loaded
    chk = LayerMaskCheck(mask, fill_px, ctx["c0_px"], ctx["s"], ctx["rot_deg"], ctx["t_px"])
    frac = chk.self_check(layer_vertices)
    if frac < SELF_CHECK_MIN_FRAC:
        bad = NoMaskCheck(f"placement_self_check_failed(frac={frac:.4f})")
        bad.self_check_frac = frac
        return bad
    return chk


def empty_totals():
    return {"layers_checked": 0, "layers_failed_closed": 0, "layers_failed_closed_detail": {},
            "fuses_planned_unchecked": 0, "fuses_attempted": 0, "fuses_accepted": 0, "fuses_rejected": 0,
            "fuses_added_by_target_backfill": 0, "strokes_unchecked": 0, "strokes_checked": 0,
            "unchecked_connector_mm": 0.0, "unchecked_offmask_mm": 0.0, "unchecked_spill_mm": 0.0,
            "accepted_connector_mm": 0.0, "accepted_offmask_mm": 0.0, "accepted_spill_mm": 0.0,
            "rejected_connector_mm": 0.0}
