r"""Boundary metrics for every case: input file, raster tracev3 received, config, SVG identity, render numbers.
Writes data/metrics.json. Reads only local copies of the inputs (never %LOCALAPPDATA%)."""
import hashlib, json, re, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from lib import parse_svg, bbox, stamp_tiled

OUT = Path(__file__).resolve().parents[1]
D = OUT / "data"
IN = OUT / "in"
FIXAB = OUT.parent / "fix-ab-2026-10-03"
BWGAP = OUT.parent / "bwphoto-detail-gap-2026-10-04"


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def raster_stats(p):
    p = Path(p)
    if not p.is_file():
        return None
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    g = im if im.ndim == 2 else cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY)
    n, _ = cv2.connectedComponents((g < 128).astype(np.uint8), connectivity=8)
    return {"file": str(p.relative_to(OUT)), "dims_wh": [int(g.shape[1]), int(g.shape[0])], "channels": 1 if im.ndim == 2 else im.shape[2],
            "sha256": sha_file(p), "black_frac_lt128": round(float((g < 128).mean()), 5),
            "gray_min": int(g.min()), "gray_mean": round(float(g.mean()), 2), "gray_max": int(g.max()),
            "frac_exact_0_or_255": round(float(((g == 0) | (g == 255)).mean()), 4),
            "black_components_8conn": int(n - 1)}


def pathset(polys, nd=3):
    keys = []
    for p in polys:
        f = tuple(map(tuple, np.round(p, nd).tolist()))
        r = f[::-1]
        keys.append(min(f, r))
    keys.sort()
    h = hashlib.sha256(repr(keys).encode()).hexdigest()
    return h


def svg_numbers(polys):
    lo, hi = bbox(polys)
    L = float(sum(np.hypot(*np.diff(p, axis=0).T).sum() for p in polys))
    return {"n_polylines": len(polys), "draw_len_mm": round(L, 1), "bbox_mm": [round(float(v), 2) for v in (*lo, *hi)],
            "bbox_w_h_mm": [round(float(hi[0] - lo[0]), 2), round(float(hi[1] - lo[1]), 2)], "sorted_pathset_sha256_nd3": pathset(polys, 3)}


def ink_numbers(polys, pw, ph, width, ppm=8, reflect=False):
    pl = [np.column_stack([pw - p[:, 0], ph - p[:, 1]]) for p in polys] if reflect else polys
    lo, hi = bbox(pl)
    x0, y0 = max(0.0, lo[0] - 2), max(0.0, lo[1] - 2)
    w, h = min(pw, hi[0] + 2) - x0, min(ph, hi[1] + 2) - y0
    w, h = round(w * 2) / 2, round(h * 2) / 2     # multiples of 0.5 mm (ppm 8, ss 40)
    cov = stamp_tiled(pl, x0, y0, w, h, width, ppm).astype(np.float64) / 255.0
    return {"stroke_mm": width, "mean_cov_pct_own_bbox_plus2": round(float(cov.mean() * 100), 3),
            "ink_area_mm2": round(float(cov.sum() / ppm / ppm), 1)}


def info_of(jd):
    fs = sorted(Path(jd).glob("*.svg.info.json"))
    if not fs:
        return None
    return json.loads(fs[-1].read_text())


def audit_fields(info):
    if not info:
        return None
    a = info.get("audit", {})
    keep = ("passes", "levels", "zero_pass_components", "fallback_strokes", "fill_px", "mm_per_px", "dropped_components",
            "dropped_ink_pct", "speck_polys_dropped", "speck_ink_pct", "lifts", "total_polys", "gray_crush_fraction",
            "n_core_passes", "n_core_regions", "n_core_rows", "core_fill", "flag_mm_per_px", "pen_id_resolved", "fill_mm_resolved")
    d = {k: a.get(k) for k in keep if k in a}
    d.update({k: info.get(k) for k in ("mode", "requested_res", "effective_res", "native_px", "bilevel_source", "upsampled", "downscaled")})
    d["front_end"] = info.get("front_end")
    return d


# (key, label, dir-or-file spec)
CASES = [
    ("A_abworker_b2", "A. exact fix-ab ab_worker invocation (heli, today) re-run", D / "ab" / "A_abworker_b2", 257, 176, False),
    ("B_route_b2", "B. B2 settings through the app route", D / "B_route_b2", 257, 176, False),
    ("C_kai_body", "C. 7e625c85 request_body verbatim on heli_cleaned.png", D / "C_kai_body", 257, 182, True),
    ("S1_b2_clahe_off", "S1. B2 with ONLY clahe=false", D / "S1_b2_clahe_off", 257, 176, False),
    ("S2_b2_hybrid", "S2. B2 with ONLY core_fill=hybrid", D / "S2_b2_hybrid", 257, 176, False),
    ("S3_b2_kai_placement", "S3. B2 with ONLY Kai's placement (paper 257x182, scale 1.2, rot 180, ...)", D / "S3_b2_kai_placement", 257, 182, True),
    ("T1_kai_scale1_p176", "T1. Kai body with ONLY placement set to B2's (paper 257x176, scale 1, rot 0)", D / "T1_kai_scale1_p176", 257, 176, False),
    ("T2_kai_clahe_on", "T2. Kai body with ONLY clahe=true", D / "T2_kai_clahe_on", 257, 182, True),
    ("R1_kai_ui_to_ref", "R1. Kai body with clahe/core_fill/small_detail/vec_merge/resolution/placement set to REF values (rest as the UI sent)", D / "R1_kai_ui_to_ref", 257, 176, False),
    ("R2_b2_penwidth05", "R2. B2 with ONLY pen_width=0.5", D / "R2_b2_penwidth05", 257, 176, False),
]
PERSISTED = [
    ("P_fixab_heli_today", "fix-ab persisted heli__today routed svg", FIXAB / "data/ab/heli__today/layout.routed.svg", 257, 176, False),
    ("P_bwgap_b2", "bwphoto-detail-gap persisted B2 routed svg", BWGAP / "data/b2_fixab_heli_sheet/layout.routed.svg", 257, 176, False),
    ("P_kai_7e625c85_preview", "Kai's ACTUAL run 7e625c85 preview.svg (app-written)", IN / "7e625c85_preview.svg", 257, 182, True),
]


def main():
    res = {"input_files": {}, "cases": {}, "persisted": {}}
    for nm in ("heli_cleaned.png", "7e625c85_cleaned.png", "1fa906f4_cleaned.png"):
        im = cv2.imread(str(IN / nm), cv2.IMREAD_UNCHANGED)
        res["input_files"][nm] = {"sha256": sha_file(IN / nm), "bytes": (IN / nm).stat().st_size,
                                  "dims_wh": [im.shape[1], im.shape[0]], "channels": 1 if im.ndim == 2 else im.shape[2],
                                  "gray_min_mean_max": [int(im.min()), round(float(im.mean()), 2), int(im.max())]}
    fa = FIXAB / "in/heli_cleaned.png"
    res["input_files"]["fix-ab in/heli_cleaned.png"] = {"sha256": sha_file(fa)}
    res["input_files"]["raster_black_frac_lt128_of_heli_cleaned"] = raster_stats(IN / "heli_cleaned.png")
    for key, label, jd, pw, ph, refl in CASES:
        if not (jd / "job.json").is_file():
            res["cases"][key] = {"label": label, "missing": True}
            continue
        job = json.loads((jd / "job.json").read_text())
        c = {"label": label, "paper": [pw, ph], "reflect180": refl, "http_error": job.get("error"),
             "src_sha256": job.get("src_sha256"), "app_saved_src_sha256": job.get("app_saved_src_sha256"),
             "stroke_count": job.get("stroke_count") or (job.get("response") or {}).get("stroke_count")}
        if job.get("error"):
            res["cases"][key] = c
            continue
        info = info_of(jd)
        c["trace_info"] = audit_fields(info)
        for nm in ("debug_gray.png", "debug_trace_mask.png", "debug_core_mask.png"):
            c[nm] = raster_stats(jd / nm)
        prov = jd / "provenance.json"
        if prov.is_file():
            pj = json.loads(prov.read_text())
            c["resolved"] = pj.get("resolved")
            c["effective_body_sha"] = hashlib.sha256(json.dumps(pj.get("effective_body"), sort_keys=True).encode()).hexdigest()[:12]
        sv = jd / "layout.routed.svg"
        polys = parse_svg(sv)
        c["svg"] = svg_numbers(polys)
        c["ink"] = {str(w): ink_numbers(polys, pw, ph, w, reflect=refl) for w in (0.12, 0.20)}
        res["cases"][key] = c
        print("case", key, c["svg"]["n_polylines"], c["ink"]["0.12"]["mean_cov_pct_own_bbox_plus2"], c["ink"]["0.2"]["mean_cov_pct_own_bbox_plus2"], flush=True)
    for key, label, sv, pw, ph, refl in PERSISTED:
        polys = parse_svg(sv)
        c = {"label": label, "svg_file": str(sv), "paper": [pw, ph], "reflect180": refl, "svg": svg_numbers(polys)}
        c["svg"]["sorted_pathset_sha256_nd2"] = pathset(polys, 2)
        c["ink"] = {str(w): ink_numbers(polys, pw, ph, w, reflect=refl) for w in (0.12, 0.20)}
        res["persisted"][key] = c
        print("persisted", key, c["svg"]["n_polylines"], flush=True)
    # nd=2 hashes for cases (preview.svg has 4 decimals, routed.svg may differ)
    for key, label, jd, pw, ph, refl in CASES:
        sv = jd / "layout.routed.svg"
        if sv.is_file():
            res["cases"][key]["svg"]["sorted_pathset_sha256_nd2"] = pathset(parse_svg(sv), 2)
        pv = jd / "layout.preview.svg"
        if pv.is_file():
            pp = parse_svg(pv)
            res["cases"][key]["preview_svg"] = {"n_polylines": len(pp), "sorted_pathset_sha256_nd2": pathset(pp, 2)}
    # F: photo path (one real bwphoto run, 0783addb): rasters entering tracev3
    res["F_photo_bwphoto_0783addb"] = {nm: raster_stats(IN / f"F_0783addb_{nm}.png") for nm in ("debug_gray", "debug_xdog", "debug_trace_mask", "debug_core_mask")}
    (D / "metrics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
