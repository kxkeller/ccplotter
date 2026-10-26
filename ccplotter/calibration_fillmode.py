"""ccplotter/calibration_fillmode.py -- four-patch fill_mode plot-scale sheet (uright_gel).

What it measures: whether the `fill_mode="zigzag"` fill primitive covers paper as well
as the default contour / level-set fill, and whether coverage depends on the machine-frame
angle of the zigzag. Four patches of the SAME source region at the SAME `fill_spacing_mm`:
three zigzag at different MACHINE-frame angles, one contour (the control).

  A contour   -- the default fill. The control.
  B zigzag 0  -- along the pen's wide direction. Predicted solid.
  C zigzag 90 -- deposition along travel is about the same (~0.29 mm) as for B, so it
                 should match B; if it does not, the directional model is incomplete.
  D zigzag 45 -- THE NEGATIVE CONTROL. Predicted visibly worse (more bare paper), from a
                 plotted pitch wedge read for paper fraction.

How to read it: compare bare-paper fraction and blotchiness across the four patches. B
and C should look alike and solid, D should show more bare paper, A is the reference.
`_geometric_bare_paper_pct` separately confirms from the DESIGN geometry that all four
should read ~0 % bare paper, so a bad patch in practice is a pen/deposition effect, not
a generator bug.

PITCH: 0.24 mm. The pitch wedge's ladder is 100/150/180/200/220/240/260/280/300/330/360/
400/500 microns and has no 0.25 mm rung. 0.24 mm (240 u) is the nearest measured rung and
gives the largest discriminating gap between 0 deg and 45 deg near 0.25 mm: bare paper
1.0 % at 0 deg vs 36.2 % at 45 deg. (0.22 mm gives 0.3 % vs 22.2 %; 0.20 mm gives 0.0 %
vs 34.8 %.)

HARD-CODED MACHINE AND PEN (limitation). The sheet is tied to the author's measured
machine: it refuses to run unless the active reach profile is EXPECT_REACH_ID
("plotter_2003_v32") with bbox EXPECT_BBOX. It is PINNED to EXPECT_PEN_ID ("uright_gel"):
it re-resolves that profile out of pen_profiles.json via load_pen_profiles() (read from
the runtime home, never a repo-root shadow copy) and hard-asserts label, pen_mm,
pen_offset, z_down and draw_speed against fixed constants. z_touch_abs is NOT pinned to
a literal, because it changes whenever the pen is recalibrated; it is asserted to be
finite and within Z_TOUCH_SANITY_RANGE_MM (5-20 mm), and the resolved value and its
source (`resolve_z_touch`) are returned in the info dict and printed in the legend.

WRONG-PEN REFUSAL. The pitch and patch D's prediction are calibrated numbers specific to
uright_gel's own wedge reading and do not measure a different pen. `fillmode_gcode`
therefore raises, before any file is written, whenever `profile` names a pen other than
EXPECT_PEN_ID; honouring an arbitrary active pen would give a sheet whose interpretation
does not hold and an on-screen header (active pen label and z_touch) that disagrees with
the g-code. `profile=None` always generates the pinned uright_gel sheet.

MIXED-MODE DESIGN. This sheet never goes through `composite.py`'s item-render pipeline.
Each of the four patches comes from an INDEPENDENT call to `trace_v3._trace_svg` (the
same production function `render.py` calls, same opts contract), once per `fill_mode`
value. The resulting polylines are placed into a shared page by pure translation (an
isometry: it preserves every element's machine-space length and axis-aligned bbox
dimensions exactly, the same guarantee `calibration_penwidth_wedge.py`'s `_pendown_check`
relies on) and emitted in ONE `gwrite` call. No single `_trace_svg` call, and therefore
no single `composite.py` linemerge / connector-router pass, sees more than one
fill_mode's geometry. (In the interactive compose flow `render_opts` is job-wide, so one
composite request carries one fill_mode; this sheet does not use that flow.)

Limitation: patch A (contour) is `_trace_svg`'s raw contour output (dt-band level sets +
`_link_pockets`, `link_mult` default 3.0) and does NOT go through `composite.py`'s
`linemerge --tolerance` or low-lift connector router, both of which a real contour job
gets. Pass pitch and per-patch coverage geometry match the app; the LIFT COUNT of patch A
is higher than a real composited contour job of the same shape. The sheet judges ink
coverage and blotchiness, not lift count.

SOURCE REGION: one 30x30 mm solid black square, not an image crop. A solid patch gives
the cleanest bare-paper readout and lets all four patches share byte-identical source
geometry via one array. Trade-off: the sheet cannot show whether zigzag differs from
contour on real image content (edges, thin strokes), only on solid fill.

WHOLESALE ZIGZAG. `_trace_opts` pins `fill_connect="none"` and
`fill_zigzag_perimeter_count=0`, so the B/C/D patches are wholesale zigzag: no
perimeter / infill split and no connector-router pass. Other zigzag surfaces in the app
(the composite-item checkbox path, or zigzag with boustrophedon connect, proximity,
split/merge and end-linking) produce different geometry. Read B/C/D as "wholesale zigzag,
isolated", not as a preview of those. The pins also keep this sheet's output from changing
if `_trace_svg`'s defaults change.

NO SWEEP: pitch and angle are pure geometry, Z and draw speed stay at the pinned
profile's stock values, ONE gwrite call.

SAFETY GUARD, adapted from calibration_penwidth_wedge.py's isometry-invariant per-element
check (some patches are 45-degree diagonal runs, so an axis-aligned-only travel guard
cannot be reused). Every emitted pen-down stroke's machine-space length AND axis-aligned
bbox width/height are checked 1:1 against the design element that produced it;
translation (plus the sheet's Y-flip) preserves both exactly, so any mismatch means a
cross-element drag. The bbox check is the explicit "no pen-down segment crosses between
patches" guard: zigzag emits thousands of short runs, so a drag from one patch's last run
to the next patch's first run could be short enough to slip past a pure length ceiling.
`pendown_segments_examined` is asserted NON-ZERO so the check cannot pass vacuously.
Angle-deviation-from-design ({0,45,90,135}) is checked only on zigzag patches (B/C/D);
patch A's contour rings are arbitrary-angle polylines.

Reached only through the generic POST /api/calibration_sheet dispatcher (id="fillmode",
registered in calibration_registry.py). It calls `trace_v3._trace_svg` directly, so it
depends on the zigzag fill (`fill_mode="zigzag"`, `_zigzag_fill`) being present.
"""
from __future__ import annotations

import math
import re
import shlex
from pathlib import Path

import numpy as np
import vpype as vp
import vpype_cli

import ccplotter.trace_v3 as trace_v3
from ccplotter.atomic_io import atomic_target
from ccplotter.config import CONFIG_PATH, PREVIEW_COLOR, PX2MM, WORK_DIR
from ccplotter.cal import resolve_pen_offset, load_cal, load_pen_profiles, resolve_z_touch
from ccplotter.gcode_post import (_gcode_lifecycle_values, validate_calibration_gcode_safety,
                                  _fix_gwrite_source_comment)
from ccplotter.reach import active_reach_profile, validate_gcode_xy_motion, pen_reach_profile
from ccplotter.svg_io import _write_paths_svg
from ccplotter.calibration_plots import rect_outline

# ---------------------------------------------------------------------------
# Expected runtime envelope -- re-verified at generation time, not assumed (same
# convention as the sibling generic-dispatcher sheets: an environment without
# reach_profiles falls back to a different default reach profile and would plot
# against the wrong envelope). Tied to the author's measured machine geometry.
# ---------------------------------------------------------------------------
EXPECT_REACH_ID = "plotter_2003_v32"
EXPECT_BBOX = (3.0, 42.5, 255.0, 254.5)
EXPECT_PEN_ID = "uright_gel"
EXPECT_PEN_LABEL = "uRight Gel refill"
EXPECT_PEN_MM = 0.255
EXPECT_PEN_OFFSET = (0.0, 0.0)
EXPECT_Z_DOWN_STOCK = -0.60
EXPECT_DRAW_SPEED_STOCK = 3000
# z_touch_abs is intentionally NOT a pinned literal -- see module docstring.
Z_TOUCH_SANITY_RANGE_MM = (5.0, 20.0)

# ---------------------------------------------------------------------------
# Paper / keep-out (mm). Same convention as calibration_dropout.py /
# calibration_penwidth_wedge.py: JIS B5 257x182mm LANDSCAPE, NOT
# config.PAPER_SIZES['b5'] (ISO 176x250).
# ---------------------------------------------------------------------------
PAPER_W_MM, PAPER_H_MM = 257.0, 182.0
HOLE_KEEPOUT_MM = 10.0
SHEET_MARGIN_MM = 5.0
_PAD_MM = 3.0
_USABLE_W = PAPER_W_MM - 2 * SHEET_MARGIN_MM
_USABLE_H = PAPER_H_MM - HOLE_KEEPOUT_MM - 2 * SHEET_MARGIN_MM

# ---------------------------------------------------------------------------
# Sheet design -- fixed. Changing the pitch means re-reading the pitch wedge (see
# module docstring).
# ---------------------------------------------------------------------------
PATCH_SRC_MM = 30.0          # solid-square ink region, per patch
BORDER_MM = 3.0              # white margin around the square in the SOURCE
                              # raster, so Otsu has two clean populations
PPMM_SRC = 20.0              # source-raster resolution (px/mm); keeps
                              # fill_px = 0.24*20 = 4.8px, comfortably above
                              # config.py's SCALE_AWARE_V2_FILL_PX_FLOOR (3px)
PITCH_MM = 0.24               # nearest MEASURED wedge rung to 0.25mm (see module
                              # docstring)
PATCHES = [
    {"key": "A", "label": "A contour", "fill_mode": "contour", "angle_deg": None},
    {"key": "B", "label": "B zigzag 0d", "fill_mode": "zigzag", "angle_deg": 0.0},
    {"key": "C", "label": "C zigzag 90d", "fill_mode": "zigzag", "angle_deg": 90.0},
    {"key": "D", "label": "D zigzag 45d", "fill_mode": "zigzag", "angle_deg": 45.0},
]
PATCH_GAP_MM = 8.0
LABEL_SIZE_PX = 11
LABEL_GAP_MM = 2.0
LEGEND_TEXT = ("FILL MODE SHEET pitch 240u=0.24mm uright_gel 2026-09-12 "
               "A=contour(control) B=zz0d C=zz90d D=zz45d(neg ctrl,predicted worse)")

# Per-patch pen-down move ceiling (worst legitimate move is one full-diagonal
# contour ring or scanline across the 30mm square) -- a length guard ON TOP
# OF the isometry length-match + bbox-match checks below, not instead of it.
PENDOWN_MAX_MM = round(PATCH_SRC_MM * math.sqrt(2.0) + 5.0, 1)


def _text_polylines_mm(s: str, size_px=LABEL_SIZE_PX):
    """Hershey-font text via vpype's own `text` command. Must be called AFTER
    vp.config_manager.load_config_file. Same idiom as
    calibration_penwidth_wedge.py's `_text_polylines_mm`. Returns
    (polys_mm, w_mm, h_mm), translated so the bbox starts at local (0,0)."""
    doc = vpype_cli.execute(f'text --size {size_px} {shlex.quote(s)}')
    polys = []
    xs, ys = [], []
    for _lid, lc in doc.layers.items():
        for line in lc:
            pts = [(p.real * PX2MM, p.imag * PX2MM) for p in line]
            polys.append(pts)
            xs += [p[0] for p in pts]
            ys += [p[1] for p in pts]
    if not xs:
        return [], 0.0, 0.0
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0, max(ys) - y0
    norm = [[(x - x0, y - y0) for x, y in p] for p in polys]
    return norm, w, h


_POLYLINE_RE = re.compile(r'<polyline points="([^"]+)"')


def _parse_trace_svg_polylines(svg_path):
    """Parse the RAW polyline coordinates `_write_paths_svg` wrote (source-
    raster px, same units as the `img` array passed to `_trace_svg`) directly
    out of the XML text -- deliberately NOT via vpype's `read_svg` (which
    interprets bare SVG px as 1/96in == config.PX2MM mm, the WRONG scale for
    this file: its units are PPMM_SRC pixels-per-mm). Returns polylines as lists of
    (x_px, y_px) tuples in that same source-raster px space."""
    text = Path(svg_path).read_text(encoding="utf-8")
    polys = []
    for m in _POLYLINE_RE.finditer(text):
        pts = []
        for pair in m.group(1).strip().split(" "):
            if not pair:
                continue
            x_s, y_s = pair.split(",")
            pts.append((float(x_s), float(y_s)))
        if len(pts) >= 2:
            polys.append(pts)
    return polys


def _poly_length_mm(pts):
    L = 0.0
    for i in range(len(pts) - 1):
        L += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
    return L


def _poly_bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(xs) - min(xs), max(ys) - min(ys)


def _make_source_image():
    """One 30x30mm solid black square on a white background, at PPMM_SRC
    px/mm, shared byte-identically by all four patches -- see module
    docstring on why a solid patch rather than an image crop."""
    w = h = int(round((PATCH_SRC_MM + 2 * BORDER_MM) * PPMM_SRC))
    img = np.full((h, w), 255, dtype=np.uint8)
    b = int(round(BORDER_MM * PPMM_SRC))
    img[b:h - b, b:w - b] = 0
    return img, w, h


def _trace_opts(fill_mode, angle_deg, pen_mm):
    fill_px = PITCH_MM * PPMM_SRC
    opts = {
        "fill_px": fill_px,
        "_mm_per_px_override": 1.0 / PPMM_SRC,
        "paper_w_mm": 0.0, "margin_mm": 0.0,  # set per-call to W/PPMM_SRC by caller
        "fill_mm": pen_mm,
        "min_mark_mm2": 0.0001,
        "hole_keep": 0.08,
        "tone": "off",
        # `fill_connect` and `fill_zigzag_perimeter_count` are PINNED, not left
        # to `_trace_svg`'s own defaults, so a future default change there cannot
        # silently change what this sheet plots. `fill_zigzag_perimeter_count=0`
        # selects wholesale zigzag (no perimeter/infill split); the default of 1
        # would produce a different geometry.
        "fill_connect": "none",
    }
    if fill_mode == "zigzag":
        opts["fill_mode"] = "zigzag"
        opts["fill_angle_deg"] = float(angle_deg)
        opts["fill_zigzag_min_run_px"] = pen_mm * PPMM_SRC
        opts["fill_zigzag_perimeter_count"] = 0
    return opts


def _build_patch_elements(pen_mm):
    """Runs `trace_v3._trace_svg` once per patch (four independent calls, so no
    single call sees more than one fill_mode -- see module docstring), reads back
    each patch's own polylines, and places them side by side in ONE row
    (same Y for all four -- keeps any position/dropout gradient orthogonal
    to the angle comparison, same reasoning as calibration_dropout.py's
    PRIMARY band). Returns (elements, patch_records, footprint_w, footprint_h,
    trace_audits)."""
    img, w_px, h_px = _make_source_image()
    patch_mm = w_px / PPMM_SRC  # full raster incl. border, in mm

    label_cache = {}
    for p in PATCHES:
        label_cache[p["key"]] = _text_polylines_mm(p["label"])
    max_label_h = max(h for _p, _w, h in label_cache.values())
    max_label_w = max(w for _p, w, _h in label_cache.values())

    # Cell width must fit the WIDER of the patch square and its own label text:
    # label "D zigzag 45deg (neg. control)" is wider than the 36mm patch square. If
    # a label falls outside the frame computed from patch_mm alone, vp.read_svg
    # silently drops its glyph strokes; `_pendown_check`'s stroke-count assertion
    # below catches that.
    cell_w = max(patch_mm, max_label_w)
    cell_h = patch_mm + LABEL_GAP_MM + max_label_h

    legend_polys, legend_w, legend_h = _text_polylines_mm(LEGEND_TEXT, size_px=10)

    footprint_w = max(len(PATCHES) * cell_w + (len(PATCHES) - 1) * PATCH_GAP_MM, legend_w)
    footprint_h = legend_h + 6.0 + cell_h

    if footprint_w + 2 * _PAD_MM > _USABLE_W:
        raise RuntimeError(
            f"fillmode sheet footprint width {footprint_w + 2 * _PAD_MM:.1f}mm exceeds "
            f"the JIS B5 landscape usable width {_USABLE_W:.1f}mm")
    if footprint_h + 2 * _PAD_MM > _USABLE_H:
        raise RuntimeError(
            f"fillmode sheet footprint height {footprint_h + 2 * _PAD_MM:.1f}mm exceeds "
            f"the JIS B5 landscape usable height {_USABLE_H:.1f}mm")

    elements = []
    legend_x = (footprint_w - legend_w) / 2.0 if legend_w < footprint_w else 0.0
    for pts in legend_polys:
        placed = [(x + legend_x, y) for x, y in pts]
        elements.append({"pts": placed, "kind": "legend"})

    grid_y0 = legend_h + 6.0
    patch_records = []
    trace_audits = {}
    for i, p in enumerate(PATCHES):
        cx = i * (cell_w + PATCH_GAP_MM)
        cy = grid_y0

        opts = _trace_opts(p["fill_mode"], p["angle_deg"], pen_mm)
        opts["paper_w_mm"] = patch_mm
        tmp_svg = WORK_DIR / f".fillmode_patch_{p['key']}.svg"
        trace_v3._trace_svg(img.copy(), tmp_svg, opts)
        raw_polys_px = _parse_trace_svg_polylines(tmp_svg)
        tmp_svg.unlink(missing_ok=True)
        trace_audits[p["key"]] = dict(trace_v3._TRACE_AUDIT)

        patch_x0 = cx + (cell_w - patch_mm) / 2.0

        patch_elem_start = len(elements)
        for poly_px in raw_polys_px:
            poly_mm = [(x / PPMM_SRC, y / PPMM_SRC) for x, y in poly_px]
            placed = [(x + patch_x0, y + cy) for x, y in poly_mm]
            elements.append({"pts": placed, "kind": "patch_fill", "patch_key": p["key"],
                              "fill_mode": p["fill_mode"], "angle_deg": p["angle_deg"]})
        patch_elem_end = len(elements)

        label_polys, label_w, _label_h = label_cache[p["key"]]
        lx0 = cx + (cell_w - label_w) / 2.0
        ly0 = cy + patch_mm + LABEL_GAP_MM
        for pts in label_polys:
            placed = [(x + lx0, y + ly0) for x, y in pts]
            elements.append({"pts": placed, "kind": "label", "patch_key": p["key"]})

        patch_records.append({
            "key": p["key"], "label": p["label"], "fill_mode": p["fill_mode"],
            "angle_deg_image_frame_requested": p["angle_deg"],
            "n_passes": len(raw_polys_px),
            "elem_range": [patch_elem_start, patch_elem_end],
            "origin_mm": [round(patch_x0, 3), round(cy, 3)],
            "patch_ink_mm": [round(PATCH_SRC_MM, 3), round(PATCH_SRC_MM, 3)],
            "patch_raster_mm": [round(patch_mm, 3), round(patch_mm, 3)],
        })

    frame = rect_outline(-2.0, -2.0, footprint_w + 4.0, footprint_h + 4.0)
    for pts in frame:
        elements.append({"pts": pts, "kind": "frame"})

    return elements, patch_records, footprint_w, footprint_h, trace_audits


# ---------------------------------------------------------------------------
# Post-emission validation. Isometry-invariant 1:1 length-match AND
# axis-aligned-bbox-match check per element (the bbox check is this sheet's
# explicit cross-patch-drag guard -- see module docstring), plus an
# angle-deviation check against {0,45,90,135} for zigzag patch-fill strokes
# only. `pendown_segments_examined` is asserted NON-ZERO so the checks cannot
# pass vacuously.
# ---------------------------------------------------------------------------
def _parse_strokes(path, z_down_abs, z_tol=1e-2):
    strokes = []
    cur = None
    x = y = None
    down = False
    with Path(path).open() as fh:
        for line in fh:
            if not line.startswith("G1"):
                continue
            mz = re.search(r"Z(-?[\d.]+)", line)
            if mz:
                was_down = down
                down = abs(round(float(mz.group(1)), 3) - round(z_down_abs, 3)) < z_tol
                if down and not was_down:
                    cur = {"pts": []}
                    if x is not None:
                        cur["pts"].append((x, y))
                    strokes.append(cur)
                elif was_down and not down:
                    cur = None
                continue
            mx = re.search(r"X(-?[\d.]+)", line)
            my = re.search(r"Y(-?[\d.]+)", line)
            nx = float(mx.group(1)) if mx else x
            ny = float(my.group(1)) if my else y
            if down and cur is not None:
                cur["pts"].append((nx, ny))
            x, y = nx, ny
    return [s for s in strokes if len(s["pts"]) >= 2]


def _pendown_check(out_gcode, z_down_abs, elements, n_pendowns_expected):
    strokes = _parse_strokes(out_gcode, z_down_abs)
    n_pendown_segments = sum(len(s["pts"]) - 1 for s in strokes)
    if n_pendown_segments <= 0:
        raise RuntimeError(
            "fillmode: pendown_segments_examined is ZERO -- the guard never ran "
            "(a vacuous pass). Do not plot this file.")
    if len(strokes) != n_pendowns_expected:
        raise RuntimeError(
            f"fillmode: stroke count mismatch: emitted {len(strokes)} pen-down cycles, "
            f"expected {n_pendowns_expected} (one per design element) -- gcode structure "
            f"does not match the design 1:1. Do not plot this file.")

    worst_len_patch = 0.0
    n_len_violations = 0
    length_mismatches = []
    bbox_mismatches = []
    angle_devs = []
    cross_patch_drags = []
    for i, (elem, stroke) in enumerate(zip(elements, strokes)):
        expected_len = _poly_length_mm(elem["pts"])
        actual_len = _poly_length_mm(stroke["pts"])
        expected_w, expected_h = _poly_bbox(elem["pts"])
        actual_w, actual_h = _poly_bbox(stroke["pts"])
        seg_lens = [math.hypot(stroke["pts"][j + 1][0] - stroke["pts"][j][0],
                                stroke["pts"][j + 1][1] - stroke["pts"][j][1])
                    for j in range(len(stroke["pts"]) - 1)]
        max_seg = max(seg_lens) if seg_lens else 0.0
        if elem["kind"] == "patch_fill":
            worst_len_patch = max(worst_len_patch, max_seg)
            if max_seg > PENDOWN_MAX_MM:
                n_len_violations += 1
        if abs(actual_len - expected_len) > max(0.05, 0.01 * expected_len):
            length_mismatches.append({"elem_idx": i, "kind": elem["kind"],
                                       "expected_mm": round(expected_len, 3),
                                       "actual_mm": round(actual_len, 3)})
        bbox_tol = max(0.05, 0.01 * max(expected_w, expected_h))
        if abs(actual_w - expected_w) > bbox_tol or abs(actual_h - expected_h) > bbox_tol:
            entry = {"elem_idx": i, "kind": elem["kind"],
                      "patch_key": elem.get("patch_key"),
                      "expected_wh_mm": [round(expected_w, 3), round(expected_h, 3)],
                      "actual_wh_mm": [round(actual_w, 3), round(actual_h, 3)]}
            bbox_mismatches.append(entry)
            if elem["kind"] == "patch_fill":
                cross_patch_drags.append(entry)
        if elem["kind"] == "patch_fill" and elem.get("fill_mode") == "zigzag" and \
           len(stroke["pts"]) == 2:
            (x0, y0), (x1, y1) = stroke["pts"]
            ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
            dev = min(abs(ang - a) for a in (0.0, 45.0, 90.0, 135.0, 180.0))
            angle_devs.append({"patch_key": elem["patch_key"], "dev_deg": dev,
                                "raw_angle_deg": ang})

    if n_len_violations != 0:
        raise RuntimeError(
            f"fillmode: {n_len_violations} patch-fill pen-down move(s) exceed "
            f"{PENDOWN_MAX_MM}mm (worst {worst_len_patch:.1f}mm). Do not plot this file.")
    if length_mismatches:
        raise RuntimeError(
            f"fillmode: {len(length_mismatches)} element(s) have machine-length != "
            f"design-length beyond tolerance: {length_mismatches[:5]}. Do not plot this file.")
    if cross_patch_drags:
        raise RuntimeError(
            f"fillmode: {len(cross_patch_drags)} patch-fill element(s) have a machine "
            f"bbox that does not match their design bbox -- this is the cross-patch-drag "
            f"guard firing: {cross_patch_drags[:5]}. Do not plot this file.")
    max_angle_dev = max((d["dev_deg"] for d in angle_devs), default=0.0)
    if max_angle_dev >= 0.5:
        raise RuntimeError(
            f"fillmode: a zigzag patch-fill stroke deviates {max_angle_dev:.2f}deg from "
            f"the nearest axis-aligned/diagonal direction -- unexpected geometry. "
            f"Do not plot this file.")

    return {
        "pendown_segments_examined": n_pendown_segments,
        "n_strokes": len(strokes),
        "n_pendowns_expected": n_pendowns_expected,
        "pendown_travel_violations": n_len_violations,
        "max_pendown_mm_patch_fill": round(worst_len_patch, 2),
        "length_mismatches": length_mismatches,
        "bbox_mismatches": bbox_mismatches,
        "cross_patch_drag_check": {"n_checked": sum(1 for e in elements if e["kind"] == "patch_fill"),
                                    "n_violations": len(cross_patch_drags)},
        "zigzag_angle_deviations": angle_devs,
        "max_zigzag_angle_deviation_deg": round(max_angle_dev, 4),
    }, strokes


def _pitch_realization_zigzag(patch_records, strokes):
    """Same idea as calibration_penwidth_wedge.py's `_pitch_realization`:
    perpendicular distance between adjacent zigzag runs' start points,
    projected onto the run direction's own normal, compared to the
    commanded PITCH_MM. Only meaningful for zigzag patches (B/C/D)."""
    def _perp_distance(p0, p1, q):
        x0, y0 = p0; x1, y1 = p1; qx, qy = q
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return 0.0
        return abs(dx * (y0 - qy) - (x0 - qx) * dy) / L

    out = {}
    for rec in patch_records:
        if rec["fill_mode"] != "zigzag":
            continue
        a, b = rec["elem_range"]
        fill_strokes = strokes[a:b]
        if len(fill_strokes) < 2:
            continue
        dists = []
        for i in range(len(fill_strokes) - 1):
            s0, s1 = fill_strokes[i], fill_strokes[i + 1]
            p0, p1 = s0["pts"][0], s0["pts"][-1]
            q = s1["pts"][0]
            dists.append(_perp_distance(p0, p1, q))
        realized = sum(dists) / len(dists)
        dev_pct = 100.0 * (realized - PITCH_MM) / PITCH_MM
        out[rec["key"]] = {
            "commanded_pitch_mm": PITCH_MM, "realized_pitch_mm": round(realized, 4),
            "deviation_pct": round(dev_pct, 2), "n_gaps_measured": len(dists),
            "min_gap_mm": round(min(dists), 4), "max_gap_mm": round(max(dists), 4),
        }
    return out


def _machine_angle_of_patch(patch_records, strokes, key):
    """Median machine-frame angle (degrees, mod 180) of a zigzag patch's
    2-point runs, read from the EMITTED gcode -- not inferred from the design
    geometry, because the sign of a rotation/flip is easy to get wrong by
    inspection."""
    rec = next(r for r in patch_records if r["key"] == key)
    a, b = rec["elem_range"]
    angs = []
    for s in strokes[a:b]:
        if len(s["pts"]) != 2:
            continue
        (x0, y0), (x1, y1) = s["pts"]
        angs.append(math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0)
    if not angs:
        return None
    angs.sort()
    return angs[len(angs) // 2]


def _geometric_bare_paper_pct(patch_records, elements, pen_mm, mm_per_check_px=0.05):
    """Sanity check independent of the physical-deposition question this
    sheet exists to test: rasterize each patch's DESIGN geometry (not the
    physical plot) at nominal pen width and report bare-paper % within the
    inked square. All four should read ~0% -- if one does not, that is a
    generator bug in THIS module, not the physical pen-direction effect
    (which this computation cannot see at all)."""
    import cv2
    ppmm = 1.0 / mm_per_check_px
    out = {}
    for rec in patch_records:
        a, b = rec["elem_range"]
        w_px = int(round(rec["patch_ink_mm"][0] * ppmm))
        h_px = int(round(rec["patch_ink_mm"][1] * ppmm))
        canvas = np.zeros((h_px, w_px), np.uint8)
        ox, oy = rec["origin_mm"]
        border_off = BORDER_MM
        for e in elements[a:b]:
            pts = np.array([[(x - ox - border_off) * ppmm, (y - oy - border_off) * ppmm]
                             for x, y in e["pts"]], np.int32)
            cv2.polylines(canvas, [pts], False, 255, max(1, int(round(pen_mm * ppmm))),
                          lineType=cv2.LINE_8)
        inked = int(np.count_nonzero(canvas))
        bare_pct = 100.0 * (1.0 - inked / max(1, canvas.size))
        out[rec["key"]] = round(bare_pct, 3)
    return out


# ---------------------------------------------------------------------------
# Emission -- single Z / single draw speed, single gwrite call, same pattern
# as calibration_dropout.dropout_gcode / calibration_penwidth_wedge.penwidth_wedge_gcode.
# ---------------------------------------------------------------------------
def _pen_id_from_profile(profile: dict, profiles_doc: dict):
    """Recover the pen_id key that CURRENTLY owns this exact profile blob, by
    re-reading pen_profiles.json (never trusting the caller's dict on its own
    -- see the refusal check in fillmode_gcode). Returns None if the blob
    matches no id (stale/foreign blob) or more than one (ambiguous)."""
    matches = [pid for pid, p in profiles_doc.get("profiles", {}).items() if p == profile]
    return matches[0] if len(matches) == 1 else None


def fillmode_gcode(out_gcode: Path, pen_mm=None, z_up=3.5,
                    travel_speed=8000, plunge_speed=500, lift_speed=2000,
                    offset_x=None, offset_y=None, end_policy="release",
                    quiet_fans=True, profile=None):
    """Emit the four-patch fill_mode sheet as ONE g-code file, ONE gwrite
    call.

    `profile`: accepted for signature parity with the generic dispatcher, and
    used for EXACTLY ONE purpose -- a refusal check. The 0.24mm pitch and patch
    D's negative-control prediction were calibrated from uright_gel's OWN
    plotted pitch wedge (see module docstring); they are not a valid
    measurement of a different pen's laid width, so this sheet does NOT
    re-target itself to `profile`. If `profile` names a pen other than
    EXPECT_PEN_ID, this refuses loudly rather than silently emitting
    uright_gel's g-code under a different pen's name (wrong pen / wrong Z).
    `profile=None` (every no-argument call site) always generates the pinned
    uright_gel sheet."""
    ppmm = 1.0 / PX2MM
    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()

    profiles = load_pen_profiles()
    if EXPECT_PEN_ID not in profiles.get("profiles", {}):
        raise RuntimeError(f"pen profile {EXPECT_PEN_ID!r} not found in pen_profiles.json")
    pen = profiles["profiles"][EXPECT_PEN_ID]

    # Refusal gate FIRST -- before the reach check and before any WORK_DIR
    # write -- so a reach mismatch can never mask the wrong-pen message, and no
    # partial/mismatched file is ever produced.
    if profile is not None and profile != pen:
        req_id = _pen_id_from_profile(profile, profiles)
        req_id_str = repr(req_id) if req_id else "unknown (blob matches no current pen)"
        req_label = profile.get("label")
        raise RuntimeError(
            f"fillmode: refusing to generate -- this sheet is PINNED to "
            f"{EXPECT_PEN_ID!r} ({EXPECT_PEN_LABEL!r}); the active/requested pen is "
            f"{req_id_str} "
            f"({req_label!r}). The 0.24mm pitch and patch D's negative-control "
            f"prediction were both calibrated from uright_gel's own plotted pitch "
            f"wedge and are not a valid measurement of a different pen's laid width. "
            f"Mount uright_gel and re-request this sheet, or ask for a dedicated "
            f"fillmode sheet design for {req_label!r}. Refusing rather than silently "
            f"emitting uright_gel's g-code under {req_label!r}'s name/Z.")

    reach0 = active_reach_profile(cal)
    if reach0["id"] != EXPECT_REACH_ID or tuple(round(v, 1) for v in reach0["bbox"]) != EXPECT_BBOX:
        raise RuntimeError(
            f"Runtime reach envelope mismatch: got id={reach0['id']!r} bbox={reach0['bbox']}, "
            f"expected id={EXPECT_REACH_ID!r} bbox={EXPECT_BBOX}. Refusing to generate against "
            f"an unexpected envelope.")

    label = pen.get("label")
    if label != EXPECT_PEN_LABEL:
        raise RuntimeError(f"pen {EXPECT_PEN_ID!r} label mismatch: got {label!r}, "
                           f"expected {EXPECT_PEN_LABEL!r}. Refusing (this is exactly the "
                           f"repo-root pen_profiles.json CWD shadow / stale-runtime-copy "
                           f"failure mode).")
    if abs(float(pen.get("pen_mm", 0.0)) - EXPECT_PEN_MM) > 1e-6:
        raise RuntimeError(f"pen {EXPECT_PEN_ID!r} pen_mm mismatch: got {pen.get('pen_mm')}, "
                           f"expected {EXPECT_PEN_MM}")
    got_offset = (float(pen.get("pen_offset_x", 0.0)), float(pen.get("pen_offset_y", 0.0)))
    if any(abs(a - b) > 1e-6 for a, b in zip(got_offset, EXPECT_PEN_OFFSET)):
        raise RuntimeError(f"pen {EXPECT_PEN_ID!r} pen_offset mismatch: got {got_offset}, "
                           f"expected {EXPECT_PEN_OFFSET}")

    z_touch, z_touch_source = resolve_z_touch(cal, pen)
    lo, hi = Z_TOUCH_SANITY_RANGE_MM
    if not (lo <= z_touch <= hi):
        raise RuntimeError(f"z_touch_abs for {EXPECT_PEN_ID} resolved to {z_touch}, outside "
                           f"the sanity range [{lo},{hi}]mm -- refusing (see module docstring "
                           f"on why this sheet checks a range, not a pinned literal).")
    z_down = float(pen.get("z_down", 0.0))
    if abs(z_down - EXPECT_Z_DOWN_STOCK) > 1e-6:
        raise RuntimeError(f"stock z_down mismatch for {EXPECT_PEN_ID}: got {z_down}, "
                           f"expected {EXPECT_Z_DOWN_STOCK}")
    draw_speed = float(pen.get("draw_speed", 0.0))
    if abs(draw_speed - EXPECT_DRAW_SPEED_STOCK) > 1e-6:
        raise RuntimeError(f"stock draw_speed mismatch for {EXPECT_PEN_ID}: got {draw_speed}, "
                           f"expected {EXPECT_DRAW_SPEED_STOCK}")

    reach = pen_reach_profile(reach0, pen)
    if pen_mm is None:
        pen_mm = EXPECT_PEN_MM

    pen_off_x, pen_off_y = resolve_pen_offset(cal, pen)

    elements, patch_records, footprint_w, footprint_h, trace_audits = _build_patch_elements(pen_mm)
    n_pendowns_expected = len(elements)

    PLACEMENT_MARGIN_MM = 20.0
    pad = _PAD_MM
    canvas_w_mm, canvas_h_mm = footprint_w + 2 * pad, footprint_h + 2 * pad

    offset_x = reach["bbox"][0] + PLACEMENT_MARGIN_MM if offset_x is None else float(offset_x)
    if offset_y is None:
        # Same vertical_flip handling as calibration_penwidth_wedge.py's
        # offset_y default.
        flip_axis_mm = max(canvas_w_mm, canvas_h_mm)
        near_edge_target = reach["bbox"][1] + PLACEMENT_MARGIN_MM
        comp_off_y = near_edge_target - flip_axis_mm + pad + footprint_h
        offset_y = comp_off_y + pen_off_y
    else:
        offset_y = float(offset_y)
    comp_off_x, comp_off_y = offset_x - pen_off_x, offset_y - pen_off_y
    park_x, park_y = reach["safe_park_point"]
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))

    W = int(round(canvas_w_mm * ppmm))
    H = int(round(canvas_h_mm * ppmm))

    placed = [[((x + pad) * ppmm, (y + pad) * ppmm) for x, y in e["pts"]] for e in elements]
    svg = WORK_DIR / f".{out_gcode.stem}.svg"
    _write_paths_svg(svg, W, H, float(pen_mm) * ppmm, polylines=placed)

    z_down_abs = z_touch + z_down
    vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({
        "z_up": float(z_up), "z_up_abs": z_touch + float(z_up),
        "z_down": z_down, "z_down_abs": z_down_abs,
        "z_touch_abs": z_touch, "z_travel_abs": z_travel_abs,
        "draw_speed": draw_speed, "travel_speed": float(travel_speed),
        "plunge_speed": float(plunge_speed), "lift_speed": float(lift_speed),
        "z_auto": "1", "origin_x": park_x, "origin_y": park_y,
        **_gcode_lifecycle_values(end_policy, quiet_fans)})

    with atomic_target(out_gcode) as tmp_gcode:
        vpype_cli.execute(f'read {shlex.quote(str(svg))} pagesize {W}x{H}px '
                          f'translate -- {comp_off_x}mm {-comp_off_y}mm '
                          f'gwrite --profile elegoo_cc {shlex.quote(str(tmp_gcode))}')
        _fix_gwrite_source_comment(tmp_gcode, out_gcode)
    svg.unlink(missing_ok=True)

    validate_gcode_xy_motion(out_gcode, reach)
    safety = validate_calibration_gcode_safety(out_gcode, z_down_abs)
    pendown_check, strokes = _pendown_check(out_gcode, z_down_abs, elements, n_pendowns_expected)
    pitch_realization = _pitch_realization_zigzag(patch_records, strokes)
    angle_B = _machine_angle_of_patch(patch_records, strokes, "B")
    angle_C = _machine_angle_of_patch(patch_records, strokes, "C")
    angle_D = _machine_angle_of_patch(patch_records, strokes, "D")
    geometric_bare_pct = _geometric_bare_paper_pct(patch_records, elements, pen_mm)

    # Plot-time estimate by arithmetic on the REAL emitted gcode: sum draw-move
    # length at draw_speed, travel-move length at travel_speed, plus a fixed
    # per-lift plunge/retract budget -- same style as other sheets' estimates.
    total_draw_mm = sum(_poly_length_mm(s["pts"]) for s in strokes)
    plot_time_s = total_draw_mm / max(1.0, draw_speed / 60.0)
    lift_budget_s = pendown_check["n_strokes"] * (
        (abs(z_up) / max(1.0, plunge_speed / 60.0)) + (abs(z_up) / max(1.0, lift_speed / 60.0)))
    plot_time_s += lift_budget_s

    prev_svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_paths_svg(prev_svg, W, H, float(pen_mm) * ppmm, polylines=placed)
    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(prev_svg))} color {PREVIEW_COLOR} '
                          f'write {shlex.quote(str(tmp_prev))}')

    return {"kind": "fillmode", "pen_mm": pen_mm, "pen_id": EXPECT_PEN_ID,
            "pen_label": EXPECT_PEN_LABEL,
            "footprint_mm": [round(canvas_w_mm, 1), round(canvas_h_mm, 1)],
            "usable_mm": [round(_USABLE_W, 1), round(_USABLE_H, 1)],
            "offset_mm": [offset_x, offset_y],
            "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
            "z_down": z_down, "z_down_abs": round(z_down_abs, 3), "draw_speed": draw_speed,
            "reach_id": reach0["id"], "reach_bbox": reach0["bbox"],
            "n_pendown_expected_total": n_pendowns_expected,
            "n_pendown_actual_total": pendown_check["n_strokes"],
            "pendown_count_matches": pendown_check["n_strokes"] == n_pendowns_expected,
            "pendown_segments_examined": pendown_check["pendown_segments_examined"],
            "deepest_z_down_abs": round(z_down_abs, 3),
            "safety_selfcheck": safety, "pendown_check": pendown_check,
            "pitch_mm": PITCH_MM,
            "pitch_realization": pitch_realization,
            "machine_frame_angle_deg": {"B": angle_B, "C": angle_C, "D": angle_D},
            "geometric_bare_paper_pct": geometric_bare_pct,
            "trace_audits": trace_audits,
            "plot_time_estimate_s": round(plot_time_s, 1),
            "plot_time_estimate_min": round(plot_time_s / 60.0, 2),
            "preview": prev.name, "svg_path": str(prev_svg), "svg_wh_px": [W, H],
            "pattern_mm": [round(canvas_w_mm, 1), round(canvas_h_mm, 1)],
            "place": {"x": offset_x, "y": offset_y, "w": canvas_w_mm, "h": canvas_h_mm},
            "z_up_abs": round(z_touch + float(z_up), 3),
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
            "patch_records": patch_records}
