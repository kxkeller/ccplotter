"""ccplotter/calibration_penwidth_wedge.py -- pitch-wedge laid-width sheet (any pen).

What it measures: the pen's laid line width, read as a PITCH rather than as a hairline.
Photographing one isolated stroke needs more resolution than a handheld phone camera has
(a phone photo returns nothing for 0.18 / 0.255 / 0.35 mm width-control bands). But the
pitch at which a filled patch of parallel passes first shows white striping between
passes IS the laid width: below that pitch the passes overlap and the patch reads solid;
at or above it a gap survives and the patch reads striped. That transition is a
centimetre-scale judgement, visible by eye or in an ordinary phone photo, because it is
read off the whole patch's texture.

How to read it: the sheet is a ladder of patches at increasing pitch, with a known-solid
anchor (well below any plausible laid width) and a known-striped anchor (well above)
bracketing it. Read the ladder as "solid ... solid ... first-striped ... striped"; the
commanded pitch of the crossing patch is the laid-width estimate. Rows are 0 deg and
45 deg, because the effective width depends on direction. Labels are plotted text in
microns, and the legend prints the pen and (in absolute mode) the bracket.

HARD-CODED MACHINE (limitation). The sheet is tied to the author's measured machine: it
refuses to run unless the active reach profile is EXPECT_REACH_ID ("plotter_2003_v32")
with bbox EXPECT_BBOX. This is about the plotter, not the pen, and does not vary by pen_id.

PEN SELECTION. The sheet accepts any pen id and measures the pen under test, so the
profile's own Z is required (no fallback):
  * `pen_id` (default DEFAULT_PEN_ID="uright_gel") is looked up by explicit key out of
    `load_pen_profiles()["profiles"][pen_id]`, re-read from disk every time (never via the
    `"active"` key, and never from a repo-root shadow copy).
  * The generic dispatcher passes a `profile` BLOB, not an id. `_pen_id_from_profile`
    recovers the id by VALUE match against a fresh disk read, and `_resolve_pen`
    re-resolves and hard-asserts that id's fields from disk. Nothing trusts the blob's own
    field values; a blob matching no id (stale or foreign) is refused. This keeps the
    pen named in the dispatcher's response header, the plotted legend, the info dict
    (`pen_id` / `pen_label`) and the g-code in agreement, so a sheet is never emitted
    with one pen's Z under another pen's name.
  * `_resolve_pen` asserts that the requested id exists and that pen_mm, z_touch_abs,
    z_down, draw_speed, pen_offset_x and pen_offset_y are all PRESENT and non-None, and
    records them in the info JSON. A pen with no `z_touch_abs` of its own (never
    calibrated, e.g. `orenz_nero_03`) fails loudly before any g-code is written, rather
    than falling back to calibration.json's global z_touch_abs (which `resolve_z_touch`
    would otherwise supply) -- a silent fallback height would misreport the pen's width.

LADDER (13 pitches total):
  * Measured-width pens (profile carries `laid_mm` or `core_mm`, e.g. uright_gel):
    `_scale_ladder_for_pen` uses an anchor at 0.4x pen_mm (well below: everything
    overlaps, patch reads solid), an anchor at 2.0x pen_mm (well above: passes plainly
    separate), and 11 ladder points log-spaced strictly between them. Log spacing keeps
    the same proportional resolution near the expected crossing whatever the pen's width.
    For uright_gel (pen_mm=0.255) this gives 0.102 ... 0.510 mm.
  * Pens with no measured laid width (e.g. bic_cristal_blue): `pen_mm` is the BALL
    DIAMETER on a ballpoint, not a laid-width estimate, so scaling by it would bracket
    the very quantity the sheet measures. Such pens instead get `_absolute_ladder`:
    a fixed ABS_LADDER_LO_MM..ABS_LADDER_HI_MM (0.10-2.00 mm) bracket, 13 log-spaced
    points (~28 % steps: a first bracket, not a fine read). The bracket is printed on the
    sheet itself.
  * An explicit per-request `ladder_mm_lo` / `ladder_mm_hi` override applies to any pen
    and wins over both defaults (see `penwidth_wedge_gcode`).

PATCH SIZE SCALES WITH THE LADDER (`_scale_patch_size`). The rule keeps roughly 10 passes
at the ladder's top pitch (`round(anchor_hi_mm * 10, 1)`), floored at 18 mm so no pen gets
a smaller patch than DEFAULT_PATCH_SIZE_MM.

FIXED DESIGN: 0 deg and 45 deg rows, one replicate patch (the ladder's pitch nearest the
pen's nominal width, at 0 deg), randomized patch-to-slot assignment with a fixed seed
(`SEED`), JIS B5 landscape 257x182 mm paper (not ISO), a 10 mm hole keep-out, 5 mm
margins, an isometry-invariant per-element length-match pen-down guard, and a
pitch-realization band classifier that re-derives each patch's pitch from the EMITTED
g-code and flags any deviation over 5 %.

Reached through the generic POST /api/calibration_sheet dispatcher (id="penwidth_wedge"),
which generates the sheet for whichever pen is active. To test a specific pen directly,
call `penwidth_wedge_gcode(out, pen_id="bic_cristal_blue")`.
"""
from __future__ import annotations

import math
import random
import re
import shlex
from pathlib import Path

import vpype as vp
import vpype_cli

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
# convention as calibration_dropout.py / calibration_singlepass.py: an
# environment without reach_profiles falls back to a different default reach
# profile and would plot against the wrong envelope). This is about the PLOTTER,
# not the pen, and does not vary by pen_id. Tied to the author's measured
# machine geometry.
# ---------------------------------------------------------------------------
EXPECT_REACH_ID = "plotter_2003_v32"
EXPECT_BBOX = (3.0, 42.5, 255.0, 254.5)

# Default pen, used by every no-argument call site (the generic
# /api/calibration_sheet dispatcher, tests).
DEFAULT_PEN_ID = "uright_gel"

REQUIRED_PEN_FIELDS = ("label", "pen_mm", "z_touch_abs", "z_down", "draw_speed",
                       "pen_offset_x", "pen_offset_y")

# ---------------------------------------------------------------------------
# Paper / keep-out (mm). Same convention and numbers as calibration_dropout.py
# / calibration_penwidth.py / calibration_singlepass.py.
# ---------------------------------------------------------------------------
PAPER_W_MM, PAPER_H_MM = 257.0, 182.0
HOLE_KEEPOUT_MM = 10.0
SHEET_MARGIN_MM = 5.0
_PAD_MM = 3.0
_USABLE_W = PAPER_W_MM - 2 * SHEET_MARGIN_MM
_USABLE_H = PAPER_H_MM - HOLE_KEEPOUT_MM - 2 * SHEET_MARGIN_MM

# ---------------------------------------------------------------------------
# Pitch ladder scaling -- see module docstring "LADDER". The multipliers
# (0.4x / 2.0x) and ladder length (11) are fixed by the sheet design.
# ---------------------------------------------------------------------------
LADDER_LO_MULT = 0.4
LADDER_HI_MULT = 2.0
LADDER_N = 11
ANGLES = (0, 45)

# ---------------------------------------------------------------------------
# ABSOLUTE bracket mode, for a pen with NO MEASURED laid width (`laid_mm` /
# `core_mm` both absent from its profile, e.g. bic_cristal_blue). The scaled
# ladder uses `pen_mm`, which for a ballpoint is the BALL DIAMETER (module
# docstring), not a laid-width estimate; bracketing the quantity it stands in for
# would be circular. For example a ladder of 0.4x-2.0x of pen_mm=1.0 runs
# 0.457-1.749mm with its "must read SOLID" anchor at 0.400mm, yet a BIC pen still
# leaves a white sliver at 0.12mm pitch, so the true crossing lies below that
# anchor and the scaled ladder cannot find it.
#
# So for such a pen an ABSOLUTE range is bracketed instead of a multiple of the
# barrel: 0.10-2.00mm, 13 log-spaced points (~28% steps -- deliberately coarse, a
# first bracket not a fine read). It can be overridden per request
# (`ladder_mm_lo`/`ladder_mm_hi`, see `penwidth_wedge_gcode`). A pen WITH a
# measured `laid_mm`/`core_mm` (uright_gel) keeps `_scale_ladder_for_pen`
# unless a caller explicitly passes ladder_mm_lo/hi (which then wins for ANY pen:
# an explicit request overrides the profile-derived default).
# ---------------------------------------------------------------------------
ABS_LADDER_LO_MM = 0.10
ABS_LADDER_HI_MM = 2.00

DEFAULT_PATCH_SIZE_MM = 18.0
# Target pass count at the ladder's own top pitch. 9.0 keeps the 18mm floor
# exactly sufficient for the widest registered pen (BIC Cristal, anchor_hi=2.0mm
# -> 18/2.0+1=10 passes) so its footprint still fits the default placement
# offset under the live reach envelope; growth only engages for a pen nominally
# wider than that (anchor_hi > 2.0mm). A target of 10 would grow the BIC patch to
# 20mm and overrun the reach envelope by 4.3mm at the default offset.
PATCH_PASSES_AT_TOP = 9.0

LABEL_SIZE_PX = 13
LABEL_GAP_MM = 1.5

SEED = 20260910           # fixed seed so patch placement is reproducible
GRID_COLS, GRID_ROWS = 9, 3   # 27 slots for 27 patches, exact fit.

CELL_PAD_MM = 2.0
CELL_GAP_X, CELL_GAP_Y = 4.0, 7.0
LEGEND_GAP_MM = 5.0


def _scale_ladder_for_pen(pen_mm: float, n_ladder: int = LADDER_N,
                          lo_mult: float = LADDER_LO_MULT, hi_mult: float = LADDER_HI_MULT):
    """13 pitches total: anchor_lo (lo_mult * pen_mm), `n_ladder` log-spaced
    intermediate points, anchor_hi (hi_mult * pen_mm). Log spacing keeps
    proportional resolution near the expected crossing regardless of the
    pen's nominal width -- see module docstring. Returns
    (anchor_lo_mm, ladder_mm_list, anchor_hi_mm), each rounded to 0.001mm
    (1 micron)."""
    n_pts = n_ladder + 2
    ratio = hi_mult / lo_mult
    pts = [round(lo_mult * (ratio ** (i / (n_pts - 1))) * pen_mm, 3) for i in range(n_pts)]
    return pts[0], pts[1:-1], pts[-1]


def _absolute_ladder(lo_mm: float, hi_mm: float, n_ladder: int = LADDER_N):
    """ABSOLUTE bracket mode -- see the module-level comment above
    ABS_LADDER_LO_MM. 13 pitches total: anchor_lo=lo_mm, `n_ladder`
    log-spaced intermediate points, anchor_hi=hi_mm -- NOT scaled by any
    pen's pen_mm. Same log-spacing/rounding convention as
    `_scale_ladder_for_pen`. Returns (anchor_lo_mm, ladder_mm_list,
    anchor_hi_mm), each rounded to 0.001mm (1 micron)."""
    if not (lo_mm > 0 and hi_mm > lo_mm):
        raise RuntimeError(
            f"penwidth_wedge: ladder bracket must satisfy 0 < lo < hi, got "
            f"lo_mm={lo_mm} hi_mm={hi_mm}")
    n_pts = n_ladder + 2
    ratio = hi_mm / lo_mm
    pts = [round(lo_mm * (ratio ** (i / (n_pts - 1))), 3) for i in range(n_pts)]
    return pts[0], pts[1:-1], pts[-1]


def _scale_patch_size(anchor_hi_mm: float, floor_mm: float = DEFAULT_PATCH_SIZE_MM,
                      passes_target: float = PATCH_PASSES_AT_TOP) -> float:
    """Patch size grows with the ladder so the top-of-ladder pitch still
    yields roughly `passes_target` passes across the patch -- see module
    docstring "PATCH SIZE SCALES WITH THE LADDER". Never smaller than
    DEFAULT_PATCH_SIZE_MM (18mm)."""
    return max(floor_mm, round(anchor_hi_mm * passes_target, 1))


def _pen_id_from_profile(profile: dict):
    """Recover the pen_id key that CURRENTLY owns this exact profile blob, by
    re-reading pen_profiles.json (never trusting the caller's dict on its own
    -- see "PEN SELECTION" in the module docstring). Returns None if
    the blob matches no id (stale/foreign blob) or more than one
    (ambiguous)."""
    profiles = load_pen_profiles().get("profiles", {})
    matches = [pid for pid, p in profiles.items() if p == profile]
    return matches[0] if len(matches) == 1 else None


def _resolve_pen(pen_id: str):
    """Resolve `pen_id` by EXPLICIT KEY out of load_pen_profiles() -- never
    via the `"active"` key (see module docstring). Hard-asserts every field
    this sheet depends on is present and non-None, and returns
    (pen_id, pen_dict) unmodified so callers can read/report the pen's own
    current values rather than a hardcoded snapshot."""
    profiles = load_pen_profiles()
    pmap = profiles.get("profiles", {})
    if pen_id not in pmap:
        raise RuntimeError(
            f"penwidth_wedge: pen id {pen_id!r} not found in pen_profiles.json. "
            f"Known ids: {sorted(pmap)}")
    pen = pmap[pen_id]
    missing = [f for f in REQUIRED_PEN_FIELDS if pen.get(f) is None]
    if missing:
        raise RuntimeError(
            f"penwidth_wedge: pen id {pen_id!r} is missing required field(s) {missing} "
            f"(present keys: {sorted(pen)}). Refusing to generate a wedge sheet against an "
            f"incompletely-calibrated pen -- e.g. orenz_nero_03 has no z_touch_abs and must "
            f"fail here, not silently plot at some fallback height.")
    return pen_id, pen


def _text_polylines_mm(s: str, size_px=LABEL_SIZE_PX):
    """Hershey-font text via vpype's own `text` command (native, no new font
    dependency). Must be called AFTER vp.config_manager.load_config_file --
    same requirement as every other gwrite call in this module. Returns
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


def _patch_hatch_0(size, pitch):
    """Axis-aligned isolated passes, alternating start edge (LR/RL) so a
    fresh-pen-down dry start (calibration_deposition's finding) does not
    concentrate on one side of every patch."""
    lines = []
    y, k = 0.0, 0
    while y <= size + 1e-9:
        pts = [(0.0, y), (size, y)] if k % 2 == 0 else [(size, y), (0.0, y)]
        lines.append(pts)
        y += pitch
        k += 1
    return lines


def _patch_hatch_45(size, pitch):
    """Diamond-clipped 45-degree isolated passes (slope +1 lines, y=x+c,
    clipped to [0,size]x[0,size]). Perpendicular spacing between adjacent
    lines is delta_c/sqrt(2); choosing delta_c=pitch*sqrt(2) makes that
    spacing exactly `pitch`. Corner slivers under 1mm are dropped -- they are
    dots, not passes, and would bias the length-based pass-count check."""
    lines = []
    dc = pitch * math.sqrt(2.0)
    c = -size + dc / 2.0
    k = 0
    while c < size:
        x_lo = max(0.0, -c)
        x_hi = min(size, size - c)
        if x_hi - x_lo > 1e-9:
            length = (x_hi - x_lo) * math.sqrt(2.0)
            if length >= 1.0:
                p0 = (x_lo, x_lo + c)
                p1 = (x_hi, x_hi + c)
                pts = [p0, p1] if k % 2 == 0 else [p1, p0]
                lines.append(pts)
                k += 1
        c += dc
    return lines


def _poly_length_mm(pts):
    L = 0.0
    for i in range(len(pts) - 1):
        L += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
    return L


def _build_slots(all_pitches, replicate_pitch, replicate_angle):
    """27 patch specs shuffled into grid slots by a fixed seed.
    Pure, no vpype dependency -- safe to call before config is loaded.
    `all_pitches` is [anchor_lo, *ladder, anchor_hi] (13 values)."""
    anchor_lo, anchor_hi = all_pitches[0], all_pitches[-1]
    patch_specs = []
    for angle in ANGLES:
        for pitch in all_pitches:
            microns = round(pitch * 1000)
            label = f"{microns}u {angle}d"
            patch_specs.append({"pitch_mm": pitch, "angle_deg": angle, "label": label,
                                 "replicate": False, "role": (
                                     "anchor_solid" if pitch == anchor_lo else
                                     "anchor_striped" if pitch == anchor_hi else "ladder")})
    _rep_microns = round(replicate_pitch * 1000)
    patch_specs.append({"pitch_mm": replicate_pitch, "angle_deg": replicate_angle,
                         "label": f"{_rep_microns}u {replicate_angle}d r",
                         "replicate": True, "role": "replicate"})
    assert len(patch_specs) == GRID_COLS * GRID_ROWS == 27, len(patch_specs)

    rng = random.Random(SEED)
    slot_order = list(range(len(patch_specs)))
    rng.shuffle(slot_order)
    slots = [patch_specs[i] for i in slot_order]
    for slot_idx, spec in enumerate(slots):
        spec["slot_idx"] = slot_idx
        spec["grid_row"] = slot_idx // GRID_COLS
        spec["grid_col"] = slot_idx % GRID_COLS
    return slots


def _build_elements(pen_id, pen_mm, patch_size_mm, all_pitches, replicate_pitch, replicate_angle,
                    legend_suffix=""):
    """Builds the ordered element list and footprint. Must run AFTER
    vp.config_manager.load_config_file (uses vpype `text`). Returns
    (elements, patch_records, footprint_w, footprint_h, grid_dims). ORDER IS
    PRESERVED end to end (no linesort/linemerge/reloop) -- each element
    becomes exactly one pen-down cycle in the gcode, in this exact order.

    `legend_suffix` (default ""): extra ASCII text appended to the printed
    legend, used by the absolute-bracket ladder mode to print the bracket ON THE
    SHEET ITSELF, because a sheet whose ladder cannot be read off the sheet
    cannot be interpreted later."""
    slots = _build_slots(all_pitches, replicate_pitch, replicate_angle)
    anchor_lo = all_pitches[0]

    label_cache = {}
    for spec in slots:
        if spec["label"] not in label_cache:
            label_cache[spec["label"]] = _text_polylines_mm(spec["label"])
    max_label_w = max(w for _p, w, _h in label_cache.values())
    max_label_h = max(h for _p, _w, h in label_cache.values())

    cell_w = max(patch_size_mm, max_label_w) + CELL_PAD_MM
    cell_h = patch_size_mm + LABEL_GAP_MM + max_label_h + CELL_PAD_MM

    legend_text = (f"PITCH IN MICRONS {round(anchor_lo * 1000)}u={anchor_lo}mm  "
                   f"0d=horizontal 45d=diagonal  r=replicate  "
                   f"{pen_id} pen_mm={pen_mm} 2026-09-12"
                   + (f"  {legend_suffix}" if legend_suffix else ""))
    legend_polys, legend_w, legend_h = _text_polylines_mm(legend_text, size_px=10)

    footprint_w = max(GRID_COLS * cell_w + (GRID_COLS - 1) * CELL_GAP_X, legend_w)
    footprint_h = legend_h + LEGEND_GAP_MM + GRID_ROWS * cell_h + (GRID_ROWS - 1) * CELL_GAP_Y

    if footprint_w + 2 * _PAD_MM > _USABLE_W:
        raise RuntimeError(
            f"penwidth_wedge: sheet footprint width {footprint_w + 2 * _PAD_MM:.1f}mm exceeds "
            f"the JIS B5 landscape usable width {_USABLE_W:.1f}mm for pen {pen_id!r} "
            f"(patch_size={patch_size_mm}mm)")
    if footprint_h + 2 * _PAD_MM > _USABLE_H:
        raise RuntimeError(
            f"penwidth_wedge: sheet footprint height {footprint_h + 2 * _PAD_MM:.1f}mm exceeds "
            f"the JIS B5 landscape usable height {_USABLE_H:.1f}mm for pen {pen_id!r} "
            f"(patch_size={patch_size_mm}mm)")

    elements = []
    legend_x = (footprint_w - legend_w) / 2.0 if legend_w < footprint_w else 0.0
    for pts in legend_polys:
        placed = [(x + legend_x, y) for x, y in pts]
        elements.append({"pts": placed, "kind": "legend"})

    grid_y0 = legend_h + LEGEND_GAP_MM
    patch_records = []
    for spec in slots:
        col, row = spec["grid_col"], spec["grid_row"]
        cx = col * (cell_w + CELL_GAP_X)
        cy = grid_y0 + row * (cell_h + CELL_GAP_Y)
        px0 = cx + (cell_w - patch_size_mm) / 2.0
        py0 = cy

        if spec["angle_deg"] == 0:
            passes = _patch_hatch_0(patch_size_mm, spec["pitch_mm"])
        else:
            passes = _patch_hatch_45(patch_size_mm, spec["pitch_mm"])

        patch_elem_start = len(elements)
        for pass_i, pts in enumerate(passes):
            placed = [(x + px0, y + py0) for x, y in pts]
            elements.append({"pts": placed, "kind": "patch_fill",
                              "slot_idx": spec["slot_idx"], "pass_i": pass_i,
                              "angle_deg": spec["angle_deg"], "pitch_mm": spec["pitch_mm"]})
        patch_elem_end = len(elements)

        label_polys, label_w, _label_h = label_cache[spec["label"]]
        lx0 = cx + (cell_w - label_w) / 2.0
        ly0 = py0 + patch_size_mm + LABEL_GAP_MM
        for pts in label_polys:
            placed = [(x + lx0, y + ly0) for x, y in pts]
            elements.append({"pts": placed, "kind": "label", "slot_idx": spec["slot_idx"]})

        patch_records.append({**spec, "n_passes": len(passes),
                              "elem_range": [patch_elem_start, patch_elem_end],
                              "origin_mm": [round(px0, 3), round(py0, 3)]})

    frame = rect_outline(-2.0, -2.0, footprint_w + 4.0, footprint_h + 4.0)
    for pts in frame:
        elements.append({"pts": pts, "kind": "frame"})

    return elements, patch_records, footprint_w, footprint_h, {
        "cell_w_mm": round(cell_w, 2), "cell_h_mm": round(cell_h, 2)}


# ---------------------------------------------------------------------------
# Post-emission validation. Isometry-invariant 1:1 length-match check per
# element (covers every kind, including frame/label -- a cross-element drag
# of any kind shows up here) plus an angle-deviation check against
# {0,45,90,135} degrees for patch-fill strokes. `pendown_segments_examined`
# is asserted non-zero so the checks cannot pass vacuously.
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


def _pendown_check(out_gcode, z_down_abs, elements, n_pendowns_expected, pendown_max_mm):
    strokes = _parse_strokes(out_gcode, z_down_abs)
    n_pendown_segments = sum(len(s["pts"]) - 1 for s in strokes)
    if n_pendown_segments <= 0:
        raise RuntimeError(
            "penwidth_wedge: pendown_segments_examined is ZERO -- the guard never ran "
            "(a vacuous pass). Do not plot this file.")
    if len(strokes) != n_pendowns_expected:
        raise RuntimeError(
            f"penwidth_wedge: stroke count mismatch: emitted {len(strokes)} pen-down "
            f"cycles, expected {n_pendowns_expected} (one per design element) -- gcode "
            f"structure does not match the design 1:1; do not trust downstream "
            f"measurements. Do not plot this file.")

    worst_len = 0.0          # worst single G1 move across PATCH_FILL elements only
    worst_len_any = 0.0
    n_len_violations = 0
    length_mismatches = []
    angle_devs = []
    for i, (elem, stroke) in enumerate(zip(elements, strokes)):
        expected_len = _poly_length_mm(elem["pts"])
        actual_len = _poly_length_mm(stroke["pts"])
        seg_lens = [math.hypot(stroke["pts"][j + 1][0] - stroke["pts"][j][0],
                                stroke["pts"][j + 1][1] - stroke["pts"][j][1])
                    for j in range(len(stroke["pts"]) - 1)]
        max_seg = max(seg_lens) if seg_lens else 0.0
        worst_len_any = max(worst_len_any, max_seg)
        if elem["kind"] == "patch_fill":
            worst_len = max(worst_len, max_seg)
            if max_seg > pendown_max_mm:
                n_len_violations += 1
        if abs(actual_len - expected_len) > max(0.05, 0.01 * expected_len):
            length_mismatches.append({"elem_idx": i, "kind": elem["kind"],
                                       "expected_mm": round(expected_len, 3),
                                       "actual_mm": round(actual_len, 3)})
        if elem["kind"] == "patch_fill" and len(stroke["pts"]) == 2:
            (x0, y0), (x1, y1) = stroke["pts"]
            ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
            dev = min(abs(ang - a) for a in (0.0, 45.0, 90.0, 135.0, 180.0))
            angle_devs.append(dev)

    if n_len_violations != 0:
        raise RuntimeError(
            f"penwidth_wedge: {n_len_violations} patch-fill pen-down move(s) exceed "
            f"{pendown_max_mm}mm (worst {worst_len:.1f}mm) -- possible cell-to-cell drag. "
            f"Do not plot this file.")
    if length_mismatches:
        raise RuntimeError(
            f"penwidth_wedge: {len(length_mismatches)} element(s) have machine-length != "
            f"design-length beyond tolerance: {length_mismatches[:5]}. Do not plot this file.")
    max_angle_dev = max(angle_devs) if angle_devs else 0.0
    if max_angle_dev >= 0.5:
        raise RuntimeError(
            f"penwidth_wedge: patch-fill stroke angle deviates {max_angle_dev:.2f}deg from "
            f"the nearest axis-aligned/diagonal direction -- unexpected geometry. "
            f"Do not plot this file.")

    return {
        "pendown_segments_examined": n_pendown_segments,
        "n_strokes": len(strokes),
        "n_pendowns_expected": n_pendowns_expected,
        "pendown_travel_violations": n_len_violations,
        "max_pendown_mm_patch_fill": round(worst_len, 2),
        "max_pendown_mm_any_kind": round(worst_len_any, 2),
        "length_mismatches": length_mismatches,
        "max_patch_fill_angle_deviation_deg": round(max_angle_dev, 4),
    }, strokes


def _perp_distance(p0, p1, q):
    x0, y0 = p0
    x1, y1 = p1
    qx, qy = q
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return 0.0
    return abs(dx * (y0 - qy) - (x0 - qx) * dy) / L


def _pitch_realization(patch_records, strokes):
    pitch_table = []
    worst_dev_pct = 0.0
    for rec in patch_records:
        a, b = rec["elem_range"]
        fill_strokes = [strokes[i] for i in range(a, b)]
        if len(fill_strokes) < 2:
            continue
        dists = []
        for i in range(len(fill_strokes) - 1):
            s0, s1 = fill_strokes[i], fill_strokes[i + 1]
            p0, p1 = s0["pts"][0], s0["pts"][-1]
            q = s1["pts"][0]
            dists.append(_perp_distance(p0, p1, q))
        realized = sum(dists) / len(dists)
        commanded = rec["pitch_mm"]
        dev_pct = 100.0 * (realized - commanded) / commanded
        worst_dev_pct = max(worst_dev_pct, abs(dev_pct))
        pitch_table.append({
            "slot_idx": rec["slot_idx"], "label": rec["label"], "role": rec["role"],
            "angle_deg": rec["angle_deg"], "commanded_pitch_mm": commanded,
            "realized_pitch_mm": round(realized, 4),
            "deviation_pct": round(dev_pct, 2),
            "n_passes": len(fill_strokes),
            "min_gap_mm": round(min(dists), 4), "max_gap_mm": round(max(dists), 4),
            "flag_over_5pct": abs(dev_pct) > 5.0,
        })
    n_flagged = sum(1 for r in pitch_table if r["flag_over_5pct"])
    return pitch_table, worst_dev_pct, n_flagged


# ---------------------------------------------------------------------------
# Emission -- single Z / single draw speed (no sweep: pitch is pure geometry,
# not a hardware setting), same single-gwrite-call pattern as
# calibration_dropout.dropout_gcode.
# ---------------------------------------------------------------------------
def penwidth_wedge_gcode(out_gcode: Path, pen_mm=None, z_up=3.5,
                         travel_speed=8000, plunge_speed=500, lift_speed=2000,
                         offset_x=None, offset_y=None, end_policy="release",
                         quiet_fans=True, profile=None, pen_id=None,
                         ladder_mm_lo=None, ladder_mm_hi=None):
    """Emit the full pitch-wedge sheet as ONE g-code file, ONE gwrite call
    (no sweep, no splice).

    Pen selection: the generic dispatcher never passes an explicit `pen_id`; it
    passes the resolved ACTIVE (or explicitly requested) pen as a `profile`
    blob, so honouring the dispatcher's pen means recovering an id from that
    blob. `_pen_id_from_profile` does that by VALUE match against a fresh
    `load_pen_profiles()` read (never trusting the blob's fields directly);
    the id it returns is then re-resolved from disk by `_resolve_pen`.
    Precedence: an explicit `pen_id` kwarg always wins (direct-call convention,
    e.g. `penwidth_wedge_gcode(out, pen_id="bic_cristal_blue")`); if both
    `pen_id` and `profile` are given they must name the same pen or this
    refuses; if only `profile` is given its matching id is used; if neither is
    given, DEFAULT_PEN_ID ("uright_gel") is used.

    `ladder_mm_lo`/`ladder_mm_hi`: an EXPLICIT per-request override of the pitch
    bracket -- see the ABS_LADDER_* comment above and `_absolute_ladder`. Both
    must be given together (0 < lo < hi) or neither; giving one without the
    other refuses. When given, this bracket is used for ANY pen (measured or
    not), taking precedence over both the profile-scaled and the
    unmeasured-default bracket. When NEITHER is given: a pen with a measured
    laid width (`laid_mm` or `core_mm` present on its profile, e.g. uright_gel)
    uses `_scale_ladder_for_pen`; a pen with NEITHER field (e.g.
    bic_cristal_blue) uses the absolute ABS_LADDER_LO_MM..ABS_LADDER_HI_MM
    (0.10-2.00mm) bracket, because for such a pen `pen_mm` is not a laid-width
    estimate (ball diameter, on a ballpoint)."""
    if pen_id is not None and profile is not None:
        blob_id = _pen_id_from_profile(profile)
        if blob_id != pen_id:
            raise RuntimeError(
                f"penwidth_wedge: explicit pen_id={pen_id!r} conflicts with the profile "
                f"blob passed in (blob matches {blob_id!r}) -- refusing rather than "
                f"guessing which pen was intended.")
    elif pen_id is None and profile is not None:
        blob_id = _pen_id_from_profile(profile)
        if blob_id is None:
            raise RuntimeError(
                "penwidth_wedge: the profile blob passed in does not match any pen "
                "currently in pen_profiles.json by value -- refusing (stale-blob "
                "hazard: the caller's profile dict may be out of date).")
        pen_id = blob_id
    pen_id = pen_id or DEFAULT_PEN_ID
    ppmm = 1.0 / PX2MM
    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()

    reach0 = active_reach_profile(cal)
    if reach0["id"] != EXPECT_REACH_ID or tuple(round(v, 1) for v in reach0["bbox"]) != EXPECT_BBOX:
        raise RuntimeError(
            f"Runtime reach envelope mismatch: got id={reach0['id']!r} bbox={reach0['bbox']}, "
            f"expected id={EXPECT_REACH_ID!r} bbox={EXPECT_BBOX}. Refusing to generate against "
            f"an unexpected envelope.")

    pen_id, pen = _resolve_pen(pen_id)
    pen_label = pen["label"]
    pen_mm_nominal = float(pen["pen_mm"])
    z_down = float(pen["z_down"])
    draw_speed = float(pen["draw_speed"])
    pen_offset_x_field = float(pen["pen_offset_x"])
    pen_offset_y_field = float(pen["pen_offset_y"])

    # z_touch_abs is required PRESENT on the profile itself (asserted in
    # _resolve_pen) -- resolve_z_touch's global-cal fallback is deliberately
    # NOT relied on here (see module docstring): this sheet exists to measure
    # a SPECIFIC pen, and a silent fallback height would misreport it.
    z_touch, z_touch_source = resolve_z_touch(cal, pen)

    reach = pen_reach_profile(reach0, pen)
    if pen_mm is None:
        pen_mm = pen_mm_nominal

    # resolve_pen_offset() = GLOBAL (calibration.json mount) + PER-PEN (this
    # profile's own pen_offset_x/y) -- see its own docstring. It is NOT
    # expected to equal the per-pen field alone whenever a global mount
    # offset is set; both are recorded in the info dict below rather than
    # asserted equal.
    pen_off_x, pen_off_y = resolve_pen_offset(cal, pen)

    # Ladder mode decision -- see penwidth_wedge_gcode's docstring and the
    # ABS_LADDER_* comment near the constants. `laid_width_measured` is
    # recorded in the info dict regardless of which mode ran (an explicit
    # override does not change what is actually known about the pen).
    laid_width_measured = pen.get("laid_mm") is not None or pen.get("core_mm") is not None
    if ladder_mm_lo is not None or ladder_mm_hi is not None:
        if ladder_mm_lo is None or ladder_mm_hi is None:
            raise RuntimeError(
                "penwidth_wedge: ladder_mm_lo and ladder_mm_hi must be given together "
                f"(got ladder_mm_lo={ladder_mm_lo!r} ladder_mm_hi={ladder_mm_hi!r}).")
        ladder_mode = "absolute_mm"
        ladder_source = "request"
        anchor_lo, ladder, anchor_hi = _absolute_ladder(float(ladder_mm_lo), float(ladder_mm_hi))
    elif laid_width_measured:
        ladder_mode = "pen_mm_multiple"
        ladder_source = "profile_default"
        anchor_lo, ladder, anchor_hi = _scale_ladder_for_pen(pen_mm_nominal)
    else:
        ladder_mode = "absolute_mm"
        ladder_source = "profile_default"
        anchor_lo, ladder, anchor_hi = _absolute_ladder(ABS_LADDER_LO_MM, ABS_LADDER_HI_MM)
    all_pitches = [anchor_lo] + ladder + [anchor_hi]

    if ladder_mode == "pen_mm_multiple":
        # Replicate sits at the ladder value nearest the pen's own nominal
        # width -- i.e. at the expected crossing (0.26mm replicate for a
        # 0.255mm pen).
        replicate_pitch = min(ladder, key=lambda p: abs(p - pen_mm_nominal))
    else:
        # For an absolute bracket, pen_mm is not a laid-width estimate
        # (may be a ball diameter), so anchoring the replicate to it would be
        # circular. Use the ladder's own middle point instead.
        replicate_pitch = ladder[len(ladder) // 2]
    replicate_angle = 0
    patch_size_mm = _scale_patch_size(anchor_hi)
    pendown_max_mm = round(patch_size_mm * math.sqrt(2.0) + 5.0, 1)

    if ladder_mode == "absolute_mm":
        # The bracket is printed whenever this mode ran; "UNMEASURED laid
        # width" is only true (and only printed) when the profile actually
        # has no laid_mm/core_mm -- an explicit ladder_mm_lo/hi override can
        # force absolute mode on a pen that DOES have a measured width, and
        # the legend must not claim otherwise on paper.
        legend_suffix = f"ABS {anchor_lo}-{anchor_hi}mm"
        if not laid_width_measured:
            legend_suffix += " UNMEASURED laid width"
    else:
        legend_suffix = ""

    elements, patch_records, footprint_w, footprint_h, grid_dims = _build_elements(
        pen_id, pen_mm_nominal, patch_size_mm, all_pitches, replicate_pitch, replicate_angle,
        legend_suffix=legend_suffix)
    n_pendowns_expected = len(elements)

    PLACEMENT_MARGIN_MM = 20.0
    pad = _PAD_MM
    canvas_w_mm, canvas_h_mm = footprint_w + 2 * pad, footprint_h + 2 * pad

    offset_x = reach["bbox"][0] + PLACEMENT_MARGIN_MM if offset_x is None else float(offset_x)
    if offset_y is None:
        # Same vertical_flip handling as calibration_penwidth.py's offset_y
        # default / calibration_dropout.py's dropout_gcode: elegoo_cc's gwrite
        # profile mirrors Y about max(canvas_w_mm, canvas_h_mm) for a landscape
        # canvas, not about the canvas height.
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
    pendown_check, strokes = _pendown_check(out_gcode, z_down_abs, elements, n_pendowns_expected,
                                            pendown_max_mm)
    pitch_table, worst_dev_pct, n_flagged = _pitch_realization(patch_records, strokes)

    prev_svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_paths_svg(prev_svg, W, H, float(pen_mm) * ppmm, polylines=placed)
    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(prev_svg))} color {PREVIEW_COLOR} '
                          f'write {shlex.quote(str(tmp_prev))}')

    return {"kind": "penwidth_wedge", "pen_mm": pen_mm, "pen_id": pen_id,
            "pen_label": pen_label, "pen_mm_nominal": pen_mm_nominal,
            "footprint_mm": [round(canvas_w_mm, 1), round(canvas_h_mm, 1)],
            "usable_mm": [round(_USABLE_W, 1), round(_USABLE_H, 1)],
            "offset_mm": [offset_x, offset_y],
            "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
            "z_down": z_down, "z_down_abs": round(z_down_abs, 3), "draw_speed": draw_speed,
            "pen_offset_mm": [pen_offset_x_field, pen_offset_y_field],
            "resolved_pen_offset_mm": [pen_off_x, pen_off_y],
            "reach_id": reach0["id"], "reach_bbox": reach0["bbox"],
            "n_pendown_expected_total": n_pendowns_expected,
            "n_pendown_actual_total": pendown_check["n_strokes"],
            "pendown_count_matches": pendown_check["n_strokes"] == n_pendowns_expected,
            "pendown_segments_examined": pendown_check["pendown_segments_examined"],
            "deepest_z_down_abs": round(z_down_abs, 3),
            "safety_selfcheck": safety, "pendown_check": pendown_check,
            "pitch_realization": pitch_table,
            "pitch_realization_worst_abs_deviation_pct": round(worst_dev_pct, 2),
            "pitch_realization_n_flagged_over_5pct": n_flagged,
            "preview": prev.name, "svg_path": str(prev_svg), "svg_wh_px": [W, H],
            "pattern_mm": [round(canvas_w_mm, 1), round(canvas_h_mm, 1)],
            "place": {"x": offset_x, "y": offset_y, "w": canvas_w_mm, "h": canvas_h_mm},
            "z_up_abs": round(z_touch + float(z_up), 3),
            "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
            "grid": {"cols": GRID_COLS, "rows": GRID_ROWS, **grid_dims},
            "seed": SEED,
            "anchor_lo_mm": anchor_lo, "anchor_hi_mm": anchor_hi, "ladder_mm": ladder,
            "ladder_lo_mult": LADDER_LO_MULT if ladder_mode == "pen_mm_multiple" else None,
            "ladder_hi_mult": LADDER_HI_MULT if ladder_mode == "pen_mm_multiple" else None,
            "ladder_mode": ladder_mode, "ladder_source": ladder_source,
            "ladder_mm_lo": anchor_lo, "ladder_mm_hi": anchor_hi,
            "laid_width_measured": laid_width_measured,
            "angles_deg": list(ANGLES),
            "replicate": {"pitch_mm": replicate_pitch, "angle_deg": replicate_angle},
            "patch_size_mm": patch_size_mm,
            "pendown_max_mm": pendown_max_mm,
            "slot_to_patch": [{"slot_idx": r["slot_idx"], "grid_row": r["grid_row"],
                               "grid_col": r["grid_col"], "pitch_mm": r["pitch_mm"],
                               "angle_deg": r["angle_deg"], "label": r["label"],
                               "role": r["role"]} for r in patch_records]}
