"""ccplotter/calibration_fidelity.py -- FIDELITY SHEET.

What it measures: how faithfully the plotter reproduces known darkness and geometry, so
that a scan of the plotted sheet can be calibrated against the commanded g-code. Nine
known-darkness single-direction hatch patches (Zone A, A0-A8), the app's own real routed
solid fill at the same nominal pitch (Zone B, currently omitted -- see "ZONE B STATUS"),
two perpendicular isolated-stroke groups for a scan-scoped pen-width read (Zone C), a
dedicated scale-fiducial frame (Zone D), and blank / saturated controls (Zone E). All
measured patches are 18 mm x 18 mm.

How to read it: scan the sheet, locate the Zone D fiducials for scale, then measure each
Zone A patch's ink coverage inside its inset window (see `inset_window`) and compare it
with the nominal darkness for that pitch. Zone C gives the laid line width (isolated
strokes in 0 and 90 degree groups, so a directional width difference shows). E1 is the
bare-paper floor and E2 the saturated ceiling for the coverage scale. The generator's
returned info dict carries the emitted per-patch line counts and geometry; use those, not
the nominal A0_A8_PITCH_MM table, for analysis.

HARD-CODED MACHINE AND PEN (limitation). The sheet is tied to the author's measured
machine: it refuses to run unless the active reach profile is EXPECT_REACH_ID
("plotter_2003_v32") with bbox EXPECT_BBOX, and it only supports pen profile
EXPECT_PEN_ID ("uright_gel") with the Z values in EXPECT_Z_TOUCH_ABS / EXPECT_Z_DOWN_STOCK.
Per-patch pitch math is derived from that pen's laid width and would mean something else
on another pen. To use another machine or pen, edit the EXPECT_* constants.

LAYOUT. The design fixes budgets (footprint ceilings, gutters, halo) and per-zone geometry
(patch size, hatch pitch and direction, fiducial anchor coordinates) but not a row-by-row
placement. The row assignment here (Zone A one row of 9, Zone B one patch under it,
Zone C's two groups beside Zone E's two controls in one band) is chosen to satisfy
every invariant: footprint within the 226 mm reach ceiling and the usable paper height,
column fiducials down to y=140 mm, row fiducials from x=12 to x=211, 5 mm horizontal /
8 mm vertical gutters, and the keep-clear halo. FOOTPRINT_W/FOOTPRINT_H (214 / 143 mm
content, 220 / 149 mm padded) follow from the fiducial anchors, which are fixed.

ZONE B STATUS. Zone B is the app's own real routed fill, obtained by POSTing to the real
/api/process route (never an in-process composite call), so composite.py is not imported
here. Fetching a Zone B fragment that way (style "tracev3", tone "off", fill_mm 0.25,
pen "uright_gel", a solid black source image) hangs inside the app's /api/process handler
after mask processing, with memory use growing past 2 GB; it reproduced for several
paper sizes and resolutions, which points at tracev3's solid-fill / connector path.
`fetch_zone_b_fragment()` performs the POST and is otherwise complete, but
`fidelity_gcode()` does not call it by default (`include_zone_b=False`), so the emitted
sheet OMITS Zone B. The unresolved hang is a known limitation.

FACTORS AND GEOMETRY:
  Zone A: 9 patches (A0-A8), single-direction (+X only) horizontal hatch at the pitches in
          A0_A8_PITCH_MM, lines centred (first and last line pitch/2 from each edge, see
          hatch_x_centered). Whole-patch measurement only, no sub-tile binning.
  Zone C: two isolated-stroke groups, 0 deg (Y-offset replicates, `stroke_group` from
          calibration_penwidth.py) and 90 deg (X-offset replicates, `stroke_group_x`
          below). `stroke_group` offsets replicates in +Y, which is parallel to a 90 deg
          line's own direction, so at 90 deg every replicate would sit on the same line.
  Zone D: scale fiducials, corner-anchored (`fiducial()` from calibration_penwidth.py).
  Zone E: E1 blank (no ink), E2 saturated (0.15 mm pitch hatch).

OVER-CROP MARGIN / HALO: m=1.5 mm, halo=2.0 mm (m + tol, tol=0.5 mm). The scan analysis
crops each patch inward by m to avoid edge effects, and the halo keeps neighbouring
objects out of that region. Every window / halo / inset number in this module derives
from M_MM / TOL_MM / HALO_MM, not re-typed values.

SAFETY AND GEOMETRY CHECKS. Each raises (RuntimeError or AssertionError) on a violation:
  - `check_halo_clear()`      -- no foreign polyline touches another object's halo
                                 (closed / non-strict boundary).
  - `check_group_invariant()` -- Zone C emitted-geometry invariant.
  - `_assert_pendown_min_y()` -- post-emission check on the lowest pen-down Y.
  - the import-time footprint asserts -- sheet fits the paper and reach ceiling.
  - `_assert_no_pendown_travel()` -- no pen-down move bridges cells.

Reached only through the calibration_registry entry (id="fidelity", ui="generic").
"""
import math
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
from ccplotter.calibration_deposition import _splice_blocks
from ccplotter.calibration_penwidth import stroke_group, fiducial, FIDUCIAL_MM

# ---------------------------------------------------------------------------
# Expected runtime envelope -- re-verified at generation time (same idiom as
# calibration_singlepass.py's EXPECT_* guards). A mismatch means this sheet is not
# safe to generate against whatever pen/reach is currently active, so it refuses
# rather than silently drawing at the wrong Z or against the wrong reach envelope.
# Tied to the author's measured machine geometry.
# ---------------------------------------------------------------------------
EXPECT_REACH_ID = "plotter_2003_v32"
EXPECT_BBOX = (3.0, 42.5, 255.0, 254.5)
EXPECT_PEN_ID = "uright_gel"
EXPECT_Z_TOUCH_ABS = 11.25
EXPECT_Z_DOWN_STOCK = -0.60

# ---------------------------------------------------------------------------
# Paper / keep-out (mm). JIS B5 182x257 landscape -- see calibration_
# penwidth.py's module docstring for why this is NOT config.PAPER_SIZES["b5"].
# ---------------------------------------------------------------------------
PAPER_W_MM, PAPER_H_MM = 257.0, 182.0
HOLE_KEEPOUT_MM = 10.0
SHEET_MARGIN_MM = 5.0

# ---------------------------------------------------------------------------
# Over-crop margin m=1.5mm, tol=0.5mm -> halo=2.0mm. Every window/halo/inset
# figure in this module derives from these three constants.
# ---------------------------------------------------------------------------
M_MM = 1.5
TOL_MM = 0.5
HALO_MM = M_MM + TOL_MM  # 2.0

PATCH_MM = 18.0
GUTTER_X_MM = 5.0   # horizontal gutter between adjacent patches in a row
GUTTER_Y_MM = 8.0   # vertical gap between stacked rows

# Nominal pitch table for A0-A8 -- NOMINAL, not authoritative; the generator's own
# emitted per-patch n_lines/C (returned in fidelity_gcode()'s response) supersede
# this at analysis time.
A0_A8_PITCH_MM = [0.250, 0.300, 0.364, 0.425, 0.510, 0.607, 0.729, 1.020, 1.700]
E2_PITCH_MM = 0.150  # saturated control

LAID_MM_FALLBACK = 0.255  # only used if the resolved profile carries no laid_mm

# Zone C geometry: 10 replicates/group, 2mm pitch, 30mm stroke length ->
# declared bbox 30x18 (0deg) / 18x30 (90deg).
C_N_REPLICATES = 10
C_PITCH_MM = 2.0
C_STROKE_LEN_MM = 30.0
C_MIN_SEP_MULT = 4.0  # min pairwise distance >= 4 x laid_mm

TICK_H = 2.0

# ---------------------------------------------------------------------------
# Line-placement convention and draw-direction pin: horizontal lines, first/last
# centred pitch/2 from each edge, ALWAYS +X (x1 > x0). NOT
# calibration_plots.hatch_h -- that helper starts a line AT y0 and loops
# `while y <= y0+h`, giving floor(h/pitch)+1 lines with one sitting on the patch
# edge, which would bias the coverage of the patch.
# ---------------------------------------------------------------------------
def hatch_x_centered(x0, y0, w, h, pitch):
    n = int(math.floor(h / pitch + 1e-9))
    return [[(x0, y0 + pitch / 2 + k * pitch), (x0 + w, y0 + pitch / 2 + k * pitch)]
            for k in range(n)]


def assert_uniform_draw_direction(polys, label):
    """Every Zone A/E hatch line runs +X. Checked on the actual emitted polyline
    list, not trusted from the pitch/loop logic that produced it."""
    bad = [p for p in polys if p[-1][0] <= p[0][0]]
    if bad:
        raise RuntimeError(
            f"{label}: {len(bad)} hatch line(s) do not run left-to-right (x1<=x0) -- "
            "the uniform-draw-direction requirement is violated.")


def stroke_group_x(x0, y0, angle_deg, n, length, pitch):
    """Replicate offset along +X instead of `stroke_group`'s +Y.
    `stroke_group` offsets every replicate by `pitch` in Y regardless of
    angle; at angle=90 a stroke's own direction IS +Y, so the offset runs
    parallel to the stroke itself and every replicate lands on the same
    line (min pairwise distance 0). Offsetting in X instead keeps the 90deg
    group's replicates a true `pitch` apart."""
    rad = math.radians(angle_deg)
    dx, dy = length * math.cos(rad), length * math.sin(rad)
    return [[(x0 + k * pitch, y0), (x0 + dx + k * pitch, y0 + dy)] for k in range(n)]


def _bbox(polys):
    xs = [x for p in polys for x, y in p]
    ys = [y for p in polys for x, y in p]
    return min(xs), min(ys), max(xs), max(ys)


def check_group_invariant(polys, declared_w, declared_h, laid_mm, label):
    """Checked on EMITTED GEOMETRY, applied identically to both Zone C groups via
    this one shared function (auditing only the 90deg helper and leaving
    `stroke_group`'s own path unchecked would be an asymmetric guard).

    1. Minimum pairwise distance between any two replicate segments >=
       4 x laid_mm. A group whose replicates are all collinear returns 0
       here and fails immediately.
    2. Realized bounding box equals the declared extent (catches a bug that
       preserves min-distance but silently changes footprint).
    """
    from shapely.geometry import LineString
    lines = [LineString(p) for p in polys]
    min_mm = float("inf")
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            d = lines[i].distance(lines[j])
            if d < min_mm:
                min_mm = d
    threshold = C_MIN_SEP_MULT * laid_mm
    if min_mm < threshold:
        raise RuntimeError(
            f"{label}: min pairwise replicate distance {min_mm:.4f}mm < "
            f"required {threshold:.4f}mm ({C_MIN_SEP_MULT}x laid_mm={laid_mm}mm) -- "
            "replicates have collapsed onto (or too near) each other.")
    x0, y0, x1, y1 = _bbox(polys)
    w, h = x1 - x0, y1 - y0
    if abs(w - declared_w) > 1e-6 or abs(h - declared_h) > 1e-6:
        raise RuntimeError(
            f"{label}: realized bbox {w:.3f}x{h:.3f}mm != declared "
            f"{declared_w}x{declared_h}mm.")
    return {"min_pairwise_distance_mm": round(min_mm, 4), "realized_bbox_mm": [round(w, 3), round(h, 3)]}


def check_halo_clear(objects):
    """Global keep-clear halo, geometric check on EMITTED polylines, CLOSED
    (non-strict) convention: a foreign polyline that merely TOUCHES another
    object's halo boundary counts as a violation (shapely's `.intersects()`
    treats boundary-touching as True, which is the closed convention).

    `objects`: list of dicts {"id": str, "bbox": (x0,y0,x1,y1), "polys": [...]}
    (bbox is the object's OWN true extent, not pre-expanded).
    Raises on the FIRST violation found; returns a summary dict otherwise.
    """
    from shapely.geometry import LineString, box
    checked = 0
    for obj in objects:
        x0, y0, x1, y1 = obj["bbox"]
        halo_box = box(x0 - HALO_MM, y0 - HALO_MM, x1 + HALO_MM, y1 + HALO_MM)
        for other in objects:
            if other["id"] == obj["id"]:
                continue
            for poly in other["polys"]:
                line = LineString(poly)
                checked += 1
                if halo_box.intersects(line):
                    raise RuntimeError(
                        f"halo violation: {other['id']!r}'s geometry intersects "
                        f"(closed/non-strict) {obj['id']!r}'s halo box "
                        f"({halo_box.bounds}) -- foreign ink inside a measurement "
                        "window would be counted as that patch's landed ink.")
    return {"objects_checked": len(objects), "segments_checked": checked, "violations": 0}


def inset_window(pitch_mm):
    """Inset formula: floor((18-2m)/pitch) x pitch, m=1.5mm -> floor(15/pitch) x
    pitch. Side is <= 18-2m (equality holds exactly when 15/pitch is an integer,
    e.g. A0/A1/E2)."""
    bound = PATCH_MM - 2 * M_MM
    n = math.floor(bound / pitch_mm + 1e-9)
    side = n * pitch_mm
    assert side <= bound + 1e-9, "inset window must never exceed 18-2m"
    return side


# ---------------------------------------------------------------------------
# Footprint. Column / row fiducial anchors are corner-anchored fiducial()
# calls, so the footprint follows their realized extents, not the nominal anchor
# coordinates. FOOTPRINT_W/H (214 / 143mm) equal the column / row bound because
# the content rows below were sized to fit inside it exactly.
# ---------------------------------------------------------------------------
DY_FIDUCIALS = [(2.5, 0.0), (2.5, 35.0), (2.5, 70.0), (2.5, 105.0), (2.5, 140.0)]
DX_FIDUCIALS = [(12.0, 2.5), (62.0, 2.5), (112.0, 2.5), (162.0, 2.5), (211.0, 2.5)]

# Content rows (this module's own placement, see module docstring):
ROW_A_XY = (12.0, 12.0)                          # Zone A: 9 patches, 1 row
ROW_B_XY = (12.0, 12.0 + PATCH_MM + GUTTER_Y_MM)  # Zone B: 1 patch under A0
ROW_CE_Y = ROW_B_XY[1] + PATCH_MM + GUTTER_Y_MM    # Zone C/E shared band

ZONE_A_W = len(A0_A8_PITCH_MM) * PATCH_MM + (len(A0_A8_PITCH_MM) - 1) * GUTTER_X_MM
C0_XY = (12.0, ROW_CE_Y)
C0_W, C0_H = C_STROKE_LEN_MM, (C_N_REPLICATES - 1) * C_PITCH_MM  # 30 x 18
C90_XY = (C0_XY[0] + C0_W + GUTTER_X_MM, ROW_CE_Y)
C90_W, C90_H = (C_N_REPLICATES - 1) * C_PITCH_MM, C_STROKE_LEN_MM  # 18 x 30
CE_BAND_H = max(C0_H, C90_H)  # 30mm
_e_y = ROW_CE_Y + (CE_BAND_H - PATCH_MM) / 2.0  # vertically centered E1/E2
E1_XY = (C90_XY[0] + C90_W + GUTTER_X_MM, _e_y)
E2_XY = (E1_XY[0] + PATCH_MM + GUTTER_X_MM, _e_y)

FOOTPRINT_W = 214.0   # D-Y column (x=0) to D-X row right edge (211+3)
FOOTPRINT_H = 143.0   # D-Y column realized bottom (140+3)

_PAD_MM = 3.0
_USABLE_W = PAPER_W_MM - 2 * SHEET_MARGIN_MM
_USABLE_H = PAPER_H_MM - HOLE_KEEPOUT_MM - 2 * SHEET_MARGIN_MM
# Reach ceiling: 232mm reachable in X, 226mm with 3mm pad.
_FOOTPRINT_W_MAX_REACH = 226.0

assert FOOTPRINT_W + 2 * _PAD_MM <= _USABLE_W, (
    f"fidelity sheet footprint width {FOOTPRINT_W + 2 * _PAD_MM:.1f}mm exceeds "
    f"the JIS B5 landscape usable width {_USABLE_W:.1f}mm.")
assert FOOTPRINT_H + 2 * _PAD_MM <= _USABLE_H, (
    f"fidelity sheet footprint height {FOOTPRINT_H + 2 * _PAD_MM:.1f}mm exceeds "
    f"the JIS B5 landscape usable height {_USABLE_H:.1f}mm.")
assert FOOTPRINT_W + 2 * _PAD_MM <= _FOOTPRINT_W_MAX_REACH, (
    f"fidelity sheet footprint width {FOOTPRINT_W + 2 * _PAD_MM:.1f}mm exceeds "
    f"the {_FOOTPRINT_W_MAX_REACH}mm reach-derived ceiling.")

# All Zone A/E content sits within the D-Y column's own y-span (0..140mm) and
# the D-X row's own x-span (12..211mm) -- checked once here at import time.
assert C0_XY[0] >= DX_FIDUCIALS[0][0], "Zone C0 x-origin falls left of the D-X row's own span"
assert E2_XY[0] + PATCH_MM <= FOOTPRINT_W, "Zone E2 right edge exceeds the footprint width"
assert ROW_CE_Y + CE_BAND_H <= DY_FIDUCIALS[-1][1] + FIDUCIAL_MM, (
    "Zone C/E band bottom exceeds the D-Y column's own realized extent")


def build_zone_a():
    """Nine A0-A8 patches, one row. Whole-patch only, no sub-tile
    binning. Returns (polys, per_patch_spec_list)."""
    polys = []
    specs = []
    x0, y0 = ROW_A_XY
    for i, pitch in enumerate(A0_A8_PITCH_MM):
        px = x0 + i * (PATCH_MM + GUTTER_X_MM)
        py = y0
        lines = hatch_x_centered(px, py, PATCH_MM, PATCH_MM, pitch)
        assert_uniform_draw_direction(lines, f"Zone A patch A{i}")
        polys += lines
        n_lines = len(lines)
        c_frac = n_lines * LAID_MM_FALLBACK * PATCH_MM / (PATCH_MM * PATCH_MM)
        inset = inset_window(pitch)
        specs.append({
            "id": f"A{i}", "bbox_mm": [px, py, px + PATCH_MM, py + PATCH_MM],
            "pitch_mm": pitch, "n_lines": n_lines, "commanded_C": round(c_frac, 4),
            "overcrop_window_mm": [px - M_MM, py - M_MM, px + PATCH_MM + M_MM, py + PATCH_MM + M_MM],
            "inset_window_side_mm": round(inset, 4),
            "inset_window_mm": [
                round(px + (PATCH_MM - inset) / 2, 4), round(py + (PATCH_MM - inset) / 2, 4),
                round(px + (PATCH_MM - inset) / 2 + inset, 4), round(py + (PATCH_MM - inset) / 2 + inset, 4)],
        })
    return polys, specs


def build_zone_c():
    """Two perpendicular isolated-stroke groups (0deg and 90deg)."""
    g0 = stroke_group(C0_XY[0], C0_XY[1], 0, C_N_REPLICATES, C_STROKE_LEN_MM, C_PITCH_MM)
    g90 = stroke_group_x(C90_XY[0], C90_XY[1], 90, C_N_REPLICATES, C_STROKE_LEN_MM, C_PITCH_MM)
    return g0, g90


def build_zone_e():
    """E1 blank (no ink -- an empty outline for scan registration only; a
    bare-paper control), E2 saturated (0.15mm pitch hatch)."""
    e1_polys = []  # deliberately empty: E1 must carry NO ink to be a blank control
    e2_lines = hatch_x_centered(E2_XY[0], E2_XY[1], PATCH_MM, PATCH_MM, E2_PITCH_MM)
    assert_uniform_draw_direction(e2_lines, "Zone E2")
    e1_inset = PATCH_MM - 2 * M_MM  # plain (18-2m) square, no pitch to derive it from
    e2_inset = inset_window(E2_PITCH_MM)
    e1_spec = {
        "id": "E1", "bbox_mm": [E1_XY[0], E1_XY[1], E1_XY[0] + PATCH_MM, E1_XY[1] + PATCH_MM],
        "overcrop_window_mm": [E1_XY[0] - M_MM, E1_XY[1] - M_MM,
                                E1_XY[0] + PATCH_MM + M_MM, E1_XY[1] + PATCH_MM + M_MM],
        "inset_window_side_mm": round(e1_inset, 4),
        "inset_window_mm": [  # centred plain square
            round(E1_XY[0] + M_MM, 4), round(E1_XY[1] + M_MM, 4),
            round(E1_XY[0] + M_MM + e1_inset, 4), round(E1_XY[1] + M_MM + e1_inset, 4)],
    }
    e2_spec = {
        "id": "E2", "bbox_mm": [E2_XY[0], E2_XY[1], E2_XY[0] + PATCH_MM, E2_XY[1] + PATCH_MM],
        "pitch_mm": E2_PITCH_MM, "n_lines": len(e2_lines),
        "overcrop_window_mm": [E2_XY[0] - M_MM, E2_XY[1] - M_MM,
                                E2_XY[0] + PATCH_MM + M_MM, E2_XY[1] + PATCH_MM + M_MM],
        "inset_window_side_mm": round(e2_inset, 4),
        "inset_window_mm": [
            round(E2_XY[0] + (PATCH_MM - e2_inset) / 2, 4), round(E2_XY[1] + (PATCH_MM - e2_inset) / 2, 4),
            round(E2_XY[0] + (PATCH_MM - e2_inset) / 2 + e2_inset, 4),
            round(E2_XY[1] + (PATCH_MM - e2_inset) / 2 + e2_inset, 4)],
    }
    return e1_polys, e2_lines, e1_spec, e2_spec


def build_zone_d():
    """Scale fiducials, corner-anchored at DY_FIDUCIALS / DX_FIDUCIALS.
    Returns (polys, dy_specs, dx_specs)."""
    polys = []
    dy_specs, dx_specs = [], []
    for x0, y0 in DY_FIDUCIALS:
        polys += fiducial(x0, y0)
        dy_specs.append({"xy_mm": [x0, y0], "size_mm": FIDUCIAL_MM})
    for x0, y0 in DX_FIDUCIALS:
        polys += fiducial(x0, y0)
        dx_specs.append({"xy_mm": [x0, y0], "size_mm": FIDUCIAL_MM})
    return polys, dy_specs, dx_specs


def fetch_zone_b_fragment(base_url, pen_id=EXPECT_PEN_ID, fill_mm=0.25, timeout=200):
    """Zone B: route through the real POST /api/process, never an in-process
    _composite() call, so composite.py is never imported by this module. NOT
    CALLED by fidelity_gcode() by default -- see the "ZONE B STATUS" section of
    this module's docstring. Kept complete and callable so the hang can be
    reproduced independently of this generator, and so Zone B can be wired in
    with a one-line change once that is fixed.

    Returns (gcode_text, response_json, post_body) on success; raises on
    any non-200 or malformed response -- never silently degrades."""
    import requests
    from PIL import Image
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        img_path = Path(td) / "zoneb_solid.png"
        Image.new("RGB", (68, 68), (0, 0, 0)).save(img_path)  # 68px @ PX2MM ~= 18mm
        with open(img_path, "rb") as f:
            up = requests.post(f"{base_url}/api/add_drawing",
                                files={"file": ("zoneb_solid.png", f)}, timeout=30)
        up.raise_for_status()
        item_id = up.json()["id"]

    body = {
        "job_type": "gcode", "style": "tracev3", "pen_id": pen_id,
        "tone": "off", "fill_mm": fill_mm, "resolution": 300,
        "connector_distance": 0, "paper": "220x160", "margin": 10,
        "offset_x": 50, "offset_y": 50,
        "z_up": 3.5, "z_down": EXPECT_Z_DOWN_STOCK, "draw_speed": 3000,
        "travel_speed": 5000, "plunge_speed": 500, "lift_speed": 2000,
        "end_policy": "release", "quiet_fans": True,
        "items": [{"id": item_id, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}],
    }
    r = requests.post(f"{base_url}/api/process", json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Zone B /api/process failed: HTTP {r.status_code}: {r.text[:500]}")
    resp = r.json()
    if resp.get("job_type") != "gcode" or "gcode_file" not in resp:
        raise RuntimeError(f"Zone B /api/process returned an unexpected shape: {sorted(resp.keys())}")
    gcode_text = (WORK_DIR / resp["gcode_file"]).read_text(encoding="utf-8")
    return gcode_text, resp, body


def _assert_pendown_min_y(path, draw_z_levels, near_edge_target, tol_mm=0.5):
    """Post-emission min-pen-down-Y check: the lowest-Y pen-DOWN point in the
    emitted file must sit within `tol_mm` of the intended near-edge target.
    Guards against the vertical_flip axis mismatch described in
    calibration_penwidth.py's offset-solve comment."""
    x = y = None
    down = False
    min_y = float("inf")
    with Path(path).open() as fh:
        for line in fh:
            if not line.startswith("G1"):
                continue
            mz = re.search(r"Z(-?[\d.]+)", line)
            if mz:
                down = round(float(mz.group(1)), 2) in draw_z_levels
                continue
            mx = re.search(r"X(-?[\d.]+)", line)
            my = re.search(r"Y(-?[\d.]+)", line)
            nx = float(mx.group(1)) if mx else x
            ny = float(my.group(1)) if my else y
            if down and ny is not None:
                min_y = min(min_y, ny)
            x, y = nx, ny
    if min_y == float("inf"):
        raise RuntimeError("no pen-down motion found in emitted g-code.")
    if abs(min_y - near_edge_target) > tol_mm:
        raise RuntimeError(
            f"min pen-down Y {min_y:.3f}mm is {abs(min_y - near_edge_target):.3f}mm "
            f"from the intended near-edge target {near_edge_target:.3f}mm "
            f"(tolerance {tol_mm}mm) -- the vertical_flip offset solve is wrong.")
    return {"min_pendown_y_mm": round(min_y, 3), "near_edge_target_mm": round(near_edge_target, 3)}


PENDOWN_MAX_MM = round(C_STROKE_LEN_MM + 5.0, 1)  # 35.0, sized for Zone C's 30mm strokes


def _assert_no_pendown_travel(path, draw_z_levels):
    """Same idiom as calibration_penwidth._assert_no_pendown_travel -- catches an
    un-anchored splice dragging the pen between cells."""
    x = y = None
    down = False
    worst = 0.0
    n = 0
    n_examined = 0
    with Path(path).open() as fh:
        for line in fh:
            if not line.startswith("G1"):
                continue
            mz = re.search(r"Z(-?[\d.]+)", line)
            if mz:
                down = round(float(mz.group(1)), 2) in draw_z_levels
                continue
            mx = re.search(r"X(-?[\d.]+)", line)
            my = re.search(r"Y(-?[\d.]+)", line)
            nx = float(mx.group(1)) if mx else x
            ny = float(my.group(1)) if my else y
            if down and x is not None and nx is not None:
                d = math.hypot(nx - x, ny - y)
                n_examined += 1
                worst = max(worst, d)
                if d > PENDOWN_MAX_MM:
                    n += 1
            x, y = nx, ny
    if n:
        raise RuntimeError(
            f"Fidelity sheet has {n} pen-DOWN move(s) longer than {PENDOWN_MAX_MM}mm "
            f"(worst {worst:.1f}mm). Do not plot this file.")
    return {
        "pendown_travel_violations": 0,
        "max_pendown_mm": round(worst, 2),
        "pendown_segments_examined": n_examined,
    }


def fidelity_gcode(out_gcode: Path, pen_mm=None, z_up=3.5, z_down=None,
                   draw_speed=None, travel_speed=8000, plunge_speed=500,
                   lift_speed=2000, offset_x=None, offset_y=None,
                   end_policy="release", quiet_fans=True, profile=None,
                   include_zone_b=False):
    """Emit the fidelity sheet (Zones A/C/D/E; Zone B omitted by default, see
    module docstring "ZONE B STATUS"). Z/reach come from the EXPECT_*
    shadow guards (uright_gel only) -- same convention as
    calibration_singlepass.py, not calibration_penwidth.py's active-profile
    fallback, because this sheet's per-patch pitch math is derived from a
    specific pen's laid_mm and would silently mean something else on any
    other pen."""
    ppmm = 1.0 / PX2MM
    vp.config_manager.load_config_file(str(CONFIG_PATH))
    cal = load_cal()

    reach0 = active_reach_profile(cal)
    if reach0["id"] != EXPECT_REACH_ID or tuple(round(v, 1) for v in reach0["bbox"]) != EXPECT_BBOX:
        raise RuntimeError(
            f"Runtime reach envelope mismatch: got id={reach0['id']!r} bbox={reach0['bbox']}, "
            f"expected id={EXPECT_REACH_ID!r} bbox={EXPECT_BBOX}.")

    profiles = load_pen_profiles()
    if EXPECT_PEN_ID not in profiles.get("profiles", {}):
        raise RuntimeError(f"pen profile {EXPECT_PEN_ID!r} not found in pen_profiles.json")
    if profile is None:
        profile = profiles["profiles"][EXPECT_PEN_ID]
    z_touch, z_touch_source = resolve_z_touch(cal, profile)
    if abs(z_touch - EXPECT_Z_TOUCH_ABS) > 1e-6:
        raise RuntimeError(f"z_touch_abs mismatch for {EXPECT_PEN_ID}: got {z_touch}, "
                           f"expected {EXPECT_Z_TOUCH_ABS}")
    stock_z_down = float(profile.get("z_down", 0.0))
    if abs(stock_z_down - EXPECT_Z_DOWN_STOCK) > 1e-6:
        raise RuntimeError(f"stock z_down mismatch for {EXPECT_PEN_ID}: got {stock_z_down}, "
                           f"expected {EXPECT_Z_DOWN_STOCK}")
    if z_down is None:
        z_down = stock_z_down
    if draw_speed is None:
        draw_speed = float(profile.get("draw_speed", 3000))

    laid_mm = float(profile.get("laid_mm") or profile.get("pen_mm") or LAID_MM_FALLBACK)

    reach = pen_reach_profile(reach0, profile)
    if pen_mm is None:
        pen_mm = float(profile.get("pen_mm", 0.255))

    pen_off_x, pen_off_y = resolve_pen_offset(cal, profile)
    PLACEMENT_MARGIN_MM = 20.0
    pad = _PAD_MM
    canvas_w_mm, canvas_h_mm = FOOTPRINT_W + 2 * pad, FOOTPRINT_H + 2 * pad
    offset_x = reach["bbox"][0] + PLACEMENT_MARGIN_MM if offset_x is None else float(offset_x)
    if offset_y is None:
        # Same vertical_flip handling as calibration_penwidth.py's offset-solve
        # comment: solve directly for the machine offset that lands the near (front)
        # edge of the content at reach.bbox[1]+PLACEMENT_MARGIN_MM, using the ACTUAL
        # flip axis (max(canvas_w_mm, canvas_h_mm) for a landscape canvas) rather
        # than assuming it is canvas_h_mm.
        flip_axis_mm = max(canvas_w_mm, canvas_h_mm)
        near_edge_target = reach["bbox"][1] + PLACEMENT_MARGIN_MM
        comp_off_y = near_edge_target - flip_axis_mm + pad + FOOTPRINT_H
        offset_y = comp_off_y + pen_off_y
    else:
        offset_y = float(offset_y)
    comp_off_x, comp_off_y = offset_x - pen_off_x, offset_y - pen_off_y
    park_x, park_y = reach["safe_park_point"]
    z_travel_abs = z_touch + max(float(cal.get("z_travel", 4.0)), float(z_up))
    near_edge_target = reach["bbox"][1] + PLACEMENT_MARGIN_MM

    W = int(round(canvas_w_mm * ppmm))
    H = int(round(canvas_h_mm * ppmm))

    # ---- build geometry ----
    a_polys, a_specs = build_zone_a()
    c0_polys, c90_polys = build_zone_c()
    e1_polys, e2_polys, e1_spec, e2_spec = build_zone_e()
    d_polys, dy_specs, dx_specs = build_zone_d()

    # ---- Zone C emitted-geometry invariant, both groups ----
    c0_check = check_group_invariant(c0_polys, C0_W, C0_H, laid_mm, "Zone C 0deg group")
    c90_check = check_group_invariant(c90_polys, C90_W, C90_H, laid_mm, "Zone C 90deg group")

    # ---- global halo-clear check on every zone's own polylines ----
    objects = []
    for spec, patch_polys in zip(a_specs, [
            hatch_x_centered(*spec["bbox_mm"][:2], PATCH_MM, PATCH_MM, spec["pitch_mm"])
            for spec in a_specs]):
        objects.append({"id": spec["id"], "bbox": tuple(spec["bbox_mm"]), "polys": patch_polys})
    objects.append({"id": "C0", "bbox": (C0_XY[0], C0_XY[1], C0_XY[0] + C0_W, C0_XY[1] + C0_H), "polys": c0_polys})
    objects.append({"id": "C90", "bbox": (C90_XY[0], C90_XY[1], C90_XY[0] + C90_W, C90_XY[1] + C90_H), "polys": c90_polys})
    objects.append({"id": "E1", "bbox": tuple(e1_spec["bbox_mm"]), "polys": e1_polys})
    objects.append({"id": "E2", "bbox": tuple(e2_spec["bbox_mm"]), "polys": e2_polys})
    for i, (x0, y0) in enumerate(DY_FIDUCIALS):
        objects.append({"id": f"DY{i}", "bbox": (x0, y0, x0 + FIDUCIAL_MM, y0 + FIDUCIAL_MM),
                        "polys": fiducial(x0, y0)})
    for i, (x0, y0) in enumerate(DX_FIDUCIALS):
        objects.append({"id": f"DX{i}", "bbox": (x0, y0, x0 + FIDUCIAL_MM, y0 + FIDUCIAL_MM),
                        "polys": fiducial(x0, y0)})
    halo_check = check_halo_clear(objects)

    all_polys = a_polys + c0_polys + c90_polys + e1_polys + e2_polys + d_polys
    placed = [[((x + pad) * ppmm, (y + pad) * ppmm) for x, y in p] for p in all_polys]

    svg = WORK_DIR / f"{out_gcode.stem}.svg"
    _write_paths_svg(svg, W, H, float(pen_mm) * ppmm, polylines=placed)

    z_down_abs = z_touch + float(z_down)
    vp.config_manager.config["gwrite"]["elegoo_cc"]["default_values"].update({
        "z_up": float(z_up), "z_up_abs": z_touch + float(z_up),
        "z_down": float(z_down), "z_down_abs": z_down_abs,
        "z_touch_abs": z_touch, "z_travel_abs": z_travel_abs,
        "draw_speed": float(draw_speed), "travel_speed": float(travel_speed),
        "plunge_speed": float(plunge_speed), "lift_speed": float(lift_speed),
        "z_auto": "1", "origin_x": park_x, "origin_y": park_y,
        **_gcode_lifecycle_values(end_policy, quiet_fans)})

    tmp = WORK_DIR / f".{out_gcode.stem}.gcode"
    vpype_cli.execute(f'read {shlex.quote(str(svg))} pagesize {W}x{H}px '
                      f'translate -- {comp_off_x}mm {-comp_off_y}mm '
                      f'gwrite --profile elegoo_cc {shlex.quote(str(tmp))}')

    texts = [tmp.read_text(encoding="utf-8")]
    labels = [f"; ZONE A/C/D/E z_down={z_down} z_down_abs={z_down_abs:.2f} draw_speed={draw_speed}"]

    zone_b_info = {"included": False,
                   "reason": "reproducible /api/process hang (see module docstring "
                             "ZONE B STATUS) -- STOP-AND-REPORT, not silently improvised"}
    if include_zone_b:
        raise RuntimeError(
            "Zone B is not wired into fidelity_gcode() -- fetch_zone_b_fragment() is "
            "implemented and callable standalone, but the real /api/process pipeline "
            "hangs on a solid-fill tracev3 request in this environment (evidence in "
            "artifacts/fidelity-sheet-build-2026-09-03/). Diagnosing/fixing that hang "
            "is out of scope for this generator; splicing an unverified fragment in "
            "would be exactly the silent improvisation this task was told to avoid.")

    combined = _splice_blocks(texts, labels, z_touch + float(z_up))
    with atomic_target(out_gcode) as tmp_gcode:
        Path(tmp_gcode).write_text(combined, encoding="utf-8")
        _fix_gwrite_source_comment(tmp_gcode, out_gcode)

    validate_gcode_xy_motion(out_gcode, reach)
    draw_z_set = {round(z_touch + float(z_down), 2)}
    pendown_min_y_check = _assert_pendown_min_y(out_gcode, draw_z_set, near_edge_target)
    pendown_travel_check = _assert_no_pendown_travel(out_gcode, draw_z_set)
    safety = validate_calibration_gcode_safety(out_gcode, z_down_abs)

    prev_svg = WORK_DIR / f"{out_gcode.stem}_pattern.svg"
    _write_paths_svg(prev_svg, W, H, float(pen_mm) * ppmm, polylines=placed)
    prev = WORK_DIR / f"{out_gcode.stem}_preview.svg"
    with atomic_target(prev) as tmp_prev:
        vpype_cli.execute(f'read {shlex.quote(str(prev_svg))} color {PREVIEW_COLOR} '
                          f'write {shlex.quote(str(tmp_prev))}')

    n_lines_total = sum(s["n_lines"] for s in a_specs) + e2_spec["n_lines"] + 2 * C_N_REPLICATES
    stroke_len_total_mm = (sum(s["n_lines"] for s in a_specs) * PATCH_MM
                           + e2_spec["n_lines"] * PATCH_MM
                           + 2 * C_N_REPLICATES * C_STROKE_LEN_MM)

    return {
        "kind": "fidelity", "pen_id": EXPECT_PEN_ID, "pen_mm": pen_mm, "laid_mm": laid_mm,
        "m_mm": M_MM, "tol_mm": TOL_MM, "halo_mm": HALO_MM,
        "footprint_mm": [round(FOOTPRINT_W + 2 * pad, 1), round(FOOTPRINT_H + 2 * pad, 1)],
        # `pattern_mm` is the key the ROUTE reads (routes/pen.py: info.get("pattern_mm"))
        # and the generic UI dereferences unguarded as f.pattern_mm[0]/[1]; omitting it
        # makes the UI fail even though the API answers 200. Same value and expression
        # as calibration_penwidth.py and calibration_singlepass.py; `footprint_mm` is
        # kept because this module's response JSON and measurement scripts read it.
        "pattern_mm": [round(FOOTPRINT_W + 2 * pad, 1), round(FOOTPRINT_H + 2 * pad, 1)],
        "offset_mm": [offset_x, offset_y],
        "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
        "z_down": z_down, "z_down_abs": round(z_down_abs, 3), "draw_speed": draw_speed,
        "zone_a": a_specs, "zone_e1": e1_spec, "zone_e2": e2_spec,
        "zone_c_0deg": c0_check, "zone_c_90deg": c90_check,
        "zone_d_y_fiducials": dy_specs, "zone_d_x_fiducials": dx_specs,
        "zone_b": zone_b_info,
        "halo_check": halo_check,
        "pendown_min_y_check": pendown_min_y_check,
        "pendown_travel_check": pendown_travel_check,
        "n_lines_total": n_lines_total, "stroke_len_total_mm": round(stroke_len_total_mm, 1),
        "safety_selfcheck": safety, "preview": prev.name,
        "svg_path": str(prev_svg), "svg_wh_px": [W, H],
        "place": {"x": offset_x, "y": offset_y,
                  "w": FOOTPRINT_W + 2 * pad, "h": FOOTPRINT_H + 2 * pad},
        "z_up_abs": round(z_touch + float(z_up), 3),
        "end_policy": str(end_policy).lower(), "quiet_fans": bool(quiet_fans),
    }
