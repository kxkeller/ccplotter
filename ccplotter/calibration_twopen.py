"""ccplotter/calibration_twopen.py -- two-pen registration and ink-wedge sheet "M10-REG v1".

What it measures: how far the second pen (black) lands from the first pen (blue) after the
paper stays put but the pen is swapped, and how each pen's ink density responds to hatch pitch.
How to read it: on each vernier find the index where the blue and black lines coincide; the
displacement is 0.1 mm times that index (0.05 mm times it for the fine pair) -- see element 1.

One sheet, TWO passes (pass 1 blue, pass 2 black), generated through the same twopen
pipeline (ccplotter/twopen.py) a real two-pen job uses. This module only builds the two
mm page-coordinate stroke layers and their geometry records; twopen turns them into two
complete gcode files (own preamble/postamble, microlift, reach check per pen profile).

Not registered in calibration_registry.py: the registry's CalSheet contract is ONE gcode
file per sheet (generator -> one out path, dispatched by POST /api/calibration_sheet).
This sheet makes two, so it is exposed as an opt-in route, POST /api/twopen_calsheet
(routes/twopen.py), and the registry and its safety sweep never see it.

Elements (all mm, page coords, y down)
  1. REGISTRATION verniers at 9 positions (3x3). Per position: an X-vernier (vertical lines,
     reads X offset) and a Y-vernier (horizontal lines, reads Y offset), each 11 lines, single
     6 mm strokes: BLUE at 1.00 mm pitch (pass 1), BLACK at 0.90 mm pitch (pass 2), centre
     lines nominally coincident. Index i (-5..+5) coincides when blue_i == black_i, i.e. when
     0.1*i == d, so d = 0.1*i mm: 0.1 mm resolution over +-0.5 mm. d is the displacement of
     the BLACK layer relative to the BLUE layer along +x (page right) / +y (page DOWN).
     Tick labels -5, 0, +5 (black) and a coarse crosshair per position in both colours.
  2. INK WEDGES per pen: 5 patches 8x8 mm at 0.96 / 0.48 / 0.24 / 0.24+cross / 0.12 mm,
     45 deg, plus one solid zigzag patch at the pen's fill pitch.
  3. OVERLAY swatches: black hatch (pass 2, 135 deg, 0.48 mm) over blue levels
     0.96 / 0.48 / 0.24 / 0.12 mm (pass 1, 45 deg), plus a blue-only and a black-only reference.
  4. IDENTITY: "M10-REG v1" in black, six black scan fiducials, tick rulers.
Stroke length rule: every stroke is >= 2 mm except the label glyph strokes (Hershey digits and
letters are built from short strokes; labels are never measured). Verniers are 6 mm.
"""
import math
from ccplotter.calibration_penwidth import fiducial as _fiducial
from ccplotter.calibration_pocket import (X0_TIP_DEFAULT, TIP_X_MIN, TIP_X_MAX, TIP_Y_MAX,
                                          PAPER_FRONT_Y_MM)

# PLACEMENT. Same pen-TIP/plate-frame convention as calibration_pocket: page x -> tip X via
# X0_TIP_DEFAULT; the sheet's FRONT-most ink edge (largest page y) -> tip Y = Y_FRONT_TIP_SHEET.
# The punched-hole strip of the paper is at the FRONT, so every mark must be >= 12 mm
# (10 mm strip + 2 mm) from the front paper edge (tip Y 1.5), i.e. tip Y >= 13.5, which is
# the same lower limit as pocket's TIP_Y_MIN. Y_FRONT_TIP_SHEET (14.0) is defined here
# rather than imported from pocket's Y_FRONT_TIP_DEFAULT. The far limit is pocket's TIP_Y_MAX.
FRONT_CLEAR_MM = 12.0
Y_FRONT_TIP_SHEET = 14.0
TIP_Y_MIN_SHEET = PAPER_FRONT_Y_MM + FRONT_CLEAR_MM            # 13.5
TIP_WINDOW = {"x": [TIP_X_MIN, TIP_X_MAX], "y": [TIP_Y_MIN_SHEET, TIP_Y_MAX]}
PAGE_H_MM = 170.0                                              # vpype page height: gwrite flips about it

SHEET_ID = "M10-REG v1"
PAPER = "250x170"                 # vpype page; the ink lives in CONTENT (237 x 156, pocket's 237 wide layout)
CONTENT = (5.0, 5.0, 242.0, 161.0)
V_LEN = 6.0
V_N = 5                           # indices -5..+5
V_PITCH_BLUE = 1.00
V_PITCH_BLACK = 0.90
V_PITCH_BLUE_FINE = 0.50          # fine pair; a 0.30 mm pitch merges into one blob when rendered, 0.50 / 0.45 stays resolvable
V_PITCH_BLACK_FINE = 0.45
MIN_STROKE_MM = 2.0
PATCH_MM = 8.0
PATCH_GAP = 3.0
LADDER = (0.96, 0.48, 0.24, "x", 0.12)          # "x" = 0.24 + cross-hatch
LADDER_LABELS = ("96", "48", "24", "24X", "12")
OVERLAY_BLUE = (0.96, 0.48, 0.24, 0.12)
OVERLAY_BLACK_PITCH = 0.48
POS_X = (28.0, 120.0, 206.0)
POS_Y = (22.0, 80.0, 138.0)
LABEL_CAP_MM = 2.0
LABEL_FONT = "futural"


# ------------------------------------------------------------------ vernier math
def vernier_lines(cx, cy, axis, colour, offset=0.0, fine=False):
    """11 single 6 mm strokes. axis "x": vertical lines at x = cx + i*pitch (+offset), centred
    on cy (reads X offset). axis "y": horizontal lines at y = cy + i*pitch (+offset), centred
    on cx. `offset` is a synthetic displacement of the layer (mm, +x right / +y down)."""
    if fine:
        pitch = V_PITCH_BLUE_FINE if colour == "blue" else V_PITCH_BLACK_FINE
    else:
        pitch = V_PITCH_BLUE if colour == "blue" else V_PITCH_BLACK
    out = []
    for i in range(-V_N, V_N + 1):
        p = i * pitch + offset
        if axis == "x":
            out.append([(cx + p, cy - V_LEN / 2), (cx + p, cy + V_LEN / 2)])
        else:
            out.append([(cx - V_LEN / 2, cy + p), (cx + V_LEN / 2, cy + p)])
    return out


def vernier_position(line, axis):
    return line[0][0] if axis == "x" else line[0][1]


def read_vernier(blue, black, axis, step_mm=0.1):
    """Index (-5..+5) of the black line that best coincides with the blue line of the SAME
    index, and the displacement it implies: d = black - blue of that pair = -(index error);
    for indices where pairs coincide, d_mm = 0.1 * index in the sheet's sign convention."""
    best = min(range(len(blue)),
               key=lambda k: abs(vernier_position(blue[k], axis) - vernier_position(black[k], axis)))
    idx = best - V_N
    return idx, round(step_mm * idx, 3)


# ------------------------------------------------------------------ primitives
def _hatch(x0, y0, w, h, pitch, angle_deg):
    """Parallel lines at angle_deg clipped to the rectangle; strokes < MIN_STROKE_MM dropped.
    Direction alternates (serpentine order) so travel between lines is short."""
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    nx, ny = -dy, dx
    cx, cy = x0 + w / 2, y0 + h / 2
    half = (w + h)
    out, k, flip = [], int(-half / pitch) - 1, False
    while k * pitch <= half:
        px, py = cx + nx * k * pitch, cy + ny * k * pitch
        t0, t1 = -1e9, 1e9
        ok = True
        for p, d, lo, hi in ((px, dx, x0, x0 + w), (py, dy, y0, y0 + h)):
            if abs(d) < 1e-12:
                if p < lo - 1e-9 or p > hi + 1e-9:
                    ok = False
            else:
                ta, tb = (lo - p) / d, (hi - p) / d
                t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        if ok and t1 - t0 >= MIN_STROKE_MM:
            seg = [(px + dx * t0, py + dy * t0), (px + dx * t1, py + dy * t1)]
            out.append(seg[::-1] if flip else seg)
            flip = not flip
        k += 1
    return out


def _zigzag(x0, y0, w, h, pitch, angle_deg=45.0):
    """ONE continuous stroke: the hatch lines joined end to end (boustrophedon)."""
    segs = _hatch(x0, y0, w, h, pitch, angle_deg)
    pts = []
    for s in segs:
        pts.extend(s)
    return [pts] if len(pts) >= 2 else []


def _rect(x0, y0, w, h):
    return [[(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h), (x0, y0)]]


def _label(text, cx, y_bottom, cap=LABEL_CAP_MM):
    """Hershey label, ASCII only, centred on cx, baseline at y_bottom (text sits ABOVE it)."""
    from vpype import text as _vt
    if not text.isascii():
        raise ValueError("label must be ASCII")
    hb = _vt.text_line("H", LABEL_FONT, 1.0).bounds()
    scale = cap / (hb[3] - hb[1])
    lc = _vt.text_line(text, LABEL_FONT, 1.0)
    b = lc.bounds()
    xm = (b[0] + b[2]) / 2.0
    return [[((z.real - xm) * scale + cx, (z.imag - hb[3]) * scale + y_bottom) for z in ln]
            for ln in lc.lines]


def _patch(kind, x, y, pitch, angle=45.0):
    if kind == "zigzag":
        return _zigzag(x, y, PATCH_MM, PATCH_MM, pitch, angle)
    segs = _hatch(x, y, PATCH_MM, PATCH_MM, pitch, angle)
    if kind == "cross":
        segs += _hatch(x, y, PATCH_MM, PATCH_MM, pitch, angle + 90.0)
    return segs + _rect(x, y, PATCH_MM, PATCH_MM)[:0]     # no outline: the hatch edge is the patch


# ------------------------------------------------------------------ the sheet
def build_layers(fill_blue=0.12, fill_black=0.12, variant="B"):
    """Returns {"blue": [polylines], "black": [polylines], "labels": [polylines in black],
    "record": {...}} -- the labels are also included in "black"; `record` describes every
    element for tests (positions, pitches, vernier centres)."""
    blue, black, label_polys = [], [], []
    rec = {"sheet": SHEET_ID, "paper": PAPER, "verniers": [], "wedges": {}, "overlays": [],
           "fiducials": [], "rulers": []}

    def lab(text, cx, yb, cap=LABEL_CAP_MM):
        polys = _label(text, cx, yb, cap)
        label_polys.extend(polys)
        black.extend(polys)

    # 1. verniers + crosshairs
    for r, cy in enumerate(POS_Y):
        for c, cx in enumerate(POS_X):
            xc, xy = cx, cy - 11.0            # X-vernier above the crosshair
            yc, yy = cx + 13.0, cy            # Y-vernier to its right
            xb, xk = vernier_lines(xc, xy, "x", "blue"), vernier_lines(xc, xy, "x", "black")
            yb_, yk = vernier_lines(yc, yy, "y", "blue"), vernier_lines(yc, yy, "y", "black")
            xfc, yfc = (cx - 13.0, cy), (cx + 27.0, cy)       # fine pair (0.50 / 0.45): X left of the crosshair, Y right
            xfb, xfk = vernier_lines(*xfc, "x", "blue", fine=True), vernier_lines(*xfc, "x", "black", fine=True)
            yfb, yfk = vernier_lines(*yfc, "y", "blue", fine=True), vernier_lines(*yfc, "y", "black", fine=True)
            blue += xb + yb_ + xfb + yfb
            black += xk + yk + xfk + yfk
            for col, tgt in (("blue", blue), ("black", black)):      # coarse crosshair, both colours
                tgt.append([(cx - 4.0, cy), (cx + 4.0, cy)])
                tgt.append([(cx, cy - 4.0), (cx, cy + 4.0)])
            for idx in (-5, 0, 5):                                   # tick labels (black)
                lab(f"{idx:+d}" if idx else "0", xc + idx * V_PITCH_BLACK, xy - V_LEN / 2 - 1.2, 1.6)
                lab(f"{idx:+d}" if idx else "0", yc + V_LEN / 2 + 3.2, yy + idx * V_PITCH_BLACK + 0.8, 1.6)
            rec["verniers"].append({"pos": [r, c], "center": [cx, cy], "x_vernier_center": [xc, xy],
                                    "y_vernier_center": [yc, yy], "x_fine_center": list(xfc),
                                    "y_fine_center": list(yfc)})

    # 2. ink wedges per pen (blue pass-1 wedge, black pass-2 wedge)
    y_w = 44.0
    for pen, x0, fill, tgt in (("blue", 14.0, fill_blue, blue), ("black", 86.0, fill_black, black)):
        patches = []
        for j, (p, lb) in enumerate(zip(LADDER, LADDER_LABELS)):
            x = x0 + j * (PATCH_MM + PATCH_GAP)
            tgt += (_patch("cross", x, y_w, 0.24) if p == "x" else _patch("hatch", x, y_w, p))
            lab(lb, x + PATCH_MM / 2, y_w - 1.2)
            patches.append({"label": lb, "pitch": 0.24 if p == "x" else p, "cross": p == "x", "origin": [x, y_w]})
        x = x0 + 5 * (PATCH_MM + PATCH_GAP)
        tgt += _patch("zigzag", x, y_w, fill)
        lab("S", x + PATCH_MM / 2, y_w - 1.2)
        patches.append({"label": "S", "pitch": fill, "zigzag": True, "origin": [x, y_w]})
        rec["wedges"][pen] = patches

    # 3. overlay swatches (blue under, black over) + references
    x0 = 158.0
    for j, p in enumerate(OVERLAY_BLUE):
        x = x0 + j * (PATCH_MM + PATCH_GAP)
        blue += _patch("hatch", x, y_w, p)
        black += _patch("hatch", x, y_w, OVERLAY_BLACK_PITCH, 135.0)
        lab(f"B{LADDER_LABELS[LADDER.index(p)]}", x + PATCH_MM / 2, y_w - 1.2)
        rec["overlays"].append({"blue_pitch": p, "black_pitch": OVERLAY_BLACK_PITCH, "origin": [x, y_w]})
    xb_ = x0 + 4 * (PATCH_MM + PATCH_GAP)
    blue += _patch("hatch", xb_, y_w, 0.24)
    lab("BL", xb_ + PATCH_MM / 2, y_w - 1.2)
    xk_ = x0 + 5 * (PATCH_MM + PATCH_GAP)
    black += _patch("hatch", xk_, y_w, OVERLAY_BLACK_PITCH, 135.0)
    lab("BK", xk_ + PATCH_MM / 2, y_w - 1.2)
    rec["overlays"] += [{"blue_only": 0.24, "origin": [xb_, y_w]}, {"black_only": OVERLAY_BLACK_PITCH, "origin": [xk_, y_w]}]

    # 4. identity, fiducials, rulers (all black)
    lab(SHEET_ID, 128.0, 104.0, 4.0)
    lab(f"SHEET {variant}", 128.0, 114.0, 3.0)
    yc_ = POS_Y[-1]
    for i, (fx, fy) in enumerate([(7.0, 7.0), (72.5, 7.0), (236.0, 7.0), (7.0, yc_ + 16.0), (72.5, yc_ + 16.0), (236.0, yc_ + 16.0)]):
        black += _fiducial(fx, fy)
        rec["fiducials"].append({"id": f"fid{i}", "origin": [fx, fy], "side": 3.0})
    # rulers: x along y=yc_+8..yc_+14 (5 mm ticks, long every 10 mm); y along x=5, y 40..140
    ry = yc_ + 8.0
    for k in range(0, 233, 5):
        black.append([(5.0 + k, ry if k % 10 == 0 else ry + 2.0), (5.0 + k, ry + 6.0)])
    for k in range(0, 101, 5):
        black.append([(5.0, 40.0 + k), (5.0 + (6.0 if k % 10 == 0 else 3.0), 40.0 + k)])
    rec["rulers"] = [{"axis": "x", "from": 5.0, "to": 237.0, "tick_every_mm": 5, "long_every_mm": 10},
                     {"axis": "y", "from": 40.0, "to": 140.0, "tick_every_mm": 5, "long_every_mm": 10}]
    rec["variant"] = variant
    rec["label_stroke_count"] = len(label_polys)
    return {"blue": blue, "black": black, "labels": label_polys, "record": rec}


def to_svg(strokes, w=250.0, h=170.0):
    body = "".join('<polyline points="%s"/>' % " ".join("%.3f,%.3f" % (x, y) for x, y in s)
                   for s in strokes if len(s) > 1)
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%gmm" height="%gmm" viewBox="0 0 %g %g" '
            'fill="none" stroke="black" stroke-width="0.3">%s</svg>' % (w, h, w, h, body))


def ink_bbox(L):
    xs = [p[0] for col in ("blue", "black") for s in L[col] for p in s]
    ys = [p[1] for col in ("blue", "black") for s in L[col] for p in s]
    return min(xs), min(ys), max(xs), max(ys)


def placement_offsets(L):
    """(offset_x, offset_y) for twopen so the ink's left edge lands at tip X = X0_TIP_DEFAULT and its
    FRONT edge (page y max) at tip Y = Y_FRONT_TIP_SHEET. gwrite maps page (x, y) to
    tip (x + offset_x, PAGE_H_MM + offset_y - y) (derived from vpype_gcode's whole-page flip and
    verified against the emitted gcode in tests/test_calibration_twopen.py)."""
    x0, _, _, y1 = ink_bbox(L)
    return X0_TIP_DEFAULT - x0, Y_FRONT_TIP_SHEET - PAGE_H_MM + y1


def sheet_body(blue_pen_id, black_pen_id, profiles=None, variant="B", **extra):
    """Request body for twopen.build_twopen. `profiles` = {pen_id: profile} supplies fill_mm
    for the solid patches (defaults 0.12 when absent)."""
    pf = profiles or {}
    fb = float((pf.get(blue_pen_id) or {}).get("fill_mm", 0.12))
    fk = float((pf.get(black_pen_id) or {}).get("fill_mm", 0.12))
    L = build_layers(fb, fk, variant)
    ox, oy = placement_offsets(L)
    body = {"blue_svg": to_svg(L["blue"]), "black_svg": to_svg(L["black"]),
            "blue_pen_id": blue_pen_id, "black_pen_id": black_pen_id, "paper": PAPER,
            "offset_x": ox, "offset_y": oy, "tip_window": TIP_WINDOW}
    body.update(extra)
    return body, L
