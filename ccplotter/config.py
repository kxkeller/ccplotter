import os
from pathlib import Path

from ccplotter import runtime_home as _rh

# ── Runtime home ──────────────────────────────────────────────────────────────
# pen_profiles.json and calibration.json are resolved from the RUNTIME HOME
# (%LOCALAPPDATA%\ccPlotter by default, or the explicit CCPLOTTER_HOME opt-in) and
# NEVER from the process CWD; see ccplotter/runtime_home.py for the full rules. A
# stale CWD copy of either file would silently carry a wrong z_touch_abs. This
# runs BEFORE any directory is created below so a refusal leaves no litter. A
# conflicting CWD copy or an untrusted (Store Python) interpreter raises here, i.e.
# the app/script refuses to start; cal.py re-checks on every access.
RUNTIME_HOME = _rh.runtime_home()
CALIBRATION_PATH = RUNTIME_HOME / "calibration.json"
PEN_PROFILES_PATH = RUNTIME_HOME / "pen_profiles.json"
_rh.guard_startup(RUNTIME_HOME, (CALIBRATION_PATH, PEN_PROFILES_PATH))

# ── Configuration ─────────────────────────────────────────────────────────────

# Printer network address. Unset (the default) means no connection is ever attempted.
PRINTER_IP = os.environ.get("CCPLOTTER_PRINTER_HOST", "").strip()
BED_MAX = 257.0  # Elegoo Carbon spec (sq)
# Legacy rectangular fallback only (machine coords).  The active physical envelope
# is the named polygon persisted in calibration.json and resolved by reach.py.
# These values retain compatibility with older calibration files. They are the
# author's measured pen-reachable extents on one machine (exact values, no
# margin); re-measure them on any other machine.
PEN_MIN_X, PEN_MIN_Y = 34.0, 27.5
PEN_MAX_X, PEN_MAX_Y = 247.0, 250.5
PLOT_W, PLOT_H = PEN_MAX_X - PEN_MIN_X, PEN_MAX_Y - PEN_MIN_Y

# Conversion: vpype default unit is the CSS pixel (96 dpi); G-code needs mm.
PX2MM = 25.4 / 96.0  # vpype works in CSS px @ 96 dpi (not 72 dpi)

# WORK_DIR is the single filesystem origin for every generated file (previews,
# routed SVGs, layout.gcode, src_* caches). It defaults to "work" (relative to
# CWD -- the data dir after the desktop launcher's os.chdir).
# CCPLOTTER_WORK_DIR overrides it so the desktop launcher can hand a second or
# later concurrent instance its own isolated dir (work/inst-<pid>). Two instances
# writing the same work/layout.routed.svg would corrupt it (see
# desktop/ccplotter_app.py acquire_single_instance_lock docstring); separate dirs
# remove that race. The /work/<f> URL is per-instance (each server serves its own
# WORK_DIR on its own port), so the frontend needs no change.
WORK_DIR = Path(os.environ.get("CCPLOTTER_WORK_DIR") or "work")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# DrawingBotV3 watch-folder drop point. Same CWD-relative pattern as
# WORK_DIR/STUDIO_DIR -- resolves to %LOCALAPPDATA%\ccPlotter\dbv3_inbox in
# the packaged desktop app (after os.chdir in desktop/ccplotter_app.py) and
# ./dbv3_inbox in the dev server. DrawingBotV3's SVG export is pointed at this
# folder.
DBV3_INBOX_DIR = Path("dbv3_inbox")
DBV3_INBOX_DIR.mkdir(exist_ok=True)

# Archive subfolder for inbox files that have already been imported into
# work/. A plain SUBFOLDER of the inbox, not a sibling directory --
# DBV3_INBOX_DIR.glob("*.svg") (non-recursive, used by both /api/dbv3_inbox and
# /api/dbv3_inbox_poll) never descends into it, so archived files stop appearing
# in listings without any listing-route change.
DBV3_ARCHIVE_DIR = DBV3_INBOX_DIR / "imported"
DBV3_ARCHIVE_DIR.mkdir(exist_ok=True)

CONFIG_PATH = Path("vpype.toml")
RASTER_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VECTOR_EXTS = {".svg", ".pdf", ".dxf"}

PAPER_SIZES = {
    "a4": (210, 297), "a5": (148, 210), "b5": (176, 250),
    "letter": (215.9, 279.4), "200x200": (200, 200), "180x180": (180, 180),
    "100x150": (100, 150)
}
RENDER_PIPELINE_VER = 9   # bump when render-engine code changes so cached _render_ SVGs regenerate
# PREVIEW_COLOR: the everyday plot preview must read as ink on paper, so tonal
# masses and shadows look as they will on the real plot, not as a bright accent
# hue. It reuses SKETCH_INK (the near-black plot-true ink token) rather than a new
# hex, so the sketch-mode preview and the general preview share one ink colour. A
# light preview colour would hide real defects (thin or missing strokes) from the
# user's eyes.
PREVIEW_COLOR = "#16181d"
SKETCH_INK = "#16181d"    # near-black ink for the plot-true sketch preview
SKETCH_PAPER = "#e7e8ee"  # soft cool "paper" (not harsh pure white — eases the dark-mode clash)

# CONNECTOR_COLOR: the X-ray overlay is baked into the SAME preview SVG as the ink
# strokes and renders over a paper-white surface (assets/templates/index.html
# --canvas). A coral (#ff6b81) measures ~2.2:1 contrast against SKETCH_PAPER,
# under the 3:1 floor for a graphical marker. #c81e3a (deep crimson) measures
# ~4.6:1 against SKETCH_PAPER, keeps the red/danger hue family (a "connector, not
# ink" signal), and stays legible on the paper preview. Display-only overlay --
# never plotted, so it has no effect on G-code.
CONNECTOR_COLOR = "#c81e3a" # X-ray overlay: router's pen-down connector segments
XRAY_LAYER = 999          # preview-only overlay layer; stripped before stats

# Fan-off command set for pen plotting (fans serve no purpose with a pen and just
# run noisily). FIRMWARE-CORRECT for the Elegoo Centauri Carbon (Klipper-based):
# fan indices are M106 (no P)=hotend, P2=part cooling, P3=chamber/case. There is
# NO P0 or P1: this firmware errors on those indices and HALTS the machine, so
# only the indices above are emitted. P3 is the chamber/case fan that is wanted
# off. Pure M-codes (no motion/Z/homing). APPENDED after the park move ONLY --
# never injected into the live print stream (a mid-plot halt risk on this
# firmware). A G-code file made only of bare M-codes with no motion also halts
# this firmware, so these are never sideloaded on their own.
FAN_OFF_GCODE = ["M107", "M106 S0", "M106 P2 S0", "M106 P3 S0"]

# Per-mode working-resolution caps (px, longest side). Lower = fewer/cleaner strokes,
# faster, more plotter-practical. Line-art tracers get a HIGH cap: clean bilevel
# sources trace at up to 2x native (thicker strokes → cleaner skeletons, tiny
# text survives) and must never be silently downscaled below native.
MODE_MAX_RES = {
    "plotready": 4800, "hybrid": 1200, "lineart": 3600, "outline": 1000, "schematic": 1600,
    "sketch": 900, "sketchv2": 4800, "tsp": 700, "flow": 700, "centerline": 1000,
    "tracev3": 2200,
    # Sketch Field (native "Sketch Lines" PFM, opt-in): the brightness map needs
    # TONE, not edge fidelity, so a lower cap suffices.
    "sketchfield": 1200,
}

# "Photo 512" mode (opt-in style "photo512"). NOT a general "photo mode": one
# specific recipe -- plain greyscale photograph, source pre-downsampled to shorter
# side 512px (the deliberate detail reduction the mode's structure depends on; 768+
# measured worse), CLAHE off, no pre-simplification, tone off, engine tracev3,
# everything else stock. This dict is the WHOLE recipe -- the one place it is
# defined. routes/core.py's api_process() merges it into the request body when
# style == "photo512" (then dispatches through the ordinary tracev3 engine), so
# every other request path is unaffected: existing styles never read
# PHOTO512_SOURCE_SIDE_PX or "source_side_cap", and CLAHE/tone here are already
# tracev3's own stock defaults (False / "off"). The only new behaviour is the
# source-side downsample, gated on the internal "this request resolved to
# photo512" flag in routes/core.py (NOT on a raw body key), which sets
# opts["source_side_cap"] only then; it is absent/None for every other request, so
# their output is unchanged.
#
# "resolution": 3000 is not just "high enough to hit the tracev3 cap". composite.py's
# scale_aware_v2 branch (default-on for tracev3, gated on
# item_opts.get("mode")=="tracev3") reads item_opts["resolution"] as `base_res` and
# MULTIPLIES it by a needed-boost factor when the analytic fill_px would otherwise
# quantize too thin (capped at SCALE_AWARE_V2_NATIVE_CAP). That branch also
# overwrites item_opts["resolution"] and sets opts["_tracev3_res_cap_override"],
# which raises render.py's own mode_cap before the source-preprocessing block. So
# whenever the boost fires, a different base_res produces a different working
# resolution, not just a differently-capped one. At the geometry this mode was
# validated at (180x180mm, platinum_preppy 0.275mm) the boost does not fire and
# 2200 vs 3000 make no observable difference, but that equivalence holds only for
# that geometry. Using the validated 3000 literally removes that bound.
PHOTO512_SOURCE_SIDE_PX = 512
PHOTO512_RECIPE = {
    "clahe": False,
    "tone": "off",
    "resolution": 3000,
    "source_side_cap": PHOTO512_SOURCE_SIDE_PX,
    # native_res would read the ORIGINAL full-res source
    # (mask._load_native_gray(raw_path), render.py's tracev3 branch) in PARALLEL
    # with the 512-pre-shrunk `img` this mode depends on; the two are incoherent
    # if both are used. The UI has no control for this key, but a direct API call
    # with style="photo512" + native_res=true could set it, so it is forced False
    # here like clahe/tone above (native_res already defaults False for every
    # tracev3 request).
    "native_res": False,
}

# "B/W photo" mode (opt-in style "bwphoto"): a fixed B/W configuration as a real
# mode -- see ccplotter/bw_photo.py. XDoG p02_e55 at shorter-side 512 (the front
# end, applied in render.py when render_opts["front_end"] == "bw_xdog"), then the
# ordinary tracev3 engine with CLAHE ON and tone OFF. Same dispatch pattern as
# PHOTO512_RECIPE: routes/core.py merges this dict into the request body when
# style == "bwphoto" and then runs the tracev3 engine; no other style reads any of
# it. "front_end" is gated on the internal "resolved to bwphoto" flag in
# routes/core.py, never on a raw caller-supplied body key.
BW_PHOTO_FRONT_END = "bw_xdog"
BW_PHOTO_RECIPE = {
    "clahe": True,
    "tone": "off",
    "resolution": 3000,   # the m09 jobs' literal value (see PHOTO512 note above)
    "native_res": False,
}

# "Line Art / Illustration" style (opt-in style "illustration"): the validated
# processing for ALREADY-CLEAN line art (a cleaned/AI/inked drawing), NOT a photo
# mode. Photo mode is separate and the photo XDoG front end never runs here. Same
# dispatch pattern as BW_PHOTO_RECIPE: routes/core.py merges this dict into the
# request body when style == "illustration" BEFORE the vcache key is built, then
# runs the ordinary tracev3 engine. There is NO front end
# (render_opts["front_end"] stays None) and NO placement field (paper, scale,
# rotation, offsets, keepout are the user's layout and are never touched).
#   clahe True          CLAHE-on mask (keeps fine structure in panels)
#   fill_mode contour   concentric contour passes (NOT zigzag); core_fill "rings" = the
#                       server's non-hybrid core ("contour" core fill; only "hybrid" is
#                       read, anything else is rings -- the literal "contour" is not a
#                       server value, so the real default is pinned)
#   small_detail off    pinned so a UI value cannot leak in
#   vec_merge 0.3       the merge distance
#   tone off / otsu     the tracev3 defaults, pinned so the style is reproducible
#   resolution 2200     BASE resolution only. composite's scale_aware_v2 then picks the
#                       working resolution. It must be 2200, not a free value: with the
#                       1.2px floor binding, the final fill_px = 1.2 * 5480 / base, so
#                       base 2600 gives fill_px 2.53 (a different drawing), and base
#                       5480 defeats the boost entirely (effective_res 2200, fill_px 1.2).
#                       It is also load-bearing for the pitch clamp: composite.py takes
#                       the v2 boost's base_res = the REQUESTED resolution, but the
#                       preliminary trace runs at min(requested,
#                       MODE_MAX_RES["tracev3"] = 2200). The two agree ONLY because this
#                       base is 2200. Changing it without the opt-in `prelim_res_basis`
#                       correction (composite.py) mis-sizes the clamp by requested/2200.
ILLUSTRATION_RECIPE = {
    "clahe": True,
    "tone": "off",
    "resolution": 2200,
    "native_res": False,
    "threshold_mode": "otsu",
    "fill_mode": "contour",
    "core_fill": "rings",
    "small_detail": "off",
    "vec_merge": 0.3,
    "scale_aware_v2": True,    # pins the route's existing default for tracev3 (routes/core.py)
    # PITCH CONTRACT (stated once, here): the pass pitch is held at
    # ILLUSTRATION_PITCH_CLAMP_FRAC x fill_mm at ANY placement; below the 1.2 px
    # trace floor it cannot be held and the danger warning fires. routes/core.py
    # reads this key only when style == "illustration" (it is merged into the body
    # then), composite.py reads it only with render_opts["illustration"]; every
    # other style never sees it. At a placement whose unclamped pitch is already
    # <= fill_mm the clamp is a no-op.
    "illustration_pitch_clamp": True,
}

# Line Art / Illustration pitch clamp (ON in ILLUSTRATION_RECIPE; read ONLY when
# render_opts["illustration"] is set). The clamped physical pass pitch is this
# fraction of the pen's fill_mm -- just under touching (a reference placement had
# 0.1089 mm pitch / 0.12 mm fill = 0.91).
ILLUSTRATION_PITCH_CLAMP_FRAC = 0.9

# Opt-in request overrides honoured ONLY under style == "bwphoto" (routes/core.py
# reads them from the body only when the resolved-bwphoto flag is set; the
# BW_PHOTO_RECIPE defaults above are NOT changed). bw_clahe: "on" (default, the
# recipe's CLAHE at clip 2.5) | "soft" (clip BW_CLAHE_SOFT_CLIP) | "off".
# bw_cut: "otsu" (default) | "mid" (fixed cut at grey 128 after the same pre-blur).
# Soft clip = half the shipped 2.5: CLAHE's local contrast gain is capped at the
# clip limit, so halving it halves the worst-case thin-ink amplification that
# was measured on sub-0.2mm ink (7-50% of it removed),
# while still lifting flat tiles (clip > 1 keeps some equalisation). One global
# value, not tuned per image.
BW_CLAHE_SOFT_CLIP = 1.25

# Hybrid contour fill (opt-in, core_fill="hybrid"): concentric rings stop once the
# remaining interior band's inscribed radius drops below this; the remaining core
# is filled with straight parallel passes. 0.45mm = calibration_pocket.R_HYBRID_MM.
CORE_R_MM_DEFAULT = 0.45
# core_fill="rows" (opt-in): rings until a region's remaining inscribed radius < this, then
# rows in the core of every LARGE region (see ccplotter/core_fill.py). 1.0 mm = about 6 pitches.
ROWS_CORE_R_MM_DEFAULT = 1.0

# Opt-in open-stroke stitching (stitch_mm, ccplotter/stitch.py): the largest endpoint gap a
# request may ask to bridge. 0 / absent = off.
STITCH_MM_MAX = 3.0

# Seed travel_speed is 5000 rather than 8000. Travel is the ONLY move class that
# reaches high speed (measured peak 133mm/s vs 50mm/s for draw), so those long
# inter-stroke moves are the individual bangs, as distinct from the continuous
# buzz of draw motion. Travel is 2.1% of plot time, so softening it costs about
# 3 min on a 293-min plot. It does NOT quiet the buzz: draw motion is 96.1% of
# total velocity change. (Measured on the author's machine.)
# ── Pen refill profiles: per-pen calibration (nominal width, REAL solid-fill
# line width from the calibration plot, Z heights, speeds). fill_mm is the pocket
# pass spacing that plots solid -- the payoff of the calibration session. ──
#
# THE SEED VALUES BELOW ARE THE AUTHOR'S MEASUREMENTS on one machine (one pen
# holder, one printer, one set of pens); they are not universal defaults.
# Re-measure z_touch_abs, fill_mm, pen offsets and speeds on any other machine
# before plotting: a wrong z_touch_abs drives the pen into the bed or draws in the
# air. The seed is written only when pen_profiles.json is missing or unusable.
# PEN_PROFILES_PATH: defined at the top of this file (runtime home) -- NOT CWD-relative.
_PEN_PROFILE_SEED = {
    "active": "energel_05",
    "profiles": {
        "energel_05": {
            "label": "Pentel EnerGel 0.5 (Blue)",
            "pen_mm": 0.5, "fill_mm": 0.25,   # author's measurement: 0.25mm real deposit width from an isolated-line calibration
            # z_touch_abs is PER-PEN: different barrels seat at different heights in
            # the holder, so a single global touch height cannot serve two pens (on
            # the author's machine EnerGel measured 12.05 vs BIC 18.75 -- a 6.7mm
            # difference; using the wrong one either draws in the air or drives the
            # pen into the bed). calibration.json's global z_touch_abs remains the
            # fallback for profiles that predate this field. Re-measure on any other
            # machine.
            "z_touch_abs": 12.05,
            "z_up": 3.5, "z_down": -0.6, "z_micro": 0.6,
            "pen_offset_x": 0.0, "pen_offset_y": 0.0,   # bare refill: no protrusion past the holder datum
            "draw_speed": 3000, "travel_speed": 5000, "plunge_speed": 500, "lift_speed": 2000,
        },
        "platinum_preppy": {
            "label": "Platinum Preppy",
            # z_touch_abs 14.85 is the author's measured contact height, confirmed
            # by plotting. Commanded draw Z is 14.25mm absolute (14.85 - 0.6).
            #
            # z_touch_abs is the CONTACT height and carries no offset of its own;
            # z_down / z_micro / z_up are the offsets relative to it. If z_touch_abs
            # is set too low, the micro-lift lands below true contact, the pen never
            # leaves the page on short hops and drags continuously, and no z_micro
            # value can fix it. A too-low contact height also over-presses the pen,
            # which widens the deposit and lets drag add uncommanded ink, so line
            # width, fill and tone readings taken at the wrong height are suspect.
            #
            # fill_mm 0.275 is the working value, set from repeated plots without a
            # coverage complaint rather than from a calibration-sheet reading; that
            # is stronger evidence for a pitch that has to lay down solid. Re-open
            # it if a plot cracks. pen_mm remains the initial 0.25mm nominal estimate
            # and is unread. pen_mm and fill_mm are NOT the same quantity: fill_mm is
            # the pass PITCH that plots solid, not a line width, so 0.275 says
            # nothing about 0.25.
            "pen_mm": 0.25, "fill_mm": 0.275,
            "z_touch_abs": 14.85,
            # PER-PEN XY offset, mm, composed with calibration.json's GLOBAL
            # holder offset by cal.resolve_pen_offset(). This is a full pen body,
            # not a bare refill, so its tip protrudes ~8mm further toward the FRONT
            # of the machine than the BIC/EnerGel refills in the same holder. Front
            # is -Y, hence -8.0. Uncorrected it plotted 8mm forward of target.
            # Estimated from that misplacement, not measured with calipers, so it is
            # working but unconfirmed. Re-measure on any other machine.
            "pen_offset_x": 0.0, "pen_offset_y": -8.0,
            "z_up": 3.5, "z_down": -0.6, "z_micro": 0.6,
            "draw_speed": 3000, "travel_speed": 5000, "plunge_speed": 500, "lift_speed": 2000,
        },
        # BIC Cristal ballpoints. z_touch_abs 18.75 is the author's measurement,
        # assumed identical for both colors (same barrel).
        #
        # fill_mm 0.19 was the author's re-measure on a clean calibration chart
        # (calibration plots bypass routing entirely, so no pen-down connector
        # segments pad the patches): band 1 (0.128mm) darkest/solid, band 2
        # (0.192mm) marginal with skips, band 3 (0.256mm) clearly showing white. An
        # independently generated wedge agrees: its 0.16mm end read basically solid
        # dark. Two instruments, ~0.16-0.19mm. The seed below carries a tighter
        # 0.12 from a later all-in-one sheet; see each profile's comment.
        #
        # A pass pitch looser than the pen actually covers leaves a real gap between
        # passes (fills crack). 0.19 is the marginal-solid point;
        # SCALE_AWARE_V2_K (0.75) brings the drawn spacing to ~0.14mm, inside the
        # solid band rather than on its edge.
        #
        # Standing caveat: a ballpoint's effective width is somewhat
        # DIRECTION-DEPENDENT and it skips intermittently, so this is a threshold
        # estimate for a pen that is not perfectly consistent, not a constant.
        # "1.0" is the BALL diameter and the point grade is MEDIUM. Ball size is not
        # line width: a medium ballpoint typically lays roughly 0.35-0.45mm of ink.
        # Neither number is measured for these pens yet -- pen_mm is the ball,
        # fill_mm is a pass pitch, and the deposited LINE width is unknown. The
        # line-width sheet (/api/calibration_linewidth) exists to measure it.
        "bic_cristal_black": {
            "label": "BIC Cristal 1.0 Black",
            "pen_mm": 1.0, "fill_mm": 0.12,   # NOT an independent paper measurement of black ink: it inherits the blue re-measure below, since both share one barrel/tip geometry. If black gets its own calibration read, that number wins.
            "z_touch_abs": 18.75,
            "z_up": 3.5, "z_down": -0.6, "z_micro": 0.6,
            "pen_offset_x": 0.0, "pen_offset_y": 0.0,   # bare refill: no protrusion past the holder datum
            "draw_speed": 3000, "travel_speed": 5000, "plunge_speed": 500, "lift_speed": 2000,
        },
        "bic_cristal_blue": {
            "label": "BIC Cristal 1.0 Blue",
            "pen_mm": 1.0, "fill_mm": 0.12,   # author's paper measurement on the all-in-one calibration sheet: 0.125 still left a sliver of white, 0.12 is the best achievable fill; residual white is a hard pen limit, not a spacing bug. Black carries the same 0.12 by inheritance (see above).
            "z_touch_abs": 18.75,
            "z_up": 3.5, "z_down": -0.6, "z_micro": 0.6,
            "pen_offset_x": 0.0, "pen_offset_y": 0.0,   # bare refill: no protrusion past the holder datum
            "draw_speed": 3000, "travel_speed": 5000, "plunge_speed": 500, "lift_speed": 2000,
        },
    }
}

# ── Final-G-code point reduction (DISABLED) ──────────────────────────────────
# Douglas-Peucker tolerance for the emitted G-code, expressed PEN-RELATIVE as a
# multiple of the active pen's fill_mm (= the pass pitch). 0 disables, and 0 is
# the value: THIS FEATURE IS OFF. Do not enable it by editing this number.
#
# Pen-relative, not absolute mm, because the governing scale for fill fidelity is
# the PASS PITCH, not the pen's nominal width. The fill is concentric passes
# fill_mm apart and each pass is simplified independently, so once the tolerance
# approaches the pitch, adjacent passes stop agreeing, cross, and the fill
# collapses into visible polygonal facets. A fixed absolute tolerance would get
# proportionally more destructive on a finer-calibrated pen.
#
# Why it stays off: simplification does not destroy ink, it MOVES it. Adjacent
# fill passes are simplified independently, so gaps open BETWEEN passes. Net
# signed ink change hides this (losses and gains nearly cancel: at 0.10mm, 2.42%
# of pixels lose ink and 1.52% gain it, net -0.90%). The metric that sees it is
# crack fraction at true deposit width, footprint fixed from the unsimplified
# geometry. Measured on one real job (11,719 strokes, 1,139,643 points), with the
# time saved by simplification:
#   tolerance   crack%    points kept   time saved
#   off          3.16        100%          0
#   0.03mm       4.46        77.1%      27.4 min
#   0.04mm       4.84        65.4%      43.0 min
#   0.10mm       9.86        40.5%      78.2 min
# At 0.036mm (0.3 x a 0.12mm fill_mm) that is roughly +30-45% RELATIVE cracking.
# Fill coverage outranks plot time, so do not add a time lever that costs coverage.
#
# The machinery is KEPT because it is correct and inert when off: output is
# byte-identical with it off, calibration_plot/fan_probe are exempt by
# construction, and it runs upstream of validate_gcode_xy_motion. Any future
# variant that leaves fill passes untouched starts from here, but must use crack
# fraction (never netted ink) as the primary metric.
SIMPLIFY_FILL_MULT = 0.0

GRAY_CRUSH_TRIGGER = 0.01   # empirical threshold -- see _gray_crush_metric docstring

NATIVE_CAP = 5200   # threshold ceiling for native-resolution tracing (memory guard).
# Consumed by mask._load_native_gray (native_res threshold_mode's source-loading
# cap). The scale_aware_v2 resolution-floor boost has its own ceiling,
# SCALE_AWARE_V2_NATIVE_CAP, below.

# scale_aware_v2 resolution-floor boost ceiling, separate from NATIVE_CAP. A
# tighter fill_mm needs proportionally MORE working resolution to hold the same
# real pass spacing, so the boost's saturation cliff (the saturation warning in
# stats_warnings.py) falls at smaller placements as fill_mm shrinks. Measured
# (plane fixture, B5 geometry, base_res=2200 -- tracev3's own MODE_MAX_RES, the
# resolution essentially every tracev3 request runs at): at fill_mm=0.19 the cliff
# sits at ~248-250mm placed width, almost exactly B5's long edge, i.e. an ordinary
# full-bleed B5 plot.
#
# The ceiling is not simply raised because the boost has a hard, pen-independent
# bound: needed_boost = SCALE_AWARE_V2_FILL_PX_FLOOR / fill_px_real, and
# fill_px_real is clamped to a 1.2px floor before that division (composite.py).
# So needed_boost never exceeds 3.0/1.2 = 2.5x and the boost never asks for more
# than 2.5 * base_res working px (5500px at base_res=2200). Any ceiling >= 5500
# would make the boost never pin against it for base_res<=2200, so the saturation
# warning could never fire, and that behaviour has to stay testable. The ceiling
# therefore stays STRICTLY BELOW 5500, with headroom above the B5-edge
# requirement (2200 * 2.3934 = 5265.5px).
#
# 5480 clears the B5-edge requirement with ~4% headroom (215px) against
# placement variance, stays 20px under the 5500 bound so larger placements
# (>=260mm on this geometry) still saturate and warn, and costs ~800MB peak RSS at
# 5480x5480 (worst-case square working raster). The 7000-8000px range would cost
# ~1.3-1.6GB for no correctness gain at base_res<=2200, since tracev3's
# working-resolution mode cap already discards any tracing benefit above that.
SCALE_AWARE_V2_NATIVE_CAP = 5480

# Scale-adaptation v2 (opt-in). The earlier scale_aware=on mode used spec-exact
# 0.25mm real spacing at every size; that spacing is what lightens small plots,
# so v2 scales spacing by a constant k instead.
#
# SCALE_AWARE_V2_K is the ratio of drawn pass spacing to the pen's fill_mm.
# Basis:
#
# (a) tracev3's pocket passes are distance-transform LEVEL SETS -- concentric
#     rings following the ink boundary inward, the same geometry as the
#     calibration pack's pocket-filled fill-test squares. The gray step wedge is
#     PARALLEL HATCH, a different geometry, so its readings do not contradict a
#     fill-test-based k.
#
# (b) One fill test (a worn Pentel EnerGel 0.5 refill; re-test with a fresh one)
#     bracketed the solid-fill threshold at a spacing/deposit-width ratio in
#     (0.8, 1.0]: last solid band 0.8x, first band showing white 1.0x. The band
#     spacing makes the bracket +/-0.2 wide, so it cannot be narrowed from that
#     data.
#
# (c) The BIC Cristal result is NOT independent corroboration: its fill_mm was set
#     from the last-solid band of its own ladder, so its ratio is 1.0 by
#     construction. The two pens disagree by 25% (0.8x vs 1.0x); this is one
#     bracket, not two-pen agreement.
#
# (d) Spacing equal to fill_mm (ratio 1.0) sits at or past the bracket's upper
#     edge, which explains white seeping through at that setting.
#
# (e) k = 0.75 is a CHOSEN starting default, not a derived constant. It sits just
#     below the (0.8, 1.0] bracket (the darker, safer side, since lightening small
#     plots is the failure being avoided), and nothing in the evidence picks 0.75
#     over another value below 0.8.
#
# (f) Even at ~128% nominal coverage NEITHER pen's fill test produced a fully
#     solid patch, so geometry alone does not guarantee solidity; ink flow,
#     pressure and direction matter too. k moves real spacing into the bracket;
#     it cannot promise no cracks.
#
# Each pen's own calibration-pack fill test re-measures its solid-band ratio, so
# the 0.8-threshold / 0.75-target relationship can be checked per pen rather than
# assumed to hold pen-to-pen.
SCALE_AWARE_V2_K = 0.75
# Working-resolution floor (v2 only): when the analytically-computed
# fill_px_real would land below this many working px, raise the working
# resolution for that item's FINAL trace pass (bounded by NATIVE_CAP) so the
# distance-transform level-set bands (ccplotter/trace_v3.py) keep enough
# pixel-space precision to avoid banding at large placements. Legacy
# (scale_aware unset) and v1 (scale_aware=True, scale_aware_v2 unset) never
# read this constant.
SCALE_AWARE_V2_FILL_PX_FLOOR = 3.0

# mm-spacing v2 (opt-in `fill_spacing_mm`). Read only when composite.py's
# fill_spacing_mm branch fires. This is the finest spacing the flag is validated
# to resolve reliably. The working-resolution boost that branch applies is sized
# so a HYPOTHETICAL request of exactly this value clears
# SCALE_AWARE_V2_FILL_PX_FLOOR px, not against whatever value is actually
# requested. Keying the boost to a fixed reference means every request from this
# value up to a coarser one shares one working resolution for a given placement,
# so the working-grid mask (and trace_v3.py's dropped_components) does not vary
# with the requested spacing. Requests finer than this value can still land under
# the precision floor; composite.py reports that per call via
# fill_spacing_mm_precision_risk.
#
# This is an OPERATING-RANGE BOUND, not a calibration constant: it marks the
# finest spacing the flag's precision guard is sized to cover, independent of any
# one pen's ink behavior.
#
# Basis: a plotted wedge (45-degree diagonal hatch, bare-paper fraction still
# showing through) read 2.7% at 0.15mm, 9.1% at 0.18mm and 34.8% at 0.20mm,
# bracketing the point where diagonal deposit stops covering paper at roughly
# ~0.17mm. 0.14 sits ~18% below that, chosen with headroom under the
# wedge-measured limit; it does not claim 0.14mm IS a pen's real deposit width.
# (The wedge is the author's, one pen.)
#
# Side effect: because the boost is sized against this ONE fixed reference, it
# pins the working grid to SCALE_AWARE_V2_FILL_PX_FLOOR / FILL_SPACING_MM_FLOOR_REF_MM
# = 3.0 / 0.14 = 21.4 px/mm for every fill_spacing_mm request from 0.14mm up,
# REGARDLESS of the base resolution the request started at (below the boost
# threshold). Measured mm_per_px_real is 0.046662 (== 1/21.4478, up to integer
# boosted_res rounding) at base resolutions 1200 and 2250. The resolution FIELD
# stops mattering in this regime.
FILL_SPACING_MM_FLOOR_REF_MM = 0.14

AI_CONFIG_PATH = Path("ai_config.json")
AI_DEFAULTS = {
    "provider": "openai",        # openai (cloud, no local GPU) | comfy (local ComfyUI)
    "openai_model": "gpt-image-2",   # cleanest/flattest for plotting (respects no-fill); see AI_OPENAI_MODELS
    "openai_quality": "medium",  # low | medium | high  (higher = more detail/cost)
    "comfyui_url": "http://127.0.0.1:8188",
    "checkpoint": "",            # empty = auto-pick the first installed checkpoint
    "steps": 22, "cfg": 7.0,
    "denoise": {"lineart": 0.65, "sketch": 0.72},
    "max_side": 1024,            # SD-friendly working size for the img2img pass
    "timeout_s": 600,
}
# OpenAI image models selectable from the UI (validated server-side). chatgpt-image-
# latest tracks the ChatGPT web UI; gpt-image-2 = newest stable (slower); 1.5/1 are
# lighter/cheaper; mini = cheapest/lowest detail.
AI_OPENAI_MODELS = ["chatgpt-image-latest", "gpt-image-2", "gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini"]

# Model Images/ lives under assets/Model Images/ in the repo. Resolved relative
# to Flask app.root_path (send_from_directory calls in routes/studio.py) or the
# process CWD (bench.py's direct Path use) -- both equal the project root in dev
# and the data dir in the frozen build, so this single relative string must match
# desktop/ccplotter_app.py's seed_defaults copy target
# (data / "assets" / "Model Images").
MODEL_IMAGES_DIR = Path("assets/Model Images")
STUDIO_DIR = WORK_DIR / "studio"
STUDIO_DIR.mkdir(exist_ok=True)
STUDIO_PAIRS = STUDIO_DIR / "pairs.json"
STUDIO_STYLECARD = STUDIO_DIR / "style_card.txt"
