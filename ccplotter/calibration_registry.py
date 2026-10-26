"""ccplotter/calibration_registry.py -- the catalogue of calibration sheets.

Why it exists: a sheet registered here is served by ONE generic handler
(POST /api/calibration_sheet) and is previewable automatically, instead of needing a
bespoke JS handler in assets/templates/index.html for each new sheet.

TWO MECHANISMS COEXIST BY DESIGN. The five bespoke flows -- plot, pack, combo, block,
deposition -- keep their own routes and handlers unchanged; rewriting a working UI to
prove a pattern has regression risk and no user-visible gain. This registry is a
CATALOGUE of every sheet (nothing is invisible), and each entry declares whether the
generic dispatcher can drive it:

  ui="generic"  -- uniform `*_gcode(out_gcode, pen_mm=..., z_up=..., ...)`
                   signature; POST /api/calibration_sheet dispatches it, and
                   the registry-parametrized safety test covers it.
  ui="bespoke"  -- has its own route and its own UI handler. `generator` is
                   None. Listed so the catalogue is complete.

THE PLOT-FROM-CANVAS TRAP, made structural. Every sheet here bakes per-cell `z_down`
and `draw_speed` into SEPARATE spliced gwrite blocks (or pins a single pen/Z). Plotting
one from the canvas would re-emit all its cells at one Z and one speed -- it would look
fine and measure NOTHING. Each entry therefore MUST carry a non-empty `plot_warning`,
validated at IMPORT time: the server refuses to start with a silent-trap sheet
registered, rather than the omission being discovered at render. The generic UI flow
additionally applies `scaleLocked` and a `calScaleNote` UNCONDITIONALLY -- there is no
registry sheet for which canvas flattening is safe, so it is not a per-entry option.

GUARDS LIVE IN THE GENERATORS, NOT HERE. Each generator keeps its own hard
`RuntimeError` shadow guards (resolved profile id, reach bbox, pen, z_touch), its own
`resolve_z_touch`, and its own reach validation. The dispatcher only dispatches; it must
never become the place a guard is skipped. `tests/test_calibration_registry_safety.py`
parametrizes over THIS registry and runs the full safety set against every
`ui="generic"` sheet, so registering a sheet automatically buys it safety coverage and a
generator that skips a guard fails a test instead of reaching a printer.

Limitation: many registered sheets pin EXPECT_REACH_ID ("plotter_2003_v32") and refuse
to run against any other reach profile; they are tied to the author's measured machine
geometry (see each generator's module docstring).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ccplotter.calibration_deposition import deposition_gcode
from ccplotter.calibration_dropout import dropout_gcode
from ccplotter.calibration_fidelity import fidelity_gcode
from ccplotter.calibration_fillmode import fillmode_gcode
from ccplotter.calibration_linestar import linestar_gcode
from ccplotter.calibration_penwidth import penwidth_gcode
from ccplotter.calibration_penwidth_wedge import penwidth_wedge_gcode
from ccplotter.calibration_fragment import fragment_bic_probe
from ccplotter.calibration_uright_short import uright_short_gcode
from ccplotter.calibration_photofe import photofe_bic
from ccplotter.calibration_pocket import pocket_bic, pocket_preppy
from ccplotter.calibration_singlepass import singlepass_gcode
from ccplotter.calibration_solidpitch import solidpitch_gcode


@dataclass(frozen=True)
class CalSheet:
    id: str
    label: str
    blurb: str
    plot_warning: str
    generator: Optional[Callable] = None
    ui: str = "generic"
    # ARCHIVED: hides the sheet from the UI's button list ONLY. The sheet stays
    # fully registered: still importable, still dispatchable by id, and still
    # covered by tests/test_calibration_registry_safety.py's parametrized sweep,
    # which walks SHEETS and not the catalogue. Archiving is decluttering, NOT
    # retirement, and it is reversible by flipping one flag.
    # /api/calibration_sheets?all=1 still returns everything.
    archived: bool = False
    # UI-HIDDEN: omitted from the default /api/calibration_sheets listing ONLY.
    # Still registered, dispatchable by id, and covered by the safety sweep.
    # ?all=1 returns it. Flip to False to restore.
    ui_hidden: bool = False


# NOTE ON WORDING: `plot_warning` is user-facing and says what goes wrong, not
# "do not do this" -- numbers and consequence, so the reason survives even if the
# sentence is skimmed.
#
# CAPTURE RULE: any sheet meant to be photographed/scanned and measured carries a
# short capture line pointing at docs/CAPTURE_RULE.md. "One exposure, whole sheet,
# no HDR" alone is NOT sufficient -- it lets a 240 dpi scan claiming "400 dpi"
# through, which makes patches unmeasurable. New measured sheets should get the
# same short line; see docs/CAPTURE_RULE.md for the full rule and per-sheet
# fiducial notes.
_SHEETS = (
    CalSheet(
        id="singlepass",
        ui_hidden=True,  # hidden from the default UI list; code and route kept
        label="Single-pass landing (speed x pressure)",
        blurb="Does an isolated single pen pass leave a mark? Sweeps draw speed "
              "500/1000/1500 against four pen pressures, with a 3000mm/min "
              "negative-control cell and a must-ink control patch per cell.",
        plot_warning="Plot the linked g-code file directly. Adding this sheet to the canvas "
                     "shows you WHERE it lands, but plotting from the canvas would re-emit "
                     "all 13 cells at one Z and one speed -- which is exactly the variable "
                     "this sheet exists to measure.",
        generator=singlepass_gcode,
        # ARCHIVED: the single-pass question was answered -- single passes land
        # fine (13 cells, zero blank, with the 3000mm/min negative-control cell
        # among the fullest on the sheet).
        archived=True,
        ui="generic",
    ),
    CalSheet(
        id="penwidth",
        ui_hidden=True,  # hidden from the default UI list; code and route kept
        label="Pen width distribution (laid-width tails)",
        blurb="Isolated single strokes across pressure x speed x angle, to measure the "
              "SPREAD of the pen's laid width rather than just its median -- the tails a "
              "defensible fill pitch needs.",
        plot_warning="Plot the linked g-code file directly. Adding this sheet to the canvas "
                     "shows you WHERE it lands, but plotting from the canvas would collapse "
                     "all 16 cells onto one Z and one speed and destroy the sweep.",
        generator=penwidth_gcode,
        # ARCHIVED, superseded by penwidth_wedge. This sheet reads each isolated
        # stroke's width directly, which needs ~600dpi; a phone camera recovers
        # nothing for 0.18/0.255/0.35mm bands. The wedge asks the same question
        # as a which-square judgement instead.
        archived=True,
        ui="generic",
    ),
    CalSheet(
        id="dropout_map",
        ui_hidden=True,  # hidden from the default UI list; code and route kept
        label="Dropout map (position x direction x repeat pass)",
        blurb="Long isolated single-pass lines across the reachable height (position), a "
              "vertical band (direction), and single-vs-double-pass pairs (repeat), to find "
              "whether ink dropout on hatch jobs is position-dependent or uniform. Pinned to "
              "uright_gel (not the half-pitch test profile).",
        plot_warning="Plot the linked g-code file directly. Adding this sheet to the canvas "
                     "shows you WHERE it lands, but plotting from the canvas risks vpype's "
                     "own linemerge/fit rescaling this sheet's exact mm spacing and dropping "
                     "the doubled repeat-band pass -- both of which would silently destroy "
                     "the measurement this sheet exists to make.",
        generator=dropout_gcode,
        # ARCHIVED: the sheet was designed against numbers measured from a phone
        # photo of a half-pitch trial, and its all-single-line geometry does not
        # carry the phenomenon it was meant to test.
        archived=True,
        ui="generic",
    ),
    CalSheet(
        id="penwidth_wedge",
        ui_hidden=True,  # hidden from the default UI list; code and route kept
        label="Pen-width pitch wedge (laid-width crossing)",
        blurb="27 patches of parallel passes at increasing pitch (0deg and 45deg), "
              "bracketed by a solid anchor and a clearly-striped anchor. The pitch where a "
              "patch first shows white striping is the laid width -- a centimetre-scale "
              "judgement a phone photo can resolve, unlike one isolated hairline. The "
              "ladder scales ~0.4x-2x of the pen's MEASURED LAID WIDTH. For a pen with no "
              "measured laid/core width it instead brackets an ABSOLUTE 0.10-2.00mm and "
              "says so in its own legend -- scaling off pen_mm there would be circular, "
              "since on a ballpoint pen_mm is the BALL DIAMETER, not the ink width. An "
              "explicit per-request ladder_mm_lo/ladder_mm_hi override is available for "
              "either kind of pen. The sheet generates for WHICHEVER PEN IS ACTIVE (or "
              "explicitly requested), re-validated from disk by id -- not always "
              "uright_gel -- so the response header and the g-code always name the same "
              "pen. See ccplotter/calibration_penwidth_wedge.py.",
        plot_warning="Plot the linked g-code file directly. Adding this sheet to the canvas "
                     "shows you WHERE it lands, but plotting from the canvas risks vpype's "
                     "own linemerge/fit rescaling this sheet's exact mm pitch ladder -- a "
                     "0.26mm patch reading as 0.28mm (7% off) would misreport the laid-width "
                     "crossing this sheet exists to find. Capture: one scan/photo, one "
                     "exposure, whole sheet in frame, no HDR/auto-enhance; >=600 dpi NATIVE, "
                     "verified from the image's own pixel size against a ruler in frame (a "
                     "declared dpi does not count -- a 240 dpi scan cannot read any rung "
                     "under ~0.21mm); PNG/TIFF or JPEG 4:4:4; this sheet has no built-in "
                     "fiducial, so lay a ruler in frame.",
        generator=penwidth_wedge_gcode,
        ui="generic",
    ),
    CalSheet(
        id="fidelity",
        ui_hidden=True,  # hidden from the default UI list; code and route kept
        label="Fidelity sheet (known-darkness patches + scale/width controls)",
        blurb="Nine known-darkness hatch patches (A0-A8), two perpendicular pen-width groups, "
              "scale fiducials, and blank/saturated controls. ZONE B (the app's own real "
              "routed fill) IS NOT INCLUDED in this build -- see the g-code header and "
              "response JSON's zone_b field for why (a reproducible pipeline hang, unrelated "
              "to this sheet, documented in ccplotter/calibration_fidelity.py).",
        plot_warning="Plot the linked g-code file directly. Adding this sheet to the canvas "
                     "shows you WHERE it lands, but plotting from the canvas would re-emit "
                     "every patch's hatch at one Z and one draw speed -- destroying the "
                     "known-darkness comparison this sheet exists to make. Zone B (the app's "
                     "real routed fill) is not present in this build; do not read the sheet "
                     "as including it.",
        generator=fidelity_gcode,
        ui="generic",
    ),
    CalSheet(
        id="fillmode",
        ui_hidden=True,  # hidden from the default UI list; code and route kept
        label="Fill mode sheet (contour vs zigzag, 0/90/45deg)",
        blurb="Four 30x30mm solid patches at the SAME 0.24mm pass pitch: A contour (the "
              "default fill, the control), B zigzag at 0deg machine frame, C zigzag at "
              "90deg, D zigzag at 45deg. D is a PRE-REGISTERED prediction of a visible "
              "failure (~36% bare paper vs ~1% at 0deg), read directly off the author's own "
              "plotted pitch wedge -- see ccplotter/calibration_fillmode.py. Pinned to "
              "uright_gel: the pitch/prediction are calibrated numbers specific to that "
              "pen's own wedge reading, not a valid measurement of a different pen, so "
              "this sheet REFUSES loudly if a different pen is "
              "active/requested rather than silently substituting uright_gel's g-code "
              "under the other pen's name. B/C/D are pinned to WHOLESALE zigzag "
              "(fill_connect=none, perimeter_count=0): NOT what the "
              "installed app checkbox produces after a rebuild, and NOT the "
              "best zigzag configuration (E+1+2+d1); read these three patches as "
              "isolated wholesale zigzag only. Depends on "
              "fill_mode=\"zigzag\" in trace_v3.py/composite.py.",
        plot_warning="Plot the linked g-code file directly. Adding this sheet to the canvas "
                     "shows you WHERE it lands, but plotting from the canvas would re-emit "
                     "all four patches through the app's own composite pipeline at one "
                     "fill_mode (render_opts is job-wide) -- destroying the very A-vs-B/C/D "
                     "comparison this sheet exists to make.",
        # ARCHIVED: B/C/D are pinned to WHOLESALE zigzag (fill_connect=none,
        # perimeter_count=0), which is not what the app's zigzag checkbox now
        # produces, nor the best zigzag configuration; superseded by the
        # checkbox wiring itself. Module and generator kept, same precedent as
        # zigzag_crop_ab / singlepass / penwidth.
        archived=True,
        generator=fillmode_gcode,
        ui="generic",
    ),
    CalSheet(
        id="linestar",
        ui_hidden=True,  # hidden from the default UI list; code and route kept
        label="Line star (16-direction pen-width, uright_gel)",
        blurb="48 isolated single 15mm passes -- 16 directions (8 angles x both travel "
              "senses, 22.5deg apart) x 3 repeats, positioned by a recorded Latin-square "
              "layout so plate position and plot order never confound the angle "
              "comparison. Separates tilt (opposite travel sense), axis asymmetry "
              "(0 vs 90deg), and diagonal anisotropy (45/135 vs 0/90) from a noisy "
              "wedge hint of a wider 0deg line. On-sheet 0/90/45deg scale "
              "fiducials with 10mm ticks, an asymmetric F orientation glyph, and a "
              "small optional coverage twin (four 0.24mm-pitch patches). Pre-registered "
              "prediction (isotropic / diagonal-anisotropy / axis-asymmetry / tilt "
              "signatures) is in the g-code header and the response's `prediction` field, "
              "before any plotting. Measurement note: the "
              "~40% diagonal-anisotropy figure is a LOWER BOUND, not a point estimate -- "
              "measure_linestar.py's width estimator is biased thinner-reads-wider, so a "
              "raw gap always UNDERSTATES the true one. The script only DECLARES "
              "'anisotropic' when the 45/135-vs-0/90 gap clears 2x that capture's own "
              "measured bias band (from ITS OWN twin-patch edge, not the synthetic self-"
              "test); below that it reports inconclusive, not isotropic -- a real effect "
              "up to ~15% can be hidden there. Raw AND deconvolved widths are both "
              "reported, never just the corrected number. Per-element direction/repeat/"
              "machine-angle is NOT printed on the sheet -- read the response's `layout` "
              "field (or the copy saved alongside the gcode) -- see "
              "ccplotter/calibration_linestar.py. Pinned to uright_gel; refuses if a "
              "different pen is active/requested. The measurement script (measure_linestar.py) is not distributed.",
        plot_warning="Plot the linked g-code file directly. Adding this sheet to the canvas "
                     "shows you WHERE it lands, but plotting from the canvas risks vpype's "
                     "own linemerge/fit touching this sheet's exact 15mm pass lengths and "
                     "22.5deg direction angles, and could reorder the 48 passes -- the "
                     "randomized-but-recorded emission order this sheet's Latin-square "
                     "design depends on to keep plate position and plot order unconfounded "
                     "with angle. Capture: one scan/photo, one exposure, whole sheet in "
                     "frame, no HDR/auto-enhance; >=600 dpi NATIVE, verified from the "
                     "image's own pixel size against this sheet's own 0/90/45deg fiducial "
                     "ticks in frame (a declared dpi does not count); PNG/TIFF or JPEG 4:4:4 "
                     "-- chroma subsampling is the leading suspect for desaturating coarse "
                     "45deg ink to gray, which would manufacture the very anisotropy this "
                     "sheet exists to measure.",
        generator=linestar_gcode,
        ui="generic",
    ),
    CalSheet(
        id="solidpitch",
        ui_hidden=True,  # hidden from the default UI list; code and route kept
        label="Solid-fill pitch ladder + positive control (contour vs zigzag, uright_gel)",
        blurb="16 solid >=15mm squares + 1 white reference (17 items total) crossing 5 fill "
              "pitches (0.229/0.20/0.18/0.15/0.118mm) x 2 geometries (contour rings, zigzag "
              "rows), a null pair (0.18mm contour at two positions, upper-bounds scan noise), "
              "a deposition control (0.118mm zigzag drawn twice over itself, isolating "
              "overdraw from pitch), and a THREE-PITCH POSITIVE-CONTROL LADDER (contour-only) "
              "at 0.35/0.50/0.85mm: predicted readings SOLID / TRANSITIONAL / OPEN "
              "respectively, from gap arithmetic (corner_gap = pitch*sqrt(2) - "
              "(laid/2)*sqrt(2) - laid/2, a round pen under-fills a ring's corner ~2x the "
              "straight gap) -- 0.85mm is the ONE unqualified OPEN prediction. "
              "INSTRUMENT-FAILURE RULE: if 0.85mm does not read visibly open/separated on a "
              "scan, the scan/photo instrument -- not the pen -- failed to resolve pitch on "
              "that run, and NO conclusion about the 0.229-0.118mm ladder may be drawn from "
              "it; measure_solidpitch.py evaluates this gate BEFORE any plateau verdict. "
              "One 9x2 grid cell is deliberately left empty (17 items, 18 cells) -- randomized "
              "along with everything else, recorded in the layout JSON's grid.unused_cells, "
              "not a missing square. Pre-registered prediction for the original 5-pitch ladder: "
              "darkness rises toward the pen's ~0.18mm solid core then may FALL below ~0.15mm "
              "as overdraw removes ink -- the question being whether tightening pitch "
              "toward the core make solid areas read fully dark before overdraw removes gel. "
              "Grid position AND emission order are independently randomized from two "
              "preregistered seeds so plate position/print order never confound the "
              "pitch/geometry comparison. Pinned to uright_gel; refuses if a different pen is "
              "active/requested. MEASUREMENT RULES: PRIMARY "
              "statistic is each square's raw mean darkness minus the white-reference square's "
              "own raw mean, in capture intensity units -- NOT a 0-1 normalization against "
              "'the darkest square' (kept only as a labelled SECONDARY figure, since assuming "
              "some square is meaningfully darkest begs the question a plateau result would "
              "contradict). If every pitch square falls within the null pair's own "
              "minimum-detectable-difference of every other, AND the instrument-failure gate "
              "above passed (0.85mm read open), that is reported as 'plateau already reached "
              "at 0.229mm (loosest tested)' -- a valid result, not a failed measurement. "
              "Capture rule: one "
              "photo/scan, one exposure, whole sheet in frame, no HDR/auto-enhance; "
              ">=600 dpi NATIVE verified from the image's own pixel size against a known "
              "physical length in frame, not a declared dpi; lossless or JPEG 4:4:4. "
              "Contour-vs-zigzag is reported both WITH and WITHOUT "
              "each square's own centre block (6x6 block map), with the centre-included "
              "figure kept as the headline. The measurement script (measure_solidpitch.py) is not distributed. "
              "See ccplotter/calibration_solidpitch.py.",
        plot_warning="Plot the linked g-code file directly. Adding this sheet to the canvas "
                     "shows you WHERE it lands, but plotting from the canvas risks vpype's own "
                     "linemerge/fit touching this sheet's exact sub-0.25mm pitch geometry and "
                     "could reorder or fuse the deposition control's two back-to-back passes -- "
                     "both of which would silently destroy the pitch/overdraw comparison this "
                     "sheet exists to make. Capture: one scan/photo, one exposure, whole "
                     "sheet in frame, no HDR/auto-enhance; >=600 dpi NATIVE, verified "
                     "from the image's own pixel size against this sheet's own 0/90deg "
                     "fiducial ticks in frame (a declared dpi does not count -- it has no "
                     "45deg fiducial, so lay a ruler in frame for a 45deg-specific check); "
                     "PNG/TIFF or JPEG 4:4:4. Check "
                     "the 0.85mm positive-control square FIRST -- if it does not read "
                     "open/separated, stop and treat the capture as failed rather than "
                     "reading the pitch ladder.",
        generator=solidpitch_gcode,
        ui="generic",
    ),
    CalSheet(
        id="pocket_bic",
        label="Dark-pocket dimple test: contour/spine/zigzag/hybrid + speed arm (BIC Cristal)",
        blurb="Synthetic dark pockets (inscribed radius 0.30-2.00mm, 7 rungs x 3 replicates x "
              "orientations 0/45/90) in capsule and disc shapes, drawn by closed-form geometry "
              "at the pen's own fill pitch: contour / medial-axis spine / zigzag / hybrid "
              "(rings down to R_HYBRID=0.45mm, then spine rows) capsules, a speed arm "
              "(contour capsules at F1500 vs F3000, n=24 each; discs n=21), disc and zigzag shape controls, "
              "9 positive-control discs with a deliberately bare core (0.25/0.40/0.60mm), "
              "12 negative-control solid squares, a prime patch, and 6 solid fiducial squares "
              "sized to survive dark_thresh=90. Question: are the light dimples inside "
              "contour-filled solids ball starvation on tiny short core paths, or ring-"
              "convergence overdraw? Predictions, detector, null and "
              "pass/fail rules were fixed before plotting (that record is not distributed). Placement "
              "and emission order are seeded (20261002/20261003/20261004). Pinned to "
              "bic_cristal_blue; label it 'nominal 0.12, laid unknown' in any preview "
              "or readout. Generated by this module only: no trace_v3 fill mode behind "
              "spine/hybrid. See ccplotter/calibration_pocket.py.",
        plot_warning="Plot the linked g-code file directly. Adding this sheet to the canvas "
                     "shows you WHERE it lands, but plotting from the canvas would let vpype "
                     "merge/reorder the sub-0.2mm ring and row geometry and flatten the "
                     "F1500/F3000 speed arm into one speed -- destroying exactly what this "
                     "sheet compares. POST with pen_id=bic_cristal_blue explicitly and re-read "
                     "the live z_touch first (BIC 18.95 vs uRight 11.45 = 7.50 mm; the active "
                     "profile drifts). Record the z_touch the sheet reports. Capture: one scan, "
                     "one exposure, whole sheet in frame, no HDR/auto-enhance; >=600 dpi NATIVE "
                     "verified from the image's own pixel size against the sheet's own 0/90deg "
                     "rulers in frame; PNG/TIFF or JPEG 4:4:4. "
                     "Do not read the sheet by eye first; use the pre-registered "
                     "detector.",
        generator=pocket_bic,
        ui="generic",
    ),
    CalSheet(
        id="pocket_preppy",
        ui_hidden=True,  # hidden from the default UI list; code and route kept
        label="Dark-pocket dimple test, Preppy twin on identical geometry (non-ball control)",
        blurb="Identical geometry and cell placement to pocket_bic, drawn with the Platinum "
              "Preppy (a fibre-free, non-ball-tip nib; fill_mm 0.275) at its own fill pitch, "
              "plus an extra 'P012' arm (contour capsules at the BIC's 0.12mm pitch) that "
              "separates pen from pitch. The Preppy-vs-BIC contrast is confounded by pitch "
              "(0.275 vs 0.12) unless the P012 arm is read first. See pocket_bic.",
        plot_warning="Plot the linked g-code file directly (same reasons as pocket_bic). POST "
                     "with pen_id=platinum_preppy explicitly; re-read z_touch (21.15) and the "
                     "angled-mount reach clamp first. Capture as for "
                     "pocket_bic.",
        generator=pocket_preppy,
        ui="generic",
    ),
    CalSheet(
        id="fragment_bic",
        label="Fragment dose-response: piece length / lifts / reversals in solid patches (BIC Cristal)",
        blurb="100 solid 6mm patches (92 at the BIC's own 0.12mm pitch with identical row coverage, plus 8 "
              "at the 0.085mm production-pitch probe: N and L03 x4) "
              "but dosed fragment structure: a piece-length ladder (0.3/0.5/1/2/4mm, each piece a "
              "pen-down), an open-start density ladder at fixed 0.5mm pieces, a reversals-only arm "
              "(continuous, no lifts, direction reversal every 0.3-4mm), bridged pen-down pairs "
              "(nearest honest synthetic of Stage E), a pocket_link 0-vs-3 EQUIVALENT on nested "
              "disc rings plus the chained square rings, continuous-null and rows-only "
              "patches, and commanded-bare-centre positive controls. Every patch carries a printed "
              "ASCII label. Question: do the BIC dimples come from fragmentation (short pieces, "
              "many starts, reversals keeping the ball from rolling)? PRE-REGISTERED hypothesis, "
              "readout and decision rule (fixed before plotting; the record is not distributed). "
              "Micro/mega lifts are applied as production does. Pinned to bic_cristal_blue "
              "(fill_mm must be 0.12); label it 'nominal 0.12, laid unknown'. Placement and "
              "emission order are seeded (20261003/20261004/20261005). See "
              "ccplotter/calibration_fragment.py.",
        plot_warning="Plot the linked g-code file directly. Adding this sheet to the canvas shows you "
                     "WHERE it lands, but plotting from the canvas would let vpype merge/reorder/"
                     "simplify the abutting row pieces and the exact 0.12mm geometry -- destroying "
                     "the fragment dose this sheet exists to apply. POST with pen_id=bic_cristal_blue "
                     "explicitly and re-read the live z_touch first (BIC 18.95 vs uRight 11.45 = "
                     "7.50 mm; the active profile drifts). Record the z_touch the sheet reports. "
                     "This is a long plot (thousands of short pen-downs): check the pen has ink "
                     "and the ball is not dry before starting. Capture: one scan, one exposure, "
                     "whole sheet in frame, no HDR/auto-enhance; >=600 dpi NATIVE verified from the "
                     "image's own pixel size against the sheet's own 0/90deg rulers in frame; "
                     "PNG/TIFF or JPEG 4:4:4. Do not read the "
                     "sheet by eye first; use the pre-registered windows.",
        generator=fragment_bic_probe,
        ui="generic",
    ),
    CalSheet(
        id="uright_short",
        ui_hidden=True,  # hidden from the default UI list; code and route kept
        label="uRight short sheet: laid-width wedge + start-skip pair + contact grid (uRight gel)",
        blurb="One short uRight sheet that separates the three candidate causes of blank areas on the "
              "paper: (1) LAID-WIDTH WEDGE, 13 absolute pitches 0.08-0.50mm at "
              "0 and 45 deg, each patch ONE continuous 6x6mm serpentine (zero lifts, so start-skipping "
              "cannot contaminate it), plus three isolated 20mm lines at 0/45/90 deg; (2) START-SKIP "
              "PAIR, N (one stroke) vs L03 (0.3mm pieces, same commanded ink) at pitch 0.13 and "
              "0.25mm, 4 replicates each, 4.5mm patches, seeded placement/order; (3) CONTACT GRID, 24 identical marks "
              "(6mm line at 0 and 90 deg + a 3x3mm serpentine at 0.13mm) over the whole page. "
              "LAID WIDTH UNMEASURED: the pitches are commanded spacings, not widths; the wedge is how "
              "it gets measured. Every item carries a printed ASCII label; fiducials and 0/90 rulers "
              "first; micro/mega lifts as production. Pinned to uright_gel (its _halfpitch/_tight3x "
              "variants share its z_touch and are generated from the base id). See "
              "ccplotter/calibration_uright_short.py.",
        plot_warning="Plot the linked g-code file directly. Adding this sheet to the canvas shows you "
                     "WHERE it lands, but plotting from the canvas would let vpype merge/reorder/"
                     "simplify the 0.3mm pieces and the absolute-pitch rows -- destroying the dose "
                     "and the wedge this sheet exists to apply. Pinned to the BASE uright_gel "
                     "profile: if uright_gel_halfpitch or uright_gel_tight3x is active the sheet is "
                     "still generated from uright_gel (same z_touch 11.45; their fill/laid values "
                     "are NOT used). Re-read the live z_touch first (uRight 11.45 vs BIC 18.95 = "
                     "7.50 mm; the active profile drifts) and record the z_touch the sheet reports. "
                     "Gel can pool on tight passes: the tightest pitch is 0.08mm and no patch is "
                     "larger than 6x6mm; use a fresh paper area and check the pen has ink. Laid "
                     "width is UNMEASURED -- do not read a width off the sheet's pitch labels.",
        generator=uright_short_gcode,
        ui="generic",
    ),
    CalSheet(
        id="photofe_bic",
        label="Photo front end on paper: 8 regions x 6 arms at plot scale (BIC Cristal, 0.12mm)",
        blurb="Does S1.2's extra source-edge recall show up USEFULLY on paper, and where do the finer "
              "settings cross from structure into texture / interpolation noise? 8 regions of 6 photos "
              "(lettering and mechanics first, then high-texture areas) x 6 arms (shipped A, S1.4, S1.2, "
              "S1.0, C1024F, NATcap = side min(1024, source short side) sigma 1.2px). Each cell is a "
              "35 x 13mm window of the REAL tracev3 output of a full-page plot (257x176, margin 10, "
              "scale 1) clipped to the window, NOT re-traced, so it is the same physical size as on a "
              "full-page plot. Every cell carries a printed ASCII label (arm, region); 0/90 rulers and "
              "six fiducials as on the fragment sheet. Judge by eye (recall is not evidence). "
              "The strokes are a precomputed asset embedded in calibration_photofe_data.py, sha256 "
              "checked at generation. Pinned to bic_cristal_blue (fill_mm 0.12); label it 'nominal "
              "0.12, laid unknown'. See ccplotter/calibration_photofe.py.",
        plot_warning="Plot the linked g-code file directly. Adding this sheet to the canvas shows you "
                     "WHERE it lands, but plotting from the canvas would let vpype merge/reorder/"
                     "simplify the pieces and destroy the real-trace geometry this sheet exists to "
                     "compare. POST with pen_id=bic_cristal_blue explicitly and re-read the live "
                     "z_touch first (BIC 18.95 vs uRight 11.45 = 7.50 mm; the active profile drifts). "
                     "Record the z_touch the sheet reports. This is a VERY LONG plot (about 16,000 "
                     "short pen-downs, ~105 m of line; see the time estimate, which is a model and "
                     "not stopwatch-validated): check the pen has ink and the ball is not dry, and "
                     "expect to re-check the pen part-way.",
        generator=photofe_bic,
        ui="generic",
    ),
    # ---- bespoke: own route AND own UI handler; listed for completeness ----
    CalSheet(
        id="deposition",
        ui_hidden=True,  # hidden from the default UI list; code and route kept
        label="Deposition sweep (overlap x pressure x speed)",
        blurb="120 fill patches measuring how much ink actually lands as passes overlap.",
        plot_warning="Plot the linked g-code file directly -- a canvas plot would flatten "
                     "the sheet's 20 cells onto one Z and one draw speed.",
        generator=None, ui="bespoke",
    ),
    CalSheet(
        id="plot", label="Calibration plot",
        blurb="The original combined calibration chart.",
        plot_warning="Plot the linked g-code file directly; canvas plotting changes the "
                     "true-size mm ladders this chart is measured off.",
        generator=None, ui="bespoke",
    ),
    CalSheet(
        id="pack", label="Calibration pack",
        blurb="The per-pen set: linewidth, min-ink, swatch, wedge.",
        plot_warning="Plot the linked g-code files directly; these are true-size charts and "
                     "rescaling them invalidates the mm ladders you measure off them.",
        generator=None, ui="bespoke",
    ),
    CalSheet(
        id="combo", label="Calibration combo",
        blurb="Combined multi-chart sheet.",
        plot_warning="Plot the linked g-code file directly; rescaling invalidates its ladders.",
        generator=None, ui="bespoke",
    ),
    CalSheet(
        id="block", label="Calibration block",
        blurb="Single-block calibration pattern.",
        plot_warning="Plot the linked g-code file directly; rescaling invalidates its ladders.",
        generator=None, ui="bespoke",
    ),
)


def _validate(sheets) -> dict:
    """Import-time validation. A malformed entry raises HERE, so the server
    refuses to start rather than the problem surfacing at render -- or worse, as
    a sheet that plots from the canvas and silently measures nothing."""
    out = {}
    for s in sheets:
        if not s.id or not s.label or not s.blurb:
            raise RuntimeError(f"calibration registry: entry {s.id!r} is missing id/label/blurb")
        if not (s.plot_warning or "").strip():
            raise RuntimeError(
                f"calibration registry: sheet {s.id!r} has no plot_warning. Every sheet here "
                f"bakes per-cell Z and draw speed into separate blocks; a canvas plot would "
                f"flatten them and measure nothing. The warning is mandatory.")
        if s.ui not in ("generic", "bespoke"):
            raise RuntimeError(f"calibration registry: sheet {s.id!r} has unknown ui={s.ui!r}")
        # The anti-escape rule: you cannot dodge the parametrized safety test by
        # omitting a generator. Absent a generator you must DECLARE ui="bespoke",
        # which is visible in review and in the catalogue.
        if s.ui == "generic" and s.generator is None:
            raise RuntimeError(
                f"calibration registry: sheet {s.id!r} is ui='generic' but has no generator. "
                f"Either provide one (and gain automatic safety-test coverage) or declare "
                f"ui='bespoke' explicitly.")
        if s.ui == "bespoke" and s.generator is not None:
            raise RuntimeError(
                f"calibration registry: sheet {s.id!r} is ui='bespoke' but carries a generator; "
                f"that is ambiguous -- a bespoke sheet is driven by its own route.")
        if s.id in out:
            raise RuntimeError(f"calibration registry: duplicate sheet id {s.id!r}")
        out[s.id] = s
    return out


SHEETS = _validate(_SHEETS)


def generic_sheets():
    """Sheets the generic dispatcher can drive -- and exactly the set the
    registry-parametrized safety test covers."""
    return [s for s in SHEETS.values() if s.ui == "generic"]


def catalogue(include_archived: bool = False):
    """The UI's button list. `ui` tells the client whether to call the generic
    dispatcher or the sheet's own bespoke handler.

    Archived sheets are omitted by default. They remain registered
    and dispatchable by id -- this only stops them crowding the button list once
    they have answered their question. Pass include_archived=True (or hit
    /api/calibration_sheets?all=1) to see them. `archived` is reported on every
    entry so a client can render them in a fold rather than hide them, without
    a second endpoint."""
    return [{"id": s.id, "label": s.label, "blurb": s.blurb,
             "ui": s.ui, "plot_warning": s.plot_warning,
             "archived": s.archived, "ui_hidden": s.ui_hidden}
            for s in SHEETS.values()
            if include_archived or not (s.archived or s.ui_hidden)]
