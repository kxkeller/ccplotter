# Pipeline

How an image becomes G-code. This is a description of what the code does; it
makes no claim about how good any mode's output looks, which depends on the image
and the pen.

```
source image -> mode (image to strokes) -> placement on paper -> optional routing
             -> vpype ordering -> G-code (vpype.toml profile, per-pen Z) -> checks
             -> preview
```

All strokes are polylines in millimetres on the page. Rendering geometry is
measured against a pen model rather than tuned by eye: the pen's pass spacing
(`fill_mm`, the spacing at which neighbouring passes lay solid ink) and width come
from the active pen profile (see [CALIBRATION.md](CALIBRATION.md)).

## Modes

The mode selector in `assets/templates/index.html` offers these (request field
`style`). Other values exist in `ccplotter/render.py` (for example `flow`,
`centerline`, `hybrid`, `lineart`, `sketchv2`, `outline`, `sketch`, `schematic`)
but are not offered in the selector.

| UI label | `style` | What it does |
|---|---|---|
| Trace (V3) (default) | `tracev3` | Source-faithful ink coverage. The image is thresholded to an ink mask (Otsu by default; adaptive, Sauvola, manual and an ink-sensitivity option are available). Each connected ink region is filled with parallel contour passes: pass k is the contour of the distance transform at (k + 0.5) x pass spacing, so a stroke of width w gets about w / spacing passes at the pen's real solid-fill spacing. Regions too thin for even one pass fall back to a skeleton centreline so nothing is dropped. There is no grey: tone is only what the mask keeps. |
| Plot-Ready Line Art | `plotready` | For images that are already clean line art. Threshold, skeletonise, then trace the skeleton graph into centreline strokes, with a quality check that warns when the input has lots of grey or shading. It does not reinterpret the image. |
| Trace V3+ | `illustration` | The Trace V3 engine with validated settings for clean drawings, and a pass-spacing clamp that holds the spacing up to the resolution limit; a warning says when the limit binds. |
| B/W photo | `bwphoto` | A fixed XDoG (difference-of-Gaussians) line filter on the greyscale photograph at native detail, then the Trace V3 engine with CLAHE on and tone off. The intermediate raster can be written to the debug output. Every parameter is a global constant, not tuned per image. |
| Black Line + Blue Tone (2 pens) | `twopenkt` | Two-pen job from one photograph. The B/W photo recipe runs first; a global separation then assigns the mid-tone tint to a blue layer and the darks and line structure to a black layer. Two complete G-code files are produced, blue first and then black, each using the Z and offsets of its own pen profile (`bic_cristal_blue`, `bic_cristal_black`) and the usual start and end code, with no pen-change code inside. Two opt-in variants exist: blue margin under black, and hatched blue (long 45-degree lines instead of solid blue). They are described in [TUNING.md](TUNING.md). |
| TSP Scribble | `tsp` | Experimental stippling: stipple points, a travelling-salesman route through them, split into strokes. Slow. |
| Sketch Field | `sketchfield` | A native, opt-in "darkness-seeking" sketch: repeatedly grow short-segment strokes along the darkest remaining pixels and subtract the ink drawn from a darkness map until a configured fraction of the image's darkness is paid off. Requires a raster source. |

`pfm_native.py` implements Sketch Field without any DrawingBotV3 code; see
[../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

A source can also be added as SVG, DXF or PDF (vector input); raster-only modes
reject vector sources. AI line-art cleanup (OpenAI or a local ComfyUI) is an
optional step before any mode and is not part of the core pipeline.

## Stages after the mode

1. **Placement.** Each drawing is scaled, rotated and fitted onto the paper
   (paper size, margin, item scale and offset). The paper is placed on the bed at
   `offset_x`/`offset_y` in machine coordinates. A keep-out margin can be applied
   so ink stays clear of notebook holes, and a Fill Page option cover-crops a
   single item.
2. **Optional routing.** If `connector_distance` or `target_z_moves` is set, a
   router joins nearby stroke ends with short connectors to reduce pen lifts.
   Jobs above 20,000 strokes skip the router (an opt-in `stay_down_over_ink` option
   raises that cap and only joins where the connector lies inside the ink mask).
   The example in this repository uses no routing.
3. **vpype ordering.** The routed strokes are read with vpype and sorted
   (`linesort`) to shorten travel. A `linesimplify` step whose tolerance is a
   multiple of the pen spacing exists, but its multiplier defaults to 0
   (disabled), so no simplification is applied unless a request sets it.
4. **G-code.** vpype's `gwrite` with the `elegoo_cc` profile from `vpype.toml`
   (plugin `vpype-gcode`) writes the file, after the drawing is translated to its
   machine position. The file uses absolute machine coordinates, never `G92`, and
   does not home (`G28` is deliberately absent; the header asks you to home first,
   with the pen removed). Z heights, speeds and offsets come from the pen
   profile, not from `vpype.toml`'s defaults.
5. **Post-processing.** Short pen travels get a low "micro-lift" hop
   (touch Z + `z_micro`) instead of a full lift; the longest travels lift higher.
   Fan commands are limited to the fan indices this firmware has, written once
   after the first positioning move and again at the end.
6. **Checks before the file is returned.** Every XY endpoint and chord is checked
   against the active reach polygon, and generation is refused if any move is
   outside it. The G-code templates and comments are written to be ASCII only (the firmware's parser was found to be sensitive to non-ASCII text); the example file was checked for this and for the absence of `G92`.
7. **Preview.** The routed strokes are rendered at the real pen width (not as
   hairlines) as an SVG, with optional views of the connectors. The preview is a
   software rendering; strokes that look separate in it can merge on paper (see
   [LIMITATIONS.md](LIMITATIONS.md)).

Results are cached per input and settings, and each generated job records a run
provenance sidecar in the runtime home with the request and the pen profile
used.
