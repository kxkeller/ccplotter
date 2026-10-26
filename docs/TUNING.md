# Tuning notes: opt-in two-pen options

This file documents the constants of two opt-in two-pen options. Both are OFF by
default and OFF is byte-identical to the behaviour without them. Several tests check
that the constants named here appear in this file, so keep it in step with
`ccplotter/twopen_keytint.py` and `ccplotter/twopen_hatch.py`.

All thresholds are expressed in pen widths of the active BLUE profile's `fill_mm`
(called W below), never in pixels, so they scale with resolution and pen.

## Blue margin under black -- `blue_trap` (default OFF) -- `ccplotter/twopen_keytint.py`

Request flag `blue_trap` (boolean). In the UI it is the checkbox "Blue margin under black
(fewer pen lifts)" in the two-pen options. The constants are also available as
`twopen_keytint.CONSTANTS["blue_trap"]`, and are echoed in the run's `info["blue_trap"]`
and in the run provenance.

- `TRAP_MARGIN_PEN = 1.0` -- the blue mask is `blue_A | (dilate(blue_A, 1 W) & black_A)`:
  blue is extended 1 W under the black so the two layers overlap instead of leaving a crack.
- `TRAP_HOLE_RADIUS_PEN = 1.5` -- holes of the blue mask smaller than pi*(1.5 W)^2 mm2
  (0.1018 mm2 at W = 0.12) are closed.
- `TRAP_FOLD_RADIUS_PEN = 1.5` -- black components smaller than pi*(1.5 W)^2 mm2 are
  reassigned to blue, never deleted (the union of ink is conserved and asserted at run time).
- `TRAP_BLUE_CONNECTOR_MM = 0.5` -- blue connectors of up to 0.5 mm are allowed, checked
  against the union of the blue and black masks. Blue is routed with `ROUTE_WALK_MAX_SEGS_TWOPEN`.

Black is the original black minus the folded specks, routed at the request's
`connector_distance` exactly as when the option is off. The trap thresholds scale with the blue
profile's `fill_mm`; the colour separation uses a fixed nominal `ROLE_WIDTH_MM` of 0.12, so
the two coincide for a 0.12 mm pen and diverge for another. Time figures for this option are
unvalidated models; nothing has been plotted.

## Hatched blue -- "Hatched blue (lighter, fewer pen lifts)" -- `hatched_blue` (default OFF) -- `ccplotter/twopen_hatch.py`

Request flag `hatched_blue` (boolean). The blue layer is drawn as long hatch lines instead of
solid fill: lighter (about half the blue coverage), a different look (paper shows between the
lines) and fewer pen lifts. A few small gaps can appear in dark areas. Tiny black specks (under
0.1 mm2) are drawn in blue. The mapping is one global rule with no per-image constants. The
constants are in `twopen_hatch.py`, echoed in `CONSTANTS["hatched_blue"]` and in the run's
`info["hatched_blue"]`:

- `SIGMA_PEN = 1.0` -- the blue target is the 3x3-blurred grey smoothed by a Gaussian of
  sigma 1 W. Coverage is `c = clip((t_blue - grey_s) / (t_blue - T), 0, 1)`, with `c = 1` on
  black; `t_blue` comes from the request's `blue_amount`.
- `LEVELS = 8`, `PITCH_PEN = 1.0` -- line pitch W (densest, c = 1) up to 8 W (c = 1/8). Line k
  at a place is drawn iff `ceil(8c) > bit-reversal(k mod 8)`: an 8-level ordered dither over
  the line index. `ANGLE_DEG = 45`.
- `CLIP_PEN = 0.5` -- hatch only where the distance to the outside of (blue u black) is at least
  0.5 W plus one raster pixel (0.11 mm at W = 0.12), so a pen of width W stays inside the region.
- `MIN_RUN_PEN = 1.0` -- runs shorter than 1 W are dropped.
- `LINK_RADIUS_PEN = 8.0` -- switchback link radius, the largest pitch the dither makes, so an
  adjacent drawn line is always reachable. Preference: forward line, then same line, then
  backward line, then nearer. A link is taken only if the straight connector passes the
  `stage_e_mask` predicate against the final trace mask of blue u black; with no candidate the
  pen lifts.
- `LINK_CANDIDATES = 40` -- a cost guard (mask tests per exit point), not a quality constant.
- `WALK_CAP_UNUSED = 1,000,000` -- a walk cap that applied only to a superseded router-linked
  variant. This variant uses an explicit linker with no walk cap. Recorded for traceability, not live.
- `HATCH_PPM = 20`, `RUN_STEP_MM = 0.025` -- the grid the clip, coverage and runs are evaluated
  on (0.05 mm per pixel; half a pixel along a line).

Precedence with the other options: hatched blue REPLACES the blue layer (the `blue_trap` blue
mask and 0.5 mm blue connector are not used). The black layer is the `blue_trap` black (specks
below pi*(1.5 W)^2 reassigned to blue, never deleted), so turning hatched blue on implies that
black whether or not "Blue margin under black" is ticked. "Stay down over ink" still applies to
the black layer; on the blue layer it is a no-op because the hatch linker already stays pen-down
only inside blue u black.

Prototype measurements on two benchmark images (models, not plotted; the job does not compute
them and they are stored in `CONSTANTS["hatched_blue"]["prototype_measurements"]`): blue
coverage of the target fell from 0.92/0.97 (solid) to 0.55/0.51 (hatched); mid tones render
lighter by about 2 to 8 L*; area uncovered by either pen went from 1.36% to 0.18% on the first
image (better) and from 0.07% to 0.27% on the second (worse); area uncovered by black is unchanged.
