# Fixture: loco-472-trestle

A source image and a reconstructed `/api/process` request body for a `tracev3` run (`requested_res` 2600)
that showed two regressions:

1. **"472" digit fragmentation**: thin digit strokes route through the skeleton-fallback branch instead of
   the width-proportional contour-pass generator, producing jittery or doubled strokes even though the source
   gaps clear the 0.30 mm visible-ink ceiling.
2. **Trestle wood-grain loss**: the dot-cull / speck-floor stage drops interior grain and crosshatch texture
   almost wholesale; the run's own audit (`dropped_ink_pct: 0.668`) trips the code's 0.5% self-check warning.

## `settings.json` is a reconstruction

The original request body was not recorded; only the derived audit block was. `settings.json` is therefore
reconstructed:

- Fields evidenced by the run record: `style: "tracev3"`, `resolution: 2600`, `job_type: "gcode"`, one item.
- Fields whose audit values match the code defaults exactly (so they were not overridden): `min_mark_mm2: 0.035`,
  `tone: "off"`, `native_res` unset (`false`), `pre_blur` unset (`3`).
- All other fields (paper size, margin, speeds, offsets, ...) are the code defaults and are not independently
  verified for the original run. `paper: "200x200"` matches the page size of the original true-width render.

`source.png` (1448x1086) is the original image, byte-for-byte. Its origin is not recorded in this repository.

Replaying this fixture reproduces the tracev3 mask and pass-generation stage that was analysed, but not
necessarily byte-identical output to the original run if a reconstructed field differs.

## Current defaults

`fill_mm` (now 0.25) and `texture_routing` (now `band_continuity_v1` for tracev3) are not set in
`settings.json`, so a replay uses the current defaults: `fill_px` 3.06, `pinch_runs_fired` 6111 (0 with the older
`off` routing). The digit-fragmentation and grain-loss issues are the ones `band_continuity_v1` addresses.
