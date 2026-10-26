# golden_tests/expected: baseline identity semantics

Each `<checkpoint>/` directory holds `source.png`, `settings.json` (the exact `/api/process`
request body), `plot.gcode`, `preview.svg`, `preview.png` and `result.json` (stats and provenance).
The replay harness (`scripts/replay_harness.py`) compares fresh runs against the recorded values in
`tests/golden_tests/replay_manifest.json`.

## What counts as a match

- `plot.gcode` SHA-256 (byte-identical).
- Numeric stats in `result.json` (`z_moves`, `stroke_count`, `pen_lifts`, `file_size_kb`).
- The true-width rasterised ink mask, `preview.png` (rendered from `preview.svg` at real pen width
  with Inkscape). Judge by the plotted render, not by a mask-level metric.

`preview.svg` is stored as reference material only; it is NOT a byte-identity assertion. A
regenerated `preview.svg` that differs byte-wise while `plot.gcode` has the same SHA-256 is expected.

## Why preview.svg is excluded from byte identity

vpype's SVG writer embeds a Dublin Core `<dc:date>` timestamp taken at generation time. Replays of
identical settings produced SVGs identical in line count, element count, points-attribute length,
viewBox and total byte length; the only differing line was the `<dc:date>` timestamp. The application
does not write that field; it is per-generation library metadata, not nondeterminism in the pipeline.
G-code carries no equivalent field, which is why it is byte-stable. The harness strips the
`<dc:date>` element before hashing SVGs.

## Recorded numbers

The two goldens are `plotready` jobs at the default `fill_mm` of 0.25:

- `alaska-737-jun14`: stroke_count 5351, z_moves 10702
- `tram-hongkong-lightrail`: stroke_count 4735, z_moves 9470

(Earlier recordings at `fill_mm` 0.15 gave 5434 / 10868 and 4781 / 9562 respectively.)
