# Fixture: dotstorm-proxy-rivets-ballast

## Purpose

Regression guard against dot-storm / beads-as-rings artifacts for any change that touches
`_component_fidelity_detail` or its gates (isolation pre-filter, area gate, decisive-margin threshold).

Rivets (fine, isolated, periodic dashes) and ballast (dense, small, irregular blobs) are the two texture
classes most similar to the failure mode: small isolated or clustered components mistaken for text-like
shapes and switched to a contour candidate that reads as a ring or scribble at true pen width instead of a
clean dot or dash.

## Pass criterion

No ring/scribble-class artifact at true pen width (not at mask level). Reference values from the baseline
measurement (one known-clean run, not a formal tolerance band; anything materially above them should be
inspected visually first):

| region | mean abs pixel diff (0-255) | fraction of pixels changed (>25) |
|---|---|---|
| handrail rivets | 0.279 | 0.29% |
| ballast/gravel | 0.217 | 0.22% |

## Contents

- `source_pointer.md`: this fixture reuses `tests/golden_tests/fixtures/loco-472-trestle/source.png`
  instead of duplicating it; both are byte-identical (SHA-256
  `56dfea7088bafa8372694b70e169169283317a977b57081796cdefaf9b3d8178`).
- `request.json`: the `/api/process` request (style `plotready`). Its item `id` is an upload id, not a file
  name; pair it with the source above.
- `crop_boxes.json`: the two reference-crop regions and their pixel sizes. No numeric crop offsets were
  recorded when the crops were made (they were located by eye on the full render), so the box offsets are
  `null` and must be re-located before any automated crop comparison.
- `handrail_rivets_strict.png`, `ballast_gravel_strict.png`: true-pen-width reference crops of the
  baseline (`strict`) isolation mode.

## How to reproduce a comparison run

1. Use the source image named in `source_pointer.md`.
2. POST `request.json` to `/api/process` (with whatever flag or gate change is under test).
3. Render the resulting preview or G-code at true pen width.
4. Re-locate the handrail-rivet and ballast-gravel regions on the full-page render, using the two reference
   crops as a guide, and diff against them.
5. Compare against the table above. Materially worse numbers, or a ring/scribble appearing where there was a
   clean dot or dash, is a regression.

## Note on the reference crops

The crops were captured at `fill_mm = 0.15`. The `plotready` default is now `fill_mm = 0.25`, and
`request.json` sets no explicit `fill_mm`, so replaying it today renders uniformly thicker strokes than the
crops. That is not a ring/dot-storm regression. The numbers above remain valid for comparing the `strict` and
`trial` isolation modes with each other at the same `fill_mm`, but the `_strict.png` files are no longer a
byte-comparable reference for the current default; re-capture them first when reusing this fixture.
