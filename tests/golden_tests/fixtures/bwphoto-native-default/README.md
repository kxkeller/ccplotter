# bwphoto-native-default

Golden job for the bwphoto recipe at the default `bw_xdog_side = native`. The request deliberately
carries no `bw_xdog_side` key, so it exercises the default itself: a change of that default (or of the
native raster path) changes the recorded G-code.

Source: `tests/fixtures/twotone/dahlia.jpg` (scikit-learn sample image, CC BY 2.0; see
`tests/fixtures/twotone/PROVENANCE.md`). The harness pins the pen (`FIXTURE_PEN_ID`).
