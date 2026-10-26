# Known limitations

Stated plainly, as of this release.

## Machine and calibration

- **Calibration is tied to the author's machine.** The shipped Z heights, reach
  polygons, pen offsets and pen profiles are measurements for one printer and one
  pen mount. A first run on another machine generates G-code with them and does not
  stop you. Re-measure before plotting; see [CALIBRATION.md](CALIBRATION.md). The
  calibration sheets additionally refuse to run unless the active reach profile is
  the author's (`plotter_2003_v32`).
- **The G-code targets one printer.** The start and end code, the fan commands
  (only the fan indices that firmware has; addressing another halts it), the
  absence of `G28` and the Z conventions are written for the Elegoo Centauri Carbon
  with its Klipper-based firmware and a pen attachment. Nothing was tried on any
  other machine or firmware.
- **Platform.** Tested on Windows with Python 3.12.0 and the exact set in
  `requirements-lock.txt`, which is specific to that platform and interpreter.
  macOS and Linux were not tested, and neither were other package versions.

## Features not verified or not working

- **Network send was not tested.** The optional path that talks to the printer over
  the network (active only when `CCPLOTTER_PRINTER_HOST` is set) was not run
  against hardware in this release. The supported path is saving the G-code and
  copying it to the printer on a USB stick.
- **The manual command route is not a safety boundary.** When
  `CCPLOTTER_PRINTER_HOST` is set, `/api/gcode` forwards any single command that
  starts with `G92`, `G28`, `G1 Z`, `G0 Z` or `M84`. Such a command can still include
  X/Y words, `G28` homes every axis with the pen attached, and `G92` resets the
  coordinate frame. None of these are checked against the reach calibration.
- **Flow mode is broken with numpy 2.4.** The optional `vpype-flow-imager`
  fails with a `TypeError` ("can't multiply sequence by non-int of type
  'numpy.bool'") inside that package when run with the pinned numpy 2.4.4. It is
  third-party code and the cause was not investigated further. Flow mode is not in
  the mode selector.
- **The desktop build is unverified.** A PyInstaller spec is included, but no
  packaged build was made or run for this release, and running
  `python desktop/ccplotter_app.py` from source was not verified either.
- **The browser UI was not exercised in a browser.** The server, the HTTP API, the
  example and the tests were run; the UI page was fetched but not clicked through.
- **The runtime home must be ASCII-only.** `scripts/launch.py` runs the server with
  the absolute work directory `<home>\work`, and that path is written into the
  G-code header comment (`; Source: ...`), where non-ASCII characters are not
  allowed. The launcher refuses a home path containing them.
- **Optional integrations** (OpenAI, ComfyUI, Inkscape, DrawingBotV3) were not
  exercised against the live services or programs.

## Known defects

- **Two-pen mode can draw stray short lines on bare paper above the router cap.**
  When a layer exceeds the router's 20,000-stroke cap and the default routing is
  in use (the opt-in "stay down over ink" is off), the key (black) layer can draw
  pen-down connectors of up to 0.5 mm over bare paper. This is known and
  undiagnosed in this release. The opt-in "stay down over ink" option joins strokes
  only where the connector lies inside the ink mask, but this release did not test
  whether it avoids this defect.
- **One golden-harness fixture mismatches.** `calibration_plot` in
  `scripts/replay_harness.py verify` reports a G-code hash different from the one
  pinned in `tests/golden_tests/replay_manifest.json`. The cause is unknown. The
  mismatch value is the same on the author's own baseline, so it is not new in
  this snapshot, and the harness exits non-zero because of it. See
  [TESTING.md](TESTING.md). The same run reports nine other fixtures matching and
  the out-of-reach "tram" fixture refused as designed.
- **The spatially indexed router walk and the reference walk disagree on one real
  layer.** `tests/test_router_walk_equivalence.py` passes on the layer it ships
  with; on the larger line-art layer `tests/fixtures/lineart_loco_v1.svg` the two
  walks produce different first strokes. This was observed once and not
  investigated.

## What the output is, and is not

- **The preview is not a physical guarantee.** It draws strokes at the pen width
  the profile says, but real ink spreads, skips and varies with direction. Strokes
  that look separate in the preview can merge on paper at pen width, and fills that
  look solid can crack. Measure on paper before trusting fine detail.
- **No grey.** A pen has one ink width. Trace-style modes reproduce only what the
  ink mask keeps; smooth tones become solid ink or bare paper.
- **Generated time estimates are model figures**, not stopwatch measurements.
- **The physical scans in the documentation have no recovered settings**, so they
  illustrate what the software can produce but cannot be re-run to the same
  G-code.

## Tests

- **Reduced test coverage.** Test images that are third-party material were not
  redistributed, so those tests skip with an explicit reason; the whole suite
  therefore covers less than the author's full suite. The list is in
  [TESTING.md](TESTING.md).
