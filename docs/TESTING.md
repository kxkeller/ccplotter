# Testing

Environment used for the results below: Windows, Python 3.12.0 (python.org), the
packages in `requirements-lock.txt` (`pytest` included). Other platforms were not
tested.

## Per-file test runner

```powershell
python scripts/run_tests.py                      # every tests/test_*.py, one file at a time
python scripts/run_tests.py -j 4                 # four files at once
python scripts/run_tests.py tests/test_pen_fill.py tests/test_stitch.py
python scripts/run_tests.py -k twopen            # files whose name contains "twopen"
python scripts/run_tests.py --log-dir some\dir   # keep per-file logs there
```

**Why per file.** Several test files import `ccplotter` at module scope and change
process-wide state (environment variables, the working directory, module globals,
the vpype configuration). Running them all in one pytest process is not
supported. The runner starts one pytest process per file, each with its own fresh,
empty, absolute `CCPLOTTER_HOME` and temp directory (removed afterwards), a
wall-clock timeout, and `CCPLOTTER_PRINTER_HOST` removed from the environment so
no test depends on a printer. It prints a summary table and exits 0 only if every
file passed; skips are not failures. Some files take several minutes.

Result of the last full run, with the pinned environment: 46 files, all pass: 555
passed, 0 failed, 18 skipped, 0 errors. (Re-run it after any change.)

## Skips and why

Test images that are third-party material are not distributed, so the tests that
need them skip with the message "fixture image not redistributable; place `<name>`
at `<path>` to run". Putting the files at those paths runs the tests. Skips:

| Test | Skips | Reason |
|---|---|---|
| `test_bw_photo.py`, raster byte-equality against the locked rasters | 5 | the ice train, bell tower, bike, A350 and giraffe images are not distributed; the tram case runs and passes |
| `test_core_fill_inset_bwmask.py` | 10 | the same five images, two benchmark tests each; the tram variants run |
| `test_gapaudit_coverage_scaling.py`, all-six-sources-by-two-papers agreement | 1 | needs all six sources; only the tram is distributed (the sibling test runs the two tram cells) |
| `test_trace_boundary.py`, cleaned-heli page hashes | 1 | derived from a third-party, AI-cleaned image; the vendored mini fixture case still runs |
| `test_runtime_home.py` | 1 | applies only to Microsoft Store Python, which is not what is used here |

Total: 18 skipped. This means the suite covers less than the author's complete
one; see [LIMITATIONS.md](LIMITATIONS.md).

## Router-walk fixture note

`tests/test_router_walk_equivalence.py` compares the spatially indexed router walk
with a reference walk on a real routed layer. It tries `tests/fixtures/router_real_layer.svg` first, then (optionally) `*.clean.svg`
files under the directory in `CCPLOTTER_TEST_SVG_DIR`, then any `*.svg` under
`tests/`, and uses the first layer with 1,500 to 19,000 strokes, so the result
depends on which file is found first. The shipped `router_real_layer.svg` is the
layer the test passes on (3,173 strokes). On the other real layer in the tree,
`tests/fixtures/lineart_loco_v1.svg`, the two walks disagree (the first stroke has
16 points in the indexed walk and 8 in the reference). This was observed once and
not investigated; see [LIMITATIONS.md](LIMITATIONS.md).

## Replay harness (golden gate)

`scripts/replay_harness.py` replays recorded requests through a real dev server and
compares the results with the pinned manifest `tests/golden_tests/replay_manifest.json`
field by field, including a SHA-256 of each G-code file. Any change that alters an
emitted byte shows up as a mismatch.

```powershell
$env:CCPLOTTER_HARNESS_PORT = "5099"      # any free port; the default is 5001
python scripts/replay_harness.py verify
```

The harness starts its own server with its own pinned runtime home
(`scripts/harness_env/`), never your real one, and never contacts a printer.

**Expected result for this release:**

- nine fixtures OK: `golden-alaska-737-jun14`, `loco-472-trestle`,
  `dotstorm-proxy-rivets-ballast`, `loco-lineart-isolation-trial`,
  `synth-tallframe-height-limited`, `plane`, `ship`, `bwphoto-native-default` and
  `fan_probe`;
- `golden-tram-hongkong-lightrail` reported as NEGATIVE-OK: it is a job that must
  be refused for leaving the pen's reachable area, and the harness is green when
  the refusal (HTTP 500, "reach safety check failed") fires;
- `calibration_plot` reports a MISMATCH: pinned `fc55488e...f7766`, produced
  `e238ff06...b66cb6`. This is a known mismatch of unknown cause, identical on the
  author's baseline, and it makes the command exit 1. See
  [LIMITATIONS.md](LIMITATIONS.md).

**Never run `record`.** `record` overwrites the pinned manifest with whatever the
current code produces, which destroys the thing the harness exists to protect. It
is only appropriate for deliberately re-pinning after a reviewed behaviour
change, and it was not run for this release.
