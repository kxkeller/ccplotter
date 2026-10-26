# Example: synthetic drawing through the default mode (Trace V3)

**This example was generated in software. It was NOT physically plotted.**
`preview.png` is a software preview, not a photograph or scan of ink on paper.
Physical plots are in [../physical-tram/](../physical-tram/README.md).

**Do not send `output.gcode` to a printer.** It was made with the author's pen
profile and the app's built-in reach rectangle, which are measurements for one
machine (see [../../docs/CALIBRATION.md](../../docs/CALIBRATION.md)). On a
different setup this file can drive the pen into the bed.

## Files

| File | What it is |
|---|---|
| `make_input.py` | Generates `input.png` with Pillow primitives only (shapes, rings, hatching, grey bars, text). No photograph or third-party artwork is involved. |
| `input.png` | 900x1200 greyscale source image (output of `make_input.py`). |
| `settings.json` | The exact request body sent to `/api/process`, plus the calibration and pen profile that were in effect. |
| `reproduce.py` | Uploads `input.png`, requests a preview and then G-code with `settings.json`, and writes `output.gcode` and `preview.png`. |
| `output.gcode` | The G-code (ASCII, no `G92`; see "Checks"). |
| `preview.png` | Software preview of the routed strokes, with a red label saying so. |
| `.gitattributes` | Keeps `output.gcode` byte-exact (no line-ending conversion) so the hash below holds after a checkout. |

## What the settings are

- Mode `tracev3` ("Trace (V3)", the default mode in the UI). It thresholds the
  image to an ink mask, so smooth tones become solid ink or bare paper: the
  gradient bar and the light grey bars are mostly lost, which is expected.
- Paper `150x200` mm, margin 10 mm, placed with its front-left corner at machine
  X 40, Y 30. Pen `energel_05`, request `z_down` -0.6, `z_up` 3.5.
- Calibration in effect: no `calibration.json` in the runtime home, so the app
  uses its built-in legacy rectangular reach, X 34 to 247, Y 27.5 to 250.5,
  zero pen offset. This is what a first source run behaves like.
- Pen profile `energel_05` as seeded by the app: `z_touch_abs` 12.05,
  `fill_mm` 0.25, `pen_mm` 0.5, `z_up` 3.5, `z_down` -0.6, `z_micro` 0.6.
  The request itself sets draw/travel/plunge/lift speeds 3000/8000/500/2000.
  Full values are in `settings.json`.

`reproduce.py` refuses to run if the server's reach or pen profile differs from
the ones recorded in `settings.json`.

## Reproduce

From the repository root, with the pinned environment installed
(`pip install -r requirements-lock.txt` in a Python 3.12 virtual environment):

```powershell
# An absolute, empty directory outside the repository (the launcher refuses a
# home inside the source tree). Use a new one for a from-scratch run.
$h = "C:\ccplotter-example-home"
New-Item -ItemType Directory -Force $h | Out-Null

python scripts/launch.py --home $h --port 5093
```

In a second terminal:

```powershell
python examples/synthetic-trace/reproduce.py --base-url http://127.0.0.1:5093 --out C:\ccplotter-example-out
Get-FileHash C:\ccplotter-example-out\output.gcode -Algorithm SHA256
```

Stop the server with Ctrl+C when done. To regenerate the input image:
`python examples/synthetic-trace/make_input.py`.

## Result

```
sha256 98b00623f73bfab4421954173a217834c7a7ea978ecd0c1ec0cc8b3a3aa940ae  output.gcode  (766,352 bytes)
```

The server stamps two header comment lines that differ on every run (the
absolute path of its work directory, and a random run id). `reproduce.py`
replaces exactly those two lines with fixed text before writing the file; no
other byte is changed, and neither line contains motion. The hash above is of the
file as written by `reproduce.py`. The file uses CRLF line endings, as the app
writes them on Windows.

The software model reported 209 strokes and about 10 minutes of plot time. That
time is a model estimate, not a stopwatch measurement.

## Checks run

- Reproduced twice from scratch (server stopped, runtime home deleted and
  recreated, server restarted each time) and then a third time into this
  directory: the G-code hash was identical every time, including after
  `make_input.py` regenerated `input.png` byte for byte.
- G-code decodes as pure ASCII; there is no `G92` command (the only occurrence
  of the text "G92" is in a header comment); fan commands use only the indices
  `S0`, `P2` and `P3`.
- Commanded X is 56.5 to 173.5 and Y is 53.4 to 206.6, inside the reach in
  effect. Z values used: 11.45 (draw), 12.65 (micro-lift), 15.55 and 16.05
  (travel).
- Environment: Windows, Python 3.12.0, packages from `requirements-lock.txt`.
  Other platforms and package versions were not tested.
