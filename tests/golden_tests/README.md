# Golden Tests — clean line-art regression suite

The clean line-art plotter workflow is the valuable, stable product:

```
OpenAI/ChatGPT clean line-art source -> CC Plotter faithful trace -> route -> G-code -> pen plot
```

These tests exist so that sketch/shading/fill/coverage experiments cannot
silently break that workflow. Before anything reaches `main`, run these inputs through the
real app/API path and confirm the clean line-art result does NOT regress.

## Layout

```
golden_tests/
  inputs/     # source images/SVGs fed into the app
  expected/   # known-good preview SVG/PNG + settings JSON, per checkpoint
```

## How to run a candidate version

1. Clear stale cache first (PowerShell):
   ```powershell
   Remove-Item work\src_*_render_*.svg -ErrorAction SilentlyContinue
   Remove-Item work\src_*_render_*.svg.info.json -ErrorAction SilentlyContinue
   Remove-Item work\debug* -ErrorAction SilentlyContinue
   ```
2. Run through the actual app/browser/API path (not an isolated backend test).
3. Settings (clean line-art mode):
   - Sketch mode: OFF
   - Fill solids: OFF
   - Ink lanes: OFF
   - Coverage repair: OFF
   - Thick-line weight: OFF
   - Connect Strokes = 0
   - Target Z moves = 0
4. Capture: source image, mask_raw, mask_clean, skeleton, final preview, stroke count, Z moves.
5. Save preview + settings JSON into `expected/<checkpoint>/`.

## Pass condition

`clean line-art source -> faithful trace -> clean preview` where the preview visually
resembles the source. It must NOT: add shading, add fill blobs, use coverage repair,
use ink lanes, use sketch hatching, mangle line-art detail more than the older working
app did, invert/hollow filled wheels weirdly, or turn clean art into noisy garbage.

If clean line art regresses, STOP — do not merge.

## Inputs (to populate)

- [ ] `inputs/clean_plane_lineart.png`  — clean OpenAI plane line-art source (not distributed; supply your own)
- [ ] `inputs/clean_tram_city_lineart.png` — clean OpenAI tram/city line-art source
- [x] `inputs/synthetic_text_rect_wheel.svg` — deterministic text + rectangle + filled wheel
- [ ] (optional, experiment-only) `inputs/dense_sketch.png` — a dense sketch input, NOT a main gate
