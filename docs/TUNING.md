# TUNING.md — the knobs, their current values, and what has already been measured

**Built 2026-08-16 at Kai's request:** so that changing direction is "sub in a value",
not "start a research task". Every value below was read from the code or an artifact,
not from memory, on 2026-08-16.

This is an INDEX, not a home. Each row points at where the fact actually lives —
`docs/CURRENT.md` for the frontier, `STATUS.md` for the backlog,
`docs/history/REVIEW_LOG.md` for gate history, `artifacts/*/RESULT.md` for evidence.
Do not copy findings into this file; point at them.

**Nothing here may be changed without the flow in CLAUDE.md rule 1** — a default flip
needs the gating audit numbers in the commit message AND a full-page plot-scale A/B
signed off by Kai. Several of these have measured alternatives that look better on
paper-free metrics; that is not permission.

---

## Tone / greyscale — `ccplotter/trace_v3.py`

| knob | current | provenance | measured alternatives | where |
|---|---|---|---|---|
| `TONE_WINDOW_HI_MULT` | **1.6** | Kai's 2026-07-21 BIC wedge at fill 0.19, patch 13-14, capped "to play it safe" one patch below the measured dead zone | 2.5 / 4.0 / 6.0 give lightest ink 39% / 25% / 17% vs 61% today — **but 1.82x-2.0x was measured as the "lines on white" dead zone at that pen.** See D1a: the cap may need LOWERING, not raising | `artifacts/tone-ramp-2026-08-15/`, `trace_v3.py:263-312` |
| `TONE_WINDOW_LO_MULT` | 0.66 | same wedge, patch 4 = first distinguishable step | none tested | same |
| `TONE_SOLID_MULT` | 0.6 | same wedge, patches 1-3 solid floor. Documentary — does not drive a computation | none tested | same |
| `TONE_LEVELS_N_DEFAULT` | 7 | Kai's reading supports ~7 levels but he was ambivalent (N3 vs N7 never decided). Committed default, **not a validated value** | N=3 never tested | same |
| `tone` mode | `off` | default | `balanced` ruled worse by Kai 08-12/08-13 — **but that negative is DOWNGRADED**, see CURRENT.md | `artifacts/tone-pictures-2026-08-15/` |

**Known plumbing gap:** `tone_levels_n` is not forwarded by `routes/core.py`, so the
engine is pinned to 7 and the band count cannot be tuned from the API or UI. In STATUS.md.

## Trace / mask — `ccplotter/routes/core.py` defaults

| knob | current | notes | where |
|---|---|---|---|
| `threshold_mode` | `otsu` | V3.1 default. Sauvola/adaptive tested: cut ice ink 46%, fragmented strokes (median 2.17→~1.4). Negative on soft input | `artifacts/tracev3-anime-tune-2026-08-09/` |
| `pre_blur` | `3` | V3.1 (3,3) Gaussian. `0` is the V3.2 opt-in | CLAUDE.md rule 4 |
| `native_res` | `False` | V3.2 opt-in, only helps downscaled sources | CLAUDE.md rule 4 |
| `hole_fill` | `-1` (auto) | bubble fix | — |
| `clahe` | `False` | **but the approved anime recipe and the locked B/W config both turn it ON per-request** | `artifacts/tracev3-anime-tune-2026-08-09/` |
| `texture_routing` | `band_continuity_v1` | default for tracev3 since 2026-07-19. **Deliberately does NOT fire on branchy/networked mesh** (`trace_v3.py:834-844`) — which is exactly what costs bike and a350 their structure | `artifacts/bw-outline-diagnosis-2026-08-15/` |

## Scale adaptation — `ccplotter/config.py`

| knob | current | notes |
|---|---|---|
| `SCALE_AWARE_V2_K` | 0.75 | regrounded 2026-07-19. Drawn spacing = 0.75 x fill_mm. Feeds the per-pixel depth threshold that drops thin structure (`0.5 * fill_px`) |
| `SCALE_AWARE_V2_FILL_PX_FLOOR` | 3.0 | floor-rescue only: bumps resolution when the placed pen would quantize below 3px |
| `SCALE_AWARE_V2_NATIVE_CAP` | 5480 | cap on that rescue |
| `BED_MAX` | 257.0 | Elegoo Carbon, square. **The preflight's paper suggestion routinely exceeds this** — see D5 |
| `SIMPLIFY_FILL_MULT` | 0.0 | **OFF PERMANENTLY.** Kai 2026-07-26: cracking outranks plot time. Do not propose as a time lever |

**The number to know (derived and verified 2026-08-16,
`artifacts/trace-at-pen-2026-08-16/`):** the structure-drop threshold collapses
algebraically to

```
threshold_mm = 0.5 * fill_px_real * mm_per_px_real  =  0.375 * fill_mm
```

Identical across all 12 test jobs regardless of image or resolution-floor boost.
**Anything whose local half-width falls below 0.375 x fill_mm is not drawn at all.**

| pen | `fill_mm` | drop threshold |
|---|---|---|
| `platinum_preppy` (mounted) | 0.275 | **0.1031 mm** |
| `energel_05` / `uright_gel` | 0.25 | 0.0938 mm |
| `bic_cristal_*` | 0.12 | **0.0450 mm** |
| `preppy_05` | 0.40 | 0.1500 mm |

### Crosshatch halves your gap — the constraint that is not obvious (2026-08-17)

**Crosshatching two directions at a shared pitch does NOT give you that pitch.** The
effective opening is roughly

```
gap ≈ pitch / √2  −  line_width
```

because the diagonal of the resulting square cell, not its side, is what the eye and the
pen see. **A pitch that is legal in one direction becomes illegal when crossed.**

Measured on `artifacts/ink-mm-2026-08-17/`: a nominal 1.0 mm crosshatch — comfortably
above DIRECTION §7's 0.55 mm visible-gap ceiling at the mounted pen — delivered an
effective **0.43 mm** and read as a solid slab. Floored properly, ice's median gap went
**0.525 mm → 0.788 mm** and the layer became genuine crosshatch texture.

So the §7 ceiling must be applied to the EFFECTIVE gap, not the nominal pitch. At
`fill_mm = 0.275`, a crosshatch needs a nominal pitch of **1.167 mm** to clear 0.55 mm.
This cost a round of debugging in the ink-synthesis campaign; it is the kind of thing
that looks fine in every number until it is on paper.

Ready reckoner at the mounted pen (`fill_mm = 0.275`, ceiling 0.550 mm), computed from
the formula and cross-checked against the measured 0.43 mm at 1.0 mm:

| nominal crosshatch pitch | effective gap | |
|---|---|---|
| 0.80 mm | 0.291 mm | SUB-CEILING |
| 1.00 mm | 0.432 mm | SUB-CEILING — this is the bug that was found |
| **1.167 mm** | **0.552 mm** | the exact floor |
| 1.30 mm | 0.644 mm | OK |
| 1.50 mm | 0.786 mm | OK |
| 2.81 mm | 1.712 mm | the loco reference's own pitch — comfortably legible |

Note the last row: the reference illustration sits far above the floor, which is part of
why it plots well and is worth keeping in view when choosing a hatch density.

For scale: a350's failing outline region has a median local half-width of **0.108 mm** —
sitting essentially ON the mounted pen's threshold. That is why it is unstable there and
clean at the BIC. bike sits at 0.104 mm and does NOT recover at any pen, because its
failure is the branchy-mesh exclusion in `band_continuity_v1`, not this threshold.

**Not implemented, often assumed to be:** render resolution does NOT adapt to plot size.
`resolution` is a manual per-item number with no link to the paper dropdown
(`render.py:56-76`, `index.html:1010`). What exists is scale-adaptation-v2, which adapts
stroke GEOMETRY to real placed scale — a different layer. Established 2026-08-15.

## Low-lift router walk cap — `ccplotter/stats_warnings.py`

`ROUTE_WALK_MAX_SEGS_PLAIN = 20000` (single-pen path, default) and `ROUTE_WALK_MAX_SEGS_TWOPEN = 100000` (two-pen key layer only, passed by `twopen_keytint.generate_layers`). A layer above its cap is not routed: WARNING logged, `route_skipped_guard` / `route_skipped_layers` set. The plain cap stays 20000 pending Kai's sign-off on `artifacts/router-guard-2026-10-21/ab/JOIN_AB.html` (D1699); raising it changes default single-pen output.

## Front end — per-request recipes, no defaults

| recipe | value | status |
|---|---|---|
| anime / sketch | `anime_style@512 + @1024 → min-blend → whiten outer 2% band → tracev3 stock + CLAHE, tone off` | approved by Kai 2026-08-11. **Per-request, never made a preset** |
| B/W | `XDoG p02_e55 @ shorter side 512 → tracev3 stock + CLAHE, tone off` | locked by Kai 2026-08-13. `p02` = **phi 2.0**, not 0.2 |
| XDoG phi | 40 / 8 / 2 tested; **2.0 locked** | phi 40 at the same 512 measured CLEANEST of four cells (median stroke 4.00 vs 2.87mm, ink gain 15.4% vs 27.7%) at 1.16-1.62x the ink — decision D2 |
| XDoG eps | 0.45 / 0.55 / 0.65 tested; **0.55 locked** | Kai: "e55 most consistently" |
| source side | 256 / 341 / 512 / 768 / 1024 / 1536 tested | 512 locked for B/W; 1024 for anime. 1536 tested, tiled inference NEGATIVE |

Inert on the tracev3 path, despite being in the UI: **`ink_sensitivity` and `gamma`** —
they only feed the lineart/sketch engines. In STATUS.md.

## Pens — and WHICH pen file you are actually reading

> **STALE, FLAGGED 2026-09-15 — the "MOUNTED" pen below is out of date. Do not read the
> mounted pen from this file.**
> This section was last written 2026-08-17 (`9886d56`). The live runtime
> `%LOCALAPPDATA%\ccPlotter\pen_profiles.json` (read via PowerShell, mtime 2026-09-14
> 20:07) records **`active` = `uright_gel`**, `pen_mm` 0.255, `fill_mm` 0.25,
> `core_mm` 0.18, `z_touch_abs` **11.45** — and every September 2026 experiment pins
> `uright_gel` explicitly. The `platinum_preppy` row below also records `z_touch_abs`
> 20.95, while the live profile now reads **21.15**.
> The numbers in this section are left unedited on purpose: they are correct *as of
> 2026-08-17* and several conclusions elsewhere were computed against them. Anything
> derived from "the mounted pen" here must be recomputed before reuse.
> **Physical confirmation of which pen is in the holder is Kai's, and is owed** — the
> uRight/Preppy `z_touch_abs` differ by 9.70mm, which is a machine-safety difference, not
> a cosmetic one (DIRECTION §10).

**Two different files are both legitimately "the active pen". Always say which.**

| context | file | active | `fill_mm` |
|---|---|---|---|
| **Kai's machine / the app** | `%LOCALAPPDATA%\ccPlotter\pen_profiles.json` | `platinum_preppy` | **0.275** |
| **Any experiment that chdirs into a harness env** | `scripts/harness_env/pen_profiles.json` | `energel_05` | **0.25** |

Config paths are CWD-relative, so an experiment measures **0.25 unless it explicitly
passes `pen_id`**. Several artifacts this week are labelled "live pen 0.275" — check
whether the script actually set `pen_id`, because the difference is not cosmetic: it
flips which benchmark cells pass the gap audit (tram@257x176 reads 100% gap survival at
0.25 and 20.5% at 0.275; belltower only collapses at 0.275, its median 0.274 straddling
the line).

### Profiles on file (runtime, not the repo seed)

| profile | `pen_mm` | `fill_mm` | notes |
|---|---|---|---|
| **`platinum_preppy`** | 0.25 | **0.275** | MOUNTED. Settled 2026-07-28. `z_touch_abs` 20.95 still unverified on paper |
| `preppy_05` | 0.5 | 0.40 | 20.75 vs 20.95 discrepancy is Kai's |
| `bic_cristal_black` / `_blue` | 1.0 | 0.12 | **the only pens where no benchmark source lands `over`** — D3 |
| `energel_05` | 0.5 | 0.25 | |
| `orenz_nero_03` | 0.3 | 0.30 | |
| `uright_gel` | 0.5 | 0.25 | |

Pen width is NOT the tonal constraint (measured: lightest tone matches within ~1.3pp
between 0.275 and 0.12). It IS a strong plottability lever.

## Reusable experiment infrastructure

- **Driving the pipeline without a socket:** `artifacts/tone-pictures-2026-08-15/01_generate_traces.py`
  — `os.chdir` into a copy of `scripts/harness_env`, then the real Flask app via
  `app.test_client()`. Required when patching a module-level constant, since a
  subprocess server cannot see the patch. Never two servers on :5001.
- **Plot-scale measurement:** `artifacts/bw-filter-vs-plot-2026-08-15/02_render_and_measure.py`
  — rasterize routed polylines at true deposited width, retained/gained/lost.
  **Use exact vector geometry (shapely buffer/union), never raster pixel counting** —
  raster counting overestimated coverage by 9-18 points and was caught only by an
  isolation test.
- **Config-driven sweeps:** `artifacts/tone-presimplify-2026-08-15/` and
  `artifacts/anime-plotscale-2026-08-16/` carry their sweep definitions in a config
  block at the top, so the next round is a value change rather than a new script.
- **Harness baseline:** `scripts/replay_harness.py verify` → expect the **sole mismatch to be `calibration_plot` `gcode_sha256`**
  (currently **9/10 + fan_probe**; the count moves whenever a fixture is added -- judge by KIND). Any other number is a real regression.

## Opt-in blue margin under black (T1 + X + F) -- `ccplotter/twopen_keytint.py`

Body flag `blue_trap` (bool, default false; UI checkbox "Blue margin under black (fewer pen lifts)" in the two-pen options). OFF is byte-identical to before. All thresholds are in pen widths of the active BLUE profile's `fill_mm` (W), never pixels, so they scale with resolution and pen. Constants (also in `twopen_keytint.CONSTANTS["blue_trap"]`, echoed in the run's `info["blue_trap"]` and provenance `resolved.keytint`):

- `TRAP_MARGIN_PEN = 1.0` -- T1 (D1709, D1713-D1720; Kai 2026-10-21): blue mask = blue_A | (dilate(blue_A, 1 W) & black_A).
- `TRAP_HOLE_RADIUS_PEN = 1.5` -- holes of the blue mask smaller than pi*(1.5 W)^2 mm2 (0.1018 mm2 at W = 0.12) are closed (floor px, strict <; trap DEVIATIONS 1).
- `TRAP_FOLD_RADIUS_PEN = 1.5` -- F (D1704-D1705): black components smaller than pi*(1.5 W)^2 mm2 are reassigned to blue, never deleted (union ink conserved, asserted at run time).
- `TRAP_BLUE_CONNECTOR_MM = 0.5` -- X (D1704-D1712): blue connectors at 0.5 mm, mask-checked against the blue + black union (D928); blue is routed with `ROUTE_WALK_MAX_SEGS_TWOPEN`.

Black is the A black minus the folded specks, routed at the body's connector_distance as OFF. Port parity vs the prototype: `artifacts/twopen-trap-2026-10-21/port_parity/`. Time figures remain unvalidated models (b)/(c); nothing plotted. W is the blue profile's `fill_mm`, so trap thresholds scale with it; `separate()` uses a fixed `ROLE_WIDTH_MM` = 0.12 nominal. They coincide for BIC and diverge for another pen (trap scales, separation does not).

## Opt-in "Hatched blue (lighter, fewer pen lifts)" -- `hatched_blue` (default OFF) -- `ccplotter/twopen_hatch.py`

Off by default. Blue is drawn as long hatch lines instead of solid fill: lighter, about half the blue coverage, a different look (paper shows between the lines), fewer pen lifts. A few small gaps can appear in dark areas. Tiny black specks (under 0.1 mm2) are drawn in blue (the blue-margin speck fold is implied, D1773). This is arm B3s of `artifacts/twopen-blue-redesign-2026-10-21/` as confirmed in `artifacts/twopen-hatchblue-confirm-2026-10-21/` (D1736-D1743, D1761-D1768). UI checkbox "Hatched blue (lighter, fewer pen lifts)". OFF is byte-identical to before.

Mapping (one global rule, no per-image constants; the prototype's per-image B1c/B3c threshold bisection and the H1/H1v gain arms are NOT part of this option). W = the BLUE profile's `fill_mm` (like `blue_trap`). Constants (in `twopen_hatch.py`, echoed in `CONSTANTS["hatched_blue"]` and the run's `info["hatched_blue"]`):

- `SIGMA_PEN = 1.0` -- the blue target is the 3x3-blurred grey smoothed by a Gaussian of sigma 1 W (B1, pen scale; sigma 0 would reproduce the traced blue). Coverage c = clip((t_blue - grey_s)/(t_blue - T), 0, 1), c = 1 on black; t_blue from the request's blue_amount.
- `LEVELS = 8`, `PITCH_PEN = 1.0` -- line pitch W (densest, c = 1) up to 8 W (c = 1/8); line k at a place is drawn iff ceil(8c) > bit-reversal(k mod 8): an 8-level ordered dither over the line index. `ANGLE_DEG = 45`.
- `CLIP_PEN = 0.5` -- hatch only where the distance to the outside of (blue u black) is >= 0.5 W + 1 raster px = 0.11 mm at W 0.12, so a pen of width W stays inside the region (redesign DEVIATIONS 8; the 1 px is the 0.05 mm hatch raster).
- `MIN_RUN_PEN = 1.0` -- runs shorter than 1 W are dropped (a dot is not a line).
- `LINK_RADIUS_PEN = 8.0` -- switchback link radius 8 W (the largest pitch the dither makes, so an adjacent drawn line is always reachable); preference forward line, then same line, then backward line, then nearer; a link is taken only if the straight connector passes the app's `stage_e_mask` predicate against the final trace mask of blue u black. No candidate -> pen lift.
- `LINK_CANDIDATES = 40` -- a COST GUARD (mask tests per exit point), not a quality constant.
- `WALK_CAP_UNUSED = 1,000,000` -- the redesign's hatch walk cap applied to the superseded router-linked arm B3 only; B3s uses the explicit linker, which has no walk cap. Recorded for traceability, not live.
- `HATCH_PPM = 20`, `RUN_STEP_MM = 0.025` -- the grid the clip / coverage / runs are evaluated on (0.05 mm/px; half a px along a line).

Precedence with the other two boxes: hatched blue REPLACES the blue layer (blue_trap's T1 blue mask and 0.5 mm blue connector are not used); the black layer is blue_trap's F black (specks below pi*(1.5 W)^2 reassigned to blue, never deleted) -- this is what the prototype used, so ON implies that black whether or not "Blue margin under black" is ticked (`info["blue_trap"]` then reads enabled false with `black_fold_implied_by_hatched_blue`). "Stay down over ink" still applies to the black layer (its own mask, as before); on the blue layer it is a no-op because the hatch linker already stays pen-down only inside blue u black (`info["stay_down_over_ink"]["layers"]["blue"]` says so).

**Disclosures (prototype numbers, models unvalidated, nothing plotted; the job does NOT compute them, they are in `CONSTANTS["hatched_blue"]["prototype_measurements"]`):** pen lifts tram 37,345 -> 20,550, heli 10,401 -> 5,247 (vs `blue_trap` solid, model (c) 9.82 -> 7.47 h tram, 3.22 -> 2.14 h heli). Blue coverage of the target 0.92/0.97 (solid) -> 0.55/0.51 (hatched), tram/heli. Tone cost (model renders): mid tones lighter by +2.2..+2.6 L* on the tram, +5..+8 L* on the heli. Crack figures: area uncovered by either pen tram 1.36% -> 0.18% (better), heli **0.07% -> 0.27% (worse)**; area uncovered by black unchanged (1.55% tram, 2.33% heli). Port parity: `artifacts/hatchblue-port-2026-10-21/` (ON == B3s pointwise, max diff 0.0 mm, tram 20,550 and heli 5,247 lifts).

## Opt-in "Stay down over ink" -- `stay_down_over_ink` (default ON since 2026-10-25)

After routing, a pen-up hop becomes a pen-down connector of any length iff the straight line stays inside that layer's own ink mask (`stage_e_mask`, D928 predicate; no usable mask = nothing joined, with a warning). **Rule (D1782, replaces the D1744 ride-along):** at or below `ROUTE_WALK_MAX_SEGS_PLAIN` (20000 strokes) OFF's ordinary "join nearby" connectors (Connect Strokes) are today's default and ON keeps them exactly, only ADDING checked stay-down joins, so ON never adds lifts on the plain route; on the two-pen key layer above the 20,000 cap it replaces unchecked joins with checked ones and can add lifts (tram black +13%, D1823). On a layer above that cap (e.g. tram: 20,129) OFF skips the router and has no joins; ON lifts the walk cap to `ROUTE_WALK_MAX_SEGS_TWOPEN` (100000) for ordering, and the ordinary short joins there are mask-checked with the same stay-down check (a layer with no usable mask joins nothing), so the only unchecked effect of the lifted cap is travel order, which puts no ink down. Layers with a connector-mask check keep it. Parity record: `artifacts/stay-down-parity-2026-10-21/`; re-measure: `artifacts/stay-down-checked-2026-10-21/`.

**Default ON (2026-10-25).** Provenance: Kai's "yes" to Question 1 on `artifacts/default-flip-audit-2026-10-21/AB_STAYDOWN.html` (chat 2026-10-25), gates D1784 / D1794-D1800, scoping D1782 (efe1912f); numbers in `artifacts/stay-down-checked-2026-10-21/TABLE.txt`. The key ABSENT from a request means ON, in both `/api/process` (`routes/core.py`) and `/api/twopen_keytint` (`routes/twopen.py`); a PRESENT `false` / `null` / junk is the old OFF path, byte-identical to the former default. The UI checkbox starts checked and always sends the explicit boolean. Library functions (`_composite`, `twopen_keytint.generate`) keep `False` as their signature default; the two-pen separation reference job is pinned OFF. **The installed build 5a2ae7ac still runs the earlier rule on layers over 20,000 strokes** until the next rebuild. outline_rows is NOT changed (waits for the heli paper plot, D1754/D1785). Install note: because absent = ON, any caller that omits the key now gets ON.

## "Row direction" -- `row_angle_rule` -- DEFAULT `a135` ("/") since 2026-11-04 for rows / outline_rows jobs (paper rule knowingly waived; select in the UI, request-body key) -- `ccplotter/core_fill.py`

**Default flip 2026-11-04 (Kai, pre-default-change gate D1989-D1990).** For `core_fill=rows` / `outline_rows` on plain Trace V3 and B/W photo (illustration pins rings; Black Line + Blue Tone does not use the key), an ABSENT or empty `row_angle_rule` now means `a135`: rows run bottom-left to top-right ("/") on the preview of a normal landscape job (item rotation 0). **Kai knowingly waived the standing paper rule** (D1831 "no row-direction default without a paper plot", former text of this entry) and chose to flip without a paper plot; nothing about this default has been seen on paper. An explicit `zero` is the former 0-deg behaviour, byte-identical to before (the UI now always sends the select value explicitly when rows / outline_rows is chosen, "As today (0)" included). Rings / hybrid / zigzag jobs are unchanged. **Cost: +25-55 % generation time on every rows / outline_rows job** (any non-zero angle; the study figure, not re-measured for the default). **Item rotation flips the sense:** the angle is in the image frame (y down), so at item rotation 90 / 270 the same default shows as "\" on the preview, and `a45` ("\" at rot 0) shows as "/". **Evidence of sense:** `artifacts/rows-slash-default-2026-11-04/`. **D690 (real pen anisotropy) is still open and unmeasured**: whether a pen lays a different stroke along "/" than along 0 deg is unknown, and the paper sheet `rowangle_bic` has not been plotted.

Applies only with `core_fill=outline_rows` or `core_fill=rows` (an explicit non-zero rule with any other core_fill is a 400). Values: `zero` (explicit: ONE global angle, ROWS_ANGLE_DEG = 0 deg of the image frame, output byte-identical to before the option existed, no audit field), `a45` (every region 45 deg of the image frame: "\" at item rot 0), `a135` (every region 135 deg = -45: "/" at item rot 0; the default; audit histogram bucket "135"), `across` (rows along each region's SHORT axis: pixel-PCA axis + 90 deg), `set` (each region 0 / 45 / 90 / 135 deg picked by an integer hash of the region's rounded centroid x, rounded centroid y and pixel count: `(round(cx)*73856093) ^ (round(cy)*19349663) ^ (n_px*83492791) % 4`; deterministic, no RNG state). **`across` fallback: a region whose pixel-PCA eigenvalue ratio is below 1.3 (near-round) or that has fewer than 3 pixels gets 45 deg instead** (12-36 % of regions on the study images; counted in the audit field). Angles are IMAGE-frame: on the page they are rotated by the item rotation, so production's 0 is vertical on the page for heli/tram (rot 90). When the rule is not `zero` (so also for the default) the trace audit carries `row_angle_rule`, `row_angle_hist` ({"0","45","90","135"} region counts) and `row_angle_fallback_regions`.

Status: study (artifacts/rowfill-angle-2026-10-26/, D1826-D1832): direction does not change solid coverage once the stamp-lattice artifact is removed (D1827/D1829); it is a LOOK choice. A non-zero angle costs +25-55 % generation time per job. ALONG (rows along the long side) and ADIR (dominant edge direction) were measured but are NOT offered. Real pen anisotropy (D690) and dimples are unmeasured offline; the paper sheet `rowangle_bic` (registry, artifacts/rowfill-paper-sheet-2026-10-26/) compares ZERO / A45 / ACROSS / SET on paper. The default was changed on Kai's explicit waiver (see top of this entry), not on a paper result.

## "Structure-aware binary" -- `bw_structure_binary` -- DEFAULT ON since 2026-11-04 (screen evidence; paper owed; B/W photo mode only) -- `ccplotter/structure_binary.py`

**Default flip 2026-11-04 (Kai, D1983-D1986; basis: the on-screen A/B page `artifacts/structure-binary-ab-2026-11-04/`, nothing plotted, paper owed).** Request key `bw_structure_binary`: key absent = ON. OFF only on a strict falsy value (`false`, `0`, `"0"`, `"false"`, `"False"`); anything else (junk, null, "", "no") = ON, so a typo cannot silently change the trace. Ignored by every style except `bwphoto` (tracev3 / illustration byte-identical). OFF output equals the pre-flip bwphoto output. UI: checkbox "Structure-aware binary", CHECKED by default, shown only in B/W photo mode, not persisted; in bwphoto the UI always sends an explicit boolean (like `stay_down_over_ink`). **Two-pen pin (D1986):** `twopen_keytint.capture_plain_bwphoto` pins `bw_structure_binary: False` on its plain reference job, so Black Line + Blue Tone output is byte-identical to before the flip (`artifacts/structure-binary-default-2026-11-04/RESULT.md`). **The mountains problem is still unsolved** (the eagle's black mountains still draw as a dark mass; mode B does not remove them; see artifacts/structure-aware-binary-p3-2026-11-01/RESULT.md). Original opt-in description follows. Port of M11 mode B (`artifacts/structure-aware-binary-p3-2026-11-01/scripts/sab_lib3.py`, `mode_B`). The bwphoto CLAHE/threshold ladder still runs, but only to yield the candidate ink K; the mode's final binary (ink = 0) replaces the raster handed to Trace V3's entry (bilevel fast path: threshold 128, no blur). Two global steps, ONE fixed parameter set (`structure_binary.PARAMS`; all scales physical mm via the job's own mm/px): (1) drop connected K components with area < 3.0 mm2 AND mean edge strength E < 0.30 AND mean local sharpness < 0.35 (diffuse texture); (2) add persistent (sigma_2 and sigma_3 zero crossings within 0.72 mm), coherent (>= 0.55), long (>= 8 mm) contours of the source luminance as 0.30 mm lines where K has no ink within 2.5 mm; then drop components < 0.035 mm2. Analysis raster pitch 0.09 mm, scales 0.12 x 2^k mm. No per-image parameters, no semantic classification.

Measured offline only (nothing plotted; offline mask-and-render figures from `artifacts/structure-aware-binary-p3-2026-11-01/RESULT.md`): M11 pass 3, 15-image benchmark, eagle pen-downs 3492 -> 2614 (-25 %), ink 98 %, sharp thin recall 0.998, crack unchanged. Provenance: M11 pass 3 approved as information (D1956-D1963) + Kai's decision 2026-11-01 to ship mode B as an experimental opt-in (mode C dropped). Recorded per job in the render `info.json` -> `audit.structure_binary` (version, full params, mm/px, K/R ink fraction, components and ink px dropped/added, binary sha). Evidence: `artifacts/structure-binary-optin-2026-11-01/RESULT.md`.

Caveats (reviewer D1964-D1968): **per-job time cost** eagle +10.6 s, belltower +12.1 s (two mode-B runs per job, prelim + final; venv; measured wall-clock of the offline parity jobs, not validated beyond those two). `native_res` is ignored when ON (Trace V3 gets the binary, never the native gray). The D994 `trace_boundary` `gray_record` hashes the PRE-mode production raster; the mode's binary hash is `audit.structure_binary.R_sha16`. With debug on, the binary is also written as `<debug_prefix>_structure_binary.png` (beside `_xdog.png`). Any plot-time figure for an ON job is a model-only, unvalidated number from the app's estimate (preview model, known ~2x high; no time figure has been stopwatch-validated).
