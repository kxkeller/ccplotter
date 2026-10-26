# Calibration

**The calibration values that ship with this code are the author's measurements
for one machine. Re-measure them on any other machine before you plot.**
G-code generated with someone else's Z heights, reach or pen offsets can drive
the pen into the bed, the frame, or off the paper. The software does not know
what your pen mount looks like; it only knows the numbers it was given.

## What is shipped, and where it lives

| Value | Where | What it is |
|---|---|---|
| Per-pen profiles: `z_touch_abs`, `z_up`, `z_down`, `z_micro`, `pen_mm`, `fill_mm`, speeds, per-pen XY offset | `ccplotter/config.py`, `_PEN_PROFILE_SEED` | The app writes this seed to `pen_profiles.json` in the runtime home when that file does not exist. Four profiles: `energel_05` (active), `platinum_preppy`, `bic_cristal_black`, `bic_cristal_blue`. |
| Legacy rectangular reach: `PEN_MIN_X/Y`, `PEN_MAX_X/Y` (34.0, 27.5 to 247.0, 250.5) and `BED_MAX` | `ccplotter/config.py` | Used when there is no `calibration.json`. |
| Reach polygons (`plotter_2001`, `plotter_2002`, `plotter_2003_v32`), the active reach profile, a global pen offset (-1.3, -43.7), `z_travel`, `mega_hop_mm`, paper and margin defaults | `assets/seed/calibration.seed.json` | A seed for `calibration.json`. It deliberately contains no `z_touch_abs`. **Only the desktop launcher copies it** into a runtime home; a source run never does. |
| `gwrite` profile `elegoo_cc`: start and end G-code, default Z heights and speeds | `vpype.toml` | The machine start/end code is specific to the Centauri Carbon. The default Z values in this file are overridden per job from the pen profile (the example's G-code uses Z values that differ from the ones in this file), but the start/end code is used as written. |
| Test-only calibration and pen profiles | `scripts/harness_env/`, `tests/fixtures/rot_fix_2026_09_01/_env/` | Pinned inputs for the tests; marked as not real calibration. Do not plot with them. |

## What a first run does

Starting from source with an empty runtime home (what `scripts/launch.py`
gives you):

- There is no `calibration.json`. The app reads it as empty (`{}`, with a
  warning on stderr), uses the built-in legacy rectangular reach, and applies
  zero pen offset.
- `pen_profiles.json` is created from the seed with the author's per-pen Z
  heights. `energel_05` is active.
- `resolve_z_touch` only refuses to generate G-code when neither the active pen
  profile nor `calibration.json` has a `z_touch_abs`. The seed always supplies
  one, so **G-code is generated, using the author's Z heights**. Nothing in the
  first run stops you from plotting with them.

The packaged desktop app additionally seeds `calibration.json` from
`assets/seed/calibration.seed.json`, which brings the author's reach polygons and
holder offset along as well. The packaged app was not verified for this release.

## Measuring your own values

Each of these is a physical measurement. Do them with the pen you will use, the
paper in place, and the pen mount in its final position.

### Touch height (`z_touch_abs`) and the Z offsets

`z_touch_abs` is the absolute machine Z, in millimetres, at which the pen tip
just touches the paper, per pen profile. Everything else is relative to it:

- draw height = `z_touch_abs` + `z_down` (negative `z_down` presses into the paper),
- lift height = `z_touch_abs` + `z_up`,
- short-hop height = `z_touch_abs` + `z_micro`.

To measure it: with the pen fitted and the paper on the bed, jog the printer's Z
by hand (from the printer's touchscreen) until the pen tip just touches the paper,
and read the printer's Z. Enter that value as the profile's "Touch Z" in the UI
("Z Heights & Speeds"), or send it with the `/api/pen_profiles/z` route. That
route refuses values outside 5 to 45 (`z_touch_abs`), -3 to 1 (`z_down`), 0.1 to 20
(`z_up`) and 0 to 20 (`z_micro`). If you start the app with
`CCPLOTTER_PRINTER_HOST` set, the `/api/calibrate_z` route can read the live Z
from the printer instead, but that path was not tested for this release.
After measuring, check the result on scrap paper with a short test plot before a
real one. Note that a different pen, a refill versus a full pen body, or a
different mount height all need their own profile.

### Reach and pen offset

Because the pen sits beside the nozzle, the area the pen tip can reach is not the
area the nozzle can reach.

- `pen_offset_x/y` in `calibration.json` is the pen-tip position relative to the
  nozzle, shared by all pens (the G-code commands the nozzle, so the app drives
  to the intended point minus the offset). A pen profile's own `pen_offset_x/y` is
  added on top of it.
- The reach profile in `calibration.json` (`reach_profiles` and
  `active_reach_profile`) is the polygon, in machine coordinates, inside which
  the pen tip may travel. G-code generation refuses to emit any move outside the
  active reach.

The app has no wizard for these. Measure them physically: command the toolhead to
known coordinates (for example with a short plot at a known placement, or the
registration-marks route), measure where the pen actually lands on the paper, and
set the offset from the difference. Find the limits of travel by moving the head
toward each edge by hand and recording the pen-tip position where the head stops,
then keep a safety margin. Write the result into `calibration.json` in your runtime
home (it is a plain JSON file; see the seed for the keys). Until you have a
`calibration.json`, the legacy rectangle from `config.py` applies, which is also the
author's measurement.

## Calibration sheets refuse on other machines

The calibration sheets (dropout, fidelity, fillmode, linestar, penwidth wedge,
pocket, singlepass and solidpitch) are bound to the author's reach. Each
one requires `EXPECT_REACH_ID`, which is `plotter_2003_v32`, and the exact reach
bounding box of that profile, and refuses to generate otherwise. Unless your
`calibration.json` defines a reach profile with that id and bounds, these sheets
will refuse. This is a safety check that was left as written, not a bug; making
the sheets usable on another machine means re-deriving their layouts for your
reach, which has not been done.

## The runtime home and the shadow guard

`pen_profiles.json` and `calibration.json` live only in the runtime home:

- By default that is `%LOCALAPPDATA%\ccPlotter`.
- `CCPLOTTER_HOME` selects another one. It must be an absolute path to an
  existing directory, or the app raises an error; it is never created or
  interpreted relative to the current directory. `scripts/launch.py --home`
  creates the directory for you and refuses a home inside the source tree.
- The server runs with the runtime home as its working directory, which is where
  `work/` and `dbv3_inbox/` appear.

The shadow guard: if the process's current directory contains its own
`pen_profiles.json` or `calibration.json` that is not the very same file as the
resolved one, the app refuses to start (`ShadowConflictError`), even if the copies
are byte-identical. A stray copy would otherwise go stale and silently supply a
wrong Z. There is no override. Do not run the app from a directory that holds
one of these files unless that directory is the runtime home.

On Windows, Python from the Microsoft Store is refused when the home is the default
`%LOCALAPPDATA%\ccPlotter` (Store Python virtualises that folder); use python.org
Python or an explicit `CCPLOTTER_HOME` elsewhere.
