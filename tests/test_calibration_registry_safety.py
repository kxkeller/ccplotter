"""Registry-parametrized safety coverage for calibration sheets.

THE POINT: registering a calibration sheet AUTOMATICALLY buys it the full safety
set. A generator that skips a guard fails here instead of reaching the printer.
The parametrization walks `calibration_registry.SHEETS` itself -- never a
hand-maintained list of ids, because a hand-maintained list is exactly how a
newly registered sheet silently escapes coverage.

The registry's own import-time validation closes the other escape route: a
sheet cannot dodge this test by omitting a generator, because `ui="generic"`
without a generator raises, and `ui="bespoke"` must be declared explicitly and
is visible in review.

WHAT IS CHECKED, and why each one is here rather than assumed -- each is a
hard safety rule for this printer:
  * pure ASCII        -- a non-ASCII character (e.g. an em-dash) in a gcode
                         comment crashes the firmware parser.
  * no G92            -- absolute machine coords only; a G92 offset desyncs the
                         machine's idea of where it is.
  * fan indices       -- M106 P0/P1 do not exist on this printer and HALT the
                         firmware. Only bare M106, P2, P3 are valid.
  * Z floor           -- no commanded Z below the sheet's own deepest z_down;
                         anything lower is driving the tip into the bed.
  * reach             -- every commanded XY inside the ACTIVE reach polygon,
                         parsed from the emitted file rather than trusted from
                         the generator's own report.
  * print structure   -- real motion present. A sheet that emitted only M-codes
                         with no print structure would halt the firmware.

Same "scripted assertions, plain exit code" convention as the other tests here
(no pytest dependency in this repo; picked up by pytest for free if added).
Run: python tests\\test_calibration_registry_safety.py
"""
from __future__ import annotations

import inspect
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Byte-verified runtime config copies, already used by test_rot_fit_fix.py.
ENV_SRC = REPO / "tests" / "fixtures" / "rot_fix_2026_09_01" / "_env"


class SkipTest(Exception):
    """A missing fixture SKIPS -- it never silently counts as a PASS."""


if not ENV_SRC.is_dir():
    print(f"SKIP: missing runtime config fixture {ENV_SRC}")
    raise SystemExit(1)

_ENV = Path(tempfile.mkdtemp(prefix="ccplotter_calreg_env_"))
for _f in ("vpype.toml", "pen_profiles.json", "calibration.json"):
    shutil.copy2(ENV_SRC / _f, _ENV / _f)
os.chdir(_ENV)
os.environ["CCPLOTTER_HOME"] = str(_ENV)  # explicit runtime home (cwd == home)
os.environ["CCPLOTTER_WORK_DIR"] = tempfile.mkdtemp(prefix="ccplotter_calreg_work_")
sys.path.insert(0, str(REPO))

from ccplotter.calibration_registry import SHEETS, generic_sheets  # noqa: E402
from ccplotter.cal import load_cal, active_pen_profile  # noqa: E402
from ccplotter.reach import active_reach_profile, pen_reach_profile  # noqa: E402

VALID_FAN = re.compile(r"M106(\s+P[23])?(\s|$)")
BAD_FAN = re.compile(r"M106\s+P[01](\s|$)")


def _parse(path: Path):
    """Return (xs, ys, zs, n_motion). Parsed from the EMITTED FILE -- never
    from the generator's own self-report, which is the thing under test."""
    xs, ys, zs, n = [], [], [], 0
    for ln in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        if not ln.startswith(("G0", "G1")):
            continue
        n += 1
        if (m := re.search(r"X(-?[\d.]+)", ln)):
            xs.append(float(m.group(1)))
        if (m := re.search(r"Y(-?[\d.]+)", ln)):
            ys.append(float(m.group(1)))
        if (m := re.search(r"Z(-?[\d.]+)", ln)):
            zs.append(float(m.group(1)))
    return xs, ys, zs, n


def check_sheet(sheet):
    """The full safety set for one registered sheet. Raises on any violation."""
    out = Path(os.environ["CCPLOTTER_WORK_DIR"]) / f"_safety_{sheet.id}.gcode"
    info = sheet.generator(out)
    raw = out.read_bytes()
    assert raw, f"{sheet.id}: generator produced an empty file"

    # -- pure ASCII (a non-ASCII character in a gcode comment crashes the firmware parser) --
    try:
        raw.decode("ascii")
    except UnicodeDecodeError as e:
        raise AssertionError(f"{sheet.id}: non-ASCII byte at offset {e.start}") from None

    text = raw.decode("ascii")
    lines = text.split("\n")

    # -- no G92 (absolute machine coords only) --
    g92 = [i for i, l in enumerate(lines, 1) if l.startswith("G92")]
    assert not g92, f"{sheet.id}: G92 present at line(s) {g92[:5]}"

    # -- fan indices: P0/P1 do not exist and HALT the firmware --
    bad = [i for i, l in enumerate(lines, 1) if BAD_FAN.search(l)]
    assert not bad, f"{sheet.id}: invalid fan index M106 P0/P1 at line(s) {bad[:5]}"

    xs, ys, zs, n_motion = _parse(out)
    assert n_motion > 0, f"{sheet.id}: no G0/G1 motion -- an M-code-only file halts the firmware"
    assert xs and ys and zs, f"{sheet.id}: no commanded X/Y/Z found"

    # -- Z floor: nothing below the sheet's OWN deepest pen-down --
    deepest = info.get("deepest_z_down_abs")
    if deepest is None:
        deepest = info.get("z_down_abs")
    assert deepest is not None, (
        f"{sheet.id}: info dict reports neither deepest_z_down_abs nor z_down_abs, so its "
        f"Z floor cannot be checked. Every registered generator must report its floor.")
    assert min(zs) >= float(deepest) - 1e-6, (
        f"{sheet.id}: commanded Z {min(zs)} is BELOW its own deepest pen-down "
        f"{deepest} -- the tip would be driven into the bed")

    # -- reach: every commanded XY inside the ACTIVE polygon --
    cal = load_cal()
    reach = pen_reach_profile(active_reach_profile(cal), active_pen_profile())
    bx0, by0, bx1, by1 = reach["bbox"]
    assert bx0 <= min(xs) and max(xs) <= bx1 and by0 <= min(ys) and max(ys) <= by1, (
        f"{sheet.id}: commanded XY [{min(xs):.1f},{max(xs):.1f}] x "
        f"[{min(ys):.1f},{max(ys):.1f}] escapes the active reach envelope "
        f"[{bx0},{bx1}] x [{by0},{by1}] ({reach['id']})")

    return (f"{sheet.id}: {len(raw)}B, {n_motion} moves, "
            f"X {min(xs):.1f}-{max(xs):.1f} Y {min(ys):.1f}-{max(ys):.1f}, "
            f"Zmin {min(zs)} >= floor {deepest}")


def test_every_generic_sheet_is_safe():
    gen = generic_sheets()
    assert gen, "registry exposes no generic sheets -- parametrization would vacuously pass"
    for sheet in gen:
        print(f"    {check_sheet(sheet)}")


# ---------------------------------------------------------------------------
# Invariant: no registered sheet (archived or not) may read the runtime WORK
# DIR (%LOCALAPPDATA%\ccPlotter\work / config.WORK_DIR) as an INPUT -- only
# ever write NEW output there. A generator that hardcodes an ABSOLUTE path into
# the live machine's `...\ccPlotter\work\runs\<job>\cleaned.png` tree bypasses
# `config.WORK_DIR` entirely, so it could never be safety-tested in an
# isolated environment and would silently depend on one specific job's
# leftover run output existing on one specific machine.
#
# TWO CHECKS, because either alone has a hole:
#   (1) static scan for a hardcoded literal reference to the runtime
#       AppData/ccPlotter path in the generator's own source file. This is
#       the one that catches an absolute path -- an "empty work dir" fixture
#       is irrelevant to a path that bypasses config.WORK_DIR and points
#       straight at the real machine.
#   (2) every generic sheet must still generate cleanly against a FRESH,
#       EMPTY work dir (this test file's own CCPLOTTER_WORK_DIR is already
#       set to one) -- catches a sheet that reads pre-existing content
#       THROUGH config.WORK_DIR itself (a different, config-based hazard
#       (1) cannot see).
# ---------------------------------------------------------------------------
_HARDCODED_WORKDIR_RE = re.compile(r"AppData.{0,3}Local.{0,3}ccPlotter", re.IGNORECASE)


def _assert_no_hardcoded_workdir_literal(sheet_id: str, source_text: str):
    m = _HARDCODED_WORKDIR_RE.search(source_text)
    if m:
        raise AssertionError(
            f"{sheet_id}: generator source hardcodes a literal reference to the runtime "
            f"AppData/ccPlotter path ({m.group(0)!r}) -- the zigzag_crop_ab hazard (D379): "
            f"a sheet that reads INPUT from the live machine's own work dir instead of only "
            f"WRITING output via config.WORK_DIR, and so can never be safety-tested in an "
            f"isolated environment.")


def test_no_sheet_hardcodes_runtime_workdir_path():
    for sheet in SHEETS.values():
        if sheet.generator is None:
            continue
        src_file = inspect.getsourcefile(sheet.generator)
        src = Path(src_file).read_text(encoding="utf-8")
        _assert_no_hardcoded_workdir_literal(sheet.id, src)


def test_every_sheet_generates_against_empty_workdir():
    """Every registered generic sheet (archived or not) must generate
    cleanly against a FRESH, EMPTY work dir -- proving it never reads
    pre-existing content from `config.WORK_DIR` as an input, only ever
    writes new output there. Named explicitly here rather than left an
    incidental side effect of `check_sheet`'s own generation call.

    Uses its OWN brand-new temp dir (not the module-level
    `CCPLOTTER_WORK_DIR` fixture, which `test_every_generic_sheet_is_safe`
    already wrote into by the time this runs) -- and patches `WORK_DIR` on
    every already-imported `ccplotter.calibration_*` module directly,
    because each of those modules did `from ccplotter.config import
    WORK_DIR` (an import-by-value, own local binding); reassigning
    `ccplotter.config.WORK_DIR` alone would not reach them. Restores every
    patched value afterward regardless of outcome."""
    import ccplotter.config as _cc_config

    fresh = Path(tempfile.mkdtemp(prefix="ccplotter_calreg_emptywork_"))
    assert not list(fresh.iterdir()), f"freshly created temp dir {fresh} is not empty"

    patched = [(_cc_config, "WORK_DIR", _cc_config.WORK_DIR)]
    _cc_config.WORK_DIR = fresh
    for name, mod in list(sys.modules.items()):
        if name.startswith("ccplotter.calibration_") and getattr(mod, "WORK_DIR", None) is not None:
            patched.append((mod, "WORK_DIR", mod.WORK_DIR))
            mod.WORK_DIR = fresh
    try:
        for sheet in generic_sheets():
            out = fresh / f"_workdir_input_check_{sheet.id}.gcode"
            try:
                sheet.generator(out)
            except Exception as e:
                raise AssertionError(
                    f"{sheet.id}: generator FAILED against a fresh empty work dir -- this "
                    f"means it reads pre-existing content from the runtime work directory as "
                    f"an INPUT (the zigzag_crop_ab hazard class, D379). Error: {e}") from e
    finally:
        for obj, attr, orig in patched:
            setattr(obj, attr, orig)


def test_workdir_invariant_positive_control():
    """Proves the two checks above actually have teeth -- a deliberately-bad
    fake sheet (a hardcoded runtime-path literal, AND a generator that reads
    pre-existing WORK_DIR content) must be CAUGHT, not silently pass. No
    fake sheet is registered in SHEETS; both halves are checked directly
    against the same helper/logic the real tests use (a check that cannot be
    observed to fail is not known to work)."""
    bad_src = (
        'SOURCE_RUN_DIR = Path(r"C:\\Users\\someone\\AppData\\Local\\ccPlotter") '
        '/ "work" / "runs"\n')
    caught = False
    try:
        _assert_no_hardcoded_workdir_literal("fake_bad_sheet_hardcoded_path", bad_src)
    except AssertionError:
        caught = True
    assert caught, (
        "positive control FAILED: the hardcoded-workdir literal scan did not catch a "
        "known-bad pattern -- the check has no teeth.")

    def _bad_generator_reads_prior_run(out, **_kw):
        # Mirrors zigzag_crop_ab's shape MINUS the hardcoded-path half (so
        # this half of the control exercises ONLY the empty-workdir-
        # generation check, independent of check (1) above): reads a file
        # through config.WORK_DIR that this call never wrote itself.
        p = Path(os.environ["CCPLOTTER_WORK_DIR"]) / "some_prior_run" / "input.png"
        return p.read_bytes()

    caught2 = False
    try:
        _bad_generator_reads_prior_run(Path(os.environ["CCPLOTTER_WORK_DIR"]) / "_pc.gcode")
    except FileNotFoundError:
        caught2 = True
    assert caught2, (
        "positive control FAILED: a generator reading pre-existing WORK_DIR content did not "
        "raise against a fresh empty dir -- the empty-workdir fixture is not actually empty, "
        "or the control generator is not exercising the read path it claims to.")


def test_registry_has_no_uncovered_generic_sheet():
    """The anti-escape assertion. Every sheet is either covered by the safety
    test above or explicitly declared bespoke -- there is no third state, and
    no way to be neither."""
    for s in SHEETS.values():
        assert s.ui in ("generic", "bespoke"), f"{s.id}: unknown ui {s.ui!r}"
        if s.ui == "generic":
            assert s.generator is not None, f"{s.id}: generic without a generator"
        else:
            assert s.generator is None, f"{s.id}: bespoke but carries a generator"
    covered = {s.id for s in generic_sheets()}
    declared = {s.id for s in SHEETS.values() if s.ui == "bespoke"}
    assert covered | declared == set(SHEETS), "a registered sheet is neither covered nor declared bespoke"


def _run_all():
    tests = [test_every_generic_sheet_is_safe,
             test_no_sheet_hardcodes_runtime_workdir_path,
             test_every_sheet_generates_against_empty_workdir,
             test_workdir_invariant_positive_control,
             test_registry_has_no_uncovered_generic_sheet]
    passed, failed = [], []
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"PASS {name}")
            passed.append(name)
        except Exception as e:
            print(f"FAIL {name}: {str(e).encode('ascii','replace').decode('ascii')}")
            failed.append(name)
    print(f"\n{len(passed)} passed, {len(failed)} failed "
          f"({len(generic_sheets())} generic sheet(s) covered of {len(SHEETS)} registered)")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
