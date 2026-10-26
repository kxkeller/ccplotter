"""pen_profiles.json / calibration.json resolve from the RUNTIME HOME,
never the process CWD.

Every refusal test is a SUBPROCESS in a scratch CWD with an explicit CCPLOTTER_HOME, and each
is paired with a POSITIVE CONTROL (same setup, no shadow / CWD == home -> succeeds), so none of
them can pass merely because the child crashed for an unrelated reason.

Run:  python -m pytest tests/test_runtime_home.py     (never with the repo root as a CWD
that holds a pen_profiles.json / calibration.json -- that is exactly what is refused.)
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ccplotter import runtime_home as rh  # noqa: E402  (import-light, no config side effects)

CAL_SEED = REPO / "assets" / "seed" / "calibration.seed.json"
PEN_A = {"active": "x", "profiles": {"x": {"label": "A", "z_touch_abs": 12.05}}}
PEN_B = {"active": "x", "profiles": {"x": {"label": "B", "z_touch_abs": 14.85}}}


def _run(code, cwd, home=None, extra_env=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env.pop("CCPLOTTER_HOME", None)
    env.pop("CCPLOTTER_WORK_DIR", None)
    if home is not None:
        env["CCPLOTTER_HOME"] = str(home)
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-c", code], cwd=str(cwd), env=env,
                          capture_output=True, text=True, timeout=120)


def _dirs():
    home = Path(tempfile.mkdtemp(prefix="d661_home_")).resolve()
    cwd = Path(tempfile.mkdtemp(prefix="d661_cwd_")).resolve()
    return home, cwd


# -- resolver ---------------------------------------------------------------

def test_default_home_is_localappdata_ccplotter(monkeypatch, tmp_path):
    monkeypatch.delenv(rh.ENV_VAR, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert rh.runtime_home() == tmp_path / "ccPlotter"


def test_explicit_home_must_be_absolute_and_existing(monkeypatch, tmp_path):
    monkeypatch.setenv(rh.ENV_VAR, str(tmp_path))
    assert rh.runtime_home() == tmp_path.resolve()                      # positive control
    monkeypatch.setenv(rh.ENV_VAR, "relative/home")
    with pytest.raises(rh.RuntimeHomeError):
        rh.runtime_home()
    monkeypatch.setenv(rh.ENV_VAR, str(tmp_path / "does_not_exist"))
    with pytest.raises(rh.RuntimeHomeError):
        rh.runtime_home()


def test_desktop_data_dir_is_the_same_resolver(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ccplotter_app_under_test",
                                                  REPO / "desktop" / "ccplotter_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.delenv(rh.ENV_VAR, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert mod.data_dir() == rh.runtime_home() == tmp_path / "ccPlotter"


def test_desktop_seed_defaults_seeds_calibration_from_the_seed_file(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ccplotter_app_under_test2",
                                                  REPO / "desktop" / "ccplotter_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res, data = tmp_path / "res", tmp_path / "data"
    res.mkdir(); data.mkdir()
    (res / "calibration.seed.json").write_bytes(CAL_SEED.read_bytes())
    mod.seed_defaults(res, data)
    assert (data / "calibration.json").read_bytes() == CAL_SEED.read_bytes()
    (data / "calibration.json").write_text('{"user": "tuned"}')           # never overwritten
    mod.seed_defaults(res, data)
    assert json.loads((data / "calibration.json").read_text()) == {"user": "tuned"}


def test_seed_file_present_and_repo_root_has_no_calibration_json():
    assert CAL_SEED.is_file()
    # Pinned over the file with LF line endings (the stored form); the author's Windows
    # checkout has CRLF, whose SHA-256 starts with 9da70921. Same content either way.
    assert hashlib.sha256(CAL_SEED.read_bytes().replace(b"\r\n", b"\n")).hexdigest().startswith("22c55168")
    assert not (REPO / "calibration.json").exists(), "repo-root calibration.json is a CWD shadow (D848)"


# -- refuse on disagreement (subprocess, with positive controls) -------------

IMPORT_CFG = "import ccplotter.config; print('CONFIG_OK')"


def test_config_import_refuses_differing_cwd_pen_profiles():
    home, cwd = _dirs()
    (home / "pen_profiles.json").write_text(json.dumps(PEN_A))
    # positive control: no CWD copy -> starts
    ok = _run(IMPORT_CFG, cwd, home)
    assert ok.returncode == 0 and "CONFIG_OK" in ok.stdout, ok.stderr[-600:]
    (cwd / "pen_profiles.json").write_text(json.dumps(PEN_B))
    bad = _run(IMPORT_CFG, cwd, home)
    assert bad.returncode != 0
    assert "ShadowConflictError" in bad.stderr and "REFUSING" in bad.stderr, bad.stderr[-800:]
    assert "CONFIG_OK" not in bad.stdout


def test_config_import_refuses_differing_cwd_calibration():
    home, cwd = _dirs()
    (home / "calibration.json").write_text(json.dumps({"z_touch_abs": 12.05}))
    (cwd / "calibration.json").write_text(json.dumps({"z_touch_abs": 16.05}))
    bad = _run(IMPORT_CFG, cwd, home)
    assert bad.returncode != 0 and "ShadowConflictError" in bad.stderr, bad.stderr[-800:]


def test_cwd_file_with_missing_home_file_is_refused_not_adopted():
    home, cwd = _dirs()                                   # home has NO pen_profiles.json
    (cwd / "pen_profiles.json").write_text(json.dumps(PEN_B))
    bad = _run(IMPORT_CFG, cwd, home)
    assert bad.returncode != 0 and "MISSING" in bad.stderr, bad.stderr[-800:]
    assert not (home / "pen_profiles.json").exists()      # nothing was seeded from the CWD copy


def test_identical_cwd_copy_is_refused_and_names_the_file():
    """Byte-identical is NOT an exemption (no escape hatch). Positive control: the SAME
    setup with the CWD copy removed, or with CWD == home (samefile), proceeds."""
    home, cwd = _dirs()
    (home / "pen_profiles.json").write_text(json.dumps(PEN_A))
    (cwd / "pen_profiles.json").write_text(json.dumps(PEN_A))
    code = ("from ccplotter.cal import load_pen_profiles\n"
            "print('Z', load_pen_profiles()['profiles']['x']['z_touch_abs'])")
    bad = _run(code, cwd, home)
    assert bad.returncode != 0 and "ShadowConflictError" in bad.stderr, bad.stderr[-800:]
    assert str(cwd / "pen_profiles.json") in bad.stderr and "BYTE-IDENTICAL" in bad.stderr
    assert "Z 12.05" not in bad.stdout
    assert (cwd / "pen_profiles.json").read_text() == json.dumps(PEN_A)   # untouched
    (cwd / "pen_profiles.json").unlink()                                   # control: no shadow
    ok = _run(code, cwd, home)
    assert ok.returncode == 0 and "Z 12.05" in ok.stdout, ok.stderr[-800:]
    ok2 = _run(code, home, home)                                           # control: CWD == home
    assert ok2.returncode == 0 and "Z 12.05" in ok2.stdout, ok2.stderr[-800:]


def test_shadow_dropped_after_startup_is_refused_on_next_access():
    """The per-access guard: a file appearing AFTER config import still blocks gcode-path reads."""
    home, cwd = _dirs()
    (home / "pen_profiles.json").write_text(json.dumps(PEN_A))
    code = ("import json, pathlib\n"
            "from ccplotter.cal import load_pen_profiles\n"
            "assert load_pen_profiles()['profiles']['x']['z_touch_abs'] == 12.05\n"
            "pathlib.Path('pen_profiles.json').write_text(json.dumps(%r))\n"
            "try:\n    load_pen_profiles()\nexcept Exception as e:\n"
            "    print('RAISED', type(e).__name__)\n"
            "else:\n    print('NO-RAISE')\n" % PEN_B)
    r = _run(code, cwd, home)
    assert r.returncode == 0, r.stderr[-800:]
    assert "RAISED ShadowConflictError" in r.stdout, r.stdout + r.stderr[-400:]


def test_load_cal_cannot_swallow_the_guard():
    home, cwd = _dirs()
    (home / "calibration.json").write_text(json.dumps({"z_touch_abs": 12.05}))
    code = ("import json, pathlib\nfrom ccplotter.cal import load_cal\n"
            "assert load_cal()['z_touch_abs'] == 12.05\n"
            "pathlib.Path('calibration.json').write_text(json.dumps({'z_touch_abs': 99}))\n"
            "try:\n    print('GOT', load_cal())\nexcept Exception as e:\n    print('RAISED', type(e).__name__)\n")
    r = _run(code, cwd, home)
    assert "RAISED ShadowConflictError" in r.stdout, r.stdout + r.stderr[-400:]


# -- never seed into the CWD; reads come from the home ----------------------

def test_seed_goes_only_into_the_resolved_home_never_the_cwd():
    home, cwd = _dirs()                                   # both empty
    r = _run("from ccplotter.cal import load_pen_profiles\n"
             "d = load_pen_profiles(); print('ACTIVE', d['active'])", cwd, home)
    assert r.returncode == 0, r.stderr[-800:]
    assert (home / "pen_profiles.json").is_file(), "seed must land in the resolved home"
    assert not (cwd / "pen_profiles.json").exists(), "seed must NEVER be written into the CWD"
    assert not (cwd / "calibration.json").exists()


def test_reads_resolve_from_home_not_cwd_and_writes_land_in_home():
    home, cwd = _dirs()
    (home / "calibration.json").write_text(json.dumps({"z_touch_abs": 12.05}))
    r = _run("from ccplotter.cal import load_cal, save_cal\n"
             "print('Z', load_cal()['z_touch_abs'])\nsave_cal({'z_micro': 0.9})", cwd, home)
    assert r.returncode == 0 and "Z 12.05" in r.stdout, r.stderr[-800:]
    assert json.loads((home / "calibration.json").read_text())["z_micro"] == 0.9
    assert not (cwd / "calibration.json").exists()


def test_cwd_equal_to_home_is_the_desktop_case_and_never_conflicts():
    home, _ = _dirs()
    (home / "pen_profiles.json").write_text(json.dumps(PEN_A))
    (home / "calibration.json").write_text(json.dumps({"z_touch_abs": 12.05}))
    r = _run("from ccplotter.cal import load_pen_profiles, load_cal\n"
             "print('OK', load_pen_profiles()['profiles']['x']['z_touch_abs'], load_cal()['z_touch_abs'])",
             home, home)
    assert r.returncode == 0 and "OK 12.05 12.05" in r.stdout, r.stderr[-800:]
    assert "WARNING" not in r.stderr


# -- Store Python (AppContainer) trap ---------------------------------------

STORE_EXE = r"C:\Users\someone\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"
STORE_PREFIX = r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.2544.0_x64__qbz5n2kfra8p0"


def test_store_python_detection():
    assert rh.is_store_python(STORE_EXE, STORE_PREFIX)
    assert not rh.is_store_python(r"C:\msys64\ucrt64\bin\python.exe", r"C:\msys64\ucrt64")
    assert not rh.is_store_python(r"C:\Users\someone\AppData\Local\Programs\ccPlotter\ccPlotter.exe", "")


def test_store_python_refused_on_the_real_runtime_dir_but_not_elsewhere(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    real = tmp_path / "ccPlotter"
    with pytest.raises(rh.UntrustedInterpreterError):
        rh.check_interpreter(real, executable=STORE_EXE, prefix=STORE_PREFIX)
    with pytest.raises(rh.UntrustedInterpreterError):
        rh.check_interpreter(real / "sub", executable=STORE_EXE, prefix=STORE_PREFIX)
    # positive controls: same interpreter, copy outside the runtime dir -> allowed
    rh.check_interpreter(tmp_path / "copy_of_runtime", executable=STORE_EXE, prefix=STORE_PREFIX)
    # non-Store interpreter on the real dir -> allowed
    rh.check_interpreter(real, executable=r"C:\msys64\ucrt64\bin\python.exe", prefix=r"C:\msys64\ucrt64")


@pytest.mark.skipif(not rh.is_store_python(), reason="live Store-Python check needs the Store interpreter")
def test_live_store_python_refuses_default_home_at_config_import():
    cwd = Path(tempfile.mkdtemp(prefix="d661_cwd_")).resolve()
    r = _run(IMPORT_CFG, cwd, home=None)                  # no CCPLOTTER_HOME -> real runtime dir
    assert r.returncode != 0 and "UntrustedInterpreterError" in r.stderr, r.stderr[-800:]
    assert not (cwd / "work").exists(), "refusal must precede any directory creation"
