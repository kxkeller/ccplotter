"""Test helper: give a test process an EXPLICIT, isolated runtime home.

pen_profiles.json / calibration.json resolve from the runtime home (ccplotter/
runtime_home.py), never the CWD. A test that os.chdir()s into the repo therefore
must NOT rely on CWD for them: it calls use_temp_home() BEFORE importing anything
from ccplotter. The temp home is seeded with the tracked calibration seed
(assets/seed/calibration.seed.json, bytes identical to the repo-root
calibration.json the tests read) and NO pen_profiles.json, so
load_pen_profiles() seeds the built-in profiles into the TEMP dir -- never the
repo root, never the live %LOCALAPPDATA%\ccPlotter.
"""
import os
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CAL_SEED = REPO / "assets" / "seed" / "calibration.seed.json"


def use_temp_home(prefix="ccplotter_test_home_", seed_calibration=True) -> Path:
    home = Path(tempfile.mkdtemp(prefix=prefix)).resolve()
    if seed_calibration:
        shutil.copyfile(CAL_SEED, home / "calibration.json")
    os.environ["CCPLOTTER_HOME"] = str(home)   # assignment, never setdefault: no ambient inheritance
    return home
