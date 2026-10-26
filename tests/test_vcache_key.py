"""Variant-cache key regression guard.

The variant key must change when the server-resolved pen profile / calibration /
reach changes (else a regenerate can return gcode at an old Z), and must keep the
legacy body-only key when no resolved fingerprint is given. Pure functions only.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root (was implicit via the CWD)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # ccplotter.config resolves pen/calibration from an explicit runtime home
_test_home.use_temp_home("ccplotter_test_home_vcache_")

from ccplotter.vcache import _resolved_inputs_fingerprint, _variant_key

BODY = {"style": "lineart", "resolution": 1024, "pen_id": "platinum_preppy",
        "job_type": "gcode", "items": [{"id": 1, "rot": 0}]}
PEN = {"z_touch_abs": 18.95, "fill_mm": 0.275, "pen_mm": 0.25, "draw_speed": 3000}
CAL = {"x_offset": -46.3, "y_offset": 0.0, "reach_profile": "plotter_2003_v32"}
REACH = {"name": "plotter_2003_v32", "x": [3.0, 255.0], "y": [42.5, 254.5]}


def test_legacy_key_unchanged_without_resolved_fp():
    legacy = hashlib.md5(json.dumps(BODY, sort_keys=True, default=str).encode()).hexdigest()[:12]
    assert _variant_key(BODY) == legacy
    assert _variant_key(BODY, None) == legacy


def test_fingerprint_moves_when_z_touch_moves():
    fp0 = _resolved_inputs_fingerprint("platinum_preppy", PEN, CAL, REACH)
    moved = dict(PEN, z_touch_abs=PEN["z_touch_abs"] + 1.0)
    fp1 = _resolved_inputs_fingerprint("platinum_preppy", moved, CAL, REACH)
    assert fp0 != fp1


def test_fingerprint_invariant_to_dict_key_order():
    reordered = dict(reversed(list(PEN.items())))
    assert list(reordered) != list(PEN)
    assert (_resolved_inputs_fingerprint("platinum_preppy", PEN, CAL, REACH)
            == _resolved_inputs_fingerprint("platinum_preppy", reordered, CAL, REACH))


def test_key_differs_across_fingerprints_on_same_body():
    fp0 = _resolved_inputs_fingerprint("platinum_preppy", PEN, CAL, REACH)
    fp1 = _resolved_inputs_fingerprint("platinum_preppy", dict(PEN, z_touch_abs=19.95), CAL, REACH)
    assert _variant_key(BODY, fp0) != _variant_key(BODY, fp1)
    assert _variant_key(BODY, fp0) == _variant_key(BODY, fp0)
    assert _variant_key(BODY, fp0) != _variant_key(BODY)
