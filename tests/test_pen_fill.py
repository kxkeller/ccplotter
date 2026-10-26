"""Per-pen fill-size slider (ccplotter/pen_fill.py + /api/pen_profiles/fill).

The user can set a pen's fill/tip size on a slider, and the value persists from one app
instance to the next. The route edits ONE number (fill_mm) of ONE profile in
pen_profiles.json -- a file that also holds z_touch_abs (bed-crash class) -- so these tests
are about what it must NOT do as much as what it does.

Every test runs against a TEMP runtime home (tests/_test_home.py), never the real
%LOCALAPPDATA%\\ccPlotter; one test stats the real file before/after to prove it.

Could-not-fail discipline: each negative assertion is paired with a control that shows
the detector can fire (a deliberately wrong write, a boundary value that must pass, ...).
"""
import difflib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home; pen/calibration never resolve from CWD
_HOME = _test_home.use_temp_home("ccplotter_test_home_penfill_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_penfill_"))
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
from PIL import Image

import ccplotter.routes.core  # noqa: F401  (/api/process, registered on the shared app)
import ccplotter.routes.pen   # noqa: F401  (/api/pen_profiles/fill)
import ccplotter.routes.runs  # noqa: F401
from ccplotter import cal, config, pen_fill
from ccplotter import runtime_home as rh
from ccplotter.config import WORK_DIR
from ccplotter.routes.core import app
from ccplotter.vcache import _resolved_inputs_fingerprint, _variant_key

PATH = cal.PEN_PROFILES_PATH
assert PATH.parent == _HOME, "test is not isolated from the real runtime home"
BOM = b"\xef\xbb\xbf"


@pytest.fixture(autouse=True)
def fresh_profiles():
    """Reseed the temp home's pen_profiles.json from the built-in seed before every test."""
    for p in list(_HOME.glob("pen_profiles*")):
        p.unlink()
    cal.load_pen_profiles()
    assert PATH.exists()
    yield


def _raw():
    return PATH.read_bytes()


def _doc():
    return json.loads(_raw().decode("utf-8"))


def _baks():
    return sorted(f for f in _HOME.glob("pen_profiles.bak-fillslider-*.json")
                  if re.fullmatch(r"pen_profiles\.bak-fillslider-\d{8}-\d{6}(-\d+)?\.json", f.name))


def _post(body):
    return app.test_client().post("/api/pen_profiles/fill", json=body)


def _changed_lines(before: bytes, after: bytes):
    """Lines that differ between two byte strings (EOL-insensitive), as (old, new) pairs."""
    a = before.decode("utf-8").splitlines()
    b = after.decode("utf-8").splitlines()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append((a[i1:i2], b[j1:j2]))
    return out


def _z_lines(raw: bytes):
    return re.findall(rb'"z_touch_abs": [^,\r\n]+', raw)


# ── what it writes ───────────────────────────────────────────────────────────

def test_writes_only_the_target_pens_fill_mm():
    before = _raw()
    d0 = _doc()
    assert d0["profiles"]["bic_cristal_blue"]["fill_mm"] == 0.12
    res = pen_fill.set_fill_mm("bic_cristal_blue", 0.2)
    assert res["changed"] and res["fill_mm"] == 0.2
    after = _raw()
    # The FIRST edit changes fill_mm, and adds one new fill_mm_calibrated line right
    # after it (holding the pre-edit value). Nothing else moves.
    diff = _changed_lines(before, after)
    flat_new = [ln for _, new in diff for ln in new]
    flat_old = [ln for old, _ in diff for ln in old]
    assert any('"fill_mm": 0.12' in ln for ln in flat_old)
    assert any('"fill_mm": 0.2' in ln for ln in flat_new)
    assert any('"fill_mm_calibrated": 0.12' in ln for ln in flat_new)
    assert not any("fill_mm_calibrated" in ln for ln in flat_old)
    assert len(flat_new) - len(flat_old) == 1 and len(flat_new) <= 3, diff
    d1 = _doc()
    assert d1["profiles"]["bic_cristal_blue"]["fill_mm"] == 0.2
    assert d1["profiles"]["bic_cristal_blue"]["fill_mm_calibrated"] == 0.12
    for pid, p in d0["profiles"].items():          # every OTHER pen and field identical
        for k, v in p.items():
            if (pid, k) != ("bic_cristal_blue", "fill_mm"):
                assert d1["profiles"][pid][k] == v, (pid, k)
    assert list(d1["profiles"]) == list(d0["profiles"])                       # key order
    for pid in d0["profiles"]:                     # other pens gained NO key at all
        if pid != "bic_cristal_blue":
            assert list(d1["profiles"][pid]) == list(d0["profiles"][pid]), pid
    keys1 = list(d1["profiles"]["bic_cristal_blue"])
    assert [k for k in keys1 if k != "fill_mm_calibrated"] == list(d0["profiles"]["bic_cristal_blue"])
    assert keys1[keys1.index("fill_mm") + 1] == "fill_mm_calibrated"
    assert d1["active"] == d0["active"]
    # CONTROL: the detector fires on a write that also touched z_touch_abs.
    tampered = after.replace(b'"z_touch_abs": 12.05', b'"z_touch_abs": 12.06')
    assert tampered != after and len(_changed_lines(before, tampered)) == len(diff) + 1


def test_z_touch_and_every_other_byte_stay_identical():
    before = _raw()
    zs = _z_lines(before)
    assert len(zs) >= 4 and all(zs)                     # the control has something to lose
    pen_fill.set_fill_mm("energel_05", 0.31)
    pen_fill.set_fill_mm("platinum_preppy", 0.3)
    after = _raw()
    assert _z_lines(after) == zs
    # Everything except the two fill_mm numbers is byte-identical, EOL style included.
    # (the one-time fill_mm_calibrated line is the only other permitted addition)
    norm = lambda b: re.sub(rb'\r?\n[ ]*"fill_mm_calibrated": [0-9.]+,?', b"",
                            re.sub(rb'("fill_mm": )[0-9.]+', rb"\1X", b))
    assert norm(after) == norm(before)
    assert (b"\r\n" in before) == (b"\r\n" in after)
    assert before.endswith(b"}") == after.endswith(b"}")
    # CONTROL: the normaliser is not blind -- changing a z value breaks equality.
    assert norm(after.replace(b'"z_touch_abs": 18.75', b'"z_touch_abs": 18.76', 1)) != norm(before)


def test_no_bom_written_even_when_the_file_had_one():
    PATH.write_text(PATH.read_text(encoding="utf-8"), encoding="utf-8-sig")
    assert _raw().startswith(BOM)                       # CONTROL: the detector can fire
    pen_fill.set_fill_mm("energel_05", 0.3)
    assert not _raw().startswith(BOM)
    assert cal.load_pen_profiles()["profiles"]["energel_05"]["fill_mm"] == 0.3
    # and the normal path writes none either
    pen_fill.set_fill_mm("energel_05", 0.28)
    assert not _raw().startswith(BOM)
    # backups are the exact OLD bytes: only the first one (the BOM'd original) has a BOM
    assert sorted(b.read_bytes().startswith(BOM) for b in _baks()) == [False, True]


def test_out_of_range_and_malformed_values_are_rejected_untouched():
    before = _raw()
    bad = [0.049, 0.0499, 0.601, 0.61, 1.0, 0, -0.1, 99, math.nan, math.inf, -math.inf,
           "0.2", "", None, True, False, [0.2], {"v": 0.2}]
    for v in bad:
        with pytest.raises(pen_fill.PenFillError):
            pen_fill.set_fill_mm("energel_05", v)
        assert _raw() == before, v
    assert not _baks()
    # CONTROL: both boundaries ARE accepted (so the rejections are the range, not a blanket no).
    assert pen_fill.set_fill_mm("energel_05", 0.05)["fill_mm"] == 0.05
    assert pen_fill.set_fill_mm("energel_05", 0.60)["fill_mm"] == 0.6
    # HTTP surface: 400 + file unchanged
    before = _raw()
    for v in (0.04, 0.7, "0.2", None):
        r = _post({"id": "energel_05", "fill_mm": v})
        assert r.status_code == 400, (v, r.get_json())
        assert _raw() == before


def test_unknown_pen_is_404_and_writes_nothing():
    before = _raw()
    r = _post({"id": "no_such_pen", "fill_mm": 0.2})
    assert r.status_code == 404 and _raw() == before and not _baks()


def test_route_refuses_any_field_but_fill_mm_and_reset():
    before = _raw()
    for extra in ("z_touch_abs", "z_down", "pen_mm", "label", "active", "pen_offset_y"):
        r = _post({"id": "energel_05", "fill_mm": 0.2, extra: 1.0})
        assert r.status_code == 400, extra
        assert _raw() == before, extra
    assert _post({"id": "energel_05"}).status_code == 400                       # nothing to do
    assert _post({"fill_mm": 0.2}).status_code == 400                           # no id
    assert _post({"id": "energel_05", "fill_mm": 0.2, "reset": True}).status_code == 400
    assert _raw() == before
    # CONTROL: the same route with only the allowed fields works.
    r = _post({"id": "energel_05", "fill_mm": 0.2})
    assert r.status_code == 200 and r.get_json()["fill_mm"] == 0.2


def test_value_is_rounded_to_a_thousandth():
    assert pen_fill.set_fill_mm("energel_05", 0.1234)["fill_mm"] == 0.123
    assert _doc()["profiles"]["energel_05"]["fill_mm"] == 0.123
    assert pen_fill.set_fill_mm("energel_05", 0.1 + 0.2)["fill_mm"] == 0.3        # no 0.30000000000000004


# ── calibrated value / reset ─────────────────────────────────────────────────

def _prof(pid):
    return _doc()["profiles"][pid]


def test_first_edit_creates_calibrated_key_second_edit_leaves_it_unchanged():
    assert "fill_mm_calibrated" not in _prof("energel_05")              # CONTROL: absent before
    r1 = pen_fill.set_fill_mm("energel_05", 0.3)
    assert r1["calibrated_key_written"] is True and r1["fill_mm_calibrated"] == 0.25
    assert _prof("energel_05")["fill_mm_calibrated"] == 0.25 and _prof("energel_05")["fill_mm"] == 0.3
    r2 = pen_fill.set_fill_mm("energel_05", 0.4)
    assert r2["calibrated_key_written"] is False
    assert _prof("energel_05")["fill_mm_calibrated"] == 0.25            # NOT the intermediate 0.3
    r3 = pen_fill.set_fill_mm("energel_05", 0.28)
    assert r3["calibrated_key_written"] is False and _prof("energel_05")["fill_mm_calibrated"] == 0.25
    # the second edit's diff is ONE line (the key is not rewritten, only fill_mm moves)
    b = _raw()
    pen_fill.set_fill_mm("energel_05", 0.29)
    d = _changed_lines(b, _raw())
    assert len(d) == 1 and len(d[0][0]) == 1 and len(d[0][1]) == 1, d
    # a no-op / rejected edit never creates the key
    pen_fill.set_fill_mm("platinum_preppy", 0.275)                       # unchanged value
    with pytest.raises(pen_fill.PenFillError):
        pen_fill.set_fill_mm("platinum_preppy", 5.0)
    assert "fill_mm_calibrated" not in _prof("platinum_preppy")
    # the key is NOT created for other pens by editing this one
    assert "fill_mm_calibrated" not in _prof("bic_cristal_blue")


def test_reset_restores_the_calibrated_value_and_keeps_the_key():
    pen_fill.set_fill_mm("bic_cristal_blue", 0.2)
    assert _prof("bic_cristal_blue")["fill_mm"] == 0.2                  # CONTROL: it moved
    r = _post({"id": "bic_cristal_blue", "reset": True})
    assert r.status_code == 200
    j = r.get_json()
    assert j["fill_mm"] == 0.12 and j["edited"] is False and j["before"] == 0.2
    assert _prof("bic_cristal_blue")["fill_mm"] == 0.12
    assert _prof("bic_cristal_blue")["fill_mm_calibrated"] == 0.12      # never removed / rewritten
    assert config._PEN_PROFILE_SEED["profiles"]["bic_cristal_blue"]["fill_mm"] == 0.12   # seed untouched


def test_reset_uses_calibrated_not_the_seed_when_they_differ():
    """A hand-measured value that is NOT the seed (e.g. EnerGel measured 0.3, seed 0.25) must
    come back on Reset; the seed is only a fallback."""
    d = _doc()
    d["profiles"]["energel_05"]["fill_mm"] = 0.3
    PATH.write_bytes(json.dumps(d, indent=2).replace("\n", "\r\n").encode())
    assert pen_fill.seed_fill_mm("energel_05") == 0.25 != 0.3            # CONTROL: they differ
    pen_fill.set_fill_mm("energel_05", 0.5)
    assert _prof("energel_05")["fill_mm_calibrated"] == 0.3
    assert pen_fill.set_fill_mm("energel_05", reset=True)["fill_mm"] == 0.3
    assert _prof("energel_05")["fill_mm"] == 0.3


def test_reset_works_on_a_seedless_pen():
    r = app.test_client().post("/api/pen_profiles/save",
                               json={"label": "Fill Test Pen", "id": "fill_test_pen",
                                     "pen_mm": 0.4, "fill_mm": 0.31})
    assert r.status_code == 200
    e = app.test_client().get("/api/pen_profiles/fill").get_json()["pens"]["fill_test_pen"]
    assert e["seed_fill_mm"] is None and e["fill_mm"] == 0.31 == e["calibrated_fill_mm"]
    assert e["edited"] is False                                          # never edited
    assert _post({"id": "fill_test_pen", "fill_mm": 0.45}).status_code == 200
    e = app.test_client().get("/api/pen_profiles/fill").get_json()["pens"]["fill_test_pen"]
    assert e["fill_mm"] == 0.45 and e["calibrated_fill_mm"] == 0.31 and e["edited"] is True
    r = _post({"id": "fill_test_pen", "reset": True})
    assert r.status_code == 200 and r.get_json()["fill_mm"] == 0.31      # restored, no seed involved
    assert _prof("fill_test_pen")["fill_mm"] == 0.31
    e = app.test_client().get("/api/pen_profiles/fill").get_json()["pens"]["fill_test_pen"]
    assert e["edited"] is False and e["calibrated_fill_mm"] == 0.31


def test_reset_on_a_never_edited_pen_is_a_noop_not_a_seed_overwrite():
    d = _doc()
    d["profiles"]["energel_05"]["fill_mm"] = 0.3                         # measured, != seed 0.25
    PATH.write_bytes(json.dumps(d, indent=2).replace("\n", "\r\n").encode())
    before = _raw()
    r = _post({"id": "energel_05", "reset": True})
    assert r.status_code == 200 and r.get_json()["changed"] is False
    assert _raw() == before and not _baks()


def test_info_marks_edited_as_differs_from_calibrated_and_laid_mm_pens():
    pen_fill.set_fill_mm("platinum_preppy", 0.3)
    d = _doc()
    d["profiles"]["bic_cristal_black"]["laid_mm"] = 0.255      # a measured-laid-width pen
    PATH.write_bytes(json.dumps(d, indent=2).replace("\n", "\r\n").encode())
    pens = app.test_client().get("/api/pen_profiles/fill").get_json()["pens"]
    assert pens["platinum_preppy"]["edited"] is True and pens["platinum_preppy"]["calibrated_fill_mm"] == 0.275
    assert pens["energel_05"]["edited"] is False and pens["energel_05"]["calibrated_fill_mm"] == 0.25
    # The per-pen `fill_drives_pitch` prediction no longer exists (it mispredicted the real spacing);
    # the UI reads the REAL spacing back per job as `realised_pitch`. laid_mm is still reported.
    assert all("fill_drives_pitch" not in p for p in pens.values())
    assert pens["bic_cristal_black"]["laid_mm"] == 0.255
    assert pens["energel_05"]["laid_mm"] is None
    # editing BACK to the calibrated number is "not edited" (differs-from-calibrated semantics)
    pen_fill.set_fill_mm("platinum_preppy", 0.275)
    assert app.test_client().get("/api/pen_profiles/fill").get_json()["pens"]["platinum_preppy"]["edited"] is False


# ── persistence ──────────────────────────────────────────────────────────────

def test_value_persists_across_a_reload_of_the_config_module():
    import importlib
    importlib.reload(config)
    importlib.reload(cal)
    importlib.reload(pen_fill)
    assert cal.load_pen_profiles()["profiles"]["bic_cristal_blue"]["fill_mm"] == 0.12   # CONTROL (pre-edit)
    pen_fill.set_fill_mm("bic_cristal_blue", 0.175)
    importlib.reload(config)
    importlib.reload(cal)
    importlib.reload(pen_fill)
    assert cal.load_pen_profiles()["profiles"]["bic_cristal_blue"]["fill_mm"] == 0.175
    assert pen_fill.fill_info()["pens"]["bic_cristal_blue"]["edited"] is True


def test_value_persists_into_a_fresh_interpreter():
    """The real 'app restart': a brand-new process (empty CWD, same explicit home) reads it."""
    pen_fill.set_fill_mm("platinum_preppy", 0.31)
    scratch = tempfile.mkdtemp(prefix="ccplotter_penfill_cwd_")
    code = ("import sys, json; sys.path.insert(0, %r)\n"
            "from ccplotter import cal, pen_fill\n"
            "print(json.dumps([cal.load_pen_profiles()['profiles']['platinum_preppy']['fill_mm'],"
            " pen_fill.fill_info()['pens']['platinum_preppy']['edited']]))\n") % str(REPO)
    env = dict(os.environ, CCPLOTTER_HOME=str(_HOME),
               CCPLOTTER_WORK_DIR=tempfile.mkdtemp(prefix="ccplotter_penfill_w_"))
    out = subprocess.run([sys.executable, "-c", code], cwd=scratch, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [0.31, True]


# ── backups ──────────────────────────────────────────────────────────────────

def test_every_write_keeps_a_timestamped_bak_of_the_exact_old_bytes():
    b0 = _raw()
    r1 = pen_fill.set_fill_mm("energel_05", 0.3)
    assert len(_baks()) == 1 and _baks()[0].read_bytes() == b0
    assert re.fullmatch(r"pen_profiles\.bak-fillslider-\d{8}-\d{6}(-\d+)?\.json", r1["backup"])
    b1 = _raw()
    r2 = pen_fill.set_fill_mm("energel_05", 0.29)            # same second -> must not clobber
    assert len(_baks()) == 2 and any(b.read_bytes() == b1 for b in _baks())
    assert r1["backup"] != r2["backup"]
    # an unchanged value writes nothing and keeps no new bak (control: the count above moved)
    r3 = pen_fill.set_fill_mm("energel_05", 0.29)
    assert r3["changed"] is False and r3["backup"] is None and len(_baks()) == 2


# ── atomicity ────────────────────────────────────────────────────────────────

def test_a_failed_replace_leaves_the_original_intact_and_no_temp_litter(monkeypatch):
    before = _raw()
    import ccplotter.atomic_io as aio

    def boom(src, dst):
        raise OSError("simulated crash at the moment of replace")
    monkeypatch.setattr(aio.os, "replace", boom)
    with pytest.raises(OSError):
        pen_fill.set_fill_mm("energel_05", 0.2)
    monkeypatch.undo()
    assert _raw() == before and json.loads(before.decode())          # intact AND parseable
    assert not list(_HOME.glob(".pen_profiles.atomictmp-*"))        # temp cleaned up
    # CONTROL: with the failure removed the very same call succeeds.
    assert pen_fill.set_fill_mm("energel_05", 0.2)["changed"]


def test_a_partial_temp_write_never_reaches_the_target(monkeypatch):
    before = _raw()
    real_write_bytes = Path.write_bytes

    def half_then_die(self, data):
        if ".atomictmp-" in self.name and self.name.startswith(".pen_profiles"):
            real_write_bytes(self, data[: len(data) // 2])
            raise OSError("disk full mid-write")
        return real_write_bytes(self, data)
    monkeypatch.setattr(Path, "write_bytes", half_then_die)
    with pytest.raises(OSError):
        pen_fill.set_fill_mm("energel_05", 0.2)
    monkeypatch.undo()
    assert _raw() == before and not list(_HOME.glob(".pen_profiles.atomictmp-*"))


def test_unparseable_file_is_refused_not_overwritten():
    PATH.write_bytes(b'{"active": "x", "profiles": {')            # truncated JSON
    junk = _raw()
    with pytest.raises(pen_fill.PenFillError):
        pen_fill.set_fill_mm("energel_05", 0.2)
    assert _raw() == junk and not _baks()


def test_concurrent_writers_and_readers_never_see_a_corrupt_file():
    pens = ["energel_05", "platinum_preppy", "bic_cristal_black", "bic_cristal_blue"]
    stop = threading.Event()
    problems, reads = [], [0]

    def reader():
        while not stop.is_set():
            try:
                raw = PATH.read_bytes()
            except PermissionError:
                continue                    # Windows: replace in flight; not a corrupt read
            finally:
                time.sleep(0.003)           # a realistic reader, not a spin that starves os.replace
            try:
                d = json.loads(raw.decode("utf-8"))
                assert set(d["profiles"]) >= set(pens) and not raw.startswith(BOM)
                reads[0] += 1
            except Exception as e:           # a torn / partial file would land here
                problems.append(repr(e))

    last = {}

    def writer(pid, base):
        try:
            for i in range(20):
                v = round(base + 0.005 * i, 3)
                pen_fill.set_fill_mm(pid, v)
                last[pid] = v
        except Exception as e:
            problems.append(f"writer {pid}: {e!r}")

    readers = [threading.Thread(target=reader) for _ in range(3)]
    writers = [threading.Thread(target=writer, args=(pid, 0.10 + 0.05 * n)) for n, pid in enumerate(pens)]
    # a second writer on the SAME pen: last-writer-wins, but never a lost OTHER-pen edit
    writers.append(threading.Thread(target=writer, args=("bic_cristal_blue", 0.4)))
    for t in readers + writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    for t in readers:
        t.join()
    assert not problems, problems[:5]
    assert reads[0] > 0                      # CONTROL: the readers really did read
    d = _doc()
    for pid in pens[:3]:                     # pens with a single writer: their last value won
        assert d["profiles"][pid]["fill_mm"] == last[pid], pid
    # two writers share bic_cristal_blue: last writer wins, one of the two final values
    assert d["profiles"]["bic_cristal_blue"]["fill_mm"] in (0.345, 0.495)
    assert len(_z_lines(_raw())) >= 4


# ── guards ───────────────────────────────────────────────────────────────────

def test_d661_cwd_shadow_guard_refuses_the_write():
    before = _raw()
    shadow_dir = Path(tempfile.mkdtemp(prefix="ccplotter_penfill_shadow_"))
    (shadow_dir / "pen_profiles.json").write_text(json.dumps({"active": "x", "profiles": {"x": {}}}))
    here = os.getcwd()
    try:
        os.chdir(shadow_dir)
        with pytest.raises(rh.ShadowConflictError):
            pen_fill.set_fill_mm("energel_05", 0.2)
        r = _post({"id": "energel_05", "fill_mm": 0.2})
        assert r.status_code == 500 and "shadow" in r.get_json()["error"].lower()
    finally:
        os.chdir(here)
    assert _raw() == before and not _baks()
    # CONTROL: from a directory with no shadow the identical call goes through.
    assert pen_fill.set_fill_mm("energel_05", 0.2)["changed"]


def test_the_real_runtime_home_is_never_touched():
    real = rh.default_home() / "pen_profiles.json"
    snap = (lambda: (real.stat().st_size, real.stat().st_mtime_ns) if real.exists() else None)
    s0 = snap()
    pen_fill.set_fill_mm("energel_05", 0.2)
    pen_fill.set_fill_mm("energel_05", None, reset=True)
    assert snap() == s0
    assert PATH.parent == _HOME and _HOME != rh.default_home()


# ── generation: the cache and the provenance ─────────────────────────────────

def test_vcache_key_changes_when_fill_mm_changes():
    body = {"style": "tracev3", "resolution": 600, "job_type": "gcode", "items": [{"id": "k"}]}
    calj, reach = cal.load_cal(), {"id": "r"}

    def key():
        pid = "bic_cristal_blue"
        return _variant_key(body, _resolved_inputs_fingerprint(pid, cal.resolve_pen_profile(pid), calj, reach))
    k0 = key()
    assert key() == k0                                         # CONTROL: stable when nothing moves
    pen_fill.set_fill_mm("bic_cristal_blue", 0.2)
    k1 = key()
    assert k1 != k0
    pen_fill.set_fill_mm("bic_cristal_blue", None, reset=True)
    # After a reset the profile keeps its (one-time) fill_mm_calibrated key, so the resolved
    # profile -- hence the key -- is NOT byte-equal to the never-edited one. That costs one
    # regenerate, never a stale hit; it is stable from then on.
    k2 = key()
    assert k2 not in (k0, k1) and key() == k2
    assert cal.resolve_pen_profile("bic_cristal_blue")["fill_mm"] == 0.12


def _make_src(did, w=700, h=500):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    g = (127 + 100 * np.sin(xx / 13.0) * np.cos(yy / 19.0)).clip(0, 255).astype(np.uint8)
    Image.fromarray(np.stack([g, g, g], -1), mode="RGB").save(WORK_DIR / f"src_{did}.jpg", quality=92)


def test_end_to_end_next_generate_misses_the_cache_and_provenance_says_edited():
    did = "penfill_src"
    _make_src(did)
    body = {"job_type": "gcode", "style": "tracev3", "paper": "120x120", "margin": 10.0,
            "resolution": 600, "pen_id": "energel_05",
            "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}]}

    def run():
        r = app.test_client().post("/api/process", json=body)
        assert r.status_code == 200, r.get_json()
        return r.get_json()
    from ccplotter.runs import _runs_load, load_run_provenance
    j1 = run()
    assert not j1.get("cached")
    j2 = run()
    assert j2.get("cached") is True                            # CONTROL: the cache works at all
    pen_fill.set_fill_mm("energel_05", 0.32)
    j3 = run()
    assert not j3.get("cached")                                # the edit forced a real regenerate
    provs = [load_run_provenance(r["id"]) for r in _runs_load() if r.get("type") == "gcode"]
    provs = [p for p in provs if p]
    edited = [p for p in provs if p["pen_fill"].get("edited_from_default") is True]
    default = [p for p in provs if p["pen_fill"].get("edited_from_default") is False]
    assert edited and default, [p["pen_fill"] for p in provs]   # the first run was unedited, the third edited
    e = edited[0]
    assert e["pen_profile"]["fill_mm"] == 0.32
    assert e["pen_fill"] == {"fill_mm": 0.32, "fill_mm_calibrated": 0.25, "seed_fill_mm": 0.25,
                             "edited_from_default": True,
                             "note": "OVERRIDE: fill_mm 0.32 (calibrated 0.25)"}
    e["pen_fill"]["note"].encode("ascii")
    # CONTROL for the marker: the unedited job says calibrated and carries the same value
    d0 = default[0]
    assert d0["pen_fill"]["fill_mm"] == 0.25 == d0["pen_profile"]["fill_mm"] == d0["pen_fill"]["fill_mm_calibrated"]
    assert d0["pen_fill"]["note"] == "calibrated fill_mm 0.25"
    # reset -> back to the calibrated pitch; (one regenerate because the profile now carries
    # the calibrated key, then cached again)
    pen_fill.set_fill_mm("energel_05", None, reset=True)
    assert not run().get("cached")
    assert run().get("cached") is True


def test_fill_provenance_helper_cases():
    fp = pen_fill.fill_provenance
    assert fp("bic_cristal_blue", {"fill_mm": 0.12})["edited_from_default"] is False
    e = fp("bic_cristal_blue", {"fill_mm": 0.1, "fill_mm_calibrated": 0.12})
    assert e["edited_from_default"] is True and e["fill_mm_calibrated"] == 0.12
    assert e["note"] == "OVERRIDE: fill_mm 0.1 (calibrated 0.12)"
    # calibrated comes from the profile, not the seed: a seedless pen is judged the same way
    n = fp("some_other_pen", {"fill_mm": 0.3, "fill_mm_calibrated": 0.4})
    assert n["seed_fill_mm"] is None and n["edited_from_default"] is True and n["fill_mm_calibrated"] == 0.4
    assert fp("some_other_pen", {"fill_mm": 0.3})["edited_from_default"] is False
    assert "error" in fp("x", object())  or fp("x", None)["fill_mm"] is None   # never raises
    for v in (e, n):
        v["note"].encode("ascii")


# ── one lock shared with the z writer ────────────────────────────────────────

def test_save_pen_profiles_and_set_fill_mm_share_one_lock():
    assert cal.PEN_PROFILES_LOCK is not None
    done = {}

    def do_fill():
        pen_fill.set_fill_mm("energel_05", 0.3)
        done["fill"] = time.monotonic()

    def do_save():
        d = cal.load_pen_profiles()
        cal.save_pen_profiles(d)
        done["save"] = time.monotonic()

    before = _raw()
    t_fill, t_save = threading.Thread(target=do_fill), threading.Thread(target=do_save)
    with cal.PEN_PROFILES_LOCK:                 # hold the module lock from this thread
        t_fill.start()
        t_save.start()
        time.sleep(0.4)
        assert not done, "a writer got past the held lock"          # BOTH are blocked on it
        assert _raw() == before
        released = time.monotonic()
    t_fill.join(10)
    t_save.join(10)
    assert set(done) == {"fill", "save"} and min(done.values()) >= released
    # CONTROL: with the lock free the same two calls complete immediately (no deadlock)
    done.clear()
    t1, t2 = threading.Thread(target=do_fill), threading.Thread(target=do_save)
    t1.start(); t2.start(); t1.join(10); t2.join(10)
    assert set(done) == {"fill", "save"}


def test_save_pen_profiles_writes_utf8_without_bom():
    d = cal.load_pen_profiles()
    d["profiles"]["energel_05"]["label"] = "Pen \u00e9"      # non-ASCII must still round-trip
    cal.save_pen_profiles(d)
    raw = _raw()
    assert not raw.startswith(BOM)
    assert json.loads(raw.decode("utf-8"))["profiles"]["energel_05"]["label"] == "Pen \u00e9"
    PATH.write_text(PATH.read_text(encoding="utf-8"), encoding="utf-8-sig")
    assert _raw().startswith(BOM)                                   # CONTROL: the detector fires
    cal.save_pen_profiles(cal.load_pen_profiles())
    assert not _raw().startswith(BOM)


# ── backup pruning ───────────────────────────────────────────────────────────

def test_backups_are_pruned_to_the_newest_20_and_foreign_files_survive():
    foreign = [_HOME / "pen_profiles.bak-before-x-20260801.json",
               _HOME / "pen_profiles.bak-fillslider-keepme.json",        # not a slider-stamped name
               _HOME / "pen_profiles.json.bak-2026-09-21",
               _HOME / "calibration.bak-fillslider-20200101-000000.json"]  # another file's backup
    for f in foreign:
        f.write_text("foreign")
    pen_fill.set_fill_mm("energel_05", 0.30)
    oldest = _baks()[0].name
    results = []
    for i in range(1, 20):                                          # writes 2..20 -> 20 backups
        results.append(pen_fill.set_fill_mm("energel_05", round(0.30 + 0.005 * i, 3)))
    assert len(_baks()) == 20 and not any(r["pruned"] for r in results)    # nothing pruned yet
    names20 = [b.name for b in _baks()]
    assert oldest in names20
    r21 = pen_fill.set_fill_mm("energel_05", 0.45)                  # the 21st write
    assert r21["pruned"] == [oldest]
    now = [b.name for b in _baks()]
    assert len(now) == 20 and oldest not in now and r21["backup"] in now
    assert set(names20) - set(now) == {oldest}                      # exactly the oldest went
    for f in foreign:                                               # nothing else was touched
        assert f.exists() and f.read_text() == "foreign", f.name
    for n in range(5):                                              # keeps holding at 20
        pen_fill.set_fill_mm("energel_05", round(0.2 + 0.01 * n, 3))
    assert len(_baks()) == 20 and all(f.exists() for f in foreign)
    # the newest backup is the exact bytes before the last write (pruning removed old ones only)
    b = _raw()
    last = pen_fill.set_fill_mm("energel_05", 0.55)
    assert (_HOME / last["backup"]).read_bytes() == b


def test_backup_naming_within_one_second_never_reuses_a_pruned_low_suffix(monkeypatch):
    # Regression (found by the final-tree rerun, flaky by the clock): with several edits in
    # one second, a prune frees suffix -0; reusing that "first free name" made the NEW backup
    # sort as the oldest and be pruned immediately, so the result named a file that was gone.
    import types
    stamp = "20261005-120000"
    monkeypatch.setattr(pen_fill, "time",
                        types.SimpleNamespace(strftime=lambda fmt: stamp, sleep=time.sleep))
    names = []
    for i in range(24):                                   # 24 edits inside the frozen second
        r = pen_fill.set_fill_mm("energel_05", round(0.30 + 0.005 * i, 3))
        assert (_HOME / r["backup"]).exists(), (i, r["backup"])        # never a ghost
        names.append(r["backup"])
    assert len(set(names)) == 24                          # no name reused
    kept = [b.name for b in _baks()]
    assert len(kept) == 20
    assert set(kept) == set(names[-20:])                  # exactly the newest 20 survive
    # CONTROL: the sort key really orders -N numerically (10 > 9), which the proof relies on
    assert pen_fill._bak_sort_key("pen_profiles.bak-fillslider-20261005-120000-10.json") > \
        pen_fill._bak_sort_key("pen_profiles.bak-fillslider-20261005-120000-9.json")
