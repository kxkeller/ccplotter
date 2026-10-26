"""ccplotter/pen_fill.py -- the per-pen PASS PITCH OVERRIDE slider's storage layer.

Purpose: bump a pen's fill/tip size on a slider, with the value persisted from one
app instance to the next.

WHAT IT EDITS. Exactly one field of one profile in pen_profiles.json: `fill_mm`, the
pass PITCH that drives coverage (trace pocket spacing, the Trace V3+ pitch clamp, the
keep-out radius, the preview stroke width, the vcache fingerprint). It deliberately does
NOT edit `pen_mm`: the stored pen_mm only seeds the "Pen Tip Width" dropdown (a per-job
`pen_width` request field read by the legacy modes and the calibration labels); the
tracev3 pass pitch never reads it. A second slider for it would move a label, not a
plot. The slider sets a TARGET pass spacing, not a guaranteed one. In Trace (V3)
and B/W photo, a pen with a measured `laid_mm` takes its target from laid_mm and ignores
fill_mm (composite.py); in Trace V3+ fill_mm only moves the spacing when it is set below
the pitch the placement would otherwise lay. Separately, for ANY pen the 1.2 px
representability floor and the resolution cap can pin the real spacing, so the code does not
try to predict, per pen, whether the slider will move it (the earlier `fill_drives_pitch`
flag did, and was wrong when laid_mm was not the binding term). Instead every
tracev3-engine job reports `realised_pitch` (composite._realised_pitch_record: the fill px the
trace ran at x the FINAL placed mm/px) and the UI shows that after Generate. fill_info()
still reports `laid_mm` so the UI can word its note.

THE CALIBRATED VALUE IS NEVER DESTROYED. On the FIRST edit of a pen, the pre-edit
fill_mm is written ONCE into the same profile as `fill_mm_calibrated` (inserted right after
fill_mm). It is never rewritten afterwards. Reset restores fill_mm from it, for every pen
(seeded or not); the built-in seed in config._PEN_PROFILE_SEED (there is no pen seed file
under assets/seed/) is only a fallback for a profile that has no fill_mm at all. "Edited"
means "differs from calibrated". A pen that has never been edited has no
`fill_mm_calibrated` key; its current fill_mm IS its calibrated value, so Reset on it is a
no-op rather than an overwrite from the seed (the seed may not equal a hand-measured value).

WHAT IT MUST NEVER DO. Touch z_touch_abs, offsets, speeds, or any other field; write the
live file non-atomically; write a BOM; reorder keys. A bed-crash-class value lives in the
same file, so every write here is: read raw bytes -> parse -> change fill_mm (and, once,
add the absent fill_mm_calibrated) -> verify those are the ONLY differences -> timestamped
.bak of the old bytes -> temp file + os.replace -> read back and confirm -> prune the
slider's own backups to the newest KEEP_BACKUPS.

The file's own formatting is preserved: every writer in this codebase produces
json.dumps(indent=2) with CRLF line endings on Windows and no trailing newline; this
module detects EOL / trailing newline / non-ASCII from the existing bytes and reproduces
them, so a one-number edit is a one-line diff (two lines on the very first edit).
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

from ccplotter import cal as _cal
from ccplotter.atomic_io import atomic_write_bytes
from ccplotter.config import _PEN_PROFILE_SEED

# Bounds. 0.05 mm is below every pen this app has ever calibrated (finest: the uRight
# 5x-overlap test profile at laid 0.051 mm; typical working pens are 0.12-0.40). 0.60 mm is
# above the coarsest stored pitch (Preppy 0.5 at 0.40) with room to spare. Below ~0.10 mm
# the trace working grid saturates SCALE_AWARE_V2_NATIVE_CAP and the existing C3 warning
# fires (loud, not silent); above 0.60 mm a pass is wider than any pen here lays, i.e.
# guaranteed cracking. Step 0.005 matches the resolution of the usual pitch values
# (0.12, 0.25, 0.275, 0.4).
FILL_MM_MIN = 0.05
FILL_MM_MAX = 0.60
FILL_MM_STEP = 0.005
_ROUND_DP = 3          # stored precision (0.001 mm); the UI step is coarser

CAL_KEY = "fill_mm_calibrated"
KEEP_BACKUPS = 20      # newest N pen_profiles.bak-fillslider-*.json kept
# The read-modify-write is serialised by cal.PEN_PROFILES_LOCK -- the SAME lock
# cal.save_pen_profiles (the z / save / active / delete writer) takes. It is looked up on
# the module at call time so a reload of cal cannot leave the two on different objects.
_MISSING = object()
_BAK_RE = re.compile(r"\.bak-fillslider-(\d{8}-\d{6})(?:-(\d+))?\.json$")


class PenFillError(ValueError):
    """A refusal with an HTTP status attached (400 bad value, 404 unknown pen, 500 file)."""
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def seed_fill_mm(pen_id: str):
    """The built-in default fill_mm for `pen_id`, or None if the seed has no such pen."""
    prof = _PEN_PROFILE_SEED.get("profiles", {}).get(pen_id)
    v = prof.get("fill_mm") if isinstance(prof, dict) else None
    return float(v) if v is not None else None


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def calibrated_fill_mm(prof, pen_id=None):
    """The calibrated pass pitch of a profile: the stored `fill_mm_calibrated` if the slider
    ever edited it, else the current fill_mm (never edited => current IS calibrated), else
    (profile has no fill_mm at all) the built-in seed, else None."""
    prof = prof if isinstance(prof, dict) else {}
    for v in (prof.get(CAL_KEY), prof.get("fill_mm")):
        if _num(v) is not None:
            return _num(v)
    return seed_fill_mm(pen_id) if pen_id else None


def _edited(cur, calibrated):
    """True when the current fill_mm differs from the calibrated value."""
    if _num(cur) is None or _num(calibrated) is None:
        return False
    return abs(float(cur) - float(calibrated)) > 1e-9


def fill_provenance(pen_id, pen_profile) -> dict:
    """Run-provenance block: is this job's fill_mm an override of the calibrated pitch?
    Pure (reads only the seed constant and the resolved profile dict the route already
    holds), never raises. `edited_from_default` is true when the
    fill differs from the calibrated value."""
    try:
        cur = (pen_profile or {}).get("fill_mm")
        cal = calibrated_fill_mm(pen_profile, pen_id)
        ed = _edited(cur, cal)
        note = (f"OVERRIDE: fill_mm {cur} (calibrated {cal})" if ed
                else f"calibrated fill_mm {cal}")
        return {"fill_mm": cur, "fill_mm_calibrated": cal, "seed_fill_mm": seed_fill_mm(pen_id),
                "edited_from_default": ed, "note": note}
    except Exception as e:      # provenance must never cost a run its archive
        return {"error": f"{type(e).__name__}: {e}"}


def fill_info() -> dict:
    """Everything the slider UI needs, read fresh from disk."""
    d = _cal.load_pen_profiles()
    pens = {}
    for pid, p in d.get("profiles", {}).items():
        cur = p.get("fill_mm")
        cal = calibrated_fill_mm(p, pid)
        laid = p.get("laid_mm")
        pens[pid] = {
            "label": p.get("label"), "fill_mm": cur, "calibrated_fill_mm": cal,
            "seed_fill_mm": seed_fill_mm(pid), "edited": _edited(cur, cal),
            "laid_mm": laid,
            # No `fill_drives_pitch` flag: whether the slider moves the REAL spacing
            # depends on engine path AND the floor/cap, so it is read back per job as
            # `realised_pitch` after Generate instead of being predicted here.
        }
    return {"ok": True, "active": d.get("active"),
            "range": {"min": FILL_MM_MIN, "max": FILL_MM_MAX, "step": FILL_MM_STEP},
            "pens": pens}


def _validate(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PenFillError("fill_mm must be a number")
    v = float(value)
    if not math.isfinite(v):
        raise PenFillError("fill_mm must be a finite number")
    if not (FILL_MM_MIN <= v <= FILL_MM_MAX):
        raise PenFillError(f"fill_mm={v} is outside the allowed range "
                           f"[{FILL_MM_MIN}, {FILL_MM_MAX}] mm -- refusing to store it")
    return round(v, _ROUND_DP)


def _backup_path(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # The suffix must be ABOVE every existing one for this second, not merely the first free
    # name: after a prune frees a low n, reusing it would sort the new backup as the oldest.
    top = -1
    for f in path.parent.glob(f"{path.stem}.bak-fillslider-{stamp}*{path.suffix}"):
        k = _bak_sort_key(f.name)
        if k is not None and k[0] == stamp:
            top = max(top, k[1])
    n = top + 1
    while True:
        cand = path.parent / (f"{path.stem}.bak-fillslider-{stamp}" + (f"-{n}" if n else "") + path.suffix)
        if not cand.exists():
            return cand
        n += 1


def _bak_sort_key(name: str):
    m = _BAK_RE.search(name)
    return (m.group(1), int(m.group(2) or 0)) if m else None


def _prune_backups(path: Path, keep: int = KEEP_BACKUPS, protect: str = "") -> list:
    """Keep the newest `keep` slider backups, delete older ones, touch NOTHING else
    (only names matching exactly <stem>.bak-fillslider-<date>-<time>[-n]<suffix>).
    Never raises: a prune failure must not cost the edit its success."""
    gone = []
    try:
        mine = []
        for f in path.parent.glob(f"{path.stem}.bak-fillslider-*{path.suffix}"):
            k = _bak_sort_key(f.name)
            if (k is not None and f.name != protect
                    and f.name.startswith(path.stem + ".bak-fillslider-")):
                mine.append((k, f))
        mine.sort(key=lambda t: t[0])
        # the file just written (protect) always survives and counts toward `keep`
        for _, f in mine[:max(0, len(mine) + (1 if protect else 0) - keep)]:
            try:
                f.unlink()
                gone.append(f.name)
            except OSError:
                pass
    except OSError:
        pass
    return gone


def _replace_with_retry(path: Path, data: bytes, tries: int = 8) -> None:
    """atomic_write_bytes, retried on the transient PermissionError Windows raises from
    os.replace when a reader has the target open for a few milliseconds."""
    for i in range(tries):
        try:
            atomic_write_bytes(path, data)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.03 * (i + 1))


def _read_profiles_raw(path: Path):
    """(raw bytes, parsed dict). Seeds via the normal loader only if the file is MISSING;
    an unreadable/odd file is never overwritten here -- we refuse instead."""
    if not path.exists():
        _cal.load_pen_profiles()          # existing seeding behaviour, runtime home only
    raw = path.read_bytes()
    try:
        d = json.loads(raw.decode("utf-8-sig"))
    except Exception as e:
        raise PenFillError(f"{path.name} is not valid JSON ({type(e).__name__}: {e}); "
                           f"refusing to write", 500)
    if not (isinstance(d, dict) and isinstance(d.get("profiles"), dict)):
        raise PenFillError(f"{path.name} has no 'profiles' object; refusing to write", 500)
    return raw, d


def _dump_like(raw: bytes, d: dict) -> bytes:
    ascii_only = all(b < 128 for b in raw)
    text = json.dumps(d, indent=2, ensure_ascii=ascii_only)
    if b"\r\n" in raw:
        text = text.replace("\n", "\r\n")
    if raw.endswith((b"\n", b"\r\n")):
        text += "\r\n" if b"\r\n" in raw else "\n"
    return text.encode("utf-8")           # never a BOM


def _with_cal_key(prof: dict, new: float, cal_value) -> dict:
    """Copy of `prof` with fill_mm=new and, ONLY when the key is absent and cal_value is
    not None, fill_mm_calibrated=cal_value inserted right after fill_mm. Other key order kept."""
    out = {}
    for k, v in prof.items():
        if k == "fill_mm":
            out[k] = new
            if CAL_KEY not in prof and cal_value is not None:
                out[CAL_KEY] = cal_value
        else:
            out[k] = v
    if "fill_mm" not in out:
        out["fill_mm"] = new
        if CAL_KEY not in prof and cal_value is not None:
            out[CAL_KEY] = cal_value
    return out


def _only_expected_changes(orig: dict, written: dict, pen_id: str, allowed: set) -> bool:
    """True iff `written` equals `orig` except that profile `pen_id` differs in fill_mm (must
    differ) and, at most, the keys in `allowed`."""
    if {k: v for k, v in written.items() if k != "profiles"} != {k: v for k, v in orig.items() if k != "profiles"}:
        return False
    if set(written["profiles"]) != set(orig["profiles"]):
        return False
    if any(written["profiles"][k] != v for k, v in orig["profiles"].items() if k != pen_id):
        return False
    a, b = orig["profiles"][pen_id], written["profiles"][pen_id]
    changed = {k for k in set(a) | set(b) if a.get(k, _MISSING) != b.get(k, _MISSING)}
    return "fill_mm" in changed and changed <= ({"fill_mm"} | set(allowed))


def set_fill_mm(pen_id: str, value=None, reset: bool = False) -> dict:
    """Set (or, with reset=True, restore to the CALIBRATED value) fill_mm of ONE profile.
    Returns a result dict; raises PenFillError on any refusal. Writes nothing on refusal
    and nothing when the value is unchanged."""
    path = _cal.PEN_PROFILES_PATH
    with _cal.PEN_PROFILES_LOCK:
        _cal._guard(path)                 # shadow / Store-Python refusal, per access
        raw, d = _read_profiles_raw(path)
        prof = d["profiles"].get(pen_id)
        if not isinstance(prof, dict):
            raise PenFillError(f"unknown pen profile '{pen_id}'", 404)
        before = prof.get("fill_mm")
        cal = calibrated_fill_mm(prof, pen_id)
        if reset:
            if cal is None:
                raise PenFillError(f"'{pen_id}' has no calibrated fill_mm to reset to")
            new = float(cal)
        else:
            new = _validate(value)
        # The first real edit records the pre-edit value once; never again.
        first_edit = ((not reset) and CAL_KEY not in prof
                      and _num(before) is not None and before != new)
        cal_after = float(before) if first_edit else cal
        out = {"ok": True, "id": pen_id, "label": prof.get("label"), "before": before,
               "fill_mm": new, "fill_mm_calibrated": cal_after,
               "seed_fill_mm": seed_fill_mm(pen_id), "edited": _edited(new, cal_after),
               "changed": before != new, "backup": None, "pruned": [],
               "calibrated_key_written": False}
        if before == new:
            out["profile"] = prof
            return out

        # Build the new document and PROVE the only differences are fill_mm and, exactly
        # once on the first edit, the absent fill_mm_calibrated key.
        orig = json.loads(json.dumps(d))
        expect = json.loads(json.dumps(d))
        expect["profiles"][pen_id] = _with_cal_key(prof, new, float(before) if first_edit else None)
        d["profiles"][pen_id] = expect["profiles"][pen_id]
        data = _dump_like(raw, d)
        written = json.loads(data.decode("utf-8"))
        allowed = {CAL_KEY} if first_edit else set()
        if (written != expect or data.startswith(b"\xef\xbb\xbf")
                or not _only_expected_changes(orig, written, pen_id, allowed)):
            raise PenFillError("internal check failed: serialised document differs from the "
                               "intended edit; nothing written", 500)

        bak = _backup_path(path)
        atomic_write_bytes(bak, raw)       # exact old bytes, before every write
        _replace_with_retry(path, data)

        back = json.loads(path.read_bytes().decode("utf-8"))
        if back != expect:
            raise PenFillError(f"read-back of {path.name} does not match the edit "
                               f"(backup: {bak.name})", 500)
        out["backup"] = bak.name
        out["pruned"] = _prune_backups(path, protect=bak.name)
        out["calibrated_key_written"] = first_edit
        out["profile"] = back["profiles"][pen_id]
        return out
