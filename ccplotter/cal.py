import json
import sys
import threading
from pathlib import Path

from ccplotter.config import PEN_PROFILES_PATH, CALIBRATION_PATH, _PEN_PROFILE_SEED
from ccplotter.runtime_home import guard as _guard

# ONE lock for every WRITE of pen_profiles.json (save_pen_profiles here and
# pen_fill.set_fill_mm). Re-entrant so set_fill_mm may call helpers that also lock.
PEN_PROFILES_LOCK = threading.RLock()

# ── Helpers ───────────────────────────────────────────────────────────────────
# Both files resolve from the RUNTIME HOME (config.CALIBRATION_PATH /
# config.PEN_PROFILES_PATH), never the CWD. Every accessor re-runs _guard() --
# it raises (never returns) if a differing CWD copy appeared since startup or the
# interpreter is the virtualising Store Python. The guard sits OUTSIDE the
# read try/except below so load_cal's "return {}" fallback can never swallow it.

def load_cal() -> dict:
    path = CALIBRATION_PATH
    _guard(path)
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as e:
        print(
            f"WARNING: load_cal failed to read/parse {path.resolve()}: "
            f"{type(e).__name__}: {e} -- returning {{}} (stale/no safety envelope)",
            file=sys.stderr,
        )
        return {}

def save_cal(data: dict):
    cur = load_cal()
    cur.update(data)
    _guard(CALIBRATION_PATH)
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(cur, indent=2), encoding="utf-8")

# ── Pen refill profiles: per-pen calibration (nominal width, REAL solid-fill
# line width from the calibration plot, Z heights, speeds). fill_mm is the pocket
# pass spacing that plots solid — the payoff of the calibration session. ──

def load_pen_profiles() -> dict:
    """Load pen_profiles.json, falling back to the built-in seed.

    Read with encoding='utf-8-sig', as load_cal() does. Windows PowerShell's
    `Out-File -Encoding utf8` writes a UTF-8 BOM, and a BOM would make a bare
    read_text() parse fail and fall through to the seed-writing fallback,
    destroying every calibrated pen profile. A hand-edited file saved with a
    BOM must load, not reset the machine's calibration.

    The fallback does not overwrite the existing file in place: an unparseable
    profiles file is first preserved as a .CORRUPT-<n> sidecar, so a measured
    z_touch_abs is never lost to a parse error. Losing those values risks
    driving the pen into the bed."""
    _guard(PEN_PROFILES_PATH)
    try:
        d = json.loads(PEN_PROFILES_PATH.read_text(encoding="utf-8-sig"))
        if isinstance(d, dict) and d.get("profiles"):
            return d
        reason = "no 'profiles' key"
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
    if PEN_PROFILES_PATH.exists():
        n = 0
        while True:
            keep = PEN_PROFILES_PATH.with_suffix(f".json.CORRUPT-{n}")
            if not keep.exists():
                break
            n += 1
        try:
            keep.write_bytes(PEN_PROFILES_PATH.read_bytes())
            print(f"WARNING: {PEN_PROFILES_PATH.resolve()} unusable ({reason}); "
                  f"preserved as {keep.name} before writing the seed profiles. "
                  f"Calibrated Z heights are in that file -- restore them before "
                  f"plotting.", file=sys.stderr)
        except Exception as e2:
            print(f"WARNING: could not preserve unusable {PEN_PROFILES_PATH}: {e2}",
                  file=sys.stderr)
    # Seed ONLY into the resolved runtime home (PEN_PROFILES_PATH is absolute), and
    # only because the file there is missing/unusable -- never into the CWD.
    PEN_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PEN_PROFILES_PATH.write_text(json.dumps(_PEN_PROFILE_SEED, indent=2))
    return json.loads(json.dumps(_PEN_PROFILE_SEED))

def save_pen_profiles(d: dict):
    # Serialised with pen_fill.set_fill_mm (the fill slider) -- Flask runs
    # threaded, and two unlocked writers could interleave. Explicit UTF-8, never a BOM
    # (the locale default encoding is not guaranteed). NOTE the lock covers the WRITE
    # only: a caller that load_pen_profiles()s, edits and saves still holds a stale copy
    # across that window (the z/save/active routes are unchanged by design).
    with PEN_PROFILES_LOCK:
        _guard(PEN_PROFILES_PATH)
        PEN_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        PEN_PROFILES_PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")

def active_pen_profile() -> dict:
    d = load_pen_profiles()
    return d["profiles"].get(d.get("active"), next(iter(d["profiles"].values())))


def resolve_pen_offset(cal: dict, profile: dict | None) -> tuple[float, float]:
    """Composed pen<->nozzle XY offset in machine mm: GLOBAL mount + PER-PEN.

    Convention (composite.py): the pen TIP sits at nozzle + offset, and G-code
    commands the NOZZLE, so the emitter drives to (intended - offset).

    The global `pen_offset_*` in calibration.json describes the HOLDER's mount
    relative to the nozzle and is shared by every pen. The per-pen
    `pen_offset_*` on the profile is how far THAT pen's tip sits from where a
    bare refill's tip would sit in the same holder -- it is a property of the
    barrel, not of the machine, so it cannot live in calibration.json.

    Example: a full pen body (rather than a bare refill) has its tip ~8mm
    further toward the FRONT of the machine than a refill. Front is -Y here, so
    its per-pen offset is y = -8.0 and, with a shared offset of -45, the
    composed offset is -45 + -8 = -53.

    Every refill profile is 0.0/0.0, so composing is a no-op for them and
    their output is unchanged.
    """
    gx = float((cal or {}).get("pen_offset_x", 0.0))
    gy = float((cal or {}).get("pen_offset_y", 0.0))
    px = float((profile or {}).get("pen_offset_x", 0.0) or 0.0)
    py = float((profile or {}).get("pen_offset_y", 0.0) or 0.0)
    return gx + px, gy + py


def resolve_z_micro(cal: dict = None, profile: dict = None) -> tuple:
    """Resolve the MICRO-LIFT height (mm ABOVE paper contact) for the pen about
    to plot. PROFILE WINS, same convention as resolve_z_touch(): how far a pen
    must rise to stop marking is a property of the tip and its mount (a wet
    fountain nib on the 45deg mount bridges a gap a ballpoint clears), not of
    the machine, so it cannot live in calibration.json. The global `z_micro`
    stays as the FALLBACK for profiles that predate the field.

    composite.py must call this rather than reading `cal.get("z_micro", 0.6)`
    directly; otherwise every pen would silently run at the global height and a
    per-pen z_micro would have no effect on the emitted G-code.

    Returns (z_micro_mm, source) with source "profile:<label>" or "global" or
    "default" so the winning source is auditable, exactly like resolve_z_touch.
    Unlike z_touch there is no bed-contact risk: an absent value falls back to
    the 0.6 mm default."""
    if cal is None:
        cal = load_cal()
    if profile is None:
        profile = active_pen_profile()
    pv = (profile or {}).get("z_micro")
    if pv is not None:
        label = (profile or {}).get("label") or (profile or {}).get("id") or "active pen"
        return float(pv), f"profile:{label}"
    gv = (cal or {}).get("z_micro")
    if gv is not None:
        return float(gv), "global"
    return 0.6, "default"


class UnknownPenProfile(RuntimeError):
    """Raised when an explicit pen_id override (request field or harness
    fixture context) does not match any profile currently in
    pen_profiles.json. Deliberately loud -- never silently falls back to
    whatever happens to be ACTIVE. This is what lets a byte-identity harness
    (or any caller) pin a specific pen and find out immediately if that pen
    profile was renamed/deleted, instead of quietly drifting onto ambient
    runtime state (see scripts/replay_harness.py, which pins its pen this way)."""
    pass


def resolve_pen_profile(pen_id: str = None) -> dict:
    """Resolve the pen profile for THIS call/request. pen_id=None (the
    default, and what every pre-existing caller does implicitly) reproduces
    today's behavior exactly: the currently ACTIVE profile in
    pen_profiles.json, i.e. identical to active_pen_profile(). Passing an
    explicit pen_id pins a SPECIFIC profile by id regardless of what is
    currently active -- and raises UnknownPenProfile (never a silent
    fallback) if that id isn't present. Strictly additive: no existing
    caller/request that omits pen_id observes any behavior change from this
    function's introduction."""
    d = load_pen_profiles()
    if pen_id is None:
        return active_pen_profile()
    if pen_id not in d.get("profiles", {}):
        raise UnknownPenProfile(
            f"pen_id '{pen_id}' not found in pen_profiles.json (known ids: "
            f"{sorted(d.get('profiles', {}))}). Refusing to silently fall "
            f"back to the active pen -- fix the request/fixture or restore "
            f"that profile."
        )
    return d["profiles"][pen_id]

# ── Per-pen Z-touch resolution ─────────────────────────────────
# z_touch_abs (the absolute machine Z at which the pen TIP touches paper) is
# stored per pen, with calibration.json holding only a global fallback.
# Different pen barrels seat at very different heights in the holder (for the
# author's machine, a Pentel EnerGel measured 12.05mm and a BIC Cristal
# 18.75mm -- a 6.7mm difference), so one global value cannot serve two pens: using the wrong one either draws in the air (wasted
# paper) or drives the pen tip into the bed (damages pen/machine). This is the
# ONE place that resolves "what Z is paper contact for the pen we're about to
# plot with" -- every consumer (composite.py, gcode_post.py) must route
# through here instead of reading cal.get("z_touch_abs") directly, so there
# is exactly one fallback rule and one failure mode to audit.
class ZTouchNotCalibrated(RuntimeError):
    """Raised when neither the active pen profile nor calibration.json (global
    fallback) supplies a z_touch_abs. Deliberately NOT a silent 0.0 default --
    Z=0 is the bed, so a silent fallback there would drive the pen into the
    bed. Callers along the /api/process and
    /api/calibration_plot routes already wrap their _composite/_calibration_gcode
    call in `except Exception -> jsonify({"error": ...}), 500` (see
    ccplotter/routes/core.py and ccplotter/routes/pen.py), so raising
    here becomes a clear user-visible error instead of a silent bad plot."""
    pass

def resolve_z_touch(cal: dict = None, profile: dict = None) -> tuple:
    """Resolve the paper-contact Z (absolute machine mm) for the pen about to
    plot. PROFILE WINS: the active pen profile's own z_touch_abs is
    authoritative (per-pen barrels seat at different heights); calibration.json's
    global z_touch_abs is only a FALLBACK for profiles that predate this field
    (e.g. a profile deliberately left uncalibrated).
    Never returns/defaults to 0.0 -- that's the bed. Returns (z_touch_abs,
    source) where source is "profile:<id-or-label>" or "global" so the winning
    source is auditable (logged by callers). Raises ZTouchNotCalibrated if
    NEITHER supplies a value -- the caller's existing except-Exception->500
    JSON error path turns that into a user-visible message instead of a
    silent Z=0.0 plunge into the bed."""
    if cal is None:
        cal = load_cal()
    if profile is None:
        profile = active_pen_profile()
    pv = (profile or {}).get("z_touch_abs")
    if pv is not None:
        label = (profile or {}).get("label") or (profile or {}).get("id") or "active pen"
        return float(pv), f"profile:{label}"
    gv = (cal or {}).get("z_touch_abs")
    if gv is not None:
        return float(gv), "global"
    label = (profile or {}).get("label") or (profile or {}).get("id") or "active pen"
    raise ZTouchNotCalibrated(
        f"No z_touch_abs for '{label}' and no global fallback in calibration.json. "
        f"Run Z calibration for this pen before plotting -- refusing to default to "
        f"Z=0.0 (that is the bed)."
    )
