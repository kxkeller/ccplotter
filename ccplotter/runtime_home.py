"""Runtime home: the ONE place pen_profiles.json and calibration.json live.

WHY THIS EXISTS: if these two files were resolved as plain CWD-relative paths, the
desktop app (which chdir()s into %LOCALAPPDATA%\\ccPlotter before importing anything)
would be safe, but every script, test and harness run from the repo root would
silently read (and, for pen_profiles.json, write a seed into) a repo-root copy instead.
A stale copy carries a wrong z_touch_abs, which risks driving the pen into the bed
or producing a wrong result, and the provenance string ("profile:...") is identical
either way. So the files live only in the runtime home, and a CWD copy is refused.

RULES (all enforced in code, see guard()):
  1. The default home is the per-user runtime dir, never the process CWD:
     %LOCALAPPDATA%\\ccPlotter (falls back to %APPDATA%, then ~). The desktop launcher's
     data_dir() calls runtime_home() too, so there is exactly one resolver.
  2. A different home is an EXPLICIT opt-in: set CCPLOTTER_HOME to an existing absolute
     directory (harness envs, tests, scratch copies). Never implicit, never CWD.
  3. REFUSE ON DISAGREEMENT: if the process CWD holds a pen_profiles.json /
     calibration.json that is not the resolved file (samefile exempts CWD == home),
     raise ShadowConflictError -- whether its bytes differ, match, or the resolved file
     is missing (a byte-identical copy REFUSES too; it is a shadow waiting to go
     stale, and there is no escape hatch). Neither side is ever silently preferred.
  4. No seed is ever written into the CWD. cal.py seeds only into the resolved home,
     and only when the file there is missing/unusable.
  5. Store Python (WindowsApps) virtualises %LOCALAPPDATA%\\ccPlotter: reads can come
     from a stale per-package snapshot and writes never reach the real file. A path cannot dodge that filesystem
     filter, so
     the resolver REFUSES to use the real runtime dir from that interpreter. Copy the
     files out (sha256-verify), point CCPLOTTER_HOME at the copy, or use another Python.

This module is import-light and side-effect-free on purpose (no mkdir, no config import):
the desktop launcher imports it before os.chdir().
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ENV_VAR = "CCPLOTTER_HOME"
GUARDED_FILES = ("pen_profiles.json", "calibration.json")


class RuntimeHomeError(RuntimeError):
    """Base class: the runtime home cannot be trusted. Never swallow this."""


class ShadowConflictError(RuntimeHomeError):
    """A CWD-relative config file disagrees with the resolved runtime-home file."""


class UntrustedInterpreterError(RuntimeHomeError):
    """Store Python would read/write a virtualised copy of the real runtime dir."""


def _local_appdata_root() -> Path:
    root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
    return Path(root)


def default_home() -> Path:
    """The per-user runtime dir. Same expression the desktop launcher always used."""
    return _local_appdata_root() / "ccPlotter"


def runtime_home() -> Path:
    """Resolve the runtime home. Pure: touches no file and creates nothing.

    CCPLOTTER_HOME (explicit opt-in) must be an ABSOLUTE, EXISTING directory -- a relative
    value would just be the CWD shadow again, and a typo that silently created a fresh
    empty home would then seed a stale profile into it."""
    env = os.environ.get(ENV_VAR)
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            raise RuntimeHomeError(
                f"{ENV_VAR}={env!r} is not an absolute path; a relative home is the CWD "
                f"shadow this setting exists to prevent")
        if not p.is_dir():
            raise RuntimeHomeError(
                f"{ENV_VAR}={env!r} is not an existing directory (refusing to invent a "
                f"home and seed profiles into it)")
        return p.resolve()
    return default_home()


# ── Store Python (AppContainer) detection ─────────────────────────────────────

def is_store_python(executable: str | None = None, prefix: str | None = None) -> bool:
    """True for the Microsoft Store interpreter (python.exe under ...\\WindowsApps\\).
    Parameters exist so the unit test can drive it without a real Store Python."""
    exe = (executable if executable is not None else sys.executable or "").lower()
    pre = (prefix if prefix is not None else sys.prefix or "").lower()
    return "\\windowsapps\\" in exe or "\\windowsapps\\" in pre


def _under(child: Path, parent: Path) -> bool:
    try:
        c = os.path.normcase(os.path.abspath(child))
        p = os.path.normcase(os.path.abspath(parent))
    except Exception:
        return False
    return c == p or c.startswith(p + os.sep)


def _real_runtime_dirs() -> list[Path]:
    dirs = [default_home(), Path.home() / "AppData" / "Local" / "ccPlotter"]
    return dirs


def check_interpreter(home: Path, *, executable: str | None = None,
                      prefix: str | None = None) -> None:
    if not is_store_python(executable, prefix):
        return
    for real in _real_runtime_dirs():
        if _under(home, real):
            raise UntrustedInterpreterError(
                f"REFUSING: this is the Microsoft Store Python ({sys.executable}), which runs in "
                f"an AppContainer and virtualises {real}: reads return a stale per-package "
                f"snapshot and writes never reach the real file (a 4 mm z_touch_abs shift is "
                f"bed-crash class). The runtime home {home} is inside that directory. Copy "
                f"pen_profiles.json / calibration.json out with PowerShell or Git-Bash cp, "
                f"sha256-verify the copy, and set {ENV_VAR} to the copy (an absolute path "
                f"outside {real}); or run a non-Store Python.")


# ── CWD-shadow check ──────────────────────────────────────────────────────────

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_cwd_shadow(path: Path) -> None:
    """Refuse (raise) if CWD/<name> exists and is not `path` itself (samefile). Identical
    bytes do NOT exempt it -- only being the very same file does."""
    try:
        cwd_file = Path.cwd() / path.name
    except OSError:
        return
    if not cwd_file.exists():
        return
    resolved_exists = path.exists()
    if resolved_exists:
        try:
            if os.path.samefile(cwd_file, path):
                return
        except OSError:
            pass
    if not resolved_exists:
        other = f"resolved file {path}: MISSING"
    else:
        same = cwd_file.read_bytes() == path.read_bytes()
        other = (f"resolved file {path}: sha256 {_sha(path)[:12]}.. {path.stat().st_size} bytes"
                 + (" (BYTE-IDENTICAL to the CWD copy, but still a separate file)" if same else ""))
    raise ShadowConflictError(
        f"REFUSING (calibration shadow guard): {cwd_file} (sha256 {_sha(cwd_file)[:12]}.., "
        f"{cwd_file.stat().st_size} bytes) shadows the runtime-home copy -- {other}. "
        f"Neither is silently preferred: a stale pen/calibration file carries a wrong "
        f"z_touch_abs (bed-crash hazard). Delete or move the CWD copy, run from a directory "
        f"without it, or set {ENV_VAR} to the directory you intend to use.")


def guard(path: Path) -> None:
    """Full per-access guard for one config file path (cheap: stat calls only)."""
    check_interpreter(path.parent)
    check_cwd_shadow(path)


def guard_startup(home: Path, paths) -> None:
    """Import-time fail-fast used by ccplotter.config: the app/scripts refuse to start."""
    check_interpreter(home)
    for p in paths:
        check_cwd_shadow(p)
