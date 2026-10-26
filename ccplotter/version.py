import sys
import json
from pathlib import Path

# ── Version / build identity ───────────────────────────────────────────────────
# Frozen (PyInstaller) builds bundle a version.json written by desktop/build.ps1
# at the resource root (sys._MEIPASS), alongside vpype.toml/ai_config.json. Dev
# and source runs have no such file -- fall back to asking git directly. Must
# never raise: an unknown version can't be allowed to block app startup.

def _load_version_info():
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            vjson = Path(meipass) / "version.json"
            try:
                if vjson.is_file():
                    data = json.loads(vjson.read_text(encoding="utf-8"))
                    data["source"] = "frozen"
                    return data
            except Exception:
                pass
    try:
        import subprocess
        here = str(Path(__file__).resolve().parent)
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=here,
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=here,
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"], cwd=here,
            stderr=subprocess.DEVNULL) != 0
        return {"commit": sha, "branch": branch, "build_date": None,
                "dirty": dirty, "source": "source"}
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "build_date": None,
                "dirty": None, "source": "unknown"}

VERSION_INFO = _load_version_info()

def _version_string():
    """Short human string for the window title / footer badge, e.g.
    '<commit> <build date>' or '<commit> source' for a dev run."""
    v = VERSION_INFO
    commit = v.get("commit") or "unknown"
    date = (v.get("build_date") or "")[:10]
    parts = [commit]
    if date:
        parts.append(date)
    elif v.get("source") == "source":
        parts.append("source")
    if v.get("dirty"):
        parts.append("(dirty)")
    return " ".join(parts)
