"""Immutable, bounded source manifests for deferred Clean/X-ray previews."""
import os
import shutil
import stat
from collections import OrderedDict
from pathlib import Path

from ccplotter.config import WORK_DIR
from ccplotter.composite import _render_clean_preview, _render_xray_preview

_ROOT = WORK_DIR / "lazy_previews"
_MAX_JOBS = 12
_JOBS = OrderedDict()


def _is_link_or_reparse(path: Path) -> bool:
    """Windows junctions/reparse points are links even when is_symlink() is false."""
    try:
        st = path.stat(follow_symlinks=False)
    except OSError:
        return True
    return path.is_symlink() or bool(getattr(st, "st_file_attributes", 0) & 0x400)


def _link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)  # same-volume immutable snapshot: no duplicate bytes
    except OSError:
        shutil.copy2(src, dst)


def _remove_dir(root: Path):
    """Remove only one known, flat, program-owned manifest directory."""
    if _is_link_or_reparse(_ROOT) or _is_link_or_reparse(root):
        return
    try:
        owned_parent = root.parent.resolve(strict=True) == _ROOT.resolve(strict=True)
    except OSError:
        return
    if not owned_parent or not root.is_dir():
        return
    for path in root.iterdir():
        # Never recurse into or unlink a link/reparse point. Normal manifests
        # contain only direct regular SVG files created by this module.
        if _is_link_or_reparse(path):
            continue
        try:
            regular = stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
        except OSError:
            regular = False
        if regular:
            try: path.unlink()
            except OSError: pass
    try: root.rmdir()
    except OSError: pass


def _remove(job):
    _remove_dir(job["dir"])


_ROOT.mkdir(parents=True, exist_ok=True)
# Manifests are session-only. A prior crash/restart cannot leave unbounded
# snapshots behind; this touches only direct children of this owned directory.
if not _is_link_or_reparse(_ROOT):
    for _orphan in list(_ROOT.iterdir()):
        _remove_dir(_orphan)


def register(job_id: str, sources: dict):
    """Snapshot server-owned SVG sources under a fresh opaque job token."""
    root = _ROOT / job_id
    if _is_link_or_reparse(_ROOT):
        raise RuntimeError("lazy preview root is a link/reparse point")
    root.mkdir(parents=True, exist_ok=False)
    paths = {}
    try:
        for kind in ("clean", "routed", "connectors"):
            src = Path(sources[kind])
            dst = root / f"{kind}.svg"
            _link_or_copy(src, dst)
            paths[kind] = dst
    except Exception:
        _remove_dir(root)
        raise
    job = {"dir": root, "paths": paths, "pen_mm": float(sources["pen_mm"]),
           "xray": bool(sources.get("xray", False)), "rendered": {}}
    _JOBS[job_id] = job
    while len(_JOBS) > _MAX_JOBS:
        _, expired = _JOBS.popitem(last=False); _remove(expired)


def render(job_id: str, kind: str) -> Path:
    if kind not in ("clean", "xray") or job_id not in _JOBS:
        raise KeyError("unknown or expired preview job")
    job = _JOBS[job_id]
    if kind == "xray" and not job["xray"]:
        raise ValueError("X-ray is unavailable because this job has no routed connectors")
    if kind in job["rendered"]:
        return job["rendered"][kind]
    out = job["dir"] / f"{kind}.preview.svg"
    if kind == "clean":
        _render_clean_preview(job["paths"]["clean"], out, job["pen_mm"])
    else:
        _render_xray_preview(job["paths"]["routed"], job["paths"]["connectors"], out, job["pen_mm"])
    job["rendered"][kind] = out
    return out
