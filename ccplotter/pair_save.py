"""ccplotter/pair_save.py -- copy several work-dir files into a user-chosen folder (two-pen Save).

Used by the desktop bridge (desktop/ccplotter_app.py _JsApi.save_files_to_folder). Kept out of the desktop module so it
is unit-testable without pywebview. Source paths are RELATIVE to the work dir (e.g. "runs/ab12cd34/gcode_blue.gcode")
and may not escape it; destination names are bare file names ending in .gcode. Nothing is overwritten unless asked.
"""
import os
import shutil
from pathlib import Path


def _safe_src(work_dir, rel):
    rel = str(rel).replace("\\", "/")
    if not rel or rel.startswith("/") or ":" in rel or any(part in ("", ".", "..") for part in rel.split("/")):
        raise ValueError(f"bad source path: {rel!r}")
    p = (Path(work_dir) / rel).resolve()
    root = Path(work_dir).resolve()
    if root not in p.parents:
        raise ValueError(f"source escapes the work folder: {rel!r}")
    return p


def _safe_name(name):
    name = str(name)
    if name != os.path.basename(name) or name in ("", ".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"bad file name: {name!r}")
    if not name.lower().endswith(".gcode"):
        raise ValueError(f"file name must end in .gcode: {name!r}")
    return name


def copy_files(work_dir, folder, items, overwrite=False):
    """items = [[work_relative_source, dest_name], ...]. All-or-nothing: every source and name is validated, and (unless
    overwrite) every destination must be free, BEFORE the first byte is written. Returns {ok, paths[]} or
    {ok: False, error[, exists[]]}."""
    try:
        folder = Path(str(folder))
        if not folder.is_dir():
            return {"ok": False, "error": f"not a folder: {folder}"}
        plan = []
        for src_rel, name in items:
            src = _safe_src(work_dir, src_rel)
            if not src.is_file():
                return {"ok": False, "error": f"not found: {src}"}
            plan.append((src, folder / _safe_name(name)))
        if len({str(d).lower() for _, d in plan}) != len(plan):
            return {"ok": False, "error": "two files would get the same name"}
        exists = [d.name for _, d in plan if d.exists()]
        if exists and not overwrite:
            return {"ok": False, "exists": exists, "error": "exists: " + ", ".join(exists)}
        for src, dest in plan:
            shutil.copyfile(str(src), str(dest))
        return {"ok": True, "paths": [str(d) for _, d in plan]}
    except Exception as e:
        return {"ok": False, "error": repr(e)}
