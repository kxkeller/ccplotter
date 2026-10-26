import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

from ccplotter.atomic_io import atomic_copy, atomic_write_bytes, atomic_write_text
from ccplotter.config import WORK_DIR, CALIBRATION_PATH
from ccplotter import runtime_home as _rh
from ccplotter.pen_fill import fill_provenance as _pen_fill_provenance
from ccplotter.version import VERSION_INFO

# ── Run archive: self-contained snapshots of expensive runs (AI cleans, gcode
# builds) so the user can compare/restore without re-running minutes of AI. ──
RUNS_DIR = WORK_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)
RUNS_INDEX = WORK_DIR / "runs.json"
RUNS_KEEP = 20
_RUNS_LOCK = threading.Lock()

def _runs_load():
    try:
        return json.loads(RUNS_INDEX.read_text())
    except Exception:
        return []

# -- Run provenance: the request that made a run, so ANY plotted job can be
# regenerated exactly. One small JSON per run (no images) under the RUNTIME HOME
# (runtime_home resolver, never the CWD): <home>/run_provenance/<run id>.json. It is a SIDECAR,
# not a runs.json field, so /api/runs stays small and old runs.json records load
# unchanged (they simply have no "provenance_file"). Kept independently of RUNS_KEEP:
# the gcode/preview snapshots are heavy and rotate at 20, these are ~KB and a plotted job
# is usually investigated long after 20 more runs.
PROVENANCE_DIRNAME = "run_provenance"
PROVENANCE_KEEP = 500
PROVENANCE_SCHEMA = 1
_TREE_STATE = None

def _tree_state() -> dict:
    """Source-tree identity beyond commit+dirty, computed ONCE per process (source runs
    only): dirty_diff_sha256 = sha256 of `git diff HEAD` (tracked files), untracked_count
    = number of `??` entries in `git status --porcelain` (a dirty tree with new
    untracked modules is not named by the diff hash alone, the count flags that gap;
    without -uall an untracked directory counts as one). None when not a source run."""
    global _TREE_STATE
    if _TREE_STATE is not None:
        return _TREE_STATE
    st = {"dirty_diff_sha256": None, "untracked_count": None}
    try:
        if VERSION_INFO.get("source") == "source":
            here = str(Path(__file__).resolve().parent)
            if VERSION_INFO.get("dirty"):
                out = subprocess.check_output(["git", "diff", "HEAD"], cwd=here,
                                              stderr=subprocess.DEVNULL, timeout=15)
                st["dirty_diff_sha256"] = hashlib.sha256(out).hexdigest()
            por = subprocess.check_output(["git", "status", "--porcelain"], cwd=here,
                                          stderr=subprocess.DEVNULL, timeout=30)
            st["untracked_count"] = sum(1 for ln in por.splitlines() if ln.startswith(b"??"))
    except Exception:
        pass
    _TREE_STATE = st
    return st

def resolved_from_stats(stats, reach_default_id=None) -> dict:
    """The resolved quantities a run actually used, read from the response-side
    stats the route already holds (trace_info.effective_res is the per-response value,
    NOT the requested resolution). None-safe; absent keys stay None."""
    ti = (stats or {}).get("trace_info") or {}
    au = ti.get("audit") or {}
    mb = (stats or {}).get("machine_bbox") or {}
    out = {
        "requested_res": ti.get("requested_res"),
        "effective_res": ti.get("effective_res"),
        "fill_px": au.get("fill_px"),
        "mm_per_px": au.get("mm_per_px"),
        "fill_mm_resolved": au.get("fill_mm_resolved"),
        "reach_profile": mb.get("reach_profile") or reach_default_id,
    }
    # Per-item REALISED pass pitch (final placed scale), added only when the job has
    # tracev3-engine raster items. NB `mm_per_px` above is the trace's own fill_mm/fill_px
    # back-solve, NOT the placed scale; realised_pitch is the honest one.
    rp = (stats or {}).get("realised_pitch")
    if rp:
        out["realised_pitch"] = rp
    return out

def build_run_provenance(request_body, effective_body, pen_id, pen_profile, resolved=None,
                         trace_boundary=None):
    """Assemble the provenance dict. request_body = the body exactly as received (before
    any recipe/default merge); effective_body = after the photo512/bwphoto/illustration recipe merge.
    trace_boundary (optional) = the ccplotter.trace_boundary.collect() block.
    Never raises: a failure here must not cost the run its gcode/preview archive."""
    try:
        return _build_run_provenance(request_body, effective_body, pen_id, pen_profile, resolved,
                                     trace_boundary)
    except Exception as e:
        return {"schema": PROVENANCE_SCHEMA, "error": f"{type(e).__name__}: {e}",
                "request_body": request_body, "pen_id": pen_id}

def _build_run_provenance(request_body, effective_body, pen_id, pen_profile, resolved,
                          trace_boundary=None):
    try:
        cal_sha = hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest()
    except Exception:
        cal_sha = None
    try:
        eb_sha = hashlib.sha256(json.dumps(effective_body, sort_keys=True, default=str,
                                           separators=(",", ":")).encode("utf-8")).hexdigest()
    except Exception as e:
        eb_sha = f"error: {type(e).__name__}: {e}"
    prov = {
        "schema": PROVENANCE_SCHEMA,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "request_body": request_body,
        "effective_body": effective_body,
        "effective_body_note": "after recipe merge (photo512/bwphoto/illustration); per-field "
                               "defaults (body.get(k, default)) are applied in code "
                               "and are NOT materialised here",
        "effective_body_sha256": eb_sha,
        "effective_body_sha256_note": "sha256 of json.dumps(effective_body, sort_keys=True, "
                                      "separators=(',',':'), default=str) utf-8",
        "pen_id": pen_id,
        "pen_profile": pen_profile,
        # Whether this job's fill_mm differs from the default (ASCII note, additive key).
        "pen_fill": _pen_fill_provenance(pen_id, pen_profile),
        "resolved": resolved,
        "calibration_sha256": cal_sha,
        "runtime_home": str(_rh.runtime_home()),
        "app": {"commit": VERSION_INFO.get("commit"), "branch": VERSION_INFO.get("branch"),
                "dirty": VERSION_INFO.get("dirty"), "source": VERSION_INFO.get("source"),
                "build_date": VERSION_INFO.get("build_date"),
                "dirty_diff_sha256": _tree_state()["dirty_diff_sha256"],
                "untracked_count": _tree_state()["untracked_count"],
                "dirty_diff_note": "git diff HEAD of tracked files only; untracked "
                                   "files are NOT covered (untracked_count says how many "
                                   "there were)"},
    }
    # Trace-boundary block (additive key; schema stays 1, old readers ignore it).
    # Absent for jobs with no traced item data; never contains an exception.
    if trace_boundary is not None:
        prov["trace_boundary"] = trace_boundary
    return prov

def _provenance_dir() -> Path:
    return _rh.runtime_home() / PROVENANCE_DIRNAME

def _provenance_write(rid: str, prov: dict) -> str:
    """Write the sidecar atomically. Returns the runtime-home-relative path stored on
    the run record. Pruning is separate (_provenance_prune) so it can be told which
    sidecars are still referenced."""
    d = _provenance_dir()
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_text(d / f"{rid}.json", json.dumps(prov, indent=1, default=str))
    return f"{PROVENANCE_DIRNAME}/{rid}.json"

def _provenance_prune(live_ids) -> None:
    """Keep the newest PROVENANCE_KEEP sidecars, and NEVER remove one whose run is still
    in runs.json (`live_ids`), whatever its mtime."""
    d = _provenance_dir()
    if not d.is_dir():
        return
    live = {str(i) for i in live_ids}
    files = sorted(d.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    for old in files[PROVENANCE_KEEP:]:
        if old.stem in live:
            continue
        try:
            old.unlink()
        except OSError:
            pass

# -- G-code header back-link. The file the user sideloads onto the printer is a raw copy of
# work/layout.gcode (desktop save_file / save_to_downloads), so the link must live in THAT
# file. One plain ';' comment line, pure ASCII, inserted right after the vpype
# "; Source:" header line (no G92, no motion; the firmware ignores comments):
#   ; ccplotter-run: id=<run id> provenance=run_provenance/<run id>.json
# A per-run id makes the raw gcode sha nondeterministic, so scripts/replay_harness.py
# (and the parity worker) strip exactly this one line before hashing, see
# strip_run_backlink(); recorded manifest hashes are of files that never had the line.
# vcache note: a vcache hit restores gcode stamped with the OLDER run's id. That is
# correct provenance by key construction -- the vcache key covers the full request, so the
# older run's sidecar describes exactly the request that produced these bytes.
BACKLINK_PREFIX = "; ccplotter-run: "

def gcode_backlink_line(rid: str, provenance_file: str) -> str:
    line = f"{BACKLINK_PREFIX}id={rid} provenance={provenance_file}"
    line.encode("ascii")        # raises on non-ASCII: never ship a firmware-hostile byte
    return line

def strip_run_backlink(data: bytes) -> bytes:
    """Remove every back-link line (any EOL style) from gcode bytes."""
    pre = BACKLINK_PREFIX.encode("ascii")
    return b"".join(ln for ln in data.splitlines(keepends=True) if not ln.startswith(pre))

SOURCE_PREFIX = "; Source:"

def strip_run_stamps(data: bytes) -> bytes:
    """Remove BOTH per-run stamps from gcode bytes: every back-link line (as
    strip_run_backlink) and every `; Source:` line. The Source line is the absolute path of
    the uploaded file, which differs between two runs of the same request made from
    different work dirs / machines, so byte-equality of two otherwise-identical runs is
    only meaningful on the stripped form."""
    pre = (BACKLINK_PREFIX.encode("ascii"), SOURCE_PREFIX.encode("ascii"))
    return b"".join(ln for ln in data.splitlines(keepends=True) if not ln.startswith(pre))

def stamp_gcode_backlink(path, rid: str, provenance_file: str) -> bool:
    """Insert (or replace) the back-link line in a gcode file, atomically, preserving the
    file's line-ending style. Returns True if written."""
    path = Path(path)
    data = strip_run_backlink(path.read_bytes())
    line = gcode_backlink_line(rid, provenance_file).encode("ascii")
    lines = data.splitlines(keepends=True)
    eol = b"\r\n" if b"\r\n" in data[:4096] else b"\n"
    at = next((i for i, ln in enumerate(lines[:20]) if ln.startswith(b"; Source:")), 0) + 1
    lines.insert(at, line + eol)
    atomic_write_bytes(path, b"".join(lines))
    return True

def record_sheet_provenance(sheet_id, out_path, pen_id, pen_profile, request_body,
                            info=None, z_touch=None, z_touch_source=None):
    """A calibration-sheet generation leaves a sidecar like a job does, so a
    plotted sheet's gcode sha256 can be recovered later. Same directory and writer as
    job provenance (<home>/run_provenance/), id "cal-<8 hex>" (no runs.json entry: sheets
    are not jobs and must not crowd the 20-run archive). The gcode is NOT touched (no
    back-link stamp). Returns {"id","provenance_file","sha256"} or None; never raises."""
    try:
        data = Path(out_path).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        rid = "cal-" + uuid4().hex[:8]
        prov = {
            "schema": PROVENANCE_SCHEMA,
            "kind": "calibration_sheet",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sheet_type": sheet_id,
            "pen_id": pen_id,
            "pen_profile": pen_profile,
            "z_touch": z_touch,
            "z_touch_source": z_touch_source,
            "request_body": request_body,
            "output_path": str(out_path),
            "gcode_sha256": sha,
            "gcode_bytes": len(data),
            "generator_info": {k: v for k, v in (info or {}).items()
                               if isinstance(v, (str, int, float, bool, type(None)))},
            "runtime_home": str(_rh.runtime_home()),
            "app": {"commit": VERSION_INFO.get("commit"), "branch": VERSION_INFO.get("branch"),
                    "dirty": VERSION_INFO.get("dirty"), "source": VERSION_INFO.get("source")},
        }
        rel = _provenance_write(rid, prov)
        try:
            import logging
            logging.getLogger("ccplotter.calsheet").info(
                "calibration_sheet sheet=%s pen_id=%s sha256=%s file=%s provenance=%s",
                sheet_id, pen_id, sha, Path(out_path).name, rel)
        except Exception:
            pass
        return {"id": rid, "provenance_file": rel, "sha256": sha}
    except Exception as e:
        print(f"  [runs] calibration-sheet provenance failed: {e}")
        return None

def load_run_provenance(rid: str):
    """Read a run's provenance back, or None (old run / pruned / unreadable)."""
    try:
        return json.loads((_provenance_dir() / f"{rid}.json").read_text(encoding="utf-8"))
    except Exception:
        return None

def _runs_add(rec: dict, copy_files: dict = None, provenance: dict = None,
              stamp_gcode: bool = False, rid: str = None, stamp_labels=("gcode",)):
    """Append a run record; copy listed files into work/runs/<id>/ so the
    snapshot survives source deletion / layout.* overwrites. Keeps RUNS_KEEP.
    `provenance` (optional) is written as a sidecar under the runtime home and
    linked from the record as rec["provenance_file"]; omitted -> record unchanged.
    `stamp_gcode` (needs provenance): write the back-link line into the SOURCE gcode
    (copy_files["gcode"], i.e. work/layout.gcode) BEFORE it is archived, so the sideloaded
    file, the archive copy and the bundle all carry it.
    `rid` (optional, used by the two-pen job) pre-chooses the run id so files can be named
    with it before the record exists; None -> a fresh uuid. `stamp_labels` (default
    ("gcode",)) lists the copy_files labels that get the back-link;
    the two-pen job stamps both of its gcode files."""
    import shutil
    with _RUNS_LOCK:
        runs = _runs_load()
        rid = rid or uuid4().hex[:8]
        rec = dict(rec)
        rec["id"] = rid
        rec["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        rdir = RUNS_DIR / rid
        rdir.mkdir(exist_ok=True)
        if provenance is not None:
            try:
                rec["provenance_file"] = _provenance_write(rid, provenance)
            except Exception as e:
                print(f"  [runs] provenance write failed: {e}")
        _stamp = [l for l in stamp_labels if (copy_files or {}).get(l)]
        if stamp_gcode and rec.get("provenance_file") and _stamp:
            try:
                for _l in _stamp:
                    stamp_gcode_backlink(copy_files[_l], rid, rec["provenance_file"])
                rec["gcode_backlink"] = True
            except Exception as e:
                print(f"  [runs] gcode back-link failed: {e}")
        rec["files"] = {}
        for label, src in (copy_files or {}).items():
            try:
                if not src:                       # skip empty/None (e.g. lineart failed)
                    continue
                p = Path(src)
                if p.is_file():                   # is_file() guards against '' -> '.' dir
                    dst = rdir / (label + p.suffix)
                    atomic_copy(p, dst)
                    rec["files"][label] = f"runs/{rid}/{dst.name}"
            except Exception as e:
                print(f"  [runs] snapshot copy failed ({label}): {e}")
        runs.insert(0, rec)
        for old in runs[RUNS_KEEP:]:
            shutil.rmtree(RUNS_DIR / old.get("id", "_"), ignore_errors=True)
        runs = runs[:RUNS_KEEP]
        try:
            atomic_write_text(RUNS_INDEX, json.dumps(runs, indent=1))
        except Exception as e:
            print(f"  [runs] index write failed: {e}")
        if provenance is not None:
            try:
                _provenance_prune([r.get("id") for r in runs])
            except Exception as e:
                print(f"  [runs] provenance prune failed: {e}")
        return rid
