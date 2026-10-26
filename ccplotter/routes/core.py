"""ccplotter/routes/core.py -- core Flask routes: index, /api/calibrate_z, /api/reconnect,
/api/version, /api/status, /api/calibration, /api/add_drawing, /api/add_text,
/api/clear_drawings, /api/delete_drawing, _PROCESS_LOCK + /api/process (the
main generation endpoint), /work/<filename>, /download_bundle, /api/send,
/api/gcode, /api/fans_off.

Shared state: _PROCESS_LOCK is defined here because it wraps the route handler
itself. It is a plain threading.Lock(), never read from any other module.

_sketch_mod (`import ccplotter.sketch as _sketch_mod`) is imported as a module
(not `from ccplotter.sketch import _SKETCH_AUDIT`) because _SKETCH_AUDIT is
rebound via `global _SKETCH_AUDIT` inside _sketch_svg; only a live module
reference sees the rebound value. Likewise `_bundle_mod._LAST_BUNDLE` is read
through its module, and the vcache group is accessed through the function
calls `_vcache_get`/`_vcache_put` rather than raw dict reads.
"""
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import threading
import time
import zipfile
from pathlib import Path
from uuid import uuid4

from flask import jsonify, render_template, request, send_file, send_from_directory
import websocket
import vpype as vp
import vpype_cli

from ccplotter.atomic_io import atomic_target, atomic_write_bytes
from ccplotter.config import (PAPER_SIZES, PRINTER_IP, BED_MAX, PEN_MIN_X, PEN_MIN_Y,
                               PEN_MAX_X, PEN_MAX_Y, WORK_DIR, RASTER_EXTS, VECTOR_EXTS,
                               PX2MM, FAN_OFF_GCODE, DBV3_INBOX_DIR, DBV3_ARCHIVE_DIR,
                               SIMPLIFY_FILL_MULT, PHOTO512_RECIPE, XRAY_LAYER,
                               BW_PHOTO_RECIPE, BW_PHOTO_FRONT_END, ILLUSTRATION_RECIPE,
                               BW_CLAHE_SOFT_CLIP, CORE_R_MM_DEFAULT, STITCH_MM_MAX, ROWS_CORE_R_MM_DEFAULT)
from ccplotter.version import VERSION_INFO, _version_string
from ccplotter.cal import (load_cal, save_cal, active_pen_profile, resolve_pen_profile, load_pen_profiles,
                           UnknownPenProfile, resolve_pen_offset)
from ccplotter.reach import active_reach_profile, pen_reach_profile
from ccplotter.mask import _inkscape_exe
from ccplotter.printer import printer
from ccplotter.composite import _composite
from ccplotter.calibration_plots import _registration_marks_gcode
from ccplotter.svg_io import _parse_dbv3_metadata
from ccplotter.stats_warnings import build_warnings
from ccplotter.vcache import _source_fingerprint, _resolved_inputs_fingerprint, _variant_key, _variant_label, _vcache_get, _vcache_put
import ccplotter.sketch as _sketch_mod
from ccplotter.runs import _runs_add, build_run_provenance, resolved_from_stats
import ccplotter.bundle as _bundle_mod
from ccplotter.bundle import _snapshot_bundle, _BUNDLE_LOCK
from ccplotter.lazy_preview import register as _lazy_preview_register, render as _lazy_preview_render
from ccplotter import coverage_audit as _coverage_audit_mod

from ccplotter.routes import app

@app.route("/")
def index(): return render_template("index.html", paper_sizes=list(PAPER_SIZES.keys()))

def _printer_unconfigured():
    """503 response when no printer host is configured (CCPLOTTER_PRINTER_HOST unset), else None."""
    if PRINTER_IP:
        return None
    return jsonify({"ok": False, "error": "No printer host configured. Set CCPLOTTER_PRINTER_HOST "
                    "to the printer's address to use network printer features."}), 503

@app.route("/api/calibrate_z", methods=["POST"])
def api_calibrate_z():
    if (_unc := _printer_unconfigured()) is not None: return _unc
    z_result, got_z = [None], threading.Event()
    def _open(ws): ws.send(printer._make_pkt(0))
    def _msg(ws, raw):
        try:
            d = json.loads(raw); coords = d.get("Status", {}).get("CurrenCoord", "")
            if coords:
                parts = coords.split(",")
                if len(parts) >= 3: z_result[0] = float(parts[2].strip()); got_z.set()
        except Exception: pass
    obj = websocket.WebSocketApp(f"ws://{PRINTER_IP}/websocket", on_open=_open, on_message=_msg, on_error=lambda ws, e: got_z.set())
    t = threading.Thread(target=lambda: obj.run_forever(ping_timeout=4), daemon=True); t.start()
    got_z.wait(timeout=12); obj.close()
    z = z_result[0] if z_result[0] is not None else printer.get_z()
    if z is None: return jsonify({"ok": False, "error": "No Z reading. Touch pen to paper first."})
    save_cal({"z_touch_abs": z}); return jsonify({"ok": True, "z_touch_abs": z, "message": f"Saved touch Z = {z:.2f}mm"})

@app.route("/api/reconnect", methods=["POST"])
def api_reconnect():
    if (_unc := _printer_unconfigured()) is not None: return _unc
    printer.disconnect(); time.sleep(1); printer.connect(); return jsonify({"ok": True})

@app.route("/api/version")
def api_version():
    v = dict(VERSION_INFO)
    v["version_string"] = _version_string()
    v["tracer"] = "inkscape"
    v["inkscape_found"] = _inkscape_exe() is not None
    return jsonify(v)

@app.route("/api/status")
def api_status():
    snap, cal = printer.get_snapshot(), load_cal()
    reach = active_reach_profile(cal)
    mn_x, mn_y, mx_x, mx_y = reach["bbox"]
    snap["plot_limits"] = {"usable_w": round(mx_x - mn_x, 1), "usable_h": round(mx_y - mn_y, 1), "bed_max": BED_MAX,
                           "profile_id": reach["id"], "profile_label": reach["label"],
                           "polygon": reach["polygon"], "bbox": reach["bbox"]}
    return jsonify(snap)

@app.route("/api/calibration", methods=["GET"])
def api_cal_get():
    cal = load_cal()
    data = dict(cal)
    # Per-pen reach: some mounts foul the printer well inside the
    # machine's own nozzle envelope in a way that is a property of the PEN, not
    # the machine (e.g. the 45-degree Preppy mount above nozzle Y 201.5; the
    # BIC has no such limit -- see ccplotter/reach.py pen_reach_profile / the
    # pen profile's optional "reach_clamp"). The plate view needs the
    # COMPOSED per-pen offset and the CLAMPED envelope for whichever pen is
    # selected, not just the global mount offset -- so an optional
    # ?pen_id=<id> resolves that specific profile (omitted = the currently
    # ACTIVE profile, matching resolve_pen_profile()'s no-arg contract).
    # Wrapped in its own try/except and added as a separate top-level key
    # ("pen_reach") so any failure here (e.g. an unknown pen_id) cannot break
    # the rest of the calibration payload; a consumer that ignores this key or
    # omits pen_id sees the same JSON as without it.
    try:
        pen_id = request.args.get("pen_id") or None
        pen_profile = resolve_pen_profile(pen_id)
        reach = pen_reach_profile(active_reach_profile(cal), pen_profile)
        pen_off_x, pen_off_y = resolve_pen_offset(cal, pen_profile)
        # FRAMES: "polygon"/"bbox" here are NOZZLE coordinates
        # -- the machine's own travel envelope, which is what G-code commands and
        # therefore what validate_gcode_xy_motion() checks against the emitted
        # file, with no conversion between the check and the bytes. The field is
        # called "pen_reach", and reading its bbox as a PEN limit is a trap: with
        # the Preppy at composed offset (+3.1,-49.9) the nozzle stops at X 255.0
        # while the tip reaches 258.1, and for a mount with a nozzle limit of
        # 236.5 the tip limit is 222.0 -- quoting the nozzle number as a pen limit
        # would let artwork placed at pen X 236.5 emit a commanded X 251.0.
        # So publish BOTH frames explicitly and let each consumer take the one it
        # means. Pen frame = nozzle frame + composed pen offset (tip = nozzle +
        # offset, per ccplotter/cal.py resolve_pen_offset).
        _pen_poly = [[float(x) + pen_off_x, float(y) + pen_off_y] for x, y in reach["polygon"]]
        _nb = reach["bbox"]
        data["pen_reach"] = {
            "pen_id": pen_id or load_pen_profiles().get("active"),
            "pen_label": pen_profile.get("label"),
            # --- nozzle frame (machine travel; what G-code commands) ---
            "polygon": reach["polygon"],
            "bbox": reach["bbox"],
            "nozzle_polygon": reach["polygon"],
            "nozzle_bbox": reach["bbox"],
            # --- pen frame (where the tip can actually draw) ---
            "pen_polygon": _pen_poly,
            "pen_bbox": [_nb[0] + pen_off_x, _nb[1] + pen_off_y,
                         _nb[2] + pen_off_x, _nb[3] + pen_off_y],
            "frames": {"polygon": "nozzle", "bbox": "nozzle",
                       "pen_polygon": "pen_tip", "pen_bbox": "pen_tip"},
            "composed_pen_offset_x": pen_off_x,
            "composed_pen_offset_y": pen_off_y,
            "clamped": bool(reach.get("pen_clamp")),
        }
    except Exception as e:
        data["pen_reach_error"] = str(e)
    resp = jsonify(data); resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'; return resp

@app.route("/api/add_drawing", methods=["POST"])
def api_add_drawing():
    if "file" not in request.files: return jsonify({"error": "No file"}), 400
    f = request.files["file"]; suffix = Path(f.filename).suffix.lower()
    if suffix not in RASTER_EXTS and suffix not in VECTOR_EXTS: return jsonify({"error": f"Unsupported format: {suffix}"}), 400
    
    # HARD LIMIT: 50MB
    f.seek(0, os.SEEK_END); size = f.tell(); f.seek(0)
    if size > 50 * 1024 * 1024:
        return jsonify({"error": f"File too large ({size/1e6:.1f}MB). Max 50MB."}), 400
        
    did = uuid4().hex[:8]; src_name = f"src_{did}{suffix}"; f.save(str(WORK_DIR / src_name))
    return jsonify({"id": did, "name": f.filename, "is_raster": suffix in RASTER_EXTS, "src_name": src_name})

@app.route("/api/dbv3_inbox")
def api_dbv3_inbox():
    """List SVGs sitting in the DrawingBotV3 watch-folder. Manual/
    on-open refresh only -- no background watcher thread, no polling loop."""
    files = []
    for p in sorted(DBV3_INBOX_DIR.glob("*.svg"), key=lambda x: -x.stat().st_mtime):
        try:
            data = p.read_bytes()
            st = p.stat()
            files.append({"filename": p.name, "mtime": st.st_mtime, "size": st.st_size,
                          "sha256_short": hashlib.sha256(data).hexdigest()[:12]})
        except Exception:
            continue
    return jsonify({"files": files, "inbox_dir": str(DBV3_INBOX_DIR.resolve())})

@app.route("/api/dbv3_import", methods=["POST"])
def api_dbv3_import():
    """Copy a file out of the DBV3 inbox into work/ as a normal src_<id>.svg --
    the same 'existing file -> fresh drawing source' pattern
    /api/runs/<rid>/use_source already uses. Everything downstream (fit,
    reach safety, routing, gcode) is the unmodified vector-source pipeline."""
    body = request.json or {}
    filename = str(body.get("filename", ""))
    if not filename or Path(filename).name != filename or Path(filename).suffix.lower() != ".svg":
        return jsonify({"error": "invalid filename"}), 400
    src = DBV3_INBOX_DIR / filename
    if not src.is_file():
        return jsonify({"error": "file not found in DBV3 inbox"}), 404
    data = src.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    did = uuid4().hex[:8]
    dst = WORK_DIR / f"src_{did}.svg"
    atomic_write_bytes(dst, data)
    # Provenance: real DBV3 1.6.25 exports carry a rich
    # XML comment (version, shape/pen-lift/travel-distance counts, page vs
    # drawing size, per-pen name+distance) right after the <svg> tag -- see
    # svg_io._parse_dbv3_metadata docstring for the exact format and why it's
    # decoded as a bounded header slice, not the whole file. None for any SVG
    # that lacks it (not every inbox file is guaranteed to be a DBV3 export).
    dbv3_meta = None
    pen_mismatch_note = None
    pen_mismatch_suggested_pen = None
    try:
        dbv3_meta = _parse_dbv3_metadata(data)
        if dbv3_meta and dbv3_meta.get("pens"):
            active = active_pen_profile()
            active_label = str(active.get("label", ""))
            exported_names = ", ".join(p["name"] for p in dbv3_meta["pens"] if p.get("name"))
            # Informational only -- a plain substring match is enough to
            # flag "these look like different pens"; no attempt to parse
            # exported pen widths (DBV3's own pen library, not ours to read).
            if exported_names and active_label and exported_names.lower() not in active_label.lower() \
                    and active_label.lower() not in exported_names.lower():
                pen_mismatch_note = (f"Exported for pen(s) '{exported_names}' in DrawingBotV3, "
                                      f"but the active ccPlotter pen profile is '{active_label}' "
                                      f"(fill {active.get('fill_mm')}mm) -- if the DBV3 pen's actual "
                                      f"width differs materially, spacing tuned in DBV3's preview won't "
                                      f"match what plots.")
                # If a SAVED pen profile plainly names the same
                # pen DBV3 exported for, offer a one-click switch instead of
                # leaving the note purely informational. Matching is
                # word-token based (not the substring test above) because
                # real DBV3 pen names are terse library entries like
                # "15 Black" -- the color/name token ("black") is the only
                # part likely to also appear in a ccPlotter profile label
                # like "BIC Cristal 1.0 Black"; the leading pen-library
                # number never will. Tokens under 3 chars (pure numbers,
                # stray initials) are dropped to avoid noise matches.
                exported_tokens = {t.lower() for t in re.split(r'[^A-Za-z]+', exported_names) if len(t) >= 3}
                if exported_tokens:
                    all_profiles = load_pen_profiles()
                    active_pid = all_profiles.get("active")
                    for pid, prof in all_profiles.get("profiles", {}).items():
                        if pid == active_pid:
                            continue
                        label_l = str(prof.get("label", "")).lower()
                        if any(tok in label_l for tok in exported_tokens):
                            pen_mismatch_suggested_pen = {"id": pid, "label": prof.get("label")}
                            break
    except Exception:
        dbv3_meta = None

    # Auto-archive on import: move the file OUT of the inbox into
    # DBV3_ARCHIVE_DIR (a subfolder, so the plain *.svg glob both listing
    # routes use never sees it again) so the panel only ever shows drawings
    # NOT YET brought in. Runs ONLY after `dst` (the work/ copy) is
    # confirmed on disk above (atomic_write_bytes already completed by this
    # point), so a failed import never loses the source file. Best-effort:
    # a failure to archive (e.g. permissions) does not fail the import --
    # the file just stays visible for a retry.
    # Configurable off via {"archive": false} in the request body; default ON.
    archived = False
    if bool(body.get("archive", True)):
        try:
            archive_target = DBV3_ARCHIVE_DIR / filename
            if archive_target.exists():
                stem, suffix = Path(filename).stem, Path(filename).suffix
                archive_target = DBV3_ARCHIVE_DIR / f"{stem}_{uuid4().hex[:8]}{suffix}"
            shutil.move(str(src), str(archive_target))
            archived = True
        except OSError:
            archived = False
    # Determinism note: DBV3's own generation isn't ours to pin --
    # record source filename + full sha for replayability via the existing
    # run archive (same mechanism as AI-clean/gcode-build provenance).
    try:
        _run_record = {"type": "dbv3_import", "source_filename": filename, "source_sha256": sha}
        if dbv3_meta:
            _run_record["dbv3_metadata"] = dbv3_meta
        _runs_add(_run_record, copy_files={"cleaned": dst})
    except Exception:
        pass
    resp = {"id": did, "name": f"DBV3: {filename}", "is_raster": False, "src_name": dst.name,
            "source_filename": filename, "source_sha256": sha, "archived": archived}
    if dbv3_meta:
        resp["dbv3_metadata"] = dbv3_meta
    if pen_mismatch_note:
        resp["pen_mismatch_note"] = pen_mismatch_note
    if pen_mismatch_suggested_pen:
        resp["pen_mismatch_suggested_pen"] = pen_mismatch_suggested_pen
    return jsonify(resp)

@app.route("/api/add_text", methods=["POST"])
def api_add_text():
    """Text mode: perfect plotter lettering via vpype's Hershey single-stroke
    fonts. The text becomes a normal vector drawing (src_<id>.svg), so all
    existing layout / scale / rotate / routing / gcode machinery applies."""
    body = request.json or {}
    txt = (body.get("text") or "").strip()
    if not txt: return jsonify({"error": "No text given."}), 400
    if len(txt) > 500: return jsonify({"error": "Text too long (max 500 chars)."}), 400
    font = str(body.get("font", "futural"))
    if not font.replace("_", "").isalnum(): return jsonify({"error": "Invalid font."}), 400
    size_mm = max(2.0, min(float(body.get("size_mm", 10.0)), 100.0))
    size_px = size_mm / PX2MM
    did = uuid4().hex[:8]
    out = WORK_DIR / f"src_{did}.svg"
    try:
        with atomic_target(out) as tmp_out:
            vpype_cli.execute(f'text --font {font} --size {size_px} -- {shlex.quote(txt)} write {shlex.quote(str(tmp_out))}')
    except Exception as e:
        return jsonify({"error": f"Text generation failed: {e}"}), 500
    if not out.exists():
        return jsonify({"error": "Text generation produced no output."}), 500
    label = txt if len(txt) <= 24 else txt[:21] + "..."
    return jsonify({"id": did, "name": f"Text: {label}", "is_raster": False, "src_name": out.name, "is_text": True})

@app.route("/api/clear_drawings", methods=["POST"])
def api_clear_drawings():
    for p in WORK_DIR.glob("src_*"):
        try: p.unlink()
        except Exception: pass
    return jsonify({"ok": True})

@app.route("/api/delete_drawing", methods=["POST"])
def api_delete_drawing():
    did = (request.json or {}).get("id", "")
    if did:
        # src_{did}* (no dot) also sweeps _vector.svg / _render_*.svg caches.
        for p in WORK_DIR.glob(f"src_{did}*"):
            try: p.unlink()
            except Exception: pass
    return jsonify({"ok": True})

_PROCESS_LOCK = threading.Lock()

@app.route("/api/process", methods=["POST"])
def api_process():
    body = request.json or {}; items = body.get("items", [])
    if not items: return jsonify({"error": "No drawings yet."}), 400
    # A registry calibration sheet imported onto the canvas is a POSITION PREVIEW.
    # Canvas gcode of it is re-routed/re-translated and is NOT the registry file, so refuse
    # (previews still allowed). Marker is written by /api/calibration_import (pen.py).
    # NOTE: the marker lives in WORK_DIR, so a cleared work dir fails OPEN (no refusal).
    if body.get("job_type") == "gcode":
        for _it in items:
            if (WORK_DIR / f"calsheetmark_{_it.get('id')}.flag").is_file():
                return jsonify({"error": "This drawing is a calibration-sheet POSITION PREVIEW. "
                                         "Generating G-code from the canvas would re-route and "
                                         "re-translate it; it is NOT the registry file. Plot the "
                                         "downloaded sheet file (calibration_<sheet>_<pen>_z....gcode) "
                                         "instead, or remove the preview from the canvas."}), 400
    # Pen context: pen_id=None (omitted) resolves to the currently ACTIVE pen
    # profile, the same as reading active_pen_profile() directly. An explicit
    # pen_id pins a specific profile regardless of what's active (used by
    # scripts/replay_harness.py so fixtures reproduce under the pen they
    # were recorded with, not whatever is selected in the UI right
    # now); an unknown pen_id fails loudly (UnknownPenProfile -> the
    # existing except-Exception->500 path) rather than silently falling
    # back. See ccplotter/cal.py resolve_pen_profile().
    try:
        pen_profile = resolve_pen_profile(body.get("pen_id"))
    except UnknownPenProfile as e:
        return jsonify({"error": str(e)}), 400
    # Resolved pen_id (provenance-only): the actual id string
    # this request resolved to -- explicit request field if pinned, else
    # whatever is currently ACTIVE. Same pattern already used by
    # routes/pen.py's calibration-plot handler. Threaded to _composite below
    # purely so the tracev3 audit dict can be stamped with it; never affects
    # generation (see composite.py/_source_svg's cache-key isolation note).
    _resolved_pen_id = body.get("pen_id") or load_pen_profiles().get("active")
    # Fill Page (cover-crop): single-item, 0/90 rotation only -- reject anything
    # else with a clean 400 rather than silently degrading.
    _fp_items = [it for it in items if bool(it.get("fill_page"))]
    if _fp_items and len(items) > 1:
        return jsonify({"error": "Fill Page is single-item only in v1 (multiple drawings present)."}), 400
    for _it in _fp_items:
        _fp_rot = float(_it.get("rot", 0.0)) % 360.0
        if min(_fp_rot % 90.0, 90.0 - (_fp_rot % 90.0)) > 1e-6:
            return jsonify({"error": "Fill Page supports 0/90 rotation only in v1."}), 400
    # Sketch Field (native PFM): image->paths algorithm -- vector sources are
    # not valid input. Clean 400, mirroring the Fill Page guard above.
    if body.get("style") == "sketchfield":
        for _it in items:
            _raw = [p for p in WORK_DIR.glob(f"src_{_it.get('id')}.*")
                    if not (p.name.endswith("_traced.svg") or "_render_" in p.name or p.name.endswith("_render.svg"))]
            if _raw and _raw[0].suffix.lower() in VECTOR_EXTS:
                return jsonify({"error": "Sketch Field requires a raster source image "
                                          "(PFM styles trace continuous grayscale, not vector paths)."}), 400
    try: paper_w, paper_h = (PAPER_SIZES[body.get("paper", "b5").lower()] if body.get("paper", "b5").lower() in PAPER_SIZES else (float(body.get("paper", "b5").split('x')[0]), float(body.get("paper", "b5").split('x')[1])))
    except Exception: return jsonify({"error": "Invalid paper size"}), 400
    margin = max(0.0, min(float(body.get("margin", 10)), min(paper_w, paper_h)/2 - 1))

    # Bare-plate registration is deliberately a fixed, non-paper drawing.  It
    # still enters through the ordinary Process Preview -> Generate G-code ->
    # Save UI contract, but must not enter _composite(): that pipeline clips to
    # the paper and applies the Kokuyo no-ink band, both of which would erase
    # the required off-sheet references / hole circles.  The one-item contract
    # also prevents a mixed job from silently omitting ordinary artwork.
    reg_items = [it for it in items if bool(it.get("registration_marks"))]
    if reg_items:
        if len(items) != 1:
            return jsonify({"error": "Registration marks are a one-time bare-plate job. Clear other canvas items before processing them."}), 400
        reg = reg_items[0].get("registration") or {}
        try:
            reg_w, reg_h = float(reg["paper_w"]), float(reg["paper_h"])
            reg_x, reg_y = float(reg["offset_x"]), float(reg["offset_y"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "Registration canvas item is missing its locked placement. Create the marks again."}), 400
        if reg_w <= reg_h:
            return jsonify({"error": "Registration marks require locked landscape paper."}), 400
        t0_registration = time.time()
        if not _PROCESS_LOCK.acquire(blocking=False):
            return jsonify({"error": "busy", "retry": False}), 429
        try:
            job_type = body.get("job_type", "preview")
            gcode_path = WORK_DIR / "layout.gcode"
            info = _registration_marks_gcode(
                gcode_path, paper_w=reg_w, paper_h=reg_h, offset_x=reg_x, offset_y=reg_y,
                pen_mm=float(body.get("pen_width", pen_profile.get("pen_mm", 0.5))),
                z_up=float(body.get("z_up", pen_profile.get("z_up", 3.5))),
                z_down=float(body.get("z_down", pen_profile.get("z_down", -1.0))),
                draw_speed=int(body.get("draw_speed", pen_profile.get("draw_speed", 3000))),
                travel_speed=int(body.get("travel_speed", pen_profile.get("travel_speed", 8000))),
                plunge_speed=int(body.get("plunge_speed", pen_profile.get("plunge_speed", 500))),
                lift_speed=int(body.get("lift_speed", pen_profile.get("lift_speed", 2000))),
                end_policy=str(body.get("end_policy", "release")),
                quiet_fans=bool(body.get("quiet_fans", True)), profile=pen_profile)
            preview_path = WORK_DIR / info["preview"]
            strokes = int(info.get("polylines", 0))
            res = {
                "gcode_file": gcode_path.name, "preview_file": preview_path.name,
                "paper": f"{reg_w}x{reg_h}mm", "item_bboxes": {}, "item_geometry": {},
                "elapsed_s": round(time.time() - t0_registration, 2), "job_type": job_type,
                "warnings": [], "pen_width_mm": float(body.get("pen_width", 0.5)),
                "small_detail": "off", "estimated_pen_lifts": strokes,
                "estimated_z_moves": strokes * 2, "stroke_count": strokes,
                "draw_length_mm": 0.0, "travel_length_mm": 0.0,
                "estimated_plot_time_min": 0.0, "connector_count": 0,
                "pen_lifts": strokes, "z_moves": strokes * 2, "draw_moves": strokes,
                "file_size_kb": round(gcode_path.stat().st_size / 1024, 1) if job_type == "gcode" else 0,
                "job_id": uuid4().hex[:8], "registration_marks": True,
            }
            if job_type == "gcode":
                _snapshot_bundle(body, gcode_path, preview_path, res)
            return jsonify(res)
        except Exception as e:
            return jsonify({"error": f"Registration marks failed: {e}"}), 500
        finally:
            _PROCESS_LOCK.release()
    
    if not _PROCESS_LOCK.acquire(blocking=False):
        return jsonify({"error": "busy", "retry": False}), 429
    
    t0_all = time.time()
    try:
        job_type = body.get("job_type", "preview")
        style = body.get("style", "outline")

        # Photo-512 (opt-in -- config.PHOTO512_RECIPE is the ONE place this
        # recipe is defined). Selecting it in the UI
        # just sets style="photo512"; merge the recipe into body BEFORE the
        # vcache key / render_opts are built below so every subsequent
        # body.get(...) call in this function (including the vcache key)
        # picks up the forced values, then dispatch through the ordinary
        # tracev3 engine. Every other request field (paper, pen, margin,
        # anything not named in the recipe) passes through unchanged. No
        # existing style value's defaults, behavior, or cache key is touched
        # by this branch -- it only fires when style == "photo512".
        _is_photo512 = (style == "photo512")
        if _is_photo512:
            body = {**body, **PHOTO512_RECIPE}
            style = "tracev3"

        # B/W photo (opt-in -- config.BW_PHOTO_RECIPE +
        # ccplotter/bw_photo.py): the XDoG p02_e55 front end (512 px by default), then
        # tracev3 with CLAHE on / tone off. Same pattern as Photo-512 above:
        # style="bwphoto" merges the recipe into body BEFORE the vcache key is
        # built (so the key covers it), then dispatches through the ordinary
        # tracev3 engine. The render_opts key that actually triggers the front
        # end ("front_end") is set below from THIS flag, never from a raw
        # caller-supplied body key. Fires only when style == "bwphoto".
        _is_bwphoto = (style == "bwphoto")
        if _is_bwphoto:
            body = {**body, **BW_PHOTO_RECIPE}
            style = "tracev3"

        # Line Art / Illustration (opt-in -- config.ILLUSTRATION_RECIPE): the
        # reference processing for already-clean line art. Merged into body BEFORE
        # the vcache key, then the ordinary tracev3 engine. NO front end is set for it
        # (render_opts["front_end"] below fires only for _is_bwphoto) and the recipe
        # carries no placement field. Fires only when style == "illustration".
        _is_illustration = (style == "illustration")
        if _is_illustration:
            body = {**body, **ILLUSTRATION_RECIPE}
            style = "tracev3"

        # ── Variant cache: instant restore when this exact (source-set, settings)
        # was already computed for the currently-loaded image. Skipped in debug
        # (debug expects freshly-written intermediates). ──
        _use_vcache = not bool(body.get("debug", False))
        _vfp = _source_fingerprint(items) if _use_vcache else None
        # The body carries only pen_id (or nothing); the server resolves the pen
        # profile / calibration / reach itself, so fold a fingerprint of those
        # RESOLVED inputs into the key -- a stored Z/pen/cal/reach change must
        # miss, never hand back gcode at the old Z (the variant cache would otherwise serve stale profile output).
        _vkey = None
        if _use_vcache:
            try:
                _cal_fp = load_cal()
                _rfp = _resolved_inputs_fingerprint(
                    _resolved_pen_id, pen_profile, _cal_fp, active_reach_profile(_cal_fp))
            except Exception as _e:
                # Unresolvable reach etc.: never share a key with a good state.
                _rfp = "unresolved:" + repr(_e)
            _vkey = _variant_key(body, _rfp)
        if _use_vcache:
            hit = _vcache_get(_vfp, _vkey)
            if hit is not None:
                hit["cached"] = True
                hit["job_id"] = uuid4().hex[:8]
                hit["elapsed_s"] = round(time.time() - t0_all, 2)
                _lazy_preview_register(hit["job_id"], {"clean": WORK_DIR / "layout.clean.svg",
                    "routed": WORK_DIR / "layout.routed.svg", "connectors": WORK_DIR / "layout.connectors.svg",
                    "pen_mm": hit.get("lazy_preview_pen_mm", 0.15), "xray": bool(hit.get("preview_xray_available", False))})
                print(f"  [vcache] HIT mode={style} job={job_type} "
                      f"(restored in {hit['elapsed_s']}s, no recompute)")
                return jsonify(hit)
        # Defaults
        if style == "plotready":
            dm, dl, ds, da, dg, dp = 0.3, 0.3, 0.06, 5, 0, 4
        elif style == "sketchv2":
            # Sketch V2: keep everything (no merge/no min-len cull), light simplify.
            dm, dl, ds, da, dg, dp = 0.0, 0.0, 0.08, 6, 0, 6
        elif style == "lineart":
            dm, dl, ds, da, dg, dp = 0.8, 0.8, 0.12, 40, 0, 8
        elif style == "hybrid":
            dm, dl, ds, da, dg, dp = 0.8, 1.0, 0.15, 40, 0, 8
        elif style in ("outline", "sketch"):
            dm, dl, ds, da, dg, dp = 0.8, 1.5, 0.3, 20, 1, 12
        else:
            dm, dl, ds, da, dg, dp = 0.3, 1.0, 0.1, 20, 1, 12

        # tone_levels_n: forwarded from the request body into render_opts for
        # trace_v3.py's opts.get("tone_levels_n", TONE_LEVELS_N_DEFAULT).
        # Absent from the body -> 7 here too, matching TONE_LEVELS_N_DEFAULT
        # exactly. Garbage/non-numeric input falls back to 7 rather than raising;
        # numeric input is clamped to [1, 32] -- trace_v3's own tone_on gate
        # already treats <=1 as a safe degenerate solid-fill fallback, and 32
        # is a generous, unmeasured ceiling against pathological multiotsu
        # class counts (not a validated upper bound).
        try:
            _tone_levels_n = int(body.get("tone_levels_n", 7))
        except (TypeError, ValueError):
            _tone_levels_n = 7
        _tone_levels_n = max(1, min(32, _tone_levels_n))

        render_opts = {
            "mode": style,
            "resolution": int(body.get("resolution", 900)),
            "clahe": bool(body.get("clahe", False)),
            # Photo-512 only (config.PHOTO512_RECIPE): shorter-side source
            # pre-downsample, in pixels. Gated on the internal _is_photo512 flag
            # set above, NOT on body.get("source_side_cap"), so a caller cannot
            # smuggle this key in under another style and trigger a pre-shrink
            # the request never opted into via style="photo512". None for every
            # other request -- render.py's pre-downsample step is a no-op unless
            # this is truthy.
            "source_side_cap": PHOTO512_RECIPE["source_side_cap"] if _is_photo512 else None,
            # B/W photo only (config.BW_PHOTO_RECIPE): selects render.py's XDoG
            # front end. None for every other request (None values are dropped
            # from the render cache hash, so existing modes' keys do not move).
            "front_end": BW_PHOTO_FRONT_END if _is_bwphoto else None,
            # Line Art / Illustration only: marks the item so composite
            # reports its physical pass pitch (warning) and, when the request carries
            # illustration_pitch_clamp, applies the style-scoped pitch clamp. None for
            # every other request (dropped from the render cache hash).
            "illustration": True if _is_illustration else None,
            "illustration_pitch_clamp": (True if (_is_illustration and bool(body.get("illustration_pitch_clamp", False))) else None),
            "channel": body.get("channel", "auto"),
            "speckle": int(body.get("filter_speckle", 10)),
            "spacing": float(body.get("hatch_spacing", 1.0)),
            "levels": int(body.get("hatch_levels", 2)),
            "canny_low": int(body.get("canny_low", 80)),
            "canny_high": int(body.get("canny_high", 160)),
            "gamma": float(body.get("gamma", 1.0)),
            "isolate_blue": bool(body.get("isolate_blue", False)),
            "contrast": float(body.get("contrast", 1.0)),
            "vec_merge": float(body.get("vec_merge", dm)),
            # Final-G-code point reduction, as a MULTIPLE of the pen's fill_mm
            # (pass pitch) -- see config.SIMPLIFY_FILL_MULT for why pen-relative
            # and for the measured sweep. 0 = OFF (no point reduction).
            "simplify_mult": float(body.get("simplify_mult", SIMPLIFY_FILL_MULT)),
            "vec_min_len": float(body.get("vec_min_len", dl)),
            "outline_simplify": float(body.get("outline_simplify", ds)),
            "lineart_min_area": int(body.get("lineart_min_area", da)),
            "lineart_gap_close": int(body.get("lineart_gap_close", dg)),
            "hole_fill": int(body.get("hole_fill", -1)),   # -1 = auto (bubble fix)
            "prune_len": int(body.get("prune_len", dp)),
            # Sketch V2 (sketchv2 mode only): dust cull + TEMP pixel pen/offset spacing.
            "sketch_min_area": int(body.get("sketch_min_area", 6)),
            "sketch_spacing_px": float(body.get("sketch_spacing_px", 0)),
            "sketch_gain": float(body.get("sketch_gain", 0.12)),
            "sketch_max_passes": int(body.get("sketch_max_passes", 2)),
            "auto_edges": bool(body.get("auto_edges", True)),
            # Threshold controls (raster-stage → in render_opts, so they bust cache).
            "threshold_mode": body.get("threshold_mode", "otsu"),
            "ink_sensitivity": body.get("ink_sensitivity", "sensitive"),
            "manual_threshold":int(body.get("manual_threshold", 128)),
            "adaptive_block": int(body.get("adaptive_block", 25)),
            "adaptive_c": int(body.get("adaptive_c", 7)),
            # Hybrid mode (raster-stage). Weights default 'auto'.
            "hybrid_preset": body.get("hybrid_preset", "balanced"),
            "geometry_weight": body.get("geometry_weight", "auto"),
            "edge_weight": body.get("edge_weight", "auto"),
            "ink_weight": body.get("ink_weight", "auto"),
            # Small-detail preservation (raster-stage): off | balanced | preserve.
            "small_detail": body.get("small_detail", "off"),
            # Micro-detail / tiny-text protection (raster-stage): off | balanced |
            # aggressive. DETAIL-FIRST — keeps fragile glyph/rivet/texture spurs from
            # being pruned; default balanced protects ALL small marks.
            "preserve_text": body.get("preserve_text", "balanced"),
            # Speckle cleanup is SEPARATE and OPT-IN: off | light | strong. Default off
            # so detail is never silently removed (light = drop lone specks only).
            "noise_filter": body.get("noise_filter", "off"),
            # Small-Detail Fidelity: on | off. Per small ink component, simulate skeleton vs
            # local super-res contour and keep whichever renders closer to the
            # component's own ink shape at pen width -- no region classification.
            # On by default; off = byte-identical baseline.
            "small_detail_fidelity": "on" if bool(body.get("small_detail_fidelity", True)) else "off",
            # Isolation gate for the fidelity trial above: strict | trial.
            # 'strict' (default) keeps the proximity pre-filter that skips small
            # components with a neighbor within ~1 pen width. 'trial' removes
            # that pre-filter so every area-gate-passing component enters the
            # fidelity trial directly.
            "fidelity_isolation": body.get("fidelity_isolation", "strict"),
            # Trace V3 coverage: real solid-fill line width (mm) from the active pen
            # profile + paper, so pocket passes are spaced to the pen, not nominal.
            "fill_mm": float(pen_profile.get("fill_mm", 0.15)),
            # MEASURED laid-line width, if this pen has ever been measured.
            # Distinct from fill_mm (pass SPACING) and from pen_mm (a nominal
            # barrel figure, wrong by 2x on uright_gel). Absent on every profile
            # that has not been measured, and absent means "behave exactly as
            # before" -- see composite.py's use of it.
            "laid_mm": (float(pen_profile["laid_mm"])
                        if pen_profile.get("laid_mm") is not None else None),
            "paper_w_mm": paper_w, "margin_mm": margin,
            # Keep faint marks: the calibration card measured a solid floor of
            # 0.4mm, but culling at that floor would drop faint detail, so the
            # cull threshold is much smaller.
            "min_mark_mm2": float(body.get("min_mark_mm2", 0.035)),
            "pocket_link": float(body.get("pocket_link", 3.0)),
            # Pocket closing: force (default) | guarded. See ccplotter/trace_v3.py
            # _link_pockets docstring. Only read by _trace_svg (mode ==
            # "tracev3"); other modes never consult this key.
            "pocket_close": body.get("pocket_close", "force"),
            # Trace V3 tonal fill knob: off (solid, byte-identical) |
            # faithful | balanced | dense. Splits ink into dark/mid/light via
            # multiotsu and fills each at its own pass spacing (tone via geometry).
            "tone": body.get("tone", "off"),
            "tone_levels_n": _tone_levels_n,
            "hole_keep": float(body.get("hole_keep", 0.08)),
            # Widen physics-marginal gaps (0.5-1.0x deposit) to one deposit
            # so hatched shadows keep their light. Geometry edit, OFF by default.
            "gap_guarantee": body.get("gap_guarantee", False),
            # Texture routing: off | centerline_v2 | band_continuity_v1.
            # Elongated-thin zero_labels components route through
            # skeletonize -> polylines -> smoothing; predicate-passing T_new
            # appendage regions are claimed the same way. Compact/closed-loop
            # components (rivets, gravel) are routed unchanged.
            # The default band_continuity_v1 pairs with the fill_mm 0.25 seed:
            # it recovers the band fragmentation a bare fill_mm change would
            # cause. Only read by _trace_svg (mode == "tracev3"); other modes
            # never consult this key. Explicit request values override.
            "texture_routing": body.get("texture_routing", "band_continuity_v1"),
            # Opt-in ink-recovery flag: default "v1" runs the speck-drop cull.
            # "no_speck_drop" skips ONLY the tiny-closed-loop cull in
            # _trace_svg, the one stage that actually discards geometry.
            "fill_routing": body.get("fill_routing", "v1"),
            # Mask options, OFF by default: native_res thresholds at native
            # resolution (only helps DOWNSCALE sources); pre_blur=0 disables the
            # pre-blur. Default pre_blur=3 is a (3,3) Gaussian.
            "native_res": bool(body.get("native_res", False)),
            "pre_blur": int(body.get("pre_blur", 3)),
            "md_radius_factor": float(body.get("md_radius_factor", 2.0)),
            "md_max_support": int(body.get("md_max_support", 14)),
            # Experimental AI texture sidecar layer merged at trace time (in the
            # cache hash so toggling re-merges instantly). DEFAULT OFF: a leftover
            # sidecar must never silently contaminate a clean run.
            "texture_layer": bool(body.get("texture_layer", False)),
            # Temporary hard-debug: force one ROI to prove the detail pipeline.
            "force_detail_roi": bool(body.get("force_detail_roi", False)),
        }
        # Sketch Field (native "Sketch Lines" PFM, style="sketchfield" only;
        # every other mode ignores these keys -- see ccplotter/pfm_native.py).
        # Parameter surface: density/line-length budget, angle policy (delta
        # cone), squiggle chain length, hard segment cap, and the seed
        # (default = content-hash of the source bytes; this key overrides it
        # explicitly). Nested under the style=="sketchfield" check (not
        # always present) so a non-sketchfield request's render_opts/opt_hash
        # is unchanged -- same cache-key discipline as the scale_aware /
        # scale_aware_v2 blocks below.
        if body.get("style") == "sketchfield":
            # Defaults favour short-to-medium strokes, a tight Drawing Delta
            # Angle (coherent squiggles), a high Line Density (photo tone
            # accuracy) and darkness floors that stop a chain before it
            # crosses into bright/empty areas. Long strokes in a wide angle
            # cone wander out of dark regions into empty space and exhaust
            # density/max_lines before covering the image.
            render_opts.update({
                "sketch_field_density": float(body.get("sketch_field_density", 0.75)),
                "sketch_field_min_len_mm": float(body.get("sketch_field_min_len_mm", 1.0)),
                "sketch_field_max_len_mm": float(body.get("sketch_field_max_len_mm", 4.0)),
                "sketch_field_angle_tests": int(body.get("sketch_field_angle_tests", 12)),
                "sketch_field_angle_delta_deg": float(body.get("sketch_field_angle_delta_deg", 25.0)),
                "sketch_field_squiggle_max_mm": float(body.get("sketch_field_squiggle_max_mm", 8.0)),
                "sketch_field_max_lines": int(body.get("sketch_field_max_lines", 20000)),
                "sketch_field_erase_amount": float(body.get("sketch_field_erase_amount", 45.0)),
                "sketch_field_seed_floor": float(body.get("sketch_field_seed_floor", 40.0)),
                "sketch_field_min_avg_darkness": float(body.get("sketch_field_min_avg_darkness", 18.0)),
                "sketch_field_seed": body.get("sketch_field_seed", None),
            })
        # Scale adaptation v1 (opt-in). OFF (default) leaves the _source_svg
        # cache key unchanged: the key is only added when true, so an off run's
        # render_opts hash does not depend on this feature. ON: composite.py computes each item's REAL placed size
        # (post fit/scale) analytically and feeds fill_px from that instead of
        # the declared paper_w_mm/margin_mm, so pass spacing and the four other
        # derived floors in ccplotter/trace_v3.py track the actual plotted mm,
        # not the paper the drawing may only partially cover. tracev3 mode only.
        if bool(body.get("scale_aware", False)):
            render_opts["scale_aware"] = True
        # Scale adaptation v2: the DEFAULT for tracev3 requests. A tracev3
        # request gets scale_aware_v2=True unless it explicitly opts out
        # (scale_aware_v2=false). Legacy (scale_aware_v2=false,
        # scale_aware=false) and v1 (scale_aware=true, scale_aware_v2=false)
        # remain reachable as explicit, non-default choices. Non-tracev3
        # requests are UNAFFECTED -- the key is only ever added when
        # style == "tracev3", so every other mode's render_opts/opt_hash is
        # unchanged (composite.py also independently gates the v2 branch on
        # item_opts.get("mode") == "tracev3", but keeping the key itself out
        # of render_opts for non-tracev3 modes preserves their cache key too).
        if style == "tracev3":
            if bool(body.get("scale_aware_v2", True)):
                render_opts["scale_aware_v2"] = True
        elif bool(body.get("scale_aware_v2", False)):
            render_opts["scale_aware_v2"] = True
        # Fill mode (opt-in zigzag). Same cache-key discipline as scale_aware
        # just above: add the key ONLY when a caller actually asks for zigzag,
        # so an off-run's render_opts/opt_hash does not change (unlike
        # fill_routing above, which is added unconditionally with a "v1"
        # default). tracev3-scoped: other modes never consult
        # fill_mode/fill_angle_deg.
        if style == "tracev3" and str(body.get("fill_mode", "contour")).lower() == "zigzag":
            render_opts["fill_mode"] = "zigzag"
            try:
                render_opts["fill_angle_deg"] = float(body.get("fill_angle_deg", 0.0))
            except (TypeError, ValueError):
                render_opts["fill_angle_deg"] = 0.0
            # Per-region angle selection (opt-in). Same cache-key discipline as
            # fill_mode above: the key is added ONLY when a caller explicitly
            # asks for it, so a request that never mentions fill_angle_mode
            # keeps the same render_opts/opt_hash.
            if str(body.get("fill_angle_mode", "fixed")).lower() == "per_region":
                render_opts["fill_angle_mode"] = "per_region"
            # Connectors between zigzag rows: fill_connect is "boustrophedon"
            # (default), "guarded", or "none". trace_v3.py raises loudly if
            # "guarded" is combined with fill_angle_mode="per_region" (not
            # implemented). The template's "Zigzag fill" checkbox sends ONLY
            # fill_mode/fill_angle_deg, never fill_connect, so a zigzag request
            # that omits fill_connect resolves to "boustrophedon" plus the
            # proximity / split-merge / endlink switches below, not to
            # unlinked rows.
            # EXPLICIT request values still win: a caller that asks for
            # fill_connect="none" (perimeter passes, no connectors) skips the
            # whole `if` below and is not promoted.
            _fill_connect_val = str(
                body.get("fill_connect", "boustrophedon")).lower()
            if _fill_connect_val in ("guarded", "boustrophedon"):
                render_opts["fill_connect"] = _fill_connect_val
                if "fill_connect_max_gap_mm" in body:
                    try:
                        render_opts["fill_connect_max_gap_mm"] = float(
                            body["fill_connect_max_gap_mm"])
                    except (TypeError, ValueError):
                        pass
                # Proximity / split-merge / chain-end-link continuation. They
                # are read straight off `opts` in trace_v3.py and MUST enter
                # this route's cache key the same way fill_connect's own value
                # does, or a cached render for one config could be served for
                # the other. Each is a plain key on `render_opts`, which
                # composite.py hashes (via `clean_opts`/`opt_hash`) into the
                # cache filename, like every other key in this function.
                # Default True (and max_drow=1) ONLY as a
                # fallback value inside `body.get(...)`, so an explicit
                # False/0 from a caller still wins; trace_v3.py itself also
                # forces all three off whenever fill_connect != "boustrophedon"
                # (`_connect_boustrophedon` gate), so a "guarded" request is
                # unaffected by these defaults even though the keys are
                # still added here.
                render_opts["fill_zigzag_proximity"] = bool(
                    body.get("fill_zigzag_proximity", True))
                render_opts["fill_zigzag_split_merge"] = bool(
                    body.get("fill_zigzag_split_merge", True))
                render_opts["fill_zigzag_endlink"] = bool(
                    body.get("fill_zigzag_endlink", True))
                try:
                    render_opts["fill_zigzag_endlink_max_drow"] = int(
                        body.get("fill_zigzag_endlink_max_drow", 1))
                except (TypeError, ValueError):
                    render_opts["fill_zigzag_endlink_max_drow"] = 1
        # Mask gap-close (opt-in). The knob is a raw kernel diameter in
        # TRACE-FRAME PIXELS, not mm: an mm value would be misleading for
        # laid_mm pens (see the `_gap_close_active` gate in trace_v3.py), and
        # any mm/pen-width UI for it would have to convert through the true
        # mm/px. Same cache-key discipline as scale_aware/fill_mode above: add
        # the key ONLY when a caller supplies a value >1px, so an off-run's
        # render_opts/opt_hash is unchanged. The key reaches trace_v3.py's
        # opts the same way fill_mode/fill_connect do (render_opts ->
        # clean_opts -> opt_hash in composite.py), so two requests differing
        # only in this value get different cache entries. tracev3-scoped: other modes never consult it. Forwarded
        # here regardless of fill_mode/tone -- trace_v3.py itself is the
        # sole place that decides which fill paths honor it (contour only,
        # not tone/zigzag -- see _mask_gap_close's call site and the
        # `_gap_close_active` gate in trace_v3.py); gating twice would only
        # risk the two gates drifting apart.
        if style == "tracev3":
            try:
                _gap_close_px_val = int(body.get("mask_gap_close_px", 0) or 0)
            except (TypeError, ValueError):
                _gap_close_px_val = 0
            if _gap_close_px_val > 1:
                render_opts["mask_gap_close_px"] = _gap_close_px_val
            # core_fill / edge_inset_mm (opt-in, all default OFF):
            # same add-only-when-active cache-key discipline. Contour
            # fill only -- trace_v3.py ignores both under fill_mode=zigzag/tone.
            # core_fill="hybrid": rings stop at inscribed radius < core_r_mm and
            # the remaining core gets straight passes. edge_inset_mm: first ring
            # this far inside the shape edge instead of 0.5 footprint.
            if str(body.get("core_fill", "rings")).lower() == "hybrid":
                render_opts["core_fill"] = "hybrid"
                try:
                    _core_r = float(body.get("core_r_mm") or CORE_R_MM_DEFAULT)
                except (TypeError, ValueError):
                    _core_r = CORE_R_MM_DEFAULT
                render_opts["core_r_mm"] = _core_r if _core_r > 0 else CORE_R_MM_DEFAULT
            elif str(body.get("core_fill", "rings")).lower() == "rows":
                # core_fill="rows" (opt-in): rings, then rows in every large region's core
                render_opts["core_fill"] = "rows"
                try:
                    _core_r = float(body.get("core_r_mm") or ROWS_CORE_R_MM_DEFAULT)
                except (TypeError, ValueError):
                    _core_r = ROWS_CORE_R_MM_DEFAULT
                render_opts["core_r_mm"] = _core_r if _core_r > 0 else ROWS_CORE_R_MM_DEFAULT
            elif str(body.get("core_fill", "rings")).lower() == "outline_rows":
                # core_fill="outline_rows" (opt-in): one outline ring, then rows over the whole interior
                render_opts["core_fill"] = "outline_rows"
            try:
                _inset_val = float(body.get("edge_inset_mm") or 0.0)
            except (TypeError, ValueError):
                _inset_val = 0.0
            if 0.0 < _inset_val <= 2.0:
                render_opts["edge_inset_mm"] = _inset_val
            if render_opts.get("core_fill") == "outline_rows" and render_opts.get("edge_inset_mm"):
                # outline_rows draws its ring at 0.5*fill_px and cannot honour an inset: refuse, never record one
                return jsonify({"error": "core_fill=outline_rows cannot be combined with edge_inset_mm > 0 "
                                         "(the outline ring is fixed at half a pen from the edge). "
                                         "Set edge_inset_mm to 0 or choose another core_fill."}), 400
            # stitch_mm (opt-in, default OFF): join open stroke ends within this
            # many mm when the whole connector lies inside the ink mask eroded by half a pen.
            # Same add-only-when-active cache-key discipline; absent / 0 / out of range
            # adds nothing, so opt_hash and bytes are untouched.
            try:
                _stitch_val = float(body.get("stitch_mm") or 0.0)
            except (TypeError, ValueError):
                _stitch_val = 0.0
            if 0.0 < _stitch_val <= STITCH_MM_MAX:
                render_opts["stitch_mm"] = _stitch_val
        # bwphoto-only mask overrides (opt-in). Read ONLY when the request
        # resolved to bwphoto above; BW_PHOTO_RECIPE defaults are untouched and
        # the recipe merge already forced everything else. Add-only-when-active.
        if _is_bwphoto:
            _bw_clahe = str(body.get("bw_clahe", "on")).lower()
            if _bw_clahe == "off":
                render_opts["clahe"] = False
                render_opts["bw_clahe"] = "off"
            elif _bw_clahe == "soft":
                render_opts["bw_clahe"] = "soft"
                render_opts["bw_clahe_clip"] = BW_CLAHE_SOFT_CLIP
            if str(body.get("bw_cut", "otsu")).lower() == "mid":
                render_opts["bw_cut"] = "mid"
            # bw_xdog_side: XDoG working-raster short side; default "native". The RESOLVED side is always
            # put in render_opts (explicit 512 included) so an explicit 512 and the native default
            # get different opt_hash / cache keys. Invalid -> clean 400.
            try:
                from ccplotter.bw_photo import parse_bw_xdog_side
                _bw_side = parse_bw_xdog_side(body.get("bw_xdog_side"))
            except ValueError as _e:
                return jsonify({"error": str(_e)}), 400
            render_opts["bw_xdog_side"] = _bw_side
        # Routing params are LAYOUT-stage, not raster-stage — keep them OUT of
        # render_opts so changing them reuses the cached vectorization.
        connector_distance = float(body.get("connector_distance", body.get("join_nearby", 0.0)))
        target_z_moves = int(body.get("target_z_moves", 0)) or None
        microlift = bool(body.get("microlift", True))
        debug = bool(body.get("debug", False))
        # connector_mask_check (opt-in, default OFF): Stage E fuses only where the
        # whole connector lies inside the ink mask (stitch predicate). LAYOUT-stage, so it stays out
        # of render_opts; composite asks trace_v3 for a mask sidecar under its own private key.
        # Only strict truthy values enable it; absent / false / junk passes nothing to _composite.
        connector_mask_check = body.get("connector_mask_check") in (True, 1, "1", "true", "True")
        # stay_down_over_ink (opt-in "Stay down over ink (no stray lines)", default OFF): after routing, a
        # pen-up travel becomes a pen-down connector of ANY length iff the straight connector stays inside the
        # layer's own ink mask (the stage_e_mask predicate). LAYOUT-stage, so out of render_opts. Same
        # strict-truthy parse as connector_mask_check; absent / false / junk passes nothing to _composite.
        stay_down_over_ink = body.get("stay_down_over_ink") in (True, 1, "1", "true", "True")
        # prelim_res_basis (experimental, no UI, default OFF): corrects the v2
        # resolution boost's base resolution (composite.py). Honoured ONLY when this request resolved to
        # scale_aware_v2; absent / "requested" passes nothing to _composite.
        # Layout-stage kwarg, deliberately NOT in render_opts, so no trace cache key moves.
        _prb = body.get("prelim_res_basis")
        if _prb not in (None, "", "requested", "prelim_effective", "final_solve"):
            return jsonify({"error": f"prelim_res_basis must be requested / prelim_effective / final_solve, got {_prb!r}"}), 400
        _prb_kw = ({"prelim_res_basis": _prb}
                   if (_prb in ("prelim_effective", "final_solve") and render_opts.get("scale_aware_v2")) else {})
        pen_width = float(body.get("pen_width", 0.5))
        reach_default = active_reach_profile(load_cal())
        gcode_path = WORK_DIR / "layout.gcode"
        try:
            preview_path, bboxes, stats = _composite(items, paper_w, paper_h, margin, gcode_path, float(body.get("z_up", 3.5)), float(body.get("z_down", -0.7)), int(body.get("draw_speed", 3000)), int(body.get("travel_speed", 8000)), int(body.get("plunge_speed", 500)), int(body.get("lift_speed", 2000)), float(body.get("offset_x", reach_default["bbox"][0])), float(body.get("offset_y", reach_default["bbox"][1])), render_opts, float(body.get("paper_rot", 0.0)), generate_gcode=(job_type=='gcode'), connector_distance=connector_distance, target_z_moves=target_z_moves, microlift=microlift, debug=debug, end_policy=str(body.get("end_policy", "release")), quiet_fans=bool(body.get("quiet_fans", True)), motion_smoothing=body.get("motion_smoothing", "off"), pen_profile=pen_profile, pen_id=_resolved_pen_id, paper_keepout=body.get("paper_keepout"), preview_reach_check=bool(body.get("registered_paper", False)), **({"connector_mask_check": True} if connector_mask_check else {}), **({"stay_down_over_ink": True} if stay_down_over_ink else {}), **_prb_kw)
        except Exception as e: return jsonify({"error": f"Pipeline failed: {e}"}), 500

        # Surface settings the warning logic needs (plottability, small-detail note).
        stats["pen_width_mm"] = pen_width
        stats["small_detail"] = render_opts["small_detail"]
        _lazy_sources = stats.pop("_lazy_preview_sources")
        _trace_boundary = stats.pop("_trace_boundary", None)   # provenance sidecar only, never the response
        warnings = build_warnings(stats)
        res = {
            "gcode_file": gcode_path.name, "preview_file": preview_path.name,
            "paper": f"{paper_w}x{paper_h}mm", "item_bboxes": bboxes,
            "item_geometry": stats.get("item_geometry", {}),
            "elapsed_s": round(time.time() - t0_all, 2),
            "job_type": job_type, "warnings": warnings,
            # Exact true-width preview input (fill_mm), retained for cache-hit
            # lazy manifests; nominal pen_width_mm is intentionally not used.
            "lazy_preview_pen_mm": _lazy_sources["pen_mm"],
            # vector-derived plotter stats (available at PREVIEW time, no gcode needed)
            **stats,
            # legacy keys the UI already reads (mapped to the vector estimates)
            "pen_lifts": stats["estimated_pen_lifts"],
            "z_moves": stats["estimated_z_moves"],
            "draw_moves": stats["stroke_count"],
            "file_size_kb": 0,
        }
        # Advisory-only paper-coverage prediction (coverage_audit.py). Runs off
        # the SAME preview geometry the response already reports (preview_path),
        # in paper mm via PX2MM -- it reads the finished SVG, adds no geometry,
        # changes no stroke. None on any pen profile without a MEASURED laid_mm
        # (unmeasured pens are unaffected, per the module's own contract); any
        # failure degrades to {"error": ...} so a preview never fails because
        # this advisory did. See ccplotter/coverage_audit.py docstring.
        res["coverage_audit"] = None
        _laid_mm = pen_profile.get("laid_mm") if pen_profile else None
        if _laid_mm is not None:
            try:
                _cov_t0 = time.time()
                _cov_doc = vp.read_multilayer_svg(str(preview_path), quantization=0.1)
                _cov_doc.layers.pop(XRAY_LAYER, None)
                _cov_strokes_mm = [[(p.real * PX2MM, p.imag * PX2MM) for p in _line]
                                    for _lc in _cov_doc.layers.values() for _line in _lc]
                # fill_mode is read straight off render_opts (only
                # ever "zigzag" when the zigzag block above set it -- absent/None
                # for every other request, so this kwarg is a no-op there and the
                # audit dict is the same on contour/plotready/lineart).
                res["coverage_audit"] = _coverage_audit_mod.audit(
                    _cov_strokes_mm, float(_laid_mm),
                    fill_mode=render_opts.get("fill_mode"))
                print(f"  [coverage_audit] {time.time() - _cov_t0:.2f}s")
            except Exception as e:
                res["coverage_audit"] = {"error": str(e)}

        # Surface the audit's own notes in the warnings bar. The audit reports
        # the fraction of intended fill that is white in the geometry (small
        # interior holes that are visible at pen scale but smear to solid at
        # the fit-to-window preview zoom), and the UI does not otherwise show
        # the `coverage_audit` field.
        #
        # SECONDARY tier deliberately: the notice fires on most real jobs, and a
        # notice that shouts every time gets ignored. Secondary collapses behind
        # the "N more notices" expander. `warnings` is the same list object
        # already stored as res["warnings"] above, so appending here reaches the
        # response.
        _cov = res.get("coverage_audit")
        if isinstance(_cov, dict):
            for _note in (_cov.get("notes") or []):
                warnings.append({"msg": _note, "level": "warning",
                                 "tier": "secondary"})

        # Only a real G-code job reports the final on-disk file size.
        if job_type == 'gcode' and gcode_path.exists():
            res["file_size_kb"] = round(gcode_path.stat().st_size / 1024, 1)
            # Rows-provenance gap: for core_fill rows/hybrid the backend default core_r_mm is applied
            # to render_opts AFTER the body is recorded, so record the RESOLVED value in a copy of
            # the post-recipe body (provenance only; `body` itself and the gcode are untouched).
            _prov_eff_body = body
            try:
                if render_opts.get("core_fill") in ("rows", "hybrid") and "core_r_mm" in render_opts:
                    _prov_eff_body = dict(body)
                    _prov_eff_body["core_r_mm"] = render_opts["core_r_mm"]
                elif render_opts.get("core_fill") == "outline_rows":
                    from ccplotter.core_fill import OUTLINE_ROWS_RINGS
                    _prov_eff_body = dict(body)
                    _prov_eff_body["core_fill"] = "outline_rows"
                    _prov_eff_body["core_outline_rings"] = OUTLINE_ROWS_RINGS
            except Exception:
                _prov_eff_body = body
            # Archive gcode builds (layout.gcode gets overwritten every run).
            try:
                _runs_add({"type": "gcode", "mode": render_opts.get("mode"),
                           "strokes_before": stats.get("strokes_before"), "strokes_after": stats.get("strokes_after"),
                           "z_moves_before": stats.get("z_moves_before"), "z_moves_after": stats.get("z_moves_after"),
                           "connector_count": stats.get("connector_count"),
                           "draw_length_mm": stats.get("draw_length_mm"), "travel_length_mm": stats.get("travel_length_mm"),
                           "estimated_plot_time_min": stats.get("estimated_plot_time_min"),
                           "connector_distance_mm": stats.get("connector_distance_mm"),
                           "trace_info": stats.get("trace_info"),
                           "source_ids": [str(i.get("id")) for i in items]},
                          copy_files={"gcode": gcode_path, "preview": preview_path,
                                      "cleaned": (WORK_DIR / f"src_{items[0]['id']}.png") if items else ""},
                          # Provenance: request as received (request.get_json is cached and
                          # `body` is only ever re-bound by the recipe merge, never mutated)
                          # + the post-recipe body + resolved pen profile + cal sha + app sha.
                          provenance=build_run_provenance(request.get_json(silent=True), _prov_eff_body,
                                                          _resolved_pen_id, pen_profile,
                                                          resolved_from_stats(stats, reach_default.get("id")),
                                                          trace_boundary=_trace_boundary),
                          stamp_gcode=True)   # adds one "; ccplotter-run:" header line to layout.gcode
                # NB: res["file_size_kb"] is deliberately NOT refreshed after the header
                # stamp: the response is unchanged (the on-disk file is ~0.1 KB larger).
            except Exception as e:
                print(f"  [runs] gcode archive failed: {e}")

        # job_id: token the inspector appends to debug URLs so the browser always
        # fetches the current run (debug files keep stable names → cache stays warm).
        res["job_id"] = uuid4().hex[:8]
        _lazy_preview_register(res["job_id"], {**_lazy_sources,
                                                "xray": bool(res.get("preview_xray_available", False))})
        if job_type == 'gcode' and gcode_path.exists():
            _snapshot_bundle(body, gcode_path, preview_path, res)
        if render_opts.get("mode") == "sketchv2":
            res["sketch_audit"] = dict(_sketch_mod._SKETCH_AUDIT)   # gain/thick/centerline/offset counts
        h = stats.get("hybrid", {})
        res["debug_metrics"] = {
            "candidate_counts": h.get("candidate_counts", {}),
            "accepted_counts": h.get("accepted_counts", {}),
            "rejected_counts": h.get("rejected_counts", {}),
            "line_length_stats_px": h.get("line_length_stats_px", {}),
            "line_length_stats_mm": stats.get("line_length_stats_mm", {}),
            "weights": h.get("weights", {}),
            "objective_score": h.get("objective_score"),
            # Rich small-detail lane block when Hybrid ran it; else just the mode.
            "small_detail": h.get("small_detail", render_opts["small_detail"]),
            "pen_width_mm": pen_width,
            "min_stroke_mm": stats.get("min_stroke_mm", 0),
            "lineart_quality": stats.get("lineart_quality", {}),
            "fused_lines": h.get("final_strokes_pre_vpype"),
            "clean_lines": stats.get("strokes_before", stats.get("stroke_count")),
            "routed_lines": stats.get("strokes_after", stats.get("stroke_count")),
        }

        routing_on = (connector_distance and connector_distance > 0) or target_z_moves
        common = (f"draw={stats['draw_length_mm']}mm travel={stats['travel_length_mm']}mm "
                  f"time={stats['estimated_plot_time_min']}min warnings={len(warnings)} elapsed={res['elapsed_s']}s")
        if routing_on:
            print(f"  [job] {job_type} mode={style} "
                  f"strokes={stats.get('strokes_before','?')}->{stats['stroke_count']} "
                  f"z_moves={stats.get('z_moves_before','?')}->{stats['estimated_z_moves']} "
                  f"connectors={stats.get('connector_count',0)} "
                  f"(+{stats.get('connector_draw_length_mm',0)}mm, longest={stats.get('longest_connector_mm',0)}mm) {common}")
        else:
            print(f"  [job] {job_type} mode={style} routing=off "
                  f"strokes={stats['stroke_count']} z_moves={stats['estimated_z_moves']} {common}")
        # Snapshot this fresh result into the current-image variant cache so an
        # A/B flip back to it is an instant restore. Cache the response WITHOUT
        # the per-call job_id (regenerated on every hit).
        if _use_vcache:
            res["cached"] = False
            _store = {k: v for k, v in res.items() if k != "job_id"}
            try: _vcache_put(_vfp, _vkey, _store, label=_variant_label(body))
            except Exception as e: print(f"  [vcache] store failed: {e}")
        return jsonify(res)
    finally:
        _PROCESS_LOCK.release()

@app.route("/api/lazy_preview/<job_id>/<kind>", methods=["POST"])
def api_lazy_preview(job_id, kind):
    """Render one immutable, server-owned optional preview for a job token."""
    if not _PROCESS_LOCK.acquire(blocking=False):
        return jsonify({"error": "busy", "retry": False}), 429
    try:
        try:
            out = _lazy_preview_render(str(job_id), str(kind))
        except KeyError:
            return jsonify({"error": "Unknown or expired preview job."}), 404
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"ok": True, "job_id": job_id, "kind": kind,
                        "preview_file": str(out.relative_to(WORK_DIR))})
    finally:
        _PROCESS_LOCK.release()

@app.route("/work/<path:filename>")
def serve_work(filename): return send_from_directory(WORK_DIR, filename)

@app.route("/download_bundle")
def download_bundle():
    """Zip of the LAST completed (job_type='gcode') generation: the preview SVG
    byte-for-byte as served to the UI, the matching gcode, the exact request
    body that produced it (doubles as a fixture input), and the info/stats
    the pipeline returned. Snapshotted at generation time (see
    _snapshot_bundle) so it never drifts from work/layout.* being overwritten
    by a later run."""
    with _BUNDLE_LOCK:
        snap = _bundle_mod._LAST_BUNDLE
    if snap is None:
        return jsonify({"error": "No G-code has been generated yet in this session. "
                                  "Click Generate G-code, then try again."}), 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(snap["preview_name"], snap["preview_svg"])
        zf.writestr(snap["gcode_name"], snap["gcode"])
        zf.writestr("request.json", snap["request_json"])
        zf.writestr("info.json", snap["info_json"])
        zf.writestr("manifest.json", json.dumps({
            "created_at": snap["created_at"],
            "job_id": snap["job_id"],
            "preview_file": snap["preview_name"],
            "gcode_file": snap["gcode_name"],
        }, indent=1))
    buf.seek(0)
    # Filename is timestamp-based so bundles can be told apart by when they
    # were made, at a glance in the Downloads folder; job_id stays in
    # manifest.json above for exact traceability back to a run.
    fname = f"ccplotter_bundle_{time.strftime('%Y-%m-%d_%H%M%S')}.zip"
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=fname)


@app.route("/api/send", methods=["POST"])
def api_send():
    if (_unc := _printer_unconfigured()) is not None: return _unc
    body = request.json or {}; gfile = body.get("gcode_file")
    if not gfile: return jsonify({"error": "No gcode_file"}), 400
    gpath = WORK_DIR / gfile
    if not gpath.exists(): return jsonify({"error": "GCode not found"}), 404
    if not printer.connected: return jsonify({"error": "Printer not connected"}), 503
    ok, msg = printer.upload_file(gpath)
    if not ok: return jsonify({"success": False, "message": msg})
    if not printer.start_print(gfile): return jsonify({"error": "Start command failed"}), 500
    time.sleep(1.0); printer.refresh(); return jsonify({"success": True, "message": f"Sent {gfile} to printer."})

@app.route("/api/gcode", methods=["POST"])
def api_gcode():
    # Manual single-command route (only reachable when a printer host is
    # configured). The filter is a plain prefix whitelist: a command is sent
    # if it starts with G92, G28, "G1 Z", "G0 Z" or M84; anything else gets
    # 403. It is NOT a safety boundary: "G1 Z..." / "G0 Z..." may also carry
    # X/Y words, G28 homes all axes (with the pen attached), and G92 resets
    # the coordinate frame. Nothing sent here is checked against the reach
    # calibration that generated G-code files go through.
    body = request.json or {}; cmd = body.get("cmd", "").strip()
    if not cmd or not any(cmd.upper().startswith(p) for p in ("G92", "G28", "G1 Z", "G0 Z", "M84")): return jsonify({"ok": False, "error": "Invalid command"}), 403
    if (_unc := _printer_unconfigured()) is not None: return _unc
    if not printer.connected or not printer._ws: return jsonify({"ok": False, "error": "Printer not connected"}), 503
    import queue; q = queue.Queue(); orig = printer._on_message
    def _wrap(ws, raw):
        orig(ws, raw)
        try:
            a = json.loads(raw).get("Data", {}).get("Data", {}).get("Ack")
            if a is not None: q.put(a)
        except: pass
    printer._ws.on_message = _wrap
    for code in [160, 161, 162, 163, 164, 224, 225, 256, 288, 320]:
        for fld in ["Gcode", "GCode", "gcode", "Command"]:
            try:
                printer._ws.send(json.dumps({"Id": "", "Data": {"Cmd": code, "Data": {fld: cmd}, "RequestID": "", "MainboardID": "", "TimeStamp": int(time.time()*1000), "From": 1}}))
                try:
                    if q.get(timeout=0.4) == 0: printer._ws.on_message = orig; return jsonify({"ok": True})
                except: pass
            except: pass
    printer._ws.on_message = orig; return jsonify({"ok": False, "error": "Not acknowledged"})

@app.route("/api/fans_off", methods=["POST"])
def api_fans_off():
    """Turn off case/auxiliary fans via standalone command injection (NOT a print
    job — so no firmware start macro / homing risk). Sends each fan M-code over
    the same websocket path used for Z-jogging, brute-forcing Cmd code + field
    name until one is acknowledged (firmware command channel is undocumented)."""
    if (_unc := _printer_unconfigured()) is not None: return _unc
    if not printer.connected or not printer._ws:
        return jsonify({"ok": False, "error": "Printer not connected"}), 503
    import queue
    orig = printer._on_message
    acked = []
    for cmd in FAN_OFF_GCODE:
        q = queue.Queue()
        def _wrap(ws, raw, _q=q):
            orig(ws, raw)
            try:
                a = json.loads(raw).get("Data", {}).get("Data", {}).get("Ack")
                if a is not None: _q.put(a)
            except Exception: pass
        printer._ws.on_message = _wrap
        ok = False
        for code in [160, 161, 162, 163, 164, 224, 225, 256, 288, 320]:
            for fld in ["Gcode", "GCode", "gcode", "Command"]:
                try:
                    printer._ws.send(json.dumps({"Id": "", "Data": {"Cmd": code, "Data": {fld: cmd},
                        "RequestID": "", "MainboardID": "", "TimeStamp": int(time.time()*1000), "From": 1}}))
                    try:
                        if q.get(timeout=0.4) == 0: ok = True; break
                    except Exception: pass
                except Exception: pass
            if ok: break
        acked.append(ok)
    printer._ws.on_message = orig
    n = sum(1 for a in acked if a)
    if n:
        return jsonify({"ok": True, "message": f"Fan-off sent ({n}/{len(FAN_OFF_GCODE)} acknowledged)."})
    return jsonify({"ok": False, "error": "No fan command was acknowledged by the firmware. "
                    "This firmware may not accept fan M-codes over the command channel."}), 502

from ccplotter.routes import twopen as _routes_twopen  # noqa: F401,E402 (opt-in /api/twopen; registration side effect only)
