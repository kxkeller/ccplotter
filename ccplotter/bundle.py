import json
import threading
import time

# ── Last-generation bundle: in-memory snapshot of the most recent COMPLETED
# (job_type='gcode') /api/process call, captured at generation time so a later
# /download_bundle always matches what was actually served/plotted -- not
# whatever happens to be sitting in work/layout.* by the time of download
# (those get overwritten by the next run). ──
_LAST_BUNDLE = None
_BUNDLE_LOCK = threading.Lock()

def _snapshot_bundle(body, gcode_path, preview_path, res):
    """Capture the exact bytes served for this generation, for /download_bundle."""
    global _LAST_BUNDLE
    try:
        snap = {
            "preview_svg": preview_path.read_bytes(),
            "preview_name": preview_path.name,
            "gcode": gcode_path.read_bytes(),
            "gcode_name": gcode_path.name,
            "request_json": json.dumps(body, indent=1).encode("utf-8"),
            "info_json": json.dumps(res, indent=1, default=str).encode("utf-8"),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "job_id": res.get("job_id"),
        }
    except Exception as e:
        print(f"  [bundle] snapshot failed: {e}")
        return
    with _BUNDLE_LOCK:
        _LAST_BUNDLE = snap
