"""ccplotter/routes/dbv3.py -- DrawingBotV3 "full integration" routes.

The /api/dbv3_inbox + /api/dbv3_import routes live in ccplotter/routes/core.py;
this module adds further routes on the shared `app` instance, like every other
routes/*.py file (see ccplotter/routes/__init__.py for why the Flask() instance
lives there and every module does `from ccplotter.routes import app`).

These routes support the manual "export -> open panel -> Refresh -> click to
import" loop:

1. POST /api/dbv3_launch -- launches the DrawingBotV3-Premium.exe GUI via a
   detached subprocess (never blocks the Flask worker thread), optionally
   passing the currently-selected drawing's RASTER source path as an argv
   argument so DBV3 opens with the image already loaded. DBV3 (a
   Processing/Java app under a native launcher) accepts a bare image path as
   its first argument and loads it; the window title then includes the
   image's basename. No headless/CLI generation exists -- this only affects
   what is loaded when the GUI window opens.

2. GET /api/dbv3_inbox_poll -- a CHEAP listing variant for the frontend's
   auto-import watcher (polls every few seconds). The existing
   /api/dbv3_inbox (core.py) reads and SHA-256-hashes every file's full
   bytes on every call -- fine for a manual one-shot "open panel" refresh,
   too expensive to run every 3-5s against potentially-large SVGs. This
   route does a bare os.stat() per file (name/mtime/size only, no read/hash)
   -- the frontend's own size-stable-across-two-polls check (using this
   route's `size` field) is what it uses to decide a file is done being
   written, and filename is enough to dedupe against already-imported
   files client-side. Also echoes `inbox_dir` (same value core.py's
   /api/dbv3_inbox returns) so the frontend's "copy inbox path" button has
   no duplicate route to maintain.

3. GET /api/dbv3_status -- best-effort "is DrawingBotV3 running" readout
   used by the panel's launch
   button for "launching / already running" feedback. Also folded into
   POST /api/dbv3_launch's response as `was_running`.
"""
import os
import subprocess
import sys
import uuid
from pathlib import Path

from flask import jsonify, request, send_from_directory

from ccplotter.config import DBV3_INBOX_DIR, WORK_DIR, RASTER_EXTS

from ccplotter.routes import app

# Staging dir for "send image straight to DBV3" uploads (skips the tracev3
# pipeline and goes straight to DBV3). Deliberately NOT
# DBV3_INBOX_DIR: the inbox is DBV3's OUTPUT drop folder, watched by the
# frontend's auto-import poll loop for .svg exports. This staging dir holds
# the raw INPUT image handed to DBV3 as argv[1] -- putting it in the inbox
# would make the watcher try to "import" a raster as if it were an export.
DBV3_STAGING_DIR = WORK_DIR / "dbv3_staging"
DBV3_STAGING_DIR.mkdir(exist_ok=True)


def _valid_inbox_filename(filename: str) -> bool:
    """Same guard as core.py's api_dbv3_import: bare filename (no path
    traversal component), .svg only, before any filesystem touch."""
    return bool(filename) and Path(filename).name == filename and Path(filename).suffix.lower() == ".svg"

# DrawingBotV3-Premium's executable, from the CCPLOTTER_DBV3_EXE environment
# variable. When it is unset the path is a bare file name that does not exist, so
# the DrawingBotV3 routes answer 404 "not found". No other code assumes a location.
DBV3_EXE_PATH = Path(os.environ.get("CCPLOTTER_DBV3_EXE") or "DrawingBotV3-Premium.exe")


# Best-effort Windows process-presence check for launch/state feedback. `tasklist /FI "IMAGENAME eq ..."`
# is the cheap, no-extra-dependency way to ask "is this exe already
# running" on Windows -- deliberately NOT psutil (no new dependency for one
# best-effort check). Never raises: any failure (non-Windows, tasklist
# missing/blocked, unexpected output) reports None (unknown), and callers
# treat None the same as "don't know, proceed anyway" -- this must never
# block a launch or turn into a hard error.
def _dbv3_process_running():
    if sys.platform != "win32":
        return None
    exe_name = DBV3_EXE_PATH.name
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/FO", "CSV", "/NH"],
            stderr=subprocess.DEVNULL, text=True, timeout=3)
    except Exception:
        return None
    return exe_name.lower() in out.lower()


@app.route("/api/dbv3_status")
def api_dbv3_status():
    """Cheap best-effort 'is DrawingBotV3 currently running' readout for the
    panel's status line / 'already running' launch feedback.
    `running: null` means unknown (non-Windows, or the
    tasklist probe failed) -- the frontend must treat that as
    non-committal, never as a false 'not running'."""
    return jsonify({"running": _dbv3_process_running(), "exe_configured": DBV3_EXE_PATH.is_file()})


def _launch_dbv3_subprocess(image_path=None):
    """Shared detached-launch mechanism for every DBV3-GUI-opening route
    (/api/dbv3_launch and /api/dbv3_send_image both use it, so the
    DETACHED_PROCESS flags logic lives in exactly one place). `image_path`,
    if given, is passed as a bare argv[1] (DBV3 loads it as the active image
    on window open). Returns the Popen args' loaded path
    (str or None) on success; raises OSError on launch failure, same as
    subprocess.Popen -- callers translate that to their own error response.
    """
    args = [str(DBV3_EXE_PATH)]
    loaded_path = None
    if image_path is not None:
        args.append(str(image_path))
        loaded_path = str(image_path)

    creationflags = 0
    if sys.platform == "win32":
        # DETACHED_PROCESS: fully decouples the child from this Flask
        # process's console/lifetime -- closing/restarting the dev server or
        # the packaged app must never take DBV3 down with it, and DBV3 must
        # never inherit stdio that could block this process's own subprocess
        # pipes.
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    subprocess.Popen(args, creationflags=creationflags, close_fds=True,
                      stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL)
    return loaded_path


@app.route("/api/dbv3_launch", methods=["POST"])
def api_dbv3_launch():
    """Launch the DrawingBotV3 GUI, optionally pre-loaded with a source
    image. Body: {"src_name": "src_<id>.png"} (optional -- omit or send a
    non-raster/missing name for a bare launch, never an error). Always
    returns promptly -- Popen is fire-and-forget, DBV3's own window opening
    (probe: several seconds, it's a JVM app under a native launcher) never
    blocks this request.

    Already-running DBV3: no focus/reuse handling attempted -- Windows will
    either open a second instance or DBV3's own single-instance logic (if
    any) takes over; we don't special-case this (undocumented, not worth
    guessing at from a closed binary). `was_running` in the response is
    purely informational (from the same best-effort tasklist probe as
    /api/dbv3_status) so the frontend can say "already running -- opening
    another window" instead of implying a fresh launch; it never changes
    what this route does.
    """
    if not DBV3_EXE_PATH.is_file():
        return jsonify({"error": f"DrawingBotV3 not found at {DBV3_EXE_PATH}. "
                                  "Set CCPLOTTER_DBV3_EXE to the full path of "
                                  "DrawingBotV3-Premium.exe."}), 404

    was_running = _dbv3_process_running()
    body = request.json or {}
    src_name = str(body.get("src_name", ""))
    image_path = None

    if src_name and Path(src_name).name == src_name:
        suffix = Path(src_name).suffix.lower()
        candidate = WORK_DIR / src_name
        if suffix in RASTER_EXTS and candidate.is_file():
            image_path = candidate.resolve()
        # Vector (.svg) sources or a missing file: fall through to a bare
        # launch -- DBV3's own art-generation modules take a raster image,
        # not an SVG, as their input; a vector-selected drawing therefore
        # degrades silently to a bare launch (the same as "no drawing loaded")
        # rather than being treated as an error.

    try:
        loaded_path = _launch_dbv3_subprocess(image_path)
    except OSError as e:
        return jsonify({"error": f"Failed to launch DrawingBotV3: {e}"}), 500

    return jsonify({"ok": True, "loaded_path": loaded_path, "was_running": was_running})


@app.route("/api/dbv3_send_image", methods=["POST"])
def api_dbv3_send_image():
    """Send a raw image straight to DrawingBotV3, skipping ccPlotter's own
    trace pipeline entirely (loading an image into ccPlotter's own
    handleFiles() also kicks off a tracev3 preview, wasted work when the
    destination is DBV3, not a plotted trace).

    Multipart upload (FormData field "image"), NOT a src_name reference --
    the whole point is this image was never loaded into ccPlotter's
    state.drawings in the first place, so there's no WORK_DIR src_<id> file
    to point at. A browser <input type=file> only hands JS a File object
    (bytes + name), never a real filesystem path (browser sandboxing) --
    but DBV3 needs a real path as argv[1] -- so this route is the bridge:
    it writes the uploaded bytes to DBV3_STAGING_DIR under a fresh uuid
    name (collision-proof, no dependence on the original filename being
    safe) preserving the original extension, then launches DBV3 against
    that staged path via the same _launch_dbv3_subprocess helper
    /api/dbv3_launch uses.

    Staged files are never cleaned up automatically here (same
    housekeeping posture as WORK_DIR's own src_ files) -- DBV3 reads the
    path once at launch and doesn't need it afterward, but deleting it out
    from under a slow-opening DBV3 window would be a race for zero benefit.
    """
    if not DBV3_EXE_PATH.is_file():
        return jsonify({"error": f"DrawingBotV3 not found at {DBV3_EXE_PATH}. "
                                  "Set CCPLOTTER_DBV3_EXE to the full path of "
                                  "DrawingBotV3-Premium.exe."}), 404

    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify({"error": "No image file uploaded."}), 400

    orig_name = Path(f.filename).name  # strip any path component client sent
    suffix = Path(orig_name).suffix.lower()
    if suffix not in RASTER_EXTS:
        return jsonify({"error": f"'{orig_name}' is not a supported raster image "
                                  f"({', '.join(sorted(RASTER_EXTS))})."}), 400

    was_running = _dbv3_process_running()
    staged_path = DBV3_STAGING_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        f.save(str(staged_path))
    except OSError as e:
        return jsonify({"error": f"Failed to stage image: {e}"}), 500

    try:
        loaded_path = _launch_dbv3_subprocess(staged_path.resolve())
    except OSError as e:
        return jsonify({"error": f"Failed to launch DrawingBotV3: {e}"}), 500

    return jsonify({"ok": True, "launched_path": loaded_path, "was_running": was_running,
                     "original_name": orig_name})


@app.route("/api/dbv3_inbox_poll")
def api_dbv3_inbox_poll():
    """Cheap inbox listing for the frontend auto-import watcher -- mtime/size
    only, no file reads or hashing (see module docstring). Polled every few
    seconds while the app is open; must stay O(number of files) stat() calls,
    nothing more."""
    files = []
    for p in sorted(DBV3_INBOX_DIR.glob("*.svg"), key=lambda x: x.name):
        try:
            st = p.stat()
            files.append({"filename": p.name, "mtime": st.st_mtime, "size": st.st_size})
        except OSError:
            continue
    return jsonify({"files": files, "inbox_dir": str(DBV3_INBOX_DIR.resolve())})


@app.route("/api/dbv3_inbox_file/<path:filename>")
def api_dbv3_inbox_file(filename):
    """Serve a raw inbox SVG's bytes -- used ONLY as an <img> thumbnail
    source in the DBV3 Inbox panel (browsers rasterize SVG natively, same
    technique already used for the main plate preview -- see
    assets/templates/index.html's `plate-img` <image href>. No thumbnail
    generation needed: this is free). Read-only, .svg only, filename
    validated against path traversal before any filesystem touch."""
    if not _valid_inbox_filename(filename):
        return jsonify({"error": "invalid filename"}), 400
    if not (DBV3_INBOX_DIR / filename).is_file():
        return jsonify({"error": "file not found in DBV3 inbox"}), 404
    return send_from_directory(str(DBV3_INBOX_DIR.resolve()), filename, mimetype="image/svg+xml")


@app.route("/api/dbv3_inbox_delete", methods=["POST"])
def api_dbv3_inbox_delete():
    """Delete a single file from the DBV3 inbox (housekeeping -- files
    accumulate there with no cleanup path otherwise). Same filename
    validation as import/serve; deleting a file that's already gone (race
    with a manual delete elsewhere) is not an error."""
    body = request.json or {}
    filename = str(body.get("filename", ""))
    if not _valid_inbox_filename(filename):
        return jsonify({"error": "invalid filename"}), 400
    target = DBV3_INBOX_DIR / filename
    try:
        target.unlink(missing_ok=True)
    except OSError as e:
        return jsonify({"error": f"Failed to delete: {e}"}), 500
    return jsonify({"ok": True})


@app.route("/api/dbv3_inbox_clear", methods=["POST"])
def api_dbv3_inbox_clear():
    """Delete every .svg currently sitting in the DBV3 inbox. Housekeeping
    action -- does not touch anything already imported into work/ (that copy
    is independent, same pattern as /api/clear_drawings vs. run archives)."""
    deleted, errors = 0, []
    for p in DBV3_INBOX_DIR.glob("*.svg"):
        try:
            p.unlink()
            deleted += 1
        except OSError as e:
            errors.append(f"{p.name}: {e}")
    return jsonify({"ok": True, "deleted": deleted, "errors": errors})
