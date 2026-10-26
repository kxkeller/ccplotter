"""
ccPlotter desktop launcher.

Boots the Flask app (app.py) hidden in a background thread and shows it in a
native window via pywebview. Designed to run both:
  - in dev:   python desktop/ccplotter_app.py           (resources = project root)
  - frozen:   ccPlotter.exe (PyInstaller one-folder)    (resources = sys._MEIPASS)

The app writes runtime files with paths relative to the CWD (work/, calibration.json,
ai_config.json, vpype.toml, ...). A read-only bundle can't be written to, so we chdir
into a per-user data dir (%LOCALAPPDATA%\\ccPlotter) BEFORE importing app.py, seeding
default config files there on first run.
"""
import os
import sys
import time
import socket
import threading
from pathlib import Path


def resource_base() -> Path:
    """Directory holding bundled read-only resources (templates/, default configs)."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent  # project root in dev


def data_dir() -> Path:
    """Writable per-user working directory for runtime files.

    Delegates to ccplotter.runtime_home.runtime_home(), the ONE resolver
    shared with config.py/cal.py (pen_profiles.json + calibration.json resolve
    from here, never from the CWD). With CCPLOTTER_HOME unset this is
    %LOCALAPPDATA%/ccPlotter. runtime_home is import-light and
    side-effect-free, so it is safe to import before the os.chdir() in main()."""
    if not getattr(sys, "frozen", False):
        _root = str(resource_base())
        if _root not in sys.path:
            sys.path.insert(0, _root)
    from ccplotter.runtime_home import runtime_home
    d = runtime_home()
    d.mkdir(parents=True, exist_ok=True)
    return d


def seed_defaults(res: Path, data: Path) -> None:
    """Copy default config files into the data dir. Two policies, deliberately
    different: USER DATA (calibration.json -- measured Z heights/offsets,
    ai_config.json) seeds ONCE and is never
    overwritten, so a rebuild can't clobber the user's own tuning. vpype.toml
    is APP LOGIC (the gcode templates), not user data -- like templates/
    index.html, it should always match the running build.

    vpype.toml is overwritten on every launch because a stale data-dir copy
    would otherwise keep serving old templates (e.g. outdated fan indices or
    postamble): CONFIG_PATH in app.py is the relative path "vpype.toml",
    resolved against the CWD (the data dir, after os.chdir), never against the
    bundle."""
    # The seed calibration lives in assets/seed/calibration.seed.json (not at the
    # repo root, where a CWD copy would shadow the runtime one); the bundle ships
    # it at the resource root under that name and it is seeded ONCE into the
    # data dir as calibration.json.
    for src_name, name in (("ai_config.json", "ai_config.json"),
                           ("calibration.seed.json", "calibration.json")):
        src, dst = res / src_name, data / name
        if src.exists() and not dst.exists():
            try:
                dst.write_bytes(src.read_bytes())
            except Exception:
                pass
    vt_src, vt_dst = res / "vpype.toml", data / "vpype.toml"
    if vt_src.exists():
        try:
            vt_dst.write_bytes(vt_src.read_bytes())
        except Exception:
            pass
    # Bundled reference images used by the Studio/style features, if present.
    # mi_src is a bundle-internal path (ccplotter.spec's datas dest stays
    # "Model Images" regardless of the repo's source layout -- see that
    # spec's comments). mi_dst must match ccplotter.config.MODEL_IMAGES_DIR
    # ("assets/Model Images"), since app.root_path is
    # overridden to `data` below and MODEL_IMAGES_DIR is resolved against it.
    mi_src, mi_dst = res / "Model Images", data / "assets" / "Model Images"
    if mi_src.is_dir() and not mi_dst.exists():
        try:
            import shutil
            mi_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(mi_src, mi_dst)
        except Exception:
            pass


_SINGLE_INSTANCE_MUTEX_NAME = "Local\\ccPlotter_SingleInstance_v1"
_SINGLE_INSTANCE_MUTEX_NAME_SELFTEST = "Local\\ccPlotter_SingleInstance_selftest_v1"
_instance_mutex_handle = None  # holds the acquired handle so it can be released on exit


def acquire_single_instance_lock() -> bool:
    """Single-instance guard for the packaged app, backed by a Windows named
    kernel mutex (not a lockfile+PID scheme).

    WHY THE GUARD EXISTS: with no guard, a user could
    open ccPlotter twice; both processes chdir into the SAME %LOCALAPPDATA%\\
    ccPlotter data dir (data_dir()) and both run their own Flask server on
    their own dynamic port, both reading/writing the same work/ directory.
    Two writers racing on work/layout.routed.svg produced a file with a
    complete SVG document followed by leftover tail bytes from the other
    writer ("junk after document element"). This guard makes the SECOND
    launch refuse to start a server or touch work/ at all.

    WHY A NAMED MUTEX AND NOT A LOCKFILE: a lockfile-with-PID scheme needs
    explicit stale-lock detection (was the PID that wrote this file killed
    without cleanup?) which means OpenProcess/PID-liveness checks and a
    window where a dead-but-not-yet-detected holder still blocks new
    launches. A Windows named kernel object (mutex) is reference-counted by
    the OS itself: ALL handles to it -- including ones held by a process
    that dies or is killed -- are closed automatically when that process
    terminates, which releases the mutex immediately. That makes "stale
    lock because the app died" structurally impossible to hit here: a
    dead holder's lock is gone the instant the process is gone, with no
    PID-liveness code needed on our side. This is the standard, robust
    Windows-native pattern for exactly this problem (Windows-only target,
    per this file's own docstring: dev + PyInstaller-frozen, both Windows).

    "Local\\" namespaces the mutex to the current user session (not
    "Global\\"), matching data_dir()'s per-user %LOCALAPPDATA% scope --
    two different Windows accounts on the same machine get independent
    data dirs and should NOT contend for the same lock.

    SELFTEST gets its OWN mutex name (see _SINGLE_INSTANCE_MUTEX_NAME_
    SELFTEST) rather than being routed around this function entirely, so
    the guard mechanism itself still runs during a selftest build-verify
    pass, while (a) a real user session already open does not block the
    selftest launch and (b) repeated selftest runs (e.g. CI re-running the
    build check) don't collide with a leftover selftest mutex from a prior
    run that already exited (which releases it anyway, per the above).

    This has no interaction with the dev Flask server (`python app.py`):
    that entry point never imports this module, so it never touches this
    mutex namespace.
    """
    global _instance_mutex_handle
    name = (_SINGLE_INSTANCE_MUTEX_NAME_SELFTEST
            if os.environ.get("CCPLOTTER_SELFTEST") == "1"
            else _SINGLE_INSTANCE_MUTEX_NAME)
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        ERROR_ALREADY_EXISTS = 183
        # bInitialOwner=True: if we create it, we own it immediately -- no
        # separate WaitForSingleObject call needed.
        handle = kernel32.CreateMutexW(None, True, name)
        if not handle:
            # Could not even create the mutex object. Fail OPEN (allow
            # startup) rather than block the app over a guard-infra error.
            return True
        already_running = (kernel32.GetLastError() == ERROR_ALREADY_EXISTS)
        if already_running:
            kernel32.CloseHandle(handle)
            return False
        _instance_mutex_handle = handle
        return True
    except Exception:
        # ctypes/kernel32 unavailable (shouldn't happen on the Windows-only
        # packaged target) -- fail open rather than block startup.
        return True


def release_single_instance_lock() -> None:
    """Release the mutex acquired above. Safe to call multiple times / when
    nothing was acquired (e.g. this process lost the race)."""
    global _instance_mutex_handle
    if _instance_mutex_handle is None:
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.ReleaseMutex(_instance_mutex_handle)
        kernel32.CloseHandle(_instance_mutex_handle)
    except Exception:
        pass
    finally:
        _instance_mutex_handle = None


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            s.close()
            return True
        s.close()
        time.sleep(0.15)
    return False


def _instance_work_dir() -> Path:
    """The work dir THIS launcher process serves from. Primary instance:
    canonical data_dir()/work (env unset -> identical to pre-multi-instance
    behavior). Secondary instances: the isolated work/inst-<pid> that main()
    exported via CCPLOTTER_WORK_DIR, so their Save / Open-folder bridges point
    at their OWN outputs, not the primary's. Mirrors ccplotter.config.WORK_DIR's
    own resolution so the native bridge and the Flask server never disagree."""
    override = os.environ.get("CCPLOTTER_WORK_DIR")
    return Path(override) if override else (data_dir() / "work")


def _pid_alive(pid: int) -> bool:
    """True if a process with this PID currently exists (Windows). Fail-safe: on
    any uncertainty, report ALIVE so orphan-sweeping never deletes a dir that
    might belong to a running instance."""
    try:
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return False  # no such process
        kernel32.CloseHandle(h)
        return True
    except Exception:
        return True


def _sweep_orphan_instance_dirs() -> None:
    """Best-effort removal of work/inst-<pid> dirs left behind by secondary
    instances that exited before their atexit cleanup ran. Only removes dirs
    whose PID is no longer alive -- never a running instance's. Cache only;
    every failure is non-fatal. The primary instance runs this at startup."""
    import shutil
    base = data_dir() / "work"
    try:
        candidates = list(base.glob("inst-*"))
    except Exception:
        return
    for d in candidates:
        if not d.is_dir():
            continue
        try:
            pid = int(d.name.split("-", 1)[1])
        except (ValueError, IndexError):
            continue
        if not _pid_alive(pid):
            shutil.rmtree(d, ignore_errors=True)



def _apply_window_icon() -> None:
    """Give the RUNNING window its own icon (taskbar, title bar, Alt-Tab).

    The .ico embedded in the exe by PyInstaller's `icon=` is what Explorer and
    the shortcut use. It does NOT reach the window pywebview creates -- with no
    icon set on that window, Windows substitutes a generic document glyph, which
    is a generic sheet-of-paper image in the taskbar.

    Done with WM_SETICON via ctypes rather than a pywebview parameter because
    the icon kwarg's availability varies across pywebview versions and GUI
    backends; sending the window message works on any of them under Windows.

    Best-effort by construction: an icon is cosmetic and must never be able to
    stop the app from starting, so every failure path is swallowed.
    """
    try:
        import ctypes
        ico = resource_base() / "ccplotter.ico"
        if not ico.is_file():
            ico = Path(__file__).resolve().parent / "ccplotter.ico"
        if not ico.is_file():
            return
        user32, WM_SETICON = ctypes.windll.user32, 0x0080
        IMAGE_ICON, LR_LOADFROMFILE, LR_DEFAULTSIZE = 1, 0x0010, 0x0040
        # Big (Alt-Tab, taskbar) and small (title bar) are separate slots.
        for wparam, cx, cy in ((1, 256, 256), (0, 16, 16)):
            h = user32.LoadImageW(None, str(ico), IMAGE_ICON, cx, cy,
                                  LR_LOADFROMFILE | (LR_DEFAULTSIZE if not cx else 0))
            if not h:
                continue
            hwnd = user32.FindWindowW(None, title_for_icon[0]) if title_for_icon else None
            if hwnd:
                user32.SendMessageW(hwnd, WM_SETICON, wparam, h)
    except Exception as e:
        print(f"[icon] window icon not applied ({e}) -- cosmetic only, continuing")


title_for_icon: list = []


def main() -> int:
    # Acquire the single-instance guard BEFORE anything touches the shared
    # data dir (seed_defaults/chdir below): a second launch must not race
    # the first over work/, calibration.json, etc. See
    # acquire_single_instance_lock()'s docstring for the corruption this
    # prevents and why a named mutex (not a lockfile+PID) was chosen.
    import atexit
    # Multi-instance model (parallel processing of several images). The guard
    # does not refuse a second launch. Instead the FIRST instance (mutex acquired) keeps the canonical work/ --
    # byte-identical to single-instance behavior when run alone -- and every
    # SUBSEQUENT instance runs isolated in work/inst-<pid>, exported via
    # CCPLOTTER_WORK_DIR BEFORE `import app` (which imports ccplotter.config,
    # where WORK_DIR reads it). Separate work dirs remove the shared-file race
    # that acquire_single_instance_lock()'s docstring describes (two writers
    # corrupting work/layout.routed.svg), so both instances can process in
    # parallel on their own dynamic ports. Shared config (calibration.json,
    # pen_profiles.json, vpype.toml, dbv3_inbox) stays common and is read-mostly;
    # concurrent EDITS (recalibrating in two windows at once, or the dbv3 inbox
    # importing the same dropped file twice) are still races and are NOT fenced.
    if acquire_single_instance_lock():
        atexit.register(release_single_instance_lock)
        _sweep_orphan_instance_dirs()  # primary removes cache dirs of secondaries that exited uncleanly
    else:
        inst_work = data_dir() / "work" / f"inst-{os.getpid()}"
        os.environ["CCPLOTTER_WORK_DIR"] = str(inst_work)
        atexit.register(lambda: __import__("shutil").rmtree(inst_work, ignore_errors=True))

    res = resource_base()
    data = data_dir()
    seed_defaults(res, data)

    # Must chdir BEFORE importing app.py (it creates work/ dirs at module load).
    os.chdir(data)

    # Persistent logging so processing errors are captured (windowed app has no
    # console). Flask logs unhandled view exceptions to app.logger -> root.
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(str(data / "app.log"), encoding="utf-8")],
    )

    import app as ccapp  # noqa: E402  (deliberate: after chdir)
    ccapp.start_printer()  # printer websocket is dialed at launch, not at import

    # Templates live in the bundle, not the writable data dir. Point Flask at them
    # explicitly (Flask's default resolution is unreliable under PyInstaller).
    from jinja2 import FileSystemLoader
    tpl = res / "templates"
    if not tpl.is_dir():              # running from a source checkout: the template lives under assets/
        tpl = res / "assets" / "templates"
    ccapp.app.template_folder = str(tpl)
    ccapp.app.jinja_loader = FileSystemLoader(str(tpl))

    # app.py serves runtime files via send_from_directory(WORK_DIR, ...) and
    # (MODEL_IMAGES_DIR, ...) with RELATIVE dirs. Flask resolves those against
    # app.root_path, which under PyInstaller is the bundle dir (_MEIPASS) — not
    # the data dir where work/ and Model Images/ actually live. Without this,
    # every /work/<preview>.svg and /model-images/<f> 404s (blank previews).
    ccapp.app.root_path = str(data)

    port = int(os.environ.get("CCPLOTTER_PORT") or free_port())

    def serve():
        # threaded=True so the UI stays responsive during long plot operations.
        ccapp.app.run(host="127.0.0.1", port=port, debug=False, threaded=True,
                      use_reloader=False)

    threading.Thread(target=serve, daemon=True).start()

    # Headless self-test: verify the bundle resolves resources + vpype plugins,
    # then exit. Used to validate the frozen build without opening a GUI window.
    if os.environ.get("CCPLOTTER_SELFTEST") == "1":
        ok = wait_for_server(port)
        import urllib.request
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10)
            served = resp.status == 200 and b"<title" in resp.read()
        except Exception as e:
            served = False
            print("SELFTEST server error:", e)
        # Real end-to-end vpype check: load the profile and actually emit gcode
        # via the gwrite plugin (plugins register lazily, so we must run one).
        try:
            import vpype as vp, vpype_cli, tempfile, shlex
            vp.config_manager.load_config_file("vpype.toml")
            has_profile = "elegoo_cc" in vp.config_manager.config.get("gwrite", {})
            gout = os.path.join(tempfile.gettempdir(), "cc_selftest.gcode")
            if os.path.exists(gout):
                os.remove(gout)
            vpype_cli.execute(
                f"line 0 0 10 10 pagesize 100mmx100mm "
                f"gwrite --profile elegoo_cc {shlex.quote(gout)}")
            gwrite_ok = os.path.exists(gout) and os.path.getsize(gout) > 0
        except Exception as e:
            has_profile = gwrite_ok = False
            print("SELFTEST vpype error:", repr(e))
        # Serve check: the app serves previews/images via send_from_directory with
        # a RELATIVE dir resolved against app.root_path. Write a probe into work/
        # and fetch it through /work to prove runtime files are actually served
        # (guards against a wrong root_path, which would blank all previews).
        try:
            (data / "work").mkdir(exist_ok=True)
            (data / "work" / "_selftest_probe.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
            pr = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/work/_selftest_probe.svg", timeout=10)
            work_serves = pr.status == 200
        except Exception as e:
            work_serves = False
            print("SELFTEST work-serve error:", repr(e))
        # Postamble fixture: the fan-off postamble and preamble fan-off are part
        # of the shared elegoo_cc gwrite profile itself (vpype.toml), so no
        # writer can produce an incomplete ending short of not calling gwrite at
        # all. Generate a real calibration G-code file with the default policy
        # on every self-test and check it for the full expected shape: fan-off
        # commands present AND M84 present (default end_policy=release).
        try:
            import tempfile as _tf
            cal_out = Path(_tf.gettempdir()) / "cc_selftest_calibration.gcode"
            ccapp._calibration_gcode(cal_out, pen_mm=0.5, z_up=3.5, z_down=-1.0)
            cal_text = cal_out.read_text()
            fans_ok = all(cmd in cal_text for cmd in ccapp.FAN_OFF_GCODE)
            m84_ok = "\nM84" in cal_text or cal_text.rstrip().endswith("M84")
            postamble_ok = fans_ok and m84_ok
        except Exception as e:
            postamble_ok = False
            print("SELFTEST postamble error:", repr(e))
        # Truthful-rendering fixture: assert on the output file, not the code.
        # A real vpype pipeline with a known fill_mm must produce a served SVG
        # whose own stroke-width attribute reflects that physical width, not
        # vpype's silent 1px fallback (see _stroke_px_for).
        try:
            import vpype_cli as _vpc, tempfile as _tf2, shlex as _sh2, re as _re2
            fill_mm_test = 0.3   # deliberately != the 0.15mm default, so a
                                 # hardcoded leftover constant would fail this
            tsvg = Path(_tf2.gettempdir()) / "cc_selftest_truthwidth.svg"
            _vpc.execute(
                f"line 0 0 50 50 pagesize 100mmx100mm "
                f"penwidth {fill_mm_test}mm write {_sh2.quote(str(tsvg))}")
            svg_text = tsvg.read_text(encoding="utf-8")
            m = _re2.search(r'stroke-width="([\d.]+)"', svg_text)
            # 100mm page -> 96dpi-px viewBox; expected px = fill_mm * 96/25.4
            expected_px = fill_mm_test * 96.0 / 25.4
            got_px = float(m.group(1)) if m else -1.0
            truthful_width_ok = m is not None and abs(got_px - expected_px) < 0.01
        except Exception as e:
            truthful_width_ok = False
            print("SELFTEST truthful-width error:", repr(e))
        line = (f"SELFTEST server_up={ok} served_index={served} "
                f"gwrite_emits_gcode={gwrite_ok} elegoo_profile={has_profile} "
                f"work_serves={work_serves} calibration_postamble_ok={postamble_ok} "
                f"truthful_width_ok={truthful_width_ok}")
        print(line)
        try:
            (data / "selftest.txt").write_text(line, encoding="utf-8")
        except Exception:
            pass
        return 0 if (ok and served and gwrite_ok and has_profile
                     and work_serves and postamble_ok and truthful_width_ok) else 2

    if not wait_for_server(port):
        # Surface a minimal error window rather than dying silently.
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, "ccPlotter failed to start its internal server.", "ccPlotter", 0x10)
        except Exception:
            pass
        return 1

    import webview

    # WebView2's DownloadStarting handler (pywebview's edgechromium backend)
    # unconditionally cancels every download -- Content-Disposition:attachment
    # responses included, e.g. /download_bundle's send_file(as_attachment=True)
    # -- unless this is set. With it True, pywebview shows a native
    # SaveFileDialog pre-seeded to the user's Downloads folder (it reads the
    # shell folder registry key itself; see webview/platforms/edgechromium.py
    # on_download_starting). This is a plain <a href> / HTTP download, not the
    # <a download>-to-disk gcode path above, which needed the JS bridge instead
    # because WebView2 ignores that attribute entirely.
    webview.settings["ALLOW_DOWNLOADS"] = True

    class _JsApi:
        """Native bridge exposed to the page as window.pywebview.api. Needed
        because WebView2 ignores <a download> / window.location.href to files —
        the printer is offline, so gcode must be written to disk (USB)."""

        def ping(self):
            return {"ok": True}

        def _src(self, work_filename):
            return _instance_work_dir() / os.path.basename(str(work_filename))

        def save_file(self, work_filename, suggested_name):
            import shutil
            try:
                src = self._src(work_filename)
                if not src.exists():
                    return {"ok": False, "error": f"not found: {src}"}
                name = str(suggested_name)
                if not name.lower().endswith(".gcode"):
                    name += ".gcode"
                win = webview.windows[0]
                # pywebview's own filter parser (util.parse_file_type) requires the
                # description to match [\w ]+ before the paren — NO hyphens allowed.
                # "G-code (*.gcode)" fails that regex and raises ValueError before the
                # dialog even opens, which every Save As... button hit. "Gcode files"
                # (no hyphen) passes.
                res = win.create_file_dialog(
                    webview.SAVE_DIALOG, save_filename=name,
                    file_types=("Gcode files (*.gcode)", "All files (*.*)"))
                if not res:
                    return {"ok": False, "cancelled": True}
                dest = res[0] if isinstance(res, (list, tuple)) else res
                dest = str(dest)
                if not dest.lower().endswith(".gcode"):
                    dest += ".gcode"
                shutil.copy(str(src), dest)
                return {"ok": True, "path": dest}
            except Exception as e:
                # This bridge call is pywebview JS<->Python IPC, not HTTP — it never
                # hits Flask/werkzeug, so without this the failure would not appear
                # in app.log.
                logging.exception("save_file failed")
                return {"ok": False, "error": repr(e)}

        def pick_folder(self):
            """Native folder chooser (two-pen Save writes both pass files into the chosen folder)."""
            try:
                win = webview.windows[0]
                fd = getattr(getattr(webview, "FileDialog", None), "FOLDER", None)
                res = win.create_file_dialog(fd if fd is not None else webview.FOLDER_DIALOG)
                if not res:
                    return {"ok": False, "cancelled": True}
                return {"ok": True, "path": str(res[0] if isinstance(res, (list, tuple)) else res)}
            except Exception as e:
                logging.exception("pick_folder failed")
                return {"ok": False, "error": repr(e)}

        def save_files_to_folder(self, folder, items, overwrite=False):
            """Copy [[work_relative_path, dest_name], ...] into `folder` (all-or-nothing; see ccplotter/pair_save.py)."""
            from ccplotter.pair_save import copy_files
            try:
                return copy_files(_instance_work_dir(), folder, items, bool(overwrite))
            except Exception as e:
                logging.exception("save_files_to_folder failed")
                return {"ok": False, "error": repr(e)}

        def save_to_downloads(self, work_filename):
            """No-dialog fallback: copy the gcode to the Downloads folder."""
            import shutil
            try:
                src = self._src(work_filename)
                if not src.exists():
                    return {"ok": False, "error": f"not found: {src}"}
                dl = Path(os.path.expanduser("~")) / "Downloads"
                dl.mkdir(parents=True, exist_ok=True)
                dest = dl / src.name
                shutil.copy(str(src), str(dest))
                return {"ok": True, "path": str(dest)}
            except Exception as e:
                logging.exception("save_to_downloads failed")
                return {"ok": False, "error": repr(e)}

        def reveal_work(self):
            try:
                os.startfile(str(_instance_work_dir()))   # this instance's work folder
                return {"ok": True}
            except Exception as e:
                return {"ok": False, "error": repr(e)}

    # ccapp.VERSION_INFO is computed at import time (above) from the bundled
    # version.json (frozen) or git (dev) -- see app.py's _load_version_info.
    title = f"ccPlotter - {ccapp._version_string()}"
    title_for_icon.clear(); title_for_icon.append(title)

    webview.create_window(
        title,
        f"http://127.0.0.1:{port}/",
        width=1400,
        height=900,
        min_size=(900, 640),
        js_api=_JsApi(),
    )
    webview.start(func=_apply_window_icon)  # blocks until the window is closed
    return 0


def _run_guarded() -> int:
    try:
        return main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        try:
            log = data_dir() / "startup_error.log"
            log.write_text(tb, encoding="utf-8")
        except Exception:
            pass
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, f"ccPlotter failed to start.\n\n{tb[-800:]}", "ccPlotter", 0x10)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(_run_guarded())
