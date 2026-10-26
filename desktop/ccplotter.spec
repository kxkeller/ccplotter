# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for ccPlotter (one-folder build).

Build from the PROJECT ROOT:
    pyinstaller desktop/ccplotter.spec --noconfirm

Output: dist/ccPlotter/  (portable folder; zip it or copy it to any Windows PC).
"""
import os
from PyInstaller.utils.hooks import collect_all, copy_metadata, collect_submodules

ROOT = os.path.abspath(os.getcwd())

datas = []
binaries = []
hiddenimports = []

# --- App resources (read-only, bundled at the root of _MEIPASS) ---
# templates/ and Model Images/ live under assets/ in the repo. The
# bundle-internal destination names below ("templates", "Model Images") are what
# desktop/ccplotter_app.py reads (res / "templates", res / "Model Images"),
# independent of where they sit in the source tree.
datas += [
    (os.path.join(ROOT, "assets", "templates", "index.html"), "templates"),
    # The .ico is ALSO embedded in the exe (see `icon=` below), which is what
    # Explorer and the shortcut show. That is a different thing from the RUNNING
    # window's icon: pywebview creates its own top-level window and, with no icon
    # set on it, Windows falls back to a generic document glyph in the taskbar and
    # Alt-Tab. Bundling it as DATA makes the file reachable at runtime so
    # ccplotter_app.py can set the window icon itself.
    (os.path.join(ROOT, "desktop", "ccplotter.ico"), "."),
    (os.path.join(ROOT, "vpype.toml"), "."),
    (os.path.join(ROOT, "ai_config.json"), "."),
    # Lives in assets/seed/ (not the repo root, where a working-directory copy
    # would shadow the runtime one); shipped at the bundle root as
    # calibration.seed.json and seeded once into the data dir by
    # desktop/ccplotter_app.py seed_defaults().
    (os.path.join(ROOT, "assets", "seed", "calibration.seed.json"), "."),
]
# Version stamp (written by desktop/build.ps1 just before this spec runs).
# Bundled at the resource root, same place app.py's _load_version_info looks
# for it (sys._MEIPASS / "version.json"). Building via `pyinstaller ...` alone
# (skipping build.ps1) leaves this missing -- app.py falls back to git, so the
# banner will read "source" instead of a stamped date; not fatal, just wrong.
#
# A STALE stamp is worse than a missing one: a missing stamp falls back to git
# and reads "source", but a stale one bundles a wrong SHA with no fallback and no
# warning, so the banner names a commit that differs from the running code.
# The check below compares the stamp against git HEAD and warns loudly.
_version_json = os.path.join(ROOT, "desktop", "version.json")
if os.path.exists(_version_json):
    datas.append((_version_json, "."))
    try:
        import json as _json
        import subprocess as _sp
        _stamped = _json.loads(open(_version_json, encoding="utf-8").read()).get("commit")
        _head = _sp.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                 stderr=_sp.DEVNULL, timeout=5).decode().strip()
        if _stamped and _head and _stamped != _head:
            print("[spec] WARNING: desktop/version.json is STALE -- stamped {} but "
                  "HEAD is {}. The banner will name the wrong commit. Run "
                  "desktop/build.ps1 (not pyinstaller directly) to restamp."
                  .format(_stamped, _head))
    except Exception:
        pass   # never let a version check break a build
else:
    print("[spec] WARNING: desktop/version.json not found -- run desktop/build.ps1 "
          "(not pyinstaller directly) to stamp the version banner.")
# --- vpype ecosystem: modules + entry-point metadata (PyInstaller drops these) ---
# The .dist-info metadata is REQUIRED so vpype can discover plugins (e.g. gwrite)
# via importlib.metadata entry_points at runtime.
_VPYPE_PKGS = ["vpype", "vpype_cli", "vpype_gcode", "vpype_flow_imager"]
_VPYPE_DISTS = ["vpype", "vpype-gcode", "vpype-flow-imager"]
for pkg in _VPYPE_PKGS:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception as e:
        print(f"[spec] skip collect_all({pkg}): {e}")
for dist in _VPYPE_DISTS:
    try:
        # recursive=True also grabs metadata for the whole dependency tree
        # (pnoise, shapely, etc.) — several read their own version at import.
        datas += copy_metadata(dist, recursive=True)
    except Exception as e:
        print(f"[spec] skip copy_metadata({dist}): {e}")
# Belt-and-suspenders for packages known to read importlib.metadata at import.
for dist in ["pnoise"]:
    try:
        datas += copy_metadata(dist)
    except Exception as e:
        print(f"[spec] skip copy_metadata({dist}): {e}")

# --- tomli (vpype's TOML parser) is mypyc-compiled: bundle it + its shared
#     mypyc runtime .pyd, which lives at the site-packages ROOT with a hashed
#     name and is missed by default. ---
try:
    import glob as _glob
    import tomli as _tomli
    d, b, h = collect_all("tomli")
    datas += d; binaries += b; hiddenimports += h
    _sp = os.path.dirname(os.path.dirname(_tomli.__file__))  # site-packages
    for _pyd in _glob.glob(os.path.join(_sp, "*__mypyc*.pyd")):
        binaries.append((_pyd, "."))  # place at bundle root so import resolves
        print(f"[spec] bundling mypyc runtime: {os.path.basename(_pyd)}")
except Exception as e:
    print(f"[spec] tomli/mypyc handling failed: {e}")

# --- Other packages PyInstaller's stock hooks don't fully cover ---
for pkg in ["svgwrite", "ezdxf", "fitz", "openai", "websocket", "webview",
            "clr_loader", "bottle"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception as e:
        print(f"[spec] skip collect_all({pkg}): {e}")

# shapely (vpype geometry dep) ships DLLs; ensure submodules are seen.
hiddenimports += collect_submodules("shapely")
# scipy/skimage/cv2/PIL/numpy are handled by PyInstaller's bundled hooks.

# --- Guard: never bundle credentials or a user's runtime state. Every data file above is listed
#     explicitly (no globbing of the working directory); this check makes that a hard failure if a
#     key file, an environment file or a calibration/pen-profile file ever ends up in `datas`.
import fnmatch as _fnmatch
_FORBIDDEN_DATA = ("*_key.txt", ".env*", "pen_profiles.json", "calibration.json", "openai_key.txt")
for _src, _dest in datas:
    _name = os.path.basename(str(_src)).lower()
    if any(_fnmatch.fnmatch(_name, _pat) for _pat in _FORBIDDEN_DATA):
        raise SystemExit("[spec] refusing to bundle {!r}: matches a credential / runtime-state pattern".format(_src))

a = Analysis(
    [os.path.join(ROOT, "desktop", "ccplotter_app.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PyQt5", "PyQt6", "PySide2", "PySide6",
              "pytest", "IPython", "notebook",
              # Heavy ML frameworks pulled in transitively but never used by app.py
              # (image generation is via the OpenAI cloud API, not local models).
              "torch", "torchvision", "torchaudio", "transformers",
              "tensorboard", "tensorflow", "jax", "jaxlib", "flax", "triton"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ccPlotter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # no terminal window
    icon=os.path.join(ROOT, "desktop", "ccplotter.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ccPlotter",
)
