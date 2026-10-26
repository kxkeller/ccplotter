"""Start the ccPlotter dev server with every piece of runtime state outside the source tree.

The app reads `vpype.toml` from the current directory and creates `work/` and
`dbv3_inbox/` there at import time, and keeps `calibration.json` and
`pen_profiles.json` in the runtime home. This launcher makes the runtime home the
working directory, so the source checkout is never written to.

    python scripts/launch.py --home D:\\ccplotter-home            # port 5001
    python scripts/launch.py --home D:\\ccplotter-home --port 5050
    set CCPLOTTER_HOME=D:\\ccplotter-home && python scripts/launch.py

The home must be an absolute path outside the source tree. It is created if it
does not exist, and `vpype.toml` is copied into it if missing (an existing copy is
never overwritten). Nothing else is seeded here; the app seeds `pen_profiles.json`
itself, and `calibration.json` is yours to create by measuring your machine.

The work directory is <home>/work (set through CCPLOTTER_WORK_DIR, absolute; the path must be ASCII-only
because it is written into G-code header comments).

The server binds 127.0.0.1 unless --host is given. No printer connection is made
unless CCPLOTTER_PRINTER_HOST is set. The server refuses to start if the chosen
port already answers.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", default=os.environ.get("CCPLOTTER_HOME", ""),
                    help="absolute runtime-home directory (default: $CCPLOTTER_HOME; required)")
    ap.add_argument("--port", type=int, default=5001, help="port to listen on (default 5001)")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1)")
    args = ap.parse_args(argv)

    if not args.home:
        ap.error("a runtime home is required: pass --home <absolute dir> or set CCPLOTTER_HOME")
    home = Path(args.home)
    if not home.is_absolute():
        ap.error(f"the runtime home must be an absolute path, got {args.home!r}")
    home = home.resolve()
    try:
        home.relative_to(ROOT)
    except ValueError:
        pass
    else:
        ap.error(f"the runtime home must be outside the source tree ({ROOT}), got {home}")

    work = home / "work"
    if not str(work).isascii():
        ap.error(f"the runtime home path must be ASCII-only (the work path is written into G-code header "
                 f"comments, and G-code comments must be pure ASCII), got {home}")

    home.mkdir(parents=True, exist_ok=True)
    toml = home / "vpype.toml"
    if not toml.exists():
        shutil.copy2(ROOT / "vpype.toml", toml)
        print(f"copied vpype.toml into {home}")

    env = dict(os.environ)
    env["CCPLOTTER_HOME"] = str(home)
    # An absolute work directory: the app serves /work/<file> with send_from_directory, which resolves a
    # RELATIVE directory against the source tree rather than the working directory (previews and G-code
    # downloads would 404). The generated G-code header names this path in a comment.
    env["CCPLOTTER_WORK_DIR"] = str(work)
    env["CCPLOTTER_DEV_PORT"] = str(args.port)
    env["CCPLOTTER_DEV_HOST"] = args.host
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    print(f"runtime home: {home}\nserving on http://{args.host}:{args.port}/", flush=True)
    try:
        return subprocess.call([sys.executable, str(ROOT / "app.py")], cwd=str(home), env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
