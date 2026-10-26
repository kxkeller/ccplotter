"""Run every test file in its own pytest process with an isolated runtime home.

Several test files import ccplotter at module scope and mutate process-wide
state (environment, cwd, module globals), so running the whole suite in one
pytest process is not supported. This runner gives each file:

  * its own process,
  * a fresh, empty, absolute CCPLOTTER_HOME (and TMP/TEMP) under a scratch
    directory that is removed afterwards,
  * a wall-clock timeout.

Usage (from the repository root, with the dependencies installed):

    python scripts/run_tests.py                       # all tests/test_*.py, one at a time
    python scripts/run_tests.py -j 4                  # four files at once
    python scripts/run_tests.py tests/test_pen_fill.py tests/test_stitch.py
    python scripts/run_tests.py -k twopen             # files whose name contains "twopen"
    python scripts/run_tests.py --log-dir some/dir    # keep per-file logs there

Exit status is 0 only if every selected file passed (skips are not failures).
Some files take several minutes; the slowest take over twenty.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SUMMARY_RE = re.compile(r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed|deselected)")


def _counts(text: str) -> dict:
    """Parse pytest's final summary line ("3 failed, 12 passed, 1 skipped in 4s")."""
    last = ""
    for line in text.splitlines():
        if re.search(r"\b(passed|failed|skipped|error|errors|no tests ran)\b.* in [\d.]+s", line):
            last = line
    out = {}
    for n, what in _SUMMARY_RE.findall(last):
        out[what.rstrip("s") if what == "errors" else what] = int(n)
    return out


def run_one(path: Path, scratch: Path, log_dir: Path, timeout: int, python: str) -> dict:
    name = path.stem
    home = scratch / name / "home"
    tmp = scratch / name / "tmp"
    home.mkdir(parents=True)
    tmp.mkdir(parents=True)
    env = dict(os.environ)
    env["CCPLOTTER_HOME"] = str(home.resolve())
    env["TMP"] = env["TEMP"] = env["TMPDIR"] = str(tmp.resolve())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("CCPLOTTER_PRINTER_HOST", None)  # tests must not depend on a configured printer
    log = log_dir / f"{name}.log"
    t0 = time.time()
    try:
        r = subprocess.run(
            [python, "-m", "pytest", str(path), "-q", "-p", "no:cacheprovider", "-rfEs"],
            cwd=str(ROOT), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout)
        rc, text = r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc, text = -1, f"TIMEOUT after {timeout}s\n" + str(e.stdout or "")
    secs = time.time() - t0
    log.write_text(text, encoding="utf-8")
    counts = _counts(text)
    if rc == 0:
        status = "PASS"
    elif rc == -1:
        status = "TIMEOUT"
    elif rc == 5:
        status = "NOTESTS"
    else:
        status = "FAIL"
    return {"file": path.name, "status": status, "rc": rc, "secs": secs,
            "counts": counts, "log": log}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="test files (default: tests/test_*.py)")
    ap.add_argument("-k", dest="match", default="", help="only files whose name contains this text")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="files to run concurrently (default 1)")
    ap.add_argument("--timeout", type=int, default=3600, help="per-file timeout in seconds (default 3600)")
    ap.add_argument("--log-dir", default="", help="directory for per-file logs (default: a temp dir)")
    ap.add_argument("--python", default=sys.executable, help="interpreter to run pytest with")
    args = ap.parse_args(argv)

    if args.files:
        files = [Path(f).resolve() for f in args.files]
    else:
        files = sorted((ROOT / "tests").glob("test_*.py"))
    if args.match:
        files = [f for f in files if args.match in f.name]
    if not files:
        print("no test files selected", file=sys.stderr)
        return 2

    scratch = Path(tempfile.mkdtemp(prefix="ccplotter-tests-")).resolve()
    log_dir = Path(args.log_dir).resolve() if args.log_dir else scratch / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} file(s); logs in {log_dir}", flush=True)

    results = []
    try:
        with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futs = [ex.submit(run_one, f, scratch, log_dir, args.timeout, args.python) for f in files]
            for fut in cf.as_completed(futs):
                res = fut.result()
                results.append(res)
                print(f"  {res['status']:<7} {res['file']}  ({res['secs']:.0f}s)", flush=True)
    finally:
        # Remove the per-file homes/temp dirs; keep the logs (they may live in scratch/logs).
        for child in list(scratch.iterdir()):
            if child.name != "logs":
                shutil.rmtree(child, ignore_errors=True)

    results.sort(key=lambda r: r["file"])
    width = max(len(r["file"]) for r in results)
    print()
    print(f"{'file':<{width}}  {'status':<7} {'passed':>6} {'failed':>6} {'skipped':>7} {'error':>5} {'secs':>6}")
    tot = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for r in results:
        c = r["counts"]
        for k in tot:
            tot[k] += c.get(k, 0)
        print(f"{r['file']:<{width}}  {r['status']:<7} {c.get('passed', 0):>6} {c.get('failed', 0):>6} "
              f"{c.get('skipped', 0):>7} {c.get('error', 0):>5} {r['secs']:>6.0f}")
    bad = [r for r in results if r["status"] != "PASS"]
    print(f"{'TOTAL':<{width}}  {len(results) - len(bad)}/{len(results)} files pass; "
          f"{tot['passed']} passed, {tot['failed']} failed, {tot['skipped']} skipped, {tot['error']} errors")
    for r in bad:
        print(f"  not passing: {r['file']} [{r['status']}] -> {r['log']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
