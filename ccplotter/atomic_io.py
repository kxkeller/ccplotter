"""ccplotter/atomic_io.py -- atomic writes for work/ pipeline outputs.

A shorter document written directly over a longer one, without the file
being truncated or replaced atomically, leaves the old tail past the new EOF:
work/layout.routed.svg would then hold a complete </svg> followed by leftover
polyline coordinates from the older file, and the gcode step fails with "junk
after document element". More generally, ANY interrupted or non-atomic write
to a work/ output can leave a partial or hybrid file that a later stage (or
the frontend, serving /work/<file> mid-write) reads as valid.

Rule: never write a pipeline output IN PLACE. Always write to a temp file in
the SAME directory as the target (same filesystem -- required for os.replace
to be atomic), then os.replace() over the target. os.replace is atomic on
both Windows and POSIX: a reader either sees the complete OLD file or the
complete NEW file, never a partial/hybrid one. A crash or exception mid-write
leaves only the orphaned temp file -- the target is untouched. A shorter new
document fully REPLACES a longer old one (no trailing garbage possible).

Four shapes, matching how this codebase actually writes files. All four are
byte-identical to the direct-write call they replace -- only the PLACEMENT
becomes atomic, nothing about the written bytes changes:

  - atomic_write_text/atomic_write_bytes: we already hold the data as a
    str/bytes (e.g. ``"\\n".join(lines) + "\\n"``) and would otherwise have
    called Path.write_text/write_bytes directly.
  - atomic_open: we open a file object ourselves and hand it to a library
    that writes into it (e.g. ``vp.write_svg(f, doc, ...)``).
  - atomic_target: we hand a PATH STRING to a writer we do not control at
    all (vpype_cli's `write`/`gwrite` CLI commands, svgwrite.Drawing(path))
    -- yields a temp Path to pass to that writer instead of the real target,
    then replaces on clean exit. This is the "writer we don't control"
    case: we never intercept inside the library, we
    just choose where it writes and replace afterward.
  - atomic_copy: shutil.copy's source-file-to-new-path idiom, made atomic at
    the destination the same way.
"""
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path


def _tmp_path_for(path) -> Path:
    """A temp path in the SAME directory as `path` (same filesystem --
    required for os.replace to be atomic), with a recognizable prefix
    (".<stem>.atomictmp-<pid>-<random><suffix>") so a leftover from a crash
    before the replace is easy to spot and can never be mistaken for -- or
    read as -- a real pipeline output (its name doesn't match anything a
    reader globs for). The ORIGINAL suffix is kept on the end (not dropped)
    because several writers we hand this path to (vpype's `write`/`gwrite`
    CLI commands, cv2.imwrite) pick their output format from the file
    extension -- an extensionless temp name made vpype/cv2 fail to write
    anything at all, which os.replace then reported as "source not found"
    (caught in the byte-identity harness before this ever reached a real
    write path)."""
    path = Path(path)
    return path.parent / f".{path.stem}.atomictmp-{os.getpid()}-{uuid.uuid4().hex[:8]}{path.suffix}"


def _cleanup(tmp: Path) -> None:
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass


def atomic_write_bytes(path, data: bytes) -> None:
    """Same result as Path(path).write_bytes(data), placed atomically."""
    path = Path(path)
    tmp = _tmp_path_for(path)
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        _cleanup(tmp)


def atomic_write_text(path, text: str, encoding=None, errors=None, newline=None) -> None:
    """Same result as Path(path).write_text(text, encoding, errors, newline)
    -- including default text-mode newline translation when `newline` is not
    given -- placed atomically."""
    path = Path(path)
    tmp = _tmp_path_for(path)
    try:
        with open(tmp, "w", encoding=encoding, errors=errors, newline=newline) as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        _cleanup(tmp)


@contextmanager
def atomic_open(path, mode="w", **kwargs):
    """``with atomic_open(path) as f: lib.write(f, ...)`` -- open a temp file
    in the same directory as `path` for writing, hand the caller the file
    object, and atomically replace `path` with it once the `with` block
    exits cleanly. Mirrors ``open(path, mode, **kwargs)`` exactly (mode /
    encoding / newline), so the written bytes are unchanged -- only the
    placement becomes atomic. On any exception inside the `with` block,
    `path` is left completely untouched and the temp file is removed."""
    path = Path(path)
    tmp = _tmp_path_for(path)
    f = open(tmp, mode, **kwargs)
    try:
        yield f
        f.close()
        os.replace(tmp, path)
    finally:
        if not f.closed:
            try:
                f.close()
            except Exception:
                pass
        _cleanup(tmp)


@contextmanager
def atomic_target(path):
    """``with atomic_target(path) as tmp: external_writer(str(tmp))`` -- for
    writers we do not control at all (vpype_cli's `write`/`gwrite` CLI
    commands, svgwrite.Drawing(path)) that take a destination PATH STRING and
    do their own internal open()/write(). Yields a temp Path in the same
    directory as `path`; atomically replaces `path` with it once the `with`
    block exits cleanly (the external writer has by then produced a complete
    file at `tmp`). A crash or exception mid-write leaves only the orphaned
    temp file -- `path` is untouched."""
    path = Path(path)
    tmp = _tmp_path_for(path)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        _cleanup(tmp)


def atomic_copy(src, dst) -> None:
    """Same result as shutil.copy(src, dst), with the destination placed
    atomically (dst is always a brand-new path in this codebase's call
    sites, but a concurrent reader of that same path -- e.g. two requests
    racing on a freshly-minted id -- still can't observe a partial copy)."""
    dst = Path(dst)
    tmp = _tmp_path_for(dst)
    try:
        shutil.copy(src, tmp)
        os.replace(tmp, dst)
    finally:
        _cleanup(tmp)
