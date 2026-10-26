"""scripts/replay_harness.py takes an exclusive lock before starting its server.

Without the lock, two concurrent `verify` runs race for :5001: the second one's
ensure_port_free() would taskkill the first one's live server, killing it on its
first fixture. A held lock, or a port that is already bound, is a loud refusal
(HarnessBusy / exit code 3), never a kill, and the lock is released when the server is
stopped. Every refusal test is paired with a control showing the same path proceeds
once the lock is released (a refusal that can never be lifted would also "pass").

No real dev server is started here; start_server's inner launch is stubbed.
"""
import os
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import scripts.replay_harness as h  # noqa: E402  (pins CCPLOTTER_HOME to the harness env)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def lockpath(tmp_path, monkeypatch):
    p = tmp_path / "harness.lock"
    monkeypatch.setattr(h, "harness_lock_path", lambda port=None: p)
    return p


def _py(code, cwd, env=None):
    return subprocess.run([sys.executable, "-c", code], cwd=str(cwd), capture_output=True,
                          text=True, env=env or os.environ.copy())


def test_second_acquire_refused_then_proceeds_after_release(lockpath):
    got = h.acquire_harness_lock(lockpath)
    assert got == lockpath and lockpath.is_file()
    with pytest.raises(h.HarnessBusy) as e:
        h.acquire_harness_lock(lockpath)
    assert str(os.getpid()) in str(e.value) and "REFUSING" in str(e.value)
    h.release_harness_lock(lockpath)
    assert not lockpath.exists()
    # control: after release the same call proceeds
    assert h.acquire_harness_lock(lockpath) == lockpath
    h.release_harness_lock(lockpath)


def test_other_process_is_refused_while_held_and_proceeds_after(lockpath, tmp_path):
    code = ("import sys; sys.path.insert(0, %r); import scripts.replay_harness as h; "
            "from pathlib import Path\n"
            "try:\n h.acquire_harness_lock(Path(%r)); print('ACQUIRED'); "
            "h.release_harness_lock(Path(%r))\n"
            "except h.HarnessBusy as e:\n print('BUSY', e)\n") % (str(REPO), str(lockpath), str(lockpath))
    h.acquire_harness_lock(lockpath)
    try:
        r = _py(code, tmp_path)
        assert "BUSY" in r.stdout and "ACQUIRED" not in r.stdout, (r.stdout, r.stderr[-500:])
    finally:
        h.release_harness_lock(lockpath)
    r2 = _py(code, tmp_path)                       # control: released -> proceeds
    assert "ACQUIRED" in r2.stdout and "BUSY" not in r2.stdout, (r2.stdout, r2.stderr[-500:])


def test_stale_lock_with_dead_pid_is_reclaimed(lockpath):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    lockpath.write_text(f"{dead.pid} 2026-01-01 00:00:00")
    assert h.acquire_harness_lock(lockpath) == lockpath
    assert h._read_lock(lockpath)[0] == os.getpid()
    h.release_harness_lock(lockpath)


def test_release_never_deletes_someone_elses_lock(lockpath):
    lockpath.write_text("999999 2026-01-01 00:00:00")
    h.release_harness_lock(lockpath)               # not ours
    assert lockpath.exists()


def test_start_server_refuses_when_lock_held_and_never_touches_the_port(lockpath, monkeypatch):
    port = _free_port()
    monkeypatch.setattr(h, "PORT", port)
    called = []
    monkeypatch.setattr(h, "_start_server_locked", lambda: called.append(1))
    h.acquire_harness_lock(lockpath)
    try:
        with pytest.raises(h.HarnessBusy):
            h.start_server()
        assert called == [], "start_server went on to launch a server despite the held lock"
    finally:
        h.release_harness_lock(lockpath)
    # control: lock free -> start_server proceeds and HOLDS the lock until stop_server
    dummy = types.SimpleNamespace(terminate=lambda: None, wait=lambda timeout=None: 0)
    monkeypatch.setattr(h, "_start_server_locked", lambda: dummy)
    proc = h.start_server()
    assert proc is dummy and lockpath.is_file()
    with pytest.raises(h.HarnessBusy):
        h.acquire_harness_lock(lockpath)           # a second verify now is refused
    h.stop_server(proc)
    assert not lockpath.exists()                   # released on stop
    assert h.acquire_harness_lock(lockpath) == lockpath
    h.release_harness_lock(lockpath)


def test_bound_port_is_refused_not_killed_and_lock_is_released(lockpath, monkeypatch):
    monkeypatch.delenv("CCPLOTTER_HARNESS_KILL_STALE", raising=False)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    port = srv.getsockname()[1]
    monkeypatch.setattr(h, "PORT", port)
    killed = []
    real_run = subprocess.run

    def spy(argv, *a, **k):
        if argv and str(argv[0]).lower().startswith("taskkill"):
            killed.append(argv)
            return types.SimpleNamespace(returncode=0)
        return real_run(argv, *a, **k)
    monkeypatch.setattr(h.subprocess, "run", spy)
    try:
        with pytest.raises(h.HarnessBusy) as e:
            h.start_server()
        assert "already bound" in str(e.value)
        assert killed == [], "the harness tried to kill a listener it did not start"
        assert not lockpath.exists(), "refusal leaked the lock"
        c = socket.socket()
        assert c.connect_ex(("127.0.0.1", port)) == 0   # the other listener is untouched
        c.close()
    finally:
        srv.close()


def test_cli_exits_3_with_message_when_lock_is_held(tmp_path):
    port = _free_port()
    lock = REPO / "work" / f"_replay_harness_{port}.lock"
    env = os.environ.copy()
    env["CCPLOTTER_HARNESS_PORT"] = str(port)
    lock.parent.mkdir(exist_ok=True)
    lock.write_text(f"{os.getpid()} 2026-10-03 00:00:00")      # held by THIS (live) process
    try:
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "replay_harness.py"), "verify"],
                           cwd=str(tmp_path), capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == 3, (r.returncode, r.stdout[-300:], r.stderr[-500:])
        assert "REFUSING" in r.stderr and str(os.getpid()) in r.stderr
        assert "dev server up" not in r.stdout
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
