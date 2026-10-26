"""replay_harness.py `record --only LABEL`.

Runs cmd_record(only=...) against a TEMP COPY of a synthetic manifest, never the real one, with the
server lifecycle and the fixture run monkeypatched out (no server, no heavy job). Pins:
  (a) an unknown label is refused and nothing is written;
  (b) a label already in the manifest is refused and the manifest is untouched;
  (c) a successful add leaves every pre-existing entry byte-identical (text of the file up to and
      including them), writes the .bak-before-only-<label> backup equal to the old manifest, keeps
      the top-level keys, and uses the existing file's indent.
Each refusal/success has a control: a run whose determinism check fails writes nothing."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts import replay_harness as rh  # noqa: E402

OLD = {"recorded_at": "2026-01-01T00:00:00", "app_sha": "abc1234", "ignored_fields": ["cached"],
       "fixtures": {"alpha": {"gcode_sha256": "aa", "n": [1, 2, 3]},
                    "zeta": {"gcode_sha256": "zz", "nested": {"k": "v"}}}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    m = tmp_path / "replay_manifest.json"
    m.write_text(json.dumps(OLD, indent=2, sort_keys=True))     # indent 2, like the real manifest
    calls = {"run": 0, "server": 0}
    state = {"flip": False}

    def fake_run(fx):
        calls["run"] += 1
        n = calls["run"] + (1 if state["flip"] and calls["run"] % 2 == 0 else 0)
        return {"kind": "process", "http_status": 200, "gcode_sha256": f"new-{fx['label']}-{n}"}

    monkeypatch.setattr(rh, "FIXTURES", [{"label": "beta", "kind": "process"}, {"label": "alpha", "kind": "process"}])
    monkeypatch.setattr(rh, "start_server", lambda: calls.__setitem__("server", calls["server"] + 1))
    monkeypatch.setattr(rh, "stop_server", lambda proc: None)
    monkeypatch.setattr(rh, "check_pen_context", lambda: None)
    monkeypatch.setattr(rh, "run_fixture", fake_run)
    # constant result per call index parity -> run1 == run2 only if not flipped
    monkeypatch.setattr(rh, "fixture_diff", lambda label, a, b, **k:
                        [] if a["gcode_sha256"].rsplit("-", 1)[0] == b["gcode_sha256"].rsplit("-", 1)[0]
                        and a == b else ["diff"])
    return m, calls, state


def _fake_equal_runs(monkeypatch):
    monkeypatch.setattr(rh, "run_fixture",
                        lambda fx: {"kind": "process", "http_status": 200, "gcode_sha256": f"new-{fx['label']}"})


def test_unknown_label_refused_and_nothing_written(env, tmp_path):
    m, calls, _ = env
    before = m.read_bytes()
    assert rh.cmd_record(m, only="nonexistent") == 2
    assert m.read_bytes() == before
    assert calls["server"] == 0 and calls["run"] == 0
    assert not list(tmp_path.glob("*.bak-before-only-*"))


def test_already_present_label_refused_and_manifest_untouched(env, tmp_path):
    m, calls, _ = env
    before = m.read_bytes()
    assert rh.cmd_record(m, only="alpha") == 2          # alpha is in FIXTURES AND in the manifest
    assert m.read_bytes() == before
    assert calls["server"] == 0 and calls["run"] == 0
    assert not list(tmp_path.glob("*.bak-before-only-*"))


def test_add_keeps_existing_entries_byte_identical_and_writes_backup(env, tmp_path, monkeypatch):
    m, calls, _ = env
    _fake_equal_runs(monkeypatch)
    old_bytes = m.read_bytes()
    assert rh.cmd_record(m, only="beta") == 0
    bak = tmp_path / "replay_manifest.json.bak-before-only-beta"
    assert bak.read_bytes() == old_bytes
    new = json.loads(m.read_text())
    assert set(new["fixtures"]) == {"alpha", "beta", "zeta"}
    assert new["fixtures"]["beta"]["gcode_sha256"] == "new-beta"
    assert {k: v for k, v in new.items() if k != "fixtures"} == {k: v for k, v in OLD.items() if k != "fixtures"}
    # byte identity of the untouched entries: re-dump each old entry with the file's indent (2) and
    # require the exact text to appear in the new file, and the old file's text outside "beta" to match
    new_text = m.read_text()
    for k, v in OLD["fixtures"].items():
        body = json.dumps(v, indent=2, sort_keys=True).replace("\n", "\n    ")
        assert f'"{k}": {body}' in new_text, k
    removed = [l for l in __import__("difflib").unified_diff(
        old_bytes.decode().splitlines(), new_text.splitlines(), lineterm="", n=0)
        if l.startswith("-") and not l.startswith("---")]
    assert removed == []                                  # a pure addition: no old line changed


def test_control_nondeterministic_fixture_writes_nothing(env, tmp_path):
    m, calls, state = env
    state["flip"] = True                                  # run 2 differs from run 1
    before = m.read_bytes()
    assert rh.cmd_record(m, only="beta") == 2
    assert m.read_bytes() == before
    assert not list(tmp_path.glob("*.bak-before-only-*"))
