"""The dev server binds loopback by default; the CCPLOTTER_DEV_HOST override is honoured."""
import ast
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home
_test_home.use_temp_home("ccplotter_test_home_devhost_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_devhost_"))
sys.path.insert(0, str(REPO))

import app as ccapp


def test_default_is_loopback(monkeypatch):
    monkeypatch.delenv("CCPLOTTER_DEV_HOST", raising=False)
    assert ccapp.resolve_dev_host() == "127.0.0.1"
    monkeypatch.setenv("CCPLOTTER_DEV_HOST", "")
    assert ccapp.resolve_dev_host() == "127.0.0.1"


def test_override_honoured(monkeypatch):
    monkeypatch.setenv("CCPLOTTER_DEV_HOST", "0.0.0.0")
    assert ccapp.resolve_dev_host() == "0.0.0.0"       # positive control: the override really changes the result
    monkeypatch.setenv("CCPLOTTER_DEV_HOST", "192.168.1.5")
    assert ccapp.resolve_dev_host() == "192.168.1.5"


def test_main_block_has_no_literal_all_interfaces_and_uses_helper():
    src = (REPO / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    main = next(n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.dump(n.test))
    seg = ast.get_source_segment(src, main)
    assert "0.0.0.0" not in seg and "resolve_dev_host()" in seg
