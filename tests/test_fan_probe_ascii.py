"""D1906: the emitted fan-probe G-code is pure ASCII (hard rule: a non-ASCII byte crashes the firmware parser), whatever the shell's
encoding mode. Control: the pre-fix comments contained em dashes, which this check would have caught."""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home
_test_home.use_temp_home("ccplotter_test_home_fanprobe_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_fanprobe_"))
sys.path.insert(0, str(REPO))

from ccplotter.gcode_post import _fan_probe_gcode


def test_fan_probe_output_is_pure_ascii_bytes(tmp_path):
    out = _fan_probe_gcode(tmp_path / "fan_probe.gcode")
    data = out.read_bytes()
    assert data.isascii(), [i for i, b in enumerate(data) if b > 127][:5]
    assert b"\xe2\x80\x94" not in data and b"\x97" not in data, "no em dash in any encoding"
    text = data.decode("ascii")
    assert "FAN PROBE -- DISABLED" in text and "M106 P2 S0" in text and "M106 P3 S0" in text
    assert "M106 P0" not in text and "M106 P1" not in text, "only valid fan indices"
    assert "G92" not in text


def test_control_the_old_em_dash_text_would_fail_the_check():
    old = "; ccPlotter FAN PROBE \u2014 DISABLED. Reference only.\n"
    assert not old.encode("utf-8").isascii()
