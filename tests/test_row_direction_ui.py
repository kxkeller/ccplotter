"""Row direction control (2026-10-29, Kai "sure"): UI for the experimental body key row_angle_rule.
Static checks on assets/templates/index.html + the real rowAngleActive() function executed in node against a stubbed ge().
Run: pytest tests/test_row_direction_ui.py  (one file). The headless-browser check lives in artifacts/row-direction-ui-2026-10-29/."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HTML = (REPO / "assets" / "templates" / "index.html").read_text(encoding="utf-8")


def test_select_present_default_a135_with_the_five_options():
    m = re.search(r'<select id="row-angle-rule"[^>]*>(.*?)</select>', HTML, re.S)
    assert m, "row-angle-rule select missing"
    opts = re.findall(r'<option value="([a-z0-9]+)"([^>]*)>([^<]*)</option>', m.group(1))
    assert [o[0] for o in opts] == ["a135", "a45", "zero", "across", "set"]
    assert [o[2] for o in opts] == ["45 deg, bottom-left to top-right (/)", "45 deg, top-left to bottom-right (\\)",
                                    "As today (0)", "Across short side", "Mixed per area"]
    assert [o[0] for o in opts if "selected" in o[1]] == ["a135"]       # 2026-11-04 default flip (Kai)


def test_field_hidden_by_default_and_tooltip_text():
    m = re.search(r'<label id="row-angle-field"([^>]*)>', HTML)
    assert m and "display:none" in m.group(1)
    assert "Look option for row fills. Coverage is the same in every direction; only the look changes. Experimental." in m.group(1)


def test_not_persisted_and_buildbody_gates_on_rowangleactive():
    assert "row-angle" not in "".join(re.findall(r"localStorage\.(?:setItem|getItem)\('([^']+)'", HTML))
    assert re.search(r"\.\.\.\(rowAngleActive\(\) \? \{ row_angle_rule: ge\('row-angle-rule'\)\.value \} : \{\}\)", HTML)
    assert len(re.findall(r"row_angle_rule\s*:", HTML)) == 1, "the key must be built in exactly one place"


NODE = shutil.which("node")


def _active(style, fill, rule):
    src = re.search(r"function rowAngleActive\(\) \{.*?\n\}", HTML, re.S).group(0)
    js = (src + "\nconst V = " + json.dumps({"img-style": style, "trace-core-fill": fill, "row-angle-rule": rule}) +
          ";\nfunction ge(id) { return { value: V[id] }; }\nprocess.stdout.write(String(rowAngleActive()));")
    return subprocess.run([NODE, "-e", js], capture_output=True, text=True, check=True).stdout == "true"


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_key_sent_explicitly_for_every_rule_with_rows_fills():
    assert _active("bwphoto", "outline_rows", "a45") is True            # positive controls
    assert _active("bwphoto", "rows", "across") is True
    assert _active("tracev3", "rows", "a45") is True
    assert _active("bwphoto", "outline_rows", "a135") is True           # the new default is sent explicitly
    assert _active("bwphoto", "outline_rows", "zero") is True           # "zero" is a CHOICE now: sent explicitly, never absent
    assert _active("bwphoto", "rings", "zero") is False                 # rings: never sent, whatever the select holds
    assert _active("bwphoto", "rings", "a45") is False                  # rings fill: absent
    assert _active("bwphoto", "hybrid", "a45") is False
    assert _active("illustration", "rows", "a45") is False              # recipe pins rings server-side
    assert _active("plotready", "rows", "a45") is False
    assert _active("twopenkt", "outline_rows", "set") is False          # D1913: no key in two-pen until a look check
