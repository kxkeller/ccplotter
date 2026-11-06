"""Ink-routed joins checkbox (2026-10-29, Kai "2"): UI for the experimental request key ink_routed_connectors.
Static checks on assets/templates/index.html + the real inkRouteActive()/inkRouteStyleOk() executed in node against a stubbed ge().
Run: pytest tests/test_inkroute_checkbox_ui.py  (one file). The headless-browser check lives in artifacts/inkroute-checkbox-ui-2026-10-29/."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HTML = (REPO / "assets" / "templates" / "index.html").read_text(encoding="utf-8")
NODE = shutil.which("node")


def test_checkbox_present_unchecked_by_default_with_label_and_tooltip():
    m = re.search(r'<input type="checkbox" id="ink-routed-connectors"([^>]*)>', HTML)
    assert m, "checkbox missing"
    assert "checked" not in m.group(1)
    assert "Route pen-down joins through ink (experimental)" in HTML
    lab = re.search(r'<label[^>]*title="([^"]*)"[^>]*>\s*<input type="checkbox" id="ink-routed-connectors"', HTML)
    assert lab and lab.group(1) == ("Off by default. Experimental. Keeps the pen down across short gaps by routing through ink already drawn, never over bare paper; "
                                    "fewer lifts; can thicken outlines by up to half a pen width; shown in the preview; adds routing time to the preview.")
    assert re.search(r'<div class="field" id="ink-routed-field" style="display:none">', HTML), "field must start hidden"


def test_not_persisted_and_buildbody_gates_on_inkrouteactive():
    assert "ink-routed" not in "".join(re.findall(r"localStorage\.(?:setItem|getItem)\('([^']+)'", HTML))
    assert re.search(r"\.\.\.\(inkRouteActive\(\) \? \{ ink_routed_connectors: true \} : \{\}\)", HTML)
    assert len(re.findall(r"ink_routed_connectors\s*:", HTML)) == 1, "the key must be built in exactly one place"


def _active(style, checked):
    fns = "".join(re.search(r"function %s\(\).*?\n" % n, HTML).group(0) for n in ("inkRouteStyleOk", "inkRouteActive"))
    js = (fns + "\nconst V = " + json.dumps({"img-style": {"value": style}, "ink-routed-connectors": {"checked": checked}}) +
          ";\nfunction ge(id) { return V[id]; }\nprocess.stdout.write(String(inkRouteActive()));")
    return subprocess.run([NODE, "-e", js], capture_output=True, text=True, check=True).stdout == "true"


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_key_sent_only_when_checked_and_style_honours_it():
    for st in ("tracev3", "bwphoto", "illustration"):
        assert _active(st, True) is True          # positive controls
        assert _active(st, False) is False        # unchecked: key absent
    for st in ("twopenkt", "plotready", "tsp", "sketchfield", "photo512"):
        assert _active(st, True) is False         # twopenkt only forwards the key to its reference job; others ignore it
