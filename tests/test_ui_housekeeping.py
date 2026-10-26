"""UI housekeeping: hidden-from-UI items stay in code.
Run: py -3.12 -m pytest tests/test_ui_housekeeping.py  (one file)."""
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home
_test_home.use_temp_home("ccplotter_test_home_uihouse_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_uihouse_"))
sys.path.insert(0, str(REPO))

import ccplotter.routes.pen  # noqa: F401
from ccplotter.routes.core import app
from ccplotter.calibration_registry import SHEETS, generic_sheets

HIDDEN = {"singlepass", "penwidth", "dropout_map", "penwidth_wedge", "fidelity",
          "fillmode", "linestar", "solidpitch", "pocket_preppy", "uright_short",
          "deposition"}
KEPT = {"pocket_bic", "fragment_bic", "photofe_bic", "plot", "pack", "combo", "block"}
HTML = (REPO / "assets" / "templates" / "index.html").read_text(encoding="utf-8")


def _ids(url):
    return {s["id"] for s in app.test_client().get(url).get_json()["sheets"]}


def test_listing_excludes_hidden_includes_kept():
    ids = _ids("/api/calibration_sheets")
    assert not (ids & HIDDEN), ids & HIDDEN
    assert KEPT <= ids, KEPT - ids


def test_hidden_still_registered_and_listed_with_all():
    assert HIDDEN <= set(SHEETS)
    assert HIDDEN <= _ids("/api/calibration_sheets?all=1")


def test_hidden_sheet_generation_route_still_200():
    tried = []
    for s in generic_sheets():
        if s.id not in HIDDEN:
            continue
        r = app.test_client().post("/api/calibration_sheet", json={"id": s.id})
        tried.append((s.id, r.status_code))
        if r.status_code == 200:
            return
    raise AssertionError(f"no hidden generic sheet generated: {tried}")


def test_index_html_keeps_photo512_option_hidden_attr():
    m = re.search(r'<option value="photo512"([^>]*)>', HTML)
    assert m, "photo512 option removed"
    assert "hidden" in m.group(1).split()


def test_index_html_hidden_panels_still_in_dom():
    for needle in ('id="ai-panel"', 'id="btn-cov-debug"', 'id="btn-gap-audit"',
                   'id="cov-status"', 'id="gap-status"', 'onclick="genDepositionSheet()"'):
        assert needle in HTML, needle


def test_photo512_not_restorable_default_is_tracev3():
    """No restore path re-selects photo512. The page persists only
    'ccplotter_registered_spot' and 'ccplotter_paper' in localStorage (no
    style key), so the style select always loads at its HTML default."""
    keys = set(re.findall(r"localStorage\.setItem\('([^']+)'", HTML))
    assert keys == {"ccplotter_registered_spot", "ccplotter_paper"}, keys
    sel = re.search(r"<select[^>]*>(?:(?!</select>).)*value=\"photo512\".*?</select>", HTML, re.S).group(0)
    selected = re.findall(r'<option value="([a-z0-9_]+)"[^>]*\bselected\b', sel)
    assert selected == ["tracev3"], selected
