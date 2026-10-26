"""WARNING when a tracev3-engine item's REALISED pass pitch is wider than the pen's fill_mm
(or its measured laid_mm), keyed ONLY on stats["realised_pitch"], deduplicated with the
Trace V3+ pitch-vs-fill danger.

Two layers:
  * stats-level (build_warnings on a hand-built stats dict): exact text, threshold + tolerance with
    controls, laid_mm reference, multi-item, dedupe, "no block -> byte-identical warnings";
  * through the real Flask route (BIC blue pinned): fires at a placement where the pitch is wider
    than 0.12 mm, silent at a placement where it is not, silent for a non-tracev3 style, one message
    (not two) for Trace V3+ where the pitch-vs-fill danger also fires.

Direction note (measured with BIC blue on a 120x120 page): realised pitch GROWS with placed size
(scale 1.0 -> 0.265 mm, scale 0.4 -> 0.106 mm, scale 0.15 -> 0.091 mm), so the advice in the text is
"plot smaller", not "plot larger".
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home BEFORE any ccplotter import
_test_home.use_temp_home("ccplotter_test_home_rpwarn_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_rpwarn_"))
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
from PIL import Image

import ccplotter.routes.runs  # noqa: F401
from ccplotter.config import WORK_DIR
from ccplotter.stats_warnings import (REALISED_PITCH_TOL_MM, build_warnings, illustration_pitch_warning,
                                      realised_pitch_warning, realised_pitch_warnings)

PEN = "bic_cristal_blue"            # seed fill_mm 0.12, no laid_mm -- pinned, never the ambient active pen
FILL = 0.12
TEXT_FILL = ("Passes {p} mm apart on this page, wider than this pen's fill {f} mm -- expect white slivers "
             "between passes; plot smaller or use a coarser pen.")


def _item(iid, pitch, fill=FILL):
    return {"id": iid, "realised_pitch_mm": pitch, "fill_mm_target": fill}


def _stats(*items, d1014=None):
    st = {"stroke_count": 10, "estimated_z_moves": 20, "item_geometry": {}}
    if items:
        vals = [i["realised_pitch_mm"] for i in items]
        st["realised_pitch"] = {"items": list(items), "min_mm": min(vals), "max_mm": max(vals), "basis": "x"}
    for iid, (p, f) in (d1014 or {}).items():
        st["item_geometry"][iid] = {"scale_aware_equivalence_prelim_approx": {
            "illustration_pitch_mm": p, "illustration_fill_mm": f}}
    return st


def _pitch_msgs(warnings):
    return [w for w in warnings if w["msg"].startswith("Passes ") or w["msg"].startswith("Item ")
            or w["msg"].startswith("Trace V3+: passes")]


# ------------------------------------------------------------------ stats level

def test_fires_with_exact_text_on_the_smoke_numbers():
    """The smoke case: a BIC tracev3 block that realised 0.1855 mm at fill 0.12."""
    w = build_warnings(_stats(_item("a", 0.1855)))
    pw = _pitch_msgs(w)
    assert len(pw) == 1
    assert pw[0]["level"] == "warning" and pw[0]["tier"] == "primary"
    exact = TEXT_FILL.format(p=f"{0.1855:.3f}", f="0.120")
    assert pw[0]["msg"] == exact
    assert exact.startswith("Passes 0.18") and "0.120 mm" in exact and exact.endswith("plot smaller or use a coarser pen.")
    # the helper is the single source of the text
    assert pw[0]["msg"] == realised_pitch_warning(0.1855, 0.12)


def test_literal_text_is_pinned():
    # one fully literal string, so a reworded constant cannot slip through the format-based check above
    assert realised_pitch_warning(0.265, 0.12) == (
        "Passes 0.265 mm apart on this page, wider than this pen's fill 0.120 mm -- expect white slivers "
        "between passes; plot smaller or use a coarser pen.")


def test_silent_at_normal_placement_and_at_equality_with_controls():
    # CONTROL (the same stats shape fires when the pitch is wide) ...
    assert len(_pitch_msgs(build_warnings(_stats(_item("a", 0.1855))))) == 1
    # ... and is silent below fill, at equality, and inside the stated tolerance (rounding noise)
    assert REALISED_PITCH_TOL_MM == 1e-4
    for p in (0.0906, 0.106, 0.1089, 0.12, 0.12 + REALISED_PITCH_TOL_MM * 0.5, 0.12 + REALISED_PITCH_TOL_MM):
        assert _pitch_msgs(build_warnings(_stats(_item("a", p)))) == [], p
    # the tolerance does not hide a real excess: just past it fires
    assert len(_pitch_msgs(build_warnings(_stats(_item("a", 0.12 + 2 * REALISED_PITCH_TOL_MM))))) == 1


def test_silent_without_the_realised_pitch_block_and_other_warnings_unchanged():
    base = build_warnings(_stats())
    assert _pitch_msgs(base) == []
    # a block that is silent leaves the whole warning list byte-identical to having no block at all
    silent = build_warnings(_stats(_item("a", 0.1089)))
    assert silent == base
    # CONTROL: a firing block changes the list by exactly one appended-in-place warning
    fired = build_warnings(_stats(_item("a", 0.1855)))
    assert len(fired) == len(base) + 1 and [w for w in fired if w not in base] == _pitch_msgs(fired)
    # empty / malformed blocks and items without a reference do not raise or fire
    for st in ({"stroke_count": 1, "estimated_z_moves": 2, "realised_pitch": {"items": []}},
               {"stroke_count": 1, "estimated_z_moves": 2, "realised_pitch": None},
               _stats({"id": "a", "realised_pitch_mm": 0.5, "fill_mm_target": None}),
               _stats({"id": "a", "realised_pitch_mm": None, "fill_mm_target": 0.12})):
        assert _pitch_msgs(build_warnings(st)) == []


def test_laid_mm_replaces_fill_as_the_reference():
    laid = 0.255
    it = _item("a", 0.27, fill=0.12)
    msgs = realised_pitch_warnings(_stats(it), laid_mm=laid)
    assert msgs == ["Passes 0.270 mm apart on this page, wider than this pen's measured line width 0.255 mm -- "
                    "expect white slivers between passes; plot smaller or use a coarser pen."]
    # CONTROL 1: a pitch above fill but within laid is fine for a laid pen (the pitch is aimed at laid)
    assert realised_pitch_warnings(_stats(_item("a", 0.20, fill=0.12)), laid_mm=laid) == []
    # CONTROL 2: the same item with no laid_mm is judged against fill and fires
    assert len(realised_pitch_warnings(_stats(_item("a", 0.20, fill=0.12)))) == 1
    # CONTROL 3: at / within tolerance of laid is silent
    assert realised_pitch_warnings(_stats(_item("a", laid + REALISED_PITCH_TOL_MM)), laid_mm=laid) == []
    # build_warnings carries it through
    assert len(_pitch_msgs(build_warnings(_stats(it), laid_mm=laid))) == 1
    assert _pitch_msgs(build_warnings(_stats(_item("a", 0.20, fill=0.12)), laid_mm=laid)) == []


def test_multi_item_names_the_item_single_item_does_not():
    st = _stats(_item("one", 0.10), _item("two", 0.30))
    msgs = [w["msg"] for w in _pitch_msgs(build_warnings(st))]
    assert msgs == ["Item two: " + realised_pitch_warning(0.30, 0.12)]
    # CONTROL: one item only -> the plain sentence, no prefix
    assert [w["msg"] for w in _pitch_msgs(build_warnings(_stats(_item("two", 0.30))))] == [realised_pitch_warning(0.30, 0.12)]


# ------------------------------------------------------------------ dedupe with the pitch-vs-fill danger

def test_dedupe_with_d1014_one_message_the_stronger():
    both = build_warnings(_stats(_item("1", 0.1357), d1014={"1": (0.1357, 0.12)}))
    pm = _pitch_msgs(both)
    assert len(pm) == 1 and pm[0]["level"] == "danger"
    assert pm[0]["msg"] == illustration_pitch_warning(0.1357, 0.12)  # pitch-vs-fill text, byte-identical
    # CONTROL A: without the pitch-vs-fill fields the new warning is the one that fires
    only_new = _pitch_msgs(build_warnings(_stats(_item("1", 0.1357))))
    assert len(only_new) == 1 and only_new[0]["level"] == "warning"
    # CONTROL B: the pitch-vs-fill danger fires for item 1 but item 2 (not covered by it) still gets the new warning
    two = _pitch_msgs(build_warnings(_stats(_item("1", 0.1357), _item("2", 0.30), d1014={"1": (0.1357, 0.12)})))
    assert [(w["level"], w["msg"]) for w in two] == [
        ("danger", illustration_pitch_warning(0.1357, 0.12)),
        ("warning", "Item 2: " + realised_pitch_warning(0.30, 0.12))]
    # CONTROL C: pitch-vs-fill present but NOT firing (pitch held) does not suppress a wider realised pitch
    held = _pitch_msgs(build_warnings(_stats(_item("1", 0.1855), d1014={"1": (0.108, 0.12)})))
    assert len(held) == 1 and held[0]["level"] == "warning"


# ------------------------------------------------------------------ through the real route

def _block(path, w=1100, h=800):
    img = np.full((h, w), 255, np.uint8)
    img[200:600, 300:800] = 20
    Image.fromarray(img).convert("RGB").save(path, quality=95)


def _run(did, style, paper="120x120", scale=1.0, job_type="preview", pen=PEN):
    from ccplotter.routes.core import app
    _block(WORK_DIR / f"src_{did}.jpg")
    body = {"job_type": job_type, "paper": paper, "margin": 10.0, "style": style, "pen_id": pen,
            "items": [{"id": did, "scale": scale, "rot": 0, "dx": 0, "dy": 0, "crop": None}]}
    resp = app.test_client().post("/api/process", json=body)
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


def test_route_fires_where_pitch_exceeds_fill_and_is_silent_where_it_does_not():
    wide = _run("rw_wide", "tracev3", scale=1.0)          # measured 0.265 mm on BIC blue (fill 0.12)
    it = wide["realised_pitch"]["items"][0]
    assert it["fill_mm_target"] == pytest.approx(FILL) and it["realised_pitch_mm"] > FILL + 0.01
    pm = _pitch_msgs(wide["warnings"])
    assert len(pm) == 1 and pm[0]["level"] == "warning"
    assert pm[0]["msg"] == realised_pitch_warning(it["realised_pitch_mm"], it["fill_mm_target"])
    assert f"{it['realised_pitch_mm']:.3f}" in pm[0]["msg"] and "0.120" in pm[0]["msg"]
    # CONTROL: the same job placed smaller realises a pitch within fill -> no message
    narrow = _run("rw_narrow", "tracev3", scale=0.4)       # measured 0.106 mm
    assert narrow["realised_pitch"]["items"][0]["realised_pitch_mm"] <= FILL
    assert _pitch_msgs(narrow["warnings"]) == []


def test_route_silent_for_non_tracev3_style():
    j = _run("rw_plain", "plotready")
    assert not j.get("realised_pitch")
    assert _pitch_msgs(j["warnings"]) == []


def test_route_dedupes_with_d1014_on_trace_v3_plus(monkeypatch):
    """Unclamped Trace V3+ at a big placement: the pitch-vs-fill danger fires AND the realised pitch exceeds fill; exactly one
    pitch message (the danger) comes back. Control: plain tracev3 at a comparable size gets the new warning."""
    import ccplotter.routes.core as core
    from ccplotter.config import ILLUSTRATION_RECIPE
    monkeypatch.setattr(core, "ILLUSTRATION_RECIPE",
                        {k: v for k, v in ILLUSTRATION_RECIPE.items() if k != "illustration_pitch_clamp"})
    j = _run("rw_il", "illustration", paper="600x600")
    it = j["realised_pitch"]["items"][0]
    assert it["realised_pitch_mm"] > it["fill_mm_target"] + REALISED_PITCH_TOL_MM     # the new check WOULD fire
    pm = _pitch_msgs(j["warnings"])
    assert len(pm) == 1 and pm[0]["level"] == "danger" and pm[0]["msg"].startswith("Trace V3+: passes")
    assert not [w for w in j["warnings"] if w["msg"].startswith("Passes ")]
    monkeypatch.undo()
    t = _run("rw_il_ctrl", "tracev3", scale=1.0)
    assert [w["level"] for w in _pitch_msgs(t["warnings"])] == ["warning"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
