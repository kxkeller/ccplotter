"""The "Line Art / Illustration" style (opt-in style="illustration").

A real style that reproduces the reference processing for an already-clean drawing
(CLAHE on, contour/rings core fill, small detail off, vec_merge 0.3, resolution 2200),
kept separate from photo mode, with NO photo XDoG front end and NO placement field.
Same dispatch pattern as bwphoto (config recipe merged in routes/core.py before the
vcache key, then style="tracev3"). This file is the standing guard:

  1. the recipe dict is pinned, and carries no front-end key and no placement field;
  2. through the real Flask route, style="illustration" == tracev3 with the recipe values
     spelled out (positive), and DIFFERS from plain tracev3 (so deleting the merge fails);
  3. a hostile body (clahe off, vec_merge 0.9, small_detail preserve, core_fill hybrid,
     resolution 300, tone on, adaptive threshold) is forced back to the recipe;
  4. the photo front end never runs for it, even if a caller smuggles front_end=bw_xdog
     (positive control: style="bwphoto" DOES fire it);
  5. existing styles are untouched: a plain tracev3 request's effective_body equals its
     requested body (no recipe merge), while illustration's shows the recipe; placement
     fields in the body (items, paper, margin) survive the merge untouched.
"""
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home  # explicit runtime home BEFORE any ccplotter import
_test_home.use_temp_home("ccplotter_test_home_illus_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_illus_"))
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
from PIL import Image

import ccplotter.bw_photo as bw_photo
import ccplotter.routes.runs  # noqa: F401
from ccplotter.config import ILLUSTRATION_RECIPE, WORK_DIR
from ccplotter.runs import _runs_load, load_run_provenance

PLACEMENT_KEYS = {"paper", "margin", "offset_x", "offset_y", "paper_rot", "paper_keepout",
                  "registered_paper", "items", "scale", "rot", "dx", "dy", "crop"}


def _make_drawing(path: Path, w=1100, h=800):
    """A clean line drawing: dark strokes of varied width on white (no photo tones)."""
    img = np.full((h, w), 255, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy, r, t in ((300, 300, 180, 7), (700, 420, 240, 13), (550, 600, 90, 4), (850, 200, 60, 20)):
        d = np.abs(np.hypot(xx - cx, yy - cy) - r)
        img[d < t] = 20
    img[395:405, 80:1000] = 20
    Image.fromarray(img).convert("RGB").save(path, quality=95)
    return path


def _points_of(text):
    pts = []
    for ps in re.findall(r'points="([^"]+)"', text):
        for tok in ps.split():
            if "," in tok:
                x, y = tok.split(",")
                pts.append((round(float(x), 3), round(float(y), 3)))
    return sorted(pts)


def _run(did, extra, job_type="preview"):
    from ccplotter.cal import load_pen_profiles
    from ccplotter.routes.core import app
    _make_drawing(WORK_DIR / f"src_{did}.jpg")
    body = {"job_type": job_type, "paper": "120x120", "margin": 10.0, "debug": True,
            "pen_id": load_pen_profiles().get("active"),
            "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}]}
    body.update(extra)
    resp = app.test_client().post("/api/process", json=body)
    assert resp.status_code == 200, resp.get_json()
    j = resp.get_json()
    routed = j.get("debug_files", {}).get("routed_svg")
    assert routed, "no debug_files.routed_svg"
    return body, j, _points_of((WORK_DIR / routed).read_text(encoding="utf-8"))


MATCHED = dict(ILLUSTRATION_RECIPE)


# ---------------------------------------------------------------- 1. recipe

def test_recipe_dict_pinned_no_frontend_no_placement():
    assert ILLUSTRATION_RECIPE == {
        "clahe": True, "tone": "off", "resolution": 2200, "native_res": False,
        "threshold_mode": "otsu", "fill_mode": "contour", "core_fill": "rings",
        "small_detail": "off", "vec_merge": 0.3, "scale_aware_v2": True,
        "illustration_pitch_clamp": True}
    assert "front_end" not in ILLUSTRATION_RECIPE
    assert not (set(ILLUSTRATION_RECIPE) & PLACEMENT_KEYS)
    # the recipe's core_fill must be a value the server reads as non-hybrid (the contour fill)
    assert ILLUSTRATION_RECIPE["core_fill"] != "hybrid"


# ---------------------------------------------------------------- 2. merge

def test_illustration_equals_matched_tracev3_and_differs_from_plain():
    _, _, pts_ill = _run("illus_a", {"style": "illustration"})
    _, _, pts_matched = _run("illus_b", {"style": "tracev3", **MATCHED})
    _, _, pts_plain = _run("illus_c", {"style": "tracev3"})
    assert pts_ill and pts_ill == pts_matched, "illustration != tracev3 with the recipe spelled out"
    # positive control: the merge is what makes it differ from default tracev3 (clahe off,
    # resolution 900, ...) -- identical geometry would mean it never fired
    assert pts_plain and pts_plain != pts_ill


# ---------------------------------------------------------------- 3. hostile body

def test_hostile_body_forced_back_to_the_recipe():
    hostile = {"clahe": False, "vec_merge": 0.9, "small_detail": "preserve", "core_fill": "hybrid",
               "resolution": 300, "tone": "balanced", "threshold_mode": "adaptive"}
    _, _, clean = _run("illus_h1", {"style": "illustration"})
    _, _, host = _run("illus_h2", {"style": "illustration", "native_res": True,
                                    "fill_mode": "zigzag", **hostile})
    assert clean and clean == host
    # positive control: the same hostile body under plain tracev3 is NOT forced back
    _, _, host_plain = _run("illus_h3", {"style": "tracev3", **hostile})
    assert host_plain != clean


# ---------------------------------------------------------------- 4. front end none

def test_photo_front_end_never_runs_for_illustration(monkeypatch):
    calls = []
    orig = bw_photo.prepare_bw_xdog_source
    monkeypatch.setattr(bw_photo, "prepare_bw_xdog_source",
                        lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    _, j, pts = _run("illus_f1", {"style": "illustration"})
    _run("illus_f2", {"style": "illustration", "front_end": "bw_xdog"})
    assert pts and not calls, "XDoG front end ran for style='illustration'"
    assert "xdog" not in (j.get("debug_files") or {})
    for p in WORK_DIR.glob("src_illus_f*_render_*.svg.info.json"):
        assert "bw_xdog" not in p.read_text()
    # positive control: the identical call under bwphoto DOES fire it
    _run("illus_f3", {"style": "bwphoto"})
    assert calls, "control failed: front end did not fire under style='bwphoto'"


# ---------------------------------------------------------------- 5. existing styles + placement

def test_existing_styles_unmerged_and_placement_untouched():
    items = [{"id": "illus_p1", "scale": 1.2, "rot": 180, "dx": -0.3, "dy": 2.0, "crop": None}]
    _run("illus_p1", {"style": "illustration", "items": items, "paper": "130x100",
                      "margin": 7.0, "offset_x": 3.0}, job_type="gcode")
    rec_i = next(r for r in _runs_load() if r.get("type") == "gcode")
    eff = load_run_provenance(rec_i["id"])["effective_body"]
    for k, v in ILLUSTRATION_RECIPE.items():
        assert eff[k] == v, k
    # placement is exactly what the caller sent
    assert eff["items"] == items and eff["paper"] == "130x100" and eff["margin"] == 7.0 and eff["offset_x"] == 3.0
    assert "front_end" not in eff

    items2 = [{"id": "illus_p2", "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}]
    _run("illus_p2", {"style": "tracev3", "items": items2, "clahe": False, "vec_merge": 0.5},
         job_type="gcode")
    rec_t = next(r for r in _runs_load() if r.get("type") == "gcode")
    prov_t = load_run_provenance(rec_t["id"])
    # tracev3: no recipe merge (effective == requested) and its values are not the recipe's
    assert prov_t["effective_body"] == prov_t["request_body"]
    assert prov_t["effective_body"]["clahe"] is False and prov_t["effective_body"]["vec_merge"] == 0.5


def test_variant_label():
    from ccplotter.vcache import _variant_label
    assert _variant_label({"style": "illustration"}).startswith("Trace V3+")


# ---------------------------------------------------------------- 6. pitch vs fill_mm

def _run_geo(did, extra, paper):
    """Like _run but returns (response, the item's equivalence block)."""
    from ccplotter.cal import load_pen_profiles
    from ccplotter.routes.core import app
    _make_drawing(WORK_DIR / f"src_{did}.jpg")
    body = {"job_type": "preview", "paper": paper, "margin": 10.0,
            "pen_id": load_pen_profiles().get("active"),
            "items": [{"id": did, "scale": 1, "rot": 0, "dx": 0, "dy": 0, "crop": None}]}
    body.update(extra)
    resp = app.test_client().post("/api/process", json=body)
    assert resp.status_code == 200, resp.get_json()
    j = resp.get_json()
    return j, j["item_geometry"][did]["scale_aware_equivalence_prelim_approx"]


def _pitch_warnings(j):
    """The pitch-vs-fill DANGER warning (the okay-level held-spacing note shares the prefix; excluded)."""
    return [w for w in j["warnings"] if w["msg"].startswith("Trace V3+:") and w["level"] == "danger"]


def _held_notes(j):
    return [w for w in j["warnings"] if w["msg"].startswith("Trace V3+: spacing held") and w["level"] == "okay"]


def _ceiling_dangers(j):
    return [w for w in j["warnings"] if w["level"] == "danger" and "resolution boost's ceiling" in w["msg"]]


def test_warning_text_is_exact_and_threshold_has_a_control():
    from ccplotter.stats_warnings import illustration_pitch_warning, illustration_pitch_warnings

    def stats(p, f):
        return {"item_geometry": {"1": {"scale_aware_equivalence_prelim_approx": {
            "illustration_pitch_mm": p, "illustration_fill_mm": f}}}}
    msg = illustration_pitch_warning(0.1357, 0.12)
    assert msg == ("Trace V3+: passes 0.136 mm apart at this placement; "
                   "your pen's fill is 0.120 mm; reduce scale/paper or use Trace (V3)")
    assert illustration_pitch_warnings(stats(0.1357, 0.12)) == [msg]
    # CONTROL: at or under fill_mm, and for a block without the fields (any other style), silent
    assert illustration_pitch_warnings(stats(0.1089, 0.12)) == []
    assert illustration_pitch_warnings(stats(0.12, 0.12)) == []
    assert illustration_pitch_warnings({"item_geometry": {"1": {
        "scale_aware_equivalence_prelim_approx": {"fill_px_real": 3.0}}}}) == []


def _no_clamp_recipe():
    return {k: v for k, v in ILLUSTRATION_RECIPE.items() if k != "illustration_pitch_clamp"}


def test_recipe_pins_the_clamp_on_with_a_control(monkeypatch):
    """HOLD THE PITCH: the clamp is part of the recipe, so a plain
    style="illustration" request (no clamp key) is clamped, and a caller cannot turn it off."""
    from ccplotter.config import ILLUSTRATION_PITCH_CLAMP_FRAC
    assert ILLUSTRATION_RECIPE["illustration_pitch_clamp"] is True
    _, on = _run_geo("illus_k1", {"style": "illustration"}, "600x600")
    fill = on["illustration_fill_mm"]
    assert on["illustration_pitch_clamp"]["applied"] is True
    assert abs(on["illustration_pitch_mm"] - ILLUSTRATION_PITCH_CLAMP_FRAC * fill) < 1e-4
    # a hostile body that asks for the clamp OFF is forced back on (recipe wins the merge)
    _, host = _run_geo("illus_k2", {"style": "illustration", "illustration_pitch_clamp": False}, "600x600")
    assert host["illustration_pitch_clamp"]["applied"] is True and host["illustration_pitch_mm"] == on["illustration_pitch_mm"]
    # CONTROL (the test can fail): the recipe WITHOUT the key does not clamp -> pitch > fill_mm
    import ccplotter.routes.core as core
    monkeypatch.setattr(core, "ILLUSTRATION_RECIPE", _no_clamp_recipe())
    _, off = _run_geo("illus_k3", {"style": "illustration"}, "600x600")
    assert off["illustration_pitch_clamp"] is None and off["illustration_pitch_mm"] > fill


def test_clamp_is_unreadable_outside_the_illustration_style():
    # tracev3 + the key (and the whole recipe spelled out) == tracev3 without it: never read there
    _, t0 = _run_geo("illus_c3", {"style": "tracev3"}, "600x600")
    _, t1 = _run_geo("illus_c4", {"style": "tracev3", "illustration_pitch_clamp": True}, "600x600")
    _, t2 = _run_geo("illus_c7", {"style": "tracev3", **MATCHED}, "600x600")
    assert t0 == t1 and "illustration_pitch_mm" not in t1 and "illustration_pitch_clamp" not in t1
    assert "illustration_pitch_mm" not in t2          # recipe incl. the clamp key, but not the illustration path


def test_noop_at_a_placement_already_within_fill(monkeypatch):
    import ccplotter.routes.core as core
    _, s1 = _run_geo("illus_c6", {"style": "illustration"}, "70x70")
    monkeypatch.setattr(core, "ILLUSTRATION_RECIPE", _no_clamp_recipe())
    _, s0 = _run_geo("illus_c5", {"style": "illustration"}, "70x70")
    assert s0 == s1 and s1["illustration_pitch_clamp"] is None   # byte-identical to the unclamped run
    assert s1["illustration_pitch_mm"] <= s1["illustration_fill_mm"]


def test_pitch_warning_silent_when_held_and_fires_with_numbers_when_it_cannot_be(monkeypatch):
    from ccplotter.stats_warnings import illustration_pitch_warning
    # held: clamp on at a big placement -> pitch <= fill_mm -> no warning
    j_ok, eq_ok = _run_geo("illus_w1", {"style": "illustration"}, "600x600")
    assert eq_ok["illustration_pitch_mm"] <= eq_ok["illustration_fill_mm"] and not _pitch_warnings(j_ok)
    # CANNOT be held: trace_v3 floors fill_px at 1.2, so a placement so large that the clamp target is
    # below 1.2 final px really lays 1.2 px, which is wider than fill_mm -> the warning fires, with the
    # REALIZED numbers and the exact pitch-vs-fill text
    j_big, eq_big = _run_geo("illus_w4", {"style": "illustration"}, "1500x1500")
    assert eq_big["illustration_pitch_floor_binds"] is True
    assert eq_big["illustration_pitch_mm"] > eq_big["illustration_fill_mm"]
    pw = _pitch_warnings(j_big)
    assert len(pw) == 1 and pw[0]["level"] == "danger" and pw[0]["tier"] == "primary"
    assert pw[0]["msg"] == illustration_pitch_warning(eq_big["illustration_pitch_mm"], eq_big["illustration_fill_mm"])
    assert f"{eq_big['illustration_pitch_mm']:.3f}" in pw[0]["msg"] and f"{eq_big['illustration_fill_mm']:.3f}" in pw[0]["msg"]
    # CONTROL 1: floor not binding at the held placement
    assert eq_ok["illustration_pitch_floor_binds"] is False
    # CONTROL 2: the unclamped recipe at 600x600 warns exactly as the pitch-vs-fill rule specifies (the original case)
    import ccplotter.routes.core as core
    monkeypatch.setattr(core, "ILLUSTRATION_RECIPE", _no_clamp_recipe())
    j_unc, eq_unc = _run_geo("illus_w5", {"style": "illustration"}, "600x600")
    assert len(_pitch_warnings(j_unc)) == 1
    monkeypatch.undo()
    # CONTROL 3: a small placement is silent; plain tracev3 at the huge placement never produces it
    j_small, _ = _run_geo("illus_w2", {"style": "illustration"}, "70x70")
    assert not _pitch_warnings(j_small)
    j_t, eq_t = _run_geo("illus_w3", {"style": "tracev3"}, "1500x1500")
    assert not _pitch_warnings(j_t) and "illustration_pitch_mm" not in eq_t


# ---------------------------------------------------------------- 7. boost-ceiling danger vs held pitch

def test_ceiling_danger_replaced_by_a_note_while_the_pitch_is_held(monkeypatch):
    """HOLD THE PITCH. At 600x600 (a large placement for the test pen) the resolution
    cap binds (needed 3.78x, capped 2.49x) but the clamp holds the pitch: no ceiling danger, an okay note."""
    j, eq = _run_geo("illus_n1", {"style": "illustration"}, "600x600")
    boost = eq["v2_res_boost"]
    assert boost["actual_boost"] < boost["needed_boost"] - 1e-6          # the cap really binds here
    assert eq["illustration_pitch_mm"] <= eq["illustration_fill_mm"]     # ... and the pitch is held
    assert not _ceiling_dangers(j) and not _pitch_warnings(j)
    notes = _held_notes(j)
    assert len(notes) == 1 and notes[0]["tier"] == "primary"
    assert notes[0]["msg"] == (
        f"Trace V3+: spacing held at {eq['illustration_pitch_mm']:.3f} mm "
        f"(pen fill {eq['illustration_fill_mm']:.3f}). Trace detail is capped at this size "
        f"({eq['fill_px_real']:.2f} px per pass; 3 is the design target).")
    # CONTROL A: plain tracev3 with the SAME cap-binding v2 settings keeps the danger, no note
    jt, et = _run_geo("illus_n2", {"style": "tracev3", "scale_aware_v2": True, "resolution": 2200}, "600x600")
    assert et["v2_res_boost"]["actual_boost"] < et["v2_res_boost"]["needed_boost"] - 1e-6
    assert len(_ceiling_dangers(jt)) == 1 and not _held_notes(jt) and not _pitch_warnings(jt)
    # CONTROL B: the same illustration placement WITHOUT the clamp (pitch > fill) keeps the danger
    # AND the pitch-vs-fill warning, and gets no note -- so the note is not unconditional
    import ccplotter.routes.core as core
    monkeypatch.setattr(core, "ILLUSTRATION_RECIPE", _no_clamp_recipe())
    ju, eu = _run_geo("illus_n3", {"style": "illustration"}, "600x600")
    assert eu["illustration_pitch_mm"] > eu["illustration_fill_mm"]
    assert len(_ceiling_dangers(ju)) == 1 and len(_pitch_warnings(ju)) == 1 and not _held_notes(ju)


def test_ceiling_danger_and_pitch_warning_both_stay_when_the_floor_binds():
    j, eq = _run_geo("illus_n4", {"style": "illustration"}, "1500x1500")
    assert eq["illustration_pitch_floor_binds"] is True and eq["illustration_pitch_mm"] > eq["illustration_fill_mm"]
    assert len(_ceiling_dangers(j)) == 1 and len(_pitch_warnings(j)) == 1 and not _held_notes(j)
    # CONTROL: the held placement of the same style has neither danger (see the test above)
    jh, _ = _run_geo("illus_n5", {"style": "illustration"}, "600x600")
    assert not _ceiling_dangers(jh) and not _pitch_warnings(jh)


def test_other_styles_ceiling_danger_text_is_unchanged():
    from ccplotter.stats_warnings import build_warnings

    def stats(extra):
        eq = {"v2_res_boost": {"needed_boost": 2.5, "actual_boost": 2.4909, "boosted_res": 5480},
              "fill_px_real": 2.989091}
        eq.update(extra)
        return {"stroke_count": 10, "estimated_z_moves": 20,
                "item_geometry": {"1": {"scale_aware_equivalence_prelim_approx": eq}}}
    plain = [w for w in build_warnings(stats({})) if "ceiling" in w["msg"]]
    assert len(plain) == 1 and plain[0]["level"] == "danger" and plain[0]["msg"].startswith(
        "This placement has hit the line-density resolution boost's ceiling (needed 2.5x, capped at 2.4909x) and can no "
        "longer hold its own spacing target (fill_px real ~2.989091px, below the 3.0px floor).")
    assert plain[0]["msg"].endswith("Plot smaller, or expect degraded line density here.")
    # illustration fields, pitch held -> note instead; pitch over fill -> danger kept (+ pitch-vs-fill warning)
    held = build_warnings(stats({"illustration_pitch_mm": 0.108, "illustration_fill_mm": 0.12}))
    assert not [w for w in held if "ceiling" in w["msg"]]
    assert [w["msg"] for w in held if w["level"] == "okay" and "spacing held" in w["msg"]] == [
        "Trace V3+: spacing held at 0.108 mm (pen fill 0.120). "
        "Trace detail is capped at this size (2.99 px per pass; 3 is the design target)."]
    over = build_warnings(stats({"illustration_pitch_mm": 0.13, "illustration_fill_mm": 0.12}))
    assert len([w for w in over if "ceiling" in w["msg"]]) == 1 and not [w for w in over if "spacing held" in w["msg"]]
    assert any(w["msg"].startswith("Trace V3+: passes 0.130") for w in over)
    # the byte-identity claim: the plain (no illustration fields) danger equals the over-fill one's danger text
    assert [w for w in over if "ceiling" in w["msg"]][0]["msg"] == plain[0]["msg"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
