"""Two-pen calibration sheet "M10-REG v1" (ccplotter/calibration_twopen.py,
POST /api/twopen_calsheet). Same env conventions as tests/test_twopen.py (temp runtime home
BEFORE any ccplotter import, temp WORK_DIR, Flask test client, never :5001)."""
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_home
_HOME = _test_home.use_temp_home("ccplotter_test_home_calsheet2_")
os.environ.setdefault("CCPLOTTER_WORK_DIR", tempfile.mkdtemp(prefix="ccplotter_test_work_calsheet2_"))
sys.path.insert(0, str(REPO))

import pytest

import ccplotter.routes.runs  # noqa: F401
from ccplotter import calibration_twopen as ct
from ccplotter.config import WORK_DIR
from ccplotter.runs import BACKLINK_PREFIX

BLUE, BLACK = "bic_cristal_blue", "bic_cristal_black"


def _client():
    from ccplotter.routes.core import app
    return app.test_client()


@pytest.fixture(scope="module")
def sheet():
    r = _client().post("/api/twopen_calsheet", json={"blue_pen_id": BLUE, "black_pen_id": BLACK})
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    files = {g["color"]: WORK_DIR / g["file"] for g in j["gcode_files"]}
    return {"resp": j, "files": files, "text": {k: v.read_text(encoding="ascii") for k, v in files.items()}}


@pytest.fixture(scope="module")
def layers():
    return ct.build_layers(0.12, 0.12)


# ------------------------------------------------------------------ safety (same scans as test_twopen)
def test_both_files_safe(sheet):
    bad_fan = re.compile(r"\bM106\s+P[01]\b", re.I)
    bad_pause = re.compile(r"^\s*(M0|M1|M600|M25|M226)\b", re.I | re.M)
    for color, p in sheet["files"].items():
        text = p.read_bytes().decode("ascii")
        code = "\n".join(l.split(";")[0] for l in text.splitlines())
        assert not re.search(r"\bG92\b", code) and not bad_fan.search(code) and not bad_pause.search(code), color
        assert text.count("M117 Plotting...") == 1 and text.count("M117 Plot complete") == 1
        assert len([l for l in text.splitlines() if l.startswith("; PASS")]) == 1
        assert "; PAIR " in text and "; PLACEMENT shared" in text and "; PEN this_pass" in text
        assert BACKLINK_PREFIX in text
        for ln in code.splitlines():
            if ln.startswith("M106"):
                assert re.fullmatch(r"M106( P[23])?( S\d+)?", ln.strip()), ln
    assert sheet["text"]["blue"] != sheet["text"]["black"]
    assert "PASS 1 OF 2: BLUE" in sheet["text"]["blue"] and "PASS 2 OF 2: BLACK" in sheet["text"]["black"]


def test_two_passes_reach_validated_with_model_b_times(sheet):
    j = sheet["resp"]
    assert [g["color"] for g in j["gcode_files"]] == ["blue", "black"]
    for g in j["gcode_files"]:
        assert g["reach_validated"] is True
        assert g["time_est"]["label"] == "model (b), not stopwatch-validated" and g["time_est"]["minutes"] > 0


def test_every_patch_and_stroke_is_inside_reach(sheet, layers):
    from ccplotter.cal import load_cal
    from ccplotter.reach import active_reach_profile
    reach = active_reach_profile(load_cal())
    x0, y0, x1, y1 = reach["bbox"]
    for p in sheet["resp"]["passes"]:
        bx0, by0, bx1, by1 = p["pendown_bbox"]
        assert x0 <= bx0 and bx1 <= x1 and y0 <= by0 and by1 <= y1, (p["color"], p["pendown_bbox"], reach["bbox"])
    # geometry stays inside the declared content area of the 250x170 paper
    cx0, cy0, cx1, cy1 = ct.CONTENT
    for col in ("blue", "black"):
        for s in layers[col]:
            for x, y in s:
                assert cx0 - 0.5 <= x <= cx1 + 0.5 and cy0 - 0.5 <= y <= cy1 + 0.5, (col, x, y)
    # every patch origin + 8 mm sits inside content
    for pen in ("blue", "black"):
        for pt in layers["record"]["wedges"][pen]:
            ox, oy = pt["origin"]
            assert cx0 <= ox and ox + ct.PATCH_MM <= cx1 and cy0 <= oy and oy + ct.PATCH_MM <= cy1


# ------------------------------------------------------------------ vernier geometry
def _vernier_from_layer(strokes, centre, axis):
    cx, cy = centre
    out = []
    for s in strokes:
        if len(s) != 2:
            continue
        (xa, ya), (xb, yb) = s
        if axis == "x" and abs(xa - xb) < 1e-9 and abs(abs(ya - yb) - ct.V_LEN) < 1e-6 and abs((ya + yb) / 2 - cy) < 1e-6 and abs(xa - cx) <= 5.0 + 1e-6:
            out.append(s)
        if axis == "y" and abs(ya - yb) < 1e-9 and abs(abs(xa - xb) - ct.V_LEN) < 1e-6 and abs((xa + xb) / 2 - cx) < 1e-6 and abs(ya - cy) <= 5.0 + 1e-6:
            out.append(s)
    key = (lambda s: s[0][0]) if axis == "x" else (lambda s: s[0][1])
    return sorted(out, key=key)


def _shift(lines, axis, d):
    return [[(x + d, y) if axis == "x" else (x, y + d) for x, y in s] for s in lines]


def test_vernier_pitches_and_counts(layers):
    for v in layers["record"]["verniers"]:
        for axis, c in (("x", v["x_vernier_center"]), ("y", v["y_vernier_center"])):
            b = _vernier_from_layer(layers["blue"], c, axis)
            k = _vernier_from_layer(layers["black"], c, axis)
            assert len(b) == 11 and len(k) == 11, (v["pos"], axis, len(b), len(k))
            pos = lambda L: [(s[0][0] if axis == "x" else s[0][1]) for s in L]
            assert [round(b2 - b1, 6) for b1, b2 in zip(pos(b), pos(b)[1:])] == [1.0] * 10
            assert [round(b2 - b1, 6) for b1, b2 in zip(pos(k), pos(k)[1:])] == [0.9] * 10
            assert abs(pos(b)[5] - pos(k)[5]) < 1e-9          # centre lines coincide
            assert all(len(s) == 2 for s in b + k)            # single strokes, 6 mm (checked in the finder)


@pytest.mark.parametrize("d, idx", [(0.0, 0), (0.3, 3), (-0.3, -3), (0.1, 1), (-0.5, -5), (0.5, 5), (0.2, 2)])
def test_synthetic_offset_reads_the_right_index_on_every_vernier(layers, d, idx):
    for v in layers["record"]["verniers"]:
        for axis, c in (("x", v["x_vernier_center"]), ("y", v["y_vernier_center"])):
            blue = _vernier_from_layer(layers["blue"], c, axis)
            black = _shift(_vernier_from_layer(layers["black"], c, axis), axis, d)
            got, dmm = ct.read_vernier(blue, black, axis)
            assert got == idx, (v["pos"], axis, d, got)
            assert dmm == pytest.approx(0.1 * idx)
    # positive control: reading is not a constant
    assert idx == 0 or ct.read_vernier(
        _vernier_from_layer(layers["blue"], layers["record"]["verniers"][0]["x_vernier_center"], "x"),
        _vernier_from_layer(layers["black"], layers["record"]["verniers"][0]["x_vernier_center"], "x"), "x")[0] == 0


def test_vernier_math_closed_form():
    blue = ct.vernier_lines(0, 0, "x", "blue")
    for k in range(-5, 6):
        black = ct.vernier_lines(0, 0, "x", "black", offset=0.1 * k)
        assert ct.read_vernier(blue, black, "x")[0] == k


# ------------------------------------------------------------------ layout contents
def test_layout_counts_and_pitches(layers):
    r = layers["record"]
    assert len(r["verniers"]) == 9 and len({tuple(v["center"]) for v in r["verniers"]}) == 9
    assert sorted({v["center"][0] for v in r["verniers"]}) == list(ct.POS_X)
    for pen in ("blue", "black"):
        w = r["wedges"][pen]
        assert [p["pitch"] for p in w[:5]] == [0.96, 0.48, 0.24, 0.24, 0.12] and w[3]["cross"] is True
        assert w[5]["zigzag"] is True and w[5]["pitch"] == 0.12
    assert len(r["overlays"]) == 6
    assert len(r["fiducials"]) == 6


def test_every_non_label_stroke_is_at_least_2mm(layers):
    labs = {id(s) for s in layers["labels"]}
    def plen(s):
        return sum(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a, b in zip(s, s[1:]))
    for col in ("blue", "black"):
        short = [s for s in layers[col] if id(s) not in labs and plen(s) < 2.0 - 1e-9]
        assert short == [], (col, len(short), short[:2])
    # the exemption is real (Hershey glyph strokes can be short) -- so the rule above is not vacuous
    assert any(plen(s) < 2.0 for s in layers["labels"])


def test_zigzag_patch_is_one_stroke_and_cross_has_both_angles(layers):
    zz = ct._zigzag(0, 0, 8, 8, 0.12)
    assert len(zz) == 1 and len(zz[0]) > 100
    cross = ct._patch("cross", 0, 0, 0.24)
    ang = {round(((s[1][1] - s[0][1]) / (s[1][0] - s[0][0])), 3) for s in cross if abs(s[1][0] - s[0][0]) > 1e-9}
    assert 1.0 in ang and -1.0 in ang


def test_identity_label_is_drawn_in_black_only(layers):
    from ccplotter.calibration_twopen import _label
    ident = _label(ct.SHEET_ID, 128.0, 104.0, 4.0)
    keyset = {tuple(map(lambda p: (round(p[0], 4), round(p[1], 4)), s)) for s in ident}
    black = {tuple(map(lambda p: (round(p[0], 4), round(p[1], 4)), s)) for s in layers["black"]}
    blue = {tuple(map(lambda p: (round(p[0], 4), round(p[1], 4)), s)) for s in layers["blue"]}
    assert keyset <= black and not (keyset & blue)
    assert "M10-REG v1" == ct.SHEET_ID and ct.SHEET_ID.isascii()


def test_not_in_the_one_file_registry():
    from ccplotter.calibration_registry import SHEETS
    assert not any("twopen" in k or "m10" in k.lower() for k in SHEETS)


def test_bad_pens_refused_4xx():
    for body in ({"blue_pen_id": BLUE, "black_pen_id": BLUE}, {"blue_pen_id": "nope", "black_pen_id": BLACK}, {"variant": "Z"}):
        r = _client().post("/api/twopen_calsheet", json=body)
        assert 400 <= r.status_code < 500, (body, r.status_code)


# ------------------------------------------------------------------ placement on the registered paper
def test_sheet_window_tracks_the_pocket_frame():
    """calibration_pocket's frame changed on fill-mode (88b60b5: holes at the FRONT, Y_FRONT_TIP_DEFAULT 8.0 -> 14.0, TIP_Y_MAX 171.5 -> 180.5). The sheet's own front
    clearance (13.5 / 14.0) is its own constant; its window upper bound is pocket's TIP_Y_MAX, whatever that currently is."""
    from ccplotter import calibration_pocket as pk
    assert pk.X0_TIP_DEFAULT == 10.0 and pk.HOLE_KEEPOUT_MM == 10.0
    assert ct.TIP_Y_MIN_SHEET == 13.5 and ct.Y_FRONT_TIP_SHEET == 14.0
    assert ct.TIP_WINDOW == {"x": [pk.TIP_X_MIN, pk.TIP_X_MAX], "y": [13.5, pk.TIP_Y_MAX]}
    assert pk.TIP_Y_MAX >= 165.8                       # the sheet's farthest pen-down point must stay inside, whichever frame is current


def test_tip_window_and_front_clearance_from_the_emitted_gcode(sheet):
    """Both passes: pen-down TIP bbox (nozzle + composed pen offset) inside the paper window, and the
    nearest mark to the FRONT paper edge (tip Y 1.5) is >= 12 mm away. Cross-checks the offset formula."""
    from ccplotter import calibration_pocket as pk
    for p in sheet["resp"]["passes"]:
        tx0, ty0, tx1, ty1 = p["tip_bbox"]
        assert pk.TIP_X_MIN <= tx0 and tx1 <= pk.TIP_X_MAX and ct.TIP_Y_MIN_SHEET <= ty0 and ty1 <= pk.TIP_Y_MAX, p["tip_bbox"]
        assert ty0 - pk.PAPER_FRONT_Y_MM >= 12.0
        assert p["tip_window"] == ct.TIP_WINDOW
    # formula check: ink left edge -> X0_TIP_DEFAULT, front-most ink -> Y_FRONT_TIP_SHEET (pen-down paths, 0.01 mm)
    L = ct.build_layers()
    x0, y0, x1, y1 = ct.ink_bbox(L)
    for p in sheet["resp"]["passes"]:
        tx0, ty0, tx1, ty1 = p["tip_bbox"]
        assert tx0 == pytest.approx(pk.X0_TIP_DEFAULT, abs=0.02) or tx0 > pk.X0_TIP_DEFAULT   # a layer may not own the left-most mark
    allx = [b for p in sheet["resp"]["passes"] for b in (p["tip_bbox"][0], p["tip_bbox"][2])]
    ally = [b for p in sheet["resp"]["passes"] for b in (p["tip_bbox"][1], p["tip_bbox"][3])]
    assert min(allx) == pytest.approx(ct.X0_TIP_DEFAULT, abs=0.02)
    assert min(ally) == pytest.approx(ct.Y_FRONT_TIP_SHEET, abs=0.02)
    assert max(allx) - min(allx) == pytest.approx(x1 - x0, abs=0.05)
    assert max(ally) - min(ally) == pytest.approx(y1 - y0, abs=0.05)


def test_tip_window_refusal_is_422_not_500():
    body, _ = ct.sheet_body(BLUE, BLACK)
    for tw in ({"x": [8.0, 100.0], "y": [13.5, 171.5]},      # right of the window
               {"x": [8.0, 249.0], "y": [40.0, 171.5]}):     # front marks leave the window
        b = dict(body, tip_window=tw)
        r = _client().post("/api/twopen", json=b)
        assert r.status_code == 422 and "paper window" in r.get_json()["error"], (tw, r.status_code, r.get_json())


def test_sheet_never_uses_the_reach_bbox_default(sheet):
    body, L = ct.sheet_body(BLUE, BLACK)
    assert body["offset_x"] != 3.0 and body["offset_y"] != 42.5 and body["tip_window"] == ct.TIP_WINDOW
    assert sheet["resp"]["passes"][0]["pen_offset_mm"] is not None


def test_content_is_237_by_156_inside_the_page():
    L = ct.build_layers()
    x0, y0, x1, y1 = ct.ink_bbox(L)
    assert x1 - x0 <= 237.0 and y1 - y0 <= 161.0 and x0 >= 5.0 and x1 <= 242.0 and y0 >= 3.0 and y1 <= 161.0
    # the top-row tick labels hang slightly above the nominal content box; they are still on the page
    assert y0 >= 3.0


# ------------------------------------------------------------------ fine verniers
@pytest.mark.parametrize("k", [-5, -2, 0, 3, 5])
def test_fine_vernier_reads_0p05_steps(layers, k):
    for v in layers["record"]["verniers"]:
        for axis, key in (("x", "x_fine_center"), ("y", "y_fine_center")):
            c = v[key]
            blue = _vernier_from_layer(layers["blue"], c, axis)
            black = _shift(_vernier_from_layer(layers["black"], c, axis), axis, 0.05 * k)
            assert len(blue) == 11 and len(black) == 11
            idx, d = ct.read_vernier(blue, black, axis, step_mm=0.05)
            assert idx == k and d == pytest.approx(0.05 * k)
    pos = lambda L: [s[0][0] for s in L]
    b = _vernier_from_layer(layers["blue"], layers["record"]["verniers"][0]["x_fine_center"], "x")
    k_ = _vernier_from_layer(layers["black"], layers["record"]["verniers"][0]["x_fine_center"], "x")
    assert round(pos(b)[1] - pos(b)[0], 6) == 0.5 and round(pos(k_)[1] - pos(k_)[0], 6) == 0.45


# ------------------------------------------------------------------ sheet A / sheet B, four files
@pytest.fixture(scope="module")
def sheet_a():
    from ccplotter.cal import load_pen_profiles, save_pen_profiles
    r0 = _client().post("/api/twopen_calsheet", json={"variant": "A"})       # profile copy absent -> refused
    assert 400 <= r0.status_code < 500, r0.status_code
    d = load_pen_profiles()
    cp = dict(d["profiles"][BLUE])
    cp["label"] = "TEST COPY for swap repeatability (D1378)"
    d["profiles"]["bic_cristal_blue_b"] = cp                                 # TEMP home only
    save_pen_profiles(d)
    r = _client().post("/api/twopen_calsheet", json={"variant": "A"})
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    return {"resp": j, "files": {g["color"]: WORK_DIR / g["file"] for g in j["gcode_files"]}}


def test_sheet_a_is_blue_then_blue_b_and_both_pass_scans(sheet_a, sheet):
    j = sheet_a["resp"]
    assert [g["pen_id"] for g in j["gcode_files"]] == [BLUE, "bic_cristal_blue_b"]
    for ident, tag in ((sheet_a, "SHEET A"), (sheet, "SHEET B")):
        for color, f in ident["files"].items():
            text = f.read_bytes().decode("ascii")
            code = "\n".join(l.split(";")[0] for l in text.splitlines())
            assert not re.search(r"\bG92\b", code) and not re.search(r"\bM106\s+P[01]\b", code)
            assert not re.search(r"^\s*(M0|M1|M600|M25|M226)\b", code, re.M)
            assert len([l for l in text.splitlines() if l.startswith("; PASS")]) == 1
            assert "; PAIR " in text and "; PLACEMENT shared" in text and "; PEN this_pass" in text
    for p in j["passes"]:
        tx0, ty0, tx1, ty1 = p["tip_bbox"]
        assert ct.TIP_Y_MIN_SHEET <= ty0 and tx1 <= ct.TIP_WINDOW["x"][1]
    # same geometry in A and B per pass (only the pen id differs): identical move counts
    assert sheet_a["files"]["blue"].read_text().count("\nG1 X") == sheet["files"]["blue"].read_text().count("\nG1 X")


def test_live_profile_files_not_written_by_the_sheet_routes():
    from ccplotter.config import PEN_PROFILES_PATH
    assert str(PEN_PROFILES_PATH).startswith(str(_HOME)), "tests must only ever touch the TEMP home"
