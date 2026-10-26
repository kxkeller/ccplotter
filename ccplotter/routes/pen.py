"""ccplotter/routes/pen.py -- pen-profile + calibration/coverage-forensics
Flask routes: /api/pen_profiles (get/save/active/delete), /api/fan_probe,
/api/calibration_plot, /api/coverage_debug, /api/gap_audit, /api/preflight.

These routes hold no module-level mutable state; they call cal.py's load/save
functions and trace_v3.py's measurement-only debug renderers, which own their
own state.
"""
from pathlib import Path
from uuid import uuid4

from flask import jsonify, request

from ccplotter.atomic_io import atomic_write_bytes
from ccplotter.config import PAPER_SIZES, RASTER_EXTS, WORK_DIR
from ccplotter.cal import (load_pen_profiles, save_pen_profiles, active_pen_profile,
                            resolve_pen_profile, UnknownPenProfile)
from ccplotter.gcode_post import _fan_probe_gcode, _calibration_gcode
from ccplotter.calibration_plots import _wedge_gcode, _combo_gcode, _calblock_gcode, CAL_BLOCKS, _registration_marks_gcode
from ccplotter.trace_v3 import _coverage_debug, _gap_audit, _preflight

from ccplotter.routes import app

@app.route("/api/pen_profiles", methods=["GET"])
def api_pen_profiles():
    return jsonify(load_pen_profiles())


@app.route("/api/pen_profiles/save", methods=["POST"])
def api_pen_profiles_save():
    """Add/update a pen refill profile from the given fields, make it active."""
    body = request.json or {}
    pid = str(body.get("id") or "").strip()
    label = str(body.get("label") or "").strip()
    if not label:
        return jsonify({"error": "label required"}), 400
    if not pid:
        import re as _re
        pid = _re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or f"pen_{uuid4().hex[:4]}"
    d = load_pen_profiles()
    prof = d["profiles"].get(pid, {})
    for k in ("pen_mm", "fill_mm", "z_up", "z_down", "z_micro",
              "draw_speed", "travel_speed", "plunge_speed", "lift_speed"):
        if body.get(k) is not None:
            prof[k] = float(body[k]) if k in ("pen_mm", "fill_mm", "z_up", "z_down", "z_micro") else int(body[k])
    prof["label"] = label
    d["profiles"][pid] = prof
    d["active"] = pid
    save_pen_profiles(d)
    return jsonify({"ok": True, "id": pid, "profiles": d})


@app.route("/api/pen_profiles/active", methods=["POST"])
def api_pen_profiles_active():
    body = request.json or {}
    pid = str(body.get("id") or "")
    d = load_pen_profiles()
    if pid not in d["profiles"]:
        return jsonify({"error": "unknown profile"}), 404
    d["active"] = pid
    save_pen_profiles(d)
    return jsonify({"ok": True, "profile": d["profiles"][pid]})


@app.route("/api/pen_profiles/delete", methods=["POST"])
def api_pen_profiles_delete():
    body = request.json or {}
    pid = str(body.get("id") or "")
    d = load_pen_profiles()
    if pid in d["profiles"] and len(d["profiles"]) > 1:
        d["profiles"].pop(pid)
        if d.get("active") == pid:
            d["active"] = next(iter(d["profiles"]))
        save_pen_profiles(d)
        return jsonify({"ok": True, "profiles": d})
    return jsonify({"error": "cannot delete (unknown or last profile)"}), 400


#  Z_FIELD_LIMITS: the guard rails on hand-typed Z values. z_touch_abs is the
#  absolute machine height at which the tip touches paper -- typing a value a
#  few mm low drives the pen into the bed, so it gets a hard floor well above
#  Z=0 (the bed) and a ceiling above the tallest measured pen (BIC 18.75, the
#  45deg Preppy 20.75). The offsets are relative to that contact height:
#  z_down is NEGATIVE = press INTO the paper, z_up/z_micro are lifts ABOVE it.
Z_FIELD_LIMITS = {
    "z_touch_abs": (5.0, 45.0),
    "z_down":      (-3.0, 1.0),
    "z_up":        (0.1, 20.0),
    "z_micro":     (0.0, 20.0),
}


@app.route("/api/pen_profiles/z", methods=["POST"])
def api_pen_profiles_z():
    """Set the Z fields of ONE pen profile from typed values.

    Why this exists: the other way to set z_touch_abs is
    /api/save_z, which reads the printer's live Z over a websocket and writes
    the GLOBAL calibration.json value. This plotter's printer is offline by
    design (gcode is sideloaded by USB), so that path is unusable, and per-pen
    contact heights -- which is what resolve_z_touch() actually consults --
    would otherwise have to be set by hand-editing pen_profiles.json, which
    allows a pen to run at an unverified contact height.

    Deliberately NOT folded into /api/pen_profiles/save: that endpoint requires
    a label and force-activates the profile it writes. This one edits Z on a
    named (default: active) profile and changes nothing else.

    This does NOT weaken the job-path invariant documented on
    /api/calibration_plot: a PLOT request still may not carry a z_touch_abs;
    it is always resolved server-side from the stored profile. This route
    writes the STORED profile, which is a separate, explicit act.
    """
    body = request.json or {}
    d = load_pen_profiles()
    pid = str(body.get("id") or d.get("active") or "")
    if pid not in d.get("profiles", {}):
        return jsonify({"error": f"unknown pen profile '{pid}'"}), 404
    prof = d["profiles"][pid]
    applied, before = {}, {}
    for k, (lo, hi) in Z_FIELD_LIMITS.items():
        if body.get(k) is None:
            continue
        try:
            v = float(body[k])
        except (TypeError, ValueError):
            return jsonify({"error": f"{k} must be a number"}), 400
        if not (lo <= v <= hi):
            return jsonify({"error": f"{k}={v} is outside the safe range "
                                     f"[{lo}, {hi}] -- refusing to store it"}), 400
        before[k] = prof.get(k)
        prof[k] = v
        applied[k] = v
    if not applied:
        return jsonify({"error": "no Z fields supplied"}), 400
    save_pen_profiles(d)
    return jsonify({"ok": True, "id": pid, "label": prof.get("label"),
                    "applied": applied, "before": before, "profile": prof})


@app.route("/api/pen_profiles/fill", methods=["GET"])
def api_pen_fill_info():
    """Per-pen fill-size slider data: range, each pen's current fill_mm, its
    calibrated value (fill_mm_calibrated) and whether it is overridden. See ccplotter/pen_fill.py."""
    from ccplotter.pen_fill import fill_info
    return jsonify(fill_info())


@app.route("/api/pen_profiles/fill", methods=["POST"])
def api_pen_fill_set():
    """Set -- or with {"reset": true} restore to the CALIBRATED value -- fill_mm of ONE pen.

    Body: {"id": <pen id>, "fill_mm": <number>}  or  {"id": <pen id>, "reset": true}.
    Any other key is refused (so a stray z_touch_abs can never ride along). Edits
    nothing but that one field; writes atomically with a timestamped .bak. Takes effect
    on the next generate (the vcache fingerprint includes the whole resolved profile)."""
    from ccplotter.pen_fill import PenFillError, set_fill_mm
    from ccplotter.runtime_home import RuntimeHomeError
    body = request.json if isinstance(request.json, dict) else {}
    extra = sorted(set(body) - {"id", "fill_mm", "reset"})
    if extra:
        return jsonify({"error": f"unsupported field(s) {extra}: this route edits fill_mm only"}), 400
    pid = str(body.get("id") or "")
    if not pid:
        return jsonify({"error": "id required"}), 400
    reset = body.get("reset") is True
    if reset and "fill_mm" in body:
        return jsonify({"error": "give either fill_mm or reset, not both"}), 400
    if not reset and "fill_mm" not in body:
        return jsonify({"error": "fill_mm required (or reset: true)"}), 400
    try:
        return jsonify(set_fill_mm(pid, body.get("fill_mm"), reset=reset))
    except PenFillError as e:
        return jsonify({"error": str(e)}), e.status
    except RuntimeHomeError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/fan_probe", methods=["POST"])
def api_fan_probe():
    """Generate the fan-probe gcode (Fix A step 1). Plot it with fans on."""
    out = WORK_DIR / "fan_probe.gcode"
    try:
        _fan_probe_gcode(out)
    except Exception as e:
        return jsonify({"error": f"Fan probe failed: {e}"}), 500
    return jsonify({"ok": True, "file": out.name, "gcode_url": f"/work/{out.name}",
                    "size_kb": round(out.stat().st_size / 1024, 1)})


@app.route("/api/calibration_plot", methods=["POST"])
def api_calibration_plot():
    """Generate the parametric calibration plot gcode (no image needed).
    Z heights (touch/up/down) and pen width/speeds come from the ACTIVE PEN
    PROFILE by default: switch the pen dropdown, hit
    Calibrate, get a plot correct for THAT pen, no manual re-entry. The request
    body may still override individual fields (the UI's z-up/z-down/pen-width
    inputs are pre-filled from the active profile by applyPenProfile() in
    index.html, so in the normal flow these already match the active pen) --
    but z_touch_abs is NEVER taken from the request; it is always resolved
    server-side from the active profile (falling back to calibration.json)
    via resolve_z_touch(), so a stale/global height can't leak into a
    per-pen plot.

    pen_id (optional): pins the profile used to
    resolve pen_mm/fill_mm/Z/speeds to a SPECIFIC saved profile regardless
    of what's currently active in pen_profiles.json. Omitted, the
    ACTIVE profile is used. An unknown pen_id fails loudly with a 400 rather than
    silently falling back to active. See ccplotter/cal.py
    resolve_pen_profile() and scripts/replay_harness.py."""
    body = request.json or {}
    out = WORK_DIR / "calibration_plot.gcode"
    try:
        profile = resolve_pen_profile(body.get("pen_id"))
    except UnknownPenProfile as e:
        return jsonify({"error": str(e)}), 400
    _pd = load_pen_profiles()
    pen_id = body.get("pen_id") or _pd.get("active")
    try:
        info = _calibration_gcode(out, pen_mm=float(body.get("pen_width", profile.get("pen_mm", 0.5))),
                                  z_up=float(body.get("z_up", profile.get("z_up", 3.5))),
                                  z_down=float(body.get("z_down", profile.get("z_down", -1.0))),
                                  draw_speed=int(body.get("draw_speed", profile.get("draw_speed", 3000))),
                                  travel_speed=int(body.get("travel_speed", profile.get("travel_speed", 8000))),
                                  plunge_speed=int(body.get("plunge_speed", profile.get("plunge_speed", 500))),
                                  lift_speed=int(body.get("lift_speed", profile.get("lift_speed", 2000))),
                                  offset_x=body.get("offset_x"), offset_y=body.get("offset_y"),
                                  end_policy=str(body.get("end_policy", "release")),
                                  quiet_fans=bool(body.get("quiet_fans", True)),
                                  profile=profile)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Calibration plot failed: {e}"}), 500
    stamp = f"?t={uuid4().hex[:6]}"
    return jsonify({"ok": True, "gcode_url": f"/work/{out.name}", "file": out.name,
                    "size_kb": round(out.stat().st_size / 1024, 1),
                    "pen_id": pen_id, "pen_label": profile.get("label"),
                    "preview_url": f"/work/{info['preview']}{stamp}",
                    "svg_name": Path(info["svg_path"]).name, **info})


@app.route("/api/calibration_pack", methods=["POST"])
def api_calibration_pack():
    """ONE button that generates the FULL
    per-pen calibration set for whichever pen is active in the pen-refill
    dropdown -- (a) the line-width/fill-spacing test (_calibration_gcode)
    and (b) the gray step wedge (_wedge_gcode) -- both driven by that pen's
    OWN resolved touch/lift/draw heights, no manual Z entry. Filenames are
    pen-identifying (calibration_<kind>_<pen_id>_z<touch*100>.gcode) so it is
    clear at a glance which pen a file is for.

    If neither the active profile nor calibration.json's global fallback
    supplies a z_touch_abs, resolve_z_touch() raises ZTouchNotCalibrated
    (ccplotter/cal.py) BEFORE any gcode is written for either file -- the
    except branch below turns that into a clear, pen-named 500 error; no
    partial/half-written pack is left behind."""
    from ccplotter.cal import resolve_z_touch, ZTouchNotCalibrated
    body = request.json or {}
    profile = active_pen_profile()
    _pd = load_pen_profiles()
    pen_id = _pd.get("active") or "pen"
    pen_label = profile.get("label") or pen_id

    try:
        z_touch, z_touch_source = resolve_z_touch(None, profile)
    except ZTouchNotCalibrated as e:
        return jsonify({"error": f"Cannot generate calibration pack for '{pen_label}': {e}"}), 400

    z_slug = f"z{int(round(z_touch * 100))}"
    id_slug = str(pen_id).strip() or "pen"
    fill_out = WORK_DIR / f"calibration_fill_{id_slug}_{z_slug}.gcode"
    wedge_out = WORK_DIR / f"calibration_wedge_{id_slug}_{z_slug}.gcode"

    common = dict(
        z_up=float(body.get("z_up", profile.get("z_up", 3.5))),
        z_down=float(body.get("z_down", profile.get("z_down", -1.0))),
        draw_speed=int(body.get("draw_speed", profile.get("draw_speed", 3000))),
        travel_speed=int(body.get("travel_speed", profile.get("travel_speed", 8000))),
        plunge_speed=int(body.get("plunge_speed", profile.get("plunge_speed", 500))),
        lift_speed=int(body.get("lift_speed", profile.get("lift_speed", 2000))),
        end_policy=str(body.get("end_policy", "release")),
        quiet_fans=bool(body.get("quiet_fans", True)),
        profile=profile,
    )

    # Placement fallback: the UI sends paper-derived offsets so a
    # calibration chart lands where the positioning plate shows it. But a
    # paper-space Y is NOT the same quantity as these generators' offset_y --
    # _wedge_gcode's own default folds in the machine's pen_offset_y (-45mm here)
    # because the two-block chart is ~176mm tall and barely fits the 212mm reach
    # span. A raw paper offset skips that and can push the chart below the reach
    # floor, in which case validate_gcode_xy_motion refuses to emit it. The wedge
    # defaults to a single ~85mm-tall block, which fits with headroom; the
    # opt-in two-block pressure_compare=True path still needs the full ~176mm
    # span.
    #
    # A calibration chart's position on the sheet carries NO measurement meaning
    # (unlike its geometry, which is emitted verbatim). So: honour the requested
    # placement when it is reachable, and fall back to the generator's own
    # known-good default when it is not, rather than failing with no chart. The fallback is reported so
    # it is never silent -- the user is told the chart moved and why.
    placement_notes = []

    def _gen_with_fallback(fn, out_path, kind, off_x, off_y, **kw):
        """Try the requested placement; on a reach failure retry at the
        generator's default (offset_x/offset_y=None) and record a note."""
        if off_x is not None or off_y is not None:
            try:
                return fn(out_path, offset_x=off_x, offset_y=off_y, **kw)
            except Exception as exc:                       # reach or layout failure
                placement_notes.append(
                    f"The {kind} did not fit the machine's reach at your paper's current "
                    f"position, so it was placed at its default safe spot instead "
                    f"(move the paper to match, or reposition the paper further from the "
                    f"front edge). Reason: {exc}")
        return fn(out_path, offset_x=None, offset_y=None, **kw)

    try:
        fill_info = _gen_with_fallback(
            _calibration_gcode, fill_out, "fill/spacing test",
            body.get("fill_offset_x"), body.get("fill_offset_y"),
            pen_mm=float(body.get("pen_width", profile.get("pen_mm", 0.5))), **common)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Calibration pack failed generating the fill test for '{pen_label}': {e}"}), 500

    try:
        wedge_info = _gen_with_fallback(
            _wedge_gcode, wedge_out, "gray wedge",
            body.get("wedge_offset_x"), body.get("wedge_offset_y"),
            pen_mm=float(body.get("fill_mm", profile.get("fill_mm", 0.25))),
            pressure_compare=bool(body.get("pressure_compare", False)), **common)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Calibration pack failed generating the gray wedge for '{pen_label}': {e}"}), 500

    stamp = f"?t={uuid4().hex[:6]}"
    return jsonify({
        "ok": True, "pen_id": pen_id, "pen_label": pen_label,
        "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
        "placement_notes": placement_notes,
        "fill": {"file": fill_out.name, "gcode_url": f"/work/{fill_out.name}",
                 "preview_url": f"/work/{fill_info['preview']}{stamp}",
                 "size_kb": round(fill_out.stat().st_size / 1024, 1),
                 "z_up_abs": fill_info["z_up_abs"], "z_down_abs": fill_info["z_down_abs"],
                 "place": fill_info["place"], "pattern_mm": fill_info["pattern_mm"],
                 "safety_selfcheck": fill_info["safety_selfcheck"],
                 "svg_name": Path(fill_info["svg_path"]).name},
        "wedge": {"file": wedge_out.name, "gcode_url": f"/work/{wedge_out.name}",
                  "preview_url": f"/work/{wedge_info['preview']}{stamp}",
                  "size_kb": round(wedge_out.stat().st_size / 1024, 1),
                  "z_up_abs": wedge_info["z_up_abs"], "z_down_abs": wedge_info["z_down_abs"],
                  "place": wedge_info["place"], "pattern_mm": wedge_info["pattern_mm"],
                  "safety_selfcheck": wedge_info["safety_selfcheck"],
                  "svg_name": Path(wedge_info["svg_path"]).name},
    })


@app.route("/api/calibration_combo", methods=["POST"])
def api_calibration_combo():
    """All-in-one calibration sheet (the UI button target; the two-plot
    /api/calibration_pack remains available): ONE true-size gcode
    file bundling the gray wedge + fill-spacing swatches + min-ink strip,
    driven by whichever pen is active in the dropdown (or an explicit
    pen_id) -- see ccplotter/calibration_plots.py _combo_gcode. Modeled
    directly on api_calibration_pack above (same pen/Z resolution, same
    pen-identifying filename convention, same placement-fallback behavior),
    just for one generator/one plot instead of two."""
    from ccplotter.cal import resolve_z_touch, ZTouchNotCalibrated
    body = request.json or {}
    try:
        profile = resolve_pen_profile(body.get("pen_id"))
    except UnknownPenProfile as e:
        return jsonify({"error": str(e)}), 400
    _pd = load_pen_profiles()
    pen_id = body.get("pen_id") or _pd.get("active") or "pen"
    pen_label = profile.get("label") or pen_id

    try:
        z_touch, z_touch_source = resolve_z_touch(None, profile)
    except ZTouchNotCalibrated as e:
        return jsonify({"error": f"Cannot generate calibration combo for '{pen_label}': {e}"}), 400

    z_slug = f"z{int(round(z_touch * 100))}"
    id_slug = str(pen_id).strip() or "pen"
    combo_out = WORK_DIR / f"calibration_combo_{id_slug}_{z_slug}.gcode"

    common = dict(
        z_up=float(body.get("z_up", profile.get("z_up", 3.5))),
        z_down=float(body.get("z_down", profile.get("z_down", -1.0))),
        draw_speed=int(body.get("draw_speed", profile.get("draw_speed", 3000))),
        travel_speed=int(body.get("travel_speed", profile.get("travel_speed", 8000))),
        plunge_speed=int(body.get("plunge_speed", profile.get("plunge_speed", 500))),
        lift_speed=int(body.get("lift_speed", profile.get("lift_speed", 2000))),
        end_policy=str(body.get("end_policy", "release")),
        quiet_fans=bool(body.get("quiet_fans", True)),
        profile=profile,
    )

    # Same placement-fallback convention as api_calibration_pack's
    # _gen_with_fallback: honour a requested offset (paper-derived, from the
    # UI) when it's reachable, else fall back to the generator's own
    # known-good default and report why -- never fail outright.
    placement_notes = []

    def _gen_with_fallback(fn, out_path, kind, off_x, off_y, **kw):
        if off_x is not None or off_y is not None:
            try:
                return fn(out_path, offset_x=off_x, offset_y=off_y, **kw)
            except Exception as exc:                       # reach or layout failure
                placement_notes.append(
                    f"The {kind} did not fit the machine's reach at your paper's current "
                    f"position, so it was placed at its default safe spot instead "
                    f"(move the paper to match, or reposition the paper further from the "
                    f"front edge). Reason: {exc}")
        return fn(out_path, offset_x=None, offset_y=None, **kw)

    try:
        combo_info = _gen_with_fallback(
            _combo_gcode, combo_out, "calibration sheet",
            body.get("combo_offset_x"), body.get("combo_offset_y"),
            pen_mm=float(body.get("pen_width", profile.get("pen_mm", 0.5))), **common)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Calibration combo failed generating the sheet for '{pen_label}': {e}"}), 500

    stamp = f"?t={uuid4().hex[:6]}"
    return jsonify({
        "ok": True, "pen_id": pen_id, "pen_label": pen_label,
        "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
        "placement_notes": placement_notes,
        "combo": {"file": combo_out.name, "gcode_url": f"/work/{combo_out.name}",
                  "preview_url": f"/work/{combo_info['preview']}{stamp}",
                  "size_kb": round(combo_out.stat().st_size / 1024, 1),
                  "z_up_abs": combo_info["z_up_abs"], "z_down_abs": combo_info["z_down_abs"],
                  "place": combo_info["place"], "pattern_mm": combo_info["pattern_mm"],
                  "safety_selfcheck": combo_info["safety_selfcheck"],
                  "sections": combo_info["sections"],
                  "svg_name": Path(combo_info["svg_path"]).name},
    })


@app.route("/api/calibration_block", methods=["POST"])
def api_calibration_block():
    """ONE calibration block per sheet.

    kind: wedge | swatch | minink | linewidth

    The user composes the sheet on the canvas from whichever blocks they want,
    so the fixed layout of the all-in-one /api/calibration_combo (and the paper
    height it has to fit) is not a constraint.

    `linewidth` measures the pen's DEPOSITED LINE WIDTH, which no other block
    does."""
    from ccplotter.cal import resolve_z_touch, ZTouchNotCalibrated
    body = request.json or {}
    kind = str(body.get("kind", "")).strip().lower()
    if kind not in CAL_BLOCKS:
        return jsonify({"error": f"Unknown calibration block '{kind}'. "
                                 f"Expected one of: {', '.join(sorted(CAL_BLOCKS))}."}), 400
    try:
        profile = resolve_pen_profile(body.get("pen_id"))
    except UnknownPenProfile as e:
        return jsonify({"error": str(e)}), 400
    _pd = load_pen_profiles()
    pen_id = body.get("pen_id") or _pd.get("active") or "pen"
    pen_label = profile.get("label") or pen_id

    try:
        z_touch, z_touch_source = resolve_z_touch(None, profile)
    except ZTouchNotCalibrated as e:
        return jsonify({"error": f"Cannot generate the {CAL_BLOCKS[kind]} for '{pen_label}': {e}"}), 400

    z_slug = f"z{int(round(z_touch * 100))}"
    id_slug = str(pen_id).strip() or "pen"
    out = WORK_DIR / f"calibration_{kind}_{id_slug}_{z_slug}.gcode"

    common = dict(
        kind=kind,
        z_up=float(body.get("z_up", profile.get("z_up", 3.5))),
        z_down=float(body.get("z_down", profile.get("z_down", -1.0))),
        draw_speed=int(body.get("draw_speed", profile.get("draw_speed", 3000))),
        travel_speed=int(body.get("travel_speed", profile.get("travel_speed", 8000))),
        plunge_speed=int(body.get("plunge_speed", profile.get("plunge_speed", 500))),
        lift_speed=int(body.get("lift_speed", profile.get("lift_speed", 2000))),
        end_policy=str(body.get("end_policy", "release")),
        quiet_fans=bool(body.get("quiet_fans", True)),
        profile=profile,
        pen_mm=float(body.get("pen_width", profile.get("pen_mm", 0.5))),
    )

    placement_notes = []
    off_x, off_y = body.get("offset_x"), body.get("offset_y")
    try:
        if off_x is not None or off_y is not None:
            try:
                info = _calblock_gcode(out, offset_x=off_x, offset_y=off_y, **common)
            except Exception as exc:
                placement_notes.append(
                    f"The {CAL_BLOCKS[kind]} did not fit the machine's reach at your paper's "
                    f"current position, so it was placed at its default safe spot instead. "
                    f"Reason: {exc}")
                info = _calblock_gcode(out, offset_x=None, offset_y=None, **common)
        else:
            info = _calblock_gcode(out, offset_x=None, offset_y=None, **common)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"{CAL_BLOCKS[kind]} failed for '{pen_label}': {e}"}), 500

    stamp = f"?t={uuid4().hex[:6]}"
    return jsonify({
        "ok": True, "kind": kind, "label": CAL_BLOCKS[kind],
        "pen_id": pen_id, "pen_label": pen_label,
        "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
        "placement_notes": placement_notes,
        "block": {"file": out.name, "gcode_url": f"/work/{out.name}",
                  "preview_url": f"/work/{info['preview']}{stamp}",
                  "size_kb": round(out.stat().st_size / 1024, 1),
                  "z_up_abs": info["z_up_abs"], "z_down_abs": info["z_down_abs"],
                  "place": info["place"], "pattern_mm": info["pattern_mm"],
                  "safety_selfcheck": info["safety_selfcheck"],
                  "svg_name": Path(info["svg_path"]).name},
    })


@app.route("/api/calibration_deposition", methods=["POST"])
def api_calibration_deposition():
    """Deposition sweep sheet -- why solid fills crack on paper.

    Deliberately NOT a `kind` of /api/calibration_block, because it cannot obey
    that route's contract. Every other block is one geometry at ONE z_down and
    ONE draw_speed, which is why the canvas flow works for them: the block's own
    g-code is a placement artifact, and the sheet that is actually plotted is
    re-emitted by /api/process from the canvas at the UI's single Z and speed.

    This sheet's entire content is a SWEEP of z_down and draw_speed across 20
    cells. Re-emitting it through the composite path would collapse all 20 cells
    onto one Z and one speed and silently destroy the experiment -- it would
    still plot, and still look like a calibration sheet, which is precisely what
    makes it dangerous. So:

      * the pattern SVG is returned for /api/calibration_import, and adding it
        to the canvas is a POSITION PREVIEW ONLY -- check it lands on the paper;
      * the file to sideload is `gcode_url` from THIS response, never a canvas
        plot.

    The response says so in `plot_warning`, and the UI repeats it.

    Placement follows api_calibration_block: offset_x/offset_y are the paper's
    current machine position, falling back to the generator's own safe default
    if the sheet does not fit the pen's reach there."""
    from ccplotter.cal import resolve_z_touch, ZTouchNotCalibrated
    from ccplotter.calibration_deposition import (deposition_gcode, Z_DOWN_LEVELS,
                                                  DRAW_SPEEDS, PITCH_MULTS)
    body = request.json or {}
    try:
        profile = resolve_pen_profile(body.get("pen_id"))
    except UnknownPenProfile as e:
        return jsonify({"error": str(e)}), 400
    _pd = load_pen_profiles()
    pen_id = body.get("pen_id") or _pd.get("active") or "pen"
    pen_label = profile.get("label") or pen_id

    try:
        z_touch, z_touch_source = resolve_z_touch(None, profile)
    except ZTouchNotCalibrated as e:
        return jsonify({"error": f"Cannot generate the deposition sheet for "
                                 f"'{pen_label}': {e}"}), 400

    z_slug = f"z{int(round(z_touch * 100))}"
    out = WORK_DIR / f"calibration_deposition_{str(pen_id).strip() or 'pen'}_{z_slug}.gcode"

    common = dict(
        z_up=float(body.get("z_up", profile.get("z_up", 3.5))),
        travel_speed=int(body.get("travel_speed", profile.get("travel_speed", 8000))),
        plunge_speed=int(body.get("plunge_speed", profile.get("plunge_speed", 500))),
        lift_speed=int(body.get("lift_speed", profile.get("lift_speed", 2000))),
        end_policy=str(body.get("end_policy", "release")),
        quiet_fans=bool(body.get("quiet_fans", True)),
        profile=profile,
    )
    # z_down and draw_speed are NOT taken from the request: they are the swept
    # axes. Accepting them here would look like it worked and quietly flatten
    # the sweep.
    placement_notes = []
    off_x, off_y = body.get("offset_x"), body.get("offset_y")
    try:
        if off_x is not None or off_y is not None:
            try:
                info = deposition_gcode(out, offset_x=off_x, offset_y=off_y, **common)
            except Exception as exc:
                placement_notes.append(
                    "The deposition sheet did not fit the machine's reach at your paper's "
                    f"current position, so it was placed at its default safe spot instead. "
                    f"Reason: {exc}")
                info = deposition_gcode(out, offset_x=None, offset_y=None, **common)
        else:
            info = deposition_gcode(out, offset_x=None, offset_y=None, **common)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Deposition sheet failed for '{pen_label}': {e}"}), 500

    stamp = f"?t={uuid4().hex[:6]}"
    return jsonify({
        "ok": True, "kind": "deposition", "label": "Deposition sweep (why fills crack)",
        "pen_id": pen_id, "pen_label": pen_label,
        "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
        "placement_notes": placement_notes,
        "plot_warning": ("Plot the linked g-code file directly. Adding this sheet to the "
                         "canvas shows you WHERE it lands, but plotting from the canvas "
                         "would re-emit all 20 cells at one Z and one speed and destroy "
                         "the sweep."),
        "sweep": {"z_down_levels": list(Z_DOWN_LEVELS), "draw_speeds": list(DRAW_SPEEDS),
                  "pitch_mults": list(PITCH_MULTS), "n_cells": info["n_cells"],
                  "n_patches": info["n_patches"],
                  "deepest_z_down_abs": info["deepest_z_down_abs"]},
        "block": {"file": out.name, "gcode_url": f"/work/{out.name}",
                  "preview_url": f"/work/{info['preview']}{stamp}",
                  "size_kb": round(out.stat().st_size / 1024, 1),
                  "z_up_abs": info["z_up_abs"], "z_down_abs": info["z_down_abs"],
                  "place": info["place"], "pattern_mm": info["pattern_mm"],
                  "safety_selfcheck": info["safety_selfcheck"],
                  "svg_name": Path(info["svg_path"]).name},
    })


@app.route("/api/calibration_penwidth", methods=["POST"])
def api_calibration_penwidth():
    """Laid-WIDTH-DISTRIBUTION sheet: `laid_mm` is only ever a median with no
    surviving tails, and this sheet measures the distribution.

    Same "sweep sheet" shape and warning as /api/calibration_deposition just
    above (which this route is modeled on): z_down and
    draw_speed are SWEPT AXES baked into 16 separate gwrite blocks and spliced
    into one g-code file (ccplotter/calibration_penwidth.py) -- they are NOT
    accepted from the request body, because doing so would look like it
    worked while silently flattening the sweep. The pattern SVG is returned
    for /api/calibration_import as a POSITION PREVIEW ONLY; the file to
    sideload is always `gcode_url` from THIS response, never a canvas plot
    re-emitted through /api/process (which would collapse all 16 cells onto
    one Z and one speed)."""
    from ccplotter.cal import resolve_z_touch, ZTouchNotCalibrated
    from ccplotter.calibration_penwidth import (penwidth_gcode, Z_DOWN_LEVELS,
                                                 DRAW_SPEEDS, ANGLES_DEG)
    body = request.json or {}
    try:
        profile = resolve_pen_profile(body.get("pen_id"))
    except UnknownPenProfile as e:
        return jsonify({"error": str(e)}), 400
    _pd = load_pen_profiles()
    pen_id = body.get("pen_id") or _pd.get("active") or "pen"
    pen_label = profile.get("label") or pen_id

    try:
        z_touch, z_touch_source = resolve_z_touch(None, profile)
    except ZTouchNotCalibrated as e:
        return jsonify({"error": f"Cannot generate the penwidth sheet for "
                                 f"'{pen_label}': {e}"}), 400

    z_slug = f"z{int(round(z_touch * 100))}"
    out = WORK_DIR / f"calibration_penwidth_{str(pen_id).strip() or 'pen'}_{z_slug}.gcode"

    common = dict(
        z_up=float(body.get("z_up", profile.get("z_up", 3.5))),
        travel_speed=int(body.get("travel_speed", profile.get("travel_speed", 8000))),
        plunge_speed=int(body.get("plunge_speed", profile.get("plunge_speed", 500))),
        lift_speed=int(body.get("lift_speed", profile.get("lift_speed", 2000))),
        end_policy=str(body.get("end_policy", "release")),
        quiet_fans=bool(body.get("quiet_fans", True)),
        profile=profile,
    )
    # z_down and draw_speed are NOT taken from the request: they are the
    # swept axes (see docstring above and calibration_deposition's identical
    # convention). Accepting them here would look like it worked and quietly
    # flatten the sweep.
    placement_notes = []
    off_x, off_y = body.get("offset_x"), body.get("offset_y")
    try:
        if off_x is not None or off_y is not None:
            try:
                info = penwidth_gcode(out, offset_x=off_x, offset_y=off_y, **common)
            except Exception as exc:
                placement_notes.append(
                    "The penwidth sheet did not fit the machine's reach at your paper's "
                    f"current position, so it was placed at its default safe spot instead. "
                    f"Reason: {exc}")
                info = penwidth_gcode(out, offset_x=None, offset_y=None, **common)
        else:
            info = penwidth_gcode(out, offset_x=None, offset_y=None, **common)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Penwidth sheet failed for '{pen_label}': {e}"}), 500

    stamp = f"?t={uuid4().hex[:6]}"
    return jsonify({
        "ok": True, "kind": "penwidth", "label": "Pen width distribution (laid-width tails)",
        "pen_id": pen_id, "pen_label": pen_label,
        "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
        "placement_notes": placement_notes,
        "plot_warning": ("Plot the linked g-code file directly. Adding this sheet to the "
                         "canvas shows you WHERE it lands, but plotting from the canvas "
                         "would re-emit all 16 cells at one Z and one speed and destroy "
                         "the sweep."),
        "sweep": {"z_down_levels": list(Z_DOWN_LEVELS), "draw_speeds": list(DRAW_SPEEDS),
                  "angles_deg": list(ANGLES_DEG), "n_cells": info["n_cells"],
                  "n_strokes": info["n_strokes"],
                  "stroke_len_total_mm": info["stroke_len_total_mm"],
                  "deepest_z_down_abs": info["deepest_z_down_abs"]},
        "block": {"file": out.name, "gcode_url": f"/work/{out.name}",
                  "preview_url": f"/work/{info['preview']}{stamp}",
                  "size_kb": round(out.stat().st_size / 1024, 1),
                  "z_up_abs": info["z_up_abs"], "z_down_abs": info["z_down_abs"],
                  "place": info["place"], "pattern_mm": info["pattern_mm"],
                  "safety_selfcheck": info["safety_selfcheck"],
                  "svg_name": Path(info["svg_path"]).name},
    })


@app.route("/api/calibration_sheets", methods=["GET"])
def api_calibration_sheets():
    """The calibration-sheet catalogue, so the UI can render its buttons from a
    registry instead of a hand-written handler per sheet; every sheet has a
    preview.

    Each entry's `ui` tells the client how to drive it: "generic" goes through
    POST /api/calibration_sheet below; "bespoke" has its own route and its own
    handler and is listed only so the catalogue is complete. The two mechanisms
    coexist DELIBERATELY -- see calibration_registry's module docstring."""
    from ccplotter.calibration_registry import catalogue
    # Archived sheets (ones whose question has been answered) are omitted by
    # default so the button list stays short; ?all=1 returns them. They stay
    # registered and dispatchable either way -- see CalSheet.archived.
    _all = str(request.args.get("all", "")).lower() in ("1", "true", "yes", "on")
    return jsonify({"ok": True, "sheets": catalogue(include_archived=_all)})


@app.route("/api/calibration_sheet", methods=["POST"])
def api_calibration_sheet():
    """Generic dispatcher for registry sheets with a uniform generator.

    THIS ONLY DISPATCHES. Every safety guard stays inside the generator it
    belongs to -- each one keeps its own hard RuntimeError shadow guards
    (resolved profile id, reach bbox, pen, z_touch) and its own reach
    validation. The dispatcher must never become the place a guard is skipped,
    so the checks below mirror the per-sheet routes exactly
    rather than relaxing any of them:
      * pen profile resolution, with UnknownPenProfile -> 400
      * resolve_z_touch, with ZTouchNotCalibrated -> 400
      * the offset placement fallback, with its user-visible note
    `z_down` and `draw_speed` are NOT accepted from the request body for any
    sheet: they are swept axes baked into spliced gwrite blocks, and taking
    them here would look like it worked while silently flattening the sweep."""
    from ccplotter.cal import resolve_z_touch, ZTouchNotCalibrated
    from ccplotter.calibration_registry import SHEETS
    body = request.json or {}
    sheet_id = str(body.get("id") or "").strip()
    sheet = SHEETS.get(sheet_id)
    if sheet is None:
        return jsonify({"error": f"Unknown calibration sheet {sheet_id!r}. "
                                 f"Known: {', '.join(sorted(SHEETS))}"}), 400
    if sheet.ui != "generic" or sheet.generator is None:
        return jsonify({"error": f"Sheet {sheet_id!r} has its own dedicated route "
                                 f"(/api/calibration_{sheet_id}); it is not driven by the "
                                 f"generic dispatcher."}), 400

    try:
        profile = resolve_pen_profile(body.get("pen_id"))
    except UnknownPenProfile as e:
        return jsonify({"error": str(e)}), 400
    _pd = load_pen_profiles()
    pen_id = body.get("pen_id") or _pd.get("active") or "pen"
    pen_label = profile.get("label") or pen_id

    try:
        z_touch, z_touch_source = resolve_z_touch(None, profile)
    except ZTouchNotCalibrated as e:
        return jsonify({"error": f"Cannot generate the {sheet.label} sheet for "
                                 f"'{pen_label}': {e}"}), 400

    z_slug = f"z{int(round(z_touch * 100))}"
    out = WORK_DIR / f"calibration_{sheet_id}_{str(pen_id).strip() or 'pen'}_{z_slug}.gcode"

    common = dict(
        z_up=float(body.get("z_up", profile.get("z_up", 3.5))),
        travel_speed=int(body.get("travel_speed", profile.get("travel_speed", 8000))),
        plunge_speed=int(body.get("plunge_speed", profile.get("plunge_speed", 500))),
        lift_speed=int(body.get("lift_speed", profile.get("lift_speed", 2000))),
        end_policy=str(body.get("end_policy", "release")),
        quiet_fans=bool(body.get("quiet_fans", True)),
        profile=profile,
    )
    # ladder_mm_lo/ladder_mm_hi is an EXPLICIT per-request override of
    # penwidth_wedge's pitch bracket -- see calibration_penwidth_wedge.py.
    # Only forwarded for that sheet (every other
    # generator's signature does not accept these kwargs and would TypeError
    # on an unexpected keyword); absent from the body, `common` is unchanged
    # and every other sheet's call is untouched.
    if sheet_id == "penwidth_wedge":
        _lad_lo, _lad_hi = body.get("ladder_mm_lo"), body.get("ladder_mm_hi")
        if _lad_lo is not None or _lad_hi is not None:
            if _lad_lo is None or _lad_hi is None:
                return jsonify({"error": "ladder_mm_lo and ladder_mm_hi must be given "
                                         "together."}), 400
            try:
                _lad_lo, _lad_hi = float(_lad_lo), float(_lad_hi)
            except (TypeError, ValueError):
                return jsonify({"error": "ladder_mm_lo/ladder_mm_hi must be numbers."}), 400
            if not (0 < _lad_lo < _lad_hi):
                return jsonify({"error": "ladder_mm_lo/ladder_mm_hi must satisfy "
                                         f"0 < lo < hi (got lo={_lad_lo} hi={_lad_hi})."}), 400
            common["ladder_mm_lo"] = _lad_lo
            common["ladder_mm_hi"] = _lad_hi
    placement_notes = []
    off_x, off_y = body.get("offset_x"), body.get("offset_y")
    try:
        if off_x is not None or off_y is not None:
            try:
                info = sheet.generator(out, offset_x=off_x, offset_y=off_y, **common)
            except Exception as exc:
                placement_notes.append(
                    f"The {sheet.label} sheet did not fit the machine's reach at your paper's "
                    f"current position, so it was placed at its default safe spot instead. "
                    f"Reason: {exc}")
                info = sheet.generator(out, offset_x=None, offset_y=None, **common)
        else:
            info = sheet.generator(out, offset_x=None, offset_y=None, **common)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"{sheet.label} sheet failed for '{pen_label}': {e}"}), 500

    try:   # flag this sheet's preview svg so its canvas import is gcode-refused
        atomic_write_bytes(WORK_DIR / f"calsheet_{Path(info['svg_path']).stem}.flag", b"1\n")
    except Exception:
        pass
    # Leave a run record (sha256 + profile + params) like a job does. Gcode untouched.
    from ccplotter.runs import record_sheet_provenance
    _rec = record_sheet_provenance(sheet_id, out, pen_id, profile, body, info,
                                   z_touch, z_touch_source)

    stamp = f"?t={uuid4().hex[:6]}"
    sweep = {k: v for k, v in info.items()
             if k not in ("preview", "svg_path", "place", "pattern_mm", "safety_selfcheck",
                          "z_up_abs", "z_down_abs", "kind", "pen_mm", "pen_id")}
    return jsonify({
        "ok": True, "kind": sheet_id, "label": sheet.label, "blurb": sheet.blurb,
        "pen_id": pen_id, "pen_label": pen_label,
        "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
        "placement_notes": placement_notes,
        "plot_warning": sheet.plot_warning,
        "sweep": sweep,
        "provenance": _rec,
        "block": {"file": out.name, "gcode_url": f"/work/{out.name}",
                  "preview_url": f"/work/{info['preview']}{stamp}",
                  "size_kb": round(out.stat().st_size / 1024, 1),
                  "z_up_abs": info.get("z_up_abs"), "z_down_abs": info.get("z_down_abs"),
                  "place": info.get("place"), "pattern_mm": info.get("pattern_mm"),
                  "safety_selfcheck": info.get("safety_selfcheck"),
                  "svg_name": Path(info["svg_path"]).name},
    })


@app.route("/api/registration_marks", methods=["POST"])
def api_registration_marks():
    """Build-plate registration marks: plot ON THE
    BARE PLATE (no paper) so the plate can be marked at the pen marks and
    real paper aligned to them afterward -- see
    ccplotter/calibration_plots.py _registration_marks_gcode for the design
    rationale (absolute-machine-coord, NOT the true_size drawing-import
    path). Modeled on api_calibration_plot for pen/Z resolution and on
    api_calibration_pack for the response shape, but with ONE material
    difference: paper_w/paper_h/offset_x/offset_y are REQUIRED here (the
    current UI paper placement, same offset_x/offset_y convention
    genCalibrationCombo() already sends) -- there is no safe generator-
    chosen default position to fall back to, because the whole point is
    marking wherever the paper ACTUALLY is. A missing/invalid placement is a
    clean 400, not a silent reposition.

    Both off-sheet edge-reference lines are mandatory (see the generator's own
    docstring): if the current placement puts either outside reach, this
    route refuses rather than returning a partial physical reference."""
    from ccplotter.cal import resolve_z_touch, ZTouchNotCalibrated
    body = request.json or {}
    try:
        paper_w = float(body["paper_w"]); paper_h = float(body["paper_h"])
        offset_x = float(body["offset_x"]); offset_y = float(body["offset_y"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "paper_w, paper_h, offset_x and offset_y are required "
                                  "(the current paper placement) -- registration marks have "
                                  "no meaningful default position."}), 400
    if paper_w <= paper_h:
        return jsonify({"error": "Registration marks require landscape paper. B5 JIS must be 257×182mm "
                                  "so the two off-sheet edge references can fit the confirmed reach."}), 400
    try:
        profile = resolve_pen_profile(body.get("pen_id"))
    except UnknownPenProfile as e:
        return jsonify({"error": str(e)}), 400
    _pd = load_pen_profiles()
    pen_id = body.get("pen_id") or _pd.get("active") or "pen"
    pen_label = profile.get("label") or pen_id

    try:
        z_touch, z_touch_source = resolve_z_touch(None, profile)
    except ZTouchNotCalibrated as e:
        return jsonify({"error": f"Cannot generate registration marks for '{pen_label}': {e}"}), 400

    z_slug = f"z{int(round(z_touch * 100))}"
    id_slug = str(pen_id).strip() or "pen"
    reg_out = WORK_DIR / f"registration_marks_{id_slug}_{z_slug}.gcode"

    try:
        info = _registration_marks_gcode(
            reg_out, paper_w=paper_w, paper_h=paper_h, offset_x=offset_x, offset_y=offset_y,
            pen_mm=float(body.get("pen_width", profile.get("pen_mm", 0.5))),
            z_up=float(body.get("z_up", profile.get("z_up", 3.5))),
            z_down=float(body.get("z_down", profile.get("z_down", -1.0))),
            draw_speed=int(body.get("draw_speed", profile.get("draw_speed", 3000))),
            travel_speed=int(body.get("travel_speed", profile.get("travel_speed", 8000))),
            plunge_speed=int(body.get("plunge_speed", profile.get("plunge_speed", 500))),
            lift_speed=int(body.get("lift_speed", profile.get("lift_speed", 2000))),
            end_policy=str(body.get("end_policy", "release")),
            quiet_fans=bool(body.get("quiet_fans", True)),
            profile=profile,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Registration marks failed for '{pen_label}': {e}"}), 500

    stamp = f"?t={uuid4().hex[:6]}"
    return jsonify({
        "ok": True, "pen_id": pen_id, "pen_label": pen_label,
        "z_touch_abs": round(z_touch, 3), "z_touch_source": z_touch_source,
        "registration": {
            "file": reg_out.name, "gcode_url": f"/work/{reg_out.name}",
            "preview_url": f"/work/{info['preview']}{stamp}",
            "size_kb": round(reg_out.stat().st_size / 1024, 1),
            "z_up_abs": info["z_up_abs"], "z_down_abs": info["z_down_abs"],
            "paper_mm": info["paper_mm"], "offset_mm": info["offset_mm"],
            "reference_edge_offset_mm": info["reference_edge_offset_mm"],
            "corners_machine": info["corners_machine"],
            "skipped_corners": info["skipped"],
            "machine_bbox": info["machine_bbox"],
            "safety_selfcheck": info["safety_selfcheck"],
        },
    })


@app.route("/api/calibration_import", methods=["POST"])
def api_calibration_import():
    """Import a generated calibration pattern SVG as a normal vector DRAWING --
    the same 'existing on-disk file -> fresh src_<id>.svg' pattern
    /api/dbv3_import (ccplotter/routes/core.py) and /api/runs/<rid>/use_source
    already use. Once imported it is an ordinary state.drawings item and goes
    through the completely unmodified /api/process pipeline: Fit, positioning,
    reach checks, preview and gcode generation all come from the same code
    every other drawing uses. This is how a calibration plot can be positioned
    on the paper and previewed like any other drawing.

    The pattern SVG is written in true physical mm (CSS px @ 96dpi, same
    PX2MM convention every other SVG source in this app uses -- see
    _calibration_gcode / _wedge_gcode in gcode_post.py / calibration_plots.py),
    so importing it does not change its measured size.

    A calibration plot is a MEASUREMENT INSTRUMENT: its ladders/spacings only
    mean what they say at scale 1.0. The response is flagged
    is_calibration=True so the frontend pins the item's scale (a client-side
    item flag the Fit/Fill-Page/scale-field paths honour -- see
    templates/index.html drawPreset/syncSelectedFromControls/pointerup)."""
    body = request.json or {}
    svg_name = str(body.get("svg_name", ""))
    if not svg_name or Path(svg_name).name != svg_name or Path(svg_name).suffix.lower() != ".svg":
        return jsonify({"error": "invalid svg_name"}), 400
    src = WORK_DIR / svg_name
    if not src.is_file():
        return jsonify({"error": "calibration pattern not found on disk -- generate the pack again"}), 404
    did = uuid4().hex[:8]
    dst = WORK_DIR / f"src_{did}.svg"
    atomic_write_bytes(dst, src.read_bytes())
    label = str(body.get("label") or "Calibration plot")
    # svg written by the registry dispatcher -> mark the imported drawing so
    # /api/process refuses canvas gcode of it (marker name does not start with src_).
    _is_sheet = (WORK_DIR / f"calsheet_{Path(svg_name).stem}.flag").is_file()
    if _is_sheet:
        atomic_write_bytes(WORK_DIR / f"calsheetmark_{did}.flag", b"registry sheet preview\n")
    return jsonify({"id": did, "name": label, "is_raster": False, "src_name": dst.name,
                    "is_calibration": True, "source_svg": svg_name,
                    "is_cal_sheet": _is_sheet})



def _deposit_mm():
    """Width of the mark the ink makes, for the gap-survivability checks.

    _gap_audit and _preflight ask "is this white gap wider than the mark the
    ink makes". That is a PEN width, not `fill_mm` (the fill pass SPACING).
    Testing gaps against fill_mm (0.175mm on uright_gel) when the pen lays
    0.255mm would call gaps survivable that the pen closes.

    Uses `laid_mm` (MEASURED) when the profile has one, and otherwise falls back
    to fill_mm. Deliberately NOT pen_mm: that is a
    nominal figure and is wrong by 2x on the one pen we have measured.
    """
    prof = active_pen_profile()
    laid = prof.get("laid_mm")
    if laid is not None:
        return float(laid)
    return float(prof.get("fill_mm", 0.15))


@app.route("/api/coverage_debug", methods=["POST"])
def api_coverage_debug():
    """Source-faithful coverage DEBUG renderer (measurement only — no strokes).
    Renders source/ink/sim-output/underfill-overfill/orientation for a base
    ('lineart' or 'sketchv2') so the coverage gap is reviewable in the app."""
    body = request.json or {}
    did = str(body.get("id", ""))
    base = str(body.get("base", "sketchv2")).lower()
    if base not in ("lineart", "sketchv2", "tracev3"):
        return jsonify({"error": "base must be 'lineart', 'sketchv2' or 'tracev3'"}), 400
    raw = [m for m in WORK_DIR.glob(f"src_{did}.*")
           if "_render_" not in m.name and not m.name.endswith("_vector.svg")]
    if not raw:
        return jsonify({"error": f"drawing {did} not found"}), 404
    if raw[0].suffix.lower() not in RASTER_EXTS:
        return jsonify({"error": "Coverage debug is for raster sources."}), 400
    try:
        paper = body.get("paper", "180x180").lower()
        pw, ph = PAPER_SIZES[paper] if paper in PAPER_SIZES else (
            float(paper.split('x')[0]), float(paper.split('x')[1]))
    except Exception:
        pw, ph = 180.0, 180.0
    try:
        r = _coverage_debug(raw[0], base_mode=base,
                            resolution=int(body.get("resolution", 2600)),
                            paper_w_mm=pw, margin_mm=float(body.get("margin", 10)),
                            pen_mm=float(body.get("pen_width", 0.5)),
                            tone=str(body.get("tone", "off")),
                            paper_h_mm=ph)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Coverage debug failed: {e}"}), 500
    stamp = f"?t={uuid4().hex[:6]}"
    return jsonify({"ok": True, "base": base, "stats": r["stats"], "paths": r["paths"],
                    "images": {k: f"/work/{v}{stamp}" for k, v in r["files"].items()}})


@app.route("/api/gap_audit", methods=["POST"])
def api_gap_audit():
    """White-hatch-gap forensics (measurement only, no strokes).
    Tracks gap survival native -> resampled -> binarized so we can see WHICH stage
    kills the shading detail before tuning anything."""
    body = request.json or {}
    did = str(body.get("id", ""))
    raw = [m for m in WORK_DIR.glob(f"src_{did}.*")
           if "_render_" not in m.name and not m.name.endswith("_vector.svg")]
    if not raw:
        return jsonify({"error": f"drawing {did} not found"}), 404
    if raw[0].suffix.lower() not in RASTER_EXTS:
        return jsonify({"error": "Gap audit is for raster sources."}), 400
    try:
        paper = body.get("paper", "180x180").lower()
        pw, ph = PAPER_SIZES[paper] if paper in PAPER_SIZES else (
            float(paper.split('x')[0]), float(paper.split('x')[1]))
    except Exception:
        pw, ph = 180.0, 180.0
    try:
        r = _gap_audit(raw[0], resolution=int(body.get("resolution", 2600)),
                       paper_w_mm=pw, margin_mm=float(body.get("margin", 10)),
                       # deposit_mm is "how wide a mark does the ink make",
                       # which decides whether a white gap survives. That is
                       # the pen's laid width, NOT fill_mm (pass spacing); on
                       # uright_gel the two differ by 1.46x.
                       deposit_mm=_deposit_mm(),
                       paper_h_mm=ph)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Gap audit failed: {e}"}), 500
    stamp = f"?t={uuid4().hex[:6]}"
    return jsonify({"ok": True, "stages": r["stages"], "meta": {k: r[k] for k in
                    ("native_px", "trace_px", "mm_per_px_native", "mm_per_px_trace",
                     "deposit_mm", "drawable_mm")}, "paths": r["paths"],
                    "images": {k: f"/work/{v}{stamp}" for k, v in r["files"].items()}})


@app.route("/api/preflight", methods=["POST"])
def api_preflight():
    """Preflight: source-suitability verdict at current paper/pen. Runs on
    load so 'why is it a slab' is a readout, not a mask-archaeology dig."""
    body = request.json or {}
    did = str(body.get("id", ""))
    raw = [m for m in WORK_DIR.glob(f"src_{did}.*")
           if "_render_" not in m.name and not m.name.endswith("_vector.svg")]
    if not raw:
        return jsonify({"error": f"drawing {did} not found"}), 404
    if raw[0].suffix.lower() not in RASTER_EXTS:
        return jsonify({"error": "Preflight is for raster sources."}), 400
    try:
        paper = body.get("paper", "180x180").lower()
        pw, ph = PAPER_SIZES[paper] if paper in PAPER_SIZES else (
            float(paper.split('x')[0]), float(paper.split('x')[1]))
    except Exception:
        pw, ph = 180.0, 180.0
    try:
        r = _preflight(raw[0], paper_w_mm=pw, paper_h_mm=ph, margin_mm=float(body.get("margin", 10)),
                       deposit_mm=_deposit_mm())
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Preflight failed: {e}"}), 500
    return jsonify({"ok": True, **r})
