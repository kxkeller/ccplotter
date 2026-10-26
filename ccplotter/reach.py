"""Validated machine-coordinate reach profiles and conservative G-code guards."""
import math
import re

from ccplotter.cal import load_cal
from ccplotter.config import PEN_MIN_X, PEN_MIN_Y, PEN_MAX_X, PEN_MAX_Y

_EPS = 1e-7
_LEGACY = {
    "id": "plotter_2001",
    "label": "Plotter 2001 (legacy rectangular)",
    "polygon": [[PEN_MIN_X, PEN_MIN_Y], [PEN_MAX_X, PEN_MIN_Y],
                [PEN_MAX_X, PEN_MAX_Y], [PEN_MIN_X, PEN_MAX_Y]],
    "safe_park_point": [100.0, 100.0],
}


def _finite_point(p):
    return isinstance(p, (list, tuple)) and len(p) == 2 and all(math.isfinite(float(v)) for v in p)


def _cross(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _on_segment(a, b, p, tol=_EPS):
    return abs(_cross(a, b, p)) <= tol and min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol


def _segments_intersect(a, b, c, d):
    ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
    cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
    if ((ab_c > _EPS and ab_d < -_EPS) or (ab_c < -_EPS and ab_d > _EPS)) and ((cd_a > _EPS and cd_b < -_EPS) or (cd_a < -_EPS and cd_b > _EPS)):
        return True
    return _on_segment(a, b, c) or _on_segment(a, b, d) or _on_segment(c, d, a) or _on_segment(c, d, b)


def _validate_polygon(poly):
    if not isinstance(poly, list) or len(poly) < 3:
        raise ValueError("reach polygon needs at least three points")
    out = [[float(p[0]), float(p[1])] for p in poly if _finite_point(p)]
    if len(out) != len(poly) or any(math.hypot(out[i][0] - out[(i + 1) % len(out)][0], out[i][1] - out[(i + 1) % len(out)][1]) <= _EPS for i in range(len(out))):
        raise ValueError("reach polygon has invalid or repeated adjacent points")
    area = sum(out[i][0] * out[(i + 1) % len(out)][1] - out[(i + 1) % len(out)][0] * out[i][1] for i in range(len(out))) / 2.0
    if abs(area) <= _EPS:
        raise ValueError("reach polygon has zero area")
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if j in (i, (i + 1) % len(out)) or i in (j, (j + 1) % len(out)):
                continue
            if _segments_intersect(out[i], out[(i + 1) % len(out)], out[j], out[(j + 1) % len(out)]):
                raise ValueError("reach polygon self-intersects")
    return out


def point_in_polygon(point, polygon, tol=_EPS):
    """Boundary-inclusive containment for a simple polygon."""
    x, y = float(point[0]), float(point[1])
    inside = False
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        if _on_segment(a, b, (x, y), tol):
            return True
        if (a[1] > y) != (b[1] > y):
            hit_x = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if hit_x > x - tol:
                inside = not inside
    return inside


def segment_in_polygon(a, b, polygon, tol=_EPS):
    """True only when the whole segment is inside; catches concave-notch chords."""
    if not point_in_polygon(a, polygon, tol) or not point_in_polygon(b, polygon, tol):
        return False
    ts = [0.0, 1.0]
    dx, dy = b[0] - a[0], b[1] - a[1]
    for i, c in enumerate(polygon):
        d = polygon[(i + 1) % len(polygon)]
        den = dx * (d[1] - c[1]) - dy * (d[0] - c[0])
        if abs(den) <= tol:
            continue
        t = ((c[0] - a[0]) * (d[1] - c[1]) - (c[1] - a[1]) * (d[0] - c[0])) / den
        u = ((c[0] - a[0]) * dy - (c[1] - a[1]) * dx) / den
        if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
            ts.append(max(0.0, min(1.0, t)))
    ts = sorted(set(round(t, 12) for t in ts))
    return all(point_in_polygon((a[0] + (b[0] - a[0]) * ((ts[i] + ts[i + 1]) / 2), a[1] + (b[1] - a[1]) * ((ts[i] + ts[i + 1]) / 2)), polygon, tol) for i in range(len(ts) - 1))


def active_reach_profile(cal=None):
    cal = load_cal() if cal is None else cal
    profiles = cal.get("reach_profiles") if isinstance(cal, dict) else None
    active = cal.get("active_reach_profile") if isinstance(cal, dict) else None
    raw = profiles.get(active) if isinstance(profiles, dict) else None
    if not isinstance(raw, dict):
        raw = dict(_LEGACY)
    polygon = _validate_polygon(raw.get("polygon", _LEGACY["polygon"]))
    bbox = [min(p[0] for p in polygon), min(p[1] for p in polygon), max(p[0] for p in polygon), max(p[1] for p in polygon)]
    park = raw.get("safe_park_point", _LEGACY["safe_park_point"])
    if not _finite_point(park) or not point_in_polygon(park, polygon):
        park = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    if not point_in_polygon(park, polygon):
        raise ValueError("reach profile has no valid safe parking point")
    return {"id": str(active or raw.get("id", "plotter_2001")), "label": str(raw.get("label", "Plotter reach")),
            "polygon": polygon, "bbox": bbox, "safe_park_point": [float(park[0]), float(park[1])]}


def pen_reach_profile(profile, pen_profile):
    """Intersect a global reach profile's polygon with a PEN-SPECIFIC
    rectangular clamp (`pen_profile["reach_clamp"]`, e.g. `{"y_max": 201.5}`
    for the 45-degree Preppy mount that physically fouls the printer above
    nozzle Y 201.5 -- the BIC has no such limit, so this is per-pen, not
    per-machine). Any subset of x_min/x_max/y_min/y_max is accepted.

    CONSERVATIVE BY CONSTRUCTION: a pen profile with no `reach_clamp` (or an
    empty/non-dict one) returns `profile` completely UNCHANGED (same object,
    same polygon, same bbox, same safe_park_point) -- byte-identical to
    calling `active_reach_profile()` alone, exactly as every caller that
    predates per-pen clamps still does. Only a profile that actually sets
    `reach_clamp` sees any behavior change.

    The clamp is applied via Sutherland-Hodgman half-plane clipping against
    each configured bound in turn, so it works correctly even against the
    active reach profile's own concave (stepped) polygon, not just a
    rectangle -- same concave-safe posture as `point_in_polygon` /
    `segment_in_polygon` above. Recomputes bbox and re-validates/relocates
    safe_park_point (falling back to the clamped bbox center, same rule
    `active_reach_profile` uses) so every field downstream (offset defaults,
    park point, paper-frame projection) reflects the PEN's envelope, not the
    machine's.

    Raises ValueError if the clamp eliminates the polygon entirely or leaves
    no valid park point -- this must never silently produce an empty/invalid
    envelope."""
    clamp = pen_profile.get("reach_clamp") if isinstance(pen_profile, dict) else None
    if not isinstance(clamp, dict) or not clamp:
        return profile

    def _clip(poly, axis, keep, limit):
        out = []
        n = len(poly)
        for i in range(n):
            cur, nxt = poly[i], poly[(i + 1) % n]
            cur_in, nxt_in = keep(cur[axis]), keep(nxt[axis])
            if cur_in:
                out.append(cur)
            if cur_in != nxt_in:
                d0, d1 = cur[axis], nxt[axis]
                t = (limit - d0) / (d1 - d0)
                if axis == 0:
                    out.append((limit, cur[1] + t * (nxt[1] - cur[1])))
                else:
                    out.append((cur[0] + t * (nxt[0] - cur[0]), limit))
        return out

    poly = [(float(p[0]), float(p[1])) for p in profile["polygon"]]
    if "x_min" in clamp:
        lim = float(clamp["x_min"])
        poly = _clip(poly, 0, lambda v, l=lim: v >= l - _EPS, lim)
    if "x_max" in clamp:
        lim = float(clamp["x_max"])
        poly = _clip(poly, 0, lambda v, l=lim: v <= l + _EPS, lim)
    if "y_min" in clamp:
        lim = float(clamp["y_min"])
        poly = _clip(poly, 1, lambda v, l=lim: v >= l - _EPS, lim)
    if "y_max" in clamp:
        lim = float(clamp["y_max"])
        poly = _clip(poly, 1, lambda v, l=lim: v <= l + _EPS, lim)

    if len(poly) < 3:
        raise ValueError(
            f"pen reach clamp {clamp} for profile {pen_profile.get('label', pen_profile.get('id', '?'))!r} "
            f"eliminates the entire reach polygon -- check the clamp values against the active reach profile"
        )
    poly = [[p[0], p[1]] for p in poly]
    bbox = [min(p[0] for p in poly), min(p[1] for p in poly), max(p[0] for p in poly), max(p[1] for p in poly)]
    park = profile["safe_park_point"]
    if not point_in_polygon(park, poly):
        park = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    if not point_in_polygon(park, poly):
        raise ValueError("pen reach clamp leaves no valid safe parking point")
    clamp_desc = ", ".join(f"{k}={v}" for k, v in clamp.items())
    label = f"{profile['label']} (pen-clamped: {clamp_desc})"
    return {"id": profile["id"], "label": label, "polygon": poly, "bbox": bbox,
            "safe_park_point": [float(park[0]), float(park[1])],
            "pen_clamp": dict(clamp), "unclamped_bbox": list(profile["bbox"])}


def validate_gcode_xy_motion(gcode_path, profile=None):
    """Reject every generated XY endpoint or chord outside the active polygon.

    G0/G00/G1/G01 establish a modal motion; coordinate-only continuations are
    checked too. G90/G91 are tracked. Any unsupported G-code with XY is a hard
    failure, because it cannot be demonstrated safe by this linear validator.
    """
    profile = active_reach_profile() if profile is None else profile
    poly, current, unsafe, moves = profile["polygon"], None, [], 0
    oob_points = []  # numeric (x,y) of geometric out-of-reach targets, for a plain-language summary
    absolute, modal_motion = True, None
    for line_no, line in enumerate(open(gcode_path, encoding="utf-8"), 1):
        code = line.split(";", 1)[0].strip().upper()
        if not code:
            continue
        mx = re.search(r"(?:^|\s)X(-?\d+(?:\.\d+)?)", code)
        my = re.search(r"(?:^|\s)Y(-?\d+(?:\.\d+)?)", code)
        gs = [int(g) for g in re.findall(r"(?:^|\s)G(\d+)(?:\s|$)", code)]
        if 90 in gs: absolute = True
        if 91 in gs: absolute = False
        if not mx and not my:
            continue
        explicit = next((g for g in reversed(gs) if g in (0, 1)), None)
        unsupported = [g for g in gs if g not in (0, 1, 90, 91)]
        if unsupported:
            unsafe.append((line_no, f"unsupported XY motion G{unsupported[-1]}")); continue
        if explicit is not None:
            modal_motion = explicit
        if modal_motion not in (0, 1):
            unsafe.append((line_no, "modal XY before G0/G1")); continue
        if current is None and not absolute:
            unsafe.append((line_no, "relative XY before a known machine point")); continue
        target = [current[0] if current else None, current[1] if current else None]
        if absolute:
            if mx: target[0] = float(mx.group(1))
            if my: target[1] = float(my.group(1))
        else:
            if mx: target[0] += float(mx.group(1))
            if my: target[1] += float(my.group(1))
        if None in target:
            unsafe.append((line_no, "partial XY before a known machine point")); continue
        moves += 1
        if not point_in_polygon(target, poly) or (current is not None and not segment_in_polygon(current, target, poly)):
            unsafe.append((line_no, f"X{target[0]:.3f} Y{target[1]:.3f}"))
            oob_points.append((target[0], target[1]))
        current = target
    if unsafe:
        detail = ", ".join(f"line {n}: {why}" for n, why in unsafe[:4])
        summary = _reach_overflow_summary(oob_points, profile)
        head = f"Generated G-code leaves active reach profile {profile['label']}"
        raise ValueError(f"{head}{summary}: {detail}")
    return {"xy_moves": moves, "profile_id": profile["id"], "profile_label": profile["label"]}


# (edge_name, bbox_index) -- shared between _reach_overflow_summary's overflow
# detection and its pen-clamp attribution, so the two never drift apart.
_EDGE_BBOX_INDEX = {"left": 0, "front": 1, "right": 2, "back": 3}
_EDGE_CLAMP_KEY = {"left": "x_min", "right": "x_max", "front": "y_min", "back": "y_max"}


def _reach_overflow_summary(oob_points, profile):
    """Turn raw out-of-reach coordinates into a plain-language clause naming
    which edge(s) of the pen envelope the job overruns and by how far, so the
    error is actionable instead of a wall of machine coordinates. Returns an
    empty string when the failures were not simple geometric overshoots (e.g.
    a parse/motion-mode fault, which leaves oob_points empty). ASCII only.

    `profile` is the (possibly pen_reach_profile()-clamped) profile dict, not
    a bare bbox -- when a clamped edge is the one that was actually hit, the
    message says so explicitly (mount limit, not "the back" of the machine),
    so the user isn't told they overran the printer frame when the real cause
    is a per-pen mount. Profiles with no clamp (`pen_clamp` absent) report the
    plain bed-edge message."""
    if not oob_points:
        return ""
    bbox = profile["bbox"] if isinstance(profile, dict) else profile
    x0, y0, x1, y1 = bbox
    xs = [p[0] for p in oob_points]
    ys = [p[1] for p in oob_points]
    over = []  # (edge_name, mm)
    left = x0 - min(xs)
    right = max(xs) - x1
    front = y0 - min(ys)   # reach Y-min is the plotter front
    back = max(ys) - y1
    for name, amt in (("left", left), ("right", right), ("front", front), ("back", back)):
        if amt > 0.05:
            over.append((name, amt))
    if not over:
        return ""
    over.sort(key=lambda e: -e[1])
    pen_clamp = profile.get("pen_clamp") if isinstance(profile, dict) else None
    unclamped_bbox = profile.get("unclamped_bbox") if isinstance(profile, dict) else None

    def _note(name):
        if not pen_clamp or unclamped_bbox is None:
            return ""
        key = _EDGE_CLAMP_KEY[name]
        if key not in pen_clamp:
            return ""
        i = _EDGE_BBOX_INDEX[name]
        if abs(bbox[i] - unclamped_bbox[i]) <= _EPS:
            return ""  # this edge is the machine's own limit, unaffected by the clamp
        return f" (this pen's own mount limit, not the machine frame -- {key}={pen_clamp[key]}mm)"

    edges = "; ".join(f"{amt:.1f}mm past the {name}{_note(name)}" for name, amt in over)
    return (f" -- the job reaches {edges} of the pen's reachable area "
            f"(X {x0:.1f}-{x1:.1f}mm, Y {y0:.1f}-{y1:.1f}mm). "
            f"Move that content inward (Fit to Paper re-places within reach), "
            f"or re-measure the reach profile if the pen physically goes further")
