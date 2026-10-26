"""Equivalence of the spatially indexed router walk against the ORIGINAL O(n^2) walk.

`_ref_route_low_lift` below is the original `_route_low_lift` from ccplotter/stats_warnings.py copied verbatim
(only the rename and the removal of the `len(segs) > 20000` bail, so it runs at any size). Order, flips and the
connector list must be equal to the reference on every tested input (incl. ties; random, tie-heavy, real layers
up to and above 20k strokes); the new walk is deterministic where the np.abs(complex) reference was not bit-stable
(1-ulp differences possible). Positive control: a perturbed
tie-break MUST fail the same assertions."""
import os

import numpy as np
import pytest
from pathlib import Path

import ccplotter.stats_warnings as sw
from ccplotter.stats_warnings import _mask_checked_fuse, _target_fuse, PX2MM


# ---- reference (verbatim copy) ----
def _ref_route_low_lift(lines, max_conn_px, target_strokes=None, hard_cap_px=None, connector_check=None, check_stats=None):
    """Greedy low-lift endpoint router (Stage E, after cleanup / before output).

    Orders strokes by a nearest-endpoint walk (travel order), then FUSES
    consecutive strokes into one polyline — drawing the gap as a PEN-DOWN
    connector segment — whenever the gap is short enough. Each fuse removes one
    pen lift (two Z-moves) at the cost of a little extra XY drawing. Smallest
    gaps are fused first; if `target_strokes` is given, fusing continues past
    `max_conn_px` (bounded by `hard_cap_px`) until the target stroke count is
    met. Returns (new_lines, connectors, connector_len_px, longest_px) where
    connectors is the list of 2-pt gap segments drawn (len = connector count).

    `connector_check` (opt-in, connector_mask_check; default None = the router exactly as
    before): an object from ccplotter.stage_e_mask with .ok(A, B) -> bool per candidate. A fuse is
    allowed only where it passes; a rejected fuse stays a pen lift. The stroke ORDER is unchanged
    (the nearest-endpoint walk ignores the mask), so on/off differ only in which gaps are fused.
    `check_stats` (dict, optional) is filled with the per-layer audit."""
    import numpy as np
    segs = [np.asarray(l) for l in lines if l is not None and len(l) >= 1]
    if len(segs) <= 1:
        return segs, [], 0.0, 0.0
    n = len(segs)
    starts = np.array([s[0] for s in segs])
    ends = np.array([s[-1] for s in segs])

    # ── 1. Greedy nearest-endpoint ordering (with per-stroke flip) ──
    used = np.zeros(n, bool)
    order = [(0, False)]; used[0] = True
    cur_end = segs[0][-1]
    for _ in range(n - 1):
        ds = np.abs(starts - cur_end); ds[used] = np.inf
        de = np.abs(ends - cur_end);   de[used] = np.inf
        i_s, i_e = int(np.argmin(ds)), int(np.argmin(de))
        i, flip = (i_s, False) if ds[i_s] <= de[i_e] else (i_e, True)
        used[i] = True
        order.append((i, flip))
        cur_end = segs[i][0] if flip else segs[i][-1]
    oriented = [segs[i][::-1] if f else segs[i] for (i, f) in order]

    # ── 2. gaps between consecutive strokes; choose which to fuse ──
    gaps = np.array([abs(oriented[k + 1][0] - oriented[k][-1]) for k in range(n - 1)])
    fuse = gaps <= max_conn_px + 1e-6      # epsilon: fuse gaps sitting exactly on the limit
    if connector_check is not None:
        fuse = _mask_checked_fuse(oriented, gaps, fuse, max_conn_px, target_strokes, hard_cap_px,
                                  connector_check, check_stats)
    elif target_strokes is not None and target_strokes >= 1:
        cap = hard_cap_px if hard_cap_px else max_conn_px * 4.0
        strokes_now = int((~fuse).sum()) + 1
        for gi in np.argsort(gaps):
            if strokes_now <= target_strokes:
                break
            if fuse[gi]:
                continue
            if gaps[gi] > cap:             # ascending → nothing smaller left to fuse
                break
            fuse[gi] = True; strokes_now -= 1

    # ── 3. build fused polylines (concatenation = implicit pen-down connector) ──
    new_lines, connectors, conn_len, longest = [], [], 0.0, 0.0
    chain = [oriented[0]]
    for k in range(n - 1):
        if fuse[k]:
            connectors.append(np.array([oriented[k][-1], oriented[k + 1][0]]))
            chain.append(oriented[k + 1])
            g = float(gaps[k]); conn_len += g; longest = max(longest, g)
        else:
            new_lines.append(np.concatenate(chain)); chain = [oriented[k + 1]]
    new_lines.append(np.concatenate(chain))
    return new_lines, connectors, conn_len, longest


# ---- harness ----
def _mk(rng, n, spread=500.0, snap=None, closed_frac=0.0):
    out = []
    for _ in range(n):
        p = rng.uniform(0, spread, 2)
        q = p + rng.normal(0, spread * 0.01, 2)
        if snap:
            p = np.round(p / snap) * snap; q = np.round(q / snap) * snap
        mid = (p + q) / 2 + rng.normal(0, 0.5, 2)
        pts = np.array([p[0] + 1j * p[1], mid[0] + 1j * mid[1], q[0] + 1j * q[1]])
        if rng.random() < closed_frac:
            pts[-1] = pts[0]
        out.append(pts)
    return out


def _same(new, ref):
    (nl, nc, ncl, nlg), (rl, rc, rcl, rlg) = new, ref
    assert len(nl) == len(rl) and len(nc) == len(rc)
    for a, b in zip(nl, rl):
        assert a.shape == b.shape and np.array_equal(a, b)      # order AND flips (reversed arrays differ)
    for a, b in zip(nc, rc):
        assert np.array_equal(a, b)
    assert ncl == rcl and nlg == rlg


def _check(lines, conn=3.0, target=None):
    cp = 4 * conn if target else None
    _same(sw._route_low_lift(lines, conn, target, cp), _ref_route_low_lift(lines, conn, target, cp))


@pytest.mark.parametrize("n", [2, 3, 10, 100, 1000, 3000])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_sets(n, seed):
    rng = np.random.default_rng(seed)
    _check(_mk(rng, n), conn=3.0)
    _check(_mk(rng, n, closed_frac=0.5), conn=3.0, target=max(1, n // 4))


@pytest.mark.parametrize("n", [50, 500, 2500])
def test_many_exact_ties_and_duplicates(n):
    rng = np.random.default_rng(7)
    lines = _mk(rng, n, spread=60.0, snap=5.0)          # integer-grid endpoints: huge numbers of exact ties
    lines += [lines[i].copy() for i in range(0, n, 3)]   # exact duplicate strokes (identical endpoints)
    lines += [lines[i][::-1].copy() for i in range(0, n, 5)]  # reversed duplicates: start of one == end of another
    rng.shuffle(lines)
    _check(lines, conn=6.0)
    _check(lines, conn=6.0, target=len(lines) // 3)
    # lattice: every nearest neighbour is a 4-way tie
    g = [np.array([x + 1j * y, x + 1j * y + 0.0]) for x in range(0, 40) for y in range(0, 40)]
    rng.shuffle(g)
    _check(g, conn=1.5)
    # all-identical points
    _check([np.array([5 + 5j, 5 + 5j]) for _ in range(300)], conn=1.0)


def _real_layer():
    import vpype as vp
    cands = [Path(__file__).resolve().parent / "fixtures" / "router_real_layer.svg"]   # shipped real layer (3173 strokes)
    extra = os.environ.get("CCPLOTTER_TEST_SVG_DIR")    # optional: a directory of real *.clean.svg layers
    if extra:
        cands += sorted(Path(extra).glob("**/*.clean.svg"))
    cands += [p for p in sorted(Path(__file__).resolve().parent.glob("**/*.svg"))]
    for p in cands:
        try:
            doc = vp.read_multilayer_svg(str(p), quantization=0.1)
        except Exception:
            continue
        for lc in doc.layers.values():
            if 1500 <= len(lc) <= 19000:
                return str(p), [np.asarray(l) for l in lc]
    return None, None


def test_real_layer_under_20k():
    p, lines = _real_layer()
    if lines is None:
        pytest.skip("no real layer between 1.5k and 19k strokes available")
    cm = 0.5 / PX2MM
    _check(lines, conn=cm, target=None)


def test_guard_logged_and_flagged(caplog):
    lines = _mk(np.random.default_rng(0), 25)
    info = {}
    with caplog.at_level("WARNING"):
        nl, cn, cl, lg = sw._route_low_lift(lines, 3.0, skip_info=info, max_walk_segs=10)
    assert info.get("route_skipped_guard") is True and cn == [] and len(nl) == 25
    assert any("SKIPPED" in r.message and "25" in r.message for r in caplog.records)


def test_plain_cap_stays_20000_and_twopen_cap_routes():
    """The plain default is 20000 (output unchanged); a 20001-stroke layer is skipped and flagged by default,
    routed when the caller passes the two-pen cap."""
    assert sw.ROUTE_WALK_MAX_SEGS_PLAIN == 20000 and sw.ROUTE_WALK_MAX_SEGS_TWOPEN == 100000
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1500, (20001, 2)); q = p + rng.normal(0, 2, (20001, 2))
    lines = [np.array([a[0] + 1j * a[1], b[0] + 1j * b[1]]) for a, b in zip(p, q)]
    info = {}
    nl, cn, _, _ = sw._route_low_lift(lines, 3.0, skip_info=info)               # default args = plain path
    assert info.get("route_skipped_guard") is True and info["strokes"] == 20001
    assert cn == [] and len(nl) == 20001 and all(np.array_equal(a, b) for a, b in zip(nl, lines))   # untouched
    info2 = {}
    nl2, cn2, _, _ = sw._route_low_lift(lines, 3.0, skip_info=info2, max_walk_segs=sw.ROUTE_WALK_MAX_SEGS_TWOPEN)
    assert not info2 and len(cn2) > 0 and len(nl2) < 20001


def test_real_layer_over_20k_walk_equals_reference():
    """Real tram blue layer, 31,084 strokes (endpoints committed as tests/fixtures/tram_blue_endpoints_31084.npz).
    Order, flips and the resulting connector gaps must equal the brute-force reference walk (~4 s)."""
    f = np.load(Path(__file__).resolve().parent / "fixtures" / "tram_blue_endpoints_31084.npz")
    starts, ends = f["starts"], f["ends"]
    assert len(starts) == 31084 > 20000
    new = sw._nn_walk_order(starts, ends, ends[0])
    ref = sw._nn_walk_order_bruteforce(starts, ends, ends[0])
    assert new == ref                                            # order AND flip, element for element
    def gaps(order):
        cur = [ends[i] if not fl else starts[i] for i, fl in order]
        nxt = [starts[i] if not fl else ends[i] for i, fl in order]
        return np.array([abs(nxt[k + 1] - cur[k]) for k in range(len(order) - 1)])
    gn, gr = gaps(new), gaps(ref)
    cm = 0.5 / PX2MM
    assert np.array_equal(gn <= cm + 1e-6, gr <= cm + 1e-6)      # identical connector decisions
    assert (gr <= cm + 1e-6).sum() > 100                         # non-vacuous: connectors really exist at 0.5 mm


# ---- positive control: a perturbed tie-break must be caught ----
def _perturbed_walk(starts, ends, first_end):
    """Same as the brute-force walk but ties resolved to the HIGHEST index: must disagree with the reference."""
    n = len(starts); used = np.zeros(n, bool); order = [(0, False)]; used[0] = True; cur = first_end
    for _ in range(n - 1):
        ds = np.abs(starts - cur); ds[used] = np.inf
        de = np.abs(ends - cur); de[used] = np.inf
        i_s = n - 1 - int(np.argmin(ds[::-1])); i_e = n - 1 - int(np.argmin(de[::-1]))
        i, f = (i_s, False) if ds[i_s] <= de[i_e] else (i_e, True)
        used[i] = True; order.append((i, f)); cur = starts[i] if f else ends[i]
    return order


def test_positive_control_perturbed_tiebreak_fails(monkeypatch):
    monkeypatch.setattr(sw, "_nn_walk_order", _perturbed_walk)
    rng = np.random.default_rng(7)
    lines = _mk(rng, 500, spread=60.0, snap=5.0)
    with pytest.raises(AssertionError):
        _check(lines, conn=6.0)
    # and the unperturbed walk passes on the very same input
    monkeypatch.undo()
    _check(lines, conn=6.0)
