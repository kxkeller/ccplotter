from pathlib import Path

from ccplotter.config import NATIVE_CAP


def _inkscape_exe():
    """Locate the Inkscape executable. GUI installers often skip PATH, so also
    probe the standard Program Files locations."""
    import shutil
    exe = shutil.which("inkscape")
    if exe:
        return exe
    for p in (r"C:\Program Files\Inkscape\bin\inkscape.exe",
              r"C:\Program Files\Inkscape\inkscape.exe",
              r"C:\Program Files (x86)\Inkscape\bin\inkscape.exe",
              r"C:\Program Files (x86)\Inkscape\inkscape.exe"):
        if Path(p).exists():
            return p
    return None

def trace_to_svg(img_path: Path, out_path: Path, filter_speckle: int = 10) -> Path:
    """Vectorize a bitmap with headless Inkscape's trace-bitmap engine. Crop and
    contrast are handled upstream (_load_gray); this just traces. The exported
    SVG keeps the embedded raster, but vpype imports only the vector geometry."""
    import subprocess
    exe = _inkscape_exe()
    if not exe:
        raise RuntimeError("Inkscape not found. Install it and add C:\\Program Files\\Inkscape\\bin to your PATH.")
    # Forward slashes + absolute paths are the most portable form inside --actions.
    in_abs = str(Path(img_path).resolve())
    out_abs = str(out_path.resolve()).replace("\\", "/")
    # Inkscape 1.4 'object-trace' arg format (7 fields):
    #   scans,smooth,stack,remove_background,speckles,smooth_corners,optimize
    # 2 stacked brightness scans + remove_background isolates the dark linework
    # (a single scan just greyscale-fills the whole page). speckles culls noise.
    speckles = max(0, int(filter_speckle))
    trace = f"object-trace:2,true,true,true,{speckles},1.0,0.2"
    actions = f"select-all;{trace};export-filename:{out_abs};export-do"
    cmd = [exe, in_abs, "--actions=" + actions]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
    except subprocess.CalledProcessError as e:
        print(f"Inkscape trace failed: {e.stderr}")
        raise RuntimeError("Inkscape tracing failed. Ensure Inkscape is installed and on your PATH.")
    except FileNotFoundError:
        raise RuntimeError("Inkscape executable not found. Install Inkscape and add it to your system PATH.")
    if not out_path.exists():
        raise RuntimeError("Inkscape produced no output SVG — check the trace-bitmap action for your version.")
    return out_path

def _load_gray(raw_path: Path, max_side: int = 1400, clahe: bool = False, channel: str = "auto", crop: list = None, isolate_color: str = None, clahe_clip: float = 2.5):
    import cv2, numpy as np, time; from PIL import Image
    t0 = time.time()

    # MEMORY GUARD: If the file is massive, use PIL to open and thumbnail it
    # BEFORE converting to a full NumPy array. This prevents MemoryErrors.
    file_size = raw_path.stat().st_size
    if file_size > 20 * 1024 * 1024: # 20MB
        print(f"    [warn] Large file ({file_size/1e6:.1f}MB), using PIL lazy-load")
        with Image.open(raw_path) as pil_img:
            # Thumbnail to 2x the max_side for safety/quality before CV2 takes over
            pil_img.thumbnail((max_side*2, max_side*2))
            bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if bgr is None: bgr = cv2.cvtColor(np.array(Image.open(raw_path).convert("RGB")), cv2.COLOR_RGB2BGR)

    print(f"    [perf] _load_gray: load {time.time()-t0:.2f}s, src {bgr.shape}")

    h, w = bgr.shape[:2]
    s = int(max_side) / max(h, w)
    if s < 1.0:
        t0 = time.time()
        bgr = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        print(f"    [perf] _load_gray: resize down {time.time()-t0:.2f}s, now {bgr.shape}")
    elif s > 1.0:
        t0 = time.time()
        # Bilevel sources (AI-cleaned line art = pure black/white) must NOT get the
        # unsharp mask: it rings on hard ink edges, the threshold then cuts through
        # the ringing, and the skeletonizer wobbles along the ragged border.
        gtest = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bilevel = float(np.mean((gtest < 40) | (gtest > 215))) > 0.97
        bgr = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_CUBIC)
        if not bilevel:
            blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=2.0)
            bgr = cv2.addWeighted(bgr, 1.5, blur, -0.5, 0)
        print(f"    [perf] _load_gray: upscale {time.time()-t0:.2f}s, now {bgr.shape}, bilevel={bilevel}")

    if crop and len(crop) == 4:
        h, w = bgr.shape[:2]
        ix0, iy0, ix1, iy1 = max(0, int(crop[0]*w)), max(0, int(crop[1]*h)), min(w, int(crop[2]*w)), min(h, int(crop[3]*h))
        if ix1 > ix0 and iy1 > iy0: bgr = bgr[iy0:iy1, ix0:ix1]

    if isolate_color == "blue":
        t0 = time.time()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (90, 50, 20), (135, 255, 255))
        bgr[mask == 0] = (255, 255, 255)
        print(f"    [perf] _load_gray: blue isolate {time.time()-t0:.2f}s")

    b, g, r = cv2.split(bgr)
    cand = {"luma": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), "red": r, "green": g, "blue": b}
    img = cv2.normalize(cand.get(channel, max(cand.values(), key=lambda c: float(cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX).std()))), None, 0, 255, cv2.NORM_MINMAX)
    if clahe: img = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8)).apply(img)
    return img


def _prepare_shorter_side_gray(raw_path: Path, side: int) -> Path:
    """Photo-512 mode's single extra preprocessing step (config.PHOTO512_RECIPE):
    pre-downsample the SOURCE to a fixed shorter-side pixel count, in plain
    greyscale, BEFORE _load_gray sees it. The detail reduction is deliberate:
    a source kept at 768 px or more carries less long-stroke structure into
    the trace than a 512 px one.

    Resize semantics: shorter-side scale (`side / min(h, w)`), single-channel
    IMREAD_GRAYSCALE decode, INTER_AREA, and no direction guard (it always
    resizes, even if the source is already smaller than `side`). Writes a
    temp PNG into WORK_DIR and returns its path. The caller (render.py) feeds
    that path to the unchanged `_load_gray`, whose own resize
    (max_side / max(h, w); this upscales the shrunk source back to the
    working trace resolution with its CUBIC + unsharp path) then runs as for
    any other source. This function is only called when
    opts["source_side_cap"] is set, so every other mode is unaffected."""
    import cv2
    import numpy as np
    from uuid import uuid4
    from ccplotter.config import WORK_DIR
    file_bytes = np.fromfile(str(raw_path), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Photo-512: could not decode source image for shorter-side downsample: {raw_path}")
    h, w = img.shape[:2]
    s = float(side) / max(1, min(h, w))
    out = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("Photo-512: failed to encode shorter-side-downsampled source.")
    tmp = WORK_DIR / f"photo512_src_{uuid4().hex[:8]}.png"
    buf.tofile(str(tmp))
    return tmp


def _stroke_px_for(opts: dict, working_w: float) -> float:
    """The pen's real physical fill width, in the SAME pixel units as the
    polylines being written. Every SVG meant for human eyes must use this
    stroke width: a fixed hairline (e.g. 0.6px or vpype's 1px default) is
    decoupled from fill_mm, so rings look open and text looks worse than it
    plots. No caller may omit it."""
    fill_mm = float((opts or {}).get("fill_mm", 0.15))
    pw_mm = float((opts or {}).get("paper_w_mm", 180.0))
    mg_mm = float((opts or {}).get("margin_mm", 10.0))
    mm_per_px = max(1e-6, (pw_mm - 2 * mg_mm)) / max(1.0, float(working_w))
    return max(0.1, fill_mm / mm_per_px)


# Junction continuation threshold: when stitching graph edges through a
# junction, only continue onto an edge whose direction keeps the stroke roughly
# straight (cos of the turn angle ≥ this). Higher = stricter / more pen lifts but
# no polygon corners; ~0.2 ≈ allow up to ~78° bend. Keeps real line continuations
# joined while letting perpendicular branches break off cleanly.
_JOIN_COS = 0.2

def _skeleton_to_polylines(skel, prune_len=12, protect_mask=None, protect_prune=2):
    """Graph-based skeleton → polylines.

    A skeleton is a graph, not a single path; a greedy global walk turns the
    pixels around junctions into spurious polygon/zig-zag artifacts. Instead we:
      1. prune short terminal spurs (pixel level),
      2. classify pixels by neighbour count — endpoint(1) / regular(2) /
         junction(3+); endpoints & junctions are "nodes",
      3. trace each graph EDGE as the run of regular pixels between two nodes,
      4. stitch edges back into long strokes through junctions ONLY when they
         stay nearly collinear (_JOIN_COS) — straight lines pass through, branches
         split off — which avoids the polygon look while keeping pen lifts low,
      5. preserve closed loops (components with no nodes),
      6. drop strokes shorter than prune_len.

    MICRO-DETAIL PROTECTION: `protect_mask` (bool HxW, True = preserve) marks
    text/label/rivet regions. Spurs and strokes lying mostly inside it use the
    much smaller `protect_prune` threshold instead of `prune_len`, so tiny glyph
    strokes survive while the rest of the drawing keeps clean spur-pruning. When
    `protect_mask is None` the clean path is used unchanged (output is
    byte-identical to a call without the argument)."""
    import numpy as np
    from collections import defaultdict
    pts = set(map(tuple, np.argwhere(skel)))   # (y, x)
    if not pts:
        return []
    PH, PW = (protect_mask.shape if protect_mask is not None else (0, 0))
    def _is_protected(path):
        """True when ≥ half of a path's pixels fall in the protect region."""
        if protect_mask is None:
            return False
        hit = 0
        for (yy, xx) in path:
            if 0 <= yy < PH and 0 <= xx < PW and protect_mask[yy, xx]:
                hit += 1
        return hit * 2 >= len(path)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx) for dy, dx in offs if (y + dy, x + dx) in pts]

    # ── 1. Prune short terminal spurs (two passes) so junctions stay clean ──
    for _ in range(2):
        for p in [q for q in pts if len(nbrs(q)) == 1]:
            path, curr = [p], p
            for _ in range(prune_len // 2):
                ns = [n for n in nbrs(curr) if n not in path]
                if len(ns) != 1: break
                curr = ns[0]; path.append(curr)
            thr = (protect_prune // 2) if _is_protected(path) else (prune_len // 2)
            if len(path) < thr:
                for pp in path: pts.discard(pp)
    if not pts:
        return []

    # ── 2. Classify nodes (degree != 2) ──
    deg = {p: len(nbrs(p)) for p in pts}
    nodes = {p for p in pts if deg[p] != 2}

    # ── 3. Trace edges. `used` tracks traversed adjacency steps so each edge and
    #       each loop is walked exactly once. ──
    used = set()                       # frozenset({a, b}) adjacency steps
    edges = []                         # each: list of (y, x) pixels, node→node
    def walk(start, first):
        path = [start, first]
        used.add(frozenset((start, first)))
        prev, curr = start, first
        while curr not in nodes:
            nxt = next((n for n in nbrs(curr) if n != prev), None)
            if nxt is None: break
            used.add(frozenset((curr, nxt)))
            path.append(nxt); prev, curr = curr, nxt
        return path

    for s in nodes:
        for n in nbrs(s):
            if frozenset((s, n)) not in used:
                edges.append(walk(s, n))

    # Loops & leftovers: any adjacency step not yet used belongs to a pure cycle
    # (no node). Sweep them into closed polylines.
    for p in list(pts):
        for n in nbrs(p):
            if frozenset((p, n)) in used: continue
            path = [p, n]; used.add(frozenset((p, n))); prev, curr = p, n
            while True:
                nxt = next((x for x in nbrs(curr) if x != prev and frozenset((curr, x)) not in used), None)
                if nxt is None: break
                used.add(frozenset((curr, nxt))); path.append(nxt); prev, curr = curr, nxt
            edges.append(path)

    if not edges:
        return []

    # ── 4. Stitch edges into strokes through junctions when nearly collinear ──
    node_inc = defaultdict(list)       # node → [(edge_idx, end) ...]; end 0=head 1=tail
    for i, e in enumerate(edges):
        node_inc[e[0]].append((i, 0))
        if e[-1] != e[0]: node_inc[e[-1]].append((i, 1))

    def out_dir(e, end):
        """Unit direction leaving the node at `end` of edge e (a few px in)."""
        k = min(4, len(e) - 1)
        v = (np.array(e[k]) - np.array(e[0])) if end == 0 else (np.array(e[-1 - k]) - np.array(e[-1]))
        nv = np.linalg.norm(v)
        return v / nv if nv > 0 else v

    edge_used = [False] * len(edges)
    def extend(poly, exclude):
        """Grow poly forward from its tail through collinear junction edges."""
        cur = exclude
        while True:
            node = poly[-1]
            inc = np.array(poly[-1]) - np.array(poly[-2])
            ni = np.linalg.norm(inc)
            if ni == 0: break
            inc = inc / ni
            best, best_dot = None, _JOIN_COS
            for (j, end) in node_inc.get(node, ()):
                if edge_used[j] or j == cur: continue
                d = float(np.dot(inc, out_dir(edges[j], end)))
                if d >= best_dot: best_dot, best = d, (j, end)
            if best is None: break
            j, end = best
            edge_used[j] = True
            seg = edges[j] if end == 0 else edges[j][::-1]
            poly.extend(seg[1:])
            cur = j

    # Start from spur/endpoint edges first so open strokes run end-to-end.
    def is_terminal(i):
        return len(node_inc.get(edges[i][0], ())) <= 1 or len(node_inc.get(edges[i][-1], ())) <= 1
    order = sorted(range(len(edges)), key=lambda i: 0 if is_terminal(i) else 1)

    polys = []
    for i in order:
        if edge_used[i]: continue
        edge_used[i] = True
        poly = list(edges[i])
        if len(poly) >= 2:
            extend(poly, i)            # grow tail
            poly.reverse()
            extend(poly, i)            # grow original head
        thr = protect_prune if _is_protected(poly) else prune_len
        if len(poly) >= thr:
            polys.append([(x, y) for (y, x) in poly])
    return polys

def _binarize_lineart(img, sensitivity: str = "normal"):
    """Ink-sensitive binarization for ALREADY-CLEAN line art (not photos).

    ChatGPT/OpenAI line art is anti-aliased: faint mountain contours and distant
    detail sit at gray 200-245, which a single hard threshold drops while keeping
    the paper. This uses HYSTERESIS plus a line-shape test so faint ink survives
    without turning the background into speckle:
      - strong ink (< strong_t) is always kept,
      - faint ink (< weak_t) is kept when its connected blob touches strong ink
        (a faint line attached to a confident one), OR when the blob is
        line-like (elongated / thin / sparse-fill) even with no strong anchor
        (uniformly-faint contours like ridgelines),
      - isolated faint dots/blobs are dropped — but counted, so they stay
        measurable and prunable downstream.
    Returns a boolean ink mask (True = draw)."""
    import cv2, numpy as np
    LEV = {                                    # (strong_t, weak_t)
        "strict":         (150, 150),          # confident dark ink only (single cut)
        "normal":         (190, 205),          # mild hysteresis ~ manual threshold 200
        "sensitive":      (200, 236),          # faint connected linework
        "very_sensitive": (210, 250),          # maximum faint detail
    }
    strong_t, weak_t = LEV.get(sensitivity, LEV["normal"])
    weak = img < weak_t
    if strong_t >= weak_t:
        return weak
    strong = img < strong_t
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), 8)
    if n <= 1:
        return weak
    has_strong = np.zeros(n, bool)
    sid = np.unique(lbl[strong])
    has_strong[sid[sid >= 0]] = True
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    ws = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
    hs = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    elong = np.maximum(ws, hs) / np.maximum(1.0, np.minimum(ws, hs))
    fill = areas / np.maximum(1.0, ws * hs)
    sc = max(img.shape) / 1000.0
    min_keep = max(6.0, 8.0 * sc * sc)         # below this an unanchored faint blob = a dot
    linelike = (areas >= min_keep) & ((elong >= 2.5) | (fill <= 0.28))
    keep = has_strong | linelike
    keep[0] = False
    return keep[lbl]

def _micro_detail_mask(mask, level: str = "balanced", noise_filter: str = "off",
                       debug_prefix: str = None, r_factor: float = 2.0, max_support: int = 14):
    """Boolean region map (True = preserve) marking text/label/rivet micro-detail.

    DETAIL-FIRST: the BASE behaviour PROTECTS EVERY small, compact connected
    component — single glyphs, bolts, rivets, short body-texture marks — so their
    fragile skeleton spurs survive pruning while big structural strokes and long
    hatch lines stay on the normal clean path. This is what keeps the BALDWIN
    placard, 472 plate, D&RGW hub text, rivets and fine locomotive texture rich.

      level='balanced'   : protect small marks up to ~30px·scale (default).
      level='aggressive' : larger size ceiling (~44px·scale) → keeps even more.

    NOISE FILTER (SEPARATE, OPT-IN, default OFF — never silently removes detail):
      'off'    : keep all protected marks (maximum detail, more speckle/Z-moves).
      'light'  : drop ONLY lone specks — a small mark with ZERO small siblings
                 within radius R (true isolated dust). Conservative.
      'strong' : also drop AREA-FILLING fields (support > max_support: dense
                 ballast/shading texture). Most aggressive cleanup.
    Vectorised; returns (bool mask HxW, n_protected_components)."""
    import cv2, numpy as np
    H, W = mask.shape
    sc = max(H, W) / 1000.0
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros((H, W), bool), 0
    small_dim = (44.0 if str(level).lower() == "aggressive" else 30.0) * sc
    min_area = max(3.0, 2.0 * sc * sc)          # below this a "component" is AA dust
    w = stats[:, cv2.CC_STAT_WIDTH].astype(float)
    h = stats[:, cv2.CC_STAT_HEIGHT].astype(float)
    a = stats[:, cv2.CC_STAT_AREA].astype(float)
    small = (np.maximum(w, h) <= small_dim) & (a >= min_area)
    small[0] = False                            # background label
    keep = small.copy()                         # BASE: protect every small mark
    idx = np.where(small)[0]
    nf = str(noise_filter).lower()
    if nf in ("light", "strong") and idx.size:
        # Count small-sibling support within radius R; gate from there.
        R = max(10.0, small_dim * float(r_factor))
        cents = cent[idx]                       # (k,2) x,y centroids
        try:
            from scipy.spatial import cKDTree
            support = cKDTree(cents).query_ball_point(cents, R, return_length=True) - 1
        except Exception:
            cell = max(1.0, R); grid = {}
            for j, (cx, cy) in enumerate(cents):
                grid.setdefault((int(cx // cell), int(cy // cell)), []).append((cx, cy))
            support = np.zeros(idx.size, int); R2 = R * R
            for j, (cx, cy) in enumerate(cents):
                kx, ky = int(cx // cell), int(cy // cell); cnt = 0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for (ox, oy) in grid.get((kx + dx, ky + dy), ()):
                            if (ox - cx) ** 2 + (oy - cy) ** 2 <= R2: cnt += 1
                support[j] = cnt - 1
        ok = support >= 1                       # light: keep marks that have ≥1 near sibling
        if nf == "strong" and int(max_support) > 0:
            ok &= support <= int(max_support)   # strong: also drop dense texture fields
        drop = idx[~ok]
        keep[drop] = False
    keep[0] = False
    protect = keep[lbl]
    if debug_prefix is not None:
        # White = protected micro-detail; gray = small marks dropped by the filter.
        vis = np.zeros((H, W), np.uint8)
        rejected = small.copy(); rejected[keep] = False
        vis[rejected[lbl]] = 90
        vis[protect] = 255
        cv2.imwrite(f"{debug_prefix}_protect.png", vis)
    return protect, int(keep.sum())

def _outline_polys(bin_mask, simplify_px: float = 0.75):
    """Trace the OUTLINE (contours) of a binary mask as closed polylines.

    Used for small text/glyph regions: skeletonization reduces a letter to its
    medial axis, which is NOT the letter's shape (bowls/junctions in A/P/R/E
    collapse). The mask outline preserves the actual glyph. RETR_CCOMP keeps
    inner contours too, so letter counters (o, e, a, P, R ...) survive. Returns
    a list of closed polylines [(x,y), ...]; coords are pixel (col,row) = the
    same (x,y) convention the skeleton polylines and _write_paths_svg use."""
    import cv2, numpy as np
    m = (np.asarray(bin_mask).astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    polys = []
    for c in cnts:
        if simplify_px > 0 and len(c) > 4:
            c = cv2.approxPolyDP(c, simplify_px, True)
        pts = c.reshape(-1, 2)
        if len(pts) < 2:
            continue
        pl = [(float(x), float(y)) for x, y in pts]
        pl.append(pl[0])                     # close the loop
        polys.append(pl)
    return polys


# Small-detail handling is per-component fidelity testing (see
# _component_fidelity_detail below), not ROI-based text detection. "Is this
# region text" cannot be decided from ROI size, area or group count: dense fur
# and a multi-line lettered plate overlap on every such metric. The test asks
# a physical question instead: does this candidate rendering reproduce the ink?

def _superres_contour_polys(gray, roi_box, med_glyph_px, target_px=46, max_up=10):
    """Local super-resolution + contour trace of one ROI. Upscales the crop so
    its glyphs reach ~target_px (where contours are clean), thresholds locally,
    traces outlines (RETR_CCOMP → counters preserved), then scales the vectors
    back to full-image (gray) coordinates. Skeleton mangles small glyphs; this
    keeps the source's actual lettering."""
    import cv2, numpy as np
    x0, y0, x1, y1 = roi_box
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    up = int(round(max(2, min(max_up, target_px / max(2.0, med_glyph_px)))))
    big = cv2.resize(crop, (crop.shape[1] * up, crop.shape[0] * up), interpolation=cv2.INTER_CUBIC)
    _, m = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = m > 0
    from skimage.morphology import remove_small_objects
    m = remove_small_objects(m, min_size=max(6, int(0.5 * up * up)))
    polys = _outline_polys(m, simplify_px=max(1.0, up * 0.35))
    return [[(x / up + x0, y / up + y0) for (x, y) in p] for p in polys]


def _component_fidelity_detail(mask, gray, fill_px, small_mult=100.0, cache=None,
                                margin=0.12, hole_thresh=0.5, debug_scores=None,
                                isolation="strict"):
    """PER-COMPONENT FIDELITY TEST for small ink components. It asks a purely
    physical question per SMALL ink component: once rendered at the pen's real
    width, which candidate -- skeleton centerline, or local super-res contour
    outline -- more faithfully reproduces this component's OWN ink shape? There
    are no boxes and no text/texture classification.

    small_mult sets the area cap as a multiple of the pen footprint
    (pi*(fill_px/2)^2). 100 covers small lettering (about 4x-64x the footprint
    at fill_px=2.81) with headroom.

    Two rules keep the test from turning dashes into rings:

    (1) ISOLATION CONSTRAINT. Any small component with another component
    within ~1 pen width (`fill_px`) is skipped entirely. It is treated as part
    of a network (rivet lines, rail-tie dashes the mask happened to break into
    several small blobs) and left for the normal whole-network skeleton pass.
    Only isolated small components (lone glyphs, lone specks) are tested.
    This is geometry (adjacency), not semantics.

    (2) SCORER: decisive-win margin. Local contour-outline tracing hugs ANY
    blob-shaped mask almost perfectly by construction (it traces the boundary),
    so a plain "iou_b > iou_a" comparison favours the contour on anything that
    is not a long uniform stroke, including dashes, which are blobs at this
    scale. The contour's IoU must therefore beat the skeleton's by >= `margin`
    relative (default 0.12). A hole-count topology check is computed
    (`cand_hole_area`) but does not veto; see the note at its computation.
    Both are geometric measurements on the rendered candidate.

    `debug_scores`, if given a list, gets one (iou_a, iou_b, switched, h, area,
    cand_hole_area / pen_footprint) tuple appended per tested (non-skipped)
    component -- for auditing the margin distribution, not used at runtime.

    Returns (exclude_mask, extra_polys, n_checked, n_switched, min_switched_h).
    n_checked counts only components that passed the isolation filter (were
    actually simulated).

    `isolation` (opt-in): 'strict' (default) applies the isolation pre-filter
    above. 'trial' removes the pre-filter: every component that passes the
    area gate enters the fidelity trial directly, and the trial's ink-space
    IoU decisive-margin test is the sole decider of switch/no-switch.
    Proximity to a neighbour does not geometrically separate 'letters in a
    word' from 'beads in a broken line' (both are clusters of small adjacent
    components), so the strict filter also keeps ordinary kerned text out of
    the trial. 'trial' adds no text/letter/dash classification; it only drops
    the proximity-based skip."""
    import cv2, numpy as np, hashlib
    from skimage.morphology import skeletonize as _sk
    from scipy.ndimage import binary_fill_holes, binary_dilation
    H, W = mask.shape
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    pen_footprint = np.pi * (fill_px / 2.0) ** 2
    small_area_cap = small_mult * pen_footprint
    exclude_mask = np.zeros((H, W), dtype=bool)
    extra_polys = []
    n_checked = n_switched = 0
    min_switched_h = None
    pw = max(1, int(round(fill_px)))
    se_r = max(1, pw)   # ~1 pen width adjacency radius for the isolation test
    se = np.ones((2 * se_r + 1, 2 * se_r + 1), dtype=bool)

    def render_iou(polys, shape, pw):
        if not polys:
            return 0.0, None
        canvas = np.zeros(shape, np.uint8)
        for p in polys:
            pts = np.array([(int(round(px)), int(round(py))) for px, py in p], np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, 1, thickness=max(1, pw))
        canvas_bool = canvas > 0
        inter = int(np.logical_and(canvas_bool, comp_mask).sum())
        union = int(np.logical_or(canvas_bool, comp_mask).sum())
        return (inter / union if union else 0.0), canvas_bool

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > small_area_cap or area < 4:
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        pad = max(2, se_r)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        comp_mask = (lbl[y0:y1, x0:x1] == i)
        if comp_mask.sum() < 4:
            continue

        # Isolation constraint: skip components with a neighbor within
        # ~1 pen width -- they're part of a connected network, stay whole.
        # The opt-in 'trial' mode skips this pre-filter entirely (see the
        # isolation= docstring above); the default 'strict' applies it.
        if isolation != "trial":
            neighbor_crop = lbl[y0:y1, x0:x1]
            dilated = binary_dilation(comp_mask, structure=se)
            if np.any((neighbor_crop != 0) & (neighbor_crop != i) & dilated):
                continue

        cache_key = None
        if cache is not None:
            cache_key = hashlib.md5(comp_mask.tobytes()).hexdigest() + f":{x1-x0}x{y1-y0}"
            hit = cache.get(cache_key)
            if hit is not None:
                switched, local_polys, cached_h = hit
                n_checked += 1
                if switched:
                    n_switched += 1
                    min_switched_h = cached_h if min_switched_h is None else min(min_switched_h, cached_h)
                    exclude_mask[y0:y1, x0:x1] |= comp_mask
                    extra_polys += [[(px + x0, py + y0) for px, py in p] for p in local_polys]
                continue

        n_checked += 1
        # Candidate A: skeleton of just this component (local coords)
        skel = _sk(comp_mask)
        polys_a = _skeleton_to_polylines(skel, prune_len=1)
        # Candidate B: local super-res contour, LOCAL roi_box (crop coords), so
        # its returned polys are already crop-local +0,0 offset -- pass gray
        # crop directly with a zero-origin box.
        crop_gray = gray[y0:y1, x0:x1]
        med_h = max(3.0, float(h))
        polys_b = _superres_contour_polys(crop_gray, (0, 0, x1 - x0, y1 - y0), med_h)

        iou_a, _ = render_iou(polys_a, comp_mask.shape, pw)
        iou_b, canvas_b = render_iou(polys_b, comp_mask.shape, pw)

        # Decisive relative margin, not a nose-ahead win.
        if iou_a <= 0:
            decisive = iou_b > 0
        else:
            decisive = (iou_b - iou_a) / iou_a >= margin

        # Ink-space topology measurement, NOT gating: does canvas_b (the
        # pen-width render, the scorer's own substrate) retain a hole where
        # the true mask is solid? A path-space check would wrongly veto real
        # glyph wins, since outline tracing is a hollow ring in path space
        # regardless of ink. In ink space, real glyph strokes and real rivet
        # dots still give overlapping hole-ratio distributions, so no
        # threshold separates them and the margin test above decides alone.
        # cand_hole_area is still computed and reported through debug_scores
        # for auditing, but it vetoes nothing.
        cand_hole_area = 0
        if canvas_b is not None:
            cand_hole_mask = binary_fill_holes(canvas_b) & ~canvas_b
            cand_hole_area = int((cand_hole_mask & comp_mask).sum())

        switched = decisive

        if debug_scores is not None:
            debug_scores.append((float(iou_a), float(iou_b), bool(switched), float(h), int(area),
                                  float(cand_hole_area / pen_footprint)))

        if cache is not None:
            cache[cache_key] = (switched, polys_b if switched else [], float(h))
        if switched:
            n_switched += 1
            min_switched_h = float(h) if min_switched_h is None else min(min_switched_h, float(h))
            exclude_mask[y0:y1, x0:x1] |= comp_mask
            extra_polys += [[(px + x0, py + y0) for px, py in p] for p in polys_b]
    return exclude_mask, extra_polys, n_checked, n_switched, (min_switched_h or 0.0)


def _binarize_mask(img, opts):
    """Threshold ladder for Trace V3 (own copy, so the other modes' thresholding
    stays byte-identical). threshold_mode: otsu (default) / manual / adaptive /
    sauvola, plus a bilevel fast path. The experimental options (pre_blur=0,
    threshold_mode="sauvola", native_res) are opt-in; omitting them gives the
    default behaviour unchanged.

    Blur: a (3,3) Gaussian is applied before thresholding by default
    (pre_blur=3; even values are bumped to the next odd size). pre_blur=0
    disables it, but then anti-aliased edge speckle survives thresholding
    attached to strokes, where the dot-cull cannot remove it, and grains the
    plot; that is why it is not the default. Bilevel sources always skip the
    blur because it rings on hard ink edges.

    Precedence: an explicit non-default threshold_mode ("manual" / "adaptive" /
    "sauvola") always wins over the bilevel fast path, so the opt-in remedy for
    faint-ink sources is reachable. "otsu" is indistinguishable from the unset
    default at this layer (callers, i.e. the UI and /api/process, always fill
    threshold_mode with "otsu" when the user has not chosen anything else), so
    it is treated as auto/default. The fast path only matters when the bilevel
    gate fires (mean((img<40)|(img>215)) > 0.9), i.e. a bold page with a faint
    minority hatch; ordinary sources sit below that gate (the highest observed
    on the reference fixtures is about 0.81) and are thresholded with Otsu."""
    import cv2, numpy as np
    tmode = str(opts.get("threshold_mode", "otsu")).lower()
    bilevel_gate = float(np.mean((img < 40) | (img > 215))) > 0.9
    explicit_mode = tmode in ("manual", "adaptive", "sauvola")
    # Bilevel sources skip the pre-blur regardless of which branch below ends up
    # firing (blurring an already-hard-edged bilevel scan rings on the edges);
    # this also holds when an explicit threshold_mode overrides the fast path.
    pre_blur = int(opts.get("pre_blur", 3))
    if bilevel_gate or pre_blur <= 0:
        src = img
    else:
        k = pre_blur if pre_blur % 2 == 1 else pre_blur + 1
        src = cv2.GaussianBlur(img, (k, k), 0)
    fast_path = bilevel_gate and not explicit_mode
    if opts.get("bw_cut") == "mid" and not fast_path and not explicit_mode:
        # bwphoto-only opt-in (routes/core.py sets this key ONLY under
        # style=bwphoto and only when non-default): fixed mid-grey cut instead of
        # Otsu (which lands ~145-169 grey on the XDoG photos). Same pre-blur as above.
        _, m = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY_INV)
    elif fast_path:
        _, m = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY_INV)
    elif tmode == "manual":
        t = int(opts.get("manual_threshold", 128))
        _, m = cv2.threshold(src, t, 255, cv2.THRESH_BINARY_INV)
    elif tmode == "adaptive":
        b = int(opts.get("adaptive_block", 25)); b = b + 1 if b % 2 == 0 else b; b = max(3, b)
        C = int(opts.get("adaptive_c", 7))
        m = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, b, C)
    elif tmode == "sauvola":
        # Sauvola: a local threshold suited to thin strokes on a light ground
        # (unevenly lit or photographic hatching). Otsu does better on clean
        # uniform-background line art, so it stays the default; Sauvola is opt-in
        # for sources where a global cut clips faint hatch.
        from skimage.filters import threshold_sauvola
        b = int(opts.get("adaptive_block", 25)); b = b + 1 if b % 2 == 0 else b; b = max(3, b)
        th = threshold_sauvola(src, window_size=b, k=float(opts.get("sauvola_k", 0.2)))
        m = ((src < th).astype(np.uint8)) * 255
    else:
        _, m = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m > 0


def _load_native_gray(raw_path, cap=NATIVE_CAP):
    """Load a source grayscale at its NATIVE resolution (capped, never upscaled).
    Used by the opt-in native_res path: gaps must be thresholded where they are
    real pixels, before any resample smears them. Distinct from _load_gray,
    which resizes to a working resolution (and upscales small sources, which
    fills in narrow hatch gaps)."""
    import cv2, numpy as np
    from PIL import Image
    g = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        g = cv2.cvtColor(np.array(Image.open(raw_path).convert("RGB")), cv2.COLOR_RGB2GRAY)
    h, w = g.shape[:2]
    m = max(h, w)
    if m > cap:
        s = cap / m
        g = cv2.resize(g, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return g


# ── Shared trace/lineart leaf utilities ─────────────────────────────────────
# _smooth_polys and _gray_crush_metric are called from both _trace_svg
# (trace_v3.py) and _lineart_svg (lineart.py). They live here so neither module
# has to import the other (trace_v3 and lineart are kept acyclic). Both are
# stateless (no module globals) and depend only on cv2/numpy.

def _smooth_polys(polys, k=5):
    """Moving-average smoothing of pixel-lattice polylines. Skeleton tracing
    emits integer pixel coords with ±1px jitter around the true centerline;
    the downstream Douglas-Peucker simplifier turns that jitter into angular
    zig-zags ("muddied" plots). Box-smooth first so DP sees the real curve.
    Open endpoints stay anchored; closed loops are smoothed circularly."""
    import numpy as np
    k = int(k) | 1                       # odd
    if k < 3:
        return polys
    out = []
    for poly in polys:
        # Adaptive window: short strokes are glyphs/ticks/small details whose
        # shape IS the signal — full-window smoothing rounds them into mush.
        ke = min(k, max(1, (len(poly) // 10)) | 1)
        if ke < 3 or len(poly) < ke + 2:
            out.append(poly); continue
        kern, pad = np.ones(ke) / ke, ke // 2
        a = np.asarray(poly, float)
        closed = np.hypot(*(a[0] - a[-1])) <= 2.0
        if closed:
            ext = np.vstack([a[-pad - 1:-1], a, a[1:pad + 1]])
        else:
            ext = np.vstack([np.repeat(a[:1], pad, 0), a, np.repeat(a[-1:], pad, 0)])
        sm = np.column_stack([np.convolve(ext[:, 0], kern, 'valid'),
                              np.convolve(ext[:, 1], kern, 'valid')])
        if closed:
            sm[-1] = sm[0]
        else:
            sm[0], sm[-1] = a[0], a[-1]
        out.append([tuple(p) for p in sm])
    return out


def _gray_crush_metric(gray, mask, mm_per_px, halfwidth_mm=1.0, lo=0.25, hi=0.85, majority=0.5):
    """Otsu shadow-crush metric. Fraction
    of the FINAL ink mask's area that sits inside 'blob' connected components
    -- components whose max local half-width (mask distance transform peak)
    exceeds `halfwidth_mm` (default 1.0mm = a 2mm-wide blob at the 0.5mm pen
    all five calibration sources used -- wider than any stroke a single/double
    pen pass could physically produce, i.e. genuinely a filled area, not a
    line) AND whose underlying PRE-THRESHOLD grayscale is majority (>=50% of
    the blob's pixels) in the `[lo,hi]` mid-gray band. This targets the
    diagnosed failure mode directly (fine gray/hatch structure that Otsu
    binarizes into one solid dark blob) while excluding thin-stroke/text AA
    edges and genuinely near-black source ink. A naive whole-mask mid-gray
    pixel fraction is not used because it scores clean sources as high as
    crushed ones. On the calibration sources the metric was 0.0% for the
    acceptable ones and 2.0% / 93.8% for the two shadow-crushed ones, which
    separates cleanly around the 1% trigger threshold used by build_warnings.

    `mm_per_px` here is the caller's paper-width/max-dimension heuristic
    (e.g. (paper_w_mm - 2*margin_mm) / working_w), not the exact drawn
    scale of this particular image -- it doesn't account for aspect-ratio
    letterboxing or per-axis scale differences. This is fine for the blob
    half-width test: the `blob_halfwidth_mult` sweep (1.0x-4.0x pen width)
    was stable across the whole range, i.e. the metric does not depend on
    getting `halfwidth_mm` -> pixels exactly right."""
    import cv2, numpy as np
    mask_u8 = mask.astype(np.uint8)
    n_ink = int(mask_u8.sum())
    if n_ink == 0:
        return 0.0
    dt = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    n, lbl, _, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if n <= 1:
        return 0.0
    g_norm = gray.astype(np.float32) / 255.0
    midband = (g_norm >= lo) & (g_norm <= hi)
    blob_thresh_px = halfwidth_mm / max(1e-6, mm_per_px)
    # Vectorized label-wise aggregation (same trick as the fallback-routing
    # block in _level_set_passes): per-label max(dt) via np.maximum.at,
    # per-label area and midband-pixel counts via np.bincount. Avoids a
    # per-component Python loop, which is slow with thousands of components.
    lbl_flat = lbl.ravel()
    maxdt = np.zeros(n, np.float64)
    np.maximum.at(maxdt, lbl_flat, dt.ravel())
    area = np.bincount(lbl_flat, minlength=n)
    midband_count = np.bincount(lbl_flat, weights=midband.ravel().astype(np.float64), minlength=n)
    midband_frac = np.divide(midband_count, area, out=np.zeros(n), where=area > 0)
    is_blob = (maxdt > blob_thresh_px) & (midband_frac >= majority)
    is_blob[0] = False  # label 0 is background
    blob_ink_px = int(area[is_blob].sum())
    return blob_ink_px / n_ink
