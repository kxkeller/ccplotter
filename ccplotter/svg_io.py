import re
from pathlib import Path

import vpype as vp

from ccplotter.atomic_io import atomic_target, atomic_write_text
from ccplotter.config import PX2MM


def _dxf_to_svg(src: Path, out: Path) -> Path:
    import ezdxf
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.svg import SVGBackend
    doc = ezdxf.readfile(src)
    msp = doc.modelspace()
    ctx = RenderContext(doc)
    out_svg = SVGBackend()
    Frontend(ctx, out_svg).draw_layout(msp)
    atomic_write_text(out, out_svg.get_xml(), encoding="utf-8")
    return out

def _pdf_to_svg(src: Path, out: Path) -> Path:
    import fitz
    doc = fitz.open(src)
    page = doc.load_page(0)
    atomic_write_text(out, page.get_svgimage(), encoding="utf-8")
    return out

def _paper_spec(w: float, h: float) -> str:
    # vpype pagesize spec (e.g. '210mmx297mm')
    return f"{w:.1f}mmx{h:.1f}mm"

def _svg_bbox_mm(svg_path: Path):
    d = vp.read_multilayer_svg(str(svg_path), quantization=0.1)
    b = d.bounds()
    return [b[0] * PX2MM, b[1] * PX2MM, b[2] * PX2MM, b[3] * PX2MM] if b else None

def _write_paths_svg(out_path: Path, w: int, h: int, stroke_px: float, segments=(), polylines=()):
    import svgwrite
    with atomic_target(out_path) as tmp:
        dwg = svgwrite.Drawing(str(tmp), size=(f"{w}px", f"{h}px"), viewBox=f"0 0 {w} {h}")
        g = dwg.g(stroke="black", stroke_width=round(float(stroke_px), 3), fill="none")
        for x0, y0, x1, y1 in segments: g.add(dwg.line((round(x0, 2), round(y0, 2)), (round(x1, 2), round(y1, 2))))
        for pts in polylines: g.add(dwg.polyline([(round(float(x), 2), round(float(y), 2)) for x, y in pts]))
        dwg.add(g); dwg.save()
    return out_path

def _parse_dbv3_metadata(data: bytes) -> dict | None:
    """Parse the provenance comment DrawingBotV3 writes right after the <svg>
    tag. Real DBV3 1.6.25 exports look like:

        <?xml version="1.0" encoding="ISO-8859-1"?>
        ...
        <svg ...><!--Created with DrawingBotV3 Premium Version: 1.6.25 Stable
        Date: 2026-07-20 at 12:54:03 EDT
        Shapes: 2573
        Total Travel: 139.64 m
        Distance Down: 133.89 m
        Distance Up: 5.75 m
        Pen Lifts: 2572
        Page Size: 148.0 mm x 210.0 mm
        Drawing Size: 147.7 mm x 209.8 mm

        Pens:
         Pen 1: 15 Black - 133.89 m
        -->

    Declared encoding is ISO-8859-1, NOT UTF-8 -- decoding with the wrong
    codec can raise on any non-ASCII byte a future export might contain (pen
    names, unicode in file paths embedded by DBV3, etc). We only need to
    decode a small HEADER slice (comment sits right after <svg ...>, well
    before any path data), so this reads a bounded prefix, not the whole
    (potentially multi-MB) file, and decodes with the file's OWN declared
    encoding read out of the XML prolog (falls back to iso-8859-1, then
    utf-8 with errors='replace' if the prolog itself is missing/unparseable).
    Returns None (not an error) for any non-DBV3 SVG that simply lacks this
    comment -- inbox files aren't guaranteed to originate from DBV3 (the
    folder is just "wherever SVGs land"), so absence is normal.
    """
    head = data[:8192]  # comment is always near the top; bound the decode cost
    enc = "iso-8859-1"
    m = re.search(rb'encoding=["\']([\w-]+)["\']', head[:200])
    if m:
        try:
            enc = m.group(1).decode("ascii")
        except Exception:
            enc = "iso-8859-1"
    try:
        text = head.decode(enc, errors="replace")
    except (LookupError, UnicodeDecodeError):
        text = head.decode("iso-8859-1", errors="replace")

    cm = re.search(r'<!--\s*Created with DrawingBotV3(.*?)-->', text, re.DOTALL)
    if not cm:
        return None
    body = cm.group(1)

    def _grab(label, pattern=r'([^\r\n]+)'):
        mm = re.search(rf'{re.escape(label)}\s*:\s*{pattern}', body)
        return mm.group(1).strip() if mm else None

    pens = []
    for pm in re.finditer(r'Pen\s+\d+:\s*([^\r\n]+?)\s*-\s*([\d.]+\s*\w+)', body):
        pens.append({"name": pm.group(1).strip(), "distance": pm.group(2).strip()})

    meta = {
        "dbv3_version": _grab("Version"),
        "export_date": _grab("Date"),
        "shape_count": _grab("Shapes"),
        "total_travel": _grab("Total Travel"),
        "distance_down": _grab("Distance Down"),
        "distance_up": _grab("Distance Up"),
        "pen_lifts": _grab("Pen Lifts"),
        "page_size": _grab("Page Size"),
        "drawing_size": _grab("Drawing Size"),
        "pens": pens,
    }
    # Only worth keeping if at least one field actually matched -- an empty
    # skeleton dict is noise, not provenance.
    if not any(v for k, v in meta.items() if k != "pens") and not pens:
        return None
    return meta


def _read_polylines_svg(svg_path) -> list:
    """Read polylines (lists of (x,y)) from an SVG written by _write_paths_svg."""
    import re
    out = []
    try:
        t = open(svg_path, encoding="utf-8").read()
    except Exception:
        return out
    for m in re.findall(r'<polyline points="([^"]+)"', t):
        pts = []
        for tok in m.split():
            if "," in tok:
                a, b = tok.split(",")[:2]
                try: pts.append((float(a), float(b)))
                except ValueError: pass
        if len(pts) >= 2:
            out.append(pts)
    return out
