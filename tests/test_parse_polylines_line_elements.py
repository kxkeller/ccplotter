"""Regression: the reference SVG parsers must see <line> strokes.

The routed-SVG writer emits every two-point stroke as <line x1 x2 y1 y2/>. A parser
that scans only points="..." silently drops those strokes (on the heli sheet, 1754 straight
hybrid core rows). Covers corrected_mapping_v2.parse_polylines_mm and the fix-ab / stitch ab_lib.parse_routed.

Positive control = the two-point stroke must be returned. Null control = an SVG with no <line> returns
exactly what the old points-only parser returned (a frozen copy of the old body lives in this file).
"""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PX2MM_TEST = 0.25


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


CM = _load(REPO / "tests" / "fixtures" / "reference_scripts" / "bw-fixarms" / "corrected_mapping_v2.py", "cm_v2_under_test")


def _old_parse_polylines_mm(svg_path, px2mm):
    """Frozen copy of the old points-only body of corrected_mapping_v2.parse_polylines_mm."""
    text = Path(svg_path).read_text(encoding="utf-8")
    polys = []
    for ps in re.findall(r'points="([^"]+)"', text):
        co = [tuple(map(float, t.split(','))) for t in ps.split() if ',' in t]
        if len(co) >= 2:
            polys.append([(x * px2mm, y * px2mm) for x, y in co])
    return polys


SVG = ('<svg xmlns="http://www.w3.org/2000/svg">\n'
       '    <polyline points="1.0,2.0 3.0,4.0 5.0,2.0"/>\n'
       '    <line x1="10.5" x2="12.5" y1="20" y2="22.25"/>\n'
       '    <polygon points="0,0 4,0 4,4 0,4"/>\n'
       '    <line y2="9" x1="7" y1="8" x2="6"/>\n'
       '</svg>\n')


def test_two_point_line_stroke_is_returned_by_parse_polylines_mm(tmp_path):
    p = tmp_path / "a.svg"
    p.write_text(SVG, encoding="utf-8")
    got = CM.parse_polylines_mm(p, PX2MM_TEST)
    k = PX2MM_TEST
    assert [(10.5 * k, 20 * k), (12.5 * k, 22.25 * k)] in got
    assert [(7 * k, 8 * k), (6 * k, 9 * k)] in got            # attribute order must not matter
    assert len(got) == 4                                       # polyline, polygon, and BOTH lines
    # the old parser provably dropped them (the control can fail)
    assert len(_old_parse_polylines_mm(p, k)) == 2


def test_svg_without_line_elements_is_unchanged_vs_old_parser(tmp_path):
    p = tmp_path / "b.svg"
    p.write_text(SVG.replace('    <line x1="10.5" x2="12.5" y1="20" y2="22.25"/>\n', "")
                    .replace('    <line y2="9" x1="7" y1="8" x2="6"/>\n', ""), encoding="utf-8")
    assert CM.parse_polylines_mm(p, PX2MM_TEST) == _old_parse_polylines_mm(p, PX2MM_TEST)


@pytest.mark.parametrize("which", ["fix-ab", "stitch"])
def test_ab_lib_parse_routed_returns_two_point_strokes(tmp_path, which):
    lib = _load(REPO / "tests" / "fixtures" / "reference_scripts" / which / "ab_lib.py", f"ab_lib_{which.replace('-', '_')}")
    lib.PX2MM = 1.0                                           # skip the repo import; scale is irrelevant here
    p = tmp_path / "c.svg"
    p.write_text(SVG, encoding="utf-8")
    got = lib.parse_routed(p)
    assert len(got) == 4
    assert any(len(a) == 2 and abs(a[0][0] - 10.5) < 1e-9 and abs(a[1][1] - 22.25) < 1e-9 for a in got)
    assert any(len(a) == 2 and abs(a[0][0] - 7) < 1e-9 and abs(a[1][0] - 6) < 1e-9 for a in got)
