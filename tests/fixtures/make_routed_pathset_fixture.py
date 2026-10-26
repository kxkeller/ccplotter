"""One-off generator for tests/fixtures/routed_pathset_mini.svg.

Takes the first 120 polylines and 120 polygons of the cleaned-heli REF routed SVG (the file whose
published path-set hash is d729a9365f), puts them in two layers, adds two <line> elements, and writes
a small routed-style SVG. The EXPECTED hash in tests/test_trace_boundary.py was computed on the output
by an independent reference implementation (tests/fixtures/reference_scripts/cleaned-heli:
lib.parse_svg + metrics.pathset(.,3)), not by ccplotter.trace_boundary.
usage: py make_routed_pathset_fixture.py <source layout.routed.svg> <out.svg>
"""
import re, sys
from pathlib import Path
t = Path(sys.argv[1]).read_text(encoding="utf-8")
head = t[:t.index("<g ")]
pl = re.findall(r'<polyline points="[^"]+"\s*/>', t)[:120]
pg = re.findall(r'<polygon points="[^"]+"\s*/>', t)[:120]
g = '  <g fill="none" id="layer%d" inkscape:groupmode="layer" inkscape:label="%d" stroke="#000000" style="display:inline">\n%s\n  </g>\n'
lines = '    <line x1="10.5" y1="20.25" x2="30.125" y2="40.5"/>\n    <line x1="31" y1="41" x2="50.0625" y2="60"/>'
out = head + g % (1, 1, "\n".join("    " + p for p in pl)) + g % (2, 2, "\n".join("    " + p for p in pg) + "\n" + lines) + "</svg>\n"
Path(sys.argv[2]).write_text(out, encoding="utf-8")
print(len(pl), len(pg), len(out))
