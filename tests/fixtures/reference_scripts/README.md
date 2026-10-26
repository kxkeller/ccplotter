# Reference scripts used by tests

Copies of small scripts from the author's experiments. Tests load them to check
that the application's own parsing and hashing code agrees with these independent implementations.
They are inputs to tests, not part of the application.

- `bw-fixarms/corrected_mapping_v2.py`: reference SVG polyline parser.
- `fix-ab/ab_lib.py`, `stitch/ab_lib.py`: reference parsers from two A/B comparisons. The only
  change from the originals is that the repository root is found relative to the file instead of a
  hard-coded path.
- `cleaned-heli/lib.py`, `metrics.py`: reference parser and path-set hash used by `test_trace_boundary.py`.
  (The images those experiments used are not distributed.)
