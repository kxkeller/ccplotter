# Source image pointer

This fixture does not carry its own copy of the source image. Use:

    tests/golden_tests/fixtures/loco-472-trestle/source.png

It is the same source as `dotstorm-proxy-rivets-ballast` (upload id `01a34e00`, SHA-256 recorded in that
fixture's `source_pointer.md`).

## Purpose

A flag-on variant fixture: it exercises `fidelity_isolation=trial` on the locomotive line-art
(`style=plotready`) request, so the opt-in isolation-gate path has its own byte-identity guard in the replay
harness.

`request.json` is `dotstorm-proxy-rivets-ballast/request.json` with exactly two changes:

- `"fidelity_isolation": "trial"` added (the flag under test).
- `"debug": false` (was `true`): debug intermediates are not among the harness's hashed fields, and this keeps
  the fixture a plain non-debug run, distinct from the debug=true case the other fixture covers.

Everything else (style `plotready`, resolution 3000, `small_detail_fidelity` true, upload id `01a34e00`) is
unchanged, so this fixture differs from `dotstorm-proxy-rivets-ballast` only in the isolation mode and the debug
flag.
