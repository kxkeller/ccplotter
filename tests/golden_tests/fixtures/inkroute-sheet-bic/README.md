D1917a golden `inkroute-sheet-bic` (registry sheet inkroute_bic, recorded 2026-10-29 with Kai's "1-yes").

81190cd7... is the HERMETIC sha: the harness runs with scripts/harness_env/calibration.json (pen offset -1.0/-45.0). The PLOT reference is f8850f3c...
(live offset -1.3/-43.7; artifacts/inkroute-paper-sheet-2026-10-29/sheet/). They differ by design -- every coordinate shifts by the offset difference and the
"; Source:" line is a relative path here. The mismatch is expected; do not "repair" either sha to match the other.

D1924 summary: 81190cd7 = the HERMETIC sha (harness pen offset -1.0/-45.0); f8850f3c = the live reference (live pen offset -1.3/-43.7, the sheet Kai plots). Both are correct for their environment.
