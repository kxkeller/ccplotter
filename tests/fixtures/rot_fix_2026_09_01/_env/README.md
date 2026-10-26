# Frozen pen profiles for tests

`pen_profiles.json` here is a frozen copy of the author's own pen profiles
(seven pens, with the author's measured contact heights, pen widths and mount
offsets). It exists only as input for the tests that read it
(`test_calibration_fragment`, `test_calibration_photofe`,
`test_calibration_registry_safety`, `test_calibration_uright_short`,
`test_rot_fit_fix`), which copy it into a temporary runtime home.

It is NOT a calibration for any other machine. Do not copy it into your own
`CCPLOTTER_HOME` or plot with it: the Z heights and offsets are specific to one
printer and one set of pens. Measure your own values (see the calibration notes).

SHA-256 of `pen_profiles.json`:
`9dc255548638e0c4144667f140063a93ce66b0afc9f208de453ddb1abee8711d`
