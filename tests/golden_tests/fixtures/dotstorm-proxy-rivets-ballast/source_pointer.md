# Source image pointer

This fixture does not carry its own copy of the source image. Use:

    tests/golden_tests/fixtures/loco-472-trestle/source.png

`request.json` refers to the source by upload id `01a34e00`. The image that request consumed and
`loco-472-trestle/source.png` were compared by hash and are byte-identical:

```
56dfea7088bafa8372694b70e169169283317a977b57081796cdefaf9b3d8178  loco-472-trestle/source.png
```

The upload ids `01a34e00` (here) and `6f2e049b` (in `loco-472-trestle`) are separate uploads of the same
image file, not different images.
