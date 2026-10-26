# Benchmark image fixtures

Only the images that are the author's own are distributed. Each is a derivative of one photograph of
a Hong Kong tram (the author's own phone photo):

- `sources/04_hktram_source.jpg`: the photograph (1405x1874). No EXIF/GPS metadata (JFIF header only).
- `m09_rasters/tram__phi2_s512.png`: the XDoG line raster of it at side 512 (locked reference output of `bw_xdog_raster`).
- `anime_hires/tram_s1024.png`: a 1024 px line-drawing derivative of it, used as the benchmark source for the preflight and gap-audit tests.

The other benchmark images used by the original test suite (an ICE train, a bell tower, a bicycle, an
Airbus A350, a giraffe) are third-party and are not distributed. Tests that need them skip with the
message "fixture image not redistributable; place <name> at <path> to run". To run them, put the files at
the paths in the message (sources named `01_bike_source.jpg`, `02_belltower_source.jpg`,
`03_giraffe_source.png`, `06_a350_source.jpg`, `07_ice1_railway_source.jpg` in `sources/`, their
`<name>__phi2_s512.png` rasters in `m09_rasters/`, and `<name>_s1024.png` in `anime_hires/`).
