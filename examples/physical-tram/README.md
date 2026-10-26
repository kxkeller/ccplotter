# Physical plots: one Hong Kong tram photograph, four ways

These are scans of drawings that were plotted on paper with ccPlotter on the
author's machine. They are real outputs, but **this directory is evidence, not a
reproducible example**: the exact settings (and the exact software version) that
produced each plot were not recovered for this release, so they cannot be
re-run to the same G-code. For a reproducible run, with G-code and hashes, see
[../synthetic-trace/](../synthetic-trace/README.md) (software only).

The source photograph is the author's own. Image metadata (EXIF, GPS) has been
stripped from every file here, and the files were downsized to about 1400 px
wide for the repository.

| File | What it is | Mode, as captioned on the project's website |
|---|---|---|
| `source-photo.jpg` | The source photograph | not a plot |
| `line-art.jpg` | Physical plot (scan) | Prepared line art |
| `trace-v3.jpg` | Physical plot (scan) | Trace V3 |
| `ink-sketch.jpg` | Physical plot (scan) | Ink Sketch |
| `two-pen.jpg` | Physical plot (scan) | Black and blue, two pens |

The captions describe the approach as follows. "Prepared line art" means the
source was first converted into line-only artwork without filled regions.
"Trace V3" also builds up filled areas with additional strokes. "Ink Sketch"
processes the photograph directly with deterministic processing. The names
"Prepared line art" and "Ink Sketch" are the website's wording; this release did
not establish which option in the current mode selector each corresponds to, so
no mapping is claimed. The two-pen plot used two pens, black and blue.

What is not known: pen, pass spacing, resolution, other options and the software
version for each plot. Do not infer settings from the images.

Physical plots differ from the software preview: strokes that look separate in a
preview can merge on paper at pen width. See
[../../docs/LIMITATIONS.md](../../docs/LIMITATIONS.md).
