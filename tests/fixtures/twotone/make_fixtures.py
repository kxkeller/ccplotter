r"""Regenerates the fixture inputs and the REFERENCE hashes from the frozen prototype script
scripts/recipe_frozen.py (read-only; the prototype directory is not part of this repository and is
located through CCPLOTTER_TWOTONE_PROTO). Committed fixtures: dahlia and pagoda (scikit-learn
sample images, CC BY 2.0; byte copies) and synth (a synthetic gradient-plus-shapes image generated here, no licence needed).
Two further images whose source could not be confirmed are deliberately not fixtures.
Run with Python 3.12; it writes only inside this folder and deletes its scratch dirs."""
import hashlib, json, os, shutil, sys
from pathlib import Path
import cv2
import numpy as np
HERE = Path(__file__).resolve().parent
if "CCPLOTTER_TWOTONE_PROTO" not in os.environ:
    raise SystemExit("set CCPLOTTER_TWOTONE_PROTO to the twotone-proto-2026-10-15 prototype directory "
                     "(it holds scripts/recipe_frozen.py); it is not part of this repository")
MAIN = Path(os.environ["CCPLOTTER_TWOTONE_PROTO"])
os.environ["CCPLOTTER_HOME"] = str(HERE / "_home")           # private; recipe_frozen only needs it to pass the runtime guard
(HERE / "_home").mkdir(exist_ok=True)
for f in ("calibration.json", "pen_profiles.json", "vpype.toml"):
    src = MAIN / "_env_import" / f
    if src.exists():
        (HERE / "_home" / f).write_bytes(src.read_bytes())
sys.path.insert(0, str(MAIN / "scripts"))


def make_synth(path, w=1200, h=800):
    """Deterministic synthetic photo: smooth tonal gradient + soft vignette (tone bands) + hard-edged shapes (edges, dark mass) + light texture."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = 235.0 - 150.0 * (xx / w) - 40.0 * (yy / h)
    g *= 1.0 - 0.25 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    img = np.stack([g * 0.95, g, g * 1.05], -1).clip(0, 255).astype(np.uint8)
    cv2.rectangle(img, (120, 120), (480, 420), (40, 40, 40), -1)               # dark mass (shadow band)
    cv2.circle(img, (760, 300), 140, (215, 215, 215), -1)                      # light disc on the gradient
    cv2.circle(img, (760, 300), 140, (20, 20, 20), 6)
    cv2.line(img, (60, 700), (1140, 560), (10, 10, 10), 5)                     # long edge
    for k in range(12):                                                        # bars of falling tone
        cv2.rectangle(img, (560 + 40 * k, 560), (590 + 40 * k, 760), (30 + 18 * k,) * 3, -1)
    rng = np.random.RandomState(7)
    img = (img.astype(np.int16) + rng.randint(-5, 6, img.shape[:2])[..., None]).clip(0, 255).astype(np.uint8)   # light texture
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


S = MAIN / "inputs12"
shutil.copyfile(S / "dahlia.jpg", HERE / "dahlia.jpg")
shutil.copyfile(S / "pagoda.jpg", HERE / "pagoda.jpg")
make_synth(HERE / "synth.jpg")
import recipe_frozen as R
out = {"source": "recipe_frozen.py", "recipe_frozen_sha256": hashlib.sha256((MAIN / "scripts" / "recipe_frozen.py").read_bytes()).hexdigest(),
       "inputs_sha256": {}, "default_257x182": {}, "legacy_257x176": {}}
for n in ("dahlia", "pagoda", "synth"):
    p = HERE / f"{n}.jpg"
    out["inputs_sha256"][n] = hashlib.sha256(p.read_bytes()).hexdigest()
    out["default_257x182"][n] = R.run(p, HERE / "_out", n)
    out["legacy_257x176"][n] = R.run(p, HERE / "_out", n, page=(257.0, 176.0), legacy=True)
(HERE / "reference_hashes.json").write_text(json.dumps(out, indent=1))
for d in ("_home", "_out", "dbv3_inbox", "work"):
    shutil.rmtree(HERE / d, ignore_errors=True)
print(json.dumps(out, indent=1))
