#!/usr/bin/env python3
"""Compare two plot-scale raster renders with deterministic grayscale SSIM."""
import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def load_grayscale(path: Path) -> np.ndarray:
    """Load an image as unresized grayscale, compositing transparency over white."""
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        flattened = Image.alpha_composite(white, rgba)
        return np.asarray(flattened.convert("L"))


def compute_ssim(reference: Path, candidate: Path) -> float:
    """Return grayscale SSIM, rejecting comparisons with unequal dimensions."""
    ref = load_grayscale(reference)
    got = load_grayscale(candidate)
    if ref.shape != got.shape:
        raise ValueError(
            "SSIM image dimensions must match exactly: "
            f"reference {ref.shape[1]}x{ref.shape[0]}, "
            f"candidate {got.shape[1]}x{got.shape[0]}"
        )
    return float(structural_similarity(ref, got, data_range=255))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path, nargs="?")
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--min", dest="minimum", type=float, default=0.999)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        ref = np.full((32, 32), 255, np.uint8); ref[8:24, 8:24] = 0
        shifted = np.roll(ref, 1, axis=1)
        same = structural_similarity(ref, ref, data_range=255)
        altered = structural_similarity(ref, shifted, data_range=255)
        with tempfile.TemporaryDirectory() as directory:
            transparent = Path(directory) / "transparent.png"
            opaque = Path(directory) / "opaque.png"
            Image.new("RGBA", (32, 32), (255, 255, 255, 0)).save(transparent)
            Image.new("RGB", (32, 32), (255, 255, 255)).save(opaque)
            alpha_same = compute_ssim(transparent, opaque)
        print(
            f"self_same={same:.6f} self_shifted={altered:.6f} "
            f"alpha_white={alpha_same:.6f}"
        )
        return 0 if same == 1.0 and altered < 1.0 and alpha_same == 1.0 else 2
    if not args.reference or not args.candidate:
        parser.error("reference and candidate are required unless --self-test is used")
    value = compute_ssim(args.reference, args.candidate)
    print(f"ssim={value:.6f} minimum={args.minimum:.6f}")
    return 0 if value >= args.minimum else 1


if __name__ == "__main__":
    raise SystemExit(main())
