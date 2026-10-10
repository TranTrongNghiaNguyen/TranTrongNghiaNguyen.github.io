"""Split the ink budget by colour family so each stroke group can be tuned.

Categories mirror the SVG structure: navy outline strokes, red fan, blue fan /
flow lines, and the near-neutral guilloche mesh.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from logo_probe import PAIRS, ink, load_reference, render
from logo_compare import denoise


def classify(arr: np.ndarray, floor: float = 0.0) -> dict[str, np.ndarray]:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cov = denoise(ink(arr), floor)
    lit = cov > 0.0
    dark = lit & (arr.min(axis=2) < 150)
    warm = lit & ~dark & (r - b > 6)
    cool = lit & ~dark & (b - r > 6)
    neutral = lit & ~dark & ~warm & ~cool
    return {"dark": dark, "warm": warm, "cool": cool, "neutral": neutral}


def summarise(arr: np.ndarray, floor: float = 0.0) -> dict[str, tuple[float, int]]:
    cov = denoise(ink(arr), floor)
    out = {}
    for key, mask in classify(arr, floor).items():
        out[key] = (float(cov[mask].sum()), int(mask.sum()))
    out["all"] = (float(cov.sum()), int((cov > 0.0).sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", choices=list(PAIRS))
    ap.add_argument("--floor", type=float, default=0.012)
    args = ap.parse_args()

    for name in args.targets or list(PAIRS):
        svg, png = PAIRS[name]
        ref = load_reference(png)
        out = render(svg, (ref.shape[1], ref.shape[0]))
        a, b = summarise(ref, args.floor), summarise(out, args.floor)

        print(f"=== {name} ===")
        print(f"  {'class':>8} {'ink_ref':>12} {'ink_out':>12} {'ratio':>7} {'px_ref':>9} {'px_out':>9} {'px_ratio':>9}")
        for key in ("dark", "warm", "cool", "neutral", "all"):
            ir, pr = a[key]
            io_, po = b[key]
            print(
                f"  {key:>8} {ir:12,.1f} {io_:12,.1f} {io_ / max(ir, 1e-9):7.3f}"
                f" {pr:9,d} {po:9,d} {po / max(pr, 1):9.3f}"
            )
        # Mean ink per lit pixel says whether strokes are too faint vs too few.
        for label, s in (("ref", a), ("out", b)):
            ir, pr = s["all"]
            print(f"  {label} mean ink per lit px: {ir / max(pr, 1):.4f}")


if __name__ == "__main__":
    main()
