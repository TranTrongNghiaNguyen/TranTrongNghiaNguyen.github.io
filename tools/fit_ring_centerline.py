"""Fit the *centreline* of a stroked ring, unbiased by stroke width.

fit_rings.py fits the outer edge of the dark band, so its radii are biased
outward by roughly half the stroke width. Here the ink is used as a weight:
for each polar angle bin the ink-weighted mean radius is taken inside a narrow
normalised band, which lands on the centre of the stroke. The cross-band ink
integral then gives the stroke width.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from fit_rings import conic_to_ellipse, fit_conic, load_gray

ROOT = Path(__file__).resolve().parent.parent


def normalised_radius(shape, p, ys, xs):
    t = math.radians(p["rot"])
    ct, st = math.cos(t), math.sin(t)
    dx, dy = xs - p["cx"], ys - p["cy"]
    u = (dx * ct + dy * st) / p["rx"]
    v = (-dx * st + dy * ct) / p["ry"]
    return np.sqrt(u**2 + v**2)


def refine(cov: np.ndarray, p: dict, half_width: float, bins: int) -> tuple[dict, float, float]:
    h, w = cov.shape
    ys, xs = np.mgrid[0:h, 0:w]
    rho = normalised_radius(cov.shape, p, ys, xs)
    scale = 0.5 * (p["rx"] + p["ry"])
    band = np.abs(rho - 1.0) * scale < half_width
    sel = band & (cov > 0.02)
    if sel.sum() < bins:
        raise SystemExit(f"only {int(sel.sum())} band pixels; widen --half-width")

    px, py, pw = xs[sel].astype(float), ys[sel].astype(float), cov[sel]
    ang = np.arctan2(py - p["cy"], px - p["cx"])
    rad = np.hypot(px - p["cx"], py - p["cy"])
    idx = ((ang + math.pi) / (2 * math.pi) * bins).astype(int) % bins

    wsum = np.bincount(idx, weights=pw, minlength=bins)
    rsum = np.bincount(idx, weights=pw * rad, minlength=bins)
    asum = np.bincount(idx, weights=pw * ang, minlength=bins)
    good = wsum > 0.05
    rbar = rsum[good] / wsum[good]
    abar = asum[good] / wsum[good]
    pts = np.column_stack([p["cx"] + rbar * np.cos(abar), p["cy"] + rbar * np.sin(abar)])

    new = conic_to_ellipse(fit_conic(pts))
    if new is None:
        raise SystemExit("refined conic was not an ellipse")

    # Ink per unit arc length -> stroke width once divided by the core ink value.
    arc = math.pi * (3 * (new["rx"] + new["ry"]) - math.sqrt((3 * new["rx"] + new["ry"]) * (new["rx"] + 3 * new["ry"])))
    cross = float(pw.sum()) / arc
    core = float(cov[sel].max())
    return new, cross, core


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="airfoil - copy.png")
    ap.add_argument("--seed", required=True, help="cx,cy,rx,ry,rot of an approximate ring")
    ap.add_argument("--half-width", type=float, default=4.0)
    ap.add_argument("--bins", type=int, default=720)
    ap.add_argument("--iters", type=int, default=4)
    args = ap.parse_args()

    path = Path(args.image) if Path(args.image).is_absolute() else ROOT / args.image
    gray = load_gray(path)
    cov = np.clip((255.0 - gray) / 255.0, 0.0, 1.0)

    cx, cy, rx, ry, rot = (float(v) for v in args.seed.split(","))
    p = {"cx": cx, "cy": cy, "rx": rx, "ry": ry, "rot": rot}
    for i in range(args.iters):
        p, cross, core = refine(cov, p, args.half_width, args.bins)
        print(
            f"iter {i}: cx={p['cx']:.3f} cy={p['cy']:.3f} rx={p['rx']:.3f} ry={p['ry']:.3f}"
            f" rot={p['rot']:.4f}  ink/arc={cross:.4f} core={core:.3f}"
            f"  -> width~{cross / max(core, 1e-6):.3f}px"
        )
    print(
        f'  <ellipse cx="{p["cx"]:.3f}" cy="{p["cy"]:.3f}" rx="{p["rx"]:.3f}" ry="{p["ry"]:.3f}"'
        f' transform="rotate({p["rot"]:.4f} {p["cx"]:.3f} {p["cy"]:.3f})"/>'
    )


if __name__ == "__main__":
    main()
