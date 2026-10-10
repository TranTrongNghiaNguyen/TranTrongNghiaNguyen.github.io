"""Fit general (rotated) ellipses to the dark rings of the reference PNG.

The two navy rings are the highest-contrast features in the emblem and carry
~46% of the reference ink, so their conic parameters are the biggest lever on
the residual error.

Method: isolate dark pixels, trace the outer boundary by taking the extreme
dark pixel on each scanline (rows and columns), then fit a general conic
  a x^2 + b x y + c y^2 + d x + e y + f = 0
by algebraic least squares and convert to centre / semi-axes / rotation.
Peel the fitted band away and repeat to reach the next ring inward.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGBA", img.size, (255, 255, 255, 255))
        flat.alpha_composite(img)
        img = flat
    return np.asarray(img.convert("RGB"), dtype=np.float64).min(axis=2)


def boundary_points(mask: np.ndarray) -> np.ndarray:
    """Extreme set pixels per row and per column -> outer boundary of mask."""
    pts: list[tuple[float, float]] = []
    h, w = mask.shape
    for y in range(h):
        xs = np.nonzero(mask[y])[0]
        if xs.size:
            pts.append((float(xs[0]), float(y)))
            pts.append((float(xs[-1]), float(y)))
    for x in range(w):
        ys = np.nonzero(mask[:, x])[0]
        if ys.size:
            pts.append((float(x), float(ys[0])))
            pts.append((float(x), float(ys[-1])))
    return np.array(pts, dtype=np.float64)


def fit_conic(pts: np.ndarray) -> np.ndarray:
    """Least-squares general conic through pts (returns A..F in pixel coords)."""
    x, y = pts[:, 0], pts[:, 1]
    mx, my = x.mean(), y.mean()
    s = max(x.std(), y.std())
    xn, yn = (x - mx) / s, (y - my) / s
    d = np.column_stack([xn**2, xn * yn, yn**2, xn, yn, np.ones_like(xn)])
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    a, b, c, dd, e, f = vt[-1]
    # Undo the conditioning substitution xn = (x - mx) / s.
    A = a / s**2
    B = b / s**2
    C = c / s**2
    D = dd / s - 2 * a * mx / s**2 - b * my / s**2
    E = e / s - 2 * c * my / s**2 - b * mx / s**2
    F = (
        f
        - dd * mx / s
        - e * my / s
        + a * mx**2 / s**2
        + b * mx * my / s**2
        + c * my**2 / s**2
    )
    return np.array([A, B, C, D, E, F], dtype=np.float64)


def conic_to_ellipse(co: np.ndarray) -> dict[str, float] | None:
    a, b, c, d, e, f = co
    m2 = np.array([[a, b / 2], [b / 2, c]])
    if np.linalg.det(m2) <= 0:
        return None  # hyperbola / parabola, not an ellipse
    cx, cy = np.linalg.solve(2 * m2, [-d, -e])
    fc = f + (d * cx + e * cy) / 2
    evals, evecs = np.linalg.eigh(m2)
    if np.any(evals == 0):
        return None
    axes2 = -fc / evals
    if np.any(axes2 <= 0):
        return None
    axes = np.sqrt(axes2)
    order = np.argsort(-axes)
    rx, ry = axes[order][0], axes[order][1]
    vec = evecs[:, order[0]]
    rot = math.degrees(math.atan2(vec[1], vec[0]))
    while rot > 90:
        rot -= 180
    while rot <= -90:
        rot += 180
    return {"cx": float(cx), "cy": float(cy), "rx": float(rx), "ry": float(ry), "rot": float(rot)}


def ellipse_residual(pts: np.ndarray, p: dict[str, float]) -> tuple[float, float]:
    """Approximate radial residual (px) of pts about the fitted ellipse."""
    t = math.radians(p["rot"])
    ct, st = math.cos(t), math.sin(t)
    dx, dy = pts[:, 0] - p["cx"], pts[:, 1] - p["cy"]
    u = (dx * ct + dy * st) / p["rx"]
    v = (-dx * st + dy * ct) / p["ry"]
    r = np.sqrt(u**2 + v**2)
    res = np.abs(r - 1.0) * (0.5 * (p["rx"] + p["ry"]))
    return float(res.mean()), float(np.percentile(res, 95))


def ring_band_mask(shape: tuple[int, int], p: dict[str, float], width: float) -> np.ndarray:
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    t = math.radians(p["rot"])
    ct, st = math.cos(t), math.sin(t)
    dx, dy = xs - p["cx"], ys - p["cy"]
    u = (dx * ct + dy * st) / p["rx"]
    v = (-dx * st + dy * ct) / p["ry"]
    r = np.sqrt(u**2 + v**2)
    return np.abs(r - 1.0) * (0.5 * (p["rx"] + p["ry"])) < width


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="airfoil - copy.png")
    ap.add_argument("--thresh", type=float, default=150.0)
    ap.add_argument("--rings", type=int, default=2)
    ap.add_argument("--peel", type=float, default=6.0)
    ap.add_argument("--crop", default="", help="x0,y0,x1,y1 to restrict the search")
    args = ap.parse_args()

    path = Path(args.image) if Path(args.image).is_absolute() else ROOT / args.image
    gray = load_gray(path)
    if args.crop:
        x0, y0, x1, y1 = (int(v) for v in args.crop.split(","))
        keep = np.zeros(gray.shape, dtype=bool)
        keep[y0:y1, x0:x1] = True
    else:
        keep = np.ones(gray.shape, dtype=bool)
    mask = (gray < args.thresh) & keep
    print(f"{path.name}: {gray.shape[1]}x{gray.shape[0]}, dark px {int(mask.sum()):,}")

    work = mask.copy()
    for i in range(args.rings):
        pts = boundary_points(work)
        if pts.shape[0] < 50:
            print(f"ring {i}: only {pts.shape[0]} boundary points, stopping")
            break
        p = conic_to_ellipse(fit_conic(pts))
        if p is None:
            print(f"ring {i}: conic fit was not an ellipse, stopping")
            break
        mean_res, p95 = ellipse_residual(pts, p)
        print(
            f"ring {i}: cx={p['cx']:.3f} cy={p['cy']:.3f} rx={p['rx']:.3f} ry={p['ry']:.3f}"
            f" rot={p['rot']:.4f}  residual mean={mean_res:.3f}px p95={p95:.3f}px n={pts.shape[0]}"
        )
        print(
            f'        <ellipse cx="{p["cx"]:.3f}" cy="{p["cy"]:.3f}" rx="{p["rx"]:.3f}"'
            f' ry="{p["ry"]:.3f}" transform="rotate({p["rot"]:.4f} {p["cx"]:.3f} {p["cy"]:.3f})"/>'
        )
        work &= ~ring_band_mask(work.shape, p, args.peel)


if __name__ == "__main__":
    main()
