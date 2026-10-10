"""Track the emblem's flow streamlines and red fan rays as continuous curves.

The previous generator (rebuild_logo_flow_and_tail.py) built the flow bands by
interpolating a handful of hand-entered guide values with PCHIP. PCHIP is only
C1, so curvature jumps at every guide point, and the per-4px cubic emission
faithfully reproduced those jumps: the rendered streamlines visibly kink instead
of flowing. The red fan was worse - straight segments to fixed end points, so
the rays cut across the blue streamlines instead of hugging them and spreading
gradually.

This module measures the strokes instead. Each stroke is followed column by
column with sub-pixel ridge detection and slope prediction, which keeps
neighbouring streamlines from being swapped, then the polyline is fitted with a
natural C2 cubic spline. Only the spline knots are written to the SVG, so the
emitted path is smooth by construction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent.parent


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGBA", img.size, (255, 255, 255, 255))
        flat.alpha_composite(img)
        img = flat
    return np.asarray(img.convert("RGB"), dtype=np.float64)


def responses(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Per-family ridge response maps.

    `ink` is total coverage; `chroma` is positive for the cool strokes and
    negative for the warm ones, which separates the red fan from the blue
    streamlines even where they cross.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ink = 255.0 - rgb.min(axis=2)
    chroma = (b + g) / 2.0 - r
    return {
        "ink": ink,
        "red": np.where(chroma < -2.0, ink, 0.0),
        "blue": np.where(chroma > 2.0, ink, 0.0),
        "dark": 255.0 - rgb.mean(axis=2) + 0.55 * np.maximum(0.0, b - r),
    }


def column_peaks(resp: np.ndarray, x: int, y0: float, y1: float, thresh: float) -> list[float]:
    """Sub-pixel local maxima of one column between y0 and y1."""
    lo, hi = max(1, int(np.floor(y0))), min(resp.shape[0] - 2, int(np.ceil(y1)))
    col = resp[:, x]
    out: list[float] = []
    for y in range(lo, hi + 1):
        if col[y] < thresh or col[y] < col[y - 1] or col[y] < col[y + 1]:
            continue
        denom = col[y - 1] - 2.0 * col[y] + col[y + 1]
        shift = 0.0 if abs(denom) < 1e-9 else 0.5 * (col[y - 1] - col[y + 1]) / denom
        out.append(y + float(np.clip(shift, -1.0, 1.0)))
    merged: list[float] = []
    for y in out:
        if merged and y - merged[-1] < 2.0:
            merged[-1] = 0.5 * (merged[-1] + y)
        else:
            merged.append(y)
    return merged


@dataclass
class Stream:
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(self.xs)
        return np.asarray(self.xs)[order], np.asarray(self.ys)[order]


def track(
    resp: np.ndarray,
    seed_x: int,
    seed_y: float,
    x_min: int,
    x_max: int,
    thresh: float,
    step: int = 2,
    window: float = 3.0,
    max_slope_change: float = 1.4,
) -> Stream:
    """Follow one ridge left and right of a seed using slope extrapolation."""
    s = Stream([float(seed_x)], [float(seed_y)])
    for direction in (+1, -1):
        x, y, slope, misses = seed_x, seed_y, 0.0, 0
        while True:
            nx = x + direction * step
            if nx < x_min or nx > x_max:
                break
            predicted = y + slope * direction * step
            cands = column_peaks(resp, int(nx), predicted - window, predicted + window, thresh)
            if not cands:
                misses += 1
                if misses > 3:
                    break
                x, y = nx, predicted
                continue
            misses = 0
            ny = min(cands, key=lambda c: abs(c - predicted))
            new_slope = (ny - y) / (direction * step)
            if abs(new_slope - slope) > max_slope_change:
                new_slope = slope + np.sign(new_slope - slope) * max_slope_change
                ny = y + new_slope * direction * step
            slope = new_slope
            x, y = nx, ny
            s.xs.append(float(x))
            s.ys.append(float(y))
    return s


def fmt(v: float) -> str:
    if abs(v) < 0.0005:
        return "0"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def spline_to_path(x: np.ndarray, y: np.ndarray, knot_step: float) -> str:
    """Emit a natural C2 cubic spline exactly as SVG cubic Beziers."""
    curve = CubicSpline(x, y, bc_type="natural")
    knots = list(np.arange(x[0], x[-1], knot_step))
    if not knots or knots[-1] != x[-1]:
        knots.append(float(x[-1]))
    knots = np.asarray(knots, dtype=float)
    parts = [f"M{fmt(knots[0])} {fmt(float(curve(knots[0])))}"]
    for a, b in zip(knots[:-1], knots[1:]):
        h = b - a
        ya, yb = float(curve(a)), float(curve(b))
        ma, mb = float(curve(a, 1)), float(curve(b, 1))
        parts.append(
            f"C{fmt(a + h / 3)} {fmt(ya + ma * h / 3)} "
            f"{fmt(b - h / 3)} {fmt(yb - mb * h / 3)} {fmt(b)} {fmt(yb)}"
        )


def report(name: str, streams: list[Stream]) -> None:
    print(f"--- {name}: {len(streams)} streams ---")
    for i, s in enumerate(streams):
        x, y = s.as_arrays()
        if x.size < 4:
            print(f"  {i:2d}: too short ({x.size} samples)")
            continue
        print(
            f"  {i:2d}: x {x[0]:6.1f}..{x[-1]:6.1f} ({x.size:3d} pts)"
            f"  y {y.min():6.1f}..{y.max():6.1f}  start {y[0]:6.1f} end {y[-1]:6.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default="airfoil - copy.png")
    ap.add_argument("--family", default="blue", choices=["blue", "red", "dark", "ink"])
    ap.add_argument("--seed-x", type=int, default=400)
    ap.add_argument("--band", default="300,430", help="y0,y1 to look for seeds in")
    ap.add_argument("--x-range", default="170,760")
    ap.add_argument("--thresh", type=float, default=10.0)
    ap.add_argument("--knot-step", type=float, default=16.0)
    ap.add_argument("--emit", action="store_true", help="print SVG path data")
    args = ap.parse_args()

    rgb = load_rgb(ROOT / args.png)
    resp = responses(rgb)[args.family]
    y0, y1 = (float(v) for v in args.band.split(","))
    x_min, x_max = (int(v) for v in args.x_range.split(","))

    seeds = column_peaks(resp, args.seed_x, y0, y1, args.thresh)
    print(f"{args.family}: {len(seeds)} seeds at x={args.seed_x} -> {[round(s, 1) for s in seeds]}")
    streams = [track(resp, args.seed_x, s, x_min, x_max, args.thresh) for s in seeds]
    report(f"{args.family} @x{args.seed_x}", streams)

    if args.emit:
        for s in streams:
            x, y = s.as_arrays()
            if x.size < 8:
                continue
            win = min(11, (x.size // 2) * 2 + 1)
            ys = savgol_filter(y, win, 3, mode="interp") if win >= 5 else y
            print(f'      <path d="{spline_to_path(x, ys, args.knot_step)}"/>')


if __name__ == "__main__":
    main()
