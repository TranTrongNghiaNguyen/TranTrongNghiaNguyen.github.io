"""Structural probes for the logo rebuild.

Answers the questions the plain MAE cannot:
  * is the render translated / scaled relative to the reference?
  * is the residual mostly stroke weight (ink density) instead of geometry?
  * where are the dark navy structures (outer rings, airfoil) in each image?
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import resvg_py
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PAIRS = {
    "emblem": (ROOT / "logo-emblem.svg", ROOT / "airfoil - copy.png"),
    "full": (ROOT / "logo.svg", ROOT / "airfoil.png"),
}


def load_reference(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGBA", img.size, (255, 255, 255, 255))
        flat.alpha_composite(img)
        img = flat
    return np.asarray(img.convert("RGB"), dtype=np.float64)


def render(svg_path: Path, size: tuple[int, int]) -> np.ndarray:
    png = resvg_py.svg_to_bytes(
        svg_path=str(svg_path),
        width=size[0],
        height=size[1],
        background="#ffffff",
        shape_rendering="geometric_precision",
        text_rendering="geometric_precision",
    )
    return np.asarray(Image.open(io.BytesIO(bytes(png))).convert("RGB"), dtype=np.float64)


def ink(arr: np.ndarray) -> np.ndarray:
    return np.clip((255.0 - arr.min(axis=2)) / 255.0, 0.0, 1.0)


def dark_mask(arr: np.ndarray, thresh: float = 150.0) -> np.ndarray:
    return arr.min(axis=2) < thresh


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (-1, -1, -1, -1)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def crossings(mask: np.ndarray, axis: str, index: int) -> list[int]:
    line = mask[index, :] if axis == "row" else mask[:, index]
    out, prev = [], False
    for i, v in enumerate(line):
        if v and not prev:
            out.append(i)
        prev = bool(v)
    return out


def best_shift(a: np.ndarray, b: np.ndarray, radius: int = 6) -> tuple[int, int, float]:
    """Integer (dx, dy) shift of b that minimises MAE against a."""
    best = (0, 0, float("inf"))
    h, w = a.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ys_a = slice(max(0, dy), h + min(0, dy))
            xs_a = slice(max(0, dx), w + min(0, dx))
            ys_b = slice(max(0, -dy), h + min(0, -dy))
            xs_b = slice(max(0, -dx), w + min(0, -dx))
            err = float(np.abs(a[ys_a, xs_a] - b[ys_b, xs_b]).mean())
            if err < best[2]:
                best = (dx, dy, err)
    return best


def best_gain(ref_ink: np.ndarray, out_ink: np.ndarray) -> tuple[float, float]:
    """Scalar k minimising |ref - k*out| in L2, plus resulting mean abs error."""
    denom = float((out_ink * out_ink).sum())
    k = float((ref_ink * out_ink).sum() / denom) if denom > 0 else 0.0
    return k, float(np.abs(ref_ink - k * out_ink).mean())


def report(name: str, svg: Path, png: Path, radius: int) -> None:
    ref = load_reference(png)
    out = render(svg, (ref.shape[1], ref.shape[0]))
    h, w = ref.shape[:2]
    ref_ink, out_ink = ink(ref), ink(out)

    print(f"=== {name} ({w}x{h}) ===")
    print(f"  ink mae            : {np.abs(ref_ink - out_ink).mean():.5f}")
    dx, dy, err = best_shift(ref_ink, out_ink, radius)
    print(f"  best integer shift : dx={dx:+d} dy={dy:+d} -> ink mae {err:.5f}")
    k, kerr = best_gain(ref_ink, out_ink)
    print(f"  best global gain   : k={k:.4f} -> ink mae {kerr:.5f}")

    for label, arr in (("ref", ref), ("out", out)):
        m = dark_mask(arr)
        x0, y0, x1, y1 = bbox(m)
        print(f"  {label} dark bbox      : x {x0}..{x1} (w {x1 - x0 + 1})  y {y0}..{y1} (h {y1 - y0 + 1})  n={int(m.sum())}")

    cy, cx = h // 2, w // 2
    for label, arr in (("ref", ref), ("out", out)):
        m = dark_mask(arr)
        print(f"  {label} dark row {cy:<4d}   : {crossings(m, 'row', cy)}")
    for label, arr in (("ref", ref), ("out", out)):
        m = dark_mask(arr)
        print(f"  {label} dark col {cx:<4d}   : {crossings(m, 'col', cx)}")

    # Ink mass centroid and spread give a scale/placement check on all strokes.
    for label, im in (("ref", ref_ink), ("out", out_ink)):
        tot = im.sum()
        ys, xs = np.mgrid[0:h, 0:w]
        mx = float((im * xs).sum() / tot)
        my = float((im * ys).sum() / tot)
        sx = float(np.sqrt((im * (xs - mx) ** 2).sum() / tot))
        sy = float(np.sqrt((im * (ys - my) ** 2).sum() / tot))
        print(f"  {label} ink centroid    : ({mx:8.3f}, {my:8.3f})  spread ({sx:7.3f}, {sy:7.3f})  mass {tot:,.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", choices=list(PAIRS))
    ap.add_argument("--radius", type=int, default=6)
    args = ap.parse_args()
    for name in args.targets or list(PAIRS):
        svg, png = PAIRS[name]
        report(name, svg, png, args.radius)


if __name__ == "__main__":
    main()
