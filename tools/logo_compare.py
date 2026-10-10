"""Render logo.svg / logo-emblem.svg with resvg and score them against the PNGs.

Reports per-pixel error metrics plus targeted structural diagnostics so the
vector reconstruction can be iterated until it matches the raster sources.
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
    "emblem-raster": (ROOT / "logo-emblem-raster-exact.svg", ROOT / "airfoil - copy.png"),
    "full-raster": (ROOT / "logo-raster-exact.svg", ROOT / "airfoil.png"),
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
    img = Image.open(io.BytesIO(bytes(png))).convert("RGB")
    return np.asarray(img, dtype=np.float64)


def ink(arr: np.ndarray) -> np.ndarray:
    """Ink coverage in [0,1]; 0 = paper white."""
    return np.clip((255.0 - arr.min(axis=2)) / 255.0, 0.0, 1.0)


def denoise(cov: np.ndarray, floor: float) -> np.ndarray:
    """Drop coverage below `floor` and rescale, so both images share a zero.

    The reference PNGs were saved lossily: their modal background level is 254,
    not 255, and a graded halo surrounds every stroke. That haze contributes
    ~2.3k ink units on the emblem, which no vector render can reproduce. Any
    metric used to steer the geometry must therefore ignore it.
    """
    if floor <= 0:
        return cov
    return np.clip((cov - floor) / (1.0 - floor), 0.0, 1.0)


def metrics(ref: np.ndarray, out: np.ndarray, floor: float = 0.0) -> dict[str, float]:
    diff = np.abs(ref - out)
    ref_ink = denoise(ink(ref), floor)
    out_ink = denoise(ink(out), floor)
    inter = np.minimum(ref_ink, out_ink).sum()
    union = np.maximum(ref_ink, out_ink).sum()
    return {
        "mae": float(diff.mean()),
        "rmse": float(np.sqrt((diff**2).mean())),
        "max_channel_err": float(diff.max()),
        "pct_pixels_gt8": float((diff.max(axis=2) > 8).mean() * 100.0),
        "pct_pixels_gt24": float((diff.max(axis=2) > 24).mean() * 100.0),
        "ink_ref": float(ref_ink.sum()),
        "ink_out": float(out_ink.sum()),
        "ink_ratio": float(out_ink.sum() / max(ref_ink.sum(), 1e-9)),
        "ink_iou": float(inter / max(union, 1e-9)),
        "ink_mae": float(np.abs(ref_ink - out_ink).mean()),
    }


def band_report(ref: np.ndarray, out: np.ndarray, rows: int = 6, cols: int = 6, floor: float = 0.0) -> str:
    ref_ink = denoise(ink(ref), floor)
    out_ink = denoise(ink(out), floor)

    h, w = ref_ink.shape
    lines = []
    for r in range(rows):
        cells = []
        for c in range(cols):
            ys = slice(r * h // rows, (r + 1) * h // rows)
            xs = slice(c * w // cols, (c + 1) * w // cols)
            a = ref_ink[ys, xs].sum()
            b = out_ink[ys, xs].sum()
            cells.append(f"{(b / a if a > 50 else 1.0):5.2f}")
        lines.append("  " + " ".join(cells))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", choices=list(PAIRS))
    ap.add_argument("--dump", action="store_true", help="write render + diff PNGs")
    ap.add_argument(
        "--floor",
        type=float,
        default=0.012,
        help="ignore ink coverage below this on both images (JPEG-style haze in the refs)",
    )
    ap.add_argument("--region", default="", help="x0,y0,x1,y1 to score only part of the canvas")
    args = ap.parse_args()

    for name in args.targets or list(PAIRS):
        svg_path, png_path = PAIRS[name]
        ref = load_reference(png_path)
        out = render(svg_path, (ref.shape[1], ref.shape[0]))
        if args.region:
            x0, y0, x1, y1 = (int(v) for v in args.region.split(","))
            ref = ref[y0:y1, x0:x1]
            out = out[y0:y1, x0:x1]
        m = metrics(ref, out, args.floor)

        print(f"=== {name}: {svg_path.name} vs {png_path.name} ({ref.shape[1]}x{ref.shape[0]}) ===")
        for k, v in m.items():
            print(f"  {k:>16}: {v:,.4f}")
        print("  ink ratio grid (render/reference):")
        print(band_report(ref, out, floor=args.floor))

        if args.dump:
            outdir = ROOT / "tools" / "_compare"
            outdir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(out.astype(np.uint8)).save(outdir / f"{name}_render.png")
            d = np.abs(ref - out).max(axis=2)
            heat = np.stack(
                [255 - np.clip(d * 3, 0, 255), 255 - np.clip(d * 3, 0, 255), np.full_like(d, 255)],
                axis=2,
            ).astype(np.uint8)
            Image.fromarray(heat).save(outdir / f"{name}_diff.png")

            # Signed residual: red = render too heavy, blue = render too light.
            ri = denoise(ink(ref), args.floor)
            oi = denoise(ink(out), args.floor)
            s = (oi - ri) * 6.0
            pos = np.clip(s, 0, 1)
            neg = np.clip(-s, 0, 1)
            signed = np.stack(
                [
                    255 - neg * 255,
                    255 - np.maximum(pos, neg) * 255,
                    255 - pos * 255,
                ],
                axis=2,
            ).astype(np.uint8)
            Image.fromarray(signed).save(outdir / f"{name}_signed.png")
            print(f"  wrote {outdir / (name + '_render.png')}, {name}_diff.png, {name}_signed.png")



if __name__ == "__main__":
    main()
