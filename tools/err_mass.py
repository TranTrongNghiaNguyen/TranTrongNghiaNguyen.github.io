"""Score an arbitrary SVG against a reference PNG and report where the error mass sits.

Unlike logo_compare (which reports ratios per grid cell), this ranks grid cells by
their share of the total absolute ink error, so tuning effort can be aimed at the
cells that actually dominate the score. Any SVG path may be passed, which lets
backup copies (logo.svg.stage2, *.bak) be scored without editing PAIRS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from logo_compare import ROOT, PAIRS, denoise, ink, load_reference, render


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("svg", help="path to any svg (absolute, or relative to repo root)")
    ap.add_argument("--ref", default="airfoil - copy.png")
    ap.add_argument("--floor", type=float, default=0.012)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--top", type=int, default=8, help="how many worst cells to list")
    ap.add_argument("--region", default="", help="x0,y0,x1,y1 to score only part of the canvas")
    args = ap.parse_args()

    def resolve(name: str) -> Path:
        """Accept an absolute path, one relative to the cwd, or one relative to the repo root."""
        p = Path(name)
        if p.is_absolute():
            return p
        return p.resolve() if p.exists() else (ROOT / name)

    svg_path = resolve(args.svg)
    ref_path = resolve(args.ref)

    ref = load_reference(ref_path)
    out = render(svg_path, (ref.shape[1], ref.shape[0]))
    if args.region:
        x0, y0, x1, y1 = (int(v) for v in args.region.split(","))
        ref, out = ref[y0:y1, x0:x1], out[y0:y1, x0:x1]

    ri = denoise(ink(ref), args.floor)
    oi = denoise(ink(out), args.floor)
    err = np.abs(ri - oi)
    total = err.sum()

    print(f"=== {svg_path.name} vs {ref_path.name} ({ri.shape[1]}x{ri.shape[0]}) ===")
    print(f"  ink_mae   : {err.mean():.6f}")
    print(f"  ink_ratio : {oi.sum() / max(ri.sum(), 1e-9):.4f}")
    print(f"  err_total : {total:,.1f}")

    h, w = ri.shape
    cells = []
    for r in range(args.rows):
        for c in range(args.cols):
            ys = slice(r * h // args.rows, (r + 1) * h // args.rows)
            xs = slice(c * w // args.cols, (c + 1) * w // args.cols)
            e = float(err[ys, xs].sum())
            a = float(ri[ys, xs].sum())
            b = float(oi[ys, xs].sum())
            cells.append((e, r, c, a, b, xs.start, xs.stop, ys.start, ys.stop))

    print("  error share grid (% of total abs ink error):")
    for r in range(args.rows):
        row = [f"{100.0 * cells[r * args.cols + c][0] / max(total, 1e-9):5.1f}" for c in range(args.cols)]
        print("   " + " ".join(row))

    print(f"  worst {args.top} cells:")
    for e, r, c, a, b, xa, xb, ya, yb in sorted(cells, reverse=True)[: args.top]:
        sign = "render heavy" if b > a else "render light"
        print(
            f"   r{r}c{c}  region {xa},{ya},{xb},{yb}  err={e:8.1f} ({100.0 * e / max(total, 1e-9):4.1f}%)"
            f"  ref={a:8.1f} out={b:8.1f} ratio={b / max(a, 1e-9):5.2f}  {sign}"
        )


if __name__ == "__main__":
    main()
