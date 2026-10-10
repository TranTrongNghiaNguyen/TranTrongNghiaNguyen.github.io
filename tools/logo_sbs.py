"""Dump a side-by-side crop of reference vs render for visual inspection.

Writes `_compare/{name}_{tag}_sbs.png`: reference on top, render below,
separated by a thin rule, so glyph shapes and positions can be compared by eye.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from logo_compare import PAIRS, load_reference, render

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=list(PAIRS))
    ap.add_argument("--region", required=True, help="x0,y0,x1,y1")
    ap.add_argument("--tag", default="crop")
    ap.add_argument("--scale", type=int, default=1)
    args = ap.parse_args()

    svg, png = PAIRS[args.target]
    ref = load_reference(png)
    out = render(svg, (ref.shape[1], ref.shape[0]))
    x0, y0, x1, y1 = (int(v) for v in args.region.split(","))
    a = ref[y0:y1, x0:x1].astype(np.uint8)
    b = out[y0:y1, x0:x1].astype(np.uint8)

    rule = np.full((3, a.shape[1], 3), 200, dtype=np.uint8)
    stacked = np.concatenate([a, rule, b], axis=0)
    img = Image.fromarray(stacked)
    if args.scale != 1:
        img = img.resize((img.width * args.scale, img.height * args.scale), Image.NEAREST)
    outdir = ROOT / "tools" / "_compare"
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"{args.target}_{args.tag}_sbs.png"
    img.save(dest)
    print(f"wrote {dest} ({img.width}x{img.height}); top = reference, bottom = render")


if __name__ == "__main__":
    main()
