"""Measure text-block geometry (bbox, baseline, advance) in ref vs render.

For each requested horizontal band the dark ink is projected onto the axes to
recover the block's left/right edges, cap height and baseline, plus the column
gaps that separate glyphs. Comparing those numbers tells whether the wordmark
needs a shift, a font-size change, or a letter-spacing change.
"""

from __future__ import annotations

import argparse

import numpy as np

from logo_compare import PAIRS, ink, load_reference, render


def block_stats(cov: np.ndarray, y0: int, y1: int, thresh: float = 0.25, x0c: int = 0, x1c: int | None = None) -> dict:
    band = cov[y0:y1, x0c:x1c]
    m = band > thresh
    cols = m.any(axis=0)
    rows = m.any(axis=1)
    if not cols.any():
        return {}
    xs = np.nonzero(cols)[0]
    ys = np.nonzero(rows)[0]
    # Glyph runs along x, so letter pitch can be recovered.
    runs = []
    start = None
    for i, v in enumerate(cols):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(cols) - 1))
    centres = [0.5 * (a + b) for a, b in runs]
    pitch = float(np.mean(np.diff(centres))) if len(centres) > 1 else float("nan")
    return {
        "x0": int(x0c + xs[0]),
        "x1": int(x0c + xs[-1]),
        "w": int(xs[-1] - xs[0] + 1),
        "top": int(y0 + ys[0]),
        "bottom": int(y0 + ys[-1]),
        "h": int(ys[-1] - ys[0] + 1),
        "runs": len(runs),
        "pitch": pitch,
        "ink": float(band[m].sum()),
    }



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=list(PAIRS))
    ap.add_argument("--band", action="append", required=True, help="y0,y1 (repeatable)")
    ap.add_argument("--thresh", type=float, default=0.25)
    ap.add_argument("--x", default="", help="x0,x1 window (defaults to full width)")
    args = ap.parse_args()

    svg, png = PAIRS[args.target]
    ref = load_reference(png)
    out = render(svg, (ref.shape[1], ref.shape[0]))
    rc, oc = ink(ref), ink(out)
    if args.x:
        xa, xb = (int(v) for v in args.x.split(","))
    else:
        xa, xb = 0, ref.shape[1]

    for spec in args.band:
        y0, y1 = (int(v) for v in spec.split(","))
        a = block_stats(rc, y0, y1, args.thresh, xa, xb)
        b = block_stats(oc, y0, y1, args.thresh, xa, xb)

        print(f"--- band y {y0}..{y1} ---")
        if not a or not b:
            print(f"    ref={a} out={b}")
            continue
        keys = ["x0", "x1", "w", "top", "bottom", "h", "runs", "pitch", "ink"]
        print("      " + "".join(f"{k:>10}" for k in keys))
        for label, s in (("ref", a), ("out", b)):
            print(f"  {label:>4}" + "".join(f"{s[k]:>10.2f}" if isinstance(s[k], float) else f"{s[k]:>10}" for k in keys))
        print(
            f"   diff  dx0={b['x0'] - a['x0']:+d} dx1={b['x1'] - a['x1']:+d}"
            f" dtop={b['top'] - a['top']:+d} dbottom={b['bottom'] - a['bottom']:+d}"
            f"  width x{b['w'] / a['w']:.4f}  height x{b['h'] / a['h']:.4f}"
        )


if __name__ == "__main__":
    main()
