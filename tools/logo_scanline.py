"""Compare 1-D scanline ink profiles between reference and render.

Prints, for a given row or column, the detected stroke peaks (position and
peak ink) for both images side by side. Mismatched peak counts mean missing or
extra geometry; matched peaks with different ink means a stroke-weight or
opacity error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from logo_probe import PAIRS, ink, load_reference, render


def peaks(profile: np.ndarray, min_ink: float) -> list[tuple[int, float]]:
    """Group contiguous runs above min_ink and report (centre, max ink)."""
    out: list[tuple[int, float]] = []
    lit = profile > min_ink
    i = 0
    n = len(profile)
    while i < n:
        if not lit[i]:
            i += 1
            continue
        j = i
        while j < n and lit[j]:
            j += 1
        seg = profile[i:j]
        centre = i + int(np.argmax(seg))
        out.append((centre, float(seg.max())))
        i = j
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=list(PAIRS))
    ap.add_argument("--row", type=int, action="append", default=[])
    ap.add_argument("--col", type=int, action="append", default=[])
    ap.add_argument("--min-ink", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    svg, png = PAIRS[args.target]
    ref = load_reference(png)
    out = render(svg, (ref.shape[1], ref.shape[0]))
    ref_ink, out_ink = ink(ref), ink(out)

    for axis, idxs in (("row", args.row), ("col", args.col)):
        for idx in idxs:
            a = ref_ink[idx, :] if axis == "row" else ref_ink[:, idx]
            b = out_ink[idx, :] if axis == "row" else out_ink[:, idx]
            pa, pb = peaks(a, args.min_ink), peaks(b, args.min_ink)
            print(f"--- {args.target} {axis} {idx}: ref {len(pa)} peaks, out {len(pb)} peaks ---")
            print(f"    ref sum {a.sum():8.3f}   out sum {b.sum():8.3f}   ratio {b.sum() / max(a.sum(), 1e-9):.3f}")
            print("    ref:", " ".join(f"{p}:{v:.2f}" for p, v in pa[: args.limit]))
            print("    out:", " ".join(f"{p}:{v:.2f}" for p, v in pb[: args.limit]))


if __name__ == "__main__":
    main()
