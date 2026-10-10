"""Solve for the per-layer ink gains that best match the reference.

At the low coverages in this artwork, stacked strokes composite close to
additively in ink space, so the reference can be modelled as
    ref ~= sum_i k_i * L_i
where L_i is the isolated ink map of top-level layer i. Non-negative least
squares over the k_i says exactly how much each layer must be scaled up or
down, which maps directly onto its opacity / stroke-width.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from logo_compare import denoise
from logo_layers import ROOT, ink, load_reference, render_string, top_level_spans


def layer_maps(svg: str, spans, size, floor):
    blank = svg
    for _, s, e in reversed(spans):
        blank = blank[:s] + blank[e:]
    base = denoise(ink(render_string(blank, size)), floor)
    maps = []
    for idx in range(len(spans)):
        only = svg
        for j, (_, s2, e2) in reversed(list(enumerate(spans))):
            if j != idx:
                only = only[:s2] + only[e2:]
        m = denoise(ink(render_string(only, size)), floor) - base
        maps.append(np.clip(m, 0.0, 1.0))
    return base, maps


def nnls(a: np.ndarray, b: np.ndarray, iters: int = 400) -> np.ndarray:
    """Projected-gradient NNLS; a is (n, k) normal-equation friendly."""
    ata = a.T @ a
    atb = a.T @ b
    k = np.ones(a.shape[1])
    step = 1.0 / (np.linalg.eigvalsh(ata).max() + 1e-12)
    for _ in range(iters):
        k = np.maximum(0.0, k - step * (ata @ k - atb))
    return k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("svg", nargs="?", default="logo-emblem.svg")
    ap.add_argument("--png", default="airfoil - copy.png")
    ap.add_argument("--group", default="emblem")
    ap.add_argument("--floor", type=float, default=0.012)
    args = ap.parse_args()

    svg_path = ROOT / args.svg
    svg = svg_path.read_text(encoding="utf-8")
    ref = load_reference(ROOT / args.png)
    size = (ref.shape[1], ref.shape[0])
    ref_ink = denoise(ink(ref), args.floor)

    spans = top_level_spans(svg, args.group)
    base, maps = layer_maps(svg, spans, size, args.floor)

    # Only fit where some layer actually paints, else the huge white area
    # dominates and drives every gain to zero.
    stack = np.stack(maps, axis=0)
    active = stack.max(axis=0) > 0.004
    a = np.stack([m[active] for m in maps], axis=1)
    b = (ref_ink - base)[active]

    print(f"{svg_path.name}: fitting {len(maps)} layers over {int(active.sum()):,} active px")
    before = np.abs(b - a.sum(axis=1)).mean()
    k = nnls(a, b)
    after = np.abs(b - a @ k).mean()
    print(f"  active-pixel ink MAE: {before:.5f} -> {after:.5f}")

    for idx, (tag, s, _e) in enumerate(spans):
        snippet = re.sub(r"\s+", " ", svg[s : s + 200])
        stroke = re.search(r'stroke="([^"]+)"', snippet)
        sw = re.search(r'stroke-width="([^"]+)"', snippet)
        op = re.search(r'opacity="([^"]+)"', snippet)
        bits = []
        if stroke:
            bits.append(stroke.group(1))
        if sw:
            bits.append(f"w={sw.group(1)}")
        if op:
            bits.append(f"op={op.group(1)}")
        print(f"  {idx}:{tag:<8} {' '.join(bits):<34} gain {k[idx]:6.3f}")


if __name__ == "__main__":
    main()
