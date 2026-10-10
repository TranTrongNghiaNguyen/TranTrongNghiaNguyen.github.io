"""Fit the individual stop opacities of one gradient against the reference.

tune_emblem.py can only scale *all* of a gradient's stops by a common factor, so
it cannot correct a gradient whose falloff is wrong. The red fan is exactly that
case: its warm ink matches the reference near the hub but dies away far too
quickly towards the far end, so the error is in the shape of the ramp, not its
overall level.

Each stop opacity is treated as an independent knob and fitted by coordinate
descent on the same blur-tolerant ink score used by tune_emblem (see the note
there on why plain ink MAE cannot be used). Optionally the score is restricted
to a colour class, so tuning the red fan is not swamped by the navy rings.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from logo_compare import denoise
from logo_ink_classes import classify
from logo_layers import ROOT, ink, load_reference, render_string


def stop_spans(svg: str, grad_id: str) -> list[tuple[int, int, float]]:
    """(start, end, opacity) of every stop-opacity attribute in the gradient."""
    m = re.search(rf'<linearGradient[^>]*id="{re.escape(grad_id)}"[^>]*>', svg)
    if not m:
        raise SystemExit(f"gradient {grad_id!r} not found")
    close = svg.index("</linearGradient>", m.end())
    out = []
    for s in re.finditer(r'stop-opacity="([0-9.]+)"', svg[m.end() : close]):
        out.append((m.end() + s.start(1), m.end() + s.end(1), float(s.group(1))))
    if not out:
        raise SystemExit(f"gradient {grad_id!r} has no stop-opacity attributes")
    return out


def set_stop(svg: str, grad_id: str, idx: int, value: float) -> str:
    spans = stop_spans(svg, grad_id)
    a, b, _ = spans[idx]
    return svg[:a] + f"{min(1.0, max(0.0, value)):.5g}" + svg[b:]


def make_scorer(ref: np.ndarray, floor: float, size, sigma: float, cls: str, ratio_w: float):
    ref_ink = denoise(ink(ref), floor)
    if cls:
        mask = classify(ref, floor)[cls]
        # Dilate the class mask so the render's own halo is scored too, else
        # thinning a stroke can hide error outside the reference's own pixels.
        mask = gaussian_filter(mask.astype(float), sigma) > 0.02
    else:
        mask = np.ones(ref_ink.shape, dtype=bool)
    ref_blur = gaussian_filter(ref_ink, sigma)
    ref_total = float(ref_ink[mask].sum())

    def score(svg: str) -> float:
        out_ink = denoise(ink(render_string(svg, size)), floor)
        base = float(np.abs(ref_blur - gaussian_filter(out_ink, sigma))[mask].mean())
        ratio = float(out_ink[mask].sum()) / max(ref_total, 1e-9)
        return base * (1.0 + ratio_w * abs(1.0 - ratio))

    return score, int(mask.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("svg", nargs="?", default="logo-emblem.svg")
    ap.add_argument("--png", default="airfoil - copy.png")
    ap.add_argument("--grad", required=True, help="gradient id whose stops are fitted")
    ap.add_argument("--class", dest="cls", default="", choices=["", "dark", "warm", "cool", "neutral"])
    ap.add_argument("--floor", type=float, default=0.012)
    ap.add_argument("--sigma", type=float, default=1.6)
    ap.add_argument("--ratio-w", type=float, default=0.4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--factors", default="0.6,0.8,0.9,1.1,1.25,1.6,2.2,3.0")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    svg_path = ROOT / args.svg
    svg = svg_path.read_text(encoding="utf-8")
    ref = load_reference(ROOT / args.png)
    size = (ref.shape[1], ref.shape[0])

    score, npx = make_scorer(ref, args.floor, size, args.sigma, args.cls, args.ratio_w)
    stops = stop_spans(svg, args.grad)
    factors = [float(v) for v in args.factors.split(",")]
    best = score(svg)
    print(f"{svg_path.name}: gradient {args.grad} has {len(stops)} stops; class={args.cls or 'all'} ({npx:,} px)")
    print(f"  start score = {best:.6f}  stops = {[s[2] for s in stops]}")

    for rnd in range(args.rounds):
        improved = False
        for idx in range(len(stops)):
            cur = stop_spans(svg, args.grad)[idx][2]
            local_best, local_svg, local_val = best, None, cur
            for f in factors:
                cand = set_stop(svg, args.grad, idx, cur * f)
                v = score(cand)
                if v < local_best - 1e-7:
                    local_best, local_svg, local_val = v, cand, cur * f
            if local_svg is not None:
                svg, best, improved = local_svg, local_best, True
                print(f"  round {rnd} stop[{idx}] {cur:.4g} -> {min(1.0, local_val):.4g}  score {best:.6f}")
        if not improved:
            print(f"  round {rnd}: no further improvement")
            break

    print(f"  final score = {best:.6f}  stops = {[s[2] for s in stop_spans(svg, args.grad)]}")
    dest = (ROOT / args.out) if args.out else svg_path
    if args.apply or args.out:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(svg, encoding="utf-8")
        print(f"  wrote {dest}")
    else:
        print("  (dry run; pass --apply or --out to write)")


if __name__ == "__main__":
    main()
