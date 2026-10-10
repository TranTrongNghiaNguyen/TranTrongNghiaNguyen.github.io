"""Render-in-the-loop tuning of the emblem's stroke weights and opacities.

The additive layer model (fit_layer_gains.py) is only approximate because the
strokes overlap and the references are lossy. This module instead evaluates the
real rasteriser for every candidate, and runs coordinate descent over a small
set of multiplicative knobs that map one-to-one onto attributes in the SVG:
stroke widths, group opacities and gradient stop opacities.

Nothing is written unless --apply is passed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from logo_compare import denoise
from logo_layers import ROOT, ink, load_reference, render_string, top_level_spans


def _num(s: str) -> float:
    return float(s)


def scale_attr(svg: str, start: int, end: int, attr: str, factor: float) -> str:
    """Multiply the first occurrence of attr="number" inside [start, end)."""
    pat = re.compile(rf'{re.escape(attr)}="([0-9.]+)"')
    m = pat.search(svg, start, end)
    if not m:
        raise KeyError(f"{attr} not found in span {start}:{end}")
    val = _num(m.group(1)) * factor
    return svg[: m.start()] + f'{attr}="{val:.5g}"' + svg[m.end() :]


def scale_stop_opacity(svg: str, grad_id: str, factor: float) -> str:
    m = re.search(rf'<linearGradient[^>]*id="{re.escape(grad_id)}"[^>]*>', svg)
    if not m:
        raise KeyError(f"gradient {grad_id} not found")
    close = svg.index("</linearGradient>", m.end())
    block = svg[m.end() : close]

    def rep(mo: re.Match) -> str:
        v = min(1.0, _num(mo.group(1)) * factor)
        return f'stop-opacity="{v:.5g}"'

    return svg[: m.end()] + re.sub(r'stop-opacity="([0-9.]+)"', rep, block) + svg[close:]


class Knob:
    """One tunable quantity, applied as a multiplicative factor."""

    def __init__(self, name: str, apply):
        self.name = name
        self.apply = apply


def build_knobs(svg: str, group: str) -> list[Knob]:
    spans = top_level_spans(svg, group)
    knobs: list[Knob] = []

    def span_width_knob(idx: int, label: str) -> Knob:
        def ap(s: str, f: float, idx=idx) -> str:
            sp = top_level_spans(s, group)
            tag, st, en = sp[idx]
            return scale_attr(s, st, en, "stroke-width", f)

        return Knob(f"{label}.stroke-width", ap)

    def span_opacity_knob(idx: int, label: str) -> Knob:
        def ap(s: str, f: float, idx=idx) -> str:
            sp = top_level_spans(s, group)
            tag, st, en = sp[idx]
            gt = s.index(">", st)
            head_end = gt + 1
            try:
                return scale_attr(s, st, head_end, "opacity", f)
            except KeyError:
                # No opacity yet: insert one, keeping any self-closing slash last.
                at = gt - 1 if s[gt - 1] == "/" else gt
                return s[:at] + f' opacity="{min(1.0, f):.5g}" ' + s[at:]

        return Knob(f"{label}.opacity", ap)


    labels = {}
    for idx, (tag, st, en) in enumerate(spans):
        head = svg[st : svg.index(">", st)]
        stroke = re.search(r'stroke="([^"]+)"', head)
        labels[idx] = f"{idx}:{tag}:{stroke.group(1) if stroke else 'fill'}"

    for idx, (tag, st, en) in enumerate(spans):
        head = svg[st : svg.index(">", st)]
        if 'stroke-width="' in head:
            knobs.append(span_width_knob(idx, labels[idx]))
        knobs.append(span_opacity_knob(idx, labels[idx]))

    for gid in ("fanRed", "fanBlue", "flowBlue"):
        if f'id="{gid}"' in svg:
            knobs.append(Knob(f"grad:{gid}.stop-opacity", lambda s, f, gid=gid: scale_stop_opacity(s, gid, f)))
    return knobs


def make_scorer(ref: np.ndarray, floor: float, size, sigma: float = 1.6, ratio_w: float = 0.4):
    """Blur-tolerant ink score.

    Plain ink MAE has a degenerate minimum: when strokes are a fraction of a
    pixel off, fading every layer towards white lowers the error, so naive
    descent bleaches the artwork. Blurring both images first makes the score
    insensitive to sub-pixel placement, so it responds to real weight errors,
    and a total-ink term pins the overall density to the reference.
    """
    ref_ink = denoise(ink(ref), floor)
    ref_blur = gaussian_filter(ref_ink, sigma)
    ref_total = float(ref_ink.sum())

    def score(svg: str) -> float:
        out_ink = denoise(ink(render_string(svg, size)), floor)
        base = float(np.abs(ref_blur - gaussian_filter(out_ink, sigma)).mean())
        ratio = float(out_ink.sum()) / max(ref_total, 1e-9)
        return base * (1.0 + ratio_w * abs(1.0 - ratio))

    return score



def coordinate_descent(svg: str, knobs, score, factors, rounds: int) -> tuple[str, float]:
    best = score(svg)
    print(f"  start ink_mae = {best:.6f}")
    for rnd in range(rounds):
        improved = False
        for knob in knobs:
            local_best, local_svg, local_f = best, None, 1.0
            for f in factors:
                try:
                    cand = knob.apply(svg, f)
                except KeyError:
                    continue
                val = score(cand)
                if val < local_best - 1e-7:
                    local_best, local_svg, local_f = val, cand, f
            if local_svg is not None:
                svg, best, improved = local_svg, local_best, True
                print(f"  round {rnd} {knob.name:<34} x{local_f:<5.3g} -> {best:.6f}")
        if not improved:
            print(f"  round {rnd}: no further improvement")
            break
    return svg, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("svg", nargs="?", default="logo-emblem.svg")
    ap.add_argument("--png", default="airfoil - copy.png")
    ap.add_argument("--group", default="emblem")
    ap.add_argument("--floor", type=float, default=0.012)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--factors", default="0.7,0.85,0.93,1.08,1.18,1.4")
    ap.add_argument("--apply", action="store_true", help="write the tuned SVG back")
    ap.add_argument("--out", default="", help="write the tuned SVG here instead of in place")
    ap.add_argument("--sigma", type=float, default=1.6)
    args = ap.parse_args()

    svg_path = ROOT / args.svg
    svg = svg_path.read_text(encoding="utf-8")
    ref = load_reference(ROOT / args.png)
    size = (ref.shape[1], ref.shape[0])

    knobs = build_knobs(svg, args.group)
    print(f"{svg_path.name}: {len(knobs)} knobs")
    score = make_scorer(ref, args.floor, size, args.sigma)
    factors = [float(v) for v in args.factors.split(",")]
    tuned, best = coordinate_descent(svg, knobs, score, factors, args.rounds)
    print(f"  final score = {best:.6f}")

    dest = (ROOT / args.out) if args.out else svg_path
    if (args.apply or args.out) and tuned != svg:
        dest.write_text(tuned, encoding="utf-8")
        print(f"  wrote {dest}")
    elif not (args.apply or args.out):
        print("  (dry run; pass --apply or --out to write)")



if __name__ == "__main__":
    main()
