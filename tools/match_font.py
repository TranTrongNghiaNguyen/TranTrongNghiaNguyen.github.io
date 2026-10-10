"""Identify which installed font the reference wordmark was set in.

resvg picks its own substitute for "Times New Roman", and the reference's cap
height does not match it, so the family must be determined empirically. Each
candidate family is rendered into the wordmark band with its font-size solved
from the reference cap height and tracking solved from the reference width, then
scored on ink error. The best family/size/tracking triple wins.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from logo_compare import PAIRS, ink, load_reference
from logo_layers import ROOT, render_string
from text_metrics import block_stats
from tune_text import get_attr, set_attr, text_spans


def set_family(svg: str, span: tuple[int, int], family: str) -> str:
    s, e = span
    head = svg[s:e]
    if 'font-family="' in head:
        head = re.sub(r'font-family="[^"]*"', f'font-family="{family}"', head, count=1)
    else:
        head = head[:-1] + f' font-family="{family}">'
    return svg[:s] + head + svg[e:]


def measure(svg: str, size, y0, y1, x0, x1):
    return block_stats(ink(render_string(svg, size)), y0, y1, 0.25, x0, x1)


DEFAULT_FAMILIES = [
    "Times New Roman",
    "Georgia",
    "Garamond",
    "Book Antiqua",
    "Palatino Linotype",
    "Cambria",
    "Constantia",
    "Bookman Old Style",
    "Century Schoolbook",
    "Sylfaen",
    "Playfair Display",
    "Baskerville Old Face",
    "Bodoni MT",
    "Perpetua",
    "Cormorant Garamond",
    "EB Garamond",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", default="logo.svg")
    ap.add_argument("--target", default="full", choices=list(PAIRS))
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--band", default="380,500")
    ap.add_argument("--x", default="840,1696")
    ap.add_argument("--families", default="")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    svg_path = ROOT / args.svg
    svg0 = svg_path.read_text(encoding="utf-8")
    _, png = PAIRS[args.target]
    ref = load_reference(png)
    size = (ref.shape[1], ref.shape[0])
    ref_ink = ink(ref)
    y0, y1 = (int(v) for v in args.band.split(","))
    x0, x1 = (int(v) for v in args.x.split(","))
    target = block_stats(ref_ink, y0, y1, 0.25, x0, x1)
    print(f"reference block: w={target['w']} h={target['h']} top={target['top']} bottom={target['bottom']}")

    families = [f.strip() for f in args.families.split(";") if f.strip()] or DEFAULT_FAMILIES

    def score(s: str) -> float:
        return float(np.abs(ref_ink[y0:y1, x0:x1] - ink(render_string(s, size))[y0:y1, x0:x1]).mean())

    results = []
    for fam in families:
        svg = set_family(svg0, text_spans(svg0)[args.index], f"{fam}, serif")
        span = text_spans(svg)[args.index]
        fs = get_attr(svg[span[0] : span[1]], "font-size", 84.0)
        ls = get_attr(svg[span[0] : span[1]], "letter-spacing", 0.0)

        st = measure(svg, size, y0, y1, x0, x1)
        if not st:
            continue
        # Solve size from cap height, then tracking from block width.
        fs2 = round(fs * target["h"] / max(st["h"], 1), 2)
        svg = set_attr(svg, text_spans(svg)[args.index], "font-size", fs2)
        st = measure(svg, size, y0, y1, x0, x1)
        if not st:
            continue
        n = max(st["runs"] - 1, 1)
        ls2 = round(ls + (target["w"] - st["w"]) / n, 3)
        svg = set_attr(svg, text_spans(svg)[args.index], "letter-spacing", ls2)

        # Snap the baseline with a short local search.
        ty = get_attr(svg[text_spans(svg)[args.index][0] : text_spans(svg)[args.index][1]], "y", 0.0)
        best_local, best_svg, best_y = score(svg), svg, ty
        for dy in (-3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3):
            cand = set_attr(svg, text_spans(svg)[args.index], "y", ty + dy)
            v = score(cand)
            if v < best_local - 1e-7:
                best_local, best_svg, best_y = v, cand, ty + dy

        st = measure(best_svg, size, y0, y1, x0, x1)
        results.append((best_local, fam, fs2, ls2, best_y, st, best_svg))
        print(
            f"  {fam:<24} size={fs2:7.2f} track={ls2:7.3f} y={best_y:7.2f}"
            f"  h={st['h']:3d} w={st['w']:4d}  score={best_local:.6f}"
        )

    results.sort(key=lambda r: r[0])
    if not results:
        raise SystemExit("no family produced a measurable block")
    best = results[0]
    print(f"\nbest: {best[1]}  size={best[2]} track={best[3]} y={best[4]}  score={best[0]:.6f}")
    if args.apply:
        svg_path.write_text(best[6], encoding="utf-8")
        print(f"wrote {svg_path}")
    else:
        print("(dry run; pass --apply to write)")


if __name__ == "__main__":
    main()

