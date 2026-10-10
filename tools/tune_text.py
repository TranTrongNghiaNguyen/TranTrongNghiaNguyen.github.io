"""Fit each <text> element's size, position and tracking to the reference.

resvg substitutes its own faces for "Times New Roman" / "Arial", so the
nominal font-size in the SVG does not reproduce the reference's cap height.
Rather than guess, each text element gets its font-size, letter-spacing, x and
y solved numerically against the raster inside its own band.

Order matters: size first (it moves everything), then tracking, then position.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from logo_compare import PAIRS, ink, load_reference
from logo_layers import ROOT, render_string
from text_metrics import block_stats

TEXT_RE = re.compile(r"<text\b[^>]*>")


def text_spans(svg: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in TEXT_RE.finditer(svg)]


def get_attr(head: str, name: str, default: float) -> float:
    m = re.search(rf'{re.escape(name)}="(-?[0-9.]+)"', head)
    return float(m.group(1)) if m else default


def set_attr(svg: str, span: tuple[int, int], name: str, value: float) -> str:
    s, e = span
    head = svg[s:e]
    pat = re.compile(rf'{re.escape(name)}="(-?[0-9.]+)"')
    if pat.search(head):
        new = pat.sub(f'{name}="{value:.4g}"', head, count=1)
    else:
        new = head[:-1] + f' {name}="{value:.4g}">'
    return svg[:s] + new + svg[e:]


def band_score(ref_ink, out_ink, y0, y1, x0, x1) -> float:
    a = ref_ink[y0:y1, x0:x1]
    b = out_ink[y0:y1, x0:x1]
    return float(np.abs(a - b).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", default="logo.svg")
    ap.add_argument("--target", default="full", choices=list(PAIRS))
    ap.add_argument(
        "--block",
        action="append",
        required=True,
        help="index:y0,y1 for the <text> element at that index (0-based, document order)",
    )
    ap.add_argument("--x", default="840,1696")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    svg_path = ROOT / args.svg
    svg = svg_path.read_text(encoding="utf-8")
    _, png = PAIRS[args.target]
    ref = load_reference(png)
    size = (ref.shape[1], ref.shape[0])
    ref_ink = ink(ref)
    x0, x1 = (int(v) for v in args.x.split(","))

    def score_all(s: str, y0: int, y1: int) -> float:
        return band_score(ref_ink, ink(render_string(s, size)), y0, y1, x0, x1)

    for spec in args.block:
        idx_s, band = spec.split(":")
        idx = int(idx_s)
        y0, y1 = (int(v) for v in band.split(","))
        spans = text_spans(svg)
        head = svg[spans[idx][0] : spans[idx][1]]
        fs = get_attr(head, "font-size", 0.0)
        ls = get_attr(head, "letter-spacing", 0.0)
        tx = get_attr(head, "x", 0.0)
        ty = get_attr(head, "y", 0.0)

        # Reference geometry for this band drives the initial analytic guess.
        a = block_stats(ref_ink, y0, y1, 0.25, x0, x1)
        b = block_stats(ink(render_string(svg, size)), y0, y1, 0.25, x0, x1)
        print(f"--- text[{idx}] band {y0}..{y1}  font-size={fs} letter-spacing={ls} x={tx} y={ty}")
        print(f"    ref w={a['w']} h={a['h']} top={a['top']}   out w={b['w']} h={b['h']} top={b['top']}")

        best = score_all(svg, y0, y1)
        # 1+2) font-size and letter-spacing must move together: enlarging the
        # glyphs also widens the block, so tracking has to absorb the change.
        # Search size candidates and, for each, solve tracking from the width.
        guess = fs * (a["h"] / max(b["h"], 1))
        cands = sorted({round(v, 2) for v in (fs, guess, guess * 0.96, guess * 0.98, guess * 1.02, guess * 1.04)})
        for cand in cands:
            trial = set_attr(svg, text_spans(svg)[idx], "font-size", cand)
            st = block_stats(ink(render_string(trial, size)), y0, y1, 0.25, x0, x1)
            if not st:
                continue
            nglyph = max(st["runs"] - 1, 1)
            ls_guess = ls + (a["w"] - st["w"]) / nglyph
            # Tracking may legitimately go negative once the glyphs are larger.
            for lc in sorted({round(v, 3) for v in (ls, ls_guess, ls_guess - 0.4, ls_guess + 0.4)}):
                t2 = set_attr(trial, text_spans(trial)[idx], "letter-spacing", lc)
                val = score_all(t2, y0, y1)
                if val < best - 1e-7:
                    best, svg, fs, ls = val, t2, cand, lc
        print(f"    font-size -> {fs:.4g}  letter-spacing -> {ls:.4g}  score {best:.6f}")



        # 3) baseline and left edge, integral and half-pixel steps.
        for name, cur_val in (("y", ty), ("x", tx)):
            local = cur_val
            for step in (4, 2, 1, 0.5):
                improved = True
                while improved:
                    improved = False
                    for cand in (local - step, local + step):
                        trial = set_attr(svg, text_spans(svg)[idx], name, cand)
                        val = score_all(trial, y0, y1)
                        if val < best - 1e-7:
                            best, svg, local, improved = val, trial, cand, True
            print(f"    {name} -> {local:.4g}  score {best:.6f}")

    if args.apply:
        svg_path.write_text(svg, encoding="utf-8")
        print(f"wrote {svg_path}")
    else:
        print("(dry run; pass --apply to write)")


if __name__ == "__main__":
    main()
