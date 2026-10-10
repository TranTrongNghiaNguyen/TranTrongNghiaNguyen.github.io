"""Per-layer ablation for the emblem / logo SVGs.

Renders variants of the file that keep only selected top-level children of a
group (default #emblem), so each visual layer's ink can be attributed and
compared against the reference inside an arbitrary elliptical annulus.
"""

from __future__ import annotations

import argparse
import io
import math
import re
from pathlib import Path

import numpy as np
import resvg_py
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
NS = "{http://www.w3.org/2000/svg}"


def load_reference(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGBA", img.size, (255, 255, 255, 255))
        flat.alpha_composite(img)
        img = flat
    return np.asarray(img.convert("RGB"), dtype=np.float64)


def render_string(svg: str, size: tuple[int, int]) -> np.ndarray:
    png = resvg_py.svg_to_bytes(
        svg_string=svg,
        width=size[0],
        height=size[1],
        background="#ffffff",
        shape_rendering="geometric_precision",
        text_rendering="geometric_precision",
    )
    return np.asarray(Image.open(io.BytesIO(bytes(png))).convert("RGB"), dtype=np.float64)


def ink(arr: np.ndarray) -> np.ndarray:
    return np.clip((255.0 - arr.min(axis=2)) / 255.0, 0.0, 1.0)


def denoise(cov: np.ndarray, floor: float) -> np.ndarray:
    if floor <= 0:
        return cov
    return np.clip((cov - floor) / (1.0 - floor), 0.0, 1.0)


def top_level_spans(svg: str, group_id: str) -> list[tuple[str, int, int]]:
    """(tag, start, end) for each direct child element of <g id="group_id">."""
    m = re.search(rf'<g[^>]*id="{re.escape(group_id)}"[^>]*>', svg)
    if not m:
        raise SystemExit(f"group id={group_id!r} not found")
    i = m.end()
    depth = 0
    spans: list[tuple[str, int, int]] = []
    tag_re = re.compile(r"<(/?)([A-Za-z][\w:-]*)([^>]*?)(/?)>")
    start = None
    name = ""
    for t in tag_re.finditer(svg, i):
        closing, tname, _, selfclose = t.groups()
        if depth == 0 and closing == "/" and tname == "g":
            break  # closes the container group
        if closing == "/":
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((name, start, t.end()))
                start = None
            continue
        if selfclose == "/":
            if depth == 0:
                spans.append((tname, t.start(), t.end()))
            continue
        if depth == 0:
            start, name = t.start(), tname
        depth += 1
    return spans


def annulus(shape: tuple[int, int], cx: float, cy: float, r_in: float, r_out: float) -> np.ndarray:
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return (r >= r_in) & (r < r_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("svg", nargs="?", default="logo-emblem.svg")
    ap.add_argument("--png", default="airfoil - copy.png")
    ap.add_argument("--group", default="emblem")
    ap.add_argument("--center", default="418.3,424.4")
    ap.add_argument("--radii", default="0,250,340,368,378,390,600")
    args = ap.parse_args()

    svg_path = ROOT / args.svg
    svg = svg_path.read_text(encoding="utf-8")
    ref = load_reference(ROOT / args.png)
    size = (ref.shape[1], ref.shape[0])
    spans = top_level_spans(svg, args.group)
    cx, cy = (float(v) for v in args.center.split(","))
    radii = [float(v) for v in args.radii.split(",")]
    bands = [annulus(ref.shape[:2], cx, cy, radii[i], radii[i + 1]) for i in range(len(radii) - 1)]
    labels = [f"{int(radii[i])}-{int(radii[i + 1])}" for i in range(len(radii) - 1)]

    ref_ink = ink(ref)
    print(f"{svg_path.name}: {len(spans)} top-level children of #{args.group}")
    header = "  ".join(f"{l:>11}" for l in labels)
    print(f"  {'layer':<28} {'ink':>10}   {header}")
    print(f"  {'REFERENCE':<28} {ref_ink.sum():10,.1f}   " + "  ".join(f"{ref_ink[b].sum():11,.1f}" for b in bands))

    # Empty baseline: everything inside the group removed.
    blank = svg
    for _, s, e in reversed(spans):
        blank = blank[:s] + blank[e:]
    base = ink(render_string(blank, size))
    print(f"  {'(nothing)':<28} {base.sum():10,.1f}   " + "  ".join(f"{base[b].sum():11,.1f}" for b in bands))

    total = np.zeros_like(ref_ink)
    for idx, (tag, s, e) in enumerate(spans):
        only = svg
        for j, (_, s2, e2) in reversed(list(enumerate(spans))):
            if j != idx:
                only = only[:s2] + only[e2:]
        layer = ink(render_string(only, size)) - base
        layer = np.clip(layer, 0.0, 1.0)
        total += layer
        snippet = re.sub(r"\s+", " ", svg[s : s + 120])
        stroke = re.search(r'stroke="([^"]+)"', snippet)
        label = f"{idx}:{tag} {stroke.group(1) if stroke else ''}"
        print(f"  {label:<28} {layer.sum():10,.1f}   " + "  ".join(f"{layer[b].sum():11,.1f}" for b in bands))

    print(f"  {'SUM OF LAYERS':<28} {total.sum():10,.1f}   " + "  ".join(f"{total[b].sum():11,.1f}" for b in bands))


if __name__ == "__main__":
    main()
