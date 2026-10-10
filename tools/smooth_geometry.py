"""Build a visually smooth candidate for the emblem's flow and airfoil tail.

This is intentionally geometry-driven rather than a literal trace of every
compressed raster wiggle. Blue streamlines are C2 splines through measured band
centres; red rays are offsets from the blue envelope that open gradually; and
the airfoil thickness tapers to zero with a shared tangent at a mitered trailing
edge. The script writes candidates by default and only touches the masters with
--apply.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from logo_layers import ROOT, top_level_spans


def fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def spline_path(x: np.ndarray, y: np.ndarray, reverse: bool = False) -> str:
    """Natural C2 x->y spline serialized exactly as cubic Beziers."""
    c = CubicSpline(x, y, bc_type="natural")
    pairs = list(zip(x[:-1], x[1:]))
    if reverse:
        pairs.reverse()
        parts = [f"L{fmt(x[-1])} {fmt(float(c(x[-1])))}"]
        for a, b in pairs:
            h = b - a
            parts.append(
                f"C{fmt(b-h/3)} {fmt(float(c(b)-c(b,1)*h/3))} "
                f"{fmt(a+h/3)} {fmt(float(c(a)+c(a,1)*h/3))} "
                f"{fmt(a)} {fmt(float(c(a)))}"
            )
    else:
        parts = [f"M{fmt(x[0])} {fmt(float(c(x[0])))}"]
        for a, b in pairs:
            h = b - a
            parts.append(
                f"C{fmt(a+h/3)} {fmt(float(c(a)+c(a,1)*h/3))} "
                f"{fmt(b-h/3)} {fmt(float(c(b)-c(b,1)*h/3))} "
                f"{fmt(b)} {fmt(float(c(b)))}"
            )
    return " ".join(parts)


# Measured centres of the outermost and innermost visible blue bands. Tail
# points are regularised so neighbouring lines converge monotonically without
# crossing or acquiring the PCHIP curvature jumps in the previous version.
FLOW_X = np.array([164, 180, 220, 250, 300, 350, 400, 450, 500, 550, 600, 650, 680, 710], float)
UP_OUT = np.array([421, 397, 365, 349, 332, 322, 319, 325, 340, 358, 380, 402, 421, 439], float)
UP_IN = np.array([421, 405, 381, 372, 364, 363, 368, 377, 389, 401, 413, 429, 436, 443], float)
LO_IN = np.array([421, 441, 462, 471, 478, 479, 475, 468, 460, 454, 450, 439, 441, 446], float)
LO_OUT = np.array([421, 449, 482, 497, 514, 524, 526, 522, 510, 496, 474, 458, 453, 450], float)


def flow_block() -> str:
    rows = []
    opacities = [0.90, 0.85, 0.80, 0.74, 0.67, 0.59, 0.51, 0.43]
    for i, blend in enumerate(np.linspace(0, 1, 8)):
        up = (1-blend)*UP_OUT + blend*UP_IN
        lo = (1-blend)*LO_IN + blend*LO_OUT
        rows.append(f'      <path d="{spline_path(FLOW_X, up)}" opacity="{opacities[i]:.2f}"/>')
        rows.append(f'      <path d="{spline_path(FLOW_X, lo)}" opacity="{opacities[i]:.2f}"/>')
    return (
        '    <!-- C2-continuous aerodynamic streamlines; regularised from PNG ridge centres. -->\n'
        '    <g fill="none" stroke="url(#flowBlue)" stroke-width="1.2638" '
        'vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round">\n'
        + "\n".join(rows) + '\n    </g>'
    )



# The red fan follows the blue outer envelope first, then opens gradually. A
# smoothstep opening has zero relative slope at the hub, so red and blue cannot
# cut through each other near the nose.
FAN_X = np.array([164, 180, 220, 250, 300, 350, 400, 450, 500, 550, 586], float)
UP_ENV = CubicSpline(FLOW_X, UP_OUT, bc_type="natural")(FAN_X)
LO_ENV = CubicSpline(FLOW_X, LO_OUT, bc_type="natural")(FAN_X)


def fan_path(sign: int, level: int) -> str:
    """One ray: level 0 hugs the blue envelope, level 7 fans out furthest."""
    end_x = 418.0 + 24.0 * (7 - level)
    x = FAN_X[FAN_X <= end_x]
    if x[-1] < end_x:
        x = np.append(x, end_x)
    env = CubicSpline(FAN_X, UP_ENV if sign < 0 else LO_ENV, bc_type="natural")(x)
    t = (x - x[0]) / (x[-1] - x[0])
    max_offset = 48.0 + 15.0 * level
    y = env + sign * max_offset * (t * t * (3.0 - 2.0 * t))
    return spline_path(x, y)


def fan_block() -> str:
    op = [0.55, 0.58, 0.62, 0.67, 0.73, 0.80, 0.88, 1.0]
    rows = [f'      <path d="{fan_path(-1, lv)}" opacity="{op[lv]:.2f}"/>' for lv in reversed(range(8))]
    rows += [f'      <path d="{fan_path(+1, lv)}" opacity="{op[lv]:.2f}"/>' for lv in range(8)]
    return (
        '    <!-- Red fan rays hug the blue envelope, then open with zero hub-relative slope. -->\n'
        '    <g fill="none" stroke="url(#fanRed)" stroke-width="0.9143" '
        'vector-effect="non-scaling-stroke" opacity="0.9384">\n'
        + "\n".join(rows) + "\n    </g>"
    )


BODY_X = np.array(
    [164, 170, 180, 200, 240, 280, 320, 360, 400, 450, 500, 550, 600, 640, 670, 685, 700, 715, 730, 735], float
)
BODY_CAMBER = np.array(
    [421, 422, 422, 421, 420.5, 420, 420, 420.5, 421, 422, 424, 427, 431, 435.5, 439.5, 441.5, 443, 444.5, 446, 447],
    float,
)
BODY_HALF = np.array(
    [0, 13, 18, 28, 39.5, 45, 46, 43.5, 39, 31, 22, 14, 7, 3.5, 1.5, 0.7, 0.35, 0.14, 0.03, 0], float
)


def body_block(stroke_width: str, opacity: str) -> str:
    """Closed airfoil whose surfaces meet at a zero-thickness trailing edge."""
    d = (
        spline_path(BODY_X, BODY_CAMBER - BODY_HALF)
        + " "
        + spline_path(BODY_X, BODY_CAMBER + BODY_HALF, reverse=True)
        + " Z"
    )
    return (
        '    <!-- Smooth airfoil body; upper/lower surfaces share a zero-thickness trailing edge. -->\n'
        f'    <path d="{d}" fill="#fff" stroke="#090e4b" stroke-width="{stroke_width}" '
        f'vector-effect="non-scaling-stroke" stroke-linejoin="miter" stroke-miterlimit="12" opacity="{opacity}"/>'
    )


def replace_layers(svg: str) -> str:
    """Swap the fan, flow and body children of #emblem for the smooth versions."""
    spans = top_level_spans(svg, "emblem")
    if len(spans) != 8:
        raise ValueError(f"expected 8 #emblem children, found {len(spans)}")
    replacements = {4: fan_block(), 6: flow_block(), 7: body_block("2.5724", "0.98")}
    for idx in sorted(replacements, reverse=True):  # back to front keeps offsets valid
        _tag, a, b = spans[idx]
        svg = svg[:a] + replacements[idx] + svg[b:]
    return svg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the masters instead of candidates")
    args = ap.parse_args()
    outdir = ROOT / "tools" / "_compare"
    outdir.mkdir(parents=True, exist_ok=True)
    for name in ("logo-emblem.svg", "logo.svg"):
        src = ROOT / name
        out = replace_layers(src.read_text(encoding="utf-8"))
        dest = src if args.apply else outdir / f"{src.stem}-smooth.svg"
        dest.write_text(out, encoding="utf-8")
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
