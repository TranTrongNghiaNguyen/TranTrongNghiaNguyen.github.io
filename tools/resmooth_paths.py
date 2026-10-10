"""Remove curvature kinks from existing emblem paths without moving them.

The flow and fan paths in the masters are accurate in position - they were fitted
to the raster - but they were emitted as one cubic every few pixels from a C1
(PCHIP) interpolant, so curvature jumps at every knot and the rendered lines
visibly kink. Rebuilding them from hand-entered guides loses accuracy (measured:
flow ink_mae 0.0253 -> 0.0277).

So this module keeps the measured samples and only changes the representation:
each path is evaluated densely, refitted with a natural C2 spline over sparse
knots, and re-emitted using the spline's own Hermite tangents. Sparse knots act
as the low-pass filter; C2 continuity removes the kinks by construction.

For the red fan an extra shaping step is applied: each ray is blended out of the
blue outer envelope with a smoothstep weight, so it leaves the hub tangent to the
streamlines - hugging them - and only opens out downstream, instead of departing
as a straight chord that cuts across them. Ray endpoints are preserved, so the
fan still lands where the reference puts it.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
from scipy.interpolate import CubicSpline

from logo_layers import ROOT, top_level_spans

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_cubics(d: str) -> tuple[tuple[float, float], list[list[float]]]:
    """Return the M point and every C segment's six numbers."""
    m = re.match(r"\s*M\s*(-?[\d.]+)[ ,]+(-?[\d.]+)", d)
    if not m:
        raise ValueError("path does not start with M")
    start = (float(m.group(1)), float(m.group(2)))
    segs = []
    for c in re.finditer(r"C([^CMZLz]+)", d):
        nums = [float(v) for v in NUM.findall(c.group(1))]
        for i in range(0, len(nums) - 5, 6):
            segs.append(nums[i : i + 6])
    return start, segs


def sample(start: tuple[float, float], segs: list[list[float]], per_seg: int = 6):
    """Dense samples of the polybezier."""
    px, py = [start[0]], [start[1]]
    cur = start
    for s in segs:
        p0 = cur
        p1, p2, p3 = (s[0], s[1]), (s[2], s[3]), (s[4], s[5])
        for i in range(1, per_seg + 1):
            t = i / per_seg
            u = 1 - t
            px.append(u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0])
            py.append(u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1])
        cur = p3
    return np.asarray(px), np.asarray(py)


def fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def emit(curve: CubicSpline, knots: np.ndarray) -> str:
    parts = [f"M{fmt(knots[0])} {fmt(float(curve(knots[0])))}"]
    for a, b in zip(knots[:-1], knots[1:]):
        h = b - a
        parts.append(
            f"C{fmt(a + h / 3)} {fmt(float(curve(a) + curve(a, 1) * h / 3))} "
            f"{fmt(b - h / 3)} {fmt(float(curve(b) - curve(b, 1) * h / 3))} "
            f"{fmt(b)} {fmt(float(curve(b)))}"
        )
    return " ".join(parts)


def monotonic_xy(x: np.ndarray, y: np.ndarray):
    """Keep only strictly increasing x so y can be treated as a function of x."""
    keep = [0]
    for i in range(1, len(x)):
        if x[i] > x[keep[-1]] + 1e-6:
            keep.append(i)
    return x[keep], y[keep]



def resmooth(d: str, knot_step: float) -> str:
    start, segs = parse_cubics(d)
    if not segs:
        return d
    x, y = monotonic_xy(*sample(start, segs))
    if x.size < 4:
        return d
    curve = CubicSpline(x, y, bc_type="natural")
    knots = np.arange(x[0], x[-1], knot_step)
    if knots.size == 0 or knots[-1] != x[-1]:
        knots = np.append(knots, x[-1])
    return emit(curve, knots)


def envelope_from(paths: list[str], upper: bool):
    """C2 spline through the outermost blue streamline, used as the fan's tangent."""
    best = None
    for d in paths:
        start, segs = parse_cubics(d)
        x, y = monotonic_xy(*sample(start, segs))
        if x.size < 4:
            continue
        score = y.min() if upper else y.max()
        if best is None or (score < best[2] if upper else score > best[2]):
            best = (x, y, score)
    if best is None:
        return None
    x, y = best[0], best[1]
    return CubicSpline(x, y, bc_type="natural")


def reshape_ray(d: str, env: CubicSpline, knot_step: float, blend_end: float = 0.55) -> str:
    """Blend a fan ray out of the streamline envelope with a smoothstep weight.

    w(t) = 1 at the hub and 0 by blend_end, and both w and w' vanish there, so
    the ray leaves the hub tangent to the envelope and rejoins its own measured
    geometry smoothly - no straight chord cutting across the blue lines.
    """
    start, segs = parse_cubics(d)
    x, y = monotonic_xy(*sample(start, segs))
    if x.size < 4:
        return d
    curve = CubicSpline(x, y, bc_type="natural")
    knots = np.arange(x[0], x[-1], knot_step)
    if knots.size == 0 or knots[-1] != x[-1]:
        knots = np.append(knots, x[-1])

    span = max(x[-1] - x[0], 1e-6)
    t = np.clip((knots - x[0]) / (span * blend_end), 0.0, 1.0)
    w = (1.0 - t) ** 2 * (1.0 + 2.0 * t)  # smoothstep complement: w(0)=1, w(1)=w'(1)=0
    ye = np.asarray(env(knots), dtype=float)
    yo = np.asarray(curve(knots), dtype=float)
    # Anchor the blend on the ray's own start so the hub position is unchanged.
    shift = float(yo[0] - ye[0])
    blended = w * (ye + shift) + (1.0 - w) * yo
    shaped = CubicSpline(knots, blended, bc_type="natural")
    return emit(shaped, knots)


def process(svg: str, knot_step: float, blend_end: float) -> tuple[str, dict[str, int]]:
    spans = top_level_spans(svg, "emblem")
    stats = {"flow": 0, "fan": 0}
    flow_idx, fan_idx = 6, 4

    flow_block = svg[spans[flow_idx][1] : spans[flow_idx][2]]
    flow_paths = re.findall(r'<path d="([^"]+)"', flow_block)
    up_env = envelope_from(flow_paths, upper=True)
    lo_env = envelope_from(flow_paths, upper=False)

    def rewrite(block: str, fn) -> str:
        out = []
        pos = 0
        for m in re.finditer(r'(<path d=")([^"]+)(")', block):
            out.append(block[pos : m.start()])
            out.append(m.group(1) + fn(m.group(2)) + m.group(3))
            pos = m.end()
        out.append(block[pos:])
        return "".join(out)

    for idx in sorted((flow_idx, fan_idx), reverse=True):
        _tag, a, b = spans[idx]
        block = svg[a:b]
        if idx == flow_idx:
            new = rewrite(block, lambda d: resmooth(d, knot_step))
            stats["flow"] = block.count("<path")
        else:
            def shape(d: str) -> str:
                _s, sg = parse_cubics(d)
                if not sg:
                    return d
                xs, ys = monotonic_xy(*sample(*parse_cubics(d)))
                if xs.size < 4:
                    return d
                # Rays above the hub belong to the upper envelope.
                env = up_env if ys[-1] < ys[0] else lo_env
                return reshape_ray(d, env, knot_step, blend_end) if env is not None else resmooth(d, knot_step)

            new = rewrite(block, shape)
            stats["fan"] = block.count("<path")
        svg = svg[:a] + new + svg[b:]
    return svg, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knot-step", type=float, default=48.0)
    ap.add_argument("--blend-end", type=float, default=0.55)
    ap.add_argument("--apply", action="store_true", help="write masters instead of candidates")
    ap.add_argument("--suffix", default="-resmooth")
    args = ap.parse_args()

    outdir = ROOT / "tools" / "_compare"
    outdir.mkdir(parents=True, exist_ok=True)
    for name in ("logo-emblem.svg", "logo.svg"):
        src = ROOT / name
        out, stats = process(src.read_text(encoding="utf-8"), args.knot_step, args.blend_end)
        dest = src if args.apply else outdir / f"{src.stem}{args.suffix}.svg"
        dest.write_text(out, encoding="utf-8")
        print(f"{name}: resmoothed {stats['flow']} flow + {stats['fan']} fan paths -> {dest.name}")


if __name__ == "__main__":
    main()
