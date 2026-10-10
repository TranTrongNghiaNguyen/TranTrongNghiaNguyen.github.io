"""Trace the red fan rays in polar coordinates about the hub.

Cartesian column tracking cannot follow these rays: near the hub at (164, 421)
they are almost vertical, so one x column crosses many rays at once and the
tracker collapses several seeds onto the same ridge. In polar coordinates about
the hub each ray is a single-valued, slowly-turning function theta(r), which is
exactly the form a tracker handles well - and it is also the physically correct
description of a fan that leaves one stagnation point.

The fitted theta(r) is then resampled as a smooth C2 spline and emitted as a
cubic-Bezier path in Cartesian space, so the ray hugs the body near the hub and
opens out gradually instead of shooting off as a straight chord.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from trace_streams import ROOT, load_rgb, responses

HUB = (164.0, 421.0)


def polar_grid(shape: tuple[int, int], hub: tuple[float, float]):
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
    dx, dy = xs - hub[0], ys - hub[1]
    return np.hypot(dx, dy), np.degrees(np.arctan2(dy, dx))


def ray_peaks(
    resp: np.ndarray,
    rad: np.ndarray,
    ang: np.ndarray,
    r: float,
    dr: float,
    a0: float,
    a1: float,
    thresh: float,
    bin_deg: float = 0.25,
) -> list[tuple[float, float]]:
    """Angular ink profile on the arc at radius r; returns (angle, strength)."""
    band = (np.abs(rad - r) < dr) & (ang >= a0) & (ang <= a1) & (resp > thresh)
    if not band.any():
        return []
    a = ang[band]
    w = resp[band]
    nbins = max(4, int((a1 - a0) / bin_deg))
    idx = np.clip(((a - a0) / (a1 - a0) * nbins).astype(int), 0, nbins - 1)
    wsum = np.bincount(idx, weights=w, minlength=nbins)
    asum = np.bincount(idx, weights=w * a, minlength=nbins)

    out: list[tuple[float, float]] = []
    for i in range(1, nbins - 1):
        if wsum[i] <= 0 or wsum[i] < wsum[i - 1] or wsum[i] < wsum[i + 1]:
            continue
        out.append((float(asum[i] / wsum[i]), float(wsum[i])))
    merged: list[tuple[float, float]] = []
    for a_, w_ in out:
        if merged and a_ - merged[-1][0] < 0.8:
            if w_ > merged[-1][1]:
                merged[-1] = (a_, w_)
        else:
            merged.append((a_, w_))
    return merged


def track_ray(
    resp: np.ndarray,
    rad: np.ndarray,
    ang: np.ndarray,
    r_seed: float,
    a_seed: float,
    r_min: float,
    r_max: float,
    thresh: float,
    step: float = 6.0,
    window: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Follow one ray inward and outward, extrapolating dtheta/dr."""
    rs, as_ = [r_seed], [a_seed]
    for direction in (+1.0, -1.0):
        r, a, slope, misses = r_seed, a_seed, 0.0, 0
        while True:
            nr = r + direction * step
            if nr < r_min or nr > r_max:
                break
            pred = a + slope * direction * step
            cands = ray_peaks(resp, rad, ang, nr, step * 0.6, pred - window, pred + window, thresh)
            if not cands:
                misses += 1
                if misses > 3:
                    break
                r, a = nr, pred
                continue
            misses = 0
            na = min(cands, key=lambda c: abs(c[0] - pred))[0]
            slope = (na - a) / (direction * step)
            r, a = nr, na
            rs.append(r)
            as_.append(a)
    order = np.argsort(rs)
    return np.asarray(rs)[order], np.asarray(as_)[order]



def fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def ray_to_path(r: np.ndarray, theta: np.ndarray, knot_step: float = 24.0) -> str:
    """Smooth theta(r), then serialize x(r),y(r) as exact cubic Hermite segments."""
    # Reject duplicate radii and denoise the angular ridge before interpolation.
    ur, ui = np.unique(r, return_index=True)
    a = theta[ui]
    win = min(9, (len(a) // 2) * 2 + 1)
    if win >= 5:
        a = savgol_filter(a, win, 2, mode="interp")
    # Every fan ray begins at the same hub; gently bridge the unmeasurable first
    # few pixels instead of extrapolating a noisy angular tangent into the apex.
    if ur[0] > 4:
        ur = np.insert(ur, 0, 0.0)
        a = np.insert(a, 0, a[0])
    sx, sy = [], []
    rr = np.arange(ur[0], ur[-1], knot_step)
    if rr.size == 0 or rr[-1] != ur[-1]:
        rr = np.append(rr, ur[-1])
    ac = CubicSpline(ur, a, bc_type="natural")
    aa = np.radians(ac(rr))
    sx = HUB[0] + rr * np.cos(aa)
    sy = HUB[1] + rr * np.sin(aa)
    # Parameterise the Cartesian samples by radius; Hermite derivatives retain C1.
    xc = CubicSpline(rr, sx, bc_type="natural")
    yc = CubicSpline(rr, sy, bc_type="natural")
    parts = [f"M{fmt(float(xc(rr[0])))} {fmt(float(yc(rr[0])))}"]
    for u, v in zip(rr[:-1], rr[1:]):
        h = v - u
        parts.append(
            f"C{fmt(float(xc(u) + xc(u, 1) * h / 3))} {fmt(float(yc(u) + yc(u, 1) * h / 3))} "
            f"{fmt(float(xc(v) - xc(v, 1) * h / 3))} {fmt(float(yc(v) - yc(v, 1) * h / 3))} "
            f"{fmt(float(xc(v)))} {fmt(float(yc(v)))}"
        )
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default="airfoil - copy.png")
    ap.add_argument("--seed-r", type=float, default=260.0)
    ap.add_argument("--angles", default="-36,-15;15,36", help="semicolon-separated angular bands")
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()

    resp = responses(load_rgb(ROOT / args.png))["red"]
    rad, ang = polar_grid(resp.shape, HUB)
    all_rays = []
    for spec in args.angles.split(";"):
        a0, a1 = (float(v) for v in spec.split(","))
        seeds = ray_peaks(resp, rad, ang, args.seed_r, 2.0, a0, a1, args.threshold)
        print(f"band {a0:g}..{a1:g}: {len(seeds)} seeds -> {[round(a, 2) for a, _ in seeds]}")
        for a, strength in seeds:
            r, theta = track_ray(resp, rad, ang, args.seed_r, a, 25, 500, args.threshold)
            all_rays.append((r, theta))
            print(
                f"  seed {a:7.2f}: r {r[0]:6.1f}..{r[-1]:6.1f} n={len(r):3d}"
                f" theta {theta.min():7.2f}..{theta.max():7.2f}"
            )
    if args.emit:
        for r, a in all_rays:
            if len(r) >= 8:
                print(f'      <path d="{ray_to_path(r, a)}"/>')


if __name__ == "__main__":
    main()
