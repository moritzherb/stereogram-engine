"""Stereogram core: depth map + pattern strip -> textured stereogram frame.

Constraint linking with hidden-surface removal after Thimbleby, Inglis &
Witten (1994), "Displaying 3D images: algorithms for single-image random-dot
stereograms". Colours come from a pattern strip and are anchored at the
centre column, so regions whose depth does not change keep identical pixels
from frame to frame (no flicker).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


def separation(z, eye_px: float, mu: float):
    """Pixel distance between the two images of a point at depth z (unrounded)."""
    return (1.0 - mu * np.asarray(z, dtype=np.float64)) * eye_px / (2.0 - mu * np.asarray(z, dtype=np.float64))


@njit(cache=True)
def _window_max(z, radius):
    """max(z[x-radius : x+radius+1]) for every x (van Herk / Gil-Werman)."""
    n = z.shape[0]
    w = 2 * radius + 1
    padded = np.zeros(n + 2 * radius, dtype=np.float32)
    padded[radius:radius + n] = z
    m = padded.shape[0]
    g = np.empty(m, dtype=np.float32)
    h = np.empty(m, dtype=np.float32)
    for i in range(m):
        if i % w == 0:
            g[i] = padded[i]
        else:
            g[i] = max(g[i - 1], padded[i])
    for i in range(m - 1, -1, -1):
        if i == m - 1 or (i + 1) % w == 0:
            h[i] = padded[i]
        else:
            h[i] = max(h[i + 1], padded[i])
    out = np.empty(n, dtype=np.float32)
    for x in range(n):
        out[x] = max(h[x], g[x + w - 1])
    return out


@njit(cache=True)
def _link_row(z, eye, mu, same, wmax):
    width = z.shape[0]
    step = 2.0 * 2.0 / (mu * eye)  # minimum growth of zt per t (conservative)
    for x in range(width):
        same[x] = x
    for x in range(width):
        zx = z[x]
        s = int(round((1.0 - mu * zx) * eye / (2.0 - mu * zx)))
        left = x - s // 2
        right = left + s
        if left < 0 or right >= width:
            continue
        # Hidden-surface removal. Fast path: nothing nearby rises above the
        # first sight-line height, so the point is visible to both eyes.
        visible = True
        if wmax[x] >= zx + (2.0 - mu * zx) * step / 2.0:
            t = 1
            while True:
                zt = zx + 2.0 * (2.0 - mu * zx) * t / (mu * eye)
                zl = z[x - t] if x - t >= 0 else 0.0
                zr = z[x + t] if x + t < width else 0.0
                visible = zl < zt and zr < zt
                t += 1
                if not (visible and zt < 1.0):
                    break
        if visible:
            l = same[left]
            while l != left and l != right:
                if l < right:
                    left = l
                else:
                    same[left] = right
                    left = right
                    right = l
                l = same[left]
            same[left] = right


@njit(parallel=True, cache=True)
def _render(depth, pattern, eye, mu, period, radius, factor):
    """Per row: upsample depth x factor, link, colour, average back down."""
    height, width_out = depth.shape
    width = width_out * factor
    ph = pattern.shape[0]
    centre = width // 2
    out = np.empty((height, width_out, 3), dtype=np.uint8)
    for y in prange(height):
        src = depth[y]
        z = np.empty(width, dtype=np.float32)
        for x in range(width):
            pos = (x + 0.5) / factor - 0.5
            if pos <= 0.0:
                z[x] = src[0]
            elif pos >= width_out - 1:
                z[x] = src[width_out - 1]
            else:
                i0 = int(pos)
                f = pos - i0
                z[x] = src[i0] * (1.0 - f) + src[i0 + 1] * f
        same = np.empty(width, dtype=np.int64)
        wmax = _window_max(z, radius)
        _link_row(z, eye, mu, same, wmax)
        root = np.empty(width, dtype=np.int64)
        for x in range(width - 1, -1, -1):
            root[x] = x if same[x] == x else root[same[x]]
        best = np.full(width, -1, dtype=np.int64)
        for x in range(width):
            r = root[x]
            if best[r] == -1 or abs(x - centre) < abs(best[r] - centre):
                best[r] = x
        prow = pattern[y % ph]
        for xo in range(width_out):
            acc0 = 0
            acc1 = 0
            acc2 = 0
            for k in range(factor):
                col = ((best[root[xo * factor + k]] - centre) % period) // factor
                acc0 += prow[col, 0]
                acc1 += prow[col, 1]
                acc2 += prow[col, 2]
            out[y, xo, 0] = (2 * acc0 + factor) // (2 * factor)
            out[y, xo, 1] = (2 * acc1 + factor) // (2 * factor)
            out[y, xo, 2] = (2 * acc2 + factor) // (2 * factor)
    return out


def render_stereogram(
    depth: np.ndarray,
    pattern: np.ndarray,
    *,
    repeat_px: int,
    depth_strength: float = 0.33,
    oversample: int = 3,
    crosseyed: bool = False,
) -> np.ndarray:
    """Render one stereogram frame.

    depth: float32 (H, W) in [0, 1], 1 = nearest. pattern: uint8 strip
    (Hp, repeat_px, 3). Returns uint8 (H, W, 3).
    """
    if pattern.shape[1] != repeat_px:
        raise ValueError(f"pattern width {pattern.shape[1]} != repeat_px {repeat_px}")
    d = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)
    if crosseyed:
        d = 1.0 - d
    d = np.ascontiguousarray(d)
    period = repeat_px * oversample
    eye = 2.0 * period
    radius = int(np.ceil(depth_strength * eye / 4.0)) + 1
    return _render(d, np.ascontiguousarray(pattern), eye, float(depth_strength), period, radius, oversample)
