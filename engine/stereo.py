"""Stereogram core: depth map + pattern strip -> textured stereogram frame.

Continuous ("flow") method: every pixel gets a fractional pattern position.
A pixel takes the pattern position of its partner one separation away, read
with interpolation, so the pattern bends smoothly with the depth. This avoids
the thin seams along depth contours that whole-pixel linking leaves, and,
together with a limit on how steep the depth may change (see
engine.depth.soften.limit_slope), the sharp copies of the object's outline.

The pattern is anchored at the centre column and propagated outwards, so
regions whose depth does not change keep identical pixels from frame to frame.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


def separation(z, eye_px: float, mu: float):
    """Pixel distance between the two images of a point at depth z (unrounded)."""
    return (1.0 - mu * np.asarray(z, dtype=np.float64)) * eye_px / (2.0 - mu * np.asarray(z, dtype=np.float64))


@njit(cache=True)
def _interp(a, pos):
    n = a.shape[0]
    if pos <= 0.0:
        return a[0]
    if pos >= n - 1:
        return a[n - 1]
    i = int(pos)
    f = pos - i
    return a[i] * (1.0 - f) + a[i + 1] * f


@njit(parallel=True, cache=True)
def _render(depth, pattern, eye, mu, period, factor, shift):
    height, width_out = depth.shape
    width = width_out * factor
    ph = pattern.shape[0]
    centre = width // 2
    out = np.empty((height, width_out, 3), dtype=np.uint8)
    for y in prange(height):
        src_row = depth[y]
        sep = np.empty(width)
        for x in range(width):
            z = _interp(src_row, (x + 0.5) / factor - 0.5)
            sep[x] = (1.0 - mu * z) * eye / (2.0 - mu * z)
        u = np.empty(width)  # pattern position (oversampled pixels)
        # Right half: copy the position from one separation to the left. The
        # separation is taken at the midpoint between the two partners.
        for x in range(centre, width):
            s = _interp(sep, x - sep[x] / 2.0)
            src = x - s
            if src < centre:
                u[x] = (x - centre) * period / s  # start strip, scaled to fit exactly
            else:
                u[x] = _interp(u, src)
        # Left half: the mirror image, copying from one separation to the right.
        for x in range(centre - 1, -1, -1):
            s = _interp(sep, x + sep[x] / 2.0)
            src = x + s
            if src >= centre:
                u[x] = (x - centre) * period / s
            else:
                u[x] = _interp(u, src)
        prow = pattern[y % ph]
        for xo in range(width_out):
            acc0 = 0
            acc1 = 0
            acc2 = 0
            for k in range(factor):
                p = u[xo * factor + k] - shift
                col = int(np.floor(p - period * np.floor(p / period))) // factor
                if col >= pattern.shape[1]:
                    col = pattern.shape[1] - 1
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
    shift: int = 0,
) -> np.ndarray:
    """Render one stereogram frame.

    depth: float32 (H, W) in [0, 1], 1 = nearest. pattern: uint8 strip
    (Hp, repeat_px, 3). Returns uint8 (H, W, 3). `shift` moves the whole
    pattern right by that many pixels (used for slow pattern drift).
    """
    if pattern.shape[1] != repeat_px:
        raise ValueError(f"pattern width {pattern.shape[1]} != repeat_px {repeat_px}")
    d = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)
    if crosseyed:
        d = 1.0 - d
    period = repeat_px * oversample
    return _render(np.ascontiguousarray(d), np.ascontiguousarray(pattern), 2.0 * period, float(depth_strength),
                   period, oversample, int(shift) * oversample)
