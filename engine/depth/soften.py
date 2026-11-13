"""Soft depth edges: the hidden object rises gradually out of the background.

A hard step from background to object makes its outline visible in the
pattern without focusing; a gentle slope hides it much better.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from engine.formats import Format

SOFTNESS_SCALE = 0.0025  # sigma as a fraction of frame width at edge_softness = 1 (keeps thin strokes)


def softness_sigma(edge_softness: float, fmt: Format) -> float:
    return float(edge_softness) * SOFTNESS_SCALE * fmt.width


def _box(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    """Box blur of radius r along axis, edges extended."""
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def soften(depth: np.ndarray, sigma_px: float) -> np.ndarray:
    """Approximate Gaussian blur (three box passes per axis)."""
    if sigma_px <= 0:
        return depth
    r = max(1, int(round((np.sqrt(4 * sigma_px**2 + 1) - 1) / 2)))
    out = depth.astype(np.float64)
    for axis in (0, 1):
        for _ in range(3):
            out = _box(out, r, axis)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


@njit(cache=True)
def limit_slope(depth: np.ndarray, fall: float) -> np.ndarray:
    """Raise the low side of steep steps so depth never changes by more than
    `fall` per pixel (cone dilation, two passes). The object keeps its full
    height; its outline becomes a gentle ramp instead of a hard step."""
    h, w = depth.shape
    out = depth.astype(np.float32).copy()
    diag = fall * 1.41421356
    for y in range(h):
        for x in range(w):
            v = out[y, x]
            if x > 0:
                v = max(v, out[y, x - 1] - fall)
            if y > 0:
                v = max(v, out[y - 1, x] - fall)
                if x > 0:
                    v = max(v, out[y - 1, x - 1] - diag)
                if x < w - 1:
                    v = max(v, out[y - 1, x + 1] - diag)
            out[y, x] = v
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            v = out[y, x]
            if x < w - 1:
                v = max(v, out[y, x + 1] - fall)
            if y < h - 1:
                v = max(v, out[y + 1, x] - fall)
                if x < w - 1:
                    v = max(v, out[y + 1, x + 1] - diag)
                if x > 0:
                    v = max(v, out[y + 1, x - 1] - diag)
            out[y, x] = v
    return out
