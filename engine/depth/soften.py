"""Soft depth edges: the hidden object rises gradually out of the background.

A hard step from background to object makes its outline visible in the
pattern without focusing; a gentle slope hides it much better.
"""
from __future__ import annotations

import numpy as np

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
