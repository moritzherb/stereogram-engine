"""Self-check: read the hidden depth back out of a stereogram and compare.

For each pixel we search the separation at which the image matches itself
(the same search the viewer's eyes perform) and convert it back to depth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from engine.stereo import separation

AMBIGUITY_RATIO = 0.8


@dataclass(frozen=True)
class SelfCheck:
    score: float
    passed: bool


@njit(parallel=True, cache=True)
def _recover(img, s_min, s_max, eye, mu, step):
    height, width, _ = img.shape
    out = np.full((height, width), np.nan, dtype=np.float32)
    for yi in prange((height + step - 1) // step):
        y = yi * step
        for x in range(0, width, step):
            best_s = -1
            best_c = 1e18
            costs = np.full(s_max - s_min + 1, 1e18)
            for s in range(s_min, s_max + 1):
                a = x - s // 2
                b = a + s
                if a - 4 < 0 or b + 4 >= width:
                    continue
                c = 0.0
                for dy in range(-1, 2):
                    yy = min(max(y + dy, 0), height - 1)
                    for dx in range(-4, 5):
                        for ch in range(3):
                            c += abs(np.float64(img[yy, a + dx, ch]) - np.float64(img[yy, b + dx, ch]))
                costs[s - s_min] = c
                if c < best_c:
                    best_c = c
                    best_s = s
            if best_s < 0:
                continue
            second = 1e18
            for s in range(s_min, s_max + 1):
                if abs(s - best_s) > 1 and costs[s - s_min] < second:
                    second = costs[s - s_min]
            if second >= 1e18 or best_c >= AMBIGUITY_RATIO * second:
                continue
            z = (eye - 2.0 * best_s) / (mu * (eye - best_s))
            out[y, x] = min(max(z, 0.0), 1.0)
    return out


def recover_depth(img: np.ndarray, *, repeat_px: int, depth_strength: float = 0.33, step: int = 1) -> np.ndarray:
    """Depth (float32, NaN where undetermined) recovered from a stereogram.

    step > 1 only evaluates every step-th row and column (others stay NaN).
    """
    eye = 2.0 * repeat_px
    s_min = int(round(float(separation(1.0, eye, depth_strength))))
    s_max = int(round(float(separation(0.0, eye, depth_strength))))
    return _recover(np.ascontiguousarray(img), s_min, s_max, eye, float(depth_strength), int(step))


def _dilate(mask: np.ndarray, rx: int, ry: int) -> np.ndarray:
    """Box dilation via cumulative sums."""
    m = mask.astype(np.int32)
    c = np.pad(np.cumsum(m, axis=1), ((0, 0), (1, 0)))
    w = m.shape[1]
    lo = np.clip(np.arange(w) - rx, 0, w)
    hi = np.clip(np.arange(w) + rx + 1, 0, w)
    m = (c[:, hi] - c[:, lo]) > 0
    c = np.pad(np.cumsum(m.astype(np.int32), axis=0), ((1, 0), (0, 0)))
    h = m.shape[0]
    lo = np.clip(np.arange(h) - ry, 0, h)
    hi = np.clip(np.arange(h) + ry + 1, 0, h)
    return (c[hi, :] - c[lo, :]) > 0


def self_check(
    depth: np.ndarray,
    img: np.ndarray,
    *,
    repeat_px: int,
    depth_strength: float = 0.33,
    tolerance: float = 0.08,
    threshold: float = 0.95,
    step: int = 1,
) -> SelfCheck:
    """Fraction of interior pixels whose recovered depth matches `depth`."""
    rec = recover_depth(img, repeat_px=repeat_px, depth_strength=depth_strength, step=step)
    gy, gx = np.gradient(depth.astype(np.float32))
    edges = _dilate(np.hypot(gx, gy) > 0.05, repeat_px, 2)
    interior = ~edges
    interior[:, :repeat_px] = False
    interior[:, -repeat_px:] = False
    sampled = np.zeros_like(interior)
    sampled[::step, ::step] = True
    interior &= sampled
    total = int(interior.sum())
    if total == 0:
        return SelfCheck(1.0, True)
    good = np.abs(rec - depth) <= tolerance  # NaN compares False -> counts as a miss
    score = float((good & interior).sum()) / total
    return SelfCheck(score, score >= threshold)
