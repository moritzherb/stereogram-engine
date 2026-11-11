"""Quality warnings, explained in plain English. Warnings never stop a render."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from engine.decode import self_check
from engine.depth.base import DepthSequence
from engine.formats import Format
from engine.patterns import DETAIL_MAX, SEAM_MAX, detail_score, seam_score

OBJECT_THRESHOLD = 0.02


@dataclass(frozen=True)
class QualityWarning:
    code: str
    message: str


def check_safe_area(seq: DepthSequence, fmt: Format, step: int = 5) -> list[QualityWarning]:
    x0, y0, x1, y1 = fmt.safe.rect(seq.width, seq.height)
    for i in range(0, len(seq), step):
        ys, xs = np.nonzero(seq.frame(i) > OBJECT_THRESHOLD)
        if len(xs) == 0:
            continue
        if xs.min() < x0 - 1 or ys.min() < y0 - 1 or xs.max() > x1 + 1 or ys.max() > y1 + 1:
            return [QualityWarning(
                "OUT_OF_SAFE_AREA",
                f"The hidden object leaves the safe area at {i / seq.fps:.1f} s, where app buttons or "
                f"captions may cover it. Try a different motion or a smaller speed.")]
    return []


def check_motion(seq: DepthSequence, threshold: float = 0.12) -> list[QualityWarning]:
    prev = None
    worst, worst_i = 0.0, 0
    for i in range(len(seq)):
        cur = seq.frame(i)[::4, ::4]
        if prev is not None:
            mask = (cur > OBJECT_THRESHOLD) | (prev > OBJECT_THRESHOLD)
            if mask.any():
                metric = float(np.abs(cur - prev)[mask].mean())
                if metric > worst:
                    worst, worst_i = metric, i
        prev = cur
    if worst > threshold:
        return [QualityWarning(
            "MOTION_TOO_FAST",
            f"The motion is very fast around {worst_i / seq.fps:.1f} s; viewers may lose the 3D effect. "
            f"Try a lower 'speed:' (for example half of the current value).")]
    return []


def check_pattern(strip: np.ndarray) -> list[QualityWarning]:
    warnings = []
    if detail_score(strip) > DETAIL_MAX:
        warnings.append(QualityWarning(
            "PATTERN_TOO_FINE",
            "The pattern has very fine detail that will smear after TikTok/YouTube compression. "
            "Use a smoother or larger-scale image."))
    if seam_score(strip) > SEAM_MAX:
        warnings.append(QualityWarning(
            "PATTERN_SEAMS",
            "The pattern has a visible seam where it repeats. Use an image without hard edges at its sides."))
    return warnings


def check_frames(depth_frames: Sequence[np.ndarray], stereo_frames: Sequence[np.ndarray], *, repeat_px: int,
                 depth_strength: float, crosseyed: bool = False) -> list[QualityWarning]:
    """Read the hidden depth back out of rendered frames and compare."""
    scores = []
    for depth, img in zip(depth_frames, stereo_frames):
        d = 1.0 - depth if crosseyed else depth
        scores.append(self_check(d, img, repeat_px=repeat_px, depth_strength=depth_strength, step=4))
    failed = [s for s in scores if not s.passed]
    if failed:
        worst = min(s.score for s in failed)
        return [QualityWarning(
            "SELF_CHECK_FAILED",
            f"The engine could only read back {worst:.0%} of the hidden shape from its own stereogram. "
            f"The 3D effect may be hard to see; try a pattern with more texture.")]
    return []
