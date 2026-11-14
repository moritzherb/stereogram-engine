"""A Blender-free depth source: a softly moving dome. Used by tests and demos."""
from __future__ import annotations

import math

import numpy as np

from engine.formats import Format
from engine.scene import Scene


class _DomeSequence:
    def __init__(self, fmt: Format, frames: int, fps: int, radius: float):
        self.fps, self.width, self.height = fps, fmt.width, fmt.height
        self._frames = frames
        x0, y0, x1, y1 = fmt.safe.rect(fmt.width, fmt.height)
        self._cx, self._cy = (x0 + x1) / 2, (y0 + y1) / 2
        self._r = radius * min(fmt.width, fmt.height)
        self._yy, self._xx = np.mgrid[0:fmt.height, 0:fmt.width].astype(np.float32)
        self.reveal = np.full((fmt.height, fmt.width, 3), 140, np.uint8)
        self.reveal[self._dome(0) > 0] = 235

    def _dome(self, i: int) -> np.ndarray:
        cx = self._cx + 0.02 * self.width * math.sin(2 * math.pi * i / max(1, self._frames))
        r2 = ((self._xx - cx) ** 2 + (self._yy - self._cy) ** 2) / self._r**2
        return np.sqrt(np.clip(1.0 - r2, 0.0, 1.0)).astype(np.float32)

    def __len__(self) -> int:
        return self._frames

    def frame(self, i: int) -> np.ndarray:
        if not 0 <= i < self._frames:
            raise IndexError(f"frame {i} out of range")
        return self._dome(i)


class FakeDepthSource:
    def __init__(self, radius: float = 0.15, fps: int = 30):
        self.radius = radius
        self.fps = fps
        self.calls = 0

    def render(self, scene: Scene, fmt: Format) -> _DomeSequence:
        self.calls += 1
        return _DomeSequence(fmt, max(1, round(scene.duration * self.fps)), self.fps, self.radius)
