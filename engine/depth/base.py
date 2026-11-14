"""The interface between depth sources (Blender, later AI depth) and the rest.

Depth frames are float32 (H, W) in [0, 1]; 1 = nearest, 0 = background.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from engine.formats import Format
    from engine.scene import Scene


class DepthSequence(Protocol):
    fps: int
    width: int
    height: int
    reveal: np.ndarray | None

    def __len__(self) -> int: ...

    def frame(self, i: int) -> np.ndarray: ...


class DepthSource(Protocol):
    def render(self, scene: "Scene", fmt: "Format") -> DepthSequence: ...


class ArrayDepthSequence:
    """Depth frames held in memory (tests, procedural sources)."""

    def __init__(self, frames: list[np.ndarray], fps: int = 30, reveal: np.ndarray | None = None):
        if not frames:
            raise ValueError("a depth sequence needs at least one frame")
        self._frames = frames
        self.fps = fps
        self.height, self.width = frames[0].shape
        self.reveal = reveal

    def __len__(self) -> int:
        return len(self._frames)

    def frame(self, i: int) -> np.ndarray:
        if not 0 <= i < len(self._frames):
            raise IndexError(f"frame {i} out of range (0..{len(self._frames) - 1})")
        return np.asarray(self._frames[i], dtype=np.float32)


class PngDepthSequence:
    """Frames depth_0001.png ... (16-bit gray) plus meta.json and optional reveal.png.

    With `size=(width, height)` frames and the reveal image are scaled to that
    size on read (depth is rendered at reduced resolution to save time).
    """

    def __init__(self, directory: Path, fps: int = 30, size: tuple[int, int] | None = None):
        self.directory = Path(directory)
        self.fps = fps
        meta = json.loads((self.directory / "meta.json").read_text())
        self._count = int(meta["frames"])
        with Image.open(self._path(0)) as first:
            self._native = first.size
        self.width, self.height = size or self._native
        reveal_path = self.directory / "reveal.png"
        self.reveal = None
        if reveal_path.exists():
            img = Image.open(reveal_path).convert("RGB")
            if img.size != (self.width, self.height):
                img = img.resize((self.width, self.height), Image.LANCZOS)
            self.reveal = np.asarray(img)

    def _path(self, i: int) -> Path:
        return self.directory / f"depth_{i + 1:04d}.png"

    def __len__(self) -> int:
        return self._count

    def frame(self, i: int) -> np.ndarray:
        if not 0 <= i < self._count:
            raise IndexError(f"frame {i} out of range (0..{self._count - 1})")
        with Image.open(self._path(i)) as img:
            arr = np.asarray(img)
        if arr.ndim == 3:
            arr = arr[..., 0]
        scale = 65535.0 if arr.dtype != np.uint8 else 255.0
        depth = (arr.astype(np.float32) / scale).clip(0.0, 1.0)
        if (self.width, self.height) != self._native:
            img = Image.fromarray(depth, mode="F").resize((self.width, self.height), Image.BILINEAR)
            depth = np.asarray(img, dtype=np.float32).clip(0.0, 1.0)
        return depth
