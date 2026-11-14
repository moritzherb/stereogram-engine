"""Long-video (16:9) depth made from the Short's depth frames.

The model, its animation and its soft edges stay exactly as in the Short:
the frames are placed in the wider frame with more background around them,
scaled evenly (never stretched) and grown by at most MAX_GROWTH.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image

from engine.compose import background_colour
from engine.depth.base import DepthSequence
from engine.formats import Format

MAX_GROWTH = 1.2  # a bigger screen allows a slightly bigger model
OBJECT_THRESHOLD = 0.02


def _paste(canvas: np.ndarray, patch: np.ndarray, ox: int, oy: int) -> None:
    h, w = patch.shape[:2]
    cy0, cx0 = max(0, oy), max(0, ox)
    cy1, cx1 = min(canvas.shape[0], oy + h), min(canvas.shape[1], ox + w)
    if cy1 > cy0 and cx1 > cx0:
        canvas[cy0:cy1, cx0:cx1] = patch[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox]


class LongFromShort:
    def __init__(self, short: DepthSequence, fmt: Format,
                 transform: Callable[[np.ndarray], np.ndarray] | None = None, sample_step: int = 5):
        self.short = short
        self.fmt = fmt
        self.fps = short.fps
        self.width, self.height = fmt.width, fmt.height
        self.prepared = transform is not None  # frames already carry lift/soft edges
        self._transform = transform or (lambda d: d)

        # Where the object goes over the whole animation (one fixed placement).
        lo_x, lo_y, hi_x, hi_y = short.width, short.height, -1, -1
        for i in list(range(0, len(short), sample_step)) + [len(short) - 1]:
            ys, xs = np.nonzero(short.frame(i) > OBJECT_THRESHOLD)
            if len(xs):
                lo_x, lo_y = min(lo_x, xs.min()), min(lo_y, ys.min())
                hi_x, hi_y = max(hi_x, xs.max()), max(hi_y, ys.max())
        if hi_x < 0:  # nothing visible: use the middle of the frame
            lo_x, lo_y, hi_x, hi_y = short.width // 4, short.height // 4, 3 * short.width // 4, 3 * short.height // 4
        obj_w, obj_h = hi_x - lo_x + 1, hi_y - lo_y + 1
        sx0, sy0, sx1, sy1 = fmt.safe.rect(fmt.width, fmt.height)
        self.scale = min(MAX_GROWTH, (sx1 - sx0) / obj_w, (sy1 - sy0) / obj_h)

        margin = max(8, round(0.03 * short.width))  # room for the softened outline
        self._crop = (max(0, lo_x - margin), max(0, lo_y - margin),
                      min(short.width, hi_x + 1 + margin), min(short.height, hi_y + 1 + margin))
        cx0, cy0, cx1, cy1 = self._crop
        self._size = (max(1, round((cx1 - cx0) * self.scale)), max(1, round((cy1 - cy0) * self.scale)))
        obj_cx, obj_cy = (lo_x + hi_x + 1) / 2 - cx0, (lo_y + hi_y + 1) / 2 - cy0
        self._offset = (round((sx0 + sx1) / 2 - obj_cx * self.scale), round((sy0 + sy1) / 2 - obj_cy * self.scale))

        self.reveal = None
        if short.reveal is not None:
            crop = short.reveal[cy0:cy1, cx0:cx1]
            patch = Image.fromarray(crop).resize(self._size, Image.LANCZOS)
            canvas = np.empty((fmt.height, fmt.width, 3), np.uint8)
            canvas[:] = background_colour(crop)  # matches the area around the model: no visible seam
            _paste(canvas, np.asarray(patch), *self._offset)
            self.reveal = canvas

    def __len__(self) -> int:
        return len(self.short)

    def frame(self, i: int) -> np.ndarray:
        cx0, cy0, cx1, cy1 = self._crop
        depth = self._transform(self.short.frame(i))[cy0:cy1, cx0:cx1]
        patch = np.asarray(Image.fromarray(np.ascontiguousarray(depth, dtype=np.float32), mode="F")
                           .resize(self._size, Image.BILINEAR), dtype=np.float32)
        canvas = np.zeros((self.height, self.width), np.float32)
        _paste(canvas, np.clip(patch, 0.0, 1.0), *self._offset)
        return canvas
