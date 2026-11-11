"""Compositor: subtle text, fades, reveals, title cards and thumbnails.

On-screen text is plain white at 85 % opacity with a soft shadow, about 3 %
of the frame height, kept in thin bands near the top and bottom.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from engine.formats import Format
from engine.scene import Scene

TEXT_SHOW = 4.0
FADE = 0.5
END_SHOW = 3.0
SHORT_REVEAL = 3.0
LONG_REVEAL = 4.0
LONG_FADE_OUT = 0.5

TEXT_OPACITY = 0.85
TEXT_HEIGHT = 0.03  # cap height as a fraction of frame height
TOP_Y = 0.12
BOTTOM_Y = 0.76
COUNTER_OPACITY = 0.60
BAR_OPACITY = 0.35
REVEAL_BG = (140, 142, 146)
CARD_BG = (12, 12, 14)


# ---------------------------------------------------------------- timelines

@dataclass(frozen=True)
class ShortAlphas:
    main: float
    end: float
    reveal: float


def _ramp(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def short_alphas(t: float, duration: float, *, has_end: bool, reveal: bool) -> ShortAlphas:
    """Opacity of the opening text, the closing text and the reveal at time t."""
    tail = SHORT_REVEAL if reveal else (END_SHOW if has_end else 0.0)
    tail_start = duration - tail
    main_end = TEXT_SHOW if tail == 0 else min(TEXT_SHOW, max(0.0, tail_start - FADE))
    main = 1.0 if t < main_end else _ramp(1.0 - (t - main_end) / FADE)
    if tail and t >= tail_start:
        main = 0.0
    ramp_in = _ramp((t - tail_start) / FADE) if tail else 0.0
    return ShortAlphas(main=main, end=ramp_in if has_end and not reveal else 0.0,
                       reveal=ramp_in if reveal else 0.0)


# ---------------------------------------------------------------- text layers

def _font(font: Path, cap_px: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font), max(8, round(cap_px / 0.72)))


class TextLayer:
    """A line of text pre-rendered once, blended per frame into a band of rows."""

    def __init__(self, text: str, frame_w: int, frame_h: int, font: Path, *, cap_px: float,
                 centre: tuple[float, float], opacity: float = TEXT_OPACITY, anchor: str = "mm"):
        size_font = _font(font, cap_px)
        while size_font.getlength(text) > 0.9 * frame_w and size_font.size > 8:
            size_font = ImageFont.truetype(str(font), size_font.size - 2)
        img = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
        shadow = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
        off = max(1, round(0.002 * frame_h))
        ImageDraw.Draw(shadow).text((centre[0] + off, centre[1] + off), text, font=size_font,
                                    fill=(0, 0, 0, 128), anchor=anchor)
        shadow = shadow.filter(ImageFilter.GaussianBlur(max(1, 0.004 * frame_h)))
        txt = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
        ImageDraw.Draw(txt).text(centre, text, font=size_font, fill=(255, 255, 255, round(255 * opacity)),
                                 anchor=anchor)
        img = Image.alpha_composite(Image.alpha_composite(img, shadow), txt)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        rows = np.nonzero(arr[..., 3].max(axis=1) > 0)[0]
        self.y0, self.y1 = (int(rows[0]), int(rows[-1]) + 1) if len(rows) else (0, 0)
        band = arr[self.y0:self.y1]
        self.alpha = band[..., 3:4]
        self.rgb = band[..., :3] * 255.0

    def blend(self, frame: np.ndarray, strength: float) -> None:
        """Blend into a float32 frame in place."""
        if strength <= 0 or self.y1 <= self.y0:
            return
        a = self.alpha * strength
        region = frame[self.y0:self.y1]
        region *= 1.0 - a
        region += self.rgb * a


def _line(text: str | None, fmt: Format, font: Path, y_frac: float) -> TextLayer | None:
    if not text:
        return None
    return TextLayer(text, fmt.width, fmt.height, font, cap_px=TEXT_HEIGHT * fmt.height,
                     centre=(fmt.width / 2, y_frac * fmt.height))


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    return np.clip(np.round(frame), 0, 255).astype(np.uint8)


def fit_reveal(img: np.ndarray, fmt: Format) -> np.ndarray:
    """Scale the reveal image to fit the frame, letterboxed on neutral grey."""
    src = Image.fromarray(img)
    scale = min(fmt.width / src.width, fmt.height / src.height)
    size = (max(1, round(src.width * scale)), max(1, round(src.height * scale)))
    canvas = Image.new("RGB", (fmt.width, fmt.height), REVEAL_BG)
    canvas.paste(src.resize(size, Image.LANCZOS), ((fmt.width - size[0]) // 2, (fmt.height - size[1]) // 2))
    return np.asarray(canvas)


# ---------------------------------------------------------------- Shorts

class ShortComposer:
    def __init__(self, scene: Scene, fmt: Format, font: Path, reveal_img: np.ndarray | None):
        self.fmt = fmt
        self.duration = scene.duration
        self.top = _line(scene.text.top, fmt, font, TOP_Y)
        self.bottom = _line(scene.text.bottom, fmt, font, BOTTOM_Y)
        self.end = _line(scene.text.end, fmt, font, BOTTOM_Y)
        self.reveal = fit_reveal(reveal_img, fmt).astype(np.float32) if reveal_img is not None else None

    def frame(self, stereo: np.ndarray, t: float) -> np.ndarray:
        a = short_alphas(t, self.duration, has_end=self.end is not None, reveal=self.reveal is not None)
        if a.main <= 0 and a.end <= 0 and a.reveal <= 0:
            return stereo
        out = stereo.astype(np.float32)
        if a.reveal > 0 and self.reveal is not None:
            out = out * (1.0 - a.reveal) + self.reveal * a.reveal
        for layer in (self.top, self.bottom):
            if layer is not None:
                layer.blend(out, a.main)
        if self.end is not None:
            self.end.blend(out, a.end)
        return _to_uint8(out)


# ---------------------------------------------------------------- long-video segments

class SegmentComposer:
    """One scene inside a long video: stereogram + counter + progress, then the reveal."""

    def __init__(self, fmt: Format, font: Path, *, index: int, total: int, title: str,
                 stereo_seconds: float, reveal_img: np.ndarray):
        self.fmt = fmt
        self.stereo_seconds = stereo_seconds
        self.total_seconds = stereo_seconds + LONG_REVEAL + LONG_FADE_OUT
        self.counter = TextLayer(f"{index} / {total}", fmt.width, fmt.height, font,
                                 cap_px=0.022 * fmt.height, opacity=COUNTER_OPACITY, anchor="lt",
                                 centre=(fmt.safe.left * fmt.width, fmt.safe.top * fmt.height))
        title_layer = _line(title, fmt, font, BOTTOM_Y)
        revealed = fit_reveal(reveal_img, fmt).astype(np.float32)
        if title_layer is not None:
            title_layer.blend(revealed, 1.0)
        self.revealed = revealed
        self.bar_h = max(2, round(4 * fmt.height / 1080))
        self.bar_y = round(fmt.height * 0.96)

    def frame(self, stereo: np.ndarray, t: float) -> np.ndarray:
        out = stereo.astype(np.float32)
        if t < self.stereo_seconds:
            fill = round(self.fmt.width * _ramp(t / self.stereo_seconds))
            if fill > 0:
                bar = out[self.bar_y:self.bar_y + self.bar_h, :fill]
                bar *= 1.0 - BAR_OPACITY
                bar += 255.0 * BAR_OPACITY
        else:
            a = _ramp((t - self.stereo_seconds) / FADE)
            out = out * (1.0 - a) + self.revealed * a
        self.counter.blend(out, 1.0)
        fade = _ramp((self.total_seconds - t) / LONG_FADE_OUT)
        if fade < 1.0:
            out *= fade
        return _to_uint8(out)


# ---------------------------------------------------------------- cards and thumbnails

def card(lines: list[str], fmt: Format, font: Path) -> np.ndarray:
    """Dark title card: first line large, the rest small, centred."""
    img = np.zeros((fmt.height, fmt.width, 3), np.float32)
    img[:] = CARD_BG
    sizes = [0.05 if i == 0 else 0.028 for i in range(len(lines))]
    gap = 0.03 * fmt.height
    heights = [s * fmt.height for s in sizes]
    total = sum(heights) + gap * (len(lines) - 1)
    y = (fmt.height - total) / 2
    for text, h in zip(lines, heights):
        TextLayer(text, fmt.width, fmt.height, font, cap_px=h, opacity=1.0,
                  centre=(fmt.width / 2, y + h / 2)).blend(img, 1.0)
        y += h + gap
    return _to_uint8(img)


def thumbnail(stereo: np.ndarray, title: str, font: Path) -> np.ndarray:
    """1280x720 still for YouTube: centre crop to 16:9 plus the title."""
    h, w = stereo.shape[:2]
    if w / h < 16 / 9:
        ch = round(w * 9 / 16)
        y0 = (h - ch) // 2
        crop = stereo[y0:y0 + ch]
    else:
        cw = round(h * 16 / 9)
        x0 = (w - cw) // 2
        crop = stereo[:, x0:x0 + cw]
    img = np.asarray(Image.fromarray(crop).resize((1280, 720), Image.LANCZOS), dtype=np.float32)
    TextLayer(title, 1280, 720, font, cap_px=0.07 * 720, opacity=1.0, centre=(640, 0.82 * 720)).blend(img, 1.0)
    return _to_uint8(img)
