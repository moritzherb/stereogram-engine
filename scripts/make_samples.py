"""Write full-screen stereogram samples for tuning on a phone.

    python scripts/make_samples.py      -> output/samples/*.png

Each image shows its settings at the bottom (repeat width, depth strength,
style). Look at them full-screen on the phone and pick what reads best.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from engine.patterns import generate_pattern
from engine.stereo import render_stereogram

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "samples"
FONT = ROOT / "assets" / "fonts" / "Lato-Regular.ttf"
W, H = 1080, 1920


def sphere() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - W / 2) ** 2 + (yy - H * 0.45) ** 2) / 330.0**2
    return np.sqrt(np.clip(1 - r2, 0, 1)).astype(np.float32)


def heart() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    x = (xx - W / 2) / 330.0
    y = -(yy - H * 0.43) / 330.0
    f = (x**2 + y**2 - 1) ** 3 - x**2 * y**3
    mask = Image.fromarray(((f <= 0) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(18))
    return (np.asarray(mask, np.float32) / 255.0) * 0.9


def word(text: str = "HI") -> np.ndarray:
    img = Image.new("L", (W, H), 0)
    font = ImageFont.truetype(str(FONT), 520)
    ImageDraw.Draw(img).text((W / 2, H * 0.45), text, font=font, fill=255, anchor="mm")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(6)), np.float32) / 255.0 * 0.85


def label(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FONT), 34)
    d.rectangle((0, H - 90, W, H), fill=(0, 0, 0))
    d.text((W / 2, H - 45), text, font=font, fill=(255, 255, 255), anchor="mm")
    return np.asarray(img)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for repeat in (140, 154, 170):
        jobs.append((f"1-repeat-{repeat}", sphere, "swirl", repeat, 0.33))
    for strength in (0.25, 0.33, 0.40):
        jobs.append((f"2-strength-{strength:.2f}", heart, "noise", 154, strength))
    for style in ("noise", "swirl", "cells"):
        jobs.append((f"3-style-{style}", word, style, 154, 0.33))
    for name, shape, style, repeat, strength in jobs:
        pattern = generate_pattern(style, repeat_px=repeat, seed=7)
        frame = render_stereogram(shape(), pattern, repeat_px=repeat, depth_strength=strength)
        text = f"{name}   repeat {repeat}px   strength {strength:.2f}   {style}"
        Image.fromarray(label(frame, text)).save(OUT / f"{name}.png")
        print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()
