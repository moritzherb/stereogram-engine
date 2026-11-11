"""Pattern strips for stereograms: generated in code or made from images.

A strip is uint8 (H, repeat_px, 3). It must tile seamlessly left-to-right
(the stereogram repeats it horizontally) and should tile top-to-bottom.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

STYLES = ("noise", "swirl", "cells", "stripes", "dots")

PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "default": [(38, 70, 120), (230, 160, 40), (120, 60, 30), (40, 150, 160), (240, 220, 170), (200, 80, 40)],
    "ocean": [(8, 30, 70), (20, 90, 150), (40, 160, 200), (150, 220, 230), (240, 250, 250)],
    "sunset": [(40, 20, 60), (140, 40, 90), (230, 90, 60), (250, 170, 60), (255, 230, 150)],
    "forest": [(20, 40, 20), (50, 90, 40), (110, 140, 60), (180, 170, 90), (90, 60, 30)],
    "candy": [(250, 120, 170), (120, 200, 250), (250, 230, 110), (160, 120, 240), (120, 230, 170)],
    "mono": [(20, 20, 20), (90, 90, 90), (160, 160, 160), (230, 230, 230)],
}

SEAM_MAX = 1.5
DETAIL_MAX = 0.12
TILE_BAND = 0.15


# ---------------------------------------------------------------- quality scores

def seam_score(strip: np.ndarray) -> float:
    """Wrap-around column difference relative to typical neighbour differences.

    Compared with the 90th percentile of all column-to-column differences, so
    patterns with flat areas and occasional sharp edges are not flagged.
    """
    a = strip.astype(np.float32)
    inner = float(np.percentile(np.abs(np.diff(a, axis=1)).mean(axis=(0, 2)), 90))
    wrap = float(np.abs(a[:, 0] - a[:, -1]).mean())
    if inner == 0:
        return 0.0 if wrap == 0 else math.inf
    return float(wrap / inner)


def detail_score(strip: np.ndarray) -> float:
    """Mean horizontal luminance gradient in [0, 1]; high = smears after compression."""
    a = strip.astype(np.float32)
    lum = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    return float(np.abs(np.diff(lum, axis=1)).mean() / 255.0)


# ---------------------------------------------------------------- image patterns

def _tile_axis1(a: np.ndarray) -> np.ndarray:
    w = a.shape[1]
    band = max(1, int(round(w * TILE_BAND)))
    keep = w - band
    out = a[:, :keep].copy()
    alpha = (np.arange(band, dtype=np.float32) / band)[None, :, None]
    out[:, :band] = a[:, :band] * alpha + a[:, keep:keep + band] * (1.0 - alpha)
    return out


def make_tileable(img: np.ndarray) -> np.ndarray:
    """Cross-blend the edges so the image wraps seamlessly in both directions.

    The result is about 15 % smaller in each direction.
    """
    a = img.astype(np.float32)
    a = _tile_axis1(a)
    a = _tile_axis1(a.transpose(1, 0, 2)).transpose(1, 0, 2)
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def load_image_pattern(path: Path, *, repeat_px: int) -> np.ndarray:
    """Turn any image (any colour mode or size) into a seamless pattern strip."""
    try:
        img = Image.open(path)
        img.load()
    except (OSError, UnidentifiedImageError) as e:
        raise ValueError(f"Could not read pattern image {path}: {e}") from None
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, rgba)
    img = img.convert("RGB")
    max_w = 8 * repeat_px
    if img.width > max_w:
        img = img.resize((max_w, max(1, round(img.height * max_w / img.width))), Image.LANCZOS)
    tiled = Image.fromarray(make_tileable(np.asarray(img)))
    new_h = max(1, round(tiled.height * repeat_px / tiled.width))
    strip = np.asarray(tiled.resize((repeat_px, new_h), Image.LANCZOS))
    reps = math.ceil(repeat_px / new_h)
    return np.ascontiguousarray(np.tile(strip, (reps, 1, 1)))


# ---------------------------------------------------------------- generated patterns

def _lattices(rng: np.random.Generator, cells_x: int, cells_y: int, octaves: int) -> list[np.ndarray]:
    return [rng.random((cells_y * 2**o, cells_x * 2**o)) for o in range(octaves)]


def _value_noise(lat: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Smooth periodic value noise; u, v in lattice units (wrap around)."""
    cy, cx = lat.shape
    fu, fv = np.floor(u), np.floor(v)
    x0 = fu.astype(np.int64) % cx
    y0 = fv.astype(np.int64) % cy
    x1, y1 = (x0 + 1) % cx, (y0 + 1) % cy
    tx, ty = u - fu, v - fv
    tx = tx * tx * (3 - 2 * tx)
    ty = ty * ty * (3 - 2 * ty)
    top = lat[y0, x0] * (1 - tx) + lat[y0, x1] * tx
    bot = lat[y1, x0] * (1 - tx) + lat[y1, x1] * tx
    return top * (1 - ty) + bot * ty


def _fbm(lats: list[np.ndarray], u: np.ndarray, v: np.ndarray) -> np.ndarray:
    total = np.zeros_like(u)
    norm = 0.0
    for o, lat in enumerate(lats):
        amp = 0.5**o
        total += amp * _value_noise(lat, u * 2**o, v * 2**o)
        norm += amp
    return total / norm


def _normalise(t: np.ndarray) -> np.ndarray:
    lo, hi = float(t.min()), float(t.max())
    return (t - lo) / (hi - lo) if hi > lo else np.zeros_like(t)


def _palette_map(t: np.ndarray, palette: list[tuple[int, int, int]]) -> np.ndarray:
    cols = np.array(palette, dtype=np.float32)
    pos = np.clip(t, 0, 1) * (len(cols) - 1)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(cols) - 1)
    f = (pos - i0)[..., None]
    return cols[i0] * (1 - f) + cols[i1] * f


def _pingpong(t: np.ndarray, bands: float) -> np.ndarray:
    x = (t * bands) % 2.0
    return np.where(x > 1.0, 2.0 - x, x)


def _wrap_dist(a: np.ndarray, b: float, period: float) -> np.ndarray:
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def generate_pattern(
    style: str,
    *,
    repeat_px: int,
    palette: str = "default",
    seed: int = 0,
    height_px: int = 512,
) -> np.ndarray:
    """Seeded, seamless pattern strip (height_px, repeat_px, 3)."""
    if style not in STYLES:
        raise ValueError(f"Unknown pattern style '{style}'. Choose one of: {', '.join(STYLES)}")
    if palette not in PALETTES:
        raise ValueError(f"Unknown palette '{palette}'. Choose one of: {', '.join(PALETTES)}")
    pal = PALETTES[palette]
    rng = np.random.default_rng(seed)
    h, w = height_px, repeat_px
    cells_x = 4
    cells_y = max(1, round(cells_x * h / w))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    u, v = xs / w * cells_x, ys / h * cells_y
    shade = 0.85 + 0.3 * _fbm(_lattices(rng, cells_x * 2, cells_y * 2, 2), u * 2, v * 2)

    if style == "noise":
        t = _normalise(_fbm(_lattices(rng, cells_x, cells_y, 3), u, v))
        rgb = _palette_map(_pingpong(t, 2.0), pal)
    elif style == "swirl":
        warp_x = _fbm(_lattices(rng, cells_x, cells_y, 2), u, v) - 0.5
        warp_y = _fbm(_lattices(rng, cells_x, cells_y, 2), u, v) - 0.5
        t = _normalise(_fbm(_lattices(rng, cells_x, cells_y, 3), u + 3.0 * warp_x, v + 3.0 * warp_y))
        rgb = _palette_map(_pingpong(t, 3.0), pal)
    elif style == "stripes":
        n = _fbm(_lattices(rng, cells_x, cells_y, 2), u, v)
        k, m = 3, max(1, round(2 * h / w / 3))
        t = 0.5 + 0.5 * np.sin(2 * np.pi * (k * xs / w + m * ys / h) + 4.0 * n)
        rgb = _palette_map(t, pal)
    elif style == "cells":
        n_pts = max(4, round(w * h / 30.0**2))
        px, py = rng.random(n_pts) * w, rng.random(n_pts) * h
        colours = np.array(pal, dtype=np.float32)[rng.integers(0, len(pal), n_pts)]
        d1 = np.full((h, w), np.inf)
        d2 = np.full((h, w), np.inf)
        idx = np.zeros((h, w), dtype=np.int64)
        for i in range(n_pts):
            d = np.hypot(_wrap_dist(xs, px[i], w), _wrap_dist(ys, py[i], h))
            closer = d < d1
            d2 = np.where(closer, d1, np.minimum(d2, d))
            idx = np.where(closer, i, idx)
            d1 = np.where(closer, d, d1)
        edge = np.clip((d2 - d1) / 4.0, 0.0, 1.0)[..., None]
        rgb = colours[idx] * (0.45 + 0.55 * edge)
    else:  # dots
        bg = _normalise(_fbm(_lattices(rng, cells_x, cells_y, 2), u, v))
        rgb = _palette_map(bg * 0.5, pal) * 0.7
        nx = 4
        ny = max(1, round(h / (w / nx)))
        sx, sy = w / nx, h / ny
        jitter = (rng.random((ny, nx, 2)) - 0.5) * 0.4
        dot_cols = np.array(pal, dtype=np.float32)[rng.integers(0, len(pal), (ny, nx))]
        radius = 0.3 * min(sx, sy)
        for j in range(ny):
            for i in range(nx):
                cx = (i + 0.5 + jitter[j, i, 0]) * sx
                cy = (j + 0.5 + jitter[j, i, 1]) * sy
                d = np.hypot(_wrap_dist(xs, cx, w), _wrap_dist(ys, cy, h))
                a = np.clip((radius - d) / 1.5 + 0.5, 0.0, 1.0)[..., None]
                rgb = rgb * (1 - a) + dot_cols[j, i] * a

    rgb = rgb * shade[..., None]
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)
