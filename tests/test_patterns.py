import numpy as np
import pytest
from PIL import Image

from engine.patterns import (
    DETAIL_MAX,
    PALETTES,
    SEAM_MAX,
    STYLES,
    detail_score,
    generate_pattern,
    load_image_pattern,
    make_tileable,
    seam_score,
)


@pytest.mark.parametrize("style", STYLES)
def test_generated_shape_and_tileable(style):
    s = generate_pattern(style, repeat_px=154, seed=3)
    assert s.shape == (512, 154, 3) and s.dtype == np.uint8
    assert seam_score(s) <= SEAM_MAX
    assert seam_score(s.transpose(1, 0, 2)) <= SEAM_MAX  # vertical seam too


@pytest.mark.parametrize("style", STYLES)
def test_generated_not_too_fine(style):
    assert detail_score(generate_pattern(style, repeat_px=154, seed=3)) <= DETAIL_MAX


def test_seed_determinism():
    a = generate_pattern("swirl", repeat_px=154, seed=1)
    assert np.array_equal(a, generate_pattern("swirl", repeat_px=154, seed=1))
    assert not np.array_equal(a, generate_pattern("swirl", repeat_px=154, seed=2))


def test_palettes_present():
    assert {"default", "ocean", "sunset", "forest", "candy", "mono"} <= set(PALETTES)
    s = generate_pattern("noise", repeat_px=154, palette="ocean", seed=1)
    assert s.shape == (512, 154, 3)


def test_unknown_style_lists_choices():
    with pytest.raises(ValueError, match="noise, swirl"):
        generate_pattern("plaid", repeat_px=154)


def test_unknown_palette_lists_choices():
    with pytest.raises(ValueError, match="ocean"):
        generate_pattern("noise", repeat_px=154, palette="rainbowz")


def _photo(tmp_path, mode, size, name="my photo.png"):
    rng = np.random.default_rng(0)
    w, h = size
    # left half dark, right half bright: a hard wrap-around seam
    arr = np.zeros((h, w, 3), np.uint8)
    arr[:, : w // 2] = 30
    arr[:, w // 2:] = 220
    arr = np.clip(arr.astype(int) + rng.integers(-10, 10, arr.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).convert(mode)
    path = tmp_path / (name if mode != "CMYK" else "my photo.jpg")
    img.save(path)
    return path


@pytest.mark.parametrize("mode", ["RGBA", "L", "CMYK", "P"])
def test_image_modes_become_rgb(tmp_path, mode):
    s = load_image_pattern(_photo(tmp_path, mode, (300, 200)), repeat_px=154)
    assert s.shape[1:] == (154, 3) and s.dtype == np.uint8 and s.shape[0] >= 154


@pytest.mark.parametrize("size", [(8000, 6000), (60, 40)])
def test_huge_and_tiny_images(tmp_path, size):
    s = load_image_pattern(_photo(tmp_path, "RGB", size), repeat_px=154)
    assert s.shape[1:] == (154, 3) and s.shape[0] >= 154


def test_image_pattern_made_seamless(tmp_path):
    s = load_image_pattern(_photo(tmp_path, "RGB", (400, 600)), repeat_px=154)
    assert seam_score(s) <= SEAM_MAX


def test_make_tileable_wraps_smoothly():
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, 50:] = 255
    t = make_tileable(img)
    assert seam_score(t) <= SEAM_MAX


def test_detail_score_flags_pixel_noise():
    fine = np.random.default_rng(0).integers(0, 256, (512, 154, 3), dtype=np.uint8)
    assert detail_score(fine) > DETAIL_MAX
