import pytest

from engine.formats import LONG, SHORT, get_format, preview_of


def test_short_and_long_constants():
    assert (SHORT.width, SHORT.height, SHORT.repeat_px) == (1080, 1920, 154)
    assert (LONG.width, LONG.height, LONG.repeat_px) == (1920, 1080, 154)  # same pattern size as Shorts


def test_safe_areas_are_centred():
    for fmt in (SHORT, LONG):
        s = fmt.safe
        assert s.left == s.right and s.top == s.bottom
    assert (SHORT.safe.top, SHORT.safe.left) == (0.20, 0.15)  # clear of app buttons and both text lines
    assert (LONG.safe.top, LONG.safe.left) == (0.08, 0.10)


def test_preview_halves():
    p = preview_of(SHORT)
    assert (p.width, p.height, p.repeat_px, p.name) == (540, 960, 77, "short-preview")
    assert p.safe == SHORT.safe


def test_get_format():
    assert get_format("short") is SHORT and get_format("long") is LONG


def test_unknown_format():
    with pytest.raises(ValueError, match="short, long"):
        get_format("square")
