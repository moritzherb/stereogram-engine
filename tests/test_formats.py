import pytest

from engine.formats import LONG, SHORT, get_format, preview_of


def test_short_and_long_constants():
    assert (SHORT.width, SHORT.height, SHORT.repeat_px) == (1080, 1920, 154)
    assert (LONG.width, LONG.height, LONG.repeat_px) == (1920, 1080, 175)


def test_preview_halves():
    p = preview_of(SHORT)
    assert (p.width, p.height, p.repeat_px, p.name) == (540, 960, 77, "short-preview")
    assert p.safe == SHORT.safe


def test_get_format():
    assert get_format("short") is SHORT and get_format("long") is LONG


def test_unknown_format():
    with pytest.raises(ValueError, match="short, long"):
        get_format("square")
