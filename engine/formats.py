"""Output formats: frame size, pattern repeat width and platform-safe area."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SafeArea:
    """Fractions of the frame kept free of the hidden object (platform UI)."""

    top: float
    bottom: float
    left: float
    right: float

    def rect(self, width: int, height: int) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) in pixels of the area the object may occupy."""
        return (
            round(self.left * width),
            round(self.top * height),
            round((1 - self.right) * width),
            round((1 - self.bottom) * height),
        )


@dataclass(frozen=True)
class Format:
    name: str
    width: int
    height: int
    repeat_px: int
    safe: SafeArea


# The hidden object is centred. Shorts: equal margins keep it clear of the apps'
# side buttons (right) and of both text lines (top 12 %, bottom 80 %).
SHORT = Format("short", 1080, 1920, 154, SafeArea(0.20, 0.20, 0.15, 0.15))
LONG = Format("long", 1920, 1080, 175, SafeArea(0.08, 0.08, 0.10, 0.10))
_FORMATS = {"short": SHORT, "long": LONG}


def get_format(name: str) -> Format:
    try:
        return _FORMATS[name]
    except KeyError:
        raise ValueError(f"Unknown format '{name}'. Choose one of: {', '.join(_FORMATS)}") from None


def preview_of(fmt: Format) -> Format:
    """Half-resolution version of a format, used by `stereo preview`."""
    return replace(
        fmt,
        name=f"{fmt.name}-preview",
        width=fmt.width // 2,
        height=fmt.height // 2,
        repeat_px=fmt.repeat_px // 2,
    )
