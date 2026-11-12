import numpy as np
import pytest

from engine.depth.base import ArrayDepthSequence
from engine.formats import LONG, SHORT
from engine.longform import MAX_GROWTH, LongFromShort


def _short_disc(cx=540, cy=960, rx=200, ry=300, frames=3):
    yy, xx = np.mgrid[0:SHORT.height, 0:SHORT.width]
    out = []
    for i in range(frames):
        r2 = ((xx - cx - 4 * i) / rx) ** 2 + ((yy - cy) / ry) ** 2
        out.append(np.sqrt(np.clip(1 - r2, 0, 1)).astype(np.float32))
    reveal = np.full((SHORT.height, SHORT.width, 3), 194, np.uint8)
    reveal[out[0] > 0] = 255
    return ArrayDepthSequence(out, fps=30, reveal=reveal)


def _bbox(f):
    ys, xs = np.nonzero(f > 0.02)
    return xs.min(), ys.min(), xs.max(), ys.max()


def test_long_frame_size_and_centre():
    long = LongFromShort(_short_disc(), LONG)
    f = long.frame(0)
    assert f.shape == (LONG.height, LONG.width) and f.dtype == np.float32
    x0, y0, x1, y1 = _bbox(f)
    assert (x0 + x1) / 2 == pytest.approx(LONG.width / 2, abs=12)
    assert (y0 + y1) / 2 == pytest.approx(LONG.height / 2, abs=12)
    assert f[:, :200].max() == 0 and f[:, -200:].max() == 0  # more background at the sides


def test_model_is_never_stretched():
    long = LongFromShort(_short_disc(rx=200, ry=300), LONG)
    sx0, sy0, sx1, sy1 = _bbox(_short_disc().frame(0))
    lx0, ly0, lx1, ly1 = _bbox(long.frame(0))
    short_aspect = (sx1 - sx0) / (sy1 - sy0)
    long_aspect = (lx1 - lx0) / (ly1 - ly0)
    assert long_aspect == pytest.approx(short_aspect, rel=0.03)


def test_small_model_grows_at_most_a_little():
    long = LongFromShort(_short_disc(rx=100, ry=100), LONG)
    assert long.scale == pytest.approx(MAX_GROWTH)
    sx0, _, sx1, _ = _bbox(_short_disc(rx=100, ry=100).frame(0))
    lx0, _, lx1, _ = _bbox(long.frame(0))
    assert (lx1 - lx0) / (sx1 - sx0) == pytest.approx(MAX_GROWTH, rel=0.05)


def test_tall_model_shrinks_to_fit_inside_the_safe_area():
    long = LongFromShort(_short_disc(rx=250, ry=560), LONG)
    x0, y0, x1, y1 = LONG.safe.rect(LONG.width, LONG.height)
    bx0, by0, bx1, by1 = _bbox(long.frame(0))
    assert by0 >= y0 - 2 and by1 <= y1 + 2 and long.scale < 1


def test_same_scale_and_position_for_every_frame():
    long = LongFromShort(_short_disc(frames=5), LONG)
    widths = [(_bbox(long.frame(i))[2] - _bbox(long.frame(i))[0]) for i in range(5)]
    assert max(widths) - min(widths) <= 2
    lefts = [_bbox(long.frame(i))[0] for i in range(5)]
    assert lefts == sorted(lefts)  # the small motion is kept, not re-centred per frame


def test_reveal_placed_the_same_way():
    long = LongFromShort(_short_disc(), LONG)
    r = long.reveal
    assert r.shape == (LONG.height, LONG.width, 3)
    assert tuple(r[5, 5]) == (194, 194, 194)  # the reveal's own background colour, no visible seam
    assert r[LONG.height // 2, LONG.width // 2].min() == 255
    assert len(long) == 3 and long.fps == 30


def test_transform_applied_before_placing():
    seq = _short_disc()
    long = LongFromShort(seq, LONG, transform=lambda d: np.where(d > 0, 0.5, 0).astype(np.float32))
    assert long.frame(0).max() == pytest.approx(0.5, abs=0.01)
