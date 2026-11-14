import numpy as np

from engine.decode import recover_depth, self_check
from engine.stereo import render_stereogram
from tests.conftest import random_strip

PAT = random_strip(200, 154, seed=5)


def _square():
    d = np.zeros((200, 1080), np.float32)
    d[50:150, 400:700] = 1.0
    return d


def _ramp():
    return np.tile(np.linspace(0, 0.8, 1080, dtype=np.float32), (200, 1))


def test_recovers_square():
    d = _square()
    img = render_stereogram(d, PAT, repeat_px=154)
    assert self_check(d, img, repeat_px=154).passed


def test_recovers_ramp():
    d = _ramp()
    img = render_stereogram(d, PAT, repeat_px=154)
    assert self_check(d, img, repeat_px=154).passed


def test_detects_wrong_depth():
    d = _square()
    img = render_stereogram(d, PAT, repeat_px=154)
    result = self_check(np.zeros_like(d), img, repeat_px=154)
    assert not result.passed


def test_recover_depth_shape_and_range():
    d = _square()
    rec = recover_depth(render_stereogram(d, PAT, repeat_px=154), repeat_px=154)
    assert rec.shape == d.shape and rec.dtype == np.float32
    finite = rec[np.isfinite(rec)]
    assert finite.min() >= 0 and finite.max() <= 1
