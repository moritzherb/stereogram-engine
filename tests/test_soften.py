import numpy as np

from engine.depth.soften import soften, softness_sigma
from engine.formats import SHORT, preview_of


def test_flat_stays_flat():
    d = np.full((50, 80), 0.4, np.float32)
    assert np.allclose(soften(d, 6.0), 0.4, atol=1e-5)


def test_step_becomes_gradual():
    d = np.zeros((60, 200), np.float32)
    d[:, 100:] = 1.0
    s = soften(d, 8.0)
    assert np.abs(np.diff(s, axis=1)).max() < 0.15
    assert s.min() >= 0 and s.max() <= 1 and s.dtype == np.float32
    assert s[30, 10] < 0.01 and s[30, 190] > 0.99  # far from the edge nothing changes


def test_zero_sigma_is_identity():
    d = np.random.default_rng(0).random((20, 30)).astype(np.float32)
    assert np.array_equal(soften(d, 0.0), d)


def test_sigma_scales_with_frame_width():
    assert softness_sigma(1.0, SHORT) == 2 * softness_sigma(1.0, preview_of(SHORT))
    assert softness_sigma(0.0, SHORT) == 0.0


def test_limit_slope_turns_steps_into_ramps():
    from engine.depth.soften import limit_slope

    d = np.zeros((40, 200), np.float32)
    d[:, 100:] = 0.9
    out = limit_slope(d, 0.02)
    assert np.abs(np.diff(out, axis=1)).max() <= 0.02 + 1e-6
    assert (out >= d).all()  # only raises the low side: the object keeps its full height
    assert out[20, 150] == np.float32(0.9) and out[20, 10] == 0  # far from the step nothing changes
