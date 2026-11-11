import numpy as np

from engine.stereo import render_stereogram, separation


def test_separation_far_plane_equals_half_eye_distance():
    assert separation(0.0, 308, 0.33) == 154
    assert round(separation(1.0, 308, 0.33)) == 124


def test_output_shape_dtype(strip):
    out = render_stereogram(np.zeros((64, 1080), np.float32), strip, repeat_px=154)
    assert out.shape == (64, 1080, 3) and out.dtype == np.uint8


def test_flat_depth_is_pure_repetition(strip):
    out = render_stereogram(np.zeros((64, 1080), np.float32), strip, repeat_px=154)
    assert np.array_equal(out[:, 154:], out[:, :-154])


def test_flat_depth_anchored_at_centre(strip):
    out = render_stereogram(np.zeros((64, 1080), np.float32), strip, repeat_px=154)
    assert np.array_equal(out[:, 540], strip[np.arange(64) % strip.shape[0], 0])


def test_raised_square_has_expected_separation(strip):
    d = np.zeros((64, 1080), np.float32)
    d[:, 400:700] = 1.0
    out = render_stereogram(d, strip, repeat_px=154, oversample=1)
    s1 = round(separation(1.0, 308, 0.33))
    assert np.array_equal(out[10, 560:600], out[10, 560 - s1:600 - s1])
    # and the far plane still repeats at 154 well left of the object
    assert np.array_equal(out[10, 154:300], out[10, 0:146])


def test_identical_depth_gives_identical_frames(strip):
    d = np.zeros((64, 1080), np.float32)
    d[20:40, 300:600] = 0.7
    a = render_stereogram(d, strip, repeat_px=154)
    b = render_stereogram(d.copy(), strip, repeat_px=154)
    assert np.array_equal(a, b)


def test_static_rows_unchanged_when_object_moves_elsewhere(strip):
    d1 = np.zeros((64, 1080), np.float32)
    d2 = d1.copy()
    d1[0:20, 300:500] = 1.0
    d2[0:20, 340:540] = 1.0
    a = render_stereogram(d1, strip, repeat_px=154)
    b = render_stereogram(d2, strip, repeat_px=154)
    assert np.array_equal(a[40:], b[40:])
    assert not np.array_equal(a[:20], b[:20])


def test_crosseyed_inverts_depth(strip):
    d = np.zeros((64, 1080), np.float32)
    d[:, 400:700] = 0.8
    a = render_stereogram(d, strip, repeat_px=154, crosseyed=True)
    b = render_stereogram(1 - d, strip, repeat_px=154)
    assert np.array_equal(a, b)
