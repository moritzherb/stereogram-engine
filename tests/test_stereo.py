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


def test_raised_square_has_expected_separation():
    from engine.patterns import generate_pattern

    d = np.zeros((64, 1080), np.float32)
    d[:, 400:700] = 1.0
    pattern = generate_pattern("swirl", repeat_px=154, seed=3)  # fractional shifts need a smooth pattern
    out = render_stereogram(d, pattern, repeat_px=154).astype(np.int16)
    s1 = round(float(separation(1.0, 308, 0.33)))  # 124
    inside = np.abs(out[10, 600:660] - out[10, 600 - s1:660 - s1]).mean()
    background = np.abs(out[10, 600:660] - out[10, 600 - 154:660 - 154]).mean()
    assert inside < 0.25 * background  # inside the object the pattern repeats at the object's separation


def test_smooth_depth_has_no_seams():
    """On smooth depth the pattern must bend smoothly: no seams along depth contours
    (they show as thin grey lines) and no seam at the centre start strip."""
    yy, xx = np.mgrid[0:300, 0:1080]
    dome = np.sqrt(np.clip(1 - ((xx - 540) ** 2 + (yy - 150) ** 2) / 200.0**2, 0, 1)).astype(np.float32) * 0.8
    a = 2 * np.pi * np.arange(154) / 154
    wheel = np.zeros((16, 154, 3), np.uint8)  # colour encodes the pattern position
    wheel[..., 0] = np.round(128 + 120 * np.cos(a))[None, :]
    wheel[..., 1] = np.round(128 + 120 * np.sin(a))[None, :]
    img = render_stereogram(dome, wheel, repeat_px=154)
    ang = np.arctan2(img[..., 1].astype(np.float32) - 128, img[..., 0].astype(np.float32) - 128)
    step = np.abs(np.angle(np.exp(1j * np.diff(ang, axis=1))))
    jumps = (step > 0.2) & (dome[:, 1:] > 0.05)  # a normal step is 2*pi/154 = 0.04
    assert jumps.sum() < 50, f"{jumps.sum()} seams"  # previous method: 8380


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


def test_pattern_shift_moves_whole_pattern(strip):
    flat = np.zeros((64, 1080), np.float32)
    base = render_stereogram(flat, strip, repeat_px=154)
    moved = render_stereogram(flat, strip, repeat_px=154, shift=10)
    assert np.array_equal(moved[:, 10:], base[:, :-10])
    assert np.array_equal(render_stereogram(flat, strip, repeat_px=154, shift=154), base)
