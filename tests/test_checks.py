import numpy as np

from engine.checks import check_frames, check_motion, check_pattern, check_safe_area
from engine.depth.base import ArrayDepthSequence
from engine.formats import SHORT, preview_of
from engine.patterns import generate_pattern
from engine.stereo import render_stereogram

FMT = preview_of(SHORT)  # 540x960, same safe-area fractions
W, H = FMT.width, FMT.height


def disc(cx, cy, r=60):
    yy, xx = np.mgrid[0:H, 0:W]
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.float32)


def codes(warnings):
    return [w.code for w in warnings]


def test_centred_static_disc_is_fine():
    seq = ArrayDepthSequence([disc(W * 0.45, H * 0.45)] * 10)
    assert check_safe_area(seq, FMT) == [] and check_motion(seq) == []


def test_disc_under_app_buttons():
    seq = ArrayDepthSequence([disc(W * 0.95, H * 0.45)] * 3)
    w = check_safe_area(seq, FMT)
    assert codes(w) == ["OUT_OF_SAFE_AREA"] and "at 0.0 s" in w[0].message


def test_disc_leaving_later_names_the_time():
    frames = [disc(W * 0.45, H * 0.45)] * 30 + [disc(W * 0.45, H * 0.97)] * 30
    w = check_safe_area(ArrayDepthSequence(frames, fps=30), FMT)
    assert "at 1.0 s" in w[0].message


def test_fast_motion_flagged():
    frames = [disc(100 + (i % 2) * 200, H * 0.45) for i in range(10)]
    w = check_motion(ArrayDepthSequence(frames))
    assert codes(w) == ["MOTION_TOO_FAST"] and "speed" in w[0].message


def test_slow_motion_fine():
    frames = [disc(150 + 2 * i, H * 0.45) for i in range(30)]
    assert check_motion(ArrayDepthSequence(frames)) == []


def test_pattern_checks():
    noisy = np.random.default_rng(0).integers(0, 256, (512, 154, 3), dtype=np.uint8)
    assert "PATTERN_TOO_FINE" in codes(check_pattern(noisy))
    seam = np.zeros((512, 154, 3), np.uint8)
    seam[:, 77:] = 255
    assert "PATTERN_SEAMS" in codes(check_pattern(seam))
    assert check_pattern(generate_pattern("swirl", repeat_px=154)) == []


def test_self_check_frames():
    d = disc(W / 2, H / 2, 150)
    pat = generate_pattern("noise", repeat_px=FMT.repeat_px, seed=1)
    img = render_stereogram(d, pat, repeat_px=FMT.repeat_px)
    assert check_frames([d], [img], repeat_px=FMT.repeat_px, depth_strength=0.33) == []
    bad = check_frames([np.zeros_like(d) + 0.5], [img], repeat_px=FMT.repeat_px, depth_strength=0.33)
    assert codes(bad) == ["SELF_CHECK_FAILED"]


def test_self_check_crosseyed():
    d = disc(W / 2, H / 2, 150)
    pat = generate_pattern("noise", repeat_px=FMT.repeat_px, seed=1)
    img = render_stereogram(d, pat, repeat_px=FMT.repeat_px, crosseyed=True)
    assert check_frames([d], [img], repeat_px=FMT.repeat_px, depth_strength=0.33, crosseyed=True) == []
