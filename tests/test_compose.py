from pathlib import Path

import numpy as np
import pytest
import yaml

from engine.compose import (
    END_SHOW,
    LONG_FADE_OUT,
    LONG_REVEAL,
    SegmentComposer,
    ShortComposer,
    card,
    fit_reveal,
    short_alphas,
    thumbnail,
)
from engine.formats import LONG, SHORT
from engine.scene import load_scene

FONT = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Montserrat-ExtraBold.ttf"


def _scene(project, **data):
    p = project.scenes / "s.yaml"
    p.write_text(yaml.safe_dump({"id": "s", "shape": "heart", **data}))
    return load_scene(p, project)


def test_main_text_timeline():
    assert short_alphas(0, 30, has_end=True, reveal=False).main == 1
    assert short_alphas(4.25, 30, has_end=True, reveal=False).main == pytest.approx(0.5)
    assert short_alphas(10, 30, has_end=True, reveal=False).main == 0


def test_end_text_last_three_seconds():
    assert short_alphas(30 - END_SHOW - 0.1, 30, has_end=True, reveal=False).end == 0
    assert short_alphas(27.6, 30, has_end=True, reveal=False).end == 1


def test_no_end_text_when_disabled():
    assert short_alphas(29, 30, has_end=False, reveal=False).end == 0


def test_reveal_replaces_end_text():
    for t in (27.5, 28, 29.9):
        a = short_alphas(t, 30, has_end=True, reveal=True)
        assert a.end == 0
    assert short_alphas(28, 30, has_end=True, reveal=True).reveal == 1
    assert short_alphas(20, 30, has_end=True, reveal=True).reveal == 0


def test_short_duration_overlap():
    for t in np.arange(0, 3, 0.05):
        a = short_alphas(float(t), 3, has_end=True, reveal=False)
        assert not (a.main > 0 and a.end > 0)


def _changed_rows(frame, base):
    return np.nonzero(np.any(frame != base, axis=(1, 2)))[0]


def _bottom_rows(comp, h=1920):
    black = np.zeros((h, 1080, 3), np.uint8)
    rows = _changed_rows(comp.frame(black, 0.0), black)
    return rows[rows > h / 2]


def test_text_stays_in_bands(tmp_project):
    comp = ShortComposer(_scene(tmp_project), SHORT, FONT, None)
    black = np.zeros((1920, 1080, 3), np.uint8)
    rows = _changed_rows(comp.frame(black, 0.0), black)
    assert len(rows) > 0
    h = 1920
    ok = ((rows >= 0.08 * h) & (rows <= 0.16 * h)) | ((rows >= 0.76 * h) & (rows <= 0.84 * h))
    assert ok.all()


@pytest.mark.parametrize("platform,centre", [("tiktok", 0.80), ("reels", 0.80), ("shorts", 0.80),
                                             ("facebook", 0.80), ("clean", 0.88)])
def test_bottom_text_position_per_platform(tmp_project, platform, centre):
    rows = _bottom_rows(ShortComposer(_scene(tmp_project, platform=platform), SHORT, FONT, None))
    assert (rows.min() + rows.max()) / 2 / 1920 == pytest.approx(centre, abs=0.006)


def test_bottom_text_position_override(tmp_project):
    rows = _bottom_rows(ShortComposer(_scene(tmp_project, text={"bottom_y": 0.85}), SHORT, FONT, None))
    assert (rows.min() + rows.max()) / 2 / 1920 == pytest.approx(0.85, abs=0.006)


def test_text_is_readable_on_busy_background(tmp_project):
    comp = ShortComposer(_scene(tmp_project), SHORT, FONT, None)
    grey = np.full((1920, 1080, 3), 128, np.uint8)
    band = comp.frame(grey, 0.0)[int(0.08 * 1920):int(0.16 * 1920)]
    assert band.max() >= 250  # solid white letters
    assert band.min() <= 60  # dark outline/shadow around them


def test_text_gone_in_the_middle(tmp_project):
    comp = ShortComposer(_scene(tmp_project), SHORT, FONT, None)
    stereo = np.random.default_rng(0).integers(0, 255, (1920, 1080, 3), dtype=np.uint8)
    assert np.array_equal(comp.frame(stereo, 15.0), stereo)


def test_disabled_text_lines(tmp_project):
    scene = _scene(tmp_project, text={"top": None, "bottom": None, "end": None})
    comp = ShortComposer(scene, SHORT, FONT, None)
    black = np.zeros((1920, 1080, 3), np.uint8)
    assert np.array_equal(comp.frame(black, 0.0), black)
    assert np.array_equal(comp.frame(black, 29.5), black)


def test_short_countdown_top_right_only_with_reveal(tmp_project):
    black = np.zeros((1920, 1080, 3), np.uint8)
    reveal = np.full((1920, 1080, 3), 200, np.uint8)
    scene = _scene(tmp_project, reveal="on", text={"top": None, "bottom": None, "end": None})
    pts = _changed(ShortComposer(scene, SHORT, FONT, reveal).frame(black, 10.0))
    assert len(pts) > 0
    assert pts[:, 0].min() > 0.14 * 1920 and pts[:, 0].max() < 0.27 * 1920 and pts[:, 1].min() > 0.78 * 1080
    no_reveal = _scene(tmp_project, text={"top": None, "bottom": None, "end": None})
    assert np.array_equal(ShortComposer(no_reveal, SHORT, FONT, None).frame(black, 10.0), black)


def test_short_reveal_at_end(tmp_project):
    reveal = np.full((1920, 1080, 3), 200, np.uint8)
    comp = ShortComposer(_scene(tmp_project, reveal="on"), SHORT, FONT, reveal)
    out = comp.frame(np.zeros((1920, 1080, 3), np.uint8), 29.0)
    assert np.array_equal(out, reveal)


def _segment():
    reveal = np.full((1080, 1920, 3), 90, np.uint8)
    return SegmentComposer(LONG, FONT, index=3, total=15, title="Hidden heart", stereo_seconds=10.0,
                           reveal_img=reveal), reveal


def test_segment_total_seconds():
    seg, _ = _segment()
    assert seg.total_seconds == 10.0 + LONG_REVEAL + LONG_FADE_OUT


def _changed(out):
    return np.argwhere(np.any(out != 0, axis=2))


def test_segment_counter_and_countdown_ring():
    seg, _ = _segment()
    black = np.zeros((1080, 1920, 3), np.uint8)
    pts = _changed(seg.frame(black, 5.0))
    top_left = pts[(pts[:, 0] < 0.15 * 1080) & (pts[:, 1] < 0.15 * 1920)]
    bottom_right = pts[(pts[:, 0] > 0.80 * 1080) & (pts[:, 1] > 0.85 * 1920)]
    assert len(top_left) > 0 and len(bottom_right) > 0
    assert len(top_left) + len(bottom_right) == len(pts)  # nothing else on screen (no progress bar)


def test_countdown_ring_runs_out():
    seg, _ = _segment()
    black = np.zeros((1080, 1920, 3), np.uint8)
    corner = (slice(int(0.8 * 1080), 1080), slice(int(0.85 * 1920), 1920))
    early = seg.frame(black, 0.5)[corner]
    late = seg.frame(black, 9.0)[corner]
    assert (early > 200).sum() > 1.5 * (late > 200).sum()  # the bright arc shrinks
    after = seg.frame(black, 11.0)
    assert np.array_equal(after[corner], seg.frame(np.full_like(black, 255), 11.0)[corner])  # gone at reveal


def test_segment_reveal_after_stereo():
    seg, reveal = _segment()
    a = seg.frame(np.zeros((1080, 1920, 3), np.uint8), 11.0)
    b = seg.frame(np.full((1080, 1920, 3), 255, np.uint8), 11.0)
    assert np.array_equal(a, b)  # fully revealed: stereogram no longer visible


def test_reveal_title_sits_below_the_model():
    red = np.zeros((1080, 1920, 3), np.uint8)
    red[..., 0] = 250
    seg = SegmentComposer(LONG, FONT, index=1, total=2, title="Hidden heart", stereo_seconds=2.0, reveal_img=red)
    out = seg.frame(np.zeros_like(red), 3.0)
    strip_top = round(0.86 * 1080)
    model_rows = np.nonzero((out[:, 300:1600] == (250, 0, 0)).all(axis=2).any(axis=1))[0]
    assert model_rows.max() < strip_top  # the reveal image stays above the strip
    title = np.argwhere(np.any(out[:, 300:1600] != fit_reveal(red, LONG, bottom_strip=0.14)[:, 300:1600], axis=2))
    assert len(title) > 0 and title[:, 0].min() >= strip_top


def test_segment_fades_to_black():
    seg, _ = _segment()
    out = seg.frame(np.full((1080, 1920, 3), 255, np.uint8), seg.total_seconds - 0.01)
    assert out.mean() < 5


def test_fit_reveal_letterboxes():
    out = fit_reveal(np.full((100, 100, 3), 255, np.uint8), LONG)
    assert out.shape == (1080, 1920, 3) and out[540, 960].min() == 255 and out[540, 10].max() < 255
    strip = fit_reveal(np.full((100, 100, 3), 255, np.uint8), LONG, bottom_strip=0.14)
    assert strip[1070, 960].max() < 255 and strip[400, 960].min() == 255


def test_card():
    img = card(["Can you see it?", "Relax your eyes"], LONG, FONT)
    assert img.shape == (1080, 1920, 3) and img.max() > 200 and tuple(img[5, 5]) == (12, 12, 14)


@pytest.mark.parametrize("shape", [(1920, 1080, 3), (1080, 1920, 3)])
def test_thumbnail_size(shape):
    t = thumbnail(np.zeros(shape, np.uint8), "Hidden heart", FONT)
    assert t.shape == (720, 1280, 3) and t.max() > 200
