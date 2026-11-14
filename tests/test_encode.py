import numpy as np
import pytest

from engine.encode import (
    VideoWriter,
    concat_music,
    copy_silent,
    list_music,
    mux_music,
    pick_music,
    probe,
)
from tests.conftest import make_sine


def _video(path, seconds=1.0, size=64):
    with VideoWriter(path, size, size, fps=30, preset="ultrafast") as w:
        for i in range(round(seconds * 30)):
            w.write(np.full((size, size, 3), i * 7 % 255, np.uint8))
    return path


@pytest.fixture
def d(tmp_path):
    p = tmp_path / "a dir with spaces"
    p.mkdir()
    return p


def test_writer_produces_h264(d):
    v = _video(d / "out.mp4")
    info = probe(v)
    assert info.duration == pytest.approx(1.0, abs=0.05)
    assert (info.width, info.height, info.has_audio) == (64, 64, False)
    assert not list(d.glob("*.part*"))


def test_writer_cleans_up_on_error(d):
    with pytest.raises(RuntimeError, match="boom"):
        with VideoWriter(d / "bad.mp4", 64, 64) as w:
            w.write(np.zeros((64, 64, 3), np.uint8))
            raise RuntimeError("boom")
    assert list(d.iterdir()) == []


def test_writer_rejects_wrong_frame_size(d):
    with pytest.raises(ValueError, match="64x64"):
        with VideoWriter(d / "bad.mp4", 64, 64) as w:
            w.write(np.zeros((32, 32, 3), np.uint8))
    assert list(d.iterdir()) == []


def test_music_filter_ignores_junk(d):
    for name in (".DS_Store", "notes.txt", "a.mp3", "B.wav", "._a.mp3"):
        (d / name).write_bytes(b"x")
    (d / "sub").mkdir()
    assert [p.name for p in list_music(d)] == ["a.mp3", "B.wav"]


def test_pick_music_empty_dir_returns_none(d):
    assert pick_music(d, "x") is None
    assert pick_music(d / "missing", "x") is None


def test_pick_music_deterministic(d):
    for name in ("a.wav", "b.wav", "c.wav"):
        (d / name).write_bytes(b"x")
    assert pick_music(d, "scene-1") == pick_music(d, "scene-1")
    assert {pick_music(d, f"s{i}").name for i in range(30)} == {"a.wav", "b.wav", "c.wav"}


def test_mux_music_trims_and_has_audio(d):
    v = _video(d / "v.mp4", 1.0)
    out = d / "with music.mp4"
    mux_music(v, make_sine(d / "tone 1.wav", 3), out, duration=1.0)
    info = probe(out)
    assert info.has_audio and info.duration == pytest.approx(1.0, abs=0.1)


def test_mux_loops_short_music(d):
    v = _video(d / "v.mp4", 3.0)
    out = d / "looped.mp4"
    mux_music(v, make_sine(d / "short.wav", 1), out, duration=3.0)
    info = probe(out)
    assert info.has_audio and info.duration == pytest.approx(3.0, abs=0.1)


def test_copy_silent(d):
    v = _video(d / "v.mp4", 1.0)
    m = d / "m.mp4"
    mux_music(v, make_sine(d / "t.wav", 2), m, duration=1.0)
    s = d / "silent.mp4"
    copy_silent(m, s)
    info = probe(s)
    assert not info.has_audio and info.duration == pytest.approx(1.0, abs=0.1)


def test_concat_music(d):
    tracks = [make_sine(d / "one.wav", 1, 440), make_sine(d / "two.wav", 1, 660)]
    out = d / "mix.wav"
    concat_music(tracks, out, duration=1.5)
    info = probe(out)
    assert info.has_audio and info.duration == pytest.approx(1.5, abs=0.1)
