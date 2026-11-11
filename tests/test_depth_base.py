import json

import numpy as np
import pytest
from PIL import Image

from engine.depth.base import ArrayDepthSequence, PngDepthSequence


def _write_seq(d, frames, reveal=True):
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames, start=1):
        Image.fromarray(np.round(f * 65535).astype(np.uint16)).save(d / f"depth_{i:04d}.png")
    if reveal:
        Image.fromarray(np.full((f.shape[0], f.shape[1], 3), 128, np.uint8)).save(d / "reveal.png")
    (d / "meta.json").write_text(json.dumps({"frames": len(frames)}))


def test_png_round_trip(tmp_path):
    f = np.linspace(0, 1, 40 * 30, dtype=np.float32).reshape(30, 40)
    _write_seq(tmp_path / "seq dir", [f, 1 - f])
    seq = PngDepthSequence(tmp_path / "seq dir", fps=30)
    assert len(seq) == 2 and (seq.width, seq.height, seq.fps) == (40, 30, 30)
    got = seq.frame(0)
    assert got.dtype == np.float32 and np.abs(got - f).max() <= 1 / 65535
    assert seq.reveal.shape == (30, 40, 3) and seq.reveal.dtype == np.uint8


def test_png_without_reveal(tmp_path):
    _write_seq(tmp_path, [np.zeros((4, 4), np.float32)], reveal=False)
    assert PngDepthSequence(tmp_path).reveal is None


def test_out_of_range(tmp_path):
    _write_seq(tmp_path, [np.zeros((4, 4), np.float32)])
    with pytest.raises(IndexError):
        PngDepthSequence(tmp_path).frame(1)


def test_array_sequence():
    frames = [np.zeros((10, 20), np.float32), np.ones((10, 20), np.float32)]
    seq = ArrayDepthSequence(frames, fps=24)
    assert len(seq) == 2 and (seq.width, seq.height, seq.fps) == (20, 10, 24)
    assert seq.frame(1).max() == 1.0 and seq.reveal is None
    with pytest.raises(IndexError):
        seq.frame(2)
