import json

import numpy as np
import pytest

from tests.blender_utils import BLENDER, SAFE, bbox, frames, make_model, run_job

pytestmark = pytest.mark.skipif(BLENDER is None, reason="Blender not installed")
W, H = 108, 192
SAFE_RECT = (SAFE["left"] * W, SAFE["top"] * H, (1 - SAFE["right"]) * W, (1 - SAFE["bottom"]) * H)


def _inside_safe(f, tol=1.5):
    x0, y0, x1, y1 = bbox(f)
    return (x0 >= SAFE_RECT[0] - tol and y0 >= SAFE_RECT[1] - tol
            and x1 <= SAFE_RECT[2] + tol and y1 <= SAFE_RECT[3] + tol)


def test_shape_heart_float_frames(tmp_path):
    proc, out = run_job(tmp_path, {"shape": "heart"}, motion="float", reveal=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    fr, seq = frames(out)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["frames"] == 10 and len(fr) == 10
    for f in fr:
        assert f.max() > 0.5 and f[0, 0] == 0 and f[-1, -1] == 0
    assert seq.reveal is not None and seq.reveal.shape == (H, W, 3)


def test_text3d(tmp_path):
    proc, out = run_job(tmp_path, {"text3d": "HI"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    fr, _ = frames(out)
    assert all(f.max() > 0.5 for f in fr)


@pytest.mark.parametrize("shape", ["sphere", "cube", "torus", "cone", "star"])
def test_every_shape_renders(tmp_path, shape):
    proc, out = run_job(tmp_path, {"shape": shape}, duration=0.2)
    assert proc.returncode == 0, proc.stderr[-2000:]
    fr, _ = frames(out)
    assert fr[0].max() > 0.5


def test_object_inside_safe_area_all_frames(tmp_path):
    proc, out = run_job(tmp_path, {"shape": "cube"}, motion="spin")
    assert proc.returncode == 0, proc.stderr[-2000:]
    fr, _ = frames(out)
    assert all(_inside_safe(f) for f in fr)


@pytest.mark.parametrize("scale,loc", [(500.0, (1000, -300, 50)), (0.01, (0, 0, 0))])
def test_odd_scale_and_offset_model(tmp_path, scale, loc):
    model = make_model(tmp_path, scale=scale, loc=loc)
    proc, out = run_job(tmp_path, {"model": str(model)}, duration=0.3)
    assert proc.returncode == 0, proc.stderr[-2000:]
    fr, _ = frames(out)
    x0, y0, x1, y1 = bbox(fr[0])
    sw, sh = SAFE_RECT[2] - SAFE_RECT[0], SAFE_RECT[3] - SAFE_RECT[1]
    assert (x1 - x0) >= 0.6 * sw or (y1 - y0) >= 0.6 * sh
    assert _inside_safe(fr[0])


def test_depth_normalised_per_scene_not_per_frame(tmp_path):
    proc, out = run_job(tmp_path, {"shape": "sphere"}, motion="fly_toward")
    assert proc.returncode == 0, proc.stderr[-2000:]
    fr, _ = frames(out)
    assert fr[0].max() < fr[-1].max() - 0.2


def test_model_clip_plays(tmp_path):
    model = make_model(tmp_path, clip="Wiggle")
    proc, out = run_job(tmp_path, {"model": str(model)}, animation="Wiggle", duration=1.2)
    assert proc.returncode == 0, proc.stderr[-2000:]
    fr, _ = frames(out)
    assert not np.allclose(fr[0], fr[6], atol=0.02)
    assert "Wiggle" in json.loads((out / "meta.json").read_text())["clips"]


def test_unknown_clip_exit_code(tmp_path):
    model = make_model(tmp_path, clip="Wiggle")
    proc, _ = run_job(tmp_path, {"model": str(model)}, animation="Swimm")
    assert proc.returncode == 3
    assert "UNKNOWN_CLIP: Swimm; available: Wiggle" in proc.stderr


def test_unreadable_model_exit_code(tmp_path):
    bad = tmp_path / "broken.glb"
    bad.write_bytes(b"not a model")
    proc, _ = run_job(tmp_path, {"model": str(bad)})
    assert proc.returncode == 4 and "IMPORT_FAILED" in proc.stderr


def test_swim_past_crosses_frame(tmp_path):
    proc, out = run_job(tmp_path, {"shape": "sphere"}, motion="swim_past")
    assert proc.returncode == 0, proc.stderr[-2000:]
    fr, _ = frames(out)

    def cx(f):
        ys, xs = np.nonzero(f > 0.02)
        return xs.mean()

    assert cx(fr[-1]) - cx(fr[0]) > 0.5 * W
    assert all(_inside_safe(f) for f in fr)
