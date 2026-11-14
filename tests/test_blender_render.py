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


def test_reveal_uses_widest_pose(tmp_path):
    proc, out = run_job(tmp_path, {"text3d": "HELLO"}, motion="spin", reveal=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["reveal_frame"] in (0, 1, 2)  # near face-on, not the middle frame (5) turned 30 degrees away


def _centroid(f):
    ys, xs = np.nonzero(f > 0.02)
    return xs.mean(), ys.mean()


@pytest.mark.parametrize("motion", ["sway", "tilt", "rock"])
def test_turning_motions_stay_in_place(tmp_path, motion):
    proc, out = run_job(tmp_path, {"shape": "heart"}, motion=motion, duration=2, motion_seconds=2)
    assert proc.returncode == 0, proc.stderr[-2000:]
    fr, _ = frames(out)
    xs, ys = zip(*(_centroid(f) for f in fr))
    assert max(xs) - min(xs) < 0.06 * W and max(ys) - min(ys) < 0.06 * H
    assert not np.allclose(fr[0], fr[5], atol=0.02)  # it does move (depth changes)


def test_motion_amount_controls_strength(tmp_path):
    changes = []
    for amount in (8, 40):
        d = tmp_path / str(amount)
        d.mkdir()
        proc, out = run_job(d, {"shape": "heart"}, motion="sway", duration=2, motion_seconds=2,
                            motion_amount=amount)
        assert proc.returncode == 0, proc.stderr[-2000:]
        fr, _ = frames(out)
        changes.append(np.abs(fr[0] - fr[5]).mean())
    assert changes[1] > 2 * changes[0]


def test_default_float_is_subtle(tmp_path):
    proc, out = run_job(tmp_path, {"shape": "heart"}, motion="float", duration=2)
    assert proc.returncode == 0, proc.stderr[-2000:]
    fr, _ = frames(out)
    ys = [_centroid(f)[1] for f in fr]
    assert max(ys) - min(ys) < 0.04 * H


def _roundness(f):
    obj = f > 0.02
    top = f[obj].max()
    slope = ((f > 0.02) & (f < 0.9 * top)).sum() / obj.sum()
    ys, xs = np.nonzero(obj)
    fill = obj.sum() / ((np.ptp(xs) + 1) * (np.ptp(ys) + 1))
    return slope, fill


@pytest.mark.parametrize("src,min_slope", [({"shape": "heart"}, 0.35), ({"shape": "star"}, 0.3)])
def test_solid_rounded_3d(tmp_path, src, min_slope):
    proc, out = run_job(tmp_path, src, width=432, height=768, duration=0.1)
    assert proc.returncode == 0, proc.stderr[-2000:]
    slope, fill = _roundness(frames(out)[0][0])
    assert slope >= min_slope, f"only {slope:.0%} of the surface is rounded"


def test_text_letters_are_solid_blocks(tmp_path):
    proc, out = run_job(tmp_path, {"text3d": "HELLO"}, width=432, height=768, duration=0.1)
    assert proc.returncode == 0, proc.stderr[-2000:]
    f = frames(out)[0][0]
    obj = f > 0.02
    plateau = (f >= 0.9 * f.max())[obj].mean()
    assert plateau >= 0.6, f"only {plateau:.0%} of the letters is at full height"  # flat, solid tops
    slope, fill = _roundness(f)
    assert slope >= 0.05  # but the edges are rounded, not cut-outs
    assert fill >= 0.56, f"letters too thin ({fill:.0%} fill)"  # ExtraBold tubes: 0.55


def test_tilting_text_stays_even(tmp_path):
    proc, out = run_job(tmp_path, {"text3d": "HELLO"}, motion="tilt", duration=2, motion_seconds=2,
                        motion_amount=10)
    assert proc.returncode == 0, proc.stderr[-2000:]
    fr, _ = frames(out)
    for f in fr:
        obj = f > 0.02
        ys, xs = np.nonzero(obj)
        third = (xs.max() - xs.min()) // 3
        left = f[:, xs.min():xs.min() + third][obj[:, xs.min():xs.min() + third]].mean()
        right = f[:, xs.max() - third:xs.max()][obj[:, xs.max() - third:xs.max()]].mean()
        assert abs(left - right) < 0.05  # the whole word stays at one height
        assert f.max() > 0.7  # present at every moment (tilt moves the letter tops back a little)


def test_short_safe_area_centres_the_object(tmp_path):
    from dataclasses import asdict

    from engine.formats import SHORT

    proc, out = run_job(tmp_path, {"shape": "heart"}, safe=asdict(SHORT.safe), duration=0.1)
    assert proc.returncode == 0, proc.stderr[-2000:]
    x0, y0, x1, y1 = bbox(frames(out)[0][0])
    assert (x0 + x1) / 2 / W == pytest.approx(0.5, abs=0.02)
    assert (y0 + y1) / 2 / H == pytest.approx(0.5, abs=0.02)


def _thirds(f, axis):
    obj = f > 0.02
    idx = np.nonzero(obj.any(axis=1 - axis))[0]
    third = (idx.max() - idx.min()) // 3
    a = [slice(None), slice(None)]
    b = [slice(None), slice(None)]
    a[axis] = slice(idx.min(), idx.min() + third)
    b[axis] = slice(idx.max() - third, idx.max())
    return f[tuple(a)][obj[tuple(a)]].mean(), f[tuple(b)][obj[tuple(b)]].mean()


def test_breathing_text_stays_even_and_hidden(tmp_path):
    proc, out = run_job(tmp_path, {"text3d": "HELLO"}, motion="breathe", duration=2, motion_seconds=2,
                        width=216, height=384)
    assert proc.returncode == 0, proc.stderr[-2000:]
    fr, _ = frames(out)
    masks = [f > 0.02 for f in fr]
    for f, m in zip(fr, masks):
        top, bottom = _thirds(f, 0)
        left, right = _thirds(f, 1)
        assert abs(top - bottom) < 0.05 and abs(left - right) < 0.05  # every letter at one height
        assert (m ^ masks[0]).mean() < 0.002  # the outline does not move (stays hidden)
    means = [f[m].mean() for f, m in zip(fr, masks)]
    assert max(means) - min(means) > 0.1  # but the word does move towards and away from the viewer


def test_tilted_text_bottom_does_not_sink(tmp_path):
    proc, out = run_job(tmp_path, {"text3d": "HELLO"}, motion="tilt", duration=2, motion_seconds=2,
                        motion_amount=10, width=216, height=384)
    assert proc.returncode == 0, proc.stderr[-2000:]
    for f in frames(out)[0]:
        top, bottom = _thirds(f, 0)
        assert abs(top - bottom) < 0.25  # a 10 degree nod is no longer exaggerated into a deep slope


def _inner(mask, steps):
    inner = mask.copy()
    for _ in range(steps):
        inner[1:-1, 1:-1] &= inner[:-2, 1:-1] & inner[2:, 1:-1] & inner[1:-1, :-2] & inner[1:-1, 2:]
    return inner[1:-1, 1:-1]


def _laplacian(a):
    a = a.astype(np.float32)
    return np.abs(a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:])


def test_heart_surface_is_smooth(tmp_path):
    proc, out = run_job(tmp_path, {"shape": "heart"}, width=432, height=768, duration=0.04, reveal=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    fr, seq = frames(out)
    f = fr[0]
    lum = seq.reveal.astype(np.float32).mean(axis=2)
    inner = _inner(f > 0.3, 8)
    crease = np.percentile(np.abs(lum[1:-1, 2:] - lum[1:-1, :-2])[inner], 99.5)
    assert crease < 10, f"sharp creases/streaks in the shading ({crease:.1f})"  # was 14.0
    grain = np.median(_laplacian(lum)[inner])
    assert grain <= 1.0, f"grainy answer image ({grain:.2f})"  # was 2.0
