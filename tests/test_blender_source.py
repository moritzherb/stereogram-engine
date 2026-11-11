from dataclasses import asdict

import pytest
import yaml

from engine.depth.blender_source import BlenderDepthSource, BlenderError, build_job
from engine.formats import SHORT
from engine.scene import load_scene


def _scene(project, name="s.yaml", **data):
    p = project.scenes / name
    p.write_text(yaml.safe_dump({"id": "s", "shape": "heart", "duration": 1, **data}))
    return load_scene(p, project)


def test_job_contents(tmp_project):
    job = build_job(_scene(tmp_project, motion="spin"), SHORT, tmp_project.cache / "x")
    assert job["safe"] == asdict(SHORT.safe)
    assert job["source"] == {"shape": "heart"} and job["motion"] == "spin"
    assert (job["width"], job["height"]) == (540, 960)  # depth rendered at half size
    assert job["fps"] == 30 and job["reveal"] is True
    assert job["out_dir"].startswith("/")


def test_render_returns_full_size_sequence(tmp_project, fake_blender):
    src = BlenderDepthSource(fake_blender.path, tmp_project.cache)
    seq = src.render(_scene(tmp_project), SHORT)
    assert len(seq) == 30 and (seq.width, seq.height) == (1080, 1920)
    f = seq.frame(0)
    assert f.shape == (1920, 1080) and 0.55 < f.max() < 0.65
    assert seq.reveal.shape == (1920, 1080, 3)


def test_cache_hit_skips_blender(tmp_project, fake_blender):
    src = BlenderDepthSource(fake_blender.path, tmp_project.cache)
    src.render(_scene(tmp_project), SHORT)
    src.render(_scene(tmp_project), SHORT)
    assert fake_blender.calls() == 1


def test_text_change_keeps_cache(tmp_project, fake_blender):
    src = BlenderDepthSource(fake_blender.path, tmp_project.cache)
    src.render(_scene(tmp_project), SHORT)
    src.render(_scene(tmp_project, text={"top": "Different"}), SHORT)
    assert fake_blender.calls() == 1


def test_motion_change_rerenders(tmp_project, fake_blender):
    src = BlenderDepthSource(fake_blender.path, tmp_project.cache)
    src.render(_scene(tmp_project, motion="spin"), SHORT)
    src.render(_scene(tmp_project, motion="bob"), SHORT)
    assert fake_blender.calls() == 2


def test_unknown_clip_becomes_blender_error(tmp_project, fake_blender, monkeypatch):
    monkeypatch.setenv("FAKE_BLENDER_MODE", "unknown_clip")
    src = BlenderDepthSource(fake_blender.path, tmp_project.cache)
    with pytest.raises(BlenderError) as e:
        src.render(_scene(tmp_project), SHORT)
    assert e.value.reason == "Animation clip 'Swimm' not found. Available: Idle, Swim"


def test_crash_has_log_tail(tmp_project, fake_blender, monkeypatch):
    monkeypatch.setenv("FAKE_BLENDER_MODE", "crash")
    src = BlenderDepthSource(fake_blender.path, tmp_project.cache)
    with pytest.raises(BlenderError) as e:
        src.render(_scene(tmp_project), SHORT)
    assert "some blender log line" in e.value.log_tail
    assert not any(p.name.endswith(".tmp") for p in tmp_project.cache.rglob("*"))


def test_partial_cache_is_redone(tmp_project, fake_blender):
    src = BlenderDepthSource(fake_blender.path, tmp_project.cache)
    seq = src.render(_scene(tmp_project), SHORT)
    (seq.directory / "meta.json").unlink()  # simulate an interrupted render
    src.render(_scene(tmp_project), SHORT)
    assert fake_blender.calls() == 2


def test_progress_is_logged(tmp_project, fake_blender):
    lines = []
    src = BlenderDepthSource(fake_blender.path, tmp_project.cache, log=lines.append)
    src.render(_scene(tmp_project), SHORT)
    assert any("30/30" in line for line in lines)
