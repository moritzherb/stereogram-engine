import json

import pytest

from engine.depth.fake import FakeDepthSource
from engine.encode import probe
from engine.pipeline import render_batch, render_scene
from engine.project import Project
from engine.scene import load_scene
from tests.conftest import make_sine, write_scene


def _scene(project, name="heart.yaml", **data):
    data.setdefault("shape", "heart")
    data.setdefault("duration", 2)
    return load_scene(write_scene(project, name, **data), project)


def test_render_short_and_long(tmp_project):
    r = render_scene(_scene(tmp_project), tmp_project, FakeDepthSource(), log=lambda m: None)
    out = tmp_project.output / "heart"
    names = {p.name for p in r.outputs}
    assert {"short.mp4", "short_silent.mp4", "long_segment.mp4", "reveal.png", "thumbnail.png",
            "credits.txt"} <= names
    short = probe(out / "short.mp4")
    assert (short.width, short.height) == (1080, 1920) and short.duration == pytest.approx(2.0, abs=0.1)
    assert probe(out / "short_silent.mp4").has_audio is False
    long = probe(out / "long_segment.mp4")
    assert (long.width, long.height) == (1920, 1080)
    assert (out / "render.json").exists() and (out / "reveal_long.png").exists()
    assert not list(out.glob("*.part*")) and not list(out.glob("*.video.mp4"))


def test_only_requested_formats(tmp_project):
    r = render_scene(_scene(tmp_project, formats=["short"]), tmp_project, FakeDepthSource(), log=lambda m: None)
    names = {p.name for p in r.outputs}
    assert "short.mp4" in names and "long_segment.mp4" not in names


def test_music_used_when_present(tmp_project):
    make_sine(tmp_project.music / "calm tune.wav", 5)
    render_scene(_scene(tmp_project, formats=["short"]), tmp_project, FakeDepthSource(), log=lambda m: None)
    out = tmp_project.output / "heart"
    assert probe(out / "short.mp4").has_audio
    assert "calm tune.wav" in (out / "credits.txt").read_text()


def test_music_none_setting(tmp_project):
    make_sine(tmp_project.music / "tune.wav", 5)
    render_scene(_scene(tmp_project, formats=["short"], music="none"), tmp_project, FakeDepthSource(),
                 log=lambda m: None)
    assert not probe(tmp_project.output / "heart" / "short.mp4").has_audio


def test_no_music_warns_but_succeeds(tmp_project):
    lines = []
    render_scene(_scene(tmp_project, formats=["short"]), tmp_project, FakeDepthSource(), log=lines.append)
    assert (tmp_project.output / "heart" / "short.mp4").exists()
    assert any("no music" in line.lower() for line in lines)


def test_skip_when_unchanged(tmp_project):
    src = FakeDepthSource()
    render_scene(_scene(tmp_project), tmp_project, src, log=lambda m: None)
    calls = src.calls
    r = render_scene(_scene(tmp_project), tmp_project, src, log=lambda m: None)
    assert r.skipped and src.calls == calls


def test_force_rerenders(tmp_project):
    src = FakeDepthSource()
    render_scene(_scene(tmp_project), tmp_project, src, log=lambda m: None)
    r = render_scene(_scene(tmp_project), tmp_project, src, force=True, log=lambda m: None)
    assert not r.skipped


def test_rerender_when_text_changes(tmp_project):
    src = FakeDepthSource()
    render_scene(_scene(tmp_project), tmp_project, src, log=lambda m: None)
    r = render_scene(_scene(tmp_project, text={"top": "Look closer"}), tmp_project, src, log=lambda m: None)
    assert not r.skipped


def test_interrupted_render_is_redone(tmp_project):
    src = FakeDepthSource()
    render_scene(_scene(tmp_project), tmp_project, src, log=lambda m: None)
    (tmp_project.output / "heart" / "render.json").unlink()  # crash before the end
    r = render_scene(_scene(tmp_project), tmp_project, src, log=lambda m: None)
    assert not r.skipped


def test_reveal_on_short_ends_on_reveal(tmp_project):
    r = render_scene(_scene(tmp_project, formats=["short"], reveal="on"), tmp_project, FakeDepthSource(),
                     log=lambda m: None)
    assert not r.skipped and (tmp_project.output / "heart" / "short.mp4").exists()


def test_batch_continues_after_failure(tmp_project):
    write_scene(tmp_project, "a.yaml", shape="heart", duration=1, formats=["short"])
    write_scene(tmp_project, "bad.yaml", model="models/missing.glb")
    write_scene(tmp_project, "c.yaml", shape="star", duration=1, formats=["short"])
    summary = render_batch([tmp_project.scenes], tmp_project, FakeDepthSource(), log=lambda m: None)
    assert summary.done == ["a", "c"]
    assert len(summary.failed) == 1 and summary.failed[0][0] == "bad.yaml"
    assert "missing.glb" in summary.failed[0][1]


def test_batch_reports_skipped(tmp_project):
    write_scene(tmp_project, "a.yaml", shape="heart", duration=1, formats=["short"])
    render_batch([tmp_project.scenes], tmp_project, FakeDepthSource(), log=lambda m: None)
    summary = render_batch([tmp_project.scenes], tmp_project, FakeDepthSource(), log=lambda m: None)
    assert summary.skipped == ["a"] and summary.done == []


def test_preview_small_and_short(tmp_project):
    r = render_scene(_scene(tmp_project, duration=8), tmp_project, FakeDepthSource(), preview=True,
                     log=lambda m: None)
    info = probe(tmp_project.output / "heart" / "preview.mp4")
    assert (info.width, info.height) == (540, 960) and info.duration == pytest.approx(5.0, abs=0.1)
    assert [p.name for p in r.outputs] == ["preview.mp4"]


def test_warnings_are_reported(tmp_project):
    lines = []
    r = render_scene(_scene(tmp_project, formats=["short"]), tmp_project, FakeDepthSource(radius=0.45),
                     log=lines.append)
    assert "OUT_OF_SAFE_AREA" in [w.code for w in r.warnings]
    assert any("safe area" in line for line in lines)


def test_project_path_with_spaces(tmp_path):
    root = tmp_path / "My Videos" / "stereo project"
    (root / "scenes").mkdir(parents=True)
    project = Project(root)
    scene = load_scene(write_scene(project, "x.yaml", shape="cube", duration=1, formats=["short"]), project)
    render_scene(scene, project, FakeDepthSource(), log=lambda m: None)
    assert (root / "output" / "x" / "short.mp4").exists()
    meta = json.loads((root / "output" / "x" / "render.json").read_text())
    assert "short.mp4" in meta["outputs"]
