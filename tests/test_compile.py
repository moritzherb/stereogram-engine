import pytest
import yaml

from engine.compile import MissingSegments, chapter_lines, compile_long_video
from engine.depth.fake import FakeDepthSource
from engine.encode import probe
from engine.pipeline import render_scene
from engine.scene import load_compilation, load_scene
from tests.conftest import make_sine, write_scene


def test_chapter_format():
    assert chapter_lines([(0, "Intro"), (5, "Stereogram 1"), (3725, "Stereogram 2")]) == [
        "0:00:00 Intro", "0:00:05 Stereogram 1", "1:02:05 Stereogram 2"]
    assert chapter_lines([(0, "Intro"), (65.4, "Stereogram 1")]) == ["00:00 Intro", "01:05 Stereogram 1"]


def _render(project, sid, title, duration=2):
    scene = load_scene(write_scene(project, f"{sid}.yaml", shape="heart", duration=duration, formats=["long"],
                                   title=title), project)
    render_scene(scene, project, FakeDepthSource(), log=lambda m: None)


def _comp(project, scenes, segment_seconds=1):
    p = project.compilations / "c.yaml"
    p.write_text(yaml.safe_dump({"id": "c", "title": "Hidden things", "scenes": scenes,
                                 "segment_seconds": segment_seconds}))
    return load_compilation(p, project)


def test_missing_segments_named(tmp_project):
    _render(tmp_project, "a", "Secret heart")
    with pytest.raises(MissingSegments) as e:
        compile_long_video(_comp(tmp_project, ["a", "ghost"]), tmp_project, log=lambda m: None)
    assert e.value.scene_ids == ["ghost"]


def test_compile_two_scenes(tmp_project):
    _render(tmp_project, "a", "Secret heart")
    _render(tmp_project, "b", "Secret star")
    video = compile_long_video(_comp(tmp_project, ["a", "b"]), tmp_project, log=lambda m: None)
    info = probe(video)
    assert (info.width, info.height) == (1920, 1080)
    assert info.duration == pytest.approx(5 + 2 * (1 + 4 + 0.5) + 4, abs=0.2)
    chapters = (video.parent / "chapters.txt").read_text().splitlines()
    assert chapters == ["00:00 Stereogram 1", "00:10 Stereogram 2"]
    assert "Secret" not in "\n".join(chapters)
    assert not list(video.parent.glob("*.part*")) and not list(video.parent.glob("*.video.mp4"))


def test_segment_shorter_than_segment_seconds_warns(tmp_project):
    _render(tmp_project, "a", "Secret heart", duration=1)
    lines = []
    video = compile_long_video(_comp(tmp_project, ["a"], segment_seconds=3), tmp_project, log=lines.append)
    assert any("shorter" in line for line in lines)
    assert probe(video).duration == pytest.approx(5 + 1 + 4.5 + 4, abs=0.2)


def test_music_spans_video(tmp_project):
    _render(tmp_project, "a", "Secret heart")
    make_sine(tmp_project.music / "one.wav", 3, 440)
    make_sine(tmp_project.music / "two.wav", 3, 660)
    video = compile_long_video(_comp(tmp_project, ["a"]), tmp_project, log=lambda m: None)
    info = probe(video)
    assert info.has_audio and info.duration == pytest.approx(5 + 5.5 + 4, abs=0.2)
