"""Regression tests for the final whole-branch review findings."""
import json
import sys
import tomllib
import tracemalloc
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from engine.cli import app
from engine.compile import compile_long_video
from engine.depth.fake import FakeDepthSource
from engine.encode import probe
from engine.pipeline import render_batch, render_scene
from engine.scene import load_compilation, load_scene
from tests.conftest import make_sine, write_scene

REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()


# Critical 1: installable on Intel Macs (numba >= 0.63 ships no x86_64 macOS wheels)
def test_intel_mac_numba_pin():
    deps = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]
    mac = 'sys_platform == "darwin" and platform_machine == "x86_64"'
    assert any(d.startswith("numba") and "<0.63" in d and mac in d for d in deps)
    assert any(d.startswith("llvmlite") and "<0.46" in d and mac in d for d in deps)
    assert "--prefer-binary" in (REPO / "README.md").read_text()


# Critical 2: compile streams frames instead of holding whole segments in memory
def test_compile_memory_is_bounded(tmp_project):
    for sid in ("a", "b"):
        scene = load_scene(write_scene(tmp_project, f"{sid}.yaml", shape="heart", duration=3, formats=["long"]),
                           tmp_project)
        render_scene(scene, tmp_project, FakeDepthSource(), log=lambda m: None)
    p = tmp_project.compilations / "c.yaml"
    p.write_text(yaml.safe_dump({"id": "c", "title": "T", "scenes": ["a", "b"], "segment_seconds": 3}))
    tracemalloc.start()
    compile_long_video(load_compilation(p, tmp_project), tmp_project, log=lambda m: None)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 200 * 1024 * 1024, f"peak {peak / 1e6:.0f} MB"


# Important 3: unquoted YAML on/off
def test_yaml_on_off_values(tmp_project):
    p = tmp_project.scenes / "s.yaml"
    p.write_text("id: s\nshape: heart\nreveal: on\nmusic: off\ntext:\n  end: off\n")
    s = load_scene(p, tmp_project)
    assert s.reveal == "on" and s.music == "none" and s.text.end is None
    p.write_text("id: s\nshape: heart\nreveal: off\n")
    assert load_scene(p, tmp_project).reveal == "off"


# Important 4: music added after a render is picked up
def test_new_music_triggers_rerender(tmp_project):
    scene = load_scene(write_scene(tmp_project, "s.yaml", shape="heart", duration=1, formats=["short"]),
                       tmp_project)
    render_scene(scene, tmp_project, FakeDepthSource(), log=lambda m: None)
    make_sine(tmp_project.music / "tune.wav", 2)
    r = render_scene(scene, tmp_project, FakeDepthSource(), log=lambda m: None)
    assert not r.skipped and probe(tmp_project.output / "s" / "short.mp4").has_audio


# Important 5: duplicate scene ids in one batch
def test_duplicate_ids_fail_clearly(tmp_project):
    write_scene(tmp_project, "a.yaml", id="dolphin", shape="heart", duration=1, formats=["short"])
    write_scene(tmp_project, "b.yaml", id="dolphin", shape="star", duration=1, formats=["short"])
    summary = render_batch([tmp_project.scenes], tmp_project, FakeDepthSource(), log=lambda m: None)
    assert summary.done == ["dolphin"]
    assert summary.failed[0][0] == "b.yaml" and "a.yaml" in summary.failed[0][1]


# Important 6: Blender's log reaches the user; preview/compile fail without tracebacks
def test_batch_shows_blender_log(tmp_project, fake_blender, monkeypatch):
    from engine.depth.blender_source import BlenderDepthSource

    monkeypatch.setenv("FAKE_BLENDER_MODE", "crash")
    write_scene(tmp_project, "a.yaml", shape="heart", duration=1)
    lines = []
    summary = render_batch([tmp_project.scenes], tmp_project,
                           BlenderDepthSource(fake_blender.path, tmp_project.cache), log=lines.append)
    assert summary.failed and "exit code 1" in summary.failed[0][1]
    assert any("some blender log line" in line for line in lines)


def test_cli_preview_blender_error_is_friendly(tmp_project, fake_blender, monkeypatch):
    monkeypatch.delenv("STEREO_DEPTH_SOURCE", raising=False)
    monkeypatch.setenv("STEREO_BLENDER", str(fake_blender.path))
    monkeypatch.setenv("FAKE_BLENDER_MODE", "unknown_clip")
    p = write_scene(tmp_project, "a.yaml", shape="heart", duration=1)
    res = runner.invoke(app, ["--project", str(tmp_project.root), "preview", str(p)])
    assert res.exit_code == 1 and res.exception is None or isinstance(res.exception, SystemExit)
    assert "Animation clip 'Swimm' not found" in res.output


def test_cli_compile_runtime_error_is_friendly(tmp_project, monkeypatch):
    monkeypatch.setenv("STEREO_DEPTH_SOURCE", "fake")
    write_scene(tmp_project, "a.yaml", shape="heart", duration=1, formats=["long"])
    root = ["--project", str(tmp_project.root)]
    assert runner.invoke(app, [*root, "render", str(tmp_project.scenes)]).exit_code == 0
    (tmp_project.output / "a" / "long_segment.mp4").write_bytes(b"corrupt")
    comp = tmp_project.compilations / "c.yaml"
    comp.write_text(yaml.safe_dump({"id": "c", "title": "T", "scenes": ["a"], "segment_seconds": 1}))
    res = runner.invoke(app, [*root, "compile", str(comp)])
    assert res.exit_code == 1 and isinstance(res.exception, SystemExit)
    assert "Compile failed" in res.output


# Important 7: YouTube chapter rules (first at 00:00, each >= 10 s)
def test_chapters_follow_youtube_rules(tmp_project):
    for sid in ("a", "b", "c"):
        scene = load_scene(write_scene(tmp_project, f"{sid}.yaml", shape="heart", duration=6, formats=["long"]),
                           tmp_project)
        render_scene(scene, tmp_project, FakeDepthSource(), log=lambda m: None)
    p = tmp_project.compilations / "c.yaml"
    p.write_text(yaml.safe_dump({"id": "c", "title": "T", "scenes": ["a", "b", "c"], "segment_seconds": 6}))
    video = compile_long_video(load_compilation(p, tmp_project), tmp_project, log=lambda m: None)
    lines = (video.parent / "chapters.txt").read_text().splitlines()
    assert lines[0].startswith("00:00 ")
    starts = [int(line[:2]) * 60 + int(line[3:5]) for line in lines] + [int(probe(video).duration)]
    assert all(b - a >= 10 for a, b in zip(starts, starts[1:])), lines


def test_short_chapters_warn(tmp_project):
    scene = load_scene(write_scene(tmp_project, "a.yaml", shape="heart", duration=1, formats=["long"]), tmp_project)
    render_scene(scene, tmp_project, FakeDepthSource(), log=lambda m: None)
    p = tmp_project.compilations / "c.yaml"
    p.write_text(yaml.safe_dump({"id": "c", "title": "T", "scenes": ["a"], "segment_seconds": 1}))
    lines = []
    compile_long_video(load_compilation(p, tmp_project), tmp_project, log=lines.append)
    assert any("YouTube" in line for line in lines)


# Review #10 (re-graded): an unreadable pattern fails before Blender runs
def test_bad_pattern_fails_before_depth(tmp_project):
    (tmp_project.patterns / "photo.heic").write_bytes(b"not an image we can read")
    scene = load_scene(write_scene(tmp_project, "a.yaml", shape="heart", duration=1, pattern="patterns/photo.heic"),
                       tmp_project)
    src = FakeDepthSource()
    with pytest.raises(ValueError, match="photo.heic"):
        render_scene(scene, tmp_project, src, log=lambda m: None)
    assert src.calls == 0
