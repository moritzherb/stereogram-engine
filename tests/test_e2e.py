"""Whole flow through the CLI: render the example scenes, then compile them."""
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from engine.cli import app
from engine.depth.blender_source import find_blender
from engine.encode import probe
from tests.conftest import make_sine

REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _copy_examples(project, duration):
    for f in sorted((REPO / "scenes").glob("example-*.yaml")):
        data = yaml.safe_load(f.read_text())
        data["duration"] = duration
        (project.scenes / f.name).write_text(yaml.safe_dump(data))
    comp = yaml.safe_load((REPO / "compilations" / "example.yaml").read_text())
    comp["segment_seconds"] = duration
    (project.compilations / "example.yaml").write_text(yaml.safe_dump(comp))


def _run_flow(project, duration):
    make_sine(project.music / "tune.wav", 4)
    root = ["--project", str(project.root)]
    res = runner.invoke(app, [*root, "render", str(project.scenes)])
    assert res.exit_code == 0, res.output
    for sid in ("example-heart", "example-text"):
        out = project.output / sid
        for name in ("short.mp4", "short_silent.mp4", "long_segment.mp4", "thumbnail.png", "credits.txt"):
            assert (out / name).exists(), f"{sid}/{name} missing"
        assert probe(out / "short.mp4").duration == pytest.approx(duration, abs=0.1)
        assert probe(out / "short.mp4").has_audio
        meta = json.loads((out / "render.json").read_text())
        assert "SELF_CHECK_FAILED" not in meta["warnings"], meta["warnings"]
    res = runner.invoke(app, [*root, "compile", str(project.compilations / "example.yaml")])
    assert res.exit_code == 0, res.output
    video = project.output / "compilations" / "example" / "video.mp4"
    info = probe(video)
    assert info.has_audio and info.duration == pytest.approx(5 + 2 * (duration + 4.5) + 4, abs=0.2)


def test_e2e_with_fake_depth(tmp_project, monkeypatch):
    monkeypatch.setenv("STEREO_DEPTH_SOURCE", "fake")
    _copy_examples(tmp_project, duration=2)
    _run_flow(tmp_project, duration=2)


@pytest.mark.skipif(find_blender() is None, reason="Blender not installed")
def test_e2e_blender(tmp_project, monkeypatch):
    monkeypatch.delenv("STEREO_DEPTH_SOURCE", raising=False)
    _copy_examples(tmp_project, duration=1)
    _run_flow(tmp_project, duration=1)
    assert (tmp_project.output / "example-heart" / "reveal.png").exists()
