from typer.testing import CliRunner

from engine.cli import app
from tests.conftest import write_scene

runner = CliRunner()


def test_cli_render_ok(tmp_project, monkeypatch):
    monkeypatch.setenv("STEREO_DEPTH_SOURCE", "fake")
    write_scene(tmp_project, "a.yaml", shape="heart", duration=1, formats=["short"])
    res = runner.invoke(app, ["--project", str(tmp_project.root), "render", str(tmp_project.scenes)])
    assert res.exit_code == 0, res.output
    assert "Done: a" in res.output


def test_cli_render_failure_exit_code(tmp_project, monkeypatch):
    monkeypatch.setenv("STEREO_DEPTH_SOURCE", "fake")
    write_scene(tmp_project, "bad.yaml", model="models/missing.glb")
    res = runner.invoke(app, ["--project", str(tmp_project.root), "render", str(tmp_project.scenes)])
    assert res.exit_code == 1
    assert "bad.yaml" in res.output and "missing.glb" in res.output


def test_cli_preview(tmp_project, monkeypatch):
    monkeypatch.setenv("STEREO_DEPTH_SOURCE", "fake")
    p = write_scene(tmp_project, "a.yaml", shape="heart", duration=2)
    res = runner.invoke(app, ["--project", str(tmp_project.root), "preview", str(p)])
    assert res.exit_code == 0, res.output
    assert (tmp_project.output / "a" / "preview.mp4").exists()


def test_cli_without_blender(tmp_project, monkeypatch, tmp_path):
    monkeypatch.delenv("STEREO_DEPTH_SOURCE", raising=False)
    monkeypatch.setenv("STEREO_BLENDER", str(tmp_path / "nope"))
    write_scene(tmp_project, "a.yaml", shape="heart", duration=1)
    res = runner.invoke(app, ["--project", str(tmp_project.root), "render", str(tmp_project.scenes)])
    assert res.exit_code == 1 and "stereo check" in res.output
