import os
import stat
import sys

from engine.check_env import run_checks
from engine.depth.blender_source import blender_version, find_blender


def _fake_blender(tmp_path, version_line):
    p = tmp_path / "fake blender"
    p.write_text(f"#!{sys.executable}\nprint({version_line!r})\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_checks_report_python_ffmpeg_numba_font():
    items = {i.name: i for i in run_checks(include_blender=False)}
    assert items["python"].ok
    assert items["ffmpeg"].ok
    assert items["numba"].ok
    assert items["font"].ok


def test_find_blender_env_override(tmp_path, monkeypatch):
    fake = _fake_blender(tmp_path, "Blender 4.5.3")
    monkeypatch.setenv("STEREO_BLENDER", str(fake))
    assert find_blender() == fake


def test_blender_version_parses(tmp_path):
    fake = _fake_blender(tmp_path, "Blender 4.2.11 LTS")
    assert blender_version(fake) == (4, 2)


def test_blender_5_rejected(tmp_path, monkeypatch):
    fake = _fake_blender(tmp_path, "Blender 5.0.1")
    monkeypatch.setenv("STEREO_BLENDER", str(fake))
    items = {i.name: i for i in run_checks(include_blender=True)}
    assert not items["blender"].ok
    assert "4.5" in items["blender"].fix


def test_blender_missing_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("STEREO_BLENDER", str(tmp_path / "nope"))
    monkeypatch.setenv("PATH", str(tmp_path))
    items = {i.name: i for i in run_checks(include_blender=True)}
    assert not items["blender"].ok
    assert "blender.org" in items["blender"].fix
