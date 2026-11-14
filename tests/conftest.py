import numpy as np
import pytest


def random_strip(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Per-pixel random colour strip: the hardest case for exact-equality tests."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


@pytest.fixture
def strip():
    return random_strip(64, 154, seed=1)


@pytest.fixture
def tmp_project(tmp_path):
    """An empty project folder (with a space in its path) plus the real font."""
    import shutil
    from pathlib import Path

    from engine.project import Project

    root = tmp_path / "My Videos" / "stereo"
    for sub in ("assets/models", "assets/patterns", "assets/music", "assets/fonts", "scenes", "compilations"):
        (root / sub).mkdir(parents=True)
    repo_fonts = Path(__file__).resolve().parents[1] / "assets" / "fonts"
    for f in repo_fonts.iterdir():
        shutil.copy(f, root / "assets" / "fonts" / f.name)
    return Project(root)


@pytest.fixture
def fake_blender(tmp_path, monkeypatch):
    """Path to an executable fake Blender plus its call-count file."""
    import stat
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent / "fake_blender.py"
    exe = tmp_path / "fake blender bin"
    exe.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{script}' \"$@\"\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    count = tmp_path / "calls.txt"
    monkeypatch.setenv("FAKE_BLENDER_COUNT", str(count))
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))

    class Fake:
        path = exe

        @staticmethod
        def calls() -> int:
            return int(count.read_text()) if count.exists() else 0

    return Fake


def make_sine(path, seconds: float, freq: int = 440):
    """Write a test tone (any container ffmpeg can infer from the suffix)."""
    import subprocess

    from engine.encode import ffmpeg_path

    subprocess.run([ffmpeg_path(), "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"sine=frequency={freq}:duration={seconds}", str(path)], check=True)
    return path


@pytest.fixture(autouse=True)
def fast_x264(monkeypatch):
    """Encode test videos quickly; real renders use the high-quality preset."""
    monkeypatch.setenv("STEREO_X264_PRESET", "ultrafast")


def write_scene(project, name: str, **data):
    import yaml

    data.setdefault("id", name.removesuffix(".yaml"))
    p = project.scenes / name
    p.write_text(yaml.safe_dump(data))
    return p
