"""`stereo check`: verifies everything the engine needs on this computer."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from engine.depth.blender_source import SCRIPT, SUPPORTED_VERSIONS, blender_version, find_blender

FONT = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Lato-Regular.ttf"
SHORT_FRAMES = 900  # 30 s at 30 fps


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def _check_python() -> CheckItem:
    v = sys.version_info
    ok = v >= (3, 11)
    return CheckItem("python", ok, f"Python {v.major}.{v.minor}.{v.micro}",
                     "" if ok else "Install Python 3.11 from python.org")


def _check_numba() -> CheckItem:
    try:
        import numba
        import numpy as np

        @numba.njit(cache=False)
        def _add(a):
            return a + 1

        _add(np.zeros(2))
        return CheckItem("numba", True, f"numba {numba.__version__}")
    except Exception as e:  # noqa: BLE001 - report anything
        return CheckItem("numba", False, str(e), "Run: pip install -e .")


def _check_ffmpeg() -> CheckItem:
    try:
        from engine.encode import ffmpeg_path
    except ImportError:
        from imageio_ffmpeg import get_ffmpeg_exe as ffmpeg_path
    try:
        exe = ffmpeg_path()
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=60).stdout
        return CheckItem("ffmpeg", True, out.splitlines()[0])
    except Exception as e:  # noqa: BLE001
        return CheckItem("ffmpeg", False, str(e), "Run: pip install -e .  (installs a bundled ffmpeg)")


def _check_font() -> CheckItem:
    try:
        from PIL import ImageFont

        ImageFont.truetype(str(FONT), 20)
        return CheckItem("font", True, FONT.name)
    except Exception as e:  # noqa: BLE001
        return CheckItem("font", False, str(e), f"Restore {FONT} from the repository")


def _check_blender() -> CheckItem:
    install_fix = ("Install Blender 4.5 LTS from blender.org into Applications "
                   "(if it won't open on your macOS, use Blender 4.2 LTS), "
                   "or set STEREO_BLENDER to its path")
    path = find_blender()
    if path is None:
        return CheckItem("blender", False, "Blender not found", install_fix)
    try:
        version = blender_version(path)
    except Exception as e:  # noqa: BLE001
        return CheckItem("blender", False, f"{path}: {e}", install_fix)
    vtext = f"Blender {version[0]}.{version[1]}"
    if version not in SUPPORTED_VERSIONS:
        return CheckItem("blender", False, f"{vtext} is not supported",
                         "Use Blender 4.5 LTS (or 4.2 LTS). Blender 5 does not run on Intel Macs.")
    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / "job.json"
        out = Path(tmp) / "out"
        job.write_text(json.dumps({"probe": True, "out_dir": str(out)}))
        proc = subprocess.run([str(path), "-b", "--factory-startup", "--python", str(SCRIPT), "--", str(job)],
                              capture_output=True, text=True, timeout=600)
        meta_file = out / "meta.json"
        if proc.returncode != 0 or not meta_file.exists():
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
            return CheckItem("blender", False, f"{vtext} failed the test render:\n{tail}",
                             "Try Blender 4.2 LTS, or send this output to your developer")
        spf = json.loads(meta_file.read_text())["seconds_per_frame"]
    minutes = spf * SHORT_FRAMES / 60
    return CheckItem("blender", True,
                     f"{vtext}, {spf:.1f} s/frame (test size) -> about {minutes:.0f} min "
                     f"of depth rendering per 30 s Short (full size is slower)")


def run_checks(include_blender: bool = True) -> list[CheckItem]:
    items = [_check_python(), _check_numba(), _check_ffmpeg(), _check_font()]
    if include_blender:
        items.append(_check_blender())
    return items
