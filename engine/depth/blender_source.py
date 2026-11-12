"""Finding and driving Blender (which runs as a separate background process)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from engine.depth.base import PngDepthSequence
from engine.formats import Format
from engine.project import BUNDLED_FONT
from engine.scene import Scene, depth_hash

MAC_APP = Path("/Applications/Blender.app/Contents/MacOS/Blender")
SUPPORTED_VERSIONS = {(4, 5), (4, 2)}
SCRIPT = Path(__file__).resolve().parents[1] / "blender" / "render_depth.py"
FPS = 30
DEPTH_SCALE = 2  # depth is rendered at 1/2 of the output size, then scaled up
SAMPLES = 4


class BlenderError(Exception):
    def __init__(self, reason: str, log_tail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.log_tail = log_tail


def find_blender() -> Path | None:
    """STEREO_BLENDER env var, then `blender` on PATH, then the macOS app."""
    env = os.environ.get("STEREO_BLENDER")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)
    if MAC_APP.is_file():
        return MAC_APP
    return None


def blender_version(path: Path) -> tuple[int, int]:
    out = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=120).stdout
    m = re.search(r"Blender (\d+)\.(\d+)", out)
    if not m:
        raise RuntimeError(f"Could not read Blender version from: {out[:200]!r}")
    return int(m.group(1)), int(m.group(2))


def blender_command(blender: Path, job_file: Path) -> list[str]:
    return [str(blender), "-b", "--factory-startup", "--python-exit-code", "1",
            "--python", str(SCRIPT), "--", str(job_file)]


def build_job(scene: Scene, fmt: Format, out_dir: Path) -> dict:
    if scene.model is not None:
        source = {"model": str(Path(scene.model).resolve())}
    elif scene.text3d is not None:
        source = {"text3d": scene.text3d}
    else:
        source = {"shape": scene.shape}
    return {
        "source": source,
        "animation": scene.animation,
        "motion": scene.motion,
        "motion_amount": scene.motion_settings.amount,
        "motion_seconds": scene.motion_settings.seconds,
        "speed": scene.speed,
        "duration": scene.duration,
        "fps": FPS,
        "width": max(2, fmt.width // DEPTH_SCALE),
        "height": max(2, fmt.height // DEPTH_SCALE),
        "samples": SAMPLES,
        "safe": asdict(fmt.safe),
        "reveal": True,
        "font": str(BUNDLED_FONT),
        "out_dir": str(Path(out_dir).resolve()),
    }


def _script_hash() -> str:
    return hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


class BlenderDepthSource:
    """Renders depth frames with Blender and caches them by content hash."""

    def __init__(self, blender: Path, cache_dir: Path, timeout_s: int = 6 * 3600,
                 log: Callable[[str], None] | None = None):
        self.blender = Path(blender)
        self.cache_dir = Path(cache_dir)
        self.timeout_s = timeout_s
        self.log = log or (lambda _msg: None)

    def cache_path(self, scene: Scene, fmt: Format) -> Path:
        key = hashlib.sha256(
            f"{depth_hash(scene, fmt)}|{_script_hash()}|{DEPTH_SCALE}|{SAMPLES}".encode()
        ).hexdigest()[:16]
        return self.cache_dir / scene.id / fmt.name / key

    def render(self, scene: Scene, fmt: Format) -> PngDepthSequence:
        final = self.cache_path(scene, fmt)
        size = (fmt.width, fmt.height)
        if (final / "meta.json").exists():
            self.log(f"[{scene.id}/{fmt.name}] depth frames from cache")
            return PngDepthSequence(final, fps=FPS, size=size)
        if final.exists():
            shutil.rmtree(final)  # interrupted earlier
        tmp = final.with_name(final.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            job = build_job(scene, fmt, tmp)
            job_file = tmp / "job.json"
            job_file.write_text(json.dumps(job))
            self._run(job_file, f"[{scene.id}/{fmt.name}]")
            (tmp / "job.json").unlink()
            tmp.rename(final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return PngDepthSequence(final, fps=FPS, size=size)

    def _run(self, job_file: Path, label: str) -> None:
        self.log(f"{label} rendering depth frames with Blender...")
        tail: deque[str] = deque(maxlen=40)
        proc = subprocess.Popen(blender_command(self.blender, job_file), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace")
        assert proc.stdout is not None
        try:
            last_logged = -1
            for line in proc.stdout:
                line = line.rstrip()
                tail.append(line)
                m = re.match(r"FRAME (\d+)/(\d+)", line)
                if m:
                    done, total = int(m.group(1)), int(m.group(2))
                    pct = done * 10 // total
                    if pct != last_logged or done == total:
                        last_logged = pct
                        self.log(f"{label} depth frame {done}/{total}")
            code = proc.wait(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise BlenderError(f"Blender took longer than {self.timeout_s // 60} minutes", "\n".join(tail))
        except BaseException:
            proc.kill()
            raise
        if code == 0:
            return
        text = "\n".join(tail)
        m = re.search(r"UNKNOWN_CLIP: (.+?); available: (.*)", text)
        if code == 3 and m:
            raise BlenderError(f"Animation clip '{m.group(1)}' not found. Available: {m.group(2)}", text)
        m = re.search(r"IMPORT_FAILED: (.*)", text)
        if code == 4 and m:
            raise BlenderError(f"Could not load the model: {m.group(1)}", text)
        raise BlenderError(f"Blender failed (exit code {code})", text)
