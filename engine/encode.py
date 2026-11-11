"""Video encoding and music, using the ffmpeg bundled with imageio-ffmpeg."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg"}
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"


def ffmpeg_path() -> str:
    env = os.environ.get("STEREO_FFMPEG")
    if env:
        return env
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()


def _part(path: Path) -> Path:
    return path.with_name(path.stem + ".part" + path.suffix)


def _run(args: list[str]) -> None:
    proc = subprocess.run([ffmpeg_path(), "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-1500:]}")


def _run_to(out: Path, args: list[str]) -> None:
    """Run ffmpeg writing to a temporary name, then move it into place."""
    out = Path(out)
    part = _part(out)
    try:
        _run([*args, str(part)])
        os.replace(part, out)
    finally:
        part.unlink(missing_ok=True)


class VideoWriter:
    """Pipe RGB frames into an H.264 MP4. The file appears only when complete."""

    def __init__(self, path: Path, width: int, height: int, fps: int = 30, crf: int = 16, preset: str = "slow"):
        self.path = Path(path)
        self.width, self.height = width, height
        self.part = _part(self.path)
        self._log = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            [ffmpeg_path(), "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-crf", str(crf),
             "-preset", preset, "-movflags", "+faststart", str(self.part)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame is {frame.shape[1]}x{frame.shape[0]}, expected {self.width}x{self.height}")
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._proc.stdin is not None
        if exc_type is not None:
            self._proc.kill()
            self._proc.wait()
            self.part.unlink(missing_ok=True)
            self._log.close()
            return
        self._proc.stdin.close()
        code = self._proc.wait()
        self._log.seek(0)
        err = self._log.read().decode(errors="replace")
        self._log.close()
        if code != 0:
            self.part.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed while encoding {self.path.name}: {err.strip()[-1500:]}")
        os.replace(self.part, self.path)


def list_music(music_dir: Path) -> list[Path]:
    music_dir = Path(music_dir)
    if not music_dir.is_dir():
        return []
    tracks = [p for p in music_dir.iterdir()
              if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in AUDIO_EXTS]
    return sorted(tracks, key=lambda p: p.name.lower())


def pick_music(music_dir: Path, key: str) -> Path | None:
    tracks = list_music(music_dir)
    if not tracks:
        return None
    return tracks[int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % len(tracks)]


def mux_music(video: Path, music: Path, out: Path, duration: float) -> None:
    """Add music (looped if short), trimmed to `duration`, faded and loudness-normalised."""
    fade_out = max(0.0, duration - 1.0)
    audio = (f"[1:a]atrim=0:{duration:.3f},asetpts=N/SR/TB,afade=t=in:st=0:d=1,"
             f"afade=t=out:st={fade_out:.3f}:d=1,{LOUDNORM}[a]")
    _run_to(out, ["-i", str(video), "-stream_loop", "-1", "-i", str(music),
                  "-filter_complex", audio, "-map", "0:v", "-map", "[a]",
                  "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                  "-t", f"{duration:.3f}", "-movflags", "+faststart"])


def concat_music(tracks: list[Path], out: Path, duration: float) -> None:
    """Join tracks back to back (repeating the list if needed) into one WAV of `duration`."""
    if not tracks:
        raise ValueError("no music tracks to join")
    args: list[str] = []
    for t in tracks:
        args += ["-i", str(t)]
    parts = "".join(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}];" for i in range(len(tracks)))
    joined = "".join(f"[a{i}]" for i in range(len(tracks)))
    graph = (f"{parts}{joined}concat=n={len(tracks)}:v=0:a=1[c];"
             f"[c]aloop=loop=-1:size=2000000000,atrim=0:{duration:.3f},asetpts=N/SR/TB[a]")
    _run_to(out, [*args, "-filter_complex", graph, "-map", "[a]", "-c:a", "pcm_s16le"])


def copy_silent(video: Path, out: Path) -> None:
    _run_to(out, ["-i", str(video), "-map", "0:v", "-c:v", "copy", "-an", "-movflags", "+faststart"])


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    width: int
    height: int
    has_audio: bool


def probe(path: Path) -> MediaInfo:
    proc = subprocess.run([ffmpeg_path(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    text = proc.stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not m:
        raise RuntimeError(f"Could not read media file {path}: {text.strip()[-500:]}")
    duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    v = re.search(r"Video: .*?(\d{2,5})x(\d{2,5})", text)
    width, height = (int(v.group(1)), int(v.group(2))) if v else (0, 0)
    return MediaInfo(duration, width, height, "Audio:" in text)
