"""Long YouTube videos: intro, many scenes with reveals, outro, music and chapters."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
from imageio_ffmpeg import read_frames
from PIL import Image

from engine.compose import LONG_FADE_OUT, LONG_REVEAL, SegmentComposer, card
from engine.encode import VideoWriter, concat_music, copy_silent, list_music, mux_music
from engine.formats import LONG
from engine.project import Project
from engine.scene import Compilation

FPS = 30
INTRO_SECONDS = 5.0
OUTRO_SECONDS = 4.0
VIEWING_HINT = "Relax your eyes and look through the screen"
OUTRO_LINES = ["Thanks for watching"]


class MissingSegments(Exception):
    def __init__(self, scene_ids: list[str]):
        super().__init__("Render these scenes first: " + ", ".join(scene_ids))
        self.scene_ids = scene_ids


def chapter_lines(starts: list[tuple[float, str]]) -> list[str]:
    long_video = any(t >= 3600 for t, _ in starts)
    lines = []
    for t, name in starts:
        s = int(t)
        h, m, sec = s // 3600, (s % 3600) // 60, s % 60
        stamp = f"{h}:{m:02d}:{sec:02d}" if long_video else f"{m:02d}:{sec:02d}"
        lines.append(f"{stamp} {name}")
    return lines


def _segment_frames(path: Path, limit: int):
    gen = read_frames(str(path))
    meta = next(gen)
    w, h = meta["size"]
    for i, raw in enumerate(gen):
        if i >= limit:
            break
        yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    gen.close()


def _title(scene_dir: Path, scene_id: str) -> str:
    try:
        return json.loads((scene_dir / "render.json").read_text()).get("title") or scene_id
    except (OSError, json.JSONDecodeError):
        return scene_id


def compile_long_video(comp: Compilation, project: Project, log: Callable[[str], None] = print) -> Path:
    missing = [sid for sid in comp.scenes if not (project.output / sid / "long_segment.mp4").exists()]
    if missing:
        raise MissingSegments(missing)
    out_dir = project.output / "compilations" / comp.id
    out_dir.mkdir(parents=True, exist_ok=True)
    video_only = out_dir / "video.video.mp4"
    final = out_dir / "video.mp4"
    font = project.font_path
    chapters: list[tuple[float, str]] = [(0.0, "Intro")]
    frames_written = 0
    total = len(comp.scenes)
    try:
        with VideoWriter(video_only, LONG.width, LONG.height, fps=FPS) as writer:
            intro = card([comp.title, VIEWING_HINT], LONG, font)
            for _ in range(round(INTRO_SECONDS * FPS)):
                writer.write(intro)
            frames_written += round(INTRO_SECONDS * FPS)
            for index, sid in enumerate(comp.scenes, start=1):
                scene_dir = project.output / sid
                chapters.append((frames_written / FPS, f"Stereogram {index}"))
                reveal_path = scene_dir / "reveal_long.png"
                reveal = (np.asarray(Image.open(reveal_path).convert("RGB")) if reveal_path.exists()
                          else np.full((LONG.height, LONG.width, 3), 140, np.uint8))
                wanted = round(comp.segment_seconds * FPS)
                stereo = list(_segment_frames(scene_dir / "long_segment.mp4", wanted))
                if len(stereo) < wanted:
                    log(f"[{comp.id}] {sid}: segment is shorter ({len(stereo) / FPS:.1f} s) than "
                        f"segment_seconds ({comp.segment_seconds:g} s); using all of it")
                composer = SegmentComposer(LONG, font, index=index, total=total, title=_title(scene_dir, sid),
                                           stereo_seconds=len(stereo) / FPS, reveal_img=reveal)
                for i, img in enumerate(stereo):
                    writer.write(composer.frame(img, i / FPS))
                tail = round((LONG_REVEAL + LONG_FADE_OUT) * FPS)
                for k in range(tail):
                    writer.write(composer.frame(stereo[-1], (len(stereo) + k) / FPS))
                frames_written += len(stereo) + tail
                log(f"[{comp.id}] added {index}/{total}: {sid}")
            chapters.append((frames_written / FPS, "Outro"))
            outro = card(OUTRO_LINES, LONG, font)
            for _ in range(round(OUTRO_SECONDS * FPS)):
                writer.write(outro)
            frames_written += round(OUTRO_SECONDS * FPS)
        duration = frames_written / FPS
        tracks = list_music(project.music)
        if tracks:
            random.Random(comp.id).shuffle(tracks)
            mix = out_dir / "music.wav"
            try:
                concat_music(tracks, mix, duration)
                mux_music(video_only, mix, final, duration)
            finally:
                mix.unlink(missing_ok=True)
        else:
            log(f"[{comp.id}] no music found in {project.music} - the video has no music")
            copy_silent(video_only, final)
    finally:
        video_only.unlink(missing_ok=True)
    (out_dir / "chapters.txt").write_text("\n".join(chapter_lines(chapters)) + "\n", encoding="utf-8")
    log(f"[{comp.id}] finished -> {final}")
    return final
