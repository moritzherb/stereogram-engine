"""Rendering scenes into finished videos (one scene, a batch, or a quick preview)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from engine.checks import QualityWarning, check_frames, check_motion, check_pattern, check_safe_area
from engine.compose import ShortComposer, thumbnail
from engine.depth.base import DepthSource
from engine.encode import VideoWriter, copy_silent, mux_music, pick_music
from engine.formats import LONG, SHORT, Format, preview_of
from engine.patterns import generate_pattern, load_image_pattern
from engine.project import Project
from engine.scene import GeneratedPattern, Scene, SceneError, load_scene, render_hash, reveal_enabled
from engine.stereo import render_stereogram

Log = Callable[[str], None]
PREVIEW_SECONDS = 5.0
SEGMENT_CRF = 12


@dataclass
class RenderResult:
    scene_id: str
    outputs: list[Path] = field(default_factory=list)
    warnings: list[QualityWarning] = field(default_factory=list)
    skipped: bool = False


@dataclass
class BatchSummary:
    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def _save_png(arr: np.ndarray, path: Path) -> Path:
    tmp = path.with_name(path.stem + ".part.png")
    Image.fromarray(arr).save(tmp)
    os.replace(tmp, path)
    return path


def _repeat_px(scene: Scene, fmt: Format) -> int:
    override = scene.advanced.pattern_repeat
    if override == "auto":
        return fmt.repeat_px
    base = SHORT if fmt.name.startswith("short") else LONG
    return max(8, round(int(override) * fmt.repeat_px / base.repeat_px))


def pattern_strip(scene: Scene, repeat_px: int) -> np.ndarray:
    p = scene.pattern
    if isinstance(p, GeneratedPattern):
        return generate_pattern(p.generate, repeat_px=repeat_px, palette=p.palette, seed=p.seed or 0)
    return load_image_pattern(Path(p), repeat_px=repeat_px)


def _music_for(scene: Scene, project: Project) -> Path | None:
    if scene.music == "none":
        return None
    if isinstance(scene.music, Path):
        return scene.music
    return pick_music(project.music, scene.id)


class _FrameRenderer:
    """Turns depth frames into stereogram frames, keeping a few for checks."""

    def __init__(self, scene: Scene, fmt: Format, seq, strip: np.ndarray):
        self.scene, self.fmt, self.seq, self.strip = scene, fmt, seq, strip
        self.repeat = _repeat_px(scene, fmt)
        n = len(seq)
        self.sample_ids = sorted({0, n // 2, n - 1})
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.middle: np.ndarray | None = None

    def frames(self, log: Log, label: str):
        n = len(self.seq)
        step = max(1, n // 10)
        for i in range(n):
            depth = self.seq.frame(i)
            img = render_stereogram(depth, self.strip, repeat_px=self.repeat,
                                    depth_strength=self.scene.advanced.depth_strength,
                                    crosseyed=self.scene.advanced.viewing == "crosseyed")
            if i in self.sample_ids:
                self.samples.append((depth, img))
            if i == n // 2:
                self.middle = img
            if (i + 1) % step == 0 or i + 1 == n:
                log(f"{label} frame {i + 1}/{n}")
            yield i, img

    def self_check(self) -> list[QualityWarning]:
        return check_frames([d for d, _ in self.samples], [s for _, s in self.samples], repeat_px=self.repeat,
                            depth_strength=self.scene.advanced.depth_strength,
                            crosseyed=self.scene.advanced.viewing == "crosseyed")


def _render_short(scene: Scene, project: Project, fmt: Format, seq, strip, out_dir: Path, *,
                  name: str, with_music: bool, log: Log) -> tuple[list[Path], _FrameRenderer, Path | None]:
    label = f"[{scene.id}/{fmt.name}]"
    reveal = seq.reveal if reveal_enabled(scene, "short") else None
    composer = ShortComposer(scene, fmt, project.font_path, reveal)
    fr = _FrameRenderer(scene, fmt, seq, strip)
    video_only = out_dir / f"{name}.video.mp4"
    try:
        with VideoWriter(video_only, fmt.width, fmt.height, fps=seq.fps) as writer:
            for i, img in fr.frames(log, label):
                writer.write(composer.frame(img, i / seq.fps))
        outputs = []
        track = _music_for(scene, project) if with_music else None
        if with_music and track is None and scene.music == "auto":
            log(f"{label} no music found in {project.music} - the Short has no music")
        main = out_dir / f"{name}.mp4"
        if track is not None:
            mux_music(video_only, track, main, duration=len(seq) / seq.fps)
        else:
            copy_silent(video_only, main)
        outputs.append(main)
        if with_music:
            silent = out_dir / f"{name}_silent.mp4"
            copy_silent(video_only, silent)
            outputs.append(silent)
    finally:
        video_only.unlink(missing_ok=True)
    return outputs, fr, track


def _render_long_segment(scene: Scene, fmt: Format, seq, strip, out_dir: Path, log: Log):
    fr = _FrameRenderer(scene, fmt, seq, strip)
    path = out_dir / "long_segment.mp4"
    with VideoWriter(path, fmt.width, fmt.height, fps=seq.fps, crf=SEGMENT_CRF) as writer:
        for _, img in fr.frames(log, f"[{scene.id}/{fmt.name}]"):
            writer.write(img)
    return path, fr


def _credits(scene: Scene, track: Path | None) -> str:
    lines = []
    if scene.model is not None:
        lines.append(f"Model: {Path(scene.model).name}")
        lic = Path(scene.model).parent / "license.txt"
        if lic.exists():
            lines.append(lic.read_text(encoding="utf-8", errors="replace").strip())
    elif scene.text3d is not None:
        lines.append(f"Hidden 3D text: {scene.text3d}")
    else:
        lines.append(f"Hidden shape: {scene.shape}")
    p = scene.pattern
    lines.append(f"Pattern: generated ({p.generate}, {p.palette})" if isinstance(p, GeneratedPattern)
                 else f"Pattern: {Path(p).name}")
    lines.append(f"Music: {track.name}" if track else "Music: none")
    lines.append("Font: Lato (SIL Open Font License)")
    return "\n".join(lines) + "\n"


def _is_up_to_date(scene: Scene, out_dir: Path) -> bool:
    meta_file = out_dir / "render.json"
    if not meta_file.exists():
        return False
    try:
        meta = json.loads(meta_file.read_text())
    except json.JSONDecodeError:
        return False
    return meta.get("render_hash") == render_hash(scene) and all((out_dir / o).exists() for o in meta["outputs"])


def render_scene(scene: Scene, project: Project, depth_source: DepthSource, *, preview: bool = False,
                 force: bool = False, log: Log = print) -> RenderResult:
    out_dir = project.output / scene.id
    out_dir.mkdir(parents=True, exist_ok=True)
    result = RenderResult(scene.id)

    def warn(ws: list[QualityWarning]) -> None:
        for w in ws:
            if w not in result.warnings:
                result.warnings.append(w)
                log(f"[{scene.id}] warning: {w.message}")

    if preview:
        short = scene.model_copy(update={"duration": min(PREVIEW_SECONDS, scene.duration)})
        fmt = preview_of(SHORT)
        seq = depth_source.render(short, fmt)
        warn(check_safe_area(seq, fmt) + check_motion(seq))
        strip = pattern_strip(short, _repeat_px(short, fmt))
        warn(check_pattern(strip))
        outputs, _, _ = _render_short(short, project, fmt, seq, strip, out_dir, name="preview",
                                      with_music=False, log=log)
        result.outputs = outputs
        return result

    if not force and _is_up_to_date(scene, out_dir):
        log(f"[{scene.id}] unchanged, skipping")
        result.skipped = True
        return result
    (out_dir / "render.json").unlink(missing_ok=True)

    track: Path | None = None
    middle_long: np.ndarray | None = None
    middle_short: np.ndarray | None = None
    for fmt_name in ("short", "long"):
        if fmt_name not in scene.formats:
            continue
        fmt = SHORT if fmt_name == "short" else LONG
        seq = depth_source.render(scene, fmt)
        warn(check_safe_area(seq, fmt) + check_motion(seq))
        strip = pattern_strip(scene, _repeat_px(scene, fmt))
        warn(check_pattern(strip))
        if fmt_name == "short":
            outputs, fr, track = _render_short(scene, project, fmt, seq, strip, out_dir, name="short",
                                               with_music=True, log=log)
            result.outputs += outputs
            middle_short = fr.middle
            if seq.reveal is not None:
                result.outputs.append(_save_png(seq.reveal, out_dir / "reveal.png"))
        else:
            path, fr = _render_long_segment(scene, fmt, seq, strip, out_dir, log)
            result.outputs.append(path)
            middle_long = fr.middle
            if seq.reveal is not None:
                result.outputs.append(_save_png(seq.reveal, out_dir / "reveal_long.png"))
        warn(fr.self_check())

    still = middle_long if middle_long is not None else middle_short
    if still is not None:
        result.outputs.append(_save_png(thumbnail(still, scene.title or scene.id, project.font_path),
                                        out_dir / "thumbnail.png"))
    credits = out_dir / "credits.txt"
    credits.write_text(_credits(scene, track), encoding="utf-8")
    result.outputs.append(credits)
    meta = {"render_hash": render_hash(scene), "title": scene.title, "outputs": [p.name for p in result.outputs],
            "warnings": [w.code for w in result.warnings]}
    tmp = out_dir / "render.json.part"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, out_dir / "render.json")
    log(f"[{scene.id}] finished -> {out_dir}")
    return result


def scene_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml")) if p.is_dir() else [p]
    return files


def render_batch(paths: list[Path], project: Project, depth_source: DepthSource, *, force: bool = False,
                 log: Log = print) -> BatchSummary:
    summary = BatchSummary()
    for f in scene_files(paths):
        try:
            scene = load_scene(f, project)
            r = render_scene(scene, project, depth_source, force=force, log=log)
        except SceneError as e:
            summary.failed.append((f.name, str(e)))
            log(f"[{f.name}] failed: {e}")
            continue
        except Exception as e:  # noqa: BLE001 - one broken scene must not stop the batch
            reason = getattr(e, "reason", None) or f"{type(e).__name__}: {e}"
            summary.failed.append((f.name, reason))
            log(f"[{f.name}] failed: {reason}")
            continue
        (summary.skipped if r.skipped else summary.done).append(scene.id)
    return summary
