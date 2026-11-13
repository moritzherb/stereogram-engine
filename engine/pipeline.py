"""Rendering scenes into finished videos (one scene, a batch, or a quick preview)."""
from __future__ import annotations

import hashlib
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
from engine.depth.soften import limit_slope, soften, softness_sigma
from engine.encode import VideoWriter, copy_silent, mux_music, pick_music
from engine.formats import LONG, SHORT, Format, preview_of
from engine.longform import LongFromShort
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


COVERAGE_DEPTH = 0.3  # edge pixels below this depth are only partly covered by the object
MAX_PATTERN_SHIFT = 0.4  # px of pattern shift per px: steeper depth tears the pattern (visible outlines)


def prepare_depth(depth: np.ndarray, scene: Scene, fmt: Format) -> np.ndarray:
    """Depth as it goes into the stereogram: the object lifted off the
    background (so it clearly stands in front of it), then gently softened."""
    lift = scene.advanced.lift
    if lift > 0:
        # Anti-aliased edge pixels get only part of the lift (in proportion to how
        # much of them the object covers); lifting them fully turns soft, slightly
        # noisy outlines into ragged steps that flicker in the stereogram.
        coverage = np.clip(depth / COVERAGE_DEPTH, 0.0, 1.0)
        depth = (coverage * lift + (1.0 - lift) * depth).astype(np.float32)
    softness = scene.advanced.edge_softness
    if softness > 0:
        # Pattern shift per unit depth is about depth_strength * repeat / 2 px.
        shift_per_depth = scene.advanced.depth_strength * _repeat_px(scene, fmt) / 2.0
        depth = limit_slope(np.ascontiguousarray(depth, dtype=np.float32),
                            MAX_PATTERN_SHIFT / softness / shift_per_depth)
    return soften(depth, softness_sigma(softness, fmt))


class _FrameRenderer:
    """Turns depth frames into stereogram frames, keeping a few for checks."""

    def __init__(self, scene: Scene, fmt: Format, seq, strip: np.ndarray):
        self.scene, self.fmt, self.seq, self.strip = scene, fmt, seq, strip
        self.repeat = _repeat_px(scene, fmt)
        base = SHORT if fmt.name.startswith("short") else LONG
        self.drift = scene.advanced.pattern_drift * fmt.width / base.width
        n = len(seq)
        self.sample_ids = sorted({0, n // 2, n - 1})
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.middle: np.ndarray | None = None

    def frames(self, log: Log, label: str):
        n = len(self.seq)
        step = max(1, n // 10)
        for i in range(n):
            depth = self.seq.frame(i)
            if not getattr(self.seq, "prepared", False):
                depth = prepare_depth(depth, self.scene, self.fmt)
            img = render_stereogram(depth, self.strip, repeat_px=self.repeat,
                                    depth_strength=self.scene.advanced.depth_strength,
                                    crosseyed=self.scene.advanced.viewing == "crosseyed",
                                    shift=round(i / self.seq.fps * self.drift))
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
    lines.append("Font: Montserrat (SIL Open Font License)")
    return "\n".join(lines) + "\n"


def engine_fingerprint() -> str:
    """Hash of the engine's own code: an engine update re-renders every scene."""
    h = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def full_hash(scene: Scene, project: Project) -> str:
    """render_hash plus the music track the scene would pick right now and the engine version."""
    track = _music_for(scene, project)
    stamp = None
    if track is not None and track.exists():
        st = track.stat()
        stamp = [str(track), st.st_mtime_ns, st.st_size]
    return hashlib.sha256(json.dumps([render_hash(scene), stamp, engine_fingerprint()]).encode()).hexdigest()


def _is_up_to_date(scene: Scene, project: Project, out_dir: Path) -> bool:
    meta_file = out_dir / "render.json"
    if not meta_file.exists():
        return False
    try:
        meta = json.loads(meta_file.read_text())
    except json.JSONDecodeError:
        return False
    return meta.get("render_hash") == full_hash(scene, project) and all((out_dir / o).exists() for o in meta["outputs"])


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
        strip = pattern_strip(short, _repeat_px(short, fmt))  # fail fast on a bad image
        warn(check_pattern(strip))
        seq = depth_source.render(short, fmt)
        warn(check_safe_area(seq, fmt) + check_motion(seq))
        outputs, _, _ = _render_short(short, project, fmt, seq, strip, out_dir, name="preview",
                                      with_music=False, log=log)
        result.outputs = outputs
        return result

    if not force and _is_up_to_date(scene, project, out_dir):
        log(f"[{scene.id}] unchanged, skipping")
        result.skipped = True
        return result
    (out_dir / "render.json").unlink(missing_ok=True)

    formats = [(name, SHORT if name == "short" else LONG) for name in ("short", "long") if name in scene.formats]
    strips = {name: pattern_strip(scene, _repeat_px(scene, fmt)) for name, fmt in formats}  # before Blender runs
    for strip in strips.values():
        warn(check_pattern(strip))
    track: Path | None = None
    middle_long: np.ndarray | None = None
    middle_short: np.ndarray | None = None
    short_seq = None
    for fmt_name, fmt in formats:
        strip = strips[fmt_name]
        if short_seq is None:
            short_seq = depth_source.render(scene, SHORT)  # one depth render serves both formats
        if fmt_name == "short":
            seq = short_seq
        else:
            seq = LongFromShort(short_seq, LONG, transform=lambda d: prepare_depth(d, scene, SHORT))
        warn(check_safe_area(seq, fmt) + check_motion(seq))
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
    meta = {"render_hash": full_hash(scene, project), "title": scene.title, "outputs": [p.name for p in result.outputs],
            "warnings": [w.code for w in result.warnings]}
    tmp = out_dir / "render.json.part"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, out_dir / "render.json")
    log(f"[{scene.id}] finished -> {out_dir}")
    return result


def log_details(e: BaseException, log: Log) -> None:
    """Show the end of Blender's log, if the error carries one."""
    tail = getattr(e, "log_tail", "")
    if tail:
        log("  Last lines of the Blender log (send these to your developer):")
        for line in tail.splitlines()[-15:]:
            log(f"    {line}")


def scene_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml")) if p.is_dir() else [p]
    return files


def render_batch(paths: list[Path], project: Project, depth_source: DepthSource, *, force: bool = False,
                 log: Log = print) -> BatchSummary:
    summary = BatchSummary()
    seen: dict[str, str] = {}
    for f in scene_files(paths):
        try:
            scene = load_scene(f, project)
            if scene.id in seen:
                raise SceneError(f"{f.name}: id: '{scene.id}' is already used by {seen[scene.id]}. "
                                 f"Give each scene its own id.")
            seen[scene.id] = f.name
            r = render_scene(scene, project, depth_source, force=force, log=log)
        except SceneError as e:
            summary.failed.append((f.name, str(e)))
            log(f"[{f.name}] failed: {e}")
            continue
        except Exception as e:  # noqa: BLE001 - one broken scene must not stop the batch
            reason = getattr(e, "reason", None) or f"{type(e).__name__}: {e}"
            summary.failed.append((f.name, reason))
            log(f"[{f.name}] failed: {reason}")
            log_details(e, log)
            continue
        (summary.skipped if r.skipped else summary.done).append(scene.id)
    return summary
