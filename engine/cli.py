"""The `stereo` command."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import typer

app = typer.Typer(help="Stereogram video engine", no_args_is_help=True, add_completion=False)


def _project(ctx: typer.Context):
    from engine.project import Project

    return Project(Path(ctx.obj["project"]).resolve())


def _depth_source(project, log):
    if os.environ.get("STEREO_DEPTH_SOURCE") == "fake":
        from engine.depth.fake import FakeDepthSource

        return FakeDepthSource()
    from engine.depth.blender_source import BlenderDepthSource, find_blender

    blender = find_blender()
    if blender is None:
        typer.echo("Blender was not found. Run `stereo check` for help installing it.")
        raise typer.Exit(1)
    return BlenderDepthSource(blender, project.cache, log=log)


@app.callback()
def main(ctx: typer.Context,
         project: Path = typer.Option(Path.cwd(), "--project", help="Project folder (default: current folder)")):
    """Stereogram video engine."""
    ctx.obj = {"project": project}


@app.command()
def check() -> None:
    """Check that Python, ffmpeg, the font and Blender all work on this computer."""
    from engine.check_env import run_checks

    items = run_checks()
    for item in items:
        status = "OK     " if item.ok else "MISSING"
        typer.echo(f"{status} {item.name:8s} {item.detail}")
        if not item.ok and item.fix:
            typer.echo(f"        -> {item.fix}")
    if not all(i.ok for i in items):
        raise typer.Exit(1)
    typer.echo("All good. You're ready to render.")


@app.command()
def render(ctx: typer.Context,
           paths: List[Path] = typer.Argument(..., help="Scene files or folders of scene files"),
           force: bool = typer.Option(False, "--force", help="Render again even if nothing changed")) -> None:
    """Render scenes into Shorts and long-video segments (output/<scene-id>/)."""
    from engine.pipeline import render_batch

    project = _project(ctx)
    summary = render_batch(paths, project, _depth_source(project, typer.echo), force=force, log=typer.echo)
    typer.echo("")
    if summary.done:
        typer.echo(f"Done: {', '.join(summary.done)}")
    if summary.skipped:
        typer.echo(f"Skipped (unchanged): {', '.join(summary.skipped)}")
    for name, reason in summary.failed:
        typer.echo(f"Failed: {name}: {reason}")
    if summary.failed:
        raise typer.Exit(1)


@app.command()
def preview(ctx: typer.Context, scene_file: Path = typer.Argument(..., help="Scene file")) -> None:
    """Quick 5-second, half-size preview of a Short (output/<scene-id>/preview.mp4)."""
    from engine.pipeline import render_scene
    from engine.scene import SceneError, load_scene

    project = _project(ctx)
    try:
        scene = load_scene(scene_file, project)
    except SceneError as e:
        typer.echo(str(e))
        raise typer.Exit(1)
    result = render_scene(scene, project, _depth_source(project, typer.echo), preview=True, log=typer.echo)
    typer.echo(f"Preview: {result.outputs[0]}")


@app.command("compile")
def compile_cmd(ctx: typer.Context,
                compilation_file: Path = typer.Argument(..., help="Compilation file")) -> None:
    """Build a long 16:9 video from rendered scenes (output/compilations/<id>/)."""
    from engine.compile import MissingSegments, compile_long_video
    from engine.scene import SceneError, load_compilation

    project = _project(ctx)
    try:
        comp = load_compilation(compilation_file, project)
        video = compile_long_video(comp, project, log=typer.echo)
    except SceneError as e:
        typer.echo(str(e))
        raise typer.Exit(1)
    except MissingSegments as e:
        typer.echo(f"{e}. Run: stereo render scenes/")
        raise typer.Exit(1)
    typer.echo(f"\nVideo: {video}")
    typer.echo("Chapters (paste into the YouTube description):")
    typer.echo((video.parent / "chapters.txt").read_text())
