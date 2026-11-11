"""The `stereo` command."""
from __future__ import annotations

import typer

app = typer.Typer(help="Stereogram video engine", no_args_is_help=True, add_completion=False)


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


@app.callback()
def main() -> None:
    """Stereogram video engine."""
