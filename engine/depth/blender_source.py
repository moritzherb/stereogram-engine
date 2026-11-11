"""Finding and driving Blender (which runs as a separate background process)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

MAC_APP = Path("/Applications/Blender.app/Contents/MacOS/Blender")
SUPPORTED_VERSIONS = {(4, 5), (4, 2)}
SCRIPT = Path(__file__).resolve().parents[1] / "blender" / "render_depth.py"


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
