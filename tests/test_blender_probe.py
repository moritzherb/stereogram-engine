import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from engine.depth.blender_source import find_blender

BLENDER = find_blender()
SCRIPT = Path(__file__).resolve().parents[1] / "engine" / "blender" / "render_depth.py"
pytestmark = pytest.mark.skipif(BLENDER is None, reason="Blender not installed")


def test_probe_renders_three_16bit_frames(tmp_path):
    job = tmp_path / "job.json"
    out = tmp_path / "out dir"
    job.write_text(json.dumps({"probe": True, "out_dir": str(out)}))
    subprocess.run([str(BLENDER), "-b", "--factory-startup", "--python", str(SCRIPT), "--", str(job)],
                   check=True, capture_output=True, timeout=600)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["frames"] == 3
    img = Image.open(out / "depth_0001.png")
    assert img.size == (108, 192)
    arr = np.asarray(img)
    assert arr.dtype in (np.uint16, np.int32)
    assert arr.max() > 0 and arr.min() == 0
