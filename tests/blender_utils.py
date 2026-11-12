import json
import subprocess
from pathlib import Path

import numpy as np

from engine.depth.base import PngDepthSequence
from engine.depth.blender_source import SCRIPT, find_blender

BLENDER = find_blender()
MAKE_MODEL = Path(__file__).resolve().parent / "blender" / "make_model.py"
SAFE = {"top": 0.10, "bottom": 0.20, "left": 0.05, "right": 0.15}


def run_job(tmp_path, source, **kw):
    out = tmp_path / "out dir"
    job = {"source": source, "animation": None, "motion": None, "speed": 1.0, "duration": 1, "fps": 10,
           "width": 108, "height": 192, "safe": SAFE, "reveal": False, "out_dir": str(out),
           "font": str(Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Montserrat-Black.ttf")}
    job.update(kw)
    jf = tmp_path / "job.json"
    jf.write_text(json.dumps(job))
    proc = subprocess.run([str(BLENDER), "-b", "--factory-startup", "--python-exit-code", "1",
                           "--python", str(SCRIPT), "--", str(jf)], capture_output=True, text=True, timeout=900)
    return proc, out


def frames(out):
    seq = PngDepthSequence(out, fps=10)
    return [seq.frame(i) for i in range(len(seq))], seq


def make_model(tmp_path, scale=1.0, loc=(0, 0, 0), clip=None, name="model.glb"):
    path = tmp_path / name
    args = [str(path), str(scale), *map(str, loc)] + ([clip] if clip else [])
    subprocess.run([str(BLENDER), "-b", "--factory-startup", "--python-exit-code", "1",
                    "--python", str(MAKE_MODEL), "--", *args], check=True, capture_output=True, timeout=300)
    return path


def bbox(frame, thresh=0.02):
    ys, xs = np.nonzero(frame > thresh)
    return xs.min(), ys.min(), xs.max(), ys.max()
