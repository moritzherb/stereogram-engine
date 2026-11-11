"""Stand-in for Blender used by tests: writes blank 16-bit frames for a job.

Behaviour is controlled by environment variables:
  FAKE_BLENDER_COUNT  file whose number is incremented on every call
  FAKE_BLENDER_MODE   "ok" (default) | "unknown_clip" | "crash"
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

count_file = os.environ.get("FAKE_BLENDER_COUNT")
if count_file:
    p = Path(count_file)
    p.write_text(str(int(p.read_text() or 0) + 1 if p.exists() else 1))
if "--version" in sys.argv:
    print("Blender 4.5.16 LTS")
    sys.exit(0)
mode = os.environ.get("FAKE_BLENDER_MODE", "ok")
if mode == "unknown_clip":
    sys.stderr.write("UNKNOWN_CLIP: Swimm; available: Idle, Swim\n")
    sys.exit(3)
if mode == "crash":
    print("some blender log line")
    sys.exit(1)
job = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
out = Path(job["out_dir"])
out.mkdir(parents=True, exist_ok=True)
n = max(1, round(job["duration"] * job["fps"]))
w, h = job["width"], job["height"]
for i in range(n):
    arr = np.zeros((h, w), np.uint16)
    arr[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = 40000
    Image.fromarray(arr).save(out / f"depth_{i + 1:04d}.png")
    print(f"FRAME {i + 1}/{n}", flush=True)
if job.get("reveal"):
    Image.fromarray(np.full((h, w, 3), 120, np.uint8)).save(out / "reveal.png")
(out / "meta.json").write_text(json.dumps({"frames": n, "clips": [], "seconds_per_frame": 0.0}))
