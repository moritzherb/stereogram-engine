"""Speed check: python tests/bench_stereo.py  (target <= 50 ms/frame on the Mac)."""
import time

import numpy as np

from engine.stereo import render_stereogram

H, W = 1920, 1080
yy, xx = np.mgrid[0:H, 0:W]
depth = np.clip(1 - ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (300**2), 0, 1).astype(np.float32)
pattern = np.random.default_rng(0).integers(0, 256, (512, 154, 3), dtype=np.uint8)
render_stereogram(depth, pattern, repeat_px=154)  # JIT warm-up
n = 10
t0 = time.perf_counter()
for _ in range(n):
    render_stereogram(depth, pattern, repeat_px=154)
print(f"{(time.perf_counter() - t0) / n * 1000:.1f} ms per 1080x1920 frame")
