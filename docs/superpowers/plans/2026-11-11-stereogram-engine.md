# Stereogram Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `stereo` command-line engine that turns scene files (3D model, animation, pattern, text, music) into upload-ready animated stereogram Shorts, 16:9 segments, and compiled long YouTube videos.

**Architecture:** Python orchestrates every stage. Blender 4.5 LTS (fallback 4.2 LTS) runs as a separate background process and writes 16-bit depth PNG frames into a cache. A Numba stereogram core turns depth and a pattern strip into RGB frames, a Pillow/NumPy compositor adds text and reveals, and ffmpeg (from `imageio-ffmpeg`) encodes and mixes music. Every stage after Blender sees only the `DepthSequence` interface.

**Tech Stack:** Python ≥3.11, numpy, numba, Pillow, PyYAML, pydantic v2, typer, imageio-ffmpeg, pytest; Blender 4.5 LTS (`bpy`, inside Blender only).

**Spec:** `docs/superpowers/specs/2026-11-11-stereogram-engine-design.md`

## Global Constraints

- `requires-python = ">=3.11"`; must run on macOS 12 Monterey, Intel (MacBook Pro 2016) and on Linux.
- No Homebrew dependency: ffmpeg comes from `imageio_ffmpeg.get_ffmpeg_exe()`, overridable with the env var `STEREO_FFMPEG`.
- Blender is found via env `STEREO_BLENDER`, then `blender` on PATH, then `/Applications/Blender.app/Contents/MacOS/Blender`. Supported major.minor: 4.5 (preferred), 4.2 (fallback). Blender 5.x is rejected with a clear message.
- **No emojis** in any generated or user-supplied on-screen text; scene validation rejects them.
- On-screen text: one bundled OFL font (`assets/fonts/Lato-Regular.ttf` + `assets/fonts/OFL.txt`), white at 85 % opacity, soft black shadow, text height ≈ 3 % of frame height.
- Video: H.264 High, `yuv420p`, 30 fps, `+faststart`; final deliverables CRF 16 preset `slow`; intermediate `long_segment.mp4` CRF 12.
- Audio: AAC 192 kbps, `loudnorm=I=-14:TP=-1.5:LRA=11`, 1 s fade in and 1 s fade out.
- Formats: short = 1080×1920, repeat 154 px; long = 1920×1080, repeat 175 px. Preview = half resolution, 5 s, short only.
- Stereogram: separation `s(z) = round((1 − μz)·E / (2 − μz))` with `E = 2·repeat_px` (far plane = repeat width), default μ = 0.33, 3× horizontal oversampling, linking anchored at the centre column.
- Depth convention everywhere: float32 in `[0, 1]`, 1 = nearest, 0 = background.
- Relative asset paths in scene files resolve against `<project>/assets/`.
- Every subprocess call passes arguments as a list (never a shell string).
- All outputs are written to a temp name in the same folder and renamed into place only when complete.

## Review Focus

1. **Paths with spaces** (`assets/patterns/my ocean pic.jpg`, project inside `~/My Videos/`): everything must still work. Pinned in Tasks 9, 10 and 12.
2. **Junk files in asset folders** (`.DS_Store`, `notes.txt` in `assets/music/`): ignored, never picked as music. Pinned in Task 9.
3. **Odd pattern images** (RGBA PNG, grayscale, CMYK JPEG, 8000 px photos, images narrower than the repeat width): converted to RGB and resized, never crash. Pinned in Task 4.
4. **Interrupted renders** (laptop lid closed, Ctrl+C): partial files never count as done; the next run redoes that scene. Pinned in Task 12.
5. **Models with odd scale or pivot** (a 500-unit model off the origin, a 0.01-unit model): auto-framing still fills the safe area. Pinned in Task 7.

---

## File Structure

```
pyproject.toml                    package "stereogram-engine", script stereo = engine.cli:app
engine/__init__.py
engine/formats.py                 Format, SafeArea, SHORT, LONG, get_format, preview_of
engine/scene.py                   Scene/Compilation models, loading, validation, hashes
engine/stereo.py                  Numba stereogram core
engine/decode.py                  depth recovery + self-check
engine/patterns.py                image + generated pattern strips, quality scores
engine/depth/__init__.py
engine/depth/base.py              DepthSequence protocol, ArrayDepthSequence, PngDepthSequence
engine/depth/blender_source.py    find_blender, BlenderDepthSource, cache
engine/blender/render_depth.py    runs INSIDE Blender (bpy only)
engine/compose.py                 overlays, timelines, cards, thumbnail
engine/encode.py                  VideoWriter, music, silent copy, probe
engine/checks.py                  quality warnings
engine/project.py                 Project paths
engine/pipeline.py                render_scene, render_batch
engine/compile.py                 compile_long_video, chapters
engine/check_env.py               `stereo check` logic
engine/cli.py                     typer app
assets/{models,patterns,music}/.gitkeep, assets/fonts/{Lato-Regular.ttf,OFL.txt}
scenes/example-heart.yaml, scenes/example-text.yaml, compilations/example.yaml
tests/...                         one test file per module
README.md                         Mac setup + usage
```

---

### Task 1: Scaffold, formats, environment check, Blender depth probe (Milestone 1)

**Files:**
- Create: `pyproject.toml`, `engine/__init__.py`, `engine/formats.py`, `engine/check_env.py`, `engine/cli.py`, `engine/depth/__init__.py`, `engine/depth/blender_source.py` (only `find_blender`, `blender_version` for now), `engine/blender/render_depth.py` (probe mode only), `.gitignore` (`output/`, `cache/`, `.venv/`, `__pycache__/`), `README.md` (Mac setup section)
- Test: `tests/test_formats.py`, `tests/test_check_env.py`, `tests/test_blender_probe.py`

**Interfaces:**
- Produces:
  - `SafeArea(top: float, bottom: float, left: float, right: float)` (frozen dataclass, fractions of frame)
  - `Format(name: str, width: int, height: int, repeat_px: int, safe: SafeArea)` (frozen)
  - `SHORT = Format("short", 1080, 1920, 154, SafeArea(0.10, 0.20, 0.05, 0.15))`
  - `LONG = Format("long", 1920, 1080, 175, SafeArea(0.06, 0.08, 0.05, 0.05))`
  - `get_format(name: str) -> Format` (ValueError for unknown), `preview_of(fmt: Format) -> Format` (half width, height and repeat_px; name `"<name>-preview"`)
  - `find_blender() -> Path | None`; `blender_version(path: Path) -> tuple[int, int]` (parses `Blender 4.5.3` from `--version`)
  - `run_checks(include_blender: bool = True) -> list[CheckItem]` where `CheckItem(name: str, ok: bool, detail: str, fix: str)`
  - CLI: `stereo check` prints one line per item (`OK`/`MISSING`, detail, fix) and exits 1 if any item fails.
  - `render_depth.py` job key `"probe": true`: renders a cube spinning, 3 frames at 108×192, writes `depth_0001.png…` (16-bit) and `meta.json` with `{"frames": 3, "seconds_per_frame": float, "engine": str}`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_formats.py
def test_short_and_long_constants():
    assert (SHORT.width, SHORT.height, SHORT.repeat_px) == (1080, 1920, 154)
    assert (LONG.width, LONG.height, LONG.repeat_px) == (1920, 1080, 175)
def test_preview_halves():
    p = preview_of(SHORT); assert (p.width, p.height, p.repeat_px, p.name) == (540, 960, 77, "short-preview")
def test_unknown_format():
    with pytest.raises(ValueError): get_format("square")

# tests/test_check_env.py
def test_checks_report_python_and_ffmpeg():
    items = {i.name: i for i in run_checks(include_blender=False)}
    assert items["python"].ok and items["ffmpeg"].ok and items["numba"].ok
def test_blender_version_rejects_5(monkeypatch):  # fake --version output "Blender 5.0.1"
    ... assert item "blender" not ok and "4.5" in item.fix
def test_find_blender_env_override(tmp_path, monkeypatch):  # STEREO_BLENDER points to a fake executable
    assert find_blender() == fake_path

# tests/test_blender_probe.py  (pytest.mark.skipif(find_blender() is None))
def test_probe_renders_three_16bit_frames(tmp_path):
    # run blender -b --python engine/blender/render_depth.py -- job.json with {"probe": true, "out_dir": ...}
    img = PIL.Image.open(tmp_path / "depth_0001.png"); assert img.mode in ("I;16", "I") and img.size == (108, 192)
    arr = np.asarray(img); assert arr.max() > 0 and arr.min() == 0   # cube visible, background zero
```

- [ ] **Step 2: Run tests, expect failures** — `pytest tests/test_formats.py tests/test_check_env.py -v` → FAIL (modules missing). The Blender test is SKIPPED in the cloud container (download.blender.org is blocked there).

- [ ] **Step 3: Implement**
  - `pyproject.toml`: deps `numpy>=1.26, numba>=0.59, pillow>=10.1, pyyaml>=6, pydantic>=2.5, typer>=0.12, imageio-ffmpeg>=0.5`; dev extra `pytest`.
  - Font: download `https://github.com/google/fonts/raw/main/ofl/lato/Lato-Regular.ttf` and `.../ofl/lato/OFL.txt` into `assets/fonts/`; commit both. The check item `font` verifies it loads with Pillow.
  - `render_depth.py` probe: Cycles, device CPU, samples 8, `max_bounces = 0`, view layer Mist pass enabled, World mist start = camera distance to cube front, depth = cube size; compositor `Render Layers.Mist → Invert → File Output` (PNG, BW, 16-bit), so 1 = near and 0 = background. Time each frame. Parse args after `--` via `sys.argv`.
  - `check_env.py` Blender item: run the probe in a temp dir with a 600 s timeout; detail = `"Blender 4.5.x, 2.1 s/frame -> about 32 min of depth rendering per 30 s Short"` (900 × s/frame).
  - README "Setup on your Mac": python.org Python 3.11 installer; Blender 4.5 LTS from blender.org into Applications (if it won't open on macOS 12, use 4.2 LTS); `git clone`, `python3 -m venv .venv`, `source .venv/bin/activate`, `pip install -e .`, `stereo check`.

- [ ] **Step 4: Run tests** — `pytest -v` → formats/check_env PASS, probe SKIPPED.

- [ ] **Step 5: Commit** — `git commit -m "feat: scaffold, formats, environment check and Blender depth probe"`

- [ ] **Step 6: Human gate (Milestone 1).** Ask the owner to follow the README on the MacBook and paste the `stereo check` output. Record the Blender version and seconds per frame in `README.md` under "Tested on". If 4.5 fails, retry with 4.2 LTS. If both fail, stop and re-plan Tasks 7–8 (pure-Python rigid-motion depth source). **Tasks 2–6 don't depend on Blender, so continue with them while waiting.**

---

### Task 2: Stereogram core

**Files:**
- Create: `engine/stereo.py`
- Test: `tests/test_stereo.py`

**Interfaces:**
- Consumes: nothing (pure numpy).
- Produces:
  - `separation(z: np.ndarray | float, eye_px: float, mu: float) -> np.ndarray | float` (unrounded)
  - `render_stereogram(depth: np.ndarray, pattern: np.ndarray, *, repeat_px: int, depth_strength: float = 0.33, oversample: int = 3, crosseyed: bool = False) -> np.ndarray`
    - `depth`: float32 (H, W) in [0, 1]; `pattern`: uint8 strip (Hp, repeat_px, 3) from Task 4; returns uint8 (H, W, 3).

- [ ] **Step 1: Write failing tests**

```python
def test_flat_depth_is_pure_repetition():
    pat = random_strip(64, 154, seed=1); out = render_stereogram(np.zeros((64, 1080), np.float32), pat, repeat_px=154)
    assert np.array_equal(out[:, 154:], out[:, :-154])           # exact period = repeat_px
def test_flat_depth_anchored_at_centre():
    # centre column 540 shows pattern column 0
    assert np.array_equal(out[:, 540], pat[np.arange(64) % pat.shape[0], 0])
def test_raised_square_has_expected_separation():
    d = np.zeros((64, 1080), np.float32); d[:, 400:700] = 1.0
    out = render_stereogram(d, pat, repeat_px=154, oversample=1)
    s1 = round(separation(1.0, 308, 0.33))                        # == 124
    assert np.array_equal(out[10, 560:600], out[10, 560 - s1:600 - s1])
def test_identical_depth_gives_identical_frames():  # no flicker
    assert np.array_equal(render_stereogram(d, pat, repeat_px=154), render_stereogram(d.copy(), pat, repeat_px=154))
def test_static_region_unchanged_when_object_moves_vertically_elsewhere():
    # object in rows 0-20 moves; rows 40-63 (flat) identical between frames
def test_crosseyed_inverts_depth():
    assert np.array_equal(render_stereogram(d, pat, repeat_px=154, crosseyed=True), render_stereogram(1 - d, pat, repeat_px=154))
def test_output_shape_dtype(): (H, W, 3), uint8
```

- [ ] **Step 2: Run** `pytest tests/test_stereo.py -v` → FAIL (import error).

- [ ] **Step 3: Implement.** Steps per frame: upsample depth horizontally ×`oversample` (linear interpolation per row); `E = 2 * repeat_px * oversample`; if crosseyed, `depth = 1 - depth`. Per row (Numba `@njit(parallel=True)` over rows, `cache=True`), Thimbleby/Inglis/Witten linking with hidden-surface removal:

```
same[x] = x for all x
for x in 0..W-1:
    s = round(separation(z[x], E, mu)); left = x - s//2; right = left + s
    if 0 <= left and right < W:
        t = 1
        do:                                   # hidden-surface removal
            zt = z[x] + 2*(2 - mu*z[x])*t/(mu*E)
            visible = z[x-t] < zt and z[x+t] < zt   (out-of-range neighbours count as 0)
            t += 1
        while visible and zt < 1
        if visible:
            l = same[left]
            while l != left and l != right:
                if l < right: left = l
                else: same[left] = right; left = right; right = l
                l = same[left]
            same[left] = right
```

  Colouring (centre-anchored): each chain of linked pixels takes the colour of its member nearest the centre column `c = W//2`. Compute `root[x]` right-to-left (`root[x] = x if same[x]==x else root[same[x]]`), keep `best[root]` = member with minimal `|x − c|`, then `col = (best − c) mod (repeat_px*oversample)` and colour from the pattern (upsampled ×oversample horizontally, nearest) at row `y % Hp`. Finally downsample by averaging each group of `oversample` columns and round to uint8.

- [ ] **Step 4: Run** `pytest tests/test_stereo.py -v` → PASS. Add `tests/bench_stereo.py` (not collected by default, run with `python tests/bench_stereo.py`) that prints ms/frame at 1080×1920. Target ≤ 50 ms on the MacBook; record the Linux number in the commit message.

- [ ] **Step 5: Commit** — `git commit -m "feat: centre-anchored stereogram core with hidden-surface removal"`

---

### Task 3: Decoder self-check

**Files:**
- Create: `engine/decode.py`
- Test: `tests/test_decode.py`

**Interfaces:**
- Consumes: `separation` from Task 2.
- Produces:
  - `recover_depth(img: np.ndarray, *, repeat_px: int, depth_strength: float = 0.33) -> np.ndarray`: float32 (H, W), NaN where undetermined.
  - `SelfCheck(score: float, passed: bool)`; `self_check(depth, img, *, repeat_px, depth_strength=0.33, tolerance=0.08, threshold=0.95) -> SelfCheck`

- [ ] **Step 1: Write failing tests**

```python
def test_recovers_square():   # noise pattern, d[:, 400:700]=1 on 1080x200
    assert self_check(d, render_stereogram(d, pat, repeat_px=154), repeat_px=154).passed
def test_recovers_ramp():     # d = linear 0..0.8 left->right
    assert self_check(...).passed
def test_detects_wrong_depth():
    img = render_stereogram(d, pat, repeat_px=154)
    assert not self_check(np.zeros_like(d), img, repeat_px=154).passed
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement.** For each pixel, test integer separations `s ∈ [round(sep(1)), round(sep(0))]`, with cost = sum of absolute RGB differences between windows centred at `x − s/2` and `x + s/2` (window 9 px wide × 3 rows); take the argmin and invert with `z = (E − 2s)/(μ(E − s))`, clipping to [0, 1]. NaN where the best cost isn't clearly below the second best (ratio > 0.8). Interior mask for `self_check`: exclude columns within `E/2` of the borders and pixels within `repeat_px` of depth edges (|∇d| > 0.05). Score = fraction of interior non-NaN pixels with `|recovered − depth| ≤ tolerance`; NaNs count as misses.

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: depth recovery self-check for stereograms"`

---

### Task 4: Patterns (Milestone 2 tuning samples)

**Files:**
- Create: `engine/patterns.py`, `scripts/make_samples.py`
- Test: `tests/test_patterns.py`

**Interfaces:**
- Produces:
  - `STYLES = ("noise", "swirl", "cells", "stripes", "dots")`; `PALETTES: dict[str, list[tuple[int, int, int]]]` with at least `default, ocean, sunset, forest, candy, mono`
  - `generate_pattern(style: str, *, repeat_px: int, palette: str = "default", seed: int = 0, height_px: int = 512) -> np.ndarray` (uint8 (height_px, repeat_px, 3), tileable both ways)
  - `load_image_pattern(path: Path, *, repeat_px: int) -> np.ndarray` (uint8 (H, repeat_px, 3))
  - `make_tileable(img: np.ndarray) -> np.ndarray`
  - `seam_score(strip) -> float`, `detail_score(strip) -> float`; constants `SEAM_MAX`, `DETAIL_MAX`

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("style", STYLES)
def test_generated_shape_and_tileable(style):
    s = generate_pattern(style, repeat_px=154, seed=3)
    assert s.shape == (512, 154, 3) and s.dtype == np.uint8 and seam_score(s) <= SEAM_MAX
def test_seed_determinism():  same seed equal, different seed not equal
def test_unknown_style_lists_choices():
    with pytest.raises(ValueError, match="noise, swirl"): generate_pattern("plaid", repeat_px=154)
@pytest.mark.parametrize("mode", ["RGBA", "L", "CMYK", "P"])
def test_image_modes_become_rgb(tmp_path, mode):          # Review Focus 3
    assert load_image_pattern(saved_img(tmp_path, mode, (300, 200)), repeat_px=154).shape[1:] == (154, 3)
def test_huge_and_tiny_images(tmp_path):                   # 8000x6000 and 60x40
    for size in [(8000, 6000), (60, 40)]: assert load_image_pattern(...).shape[1] == 154
def test_image_pattern_made_seamless(tmp_path):  # photo with hard left/right difference -> seam_score <= SEAM_MAX
def test_detail_score_flags_pixel_noise():
    fine = rng.integers(0, 256, (512, 154, 3), dtype=np.uint8)
    assert detail_score(fine) > DETAIL_MAX >= detail_score(generate_pattern("swirl", repeat_px=154))
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement.** Images: `ImageOps.exif_transpose`, `convert("RGB")` (CMYK/P/L/RGBA all go through this; RGBA is composited on black first), resize to width `repeat_px` keeping aspect (LANCZOS), then tile vertically until height ≥ `repeat_px`; `make_tileable` cross-blends a 15 % band at the left and right edges and at the top and bottom edges with the opposite side. Vertical anti-repetition: stack 3 copies of the tile, each shifted horizontally by `round(k · 0.382 · repeat_px)` (k = 0, 1, 2), giving a 3×-tall strip. Generators: smooth value noise (≥ 4 px features at repeat 154) built from periodic noise so they tile by construction, then mapped through the palette. `seam_score` = mean |col0 − col−1| ÷ mean adjacent-column difference (SEAM_MAX = 1.5). `detail_score` = mean absolute horizontal gradient of the luminance (DETAIL_MAX tuned so that per-pixel uniform noise fails and every generated style passes).
  - `scripts/make_samples.py`: writes `output/samples/*.png`. These are 1080×1920 stereograms of a sphere, a heart silhouette and the word "HI" (depth from simple numpy masks), using the noise, swirl and cells styles at repeat 140, 154 and 170 and depth strength 0.25, 0.33 and 0.40, with the settings printed on a small label at the bottom.

- [ ] **Step 4: Run** → PASS. Run `python scripts/make_samples.py`.

- [ ] **Step 5: Commit** — `git commit -m "feat: generated and image patterns with quality scores; tuning samples"`

- [ ] **Step 6: Human gate (Milestone 2).** Send the sample PNGs to the owner to view on their iPhone; ask which repeat and strength read best. Update the `SHORT`/`LONG` `repeat_px` and the default μ if they pick different values (update constants, tests and the spec together).

---

### Task 5: Scene and compilation files

**Files:**
- Create: `engine/scene.py`, `engine/project.py`, `scenes/example-heart.yaml`, `scenes/example-text.yaml`, `compilations/example.yaml`
- Test: `tests/test_scene.py`

**Interfaces:**
- Produces:
  - `Project(root: Path)` with properties `assets, models, patterns, music, fonts, scenes, compilations, output, cache` (Paths under root); `font_path` → `assets/fonts/Lato-Regular.ttf`.
  - `MOTIONS = ("spin", "float", "bob", "zoom_in", "fly_toward", "swim_past", "ripple", "pulse")`; `SHAPES = ("sphere", "cube", "torus", "cone", "heart", "star")`
  - `TextConfig(top: str | None = "This is a stereogram", bottom: str | None = "What do you see?", end: str | None = "Comment what you see")`
  - `GeneratedPattern(generate: str, palette: str = "default", seed: int | None = None)`
  - `Advanced(depth_strength: float = 0.33 (0.1–0.6), pattern_repeat: int | Literal["auto"] = "auto", viewing: Literal["parallel", "crosseyed"] = "parallel")`
  - `Scene(id, model: Path | None, text3d: str | None, shape: str | None, animation: str | None, motion: str | None, speed: float = 1.0, duration: float = 30, pattern: Path | GeneratedPattern, formats: list[Literal["short", "long"]] = ["short", "long"], text: TextConfig, music: Path | Literal["auto", "none"] = "auto", reveal: Literal["auto", "on", "off"] = "auto", title: str | None, advanced: Advanced)`
    - Exactly one of `model`, `text3d`, `shape`. `id` matches `^[a-z0-9][a-z0-9-]*$`. `duration` is 1–180, `speed` > 0. Default pattern = `GeneratedPattern("noise")`; a `None` seed becomes `int(sha256(id)[:8], 16)`. Default `title` = id with dashes → spaces, capitalised.
  - `Compilation(id, title: str, scenes: list[str] (min 1), segment_seconds: float = 35)`
  - `class SceneError(Exception)`: message format `"<file>: <field>: <problem>. <fix>"`
  - `load_scene(path: Path, project: Project) -> Scene` (resolves relative paths against `project.assets`, checks that the files exist)
  - `load_compilation(path: Path, project: Project) -> Compilation`
  - `contains_emoji(text: str) -> bool`
  - `reveal_enabled(scene: Scene, fmt_name: str) -> bool` (auto: short → False, long → True)
  - `depth_hash(scene: Scene, fmt: Format) -> str` (only fields affecting Blender: source, animation, motion, speed, duration, fmt name, size and safe area); `render_hash(scene: Scene) -> str` (all fields + file mtimes of the referenced assets)

- [ ] **Step 1: Write failing tests**

```python
def test_minimal_scene_defaults(tmp_project):   # yaml: {id: heart, shape: heart}
    s = load_scene(p, tmp_project)
    assert s.duration == 30 and s.formats == ["short", "long"] and s.text.bottom == "What do you see?"
    assert isinstance(s.pattern, GeneratedPattern) and s.pattern.seed is not None and s.title == "Heart"
def test_relative_paths_resolve_against_assets_with_spaces(tmp_project):   # Review Focus 1
    # pattern: "patterns/my ocean pic.jpg" exists -> resolved absolute path
def test_emoji_rejected_with_field_name():
    with pytest.raises(SceneError, match=r"text\.top: .*emoji"): load_scene(yaml_with(text={"top": "Look 👀"}))
def test_exactly_one_source():  both model+shape -> SceneError mentioning "model, text3d or shape"
def test_missing_model_file_message():  SceneError contains the resolved path and "assets/models"
def test_bad_id(): "Dolphin Jump" -> SceneError suggesting "dolphin-jump"
def test_unknown_motion_lists_presets(): match "spin, float, bob"
def test_reveal_auto(): reveal_enabled(s, "short") is False and reveal_enabled(s, "long") is True
def test_depth_hash_ignores_text_and_pattern(): change text/pattern -> same depth_hash, different render_hash
def test_compilation_needs_scenes(): scenes: [] -> SceneError
def test_examples_load(): every file in scenes/ and compilations/ loads
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** with pydantic v2 models; convert `ValidationError` into `SceneError` using the first error's `loc` joined by dots. Emoji detection: any character in the Unicode ranges U+1F000–U+1FAFF, U+2600–U+27BF, U+FE0F or U+200D. Example scenes: `example-heart` (shape heart, motion float, generate swirl, palette sunset) and `example-text` (text3d "HELLO", motion spin, generate cells, palette ocean); `compilations/example.yaml` lists both.

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: scene and compilation files with validation"`

---

### Task 6: Depth sequence interface

**Files:**
- Create: `engine/depth/base.py`
- Test: `tests/test_depth_base.py`

**Interfaces:**
- Produces:
  - `class DepthSequence(Protocol)`: attrs `fps: int, width: int, height: int, reveal: np.ndarray | None` (uint8 RGB (H, W, 3)); `__len__() -> int`; `frame(i: int) -> np.ndarray` (float32 (H, W) in [0, 1])
  - `ArrayDepthSequence(frames: list[np.ndarray], fps: int = 30, reveal: np.ndarray | None = None)`
  - `PngDepthSequence(directory: Path, fps: int = 30)`: reads `depth_%04d.png` (16-bit, value / 65535) and optional `reveal.png`; `len` comes from `meta.json["frames"]`
  - `class DepthSource(Protocol)`: `render(scene: Scene, fmt: Format) -> DepthSequence`

- [ ] **Step 1: Tests:** PNG round-trip (write a 16-bit PNG with known values → `frame(0)` within 1/65535); `len` from meta; reveal loaded when present and `None` when absent; `IndexError` for out-of-range frames.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: depth sequence interface and PNG sequence reader"`

---

### Task 7: Blender scene builder (Milestone 3, part 1)

**Files:**
- Modify: `engine/blender/render_depth.py` (add full job mode; keep probe)
- Test: `tests/test_blender_render.py` (skipped without Blender)

**Interfaces:**
- Consumes: job JSON written by Task 8.
- Produces: job schema and outputs:

```json
{"source": {"model": "/abs/x.glb"} | {"text3d": "HELLO"} | {"shape": "heart"},
 "animation": "Swim" | null, "motion": "float" | null, "speed": 1.0,
 "duration": 30, "fps": 30, "width": 1080, "height": 1920,
 "safe": {"top": 0.1, "bottom": 0.2, "left": 0.05, "right": 0.15},
 "reveal": true, "out_dir": "/abs/cache/..."}
```

  Outputs in `out_dir`: `depth_0001.png…` (16-bit gray, 1 = near), `reveal.png` (RGB, if requested), and `meta.json` `{"frames": N, "clips": [names], "seconds_per_frame": f, "blender": "4.5.x"}` written **last**. Exit code 3 with stderr line `UNKNOWN_CLIP: <name>; available: a, b` for a bad clip; exit code 4 with `IMPORT_FAILED: <reason>` for an unreadable model.

- [ ] **Step 1: Write failing tests** (all at 108×192, duration 1 s, fps 10 → 10 frames)

```python
def test_shape_heart_float_frames(tmp_path): meta["frames"] == 10; every frame has object pixels and zero background
def test_text3d(tmp_path): object pixels present; reveal.png exists
def test_object_inside_safe_area_all_frames(tmp_path):  # spin cube
    # bounding box of pixels > 0.02 lies within safe rect (1-px tolerance)
def test_odd_scale_and_offset_model(tmp_path):          # Review Focus 5
    # glb cube exported with scale 500 at (1000, -300, 50), and one with scale 0.01:
    # object bbox width or height >= 60 % of the safe area's width or height
def test_depth_normalised_per_scene_not_per_frame(tmp_path):  # fly_toward: max depth increases over time
    assert frames[0].max() < frames[-1].max()
def test_unknown_clip_exit_code(tmp_path): returncode == 3 and "available:" in stderr
def test_swim_past_crosses_frame(tmp_path): object centroid x moves left->right by > 50 % of width
```

  Test models are generated in the test with Blender itself (`bpy.ops.export_scene.gltf`) via a tiny helper script, so no binary fixtures are committed.

- [ ] **Step 2: Run** → FAIL on a machine with Blender (SKIP in cloud).

- [ ] **Step 3: Implement** (inside Blender, `bpy` only):
  - Empty scene; import `.glb/.gltf` (`import_scene.gltf`) or `.fbx` (`import_scene.fbx`). For `text3d`: a text object, extrude 0.1, bevel 0.01, Center alignment. Shapes: mesh primitives; `heart` and `star` are extruded 2D curves.
  - Parent all imported objects to an Empty `ROOT`. Normalise: scale ROOT so the rest-pose bounding-box diagonal = 2, centred at the origin.
  - Animation: if `animation` is set, find the action by name across armatures and objects (else exit 3), assign it, loop via NLA repeat to cover `duration·fps` frames, scale with `speed`. If not set and the model has exactly one action, use it.
  - Motion presets on ROOT (period seconds ÷ speed): `spin` Z 360° / 6 s; `float` Z sine amplitude 0.08 / 4 s + Z rotation ±15° / 7 s; `bob` Z sine 0.15 / 1.2 s; `zoom_in` scale 0.6 → 1.0 ease; `fly_toward` Y from +3 to 0 with scale 0.5 → 1.0; `swim_past` X from −1 to +1 frame-widths with Z sine 0.05; `ripple` Wave modifier on every mesh (height 0.05, width 0.5); `pulse` scale 1.0 → 1.08 → 1.0 every 1 s with a sharp rise.
  - Camera: orthographic, looking along +Y from −Y (front view). Evaluate world bounding boxes at every frame (depsgraph). Framing box = union over all frames, except `swim_past`, which uses the largest single-frame box with the X path scaled to the safe width. Set `ortho_scale` and `shift_x/shift_y` so the framing box fills the safe rect (5 % margin), keeping the aspect ratio.
  - Depth: Mist pass, `mist_settings.start` = distance to the box front, `depth = max(box_depth, 0.6·max(box_w, box_h))`, falloff linear; compositor inverts to 1 = near. Render engine and samples as chosen in Milestone 1. Frames go to `out_dir`, then `meta.json`.
  - Reveal: one extra render at the middle frame with Workbench (studio lighting, matcap, neutral grey background, colour from the material), saved as `reveal.png`. Fall back to Cycles at 16 samples if Workbench fails headless.

- [ ] **Step 4: Run on the owner's Mac** (or any machine with Blender 4.5): `pytest tests/test_blender_render.py -v` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: Blender scene builder with presets, auto-framing, depth and reveal renders"`

---

### Task 8: Blender depth source and cache (Milestone 3, part 2)

**Files:**
- Modify: `engine/depth/blender_source.py`
- Test: `tests/test_blender_source.py`

**Interfaces:**
- Consumes: `Scene`, `depth_hash` (Task 5), `Format` (Task 1), `PngDepthSequence` (Task 6), job schema (Task 7).
- Produces:
  - `class BlenderError(Exception)`: attrs `reason: str`, `log_tail: str` (last 40 lines)
  - `BlenderDepthSource(blender: Path, cache_dir: Path, timeout_s: int = 6 * 3600)`; `.render(scene, fmt) -> PngDepthSequence`
  - `build_job(scene: Scene, fmt: Format, out_dir: Path) -> dict`
  - Cache dir: `cache/<scene.id>/<fmt.name>/<depth_hash[:16]>/`; a cache hit means `meta.json` exists. The script's own sha256 is part of the hash.

- [ ] **Step 1: Write failing tests** using a **fake Blender**: a Python script made executable that reads the job JSON and writes N blank 16-bit PNGs plus `meta.json`, or exits 3 with `UNKNOWN_CLIP` text.

```python
def test_job_contents(): build_job(scene, SHORT, p)["safe"] == asdict(SHORT.safe) and paths absolute
def test_cache_hit_skips_blender(): second render() doesn't invoke fake (call-count file unchanged)
def test_text_change_keeps_cache(): scene with different text -> same cache dir
def test_unknown_clip_becomes_blender_error(): BlenderError.reason == "Animation clip 'Swimm' not found. Available: Swim, Idle"
def test_partial_cache_is_redone(): dir with frames but no meta.json -> fake runs again   # Review Focus 4
def test_cache_and_project_paths_with_spaces(tmp_path): project in "my videos/" works      # Review Focus 1
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (render into `<dir>.tmp`, then rename when Blender exits 0). **Step 4: Run** → PASS (no real Blender needed).
- [ ] **Step 5: Commit** — `git commit -m "feat: Blender depth source with content-hash cache"`

---

### Task 9: Encoding and music

**Files:**
- Create: `engine/encode.py`
- Test: `tests/test_encode.py`

**Interfaces:**
- Produces:
  - `ffmpeg_path() -> str` (`STEREO_FFMPEG` or the imageio-ffmpeg binary)
  - `VideoWriter(path: Path, width: int, height: int, fps: int = 30, crf: int = 16, preset: str = "slow")`: context manager; `.write(frame: np.ndarray)` (uint8 RGB); writes `path.with_suffix(".part.mp4")` and renames on a clean `__exit__`, deleting the part file on exception
  - `AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg"}`
  - `list_music(music_dir: Path) -> list[Path]` (sorted, extension filter, skips dotfiles)
  - `pick_music(music_dir: Path, key: str) -> Path | None` (index `int(sha256(key)[:8], 16) % n`)
  - `mux_music(video: Path, music: Path, out: Path, duration: float) -> None` (loops music if it is shorter)
  - `concat_music(tracks: list[Path], out: Path, duration: float) -> None` (for long videos)
  - `copy_silent(video: Path, out: Path) -> None`
  - `probe(path: Path) -> MediaInfo(duration: float, width: int, height: int, has_audio: bool)` (parses `ffmpeg -i` stderr)

- [ ] **Step 1: Write failing tests**

```python
def test_writer_produces_h264(tmp_path): 30 frames 64x64 -> probe: duration≈1.0, (64,64), no audio; no .part file left
def test_writer_cleans_up_on_error(tmp_path): exception inside with-block -> neither file exists
def test_music_filter_ignores_junk(tmp_path):              # Review Focus 2
    # music dir: .DS_Store, notes.txt, a.mp3, B.wav -> list_music == [B.wav, a.mp3] sorted case-insensitively
def test_pick_music_empty_dir_returns_none(tmp_path)
def test_pick_music_deterministic(tmp_path)
def test_mux_music_trims_and_has_audio(tmp_path): 1 s video + 3 s generated sine (ffmpeg lavfi) -> duration≈1.0, has_audio
def test_mux_loops_short_music(tmp_path): 3 s video + 1 s sine -> duration≈3.0
def test_copy_silent(tmp_path): has_audio False, same duration
def test_paths_with_spaces(tmp_path): all of the above inside "a dir with spaces/"   # Review Focus 1
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (rawvideo `rgb24` on stdin → libx264 with the constraint flags; audio filter `aloop=-1:size=2e9` when looping, `atrim=0:{d}`, `afade=t=in:d=1`, `afade=t=out:st={d-1}:d=1`, loudnorm; `-c:v copy -shortest`). **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: ffmpeg video writer, music selection and muxing"`

---

### Task 10: Compositor

**Files:**
- Create: `engine/compose.py`
- Test: `tests/test_compose.py`

**Interfaces:**
- Consumes: `Scene`, `TextConfig` (Task 5), `Format` (Task 1).
- Produces:
  - Timing constants: `TEXT_SHOW = 4.0, FADE = 0.5, END_SHOW = 3.0, SHORT_REVEAL = 3.0, LONG_REVEAL = 4.0, LONG_FADE_OUT = 0.5`
  - `short_alphas(t: float, duration: float, *, has_end: bool, reveal: bool) -> ShortAlphas(main: float, end: float, reveal: float)` (pure)
  - `ShortComposer(scene: Scene, fmt: Format, font: Path, reveal_img: np.ndarray | None)`; `.frame(stereo: np.ndarray, t: float) -> np.ndarray`
  - `SegmentComposer(fmt: Format, font: Path, *, index: int, total: int, title: str, stereo_seconds: float, reveal_img: np.ndarray)`; `.frame(stereo: np.ndarray, t: float) -> np.ndarray` (for `t ≥ stereo_seconds` the caller passes the last stereogram frame, which the reveal cross-fades from); `.total_seconds` = stereo_seconds + LONG_REVEAL + LONG_FADE_OUT
  - `card(lines: list[str], fmt: Format, font: Path) -> np.ndarray` (dark background #0C0C0E, centred lines; first line large, others small)
  - `thumbnail(stereo: np.ndarray, title: str, font: Path) -> np.ndarray` (1280×720; centre-crops a 9:16 source to 16:9)
  - `fit_reveal(img: np.ndarray, fmt: Format) -> np.ndarray` (letterbox onto neutral grey)

- [ ] **Step 1: Write failing tests**

```python
def test_main_text_timeline():
    assert short_alphas(0, 30, has_end=True, reveal=False).main == 1
    assert short_alphas(4.25, 30, ...).main == pytest.approx(0.5)
    assert short_alphas(10, 30, ...).main == 0
def test_end_text_last_three_seconds(): end==0 at t=26.9, ==1 at t=27.6
def test_reveal_replaces_end_text(): reveal=True -> end==0 always, reveal==1 at t=28
def test_short_duration_overlap():   # duration 3: main and end never both > 0 at the same t
def test_text_stays_in_bands():
    # render overlay on a black frame: all changed pixels in rows [0.08H, 0.16H] or [0.72H, 0.80H]
def test_text_is_subtle(): max overlay luminance <= 0.85*255 + 1 (opacity cap)
def test_segment_counter_and_progress():
    # at t = stereo_seconds/2, progress bar fills about 50 % of the width (±2 %); counter pixels in the top-left 15 % box
def test_segment_reveal_after_stereo(): t = stereo_seconds+1 -> frame equals reveal composite (title visible)
def test_segment_fades_to_black(): t = total_seconds - 0.01 -> mean < 5
def test_thumbnail_size(): (720, 1280, 3) from both 1080x1920 and 1920x1080 inputs
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** Pre-render each text line once as an RGBA layer (Pillow, `font size = round(0.03·H / 0.72)`, shadow = blurred black copy at 50 % with offset 0.002·H). Per frame, alpha-blend in numpy. Text bands: top line centred at 12 % of H, bottom and end at 76 %. Progress bar: 4 px tall (long), white at 35 %, at y = H − 0.04·H, left to right. Counter "3 / 15": 2.2 % of H text at 60 % opacity, top-left inside the safe area. Reveal cross-fade over FADE; long reveal shows `title` in the bottom band. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: compositor for Shorts, long segments, cards and thumbnails"`

---

### Task 11: Quality checks

**Files:**
- Create: `engine/checks.py`
- Test: `tests/test_checks.py`

**Interfaces:**
- Consumes: `DepthSequence` (Task 6), `Format` (Task 1), `seam_score/detail_score/SEAM_MAX/DETAIL_MAX` (Task 4), `self_check` (Task 3).
- Produces:
  - `QualityWarning(code: str, message: str)`; codes `OUT_OF_SAFE_AREA, MOTION_TOO_FAST, PATTERN_TOO_FINE, PATTERN_SEAMS, SELF_CHECK_FAILED`
  - `check_safe_area(seq, fmt, step: int = 5) -> list[QualityWarning]` (object = depth > 0.02; the message names the first offending second)
  - `check_motion(seq, threshold: float = 0.12) -> list[QualityWarning]` (metric: mean |d_t − d_{t−1}| over pixels where either frame > 0.02; the message suggests a lower `speed:`)
  - `check_pattern(strip) -> list[QualityWarning]`
  - `check_frames(depth_frames, stereo_frames, fmt, depth_strength) -> list[QualityWarning]` (runs `self_check` on the given pairs)

- [ ] **Step 1: Tests:** a centred static disc gives no warnings; a disc touching the right 10 % in SHORT gives OUT_OF_SAFE_AREA with "at 0.0 s"; a disc jumping 200 px per frame gives MOTION_TOO_FAST; a slowly moving disc (2 px per frame) gives none; uniform noise strip gives PATTERN_TOO_FINE; a stereogram/depth mismatch gives SELF_CHECK_FAILED.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: quality warnings for safe area, motion, patterns and self-check"`

---

### Task 12: Render pipeline, batch, preview, CLI (Milestones 4–5)

**Files:**
- Create: `engine/pipeline.py`
- Modify: `engine/cli.py` (add `preview`, `render`; global `--project PATH`, default cwd)
- Test: `tests/test_pipeline.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above. Tests inject a `FakeDepthSource` that returns an `ArrayDepthSequence` (moving disc) so they run without Blender.
- Produces:
  - `RenderResult(scene_id: str, outputs: list[Path], warnings: list[QualityWarning], skipped: bool)`
  - `render_scene(scene: Scene, project: Project, depth_source: DepthSource, *, preview: bool = False, force: bool = False, log: Callable[[str], None] = print) -> RenderResult`
  - `BatchSummary(done: list[str], skipped: list[str], failed: list[tuple[str, str]])`; `render_batch(paths: list[Path], project, depth_source, *, force=False, log=print) -> BatchSummary`
  - Output files per spec section 10 plus internal `reveal_long.png` and `render.json` (`{"render_hash": ..., "outputs": [...]}`), written last.
  - Preview: `output/<id>/preview.mp4`, using `preview_of(SHORT)` and duration `min(5, scene.duration)`; no music; skip logic not applied.

- [ ] **Step 1: Write failing tests**

```python
def test_render_short_and_long(tmp_project):
    r = render_scene(scene_2s, proj, FakeDepthSource())
    names = {p.name for p in r.outputs}
    assert {"short.mp4", "short_silent.mp4", "long_segment.mp4", "reveal.png", "thumbnail.png", "credits.txt"} <= names
    assert probe(out/"short.mp4").width == 1080 and probe(out/"short_silent.mp4").has_audio is False
def test_music_used_when_present(tmp_project): sine in assets/music -> short.mp4 has_audio; credits.txt names the track
def test_no_music_warns_but_succeeds(tmp_project): short.mp4 exists, log contains "no music"
def test_skip_when_unchanged(tmp_project): second call -> skipped True, FakeDepthSource call count unchanged
def test_rerender_when_text_changes(tmp_project)
def test_interrupted_render_is_redone(tmp_project):        # Review Focus 4
    # short.mp4 present but no render.json (simulated crash) -> not skipped
def test_batch_continues_after_failure(tmp_project):
    # 3 scene files, one with a missing model -> summary.failed == [("bad.yaml", "<SceneError message>")], 2 done
def test_preview_small_and_short(tmp_project): preview.mp4 is 540x960, duration ≈ min(5, scene.duration)
def test_project_path_with_spaces(tmp_path): project "My Videos/stereo" renders     # Review Focus 1
def test_cli_render_exit_codes(tmp_project): CliRunner: all ok -> 0, any failure -> 1 and summary printed
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `render_scene`: load the pattern strip per format (generated or image, `repeat_px` from the format or `advanced.pattern_repeat`); `check_pattern`. Per format: `seq = depth_source.render(scene, fmt)`; run `check_safe_area` and `check_motion`, and log the warnings. Stream frames: `render_stereogram` → composer (short) or plain (long segment, CRF 12) → `VideoWriter`. Run `check_frames` on the first, middle and last frame. Music via `pick_music(project.music, scene.id)` unless `music` is `"none"` or an explicit path. Write `reveal.png` (short framing), `reveal_long.png`, `thumbnail.png` (from the long middle frame if available, else short), `credits.txt` (model with its `license.txt` contents if present, pattern, music), then `render.json`. CLI picks `BlenderDepthSource(find_blender(), project.cache)` and prints a clear "Blender not found, run `stereo check`" if missing.
- [ ] **Step 4: Run** `pytest -v` → all PASS (Blender tests skipped in cloud).
- [ ] **Step 5: Commit** — `git commit -m "feat: render pipeline, batch rendering, preview and CLI"`
- [ ] **Step 6: Human gate (Milestones 3–5).** The owner runs `stereo render scenes/` on the Mac, then watches `output/example-heart/short.mp4` and `output/example-text/short.mp4` on their iPhone and confirms the 3D effect, text subtlety and speed. Record the render time per Short in the README.

---

### Task 13: Long-video compilation (Milestone 6)

**Files:**
- Create: `engine/compile.py`
- Modify: `engine/cli.py` (add `compile`)
- Test: `tests/test_compile.py`

**Interfaces:**
- Consumes: `Compilation`, `load_compilation` (Task 5), `SegmentComposer`, `card` (Task 10), `VideoWriter`, `list_music`, `concat_music`, `mux_music` (Task 9), outputs of Task 12.
- Produces:
  - `INTRO_SECONDS = 5.0, OUTRO_SECONDS = 4.0`; intro lines: `[title, "Relax your eyes and look through the screen"]`; outro lines: `["Thanks for watching"]`
  - `class MissingSegments(Exception)`: attr `scene_ids: list[str]`
  - `compile_long_video(comp: Compilation, project: Project, log=print) -> Path` → `output/compilations/<id>/video.mp4` and `chapters.txt`
  - `chapter_lines(starts: list[tuple[float, str]]) -> list[str]` (`MM:SS` format, or `H:MM:SS` from 1 h; the first entry is `00:00 Intro`; the rest are `Stereogram N`)

- [ ] **Step 1: Write failing tests**

```python
def test_chapter_format():
    assert chapter_lines([(0, "Intro"), (5, "Stereogram 1"), (3725, "Stereogram 2")]) == ["0:00:00 Intro", "0:00:05 Stereogram 1", "1:02:05 Stereogram 2"]
    assert chapter_lines([(0, "Intro"), (65.4, "Stereogram 1")]) == ["00:00 Intro", "01:05 Stereogram 1"]
def test_missing_segments_named(tmp_project):
    with pytest.raises(MissingSegments) as e: compile_long_video(comp_with(["a", "ghost"]), proj)
    assert e.value.scene_ids == ["ghost"]
def test_compile_two_scenes(tmp_project):   # scenes pre-rendered via FakeDepthSource, 2 s each, segment_seconds=1
    v = compile_long_video(comp, proj); info = probe(v)
    assert (info.width, info.height) == (1920, 1080)
    assert info.duration == pytest.approx(5 + 2 * (1 + 4 + 0.5) + 4, abs=0.2)
    assert open(chapters).read().splitlines() == ["00:00 Intro", "00:05 Stereogram 1", "00:10 Stereogram 2", "00:16 Outro"]
def test_chapters_never_contain_titles(tmp_project): scene titles absent from chapters.txt
def test_segment_shorter_than_segment_seconds_warns(tmp_project)
def test_music_spans_video(tmp_project): two 3 s sines -> has_audio and duration matches video
```

  (In the chapter test, 10.5 s and 15.9 s are floored to whole seconds.)

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement:** decode each `long_segment.mp4` with `imageio_ffmpeg.read_frames` (first `segment_seconds·fps` frames), compose with `SegmentComposer(index=i+1, total=n, title=scene.title, reveal_img=reveal_long.png)`, and write everything into one `VideoWriter` (CRF 16). Track chapter start times by frame count. Music: tracks from `list_music` in a deterministic order seeded by `comp.id`, enough to cover the duration; `concat_music` then `mux_music`. CLI prints the paths and the chapters text, ready to paste into a YouTube description.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: long video compilation with reveals and chapters"`

---

### Task 14: End-to-end test and README

**Files:**
- Create: `tests/test_e2e.py`
- Modify: `README.md`

- [ ] **Step 1: Test** `test_e2e_with_fake_depth`: copy the example scenes into a temp project, `CliRunner` `render scenes/` then `compile compilations/example.yaml` with a fake depth source (env `STEREO_DEPTH_SOURCE=fake`, honoured only by the CLI's source factory, used for tests). Assert that every output exists, the durations match and the self-check produces no warnings. Add `test_e2e_blender` (skipped without Blender): the same flow with real Blender at preview size.
- [ ] **Step 2: Run** `pytest -v` → PASS.
- [ ] **Step 3: README:** quick start (three commands); scene file reference (every field, default and allowed values, copied from Task 5); the motion presets with tips (big shapes, moderate speed, toward-camera motion reads best); finding free models (CC0 sources, put `license.txt` next to the model); music folder and silent exports; what each warning means and how to fix it; uploading tips (Shorts use `short.mp4` or `short_silent.mp4`; YouTube long videos paste `chapters.txt`).
- [ ] **Step 4: Commit** — `git commit -m "test: end-to-end pipeline test; docs: README usage guide"`
