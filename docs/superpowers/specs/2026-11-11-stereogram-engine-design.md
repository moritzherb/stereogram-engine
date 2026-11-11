# Stereogram Engine: Design Spec

Date: 2026-11-11
Status: Draft, awaiting owner review

## 1. Goal

A local engine that turns animated 3D scenes into **animated stereogram videos**
("Magic Eye" style: a patterned surface in which a moving 3D object becomes
visible when the viewer looks "through" the screen). The videos are uploaded
manually to TikTok, Instagram Reels, Facebook, YouTube Shorts (vertical) and
YouTube (horizontal long videos).

Reference style: two example TikTok clips the owner shared. Both are 9:16,
about 30 s at 30 fps with music. They use a tiled texture pattern (globes, or an
abstract multicolour texture) and a hidden object that moves while the pattern
stays stable. Text sits at the top ("This is a Stereogram") and bottom ("What do
you see?").

**Success:** the owner writes a short scene file, runs one command on their
MacBook Pro 2016, and gets an upload-ready Short (with and without music) and a
16:9 segment. Many segments can be compiled into a long YouTube video with
reveals and chapters. The 3D effect is easy to see on an iPhone screen and
survives platform recompression.

## 2. Decisions

### Stated by the owner
- Animated stereograms like the reference clips.
- On-screen text is **subtle** and never blocks the picture. **No emojis** in
  any generated text.
- Both Shorts (vertical) and long videos (horizontal); long videos are
  compilations of the same scenes.
- Hidden objects come from 3D models and code-built shapes and text now; AI depth
  from photos comes later (option "D").
- Command-line batch engine first, web UI later (option "C").
- Patterns: generated in code **and** from the owner's own images (option "C").
- Music: a folder of owner-supplied royalty-free tracks, **plus** an optional
  silent export for adding trending sounds in-app (option "3").
- Reveal: per-video setting; default off for Shorts, always on in long
  videos (option "C").
- 3D/animation: **Blender + Python** (owner wants real character animations).
- Rendering machine: **MacBook Pro 2016** (Intel CPU, max macOS 12 Monterey).
  The owner directs the work from an iPhone.

### Assumptions (owner may correct)
- Default viewing method is parallel ("wall-eyed", look through the screen), the
  standard Magic Eye method. Cross-eyed mode is a setting, not a default.
- 30 fps, H.264 MP4, 1080×1920 for Shorts, 1920×1080 for long videos.
- Uploading stays manual in v1.
- The owner is responsible for music licences; model licences are recorded per
  model (see section 9).

## 3. Scope

**In v1:** scene files, Blender depth rendering (models with built-in
animations, motion presets, 3D text and primitives), stereogram generation,
patterns (generated and from images), text overlays, music and silent exports,
reveal images, thumbnails, previews, batch rendering, long-video compilation
with chapters, quality warnings, tests.

**Out of v1 (later, separate projects):** web UI, AI depth from photos or
generated images, automatic uploading to platforms, title and hashtag
generation.

## 4. Architecture

```
scene.yaml ─► [Scene loader] ─► [Depth source: Blender] ─► depth frames (cache)
                                                              │
pattern (image or generator) ─────────────────────────────────┤
                                                              ▼
                                                    [Stereogram core]
                                                              │ RGB frames
                                                              ▼
                              [Compositor: text, fades, reveal, progress bar]
                                                              │
                                                              ▼
                              [Encoder: ffmpeg, music, silent variant]
                                                              │
                                                              ▼
                       output/<scene-id>/  ─► [Compiler] ─► long video + chapters
```

Python drives every stage. Blender runs as a **separate background process**
(`blender -b --python engine/blender/render_depth.py -- <job.json>`), so the
engine's own Python environment never depends on Blender's bundled Python.

**Swappable depth source.** Every stage after the depth source only sees a
`DepthSequence`: a frame count, fps, resolution, and float32 depth frames in
`[0, 1]` (1 = nearest, 0 = background), plus an optional reveal image. Blender
is the v1 implementation. A pure-Python rigid-motion source (fallback) and an
AI-depth source (later) plug into the same interface.

## 5. Components

Each unit has one job and is testable on its own.

| Module | Responsibility | Depends on |
|---|---|---|
| `engine/scene.py` | Load and validate scene and compilation YAML, apply defaults, produce typed config objects; clear error messages | pydantic, PyYAML |
| `engine/depth/base.py` | `DepthSource` interface and `DepthSequence` type | numpy |
| `engine/depth/blender_source.py` | Write job JSON, run Blender, read frames back, manage cache | subprocess |
| `engine/blender/render_depth.py` | Runs **inside** Blender: import model, pick animation clip or motion preset, build 3D text and primitives, auto-frame the camera, render depth and reveal image | Blender `bpy` only |
| `engine/patterns.py` | Load an image, make it seamlessly tileable, scale to strip width; procedural generators (noise, swirl, cells, stripes) with seeded palettes | numpy, Pillow |
| `engine/stereo.py` | Stereogram core (Numba): depth frame + pattern → RGB frame | numpy, numba |
| `engine/decode.py` | Self-check: recover depth from a stereogram and compare | numpy, numba |
| `engine/compose.py` | Text overlays, fades, progress bar, scene counter, reveal cross-fade, intro and outro cards, thumbnail | Pillow, numpy |
| `engine/encode.py` | Pipe frames into ffmpeg; music selection, trim, fade, loudness; silent variant | ffmpeg |
| `engine/compile.py` | Concatenate rendered 16:9 segments with intro and outro; write chapters file | ffmpeg |
| `engine/checks.py` | Pre-flight and post-render warnings (section 8) | numpy |
| `engine/cli.py` | `stereo` command: `check`, `preview`, `render`, `compile` | typer |

### 5.1 Blender stage
- **Version:** Blender **4.5 LTS** (the last release with official Intel Mac
  builds; Blender 5 dropped Intel Macs). Reports conflict on whether 4.5 runs
  on macOS 12, so **Milestone 1 verifies this on the owner's Mac**. Fallback
  order: 4.5 LTS → 4.2 LTS (macOS 11.2+) → pure-Python rigid-motion source.
- **Inputs:** `.glb`/`.gltf` (preferred) and `.fbx` models; `text3d: "HELLO"`;
  primitives (`sphere`, `torus`, `heart`, …).
- **Animation:** `animation: <clip name>` plays a built-in clip (looped to the
  requested duration, speed adjustable with `speed:`). Motion presets work on any
  object: `spin`, `float`, `bob`, `zoom_in`, `fly_toward`, `swim_past`,
  `ripple`, `pulse`. A clip and a preset can combine, e.g. a swimming dolphin
  that also drifts across the frame.
- **Camera:** orthographic by default (stable depth, no perspective
  distortion). The camera is auto-framed so the object's bounding box over the
  **whole** animation fits inside the format's safe area.
- **Depth output:** rendered at output resolution, normalised per scene
  (not per frame, so depth doesn't "breathe"), saved as 16-bit grayscale PNG
  frames. Milestone 1 picks the render engine that works headless on the Mac
  (Cycles CPU with a Mist/Z pass at 1 sample, or Workbench).
- **Reveal image:** a simple shaded render of the object at a key frame on a
  neutral background.

### 5.2 Stereogram core
- Algorithm: symmetric constraint linking with hidden-surface removal
  (Thimbleby, Inglis & Witten, 1994), per row. Separation
  `s(z) = round((1 − μz)·E / (2 − μz))`, with depth factor μ ≈ 0.33 by default.
  The colour comes from the pattern image, so it's a textured stereogram, not
  random dots.
- **Oversampling:** computed at 3× horizontal resolution, then downscaled with
  an area filter, which gives smooth depth steps and no jaggies or ghosting.
- **Temporal stability:** linking starts from the image centre column, and the
  pattern is anchored identically every frame. Regions whose depth doesn't
  change produce identical pixels frame to frame, so there's no flicker.
- **Default pattern repeat width:** vertical about width/7 (about 154 px at 1080
  wide), horizontal about width/11 (about 175 px at 1920 wide). Both get final
  tuning by viewing tests on the owner's iPhone in Milestone 2.
- **Performance target:** ≤ 50 ms per 1080×1920 frame on the 2016 MacBook Pro.

### 5.3 Patterns
- Image patterns: crop, then make seamless (mirror and cross-blend seams),
  scale to the strip width; vertical tiling uses a per-row offset to hide
  repetition.
- Generated patterns: seeded, so the same scene always gives the same look.
  Styles: `noise`, `swirl`, `cells`, `stripes`, `dots`, with named palettes.
- Detail check: warn if the pattern's high-frequency energy is so high it will
  smear after platform compression.

### 5.4 Encoding
- ffmpeg from the `imageio-ffmpeg` package (bundled binary, no Homebrew needed
  on macOS 12), with a configurable path to a system ffmpeg.
- Video: H.264 High, `yuv420p`, 30 fps, CRF 16 / preset `slow`,
  `+faststart`. A high bitrate is deliberate because platforms recompress.
- Audio: a track is picked deterministically per scene from `assets/music/`,
  trimmed to length, 1 s fade in and out, loudness normalised to −14 LUFS,
  AAC 192 kbps. `short_silent.mp4` re-uses the video stream with no audio.

## 6. Scene file

```yaml
id: dolphin-jump                # required, used for output folder
model: models/dolphin.glb       # or: text3d: "HELLO"  or: shape: heart
animation: Swim                 # built-in clip name, optional
motion: swim_past               # preset, optional (combines with animation)
speed: 1.0
duration: 30                    # seconds
pattern: patterns/ocean.jpg     # or: { generate: swirl, palette: ocean, seed: 7 }
formats: [short, long]          # short = 1080x1920, long = 1920x1080 segment
text:
  top: "This is a stereogram"
  bottom: "What do you see?"
  end: "Comment what you see"   # shown in the final seconds of a Short
music: auto                     # auto | path | none
reveal: auto                    # auto (Short: off, long: on) | on | off
title: "Hidden animal"          # used on thumbnail and in long-video chapters
advanced:                       # all optional
  depth_strength: 0.33
  pattern_repeat: auto
  viewing: parallel             # parallel | crosseyed
```

Compilation file:

```yaml
id: ocean-animals
title: "Can you see the hidden ocean animals?"
scenes: [dolphin-jump, shark-attack, whale-glide]
segment_seconds: 35             # stereogram time per scene before the reveal
```

Validation errors name the file, the field and the fix, in plain English.

## 7. Video layouts

All on-screen text: one bundled clean sans-serif font (OFL-licensed), plain
white at about 85 % opacity with a soft shadow, **no emojis** (any emoji in a
text field is rejected at validation), and short single lines.

### Short (1080×1920)
- The stereogram fills the frame. The hidden object is framed inside a safe area
  clear of the platform UI: top 10 %, bottom 20 % and right 15 % stay free.
- Top line at about 12 % of the height, bottom line at about 76 %. Text height is
  about 3 % of the frame height.
- Timeline: text visible for 0–4 s, then fades out over 0.5 s. `end` text fades
  in for the last 3 s. If reveal is on, it replaces the `end` text: the last 3 s
  cross-fade to the reveal image instead.
- Text may briefly overlap the object's area; this is acceptable because it is
  only visible during the first 4 s and the last 3 s.

### Long-video segment (1920×1080)
- Small scene counter "3 / 15" in the top-left corner and a thin progress bar
  along the bottom (the countdown), both low-contrast.
- Stereogram for `segment_seconds`, then a 4 s reveal (cross-fade to the reveal
  image with the scene title), then a 0.5 s fade to the next scene.

### Long video
- Intro card (about 5 s): title plus one-line viewing hint ("Relax your eyes and
  look through the screen").
- The segments in order, then a plain outro card.
- `chapters.txt`: `00:00 Intro`, `00:05 Stereogram 1`, …. Chapter names never
  reveal the answer.

### Thumbnail
- A 1280×720 still frame of the stereogram with the title, for YouTube.

## 8. Quality checks and error handling

**Pre-flight (before rendering, also run by `preview`):**
- The object leaves the safe area at any point in the animation.
- Motion too fast: the depth change between frames exceeds a threshold, so
  viewers would lose the lock.
- Pattern too fine for compression, or visible tile seams.
- Missing files, unknown animation clip names (the warning lists the available
  clips), emojis in text.

**Batch behaviour:**
- Finished scenes are skipped, judged by output files plus a hash of the scene
  settings, so batches can be restarted.
- A failing scene is logged and the batch continues. At the end, a summary lists
  every failure with a plain-English reason.
- Depth frames are cached under `cache/` by a hash of the depth-relevant
  settings, so changing only the pattern, text or music skips Blender entirely.
- `compile` refuses to run if a listed scene has no rendered segment and names
  the missing scenes.
- Blender errors (crash, unsupported model) are caught and the relevant part of
  the Blender log is shown.

## 9. Assets and licensing

- `assets/models/<name>/` holds the model plus a `license.txt`.
  Preferred sources: CC0 or commercial-use-allowed packs (e.g. Quaternius).
- `stereo render` writes `credits.txt` next to each output, listing the model,
  pattern and music used, for video descriptions when attribution is required.

## 10. Folder structure

```
stereogram-engine/
  assets/{models,patterns,music,fonts}/
  scenes/            one .yaml per video
  compilations/      long-video lists
  output/<scene-id>/ short.mp4, short_silent.mp4, long_segment.mp4,
                     reveal.png, thumbnail.png, credits.txt
  output/compilations/<id>/ video.mp4, chapters.txt
  cache/             depth frames (safe to delete)
  engine/            code
  tests/
```

## 11. Commands

- `stereo check`: verifies Python packages, Blender (version and headless depth
  render of a test cube), ffmpeg and fonts; prints fixes for anything missing.
- `stereo preview <scene>`: 5 s, half resolution, short only, fast.
- `stereo render <scene.yaml | folder>`: full render of one scene or a batch.
- `stereo compile <compilation.yaml>`: builds the long video and chapters.

## 12. Tech stack

Python 3.11 (python.org installer; supports macOS 12 Intel), numpy, numba,
Pillow, PyYAML, pydantic, typer, imageio-ffmpeg, pytest. Blender 4.5 LTS
installed as a normal Mac app; the engine finds it at
`/Applications/Blender.app` or via a configured path. Code stays
cross-platform so development and tests also run on Linux.

## 13. Testing

- **Unit tests (stereogram maths):** a flat depth map gives a pure repetition at
  exactly the far-plane width; a step shape gives exactly the expected
  separations; identical depth frames give identical output frames
  (no flicker).
- **Self-check (decoder):** recover depth from generated stereograms and require
  ≥ 95 % of interior pixels within tolerance of the source depth. This also runs
  after every real render as a quality gate (warning, not failure).
- **Pattern, compose and encode tests:** tileability, text placement inside safe
  areas, emoji rejection, output duration, resolution, audio or no audio.
- **Blender tests** (skipped when Blender isn't installed): render a cube with a
  spin preset; check frame count, depth range and the cache hit on the second run.
- **End-to-end:** a 2 s scene through every stage, plus a 2-scene compilation
  with chapters.
- **Human check** at each milestone: the owner watches the test video on their
  iPhone and confirms the 3D effect.

## 14. Milestones

1. **Mac check:** Blender 4.5 (or the fallback) renders depth frames headless on
   the owner's MacBook Pro 2016; measure speed; choose the render engine.
   Includes `stereo check`.
2. **Still stereograms:** depth image + pattern → stereogram PNG; decoder
   self-check; owner tunes repeat width and depth strength on iPhone.
3. **Animated:** Blender animation → depth frames → stereogram video (no text or
   music).
4. **Finishing:** text, fades, music, silent variant, reveal, thumbnail,
   credits.
5. **Batch and preview:** folder rendering, caching, skip and resume, warnings,
   summary.
6. **Long videos:** 16:9 segments, intro and outro, compilation, chapters.

## 15. Risks

| Risk | Mitigation |
|---|---|
| Blender 4.5 doesn't run on macOS 12 | Milestone 1 tests first; fall back to 4.2 LTS, then to the pure-Python rigid-motion depth source |
| Rendering too slow on the 2016 Mac | Depth-only renders at minimal samples; depth caching; overnight batches |
| Platform compression smears the pattern | High-bitrate export, pattern detail check, coarser default patterns |
| Fast or detailed animations don't read in 3D | Motion-speed warning, preset speeds tuned for stereograms, guidance in README |
| Copyright claims on music or models | Owner-supplied licensed music, licence file per model, `credits.txt` |
