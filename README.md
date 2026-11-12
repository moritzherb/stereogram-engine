# Stereogram Engine

Turns 3D scenes into animated stereogram videos ("Magic Eye" style) for
TikTok, Reels, YouTube Shorts and long YouTube videos.

## Setup on your Mac

You only do this once. Open the **Terminal** app (Applications > Utilities)
and follow the steps.

1. **Python 3.11:** download the macOS installer from
   <https://www.python.org/downloads/release/python-3119/> and run it.
2. **Blender 4.5 LTS:** download it from <https://www.blender.org/download/lts/4-5/>
   (choose the **Intel** version), and drag Blender into Applications.
   If it won't open on your macOS, use **Blender 4.2 LTS** instead:
   <https://www.blender.org/download/lts/4-2/>.
3. **This engine:**

   ```bash
   cd ~
   git clone https://github.com/moritzherb/stereogram-engine.git
   cd stereogram-engine
   git checkout claude/stereogram-engine-plan-9la0b4
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --prefer-binary -e .
   ```

4. **Check everything works:**

   ```bash
   stereo check
   ```

   Every line should say `OK`. The Blender line also tells you how fast your
   Mac renders. Copy the whole output and send it to your developer.

Every time you open a new Terminal window later, first run:

```bash
cd ~/stereogram-engine
source .venv/bin/activate
```

## Quick start

```bash
stereo preview scenes/example-heart.yaml   # 5-second check, a few seconds to render
stereo render scenes/                      # every scene in the folder (Shorts + 16:9 segments)
stereo compile compilations/example.yaml   # long YouTube video from rendered scenes
```

Finished files land in `output/`:

| File | Use it for |
| --- | --- |
| `output/<scene>/short.mp4` | TikTok, Reels, YouTube Shorts, Facebook (with music) |
| `output/<scene>/short_silent.mp4` | Same, without music, so you can add a trending sound in the app |
| `output/<scene>/reveal.png` | The answer, for a comment or a follow-up post |
| `output/<scene>/thumbnail.png` | YouTube thumbnail |
| `output/<scene>/credits.txt` | What was used (model, pattern, music), for the video description |
| `output/compilations/<id>/video.mp4` | The long YouTube video |
| `output/compilations/<id>/chapters.txt` | Paste into the YouTube description to get chapters |

Rendering again skips scenes that haven't changed. Use `stereo render scenes/ --force` to redo everything.
A batch can be stopped (Ctrl+C or closing the laptop) and simply started again.

## Writing a scene

Create a file in `scenes/`, for example `scenes/dolphin-jump.yaml`.
Only `id` and one of `model`, `text3d` or `shape` are required:

```yaml
id: dolphin-jump                # lowercase letters, numbers and dashes
model: models/dolphin/dolphin.glb   # or: text3d: "HELLO"   or: shape: heart
animation: Swim                 # a clip inside the model (optional)
motion: sway                    # a movement preset (optional, combines with animation)
motion_settings:                # optional; leave out for the gentle defaults
  amount: 20                    # how far: degrees for turns, % of the size for movement
  seconds: 8                    # how slow: length of one full cycle
speed: 1.0                      # lower = slower
duration: 30                    # seconds (1 to 180)
pattern: patterns/ocean.jpg     # your image, or a generated pattern:
#   pattern: { generate: swirl, palette: ocean, seed: 7 }
formats: [short, long]          # short = 1080x1920, long = 1920x1080 segment
text:
  top: "This is a stereogram"   # shown for the first 4 seconds
  bottom: "What do you see?"
  end: "Comment what you see"   # shown for the last 3 seconds (null = off)
  bottom_y: 0.85                # optional: exact height of the bottom line (0.5 to 0.95)
platform: tiktok                # tiktok, reels, shorts, facebook or clean (where the bottom line sits)
music: auto                     # auto (picks from assets/music), none, or music/my-song.mp3
reveal: auto                    # auto = Shorts: no answer, long videos: answer shown
title: "Hidden dolphin"         # used for the thumbnail and long-video reveal
advanced:
  depth_strength: 0.33          # 0.1 to 0.6, how far the object pops out
  pattern_repeat: auto          # pattern width in pixels
  viewing: parallel             # parallel (look through) or crosseyed
  lift: 0.35                    # 0 to 0.8: how far the whole object stands out in front of the background
  edge_softness: 1.0            # 0 to 3: how soft the object's outline is
  pattern_drift: 0              # pixels per second the whole pattern slides sideways (try 10-20)
```

Text must be plain: emojis are rejected.

**Shapes:** `sphere`, `cube`, `torus`, `cone`, `heart`, `star`.

**Generated patterns:** styles `noise`, `swirl`, `cells`, `stripes`, `dots`;
palettes `default`, `ocean`, `sunset`, `forest`, `candy`, `mono`.

**Motions:**

| Motion | What it does | `amount` means (default) | `seconds` (default) |
| --- | --- | --- | --- |
| `sway` | Turns to one side and back, then to the other side and back | degrees (20) | one full cycle (8) |
| `tilt` | Nods forward and back | degrees (12) | 8 |
| `rock` | Tips left and right | degrees (10) | 8 |
| `spin` | Turns all the way around | degrees per cycle (360) | 12 |
| `float` | Drifts gently up and down | % of size (3) | 6 |
| `bob` | Quicker up-and-down bounce | % of size (5) | 2.5 |
| `zoom_in` | Grows to full size | start size % (70) | whole video |
| `fly_toward` | Comes towards the viewer | start distance % of size (75) | whole video |
| `swim_past` | Crosses the screen left to right | object width % of the screen (30) | whole video |
| `ripple` | Waves run across the surface | wave height % (3) | 2 |
| `pulse` | Beats like a heart | growth % (5) | one beat (1) |

**Keeping it hidden:** the turning motions (`sway`, `tilt`, `rock`) keep the object in place, so it
only appears when the viewer focuses. Motions that move the object around (`float`, `bob`,
`swim_past`) give it away more easily without focusing. Soft edges (`edge_softness`) and, optionally,
`pattern_drift` hide it even better. `lift` balances the other way: higher makes the object pop out
more clearly once seen, lower blends it into the background. Smooth, organic patterns (`swirl`, `noise`, busy photos) hide
the object much better than patterns with straight, high-contrast lines (`cells`, `stripes`).

Tips: big, bold shapes read best; fine details (fingers, thin legs) disappear.
Hidden 3D text uses solid, rounded letters; short words (up to about 8 letters) work best.

**Text position per platform:** the bottom line sits at 80 % of the height for `tiktok`, `reels`,
`shorts` and `facebook` (just above the apps' caption area), and at 88 % for `clean` (posted without
a caption). These are starting values; use `text.bottom_y` to fine-tune after your first uploads.

**Countdown:** when the answer is revealed (always in long videos, `reveal: on` in Shorts), a small
ring with the seconds left counts down until the reveal: bottom-right in long videos, top-right in
Shorts (the apps' buttons cover the bottom-right corner there).

## Long videos

Create a file in `compilations/`:

```yaml
id: ocean-animals
title: "Can you see the hidden ocean animals?"
scenes: [dolphin-jump, shark-attack, whale-glide]   # scene ids, in order
segment_seconds: 35                                 # stereogram time before each reveal
```

Render the scenes first (`stereo render scenes/`), then `stereo compile compilations/ocean-animals.yaml`.

## Models, patterns and music

- **Models:** put each model in its own folder under `assets/models/` (`.glb` preferred, `.fbx` works),
  with a `license.txt` next to it. Use models that allow commercial use, e.g. CC0 packs from
  Quaternius or Kenney, so monetised videos are safe. Animated models (with clips like "Swim" or "Run")
  move on their own; the engine tells you the clip names if you mistype one.
- **Patterns:** any image in `assets/patterns/` works; the engine makes it seamless. Busy, colourful
  images with medium-sized detail work best.
- **Music:** put royalty-free tracks (`.mp3`, `.m4a`, `.wav`, ...) in `assets/music/`. Each scene picks
  one automatically; it loops and fades to fit.

## Warnings and what to do

| Warning | Meaning | Fix |
| --- | --- | --- |
| leaves the safe area | The object goes under TikTok's buttons or caption | Different motion or lower speed |
| motion is very fast | Viewers may lose the 3D effect | Lower `speed:` |
| very fine detail | The pattern will smear after upload compression | Smoother pattern image |
| visible seam | The pattern image has hard edges at its sides | Different image |
| could only read back ...% | The engine's own self-check struggled | Pattern with more texture |

## Tested on

- Linux, Blender 4.5.16 LTS (cloud development machine): about 0.4 s per depth frame,
  0.1 s per stereogram frame
