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
   pip install -e .
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

## Tested on

- Linux (development, without Blender)
