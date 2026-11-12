"""Scene and compilation files: loading, defaults and plain-English validation."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from engine.formats import Format
from engine.patterns import PALETTES, STYLES
from engine.project import Project

MOTIONS = ("sway", "tilt", "rock", "spin", "float", "bob", "zoom_in", "fly_toward", "swim_past", "ripple", "pulse")
SHAPES = ("sphere", "cube", "torus", "cone", "heart", "star")
PLATFORMS = ("tiktok", "reels", "shorts", "facebook", "clean")
MODEL_EXTS = (".glb", ".gltf", ".fbx")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")


class SceneError(Exception):
    """A problem in a scene or compilation file, explained in plain English."""


def contains_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text))


def _no_emoji(value: str | None) -> str | None:
    if value is not None and contains_emoji(value):
        raise ValueError("contains an emoji. Remove it: on-screen text must be plain text")
    return value


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "my-scene"


def _check_id(value: str) -> str:
    if not _ID_RE.match(value):
        raise ValueError(f"'{value}' is not allowed. Use lowercase letters, numbers and dashes, "
                         f"e.g. '{_slug(value)}'")
    return value


def _resolve(value: Any, info: ValidationInfo, folder_hint: str) -> Path:
    assets: Path = info.context["assets"]
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = assets / p
    if not p.is_file():
        raise ValueError(f"file not found: {p}. Put the file in {assets / folder_hint} or fix the path")
    return p


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextConfig(_Strict):
    top: str | None = "This is a stereogram"
    bottom: str | None = "What do you see?"
    end: str | None = "Comment what you see"
    bottom_y: float | None = Field(None, ge=0.5, le=0.95)  # overrides the platform's position

    _plain = field_validator("top", "bottom", "end")(classmethod(lambda cls, v: _no_emoji(v)))


class GeneratedPattern(_Strict):
    generate: str
    palette: str = "default"
    seed: int | None = None

    @field_validator("generate")
    @classmethod
    def _style(cls, v: str) -> str:
        if v not in STYLES:
            raise ValueError(f"unknown style '{v}'. Choose one of: {', '.join(STYLES)}")
        return v

    @field_validator("palette")
    @classmethod
    def _palette(cls, v: str) -> str:
        if v not in PALETTES:
            raise ValueError(f"unknown palette '{v}'. Choose one of: {', '.join(PALETTES)}")
        return v


class MotionSettings(_Strict):
    """How strong and how slow a motion is. Empty = the motion's gentle default."""

    amount: float | None = Field(None, gt=0, le=360)  # degrees for turns, % of size for movement
    seconds: float | None = Field(None, gt=0, le=120)  # length of one full cycle


class Advanced(_Strict):
    depth_strength: float = Field(0.33, ge=0.1, le=0.6)
    edge_softness: float = Field(1.0, ge=0.0, le=3.0)
    pattern_drift: float = Field(0.0, ge=0.0, le=200.0)  # pixels per second, 0 = off
    pattern_repeat: int | Literal["auto"] = "auto"
    viewing: Literal["parallel", "crosseyed"] = "parallel"


class Scene(_Strict):
    id: str
    model: Path | None = None
    text3d: str | None = None
    shape: str | None = None
    animation: str | None = None
    motion: str | None = None
    motion_settings: MotionSettings = Field(default_factory=MotionSettings)
    speed: float = Field(1.0, gt=0, le=10)
    duration: float = Field(30, ge=1, le=180)
    pattern: Path | GeneratedPattern | None = None
    formats: list[Literal["short", "long"]] = Field(default_factory=lambda: ["short", "long"], min_length=1)
    text: TextConfig = Field(default_factory=TextConfig)
    music: Path | Literal["auto", "none"] = "auto"
    reveal: Literal["auto", "on", "off"] = "auto"
    title: str | None = None
    platform: Literal["tiktok", "reels", "shorts", "facebook", "clean"] = "tiktok"
    advanced: Advanced = Field(default_factory=Advanced)

    _id = field_validator("id")(classmethod(lambda cls, v: _check_id(v)))
    _plain = field_validator("title", "text3d")(classmethod(lambda cls, v: _no_emoji(v)))

    @model_validator(mode="before")
    @classmethod
    def _yaml_booleans(cls, data: Any) -> Any:
        """YAML reads unquoted on/off/yes/no as true/false; map them back."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if isinstance(data.get("reveal"), bool):
            data["reveal"] = "on" if data["reveal"] else "off"
        if isinstance(data.get("music"), bool):
            data["music"] = "auto" if data["music"] else "none"
        if isinstance(data.get("text"), dict):
            data["text"] = {k: (None if v is False else v) for k, v in data["text"].items()}
        return data

    @field_validator("model", mode="before")
    @classmethod
    def _model(cls, v: Any, info: ValidationInfo) -> Path | None:
        if v is None:
            return None
        p = _resolve(v, info, "models")
        if p.suffix.lower() not in MODEL_EXTS:
            raise ValueError(f"'{p.name}' is not a supported model. Use {', '.join(MODEL_EXTS)}")
        return p

    @field_validator("shape")
    @classmethod
    def _shape(cls, v: str | None) -> str | None:
        if v is not None and v not in SHAPES:
            raise ValueError(f"unknown shape '{v}'. Choose one of: {', '.join(SHAPES)}")
        return v

    @field_validator("motion")
    @classmethod
    def _motion(cls, v: str | None) -> str | None:
        if v is not None and v not in MOTIONS:
            raise ValueError(f"unknown motion '{v}'. Choose one of: {', '.join(MOTIONS)}")
        return v

    @field_validator("pattern", mode="before")
    @classmethod
    def _pattern(cls, v: Any, info: ValidationInfo) -> Any:
        if v is None or isinstance(v, GeneratedPattern):
            return v
        if isinstance(v, dict):
            try:
                return GeneratedPattern.model_validate(v)
            except ValidationError as e:
                first = e.errors()[0]
                field = ".".join(str(p) for p in first["loc"])
                raise ValueError(f"{field} {first['msg'].removeprefix('Value error, ')}") from None
        return _resolve(v, info, "patterns")

    @field_validator("music", mode="before")
    @classmethod
    def _music(cls, v: Any, info: ValidationInfo) -> Any:
        if v in ("auto", "none"):
            return v
        return _resolve(v, info, "music")

    @model_validator(mode="after")
    def _finish(self) -> "Scene":
        sources = [s for s in (self.model, self.text3d, self.shape) if s is not None]
        if len(sources) != 1:
            raise ValueError("use exactly one of model, text3d or shape")
        if self.pattern is None:
            self.pattern = GeneratedPattern(generate="noise")
        if isinstance(self.pattern, GeneratedPattern) and self.pattern.seed is None:
            self.pattern.seed = int(hashlib.sha256(self.id.encode()).hexdigest()[:8], 16)
        if self.title is None:
            self.title = self.id.replace("-", " ").capitalize()
        return self


class Compilation(_Strict):
    id: str
    title: str
    scenes: list[str] = Field(min_length=1)
    segment_seconds: float = Field(35, ge=1, le=600)

    _id = field_validator("id")(classmethod(lambda cls, v: _check_id(v)))
    _plain = field_validator("title")(classmethod(lambda cls, v: _no_emoji(v)))

    @field_validator("scenes")
    @classmethod
    def _scene_ids(cls, v: list[str]) -> list[str]:
        for scene_id in v:
            _check_id(scene_id)
        return v


def _explain(path: Path, err: ValidationError) -> SceneError:
    first = err.errors()[0]
    loc = ".".join(str(p) for p in first["loc"]) or "file"
    msg = first["msg"].removeprefix("Value error, ")
    if first["type"] == "missing":
        msg = "is required"
    elif first["type"] == "extra_forbidden":
        msg = "is not a known setting. Check the spelling"
    return SceneError(f"{path.name}: {loc}: {msg}.")


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise SceneError(f"{path.name}: file: could not be read ({e}).") from None
    if not isinstance(data, dict):
        raise SceneError(f"{path.name}: file: expected settings like 'id: my-scene'.")
    return data


def load_scene(path: Path, project: Project) -> Scene:
    data = _read_yaml(path)
    try:
        return Scene.model_validate(data, context={"assets": project.assets})
    except ValidationError as e:
        raise _explain(path, e) from None


def load_compilation(path: Path, project: Project) -> Compilation:
    data = _read_yaml(path)
    try:
        return Compilation.model_validate(data, context={"assets": project.assets})
    except ValidationError as e:
        raise _explain(path, e) from None


def reveal_enabled(scene: Scene, fmt_name: str) -> bool:
    if scene.reveal == "auto":
        return fmt_name.startswith("long")
    return scene.reveal == "on"


def _file_stamp(p: Path | None) -> list | None:
    if p is None or not isinstance(p, Path):
        return None
    st = p.stat()
    return [str(p), st.st_mtime_ns, st.st_size]


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def depth_hash(scene: Scene, fmt: Format) -> str:
    """Changes only when something Blender renders changes."""
    return _sha({
        "model": _file_stamp(scene.model),
        "text3d": scene.text3d,
        "shape": scene.shape,
        "animation": scene.animation,
        "motion": scene.motion,
        "motion_settings": scene.motion_settings.model_dump(),
        "speed": scene.speed,
        "duration": scene.duration,
        "format": [fmt.name, fmt.width, fmt.height, fmt.safe.top, fmt.safe.bottom, fmt.safe.left, fmt.safe.right],
    })


def render_hash(scene: Scene) -> str:
    """Changes when anything affecting the finished videos changes."""
    return _sha({
        "scene": scene.model_dump(mode="json"),
        "files": [_file_stamp(scene.model),
                  _file_stamp(scene.pattern if isinstance(scene.pattern, Path) else None),
                  _file_stamp(scene.music if isinstance(scene.music, Path) else None)],
    })
