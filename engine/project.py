"""Where everything lives inside a project folder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BUNDLED_FONT = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Lato-Regular.ttf"


@dataclass(frozen=True)
class Project:
    root: Path

    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def models(self) -> Path:
        return self.assets / "models"

    @property
    def patterns(self) -> Path:
        return self.assets / "patterns"

    @property
    def music(self) -> Path:
        return self.assets / "music"

    @property
    def fonts(self) -> Path:
        return self.assets / "fonts"

    @property
    def font_path(self) -> Path:
        """The project's own font, or the one shipped with the engine."""
        own = self.fonts / "Lato-Regular.ttf"
        return own if own.exists() else BUNDLED_FONT

    @property
    def scenes(self) -> Path:
        return self.root / "scenes"

    @property
    def compilations(self) -> Path:
        return self.root / "compilations"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def cache(self) -> Path:
        return self.root / "cache"
