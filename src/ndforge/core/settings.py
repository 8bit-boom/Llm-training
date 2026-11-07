"""Small app-wide settings (recent projects, simple/expert mode, theme). Stored as JSON in the config folder."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .paths import config_dir


@dataclass
class Settings:
    expert_mode: bool = False
    show_help_panel: bool = True
    welcome_seen: bool = False
    recent_projects: list[str] = field(default_factory=list)

    @staticmethod
    def _file() -> Path:
        return config_dir() / "settings.json"

    @classmethod
    def load(cls) -> Settings:
        try:
            data = json.loads(cls._file().read_text(encoding="utf-8"))
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self) -> None:
        try:
            self._file().parent.mkdir(parents=True, exist_ok=True)
            self._file().write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except OSError:
            pass  # settings are a convenience; never stop the app for them

    def remember_project(self, folder: Path) -> None:
        s = str(folder)
        self.recent_projects = [s] + [p for p in self.recent_projects if p != s][:9]
