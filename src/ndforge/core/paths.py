"""Where ND Lore Forge keeps things."""
from __future__ import annotations

import os
from pathlib import Path

import platformdirs

APP = "NDLoreForge"


def config_dir() -> Path:
    override = os.environ.get("NDFORGE_CONFIG_DIR")
    return Path(override) if override else Path(platformdirs.user_config_dir(APP, appauthor=False))


def default_projects_dir() -> Path:
    override = os.environ.get("NDFORGE_PROJECTS_DIR")
    return Path(override) if override else Path.home() / "NDLoreForge" / "Projects"
