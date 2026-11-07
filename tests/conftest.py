import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("NDFORGE_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("NDFORGE_PROJECTS_DIR", str(tmp_path / "projects"))
