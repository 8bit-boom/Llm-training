import pytest

from ndforge.core.project import Project, ProjectError
from ndforge.ingest.bundle import open_bundle
from sample_bundle import WORLD, build


def test_create_open_roundtrip(tmp_path):
    p = Project.create(tmp_path, "My World")
    assert (tmp_path / "My World" / "project.toml").exists()
    assert Project.open(p.folder).name == "My World"
    with pytest.raises(ProjectError, match="already exists"):
        Project.create(tmp_path, "My World")
    with pytest.raises(ProjectError, match="name"):
        Project.create(tmp_path, "  ")
    with pytest.raises(ProjectError, match="not an ND Lore Forge project"):
        Project.open(tmp_path)


def test_import_progress_and_unlock(tmp_path):
    p = Project.create(tmp_path, "W")
    assert p.unlocked() == {"data": True, "dataset": False, "train": False, "test": False, "use": False}
    p.import_bundle(open_bundle(build(tmp_path / "a.ndtrain")))
    assert p.progress()["data"] and p.unlocked()["dataset"] and not p.unlocked()["train"]
    again = Project.open(p.folder)
    assert again.world_name == "Ashfall Reach" and again.current_bundle().counts["entities.jsonl"] == 2


def test_reimport_rules(tmp_path):
    p = Project.create(tmp_path, "W")
    first = open_bundle(build(tmp_path / "1.ndtrain", 1))
    p.import_bundle(first)
    with pytest.raises(ProjectError, match="same file"):
        p.import_bundle(first)
    newer = open_bundle(build(tmp_path / "2.ndtrain", 2))
    diff, problem = p.check_new_bundle(newer)
    assert problem is None and diff.total == 3
    p.import_bundle(newer)
    assert len(p.sources) == 2
    with pytest.raises(ProjectError, match="older"):
        p.import_bundle(first)
    other = open_bundle(build(tmp_path / "o.ndtrain", 1, world=dict(WORLD, slug="other", name="Other")))
    with pytest.raises(ProjectError, match="different world|for the world"):
        p.import_bundle(other)
