"""A project = one world. Lives in a folder:

    project.toml            name, world, choices made so far
    sources/<sha12>.ndtrain every imported bundle (kept, so a diff to the previous one is possible)
    datasets/vN/            saved training-data versions   (step 2)
    runs/<name>/            training runs                    (step 3)
    exports/                GGUF / Modelfile                 (step 5)
"""
from __future__ import annotations

import shutil
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import tomli_w

from ..ingest.bundle import Bundle, BundleError, open_bundle
from ..ingest.diff import BundleDiff, diff_bundles, same_world

STEPS = ("data", "dataset", "train", "test", "use")


class ProjectError(Exception):
    """Plain-language problem with a project."""


@dataclass
class Project:
    folder: Path
    name: str
    world_name: str = ""
    world_slug: str = ""
    current_source: str = ""          # file name inside sources/
    sources: list[str] = field(default_factory=list)  # oldest first
    created: str = ""
    choices: dict = field(default_factory=dict)       # model, training type, stages, preset ... (filled by later steps)

    # ----------------------------------------------------------------- files
    @property
    def file(self) -> Path:
        return self.folder / "project.toml"

    @classmethod
    def create(cls, parent: Path, name: str) -> Project:
        name = name.strip()
        if not name:
            raise ProjectError("Please give the project a name.")
        safe = "".join(c if c.isalnum() or c in " -_." else "_" for c in name).strip(" .") or "project"
        folder = Path(parent) / safe
        if folder.exists() and any(folder.iterdir()):
            raise ProjectError(f"The folder {folder} already exists and is not empty. Choose another name.")
        for sub in ("sources", "datasets", "runs", "exports"):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        p = cls(folder=folder, name=name, created=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
        p.save()
        return p

    @classmethod
    def open(cls, folder: str | Path) -> Project:
        folder = Path(folder)
        f = folder / "project.toml"
        try:
            data = tomllib.loads(f.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ProjectError(f"{folder} is not an ND Lore Forge project (no project.toml).") from None
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ProjectError(f"The project file is damaged: {exc}") from exc
        w = data.get("world", {})
        return cls(folder=folder, name=data.get("name", folder.name), world_name=w.get("name", ""),
                   world_slug=w.get("slug", ""), current_source=data.get("source", {}).get("current", ""),
                   sources=list(data.get("source", {}).get("all", [])), created=data.get("created", ""),
                   choices=dict(data.get("choices", {})))

    def save(self) -> None:
        doc = {"name": self.name, "created": self.created,
               "world": {"name": self.world_name, "slug": self.world_slug},
               "source": {"current": self.current_source, "all": self.sources},
               "choices": self.choices}
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(tomli_w.dumps(doc), encoding="utf-8")
        tmp.replace(self.file)

    # --------------------------------------------------------------- sources
    def current_bundle(self) -> Bundle | None:
        if not self.current_source:
            return None
        try:
            return open_bundle(self.folder / "sources" / self.current_source)
        except (BundleError, OSError):
            return None

    def check_new_bundle(self, bundle: Bundle) -> tuple[BundleDiff | None, str | None]:
        """Returns (diff against the current snapshot or None for a first import, problem text or None)."""
        cur = self.current_bundle()
        if cur is None:
            return None, None
        if not same_world(cur, bundle):
            return None, (f"This file is for the world '{bundle.world_name}', but this project is for "
                          f"'{cur.world_name}'. Create a new project for a different world.")
        if bundle.sha256 == cur.sha256:
            return None, "This is the same file that is already imported."
        if bundle.manifest.get("created", "") < cur.manifest.get("created", ""):
            return None, "This file is older than the one already imported."
        return diff_bundles(cur, bundle), None

    def import_bundle(self, bundle: Bundle) -> Bundle:
        """Copy a verified bundle into the project and make it the current snapshot."""
        _, problem = self.check_new_bundle(bundle)
        if problem:
            raise ProjectError(problem)
        name = f"{bundle.sha256[:12]}.ndtrain"
        dest = self.folder / "sources" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(bundle.path, dest)
        self.current_source = name
        if name not in self.sources:
            self.sources.append(name)
        self.world_name, self.world_slug = bundle.world_name, bundle.world_slug
        self.save()
        return open_bundle(dest)

    # -------------------------------------------------------------- progress
    def progress(self) -> dict[str, bool]:
        """Which steps are finished; drives the ticks and unlocking in the sidebar."""
        datasets = [d for d in (self.folder / "datasets").glob("v*") if (d / "manifest.json").exists()]
        runs = [r for r in (self.folder / "runs").glob("*") if (r / "result.json").exists()]
        tested = [r for r in runs if (r / "test.json").exists()]
        exports = list((self.folder / "exports").glob("*.gguf"))
        return {"data": bool(self.current_source), "dataset": bool(datasets), "train": bool(runs),
                "test": bool(tested), "use": bool(exports)}

    def unlocked(self) -> dict[str, bool]:
        """A step opens when the steps it needs are done. Testing is optional: "use" opens once something is trained."""
        done = self.progress()
        return {"data": True, "dataset": done["data"], "train": done["data"] and done["dataset"],
                "test": done["train"], "use": done["train"]}
