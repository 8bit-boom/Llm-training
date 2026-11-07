"""Read, verify and (for tests / reference) write nd-world Training Bundles. Format: docs/BUNDLE_FORMAT.md."""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORMAT = "nd-training-bundle"
SUPPORTED_VERSION = 1
REQUIRED = ("manifest.json", "world.json", "entities.jsonl")
# Order in which members are written; also the list of members the reader knows about.
MEMBER_ORDER = (
    "manifest.json", "world.json", "entities.jsonl", "entity_notes.jsonl", "relations.jsonl", "rules.md",
    "rules_sheets.jsonl", "sessions.jsonl", "facts.jsonl", "quests.jsonl", "tables.jsonl", "ai_instructions.jsonl",
    "characters.jsonl",
)
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)

# Plain-language names used in summaries.
LABELS = {
    "entities.jsonl": "entities", "entity_notes.jsonl": "entity notes", "relations.jsonl": "relations",
    "rules_sheets.jsonl": "sheet rules", "sessions.jsonl": "session recaps", "facts.jsonl": "facts",
    "quests.jsonl": "quests", "tables.jsonl": "random tables", "ai_instructions.jsonl": "AI instructions",
    "characters.jsonl": "player characters",
}


class BundleError(Exception):
    """The file is not a usable training bundle. The message is written for the user."""


@dataclass
class Bundle:
    path: Path
    manifest: dict
    world: dict
    sha256: str  # hash of the whole file, identifies this snapshot
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def world_name(self) -> str:
        return self.manifest["world"].get("name", "")

    @property
    def world_slug(self) -> str:
        return self.manifest["world"].get("slug", "")

    def summary_lines(self) -> list[str]:
        out = []
        for member, label in LABELS.items():
            n = self.counts.get(member)
            if n:
                out.append(f"{n} {label}")
        if self.counts.get("rules.md"):
            out.append("world rules text")
        return out

    def read_text(self, member: str) -> str:
        with zipfile.ZipFile(self.path) as z:
            try:
                return z.read(member).decode("utf-8")
            except KeyError:
                return ""

    def records(self, member: str) -> Iterator[dict]:
        """Yield the records of a .jsonl member (nothing when the member is absent)."""
        with zipfile.ZipFile(self.path) as z:
            if member not in z.namelist():
                return
            with z.open(member) as fh:
                for n, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line.decode("utf-8"))
                    except ValueError as exc:
                        raise BundleError(f"{member}, line {n}: not valid JSON ({exc})") from exc


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def open_bundle(path: str | Path) -> Bundle:
    """Open and fully verify a bundle. Raises BundleError with a plain-language message."""
    path = Path(path)
    if not path.is_file():
        raise BundleError(f"The file {path.name} was not found.")
    if not zipfile.is_zipfile(path):
        raise BundleError("This file is not a training bundle (it is not a zip file). "
                          "Export it from nd-world: Tools > Export & Backup > Training bundle.")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if "manifest.json" not in names:
            raise BundleError("This zip file has no manifest.json, so it is not a training bundle.")
        try:
            manifest = json.loads(z.read("manifest.json").decode("utf-8"))
        except ValueError as exc:
            raise BundleError(f"manifest.json is damaged ({exc}).") from exc
        if manifest.get("format") != FORMAT:
            raise BundleError("This file is not a training bundle (wrong format name in manifest.json).")
        version = manifest.get("version")
        if not isinstance(version, int) or version > SUPPORTED_VERSION:
            raise BundleError(f"This bundle uses format version {version}, but this program only understands "
                              f"version {SUPPORTED_VERSION}. Update ND Lore Forge.")
        listed = manifest.get("files") or {}
        for req in REQUIRED:
            if req != "manifest.json" and req not in names:
                raise BundleError(f"The bundle is missing {req}.")
        extra = [n for n in names if n != "manifest.json" and n not in listed]
        if extra:
            raise BundleError(f"The bundle contains files that are not in its manifest: {', '.join(extra)}.")
        counts: dict[str, int] = {}
        for member, info in listed.items():
            if member not in names:
                raise BundleError(f"The bundle is missing {member}, which its manifest lists.")
            data = z.read(member)
            if len(data) != info.get("bytes", len(data)) or hashlib.sha256(data).hexdigest() != info.get("sha256"):
                raise BundleError(f"{member} is damaged (checksum does not match). Export the bundle again.")
            if member.endswith(".jsonl"):
                counts[member] = sum(1 for ln in data.split(b"\n") if ln.strip())
            elif member == "rules.md" and data.strip():
                counts[member] = 1
        try:
            world = json.loads(z.read("world.json").decode("utf-8"))
        except ValueError as exc:
            raise BundleError(f"world.json is damaged ({exc}).") from exc
    return Bundle(path=path, manifest=manifest, world=world, sha256=_file_sha256(path), counts=counts)


# ---------------------------------------------------------------- writer (reference implementation of the zip rules)

def _jsonl(records: list[dict]) -> bytes:
    return b"".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n" for r in records)


def write_bundle(path: str | Path, *, world: dict, members: dict[str, Any], nd_build: str = "test",
                 created: str | None = None) -> Path:
    """Write a bundle following docs/BUNDLE_FORMAT.md. `world` is the world.json document (needs id/name/slug/description);
    `members` maps member names (without manifest/world) to a list of records (.jsonl) or text (rules.md)."""
    payload: dict[str, bytes] = {"world.json": (json.dumps(world, ensure_ascii=False, indent=2) + "\n").encode("utf-8")}
    for name in MEMBER_ORDER:
        if name in ("manifest.json", "world.json") or name not in members:
            continue
        value = members[name]
        payload[name] = _jsonl(value) if name.endswith(".jsonl") else (value.rstrip("\n") + "\n").encode("utf-8")
    unknown = set(members) - set(MEMBER_ORDER)
    if unknown:
        raise ValueError(f"unknown bundle members: {sorted(unknown)}")
    files: dict[str, dict] = {}
    for name, data in payload.items():
        info: dict[str, Any] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        if name.endswith(".jsonl"):
            info["records"] = sum(1 for ln in data.split(b"\n") if ln.strip())
        files[name] = info
    manifest = {
        "format": FORMAT, "version": SUPPORTED_VERSION,
        "created": created or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nd_build": nd_build,
        "world": {k: world.get(k) for k in ("id", "name", "slug", "description")},
        "files": files,
    }
    payload["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path = Path(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in MEMBER_ORDER:
            if name in payload:
                zi = zipfile.ZipInfo(name, _FIXED_TIME)
                zi.external_attr = 0o644 << 16
                zi.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(zi, payload[name], compresslevel=6)
    return path
