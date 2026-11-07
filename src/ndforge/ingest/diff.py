"""Compare two bundles of the same world: what was added, changed or removed (shown before "Retrain with the new export")."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .bundle import LABELS, Bundle

# Fields that do not change the meaning of a record.
_IGNORE = {"updated_at", "created_at"}


@dataclass
class KindDiff:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.added) + len(self.changed) + len(self.removed)


@dataclass
class BundleDiff:
    kinds: dict[str, KindDiff]

    @property
    def total(self) -> int:
        return sum(k.total for k in self.kinds.values())

    def summary(self) -> str:
        if not self.total:
            return "Nothing changed."
        parts = []
        for member, d in self.kinds.items():
            if d.total:
                parts.append(f"{LABELS.get(member, member)}: {len(d.added)} new, {len(d.changed)} changed, {len(d.removed)} removed")
        return "; ".join(parts)


def _digest(rec: dict) -> str:
    clean = {k: v for k, v in rec.items() if k not in _IGNORE}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _label(rec: dict) -> str:
    return str(rec.get("name") or rec.get("title") or rec.get("content", "")[:40] or rec.get("id"))


def diff_bundles(old: Bundle, new: Bundle) -> BundleDiff:
    kinds: dict[str, KindDiff] = {}
    for member in LABELS:
        if member not in old.counts and member not in new.counts:
            continue
        a = {r.get("id"): r for r in old.records(member)}
        b = {r.get("id"): r for r in new.records(member)}
        d = KindDiff()
        for rid, rec in b.items():
            if rid not in a:
                d.added.append(_label(rec))
            elif _digest(a[rid]) != _digest(rec):
                d.changed.append(_label(rec))
        d.removed = [_label(rec) for rid, rec in a.items() if rid not in b]
        kinds[member] = d
    return BundleDiff(kinds)


def same_world(a: Bundle, b: Bundle) -> bool:
    wa, wb = a.manifest["world"], b.manifest["world"]
    return wa.get("slug") == wb.get("slug")
