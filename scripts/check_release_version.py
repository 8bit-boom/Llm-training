#!/usr/bin/env python3
"""Fails (exit 1) when the release tag and the program's version disagree. Run by .github/workflows/release.yml before anything is built.

    python3 scripts/check_release_version.py v0.2.0     # the tag of the release
    python3 scripts/check_release_version.py            # no tag (a dry run): only the files are checked against each other

The one source of truth is `version` under [workspace.package] in Cargo.toml. src-tauri/tauri.conf.json must not carry a version of its own (Tauri then uses Cargo's),
the crates must take it from the workspace, and the Inno Setup script's fallback must be the same number (the build passes /DAppVersion from Cargo.toml anyway).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$")


def problems(tag, root=ROOT):
    out = []
    cargo = (root / "Cargo.toml").read_text(encoding="utf-8")
    m = re.search(r"\[workspace\.package\][^\[]*?^version\s*=\s*\"([^\"]+)\"", cargo, re.S | re.M)
    if not m:
        return ["Cargo.toml has no version under [workspace.package]."]
    version = m.group(1)
    if not SEMVER.match(version):
        out.append(f"The version {version!r} in Cargo.toml is not a semantic version (x.y.z).")
    if tag is not None and tag != f"v{version}":
        out.append(f"The tag is {tag!r} but Cargo.toml says version {version!r}; the tag must be v{version}. Change the version (one place: Cargo.toml) or tag again.")
    conf = json.loads((root / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
    if "version" in conf and conf["version"] != version:
        out.append(f"src-tauri/tauri.conf.json says version {conf['version']!r}; remove that line (Cargo.toml is the only source) or make it {version!r}.")
    for crate in ("src-tauri/Cargo.toml", "crates/ndforge-core/Cargo.toml"):
        text = (root / crate).read_text(encoding="utf-8")
        if not re.search(r"^version\.workspace\s*=\s*true", text, re.M):
            out.append(f"{crate} must take its version from the workspace (version.workspace = true).")
    iss = (root / "packaging" / "windows" / "ndforge.iss").read_text(encoding="utf-8")
    d = re.search(r'#define AppVersion "([^"]+)"', iss)
    if d and d.group(1) != version:
        out.append(f"packaging/windows/ndforge.iss falls back to AppVersion {d.group(1)!r}, not {version!r}.")
    return out


def main(argv):
    tag = argv[1] if len(argv) > 1 and argv[1] else None
    found = problems(tag)
    for p in found:
        print("ERROR:", p)
    if not found:
        print("Version check passed" + (f": tag {tag} matches Cargo.toml." if tag else " (no tag given): the files agree."))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
