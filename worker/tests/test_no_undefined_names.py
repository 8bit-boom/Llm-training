"""The lint CI runs (`ruff check --isolated --select E9,F63,F7,F82 worker/ndforge_worker`) as a test too: an undefined name in a path that only a real model reaches
(a rename that missed one use) is a NameError at run time, and no torch-free test would notice."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "ndforge_worker"


def _ruff():
    exe = shutil.which("ruff")
    if exe:
        return [exe]
    probe = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True)
    return [sys.executable, "-m", "ruff"] if probe.returncode == 0 else None


def test_ruff_finds_no_undefined_names_in_the_worker():
    cmd = _ruff()
    if cmd is None:
        pytest.skip("ruff is not installed")
    p = subprocess.run(cmd + ["check", "--isolated", "--select", "E9,F63,F7,F82", str(PKG)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_the_old_tokenizer_helper_name_is_only_used_where_it_is_defined():
    import re
    bad = [f"{f.name}:{n}" for f in PKG.glob("*.py") if f.name != "data.py"
           for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1) if re.search(r"(?<![\w.])_inner\(", line)]
    assert not bad, bad
