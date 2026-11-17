"""Guard: a test run must not leave new files directly in worker/ (stray build output, outputs written to the cwd)."""
import os
from pathlib import Path

_WORKER = Path(__file__).resolve().parents[1]
_before = set()


def pytest_sessionstart(session):
    global _before
    _before = set(os.listdir(_WORKER))


def pytest_sessionfinish(session, exitstatus):
    # Ignore caches pytest and Python create on their own.
    new = sorted(n for n in set(os.listdir(_WORKER)) - _before if n not in (".pytest_cache", "__pycache__", ".ruff_cache"))
    if new:
        print(f"\nERROR: the tests left new files in worker/: {', '.join(new)}")
        session.exitstatus = 1
