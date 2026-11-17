"""Small file helpers shared by the jobs (no torch).

`replace_retry`: on Windows a rename or replace can fail for a moment (a virus scanner or the search indexer still holds the file or folder), so it is tried again a few times.
A big output (a GGUF file, a model folder) is written under a `.partial` name and moved to its real name only when it is complete, so a half file is never taken for a finished one."""
import os
import time
from pathlib import Path

PARTIAL = ".partial"


def replace_retry(src, dst, tries: int = 10, pause: float = 0.2) -> None:
    """os.replace, tried again (10 times, 0.2 s apart) when the operating system answers "permission denied" / "in use"."""
    for attempt in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt + 1 >= tries:
                raise
            time.sleep(pause)


def partial_of(path) -> Path:
    """`model-q4_k_m.gguf` -> `model-q4_k_m.gguf.partial`"""
    p = Path(path)
    return p.with_name(p.name + PARTIAL)


def remove_partials(folder, pattern: str = "*.gguf" + PARTIAL) -> list:
    """Deletes the half files an earlier, interrupted job left in `folder` (not below it). Returns the names removed."""
    gone = []
    f = Path(folder)
    if f.is_dir():
        for p in f.glob(pattern):
            try:
                p.unlink()
                gone.append(p.name)
            except OSError:
                pass
    return gone
