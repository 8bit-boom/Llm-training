"""Which training checkpoints can be resumed from (torch-free).

A checkpoint written while the computer crashed, the power failed or the disk filled can be cut short: the folder exists, but its weights or optimizer file are empty or missing.
`resume_from_checkpoint=True` would pick that newest folder and fail. `usable()` looks at every `checkpoint-<step>` folder, renames the broken newest ones to
`checkpoint-<step>.broken` (kept for a look, never picked again) and returns the good ones, oldest first; the training resumes from the last of them."""
import json
import re
from pathlib import Path

from . import protocol

_NAME = re.compile(r"^checkpoint-(\d+)$")
WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin", "model.safetensors", "model-*.safetensors", "pytorch_model.bin", "pytorch_model-*.bin")
OPTIMIZER = ("optimizer.pt", "optimizer.bin")


def step_of(path) -> int:
    m = _NAME.match(Path(path).name)
    return int(m.group(1)) if m else -1


def _files(folder: Path, names) -> list:
    out = []
    for n in names:
        out.extend(p for p in folder.glob(n) if p.is_file())
    return out


def problem(folder) -> str:
    """Why this checkpoint cannot be resumed from, or "" when it can."""
    folder = Path(folder)
    state = folder / "trainer_state.json"
    try:
        json.loads(state.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "trainer_state.json is missing or cut short"
    weights = _files(folder, WEIGHTS)
    if not weights or any(p.stat().st_size == 0 for p in weights):
        return "the saved weights are missing or empty"
    opt = _files(folder, OPTIMIZER)
    if not opt or any(p.stat().st_size == 0 for p in opt):
        return "the saved optimizer state is missing or empty"
    return ""


def usable(ckdir) -> list:
    """The checkpoints of `ckdir` that can be resumed from, sorted by step (oldest first). Broken ones are renamed `checkpoint-N.broken` and logged."""
    ckdir = Path(ckdir)
    if not ckdir.is_dir():
        return []
    found = sorted((p for p in ckdir.iterdir() if p.is_dir() and _NAME.match(p.name)), key=step_of)
    good = []
    for p in found:
        why = problem(p)
        if not why:
            good.append(p)
            continue
        target = p.with_name(p.name + ".broken")
        n = 1
        while target.exists():
            n += 1
            target = p.with_name(f"{p.name}.broken{n}")
        try:
            p.rename(target)
            protocol.log(f"The checkpoint {p.name} cannot be used ({why}); it was renamed to {target.name}. Going on from the one before it.")
        except OSError:
            protocol.log(f"The checkpoint {p.name} cannot be used ({why}) and could not be renamed; it is skipped.")
    return good
