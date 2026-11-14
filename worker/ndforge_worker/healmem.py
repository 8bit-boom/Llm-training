"""Healing and the graphics memory: the automatic settings by student size, the memory estimate, the out-of-memory detection and the "safer settings" ladder.

Pure Python (no torch import), so it is tested without a graphics card. The ladder and the activation estimate are the SAME rules as in the Rust program
(crates/ndforge-core/src/reshape/plan.rs: `heal_next_safer`, `heal_activation_gb`); both sides are tested against the shared cases in
worker/tests/data/heal_cases.json, so the page ("Try again with safer settings") and the worker (the automatic retries) agree.

The order of the ladder (none of these steps changes what the model learns):
    1. gradient checkpointing on        (the layers' activations are made again in the backward pass: about a third slower, far less memory)
    2. half the batch per step, twice the gradient accumulation   (the same number of sequences per update, so the same steps and the same checkpoints)
    3. a smaller KL chunk               (the output layer is applied to fewer tokens at a time; slower, less memory)
Changes that DO alter the maths (a shorter sequence length, a smaller LoRA rank) are never made automatically: the person is told about them.
"""
from __future__ import annotations

from .reshape import ReshapeError


class HealOutOfMemory(ReshapeError):
    """The final, plain-language message after the automatic retries ran out (shown on the card as it is, without a traceback)."""


MAX_RETRIES = 3                 # automatic retries after an out-of-memory error; each one loads the model again and goes on from the last checkpoint
MIN_CHUNK = 64
EFFECTIVE_BATCH = 32            # sequences per optimizer update: batch x accumulation stays at this unless the person chose otherwise
BIG_STUDENT_B = 15.0            # students of this size (billions of parameters) or more get the cautious defaults
CHECKPOINT_ALWAYS_B = 8.0       # gradient checkpointing is always on for students of this size or more
ACTIVATION_SHARE = 0.60         # auto: checkpointing on when the activations would take more than this part of the free memory


def is_oom(exc: BaseException) -> bool:
    """True for a graphics-memory failure: torch.OutOfMemoryError (torch.cuda.OutOfMemoryError), or a RuntimeError / torch.AcceleratorError whose text says
    "out of memory" ("CUDA error: out of memory" comes as an AcceleratorError in recent torch). A MemoryError (the computer's own memory) is not one."""
    if isinstance(exc, MemoryError):
        return False
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower() and not isinstance(exc, (KeyboardInterrupt, SystemExit))


def auto_defaults(student_b: float) -> dict:
    """The settings for a student of `student_b` billion parameters when the person chose nothing (owner decisions, docs/HEAL_SETTINGS.md)."""
    big = student_b >= BIG_STUDENT_B
    batch = 1 if student_b >= 25 else 2 if big else 4
    return {"max_seq": 1024 if big else 2048, "rank": 32 if big else 64, "batch": batch, "grad_accum": max(1, EFFECTIVE_BATCH // batch),
            "chunk": 128 if big else 256}


def activation_gb(layers: int, hidden: int, seq: int, batch: int, checkpointing: bool) -> float:
    """Activations kept for the backward pass, in GB. Per token and layer about 34 x hidden bytes without checkpointing (bf16; the figure of Korthikanti et al.
    for attention that does not keep the score matrix, MLP of 3 x hidden). With checkpointing only each layer's input is kept (2 x hidden bytes) plus the
    activations of the one layer that is being made again (34 x hidden)."""
    tokens = max(1, seq) * max(1, batch)
    per_token = (2 * layers * hidden + 34 * hidden) if checkpointing else 34 * layers * hidden
    return per_token * tokens / 1e9


def checkpointing_auto(student_b: float, activations_gb: float, free_gb: float | None) -> bool:
    """Auto: on for students of 8 B or more; below that on when the activations (without checkpointing) would take more than 60 % of the free memory. Unknown card: on."""
    if student_b >= CHECKPOINT_ALWAYS_B or free_gb is None:
        return True
    return activations_gb > ACTIVATION_SHARE * free_gb


def next_safer(s: dict):
    """The next safer setting that keeps the maths, from {"checkpointing": bool, "batch": int, "grad_accum": int, "chunk": int}.
    Returns (new settings, short label) or None when the ladder is used up. The sequences per update (batch x grad_accum) never change."""
    s = dict(s)
    if not s.get("checkpointing"):
        s["checkpointing"] = True
        return s, "gradient checkpointing"
    batch, acc = int(s["batch"]), int(s["grad_accum"])
    if batch > 1:
        new = batch // 2 if batch % 2 == 0 else 1
        s["batch"], s["grad_accum"] = new, acc * batch // new
        return s, f"a smaller batch ({new} per step, {s['grad_accum']} steps added up)"
    chunk = int(s["chunk"])
    if chunk > MIN_CHUNK:
        s["chunk"] = max(MIN_CHUNK, chunk // 2)
        return s, f"a smaller KL chunk ({s['chunk']})"
    return None


def describe(s: dict) -> str:
    """The settings in one plain line for logs and messages."""
    return (f"sequence length {s.get('max_seq')}, batch {s.get('batch')} x {s.get('grad_accum')} steps added up, "
            f"gradient checkpointing {'on' if s.get('checkpointing') else 'off'}, LoRA rank {s.get('rank')}, KL chunk {s.get('chunk')}, {s.get('method', 'lora').upper()}")


def advice(s: dict, tried: list) -> str:
    """The plain message when the card is still too small after the automatic retries: which step, the numbers, what was tried and what to do next, in order."""
    qlora = str(s.get("method", "")).lower() == "qlora"
    steps = [f"a shorter sequence length (now {s.get('max_seq')}; 512 or 1024 helps most)"]
    if not qlora:
        steps.append("QLoRA (4-bit) instead of LoRA 16-bit")
    steps += [f"a lower LoRA rank (now {s.get('rank')})", "a smaller model"]
    done = ("What was tried automatically: " + ", then ".join(tried) + ". ") if tried else ""
    return ("The graphics card ran out of memory during step train. " + f"Settings used: {describe(s)}. " + done +
            "What to change next, in this order: " + "; ".join(f"{i}. {t}" for i, t in enumerate(steps, 1)) +
            ". Changing the sequence length makes the original model answer again (the saved answers are for the old length); the other settings keep them.")


def set_alloc_conf() -> bool:
    """Before torch starts: let the graphics memory allocator grow in pieces that can be joined again ("expandable segments"), which keeps the memory from
    being cut into holes during long runs. Left alone when the person set it themselves, and on Windows (torch does not support it there and only warns).
    Returns True when it was set."""
    import os
    import sys
    if sys.platform == "win32" or os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF"):
        return False
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return True


def run_with_oom_retry(attempt, settings: dict, on_retry=None, free=None, max_retries: int = MAX_RETRIES):
    """attempt(settings, retry_number) trains (resuming from the last checkpoint). When it fails with an out-of-memory error the memory is freed (`free()`),
    the settings go one step down the ladder and it is called again, at most `max_retries` times. Other errors pass through. When the ladder or the
    retries are used up a HealOutOfMemory with the plain advice is raised (the numbers used are in the text). Returns (result, final settings, labels tried).
    The failed attempt's frames are released before `free()` runs (a caught exception keeps its traceback, and with it the model, alive)."""
    tried: list = []
    s = dict(settings)
    n = 0
    while True:
        failure = None
        try:
            return attempt(s, n), s, tried
        except Exception as exc:                         # noqa: BLE001 - only out-of-memory is handled here, the rest is raised again
            if not is_oom(exc):
                raise
            failure = f"{type(exc).__name__}: {str(exc)[:160]}"
        if free is not None:
            free()
        step = next_safer(s) if n < max_retries else None
        if step is None:
            raise HealOutOfMemory(advice(s, tried) + f" ({failure})")
        s, label = step
        tried.append(label)
        n += 1
        if on_retry is not None:
            on_retry(n, label, s)
