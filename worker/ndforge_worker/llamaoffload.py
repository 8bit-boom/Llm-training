"""The tuner's test of llama.cpp's offload (job `llama_offload_tune`): how many layers of a GGUF go to the graphics card (`-ngl`).

The program's own llama.cpp jobs (the importance matrix, Compare quants) always asked for `-ngl 99`: every layer on the card. A 16-bit GGUF of a big model does not fit a card, and llama.cpp then stops
with an allocation error instead of using the processor for the rest. This job runs a SHORT `llama-imatrix` pass (a few chunks of a neutral repeated text: its speed does not depend on the words) on the
GGUF the caller names, from the most layers to the fewest, and stops at the first number of layers that runs (more layers on the card is never slower, and the next lower one is not tried). Every try is a
row: seconds, seconds per pass when llama.cpp says it, the tokens per second, or why it failed (an out-of-memory text is marked). Nothing is written but the result; the temporary matrix is removed.

job: {"job": "llama_offload_tune", "gguf": file, "ctx": 512, "chunks": 6, "candidates": [99, 40, 30, ...], "work": folder, "stop_file": path}
result: {"rows": [...], "gguf": name, "gguf_mb", "tool": version line, "stopped"}
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from . import imatrix, llamacpp, protocol

PARAGRAPH = ("The old road crossed the river at the ford, where the three families had met every spring for as long as anyone remembered. "
             "Nobody wrote the terms down; each year the eldest of them said the words again and the others nodded. ")
OOM_MARKS = ("out of memory", "failed to allocate", "cudamalloc failed", "unable to allocate", "alloc_tensor_range: failed", "insufficient memory", "cuda error: out of memory")
PER_PASS = re.compile(r"([\d.]+)\s+seconds per pass")


class OffloadError(Exception):
    """A plain-language failure of the test."""


def is_oom(lines) -> bool:
    t = "\n".join(str(x) for x in lines).lower()
    return any(m in t for m in OOM_MARKS)


def seconds_per_pass(lines):
    """The first 'N seconds per pass' llama-imatrix prints after its first chunk, or None."""
    for line in lines:
        m = PER_PASS.search(str(line))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def calibration_text(chunks: int, ctx: int) -> str:
    """Enough repeated text for `chunks` contexts of `ctx` tokens (about 4 characters a token, with a good margin)."""
    need = int(chunks * ctx * 6) + 4000
    return (PARAGRAPH * (need // len(PARAGRAPH) + 1))[:need]


def row_of(ngl: int, rc: int, tail: list, seconds: float, out_exists: bool, chunks: int, ctx: int) -> dict:
    ok = rc == 0 and out_exists
    row = {"ngl": int(ngl), "ok": ok, "seconds": round(seconds, 2)}
    if ok:
        spp = seconds_per_pass(tail)
        row["sec_per_pass"] = spp
        row["tokens_per_s"] = round(ctx / spp, 1) if spp else round(chunks * ctx / max(seconds, 1e-9), 1)
        row["timed_by"] = "llama.cpp's own seconds per pass" if spp else "the whole run, including loading the file"
    else:
        row["oom"] = is_oom(tail)
        last = [str(x) for x in tail if str(x).strip()][-3:]
        row["error"] = protocol.scrub(" | ".join(last))[:300] or f"llama-imatrix ended with code {rc}"
    return row


def measure(candidates, run_one, stop_requested=lambda: False) -> tuple:
    """Runs `run_one(ngl) -> row` for the candidates in the order given and stops after the first row that is ok. -> (rows, stopped)."""
    rows = []
    for ngl in candidates:
        if stop_requested():
            return rows, True
        r = run_one(int(ngl))
        rows.append(r)
        if r["ok"]:
            break
    return rows, False


def run(job: dict) -> int:
    protocol.reserve_stdout()
    gguf = Path(job["gguf"])
    if not gguf.is_file():
        raise OffloadError(f"The GGUF file {gguf.name} is not there any more.")
    conv, quant = llamacpp.find_llama_cpp()
    tool = imatrix.find_llama_imatrix(quant) if quant else llamacpp.find_tool("llama-imatrix", None)
    if tool is None:
        raise OffloadError(llamacpp.not_found_text("the offload cannot be tested"))
    ctx, chunks = int(job.get("ctx") or 512), max(2, int(job.get("chunks") or 6))
    cands = [int(x) for x in job.get("candidates") or [99]]
    work = Path(job["work"])
    work.mkdir(parents=True, exist_ok=True)
    text = work / "offload_text.txt"
    text.write_text(calibration_text(chunks, ctx), encoding="utf-8")
    stop_file = job.get("stop_file")
    im = {"stop_file": stop_file}
    out = work / "offload_matrix.gguf"

    def stopped() -> bool:
        return bool(stop_file) and os.path.exists(stop_file)

    def run_one(ngl: int) -> dict:
        out.unlink(missing_ok=True)
        protocol.progress("tune_offload", 0.0, f"llama.cpp with {ngl} layers on the card")
        cmd = imatrix.launcher(tool) + ["-m", str(gguf), "-f", str(text), "-o", str(out), "--no-ppl", "--chunks", chunks, "-ngl", ngl, "-c", ctx]
        t0 = time.monotonic()
        rc, tail = imatrix.run_streaming(cmd, im, chunks, label=None, tail_len=400, log_tail=False)
        row = row_of(ngl, rc, tail, time.monotonic() - t0, out.is_file(), chunks, ctx)
        out.unlink(missing_ok=True)
        protocol.log(f"llama.cpp, {ngl} layers on the card: " + (f"{row['tokens_per_s']} tokens/s" if row["ok"] else f"did not run ({'out of memory' if row.get('oom') else row['error']})"))
        return row

    try:
        rows, was_stopped = measure(cands, run_one, stopped)
    except imatrix.Stopped:
        rows, was_stopped = [], True
    finally:
        text.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
    protocol.result({"rows": rows, "gguf": gguf.name, "gguf_mb": round(gguf.stat().st_size / 1e6), "tool": imatrix.tool_version(tool), "chunks": chunks, "ctx": ctx, "stopped": was_stopped})
    return 0
