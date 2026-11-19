"""trace_gen: the original (or any model) writes thinking answers for prompts, so a heal can learn from them.

    EXPERIMENTAL on the vLLM way (docs/VLLM.md: built from a research report, not yet run on the owner's PC; throughput numbers there are estimates). The standard engine and llama-server work as they do
    for the thinking check; the way is chosen in this order unless the job names one: vLLM (when an engine is set up), llama-server (a GGUF file and llama.cpp), the standard engine.

    job: {"model": folder, GGUF file or Hub id, "backend": "auto" | "vllm" | "llama" | "hf",
          "source": {"file": jsonl/json/csv/txt} or {"id": Hugging Face dataset, "split", "config"}, "column": optional,   (prompts: the USER turns of the rows)
          "out_file": the jsonl to append to, "limit": prompts to try in this run (default 200), "max_new": 6144, "stop_file": path,
          "engine": {...} (vLLM), "parallel": 40, "load": "auto" | ..., "batch": 8 (standard engine)}

Output (`out_file`, one JSON object per line, the format of a `file` general source of a heal, `generaltext.py`):
    {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": "<think>\\n...\\n</think>\\n\\nanswer"}], "prompt_hash": "...", "tokens": n, "backend": "..."}
Rejected answers are not written to it: the reason is counted, and the prompt's hash goes to `<out_file>.rejected.jsonl` so a resumed run does not ask again (delete that file to retry).
An answer is kept when `</think>` was written, the answer after it is not empty, the limit did not cut it, it is no loop (`thinkcheck.loop_detect`) and not garbled (`thinkcheck.garbage_detect`).
Resumable: prompts whose hash is in either file are skipped; every kept answer is written (and flushed) as soon as it arrives; Stop ends between answers and keeps what was written.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import generaltext, protocol, thinkcheck, vllmengine
from .thinkcheck import ThinkCheckError

DEFAULT_LIMIT = 200
DEFAULT_MAX_NEW = 6144
MAX_PROMPT_CHARS = 6000
REASONS = ("not_closed", "empty_answer", "cut_by_limit", "loop", "garbled")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.strip().encode("utf-8")).hexdigest()[:20]


def _hashes(path: Path) -> set:
    out = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                h = json.loads(line).get("prompt_hash")
            except ValueError:
                continue
            if h:
                out.add(h)
    except OSError:
        pass
    return out


def prompts(source: dict, column: str | None = None):
    """The user turns of a dataset or file, in order, each once: a chat row gives its first user turn, a question+answer row its question, a text row its text."""
    seen = set()
    spec = None
    for row in generaltext.open_rows(source):
        if spec is None:
            spec = generaltext.detect(row, column)
        try:
            item = generaltext.row_item(row, spec) if spec["kind"] != "text" else ("text", generaltext.text_of(row.get(spec["column"])))
        except (KeyError, TypeError):
            continue
        if not item:
            continue
        kind, v = item
        p = v if kind == "text" else next((m["content"] for m in v if m["role"] == "user"), "")
        p = (p or "").strip()
        if not p or len(p) > MAX_PROMPT_CHARS:
            continue
        h = prompt_hash(p)
        if h not in seen:
            seen.add(h)
            yield p, h


def judge(raw: dict) -> tuple[str | None, str]:
    """(reason it is rejected | None, the text as `<think>...</think>` + answer): the filters of the report."""
    text = raw.get("text") or ""
    parts = thinkcheck.split_output(text, started_in_think=True)
    if raw.get("truncated"):
        return "cut_by_limit", text
    if not parts["closed"]:
        return "not_closed", text
    if not parts["answer"]:
        return "empty_answer", text
    if thinkcheck.loop_detect(text.split(), text):
        return "loop", text
    if thinkcheck.garbage_detect(text):
        return "garbled", text
    return None, f"<think>\n{parts['think']}\n</think>\n\n{parts['answer']}"


def choose_chain(job: dict) -> list:
    """The ways to try, in order: the named one alone, or (auto) vLLM when an engine is set up, llama-server for a GGUF file, then the standard engine."""
    want = str(job.get("backend") or "auto")
    if want != "auto":
        return [want]
    chain = []
    if job.get("engine"):
        try:
            vllmengine.engine_config(job)
            chain.append("vllm")
        except vllmengine.EngineError as exc:
            protocol.log(f"The vLLM engine is not available ({exc}); trying the next way.")
    model = str(job.get("model") or "")
    if model.lower().endswith(".gguf"):
        from . import llamacpp
        if llamacpp.find_tool("llama-server"):
            chain.append("llama")
    chain.append("hf")
    return chain


def run(job: dict) -> int:
    protocol.reserve_stdout()
    t0 = time.monotonic()
    model = str(job.get("model") or "")
    if not model or not job.get("out_file") or not isinstance(job.get("source"), dict):
        raise ThinkCheckError("trace_gen needs a model, a source of prompts and an output file.", "trace_gen_failed")
    out = Path(job["out_file"])
    rej_path = out.with_name(out.name + ".rejected.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done_hashes = _hashes(out) | _hashes(rej_path)
    limit = int(job.get("limit") or DEFAULT_LIMIT)
    todo = []
    skipped = 0
    try:
        for p, h in prompts(job["source"], job.get("column")):
            if h in done_hashes:
                skipped += 1
                continue
            todo.append((p, h))
            if len(todo) >= limit:
                break
    except generaltext.GeneralTextError as exc:
        raise ThinkCheckError(exc.plain, "trace_gen_failed") from exc
    if not todo:
        protocol.result({"out_file": str(out), "attempted": 0, "kept": 0, "rejected": {}, "skipped_existing": skipped, "backend": None, "stopped": False, "seconds": 0.0,
                         "note": "Nothing new to ask: every prompt of this source is already in the file (or was rejected before)."})
        return 0
    stop_file = Path(job["stop_file"]) if job.get("stop_file") else None
    should_stop = (lambda: stop_file.exists()) if stop_file else (lambda: False)
    settings = thinkcheck.settings_of({"max_new": int(job.get("max_new") or DEFAULT_MAX_NEW)})
    by_id = {h: p for p, h in todo}
    stats = {"kept": 0, "tokens": 0, "rejected": {r: 0 for r in REASONS}}
    answered: set = set()
    notes: list = []
    used = None

    def on_result(raw: dict) -> None:
        h = raw["id"]
        answered.add(h)
        stats["tokens"] += int(raw.get("gen_tokens") or 0)
        why, text = judge(raw)
        if why:
            stats["rejected"][why] += 1
            with open(rej_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"prompt_hash": h, "reason": why}) + "\n")
        else:
            row = {"messages": [{"role": "user", "content": by_id[h]}, {"role": "assistant", "content": text}], "prompt_hash": h, "tokens": int(raw.get("gen_tokens") or 0), "backend": used}
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["kept"] += 1
        n = len(answered)
        protocol.progress("trace", n / len(todo), f"Answer {n} of {len(todo)} ({stats['kept']} kept)", protocol.make_detail("answers", n, len(todo), t0, rate_unit="answers/s"))

    chain = choose_chain(job)
    last_exc = None
    for name in chain:
        used = name
        remaining = [(p, h) for p, h in todo if h not in answered]
        if not remaining:
            break
        reqs = [{"id": h, "messages": [{"role": "user", "content": p}], "thinking": True, "max_new": settings["max_new"]} for p, h in remaining]
        try:
            backend = thinkcheck.load_backend(name)
            backend.generate(reqs, settings, on_result, should_stop, job)
            last_exc = None
            break
        except ThinkCheckError as exc:
            last_exc = exc
            if len(chain) > 1 and name != chain[-1]:
                notes.append(f"The {name} way failed ({protocol.scrub(str(exc))[:160]}); the next way is used.")
                protocol.log(notes[-1])
                continue
            raise
    if last_exc is not None:
        raise last_exc
    spent = time.monotonic() - t0
    protocol.result({"out_file": str(out), "attempted": len(answered), "kept": stats["kept"], "rejected": {k: v for k, v in stats["rejected"].items() if v}, "skipped_existing": skipped,
                     "backend": used, "tokens": stats["tokens"], "tokens_per_s": round(stats["tokens"] / spent, 1) if spent > 0 and stats["tokens"] else None,
                     "stopped": len(answered) < len(todo), "seconds": round(spent, 1), "notes": notes,
                     "estimate_note": "Speeds are estimates from a research report until the vLLM self-test has been run on this PC."})
    return 0
