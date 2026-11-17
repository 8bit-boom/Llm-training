"""The thinking check: does a model still think, and think sensibly, after it was cut and healed?

A small suite of questions (`suites/think_suite.jsonl`: 32 questions in four topics, written for this program, plus 8 thinking-off control questions) is given to the
model with thinking ON and the sampling Qwen recommends for thinking (temperature 0.6, top-p 0.95, top-k 20, a fixed seed). The answers are scored and summed up:

    think_rate    a non-empty reasoning trace was written
    closed_rate   the trace was closed (`</think>` came) before the answer
    accuracy      the answer after `</think>` is the right one
    loop_rate     the text repeats itself (a 32-token run repeated 4 times in the last 1024 tokens, or it compresses to under 15 %)
    garbage_rate  replacement or control characters, or text in another script
    truncated_rate  the answer was cut by the token limit
    median / p90 trace tokens

and `gates` compares a cut model with the original (numbers in docs/THINKING_CHECK.md; the same cases are tested in Rust, worker/tests/data/think_gate_cases.json).

This file is the part that needs no torch: the suite, reading an output, scoring, the summary, the gates and the job around a generation backend. A backend
(`thinkcheck_hf.py`, `thinkcheck_llama.py`) only generates: backend.generate(requests, settings, on_result, should_stop, job): it calls on_result({"id", "text", "gen_tokens", "truncated", ...}) for each finished request.
"""
from __future__ import annotations

import importlib
import json
import math
import re
import time
import unicodedata
import zlib
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import protocol

SUITE_VERSION = 1
SUITE_FILE = Path(__file__).parent / "suites" / "think_suite.jsonl"
QUICK_ITEMS = 16                                                  # the first 16 main questions: four of each topic
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 3407}
MAX_NEW = {"full": 3072, "quick": 2048, "control": 512}
BACKENDS = {"hf": "ndforge_worker.thinkcheck_hf", "llama": "ndforge_worker.thinkcheck_llama"}
EST_TOKENS_PER_ITEM = 1500                                        # an average main answer with its trace (rough: about 50,000 tokens for the full suite)
EST_TOKENS_PER_CONTROL = 60
EST_TOKENS_PER_SECOND = {"hf": 55.0, "llama": 165.0}              # rough, for a 27B model on a 5090 (smaller models are faster)

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
LOOP_NGRAM, LOOP_WINDOW, LOOP_REPEATS, LOOP_RATIO = 32, 1024, 4, 0.15
GARBAGE_SHARE = 0.01
WRONG_SCRIPT_SHARE = 0.2


class ThinkCheckError(Exception):
    """A problem with a plain-language message and a short `code`."""

    def __init__(self, message: str, code: str = "think_check_failed"):
        super().__init__(message)
        self.code = code


# ------------------------------------------------------------------------------------------------------------------ the suite

def load_suite(path=None) -> dict:
    """{"version", "items": [32 main questions], "control": [8 thinking-off questions]}; each item {id, set, topic, prompt, answer, kind: number | exact | choice}."""
    p = Path(path) if path else SUITE_FILE
    version, items, control = None, [], []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "suite_version" in row:
            version = int(row["suite_version"])
            continue
        for k in ("id", "prompt", "answer", "kind"):
            if not row.get(k):
                raise ThinkCheckError(f"{p.name} line {n}: the item has no '{k}'.")
        if row["kind"] not in ("number", "exact", "choice"):
            raise ThinkCheckError(f"{p.name} line {n}: unknown kind {row['kind']!r}.")
        (control if row.get("set") == "control" else items).append(row)
    if version is None:
        raise ThinkCheckError(f"{p.name} has no suite_version line.")
    return {"version": version, "items": items, "control": control}


def settings_of(job: dict) -> dict:
    """The settings a check runs with (and its cache is keyed by): quick or full, sampling, token limits, and whether the control questions are asked."""
    quick = bool(job.get("quick", False))
    s = {**SAMPLING, **{k: job[k] for k in SAMPLING if job.get(k) is not None}}
    s.update(quick=quick, max_new=int(job.get("max_new") or MAX_NEW["quick" if quick else "full"]), control_max_new=MAX_NEW["control"],
             control=bool(job.get("control", True)), suite_version=SUITE_VERSION)
    return s


def questions(suite: dict, settings: dict) -> tuple[list, list]:
    main = suite["items"][:QUICK_ITEMS] if settings["quick"] else list(suite["items"])
    return main, (list(suite["control"]) if settings["control"] else [])


# ------------------------------------------------------------------------------------------------------------------ reading an output

def split_output(text: str, started_in_think: bool = True) -> dict:
    """{"think": the reasoning text or "", "answer": what follows it, "closed": </think> was written}. With thinking on, the prompt already ends with `<think>\\n`, so the
    generated text starts INSIDE the reasoning; a `<think>` at its start (a template that did not open the block) is dropped. Without a `</think>` the whole text is
    reasoning that never ended (answer "") when thinking is on, and the answer when it is not."""
    t = text or ""
    if THINK_CLOSE in t:
        head, _, answer = t.partition(THINK_CLOSE)
        think = head.split(THINK_OPEN)[-1] if THINK_OPEN in head else head
        return {"think": think.strip(), "answer": answer.strip(), "closed": True}
    if started_in_think or t.lstrip().startswith(THINK_OPEN):
        think = t.lstrip()[len(THINK_OPEN):] if t.lstrip().startswith(THINK_OPEN) else t
        return {"think": think.strip(), "answer": "", "closed": False}
    return {"think": "", "answer": t.strip(), "closed": False}


_ANSWER_LINE = re.compile(r"answer\s*(?:is\s*)?[:=]\s*(.+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def _clean(s: str) -> str:
    s = s.strip().strip("*_`\"'“”‘’ ")
    s = re.sub(r"^\\boxed\{(.*)\}$", r"\1", s)
    return s.strip().rstrip(".!").strip().strip("*_`\"'“”‘’ ")


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", _clean(s)).lower()


def _as_decimal(s: str):
    try:
        return Decimal(s.replace(",", ""))
    except InvalidOperation:
        return None


def _answer_part(text: str) -> str:
    """The text that holds the answer: what follows the last 'Answer:' (on its line), else the last non-empty line."""
    found = _ANSWER_LINE.findall(text or "")
    if found:
        return found[-1]
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def extract_answer(text: str, kind: str):
    """The model's answer from the text after </think>, normalised so it can be compared: a number as a plain decimal string, a choice as one capital letter, anything else as
    lower-case text with the markdown and the end punctuation taken off. None when there is nothing to read."""
    part = _answer_part(text)
    if not part.strip():
        return None
    if kind == "number":
        nums = _NUMBER.findall(part) or ([] if _ANSWER_LINE.search(text or "") else _NUMBER.findall(text or ""))
        d = _as_decimal(nums[-1]) if nums else None
        if d is None:
            return None
        s = format(d.normalize(), "f")
        return "0" if s in ("-0", "") else s
    if kind == "choice":
        c = _clean(part)
        m = re.fullmatch(r"\(?([A-Da-d])\)?[.):]?(?:\s.*)?", c)
        if m:
            return m.group(1).upper()
        found = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", part)
        return found[-1] if found else None
    return _norm_text(part) or None


def gold_answer(item: dict):
    """The item's right answer in the same normal form `extract_answer` gives."""
    kind, a = item["kind"], str(item["answer"])
    if kind == "number":
        d = _as_decimal(a)
        s = format(d.normalize(), "f") if d is not None else a
        return "0" if s in ("-0", "") else s
    if kind == "choice":
        return a.strip().upper()
    return _norm_text(a)


def is_correct(item: dict, answer_text: str) -> bool:
    got = extract_answer(answer_text, item["kind"])
    return got is not None and got == gold_answer(item)


# ------------------------------------------------------------------------------------------------------------------ what can go wrong

def loop_detect(tokens, text: str | None = None) -> bool:
    """True when the text repeats itself: some run of 32 tokens occurs 4 or more times in the last 1024 tokens, or (given `text` of at least 2,000 bytes) it
    compresses to less than 15 % of its size. `tokens` is a list of token ids (any hashable items; the llama backend passes words)."""
    toks = list(tokens or [])[-LOOP_WINDOW:]
    if len(toks) >= LOOP_NGRAM * LOOP_REPEATS:
        seen: dict = {}
        for i in range(len(toks) - LOOP_NGRAM + 1):
            g = tuple(toks[i:i + LOOP_NGRAM])
            seen[g] = seen.get(g, 0) + 1
            if seen[g] >= LOOP_REPEATS:
                return True
    if text:
        raw = text.encode("utf-8", "replace")
        if len(raw) >= 2000 and len(zlib.compress(raw, 6)) / len(raw) < LOOP_RATIO:
            return True
    return False


_FOREIGN = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "ARABIC", "HEBREW", "THAI", "DEVANAGARI", "BENGALI", "GREEK")


def garbage_detect(text: str) -> bool:
    """True for text that is broken: more than 1 % replacement characters or control characters (other than newline, return and tab), or more than 20 % of its letters
    in another script than the suite's (the questions are English)."""
    t = text or ""
    if not t:
        return False
    bad = sum(1 for c in t if c == "�" or (unicodedata.category(c) == "Cc" and c not in "\n\r\t"))
    if bad / len(t) > GARBAGE_SHARE:
        return True
    letters = foreign = 0
    for c in t:
        if c.isalpha():
            letters += 1
            if c > "ɏ" and unicodedata.name(c, "").startswith(_FOREIGN):
                foreign += 1
    return letters >= 20 and foreign / letters > WRONG_SCRIPT_SHARE


def score_item(item: dict, raw: dict, thinking: bool = True) -> dict:
    """One answered question. `raw`: {"text": the generated text, "gen_tokens": how many tokens, "truncated": the limit cut it, "tokens": ids (optional),
    "think_tokens": the length of the reasoning in tokens (optional; else about a quarter of its characters)}."""
    text = raw.get("text") or ""
    parts = split_output(text, started_in_think=thinking)
    answer = parts["answer"] if (parts["closed"] or not thinking) else ""
    got = extract_answer(answer, item["kind"]) if answer else None
    tt = raw.get("think_tokens")
    if tt is None:
        tt = int(round(len(parts["think"]) / 4)) if parts["think"] else 0
    return {"id": item["id"], "topic": item.get("topic"), "kind": item["kind"], "ok": bool(got is not None and got == gold_answer(item)), "got": got, "gold": gold_answer(item),
            "think": bool(parts["think"]), "closed": parts["closed"], "think_tokens": int(tt), "gen_tokens": int(raw.get("gen_tokens") or 0),
            "truncated": bool(raw.get("truncated")), "loop": loop_detect(raw.get("tokens") or text.split(), text), "garbage": garbage_detect(text)}


# ------------------------------------------------------------------------------------------------------------------ the summary and the gates

def _percentile(values: list, q: float) -> int:
    if not values:
        return 0
    v = sorted(values)
    return int(v[max(0, math.ceil(q * len(v)) - 1)])


def summarize(results: list) -> dict:
    """The numbers of one set of answered questions (all rates are shares of the questions asked, 0 to 1)."""
    n = len(results)
    if n == 0:
        return {"n": 0, "think_rate": 0.0, "closed_rate": 0.0, "accuracy": 0.0, "loop_rate": 0.0, "garbage_rate": 0.0, "truncated_rate": 0.0,
                "median_think_tokens": 0, "p90_think_tokens": 0}
    rate = lambda f: round(sum(1 for r in results if f(r)) / n, 4)      # noqa: E731
    traces = [r["think_tokens"] for r in results if r["think"]]
    return {"n": n, "think_rate": rate(lambda r: r["think"]), "closed_rate": rate(lambda r: r["closed"]), "accuracy": rate(lambda r: r["ok"]),
            "loop_rate": rate(lambda r: r["loop"]), "garbage_rate": rate(lambda r: r["garbage"]), "truncated_rate": rate(lambda r: r["truncated"]),
            "median_think_tokens": _percentile(traces, 0.5), "p90_think_tokens": _percentile(traces, 0.9)}


WARN_MARGIN = 0.05
GATES = ("closed_rate", "accuracy", "loop_rate", "garbage_rate", "median_think_tokens")


def gate_limits(original: dict) -> dict:
    """The limit of each gate, from the ORIGINAL model's numbers: {name: (direction, limit)}; "min": the value must reach the limit, "max": must not pass it."""
    o_closed, o_acc, o_loop, o_med = (float(original.get(k, 0) or 0) for k in ("closed_rate", "accuracy", "loop_rate", "median_think_tokens"))
    return {"closed_rate": ("min", max(0.9 * o_closed, 0.75)), "accuracy": ("min", 0.8 * o_acc), "loop_rate": ("max", o_loop + 0.10),
            "garbage_rate": ("max", 0.05), "median_think_tokens": ("max", 2.5 * max(o_med, 1.0))}


def gates(original: dict, candidate: dict) -> dict:
    """{"verdict": "pass" | "warn" | "fail", "checks": [{name, direction, limit, value, ok, warn}], "failed": [names], "warned": [names]}.
    Fail: any gate missed (closed_rate at least 0.9x the original's and 0.75; accuracy at least 0.8x; loop_rate at most the original's + 0.10; garbage_rate at most 0.05;
    the median trace at most 2.5x the original's). Warn: all met, but one is within 5 % of its limit."""
    eps = 1e-9
    checks = []
    for name, (direction, limit) in gate_limits(original).items():
        v = float(candidate.get(name, 0) or 0)
        if direction == "min":
            ok = v >= limit - eps
            warn = ok and limit > 0 and v < limit * (1 + WARN_MARGIN) - eps
        else:
            ok = v <= limit + eps
            warn = ok and limit > 0 and v > limit * (1 - WARN_MARGIN) + eps
        checks.append({"name": name, "direction": direction, "limit": round(limit, 4), "value": round(v, 4), "ok": ok, "warn": warn})
    failed = [c["name"] for c in checks if not c["ok"]]
    warned = [c["name"] for c in checks if c["ok"] and c["warn"]]
    return {"verdict": "fail" if failed else "warn" if warned else "pass", "checks": checks, "failed": failed, "warned": warned}


# ------------------------------------------------------------------------------------------------------------------ the jobs

def requests_for(main: list, control: list, settings: dict) -> list:
    """What the backend is asked to generate: [{"id", "messages", "thinking", "max_new"}]; the control questions are asked with thinking off."""
    out = [{"id": it["id"], "messages": [{"role": "user", "content": it["prompt"]}], "thinking": True, "max_new": settings["max_new"]} for it in main]
    out += [{"id": it["id"], "messages": [{"role": "user", "content": it["prompt"]}], "thinking": False, "max_new": settings["control_max_new"]} for it in control]
    return out


def plan(job: dict) -> dict:
    """What a check would do, without loading anything: the question counts, the limits, the sampling and a rough size (tokens and minutes)."""
    suite = load_suite()
    settings = settings_of(job)
    main, control = questions(suite, settings)
    backend = job.get("backend") or "hf"
    tokens = len(main) * EST_TOKENS_PER_ITEM * (settings["max_new"] / MAX_NEW["full"] if settings["quick"] else 1) + len(control) * EST_TOKENS_PER_CONTROL
    rate = EST_TOKENS_PER_SECOND.get(backend, EST_TOKENS_PER_SECOND["hf"])
    return {"suite_version": suite["version"], "items": len(main), "control_items": len(control), "backend": backend, "settings": settings,
            "est_tokens": int(tokens), "est_minutes": round(tokens / rate / 60, 1),
            "note": "A rough guess for a 27B model on a fast card; a smaller model is quicker. Thinking can run to the limit of " + str(settings["max_new"]) + " new tokens per question."}


def run_plan(job: dict) -> int:
    """job think_check_plan: {"backend": "hf" | "llama", "quick": false}. Result: see `plan`. No model is loaded."""
    protocol.reserve_stdout()
    protocol.result(plan(job))
    return 0


def load_backend(name: str):
    mod = BACKENDS.get(name)
    if not mod:
        raise ThinkCheckError(f"Unknown way to run the check: {name!r}. Use 'hf' or 'llama'.")
    try:
        return importlib.import_module(mod)
    except ImportError as exc:
        raise ThinkCheckError(f"The way to run the check ({name}) is not available: {exc}") from exc


def run_items(backend, main: list, control: list, settings: dict, should_stop=lambda: False, job: dict | None = None) -> dict:
    """Asks the backend, scores each answer as it arrives and emits `think_item` events. Returns {"items": [...], "control": [...], "stopped": bool}."""
    by_id = {it["id"]: (it, True) for it in main} | {it["id"]: (it, False) for it in control}
    reqs = requests_for(main, control, settings)
    done: list = []

    def on_result(raw: dict) -> None:
        item, is_main = by_id[raw["id"]]
        r = score_item(item, raw, thinking=is_main)
        r["control"] = not is_main
        done.append(r)
        protocol.emit("think_item", id=r["id"], ok=r["ok"], closed=r["closed"], think_tokens=r["think_tokens"], control=r["control"])
        protocol.progress("think", len(done) / max(1, len(reqs)), f"Question {len(done)} of {len(reqs)}")

    backend.generate(reqs, settings, on_result, should_stop, job or {})
    return {"items": [r for r in done if not r["control"]], "control": [r for r in done if r["control"]], "stopped": len(done) < len(reqs)}


def result_of(model: str, backend: str, settings: dict, run: dict, seconds: float) -> dict:
    return {"model": model, "backend": backend, "suite_version": SUITE_VERSION, "settings": settings, "main": summarize(run["items"]), "control": summarize(run["control"]),
            "items": run["items"] + run["control"], "stopped": run["stopped"], "seconds": round(seconds, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


CACHE_FORMAT = 1


def fingerprint(model: str) -> str:
    """What identifies the model that is checked: a GGUF file by name, size and time; a folder by its config and weight files (`reshapetry.weights_fingerprint`);
    a Hub id by the id itself (the revision is not looked at: the answer of an original that was updated on the Hub is not noticed)."""
    p = Path(model)
    if p.is_file():
        st = p.stat()
        return f"file:{p.name}:{st.st_size}:{int(st.st_mtime)}"
    if p.is_dir():
        from . import reshapetry
        return reshapetry.weights_fingerprint(p)
    return f"repo:{model}"


def cache_key(job: dict, backend: str, settings: dict) -> dict:
    """What the numbers depend on: the model, the way it is run and loaded, the suite and every setting. The same key is the same question."""
    return {"format": CACHE_FORMAT, "model": fingerprint(str(job.get("model") or "")), "backend": backend, "load": job.get("load") or "auto", "suite_version": SUITE_VERSION,
            "settings": settings}


def _cached(path: Path, key: dict):
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    res = saved.get("result") if isinstance(saved, dict) else None
    return res if saved.get("key") == key and isinstance(res, dict) and not res.get("stopped") else None


def _save_cache(path: Path, key: dict, res: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps({"key": key, "result": res}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run(job: dict) -> int:
    """job think_check: {"model": folder, GGUF file or Hub id, "backend": "hf" | "llama", "quick": false, "control": true, "load": "auto|4bit|16bit", "stop_file": path,
    "cache_file": path (optional)}. With a cache_file, a finished check of the same model, backend, load, suite and settings is answered from it ("cached": true) and a new
    one is saved there (the original model is checked once, however often it is compared). Result: see `result_of` (the numbers of `summarize` for the questions and for the
    thinking-off control, and one entry per question)."""
    protocol.reserve_stdout()
    started = time.monotonic()
    name = job.get("backend") or "hf"
    backend = load_backend(name)
    suite = load_suite()
    settings = settings_of(job)
    cache = Path(job["cache_file"]) if job.get("cache_file") else None
    key = cache_key(job, name, settings) if cache else None
    if cache and (hit := _cached(cache, key)) is not None:
        protocol.log("This model was checked before with the same settings; the saved numbers are used.")
        protocol.result({**hit, "cached": True})
        return 0
    main, control = questions(suite, settings)
    stop_file = Path(job["stop_file"]) if job.get("stop_file") else None
    should_stop = (lambda: stop_file.exists()) if stop_file else (lambda: False)
    run_ = run_items(backend, main, control, settings, should_stop, job)
    res = {**result_of(str(job.get("model") or ""), name, settings, run_, time.monotonic() - started), "cached": False}
    if cache and not res["stopped"]:
        _save_cache(cache, key, res)
    protocol.result(res)
    return 0
