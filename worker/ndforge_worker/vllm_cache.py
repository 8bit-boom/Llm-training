"""The teacher cache made by vLLM: a STANDALONE script that runs INSIDE the vLLM container (EXPERIMENTAL, off by default).

    python vllm_cache.py --model <folder or repo> --data <folder with rows.npz> --out <cache folder> --meta <meta.json> [--first-row N --last-row M] ...

It imports only the standard library, numpy, vllm (inside `build_scorer`) and `cachewrite.py` that lies next to it (no torch of ours, no other module of the worker package), reads
`rows.npz` (the token ids of the rows, as the cache job saves them), asks vLLM for the top-k log-probabilities at every position and writes the parts with the same numpy writer the
worker uses (byte-identical format, see tests/test_cachewrite.py).

    BUILT FROM A RESEARCH REPORT, NOT RUN ON THE OWNER'S PC. Everything vLLM-specific is marked "report" below and lives in ONE place each:
    - `build_scorer`            the `LLM(...)` and `SamplingParams(...)` arguments (report: max_logprobs=k, language_model_only, gpu_memory_utilization 0.88, `max_tokens=1`,
                                `prompt_logprobs=k`, `logprobs=k`, `flat_logprobs=True`, `detokenize=False`). A keyword the installed vLLM does not know is dropped with a plain note
                                (only the ones in OPTIONAL_*), everything else fails loudly.
    - `positions_of` / `convert_row`  the ONE place that turns vLLM's answer for a row into arrays. It understands (a) `FlatLogprobs`-like objects (flat arrays + start/end indices),
                                (b) a list with one `{token_id: Logprob | float}` dict per position (the older shape), (c) a pair of 2-D arrays; anything else is a plain error.

What a row's answer is: the cache holds, for every position i of a row, the distribution that predicts token i + 1 (what the HF teacher's hidden state i gives). vLLM's `prompt_logprobs[j]`
(j >= 1) is the distribution that predicts prompt token j, so position i is `prompt_logprobs[i + 1]`, and the LAST position (nothing follows it in the row) is the distribution of the one
token vLLM generates (`max_tokens=1`, greedy, `logprobs=k`, raw log-probabilities: report). The leftover mass is `max(0, 1 - sum(exp(top-k)))`, exactly `distilljob.topk_from_logits`;
the (k+1)-th entry vLLM adds when the actual token is outside the top k is dropped.

Output on stdout: JSON lines as in `protocol.py` (`log`, `progress` with a `detail` made like `protocol.make_detail`, a last `result`); errors as `{"type": "error", ...}`; exit 0 / 1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from . import cachewrite
except ImportError:                                      # run as a script: cachewrite.py lies next to this file
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cachewrite

PHASES = ("importing vllm", "starting engine and loading the model", "scoring")
OPTIONAL_LLM = ("language_model_only", "logprobs_mode", "generation_config", "enable_prefix_caching", "max_num_batched_tokens", "max_num_seqs", "disable_log_stats", "quantization")
OPTIONAL_SAMPLING = ("flat_logprobs", "detokenize")


class CacheScriptError(Exception):
    """A plain-language problem."""


# ----------------------------------------------------------------------------------------------------------------- output lines (the protocol, without importing it)

def emit(kind: str, **fields) -> None:
    sys.stdout.write(json.dumps({"type": kind, **fields}, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(text: str) -> None:
    emit("log", text=text)


def make_detail(unit: str, done, total, started: float | None = None, now: float | None = None, rate_unit: str | None = None, done_at_start=0) -> dict:
    """Same as `protocol.make_detail` (a test compares them)."""
    d: dict = {"unit": unit, "done": round(float(done), 3), "total": round(float(total), 3)}
    if started is not None:
        spent = (now if now is not None else time.monotonic()) - started
        gained = float(done) - float(done_at_start)
        if spent > 0 and gained > 0:
            rate = gained / spent
            d["rate"] = round(rate, 3)
            if rate_unit:
                d["rate_unit"] = rate_unit
            d["eta_s"] = round(max(0.0, float(total) - float(done)) / rate, 1)
    return d


def progress(step: str, value: float, text: str = "", detail: dict | None = None) -> None:
    if detail:
        emit("progress", step=step, value=round(float(value), 4), text=text, detail=detail)
    else:
        emit("progress", step=step, value=round(float(value), 4), text=text)


def timing(stage: str, seconds: float, **fields) -> dict:
    rec = {"stage": stage, "seconds": round(float(seconds), 2), **{k: v for k, v in fields.items() if v is not None}}
    log("timing: " + json.dumps(rec, ensure_ascii=False, sort_keys=True))
    return rec


# --------------------------------------------------------------------------------------------------------------------- batching

def plan_batches(lengths: list, first: int, last: int, budget: int, max_rows: int = 256) -> list:
    """[(a, b)] spans of consecutive rows `first..last` whose token total stays within `budget` (a row longer than the budget is a span of its own), at most `max_rows` rows each."""
    spans, a = [], first
    while a < last:
        b, tokens = a, 0
        while b < last and (b == a or (tokens + int(lengths[b]) <= budget and b - a < max_rows)):
            tokens += int(lengths[b])
            b += 1
        spans.append((a, b))
        a = b
    return spans


# ------------------------------------------------------------------------------------------------- vLLM's answer -> arrays (the ONE conversion, from the report)

def _logprob_value(x) -> float:
    """A `Logprob` object (`.logprob`), a dict with `logprob`, or a plain number."""
    if hasattr(x, "logprob"):
        return float(x.logprob)
    if isinstance(x, dict) and "logprob" in x:
        return float(x["logprob"])
    try:
        return float(x)
    except (TypeError, ValueError):
        raise CacheScriptError(f"A vLLM log-probability entry is a {type(x).__name__}; a Logprob object or a number was expected.") from None


def _flat_fields(obj):
    """(token_ids, logprobs, starts, ends) of a FlatLogprobs-like object (attributes or dict keys), or None."""
    def get(name):
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)
    ids, lps, st, en = get("token_ids"), get("logprobs"), get("start_indices"), get("end_indices")
    if ids is None or lps is None or st is None or en is None:
        return None
    return ids, lps, st, en


def positions_of(obj, np) -> list:
    """The per-position entries of one vLLM `prompt_logprobs` / `logprobs` value as a list: one `(ids int64 [m], logp float32 [m])` per position, `None` for a position without entries
    (position 0 of `prompt_logprobs` is `None` in vLLM). Shapes understood (report; unverified on a real vLLM): FlatLogprobs-like (flat arrays + start/end indices), a list of
    `{token_id: Logprob | float}` dicts (or None), a pair `(ids [p, m], logp [p, m])` of 2-D arrays. Anything else: a plain error naming what was seen."""
    if obj is None:
        return []
    flat = _flat_fields(obj)
    if flat is not None:
        ids, lps, st, en = (np.asarray(x) for x in flat)
        if len(st) != len(en):
            raise CacheScriptError(f"vLLM's flat log-probabilities have {len(st)} start and {len(en)} end indices.")
        out = []
        for a, b in zip(st.tolist(), en.tolist()):
            out.append(None if b <= a else (ids[a:b].astype(np.int64), lps[a:b].astype(np.float32)))
        return out
    if isinstance(obj, (list, tuple)) and len(obj) == 2 and all(hasattr(x, "ndim") and getattr(x, "ndim", 0) == 2 for x in obj):
        ids, lps = np.asarray(obj[0]), np.asarray(obj[1])
        if ids.shape != lps.shape:
            raise CacheScriptError(f"vLLM's log-probability arrays differ in shape: {ids.shape} and {lps.shape}.")
        return [(ids[i].astype(np.int64), lps[i].astype(np.float32)) for i in range(ids.shape[0])]
    if isinstance(obj, (list, tuple)):
        out = []
        for i, d in enumerate(obj):
            if d is None:
                out.append(None)
            elif isinstance(d, dict):
                items = [(int(t), _logprob_value(v)) for t, v in d.items()]
                out.append((np.array([t for t, _ in items], dtype=np.int64), np.array([v for _, v in items], dtype=np.float32)))
            else:
                raise CacheScriptError(f"Position {i} of vLLM's log-probabilities is a {type(d).__name__}; a dict of token id -> log-probability was expected.")
        return out
    raise CacheScriptError(f"vLLM's log-probabilities came back as a {type(obj).__name__}, a shape this program does not know (expected flat arrays, a list of dicts or two 2-D arrays). "
                           "Run the vLLM self-test and report this message.")


def top_k_of(entry, k: int, np, where: str):
    """(ids int64 [k], logp float32 [k]) of one position: the k largest by log-probability, in descending order (a stable sort, ties keep vLLM's order); the (k + 1)-th entry
    (the actual token when it lies outside the top k) is dropped."""
    if entry is None:
        raise CacheScriptError(f"vLLM gave no log-probabilities at {where}.")
    ids, lp = entry
    if len(ids) < k:
        raise CacheScriptError(f"vLLM gave {len(ids)} log-probabilities at {where}, {k} were asked for (is max_logprobs set to {k} or more?).")
    order = np.argsort(-lp, kind="stable")[:k]
    return ids[order], lp[order]


def convert_row(prompt_logprobs, gen_logprobs, n: int, k: int, np):
    """The cache arrays of ONE row of `n` tokens: (ids int64 [n, k], logp float32 [n, k], resid float32 [n]).
    Position i (i < n - 1) is `prompt_logprobs[i + 1]`; position n - 1 is the first (only) entry of `gen_logprobs`. See the module text."""
    if n < 1:
        raise CacheScriptError("A row without tokens cannot be scored.")
    pl = positions_of(prompt_logprobs, np)
    gl = positions_of(gen_logprobs, np)
    if n > 1 and len(pl) != n:
        raise CacheScriptError(f"vLLM gave log-probabilities for {len(pl)} prompt positions of a row with {n} tokens.")
    if not gl:
        raise CacheScriptError("vLLM gave no log-probabilities for the generated token (the position after the last token of the row).")
    ids = np.empty((n, k), dtype=np.int64)
    logp = np.empty((n, k), dtype=np.float32)
    for i in range(n - 1):
        ids[i], logp[i] = top_k_of(pl[i + 1], k, np, f"prompt position {i + 1}")
    ids[n - 1], logp[n - 1] = top_k_of(gl[0], k, np, "the generated token")
    resid = np.maximum(np.float32(0.0), np.float32(1.0) - np.exp(logp).sum(axis=-1, dtype=np.float32)).astype(np.float32)
    return ids, logp, resid


# ------------------------------------------------------------------------------------------------------------------------- the engine (vLLM)

def llm_kwargs(args) -> dict:
    """The `LLM(...)` arguments, from the report (unverified): see OPTIONAL_LLM for the ones dropped when the installed vLLM does not know them."""
    kw = {"model": args.model, "max_logprobs": int(args.k), "max_model_len": int(args.max_seq) + 8, "gpu_memory_utilization": float(args.gpu_util),
          "language_model_only": True, "generation_config": "vllm", "disable_log_stats": True, "logprobs_mode": "raw_logprobs", "enable_prefix_caching": False}
    if args.quantization:
        kw["quantization"] = args.quantization
    # ALWAYS explicit, "auto" as well: the engine must not take the type from anywhere else. An unknown keyword is NOT dropped: the identity and the result say which type this is
    kw["kv_cache_dtype"] = "fp8" if getattr(args, "kv_cache_dtype", "auto") == "fp8" else "auto"
    if args.max_num_batched_tokens:
        kw["max_num_batched_tokens"] = int(args.max_num_batched_tokens)
    if int(getattr(args, "max_num_seqs", 0) or 0) > 0:
        kw["max_num_seqs"] = int(args.max_num_seqs)           # vLLM's own default is 256: a hybrid (Mamba) model has far fewer cache blocks than that and refuses to start (the owner's log)
    kw.update(json.loads(args.llm_kwargs or "{}"))
    return kw


# vLLM refuses to start when max_num_seqs is more than its cache can hold: "max_num_seqs (256) exceeds available Mamba cache blocks (171) ... Please lower max_num_seqs to at most 171"
SEQS_LIMIT = re.compile(r"(?:lower max_num_seqs to at most|available Mamba cache blocks \()\s*(\d+)", re.I)


def seqs_limit(text: str):
    """The number N from vLLM's "max_num_seqs (256) exceeds available Mamba cache blocks (N)" / "lower max_num_seqs to at most N" error, or None for any other error."""
    m = SEQS_LIMIT.search(str(text or ""))
    return int(m.group(1)) if m and "max_num_seqs" in str(text) else None


def retry_seqs(limit: int, previous) -> int:
    """The value to try again with: at most what vLLM said, and at most half of what failed (so the second start does not land exactly on a limit that moves with the memory), never below 1."""
    cands = [int(limit)]
    if previous:
        cands.append(max(1, int(previous) // 2))
    return max(1, min(cands))


def sampling_kwargs(args) -> dict:
    return {"max_tokens": 1, "temperature": 0.0, "prompt_logprobs": int(args.k), "logprobs": int(args.k), "flat_logprobs": True, "detokenize": False}


def _unknown_kw(exc: Exception, allowed: tuple):
    m = re.search(r"unexpected keyword argument '(\w+)'", str(exc)) or re.search(r"got an unexpected keyword argument (\w+)", str(exc))
    return m.group(1) if m and m.group(1) in allowed else None


def build_scorer(args):
    """(scorer, notes): `scorer.score(rows)` -> the vLLM outputs for `rows` (lists of token ids), in order. Imports vllm here (and nowhere else)."""
    notes: list = []
    t0 = time.monotonic()
    import vllm                                               # noqa: PLC0415
    from vllm import LLM, SamplingParams                      # noqa: PLC0415
    try:
        from vllm.inputs import TokensPrompt                  # noqa: PLC0415
    except ImportError:
        TokensPrompt = None                                   # noqa: N806 - older vLLM: a dict with prompt_token_ids is accepted
    timing("import_vllm", time.monotonic() - t0, vllm=getattr(vllm, "__version__", "unknown"))
    skw = sampling_kwargs(args)
    for _ in range(len(OPTIONAL_SAMPLING) + 1):
        try:
            sp = SamplingParams(**skw)
            break
        except TypeError as exc:
            bad = _unknown_kw(exc, OPTIONAL_SAMPLING)
            if not bad or bad not in skw:
                raise
            notes.append(f"This vLLM does not know SamplingParams({bad}=...): left out.")
            skw.pop(bad)
    kw = llm_kwargs(args)
    t1 = time.monotonic()
    llm = None
    retried = False
    for _ in range(len(OPTIONAL_LLM) + 3):
        try:
            llm = LLM(**kw)
            break
        except TypeError as exc:
            bad = _unknown_kw(exc, OPTIONAL_LLM)
            if not bad or bad not in kw:
                raise
            notes.append(f"This vLLM does not know LLM({bad}=...): left out.")
            kw.pop(bad)
        except (ValueError, RuntimeError) as exc:
            limit = seqs_limit(str(exc))
            if limit is None or retried:
                if limit is not None:
                    raise CacheScriptError(f"vLLM needs a smaller max_num_seqs; the program already retried once (with {kw.get('max_num_seqs')}) and it still failed: {str(exc)[:300]}") from exc
                raise
            retried = True
            new = retry_seqs(limit, kw.get("max_num_seqs"))
            log(f"vLLM said at most {limit} sequences fit its cache (max_num_seqs was {kw.get('max_num_seqs', 'its default 256')}): trying again with max_num_seqs={new}.")
            notes.append(f"vLLM needs a smaller max_num_seqs; the program retried with {new}.")
            kw["max_num_seqs"] = new
    timing("engine_start_and_model_load", time.monotonic() - t1, **{key: kw[key] for key in ("quantization", "max_model_len") if key in kw})
    for n in notes:
        log(n)

    class Scorer:
        def __init__(self):
            self.notes = notes
            self.flat = bool(skw.get("flat_logprobs"))

        def score(self, rows: list) -> list:
            prompts = [TokensPrompt(prompt_token_ids=r) if TokensPrompt is not None else {"prompt_token_ids": r} for r in rows]
            return llm.generate(prompts, sp, use_tqdm=False)

    return Scorer(), notes


def outputs_to_rows(outputs, lengths: list, k: int, np) -> list:
    """[(ids, logp, resid)] for the outputs of one batch (one RequestOutput per row, in order)."""
    if len(outputs) != len(lengths):
        raise CacheScriptError(f"vLLM answered {len(outputs)} rows of {len(lengths)}.")
    rows = []
    for out, n in zip(outputs, lengths):
        gen = getattr(out, "outputs", None)
        if not gen:
            raise CacheScriptError("A vLLM answer has no generated token (outputs is empty).")
        rows.append(convert_row(getattr(out, "prompt_logprobs", None), getattr(gen[0], "logprobs", None), int(n), k, np))
    return rows


# ------------------------------------------------------------------------------------------------------------------------------ the job

def run(args, scorer=None, np=None) -> dict:
    """Writes the answers of rows `first..last` to `args.out` (resumable) and returns the result. `scorer` (a fake in the tests) answers `score(rows)`."""
    if np is None:
        import numpy as np                                    # noqa: PLC0415
    out = Path(args.out)
    stop_file = Path(args.stop_file) if args.stop_file else None
    ids_all, offsets = cachewrite.load_row_ids(args.data)
    total_rows = len(offsets) - 1
    first = int(args.first_row)
    last = total_rows if args.last_row is None or int(args.last_row) < 0 else min(int(args.last_row), total_rows)
    lengths = np.diff(offsets).tolist()
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8")) if args.meta else {}
    meta = {**meta, "rows": last - first}                    # this folder holds the chunk first..last only: its own `complete` means the chunk
    writer = cachewrite.CacheWriter(out, meta, int(args.part_tokens))
    start = first + writer.done_rows
    chunk_tokens = int(sum(lengths[first:last]))
    tokens_before = int(sum(lengths[first:start]))
    log(f"Rows {first} to {last} of {total_rows}; {writer.done_rows} of them are already in {out}." if writer.done_rows else f"Rows {first} to {last} of {total_rows}.")
    result = {"first_row": first, "last_row": last, "rows": last - first, "tokens": chunk_tokens, "stopped": False, "notes": []}
    if start >= last:
        writer.close()
        result.update(done_rows=writer.done_rows, tokens_scored=0, tokens_per_s=0.0, seconds=0.0, resumed=True)
        return result
    if scorer is None:
        progress("cache", 0.0, "Starting the vLLM engine and loading the model")
        scorer, notes = build_scorer(args)
        result["notes"] += notes
    t0 = time.monotonic()
    scored = 0
    try:
        for a, b in plan_batches(lengths, start, last, int(args.batch_tokens), int(args.max_batch_rows)):
            if stop_file is not None and stop_file.exists():
                result["stopped"] = True
                break
            rows = [ids_all[offsets[i]:offsets[i + 1]].astype(np.int64).tolist() for i in range(a, b)]
            got = outputs_to_rows(scorer.score(rows), lengths[a:b], int(args.k), np)
            for r in got:
                writer.add_row(*r)
            scored += int(sum(lengths[a:b]))
            progress("cache", (tokens_before + scored) / max(chunk_tokens, 1), f"vLLM scores your texts: {b - first} of {last - first}",
                     make_detail("tokens", tokens_before + scored, chunk_tokens, t0, rate_unit="tokens/s", done_at_start=tokens_before))
    finally:
        writer.close()
    secs = max(time.monotonic() - t0, 1e-9)
    if scored:
        timing("vllm_scoring", secs, tokens=scored, tokens_per_s=round(scored / secs, 1), rows=writer.done_rows)
    result.update(done_rows=writer.done_rows, tokens_scored=scored, tokens_per_s=round(scored / secs, 1), seconds=round(secs, 1))
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Teacher cache through vLLM (runs inside the vLLM container)")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="folder with rows.npz")
    p.add_argument("--out", required=True, help="cache folder (parts + index.json), written for rows first..last")
    p.add_argument("--meta", default="", help="JSON file with the cache meta (the keys distilljob.cache_meta makes)")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument("--first-row", type=int, default=0)
    p.add_argument("--last-row", type=int, default=None)
    p.add_argument("--part-tokens", type=int, default=262144)
    p.add_argument("--batch-tokens", type=int, default=32768, help="rows are sent to vLLM together up to this many tokens (not a fixed row count)")
    p.add_argument("--max-batch-rows", type=int, default=256)
    p.add_argument("--gpu-util", type=float, default=0.88)
    p.add_argument("--quantization", default="", help="passed to LLM(quantization=...), e.g. the bitsandbytes route (UNVERIFIED)")
    p.add_argument("--max-num-batched-tokens", type=int, default=0)
    p.add_argument("--max-num-seqs", type=int, default=32, help="LLM(max_num_seqs=...): rows scored at once; 0 = vLLM's default (256, too many for a hybrid model's cache)")
    p.add_argument("--llm-kwargs", default="", help="JSON object of more LLM(...) arguments")
    p.add_argument("--stop-file", default="")
    p.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
        res = run(args)
        emit("result", data=res)
        return 0
    except CacheScriptError as exc:
        emit("error", code="vllm_cache_failed", message=str(exc))
    except Exception as exc:                                  # noqa: BLE001 - the worker reads the message and falls back to the standard teacher
        emit("error", code="vllm_cache_failed", message=f"{type(exc).__name__}: {str(exc)[:500]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
