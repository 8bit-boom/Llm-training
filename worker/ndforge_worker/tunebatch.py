"""The tuner's batch-size tests for the model's own forward passes (job `tune_batch_bench`; nothing is saved, only the result):

    what "calib"   Try first / the cut's measuring: how many calibration rows go through the ORIGINAL together (`calibration.batch`, 4 until now). The same rows, the same maths: every size reports the
                   mean token loss it measured, so that a size that changes the numbers can be refused. One forward pass per batch with hooks on the block boundaries, as `reshapejob._measure` does.
    what "think"   The thinking check: how many questions are asked together (`batch`, 8 until now). The same questions are generated with sampling at every size (the same seed); the speed is the
                   generated tokens per second. Sampling makes the SAME-ANSWER guarantee statistical only, so the program calls this class "same on average".

The sizes come from the caller (`sizes`, 0 = the program's own default). The model is loaded once; `healspeed.sweep` runs every size (an out-of-memory size is a row, the larger ones are skipped), measures the
noise by repeating the baseline and the fastest, and recommends. The result is shaped like the student's token-budget sweep (`batch_sweep.results` with `batch_tokens` = the size).

job: {"job": "tune_batch_bench", "what": "calib" | "think", "model": folder or hub id, "load": "auto" | "16bit" | "4bit" | "offload", "sizes": [0, 1, 2, 8, 16], "family": "qwen3_5",
      "calibration": {train, samples, max_len, seed, ...} (calib), "max_new": 96 (think), "repeat": 2}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import healspeed, protocol, reshape, reshapejob

DEFAULTS = {"calib": 4, "think": 8}
THINK_QUESTIONS = 16
THINK_MAX_NEW = 96


class BenchError(Exception):
    """A plain-language failure of the test."""


def effective(what: str, size: int) -> int:
    """The real batch of a size: 0 is the program's own default."""
    return int(size) if int(size) > 0 else DEFAULTS[what]


def prompts_in_order(inner, requests: list, prompt_of) -> tuple:
    """The requests with their prompts, shortest first (less padding), as the check itself orders them."""
    prompts = {r["id"]: prompt_of(inner, r) for r in requests}
    return sorted(requests, key=lambda r: (not r["thinking"], len(prompts[r["id"]]))), prompts


def think_requests(max_new: int = THINK_MAX_NEW, n: int = THINK_QUESTIONS) -> list:
    from . import thinkcheck
    settings = thinkcheck.settings_of({"quick": True, "max_new": max_new, "control": False})
    main, _ = thinkcheck.questions(thinkcheck.load_suite(), settings)
    return thinkcheck.requests_for(main[:n], [], settings)


def run(job: dict) -> int:
    protocol.reserve_stdout()
    what = job.get("what")
    if what not in DEFAULTS:
        raise BenchError(f"Unknown batch test {what!r}.")
    import torch
    sizes = [int(x) for x in job.get("sizes") or [0]]
    folder = reshapejob._resolve_source(str(job["model"]))
    from . import safetensors_io
    est = sum(r.nbytes for r in safetensors_io.list_tensors(folder).values())
    mode = reshapejob._choose_load(job.get("load", "auto"), est)
    protocol.progress("tune_batch", 0.0, f"Loading the model ({mode})")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(folder))
    model = reshapejob._load(folder, mode)
    cuda = torch.cuda.is_available()
    card = torch.cuda.mem_get_info()[1] / 1e6 if cuda else None
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    def timed(fn) -> dict:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        sync()
        t0 = time.monotonic()
        out = fn()
        sync()
        out["seconds"] = round(time.monotonic() - t0, 3)
        out["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6) if cuda else None
        out["tokens_per_s"] = round(out["tokens"] / max(out["seconds"], 1e-9), 2)
        return out

    def guarded(fn):
        def run_one(size: int) -> dict:
            protocol.progress("tune_batch", 0.5, f"Batch of {effective(what, size)}")
            try:
                return timed(lambda: fn(effective(what, size)))
            except RuntimeError as exc:
                reshapejob._free()
                if reshapejob._is_oom(exc):
                    return {"tokens_per_s": None, "error": "out of memory"}
                raise
        return run_one

    try:
        if what == "calib":
            config = json.loads((Path(folder) / "config.json").read_text(encoding="utf-8"))
            tc = reshape.text_config(config)
            n_layers = int(tc["num_hidden_layers"])
            period = reshapejob.check_period(config, reshape.layer_types_of(tc))
            layers = reshapejob._find_layers(model, n_layers)
            cal = dict(job.get("calibration") or {})
            family = job.get("family", "generic")

            def one_calib(bs: int) -> dict:
                batches = reshapejob._calibration(tok, {**cal, "batch": bs}, family)
                D, D1, D2, loss, tokens = reshapejob._measure(model, layers, period, batches, mode)
                return {"tokens": tokens, "first_loss": round(loss, 6) if loss is not None else None, "batches": len(batches)}
            run_one = guarded(one_calib)
        else:
            from . import thinkcheck_hf as hf
            from .data import text_tokenizer
            inner = text_tokenizer(tok)
            inner.padding_side = "left"
            if inner.pad_token_id is None:
                inner.pad_token = inner.eos_token
            reqs = think_requests(int(job.get("max_new") or THINK_MAX_NEW))
            order, prompts = prompts_in_order(inner, reqs, hf.prompt_of)
            settings = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 3407}
            stops = hf.stop_ids(inner, model)
            model.config.use_cache = True

            def one_think(bs: int) -> dict:
                tokens = 0
                for i in range(0, len(order), bs):
                    part = order[i:i + bs]
                    enc = inner([prompts[r["id"]] for r in part], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                    gen = hf._generate_batch(model, enc, settings, part[0]["max_new"], stops, inner.pad_token_id, settings["seed"])
                    tokens += sum(len(hf.split_generated(row, stops, inner.pad_token_id)[0]) for row in gen)
                return {"tokens": tokens, "first_loss": None, "questions": len(order)}
            run_one = guarded(one_think)
        sweep = healspeed.sweep(run_one, tuple(sizes), card, int(job.get("repeat") or 2))
    finally:
        model = None
        reshapejob._free()
    sweep["card_mb"] = round(card) if card else None
    protocol.progress("tune_batch", 1.0, "")
    protocol.result({"batch_sweep": sweep, "what": what, "load": mode, "stopped": False})
    return 0
