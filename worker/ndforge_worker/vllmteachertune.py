"""The tuner's measurement of the vLLM teacher's engine settings (job `vllm_teacher_tune`; EXPERIMENTAL like the vLLM teacher, off unless the Auto page's test is ticked and vLLM is switched on)

Like the teacher check (`vllmteachercheck.py`) it makes ONE reference (an existing complete standard cache of the same original, else the standard 4-bit teacher on the same rows, freed before any container
starts) and then scores the SAME rows through `vllmcache.run_chunk` once per variant: the engine's own settings (`max_num_seqs`, `batch_tokens`, `gpu_memory_utilization`) one at a time from the
person's, the baseline twice for the noise. Every variant is a row: the scoring speed, the start-up, and how much its answers differ from the reference (top-1 agreement, mean KL), so that a faster
setting that changes the answers can be refused by the program. `run_chunk` always stops and removes its container, also on a failure or a stop; a variant that fails (memory, the engine) is a row with a
plain error and the next one is tried. Nothing is saved but the result.

job: the `vllm_teacher_check` job (teacher, student, model, data, k, max_seq, seed, engine, reference_cache, work, stop_file) without `out`, plus
     "variants": [{"id", "knob", "value", "label", "engine": {the engine keys this variant changes}}], "rows": n (default 128).
"""
from __future__ import annotations

import time
from pathlib import Path

from . import cachewrite, protocol, vllmcache, vllmteachercheck
from .vllmteachercheck import TeacherCheckError

DEFAULT_ROWS = 128


def is_oom(text: str) -> bool:
    t = (text or "").lower()
    return "out of memory" in t or "oom" in t.split() or "cuda error: out of memory" in t


def variant_engine(job: dict, v: dict) -> dict:
    """The job with the engine settings of one variant (the person's engine object with the variant's keys on top)."""
    return {**job, "engine": {**(job.get("engine") or {}), **(v.get("engine") or {})}}


def row_of(v: dict, res: dict, metrics: dict) -> dict:
    """The row of one finished variant."""
    return {"id": v["id"], "knob": v.get("knob") or "", "value": v.get("value"), "label": v.get("label") or v["id"], "ok": True, "tokens_per_s": res.get("tokens_per_s"), "tokens": res.get("tokens"),
            "startup_seconds": res.get("startup_seconds"), "seconds": res.get("seconds"), "top1": metrics.get("top1"), "kl": metrics.get("kl"), "jaccard": metrics.get("jaccard"),
            "no_overlap": metrics.get("no_overlap"), "gpu_util": (res.get("gpu") or {}).get("util"), "gpu_lowered": bool((res.get("gpu") or {}).get("lowered"))}


def run(job: dict) -> int:
    protocol.reserve_stdout()
    started = time.monotonic()
    import numpy as np
    from . import distilljob, reshapejob, safetensors_io
    k = int(job.get("k", 32))
    n_want = int(job.get("rows", DEFAULT_ROWS))
    max_seq = int(job.get("max_seq", 2048))
    repo = job["teacher"]["repo"]
    work = Path(job["work"])
    work.mkdir(parents=True, exist_ok=True)
    stop_file = Path(job["stop_file"]) if job.get("stop_file") else None
    if not vllmcache.wanted(job):
        raise TeacherCheckError("The job needs engine: {\"kind\": \"vllm\", ...} (the same settings the heal would use).")
    variants = [v for v in job.get("variants") or [] if isinstance(v, dict) and v.get("id")]
    if not variants:
        raise TeacherCheckError("The job lists no variants to measure.")
    teacher_dir = reshapejob._resolve_source(repo)
    weights = safetensors_io.weights_manifest(teacher_dir)
    rows, ref, max_seq, ref_source = vllmteachercheck.prepare(job, np, k, n_want, max_seq, teacher_dir, weights, stop_file)
    tokens = sum(len(r["input_ids"]) for r in rows)
    vin = work / "vllm_in"
    vin.mkdir(parents=True, exist_ok=True)
    distilljob.save_rows(vin, rows)
    out_rows: list = []
    stopped = False
    for n, v in enumerate(variants):
        if stop_file is not None and stop_file.exists():
            stopped = True
            break
        j = variant_engine(job, v)
        protocol.progress("tune_vllm", n / len(variants), f"vLLM teacher, setting {n + 1} of {len(variants)}: {v.get('label') or v['id']}")
        try:
            ident = vllmcache.identity(j, repo, teacher_dir)
            vout = work / f"vllm_out_{n}"
            vout.mkdir(parents=True, exist_ok=True)
            for f in vout.glob("*"):
                f.unlink()
            meta = {"teacher": repo, "k": k, "rows": len(rows), "data_hash": distilljob.data_hash(rows), "tokenizer_hash": "", "max_len": max_seq, "general_key": {}, "general_rows": 0, **ident}
            writer = cachewrite.CacheWriter(vout, meta, 262144)
            try:
                res = vllmcache.run_chunk({**j, "part_tokens": 262144}, writer, vin, meta, repo, teacher_dir, 0, len(rows), k, max_seq, stop_file)
            finally:
                writer.close()
            if res.get("stopped") or writer.done_rows < len(rows):
                stopped = True
                break
            rv = distilljob.CacheReader(vout)
            metrics = vllmteachercheck.compare(ref, [rv.get(i) for i in range(len(rows))], np)["overall"]
            row = row_of(v, res, metrics)
        except (vllmcache.VllmCacheError, vllmcache.vllmengine.EngineError, TeacherCheckError) as exc:
            msg = protocol.scrub(str(exc))[:300]
            row = {"id": v["id"], "knob": v.get("knob") or "", "value": v.get("value"), "label": v.get("label") or v["id"], "ok": False, "error": msg, "oom": is_oom(msg)}
        out_rows.append(row)
        protocol.log(f"vLLM teacher, {row['label']}: " + (f"{row['tokens_per_s']} tokens/s, top-1 agreement {row['top1']}" if row["ok"] else f"failed: {row['error']}"))
    protocol.progress("tune_vllm", 1.0, "")
    protocol.result({"rows": out_rows, "reference": ref_source, "rows_scored": len(rows), "tokens": tokens, "k": k, "max_seq": max_seq, "stopped": stopped, "seconds": round(time.monotonic() - started, 1)})
    return 0
