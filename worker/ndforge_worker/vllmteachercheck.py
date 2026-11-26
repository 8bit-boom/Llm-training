"""Is the vLLM teacher the same teacher? (job `vllm_teacher_check`; EXPERIMENTAL, off by default)

    BUILT FROM A RESEARCH REPORT, NOT RUN ON THE OWNER'S PC. The thresholds below are HEURISTICS (chosen here, not measured or taken from a paper); they are written next to every verdict.

The heal learns from the teacher's top-32 answers. When vLLM makes them (`vllmcache.py`), the weights can differ from the standard teacher's (4-bit NF4 through Hugging Face): a pre-quantized
copy (AWQ) is another quantization of the same model, the in-flight route should use the same NF4 numbers. This job measures how much the two differ on N real rows (default 100) of the
project's heal data at the heal's `max_seq`:

    reference   an existing, complete standard (Hugging Face, 4-bit) cache of the SAME original (its `rows.npz` gives the rows; no GPU is needed for the reference), else the standard teacher
                is run on the same rows first. The two models are never on the GPU at the same time (the standard teacher is freed before the container starts).
    vLLM        the rows go through `vllmcache.run_chunk` exactly as a cache job does (container, cp, exec, script), also timing the start-up phases and the scoring speed.
    metrics     top-1 agreement, top-32 set overlap (mean Jaccard), mean KL(reference || vLLM) over the support both give (each side renormalised on it), the difference of the leftover
                ("tail") mass, and the same by position in the row.
    verdict     "same" | "close" | "different" with the thresholds in THRESHOLDS and a plain sentence. "different" makes the heal and Auto-cut refuse the vLLM teacher; not having run the check only warns.

Result and file `vllm_teacher_check.json` (job key `out`; the program puts it in `models/<name>/`).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import cachewrite, protocol, vllmcache

# HEURISTICS (not measured): both numbers must hold for the better verdict.
THRESHOLDS = {"same": {"top1": 0.97, "kl": 0.02}, "close": {"top1": 0.93, "kl": 0.08}, "max_no_overlap": 0.01}
POSITION_BUCKETS = ((0, 128), (128, 512), (512, 1024), (1024, 2048), (2048, 10 ** 9))
DEFAULT_ROWS = 100


class TeacherCheckError(Exception):
    """A plain-language failure of the check."""


# ----------------------------------------------------------------------------------------------------------------------------- metrics (numpy only)

def position_stats(a, b, np):
    """Per-position numbers of two answers for the SAME row: a = (ids [n, k], logp [n, k], resid [n]) the reference, b = the vLLM answers.
    -> dict of arrays [n]: top1 (0/1), jaccard, kl (nan where the two share no token), tail_diff (b - a), shared (the reference's probability mass on the shared tokens)."""
    ia, la, ra = a[0].astype(np.int64), a[1].astype(np.float32), np.asarray(a[2], dtype=np.float32)
    ib, lb, rb = b[0].astype(np.int64), b[1].astype(np.float32), np.asarray(b[2], dtype=np.float32)
    if ia.shape != ib.shape:
        raise TeacherCheckError(f"The two answers have different shapes ({ia.shape} and {ib.shape}).")
    eq = ia[:, :, None] == ib[:, None, :]                          # [n, k, k]
    inter = eq.sum(axis=(1, 2)).astype(np.float32)
    k = ia.shape[1]
    jac = inter / np.maximum(2 * k - inter, 1)
    # top-1: the same token on the largest entry of both (the writers store them in descending order, but sort anyway)
    t1 = ia[np.arange(len(ia)), la.argmax(axis=1)] == ib[np.arange(len(ib)), lb.argmax(axis=1)]
    shared_a = eq.any(axis=2)                                      # entry i of the reference is also in vLLM's top-k
    lq = np.where(eq, lb[:, None, :], -np.inf).max(axis=2)         # vLLM's log-probability of the reference's entry i
    p = np.where(shared_a, np.exp(la), 0.0)
    q = np.where(shared_a, np.exp(np.where(shared_a, lq, 0.0)), 0.0)
    ps, qs = p.sum(axis=1), q.sum(axis=1)
    ok = (ps > 0) & (qs > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pn, qn = p / ps[:, None], q / qs[:, None]
        terms = np.where(shared_a & (pn > 0), pn * (np.log(np.where(pn > 0, pn, 1.0)) - np.log(np.where(qn > 0, qn, 1.0))), 0.0)
    kl = np.where(ok, terms.sum(axis=1), np.nan)
    return {"top1": t1.astype(np.float32), "jaccard": jac, "kl": kl.astype(np.float32), "tail_diff": (rb - ra).astype(np.float32), "shared": ps.astype(np.float32)}


def _agg(parts: dict, np) -> dict:
    n = int(len(parts["top1"]))
    if n == 0:
        return {"positions": 0}
    kl = parts["kl"]
    good = ~np.isnan(kl)
    return {"positions": n, "top1": round(float(parts["top1"].mean()), 5), "jaccard": round(float(parts["jaccard"].mean()), 5),
            "kl": round(float(kl[good].mean()), 6) if good.any() else None, "no_overlap": round(float((~good).mean()), 5),
            "tail_diff_mean": round(float(parts["tail_diff"].mean()), 6), "tail_diff_abs": round(float(np.abs(parts["tail_diff"]).mean()), 6),
            "shared_mass": round(float(parts["shared"].mean()), 5)}


def compare(ref_rows: list, vllm_rows: list, np) -> dict:
    """{"overall": {...}, "by_position": [{"from", "to", ...}], "rows": n}: see the module text. Both lists hold (ids, logp, resid) per row, in the same order."""
    if len(ref_rows) != len(vllm_rows):
        raise TeacherCheckError(f"The reference has {len(ref_rows)} rows and vLLM {len(vllm_rows)}.")
    cols: dict = {}
    pos = []
    for a, b in zip(ref_rows, vllm_rows):
        s = position_stats(a, b, np)
        for kk, v in s.items():
            cols.setdefault(kk, []).append(v)
        pos.append(np.arange(len(a[2])))
    if not pos:
        return {"overall": {"positions": 0}, "by_position": [], "rows": 0}
    cols = {kk: np.concatenate(v) for kk, v in cols.items()}
    where = np.concatenate(pos)
    buckets = []
    for lo, hi in POSITION_BUCKETS:
        sel = (where >= lo) & (where < hi)
        if sel.any():
            buckets.append({"from": lo, "to": None if hi >= 10 ** 9 else hi, **_agg({kk: v[sel] for kk, v in cols.items()}, np)})
    return {"overall": _agg(cols, np), "by_position": buckets, "rows": len(ref_rows)}


def verdict_of(overall: dict) -> dict:
    """{"verdict": "same" | "close" | "different", "text": plain sentence, "thresholds"}. HEURISTIC thresholds, see THRESHOLDS."""
    t1, kl, miss = overall.get("top1"), overall.get("kl"), overall.get("no_overlap") or 0.0
    th = THRESHOLDS
    if t1 is None or kl is None or miss > th["max_no_overlap"]:
        v, why = "different", "The two teachers share no answers at too many positions to be compared."
    elif t1 >= th["same"]["top1"] and kl <= th["same"]["kl"]:
        v, why = "same", f"The vLLM teacher gives the same top answer as the standard teacher at {t1 * 100:.1f}% of the positions and the two distributions are very close (mean KL {kl:.4f})."
    elif t1 >= th["close"]["top1"] and kl <= th["close"]["kl"]:
        v, why = "close", (f"The vLLM teacher agrees with the standard teacher at {t1 * 100:.1f}% of the positions (mean KL {kl:.4f}): close but not the same, the way two quantizations of one model differ. "
                           "Healing with it works, the student learns a slightly different teacher.")
    else:
        v, why = "different", f"The vLLM teacher differs too much from the standard teacher (top answer agrees at {t1 * 100:.1f}% of the positions, mean KL {kl:.4f}): it is NOT used for healing."
    return {"verdict": v, "text": why, "thresholds": {"same": th["same"], "close": th["close"], "max_no_overlap": th["max_no_overlap"],
                                                     "note": "Heuristics chosen for this program, not measured or published values."}}


def effective_job(job: dict, kvd: str) -> dict:
    """The job with the KV cache type of THIS measurement and no other: "fp8" turns both switches on (asking for the measurement is the confirmation); "auto" turns them OFF, whatever the owner's engine settings
    say (the settings had made an "auto" check run with an 8-bit cache: the owner's log)."""
    e = {**(job.get("engine") or {}), "kv_cache_dtype": kvd, "kv_cache_dtype_teacher": kvd == "fp8"}
    return {**job, "engine": e}


def kv_confirmation(requested: str, res: dict) -> dict:
    """{"requested", "seen" (from the engine's own log: fp8 / auto / None), "evidence", "agrees": True | False | None (the log said nothing)}."""
    seen = (res.get("kv_cache_seen") or {})
    s = seen.get("seen")
    return {"requested": requested, "seen": s, "evidence": seen.get("evidence") or [], "agrees": None if s is None else s == requested}


def tail_note(overall: dict, k: int) -> str:
    """The systematic tail difference, said plainly (the owner's runs: about -0.028 at every position range)."""
    t = overall.get("tail_diff_mean")
    if t is None or t > -0.01:
        return ""
    return (f" The vLLM teacher puts about {abs(t) * 100:.1f} % less probability outside its top {k} than the standard teacher does; the heal's tail term is affected a little; the cause is not known.")


def with_kv(v: dict, conf: dict, overall: dict, k: int) -> dict:
    """The verdict with the KV cache type it measured in its text; INVALID when the engine's log and the request disagree (the numbers then measure the other cache type)."""
    kvd = conf["requested"]
    if conf["agrees"] is False:
        return {**v, "verdict": "invalid", "valid": False,
                "text": f"INVALID: this check asked for the {kvd} KV cache, but the engine's own log says it used {conf['seen']}. The numbers measure the {conf['seen']} cache, not the {kvd} one: do not trust the verdict ({v['verdict']}) for the {kvd} cache."}
    how = "confirmed by the engine's log" if conf["agrees"] else "the engine's log did not say; taken as asked"
    return {**v, "valid": True, "text": v["text"] + f" Measured with the {kvd} KV cache ({how})." + tail_note(overall, k)}


def kv_difference(a: dict, b: dict) -> dict:
    """fp8 minus auto for the numbers that matter (None where one is missing)."""
    out = {}
    for key in ("top1", "jaccard", "kl", "tail_diff_mean", "tail_diff_abs", "shared_mass", "no_overlap"):
        x, y = a.get(key), b.get(key)
        out[key] = None if x is None or y is None else round(y - x, 6)
    return out


def pick_rows(n_available: int, n: int, np) -> list:
    """`n` indices spread evenly over `n_available` rows (all of them when fewer)."""
    if n_available <= 0:
        return []
    if n_available <= n:
        return list(range(n_available))
    return sorted({int(i) for i in np.linspace(0, n_available - 1, n)})


# ---------------------------------------------------------------------------------------------------------------------------------- the job

def prepare(job: dict, np, k: int, n_want: int, max_seq: int, teacher_dir, weights: str, stop_file):
    """The rows and the reference answers of a check or a tuning measurement: an existing complete standard cache of the SAME original (job `reference_cache`) gives both without a GPU; else the standard teacher
    (4-bit) answers `n_want` rows of the project's data first and is freed before the engine's container starts. -> (rows, ref, max_seq, ref_source)."""
    from . import distilljob, reshapejob
    ref_dir = job.get("reference_cache")
    idx = _usable_reference(ref_dir, weights, k) if ref_dir else None
    ref_source = None
    if idx is not None:
        all_rows = distilljob.load_rows(ref_dir)
        have = min(int(idx.get("done_rows", 0)), len(all_rows))
        picks = pick_rows(have, n_want, np)
        rows = [all_rows[i] for i in picks]
        max_seq = int(idx.get("max_len") or max_seq)
        rd = distilljob.CacheReader(ref_dir)
        ref = [rd.get(i) for i in picks]
        ref_source = {"kind": "existing_cache", "folder": str(ref_dir), "note": f"An existing complete standard cache of the same original was used as the reference ({len(rows)} of its rows); no GPU was needed for it."}
        protocol.log(ref_source["note"])
    else:
        from transformers import AutoTokenizer
        student = Path(job["student"])
        tok = AutoTokenizer.from_pretrained(str(student))
        all_rows, _stats = distilljob.build_rows(tok, job["data"], job["model"].get("family", "generic"), max_seq, int(job.get("seed", 3407)))
        proj = [r for r in all_rows if r.get("source") == "project"] or all_rows
        rows = [proj[i] for i in pick_rows(len(proj), n_want, np)]
        from .data import text_tokenizer
        inner = text_tokenizer(tok)
        pad = inner.pad_token_id if inner.pad_token_id is not None else (inner.eos_token_id or 0)
        ref = _hf_reference(teacher_dir, rows, k, pad, stop_file)
        reshapejob.release_gpu("the 4-bit original")             # the engine's container needs the card: nothing of the reference run may stay on it
        ref_source = {"kind": "ran_standard_teacher", "note": f"The standard teacher (4-bit) answered the same {len(rows)} rows first; the two models were never on the card together."}
    if not rows:
        raise TeacherCheckError("There are no rows to compare (the project's heal data is empty).")
    return rows, ref, max_seq, ref_source



def _usable_reference(folder, origin_weights: str, k: int):
    """The index of `folder` when it is a complete standard (not vLLM) cache of this original with this k; else None."""
    idx = cachewrite.read_cache_index(folder)
    if not idx or idx.get("engine") or idx.get("teacher_quant"):
        return None
    if idx.get("k") != k or int(idx.get("done_rows", 0)) < 1:
        return None
    if idx.get("teacher_weights") and origin_weights and idx["teacher_weights"] != origin_weights:
        return None
    if (idx.get("teacher_load") or "auto") not in ("auto", "4bit"):
        return None                                                 # a 16-bit or offloaded teacher is not the 4-bit teacher the heal normally uses
    return idx


def _hf_reference(teacher_dir, rows: list, k: int, pad_cfg, stop_file):
    """The standard teacher (4-bit NF4) on `rows`: [(ids, logp, resid)]. Frees the model and the card before it returns."""
    from . import distilljob, reshapejob
    mode = "4bit"
    protocol.progress("check_reference", 0.0, "Loading the original model (4-bit) to make the reference answers")
    model = reshapejob._load(teacher_dir, mode, "load_teacher")
    out = []
    try:
        pad = int(pad_cfg or 0)
        for i, r in enumerate(rows):
            if stop_file is not None and Path(stop_file).exists():
                raise TeacherCheckError("Stopped.")
            out += distilljob._teacher_answers(model, [r], k, 256, pad)
            protocol.progress("check_reference", (i + 1) / len(rows), f"Reference answers: {i + 1} of {len(rows)}")
    finally:
        del model
        reshapejob._free()
    return out


def run(job: dict) -> int:
    protocol.reserve_stdout()
    started = time.monotonic()
    import numpy as np
    from . import distilljob, reshapejob, safetensors_io
    k = int(job.get("k", 32))
    n_want = int(job.get("rows", DEFAULT_ROWS))
    max_seq = int(job.get("max_seq", 2048))
    repo = job["teacher"]["repo"]
    out_file = Path(job["out"])
    work = Path(job.get("work") or out_file.parent / "vllm_teacher_check_work")
    work.mkdir(parents=True, exist_ok=True)
    stop_file = Path(job["stop_file"]) if job.get("stop_file") else None
    if not vllmcache.wanted(job):
        raise TeacherCheckError("The job needs engine: {\"kind\": \"vllm\", ...} (the same settings the heal would use).")
    kv = str(job.get("kv_cache_dtype") or "auto").lower()
    if kv not in ("auto", "fp8"):
        raise TeacherCheckError("kv_cache_dtype must be auto or fp8.")
    compare_kv = bool(job.get("compare_kv"))
    if compare_kv:
        kv = "auto"                                              # the comparison measures both: exact first, then 8-bit
    job = effective_job(job, kv)                                 # the job's own choice overrides the engine settings, in both directions
    teacher_dir = reshapejob._resolve_source(repo)
    weights = safetensors_io.weights_manifest(teacher_dir)
    try:
        ident = vllmcache.identity(job, repo, teacher_dir)
    except (vllmcache.VllmCacheError, vllmcache.vllmengine.EngineError) as exc:
        raise TeacherCheckError(str(exc)) from None
    # 1. the rows and the reference
    rows, ref, max_seq, ref_source = prepare(job, np, k, n_want, max_seq, teacher_dir, weights, stop_file)
    tokens = sum(len(r["input_ids"]) for r in rows)
    # 2. vLLM on the same rows (once, or twice with compare_kv: the exact cache, then the 8-bit one)
    vin = work / "vllm_in"
    vin.mkdir(parents=True, exist_ok=True)
    distilljob.save_rows(vin, rows)

    def score(kvd: str):
        j = effective_job(job, kvd)
        try:
            ident_v = vllmcache.identity(j, repo, teacher_dir)
        except (vllmcache.VllmCacheError, vllmcache.vllmengine.EngineError) as exc:
            raise TeacherCheckError(str(exc)) from None
        vout = work / ("vllm_out" if not compare_kv else f"vllm_out_{kvd}")
        vout.mkdir(parents=True, exist_ok=True)
        for f in vout.glob("*"):
            f.unlink()
        meta = {"teacher": repo, "k": k, "rows": len(rows), "data_hash": distilljob.data_hash(rows), "tokenizer_hash": "", "max_len": max_seq, "general_key": {}, "general_rows": 0, **ident_v}
        writer = cachewrite.CacheWriter(vout, meta, 262144)
        protocol.log(f"Scoring the rows with vLLM and the {kvd} KV cache.")
        try:
            res = vllmcache.run_chunk({**j, "part_tokens": 262144}, writer, vin, meta, repo, teacher_dir, 0, len(rows), k, max_seq, stop_file)
        except (vllmcache.VllmCacheError, vllmcache.vllmengine.EngineError) as exc:
            raise TeacherCheckError(f"vLLM could not score the rows: {exc}") from None
        writer.close()
        if res.get("stopped") or writer.done_rows < len(rows):
            raise TeacherCheckError("The check was stopped before vLLM had scored all the rows.")
        rv = distilljob.CacheReader(vout)
        got = [rv.get(i) for i in range(len(rows))]
        conf = kv_confirmation(kvd, res)
        if conf["agrees"] is False:
            protocol.log(f"The engine's log says the {conf['seen']} KV cache, but {kvd} was asked for: this measurement is invalid.")
        cmp_ = compare(ref, got, np)
        return res, cmp_, conf, ident_v

    VLLM_KEYS = ("engine", "teacher_quant", "route", "reason", "startup_seconds", "load_seconds", "stage_seconds", "tokens_per_s", "tokens", "load_phases", "timings", "seconds")
    res, cmp_, conf, ident = score(kv)
    v = with_kv(verdict_of(cmp_["overall"]), conf, cmp_["overall"], k)
    report = {"v": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **v, "rows": len(rows), "tokens": tokens, "k": k, "max_seq": max_seq, "metrics": cmp_["overall"],
              "by_position": cmp_["by_position"], "reference": ref_source, "identity": ident, "kv_cache_dtype": kv, "kv_cache_confirmed": conf,
              "vllm": {kk: res.get(kk) for kk in VLLM_KEYS},
              "seconds": round(time.monotonic() - started, 1)}
    if compare_kv:
        res8, cmp8, conf8, ident8 = score("fp8")
        v8 = with_kv(verdict_of(cmp8["overall"]), conf8, cmp8["overall"], k)
        diff = kv_difference(cmp_["overall"], cmp8["overall"])
        both_valid = bool(v["valid"] and v8["valid"])
        report["compare_kv"] = {"auto": {"metrics": cmp_["overall"], "by_position": cmp_["by_position"], "verdict": v["verdict"], "text": v["text"], "kv_cache_confirmed": conf, "vllm": {kk: res.get(kk) for kk in VLLM_KEYS}},
                                "fp8": {"metrics": cmp8["overall"], "by_position": cmp8["by_position"], "verdict": v8["verdict"], "text": v8["text"], "kv_cache_confirmed": conf8, "identity": ident8, "vllm": {kk: res8.get(kk) for kk in VLLM_KEYS}},
                                "difference": diff, "valid": both_valid,
                                "text": ((f"The 8-bit KV cache against the exact one, on the same {len(rows)} rows: top-1 agreement {diff['top1'] * 100:+.2f} points, mean KL {diff['kl']:+.5f}, tail difference {diff['tail_diff_mean']:+.4f}, speed "
                                          f"{res.get('tokens_per_s') or 'n/a'} against {res8.get('tokens_per_s') or 'n/a'} tokens/s (exact against 8-bit).") if both_valid and None not in (diff["top1"], diff["kl"], diff["tail_diff_mean"])
                                         else "The comparison is not valid: at least one of the two runs did not use the KV cache type that was asked for (see the confirmation of each).")}
        report["text"] = report["text"] + " " + report["compare_kv"]["text"]
        report["seconds"] = round(time.monotonic() - started, 1)
    tmp = out_file.with_name(out_file.name + ".partial")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, out_file)
    protocol.result({**{kk: report.get(kk) for kk in ("verdict", "text", "rows", "tokens", "k", "max_seq", "metrics", "by_position", "reference", "identity", "vllm", "thresholds", "seconds", "kv_cache_dtype", "kv_cache_confirmed", "compare_kv", "valid")}, "report": str(out_file)})
    return 0
