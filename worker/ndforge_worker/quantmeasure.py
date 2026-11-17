"""Measuring what each quantization type does to each tensor of YOUR weights, weighted by the importance matrix (the measured fit, docs/MIXED_QUANTS.md "Fit to a size (measured)").

The metric, the same for every engine. The importance matrix holds, for every weight tensor, the sum over the calibration tokens of the squared input of each input channel
(`in_sum2`); divided by the number of tokens N this is a[e][c] = how much energy arrives on input channel c (of expert e: the probability of being routed there times the mean squared
input when routed). For a tensor t and a type T the error is

    D_t(T) = (R / n) * sum over the n sampled rows r of sum over channels c of  a[e(r)][c] * (w[r][c] - w_hat[r][c])^2

with R the number of rows of the whole tensor and w_hat the row after quantizing to T and decoding. `N_t = (R / n) * sum a * w^2` (the same sum for the weights themselves) is kept too,
and the unweighted squared error. The figure reported for a plan is sqrt(sum D / sum N).

Four caveats are part of the method and are repeated wherever a plan is shown: (1) the sum is a DIAGONAL approximation: it ignores that input channels are correlated; (2) equal output
error is treated as equally harmful in every tensor and every layer: nothing runs the model; (3) the sample is about 1 to 3 percent of the rows; (4) with the `reference` engine the rounding is
llama.cpp's reference rounding WITHOUT the importance matrix, which is not the rounding the imatrix export does, so the weighted error is overstated and the ranking is only an aid. The judge is
always Compare quants (KL divergence), never this number.

Engine `reference` (this module): numpy encoders and decoders (kquants.py, quantblocks.py). Engine `llama` (measure_llama, further down) runs the user's own llama-quantize on a small sample file."""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from . import fsutil, ggufinfo, ggufrows, mixedquant, quantblocks

MEASURE_VERSION = 1
RUNGS = ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0")
DEFAULT_SAMPLE_ROWS = 128
CAVEATS = ("The error is measured with a diagonal approximation of the importance matrix (it ignores how input channels move together). Every tensor and layer is treated as equally harmful for the "
           "same output error; nothing runs the model. Only about 1 to 3 percent of the rows of each tensor were looked at. With the reference engine the rounding is not the rounding of an "
           "importance-matrix export. This is a ranking aid; Compare quants (KL divergence) is the judge.")
FINGERPRINT_PROBES = 16
PROBE_BYTES = 1 << 20


class MeasureError(ValueError):
    """The measurement cannot be done; the text is plain language (a bad matrix, a bad file)."""


class Stopped(Exception):
    """The STOP file appeared; what was measured so far is in the cache and a new run resumes."""


# ------------------------------------------------------------------------------------------------------------------ fingerprints and the cache key

def fingerprint_file(path) -> str:
    """sha256 of the whole file (the importance matrix: small)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_model(path, info: dict | None = None) -> str:
    """Content-based, quick, mtime ignored: sha256 of the file size, of the header bytes [0:data_start] and of 1 MiB read at 16 evenly spaced offsets."""
    info = info or ggufinfo.read(path)
    size = Path(path).stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as fh:
        h.update(hashlib.sha256(fh.read(info["data_start"])).digest())
        span = max(size - PROBE_BYTES, 0)
        for i in range(FINGERPRINT_PROBES):
            fh.seek(i * span // (FINGERPRINT_PROBES - 1))
            h.update(hashlib.sha256(fh.read(PROBE_BYTES)).digest())
    return h.hexdigest()


def cache_key(model_fp: str, matrix_fp: str, sample_rows: int, seed: int, engine: str, extra: str = "") -> str:
    """16 hex characters naming the cache folder: everything that changes the numbers is in it (`extra`: for the llama engine the llama.cpp version line)."""
    raw = json.dumps([model_fp, matrix_fp or "none", int(sample_rows), int(seed), engine, MEASURE_VERSION, extra], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------------------------ importance from the matrix

def importance(matrix_path, info: dict) -> tuple:
    """({tensor name: float64 array (experts, cols)}, notes) for every sampled tensor of `info` (a 16-bit GGUF's header). `matrix_path` None gives a = 1 everywhere with a note.
    The matrix is a GGUF with `<tensor>.in_sum2` (1-D `[n_in]`, or `[n_in, n_expert]`: both are read the same way, group-major) and `<tensor>.counts`. The divisor N is the largest count among the
    tensors with a single count (the dense ones: all see every token), else the largest count at all. A tensor whose number of statistics is not cols x experts stops the job (the matrix is
    not of this model); a sampled tensor the matrix does not have gets a constant vector, the median of the mean importance of the tensors it has (counted in a note)."""
    notes = []
    want = {t["name"]: t for t in info["tensors"] if ggufrows.is_sampled(t)}
    if matrix_path is None:
        notes.append("No importance matrix was given, so every input channel counts the same (a plain weight error). Choose a matrix: with one the plan follows what the model actually reads.")
        return {n: np.ones((ggufrows.geometry(t)[2], t["dims"][0]), np.float64) for n, t in want.items()}, notes
    p = Path(matrix_path)
    if p.suffix.lower() == ".dat":
        raise MeasureError(f"{p.name} is a legacy .dat matrix. The measured fit needs a GGUF matrix (imatrix.gguf); make one with the importance-matrix step or use one from the library.")
    try:
        minfo = ggufinfo.read(p)
    except (OSError, ValueError) as exc:
        raise MeasureError(f"The matrix {p.name} could not be read as a GGUF importance matrix ({exc}). A legacy .dat file cannot be used here.")
    by = {t["name"]: t for t in minfo["tensors"]}

    def floats(t):
        if t["type"] != 0:
            raise MeasureError(f"The matrix {p.name} holds its statistics in a form this program does not read (not float32).")
        n = ggufinfo.numel(t)
        a = np.fromfile(p, dtype="<f4", count=n, offset=minfo["data_start"] + t["offset"])
        if a.size != n:
            raise MeasureError(f"The matrix {p.name} is cut short.")
        return a.astype(np.float64)

    dense_counts, all_counts = [], []
    for name, t in by.items():
        if name.endswith(".counts"):
            c = floats(t)
            all_counts.append(float(c.max()) if c.size else 0.0)
            if c.size == 1:
                dense_counts.append(float(c[0]))
    n_tokens = max(dense_counts) if dense_counts else (max(all_counts) if all_counts else 0.0)
    if n_tokens <= 0:
        raise MeasureError(f"The matrix {p.name} has no counts, so it cannot be turned into an importance. Make it again, or use another matrix.")
    out, missing = {}, []
    for name, t in want.items():
        s = by.get(name + ".in_sum2")
        if s is None:
            missing.append(name)
            continue
        cols, _rows, experts = ggufrows.geometry(t)
        if ggufinfo.numel(s) != cols * experts:
            raise MeasureError(f"The matrix does not fit this model: {name} has {cols} input channels x {experts} expert(s) but the matrix holds {ggufinfo.numel(s)} numbers for it. "
                               "This matrix is not of this model; adapt it first (Importance matrix page, Adapt).")
        out[name] = (floats(s) / n_tokens).reshape(experts, cols)
    if not out:
        raise MeasureError(f"None of the tensors of the model is in the matrix {p.name}. This matrix is not of this model; adapt it first (Importance matrix page, Adapt).")
    if missing:
        med = statistics.median(float(v.mean()) for v in out.values())
        for name in missing:
            cols, _rows, experts = ggufrows.geometry(want[name])
            out[name] = np.full((experts, cols), med, np.float64)
        notes.append(f"{len(missing)} tensors are not in the matrix (for example {missing[0]}); they were given the median importance of the others, so the plan treats them like an average tensor.")
    return out, notes


# ------------------------------------------------------------------------------------------------------------------ measuring one tensor

def tensor_record(t: dict, pick: dict, rows: np.ndarray, a: np.ndarray, rungs=RUNGS, encode=None) -> dict:
    """The record of one tensor: {"rows", "cols", "experts", "sampled", "norm2_w", "types": {rung: {"eff", "bytes", "err", "err_plain"}}}. `rows` float32 `[n, cols]` are the sampled rows
    (expert-major, as ggufrows.flat_rows), `a` float64 `(experts, cols)`. `encode(x, type)` returns the bytes of the rows in that type (default: quantblocks.encode)."""
    encode = encode or quantblocks.encode
    cols, per, experts = pick["cols"], pick["rows_per_expert"], pick["experts"]
    n = rows.shape[0]
    total_rows = per * experts
    scale = total_rows / n
    k = len(pick["rows"])
    arow = np.repeat(a, k, axis=0)        # one expert's importance per sampled row (rows are expert-major)
    w64 = rows.astype(np.float64)
    rec = {"rows": total_rows, "cols": cols, "experts": experts, "sampled": n, "norm2_w": float(scale * np.sum(arow * w64 * w64)), "types": {}}
    done = {}
    numel = cols * total_rows
    for rung in rungs:
        eff = mixedquant.effective_type(rung, cols)
        if eff not in done:
            raw = encode(rows, eff)
            dec = quantblocks.decode(raw, eff, n, cols)
            diff = w64 - dec.astype(np.float64)
            sq = diff * diff
            done[eff] = (float(scale * np.sum(arow * sq)), float(scale * np.sum(sq)))
        err, plain = done[eff]
        rec["types"][rung] = {"eff": eff, "bytes": int(numel * ggufinfo.BPW[eff] / 8), "err": err, "err_plain": plain}
    return rec


# ------------------------------------------------------------------------------------------------------------------ the cache

def _load_cache(cache_dir: Path, key: str) -> dict:
    p = Path(cache_dir) / "measure.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("version") == MEASURE_VERSION and d.get("key") == key and isinstance(d.get("tensors"), dict):
            return d
    except (OSError, ValueError):
        pass
    return {"version": MEASURE_VERSION, "key": key, "tensors": {}, "skipped": [], "notes": [], "done": False}


def save_cache(cache_dir, data: dict) -> None:
    """measure.json, written under a temporary name and moved into place (a stop or a crash never leaves half a file)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / "measure.json.tmp"
    tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    fsutil.replace_retry(tmp, cache_dir / "measure.json")


def default_threads() -> int:
    return max(1, min((os.cpu_count() or 2) - 1, 16))


def measure(src: dict, imp: dict, opts: dict, cache_dir, should_stop=lambda: False, progress=lambda done, total, name: None) -> dict:
    """Measures every sampled tensor with the numpy engine. `opts`: {"rungs", "sample_rows", "seed", "threads", "key" (the cache key; else made from the model file alone)}. Resumes from
    `cache_dir/measure.json` when its key matches; writes it after every tensor; raises Stopped when `should_stop()` turns true (after the tensors in flight finish). Returns the measures:
    {"version", "key", "engine": "reference", "sample_rows", "seed", "rungs", "tensors": {name: record}, "skipped": [names], "notes", "done": True}."""
    rungs = tuple(opts.get("rungs") or RUNGS)
    n_rows = int(opts.get("sample_rows") or DEFAULT_SAMPLE_ROWS)
    seed = int(opts.get("seed") or 0)
    key = opts.get("key") or cache_key(fingerprint_model(src["path"], src), "none", n_rows, seed, "reference")
    picks = ggufrows.sample_plan(src, n_rows, seed)
    by = {t["name"]: t for t in src["tensors"]}
    data = _load_cache(cache_dir, key)
    data.update({"engine": "reference", "sample_rows": n_rows, "seed": seed, "rungs": list(rungs), "done": False})
    todo = [n for n in sorted(picks) if n not in data["tensors"] and n not in data["skipped"]]
    total = len(picks)
    lock = threading.Lock()

    def work(name):
        if should_stop():
            return name, None
        pick = picks[name]
        rows = ggufrows.read_rows(src, by[name], ggufrows.flat_rows(pick))
        if not np.isfinite(rows).all():
            return name, "skipped"
        return name, tensor_record(by[name], pick, rows, imp[name], rungs)

    threads = max(1, int(opts.get("threads") or default_threads()))
    done_count = len(data["tensors"]) + len(data["skipped"])
    stopped = False
    it = iter(todo)
    pending = set()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        def fill():
            while len(pending) < threads * 2 and not should_stop():
                try:
                    pending.add(pool.submit(work, next(it)))
                except StopIteration:
                    return
        fill()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                pending.discard(f)
                name, rec = f.result()
                if rec is None:
                    stopped = True
                    continue
                with lock:
                    if rec == "skipped":
                        data["skipped"].append(name)
                        data["notes"].append(f"{name} holds numbers that are not finite (NaN or infinity) in the sampled rows; it was left out of the measurement and is kept at a high type.")
                    else:
                        data["tensors"][name] = rec
                    done_count += 1
                    save_cache(cache_dir, data)
                progress(done_count, total, name)
            if should_stop():
                stopped = True
            fill()
    if stopped or should_stop():
        save_cache(cache_dir, data)
        raise Stopped()
    data["done"] = True
    data["notes"] = sorted(set(data["notes"]))
    save_cache(cache_dir, data)
    return data


# ------------------------------------------------------------------------------------------------------------------ speed probe

_SPEED: dict = {}


def speed_probe(types=("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"), weights: int = 1 << 16) -> dict:
    """{type: weights encoded per second on one thread}: 64k random weights of each K type, once per process (the estimate of the minutes uses it)."""
    x = (np.random.default_rng(7).laplace(size=(weights // 256, 256)) * 0.02).astype(np.float32)
    for t in types:
        if t not in _SPEED:
            t0 = time.perf_counter()
            quantblocks.encode(x, t)
            _SPEED[t] = weights / max(time.perf_counter() - t0, 1e-6)
    return {t: _SPEED[t] for t in types}
