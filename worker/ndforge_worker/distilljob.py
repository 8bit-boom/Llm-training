"""Healing a thinned model by distillation ("Slim or grow a model" version 2).

    distill_cache   the original model (teacher) writes its top-k next-token probabilities for every token of the training rows to disk
    distill         the thinned model (student, LoRA) learns to give the same probabilities: KL(teacher || student) per token
    distill_merge   the LoRA is added into the student's weights (numpy, no torch) -> a new base model folder

The loss follows NVIDIA's Minitron recipe (KL on the logits, temperature 1, alpha 1.0 = no cross-entropy term) with the teacher's distribution cut to its
top-k entries plus one "everything else" bucket (Model-Optimizer's logit_kl_top_k idea). Full logits are never held in memory: the vocabulary of
Qwen3.5 has 248,320 entries, 2 GB per 2048-token row in float32, so the loss is computed in chunks of tokens that are recomputed in the backward pass.
torch and numpy are imported inside the functions that need them.
"""
from __future__ import annotations

import math

TINY = 1e-10


# ------------------------------------------------------------------------------------------------------- the math (torch)

def topk_from_logits(logits, k: int):
    """Teacher side. logits [n, V] -> (ids int64 [n, k], logp float32 [n, k] = log-probabilities of the k most likely tokens, resid float32 [n] = the
    probability left over for all other tokens)."""
    import torch
    lp = torch.log_softmax(logits.float(), dim=-1)
    top, ids = lp.topk(min(k, lp.shape[-1]), dim=-1)
    resid = (1.0 - top.exp().sum(-1)).clamp(min=0.0)
    return ids, top, resid


def kd_topk_loss(student_logp, ids, logp, resid, mask):
    """Student side, for a chunk of tokens: student_logp [n, V] log-probabilities, the teacher's (ids, logp, resid) as made by topk_from_logits, mask [n]
    (1 for real tokens). KL = sum_k p_T,k (log p_T,k - log p_S,k) + r_T (log r_T - log r_S) with r = the probability of everything outside the top-k.
    Returns (sum of the per-token KL over the real tokens, number of real tokens)."""
    import torch
    ls = student_logp.float().gather(-1, ids)
    lt = logp.float()
    main = (lt.exp() * (lt - ls)).sum(-1)
    r_t = resid.float()
    r_s = (1.0 - ls.exp().sum(-1)).clamp(min=TINY)
    rest = torch.where(r_t > 0, r_t * (torch.log(r_t.clamp(min=TINY)) - torch.log(r_s)), torch.zeros_like(r_t))
    m = mask.float()
    return ((main + rest) * m).sum(), m.sum()


def _chunk_losses(h, weight, ids, logp, resid, mask, labels, label_mask):
    import torch
    import torch.nn.functional as F
    logits = (h @ weight.t()).float()
    kd, _ = kd_topk_loss(torch.log_softmax(logits, dim=-1), ids, logp, resid, mask)
    if labels is None:
        return kd, kd.new_zeros(())
    ce = F.cross_entropy(logits, labels.clamp(min=0), reduction="none")
    return kd, (ce * label_mask.float()).sum()


def chunked_kd_loss(hidden, weight, targets, mask, chunk: int = 256, labels=None, label_mask=None):
    """hidden [n, C] (the student's last hidden states), weight [V, C] (its output layer), targets = the teacher's (ids [n, k], logp [n, k], resid [n]),
    mask [n]. The logits are made `chunk` tokens at a time inside torch.utils.checkpoint, so at most chunk x V floats exist (the backward pass makes them again).
    With labels [n] (the next token, any value where label_mask is 0) the cross-entropy over label_mask is returned as well.
    Returns (kd_sum, n_tokens, ce_sum, n_label_tokens); divide to get means."""
    import torch
    from torch.utils.checkpoint import checkpoint
    ids, logp, resid = targets
    n = hidden.shape[0]
    kd_total, ce_total = hidden.new_zeros((), dtype=torch.float32), hidden.new_zeros((), dtype=torch.float32)
    for a in range(0, n, max(1, int(chunk))):
        b = min(n, a + chunk)
        args = (hidden[a:b], weight, ids[a:b], logp[a:b], resid[a:b], mask[a:b], None if labels is None else labels[a:b], None if label_mask is None else label_mask[a:b])
        kd, ce = checkpoint(_chunk_losses, *args, use_reentrant=False)
        kd_total = kd_total + kd
        ce_total = ce_total + ce
    n_label = label_mask.float().sum() if label_mask is not None else hidden.new_zeros(())
    return kd_total, mask.float().sum(), ce_total, n_label


def forward_hidden(model, input_ids, attention_mask):
    """(hidden [b, t, C], the model's output): the hidden states after the final norm of every position, taken with a hook on the final norm. The model is
    called with logits_to_keep=1, so only the last position goes through the (huge) output layer; the training loss makes the other logits in chunks."""
    base = model
    for name in ("get_decoder",):
        if hasattr(base, name):
            try:
                base = getattr(base, name)()
            except Exception:       # noqa: BLE001 - some wrappers have a get_decoder that needs arguments
                base = model
    norm = getattr(base, "norm", None) or getattr(getattr(model, "model", None), "norm", None)
    if norm is None:
        raise ValueError("The final normalisation layer of this model was not found.")
    held: dict = {}
    handle = norm.register_forward_hook(lambda m, a, o: held.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    try:
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=1)
        except TypeError:                                  # [A] an older transformers without logits_to_keep: full logits are made once (memory!)
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return held["h"], out


# ------------------------------------------------------------------------------------------------------- numpy twin (tests)

def kd_topk_reference(student_logits, teacher_logits, k: int) -> float:
    """The same loss in plain float64 numpy, one row [n, V] of logits each: the mean KL per token of the teacher's top-k distribution (plus its bucket)
    against the student's. Used by the tests as an independent check of kd_topk_loss."""
    import numpy as np
    t = np.asarray(teacher_logits, dtype=np.float64)
    s = np.asarray(student_logits, dtype=np.float64)
    lt = t - t.max(-1, keepdims=True)
    lt = lt - np.log(np.exp(lt).sum(-1, keepdims=True))
    ls = s - s.max(-1, keepdims=True)
    ls = ls - np.log(np.exp(ls).sum(-1, keepdims=True))
    total = 0.0
    for i in range(t.shape[0]):
        top = np.argsort(-lt[i], kind="stable")[:k]
        pt = np.exp(lt[i][top])
        ps = np.exp(ls[i][top])
        r_t, r_s = max(1.0 - pt.sum(), 0.0), max(1.0 - ps.sum(), TINY)
        kl = float((pt * (lt[i][top] - ls[i][top])).sum())
        if r_t > 0:
            kl += r_t * (math.log(max(r_t, TINY)) - math.log(r_s))
        total += kl
    return total / t.shape[0]


# ============================================================================================================ the data

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from . import protocol, safetensors_io  # noqa: E402
from .reshape import ReshapeError  # noqa: E402

CACHE_FORMAT = 1
DEFAULT_GENERAL = "nvidia/Nemotron-Pretraining-SFT-v1"      # NVIDIA's recommendation for distillation data; [A] the subset and licence are checked on the owner's PC
MIN_GENERAL_ROW = 16                                         # windows shorter than this many tokens are dropped


def tokenizer_hash(folder) -> str:
    """sha256 over the tokenizer files of a model folder: the teacher and the student must see exactly the same token ids"""
    h = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model", "added_tokens.json"):
        f = Path(folder) / name
        if f.is_file():
            h.update(name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def data_hash(rows: list) -> str:
    """sha256 over the token ids (and lengths) of the rows, in order"""
    h = hashlib.sha256()
    for r in rows:
        ids = r["input_ids"]
        h.update(len(ids).to_bytes(4, "little"))
        h.update(b"".join(int(x).to_bytes(4, "little", signed=False) for x in ids))
    return h.hexdigest()


def _text_of(value) -> str:
    """a dataset cell as text: a string, or a list of chat messages joined as plain lines"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in value)
    return "" if value is None else str(value)


def _general_texts(source: dict):
    """An iterator over the general texts: a local JSONL file (source["file"], used by the tests and by people who bring their own text) or a Hugging Face
    dataset streamed from the Hub (source["id"], never downloaded as a whole)."""
    column = source.get("column") or "text"
    if source.get("file"):
        with open(source["file"], encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield _text_of(json.loads(line).get(column))
        return
    from datasets import load_dataset
    ds = load_dataset(source.get("id") or DEFAULT_GENERAL, name=source.get("config"), split=source.get("split") or "train", streaming=True)
    for item in ds:
        yield _text_of(item.get(column))


def general_rows(source: dict, tok_fn, max_seq: int, budget: int, seed: int = 3407):
    """General text cut into rows of at most max_seq tokens (windows of long texts), about `budget` tokens in all. tok_fn(text) -> list of token ids.
    The order is the dataset's own (deterministic); returns (rows, tokens). Raises whatever the source raises (the caller explains it)."""
    rows, tokens = [], 0
    for text in _general_texts(source):
        ids = tok_fn(text)
        for a in range(0, len(ids), max_seq):
            w = ids[a:a + max_seq]
            if len(w) < MIN_GENERAL_ROW:
                continue
            rows.append({"input_ids": w, "completion_mask": [1] * len(w), "kind": "general", "source": "general"})
            tokens += len(w)
            if tokens >= budget:
                return rows, tokens
    return rows, tokens


def build_rows(tokenizer, data_cfg: dict, fam_name: str, max_seq: int, seed: int = 3407):
    """The unique rows the teacher answers and the student learns from: the project's saved training data (as in training) followed by general text.
    data_cfg: {"train": path, "tokens": total heal tokens, "general": {"share": 0.5, "id" | "file", "column", "config", "split"} | None}.
    The general text is cut to its share of the heal budget (project data repeats when it is smaller than its share; see training_order). When the general
    text cannot be read (offline ...), healing goes on with the project's data only and stats["general_error"] says why.
    Returns (rows, stats)."""
    from . import data
    proj, pstats = data.load_training_rows(tokenizer, data_cfg["train"], fam_name, max_seq, seed)
    for r in proj:
        r["source"] = "project"
    stats = {"project_rows": len(proj), "project_tokens": pstats["tokens"], "general_rows": 0, "general_tokens": 0, "general_error": None, "general_source": None}
    g = data_cfg.get("general")
    rows = list(proj)
    if g and float(g.get("share", 0)) > 0:
        total = int(data_cfg.get("tokens") or 0)
        budget = int(total * float(g["share"]))
        name = g.get("file") or g.get("id") or DEFAULT_GENERAL
        stats["general_source"] = name
        inner = data._inner(tokenizer)
        protocol.progress("data", 0.0, f"Getting the general text ({name})")
        try:
            grows, gtok = general_rows(g, lambda t: inner(t, add_special_tokens=False)["input_ids"], max_seq, budget, seed)
        except Exception as exc:      # noqa: BLE001 - offline, gated, wrong id or column: all end the same plain way
            grows, gtok = [], 0
            stats["general_error"] = (f"The general text ({name}) could not be downloaded: {str(exc)[:200]}. "
                                      "Healing goes on with your world's text only; try again later for better results.")
            protocol.log(stats["general_error"])
        rows += grows
        stats.update(general_rows=len(grows), general_tokens=gtok)
        protocol.progress("data", 1.0, "")
    return rows, stats


def training_order(rows: list, tokens: int, general_share: float, seed: int = 3407, block: int = 64) -> list[int]:
    """Row indices (repeated as needed) for a run of about `tokens` tokens: the share `general_share` of them from the rows with source "general", the rest
    from "project" rows, each group cycled through in a fresh shuffled order. Locality: the order is shuffled in blocks of `block` consecutive rows, so
    that the cached teacher answers of nearby rows live in the same file. With only one kind of row present all tokens come from that kind."""
    by = {"project": [i for i, r in enumerate(rows) if r.get("source") != "general"], "general": [i for i, r in enumerate(rows) if r.get("source") == "general"]}
    if not by["general"]:
        share = {"project": 1.0, "general": 0.0}
    elif not by["project"]:
        share = {"project": 0.0, "general": 1.0}
    else:
        share = {"project": 1.0 - general_share, "general": general_share}
    rng = random.Random(seed)
    chosen: list[int] = []
    for kind in ("project", "general"):
        idx, want, got = by[kind], int(tokens * share[kind]), 0
        while idx and got < want:
            blocks = [idx[a:a + block] for a in range(0, len(idx), block)]
            rng.shuffle(blocks)
            for b in blocks:
                b = list(b)
                rng.shuffle(b)
                for i in b:
                    if got >= want:
                        break
                    chosen.append(i)
                    got += len(rows[i]["input_ids"])
    # the two kinds are interleaved by block so that every stretch of the run has both
    rng2 = random.Random(seed + 1)
    chunks = [chosen[a:a + block] for a in range(0, len(chosen), block)]
    rng2.shuffle(chunks)
    return [i for c in chunks for i in c]


# ============================================================================================================ the cache

class CacheWriter:
    """Teacher answers on disk: part-00001.npz ... (ids uint32 [n, k], logp float16 [n, k], resid float32 [n], row_offsets int64 [rows + 1]) and index.json.
    Rows are added in order; a part is written whenever it holds `part_tokens` tokens, and index.json (with done_rows) after every part, so a stopped job
    can continue where it was. Needs numpy."""

    def __init__(self, folder, meta: dict, part_tokens: int = 262144):
        from . import slice_io
        self.np = slice_io.np()
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {"format": CACHE_FORMAT, **meta}
        self.part_tokens = part_tokens
        old = read_cache_index(self.dir)
        self.parts: list = list(old["parts"]) if old and cache_matches(old, meta) else []
        if not self.parts:
            for f in self.dir.glob("part-*.npz"):
                f.unlink()
        self.done_rows = sum(p["rows"] for p in self.parts)
        self.tokens = sum(p["tokens"] for p in self.parts)
        self._buf: list = []
        self._buf_tokens = 0

    def add_row(self, ids, logp, resid) -> None:
        np = self.np
        self._buf.append((np.asarray(ids, dtype=np.uint32), np.asarray(logp, dtype=np.float16), np.asarray(resid, dtype=np.float32)))
        self._buf_tokens += len(self._buf[-1][2])
        if self._buf_tokens >= self.part_tokens:
            self.flush()

    def flush(self) -> None:
        np = self.np
        if not self._buf:
            return
        n = len(self.parts) + 1
        name = f"part-{n:05d}.npz"
        offsets = np.cumsum([0] + [len(b[2]) for b in self._buf]).astype(np.int64)
        tmp = self.dir / f"part-{n:05d}.partial.npz"
        np.savez(tmp, ids=np.concatenate([b[0] for b in self._buf]), logp=np.concatenate([b[1] for b in self._buf]),
                 resid=np.concatenate([b[2] for b in self._buf]), row_offsets=offsets)
        os.replace(tmp, self.dir / name)
        self.parts.append({"file": name, "first_row": self.done_rows, "rows": len(self._buf), "tokens": int(offsets[-1])})
        self.done_rows += len(self._buf)
        self.tokens += int(offsets[-1])
        self._buf, self._buf_tokens = [], 0
        self._write_index()

    def _write_index(self) -> None:
        idx = {**self.meta, "done_rows": self.done_rows, "tokens": self.tokens, "parts": self.parts, "complete": self.done_rows >= self.meta["rows"],
               "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        tmp = self.dir / "index.partial.json"
        tmp.write_text(json.dumps(idx), encoding="utf-8")
        os.replace(tmp, self.dir / "index.json")

    def close(self) -> None:
        self.flush()
        if not self.parts:
            self._write_index()


def read_cache_index(folder):
    try:
        idx = json.loads((Path(folder) / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return idx if isinstance(idx, dict) and idx.get("format") == CACHE_FORMAT else None


def cache_matches(index: dict, expect: dict) -> bool:
    """True when the saved teacher answers belong to this data, tokenizer, length, k and teacher"""
    return all(index.get(k) == expect.get(k) for k in ("teacher", "k", "rows", "data_hash", "tokenizer_hash", "max_len"))


class CacheReader:
    """Reads the rows of a cache back (two parts are kept in memory). get(i) -> (ids [n, k] int64, logp [n, k] float16, resid [n] float32)."""

    def __init__(self, folder):
        from . import slice_io
        self.np = slice_io.np()
        self.dir = Path(folder)
        self.index = read_cache_index(folder)
        if not self.index:
            raise ReshapeError("The teacher's saved answers are missing or unreadable; they will be made again.")
        self.parts = self.index["parts"]
        self._loaded: dict = {}

    @property
    def done_rows(self) -> int:
        return int(self.index["done_rows"])

    def _part(self, p: int):
        if p not in self._loaded:
            if len(self._loaded) >= 2:
                self._loaded.pop(next(iter(self._loaded)))
            with self.np.load(self.dir / self.parts[p]["file"]) as z:
                self._loaded[p] = {k: z[k] for k in z.files}
        return self._loaded[p]

    def get(self, i: int):
        for p, meta in enumerate(self.parts):
            if meta["first_row"] <= i < meta["first_row"] + meta["rows"]:
                z = self._part(p)
                a, b = int(z["row_offsets"][i - meta["first_row"]]), int(z["row_offsets"][i - meta["first_row"] + 1])
                return z["ids"][a:b].astype(self.np.int64), z["logp"][a:b], z["resid"][a:b]
        raise ReshapeError(f"The teacher's saved answers do not contain row {i}.")


def _teacher_answers(model, batch_rows: list, k: int, chunk: int, pad_id: int):
    """[(ids [n, k], logp [n, k], resid [n]) as numpy] for the rows of a batch: the teacher's top-k next-token answers at every position"""
    import torch
    n = max(len(r["input_ids"]) for r in batch_rows)
    dev = next(model.parameters()).device
    ids = torch.full((len(batch_rows), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch_rows), n), dtype=torch.long)
    for i, r in enumerate(batch_rows):
        ids[i, :len(r["input_ids"])] = torch.tensor(r["input_ids"])
        mask[i, :len(r["input_ids"])] = 1
    out = []
    with torch.no_grad():
        hidden, _ = forward_hidden(model, ids.to(dev), mask.to(dev))
        weight = model.get_output_embeddings().weight
        for i, r in enumerate(batch_rows):
            h = hidden[i, :len(r["input_ids"])]
            parts = []
            for a in range(0, h.shape[0], chunk):
                parts.append([t.cpu() for t in topk_from_logits(h[a:a + chunk] @ weight.t(), k)])
            out.append(tuple(torch.cat([p[j] for p in parts]).numpy() for j in range(3)))
    return out


def run_cache(cfg: dict) -> int:
    """job distill_cache: {"teacher": {"repo", "load": "auto|4bit|16bit|offload"}, "student": folder of the thinned model (its tokenizer is used),
    "model": {"family"}, "data": {"train", "tokens", "general": {...}}, "k": 32, "max_seq": 2048, "batch": 4, "chunk": 256, "part_tokens": 262144,
    "out": cache folder, "stop_file": path}. Saves the teacher's top-k answers for every row (resumable). Result: rows, tokens, bytes, reused, stopped, notes."""
    protocol.reserve_stdout()
    started = time.monotonic()
    from . import reshapejob, slice_io
    slice_io.np()
    out = Path(cfg["out"])
    stop_file = Path(cfg.get("stop_file") or out.parent / "STOP")
    k, max_seq = int(cfg.get("k", 32)), int(cfg.get("max_seq", 2048))
    student = Path(cfg["student"])
    teacher_dir = reshapejob._resolve_source(cfg["teacher"]["repo"])
    if tokenizer_hash(teacher_dir) != tokenizer_hash(student):
        raise ReshapeError("The original model and the thinned model do not use the same tokenizer, so the original's answers cannot be used to teach it.")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(student))
    rows, stats = build_rows(tok, cfg["data"], cfg["model"].get("family", "generic"), max_seq, int(cfg.get("seed", 3407)))
    meta = {"teacher": cfg["teacher"]["repo"], "k": k, "rows": len(rows), "data_hash": data_hash(rows), "tokenizer_hash": tokenizer_hash(student), "max_len": max_seq,
            "project_rows": stats["project_rows"], "general_rows": stats["general_rows"]}
    writer = CacheWriter(out, meta, int(cfg.get("part_tokens", 262144)))
    notes = [stats["general_error"]] if stats["general_error"] else []
    total_tokens = sum(len(r["input_ids"]) for r in rows)
    result = {"rows": len(rows), "tokens": total_tokens, "k": k, "dir": str(out), "stats": stats, "notes": notes, "stopped": False, "reused": False}
    if writer.done_rows >= len(rows):
        protocol.log("The teacher's saved answers are already complete for this data.")
        result.update(reused=True, bytes=_dir_bytes(out), seconds=round(time.monotonic() - started, 1))
        protocol.result(result)
        return 0
    need = (total_tokens - writer.tokens) * (k * 6 + 4)
    free = shutil.disk_usage(out)[2]
    if free < need * 1.05:
        protocol.error("disk_full", f"The teacher's answers need {need / 1e9:.1f} GB in {out}; {free / 1e9:.1f} GB free.")
        return 1
    est = sum(r.nbytes for r in safetensors_io.list_tensors(teacher_dir).values())
    mode = reshapejob._choose_load(cfg["teacher"].get("load", "auto"), est)
    protocol.progress("cache", 0.0, f"Loading the original model ({mode})")
    model = reshapejob._load(teacher_dir, mode)
    inner = getattr(tok, "tokenizer", tok)
    pad_id = inner.pad_token_id if inner.pad_token_id is not None else (inner.eos_token_id or 0)
    batch, chunk = max(1, int(cfg.get("batch", 4))), int(cfg.get("chunk", 256))
    try:
        i = writer.done_rows
        if i:
            protocol.log(f"Continuing the teacher's answers after row {i} of {len(rows)}.")
        while i < len(rows):
            if stop_file.exists():
                result["stopped"] = True
                break
            part = rows[i:i + batch]
            try:
                answers = _teacher_answers(model, part, k, chunk, pad_id)
            except RuntimeError as exc:
                if not reshapejob._is_oom(exc) or len(part) == 1:
                    raise
                protocol.log("Out of memory; the original model answers one text at a time.")
                reshapejob._free()
                batch = 1
                continue
            for a in answers:
                writer.add_row(*a)
            i += len(part)
            protocol.progress("cache", i / len(rows), f"The original model answers your texts: {i} of {len(rows)}")
    finally:
        writer.close()
        del model
        reshapejob._free()
    result.update(bytes=_dir_bytes(out), seconds=round(time.monotonic() - started, 1), done_rows=writer.done_rows)
    protocol.result(result)
    return 0


def _dir_bytes(folder) -> int:
    return sum(f.stat().st_size for f in Path(folder).glob("*") if f.is_file())
