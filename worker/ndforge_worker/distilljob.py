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

from . import generaltext, protocol, safetensors_io  # noqa: E402
from .reshape import ReshapeError  # noqa: E402

CACHE_FORMAT = 1
from .generaltext import DEFAULT_ID as DEFAULT_GENERAL      # mlabonne/open-perfectblend: open (Apache-2.0); the gated Nemotron set stays selectable in the app
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
    """An iterator over the general texts: a local file (source["file"]: .jsonl .json .txt .csv .parquet) or a Hugging Face dataset streamed from the Hub
    (source["id"], never downloaded as a whole). Text, chat (messages / conversations) and prompt+completion rows are all read (generaltext.py);
    source["render"] turns chat rows into text with the model's chat template."""
    from . import generaltext
    yield from generaltext.texts(source)


def general_rows(source: dict, tok_fn, max_seq: int, budget: int, seed: int = 3407):
    """General text cut into rows of at most max_seq tokens (windows of long texts), about `budget` tokens in all. tok_fn(text) -> list of token ids.
    The order is the dataset's own, or mixed by source["seed"] when it has one (the same seed gives the same sample); returns (rows, tokens). Raises whatever the source raises (the caller explains it)."""
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
    stats = {"project_rows": len(proj), "project_tokens": pstats["tokens"], "general_rows": 0, "general_tokens": 0, "general_error": None, "general_error_code": None,
             "general_source": None, "general_detected": None, "general_texts_skipped": 0}
    g = data_cfg.get("general")
    rows = list(proj)
    if g and float(g.get("share", 0)) > 0:
        total = int(data_cfg.get("tokens") or 0)
        budget = int(total * float(g["share"]))
        g = {**g}
        if not g.get("file") and not g.get("id"):
            g["id"] = DEFAULT_GENERAL
        name = g.get("file") or g.get("id")
        stats["general_source"] = name
        inner = data._inner(tokenizer)
        report: dict = {}
        g.update(seed=seed, report=report, render=lambda msgs: data.render_chat(tokenizer, msgs, fam_name)[1])
        protocol.progress("data", 0.0, f"Getting the general text ({name})")
        try:
            grows, gtok = general_rows(g, lambda t: inner(t, add_special_tokens=False)["input_ids"], max_seq, budget, seed)
        except Exception as exc:      # noqa: BLE001 - offline, gated, wrong id or column: all end the same plain way
            grows, gtok = [], 0
            stats["general_error"], stats["general_error_code"] = generaltext.failure_note(
                exc, g, "downloaded", "Healing goes on with your world's text only; try again later or choose another dataset.", DEFAULT_GENERAL)
            protocol.log(stats["general_error"])
        stats.update(general_detected=report.get("detected"), general_texts_skipped=report.get("skipped", 0))
        rows += grows
        stats.update(general_rows=len(grows), general_tokens=gtok)
        if grows:
            protocol.log(f"General text used: {name} ({len(grows)} rows, {gtok:,} tokens).")
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


def save_rows(folder, rows: list) -> None:
    """rows.npz next to the cache: the token ids of the rows the teacher answered, so the student learns from exactly these rows without making them again
    (and without a second download of the general text). Needs numpy."""
    from . import slice_io
    np = slice_io.np()
    lens = np.array([len(r["input_ids"]) for r in rows], dtype=np.int64)
    kinds = sorted({str(r.get("kind", "")) for r in rows})
    np.savez(Path(folder) / "rows.partial.npz", ids=np.array([x for r in rows for x in r["input_ids"]], dtype=np.int32),
             mask=np.array([x for r in rows for x in r["completion_mask"]], dtype=np.uint8), offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
             general=np.array([r.get("source") == "general" for r in rows], dtype=np.uint8), kind=np.array([kinds.index(str(r.get("kind", ""))) for r in rows], dtype=np.int16),
             kinds=np.array(kinds))
    os.replace(Path(folder) / "rows.partial.npz", Path(folder) / "rows.npz")


def load_rows(folder) -> list:
    from . import slice_io
    np = slice_io.np()
    try:
        with np.load(Path(folder) / "rows.npz") as z:
            ids, mask, off, gen, kind, kinds = (z[k] for k in ("ids", "mask", "offsets", "general", "kind", "kinds"))
    except (OSError, ValueError, KeyError):
        raise ReshapeError("The teacher's saved answers are missing or unreadable; they will be made again.") from None
    return [{"input_ids": ids[off[i]:off[i + 1]].tolist(), "completion_mask": mask[off[i]:off[i + 1]].tolist(), "kind": str(kinds[kind[i]]),
             "source": "general" if gen[i] else "project"} for i in range(len(off) - 1)]


def read_cache_index(folder):
    try:
        idx = json.loads((Path(folder) / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return idx if isinstance(idx, dict) and idx.get("format") == CACHE_FORMAT else None


def cache_matches(index: dict, expect: dict) -> bool:
    """True when the saved teacher answers belong to this data, tokenizer, length, k and teacher"""
    return all(index.get(k) == expect.get(k) for k in ("teacher", "k", "rows", "data_hash", "tokenizer_hash", "max_len", "general_key", "general_rows"))


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
            "project_rows": stats["project_rows"], "general_rows": stats["general_rows"],
            "general_key": generaltext.general_key((cfg["data"] or {}).get("general"), ((cfg["data"] or {}).get("general") or {}).get("share", 0), int(cfg.get("seed", 3407))),
            "general_source": stats["general_source"]}
    writer = CacheWriter(out, meta, int(cfg.get("part_tokens", 262144)))
    save_rows(out, rows)
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


# ============================================================================================================ the student

DEFAULT_HYPER = {"rank": 64, "alpha": 64, "lr": 1e-4, "max_seq": 2048, "batch": 4, "grad_accum": 8, "warmup": 30, "weight_decay": 0.0, "scheduler": "cosine",
                 "seed": 3407, "save_steps": 50, "logging_steps": 1, "tokens": 20_000_000}


def _distill_collator(pad_id: int, reader):
    """Batches of rows: ids, attention mask, labels (as in training) and, with a cache, the teacher's answers for every real position, in row order."""
    import numpy as np
    import torch

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            t = torch.tensor(b["input_ids"], dtype=torch.long)
            ids[i, :L] = t
            att[i, :L] = 1
            lab[i, :L] = torch.where(torch.tensor(b["completion_mask"], dtype=torch.bool), t, torch.full_like(t, -100))
        out = {"input_ids": ids, "attention_mask": att, "labels": lab}
        if reader is not None:
            parts = [reader.get(int(b["row"])) for b in batch]
            for b, (tid, _, _) in zip(batch, parts):
                if len(tid) != len(b["input_ids"]):
                    raise ReshapeError("The teacher's saved answers do not fit the rows; they will be made again.")
            out["t_ids"] = torch.from_numpy(np.concatenate([p[0] for p in parts]))
            out["t_logp"] = torch.from_numpy(np.concatenate([p[1] for p in parts]).astype("float32"))
            out["t_resid"] = torch.from_numpy(np.concatenate([p[2] for p in parts]))
        return out
    return collate


def _teacher_targets(teacher, ids, att, k: int, chunk: int):
    """the same (ids, logp, resid) for the real positions of a batch, made now by the teacher (online mode)"""
    import torch
    with torch.no_grad():
        hidden, _ = forward_hidden(teacher, ids.to(next(teacher.parameters()).device), att.to(next(teacher.parameters()).device))
        h = hidden[att.bool().to(hidden.device)]
        weight = teacher.get_output_embeddings().weight
        parts = [topk_from_logits(h[a:a + chunk] @ weight.t(), k) for a in range(0, h.shape[0], chunk)]
    return tuple(torch.cat([p[j] for p in parts]).to(ids.device) for j in range(3))


def student_loss(model, inputs, targets_fn, alpha: float, chunk: int):
    """(loss, kd mean, ce mean) of a batch. The student's hidden states of the real positions go through the output layer in chunks; the teacher's answers
    come from targets_fn(inputs) -> (ids, logp, resid) for those positions. alpha = 1: only the KL term."""
    import torch
    att = inputs["attention_mask"]
    hidden, _ = forward_hidden(model, inputs["input_ids"], att)
    sel = att.bool().to(hidden.device)
    h = hidden[sel]
    weight = model.get_output_embeddings().weight
    targets = tuple(t.to(h.device) for t in targets_fn(inputs))
    labels = lab_mask = None
    if alpha < 1.0:
        nxt = torch.cat([inputs["labels"][:, 1:], torch.full_like(inputs["labels"][:, :1], -100)], dim=1).to(h.device)[sel]
        lab_mask, labels = nxt != -100, nxt.clamp(min=0)
    kd, n, ce, n_lab = chunked_kd_loss(h, weight, targets, torch.ones(h.shape[0], device=h.device), chunk, labels, lab_mask)
    kd_mean = kd / n.clamp(min=1)
    ce_mean = ce / n_lab.clamp(min=1) if labels is not None else kd_mean.new_zeros(())
    loss = alpha * kd_mean + (1.0 - alpha) * ce_mean if alpha < 1.0 else kd_mean
    return loss, kd_mean, ce_mean


UNSTABLE = "Healing stopped: the training became unstable (the error went to infinity). Lower the learning rate (Expert) and continue."


def _trainer_class(base):
    class DistillTrainer(base):
        """SFTTrainer with the distillation loss. self.distill = {"targets": fn, "alpha", "chunk", "kd": [...], "ce": [...]}"""

        def _get_train_sampler(self, *args, **kwargs):
            from torch.utils.data import SequentialSampler
            return SequentialSampler(self.train_dataset)       # the order was made by training_order (shuffled in blocks, close rows share a cache file)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch
            d = self.distill
            loss, kd, ce = student_loss(model, inputs, d["targets"], d["alpha"], d["chunk"])
            if not torch.isfinite(loss):
                raise ReshapeError(UNSTABLE)
            d["kd"].append(float(kd))
            d["ce"].append(float(ce))
            d["tokens"] += int(inputs["attention_mask"].sum())
            return (loss, None) if return_outputs else loss
    return DistillTrainer


def calib_loss(model, batches: list, chunk: int = 256):
    """Mean next-token cross-entropy of a model on the measuring batches (the number the thinned model's result card shows as "before healing"),
    computed from the hidden states in chunks, so that no full logits exist. The model is put back into training mode afterwards."""
    import torch
    import torch.nn.functional as F
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        with torch.no_grad():
            dev = next(model.parameters()).device
            for b in batches:
                ids, mask = b["input_ids"].to(dev), b["attention_mask"].to(dev)
                hidden, _ = forward_hidden(model, ids, mask)
                weight = model.get_output_embeddings().weight
                for i in range(ids.shape[0]):
                    n = int(mask[i].sum())
                    if n < 2:
                        continue
                    for a in range(0, n - 1, chunk):
                        c = min(chunk, n - 1 - a)
                        logits = (hidden[i, a:a + c] @ weight.t()).float()
                        total += float(F.cross_entropy(logits, ids[i, a + 1:a + 1 + c], reduction="sum"))
                    count += n - 1
    finally:
        if was_training:
            model.train()
    return total / count if count else None


def _callback_class():
    from .callbacks import LiveCallback

    class HealCallback(LiveCallback):
        """LiveCallback (metrics, stop file) plus the distillation numbers and the calibration loss every `eval_every` steps."""

        def __init__(self, *args, eval_fn=None, eval_every: int = 0, distill=None, **kw):
            super().__init__(*args, **kw)
            self.eval_fn, self.eval_every, self.distill = eval_fn, eval_every, distill
            self.history: list = []          # (step, kd, ce)
            self.calib: list = []            # (step, loss)
            self._seen = 0

        def on_log(self, args, state, control, logs=None, **kw):
            super().on_log(args, state, control, logs=logs, **kw)
            d = self.distill
            if d is None or not logs or "loss" not in logs:
                return
            kd, ce = d["kd"][self._seen:], d["ce"][self._seen:]
            self._seen = len(d["kd"])
            if kd:
                row = (self.offset + state.global_step, sum(kd) / len(kd), sum(ce) / len(ce))
                self.history.append(row)
                protocol.emit("distill_metric", step=row[0], kd=row[1], ce=row[2], tokens=d["tokens"], total_steps=self.total)

        def _evaluate(self, step):
            if self.eval_fn is None:
                return
            try:
                v = self.eval_fn()
            except Exception as exc:         # noqa: BLE001 - the measurement must never stop the healing
                protocol.log(f"The check on your text was skipped: {type(exc).__name__}: {exc}")
                return
            if v is not None:
                self.calib.append((step, v))
                protocol.emit("distill_eval", step=step, calib_loss=v)

        def on_step_end(self, args, state, control, **kw):
            super().on_step_end(args, state, control, **kw)
            if self.eval_every and not self.stopped and state.global_step % self.eval_every == 0:
                self._evaluate(self.offset + state.global_step)
    return HealCallback


def run(cfg: dict) -> int:
    """job distill: {"model": {"repo": the thinned model's folder, "family"}, "method": "lora|qlora", "backend": "hf", "dataset": {"train", "general": {...}},
    "hyper": {rank, alpha, lr, max_seq, batch, grad_accum, warmup, weight_decay, scheduler, seed, save_steps, logging_steps, tokens, max_steps?},
    "distill": {"mode": "cache|online", "cache": folder, "teacher": {"repo", "load"}, "alpha": 1.0, "k": 32, "chunk": 256, "eval_every": 50},
    "eval": {"calibration": {...}, "teacher_loss", "slim_loss"} | None, "out": folder, "resume": bool, "stop_file": path}.
    Writes the LoRA to out/adapter. Result: steps, tokens_seen, kd_first, kd_last, calib_loss_before / _after, teacher_calib_loss, adapter_dir, stopped."""
    protocol.reserve_stdout()
    from . import backends
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    stop_file = Path(cfg.get("stop_file") or out / "STOP")
    stop_file.unlink(missing_ok=True)
    d = cfg["distill"]
    mode = d.get("mode", "cache")
    if mode not in ("cache", "online"):
        raise ReshapeError(f"Unknown way of healing: {mode!r}.")
    if cfg.get("backend", "hf") not in ("hf", None):
        protocol.log("Healing uses the standard engine (Transformers + PEFT): the fused loss of the faster engine was not checked for this loss.")
    backends.prepare("hf")
    import torch
    from datasets import Dataset
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer
    from . import data, reshapejob

    h = {**DEFAULT_HYPER, **(cfg.get("hyper") or {})}
    k, chunk, alpha = int(d.get("k", 32)), int(d.get("chunk", 256)), float(d.get("alpha", 1.0))
    set_seed(h["seed"])
    protocol.log(f"Healing: {mode} mode, precision {backends.precision_name()}, LoRA rank {h['rank']}, learning rate {h['lr']}, KL loss weight {alpha}.")
    student_dir = cfg["model"]["repo"]
    reader = teacher = None
    if mode == "cache":
        reader = CacheReader(d["cache"])
        idx = reader.index
        rows = load_rows(d["cache"])
        if (idx["tokenizer_hash"] != tokenizer_hash(student_dir) or idx["max_len"] != h["max_seq"] or idx["rows"] != len(rows)
                or data_hash(rows) != idx["data_hash"] or reader.done_rows < idx["rows"]):
            raise ReshapeError("The saved teacher answers belong to other data; they will be made again.")
        share = float(((cfg.get("dataset") or {}).get("general") or {}).get("share", 0.0))
        g_cfg = (cfg.get("dataset") or {}).get("general")
        # the rows come from the cache itself, so the training run's seed cannot change them (see generaltext.same_general)
        if not generaltext.same_general(idx.get("general_key"), g_cfg, share):
            raise ReshapeError("The saved teacher answers belong to other data (another general text); they will be made again.")
    else:
        share = float(((cfg.get("dataset") or {}).get("general") or {}).get("share", 0.0))
    lcfg = {"model": cfg["model"], "method": cfg["method"], "hyper": h}
    model, tokenizer, _ = backends.load(lcfg, "hf")
    if mode == "online":
        dcfg = {"train": cfg["dataset"]["train"], "tokens": int(h["tokens"]), "general": (cfg["dataset"] or {}).get("general")}
        rows, stats = build_rows(tokenizer, dcfg, cfg["model"].get("family", "generic"), int(h["max_seq"]), int(h["seed"]))
        if stats["general_error"]:
            share = 0.0
        t_dir = reshapejob._resolve_source(d["teacher"]["repo"])
        if tokenizer_hash(t_dir) != tokenizer_hash(student_dir):
            raise ReshapeError("The original model and the thinned model do not use the same tokenizer, so the original's answers cannot be used to teach it.")
        est = sum(r.nbytes for r in safetensors_io.list_tensors(t_dir).values())
        teacher = reshapejob._load(t_dir, reshapejob._choose_load(d["teacher"].get("load", "auto"), est))
        for p in teacher.parameters():
            p.requires_grad_(False)
    elif not any(r.get("source") == "general" for r in rows):
        share = 0.0
    order = training_order(rows, int(h["tokens"]), share, int(h["seed"]))
    if not order:
        raise ReshapeError("There is nothing to learn from: the training data is empty.")
    ds = Dataset.from_list([{"input_ids": rows[i]["input_ids"], "completion_mask": rows[i]["completion_mask"], "row": i} for i in order])
    per_step = max(1, int(h["batch"]) * int(h["grad_accum"]))
    steps = int(h["max_steps"]) if h.get("max_steps") else max(1, -(-len(order) // per_step))
    tokens_planned = sum(len(rows[i]["input_ids"]) for i in order)
    protocol.log(f"{len(order)} rows, {tokens_planned:,} tokens in {steps} steps ({sum(1 for i in order if rows[i].get('source') == 'general')} rows of general text).")

    inner = data._inner(tokenizer)
    pad_id = inner.pad_token_id if inner.pad_token_id is not None else inner.eos_token_id
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.get_device_capability(0)[0] >= 8
    optim = "adamw_8bit" if cuda else "adamw_torch"
    if optim == "adamw_8bit":
        try:
            import bitsandbytes  # noqa: F401
        except Exception:
            optim = "adamw_torch"
    fields = getattr(SFTConfig, "__dataclass_fields__", {})
    ckdir = out / "checkpoints"
    kwargs = dict(output_dir=str(ckdir), per_device_train_batch_size=int(h["batch"]), gradient_accumulation_steps=int(h["grad_accum"]), learning_rate=float(h["lr"]),
                  warmup_steps=int(h["warmup"]), weight_decay=float(h["weight_decay"]), lr_scheduler_type=h["scheduler"], logging_steps=int(h["logging_steps"]),
                  save_strategy="steps", save_steps=int(h["save_steps"]), save_total_limit=2, optim=optim, seed=int(h["seed"]), report_to="none", disable_tqdm=True,
                  bf16=bf16, fp16=cuda and not bf16, remove_unused_columns=False, gradient_checkpointing=False, dataloader_num_workers=0, use_cpu=not cuda,
                  max_steps=steps)
    kwargs["max_length" if "max_length" in fields else "max_seq_length"] = int(h["max_seq"])
    if "dataset_kwargs" in fields:
        kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
    if "packing" in fields:
        kwargs["packing"] = False
    args = SFTConfig(**kwargs)

    ev = cfg.get("eval") or None
    eval_fn = None
    if ev and ev.get("calibration"):
        batches = reshapejob._calibration(tokenizer, ev["calibration"], cfg["model"].get("family", "generic"))
        eval_fn = lambda: calib_loss(model, batches, chunk)  # noqa: E731
    state = {"kd": [], "ce": [], "tokens": 0}
    Heal = _callback_class()
    live = Heal(str(stop_file), None, [], 0, eval_fn=eval_fn, eval_every=int(d.get("eval_every", 0)), distill=state)
    trainer_kw = {"model": model, "args": args, "train_dataset": ds, "data_collator": _distill_collator(pad_id, reader), "callbacks": [live]}
    trainer_kw["processing_class" if "processing_class" in SFTTrainer.__init__.__code__.co_varnames else "tokenizer"] = tokenizer
    trainer = _trainer_class(SFTTrainer)(**trainer_kw)
    trainer.model_accepts_loss_kwargs = False           # the trainer then divides the loss by the gradient accumulation itself
    targets = (lambda inp: (inp["t_ids"], inp["t_logp"], inp["t_resid"])) if mode == "cache" else (lambda inp: _teacher_targets(teacher, inp["input_ids"], inp["attention_mask"], k, chunk))
    trainer.distill = {**state, "targets": targets, "alpha": alpha, "chunk": chunk}
    live.distill = trainer.distill
    before = ev.get("slim_loss") if ev else None
    if before is None and eval_fn is not None:
        before = eval_fn()
        if before is not None:
            protocol.emit("distill_eval", step=0, calib_loss=before)
    ckpts = sorted(ckdir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if ckdir.exists() else []
    resume = bool(cfg.get("resume")) and bool(ckpts)
    if cfg.get("resume") and not ckpts:
        protocol.log("No checkpoint to continue from; starting from the beginning.")
    t0 = time.time()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=True if resume else None)
    after = None
    if eval_fn is not None and not live.stopped:
        after = eval_fn()
        if after is not None:
            protocol.emit("distill_eval", step=trainer.state.global_step, calib_loss=after)
    target = out / "adapter"
    trainer.model.save_pretrained(str(target))
    hist = live.history
    gen_info = {k: stats.get(k) for k in ("general_source", "general_rows", "general_tokens", "general_error", "general_error_code", "general_detected")} if mode == "online" else \
        {"general_source": idx.get("general_source"), "general_rows": idx.get("general_rows")}
    summary = {"general": gen_info, "backend": "hf", "method": cfg["method"], "mode": mode, "steps": trainer.state.global_step, "total_steps": steps, "tokens_seen": trainer.distill["tokens"],
               "kd_first": hist[0][1] if hist else None, "kd_last": hist[-1][1] if hist else None, "calib_loss_before": before,
               "calib_loss_after": after if after is not None else (live.calib[-1][1] if live.calib else None), "teacher_calib_loss": ev.get("teacher_loss") if ev else None,
               "train_seconds": round(time.time() - t0, 1), "peak_vram_mb": live._vram(), "adapter_dir": str(target), "stopped": live.stopped,
               "rows": len(order), "tokens_planned": tokens_planned}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    protocol.result(summary)
    return 0


# ============================================================================================================ the merge

def merge(cfg: dict) -> int:
    """job distill_merge: {"base": the thinned model's folder, "adapter": the LoRA folder, "out": ".../healed.partial", "final": ".../healed",
    "summary": the distill job's result (optional, recorded), "stop_file"}. Adds the LoRA into the weights (numpy, no torch, one tensor at a time) and writes
    a new, self-contained base model folder. Result: model_dir, merged, bytes_written."""
    protocol.reserve_stdout()
    started = time.monotonic()
    from . import reshapejob, slice_io
    base, adapter = Path(cfg["base"]), Path(cfg["adapter"])
    out, final = Path(cfg["out"]), Path(cfg["final"])
    stop_file = Path(cfg.get("stop_file") or out.parent / "STOP")
    if final.exists():
        raise ReshapeError(f"{final} already exists.")
    tensors = safetensors_io.list_tensors(base)
    need = sum(t.nbytes for t in tensors.values())
    free = shutil.disk_usage(out.parent)[2]
    if free < need * 1.1:
        protocol.error("disk_full", f"There is not enough free space: {need * 1.1 / 1e9:.1f} GB needed in {out.parent}, {free / 1e9:.1f} GB free.")
        return 1
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    last = [-1.0]

    def on_progress(done, total):
        if total and (done / total - last[0] >= 0.01 or done == total):
            last[0] = done / total
            protocol.progress("merge", done / total, f"Saving the healed model: {done / 1e9:.1f} of {total / 1e9:.1f} GB")

    try:
        res = slice_io.merge_lora(tensors, adapter, out, on_progress=on_progress, should_stop=lambda: stop_file.exists())
        shutil.copy2(base / "config.json", out / "config.json")
        reshapejob._copy_side_files(base, out)
        if (base / "width_kept.npz").is_file():
            shutil.copy2(base / "width_kept.npz", out / "width_kept.npz")
        try:
            meta = json.loads((base / "ndforge_reshape.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {"format": 2, "tool": "ND Lore Forge"}
        s = cfg.get("summary") or {}
        meta["healed"] = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "from": base.name if base.name != "model" else base.parent.name, "merged_weights": res["merged"],
                          "lora_rank": res["rank"], "lora_scaling": res["scaling"], **{k: s.get(k) for k in (
                              "mode", "general", "steps", "tokens_seen", "kd_first", "kd_last", "calib_loss_before", "calib_loss_after", "teacher_calib_loss") if k in s}}
        (out / "ndforge_reshape.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "README.md").write_text(f"---\n{reshapejob._licence_front_matter(base / 'README.md')}---\n# {out.parent.name} (healed)\n\n"
                                       f"{base.parent.name}, taught by its original model (distillation).\n\n"
                                       "Made with ND Lore Forge (Slim or grow a model). See ndforge_reshape.json for the details.\n", encoding="utf-8")
        out.rename(final)
    except safetensors_io.Cancelled:
        shutil.rmtree(out, ignore_errors=True)
        protocol.result({"stopped": True, "model_dir": None})
        return 0
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    protocol.progress("done", 1.0, "")
    protocol.result({"model_dir": str(final), "merged": res["merged"], "bytes_written": need, "seconds": round(time.monotonic() - started, 1), "stopped": False})
    return 0
