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
    logits = (h.to(weight.dtype) @ weight.t()).float()      # the output layer may be 16-bit (kbit_upcast off): the matrix product runs in its type, everything after it in 32-bit
    kd, _ = kd_topk_loss(torch.log_softmax(logits, dim=-1), ids, logp, resid, mask)
    if labels is None:
        return kd, kd.new_zeros(())
    ce = F.cross_entropy(logits, labels.clamp(min=0), reduction="none")
    return kd, (ce * label_mask.float()).sum()


def chunked_kd_loss(hidden, weight, targets, mask, chunk: int = 256, labels=None, label_mask=None, checkpoint_head: bool = True):
    """hidden [n, C] (the student's last hidden states), weight [V, C] (its output layer), targets = the teacher's (ids [n, k], logp [n, k], resid [n]),
    mask [n]. The logits are made `chunk` tokens at a time inside torch.utils.checkpoint, so at most chunk x V floats exist (the backward pass makes them again).
    With labels [n] (the next token, any value where label_mask is 0) the cross-entropy over label_mask is returned as well.
    `checkpoint_head=False` keeps every chunk's logits for the backward pass instead of making them again: faster, and the same numbers, but about tokens x V x 10 bytes of memory.
    Returns (kd_sum, n_tokens, ce_sum, n_label_tokens); divide to get means."""
    import torch
    from torch.utils.checkpoint import checkpoint
    ids, logp, resid = targets
    n = hidden.shape[0]
    kd_total, ce_total = hidden.new_zeros((), dtype=torch.float32), hidden.new_zeros((), dtype=torch.float32)
    for a in range(0, n, max(1, int(chunk))):
        b = min(n, a + chunk)
        args = (hidden[a:b], weight, ids[a:b], logp[a:b], resid[a:b], mask[a:b], None if labels is None else labels[a:b], None if label_mask is None else label_mask[a:b])
        kd, ce = checkpoint(_chunk_losses, *args, use_reentrant=False) if checkpoint_head else _chunk_losses(*args)
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

from . import ckpt, fsutil, generaltext, healmem, protocol, safetensors_io  # noqa: E402
from .cachewrite import CACHE_FORMAT, CacheWriter, SharedCacheWriter, cache_matches, read_cache_index  # noqa: E402,F401
from .reshape import ReshapeError  # noqa: E402

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


def _require_same_ids(teacher_dir, student_dir) -> None:
    """The pair decision of the Heal Check (ids hash, then the probe texts): a pair the Check accepted is accepted here. The cache is still keyed by
    `tokenizer_hash(student)`; only the question 'may this teacher teach this student' uses the ids."""
    from . import tokpair
    d = tokpair.decide_pair(student_dir, teacher_dir)
    if not d["ok"]:
        raise ReshapeError("The original model and the thinned model do not use the same tokenizer, so the original's answers cannot be used to teach it. " + d["text"])
    if d["level"] == "warn":
        protocol.log(d["text"])


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


def data_length_labels():
    from . import data
    return data.LENGTH_LABELS


def data_length_bucket(n: int) -> str:
    from . import data
    return data.length_bucket(n)


def general_rows(source: dict, tok_fn, max_seq: int, budget: int, seed: int = 3407, stats: dict | None = None, probe=None, holdout: int = 0):
    """General text cut into rows of at most max_seq tokens, about `budget` tokens in all. tok_fn(text) -> list of token ids.
    A plain-text source is windowed: a long text gives several rows. A chat-shaped source (messages, conversations, prompt + answer) gives ONE row per conversation, its first
    max_seq tokens (a later window would have neither the question nor the chat frame); a conversation longer than twice max_seq is skipped. `stats` (a dict, optional)
    receives {"cut_rows": conversations cut to max_seq, "skipped_long": conversations skipped as too long, "length_buckets": every text or conversation counted by its length
    before cutting (the ones left out included), "skipped_reasoning": skipped conversations for which `probe(ids)` (optional) says they were reasoning rows,
    "holdout_rows": with `holdout` > 0, that many rows read PAST the ones returned (rendered and cut the same way, never a later window of a text a returned row came from)}.
    The order is the dataset's own, or mixed by source["seed"] when it has one (the same seed gives the same sample); returns (rows, tokens). Raises whatever the source raises (the caller explains it)."""
    stats = stats if stats is not None else {}
    stats.setdefault("cut_rows", 0)
    stats.setdefault("skipped_long", 0)
    stats.setdefault("skipped_reasoning", 0)
    if holdout > 0:
        stats.setdefault("holdout_rows", [])
    buckets = stats.setdefault("length_buckets", {label: 0 for label in data_length_labels()})
    report = source.get("report")
    if not isinstance(report, dict):
        report = {}
        source = {**source, "report": report}                 # the reader fills it: what kind of rows these are is known after the first one
    rows, tokens = [], 0
    for text in _general_texts(source):
        ids = tok_fn(text)
        buckets[data_length_bucket(len(ids))] += 1
        if (report.get("detected") or {}).get("kind", "text") != "text":
            if len(ids) > 2 * max_seq:
                stats["skipped_long"] += 1
                if probe is not None and probe(ids):
                    stats["skipped_reasoning"] += 1
                continue
            if len(ids) > max_seq:
                stats["cut_rows"] += 1
            windows = [ids[:max_seq]]
        else:
            windows = [ids[a:a + max_seq] for a in range(0, len(ids), max_seq)]
        for w in windows:
            if len(w) < MIN_GENERAL_ROW:
                continue
            row = {"input_ids": w, "completion_mask": [1] * len(w), "kind": "general", "source": "general"}
            if tokens >= budget:                                         # past the budget: only with a held-out set to fill
                held = stats.setdefault("holdout_rows", [])
                held.append(row)
                if len(held) >= holdout:
                    return rows, tokens
                break                                                    # one row of a text: its other windows would be next to a row that was trained on
            rows.append(row)
            tokens += len(w)
            if tokens >= budget:
                if holdout <= 0:
                    return rows, tokens
                break                                                    # the rest of this text is not held out
    return rows, tokens


# ------------------------------------------------------------------------------------------------------------ what the heal's rows cover

def marker_ids(tokenizer, fam_name: str) -> dict:
    """The token ids that mark a chat: {"assistant": ids of the start of an assistant turn ("<|im_start|>assistant\\n") or None, "end": the end-of-turn id or None,
    "think_open", "think_close": the reasoning markers or None}. A marker the tokenizer does not know as one token is None (the row statistics then say less, never more)."""
    from . import data
    from .families import family
    tok = data.text_tokenizer(tokenizer)
    unk = getattr(tok, "unk_token_id", None)

    def one(text):
        try:
            i = tok.convert_tokens_to_ids(text)
        except Exception:                                   # noqa: BLE001 - a tokenizer without the call
            return None
        return None if i is None or i == unk or not isinstance(i, int) or i < 0 else i

    out = {"assistant": None, "end": None, "think_open": one("<think>"), "think_close": one("</think>")}
    end = (family(fam_name).get("end") or "").strip()
    out["end"] = one(end) if end else (tok.eos_token_id if isinstance(getattr(tok, "eos_token_id", None), int) else None)
    if one("<|im_start|>") is not None:
        try:
            out["assistant"] = list(tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"])
        except Exception:                                   # noqa: BLE001
            pass
    return out


def _find_last(ids: list, pat: list) -> int:
    """The index just after the last occurrence of the id list `pat` in `ids`, or -1."""
    n = len(pat)
    for a in range(len(ids) - n, -1, -1):
        if ids[a:a + n] == pat:
            return a + n
    return -1


BLANK_THINK = 2                 # a reasoning block of this many tokens or fewer (the newlines of an empty one) is not reasoning


def reasoning_probe(markers: dict):
    """A function ids -> True when the row holds a non-empty reasoning block (or a reasoning block that was cut), None when the tokenizer has no such markers."""
    o, c = markers.get("think_open"), markers.get("think_close")
    if o is None or c is None:
        return None

    def probe(ids: list) -> bool:
        try:
            a = ids.index(o)
        except ValueError:
            return False
        try:
            b = ids.index(c, a)
        except ValueError:
            return True
        return b - a - 1 > BLANK_THINK
    return probe


def classify_row(row: dict, markers: dict, max_seq: int, chat: bool) -> dict:
    """How one kept row splits: {"chat", "complete", "think": "none"|"complete"|"cut_reasoning"|"cut_answer", "prompt", "reasoning", "answer", "plain"} in tokens.
    A text row is all `plain`. A chat row is split at the start of the assistant turn (the marker ids, else the project's completion mask); with the reasoning markers the block
    from <think> to </think> is `reasoning` and what follows is `answer`. `complete`: the end-of-turn token closes the row (families without one: the row is shorter than max_seq)."""
    ids = row["input_ids"]
    n = len(ids)
    end = markers.get("end")
    closed = (end is not None and end in ids[-3:]) if end is not None else n < max_seq
    if not chat:
        return {"chat": False, "complete": n < max_seq, "think": "none", "prompt": 0, "reasoning": 0, "answer": 0, "plain": n}
    start = _find_last(ids, markers["assistant"]) if markers.get("assistant") else -1
    if start < 0:
        mask = row.get("completion_mask") or []
        first = next((i for i, m in enumerate(mask) if m), None) if len(mask) == n else None
        if first is None or first == 0:
            return {"chat": True, "complete": closed, "think": "none", "prompt": 0, "reasoning": 0, "answer": 0, "plain": n}
        start = first
    tail = ids[start:]
    prompt, reasoning, think = start, 0, "none"
    o, c = markers.get("think_open"), markers.get("think_close")
    if o is not None and c is not None and o in tail:
        a = tail.index(o)
        if c in tail[a:]:
            b = tail.index(c, a)
            if b - a - 1 > BLANK_THINK:
                prompt, reasoning, think = start + a, b - a + 1, "complete" if closed else "cut_answer"
            else:
                prompt = start + b + 1                      # the empty block of a template that switched thinking off belongs to the prompt
        else:
            prompt, reasoning, think = start + a, len(tail) - a, "cut_reasoning"
    answer = n - prompt - reasoning
    return {"chat": True, "complete": closed, "think": think, "prompt": prompt, "reasoning": reasoning, "answer": answer, "plain": 0}


def coverage(rows: list, markers: dict, max_seq: int, stats: dict | None = None, general_chat: bool = False) -> dict:
    """What the rows of a heal cover (docs/HEAL_SETTINGS.md): the row counts by kind, how many chat rows are complete, cut or left out, the same for reasoning rows, the share of the
    kept tokens that is prompt / reasoning / answer / plain text, and the lengths of all rows before cutting (the ones left out included). `stats` is build_rows' (the left-out
    counts and `length_buckets`); `general_chat`: the general rows are conversations. Pure: no tokenizer, no files."""
    stats = stats or {}
    kinds = {"project_text": 0, "project_chat": 0, "general_text": 0, "general_chat": 0}
    chat = {"complete": 0, "cut": 0}
    think = {"complete": 0, "cut_reasoning": 0, "cut_answer": 0}
    tokens = {"prompt": 0, "reasoning": 0, "answer": 0, "plain": 0}
    for r in rows:
        is_general = r.get("source") == "general"
        is_chat = general_chat if is_general else r.get("kind") == "chat"
        c = classify_row(r, markers, max_seq, is_chat)
        kinds[("general_" if is_general else "project_") + ("chat" if is_chat else "text")] += 1
        if is_chat:
            chat["complete" if c["complete"] else "cut"] += 1
            if c["think"] != "none":
                think[c["think"]] += 1
        for key in tokens:
            tokens[key] += c[key]
    total = sum(tokens.values())
    skipped = {"project_chat": int(stats.get("skipped_too_long", 0)), "general_chat": int(stats.get("general_skipped_long", 0))}
    skipped_reasoning = int(stats.get("skipped_reasoning", 0)) + int(stats.get("general_skipped_reasoning", 0))
    reasoning_rows = sum(think.values())
    out = {"v": 1, "max_seq": int(max_seq), "rows": kinds, "chat_rows": {**chat, "skipped": skipped["project_chat"] + skipped["general_chat"], "skipped_by": skipped},
           "reasoning_rows": {**think, "skipped": skipped_reasoning, "total": reasoning_rows + skipped_reasoning, "known": markers.get("think_open") is not None and markers.get("think_close") is not None},
           "token_share": {k: (round(v / total, 4) if total else 0.0) for k, v in tokens.items()}, "tokens": tokens,
           "length_buckets": dict(stats.get("length_buckets") or {}), "notes": []}
    gb = stats.get("general_length_buckets") or {}
    if gb:
        out["length_buckets"] = {k: out["length_buckets"].get(k, 0) + gb.get(k, 0) for k in set(out["length_buckets"]) | set(gb)}
        out["length_buckets"] = {k: out["length_buckets"][k] for k in data_length_labels() if k in out["length_buckets"]}
    lost = think["cut_reasoning"] + think["cut_answer"] + skipped_reasoning
    total_reasoning = out["reasoning_rows"]["total"]
    if total_reasoning and lost / total_reasoning > 0.10:
        pct = round(100 * lost / total_reasoning)
        if max_seq < 8192:
            out["notes"].append(f"{pct}% of the reasoning rows lost their answer or were left out because they are longer than {max_seq} tokens: choose 8192 in Heal settings if the plan says it fits.")
        else:
            out["notes"].append(f"{pct}% of the reasoning rows lost their answer or were left out: these traces are longer than {max_seq} tokens, and a longer bucket is not available yet.")
    if total_reasoning and out["token_share"]["reasoning"] < 0.75 and think["complete"] + think["cut_answer"] + think["cut_reasoning"]:
        out["notes"].append(f"Reasoning is {round(100 * out['token_share']['reasoning'])}% of the kept tokens; Unsloth's guidance for keeping thinking is 75% or more (a guide, not a rule).")
    return out


def build_rows(tokenizer, data_cfg: dict, fam_name: str, max_seq: int, seed: int = 3407, holdout: int = 0):
    """The unique rows the teacher answers and the student learns from: the project's saved training data (as in training) followed by general text.
    data_cfg: {"train": path, "tokens": total heal tokens, "general": {"share": 0.5, "id" | "file", "column", "config", "split"} | None}.
    The general text is cut to its share of the heal budget (project data repeats when it is smaller than its share; see training_order). When the general
    text cannot be read (offline ...), healing goes on with the project's data only and stats["general_error"] says why.
    Returns (rows, stats)."""
    from . import data
    markers = marker_ids(tokenizer, fam_name)
    probe = reasoning_probe(markers)
    proj, pstats = data.load_training_rows(tokenizer, data_cfg["train"], fam_name, max_seq, seed, **({"probe": probe} if probe else {}))
    for r in proj:
        r["source"] = "project"
    stats = {"project_rows": len(proj), "project_tokens": pstats["tokens"], "general_rows": 0, "general_tokens": 0, "general_error": None, "general_error_code": None,
             "general_source": None, "general_detected": None, "general_texts_skipped": 0, "general_cut_rows": 0, "general_skipped_long": 0}
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
        inner = data.text_tokenizer(tokenizer)
        report: dict = {}
        g.update(seed=seed, report=report, render=lambda msgs: data.render_chat_full(tokenizer, msgs, fam_name)[1])
        protocol.progress("data", 0.0, f"Getting the general text ({name})")
        long_rows: dict = {}
        try:
            grows, gtok = general_rows(g, lambda t: inner(t, add_special_tokens=False)["input_ids"], max_seq, budget, seed, long_rows, probe, **({"holdout": holdout} if holdout > 0 else {}))
            if holdout > 0:
                stats["holdout_rows"] = long_rows.pop("holdout_rows", [])
        except Exception as exc:      # noqa: BLE001 - offline, gated, wrong id or column: all end the same plain way
            grows, gtok = [], 0
            stats["general_error"], stats["general_error_code"] = generaltext.failure_note(
                exc, g, "downloaded", "Healing goes on with your world's text only; try again later or choose another dataset.", DEFAULT_GENERAL)
            protocol.log(stats["general_error"])
        stats.update(general_detected=report.get("detected"), general_texts_skipped=report.get("skipped", 0), general_cut_rows=long_rows.get("cut_rows", 0),
                     general_skipped_long=long_rows.get("skipped_long", 0), general_skipped_reasoning=long_rows.get("skipped_reasoning", 0),
                     general_length_buckets=long_rows.get("length_buckets") or {})
        rows += grows
        stats.update(general_rows=len(grows), general_tokens=gtok)
        if grows:
            protocol.log(f"General text used: {name} ({len(grows)} rows, {gtok:,} tokens).")
            if stats["general_cut_rows"] or stats["general_skipped_long"]:
                protocol.log(f"Long conversations: {stats['general_cut_rows']} were cut to {max_seq} tokens, {stats['general_skipped_long']} were longer than {2 * max_seq} tokens and left out.")
        protocol.progress("data", 1.0, "")
    stats["length_buckets"] = pstats.get("length_buckets") or {}
    stats["skipped_too_long"], stats["skipped_reasoning"] = pstats.get("skipped_too_long", 0), pstats.get("skipped_reasoning", 0)
    stats["coverage"] = coverage(rows, markers, max_seq, stats, (stats.get("general_detected") or {}).get("kind", "text") != "text")
    return rows, stats


def _cycle(rows: list, idx: list, want: int, rng, block: int) -> list[int]:
    """Row indices from `idx` (cycled in a fresh shuffled order, shuffled in blocks) until their tokens reach `want`."""
    chosen: list[int] = []
    got = 0
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
    return chosen


def _interleave(chosen: list[int], seed: int, block: int) -> list[int]:
    """the two kinds are interleaved by block so that every stretch of the run has both"""
    rng2 = random.Random(seed + 1)
    chunks = [chosen[a:a + block] for a in range(0, len(chosen), block)]
    rng2.shuffle(chunks)
    return [i for c in chunks for i in c]


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
        chosen += _cycle(rows, by[kind], int(tokens * share[kind]), rng, block)
    return _interleave(chosen, seed, block)


# ------------------------------------------------------------------------------------------------------------ healing in rounds
# One heal is trained in rounds (docs/HEAL_ROUNDS.md): each round learns from all project rows (cycled as in a normal run) and the general rows the teacher answered FOR THIS
# ROUND (rows general_from .. general_to-1 of the run's row list), so no general row is repeated across rounds. The optimizer and the scheduler position go on from round
# to round; the learning rate of a round has its own warm-up, plateau and cool-down, so that every round ends with an annealed adapter that can be checked.

DEFAULT_ROUND_SCHEDULE = {"v": 1, "gamma": 0.85, "floor": 0.35, "cooldown": 0.2, "min_ratio": 0.05, "rewarm_share": 0.1, "rewarm_max": 10}


def round_lr_factor(step: int, s_start: int, steps: int, index: int, warmup: int, sch: dict) -> float:
    """The learning-rate factor (times the base rate) at global optimizer step `step` for round `index` (1, 2, ...), which has `steps` steps and starts at global step
    `s_start`. Round 1 warms up from 0 (over the job's warm-up, at most a quarter of the round); a later round re-warms from `min_ratio` x its peak over `rewarm_share`
    of the round (at most `rewarm_max` steps). The peak falls by `gamma` per round down to `floor`. The last `cooldown` share of every round decays as 1 - sqrt(u) to
    `min_ratio` x the peak. Pure (no torch): the trainer wraps it in a LambdaLR."""
    t = min(max(int(step) - int(s_start), 0), max(int(steps) - 1, 0))
    peak = max(sch["floor"], sch["gamma"] ** (index - 1))
    if index == 1:
        w, lo = min(int(warmup), math.ceil(0.25 * steps)), 0.0
    else:
        w, lo = min(sch["rewarm_max"], math.ceil(sch["rewarm_share"] * steps)), sch["min_ratio"] * peak
    w = max(w, 1)
    c = max(1, math.ceil(sch["cooldown"] * steps))
    if t < w:
        return lo + (peak - lo) * (t + 1) / w
    if t >= steps - c:
        u = (t - (steps - c) + 1) / c
        return peak * (sch["min_ratio"] + (1 - sch["min_ratio"]) * (1 - math.sqrt(min(u, 1.0))))
    return peak


def round_order(rows: list, general_from: int, general_to: int, round_tokens: int, general_share: float, seed: int, block: int = 64) -> list[int]:
    """The row indices of one round: every project row (cycled in a fresh shuffled order, as in training_order) and the general rows `general_from` .. `general_to - 1`
    (each once, never more than they hold), about `round_tokens` tokens in all, the general part at most `general_share` of them. Interleaved in blocks like training_order."""
    proj = [i for i, r in enumerate(rows) if r.get("source") != "general"]
    gen = [i for i in range(max(0, general_from), min(general_to, len(rows))) if rows[i].get("source") == "general"]
    gen_tokens = sum(len(rows[i]["input_ids"]) for i in gen)
    want_g = min(int(round_tokens * general_share), gen_tokens) if proj else min(round_tokens, gen_tokens)
    want_p = round_tokens - want_g if proj else 0
    rng = random.Random(seed)
    chosen = _cycle(rows, proj, want_p, rng, block) + _cycle(rows, gen, want_g, rng, block)
    return _interleave(chosen, seed, block)


def round_slice(order: list, steps_done: int, per_step: int) -> list:
    """What is left of a round's order after `steps_done` optimizer steps of `per_step` rows: a crash in the middle of a round continues exactly here."""
    return list(order[max(0, int(steps_done)) * max(1, int(per_step)):])


def round_start(ckdir, index: int, s_start: int, out) -> dict:
    """Where training round `index` (global steps s_start ..) starts from, decided from the checkpoints of the run (torch-free):
    "continue": a usable checkpoint at or after the round's first step exists (the forced save at the end of the previous round, or one from inside this round): training
    resumes from the newest one with the optimizer and scheduler as they were, HF's global step continues (gs_start = s_start for a round that has not begun);
    "fresh": no such checkpoint (they were deleted): the checkpoints that are left are removed, the previous round's adapter is the start and the optimizer begins anew
    (HF's step counts from 0: gs_start = 0, the round's steps are shifted by the caller). A marker file lets a restart inside such a fresh round resume from its own checkpoints.
    Returns {"mode", "gs_start", "init_adapter", "note"}."""
    ckdir, out = Path(ckdir), Path(out)
    ck = ckpt.usable(ckdir)
    marker = ckdir / "FRESH_ROUND"
    try:
        marked = int(marker.read_text(encoding="utf-8").strip()) == int(index)
    except (OSError, ValueError):
        marked = False
    prev = out / "rounds" / f"r{int(index) - 1}" / "adapter"
    if int(index) <= 1:
        return {"mode": "continue", "gs_start": 0, "init_adapter": None, "note": None}
    if marked:
        return {"mode": "fresh", "gs_start": 0, "init_adapter": str(prev) if prev.is_dir() else None, "note": None}
    if any(ckpt.step_of(c) >= int(s_start) for c in ck):
        return {"mode": "continue", "gs_start": int(s_start), "init_adapter": None, "note": None}
    if not prev.is_dir():
        raise ReshapeError("The previous round's adapter and its checkpoint are both missing, so healing cannot go on from the end of the previous round. Heal again from the start.")
    import shutil as _sh
    for c in ckdir.glob("checkpoint-*"):
        _sh.rmtree(c, ignore_errors=True)
    ckdir.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(int(index)), encoding="utf-8")
    return {"mode": "fresh", "gs_start": 0, "init_adapter": str(prev), "note": f"No checkpoint of the end of round {int(index) - 1} was found: round {int(index)} starts from that round's adapter with a fresh optimizer."}


# ============================================================================================================ the cache

def row_hashes(rows: list) -> list:
    """sha1 of the token ids of every row (20 bytes each): what tells a shared cache that the new rows start with the rows it already answered."""
    from array import array
    return [hashlib.sha1(array("I", [int(x) for x in r["input_ids"]]).tobytes()).digest() for r in rows]


def _file_sha256(path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def shared_store_id(cfg: dict, teacher_dir, student, k: int, max_seq: int, seed: int, version: int = 2, engine_id: dict | None = None) -> str:
    """The folder name of the shared cache these rows belong to: the teacher (its folder and config), the tokenizer, k, the length, how the teacher is loaded, the general text
    WITHOUT its share (the share is the budget: a bigger budget only adds rows), the seed, the model family and the project's training file.
    Version 2 adds the identity of the teacher's weight files (`safetensors_io.weights_manifest`); version 1 is the id older caches were saved under.
    `engine_id` (only for the vLLM teacher: `vllmcache.identity`: engine version, teacher quantization, the original's repo and weights) is part of the id when given, so a vLLM cache never
    shares a store with the standard (Hugging Face, 4-bit) teacher; without it the id is exactly what it always was."""
    data_cfg = cfg.get("data") or {}
    g = data_cfg.get("general")
    used = bool(g) and float(g.get("share", 0) or 0) > 0
    gen = generaltext.general_key(g, 1.0 if used else 0, seed)
    gen.pop("share", None)
    ident = {"v": 1, "teacher": str(Path(teacher_dir).resolve()), "teacher_config": _file_sha256(Path(teacher_dir) / "config.json"), "tokenizer": tokenizer_hash(student), "k": int(k),
             "max_len": int(max_seq), "load": (cfg.get("teacher") or {}).get("load", "auto"), "general": gen, "seed": int(seed),
             "family": (cfg.get("model") or {}).get("family", "generic"), "train": _file_sha256(data_cfg.get("train") or "")}
    if int(version) >= 2:
        ident["v"] = 2
        ident["teacher_weights"] = safetensors_io.weights_manifest(teacher_dir)
    if engine_id:
        ident["engine_id"] = engine_id
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode("utf-8")).hexdigest()[:20]


ADOPT_NOTE = ("These answers were made before the program checked the original's weight files; if you replaced them in place, delete the saved answers.")


def choose_store(base, cfg: dict, teacher_dir, student, k: int, max_seq: int, seed: int, meta: dict, hashes: list, engine_id: dict | None = None):
    """(folder, note) of the shared cache for these rows. The id of the current version first; when no store of that id exists yet but one made under the old id (before the
    weights were part of it) does, that store is adopted once: it gets `teacher_weights` in its index and a note says so. An old store whose index already names other weights is
    not used."""
    base = Path(base)
    new_id = shared_store_id(cfg, teacher_dir, student, k, max_seq, seed, 2, engine_id)
    if engine_id:                                       # a vLLM cache has no older id to adopt from
        return pick_store(base, new_id, meta, hashes), None
    if read_cache_index(base / new_id) is None:
        old_dir = base / shared_store_id(cfg, teacher_dir, student, k, max_seq, seed, 1)
        idx = read_cache_index(old_dir)
        if idx is not None and idx.get("teacher_weights") in (None, meta.get("teacher_weights")) and store_extends(old_dir, {**meta, "teacher_weights": None}, hashes):
            note = None
            if idx.get("teacher_weights") is None:
                idx["teacher_weights"] = meta.get("teacher_weights")
                tmp = old_dir / "index.partial.json"
                tmp.write_text(json.dumps(idx), encoding="utf-8")
                os.replace(tmp, old_dir / "index.json")
                note = ADOPT_NOTE
            return old_dir, note
    return pick_store(base, new_id, meta, hashes), None


def _read_hashes(folder):
    from . import slice_io
    np = slice_io.np()
    try:
        arr = np.load(Path(folder) / "rowhashes.npy")
    except (OSError, ValueError):
        return None
    return arr if arr.ndim == 2 and arr.shape[1] == 20 else None


def store_extends(folder, meta: dict, hashes: list) -> bool:
    """True when the shared cache in `folder` (none yet counts as true) is for this teacher, tokenizer, k and length AND the rows it has answered are the first rows of `hashes`
    (or `hashes` are the first rows of what it has answered: a smaller run)."""
    old = read_cache_index(folder)
    if old is None:
        return True
    if not all(old.get(key) == meta.get(key) for key in ("teacher", "k", "tokenizer_hash", "max_len", "engine", "teacher_quant", "teacher_origin", "kv_cache")):
        return False
    if old.get("teacher_weights") and meta.get("teacher_weights") and old["teacher_weights"] != meta["teacher_weights"]:
        return False
    done = int(old.get("done_rows", 0))
    if done == 0:
        return True
    have = _read_hashes(folder)
    if have is None or len(have) < done:
        return False
    m = min(done, len(hashes))
    return all(bytes(have[i]) == hashes[i] for i in range(m))


def pick_store(base, store_id: str, meta: dict, hashes: list) -> Path:
    """`base/<id>`, or `base/<id>-2` ... when the first ones hold answers for other rows (never overwritten: another model's view may still point at them)."""
    base = Path(base)
    for n in range(1, 100):
        d = base / (store_id if n == 1 else f"{store_id}-{n}")
        if store_extends(d, meta, hashes):
            return d
    raise ReshapeError("Too many different saved answers for this teacher: delete some of the models under models/_teacher_cache.")


def save_rows(folder, rows: list, name: str = "rows") -> None:
    """rows.npz next to the cache: the token ids of the rows the teacher answered, so the student learns from exactly these rows without making them again
    (and without a second download of the general text). Needs numpy."""
    from . import slice_io
    np = slice_io.np()
    lens = np.array([len(r["input_ids"]) for r in rows], dtype=np.int64)
    kinds = sorted({str(r.get("kind", "")) for r in rows})
    np.savez(Path(folder) / f"{name}.partial.npz", ids=np.array([x for r in rows for x in r["input_ids"]], dtype=np.int32),
             mask=np.array([x for r in rows for x in r["completion_mask"]], dtype=np.uint8), offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
             general=np.array([r.get("source") == "general" for r in rows], dtype=np.uint8), kind=np.array([kinds.index(str(r.get("kind", ""))) for r in rows], dtype=np.int16),
             kinds=np.array(kinds))
    os.replace(Path(folder) / f"{name}.partial.npz", Path(folder) / f"{name}.npz")


def load_rows(folder, name: str = "rows") -> list:
    from . import slice_io
    np = slice_io.np()
    try:
        with np.load(Path(folder) / f"{name}.npz") as z:
            ids, mask, off, gen, kind, kinds = (z[k] for k in ("ids", "mask", "offsets", "general", "kind", "kinds"))
    except (OSError, ValueError, KeyError):
        raise ReshapeError("The teacher's saved answers are missing or unreadable; they will be made again.") from None
    return [{"input_ids": ids[off[i]:off[i + 1]].tolist(), "completion_mask": mask[off[i]:off[i + 1]].tolist(), "kind": str(kinds[kind[i]]),
             "source": "general" if gen[i] else "project"} for i in range(len(off) - 1)]


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
        # a shared cache keeps its parts in another folder (SharedCacheWriter); the index of this folder says where
        store = self.index.get("store")
        self.parts_dir = (self.dir / store).resolve() if store else self.dir
        self._loaded: dict = {}

    @property
    def done_rows(self) -> int:
        return int(self.index["done_rows"])

    def _part(self, p: int):
        if p not in self._loaded:
            if len(self._loaded) >= 2:
                self._loaded.pop(next(iter(self._loaded)))
            with self.np.load(self.parts_dir / self.parts[p]["file"]) as z:
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
            # the chunks stay on the card (the logits of one chunk at a time bound the memory), and the row comes back to the host in one go
            parts = [topk_from_logits(h[a:a + chunk].to(weight.dtype) @ weight.t(), k) for a in range(0, h.shape[0], chunk)]
            out.append(tuple(torch.cat([p[j] for p in parts]).cpu().numpy() for j in range(3)))
    return out


WINDOW_ROWS = 64            # rows answered together in length order before they are written in their own order
SPEED_LOG_EVERY = 60.0      # seconds between the "cache_speed" lines


TEACHER_MAX_ROWS = 16               # the most rows of one forward pass when the rows are taken by a token budget


def rows_in_budget(lengths_longest_first: list, budget: int, cap: int = TEACHER_MAX_ROWS) -> int:
    """How many of the rows (sorted from the longest, as `length_order` gives them) go through the original together: as many as fit `budget` tokens when each is padded to the longest of the batch
    (the first), at least 1, at most `cap`. A budget of 0 is "the fixed number of rows" (the caller's)."""
    if not lengths_longest_first:
        return 0
    return max(1, min(cap, int(budget) // max(1, int(lengths_longest_first[0]))))


def shrink_budget(budget: int, longest: int) -> int:
    """After an out-of-memory error: half the token budget (a budget below the longest row means one row at a time)."""
    return max(int(budget) // 2, 0) if int(budget) // 2 >= longest else 0


def length_order(window: list) -> list:
    """The positions of `window` from the longest row to the shortest (equal lengths keep their order): the order the teacher answers them in."""
    return sorted(range(len(window)), key=lambda j: -len(window[j]["input_ids"]))


def cache_meta(cfg: dict, rows: list, stats: dict, student, k: int, max_seq: int, teacher_dir=None, engine_id: dict | None = None) -> dict:
    """What index.json says about a teacher cache (the first eight keys decide `cache_matches`). `engine_id` (vLLM teacher only) adds `engine`, `teacher_quant` and `teacher_origin`,
    which `cache_matches` and `store_extends` compare too: a cache made by another engine or another teacher quantization is never taken over."""
    gen = (cfg.get("data") or {}).get("general")
    extra = {"teacher_weights": safetensors_io.weights_manifest(teacher_dir)} if teacher_dir else {}
    extra.update(engine_id or {})
    return {**extra, "teacher": cfg["teacher"]["repo"], "k": k, "rows": len(rows), "data_hash": data_hash(rows), "tokenizer_hash": tokenizer_hash(student), "max_len": max_seq,
            "project_rows": stats["project_rows"], "general_rows": stats["general_rows"],
            "general_key": generaltext.general_key(gen, (gen or {}).get("share", 0), int(cfg.get("seed", 3407))),
            "general_source": stats["general_source"],
            # who made the answers and for which rows: the program compares `rows_key` before it skips the teacher pass (it is not part of cache_matches: the rows' own hash is)
            "teacher_load": cfg["teacher"].get("load", "auto"), "rows_key": cfg.get("rows_key")}


def rows_upto(rows: list, general_tokens) -> int:
    """How many leading rows a partial teacher pass answers (healing in rounds): every project row, then general rows until their tokens reach `general_tokens`
    (the row that reaches it is included). `rows` are the project rows followed by the general rows, as build_rows makes them. 0 or less = the project rows only."""
    want, got = int(general_tokens or 0), 0
    for i, r in enumerate(rows):
        if r.get("source") == "general":
            if got >= want:
                return i
            got += len(r["input_ids"])
    return len(rows)


def _reusable_rows(out, cfg: dict, k: int, max_seq: int, student):
    """The rows an earlier call of this run saved next to the cache (rows.npz), when they are exactly the rows this call would build: the cache's own index says for which
    data they were made (rows_key, general text, k, length, tokenizer, teacher) and the rows' hash must match. None otherwise (the caller then builds the rows)."""
    idx = read_cache_index(out)
    if not idx or not cfg.get("rows_key") or idx.get("rows_key") != cfg.get("rows_key"):
        return None
    gen = (cfg.get("data") or {}).get("general")
    want_key = generaltext.general_key(gen, (gen or {}).get("share", 0), int(cfg.get("seed", 3407)))
    if (idx.get("k") != k or idx.get("max_len") != max_seq or idx.get("teacher") != cfg["teacher"]["repo"] or idx.get("general_key") != want_key
            or idx.get("tokenizer_hash") != tokenizer_hash(student)):
        return None
    try:
        rows = load_rows(out)
    except ReshapeError:
        return None
    if len(rows) != idx.get("rows") or data_hash(rows) != idx.get("data_hash"):
        return None
    proj = [r for r in rows if r.get("source") != "general"]
    gen_rows = [r for r in rows if r.get("source") == "general"]
    stats = {"project_rows": len(proj), "project_tokens": sum(len(r["input_ids"]) for r in proj), "general_rows": len(gen_rows),
             "general_tokens": sum(len(r["input_ids"]) for r in gen_rows), "general_error": None, "general_error_code": None, "general_source": idx.get("general_source"),
             "general_detected": None, "general_texts_skipped": 0, "general_cut_rows": 0, "general_skipped_long": 0}
    return rows, stats


def _read_holdout_meta(folder):
    try:
        m = json.loads((Path(folder) / "holdout.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return m if isinstance(m, dict) else None


def _read_coverage(folder):
    try:
        c = json.loads((Path(folder) / "coverage.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return c if isinstance(c, dict) and c.get("v") == 1 else None


def _write_coverage(folder, cov: dict) -> None:
    tmp = Path(folder) / "coverage.partial.json"
    tmp.write_text(json.dumps(cov, indent=2), encoding="utf-8")
    os.replace(tmp, Path(folder) / "coverage.json")


def _early_vllm_probe(cfg: dict) -> None:
    """vLLM is required: find out NOW (seconds, before the model is read and the rows are made) whether files can be moved into the container, not after an hour of preparation."""
    from . import vllmcache as _vc
    if not (_vc.wanted(cfg) and (cfg.get("engine") or {}).get("required")):
        return
    protocol.progress("cache", 0.0, "Testing the vLLM container: can files go in and out?")
    pr = _vc.probe(cfg)
    protocol.log(f"Container copy test: {'ok' if pr['ok'] else 'FAILED'} after {pr.get('seconds')} s. {pr['text']}")
    if not pr["ok"]:
        raise ReshapeError(f"vLLM was required for the teacher's answers but the container cannot be used: {pr['text']}" + ("" if _vc.REMEDY_OFF in pr["text"] else " " + _vc.REMEDY_OFF))


def run_cache(cfg: dict) -> int:
    """job distill_cache: {"teacher": {"repo", "load": "auto|4bit|16bit|offload"}, "student": folder of the thinned model (its tokenizer is used),
    "model": {"family"}, "data": {"train", "tokens", "general": {...}}, "k": 32, "max_seq": 2048, "batch": 4, "chunk": 256, "part_tokens": 262144,
    "out": cache folder, "stop_file": path, "upto_general_tokens": N (optional: answer the project rows and the general rows up to N general tokens only, then stop; healing in
    rounds), "reuse_rows": bool (optional: take the rows from out/rows.npz when they are the rows of this run, instead of reading the general text again)}.
    Saves the teacher's top-k answers for every row (resumable). Result: rows, tokens, bytes, reused, stopped, notes, done_rows, general_rows_done, general_tokens_done, rows_before."""
    protocol.reserve_stdout()
    healmem.set_alloc_conf()
    started = time.monotonic()
    from . import reshapejob, slice_io
    slice_io.np()
    out = Path(cfg["out"])
    stop_file = Path(cfg.get("stop_file") or out.parent / "STOP")
    k, max_seq = int(cfg.get("k", 32)), int(cfg.get("max_seq", 2048))
    student = Path(cfg["student"])
    _early_vllm_probe(cfg)
    teacher_dir = reshapejob._resolve_source(cfg["teacher"]["repo"])
    _require_same_ids(teacher_dir, student)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(student))
    again = _reusable_rows(out, cfg, k, max_seq, student) if cfg.get("reuse_rows") else None
    if again:
        rows, stats = again
        protocol.log(f"The rows of this run are taken from the saved ones ({len(rows)} rows); the general text is not read again.")
    else:
        hold_n = int((cfg.get("holdout") or {}).get("rows", 0) or 0) if isinstance(cfg.get("holdout"), dict) else 0
        rows, stats = build_rows(tok, cfg["data"], cfg["model"].get("family", "generic"), max_seq, int(cfg.get("seed", 3407)), hold_n)
    hold = stats.pop("holdout_rows", None) or []
    from . import vllmcache
    veng = None                                         # the vLLM teacher (job key `engine: {"kind": "vllm"}`, off when absent): its identity is part of the cache's
    if vllmcache.wanted(cfg):
        try:
            veng = vllmcache.identity(cfg, cfg["teacher"]["repo"], teacher_dir)
        except (vllmcache.VllmCacheError, vllmcache.vllmengine.EngineError) as exc:
            if (cfg.get("engine") or {}).get("required"):
                raise ReshapeError(f"vLLM was required for the teacher's answers but cannot be used: {exc} {vllmcache.REMEDY}") from None
            protocol.log(f"vLLM is not used for the teacher's answers: {exc} The standard teacher is used.")
    meta = cache_meta(cfg, rows, stats, student, k, max_seq, teacher_dir, veng)
    upto = cfg.get("upto_general_tokens")
    target = len(rows) if upto is None else rows_upto(rows, upto)
    legacy = read_cache_index(out)
    adopt_note = None
    hashes = row_hashes(rows) if cfg.get("shared_cache") and cfg.get("shared_dir") else None

    def open_writer(meta: dict, engine_id: dict | None):
        nonlocal adopt_note
        part_tokens = int(cfg.get("part_tokens", 262144))
        if cfg.get("shared_cache") and cfg.get("shared_dir") and not (legacy and "store" not in legacy and legacy.get("parts")):
            # parts in the shared folder, this model's own index and rows next to it (see "the shared cache")
            store, note = choose_store(cfg["shared_dir"], cfg, teacher_dir, student, k, max_seq, int(cfg.get("seed", 3407)), meta, hashes, engine_id)
            adopt_note = adopt_note or note
            if note:
                protocol.log(note)
            w = SharedCacheWriter(store, out, meta, hashes, part_tokens)
            protocol.log(f"The teacher's answers are kept in the shared folder {store.name} ({w.done_rows} of {len(rows)} rows are already there).")
            return w
        return CacheWriter(out, meta, part_tokens)

    writer = open_writer(meta, veng)
    if not again:
        save_rows(out, rows)
        if hold:
            save_rows(out, hold, "holdout")                 # general rows read past the ones the heal learns from: only the loss is measured on them, never trained on
            (out / "holdout.json").write_text(json.dumps({"rows": len(hold), "tokens": sum(len(r["input_ids"]) for r in hold), "teacher_loss": None}), encoding="utf-8")
    cov = stats.get("coverage") or _read_coverage(out) or coverage(rows, marker_ids(tok, cfg["model"].get("family", "generic")), max_seq, stats,
                                                                   ((stats.get("general_detected") or {}).get("kind", "text") != "text"))
    _write_coverage(out, cov)
    notes = [stats["general_error"]] if stats["general_error"] else []
    if adopt_note:
        notes.append(adopt_note)
    total_tokens = sum(len(r["input_ids"]) for r in rows)
    result = {"rows": len(rows), "tokens": total_tokens, "k": k, "dir": str(out), "stats": stats, "notes": notes, "stopped": False, "reused": False, "coverage": cov}
    if hold:
        result["holdout"] = {"rows": len(hold), "tokens": sum(len(r["input_ids"]) for r in hold)}
    rows_before = writer.done_rows
    target_tokens = sum(len(r["input_ids"]) for r in rows[:target]) if target < len(rows) else total_tokens

    def counts() -> dict:
        got = rows[:min(writer.done_rows, len(rows))]
        gen = [r for r in got if r.get("source") == "general"]
        return {"done_rows": writer.done_rows, "general_rows_done": len(gen), "general_tokens_done": sum(len(r["input_ids"]) for r in gen), "rows_before": rows_before}

    if writer.done_rows >= target:
        protocol.log("The teacher's saved answers are already complete for this data." if target >= len(rows) else "The teacher has already answered the rows of this round.")
        result.update(reused=True, bytes=_dir_bytes(writer.dir), seconds=round(time.monotonic() - started, 1), **counts())
        protocol.result(result)
        return 0
    need = (sum(len(r["input_ids"]) for r in rows[:target]) - writer.tokens) * (k * 6 + 4)
    free = shutil.disk_usage(writer.dir)[2]
    if free < need * 1.05:
        protocol.error("disk_full", f"The teacher's answers need {need / 1e9:.1f} GB in {out}; {free / 1e9:.1f} GB free.")
        return 1
    if veng:
        outcome, note = _vllm_teacher(cfg, writer, rows, out, meta, teacher_dir, k, max_seq, target, stop_file, result)
        if outcome == "done":
            result.update(bytes=_dir_bytes(writer.dir), seconds=round(time.monotonic() - started, 1), **counts())
            protocol.result(result)
            return 0
        if outcome == "stopped":
            result.update(stopped=True, bytes=_dir_bytes(writer.dir), seconds=round(time.monotonic() - started, 1), **counts())
            protocol.result(result)
            return 0
        # the vLLM teacher failed (and was not required): the standard teacher answers the rest, into a cache of ITS OWN identity (never mixed with the vLLM rows)
        writer.close()
        meta = cache_meta(cfg, rows, stats, student, k, max_seq, teacher_dir)
        writer = open_writer(meta, None)
        rows_before = writer.done_rows
        protocol.log(note)
        result["notes"].append(note)
        if writer.done_rows >= target:
            result.update(reused=True, bytes=_dir_bytes(writer.dir), seconds=round(time.monotonic() - started, 1), **counts())
            protocol.result(result)
            return 0
    est = sum(r.nbytes for r in safetensors_io.list_tensors(teacher_dir).values())
    mode = reshapejob._choose_load(cfg["teacher"].get("load", "auto"), est)
    protocol.progress("cache", 0.0, f"Loading the original model ({mode})")
    model = reshapejob._load(teacher_dir, mode, "load_teacher")
    from . import speed
    speed.report_kernels(model)
    from .data import text_tokenizer
    inner = text_tokenizer(tok)
    pad_id = inner.pad_token_id if inner.pad_token_id is not None else (inner.eos_token_id or 0)
    batch, chunk = max(1, int(cfg.get("batch", 4))), int(cfg.get("chunk", 256))
    budget = max(0, int(cfg.get("teacher_batch_tokens") or 0))          # 0 / absent: today's fixed number of rows per forward pass; the cache's rows, identity and rows_key do not depend on it
    window_rows = max(WINDOW_ROWS, batch)
    held = _read_holdout_meta(out)
    if held is not None and held.get("teacher_loss") is None and (out / "holdout.npz").is_file():
        # the original's own error on the held-out text, measured while it is loaded: what the student's number is compared with
        try:
            hb = reshapejob._pad_batches([{"input_ids": r["input_ids"]} for r in load_rows(out, "holdout")], 4, pad_id)
            held["teacher_loss"] = calib_loss(model, hb, chunk)
            (out / "holdout.json").write_text(json.dumps(held), encoding="utf-8")
            protocol.log(f"The original's error on {held['rows']} held-out general rows: {held['teacher_loss']:.3f}.")
        except Exception as exc:             # noqa: BLE001 - a measurement on the side must never stop the teacher pass
            protocol.log(f"The original's error on the held-out rows was not measured: {type(exc).__name__}: {exc}")
    t0 = last_log = time.monotonic()
    seen_tokens = 0
    tokens_before = writer.tokens
    try:
        i = writer.done_rows
        if i:
            protocol.log(f"Continuing the teacher's answers after row {i} of {len(rows)}.")
        while i < target:
            if stop_file.exists():
                result["stopped"] = True
                break
            # a window of rows is answered longest first (batches of similar length waste little on padding) and written in the order of the rows
            window = rows[i:min(i + window_rows, target)]
            answers: dict = {}
            order = length_order(window)
            pos, cut_short = 0, False
            while pos < len(order):
                if stop_file.exists():
                    cut_short = True
                    break
                if budget > 0:
                    idx = order[pos:pos + rows_in_budget([len(window[j]["input_ids"]) for j in order[pos:pos + TEACHER_MAX_ROWS]], budget)]
                else:
                    idx = order[pos:pos + batch]
                try:
                    got = _teacher_answers(model, [window[j] for j in idx], k, chunk, pad_id)
                except RuntimeError as exc:
                    if not reshapejob._is_oom(exc) or len(idx) == 1:
                        raise
                    reshapejob._free()
                    if budget > 0:
                        budget = shrink_budget(budget, len(window[idx[0]]["input_ids"]))
                        protocol.log(f"Out of memory; the original model takes {'about half as many tokens' if budget else 'one text'} per pass.")
                        if budget == 0:
                            batch = 1
                    else:
                        protocol.log("Out of memory; the original model answers one text at a time.")
                        batch = 1
                    continue
                for j, a in zip(idx, got):
                    answers[j] = a
                    seen_tokens += len(window[j]["input_ids"])
                pos += len(idx)
                protocol.progress("cache", (i + len(answers)) / target, f"The original model answers your texts: {i + len(answers)} of {target}",
                                  protocol.make_detail("tokens", min(tokens_before + seen_tokens, target_tokens), target_tokens, t0, rate_unit="tokens/s", done_at_start=tokens_before))
                if time.monotonic() - last_log >= SPEED_LOG_EVERY:
                    last_log = time.monotonic()
                    protocol.log(f"cache_speed: {seen_tokens / max(last_log - t0, 1e-9):,.0f} tokens/s ({seen_tokens:,} tokens so far; this row: {len(window[idx[0]]['input_ids']):,} tokens)")
            # the rows that were answered, from the first one on without a gap, go to the cache in their own order
            written = 0
            while written < len(window) and written in answers:
                writer.add_row(*answers[written])
                written += 1
                if stop_file.exists():                  # the Stop button was pressed while the cache was written (a part was saved just now)
                    break
            i += written
            if cut_short:
                result["stopped"] = True
                break
    finally:
        writer.close()
        del model
        reshapejob._free()
    tps = seen_tokens / max(time.monotonic() - t0, 1e-9)
    if seen_tokens:
        protocol.log(f"cache_speed: {tps:,.0f} tokens/s ({seen_tokens:,} tokens answered in {time.monotonic() - t0:,.0f} s)")
        answered = max(1, writer.done_rows - rows_before)
        protocol.timing("teacher_pass", time.monotonic() - t0, tokens=seen_tokens, tokens_per_s=round(tps, 1), mean_row_tokens=round(seen_tokens / answered, 1), rows=answered)
    result.update(bytes=_dir_bytes(writer.dir), seconds=round(time.monotonic() - started, 1), tokens_per_s=round(tps, 1), **counts())
    protocol.result(result)
    return 0


def _vllm_teacher(cfg: dict, writer, rows: list, out, meta: dict, teacher_dir, k: int, max_seq: int, target: int, stop_file, result: dict):
    """The vLLM teacher answers the rows after `writer.done_rows` up to `target` (vllmcache.run_chunk). ("done" | "stopped" | "fallback", note). A failure is a "fallback" with a plain
    note (the heal goes on with the standard teacher) unless the job says `engine.required`; then it is an error. The record of what happened goes to `result["teacher_engine"]`."""
    from . import vllmcache
    eng = cfg.get("engine") or {}
    check = eng.get("teacher_check") or {}
    rec = {"engine": meta.get("engine"), "teacher_quant": meta.get("teacher_quant"), "used": True, "fallback": False, "teacher_check": check.get("verdict")}
    result["teacher_engine"] = rec
    problem = None
    if str(check.get("verdict") or "").lower() == "different":
        problem = "The check of the vLLM teacher says it is a different teacher than the standard one, so it is not used."
    else:
        if not check.get("verdict"):
            protocol.log("The vLLM teacher was not checked against the standard teacher yet (job vllm_teacher_check): its answers are used as they are.")
            rec["unchecked"] = True
        try:
            r = vllmcache.run_chunk(cfg, writer, out, meta, cfg["teacher"]["repo"], teacher_dir, writer.done_rows, target, k, max_seq, stop_file)
            rec.update({kk: r.get(kk) for kk in ("startup_seconds", "load_seconds", "tokens", "tokens_per_s", "route", "reason", "load_phases", "timings", "stage_seconds", "rows")})
            writer.flush()
            result["notes"] += r.get("notes") or []
            if r.get("stopped"):
                return "stopped", ""
            if writer.done_rows < target:
                problem = f"vLLM stopped after {writer.done_rows} of {target} rows."
            else:
                if r.get("tokens_per_s"):
                    result["tokens_per_s"] = r["tokens_per_s"]
                return "done", ""
        except (vllmcache.VllmCacheError, vllmcache.vllmengine.EngineError) as exc:
            problem = str(exc)
    rec.update(used=False, fallback=True, problem=problem)
    if eng.get("required"):
        raise ReshapeError(f"vLLM was required for the teacher's answers and it did not work: {problem}" + ("" if vllmcache.REMEDY_OFF in problem else " " + vllmcache.REMEDY_OFF))
    return "fallback", f"vLLM could not make the teacher's answers ({problem}) The standard teacher makes them instead."


def _dir_bytes(folder) -> int:
    return sum(f.stat().st_size for f in Path(folder).glob("*") if f.is_file())


# ============================================================================================================ the student

DEFAULT_HYPER = {"rank": 64, "alpha": 64, "lr": 1e-4, "max_seq": 2048, "batch": 4, "grad_accum": 8, "warmup": 30, "weight_decay": 0.0, "scheduler": "cosine",
                 "seed": 3407, "save_steps": 50, "logging_steps": 1, "tokens": 20_000_000}


def _student_config(student_dir) -> dict:
    try:
        conf = json.loads((Path(student_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return conf.get("text_config") or conf


def _student_b(student_dir) -> float:
    """Billions of parameters of the student, from its weight files (2 bytes each); 0 when unknown."""
    try:
        return sum(r.nbytes for r in safetensors_io.list_tensors(student_dir).values()) / 2e9
    except Exception:                       # noqa: BLE001 - an unreadable folder is reported by the loader later
        return 0.0


def resolve_hyper(cfg: dict, student_dir) -> dict:
    """The training settings: the built-in ones, then the automatic ones for the student's size (healmem.auto_defaults: a longer sequence, a bigger LoRA and a
    bigger batch only for small students), then what the job names. `batch` named without `grad_accum` keeps 32 sequences per update. Also sets
    "student_b" and "checkpointing" (the person's "auto" is decided after the model is loaded, see _decide_checkpointing)."""
    user = dict(cfg.get("hyper") or {})
    sb = float(user.get("student_b") or 0) or _student_b(student_dir)
    auto = healmem.auto_defaults(sb)
    h = {**DEFAULT_HYPER, "max_seq": auto["max_seq"], "rank": auto["rank"], "alpha": auto["rank"], "batch": auto["batch"], "grad_accum": auto["grad_accum"],
         "chunk": auto["chunk"]}
    if "rank" in user and "alpha" not in user:
        user["alpha"] = user["rank"]
    if "batch" in user and "grad_accum" not in user:
        user["grad_accum"] = max(1, -(-healmem.EFFECTIVE_BATCH // max(1, int(user["batch"]))))
    h.update({k: v for k, v in user.items() if v is not None})
    h["student_b"] = sb
    return h


def _decide_checkpointing(h: dict, student_dir, free_gb) -> bool:
    """checkpointing "on" / "off" / True / False as the job says; "auto" (or missing): healmem.checkpointing_auto from the activation estimate."""
    v = h.get("checkpointing", "auto")
    if v in (True, "on", "true", "yes"):
        return True
    if v in (False, "off", "false", "no"):
        return False
    c = _student_config(student_dir)
    layers, hidden = int(c.get("num_hidden_layers") or 0), int(c.get("hidden_size") or 0)
    act = healmem.activation_gb(layers, hidden, int(h["max_seq"]), int(h["batch"]), False) if layers and hidden else 0.0
    return healmem.checkpointing_auto(float(h.get("student_b") or 0), act, free_gb)


def _apply_checkpointing(model, on: bool) -> None:
    """Switches gradient checkpointing on (the Hugging Face way: not reentrant, the inputs require grads so that the LoRA layers get gradients) or off. The model
    may already be switched on by the loader (LoRA and QLoRA on a graphics card), so "off" has to be done explicitly."""
    if on:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        try:
            model.enable_input_require_grads()
        except Exception:                   # noqa: BLE001 - models without an input embedding hook; the non-reentrant mode does not need it
            pass
    else:
        model.gradient_checkpointing_disable()
    conf = getattr(model, "config", None)
    if conf is not None:
        conf.use_cache = False


def _trim() -> None:
    """Gives cached, unused graphics memory back (after saving a checkpoint or a measurement), so that the next step starts from a clean pool."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                       # noqa: BLE001
        pass


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
        parts = [topk_from_logits(h[a:a + chunk].to(weight.dtype) @ weight.t(), k) for a in range(0, h.shape[0], chunk)]
    return tuple(torch.cat([p[j] for p in parts]).to(ids.device) for j in range(3))


def student_loss(model, inputs, targets_fn, alpha: float, chunk: int, checkpoint_head: bool = True):
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
    kd, n, ce, n_lab = chunked_kd_loss(h, weight, targets, torch.ones(h.shape[0], device=h.device), chunk, labels, lab_mask, checkpoint_head)
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

        def create_scheduler(self, num_training_steps, optimizer=None):
            rl = getattr(self, "round_lr", None)
            if not rl:
                return super().create_scheduler(num_training_steps, optimizer)
            from torch.optim.lr_scheduler import LambdaLR
            if self.lr_scheduler is None:       # LambdaLR saves only its step counter, so a resumed round gets the same rates (the function is rebuilt from the job)
                self.lr_scheduler = LambdaLR(optimizer if optimizer is not None else self.optimizer, lambda st: round_lr_factor(st + rl["shift"], rl["s_start"], rl["steps"], rl["index"], rl["warmup"], rl["sch"]))
                self._created_lr_scheduler = True
            return self.lr_scheduler

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch
            d = self.distill
            loss, kd, ce = student_loss(model, inputs, d["targets"], d["alpha"], d["chunk"], d.get("head_checkpoint", True))
            if not torch.isfinite(loss):
                raise ReshapeError(UNSTABLE)
            d["kd"].append(float(kd))
            d["ce"].append(float(ce))
            d["tokens"] += int(inputs["attention_mask"].sum())
            return (loss, None) if return_outputs else loss
    return DistillTrainer


def _held_loss(fn):
    """The loss on the held-out rows (None without them, or when the measurement fails: it never stops the healing)."""
    if fn is None:
        return None
    try:
        return fn()
    except Exception as exc:             # noqa: BLE001
        protocol.log(f"The check on the held-out text was skipped: {type(exc).__name__}: {exc}")
        return None


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
                        logits = (hidden[i, a:a + c].to(weight.dtype) @ weight.t()).float()
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

        def __init__(self, *args, eval_fn=None, eval_every: int = 0, distill=None, lr_at=None, save_at: int = 0, hold_fn=None, **kw):
            super().__init__(*args, **kw)
            self.eval_fn, self.eval_every, self.distill = eval_fn, eval_every, distill
            self.hold_fn, self.hold = hold_fn, []
            self.lr_at, self.save_at = lr_at, save_at       # rounds: the rate for a step (set at the start of a resumed round) and the last step of the round (always saved)
            self.history: list = []          # (step, kd, ce)
            self.calib: list = []            # (step, loss)
            self._seen = 0

        def on_log(self, args, state, control, logs=None, **kw):
            opt = kw.get("optimizer")
            if self.lr_at is not None and opt is not None and logs and "loss" in logs:
                logs["learning_rate"] = float(opt.param_groups[0]["lr"])     # a round: the rate the step really used
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
            _trim()
            if v is not None:
                self.calib.append((step, v))
                hv = _held_loss(self.hold_fn)
                if hv is not None:
                    self.hold.append((step, hv))
                protocol.emit("distill_eval", step=step, calib_loss=v, **({"holdout_loss": hv} if hv is not None else {}))

        def on_save(self, args, state, control, **kw):
            _trim()                                     # a checkpoint was just written: give the unused cached memory back

        def on_step_begin(self, args, state, control, **kw):
            super().on_step_begin(args, state, control, **kw)
            opt = kw.get("optimizer")
            if self.lr_at is not None and opt is not None:
                # a round sets the rate of EVERY step itself (the function of the global step): whatever the scheduler or a resumed optimizer state says, the step uses this one
                lr = self.lr_at(state.global_step)
                for g in opt.param_groups:
                    g["lr"] = lr

        def on_step_end(self, args, state, control, **kw):
            super().on_step_end(args, state, control, **kw)
            if self.save_at and state.global_step >= self.save_at:
                control.should_save = True
            if self.eval_every and not self.stopped and state.global_step % self.eval_every == 0:
                self._evaluate(self.offset + state.global_step)
    return HealCallback


BENCH_STEPS, BENCH_WARMUP = 12, 3


def _bench_targets(k: int, vocab: int, seed: int):
    """Seeded synthetic teacher answers with the shapes (and so the cost) of real ones: targets_fn(inputs) -> (ids [n, k], logp [n, k], resid [n]) for the real positions."""
    import torch
    gen = torch.Generator().manual_seed(seed)

    def targets(inputs):
        n = int(inputs["attention_mask"].sum())
        ids = torch.randint(0, vocab, (n, k), generator=gen)
        logp = torch.log_softmax(torch.randn(n, k, generator=gen), dim=-1) + math.log(0.9)
        return ids, logp, torch.full((n,), 0.1)
    return targets


def run_bench(cfg: dict) -> int:
    """job distill with `bench` (see `_run_bench`); with `bench.out_dir` the heal speed report is written on EVERY end, a failure included."""
    box: dict = {}
    try:
        return _run_bench(cfg, box)
    except BaseException as exc:        # noqa: BLE001 - the report first, then the failure goes on its way
        if box.get("write"):
            box["write"]("failed", f"{type(exc).__name__}: {protocol.scrub(str(exc))[:300]}")
        raise


def _run_bench(cfg: dict, box: dict) -> int:
    """job distill with "bench": {"steps": 12, "warmup": 3, "targets": "cache" | "synthetic"}: the speed of the healing step at the chosen settings, nothing is saved (no checkpoint,
    no adapter, no evaluation, no output folder). Real rows of the project at the chosen sequence length; the teacher's answers from the complete cache when it is there and asked
    for, else seeded synthetic ones with the same shapes (no teacher is loaded). Plain training steps (forward, the KL loss, backward, the optimizer): the first `warmup` are not
    timed. Emits timing `heal_bench` {tokens_per_s, peak_vram_mb, chunk, head_checkpoint, student_b} and the same numbers as the result. A card that runs out of memory ends the
    job with the error distill_oom (the caller reports that setting as failed and goes on with the next)."""
    protocol.reserve_stdout()
    healmem.set_alloc_conf()
    from . import backends
    backends.prepare("hf")
    import torch
    from transformers import set_seed
    from . import data
    b = cfg["bench"] if isinstance(cfg.get("bench"), dict) else {}
    steps, warm = max(1, int(b.get("steps", BENCH_STEPS))), max(0, int(b.get("warmup", BENCH_WARMUP)))
    from . import healspeed
    report = {"created": healspeed.now(), "status": "running", "environment": healspeed.environment_facts(), "windows_hints": b.get("windows_hints") or [], "notes": []} if b.get("out_dir") else None

    def write_report(status: str, error: str | None = None) -> None:
        """the heal speed report (only when the Heal card's "Measure heal speed" asked for it): written on EVERY end"""
        if report is not None:
            report.update(status=status, error=error)
            try:
                protocol.add_extra("heal_speed_report", healspeed.write_report(b["out_dir"], report))
            except Exception as exc:        # noqa: BLE001 - the report must never hide the job's own result
                protocol.log(f"The heal speed report could not be written: {type(exc).__name__}: {exc}")
    d = cfg.get("distill") or {}
    student_dir = cfg["model"]["repo"]
    h = resolve_hyper(cfg, student_dir)
    k, chunk = int(d.get("k", 32)), int(d.get("chunk") or h["chunk"])
    head_ckpt = str(d.get("head_checkpoint", "on")).lower() not in ("off", "false", "0")
    set_seed(h["seed"])
    method = cfg.get("method", "lora")
    lcfg = {"model": cfg["model"], "method": method, "hyper": h, "kbit_upcast": method == "qlora" and bool(d.get("kbit_upcast", False))}
    protocol.progress("bench", 0.0, "Loading the model")
    model, tokenizer, _ = backends.load(lcfg, "hf")
    cuda = torch.cuda.is_available()
    from . import speed
    kern = speed.report_kernels(model)
    box["write"] = write_report
    if report is not None:
        report["kernels"] = kern
    free_gb = torch.cuda.mem_get_info()[0] / 1e9 if cuda else None
    _apply_checkpointing(model, _decide_checkpointing(h, student_dir, free_gb))
    reader, use = None, "synthetic"
    cache = d.get("cache")
    if b.get("targets", "cache") == "cache" and cache:
        idx = read_cache_index(cache)
        if idx and idx.get("complete") and idx.get("max_len") == int(h["max_seq"]):
            try:
                rows, reader, use = load_rows(cache), CacheReader(cache), "cache"
            except ReshapeError:
                reader = None
    if reader is None:
        rows, _ = build_rows(tokenizer, {"train": cfg["dataset"]["train"], "tokens": int(h["max_seq"]) * int(h["batch"]) * (steps + warm), "general": None},
                             cfg["model"].get("family", "generic"), int(h["max_seq"]), int(h["seed"]))
    if not rows:
        raise ReshapeError("There is nothing to measure the speed on: the training data is empty.")
    inner = data.text_tokenizer(tokenizer)
    pad_id = inner.pad_token_id if inner.pad_token_id is not None else (inner.eos_token_id or 0)
    collate = _distill_collator(pad_id, reader)
    batch = max(1, int(h["batch"]))
    vocab = int(model.get_output_embeddings().weight.shape[0])
    synth = _bench_targets(k, vocab, int(h["seed"]))
    targets = (lambda inp: (inp["t_ids"], inp["t_logp"], inp["t_resid"])) if reader is not None else synth
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ReshapeError("The student has no trainable weights, so its speed cannot be measured.")
    opt = torch.optim.AdamW(params, lr=1e-6)
    model.train()
    dev = next(model.parameters()).device
    tokens, started = 0, None
    try:
        for step in range(warm + steps):
            if step == warm:
                if cuda:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
            sel = [{**rows[(step * batch + j) % len(rows)], "row": (step * batch + j) % len(rows)} for j in range(batch)]
            inputs = {kk: v.to(dev) for kk, v in collate(sel).items()}
            loss, _, _ = student_loss(model, inputs, targets, 1.0, chunk, head_ckpt)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step >= warm:
                tokens += int(inputs["attention_mask"].sum())
            protocol.progress("bench", (step + 1) / (warm + steps), f"Step {step + 1} of {warm + steps}")
        if cuda:
            torch.cuda.synchronize()
    except RuntimeError as exc:
        if healmem.is_oom(exc):
            write_report("failed", "out of memory")
            protocol.error("distill_oom", f"The graphics card ran out of memory at sequence length {h['max_seq']}, batch {batch}, KL chunk {chunk}, output-layer checkpointing {'on' if head_ckpt else 'off'}.")
            return 1
        raise
    secs = max(time.monotonic() - started, 1e-6)
    peak = round(torch.cuda.max_memory_allocated() / 1e6) if cuda else None
    res = {"tokens_per_s": round(tokens / secs, 2), "peak_vram_mb": peak, "chunk": chunk, "head_checkpoint": head_ckpt, "student_b": h["student_b"], "steps": steps, "warmup": warm,
           "tokens": tokens, "seconds": round(secs, 2), "targets": use, "max_seq": int(h["max_seq"]), "batch": batch, "stopped": False}
    protocol.timing("heal_bench", secs, tokens=tokens, tokens_per_s=res["tokens_per_s"], peak_vram_mb=peak, chunk=chunk, head_checkpoint=head_ckpt, student_b=h["student_b"])
    if report is not None:
        report["bench"] = {k_: res[k_] for k_ in ("tokens_per_s", "peak_vram_mb", "chunk", "max_seq", "batch", "student_b", "targets")}
        try:
            if b.get("profile"):
                protocol.progress("bench", 0.97, "Profiling two steps")
                res["heal_profile"] = report["heal_profile"] = healspeed.profile_steps(model, rows, collate, targets, chunk, head_ckpt, opt, batch, dev, 2, (warm + steps) * batch)
            if b.get("microbench"):
                report["microbench"] = healspeed.gdn_microbench()
            if b.get("teacher_bench") and (d.get("teacher") or {}).get("repo"):
                del opt
                model = None
                import gc
                gc.collect()
                if cuda:
                    torch.cuda.empty_cache()
                from . import reshapejob
                tdir = reshapejob._resolve_source(d["teacher"]["repo"])
                tmodel = reshapejob._load(tdir, reshapejob._choose_load(d["teacher"].get("load", "auto"), sum(r.nbytes for r in safetensors_io.list_tensors(tdir).values())), "load_teacher")
                try:
                    report["teacher_bench"] = healspeed.teacher_bench(rows, lambda br: _teacher_answers(tmodel, br, k, 256, pad_id), sync=torch.cuda.synchronize if cuda else (lambda: None))
                finally:
                    del tmodel
                    reshapejob._free()
        except Exception as exc:        # noqa: BLE001 - the plain speed number is still a result; the report says what failed
            report["notes"].append(f"An extra measurement failed: {type(exc).__name__}: {protocol.scrub(str(exc))[:240]}")
        write_report("done")
    protocol.progress("bench", 1.0, "")
    protocol.result(res)
    return 0


def run(cfg: dict) -> int:
    """job distill: {"model": {"repo": the thinned model's folder, "family"}, "method": "lora|qlora", "backend": "hf", "dataset": {"train", "general": {...}},
    "hyper": {rank, alpha, lr, max_seq, batch, grad_accum, warmup, weight_decay, scheduler, seed, save_steps, logging_steps, tokens, max_steps?,
              checkpointing: "auto|on|off", optimizer: "auto|adamw_8bit|adamw", student_b?},
    "distill": {"mode": "cache|online", "cache": folder, "teacher": {"repo", "load"}, "alpha": 1.0, "k": 32, "chunk": 256, "eval_every": 50,
                "kbit_upcast": false (QLoRA only: true = PEFT's 32-bit copy of the embeddings and the output layer; student-side numerics, not part of any cache key),
                 "head_checkpoint": "on" (default) | "off" (the output layer's logits of every chunk are kept for the backward pass instead of being made again: faster, the same loss,
                 about tokens x vocab x 10 bytes more memory; student-side only, not part of any cache key; the out-of-memory ladder turns it back on first)},
    "eval": {"calibration": {...}, "teacher_loss", "slim_loss"} | None, "out": folder, "resume": bool, "stop_file": path,
    "rounds": {"index", "round_tokens", "general_from", "general_to", "s_start", "schedule": {...}} (optional: this job trains ONE round of a heal that is trained in rounds,
               see docs/HEAL_ROUNDS.md; cache mode only)}.
    Settings the job leaves out are chosen by the student's size (resolve_hyper). When the graphics card runs out of memory the training is tried again, at
    most healmem.MAX_RETRIES times, from the last checkpoint with the next safer setting (healmem.next_safer); the final failure is a plain message.
    Writes the LoRA to out/adapter (a round also to out/rounds/r<n>/adapter). Result: steps, tokens_seen, kd_first, kd_last, calib_loss_before / _after, teacher_calib_loss,
    adapter_dir, stopped, settings, oom_retries (and `round` with rounds)."""
    if cfg.get("bench"):
        return run_bench(cfg)                 # the speed test: nothing is saved, no folder is made
    protocol.reserve_stdout()
    healmem.set_alloc_conf()
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

    student_dir = cfg["model"]["repo"]
    h = resolve_hyper(cfg, student_dir)
    k, alpha = int(d.get("k", 32)), float(d.get("alpha", 1.0))
    chunk = int(d.get("chunk") or h["chunk"])
    set_seed(h["seed"])
    protocol.log(f"Healing: {mode} mode, precision {backends.precision_name()}, LoRA rank {h['rank']}, learning rate {h['lr']}, KL loss weight {alpha}.")
    reader = teacher = None
    rd = cfg.get("rounds") if isinstance(cfg.get("rounds"), dict) else None
    if rd and mode != "cache":
        raise ReshapeError("Healing in rounds needs the teacher's saved answers (cache mode).")
    if mode == "cache":
        reader = CacheReader(d["cache"])
        idx = reader.index
        rows = load_rows(d["cache"])
        if (idx["tokenizer_hash"] != tokenizer_hash(student_dir) or idx["max_len"] != h["max_seq"] or idx["rows"] != len(rows)
                or data_hash(rows) != idx["data_hash"]):
            raise ReshapeError("The saved teacher answers belong to other data; they will be made again.")
        if rd:
            if not 0 <= int(rd["general_from"]) <= int(rd["general_to"]) <= len(rows):
                raise ReshapeError("The rows of this round do not exist in the saved teacher answers; they will be made again.")
            if reader.done_rows < int(rd["general_to"]):
                raise ReshapeError(f"The teacher has answered {reader.done_rows} rows, but this round needs the first {int(rd['general_to'])}. The answers for this round are made first.")
        elif reader.done_rows < idx["rows"]:
            raise ReshapeError("The saved teacher answers belong to other data; they will be made again.")
        share = float(((cfg.get("dataset") or {}).get("general") or {}).get("share", 0.0))
        g_cfg = (cfg.get("dataset") or {}).get("general")
        # the rows come from the cache itself, so the training run's seed cannot change them (see generaltext.same_general)
        if not generaltext.same_general(idx.get("general_key"), g_cfg, share):
            raise ReshapeError("The saved teacher answers belong to other data (another general text); they will be made again.")
    else:
        share = float(((cfg.get("dataset") or {}).get("general") or {}).get("share", 0.0))
    method = cfg["method"]
    upcast = method == "qlora" and bool(d.get("kbit_upcast", False))     # the old 32-bit copy of the embeddings and the output layer (QLoRA only); off for healing
    lcfg = {"model": cfg["model"], "method": method, "hyper": h, "kbit_upcast": upcast}
    ckdir = out / "checkpoints"
    start = {"mode": "continue", "gs_start": 0, "init_adapter": None, "note": None}
    r_index = r_start = 0
    if rd:
        r_index, r_start = int(rd["index"]), int(rd["s_start"])
        start = round_start(ckdir, r_index, r_start, out)
        if start["note"]:
            protocol.log(start["note"])
        if start["init_adapter"]:
            lcfg["init_adapter"] = start["init_adapter"]
    with protocol.timed("load_student", method=method):
        model, tokenizer, _ = backends.load(lcfg, "hf")
    from . import speed
    speed.report_kernels(model)
    if mode == "online":
        dcfg = {"train": cfg["dataset"]["train"], "tokens": int(h["tokens"]), "general": (cfg["dataset"] or {}).get("general")}
        rows, stats = build_rows(tokenizer, dcfg, cfg["model"].get("family", "generic"), int(h["max_seq"]), int(h["seed"]))
        if stats["general_error"]:
            share = 0.0
        t_dir = reshapejob._resolve_source(d["teacher"]["repo"])
        _require_same_ids(t_dir, student_dir)
        est = sum(r.nbytes for r in safetensors_io.list_tensors(t_dir).values())
        teacher = reshapejob._load(t_dir, reshapejob._choose_load(d["teacher"].get("load", "auto"), est), "load_teacher")
        for p in teacher.parameters():
            p.requires_grad_(False)
    elif not any(r.get("source") == "general" for r in rows):
        share = 0.0
    sch = {**DEFAULT_ROUND_SCHEDULE, **((rd or {}).get("schedule") or {})}
    if rd:
        order = round_order(rows, int(rd["general_from"]), int(rd["general_to"]), int(rd["round_tokens"]), share, int(h["seed"]) + r_index)
    else:
        order = training_order(rows, int(h["tokens"]), share, int(h["seed"]))
    if not order:
        raise ReshapeError("There is nothing to learn from: the training data is empty.")

    def make_ds(sel: list):
        return Dataset.from_list([{"input_ids": rows[i]["input_ids"], "completion_mask": rows[i]["completion_mask"], "row": i} for i in sel])

    ds = None if rd else make_ds(order)
    per_step = max(1, int(h["batch"]) * int(h["grad_accum"]))               # the sequences per update: it stays the same when a retry halves the batch
    steps = int(h["max_steps"]) if h.get("max_steps") else max(1, -(-len(order) // per_step))       # a round: the steps of this round
    tokens_planned = sum(len(rows[i]["input_ids"]) for i in order)
    gs_start = int(start["gs_start"])
    shift = r_start - gs_start                                              # HF's step counter of a fresh round starts at 0; the round's rates and the reports use global steps
    max_steps = gs_start + steps if rd else steps
    protocol.log(f"{len(order)} rows, {tokens_planned:,} tokens in {steps} steps ({sum(1 for i in order if rows[i].get('source') == 'general')} rows of general text)"
                 + (f"; round {r_index}, global steps {r_start} to {r_start + steps}." if rd else "."))

    inner = data.text_tokenizer(tokenizer)
    pad_id = inner.pad_token_id if inner.pad_token_id is not None else inner.eos_token_id
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.get_device_capability(0)[0] >= 8
    want = str(h.get("optimizer") or "auto").lower()
    optim = "adamw_torch" if want in ("adamw", "adamw_torch") or not cuda else "adamw_8bit"
    if optim == "adamw_8bit":
        try:
            import bitsandbytes  # noqa: F401
        except Exception:
            optim = "adamw_torch"
            if want in ("adamw_8bit", "8bit"):
                protocol.log("The 8-bit optimizer (bitsandbytes) is not available here; using the standard AdamW (it needs more memory).")
    fields = getattr(SFTConfig, "__dataclass_fields__", {})
    ev = cfg.get("eval") or None
    batches = reshapejob._calibration(tokenizer, ev["calibration"], cfg["model"].get("family", "generic")) if ev and ev.get("calibration") else None
    hold_batches, hold_meta = None, None
    if ev and isinstance(ev.get("holdout"), dict) and mode == "cache" and (Path(d["cache"]) / "holdout.npz").is_file():
        # general rows that were read past the ones the heal learns from (the teacher pass saved them): only their loss is measured, they are never trained on
        try:
            hold_meta = _read_holdout_meta(d["cache"]) or {}
            hold_batches = reshapejob._pad_batches([{"input_ids": r["input_ids"]} for r in load_rows(d["cache"], "holdout")], 4, pad_id) or None
        except ReshapeError:
            hold_batches = None
    holder = {"model": model}
    del model                                                # only the holder keeps the model, so that a retry can really free it
    Heal = _callback_class()
    hist_all: list = []
    calib_all: list = []
    done = {"before": ev.get("slim_loss") if ev else None, "t0": time.time(), "tokens": 0, "hbefore": None}
    hold_all: list = []
    free_gb = None
    if cuda:
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
    settings = {"checkpointing": _decide_checkpointing(h, student_dir, free_gb), "batch": int(h["batch"]), "grad_accum": int(h["grad_accum"]), "chunk": chunk,
                "max_seq": int(h["max_seq"]), "rank": int(h["rank"]), "method": method, "upcast": upcast}
    head_ckpt = str(d.get("head_checkpoint", "on")).lower() not in ("off", "false", "0")        # job key distill.head_checkpoint "on" (default) | "off"
    if not head_ckpt:
        settings["head_checkpoint"] = False                  # only an explicit "off" is part of the settings: a default job's settings are unchanged
    protocol.log("Healing settings: " + healmem.describe(settings) + f", optimizer {optim}.")
    protocol.emit("heal_settings", settings=settings, optimizer=optim)

    def attempt(s: dict, n: int):
        if n > 0:                                            # a retry: a fresh model and trainer, from the last checkpoint
            lcfg["kbit_upcast"] = bool(s.get("upcast"))      # the ladder's first step turns the 32-bit embedding copy off
            with protocol.timed("load_student", method=method, retry=True):
                holder["model"] = backends.load(lcfg, "hf")[0]
        mdl = holder["model"]
        ckpts = ckpt.usable(ckdir)       # a checkpoint cut short by a crash is renamed *.broken and left out
        if rd:
            ckpts = [c for c in ckpts if ckpt.step_of(c) <= max_steps]
        resume = (bool(cfg.get("resume")) or n > 0 or (bool(rd) and r_index > 1)) and bool(ckpts)
        done_steps = max(0, ckpt.step_of(ckpts[-1]) - gs_start) if (rd and resume) else 0
        if rd and resume and done_steps >= steps:
            # the round was trained and its checkpoint saved, but the job ended before the adapter was written: the checkpoint's adapter IS the round's adapter
            protocol.log(f"Round {r_index} was already trained (checkpoint {ckpts[-1].name}); its adapter is taken from there.")
            for target_dir in (out / "adapter", out / "rounds" / f"r{r_index}" / "adapter"):
                target_dir.mkdir(parents=True, exist_ok=True)
                for name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"):
                    if (ckpts[-1] / name).is_file():
                        shutil.copy2(ckpts[-1] / name, target_dir / name)
            return {"after": None, "steps": ckpt.step_of(ckpts[-1]), "tokens": 0, "stopped": False, "vram": 0, "target": out / "adapter"}
        _apply_checkpointing(mdl, bool(s["checkpointing"]))
        kwargs = dict(output_dir=str(ckdir), per_device_train_batch_size=int(s["batch"]), gradient_accumulation_steps=int(s["grad_accum"]), learning_rate=float(h["lr"]),
                      warmup_steps=int(h["warmup"]), weight_decay=float(h["weight_decay"]), lr_scheduler_type=h["scheduler"], logging_steps=int(h["logging_steps"]),
                      save_strategy="steps", save_steps=int(h["save_steps"]), save_total_limit=2, optim=optim, seed=int(h["seed"]), report_to="none", disable_tqdm=True,
                      bf16=bf16, fp16=cuda and not bf16, remove_unused_columns=False, gradient_checkpointing=bool(s["checkpointing"]), dataloader_num_workers=0,
                      use_cpu=not cuda, max_steps=max_steps)
        if rd:
            kwargs["ignore_data_skip"] = True               # the round's rows are sliced by this job, not skipped by the trainer
        if s["checkpointing"]:
            kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        kwargs["max_length" if "max_length" in fields else "max_seq_length"] = int(h["max_seq"])
        if "dataset_kwargs" in fields:
            kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
        if "packing" in fields:
            kwargs["packing"] = False
        args = SFTConfig(**kwargs)
        eval_fn = (lambda: calib_loss(holder["model"], batches, int(s["chunk"]))) if batches else None        # noqa: E731
        hold_fn = (lambda: calib_loss(holder["model"], hold_batches, int(s["chunk"]))) if hold_batches else None        # noqa: E731
        state = {"kd": [], "ce": [], "tokens": 0}
        lr_at = (lambda st: float(h["lr"]) * round_lr_factor(st + shift, r_start, steps, r_index, int(h["warmup"]), sch)) if rd else None   # noqa: E731
        live = Heal(str(stop_file), None, [], 0, offset=shift if rd else 0, eval_fn=eval_fn, eval_every=int(d.get("eval_every", 0)), distill=state, lr_at=lr_at, save_at=max_steps if rd else 0, hold_fn=hold_fn)
        trainer_kw = {"model": mdl, "args": args, "train_dataset": make_ds(round_slice(order, done_steps, per_step)) if rd else ds, "data_collator": _distill_collator(pad_id, reader), "callbacks": [live]}
        trainer_kw["processing_class" if "processing_class" in SFTTrainer.__init__.__code__.co_varnames else "tokenizer"] = tokenizer
        trainer = _trainer_class(SFTTrainer)(**trainer_kw)
        trainer.model_accepts_loss_kwargs = False           # the trainer then divides the loss by the gradient accumulation itself
        if rd:
            trainer.round_lr = {"shift": shift, "s_start": r_start, "steps": steps, "index": r_index, "warmup": int(h["warmup"]), "sch": sch}
        targets = (lambda inp: (inp["t_ids"], inp["t_logp"], inp["t_resid"])) if mode == "cache" else (lambda inp: _teacher_targets(teacher, inp["input_ids"], inp["attention_mask"], k, int(s["chunk"])))
        trainer.distill = {**state, "targets": targets, "alpha": alpha, "chunk": int(s["chunk"]), "head_checkpoint": s.get("head_checkpoint", True) is not False}
        live.distill = trainer.distill
        if done["before"] is None and eval_fn is not None and not (rd and r_index > 1):
            with protocol.timed("eval", when="before"):
                done["before"] = eval_fn()
            if done["before"] is not None:
                if done["hbefore"] is None:
                    done["hbefore"] = _held_loss(hold_fn)
                protocol.emit("distill_eval", step=0, calib_loss=done["before"], **({"holdout_loss": done["hbefore"]} if done["hbefore"] is not None else {}))
        if cfg.get("resume") and not ckpts and n == 0:
            protocol.log("No checkpoint to continue from; starting from the beginning.")
        if n > 0:
            at = int(ckpts[-1].name.split("-")[1]) if ckpts else 0
            hist_all[:] = [r for r in hist_all if r[0] <= at]
            calib_all[:] = [r for r in calib_all if r[0] <= at]
            protocol.log(f"Going on from {'step ' + str(at) if ckpts else 'the beginning (no checkpoint was saved yet)'}.")
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        try:
            t_train = time.monotonic()
            tokens_before = trainer.distill["tokens"]
            trainer.train(resume_from_checkpoint=str(ckpts[-1]) if resume else None)
            spent = time.monotonic() - t_train
            seen = trainer.distill["tokens"] - tokens_before
            protocol.timing("train", spent, steps=trainer.state.global_step, tokens=seen, tokens_per_s=round(seen / spent, 1) if spent > 0 and seen > 0 else None)
            after = None
            if eval_fn is not None and not live.stopped:
                with protocol.timed("eval", when="after"):
                    after = eval_fn()
                if after is not None:
                    hafter = _held_loss(hold_fn)
                    if hafter is not None:
                        live.hold.append((trainer.state.global_step, hafter))
                    protocol.emit("distill_eval", step=trainer.state.global_step, calib_loss=after, **({"holdout_loss": hafter} if hafter is not None else {}))
            target = out / "adapter"
            trainer.model.save_pretrained(str(target))
            if rd and not live.stopped:
                trainer.model.save_pretrained(str(out / "rounds" / f"r{r_index}" / "adapter"))
            return {"after": after, "steps": trainer.state.global_step, "tokens": trainer.distill["tokens"], "stopped": live.stopped, "vram": live._vram(), "target": target}
        finally:
            hist_all.extend(live.history)               # also of a failed attempt: the steps up to its last checkpoint count
            calib_all.extend(live.calib)
            hold_all.extend(live.hold)
            del trainer, live

    def free_memory():
        holder["model"] = None
        reshapejob._free()

    def on_retry(n, label, s):
        protocol.log(f"Out of memory: trying again with {label}. The sequences per update stay {int(s['batch']) * int(s['grad_accum'])}; training goes on from the last saved checkpoint.")
        protocol.emit("heal_retry", attempt=n, change=label, settings=s)

    r, used, tried = healmem.run_with_oom_retry(attempt, settings, on_retry=on_retry, free=free_memory)
    hist = hist_all
    gen_info = {k: stats.get(k) for k in ("general_source", "general_rows", "general_tokens", "general_error", "general_error_code", "general_detected")} if mode == "online" else \
        {"general_source": idx.get("general_source"), "general_rows": idx.get("general_rows")}
    cov = stats.get("coverage") if mode == "online" else _read_coverage(d["cache"])
    summary = {"general": gen_info, "backend": "hf", "method": method, "mode": mode, "steps": r["steps"], "total_steps": steps, "tokens_seen": r["tokens"],
               "kd_first": hist[0][1] if hist else None, "kd_last": hist[-1][1] if hist else None, "calib_loss_before": done["before"],
               "calib_loss_after": r["after"] if r["after"] is not None else (calib_all[-1][1] if calib_all else None), "teacher_calib_loss": ev.get("teacher_loss") if ev else None,
               "train_seconds": round(time.time() - done["t0"], 1), "peak_vram_mb": r["vram"], "adapter_dir": str(r["target"]), "stopped": r["stopped"],
               "rows": len(order), "tokens_planned": tokens_planned, "settings": used, "optimizer": optim, "oom_retries": tried}
    if hold_batches:
        # the loss on rows the heal never learned from (the "calibration" loss above is measured on text the heal also trains on, so it reads better than it should)
        summary.update(holdout_rows=(hold_meta or {}).get("rows"), holdout_loss_before=done["hbefore"], holdout_loss_after=hold_all[-1][1] if hold_all else None,
                       teacher_holdout_loss=(hold_meta or {}).get("teacher_loss"))
    if cov:
        summary["coverage"] = cov
    if rd:
        # r["steps"] is HF's step counter: the round's own steps are what it counted since the round began
        done_in_round = max(0, r["steps"] - gs_start)
        summary["steps"] = r_start + done_in_round
        summary["round"] = {"index": r_index, "s_start": r_start, "s_end": r_start + done_in_round, "steps": steps, "tokens": tokens_planned,
                            "calib_loss_after": summary["calib_loss_after"], "complete": done_in_round >= steps and not r["stopped"]}
        if summary["round"]["complete"]:
            rdir = out / "rounds" / f"r{r_index}"
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    free = fsutil.free_bytes(out.parent)
    if free < need * 1.1:
        protocol.error("disk_full", f"There is not enough free space: {need * 1.1 / 1e9:.1f} GB needed in {out.parent}, {free / 1e9:.1f} GB free.")
        return 1
    if out.exists():
        shutil.rmtree(out)
    fsutil.ensure_parent(out)
    fsutil.ensure_parent(final)
    out.mkdir(parents=True)
    last = [-1.0]
    t_merge_start = time.monotonic()

    def on_progress(done, total):
        if total and (done / total - last[0] >= 0.01 or done == total):
            last[0] = done / total
            protocol.progress("merge", done / total, f"Saving the healed model: {done / 1e9:.1f} of {total / 1e9:.1f} GB",
                              protocol.make_detail("GB", done / 1e9, total / 1e9, t_merge_start, rate_unit="GB/s"))

    try:
        clock: dict = {}
        t_merge = time.monotonic()
        res = slice_io.merge_lora(tensors, adapter, out, on_progress=on_progress, should_stop=lambda: stop_file.exists(), clock=clock)
        protocol.timing("merge_read", clock.get("read", 0.0), tensors=res["merged"])
        protocol.timing("merge_compute", clock.get("compute", 0.0), tensors=res["merged"])
        protocol.timing("merge_total", time.monotonic() - t_merge, bytes=need)
        shutil.copy2(base / "config.json", out / "config.json")
        reshapejob._copy_side_files(base, out)
        if (base / "width_kept.npz").is_file():
            shutil.copy2(base / "width_kept.npz", out / "width_kept.npz")
        try:
            meta = json.loads((base / "ndforge_reshape.json").read_text(encoding="utf-8"))
            external = False
        except (OSError, ValueError):                  # a model that was not made on the Slim page (Heal a model): there is nothing to carry over
            meta = {"format": 2, "tool": "ND Lore Forge", "mode": "heal"}
            external = True
        origin = cfg.get("origin")
        if isinstance(origin, dict) and origin.get("repo") and (external or not (meta.get("origin") or {}).get("repo")):
            meta["origin"] = origin                    # the teacher: Step 3 and Step 5 read the model it was healed from here
        if external:
            conf = _student_config(base)
            meta["layers"] = conf.get("num_hidden_layers")
            meta["params"] = sum(math.prod(t.shape) for t in tensors.values())
            meta["healed_from"] = cfg.get("student_label") or str(base)
        s = cfg.get("summary") or {}
        meta["healed"] = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "from": cfg.get("student_label") or (base.name if base.name != "model" else base.parent.name), "merged_weights": res["merged"],
                          "lora_rank": res["rank"], "lora_scaling": res["scaling"], **{k: s.get(k) for k in (
                              "mode", "general", "steps", "tokens_seen", "kd_first", "kd_last", "calib_loss_before", "calib_loss_after", "teacher_calib_loss") if k in s}}
        if cfg.get("recipe"):
            meta["healed"]["recipe"] = str(cfg["recipe"])         # "thinking": the export keeps the model's own chat template (exportjob thinking_default)
        (out / "ndforge_reshape.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        label = cfg.get("student_label") or base.parent.name
        teacher = (meta.get("origin") or {}).get("repo")
        (out / "README.md").write_text(f"---\n{reshapejob._licence_front_matter(base / 'README.md')}---\n# {out.parent.name} (healed)\n\n"
                                       f"{label}, taught by {'its original model' if not external else (teacher or 'another model')} (distillation).\n\n"
                                       f"Made with ND Lore Forge ({'Heal a model' if external else 'Slim or grow a model'}). See ndforge_reshape.json for the details.\n", encoding="utf-8")
        fsutil.replace_retry(out, final)
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
