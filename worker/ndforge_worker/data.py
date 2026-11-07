"""Training data: the project's train.jsonl (world-book texts and practice conversations) as token ids with a completion mask.

A text row (world book): the whole text is learned. A chat row: only the assistant's answer and the end marker are learned; the system and
user turns are context (mask 0), exactly like Unsloth's train_on_responses_only but computed from the template, not from marker strings.
"""
import json
import random

from . import protocol
from .families import family


def read_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError(f"{path}, line {n}: not valid JSON ({exc})") from exc
    return rows


def _inner(tokenizer):
    """Multimodal models hand back a processor; plain tokenization and the chat template live on its tokenizer."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def render_chat(tokenizer, messages: list, fam_name: str):
    """(prompt text, full text) for a [system?, user, assistant] conversation."""
    fam = family(fam_name)
    tok = _inner(tokenizer)
    if messages[-1]["role"] != "assistant":
        raise ValueError("the last message of a practice conversation must be the assistant's answer")
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, **fam["prompt_kwargs"])
    end = fam["end"] if fam["end"] is not None else (tok.eos_token or "")
    return prompt, prompt + messages[-1]["content"] + end


def encode_rows(tokenizer, rows: list, fam_name: str, max_len: int, seed: int = 3407):
    """[{input_ids, completion_mask, kind: "text"|"chat"}] and counts. Rows whose prompt alone does not fit are skipped; longer rows are cut at max_len."""
    tok = _inner(tokenizer)
    bos = getattr(tok, "bos_token", None)
    out, stats = [], {"text": 0, "chat": 0, "skipped_too_long": 0, "cut": 0, "empty": 0}
    for r in rows:
        if r.get("type") == "text":
            text = (r.get("text") or "").strip()
            if not text:
                stats["empty"] += 1
                continue
            ids = tok(text, add_special_tokens=True)["input_ids"]
            mask = [1] * len(ids)
            kind = "text"
        else:
            prompt, full = render_chat(tokenizer, r["messages"], fam_name)
            # the template already writes the beginning-of-sequence token, so none is added twice
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            ids = tok(full, add_special_tokens=False)["input_ids"]
            if ids[: len(p_ids)] != p_ids:
                # the answer's first token merged with the end of the prompt: fall back to a character-based boundary
                p_ids = tok(prompt.rstrip(), add_special_tokens=False)["input_ids"]
                if ids[: len(p_ids)] != p_ids:
                    p_ids = p_ids[: max(0, len(p_ids) - 1)]
            mask = [0] * len(p_ids) + [1] * (len(ids) - len(p_ids))
            kind = "chat"
            if len(p_ids) >= max_len - 1:
                stats["skipped_too_long"] += 1
                continue
        if len(ids) > max_len:
            ids, mask = ids[:max_len], mask[:max_len]
            stats["cut"] += 1
        stats[kind] += 1
        out.append({"input_ids": ids, "completion_mask": mask, "kind": kind})
    random.Random(seed).shuffle(out)
    stats["tokens"] = sum(len(x["input_ids"]) for x in out)
    stats["learned_tokens"] = sum(sum(x["completion_mask"]) for x in out)
    return out, stats


def load_training_rows(tokenizer, path: str, fam_name: str, max_len: int, seed: int):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("The training file is empty. Save a version in Step 2 first.")
    enc, stats = encode_rows(tokenizer, rows, fam_name, max_len, seed)
    if not enc:
        raise ValueError("Nothing in the training file fits the chosen sequence length. Raise the length or shorten the examples.")
    protocol.log(f"Training data: {stats['text']} world-book texts and {stats['chat']} practice conversations, {stats['tokens']:,} tokens "
                 f"({stats['learned_tokens']:,} learned); {stats['cut']} cut to {max_len} tokens, {stats['skipped_too_long']} skipped (too long).")
    return enc, stats
