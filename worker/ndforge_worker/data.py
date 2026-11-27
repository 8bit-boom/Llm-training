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


def text_tokenizer(tokenizer):
    """The text tokenizer inside what a loader handed back. A multimodal model (and Unsloth's wrapper) returns a processor whose `__call__` and
    `apply_chat_template` look for images and fail on plain text; tokenization, the chat template and the pad / eos ids live on its `.tokenizer`.
    A plain tokenizer is returned as it is. Everything that renders or tokenizes text goes through this."""
    return getattr(tokenizer, "tokenizer", tokenizer)


_inner = text_tokenizer         # the old name, kept


def render_chat(tokenizer, messages: list, fam_name: str):
    """(prompt text, full text) for a [system?, user, assistant] conversation."""
    fam = family(fam_name)
    tok = _inner(tokenizer)
    if messages[-1]["role"] != "assistant":
        raise ValueError("the last message of a practice conversation must be the assistant's answer")
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, **fam["prompt_kwargs"]) + fam.get("answer_open", "")
    end = fam["end"] if fam["end"] is not None else (tok.eos_token or "")
    return prompt, prompt + messages[-1]["content"] + end


THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def split_thinking(content: str):
    """(trace, answer) of an assistant text that starts with a reasoning block, as the Qwen template reads it: the text before the first </think> (after the last
    <think> in it), and what follows. (None, content) when there is no closed block."""
    if THINK_CLOSE not in content:
        return None, content
    head, _, answer = content.partition(THINK_CLOSE)
    trace = head.split(THINK_OPEN)[-1].strip()
    return trace, answer.lstrip("\n")


def has_thinking(messages: list) -> bool:
    """True when the last assistant turn carries a reasoning trace that is not empty: inline (<think>...</think>) or in the separate field `reasoning_content`."""
    if not messages or messages[-1].get("role") != "assistant":
        return False
    last = messages[-1]
    r = last.get("reasoning_content")
    if isinstance(r, str) and r.strip():
        return True
    c = last.get("content")
    if isinstance(c, str):
        trace, _ = split_thinking(c)
        return bool(trace)
    return False


def render_chat_full(tokenizer, messages: list, fam_name: str, thinking=None):
    """(prompt text, full text) like render_chat, but a conversation whose last assistant turn has thinking is rendered the way the model writes it with thinking ON:

        ...<|im_start|>assistant\n<think>\n{trace}\n</think>\n\n{answer}<|im_end|>\n      prompt = everything up to and including "assistant\n<think>\n"

    (the model's own template over the whole conversation, enable_thinking=True, the trace passed as reasoning_content). render_chat would put its empty thinking-off
    block in front of such an answer, which then holds a second block. Without thinking the result is render_chat's. `thinking`: None = decide with has_thinking."""
    fam = family(fam_name)
    on = has_thinking(messages) if thinking is None else bool(thinking)
    if not (on and fam.get("thinks")) or not messages or messages[-1]["role"] != "assistant":
        return render_chat(tokenizer, _without_empty_block(messages, fam) if fam.get("thinks") else messages, fam_name)
    tok = _inner(tokenizer)
    last = messages[-1]
    trace, answer = split_thinking(last["content"]) if isinstance(last.get("content"), str) else (None, "")
    if not trace:
        r = last.get("reasoning_content")
        trace, answer = (r.strip() if isinstance(r, str) else ""), (last.get("content") or "")
    if not trace:
        return render_chat(tokenizer, messages, fam_name)
    turn = {"role": "assistant", "content": answer.strip(), "reasoning_content": trace}
    full = tok.apply_chat_template(messages[:-1] + [turn], tokenize=False, add_generation_prompt=False, enable_thinking=True)
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if not full.startswith(prompt):
        raise ValueError("the chat template did not write the reasoning block where the prompt ends")
    return prompt, full


def _without_empty_block(messages: list, fam: dict) -> list:
    """The last assistant turn without a leading EMPTY reasoning block ("<think>\n\n</think>\n\n"): the thinking-off prompt already ends with one."""
    if not messages or messages[-1].get("role") != "assistant" or not isinstance(messages[-1].get("content"), str):
        return messages
    c = messages[-1]["content"]
    trace, answer = split_thinking(c)
    if trace == "" and c.lstrip().startswith(THINK_OPEN):
        return messages[:-1] + [{**messages[-1], "content": answer}]
    return messages


LENGTH_EDGES = (512, 1024, 2048, 4096, 8192, 16384)
LENGTH_LABELS = ("<=512", "<=1k", "<=2k", "<=4k", "<=8k", "<=16k", ">16k")


def length_bucket(n: int) -> str:
    """The label of the length bucket of a row of `n` tokens (before any cutting)."""
    for edge, label in zip(LENGTH_EDGES, LENGTH_LABELS):
        if n <= edge:
            return label
    return LENGTH_LABELS[-1]


def encode_rows(tokenizer, rows: list, fam_name: str, max_len: int, seed: int = 3407, probe=None):
    """[{input_ids, completion_mask, kind: "text"|"chat"}] and counts. Rows whose prompt alone does not fit are skipped; longer rows are cut at max_len.
    `stats["length_buckets"]` counts every row by its length before cutting (skipped rows included); `probe(ids) -> bool` (optional) says whether a skipped row was a reasoning row
    (`stats["skipped_reasoning"]`)."""
    tok = _inner(tokenizer)
    bos = getattr(tok, "bos_token", None)
    out, stats = [], {"text": 0, "chat": 0, "skipped_too_long": 0, "cut": 0, "empty": 0, "length_buckets": {label: 0 for label in LENGTH_LABELS}, "skipped_reasoning": 0}
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
            stats["length_buckets"][length_bucket(len(ids))] += 1
            if len(p_ids) >= max_len - 1:
                stats["skipped_too_long"] += 1
                if probe is not None and probe(ids):
                    stats["skipped_reasoning"] += 1
                continue
        if kind == "text":
            stats["length_buckets"][length_bucket(len(ids))] += 1
        if len(ids) > max_len:
            ids, mask = ids[:max_len], mask[:max_len]
            stats["cut"] += 1
        stats[kind] += 1
        out.append({"input_ids": ids, "completion_mask": mask, "kind": kind})
    random.Random(seed).shuffle(out)
    stats["tokens"] = sum(len(x["input_ids"]) for x in out)
    stats["learned_tokens"] = sum(sum(x["completion_mask"]) for x in out)
    return out, stats


def load_training_rows(tokenizer, path: str, fam_name: str, max_len: int, seed: int, probe=None):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("The training file is empty. Save a version in Step 2 first.")
    enc, stats = encode_rows(tokenizer, rows, fam_name, max_len, seed, probe)
    if not enc:
        raise ValueError("Nothing in the training file fits the chosen sequence length. Raise the length or shorten the examples.")
    protocol.log(f"Training data: {stats['text']} world-book texts and {stats['chat']} practice conversations, {stats['tokens']:,} tokens "
                 f"({stats['learned_tokens']:,} learned); {stats['cut']} cut to {max_len} tokens, {stats['skipped_too_long']} skipped (too long).")
    return enc, stats
