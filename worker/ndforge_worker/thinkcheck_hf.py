"""The transformers backend of the thinking check: loads the model like the other jobs do (`reshapejob._load`: 16-bit, 4-bit or spread over card and memory), asks the
questions in batches with sampling, left padding and the model's own chat template (thinking on: the prompt ends with `<think>\\n`; off: with the empty block).

    job: {"model": folder or Hub id, "load": "auto" | "16bit" | "4bit" | "offload", "batch": 8, ...}
         optional "adapter": a LoRA folder put on top of the loaded model WITHOUT merging it (the model as it is while it is trained; a merged 4-bit model is a different model),
         "double_quant": true loads a 4-bit base like the student is loaded for training (NF4 with double quantization).

Batched generation of a model with linear-attention layers and left padding is not proven to give the same answers as one question at a time (see docs/THINKING_CHECK.md);
`"batch": 1` asks one at a time. An out-of-memory error halves the batch and goes on.
"""
from __future__ import annotations

import time

from . import families, protocol
from .data import text_tokenizer
from .thinkcheck import ThinkCheckError

DEFAULT_BATCH = 8


def prompt_of(inner, request: dict, family: str | None = None) -> str:
    """The text the model is given: its chat template with the generation prompt, and enable_thinking as the request says (a family with its own template argument, Inkling's
    `reasoning_effort`, gets that one)."""
    try:
        kw = families.think_kwargs(family, bool(request["thinking"])) if family else {"enable_thinking": bool(request["thinking"])}
        return inner.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True, **kw)
    except Exception as exc:        # noqa: BLE001 - a tokenizer with no chat template
        raise ThinkCheckError(f"The model's tokenizer cannot write the questions with a chat template ({type(exc).__name__}: {str(exc)[:120]}).") from exc


def stop_ids(inner, model) -> list:
    """Every token id that ends an answer: the tokenizer's end-of-sequence, the model's generation config and the Qwen turn markers."""
    ids = []
    cfg = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    for v in [inner.eos_token_id, *(cfg if isinstance(cfg, (list, tuple)) else [cfg])]:
        if v is not None and v not in ids:
            ids.append(int(v))
    unk = getattr(inner, "unk_token_id", None)
    for name in ("<|im_end|>", "<|endoftext|>", "<|content_model_end_sampling|>"):
        tid = inner.convert_tokens_to_ids(name)
        if tid is not None and tid != unk and tid not in ids:
            ids.append(int(tid))
    return ids


def split_generated(ids: list, stops: list, pad_id) -> tuple[list, bool]:
    """(the answer's token ids, cut by the limit): the ids up to the first stop token; without one the limit ended it (trailing padding taken off)."""
    for i, t in enumerate(ids):
        if t in stops:
            return ids[:i], False
    while ids and pad_id is not None and ids[-1] == pad_id:
        ids = ids[:-1]
    return ids, True


def _generate_batch(model, enc, settings: dict, max_new: int, stops: list, pad_id: int, seed: int):
    import torch
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=float(settings["temperature"]), top_p=float(settings["top_p"]), top_k=int(settings["top_k"]),
                             max_new_tokens=int(max_new), pad_token_id=pad_id, eos_token_id=stops, use_cache=True)
    return out[:, enc["input_ids"].shape[1]:].tolist()


def generate(requests: list, settings: dict, on_result, should_stop, job: dict) -> None:
    from transformers import AutoTokenizer

    from . import reshapejob, safetensors_io, speed
    folder = reshapejob._resolve_source(str(job["model"]))
    tok = AutoTokenizer.from_pretrained(str(folder))
    inner = text_tokenizer(tok)
    inner.padding_side = "left"
    if inner.pad_token_id is None:
        inner.pad_token = inner.eos_token
    est = sum(r.nbytes for r in safetensors_io.list_tensors(folder).values())
    mode = reshapejob._choose_load(job.get("load", "auto"), est)
    protocol.progress("load", 0.0, f"Loading the model ({mode})")
    adapter = job.get("adapter")
    model = reshapejob._load(folder, mode, "load_model", True) if (adapter and job.get("double_quant")) else reshapejob._load(folder, mode)
    if adapter:
        from peft import PeftModel
        protocol.progress("load", 0.8, "Putting the LoRA on the model")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()
    kernels = speed.report_kernels(model)
    model.config.use_cache = True
    protocol.progress("load", 1.0, "")
    stops = stop_ids(inner, model)
    close_id = inner.convert_tokens_to_ids("</think>")
    if close_id == getattr(inner, "unk_token_id", None):
        close_id = None
    pad_id = inner.pad_token_id
    batch = max(1, int(job.get("batch") or DEFAULT_BATCH))
    # the same kind of question together (same limit), the shortest prompts first: less padding
    prompts = {r["id"]: prompt_of(inner, r, job.get("family")) for r in requests}
    order = sorted(requests, key=lambda r: (not r["thinking"], len(prompts[r["id"]])))
    seed, n_batch, i = int(settings["seed"]), 0, 0
    counted = {"tokens": 0}
    inner_result = on_result

    def on_result(item):                                    # noqa: F811 - counts the generated tokens for the timing line
        counted["tokens"] += int(item.get("gen_tokens") or 0)
        inner_result(item)

    t_gen = time.monotonic()
    try:
        while i < len(order):
            if should_stop():
                return
            first = order[i]
            part = [r for r in order[i:i + batch] if r["thinking"] == first["thinking"] and r["max_new"] == first["max_new"]]
            enc = inner([prompts[r["id"]] for r in part], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            try:
                gen = _generate_batch(model, enc, settings, first["max_new"], stops, pad_id, seed + n_batch)
            except RuntimeError as exc:
                if not reshapejob._is_oom(exc) or batch == 1:
                    raise
                batch = max(1, batch // 2)
                protocol.log(f"Out of memory; the questions are asked {batch} at a time.")
                del enc
                reshapejob._free()
                continue
            for r, row in zip(part, gen):
                ids, truncated = split_generated(row, stops, pad_id)
                think_tokens = None
                if r["thinking"]:
                    think_tokens = ids.index(close_id) if close_id is not None and close_id in ids else len(ids)
                on_result({"id": r["id"], "text": inner.decode(ids, skip_special_tokens=False), "gen_tokens": len(ids), "truncated": truncated, "tokens": ids,
                           "think_tokens": think_tokens})
            i += len(part)
            n_batch += 1
    finally:
        spent = time.monotonic() - t_gen
        protocol.timing("think_generate", spent, generated_tokens=counted["tokens"], tokens_per_s=round(counted["tokens"] / spent, 1) if spent > 0 and counted["tokens"] else None,
                        final_batch=batch, kernels=(kernels or {}).get("delta_rule") if kernels else None)
        del model
        reshapejob._free()
