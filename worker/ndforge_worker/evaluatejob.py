"""The evaluate job: answer questions with the original model and with the trained one (same prompts, greedy decoding), so the window can score them.

Config keys: model {repo, family}, adapter (folder of a LoRA run) OR full_model (folder of a full fine-tune), questions [{id, messages: [{role, content}...]}]
(the conversation up to and including the last user turn), which (["original", "trained"]), max_new (tokens), out (JSON file to write).
Writes {"original": {id: text}, "trained": {id: text}} to `out`. For a LoRA run the original answers come from the same model with the adapter switched off."""
import gc
import json
from pathlib import Path

from . import protocol
from .families import family


def _dtype():
    import torch
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    return torch.float32


def _load(repo):
    import torch
    from transformers import AutoModelForCausalLM
    kw = {"device_map": {"": 0}} if torch.cuda.is_available() else {}
    try:
        return AutoModelForCausalLM.from_pretrained(repo, dtype=_dtype(), **kw)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(repo, torch_dtype=_dtype(), **kw)


def _answers(model, tok, fam_name, questions, max_new, label, done, total):
    import torch
    from .data import _inner, render_chat
    inner = _inner(tok)
    model.eval()
    dev = next(model.parameters()).device
    out = {}
    for q in questions:
        msgs = list(q["messages"]) + [{"role": "assistant", "content": ""}]
        prompt, _ = render_chat(tok, msgs, fam_name)
        enc = inner(prompt, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            ids = model.generate(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev), max_new_tokens=max_new, do_sample=False,
                                 use_cache=True, pad_token_id=inner.pad_token_id if inner.pad_token_id is not None else inner.eos_token_id)
        out[q["id"]] = inner.decode(ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        done += 1
        protocol.progress("evaluate", done / total, f"Answering ({label}): {done} of {total}")
    return out, done


def run(cfg: dict) -> int:
    protocol.reserve_stdout()
    import torch
    from transformers import AutoTokenizer
    fam_name = cfg["model"]["family"]
    fam = family(fam_name)  # noqa: F841  (fails early for a family the worker does not know; "generic" is accepted)
    which = cfg.get("which") or ["original", "trained"]
    qs = cfg["questions"]
    total = len(qs) * len(which)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["repo"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    max_new = int(cfg.get("max_new", 160))
    answers, done = {}, 0
    full = cfg.get("full_model")
    if not full and not cfg.get("adapter"):
        raise RuntimeError("This run has no saved model to test.")
    protocol.progress("load", 0.0, "Loading the original model")
    base = _load(cfg["model"]["repo"])
    if full:
        if "original" in which:
            answers["original"], done = _answers(base, tok, fam_name, qs, max_new, "original", done, total)
        del base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if "trained" in which:
            protocol.progress("load", 0.0, "Loading the trained model")
            trained = _load(full)
            answers["trained"], done = _answers(trained, tok, fam_name, qs, max_new, "trained", done, total)
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, cfg["adapter"])
        if "original" in which:
            with model.disable_adapter():
                answers["original"], done = _answers(model, tok, fam_name, qs, max_new, "original", done, total)
        if "trained" in which:
            answers["trained"], done = _answers(model, tok, fam_name, qs, max_new, "trained", done, total)
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
    protocol.result({"out": str(out), "n": len(qs)})
    return 0
