"""The evaluate job: answer questions with the original model and with the trained one (same prompts, greedy decoding), so the window can score them.

Config keys: model {repo, family}, adapter (folder of a LoRA run) OR full_model (folder of a full fine-tune), questions [{id, messages: [{role, content}...]}]
(the conversation up to and including the last user turn), which (["original", "trained"]), max_new (tokens), out (JSON file to write).
Optional: trained_base (the folder the adapter was trained on when that is not model.repo: a run on a slimmed / grown model, compared with the real original),
load_4bit {"original": bool, "trained": bool} (load that side in 4-bit so a big model fits the card).
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


def plan_sides(cfg: dict) -> dict:
    """What the job loads, as plain data (so it can be tested without torch):
    {"kind": "adapter" | "adapter_on_other_base" | "full", "original": (repo, four_bit), "trained": (folder to load, four_bit), "loaded": {...}}.
    One model carries both sides of a plain adapter test (the adapter is switched off for the original), so both share the loading mode."""
    four = cfg.get("load_4bit") or {}
    fo, ft = bool(four.get("original")), bool(four.get("trained"))
    repo, full, adapter, other = cfg["model"]["repo"], cfg.get("full_model"), cfg.get("adapter"), cfg.get("trained_base")
    if not full and not adapter:
        raise RuntimeError("This run has no saved model to test.")
    name = lambda b: "4-bit" if b else "16-bit"  # noqa: E731
    if full:
        return {"kind": "full", "original": (repo, fo), "trained": (full, ft), "loaded": {"original": name(fo), "trained": name(ft)}}
    if other:
        return {"kind": "adapter_on_other_base", "original": (repo, fo), "trained": (other, ft), "loaded": {"original": name(fo), "trained": name(ft)}}
    shared = fo or ft
    return {"kind": "adapter", "original": (repo, shared), "trained": (repo, shared), "loaded": {"original": name(shared), "trained": name(shared)}}


def _load(repo, four_bit=False):
    import torch
    from transformers import AutoModelForCausalLM
    kw = {"device_map": {"": 0}} if torch.cuda.is_available() else {}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=_dtype())
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
    sides = plan_sides(cfg)
    protocol.progress("load", 0.0, "Loading the original model")
    if sides["kind"] in ("full", "adapter_on_other_base"):
        if "original" in which:
            base = _load(*sides["original"])
            answers["original"], done = _answers(base, tok, fam_name, qs, max_new, "original", done, total)
            del base
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if "trained" in which:
            protocol.progress("load", 0.0, "Loading the trained model")
            trained = _load(*sides["trained"])
            if sides["kind"] == "adapter_on_other_base":
                from peft import PeftModel
                trained = PeftModel.from_pretrained(trained, cfg["adapter"])
            answers["trained"], done = _answers(trained, tok, fam_name, qs, max_new, "trained", done, total)
    else:
        from peft import PeftModel
        base = _load(*sides["original"])
        model = PeftModel.from_pretrained(base, cfg["adapter"])
        if "original" in which:
            with model.disable_adapter():
                answers["original"], done = _answers(model, tok, fam_name, qs, max_new, "original", done, total)
        if "trained" in which:
            answers["trained"], done = _answers(model, tok, fam_name, qs, max_new, "trained", done, total)
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
    protocol.result({"out": str(out), "n": len(qs), "loaded": sides["loaded"]})
    return 0
