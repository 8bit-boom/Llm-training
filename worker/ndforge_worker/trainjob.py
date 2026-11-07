"""The training job: PEFT (LoRA / QLoRA) + SFT on a saved dataset version. Config comes from the Rust program (see run() for the keys).

Events (protocol.py): start, metric, sample, progress, log; the result has the adapter folder, peak memory, steps and the final loss."""
import json
import os
import time
from pathlib import Path

from . import backends, protocol


def _collator(pad_id: int):
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
            m = torch.tensor(b["completion_mask"], dtype=torch.bool)
            lab[i, :L] = torch.where(m, t, torch.full_like(t, -100))
        return {"input_ids": ids, "attention_mask": att, "labels": lab}
    return collate


def _make_answer_fn(model, tokenizer, fam_name, backend, loader, max_new):
    import torch
    from .data import _inner, render_chat

    def answer(q):
        msgs = [{"role": "system", "content": q["system"]}, {"role": "user", "content": q["question"]}, {"role": "assistant", "content": ""}]
        prompt, _ = render_chat(tokenizer, msgs, fam_name)
        tok = _inner(tokenizer)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        dev = next(model.parameters()).device
        backends.to_inference(backend, loader, model)
        try:
            with torch.no_grad():
                out = model.generate(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev), max_new_tokens=max_new,
                                     do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        finally:
            backends.to_training(backend, loader, model)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return answer


def run(cfg: dict) -> int:
    """cfg: {model: {repo, family}, method: lora|qlora, backend: auto|unsloth|hf, dataset: {train}, hyper: {rank, alpha, lr, epochs, max_steps,
    batch, grad_accum, max_seq, warmup, weight_decay, scheduler, seed, save_steps, logging_steps}, out, resume, probes: [{system, question}],
    probe_every, probe_tokens, optim}"""
    protocol.reserve_stdout()
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    stop_file = out / "STOP"
    stop_file.unlink(missing_ok=True)
    backend = backends.prepare(cfg.get("backend", "auto"))   # before anything imports transformers
    import torch
    from datasets import Dataset
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer
    from . import data
    from .callbacks import LiveCallback

    h = cfg["hyper"]
    set_seed(h["seed"])
    protocol.log(f"Backend: {backend}; method: {cfg['method']}; precision: {backends.precision_name()}")
    model, tokenizer, loader = backends.load(cfg, backend)
    rows, stats = data.load_training_rows(tokenizer, cfg["dataset"]["train"], cfg["model"]["family"], h["max_seq"], h["seed"])
    inner = data._inner(tokenizer)
    pad_id = inner.pad_token_id if inner.pad_token_id is not None else inner.eos_token_id
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.get_device_capability(0)[0] >= 8
    optim = cfg.get("optim") or ("adamw_8bit" if cuda else "adamw_torch")
    if optim == "adamw_8bit":
        try:
            import bitsandbytes  # noqa: F401
        except Exception:
            optim = "adamw_torch"
    kwargs = dict(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=h["batch"], gradient_accumulation_steps=h["grad_accum"],
        learning_rate=h["lr"], warmup_steps=h["warmup"], weight_decay=h["weight_decay"], lr_scheduler_type=h["scheduler"],
        logging_steps=h["logging_steps"], save_strategy="steps", save_steps=h["save_steps"], save_total_limit=2, optim=optim,
        seed=h["seed"], report_to="none", disable_tqdm=True, bf16=bf16, fp16=cuda and not bf16, remove_unused_columns=False,
        gradient_checkpointing=False, dataloader_num_workers=0, use_cpu=not cuda,
    )
    if h.get("max_steps"):
        kwargs["max_steps"] = int(h["max_steps"])
    else:
        kwargs["num_train_epochs"] = h["epochs"]
    fields = getattr(SFTConfig, "__dataclass_fields__", {})
    kwargs["max_length" if "max_length" in fields else "max_seq_length"] = h["max_seq"]
    if "dataset_kwargs" in fields:
        kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
    if "packing" in fields:
        kwargs["packing"] = False
    args = SFTConfig(**kwargs)
    probes = cfg.get("probes") or []
    answer_fn = _make_answer_fn(model, tokenizer, cfg["model"]["family"], backend, loader, cfg.get("probe_tokens", 120)) if probes else None
    live = LiveCallback(str(stop_file), answer_fn, probes, cfg.get("probe_every", 0))
    ds = Dataset.from_list(rows)
    trainer_kw = {"model": model, "args": args, "train_dataset": ds, "data_collator": _collator(pad_id), "callbacks": [live]}
    trainer_kw["processing_class" if "processing_class" in SFTTrainer.__init__.__code__.co_varnames else "tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kw)
    ckpts = sorted((out / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if (out / "checkpoints").exists() else []
    resume = bool(cfg.get("resume")) and bool(ckpts)
    if cfg.get("resume") and not ckpts:
        protocol.log("No checkpoint to continue from; starting from the beginning.")
    t0 = time.time()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=True if resume else None)
    seconds = round(time.time() - t0, 1)
    adapter = out / "adapter"
    trainer.model.save_pretrained(str(adapter))
    try:
        tokenizer.save_pretrained(str(adapter))
    except Exception as exc:
        protocol.log(f"Tokenizer not saved with the adapter: {exc}")
    summary = {"backend": backend, "method": cfg["method"], "precision": backends.precision_name(), "steps": trainer.state.global_step,
               "total_steps": trainer.state.max_steps, "final_loss": live.last_loss, "train_seconds": seconds, "peak_vram_mb": live._vram(),
               "adapter_dir": str(adapter), "stopped": live.stopped, "data": stats, "model": cfg["model"]["repo"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    protocol.result(summary)
    return 0
