"""The training job: PEFT (LoRA / QLoRA) or full fine-tuning, with SFT on a saved dataset version, in one or two stages.

Config comes from the Rust program (see run() for the keys). Events (protocol.py): start (per stage), metric, sample, progress, log; the result has the
saved model (an adapter folder, or the whole model for full fine-tuning), peak memory, steps and the final loss.

Stages: [{"name", "filter": "text"|"chat"|"all", "epochs"?, "max_steps"?, "lr"?}]. One stage "all" trains the world-book texts and the practice
conversations together (the default). Two stages, "text" then "chat", read the world book first and then practise answering; the same model carries on
from one stage to the next, and a stopped run continues in the stage it was in."""
import json
import math
import os
import time
from pathlib import Path

from . import backends, protocol, speed


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


def _trainer_class(base, sampler_factory):
    """The trainer with batches of similar length (see speed.py)."""
    class BucketedTrainer(base):
        def _get_train_sampler(self, *args, **kwargs):
            return sampler_factory()
    return BucketedTrainer


def _decide_checkpointing(model, rows, batch, pad_id, bf16):
    """Plain-libraries path only (Unsloth manages its own): gradient checkpointing saves memory but costs about a third more time, so it is switched off when the
    longest batch runs without it and leaves memory to spare. Returns (checkpointing on, peak bytes or None)."""
    import torch
    longest = sorted(rows, key=lambda r: len(r["input_ids"]), reverse=True)[:batch]
    total = torch.cuda.get_device_properties(0).total_memory
    try:
        model.gradient_checkpointing_disable()
        peak, oom = speed.probe_checkpointing(model, _collator(pad_id)(longest), bf16)
    except Exception as exc:   # whatever goes wrong, the safe setting stays
        protocol.log(f"Could not test training without checkpointing ({type(exc).__name__}: {exc}); keeping it on.")
        peak, oom = None, True
    use = speed.checkpointing_choice(peak, total, oom)
    if use:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    return use, peak


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


def _steps(n_rows: int, h: dict, stage: dict, single: bool) -> int:
    """Optimizer steps a stage will take (the Trainer's own rule, rounded the same way)."""
    if single and h.get("max_steps"):
        return int(h["max_steps"])
    if stage.get("max_steps"):
        return int(stage["max_steps"])
    per_step = max(1, h["batch"] * h["grad_accum"])
    return max(1, math.ceil(n_rows / per_step * float(stage.get("epochs", h["epochs"]))))


def _read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"completed": []}


def run(cfg: dict) -> int:
    """cfg: {model: {repo, family}, method: lora|qlora|full, backend: auto|unsloth|hf, dataset: {train}, hyper: {rank, alpha, lr, epochs, max_steps,
    batch, grad_accum, max_seq, warmup, weight_decay, scheduler, seed, save_steps, logging_steps}, out, resume, stages?, probes: [{system, question}],
    probe_every, probe_tokens, optim, init_adapter? (continue a LoRA), init_model? (continue full fine-tuning from a saved model)}"""
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
    full = cfg["method"] == "full"
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
    stages = cfg.get("stages") or [{"name": "all", "filter": "all"}]
    single = len(stages) == 1
    stage_rows = [[r for r in rows if st["filter"] == "all" or r["kind"] == st["filter"]] for st in stages]
    estimates = [_steps(len(sr), h, st, single) if sr else 0 for sr, st in zip(stage_rows, stages)]
    state_path = out / "stage_state.json"
    prior = _read_state(state_path) if cfg.get("resume") else {"completed": []}
    # finished stages are skipped when a two-stage run continues; a single stage always continues from its own checkpoint
    done_stages = {} if single else {c["name"]: c["steps"] for c in prior["completed"]}
    probes = cfg.get("probes") or []
    answer_fn = _make_answer_fn(model, tokenizer, cfg["model"]["family"], backend, loader, cfg.get("probe_tokens", 120)) if probes else None
    ck_note, ck_state = None, None
    if backend == "unsloth":
        ck_state = "unsloth"
    elif cuda and not full:
        use, peak = _decide_checkpointing(model, rows, h["batch"], pad_id, bf16)
        ck_state = "on" if use else "off"
        gb = lambda b: f"{b / 1024 ** 3:.1f} GB"
        ck_note = (f"Gradient checkpointing is on: the longest batch needs more than 85 % of the card's memory without it." if use and peak
                   else "Gradient checkpointing is on: the longest batch did not fit without it." if use else f"Gradient checkpointing is off (faster): the longest batch needs {gb(peak)} without it.")
        protocol.log(ck_note)
    resumed_any = False
    t0 = time.time()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    offset, live_total, stopped, last_loss, peak = 0, 0, False, None, 0
    fields = getattr(SFTConfig, "__dataclass_fields__", {})
    for i, (st, sr) in enumerate(zip(stages, stage_rows)):
        name = st["name"]
        if not sr:
            protocol.log(f"Stage \"{name}\" has no examples in this data version; skipped.")
            continue
        if name in done_stages:
            offset += done_stages[name]
            protocol.log(f"Stage \"{name}\" was finished before; continuing after it.")
            continue
        if len(stages) > 1:
            protocol.log(f"Stage {i + 1} of {len(stages)}: {'reading the world book' if st['filter'] == 'text' else 'practising answers'} ({len(sr)} examples).")
        ckdir = out / "checkpoints" if single else out / "checkpoints" / name
        kwargs = dict(
            output_dir=str(ckdir), per_device_train_batch_size=h["batch"], gradient_accumulation_steps=h["grad_accum"],
            learning_rate=float(st.get("lr", h["lr"])), warmup_steps=h["warmup"], weight_decay=h["weight_decay"], lr_scheduler_type=h["scheduler"],
            logging_steps=h["logging_steps"], save_strategy="steps", save_steps=h["save_steps"], save_total_limit=1 if full else 2, optim=optim,
            seed=h["seed"], report_to="none", disable_tqdm=True, bf16=bf16, fp16=cuda and not bf16, remove_unused_columns=False,
            gradient_checkpointing=False, dataloader_num_workers=0, use_cpu=not cuda,
        )
        if single and h.get("max_steps"):
            kwargs["max_steps"] = int(h["max_steps"])
        elif st.get("max_steps"):
            kwargs["max_steps"] = int(st["max_steps"])
        else:
            kwargs["num_train_epochs"] = float(st.get("epochs", h["epochs"]))
        kwargs["max_length" if "max_length" in fields else "max_seq_length"] = h["max_seq"]
        if "dataset_kwargs" in fields:
            kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
        if "packing" in fields:
            kwargs["packing"] = False
        model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
        extra, padding_free, adv_notes = speed.advanced_options(h, cfg["model"]["family"], model_type, backend, cuda, torch.cuda.get_device_capability(0) if cuda else None, fields)
        kwargs.update(extra)
        args = SFTConfig(**kwargs)
        steps_after = sum(estimates[i + 1:])
        live = LiveCallback(str(stop_file), answer_fn, probes, cfg.get("probe_every", 0), stage=name, stage_index=i, stage_count=len(stages), offset=offset, steps_after=steps_after)
        lengths = [len(r["input_ids"]) for r in sr]
        pad_random, pad_bucketed = speed.padding_estimate(lengths, h["batch"], h["seed"])
        bucket = speed.should_bucket(h.get("bucket", "auto"), pad_random, len(sr), h["batch"]) and not padding_free
        counter = speed.TokenCounter()
        live.counter = counter
        notes = speed.kernel_notes(cfg["model"]["family"], backend, cuda, torch.cuda.get_device_capability(0) if cuda else None) + adv_notes
        protocol.emit("speed", stage=name, bucketed=bucket, pad_random_pct=round(100 * pad_random, 1), pad_bucketed_pct=round(100 * pad_bucketed, 1),
                      mean_len=round(sum(lengths) / len(lengths)), max_len=max(lengths), batch=h["batch"], checkpointing=ck_state, checkpointing_note=ck_note, engine=backend, padding_free=padding_free, options=sorted(extra) + (["padding_free"] if padding_free else []),
                      attn=getattr(getattr(model, "config", None), "_attn_implementation", None), notes=notes)
        ds = Dataset.from_list([{"input_ids": r["input_ids"], "completion_mask": r["completion_mask"]} for r in sr])
        trainer_kw = {"model": model, "args": args, "train_dataset": ds, "data_collator": counter.wrap(speed.flatten_collator(pad_id) if padding_free else _collator(pad_id)), "callbacks": [live]}
        trainer_kw["processing_class" if "processing_class" in SFTTrainer.__init__.__code__.co_varnames else "tokenizer"] = tokenizer
        trainer = (_trainer_class(SFTTrainer, lambda: speed.make_sampler(lengths, h["batch"], h["seed"])) if bucket else SFTTrainer)(**trainer_kw)
        ckpts = sorted(ckdir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if ckdir.exists() else []
        resume = bool(cfg.get("resume")) and bool(ckpts)
        if cfg.get("resume") and not ckpts and not resumed_any:
            protocol.log("No checkpoint to continue from; starting from the beginning.")
        resumed_any = resumed_any or resume
        trainer.train(resume_from_checkpoint=True if resume else None)
        done_steps = trainer.state.global_step
        last_loss = live.last_loss if live.last_loss is not None else last_loss
        peak = max(peak, live._vram())
        model = trainer.model   # the same model carries on into the next stage
        if live.stopped:
            stopped = True
            break
        prior["completed"] = [c for c in prior["completed"] if c["name"] != name] + [{"name": name, "steps": done_steps}]
        state_path.write_text(json.dumps(prior), encoding="utf-8")
        offset += done_steps
    seconds = round(time.time() - t0, 1)
    if full:
        target = out / "model"
        model.save_pretrained(str(target))
    else:
        target = out / "adapter"
        model.save_pretrained(str(target))
    try:
        tokenizer.save_pretrained(str(target))
    except Exception as exc:
        protocol.log(f"Tokenizer not saved with the model: {exc}")
    summary = {"backend": backend, "method": cfg["method"], "precision": backends.precision_name(), "steps": offset, "total_steps": max(offset, sum(estimates)),
               "final_loss": last_loss, "train_seconds": seconds, "peak_vram_mb": peak,
               "adapter_dir": None if full else str(target), "model_dir": str(target) if full else None, "stopped": stopped, "data": stats,
               "model": cfg["model"]["repo"], "stages": [c for c in prior["completed"]]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    protocol.result(summary)
    return 0
