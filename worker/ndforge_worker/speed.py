"""Training speed helpers: how much of each batch is padding, batches of similar length, notes about the fast kernels, and the checkpointing decision.

Why: a batch is padded to its longest example, and the padding is computed on and thrown away. Our examples range from a one-line answer to a long
world-book piece, so random batches waste a lot. Putting examples of similar length in the same batch removes most of it (the batch order is still
random, and every epoch groups the examples differently). Nothing here needs a graphics card, except `probe_checkpointing`."""
import importlib.util
import random


def _batches(order, batch):
    return [order[i:i + batch] for i in range(0, len(order), batch)]


def padding_fraction(lengths, batches) -> float:
    """Share of the computed tokens that are padding when the examples are processed in these batches of indices."""
    real = padded = 0
    for b in batches:
        longest = max(lengths[i] for i in b)
        padded += longest * len(b)
        real += sum(lengths[i] for i in b)
    return 0.0 if padded == 0 else 1.0 - real / padded


def bucket_order(lengths, batch, seed, epoch, mega=8):
    """Indices in an order where consecutive groups of `batch` are of similar length: shuffle, cut into pieces of `mega` batches, sort each piece by
    length, cut into batches and shuffle the batches. The grouping differs every epoch."""
    rng = random.Random(seed * 1_000_003 + epoch)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    size = max(batch, batch * mega)
    batches = []
    for i in range(0, len(idx), size):
        piece = sorted(idx[i:i + size], key=lambda j: lengths[j], reverse=True)
        batches.extend(_batches(piece, batch))
    rng.shuffle(batches)
    return [j for b in batches for j in b], batches


def padding_estimate(lengths, batch, seed=1, trials=6):
    """(random batches, bucketed batches): the average padding share of each way of making batches."""
    if batch <= 1 or len(lengths) < 2:
        return 0.0, 0.0
    rnd = bucketed = 0.0
    for t in range(trials):
        rng = random.Random(seed + t)
        idx = list(range(len(lengths)))
        rng.shuffle(idx)
        rnd += padding_fraction(lengths, _batches(idx, batch))
        bucketed += padding_fraction(lengths, bucket_order(lengths, batch, seed, t)[1])
    return rnd / trials, bucketed / trials


def should_bucket(setting, pad_random: float, n: int, batch: int) -> bool:
    """`setting`: "auto" (when at least 8 % of a random batch is padding), "on" or "off"."""
    if batch <= 1 or n < 2 * batch or setting == "off":
        return False
    return setting == "on" or pad_random >= 0.08


def make_sampler(lengths, batch, seed, mega=8):
    """A torch sampler that yields `bucket_order` and a new grouping every epoch."""
    from torch.utils.data import Sampler

    class BucketSampler(Sampler):
        def __init__(self):
            self.epoch = 0
            self.auto = True

        def set_epoch(self, epoch):
            self.epoch, self.auto = epoch, False

        def __iter__(self):
            order, _ = bucket_order(lengths, batch, seed, self.epoch, mega)
            if self.auto:
                self.epoch += 1
            return iter(order)

        def __len__(self):
            return len(lengths)

    return BucketSampler()


class TokenCounter:
    """Counts the real and the padded tokens of every batch the collator makes (the loader runs in this process)."""

    def __init__(self):
        self.real = 0
        self.padded = 0

    def wrap(self, collate):
        def counted(batch):
            out = collate(batch)
            mask = out.get("attention_mask")
            if mask is None:      # padding-free: every token is real
                self.real += int(out["input_ids"].numel())
                self.padded += int(out["input_ids"].numel())
            else:
                self.real += int(mask.sum())
                self.padded += int(mask.numel())
            return out
        return counted


def kernel_notes(family: str, backend: str, cuda: bool, capability=None) -> list:
    """Plain-language notes about speed that the window shows next to the run."""
    notes = []
    if family == "qwen3_5":
        if backend == "unsloth":
            notes.append("Qwen3.5 uses Unsloth's own fast kernels; the first steps can be slow while they compile.")
        else:
            missing = [n for n, mod in (("flash-linear-attention", "fla"), ("causal-conv1d", "causal_conv1d")) if importlib.util.find_spec(mod) is None]
            if missing:
                notes.append("Qwen3.5 has linear-attention layers, and its fast kernels (" + " and ".join(missing) + ") are not installed: Transformers then falls back to a much slower "
                             "plain-PyTorch version. Install them in the training environment, or use the Unsloth backend (Training setup).")
    if cuda and capability is not None and capability[0] < 8:
        notes.append("This graphics card has no bf16 and no FlashAttention: training uses fp16 and PyTorch's memory-efficient attention.")
    return notes


def checkpointing_choice(peak_bytes, total_bytes, out_of_memory: bool, limit: float = 0.85) -> bool:
    """True = use gradient checkpointing (less memory, about a third slower). Off only when the longest batch ran without it and stayed under `limit` of the memory."""
    if out_of_memory or peak_bytes is None or not total_bytes:
        return True
    return peak_bytes > limit * total_bytes


def probe_checkpointing(model, batch_of_longest, bf16: bool):
    """Runs one forward and backward pass of the longest batch with checkpointing OFF; returns (peak bytes or None, ran out of memory). Needs a CUDA card."""
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16):
            loss = model(**{k: v.to("cuda") for k, v in batch_of_longest.items()}).loss
        loss.backward()
        peak = torch.cuda.max_memory_allocated()
        return peak, False
    except torch.cuda.OutOfMemoryError:
        return None, True
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ------------------------------------------------------------------------------------------------------------ advanced options (all off by default)
def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def padding_free_status(family: str, backend: str, cuda: bool, capability):
    """(possible, reason). Padding-free batching puts the examples of a batch end to end in ONE row and tells FlashAttention-2 where each begins, so no
    padding is computed at all. It needs the flash-attn package, an Ampere or newer card and a model whose layers all understand that. Qwen3.5 (linear
    attention) and Gemma 4 (sliding-window attention) have not been checked for it, so it stays off for them."""
    if backend != "hf":
        return False, "padding-free batching works only on the plain-libraries path (Unsloth manages its own batching)"
    if not cuda or capability is None or capability[0] < 8:
        return False, "padding-free batching needs a graphics card of the RTX 30 series or newer"
    if not _have("flash_attn"):
        return False, "padding-free batching needs the flash-attn package, which is not installed (on Windows only community builds exist)"
    if family in ("qwen3_5", "gemma4"):
        return False, "padding-free batching has not been checked for this model family's attention layers, so it stays off"
    return True, ""


ATTENTION_CHOICES = ("auto", "sdpa", "flash_attention_2", "flex_attention")


def attention_request(choice: str, cuda: bool, capability) -> tuple:
    """(attn_implementation to ask for or None, note or ""). "auto" leaves the choice to the library (Unsloth picks its own, Transformers uses sdpa).
    An explicit choice that cannot work here is dropped with a reason instead of failing the run."""
    choice = choice or "auto"
    if choice == "auto":
        return None, ""
    if choice not in ATTENTION_CHOICES:
        return None, f"unknown attention setting \"{choice}\", using the default"
    if choice == "sdpa":
        return "sdpa", ""
    if not cuda:
        return None, f"{choice} needs a graphics card; using the default attention"
    if choice == "flash_attention_2":
        if capability is None or capability[0] < 8:
            return None, "FlashAttention-2 needs a graphics card of the RTX 30 series or newer; using the default attention"
        if not _have("flash_attn"):
            return None, "FlashAttention-2 needs the flash-attn package, which is not installed (on Windows only community builds exist); using the default attention"
        return "flash_attention_2", ""
    try:
        import torch
        major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if (major, minor) < (2, 5):
            return None, "flex attention needs PyTorch 2.5 or newer; using the default attention"
    except Exception:
        return None, "flex attention could not be checked; using the default attention"
    return "flex_attention", ""


def liger_status(model_type: str, backend: str):
    """(possible, reason) for the fused Liger kernels (they save memory and time on the loss with huge vocabularies)."""
    if backend != "hf":
        return False, "Liger kernels are for the plain-libraries path (Unsloth has its own)"
    if not _have("liger_kernel"):
        return False, "the liger-kernel package is not installed"
    try:
        from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
        if model_type not in MODEL_TYPE_TO_APPLY_LIGER_FN:
            return False, f"Liger has no kernels for the model type \"{model_type}\""
    except Exception:
        return False, "the installed liger-kernel could not be asked which models it supports"
    return True, ""


def compile_status(backend: str, cuda: bool):
    """(possible, reason) for torch.compile. It needs Triton on a graphics card; batches of changing length make it compile again, so it is for experiments."""
    if backend != "hf":
        return False, "torch.compile is for the plain-libraries path (Unsloth compiles its own parts)"
    if not cuda:
        return False, "torch.compile is used only with a graphics card here"
    if not _have("triton"):
        return False, "torch.compile needs Triton, which is not installed (on Windows only community builds exist)"
    return True, ""


def advanced_options(h: dict, family: str, model_type: str, backend: str, cuda: bool, capability, fields) -> tuple:
    """(extra keyword arguments for the trainer's config, padding_free active, [plain-language notes]) for the settings compile / liger / padding_free."""
    kwargs, notes, pf = {}, [], False
    if h.get("compile") == "on":
        ok, why = compile_status(backend, cuda)
        if ok and "torch_compile" in fields:
            kwargs["torch_compile"] = True
            notes.append("torch.compile is on: the first steps are slow while it compiles, and batches of a new length compile again.")
        else:
            notes.append(f"torch.compile was asked for but is off: {why or 'this version of Transformers has no such setting'}.")
    if h.get("liger") == "on":
        ok, why = liger_status(model_type, backend)
        if ok and "use_liger_kernel" in fields:
            kwargs["use_liger_kernel"] = True
            notes.append("Liger kernels are on.")
        else:
            notes.append(f"Liger kernels were asked for but are off: {why or 'this version of Transformers has no such setting'}.")
    if h.get("padding_free") == "on":
        ok, why = padding_free_status(family, backend, cuda, capability)
        if ok:
            pf = True
            notes.append("Padding-free batching is on: no padding is computed.")
        else:
            notes.append(f"Padding-free batching was asked for but is off: {why}.")
    return kwargs, pf, notes


def flatten_collator(pad_id):
    """A collator for padding-free batches: the examples end to end in one row, with position ids that restart for each example (that is how
    FlashAttention-2 finds the boundaries) and no label across a boundary (the first token of an example is never predicted from the last token of the one before)."""
    import torch

    def collate(batch):
        ids, labels, pos = [], [], []
        for b in batch:
            t = torch.tensor(b["input_ids"], dtype=torch.long)
            m = torch.tensor(b["completion_mask"], dtype=torch.bool)
            lab = torch.where(m, t, torch.full_like(t, -100))
            lab[0] = -100
            ids.append(t)
            labels.append(lab)
            pos.append(torch.arange(len(t)))
        return {"input_ids": torch.cat(ids)[None], "labels": torch.cat(labels)[None], "position_ids": torch.cat(pos)[None]}
    return collate
