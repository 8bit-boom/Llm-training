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
            mask = out["attention_mask"]
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
