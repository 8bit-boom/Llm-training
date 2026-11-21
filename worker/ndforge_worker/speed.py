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


def split_by_token_budget(lengths, budget):
    """The rows of ONE optimizer update (their `lengths`) split into micro-batches that each hold at most `budget` tokens once padded to their longest row: the rows are taken from the longest
    to the shortest (ties in their order), a micro-batch is closed when one more row would pass the budget; a row longer than the budget is a micro-batch of its own.
    -> list of lists of positions (into `lengths`), every position exactly once. `budget` <= 0: one row per micro-batch, in the given order (today's way)."""
    if budget <= 0:
        return [[i] for i in range(len(lengths))]
    order = sorted(range(len(lengths)), key=lambda j: (-lengths[j], j))
    out, cur = [], []
    for j in order:
        if cur and (len(cur) + 1) * lengths[cur[0]] > budget:
            out.append(cur)
            cur = []
        cur.append(j)
    if cur:
        out.append(cur)
    return out


def micro_batches(lengths, budget, seed, step):
    """`split_by_token_budget` with the order of the micro-batches shuffled by a seed and the update's number (the same update gets the same order on every run, another update another one:
    the order does not change the sum of the gradients, it only avoids always running the longest rows first)."""
    batches = split_by_token_budget(lengths, budget)
    random.Random(int(seed) * 1_000_003 + int(step)).shuffle(batches)
    return batches


def padded_share(lengths, batches) -> float:
    """The share of the computed tokens that is padding in these micro-batches (positions into `lengths`)."""
    return padding_fraction(lengths, batches)


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


def _origin(fn) -> str:
    return str(getattr(fn, "__module__", "") or "")


# transformers 5.23 chooses the delta-rule implementation ONCE, at import time, inside the modeling module (there is no `chunk_gated_delta_rule` attribute on the layers any more): the
# module's own globals, the closure of its torch reference, and the warning it logs when the fast path is not available are the evidence.
FALLBACK_WARNINGS: list = []
_FALLBACK_WORDS = ("fast path is not available", "falling back", "fall back", "flash-linear-attention", "causal-conv1d")


def install_fallback_watch() -> None:
    """Remembers the warnings transformers logs about a missing fast path (`FALLBACK_WARNINGS`); called when this module is imported, before any model is loaded."""
    import logging
    if any(getattr(h, "_nd_watch", False) for h in logging.getLogger("transformers").handlers):
        return

    class Watch(logging.Handler):
        _nd_watch = True

        def emit(self, record):
            try:
                msg = record.getMessage()
            except Exception:       # noqa: BLE001
                return
            if any(w in msg.lower() for w in _FALLBACK_WORDS) and len(FALLBACK_WARNINGS) < 20:
                FALLBACK_WARNINGS.append(msg[:300])

    logging.getLogger("transformers").addHandler(Watch(level=logging.WARNING))


try:
    install_fallback_watch()
except Exception:       # noqa: BLE001 - never stop an import
    pass


def _is_fla(fn) -> bool:
    return _origin(fn).split(".")[0] in ("fla", "flash_linear_attention")


def _callables_in(fn):
    """The values a function can see besides its arguments: the cells of its closure and its module globals' callables of the same names."""
    out = []
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            out.append(cell.cell_contents)
        except ValueError:
            pass
    return out


def _delta_rule(model_module) -> tuple:
    """(\"fla\" | \"torch\" | \"unknown\", how) from the modeling module of the layers (a module object or None)."""
    import sys
    mod = sys.modules.get(model_module) if isinstance(model_module, str) else model_module
    if mod is not None:
        fn = getattr(mod, "chunk_gated_delta_rule", None)
        if fn is not None:
            return ("fla" if _is_fla(fn) else "torch"), "the module's chunk_gated_delta_rule"
        flag = getattr(mod, "is_fast_path_available", None)
        if isinstance(flag, bool):
            return ("fla" if flag else "torch"), "the module's is_fast_path_available flag"
        ref = getattr(mod, "torch_chunk_gated_delta_rule", None)
        if ref is not None:
            seen = _callables_in(ref) + [v for v in vars(mod).values() if callable(v) and not isinstance(v, type)]
            if any(_is_fla(v) for v in seen if callable(v)):
                return "fla", "an fla callable next to the torch reference"
    if FALLBACK_WARNINGS:
        return "torch", "transformers warned about a missing fast path: " + FALLBACK_WARNINGS[0][:120]
    return "unknown", "no evidence in the module or in the warnings"


def _conv_of(fn) -> str:
    """Where the short-convolution function comes from, by its module: the causal-conv1d package, the kernels hub, or a plain PyTorch reference (a function that lives in transformers or anywhere else)."""
    origin = _origin(fn)
    top = origin.split(".")[0]
    if top in ("kernels", "hf_kernels") or "kernels" in origin.split(".")[:2]:
        return "kernels-hub"
    if top == "causal_conv1d":
        return "causal_conv1d"
    return "torch"


def _conv_fallback_warned() -> bool:
    """transformers logged "`causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed" (or the package cannot be found)."""
    for w in FALLBACK_WARNINGS:
        low = w.lower()
        if "causal_conv1d" in low and ("falling back" in low or "not installed" in low or "reference" in low):
            return True
    return False


def gdn_kernels(model):
    """Which kernels the GatedDeltaNet (linear-attention) layers of a loaded model really use, found by looking at the model, not at what is installed: {"delta_rule": "fla" | "torch" | "unknown",
    "conv": / "causal_conv1d": "causal_conv1d" | "kernels-hub" | "torch" | "unknown", "how": what the answer is based on, "module": the layers' module}. None when the model has no such layer (or cannot
    be looked at). Works for the older layers that carry `chunk_gated_delta_rule` AND for transformers 5.23, where the choice sits in the modeling module."""
    try:
        layer = None
        for m in model.modules():
            if "gateddeltanet" in type(m).__name__.lower() or getattr(m, "chunk_gated_delta_rule", None) is not None:
                layer = m
                break
        if layer is None:
            return None
        fn = getattr(layer, "chunk_gated_delta_rule", None)
        if fn is not None:                                  # older versions: the layer holds the function
            delta, how = ("fla" if _is_fla(fn) else "torch"), "the layer's chunk_gated_delta_rule"
        else:
            delta, how = _delta_rule(type(layer).__module__)
        conv_fn = getattr(layer, "causal_conv1d_fn", None)
        if conv_fn is None and not hasattr(layer, "causal_conv1d_fn"):
            import sys
            mod = sys.modules.get(type(layer).__module__)
            cf = getattr(mod, "causal_conv1d_fn", "missing") if mod is not None else "missing"
            conv_fn = None if cf == "missing" else cf
            conv = "unknown" if cf == "missing" else ("torch" if cf is None else _conv_of(cf))
        elif conv_fn is None:
            conv = "torch"
        else:
            conv = _conv_of(conv_fn)
        if conv in ("causal_conv1d", "kernels-hub", "unknown") and _conv_fallback_warned():
            conv = "torch"                                  # the package is not there: whatever name the function has, transformers said it runs its reference PyTorch version
        return {"delta_rule": delta, "conv": conv, "causal_conv1d": conv, "how": how, "module": type(layer).__module__}
    except Exception:       # noqa: BLE001 - looking must never stop a job
        return None


def gdn_note(info) -> str:
    """A plain sentence about slow kernels, or "" when the fast ones run (or there is nothing to say)."""
    if not info:
        return ""
    if info.get("delta_rule") == "unknown":
        return "It could not be found out which GatedDeltaNet kernels run (" + str(info.get("how", "")) + "): measure the speed (Heal card, Measure heal speed) before trusting it."
    if info.get("delta_rule") != "fla":
        return "GatedDeltaNet runs on the slow PyTorch kernels (flash-linear-attention not installed or its import failed)."
    if info.get("conv") == "torch":
        return "The short convolution of GatedDeltaNet uses the PyTorch version (causal-conv1d not installed); that is a small part of the time."
    return ""


def report_kernels(model):
    """Looks at the loaded model, logs `kernels: {...}` once (and the note when the slow ones run) and adds `kernels` to the job's result. Returns the info or None."""
    info = gdn_kernels(model)
    if info is None:
        return None
    import json
    from . import protocol
    if protocol.EXTRAS.get("kernels") != info:
        protocol.log("kernels: " + json.dumps(info, sort_keys=True))
        note = gdn_note(info)
        if note:
            protocol.log(note)
        protocol.add_extra("kernels", info)
    return info


def kernel_notes(family: str, backend: str, cuda: bool, capability=None) -> list:
    """Plain-language notes about speed that the window shows next to a TRAINING run (the other jobs report what really ran: `report_kernels`)."""
    notes = []
    if family == "qwen3_5":
        if backend == "unsloth":
            notes.append("Training with Unsloth: Qwen3.5 uses Unsloth's own fast kernels; the first steps can be slow while they compile. Measuring, healing and checking do not use Unsloth: they use the "
                         "flash-linear-attention and causal-conv1d packages when installed.")
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


ATTENTION_CHOICES = ("auto", "sdpa", "cudnn", "flash_attention_2", "flex_attention")


_cudnn_stack = None


def cudnn_status(cuda: bool) -> tuple:
    """(possible, reason) for cuDNN attention: PyTorch's own scaled_dot_product_attention backend that calls NVIDIA's cuDNN (PyTorch 2.5+, CUDA only)."""
    if not cuda:
        return False, "cuDNN attention needs a graphics card"
    try:
        import torch
        from torch.nn.attention import SDPBackend
        if not hasattr(SDPBackend, "CUDNN_ATTENTION") or not torch.backends.cuda.is_available() or not torch.backends.cudnn.is_available():
            return False, "this PyTorch has no cuDNN attention (needs PyTorch 2.5 or newer with cuDNN)"
    except Exception as exc:
        return False, f"cuDNN attention could not be checked ({type(exc).__name__})"
    return True, ""


def enable_cudnn() -> str:
    """Puts cuDNN first among PyTorch's attention backends for the rest of this process, the others behind it (a case cuDNN cannot do, such as an
    unsupported mask or head size, falls to the next one instead of failing). Returns "" or a note."""
    global _cudnn_stack
    import contextlib
    from torch.nn.attention import SDPBackend, sdpa_kernel
    order = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    stack = contextlib.ExitStack()
    try:
        stack.enter_context(sdpa_kernel(order, set_priority=True))
    except TypeError:   # PyTorch 2.5: no priority argument, so cuDNN is only allowed, not preferred
        return "this PyTorch cannot rank attention backends (needs 2.6+), so cuDNN is allowed but may not be the one chosen"
    _cudnn_stack = stack
    return ""


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
    if choice == "cudnn":
        ok, why = cudnn_status(cuda)
        return ("sdpa", "") if ok else (None, why + "; using the default attention")
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
