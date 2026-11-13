"""Loading a model with LoRA attached: Unsloth (preferred: faster, much less memory) or plain Transformers + PEFT (fallback).

`unsloth` must be imported before transformers / trl / peft, so `prepare()` is the first thing a training job calls."""
from . import protocol
from .families import QWEN_TARGETS, family

_unsloth = None


def prepare(backend: str) -> str:
    """Import Unsloth first when it may be used. Returns the backend that will really be used: "unsloth" or "hf"."""
    global _unsloth
    if backend in ("auto", "unsloth"):
        try:
            import unsloth  # noqa: F401  (the import itself patches transformers; order matters)
            _unsloth = unsloth
            return "unsloth"
        except Exception as exc:  # ImportError, or a CUDA/driver problem while importing
            if backend == "unsloth":
                raise RuntimeError(f"Unsloth could not be loaded: {exc}. Choose the standard engine instead: Expert mode, \"Training engine\" = standard "
                                   "(or install the training software without Unsloth in Training setup).") from exc
            protocol.log(f"Unsloth is not available ({type(exc).__name__}: {exc}); using the plain Transformers + PEFT fallback.")
    return "hf"


def _precision():
    import torch
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return ("bf16", torch.bfloat16) if major >= 8 else ("fp16", torch.float16)
    return "fp32", torch.float32


def _attention(cfg: dict):
    """The attn_implementation to ask for (None = library default); a note about a refused choice goes to the log."""
    import torch
    from . import speed
    cuda = torch.cuda.is_available()
    choice = cfg["hyper"].get("attention", "auto")
    attn, note = speed.attention_request(choice, cuda, torch.cuda.get_device_capability(0) if cuda else None)
    if choice == "cudnn" and attn:
        note = speed.enable_cudnn()
    if note:
        protocol.log("Attention: " + note + ".")
    return attn


def _load_with_attention(load, kwargs: dict, attn):
    """load(**kwargs, attn_implementation=attn); when the library refuses that attention for this model, load again with its default."""
    if not attn:
        return load(**kwargs)
    try:
        return load(attn_implementation=attn, **kwargs)
    except Exception as exc:
        protocol.log(f"Attention \"{attn}\" could not be used for this model ({type(exc).__name__}: {str(exc)[:200]}); loading again with the default attention.")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return load(**kwargs)


def lora_targets(h: dict) -> list:
    """The LoRA target modules: the standard list plus `hyper.extra_targets` (a grown model's switched-off layers need their output projection trained)."""
    out = list(QWEN_TARGETS)
    for t in h.get("extra_targets") or []:
        if t not in out:
            out.append(t)
    return out


def lora_layers(h: dict):
    """`hyper.layers_to_transform` (indices of the layers LoRA is attached to) as a list, or None for every layer."""
    layers = h.get("layers_to_transform")
    return [int(i) for i in layers] if layers else None


def load_unsloth(cfg: dict):
    from unsloth import FastLanguageModel, FastModel
    fam = family(cfg["model"]["family"])
    loader = FastModel if fam["loader"] == "FastModel" else FastLanguageModel
    h = cfg["hyper"]
    full = cfg["method"] == "full"
    kw = {"model_name": cfg.get("init_model") or cfg["model"]["repo"], "max_seq_length": h["max_seq"], "dtype": None, "full_finetuning": full}
    if cfg["method"] == "qlora":
        kw["load_in_4bit"] = True
    else:
        kw.update(load_in_4bit=False, load_in_16bit=True)
    init = cfg.get("init_adapter") if not full else None
    if init:   # continue from an earlier LoRA: Unsloth loads a LoRA folder as the model (base model + trainable adapter)
        kw["model_name"] = init
    protocol.progress("load", 0.05, "Loading the model")
    model, tokenizer = _load_with_attention(loader.from_pretrained, kw, _attention(cfg))
    if init or full:   # a loaded adapter, or full fine-tuning: every weight is trained, no LoRA layers to add
        return model, tokenizer, loader
    protocol.progress("load", 0.8, "Adding the LoRA layers")
    common = {"r": h["rank"], "lora_alpha": h["alpha"], "lora_dropout": 0, "bias": "none", "random_state": h["seed"]}
    if loader is FastModel:
        model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                         finetune_attention_modules=True, finetune_mlp_modules=True, **common)
    else:
        if h.get("extra_targets"):
            protocol.log("Also training: " + ", ".join(h["extra_targets"]) + " (the model has layers that were switched off when it was grown).")
        peft_kw = dict(target_modules=lora_targets(h), use_gradient_checkpointing="unsloth", max_seq_length=h["max_seq"], **common)
        layers = lora_layers(h)
        if layers:
            try:
                return FastLanguageModel.get_peft_model(model, layers_to_transform=layers, **peft_kw), tokenizer, loader
            except TypeError:
                protocol.log("Training only some layers is not supported by Unsloth here; training all layers.")
        model = FastLanguageModel.get_peft_model(model, **peft_kw)
    return model, tokenizer, loader


def load_hf(cfg: dict):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    h = cfg["hyper"]
    _, dtype = _precision()
    full = cfg["method"] == "full"
    if full and torch.cuda.is_available() and dtype != torch.bfloat16:
        raise RuntimeError("Full fine-tuning needs a graphics card with bf16 (RTX 30 series or newer). Use LoRA on this card.")
    repo = cfg.get("init_model") or cfg["model"]["repo"]
    protocol.progress("load", 0.05, "Loading the model")
    tokenizer = AutoTokenizer.from_pretrained(repo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {}
    from . import speed
    attn = _attention(cfg)
    if attn:
        kwargs["attn_implementation"] = attn
    if h.get("padding_free") == "on" and speed.padding_free_status(cfg["model"]["family"], "hf", torch.cuda.is_available(), torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)[0]:
        kwargs["attn_implementation"] = "flash_attention_2"     # padding-free batching tells FlashAttention-2 where each example starts
    if cfg["method"] == "qlora":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    def load(**kw):
        try:
            return AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, **kw)
        except TypeError:  # an older transformers still calls it torch_dtype
            return AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, **kw)
    ask = kwargs.pop("attn_implementation", None)
    model = _load_with_attention(load, kwargs, ask)
    if full:
        # every weight is trained: nothing to attach; checkpointing keeps the activations small
        if torch.cuda.is_available():
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        model.config.use_cache = False
        return model, tokenizer, None
    protocol.progress("load", 0.8, "Adding the LoRA layers")
    if cfg["method"] == "qlora":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    elif torch.cuda.is_available():
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if cfg.get("init_adapter"):
        from peft import PeftModel
        return PeftModel.from_pretrained(model, cfg["init_adapter"], is_trainable=True), tokenizer, None
    lora_kw = dict(r=h["rank"], lora_alpha=h["alpha"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
    if lora_layers(h):
        lora_kw.update(layers_to_transform=lora_layers(h), layers_pattern="layers")
    try:   # the image / audio parts of a multimodal model stay as they are: only the language layers learn (so the base model's vision file still fits)
        lora = LoraConfig(exclude_modules=r".*(vision|visual|audio|image|multi_modal|mm_projector|embed_vision|embed_audio).*", **lora_kw)
    except TypeError:   # an older PEFT has no exclude_modules
        lora = LoraConfig(**lora_kw)
    return get_peft_model(model, lora), tokenizer, None


def load(cfg: dict, backend: str):
    if backend == "unsloth":
        return load_unsloth(cfg)
    return load_hf(cfg)


def precision_name() -> str:
    return _precision()[0]


def to_inference(backend: str, loader, model):
    if backend == "unsloth" and loader is not None:
        loader.for_inference(model)
    model.eval()


def to_training(backend: str, loader, model):
    if backend == "unsloth" and loader is not None:
        loader.for_training(model)
    model.train()
