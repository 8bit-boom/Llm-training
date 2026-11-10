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
                raise RuntimeError(f"Unsloth could not be loaded: {exc}") from exc
            protocol.log(f"Unsloth is not available ({type(exc).__name__}: {exc}); using the plain Transformers + PEFT fallback.")
    return "hf"


def _precision():
    import torch
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return ("bf16", torch.bfloat16) if major >= 8 else ("fp16", torch.float16)
    return "fp32", torch.float32


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
    model, tokenizer = loader.from_pretrained(**kw)
    if init or full:   # a loaded adapter, or full fine-tuning: every weight is trained, no LoRA layers to add
        return model, tokenizer, loader
    protocol.progress("load", 0.8, "Adding the LoRA layers")
    common = {"r": h["rank"], "lora_alpha": h["alpha"], "lora_dropout": 0, "bias": "none", "random_state": h["seed"]}
    if loader is FastModel:
        model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                         finetune_attention_modules=True, finetune_mlp_modules=True, **common)
    else:
        model = FastLanguageModel.get_peft_model(model, target_modules=QWEN_TARGETS, use_gradient_checkpointing="unsloth",
                                                 max_seq_length=h["max_seq"], **common)
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
    if h.get("padding_free") == "on" and speed.padding_free_status(cfg["model"]["family"], "hf", torch.cuda.is_available(), torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)[0]:
        kwargs["attn_implementation"] = "flash_attention_2"     # padding-free batching tells FlashAttention-2 where each example starts
    if cfg["method"] == "qlora":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, **kwargs)
    except TypeError:  # an older transformers still calls it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, **kwargs)
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
    lora = LoraConfig(r=h["rank"], lora_alpha=h["alpha"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
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
