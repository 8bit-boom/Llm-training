"""Which weights the vLLM engine loads (EXPERIMENTAL; the sizes are the research report's estimates, not measured here).

`pick` is a pure function (a table of rules) so it is tested without a GPU:

    1. the engine's `quant` is given ("none", "fp8", or a pre-quantized id): it is used as it is.
    2. the folder already holds quantized weights (its config.json has a `quantization_config`): loaded as stored, nothing is added.
    3. bf16 fits next to the cache (weights <= the usable memory minus CACHE_GB): "none".
    4. the model is the ORIGINAL (`engine.original`) and a pre-quantized checkpoint is configured (`engine.teacher_repo`): that checkpoint is served instead ("prequant:<repo>").
    5. otherwise (a cut): `--quantization fp8` at load when the 8-bit weights fit (about FP8_SHARE of the bf16 size: the embedding and the output layer stay 16-bit); else it does not fit.

`--language-model-only` is ALWAYS passed (vllmengine.serve_args). A weight size that is not known (a Hub id that is not the original) is assumed to fit and the engine says so if not.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_VRAM_GB = 32.0
GPU_UTIL = 0.88                    # vllmengine.DEFAULT_GPU_UTIL (the share of the card the engine reserves)
CACHE_GB = 6.0                     # what the attention cache and the linear-attention state need for ~25-40 thinking answers (report: 6-9 GB left)
FP8_SHARE = 0.56                   # about 14.5 GB body + 5.1 GB bf16 embedding / head of 34 GB for the 17 B cut in the report (19.6 / 34)


def weights_gb(model: str) -> float | None:
    """The size of the weights of a model FOLDER in GB (its .safetensors files); None for a Hub id or an empty folder."""
    p = Path(str(model or ""))
    if not p.is_dir():
        return None
    total = sum(f.stat().st_size for f in p.glob("*.safetensors"))
    return round(total / 1e9, 2) if total else None


def stored_quantization(model: str) -> str | None:
    try:
        cfg = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    q = cfg.get("quantization_config") if isinstance(cfg, dict) else None
    return str((q or {}).get("quant_method") or "quantized") if q else None


def pick(model: str, engine: dict | None = None) -> dict:
    """{"model": what the server loads, "quant": "none" | "fp8" | "prequant:<repo>" | "stored:<method>", "fits": bool | None, "weights_gb", "usable_gb", "reason": a plain sentence}."""
    e = engine or {}
    vram = float(e.get("vram_gb") or DEFAULT_VRAM_GB)
    usable = round(vram * float(e.get("gpu_memory_utilization") or GPU_UTIL) - CACHE_GB, 1)
    given = str(e.get("quant") or "auto")
    gb = float(e["weights_gb"]) if e.get("weights_gb") else weights_gb(model)
    base = {"model": model, "weights_gb": gb, "usable_gb": usable}
    if given != "auto":
        if given in ("none", "fp8"):
            return {**base, "quant": given, "fits": None, "reason": f"As you chose: {given}."}
        return {**base, "model": given, "quant": f"prequant:{given}", "fits": None, "reason": f"As you chose: the pre-quantized checkpoint {given} is served."}
    stored = stored_quantization(model)
    if stored:
        return {**base, "quant": f"stored:{stored}", "fits": None, "reason": f"The folder already holds {stored} weights; they are loaded as stored."}
    if gb is None and not e.get("original"):
        return {**base, "quant": "none", "fits": None, "reason": "The size of the weights is not known (a Hub id): loaded as they are."}
    if gb is not None and gb <= usable:
        return {**base, "quant": "none", "fits": True, "reason": f"The 16-bit weights (about {gb} GB) fit next to the cache ({usable} GB usable)."}
    repo = str(e.get("teacher_repo") or "")
    if e.get("original") and repo:
        return {**base, "model": repo, "quant": f"prequant:{repo}", "fits": True,
                "reason": f"The original does not fit in 16 bits (usable {usable} GB); the pre-quantized checkpoint {repo} is served instead (its answers differ a little from the original's)."}
    if gb is not None and gb * FP8_SHARE <= usable:
        return {**base, "quant": "fp8", "fits": True, "reason": f"The 16-bit weights (about {gb} GB) do not fit ({usable} GB usable); they are converted to 8 bits when loaded (about {round(gb * FP8_SHARE, 1)} GB)."}
    return {**base, "quant": "fp8" if gb is not None else "none", "fits": False,
            "reason": f"The weights do not fit on this card even in 8 bits (about {round((gb or 0) * FP8_SHARE, 1)} GB wanted, {usable} GB usable): use the standard engine."}
