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
# THE IDENTITY RULE (one place: the teacher cache, the thinking checks, trace generation and the self-test's "original" runs all use `select_original`): a pre-quantized checkpoint is a quantization of ONE
# repo, so it stands in for an original only when the original's id is in the table of confirmed pairs (default: the base model and nothing else) or the owner confirmed that copy for THIS original.
# Anything else (a finetune) is served from its own folder, quantized to 4 bits while loading (bitsandbytes: UNVERIFIED how vLLM 0.34 spells it), or refused: never the base model's checkpoint.
DEFAULT_PAIRS = {"Qwen/Qwen3.8-27B": "cyankiwi/Qwen3.8-27B-AWQ-INT4"}
BNB_ARGS = ["--quantization", "bitsandbytes"]                               # UNVERIFIED (the report: bitsandbytes moved into the plugin `vllm-bnb-plugin` 0.0.3)
BNB_SHARE = 1 / 3.6                                                         # 4-bit weights are about this share of the 16-bit bytes (as the program's own estimates)


class IdentityError(Exception):
    """The identity rule refuses (plain words)."""


def norm_id(x) -> str:
    return str(x or "").strip().rstrip("/").lower()
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
    if e.get("original"):
        # the owner's ORIGINAL does not fit in 16 bits: the identity rule decides what stands in for it (a confirmed copy, or its own folder in 4 bits), never the base model's checkpoint
        origin = str(e.get("original_repo") or model)
        try:
            sel = select_original(e.get("teacher") or {}, origin, model)
        except IdentityError as exc:
            return {**base, "quant": "none", "fits": False, "reason": str(exc)}
        if sel["route"] == "prequant":
            return {**base, "model": sel["model"], "quant": f"prequant:{sel['model']}", "fits": True, "served": f"confirmed copy {sel['model']}",
                    "reason": f"The original does not fit in 16 bits (usable {usable} GB); {sel['reason']} Its answers differ a little from the original's."}
        need = round(gb * BNB_SHARE, 1) if gb is not None else None
        if need is not None and need > usable:
            return {**base, "quant": "bnb-nf4", "fits": False, "reason": f"{origin} does not fit even in 4 bits ({need} GB wanted, {usable} GB usable) and no confirmed quantized copy of it is set. The standard engine is used; the base model's pre-quantized checkpoint is NOT used for another model."}
        return {**base, "quant": "bnb-nf4", "fits": None if need is None else True, "served": f"original: {model} (4-bit on the fly)",
                "reason": f"The original does not fit in 16 bits (usable {usable} GB); it is served from its own folder, quantized to 4 bits while loading (the in-flight bitsandbytes route: UNVERIFIED, the engine may refuse it and the standard engine then answers)."}
    if gb is not None and gb * FP8_SHARE <= usable:
        return {**base, "quant": "fp8", "fits": True, "reason": f"The 16-bit weights (about {gb} GB) do not fit ({usable} GB usable); they are converted to 8 bits when loaded (about {round(gb * FP8_SHARE, 1)} GB)."}
    return {**base, "quant": "fp8" if gb is not None else "none", "fits": False,
            "reason": f"The weights do not fit on this card even in 8 bits (about {round((gb or 0) * FP8_SHARE, 1)} GB wanted, {usable} GB usable): use the standard engine."}


def select_original(tcfg: dict, origin: str, origin_dir) -> dict:
    """Which weights the engine loads for the original `origin` (the original's repo id or folder, a Hub id or a folder; `origin_dir` is its local folder).
    `tcfg`: {"route": "auto" | "bnb" | "prequant", "pairs": {original: quantized}, "confirmed": {original: quantized}, "quantized_repo": str, "quantized_rev": str, "bnb_args": [...], "bnb_pip": str}.
    -> {"route": "prequant" | "bnb", "model", "quant": "bnb-nf4" | "awq:<repo>@<rev>", "quantized_repo", "reason"}; IdentityError when the route asked for cannot be used for this original."""
    route = str(tcfg.get("route") or "auto").lower()
    if route not in ("auto", "bnb", "prequant"):
        raise IdentityError(f"Unknown vLLM teacher route {route!r}: use auto, bnb or prequant.")
    pairs = {norm_id(k): str(v) for k, v in {**DEFAULT_PAIRS, **(tcfg.get("pairs") or {})}.items()}
    confirmed = {norm_id(k): str(v) for k, v in (tcfg.get("confirmed") or {}).items()}
    mine = norm_id(origin)
    paired = pairs.get(mine) or confirmed.get(mine)
    asked = str(tcfg.get("quantized_repo") or "")
    if asked and norm_id(asked) != norm_id(paired) and norm_id(confirmed.get(mine)) != norm_id(asked):
        raise IdentityError(f"{asked} is not a confirmed quantized copy of {origin}. Tick \"this repo is a quantized copy of my original model\" for it, or use the in-flight 4-bit route on your own folder.")
    if asked and not paired:
        paired = asked if norm_id(confirmed.get(mine)) == norm_id(asked) else None
    if route == "prequant" and not paired:
        raise IdentityError(f"There is no confirmed quantized copy of {origin}: the pre-quantized checkpoint of another model would teach the wrong model. Use the in-flight 4-bit route, or confirm a copy in the settings.")
    if route in ("auto", "prequant") and paired:
        rev = str(tcfg.get("quantized_rev") or "unpinned")
        return {"route": "prequant", "model": paired, "quant": f"awq:{paired}@{rev}", "quantized_repo": paired,
                "reason": f"{paired} is the confirmed quantized copy of {origin}."}
    return {"route": "bnb", "model": str(origin_dir), "quant": "bnb-nf4", "quantized_repo": "",
            "reason": f"No confirmed quantized copy of {origin}: the original's own folder is loaded and quantized to 4 bits while loading (the in-flight bitsandbytes route)."}


