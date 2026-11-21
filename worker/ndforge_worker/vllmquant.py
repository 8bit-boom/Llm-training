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
import re
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


def is_local_folder(model: str) -> bool:
    """A folder on this computer (an absolute path, a relative one, a Windows drive path, or an existing folder), not a Hub id."""
    m = str(model or "")
    return Path(m).is_dir() or Path(m).is_absolute() or m.startswith(("./", "../", "~", ".\\", "..\\")) or bool(re.match(r"^[A-Za-z]:[\\/]", m)) or m.startswith("\\\\")


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


# ------------------------------------------------------------------------------------------------------------------ the fit: an explicit sum, not a fixed reserve
# Measured on the owner's PC (4th and 5th self-test, 27B AWQ): weights 18.2 GiB, peak activation 0.29, CUDA graphs 0.27, transient 0.87 (about 1.5 GB of overhead in all); the KV cache is about
# 0.118 GB per 1000 tokens (71,141 tokens = 8.4 GB). One 8,192-token sequence needs about 1 GB of cache, so a minimum of 2.5 GB (about 21 thousand tokens) leaves plenty of room.
OVERHEAD_GB = 1.5
MIN_KV_GB = 2.5
KV_GB_PER_1K = 0.118
ANSWER_TOKENS = 3300               # a typical thinking answer, for "how many answers fit at once"
FREE_MARGIN_GB = 0.6               # what vllmengine.plan_gpu_util keeps free next to the engine
MAX_UTIL = 0.92                    # the share of the card the engine may be raised to, never more
DEFAULT_MAX_SEQS = 40              # vllmengine.DEFAULT_MAX_SEQS
NF4_BODY_BYTES = 0.53              # 4 bits plus the block scales of bitsandbytes (double quantization)
KEEP_16BIT = re.compile(r"(embed|lm_head|wte|visual|vision|mm_proj|multi_modal|audio|patch_embed)", re.I)
_DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def route_weights_gb(model: str) -> dict | None:
    """The size of the weights in GB for each way of loading, from the safetensors HEADERS of a model folder: {"bf16", "fp8", "nf4"}. The embedding, the output layer and the vision/audio tensors stay
    16-bit (2 bytes); the transformer body (every other 2-D weight) is 1 byte in 8-bit and about 0.53 byte in 4-bit; small tensors (norms, biases) keep their size. None when the headers cannot be read."""
    p = Path(str(model or ""))
    files = sorted(p.glob("*.safetensors")) if p.is_dir() else []
    if not files:
        return None
    total = fp8 = nf4 = 0.0
    try:
        for f in files:
            with open(f, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                if not 2 <= n <= 100_000_000:
                    raise ValueError("not a safetensors header")
                header = json.loads(fh.read(n).decode("utf-8"))
            for name, t in header.items():
                if name == "__metadata__" or not isinstance(t, dict):
                    continue
                size = _DTYPE_BYTES[t["dtype"]]
                numel = 1
                for d in t["shape"]:
                    numel *= int(d)
                raw = numel * size
                body = len(t["shape"]) >= 2 and not KEEP_16BIT.search(name) and t["dtype"] in ("BF16", "F16", "F32")
                total += raw
                fp8 += numel * 1.0 if body else raw
                nf4 += numel * NF4_BODY_BYTES if body else raw
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return {"bf16": round(total / 1e9, 2), "fp8": round(fp8 / 1e9, 2), "nf4": round(nf4 / 1e9, 2)}


def _budgets(vram: float, util: float, free: float | None) -> dict:
    """The memory the engine may use: `normal` = the configured share of the card (lowered to the real free memory minus 0.6 GB when that is known), `raised` = up to the free memory minus
    0.6 GB but never above MAX_UTIL of the card (equal to `normal` when the free memory is not known)."""
    conf = vram * util
    def one(gb: float, why: str) -> dict:
        return {"gb": round(gb, 1), "util": round(max(gb / vram - 0.0049, 0.0), 2), "why": why}
    if free is None:
        n = one(conf, f"{util:g} of the {vram:g} GB card; the free memory is not known")
        return {"normal": n, "raised": n}
    cap = free - FREE_MARGIN_GB
    normal = one(min(conf, cap), f"{util:g} of the {vram:g} GB card" if conf <= cap else f"free {free:g} GB minus {FREE_MARGIN_GB:g}")
    raised = one(min(vram * MAX_UTIL, cap), f"{min(vram * MAX_UTIL, cap) / vram:.2f} of the card: free {free:g} GB minus {FREE_MARGIN_GB:g}" if cap < vram * MAX_UTIL else f"{MAX_UTIL:g} of the {vram:g} GB card")
    return {"normal": normal, "raised": raised if raised["gb"] > normal["gb"] else normal}


def _place(w: float, bud: dict) -> dict:
    """Does `w` GB of weights fit? {"ok", "budget": the budget used, "raised", "need", "left_cache_gb", "answers", "arith": plain words}. The normal budget first; the raised one only when needed."""
    need = round(w + OVERHEAD_GB + MIN_KV_GB, 1)
    for key in ("normal", "raised"):
        b = bud[key]
        if need <= b["gb"]:
            cache = round(b["gb"] - w - OVERHEAD_GB, 1)
            answers = max(1, int(cache / (KV_GB_PER_1K * ANSWER_TOKENS / 1000)))
            return {"ok": True, "budget": b, "raised": key == "raised" and bud["raised"] is not bud["normal"], "need": need, "left_cache_gb": cache, "answers": answers,
                    "arith": f"weights {w:g} + overhead {OVERHEAD_GB:g} = {round(w + OVERHEAD_GB, 1)} GB of {b['gb']} GB ({b['why']}), leaving {cache} GB of cache: about {answers} answers of {ANSWER_TOKENS / 1000:g}k tokens at once"}
    b = bud["raised"]
    return {"ok": False, "budget": b, "raised": False, "need": need, "left_cache_gb": None, "answers": 0,
            "arith": f"weights {w:g} + overhead {OVERHEAD_GB:g} + minimum cache {MIN_KV_GB:g} = {need} GB; budget {b['gb']} GB ({b['why']})"}


def _fitted(base: dict, quant: str, pl: dict, reason: str, **more) -> dict:
    out = {**base, "quant": quant, "fits": pl["ok"], "reason": reason, "gpu_util": pl["budget"]["util"], "arith": pl["arith"], **more}
    if pl["ok"]:
        out["max_seqs"] = min(DEFAULT_MAX_SEQS, pl["answers"])
        if pl["raised"]:
            out["reason"] += f" The share of the card is raised to {pl['budget']['util']:g} for this (the free memory allows it)."
    return out


def seq_cap(weights_gb: float, vram_gb: float, util: float, free_gb, tokens_per_row: int, default: int = 32) -> tuple:
    """(max_num_seqs, a plain sentence) for the in-container scoring script: `default` rows at once, at most what the cache left after the weights and the overhead can hold at `tokens_per_row` tokens
    each (about 0.118 GB per 1000 tokens, measured; a hybrid model needs less for its attention layers and has its own block limit, which the engine reports and the script then retries with), never
    below 1. The sentence goes to the job log before the engine starts."""
    bud = _budgets(float(vram_gb), float(util), None if free_gb is None else float(free_gb))
    b = bud["normal"]["gb"]                                  # the engine's own budget (the configured share, lowered to what is free), not the raised one
    cache = max(0.0, b - float(weights_gb) - OVERHEAD_GB)
    per_row = KV_GB_PER_1K * max(1, int(tokens_per_row)) / 1000
    fit = max(1, int(cache / per_row)) if per_row else default
    n = max(1, min(int(default), fit))
    return n, (f"Rows scored at once: {n} (cache left after the weights and overhead about {cache:.1f} GB of the {b} GB budget, about {per_row:.2f} GB per row of {int(tokens_per_row)} tokens; "
               "a hybrid Mamba/attention model also has a block limit of its own that the engine reports at its start, and the script then retries once with that number).")


def pick(model: str, engine: dict | None = None) -> dict:
    """{"model": what the server loads, "quant": "none" | "fp8" | "bnb-nf4" | "prequant:<repo>" | "stored:<method>", "fits": bool | None, "weights_gb", "usable_gb", "reason": a plain sentence,
    and where a fit was calculated: "gpu_util", "max_seqs", "arith", "route_hint", "like_for_like"}. The engine keys read here: `vram_gb`, `gpu_memory_utilization`, `gpu_free_gb` (the real free
    memory, when known), `weights_gb`, `quant`, `original`, `route_hint`."""
    e = engine or {}
    vram = float(e.get("vram_gb") or DEFAULT_VRAM_GB)
    util = float(e.get("gpu_memory_utilization") or GPU_UTIL)
    free = float(e["gpu_free_gb"]) if e.get("gpu_free_gb") not in (None, "") else None
    bud = _budgets(vram, util, free)
    usable = round(bud["normal"]["gb"] - OVERHEAD_GB - MIN_KV_GB, 1)           # the most weights that fit next to the overhead and the minimum cache
    given = str(e.get("quant") or "auto")
    sizes = None if e.get("weights_gb") else route_weights_gb(model)
    gb = float(e["weights_gb"]) if e.get("weights_gb") else (sizes["bf16"] if sizes else weights_gb(model))
    base = {"model": model, "weights_gb": gb, "usable_gb": usable}
    if given == "bnb-nf4":
        # asked for: the in-flight 4-bit route on THIS model (the self-test "Test the in-flight 4-bit route"): the fit is calculated, nothing else is decided
        if gb is None:
            return {**base, "quant": "bnb-nf4", "fits": None, "reason": "As you chose: the in-flight 4-bit route (UNVERIFIED); the size of the weights is not known, so the fit is not checked."}
        w4_ = sizes["nf4"] if sizes else round(gb * BNB_SHARE, 1)
        pl = _place(w4_, bud)
        return _fitted(base, "bnb-nf4", pl, f"As you chose: the in-flight 4-bit route (UNVERIFIED): {pl['arith']}.", served=f"{model} (4-bit on the fly)")
    if given != "auto":
        if given in ("none", "fp8"):
            return {**base, "quant": given, "fits": None, "reason": f"As you chose: {given}."}
        return {**base, "model": given, "quant": f"prequant:{given}", "fits": None, "reason": f"As you chose: the pre-quantized checkpoint {given} is served."}
    stored = stored_quantization(model)
    if stored:
        return {**base, "quant": f"stored:{stored}", "fits": None, "reason": f"The folder already holds {stored} weights; they are loaded as stored."}
    if gb is None and not e.get("original"):
        return {**base, "quant": "none", "fits": None, "reason": "The size of the weights is not known (a Hub id): loaded as they are."}
    hint = str(e.get("route_hint") or "")
    w8 = lambda: (sizes["fp8"] if sizes else round(gb * FP8_SHARE, 1))          # noqa: E731 - the 8-bit size: from the headers, else the share
    w4 = lambda: (sizes["nf4"] if sizes else round(gb * BNB_SHARE, 1))          # noqa: E731
    if gb is not None:
        pl = _place(gb, bud)
        if pl["ok"]:
            out = _fitted(base, "none", pl, f"The 16-bit weights fit: {pl['arith']}.")
            if hint and not e.get("original"):
                out.update(route_hint=hint, like_for_like=hint == "none")
                if hint != "none":
                    out["reason"] += f" The original of this run was served as {hint}: different quantization, not like-for-like."
            return out
    if e.get("original"):
        # the owner's ORIGINAL does not fit in 16 bits: the identity rule decides what stands in for it (a confirmed copy, or its own folder in 4 bits), never the base model's checkpoint
        origin = str(e.get("original_repo") or model)
        try:
            sel = select_original(e.get("teacher") or {}, origin, model)
        except IdentityError as exc:
            return {**base, "quant": "none", "fits": False, "reason": str(exc)}
        if sel["route"] == "prequant":
            return {**base, "model": sel["model"], "quant": f"prequant:{sel['model']}", "fits": True, "served": f"confirmed copy {sel['model']}",
                    "reason": f"The original does not fit in 16 bits ({usable} GB of weights fit); {sel['reason']} Its answers differ a little from the original's."}
        if gb is None:
            return {**base, "quant": "bnb-nf4", "fits": None, "served": f"original: {model} (4-bit on the fly)", "reason": "The original's size is not known; it is served from its own folder, quantized to 4 bits while loading (UNVERIFIED)."}
        pl = _place(w4(), bud)
        if not pl["ok"]:
            return {**base, "quant": "bnb-nf4", "fits": False, "gpu_util": pl["budget"]["util"], "arith": pl["arith"],
                    "reason": f"{origin} does not fit even in 4 bits ({pl['arith']}) and no confirmed quantized copy of it is set. The standard engine is used; the base model's pre-quantized checkpoint is NOT used for another model."}
        return _fitted(base, "bnb-nf4", pl, f"The original does not fit in 16 bits ({usable} GB of weights fit); it is served from its own folder, quantized to 4 bits while loading: {pl['arith']} "
                       "(the in-flight bitsandbytes route: UNVERIFIED, the engine may refuse it and the standard engine then answers).", served=f"original: {model} (4-bit on the fly)")
    if gb is not None and is_local_folder(model):
        return _own_folder(base, gb, w8(), w4(), bud, hint)
    if gb is not None:
        pl = _place(w8(), bud)
        if pl["ok"]:
            return _fitted(base, "fp8", pl, f"The 16-bit weights (about {gb} GB) do not fit; they are converted to 8 bits when loaded (about {w8()} GB): {pl['arith']}.")
        return _fitted(base, "fp8", pl, f"The weights do not fit on this card even in 8 bits ({pl['arith']}): use the standard engine.")
    return {**base, "quant": "none", "fits": False, "reason": "The weights do not fit."}


def _own_folder(base: dict, gb: float, w8: float, w4: float, bud: dict, hint: str) -> dict:
    """A model folder that is NOT the original (a cut, a healed or merged model, the owner's own finetune) that does not fit in 16 bits: 8 bits when it fits, else the in-flight 4-bit route
    (bitsandbytes, the same one the original gets; UNVERIFIED for this model) with the same explicit fit. NEVER a stand-in checkpoint. `hint` is the route the original of the same run was served
    with: it is preferred when it fits (a comparison of the original, the cut and the healed model is only fair when they are quantized the same way); the result says when it could not be matched."""
    size = {"fp8": w8, "bnb-nf4": w4}
    order = ["fp8", "bnb-nf4"]
    if hint in size:
        order.remove(hint)
        order.insert(0, hint)
    placed = {r: _place(size[r], bud) for r in order}
    for route in order:
        pl = placed[route]
        if not pl["ok"]:
            continue
        like = None if not hint else hint == route
        if route == "fp8":
            why = f"The 16-bit weights (about {gb} GB) do not fit; they are converted to 8 bits when loaded (about {w8} GB): {pl['arith']}."
        else:
            miss = f" 8 bits would not fit ({placed['fp8']['arith']})." if not placed["fp8"]["ok"] else ""
            why = (f"The 16-bit weights (about {gb} GB) do not fit.{miss} They are quantized to 4 bits while loading (about {w4} GB): {pl['arith']}. This is the in-flight bitsandbytes route"
                   f"{', the same as the original of this run, so the comparison is like-for-like' if hint == 'bnb-nf4' else ''}; UNVERIFIED for this model (the original uses it; the engine may still refuse it "
                   "and the standard engine then answers). No stand-in checkpoint is ever used for your own model.")
        out = _fitted(base, route, pl, why, served=f"{base['model']} ({'8-bit' if route == 'fp8' else '4-bit'} on the fly)", route_hint=hint or None, like_for_like=like)
        if like is False:
            out["reason"] += f" The original of this run was served as {hint}: different quantization, not like-for-like."
        return out
    last = placed["bnb-nf4"]
    return {**base, "quant": "bnb-nf4", "fits": False, "route_hint": hint or None, "like_for_like": None if not hint else False, "gpu_util": last["budget"]["util"], "arith": last["arith"],
            "reason": (f"The weights do not fit on this card even in 4 bits. 8 bits: {placed['fp8']['arith']}. 4 bits: {last['arith']}. Nothing was changed: more free memory on the card "
                       "(close other programs) or a smaller model would be needed. The standard engine is used.")}


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


