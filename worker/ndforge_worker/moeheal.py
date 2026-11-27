"""Healing a cut Inkling-2-Mini (docs/MOE_GUIDE.md section 4): the parts that differ from a dense heal.

* the student keeps its experts packed and FROZEN (`PackedExperts` holds buffers, never parameters); LoRA goes on the dense parts only (`lora_targets`);
* the loss sees the model's own output: logits bounded by the final softcap (30) over the unpadded vocabulary (200,058 rows; the checkpoint's padded rows are cut at load);
* training and teacher text use Inkling's segment format (`format_turn`).
The packed in-process teacher is the loaded packed model itself (the same `load_packed`); vLLM nightly is tried first by the caller and this is its fallback.
"""
from __future__ import annotations

from . import families

DENSE_LORA_SUFFIXES = ("q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear")


def lora_targets(module_names) -> list[str]:
    """names of the linear modules that get a LoRA: attention projections of every layer; never anything under `mlp.experts` (packed), the routers or the embeddings"""
    out = []
    for n in module_names:
        if ".mlp." in n or "embed" in n or "lm_head" in n:
            continue
        if n.endswith(DENSE_LORA_SUFFIXES):
            out.append(n)
    return out


def freeze_packed(model) -> int:
    """Makes sure no packed-experts tensor trains: returns how many trainable parameters sit under an experts module (must be 0)"""
    bad = 0
    for n, p in model.named_parameters():
        if ".mlp.experts." in n:
            p.requires_grad_(False)
        if ".mlp.experts." in n and p.requires_grad:
            bad += 1
    return bad


def format_turn(thinking: str, answer: str) -> str:
    """One assistant turn in Inkling's segments: a reasoning segment (when there is a trace) then the answer segment, ended with the sampling-end token"""
    t = families.family("inkling2")["think"]
    out = ""
    if thinking and thinking.strip():
        out += f"{t['turn_open']}{t['open']}{thinking.strip()}{t['close']}"
    return out + f"{t['turn_open']}{t['answer_open']}{answer.strip()}{t['close']}<|content_model_end_sampling|>"


SOFTCAP = 30.0
UNPADDED_VOCAB = 200_058


def softcap_of(cfg: dict) -> float | None:
    """the final logit softcap a model's config asks for (None when it has none: every dense model)"""
    tc = cfg.get("text_config") or cfg
    v = tc.get("final_logit_softcapping")
    return float(v) if v else None
