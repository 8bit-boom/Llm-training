"""Loading Inkling-2-Mini for measuring, Try first and healing without decoding its 20 GB of packed experts to 16-bit (which needs 48 GB).

Two paths:

* "hf"      transformers' own conversion (version 5.24 or newer): the experts are decoded to 16-bit while loading (about 48 GB; with `device_map="auto"` it spills to the computer's memory).
            Slow, but it is the library's own code, so it is the reference the packed path is checked against.
* "packed"  `load_packed`: the model is built on the meta device, every tensor that is not an expert is read and renamed with the table below (a copy of transformers' `inkling2_text` and
            `inkling2_loop_text` rules), and every `mlp.experts` is replaced by `PackedExperts`, which keeps the authors' NVFP4 / MXFP8 bytes on the card (about 16 GB) and decodes only the
            experts a batch uses, a few at a time, with the formula of transformers' `Inkling2LoopDequantizeExperts` (golden test in tests/test_inkling_load.py).

`dequant_*`, `PackedExperts` and `hf_name` need only torch; the model classes are imported from transformers when a model is built. The check of the table against the installed
transformers is `tests/test_inkling_load.py::test_the_rename_table_loads_a_tiny_real_model` (it skips on a transformers older than 5.24).
"""
from __future__ import annotations

import re

from . import moe_layout

MIN_TRANSFORMERS = "5.24.0"
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
EXPERTS_PER_CHUNK = 8


class InklingLoadError(RuntimeError):
    """plain-language message (the job turns it into an error code)"""


def transformers_ok() -> tuple[bool, str]:
    """(new enough, the installed version or "")"""
    try:
        from importlib import metadata
        v = metadata.version("transformers")
    except Exception:
        return False, ""
    parts = [int(x) for x in re.findall(r"\d+", v)[:3]] + [0, 0, 0]
    need = [int(x) for x in MIN_TRANSFORMERS.split(".")]
    return parts[:3] >= need, v


def require_transformers() -> None:
    ok, v = transformers_ok()
    if not ok:
        raise InklingLoadError(f"Inkling-2-Mini needs the training software updated: transformers {MIN_TRANSFORMERS} or newer"
                               + (f" (this computer has {v})." if v else " (it is not installed).") + " The Slim page shows the command; Qwen and Gemma keep working with the newer version.")


# --------------------------------------------------------------------------------------------------------------------- decoding

def dequant_nvfp4(weight, scale, amax, denominator, dtype=None):
    """NVFP4 experts -> [E, rows, 2 x bytes]: two E2M1 codes per byte (the even element in the low nibble), one fp8 scale per 16 values, and the per-expert factor amax / denominator."""
    import torch
    dtype = dtype or torch.bfloat16
    e, rows = weight.shape[:2]
    cols = weight.shape[2] * 2
    values = torch.tensor(E2M1, device=weight.device)
    byte = torch.arange(256, device=weight.device)
    table = torch.stack((values[byte & 0xF], values[byte >> 4]), dim=-1)
    block = scale.float()
    out = table[weight.to(torch.uint8).long()].reshape(e, rows, cols // 16, 16) * block[..., None]
    out = out * (amax.float() / denominator.float()).reshape(-1, 1, 1, 1)
    return out.reshape(e, rows, cols).to(dtype)


def dequant_mxfp8(weight, scale, dtype=None):
    """MXFP8 experts -> [E, rows, cols]: fp8 values, one power-of-two scale 2 ** (scale - 127) per 32 values"""
    import torch
    dtype = dtype or torch.bfloat16
    e, rows, cols = weight.shape
    out = weight.float().reshape(e, rows, cols // 32, 32) * torch.exp2(scale.float() - 127)[..., None]
    return out.reshape(e, rows, cols).to(dtype)


def softcapped_swiglu(gate, up, gate_cap: float, up_cap: float):
    """Inkling2LoopSoftcappedSwiGLU: the sigmoid sees the raw gate, the softcaps only bound the linear gate factor and `up`"""
    import torch
    dtype = gate.dtype
    g, u = gate.float(), up.float()
    return (torch.tanh(g / gate_cap) * gate_cap * torch.sigmoid(g) * torch.tanh(u / up_cap) * up_cap).to(dtype)


def make_packed_experts_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class PackedExperts(nn.Module):
        """The experts of one layer in the authors' packed bytes. Same forward signature and arithmetic as Inkling2LoopExperts; rows of w13 are interleaved (gate0, up0, gate1, ...)."""

        def __init__(self, tensors: dict, gate_cap: float, up_cap: float, dtype=None, chunk: int = EXPERTS_PER_CHUNK):
            super().__init__()
            self.gate_cap, self.up_cap, self.chunk = gate_cap, up_cap, chunk
            self.dtype = dtype or torch.bfloat16
            self.packed = "w13_weight_amax" in tensors
            for k, v in tensors.items():                       # buffers: they move with the module and are never trained
                self.register_buffer(k, v, persistent=False)
            self.num_experts = tensors["w13_weight"].shape[0]

        def _decode(self, which: str, ids):
            w, s = getattr(self, f"{which}_weight")[ids], getattr(self, f"{which}_weight_scale")[ids]
            if self.packed:
                return dequant_nvfp4(w, s, getattr(self, f"{which}_weight_amax")[ids], getattr(self, f"{which}_weight_outer_scale_denominator"), self.dtype)
            return dequant_mxfp8(w, s, self.dtype)

        def _act(self, gu):
            return softcapped_swiglu(gu[..., 0::2], gu[..., 1::2], self.gate_cap, self.up_cap)

        def expert_forward(self, e: int, x):
            ids = torch.tensor([e], device=x.device)
            return F.linear(self._act(F.linear(x.to(self.dtype), self._decode("w13", ids)[0])), self._decode("w2", ids)[0])

        def forward(self, hidden_states, top_k_index, top_k_weights):
            out = torch.zeros_like(hidden_states, dtype=torch.float32)
            hit = torch.unique(top_k_index[top_k_index < self.num_experts]).tolist()
            for s in range(0, len(hit), self.chunk):
                group = hit[s:s + self.chunk]
                ids = torch.tensor(group, device=hidden_states.device)
                w13, w2 = self._decode("w13", ids), self._decode("w2", ids)
                for j, e in enumerate(group):
                    slot_tok, slot_k = torch.where(top_k_index == e)
                    y = F.linear(self._act(F.linear(hidden_states[slot_tok].to(self.dtype), w13[j])), w2[j]) * top_k_weights[slot_tok, slot_k, None]
                    out.index_add_(0, slot_tok, y.to(out.dtype))
            return out.to(hidden_states.dtype)

    return PackedExperts


# --------------------------------------------------------------------------------------------------------------------- names

_RULES = [          # (regex on the checkpoint name after "model.llm.", replacement), applied in order; a copy of transformers' inkling2_text + inkling2_loop_text renames (the loop's rules first, as there)
    (r"controllers\.attn\.", "q_message_gate."), (r"controllers\.k\.", "k_message_gate."), (r"controllers\.v\.", "v_message_gate."), (r"controllers\.mlp\.", "mlp_message_gate."),
    (r"_message_gate\.down_weight", "_message_gate.down_proj.weight"), (r"_message_gate\.gain_weight", "_message_gate.message_gate_proj.weight"),
    (r"_message_gate\.forget_weight", "_message_gate.forget_gate_proj.weight"),
    (r"^embed\.weight$", "embed_tokens.weight"), (r"^embed_norm\.", "embed_tokens.embed_norm."),
    (r"mlp\.gate\.bias$", "mlp.gate.e_score_correction_bias"),
    (r"\.attn\.", ".self_attn."),
    (r"self_attn\.wq_du", "self_attn.q_proj.linear"), (r"self_attn\.wk_dv", "self_attn.k_proj.linear"), (r"self_attn\.wv_dv", "self_attn.v_proj.linear"),
    (r"self_attn\.k_sconv\.", "self_attn.k_proj.sconv."), (r"self_attn\.v_sconv\.", "self_attn.v_proj.sconv."),
    (r"self_attn\.rel_logits_proj\.", "self_attn.relative_position_bias."), (r"self_attn\.wr_du", "self_attn.relative_position_bias.linear"),
    (r"self_attn\.q_l2_clip\.", "self_attn.q_proj.norm."), (r"self_attn\.rel_l2_clip\.", "self_attn.relative_position_bias.norm."),
    (r"self_attn\.wo_ud", "self_attn.o_proj.linear"), (r"self_attn\.gate_attn", "self_attn.o_proj.gate_proj"), (r"self_attn\.attn_sink_proj", "self_attn.sink_proj"),
    (r"attn_sconv\.", "post_attention_convolution."), (r"mlp_sconv\.", "post_mlp_convolution."),
    (r"(convolution|sconv)\.weight(?!_)", r"\1.conv1d.weight"), (r"(convolution|sconv)\.static_w", r"\1.static_weight"), (r"(convolution|sconv)\.weight_proj", r"\1.weight_proj.weight"),
    (r"mlp_norm", "post_attention_layernorm"), (r"attn_norm", "input_layernorm"), (r"(q|k|v)_input_norm", r"\1_input_layernorm"),
    (r"attn_res\.mlp_queries\.(\d+)", r"layers.\1.mlp_input.query"), (r"attn_res\.output_query", "final_input.query"),
]


def hf_name(name: str):
    """The parameter name in transformers' text model for a checkpoint tensor, or None when it is not loaded here (experts, vision, the drafter, the two loop query tables that
    `merge_message_queries` handles). `model.llm.unembed.weight` -> `lm_head.weight` (the caller drops the padded rows)."""
    layer, role, _ = moe_layout.classify(name)
    if role in ("vision", "mtp", "experts", "expert_companion", "expert_shared", "loop_query"):
        return None
    if name == "model.llm.unembed.weight":
        return "lm_head.weight"
    if not name.startswith("model.llm."):
        return None
    n = name[len("model.llm."):]
    m = re.match(r"attn_res\.attn_queries\.(\d+)$", n)
    if m:
        return f"model.layers.{int(m.group(1)) + 1}.attention_input.query"
    for pat, rep in _RULES:
        n = re.sub(pat, rep, n)
    return "model." + n


def merge_message_queries(queries, kv_queries):
    """middle_cycle_cross_pass_queries + _kv_queries (each [2 x loop layers, H]) -> message_input.query [loop layers, 4, H] in the order query, key, value, mlp"""
    import torch
    q = queries.unflatten(0, (-1, 2))
    kv = kv_queries.unflatten(0, (-1, 2))
    return torch.cat([q[:, :1], kv, q[:, 1:]], dim=1)


# --------------------------------------------------------------------------------------------------------------------- building the model

def text_layers(model):
    """the decoder layers of a loaded Inkling model, whether it is the text model or the full conditional-generation one"""
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise InklingLoadError("The layers of the loaded model could not be found.")


def load_hf(folder, mode: str = "auto", dtype=None):
    """The library's own loading (needs about 48 GB: with device_map="auto" the rest goes to the computer's memory). Returns the full model; use `text_layers` for the layers."""
    require_transformers()
    import torch
    from transformers import AutoModelForImageTextToText
    kw: dict = {"dtype": dtype or torch.bfloat16}
    if torch.cuda.is_available():
        kw["device_map"] = "auto"
        free = torch.cuda.mem_get_info()[0] / 2**30
        kw["max_memory"] = {0: f"{max(1, int(free - 4))}GiB", "cpu": "200GiB"}
    model = AutoModelForImageTextToText.from_pretrained(str(folder), **kw)
    model.eval()
    return model


def load_packed(folder, device=None, dtype=None, on_tensor=None):
    """The text model with its experts kept packed (see the top of this file). Returns the model on `device`. `on_tensor(done, total)` reports progress."""
    require_transformers()
    import json
    import sys
    from pathlib import Path

    import torch
    import torch.nn as nn
    from safetensors import safe_open
    from transformers.models.inkling2_loop import configuration_inkling2_loop as cmod
    from transformers.models.inkling2_loop import modeling_inkling2_loop as mmod

    from . import safetensors_io
    folder = Path(folder)
    dtype = dtype or torch.bfloat16
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    tc_dict = {k: v for k, v in moe_layout.text_cfg(cfg).items() if k != "model_type"}
    tc = cmod.Inkling2LoopTextConfig(**tc_dict)

    class _NoExperts(nn.Module):                            # the real experts are 48 GB: they are never built here
        def __init__(self, config):
            super().__init__()
    real = mmod.Inkling2LoopExperts
    mmod.Inkling2LoopExperts = _NoExperts
    old_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = mmod.Inkling2LoopForCausalLM(tc)
    finally:
        torch.set_default_dtype(old_default)
        mmod.Inkling2LoopExperts = real
    refs = safetensors_io.list_tensors(folder)
    n_layers = int(tc_dict["num_hidden_layers"])
    sd: dict = {}
    experts: dict = {i: {} for i in range(n_layers)}
    pending_q: dict = {}
    handles: dict = {}
    total, done = len(refs), 0
    try:
        for name, ref in refs.items():
            done += 1
            layer, role, _ = moe_layout.classify(name)
            if role in ("vision", "mtp"):
                continue
            f = handles.get(ref.file) or handles.setdefault(ref.file, safe_open(ref.file, framework="pt"))
            if role in ("experts", "expert_companion", "expert_shared"):
                experts[layer][name.split(".mlp.experts.")[1]] = f.get_tensor(name).to(device)
            elif role == "loop_query":
                pending_q[name.rsplit(".", 1)[-1]] = f.get_tensor(name)
            else:
                target = hf_name(name)
                if target is None:
                    continue
                t = f.get_tensor(name)
                if target == "lm_head.weight":
                    t = t[: int(tc_dict["unpadded_vocab_size"])]
                sd[target] = t
            if on_tensor and done % 50 == 0:
                on_tensor(done, total)
    finally:
        handles.clear()
    if pending_q:
        sd["model.message_input.query"] = merge_message_queries(pending_q["middle_cycle_cross_pass_queries"], pending_q["middle_cycle_cross_pass_kv_queries"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if ".mlp.experts." not in k]
    if unexpected or missing:
        raise InklingLoadError("The weights do not fit the model code of the installed transformers (the names changed?): "
                               f"missing {missing[:4]}, unexpected {list(unexpected)[:4]}. Update the program, or load with the library's own path.")
    cls = make_packed_experts_class()
    for i, layer in enumerate(text_layers(model)):
        layer.mlp.experts = cls(experts[i], float(tc_dict["mlp_gate_softcap"]), float(tc_dict["mlp_up_softcap"]), dtype)
        experts[i] = None
    model.to(device)
    model.eval()
    model.config.use_cache = False
    return model
