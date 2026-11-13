"""Reading and slicing tensors with numpy, for width pruning ("Slim or grow a model" version 2). numpy is imported lazily: depth pruning and growing
(reshape.py, reshapejob.py) never need it.

bf16 and f16 tensors are cut as raw 16-bit integers: slicing never looks at the values, so it is bit-exact and needs no bf16 type.
"""
from __future__ import annotations

import json
import math

from .reshape import ReshapeError
from .safetensors_io import TensorRef

# little-endian raw element types (safetensors dtype -> numpy dtype string)
RAW = {"BF16": "<u2", "F16": "<u2", "F32": "<f4", "F64": "<f8", "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "i1", "U8": "u1", "BOOL": "u1"}
NUMPY_MISSING = "The training software is incomplete (numpy). Run Training setup again."


def np():
    """numpy, or a plain ReshapeError"""
    try:
        import numpy
    except ImportError:
        raise ReshapeError(NUMPY_MISSING) from None
    return numpy


def read_array(ref: TensorRef):
    """The tensor as a numpy array of its raw element type (bf16 / f16 as uint16). Reads exactly this tensor from its file."""
    n = np()
    if ref.dtype not in RAW:
        raise ReshapeError(f"Tensors of type {ref.dtype} cannot be thinned.")
    arr = n.fromfile(ref.file, dtype=RAW[ref.dtype], count=ref.numel, offset=ref.start)
    if arr.size != ref.numel:
        raise ReshapeError(f"{ref.file} ended early while reading a tensor.")
    return arr.reshape(ref.shape)


def sliced_bytes(ref: TensorRef, axis: int, idx) -> bytes:
    """The bytes of `ref` keeping only the entries `idx` (in that order) along `axis`, in the tensor's own element type."""
    n = np()
    arr = read_array(ref)
    if not 0 <= axis < arr.ndim:
        raise ReshapeError(f"A tensor of shape {tuple(ref.shape)} has no axis {axis}.")
    index = n.asarray(list(idx), dtype=n.int64)
    if index.size and (index.min() < 0 or index.max() >= arr.shape[axis]):
        raise ReshapeError("A kept index lies outside the tensor.")
    return n.ascontiguousarray(n.take(arr, index, axis=axis)).tobytes()


def bf16_to_f32(u16):
    """bf16 bit patterns (uint16 array) -> float32 values"""
    n = np()
    return (n.asarray(u16, dtype="<u2").astype(n.uint32) << 16).view(n.float32)


def f32_to_bf16(f32):
    """float32 array -> bf16 bit patterns (uint16), round to nearest even; NaN stays NaN"""
    n = np()
    f = n.ascontiguousarray(f32, dtype=n.float32)
    u = f.view(n.uint32).astype(n.uint64)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(n.uint16)
    nan = n.isnan(f)
    if nan.any():
        rounded = n.where(nan, ((u >> 16) | 0x40).astype(n.uint16), rounded)
    return rounded


def _as_f64(ref: TensorRef):
    n = np()
    a = read_array(ref)
    if ref.dtype == "BF16":
        return bf16_to_f32(a).astype(n.float64)
    return a.astype(n.float64)


def weight_norm_scores(tensors: dict, dims: dict) -> dict:
    """Importance of the structures of the multi-token head from the weights alone (it is never run, so nothing can be measured):
    neuron i -> ||down[:, i]|| * ||up[i, :]||, query head h -> ||o_proj[:, rows(h)]||. {layer: {"mlp": [I floats], "heads": [nh floats] or None}} per
    `mtp.layers.N`, from the tensors {name: TensorRef}."""
    n = np()
    out: dict = {}
    layers = sorted({int(k.split(".")[2]) for k in tensors if k.startswith("mtp.layers.") and k.split(".")[2].isdigit()})
    hd = dims["head_dim"]
    for i in layers:
        pre = f"mtp.layers.{i}."
        down, up = tensors.get(pre + "mlp.down_proj.weight"), tensors.get(pre + "mlp.up_proj.weight")
        if down is None or up is None:
            raise ReshapeError("The multi-token head of this model has no plain MLP; it cannot be thinned.")
        d, u = _as_f64(down), _as_f64(up)
        entry = {"mlp": [float(x) for x in n.sqrt((d ** 2).sum(axis=0)) * n.sqrt((u ** 2).sum(axis=1))], "heads": None}
        o = tensors.get(pre + "self_attn.o_proj.weight")
        if o is not None:
            w = _as_f64(o)
            entry["heads"] = [float(n.sqrt((w[:, h * hd:(h + 1) * hd] ** 2).sum())) for h in range(w.shape[1] // hd)]
        out[i] = entry
    return out


# ------------------------------------------------------------------------------------------------------------- the LoRA merge

LORA_KEY = "base_model.model."
MISMATCH = "The healing result does not fit the slim model (different layers). Nothing was changed; please make a problem report."


def _adapter_pairs(adapter: dict) -> dict:
    """{module path: (A name, B name)} from the adapter's tensor names (base_model.model.<module>.lora_A.weight)"""
    pairs: dict = {}
    for name in adapter:
        if not name.startswith(LORA_KEY):
            continue
        for tag in ("lora_A", "lora_B"):
            marker = f".{tag}."
            if marker in name:
                module = name[len(LORA_KEY):name.index(marker)]
                pairs.setdefault(module, {})[tag] = name
    bad = [m for m, v in pairs.items() if set(v) != {"lora_A", "lora_B"}]
    if bad:
        raise ReshapeError(MISMATCH)
    return pairs


def _base_name(module: str, names: set) -> str:
    """the base tensor of an adapter module: the same path + ".weight", or with the language model's prefix of a combined (vision + text) checkpoint"""
    cands = [module + ".weight"]
    if module.startswith("model."):
        cands.append("model.language_model." + module[len("model."):] + ".weight")
        cands.append("language_model.model." + module[len("model."):] + ".weight")
    for c in cands:
        if c in names:
            return c
    raise ReshapeError(MISMATCH)


def _to_f32(ref: TensorRef):
    n = np()
    a = read_array(ref)
    if ref.dtype == "BF16":
        return bf16_to_f32(a)
    if ref.dtype == "F16":
        return a.view(n.float16).astype(n.float32)
    if ref.dtype in ("F32", "F64"):
        return a.astype(n.float32)
    raise ReshapeError(f"Tensors of type {ref.dtype} cannot be merged.")


def _from_f32(x, dtype: str) -> bytes:
    n = np()
    x = n.ascontiguousarray(x, dtype=n.float32)
    if dtype == "BF16":
        return f32_to_bf16(x).astype("<u2").tobytes()
    if dtype == "F16":
        return x.astype("<f2").tobytes()
    if dtype == "F32":
        return x.astype("<f4").tobytes()
    raise ReshapeError(f"Tensors of type {dtype} cannot be merged.")


def merge_lora(base: dict, adapter_dir, out_dir, on_progress=None, should_stop=None, shard_bytes: int = 5 * 2**30) -> dict:
    """Adds a plain LoRA into the base weights without loading the model: W' = W + (alpha / r) * B @ A for every adapted Linear weight (numpy, float32, one tensor
    at a time), rounded back to the tensor's own type (bf16: to nearest even). Everything else is copied as raw bytes. base: {name: TensorRef} of the base
    folder; adapter_dir: a PEFT adapter folder (adapter_model.safetensors + adapter_config.json). Writes the shards (and index) to out_dir.
    Returns {"merged": number of weights changed, "scaling", "rank"}."""
    from pathlib import Path

    from . import safetensors_io as st
    n = np()
    adapter_dir = Path(adapter_dir)
    try:
        conf = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ReshapeError("The healing result has no adapter_config.json.") from None
    if conf.get("use_dora") or conf.get("rank_pattern") or conf.get("alpha_pattern") or conf.get("modules_to_save") or conf.get("peft_type", "LORA") != "LORA":
        raise ReshapeError("Only a plain LoRA can be merged (no DoRA, no per-layer sizes, no extra saved modules).")
    r, alpha = int(conf["r"]), float(conf["lora_alpha"])
    scaling = alpha / math.sqrt(r) if conf.get("use_rslora") else alpha / r
    adapter = st.list_tensors(adapter_dir)
    pairs = _adapter_pairs(adapter)
    names = set(base)
    targets = {_base_name(m, names): v for m, v in pairs.items()}
    merged = [0]

    def producer(name, ref, ab):
        def make():
            if should_stop and should_stop():
                raise st.Cancelled()
            w = _to_f32(ref)
            a, b = _to_f32(adapter[ab["lora_A"]]), _to_f32(adapter[ab["lora_B"]])
            if a.shape[0] != r or b.shape[1] != r or b.shape[0] != w.shape[0] or a.shape[1] != w.shape[1]:
                raise ReshapeError(MISMATCH)
            merged[0] += 1
            return _from_f32(w + scaling * (b @ a), ref.dtype)
        return make

    plan = []
    for name, ref in base.items():
        if name in targets:
            plan.append(st.OutTensor(name, ref.dtype, tuple(ref.shape), None, None, producer(name, ref, targets[name])))
        else:
            plan.append(st.OutTensor(name, ref.dtype, tuple(ref.shape), ref))
    st.write_shards(plan, out_dir, shard_bytes=shard_bytes, on_progress=on_progress, should_stop=should_stop)
    return {"merged": merged[0], "scaling": scaling, "rank": r}
