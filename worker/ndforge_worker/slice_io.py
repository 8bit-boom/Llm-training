"""Reading and slicing tensors with numpy, for width pruning ("Slim or grow a model" version 2). numpy is imported lazily: depth pruning and growing
(reshape.py, reshapejob.py) never need it.

bf16 and f16 tensors are cut as raw 16-bit integers: slicing never looks at the values, so it is bit-exact and needs no bf16 type.
"""
from __future__ import annotations

import json
import math
import time

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
    u = n.asarray(u16, dtype="<u2").astype(n.uint32)           # the only copy: shifted in place, then looked at as float32
    u <<= 16
    return u.view(n.float32)


def f32_to_bf16(f32):
    """float32 array -> bf16 bit patterns (uint16), round to nearest even; NaN stays NaN"""
    n = np()
    f = n.ascontiguousarray(f32, dtype=n.float32)
    u = f.view(n.uint32)
    r = (u >> 16) & 1                       # one uint32 temporary; the rounding is done on it in place (no uint64)
    r += u
    r += 0x7FFF
    r >>= 16
    rounded = r.astype(n.uint16)
    del r
    nan = n.isnan(f)
    if nan.any():                           # the sum can only wrap (modulo 2**32) for NaN patterns, which are replaced here
        rounded = n.where(nan, ((u >> 16) | 0x40).astype(n.uint16), rounded)
    return rounded


def _as_f64(ref: TensorRef):
    n = np()
    a = read_array(ref)
    if ref.dtype == "BF16":
        return bf16_to_f32(a).astype(n.float64)
    return a.astype(n.float64)


def down_col_sq_norms(tensors: dict, n_layers: int):
    """float32 [n_layers, intermediate]: the squared L2 norm of every column of each decoder layer's `mlp.down_proj.weight` ([hidden, intermediate]), read from the weight
    files one tensor at a time (bf16 / f16 / f32; no torch). The squared size of what a neuron adds to the layer's output per unit of its activation."""
    from .reshape import split_key       # (imported here only to keep the module's top as it was)
    n = np()
    found: dict = {}
    for name, ref in tensors.items():
        sp = split_key(name)
        if sp is not None and sp[2] == "mlp.down_proj.weight":
            found[sp[1]] = ref
    missing = [i for i in range(n_layers) if i not in found]
    if missing:
        raise ReshapeError(f"Layer {missing[0]} has no plain MLP output projection, so its neurons cannot be ranked by their output.")
    rows = []
    for i in range(n_layers):
        w = _as_f64(found[i])
        rows.append((w ** 2).sum(axis=0).astype(n.float32))
    return n.stack(rows)


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


CHUNK_F32_BYTES = 64 * 2**20          # the merge works on this many bytes of float32 at a time


class _RawWeight:
    """A weight as it is stored (bf16 / f16 as raw uint16, f32 as float32); converted to float32 a block of rows at a time."""

    def __init__(self, raw, dtype: str):
        if dtype not in ("BF16", "F16", "F32", "F64"):
            raise ReshapeError(f"Tensors of type {dtype} cannot be merged.")
        self.raw, self.dtype, self.shape = raw, dtype, tuple(raw.shape)

    def f32(self, lo: int, hi: int):
        n = np()
        part = self.raw[lo:hi]
        if self.dtype == "BF16":
            return bf16_to_f32(part)
        if self.dtype == "F16":
            return part.view(n.float16).astype(n.float32)
        return part.astype(n.float32)


def _merged_bytes(w: "_RawWeight", a, b, scaling: float, dtype: str):
    """W + scaling * (B @ A) in the weight's own type, a block of rows at a time (about 64 MB of float32 per block, not six full-size copies): returns a 1-D uint8 memoryview."""
    n = np()
    if dtype not in ("BF16", "F16", "F32"):
        raise ReshapeError(f"Tensors of type {dtype} cannot be merged.")
    rows = w.shape[0]
    cols = int(n.prod(w.shape[1:])) if len(w.shape) > 1 else 1
    step = max(1, CHUNK_F32_BYTES // (4 * max(1, cols)))
    out = n.empty(w.shape, dtype="<f4" if dtype == "F32" else "<u2")
    for lo in range(0, rows, step):
        hi = min(rows, lo + step)
        blk = w.f32(lo, hi)
        d = b[lo:hi] @ a
        d *= scaling
        blk += d
        del d
        if dtype == "BF16":
            out[lo:hi] = f32_to_bf16(blk)
        elif dtype == "F16":
            out[lo:hi] = blk.astype("<f2").view("<u2")
        else:
            out[lo:hi] = blk
    return memoryview(out.reshape(-1).view(n.uint8))


def merge_lora(base: dict, adapter_dir, out_dir, on_progress=None, should_stop=None, shard_bytes: int = 5 * 2**30, clock: dict | None = None) -> dict:
    """Adds a plain LoRA into the base weights without loading the model: W' = W + (alpha / r) * B @ A for every adapted Linear weight (numpy, float32, one tensor
    at a time), rounded back to the tensor's own type (bf16: to nearest even). Everything else is copied as raw bytes. base: {name: TensorRef} of the base
    folder; adapter_dir: a PEFT adapter folder (adapter_model.safetensors + adapter_config.json). Writes the shards (and index) to out_dir.
    Returns {"merged": number of weights changed, "scaling", "rank"}. `clock`, when given, is filled with the seconds spent reading ("read") and computing ("compute") the changed weights."""
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
            t0 = time.monotonic()
            raw = read_array(ref)
            a, b = _to_f32(adapter[ab["lora_A"]]), _to_f32(adapter[ab["lora_B"]])
            t1 = time.monotonic()
            w = _RawWeight(raw, ref.dtype)
            if a.shape[0] != r or b.shape[1] != r or b.shape[0] != w.shape[0] or a.shape[1] != w.shape[1]:
                raise ReshapeError(MISMATCH)
            merged[0] += 1
            out = _merged_bytes(w, a, b, scaling, ref.dtype)
            if clock is not None:
                clock["read"] = clock.get("read", 0.0) + (t1 - t0)
                clock["compute"] = clock.get("compute", 0.0) + (time.monotonic() - t1)
            return out
        return make

    plan = []
    for name, ref in base.items():
        if name in targets:
            plan.append(st.OutTensor(name, ref.dtype, tuple(ref.shape), None, None, producer(name, ref, targets[name])))
        else:
            plan.append(st.OutTensor(name, ref.dtype, tuple(ref.shape), ref))
    st.write_shards(plan, out_dir, shard_bytes=shard_bytes, on_progress=on_progress, should_stop=should_stop)
    return {"merged": merged[0], "scaling": scaling, "rank": r}
