"""Reading and slicing tensors with numpy, for width pruning ("Slim or grow a model" version 2). numpy is imported lazily: depth pruning and growing
(reshape.py, reshapejob.py) never need it.

bf16 and f16 tensors are cut as raw 16-bit integers: slicing never looks at the values, so it is bit-exact and needs no bf16 type.
"""
from __future__ import annotations

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
