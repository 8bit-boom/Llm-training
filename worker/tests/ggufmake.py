"""A tiny GGUF writer for tests, independent of the worker's code (stdlib struct + numpy for the numbers): a key-value section and tensors of F32 / F16 / BF16 (or raw bytes
for any other type number)."""
import struct

import numpy as np

F32, F16, BF16, Q4_K = 0, 1, 30, 12
ELEM = {F32: 4, F16: 2, BF16: 2}


def kv_bytes(kvs) -> bytes:
    """kvs: list of (key, kind, value); kind "u32", "str" or "strs" (array of strings)."""
    out = b""
    for key, kind, v in kvs:
        kb = key.encode()
        out += struct.pack("<Q", len(kb)) + kb
        if kind == "u32":
            out += struct.pack("<II", 4, v)
        elif kind == "str":
            vb = v.encode()
            out += struct.pack("<IQ", 8, len(vb)) + vb
        elif kind == "strs":
            out += struct.pack("<IIQ", 9, 8, len(v)) + b"".join(struct.pack("<Q", len(s.encode())) + s.encode() for s in v)
        else:
            raise ValueError(kind)
    return out


def to_bytes(a, type_id) -> bytes:
    a = np.ascontiguousarray(a, dtype=np.float32)
    if type_id == F32:
        return a.astype("<f4").tobytes()
    if type_id == F16:
        return a.astype("<f2").tobytes()
    if type_id == BF16:      # truncate the low half: the test values are chosen so this is the value itself
        return (a.view(np.uint32) >> 16).astype("<u2").tobytes()
    raise ValueError(type_id)


def write(path, tensors, kvs=(), align=32):
    """tensors: list of (name, dims (ggml order, cols first), type_id, data) where data is an array shaped like dims reversed (numpy order) or raw bytes."""
    kv = kv_bytes(list(kvs))
    blobs, offs, off = [], [], 0
    for _n, _d, ty, data in tensors:
        b = data if isinstance(data, (bytes, bytearray)) else to_bytes(data, ty)
        blobs.append(b)
        offs.append(off)
        off = (off + len(b) + align - 1) // align * align
    infos = b""
    for (name, dims, ty, _), o in zip(tensors, offs):
        nb = name.encode()
        infos += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ty, o)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(list(kvs))) + kv + infos
    head += b"\0" * ((-len(head)) % align)
    with open(path, "wb") as fh:
        fh.write(head)
        for b, o in zip(blobs, offs):
            fh.seek(len(head) + o)
            fh.write(b)
        fh.truncate(len(head) + off)
    return path


def bf16_values(rng, shape, scale=0.02):
    """float32 values that are exactly representable in bfloat16"""
    a = (rng.laplace(size=shape) * scale).astype(np.float32)
    return (a.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def tiny_model(path, rng=None, rows=300, cols=512, n_layer=2, experts=4, kind=BF16, tied=True, odd_cols=True):
    """A qwen35moe-like 16-bit GGUF with 2-D weights, 3-D expert weights, norms, an embedding (and a cols % 256 != 0 tensor when asked)."""
    rng = rng or np.random.default_rng(0)
    ts = []

    def add(name, dims, ty=kind, scale=0.02):
        ts.append((name, dims, ty, bf16_values(rng, tuple(reversed(dims)), scale) if ty == BF16 else (rng.laplace(size=tuple(reversed(dims))) * scale).astype(np.float32)))
    add("token_embd.weight", [cols, 400])
    for i in range(n_layer):
        add(f"blk.{i}.attn_norm.weight", [cols], F32, 1.0)
        add(f"blk.{i}.attn_q.weight", [cols, rows])
        add(f"blk.{i}.attn_k.weight", [cols, 64])
        add(f"blk.{i}.ffn_up_exps.weight", [cols, 96, experts])
        add(f"blk.{i}.ffn_gate_inp.weight", [cols, experts], F32, 0.1)
        if odd_cols and i == 0:
            add(f"blk.{i}.ssm_out.weight", [320, 64])
    if not tied:
        add("output.weight", [cols, 400])
    return write(path, ts, [("general.architecture", "str", "qwen35moe"), ("general.alignment", "u32", 32), ("tokenizer.ggml.tokens", "strs", ["a", "b", "c"])])
