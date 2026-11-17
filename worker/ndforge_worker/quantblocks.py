"""Encoders and decoders of the ggml block formats, in numpy: what the measured fit (quantfit.py, quantmeasure.py) needs to quantize a row and read it back.

Encoders: the K-quants (Q2_K..Q6_K) come from kquants.py (a verbatim port of quant-convert-gui, see its header); Q8_0, Q4_0, Q5_0 and Q5_1 are ported from
gguf-py's `gguf/quants.py` (part of llama.cpp), which is bit-exact with ggml's reference implementation. Decoders: all types are ported from gguf-py's numpy
dequantizers (the layouts are those of llama.cpp's `ggml/src/ggml-common.h`). `np_roundf` rounds half away from zero like C's roundf.

gguf-py and llama.cpp are MIT licensed, Copyright (c) 2023-2026 The ggml authors; the notice is in THIRD_PARTY_NOTICES.md and in the header of kquants.py.
tests/test_quantblocks.py compares every function with gguf-py itself (`importorskip`).

Everything works on 2-D float32 arrays `[rows, cols]` (a row is what a block runs along) and returns bytes as `uint8 [rows, row_bytes]`."""
from __future__ import annotations

import numpy as np

from . import kquants

QK_K = 256
# type name -> (values per block, bytes per block)
BLOCK = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q4_0": (32, 18), "Q5_0": (32, 22), "Q5_1": (32, 24), "Q8_0": (32, 34),
    "Q2_K": (QK_K, 84), "Q3_K": (QK_K, 110), "Q4_K": (QK_K, 144), "Q5_K": (QK_K, 176), "Q6_K": (QK_K, 210),
}
QUANTIZED = tuple(t for t in BLOCK if t not in ("F32", "F16", "BF16"))


def row_bytes(t: str, cols: int) -> int:
    """Bytes of one row of `cols` values in type `t`. ValueError when `cols` is not a multiple of the block size."""
    if t not in BLOCK:
        raise ValueError(f"unknown type {t}")
    n, size = BLOCK[t]
    if cols % n:
        raise ValueError(f"{cols} values per row is not a multiple of {n}, the block size of {t}")
    return cols // n * size


def np_roundf(n):
    a = abs(n)
    floored = np.floor(a)
    b = floored + np.floor(2 * (a - floored))
    return np.sign(n) * b


# ------------------------------------------------------------------------------------------------------------------ legacy encoders (gguf-py)

def _q8_0(blocks):
    d = abs(blocks).max(axis=1, keepdims=True) / 127
    with np.errstate(divide="ignore"):
        id_ = np.where(d == 0, 0, 1 / d)
    qs = np_roundf(blocks * id_)
    d = d.astype(np.float16).view(np.uint8)
    qs = qs.astype(np.int8).view(np.uint8)
    return np.concatenate([d, qs], axis=1)


def _q4_0(blocks):
    n_blocks = blocks.shape[0]
    imax = abs(blocks).argmax(axis=-1, keepdims=True)
    mx = np.take_along_axis(blocks, imax, axis=-1)
    d = mx / -8
    with np.errstate(divide="ignore"):
        id_ = np.where(d == 0, 0, 1 / d)
    qs = np.trunc((blocks * id_) + np.float32(8.5), dtype=np.float32).astype(np.uint8).clip(0, 15)
    qs = qs.reshape((n_blocks, 2, 16))
    qs = qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))
    d = d.astype(np.float16).view(np.uint8)
    return np.concatenate([d, qs], axis=-1)


def _q5_0(blocks):
    n_blocks = blocks.shape[0]
    imax = abs(blocks).argmax(axis=-1, keepdims=True)
    mx = np.take_along_axis(blocks, imax, axis=-1)
    d = mx / -16
    with np.errstate(divide="ignore"):
        id_ = np.where(d == 0, 0, 1 / d)
    q = np.trunc((blocks * id_) + np.float32(16.5), dtype=np.float32).astype(np.uint8).clip(0, 31)
    qs = q.reshape((n_blocks, 2, 16))
    qs = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))
    qh = np.packbits(q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=-1, bitorder="little").reshape(n_blocks, 4)
    d = d.astype(np.float16).view(np.uint8)
    return np.concatenate([d, qh, qs], axis=-1)


def _q5_1(blocks):
    n_blocks = blocks.shape[0]
    mx = blocks.max(axis=-1, keepdims=True)
    mn = blocks.min(axis=-1, keepdims=True)
    d = (mx - mn) / 31
    with np.errstate(divide="ignore"):
        id_ = np.where(d == 0, 0, 1 / d)
    q = np.trunc((blocks - mn) * id_ + np.float32(0.5), dtype=np.float32).astype(np.uint8).clip(0, 31)
    qs = q.reshape((n_blocks, 2, 16))
    qs = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))
    qh = np.packbits(q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=-1, bitorder="little").reshape(n_blocks, 4)
    d = d.astype(np.float16).view(np.uint8)
    m = mn.astype(np.float16).view(np.uint8)
    return np.concatenate([d, m, qh, qs], axis=-1)


_LEGACY_ENCODERS = {"Q8_0": _q8_0, "Q4_0": _q4_0, "Q5_0": _q5_0, "Q5_1": _q5_1}


def _bf16_bytes(x):
    """float32 -> bfloat16 bytes, round to nearest even, quiet NaNs (ggml_compute_fp32_to_bf16)."""
    n = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    n = np.where((n & 0x7fffffff) > 0x7f800000, (n & np.uint32(0xffff0000)) | np.uint32(64 << 16), n)
    n = (n.astype(np.uint64) + (0x7fff + ((n >> 16) & 1))) >> 16
    return n.astype("<u2").view(np.uint8)


def encode(x, t: str):
    """float32 `[rows, cols]` -> bytes `uint8 [rows, row_bytes(t, cols)]`."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("encode wants a 2-D array [rows, cols]")
    rows, cols = x.shape
    nbytes = row_bytes(t, cols)
    x = np.ascontiguousarray(x)
    if t == "F32":
        return x.astype("<f4").view(np.uint8).reshape(rows, nbytes)
    if t == "F16":
        return x.astype("<f2").view(np.uint8).reshape(rows, nbytes)
    if t == "BF16":
        return _bf16_bytes(x).reshape(rows, nbytes)
    if t in kquants.SUPPORTED:
        return np.asarray(kquants.quantize(x, t), dtype=np.uint8).reshape(rows, nbytes)
    blocks = x.reshape(-1, BLOCK[t][0])
    return _LEGACY_ENCODERS[t](blocks).reshape(rows, nbytes)


# ------------------------------------------------------------------------------------------------------------------ decoders (gguf-py)

def _dq_q8_0(b):
    d, x = np.split(b, [2], axis=1)
    return x.view(np.int8).astype(np.float32) * d.view(np.float16).astype(np.float32)


def _dq_q4_0(b):
    n = b.shape[0]
    d, qs = np.hsplit(b, [2])
    d = d.view(np.float16).astype(np.float32)
    qs = qs.reshape((n, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n, -1)).astype(np.int8) - np.int8(8)
    return d * qs.astype(np.float32)


def _q5_high(qh, n):
    qh = qh.view(np.uint32).reshape((n, 1)) >> np.arange(32, dtype=np.uint32).reshape((1, 32))
    return (qh & np.uint32(0x01)).astype(np.uint8)


def _dq_q5_0(b):
    n = b.shape[0]
    d, rest = np.hsplit(b, [2])
    qh, qs = np.hsplit(rest, [4])
    d = d.view(np.float16).astype(np.float32)
    qh = _q5_high(np.ascontiguousarray(qh), n)
    ql = qs.reshape((n, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n, -1))
    q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(16)
    return d * q.astype(np.float32)


def _dq_q5_1(b):
    n = b.shape[0]
    d, rest = np.hsplit(b, [2])
    m, rest = np.hsplit(rest, [2])
    qh, qs = np.hsplit(rest, [4])
    d = d.view(np.float16).astype(np.float32)
    m = m.view(np.float16).astype(np.float32)
    qh = _q5_high(np.ascontiguousarray(qh), n)
    ql = qs.reshape((n, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n, -1))
    q = (ql | (qh << np.uint8(4))).astype(np.float32)
    return d * q + m


def _dq_q2_k(b):
    n = b.shape[0]
    scales, rest = np.hsplit(b, [QK_K // 16])
    qs, rest = np.hsplit(rest, [QK_K // 4])
    d, dmin = np.hsplit(rest, [2])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)
    dl = (d * (scales & 0xF).astype(np.float32)).reshape((n, QK_K // 16, 1))
    ml = (dmin * (scales >> 4).astype(np.float32)).reshape((n, QK_K // 16, 1))
    shift = np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n, -1, 1, 32)) >> shift) & np.uint8(3)
    qs = qs.reshape((n, QK_K // 16, 16)).astype(np.float32)
    return (dl * qs - ml).reshape((n, -1))


def _dq_q3_k(b):
    n = b.shape[0]
    hmask, rest = np.hsplit(b, [QK_K // 8])
    qs, rest = np.hsplit(rest, [QK_K // 4])
    scales, d = np.hsplit(rest, [12])
    d = d.view(np.float16).astype(np.float32)
    lscales, hscales = np.hsplit(scales, [8])
    lscales = lscales.reshape((n, 1, 8)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n, 16))
    hscales = hscales.reshape((n, 1, 4)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n, 16))
    scales = (lscales & np.uint8(0x0F)) | ((hscales & np.uint8(0x03)) << np.uint8(4))
    scales = (scales.astype(np.int8) - np.int8(32)).astype(np.float32)
    dl = (d * scales).reshape((n, 16, 1))
    ql = qs.reshape((n, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n, -1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n, 16, QK_K // 16)) & np.uint8(3)
    qh = qh.reshape((n, 16, QK_K // 16)) & np.uint8(1)
    qh = qh ^ np.uint8(1)  # the offset is zero when the bit in the mask is set
    q = (ql.astype(np.int8) - (qh << np.uint8(2)).astype(np.int8)).astype(np.float32)
    return (dl * q).reshape((n, QK_K))


def _scale_min(scales):
    n = scales.shape[0]
    scales = scales.reshape((n, 3, 4))
    d, m, m_d = np.split(scales, 3, axis=-2)
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.reshape((n, 8)), mn.reshape((n, 8))


def _dq_q4_k(b):
    n = b.shape[0]
    d, rest = np.hsplit(b, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, qs = np.hsplit(rest, [12])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)
    sc, m = _scale_min(np.ascontiguousarray(scales))
    d = (d * sc.astype(np.float32)).reshape((n, -1, 1))
    dm = (dmin * m.astype(np.float32)).reshape((n, -1, 1))
    qs = qs.reshape((n, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n, -1, 32)).astype(np.float32)
    return (d * qs - dm).reshape((n, QK_K))


def _dq_q5_k(b):
    n = b.shape[0]
    d, rest = np.hsplit(b, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, rest = np.hsplit(rest, [12])
    qh, qs = np.hsplit(rest, [QK_K // 8])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)
    sc, m = _scale_min(np.ascontiguousarray(scales))
    d = (d * sc.astype(np.float32)).reshape((n, -1, 1))
    dm = (dmin * m.astype(np.float32)).reshape((n, -1, 1))
    ql = qs.reshape((n, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n, -1, 1, 32)) >> np.arange(8, dtype=np.uint8).reshape((1, 1, 8, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n, -1, 32))
    qh = (qh & np.uint8(0x01)).reshape((n, -1, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.float32)
    return (d * q - dm).reshape((n, QK_K))


def _dq_q6_k(b):
    n = b.shape[0]
    ql, rest = np.hsplit(b, [QK_K // 2])
    qh, rest = np.hsplit(rest, [QK_K // 4])
    scales, d = np.hsplit(rest, [QK_K // 16])
    scales = np.ascontiguousarray(scales).view(np.int8).astype(np.float32)
    d = d.view(np.float16).astype(np.float32)
    d = (d * scales).reshape((n, QK_K // 16, 1))
    ql = ql.reshape((n, -1, 1, 64)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n, -1, 32))
    qh = qh.reshape((n, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = (qh & np.uint8(0x03)).reshape((n, -1, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
    q = q.reshape((n, QK_K // 16, -1)).astype(np.float32)
    return (d * q).reshape((n, QK_K))


_DECODERS = {"Q8_0": _dq_q8_0, "Q4_0": _dq_q4_0, "Q5_0": _dq_q5_0, "Q5_1": _dq_q5_1, "Q2_K": _dq_q2_k, "Q3_K": _dq_q3_k, "Q4_K": _dq_q4_k,
             "Q5_K": _dq_q5_k, "Q6_K": _dq_q6_k}


def decode(raw, t: str, rows: int, cols: int):
    """Bytes (a bytes-like object or a uint8 array) of `rows` rows of `cols` values in type `t` -> float32 `[rows, cols]`. ValueError on a wrong length."""
    nbytes = row_bytes(t, cols)
    buf = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray, memoryview)) else np.ascontiguousarray(raw).reshape(-1).view(np.uint8)
    if buf.size != rows * nbytes:
        raise ValueError(f"{t}: {buf.size} bytes is not {rows} rows of {cols} values ({rows * nbytes} bytes)")
    if t == "F32":
        return buf.view("<f4").astype(np.float32).reshape(rows, cols)
    if t == "F16":
        return buf.view("<f2").astype(np.float32).reshape(rows, cols)
    if t == "BF16":
        return (buf.view("<u2").astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)
    blocks = buf.reshape(-1, BLOCK[t][1])
    with np.errstate(all="ignore"):
        out = _DECODERS[t](blocks)
    return np.ascontiguousarray(out, dtype=np.float32).reshape(rows, cols)
