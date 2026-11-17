"""Rows of a 16-bit GGUF, and the small "sample" GGUF made of some of them (the measured fit, docs/MIXED_QUANTS.md "Fit to a size (measured)").

The measured fit does not need every weight: it looks at a few rows of every tensor (a row is what a quantization block runs along, so one quantized row is exactly what the full
export makes of that row). `sample_plan` chooses the rows (stratified random, seeded, the same for the same file), `read_rows` reads them with plain seek and read (no memory map: on
Windows a mapped file cannot be deleted while it is open), `write_sample` writes a valid GGUF holding only those rows of every quantizable tensor (everything else is copied whole) so
the user's own llama-quantize can be run on it.

Shapes follow GGUF: `dims[0]` is the row length (the columns); a 2-D tensor is `[cols, rows]`, a mixture-of-experts tensor `[cols, rows, n_expert]`. Errors are SourceError (a
ValueError with a plain-language text)."""
from __future__ import annotations

import math
import os
import struct
import zlib
from pathlib import Path

import numpy as np

from . import fsutil, ggufinfo, mixedquant

SEED_XOR = 1234
RAW = {0: ("F32", 4), 1: ("F16", 2), 30: ("BF16", 2)}      # ggml type number -> (name, bytes per value)
CHUNK_BYTES = 64 << 20


class SourceError(ValueError):
    """The file cannot be used for the measured fit; the text says why in plain words."""


def open_source(path) -> dict:
    """ggufinfo.read of `path` plus {"path", "size"}, refusing what the measured fit cannot use: not a GGUF, a split file (`split.count` above 1: merge it with `llama-gguf-split --merge`),
    a file with quantized weights (it needs the 16-bit file), a file cut short."""
    p = Path(path)
    try:
        info = ggufinfo.read(p)
        size = p.stat().st_size
    except OSError as exc:
        raise SourceError(f"The model file {p.name} could not be opened ({exc.strerror or type(exc).__name__}).")
    except ValueError:
        raise SourceError(f"{p.name} is not a readable GGUF file.")
    split = info["meta"].get("split.count")
    if isinstance(split, int) and not isinstance(split, bool) and split > 1:
        raise SourceError(f"{p.name} is one part of a split GGUF ({split} parts). Merge it into one file first (llama-gguf-split --merge first-part.gguf merged.gguf), then use that.")
    for t in info["tensors"]:
        name = ggufinfo.type_name(t["type"])
        if t["type"] not in RAW and (name.startswith(("Q", "IQ", "TQ")) or mixedquant.quantizable(t["name"], t["dims"])):
            raise SourceError(f"{p.name} holds quantized weights ({t['name']} is {name}). The measured fit needs the 16-bit file (F16 or BF16): the one the export makes first, model-f16.gguf.")
        if t["type"] in RAW and t["offset"] + ggufinfo.numel(t) * RAW[t["type"]][1] + info["data_start"] > size:
            raise SourceError(f"{p.name} is cut short (the data of {t['name']} is not all there).")
    info["path"] = str(p)
    info["size"] = size
    return info


def value_bytes(t: dict) -> int:
    return RAW[t["type"]][1]


def is_sampled(t: dict) -> bool:
    """A tensor the measured fit looks at: one llama-quantize would quantize, stored in 16 or 32 bits."""
    return t["type"] in RAW and mixedquant.quantizable(t["name"], t["dims"])


def geometry(t: dict) -> tuple:
    """(cols, rows per expert, experts): a 3-D tensor is `[cols, rows, experts]`; anything else is flattened to `[cols, rows]` with one expert."""
    d = t["dims"]
    cols = d[0]
    if len(d) == 3:
        return cols, d[1], d[2]
    rows = 1
    for x in d[1:]:
        rows *= x
    return cols, rows, 1


def _strata(n_rows: int, n: int, rng) -> list:
    if n >= n_rows:
        return list(range(n_rows))
    edges = [i * n_rows // n for i in range(n + 1)]
    return [int(rng.integers(edges[i], edges[i + 1])) for i in range(n)]


def sample_plan(info: dict, n: int = 128, seed: int = 0) -> dict:
    """{tensor name: pick} for every sampled tensor. A pick is {"name", "cols", "rows_per_expert", "experts", "rows": [row index within one expert, ascending], "exact": every row is in it}.
    Stratified: the rows are cut into `n` equal strata and one row is drawn from each (numpy default_rng seeded with crc32(name) ^ 1234 ^ seed), so the same file and settings always give the
    same rows. A tensor with `n` rows or fewer is taken whole (exact). A 3-D tensor takes k = ceil(n / experts) rows (at least 1), the same rows in every expert."""
    n = max(1, int(n))
    picks = {}
    for t in info["tensors"]:
        if not is_sampled(t):
            continue
        cols, rows, experts = geometry(t)
        want = n if experts == 1 else max(1, math.ceil(n / experts))
        rng = np.random.default_rng((zlib.crc32(t["name"].encode("utf-8")) ^ SEED_XOR ^ int(seed)) & 0xFFFFFFFF)
        chosen = _strata(rows, want, rng)
        picks[t["name"]] = {"name": t["name"], "cols": cols, "rows_per_expert": rows, "experts": experts, "rows": chosen, "exact": len(chosen) == rows}
    return picks


def flat_rows(pick: dict) -> list:
    """The row numbers of a pick inside the source tensor (expert-major), ascending."""
    r, k = pick["rows_per_expert"], pick["rows"]
    return [e * r + i for e in range(pick["experts"]) for i in k]


def sample_dims(pick: dict) -> list:
    k = len(pick["rows"])
    return [pick["cols"], k] if pick["experts"] == 1 else [pick["cols"], k, pick["experts"]]


def _read_raw(fh, base: int, row_len: int, rows) -> bytes:
    """The bytes of rows (ascending row numbers) of a tensor that starts at file offset `base`: neighbouring rows are read in one go."""
    out = []
    i = 0
    rows = list(rows)
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1] == rows[j] + 1 and (j + 1 - i + 1) * row_len <= CHUNK_BYTES:
            j += 1
        count = j - i + 1
        fh.seek(base + rows[i] * row_len)
        raw = fh.read(count * row_len)
        if len(raw) != count * row_len:
            raise SourceError("The model file is cut short (a row could not be read).")
        out.append(raw)
        i = j + 1
    return b"".join(out)


def to_float32(raw: bytes, type_id: int, shape) -> np.ndarray:
    if type_id == 0:
        a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif type_id == 1:
        a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif type_id == 30:
        a = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        raise SourceError(f"The type {ggufinfo.type_name(type_id)} cannot be read as numbers here.")
    return a.reshape(shape)


def read_rows(src: dict, tensor: dict, rows) -> np.ndarray:
    """float32 `[len(rows), cols]`: the given rows (row numbers within the whole tensor, any order) of `tensor` (an entry of src["tensors"]). BF16 is widened exactly."""
    if tensor["type"] not in RAW:
        raise SourceError(f"{tensor['name']} is {ggufinfo.type_name(tensor['type'])}, not a 16-bit tensor.")
    cols, per, experts = geometry(tensor)
    total = per * experts
    rows = [int(r) for r in rows]
    if any(r < 0 or r >= total for r in rows):
        raise ValueError(f"row out of range for {tensor['name']}")
    order = sorted(range(len(rows)), key=lambda i: rows[i])
    row_len = cols * value_bytes(tensor)
    with open(src["path"], "rb") as fh:
        raw = _read_raw(fh, src["data_start"] + tensor["offset"], row_len, [rows[i] for i in order])
    a = to_float32(raw, tensor["type"], (len(rows), cols))
    out = np.empty_like(a)
    out[order] = a
    return out


def _aligned(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_sample(src: dict, out_path, picks: dict) -> dict:
    """Writes the sample GGUF: the header and the whole key-value section of `src` byte for byte, then every sampled tensor cut to its picked rows and every other tensor whole, in the
    source order. Written under a `.partial` name and renamed when complete. Returns {"path", "bytes", "tensors", "sampled_weights"}."""
    align = src["alignment"]
    tensors = src["tensors"]
    layout, off = [], 0
    for t in tensors:
        pk = picks.get(t["name"]) if is_sampled(t) else None
        dims = sample_dims(pk) if pk else list(t["dims"])
        n = 1
        for d in dims:
            n *= d
        nbytes = n * value_bytes(t) if t["type"] in RAW else None
        if nbytes is None:
            nbytes = _source_bytes(t)
        layout.append((t, pk, dims, off, nbytes))
        off = _aligned(off + nbytes, align)
    with open(src["path"], "rb") as fh:
        head = fh.read(src["kv_end"])
    version = struct.unpack("<I", head[4:8])[0]
    n_kv = struct.unpack("<Q", head[16:24])[0]
    infos = b"".join(_string(t["name"]) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<I", t["type"]) + struct.pack("<Q", o)
                     for t, _pk, dims, o, _n in layout)
    header = b"GGUF" + struct.pack("<I", version) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", n_kv) + head[24:src["kv_end"]]
    start = _aligned(len(header) + len(infos), align)
    out = fsutil.ensure_parent(out_path)
    part = fsutil.partial_of(out)
    sampled = 0
    try:
        with open(src["path"], "rb") as fin, open(part, "wb") as fo:
            fo.write(header + infos)
            fo.write(b"\0" * (start - len(header) - len(infos)))
            pos = 0
            for t, pk, dims, o, nbytes in layout:
                if o > pos:
                    fo.write(b"\0" * (o - pos))
                base = src["data_start"] + t["offset"]
                if pk is not None:
                    rl = pk["cols"] * value_bytes(t)
                    flat = flat_rows(pk)
                    step = max(1, CHUNK_BYTES // max(rl, 1))
                    for i in range(0, len(flat), step):
                        fo.write(_read_raw(fin, base, rl, flat[i:i + step]))
                    sampled += len(flat) * pk["cols"]
                else:
                    fin.seek(base)
                    left = nbytes
                    while left:
                        buf = fin.read(min(left, CHUNK_BYTES))
                        if not buf:
                            raise SourceError("The model file is cut short (a tensor could not be copied).")
                        fo.write(buf)
                        left -= len(buf)
                pos = o + nbytes
        fsutil.replace_retry(part, out)
    except BaseException:
        try:
            os.unlink(part)
        except OSError:
            pass
        raise
    return {"path": str(out), "bytes": Path(out).stat().st_size, "tensors": len(tensors), "sampled_weights": sampled}


def _source_bytes(t: dict) -> int:
    """Bytes of a tensor of a type this module does not read as numbers (copied whole): taken from the ggml block sizes."""
    name = ggufinfo.type_name(t["type"])
    bits = ggufinfo.BPW.get(name)
    if bits is None:
        raise SourceError(f"The tensor {t['name']} has a type ({name}) this program cannot copy.")
    return int(ggufinfo.numel(t) * bits / 8)
