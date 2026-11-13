"""An independent, minimal .safetensors encoder/decoder for the tests (deliberately NOT ndforge_worker.safetensors_io, so that module is
checked against a second implementation of the format)."""
import json
import struct
from pathlib import Path

BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8}


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def fill(i, n):
    """deterministic bytes of tensor number i"""
    return bytes((i * 7 + k) % 256 for k in range(n))


def write_file(path, tensors):
    """tensors: list of (name, dtype, shape, bytes)"""
    header, pos = {"__metadata__": {"format": "pt"}}, 0
    for name, dtype, shape, data in tensors:
        assert len(data) == numel(shape) * BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [pos, pos + len(data)]}
        pos += len(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw)
        for t in tensors:
            fh.write(t[3])


def read_file(path):
    """{name: (dtype, shape, bytes)}"""
    blob = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", blob[:8])
    header = json.loads(blob[8:8 + n])
    header.pop("__metadata__", None)
    return {k: (v["dtype"], tuple(v["shape"]), blob[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]]) for k, v in header.items()}


def read_folder(folder):
    out = {}
    for p in sorted(Path(folder).glob("*.safetensors")):
        out.update(read_file(p))
    return out


def slice_reference(data, shape, width, axis, idx):
    """pure Python reference for slicing: the bytes of a tensor (shape, `width` bytes per element) keeping only the entries `idx` along `axis`"""
    elems = [data[i:i + width] for i in range(0, len(data), width)]
    strides = [numel(shape[k + 1:]) for k in range(len(shape))]
    out = []

    def walk(dim, base, picked):
        if dim == len(shape):
            out.append(elems[base])
            return
        rng = idx if dim == axis else range(shape[dim])
        for i in rng:
            walk(dim + 1, base + i * strides[dim], picked)
    walk(0, 0, None)
    return b"".join(out)
