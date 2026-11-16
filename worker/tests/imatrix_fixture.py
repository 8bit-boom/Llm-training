"""A scaled-down importance matrix with EXACTLY the layout of the real published file (`imatrix_unsloth.gguf` of unsloth/Qwen3.8-27B-GGUF, read on 2026-11-16; docs/IMATRIX.md,
"Verified on the real published file"): GGUF version 3, alignment 32 (no `general.alignment` key), four key-values (`general.type`, `imatrix.datasets` as an array of strings,
`imatrix.chunk_count` and `imatrix.chunk_size` as uint32), no `general.architecture`, every tensor float32 and ONE-dimensional, per weight `<name>.in_sum2` (the input width) followed
by `<name>.counts` (one value), the tensors sorted by the name of the weight as text (blk.0, blk.1, blk.10, ... blk.19, blk.2, ...), the full-attention layers at every `interval`th
layer (3, 7, 11, ...), every layer with ffn_down / ffn_gate / ffn_up. Only the widths are small. It is written with the standard library only (struct, array), independent of the
worker's own writer, so that a test of the worker's reader and writer is a real comparison. Used by every test that needs a matrix; no big files, the same on Windows and Linux."""
import re
import struct
import sys
from array import array
from pathlib import Path

LINEAR_KINDS = ("attn_gate", "attn_qkv", "ssm_alpha", "ssm_beta", "ssm_out")
FULL_KINDS = ("attn_q", "attn_k", "attn_v", "attn_output")
ALL_KINDS = ("ffn_down", "ffn_gate", "ffn_up")
# the kinds whose input is the attention output (6144 in the real file), the MLP's own width (17408), or the hidden size (5120)
ATTN_OUT_KINDS = ("attn_output", "ssm_out")
REAL = {"layers": 64, "interval": 4, "hidden": 5120, "inter": 17408, "attn_out": 6144, "tokens": 1149614.0, "chunk_count": 1251, "chunk_size": 8192}
DUMP = Path(__file__).parent / "data" / "imatrix_unsloth_qwen38_27b_header.txt"


def is_full(layer: int, interval: int = 4) -> bool:
    return layer % interval == interval - 1


def kinds_of(layer: int, interval: int = 4) -> tuple:
    return (FULL_KINDS if is_full(layer, interval) else LINEAR_KINDS) + ALL_KINDS


def width(kind: str, hidden: int, inter: int, attn_out: int) -> int:
    return inter if kind == "ffn_down" else attn_out if kind in ATTN_OUT_KINDS else hidden


def value(layer: int, kind: str, i: int) -> float:
    """The statistic of channel `i` of `blk.<layer>.<kind>`: distinct for every (layer, kind, channel), exactly representable as float32, so a test can predict any entry."""
    return float((layer + 1) * 1000 + sorted(LINEAR_KINDS + FULL_KINDS + ALL_KINDS).index(kind) * 64 + i)


def weight_names(layers: int = 8, interval: int = 4) -> list:
    """`blk.N.<kind>.weight` for every layer, sorted as text the way the real file is."""
    return sorted(f"blk.{L}.{k}.weight" for L in range(layers) for k in kinds_of(L, interval))


def _s(x: str) -> bytes:
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def gguf_bytes(kvs: list, tensors: list, alignment: int = 32) -> bytes:
    """GGUF version 3 bytes. kvs: [(key, type id, encoded value)]; tensors: [(name, [floats])] all float32 and 1-D. The data region starts at the aligned end of the tensor-info block
    and every tensor starts at an aligned offset (the real file pads every tensor, also the one-value `counts`, to 32 bytes)."""
    info, blob, off = b"", b"", 0
    for name, vals in tensors:
        raw = array("f", vals)
        if sys.byteorder == "big":
            raw.byteswap()
        data = raw.tobytes()
        info += _s(name) + struct.pack("<I", 1) + struct.pack("<Q", len(vals)) + struct.pack("<IQ", 0, off)
        pad = (-len(data)) % alignment
        blob += data + b"\0" * pad
        off += len(data) + pad
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs))
    for k, t, v in kvs:
        head += _s(k) + struct.pack("<I", t) + v
    head += info
    return head + b"\0" * ((-len(head)) % alignment) + blob


def kv_string(k: str, v: str):
    return (k, 8, _s(v))


def kv_u32(k: str, v: int):
    return (k, 4, struct.pack("<I", v))


def kv_strings(k: str, items: list):
    return (k, 9, struct.pack("<IQ", 8, len(items)) + b"".join(_s(x) for x in items))


def write_imatrix(path, layers: int = 8, interval: int = 4, hidden: int = 16, inter: int = 24, attn_out: int = 20, tokens: float = REAL["tokens"],
                  chunk_count: int = REAL["chunk_count"], chunk_size: int = REAL["chunk_size"], datasets=("unsloth_calibration_dataset",), scale: float = 1.0) -> dict:
    """Writes the matrix and returns {"names": [tensor names in file order], "values": {tensor name: [floats]}} so that a test can compare anything it wants. `scale` multiplies every
    statistic (`counts` too stays `tokens`), to make two matrices of different content."""
    tensors, values = [], {}
    for base in weight_names(layers, interval):
        m = re.match(r"blk\.(\d+)\.(.+)\.weight$", base)
        L, kind = int(m.group(1)), m.group(2)
        n = width(kind, hidden, inter, attn_out)
        sums = [value(L, kind, i) * scale for i in range(n)]
        tensors += [(base + ".in_sum2", sums), (base + ".counts", [tokens])]
        values[base + ".in_sum2"], values[base + ".counts"] = sums, [tokens]
    kvs = [kv_string("general.type", "imatrix"), kv_strings("imatrix.datasets", list(datasets)), kv_u32("imatrix.chunk_count", chunk_count), kv_u32("imatrix.chunk_size", chunk_size)]
    Path(path).write_bytes(gguf_bytes(kvs, tensors))
    return {"names": [t[0] for t in tensors], "values": values}


# ------------------------------------------------------------------------------------------------------------------ the committed evidence: the header dump of the real file

def parse_dump(path=DUMP) -> dict:
    """The real file's structure from the committed header dump: {"version", "tensors", "kv": {key: text}, "layers", "kinds": {(kind, suffix): (count, [length])}}."""
    text = Path(path).read_text(encoding="utf-8")
    head = re.search(r"GGUF version (\d+) \| tensors (\d+) \| key-values (\d+)", text)
    kv = dict(re.findall(r"^ kv (\S+) = (.*)$", text, re.M))
    layers = re.search(r"layers: (\d+) \((\d+), (\d+)\)", text)
    kinds = {}
    for m in re.finditer(r"blk\.N\.(\w+)\.weight\.(in_sum2|counts) x (\d+) \| example \S+ \[(\d+)\] type (\d+) \| shapes: \[\((\d+),\)\]", text):
        kinds[(m.group(1), m.group(2))] = (int(m.group(3)), int(m.group(4)), int(m.group(5)))
    return {"version": int(head.group(1)), "tensors": int(head.group(2)), "kv_count": int(head.group(3)), "kv": kv, "layers": int(layers.group(1)), "kinds": kinds}


def structure_of(path) -> dict:
    """{(kind, suffix): (count, length of the first, float type)} of a matrix file, read with the worker's header reader, in the shape `parse_dump` has."""
    from ndforge_worker import ggufinfo
    info = ggufinfo.read(path)
    out: dict = {}
    for t in info["tensors"]:
        m = re.match(r"blk\.\d+\.(\w+)\.weight\.(in_sum2|counts)$", t["name"])
        key = (m.group(1), m.group(2))
        count, length, typ = out.get(key, (0, ggufinfo.numel(t), t["type"]))
        out[key] = (count + 1, length, typ)
    return out
