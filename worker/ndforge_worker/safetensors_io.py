"""Reading and writing .safetensors files with the standard library only (no numpy, no torch).

Used by the "Slim or grow a model" job: layers are renamed and copied as raw bytes, so a 56 GB model can be reshaped without loading it.

File layout: 8 bytes little-endian header length N, N bytes of JSON (name -> {dtype, shape, data_offsets}), then the data area.
`data_offsets` are relative to the start of the data area; `TensorRef.start/end` are absolute file offsets.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}

INDEX_NAME = "model.safetensors.index.json"


class Cancelled(Exception):
    """The caller asked to stop while shards were being written."""


@dataclass(frozen=True)
class TensorRef:
    file: str          # absolute path of the shard
    dtype: str
    shape: tuple
    start: int         # absolute byte offset in the file (8 + header_len + data_offsets[0])
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


def _dtype_bytes(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise ValueError(f"unsupported tensor type {dtype!r}") from None


def read_header(path) -> tuple[dict, int]:
    """(header without "__metadata__", absolute offset of the data area). Raises ValueError("not a safetensors file: ...") on a bad length/JSON."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"not a safetensors file: {path.name} (too short)")
        (n,) = struct.unpack("<Q", raw)
        if n > size - 8:
            raise ValueError(f"not a safetensors file: {path.name} (header length {n} is larger than the file)")
        try:
            header = json.loads(fh.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"not a safetensors file: {path.name} (header is not JSON: {exc})") from None
    if not isinstance(header, dict):
        raise ValueError(f"not a safetensors file: {path.name} (header is not an object)")
    header.pop("__metadata__", None)
    base = 8 + n
    for name, info in header.items():
        try:
            a, b = info["data_offsets"]
            ok = isinstance(info["dtype"], str) and isinstance(info["shape"], list) and 0 <= a <= b and base + b <= size
        except (KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"not a safetensors file: {path.name} (bad entry for tensor {name!r})")
    return header, base


def _file_tensors(path: Path) -> dict[str, TensorRef]:
    header, base = read_header(path)
    items = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    return {name: TensorRef(str(path.resolve()), info["dtype"], tuple(info["shape"]), base + info["data_offsets"][0], base + info["data_offsets"][1])
            for name, info in items}


def list_tensors(folder) -> dict[str, TensorRef]:
    """All tensors of a model folder: model.safetensors.index.json (weight_map) when present, else every *.safetensors file. Duplicate names -> ValueError."""
    folder = Path(folder)
    index = folder / INDEX_NAME
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{INDEX_NAME} is not readable: {exc}") from None
        files = sorted(set(weight_map.values()))
        missing = [f for f in files if not (folder / f).is_file()]
        if missing:
            raise ValueError(f"{INDEX_NAME} names files that are missing: {', '.join(missing[:3])}")
        paths = [folder / f for f in files]
    else:
        paths = sorted(folder.glob("*.safetensors"))
    if not paths:
        raise ValueError(f"no .safetensors files in {folder}")
    out: dict[str, TensorRef] = {}
    for p in paths:
        for name, ref in _file_tensors(p).items():
            if name in out:
                raise ValueError(f"tensor {name!r} appears in more than one file")
            out[name] = ref
    return out


def tensors_from_hub_metadata(meta) -> dict[str, tuple[str, tuple]]:
    """{name: (dtype, shape)} from HfApi.get_safetensors_metadata(...).files_metadata (inspect without download)."""
    out: dict[str, tuple[str, tuple]] = {}
    for fmeta in meta.files_metadata.values():
        for name, info in fmeta.tensors.items():
            if name in out:
                raise ValueError(f"tensor {name!r} appears in more than one file")
            out[name] = (str(info.dtype), tuple(info.shape))
    return out


@dataclass
class OutTensor:
    name: str
    dtype: str
    shape: tuple
    source: TensorRef | None    # None => write zeros (or `data`)
    slice: tuple | None = None  # (axis, kept indices): write only these entries of the source along `axis` (width pruning; `shape` is the new shape)
    data: bytes | None = None   # bytes held in memory (a small tensor made by the caller); source must be None

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * _dtype_bytes(self.dtype)


def _plan_shards(tensors: list[OutTensor], shard_bytes: int) -> list[list[OutTensor]]:
    shards: list[list[OutTensor]] = []
    cur: list[OutTensor] = []
    used = 0
    for t in tensors:
        if cur and used + t.nbytes > shard_bytes:
            shards.append(cur)
            cur, used = [], 0
        cur.append(t)
        used += t.nbytes
    if cur:
        shards.append(cur)
    return shards


def _header_bytes(shard: list[OutTensor]) -> bytes:
    header: dict = {"__metadata__": {"format": "pt"}}
    pos = 0
    for t in shard:
        header[t.name] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": [pos, pos + t.nbytes]}
        pos += t.nbytes
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return raw + b" " * (-len(raw) % 8)


def write_shards(tensors: list[OutTensor], out_dir, shard_bytes: int = 5 * 2**30, chunk: int = 64 * 2**20,
                 on_progress=None, should_stop=None) -> dict:
    """Writes model-00001-of-0000N.safetensors (+ model.safetensors.index.json when N > 1, always with metadata.total_size)
    in the given order, copying bytes in `chunk` pieces (never holds a tensor in memory), zeros for source None.
    Header JSON is padded with spaces to a multiple of 8 bytes; "__metadata__": {"format": "pt"}.
    on_progress(done_bytes, total_bytes); should_stop() -> True raises Cancelled. Returns the index dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [t.name for t in tensors]
    if len(set(names)) != len(names):
        raise ValueError("two output tensors have the same name")
    for t in tensors:
        if t.slice is not None and t.source is None:
            raise ValueError(f"tensor {t.name!r}: a slice needs a source")
        if t.data is not None and (t.source is not None or len(t.data) != t.nbytes):
            raise ValueError(f"tensor {t.name!r}: data must have exactly {t.nbytes} bytes and no source")
        if t.source is not None and t.slice is None and t.source.nbytes != t.nbytes:
            raise ValueError(f"tensor {t.name!r}: source has {t.source.nbytes} bytes, expected {t.nbytes}")
    chunk = max(1, int(chunk))
    total = sum(t.nbytes for t in tensors)
    shards = _plan_shards(tensors, shard_bytes)
    n = len(shards)
    done = 0
    weight_map: dict[str, str] = {}
    handles: dict[str, object] = {}
    try:
        for i, shard in enumerate(shards, 1):
            fname = "model.safetensors" if n == 1 else f"model-{i:05d}-of-{n:05d}.safetensors"
            header = _header_bytes(shard)
            with open(out_dir / fname, "wb") as dst:
                dst.write(struct.pack("<Q", len(header)))
                dst.write(header)
                for t in shard:
                    left = t.nbytes
                    if t.slice is not None or t.data is not None:
                        if t.slice is not None:
                            from . import slice_io          # numpy is needed only for width pruning
                            blob = slice_io.sliced_bytes(t.source, t.slice[0], t.slice[1])
                        else:
                            blob = t.data
                        if len(blob) != t.nbytes:
                            raise ValueError(f"tensor {t.name!r}: produced {len(blob)} bytes, expected {t.nbytes}")
                        for pos in range(0, len(blob), chunk):
                            if should_stop and should_stop():
                                raise Cancelled()
                            piece = blob[pos:pos + chunk]
                            dst.write(piece)
                            done += len(piece)
                            if on_progress:
                                on_progress(done, total)
                        del blob
                    elif t.source is None:
                        while left > 0:
                            if should_stop and should_stop():
                                raise Cancelled()
                            piece = min(chunk, left)
                            dst.write(bytes(piece))
                            left -= piece
                            done += piece
                            if on_progress:
                                on_progress(done, total)
                    else:
                        src = handles.get(t.source.file)
                        if src is None:
                            src = handles[t.source.file] = open(t.source.file, "rb")
                        src.seek(t.source.start)
                        while left > 0:
                            if should_stop and should_stop():
                                raise Cancelled()
                            data = src.read(min(chunk, left))
                            if not data:
                                raise ValueError(f"{Path(t.source.file).name} ended early while copying {t.name!r}")
                            dst.write(data)
                            left -= len(data)
                            done += len(data)
                            if on_progress:
                                on_progress(done, total)
                    weight_map[t.name] = fname
    finally:
        for h in handles.values():
            h.close()
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    if n > 1:
        (out_dir / INDEX_NAME).write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index
