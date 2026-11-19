"""The teacher cache's on-disk format and its writers, numpy only (no torch, no other module of this package).

    Both the worker (`distilljob.run_cache`, HF teacher) and the standalone script `vllm_cache.py` that runs INSIDE the vLLM container write the cache with these classes, so the two
    environments make byte-identical parts and indexes. This file must stay importable by itself (`python cachewrite.py`-style: it imports only the standard library and, lazily, numpy).

    part-00001.npz ... : ids uint32 [n, k], logp float16 [n, k], resid float32 [n], row_offsets int64 [rows + 1]; index.json (format, meta, done_rows, tokens, parts, complete, updated).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

CACHE_FORMAT = 1


def _numpy():
    try:
        import numpy
    except ImportError:
        raise ImportError("numpy is needed to write the teacher's saved answers.") from None
    return numpy


def read_cache_index(folder):
    try:
        idx = json.loads((Path(folder) / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return idx if isinstance(idx, dict) and idx.get("format") == CACHE_FORMAT else None


def cache_matches(index: dict, expect: dict) -> bool:
    """True when the saved teacher answers belong to this data, tokenizer, length, k and teacher"""
    if index.get("teacher_weights") and expect.get("teacher_weights") and index["teacher_weights"] != expect["teacher_weights"]:
        return False                                    # an older cache without the field is accepted as before
    return all(index.get(k) == expect.get(k) for k in ("teacher", "k", "rows", "data_hash", "tokenizer_hash", "max_len", "general_key", "general_rows"))


class CacheWriter:
    """Teacher answers on disk: part-00001.npz ... (ids uint32 [n, k], logp float16 [n, k], resid float32 [n], row_offsets int64 [rows + 1]) and index.json.
    Rows are added in order; a part is written whenever it holds `part_tokens` tokens, and index.json (with done_rows) after every part, so a stopped job
    can continue where it was. Needs numpy."""

    def __init__(self, folder, meta: dict, part_tokens: int = 262144):
        self.np = _numpy()
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {"format": CACHE_FORMAT, **meta}
        self.part_tokens = part_tokens
        old = read_cache_index(self.dir)
        self.parts: list = list(old["parts"]) if old and cache_matches(old, meta) else []
        if not self.parts:
            for f in self.dir.glob("part-*.npz"):
                f.unlink()
        self.done_rows = sum(p["rows"] for p in self.parts)
        self.tokens = sum(p["tokens"] for p in self.parts)
        self._buf: list = []
        self._buf_tokens = 0

    def add_row(self, ids, logp, resid) -> None:
        np = self.np
        self._buf.append((np.asarray(ids, dtype=np.uint32), np.asarray(logp, dtype=np.float16), np.asarray(resid, dtype=np.float32)))
        self._buf_tokens += len(self._buf[-1][2])
        if self._buf_tokens >= self.part_tokens:
            self.flush()

    def flush(self) -> None:
        np = self.np
        if not self._buf:
            return
        n = len(self.parts) + 1
        name = f"part-{n:05d}.npz"
        offsets = np.cumsum([0] + [len(b[2]) for b in self._buf]).astype(np.int64)
        tmp = self.dir / f"part-{n:05d}.partial.npz"
        np.savez(tmp, ids=np.concatenate([b[0] for b in self._buf]), logp=np.concatenate([b[1] for b in self._buf]),
                 resid=np.concatenate([b[2] for b in self._buf]), row_offsets=offsets)
        os.replace(tmp, self.dir / name)
        self.parts.append({"file": name, "first_row": self.done_rows, "rows": len(self._buf), "tokens": int(offsets[-1])})
        self.done_rows += len(self._buf)
        self.tokens += int(offsets[-1])
        self._buf, self._buf_tokens = [], 0
        self._write_index()

    def _write_index(self) -> None:
        idx = {**self.meta, "done_rows": self.done_rows, "tokens": self.tokens, "parts": self.parts, "complete": self.done_rows >= self.meta["rows"],
               "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        tmp = self.dir / "index.partial.json"
        tmp.write_text(json.dumps(idx), encoding="utf-8")
        os.replace(tmp, self.dir / "index.json")

    def close(self) -> None:
        self.flush()
        if not self.parts:
            self._write_index()


# ------------------------------------------------------------------------------------------------------------ the shared cache
# The teacher's answers for a row only depend on the row, the teacher, k and the length, not on which model is being healed or how many tokens the run wants. The parts of
# a cache can therefore live in one shared folder (`shared_dir/<id>/`, job keys `shared_cache` + `shared_dir`) and be used by several heals: a bigger run only adds the rows
# its smaller sibling did not have. The per-model folder (`out`) keeps its own index.json (a "view": the parts of the store this model's rows need, `store` = where they
# are), rows.npz and nothing heavy, so everything that reads a cache (the student, the program's skip decision) works as before.

class SharedCacheWriter(CacheWriter):
    """A CacheWriter whose parts live in `store` (see above) and that keeps the model's own index.json (the view) in `view`. The caller has chosen `store` with `pick_store`, so
    whatever it already holds is a start of these rows: its parts are kept and the writer goes on after them."""

    def __init__(self, store, view, meta: dict, hashes: list, part_tokens: int = 262144):
        self.np = _numpy()
        self.dir = Path(store)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.view = Path(view)
        self.view.mkdir(parents=True, exist_ok=True)
        self.meta = {"format": CACHE_FORMAT, **meta}
        self.part_tokens = part_tokens
        self.hashes = hashes
        old = read_cache_index(self.dir)
        self.parts = list(old["parts"]) if old else []
        if not self.parts:
            for f in self.dir.glob("part-*.npz"):
                f.unlink()
        self.done_rows = sum(p["rows"] for p in self.parts)
        self.tokens = sum(p["tokens"] for p in self.parts)
        self._buf: list = []
        self._buf_tokens = 0
        for f in self.view.glob("part-*.npz"):          # parts of an older, unshared cache of this model
            f.unlink()
        self._write_view()

    def _write_hashes(self) -> None:
        np = self.np
        n = min(self.done_rows, len(self.hashes))
        arr = np.frombuffer(b"".join(self.hashes[:n]), dtype=np.uint8).reshape(-1, 20) if n else np.zeros((0, 20), dtype=np.uint8)
        tmp = self.dir / "rowhashes.partial.npy"
        np.save(tmp, arr)
        os.replace(tmp, self.dir / "rowhashes.npy")

    def _store_ref(self) -> str:
        try:
            return os.path.relpath(self.dir, self.view).replace("\\", "/")
        except ValueError:                              # another drive on Windows: the full path
            return str(self.dir)

    def _write_view(self) -> None:
        need = [p for p in self.parts if p["first_row"] < self.meta["rows"]]
        idx = {**self.meta, "done_rows": self.done_rows, "tokens": sum(p["tokens"] for p in need), "parts": need, "complete": self.done_rows >= self.meta["rows"],
               "store": self._store_ref(), "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        tmp = self.view / "index.partial.json"
        tmp.write_text(json.dumps(idx), encoding="utf-8")
        os.replace(tmp, self.view / "index.json")

    def _write_index(self) -> None:
        self._write_hashes()
        super()._write_index()
        self._write_view()

    def close(self) -> None:
        self.flush()
        if not self.parts:
            self._write_index()
        else:
            self._write_view()


def load_row_ids(folder, name: str = "rows"):
    """(ids int32 [total], offsets int64 [rows + 1]) of `folder/<name>.npz`, the token ids of the rows a teacher answers (see `distilljob.save_rows`)."""
    np = _numpy()
    with np.load(Path(folder) / f"{name}.npz") as z:
        return z["ids"], z["offsets"]
