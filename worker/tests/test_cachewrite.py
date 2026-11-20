"""The numpy-only cache writer (cachewrite.py) against a frozen copy of the writer as it was in distilljob.py before the extraction: same bytes on disk."""
import json
import os
import time
import zipfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from ndforge_worker import cachewrite, distilljob  # noqa: E402
from ndforge_worker import slice_io  # noqa: E402,F401

CACHE_FORMAT = 1


# ---- frozen reference: the code of distilljob.py at d3d8e35 (only renamed)
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


class RefCacheWriter:
    """Teacher answers on disk: part-00001.npz ... (ids uint32 [n, k], logp float16 [n, k], resid float32 [n], row_offsets int64 [rows + 1]) and index.json.
    Rows are added in order; a part is written whenever it holds `part_tokens` tokens, and index.json (with done_rows) after every part, so a stopped job
    can continue where it was. Needs numpy."""

    def __init__(self, folder, meta: dict, part_tokens: int = 262144):
        self.np = np
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

class RefSharedWriter(RefCacheWriter):
    """A CacheWriter whose parts live in `store` (see above) and that keeps the model's own index.json (the view) in `view`. The caller has chosen `store` with `pick_store`, so
    whatever it already holds is a start of these rows: its parts are kept and the writer goes on after them."""

    def __init__(self, store, view, meta: dict, hashes: list, part_tokens: int = 262144):
        self.np = np
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




def synth_rows(n=7, k=4, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        m = int(rng.integers(3, 40))
        ids = rng.integers(0, 50000, size=(m, k)).astype(np.int64)
        logp = -np.sort(rng.random((m, k)).astype(np.float32) * 5, axis=1)
        resid = rng.random(m).astype(np.float32) * 0.2
        out.append((ids, logp, resid))
    return out


META = {"teacher": "t/x", "k": 4, "rows": 7, "data_hash": "h", "tokenizer_hash": "tk", "max_len": 64, "general_key": {}, "general_rows": 0}


def members(path):
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n) for n in sorted(z.namelist())}


def index_of(folder):
    d = json.loads((Path(folder) / "index.json").read_text())
    d.pop("updated")
    return d


def test_same_module_is_used_by_distilljob():
    assert distilljob.CacheWriter is cachewrite.CacheWriter
    assert distilljob.SharedCacheWriter is cachewrite.SharedCacheWriter
    assert distilljob.CACHE_FORMAT == cachewrite.CACHE_FORMAT == 1
    assert distilljob.read_cache_index is cachewrite.read_cache_index and distilljob.cache_matches is cachewrite.cache_matches


@pytest.mark.parametrize("part_tokens", [1, 50, 120, 10 ** 6])
def test_plain_writer_is_byte_identical(tmp_path, part_tokens):
    rows = synth_rows()
    a, b = RefCacheWriter(tmp_path / "ref", META, part_tokens), cachewrite.CacheWriter(tmp_path / "new", META, part_tokens)
    for r in rows:
        a.add_row(*r)
        b.add_row(*r)
    a.close()
    b.close()
    files = sorted(p.name for p in (tmp_path / "ref").iterdir())
    assert files == sorted(p.name for p in (tmp_path / "new").iterdir())
    for f in files:
        if f.startswith("part-"):
            assert members(tmp_path / "ref" / f) == members(tmp_path / "new" / f), f
    assert index_of(tmp_path / "ref") == index_of(tmp_path / "new")


def test_resume_is_identical(tmp_path):
    rows = synth_rows()
    for cls, name in ((RefCacheWriter, "ref"), (cachewrite.CacheWriter, "new")):
        w = cls(tmp_path / name, META, 60)
        for r in rows[:4]:
            w.add_row(*r)
        w.flush()
        w2 = cls(tmp_path / name, META, 60)               # a second job continues the folder
        assert w2.done_rows == w.done_rows
        for r in rows[w2.done_rows:]:
            w2.add_row(*r)
        w2.close()
    assert index_of(tmp_path / "ref") == index_of(tmp_path / "new")
    for p in (tmp_path / "ref").glob("part-*.npz"):
        assert members(p) == members(tmp_path / "new" / p.name)


def test_shared_writer_is_byte_identical(tmp_path):
    rows = synth_rows()
    hashes = [bytes([i]) * 20 for i in range(len(rows))]
    for cls, name in ((RefSharedWriter, "ref"), (cachewrite.SharedCacheWriter, "new")):
        w = cls(tmp_path / name / "store", tmp_path / name / "view", META, hashes, 70)
        for r in rows:
            w.add_row(*r)
        w.close()
    for sub in ("store", "view"):
        assert sorted(p.name for p in (tmp_path / "ref" / sub).iterdir()) == sorted(p.name for p in (tmp_path / "new" / sub).iterdir())
    for p in (tmp_path / "ref" / "store").glob("part-*.npz"):
        assert members(p) == members(tmp_path / "new" / "store" / p.name)
    assert (tmp_path / "ref" / "store" / "rowhashes.npy").read_bytes() == (tmp_path / "new" / "store" / "rowhashes.npy").read_bytes()
    assert index_of(tmp_path / "ref" / "store") == index_of(tmp_path / "new" / "store")
    assert index_of(tmp_path / "ref" / "view") == index_of(tmp_path / "new" / "view")


def test_cachewrite_imports_nothing_of_the_package():
    src = Path(cachewrite.__file__).read_text(encoding="utf-8")
    assert "from ." not in src and "import ndforge_worker" not in src and "torch" not in src.replace("no torch", "")


def test_read_back_with_the_reader(tmp_path):
    rows = synth_rows()
    w = cachewrite.CacheWriter(tmp_path / "c", META, 50)
    for r in rows:
        w.add_row(*r)
    w.close()
    rd = distilljob.CacheReader(tmp_path / "c")
    for i, (ids, logp, resid) in enumerate(rows):
        g = rd.get(i)
        assert (g[0] == ids).all() and np.allclose(g[1], logp.astype(np.float16)) and np.allclose(g[2], resid)
