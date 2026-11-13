import json
import struct

import pytest

from ndforge_worker import safetensors_io as st
from stfile import fill, read_file, read_folder, write_file


def make_source(tmp_path, split=False):
    """three tensors with deterministic bytes; optionally spread over two shards + index"""
    t = [("a.weight", "BF16", (3, 4), fill(0, 24)), ("b.weight", "F32", (5,), fill(1, 20)), ("c.bias", "F16", (7,), fill(2, 14))]
    if not split:
        write_file(tmp_path / "model.safetensors", t)
        return t
    write_file(tmp_path / "model-00001-of-00002.safetensors", t[:2])
    write_file(tmp_path / "model-00002-of-00002.safetensors", t[2:])
    wm = {"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00001-of-00002.safetensors", "c.bias": "model-00002-of-00002.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    return t


def test_list_tensors_single_and_indexed(tmp_path):
    for split in (False, True):
        d = tmp_path / f"s{split}"
        d.mkdir()
        t = make_source(d, split)
        refs = st.list_tensors(d)
        assert list(refs) == ["a.weight", "b.weight", "c.bias"]
        r = refs["b.weight"]
        assert (r.dtype, r.shape, r.nbytes, r.numel) == ("F32", (5,), 20, 5)
        with open(r.file, "rb") as fh:
            fh.seek(r.start)
            assert fh.read(r.nbytes) == t[1][3]


def test_read_header_rejects_garbage(tmp_path):
    p = tmp_path / "x.safetensors"
    p.write_bytes(b"abc")
    with pytest.raises(ValueError, match="not a safetensors file"):
        st.read_header(p)
    p.write_bytes(struct.pack("<Q", 999) + b"{}")
    with pytest.raises(ValueError, match="not a safetensors file"):
        st.read_header(p)
    p.write_bytes(struct.pack("<Q", 4) + b"nope")
    with pytest.raises(ValueError, match="not a safetensors file"):
        st.read_header(p)
    # offsets that point past the end of the file
    raw = json.dumps({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}).encode()
    p.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)
    with pytest.raises(ValueError, match="bad entry"):
        st.read_header(p)


def test_duplicate_names_and_empty_folder(tmp_path):
    write_file(tmp_path / "a.safetensors", [("w", "F32", (1,), fill(0, 4))])
    write_file(tmp_path / "b.safetensors", [("w", "F32", (1,), fill(1, 4))])
    with pytest.raises(ValueError, match="more than one file"):
        st.list_tensors(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no .safetensors"):
        st.list_tensors(empty)


def outs(src_dir):
    refs = st.list_tensors(src_dir)
    return refs, [
        st.OutTensor("b.weight", "F32", (5,), refs["b.weight"]),
        st.OutTensor("new.zero", "BF16", (2, 3), None),
        st.OutTensor("a.weight", "BF16", (3, 4), refs["a.weight"]),
        st.OutTensor("a2.weight", "BF16", (3, 4), refs["a.weight"]),      # the same source twice (a duplicated layer)
    ]


def test_round_trip_is_byte_exact_with_tiny_chunks(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    t = make_source(src)
    refs, plan = outs(src)
    out = tmp_path / "out"
    seen = []
    index = st.write_shards(plan, out, chunk=7, on_progress=lambda d, n: seen.append((d, n)))
    assert [p.name for p in out.iterdir()] == ["model.safetensors"]        # one shard: no index file
    assert index["metadata"]["total_size"] == 20 + 12 + 24 + 24 and seen[-1] == (80, 80)
    blob = (out / "model.safetensors").read_bytes()
    (n,) = struct.unpack("<Q", blob[:8])
    assert n % 8 == 0
    meta = json.loads(blob[8:8 + n])
    assert meta["__metadata__"] == {"format": "pt"}
    back = read_file(out / "model.safetensors")
    assert back["b.weight"] == ("F32", (5,), t[1][3])
    assert back["new.zero"] == ("BF16", (2, 3), bytes(12))
    assert back["a.weight"][2] == t[0][3] and back["a2.weight"][2] == t[0][3] and back["a.weight"][:2] == ("BF16", (3, 4))
    # order of the tensors in the file is the order given
    assert [k for k, _ in sorted(meta.items(), key=lambda kv: kv[1]["data_offsets"][0] if kv[0] != "__metadata__" else -1)][1:] == [p.name for p in plan]
    assert list(st.list_tensors(out)) == [p.name for p in plan]


def test_sharding_and_index(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_source(src)
    refs, plan = outs(src)
    out = tmp_path / "out"
    index = st.write_shards(plan, out, shard_bytes=30)       # 20 | 12+24 (36 > 30 alone is fine) | 24 ...
    files = sorted(p.name for p in out.glob("*.safetensors"))
    n = len(files)
    assert n >= 3 and files[0] == f"model-00001-of-{n:05d}.safetensors"
    on_disk = json.loads((out / "model.safetensors.index.json").read_text())
    assert on_disk == index and on_disk["metadata"]["total_size"] == 80
    assert set(on_disk["weight_map"]) == {p.name for p in plan} and set(on_disk["weight_map"].values()) == set(files)
    data = read_folder(out)
    assert data["a2.weight"][2] == fill(0, 24)
    assert list(st.list_tensors(out)) == sorted(on_disk["weight_map"], key=lambda k: (on_disk["weight_map"][k], [p.name for p in plan].index(k)))


def test_a_tensor_bigger_than_a_shard_gets_its_own(tmp_path):
    big = [st.OutTensor("x", "F32", (100,), None), st.OutTensor("y", "F32", (1,), None)]
    index = st.write_shards(big, tmp_path, shard_bytes=16)
    assert sorted(set(index["weight_map"].values())) == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def test_stop_raises_and_leaves_no_index(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_source(src)
    refs, plan = outs(src)
    out = tmp_path / "out"
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 3

    with pytest.raises(st.Cancelled):
        st.write_shards(plan, out, shard_bytes=30, chunk=5, should_stop=stop)
    assert not (out / "model.safetensors.index.json").exists()


def test_bad_plans_are_refused(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_source(src)
    refs = st.list_tensors(src)
    with pytest.raises(ValueError, match="same name"):
        st.write_shards([st.OutTensor("a", "F32", (1,), None), st.OutTensor("a", "F32", (1,), None)], tmp_path / "o1")
    with pytest.raises(ValueError, match="expected"):
        st.write_shards([st.OutTensor("a", "F32", (2,), refs["a.weight"])], tmp_path / "o2")
    with pytest.raises(ValueError, match="unsupported tensor type"):
        st.write_shards([st.OutTensor("a", "NOPE", (2,), None)], tmp_path / "o3")


def test_hub_metadata_adapter():
    class Info:
        def __init__(self, dtype, shape):
            self.dtype, self.shape = dtype, shape

    class F:
        def __init__(self, tensors):
            self.tensors = tensors

    class Meta:
        files_metadata = {"m1.safetensors": F({"a": Info("BF16", [2, 3])}), "m2.safetensors": F({"b": Info("F32", [4])})}

    assert st.tensors_from_hub_metadata(Meta) == {"a": ("BF16", (2, 3)), "b": ("F32", (4,))}


def test_real_safetensors_library_reads_our_files(tmp_path):
    safe_open = pytest.importorskip("safetensors").safe_open
    pytest.importorskip("numpy")
    src = tmp_path / "src"
    src.mkdir()
    write_file(src / "model.safetensors", [("a", "F32", (3, 4), fill(0, 48)), ("b", "F16", (7,), fill(1, 14))])
    refs = st.list_tensors(src)
    out = tmp_path / "out"
    st.write_shards([st.OutTensor("a", "F32", (3, 4), refs["a"]), st.OutTensor("z", "F16", (2,), None), st.OutTensor("b", "F16", (7,), refs["b"])], out)
    with safe_open(str(out / "model.safetensors"), framework="numpy") as fh:
        assert sorted(fh.keys()) == ["a", "b", "z"]
        assert fh.get_tensor("a").tobytes() == fill(0, 48) and fh.get_tensor("b").tobytes() == fill(1, 14)
        assert not fh.get_tensor("z").any()
