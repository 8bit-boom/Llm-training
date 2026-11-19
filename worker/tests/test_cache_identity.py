"""Which weights a teacher cache, a Try result and a measurement belong to (numpy only, no torch)."""
import json
import shutil
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from ndforge_worker import distilljob as dj  # noqa: E402
from ndforge_worker import reshapetry, safetensors_io  # noqa: E402
from stfile import fill, write_file  # noqa: E402


def make_model(folder, n=40000, tensors=2):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text('{"a": 1}')
    write_file(folder / "model-00001.safetensors", [(f"w{i}", "BF16", (n,), fill(i, 2 * n)) for i in range(tensors)])
    return folder


def test_manifest_is_stable_across_a_copy(tmp_path):
    a = make_model(tmp_path / "a")
    b = tmp_path / "b"
    shutil.copytree(a, b)                                   # new file times, same bytes
    assert safetensors_io.weights_manifest(a) == safetensors_io.weights_manifest(b)
    assert safetensors_io.weights_manifest(a).startswith("dir:")


def test_manifest_changes_when_a_tensor_a_file_or_a_size_changes(tmp_path):
    base = safetensors_io.weights_manifest(make_model(tmp_path / "a"))
    other = make_model(tmp_path / "b")
    write_file(other / "model-00001.safetensors", [("w0", "BF16", (40000,), fill(9, 80000)), ("w1", "BF16", (40000,), fill(1, 80000))])
    assert safetensors_io.weights_manifest(other) != base, "a whole tensor changed"
    third = make_model(tmp_path / "c")
    write_file(third / "model-00002.safetensors", [("x", "BF16", (8,), fill(3, 16))])
    assert safetensors_io.weights_manifest(third) != base, "a file was added"
    fourth = make_model(tmp_path / "d", n=40001)
    assert safetensors_io.weights_manifest(fourth) != base, "a size changed"
    fifth = make_model(tmp_path / "e")
    (fifth / "config.json").write_text('{"a": 2}')
    assert safetensors_io.weights_manifest(fifth) != base, "another config"


def test_manifest_of_a_small_file_reads_all_of_it(tmp_path):
    a = make_model(tmp_path / "a", n=8)
    b = make_model(tmp_path / "b", n=8)
    write_file(b / "model-00001.safetensors", [("w0", "BF16", (8,), fill(0, 16)), ("w1", "BF16", (8,), fill(5, 16))])
    assert safetensors_io.weights_manifest(a) != safetensors_io.weights_manifest(b)


def test_a_hub_snapshot_is_named_by_its_commit(tmp_path):
    snap = tmp_path / "models--org--m" / "snapshots" / "abc123"
    make_model(snap)
    assert safetensors_io.weights_manifest(snap) == "hub:abc123"


def test_a_folder_without_weights_still_has_a_manifest(tmp_path):
    (tmp_path / "t").mkdir()
    assert safetensors_io.weights_manifest(tmp_path / "t").startswith("dir:")


# ---------------------------------------------------------------------------------------------------------- the shared teacher cache

META = {"teacher": "org/T", "k": 8, "tokenizer_hash": "t1", "max_len": 128, "rows": 4, "data_hash": "d", "general_key": {}, "general_rows": 0, "project_rows": 4}


def rows_of(n, length=4):
    return [{"input_ids": [i + 1] * length} for i in range(n)]


def cfg_for(train):
    return {"teacher": {"repo": "org/T", "load": "auto"}, "model": {"family": "qwen3_5"}, "data": {"train": str(train), "tokens": 1000, "general": None}}


def setup(tmp_path):
    teacher = make_model(tmp_path / "teacher")
    student = tmp_path / "student"
    student.mkdir()
    (student / "tokenizer.json").write_text('{"v": 1}')
    train = tmp_path / "train.jsonl"
    train.write_text('{"x": 1}\n')
    return teacher, student, cfg_for(train)


def fake(n):
    k = 8
    return (np.zeros((n, k), dtype=np.uint32), np.zeros((n, k), dtype=np.float16), np.zeros(n, dtype=np.float32))


def write_store(store, view, rows, meta):
    w = dj.SharedCacheWriter(store, view, meta, dj.row_hashes(rows), 8)
    for r in rows:
        w.add_row(*fake(len(r["input_ids"])))
    w.close()


def test_the_store_id_v2_follows_the_weights_and_v1_does_not(tmp_path):
    teacher, student, cfg = setup(tmp_path)
    v1, v2 = (dj.shared_store_id(cfg, teacher, student, 8, 128, 3407, v) for v in (1, 2))
    assert v1 != v2 and len(v2) == 20
    write_file(teacher / "model-00001.safetensors", [("w0", "BF16", (40000,), fill(8, 80000))])
    assert dj.shared_store_id(cfg, teacher, student, 8, 128, 3407, 1) == v1
    assert dj.shared_store_id(cfg, teacher, student, 8, 128, 3407, 2) != v2


def test_a_legacy_store_is_adopted_once_with_a_note_then_a_mismatch_picks_the_next(tmp_path):
    teacher, student, cfg = setup(tmp_path)
    base = tmp_path / "shared"
    rows = rows_of(4)
    legacy = base / dj.shared_store_id(cfg, teacher, student, 8, 128, 3407, 1)
    write_store(legacy, tmp_path / "view0", rows, {**META})                     # made before the weights were recorded: no teacher_weights
    meta = {**META, "teacher_weights": safetensors_io.weights_manifest(teacher)}
    store, note = dj.choose_store(base, cfg, teacher, student, 8, 128, 3407, meta, dj.row_hashes(rows))
    assert store == legacy and note and "weight files" in note
    assert json.loads((legacy / "index.json").read_text())["teacher_weights"] == meta["teacher_weights"]
    store, note = dj.choose_store(base, cfg, teacher, student, 8, 128, 3407, meta, dj.row_hashes(rows))
    assert store == legacy and note is None, "adopted once: no note the second time"
    # the weights are replaced: the adopted store no longer fits and a fresh one is used
    write_file(teacher / "model-00001.safetensors", [("w0", "BF16", (40000,), fill(8, 80000))])
    meta2 = {**META, "teacher_weights": safetensors_io.weights_manifest(teacher)}
    store, note = dj.choose_store(base, cfg, teacher, student, 8, 128, 3407, meta2, dj.row_hashes(rows))
    assert store != legacy and note is None
    assert store.name == dj.shared_store_id(cfg, teacher, student, 8, 128, 3407, 2)


def test_two_weight_identities_in_one_store_do_not_share_answers(tmp_path):
    base = tmp_path / "s"
    rows = rows_of(4)
    write_store(base / "id", tmp_path / "view", rows, {**META, "teacher_weights": "dir:a"})
    assert dj.store_extends(base / "id", {**META, "teacher_weights": "dir:a"}, dj.row_hashes(rows))
    assert not dj.store_extends(base / "id", {**META, "teacher_weights": "dir:b"}, dj.row_hashes(rows))
    assert dj.pick_store(base, "id", {**META, "teacher_weights": "dir:b"}, dj.row_hashes(rows)).name == "id-2"
    assert dj.store_extends(base / "id", dict(META), dj.row_hashes(rows)), "a side without the field matches"
    assert dj.cache_matches({**META, "teacher_weights": "dir:a"}, {**META, "teacher_weights": "dir:a"})
    assert not dj.cache_matches({**META, "teacher_weights": "dir:a"}, {**META, "teacher_weights": "dir:b"})
    assert dj.cache_matches(dict(META), {**META, "teacher_weights": "dir:b"}), "an old index without the field is accepted"


def test_cache_meta_carries_the_weights_only_when_asked(tmp_path):
    teacher, student, cfg = setup(tmp_path)
    stats = {"project_rows": 1, "general_rows": 0, "general_source": None}
    assert "teacher_weights" not in dj.cache_meta(cfg, rows_of(1), stats, student, 8, 128)
    assert dj.cache_meta(cfg, rows_of(1), stats, student, 8, 128, teacher)["teacher_weights"] == safetensors_io.weights_manifest(teacher)


# ---------------------------------------------------------------------------------------------------------- Try results and measurements

CALIB = {"version": 3, "samples": 8, "max_len": 64, "seed": 3407}


def test_a_try_key_without_weights_sample_still_matches_and_a_new_one_must(tmp_path):
    old = reshapetry.build_key("org/m", "dir:x", CALIB, "h", "auto")
    assert "weights_sample" not in old, "no new key unless it is given"
    new = reshapetry.build_key("org/m", "dir:x", CALIB, "h", "auto", "dir:s1")
    assert reshapetry.key_diff(old, new) is None, "answers saved before the field existed are accepted"
    assert reshapetry.key_diff(new, new) is None
    other = reshapetry.build_key("org/m", "dir:x", CALIB, "h", "auto", "dir:s2")
    assert reshapetry.key_diff(new, other) == "other weights of the model"
    assert reshapetry.key_diff(old, {**new, "weights": "dir:y"}) == "other weights of the model"


def test_the_measurement_is_refused_only_for_other_weights(tmp_path, monkeypatch):
    from ndforge_worker import reshapejob
    np_ = np
    folder = tmp_path / "scores"
    folder.mkdir()
    dims = {"intermediate": 4, "heads": 2, "lin_v_heads": 2}
    types = ["linear_attention", "full_attention"]
    meta = {"format": reshapejob.SCORES_FORMAT, "source": "org/m", "data_version": 3, "samples": 8, "max_len": 64, "reasoning": reshapejob.reasoning_key({})}
    arrays = {}
    for name, shape in (("mlp", (2, 4)), ("attn", (1, 2)), ("gdn", (1, 2))):
        for suffix in ("", "_h1", "_h2"):
            arrays[name + suffix] = np_.ones(shape, dtype=np_.float32)
    np_.savez(folder / "scores.npz", **arrays)
    cfg = {"source": {"repo": "org/m"}, "calibration": {"version": 3, "samples": 8, "max_len": 64}}

    def read(**kw):
        return reshapejob._read_scores(folder, cfg, 2, dims, types, **kw)
    (folder / "scores.json").write_text(json.dumps(meta))
    assert read() is not None and read(weights="dir:a") is not None, "an old measurement without the field is accepted"
    (folder / "scores.json").write_text(json.dumps({**meta, "weights": "dir:a"}))
    assert read(weights="dir:a") is not None and read() is not None
    assert read(weights="dir:b") is None
