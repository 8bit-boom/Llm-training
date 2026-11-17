"""Checkpoints cut short by a crash are not resumed from (ckpt.py)."""
import json

from ndforge_worker import ckpt, protocol


def make(root, step, weights=b"w" * 10, optimizer=b"o" * 10, state="{}", adapter=True):
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    if state is not None:
        (d / "trainer_state.json").write_text(state, encoding="utf-8")
    if weights is not None:
        (d / ("adapter_model.safetensors" if adapter else "model.safetensors")).write_bytes(weights)
    if optimizer is not None:
        (d / "optimizer.pt").write_bytes(optimizer)
    return d


def test_good_checkpoints_are_returned_oldest_first_by_step_not_by_text(tmp_path):
    for step in (100, 20, 3):
        make(tmp_path, step)
    assert [p.name for p in ckpt.usable(tmp_path)] == ["checkpoint-3", "checkpoint-20", "checkpoint-100"]
    assert ckpt.usable(tmp_path / "nowhere") == []


def test_a_cut_short_newest_checkpoint_is_renamed_and_the_one_before_is_used(tmp_path, monkeypatch):
    logs = []
    monkeypatch.setattr(protocol, "log", logs.append)
    make(tmp_path, 50)
    make(tmp_path, 100, optimizer=b"")                      # the power went while the optimizer was written
    got = ckpt.usable(tmp_path)
    assert [p.name for p in got] == ["checkpoint-50"]
    assert (tmp_path / "checkpoint-100.broken").is_dir() and not (tmp_path / "checkpoint-100").exists()
    assert any("checkpoint-100" in m and "optimizer" in m for m in logs)
    # the broken folder is never picked again, and a second look changes nothing
    assert [p.name for p in ckpt.usable(tmp_path)] == ["checkpoint-50"]


def test_every_kind_of_damage_is_found(tmp_path):
    cases = {
        "state_missing": dict(state=None),
        "state_cut": dict(state='{"global_step": 5, "log_hist'),
        "weights_empty": dict(weights=b""),
        "weights_missing": dict(weights=None),
        "optimizer_missing": dict(optimizer=None),
    }
    for i, kw in enumerate(cases.values(), start=1):
        make(tmp_path, i * 10, **kw)
    make(tmp_path, 1000)
    assert [p.name for p in ckpt.usable(tmp_path)] == ["checkpoint-1000"]
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.endswith(".broken")) == sorted(f"checkpoint-{i * 10}.broken" for i in range(1, 6))


def test_a_full_model_checkpoint_and_a_second_broken_copy_are_handled(tmp_path):
    make(tmp_path, 7, adapter=False)
    (tmp_path / "checkpoint-9.broken").mkdir()              # an earlier broken one with the same name
    make(tmp_path, 9, weights=b"")
    assert [p.name for p in ckpt.usable(tmp_path)] == ["checkpoint-7"]
    assert (tmp_path / "checkpoint-9.broken2").is_dir()
    # a shard of a bigger model is a weights file too; one empty shard is damage
    d = make(tmp_path, 11, weights=None)
    (d / "model-00001-of-00002.safetensors").write_bytes(b"x")
    (d / "model-00002-of-00002.safetensors").write_bytes(b"")
    assert [p.name for p in ckpt.usable(tmp_path)] == ["checkpoint-7"]


def test_the_state_file_must_be_json(tmp_path):
    d = make(tmp_path, 1, state=json.dumps({"global_step": 1}))
    assert ckpt.problem(d) == ""
    assert "trainer_state" in ckpt.problem(make(tmp_path, 2, state="not json"))
