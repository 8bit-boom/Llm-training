"""Reasoning-aware calibration (reshapejob.reasoning_key / _reasoning_rows / _calibration): keys without torch, one batch test with torch (CI)."""
import json

import pytest

from ndforge_worker import generaltext, reshapejob, reshapetry
from ndforge_worker.reshape import ReshapeError
from test_thinking_render import JinjaTok, PLAIN, Q, THINK


class WordTok(JinjaTok):
    """the real Qwen3.5 template, one token per word"""
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [hash(w) % 997 + 2 for w in text.split()]}


def source_rows(n_think=6, n_plain=3):
    rows = []
    for i in range(n_think):
        rows.append({"messages": [Q, {"role": "assistant", "content": f"<think>\nStep one of {i}. Step two. Step three.\n</think>\n\nAnswer {i}."}]})
    for i in range(n_plain):
        rows.append({"messages": [Q, {"role": "assistant", "content": f"Plain {i}."}]})
    return rows


@pytest.fixture
def hub(monkeypatch):
    seen = {}

    def loader(source, token):
        seen["source"] = dict(source)
        return iter(source_rows())
    monkeypatch.setattr(generaltext, "hub_loader", loader)
    return seen


def test_the_key_is_none_without_reasoning_and_names_every_part():
    assert reshapejob.reasoning_key({}) is None and reshapejob.reasoning_key({"reasoning": None}) is None
    assert reshapejob.reasoning_key({"reasoning": {"source": {}, "share": 0}}) is None
    k = reshapejob.reasoning_key({"reasoning": {}})
    assert k == {"id": reshapejob.REASONING_ID, "file": None, "config": None, "split": None, "column": None, "share": 0.5, "max_len": 2048, "render": generaltext.RENDER_VERSION}
    f = reshapejob.reasoning_key({"reasoning": {"source": {"file": "x.jsonl"}, "share": 0.3, "max_len": 1024}})
    assert f["id"] is None and f["file"] == "x.jsonl" and f["share"] == 0.3 and f["max_len"] == 1024
    assert reshapejob.reasoning_key({"reasoning": {"share": 5}})["share"] == 0.9            # never the whole sample


def test_the_try_key_and_the_measurement_know_the_reasoning_text():
    calib = {"version": 1, "samples": 64, "max_len": 512}
    plain = reshapetry.build_key("m", "w", calib, "h", "auto")
    assert plain["reasoning"] is None
    with_r = reshapetry.build_key("m", "w", {**calib, "reasoning": {}}, "h", "auto")
    assert with_r["reasoning"]["id"] == reshapejob.REASONING_ID
    assert reshapetry.key_diff(plain, with_r) == "another reasoning text" and reshapetry.key_diff(with_r, plain) == "another reasoning text"
    old = {k: v for k, v in plain.items() if k != "reasoning"}              # a file from before reasoning existed still fits a plain question
    assert reshapetry.key_diff(old, plain) is None and reshapetry.key_diff(old, with_r) == "another reasoning text"


def test_a_measurement_of_other_text_is_not_reused(tmp_path):
    np = pytest.importorskip("numpy")
    dims = {"intermediate": 4, "heads": 2, "lin_v_heads": 2}
    types = ["linear_attention", "full_attention"]
    meta = {"format": reshapejob.SCORES_FORMAT, "source": "m", "samples": 8, "max_len": 64, "data_version": 1, "reasoning": None}
    (tmp_path / "scores.json").write_text(json.dumps(meta), encoding="utf-8")
    arrays = {}
    for name, shape in {"mlp": (2, 4), "attn": (1, 2), "gdn": (1, 2)}.items():
        for suffix in ("", "_h1", "_h2"):
            arrays[name + suffix] = np.zeros(shape, dtype=np.float32)
    np.savez(tmp_path / "scores.npz", **arrays)
    cfg = {"source": {"repo": "m"}, "calibration": {"version": 1, "samples": 8, "max_len": 64}}
    assert reshapejob._read_scores(tmp_path, cfg, 2, dims, types) is not None
    cfg["calibration"]["reasoning"] = {}
    assert reshapejob._read_scores(tmp_path, cfg, 2, dims, types) is None
    meta["reasoning"] = reshapejob.reasoning_key(cfg["calibration"])
    (tmp_path / "scores.json").write_text(json.dumps(meta), encoding="utf-8")
    assert reshapejob._read_scores(tmp_path, cfg, 2, dims, types) is not None


def test_reasoning_rows_are_thinking_conversations_cut_to_the_length(hub):
    key = reshapejob.reasoning_key({"reasoning": {"source": {"id": "a/b"}, "max_len": 20}})
    rows = reshapejob._reasoning_rows(WordTok(), key, "qwen3_5", 4, 3407)
    assert len(rows) == 4 and all(r["kind"] == "reasoning" and len(r["input_ids"]) <= 20 and len(r["completion_mask"]) == len(r["input_ids"]) for r in rows)
    assert hub["source"]["id"] == "a/b" and hub["source"]["seed"] == 3407
    # the plain answers (no trace) are passed over for a family that thinks, so only the six conversations with a trace can be used
    assert len(reshapejob._reasoning_rows(WordTok(), key, "qwen3_5", 6, 3407)) == 6


def test_a_source_without_traces_and_an_unreadable_source_are_plain_errors(monkeypatch):
    monkeypatch.setattr(generaltext, "hub_loader", lambda s, t: iter([{"messages": [Q, PLAIN]}] * 5))
    key = reshapejob.reasoning_key({"reasoning": {}})
    with pytest.raises(ReshapeError, match="no conversations with a thinking trace"):
        reshapejob._reasoning_rows(WordTok(), key, "qwen3_5", 2, 1)

    def broken(s, t):
        raise ConnectionError("Failed to resolve huggingface.co")
    monkeypatch.setattr(generaltext, "hub_loader", broken)
    with pytest.raises(ReshapeError, match="could not be read for measuring"):
        reshapejob._reasoning_rows(WordTok(), key, "qwen3_5", 2, 1)


def test_the_calibration_batches_hold_the_project_rows_and_the_thinking_rows(hub, tmp_path):
    pytest.importorskip("torch")
    train = tmp_path / "train.jsonl"
    train.write_text("".join(json.dumps(r) + "\n" for r in [{"type": "text", "text": "word " * 30}] * 6 + [{"type": "chat", "messages": [Q, PLAIN]}] * 6), encoding="utf-8")
    cfg = {"train": str(train), "samples": 8, "max_len": 64, "seed": 3407, "batch": 2, "reasoning": {"source": {"id": "a/b"}, "share": 0.5, "max_len": 24}}
    batches = reshapejob._calibration(WordTok(), cfg, "qwen3_5")
    total = sum(len(b["input_ids"]) for b in batches)
    assert total == 8 and all(b["input_ids"].shape == b["attention_mask"].shape for b in batches)
    longest = max(int(b["attention_mask"].sum(dim=1).max()) for b in batches)
    assert longest <= 64
    again = reshapejob._calibration(WordTok(), cfg, "qwen3_5")
    assert all((a["input_ids"] == b["input_ids"]).all() for a, b in zip(batches, again)) and len(batches) == len(again)
    plain = reshapejob._calibration(WordTok(), {k: v for k, v in cfg.items() if k != "reasoning"}, "qwen3_5")
    count = lambda bs: sum(int(b["attention_mask"].sum()) for b in bs)      # noqa: E731
    assert sum(len(b["input_ids"]) for b in plain) == 8 and count(plain) != count(batches)
