"""Healing by distillation with real (tiny, random) models on a CPU. Skipped when torch / transformers are not installed (CI has them)."""
import io
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

from test_reshape_torch import IDS, load  # noqa: E402
from tinymodel import build_arch_dir, sample_rows, write_jsonl  # noqa: E402

from ndforge_worker import distilljob as dj  # noqa: E402


def run_in_process(monkeypatch, fn, cfg):
    from ndforge_worker import protocol
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    monkeypatch.setattr(protocol, "reserve_stdout", lambda: None)
    code = fn(cfg)
    return code, [json.loads(line) for line in buf.getvalue().splitlines()]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return build_arch_dir(tmp_path_factory.mktemp("distill") / "base", "qwen3_5", num_layers=4)


def write_data(tmp_path):
    train = tmp_path / "train.jsonl"
    write_jsonl(train, sample_rows())
    general = tmp_path / "general.jsonl"
    general.write_text("".join(json.dumps({"text": f"General text number {i}. " * 30}) + "\n" for i in range(4)), encoding="utf-8")
    return str(train), str(general)


def cache_cfg(tmp_path, base, **over):
    train, general = write_data(tmp_path)
    return {"job": "distill_cache", "teacher": {"repo": str(base), "load": "16bit"}, "student": str(base), "model": {"family": "qwen3_5"},
            "data": {"train": train, "tokens": 2000, "general": {"share": 0.5, "file": general}}, "k": 8, "max_seq": 64, "batch": 2, "chunk": 16, "part_tokens": 300,
            "out": str(tmp_path / "distill" / "cache"), "stop_file": str(tmp_path / "STOP"), **over}


def test_the_teacher_writes_its_top_k_answers_for_every_row(tmp_path, base, monkeypatch):
    cfg = cache_cfg(tmp_path, base)
    code, ev = run_in_process(monkeypatch, dj.run_cache, cfg)
    assert code == 0 and ev[-1]["type"] == "result", ev[-1]
    r = ev[-1]["data"]
    idx = json.loads((tmp_path / "distill" / "cache" / "index.json").read_text())
    assert idx["rows"] == r["rows"] == idx["done_rows"] and idx["complete"] and idx["k"] == 8 and idx["max_len"] == 64 and len(idx["parts"]) > 1
    assert idx["tokens"] == r["tokens"] and idx["teacher"] == str(base) and len(idx["data_hash"]) == 64 and len(idx["tokenizer_hash"]) == 64
    assert r["stats"]["project_rows"] == 16 and r["stats"]["general_rows"] > 0 and not r["notes"] and r["reused"] is False
    # row 0 against a direct forward pass of the same teacher
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(base))
    rows, _ = dj.build_rows(tok, cfg["data"], "qwen3_5", 64)
    assert dj.data_hash(rows) == idx["data_hash"]
    rd = dj.CacheReader(tmp_path / "distill" / "cache")
    model = load(base)
    for i in (0, len(rows) - 1):
        ids, lp, rs = rd.get(i)
        x = torch.tensor([rows[i]["input_ids"]])
        with torch.no_grad():
            ref_ids, ref_lp, ref_rs = dj.topk_from_logits(model(input_ids=x, use_cache=False).logits[0], 8)
        assert ids.shape == (x.shape[1], 8) and lp.shape == ids.shape and rs.shape == (x.shape[1],)
        assert np.allclose(lp.astype(np.float32), ref_lp.numpy(), atol=2e-2) and np.allclose(rs, ref_rs.numpy(), atol=2e-2)
        assert np.allclose(np.exp(lp.astype(np.float32)).sum(-1) + rs, 1.0, atol=2e-2) and (lp <= 0).all() and ids.max() < model.config.vocab_size


def test_a_stopped_cache_job_continues_with_the_remaining_rows_only(tmp_path, base, monkeypatch):
    cfg = cache_cfg(tmp_path, base)
    real_flush = dj.CacheWriter.flush

    def flush_then_stop(self):
        real_flush(self)
        (tmp_path / "STOP").write_text("")                       # the person pressed Stop right after the first part was saved
    monkeypatch.setattr(dj.CacheWriter, "flush", flush_then_stop)
    code, ev = run_in_process(monkeypatch, dj.run_cache, cfg)
    r = ev[-1]["data"]
    assert code == 0 and r["stopped"] is True and 0 < r["done_rows"] < r["rows"]
    done = r["done_rows"]
    monkeypatch.setattr(dj.CacheWriter, "flush", real_flush)
    (tmp_path / "STOP").unlink()
    calls = []
    real = dj._teacher_answers
    monkeypatch.setattr(dj, "_teacher_answers", lambda model, part, *a: calls.append(len(part)) or real(model, part, *a))
    code, ev = run_in_process(monkeypatch, dj.run_cache, cfg)
    r2 = ev[-1]["data"]
    assert code == 0 and not r2["stopped"] and sum(calls) == r["rows"] - done and r2["done_rows"] == r["rows"]
    assert json.loads((tmp_path / "distill" / "cache" / "index.json").read_text())["complete"] is True
    # a complete cache is reused without loading the teacher
    monkeypatch.setattr(dj, "_teacher_answers", lambda *a: (_ for _ in ()).throw(AssertionError("must not run")))
    code, ev = run_in_process(monkeypatch, dj.run_cache, cfg)
    assert code == 0 and ev[-1]["data"]["reused"] is True


def test_offline_general_text_is_reported_and_the_cache_covers_the_project_rows(tmp_path, base, monkeypatch):
    cfg = cache_cfg(tmp_path, base)
    cfg["data"]["general"] = {"share": 0.5, "id": "org/nothing-here"}
    monkeypatch.setattr(dj, "_general_texts", lambda source: (_ for _ in ()).throw(OSError("offline")))
    code, ev = run_in_process(monkeypatch, dj.run_cache, cfg)
    r = ev[-1]["data"]
    assert code == 0 and r["rows"] == 16 and r["stats"]["general_rows"] == 0 and "could not be downloaded" in r["notes"][0]


def test_a_different_tokenizer_is_refused(tmp_path, base, monkeypatch):
    other = tmp_path / "student"
    other.mkdir()
    (other / "tokenizer.json").write_text('{"different": true}')
    from ndforge_worker.reshape import ReshapeError
    with pytest.raises(ReshapeError, match="same tokenizer"):
        run_in_process(monkeypatch, dj.run_cache, cache_cfg(tmp_path, base, student=str(other)))
