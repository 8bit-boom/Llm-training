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


# ------------------------------------------------------------------------------------------------------------ the student

from test_reshape_width_torch import perturb, run_width  # noqa: E402


@pytest.fixture(scope="module")
def teacher(tmp_path_factory):
    d = build_arch_dir(tmp_path_factory.mktemp("teacher") / "base", "qwen3_5", num_layers=4)
    perturb(d)                                   # logits with some spread, so there is something to learn
    return d


def thin_student(tmp_path, teacher):
    j, _ = run_width(tmp_path / "thin", teacher, {"intermediate_size": 80}, layers=4)
    return j["final"]


def distill_cfg(tmp_path, student, teacher, mode, **over):
    train, general = write_data(tmp_path)
    hyper = {"rank": 8, "alpha": 8, "lr": 3e-3, "max_seq": 64, "batch": 2, "grad_accum": 1, "warmup": 0, "save_steps": 100, "logging_steps": 1, "tokens": 1500,
             "max_steps": 20, "seed": 1}
    hyper.update(over.pop("hyper", {}))
    cfg = {"job": "distill", "model": {"repo": str(student), "family": "qwen3_5"}, "method": "lora", "backend": "hf",
           "dataset": {"train": train, "general": {"share": 0.5, "file": general}}, "hyper": hyper,
           "distill": {"mode": mode, "cache": str(tmp_path / "distill" / "cache"), "teacher": {"repo": str(teacher), "load": "16bit"}, "alpha": 1.0, "k": 16, "chunk": 16,
                       "eval_every": 10},
           "eval": {"calibration": {"train": train, "samples": 8, "max_len": 64, "batch": 2, "version": 1}, "teacher_loss": 1.0}, "out": str(tmp_path / "distill" / "run")}
    cfg.update(over)
    return cfg


def make_cache(tmp_path, student, teacher, monkeypatch, k=16):
    cfg = cache_cfg(tmp_path, teacher, student=str(student), k=k)
    cfg["data"]["tokens"] = 1500
    code, ev = run_in_process(monkeypatch, dj.run_cache, cfg)
    assert code == 0, ev[-1]


def metrics(ev, name="distill_metric"):
    return [e for e in ev if e["type"] == name]


def test_healing_in_cache_mode_lowers_the_kl_and_saves_a_lora(tmp_path, teacher, monkeypatch):
    student = thin_student(tmp_path, teacher)
    make_cache(tmp_path, student, teacher, monkeypatch)
    code, ev = run_in_process(monkeypatch, dj.run, distill_cfg(tmp_path, student, teacher, "cache"))
    assert code == 0 and ev[-1]["type"] == "result", ev[-1]
    r = ev[-1]["data"]
    kd = [e["kd"] for e in metrics(ev)]
    assert len(kd) >= 10 and all(np.isfinite(kd)) and r["kd_first"] == kd[0] and r["kd_last"] == kd[-1] and r["steps"] == 20 and r["mode"] == "cache"
    assert sum(kd[-5:]) / 5 < 0.95 * sum(kd[:5]) / 5, kd
    assert r["tokens_seen"] > 0 and r["stopped"] is False and Path(r["adapter_dir"], "adapter_config.json").is_file()
    cal = [e["calib_loss"] for e in metrics(ev, "distill_eval")]
    assert len(cal) >= 3 and r["calib_loss_after"] is not None and r["calib_loss_before"] is not None and r["teacher_calib_loss"] == 1.0
    assert {"metric", "start", "progress"} <= {e["type"] for e in ev}                  # the existing loss chart keeps working
    assert all(np.isfinite([e["loss"] for e in ev if e["type"] == "metric"]))


def test_the_first_step_loss_is_the_same_in_cache_and_online_mode(tmp_path, teacher, monkeypatch):
    student = thin_student(tmp_path, teacher)
    V = json.loads((Path(teacher) / "config.json").read_text())["text_config"]["vocab_size"]
    make_cache(tmp_path, student, teacher, monkeypatch, k=V)                               # k = the whole vocabulary: the top-k answer is exact
    first = {}
    for mode in ("cache", "online"):
        cfg = distill_cfg(tmp_path, student, teacher, mode, hyper={"max_steps": 1}, out=str(tmp_path / f"run-{mode}"))
        cfg["distill"]["k"] = V
        cfg["distill"]["eval_every"] = 0
        cfg["eval"] = None
        code, ev = run_in_process(monkeypatch, dj.run, cfg)
        assert code == 0, ev[-1]
        first[mode] = metrics(ev)[0]["kd"]
    assert first["cache"] > 0 and first["online"] == pytest.approx(first["cache"], rel=2e-2)


def test_a_cache_of_other_data_is_refused_and_a_stop_file_stops_with_a_saved_adapter(tmp_path, teacher, monkeypatch):
    from ndforge_worker.reshape import ReshapeError
    student = thin_student(tmp_path, teacher)
    make_cache(tmp_path, student, teacher, monkeypatch)
    idx = json.loads((tmp_path / "distill" / "cache" / "index.json").read_text())
    idx["max_len"] = 32
    (tmp_path / "distill" / "cache" / "index.json").write_text(json.dumps(idx))
    with pytest.raises(ReshapeError, match="other data"):
        run_in_process(monkeypatch, dj.run, distill_cfg(tmp_path, student, teacher, "cache"))
    idx["max_len"] = 64
    (tmp_path / "distill" / "cache" / "index.json").write_text(json.dumps(idx))
    stop = tmp_path / "run-stop" / "STOP"
    cfg = distill_cfg(tmp_path, student, teacher, "cache", out=str(tmp_path / "run-stop"), stop_file=str(stop))
    from ndforge_worker import callbacks
    real = callbacks.LiveCallback.on_step_end

    def stop_after_two(self, args, state, control, **kw):
        if state.global_step == 2:
            stop.parent.mkdir(parents=True, exist_ok=True)
            stop.write_text("")
        return real(self, args, state, control, **kw)
    monkeypatch.setattr(callbacks.LiveCallback, "on_step_end", stop_after_two)
    code, ev = run_in_process(monkeypatch, dj.run, cfg)
    r = ev[-1]["data"]
    assert code == 0 and r["stopped"] is True and r["steps"] == 2 and Path(r["adapter_dir"]).is_dir()
    assert any((tmp_path / "run-stop" / "checkpoints").glob("checkpoint-*"))           # a checkpoint was saved for "Continue"
