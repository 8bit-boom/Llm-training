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
            "data": {"train": train, "tokens": 2000, "general": {"share": 0.5, "file": general}}, "k": 8, "max_seq": 128, "batch": 2, "chunk": 16, "part_tokens": 300,
            "out": str(tmp_path / "distill" / "cache"), "stop_file": str(tmp_path / "STOP"), **over}


def test_the_teacher_writes_its_top_k_answers_for_every_row(tmp_path, base, monkeypatch):
    cfg = cache_cfg(tmp_path, base)
    code, ev = run_in_process(monkeypatch, dj.run_cache, cfg)
    assert code == 0 and ev[-1]["type"] == "result", ev[-1]
    r = ev[-1]["data"]
    idx = json.loads((tmp_path / "distill" / "cache" / "index.json").read_text())
    assert idx["rows"] == r["rows"] == idx["done_rows"] and idx["complete"] and idx["k"] == 8 and idx["max_len"] == 128 and len(idx["parts"]) > 1
    assert idx["tokens"] == r["tokens"] and idx["teacher"] == str(base) and len(idx["data_hash"]) == 64 and len(idx["tokenizer_hash"]) == 64
    assert r["stats"]["project_rows"] == 16 and r["stats"]["general_rows"] > 0 and not r["notes"] and r["reused"] is False
    # row 0 against a direct forward pass of the same teacher
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(base))
    rows, _ = dj.build_rows(tok, cfg["data"], "qwen3_5", 128)
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
    hyper = {"rank": 8, "alpha": 8, "lr": 3e-3, "max_seq": 128, "batch": 2, "grad_accum": 1, "warmup": 0, "save_steps": 100, "logging_steps": 1, "tokens": 1500,
             "max_steps": 20, "seed": 1}
    hyper.update(over.pop("hyper", {}))
    cfg = {"job": "distill", "model": {"repo": str(student), "family": "qwen3_5"}, "method": "lora", "backend": "hf",
           "dataset": {"train": train, "general": {"share": 0.5, "file": general}}, "hyper": hyper,
           "distill": {"mode": mode, "cache": str(tmp_path / "distill" / "cache"), "teacher": {"repo": str(teacher), "load": "16bit"}, "alpha": 1.0, "k": 16, "chunk": 16,
                       "eval_every": 10},
           "eval": {"calibration": {"train": train, "samples": 8, "max_len": 128, "batch": 2, "version": 1}, "teacher_loss": 1.0}, "out": str(tmp_path / "distill" / "run")}
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
    conf = json.loads((Path(teacher) / "config.json").read_text())
    V = (conf.get("text_config") or conf)["vocab_size"]                                  # the tiny model's config is flat
    make_cache(tmp_path, student, teacher, monkeypatch, k=V)                               # k = the whole vocabulary: the top-k answer is exact
    first = {}
    for mode in ("cache", "online"):
        cfg = distill_cfg(tmp_path, student, teacher, mode, hyper={"max_steps": 1, "seed": 3407}, out=str(tmp_path / f"run-{mode}"))      # the cache was built with the default seed: the online rows must be the same rows
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
    idx["max_len"] = 128
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


def test_the_numpy_merge_equals_pefts_own_merge(tmp_path):
    """validates the adapter's key names and the scaling against PEFT's merge_and_unload on a real (tiny) Qwen3.5 text model"""
    import shutil

    from peft import LoraConfig, get_peft_model

    from ndforge_worker import safetensors_io, slice_io
    src = build_arch_dir(tmp_path / "b", "qwen3_5", num_layers=4)
    model = load(src)
    pm = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
    gen = torch.Generator().manual_seed(3)
    for name, p in pm.named_parameters():
        if "lora_B" in name:
            p.data.copy_(torch.randn(p.shape, generator=gen) * 0.05)           # a trained adapter, not the all-zero start
    pm.save_pretrained(tmp_path / "adapter")
    reference = pm.merge_and_unload()
    out = tmp_path / "merged"
    res = slice_io.merge_lora(safetensors_io.list_tensors(src), tmp_path / "adapter", out)
    assert res["merged"] > 10 and res["scaling"] == 2.0
    shutil.copy2(src / "config.json", out / "config.json")
    ours = load(out)
    ref_state = reference.state_dict()
    assert set(ref_state) == set(ours.state_dict())
    for name, p in ours.state_dict().items():
        assert torch.allclose(p, ref_state[name], atol=1e-5), name
    assert torch.allclose(ours(input_ids=IDS, use_cache=False).logits, reference(input_ids=IDS, use_cache=False).logits, atol=1e-4)


# ------------------------------------------------------------------------------------------------- memory settings and out-of-memory retries

OOM = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


def fail_at(monkeypatch, calls_to_fail, text="CUDA out of memory. Tried to allocate 2.00 GiB"):
    """student_loss raises an out-of-memory error at the given (1-based) call numbers: the same place where a real card runs out of memory."""
    real = dj.student_loss
    count = {"n": 0}

    def flaky(*a, **kw):
        count["n"] += 1
        if count["n"] in calls_to_fail:
            raise OOM(text)
        return real(*a, **kw)
    monkeypatch.setattr(dj, "student_loss", flaky)
    return count


def logs(ev):
    return [e["text"] for e in ev if e["type"] == "log"]


def test_an_out_of_memory_error_is_retried_with_checkpointing_and_finishes_with_the_same_steps(tmp_path, teacher, monkeypatch):
    student = thin_student(tmp_path, teacher)
    make_cache(tmp_path, student, teacher, monkeypatch)
    count = fail_at(monkeypatch, {8})                                    # step 8 of 20; a checkpoint was saved after step 5
    cfg = distill_cfg(tmp_path, student, teacher, "cache", hyper={"save_steps": 5, "checkpointing": "off"})
    code, ev = run_in_process(monkeypatch, dj.run, cfg)
    assert code == 0 and ev[-1]["type"] == "result", ev[-1]
    r = ev[-1]["data"]
    assert r["steps"] == 20 and r["stopped"] is False and r["oom_retries"] == ["gradient checkpointing"] and r["settings"]["checkpointing"] is True
    assert r["settings"]["batch"] == 2 and r["settings"]["max_seq"] == 128 and r["settings"]["rank"] == 8       # nothing that changes the maths
    assert any("Out of memory: trying again with gradient checkpointing" in t for t in logs(ev)) and any("Going on from step 5" in t for t in logs(ev))
    assert [e["change"] for e in ev if e["type"] == "heal_retry"] == ["gradient checkpointing"]
    first = [e for e in ev if e["type"] == "heal_settings"][0]["settings"]
    assert first["checkpointing"] is False
    assert count["n"] == 7 + 1 + 15                                      # 7 good calls, the failing one, then steps 6..20 again
    kd = [e["kd"] for e in metrics(ev)]
    assert all(np.isfinite(kd)) and r["kd_last"] == kd[-1] and Path(r["adapter_dir"], "adapter_config.json").is_file()


def test_a_second_out_of_memory_error_halves_the_batch_and_the_run_still_has_the_same_updates(tmp_path, teacher, monkeypatch):
    student = thin_student(tmp_path, teacher)
    make_cache(tmp_path, student, teacher, monkeypatch)
    fail_at(monkeypatch, {8, 12})                                         # the second failure comes during the retry (calls 8 .. 14 are steps 6 .. 12)
    cfg = distill_cfg(tmp_path, student, teacher, "cache", hyper={"save_steps": 5, "checkpointing": "off"})
    code, ev = run_in_process(monkeypatch, dj.run, cfg)
    r = ev[-1]["data"]
    assert code == 0 and r["steps"] == 20 and r["total_steps"] == 20, r
    assert r["oom_retries"][0] == "gradient checkpointing" and r["oom_retries"][1].startswith("a smaller batch")
    assert (r["settings"]["batch"], r["settings"]["grad_accum"]) == (1, 2)           # 2 x 1 -> 1 x 2: the same 2 sequences per update
    assert len([e for e in ev if e["type"] == "heal_retry"]) == 2


def test_when_every_retry_fails_the_message_names_the_numbers_and_the_next_steps(tmp_path, teacher, monkeypatch):
    from ndforge_worker import healmem
    student = thin_student(tmp_path, teacher)
    make_cache(tmp_path, student, teacher, monkeypatch)
    fail_at(monkeypatch, set(range(1, 400)))
    cfg = distill_cfg(tmp_path, student, teacher, "cache", hyper={"checkpointing": "off"})
    with pytest.raises(healmem.HealOutOfMemory) as e:
        run_in_process(monkeypatch, dj.run, cfg)
    text = str(e.value)
    assert "step train" in text and "sequence length 128" in text and "LoRA rank 8" in text and "shorter sequence length" in text and "gradient checkpointing" in text


def test_a_failure_that_is_not_out_of_memory_is_not_retried(tmp_path, teacher, monkeypatch):
    student = thin_student(tmp_path, teacher)
    make_cache(tmp_path, student, teacher, monkeypatch)
    count = fail_at(monkeypatch, set())
    monkeypatch.setattr(dj, "student_loss", lambda *a, **kw: (_ for _ in ()).throw(ValueError("not memory")))
    with pytest.raises(ValueError, match="not memory"):
        run_in_process(monkeypatch, dj.run, distill_cfg(tmp_path, student, teacher, "cache"))
    assert count["n"] == 0


def captured_trainer(monkeypatch):
    got = []
    real = dj._trainer_class

    def wrap(base):
        cls = real(base)

        class Spy(cls):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                got.append(self)
        return Spy
    monkeypatch.setattr(dj, "_trainer_class", wrap)
    return got


def test_the_settings_reach_the_trainer(tmp_path, teacher, monkeypatch):
    student = thin_student(tmp_path, teacher)
    make_cache(tmp_path, student, teacher, monkeypatch)
    got = captured_trainer(monkeypatch)
    cfg = distill_cfg(tmp_path, student, teacher, "cache", hyper={"batch": 1, "grad_accum": 4, "max_steps": 1, "checkpointing": "on", "optimizer": "adamw"})
    cfg["distill"]["chunk"] = 32
    code, ev = run_in_process(monkeypatch, dj.run, cfg)
    assert code == 0, ev[-1]
    t = got[0]
    a = t.args
    assert (a.per_device_train_batch_size, a.gradient_accumulation_steps, a.max_length, a.gradient_checkpointing) == (1, 4, 128, True)
    assert a.gradient_checkpointing_kwargs == {"use_reentrant": False} and getattr(a.optim, "value", a.optim) == "adamw_torch" and t.distill["chunk"] == 32
    assert t.model.is_gradient_checkpointing is True
    cfg = distill_cfg(tmp_path, student, teacher, "cache", hyper={"max_steps": 1, "checkpointing": "off"}, out=str(tmp_path / "run2"))
    code, ev = run_in_process(monkeypatch, dj.run, cfg)
    assert code == 0 and got[1].args.gradient_checkpointing is False and got[1].args.gradient_checkpointing_kwargs in (None, {})
    assert [e for e in ev if e["type"] == "heal_settings"][0]["settings"]["checkpointing"] is False


def test_the_first_step_loss_is_the_same_with_and_without_gradient_checkpointing(tmp_path, teacher, monkeypatch):
    student = thin_student(tmp_path, teacher)
    make_cache(tmp_path, student, teacher, monkeypatch)
    first = {}
    for ck in ("off", "on"):
        cfg = distill_cfg(tmp_path, student, teacher, "cache", hyper={"max_steps": 1, "checkpointing": ck}, out=str(tmp_path / f"ck-{ck}"))
        cfg["distill"]["eval_every"] = 0
        cfg["eval"] = None
        code, ev = run_in_process(monkeypatch, dj.run, cfg)
        assert code == 0, ev[-1]
        first[ck] = metrics(ev)[0]["kd"]
    assert first["off"] > 0 and first["on"] == pytest.approx(first["off"], rel=1e-4)


def test_the_settings_follow_the_students_size_unless_the_job_names_them(tmp_path, base):
    cfg = {"hyper": {"student_b": 21.0}}
    h = dj.resolve_hyper(cfg, base)
    assert (h["max_seq"], h["rank"], h["alpha"], h["batch"], h["grad_accum"], h["chunk"]) == (1024, 32, 32, 2, 16, 128)
    h = dj.resolve_hyper({"hyper": {"student_b": 3.0}}, base)
    assert (h["max_seq"], h["rank"], h["batch"], h["grad_accum"], h["chunk"]) == (2048, 64, 4, 8, 256)
    h = dj.resolve_hyper({"hyper": {"student_b": 21.0, "batch": 1, "rank": 16, "max_seq": 512}}, base)
    assert (h["max_seq"], h["rank"], h["alpha"], h["batch"], h["grad_accum"]) == (512, 16, 16, 1, 32)            # batch named: 32 sequences per update stay
    h = dj.resolve_hyper({}, base)                                                                           # the tiny model: its size comes from its files
    assert h["student_b"] < 0.01 and h["max_seq"] == 2048
    assert dj._decide_checkpointing({"student_b": 21.0, "max_seq": 1024, "batch": 2}, base, 100.0) is True
    assert dj._decide_checkpointing({"student_b": 0.1, "max_seq": 128, "batch": 2}, base, 100.0) is False
    assert dj._decide_checkpointing({"student_b": 0.1, "max_seq": 128, "batch": 2, "checkpointing": "on"}, base, 100.0) is True
