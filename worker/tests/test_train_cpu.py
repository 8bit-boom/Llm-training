"""The training pipeline end to end on a CPU with a tiny random model and the REAL Qwen3.5 / Gemma 4 chat templates."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from tinymodel import build_arch_dir, build_llama_dir, build_tokenizer, sample_rows, write_jsonl  # noqa: E402

WORKER = Path(__file__).resolve().parent.parent


def encode(family):
    from ndforge_worker import data
    tok = build_tokenizer(family)
    return tok, data.encode_rows(tok, sample_rows(3, 1), family, 256)


@pytest.mark.parametrize("family,marker_end", [("qwen3_5", "<|im_end|>\n"), ("gemma4", "<turn|>\n")])
def test_only_the_answer_is_learned(family, marker_end):
    tok, (rows, stats) = encode(family)
    assert stats["chat"] == 3 and stats["text"] == 1 and stats["skipped_too_long"] == 0
    chat = [r for r in rows if not all(r["completion_mask"])][0]
    learned = [i for i, m in zip(chat["input_ids"], chat["completion_mask"]) if m]
    text = tok.decode(learned)
    assert text.startswith("Anders de Vallière") and text.endswith(marker_end.strip()) or marker_end.strip() in text, text
    # the system and user turns and the assistant opening are context, not learned
    context = tok.decode([i for i, m in zip(chat["input_ids"], chat["completion_mask"]) if not m])
    assert "Who is Anders" in context and "GM secret" not in context
    # a world-book text is learned whole
    book = [r for r in rows if all(r["completion_mask"])][0]
    assert "Hunts colossal beasts" in tok.decode(book["input_ids"])


def test_qwen_prompt_has_the_empty_thinking_block_and_gemma_prompt_ends_with_the_model_turn():
    from ndforge_worker import data
    q = build_tokenizer("qwen3_5")
    prompt, full = data.render_chat(q, sample_rows(1, 0)[0]["messages"], "qwen3_5")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") and full.endswith("fire.)<|im_end|>\n")
    g = build_tokenizer("gemma4")
    prompt, full = data.render_chat(g, sample_rows(1, 0)[0]["messages"], "gemma4")
    assert prompt.startswith("<bos>") and prompt.rstrip().endswith("<|turn>model") and full.endswith("fire.)<turn|>\n")


def test_rows_that_do_not_fit_are_skipped_or_cut():
    from ndforge_worker import data
    tok = build_tokenizer("qwen3_5")
    rows = sample_rows(1, 0) + [{"type": "text", "text": "word " * 500}]
    enc, st = data.encode_rows(tok, rows, "qwen3_5", 64)
    assert st["cut"] == 1 and all(len(r["input_ids"]) <= 64 for r in enc)
    enc, st = data.encode_rows(tok, rows[:1], "qwen3_5", 8)
    assert enc == [] and st["skipped_too_long"] == 1


def run_job(tmp_path, family, arch="llama", **over):
    model_dir = tmp_path / "model"
    if not model_dir.exists():
        (build_arch_dir if arch == "real" else build_llama_dir)(model_dir, family)
    train = tmp_path / "train.jsonl"
    write_jsonl(train, sample_rows(12, 4))
    out = tmp_path / "run"
    job = {"job": "train", "model": {"repo": str(model_dir), "family": family}, "method": "lora", "backend": "hf", "dataset": {"train": str(train)},
           "hyper": {"rank": 4, "alpha": 4, "lr": 5e-3, "epochs": 1, "max_steps": 6, "batch": 2, "grad_accum": 1, "max_seq": 128, "warmup": 1,
                     "weight_decay": 0.0, "scheduler": "linear", "seed": 1, "save_steps": 3, "logging_steps": 1},
           "out": str(out), "resume": False, "probe_every": 3, "probe_tokens": 6,
           "probes": [{"system": "You are the assistant.", "question": "Who is Anders?"}]}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(job.get(k), dict):
            job[k].update(v)
        else:
            job[k] = v
    f = tmp_path / "job.json"
    f.write_text(json.dumps(job), encoding="utf-8")
    p = subprocess.run([sys.executable, "-X", "utf8", "-m", "ndforge_worker", str(f)], cwd=WORKER, capture_output=True, text=True, timeout=600)
    events = [json.loads(line) for line in p.stdout.splitlines() if line.startswith("{")]
    return p, events, out


@pytest.mark.parametrize("family,arch", [("qwen3_5", "llama"), ("gemma4", "llama"), ("qwen3_5", "real"), ("gemma4", "real")])
def test_a_tiny_training_run_reports_everything_and_saves_an_adapter(tmp_path, family, arch):
    p, ev, out = run_job(tmp_path, family, arch)
    assert p.returncode == 0, p.stderr[-3000:] + p.stdout[-1000:]
    kinds = [e["type"] for e in ev]
    assert kinds.count("start") == 1 and kinds.count("metric") >= 5 and kinds[-1] == "result"
    metrics = [e for e in ev if e["type"] == "metric"]
    assert all(m["loss"] == m["loss"] for m in metrics) and metrics[-1]["step"] == 6 and metrics[-1]["total_steps"] == 6
    assert metrics[0]["loss"] > metrics[-1]["loss"], [m["loss"] for m in metrics]          # it learns
    samples = [e for e in ev if e["type"] == "sample"]
    assert {s["step"] for s in samples} >= {0, 6} and all(s["question"] == "Who is Anders?" for s in samples)
    res = ev[-1]["data"]
    assert res["steps"] == 6 and res["stopped"] is False and res["backend"] == "hf" and Path(res["adapter_dir"], "adapter_config.json").exists()
    assert list((out / "checkpoints").glob("checkpoint-*")) and (out / "summary.json").exists()
    assert res["data"]["chat"] == 12 and res["data"]["text"] == 4


def test_resume_continues_from_the_last_checkpoint(tmp_path):
    p, ev, out = run_job(tmp_path, "qwen3_5")
    assert p.returncode == 0, p.stderr[-2000:]
    p2, ev2, _ = run_job(tmp_path, "qwen3_5", hyper={"max_steps": 9}, resume=True)
    assert p2.returncode == 0, p2.stderr[-2000:]
    start = [e for e in ev2 if e["type"] == "start"][0]
    assert start["resumed_from"] == 6
    assert [e["step"] for e in ev2 if e["type"] == "metric"] == [7, 8, 9]


def test_resume_without_a_checkpoint_starts_over(tmp_path):
    p, ev, _ = run_job(tmp_path, "gemma4", resume=True)
    assert p.returncode == 0 and any(e["type"] == "log" and "No checkpoint" in e["text"] for e in ev)


def test_errors_are_plain(tmp_path):
    p, ev, _ = run_job(tmp_path, "qwen3_5", dataset={"train": str(tmp_path / "missing.jsonl")})
    assert p.returncode == 1 and ev[-1]["type"] == "error" and ev[-1]["code"] == "train_failed"
    (tmp_path / "empty.jsonl").write_text("")
    p, ev, _ = run_job(tmp_path, "qwen3_5", dataset={"train": str(tmp_path / "empty.jsonl")})
    assert "Save a version in Step 2" in ev[-1]["message"]


def test_stop_file_ends_training_early_but_saves(tmp_path):
    from ndforge_worker.callbacks import LiveCallback

    class Ctl:
        should_save = False
        should_training_stop = False

    stop = tmp_path / "STOP"
    cb = LiveCallback(str(stop))
    ctl = Ctl()
    cb.on_step_end(None, type("S", (), {"global_step": 1})(), ctl)
    assert not ctl.should_training_stop
    stop.write_text("")
    cb.on_step_end(None, type("S", (), {"global_step": 2})(), ctl)
    assert ctl.should_save and ctl.should_training_stop and cb.stopped
