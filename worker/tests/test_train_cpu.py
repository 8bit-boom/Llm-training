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


# ---------------------------------------------------------------------------------------------------------------- export and warm start

def run_export(tmp_path, family, run_out, env=None, **over):
    job = {"job": "export", "adapter": str(run_out / "adapter"), "base": {"repo": str(tmp_path / "model"), "family": family}, "out": str(tmp_path / "exp"),
           "quant": "q4_k_m", "backend": "hf", "gguf": False}
    job.update(over)
    f = tmp_path / "export-job.json"
    f.write_text(json.dumps(job), encoding="utf-8")
    import os
    p = subprocess.run([sys.executable, "-X", "utf8", "-m", "ndforge_worker", str(f)], cwd=WORKER, capture_output=True, text=True, timeout=600, env={**os.environ, **(env or {})})
    return p, [json.loads(line) for line in p.stdout.splitlines() if line.startswith("{")]


def test_qwen_template_answers_without_thinking_by_default_after_export():
    from ndforge_worker.families import default_non_thinking
    from tinymodel import DATA
    tpl = (DATA / "qwen3_5_chat_template.jinja").read_text(encoding="utf-8")
    patched = default_non_thinking(tpl)
    assert patched != tpl and default_non_thinking(patched) == patched          # changed once, then stable
    from jinja2 import Environment
    env = Environment()
    render = lambda t, **kw: env.from_string(t).render(messages=[{"role": "user", "content": "hi"}], add_generation_prompt=True, **kw)  # noqa: E731
    assert render(tpl).endswith("<|im_start|>assistant\n<think>\n")                                   # the original thinks by default
    assert render(patched).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")                 # the exported one does not
    assert render(patched, enable_thinking=True).endswith("<|im_start|>assistant\n<think>\n")         # unless asked
    assert render(patched, enable_thinking=False).endswith("<think>\n\n</think>\n\n")


@pytest.mark.parametrize("family", ["qwen3_5", "gemma4"])
def test_export_merges_the_adapter_into_the_model_without_changing_its_answers(tmp_path, family):
    p, ev, run_out = run_job(tmp_path, family, "real")
    assert p.returncode == 0, p.stderr[-2000:]
    p2, ev2 = run_export(tmp_path, family, run_out)
    assert p2.returncode == 0, p2.stderr[-3000:]
    res = ev2[-1]["data"]
    merged = Path(res["merged_dir"])
    assert res["gguf"] is None and (merged / "config.json").exists() and list(merged.glob("*.safetensors"))
    # the merged model gives the same logits as base + adapter
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(str(tmp_path / "model"))
    peft_model = PeftModel.from_pretrained(base, str(run_out / "adapter")).eval()
    merged_model = AutoModelForCausalLM.from_pretrained(str(merged)).eval()
    ids = torch.randint(0, 100, (1, 12))
    with torch.no_grad():
        assert torch.allclose(peft_model(input_ids=ids).logits, merged_model(input_ids=ids).logits, atol=1e-4)
    if family == "qwen3_5":   # the template shipped with the merged model does not think by default
        tpl = (merged / "chat_template.jinja").read_text(encoding="utf-8") if (merged / "chat_template.jinja").exists() else json.loads((merged / "tokenizer_config.json").read_text())["chat_template"]
        assert "enable_thinking is not defined or enable_thinking is false" in tpl


def _fake_llama_cpp(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "convert_hf_to_gguf.py").write_text(
        "import sys\nargs=sys.argv\nout=args[args.index('--outfile')+1]\nassert args[args.index('--outtype')+1]=='f16'\nopen(out,'wb').write(b'GGUF-f16 '+open(args[1]+'/config.json','rb').read()[:20])\nprint('converted')\n")
    q = root / "llama-quantize"
    q.write_text("#!/bin/sh\necho \"quantizing $1 to $3\"\nprintf 'GGUF-%s' \"$3\" > \"$2\"\n")
    q.chmod(0o755)


@pytest.mark.skipif(sys.platform == "win32", reason="the fake llama-quantize is a shell script")
def test_gguf_is_made_with_llama_cpp_and_the_big_files_are_cleaned_up(tmp_path):
    p, ev, run_out = run_job(tmp_path, "qwen3_5")
    assert p.returncode == 0
    _fake_llama_cpp(tmp_path / "lcpp")
    p2, ev2 = run_export(tmp_path, "qwen3_5", run_out, env={"LLAMA_CPP_DIR": str(tmp_path / "lcpp")}, gguf=True, quant="q5_k_m")
    assert p2.returncode == 0, p2.stderr[-2000:] + p2.stdout[-1000:]
    res = ev2[-1]["data"]
    out = tmp_path / "exp"
    assert Path(res["gguf"]) == out / "model-q5_k_m.gguf" and (out / "model-q5_k_m.gguf").read_bytes() == b"GGUF-Q5_K_M"
    assert not (out / "merged").exists() and not (out / "model-f16.gguf").exists() and res["merged_dir"] is None
    assert any(e["type"] == "log" and "quantizing" in e["text"] for e in ev2)
    # f16 is the converter's own output, renamed; keep_merged keeps the merged model
    p3, ev3 = run_export(tmp_path, "qwen3_5", run_out, env={"LLAMA_CPP_DIR": str(tmp_path / "lcpp")}, gguf=True, quant="f16", keep_merged=True)
    assert p3.returncode == 0 and (out / "model-f16.gguf").read_bytes().startswith(b"GGUF-f16") and (out / "merged").exists()


def test_without_llama_cpp_the_error_says_what_to_do(tmp_path):
    p, ev, run_out = run_job(tmp_path, "qwen3_5")
    p2, ev2 = run_export(tmp_path, "qwen3_5", run_out, env={"LLAMA_CPP_DIR": str(tmp_path / "nothing"), "HOME": str(tmp_path / "home")}, gguf=True)
    assert p2.returncode == 1 and ev2[-1]["code"] == "export_failed" and "llama.cpp is needed" in ev2[-1]["message"]
    p3, ev3 = run_export(tmp_path, "qwen3_5", run_out, quant="q2")
    assert p3.returncode == 1 and "Unknown quantisation" in ev3[-1]["message"]


def test_warm_start_continues_from_an_earlier_adapter(tmp_path):
    p, ev, run_out = run_job(tmp_path, "qwen3_5", hyper={"max_steps": 25, "lr": 2e-2, "warmup": 0})
    assert p.returncode == 0, p.stderr[-2000:]
    first = [e for e in ev if e["type"] == "metric"]
    cold_start_loss, last_loss = first[0]["loss"], first[-1]["loss"]
    assert cold_start_loss - last_loss > 0.3, (cold_start_loss, last_loss)                       # the first run really learned something
    # a second run that starts from that adapter (and barely moves: tiny learning rate) begins where the first one ended
    p2, ev2, _ = run_job(tmp_path, "qwen3_5", init_adapter=str(run_out / "adapter"), out=str(tmp_path / "run2"), hyper={"max_steps": 2, "lr": 1e-7, "warmup": 0})
    assert p2.returncode == 0, p2.stderr[-2000:]
    warm = [e for e in ev2 if e["type"] == "metric"]
    assert abs(warm[0]["loss"] - last_loss) < 0.5 * (cold_start_loss - last_loss), (warm[0]["loss"], last_loss, cold_start_loss)
    # without init_adapter the same settings start from scratch
    p3, ev3, _ = run_job(tmp_path, "qwen3_5", out=str(tmp_path / "run3"), hyper={"max_steps": 2, "lr": 1e-7, "warmup": 0})
    assert [e for e in ev3 if e["type"] == "metric"][0]["loss"] > last_loss + 0.5 * (cold_start_loss - last_loss)


# --------------------------------------------------------------------------------------------- full fine-tuning and two stages

def _params(model_dir):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(model_dir))
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


@pytest.mark.parametrize("family", ["qwen3_5", "gemma4"])
def test_full_fine_tuning_changes_the_weights_and_saves_the_whole_model(tmp_path, family):
    p, ev, out = run_job(tmp_path, family, "real", method="full", hyper={"max_steps": 8, "lr": 1e-2, "warmup": 0})
    assert p.returncode == 0, p.stderr[-3000:]
    res = ev[-1]["data"]
    assert res["adapter_dir"] is None and res["method"] == "full" and Path(res["model_dir"], "config.json").exists() and list(Path(res["model_dir"]).glob("*.safetensors"))
    assert not (out / "adapter").exists()
    metrics = [e for e in ev if e["type"] == "metric"]
    assert metrics[0]["loss"] - metrics[-1]["loss"] > 0.3, [m["loss"] for m in metrics]              # it learns (full steps are strong)
    before, after = _params(tmp_path / "model"), _params(res["model_dir"])
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert len(changed) > 0.8 * len(before), f"only {len(changed)} of {len(before)} tensors changed"   # every weight is trained, not a few
    assert any(e["type"] == "sample" for e in ev)


def test_full_fine_tuning_refuses_a_card_without_bf16_with_a_plain_message(tmp_path, monkeypatch):
    # simulated here by calling the loader directly with a fake fp16-only card
    from ndforge_worker import backends
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    cfg = {"method": "full", "model": {"repo": "x", "family": "qwen3_5"}, "hyper": {"rank": 4, "alpha": 4, "max_seq": 64, "seed": 1}}
    with pytest.raises(RuntimeError, match="needs a graphics card with bf16"):
        backends.load_hf(cfg)


STAGES = [{"name": "world_book", "filter": "text", "epochs": 3}, {"name": "practice", "filter": "chat", "epochs": 2}]


def test_two_stages_read_the_world_book_first_then_practise_with_continuous_steps(tmp_path):
    p, ev, out = run_job(tmp_path, "qwen3_5", stages=STAGES, hyper={"max_steps": None, "batch": 2, "grad_accum": 1, "lr": 5e-3, "save_steps": 100})
    assert p.returncode == 0, p.stderr[-3000:]
    starts = [e for e in ev if e["type"] == "start"]
    assert [(s["stage"], s["stage_index"], s["stage_count"]) for s in starts] == [("world_book", 0, 2), ("practice", 1, 2)]
    # 4 texts x 3 epochs / batch 2 = 6 steps, then 12 conversations x 2 epochs / 2 = 12 steps; the numbering carries on across stages
    metrics = [e for e in ev if e["type"] == "metric"]
    assert [m["stage"] for m in metrics] == ["world_book"] * 6 + ["practice"] * 12
    assert [m["step"] for m in metrics] == list(range(1, 19))
    assert {m["total_steps"] for m in metrics} == {18}
    res = ev[-1]["data"]
    assert res["steps"] == 18 and [c["name"] for c in res["stages"]] == ["world_book", "practice"] and Path(res["adapter_dir"], "adapter_config.json").exists()
    # a stage with nothing to learn is skipped, not an error
    (tmp_path / "b").mkdir()
    p2, ev2, _ = run_job(tmp_path / "b", "qwen3_5", stages=[{"name": "world_book", "filter": "text"}, {"name": "practice", "filter": "chat"}],
                         dataset={"train": str(_only_chat(tmp_path))}, hyper={"max_steps": None, "batch": 2, "grad_accum": 1, "epochs": 1})
    assert p2.returncode == 0 and any(e["type"] == "log" and "has no examples" in e["text"] for e in ev2)


def _only_chat(tmp_path):
    f = tmp_path / "chat_only.jsonl"
    write_jsonl(f, sample_rows(8, 0))
    return f


def test_a_stopped_two_stage_run_continues_in_the_stage_it_was_in(tmp_path):
    # stop during stage 2 by letting a first run finish stage 1 and then die: simulated by running only stage 1 first with its state saved
    p, ev, out = run_job(tmp_path, "qwen3_5", stages=STAGES[:1], hyper={"max_steps": None, "batch": 2, "grad_accum": 1, "save_steps": 2})
    assert p.returncode == 0
    state = json.loads((out / "stage_state.json").read_text())
    assert state["completed"] == [{"name": "world_book", "steps": 6}]
    # continue with both stages: stage 1 is skipped, stage 2 runs and the step numbers continue after stage 1
    p2, ev2, _ = run_job(tmp_path, "qwen3_5", stages=STAGES, resume=True, hyper={"max_steps": None, "batch": 2, "grad_accum": 1, "save_steps": 2})
    assert p2.returncode == 0, p2.stderr[-2500:]
    assert any(e["type"] == "log" and 'Stage "world_book" was finished before' in e["text"] for e in ev2)
    starts = [e for e in ev2 if e["type"] == "start"]
    assert [s["stage"] for s in starts] == ["practice"]
    assert [e["step"] for e in ev2 if e["type"] == "metric"][0] == 7
    assert ev2[-1]["data"]["steps"] == 18 and [c["name"] for c in ev2[-1]["data"]["stages"]] == ["world_book", "practice"]


def test_a_full_model_continues_from_a_saved_full_model_and_is_exported_without_a_merge(tmp_path):
    p, ev, out = run_job(tmp_path, "qwen3_5", "real", method="full", hyper={"max_steps": 20, "lr": 1e-2, "warmup": 0})
    assert p.returncode == 0, p.stderr[-2000:]
    first = [e for e in ev if e["type"] == "metric"]
    cold, last = first[0]["loss"], first[-1]["loss"]
    assert cold - last > 0.3
    model_dir = ev[-1]["data"]["model_dir"]
    # warm start: the second run loads the first run's model, so it begins near where the first ended
    p2, ev2, _ = run_job(tmp_path, "qwen3_5", "real", method="full", init_model=model_dir, out=str(tmp_path / "run2"), hyper={"max_steps": 2, "lr": 1e-9, "warmup": 0})
    assert p2.returncode == 0, p2.stderr[-2000:]
    assert abs([e for e in ev2 if e["type"] == "metric"][0]["loss"] - last) < 0.5 * (cold - last)
    # export: nothing to merge; the template of the model folder is the non-thinking one; a GGUF comes from the model folder itself
    if sys.platform != "win32":
        _fake_llama_cpp(tmp_path / "lcpp")
    job = {"job": "export", "full_model": model_dir, "base": {"repo": str(tmp_path / "model"), "family": "qwen3_5"}, "out": str(tmp_path / "expfull"), "quant": "q4_k_m",
           "backend": "hf", "gguf": sys.platform != "win32"}
    f = tmp_path / "ej.json"
    f.write_text(json.dumps(job), encoding="utf-8")
    import os
    pe = subprocess.run([sys.executable, "-X", "utf8", "-m", "ndforge_worker", str(f)], cwd=WORKER, capture_output=True, text=True, timeout=600,
                        env={**os.environ, "LLAMA_CPP_DIR": str(tmp_path / "lcpp")})
    assert pe.returncode == 0, pe.stderr[-2000:]
    res = [json.loads(line) for line in pe.stdout.splitlines() if line.startswith("{")][-1]["data"]
    assert res["full_model"] is True and Path(model_dir, "config.json").exists()               # the trained model is kept
    tpl = Path(model_dir, "chat_template.jinja")
    text = tpl.read_text(encoding="utf-8") if tpl.exists() else json.loads(Path(model_dir, "tokenizer_config.json").read_text())["chat_template"]
    assert "enable_thinking is not defined or enable_thinking is false" in text
    if sys.platform != "win32":
        assert Path(res["gguf"]).read_bytes() == b"GGUF-Q4_K_M"
