"""The importance matrix step of the GGUF export, with stand-in llama.cpp tools (small Python scripts run through this interpreter: the same on Windows and
Linux) and a stand-in tokenizer. No torch, no llama.cpp, no network."""
import json
import os
import re
import threading
import time
from pathlib import Path

import pytest

from ndforge_worker import exportjob, imatrix, protocol

CONVERT = '''import sys
a = sys.argv
out = a[a.index("--outfile") + 1]
open(out, "wb").write(b"GGUF-f16")
print("converted")
'''

QUANTIZE = '''import json, os, sys
a = sys.argv[1:]
open(os.environ["STUB_LOG"], "a").write(json.dumps({"tool": "quantize", "args": a}) + "\\n")
mat = None
if "--imatrix" in a:
    i = a.index("--imatrix")
    mat = a[i + 1]
    a = a[:i] + a[i + 2:]
    if os.environ.get("QUANTIZE_REFUSES_IMATRIX"):
        print("error: invalid argument: --imatrix")
        sys.exit(1)
src, dst, quant = a[0], a[1], a[2]
open(dst, "wb").write(("GGUF-" + quant + ("+imatrix" if mat else "")).encode())
print("quantizing", quant)
'''

IMATRIX = '''import json, os, sys, time
a = sys.argv[1:]
if a == ["--version"]:
    print("version: 9999 (stub1234)")
    sys.exit(0)
log = os.environ["STUB_LOG"]
if a == ["--help"]:
    open(log, "a").write(json.dumps({"tool": "imatrix-help", "args": a}) + "\\n")
    print("usage: llama-imatrix [options]\\n  --chunks N\\n" + ("" if os.environ.get("NEXTN_MODE") == "unsupported" else "  --nextn   collects data for MTP/NextN layers if present\\n"))
    sys.exit(0)
calib = a[a.index("-f") + 1]
open(log, "a").write(json.dumps({"tool": "imatrix", "args": a, "calibration": open(calib, encoding="utf-8").read()}) + "\\n")
mode = os.environ.get("IMATRIX_MODE", "")
if "--nextn" in a and os.environ.get("NEXTN_MODE") == "refuse":
    print("error: --nextn cannot be used on models where NextN layers share the trunk's KV cache")
    sys.exit(1)
if "--nextn" in a and os.environ.get("NEXTN_MODE") == "other":
    print("CUDA out of memory")
    sys.exit(2)
if mode == "old" and "--output-format" in a:
    print("error: invalid argument: --output-format")
    sys.exit(1)
if mode == "fail":
    print("llama_model_load: error loading model")
    sys.exit(3)
if mode == "hang":
    open(os.environ["STUB_STARTED"], "w").write("x")
    for _ in range(600):          # at most about 30 seconds: the test ends it long before
        time.sleep(0.05)
    open(os.environ["STUB_STARTED"] + ".finished", "w").write("x")
    sys.exit(0)
n = int(a[a.index("--chunks") + 1])
for i in range(1, n + 1):
    print("[%d]1.0," % i)
open(a[a.index("-o") + 1], "wb").write(b"IMATRIX-FILE")
'''


class StubTok:
    """one token per whitespace-separated word; ChatML chat template"""
    eos_token = "<|im_end|>"

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(text.split())))}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + ("<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if kw.get("enable_thinking") is False else "") if add_generation_prompt else "")


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def project_rows(n_chat=12, n_text=6, words=40):
    rows = [{"type": "chat", "messages": [{"role": "system", "content": "You know the world."}, {"role": "user", "content": f"Who is hero {i}?"},
                                         {"role": "assistant", "content": " ".join(f"fact{i}_{j}" for j in range(words))}]} for i in range(n_chat)]
    rows += [{"type": "text", "text": " ".join(f"lore{i}_{j}" for j in range(words * 2))} for i in range(n_text)]
    return rows


def general_file(tmp_path, n=30, words=60):
    return write_rows(tmp_path / "general.jsonl", [{"text": " ".join(f"gen{i}_{j}" for j in range(words))} for i in range(n)])


@pytest.fixture
def world(tmp_path, monkeypatch):
    """stand-in llama.cpp, a merged model folder, the project's training file and a job config; every tool call is logged in STUB_LOG"""
    lcpp = tmp_path / "lcpp"
    lcpp.mkdir()
    for name, code in (("convert_hf_to_gguf.py", CONVERT), ("llama-quantize.py", QUANTIZE), ("llama-imatrix.py", IMATRIX)):
        (lcpp / name).write_text(code, encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("STUB_LOG", str(log))
    monkeypatch.setenv("STUB_STARTED", str(tmp_path / "started"))
    for var in ("IMATRIX_MODE", "QUANTIZE_REFUSES_IMATRIX", "NEXTN_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: (lcpp / "convert_hf_to_gguf.py", lcpp / "llama-quantize.py"))
    monkeypatch.setattr(imatrix, "find_llama_imatrix", lambda q: lcpp / "llama-imatrix.py")
    monkeypatch.setattr(imatrix, "load_tokenizer", lambda sources: StubTok())
    events = []
    monkeypatch.setattr(protocol, "emit", lambda kind, **f: events.append({"type": kind, **f}))
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    train = write_rows(tmp_path / "train.jsonl", project_rows())

    class W:
        pass

    w = W()
    w.root, w.log, w.events, w.merged, w.train, w.lcpp = tmp_path, log, events, merged, train, lcpp
    w.general = general_file(tmp_path)

    def cfg(**im):
        base = {"enabled": True, "model": "run1", "data_version": 3, "train": train, "chunks": 64, "ctx": 128, "general_share": 0.2, "reuse": True, "reuse_dirs": [],
                "general": {"file": w.general}, "stop_file": str(tmp_path / "STOP")}
        base.update(im)
        return {"adapter": str(tmp_path / "adapter"), "base": {"repo": "org/base", "family": "qwen3_5"}, "imatrix": base}

    def calls(tool=None):
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        return [r for r in rows if tool is None or r["tool"] == tool]

    def export(quants, out=None, report=None, **im):
        out = out or (tmp_path / "out")
        out.mkdir(exist_ok=True)
        rep = exportjob.new_report() if report is None else report
        files = exportjob.gguf_with_llama_cpp(cfg(**im), w.merged, out, quants, rep)
        return files, rep, out

    w.cfg, w.calls, w.export = cfg, calls, export
    return w


def flag(args, name, n=1):
    i = args.index(name)
    return args[i + 1] if n == 1 else args[i + 1:i + 1 + n]


# ------------------------------------------------------------------------------------------------------------------ the command lines

def test_the_matrix_is_made_once_and_given_only_to_the_sizes_below_q8(world):
    files, rep, out = world.export(["q4_k_m", "q6_k", "q8_0", "f16"])
    (im,) = world.calls("imatrix")
    a = im["args"]
    assert flag(a, "-m") == str(out / "model-f16.gguf") and flag(a, "-f").endswith("imatrix-calibration.txt")
    assert Path(flag(a, "-o")).name == "imatrix.partial.gguf"
    assert flag(a, "--output-format") == "gguf" and "--no-ppl" in a and "--parse-special" in a
    assert flag(a, "--chunks") == "64" and flag(a, "-c") == "128" and flag(a, "-ngl") == "99" and "--process-output" not in a
    q = {c["args"][-1]: c["args"] for c in world.calls("quantize")}
    assert set(q) == {"Q4_K_M", "Q6_K", "Q8_0"}                       # f16 is the converter's own file, not quantised
    for size in ("Q4_K_M", "Q6_K"):
        assert q[size][0] == "--imatrix" and q[size][1] == str(out / "imatrix.gguf") and q[size][2:4] == [str(out / "model-f16.gguf"), str(out / f"model-{size.lower()}.gguf")]
    assert "--imatrix" not in q["Q8_0"]
    assert files["q4_k_m"].read_bytes() == b"GGUF-Q4_K_M+imatrix" and files["q8_0"].read_bytes() == b"GGUF-Q8_0" and files["f16"].read_bytes() == b"GGUF-f16"
    assert (out / "imatrix.gguf").read_bytes() == b"IMATRIX-FILE" and not (out / "imatrix.partial.gguf").exists() and not (out / "imatrix-calibration.txt").exists()
    assert rep["made"] and not rep["reused"] and rep["applied"] == ["q4_k_m", "q6_k"] and list(rep["not_applied"]) == ["q8_0"] and rep["notes"] == []
    meta = json.loads((out / "imatrix.json").read_text(encoding="utf-8"))
    assert (meta["model"], meta["data_version"], meta["chunks"], meta["ctx"], meta["general_share"], meta["general_used"]) == ("run1", 3, 64, 128, 0.2, True)
    assert meta["data_hash"] == imatrix.file_sha256(world.train) and meta["llama_cpp"] == "version: 9999 (stub1234)" and meta["base_repo"] == "org/base"
    assert meta["project_tokens"] > 0 and meta["general_tokens"] > 0


def test_the_progress_is_the_imatrix_phase_and_the_tool_output_goes_to_the_log(world):
    world.export(["q4_k_m"])
    prog = [e for e in world.events if e["type"] == "progress" and e["step"] == "imatrix"]
    assert prog[0]["value"] == 0.0 and prog[-1]["value"] == 1.0
    chunk = [e for e in prog if e["text"].startswith("Chunk ")]
    assert chunk and chunk[-1]["text"] == "Chunk 64 of 64" and all(a["value"] <= b["value"] for a, b in zip(prog, prog[1:]))
    logs = [e["text"] for e in world.events if e["type"] == "log"]
    assert any(t.startswith("$ ") and "--parse-special" in t for t in logs) and "[64]1.0," in logs


@pytest.mark.parametrize("quants", [["q8_0"], ["f16"], ["q8_0", "f16"]])
def test_without_a_size_below_q8_nothing_is_measured(world, quants):
    files, rep, out = world.export(quants)
    assert world.calls("imatrix") == [] and not (out / "imatrix.gguf").exists() and not rep["made"] and rep["applied"] == []
    assert "below Q8_0" in " ".join(rep["notes"])
    assert all("--imatrix" not in c["args"] for c in world.calls("quantize")) and set(files) == set(quants)


def test_the_tools_are_found_next_to_each_other(tmp_path, monkeypatch):
    exe = "llama-imatrix.exe" if os.name == "nt" else "llama-imatrix"
    quant = tmp_path / "build" / "bin" / ("llama-quantize.exe" if os.name == "nt" else "llama-quantize")
    quant.parent.mkdir(parents=True)
    quant.write_text("x")
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    assert imatrix.find_llama_imatrix(quant) is None                  # an older llama.cpp without it
    (quant.parent / exe).write_text("x")
    assert imatrix.find_llama_imatrix(quant) == quant.parent / exe
    assert imatrix.launcher("tool.py")[1:] == ["tool.py"] and imatrix.launcher("llama-imatrix") == ["llama-imatrix"]


# ------------------------------------------------------------------------------------------------------------------ reuse

def test_a_later_export_of_the_same_model_and_data_reuses_the_matrix(world):
    _, rep1, out1 = world.export(["q4_k_m"])
    assert rep1["made"] and len(world.calls("imatrix")) == 1
    out2 = world.root / "out2"
    files, rep2, _ = world.export(["q4_k_m", "q5_k_m"], out=out2, reuse_dirs=[str(out1)])
    assert len(world.calls("imatrix")) == 1, "not measured again"
    assert rep2["reused"] and not rep2["made"] and rep2["applied"] == ["q4_k_m", "q5_k_m"] and (out2 / "imatrix.gguf").read_bytes() == b"IMATRIX-FILE" and (out2 / "imatrix.json").is_file()
    # in place (the same folder): nothing is copied onto itself
    _, rep3, _ = world.export(["q4_k_m"], out=out1, reuse_dirs=[str(out1)])
    assert rep3["reused"] and (out1 / "imatrix.gguf").read_bytes() == b"IMATRIX-FILE"


def test_a_saved_matrix_is_not_reused_when_it_no_longer_fits(world, tmp_path):
    _, _, out1 = world.export(["q4_k_m"])
    n = lambda: len(world.calls("imatrix"))                                          # noqa: E731
    assert n() == 1
    world.export(["q4_k_m"], out=world.root / "o_off", reuse_dirs=[str(out1)], reuse=False)              # the box is unticked
    assert n() == 2
    world.export(["q4_k_m"], out=world.root / "o_chunks", reuse_dirs=[str(out1)], chunks=128)
    world.export(["q4_k_m"], out=world.root / "o_ctx", reuse_dirs=[str(out1)], ctx=256)
    world.export(["q4_k_m"], out=world.root / "o_model", reuse_dirs=[str(out1)], model="run2")
    assert n() == 5
    rows = project_rows() + [{"type": "text", "text": "one more text about the world " * 5}]                  # the data changed
    world.export(["q4_k_m"], out=world.root / "o_data", reuse_dirs=[str(out1)], train=write_rows(tmp_path / "train2.jsonl", rows))
    assert n() == 6
    adapter = tmp_path / "adapter"                                                                           # the same run trained further
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"W")
    world.export(["q4_k_m"], out=world.root / "o_weights", reuse_dirs=[str(out1)])
    assert n() == 7
    _, rep, _ = world.export(["q4_k_m"], out=world.root / "o_same", reuse_dirs=[str(world.root / "o_weights"), str(out1)])
    assert rep["reused"] and n() == 7


def test_a_matrix_made_without_general_text_is_not_reused_when_general_text_is_wanted(world):
    _, rep1, out1 = world.export(["q4_k_m"], general={"file": str(world.root / "missing.jsonl")})
    assert rep1["made"] and rep1["info"]["general_used"] is False
    _, rep2, _ = world.export(["q4_k_m"], out=world.root / "o2", reuse_dirs=[str(out1)])
    assert rep2["made"] and not rep2["reused"] and rep2["info"]["general_used"] is True
    _, rep3, _ = world.export(["q4_k_m"], out=world.root / "o3", reuse_dirs=[str(out1)], general_share=0.0, general=None)       # nothing wanted: the old one fits
    assert rep3["made"]                                                                                      # a different share is a different matrix


# ------------------------------------------------------------------------------------------------------------------ never fatal

def test_without_llama_imatrix_the_sizes_are_made_normally_and_it_is_said(world, monkeypatch):
    monkeypatch.setattr(imatrix, "find_llama_imatrix", lambda q: None)
    files, rep, out = world.export(["q4_k_m", "q8_0"])
    assert world.calls("imatrix") == [] and files["q4_k_m"].read_bytes() == b"GGUF-Q4_K_M" and not rep["made"] and rep["applied"] == []
    assert "no llama-imatrix" in rep["notes"][0] and "older" in rep["notes"][0]


def test_a_failing_llama_imatrix_becomes_a_note(world, monkeypatch):
    monkeypatch.setenv("IMATRIX_MODE", "fail")
    files, rep, out = world.export(["q4_k_m"])
    assert files["q4_k_m"].read_bytes() == b"GGUF-Q4_K_M" and not rep["made"] and "exit code 3" in rep["notes"][0] and "without the importance matrix" in rep["notes"][0]
    assert not (out / "imatrix.gguf").exists() and not (out / "imatrix.partial.gguf").exists() and not (out / "imatrix-calibration.txt").exists()
    assert all("--imatrix" not in c["args"] for c in world.calls("quantize"))
    assert any(e["type"] == "log" and "error loading model" in e["text"] for e in world.events)          # the output tail is in the log


def test_an_older_llama_imatrix_without_output_format_is_tried_again_without_it(world, monkeypatch):
    monkeypatch.setenv("IMATRIX_MODE", "old")
    files, rep, out = world.export(["q4_k_m"])
    first, second = world.calls("imatrix")
    assert "--output-format" in first["args"] and "--output-format" not in second["args"]
    assert rep["made"] and rep["applied"] == ["q4_k_m"] and files["q4_k_m"].read_bytes().endswith(b"+imatrix")


def test_a_quantiser_that_refuses_the_matrix_makes_that_size_without_it(world, monkeypatch):
    monkeypatch.setenv("QUANTIZE_REFUSES_IMATRIX", "1")
    files, rep, out = world.export(["q4_k_m", "q6_k"])
    assert files["q4_k_m"].read_bytes() == b"GGUF-Q4_K_M" and files["q6_k"].read_bytes() == b"GGUF-Q6_K"
    assert rep["made"] and rep["applied"] == [] and set(rep["not_applied"]) == {"q4_k_m", "q6_k"} and "did not accept" in rep["notes"][0]


def test_offline_general_text_falls_back_to_the_projects_text_with_a_note(world):
    files, rep, out = world.export(["q4_k_m"], general={"file": str(world.root / "missing.jsonl")})
    (im,) = world.calls("imatrix")
    assert rep["made"] and rep["info"]["general_used"] is False and rep["info"]["general_tokens"] == 0
    assert "general text" in rep["notes"][0] and "your project's text only" in rep["notes"][0]
    assert "gen0_" not in im["calibration"] and "fact0_0" in im["calibration"]
    assert any(e["type"] == "log" and "your project's text only" in e["text"] for e in world.events)


def test_a_dataset_that_cannot_be_downloaded_is_the_same_fallback(world, monkeypatch):
    from ndforge_worker import distilljob

    def offline(source):
        raise OSError("Network is unreachable")
        yield   # pragma: no cover

    monkeypatch.setattr(distilljob, "_general_texts", offline)
    _, rep, _ = world.export(["q4_k_m"], general={"id": "some/dataset"})
    assert rep["made"] and rep["info"]["general_used"] is False and "Network is unreachable" in rep["notes"][0] and "some/dataset" in rep["notes"][0]


def test_the_default_general_text_is_the_healing_default(world, monkeypatch):
    from ndforge_worker import distilljob
    seen = []
    monkeypatch.setattr(distilljob, "_general_texts", lambda source: (seen.append(source), iter(["word " * 50] * 400))[1])
    _, rep, _ = world.export(["q4_k_m"], general=None)
    assert seen[0]["id"] == distilljob.DEFAULT_GENERAL and rep["info"]["general_used"] and rep["info"]["general_source"] == "default"


@pytest.mark.parametrize("what", ["no_train", "no_tokenizer", "tokenizer_crash"])
def test_other_problems_are_notes_too(world, monkeypatch, what):
    if what == "no_train":
        files, rep, _ = world.export(["q4_k_m"], train=str(world.root / "gone.jsonl"))
        assert "training data of this run is missing" in rep["notes"][0]
    else:
        def broken(sources):
            raise OSError("no tokenizer files") if what == "no_tokenizer" else RuntimeError("boom")
        monkeypatch.setattr(imatrix, "load_tokenizer", broken)
        files, rep, _ = world.export(["q4_k_m"])
        assert "could not be made" in rep["notes"][0] and ("no tokenizer files" in rep["notes"][0] or "boom" in rep["notes"][0])
    assert files["q4_k_m"].read_bytes() == b"GGUF-Q4_K_M" and world.calls("imatrix") == []


def test_without_the_imatrix_block_nothing_changes(world):
    out = world.root / "plain"
    out.mkdir()
    cfg = world.cfg()
    del cfg["imatrix"]
    files = exportjob.gguf_with_llama_cpp(cfg, world.merged, out, ["q4_k_m"])
    assert files["q4_k_m"].read_bytes() == b"GGUF-Q4_K_M" and world.calls("imatrix") == []
    assert world.calls("quantize")[0]["args"][0] == str(out / "model-f16.gguf")


# ------------------------------------------------------------------------------------------------------------------ the stop file

def _stop_when_started(world, delay=0.0):
    def waiter():
        for _ in range(400):
            if (world.root / "started").exists():
                break
            time.sleep(0.05)
        time.sleep(delay)
        (world.root / "STOP").write_text("")
    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    return t


def test_the_stop_file_kills_llama_imatrix_and_ends_the_export(world, monkeypatch):
    monkeypatch.setenv("IMATRIX_MODE", "hang")
    t = _stop_when_started(world)
    t0 = time.time()
    with pytest.raises(imatrix.Stopped):
        world.export(["q4_k_m"])
    t.join()
    assert time.time() - t0 < 15, "the tool (it would run about 30 s) was killed, not waited for"
    assert (world.root / "started").exists() and not (world.root / "started.finished").exists()
    out = world.root / "out"
    assert not (out / "model-f16.gguf").exists() and not (out / "imatrix.partial.gguf").exists() and not (out / "imatrix.gguf").exists() and not (out / "model-q4_k_m.gguf").exists()
    assert world.calls("quantize") == []


def test_the_stop_file_also_stops_between_sizes(world):
    (world.root / "STOP").write_text("")
    with pytest.raises(imatrix.Stopped):
        world.export(["q8_0", "q4_k_m"], general_share=0.2, reuse_dirs=[], reuse=False)
    assert world.calls("quantize") == []


def test_the_worker_reports_a_stop_as_cancelled(world, monkeypatch, capsys):
    from ndforge_worker import __main__ as entry
    monkeypatch.setattr(exportjob, "run", lambda job: (_ for _ in ()).throw(imatrix.Stopped()))
    job = world.root / "job.json"
    job.write_text(json.dumps({"job": "export"}))
    assert entry.main(["x", str(job)]) == 1
    assert [e for e in world.events if e["type"] == "error"] == [{"type": "error", "code": "export_stopped", "message": "Cancelled."}]


# ------------------------------------------------------------------------------------------------------------------ the calibration text

def test_the_calibration_text_uses_the_chat_template_and_keeps_special_tokens():
    rows = project_rows(n_chat=3, n_text=2, words=10)
    text, stats = imatrix.build_calibration(StubTok(), rows, "qwen3_5", ctx=64, chunks=4, general=[])
    # a practice conversation: the model's own template (thinking off for Qwen3.5), the answer and the end marker
    assert "<|im_start|>system\nYou know the world.<|im_end|>\n<|im_start|>user\nWho is hero 0?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nfact0_0" in text
    assert text.count("<|im_start|>assistant") >= 3 and "fact0_9<|im_end|>\n" in text
    # a world-book text as it is, with no template around it
    assert "lore0_0 lore0_1" in text and "<|im_start|>lore" not in text
    assert stats["general_rows"] == 0 and stats["tokens"] >= 64 * 4 and stats["project_rows"] == 5


def test_a_gemma_row_ends_with_the_gemma_marker():
    rows = [{"type": "chat", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}]}]
    text, _ = imatrix.build_calibration(StubTok(), rows, "gemma4", ctx=8, chunks=2, general=[])
    assert "hello there<turn|>" in text


def test_about_a_fifth_is_general_text_mixed_through_the_file():
    tok = StubTok()
    general = [" ".join(f"gen{i}_{j}" for j in range(50)) for i in range(60)]
    text, stats = imatrix.build_calibration(tok, project_rows(n_chat=40, n_text=20, words=30), "qwen3_5", ctx=128, chunks=64, general=general)
    share = stats["general_tokens"] / stats["tokens"]
    assert 0.15 <= share <= 0.25, share
    assert stats["tokens"] >= 128 * 64 and stats["wanted_tokens"] == 128 * 64
    words = text.split()
    gen_pos = [i for i, w in enumerate(words) if w.startswith("gen")]
    assert gen_pos and gen_pos[0] < len(words) * 0.3 and gen_pos[-1] > len(words) * 0.7, "general text is spread over the whole file, not one block"
    assert imatrix.count_tokens(tok, text) >= 128 * 64


def test_the_same_data_always_gives_the_same_text_and_other_data_another():
    rows, general = project_rows(), [f"general text number {i} " * 20 for i in range(20)]
    a, _ = imatrix.build_calibration(StubTok(), rows, "qwen3_5", 64, 8, general)
    b, _ = imatrix.build_calibration(StubTok(), list(rows), "qwen3_5", 64, 8, list(general))
    c, _ = imatrix.build_calibration(StubTok(), rows[:-1], "qwen3_5", 64, 8, general)
    d, _ = imatrix.build_calibration(StubTok(), rows, "qwen3_5", 64, 8, general, seed=1)
    assert a == b and a != c and a != d


def test_small_data_is_repeated_to_fill_the_chunks_and_odd_rows_are_skipped():
    rows = [{"type": "text", "text": "tiny world book " * 5}, {"type": "chat", "messages": [{"role": "user", "content": "no answer yet"}]}, {"type": "text", "text": "  "}]
    text, stats = imatrix.build_calibration(StubTok(), rows, "generic", ctx=32, chunks=4, general=[])
    assert stats["repeated"] and stats["project_rows"] == 1 and stats["skipped_rows"] == 2 and stats["tokens"] >= 32 * 4
    with pytest.raises(ValueError, match="no text"):
        imatrix.build_calibration(StubTok(), rows[1:], "generic", 32, 4, [])


def test_the_text_file_reaches_llama_imatrix_unchanged(world):
    world.export(["q4_k_m"])
    (im,) = world.calls("imatrix")
    assert "<|im_start|>assistant\n<think>" in im["calibration"] and "<|im_end|>" in im["calibration"] and "gen0_0" in im["calibration"] and "lore0_0" in im["calibration"]
    assert imatrix.count_tokens(StubTok(), im["calibration"]) >= 64 * 128


# ------------------------------------------------------------------------------------------------------------------ the whole job

def test_which_backend_makes_the_gguf(world, monkeypatch):
    cfg = world.cfg()
    rep = exportjob.new_report()
    assert exportjob.use_unsloth_for("hf", cfg, ["q4_k_m"], rep) is False
    assert exportjob.use_unsloth_for("unsloth", cfg, ["q4_k_m"], rep) is False and rep["notes"] == []        # llama.cpp has everything: the matrix needs it
    assert exportjob.use_unsloth_for("unsloth", cfg, ["q4_k_m"], None) is True                                # no matrix wanted: Unsloth as before
    assert exportjob.use_unsloth_for("unsloth", {**cfg, "gguf": False}, ["q4_k_m"], rep) is False
    rep = exportjob.new_report()
    assert exportjob.use_unsloth_for("unsloth", cfg, ["q8_0", "f16"], rep) is True and "below Q8_0" in rep["notes"][0]
    monkeypatch.setattr(imatrix, "find_llama_imatrix", lambda q: None)
    rep = exportjob.new_report()
    assert exportjob.use_unsloth_for("unsloth", cfg, ["q4_k_m"], rep) is True and "through Unsloth without it" in rep["notes"][0]
    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: (None, None))
    rep = exportjob.new_report()
    assert exportjob.use_unsloth_for("unsloth", cfg, ["q4_k_m"], rep) is True and "llama.cpp (with llama-imatrix) was not found" in rep["notes"][0]


def test_the_whole_export_job_returns_the_imatrix_report(world, monkeypatch):
    monkeypatch.setattr(protocol, "reserve_stdout", lambda: None)
    monkeypatch.setattr(exportjob.backends, "prepare", lambda b: "hf")
    monkeypatch.setattr(exportjob, "merge_hf", lambda cfg, merged: merged.mkdir(parents=True, exist_ok=True))
    out = world.root / "job_out"
    cfg = {**world.cfg(), "out": str(out), "quants": ["q4_k_m", "q8_0"], "quant": "q4_k_m", "backend": "hf", "gguf": True, "vision": False}
    assert exportjob.run(cfg) == 0
    result = [e for e in world.events if e["type"] == "result"][0]["data"]
    im = result["imatrix"]
    assert (im["made"], im["applied"], im["note"], Path(im["file"]).name) == (True, ["q4_k_m"], "", "imatrix.gguf") and im["info"]["chunks"] == 64
    assert result["ggufs"].keys() == {"q4_k_m", "q8_0"} and json.loads((out / "export.json").read_text(encoding="utf-8"))["imatrix"]["made"] is True
    # not asked for: the result says null
    cfg2 = {k: v for k, v in cfg.items() if k != "imatrix"}
    cfg2["out"] = str(world.root / "job_out2")
    world.events.clear()
    assert exportjob.run(cfg2) == 0 and [e for e in world.events if e["type"] == "result"][0]["data"]["imatrix"] is None
    assert len(world.calls("imatrix")) == 1
