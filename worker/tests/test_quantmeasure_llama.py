"""The llama engine of the measured fit (quantmeasure.measure_llama / measure_any) against a stand-in llama-quantize that really quantizes with the numpy encoders
(tests/tools/llama_quantize_standin.py). This proves the plumbing, the read-back, the self-check, the stop and the fallbacks. It proves nothing about llama.cpp itself."""
import json
import os

from pathlib import Path

import numpy as np
import pytest
from ggufmake import tiny_model
from test_quantmeasure import make_matrix

from ndforge_worker import ggufrows, imatrix, llamacpp, quantmeasure as qm

STANDIN = Path(__file__).parent / "tools" / "llama_quantize_standin.py"
RUNGS = ("Q3_K", "Q4_K", "Q6_K")


@pytest.fixture
def world(tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    llamacpp._HELP_CACHE.clear()
    monkeypatch.setenv("STANDIN_LOG", str(log))
    for v in ("STANDIN_NO_TYPEFILE", "STANDIN_FLIP", "STANDIN_FAIL", "STANDIN_SLOW"):
        monkeypatch.delenv(v, raising=False)
    src = ggufrows.open_source(tiny_model(tmp_path / "model.gguf", rows=300, cols=512))
    matrix = make_matrix(tmp_path / "mx.gguf", src, np.random.default_rng(3))
    imp, _ = qm.importance(matrix, src)

    class W:
        pass
    w = W()
    w.src, w.matrix, w.imp, w.tmp, w.log = src, matrix, imp, tmp_path, log
    w.opts = {"sample_rows": 24, "seed": 1, "threads": 2, "rungs": RUNGS, "key": "k" * 16, "engine": "llama"}
    w.calls = lambda: [json.loads(x) for x in log.read_text().splitlines() if "--help" not in x and "--version" not in x] if log.exists() else []
    return w


def run(w, cache="cache", **kw):
    return qm.measure_any(w.src, w.imp, w.opts, w.tmp / cache, quantize=kw.pop("quantize", STANDIN), matrix=kw.pop("matrix", w.matrix), im=kw.pop("im", {}), **kw)


def test_the_llama_engine_gives_the_numpy_engines_numbers_with_the_standin_and_reads_back_the_actual_types(world):
    w = world
    res = run(w)
    assert res["engine"] == "llama" and res["done"] and res["selfcheck"]["passed"] and res["selfcheck"]["identical"] == 1.0
    ref = qm.measure(w.src, w.imp, {**w.opts, "key": "r" * 16}, w.tmp / "ref")
    assert set(res["tensors"]) == set(ref["tensors"])
    for name, rec in res["tensors"].items():
        for r in RUNGS:
            a, b = rec["types"][r], ref["tensors"][name]["types"][r]
            assert a["eff"] == b["eff"] and a["bytes"] == b["bytes"] and a["err"] == pytest.approx(b["err"], rel=1e-9) and a["err_plain"] == pytest.approx(b["err_plain"], rel=1e-9)
        assert rec["norm2_w"] == pytest.approx(ref["tensors"][name]["norm2_w"], rel=1e-9)
    assert res["tensors"]["blk.0.ssm_out.weight"]["types"]["Q3_K"]["eff"] == "Q4_0", "the type llama-quantize really gave (the fallback of a 320-column tensor) is the one recorded"
    sim = res["simulated"]["Q4_K_M"]
    assert sim["blk.0.attn_q.weight"]["eff"] == "Q4_K" and sim["output.weight" if "output.weight" in sim else "token_embd.weight"]["eff"] in ("Q4_K", "Q6_K")
    assert not (w.tmp / "cache" / "sample.gguf").exists(), "the sample file is deleted when the measurement is complete"
    assert json.loads((w.tmp / "cache" / "measure.json").read_text())["done"]
    # the same call again is served from the cache without running anything
    n = len(w.calls())
    assert run(w)["done"] and len(w.calls()) == n


def test_the_commands_the_tool_gets(world):
    w = world
    run(w)
    calls = w.calls()
    selfcheck, *rest = calls
    assert "--pure" in selfcheck and "--imatrix" not in selfcheck and selfcheck[-1] == "Q4_K_M"
    rung_calls = [c for c in rest if "--pure" in c]
    assert [c[-1] for c in rung_calls] == ["Q3_K_M", "Q4_K_M", "Q6_K"]
    for c in rung_calls:
        assert c[c.index("--imatrix") + 1] == str(w.matrix) and "--tensor-type-file" in c and c.index(str(w.tmp / "cache" / "sample.gguf")) > c.index("--tensor-type-file")
    plain = [c for c in rest if "--pure" not in c]
    assert [c[-1] for c in plain] == ["Q4_K_M", "Q5_K_M"] and all("--imatrix" in c and "--tensor-type-file" not in c for c in plain)
    tf = (w.tmp / "cache" / "rung-q3_k.tensor-types.txt").read_text().splitlines()
    assert tf[-1] == ".*=Q3_K" and any(l.endswith("=Q4_0") and "ssm_out" in l for l in tf)


def test_without_a_matrix_no_imatrix_flag_is_passed(world):
    w = world
    imp, _ = qm.importance(None, w.src)
    w.imp = imp
    qm.measure_any(w.src, imp, w.opts, w.tmp / "c", quantize=STANDIN, matrix=None, im={})
    assert all("--imatrix" not in c for c in w.calls())


def test_tensor_type_rules_as_arguments_when_the_tool_has_no_type_file(world, monkeypatch):
    monkeypatch.setenv("STANDIN_NO_TYPEFILE", "1")
    w = world
    assert not llamacpp.quantize_supports(STANDIN, "--tensor-type-file") and llamacpp.quantize_supports(STANDIN, "--pure")
    res = run(w)
    assert res["engine"] == "llama"
    rung = [c for c in w.calls() if "--pure" in c and c[-1] == "Q3_K_M"][0]
    assert "--tensor-type-file" not in rung and rung.count("--tensor-type") >= 2 and ".*=Q3_K" in rung


def test_the_type_file_is_used_when_the_help_lists_it(world):
    assert llamacpp.quantize_supports(STANDIN, "--tensor-type-file")
    run(world)
    assert (world.tmp / "cache" / "rung-q4_k.tensor-types.txt").is_file()


def test_the_command_length_guard(world, monkeypatch):
    monkeypatch.setenv("STANDIN_NO_TYPEFILE", "1")
    monkeypatch.setattr(qm, "COMMAND_LIMIT", 300)
    res = run(world)
    assert res["engine"] == "reference" and any("update llama.cpp" in n and "--tensor-type-file" in n for n in res["notes"])


def test_a_failed_self_check_falls_back_to_the_numpy_engine_with_a_note(world, monkeypatch):
    monkeypatch.setenv("STANDIN_FLIP", "1")
    res = run(world)
    assert res["engine"] == "reference" and res["done"]
    assert any("self-check failed" in n and "0.00 percent" in n for n in res["notes"]) and any("numpy engine measured instead" in n for n in res["notes"])
    ref = qm.measure(world.src, world.imp, {**world.opts, "key": "r" * 16}, world.tmp / "ref")
    assert json.dumps(res["tensors"], sort_keys=True) == json.dumps(ref["tensors"], sort_keys=True)


def test_a_tool_that_is_missing_or_exits_nonzero_falls_back_with_a_note(world, monkeypatch):
    res = run(world, cache="a", quantize=None)
    assert res["engine"] == "reference" and "llama-quantize was not found" in res["notes"][0]
    res = run(world, cache="b", quantize=world.tmp / "nope.py")
    assert res["engine"] == "reference"
    monkeypatch.setenv("STANDIN_FAIL", "1")
    res = run(world, cache="c")
    assert res["engine"] == "reference" and any("exited with code 1" in n and "standin told to fail" in n for n in res["notes"])


def test_the_reference_engine_asked_for_never_runs_the_tool(world):
    w = world
    w.opts["engine"] = "reference"
    res = run(w)
    assert res["engine"] == "reference" and w.calls() == []


def test_a_matrix_that_does_not_fit_makes_the_tool_fail_and_the_numpy_engine_takes_over(world):
    w = world
    bad = make_matrix(w.tmp / "bad.gguf", w.src, np.random.default_rng(1), bad="blk.1.attn_k.weight")
    res = run(w, matrix=bad)
    assert res["engine"] == "reference" and any("exited with code 1" in n for n in res["notes"])


def test_stop_between_runs_then_resume_runs_only_the_rest(world):
    w = world
    stop = w.tmp / "STOP"
    seen = []

    def progress(done, total, name):
        seen.append(name)
        if len(seen) == 2:
            stop.write_text("")
    with pytest.raises(qm.Stopped):
        run(w, im={"stop_file": str(stop)}, progress=progress)
    cache = json.loads((w.tmp / "cache" / "measure.json").read_text())
    assert not cache["done"] and set(cache["llama_runs"]) == {"Q3_K"} and cache["selfcheck"]["passed"]
    first = len(w.calls())
    assert first == 2                                              # the self-check and one rung
    stop.unlink()
    res = run(w, im={"stop_file": str(stop)})
    assert res["done"] and res["engine"] == "llama"
    later = w.calls()[first:]
    assert [c[-1] for c in later] == ["Q4_K_M", "Q6_K", "Q4_K_M", "Q5_K_M"] and not any(c[-1] == "Q3_K_M" for c in later)


def test_a_stop_kills_the_running_tool(world, monkeypatch):
    monkeypatch.setenv("STANDIN_SLOW", "30")
    w = world
    stop = w.tmp / "STOP"
    stop.write_text("")                                            # already there: the first run is killed at once
    import time
    t0 = time.time()
    with pytest.raises(qm.Stopped):
        run(w, im={"stop_file": str(stop)})
    assert time.time() - t0 < 20


def test_a_windows_style_exe_name_goes_through_the_launcher_unchanged(world, monkeypatch, tmp_path):
    assert imatrix.launcher("C:\\llama\\llama-quantize.exe") == ["C:\\llama\\llama-quantize.exe"]
    exe = tmp_path / "llama-quantize.exe"
    exe.write_text("not a program")
    seen = []

    def fake(cmd, im, chunks, label=None, **kw):
        seen.append(cmd)
        return 1, ["boom"]
    monkeypatch.setattr(imatrix, "run_streaming", fake)
    res = run(world, quantize=exe)
    assert seen and seen[0][0] == str(exe) and res["engine"] == "reference"


def test_not_enough_disk_space_falls_back(world, monkeypatch):
    monkeypatch.setattr(qm.fsutil, "free_bytes", lambda p: 10)
    res = run(world)
    assert res["engine"] == "reference" and any("free disk space" in n for n in res["notes"])


def test_the_sample_file_is_smaller_than_the_model(world):
    w = world
    picks = ggufrows.sample_plan(w.src, 24, 1)
    info = ggufrows.write_sample(w.src, w.tmp / "s.gguf", picks)
    assert info["bytes"] < w.src["size"]


def test_help_is_read_once_per_program_and_modification_time(tmp_path, monkeypatch):
    tool = tmp_path / "llama-quantize.py"
    tool.write_text("import sys\nprint('usage: x [--tensor-type-file]\\n  --tensor-type-file F: rules', file=sys.stderr)\nsys.exit(1)\n", encoding="utf-8")
    llamacpp._HELP_CACHE.clear()
    assert llamacpp.quantize_supports(tool, "--tensor-type-file") and not llamacpp.quantize_supports(tool, "--nothing")
    tool.write_text("import sys\nprint('usage: x')\nsys.exit(1)\n", encoding="utf-8")
    os.utime(tool, (1_700_000_000, 1_700_000_000))
    assert not llamacpp.quantize_supports(tool, "--tensor-type-file")
    assert not llamacpp.quantize_supports(tmp_path / "missing", "--pure")
