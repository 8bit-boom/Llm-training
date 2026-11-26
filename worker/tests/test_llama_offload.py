"""The tuner's llama.cpp offload test: the rows, the stop at the first number of layers that runs, an allocation error as an out-of-memory row, and the job with a stand-in llama-imatrix."""
import json
import sys
import textwrap

import pytest

from ndforge_worker import imatrix, llamacpp, llamaoffload as lo, protocol

FAKE = textwrap.dedent('''
    import sys
    a = sys.argv
    ngl = int(a[a.index("-ngl") + 1])
    if ngl > 30:
        print("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 31000 MiB on device 0: cudaMalloc failed: out of memory")
        sys.exit(1)
    print("compute_imatrix: 2.50 seconds per pass - ETA 0.20 minutes")
    open(a[a.index("-o") + 1], "wb").write(b"x")
''')


def test_the_text_is_long_enough_and_an_allocation_error_is_an_oom():
    assert len(lo.calibration_text(6, 512)) >= 6 * 512 * 5
    assert lo.is_oom(["ggml_backend_cuda_buffer_type_alloc_buffer: ... cudaMalloc failed: out of memory"]) and lo.is_oom(["llama_model_load: failed to allocate buffer"])
    assert not lo.is_oom(["error: invalid magic characters"])
    assert lo.seconds_per_pass(["x", "compute_imatrix: 4.25 seconds per pass - ETA 1 minutes"]) == 4.25 and lo.seconds_per_pass(["nothing"]) is None


def test_the_measurement_stops_at_the_first_number_of_layers_that_runs_and_honours_stop():
    seen = []

    def run_one(ngl):
        seen.append(ngl)
        return {"ngl": ngl, "ok": ngl <= 40}
    rows, stopped = lo.measure([99, 60, 40, 30], run_one)
    assert seen == [99, 60, 40] and [r["ok"] for r in rows] == [False, False, True] and not stopped
    rows, stopped = lo.measure([99, 60], run_one, stop_requested=lambda: True)
    assert rows == [] and stopped


def test_a_row_says_how_it_was_timed_and_why_it_failed():
    ok = lo.row_of(40, 0, ["compute_imatrix: 2.00 seconds per pass"], 30.0, True, 6, 512)
    assert ok["ok"] and ok["tokens_per_s"] == 256.0 and "seconds per pass" in ok["timed_by"]
    whole = lo.row_of(40, 0, [], 12.0, True, 6, 512)
    assert whole["tokens_per_s"] == 256.0 and "including loading" in whole["timed_by"]
    bad = lo.row_of(99, 1, ["cudaMalloc failed: out of memory"], 3.0, False, 6, 512)
    assert not bad["ok"] and bad["oom"] and "out of memory" in bad["error"]


def test_the_job_tries_fewer_layers_until_one_runs_and_removes_its_files(tmp_path, monkeypatch, capsys):
    tool = tmp_path / "llama-imatrix.py"
    tool.write_text(FAKE, encoding="utf-8")
    gguf = tmp_path / "model-f16.gguf"
    gguf.write_bytes(b"GGUF" + b"0" * 2000)
    monkeypatch.setattr(llamacpp, "find_llama_cpp", lambda: (None, None))
    monkeypatch.setattr(llamacpp, "find_tool", lambda name, quantize=None: tool)
    monkeypatch.setattr(protocol, "_out", sys.stdout)
    work = tmp_path / "work"
    rc = lo.run({"job": "llama_offload_tune", "gguf": str(gguf), "ctx": 64, "chunks": 4, "candidates": [99, 60, 24, 12], "work": str(work), "stop_file": str(tmp_path / "STOP")})
    res = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    data = [x for x in res if x["type"] == "result"][0]["data"]
    assert rc == 0 and [r["ngl"] for r in data["rows"]] == [99, 60, 24] and [r["ok"] for r in data["rows"]] == [False, False, True]
    assert data["rows"][0]["oom"] and data["rows"][2]["tokens_per_s"] == 25.6 and data["gguf"] == "model-f16.gguf" and not data["stopped"]
    assert not (work / "offload_text.txt").exists() and not (work / "offload_matrix.gguf").exists()


def test_a_missing_llama_cpp_or_file_is_a_plain_error(tmp_path, monkeypatch):
    monkeypatch.setattr(llamacpp, "find_llama_cpp", lambda: (None, None))
    monkeypatch.setattr(llamacpp, "find_tool", lambda name, quantize=None: None)
    with pytest.raises(lo.OffloadError, match="not there any more"):
        lo.run({"gguf": str(tmp_path / "no.gguf"), "work": str(tmp_path / "w")})
    g = tmp_path / "m.gguf"
    g.write_bytes(b"x")
    with pytest.raises(lo.OffloadError, match="llama.cpp was not found"):
        lo.run({"gguf": str(g), "work": str(tmp_path / "w")})
    c = json.loads((__import__("pathlib").Path(lo.__file__).parent / "contract.json").read_text())
    assert "llama_offload_tune" in c["jobs"] and "llama_offload_tune_failed" in c["error_codes"]
