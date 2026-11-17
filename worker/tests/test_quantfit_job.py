"""The jobs `quantfit_estimate` and `quantfit` end to end on a synthetic 16-bit GGUF with the stand-in llama-quantize: measure, plan, make the file, read the WHOLE output back and compare the
predicted weighted error with the measured one. This proves the pipeline is consistent (sampling, bookkeeping, rules, the file that comes out); it says nothing about llama.cpp itself."""
import json
from pathlib import Path

import numpy as np
import pytest
from ggufmake import BF16, F32, bf16_values, write
from test_quantmeasure import make_matrix

from ndforge_worker import __main__ as entry
from ndforge_worker import ggufinfo, ggufrows, imatrix, llamacpp, mixedquant, protocol, quantblocks, quantfit, quantmeasure as qm

STANDIN = Path(__file__).parent / "standins" / "llama_quantize_standin.py"
KV = [("general.architecture", "str", "qwen35"), ("general.alignment", "u32", 32), ("qwen35.block_count", "u32", 4)]


def synthetic_model(path, seed=21):
    """4 layers, rows 256-1024, columns 512 / 1024, Laplace weights with a different scale in every tensor (the reason a mixed plan can win)."""
    rng = np.random.default_rng(seed)
    ts = []

    def add(name, dims, scale):
        ts.append((name, dims, BF16, bf16_values(rng, tuple(reversed(dims)), scale)))
    add("token_embd.weight", [512, 600], 0.02)
    ts.append(("output_norm.weight", [512], F32, np.ones(512, np.float32)))
    add("output.weight", [512, 600], 0.02)
    for i in range(4):
        s = rng.uniform(0.004, 0.09, 6)
        ts.append((f"blk.{i}.attn_norm.weight", [512], F32, np.ones(512, np.float32)))
        add(f"blk.{i}.attn_q.weight", [512, 1024], s[0])
        add(f"blk.{i}.attn_k.weight", [512, 256], s[1])
        add(f"blk.{i}.attn_v.weight", [512, 256], s[2])
        add(f"blk.{i}.attn_output.weight", [1024, 512], s[3])
        add(f"blk.{i}.ffn_gate.weight", [512, 1024], s[4])
        add(f"blk.{i}.ffn_up.weight", [512, 1024], s[5])
        add(f"blk.{i}.ffn_down.weight", [1024, 512], s[0])
    return write(path, ts, KV)


@pytest.fixture
def world(tmp_path, monkeypatch):
    llamacpp._HELP_CACHE.clear()
    for v in ("STANDIN_NO_TYPEFILE", "STANDIN_FLIP", "STANDIN_FAIL", "STANDIN_SLOW"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("STANDIN_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.setattr(quantfit, "_llama_quantize", lambda: STANDIN)
    events = []
    monkeypatch.setattr(protocol, "emit", lambda kind, **f: events.append({"type": kind, **f}))
    gguf = synthetic_model(tmp_path / "model-f16.gguf")
    src = ggufrows.open_source(gguf)
    matrix = make_matrix(tmp_path / "imatrix.gguf", src, np.random.default_rng(5))

    class W:
        pass
    w = W()
    w.tmp, w.gguf, w.src, w.matrix, w.events = tmp_path, gguf, src, matrix, events
    w.imp, _ = qm.importance(matrix, src)
    return w


def full_weighted_error(w, out_path, only=None):
    """sum over quantizable tensors of sum a * (w - w_hat)^2, over ALL rows, reading the file llama-quantize (the stand-in) wrote"""
    info = ggufinfo.read(out_path)
    by = {t["name"]: t for t in info["tensors"]}
    total, types = 0.0, {}
    for t in w.src["tensors"]:
        if not ggufrows.is_sampled(t):
            continue
        cols, per, experts = ggufrows.geometry(t)
        o = by[t["name"]]
        ty = ggufinfo.type_name(o["type"])
        types[t["name"]] = ty
        raw = np.fromfile(out_path, dtype=np.uint8, count=per * experts * quantblocks.row_bytes(ty, cols), offset=info["data_start"] + o["offset"])
        dec = quantblocks.decode(raw, ty, per * experts, cols).astype(np.float64)
        src_rows = ggufrows.read_rows(w.src, t, range(per * experts)).astype(np.float64)
        total += float(np.sum(np.repeat(w.imp[t["name"]], per, axis=0) * (src_rows - dec) ** 2))
    return total, types


def job(w, **kw):
    return {"job": "quantfit", "gguf": str(w.gguf), "matrix": str(w.matrix), "target_gb": kw.pop("target_gb", 0.0), "sample_rows": 32, "engine": "auto", "cache_dir": str(w.tmp / "quantfit"),
            "stop_file": str(w.tmp / "STOP"), "make": True, "neighbours": False, "out_dir": str(w.tmp / "out"), **kw}, None


def test_predicted_error_equals_the_measured_error_of_the_file_that_comes_out(world, capsys):
    w = world
    # the plain file at the uniform Q4_K size, made with the stand-in exactly as llama-quantize --pure would
    uni = w.tmp / "uniform-q4k.gguf"
    import subprocess
    import sys
    subprocess.run([sys.executable, str(STANDIN), "--pure", "--imatrix", str(w.matrix), str(w.gguf), str(uni), "Q4_K_M"], check=True)
    uni_bytes = uni.stat().st_size
    j, _ = job(w, target_gb=uni_bytes / 1e9)
    res = quantfit.run_job(j)
    assert res["fits"] and res["engine"] == "llama" and len(res["files"]) == 1
    made = Path(res["files"][0]["path"])
    assert made.name == "model-mixed_measured.gguf" and res["files"][0]["mixed"] == "mixed_measured" and res["files"][0]["matrix"] is True
    plan = res["plan"]
    # the types in the file are the types in the plan, tensor by tensor
    measured_err, types = full_weighted_error(w, made)
    want = {**plan["assign"], **plan["pinned_assign"]}
    assert types == want
    # prediction: the sum of the planned and pinned tensors' sampled estimates; measurement: all rows
    rep = {r["id"]: r for r in res["report"]["rows"]}
    norm2 = res["report"]["norm2"]
    predicted_rel, measured_rel = rep["plan"]["error"], float(np.sqrt(measured_err / norm2))
    # uniform Q4_K
    uni_err, uni_types = full_weighted_error(w, uni)
    assert set(uni_types.values()) == {"Q4_K"}
    uni_rel = float(np.sqrt(uni_err / norm2))
    print(f"\npredicted plan error {predicted_rel:.5f}, measured {measured_rel:.5f}; predicted uniform Q4_K {rep['uniform_q4_k']['error']:.5f}, measured {uni_rel:.5f}; "
          f"plan {made.stat().st_size} bytes (predicted {plan['total']}), uniform Q4_K {uni_bytes} bytes")
    assert predicted_rel == pytest.approx(measured_rel, rel=0.05)
    assert rep["uniform_q4_k"]["error"] == pytest.approx(uni_rel, rel=0.05)
    assert measured_rel < uni_rel, "the measured plan has less weighted error than uniform Q4_K ..."
    assert made.stat().st_size <= uni_bytes, "... in a file that is not bigger"
    # the file is the size the plan said (the stand-in writes no padding beyond the alignment)
    assert abs(made.stat().st_size - plan["total"]) <= 32 * len(w.src["tensors"]) + 64
    assert len(set(plan["assign"].values())) > 1, "different tensors got different types"
    # the saved plan, the type file and the events
    saved = list((w.tmp / "quantfit").glob("*/plan-*.json"))
    assert len(saved) == 1 and json.loads(saved[0].read_text())["fits"]
    assert (w.tmp / "out" / "mixed_measured.tensor-types.txt").read_text().splitlines() == res["rules"]
    assert "--tensor-type-file" in res["flags"] and res["flags"][0] == "--output-tensor-type"
    stages = [e["stage"] for e in protocol.timings()] if hasattr(protocol, "timings") else []
    assert {"allocate", "quantize"} <= set(stages)


def test_a_second_run_uses_the_cache_and_another_size_reuses_it(world):
    w = world
    j, _ = job(w, target_gb=0.0045, make=False)
    first = quantfit.run_job(j)
    def calls():
        return [c for c in (w.tmp / "calls.jsonl").read_text().splitlines() if "--version" not in c and "--help" not in c]
    before = calls()
    again = quantfit.run_job(j)
    assert calls() == before, "nothing was measured again"
    assert again["plan"]["assign"] == first["plan"]["assign"] and again["measure_key"] == first["measure_key"]
    j2 = {**j, "target_gb": 0.0055}
    assert quantfit.run_job(j2)["measure_key"] == first["measure_key"], "another size reuses the same measures"
    j3 = {**j, "sample_rows": 64}
    assert quantfit.run_job(j3)["measure_key"] != first["measure_key"]


def test_below_the_minimum_the_floor_plan_is_returned_and_nothing_is_made(world):
    w = world
    j, _ = job(w, target_gb=0.001)
    res = quantfit.run_job(j)
    assert not res["fits"] and res["files"] == [] and any("not made" in n for n in res["notes"]) and res["min_gb"] > 0.001
    assert set(res["plan"]["rung"].values()) == {"Q3_K"}


def test_the_neighbours_are_made_and_the_matrix_refusal_is_retried(world):
    w = world
    j, _ = job(w, target_gb=0.0055, neighbours=True)
    res = quantfit.run_job(j)
    groups = [f["group"] for f in res["files"]]
    assert groups[0] == quantfit.LABEL and all(Path(f["path"]).is_file() for f in res["files"]) and len(groups) >= 2
    assert all(f["matrix"] is True for f in res["files"] if f["mixed"] is None)


def test_estimate_job(world):
    w = world
    r = quantfit.estimate_job({"job": "quantfit_estimate", "gguf": str(w.gguf), "matrix": str(w.matrix), "sample_rows": 32, "engine": "auto", "target_gb": 0.005})
    assert r["engine"] == "llama" and r["tensor_type_file"] is True and r["matrix_ok"] is True and r["llama_quantize"].endswith("llama_quantize_standin.py")
    assert r["params"] == ggufinfo.n_parameters(w.src) and 0 < r["sampled_weights"] < r["params"] and r["read_bytes"] == r["sampled_weights"] * 2
    assert 0 < r["min_gb"] < r["max_gb"] and set(r["plain_estimates"]) == set(quantfit.PLAIN_SIZES) and r["speed"]["Q4_K"] > 0
    assert r["minutes_estimate"]["chosen"] >= 0 and "an estimate" in r["minutes_note"] and "assumptions" in r["minutes_note"]
    assert "version: 4321" in r["llama_version"]
    r = quantfit.estimate_job({"job": "quantfit_estimate", "gguf": str(w.gguf), "matrix": None, "sample_rows": 32, "engine": "reference"})
    assert r["engine"] == "reference" and r["matrix_ok"] is None and any("No importance matrix" in n for n in r["notes"])
    bad = make_matrix(w.tmp / "bad.gguf", w.src, np.random.default_rng(1), bad="blk.1.attn_k.weight")
    r = quantfit.estimate_job({"job": "quantfit_estimate", "gguf": str(w.gguf), "matrix": str(bad)})
    assert r["matrix_ok"] is False and any("not of this model" in n for n in r["notes"])
    with pytest.raises(quantfit.QuantFitError, match="could not be opened"):
        quantfit.estimate_job({"gguf": str(w.tmp / "x.gguf")})


def test_the_worker_entry_runs_both_jobs_and_maps_the_errors(world, tmp_path):
    w = world
    jobfile = w.tmp / "job.json"
    jobfile.write_text(json.dumps({"job": "quantfit_estimate", "gguf": str(w.gguf), "matrix": str(w.matrix)}), encoding="utf-8")
    assert entry.main(["x", str(jobfile)]) == 0 and [e for e in w.events if e["type"] == "result"][-1]["data"]["engine"] == "llama"
    j, _ = job(w, target_gb=0.0055, make=False)
    jobfile.write_text(json.dumps(j), encoding="utf-8")
    assert entry.main(["x", str(jobfile)]) == 0 and [e for e in w.events if e["type"] == "result"][-1]["data"]["fits"]
    # plain failures
    for patch, code in (({"target_gb": 0}, "quantfit_failed"), ({"gguf": str(w.tmp / "gone.gguf")}, "quantfit_failed")):
        jobfile.write_text(json.dumps({**j, **patch}), encoding="utf-8")
        w.events.clear()
        assert entry.main(["x", str(jobfile)]) == 1
        e = [x for x in w.events if x["type"] == "error"][0]
        assert e["code"] == code and "Traceback" not in e["message"]
    # a stop: the STOP file is there before the job starts
    (w.tmp / "STOP").write_text("")
    jobfile.write_text(json.dumps({**j, "cache_dir": str(w.tmp / "fresh")}), encoding="utf-8")
    w.events.clear()
    assert entry.main(["x", str(jobfile)]) == 1
    assert [x for x in w.events if x["type"] == "error"][0]["code"] == "quantfit_stopped"


def test_make_without_llama_cpp_reports_the_missing_code(world, monkeypatch):
    w = world
    monkeypatch.setattr(quantfit, "_llama_quantize", lambda: None)
    j, _ = job(w, target_gb=0.0055)
    jobfile = w.tmp / "job.json"
    jobfile.write_text(json.dumps(j), encoding="utf-8")
    assert entry.main(["x", str(jobfile)]) == 1
    assert [e for e in w.events if e["type"] == "error"][0]["code"] == "llama_cpp_missing"
    # measuring only, without llama.cpp: the numpy engine, with a note
    jobfile.write_text(json.dumps({**j, "make": False}), encoding="utf-8")
    w.events.clear()
    assert entry.main(["x", str(jobfile)]) == 0
    res = [e for e in w.events if e["type"] == "result"][0]["data"]
    assert res["engine"] == "reference"


def test_contract_lists_the_new_jobs_and_codes():
    c = json.loads((Path(entry.__file__).parent / "contract.json").read_text())
    assert {"quantfit", "quantfit_estimate"} <= set(c["jobs"]) and {"quantfit_failed", "quantfit_stopped"} <= set(c["error_codes"])
    assert c["jobs"] == sorted(c["jobs"]) and c["error_codes"] == sorted(c["error_codes"])
