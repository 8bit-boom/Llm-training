"""The tuner's vLLM engine measurement: one reference, the same rows through the engine once per variant (fake vLLM here), a failing variant is a row, a stop ends it, nothing is left running."""
import json

import pytest

np = pytest.importorskip("numpy")

from ndforge_worker import protocol, vllmcache, vllmteachertune as tt  # noqa: E402
from test_cache_identity import make_model  # noqa: E402
from test_vllm_teacher_check import K, fake_chunk, job_for, make_reference  # noqa: E402

VARIANTS = [{"id": "baseline", "label": "Your settings"}, {"id": "baseline2", "label": "Your settings again"},
            {"id": "max_num_seqs=64", "knob": "max_num_seqs", "value": 64, "label": "64 rows at once", "engine": {"max_num_seqs": 64}},
            {"id": "gpu_util_pct=92", "knob": "gpu_util_pct", "value": 92, "label": "92 % of the card", "engine": {"gpu_memory_utilization": 0.92}}]


def run_tune(tmp_path, monkeypatch, capsys, chunk, variants=VARIANTS, **kw):
    teacher = make_model(tmp_path / "origin")
    ref = make_reference(tmp_path, teacher)
    monkeypatch.setattr(vllmcache, "run_chunk", chunk)
    monkeypatch.setattr(protocol, "_out", __import__("sys").stdout)
    job = job_for(tmp_path, ref, variants=variants, work=str(tmp_path / "work"), **kw)
    job.pop("out")
    rc = tt.run(job)
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    return rc, [x for x in lines if x["type"] == "result"][0]["data"]


def test_every_variant_is_scored_on_the_same_rows_with_its_own_engine_settings(tmp_path, monkeypatch, capsys):
    engines: list = []
    inner = fake_chunk()

    def chunk(job, *a, **k):
        engines.append(job["engine"])
        r = inner(job, *a, **k)
        r["tokens_per_s"] = 2000.0 + 100 * len(engines)
        r["gpu"] = {"util": job["engine"].get("gpu_memory_utilization", 0.88), "lowered": False}
        return r
    rc, res = run_tune(tmp_path, monkeypatch, capsys, chunk)
    assert rc == 0 and [r["id"] for r in res["rows"]] == ["baseline", "baseline2", "max_num_seqs=64", "gpu_util_pct=92"] and not res["stopped"]
    assert all(r["ok"] and r["top1"] == 1.0 and r["kl"] is not None for r in res["rows"]) and res["reference"]["kind"] == "existing_cache"
    assert engines[2]["max_num_seqs"] == 64 and "max_num_seqs" not in engines[0] and engines[3]["gpu_memory_utilization"] == 0.92
    assert res["rows"][3]["tokens_per_s"] == 2400.0 and res["rows"][3]["gpu_util"] == 0.92 and res["rows_scored"] == 5


def test_a_variant_that_fails_is_a_row_with_a_plain_error_and_the_others_go_on(tmp_path, monkeypatch, capsys):
    inner = fake_chunk()
    n = {"i": 0}

    def chunk(job, *a, **k):
        n["i"] += 1
        if n["i"] == 3:
            raise vllmcache.VllmCacheError("CUDA error: out of memory while starting the engine")
        return inner(job, *a, **k)
    rc, res = run_tune(tmp_path, monkeypatch, capsys, chunk)
    bad = res["rows"][2]
    assert rc == 0 and bad["ok"] is False and bad["oom"] is True and "out of memory" in bad["error"] and res["rows"][3]["ok"] is True


def test_a_variant_whose_answers_differ_shows_it_in_its_agreement(tmp_path, monkeypatch, capsys):
    inner_ok, inner_flip = fake_chunk(), fake_chunk(flip=True)
    n = {"i": 0}

    def chunk(job, *a, **k):
        n["i"] += 1
        return (inner_flip if n["i"] == 3 else inner_ok)(job, *a, **k)
    _, res = run_tune(tmp_path, monkeypatch, capsys, chunk)
    assert res["rows"][0]["top1"] == 1.0 and res["rows"][2]["top1"] < 0.5


def test_stop_ends_the_run_between_variants_and_a_stopped_chunk_is_not_a_row(tmp_path, monkeypatch, capsys):
    inner = fake_chunk()
    n = {"i": 0}

    def chunk(job, *a, **k):
        n["i"] += 1
        r = inner(job, *a, **k)
        if n["i"] == 2:
            r["stopped"] = True
        return r
    _, res = run_tune(tmp_path, monkeypatch, capsys, chunk)
    assert res["stopped"] is True and [r["id"] for r in res["rows"]] == ["baseline"]


def test_the_job_needs_variants_and_is_in_the_contract():
    c = json.loads((__import__("pathlib").Path(tt.__file__).parent / "contract.json").read_text())
    assert "vllm_teacher_tune" in c["jobs"] and "vllm_teacher_tune_failed" in c["error_codes"]
    assert tt.is_oom("CUDA out of memory") and not tt.is_oom("the container did not answer")
