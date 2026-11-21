"""Which KV cache type a vLLM measurement really used: the job's choice overrides the engine settings, the engine's own log confirms it, a disagreement makes the result invalid, and one job can compare auto and 8-bit."""
import itertools
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).parent))

from ndforge_worker import vllmcache, vllmengine as ve, vllmteachercheck as tc  # noqa: E402
from test_cache_identity import make_model  # noqa: E402
from test_vllm_cache_job import FakeWorld, make_rows, new_writer, run_it  # noqa: E402
from test_vllm_teacher_check import K, fake_chunk, job_for, make_reference, probs, run_job  # noqa: E402

# the owner's engine log lines (vLLM 0.34, the in-flight route)
CONFIG_FP8 = "(EngineCore pid=212) INFO [core.py:77] Initializing a V1 LLM engine with config: model='/m', quantization=bitsandbytes, kv_cache_dtype='fp8', max_num_seqs=15"
USING_FP8 = "(EngineCore pid=212) INFO [kv_cache.py:301] Using fp8 data type to store kv cache. It reduces the GPU memory footprint."
FLASHINFER = "(EngineCore pid=212) INFO [flashinfer.py:88] FlashInfer: q_data_type=torch.bfloat16 kv_cache_dtype=torch.float8_e4m3fn"
CONFIG_AUTO = "(EngineCore pid=212) INFO [core.py:77] Initializing a V1 LLM engine with config: model='/m', quantization=bitsandbytes, 'kv_cache_dtype': 'auto', max_num_seqs=15"
DICT_FP8 = "engine_args: {'model': '/m', 'kv_cache_dtype': 'fp8', 'max_num_seqs': 15}"


def test_the_engine_log_says_which_kv_cache_it_used():
    for line in (CONFIG_FP8, USING_FP8, FLASHINFER, DICT_FP8):
        assert ve.kv_cache_from_log(line)["seen"] == "fp8", line
    assert ve.kv_cache_from_log(CONFIG_AUTO)["seen"] == "auto"
    assert ve.kv_cache_from_log("kv_cache_dtype=torch.bfloat16")["seen"] == "auto"
    assert ve.kv_cache_from_log("INFO nothing about it\nINFO other")["seen"] is None
    mixed = ve.kv_cache_from_log(CONFIG_AUTO + "\n" + USING_FP8)
    assert mixed["seen"] == "fp8" and USING_FP8 in mixed["evidence"][0], "any fp8 line wins over an auto default"


def test_the_job_overrides_the_engine_settings_in_both_directions():
    owner = {"engine": {"kind": "vllm", "kv_cache_dtype": "fp8", "kv_cache_dtype_teacher": True, "container": {"kind": "docker"}}}
    assert vllmcache.teacher_kv(owner) == "fp8", "the owner's settings alone make fp8"
    auto = tc.effective_job(owner, "auto")
    assert vllmcache.teacher_kv(auto) == "auto" and auto["engine"]["kv_cache_dtype"] == "auto" and auto["engine"]["kv_cache_dtype_teacher"] is False
    plain = {"engine": {"kind": "vllm", "container": {"kind": "docker"}}}
    assert vllmcache.teacher_kv(tc.effective_job(plain, "fp8")) == "fp8"
    assert vllmcache.vllmengine and "--kv-cache-dtype" in vllmcache.script_args(auto, {"route": "prequant"}, "/m", {}, 0, 1, 64, 4) and vllmcache.script_args(auto, {"route": "prequant"}, "/m", {}, 0, 1, 64, 4)[-2:] != ["--kv-cache-dtype", "fp8"]


def kv_chunk(seen_jobs, force=None, noise8=0.0, tail=0.0):
    """A fake run_chunk that records the kv type it was asked for and reports what 'the engine log' says (the asked type unless `force`)."""
    base = fake_chunk()

    def chunk(job, writer, rows_dir, meta, origin, origin_dir, first, last, k, max_seq, stop_file, **kw):
        asked = vllmcache.teacher_kv(job)
        seen_jobs.append(asked)
        r = fake_chunk(noise=(noise8 if asked == "fp8" else 0.0) + tail)(job, writer, rows_dir, meta, origin, origin_dir, first, last, k, max_seq, stop_file)
        seen = force or asked
        r["kv_cache_seen"] = {"seen": seen, "evidence": [USING_FP8 if seen == "fp8" else CONFIG_AUTO]}
        return r
    return chunk


def result(lines):
    return [x for x in lines if x["type"] == "result"][0]["data"]


def test_an_auto_check_is_auto_even_when_the_owners_settings_say_fp8_and_the_result_says_confirmed(tmp_path, monkeypatch, capsys):
    teacher = make_model(tmp_path / "origin")
    ref = make_reference(tmp_path, teacher)
    jobs: list = []
    owner_engine = {"engine": {"kind": "vllm", "teacher": {}, "container": {"kind": "docker"}, "kv_cache_dtype": "fp8", "kv_cache_dtype_teacher": True}}
    rc, lines = run_job(tmp_path, monkeypatch, capsys, ref, kv_chunk(jobs), **owner_engine)
    res = result(lines)
    assert rc == 0 and jobs == ["auto"], "the engine was asked for auto, not for the owner's fp8"
    assert res["kv_cache_dtype"] == "auto" and res["kv_cache_confirmed"]["seen"] == "auto" and res["kv_cache_confirmed"]["agrees"] is True and res["valid"] is True
    assert "Measured with the auto KV cache (confirmed by the engine's log)" in res["text"]
    assert "kv_cache" not in res["identity"], "an exact-cache identity has no fp8 key"


def test_fp8_only_when_asked_and_its_identity_says_so(tmp_path, monkeypatch, capsys):
    teacher = make_model(tmp_path / "origin")
    ref = make_reference(tmp_path, teacher)
    jobs: list = []
    rc, lines = run_job(tmp_path, monkeypatch, capsys, ref, kv_chunk(jobs), kv_cache_dtype="fp8")
    res = result(lines)
    assert jobs == ["fp8"] and res["kv_cache_dtype"] == "fp8" and res["identity"]["kv_cache"] == "fp8" and res["kv_cache_confirmed"]["agrees"] is True
    assert "Measured with the fp8 KV cache (confirmed by the engine's log)" in res["text"]


def test_a_log_that_disagrees_with_the_request_makes_the_result_invalid(tmp_path, monkeypatch, capsys):
    teacher = make_model(tmp_path / "origin")
    ref = make_reference(tmp_path, teacher)
    rc, lines = run_job(tmp_path, monkeypatch, capsys, ref, kv_chunk([], force="fp8"))
    res = result(lines)
    assert res["verdict"] == "invalid" and res["valid"] is False and res["kv_cache_confirmed"] == {**res["kv_cache_confirmed"], "requested": "auto", "seen": "fp8", "agrees": False}
    assert res["text"].startswith("INVALID: this check asked for the auto KV cache, but the engine's own log says it used fp8")


def test_a_log_that_says_nothing_is_taken_as_asked_and_it_is_said(tmp_path, monkeypatch, capsys):
    teacher = make_model(tmp_path / "origin")
    ref = make_reference(tmp_path, teacher)
    quiet = kv_chunk([])

    def silent(*a, **k):
        r = quiet(*a, **k)
        r["kv_cache_seen"] = {"seen": None, "evidence": []}
        return r
    rc, lines = run_job(tmp_path, monkeypatch, capsys, ref, silent)
    res = result(lines)
    assert "the engine's log did not say; taken as asked" in res["text"] and res["valid"] is True and res["kv_cache_confirmed"]["agrees"] is None


def test_compare_kv_scores_the_same_rows_twice_and_prints_both_and_the_difference(tmp_path, monkeypatch, capsys):
    teacher = make_model(tmp_path / "origin")
    ref = make_reference(tmp_path, teacher)
    jobs: list = []
    rc, lines = run_job(tmp_path, monkeypatch, capsys, ref, kv_chunk(jobs, noise8=0.05), compare_kv=True)
    res = result(lines)
    assert rc == 0 and jobs == ["auto", "fp8"]
    c = res["compare_kv"]
    assert c["valid"] is True and c["auto"]["kv_cache_confirmed"]["seen"] == "auto" and c["fp8"]["kv_cache_confirmed"]["seen"] == "fp8"
    assert c["auto"]["metrics"]["top1"] == 1.0 and c["fp8"]["metrics"]["tail_diff_mean"] > c["auto"]["metrics"]["tail_diff_mean"]
    assert c["difference"]["tail_diff_mean"] == pytest.approx(c["fp8"]["metrics"]["tail_diff_mean"] - c["auto"]["metrics"]["tail_diff_mean"], abs=1e-5)
    assert "The 8-bit KV cache against the exact one" in c["text"] and "exact against 8-bit" in c["text"] and c["text"] in res["text"]
    assert res["kv_cache_dtype"] == "auto" and res["identity"].get("kv_cache") is None, "the headline numbers are the exact cache's"
    assert c["fp8"]["identity"]["kv_cache"] == "fp8"


def test_an_invalid_run_makes_the_comparison_invalid(tmp_path, monkeypatch, capsys):
    teacher = make_model(tmp_path / "origin")
    ref = make_reference(tmp_path, teacher)
    rc, lines = run_job(tmp_path, monkeypatch, capsys, ref, kv_chunk([], force="fp8"), compare_kv=True)
    r2 = result(lines)
    assert r2["compare_kv"]["valid"] is False and "not valid" in r2["compare_kv"]["text"]


def test_the_systematic_tail_difference_is_said_plainly_without_alarm(tmp_path, monkeypatch, capsys):
    teacher = make_model(tmp_path / "origin")
    ref = make_reference(tmp_path, teacher)
    rc, lines = run_job(tmp_path, monkeypatch, capsys, ref, kv_chunk([], tail=-0.1))
    t = result(lines)["text"]
    assert "puts about" in t and "less probability outside its top 4 than the standard teacher does" in t and "the heal's tail term is affected a little; the cause is not known" in t


def test_no_tail_note_without_a_tail_difference(tmp_path, monkeypatch, capsys):
    teacher = make_model(tmp_path / "origin")
    ref = make_reference(tmp_path, teacher)
    rc, lines = run_job(tmp_path, monkeypatch, capsys, ref, kv_chunk([]))
    assert "less probability outside" not in result(lines)["text"]


def test_the_scoring_run_records_what_the_engine_log_said(tmp_path, monkeypatch):
    for job_kv, log_line, want in ((True, USING_FP8, "fp8"), (False, CONFIG_AUTO, "auto")):
        sub = tmp_path / ("a" if job_kv else "b")
        sub.mkdir()
        rows = sub / "rows"
        rows.mkdir()
        make_rows(rows, [10, 10])
        world = FakeWorld(sub, monkeypatch, bind="ok")
        orig = world.popen

        def popen(cmd, _o=orig, _l=log_line, **kw):
            p = _o(cmd, **kw)
            p.stdout = itertools.chain([_l + "\n"], p.stdout)
            return p
        monkeypatch.setattr(ve, "_popen", popen)
        job = {"engine": {"kind": "vllm", "teacher": {}, "container": {"kind": "docker"}, **({"kv_cache_dtype": "fp8", "kv_cache_dtype_teacher": True} if job_kv else {})}, "part_tokens": 30}
        r = run_it(sub, world, new_writer(sub, 2), rows, 0, 2, job=job)
        assert r["kv_cache_requested"] == want and r["kv_cache_seen"]["seen"] == want and r["kv_cache_seen"]["evidence"]
