"""The optional 8-bit KV cache: command builders, identity separation, the teacher's extra confirmation, the self-test's second pass (fake engine and logs), the llama.cpp cache type."""
import json
import types

import pytest

np = pytest.importorskip("numpy")

from ndforge_worker import thinkcheck as tc, thinkcheck_llama as tl, thinkcheck_vllm as tv, vllm_cache as vc, vllmcache, vllmengine as ve, vllmselftest as st  # noqa: E402
from test_cache_identity import make_model  # noqa: E402


def test_the_flag_is_added_only_when_fp8_is_set_for_every_engine_kind():
    assert "--kv-cache-dtype" not in ve.serve_args("m") and "--kv-cache-dtype" not in ve.serve_args("m", kv_cache_dtype="auto")
    a = ve.serve_args("m", kv_cache_dtype="fp8")
    assert a[a.index("--kv-cache-dtype") + 1] == "fp8"
    for kind, extra in (("docker", {}), ("docker_wsl", {}), ("wslc", {}), ("wsl", {})):
        c = ve.engine_config({"engine": {"kind": kind, "kv_cache_dtype": "fp8", **extra}})
        cmd = ve.Engine(c, "org/model").build()
        assert "--kv-cache-dtype" in cmd and cmd[cmd.index("--kv-cache-dtype") + 1] == "fp8", kind
        off = ve.Engine(ve.engine_config({"engine": {"kind": kind}}), "org/model").build()
        assert "--kv-cache-dtype" not in off, kind
    url = ve.Engine(ve.engine_config({"engine": {"kind": "url", "url": "http://x:1", "kv_cache_dtype": "fp8"}}), "m")
    assert url.build() == [] and url.kv_cache_dtype == "fp8", "the owner's own server: only recorded"
    with pytest.raises(ve.EngineError):
        ve.engine_config({"engine": {"kind": "docker", "kv_cache_dtype": "int4"}})


def test_a_check_with_fp8_has_its_own_key_and_says_so(tmp_path):
    base = {"model": "org/cut", "backend": "vllm", "engine": {"kind": "docker", "vram_gb": 32, "weights_gb": 20}}
    fp8 = {**base, "engine": {**base["engine"], "kv_cache_dtype": "fp8"}}
    k0, k1 = tc.cache_key(base, "vllm", tc.settings_of({}))["engine"], tc.cache_key(fp8, "vllm", tc.settings_of({}))["engine"]
    assert "kv_cache" not in k0 and k1["kv_cache"] == "fp8", "the key of an exact check is exactly what it was"
    e = tv.make_engine(fp8)
    assert e.kv_cache_dtype == "fp8" and "--kv-cache-dtype" in e.build() and fp8["_engine_report"]["kv_cache"] == "fp8"
    plain = {**base}
    tv.make_engine(plain)
    assert "kv_cache" not in plain["_engine_report"]


def test_the_teachers_cache_needs_the_second_confirmation_and_carries_it_in_its_identity(tmp_path):
    origin = make_model(tmp_path / "origin")
    job = lambda **e: {"engine": {"kind": "vllm", "teacher": {}, "container": {"kind": "docker"}, **e}}      # noqa: E731
    assert vllmcache.teacher_kv(job(kv_cache_dtype="fp8")) == "auto", "fp8 alone does not reach the teacher"
    assert vllmcache.teacher_kv(job(kv_cache_dtype_teacher=True)) == "auto"
    assert vllmcache.teacher_kv(job(kv_cache_dtype="fp8", kv_cache_dtype_teacher=True)) == "fp8"
    exact = vllmcache.identity(job(kv_cache_dtype="fp8"), "Someone/Finetune", origin)
    assert "kv_cache" not in exact
    fp8 = vllmcache.identity(job(kv_cache_dtype="fp8", kv_cache_dtype_teacher=True), "Someone/Finetune", origin)
    assert fp8["kv_cache"] == "fp8" and {k: v for k, v in fp8.items() if k != "kv_cache"} == exact
    from ndforge_worker import distilljob as dj
    meta = {"teacher": "t", "k": 4, "rows": 1, "data_hash": "h", "tokenizer_hash": "x", "max_len": 8, "general_key": {}, "general_rows": 0}
    assert not dj.cache_matches({**meta, **exact, "format": 1}, {**meta, **fp8}) and dj.cache_matches({**meta, **fp8, "format": 1}, {**meta, **fp8})
    assert not dj.cache_matches({**meta, **fp8, "format": 1}, {**meta, **exact}), "an fp8 cache is never taken for an exact one"
    teacher, student = make_model(tmp_path / "t2"), tmp_path / "s"
    student.mkdir()
    (student / "tokenizer.json").write_text("{}")
    cfg = {"teacher": {"repo": "o/T", "load": "auto"}, "model": {"family": "g"}, "data": {"train": str(tmp_path / "t.jsonl"), "general": None}}
    (tmp_path / "t.jsonl").write_text("{}\n")
    assert dj.shared_store_id(cfg, teacher, student, 8, 64, 1, 2, exact) != dj.shared_store_id(cfg, teacher, student, 8, 64, 1, 2, fp8)


def test_the_script_passes_the_flag_only_when_asked_and_never_drops_it():
    args = vc.parse_args(["--model", "m", "--data", "d", "--out", "o", "--kv-cache-dtype", "fp8"])
    assert vc.llm_kwargs(args)["kv_cache_dtype"] == "fp8"
    assert "kv_cache_dtype" not in vc.llm_kwargs(vc.parse_args(["--model", "m", "--data", "d", "--out", "o"]))
    assert "kv_cache_dtype" not in vc.OPTIONAL_LLM, "an identity that says fp8 must really be fp8"


def test_the_teacher_check_can_measure_the_cost_of_fp8(tmp_path, monkeypatch, capsys):
    import test_vllm_teacher_check as t
    from ndforge_worker import protocol, vllmteachercheck as tcheck
    teacher = make_model(tmp_path / "origin")
    ref = t.make_reference(tmp_path, teacher)
    seen = []

    def chunk(job, writer, rows_dir, meta, origin, origin_dir, first, last, k, max_seq, stop_file, **kw):
        seen.append((job["engine"].get("kv_cache_dtype"), job["engine"].get("kv_cache_dtype_teacher"), meta.get("kv_cache")))
        return t.fake_chunk()(job, writer, rows_dir, meta, origin, origin_dir, first, last, k, max_seq, stop_file)

    monkeypatch.setattr(vllmcache, "run_chunk", chunk)
    monkeypatch.setattr(protocol, "_out", __import__("sys").stdout)
    assert tcheck.run({**t.job_for(tmp_path, ref), "kv_cache_dtype": "fp8"}) == 0
    rep = json.loads((tmp_path / "heal" / "vllm_teacher_check.json").read_text())
    assert seen[0] == ("fp8", True, "fp8") and rep["kv_cache_dtype"] == "fp8" and rep["identity"]["kv_cache"] == "fp8" and "top1" in rep["metrics"] and "kl" in rep["metrics"]
    capsys.readouterr()
    tcheck.run(t.job_for(tmp_path, ref))
    assert json.loads((tmp_path / "heal" / "vllm_teacher_check.json").read_text())["kv_cache_dtype"] == "auto" and seen[1][0] in (None, "auto") and "kv_cache" not in json.loads((tmp_path / "heal" / "vllm_teacher_check.json").read_text())["identity"]


def test_llama_cache_type_is_off_by_default_and_recorded():
    base = tl.command("llama-server", "m.gguf", 1, {})
    assert "-ctk" not in base and "-fa" not in base
    q = tl.command("llama-server", "m.gguf", 1, {"cache_type": "q8_0"})
    assert q[q.index("-ctk") + 1] == "q8_0" and q[q.index("-ctv") + 1] == "q8_0" and q[q.index("-fa") + 1] == "on"
    assert tl.command("llama-server", "m.gguf", 1, {"cache_type": "q8_0", "flash_attn_args": ["-fa"]})[-1] == "-fa"
    with pytest.raises(tc.ThinkCheckError):
        tl.command("llama-server", "m.gguf", 1, {"cache_type": "q4_0"})
    k0 = tc.cache_key({"model": "m.gguf", "backend": "llama"}, "llama", tc.settings_of({}))
    k1 = tc.cache_key({"model": "m.gguf", "backend": "llama", "cache_type": "q8_0"}, "llama", tc.settings_of({}))
    assert "llama_cache_type" not in k0 and k1["llama_cache_type"] == "q8_0"


FP8_LOG = """INFO Using FLASH_ATTN attention backend
INFO GPU KV cache size: 142,282 tokens
INFO Maximum concurrency for 8,192 tokens per request: 17.37x
"""


def test_the_fp8_pass_reports_both_settings_and_a_refusal_is_a_plain_message(monkeypatch):
    log = "INFO GPU KV cache size: 71,141 tokens\nINFO Maximum concurrency for 8,192 tokens per request: 8.68x\nINFO Using FLASH_ATTN attention backend"
    s = st.kv_summary("auto", log, {"long_tokens_per_s": 500.0})
    assert s["kv_tokens"] == 71141 and s["max_concurrency"] == 8.68 and "FLASH_ATTN" in s["attention_backend"] and s["refuses_fp8"] is None
    r = st.kv_summary("fp8", "ValueError: kv_cache_dtype fp8 is not supported with the TRITON_ATTN attention backend")
    assert r["refuses_fp8"] and "not supported" in r["refuses_fp8"]

    class Fake(ve.Engine):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.stopped_n = 0

    started = []

    class Eng2:
        def __init__(self, cfg, model, **k):
            started.append(k.get("kv_cache_dtype"))
            self.port, self.exit_logs, self.commands = 4242, "", []

        def start(self):
            pass

        def wait_ready(self, *a, **k):
            raise ve.EngineError("The vLLM engine's container is not running any more (it stopped while starting). Its last lines: ValueError: kv_cache_dtype fp8 is not supported with the TRITON_ATTN attention backend")

        def logs_tail(self, n):
            return "ValueError: kv_cache_dtype fp8 is not supported with the TRITON_ATTN attention backend"

        def stop(self):
            started.append("stopped")

    monkeypatch.setattr(ve, "Engine", Eng2)
    rep = {"_full_log": log, "long_tokens_per_s": 500.0, "suite": {"think_rate": 1.0, "closed_rate": 1.0, "accuracy": 0.75}}
    first = types.SimpleNamespace(stop=lambda: started.append("first stopped"))
    st.kv_fp8_pass({"kind": "docker"}, "m", "none", 0.88, None, rep, first, [], {}, {}, lambda: False, "model")
    k = rep["kv_fp8"]
    assert started == ["first stopped", "fp8", "stopped"] and k["fallback"] == "auto" and "stays auto" in k["fp8_failed"] and "not supported" in k["fp8_failed"] and k["auto"]["kv_tokens"] == 71141
    md = st.markdown({"version": 1, "created": "t", "model": "m", "engine": {"kind": "docker"}, "note": "n", "verdict": "fail", "verdict_text": "x", "prerequisites": [], "criteria": [], "comparisons": [], "hints": [], "commands": [], "kv_fp8": k})
    assert "8-bit KV cache (fp8)" in md and "71141 tokens" in md and "stays auto" in md
