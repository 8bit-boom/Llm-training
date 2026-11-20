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
    log = "INFO GPU KV cache size: 71,141 tokens\nINFO Maximum concurrency for 8,192 tokens per request: 8.68x\n(EngineCore pid=137) INFO 11-20 13:46:41 [cuda.py:549] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'TRITON_ATTN']."
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


# ---- the 5th report's findings (the real lines of the owner's log)

REAL = """(EngineCore pid=137) INFO 11-20 13:46:40 [mm_encoder_attention.py:93] Using FLASH_ATTN backend for vit attention (MMEncoderAttention).
(EngineCore pid=137) INFO 11-20 13:46:40 Attention backends without mm_prefix support may now be selected.
(EngineCore pid=137) INFO 11-20 13:46:41 [cuda.py:549] Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN'].
(EngineCore pid=137) INFO 11-20 13:46:42 FlashInfer resolved query dtypes: q=bfloat16, decode_backend=xqa, kv_cache_dtype=torch.float8_e4m3fn, arch=sm120
(EngineCore pid=137) INFO 11-20 13:46:43 Using FlashInfer GDN prefill kernel
(EngineCore pid=137) INFO 11-20 13:46:43 Using MarlinLinearKernel for CompressedTensorsWNA16
(EngineCore pid=137) INFO 11-20 13:46:44 Using fp8 data type to store kv cache. It reduces the GPU memory footprint
(EngineCore pid=137) INFO 11-20 13:48:00 GPU KV cache size: 85,196 tokens
(EngineCore pid=137) INFO 11-20 13:48:00 Maximum concurrency for 8,192 tokens per request: 10.4x
"""


def test_the_backend_lines_are_the_real_ones_and_the_vision_encoder_is_kept_apart():
    b = st.backend_facts(REAL)
    assert b["text_attention"] == "FLASHINFER" and b["decode_backend"] == "xqa" and b["flashinfer_kv_dtype"] == "torch.float8_e4m3fn" and b["gdn_prefill_kernel"] == "FlashInfer" and b["linear_kernel"] == "MarlinLinearKernel"
    assert "vit attention" in b["vision_encoder_line"] and b["log_says_fp8_kv"] is True
    assert b["summary"] == "attention backend (text layers): FLASHINFER (xqa decode), GDN prefill kernel: FlashInfer, linear kernel: Marlin, KV dtype in the log: torch.float8_e4m3fn"
    assert "mm_prefix" not in b["summary"] and "mm_prefix" not in (st.kv_summary("fp8", REAL, requested="fp8")["attention_backend"])
    exact = st.backend_facts(REAL.replace("Using fp8 data type to store kv cache", "x").replace("torch.float8_e4m3fn", "auto"))
    assert "KV dtype in the log: auto" in exact["summary"]


def make_world(monkeypatch, fp8_in_auto_log=True):
    """A fake wslc: remembers what is running, answers list/inspect/logs/stop/remove."""
    w = types.SimpleNamespace(running=set(), calls=[], started=[])

    def run(cmd, **kw):
        w.calls.append(list(cmd))
        out = b""
        if cmd[1] == "list":
            out = ("NAME STATE\n" + "".join(f"{n} running\n" for n in sorted(w.running))).encode() if "--format" not in cmd else b""     # the JSON listing prints nothing, like the owner's wslc
        elif cmd[1] == "inspect":
            out = (b'{"Name": "x", "State": {"Running": true}}' if cmd[2] in w.running else b"Object not found")
        elif cmd[1] == "logs":
            out = REAL.encode() if cmd[-1] in w.running or True else b""
        elif cmd[1] == "run":
            n = cmd[cmd.index("--name") + 1]
            w.running.add(n)
        elif cmd[1] in ("stop", "remove"):
            w.running.discard(cmd[-1]) if cmd[1] == "remove" else None
        return types.SimpleNamespace(returncode=0, stdout=out, stderr=b"")

    monkeypatch.setattr(ve, "_run", run)
    monkeypatch.setattr(st, "_run", lambda cmd, timeout=600: {"cmd": cmd, "returncode": 0, "output": run(cmd).stdout.decode()})
    monkeypatch.setattr(ve, "_popen", lambda cmd, **kw: (w.started.append(cmd), types.SimpleNamespace(poll=lambda: None, returncode=None, terminate=lambda: None, wait=lambda t=None: 0))[1])
    return w


def test_the_baseline_engine_of_the_pass_is_started_with_auto_whatever_the_setting_says():
    cfg = ve.engine_config({"engine": {"kind": "wslc", "kv_cache_dtype": "fp8"}})
    assert ve.Engine(cfg, "m").kv_cache_dtype == "fp8", "the setting alone"
    first = ve.Engine(cfg, "m", kv_cache_dtype="auto")
    assert first.kv_cache_dtype == "auto" and "--kv-cache-dtype" not in first.build()
    second = ve.Engine(cfg, "m", kv_cache_dtype="fp8")
    assert second.build()[second.build().index("--kv-cache-dtype") + 1] == "fp8"
    import inspect
    assert 'kv_cache_dtype="auto" if job.get("kv_fp8_test") else None' in inspect.getsource(st.run)


def test_an_auto_run_whose_log_says_fp8_gets_no_comparison():
    rep = {"_full_log": REAL, "long_tokens_per_s": 500.0, "suite": {}}

    class Dead:
        def stop(self):
            pass

    class Eng:
        def __init__(self, *a, **k):
            self.port, self.exit_logs, self.commands = 0, "", []

        def start(self):
            raise ve.EngineError("no")

        def logs_tail(self, n):
            return ""

        def stop(self):
            pass

    import ndforge_worker.vllmengine as vmod
    old = vmod.Engine
    vmod.Engine = Eng
    try:
        st.kv_fp8_pass({"kind": "docker"}, "m", "none", 0.88, None, rep, Dead(), [], {}, {}, lambda: False, "model")
    finally:
        vmod.Engine = old
    k = rep["kv_fp8"]
    assert "comparison_refused" in k and "says it used the 8-bit KV cache" in k["comparison_refused"] and k["auto"]["log_confirms_fp8"] is True and k["auto"]["requested_dtype"] == "auto"
    md = st.markdown({"version": 1, "created": "t", "model": "m", "engine": {"kind": "docker"}, "note": "n", "verdict": "pass", "verdict_text": "x", "prerequisites": [], "criteria": [], "comparisons": [], "hints": [], "commands": [], "kv_fp8": k,
                      "start_command": ["wslc", "run", "--kv-cache-dtype", "fp8"], "kv_requested": {"first": "auto", "second": "fp8"}})
    assert "no comparison is printed" in md and "KV cache type passed to this engine: auto" in md and "the log CONFIRMS an fp8 KV cache" in md


def test_the_second_sample_follows_health_uses_the_second_name_and_a_wslc_list_that_is_empty_is_a_finding(monkeypatch):
    w = make_world(monkeypatch)
    w.running.add("ndforge-vllm-2")
    rep: dict = {}
    st.list_sample(rep, {"kind": "wslc"}, "right after /health answered (the fp8 engine)", "ndforge-vllm-2")
    assert rep["wslc"]["list_samples"][0]["inspect"]["output"].startswith("{")
    assert any(c[:2] == ["wslc", "inspect"] and c[2] == "ndforge-vllm-2" for c in w.calls)
    w.running.clear()
    r2: dict = {}
    monkeypatch.setattr(st, "_run", lambda cmd, timeout=600: {"cmd": cmd, "returncode": 0, "output": "" if cmd[1] == "list" else "State: running"})
    st.list_sample(r2, {"kind": "wslc"}, "after health", "ndforge-vllm-2")
    assert any("`wslc list` is empty while the container ndforge-vllm-2 runs" in f for f in r2["wslc"]["findings"])


def test_leftover_containers_are_found_and_removed_by_name_and_the_fp8_commands_are_printed(monkeypatch):
    w = make_world(monkeypatch)
    w.running.update({"ndforge-vllm-52038", "ndforge-vllm-63952", "other"})
    eng = types.SimpleNamespace(kind="wslc", cfg=ve.engine_config({"engine": {"kind": "wslc"}}))
    lo = st.remove_leftovers(eng)
    assert lo["found"] == ["ndforge-vllm-52038", "ndforge-vllm-63952"] and lo["removed"] == lo["found"] and "other" in w.running
    assert ["wslc", "remove", "-f", "ndforge-vllm-52038"] in w.calls
    assert st.remove_leftovers(eng)["found"] == [], "nothing left on the second look"
    md = st.markdown({"version": 1, "created": "t", "model": "m", "engine": {"kind": "wslc"}, "note": "n", "verdict": "pass", "verdict_text": "x", "prerequisites": [], "criteria": [], "comparisons": [], "hints": [], "commands": [["wslc", "stop", "a"]],
                      "fp8_commands": [["wslc", "run", "x"], ["wslc", "stop", "-t", "30", "ndforge-vllm-52038"], ["wslc", "remove", "-f", "ndforge-vllm-52038"]], "fp8_engine_name": "ndforge-vllm-52038", "leftovers": lo})
    assert "Commands of the fp8 engine (ndforge-vllm-52038)" in md and "wslc remove -f ndforge-vllm-52038" in md and "Containers left behind after the stops: ndforge-vllm-52038, ndforge-vllm-63952 (removed:" in md


def test_the_fp8_engines_own_log_lines_are_printed():
    fp8 = st.kv_summary("fp8", REAL, requested="fp8")
    assert any("Using FLASHINFER attention backend" in h for h in fp8["highlights"]) and fp8["log_confirms_fp8"]
    md = st.markdown({"version": 1, "created": "t", "model": "m", "engine": {"kind": "docker"}, "note": "n", "verdict": "pass", "verdict_text": "x", "prerequisites": [], "criteria": [], "comparisons": [], "hints": [], "commands": [],
                      "kv_fp8": {"auto": st.kv_summary("auto", "INFO GPU KV cache size: 71,141 tokens", requested="auto"), "fp8": fp8, "text": "KV cache: 71,141 tokens with auto, 85,196 with fp8 (1.20 times)."}})
    assert "Kernel / backend / cache lines of the fp8 engine:" in md and "Using FLASHINFER attention backend" in md and "1.20 times" in md
