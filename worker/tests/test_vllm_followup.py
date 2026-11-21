"""wslc list parsing (real field names), log highlights without progress bars, the backend note of the fp8 comparison."""
import json

from ndforge_worker import vllmengine, vllmselftest

BASE = {"verdict": "pass", "verdict_text": "ok", "created": "2026-11-20"}
ONE = {"Command": "vllm serve", "CreatedAt": "2026-11-20", "HealthStatus": "", "ID": "ab12cd34ef56", "Image": "vllm/vllm-openai:v0.34", "Labels": {}, "LocalVolumes": 0, "Mounts": [], "Names": "ndforge-vllm-8000",
       "Networks": [], "Platform": "linux", "Ports": "8000", "RunningFor": "2 minutes", "Size": "", "State": "running", "Status": "Up 2 minutes"}
GONE = {**ONE, "Names": "ndforge-vllm-8001", "ID": "ff00ee11dd22", "State": "exited", "Status": "Exited (0)"}


def test_the_list_is_read_as_an_array_one_object_or_one_object_per_line():
    for text in (json.dumps([ONE]), json.dumps(ONE), json.dumps(ONE) + "\n", json.dumps(ONE, indent=2), json.dumps(ONE) + "\n" + json.dumps(GONE)):
        assert vllmengine.wslc_running(text, "ndforge-vllm-8000") is True, text
        assert "ndforge-vllm-8000" in vllmengine.list_names(text, "ndforge-vllm-")
    assert vllmengine.wslc_running(json.dumps([ONE, GONE]), "ndforge-vllm-8001") is False
    assert vllmengine.wslc_running(json.dumps(ONE) + "\n" + json.dumps(GONE), "ndforge-vllm-8001") is False
    assert vllmengine.wslc_running(json.dumps(GONE), "ndforge-vllm-8000") is False, "not in the list"
    assert vllmengine.wslc_running("", "ndforge-vllm-8000") is None and vllmengine.wslc_running("[]", "x") is False
    assert vllmengine.list_names(json.dumps(ONE) + "\n" + json.dumps(GONE), "ndforge-vllm-") == ["ndforge-vllm-8000", "ndforge-vllm-8001"]
    assert vllmengine.list_names(json.dumps({**ONE, "Names": ["/ndforge-vllm-9"]}), "ndforge-vllm-") == ["ndforge-vllm-9"]


def test_a_table_is_still_understood():
    table = "ID  NAMES  STATE\nab12  ndforge-vllm-8000  running\n"
    assert vllmengine.parse_list(table) is None
    assert vllmengine.list_names(table) == ["ndforge-vllm-8000"]
    assert vllmengine.wslc_running(table, "ndforge-vllm-8000") is True


def test_progress_bars_never_push_the_cache_lines_out():
    bars = [f"Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): {i}%|#| {i}/80 [00:0{i % 10}<00:01, 10.0it/s]" for i in range(100)]
    kernels = [f"INFO attention backend line {i} flashinfer" for i in range(60)]
    log = "\n".join(kernels + bars + ["INFO GPU KV cache size: 85,196 tokens", "INFO Maximum concurrency for 3,300 tokens per request: 25.8x", "INFO init engine (profile, create kv cache, warmup model) took 61.2 seconds"])
    out = vllmengine.log_highlights(log, 50)
    assert len(out) <= 50 and not any("it/s]" in x or "Capturing CUDA graphs" in x for x in out)
    assert any("GPU KV cache size" in x for x in out) and any("init engine" in x for x in out) and any("Maximum concurrency" in x for x in out)


def test_the_fp8_section_says_when_the_backend_changed_too():
    kf = {"auto": {"backends": {"text_attention": "FLASH_ATTN"}}, "fp8": {"backends": {"text_attention": "FLASHINFER"}}}
    assert "the attention backend changed from FLASH_ATTN to FLASHINFER" in vllmselftest.backend_note(kf)
    assert vllmselftest.backend_note({"auto": {"backends": {"text_attention": "X"}}, "fp8": {"backends": {"text_attention": "X"}}}) is None
    md = vllmselftest.markdown({**BASE, "kv_fp8": {**kf, "auto": {**kf["auto"], "requested_dtype": "auto"}, "fp8": {**kf["fp8"], "requested_dtype": "fp8"}}})
    assert "Not only the cache type changed" in md


def test_the_late_list_is_labelled_and_the_hint_names_what_is_verified():
    rep = {"wslc": {"help": {"wslc list -a --format json": {"returncode": 0, "output": ""}}}}
    md = vllmselftest.markdown({**BASE, **rep})
    assert "taken after the engines were removed: empty is expected" in md
    assert "Still untested: `cp`, `exec`" in vllmselftest.WSLC_VERIFIED and "HealthStatus" in vllmselftest.WSLC_VERIFIED
