"""A round (model + LoRA) checked with vLLM: LoRA serving flags, the rank, and the fallback to the standard engine with a plain note (never a failed round)."""
import io
import json

import pytest

import test_think_check_vllm as ttv
from ndforge_worker import protocol, thinkcheck as tc, vllmengine as ve


class FakeHf:
    asked = 0

    def generate(self, requests, settings, on_result, should_stop, job):
        FakeHf.asked += 1
        suite = tc.load_suite()
        ans = {i["prompt"]: i for i in suite["items"] + suite["control"]}
        for r in requests:
            on_result({"id": r["id"], "text": "think\n</think>\nAnswer: " + str(ans[r["messages"][0]["content"]]["answer"]), "gen_tokens": 5, "truncated": False})


def adapter(tmp_path, **cfg):
    d = tmp_path / "ad"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"r": 16, **cfg}))
    (d / "adapter_model.safetensors").write_bytes(b"x")
    return d


def out(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    return lambda: [json.loads(x) for x in buf.getvalue().splitlines() if x.startswith("{")]


def base(tmp_path, f, **kw):
    return {"job": "think_check", "model": "org/student", "backend": "vllm", "quick": True, "fallback_hf": True, "adapter": str(adapter(tmp_path)), "start_timeout": 5,
            "engine": {"kind": "url", "url": f"http://127.0.0.1:{f.port}"}, **kw}


@pytest.fixture
def hf(monkeypatch):
    FakeHf.asked = 0
    real = tc.load_backend
    monkeypatch.setattr(tc, "load_backend", lambda n: FakeHf() if n == "hf" else real(n))
    return FakeHf


def test_the_lora_flags_and_the_rank(tmp_path):
    for r, want in ((8, 8), (16, 16), (48, 64), (64, 64), (100, 128), (600, 512)):
        d = tmp_path / f"a{r}"
        d.mkdir()
        (d / "adapter_config.json").write_text(json.dumps({"r": r}))
        assert ve.lora_rank_for(str(d)) == want
    assert ve.lora_rank_for(str(tmp_path / "none")) == 64
    d = tmp_path / "a16"
    cmd = ve.Engine(ve.engine_config({"engine": {"kind": "docker"}}), "org/m", adapter=str(d)).build()
    assert cmd[cmd.index("--max-lora-rank") + 1] == "16" and "--enable-lora" in cmd and "round=/models/adapter" in cmd


def test_a_round_is_checked_with_vllm_when_it_works(tmp_path, monkeypatch, hf):
    f = ttv.Fake()
    try:
        o = out(monkeypatch)
        assert tc.run(base(tmp_path, f)) == 0
        res = o()[-1]["data"]
        assert res["backend"] == "vllm" and "fallback" not in res and res["engine"]["lora"] is True and hf.asked == 0
        assert {r["model"] for r in f.requests} == {"round"}
    finally:
        f.close()


def test_a_dora_adapter_falls_back_to_hf_with_a_note(tmp_path, monkeypatch, hf):
    f = ttv.Fake()
    try:
        o = out(monkeypatch)
        job = base(tmp_path, f)
        (tmp_path / "ad" / "adapter_config.json").write_text(json.dumps({"r": 8, "use_dora": True}))
        assert tc.run(job) == 0
        res = o()[-1]["data"]
        assert res["backend"] == "hf" and "DoRA" in res["fallback"] and "standard engine" in res["fallback"] and hf.asked == 1 and f.requests == []
        assert "engine" not in res
    finally:
        f.close()


def test_a_vllm_failure_falls_back_to_hf_and_the_round_does_not_fail(tmp_path, monkeypatch, hf):
    o = out(monkeypatch)
    job = {"job": "think_check", "model": "org/student", "backend": "vllm", "quick": True, "fallback_hf": True, "adapter": str(adapter(tmp_path)), "start_timeout": 0.3,
           "engine": {"kind": "url", "url": "http://127.0.0.1:1"}}
    assert tc.run(job) == 0
    res = o()[-1]["data"]
    assert res["backend"] == "hf" and res["fallback"].startswith("The vLLM engine could not check this round") and hf.asked == 1


def test_without_the_fallback_flag_a_vllm_failure_still_raises(tmp_path):
    job = {"job": "think_check", "model": "org/student", "backend": "vllm", "quick": True, "adapter": str(adapter(tmp_path)), "start_timeout": 0.3, "engine": {"kind": "url", "url": "http://127.0.0.1:1"}}
    with pytest.raises(tc.ThinkCheckError):
        tc.run(job)
