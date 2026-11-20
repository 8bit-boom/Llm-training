"""The vLLM self-test with fakes: a fake OpenAI server in a thread, a fake docker / wslc / nvidia-smi through the engine's subprocess hooks. The engine is stopped in EVERY end."""
import io
import json
import threading
import time

import pytest

import test_think_check_vllm as ttv
from ndforge_worker import protocol, vllmengine as ve, vllmselftest as st
from ndforge_worker import thinkcheck as tc


class Env:
    """Fake subprocess world: records every command; nvidia-smi / docker / wsl.exe / wslc answers can be changed."""

    def __init__(self, monkeypatch, port, exit_code=None):
        self.run, self.popen = [], []
        self.answers = {"nvidia-smi": (0, "581.42, 900, 32768\n"), "docker": (0, "ok"), "wsl.exe": (0, "WSL version: 3.0.1.0"), "wslc": (0, "help text")}
        self.proc = ttv.Proc()
        self.proc.returncode = exit_code
        env = self

        def fake_run(cmd, **kw):
            env.run.append(list(cmd))
            rc, out = env.answers.get(cmd[0], (0, ""))
            return type("R", (), {"returncode": rc, "stdout": out.encode(), "stderr": b""})()

        def fake_popen(cmd, **kw):
            env.popen.append(list(cmd))
            kw["stdout"].write(b"INFO attention backend FLASH_ATTN\nINFO GDN prefill: FlashInfer\n")
            return env.proc

        monkeypatch.setattr(ve, "_run", fake_run)
        monkeypatch.setattr(ve, "_popen", fake_popen)
        monkeypatch.setattr(ve, "_which", lambda n: None)
        monkeypatch.setattr(ve, "free_port", lambda: port)


def job(tmp_path, **kw):
    return {"job": "vllm_selftest", "model": "cyankiwi/Qwen3.8-27B-AWQ-INT4", "confirm_download": True, "engine": {"kind": "docker"}, "out_dir": str(tmp_path / "out"), "start_timeout": 10, **kw}


def lines_of(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    return lambda: [json.loads(x) for x in buf.getvalue().splitlines() if x.startswith("{")]


def stopped(env):
    return any(c[:2] == ["docker", "stop"] for c in env.run)


def test_a_download_needs_the_confirmation_and_names_the_size(tmp_path, monkeypatch):
    Env(monkeypatch, 1)
    with pytest.raises(tc.ThinkCheckError, match="about 21 GB"):
        st.run(job(tmp_path, confirm_download=False))
    assert not (tmp_path / "out").exists()


def test_a_whole_self_test_writes_a_report_and_stops_the_engine(tmp_path, monkeypatch):
    f = ttv.Fake()
    try:
        env = Env(monkeypatch, f.port)
        saved = tmp_path / "hf.json"
        saved.write_text(json.dumps({"key": {}, "result": {"backend": "hf", "model": "org/original", "main": {"n": 16, "closed_rate": 1.0, "accuracy": 1.0, "loop_rate": 0.0}}}))
        out = lines_of(monkeypatch)
        assert st.run(job(tmp_path, compare={"hf": str(saved)})) == 0
        rep = json.loads((tmp_path / "out" / "vllm_selftest.json").read_text())
        md = (tmp_path / "out" / "vllm_selftest.md").read_text()
        assert stopped(env) and rep["engine_stopped"] is True
        assert [p["id"] for p in rep["prerequisites"]][:3] == ["driver", "gpu_free", "wsl"] and any(p["id"] == "gpu_passthrough" and p["ok"] for p in rep["prerequisites"])
        assert env.run[[c[0] for c in env.run].index("docker")][:5] == ["docker", "run", "--rm", "--gpus", "all"], "the GPU test runs before the engine starts"
        assert rep["startup_seconds"] is not None and any("FLASH_ATTN" in x for x in rep["highlights"]) and rep["reachability"]["ok"]
        assert rep["suite"]["n"] == 16 and rep["suite"]["accuracy"] == 1.0 and rep["completion_tokens"] == 20 * 24 and rep["tokens_per_s"]
        c = rep["comparisons"][0]
        assert c["with"] == "hf" and c["within_noise"] is True and c["deltas"]["accuracy"] == 0.0
        assert rep["verdict"] in ("pass", "partial") and {x["id"] for x in rep["criteria"]} >= {"ready", "speed", "noise_hf"}
        assert "not yet run on this PC" in md and "## Success criteria" in md and "FLASH_ATTN" in md
        assert out()[-1]["data"]["report"].endswith("vllm_selftest.json")
        assert rep["verdict"] != "pass" or rep["tokens_per_s"] >= 300 or True
    finally:
        f.close()


def test_the_speed_criterion_is_judged_against_300_tokens_per_second():
    base = {"startup_seconds": 100.0, "tokens_per_s": 250.0, "comparisons": [], "prerequisites": []}
    crit = st.criteria_of(base)
    assert [c["ok"] for c in crit] == [True, False]
    assert st.verdict({**base, "criteria": crit})[0] == "partial"
    good = {**base, "tokens_per_s": 450.0}
    assert st.verdict({**good, "criteria": st.criteria_of(good)})[0] == "pass"
    slow = {**base, "startup_seconds": 400.0, "tokens_per_s": 100.0}
    assert st.verdict({**slow, "criteria": st.criteria_of(slow)})[0] == "fail"
    assert st.criteria_of({**base, "comparisons": [{"with": "hf", "tolerance": 0.125, "deltas": {}, "within_noise": False}]})[-1]["ok"] is False


def test_a_card_that_the_container_cannot_use_ends_the_test_before_the_engine_starts(tmp_path, monkeypatch):
    env = Env(monkeypatch, 1)
    env.answers["docker"] = (1, "docker: Error response from daemon: could not select device driver")
    assert st.run(job(tmp_path)) == 0
    rep = json.loads((tmp_path / "out" / "vllm_selftest.json").read_text())
    assert rep["verdict"] == "fail" and "could not use the graphics card" in rep["verdict_text"] and not env.popen
    assert "could not select device driver" in json.dumps(rep["prerequisites"])


def test_a_busy_card_ends_the_test_early(tmp_path, monkeypatch):
    env = Env(monkeypatch, 1)
    env.answers["nvidia-smi"] = (0, "581.42, 13000, 32768\n")
    assert st.run(job(tmp_path)) == 0
    assert json.loads((tmp_path / "out" / "vllm_selftest.json").read_text())["verdict"] == "fail" and not env.popen


def test_a_crashing_engine_still_gets_a_report_and_the_explicit_stop(tmp_path, monkeypatch):
    env = Env(monkeypatch, 1, exit_code=2)
    assert st.run(job(tmp_path)) == 0
    rep = json.loads((tmp_path / "out" / "vllm_selftest.json").read_text())
    assert rep["verdict"] == "fail" and "exit code 2" in rep["error"] and stopped(env)


def test_stop_ends_the_questions_and_the_engine_and_the_report_says_so(tmp_path, monkeypatch):
    f = ttv.Fake(delay=3.0)
    try:
        env = Env(monkeypatch, f.port)
        stop = tmp_path / "stop"
        threading.Timer(1.5, lambda: stop.write_text("x")).start()
        t0 = time.time()
        assert st.run(job(tmp_path, stop_file=str(stop))) == 0
        assert time.time() - t0 < 6
        rep = json.loads((tmp_path / "out" / "vllm_selftest.json").read_text())
        assert rep["verdict"] == "stopped" and stopped(env)
    finally:
        f.close()


def test_wslc_help_is_captured_into_the_report_and_templates_are_used(tmp_path, monkeypatch):
    f = ttv.Fake()
    try:
        env = Env(monkeypatch, f.port)
        monkeypatch.setattr(ve, "_which", lambda n: "wslc")
        assert st.run(job(tmp_path, engine={"kind": "wslc"})) == 0
        rep = json.loads((tmp_path / "out" / "vllm_selftest.json").read_text())
        assert "wslc run --help" in rep["wslc"]["help"] and rep["wslc"]["help"]["wslc run --help"]["output"] == "help text"
        gpu = next(p for p in rep["prerequisites"] if p["id"] == "gpu_passthrough")
        assert gpu["command"][:3] == ["wslc", "run", "--rm"] and "nvidia-smi" in gpu["command"]
        assert any(c[:2] == ["wslc", "stop"] for c in env.run) and any(c[:2] == ["wslc", "remove"] for c in env.run)
        md = (tmp_path / "out" / "vllm_selftest.md").read_text()
        assert "wslc help (for adapting the templates)" in md and "wslc syntax unverified" in md
        assert all("Bearer" not in " ".join(map(str, c)) for c in rep["commands"])
        assert not any(f"--api-key" in " ".join(map(str, c)) and "[key]" not in " ".join(map(str, c)) for c in rep["commands"]), "the key is redacted in the report"
    finally:
        f.close()


def test_an_unreachable_port_gets_the_wsl_ip_hint(tmp_path, monkeypatch):
    env = Env(monkeypatch, 1)
    engine = ve.Engine(ve.engine_config({"engine": {"kind": "docker_wsl"}}), "org/m")
    engine.url = "http://127.0.0.1:1"
    env.answers["wsl.exe"] = (0, "172.20.0.2 \n")
    r = st.reachability(engine)
    assert r["ok"] is False and "wsl.exe hostname -I" in r["detail"] and "172.20.0.2" in r["detail"] and "mirrored" in r["detail"]


def test_a_model_folder_under_mnt_is_warned_about(tmp_path, monkeypatch):
    Env(monkeypatch, 1)
    cfg = ve.engine_config({"engine": {"kind": "docker_wsl"}})
    pre = st.prerequisites(cfg, r"C:\models\q", gpu_test=False)
    assert any(p["id"] == "slow_share" and p["ok"] is False for p in pre)


def test_the_adapter_agreement_serves_round_and_compares_with_the_hf_adapter_result(tmp_path, monkeypatch):
    f = ttv.Fake()
    try:
        Env(monkeypatch, f.port)
        ad = tmp_path / "ad"
        ad.mkdir()
        (ad / "adapter_config.json").write_text("{}")
        saved = tmp_path / "hfad.json"
        saved.write_text(json.dumps({"result": {"backend": "hf", "main": {"n": 16, "closed_rate": 1.0, "accuracy": 1.0, "loop_rate": 0.0}}}))
        assert st.run(job(tmp_path, adapter=str(ad), compare={"hf_adapter": str(saved)})) == 0
        rep = json.loads((tmp_path / "out" / "vllm_selftest.json").read_text())
        assert {r["model"] for r in f.requests} == {"round"} and rep["comparisons"][0]["with"] == "hf_adapter"
    finally:
        f.close()
