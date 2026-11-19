"""The vLLM backend of the thinking check against a fake OpenAI-style server running in a thread (no Docker, WSL or vLLM): the request body, parallel asking, the Stop file,
a health timeout, a crash tail, the explicit stop in every end, the cache key, the plan, adapters. Runs on every OS."""
import http.server
import io
import json
import threading
import time

import pytest

from ndforge_worker import protocol
from ndforge_worker import thinkcheck as tc
from ndforge_worker import thinkcheck_vllm as tv
from ndforge_worker import vllmengine as ve

SUITE = tc.load_suite()
ANSWERS = {i["prompt"]: i for i in SUITE["items"] + SUITE["control"]}


class Fake:
    def __init__(self, mode="ok", delay=0.0):
        self.mode, self.delay = mode, delay
        self.requests, self.live, self.peak = [], 0, 0
        self.lock = threading.Lock()
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_GET(self):
                code = 503 if fake.mode == "loading" else 200
                self.send_response(code)
                self.end_headers()
                self.wfile.write(b'{"version": "0.34.0"}' if self.path == "/version" else b"{}")

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake.lock:
                    fake.requests.append(body)
                    fake.live += 1
                    fake.peak = max(fake.peak, fake.live)
                time.sleep(fake.delay)
                item = ANSWERS[body["messages"][0]["content"]]
                think = body["chat_template_kwargs"]["enable_thinking"]
                if fake.mode == "split" and think:
                    msg = {"reasoning": "work it out", "content": "Answer: " + str(item["answer"])}
                elif think:
                    msg = {"content": "work it out\n</think>\n\nAnswer: " + str(item["answer"])}
                else:
                    msg = {"content": "Answer: " + str(item["answer"])}
                out = json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {"completion_tokens": 20}}).encode()
                with fake.lock:
                    fake.live -= 1
                self.send_response(200)
                self.end_headers()
                self.wfile.write(out)

        class Srv(http.server.ThreadingHTTPServer):
            request_queue_size = 128                        # 40 questions connect at once

        self.srv = Srv(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


def lines_of(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    return lambda: [json.loads(x) for x in buf.getvalue().splitlines() if x.startswith("{")]


class Proc:
    returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, t=None):
        return self.returncode


@pytest.fixture
def fake():
    f = Fake()
    yield f
    f.close()


def url_job(f, **kw):
    return {"job": "think_check", "model": "org/some-model", "backend": "vllm", "start_timeout": 10, "engine": {"kind": "url", "url": f"http://127.0.0.1:{f.port}"}, **kw}


@pytest.fixture
def docker_fake(monkeypatch):
    """kind docker with a fake `docker`: records every command; the 'container' is the thread server."""
    st = {"popen": [], "run": [], "proc": Proc(), "port": 0}
    monkeypatch.setattr(ve, "_popen", lambda cmd, **kw: (st["popen"].append(cmd), kw["stdout"].write(b"INFO attention backend: FLASH_ATTN\n"), st["proc"])[2])
    monkeypatch.setattr(ve, "_run", lambda cmd, **kw: st["run"].append(cmd))
    monkeypatch.setattr(ve, "free_port", lambda: st["port"])
    return st


def test_a_whole_check_with_the_request_body_and_parallel_asking(monkeypatch):
    f = Fake(delay=0.2)
    try:
        out = lines_of(monkeypatch)
        assert tc.run(url_job(f)) == 0
        res = out()[-1]["data"]
        assert res["backend"] == "vllm" and res["main"]["n"] == 32 and res["control"]["n"] == 8 and res["main"]["accuracy"] == 1.0 and res["main"]["closed_rate"] == 1.0
        assert res["engine"]["vllm"] == "unknown" and res["engine"]["quant"] == "none" and res["engine"]["lora"] is False and res["engine"]["server_version"] == "0.34.0"
        assert res["engine"]["completion_tokens"] == 800 and res["engine"]["kind"] == "url"
        assert len(f.requests) == 40 and f.peak > 10, "all questions are asked at once"
        r = next(x for x in f.requests if x["chat_template_kwargs"]["enable_thinking"])
        assert (r["model"], r["temperature"], r["top_p"], r["top_k"], r["seed"], r["max_tokens"], r["stream"]) == ("model", 0.6, 0.95, 20, 3407, 3072, False)
        assert (r["repetition_penalty"], r["presence_penalty"], r["min_p"]) == (1.0, 0.0, 0.0)
    finally:
        f.close()


def test_a_server_that_splits_the_reasoning_gives_the_same_numbers(monkeypatch):
    f = Fake("split")
    try:
        out = lines_of(monkeypatch)
        assert tc.run(url_job(f)) == 0
        assert out()[-1]["data"]["main"]["accuracy"] == 1.0
    finally:
        f.close()


def test_docker_lifecycle_starts_waits_asks_and_stops_explicitly(fake, docker_fake, monkeypatch):
    docker_fake["port"] = fake.port
    out = lines_of(monkeypatch)
    job = {"job": "think_check", "model": "org/some-model", "backend": "vllm", "start_timeout": 10, "engine": {"kind": "docker"}}
    assert tc.run(job) == 0
    res = out()[-1]["data"]
    assert docker_fake["popen"][0][:2] == ["docker", "run"] and "ndforge-vllm-%d" % fake.port in docker_fake["popen"][0]
    assert docker_fake["run"][0][:2] == ["docker", "stop"], "the explicit stop must run when the check is done"
    assert res["engine"]["vllm"] == "0.34.0" and res["engine"]["highlights"] == ["INFO attention backend: FLASH_ATTN"] and res["engine"]["ready_seconds"] is not None


def test_the_stop_file_stops_the_questions_and_the_engine(tmp_path, docker_fake, monkeypatch):
    f = Fake(delay=3.0)
    try:
        docker_fake["port"] = f.port
        stop = tmp_path / "stop"
        threading.Timer(1.0, lambda: stop.write_text("x")).start()
        out = lines_of(monkeypatch)
        t0 = time.time()
        assert tc.run({"job": "think_check", "model": "org/m", "backend": "vllm", "engine": {"kind": "docker"}, "stop_file": str(stop)}) == 0
        assert time.time() - t0 < 3.0
        assert out()[-1]["data"]["stopped"] is True
        assert docker_fake["run"] and docker_fake["run"][0][:2] == ["docker", "stop"]
    finally:
        f.close()


def test_health_timeout_is_plain_and_stops_the_engine(docker_fake):
    f = Fake("loading")
    try:
        docker_fake["port"] = f.port
        with pytest.raises(tc.ThinkCheckError, match="did not become ready"):
            tc.run({"job": "think_check", "model": "org/m", "backend": "vllm", "engine": {"kind": "docker"}, "start_timeout": 0.5})
        assert docker_fake["run"][0][:2] == ["docker", "stop"]
    finally:
        f.close()


def test_a_crash_shows_the_last_lines_and_stops_the_engine(docker_fake, monkeypatch):
    docker_fake["port"] = 1
    proc = Proc()
    proc.returncode = 2
    docker_fake["proc"] = proc
    monkeypatch.setattr(ve, "_popen", lambda cmd, **kw: (kw["stdout"].write(b"CUDA out of memory\n"), proc)[1])
    with pytest.raises(tc.ThinkCheckError, match="exit code 2.*CUDA out of memory"):
        tc.run({"job": "think_check", "model": "org/m", "backend": "vllm", "engine": {"kind": "docker"}, "start_timeout": 5})
    assert docker_fake["run"][0][:2] == ["docker", "stop"]


def test_a_failing_question_stops_the_engine(docker_fake):
    docker_fake["port"] = 1                                 # nothing listens: health wait would time out
    with pytest.raises(tc.ThinkCheckError):
        tc.run({"job": "think_check", "model": "org/m", "backend": "vllm", "engine": {"kind": "docker"}, "start_timeout": 0.3})
    assert docker_fake["run"]


def test_a_gguf_file_and_a_bad_engine_are_refused_plainly(tmp_path):
    g = tmp_path / "m.gguf"
    g.write_bytes(b"GGUF")
    with pytest.raises(tc.ThinkCheckError, match="model folder or a Hub id"):
        tc.run({"job": "think_check", "model": str(g), "backend": "vllm"})
    with pytest.raises(tc.ThinkCheckError, match="Unknown vLLM engine kind"):
        tc.run({"job": "think_check", "model": "org/m", "backend": "vllm", "engine": {"kind": "podman"}})


def test_the_cache_key_separates_vllm_and_leaves_old_keys_alone(tmp_path):
    base = {"model": "org/x", "load": "auto"}
    s = tc.settings_of({})
    hf, llama = tc.cache_key(base, "hf", s), tc.cache_key(base, "llama", s)
    assert "engine" not in hf and "engine" not in llama
    assert hf == {"format": 1, "model": "repo:org/x", "backend": "hf", "load": "auto", "suite_version": 1, "settings": s}
    v = tc.cache_key({**base, "engine": {"kind": "docker"}}, "vllm", s)
    assert v["backend"] == "vllm" and v["engine"] == {"vllm": "0.34.0", "quant": "none", "lora": False}
    assert tc.cache_key({**base, "engine": {"kind": "docker", "quant": "fp8"}}, "vllm", s)["engine"]["quant"] == "fp8"
    assert tc.cache_key({**base, "engine": {"kind": "docker", "image": "vllm/vllm-openai:v0.35.0"}}, "vllm", s) != v


def test_a_cached_vllm_result_is_used_the_second_time(fake, monkeypatch, tmp_path):
    out = lines_of(monkeypatch)
    job = url_job(fake, cache_file=str(tmp_path / "c.json"))
    assert tc.run(job) == 0
    n = len(fake.requests)
    assert tc.run(url_job(fake, cache_file=str(tmp_path / "c.json"))) == 0
    assert len(fake.requests) == n and out()[-1]["data"]["cached"] is True


def test_the_plan_knows_the_start_up_and_the_speed():
    hf, vl = tc.plan({"backend": "hf"}), tc.plan({"backend": "vllm"})
    assert "startup_minutes" not in hf and vl["startup_minutes"] == 3.5 and tc.EST_TOKENS_PER_SECOND["vllm"] == 400.0
    assert vl["est_minutes"] > vl["startup_minutes"] and "estimates from a research report" in vl["note"]
    assert tc.plan({"backend": "llama"}).keys() == hf.keys()


def lora_dir(tmp_path, **cfg):
    d = tmp_path / "ad"
    d.mkdir(exist_ok=True)
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    return d


def test_an_adapter_is_served_as_round_and_changes_the_key(fake, tmp_path, monkeypatch):
    out = lines_of(monkeypatch)
    ad = lora_dir(tmp_path, r=16)
    (ad / "adapter_model.safetensors").write_bytes(b"x")
    assert tc.run(url_job(fake, adapter=str(ad), quick=True)) == 0
    res = out()[-1]["data"]
    assert res["engine"]["lora"] is True and {r["model"] for r in fake.requests} == {"round"}
    assert tc.cache_key({"model": "org/x", "adapter": str(ad)}, "vllm", {})["engine"]["lora"] is True


def test_dora_and_loftq_adapters_keep_raising_a_plain_message(fake, tmp_path):
    with pytest.raises(tc.ThinkCheckError, match="DoRA"):
        tc.run(url_job(fake, adapter=str(lora_dir(tmp_path, use_dora=True))))
    with pytest.raises(tc.ThinkCheckError, match="LoftQ"):
        tc.run(url_job(fake, adapter=str(lora_dir(tmp_path, init_lora_weights="loftq"))))
    assert fake.requests == []


def test_llama_still_refuses_an_adapter(tmp_path):
    with pytest.raises(tc.ThinkCheckError, match="LoRA"):
        tc.run({"job": "think_check", "model": "org/x", "backend": "llama", "adapter": str(lora_dir(tmp_path))})
