"""The vLLM launcher with a mocked subprocess: built command lines, Windows -> WSL paths, the explicit stop (always called), the readiness wait.
The real launch needs the owner's PC (nothing here starts Docker, WSL or vLLM)."""
import http.server
import threading

import pytest

from ndforge_worker import vllmengine as ve


class FakeProc:
    def __init__(self, exit_code=None):
        self.returncode = exit_code
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, t=None):
        return self.returncode


@pytest.fixture
def fakes(monkeypatch):
    calls = {"popen": [], "run": [], "proc": FakeProc()}

    def popen(cmd, **kw):
        calls["popen"].append(cmd)
        kw["stdout"].write(b"INFO attention backend FLASH_ATTN\nnoise\nINFO GDN prefill kernel: FlashInfer\n")
        return calls["proc"]

    def run(cmd, **kw):
        calls["run"].append(cmd)
        return None

    monkeypatch.setattr(ve, "_popen", popen)
    monkeypatch.setattr(ve, "_run", run)
    return calls


def server(status=200):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(status if self.path == "/health" else 200)
            self.end_headers()
            self.wfile.write(b'{"version": "0.34.0"}')

    s = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def test_wsl_path_translation():
    assert ve.wsl_path(r"C:\Users\me\models\q") == "/mnt/c/Users/me/models/q"
    assert ve.wsl_path("D:/x/y/") == "/mnt/d/x/y"
    assert ve.wsl_path("C:\\") == "/mnt/c"
    assert ve.wsl_path(r"\\wsl.localhost\Ubuntu\home\me\m") == "/home/me/m"
    assert ve.wsl_path("/home/me/m") == "/home/me/m"
    assert ve.is_local_path(r"C:\m") and ve.is_local_path("/m") and not ve.is_local_path("cyankiwi/Qwen3.8-27B-AWQ-INT4")


def test_image_version():
    assert ve.image_version("vllm/vllm-openai:v0.34.0") == "0.34.0"
    assert ve.image_version("vllm/vllm-openai:latest") == "latest"
    assert ve.image_version("localhost:5000/vllm") == "unknown"


def test_serve_args_all_flags():
    a = ve.serve_args("/models/base")
    assert a[:2] == ["--model", "/models/base"]
    for flag in ("--language-model-only", "--generation-config", "vllm", "--served-model-name", "model", "--max-model-len", "8192", "--max-num-seqs", "40", "0.88"):
        assert flag in a
    assert "--quantization" not in a and "--enable-lora" not in a and "--reasoning-parser" not in a
    f = ve.serve_args("m", quant="fp8", adapter="/a")
    assert f[f.index("--quantization") + 1] == "fp8"
    assert f[f.index("--lora-modules") + 1] == "round=/a" and f[f.index("--max-lora-rank") + 1] == "64" and "--enable-lora" in f
    assert "--quantization" not in ve.serve_args("m", quant="awq:x")


def test_engine_config_defaults_and_errors():
    c = ve.engine_config({"engine": {"kind": "docker"}})
    assert c["image"] == "vllm/vllm-openai:v0.34.0" and c["version"] == "0.34.0"
    assert ve.engine_config({})["kind"] == "docker"
    with pytest.raises(ve.EngineError):
        ve.engine_config({"engine": {"kind": "podman"}})
    with pytest.raises(ve.EngineError):
        ve.engine_config({"engine": {"kind": "url"}})
    with pytest.raises(ve.EngineError, match="ndforge-vllm-"):
        ve.engine_config({"engine": {"hf_volume": "my-own-volume"}})
    assert ve.engine_identity({"engine": {"kind": "docker"}}, "fp8", True) == {"vllm": "0.34.0", "quant": "fp8", "lora": True}


def test_docker_command_hf_id_uses_named_volume(fakes, monkeypatch):
    monkeypatch.setattr(ve, "free_port", lambda: 18123)
    e = ve.Engine(ve.engine_config({}), "cyankiwi/Qwen3.8-27B-AWQ-INT4")
    cmd = e.build()
    assert cmd[:2] == ["docker", "run"] and "--rm" not in cmd and "--ipc=host" in cmd
    assert cmd[cmd.index("--gpus") + 1] == "all" and cmd[cmd.index("--name") + 1] == "ndforge-vllm-18123"
    assert cmd[cmd.index("-p") + 1] == "127.0.0.1:18123:8000"
    assert cmd[cmd.index("-v") + 1] == "ndforge-vllm-hf:/root/.cache/huggingface:rw"
    assert cmd[cmd.index("vllm/vllm-openai:v0.34.0") + 1:][:2] == ["--model", "cyankiwi/Qwen3.8-27B-AWQ-INT4"]


def test_docker_command_local_folder_and_adapter_are_mounted(monkeypatch):
    monkeypatch.setattr(ve, "free_port", lambda: 1)
    e = ve.Engine(ve.engine_config({}), r"C:\models\q", adapter=r"C:\adapters\r1", quant="fp8")
    cmd = e.build()
    assert r"C:\models\q:/models/base:ro" in cmd and r"C:\adapters\r1:/models/adapter:ro" in cmd
    assert "round=/models/adapter" in cmd and "/models/base" in cmd and "fp8" in cmd


def test_wsl_command(monkeypatch):
    monkeypatch.setattr(ve, "free_port", lambda: 18200)
    e = ve.Engine(ve.engine_config({"engine": {"kind": "wsl", "distro": "Ubuntu-24.04"}}), r"C:\models\q", adapter=r"D:\a")
    cmd = e.build()
    assert cmd[:6] == ["wsl.exe", "-d", "Ubuntu-24.04", "--", "~/.ndforge/vllm/bin/vllm", "serve"]
    assert cmd[6] == "/mnt/c/models/q" and "round=/mnt/d/a" in cmd
    assert cmd[-4:] == ["--host", "127.0.0.1", "--port", "18200"] and "--model" not in cmd


def test_stop_commands():
    d = ve.engine_config({})
    assert ve.stop_commands(d, 5) == [["docker", "stop", "-t", "20", "ndforge-vllm-5"], ["docker", "rm", "-f", "ndforge-vllm-5"]]
    w = ve.engine_config({"engine": {"kind": "wsl", "distro": "U"}})
    assert ve.stop_commands(w, 7) == [["wsl.exe", "-d", "U", "--", "pkill", "-f", "vllm serve.*--port 7"]]
    assert ve.stop_commands(ve.engine_config({"engine": {"kind": "url", "url": "http://x"}}), 1) == []


def test_stop_runs_even_when_the_process_already_ended(fakes, monkeypatch):
    monkeypatch.setattr(ve, "free_port", lambda: 9)
    fakes["proc"] = FakeProc(exit_code=1)
    e = ve.Engine(ve.engine_config({}), "org/m")
    e.start()
    e.stop()
    assert fakes["run"][0] == ["docker", "stop", "-t", "20", "ndforge-vllm-9"] and fakes["run"][1][:3] == ["docker", "rm", "-f"]
    e.stop()                                                  # twice is fine
    assert e.stopped


def test_stop_terminates_the_client_and_survives_failing_stop_commands(fakes, monkeypatch):
    monkeypatch.setattr(ve, "free_port", lambda: 9)
    monkeypatch.setattr(ve, "_run", lambda *a, **k: (_ for _ in ()).throw(OSError("no docker")))
    e = ve.Engine(ve.engine_config({}), "org/m")
    e.start()
    e.stop()
    assert fakes["proc"].terminated


def test_start_failure_is_plain(monkeypatch):
    monkeypatch.setattr(ve, "free_port", lambda: 9)
    monkeypatch.setattr(ve, "_popen", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("docker")))
    with pytest.raises(ve.EngineError, match="Docker Desktop"):
        ve.Engine(ve.engine_config({}), "org/m").start()


def test_wait_ready_and_highlights(fakes, monkeypatch):
    srv = server()
    monkeypatch.setattr(ve, "free_port", lambda: srv.server_address[1])
    e = ve.Engine(ve.engine_config({}), "org/m")
    e.start()
    e.wait_ready(10)
    assert e.ready_seconds is not None and e.server_version() == "0.34.0"
    lines = ve.log_highlights(e.log_text())
    assert any("FLASH_ATTN" in x for x in lines) and any("FlashInfer" in x for x in lines) and not any(x == "noise" for x in lines)
    e.stop()
    srv.shutdown()


def test_wait_ready_crash_and_timeout_and_stop(fakes, monkeypatch):
    srv = server(status=503)
    monkeypatch.setattr(ve, "free_port", lambda: srv.server_address[1])
    e = ve.Engine(ve.engine_config({}), "org/m")
    e.start()
    with pytest.raises(ve.EngineError, match="did not become ready"):
        e.wait_ready(0.3)
    e.wait_ready(5, should_stop=lambda: True)                 # Stop returns without error
    fakes["proc"].returncode = 3
    with pytest.raises(ve.EngineError, match="exit code 3"):
        e.wait_ready(5)
    e.stop()
    srv.shutdown()


def test_url_engine_starts_and_stops_nothing(fakes):
    srv = server()
    e = ve.Engine(ve.engine_config({"engine": {"kind": "url", "url": f"http://127.0.0.1:{srv.server_address[1]}/"}}), "model")
    e.start()
    e.wait_ready(5)
    e.stop()
    assert fakes["popen"] == [] and fakes["run"] == []
    srv.shutdown()


# ---- docker inside WSL, the WSL version, "auto"

@pytest.mark.parametrize("text,version,ok", [
    ("WSL version: 2.7.3.0\nKernel version: 6.6", (2, 7, 3, 0), True),
    ("WSL-Version: 3.0.1.0", (3, 0, 1, 0), True),
    ("Version de WSL : 3.0.1", (3, 0, 1), True),
    ("WSL version: 2.6.1.0", (2, 6, 1, 0), False),
    ("WSL version: 2.10.0", (2, 10, 0), True),                  # numbers, not strings: 2.10 is newer than 2.7
    ("something went wrong", None, None),
    ("", None, None),
])
def test_wsl_version_parser(text, version, ok):
    assert ve.parse_wsl_version(text) == version
    n = ve.wsl_version_note(text)
    assert n["ok"] is ok
    if ok is None:
        assert "unknown, continuing" in n["text"]
    if ok is False:
        assert "wsl --update" in n["text"]


def test_docker_in_wsl_commands_with_and_without_a_distro(monkeypatch):
    monkeypatch.setattr(ve, "free_port", lambda: 18300)
    e = ve.Engine(ve.engine_config({"engine": {"kind": "docker_wsl"}}), r"C:\models\q", adapter=r"D:\a")
    cmd = e.build()
    assert cmd[:5] == ["wsl.exe", "--", "docker", "run", "--gpus"]
    assert "/mnt/c/models/q:/models/base:ro" in cmd and "/mnt/d/a:/models/adapter:ro" in cmd
    e2 = ve.Engine(ve.engine_config({"engine": {"kind": "docker_wsl", "distro": "Ubuntu"}}), "org/m")
    c2 = e2.build()
    assert c2[:7] == ["wsl.exe", "-d", "Ubuntu", "--", "docker", "run", "--gpus"] and "ndforge-vllm-hf:/root/.cache/huggingface:rw" in c2
    assert ve.stop_commands(ve.engine_config({"engine": {"kind": "docker_wsl", "distro": "Ubuntu"}}), 4) == [
        ["wsl.exe", "-d", "Ubuntu", "--", "docker", "stop", "-t", "20", "ndforge-vllm-4"], ["wsl.exe", "-d", "Ubuntu", "--", "docker", "rm", "-f", "ndforge-vllm-4"]]
    assert ve.engine_config({"engine": {"kind": "docker_wsl"}})["version"] == "0.34.0"


def test_docker_in_wsl_stop_is_always_called(fakes, monkeypatch):
    monkeypatch.setattr(ve, "free_port", lambda: 9)
    e = ve.Engine(ve.engine_config({"engine": {"kind": "docker_wsl"}}), "org/m")
    e.start()
    e.stop()
    assert fakes["run"][0][:5] == ["wsl.exe", "--", "docker", "stop", "-t"] and fakes["run"][1][2:5] == ["docker", "rm", "-f"]


def test_auto_prefers_windows_docker_then_docker_in_wsl(monkeypatch):
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        return type("R", (), {"returncode": 0 if cmd[0] == "wsl.exe" or cmd[0] == "docker" and AUTO["docker"] else 1})()

    AUTO = {"docker": True}
    monkeypatch.setattr(ve, "_run", run)
    assert ve.engine_config({"engine": {"kind": "auto"}})["kind"] == "docker"
    AUTO["docker"] = False
    assert ve.engine_config({"engine": {"kind": "auto", "distro": "U"}})["kind"] == "docker_wsl"
    assert seen[-1] == ["wsl.exe", "-d", "U", "--", "docker", "version"]
    monkeypatch.setattr(ve, "_run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(ve.EngineError, match="No engine answered"):
        ve.engine_config({"engine": {"kind": "auto"}})


def test_models_under_mnt_are_flagged_as_slow():
    assert ve.under_mnt(r"C:\models\q") and ve.under_mnt("/mnt/d/x")
    assert not ve.under_mnt("/home/me/q") and not ve.under_mnt("org/model")


# ---- wslc (options from the owner's `wslc --help`, 3.0.x; the combination untested: templates)

def test_template_renderer_with_the_real_default_template():
    v = {"name": "n", "port": 5, "image": "img", "model_dir": r"C:\My Models\q", "args": ["--model", "/models", "--x"], "gpu_flag": "--gpus all", "shm": "16g",
         "publish": "5:8000", "env": []}
    out = ve.render_template(ve.WSLC_TEMPLATES["run"], v)
    assert out == ["wslc", "run", "-d", "--name", "n", "--gpus", "all", "--shm-size", "16g", "-p", "5:8000", "-v", r"C:\My Models\q:/models", "img", "--model", "/models", "--x"]
    assert "--ipc=host" not in out and not any(x.startswith("--ipc") for x in out), "wslc has no --ipc option"
    assert ve.render_template("a {gpu_flag} b {env}", {**v, "gpu_flag": ""}) == ["a", "b"]
    assert ve.render_template("a {env}", {**v, "env": ["-e", "K=V W"]}) == ["a", "-e", "K=V W"]
    with pytest.raises(ve.EngineError, match="not known"):
        ve.render_template("wslc run {nope}", v)


def test_wslc_commands_come_from_editable_templates(monkeypatch):
    monkeypatch.setattr(ve, "free_port", lambda: 7)
    c = ve.engine_config({"engine": {"kind": "wslc", "wslc": {"shm": "8g", "stop": "wslc kill {name}", "publish": "127.0.0.1:{port}:8000", "env": ["-e", "A=1"]}}})
    e = ve.Engine(c, r"C:\my models\q")
    cmd = e.build()
    assert cmd[:3] == ["wslc", "run", "-d"] and cmd[cmd.index("--shm-size") + 1] == "8g" and cmd[cmd.index("-p") + 1] == "127.0.0.1:7:8000"
    assert r"C:\my models\q:/models" in cmd and cmd[cmd.index("--model") + 1] == "/models" and "A=1" in cmd and cmd[cmd.index("--gpus") + 1] == "all"
    assert ve.stop_commands(c, 7) == [["wslc", "kill", "ndforge-vllm-7"], ["wslc", "remove", "-f", "ndforge-vllm-7"]]
    default = ve.Engine(ve.engine_config({"engine": {"kind": "wslc"}}), "org/m").build()
    assert default[default.index("-p") + 1] == "7:8000" and "ndforge-vllm-hf:/models" in default and "--download-dir" in default
    with pytest.raises(ve.EngineError, match="Unknown wslc setting"):
        ve.engine_config({"engine": {"kind": "wslc", "wslc": {"bogus": "x"}}})
    assert ve.wslc_values(c, 7, "", [], loopback=True)["publish"] == "127.0.0.1:7:8000"


def test_wslc_stop_is_always_called_and_detached_exit_is_not_a_crash(fakes, monkeypatch):
    fakes["proc"] = FakeProc(exit_code=0)                       # `wslc run -d` returns at once
    srv = server()
    monkeypatch.setattr(ve, "free_port", lambda: srv.server_address[1])
    e = ve.Engine(ve.engine_config({"engine": {"kind": "wslc"}}), "org/m")
    e.start()
    e.wait_ready(5)
    e.stop()
    assert fakes["run"][-2][:2] == ["wslc", "stop"] and fakes["run"][-1][:2] == ["wslc", "remove"]
    srv.shutdown()


def test_wslc_help_is_captured_verbatim_and_limited(monkeypatch):
    monkeypatch.setattr(ve, "_which", lambda n: None)
    assert ve.wslc_help() == {}
    monkeypatch.setattr(ve, "_which", lambda n: "wslc")

    def run(cmd, **kw):
        if cmd[:2] == ["wslc", "registry"]:
            raise OSError("not a command")
        return type("R", (), {"returncode": 0, "stdout": ("help of " + " ".join(cmd) + "\n" + "x" * 100).encode(), "stderr": b""})()

    monkeypatch.setattr(ve, "_run", run)
    h = ve.wslc_help(limit=40)
    assert list(h) == [" ".join(c) for c in ve.WSLC_HELP_COMMANDS]
    for need in ("wslc --version", "wslc info", "wslc stop --help", "wslc remove --help", "wslc list --help", "wslc logs --help", "wslc pull --help", "wslc volume create --help",
                 "wslc network --help", "wslc registry --help", "wslc system --help", "wslc container --help"):
        assert need in h
    assert h["wslc run --help"]["output"].startswith("help of wslc run --help") and len(h["wslc run --help"]["output"]) == 40
    assert "could not run" in h["wslc registry --help"]["output"]


def test_auto_detection_order_with_fakes(monkeypatch):
    answers = {}
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        key = "docker" if cmd[0] == "docker" else "wslc" if cmd[0] == "wslc" else "docker_wsl" if "docker" in cmd else "wsl"
        return type("R", (), {"returncode": 0 if answers.get(key) else 1})()

    monkeypatch.setattr(ve, "_run", run)
    for have, want in (({"docker": 1, "wslc": 1, "wsl": 1}, "docker"), ({"docker_wsl": 1, "wslc": 1}, "docker_wsl"), ({"wslc": 1, "wsl": 1}, "wslc"), ({"wsl": 1}, "wsl")):
        answers.clear()
        answers.update(have)
        seen.clear()
        assert ve.detect_kind({}) == want
    assert seen[-1][:4] == ["wsl.exe", "--", "test", "-x"]


# ---- the per-run API key (every kind that starts a server)

@pytest.mark.parametrize("kind", ["docker", "docker_wsl", "wslc", "wsl"])
def test_every_started_server_gets_a_fresh_api_key_that_is_never_logged(kind, fakes, monkeypatch):
    from ndforge_worker import protocol
    monkeypatch.setattr(ve, "free_port", lambda: 11)
    a, b = (ve.Engine(ve.engine_config({"engine": {"kind": kind}}), "org/m") for _ in range(2))
    assert a.api_key != b.api_key and len(a.api_key) >= 24
    cmd = a.build()
    assert cmd[cmd.index("--api-key") + 1] == a.api_key
    a.start()
    a.stop()
    assert all(a.api_key not in " ".join(c) for c in a.commands), "the recorded commands are redacted"
    assert "[key]" in " ".join(a.commands[0])
    assert protocol.scrub(f"curl -H 'Bearer {a.api_key}'") == "curl -H 'Bearer [token]'"
    assert a.headers() == {"Authorization": f"Bearer {a.api_key}"}


def test_a_url_engine_uses_the_given_key_and_never_adds_one():
    e = ve.Engine(ve.engine_config({"engine": {"kind": "url", "url": "http://x", "api_key": "sekret-key-123"}}), "model")
    assert e.api_key == "sekret-key-123"
    assert ve.Engine(ve.engine_config({"engine": {"kind": "url", "url": "http://x"}}), "model").headers() == {}
    assert "--api-key" not in ve.serve_args("m")


# ---- a container that crashes at start: no waiting for the timeout, the logs are read before it is removed, no `--rm`

def test_a_crashed_wslc_container_fails_fast_with_its_logs_and_is_still_removed(monkeypatch):
    import types
    monkeypatch.setattr(ve, "free_port", lambda: 54504)
    calls = []

    def run(cmd, **kw):
        calls.append(list(cmd))
        out = {"list": b"[]", "logs": b"INFO loading\nCUDA out of memory. Tried to allocate 20 GiB\n"}.get(cmd[1], b"")
        return types.SimpleNamespace(returncode=0, stdout=out, stderr=b"")

    monkeypatch.setattr(ve, "_run", run)
    monkeypatch.setattr(ve, "_popen", lambda cmd, **kw: types.SimpleNamespace(poll=lambda: 0, returncode=0, terminate=lambda: None, wait=lambda t=None: 0))
    monkeypatch.setattr(ve, "STATE_EVERY", 0.0)
    c = ve.engine_config({"engine": {"kind": "wslc"}})
    e = ve.Engine(c, "org/model")
    e.start()
    assert "--rm" not in e.commands[0]
    import time
    t0 = time.monotonic()
    with pytest.raises(ve.EngineError, match="not running any more") as x:
        e.wait_ready(1500, sleep=lambda s: None)
    assert time.monotonic() - t0 < 5, "no 1500 s wait"
    assert "CUDA out of memory" in str(x.value) and "CUDA out of memory" in e.exit_logs
    e.stop()
    names = [c_[1] for c_ in calls]
    assert names.index("logs") < names.index("stop") < names.index("remove"), "logs first, then the explicit stop and remove"


def test_docker_inspect_is_read_as_running_or_not(monkeypatch):
    import types
    monkeypatch.setattr(ve, "free_port", lambda: 5)
    e = ve.Engine(ve.engine_config({"engine": {"kind": "docker"}}), "org/m")
    e.port = 5
    for text, want in [("true\n", True), ("false\n", False), ("Error: No such object: ndforge-vllm-5", False), ("", None)]:
        monkeypatch.setattr(ve, "_run", lambda cmd, t=text, **kw: types.SimpleNamespace(returncode=0, stdout=t.encode(), stderr=b""))
        assert e.running() is want, text


def test_a_stop_of_a_container_that_is_gone_is_fine(monkeypatch):
    import types
    monkeypatch.setattr(ve, "_run", lambda cmd, **kw: types.SimpleNamespace(returncode=1, stdout=b"", stderr=b"WSLC_E_CONTAINER_NOT_FOUND"))
    e = ve.Engine(ve.engine_config({"engine": {"kind": "wslc"}}), "org/m")
    e.port = 9
    e.stop()
    assert e.stopped
