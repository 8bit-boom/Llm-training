"""The llama-server backend of the thinking check against a stand-in `llama-server` (a small Python program on the PATH that speaks the same HTTP): starting and stopping the
server, the questions with chat_template_kwargs, both shapes of reasoning, plain failures. Unix only (the stand-in is a script with a shebang)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ndforge_worker import llamacpp, protocol
from ndforge_worker import thinkcheck as tc
from ndforge_worker import thinkcheck_llama as tl

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the stand-in server is a script with a shebang")
WORKER = Path(__file__).resolve().parent.parent

SERVER = r'''#!%(py)s
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, %(worker)r)
from ndforge_worker import thinkcheck as tc
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
state = os.environ["FAKE_STATE"]
mode = os.environ.get("FAKE_MODE", "inline")
open(os.path.join(state, "pid"), "w").write(str(os.getpid()))
open(os.path.join(state, "args.json"), "w").write(json.dumps(args))
if mode == "crash":
    sys.stderr.write("error: unable to load model\n"); sys.exit(3)
suite = tc.load_suite()
answers = {i["prompt"]: i for i in suite["items"] + suite["control"]}
t0 = time.time()
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        ready = mode != "loading" and time.time() - t0 > 0.5
        self.send_response(200 if ready else 503); self.end_headers(); self.wfile.write(b"{}")
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with open(os.path.join(state, "requests.jsonl"), "a") as fh:
            fh.write(json.dumps(body) + "\n")
        item = answers[body["messages"][0]["content"]]
        think = body["chat_template_kwargs"]["enable_thinking"]
        if mode == "slow":
            time.sleep(30)
        msg, finish = {}, "stop"
        if mode == "loop" and think:
            msg, finish = {"content": "wait " * 50}, "length"
        elif think and mode == "split":
            msg = {"reasoning_content": "work it out for " + item["id"], "content": "Answer: " + str(item["answer"])}
        elif think:
            msg = {"content": "work it out for " + item["id"] + "\n</think>\n\nAnswer: " + str(item["answer"])}
        else:
            msg = {"content": "Answer: " + str(item["answer"])}
        out = json.dumps({"choices": [{"message": msg, "finish_reason": finish}], "usage": {"completion_tokens": 17}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "llama-server"
    exe.write_text(SERVER % {"py": sys.executable, "worker": str(WORKER)})
    exe.chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    for var in ("LLAMA_CPP_DIR", "NDFORGE_LLAMA_CPP_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_STATE", str(state))
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    return type("S", (), {"state": state, "gguf": gguf, "bin": bin_dir, "tmp": tmp_path})


def lines_of(monkeypatch):
    import io
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    return lambda: [json.loads(x) for x in buf.getvalue().splitlines() if x.startswith("{")]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return True


def gone(pid: int, wait: float = 5.0) -> bool:
    end = time.time() + wait
    while time.time() < end:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return not alive(pid)


def job(s, **kw):
    return {"job": "think_check", "model": str(s.gguf), "backend": "llama", "start_timeout": 30, **kw}


def test_the_server_is_found_with_the_other_llama_cpp_tools(stand_in):
    assert llamacpp.find_tool("llama-server") == stand_in.bin / "llama-server"


def test_a_whole_check_through_the_server(stand_in, monkeypatch):
    out = lines_of(monkeypatch)
    assert tc.run(job(stand_in)) == 0
    res = out()[-1]["data"]
    assert res["backend"] == "llama" and res["main"]["n"] == 32 and res["control"]["n"] == 8
    assert res["main"]["accuracy"] == 1.0 and res["control"]["accuracy"] == 1.0 and res["main"]["closed_rate"] == 1.0 and res["control"]["think_rate"] == 0.0
    assert sum(1 for x in out() if x["type"] == "think_item") == 40
    reqs = [json.loads(x) for x in (stand_in.state / "requests.jsonl").read_text().splitlines()]
    assert len(reqs) == 40 and sum(1 for r in reqs if r["chat_template_kwargs"]["enable_thinking"]) == 32
    main = next(r for r in reqs if r["chat_template_kwargs"]["enable_thinking"])
    assert (main["temperature"], main["top_p"], main["top_k"], main["seed"], main["max_tokens"], main["stream"]) == (0.6, 0.95, 20, 3407, 3072, False)
    assert all(r["max_tokens"] == 512 for r in reqs if not r["chat_template_kwargs"]["enable_thinking"])
    args = json.loads((stand_in.state / "args.json").read_text())
    assert "--jinja" in args and args[args.index("-np") + 1] == "4" and args[args.index("-c") + 1] == "16384" and args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("-m") + 1] == str(stand_in.gguf)
    assert gone(int((stand_in.state / "pid").read_text())), "the server must be stopped when the check is done"


def test_a_server_that_splits_the_reasoning_off_gives_the_same_numbers(stand_in, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "split")
    out = lines_of(monkeypatch)
    assert tc.run(job(stand_in, quick=True, control=False)) == 0
    res = out()[-1]["data"]
    assert res["main"]["n"] == 16 and res["main"]["accuracy"] == 1.0 and res["main"]["closed_rate"] == 1.0 and res["main"]["think_rate"] == 1.0


def test_an_answer_cut_by_the_limit_is_truncated_and_unclosed(stand_in, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "loop")
    out = lines_of(monkeypatch)
    assert tc.run(job(stand_in, quick=True, control=False)) == 0
    m = out()[-1]["data"]["main"]
    assert m["truncated_rate"] == 1.0 and m["closed_rate"] == 0.0 and m["accuracy"] == 0.0


def test_text_of_both_shapes():
    assert tl.text_of({"content": "a\n</think>\n\nb"}, "stop", True) == "a\n</think>\n\nb"
    assert tl.text_of({"reasoning_content": "a", "content": "b"}, "stop", True) == "a\n</think>\n\nb"
    assert tl.text_of({"reasoning_content": "a", "content": ""}, "length", True) == "a"
    assert tl.text_of({"content": "plain"}, "stop", False) == "plain"


def test_a_server_that_cannot_start_is_a_plain_error(stand_in, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "crash")
    lines_of(monkeypatch)
    with pytest.raises(tc.ThinkCheckError, match="stopped while starting.*unable to load model"):
        tc.run(job(stand_in))


def test_a_server_that_never_gets_ready_is_stopped_after_the_timeout(stand_in, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "loading")
    lines_of(monkeypatch)
    with pytest.raises(tc.ThinkCheckError, match="did not become ready within 1 seconds"):
        tc.run(job(stand_in, start_timeout=1))
    assert gone(int((stand_in.state / "pid").read_text()))


def test_stop_ends_the_check_and_the_server(stand_in, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "slow")
    out = lines_of(monkeypatch)
    stop = stand_in.tmp / "STOP"
    import threading
    threading.Timer(2.0, lambda: stop.write_text("")).start()
    t0 = time.time()
    assert tc.run(job(stand_in, stop_file=str(stop))) == 0
    assert time.time() - t0 < 25 and out()[-1]["data"]["stopped"] is True
    assert gone(int((stand_in.state / "pid").read_text()))


def test_no_llama_server_gives_the_error_code_the_page_uses(stand_in, monkeypatch, tmp_path):
    (stand_in.bin / "llama-server").unlink()
    cfg = tmp_path / "job.json"
    cfg.write_text(json.dumps(job(stand_in)))
    p = subprocess.run([sys.executable, "-X", "utf8", "-m", "ndforge_worker", str(cfg)], cwd=WORKER, capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PATH": str(tmp_path)})
    ev = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]
    assert p.returncode == 1 and ev[-1]["code"] == "llama_cpp_missing" and "llama-server" in ev[-1]["message"]


def test_a_model_that_is_not_a_gguf_is_refused(stand_in, monkeypatch):
    lines_of(monkeypatch)
    with pytest.raises(tc.ThinkCheckError, match="needs a GGUF file"):
        tc.run(job(stand_in, model=str(stand_in.tmp)))
