"""The llama-server backend of the thinking check: checks a GGUF file the way llama.cpp (and so Unsloth Studio, LM Studio, Ollama's engine) will really run it, chat template
and quantisation included. It starts `llama-server` on a free local port, waits until it is ready, asks the questions through the OpenAI-style endpoint (`/v1/chat/completions`
with `chat_template_kwargs: {"enable_thinking": ...}`, 4 at a time), and stops the server again when it is done, stopped or failed.

    job: {"model": "<file>.gguf", "ctx": 16384, "parallel": 4, "ngl": 99, "start_timeout": 900, "server": "<path of llama-server>" (optional)}

`-c` is shared between the parallel slots (16384 over 4 = 4096 tokens each: a question of 3072 new tokens plus its prompt fits).
Servers that split the reasoning into `reasoning_content` and servers that leave `<think>...</think>` inside `content` are both understood.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from . import llamacpp, protocol
from .thinkcheck import ThinkCheckError

DEFAULT_CTX = 16384
DEFAULT_PARALLEL = 4
DEFAULT_START_TIMEOUT = 900          # seconds: a big GGUF can take minutes to load
REQUEST_TIMEOUT = 1800


def find_server(job: dict) -> str:
    """The llama-server program: the job's own path, else the same places llama.cpp is looked for everywhere else (llamacpp.find_tool)."""
    given = job.get("server")
    if given:
        if not Path(given).is_file():
            raise ThinkCheckError(f"llama-server was not found at {given}.")
        return str(given)
    found = llamacpp.find_tool("llama-server")
    if not found:
        raise llamacpp.LlamaCppMissing(llamacpp.not_found_text("the thinking check can run the GGUF file (it needs llama-server)"))
    return str(found)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


CACHE_TYPES = ("f16", "q8_0")


def command(exe: str, gguf: str, port: int, job: dict) -> list:
    cmd = [exe, "-m", gguf, "--host", "127.0.0.1", "--port", str(port), "--jinja", "-np", str(int(job.get("parallel") or DEFAULT_PARALLEL)),
           "-c", str(int(job.get("ctx") or DEFAULT_CTX)), "-ngl", str(int(job.get("ngl", 99)))]
    ct = str(job.get("cache_type") or "f16").lower()
    if ct not in CACHE_TYPES:
        raise ThinkCheckError(f"The KV cache type must be f16 or q8_0, not {ct!r}.")
    if ct != "f16":
        # an 8-bit KV cache (off by default): `-ctk/-ctv q8_0`; llama.cpp needs flash attention for a quantized V cache. The flash-attention spelling (`-fa on` of newer builds, `-fa` alone of older ones)
        # is UNVERIFIED here: `flash_attn_args` of the job replaces it.
        cmd += ["-ctk", ct, "-ctv", ct] + [str(x) for x in (job.get("flash_attn_args") or ["-fa", "on"])]
    return cmd


def _die_with_parent():
    """On Linux the server ends when the worker does (the Stop button kills the worker; the server must not stay behind)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 15)        # PR_SET_PDEATHSIG, SIGTERM
    except Exception:           # noqa: BLE001 - not Linux: the finally below does the work
        pass


class Server:
    """A running llama-server: start (waits until it is ready), stop (always)."""

    def __init__(self, exe: str, gguf: str, job: dict):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = tempfile.TemporaryFile()
        kw = {"preexec_fn": _die_with_parent} if sys.platform.startswith("linux") else {}
        self.proc = subprocess.Popen(command(exe, gguf, self.port, job), stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)

    def tail(self, n: int = 12) -> str:
        try:
            self.log.seek(0)
            lines = self.log.read().decode("utf-8", "replace").strip().splitlines()
        except (OSError, ValueError):
            return ""
        return protocol.scrub("\n".join(lines[-n:]))

    def wait_ready(self, timeout: float, should_stop=lambda: False) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if self.proc.poll() is not None:
                raise ThinkCheckError(f"llama-server stopped while starting (exit code {self.proc.returncode}). Its last lines: {self.tail()}")
            if should_stop():
                return
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass                                               # not listening yet, or 503 while the model loads
            if time.monotonic() > deadline:
                raise ThinkCheckError(f"llama-server did not become ready within {int(timeout)} seconds (the model may be too big for this computer). Its last lines: {self.tail()}")
            time.sleep(0.25)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(10)
        try:
            self.log.close()
        except OSError:
            pass


def body_of(request: dict, settings: dict, family: str | None = None) -> dict:
    from .families import think_kwargs
    kwargs = think_kwargs(family, bool(request["thinking"])) if family else {"enable_thinking": bool(request["thinking"])}
    return {"model": "model", "messages": request["messages"], "temperature": settings["temperature"], "top_p": settings["top_p"], "top_k": settings["top_k"],
            "seed": settings["seed"], "max_tokens": request["max_new"], "stream": False, "chat_template_kwargs": kwargs}


def text_of(message: dict, finish: str, thinking: bool) -> str:
    """The generated text as the check reads it: reasoning, `</think>`, answer. A server that splits the reasoning off gets it put back; one that leaves it in `content` is used as it is."""
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if reasoning and thinking:
        return reasoning + ("\n</think>\n\n" + content if content or finish != "length" else "")
    return content


def ask(url: str, request: dict, settings: dict, family: str | None = None) -> dict:
    data = json.dumps(body_of(request, settings, family)).encode("utf-8")
    req = urllib.request.Request(url + "/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
            out = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ThinkCheckError(f"llama-server refused a question (HTTP {exc.code}): {protocol.scrub(exc.read().decode('utf-8', 'replace')[:200])}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ThinkCheckError(f"llama-server did not answer a question ({type(exc).__name__}: {str(exc)[:120]}).") from exc
    choice = (out.get("choices") or [{}])[0]
    finish = choice.get("finish_reason") or ""
    return {"id": request["id"], "text": text_of(choice.get("message") or {}, finish, request["thinking"]), "truncated": finish == "length",
            "gen_tokens": int((out.get("usage") or {}).get("completion_tokens") or 0)}


def generate(requests: list, settings: dict, on_result, should_stop, job: dict) -> None:
    gguf = str(job.get("model") or "")
    if not Path(gguf).is_file() or not gguf.lower().endswith(".gguf"):
        raise ThinkCheckError(f"The llama-server way needs a GGUF file; {gguf or 'no model'} is not one.")
    exe = find_server(job)
    protocol.progress("load", 0.0, "Starting llama-server")
    server = Server(exe, gguf, job)
    pool = None
    counted = {"tokens": 0}
    t_gen = [None]
    try:
        server.wait_ready(float(job.get("start_timeout") or DEFAULT_START_TIMEOUT), should_stop)
        if should_stop():
            return
        protocol.progress("load", 1.0, "")
        t_gen[0] = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=int(job.get("parallel") or DEFAULT_PARALLEL))
        pending = {pool.submit(ask, server.url, r, settings, job.get("family")) for r in requests}
        while pending and not should_stop():                    # looks at the Stop file twice a second while the questions run
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for f in done:
                res = f.result()
                counted["tokens"] += int((res or {}).get("gen_tokens") or 0)
                on_result(res)
    finally:
        if t_gen[0] is not None:
            spent = time.monotonic() - t_gen[0]
            protocol.timing("think_generate", spent, backend="llama", generated_tokens=counted["tokens"], tokens_per_s=round(counted["tokens"] / spent, 1) if spent > 0 and counted["tokens"] else None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        server.stop()
