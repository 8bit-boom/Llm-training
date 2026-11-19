"""The vLLM backend of the thinking check (EXPERIMENTAL, off by default).

    BUILT FROM A RESEARCH REPORT, NOT YET RUN ON THE OWNER'S PC. The vLLM flags and behaviours are in `vllmengine.py`; run the "vLLM self-test" first.

It starts (or connects to) a vLLM server through `vllmengine.Engine` (Docker, WSL or an address), waits until `/health` answers, asks all the questions at once (40 in parallel:
vLLM batches them, that is the point), and ALWAYS stops the server again (`finally`, the Stop file, a failure), explicitly: killing `docker.exe` / `wsl.exe` on Windows does not end
the container or the Linux process.

    job: {"model": folder or Hub id, "backend": "vllm", "parallel": 40, "start_timeout": 1500,
          "engine": {"kind": "docker" | "wsl" | "url", "image": ..., "distro": ..., "url": ..., "quant": "none" | "fp8" | "<pre-quantized id>", ...},
          "adapter": LoRA folder (optional: served as `round` with --enable-lora)}

The request is the llama-server request (`thinkcheck_llama.body_of`) plus `repetition_penalty: 1.0, presence_penalty: 0.0, min_p: 0.0`: every sampling value explicit, so the model's own
generation_config (also switched off with `--generation-config vllm`) cannot leak in. The reasoning is NOT split off by a server parser: the chat template ends the prompt with `<think>\\n`,
so `content` starts inside the reasoning, which is what `thinkcheck.split_output` expects (a server that does split it is read as well, `thinkcheck_llama.text_of`).
Numbers are not identical to the transformers backend (same distribution, other random numbers): gates are always read against the original run on the SAME backend.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from . import protocol, vllmengine, vllmquant
from .thinkcheck import ThinkCheckError
from .thinkcheck_llama import REQUEST_TIMEOUT, text_of

DEFAULT_PARALLEL = 40
EXTRA_SAMPLING = {"repetition_penalty": 1.0, "presence_penalty": 0.0, "min_p": 0.0}


def pick_of(job: dict) -> dict:
    """What the server loads and how (`vllmquant.pick`): the model (the original's pre-quantized checkpoint when the original does not fit), "none" | "fp8" | "prequant:<repo>" | "stored:<m>"."""
    return vllmquant.pick(str(job.get("model") or ""), job.get("engine") or {})


def quant_of(job: dict) -> str:
    """How the weights are loaded in the server (the cache key and the result say it): "none", "fp8", "prequant:<repo>" or "stored:<method>"."""
    return pick_of(job)["quant"]


def served_model(job: dict) -> str:
    return vllmengine.ADAPTER_NAME if job.get("adapter") else vllmengine.SERVED_NAME


def body_of(request: dict, settings: dict, model: str = vllmengine.SERVED_NAME) -> dict:
    return {"model": model, "messages": request["messages"], "temperature": settings["temperature"], "top_p": settings["top_p"], "top_k": settings["top_k"], **EXTRA_SAMPLING,
            "seed": settings["seed"], "max_tokens": request["max_new"], "stream": False, "chat_template_kwargs": {"enable_thinking": bool(request["thinking"])}}


def ask(url: str, request: dict, settings: dict, model: str = vllmengine.SERVED_NAME, api_key: str = "") -> dict:
    data = json.dumps(body_of(request, settings, model)).encode("utf-8")
    req = urllib.request.Request(url + "/v1/chat/completions", data=data, headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
            out = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ThinkCheckError(f"The vLLM engine refused a question (HTTP {exc.code}): {protocol.scrub(exc.read().decode('utf-8', 'replace')[:200])}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ThinkCheckError(f"The vLLM engine did not answer a question ({type(exc).__name__}: {str(exc)[:120]}).") from exc
    choice = (out.get("choices") or [{}])[0]
    finish = choice.get("finish_reason") or ""
    return {"id": request["id"], "text": text_of(choice.get("message") or {}, finish, request["thinking"]), "truncated": finish == "length",
            "gen_tokens": int((out.get("usage") or {}).get("completion_tokens") or 0)}


def make_engine(job: dict) -> "vllmengine.Engine":
    model = str(job.get("model") or "")
    if not model or model.lower().endswith(".gguf") or Path(model).is_file():
        raise ThinkCheckError(f"The vLLM way needs a model folder or a Hub id; {model or 'no model'} is not one (a GGUF file can use the llama-server way).")
    try:
        cfg = vllmengine.engine_config(job)
    except vllmengine.EngineError as exc:
        raise ThinkCheckError(str(exc)) from exc
    e = job.get("engine") or {}
    total, used = vllmengine.gpu_mb()
    g = vllmengine.plan_gpu_util(total, used, float(e.get("gpu_memory_utilization") or vllmengine.DEFAULT_GPU_UTIL))
    if g["ok"] is False:
        raise ThinkCheckError(g["text"])
    job["engine"] = e = {**e, "gpu_memory_utilization": g["util"]}          # lowered when other programs use the card; the weights' fit below uses it
    job.setdefault("_engine_report", {})["gpu"] = g
    pick = pick_of(job)
    if pick["fits"] is False:
        raise ThinkCheckError(pick["reason"])
    job.setdefault("_engine_report", {})["pick"] = {k: pick[k] for k in ("quant", "reason", "weights_gb", "usable_gb")}
    return vllmengine.Engine(cfg, pick["model"], adapter=str(job["adapter"]) if job.get("adapter") else None, quant=pick["quant"], extra_args=e.get("extra_args"),
                             max_len=int(e.get("max_model_len") or vllmengine.DEFAULT_MAX_MODEL_LEN), max_seqs=int(e.get("max_num_seqs") or vllmengine.DEFAULT_MAX_SEQS),
                             gpu_util=float(e.get("gpu_memory_utilization") or vllmengine.DEFAULT_GPU_UTIL))


def generate(requests: list, settings: dict, on_result, should_stop, job: dict) -> None:
    engine = make_engine(job)
    report = job.setdefault("_engine_report", {})
    protocol.progress("load", 0.0, "Starting the vLLM engine (experimental)")
    pool = None
    counted = {"tokens": 0}
    t_gen = [None]
    try:
        try:
            engine.start()
            engine.wait_ready(float(job.get("start_timeout") or vllmengine.DEFAULT_START_TIMEOUT), should_stop)
        except vllmengine.EngineError as exc:
            raise ThinkCheckError(str(exc)) from exc
        report.update(kind=engine.kind, ready_seconds=engine.ready_seconds, server_version=engine.server_version(), highlights=vllmengine.log_highlights(engine.log_text()))
        if should_stop():
            return
        protocol.progress("load", 1.0, "")
        t_gen[0] = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=int(job.get("parallel") or DEFAULT_PARALLEL))
        model = served_model(job)
        pending = {pool.submit(ask, engine.url, r, settings, model, engine.api_key) for r in requests}
        while pending and not should_stop():                    # looks at the Stop file twice a second while the questions run
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for f in done:
                res = f.result()
                counted["tokens"] += int((res or {}).get("gen_tokens") or 0)
                on_result(res)
    finally:
        if t_gen[0] is not None:
            spent = time.monotonic() - t_gen[0]
            report.update(generate_seconds=round(spent, 1), completion_tokens=counted["tokens"], tokens_per_s=round(counted["tokens"] / spent, 1) if spent > 0 and counted["tokens"] else None)
            protocol.timing("think_generate", spent, backend="vllm", generated_tokens=counted["tokens"], tokens_per_s=report["tokens_per_s"])
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        engine.stop()
