"""The vLLM self-test (job `vllm_selftest`): runs the research report's 1-hour experiment for the owner, as far as the app can, and writes `vllm_selftest.json` / `.md`.

    BUILT FROM A RESEARCH REPORT, NOT YET RUN ON THIS PC. The report's success criteria are the verdict: engine ready in under 5 minutes, at least 300 completion tokens/s with 40 parallel
    requests, and thinking numbers within the noise of the saved standard-engine / llama.cpp results of the same questions.

    job: {"model": folder or Hub id (default: the configured pre-quantized 27 B), "confirm_download": true (needed for a Hub id: it is downloaded, about 21 GB for the default),
          "engine": {...}, "out_dir": folder, "stop_file": path, "quick": true, "gpu_test": true,
          "compare": {"hf": result file, "llama": result file} (optional saved results, either the cache file `{"key", "result"}` or a plain result),
          "adapter": LoRA folder (optional: serve it and compare with `compare.hf_adapter`), "start_timeout": 1500}

Steps: prerequisites (driver, WSL version, GPU passthrough in a throw-away container, `wslc` help captured verbatim) -> start the engine -> wait for /health (startup time, attention / GDN kernel
log lines, reachability from this side) -> the thinking suite with 40 parallel requests (wall time, completion tokens/s, closed / accuracy / loop rates) -> compare -> stop the engine (ALWAYS,
in a `finally`) -> the report with a plain verdict. Stop writes the stop file; nothing stays running; a report of a stopped test says so.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from . import protocol, thinkcheck, thinkcheck_vllm, vllmengine
from .thinkcheck import ThinkCheckError

REPORT_VERSION = 1
NOTE = "Built from a research report and not yet run on this PC: this self-test is the experiment. Speeds elsewhere in the app for vLLM are estimates."
CRITERIA = {"ready_seconds": 300.0, "tokens_per_s": 300.0}
NOISE_QUESTIONS = 2                                   # the report: within suite noise = +-1-2 questions
OUT_LIMIT = 4000
DEFAULT_MODEL = "cyankiwi/Qwen3.8-27B-AWQ-INT4"
DOWNLOAD_GB = {"cyankiwi/Qwen3.8-27B-AWQ-INT4": 21.0}


def download_note(model: str) -> str:
    gb = DOWNLOAD_GB.get(model)
    return f"{model} is downloaded from the Hugging Face Hub the first time ({'about %.0f GB' % gb if gb else 'its size depends on the model'}); the engine's volume keeps it for later." if not vllmengine.is_local_path(model) else ""


# ---- what the model is (the numbers of a self-test belong to THIS model: the program scales them by size, or refuses to, when it plans for another one)

_QUANT_WORDS = re.compile(r"(awq|gptq|nvfp4|fp4|fp8|int4|int8|4bit|8bit|bnb|gguf|w4a16|w8a8)", re.I)


def param_count_b(model: str, safetensors_total=None, weights_gb=None):
    """(billions of parameters, where it comes from): the size in the name (`...-9B...`), else the Hub's safetensors count, else a local folder's 16-bit bytes / 2; (None, "unknown")."""
    m = re.search(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", str(model).replace("-", " ").replace("_", " "))
    if m:
        return float(m.group(1)), "the size in the model's name"
    if safetensors_total:
        return round(float(safetensors_total) / 1e9, 2), "the Hub's parameter count"
    if weights_gb and vllmengine.is_local_path(model):
        return round(float(weights_gb) / 2.0, 2), "the folder's weight bytes at 16 bits"
    return None, "unknown"


def model_facts(model: str, quant: str = "none") -> dict:
    """{"model", "params_b", "params_source", "weights_gb", "weights_source", "quant", "note"}: from the folder, or for a Hub id from the Hub's file list (a plain note when it cannot be read)."""
    from . import vllmquant
    gb, src, total, note = None, "unknown", None, ""
    if vllmengine.is_local_path(model):
        gb, src = vllmquant.weights_gb(model), "the folder's .safetensors files"
    else:
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(model, files_metadata=True)
            size = sum(int(getattr(x, "size", 0) or 0) for x in (info.siblings or []) if str(getattr(x, "rfilename", "")).endswith(".safetensors"))
            gb, src = (round(size / 1e9, 1) if size else None), "the Hub's file list"
            total = ((getattr(info, "safetensors", None) or {}).get("total") if isinstance(getattr(info, "safetensors", None), dict) else getattr(getattr(info, "safetensors", None), "total", None))
        except Exception as exc:                          # noqa: BLE001 - offline, gated, unknown id: the size stays unknown and the test says so
            note = f"The size of {model} could not be read from the Hugging Face Hub ({type(exc).__name__}); it is not checked against the card."
    params, psrc = param_count_b(model, total, gb)
    q = quant if quant not in ("none", "") else (vllmquant.stored_quantization(model) if vllmengine.is_local_path(model) else None) or ((_QUANT_WORDS.search(model) or [None])[0] and f"quantized ({_QUANT_WORDS.search(model).group(1).lower()})") or "16-bit"
    return {"model": model, "params_b": params, "params_source": psrc, "weights_gb": gb, "weights_source": src, "quant": q, "note": note}


def fit_problem(facts: dict, vram_gb, default_model: str = "", util: float | None = None) -> str | None:
    """A plain refusal when the weights (16-bit) cannot fit the card next to the cache; None when they fit or the size is not known."""
    from . import vllmquant
    gb = facts.get("weights_gb")
    if not gb or not vram_gb or facts.get("quant") != "16-bit":
        return None
    usable = vram_gb * (util or vllmquant.GPU_UTIL) - vllmquant.CACHE_GB
    if gb <= usable:
        return None
    return (f"{facts['model']} is about {gb:.0f} GB in 16-bit; your card has {vram_gb:.0f} GB (about {usable:.0f} GB are usable by the engine). Use the pre-quantized id (leave the model field empty for the default "
            f"{default_model or DEFAULT_MODEL}, about 21 GB) or a smaller model.")


LONG_REQUESTS, LONG_TOKENS = 8, 2048


def answer_stats(tokens: list) -> dict:
    """Tokens per answer of the thinking questions: {"n", "total", "mean", "median", "p90"} (empty list: zeros)."""
    t = sorted(int(x) for x in tokens)
    if not t:
        return {"n": 0, "total": 0, "mean": 0, "median": 0, "p90": 0}
    pick = lambda q: t[min(len(t) - 1, int(q * (len(t) - 1) + 0.5))]    # noqa: E731
    return {"n": len(t), "total": sum(t), "mean": round(sum(t) / len(t), 1), "median": pick(0.5), "p90": pick(0.9)}


def generic_note(model) -> str:
    return (f"This is a generic engine test on `{model}` (the configured checkpoint or the model you chose). It does NOT test your own original model: a finetune is served from its own folder, quantized to 4 bits while loading "
            "(the in-flight bitsandbytes route, unverified), and a pre-quantized checkpoint stands in only for the original it was made from or a copy you confirmed.")


GENERIC_RANGE = "100-250"


def assess_answers(stats: dict, suite: dict, think_blocks: int, n_answers: int, reference: dict | None = None) -> dict:
    """Are the answers of the Quick suite plausible thinking answers? -> {"level": "ok" | "note" | "warn", "text", "reference_median"}.
    The yardstick is the saved standard-engine check of the same suite when there is one (its `median_think_tokens`): a WARNING only when the vLLM median is below half of it, when the thinking rate
    is low, or when no answer has a `<think>` block. Without a reference the median is only a NOTE (a 27 B original typically writes a median of about 100-250 tokens on this suite); a fixed
    token limit is not a rule. "did not think" is said only when the thinking rate or the `<think>` blocks say so."""
    med, rate = stats.get("median", 0), (suite or {}).get("think_rate")
    ref = None
    if reference and isinstance(reference.get("main"), dict):
        ref = reference["main"].get("median_think_tokens")
    if stats.get("n", 0) == 0:
        return {"level": "note", "text": "No thinking answers were measured.", "reference_median": ref}
    if (rate is not None and rate < 0.2) or (n_answers > 0 and think_blocks == 0):
        return {"level": "warn", "reference_median": ref, "text": f"The model did not think: think_rate {rate}, {think_blocks} of {n_answers} answers have a `<think>` block. Check `enable_thinking` and the `<think>` prefix of the chat template in the sample answers. "
                "This speed does not predict a real thinking check; the long-answer speed below does."}
    if ref:
        if med < 0.5 * float(ref):
            return {"level": "warn", "reference_median": ref, "text": f"The answers are much shorter than your saved standard-engine check of this suite (median {med} tokens against {ref}): the quantized model may think less, or max_tokens or the chat template differ. "
                    "Look at the sample answers. This speed does not predict a real thinking check; the long-answer speed below does."}
        return {"level": "ok", "reference_median": ref, "text": f"The answers are as long as your saved standard-engine check of this suite (median {med} tokens against {ref})."}
    return {"level": "note", "reference_median": None, "text": f"The answers are short (median {med} tokens): the original thinking check of a 27 B model on this suite typically has a median around {GENERIC_RANGE} tokens; compare with your standard check "
            "(no saved standard-engine numbers were given). The long-answer speed below is the one the estimates use."}


def long_decode(engine, should_stop, model_name: str = vllmengine.SERVED_NAME, n: int = LONG_REQUESTS, tokens: int = LONG_TOKENS) -> dict:
    """The decode speed under thinking-like length: `n` requests that must write `tokens` tokens each (`min_tokens` and `ignore_eos` are vLLM request extras: UNVERIFIED on this version; a
    server that refuses them gives a plain error here and the run goes on without this number)."""
    import urllib.request
    url = engine.url + "/v1/chat/completions"

    def one(i: int) -> int:
        body = {"model": model_name, "messages": [{"role": "user", "content": f"Think step by step, at great length, about how a small village could design its own water supply (variant {i})."}],
                "temperature": 0.6, "top_p": 0.95, "max_tokens": tokens, "min_tokens": tokens, "ignore_eos": True, "stream": False, "chat_template_kwargs": {"enable_thinking": True}}
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {engine.api_key}"} if engine.api_key else {})})
        with urllib.request.urlopen(req, timeout=thinkcheck_vllm.REQUEST_TIMEOUT) as r:
            return int((json.loads(r.read().decode("utf-8")).get("usage") or {}).get("completion_tokens") or 0)

    t0 = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=n) as pool:
            got = list(pool.map(one, range(n)))
    except Exception as exc:                               # noqa: BLE001 - one number less, not a failed test
        return {"long_error": f"{type(exc).__name__}: {protocol.scrub(str(exc))[:200]}"}
    spent = time.monotonic() - t0
    total = sum(got)
    return {"long_requests": n, "long_tokens": total, "long_wall_seconds": round(spent, 1), "long_tokens_per_s": round(total / spent, 1) if spent > 0 and total else None}


def _run(cmd: list, timeout: int = 600) -> dict:
    try:
        r = vllmengine._run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout, **vllmengine._win_kw())
        raw = (getattr(r, "stdout", b"") or b"") + (getattr(r, "stderr", b"") or b"")
        text = raw.decode("utf-8", "replace").replace("\x00", "") if isinstance(raw, bytes) else str(raw)
        return {"cmd": vllmengine.redact(cmd, ""), "returncode": getattr(r, "returncode", None), "output": protocol.scrub(text)[-OUT_LIMIT:]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"cmd": cmd, "returncode": None, "output": f"could not run: {exc}"}


def gpu_test_command(cfg: dict) -> list | None:
    """The throw-away container that proves the card is visible (`nvidia-smi`), per kind; None for a server that already runs."""
    kind = cfg["kind"]
    if kind in ("docker", "docker_wsl"):
        return vllmengine.docker_prefix(cfg) + ["run", "--rm", "--gpus", "all", cfg.get("gpu_test_image") or vllmengine.WSLC_TEMPLATES["test_image"], "nvidia-smi"]
    if kind == "wslc":
        return vllmengine.render_template(vllmengine.wslc_templates(cfg)["gputest"], vllmengine.wslc_values(cfg, 0, "", []))
    if kind == "wsl":
        return ["wsl.exe"] + (["-d", cfg["distro"]] if cfg.get("distro") else []) + ["--", "nvidia-smi"]
    return None


def prerequisites(cfg: dict, model: str, gpu_test: bool = True, util: float = vllmengine.DEFAULT_GPU_UTIL) -> list:
    """[{"id", "ok": True | False | None, "detail", "output"?}]: None = unknown (continuing)."""
    out = []
    smi = _run(["nvidia-smi", "--query-gpu=driver_version,memory.used,memory.total", "--format=csv,noheader,nounits"], 30)
    parts = [x.strip() for x in smi["output"].splitlines()[0].split(",")] if smi["returncode"] == 0 and smi["output"].strip() else []
    if parts:
        major = int(parts[0].split(".")[0]) if parts[0].split(".")[0].isdigit() else None
        out.append({"id": "driver", "ok": None if major is None else major >= 580, "detail": f"NVIDIA driver {parts[0]} (580 or newer is wanted for the CUDA 13 images)"})
        used = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        total = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
        g = vllmengine.plan_gpu_util(total, used, util)
        out.append({"id": "gpu_free", "ok": g["ok"], "detail": g["text"], "gpu_util": g["util"], "lowered": g["lowered"]})
    else:
        out.append({"id": "driver", "ok": None, "detail": "nvidia-smi did not answer on this side (unknown, continuing)", "output": smi["output"]})
    w = _run(["wsl.exe", "--version"], 30)
    if w["returncode"] == 0:
        note = vllmengine.wsl_version_note(w["output"])
        out.append({"id": "wsl", "ok": note["ok"], "detail": note["text"]})
    if vllmengine.under_mnt(model):
        out.append({"id": "slow_share", "ok": False, "detail": "The model folder is on a Windows drive (/mnt/...), which WSL reads through a slow share; a named volume or a folder inside WSL loads much faster."})
    cmd = gpu_test_command(cfg) if gpu_test else None
    if cmd:
        r = _run(cmd, 900)
        out.append({"id": "gpu_passthrough", "ok": r["returncode"] == 0, "detail": "the graphics card is visible inside a container" if r["returncode"] == 0 else f"the container could not use the graphics card (exit code {r['returncode']})",
                    "command": r["cmd"], "output": r["output"]})
    return out


def _load_result(path) -> dict | None:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    r = d.get("result") if isinstance(d, dict) and isinstance(d.get("result"), dict) else d
    return r if isinstance(r, dict) and isinstance(r.get("main"), dict) else None


def compare(main: dict, saved: dict, label: str, model: str = "") -> dict:
    """Our numbers against a saved result's: within noise = every rate differs by at most 2 questions."""
    n = max(1, int(main.get("n") or 1))
    tol = NOISE_QUESTIONS / n
    deltas = {k: round(float(main.get(k, 0) or 0) - float(saved["main"].get(k, 0) or 0), 4) for k in ("closed_rate", "accuracy", "loop_rate")}
    return {"with": label, "backend": saved.get("backend"), "model": saved.get("model"), "deltas": deltas, "tolerance": round(tol, 4), "within_noise": all(abs(v) <= tol + 1e-9 for v in deltas.values()),
            "other_model": saved.get("model") != model}


def verdict(rep: dict) -> tuple[str, str]:
    """("pass" | "partial" | "fail" | "stopped", a plain sentence) from the report's success criteria."""
    if rep.get("stopped"):
        return "stopped", "The test was stopped before it finished; nothing is concluded."
    if rep.get("error"):
        return "fail", "The test could not finish: " + rep["error"]
    bad = [p["id"] for p in rep.get("prerequisites", []) if p.get("ok") is False and p["id"] in ("gpu_passthrough", "gpu_free")]
    if bad:
        return "fail", "The engine cannot run here yet: " + ", ".join(bad) + ". See the prerequisites in the report."
    crit = rep.get("criteria", [])
    failed = [c["label"] for c in crit if c["ok"] is False]
    scored = [c for c in crit if c["ok"] is not None]
    if not scored:
        return "fail", "No measurement was made."
    if not failed:
        return "pass", "All the report's criteria were met on this PC: the engine is worth using for the thinking check and for making thinking answers (still check a few answers by eye)."
    if len(failed) < len(scored):
        return "partial", "Some criteria were not met: " + "; ".join(failed) + ". Use vLLM only where those do not matter, or keep the standard engine."
    return "fail", "No criterion was met: " + "; ".join(failed) + ". Keep the standard engine."


def criteria_of(rep: dict) -> list:
    out = []
    rs, tps = rep.get("startup_seconds"), rep.get("tokens_per_s")
    out.append({"id": "ready", "label": f"engine ready in under {int(CRITERIA['ready_seconds'] / 60)} minutes", "limit": CRITERIA["ready_seconds"], "value": rs, "ok": None if rs is None else rs < CRITERIA["ready_seconds"]})
    lt = rep.get("long_tokens_per_s")
    tps = lt if lt is not None else tps          # the speed on LONG answers when it was measured: short answers overstate a thinking check
    out.append({"id": "speed", "label": f"at least {int(CRITERIA['tokens_per_s'])} tokens/s with 40 requests at once" + (" (long answers)" if lt is not None else " (short answers: not measured on long ones)"), "limit": CRITERIA["tokens_per_s"], "value": tps,
                "ok": None if tps is None else tps >= CRITERIA["tokens_per_s"]})
    for c in rep.get("comparisons", []):
        if c.get("other_model"):
            out.append({"id": "noise_" + c["with"], "label": f"numbers within noise of {c['with']} (+-{NOISE_QUESTIONS} questions): not like-for-like (reference: {c.get('model')})", "limit": c["tolerance"], "value": c["deltas"], "ok": None, "not_like_for_like": True})
        else:
            out.append({"id": "noise_" + c["with"], "label": f"numbers within noise of {c['with']} (+-{NOISE_QUESTIONS} questions)", "limit": c["tolerance"], "value": c["deltas"], "ok": c["within_noise"]})
    return out


WSLC_VERIFIED = ("wslc: your runs verified `run -d` with --gpus, --shm-size, -p and -v, `stop -t`, `remove -f`, `pull`, `logs -n N`, `inspect <name>`, `list -a` (table) and `list -a --format json` "
                 "(fields Command, CreatedAt, HealthStatus, ID, Image, Labels, LocalVolumes, Mounts, Names, Networks, Platform, Ports, RunningFor, Size, State, Status; one JSON object per container, "
                 "a single object and not an array when there is one container), and that the published port is reachable from Windows. Still untested: `cp` and `exec`. The commands are templates you can edit.")


def backend_note(kf: dict):
    """When the two engines of the fp8 comparison use another text-layer attention backend: the sentence that says the comparison is dtype and backend together. None when they match or are unknown."""
    a = ((kf.get("auto") or {}).get("backends") or {}).get("text_attention")
    b = ((kf.get("fp8") or {}).get("backends") or {}).get("text_attention")
    if a and b and a != b:
        return (f"Not only the cache type changed: the attention backend changed from {a} to {b} (vLLM switches because the backend of the standard engine has no fp8 KV cache on this card). "
                "The comparison is the cache type and the backend together; a difference in speed or in the answers cannot be put on the cache type alone.")
    return None


def markdown(rep: dict) -> str:
    L = ["# vLLM self-test", "", f"**{rep['verdict'].upper()}**: {rep['verdict_text']}", "", f"> {NOTE}", "", f"> {generic_note(rep.get('model'))}", "", f"Created {rep['created']} - model `{rep.get('model')}` - engine {rep.get('engine', {}).get('kind')} ({rep.get('engine', {}).get('image', '')})", ""]
    L += ["## Success criteria (from the report)", ""] + [f"- [{'x' if c['ok'] else ' ' if c['ok'] is False else '?'}] {c['label']}: {'not reached' if c.get('not_reached') else 'n/a' if c.get('not_like_for_like') else c['value']}" for c in rep.get("criteria", [])] + [""]
    L += ["## Prerequisites", ""] + [f"- [{'x' if p['ok'] else ' ' if p['ok'] is False else '?'}] {p['id']}: {p['detail']}" for p in rep.get("prerequisites", [])] + [""]
    if rep.get("download_note"):
        L += [rep["download_note"], ""]
    mi = rep.get("model_info")
    if mi:
        L += ["## What was tested", "", f"Model `{mi['model']}`: {mi.get('params_b') if mi.get('params_b') is not None else 'unknown'} B parameters ({mi.get('params_source')}), {mi.get('quant')}, weights {mi.get('weights_gb') if mi.get('weights_gb') is not None else 'unknown'} GB ({mi.get('weights_source')}). "
              f"GPU: {(rep.get('gpu') or {}).get('name') or 'unknown'}, {(rep.get('gpu') or {}).get('memory_gb') or 'unknown'} GB, driver {(rep.get('gpu') or {}).get('driver') or 'unknown'}. Engine {rep.get('engine', {}).get('kind')} {rep.get('engine', {}).get('wslc_version') or rep.get('engine', {}).get('version')}; vLLM takes {rep.get('gpu_util', 'the configured share')} of the card.",
              "The speeds below belong to THIS model: the program scales them by size when it plans for another one.", ""] + ([mi["note"], ""] if mi.get("note") else [])
    if rep.get("error"):
        L += ["## Error", "", rep["error"], ""]
    if rep.get("start_command"):
        kr = rep.get("kv_requested") or {}
        L += ["## Start command (the API key is replaced)", "", f"KV cache type passed to this engine: {kr.get('first', 'auto')}.", "", "    " + " ".join(str(x) for x in rep["start_command"]), ""]
    if rep.get("start_command_fp8"):
        L += ["## Start command of the fp8 engine (the API key is replaced)", "", "KV cache type passed to this engine: fp8.", "", "    " + " ".join(str(x) for x in rep["start_command_fp8"]), ""]
    if rep.get("exit_logs"):
        L += ["## Last lines of the container's log (read before it was removed)", ""] + [f"    {x}" for x in str(rep["exit_logs"]).splitlines()[-60:]] + [""]
    if rep.get("timings"):
        L += ["## Time so far", "", ", ".join(f"{k} {v}" for k, v in rep["timings"].items()) + f", whole test {rep.get('seconds', '?')} s.", ""]
    at, su = rep.get("answer_tokens"), rep.get("suite")
    if su:
        L += ["## Thinking suite on this engine", "", f"{rep.get('completion_tokens')} tokens generated in {rep.get('wall_seconds')} s ({rep.get('tokens_per_s')} tokens/s). Tokens per thinking answer: " + (f"mean {at['mean']}, median {at['median']}, 90th percentile {at['p90']} ({at['n']} answers)." if at else "not known."), "",
              "| think_rate | closed_rate | accuracy | loop_rate | truncated_rate |", "|---|---|---|---|---|", f"| {su.get('think_rate')} | {su.get('closed_rate')} | {su.get('accuracy')} | {su.get('loop_rate')} | {su.get('truncated_rate')} |", ""]
        ck = rep.get("answer_check")
        L += [f"Thinking answers: think_rate {su.get('think_rate')}, closed_rate {su.get('closed_rate')}, {rep.get('think_blocks', '?')} of {at['n'] if at else '?'} with a `<think>` block."]
        if ck:
            L += ["", ("**" + ck["text"] + "**") if ck["level"] == "warn" else ck["text"]]
        L += [""]
        for smp in [x for x in rep.get("samples", []) if x.get("text")][:2]:
            L += [f"First 300 characters of sample answer `{smp['id']}`:", "", "    " + str(smp["text"])[:300].replace("\n", "\n    "), ""]
        if rep.get("request_check"):
            L += [rep["request_check"]["note"], ""]
        if rep.get("long_tokens_per_s") is not None:
            L += [f"Speed on long answers: {rep['long_requests']} requests of {LONG_TOKENS} tokens each: {rep['long_tokens']} tokens in {rep['long_wall_seconds']} s = {rep['long_tokens_per_s']} tokens/s (this is the number the app's estimates use).", ""]
        elif rep.get("long_error"):
            L += [f"The long-answer speed could not be measured: {rep['long_error']}", ""]
        if rep.get("comparisons"):
            for c in rep["comparisons"]:
                L += [f"Compared with the saved {c['with']} numbers{' of ' + str(c.get('model')) if c.get('model') else ''}{' (ANOTHER model than this test: not a like-for-like comparison)' if c.get('other_model') else ''}: differences {c['deltas']}, {'within' if c['within_noise'] else 'outside'} the noise (+-{c['tolerance']}).", ""]
        else:
            L += ["No saved standard-engine (hf / llama.cpp) numbers for this model were given, so nothing is compared.", ""]
    if rep.get("startup_seconds") is not None:
        L += ["## Start-up", "", f"Ready after {rep['startup_seconds']} s. Server version: {rep.get('server_version') or 'unknown'}.", "", "Log lines about the attention / GDN kernel, the backend, the quantization and the cache (kernel lines first):", ""] + [f"    {x}" for x in rep.get("highlights", [])] + [""]
        if rep.get("startup_breakdown"):
            L += [rep["startup_breakdown"] + ".", ""]
        if (rep.get("kv") or {}).get("note"):
            L += [rep["kv"]["note"], ""]
    if rep.get("reachability"):
        L += ["## Reachability from this side", "", rep["reachability"]["detail"], ""]
    if rep.get("tokens_per_s") is not None:
        s = rep.get("suite", {})
        L += ["## Thinking suite (Quick, 40 requests at once)", "", f"Wall time {rep.get('wall_seconds')} s, {rep.get('completion_tokens')} completion tokens, {rep['tokens_per_s']} tokens/s.", "",
              f"closed {s.get('closed_rate')}, accuracy {s.get('accuracy')}, loops {s.get('loop_rate')}, garbage {s.get('garbage_rate')}, cut by the limit {s.get('truncated_rate')}", ""]
    for c in rep.get("comparisons", []):
        L += [f"Against {c['with']} ({c.get('backend')}, model {c.get('model')}): differences {c['deltas']}, {'within' if c['within_noise'] else 'OUTSIDE'} the noise of +-{c['tolerance']}. The saved result may be of another model (the original): a difference is not necessarily a fault.", ""]
    kf = rep.get("kv_fp8")
    if kf:
        L += ["## 8-bit KV cache (fp8), the optional second pass", ""]
        for key in ("auto", "fp8"):
            x = kf.get(key)
            if x and kf.get("comparison_refused") and key == "fp8":
                continue
            if x:
                L += [f"- **{key}** (passed: {x.get('requested_dtype')}; the log {'CONFIRMS' if x.get('log_confirms_fp8') else 'does not mention'} an fp8 KV cache): KV cache {x.get('kv_tokens') if x.get('kv_tokens') is not None else 'unknown'} tokens, max concurrency {x.get('max_concurrency') if x.get('max_concurrency') is not None else 'unknown'}x, long-answer speed {x.get('long_tokens_per_s')} tokens/s, "
                      f"think_rate {x.get('think_rate')}, closed_rate {x.get('closed_rate')}, accuracy {x.get('accuracy')}; attention backend: {x.get('attention_backend') or 'not found in the log'}."]
        bn = kf.get("backend_note") or backend_note(kf)
        if bn and not kf.get("comparison_refused"):
            L += ["", bn]
        for k in ("comparison_refused", "text", "warning", "fp8_failed"):
            if kf.get(k) and not (k == "text" and kf.get("text") == kf.get("comparison_refused")):
                L += ["", kf[k]]
        for key in ("auto", "fp8"):
            x = kf.get(key)
            if x and x.get("highlights") and not (kf.get("comparison_refused") and key == "fp8"):
                L += ["", f"Kernel / backend / cache lines of the {key} engine:", ""] + [f"    {h}" for h in x["highlights"]]
        if kf.get("exit_logs"):
            L += ["", "Last lines of the fp8 engine's log:", ""] + [f"    {x}" for x in str(kf["exit_logs"]).splitlines()[-40:]]
        L += [""]
    if rep.get("think_samples"):
        L += ["## Two thinking answers (one short, one long: the start and the end of the `<think>` block)", ""]
        for x in rep["think_samples"]:
            L += [f"### {x['id']} ({x.get('gen_tokens')} tokens)", "", "start:", "", "    " + x["start"].replace("\n", "\n    "), "", "around the end of the block:", "", "    " + x["end"].replace("\n", "\n    "), ""]
    if rep.get("samples"):
        L += ["## Sample answers (look at them: garbled text means a problem)", ""] + [f"### {x['id']}\n\n    {x['text'][:600]!r}\n" for x in rep["samples"]]
    if rep.get("wslc"):
        L += ["## wslc help (for adapting the templates)", "", WSLC_VERIFIED, ""]
        for f in rep["wslc"].get("findings", []):
            L += [f"**Finding:** {f}", ""]
        for smp in rep["wslc"].get("list_samples", []):
            j = smp.get("json") or {}
            L += [f"### `wslc list -a --format json` {smp.get('when') or ''} at {smp['at']} (exit code {j.get('returncode')}): the real field names", ""]
            L += ["    " + (j.get("output") or "").replace("\n", "\n    ")] if (j.get("output") or "").strip() else ["    (empty while the container ran: `wslc list --format json` printed nothing)"]
            for key, title in (("table", "`wslc list -a` (table)"), ("inspect", "`wslc inspect <name>`")):
                if smp.get(key):
                    L += ["", f"{title} at the same moment (exit code {smp[key]['returncode']}):", "", "    " + (smp[key]["output"] or "(empty)").replace("\n", "\n    ")]
            L += [""]
        for k, v in rep["wslc"].get("help", {}).items():
            late = " (taken after the engines were removed: empty is expected)" if k.endswith("list -a --format json") else ""
            L += [f"### {k} (exit code {v['returncode']}){late}", "", "```", v["output"] or ("(empty: expected, no container is left)" if late else ""), "```", ""]
        if rep["wslc"].get("loopback"):
            L += ["Loopback publish test: " + rep["wslc"]["loopback"], ""]
    if rep.get("hints"):
        L += ["## Hints", ""] + [f"- {h}" for h in rep["hints"]] + [""]
    L += ["## Commands that were run", ""] + [f"    {' '.join(map(str, c))}" for c in rep.get("commands", [])]
    if rep.get("fp8_commands"):
        L += ["", f"Commands of the fp8 engine ({rep.get('fp8_engine_name')}):", ""] + [f"    {' '.join(map(str, c))}" for c in rep["fp8_commands"]]
    lo = rep.get("leftovers")
    if lo:
        L += ["", "Containers left behind after the stops: " + (", ".join(lo["found"]) + (f" (removed: {', '.join(lo['removed']) or 'none'})" if lo["found"] else "") if lo["found"] else "none (a final `list -a` found no ndforge-vllm- container).")]
    return "\n".join(L) + "\n"


def list_sample(rep: dict, cfg: dict, when: str = "", name: str = "") -> None:
    """A sample of what wslc says about the containers WHILE the engine's container exists (after /health answered, right before the stop, and in the failure path before the removal):
    `wslc list -a --format json` (the real field names), `wslc list -a` (the table) and `wslc inspect <name>`. An empty JSON is said to be empty."""
    if cfg.get("kind") != "wslc":
        return
    smp = {"at": time.strftime("%H:%M:%S"), "when": when}
    for key, cmd in (("json", ["wslc", "list", "-a", "--format", "json"]), ("table", ["wslc", "list", "-a"]), ("inspect", ["wslc", "inspect", name] if name else None)):
        if cmd:
            r = _run(cmd, 60)
            smp[key] = {"returncode": r["returncode"], "output": tidy(r["output"])[-3000:]}
    rep.setdefault("wslc", {}).setdefault("list_samples", []).append(smp)
    ins = ((smp.get("inspect") or {}).get("output") or "").lower()
    empty = not ((smp.get("json") or {}).get("output") or "").strip().strip("[]").strip() and not ((smp.get("table") or {}).get("output") or "").strip()
    if empty and name and re.search(r'"running"\s*:\s*true|status"?\s*[:=]\s*"?running|\brunning\b', ins):
        rep["wslc"].setdefault("findings", []).append(f"{when or 'at ' + smp['at']}: `wslc list` is empty while the container {name} runs (`wslc inspect` says it is running).")


FP8_REFUSED = re.compile(r"(fp8|kv.cache.dtype|kv_cache_dtype)[^\n]*(not supported|unsupported|incompatible|not implemented|does not support|cannot|invalid|error)|(not supported|unsupported|does not support)[^\n]*(fp8|kv.cache)", re.I)
TEXT_BACKEND = re.compile(r"Using (\w+) attention backend out of potential backends:?\s*(\[[^\]]*\])?")
VIT_BACKEND = re.compile(r"[^\n]*(?:for vit attention|MMEncoderAttention)[^\n]*", re.I)
DECODE_BACKEND = re.compile(r"decode_backend=(\w+)")
FLASHINFER_KV = re.compile(r"FlashInfer resolved query dtypes:[^\n]*?kv_cache_dtype=([\w.]+)")
GDN_KERNEL = re.compile(r"Using (\w+) GDN prefill kernel")
LINEAR_KERNEL = re.compile(r"Using (\w*LinearKernel)")
FP8_KV_SAYS = re.compile(r"Using fp8 data type to store kv cache|kv_cache_dtype=torch\.float8", re.I)


def backend_facts(log: str) -> dict:
    """What the log names, from the real lines of vLLM 0.34: the TEXT layers' attention backend ("Using FLASHINFER attention backend out of potential backends: [...]"), the decode backend and the KV dtype
    FlashInfer resolved, the GDN prefill kernel, the linear (Marlin) kernel; the vision encoder's attention line is kept apart (it is not the text layers). `summary` is the sentence of the report."""
    log = log or ""
    m, d, k, g, l_, v = TEXT_BACKEND.search(log), DECODE_BACKEND.search(log), FLASHINFER_KV.search(log), GDN_KERNEL.search(log), LINEAR_KERNEL.search(log), VIT_BACKEND.search(log)
    out = {"text_attention": m.group(1) if m else None, "decode_backend": d.group(1) if d else None, "flashinfer_kv_dtype": k.group(1) if k else None, "gdn_prefill_kernel": g.group(1) if g else None,
           "linear_kernel": l_.group(1) if l_ else None, "vision_encoder_line": v.group(0).strip()[:200] if v else None, "log_says_fp8_kv": bool(FP8_KV_SAYS.search(log))}
    bits = [f"attention backend (text layers): {out['text_attention'] or 'not found in the log'}" + (f" ({out['decode_backend']} decode)" if out["decode_backend"] else "")]
    if out["gdn_prefill_kernel"]:
        bits.append(f"GDN prefill kernel: {out['gdn_prefill_kernel']}")
    if out["linear_kernel"]:
        bits.append(f"linear kernel: {out['linear_kernel'].replace('LinearKernel', '')}")
    bits.append(f"KV dtype in the log: {out['flashinfer_kv_dtype'] or ('fp8 (\"Using fp8 data type to store kv cache\")' if out['log_says_fp8_kv'] else 'not named (auto)')}")
    out["summary"] = ", ".join(bits)
    return out


def kv_summary(label: str, log: str, extra: dict | None = None, requested: str | None = None) -> dict:
    """What one engine start says about its KV cache: tokens, concurrency, the backends the log names, the dtype that was PASSED (`requested`) and whether the log confirms fp8, and whether the log refuses fp8."""
    kv = vllmengine.kv_facts(log)
    bf = backend_facts(log)
    refused = FP8_REFUSED.search(log or "")
    return {"setting": label, "requested_dtype": requested or label, "log_confirms_fp8": bf["log_says_fp8_kv"], **kv, "attention_backend": bf["summary"], "backends": bf,
            "refuses_fp8": refused.group(0).strip()[:200] if refused else None, "highlights": vllmengine.log_highlights(log, 40), **(extra or {})}


def kv_fp8_pass(cfg: dict, model: str, quant: str, util: float, extra_args, rep: dict, first_engine, reqs: list, by_id: dict, settings: dict, should_stop, model_name: str) -> None:
    """The optional second pass ("also test the 8-bit KV cache"): stops the first engine, starts the same model with `--kv-cache-dtype fp8`, and reports for BOTH settings the KV cache size in tokens and the maximum
    concurrency, the speed on the same long-answer test and the Quick suite's rates (think_rate, closed_rate, accuracy), the attention backend the log names, and whether the log refuses fp8. A start that fails
    (or a log that refuses fp8) is a plain message, never a hang (the start fails fast and times out), and the result says the setting stays "auto". Writes `rep["kv_fp8"]`."""
    out: dict = {"auto": kv_summary("auto", rep.get("_full_log", ""), {"long_tokens_per_s": rep.get("long_tokens_per_s"), "think_rate": (rep.get("suite") or {}).get("think_rate"),
                                                                     "closed_rate": (rep.get("suite") or {}).get("closed_rate"), "accuracy": (rep.get("suite") or {}).get("accuracy")}, requested="auto")}
    rep["kv_fp8"] = out
    if out["auto"]["log_confirms_fp8"]:
        out["comparison_refused"] = ("The log of the \"auto\" run says it used the 8-bit KV cache (\"Using fp8 data type to store kv cache\"), so the two runs are not different settings: no comparison is printed. "
                                     "Check that nothing else passes --kv-cache-dtype fp8 (the engine's extra arguments, the engine settings).")
    first_engine.stop()
    eng = vllmengine.Engine(cfg, model, quant=quant, gpu_util=util, extra_args=extra_args, kv_cache_dtype="fp8")
    try:
        eng.start()
        eng.wait_ready(float(cfg.get("start_timeout") or vllmengine.DEFAULT_START_TIMEOUT), should_stop)
        list_sample(rep, cfg, "right after /health answered (the fp8 engine)", vllmengine.container_name(eng.port))
        log = eng.logs_big(800) + "\n" + eng.log_text()
        fp8 = kv_summary("fp8", log, requested="fp8")
        rep["start_command_fp8"] = eng.commands[0] if eng.commands else None
        results, tokens, t0 = [], 0, time.monotonic()
        with ThreadPoolExecutor(max_workers=thinkcheck_vllm.DEFAULT_PARALLEL) as pool:
            for raw in pool.map(lambda r: thinkcheck_vllm.ask(eng.url, r, settings, model_name, eng.api_key), reqs):
                item, is_main = by_id[raw["id"]]
                tokens += int(raw.get("gen_tokens") or 0)
                results.append({**thinkcheck.score_item(item, raw, thinking=is_main), "control": not is_main})
        suite = thinkcheck.summarize([r for r in results if not r["control"]])
        fp8.update(think_rate=suite.get("think_rate"), closed_rate=suite.get("closed_rate"), accuracy=suite.get("accuracy"), suite_tokens_per_s=round(tokens / max(time.monotonic() - t0, 1e-9), 1))
        fp8.update(long_decode(eng, should_stop, model_name))
        out["fp8"] = fp8
        a, b = out["auto"].get("kv_tokens"), fp8.get("kv_tokens")
        if out.get("comparison_refused"):
            out["text"] = out["comparison_refused"]
        elif not fp8["log_confirms_fp8"] and not fp8.get("refuses_fp8"):
            out["text"] = "The fp8 engine's log does NOT say it used an 8-bit KV cache (no \"Using fp8 data type to store kv cache\"): the numbers below are not an fp8 measurement."
        else:
            out["text"] = (f"KV cache: {a:,} tokens with auto, {b:,} with fp8 ({b / a:.2f} times)." if a and b else "The KV cache size was not found in the logs.")
        bn = backend_note(out)
        if bn:
            out["backend_note"] = bn                      # additive: the comparison is the cache type AND the backend
        if fp8.get("refuses_fp8"):
            out["warning"] = f"The log mentions a problem with fp8: {fp8['refuses_fp8']}"
    except vllmengine.EngineError as exc:
        logs = eng.exit_logs or eng.logs_tail(60)
        refused = FP8_REFUSED.search(logs or "")
        out["fp8_failed"] = f"vLLM did not run with the 8-bit KV cache{' (the log refuses it: ' + refused.group(0).strip()[:160] + ')' if refused else ''}: {str(exc)[:300]}. The setting stays auto."
        out["fallback"] = "auto"
        out["exit_logs"] = logs
    except Exception as exc:                                  # noqa: BLE001 - the second pass must never lose the report
        out["fp8_failed"] = f"The fp8 pass failed ({type(exc).__name__}: {protocol.scrub(str(exc))[:200]}). The setting stays auto."
        out["fallback"] = "auto"
    finally:
        if eng.port and not out.get("fp8_failed"):
            list_sample(rep, cfg, "right before the stop (the fp8 engine)", vllmengine.container_name(eng.port))
        eng.stop()
        rep["fp8_commands"] = eng.commands
        rep["fp8_engine_name"] = vllmengine.container_name(eng.port) if eng.port else None


def remove_leftovers(engine) -> dict:
    """After every stop: a final `list -a` (wslc) / `ps -a` (docker) and a removal of any `ndforge-vllm-*` container still there, by its own name; says what it found. Never raises."""
    out = {"found": [], "removed": [], "listing": None}
    try:
        if engine.kind == "wslc":
            r = _run(["wslc", "list", "-a", "--format", "json"], 60)
            if r["returncode"] != 0 or vllmengine.parse_list(r["output"]) is None:
                r = _run(["wslc", "list", "-a"], 60)                 # the table, when the JSON is not there
            names = vllmengine.list_names(r["output"], "ndforge-vllm-")
            tmpl = vllmengine.wslc_templates(engine.cfg)["remove"]
            mk = lambda n: vllmengine.render_template(tmpl, {**vllmengine.wslc_values(engine.cfg, 0, "", []), "name": n})      # noqa: E731
        elif engine.kind in ("docker", "docker_wsl"):
            pre = vllmengine.docker_prefix(engine.cfg)
            r = _run(pre + ["ps", "-a", "--filter", "name=ndforge-vllm-", "--format", "{{.Names}}"], 60)
            names = re.findall(r"ndforge-vllm-\d+", r["output"])
            mk = lambda n: pre + ["rm", "-f", n]      # noqa: E731
        else:
            return out
        out["listing"] = tidy(r["output"])[-1500:]
        for n in sorted(set(names)):
            out["found"].append(n)
            res = _run(mk(n), 60)
            if res["returncode"] == 0:
                out["removed"].append(n)
    except Exception as exc:                                  # noqa: BLE001 - a clean-up check must never lose the report
        out["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return out


def startup_parts(ready_s, log: str) -> dict:
    """What the engine's log says about the cache and where the start-up went: `kv` (KV cache tokens, concurrency, the note), `init_engine_s` and the sentence "container + model load X s, engine init Y s"."""
    out: dict = {}
    kv = vllmengine.kv_facts(log)
    if kv.get("kv_tokens"):
        fit = kv["kv_tokens"] // 3300
        kv["note"] = (f"The KV cache holds {kv['kv_tokens']:,} tokens" + (f" (maximum concurrency {kv['max_concurrency']}x for {kv['per_request_tokens']:,}-token requests)" if kv.get("max_concurrency") else "")
                      + f": with answers of about 3.3k tokens only about {fit} fit at once; the other requests wait, so 40 questions at once are served in waves.")
        out["kv"] = kv
    init = vllmengine.init_engine_seconds(log)
    if init is not None and ready_s:
        out["init_engine_s"] = init
        out["startup_breakdown"] = f"startup: container + model load {round(float(ready_s) - init, 1)} s, engine init (profile, create kv cache, warmup model) {init} s"
    return out


def thinking_samples(raws: list, main_ids: set) -> list:
    """One short and one long THINKING answer: the start (300 characters) and the place where the `<think>` block ends, so both ends of the block are visible."""
    mains = sorted([r for r in raws if r["id"] in main_ids and r.get("text")], key=lambda r: r.get("gen_tokens") or 0)
    pick = [mains[0], mains[-1]] if len(mains) > 1 else mains
    out = []
    for r in pick:
        t = r["text"]
        i = t.find("</think>")
        out.append({"id": r["id"], "gen_tokens": r.get("gen_tokens"), "start": t[:300], "end": t[max(0, i - 150): i + 150] if i >= 0 else "(no </think> in this answer)"})
    return out


def tidy(text: str) -> str:
    """Captured console output: CRLF and lone CR become newlines and runs of blank lines collapse (wslc prints CRLF that was converted twice)."""
    t = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", t)


def _write(out_dir: Path, rep: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # the report leaves the PC (the person pastes it into an issue): no user name, no home folder in it
    (out_dir / "vllm_selftest.json").write_text(protocol.redact_user(protocol.scrub(json.dumps(rep, ensure_ascii=False, indent=1))), encoding="utf-8")
    (out_dir / "vllm_selftest.md").write_text(tidy(protocol.redact_user(protocol.scrub(markdown(rep)))), encoding="utf-8")


def reachability(engine: "vllmengine.Engine") -> dict:
    """From this side (Windows): does `<url>/health` answer? If not: the WSL IP workaround, as a plain hint."""
    try:
        engine.wait_ready(5)
        return {"ok": True, "detail": f"{engine.url}/health answered from this side."}
    except vllmengine.EngineError:
        hint = ""
        if engine.kind in ("docker_wsl", "wslc", "wsl"):
            r = _run(["wsl.exe", "hostname", "-I"], 30)
            hint = f" Workaround: use the WSL address instead (`wsl.exe hostname -I` says: {r['output'].strip()[:80] or 'nothing'}) as the server address (way: url). If WSL's networking mode is 'mirrored' that may matter (unverified)."
        return {"ok": False, "detail": f"{engine.url}/health did not answer from this side.{hint}"}


def run(job: dict) -> int:
    protocol.reserve_stdout()
    t_start = time.monotonic()
    model = str(job.get("model") or DEFAULT_MODEL)
    adapter = str(job["adapter"]) if job.get("adapter") else None
    out_dir = Path(job["out_dir"]) if job.get("out_dir") else Path(".")
    stop_file = Path(job["stop_file"]) if job.get("stop_file") else None
    should_stop = (lambda: stop_file.exists()) if stop_file else (lambda: False)
    if not vllmengine.is_local_path(model) and not job.get("confirm_download"):
        raise ThinkCheckError(f"{download_note(model)} Confirm the download first (tick the box), or choose a model folder.", "vllm_selftest_failed")
    try:
        cfg = vllmengine.engine_config(job)
    except vllmengine.EngineError as exc:
        raise ThinkCheckError(str(exc), "vllm_selftest_failed") from exc
    rep: dict = {"generic_test": generic_note(model), "version": REPORT_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note": NOTE, "model": model, "adapter": adapter, "download_note": download_note(model),
                 "engine": {k: cfg.get(k) for k in ("kind", "image", "distro", "venv", "url", "version")}, "prerequisites": [], "comparisons": [], "hints": [], "commands": [], "stopped": False}
    engine = None
    try:
        protocol.progress("selftest", 0.05, "Checking the prerequisites")
        smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], 30)
        gpu = [x.strip() for x in smi["output"].splitlines()[0].split(",")] if smi["returncode"] == 0 and smi["output"].strip() else []
        vram_gb = float((job.get("engine") or {}).get("vram_gb") or 0) or (round(int(gpu[1]) / 1024, 1) if len(gpu) > 1 and gpu[1].isdigit() else None)
        rep["gpu"] = {"name": gpu[0] if gpu else None, "memory_gb": vram_gb, "driver": gpu[2] if len(gpu) > 2 else None}
        rep["model_info"] = model_facts(model, thinkcheck_vllm.quant_of(job))
        rep["model_params_b"], rep["model_quant"] = rep["model_info"]["params_b"], rep["model_info"]["quant"]
        rep["timings"] = {}
        refuse = fit_problem(rep["model_info"], vram_gb)
        if refuse:
            rep["error"] = refuse                           # nothing is started and nothing is waited for
            return _finish(rep, out_dir, t_start)
        t_pre = time.monotonic()
        rep["prerequisites"] = prerequisites(cfg, model, bool(job.get("gpu_test", True)), float((job.get("engine") or {}).get("gpu_memory_utilization") or vllmengine.DEFAULT_GPU_UTIL))
        gfree = next((p for p in rep["prerequisites"] if p["id"] == "gpu_free"), None)
        util_now = float(gfree["gpu_util"]) if gfree and gfree.get("gpu_util") else float((job.get("engine") or {}).get("gpu_memory_utilization") or vllmengine.DEFAULT_GPU_UTIL)
        rep["gpu_util"] = util_now
        refuse = fit_problem(rep["model_info"], vram_gb, util=util_now)          # the card's free memory counts: usable = utilization x card - cache
        if refuse:
            rep["error"] = refuse
            return _finish(rep, out_dir, t_start)
        rep["timings"]["prerequisites_s"] = round(time.monotonic() - t_pre, 1)
        if cfg["kind"] == "wslc" or vllmengine._which("wslc"):
            rep["wslc"] = {"help": vllmengine.wslc_help()}
            ver = (rep["wslc"]["help"].get("wslc --version") or {}).get("output", "").strip().splitlines()
            rep["engine"]["wslc_version"] = ver[0].strip() if ver else None
        if should_stop():
            rep["stopped"] = True
            return _finish(rep, out_dir, t_start)
        blocked = [p for p in rep["prerequisites"] if p["id"] in ("gpu_passthrough", "gpu_free") and p["ok"] is False]
        if blocked:
            rep["error"] = blocked[0]["detail"]
            return _finish(rep, out_dir, t_start)
        quick = bool(job.get("quick", True))
        settings = thinkcheck.settings_of({"quick": quick})
        main, control = thinkcheck.questions(thinkcheck.load_suite(), settings)
        reqs = thinkcheck.requests_for(main, control, settings)
        e = job.get("engine") or {}
        engine = vllmengine.Engine(cfg, model, adapter=adapter, quant=thinkcheck_vllm.quant_of(job), extra_args=e.get("extra_args"), gpu_util=util_now,
                                   kv_cache_dtype="auto" if job.get("kv_fp8_test") else None)          # the baseline of the fp8 pass is "auto" whatever the setting says
        protocol.progress("selftest", 0.15, "Starting the engine (a first start downloads the image and the model)")
        t_wait = time.monotonic()
        try:
            engine.start()
            engine.wait_ready(float(job.get("start_timeout") or vllmengine.DEFAULT_START_TIMEOUT), should_stop)
        except vllmengine.EngineError as exc:
            rep["error"] = str(exc)
            rep["timings"]["waited_for_ready_s"] = round(time.monotonic() - t_wait, 1)
            list_sample(rep, cfg, "in the failure path, before the container is removed", vllmengine.container_name(engine.port))
            rep["exit_logs"] = engine.exit_logs or engine.logs_tail(60)       # read before `_finish` removes the container
            rep["highlights"] = vllmengine.log_highlights(rep["exit_logs"] + "\n" + engine.log_text())
            return _finish(rep, out_dir, t_start, engine)
        full_log = engine.logs_big(800) + "\n" + engine.log_text()
        rep["_full_log"] = full_log
        rep.update(startup_seconds=engine.ready_seconds, server_version=engine.server_version(), highlights=vllmengine.log_highlights(full_log, 60))
        rep.update(startup_parts(engine.ready_seconds, full_log))
        list_sample(rep, cfg, "right after /health answered", vllmengine.container_name(engine.port))
        rep["reachability"] = reachability(engine)
        if should_stop():
            rep["stopped"] = True
            return _finish(rep, out_dir, t_start, engine)
        protocol.progress("selftest", 0.4, "Asking the thinking questions, 40 at once")
        results, tokens = [], 0
        by_id = {it["id"]: (it, True) for it in main} | {it["id"]: (it, False) for it in control}
        model_name = vllmengine.ADAPTER_NAME if adapter else vllmengine.SERVED_NAME
        t0 = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=thinkcheck_vllm.DEFAULT_PARALLEL)
        try:
            pending = {pool.submit(thinkcheck_vllm.ask, engine.url, r, settings, model_name, engine.api_key) for r in reqs}
            raws = []
            while pending and not should_stop():
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for f in done:
                    raw = f.result()
                    tokens += int(raw.get("gen_tokens") or 0)
                    item, is_main = by_id[raw["id"]]
                    results.append({**thinkcheck.score_item(item, raw, thinking=is_main), "control": not is_main})
                    raws.append(raw)
                    protocol.progress("selftest", 0.4 + 0.5 * len(results) / len(reqs), f"Question {len(results)} of {len(reqs)}")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        spent = time.monotonic() - t0
        if len(results) < len(reqs):
            rep["stopped"] = True
        else:
            mains = [r for r in results if not r["control"]]
            rep["suite"] = thinkcheck.summarize(mains)
            rep.update(wall_seconds=round(spent, 1), completion_tokens=tokens, tokens_per_s=round(tokens / spent, 1) if spent > 0 and tokens else None)
            rep["samples"] = [{"id": r["id"], "text": r["text"], "gen_tokens": r.get("gen_tokens")} for r in raws[:5]]
            main_ids = {it["id"] for it in main}
            rep["answer_tokens"] = answer_stats([r.get("gen_tokens") or 0 for r in raws if r["id"] in main_ids])
            rep["think_samples"] = thinking_samples(raws, main_ids)
            rep["think_blocks"] = sum(1 for r in raws if r["id"] in main_ids and ("</think>" in (r.get("text") or "") or "<think>" in (r.get("text") or "")))
            rep["request_check"] = {"enable_thinking_sent": True, "same_request_as_the_thinking_check": True, "note": "Sent through thinkcheck_vllm.ask, the request of the real vLLM check: enable_thinking true for the questions, false for the controls; the `<think>` prefix is written by the server's chat template (see the samples)."}
            protocol.progress("selftest", 0.92, "Measuring the speed on long answers")
            if not should_stop():
                rep.update(long_decode(engine, should_stop, model_name))
            if job.get("kv_fp8_test") and not should_stop() and engine.kind != "url":
                protocol.progress("selftest", 0.95, "Testing the 8-bit KV cache (a second engine start)")
                kv_fp8_pass(cfg, model, thinkcheck_vllm.quant_of(job), util_now, e.get("extra_args"), rep, engine, reqs, by_id, settings, should_stop, model_name)
            refs: list = []
            for label in ("hf", "llama") + (("hf_adapter",) if adapter else ()):
                saved = _load_result((job.get("compare") or {}).get(label)) if (job.get("compare") or {}).get(label) else None
                if saved:
                    rep["comparisons"].append(compare(rep["suite"], saved, label, model))
                    refs.append(saved)
            rep["answer_check"] = assess_answers(rep["answer_tokens"], rep["suite"], rep["think_blocks"], rep["answer_tokens"]["n"], refs[0] if refs else None)
            rep["short_answers"] = rep["answer_check"]["level"] == "warn"
    except Exception as exc:                          # noqa: BLE001 - the report must still be written and the engine stopped (a ThinkCheckError too)
        rep["error"] = f"{type(exc).__name__}: {protocol.scrub(str(exc))[:300]}"
    return _finish(rep, out_dir, t_start, engine)


def _finish(rep: dict, out_dir: Path, t_start: float, engine=None) -> int:
    if engine is not None:
        if engine.kind == "wslc" and not rep.get("error") and engine.port and not rep.get("kv_fp8"):
            list_sample(rep, {"kind": "wslc"}, "right before the stop", vllmengine.container_name(engine.port))
        engine.stop()                                 # explicit stop, always (also after a failure and a Stop)
        rep["commands"] = engine.commands                 # the start command is in it, the API key replaced by [key]
        rep["engine_stopped"] = engine.stopped
        rep["start_command"] = engine.commands[0] if engine.commands else None
        rep["kv_requested"] = {"first": engine.kv_cache_dtype, "second": "fp8" if rep.get("kv_fp8") else None}
        rep["leftovers"] = remove_leftovers(engine)
        if engine.exit_logs and not rep.get("exit_logs"):
            rep["exit_logs"] = engine.exit_logs
    if engine is not None and engine.kind == "wslc":
        rep.setdefault("hints", []).append(WSLC_VERIFIED)
    rep.pop("_full_log", None)
    rep["criteria"] = criteria_of(rep) if not rep.get("error") and not rep.get("stopped") else [{**c, "value": None, "ok": None, "not_reached": True} for c in criteria_of({})]
    rep["verdict"], rep["verdict_text"] = verdict(rep)
    if (rep.get("answer_check") or {}).get("level") == "warn":
        rep["verdict_text"] += " " + rep["answer_check"]["text"]
    rep["seconds"] = round(time.monotonic() - t_start, 1)
    _write(out_dir, rep)
    protocol.result({k: rep[k] for k in ("verdict", "verdict_text", "criteria", "startup_seconds", "tokens_per_s", "wall_seconds", "suite", "comparisons", "stopped", "seconds", "answer_tokens", "answer_check", "think_blocks", "short_answers", "long_tokens_per_s", "long_wall_seconds", "model_info", "model_params_b", "model_quant", "gpu") if k in rep}
                    | {"report": str(out_dir / "vllm_selftest.json"), "report_md": str(out_dir / "vllm_selftest.md")})
    return 0
