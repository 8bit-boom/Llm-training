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


SHORT_MEDIAN_TOKENS = 300          # a thinking answer is hundreds to thousands of tokens: a median below this means the model did not think (or the template / limit differ)
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


def short_sentence(stats: dict) -> str:
    return (f"The answers are short (median {stats['median']} tokens, mean {stats['mean']}): either the model did not think, or max_tokens or the chat template differ from a real thinking check "
            "(look at `enable_thinking` and the `<think>` prefix in the sample answers). This speed does not predict a real thinking check; the long-answer speed below does.")


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
    if not crit:
        return "fail", "No measurement was made."
    if not failed:
        return "pass", "All the report's criteria were met on this PC: the engine is worth using for the thinking check and for making thinking answers (still check a few answers by eye)."
    if len(failed) < len(crit):
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
        out.append({"id": "noise_" + c["with"], "label": f"numbers within noise of {c['with']} (+-{NOISE_QUESTIONS} questions)", "limit": c["tolerance"], "value": c["deltas"], "ok": c["within_noise"]})
    return out


def markdown(rep: dict) -> str:
    L = ["# vLLM self-test", "", f"**{rep['verdict'].upper()}**: {rep['verdict_text']}", "", f"> {NOTE}", "", f"> {generic_note(rep.get('model'))}", "", f"Created {rep['created']} - model `{rep.get('model')}` - engine {rep.get('engine', {}).get('kind')} ({rep.get('engine', {}).get('image', '')})", ""]
    L += ["## Success criteria (from the report)", ""] + [f"- [{'x' if c['ok'] else ' ' if c['ok'] is False else '?'}] {c['label']}: {'not reached' if c.get('not_reached') else c['value']}" for c in rep.get("criteria", [])] + [""]
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
        L += ["## Start command (the API key is replaced)", "", "    " + " ".join(str(x) for x in rep["start_command"]), ""]
    if rep.get("exit_logs"):
        L += ["## Last lines of the container's log (read before it was removed)", ""] + [f"    {x}" for x in str(rep["exit_logs"]).splitlines()[-60:]] + [""]
    if rep.get("timings"):
        L += ["## Time so far", "", ", ".join(f"{k} {v}" for k, v in rep["timings"].items()) + f", whole test {rep.get('seconds', '?')} s.", ""]
    at, su = rep.get("answer_tokens"), rep.get("suite")
    if su:
        L += ["## Thinking suite on this engine", "", f"{rep.get('completion_tokens')} tokens generated in {rep.get('wall_seconds')} s ({rep.get('tokens_per_s')} tokens/s). Tokens per thinking answer: " + (f"mean {at['mean']}, median {at['median']}, 90th percentile {at['p90']} ({at['n']} answers)." if at else "not known."), "",
              "| think_rate | closed_rate | accuracy | loop_rate | truncated_rate |", "|---|---|---|---|---|", f"| {su.get('think_rate')} | {su.get('closed_rate')} | {su.get('accuracy')} | {su.get('loop_rate')} | {su.get('truncated_rate')} |", ""]
        if rep.get("short_answers") and at:
            L += [f"**{short_sentence(at)}**", ""]
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
        L += ["## Start-up", "", f"Ready after {rep['startup_seconds']} s. Server version: {rep.get('server_version') or 'unknown'}.", "", "Log lines about the attention / GDN kernel and the cache:", ""] + [f"    {x}" for x in rep.get("highlights", [])] + [""]
    if rep.get("reachability"):
        L += ["## Reachability from this side", "", rep["reachability"]["detail"], ""]
    if rep.get("tokens_per_s") is not None:
        s = rep.get("suite", {})
        L += ["## Thinking suite (Quick, 40 requests at once)", "", f"Wall time {rep.get('wall_seconds')} s, {rep.get('completion_tokens')} completion tokens, {rep['tokens_per_s']} tokens/s.", "",
              f"closed {s.get('closed_rate')}, accuracy {s.get('accuracy')}, loops {s.get('loop_rate')}, garbage {s.get('garbage_rate')}, cut by the limit {s.get('truncated_rate')}", ""]
    for c in rep.get("comparisons", []):
        L += [f"Against {c['with']} ({c.get('backend')}, model {c.get('model')}): differences {c['deltas']}, {'within' if c['within_noise'] else 'OUTSIDE'} the noise of +-{c['tolerance']}. The saved result may be of another model (the original): a difference is not necessarily a fault.", ""]
    if rep.get("samples"):
        L += ["## Sample answers (look at them: garbled text means a problem)", ""] + [f"### {x['id']}\n\n    {x['text'][:600]!r}\n" for x in rep["samples"]]
    if rep.get("wslc"):
        L += ["## wslc help (for adapting the templates)", "", "wslc syntax unverified.", ""]
        for smp in rep["wslc"].get("list_samples", []):
            L += [f"### `wslc list -a --format json` at {smp['at']} (exit code {smp['returncode']}): the real field names", "", "    " + (smp["output"] or "(empty)").replace("\n", "\n    "), ""]
        for k, v in rep["wslc"].get("help", {}).items():
            L += [f"### {k} (exit code {v['returncode']})", "", "```", v["output"], "```", ""]
        if rep["wslc"].get("loopback"):
            L += ["Loopback publish test: " + rep["wslc"]["loopback"], ""]
    if rep.get("hints"):
        L += ["## Hints", ""] + [f"- {h}" for h in rep["hints"]] + [""]
    L += ["## Commands that were run", ""] + [f"    {' '.join(map(str, c))}" for c in rep.get("commands", [])]
    return "\n".join(L) + "\n"


def list_sample(rep: dict, cfg: dict) -> None:
    """One `wslc list -a --format json` sample while the engine's container exists (and again in the failure path before it is removed): the real field names of the JSON."""
    if cfg.get("kind") != "wslc":
        return
    r = _run(["wslc", "list", "-a", "--format", "json"], 60)
    rep.setdefault("wslc", {}).setdefault("list_samples", []).append({"at": time.strftime("%H:%M:%S"), "output": tidy(r["output"])[-3000:], "returncode": r["returncode"]})


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
        engine = vllmengine.Engine(cfg, model, adapter=adapter, quant=thinkcheck_vllm.quant_of(job), extra_args=e.get("extra_args"), gpu_util=util_now)
        protocol.progress("selftest", 0.15, "Starting the engine (a first start downloads the image and the model)")
        t_wait = time.monotonic()
        try:
            engine.start()
            list_sample(rep, cfg)
            engine.wait_ready(float(job.get("start_timeout") or vllmengine.DEFAULT_START_TIMEOUT), should_stop)
        except vllmengine.EngineError as exc:
            rep["error"] = str(exc)
            rep["timings"]["waited_for_ready_s"] = round(time.monotonic() - t_wait, 1)
            list_sample(rep, cfg)
            rep["exit_logs"] = engine.exit_logs or engine.logs_tail(60)       # read before `_finish` removes the container
            rep["highlights"] = vllmengine.log_highlights(rep["exit_logs"] + "\n" + engine.log_text())
            return _finish(rep, out_dir, t_start, engine)
        rep.update(startup_seconds=engine.ready_seconds, server_version=engine.server_version(), highlights=vllmengine.log_highlights(engine.log_text()))
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
            rep["short_answers"] = rep["answer_tokens"]["n"] > 0 and rep["answer_tokens"]["median"] < SHORT_MEDIAN_TOKENS
            rep["request_check"] = {"enable_thinking_sent": True, "same_request_as_the_thinking_check": True, "note": "Sent through thinkcheck_vllm.ask, the request of the real vLLM check: enable_thinking true for the questions, false for the controls; the `<think>` prefix is written by the server's chat template (see the samples)."}
            protocol.progress("selftest", 0.92, "Measuring the speed on long answers")
            if not should_stop():
                rep.update(long_decode(engine, should_stop, model_name))
            for label in ("hf", "llama") + (("hf_adapter",) if adapter else ()):
                saved = _load_result((job.get("compare") or {}).get(label)) if (job.get("compare") or {}).get(label) else None
                if saved:
                    rep["comparisons"].append(compare(rep["suite"], saved, label, model))
    except Exception as exc:                          # noqa: BLE001 - the report must still be written and the engine stopped (a ThinkCheckError too)
        rep["error"] = f"{type(exc).__name__}: {protocol.scrub(str(exc))[:300]}"
    return _finish(rep, out_dir, t_start, engine)


def _finish(rep: dict, out_dir: Path, t_start: float, engine=None) -> int:
    if engine is not None:
        engine.stop()                                 # explicit stop, always (also after a failure and a Stop)
        rep["commands"] = engine.commands                 # the start command is in it, the API key replaced by [key]
        rep["engine_stopped"] = engine.stopped
        rep["start_command"] = engine.commands[0] if engine.commands else None
        if engine.exit_logs and not rep.get("exit_logs"):
            rep["exit_logs"] = engine.exit_logs
    if engine is not None and engine.kind == "wslc":
        rep.setdefault("hints", []).append("wslc syntax unverified: the commands above are templates; adapt them from the help output in this report.")
    rep["criteria"] = criteria_of(rep) if not rep.get("error") and not rep.get("stopped") else [{**c, "value": None, "ok": None, "not_reached": True} for c in criteria_of({})]
    rep["verdict"], rep["verdict_text"] = verdict(rep)
    if rep.get("short_answers") and rep.get("answer_tokens"):
        rep["verdict_text"] += " " + short_sentence(rep["answer_tokens"])
    rep["seconds"] = round(time.monotonic() - t_start, 1)
    _write(out_dir, rep)
    protocol.result({k: rep[k] for k in ("verdict", "verdict_text", "criteria", "startup_seconds", "tokens_per_s", "wall_seconds", "suite", "comparisons", "stopped", "seconds", "answer_tokens", "short_answers", "long_tokens_per_s", "long_wall_seconds", "model_info", "model_params_b", "model_quant", "gpu") if k in rep}
                    | {"report": str(out_dir / "vllm_selftest.json"), "report_md": str(out_dir / "vllm_selftest.md")})
    return 0
