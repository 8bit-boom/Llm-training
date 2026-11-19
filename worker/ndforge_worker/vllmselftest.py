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


def prerequisites(cfg: dict, model: str, gpu_test: bool = True) -> list:
    """[{"id", "ok": True | False | None, "detail", "output"?}]: None = unknown (continuing)."""
    out = []
    smi = _run(["nvidia-smi", "--query-gpu=driver_version,memory.used,memory.total", "--format=csv,noheader,nounits"], 30)
    parts = [x.strip() for x in smi["output"].splitlines()[0].split(",")] if smi["returncode"] == 0 and smi["output"].strip() else []
    if parts:
        major = int(parts[0].split(".")[0]) if parts[0].split(".")[0].isdigit() else None
        out.append({"id": "driver", "ok": None if major is None else major >= 580, "detail": f"NVIDIA driver {parts[0]} (580 or newer is wanted for the CUDA 13 images)"})
        used = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        out.append({"id": "gpu_free", "ok": None if used is None else used <= 2048, "detail": f"{used} MB of the card are in use (vLLM reserves about 88 %: close other programs)" if used is not None else "the memory in use could not be read"})
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


def compare(main: dict, saved: dict, label: str) -> dict:
    """Our numbers against a saved result's: within noise = every rate differs by at most 2 questions."""
    n = max(1, int(main.get("n") or 1))
    tol = NOISE_QUESTIONS / n
    deltas = {k: round(float(main.get(k, 0) or 0) - float(saved["main"].get(k, 0) or 0), 4) for k in ("closed_rate", "accuracy", "loop_rate")}
    return {"with": label, "backend": saved.get("backend"), "model": saved.get("model"), "deltas": deltas, "tolerance": round(tol, 4), "within_noise": all(abs(v) <= tol + 1e-9 for v in deltas.values()),
            "other_model": True}


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
    out.append({"id": "speed", "label": f"at least {int(CRITERIA['tokens_per_s'])} tokens/s with 40 requests at once", "limit": CRITERIA["tokens_per_s"], "value": tps, "ok": None if tps is None else tps >= CRITERIA["tokens_per_s"]})
    for c in rep.get("comparisons", []):
        out.append({"id": "noise_" + c["with"], "label": f"numbers within noise of {c['with']} (+-{NOISE_QUESTIONS} questions)", "limit": c["tolerance"], "value": c["deltas"], "ok": c["within_noise"]})
    return out


def markdown(rep: dict) -> str:
    L = ["# vLLM self-test", "", f"**{rep['verdict'].upper()}**: {rep['verdict_text']}", "", f"> {NOTE}", "", f"Created {rep['created']} - model `{rep.get('model')}` - engine {rep.get('engine', {}).get('kind')} ({rep.get('engine', {}).get('image', '')})", ""]
    L += ["## Success criteria (from the report)", ""] + [f"- [{'x' if c['ok'] else ' ' if c['ok'] is False else '?'}] {c['label']}: {c['value']}" for c in rep.get("criteria", [])] + [""]
    L += ["## Prerequisites", ""] + [f"- [{'x' if p['ok'] else ' ' if p['ok'] is False else '?'}] {p['id']}: {p['detail']}" for p in rep.get("prerequisites", [])] + [""]
    if rep.get("download_note"):
        L += [rep["download_note"], ""]
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
        for k, v in rep["wslc"].get("help", {}).items():
            L += [f"### {k} (exit code {v['returncode']})", "", "```", v["output"], "```", ""]
        if rep["wslc"].get("loopback"):
            L += ["Loopback publish test: " + rep["wslc"]["loopback"], ""]
    if rep.get("hints"):
        L += ["## Hints", ""] + [f"- {h}" for h in rep["hints"]] + [""]
    L += ["## Commands that were run", ""] + [f"    {' '.join(map(str, c))}" for c in rep.get("commands", [])]
    return "\n".join(L) + "\n"


def _write(out_dir: Path, rep: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vllm_selftest.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "vllm_selftest.md").write_text(markdown(rep), encoding="utf-8")


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
    rep: dict = {"version": REPORT_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note": NOTE, "model": model, "adapter": adapter, "download_note": download_note(model),
                 "engine": {k: cfg.get(k) for k in ("kind", "image", "distro", "venv", "url", "version")}, "prerequisites": [], "comparisons": [], "hints": [], "commands": [], "stopped": False}
    engine = None
    try:
        protocol.progress("selftest", 0.05, "Checking the prerequisites")
        rep["prerequisites"] = prerequisites(cfg, model, bool(job.get("gpu_test", True)))
        if cfg["kind"] == "wslc" or vllmengine._which("wslc"):
            rep["wslc"] = {"help": vllmengine.wslc_help()}
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
        engine = vllmengine.Engine(cfg, model, adapter=adapter, quant=thinkcheck_vllm.quant_of(job), extra_args=e.get("extra_args"))
        protocol.progress("selftest", 0.15, "Starting the engine (a first start downloads the image and the model)")
        try:
            engine.start()
            engine.wait_ready(float(job.get("start_timeout") or vllmengine.DEFAULT_START_TIMEOUT), should_stop)
        except vllmengine.EngineError as exc:
            rep["error"] = str(exc)
            rep["highlights"] = vllmengine.log_highlights(engine.log_text())
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
            rep["samples"] = [{"id": r["id"], "text": r["text"]} for r in raws[:5]]
            for label in ("hf", "llama") + (("hf_adapter",) if adapter else ()):
                saved = _load_result((job.get("compare") or {}).get(label)) if (job.get("compare") or {}).get(label) else None
                if saved:
                    rep["comparisons"].append(compare(rep["suite"], saved, label))
    except ThinkCheckError:
        raise
    except Exception as exc:                          # noqa: BLE001 - the report must still be written and the engine stopped
        rep["error"] = f"{type(exc).__name__}: {protocol.scrub(str(exc))[:300]}"
    return _finish(rep, out_dir, t_start, engine)


def _finish(rep: dict, out_dir: Path, t_start: float, engine=None) -> int:
    if engine is not None:
        engine.stop()                                 # explicit stop, always (also after a failure and a Stop)
        rep["commands"] = engine.commands
        rep["engine_stopped"] = engine.stopped
    if engine is not None and engine.kind == "wslc":
        rep.setdefault("hints", []).append("wslc syntax unverified: the commands above are templates; adapt them from the help output in this report.")
    rep["criteria"] = criteria_of(rep) if not rep.get("error") and not rep.get("stopped") else []
    rep["verdict"], rep["verdict_text"] = verdict(rep)
    rep["seconds"] = round(time.monotonic() - t_start, 1)
    _write(out_dir, rep)
    protocol.result({k: rep[k] for k in ("verdict", "verdict_text", "criteria", "startup_seconds", "tokens_per_s", "wall_seconds", "suite", "comparisons", "stopped", "seconds") if k in rep}
                    | {"report": str(out_dir / "vllm_selftest.json"), "report_md": str(out_dir / "vllm_selftest.md")})
    return 0
