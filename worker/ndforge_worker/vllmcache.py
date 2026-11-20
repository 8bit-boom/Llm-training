"""The teacher cache made by vLLM, worker side (EXPERIMENTAL, off by default).

    BUILT FROM A RESEARCH REPORT, NOT RUN ON THE OWNER'S PC. The container commands of `wslc` (cp, exec, run -d with another entrypoint) are GUESSES with docker-like syntax: the
    self-test saves `wslc cp --help` and `wslc exec --help`, and every command is a template the owner can edit (`engine["wslc"]`, `engine["exec_templates"]`).

`run_cache` (distilljob.py) calls `run_chunk` when the job has `engine: {"kind": "vllm", "teacher": {...}}`. This module runs in OUR environment (no vLLM, no torch needed here):

    1. `select_teacher`   which weights the engine scores with. THE TEACHER MUST BE THE OWNER'S ORIGINAL MODEL. A pre-quantized checkpoint is a quantization of ONE repo (the configured
                          pair `{original repo id: quantized repo id}`, default `Qwen/Qwen3.8-27B` -> `cyankiwi/Qwen3.8-27B-AWQ-INT4`); it is used only for that original, or when the
                          owner confirmed "this repo is a quantized copy of my original" (`confirmed`, stored per original). Everything else: the in-flight bitsandbytes route on the owner's
                          own original folder (the same NF4 numerics as the standard teacher; UNVERIFIED how vLLM spells it and how fast it is). Never a silent fall back to the base AWQ.
    2. `identity`         `engine: "vllm-<version>"`, `teacher_quant: "bnb-nf4" | "awq:<repo>@<rev>"`, `teacher_origin: {repo, weights}`: goes into the cache's index and into the
                          shared store id, so a vLLM cache is never adopted into a store made by the standard teacher (and the other way round).
    3. `Container`        an idle container (docker, docker_wsl, wslc) or WSL, with the model; `cp_in`, `cp_out`, `exec` command templates; ALWAYS stopped and removed in a `finally`.
    4. `run_chunk`        stages rows.npz and the scripts INTO the container (no bind mount of the project folder), runs `vllm_cache.py` there by exec, follows its progress lines,
                          copies every finished part out while it runs and writes it into the worker's own cache writer (so a stop or a crash loses little), final sync, container removed.
"""
from __future__ import annotations

import json
import queue
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from . import protocol, vllmengine, vllmquant

DEFAULT_PAIRS = vllmquant.DEFAULT_PAIRS
BNB_ARGS = ["--quantization", "bitsandbytes"]                               # UNVERIFIED: the report says bitsandbytes moved into the plugin `vllm-bnb-plugin` 0.0.3; its flags are unknown
BNB_PIP = "vllm-bnb-plugin==0.0.3"                                          # UNVERIFIED: installed into the throw-away container before the run (needs network)
WORK = "/work"
SYNC_SECONDS = 60.0
READY_TIMEOUT = 180.0
LOAD_LINES = re.compile(r"(download|loading (model )?weights|weights took|model loading took|torch\.compile|compil|cuda graph|capturing|init engine|engine core|memory profiling|kv cache|"
                        r"warmup|application startup|quantiz|bitsandbytes|error|traceback)", re.I)
SCRIPT_FILES = ("vllm_cache.py", "cachewrite.py")


class VllmCacheError(Exception):
    """A plain-language reason why the vLLM teacher could not be used (the caller falls back to the standard teacher unless the owner required vLLM)."""


# ----------------------------------------------------------------------------------------------------------------- which weights (the identity rule)

_key = vllmquant.norm_id


def select_teacher(tcfg: dict, origin: str, origin_dir) -> dict:
    """Which weights the engine loads for the original `origin`: the ONE identity rule of `vllmquant.select_original` (shared with the thinking checks, trace generation and the self-test);
    a refusal is a VllmCacheError."""
    try:
        return vllmquant.select_original(tcfg, origin, origin_dir)
    except vllmquant.IdentityError as exc:
        raise VllmCacheError(str(exc)) from None


def identity(job: dict, origin: str, origin_dir) -> dict:
    """The keys a vLLM cache adds to its index (`cache_meta`) and to the shared store id: {"engine": "vllm-0.34.0", "teacher_quant": ..., "teacher_origin": {"repo", "weights"}}.
    A pure function of the job and the original's files: no GPU, no container."""
    from . import safetensors_io
    e = container_config(job)
    sel = select_teacher((job.get("engine") or {}).get("teacher") or {}, origin, origin_dir)
    out = {"engine": f"vllm-{e['version']}", "teacher_quant": sel["quant"], "teacher_origin": {"repo": str(origin), "weights": safetensors_io.weights_manifest(origin_dir)}}
    if teacher_kv(job) == "fp8":
        out["kv_cache"] = "fp8"
    return out


def teacher_kv(job: dict) -> str:
    """The KV cache type of the teacher's answers: "fp8" ONLY when the 8-bit KV cache is on AND the owner confirmed it for the teacher (`kv_cache_dtype_teacher`): the teacher's logits are the training
    target, so the exact cache is the default. A cache made with fp8 carries it in its identity and is never mixed with exact ones."""
    e = job.get("engine") or {}
    return "fp8" if str(e.get("kv_cache_dtype") or "auto").lower() == "fp8" and e.get("kv_cache_dtype_teacher") is True else "auto"


def container_config(job: dict) -> dict:
    """The engine settings (kind docker | docker_wsl | wslc | wsl | auto, image, distro, ...) of `job["engine"]["container"]`, as `vllmengine.engine_config` reads them (default: docker, the pinned image)."""
    return vllmengine.engine_config({"engine": dict((job.get("engine") or {}).get("container") or {})})


def wanted(job: dict) -> bool:
    e = job.get("engine")
    return isinstance(e, dict) and str(e.get("kind") or "").lower() == "vllm"


# ------------------------------------------------------------------------------------------------------------------------ commands (cp_in, cp_out, exec)

EXEC_DEFAULTS = {"cp_in": "{docker} cp {src} {name}:{dst}", "cp_out": "{docker} cp {name}:{src} {dst}", "exec": "{docker} exec {name} {args}",
                 "run_idle": "{docker} run -d --gpus all --ipc=host --name {name} {mounts} --entrypoint sleep {image} infinity"}
WSL_DEFAULTS = {"cp_in": "{wsl} cp {src} {dst}", "cp_out": "{wsl} cp {src} {dst}", "exec": "{wsl} {args}"}       # kind "wsl": no container, plain commands in the distro


def exec_templates(cfg: dict) -> dict:
    """The command templates of this engine kind: docker-like defaults (GUESS for wslc: its own keys in `engine["wslc"]`), overridable with `engine["exec_templates"]`."""
    kind = cfg["kind"]
    if kind == "wslc":
        t = vllmengine.wslc_templates(cfg)
        base = {k: t[k] for k in ("cp_in", "cp_out", "exec", "run_idle")}
    elif kind == "wsl":
        base = dict(WSL_DEFAULTS)
    else:
        base = dict(EXEC_DEFAULTS)
    over = cfg.get("exec_templates") or {}
    for k in over:
        if k not in base and k not in EXEC_DEFAULTS:
            raise vllmengine.EngineError(f"Unknown command template {k!r}. Known: {', '.join(EXEC_DEFAULTS)}.")
    return {**base, **{k: str(v) for k, v in over.items() if k in base}}


class Container:
    """The place the script runs: an idle container (docker / docker_wsl / wslc) or a WSL distro (wsl). `_popen` / `_run` are the hooks of vllmengine (tests replace them)."""

    def __init__(self, cfg: dict, model: str, *, hf_volume_dir: str | None = None):
        if cfg["kind"] == "url":
            raise VllmCacheError("The teacher cache cannot use an engine that is only an address (the url way): it needs a container or a WSL install it can copy files into.")
        self.cfg, self.model = cfg, model
        self.kind = cfg["kind"]
        self.port = vllmengine.free_port()
        self.name = vllmengine.container_name(self.port)
        self.t = exec_templates(cfg)
        self.started = False
        self.commands: list = []
        self.model_arg = model                                  # what the script's --model is, as the container sees it
        self.extra_llm: dict = {}

    # --- paths: Docker inside WSL and WSL want Linux paths
    def host(self, p) -> str:
        return vllmengine.wsl_path(str(p)) if self.kind in ("docker_wsl", "wsl") else str(p)

    def _values(self, **kw) -> dict:
        d = {"name": self.name, "docker": vllmengine.docker_prefix(self.cfg) if self.kind in ("docker", "docker_wsl") else [],
             "wsl": ["wsl.exe"] + (["-d", self.cfg["distro"]] if self.cfg.get("distro") else []) + ["--"], "image": self.cfg["image"], "mounts": [], "args": [], "src": "", "dst": ""}
        d.update(kw)
        return d

    def mounts(self) -> list:
        """[(host path or volume, container path, mode)]: a local model folder read-only (a Windows drive goes through a slow share: the caller warns), the Hub cache as a NAMED volume."""
        if vllmengine.is_local_path(self.model):
            self.model_arg = "/models/base" if self.kind != "wslc" else "/models"
            return [(self.host(self.model), "/models/base", "ro")]
        self.model_arg = self.model
        return [(self.cfg["hf_volume"], "/root/.cache/huggingface", "rw")] if self.kind != "wslc" else []

    def start_command(self) -> list:
        if self.kind == "wsl":
            self.model_arg = vllmengine.wsl_path(self.model) if vllmengine.is_local_path(self.model) else self.model
            return []
        mounts = self.mounts()
        if self.kind == "wslc":
            v = vllmengine.wslc_values(self.cfg, self.port, self.model if vllmengine.is_local_path(self.model) else self.cfg["hf_volume"], [])
            if not vllmengine.is_local_path(self.model):
                self.extra_llm = {"download_dir": "/models"}
            return vllmengine.render_template(self.t["run_idle"], v)
        flat: list = []
        for src, dst, mode in mounts:
            flat += ["-v", f"{src}:{dst}:{mode}"]
        return vllmengine.render_template(self.t["run_idle"], self._values(mounts=flat))

    def cp_in_command(self, src, dst) -> list:
        return vllmengine.render_template(self.t["cp_in"], self._values(src=self.host(src), dst=dst))

    def cp_out_command(self, src, dst) -> list:
        return vllmengine.render_template(self.t["cp_out"], self._values(src=src, dst=self.host(dst)))

    def exec_command(self, args: list) -> list:
        return vllmengine.render_template(self.t["exec"], self._values(args=[str(a) for a in args]))

    # --- running them
    def run(self, cmd: list, timeout: float = 600, what: str = "") -> str:
        self.commands.append(cmd)
        try:
            r = vllmengine._run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout, **vllmengine._win_kw())
        except (OSError, subprocess.SubprocessError) as exc:
            raise VllmCacheError(f"{what or cmd[0]} failed: {exc}") from exc
        out = ((getattr(r, "stdout", b"") or b"") + (getattr(r, "stderr", b"") or b"")).decode("utf-8", "replace")
        if getattr(r, "returncode", 0) != 0:
            raise VllmCacheError(f"{what or ' '.join(cmd[:3])} failed (exit code {r.returncode}): {protocol.scrub(out.strip()[-400:])}")
        return out

    def start(self, timeout: float = 3600) -> None:
        cmd = self.start_command()
        if cmd:
            self.started = True                                  # the container may exist even when the command fails half way: `stop` removes it either way
            self.run(cmd, timeout, "Starting the container")
        deadline = time.monotonic() + READY_TIMEOUT
        while True:
            try:
                self.run(self.exec_command(["true"]), 60, "The container did not answer")
                return
            except VllmCacheError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1.0)

    def stop(self) -> None:
        """Always the explicit stop and remove; never raises."""
        if self.kind in ("wsl",) and not self.started:
            return
        for cmd in vllmengine.stop_commands(self.cfg, self.port):
            self.commands.append(cmd)
            try:
                vllmengine._run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, timeout=vllmengine.STOP_TIMEOUT, **vllmengine._win_kw())
            except (OSError, subprocess.SubprocessError):
                pass
        self.started = False


# ----------------------------------------------------------------------------------------------------------------------------- the chunk

def script_args(job: dict, sel: dict, model_arg: str, extra_llm: dict, first: int, last: int, max_seq: int, k: int) -> list:
    """The command line of vllm_cache.py inside the container."""
    e = job.get("engine") or {}
    a = ["python3", f"{WORK}/vllm_cache.py", "--model", model_arg, "--data", f"{WORK}/data", "--out", f"{WORK}/out", "--meta", f"{WORK}/meta.json", "--k", str(k), "--max-seq", str(max_seq),
         "--first-row", str(first), "--last-row", str(last), "--part-tokens", str(int(job.get("part_tokens", 262144))), "--stop-file", f"{WORK}/STOP",
         "--gpu-util", str(float(e.get("gpu_memory_utilization") or vllmengine.DEFAULT_GPU_UTIL)), "--batch-tokens", str(int(e.get("batch_tokens") or 32768))]
    if sel["route"] == "bnb":
        tail = list((e.get("teacher") or {}).get("bnb_args") or BNB_ARGS)
        quant = tail[tail.index("--quantization") + 1] if "--quantization" in tail and tail.index("--quantization") + 1 < len(tail) else "bitsandbytes"
        a += ["--quantization", quant]
    kw = {**extra_llm, **(e.get("llm_kwargs") or {})}
    if teacher_kv(job) == "fp8":
        a += ["--kv-cache-dtype", "fp8"]
    if kw:
        a += ["--llm-kwargs", json.dumps(kw)]
    return a


class Syncer:
    """Copies the parts the container has finished out and writes their rows into the worker's cache writer, in order, once each."""

    def __init__(self, box: Container, writer, tmp: Path, first_row: int):
        self.box, self.writer, self.tmp = box, writer, tmp
        self.done_parts: set = set()
        self.first_row = first_row
        self.rows = 0

    def sync(self) -> int:
        """Ingests new parts; returns the rows added."""
        import numpy as np
        idx_path = self.tmp / "index.json"
        try:
            self.box.run(self.box.cp_out_command(f"{WORK}/out/index.json", idx_path), 120, "Copying the index out")
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
        except (VllmCacheError, OSError, ValueError):
            return 0                                             # nothing written yet (or a copy raced the writer): the next sync tries again
        added = 0
        for part in idx.get("parts", []):
            name = part["file"]
            if name in self.done_parts:
                continue
            dst = self.tmp / name
            self.box.run(self.box.cp_out_command(f"{WORK}/out/{name}", dst), 600, f"Copying {name} out")
            with np.load(dst) as z:
                ids, logp, resid, off = z["ids"], z["logp"], z["resid"], z["row_offsets"]
            for r in range(len(off) - 1):
                a, b = int(off[r]), int(off[r + 1])
                self.writer.add_row(ids[a:b], logp[a:b], resid[a:b])
                added += 1
            self.done_parts.add(name)
            try:
                dst.unlink()
            except OSError:
                pass
        self.rows += added
        return added


def _reader(proc, q: "queue.Queue") -> None:
    try:
        for line in proc.stdout:
            q.put(line if isinstance(line, str) else line.decode("utf-8", "replace"))
    except (OSError, ValueError):
        pass
    q.put(None)


def run_chunk(job: dict, writer, rows_dir, meta: dict, origin: str, origin_dir, first: int, last: int, k: int, max_seq: int, stop_file, *, on_progress=None, sleep=time.sleep) -> dict:
    """Scores rows `first..last` of `rows_dir/rows.npz` with vLLM and writes them into `writer` (the worker's cache writer, which already holds rows before `first`).
    -> {"engine", "teacher_quant", "startup_seconds", "tokens_per_s", "tokens", "load_phases", "stopped", "rows", "seconds", "notes"}; VllmCacheError when vLLM could not do it
    (the rows written before the failure stay in the cache). The container is ALWAYS stopped and removed."""
    e = container_config(job)
    tcfg = (job.get("engine") or {}).get("teacher") or {}
    sel = select_teacher(tcfg, origin, origin_dir)
    total, used = vllmengine.gpu_mb()
    g = vllmengine.plan_gpu_util(total, used, float((job.get("engine") or {}).get("gpu_memory_utilization") or vllmengine.DEFAULT_GPU_UTIL))
    if g["ok"] is False:
        raise VllmCacheError(g["text"])
    job = {**job, "engine": {**(job.get("engine") or {}), "gpu_memory_utilization": g["util"]}}
    box = Container(e, sel["model"])
    t0 = time.monotonic()
    result: dict = {"engine": f"vllm-{e['version']}", "teacher_quant": sel["quant"], "route": sel["route"], "reason": sel["reason"], "stopped": False, "notes": [],
                    "load_phases": [], "gpu": g, "startup_seconds": None, "rows": 0, "tokens": 0, "tokens_per_s": 0.0}
    if sel["route"] == "bnb" and vllmengine.under_mnt(str(origin_dir)) and box.kind in ("wsl", "docker_wsl"):
        result["notes"].append("The original's folder is on a Windows drive: the engine reads it through a slow share (9p); copying it into the WSL filesystem makes the start much faster.")
    tmp_ctx = tempfile.TemporaryDirectory(prefix="ndforge-vllm-")
    tmp = Path(tmp_ctx.name)
    proc = None
    try:
        protocol.progress("cache", 0.0, "Starting the vLLM container (the first time an image download and a model download can take long)")
        box.start(float(e.get("start_timeout") or vllmengine.DEFAULT_START_TIMEOUT) + 2100)
        if sel["route"] == "bnb":
            pip = str((tcfg or {}).get("bnb_pip") or BNB_PIP)
            if pip.lower() != "none":
                protocol.log(f"Installing {pip} into the container (the in-flight 4-bit route; UNVERIFIED: how vLLM 0.34 spells it is not known).")
                box.run(box.exec_command(["pip", "install", pip]), 1800, "Installing the bitsandbytes plugin")
        (tmp / "meta.json").write_text(json.dumps({**meta, "engine": result["engine"], "teacher_quant": sel["quant"]}), encoding="utf-8")
        box.run(box.exec_command(["mkdir", "-p", f"{WORK}/data", f"{WORK}/out"]), 60, "Making the work folders")
        here = Path(__file__).resolve().parent
        for f in SCRIPT_FILES:
            box.run(box.cp_in_command(here / f, f"{WORK}/{f}"), 120, f"Copying {f} into the container")
        box.run(box.cp_in_command(Path(rows_dir) / "rows.npz", f"{WORK}/data/rows.npz"), 1800, "Copying the rows into the container")
        box.run(box.cp_in_command(tmp / "meta.json", f"{WORK}/meta.json"), 120, "Copying the cache description into the container")
        staged = time.monotonic()
        result["stage_seconds"] = round(staged - t0, 1)
        cmd = box.exec_command(script_args(job, sel, box.model_arg, box.extra_llm, first, last, max_seq, k))
        box.commands.append(cmd)
        protocol.progress("cache", 0.0, "Loading the model into vLLM")
        try:
            proc = vllmengine._popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, bufsize=1, **vllmengine._win_kw())
        except OSError as exc:
            raise VllmCacheError(f"The script could not be started in the container: {exc}") from exc
        q: queue.Queue = queue.Queue()
        threading.Thread(target=_reader, args=(proc, q), daemon=True).start()
        syncer = Syncer(box, writer, tmp, first)
        last_sync = time.monotonic()
        stop_sent, script_t0, first_scoring = False, time.monotonic(), None
        failure = None
        while True:
            try:
                line = q.get(timeout=1.0)
            except queue.Empty:
                line = ""
            if line is None:
                break
            if line:
                obj = _parse(line)
                if obj is None:
                    if LOAD_LINES.search(line) and len(result["load_phases"]) < 80:
                        result["load_phases"].append({"t": round(time.monotonic() - script_t0, 1), "line": protocol.scrub(line.strip())[:300]})
                elif obj.get("type") == "log":
                    protocol.log(str(obj.get("text", "")))
                    _timing_line(result, str(obj.get("text", "")))
                elif obj.get("type") == "progress":
                    det = obj.get("detail") or {}
                    if det and first_scoring is None:
                        first_scoring = time.monotonic()
                        result["startup_seconds"] = round(first_scoring - t0, 1)
                        result["load_seconds"] = round(first_scoring - staged, 1)
                    protocol.progress(str(obj.get("step", "cache")), float(obj.get("value", 0.0)), str(obj.get("text", "")), det or None)
                    if on_progress:
                        on_progress(obj)
                elif obj.get("type") == "error":
                    failure = str(obj.get("message") or "vLLM failed")
                elif obj.get("type") == "result":
                    data = obj.get("data") or {}
                    result["tokens"] = int(data.get("tokens_scored") or 0)
                    result["stopped"] = bool(data.get("stopped"))
                    result["tokens_per_s"] = data.get("tokens_per_s") or result["tokens_per_s"]
                    result["notes"] += [str(x) for x in data.get("notes") or []]
            if stop_file is not None and Path(stop_file).exists() and not stop_sent:
                stop_sent = True
                try:
                    box.run(box.exec_command(["touch", f"{WORK}/STOP"]), 60, "Asking the script to stop")
                except VllmCacheError:
                    pass
            if time.monotonic() - last_sync >= SYNC_SECONDS:
                last_sync = time.monotonic()
                syncer.sync()
        try:
            proc.wait(60)
        except (subprocess.TimeoutExpired, OSError):
            pass
        syncer.sync()                                            # the last parts, also after a failure: a later run goes on after them
        code = getattr(proc, "returncode", 0)
        if failure:
            raise VllmCacheError(failure)
        if code not in (0, None):
            raise VllmCacheError(f"The vLLM script ended with exit code {code}. Its last lines: " + " | ".join(x["line"] for x in result["load_phases"][-4:]))
        result["stopped"] = bool(result["stopped"] or (stop_sent and writer.done_rows < last))
        result["rows"] = syncer.rows
        result["seconds"] = round(time.monotonic() - t0, 1)
        return result
    except vllmengine.EngineError as exc:
        raise VllmCacheError(str(exc)) from exc
    finally:
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except (OSError, subprocess.SubprocessError):
                pass
        box.stop()
        tmp_ctx.cleanup()


def _parse(line: str):
    s = line.strip()
    if not s.startswith("{"):
        return None
    try:
        obj = json.loads(s)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) and "type" in obj else None


def _timing_line(result: dict, text: str) -> None:
    if not text.startswith("timing: "):
        return
    try:
        rec = json.loads(text[len("timing: "):])
    except ValueError:
        return
    if rec.get("stage") in ("import_vllm", "engine_start_and_model_load", "vllm_scoring"):
        result.setdefault("timings", {})[rec["stage"]] = rec.get("seconds")
        if rec["stage"] == "vllm_scoring":
            result["tokens_per_s"] = rec.get("tokens_per_s") or result.get("tokens_per_s")
