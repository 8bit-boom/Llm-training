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
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import types
from collections import deque
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


KV_FP8_WARNING = ("8-bit KV cache for the teacher is on: measured on the owner's PC it gave 1.2x more cached tokens and no speed gain (525 against 520 tokens/s), the targets get a little numeric noise, "
                  "and the cache is stored under its own identity. Turn it off (Settings > vLLM > \"Also use it for the original's answers\") for exact targets.")


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


REMEDY_OFF = ("To go on without vLLM: turn off \"Make the original's answers with vLLM\" on the Heal card, or in Auto-cut open \"Change settings and continue\" and switch the vLLM teacher cache off "
              "(the standard teacher then answers).")
REMEDY = REMEDY_OFF + " \"Test the container copy\" on the Heal card finds out which way of getting files into the container works on this PC."
PROBE_TEXT = ("ndforge container copy probe 0123456789 " * 26)[:1024]             # 1 KB of plain text
CP_IN_VARIANTS = ("file", "dir", "touch", "fwd")                                  # `cp src name:/work/f`, `cp src name:/work/` (+ mv), `touch` first, forward slashes in the source
CP_OUT_VARIANTS = ("file", "dir", "fwd")


def cmd_text(cmd: list) -> str:
    """A command line as one readable string (words with a space in quotes), with secrets scrubbed."""
    return protocol.scrub(" ".join(f'"{a}"' if (" " in str(a) or not str(a)) else str(a) for a in cmd))


class Container:
    """The place the script runs: an idle container (docker / docker_wsl / wslc) or a WSL distro (wsl). `_popen` / `_run` are the hooks of vllmengine (tests replace them).

    Files go in and out by the first way that works on this PC (`start` finds out and records it in `self.route_in`, `self.route_out`, `self.tried`):
      1. `bind`   a host work folder mounted at /work (`-v <host>:/work`): nothing is copied by a container command at all. Several spellings of the host path are tried.
      2. `cp:*`   the `cp_in` / `cp_out` templates, in the file form, the directory form (`name:/work/`), after a `touch`, and with forward slashes in the source.
      3. `stdin`  `exec -i name sh -c 'cat > /work/f' < file` (and `exec name cat /work/f > file` out).
    Every failed command is reported with its exact command line, its stderr and a look inside the container (`diagnose`)."""

    def __init__(self, cfg: dict, model: str, *, hf_volume_dir: str | None = None, work_dir=None):
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
        self.work_dir = Path(work_dir) if work_dir is not None else None      # the host folder that is /work when the bind mount works
        self.bound: str | None = None                           # the spelling of the host folder that the container accepted
        self.route_in: str | None = None
        self.route_out: str | None = None
        self.tried: list = []                                   # [{"route", "ok", "detail"}] of every way that was tried
        self.start_seconds: float | None = None
        self._diagnosing = False

    # --- paths: Docker inside WSL and WSL want Linux paths; the others the plain Windows spelling (no \\?\ prefix)
    def host(self, p) -> str:
        return vllmengine.wsl_path(str(p)) if self.kind in ("docker_wsl", "wsl") else vllmengine.plain_path(str(p))

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

    def start_command(self, bind: str | None = None) -> list:
        if self.kind == "wsl":
            self.model_arg = vllmengine.wsl_path(self.model) if vllmengine.is_local_path(self.model) else self.model
            return []
        mounts = self.mounts()
        if self.kind == "wslc":
            v = vllmengine.wslc_values(self.cfg, self.port, self.host(self.model) if vllmengine.is_local_path(self.model) else self.cfg["hf_volume"], [])
            if not vllmengine.is_local_path(self.model):
                self.extra_llm = {"download_dir": "/models"}
            cmd = vllmengine.render_template(self.t["run_idle"], v)
        else:
            flat: list = []
            for src, dst, mode in mounts:
                flat += ["-v", f"{src}:{dst}:{mode}"]
            cmd = vllmengine.render_template(self.t["run_idle"], self._values(mounts=flat))
        if bind:
            at = cmd.index("run") + 1 if "run" in cmd else 1
            cmd = cmd[:at] + ["-v", f"{bind}:{WORK}"] + cmd[at:]       # the host work folder is /work
        return cmd

    def bind_spellings(self) -> list:
        """The ways the host work folder can be written for the engine: as it is, with forward slashes, and the WSL form `/mnt/c/...`. The first one the engine accepts is used and logged."""
        if self.work_dir is None or self.kind == "wsl":
            return []
        h = self.host(self.work_dir)
        out = [h, h.replace("\\", "/")]
        m = re.match(r"^([A-Za-z]):[\\/]+(.*)$", h)
        if m:
            out.append(f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}")
        seen: list = []
        for x in out:
            if x not in seen:
                seen.append(x)
        return seen

    def cp_in_command(self, src, dst, *, spell: str | None = None) -> list:
        return vllmengine.render_template(self.t["cp_in"], self._values(src=spell if spell is not None else self.host(src), dst=dst))

    def cp_out_command(self, src, dst, *, spell: str | None = None) -> list:
        return vllmengine.render_template(self.t["cp_out"], self._values(src=src, dst=spell if spell is not None else self.host(dst)))

    def exec_command(self, args: list, *, stdin: bool = False) -> list:
        cmd = vllmengine.render_template(self.t["exec"], self._values(args=[str(a) for a in args]))
        if stdin and "exec" in cmd:
            at = cmd.index("exec") + 1
            if "-i" not in cmd[at:at + 2]:
                cmd = cmd[:at] + ["-i"] + cmd[at:]
        return cmd

    # --- running them
    def run(self, cmd: list, timeout: float = 600, what: str = "", *, diagnose: bool = True, stdin_path=None, stdout_path=None, quiet: bool = False) -> str:
        self.commands.append(cmd)
        fin = fout = None
        try:
            kw: dict = {}
            kw["stdin"] = open(stdin_path, "rb") if stdin_path else subprocess.DEVNULL
            fin = kw["stdin"] if stdin_path else None
            if stdout_path:
                fout = open(stdout_path, "wb")
                kw.update(stdout=fout, stderr=subprocess.PIPE)
            else:
                kw["capture_output"] = True
            r = vllmengine._run(cmd, timeout=timeout, **kw, **vllmengine._win_kw())
        except (OSError, subprocess.SubprocessError) as exc:
            raise self._failure(cmd, what, str(exc), None, diagnose, quiet) from exc
        finally:
            for f in (fin, fout):
                if f is not None:
                    try:
                        f.close()
                    except OSError:
                        pass
        out = ((getattr(r, "stdout", b"") or b"") + (getattr(r, "stderr", b"") or b"")) if not stdout_path else (getattr(r, "stderr", b"") or b"")
        out = out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out)
        if getattr(r, "returncode", 0) != 0:
            raise self._failure(cmd, what, protocol.scrub(out.strip()[-600:]), r.returncode, diagnose, quiet)
        return out

    def _failure(self, cmd: list, what: str, detail: str, code, diagnose: bool, quiet: bool = False) -> VllmCacheError:
        head = f"{what or cmd[0]}: `{cmd_text(cmd)}` failed" + (f" (exit code {code})" if code is not None else "") + (f": {detail}" if detail else "")
        if not quiet:
            protocol.log(head)
        if diagnose and not quiet:
            head += " " + self.diagnose(cmd)
        return VllmCacheError(head)

    def diagnose(self, cmd: list | None = None) -> str:
        """Looks inside the container right after a failure: the work folder, the mounts, `cp --help` when a copy failed. Everything goes to the job log; a short form is returned for the message."""
        if self._diagnosing or not self.started and self.kind != "wsl":
            return ""
        self._diagnosing = True
        notes: list = []
        try:
            probes = [("ls -la /work", self.exec_command(["ls", "-la", WORK]))]
            if self.kind in ("docker", "docker_wsl"):
                probes.append(("mounts", vllmengine.docker_prefix(self.cfg) + ["inspect", "-f", "{{json .Mounts}}", self.name]))
            elif self.kind == "wslc":
                probes.append(("inspect", ["wslc", "inspect", self.name]))
                if cmd and any(str(x) == "cp" for x in cmd[:3]):
                    probes.append(("wslc cp --help", ["wslc", "cp", "--help"]))
            for label, c in probes:
                try:
                    text = self.run(c, 30, label, diagnose=False).strip()
                except VllmCacheError as exc:
                    text = f"(could not run: {str(exc)[:200]})"
                protocol.log(f"Diagnostics, {label}: `{cmd_text(c)}` ->\n{text[:1800]}")
                notes.append(f"{label}: " + " ".join(text.split())[:300])
        finally:
            self._diagnosing = False
        return ("Checked right then: " + "; ".join(notes) + ".") if notes else ""

    # --- start, and finding the way files go in and out
    def _start_once(self, timeout: float, bind: str | None) -> None:
        cmd = self.start_command(bind)
        if cmd:
            self.started = True                                  # the container may exist even when the command fails half way: `stop` removes it either way
            self.run(cmd, timeout, "Starting the container", diagnose=False)
        deadline = time.monotonic() + READY_TIMEOUT
        while True:
            try:
                self.run(self.exec_command(["true"]), 60, "The container did not answer", diagnose=False)
                return
            except VllmCacheError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1.0)

    def start(self, timeout: float = 3600) -> None:
        """Starts the container and finds a working way to move files (bind mount, cp, stdin). Raises a VllmCacheError that says what was tried and what to do."""
        t0 = time.monotonic()
        start_errors: list = []
        for sp in self.bind_spellings():
            try:
                self._start_once(timeout, sp)
            except VllmCacheError as exc:
                self.tried.append({"route": f"bind {sp}", "ok": False, "detail": f"the container did not start with it: {str(exc)[:300]}"})
                protocol.log(f"The host work folder as `{sp}` was refused: {str(exc)[:300]}")
                start_errors.append(str(exc))
                self.stop()
                continue
            ok, why = self._probe_bind(sp)
            self.tried.append({"route": f"bind {sp}", "ok": ok, "detail": why})
            if ok:
                self.bound = sp
                self.route_in = self.route_out = "bind"
                self._done_start(t0, f"the host work folder mounted as `{sp}` (no copy command is used)")
                return
            protocol.log(f"The host work folder as `{sp}` did not work: {why}")
            self.stop()
        try:
            self._start_once(timeout, None)
        except VllmCacheError as exc:
            self.tried.append({"route": "start", "ok": False, "detail": str(exc)[:400]})
            raise VllmCacheError(f"{exc} {self._summary()} {REMEDY}") from exc
        self.run(self.exec_command(["mkdir", "-p", WORK]), 60, "Making /work", diagnose=False) if self.kind != "wsl" else None
        self.route_in = self._probe_direction("in")
        self.route_out = self._probe_direction("out")
        if not self.route_in or not self.route_out:
            diag = self.diagnose()
            raise VllmCacheError(f"No way of moving files {'into' if not self.route_in else 'out of'} the container works. {self._summary()} {diag} {REMEDY}".replace("  ", " "))
        self._done_start(t0, f"cp/stdin: in by {self.route_in}, out by {self.route_out}")

    def _done_start(self, t0: float, how: str) -> None:
        self.start_seconds = round(time.monotonic() - t0, 1)
        protocol.log(f"The container is ready after {self.start_seconds} s; files go in and out by {how}.")

    def _summary(self) -> str:
        return "Tried: " + "; ".join(f"{t['route']}: {'worked' if t['ok'] else 'failed'}" + ("" if t["ok"] else f" ({' '.join(str(t['detail']).split())[:160]})") for t in self.tried) + "."

    def _probe_bind(self, spelling: str) -> tuple:
        """Writes 1 KB into the host work folder and asks the container to read it, and the other way round."""
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            (self.work_dir / "probe_in.txt").write_text(PROBE_TEXT, encoding="utf-8")
            got = self.run(self.exec_command(["cat", f"{WORK}/probe_in.txt"]), 60, "Reading the probe file in the container", diagnose=False)
            if got.strip() != PROBE_TEXT.strip():
                return False, "the container does not see the file written on the host (it sees something else: " + " ".join(got.split())[:100] + ")"
            self.run(self.exec_command(["sh", "-c", f"printf %s ok-from-container > {WORK}/probe_out.txt"]), 60, "Writing the probe file in the container", diagnose=False)
            for _ in range(10):
                f = self.work_dir / "probe_out.txt"
                if f.is_file() and f.read_text(encoding="utf-8", errors="replace").strip() == "ok-from-container":
                    return True, "1 KB went in and a line came back"
                time.sleep(0.3)
            return False, "the container wrote a file that does not appear in the host folder"
        except (VllmCacheError, OSError) as exc:
            return False, str(exc)[:300]

    def _probe_direction(self, direction: str) -> str | None:
        """Finds the first working route for one direction with a 1 KB file in /work, records every attempt."""
        routes = [f"cp:{v}" for v in (CP_IN_VARIANTS if direction == "in" else CP_OUT_VARIANTS)] + ["stdin"]
        tmp = Path(tempfile.mkdtemp(prefix="ndforge-probe-"))
        try:
            src = tmp / "probe_in.txt"
            src.write_text(PROBE_TEXT, encoding="utf-8")
            if direction == "out":
                try:
                    self.run(self.exec_command(["sh", "-c", f"printf %s '{PROBE_TEXT}' > {WORK}/probe_out.txt"]), 60, "Writing the probe file in the container", diagnose=False)
                except VllmCacheError as exc:
                    self.tried.append({"route": "out", "ok": False, "detail": str(exc)[:300]})
                    return None
            for route in routes:
                try:
                    if direction == "in":
                        self._put_route(route, src, "probe_in.txt")
                        got = self.run(self.exec_command(["cat", f"{WORK}/probe_in.txt"]), 60, "Reading the probe file in the container", diagnose=False)
                        ok = got.strip() == PROBE_TEXT.strip()
                    else:
                        dst = tmp / "probe_back.txt"
                        self._get_route(route, "probe_out.txt", dst)
                        ok = dst.is_file() and dst.read_text(encoding="utf-8", errors="replace").strip() == PROBE_TEXT.strip()
                    why = "1 KB went " + ("in" if direction == "in" else "out") if ok else "the file arrived changed or empty"
                except (VllmCacheError, OSError) as exc:
                    ok, why = False, str(exc)[:300]
                self.tried.append({"route": f"{route} ({direction})", "ok": ok, "detail": why})
                if ok:
                    return route
            return None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # --- moving files by the route that was found
    def put(self, src, rel: str, timeout: float = 600, what: str = "") -> None:
        """Copies the host file `src` to `/work/<rel>` in the container."""
        what = what or f"Copying {Path(str(src)).name} into the container"
        try:
            self._put_route(self.route_in or "cp:file", src, rel, timeout, what)
        except OSError as exc:
            raise VllmCacheError(f"{what}: {exc}. {REMEDY}") from exc

    def get(self, rel: str, dst, timeout: float = 600, what: str = "", quiet: bool = False) -> None:
        """Copies `/work/<rel>` of the container to the host file `dst`."""
        what = what or f"Copying {rel} out of the container"
        try:
            self._get_route(self.route_out or "cp:file", rel, dst, timeout, what, quiet)
        except OSError as exc:
            raise VllmCacheError(f"{what}: {exc}. {REMEDY}") from exc

    def _put_route(self, route: str, src, rel: str, timeout: float = 120, what: str = "") -> None:
        dst = f"{WORK}/{rel}"
        parent = dst.rsplit("/", 1)[0]
        if route == "bind":
            target = self.work_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
            return
        if route == "stdin":
            self.run(self.exec_command(["sh", "-c", f"mkdir -p {parent} && cat > {dst}"], stdin=True), timeout, what, stdin_path=src)
            return
        variant = route.split(":", 1)[1]
        base = Path(str(src)).name
        if variant == "dir":
            self.run(self.cp_in_command(src, parent + "/"), timeout, what)
            if base != dst.rsplit("/", 1)[1]:
                self.run(self.exec_command(["mv", f"{parent}/{base}", dst]), 60, what)
        elif variant == "touch":
            self.run(self.exec_command(["sh", "-c", f"mkdir -p {parent} && touch {dst}"]), 60, what)
            self.run(self.cp_in_command(src, dst), timeout, what)
        elif variant == "fwd":
            self.run(self.cp_in_command(src, dst, spell=self.host(src).replace("\\", "/")), timeout, what)
        else:
            self.run(self.cp_in_command(src, dst), timeout, what)

    def _get_route(self, route: str, rel: str, dst, timeout: float = 120, what: str = "", quiet: bool = False) -> None:
        src = f"{WORK}/{rel}"
        dst = Path(dst)
        if route == "bind":
            shutil.copyfile(self.work_dir / rel, dst)
            return
        if route == "stdin":
            self.run(self.exec_command(["cat", src]), timeout, what, stdout_path=dst, quiet=quiet)
            return
        variant = route.split(":", 1)[1]
        if variant == "dir":
            self.run(self.cp_out_command(src, str(dst.parent) + ("\\" if "\\" in str(dst) else "/")), timeout, what, quiet=quiet)
            got = dst.parent / src.rsplit("/", 1)[1]
            if got != dst and got.is_file():
                os.replace(got, dst)
        elif variant == "fwd":
            self.run(self.cp_out_command(src, dst, spell=self.host(dst).replace("\\", "/")), timeout, what, quiet=quiet)
        else:
            self.run(self.cp_out_command(src, dst), timeout, what, quiet=quiet)

    def evidence(self, script_lines=()) -> str:
        """Why a run failed, collected BEFORE the container is removed: the root-cause lines of the script's own output (every line it printed, kept by the caller) and of the container's log, written to the
        job log, and the last ones as text for the error."""
        lines = list(script_lines)
        rc = vllmengine.root_cause("\n".join(lines))
        try:
            if self.started and self.kind != "wsl":
                big = vllmengine.Engine.logs_big(types.SimpleNamespace(kind=self.kind, port=self.port, cfg=self.cfg, log_text=lambda: ""), 800)
                rc = (vllmengine.root_cause(big) + rc)[-25:] if big.strip() else rc
        except Exception:                                       # noqa: BLE001 - the evidence must never hide the failure itself
            pass
        if rc:
            protocol.log("Root cause lines (script and container log):\n" + "\n".join(rc))
        hints = vllmengine.hints_for("\n".join(rc) + "\n" + "\n".join(lines))
        for h in hints:
            protocol.log("Known cause: " + h)
        tail = [protocol.scrub(x.strip())[:300] for x in lines if x.strip()][-8:]
        return (("Root cause: " + " | ".join(rc[-12:]) + ". ") if rc else "No line of the engine's output names a cause. ") + (("What it probably means: " + " ".join(hints) + " ") if hints else "") + ("Last lines: " + " | ".join(tail) if tail else "The script printed nothing.")

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


def probe(job: dict) -> dict:
    """The small test behind \"Test the container copy\" and the automatic check before a teacher cache: starts the idle container, tries every way of moving a 1 KB file in and out, removes the container.
    -> {"ok", "route_in", "route_out", "bind", "tried", "seconds", "text"}. The container is ALWAYS removed."""
    e = container_config(job)
    t0 = time.monotonic()
    tmp = tempfile.TemporaryDirectory(prefix="ndforge-vllm-probe-")
    box = Container(e, "", work_dir=Path(tmp.name))
    out: dict = {"ok": False, "route_in": None, "route_out": None, "bind": None, "tried": box.tried, "text": ""}
    try:
        box.start(float(e.get("start_timeout") or vllmengine.DEFAULT_START_TIMEOUT) + 300)
        out.update(ok=True, route_in=box.route_in, route_out=box.route_out, bind=box.bound)
        out["text"] = (f"Files go in and out through a mounted folder (`{box.bound}`)." if box.bound
                       else f"Files go in by {box.route_in} and out by {box.route_out}. A mounted folder was refused here: " + box._summary())
    except (VllmCacheError, vllmengine.EngineError) as exc:
        out["text"] = str(exc)
    finally:
        box.stop()
        tmp.cleanup()
    out["seconds"] = round(time.monotonic() - t0, 1)
    return out


# ----------------------------------------------------------------------------------------------------------------------------- the chunk

def script_args(job: dict, sel: dict, model_arg: str, extra_llm: dict, first: int, last: int, max_seq: int, k: int, max_seqs: int | None = None) -> list:
    """The command line of vllm_cache.py inside the container."""
    e = job.get("engine") or {}
    a = ["python3", f"{WORK}/vllm_cache.py", "--model", model_arg, "--data", f"{WORK}/data", "--out", f"{WORK}/out", "--meta", f"{WORK}/meta.json", "--k", str(k), "--max-seq", str(max_seq),
         "--first-row", str(first), "--last-row", str(last), "--part-tokens", str(int(job.get("part_tokens", 262144))), "--stop-file", f"{WORK}/STOP",
         "--gpu-util", str(float(e.get("gpu_memory_utilization") or vllmengine.DEFAULT_GPU_UTIL)), "--batch-tokens", str(int(e.get("batch_tokens") or 32768)),
         "--max-num-seqs", str(int(e.get("max_num_seqs") or max_seqs or 32))]
    if sel["route"] == "bnb":
        tail = list((e.get("teacher") or {}).get("bnb_args") or BNB_ARGS)
        quant = tail[tail.index("--quantization") + 1] if "--quantization" in tail and tail.index("--quantization") + 1 < len(tail) else "bitsandbytes"
        a += ["--quantization", quant]
    kw = {**extra_llm, **(e.get("llm_kwargs") or {})}
    a += ["--kv-cache-dtype", teacher_kv(job)]                          # explicit for "auto" too: the script must not leave it to a default
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
            self.box.get("out/index.json", idx_path, 120, "Copying the index out", quiet=True)
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
        except (VllmCacheError, OSError, ValueError):
            return 0                                             # nothing written yet (or a copy raced the writer): the next sync tries again
        added = 0
        for part in idx.get("parts", []):
            name = part["file"]
            if name in self.done_parts:
                continue
            dst = self.tmp / name
            self.box.get(f"out/{name}", dst, 600, f"Copying {name} out")
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


def _log_gpu_before_start(g: dict, total, used) -> None:
    """The arithmetic of the engine's memory, in the job log, before any engine starts."""
    if total and used is not None:
        free = (float(total) - float(used)) / 1024
        protocol.log(f"GPU free before the engine start: {free:.1f} GB of {float(total) / 1024:.1f} GB; the engine takes {g['util'] * float(total) / 1024:.1f} GB ({g['util']:g} of the card). {g['text']}")
    else:
        protocol.log(f"GPU memory before the engine start is not known (nvidia-smi did not answer). {g['text']}")


def _check_bnb_fit(origin_dir, g: dict, total, used, max_seq: int = 2048):
    """The in-flight 4-bit route needs its weights + overhead + a minimum cache within the engine's budget: refused early with the arithmetic when it cannot (never after the rows were staged)."""
    from . import vllmquant
    sizes = vllmquant.route_weights_gb(str(origin_dir))
    if not sizes or not total or used is None:
        return None
    free = round((float(total) - float(used)) / 1024, 1)
    bud = vllmquant._budgets(float(total) / 1024, g["util"], free)
    pl = vllmquant._place(sizes["nf4"], bud)
    protocol.log(f"4-bit fit of the original ({sizes['bf16']} GB in 16 bits): {pl['arith']}.")
    if not pl["ok"]:
        raise VllmCacheError(f"The original does not fit in 4 bits next to the engine's needs: {pl['arith']}. Free more memory on the card (close other programs) or use the standard engine. {REMEDY_OFF}")
    n, text = vllmquant.seq_cap(sizes["nf4"], float(total) / 1024, g["util"], free, max_seq)
    protocol.log(text)
    return n


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
    _log_gpu_before_start(g, total, used)
    seq_cap = None
    if sel["route"] == "bnb":
        seq_cap = _check_bnb_fit(origin_dir, g, total, used, max_seq)
    tmp_ctx = tempfile.TemporaryDirectory(prefix="ndforge-vllm-")
    tmp = Path(tmp_ctx.name)
    work = tmp / "work"                                          # the host folder that is /work in the container when the bind mount works
    box = Container(e, sel["model"], work_dir=work)
    t0 = time.monotonic()
    if teacher_kv(job) == "fp8":
        protocol.log(KV_FP8_WARNING)
    result: dict = {"engine": f"vllm-{e['version']}", "teacher_quant": sel["quant"], "route": sel["route"], "reason": sel["reason"], "stopped": False, "notes": [],
                    "load_phases": [], "gpu": g, "startup_seconds": None, "rows": 0, "tokens": 0, "tokens_per_s": 0.0}
    if sel["route"] == "bnb" and vllmengine.under_mnt(str(origin_dir)) and box.kind in ("wsl", "docker_wsl"):
        result["notes"].append("The original's folder is on a Windows drive: the engine reads it through a slow share (9p); copying it into the WSL filesystem makes the start much faster.")
    proc = None
    try:
        protocol.progress("cache", 0.0, "Starting the vLLM container (the first time an image download and a model download can take long)")
        box.start(float(e.get("start_timeout") or vllmengine.DEFAULT_START_TIMEOUT) + 2100)
        if sel["route"] == "bnb":
            pip = str((tcfg or {}).get("bnb_pip") or BNB_PIP)
            if pip.lower() != "none":
                protocol.log(f"Installing {pip} into the container (the in-flight 4-bit route; UNVERIFIED: how vLLM 0.34 spells it is not known).")
                t_pip = time.monotonic()
                box.run(box.exec_command(["pip", "install", pip]), 1800, "Installing the bitsandbytes plugin")
                protocol.log(f"Installing {pip} took {time.monotonic() - t_pip:.0f} s (it repeats at every start: the container is new each time).")
        (tmp / "meta.json").write_text(json.dumps({**meta, "engine": result["engine"], "teacher_quant": sel["quant"]}), encoding="utf-8")
        box.run(box.exec_command(["mkdir", "-p", f"{WORK}/data", f"{WORK}/out"]), 60, "Making the work folders")
        here = Path(__file__).resolve().parent
        for f in SCRIPT_FILES:
            box.put(here / f, f, 120, f"Copying {f} into the container")
        box.put(Path(rows_dir) / "rows.npz", "data/rows.npz", 1800, "Copying the rows into the container")
        box.put(tmp / "meta.json", "meta.json", 120, "Copying the cache description into the container")
        staged = time.monotonic()
        result["stage_seconds"] = round(staged - t0, 1)
        cmd = box.exec_command(script_args(job, sel, box.model_arg, box.extra_llm, first, last, max_seq, k, seq_cap))
        box.commands.append(cmd)
        protocol.progress("cache", 0.0, "Loading the model into vLLM")
        try:
            proc = vllmengine._popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, bufsize=1, **vllmengine._win_kw())
        except OSError as exc:
            raise VllmCacheError(f"The script could not be started in the container: {exc}") from exc
        q: queue.Queue = queue.Queue()
        threading.Thread(target=_reader, args=(proc, q), daemon=True).start()
        syncer = Syncer(box, writer, tmp, first)
        seen_lines: deque = deque(maxlen=800)                   # every line the script printed: the cause of a failure is among them
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
                seen_lines.append(line.rstrip())
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
            raise VllmCacheError(f"{failure} {box.evidence(seen_lines)}")
        if code not in (0, None):
            raise VllmCacheError(f"The vLLM script ended with exit code {code}. {box.evidence(seen_lines)}")
        result["stopped"] = bool(result["stopped"] or (stop_sent and writer.done_rows < last))
        seen = vllmengine.kv_cache_from_log("\n".join(seen_lines))
        result["kv_cache_requested"] = teacher_kv(job)
        result["kv_cache_seen"] = seen
        protocol.log(f"KV cache type: asked for {teacher_kv(job)}; the engine's log says {seen['seen'] or 'nothing about it'}." + ("" if not seen["evidence"] else " (" + seen["evidence"][0][:160] + ")"))
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
