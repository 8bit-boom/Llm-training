"""The vLLM engine launcher (EXPERIMENTAL, off by default): builds the command lines that start a vLLM server and stops it again, explicitly.

    BUILT FROM A RESEARCH REPORT, NOT YET RUN ON THE OWNER'S PC. Every vLLM flag below (the names, the values, the image tag, the `/health` and `/version` routes, the log
    lines looked for) is "from the research report, untested". They are all in this one file, in the constants at the top and in `serve_args`, so one place changes when
    the self-test (job `think_check` with backend "vllm", or the Settings card "vLLM self-test") shows something different.

Five ways to have a server (`engine["kind"]`; "auto" tries docker, docker_wsl, wslc, wsl in that order and takes the first that answers):
    docker      `docker run --rm --gpus all --ipc=host -p 127.0.0.1:<port>:8000 --name ndforge-vllm-<port> <image> <vllm flags>` (Windows docker.exe / Docker Desktop)
    docker_wsl  the same `docker ...` commands, each run as `wsl.exe [-d <distro>] -- docker ...` (a Docker engine installed INSIDE WSL; Linux paths, `C:` drive paths -> `/mnt/c/...`)
    wsl         `wsl.exe -d <distro> -- <venv>/bin/vllm serve <model> <vllm flags>` (Windows paths become /mnt/<drive>/...)
    wslc        WSLc, the container command of WSL 3.x (`wslc run ...`): options from the owner's `wslc --help`, the combination untested, the command lines are TEMPLATES the owner can edit (WSLC_TEMPLATES)
    url         an already running server (`engine["url"]`): nothing is started and nothing is stopped.
What WSL 3.x changes is untested by us (the report only covered WSL 2.7.x): versions are compared as numbers, 2.7.3 and newer (3.x too) are fine, an unreadable version is "unknown".

Windows: killing `docker.exe` or `wsl.exe` does NOT end the container / the Linux process tree. So `Engine.stop()` ALWAYS runs the explicit stop (`docker stop`, `pkill` in
the distro) and the callers put it in a `finally`.

Pure functions (CI-tested with a mocked subprocess): `wsl_path`, `serve_args`, `docker_command`, `wsl_command`, `stop_commands`, `engine_identity`, `log_highlights`.
The real launch needs the owner's PC.
"""
from __future__ import annotations

import json
import re
import shlex
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from . import protocol

DEFAULT_IMAGE = "vllm/vllm-openai:v0.34.0"              # from the research report: pinned, bump deliberately
DEFAULT_VERSION = "0.34.0"
DEFAULT_WSL_VENV = "~/.ndforge/vllm"                    # from the research report: `uv venv ~/.ndforge/vllm --python 3.12 && uv pip install vllm==0.34.0 --torch-backend=cu130`
DEFAULT_HF_VOLUME = "ndforge-vllm-hf"                   # the only Docker volume names this program touches start with "ndforge-vllm-"
VOLUME_PREFIX = "ndforge-vllm-"
CONTAINER_PREFIX = "ndforge-vllm-"
SERVED_NAME = "model"
ADAPTER_NAME = "round"
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_MAX_SEQS = 40
DEFAULT_GPU_UTIL = 0.88
DEFAULT_LORA_RANK = 64
DEFAULT_START_TIMEOUT = 1500                            # seconds: image pull + model load + graph capture; the report expects 2-5 minutes
STOP_TIMEOUT = 60
KINDS = ("docker", "docker_wsl", "wslc", "wsl", "url")
# WSLc (WSL 3.x containers). The options below are "from the owner's `wslc --help` / `wslc run --help` (3.0.x)"; the COMBINATION is UNTESTED, and the syntax of `wslc stop`, `wslc remove`,
# `wslc volume ...` is not yet seen. So every command line is a TEMPLATE the owner can edit (engine["wslc"] = {"run": ..., "stop": ..., ...}). wslc has no `--ipc`: the shared memory
# vLLM needs is `--shm-size`. Whether `-p 127.0.0.1:<port>:8000` is accepted (loopback only) is UNKNOWN: the default publishes `<port>:8000`, which may be reachable from the network,
# which is one reason the engine always starts vLLM with a per-run API key (`--api-key`).
WSLC_TEMPLATES = {"run": "wslc run -d --rm --name {name} {gpu_flag} --shm-size {shm} -p {publish} -v {model_dir}:/models {env} {image} {args}",
                  "stop": "wslc stop {name}", "remove": "wslc remove {name}", "logs": "wslc logs {name}", "list": "wslc list", "pull": "wslc pull {image}",
                  "gputest": "wslc run --rm {gpu_flag} {test_image} nvidia-smi",
                  "gpu_flag": "--gpus all", "shm": "16g", "publish": "{port}:8000", "publish_loopback": "127.0.0.1:{port}:8000", "env": [], "test_image": "nvidia/cuda:13.0.0-base-ubuntu24.04"}
WSLC_HELP_COMMANDS = (["wslc", "--version"], ["wslc", "info"], ["wslc", "--help"], ["wslc", "run", "--help"], ["wslc", "system", "--help"], ["wslc", "container", "--help"],
                      ["wslc", "stop", "--help"], ["wslc", "remove", "--help"], ["wslc", "list", "--help"], ["wslc", "logs", "--help"], ["wslc", "pull", "--help"],
                      ["wslc", "volume", "--help"], ["wslc", "volume", "create", "--help"], ["wslc", "network", "--help"], ["wslc", "registry", "--help"])
HELP_LIMIT = 8000
KIND_NAMES = {"docker": "Docker Desktop", "docker_wsl": "Docker inside WSL", "wslc": "WSLc (WSL 3.x containers)"}
MIN_WSL = (2, 7, 3)                                     # from the research report (the CUDA-graph crash fix); newer, 3.x included, is assumed fine
# log lines worth keeping for the self-test report (from the research report: which attention kernel, which GDN prefill kernel, how much cache)
HIGHLIGHT = re.compile(r"(attention backend|attn backend|flash_attn|flashinfer|gdn|triton|kv cache|cache memory|gpu memory|model loading took|cuda graph|error|warning: .*sm_?120)", re.I)


class EngineError(Exception):
    """A plain-language problem with the vLLM engine."""


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _win_kw() -> dict:
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if sys.platform == "win32" else {}


_popen = subprocess.Popen
_run = subprocess.run
_which = shutil.which


# ------------------------------------------------------------------------------------------------------------------ pure helpers

def wsl_path(path: str) -> str:
    r"""`C:\Users\me\m` -> `/mnt/c/Users/me/m`; `\\wsl$\...` / `\\wsl.localhost\<distro>\home\x` -> `/home/x`; a path that already is POSIX stays as it is."""
    p = str(path or "")
    m = re.match(r"^\\\\wsl(?:\$|\.localhost)\\[^\\]+(\\.*)?$", p, re.I)
    if m:
        return (m.group(1) or "\\").replace("\\", "/")
    m = re.match(r"^([A-Za-z]):[\\/]*(.*)$", p)
    if m:
        rest = m.group(2).replace("\\", "/").rstrip("/")
        return f"/mnt/{m.group(1).lower()}" + (f"/{rest}" if rest else "")
    return p.replace("\\", "/") if re.match(r"^[A-Za-z]:", p) else p


def parse_wsl_version(text: str):
    """The WSL version from `wsl --version` as a tuple of ints, tolerant of localized labels: the first x.y.z (or x.y) after the first line's label; None when none is found.
    "WSL version: 2.7.3.0" -> (2, 7, 3, 0)."""
    for line in str(text or "").replace("\x00", "").splitlines():       # `wsl --version` prints UTF-16: NUL bytes between the letters when read as bytes
        m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\.(\d+))?", line)
        if m:
            return tuple(int(g) for g in m.groups() if g is not None)
    return None


def wsl_version_note(text: str) -> dict:
    """{"version": "3.0.1" | "", "ok": True | False | None, "text": plain sentence}: ok None = unknown (continue), False = older than 2.7.3 (run `wsl --update`)."""
    v = parse_wsl_version(text)
    if v is None:
        return {"version": "", "ok": None, "text": "The WSL version could not be read (unknown, continuing)."}
    shown = ".".join(str(x) for x in v)
    pad = tuple(v) + (0,) * (3 - len(v))
    if pad[:3] >= MIN_WSL:
        return {"version": shown, "ok": True, "text": f"WSL {shown} (2.7.3 or newer is wanted; the behaviour of 3.x is untested by us, the self-test tells what works)."}
    return {"version": shown, "ok": False, "text": f"WSL {shown} is older than 2.7.3; run `wsl --update` (older versions can crash the whole WSL machine when the model loads)."}


def under_mnt(path: str) -> bool:
    """True when the path, as WSL sees it, is on a Windows drive (/mnt/c/...: a slow 9p share; a named volume or a folder in the WSL home is much faster)."""
    return wsl_path(path).startswith("/mnt/") if is_local_path(path) else False


def is_local_path(model: str) -> bool:
    """A folder path (Windows or POSIX), as opposed to a Hub id like `org/name`."""
    m = str(model or "")
    return bool(re.match(r"^[A-Za-z]:[\\/]", m) or m.startswith(("/", "\\\\", "~", "./", ".\\", "../")))


def image_version(image: str) -> str:
    """`vllm/vllm-openai:v0.34.0` -> `0.34.0`; no or a floating tag -> the tag or `unknown`."""
    tag = str(image or "").rsplit(":", 1)[-1] if ":" in str(image or "").rsplit("/", 1)[-1] else ""
    m = re.match(r"^v?(\d+\.\d+(?:\.\d+)?)", tag)
    return m.group(1) if m else (tag or "unknown")


def render_template(template: str, values: dict) -> list:
    """A command line from a template, as an argument LIST (no shell: a path with spaces stays one argument). The template is split into words first; `{name}`-style
    placeholders are replaced inside a word, a word that is only `{args}` or `{gpu_flag}` becomes zero or more words (EMPTY drops it). An unknown placeholder is a plain error."""
    out: list = []
    for word in shlex.split(template, posix=True):
        names = re.findall(r"\{(\w+)\}", word)
        for n in names:
            if n not in values:
                raise EngineError(f"The command template uses {{{n}}}, which is not known here. Known: {', '.join('{' + k + '}' for k in sorted(values))}.")
        if len(names) == 1 and word == "{" + names[0] + "}" and isinstance(values[names[0]], (list, tuple)):
            out += [str(x) for x in values[names[0]]]
        elif len(names) == 1 and word == "{" + names[0] + "}" and names[0] == "gpu_flag":
            out += shlex.split(str(values["gpu_flag"] or ""), posix=True)
        else:
            out.append(re.sub(r"\{(\w+)\}", lambda m: str(values[m.group(1)]), word))
    return out


def redact(cmd: list, secret: str) -> list:
    """The command line without the API key (it is kept in `Engine.commands` and shown in reports)."""
    return [str(x).replace(secret, "[key]") if secret else str(x) for x in cmd]


def wslc_templates(cfg: dict) -> dict:
    return {**WSLC_TEMPLATES, **{k: v for k, v in (cfg.get("wslc") or {}).items() if k in WSLC_TEMPLATES}}


def wslc_help(limit: int = HELP_LIMIT) -> dict:
    """Runs and CAPTURES (verbatim, size-limited) the help of `wslc` for the self-test report ("wslc help (for adapting the templates)"); {} when `wslc` is not on the PATH.
    {"<command line>": {"returncode": n, "output": "..."}}."""
    if not _which("wslc"):
        return {}
    out = {}
    for cmd in WSLC_HELP_COMMANDS:
        try:
            r = _run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=30, **_win_kw())
            raw = (getattr(r, "stdout", b"") or b"") + (getattr(r, "stderr", b"") or b"")
            text = raw.decode("utf-16-le", "replace") if raw[:2] in (b"\xff\xfe",) or (len(raw) > 4 and raw[1:2] == b"\x00") else raw.decode("utf-8", "replace")
            out[" ".join(cmd)] = {"returncode": getattr(r, "returncode", None), "output": protocol.scrub(text.replace("\x00", ""))[:limit]}
        except (OSError, subprocess.SubprocessError) as exc:
            out[" ".join(cmd)] = {"returncode": None, "output": f"could not run: {exc}"}
    return out


def detect_kind(e: dict) -> str:
    """"auto": `docker version` on Windows, else `wsl.exe [-d distro] -- docker version`, else `wslc --version` (only when the others are missing), else a vLLM venv in WSL;
    a plain error when none answers."""
    wsl = ["wsl.exe"] + (["-d", e["distro"]] if e.get("distro") else []) + ["--"]
    venv = str(e.get("venv") or DEFAULT_WSL_VENV).rstrip("/")
    for kind, cmd in (("docker", ["docker", "version"]), ("docker_wsl", docker_prefix({"kind": "docker_wsl", "distro": e.get("distro") or ""}) + ["version"]),
                      ("wslc", ["wslc", "--version"]), ("wsl", wsl + ["test", "-x", f"{venv}/bin/vllm"])):
        try:
            r = _run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, timeout=30, **_win_kw())
        except (OSError, subprocess.SubprocessError):
            continue
        if getattr(r, "returncode", 1) == 0:
            return kind
    raise EngineError("No engine answered (`docker version`, `wsl.exe -- docker version`, `wslc --version` and a vLLM install in WSL were all tried). Install Docker Desktop, or Docker inside WSL, or choose the url way in the settings.")


def docker_prefix(cfg: dict) -> list:
    """The words in front of every docker command: `docker`, or `wsl.exe [-d distro] -- docker` for a Docker engine inside WSL."""
    if cfg["kind"] == "docker_wsl":
        return ["wsl.exe"] + (["-d", cfg["distro"]] if cfg.get("distro") else []) + ["--", "docker"]
    return ["docker"]


def engine_config(job: dict) -> dict:
    """The job's `engine` object with its defaults: {kind, image, distro, venv, url, model_volume, hf_volume, version, quant, lora, port, ...}."""
    e = dict(job.get("engine") or {})
    kind = str(e.get("kind") or "docker").lower()
    if kind == "auto":
        kind = detect_kind(e)
    if kind not in KINDS:
        raise EngineError(f"Unknown vLLM engine kind {kind!r}. Use docker, docker_wsl, wslc, wsl, url or auto.")
    e["kind"] = kind
    e["image"] = str(e.get("image") or DEFAULT_IMAGE)
    e["venv"] = str(e.get("venv") or DEFAULT_WSL_VENV)
    e["hf_volume"] = str(e.get("hf_volume") or DEFAULT_HF_VOLUME)
    if not e["hf_volume"].startswith(VOLUME_PREFIX) and e["hf_volume"]:
        raise EngineError(f"The Docker volume must be named {VOLUME_PREFIX}<something>: this program only touches those.")
    if kind == "wslc":
        wslc_templates(e)
        for k in (e.get("wslc") or {}):
            if k not in WSLC_TEMPLATES:
                raise EngineError(f"Unknown wslc setting {k!r}. Known: {', '.join(WSLC_TEMPLATES)}.")
    if kind in ("wsl", "docker_wsl") and not e.get("distro"):
        e["distro"] = ""                                    # the default distro
    if kind == "url" and not e.get("url"):
        raise EngineError("The vLLM engine is set to 'url' but no address is given.")
    e["version"] = str(e.get("version") or (image_version(e["image"]) if kind in ("docker", "docker_wsl") else DEFAULT_VERSION if kind == "wsl" else "unknown"))
    return e


def serve_args(model: str, *, quant: str = "none", adapter: str | None = None, max_len: int = DEFAULT_MAX_MODEL_LEN, max_seqs: int = DEFAULT_MAX_SEQS,
               gpu_util: float = DEFAULT_GPU_UTIL, lora_rank: int = DEFAULT_LORA_RANK, extra: list | None = None, api_key: str | None = None) -> list:
    """The vLLM flags (all from the research report, untested): no reasoning parser (the template ends the prompt with `<think>\\n`), model defaults off
    (`--generation-config vllm`), vision tower off, the served name `model`. `quant`: "none" | "fp8" (`--quantization fp8`) | anything else is a pre-quantized checkpoint (nothing added:
    vLLM reads its own quantization config). `adapter`: the path (as the server sees it) of a LoRA folder, served as `round`."""
    args = ["--model", model, "--served-model-name", SERVED_NAME, "--language-model-only", "--generation-config", "vllm", "--max-model-len", str(int(max_len)),
            "--max-num-seqs", str(int(max_seqs)), "--gpu-memory-utilization", f"{float(gpu_util):g}"]
    if api_key:
        args += ["--api-key", api_key]                      # the port may not be loopback-only (WSLc): every request must carry `Authorization: Bearer <key>`
    if quant == "fp8":
        args += ["--quantization", "fp8"]
    if adapter:
        args += ["--enable-lora", "--max-lora-rank", str(int(lora_rank)), "--lora-modules", f"{ADAPTER_NAME}={adapter}"]
    return args + [str(x) for x in (extra or [])]


def container_name(port: int) -> str:
    return f"{CONTAINER_PREFIX}{int(port)}"


def docker_command(cfg: dict, port: int, args: list, *, mounts: list | None = None) -> list:
    """`docker run` for the pinned image. `mounts`: [(host path or named volume, container path, "ro"|"rw")]. The Hub cache is a NAMED volume (a Windows bind mount is slow)."""
    cmd = docker_prefix(cfg) + ["run", "--rm", "--gpus", "all", "--ipc=host", "--name", container_name(port), "-p", f"127.0.0.1:{int(port)}:8000"]
    for src, dst, mode in mounts or []:
        cmd += ["-v", f"{src}:{dst}:{mode}"]
    return cmd + [cfg["image"]] + list(args)


def wsl_command(cfg: dict, port: int, args: list, *, exe: str = "wsl.exe") -> list:
    """`wsl.exe -d <distro> -- <venv>/bin/vllm serve <flags> --host 127.0.0.1 --port <port>`; vLLM's `serve` takes the model as the first argument."""
    cmd = [exe] + (["-d", cfg["distro"]] if cfg.get("distro") else []) + ["--", f"{cfg['venv'].rstrip('/')}/bin/vllm", "serve"]
    rest = list(args)
    if rest[:1] == ["--model"] and len(rest) >= 2:          # `vllm serve <model> ...`
        cmd.append(rest[1])
        rest = rest[2:]
    return cmd + rest + ["--host", "127.0.0.1", "--port", str(int(port))]


def wslc_values(cfg: dict, port: int, model_dir: str, args: list, loopback: bool = False) -> dict:
    t = wslc_templates(cfg)
    env: list = []
    for item in t["env"] if isinstance(t["env"], list) else []:      # ["-e", "K=V"] words or "K=V" strings
        env.append(str(item))
    publish = str(t["publish_loopback"] if loopback else t["publish"]).replace("{port}", str(int(port)))
    return {"name": container_name(port), "port": int(port), "image": cfg["image"], "model_dir": model_dir, "args": list(args), "gpu_flag": t["gpu_flag"], "shm": t["shm"],
            "publish": publish, "env": env, "test_image": t["test_image"]}


def stop_commands(cfg: dict, port: int, *, wsl: str = "wsl.exe") -> list:
    """The explicit stop, as a list of command lines (each ignored when it fails). url: none."""
    if cfg["kind"] in ("docker", "docker_wsl"):
        d = docker_prefix(cfg)
        return [d + ["stop", "-t", "20", container_name(port)], d + ["rm", "-f", container_name(port)]]
    if cfg["kind"] == "wslc":
        t = wslc_templates(cfg)
        v = wslc_values(cfg, port, "", [])
        return [render_template(t["stop"], v), render_template(t["remove"], v)]
    if cfg["kind"] == "wsl":
        return [[wsl] + (["-d", cfg["distro"]] if cfg.get("distro") else []) + ["--", "pkill", "-f", f"vllm serve.*--port {int(port)}"]]
    return []


def log_highlights(text: str, limit: int = 40) -> list:
    """The lines of the server log that name the attention / GDN kernel, the cache size and problems (for the self-test report)."""
    out, seen = [], set()
    for ln in (text or "").splitlines():
        s = ln.strip()
        if s and HIGHLIGHT.search(s) and s not in seen:
            seen.add(s)
            out.append(protocol.scrub(s)[:300])
            if len(out) >= limit:
                break
    return out


def engine_identity(job: dict, quant: str, lora: bool) -> dict:
    """What the result cache key adds for backend vllm: {"vllm": version, "quant": ..., "lora": bool} (additive: other backends' keys do not have it)."""
    cfg = engine_config(job)
    return {"vllm": cfg["version"], "quant": quant, "lora": bool(lora)}


# ------------------------------------------------------------------------------------------------------------------ the running engine

class Engine:
    """A vLLM server: start() (does not wait), wait_ready(), stop() (always the explicit stop). `_popen` and `_run` are the module-level hooks tests replace."""

    def __init__(self, cfg: dict, model: str, *, adapter: str | None = None, quant: str = "none", extra_args: list | None = None, max_len: int = DEFAULT_MAX_MODEL_LEN,
                 max_seqs: int = DEFAULT_MAX_SEQS, gpu_util: float = DEFAULT_GPU_UTIL):
        self.cfg, self.model, self.adapter, self.quant = cfg, model, adapter, quant
        self.extra_args, self.max_len, self.max_seqs, self.gpu_util = extra_args, max_len, max_seqs, gpu_util
        self.kind = cfg["kind"]
        self.port = 0
        self.url = str(cfg.get("url") or "").rstrip("/")
        self.proc = None
        self.log = None
        self.started_at = None
        self.ready_seconds = None
        self.stopped = False
        self.commands: list = []
        self.api_key = str(cfg.get("api_key") or "") if self.kind == "url" else secrets.token_urlsafe(24)   # a started server gets a fresh key every run
        protocol.add_secret(self.api_key)

    # --- what to run
    def build(self) -> list:
        """The start command line (docker or wsl); [] for url. Sets `self.port`."""
        if self.kind == "url":
            return []
        self.port = self.port or free_port()
        mounts: list = []
        model, adapter = self.model, self.adapter
        if self.kind in ("docker", "docker_wsl"):
            host = wsl_path if self.kind == "docker_wsl" else (lambda x: x)      # a Docker engine inside WSL wants Linux paths
            if is_local_path(model):
                mounts.append((host(model), "/models/base", "ro"))
                model = "/models/base"
            elif self.cfg.get("hf_volume"):
                mounts.append((self.cfg["hf_volume"], "/root/.cache/huggingface", "rw"))
            if adapter:
                mounts.append((host(adapter), "/models/adapter", "ro"))
                adapter = "/models/adapter"
        elif self.kind == "wsl":
            model = wsl_path(model) if is_local_path(model) else model
            adapter = wsl_path(adapter) if adapter else adapter
        if self.kind == "wslc":                                # the folder is mounted at /models; a Hub id downloads into the named volume mounted there (UNVERIFIED)
            extra = list(self.extra_args or [])
            if is_local_path(model):
                model_dir, model = model, "/models"
            else:
                model_dir = self.cfg["hf_volume"]
                extra += ["--download-dir", "/models"]
            if adapter:
                raise EngineError("A LoRA on top is not set up for the wslc way yet (use docker or the url way).")
            args = serve_args(model, quant=self.quant, max_len=self.max_len, max_seqs=self.max_seqs, gpu_util=self.gpu_util, extra=extra, api_key=self.api_key)
            return render_template(wslc_templates(self.cfg)["run"], wslc_values(self.cfg, self.port, model_dir, args))
        args = serve_args(model, quant=self.quant, adapter=adapter, max_len=self.max_len, max_seqs=self.max_seqs, gpu_util=self.gpu_util, extra=self.extra_args, api_key=self.api_key)
        return docker_command(self.cfg, self.port, args, mounts=mounts) if self.kind in ("docker", "docker_wsl") else wsl_command(self.cfg, self.port, args)

    def start(self) -> None:
        self.started_at = time.monotonic()
        if self.kind == "url":
            return
        cmd = self.build()
        self.commands.append(redact(cmd, self.api_key))
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = tempfile.TemporaryFile()
        try:
            self.proc = _popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **_win_kw())
        except OSError as exc:
            raise EngineError(f"The vLLM engine could not be started ({cmd[0]}: {exc}). Is {KIND_NAMES.get(self.kind, 'WSL')} installed and running?") from exc

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def log_text(self) -> str:
        if self.kind == "wslc" and self.proc is not None:                  # detached: ask for the container's log
            try:
                r = _run(render_template(wslc_templates(self.cfg)["logs"], wslc_values(self.cfg, self.port, "", [])), capture_output=True, stdin=subprocess.DEVNULL, timeout=30, **_win_kw())
                return (getattr(r, "stdout", b"") or b"").decode("utf-8", "replace") + (getattr(r, "stderr", b"") or b"").decode("utf-8", "replace")
            except (OSError, subprocess.SubprocessError, EngineError):
                pass
        if self.log is None:
            return ""
        try:
            self.log.seek(0)
            return self.log.read().decode("utf-8", "replace")
        except (OSError, ValueError):
            return ""

    def tail(self, n: int = 12) -> str:
        return protocol.scrub("\n".join(self.log_text().strip().splitlines()[-n:]))

    def wait_ready(self, timeout: float, should_stop=lambda: False, sleep=time.sleep) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if self.proc is not None and self.proc.poll() is not None and not (self.kind == "wslc" and self.proc.returncode == 0):     # `wslc run -d` ends at once, fine
                raise EngineError(f"The vLLM engine stopped while starting (exit code {self.proc.returncode}). Its last lines: {self.tail()}")
            if should_stop():
                return
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url + "/health", headers=self.headers()), timeout=5) as r:
                    if r.status == 200:
                        self.ready_seconds = round(time.monotonic() - (self.started_at or time.monotonic()), 1)
                        return
            except (urllib.error.URLError, OSError):
                pass                                        # not listening yet
            if time.monotonic() > deadline:
                raise EngineError(f"The vLLM engine did not become ready within {int(timeout)} seconds (an image pull or a model download can take that long the first time). Its last lines: {self.tail()}")
            sleep(0.25)

    def server_version(self) -> str:
        """The version the running server reports (`GET /version`, from the research report; "" when it does not answer)."""
        try:
            with urllib.request.urlopen(urllib.request.Request(self.url + "/version", headers=self.headers()), timeout=5) as r:
                return str(json.loads(r.read().decode("utf-8")).get("version") or "")
        except (urllib.error.URLError, OSError, ValueError, AttributeError):
            return ""

    def stop(self) -> None:
        """ALWAYS runs the explicit stop (even when the process already ended), then ends the client process; never raises."""
        self.stopped = True
        if self.kind != "url":
            for cmd in stop_commands(self.cfg, self.port):
                self.commands.append(redact(cmd, self.api_key))
                try:
                    _run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, timeout=STOP_TIMEOUT, **_win_kw())
                except (OSError, subprocess.SubprocessError):
                    pass
        proc = self.proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(10)
            except (OSError, subprocess.SubprocessError):
                pass
        if self.log is not None:
            try:
                self.log.close()
            except OSError:
                pass
