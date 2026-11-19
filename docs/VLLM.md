# vLLM engine (EXPERIMENTAL, off by default)

> **Built from a research report, not yet run on this PC. Run the "vLLM self-test" first.** Every vLLM flag, image tag, speed and memory figure here comes from that
> report (its claims about vLLM 0.34.0 were not verified by us). Speeds are estimates, not measurements. Nothing here is installed for you: Docker, WSL and the NVIDIA driver
> are yours to install (guided text in the app). Nothing changes unless you switch it on.

## What it is for

| Use | Verdict of the report | Status here |
|---|---|---|
| The thinking check (many answers at once) | go after the 1-hour experiment (generation about 5-9x faster than the standard engine, whole check about 1.5-3x, start-up 2-5 minutes) | backend "vLLM (experimental)" |
| Checks of a model with a LoRA on top, every round | maybe: LoRA output on the linear-attention layers is reported garbled in some versions | setting, default OFF; the standard engine stays the exact path |
| Thinking answers written by the original model (`trace_gen`) | strongest case (5-10x) | job + card |
| Teacher cache | maybe, effort L | (last step, if built) |
| Everyday engine | no | not offered |

## Ways to run the engine (`engine.kind`)

1. **docker**: Windows `docker.exe` (Docker Desktop). `docker run --rm --gpus all --ipc=host -p 127.0.0.1:<port>:8000 --name ndforge-vllm-<port> vllm/vllm-openai:v0.34.0 ...`.
2. **docker_wsl**: a Docker engine installed INSIDE WSL; every docker command runs as `wsl.exe [-d <distro>] -- docker ...`; paths become Linux paths (`C:\x` -> `/mnt/c/x`).
3. **wslc**: WSLc, the container command of WSL 3.x (see below). Syntax unverified, templates you can edit.
4. **wsl**: a Python environment with vLLM inside WSL, started as `wsl.exe -d <distro> -- ~/.ndforge/vllm/bin/vllm serve ...`.
5. **url**: a server that already runs (nothing is started or stopped).
6. **auto**: tries 1, 2, 3, 4 in that order (`docker version`, `wsl.exe -- docker version`, `wslc --version` only if the first two are missing, a vLLM install in WSL). The app shows which was found and you can override it.

WSL 3.x behaviour is untested by us; the self-test tells you what works on your PC. WSL versions are compared as numbers: 2.7.3 and newer (3.x included) are fine; an unreadable
version is reported as "unknown, continuing".

The engine is always stopped explicitly (`docker stop`, `pkill` in the distro, the configured `wslc` stop template), in a `finally`, on Stop and on failure: killing `docker.exe`
or `wsl.exe` on Windows does not end the container. The app only touches containers and volumes named `ndforge-vllm-*`.

Models: prefer a NAMED volume (`ndforge-vllm-hf`, the default for a Hub id) or a folder inside the WSL filesystem. A folder on a Windows drive (`/mnt/c/...`) goes through a slow
9p share; the self-test report warns when your model folder is there.

## In the app (step 3)

- **Training setup, "vLLM engine (experimental)"**: the switch (off), the way to run it (auto / docker / docker in WSL / WSLc / WSL / url), the pinned image (`vllm/vllm-openai:v0.34.0`), the WSL
  distribution and folder, the Docker volume for models (`ndforge-vllm-*` only), an optional model folder, the pre-quantized original (`cyankiwi/Qwen3.8-27B-AWQ-INT4`), the round-check switch (off) and
  the WSLc templates. "Look for the engine" asks `docker version`, `wsl.exe -- docker version`, `wslc --version`, a vLLM install in WSL, `wsl --version` and `nvidia-smi`; it starts and installs
  nothing and prints plain install text when nothing is found (Docker Desktop or Docker in WSL or WSLc or a vLLM environment; Windows driver 580 or newer; about 60 GB of disk).
- **Thinking check card**: the engine choice and the separate vLLM table (see docs/THINKING_CHECK.md).
- **Before a vLLM check** the checklist asks: an engine is there, the card is free (nothing else above 2 GB), driver and WSL not too old (versions are compared as numbers; unreadable = "unknown, continuing").

## Choosing the weights (step 4)

`vllmquant.pick` decides what the engine loads, from the report's estimates (usable memory = card x 0.88 minus about 6 GB for the cache): the 16-bit weights when they fit; for the 27 B
ORIGINAL the configured pre-quantized checkpoint (default `cyankiwi/Qwen3.8-27B-AWQ-INT4`: its answers differ a little from the original's, and the result says which weights ran);
for a cut that does not fit `--quantization fp8` at load (no tools needed); a folder that is already quantized is loaded as stored; if even 8 bits do not fit, a plain message says so.
`--language-model-only` is always passed. Whether it really fits is for your card: the self-test shows it.

## Rounds of a heal with vLLM (step 5, off by default)

"Check the rounds of a heal (model + LoRA) with vLLM" (Training setup, only with the engine on) serves the base model with `--enable-lora --max-lora-rank <rank> --lora-modules round=<folder>`.
**Untested and risky**: the report found a forum report of garbled output for LoRA on the linear-attention layers, and our 4-bit double-quantized base cannot be reproduced in vLLM (an 8-bit base
plus the adapter is "closer to what ships", not "as trained"). Round numbers from vLLM are judged against the original checked the standard way, so keep this OFF until the agreement check of
the self-test (LoRA folder field) shows the same numbers and sensible answers on 2-3 rounds. DoRA and LoftQ adapters are not served by vLLM: a round with one falls back to the standard
engine; so does any vLLM problem; the result carries a plain `fallback` note and the log says it. A round check never fails because of the experiment.

## The vLLM self-test (Training setup, vLLM engine card)

The research report's 1-hour experiment, run for you as far as the app can. You choose a model (a folder or a Hugging Face id; the default is the configured pre-quantized original, about 21 GB:
the card asks you to confirm the download) and press **Run the vLLM self-test**. It:

1. checks the NVIDIA driver (580 or newer), the WSL version, that nothing else uses more than 2 GB of the card, that the model folder is not on a slow `/mnt/` share, and that a throw-away
   container can see the card (`nvidia-smi` in `nvidia/cuda:13.0.0-base-ubuntu24.04`; the way it is run depends on the engine kind). When `wslc` is on the PATH its help is captured verbatim.
2. starts the engine, waits for `/health`, records the start-up time, the attention / GDN kernel and cache log lines, and checks that `/health` answers from the Windows side (if not: the WSL
   address workaround `wsl.exe hostname -I` as a hint; mirrored networking may matter, unverified).
3. asks the Quick thinking suite with 40 requests at once: wall time, completion tokens per second, closed / accuracy / loop rates, five sample answers to look at.
4. compares with your saved standard-engine (and llama.cpp) numbers (note: those may be of the original, not of the quantized model you test with). With a LoRA folder it serves the adapter and
   compares with a saved check of the same adapter on the standard engine ("agreement check").
5. stops the engine (always: also after an error and on Stop), writes `vllm_selftest.json` and `vllm_selftest.md` (also in the problem-report bundle) with a plain verdict.

Success criteria (from the report): engine ready in under 5 minutes; at least 300 tokens per second; numbers within the noise of +-2 questions. **Pass** = all, **Partly** = some, **Fail** =
none or the engine could not run. The report says which; nothing in it is a promise for other models or other PCs.

## WSLc (WSL 3.x containers)

What we know, from the owner's `wslc --help` and `wslc run --help` (3.0.x): the global option `--session`; top-level commands `container image network registry settings system volume attach build cp create exec events export images import info inspect kill list load login logout logs pull push remove restart rmi run save start stats stop tag version`. `wslc run [options] <image> [<command>] [<arguments>...]` has `-d/--detach`, `--gpus` ("Add GPU devices to the container ('all' to pass all GPUs)"), `-v/--volume`, `--mount`, `-p/--publish`, `--name`, `--rm`, `-e/--env`, `--shm-size`, `--pull (always|missing|never)`, `--entrypoint`, `--network`, and more. **There is no `--ipc` option** (docker's `--ipc=host` does not exist here): the shared memory vLLM needs is `--shm-size` (default 16g).

What we have NOT seen: the options of `wslc stop`, `wslc remove`, `wslc volume ...`. **Every command is a template you can edit** in Settings (the combination is untested):

    run      wslc run -d --rm --name {name} {gpu_flag} --shm-size {shm} -p {publish} -v {model_dir}:/models {env} {image} {args}
    stop     wslc stop {name}
    remove   wslc remove {name}        (in a finally; "no such container" is fine because of --rm)
    logs     wslc logs {name}          list: wslc list      pull: wslc pull {image}
    gputest  wslc run --rm {gpu_flag} {test_image} nvidia-smi

Placeholders: `{name} {port} {image} {model_dir} {args} {gpu_flag} {shm} {publish} {env} {test_image}`; defaults `gpu_flag` = `--gpus all`, `shm` = `16g`, `publish` = `{port}:8000`
(`publish_loopback` = `127.0.0.1:{port}:8000`), `test_image` = `nvidia/cuda:13.0.0-base-ubuntu24.04`.

**Security note.** Whether `-p 127.0.0.1:<port>:8000` is accepted (loopback only) is unknown, so the default publishes `<port>:8000`, which may be reachable from your network. For that
reason the engine ALWAYS starts vLLM with a fresh random `--api-key` per run (every engine kind) and the worker sends `Authorization: Bearer <key>`; the key is never logged (it is on
the scrub list and removed from recorded command lines). The self-test tries the loopback form once and records which works.

Unknowns the self-test is meant to answer: whether Docker Hub images (`vllm/vllm-openai`) pull unmodified (`--pull missing` suggests registries work; `wslc registry` and `wslc login`
exist), whether `-v C:\path:/models` bind mounts work and how slow they are (`wslc volume` would be the fast way, its syntax is unseen), whether `--gpus all` exposes CUDA to
PyTorch and vLLM inside WSLc (release notes mention "WSLC GPU containers" and "enable inbox gpu libs"), and whether the images' CUDA 13 userspace works with the WSLc GPU libraries.

The self-test saves the verbatim output of `wslc --version`, `info`, `--help`, `run --help`, `system --help`, `container --help`, `stop --help`, `remove --help`, `list --help`,
`logs --help`, `pull --help`, `volume --help`, `volume create --help`, `network --help` and `registry --help` into `vllm_selftest.json/.md` ("wslc help (for adapting the templates)"),
runs the GPU test image first, and says "wslc syntax unverified". Please paste the report into an issue so the real defaults can be set.
