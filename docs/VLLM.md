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

## WSLc (WSL 3.x containers)

What we know: only release-note titles of WSL 3.0.x (3.0.1 is "generally available", 29 September 2026): a built-in Linux container CLI/SDK with its own `wslc` command; titles mention
`wslc run`, `wslc create`, `--cap-add/--cap-drop`, `wslc network create --ipv6`, "WSLC GPU containers", "Fix WSLC GPU hook execution and library discovery", "Expose VGEM DRM nodes to
WSLC GPU containers". We could not read the `wslc` reference. **We do not know** its option names, the GPU option, registry support, bind-mount syntax, port forwarding, or whether it
accepts an unmodified Docker Hub image such as `vllm/vllm-openai`.

So the command lines are TEMPLATES (a guess modelled on docker; `{gpu_flag}` is empty because we do not know the GPU option):

    run     wslc run -d --name {name} -p 127.0.0.1:{port}:8000 -v {model_dir}:/models {gpu_flag} {image} {args}
    stop    wslc stop {name}
    remove  wslc rm -f {name}
    logs    wslc logs {name}

Placeholders: `{name} {port} {image} {model_dir} {args} {gpu_flag}`. Edit them in Settings (engine "wslc" templates) after reading `wslc run --help`. The self-test saves the verbatim
output of `wslc --version`, `--help`, `run --help`, `pull --help`, `volume --help`, `network --help` into `vllm_selftest.json/.md` ("wslc help (for adapting the templates)").
Please paste `wslc run --help` into an issue so the real defaults can be set. The report says "wslc syntax unverified".
