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
| Teacher cache (the original's answers) | maybe, effort L | job key `engine` of `distill_cache`, Heal card option, Auto-cut option (see "The teacher's answers through vLLM") |
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
- **Before a vLLM check** the checklist asks: an engine is there, the card has room (the engine reserves 0.88 of it; when other programs such as the desktop use some, the share is LOWERED to fit, down to 0.70, and only below that is the start refused), driver and WSL not too old (versions are compared as numbers; unreadable = "unknown, continuing").

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

## Thinking answers written by the original (step 6)

Under a made model, "Make thinking answers with the original model": the original (or a model of your choice) answers prompts WITH its thinking; kept answers go to
`models/<name>/trace_gen/answers.jsonl` in the format of a heal's "My own file" general source (`messages` with `<think>...</think>` inline). The way is chosen by the worker: the experimental vLLM
engine when it is switched on and works, else llama-server (a GGUF file and llama.cpp), else the standard engine; a way that fails falls back to the next with a plain note. For the 27 B original
the configured pre-quantized checkpoint is served (its answers differ a little from the original's).

An answer is kept when `</think>` was written, the answer after it is not empty, the token limit (default 6144) did not cut it, it is no loop (`thinkcheck.loop_detect`) and not garbled
(`garbage_detect`). Rejected prompts are remembered (`answers.jsonl.rejected.jsonl`; delete it to retry) and a second run goes on with new prompts (resume by the hash of the prompt); Stop keeps
what was written. **Speed**: the report estimates 350-550 tokens per second on vLLM (1.3-2.0 million tokens per hour) against about 0.2 million on the standard engine, i.e. 5-10 times: an
estimate from a research report, not measured here; the self-test measures your PC. llama-server with many slots was never measured on this model.

## The vLLM self-test (Training setup, vLLM engine card)

The research report's 1-hour experiment, run for you as far as the app can. You choose a model (a folder or a Hugging Face id; the default is the configured pre-quantized original, about 21 GB:
the card asks you to confirm the download) and press **Run the vLLM self-test**. It:

1. checks the NVIDIA driver (580 or newer), the WSL version, that the card has room next to what the desktop already uses (`needed = 0.88 x card`, `free = card - used`; when `free < needed + 0.5 GB` the utilization is lowered to `(free - 0.6 GB) / card`, below 0.70 the test refuses and names the amount; an unreadable nvidia-smi continues with a note; the lowered value is used for the weights-fit check and recorded in the report), that the model folder is not on a slow `/mnt/` share, and that a throw-away
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


## The teacher's answers through vLLM (experimental, off)

The heal learns from the original's top-32 next-token answers (`distill_cache`). With the option on, **vLLM makes those answers** (offline scoring: `max_tokens=1, prompt_logprobs=32, logprobs=32`, the cache's own format) instead of the Hugging Face teacher. It is a long job (hours), so the engine's start-up is paid once and the faster scoring can pay back; a heal in rounds starts the engine for every round's chunk and the card warns about it (no "cache ahead" option was built).

**Everything vLLM-specific below is from the research report and unverified on this program's side**: `flat_logprobs`, `logprobs_mode`, the shape of `prompt_logprobs`, `language_model_only` as an `LLM(...)` argument, `min_tokens`/`ignore_eos` of the long-answer speed test. They are isolated: the conversion of vLLM's answer is ONE function (`vllm_cache.convert_row`, tested with fake answers in four shapes, a plain error for an unknown shape); a keyword this vLLM does not know is dropped with a note (`OPTIONAL_*`).

### The teacher must be YOUR original (the identity rule)

A pre-quantized checkpoint (such as `cyankiwi/Qwen3.8-27B-AWQ-INT4`) is a quantization of the **base** model. Using it as the teacher for a finetune would silently teach the wrong model. So:

- a pre-quantized repo is accepted only for the original it was made from: the settings table `pairs` ({original repo id: quantized repo id}; the default table has exactly `Qwen/Qwen3.8-27B` -> `cyankiwi/Qwen3.8-27B-AWQ-INT4`), or a copy **you confirm** ("this repository is a quantized copy of my original model", stored per original in `confirmed`);
- every other original uses the **in-flight bitsandbytes (NF4) route on your own folder**: the same 4-bit numbers as the standard teacher, speed unknown. **UNVERIFIED**: the report says vLLM moved bitsandbytes into the plugin `vllm-bnb-plugin` 0.0.3; this program pip-installs it into the throw-away container and passes `quantization="bitsandbytes"` (both editable: `engine.teacher.bnb_pip`, `bnb_args`). It may not work for the Qwen3.5/3.8 architecture; the job then falls back to the standard teacher with a note (or stops, if you ticked "a heal stops when vLLM does not work");
- the route `prequant` refuses an original without a confirmed copy; there is never a silent fall back to the base AWQ.

The cache records `engine: "vllm-<version>"`, `teacher_quant: "bnb-nf4" | "awq:<repo>@<rev>"` and `teacher_origin` (the original's repo and its weights manifest). They are part of `cache_matches`, `store_extends`, the shared store id (`shared_store_id(..., engine_id)`) and the program's `rows_key`: a vLLM cache is **never** adopted into a store made by the standard teacher and the other way round (separate store ids; tests in `test_vllm_cache_job.py` and `vllm_teacher.rs`). Without the key all ids are exactly what they were. If vLLM fails after some rows, the rest is made by the standard teacher into a store of ITS OWN identity; in a per-model (unshared) cache folder the vLLM rows are then discarded.

### How it runs

`vllmcache.py` (our environment) starts an idle container (`docker run -d ... --entrypoint sleep`, or the WSLc template `run_idle`), copies `rows.npz`, `vllm_cache.py` and `cachewrite.py` **into** it (`cp_in`; no bind mount of the project), runs the script by `exec`, follows its JSON progress lines, copies each finished part out while it runs (`cp_out`) and writes it into the worker's own cache writer (a stop or crash loses little), then stops **and removes** the container in a `finally`. The model folder is mounted read-only (a Windows drive goes through a slow 9p share: the job says so); Hub ids go into the named volume. `cp_in`, `cp_out`, `exec`, `run_idle` are command templates (`engine.exec_templates`, or `engine.wslc` for WSLc). **The WSLc `cp`/`exec` syntax is a GUESS** (docker-like); the self-test now captures `wslc cp --help`, `wslc exec --help`, `wslc image --help` and one `wslc list -a --format json` sample.

The pure-numpy writer is `cachewrite.py`; a test compares its parts and indexes byte for byte with the writer as it was before the extraction.

### Is it the same teacher? "Check the vLLM teacher"

Job `vllm_teacher_check` (Heal card button): 100 real rows of your heal data at the heal's sequence length; the reference is an existing complete standard cache of the same original when there is one (no GPU needed), else the standard 4-bit teacher runs first (never both models on the card). Metrics: top-1 agreement, top-32 overlap (Jaccard), KL(reference || vLLM) over the shared support (each side renormalised on it), the leftover-mass difference, and the same by position. **Verdict thresholds are heuristics chosen here** (not measured or published): `same` = top-1 >= 0.97 and KL <= 0.02; `close` = top-1 >= 0.93 and KL <= 0.08; else `different`; more than 1 % of positions without a shared token is `different`. `different` makes the heal refuse the vLLM teacher; an unchecked teacher only warns. The result (`models/<name>/vllm_teacher_check.json`) also holds the engine's start-up phases (lines of the container's log with times) and the scoring speed, so you learn where the start-up goes.

### Measured numbers: whose they are

The estimates use numbers from **your own files** and say so; nothing is shown as measured unless it is:

- `vllm_selftest.json`: start-up seconds, the model id, its size in billions of parameters, its quantization, the GPU; the speed on **long** answers (8 requests of 2,048 tokens, `min_tokens` forced) and the Quick suite's answer lengths. Numbers are used for another model only when its size is within 2 times of the model that was measured (start-up scaled by size, generation time scaled by parameters: a first-order rule, labelled an estimate); beyond that, or in an older report that does not say which model it ran, the report's formulas are used and the plan says why. The Quick suite's wall time is used only when its answers were thinking-long (median at least 300 tokens): short answers overstate a thinking check.
- `vllm_teacher_check.json`: the scoring speed of the teacher's answers. Without it the report's 3,000 to 3,800 tokens per second is used and labelled "not measured"; the self-test's generation speed is never used for scoring.

Example, **your PC, one run each, not a promise**: a 9 B model (RTX 5090, Windows, WSLc 3.0.5.0, driver 621.24): engine ready after 510.6 s, 659.2 tokens/s at 40 requests; the default 27 B AWQ-INT4: ready after 263.5 s, 447.6 tokens/s, but in 7.7 s: about 86 tokens per answer, i.e. short answers that do not predict a thinking check. This is why the self-test now reports the answer lengths, warns when the median is below 300 tokens and measures the long-answer speed.

### Engine containers: no `--rm`, fail fast

The app no longer starts engine containers with `--rm`: a container that crashed at start (an out-of-memory error) stays, so its last 60 log lines are read **before** it is removed (explicitly, in a `finally`). While waiting for `/health` the state is polled every few seconds (docker: `docker inspect`; WSLc: `wslc list -a --format json`, read defensively, field names unknown) and the start fails at once when the container is not running any more (no 25-minute wait); `WSLC_E_CONTAINER_NOT_FOUND` in a stop or remove counts as success. Before any start the self-test checks the model's real size (Hub file list) against the card and refuses a 16-bit model that cannot fit ("about 55 GB in 16-bit; your card has 32 GB ..."). The report is written on every end, with the start command (key replaced), the container's last lines and the timings.

### WSLc commands (verified from `--help` of wslc 3.0.5.0)

`wslc stop -t 30 <name>`, `wslc remove -f <name>`, `wslc logs -n 60 <name>`, `wslc pull -q <image>`, `wslc list -a --format json`, `wslc volume create <name>`; the engine command is the one in "WSLc" above, without `--rm`. Still guesses: `cp`, `exec`, `inspect`-style state fields.

### The report and your privacy

`vllm_selftest.md/.json` are written to `<config folder>/vllm_selftest/` (Windows: `%APPDATA%\NDLoreForge\vllm_selftest\`, i.e. Roaming). The card shows that path as text with **Copy path**, **Open the folder** and **Copy the whole report**, and the report text inline (also after a failure); the Heal card does the same for the teacher check. The files and the diagnostics bundle have your user name and home folder replaced (`<user>`, `~`).

### Unknowns (the self-test and the teacher check are meant to answer them)

Whether `vllm-bnb-plugin` works for this architecture and how fast it is; the real shape of `prompt_logprobs` with `flat_logprobs`; the scoring speed on your PC; the WSLc `cp`/`exec` syntax; whether the AWQ teacher is "close" to the NF4 one on your data; how much of the start-up is download, weight load or graph capture (the check's `load_phases`).

### Update: the real `wslc cp` / `exec` syntax and what the last report showed

From the owner's report (wslc 3.0.5): `wslc cp [-a] [-L] [-q] <source> <target>` (source: local path, `CONTAINER:PATH` or `-`; target: local path or `CONTAINER:PATH`) and `wslc exec [-d] [-e KEY=VAL] [--env-file] [-i] [-t] [-u user] [-w dir] <container-id> <command> [<arguments>...]`: the defaults are `wslc cp {src} {name}:{dst}`, `wslc cp {name}:{src} {dst}`, `wslc exec {name} {args}` and (detached) `wslc exec -d {name} {args}` (editable; the combination is untested). `wslc list -a --format json` printed nothing when no container existed, so its field names are still unknown: the self-test saves a sample right after its own container starts and in the failure path, and the running check accepts several shapes (a name found in the output with no stop word counts as running). Captured console text is tidied (CRLF, blank-line runs) and a failed run still prints the criteria as "not reached".


### The identity rule also holds for checks, trace generation and Auto-cut (one function)

The same decision (`vllmquant.select_original`, used by the cache runner, `thinkcheck_vllm`, trace generation and the self-test's "original" runs; the program's `vllm::teacher_route` / `served_text` show it) applies wherever vLLM stands in for the owner's ORIGINAL (a thinking check of the original, trace generation by the original, the original's check inside Auto-cut): if the original fits the card in 16 bits it is served from its own folder; if not, a pre-quantized checkpoint is used ONLY when the original's repo id is in the confirmed-pairs table (default: `Qwen/Qwen3.8-27B` -> the configured AWQ, nothing else) or you confirmed that copy for THIS original; otherwise it is served from its own folder quantized to 4 bits while loading (bitsandbytes, **unverified**), and if that does not fit or the route is refused (`prequant` without a copy), the vLLM check refuses with a plain message and the standard engine does the check (result key `fallback`, standard result file, byte-identical keys). The thinking gates compare each cut with the ORIGINAL's numbers, so checking the base model instead would set the wrong yardstick; trace generation would write in the wrong model's voice. The result and cache key of a vLLM check of an original carry `quant` (`bnb-nf4` or `prequant:<repo>`) and `teacher_origin`; other checks have no such key. The preflight and the check card say which model vLLM loads ("original: <id> (4-bit on the fly)" or "confirmed copy <repo>"). The self-test is a **generic engine test on the configured checkpoint**: its report says it does NOT test your own original.
