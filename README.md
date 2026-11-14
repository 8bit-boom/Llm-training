# ND Lore Forge

A Windows/Linux desktop program (GUI) that fine-tunes small LLMs (Qwen3.5 2B/4B/9B, Gemma 4 E2B/E4B) on the lore, rules and
mechanics of one nd-world world. Written in **Rust** ([Tauri 2](https://tauri.app) window + `ndforge-core`); the training itself
runs in a small Python worker (Unsloth) that the program starts. Plan: [PLAN.md](PLAN.md). The file format shared with nd-world:
[docs/BUNDLE_FORMAT.md](docs/BUNDLE_FORMAT.md).

## Status

| Milestone | State |
|---|---|
| M0 Bundle format spec + sample bundle | done |
| M1 Guided app (5 steps, help panel, projects), Rust + Tauri | done |
| M3 Step 1: import `.ndtrain`, verification, diff | done |
| M2 nd-world "Training bundle" download | built in the nd-world repo; waiting for its full test suite |
| M4 Training setup: GPU/driver checks, uv install, Python worker, GPU self-test | built; needs a run on a real NVIDIA PC (the RTX 5090) |
| M5 Step 2: make training data (world book + simple facts), review, save versions | done |
| M6 Step 3: PEFT + SFT training (Unsloth), live charts, stop/resume | built; needs a first run on the RTX 5090 |
| M7 Step 5: GGUF + Modelfile, install into Ollama (PC or TrueNAS), "Retrain with the new export" | built; needs a first run with a real model and Ollama |
| M8 Train on the TrueNAS server (worker server + client + compose) | built; the container image still has to be built once |
| M15 Train in Unsloth Desktop (third "Where to train" option, adapter copied back when it runs on this computer) | built and tested against a stand-in Unsloth; not tried with a real Desktop (docs/UNSLOTH_DESKTOP.md, "Not verified yet") |
| M9 Full fine-tune + "world book first, then practice" stages, type cards | built; needs a first run on the RTX 5090 |
| M10 Teacher-generated examples + checking (Step 2) | built; needs a first run with a real teacher model. "Better scores" can only be shown by the M11 test page |
| M11 Step 4 Test: scores, side-by-side chat, transcripts, run comparison | built; needs a first run with a real model |
| M12 Installers (Inno Setup, AppImage), in-app guide with screenshots, troubleshooting, problem report | Linux AppImage and .deb built and started here; the Windows installer and clean-machine runs are not verified (docs/INSTALL.md) |
| M13 Slim or grow a model: layer pruning and depth up-scaling as a new base model | built and tested without a GPU (worker on stand-ins, a fake worker in the Rust tests, the page in a stand-in browser); needs a first run on the RTX 5090 (docs/RESHAPE.md, "Not verified yet") |
| M14 Thinner layers (width pruning) + healing by distillation | built and tested without a GPU or torch (numpy stand-ins, fake workers, the page in a stand-in browser; the CPU torch tests are written and have never run); needs a first run on the RTX 5090 (docs/RESHAPE.md, "Not verified yet") |
| Importance matrix for the small GGUF sizes (optional, Step 5) | built and tested with stand-in llama.cpp tools; not run with a real llama.cpp or GPU (docs/IMATRIX.md, "Not verified yet") |
| Several worlds in one project (train on all at once) | built and tested on small sample worlds (datasets, teacher examples, test probes); not tried with real worlds or a real training run |

## Install

- **Installers:** Windows `.exe`, Linux AppImage and `.deb` on the [Releases page](https://github.com/8bit-boom/Llm-training/releases).
- **From a git clone:** `git clone https://github.com/8bit-boom/Llm-training && cd Llm-training && ./install.sh && ./run.sh` (Linux); on Windows double-click `install.bat`, then `run.bat` (and `update.bat` later to get the newest version). Details and prerequisites: `docs/INSTALL.md`.

## Build and run

Needs Rust (stable). On Linux also the Tauri libraries: `libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev libayatana-appindicator3-dev patchelf`.

```
cargo test --workspace            # the core (no window needed)
cargo run -p nd-lore-forge        # the program (development build)
cargo install tauri-cli --version "^2" && cargo tauri build     # installer (MSI/NSIS, AppImage/deb)
```

To try step 1 without nd-world, choose `tests/fixtures/sample.ndtrain` (regenerate: `cargo run -p ndforge-core --example make_sample`).
`NDFORGE_CONFIG_DIR` and `NDFORGE_PROJECTS_DIR` move the settings and the projects folder.

## Training setup (M4)

Open **⚙ Training setup** in the window. It checks the NVIDIA card and driver (RTX 50 cards need driver 570+), free disk space and
[uv](https://docs.astral.sh/uv/), installs a Python environment with the right CUDA build of PyTorch, and runs a 30-second GPU self-test
(`worker/`, `python -m ndforge_worker job.json`, JSON lines on stdout; see `worker/ndforge_worker/protocol.py`). The worker has its
own tests: `cd worker && python -m pytest tests`. `NDFORGE_ENV_DIR`, `NDFORGE_WORKER_DIR` and `NDFORGE_UV` override where things are found.

## Training data (M5)

Step 2 turns the world into a **world book** (every entry, rules section, session, quest, table and character as readable text) and
**simple-fact examples** (who/what is X, other names, connections, details, lists, random tables, sessions and quests, rules,
characters, GM secrets). No AI model is used: the same world and recipe always give the same examples (stable ids). GM secrets
(`[gmonly]`, `[for=]`, `[reveal=]`, `:::gm`) are kept but labelled, e.g. `(GM secret: ...)`. The draft (`datasets/draft/`) keeps your Drop / Keep /
Edit decisions when it is made again; **Save this version** freezes `datasets/vN/{train,test}.jsonl` + `manifest.json`.
Code: `crates/ndforge-core/src/dataset/` (`markup`, `docs`, `chunk`, `builders`, `split`, `store`).

## Training (M6)

Step 3 trains a model with **LoRA** (16-bit by default; 4-bit QLoRA when memory is short) on the newest saved data version. `catalog/models.toml` lists the
six models (repos checked on Hugging Face). The Rust program plans the run (`train/plan.rs`: presets Quick / Normal / Best, memory estimate, lowers batch and
sequence length to fit, replaces the estimate with the peak measured on this PC) and starts the Python worker (`worker/ndforge_worker/trainjob.py`), which
reports JSON lines: `start`, `metric` (loss, learning rate, memory, GPU temperature, time left), `sample` (answers to fixed questions before/during/after
training), `progress`, `log`, `result`. **Stop** writes a STOP file; the worker saves a checkpoint after the current step and exits (killed after 2 minutes
otherwise); **Continue** resumes from the last checkpoint. Each run lives in `runs/<name>/` (`config.json`, `state.json`, `metrics.jsonl`, `checkpoints/`, `adapter/`).

Worker tests (CPU, tiny random models, no downloads): `pip install torch transformers trl peft datasets pytest && cd worker && python -m pytest tests`.

## Export and retrain (M7)

**Step 5** makes `exports/<run>-<size>/` with `model-<size>.gguf`, a `Modelfile` and a model card. The worker merges the trained LoRA into the model (16-bit) and makes the
GGUF with Unsloth's `save_pretrained_gguf`, or, without Unsloth, with a llama.cpp it finds (`LLAMA_CPP_DIR`, `~/llama.cpp`, a `llama.cpp` folder next to the program).
Qwen3.5's chat template is edited to answer without thinking by default. An optional **importance matrix** ("Better small sizes", off by default) makes the Q6_K / Q5_K_M / Q4_K_M files lose less quality: llama.cpp's `llama-imatrix` reads your own data plus some general text first (`docs/IMATRIX.md`; built with stand-in tools, not yet run with a real llama.cpp). **Install into Ollama** uploads the file as a blob and creates the model through Ollama's HTTP API
(`/api/blobs`, `/api/create`), so it works for this PC and for a server such as the TrueNAS box (`http://truenas:11434`; Ollama there needs `OLLAMA_HOST=0.0.0.0`).
**Test it** asks the installed model one of the training questions. The page lists what to set in nd-world (`OLLAMA_MODEL`, or `UNSLOTH_MODEL` + `LLM_CONTEXT_TOKENS`) and warns
that the model knows all GM secrets.

**Retrain with the new export** (Step 1, after a newer `.ndtrain` was imported): makes the examples again (drops and edits stay), saves a new data version, trains a new run with the
last finished run's settings (optionally continuing from its adapter with a shorter run, suggested when under a quarter of the world changed) and can then make the files and
install them. Earlier runs, versions and files are kept.

Tests: Rust `cargo test --workspace` (the template test uses `go` when installed), worker `cd worker && python -m pytest tests`.

## Training on the TrueNAS server (M8)

`crates/ndforge-server` is the worker that runs in a container on the server (`Dockerfile.worker`, `truenas-worker-compose.yml`): token-protected HTTP, one job at a time, runs the same
Python worker as a local run, keeps checkpoints, frees the card by unloading Ollama. In the window, **Step 3 → Where to train** picks this PC or a server; the run looks the same, the finished adapter is copied back.
Guide: [docs/TRUENAS_WORKER.md](docs/TRUENAS_WORKER.md). Protocol: `crates/ndforge-core/src/remote/mod.rs`.

## Training in Unsloth Desktop (M15)

Step 3 → Where to train can also send the saved data to **Unsloth Desktop** (Training setup: address, API key, Test) and follow the job there with the same charts; the adapter is copied back when
Unsloth runs on this computer. Local training stays the default. A small client over Unsloth's OpenAI-shaped fine-tuning API: `crates/ndforge-core/src/unsloth/`; what Desktop is, the failure messages and the checklist for a first
real run: [docs/UNSLOTH_DESKTOP.md](docs/UNSLOTH_DESKTOP.md); the calls both products share with nd-world: [docs/UNSLOTH_API_CONTRACT.md](docs/UNSLOTH_API_CONTRACT.md) with `tests/fixtures/unsloth/`.

## Full fine-tuning and two stages (M9)

Step 3 asks two questions. **How much of the model changes?** LoRA (default), QLoRA, or **Full fine-tune** (every weight; `method: "full"`, learning rate 1e-5, batch 1, 8-bit optimizer, gradient checkpointing;
the whole model is saved in `runs/<name>/model/`, exported without a merge, and a retrain can continue from it). A card that cannot do it is greyed out with the reason: this computer only, a bf16 card (RTX 30 series or newer) is needed,
memory (an estimate of about 6 bytes per parameter plus activations, replaced by the measured peak after a run), disk for the checkpoints. **What should it learn from?** Both together (default), or two stages: the world book as text, then
the practice answers (`stages` in the job; the step numbers continue, and Continue skips a finished stage). Details: `crates/ndforge-core/src/train/plan.rs`, `worker/ndforge_worker/trainjob.py`.

## Teacher examples (M10)

On Step 2, after the examples are made, an optional card lets a bigger model (any OpenAI-compatible server: Ollama's `/v1`, LM Studio, llama.cpp, vLLM, a hosted API; the key is kept in `secrets.json`) write deeper questions, rules practice, NPC role-play,
campaign history, lore search (nd-world's "Relevant world lore" context block), "not in the lore" answers, ~10 % general chat and house-style answers from the world's AI instructions. Every answer is cached in `datasets/teacher_cache/` (a stopped
or repeated run only pays for what is new), then a **judge** pass compares it with its source: *unsupported* is dropped (reviewable), *partly* is kept and marked, general chat / house style are marked "not checked". "Not in the lore" questions
use only invented names that appear nowhere in the world; NPC prompts never contain GM secrets. Teacher examples survive "Make the examples again", keep your Drop/Edit decisions, and general chat / house style never go to the test file.
Code: `crates/ndforge-core/src/teacher/`, tests with a mock server: `crates/ndforge-core/tests/teacher.rs`.

## Test page (M11)

Step 4 puts the held-out questions of the run's data version to the **original** and the **trained** model (the worker's `evaluate` job: the same weights with the LoRA layers switched off, or a fresh copy for a full fine-tune; greedy decoding) and scores them in Rust
(`crates/ndforge-core/src/evaluate/`): lore and rules by word overlap with the expected answer (or graded by the teacher model, optional), "made-up answers" with 8 invented names that appear nowhere in the world (did it decline or invent?), and
6 general-chat prompts (sensible or not). The verdict is plain ("Better than the original on 86 % of lore questions"), with warnings when the model invents lore or forgot general chat. Also: ask both models a free question side by side, save the chat to
`transcripts/`, and a table comparing every tested run. Results: `runs/<run>/test.json`. The word-overlap score is a rough guide, not a grade.

**From nd-world:** the teacher step can also run inside nd-world (Export & Backup → Better training examples, Python-free image). Its examples travel in the `.ndtrain` file as `teacher_examples.jsonl` and appear in Step 2 already checked; see `docs/BUNDLE_FORMAT.md`.

## Your own models from Hugging Face

Step 3 has **＋ Add a model from Hugging Face…**: a name like `unsloth/Qwen3-4B` or a pasted address. `crates/ndforge-core/src/train/hub.rs` reads `GET /api/models/<repo>` and refuses what cannot be trained (gated / private, GGUF-only, no chat template, needs `trust_remote_code`, pre-quantized, not a language model); otherwise it works out size, licence, a memory estimate and the chat-format family (Qwen3.5, Gemma 4, or `generic` = the model's own chat template). Added models are kept in `custom_models.json` in the settings folder. For `generic` models the Ollama install leaves the template to the GGUF. Tested with a stand-in Hugging Face server (`NDFORGE_HF_URL`) and a tiny model through the real worker; not tried against real Hugging Face repositories or real large models.

## Slim or grow a model (M13)

The sidebar item **🧬 Slim or grow a model** makes a new base model from one you have, by cutting **whole blocks of layers out** (slim, for example Qwen3.5 27B: 64 → 48 layers, about 21.6 B) or **copying a stretch in** (grow, for example Qwen3.5 9B: 32 → 48 layers). It appears in Step 3's model list; Step 4 can compare a run on it with the original; Step 5 reuses the original's vision file.

- **Slim** measures, on a sample of the project's own training text, how much each block of layers changes the hidden state (the angular distance between block boundaries, Gromov et al. 2024, "The Unreasonable Ineffectiveness of the Deeper Layers"; ShortGPT's Block Influence for separate blocks) and removes the least useful run of blocks. Big models are measured in 4-bit. **A slimmed model answers worse until it is trained again** (Step 3); the page, the help and the test say so.
- **Grow** is depth up-scaling (SOLAR 10.7B, mergekit "passthrough"). New layers start with zeroed output weights (LLaMA-Pro style), so the grown model first answers **exactly like the original**; the program checks this on the project's text and refuses to keep a model that fails. **Honest expectation: little or no gain from one world's text, about 40 % more memory, disk space and time.**
- Layers are moved as raw bytes between safetensors files (no model load), whole pattern periods only (Qwen3.5: 4 layers), so old and new Transformers / llama.cpp agree on the layer pattern. Gemma 4 and quantized models are refused with the reason.
- Unsloth Studio can only prune MoE experts (REAP), not layers, and not dense or hybrid Qwen3.5: that is why this is built here (docs/RESHAPE.md, section "Could Unsloth Studio do it?").

**Try first** (a card on the same page) finds out what a cut will cost before any model is written: the original is loaded once, a cut is imitated with hooks and the same text-prediction error as the made model's before / after numbers is measured for every block, every run of blocks, every width or your own choice (`docs/TRY_FIRST.md`; it has not been run on a real model yet).

Code: worker `reshape.py`, `reshapejob.py`, `safetensors_io.py`; Rust `crates/ndforge-core/src/reshape/`; page `ui/app.js` (smoke test `ui/smoke/`). Method, file formats, deviations from the plan and the list of what **has not been verified without a GPU**: `docs/RESHAPE.md`.

### Thinner layers and healing (M14, version 2)

The same page can also make **every layer thinner** (width pruning, the Minitron method: NVIDIA, arXiv 2407.14679 / 2408.11796): the neurons of each layer's MLP that matter least on the project's own text are taken out, in every layer by the same number (Qwen3.5 / 3.8 only). For the 27B the slider starts on **MLP 13,312 = 23.7 B parameters**; 21.1 B is on the same slider, and **Both** removes a block and thins the rest. The sizes are exact before anything is measured.

- **Healing** (the *Heal it* card): the original model teaches the thinner one to answer like it did (KL on its top-32 next-token probabilities, LoRA / QLoRA, 20 M tokens for 15 B and up, 40 M below, 10 to 80 M on offer; the original answers first and saves its answers to disk when both do not fit the card). About half of the text is general text **streamed from Hugging Face** (open default `mlabonne/open-perfectblend`; choose another dataset or your own file in the *General text* list, with a *Check it* button; see docs/GENERAL_TEXT.md; offline or gated falls back to the project's text). The LoRA is merged into a new base model folder (`models/<name>/healed`, numpy, no torch); the checkpoints are deleted, the teacher's answers kept until the model is deleted. Step 4's table shows **original · thinner · healed**.
- **Memory and settings:** healing chooses cautious settings by model size (sequence 1024, rank 32, batch 2, gradient checkpointing from 15 B), shows a live memory estimate, offers Expert *Healing settings*, retries by itself after an out-of-memory error, and offers *Try again with safer settings* (docs/HEAL_SETTINGS.md).
- **Honest expectation:** a thinned model is clearly weaker until healed, and 20 to 40 M tokens is a small fraction of the 94 B NVIDIA used; expect a usable model, not one as good as the original. Thinning the attention layers is offered in Expert mode, labelled experimental, and stays off until the GPU checklist is done.
- Code: worker `reshape_width.py`, `slice_io.py`, `distilljob.py`; Rust `reshape/plan.rs`, `reshape/runs.rs`. Plan: `docs/RESHAPE_V2_PLAN.md`; what was built, owner decisions, deviations and the **not verified yet** list: `docs/RESHAPE.md`; the practical guide: `docs/PRUNE_QWEN38_27B.md`.

## Training speed

What was researched (FlashAttention, SageAttention, `torch.compile`, quantization, FP8 and others), what is built (batches of similar length, tokens-per-second and padding numbers, a warning when Qwen3.5's fast kernels are missing, gradient checkpointing only when memory needs it on the plain-libraries path) and what is next: `docs/TRAINING_SPEED.md`. Not yet measured on the RTX 5090.

## Without Unsloth

Training setup has **Install without Unsloth** (standard libraries only; the program fetches Python and PyTorch itself) and Expert mode has *Training engine* (auto / unsloth / standard). A model can also be a folder on this computer (Step 3, *Add a model from Hugging Face…* → *Choose a folder…*; checked like a Hugging Face model, trained on this computer only). Details: `docs/INSTALL.md`, `docs/TRAINING_SPEED.md`.
