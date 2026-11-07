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
| M9 Full fine-tune + "world book first, then practice" stages, type cards | built; needs a first run on the RTX 5090 |
| M10 onward (teacher examples, test page, installers ...) | not started |

## Build and run

Needs Rust (stable). On Linux also the Tauri libraries: `libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev libayatana-appindicator3-dev patchelf`.

```
cargo test --workspace            # the core (no window needed)
cargo run -p nd-lore-forge        # the program
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
five models (repos checked on Hugging Face). The Rust program plans the run (`train/plan.rs`: presets Quick / Normal / Best, memory estimate, lowers batch and
sequence length to fit, replaces the estimate with the peak measured on this PC) and starts the Python worker (`worker/ndforge_worker/trainjob.py`), which
reports JSON lines: `start`, `metric` (loss, learning rate, memory, GPU temperature, time left), `sample` (answers to fixed questions before/during/after
training), `progress`, `log`, `result`. **Stop** writes a STOP file; the worker saves a checkpoint after the current step and exits (killed after 2 minutes
otherwise); **Continue** resumes from the last checkpoint. Each run lives in `runs/<name>/` (`config.json`, `state.json`, `metrics.jsonl`, `checkpoints/`, `adapter/`).

Worker tests (CPU, tiny random models, no downloads): `pip install torch transformers trl peft datasets pytest && cd worker && python -m pytest tests`.

## Export and retrain (M7)

**Step 5** makes `exports/<run>-<size>/` with `model-<size>.gguf`, a `Modelfile` and a model card. The worker merges the trained LoRA into the model (16-bit) and makes the
GGUF with Unsloth's `save_pretrained_gguf`, or, without Unsloth, with a llama.cpp it finds (`LLAMA_CPP_DIR`, `~/llama.cpp`, a `llama.cpp` folder next to the program).
Qwen3.5's chat template is edited to answer without thinking by default. **Install into Ollama** uploads the file as a blob and creates the model through Ollama's HTTP API
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

## Full fine-tuning and two stages (M9)

Step 3 asks two questions. **How much of the model changes?** LoRA (default), QLoRA, or **Full fine-tune** (every weight; `method: "full"`, learning rate 1e-5, batch 1, 8-bit optimizer, gradient checkpointing;
the whole model is saved in `runs/<name>/model/`, exported without a merge, and a retrain can continue from it). A card that cannot do it is greyed out with the reason: this computer only, a bf16 card (RTX 30 series or newer) is needed,
memory (an estimate of about 6 bytes per parameter plus activations, replaced by the measured peak after a run), disk for the checkpoints. **What should it learn from?** Both together (default), or two stages: the world book as text, then
the practice answers (`stages` in the job; the step numbers continue, and Continue skips a finished stage). Details: `crates/ndforge-core/src/train/plan.rs`, `worker/ndforge_worker/trainjob.py`.
