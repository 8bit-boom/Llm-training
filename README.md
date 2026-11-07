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
| M5 onward (data, training, export ...) | not started |

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
