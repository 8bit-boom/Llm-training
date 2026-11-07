# ND Lore Forge

A Windows/Linux desktop program (GUI) that fine-tunes small LLMs (Qwen3.5 2B/4B/9B, Gemma 4 E2B/E4B) on the lore, rules and
mechanics of one nd-world world. The plan is in [PLAN.md](PLAN.md); the file format shared with nd-world is in
[docs/BUNDLE_FORMAT.md](docs/BUNDLE_FORMAT.md).

## Status

| Milestone | State |
|---|---|
| M0 Bundle format spec + sample bundle | done |
| M1 Guided app skeleton (5 steps, help panel, projects) | done |
| M3 Step 1: import `.ndtrain`, verification, diff | done |
| M2 nd-world "Training bundle" download | next (nd-world repo) |
| M4 onward (environment, data, training, export ...) | not started |

## Run from source

```
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ndforge                                           # or: python -m ndforge
pytest -q                                         # QT_QPA_PLATFORM=offscreen on a machine without a display
```

On Linux the Qt libraries need `libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3`.

To try step 1 without nd-world, use the sample file `tests/fixtures/sample.ndtrain` (regenerate it with `python tests/sample_bundle.py`).
