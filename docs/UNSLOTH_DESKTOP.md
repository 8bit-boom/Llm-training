# Unsloth Desktop in ND Lore Forge

Unsloth Desktop is a native app (Windows, macOS, Linux) that runs Unsloth Studio, the training and chat interface, on your own computer. Lore Forge can send a saved
data version to it and follow the training, instead of training with its own worker. The calls both products use are in `docs/UNSLOTH_API_CONTRACT.md`.

## Owner decisions

1. Desktop is a **third "Where to train" option**, never the default. The worker on this computer stays the main trainer (it does two stages, full fine-tuning, continuing after a stop and exact memory figures).
2. The trained adapter is **copied back from Desktop's run folder only when Desktop runs on the same computer**. A Desktop on another computer ends with "the adapter stays there" in this version.
3. Experimental items (a GGUF made inside Unsloth, loading it for chat, an Ollama import) are **not built**; steps that need the real Desktop and a graphics card are marked below.
4. **No shared crate** with nd-world: Lore Forge has its own small client (`ureq`, already a dependency) and the contract document plus the JSON fixtures in `tests/fixtures/unsloth/`.
