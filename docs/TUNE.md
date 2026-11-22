# Tune this PC for this model (auto-tune and presets)

Plan: `AUTOTUNE_PRESETS_PLAN.md` (planning notes). Code: `crates/ndforge-core/src/tune/`. Owner's request: press ONE button, every setting of the Heal card is tested for the selected model, the winners are applied after one confirmation, and the settings are saved as a preset that can be selected later.

## Identity (`tune/identity.rs`)
* `shape_id` (`s-` + 16 hex) is the hash of the canonical speed-relevant facts of `config.json` (architecture, layer count and types, widths, heads, linear-attention sizes, vocabulary, tied embedding, vision/MTP/pre-quantized flags). Key order and spacing never change it. **Speed depends on the shape, not on the weight values**, so a healed or merged copy of a cut has the cut's shape.
* `weights_id` (`w-` + 16 hex) hashes `config.json` and, per `*.safetensors`, the name, size and header: no modification times. It tells entries apart; it never gates reuse.
* `shape_of_plan` gives the shape a planned cut will have (layers kept, new width), as the worker's `fix_config` writes them, so Auto-cut can find a preset before the cut exists.
* `MachineId.key` is the card's name and memory only; driver, torch, transformers, peft, bitsandbytes, triton, fla, vLLM and llama.cpp versions are `software` facts that make single results stale.
