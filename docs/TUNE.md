# Tune this PC for this model (auto-tune and presets)

Plan: `AUTOTUNE_PRESETS_PLAN.md` (planning notes). Code: `crates/ndforge-core/src/tune/`. Owner's request: press ONE button, every setting of the Heal card is tested for the selected model, the winners are applied after one confirmation, and the settings are saved as a preset that can be selected later.

## Identity (`tune/identity.rs`)
* `shape_id` (`s-` + 16 hex) is the hash of the canonical speed-relevant facts of `config.json` (architecture, layer count and types, widths, heads, linear-attention sizes, vocabulary, tied embedding, vision/MTP/pre-quantized flags). Key order and spacing never change it. **Speed depends on the shape, not on the weight values**, so a healed or merged copy of a cut has the cut's shape.
* `weights_id` (`w-` + 16 hex) hashes `config.json` and, per `*.safetensors`, the name, size and header: no modification times. It tells entries apart; it never gates reuse.
* `shape_of_plan` gives the shape a planned cut will have (layers kept, new width), as the worker's `fix_config` writes them, so Auto-cut can find a preset before the cut exists.
* `MachineId.key` is the card's name and memory only; driver, torch, transformers, peft, bitsandbytes, triton, fla, vLLM and llama.cpp versions are `software` facts that make single results stale.

## Presets (`tune/preset.rs`, `tune/store.rs`)
* A preset (`format: 1`) holds the model identity, the machine, the tested functions (`heal.student_batch`, `heal.loss`, ...), and the WHOLE Heal card as tested (`card`), so it can be applied as a whole. Each function has `chosen` and `default` values, the gain, the measured spread, its class and tier, the guards that were checked, every candidate that was tried (an out-of-memory is a row), what it `depends_on`, which knobs are `transferable` to another model, and the `context` (sequence length, method, rank ...) it assumed. Unknown fields are kept (`extra`), so a later program version can add some.
* Where: `<project>/tune/presets/<id>.json` and the app-wide `<config folder>/tune_library/<id>.json`. Neither folder is a Storage root. A file that cannot be read is set aside as `.broken-<time>` and listed; a preset of a newer format is left where it is and listed.
* Matching, best first: **exact** (same weights and machine), **same shape** (the speed should be the same), **from a parent** (only the transferable knobs), **nearest** (same architecture within 0.67 to 1.5 times the size; transferable knobs, re-test recommended); **another PC** is listed and never offered.
* Stale: a function is stale when a component it depends on changed (driver major, a package's minor, the vLLM image, the llama.cpp version). Age alone never makes a result stale. A component that is not known now is not a change.
* Delete needs the preset's exact name typed. Export writes `<name>.ndtune.json` (`{"ndtune": 1, "preset": ...}`, the home folder replaced by `~`); import goes to the library, refuses a newer format, and needs `overwrite` for an existing id.
