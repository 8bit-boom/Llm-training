# Slim or grow a model (M13): what was built, what was decided, what is not verified

The full plan is `docs/RESHAPE_PLAN.md` (written before the code; it is kept as it was). This file is what the code actually does, where it differs
from that plan, and above all **what has not been seen working yet**.

## What it is

A page (**🧬 Slim or grow a model**) that makes a new base model from one the program already knows, by changing the number of decoder layers:

- **Slim (layer pruning).** Remove whole blocks of layers (a block = one period of the layer pattern; Qwen3.5: 3 linear-attention + 1 full-attention layer).
  Which blocks: the worker runs a sample of the project's own training text (`datasets/vN/train.jsonl`, half world book, half practice conversations)
  through the model and records the hidden state at every block boundary. `D[a][b]` is the mean angular distance `arccos(cos(h_a, h_b)) / pi` over the
  tokens. *Contiguous* (default): remove the `n` blocks `a .. a+n-1` whose boundaries `a` and `a+n` are closest (Gromov et al. 2024, "The Unreasonable
  Ineffectiveness of the Deeper Layers"). *Separate*: the `n` blocks with the smallest `D[k][k+1]` (ShortGPT's Block Influence on blocks). The first and
  last block are protected; at most half of the blocks go. The calibration set is split into two halves by batch parity; `stable` says whether both
  halves chose the same blocks.
- **Grow (depth up-scaling).** SOLAR 10.7B / mergekit "passthrough": layers `0 .. half-1` followed by `half-m .. L-1`, with `m` = added layers and `half`
  rounded down to a whole period. Start `identity` (default): the second copy of every duplicated layer has its output projections
  (`linear_attn.out_proj` / `self_attn.o_proj`, `mlp.down_proj`) set to zero, so the grown model computes exactly the original (LLaMA-Pro idea). Start
  `copy`: plain copies. The identity start is checked: with torch available, the loss on the calibration batches before and after must agree within
  1e-3 relative, otherwise the model is not kept (`grow_check_failed`).
- The tensors are moved **as raw bytes** between safetensors files (`safetensors_io.py`, standard library only): no model load, a 56 GB model can be
  reshaped on a small machine. Vision tensors (`model.visual.*`), the multi-token head (`mtp.*`), embeddings and the output layer are copied unchanged;
  `config.json` gets the new `num_hidden_layers` and a remapped `layer_types`; tokenizer, chat template and processor files are copied.
- The result lives in `models/<name>/model/` of the project (with `ndforge_reshape.json` that says exactly which old layers make up each new layer)
  and is added to the model list as `custom-local-<name>`, with `derived_from` and `reshape` fields.

Where the code is: worker `safetensors_io.py`, `reshape.py` (pure), `reshapejob.py` (jobs `reshape_inspect` and `reshape`); hooks in `evaluatejob.py`
(`trained_base`, `load_4bit`, `loaded`), `exportjob.py` (`base.vision_repo`), `backends.py` (`extra_targets`, `layers_to_transform`); Rust
`crates/ndforge-core/src/reshape/` (`plan.rs`, `runs.rs`), `evaluate/runs.rs` (`Side`, `Pair`), `export/runs.rs`, `train/runs.rs`; window commands in
`src-tauri/src/lib.rs`, page in `ui/app.js` (smoke test `ui/smoke/reshape.test.mjs`).

## Owner decisions (answers to the plan's section 12)

1. **4-bit scoring is acceptable** and is the default for big models (a 27B on a 32 GB card is measured in NF4). CPU-offload in 16-bit stays the
   fallback (`load: offload`, Expert mode, and what `auto` picks when even 4-bit does not fit).
2. **Reshaped models live inside the project** (`models/<name>/`), not in a shared folder.
3. **Grow is offered in normal mode**, not only in Expert mode. The honest expectation stays visible next to it: little or no gain from one world's text,
   about 40 % more memory, disk space and time (page intro, the radio label, the per-size note, the help text and the README).

## Could Unsloth Studio do it instead? (plan section 0, checked against the sparse clone at commit 1bbf9c3)

Only MoE **expert** pruning (REAP, `studio/backend/core/export/reap.py`, applied at export time through `FastLanguageModel.reap_prune`). It removes routed
experts inside every MoE layer; it never changes the number of layers, and it does not apply to dense or hybrid models such as Qwen3.5 2B / 4B / 9B /
27B (`moe_info_from_config` returns nothing for them). A search of `studio/` for pruning, up-scaling, passthrough, `num_hidden_layers` editing or
`layers_to_transform` finds nothing else. So layer pruning and depth up-scaling are built here. A folder made here can be opened in Studio afterwards.
Expert pruning for `qwen3_5_moe` bases is a possible follow-up (below).

## Facts the design depends on (from the plan; V = read in source, A = assumption)

| # | Fact | |
|---|---|---|
| F1 | Qwen3.5 text config: `num_hidden_layers`, `layer_types`, `full_attention_interval` (4), `mtp_num_hidden_layers`, `mlp_only_layers`; nested in `text_config` next to `vision_config` | V (real 4B config in `worker/tests/data`) |
| F2 | Without `layer_types`, transformers builds it with the interval rule | V |
| F3 | `len(layer_types) == num_hidden_layers` is validated | V |
| F4 | Layers are built from `layer_types[layer_idx]`; renumbering 0..N-1 with a matching list is all a reload needs | V (reading); A (a real reload: `test_reshape_torch.py`, not run here) |
| F5 | A layer is `h + mixer(norm(h))`, `h + mlp(norm(h))`; zeroing mixer-out and `down_proj` makes it an identity | V (reading); A (numerically, GPU and CPU torch test not run here) |
| F6 | VLM key layout `model.language_model.layers.N.*`, `mtp.*`, `model.visual.*`; `AutoModelForCausalLM` loads the text part | V |
| F7 / F8 | llama.cpp writes `recurrent_layers` from `layer_types` and keeps `mtp.*`; the interval pattern is kept so old readers agree | V (reading) |
| F10 | `HfApi.get_safetensors_metadata` gives dtypes and shapes without downloading | V (reading); only a stand-in is tested here |
| F13-F15 | A local folder can be a base model; export falls back to `--mmproj` conversion; the evaluate job can load two models in turn | V |

## File formats

Unchanged from the plan: job `reshape` (section 5.1), `models/<name>/model/ndforge_reshape.json` (5.2), the result (5.3), `models/<name>/state.json` (5.4),
the two optional catalog fields (5.5). Extras that were added: `models/<name>/scores.json` (the `reshape_scores` event, for the chart after a restart),
`models/<name>/config.json` (the job as sent). `ndforge_reshape.json` also records `calibration.version` (the data version used) and
`calibration.tokens` (how many tokens were measured).

## Deviations from the plan

Each is the smallest change that made something work or kept something honest.

1. **`plan_tensors` is stricter than the plan.** The plan raises only when a layer has neither a mixer output nor a `down_proj`; the code requires **both**,
   because zeroing one of them does not make a layer an identity. A layer without both (for example a MoE layer whose MLP output lives in experts) cannot start
   as an exact copy; the message offers "copy the layers". Affects only `qwen3_5_moe` / `qwen3_moe` grows.
2. **`_score` returns a fifth value**: `(D, D_half1, D_half2, loss_before, {"load", "tokens"})`, because the result must say how the model was loaded and
   how many tokens were used.
3. **Stop is a result, not an error.** On the STOP file the worker removes `model.partial` and answers `{"stopped": true, "model_dir": null}` (exit 0);
   the Rust side maps that to state `stopped`. A killed worker (after the 60 s grace) is "Stopped." with the same cleanup.
4. **`layer_types_of` assumes the transformers default interval (4) for `qwen3_5*`** when neither `layer_types` nor `full_attention_interval` is in
   `config.json`, because `save_pretrained` leaves default values out of the file.
5. **`evaluate::runs::Pair` has a fifth field, `against`** ("base" or "origin"), so `execute_pair` can write it into `test.json`.
   `test.json` keeps its old keys and gains `against` and `loaded`. `compare()` rows for made models have `method` `slim` / `grow` and `steps` null.
6. **`Hyper` has a new field `only_new_layers`** (default false). The expert checkbox *Train only the new layers* has to go through the same override
   mechanism as every other setting, which only accepts keys that exist in `Hyper`. `extra_targets` and `layers_to_transform` themselves are added to the job only,
   as planned.
7. **`reshape_options` returns the warnings and the disk space per size** (`options[].warnings`, `options[].disk_needed_gb`) instead of one list for the
   default size, so the page does not repeat Rust's rules. `default_blocks` (a quarter of the blocks) is the slider's first position.
8. **`test_ask` takes `against` too** (not only `test_start`), so the free chat on Step 4 can use the same choice.
9. **`scoring_load` with no known card answers "4bit"** (as the plan's test says); the worker itself loads 16-bit on a computer without CUDA. The value only
   describes the default on the page.
10. **Extra helpers**: `check_whole_blocks` (explicit layer lists), `reshape::runs::execute_with_grace`, `AppModel::reshape_options_for` (so the window can
    run the first look on its own thread), a cache of `reshape_inspect` answers per model (`reshape_inspect.json` in the settings folder), and
    `ui/smoke/` with a CI step (`node --test ui/smoke/*.test.mjs`).
11. **The grow self-check without a training file** uses four built-in sentences; without torch it is skipped with a note in the result (the plan only
    mentioned "when torch is present").
12. **Names**: two models made in the same second get a `-2` suffix.
13. **`train::runs::drive`'s STOP file** is `models/<name>/STOP`; the worker looks at `Path(out).parent / "STOP"`, the same folder.

## Not verified yet (needs a GPU run)

Everything below is **implemented as the plan says and has not been seen working**. None of it should be described as working.

**Never executed in this environment (no torch, transformers or safetensors here; the network proxy blocks the PyTorch host):**
- `reshapejob._load`, `_find_layers`, `_calibration`, `_measure`, `_score`, `_loss`, the loss check and the exact-copy self-test with a real model. Their control
  flow in `run` is tested with stand-ins (`test_reshape.py`: scoring flow, unstable halves, failed and passed self-test, failing and skipped check); the
  torch code itself is not.
- The seven torch tests of the plan, written in `worker/tests/test_reshape_torch.py` (pruned model equals the original with layers skipped; identity grow is
  exact, also through the job's check; copy-grow loads and generates; scoring picks the middle block and reports finite numbers; LoRA run on a pruned base
  then evaluate with `trained_base`; extra targets reach `out_proj`; GGUF with a real llama.cpp). They are skipped here (`importorskip`) and run in CI
  (`.github/workflows/ci.yml`, job `worker`, CPU torch). **They have never run once.** If one fails in CI, fix the test or the code; do not weaken it.
- That `save_pretrained` of the tiny Qwen3.5 text model uses key names `LAYER_KEY` knows (`model.layers.N.`; `model.language_model.layers.N.` is also handled).
- `layers_to_transform` in Unsloth's `get_peft_model` (the code tries it and falls back with a log line); `layers_to_transform` together with
  `target_modules="all-linear"` in PEFT on the plain-libraries path.
- A real Hugging Face download (`snapshot_download` with the allow-list) and `get_safetensors_metadata` (both are stubbed in tests).
- The window in a real Tauri window: the page, the slider and the buttons are tested with a stand-in browser (`ui/smoke/`), and the Rust side builds
  (`cargo build -p nd-lore-forge` succeeds here), but nobody has clicked through the real window. The fake-worker tests of the Rust side run on unix only.

**The GPU checklist of the plan (section 9.4), to be done on the RTX 5090 and recorded here:**

| # | Check | Result |
|---|---|---|
| 1 | 9B: measure in 16-bit and in 4-bit: the same span chosen? time and peak VRAM (validates 4-bit measuring) | not run |
| 2 | 27B: slim 64 → 48 layers, 4-bit measuring: time, peak VRAM / RAM, disk; the output loads in Transformers and in Unsloth `FastLanguageModel` | not run |
| 3 | Test page, original vs slim (both 4-bit): lore / rules / general numbers; then QLoRA healing (Step 3) and test against the original | not run |
| 4 | 9B: grow 32 → 48 identity: the check passes on the GPU (loss equal); does 16-bit LoRA fit; test against the original after training | not run |
| 5 | Export: Unsloth GGUF and llama.cpp GGUF of the 48-layer model; `llama-server -m ... --mmproj <official>` answers an image question; Ollama install and *Test it* | not run |
| 6 | Speed in tokens/s against the original (the benchmark on the made model) | not run |

**Assumptions the plan marked [A] / [GPU] that remain open:** Unsloth loads a 48-layer local Qwen3.5 folder; Unsloth's bundled llama.cpp and Ollama's
Qwen3.5 engine accept 48 layers; 4-bit measuring ranks blocks like 16-bit; the quality after healing (QLoRA on Qwen3.5 is discouraged by Unsloth); the
estimated sizes (a 27B slimmed by 4 blocks is about 21.6 B; the 9B grown by 4 blocks about 13.1 B; the page computes them from the real tensor shapes, the
plan's figures were estimates); measuring time and memory; whether a world's text gives any gain from growing (expected: no).

## Follow-ups (optional steps 18-20 of the plan, not built)

- **Distillation heal** (`distill_cache` job and a KL term in `trainjob.py`): only worth building if the checklist shows that plain training does not win
  back enough quality after slimming.
- **MoE expert pruning (REAP)** for `qwen3_5_moe` bases through `FastLanguageModel.reap_prune`, when Unsloth is installed. The dataset format Core expects
  is not in the sparse clone and has to be read first.
- **Gemma 4**: slice `embed_tokens_per_layer` / `per_layer_model_projection` by layer and keep `num_kv_shared_layers` consistent. Refused with a reason until then.
