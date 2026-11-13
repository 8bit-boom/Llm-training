# Slim or grow a model (M13): what was built, what was decided, what is not verified

The full plan of version 1 is `docs/RESHAPE_PLAN.md`, the plan of version 2 (thinner layers and healing) is `docs/RESHAPE_V2_PLAN.md`; both were written before the code and are kept
as they were. This file is what the code actually does, where it differs from those plans, and above all **what has not been seen working yet**.
Version 1 (whole layers) is described first; **version 2 (thinner layers + healing) starts at "Version 2"** below.

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

## Version 2: thinner layers (width pruning) and healing

### What it is

The same page, a new way of slimming: instead of removing whole layers, **every layer is made narrower**. The MLP of each layer (62 % of a Qwen3.5 27B: 17.1 of 27.8 B parameters) loses the
neurons that matter least, measured on the project's own text. In Expert mode the linear-attention layers can lose value heads and the full-attention layers query heads (experimental, switched off
until the GPU checklist below is done). Then **healing** gives the thinner model back what thinning took away: the original model (teacher) teaches it to answer like it did (distillation),
on the project's text and on general text; the LoRA that learns this is merged into a new base model folder.

- **How:** *Remove whole layers* (version 1, the result runs faster), *Make every layer thinner* (keeps more quality for the same size, needs healing), or *Both* (remove blocks and thin the rest).
  NVIDIA's own comparison: width pruning gives better accuracy at a fixed parameter count, depth pruning gives faster inference at a fixed parameter count.
- **Method:** Minitron (Muralidharan et al. 2024, arXiv 2407.14679, "Compact Language Models via Pruning and Knowledge Distillation"; Sreenivas et al. 2024, arXiv 2408.11796, "LLM Pruning and
  Distillation in Practice: The Minitron Approach"; Minitron-SSM, arXiv 2504.11409, for the group-aware pruning of state-space heads) and NVIDIA's Model-Optimizer
  (`modelopt/torch/prune/plugins/mcore_minitron.py`) as the reference implementation. The importance of a structure is the L2 norm over the measuring texts of the mean absolute activation that
  feeds the output projection of its block (`mlp.down_proj`, `self_attn.o_proj`, `linear_attn.out_proj`): it is the quantity the removed columns would have multiplied. The count of kept structures
  is the same in every layer (transformers and llama.cpp build every layer from one config value); which ones are kept differs per layer. The paper bodies were not read when this was planned
  (the sites were blocked); claims about them come from the abstracts, NVIDIA's model cards and Model-Optimizer's code and guidelines.
- **Not done:** `hidden_size` (shared with the vision merger, which the text-only healing never trains, and the official mmproj file would no longer fit), head dimensions (llama.cpp needs equal
  key and value sizes), key heads and KV heads, per-layer sizes, LLM-Pruner (no Qwen support), MoE models.
- **Dense Qwen3.5 / Qwen3.8 only** (`qwen3_5`, `qwen3_5_text`); every other model keeps the layer options and the page says why width is not offered.

### What can be pruned, and how (the numerically checked slicing)

A CI test (`worker/tests/test_reshape_width_torch.py`) checks the MLP, the value heads, the query heads, all three together, and depth plus width: the thinned model's logits equal the original's with the removed structures masked (their input columns of the output
projection set to zero), atol 1e-4 in float32. The same check was made once by hand on CPU torch before the code was written (maximum difference 1.9e-6; see `docs/RESHAPE_V2_PLAN.md`).

| Dimension | Tensors touched (per layer) | Constraint | Config key | Offered |
|---|---|---|---|---|
| MLP `intermediate_size` | `mlp.gate_proj` rows, `mlp.up_proj` rows, `mlp.down_proj` columns; all layers and `mtp.layers.0.mlp.*` | same count in every layer; multiple of 256; at least half kept | `intermediate_size` | normal |
| GatedDeltaNet value heads | `in_proj_qkv` rows of the v part, `conv1d` the same channels, `in_proj_z` rows, `in_proj_b` / `in_proj_a` rows, `A_log`, `dt_bias`, `out_proj` columns; the gated norm is shared and untouched | equal count per key head; one step (48 to 32 on the 27B, 32 to 16 on the 9B) | `linear_num_value_heads` | Expert, experimental, off |
| Full-attention query heads | `q_proj` rows (query and gate of each head, `2 * head_dim` rows), `o_proj` columns; `q_norm` / `k_norm` untouched; also `mtp.layers.0.self_attn.*` | equal count per KV group; multiples of the KV heads, at least two thirds | `num_attention_heads` | Expert, experimental, off |
| Everything else (embeddings, norms, vision, `mtp.fc`, k/v projections) | byte copy | | | |

`head_dim` is written explicitly into the config when heads change (transformers would otherwise derive it from `hidden_size // num_attention_heads`). The multi-token head is never run by
transformers, so it is cut by weight size (`||down[:, i]|| * ||up[i, :]||` per neuron, `||o_proj columns||` per head) and the result says it was **not measured or healed**.
bf16 tensors are cut as raw `uint16` with numpy: bit-exact, no bf16 type needed.

### Sizes (exact, before anything is measured)

Qwen3.5 / Qwen3.8 27B (64 layers, MLP 17,408, 48 value heads, 24 heads; the formula reproduces the Hugging Face total 27,781,427,952 exactly, checked in `worker/tests/test_reshape_width.py`):

| Variant | Layers | MLP | Parameters | Disk, 16-bit |
|---|---|---|---|---|
| original | 64 | 17,408 | 27.78 B | 56 GB |
| **mild cut, the slider's first position** | 64 | 13,312 (−24 %) | **23.69 B** | 47 GB |
| thinner | 64 | 12,288 (−29 %) | 22.67 B | 45 GB |
| the 21 B cut (needs healing) | 64 | 10,752 (−38 %) | 21.14 B | 42 GB |
| **Both**: one block less | 60 | 12,288 | 21.46 B | 43 GB |
| Both: two blocks less | 56 | 13,312 | 21.15 B | 42 GB |
| most the page allows | 64 | 8,704 (−50 %) | 19.09 B | 38 GB |

The Expert axes lower this further (13,312 with 32 value heads and 16 heads: 21.64 B). For the 9B (MLP 12,288, 32 value heads, 16 heads): MLP 9,216 gives 8.41 B; 9,216 with 16 value heads 7.80 B.
The Rust side computes the same numbers (`reshape::plan::width_params`), tested against `worker/tests/data/width_params.json`, which both test suites read.

### Owner decisions for version 2 (answers to the open questions of the plan, section 7)

1. **The slider's first (suggested) position for the 27B is the mild cut: 23.7 B (MLP 13,312).** 21.1 B stays reachable on the same slider with a "needs healing" warning; **Both** is offered as well
   (one block removed plus MLP 12,288 = 21.5 B). The page starts on 13/17 of the MLP, and on one block plus 12/17 for Both (`plan::suggested_widths`).
2. **General text for healing is on by default** (about 50 % of the healing tokens), switched off in Expert mode (share 0). The page says plainly that it is **downloaded from Hugging Face**
   (default dataset `nvidia/Nemotron-Pretraining-SFT-v1`, NVIDIA's recommendation for distillation data) and that this needs an internet connection. **Offline** (or any download error) is not an
   error: healing goes on with the project's own text only, with a plain-language note on the Heal card and in the result.
3. **Heal budget:** 20 M tokens for students of 15 B or more, 40 M below; the page allows 10 / 20 / 40 / 80 M and shows an estimate of the time and of the disk space for the teacher's answers.
4. **The healed adapter is merged into a NEW base model folder** (`models/<name>/healed/`, numpy, one tensor at a time, no torch). The distillation checkpoints are deleted afterwards; the
   teacher's saved answers (`distill/cache/`) are kept until the model is deleted (the LoRA itself, small, stays in `distill/run/adapter/`).
5. **Expert axes** (GatedDeltaNet value heads, attention heads) are implemented end to end (worker, Rust, page) but offered **in Expert mode only, labelled "experimental", and disabled** until the GPU
   checklist below is marked passed. The switch is the constant `EXPERIMENTAL_AXES_PASSED` at the top of the page's section in `ui/app.js`; flip it when the checklist is done.

### How healing works

`models/<name>/heal.json` holds the state; three worker jobs run one after the other, each resumable, and `reshape::runs::heal` skips what is finished (so *Continue* after a stop, a failure
or closing the program goes on where it was):

1. `distill_cache` (when the teacher and the student do not fit the card together; `reshape::plan::heal_plan` decides): the original, loaded alone (16-bit, or 4-bit when it does not fit), writes its
   **top 32 next-token probabilities plus the leftover probability** for every token of the rows to `distill/cache/part-NNNNN.npz` (ids `uint32`, log-probabilities `float16`, leftover `float32`,
   196 bytes per token: 20 M tokens are 3.9 GB) and `index.json` (teacher, k, rows, data and tokenizer hashes, `done_rows`, `complete`). The rows themselves (token ids) are saved in `rows.npz`.
   The student job refuses a cache whose hashes or length differ ("The saved teacher answers belong to other data; they will be made again.").
2. `distill`: the thinned model (QLoRA from 15 B, else 16-bit LoRA, rank 64, learning rate 1e-4 with a cosine decay after 30 warm-up steps, 4 x 8 rows of up to 2048 tokens per step, checkpoints every 50 steps) learns
   `KL(teacher || student)` per token, `alpha = 1.0` (no cross-entropy term; Expert can lower it). Full logits are never held: the student's last hidden states go through the output layer in chunks
   of 256 tokens inside `torch.utils.checkpoint` (peak about 250 MB instead of 2 GB per row). The mode is `cache` (default when both do not fit) or `online` (teacher 4-bit and student in one process).
   Every 50 steps the student's text-prediction error on the measuring batches is computed and shown next to the original's.
3. `distill_merge`: `W' = W + (alpha / r) * B @ A` per adapted weight in numpy float32, rounded back to the tensor's own type (bf16 to nearest even), everything else copied as raw bytes; the folder
   gets `config.json`, the tokenizer files, `ndforge_reshape.json` (the thinned model's, with `healed: {...}`) and `width_kept.npz`. Only a plain LoRA is merged (no DoRA, no per-layer ranks).

The healed model is added to the model list as `custom-local-<name>-healed`; Step 3 trains on it, Step 4 compares original, thinner and healed (`compare()` rows `model:<name>` with method `slim`
(layers), `slim-width` or `slim-both`, and `model:<name>-healed` with method `healed`), Step 5 exports it with the original's vision file (the hidden size is unchanged).

### File formats (version 2)

- **Job `reshape`** (version 1 keys unchanged) plus `"width": {"intermediate_size": 10752, "linear_num_value_heads": null, "num_attention_heads": null}`, `"scores_dir": "<project>/models/_scores/<key>"`
  (`<key>` = `slug(source)-v<data version>-<samples>x<length>-<load>`), `"blocks": 0` allowed with width, `"ffn_divisor"` (default 256; the tests use 4 on a tiny model). **Result** adds `shape`
  (`depth` / `width` / `both`) and `width: {before, after, stable_overlap, scores_key, mtp}`; the event `width_scores` carries the mean MLP score of every layer, the share of the score that stays
  and the stability.
- **`models/_scores/<key>/`**: `scores.npz` (float32 `mlp [L, I]`, `attn [n_full, nh]`, `gdn [n_lin, nv]` and the same with `_h1` / `_h2` for the two halves of the text) and `scores.json`
  (`format`, `source`, `data_version`, `samples`, `max_len`, `load`, `tokens`, the block-boundary distance matrices, `loss_before`). One measurement serves every size on the same data; a measurement
  of other data, another model or other settings is not used (a log line says why). Job `reshape_measure` makes one on its own.
- **`ndforge_reshape.json` format 2** (width models) adds `shape` and `width: {intermediate_size: [before, after], linear_num_value_heads, num_attention_heads, kept: "width_kept.npz",
  scores_key, stable_overlap, mtp}`; `width_kept.npz` holds the kept indices of every layer of the new model (int32; `mlp_<j>`, `heads_<j>`, `gdn_<j>`, `src`, `mtp_*`), so the surgery is reproducible.
- **Jobs `distill_cache`, `distill`, `distill_merge`**: keys in the docstrings of `worker/ndforge_worker/distilljob.py`. Events: `distill_metric` (KL, cross-entropy, tokens), `distill_eval`
  (calibration loss) next to the usual `metric` / `progress` / `log`; every healing event also carries `phase: "heal"` and `heal_step` (`cache` / `train` / `merge`).
- **`models/<name>/`** gains `heal.json`, `heal_config.json`, `heal_result.json`, `heal_metrics.jsonl`, `width_scores.json`, `test_healed.json`, `distill/cache/`, `distill/run/`, `healed/`.

### Where the code is (version 2)

Worker: `reshape_width.py` (pure: dimensions, sizes, exact parameter count, picking, slicing rules, config rewrite), `slice_io.py` (numpy: slicing, bf16, weight scores, the LoRA merge),
`reshapejob.py` (`_measure_all`, `_scores_for`, `run_measure`, width in `run`), `distilljob.py` (math, data, cache, student, merge), `safetensors_io.py` (`OutTensor.slice` / `data`).
Rust: `reshape/plan.rs` (`Width`, `WidthDims`, `Shape`, `width_params`, `width_options`, `suggested_widths`, `heal_plan`), `reshape/runs.rs` (`measure`, `heal`, `HealSpec`, `heal_options`, `register_healed`,
`test_target`), `evaluate/runs.rs` (`compare`), window commands `reshape_options` (with `shape`), `reshape_measurement`, `reshape_measure`, `heal_options`, `heal_start`, `heal_stop`, `reshape_test` (with `target`).
Page: `ui/app.js` (smoke tests in `ui/smoke/reshape.test.mjs`).

### Deviations from the plan (version 2)

Each is the smallest change that made something work or kept something honest.

1. **Value heads for the 9B are 32 → 16, not 24.** The plan's table says 24, but its own rule (a multiple of the 16 key heads, one step) allows only 32 and 16.
2. **`plan_width` takes the old layer types and no `tensors` argument** (`plan_width(out, src, types, dims, width, keep, mtp_keep)`): the tensors come in through `out`, the types decide which rows a layer has.
   `weight_norm_scores` returns one entry per multi-token layer (`{layer: {"mlp", "heads"}}`).
3. **The student's hidden states are taken with a hook on the final norm, not on `lm_head`'s input.** `logits_to_keep=1` hands `lm_head` only the last position; the final norm's output still has all of
   them. The call still passes `logits_to_keep=1` so no full logits are made (an older transformers without it falls back to one full forward).
4. **Healing never uses Unsloth for now.** The plan tries it only when a `distill_selftest` passes; that test was not written (nothing here can run Unsloth), so the plain-libraries path is always used and the
   log says so. Checklist item 6 stays open.
5. **The rows are saved with the cache** (`rows.npz`), so the student learns from exactly the rows the teacher answered and the general text is downloaded once, not twice.
6. **No temperature setting** (fixed 1.0, which is also NVIDIA's default); the plan listed `temperature` among the keys.
7. **Offline general text falls back to the project's text with a note** instead of the plan's error message plus an Expert switch (owner decision 2).
8. **Compare rows:** plain layer removal keeps the version-1 label `slim` (the plan wrote `slim-depth`; version-1 tests and pages read `slim`); new labels `slim-width`, `slim-both`, `healed`.
9. **Healing events:** `distill_metric` and `distill_eval` are their own event types next to `metric` (which keeps the existing loss chart working) instead of extra fields on `metric`.
10. **`OutTensor.data` may be a function** that makes the bytes when the tensor is written, so the merge holds one tensor in memory at a time instead of the whole model.
11. **`reshape_options` returns `width_options`** (one group per number of removed blocks, one stop per MLP width, each with exact size, disk space and warnings) and a `width` block (support, reason,
    choices, suggested stops, heal budgets); `reshape_measurement` was added so the page can say "measured earlier: reused".
12. **`heal_plan` numbers:** teacher in 4-bit = 0.6 bytes per parameter, a step with the teacher's answers saved first needs `max(teacher, student) + 2 GB`, both at once `teacher + student + 4 GB`. A
    27B → 21B on a 16 GB card therefore says "about 19 GB" (the plan's example said 23).
13. **`ffn_divisor`** (default 256) is a job key only so that tests can use a tiny model (MLP 32, divisor 4).
14. **Existing tests were extended, not weakened** where the new fields demanded it: the inspect and result key sets (`INSPECT_KEYS`, `RESULT_KEYS`) have the new keys, the Rust `inspect()` helper uses
    `..Default::default()`, `fake_worker` takes the inspect answer as a parameter, `stfile.BYTES` knows `I64`.

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
Qwen3.5 engine accept 48 layers; 4-bit measuring ranks blocks like 16-bit; the quality after healing (QLoRA is fine for dense Qwen3.5, Unsloth's caution is for MoE; it still loses a little more than 16-bit); the
estimated sizes (a 27B slimmed by 4 blocks is about 21.6 B; the 9B grown by 4 blocks about 13.1 B; the page computes them from the real tensor shapes, the
plan's figures were estimates); measuring time and memory; whether a world's text gives any gain from growing (expected: no).

### Version 2 (thinner layers and healing): not verified yet

**Everything here was implemented as the plan says and has not been seen working on a real model.** Without torch, transformers and a GPU the following was tested with stand-ins and numpy (all green):
the dimension, size and parameter-count helpers (the Hugging Face totals of the 27B and 9B are reproduced exactly), picking structures, the slicing rules against hand-written index lists, byte-exact slicing
and writing, the whole `reshape` job with a width from a written measurement (output folder, config, `ndforge_reshape.json`, `width_kept.npz`, parameter count of the written shapes against the formula), the KL
math against an independent numpy version, the data order, the resumable teacher cache, the LoRA merge against `W + (alpha / r) * B @ A`, the Rust plan, the runs and the heal flow with fake workers, and the
page in a stand-in browser.

**Written but never executed here (they are skipped without torch; CI runs them on CPU and they have never run once; if one fails there, fix the test or the code, do not weaken it):**
- `worker/tests/test_reshape_width_torch.py`: thinned equals original-with-structures-switched-off (MLP; MLP + heads + value heads; each axis alone; depth + width), the measuring (finite, deterministic, cached,
  equal to a hand-computed aggregation), the job measuring by itself and reusing its measurement, the GGUF export with a real llama.cpp (skips without it).
- `worker/tests/test_distill_math.py` (torch part), `worker/tests/test_distill_torch.py`: the chunked loss equals the full-logits loss with equal gradients, the hidden states through the final norm
  reproduce the logits, the teacher cache against a direct forward pass, resume of the cache job, healing in cache mode lowers the KL, cache and online mode give the same first-step loss (k = whole vocabulary),
  stop and checkpoint, the numpy merge equals PEFT's own `merge_and_unload()`.
- The code those tests cover: `_measure_all`, `_measure_and_save`, `run_measure`, `forward_hidden`, `calib_loss`, `_teacher_answers`, `run_cache`, `run` of `distilljob.py`, the Trainer subclass
  (`compute_loss`, `model_accepts_loss_kwargs`, the sampler).
- The window in a real Tauri window (the page was tested in a stand-in browser; `cargo build -p nd-lore-forge` succeeds).

**Assumptions the plan marked [A] / [GPU] that remain open:** a hook on `Linear4bit` sees the bf16 activation; `Qwen3_5ForCausalLM.forward` accepts `logits_to_keep` in the transformers that ship (the code falls
back without it, at the price of memory); PEFT's `all-linear` covers the linear-attention projections and the key names of the saved adapter match (the merge test checks this on CPU); QLoRA adapters merged into
bf16 weights differ slightly from the 4-bit model they were trained on; a 4-bit teacher gives a slightly noisier target; llama.cpp quantises MLP rows of 13,312 or 10,752 values as k-quants (multiples of 256);
Unsloth's bundled llama.cpp and Ollama accept a non-standard MLP width; the multi-token head cut by weight size loads in llama.cpp and is useless for speculative decoding; the general dataset
(`nvidia/Nemotron-Pretraining-SFT-v1`: the right column is `text`, the licence suits the project) must be checked on the owner's PC; the time and memory estimates of the Heal card; and above all **the quality
after healing with 20 to 40 million tokens** (NVIDIA used 94 billion for its models: expect a usable model, not one as good as the original).

**The GPU checklist of version 2** (RTX 5090; record the numbers here):

| # | Check | Result |
|---|---|---|
| 1 | 9B: measure in 16-bit and in 4-bit: Jaccard overlap of the kept MLP sets at MLP 9,216 (0.9 or more expected), time and peak VRAM | not run |
| 2 | 27B: measure in 4-bit (256 samples x 512 tokens): time and peak VRAM; make MLP 13,312 and 10,752: time and disk; load them in transformers and in Unsloth; loss before and after | not run |
| 3 | Test page original against thinner (both 4-bit), then heal (20 M tokens, cache mode): cache time and size, student tokens per second, peak VRAM, the KL curve, the calibration loss against the teacher's; test original against healed; then Step 3 on the healed model and test | not run |
| 4 | The same size by layers only (48 layers), by width only (MLP 10,752) and both: Test-page scores and tokens per second side by side (the owner's comparison) | not run |
| 5 | Export: GGUF (Unsloth and llama.cpp) of the width model; Q4_K_M quantises all MLP tensors as k-quants; `llama-server` with the official mmproj answers an image question; Ollama install and *Test it* | not run |
| 6 | Healing with Unsloth: does a hook see the hidden states under its fused loss? (needs the self-test of deviation 4 first) | not run |
| 7 | Merge: loss of the merged healed folder (16-bit) against the adapter on the 4-bit student | not run |
| 8 | The Expert axes (value heads 48 to 32, attention heads 24 to 20 / 16) on the 9B: quality after healing against MLP-only at the same size; if good, set `EXPERIMENTAL_AXES_PASSED` to true | not run |

**Steps of the plan that were not done:** step 20 (the GPU run itself: this checklist); step 21 *(optional)* hidden-size pruning for a text-only export; step 22 *(optional)* an intermediate-state loss for
"Both" with three or more blocks removed.

## Follow-ups (optional steps 18-20 of the plan of version 1, and the open steps of version 2)

- **Distillation heal**: built in version 2 (see above; it is a job of its own, `distilljob.py`, not a KL term in `trainjob.py`).
- **Hidden-size pruning for a text-only export** (version 2, step 21): drops the vision tower, slices the embeddings, every norm and every projection and the multi-token head's `fc`; needs its own design.
- **An intermediate-state loss for "Both"** with three or more blocks removed (version 2, step 22), if the checklist shows that depth hurts more after healing.
- **Healing with Unsloth** behind a one-step self-test (version 2, deviation 4), for speed.
- **Switching on the Expert axes** once checklist item 8 passes.
- **A temperature setting** for the distillation loss (the teacher's answers are saved at 1.0).
- **MoE expert pruning (REAP)** for `qwen3_5_moe` bases through `FastLanguageModel.reap_prune`, when Unsloth is installed. The dataset format Core expects
  is not in the sparse clone and has to be read first.
- **Gemma 4**: slice `embed_tokens_per_layer` / `per_layer_model_projection` by layer and keep `num_kv_shared_layers` consistent. Refused with a reason until then.

A practical guide with recommendations for Qwen3.8 27B: `docs/PRUNE_QWEN38_27B.md`.
