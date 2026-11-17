# Importance matrix (imatrix) for the GGUF export

An optional step of Step 5 ("Better small sizes (importance matrix)"), and, since the matrix library, a function of its own (page "Importance matrices", see "The matrix library"). Built and tested with stand-in tools; **never run with a real llama.cpp or a GPU** (see "Not verified yet").

## What it is, and when it helps

When a model is shrunk to 4 or 5 bits, every weight is rounded. An **importance matrix** records, for a sample of text, how strongly each input of each layer is used. llama.cpp's
`llama-quantize --imatrix` then spends its precision where it matters, so **Q6_K, Q5_K_M and Q4_K_M lose less quality** than without it. It mainly helps at about 4 bits and
below; for **Q8_0, F16 and BF16 it does (almost) nothing**, so those sizes are never given the matrix.

The matrix is a measurement of the model on text, so it should look like what the model will be asked: the project's own training data, as the model sees it.

## How it is made

All in `worker/ndforge_worker/imatrix.py`, called from `exportjob.gguf_with_llama_cpp` after the f16 conversion (the llama.cpp way) or from `exportjob.matrix_for_unsloth` before Unsloth is loaded (the Unsloth way, see "Who makes the GGUF"):

1. **Calibration text** (`build_calibration`), about `ctx x chunks` tokens (+10 % slack):
   - about **80 %** the project's saved data version (the one the run trained on: `train.jsonl`): a practice conversation rendered with the **model's own chat template** (Qwen3.5 and Gemma 4 with thinking off, the family's end marker; special tokens such as `<|im_start|>` stay in the text), a world-book text as it is. Rows are shuffled with a fixed seed; data smaller than its share is repeated.
   - about **20 %** general text, through the healing code (`distilljob._general_texts`, default dataset `distilljob.DEFAULT_GENERAL` = `mlabonne/open-perfectblend`, or the dataset chosen in the box under the tick; chat rows go through the model's chat template; streamed, never downloaded whole; see docs/GENERAL_TEXT.md; a long text counts for at most two chunks).
   - both kinds are mixed evenly through the file, so `--chunks` never cuts one kind away. The same data and settings always give the same file.
   - The tokenizer is loaded from the merged model folder (so it carries the patched, non-thinking template), then from the base model.
2. **llama-imatrix** (the tool is looked up next to `llama-quantize`, then on `PATH`):

   ```
   llama-imatrix -m model-f16.gguf -f imatrix-calibration.txt -o imatrix.partial.gguf --output-format gguf --no-ppl --chunks N --parse-special -ngl 99 -c 512
   ```

   `--output-format gguf` is dropped and the call repeated if this llama-imatrix does not know it (older builds). `--process-output` is not used; `--nextn` only as described under "The speculative-decoding (MTP) head". `-ngl 99` offloads to the GPU and is harmless on CPU-only builds. The file is renamed to `imatrix.gguf` only when the tool succeeded.
3. **llama-quantize** for every size below Q8_0: `llama-quantize --imatrix imatrix.gguf model-f16.gguf model-<size>.gguf <SIZE>`. Q8_0 is made as before; F16 is the converter's own file.

The matrix is made **once per export** (all chosen sizes share it).

## Who makes the GGUF (Unsloth or llama.cpp)

The matrix is always **measured by llama.cpp's `llama-imatrix`** (Unsloth has no measuring step, see "What Unsloth's `save_pretrained_gguf` does"). What differs is who rounds the sizes:

| Back end | Build can apply a matrix | llama.cpp + `llama-imatrix` found | Result (`via`) |
|---|---|---|---|
| Unsloth | yes | yes | llama.cpp measures, **Unsloth** makes the GGUF with `save_pretrained_gguf(..., imatrix_file=<path>)`: `via: "unsloth"` |
| Unsloth | no | yes | the GGUF is made the llama.cpp way (merge with Transformers + PEFT, convert, `llama-imatrix`, `llama-quantize`): `via: "llama.cpp"`, as before this change |
| Unsloth | any | no | Unsloth without a matrix and a note: `via: "unsloth"` |
| other | - | - | the llama.cpp way: `via: "llama.cpp"` (needs llama.cpp anyway) |
| Unsloth, only Q8_0 / F16 chosen | any | any | nothing to improve: Unsloth as before, and a note |

The decision is made by `exportjob.plan_gguf`, logged, and recorded: `via` is in the worker's result and `export.json` (every GGUF export, matrix or not), in `meta.json` (`via`, and `imatrix.via`), and the files list of the export page says "GGUF made by Unsloth / llama.cpp" for exports that asked for a matrix.

**Can the build apply a matrix?** `exportjob.unsloth_can_apply_imatrix(save_fn, has_resolver=None)` mirrors Unsloth Studio's `_imatrix_export_supported` (Studio: `studio/backend/core/export/export.py`): true when `save_pretrained_gguf` has a parameter named `imatrix_file` (positional-only does not count), or takes `**kwargs` **and** `unsloth_zoo.llama_cpp.resolve_imatrix_file` can be imported (a bare `**kwargs` may be a function that filters the keyword away). One difference, on purpose and stricter: a function from `unsloth_zoo.cpu*` is refused, because that loader takes `**kwargs` and silently drops `imatrix_file` (version 2026.11.4, read in the source). The function is looked at **before** the model is loaded (`unsloth.save.unsloth_save_pretrained_gguf`, the one Unsloth puts on every model), so the matrix can be measured while no model is in memory; after loading, the model's own `save_pretrained_gguf` is checked again (a mismatch is a note: Unsloth without the matrix).

**The flow when Unsloth applies it** (`exportjob.matrix_for_unsloth`, then `run`):

1. The matrix is reused if a saved one fits (nothing is merged or converted then). Otherwise the 16-bit GGUF to measure on is `model-f16.gguf` of the export folder when it is newer than the weights, else it is made now: merge with Transformers + PEFT into `merged/` (a full fine-tune is used as it is), then llama.cpp's converter into `imatrix-f16.gguf`. Both are removed again afterwards (Unsloth merges by itself; with *keep merged* it saves the merged model as before).
2. `llama-imatrix` measures (with `--nextn` when it applies), saved as `imatrix.gguf` + `imatrix.json` as always.
3. Unsloth is loaded and `save_pretrained_gguf(tmp, tokenizer, quantization_method=<sizes>, imatrix_file="<export folder>/imatrix.gguf")` is called **once for the whole batch**. Vision file and *keep merged* are handled exactly as before.
4. Report: `applied` = the sizes below Q8_0, `not_applied` = Q8_0 / F16 ("high precision").

**Fallbacks** (never an error): the measuring fails or llama.cpp cannot be found: Unsloth without the matrix and a note; the merge or conversion for the measuring fails (for example out of memory): a note, Unsloth without the matrix; Unsloth raises an error that mentions the imatrix: one more call without it and a note ("Unsloth did not accept the importance matrix ..."); any other Unsloth error is not hidden (it would fail without a matrix too). Only the Stop button ends the export early, as before.

## What Unsloth's `save_pretrained_gguf` does (read, not run)

Read in the published wheels **unsloth 2026.11.4** and **unsloth_zoo 2026.11.4** (PyPI, pure Python; unpacked into an empty folder and only read, never imported or executed):

- `unsloth.save.unsloth_save_pretrained_gguf(self, save_directory, tokenizer=None, quantization_method="fast_quantized", first_conversion=None, push_to_hub=False, token=None, private=None, is_main_process=True, state_dict=None, save_function=torch.save, max_shard_size="5GB", safe_serialization=True, variant=None, save_peft_format=True, tags=None, temporary_location="_unsloth_temporary_saved_buffers", maximum_memory_usage=0.85, save_method=None, imatrix_file=None, merge_is_disposable=True)`.
- `imatrix_file`: `None`/`False` = off; a **path** = that file (a `*.gguf_file` is renamed to `*.gguf`; `.dat` and `.gguf` both named in its messages); `True` = download Unsloth's own published matrix (`imatrix_unsloth.dat` / `imatrix_unsloth.gguf_file`) from `unsloth/<base>-GGUF` on the Hub. It is resolved **before** the merge, so a bad path or an unavailable upstream matrix fails early.
- **Unsloth does not measure.** It never runs `llama-imatrix`; it only passes `--imatrix <file>` to its own `llama-quantize` (`unsloth_zoo.llama_cpp.quantize_gguf(..., imatrix=...)`, which checks that the file exists). `save_to_gguf` refuses early (`RuntimeError`, "your installed unsloth_zoo's quantize_gguf does not support imatrix") when the installed zoo's `quantize_gguf` has no `imatrix` parameter. So **`llama-imatrix` is not needed in Unsloth's own llama.cpp build, but the matrix must come from somewhere**: here, from the user's llama.cpp.
- The matrix is given to **every size in the call**, Q8_0 included (llama.cpp's `llama-quantize` has nothing to do with a matrix for Q8_0). The IQ low-bit sizes are only valid with one; Lore Forge does not offer them.
- `unsloth_zoo.llama_cpp.resolve_imatrix_file(imatrix_file, dest_dir, repo_candidates=(), token=None)` and `quant_requires_imatrix` exist in the zoo. The zoo's **CPU** loader (`unsloth_zoo/cpu/loader.py`) has `save_pretrained_gguf(self, save_directory, tokenizer=None, quantization_method="q8_0", **kwargs)` and ignores `imatrix_file`; the MLX one has an explicit parameter.
- **`imatrix_file=True` is not offered** ("Use Unsloth's published matrix"). The source resolves it by the base model's name to `unsloth/<name>-GGUF`; it is a matrix measured on the **unmodified** base model, needs the network and, for a LoRA or a full fine-tune made here, is not the model being quantised (and for a base made by "Slim or grow a model" it would not fit at all). The data-driven matrix measured on the trained model is the point of this feature; it was not worth the risk of a quietly wrong matrix.

## Files

Per export folder (`exports/<run>-<size>/`), next to the GGUFs, in **every** export of one batch (so a later export finds it whichever size was kept; it is small):

- `imatrix.gguf` - the matrix.
- `imatrix.json` - what it was made from: `model` (run name), `weights` (a fingerprint of the run's safetensors files, so the same run trained further does not match), `data_hash` (sha256 of `train.jsonl`), `data_version`, `chunks`, `ctx`, `general_share`, `general_used`, `general_source`, `nextn`, `nextn_asked`, `nextn_note`, `project_tokens`, `general_tokens`, `llama_cpp` (the version line of `llama-imatrix --version`, when obtainable), `base_repo`, `created`.
- `meta.json` gets `imatrix`: `null` when it was not asked for, else `requested`, `applied` (this size used it), `made`, `reused`, `file`, `chunks`, `ctx`, `general_used`, `llama_cpp`, `reason` (why this size did not use it), `note`, `nextn` (the head is in the matrix), `nextn_asked`, `via` (`"unsloth"` / `"llama.cpp"`: who made the GGUF); `meta.json` itself has `via` too.
- `export.json` (the worker's summary) has the whole `imatrix` report and `via`.

**Reuse.** A later export of the same run reuses a saved matrix instead of measuring again when `model`, `weights`, `data_hash`, `chunks`, `ctx`, `general_share`, `general` (the dataset: id or file, config, split, column, share, seed; a matrix saved before this key existed counts as made from other data) and `nextn` (see below) are all equal (the checkbox "Reuse the saved matrix" is on whenever one exists; untick it to measure again). A matrix that was made without general text (offline at the time) is **not** reused while general text is wanted. The page checks only that a matrix of the run exists; the worker does the exact comparison.

## Defaults (owner decisions)

| | |
|---|---|
| Off by default | yes; the checkbox is shown only for GGUF exports |
| Amount of text | 64 / **128** / 256 chunks; context 512 tokens |
| Share | about 80 % project data, 20 % general text |
| Sizes that get it | every size below Q8_0 that the exporter offers (Q6_K, Q5_K_M, Q4_K_M); not Q8_0, F16, BF16 |
| Vision (mmproj) file | unaffected |
| Stop | the Cancel button on the page writes a `STOP` file into the export folder; the worker kills `llama-imatrix` at once, removes the half-made files and ends with "Cancelled."; the program kills the worker if it has not ended 20 s later. Without the matrix, Cancel kills the worker at once, as before |
| Progress | the existing events, step `imatrix` (chunk markers `[12]` of llama-imatrix when it prints them, else a heartbeat with the elapsed time); the last 40 lines of the tool's output go to the log |
| MTP head (`--nextn`) | on by default **for a model that has a head** and a llama-imatrix that knows `--nextn`; see below |
| Who makes the GGUF | Unsloth when the build can apply a matrix, else llama.cpp; see above |

## The speculative-decoding (MTP) head (`--nextn`)

A model cut in width has no head any more (the cut removes it: `docs/RESHAPE.md`), so none is offered for it; a depth-only cut keeps its head.

llama.cpp's `tools/imatrix` README: `--nextn` "collects data for MTP/NextN layers if present. Needs a single sequence per batch (i.e. set `--batch-size` <= `--ctx-size`). Cannot be used on models where NextN layers share the trunk's KV cache." (`--model-draft FNAME` supplies an external draft file, only for a model without NextN layers; **not used**.)

Owner decisions (defaults) as built:

1. **Does the model have a head?** `imatrix.model_has_mtp`: the converted 16-bit GGUF when readable (`gguf_has_mtp`: a key ending `nextn_predict_layers` above 0, or a tensor named `...nextn...` / `mtp...`; only the header is read), else a model folder (the merged one, the full model, the base model as a folder or in the Hugging Face cache): `config.json` keys `mtp_num_hidden_layers` / `num_nextn_predict_layers` (also inside `text_config`) or tensor names starting `mtp.` (`model.mtp.`) from `model.safetensors.index.json` or the safetensors headers (`folder_has_mtp`). The GGUF has priority because it is what llama-imatrix reads: a converter that dropped the head means there is nothing to measure. When nothing can be read, the head is treated as absent and a note says "it could not be told whether the model has one" (through Unsloth the model is merged only when the 16-bit file is needed, and if the base model is on no disk the worker makes that file first to find out). The Rust side (`export::mtp_head_in`, same rules, plus the cache) tells the page per run.
2. **Does llama-imatrix support it?** `llama-imatrix --help` is run **once per matrix** (only when the option is asked for and the model has a head) and searched for `--nextn`.
3. **The option** "Include the speculative-decoding head (MTP) in the matrix" (`imatrix.nextn` in the job, default on): shown on the page only for a model that has a head. When on and possible, the call gets `--nextn` and `-b <ctx>` next to `-c <ctx>`. If llama-imatrix exits non-zero and its output mentions `nextn`, `KV cache` or `mtp`, it is called **once more** without `--nextn` and `-b`, and the result says "The MTP head was left out: <that line>." Any other failure is the ordinary "llama-imatrix failed" (no retry; the sizes are made without a matrix).
4. **Reuse key**: `nextn` is part of it, **exact match**: a matrix made without the head is not reused when the head is asked for (and possible), and the reverse is not reused either. `imatrix.json` has `nextn` (what the file really contains) and `nextn_asked` (what was asked and possible when it was made: a matrix whose head was refused still counts as the answer for the same request, so it is not measured again and again); a file saved before this key counts as made without the head.
5. Never fails an export; Stop works as before (it also ends a `--nextn` run at once).

## When something is wrong: never fatal

The export is never failed by the matrix step. Every problem is a plain-language **note** (shown on the page, in `meta.json` and in the log) and the sizes are made without it:

| Situation | Note (shortened) |
|---|---|
| llama.cpp has no `llama-imatrix` | "This llama.cpp has no llama-imatrix (an older install), so the sizes were made without the importance matrix. Update llama.cpp ..." |
| Unsloth back end and no llama.cpp | "An importance matrix was asked for, but llama.cpp (with llama-imatrix) was not found, so the GGUF was made through Unsloth without it. ..." |
| Unsloth cannot take the matrix after all, or refuses it | "The loaded Unsloth could not take the importance matrix after all, ..." / "Unsloth did not accept the importance matrix (...), so the GGUF was made through Unsloth without it." |
| the MTP head cannot be measured | "The MTP head was left out: <reason>." (no head in the model / this llama-imatrix has no --nextn option (an older build) / it could not be told whether the model has one / the line llama-imatrix printed) |
| no size below Q8_0 chosen | "The importance matrix helps the sizes below Q8_0; none of them was chosen, so it was not made." |
| general text cannot be read (offline, gated, not found, no usable column, nothing read, no `datasets`) | "The general text (...) could not be read: <plain reason>. The matrix is made from your project's text only." (the matrix is still made; docs/GENERAL_TEXT.md lists the reasons) |
| `llama-imatrix` fails | "llama-imatrix failed (exit code N; see the log), so the sizes were made without the importance matrix." |
| `llama-quantize` refuses `--imatrix` | "llama-quantize did not accept the importance matrix for Q4_K_M (exit code N); that size was made without it." |
| training file or tokenizer missing, any other error | "The importance matrix could not be made (...), so the sizes were made without it." |

Only the Stop button ends the export early.

## Tests

- Worker: `worker/tests/test_imatrix_nextn.py` (head detection from config keys, safetensors names and index, GGUF header and the Hub cache; the exact command lines with and without `--nextn`, `-b`/`-c`, the `--help` probe asked once, the refusal retried once, the other failure not retried, the reuse key, a saved matrix of before the key) and `worker/tests/test_imatrix_unsloth.py` (the feature detection with stand-in save functions: `imatrix_file` parameter, **kwargs only with and without the zoo resolver, none, positional-only, the zoo CPU loader; the decision table; whole exports through a stand-in `unsloth` module: one call with `imatrix_file`, measured before loading, reuse without merging, an f16 already in the folder, full model, keep merged, cannot-apply goes the llama.cpp way, no llama.cpp, a refusal retried without the matrix, a stop), and `worker/tests/test_imatrix.py` (stand-in `llama-imatrix`, `llama-quantize` and convert scripts written as small Python files and started through the current interpreter, a stand-in tokenizer; no torch, llama.cpp or network; the same on Windows and Linux): exact command lines (`--imatrix` only for the right sizes, `--parse-special`, `-c`, `--chunks`, `-o`), reuse and every reason not to reuse, each fallback, the stop file (a tool that would run 30 s is killed at once), the calibration text (chat template, special tokens, 20 % general text mixed in, deterministic, repeats small data), which back end makes the GGUF, and a whole export job.
- Rust: `crates/ndforge-core/tests/export.rs` (options default off and are checked, the job block, saved matrices and time guesses, the matrix and its json kept in every export of a batch, meta, graceful Stop with a fake worker) and `export/runs.rs` (STOP file first, kill after the grace time).
- Window: `ui/smoke/export.test.mjs` (off by default, 64/128/256, reuse box, only for GGUF, options sent, the files list, Cancel, the head option shown only with a head and greyed with a reason when unsupported, the explanation of who makes the files, "GGUF made by").

## The matrix library

A matrix can also be made **on its own**, without exporting any GGUF: page **Importance matrices** (main navigation, next to "Slim or grow a model", and a link from Step 5). The matrices are kept in the project, used again by exports and copied out. Built and tested with stand-in tools only (see "Not verified yet").

### Owner decisions (defaults) as built

| | |
|---|---|
| Where | `<project>/imatrix/<id>/`, id = `yyyymmdd-hhmm-<short model name>` (local time, a number is added if the name is taken). Old projects have no such folder: it is made with the first matrix |
| Files | `imatrix.gguf` (what llama.cpp reads), optional `imatrix.dat` (legacy), `imatrix.json` |
| `imatrix.json` | the export's fields (`format`, `model`, `weights`, `data_hash`, `chunks`, `ctx`, `general_share`, `general`, `nextn`, `nextn_asked`, `data_version`, `base_repo`, `general_used`, `general_source`, `general_error`, `project_tokens`, `general_tokens`, `nextn_note`, `llama_cpp`, `created`) plus `id`, `label` (at most 80 characters), `source` (what it was made from, in words) and `source_kind` (`run` / `model` / `gguf` / `imported` / `export`), `calibration` / `calibration_kind` (`data` / `text`), `run`, `size` (bytes), `arch`, `layers`, `entries` (read from the matrix header when it can be), `via` (`llama.cpp` / `imported`; for a matrix added from an export, what that export says), `has_dat`, `notes`, `statistics` |
| Matrices saved next to past exports | **not moved, not changed, not migrated.** "Add to library" (page, list "or from an earlier export") copies one in (`copied_from` names the export) |
| Model sources | (a) **a trained run**: its adapter merged into its base with the export's own `merge_hf` (a full fine-tune is used as it is), then converted with `convert_hf_to_gguf.py --outtype f16`; **an f16 GGUF of an export of that run is used instead** when it is newer than the weights (`model-f16.gguf` in an export folder whose `meta.json` names the run); (b) **a model**: an entry of the model list (also made, slimmed or healed models: they are list entries), a typed folder, or a Hugging Face id (the local cache first, else `snapshot_download`; the token only through `HF_TOKEN`), converted exactly as in (a); (c) **a GGUF file** (F16 / BF16 / Q8_0 recommended, any GGUF `llama-imatrix` accepts; the page says a more precise file gives a better matrix) |
| Calibration | the project's latest saved data version through the model's chat template (about 80 %) plus general text (the shared chooser, scope `imatrix`; default `mlabonne/open-perfectblend`, share default 20 %, 0 to 50 % allowed), or **a text file of the person's own**, used as it is (no general text, no template; `--parse-special` only when it contains template tokens such as `<|im_start|>`) |
| Options | chunks 64 / **128** / 256, context 512, `-ngl 99`, `--parse-special` (data), the `--nextn` option as in the export (probe with `--help` once, `-b` = ctx, one retry without when llama-imatrix refuses, a note), output format `gguf`, **also make a legacy .dat** (off), a label |
| Statistics | `llama-imatrix -m <model> --in-file imatrix.gguf --show-statistics` after the measuring, parsed leniently (a header line with tabs and `Min` / `Max` / `μ` columns and a `(N tensors)` line) into a few lines (tensors measured, largest value, mean of the means, share of active weights); skipped without a word when the build has no such option or nothing sensible is found |
| Legacy `.dat` | the README of llama.cpp's `tools/imatrix` documents `llama-imatrix --in-file x.gguf --output-format dat -o y.dat`; that is tried first, then the same with only the file extension (`-o y.dat`), then a note ("The legacy .dat file could not be made ...") and the matrix is still kept. "Make a .dat (legacy)" on an entry runs the same conversion as its own job (`imatrix_dat`) |
| Stop | the STOP file, as in the export: the tool is killed at once (also during the model conversion, which runs through the same streaming runner), the half-made folder is removed, the job ends with "Cancelled." (`imatrix_stopped`); the program kills the worker 20 s later if it has not ended. Merging a run (Transformers + PEFT) cannot be interrupted: the STOP file is looked at after it |
| Progress | step `imatrix`, one bar that never goes back: preparing / merging, converting (0.02 to 0.2), general text, measuring (0.2 to 0.95, the chunk markers), statistics, .dat; the last 40 lines of every tool's output go to the log |
| Failures | always a `MatrixError` with plain words (`imatrix_failed`): llama.cpp not found (error code `llama_cpp_missing`; says where to get it, that the program can install it itself, and the real search order: `LLAMA_CPP_DIR`, the folder the program installed into (`NDFORGE_LLAMA_CPP_DIR`), `~/llama.cpp`, a `llama.cpp` folder next to the program; `llama-imatrix` also on the PATH, the converter never; see docs/LLAMA_CPP_INSTALL.md), no `convert_hf_to_gguf.py`, model not convertible, not enough free space (checked before converting: the size of the model's safetensors), disk full while writing, a Hugging Face model that is gated / not found / unreachable, calibration unreadable (text file missing or empty, `train.jsonl` missing or unreadable, no text in the data, tokenizer unreadable), llama-imatrix failed (with the usual causes), a GGUF with saved data but no model to take the chat template from. Any other exception is turned into "The matrix could not be made (...)" |
| Result | `id`, `dir`, `file`, `dat`, `size_mb`, `info` (the json), `notes`, `statistics` |

### Using a matrix in an export

The export box (Step 5, after ticking *Better small sizes*) has **Use a saved matrix**: a list of the library, **Add a matrix file from my computer...** (a `.gguf` or `.dat`, with a label; copied into the library and selected) and **Open the matrix library**. Choosing one hides the measuring options. The job gets `imatrix.use_file` (the library file) and `imatrix.use_id`; the worker's `imatrix.use_saved_file` copies it into the export folder as `imatrix.gguf` (or `imatrix.dat`) with its `imatrix.json` (`from_library` added) and returns it from `prepare` exactly like a freshly measured matrix, so llama-quantize and Unsloth receive it through the same code. Nothing is measured, so llama.cpp (llama-imatrix) is not needed to apply it through Unsloth. The report says `reused`, `from_library`; `meta.json` has `imatrix.library`.

**Light compatibility check** (never a refusal; yellow warning on the page, `Warning:` note in the worker): the header of the matrix is read (`imatrix.read_gguf_header`; the minimal GGUF header reader that `gguf_has_mtp` now uses too; the Rust twin is `export/gguf_meta.rs`) for `general.architecture` (when present), the layers (highest `blk.N.` plus one) and the entries, and compared with the model: from the 16-bit GGUF (`<arch>.block_count`) when the worker has one, else from a model folder's `config.json` (`model_type`, `num_hidden_layers`, also inside `text_config`; the MTP layers count as allowed extra layers). A different architecture or layer count is a warning; a file that cannot be read (a legacy `.dat`, an unusual file) is a warning; a model whose layers cannot be read is a warning on the page; `llama-quantize` decides.

While an export uses a library matrix (and while a `.dat` is being made) the matrix is **in use** and cannot be deleted (a counter in the program, `imatrix_lib::mark_in_use`).

### Actions on an entry

- **Save a copy...**: `imatrix.gguf`, `imatrix.dat` if present, and `imatrix.json` as `imatrix-info.json`, into a **new folder `<chosen folder>/<id>`** (`<id>-2`, `-3` ... when it exists; nothing is overwritten). The free space is checked first (the function is injected in the tests); a failed copy removes what it made.
- **Open folder**: the system's own file manager (`explorer` / `open` / `xdg-open`); the program had no such command, so `imatrix_open_folder` was added.
- **Make a .dat (legacy)** when missing; **Delete** with a question, refused while in use. Importing refuses names that are not `.gguf` / `.dat`, empty files and too little free space; a file that is not readable as a GGUF is still added with a warning.

### Tauri commands

`imatrix_list` (state of the card), `imatrix_create` (events on `imatrix-event`), `imatrix_stop`, `imatrix_import` (`file` or `export`, `label`), `imatrix_save_copy`, `imatrix_open_folder`, `imatrix_make_dat`, `imatrix_delete`, `imatrix_check`. The export command `export_start` takes `imatrix.use_saved` (a library id; old specs without it load). Job types `imatrix_make` and `imatrix_dat` are new; existing job types and events are unchanged. The Hugging Face token reaches the worker only as `HF_TOKEN` (`job_env`), never in a job file or a log.

### Tests of the library

- Worker `worker/tests/test_imatrix_library.py`: both the export and the new job call `imatrix.make_matrix`; every model source (run: merge, f16 of an earlier export reused or too old, full fine-tune; folder; Hub id stand-in and the Hub error texts; GGUF), the command lines (`-m`, `-f`, `-o`, `--chunks`, `-c`, `--parse-special`, `-ngl`, `--nextn` and its retry, `--output-format dat`, `--in-file`), the folder contents and json fields, own text (as it is, `--parse-special` only with template tokens), progress never going back, Stop (a hanging tool is killed, no folder left), every failure text, statistics present and absent and odd, `.dat` fallbacks and "make a .dat later", the GGUF header reader and the check with hand-built headers, and a saved matrix in an export (llama-quantize gets the file, a `.dat`, warnings, a vanished file, only Q8_0, through Unsloth without llama.cpp). The existing export tests are unchanged and green.
- Rust `crates/ndforge-core/tests/imatrix_library.rs` and `export/gguf_meta.rs`: list (old project, odd folders), import (file, `.dat`, unreadable, nonsense, free space), from an export, save a copy (numbered names, free space injected, a file instead of a folder), delete and refuse while in use, the check, spec serde (old export specs), the job for each source (and the data, text, share and chunk refusals), the token only in the environment, the export block, and fake-worker runs (unix only: a whole job, Stop, a failure, a `.dat`, an export with a saved matrix).
- Window `ui/smoke/imatrix.test.mjs`: the card in the navigation, the explanation, the list and its actions, copy / delete / open / `.dat`, the three sources, the chooser, create / progress / Stop / result with statistics, add from a file and from an export, and in the export box: the list, the check and its yellow warning, `use_saved` sent only when chosen.

### Deviations

- **Legacy `.dat` command**: the brief named `--in-file x.gguf -o y.dat`; the README of llama.cpp (read for this change) documents `--in-file x.gguf --output-format dat -o y.dat`. That is tried first, the brief's form is the fallback.
- **Open folder**: the program had no opener, so a small command `imatrix_open_folder` starts the system's file manager (not in the brief's list of names).
- **Save a copy** writes into a new numbered folder named after the matrix, not straight into the chosen folder (all matrices are called `imatrix.gguf`; this is the only way never to overwrite).
- **A GGUF source with the saved data** needs the model it was made from (folder or Hugging Face id) for the chat template; without it the job says so and offers the own-text calibration. A typed folder or Hub id that is not in the model list uses the chat family `generic`.
- **The compatibility check lives twice**: in the worker (used at export time, notes) and in Rust (the page, before the export; a small second GGUF header reader, `export/gguf_meta.rs`), with the same texts.
- `ImatrixOptions` got `use_saved`; one existing struct literal in `crates/ndforge-core/tests/export.rs` needed `use_saved: None` to compile (no assertion changed).
- **In use** is a counter inside the running program, not a lock on disk: a second program on the same project would not see it.
- **No label editing** after a matrix is made (the label is asked when it is created or added).
- The time guess is the export's formula; the page adds "plus the time to convert the model" for a model source.

## Merging matrices

In the card **Importance matrices**: tick two or more entries (and/or add matrix files, and/or use **Unsloth's published matrix for a base model**), press **Merge**. A new worker job (`imatrix_merge`, new; the existing jobs are untouched) runs llama.cpp and registers a new library entry. Built and tested with stand-in tools and a stand-in Hub; **never run with a real llama.cpp or the real Hub** (see "Not verified yet").

### The command, from llama.cpp's README and source

```
llama-imatrix --in-file a.gguf --in-file b.gguf -o imatrix.partial.gguf --output-format gguf
```

`tools/imatrix/README.md` documents exactly this ("combine existing imatrices": `--in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf -o imatrix-combined.gguf`). **`-m` and `-f` are not needed with `--in-file`**: `imatrix.cpp` loads every `--in-file` and, with no prompt, prints "No prompt provided; combining precomputed matrices only." and saves (the README's usage line calls `-m`/`-f` mandatory, its own combine example has neither). A legacy `.dat` is read too (`--in-file` accepts both formats). `--output-format gguf` is passed first and dropped when this build does not know it (as for the measuring). **What it does:** `load_imatrix` **adds** the sums (`<tensor>.in_sum2`) and the counts (`<tensor>.counts`) of every input, so the merged matrix is the statistics of all the text, and an input measured on more chunks counts for more. That is why the page says, before and after: *a large published matrix can outweigh a small one of your own* (the result lists each parent's chunks and its share of the total; a note is added when one parent carries more than three quarters). The merged file's `imatrix.chunk_count` is whatever llama-imatrix writes; the entry's `chunks` is the sum of the parents' (`imatrix.chunk_count` read from their headers, or the library's own figure).

### Owner decisions, as built

| | |
|---|---|
| Inputs | entries of the library (a `.dat`-only entry is merged through its `.dat`), matrix files (`.gguf` / `.dat`, by path or Browse), and Unsloth's published matrix for a base model; at least two, the same file twice is refused (it would count double) |
| The result | `imatrix/<id>/imatrix.gguf` + `imatrix.json` with `source_kind: "merged"`, `source: "a merge of <labels>"`, `via: "llama.cpp"`, `parents` (kind, label, library id, path, SHA-256, size, chunks; for the published one also `repo`, `file`, `base`, `url`), `merge_weights` (each parent's chunks and share), `chunks` (the sum when all are known), `notes` (the warnings, each starting "Warning:"), `llama_cpp`. Like every entry: usable in an export ("Use a saved matrix"), in Compare quants, saved as a copy, deleted. The parents cannot be deleted while the merge runs (the "in use" counter) |
| Published matrix | repo and file name **as Unsloth's own code derives them** (see below), downloaded with plain HTTPS (`urllib`, `HF_ENDPOINT` for a mirror and for the tests) into `<project>/imatrix_downloads/<owner>--<repo>/imatrix_unsloth.gguf` (**not** inside the library, so it is not a library entry), once: a second merge finds the file with the size the Hub states and downloads nothing. The page's **Look it up** (`imatrix_published`) shows repo, file, **size** and the page address and says explicitly "This merge downloads N MB from huggingface.co ... once; nothing else is downloaded" before anything is fetched |
| The token | only through the environment variable `HF_TOKEN` (set by the program from the token box, as in the other jobs); it is sent to the Hub's host only, **never to the download server a redirect leads to** (redirects are followed by hand), and scrubbed from every log line and error |
| Compatibility | from the GGUF headers (the existing reader): architecture, layer count, and the tensors that only some of the inputs have; a mismatch is a yellow warning and the merge still runs; a file that cannot be read (a legacy `.dat`) gets a warning that it could not be checked. llama-imatrix's own refusal is quoted in plain words ("refused to combine them: its message is ... The matrices were probably made from different models", "could not combine the matrices ... If one of the files is not an importance matrix, take it out") |
| Weights | each input may have a weight (a number above 0, default 1; not sent when it is 1, so an old merge is the same job as before): the worker writes a COPY of that matrix with `in_sum2` and `counts` multiplied by it (in a work folder of the new entry, removed afterwards; the parent is untouched) and llama-imatrix reads the copies. The mean square per channel is unchanged, the matrix just counts that many times as strongly. The shares in the result are computed from the rows of text (`counts`) when every parent has them, else from the chunks; the result lists each parent's rows, weight and share (`merge_weights`: `chunks`, `tokens`, `weight`, `share`). A weight alone needs real data in the file (a header-only stand-in cannot be scaled) |
| A model cut on the Slim page | the published matrix describes the ORIGINAL model. It is **never merged raw** with a matrix of a model that has other layers or another MLP width: when a library parent was measured on a cut model (its record has `cut_model`, written when a matrix is made from a model folder, a run on one, or a GGUF whose model folder is one), the published input gets `adapt_to` that model and is adapted first (see "Adapting a matrix to a cut model"; the result says `adapted` in the parent); otherwise, when the worker finds a published matrix and another parent with different layers or a different `ffn_down` width, the merge is refused before llama-imatrix starts (`adapt_cut_model`) with a plain text that offers the Adapt card. A published matrix and a matching matrix are merged raw as before. A cut model that cannot be adapted (head axes, grown) refuses the merge with that reason |
| Honest notes | always, when the published matrix is a parent: "describes the UNMODIFIED base model, not your fine-tune: a reasonable start for a light LoRA, less so for a heavily changed model" |
| Create form | the tick **Also merge Unsloth's published matrix for the base model** (shown only when the model has a base on the Hub) makes a second, merged entry after the creation; the first is kept; if the base is a folder, the Hub is unreachable or there is no published matrix, the matrix you made is kept and the reason is shown next to it (`merge_error`). **For a model made on the Slim page** (a run on it, or the model in the list) the base is the original named in its record (`origin.repo`) and the published matrix is adapted to the cut model first, never merged raw (the page says so under the tick) |
| Stop | the STOP file as for the other jobs: a download is ended at once (the `.part` file removed), `llama-imatrix` is killed, the half-made folder removed, "Cancelled."; the program kills the worker 20 s later if it has not ended |

### Which repository? (read, not guessed)

`unsloth/save.py` `_gguf_repo_candidates(model)` (unsloth 2026.11.4): for the model's base name (a local folder has none), after `get_model_name(name, load_in_4bit=False)`, `repo = name if name.endswith("-GGUF") else f"unsloth/{name.split('/')[-1]}-GGUF"`. `unsloth_zoo.llama_cpp.resolve_imatrix_file` lists the repo's files and takes the first of `imatrix_unsloth.dat`, `imatrix_unsloth.gguf_file`, `imatrix_unsloth.gguf` that exists; a `.gguf_file` is renamed `.gguf` locally ("so the Hub does not list it as a model"). `imatrixmerge.published_repo` and `UPSTREAM_NAMES` do the same. Of `get_model_name` only the well-known suffixes `-unsloth-bnb-4bit` and `-bnb-4bit` are undone here (its full table is Unsloth's and cannot be read without running it). For example `Qwen/Qwen3.8-27B` gives `unsloth/Qwen3.8-27B-GGUF`, which holds `imatrix_unsloth.gguf` (reported as present and not gated by the person who asked for this feature, who checked the Hub API; this program has never run the lookup against the real Hub).

### Tests of merging

- Worker `worker/tests/test_imatrix_merge.py` (a stand-in Hub as a local HTTP server, a stand-in `llama-imatrix`): the repo derivation table and the file-name order; the lookup (repo, file, size, page, gated, cached); `.gguf_file` renamed; the download with progress, once only, a cached file of another size downloaded again, the token sent to the Hub and **not** to the redirect target, a cut-off download removed, Stop; every Hub problem (not found, private, no file, gated, local folder, HTTP 500, offline) with its message; the exact merge command line (no `-m`, no `-f`), the registration (files, json fields, parents with hashes, weights), progress that never goes back, the published matrix as a parent with its warnings, three inputs, models that differ, a merged entry used by an export (`use_saved_file`), compatibility warnings (architecture, layers, tensor sets, a `.dat`), every refusal, each llama-imatrix failure message, the retry without `--output-format`, no llama.cpp, Stop (a hanging tool is killed, no folder left), not enough disk, the worker entry.
- Rust `crates/ndforge-core/tests/imatrix_merge.rs`: the choice and old specs, Hub ids, the base of a source, the job and every refusal, the base models offered, fake-worker runs (unix only: a merge holding its parents, Stop, failures, the lookup, a creation that also merges, keeps its own matrix when the merge fails or the base is a folder).
- Window `ui/smoke/imatrix_merge.test.mjs`: the ticks, the honest text, files, the published matrix (choice, lookup, size and download notice, cached, gated, every error), the spec that is sent, progress / Stop / result with warnings and weights, the Create tick, and a creation that also merged or failed to.

### Deviations (merging)

- **`source_kind: "merged"`**, not `source: "merged"`: in this project's entries `source` is the sentence ("the trained run run1") and `source_kind` the kind (`run` / `model` / `gguf` / `imported` / `export`); the merged entry follows that (its `source` says "a merge of ...").
- The published matrix is downloaded with `urllib` against the Hub's plain HTTP API (`/api/models/<repo>?blobs=true`, `/<repo>/resolve/main/<file>`), not `huggingface_hub`: nothing to import, testable with a local server.
- The merged entry has no statistics (`--show-statistics` needs a model file next to it) and no `ctx`.
- A merged entry's `chunks` is the sum of its parents' counts, which is what the page shows; llama-imatrix's own `imatrix.chunk_count` in the file may differ (it counts the chunks it saw).

## Verified on the real published file

On 2026-11-16 the file `imatrix_unsloth.gguf` of the Hub repository `unsloth/Qwen3.8-27B-GGUF` (13,642,656 bytes, 13.6 MB; downloaded by the owner) was read by every reader of this program. It was only parsed, never executed. The evidence is committed: `worker/tests/data/imatrix_unsloth_qwen38_27b_header.txt` (31 lines, the header dump), the scaled-down generator `worker/tests/imatrix_fixture.py` that writes exactly this layout (used by every CI test that needs a matrix), and `worker/tests/test_imatrix_real_format.py` (the dump's structure is reproduced for 64 layers; the readers are checked on the fixture and, with `NDFORGE_REAL_IMATRIX=<path>`, on the real file; the Rust twin: `NDFORGE_REAL_IMATRIX=<path> cargo test -p ndforge-core real_published`).

| Fact | Value |
|---|---|
| Format | GGUF version 3, alignment 32 (no `general.alignment` key); the tensor data starts at the 32-aligned end of the tensor-info block and every tensor, also the one-value `counts`, is padded to 32 bytes |
| Key-values | exactly four: `general.type = "imatrix"`, `imatrix.datasets` (array of strings: `["unsloth_calibration_dataset"]`), `imatrix.chunk_count` (uint32, 1251), `imatrix.chunk_size` (uint32, 8192). **No `general.architecture`**: a check that needs the model must infer it from names and shapes |
| Tensors | 992, all float32 (type 0), all **one-dimensional** (`[5120]`, not `[1, N]`); sorted by the name of the weight as text (`blk.0`, `blk.1`, `blk.10` ... `blk.19`, `blk.2` ...), `<weight>.in_sum2` then `<weight>.counts` |
| Names | `blk.<L>.<tensor>.weight.in_sum2` (length = the input width of that weight: the sum over tokens of the squared input per channel) and `blk.<L>.<tensor>.weight.counts` (length 1) |
| Counts | `counts` is 1149614.0 for every tensor; `in_sum2 / counts` is the mean square activation per channel. No NaN, no zero |
| Layers | 0 to 63. Full attention at 3, 7, ..., 63 (16 layers): `attn_q`, `attn_k`, `attn_v` (input 5120), `attn_output` (input 6144). The other 48 (linear attention): `attn_gate`, `attn_qkv`, `ssm_alpha`, `ssm_beta` (input 5120), `ssm_out` (input 6144). All 64: `ffn_down` (input 17408, the MLP intermediate size), `ffn_gate`, `ffn_up` (input 5120, the hidden size). 48 x 10 + 16 x 8 + 64 x 6 = 992 |
| Not in it | no `output.weight`, no token embedding, no MTP / NextN entries |
| Scale | the mean of `in_sum2` of `ffn_down` is about 103 in layer 0 and about 2.8 million in layer 63 (per count: 9e-5 and 2.5): layers differ by orders of magnitude, which is why layers must be dropped, not averaged |

What the readers did with it: the header reader, `matrix_summary`, `read_importance` (`in_sum2` one-dimensional with one count: its `n_in x n_experts` description now says both layouts) and the Rust `gguf_meta.rs` read the file correctly; what changed is what they report and compare. `matrix_summary` also returns `chunk_size`, `datasets`, `kinds` (how many layers have each tensor kind), `hidden` (input of `ffn_gate`), `intermediate` (input of `ffn_down`) and `tokens` (the `counts` value: what a merge adds up); `ggufinfo.read` keeps a short string array under `<key>#items`; `compat_warnings` compares the hidden size and the MLP size when the matrix has no architecture (and says what a thinner `ffn_down` means); the Rust twin has the same fields and texts (`read_tensors` returns the dimensions).

## Adapting a matrix to a cut model

Built for a model cut on the **Slim or grow a model** page (layers removed in blocks and/or the MLP made thinner; the record is `models/<name>/model/ndforge_reshape.json`, the kept neurons are `width_kept.npz`, see `worker/ndforge_worker/reshapejob.py`). Unsloth publishes the matrix it used for the original model (the Hub repository `unsloth/<model>-GGUF`, `imatrix_unsloth.gguf`; the real layout is in "Verified on the real published file"). The Dynamic recipe is unpublished and cannot be reproduced; a matrix is per layer and per tensor, so it can be **adapted** to a cut model without measuring anything. The code is `worker/ndforge_worker/imatrixadapt.py` (pure numpy, no torch; jobs `imatrix_adapt` and `imatrix_adapt_preview`), `crates/ndforge-core/src/export/imatrix_adapt.rs` and the card **Adapt to my cut model** in Importance matrices.

### Owner decisions (defaults) as built

| | |
|---|---|
| Inputs | a source matrix: an entry of the library (a `.gguf`), a matrix file (`.gguf`), or **Unsloth's published matrix for a base model** (looked up and downloaded once with the same `imatrixmerge.lookup` / `download` as in a merge, into `imatrix_downloads/`; the page's Look it up shows the size first); and a made model of the project (`models/<name>`) |
| What the record gives | `origin.layers` (the original layer count), `src_layers` (the original layer that became each new layer: the exact old to new map, **already recorded** by the Slim job, so the job did not need extending), `dropped`, `width.intermediate_size` `[before, after]` and `width_kept.npz` (`mlp_<new layer>`: the kept neuron numbers of the ORIGINAL numbering, in the order the weights were sliced; `src` is checked against the record). A record that lists only the dropped layers is enough |
| Older records | a model made by an older version that recorded only counts (no `src_layers`, no `dropped`): **the person is asked** (the card shows a box "Layers of the original it kept", numbers counted from 0) and the answer is checked (different layers, in order, of the original, as many as the record says); without an answer the job is refused with that question |
| Depth | only the entries `blk.<old>.` of kept layers stay, renamed `blk.<new>.`; tensor order is the source's order with the kept entries (the real file is sorted by the weight's name as text; it is not re-sorted: llama.cpp does not care) |
| Width | `blk.<L>.ffn_down.weight.in_sum2` (one value per MLP neuron) is sliced with that layer's kept neurons; `ffn_gate`, `ffn_up`, the attention and `ssm_*` entries are copied (their input is the hidden size, which Slim never changes); `counts` are copied |
| Weight | optional `w` (default 1, above 0, at most 1,000,000) multiplies BOTH `in_sum2` and `counts`, in float64 then rounded to float32 (`w = 1` is bit-identical). The published matrix counts about 1,149,614 rows, a matrix of 128 chunks of 512 about 65,000 (65,536): in a plain merge it outweighs the other about 17 to 1; a weight of 0.05 makes them about even. The card, the Merge card and the help say so |
| Refusals (plain words, each with its own error code `adapt_<reason>`) | `no_metadata` (no `ndforge_reshape.json`, or no `origin.layers`: not a model of the Slim page, or healed from another model), `heads` (the attention heads or the linear-attention value heads were cut: the experimental head axes change the input of `attn_output` / `ssm_out`, which a matrix of the original cannot be cut to), `grown` (a model with added layers: nothing was measured for a copy), `needs_layer_map`, `bad_kept` (kept layers or `width_kept.npz` missing, damaged or not fitting the record), `wrong_original` (the matrix's layer count is not the original's, a layer lacks `ffn_down` / `ffn_gate` / `ffn_up`, or its `ffn_down` width is not the recorded one), `already_adapted` (a library entry with `source_kind: "adapted"`, or `imatrix.datasets` that has an "(adapted to" entry), `bad_matrix` (not the real layout: a `.dat`, truncated, a tensor that is not float32 or not one-dimensional, an `in_sum2` without its `counts`), `bad_weight`. Everything is checked before anything is downloaded or written |
| Output | `imatrix/<id>/imatrix.gguf` + `imatrix.json`, `source_kind: "adapted"`, `via: "adapted"`, `parents` (the source with its SHA-256), `adapt` (model, model folder, origin repo, layers before and after, `kept_layers`, `dropped_layers`, `ffn_down_before` / `after`, `weight`, tensors before and after), `chunks` (the source's), `tokens` (the rows after the weight), `cut_model` (so a later merge adapts the published matrix to the same model). The GGUF is written with exactly the real layout: the source's key-values in their order and types (`imatrix.datasets` gets an entry `(adapted to <model>)`; `imatrix.chunk_count` and `imatrix.chunk_size` are kept), 1-D float32, alignment 32, and the predicted size is the written size |
| Checks after writing | the file is read back with the real-layout reader and compared with the plan: tensor count, names, lengths, the header with the other reader (1-D, float32, aligned), and for every kept entry the values (and their sums) equal the source's at the kept places (times the weight, to float32 precision); a mismatch removes the file and says so. When `llama-imatrix` is found, `llama-imatrix --in-file adapted.gguf -o check.gguf` must exit 0 and write the same tensors (a stand-in tool in the tests; a refusal is a **warning** in the notes and `llama_check`, the entry is kept) |
| Merge with my own matrix | the merge job takes, per input, a `weight` and an `adapt_to` (`{"model_dir", "model_name"}`): "adapted published matrix + my own matrix" is one job. The adapted and weighted copies are written in a work folder of the new entry; the result lists the weights used. When a library parent was measured on a cut model, the published input is adapted to it automatically (weight 1 unless the person sets one) |
| Safety | see "Merging matrices": a published matrix is never merged raw with a matrix of a cut model |
| UI | Importance matrices, card **Adapt to my cut model**: the matrix source, the cut model (with how it was cut; a model that cannot be adapted shows why), the weight with the numbers above, a label, a preview (`keeps 32 of 64 layers; ffn_down 17,408 -> 10,752 per layer; exactly / about N MB`, from the file's header when it is at hand, else estimated from the size the Hub states), **Adapt**, progress, **Stop**, the result and its notes; the plain statement that the matrix still describes the unmodified model's activations (an approximation after cutting and healing; best merged with a matrix of your own measured on the healed model) and is not the unpublished recipe behind Unsloth's own GGUF sizes |

### Commands

Worker jobs `imatrix_adapt`, `imatrix_adapt_preview` (new; no existing job type or event was renamed: the nd-world repository embeds this worker). Tauri commands `imatrix_adapt_preview` (the numbers, in Rust, from the model's record and the matrix header: no Python needed) and `imatrix_adapt` (events on `imatrix-event`, like `imatrix_create`; Stop is `imatrix_stop`). Old specs load: `MergeInput` got `weight` and `adapt_to`, `AdaptSpec` is new, all with serde defaults.

### Tests

- Worker `worker/tests/test_imatrix_real_format.py` (the fixture reproduces the committed dump's structure for 64 layers; the readers on the real layout; the real file when `NDFORGE_REAL_IMATRIX` is set) and `worker/tests/test_imatrix_adapt.py` (the worker's writer is byte for byte the independent fixture generator's; depth, width, both, the model made by the real Slim job, the weight, every refusal, the job with library / file / published sources, the stand-in llama-imatrix, Stop, a wrong write caught by the verification, the preview, the worker entry with its error codes, merging with weights and with an adapted published matrix, the safety rule, and the real file adapted to a 64 -> 32 layer, 17,408 -> 10,752 cut when `NDFORGE_REAL_IMATRIX` is set).
- Rust `crates/ndforge-core/tests/imatrix_adapt.rs` (spec and old specs, the record and its refusals, preview numbers, exact against the sizes the worker's writer gives for the same layout, the jobs, the merge job with weights and the automatic adaptation, the state of the page, fake-worker runs: unix only) and `export/gguf_meta.rs` (the real layout; the real file with `NDFORGE_REAL_IMATRIX`).
- Window `ui/smoke/imatrix_adapt.test.mjs`.

### Deviations

- **No extension of the Slim job** was needed: it already writes `src_layers`, `dropped`, `origin.layers` and `width` (`intermediate_size` `[before, after]`, `kept`). Only records without `src_layers` and `dropped` (older versions) need the person's answer.
- **Grown models are refused** (the brief only mentioned layers removed in blocks): a duplicated layer has no statistics of its own.
- **Entries beyond the original's layers** (a speculative-decoding head in a matrix measured with `--nextn`) are left out with a note, not refused.
- **The shares of a merge use the rows of text (`counts`)**, not `imatrix.chunk_count`, when every parent has them: the published matrix's `chunk_count` is 1251 with a chunk size of 8192, but its `counts` is 1,149,614 (the statistics were summed over fewer rows than chunks x size). Without readable counts the chunk figures are used as before.
- **The scale figures of the brief** ("ffn_down mean-square from about 1.6 to about 42,657") were not reproduced: measured on the real file, the mean of `in_sum2` of `ffn_down` is 103 in layer 0 and 2.8 million in layer 63 (per count: 9e-5 and 2.5). The conclusion (layers differ by orders of magnitude) is the same.
- **`tensor order as in the source`**: kept entries stay in the source's order, which is the real file's text sort of the OLD layer names (blk.10 before blk.2); the renumbered file is therefore not sorted by its new names. llama.cpp reads tensors by name.
- **`matrix_summary` returns more keys** (an existing test that compared the whole dict was updated to the new keys; no assertion about the old keys was changed).
- **`ModelFacts` (Rust) got `hidden` and `intermediate`**; struct literals in `gguf_meta.rs`'s own tests use `..Default::default()`.

## Not verified yet (needs llama.cpp and a GPU)

Nothing below has run. Do not rely on any of it until checked on the owner's PC:

- **`--nextn` on a real model**: that it works on a Qwen3.x GGUF with an MTP head, what it prints (the retry looks for `nextn`, `KV cache` or `mtp` in the output), that the head is in the converted GGUF at all (the Transformers + PEFT merge may drop `mtp.*` tensors; the worker then sees a GGUF without a head and says so), the key / tensor names of the head in a real GGUF (`gguf_has_mtp` looks for `*nextn_predict_layers` and `...nextn...` / `mtp...`: a guess from other model families), and that `llama-quantize` accepts a matrix made with `--nextn`.
- **That the installed Unsloth really applies `imatrix_file`**: only the source of 2026.11.4 was read. Whether Unsloth's own `llama-quantize` build reads the **GGUF-format** matrix this feature writes (its messages name `.dat` and `.gguf`), whether it works for a LoRA model loaded the way the export does, and that `imatrix_file` of a path inside the export folder is left alone by its clean-up of the output folder.
- That one call for the whole batch (Q8_0 included, which ignores the matrix) behaves, and the time and memory of merging with Transformers + PEFT for the measuring *and* again inside Unsloth (the matrix path merges twice; the first copy is removed before Unsloth loads).

- The exact flags on the user's llama.cpp build: `--output-format`, `--no-ppl`, `--parse-special`, `--chunks`, `-ngl`, `-c`, and the position of `--imatrix` for `llama-quantize` (options first, then the files).
- What `llama-imatrix` prints with `--no-ppl`: the progress parsing (`[12]` markers) is a guess; without markers only the heartbeat shows.
- Whether `llama-quantize` accepts a GGUF-format matrix on that build (older builds read `.dat`), and the text of the errors that make the page fall back.
- The time of the matrix for a **9B and a 27B** model (the time guess on the page is a rough formula, not a measurement), the memory it needs next to the 16-bit GGUF, and whether **`-ngl 99` helps** (and how a 27B f16 behaves on a 32 GB card).
- That the calibration text, tokenized by llama.cpp with `--parse-special`, really gives the chat-template special tokens as special tokens, and that its token count matches the count made here (a 10 % slack is added).
- The default general dataset (`mlabonne/open-perfectblend`, chat, rendered through the model's template; see docs/GENERAL_TEXT.md) as streamed here (it is the healing code's default, itself unverified), and how long the failed download of an offline PC takes before the note appears.
- **Quality of Q4_K_M with and without the matrix** on the Test page (the point of the feature): make both, run the Step 4 test on each.
- Whether killing `llama-imatrix` on Windows leaves anything running.

Matrix library (everything above, and in addition):

- **`--show-statistics` output**: its real text (the README only describes the columns); the parser looks for a `(N tensors)` line and a tab-separated header with `Min` / `Max` / `μ` / `% Active`, and whether `-m <model>` is needed next to `--in-file` (the README example has both).
- **`--in-file x.gguf --output-format dat -o y.dat`** (and the extension-only fallback) on the user's build, that the `.dat` it writes is read by `llama-quantize` and by Unsloth's quantiser, and that a `.dat` imported from elsewhere is accepted by them.
- **Compatibility between matrices and files**: that a matrix made on a merged model fits quantising a different source file of the same model (for example the one an export converts, or another GGUF of the same weights), and that a matrix made on a Q8_0 GGUF is as good as one made on F16.
- **The layer comparison for hybrid or MTP models** (the MTP layers are allowed as extra): a guess from the configs; the published matrix has no MTP entries (see below), so only a matrix measured with `--nextn` shows how an MTP head is named in the statistics.
- **Hugging Face download** of a model folder with `snapshot_download` (ignored patterns: gguf, onnx, msgpack, h5, pth, `original/*`, md), gated models and the token path, the disk use of a big model, and that `convert_hf_to_gguf.py` accepts the downloaded folder.
- Merging a trained run for the library (memory, time, that STOP during the merge ends within the 20 s grace), the 16-bit file of an export being really newer than the weights check, and the time of the conversion (the page only says "plus the time to convert the model").
- The chat template of a typed folder or Hub id with the family `generic`, and of a GGUF whose "made from" model is a different quantisation of the same family.
- `explorer` / `open` / `xdg-open` for "Open folder" on Windows, macOS and Linux (the program only reports a failure to start the tool).
- That `llama-imatrix --in-file ... --show-statistics` and `--output-format dat` do not take long enough on a large matrix to need their own progress (no progress is shown for them).
- Windows paths with spaces or non-ASCII characters through the stand-in tools only (the tests do not use real llama.cpp).

Adapting a matrix to a cut model (in addition; **nothing here has run with a real llama.cpp, a real cut model or a GPU**):

- **That `llama-imatrix` and `llama-quantize` accept an adapted file.** The layout is the real one (verified byte for byte against an independent generator of it, and the real published file adapted and read back), but no real llama.cpp has read the output: the check after writing is tested with a stand-in tool only. Whether `llama-quantize --imatrix` is content with a matrix whose layers are numbered 0 to N-1 for a model with N layers, and with an `ffn_down` of the cut width, is not known until it is tried.
- **How much an adapted matrix improves a cut model's quants.** That is what **Compare quants** measures: make the same size with and without the adapted matrix (and with the adapted matrix merged with the owner's own matrix, at several weights) and compare. Nothing says yet that the statistics of the unmodified model are a good guide for a cut and healed one; the only claim made is that they are an approximation.
- **A real cut model.** The tests use the files the Slim job's own functions write (`_save_kept`, the record's keys, and a tiny model made by the real job), not a 16B model; the layers and neurons of the owner's real cut model were not looked at.
- **The weight to use.** 0.05 to 0.1 for the published matrix next to a matrix of 128 chunks is arithmetic on the token counts, not a measured optimum.
- **Memory and time on a real matrix**: the whole matrix is held in memory while it is adapted (13.6 MB for the 27B's; a mixture-of-experts matrix with two-dimensional tensors is refused as not the real layout).

Merging matrices (in addition):

- **`llama-imatrix --in-file a --in-file b -o out`** on the user's build: that it needs neither `-m` nor `-f` (the README's usage line says they are mandatory; its combine example and `imatrix.cpp` say otherwise), that `--output-format gguf` is accepted next to it, and the exact text it prints when the inputs do not fit (the page quotes the line that mentions the mismatch).
- **That `llama-quantize` accepts a merged matrix** (the merged file's header, its `imatrix.datasets`, a chunk count that is the sum), and that a merge of a **legacy `.dat`** with a GGUF matrix works on that build.
- **Unsloth's real published matrix**: that `unsloth/<Model>-GGUF` has the file named as expected for the models in this project's list (checked by hand for `Qwen/Qwen3.8-27B` only), that the Hub's `siblings` list carries the `size` with `?blobs=true` as the stand-in does, how big the real files are, that a real redirect to the download server works with the manual redirect following and that Windows corporate proxies let it through; the name derivation for models whose Unsloth mirror has another name (`get_model_name`).
- Whether a published matrix measured on the unmodified base is a **useful** start for the person's fine-tune, and with what weight: that is a question for Compare quants (make the same size with and without the merge and compare).

## Deviations from the brief

- **(A)** The page cannot know whether `llama-imatrix` has `--nextn` (the program does not find llama.cpp; the worker does), so the greyed state with a reason is driven by `imatrix.nextn_supported` of the page state, which is `null` for now: the real answer comes in a note after the run. The tick is shown for a model known to have a head (`mtp` true); with unknown (`null`) it is not offered.
- **(A)** `nextn` in `imatrix.json` is what the file contains; the reuse key is compared with `nextn_asked` (a refused head would otherwise make every later export measure again).
- **(B)** Unsloth gets **one** call for the whole batch with the matrix, Q8_0 included, instead of a second call without it for Q8_0 / F16 (that would merge the model twice more). The report marks those sizes "high precision".
- **(B)** The cannot-apply test is stricter than Studio's: the zoo CPU loader's `**kwargs` is refused.
- **(B)** `imatrix_file=True` (Unsloth's published matrix) is not offered, see above.

- The calibration text and the matrix are made on the llama.cpp path even when the Unsloth back end is selected (see above); this was the brief's own fallback rule, applied because the installed Unsloth's `save_pretrained_gguf` could not be confirmed to take a matrix.
- `--output-format gguf` is passed, with a retry without it for older builds (the brief listed it as optional).
- The progress of `llama-quantize` is not parsed (only the existing "Shrinking to ..." step).

## Follow-ups

- Ask for `--nextn` support before the run (the page now learns it only from a note afterwards).
- Accept a saved matrix made with more chunks than asked for.

## Thinking models: conversations with thinking in the calibration text

For a model that thinks (Qwen3.5 / 3.8, a cut one above all), the calibration text can hold conversations **with their thinking**, written the way the model writes them with thinking on
(`data.render_chat_full`: the model's own template over the whole conversation, `<|im_start|>assistant\n<think>\n{trace}\n</think>\n\n{answer}`, no empty block in front of a trace). The importance matrix then
weighs the weights by text that looks like what the quantised model will be asked, not only by thinking-off practice conversations. It matters for the small sizes (Q4_K_M and below), where the matrix steers the rounding.

- **Option**: `reasoning` in the worker's calibration, `{"source": {"id" | "file"}, "share": 0.5, "max_len": 2048}` (the same request and the same key as the reshape measuring's; default source the teacher's own
  traces, `faunix/Qwen3.8-27B-Distillation-40K`). In Rust: `ImatrixOptions.reasoning` (export) and `MakeSpec.reasoning` (matrix library), both a `ReasoningCalib`. Off by default: without it nothing changes.
- **How**: `share` of the calibration tokens (at most 90 %) are conversations with a thinking trace, each cut to its first `max_len` tokens (a later part would lack the question and the chat frame); conversations without
  a trace are passed over for a model that thinks. They are mixed evenly with the project's text and the general text through the file, so `--chunks` cuts none of the kinds out.
- **Key**: the reasoning request is part of what a saved matrix was made from (`reasoning` in `imatrix.json`, with `reasoning_used` and `reasoning_tokens`). A matrix made without it is not used when it is wanted, another share or
  row length is another matrix, and one made while the text could not be read (offline: a note, the matrix is made without it) is not used when the text is wanted.
- **Auto-cut** switches the importance matrix on for its export and passes this option when *keep it able to think* is on: the matrix of the finished model is measured on project text, general text and reasoning text.
  Tests: `worker/tests/test_imatrix.py` (the text, the cut, the fallback, the key), `tests/export.rs`, `tests/imatrix_library.rs` (the job's request).
