# Importance matrix (imatrix) for the GGUF export

An optional step of Step 5 ("Better small sizes (importance matrix)"). Built and tested with stand-in tools; **never run with a real llama.cpp or a GPU** (see "Not verified yet").

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
