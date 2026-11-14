# Importance matrix (imatrix) for the GGUF export

An optional step of Step 5 ("Better small sizes (importance matrix)"). Built and tested with stand-in tools; **never run with a real llama.cpp or a GPU** (see "Not verified yet").

## What it is, and when it helps

When a model is shrunk to 4 or 5 bits, every weight is rounded. An **importance matrix** records, for a sample of text, how strongly each input of each layer is used. llama.cpp's
`llama-quantize --imatrix` then spends its precision where it matters, so **Q6_K, Q5_K_M and Q4_K_M lose less quality** than without it. It mainly helps at about 4 bits and
below; for **Q8_0, F16 and BF16 it does (almost) nothing**, so those sizes are never given the matrix.

The matrix is a measurement of the model on text, so it should look like what the model will be asked: the project's own training data, as the model sees it.

## How it is made

All in `worker/ndforge_worker/imatrix.py`, called from `exportjob.gguf_with_llama_cpp` after the f16 conversion:

1. **Calibration text** (`build_calibration`), about `ctx x chunks` tokens (+10 % slack):
   - about **80 %** the project's saved data version (the one the run trained on: `train.jsonl`): a practice conversation rendered with the **model's own chat template** (Qwen3.5 and Gemma 4 with thinking off, the family's end marker; special tokens such as `<|im_start|>` stay in the text), a world-book text as it is. Rows are shuffled with a fixed seed; data smaller than its share is repeated.
   - about **20 %** general text, through the healing code (`distilljob._general_texts`, default dataset `distilljob.DEFAULT_GENERAL` = `mlabonne/open-perfectblend`, or the dataset chosen in the box under the tick; chat rows go through the model's chat template; streamed, never downloaded whole; see docs/GENERAL_TEXT.md; a long text counts for at most two chunks).
   - both kinds are mixed evenly through the file, so `--chunks` never cuts one kind away. The same data and settings always give the same file.
   - The tokenizer is loaded from the merged model folder (so it carries the patched, non-thinking template), then from the base model.
2. **llama-imatrix** (the tool is looked up next to `llama-quantize`, then on `PATH`):

   ```
   llama-imatrix -m model-f16.gguf -f imatrix-calibration.txt -o imatrix.partial.gguf --output-format gguf --no-ppl --chunks N --parse-special -ngl 99 -c 512
   ```

   `--output-format gguf` is dropped and the call repeated if this llama-imatrix does not know it (older builds). `--process-output` and `--nextn` are not used. `-ngl 99` offloads to the GPU and is harmless on CPU-only builds. The file is renamed to `imatrix.gguf` only when the tool succeeded.
3. **llama-quantize** for every size below Q8_0: `llama-quantize --imatrix imatrix.gguf model-f16.gguf model-<size>.gguf <SIZE>`. Q8_0 is made as before; F16 is the converter's own file.

The matrix is made **once per export** (all chosen sizes share it).

## Why llama.cpp, not Unsloth's `save_pretrained_gguf`

Unsloth Studio's export code (`studio/backend/core/export/export.py`) passes an `imatrix_file` to `save_pretrained_gguf` and refuses it with "This Unsloth build does not support GGUF imatrix export" when the installed build's function neither has that parameter nor `unsloth_zoo.llama_cpp.resolve_imatrix_file`. That was **not confirmed for the Unsloth that Lore Forge installs**, and a matrix has to be measured on a GGUF first anyway (which needs llama.cpp). So when a matrix is asked for **and** llama.cpp with `llama-imatrix` is found, the GGUF is made the llama.cpp way (merge with Transformers + PEFT, convert, imatrix, quantize) even if the Unsloth back end is selected. Without llama.cpp (or without `llama-imatrix`, or with no size below Q8_0) the export goes through Unsloth as before and says so in a note.

## Files

Per export folder (`exports/<run>-<size>/`), next to the GGUFs, in **every** export of one batch (so a later export finds it whichever size was kept; it is small):

- `imatrix.gguf` - the matrix.
- `imatrix.json` - what it was made from: `model` (run name), `weights` (a fingerprint of the run's safetensors files, so the same run trained further does not match), `data_hash` (sha256 of `train.jsonl`), `data_version`, `chunks`, `ctx`, `general_share`, `general_used`, `general_source`, `project_tokens`, `general_tokens`, `llama_cpp` (the version line of `llama-imatrix --version`, when obtainable), `base_repo`, `created`.
- `meta.json` gets `imatrix`: `null` when it was not asked for, else `requested`, `applied` (this size used it), `made`, `reused`, `file`, `chunks`, `ctx`, `general_used`, `llama_cpp`, `reason` (why this size did not use it), `note`.
- `export.json` (the worker's summary) has the whole `imatrix` report.

**Reuse.** A later export of the same run reuses a saved matrix instead of measuring again when `model`, `weights`, `data_hash`, `chunks`, `ctx`, `general_share` and `general` (the dataset: id or file, config, split, column, share, seed; a matrix saved before this key existed counts as made from other data) are all equal (the checkbox "Reuse the saved matrix" is on whenever one exists; untick it to measure again). A matrix that was made without general text (offline at the time) is **not** reused while general text is wanted. The page checks only that a matrix of the run exists; the worker does the exact comparison.

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
| `--nextn` (MTP heads) | not used in this version (follow-up) |

## When something is wrong: never fatal

The export is never failed by the matrix step. Every problem is a plain-language **note** (shown on the page, in `meta.json` and in the log) and the sizes are made without it:

| Situation | Note (shortened) |
|---|---|
| llama.cpp has no `llama-imatrix` | "This llama.cpp has no llama-imatrix (an older install), so the sizes were made without the importance matrix. Update llama.cpp ..." |
| Unsloth back end and no llama.cpp | "An importance matrix was asked for, but llama.cpp (with llama-imatrix) was not found, so the GGUF was made through Unsloth without it. ..." |
| no size below Q8_0 chosen | "The importance matrix helps the sizes below Q8_0; none of them was chosen, so it was not made." |
| general text cannot be read (offline, gated, not found, no usable column, nothing read, no `datasets`) | "The general text (...) could not be read: <plain reason>. The matrix is made from your project's text only." (the matrix is still made; docs/GENERAL_TEXT.md lists the reasons) |
| `llama-imatrix` fails | "llama-imatrix failed (exit code N; see the log), so the sizes were made without the importance matrix." |
| `llama-quantize` refuses `--imatrix` | "llama-quantize did not accept the importance matrix for Q4_K_M (exit code N); that size was made without it." |
| training file or tokenizer missing, any other error | "The importance matrix could not be made (...), so the sizes were made without it." |

Only the Stop button ends the export early.

## Tests

- Worker: `worker/tests/test_imatrix.py` (stand-in `llama-imatrix`, `llama-quantize` and convert scripts written as small Python files and started through the current interpreter, a stand-in tokenizer; no torch, llama.cpp or network; the same on Windows and Linux): exact command lines (`--imatrix` only for the right sizes, `--parse-special`, `-c`, `--chunks`, `-o`), reuse and every reason not to reuse, each fallback, the stop file (a tool that would run 30 s is killed at once), the calibration text (chat template, special tokens, 20 % general text mixed in, deterministic, repeats small data), which back end makes the GGUF, and a whole export job.
- Rust: `crates/ndforge-core/tests/export.rs` (options default off and are checked, the job block, saved matrices and time guesses, the matrix and its json kept in every export of a batch, meta, graceful Stop with a fake worker) and `export/runs.rs` (STOP file first, kill after the grace time).
- Window: `ui/smoke/export.test.mjs` (off by default, 64/128/256, reuse box, only for GGUF, options sent, the files list, Cancel).

## Not verified yet (needs llama.cpp and a GPU)

Nothing below has run. Do not rely on any of it until checked on the owner's PC:

- The exact flags on the user's llama.cpp build: `--output-format`, `--no-ppl`, `--parse-special`, `--chunks`, `-ngl`, `-c`, and the position of `--imatrix` for `llama-quantize` (options first, then the files).
- What `llama-imatrix` prints with `--no-ppl`: the progress parsing (`[12]` markers) is a guess; without markers only the heartbeat shows.
- Whether `llama-quantize` accepts a GGUF-format matrix on that build (older builds read `.dat`), and the text of the errors that make the page fall back.
- The time of the matrix for a **9B and a 27B** model (the time guess on the page is a rough formula, not a measurement), the memory it needs next to the 16-bit GGUF, and whether **`-ngl 99` helps** (and how a 27B f16 behaves on a 32 GB card).
- That the calibration text, tokenized by llama.cpp with `--parse-special`, really gives the chat-template special tokens as special tokens, and that its token count matches the count made here (a 10 % slack is added).
- The default general dataset (`mlabonne/open-perfectblend`, chat, rendered through the model's template; see docs/GENERAL_TEXT.md) as streamed here (it is the healing code's default, itself unverified), and how long the failed download of an offline PC takes before the note appears.
- **Quality of Q4_K_M with and without the matrix** on the Test page (the point of the feature): make both, run the Step 4 test on each.
- Whether killing `llama-imatrix` on Windows leaves anything running.

## Deviations from the brief

- The calibration text and the matrix are made on the llama.cpp path even when the Unsloth back end is selected (see above); this was the brief's own fallback rule, applied because the installed Unsloth's `save_pretrained_gguf` could not be confirmed to take a matrix.
- `--output-format gguf` is passed, with a retry without it for older builds (the brief listed it as optional).
- The progress of `llama-quantize` is not parsed (only the existing "Shrinking to ..." step).

## Follow-ups

- `--nextn` for models with MTP heads.
- Pass the matrix to Unsloth's `save_pretrained_gguf(imatrix_file=...)` when the installed build is shown to accept it (`inspect.signature`), so llama.cpp would be needed only for the measuring.
- Accept a saved matrix made with more chunks than asked for.
