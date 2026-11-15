# Compare quants (KL divergence)

A page of its own, **Compare quants** (main navigation, after "Importance matrices"; also reached from Step 5 with the button **Compare these sizes**). It measures how far each quantised GGUF of a model is from the 16-bit file, with llama.cpp's `llama-perplexity`, and shows a table, a chart and plain words. Built and tested with stand-in tools; **never run with a real llama.cpp, model or GPU** (see "Not verified yet").

Code: `worker/ndforge_worker/quantcheck.py` (jobs `quantcheck` and `quantcheck_plan`), `worker/ndforge_worker/ggufinfo.py` (header reader), `crates/ndforge-core/src/export/quantcheck.rs`, the page in `ui/app.js` (`renderQuantcheck`), help `crates/ndforge-core/help/quantcheck.md`.

## What it measures, and what it does not

1. `llama-perplexity -m reference.gguf -f test.txt --kl-divergence-base base.kld -c 512 --chunks N -ngl 99`: the 16-bit model reads the test text and writes, for the second half of every chunk, the log-probabilities of all words of the vocabulary (squeezed into 16 bits).
2. For every quant: `llama-perplexity -m quant.gguf --kl-divergence-base base.kld --kl-divergence -c 512 --chunks N -ngl 99`. It prints (read leniently, see below) the mean KL divergence with its uncertainty, the median, the 99.9 / 99 / 95 / 90 percentiles, the maximum, the percentage of places where both files pick the same top word ("same top p"), the RMS change of the correct word's probability, and the perplexity of the quant and of the base.

Per file the page shows: size (GB = 10^9 bytes), **bits per weight** (file size x 8 / the number of weights read from the GGUF header, all tensors), mean KLD, median, 99 %, maximum, same-top-word, PPL. KLD is **a proxy**: it says how much the probabilities moved on this text. Unsloth reports it for its quants too, but it does not replace trying the model on your own questions (Step 4 · Test); the page says so in its verdict.

Not measured here: answer quality, speed, memory, long contexts. The test is `ctx 512`; quantisation damage at long contexts is not seen.

## Owner decisions, as built

| | |
|---|---|
| Reference | a 16-bit (F16 / BF16) GGUF: one already in an export folder (an export of size "F16", a `model-f16.gguf` left in an export folder, a reference an earlier comparison kept), one chosen by path, or **made by the worker** with llama.cpp's converter from a trained run or a model of the list (the matrix library's own source code, `imatrixjob.source_f16`; merged first for a run). A made reference is kept in the comparison's folder (`reference-f16.gguf`) and deleted with the comparison. A reference that is not 16-bit (by `general.file_type`, else by the tensor types) is allowed with a note: the figures then say how far the others are from it |
| Files to compare | any of the project's export GGUFs (ticked) and any path typed or browsed; with matrix / without matrix / mixed-precision variants in one table (the words "with matrix", "no matrix" come from each export's `meta.json`: `imatrix.applied`; a mixed-precision export names its preset in `meta.json` `mixed`) |
| Test text | **default: text that was not used for the matrix calibration**: the held-out rows of the saved data version (`datasets/vN/test.jsonl`); when that version has none, a sample of its training rows with seed 1234 (the matrix calibration uses 3407), and a note says so. Rendered with the model's own chat template, then the special markers taken out (see "Deviations"). Option: a text file of the person's own, used as it is. The tokenizer comes from the run the files were made from (its add-on's tokenizer, then its base model) or a typed folder / Hugging Face id |
| Amount of text | context 512; chunks 4 / 8 / **16** / 32 / 64 / 128; the page shows the disk need of each before anything runs. If the text holds fewer chunks than asked, fewer are used and a note says so |
| Disk | the base file's size is computed from llama.cpp's own source and shown (see below), the free space is asked (`quantcheck.free_bytes`, replaced in the tests), a tenth of the disk is kept in reserve. Too little: a refusal in plain words that names the numbers, and, if a smaller amount would fit, says so. Nothing is written before that check. `base.kld` is deleted when every file was measured, unless "Keep the 16-bit model's answers" is ticked; after Stop, an error or a failed row it is kept (so a retry does not read the text again) and a button deletes it |
| Passes | one base pass, then one pass per file; one progress bar over all passes (`quantcheck` step, the chunk table's rows give the chunk marks); `results.json` is rewritten after every pass |
| Stop | the STOP file (as for the matrix): the tool is killed at once, the finished passes stay in `results.json` (state `stopped`), a half-written `base.kld` is removed, the page says "Stopped. The files that were already measured are kept". The program kills the worker 20 s later if it has not ended |
| Continue | **Continue** on a saved comparison: the base pass is skipped when `base.kld` still exists and the **key** matches (`results.json` `key`: fingerprint of the reference file (size, time, hash of its first MB), SHA-256 of the test text, chunks, ctx, the version line of `llama-perplexity`); files already measured (same path, size and time) are skipped; files ticked in the form are added. The saved `test.txt` is measured again, so the text is the same. A different key with the base file still there is refused in plain words; without the base file the old rows are dropped and everything is measured again |
| Files | `quantcheck/<id>/results.json`, `test.txt`, `base.kld` (temporary), `reference-f16.gguf` (only when the program made it); id = `yyyymmdd-hhmm-<short reference name>` |

## The size of the base file

`--kl-divergence-base` is documented in the README of llama.cpp's `tools/perplexity` only as "very large"; the format was read from its source (`tools/perplexity/perplexity.cpp`, master, 2026-11). The file holds `_logits_` (8 bytes), `n_ctx`, `n_vocab`, `n_chunk` (4 bytes each), the tokens of every chunk (4 bytes each) and then, for the **second half of every chunk only** (`n_ctx - 1 - n_ctx/2` tokens: 255 of 512), one record of `2 x (2 x ceil(n_vocab / 2) + 4)` bytes per token. So:

`bytes = 20 + 4 x n_ctx x chunks + chunks x (n_ctx - 1 - n_ctx/2) x 2 x (2 x ceil(n_vocab/2) + 4)`

For a 248,320-word vocabulary: 16 chunks of 512 = about **2.0 GB** (4,080 compared words), 32 chunks 4.1 GB, 128 chunks 16.2 GB. This is about half of "tokens x vocabulary x 2 bytes", because only half of each chunk is compared. The vocabulary is read from the reference's header (the length of `tokenizer.ggml.tokens`); if it is not readable, 250,000 words are assumed and the page says so.

## Reading the tool's output (lenient on purpose)

`quantcheck.parse_kld` looks at each line for a label and the first number after it: `Mean    KLD:   0.012345 ± 0.000100`, a table row `| Mean KLD | 0.0123 ± 0.0001 |`, `Same top p: 97.7 ± 0.01 %`, `Mean PPL(Q) : 6.22`, percentile lines (`99.0%   KLD:`), `RMS Δp`. Lines it does not know are ignored; the Greek Delta may arrive as another character. If the final statistics are missing but the per-chunk table was printed (the tool prints no statistics for fewer than 100 compared words), the last row of that table gives the mean KLD, same-top, RMS and PPL and the row is marked "running figure" (`partial`). If no figure is found the row **fails with a reason** (the log keeps the tool's last lines); the tool's exit code is not trusted either way, because llama-perplexity returns 0 after a failed KL pass (`kl_divergence` only logs the error).

## The table, the chart and the words

Sorted by size. The chart plots mean KLD (y, log scale when the values differ by more than a factor of 30) against size (x); the **Pareto line** joins the files that no smaller file beats (lower left is better), computed in the worker (`quantcheck.pareto`) and stored in `results.json`. The words (`quantcheck.verdict`): the smallest and the closest file, which files are on the line, which are not worth it (a smaller file is at least as close), "at about X GB, Y keeps the most", for sizes that exist with and without a matrix "Q4_K_M with an importance matrix is N % closer than without", for mixed-precision files the plain size nearest in file size, and the last sentence that KLD is a proxy.

## Everything that can go wrong (never fatal)

| Situation | What the page says (shortened) |
|---|---|
| `llama-perplexity` not found | "llama-perplexity was not found ... Install a current llama.cpp ..., set LLAMA_CPP_DIR ... looks in LLAMA_CPP_DIR, then the folder the program installed llama.cpp into, then a llama.cpp folder in your home folder, then next to the program; also on the PATH" (also its `build/bin` folders), error code `llama_cpp_missing`: the page then shows **Install llama.cpp now** (docs/LLAMA_CPP_INSTALL.md) |
| the training software is not installed | "Open Training setup first" (the worker runs the job; no torch is needed for the comparison itself) |
| the reference is missing / not a GGUF | "was not found" / "could not be read as a GGUF file ... Choose a 16-bit GGUF" |
| not enough disk | the numbers, "a tenth is kept in reserve", and the largest amount that would fit |
| disk full during the base pass | "the disk is full. Free some space" |
| a file cannot be read / has another vocabulary or architecture | only its row fails, with the reason; the others go on |
| out of memory / cannot load / text too short | named from the tool's output |
| the base pass wrote nothing | "did not write the file of the 16-bit model's answers ... too old?" |
| no test text | "too little test text" with the minimum; the tokenizer cannot be read: "Tell the page which model the files were made from" |

## Tests

- Worker `worker/tests/test_quantcheck.py` (stand-in `llama-perplexity` as a small Python script, hand-built GGUF headers with a token list and tensor infos, a stand-in tokenizer): the exact command lines of both passes, the base file size formula against hand-computed numbers and the real vocabulary, the plan (default, shrinking, refusal), the classic output, the README table, odd spacing and the running table, junk, `strip_specials`, the Pareto line and every sentence of the verdict (also matrix and mixed), text from a file / held-out rows / a training sample (deterministic) / too little data, the disk refusal before anything runs, the disk filling during the base pass, Stop (a hanging tool is killed; finished passes and the base file kept), Continue (no second base pass, finished rows skipped, an added file), a changed key, an existing folder, every failure message, making the reference with a stand-in converter, the GGUF reader, the worker entry (result, plan, stopped, failed, unexpected).
- Rust `crates/ndforge-core/tests/quantcheck.rs`: the choice and its old / partial forms, the words of the table, the job and every refusal, the data text job (held-out file, tokenizer sources, family), a reference made from a run, the files the page offers, listing / reading / deleting, delete only the logits, ids, and fake-worker runs (unix only: a whole comparison, the running guard, Stop, failure, no environment, the plan job).
- Window `ui/smoke/quantcheck.test.mjs`: the navigation, the explanation, the form, the disk table and its refusals, the choice that is sent, progress / Stop / failure, the result table, the chart and the words, the list actions, and the **Compare these sizes** button of Step 5.

## Deviations

- **The base file is about half of the brief's rule of thumb** (tokens x vocabulary x 2 bytes): only the second half of each chunk is stored (read from `perplexity.cpp`). The brief's figures (8,192 tokens = 4 GB) become 4,080 compared words = 2 GB. The default (16 chunks of 512) is the same as the brief's.
- **The chat-template text cannot keep its special markers.** `llama-perplexity` has no `--parse-special` (it tokenises with special parsing off: `common_tokenize(ctx, prompt, true)` in `perplexity.cpp`, and the option is registered for the imatrix tool only in `common/arg.cpp`), so `<|im_start|>` would be read as plain letters. The markers are taken out and the role names and line breaks stay. This is the same for every file, so the ranking is fair, but the text is a plain conversation, not the exact token sequence the model is used with.
- The comparison pass does not pass `-f` (the tool reads the tokens from the base file).
- A comparison's rows are saved after every pass in `results.json`; a stopped comparison is not "an error": its state is `stopped`.

## Not verified yet (needs llama.cpp, a real model, a GPU)

- Everything about a **real `llama-perplexity`**: that the flags `--kl-divergence-base`, `--kl-divergence`, `-c`, `--chunks`, `-ngl` mean what the master README and `common/arg.cpp` say on the user's build, that the comparison pass runs without `-f`, the exact output text of that build (the parser is lenient but was only tried on the format read from the source and the README table), and that exit code 0 after a failed pass is caught (the tests simulate it).
- **That the base file's size matches the estimate** for a real model (the formula comes from the source of one revision).
- The progress marks: the base pass prints `[1]`..., the comparison pass its per-chunk table; a build that prints something else shows only the heartbeat.
- That a 16-bit reference **made by the program** is the same file an export would convert (it uses the matrix library's merge + convert).
- Whether KLD on 4,000 compared words ranks the sizes the way a long test would: more chunks give steadier figures; the page shows the uncertainty only in the worker's log.
- Windows: killing `llama-perplexity` on Stop; paths with spaces or non-ASCII characters (the tests use stand-in tools).
- The held-out text as the model sees it with the markers taken out: whether the ranking of sizes changes with the marker handling.

## Follow-ups

- Show the uncertainty (`±`) per figure, and the Δp percentiles.
- A second text (for example the person's own questions) in one comparison.
- Compare KLD between two exports of different runs (the table already allows it, but the verdict assumes one model).
