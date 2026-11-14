# Try before cutting ("Try first" on the Slim-or-grow page)

Before a new model folder (tens of GB) is written, the page can tell what a cut will cost. The original is loaded once, in memory; a cut is **imitated with hooks**; the
project's own calibration text is run through it; the text-prediction error ("loss") is the same number as the `loss_before` / `loss_after` of a made model. Nothing is
written except one small file, `reshape_try.json`.

## What it answers (owner decisions = the defaults)

1. **Baseline:** the loss of the original on the calibration batches (same data, samples and load mode as measuring: 4-bit for models that do not fit in 16-bit).
2. **Depth, per block:** for every block of 4 layers except the first and the last (never candidates, as in the existing rules; 14 of 16 for the 27B) the loss with only that block skipped: loss, change, relative rise, perplexity, and the program's influence score (the block-boundary distance of the measurement) next to it.
3. **Depth, runs:** for the run length on the size slider, the loss of every window of that many neighbouring blocks, the cheapest window, and the program's default choice for comparison ("the default costs X, the cheapest run costs Y"). Asked on demand, kept per length.
4. **Preview my choice:** one pass for any set of blocks (clicked in the chart or typed as layers in Expert mode). Two blocks cost more than the two single costs added up; this pass is the honest number.
5. **Width:** for each MLP width the slider offers (17,408 to 13,312, 12,288, 10,752 ...), the loss with the neurons below the cut switched off: a quality-against-size curve. It needs the width measurement; without it the card offers *Measure first* (the existing job). **Both:** one pass for the chosen blocks and width together.
6. **Verdict** in plain words with a table and charts: the rise of the error and the perplexity (e to the loss), colour bands by the relative rise. A warning when the two halves of the text disagree about the ranking.
7. **Use this choice** fills the Make form (layer list, width, or both) so exactly what was tried is cut.
8. **Cost before running:** the number of passes and a time estimate. Quick (64 samples) or Full (128); Quick is the default for models of 15 B or more. Progress per pass; Stop keeps the passes done.
9. **Kept answers:** `reshape_try.json` in the model's measurement folder (`models/_scores/<key>/`); a repeated Try skips the passes it has.

Scope: dense Qwen3.5 / 3.8 (the families the Slim page supports for width). Gemma and quantized checkpoints keep the existing refusal.

## How it works

- **A pass** is one forward pass of every calibration batch with a cut imitated: the layers of a skipped block get a forward that returns its input (`use_cache=False`, so no attention or recurrent state is needed); the MLP neurons a width cut removes are multiplied by zero in front of `down_proj`, using the neurons the written model would keep (`reshape_width.keep_top` on the measurement's scores of the kept layers). The existing torch tests prove that a thinned or pruned model equals the original with those structures switched off; `worker/tests/test_reshape_try_torch.py` proves that the predicted loss equals the calibration loss of the folder the `reshape` job writes (below).
- **A pass key** names the question: `d=<blocks>|w=<mlp width>` (`d=|w=` is the original). Single blocks, windows, widths and combinations are all passes, so a window of one block, a typed choice that equals a window and a window asked for twice are computed once.
- **Order:** the original, the person's own choices, single blocks, runs, widths. Results are saved atomically after every pass; the stop file (`models/_scores/STOP`, the same as measuring) is looked at before and between the batches of a pass; a pass that was running is dropped, the finished ones stay.
- **Halves:** the batches of even and odd position give two losses per pass (as the measurement's two halves). `stability` compares them for the blocks, for every run length and for the widths: the cheapest must be the same and at least 80 % of the pairs ordered the same way (`AGREE_PAIRS`).
- **Time:** `seconds_per_pass` is measured (mean of the passes of the file) and kept in the results; a plan before the first pass uses the measurement's own timing (`seconds_per_pass` in `scores.json`, written by newer measurements; it includes the hooks of measuring, so it overestimates a little) or, failing that, a formula (2 operations per parameter and token at a guessed speed per load mode: 40 / 15 / 2 TFLOP/s for 16-bit / 4-bit / offload) and says it is a guess. A timed pass is scaled by the tokens of the text.
- **Plan:** the card asks the worker (`reshape_try` with `plan_only`) how many passes are needed and how many are kept; nothing is loaded and torch is not needed.

### Colour bands (rules of thumb)

The rise of the error is `loss / original loss - 1`. The bands are the constant `BAND_EDGES = (0.03, 0.10)` in `worker/ndforge_worker/reshapetry.py`, the only place:

| Rise | Name | Colour | Rough meaning |
|---|---|---|---|
| under +3 % | small | green | usually a mild cut |
| +3 to +10 % | noticeable | amber | healing will have real work to do |
| over +10 % | large | red | a damaged model until healed |

They are **rules of thumb chosen by the owner, not measurements**: nobody has measured how the error before healing relates to the quality after healing for these models (see "Not verified yet"). The page and the help say this next to every verdict.

### The file `reshape_try.json`

`format`, `key` (source repo, weights fingerprint, data version, data hash, samples, max length, seed, requested load mode), `passes` (`{pass key: {loss, loss_h1, loss_h2, tokens, seconds}}`), `tokens`, `load_used`, `seconds_per_pass`, `analysis` (what the page shows: base, bands, per-block rows, runs with windows / best / default / stability, widths, other choices). Another key is recomputed and a log line names the reason ("another number of samples", "other weights of the model", ...). The weights fingerprint is a hash of `config.json` and the names and sizes of the weight files of a folder, or the commit of a Hugging Face snapshot. Loaded in another way than before (16-bit instead of 4-bit) also means new answers.

### Job and commands

- Worker job `reshape_try` (`worker/ndforge_worker/reshapetry.py`, dispatched in `__main__.py`, error code `try_failed`). Events: `progress` (step `try`), `try_pass` (key, loss, `done`, `total`, the whole `analysis`), `log`. Result: `analysis`, `passes_total`, `passes_done`, `stopped`, `seconds_per_pass`, `load`, `tokens`, `dir`.
- Rust `reshape/runs.rs`: `TrySpec`, `try_job`, `try_plan`, `try_run`, `try_results`; `reshape/plan.rs`: `default_try_samples`. Window commands `reshape_try_plan`, `reshape_try_results`, `reshape_try_start` (events on `reshape-event` with `phase: "try"`; Stop is `reshape_stop`).
- Page: `ui/app.js` (`tryCard` and the functions before `renderReshape`), smoke tests `ui/smoke/try.test.mjs`. In-app help: `crates/ndforge-core/help/reshape.md`.

## Tests

- `worker/tests/test_reshape_try.py` (no torch): keys, windows, choices, the order of passes, bands, rows, stability, cheapest window, the analysis, the cache key and its reasons, results file, time estimate, and the plan job through the real worker entry point.
- `worker/tests/test_reshape_try_torch.py` (tiny random Qwen3.5 model, 24 layers, CPU): **(a)** for five block sets the loss via skip hooks equals `_loss` of the folder the `reshape` job writes; **(b)** for two widths the loss via masks equals `_loss` of the thinned folder; **(c)** depth and width together; also the job's own `loss_before` / `loss_after` check; **(d)** the baseline equals the original's loss and the measurement's `loss_before`; **(e)** every window equals the same set asked for on its own in another folder and order, and the cheapest is the lowest; **(f)** a stop in the middle, then a second run, gives exactly the results of one uninterrupted run and runs only what is left; further: a pass leaves the model as it was, a repeated Try loads nothing, another key recomputes, a stop before the first pass, a width without a measurement is refused.
- Rust (`crates/ndforge-core/tests/reshape.rs`): defaults and refusals of `TrySpec`, the job's folders and keys, plan, run and results with a fake worker (unix only, like the other fake-worker tests).

## Deviations from the brief

1. **Width means the MLP width only.** The Expert axes (linear-attention value heads, attention heads) are switched off in the page (`EXPERIMENTAL_AXES_PASSED`) and are not tried; the card does not offer a preview for them.
2. **The answers live in the folder of the Try's own samples** (`models/_scores/<slug>-v<version>-<samples>x<len>-<load>/`), which is the measurement's own folder only when the Try and the measurement use the same number of samples (64 Quick against 128 Full against the width measurement's 256 by default: usually not). The neuron scores and the influence are read from the measurement's folder (`measure_dir`). This keeps Quick and Full answers side by side instead of one deleting the other.
3. **Single blocks are runs of length one** in the analysis (`runs["1"]`), so one table and one rule serve both, and the program's default for one block is shown.
4. **Rust does not check the weights fingerprint** when it shows kept answers (it checks model, data version and the folder's settings); the worker checks everything when it plans or runs, and drops answers of other weights with a log line.
5. **Needs transformers 5 or later** for the skipped layers (a decoder layer returns the tensor itself there); with an older one the job says so instead of risking a wrong answer.
6. **Comparability with a made model** holds on the same calibration: the original's loss from a Try with 64 samples is not the `loss_before` of a made model measured on 256 samples (the numbers differ by sampling noise, not by method). The torch tests use the same batches for both. A made model's own check always uses the page's samples.
7. **After an out-of-memory retry** (one text at a time) the two halves are by text instead of by batch; the total loss is unchanged.
8. **`seconds_per_pass` in `scores.json`** is written by measurements made from now on (an older measurement has none and the formula is used).
9. The step "worker, torch" and "UI + help" commits are ordered as asked; the Rust part was committed before the UI so the page can call it.

## Not verified yet (needs a GPU run)

Everything above is tested on tiny random models on a CPU (logic, not speed or quality). Not seen on a real model:

- the time per pass on a 27B in 4-bit, and how well the formula guesses it (the formula is a guess);
- memory: whether one pass fits next to the weights on the 32 GB card, whether the 4-bit layers accept the neuron mask hook (`Linear4bit` pre-hook sees the half-precision activation: assumed), and the OOM fallback;
- whether the 4-bit and the 16-bit rankings of blocks, runs and widths agree;
- **how well the loss before healing predicts the quality after healing.** This is the key question: the colour bands are rules of thumb until it is measured. The way to measure it: heal several cuts of one model (cheap, middle, expensive by this card) with the same budget and compare the healed loss and the Test page scores;
- how many samples are enough (Quick 64 against Full 128): how often the two halves disagree on real data;
- the window in a real Tauri window (the page was tested in a stand-in browser; the Rust side builds in the tests and the commands were written, not run in a window).
