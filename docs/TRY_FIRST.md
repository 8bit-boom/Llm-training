# Try before cutting ("Try first" on the Slim-or-grow page)

Before a new model folder (tens of GB) is written, the page can tell what a cut will cost. The original is loaded once, in memory; a cut is **imitated with hooks**; the
project's own calibration text is run through it; the text-prediction error ("loss") is the same number as the `loss_before` / `loss_after` of a made model. Nothing is
written except one small file, `reshape_try.json`.

## Loss is not a thinking test

Try first measures how well a cut predicts text. For a model that thinks (Qwen3.5 / 3.8) that is NOT the same as still being able to think: in a real run (docs/REAL_RUN_17B.md) the cut with the
lowest error finished its reasoning in half of the answers, against 97 % for the original. The verdict therefore always says so and what to do: make the cut, then run "Check the cut model" in the
Thinking check before healing. A choice whose error is *lower than the original's* is flagged: a cut that predicts text better than the model it came from is a reason to check that both were
measured the same way (a 4-bit or offloaded load against a 16-bit one, calibration text that suits the cut). Auto-cut does the thinking check for its best cuts by itself (docs/AUTO_CUT.md, "The
thinking check before healing").

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

## The automatic search ("Find the best cut for me")

The first option of the card. For a size (parameters, the size of the Q4_K_M file, a share to keep, or "the best trade-off") the program decides which cuts to measure, measures them in **one** model load and ranks them.

**Where it lives.** It is a key of the same job: `reshape_try` with `want.search` (`reshapetry._search_loop`, no new job or event type, no contract change). The rounds are planned by `worker/ndforge_worker/reshapesearch.py` (pure Python: from the passes already measured and a context it says which passes come next, deterministically; a stop and a repeat goes on where it was). The ranking, the confidence, the confirmation and the thinking check are in `crates/ndforge-core/src/reshape/search.rs`. Every pass uses the ordinary pass keys (`d=3,7,11|w=10752`) in the ordinary `reshape_try.json`, so the passes are shared with the manual card and with Auto-cut, and a repeated search costs nothing for the passes it has.

**The target.** The target size fixes how many blocks go together with which MLP width (`auto_plan::candidates`, inside the guardrails: the width-first floor and at most 4 blocks; Expert can allow more). What is unknown is *which* blocks. Trade-off mode looks at five sizes (90, 85, 80, 76 and 70 % of the parameters; not below 20 B for a model that thinks).

**The rounds**, in the order that keeps the most useful answers when the time runs out:

1. `singles`: the original and every block that may be cut, on its own (about 15 passes for 16 blocks).
2. `width_refs`: the widths of the cuts, with no block removed.
3. `sets_k`: for every number of blocks k of 2 or more: the six best sets by an additive prediction (the original plus the sum of the single increases, over the best k + 3 singles), the two best neighbouring runs by the same prediction and the program's own choice from the measurement. Only depth.
4. `combined`: the best three sets of each k at the width that k's target size needs: the cuts that could really be made.
5. `refine1`, `refine2`: around the best two cuts, one block swapped for one of the best unused singles; the six best predictions are measured.

About 50 to 75 passes for a 27 B model. "Every neighbouring run too" (Expert) adds the older sweep. The finalists are the best three, kept diverse: at most two with the same number of blocks, and a set that differs from a better one by a single block gives way to the next.

**The time.** One pass takes about 16 s for a 27 B in 4-bit (a guess until this computer has measured one; the plan line says "an estimate") plus loading once. Before every pass the worker checks that the elapsed time plus one more pass fits the minutes given (10, 20, 60 or your own number, 5 to 240); if not it stops with `stop_reason: "budget"` and everything measured is ranked. The other reasons are `done`, `stopped` (Stop keeps everything) and `max_passes` (at most 400). The plan line says how many passes fit ("40 fit the 20 minutes: rounds 1 to 2").

**What you see.** While it runs: "Round 3 of 5 (sets of blocks) · pass 34 of about 72 · up to 9 min left", a bar and a table of the best ten cuts so far (blocks out, MLP, the change of the error with its colour band, whether the two halves of the text agree). At the end: the best cut, a reason in plain words, how sure the call is (**clear**; **a close call**: the best two differ by less than the larger of their noise, the noise being the difference of the two halves of the text; **uncertain**: the halves name different winners), guardrail flags, and:

- *Use this cut in Auto-cut* (the cut is pinned on the Auto-cut card and Auto-cut does not search again),
- *Cut it now* (fills the Make form with exactly these blocks and the width),
- *Cut and heal it (Auto-cut, one stage)*.

A close call is asked again with 128 samples (one load, a few passes at twice the length) when the first answer used fewer; the new numbers decide among those cuts. In trade-off mode the recommendation is the smallest size whose best cut raises the error by at most 10 % (the second colour edge) and stays inside the guardrails: a **rule of thumb**, labelled so.

**The thinking check of the best two** (for a model that thinks; on by default when the disk allows, or by the button "Check the thinking of the top 2"): the program writes each of the two cuts, checks the original once and the cut, and deletes the cut again. One cut is on the disk at a time; nothing is kept. A stop in the middle continues with the cut that was written (`models/_search/<id>/search.json` keeps it as `pending`). When the check has run, `rank_thinking` decides between the two (the error breaks a tie); until it has run, the answer says "Thinking not checked".

**What it can and cannot know.**

- It measures the **text-prediction error on your own text** (with reasoning text when the model keeps thinking), with the cut imitated by hooks; the torch tests prove that equals the cut that is written. It does **not** measure the quality after healing.
- For a model that thinks, **a low error does not prove it can still think.** In the owner's own run one cut had 16.9 % less error and closed its reasoning in 0.50 of the answers against 0.97. Only the thinking check decides, and a result without it says so.
- With about 70 candidates the winner is partly luck (the winner's curse): the noise column, the halves check and the confirmation with more text show how close it is. A confident-looking first place can be a coin toss.
- The estimates (time per pass, passes that fit) are estimates until this computer has measured them.

**Auto-cut.** With `search_minutes` (the speed presets set it: Thorough 30, Balanced 20, Fast 10; none for a spec saved earlier) the step `try` is this search for the one target size (loss only), writes its best three cuts as the candidates of the thinking check before healing (`AutoState.try_cands`) and chooses the lowest error. With a pinned cut (`AutoSpec.cut`) there is no search and no thinking check before healing. With two stages the searched blocks are cut exactly when stage 1 takes out all of them; otherwise a later stage measures again on the healed model, as before.

Code: `reshapesearch.py`, `reshapetry.py` (`_search_loop`, `search_summary`), `reshape/search.rs`, `runs.rs` (`TrySpec.search`), commands `reshape_search_plan`, `reshape_search_start`, `reshape_search_status`, `ui/app.js` (`trySearchBlock`). Tests: `worker/tests/test_reshapesearch.py`, `test_reshapetry_search.py`, `test_reshape_try_torch.py` (the search on a tiny model), `crates/ndforge-core/tests/search.rs`, `ui/smoke/try.test.mjs`.

Not verified: the real time per pass and load on a 27 B; whether about 70 passes find a better cut than the 66-pass sweep of neighbouring runs; whether non-contiguous sets heal as well as neighbouring ones; how Quick and Full thinking checks agree on close calls.

## Ranking the neurons by their output (experimental, Expert)

Today the neurons of the MLP that stay are the ones with the largest score of the measurement: for every text the mean of the absolute activation over its tokens, squared, summed over the texts, square-rooted (`reshapejob._measure_all`). That score looks only at how much a neuron fires. The option `width_scoring: "output"` ranks them by the expected squared size of what each neuron adds to the layer's output instead: **the mean over all real tokens of the squared activation** (not the square of the score above, so it is a second accumulator that the measurement now always records: `mlp_ms`, `mlp_ms_h1`, `mlp_ms_h2` in `scores.npz`, `"mean_square": true` in `scores.json`) **times the squared norm of the neuron's column of `down_proj`** (read from the weight files with no torch, kept in `down_norms.npz` next to the measurement with the identity of the weights). The two halves of the text use their own mean squares, so the stability overlap is computed the same way.

* Default unchanged: `activation`. A job without the key, a spec without `width_scoring` and a Try pass key without a suffix are exactly what they were. Only `width_scoring: "output"` is sent to the worker; `ndforge_reshape.json` and the result carry `width.scoring` only then.
* A measurement made before this recorded no mean squares: the cut job measures again (logged) only when the output ranking is asked for, `reshape_measure` measures such a folder again, and Try first says "Measure the model again" instead of guessing.
* Try first compares the two on your text: a choice or a list of widths with `scoring: "output"` also tries the plain cut, the pass key of the variant has the suffix `|s=output` (`d=2|w=13312|s=output`), and the analysis has a `variants` list: the error, the change against the original, the change against the same cut by activation, and **whether it is better in both halves of the text**. One half alone can be luck, so the program calls only both a gain. The Try card and the Slim page say that nobody measured this here, that it is measured before healing only and may not survive healing, and that the citations behind the formula have not been checked by this program (it is algebra: the expected squared contribution of a neuron when the neurons are treated as independent).
* The model written by the cut uses the same scores, so the Try pass loss of a variant equals the loss of the written model (a torch test checks it on a tiny model).

## Refitting the kept neurons (experimental, Expert, off by default)

When a width cut removes MLP neurons, the layer's output changes by what those neurons used to add. Refitting the kept neurons' output weights by ridge least squares can give part of it back with no training. With `A` the activations entering `down_proj` on real tokens, `K` the kept and `R` the removed neurons and `W0` the original `down_proj`: find `D` so that `A_K (W0_K + D)^T` is close to `A W0^T`, i.e. `D^T = (G + lambda I)^-1 C` with `G = A_K^T A_K`, `C = A_K^T E`, `E = A_R W0_R^T`, `lambda = 0.01 x trace(G)/k`. Only `G`, `C` and `||E||^2` are kept (about 380 MB + 200 MB per layer at a width of 9,728 on the 27 B). The maths is in `worker/ndforge_worker/reshape_recon.py` (numpy twin tested against `numpy.linalg.lstsq`); it is standard algebra, **but whether it keeps more quality is not known for any real model**: it is off by default and Try first measures it before anything is written. A gain before healing may not survive healing.

* How: choices (or every width, with the Expert box) carry `reconstruct: "pruned" | "original"`; pass keys get the suffix `|r=pruned` / `|r=original` (after `|s=` when both are on; keys without a suffix are exactly the old ones). The plain cut is tried too. "pruned" (the default variant) fits every layer on what the already-cut model feeds it (exact for the written model); "original" fits it on the uncut model's activations. The refit runs after the plain passes.
* Layer-wise replay: one partial forward captures the input of layer 0 and the keyword arguments that each layer type is called with (linear-attention and full-attention layers have different masks and position embeddings); the hidden states then stay in CPU memory (about 2 GB per stream for 200,000 tokens at hidden size 5,120; the job refuses with a plain message when they would take more than half of the computer's memory) and each layer is run from there. Statistics are accumulated on the card in float32 (TF32 off), solved there with a Cholesky factorisation and a residual check; a failed check is solved again on the CPU in float64. The original `down_proj` is read from the weight files, so a 4-bit loaded model is refitted against the original weights (the activations come from the 4-bit model: a small inconsistency).
* Honest measurement: the statistics are kept per half of the text (batch parity). The refit is fitted on one half and scored on the other exactly from the statistics (`err = tr(D^T G2 D) - 2 tr(D^T C2) + ||E2||^2`); `removed_share = 1 - err/||E||^2` per layer, shown as the median and the worst layer, and the final fit uses both halves. The rows of the refit come from the END of the seeded shuffle (Try first, the measurement and the cut check take the head), the reasoning rows skip the ones the measurement used, and an overlap count by the hash of the token ids is shown (a warning when it is not 0 or the data is too small). The program-level measure is the Try loss of the refitted cut next to the plain cut, "better in both halves" as for the other variants.
* Output: the refitted `down_proj` of every layer, bf16 `[H, k]`, in a sidecar `<measurement folder>/recon/<tag>/` (`model.safetensors` + `recon.json` with the kept indices, the held-out shares, the overlap and the time); the tag is a hash of the weights' identity, the width, the ranking, the skipped blocks, the variant, alpha and the rows, and at most the two newest tags are kept (about 6.4 GB per tag at a width of 9,728 on the 27 B). A Try pass uses these exact numbers, so its loss is the loss of the model the cut would write with them.
* Time (a formula, not a measurement): the plan counts three extra passes over the text plus the statistics and solves per refit (`recon_passes`, `recon_seconds`); for the 27 B 4-bit on a 5090 a refit is guessed at 10 to 20 minutes per variant. Nothing has been run on a real card here; the owner's numbers decide.

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
