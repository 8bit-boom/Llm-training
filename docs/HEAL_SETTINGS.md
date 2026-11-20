# Healing settings and graphics memory

Healing (distillation, `worker/ndforge_worker/distilljob.py`) ran out of graphics memory on a ~16 B thinned Qwen3.x student on an RTX 5090 (32 GB), in step *train*:
"The graphics card ran out of memory ... (AcceleratorError: CUDA error: out of memory)". This page records what was found, what was built, the owner decisions
(the defaults) and what has **not** been verified, because it needs a real card.

## What was found (reading the code)

1. `gradient_checkpointing=False` was hard-coded in the `SFTConfig`. **Correction to the first suspicion:** `backends.load_hf` already switches checkpointing on
   in the model for LoRA on a graphics card, and `prepare_model_for_kbit_training` does it for QLoRA, and the Trainer does not switch it off when the flag is
   `False`. So checkpointing was probably already on in the failed run; the flag was misleading, not the cause. What was missing is that the setting was neither
   visible nor controllable. It is now explicit in both directions (`_apply_checkpointing`: on = non-reentrant, `enable_input_require_grads`, `use_cache=False`; off = `gradient_checkpointing_disable`).
2. `DEFAULT_HYPER` (rank 64, 2048 tokens, batch 4 x 8) did not depend on the model or the card.
3. `HealSpec` could not set batch, accumulation, sequence length, checkpointing, optimizer, KL chunk, method, top-k, teacher load or warm-up.
4. `heal_plan` counted only weights (`estimate_vram`), not activations, the LoRA/optimizer state, the output layer's chunk or - the big one for Qwen3.5 -
   **the 32-bit copy of the embedding and the output layer** (248,320 x hidden each) that PEFT's kbit preparation makes in QLoRA (see "The 32-bit copy" below). At hidden size 4096 a 16 B student
   in QLoRA was therefore ~16 GB of weights, not ~9 GB; the copy is now off by default (~12 GB). This (with the long sequences and batch 4) is the most likely real cause; it is **not measured** (see below).
5. "Continue" resumed with the same settings. No retry, no `expandable_segments`, no `empty_cache` between steps.

## The settings (Heal card, Expert mode: "Healing settings"; the safe defaults apply without Expert mode)

(The same card heals any model chosen in the *Heal a model* section, not only a made one: docs/HEAL_ANY_MODEL.md. The sizes it needs come from the made model's record or, for another model, from the Check.)

| Setting | What it does | Default |
|---|---|---|
| Sequence length 512/1024/2048/4096/8192 | tokens per training text. Memory grows with it. 8192 is for long reasoning rows (the faunix set averages about 5,750 tokens); the estimate of a 16 B student with QLoRA, checkpointing and batch 1 is about 18.5 GB at 4096 and 20.6 GB at 8192 (shared cases in `worker/tests/data/heal_cases.json`, estimates not measurements). Changing it makes the original answer again (the cache is for one length) and restarts the learning. | 1024 from 15 B, else 2048 |
| Batch per step 1/2/4 | texts on the card at once. | 2 from 15 B (1 from 25 B), else 4 |
| Sequences per update | batch x accumulation; the accumulation is computed and shown. Keeps the learning the same whatever the batch. | 32 |
| Gradient checkpointing Auto/On/Off | makes layer activations again in the backward pass: much less memory, about a third slower. Auto: on from 8 B; below that on when the activations would take more than 60 % of the memory left after the weights; card unknown: on. | Auto |
| LoRA size (rank) 8-128 | size of the learned part. Changing it restarts the learning. | 32 from 15 B, else 64 |
| Method QLoRA 4-bit / LoRA 16-bit | how the student is held. Automatic: QLoRA from 15 B, and also when the 16-bit estimate is above 85 % of the card. | automatic |
| Optimizer 8-bit AdamW / AdamW | 8-bit needs `bitsandbytes`; without it AdamW is used (and logged). | 8-bit AdamW |
| Loss chunk 64-512 | tokens per piece when the 248 k-word output layer is applied for the KL loss. | 128 from 15 B, else 256 |
| Answers kept per token (top-k) 8-128 | the teacher's answers kept in the cache; changing it makes the original answer again. | 32 |
| The original is loaded as Auto/4-bit/16-bit | for the teacher steps. | Auto |
| Learning rate, KL weight, warm-up, tokens, way (cache/online) | as before (warm-up 30 steps). | as before |
| Keep the embeddings and output layer in 32-bit (QLoRA only) | the old PEFT behaviour; the box says how many GB more it needs (from the estimate). Changing it keeps the teacher's answers and the learning done so far. | off |

"Reset to safe defaults" puts all of these back to automatic. The card shows the estimate live (core command `heal_estimate`, one source for page and job).

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set at the top of the distill jobs (before torch starts) unless the person set `PYTORCH_CUDA_ALLOC_CONF` or `PYTORCH_ALLOC_CONF`
themselves; not on Windows (torch does not support it there and only warns). `torch.cuda.empty_cache()` runs after every saved checkpoint and every measurement.

## The estimate (`reshape::plan::heal_memory`) - an estimate, not a measurement

GB = (weights + LoRA/optimizer + activations + loss chunk + 1) x 1.10, with E = vocab x hidden:

* weights: LoRA 16-bit 2 bytes per parameter. QLoRA: 0.55 bytes per parameter for the 4-bit layers, plus the embedding and the output layer (2 E parameters, counted untied) as 2-byte values (default) or, with "Keep ... in 32-bit" on, as 4-byte floats.
* LoRA: rank x layers x 20 x hidden parameters x 16 bytes (weight, gradient, two Adam states; the 8-bit optimizer needs less).
* activations per token: 34 x layers x hidden bytes without checkpointing (Korthikanti et al., bf16, attention without a stored score matrix); with checkpointing 2 x layers x hidden + 34 x hidden.
* loss chunk: chunk x vocab x 16 bytes (logits, log-softmax, gradient, recomputation). 1 GB for the driver's context, 10 % for fragmentation and what is not counted.

Example, 16 B student (40 layers, hidden 4096, vocab 248,320), QLoRA, rank 32: defaults (1024 tokens, batch 2, checkpointing) = about 17.5 GB with the tables in 16-bit (about 22 GB with the 32-bit copy), "fits" on 32 GB; the old defaults
(2048 tokens, batch 4, no checkpointing, 32-bit copy) = about 73 GB at rank 64 (the activations alone are 45 GB), "will probably run out of memory". Words on the page: *fits* (up to 85 % of the card),
*tight* (up to 100 %), *will probably run out of memory* (more); healing is refused only for the last one. The teacher's step is estimated as before (4-bit, 0.6 bytes per parameter + 2 GB) and
the heavier of the two steps counts (both together when they fit: "online").

## The thinking recipe and these settings

With **Keep thinking** ticked (Qwen3.5 / 3.8 students) the automatic values change: sequence length 8192 when the estimate fits (else 4096, 2048, 1024), batch 1, LoRA size 64; 40 or 80 million tokens
(`plan::thinking_knobs`, `thinking_tokens_m`; docs/RESHAPE.md has the table). What is chosen in Expert mode wins, and a sequence length or LoRA size that differs from the stopped run's still makes the
old checkpoints go (`train_key`). The out-of-memory ladder below never changes the sequence length or the rank, so a recipe run that is too big ends with the plain message and the choice to pick 4096 or 2048.

## Out-of-memory recovery

In the worker (`healmem.run_with_oom_retry`): a torch `OutOfMemoryError`, or a `RuntimeError` / `AcceleratorError` whose text says "out of memory" (not Python's `MemoryError`) during the training step frees the model and
trainer (`gc` + `empty_cache`), loads the model again and goes on from the last checkpoint with the next safer setting, at most 3 times. Order (none of it changes what is learned: the sequences per update stay):

0. only when the 32-bit embedding copy is on (QLoRA): turn it off ("the 32-bit embedding copy turned off"; the model is loaded again without it and goes on from the trainer's last checkpoint; the maths hardly changes because the tables are frozen);
1. gradient checkpointing on; 2. half the batch, twice the accumulation (down to 1); 3. half the loss chunk (down to 64).

Each retry writes "Out of memory: trying again with ..." to the log and a `heal_retry` event; the settings in use are a `heal_settings` event. When everything is used up the error says which step, the numbers used,
what was tried, and what to change next in this order: shorter sequence length, QLoRA (if not already), lower LoRA size, smaller model. The page then shows that text, the retries, and **Try again with safer settings**,
which fills in the next rung (`plan::heal_next_safer`, the same rules as `healmem.next_safer`; both are tested against `worker/tests/data/heal_cases.json`) instead of a blind Continue. Other failures keep "Continue".
Settings that change the maths (sequence length, rank, method, top-k, learning rate, tokens, warm-up, sequences per update) are kept in `heal.json` as `train_key`; if it differs the old checkpoints are deleted and the learning starts again, and a changed sequence length or top-k makes the teacher answer again
(`cache_complete` checks `k` and `max_len`). The teacher's answers are **kept** when only student-side settings (batch, accumulation, checkpointing, optimizer, loss chunk) change; a change of the teacher's loading (`auto` or `4bit`) or of the amount of text makes it answer again (see `rows_key` below).
An out-of-memory error in the teacher step or while loading gets its own plain message with the next actions.

## Deviations from the brief

* `heal_memory_gb` takes a struct (`HealDims`, `HealKnobs`) instead of eight loose arguments (and also the vocabulary and the loss chunk).
* Cause 1 was wrong as described (see above).
* The 8 B student next to a 4-bit teacher no longer fits "online" on 32 GB by the new estimate, so the older 8 B test cases use a 4 B student for the online path; the 8 B case is "cache" mode.
* `HealSpec.rank` is now `Option<u32>` (None = by size); older saved specs with a rank keep it. The heal tests for the 21 B student now expect rank 32 and 1024 tokens.
* `PYTORCH_CUDA_ALLOC_CONF` is set in the Python job entry only, not in the Rust spawn.
* The retry reloads the model in the same process; a fresh process would be more robust against a CUDA context left in a bad state by an `AcceleratorError`. "Try again with safer settings" on the page is the fresh-process route.
* The earlier note here, "about 8 GB" for the 32-bit copy, was too high: it was the size of the 32-bit tables (2 x 248,320 x 4096 x 4 bytes = 8.1 GB), not what the copy costs on top of 16-bit. The real extra is 2 bytes per value: 4.1 GB at hidden size 4096, 5.1 GB at 5120 (both tables), see below.
* The brief expected a Python twin of `heal_memory_gb` in `healmem.py`; only the activation estimate existed there. `healmem.heal_memory_gb` was added (the same arithmetic as `plan::heal_memory`) so that the new "memory" cases in `heal_cases.json` are checked on both sides.
* The setting only exists for QLoRA: for 16-bit LoRA there is no kbit preparation and no copy, so `heal_choice` forces `upcast` off there (the box is greyed out) and the estimate ignores it.
* The "saves about X GB" log line counts every non-4-bit 16-bit matrix that the PEFT cast would have doubled (the two tables and any other weight bitsandbytes leaves alone), at 2 bytes per value; a tied table is counted once. The estimate counts both tables as if untied (a tied model needs less).

## The 32-bit copy of the embeddings and the output layer (follow-up)

**What the cast does.** Healing with QLoRA loads the student in 4-bit and then called PEFT's `prepare_model_for_kbit_training`. That function (1) freezes every parameter, (2) turns every 16-bit parameter that is not
4-bit into 32-bit floats (`param.data = param.data.to(torch.float32)`), and (3) switches on gradient checkpointing and input gradients. Step 2 includes the input embedding table and the output layer (`lm_head`),
which bitsandbytes does not quantize, and the small parameters (norm weights, per-head decay vectors).

**The real numbers** (Qwen3.x with a 248,320-word vocabulary and hidden size 5120, untied tables; check `tie_word_embeddings` in the model's config): one table is 248,320 x 5,120 = 1.27 billion values =
2.54 GB in 16-bit and 5.09 GB in 32-bit. Both tables: 5.09 GB in 16-bit, 10.17 GB in 32-bit, so the cast costs **5.09 GB more** (5.6 GB with the estimate's 10 % margin). At hidden size 4096 the extra is 4.07 GB
(4.5 GB with the margin). Not "about 8 GB".

**What is done now.** `backends.prepare_kbit_keep_dtype` (the "light preparation") freezes everything, makes only the parameters with one dimension or none 32-bit (norm weights, biases, per-head scalars, for stability),
keeps every matrix that is not 4-bit (the two tables and any other) in the 16-bit type it was loaded in, switches on gradient checkpointing (`use_reentrant=False`) and `enable_input_require_grads()`, and logs one
plain line with the real saving ("Keeping the embeddings and the output layer in 16-bit: saves about X GB"). The tables are frozen while healing (only the LoRA matrices learn), so the 32-bit copies buy nothing for the
KL loss, which is computed in chunks from the hidden states and the output layer's weight (`distilljob.chunked_kd_loss`). To keep that safe with 16-bit tables, every product with the output layer is now
`(h.to(weight.dtype) @ weight.t()).float()` and the log-softmax, top-k and the leftover probability run in 32-bit (`_chunk_losses`, `_teacher_answers`, `_teacher_targets`, `calib_loss`, `topk_from_logits`).

**The setting.** `HealSpec.kbit_upcast` (None = off) goes to the worker as `distill.kbit_upcast`; `backends.load_hf` takes `cfg.get("kbit_upcast", True)`: True is the old behaviour (normal training, `trainjob.py`, is unchanged and
does not pass it), False uses the light preparation. The Heal card's Expert section has the box "Keep the embeddings and output layer in 32-bit (the old behaviour: about X GB more memory)", unchecked, X from the estimate.
The estimate counts the 32-bit copy only when it is on (`HealKnobs.upcast`, shared cases "memory" in `heal_cases.json`).

**The ladder.** When it is on, the first automatic step after an out-of-memory error is "the 32-bit embedding copy turned off" (the model is loaded again without it; the trainer's checkpoint resumes). "Try again with safer settings" does the same on the card.

**The keys.** The setting is student-side numerics of frozen weights only. `heal_train_key` does not contain it and the teacher's cache check (`cache_complete`: general text key, `k`, sequence length) does not either, so **changing it keeps the saved teacher answers and the checkpoints**
(tested: `the_32_bit_embedding_copy_reaches_the_job_and_changing_it_keeps_the_teachers_answers_and_the_checkpoints`). A run that was started with one setting and continued with the other has LoRA matrices trained against slightly different frozen tables; that is the same kind of change as a checkpoint resumed after the retry ladder's other steps, but it is not bit-identical.

**Normal QLoRA training** (`trainjob.py`, `load_hf` with the default `kbit_upcast=True`) has the same cast and may get the same opt-in later. It was **not** changed now: it needs a GPU run and it is not what failed.

**Tests (CPU, tiny bf16 model of the real architecture, no bitsandbytes):** `worker/tests/test_kbit_keep_dtype.py` (frozen; tables stay bf16; one-dimensional parameters fp32; checkpointing and input gradients on; the logged saving equals the independent sum and equals the extra bytes of PEFT's own function on a copy;
a LoRA on the light-prepared model gets non-zero gradients from the KL loss; `load_hf` picks the preparation by `kbit_upcast` with the loaders replaced; the retry reloads without the copy). The first-step KL with 16-bit tables is within 1 % of the one with PEFT's 32-bit cast
(measured 0.003 % to 0.14 % over ten student/noise combinations under bf16 autocast, as the trainer runs); the test's comment states the measurement and why 1 %.

## Not verified yet (needs a GPU run)

* **The new settings:** the real memory saved on the RTX 5090 by keeping the tables in 16-bit (the arithmetic says 5.1 GB at hidden size 5120; the card has not measured it), whether the out-of-memory error of the 16 B student is gone with it, whether training behaves the same (loss curve, final KL) with 16-bit frozen tables on the real 4-bit model, and the speed (no per-chunk 32-bit-to-16-bit cast of the output layer, so possibly faster; not measured). `prepare_kbit_keep_dtype` itself was only run on a CPU without bitsandbytes: that the real `Params4bit` layers are left alone by it (it checks the class name, as PEFT does) was not run with the real library.

* The real memory use of a 16 B student on an RTX 5090, and whether the defaults (and the estimate's 85 % / 100 % thresholds) are right at 24 and 32 GB. The estimate's constants (34 x hidden per token, 16 bytes per LoRA parameter, 0.55 for 4-bit, the 10 % margin) are rules of thumb, not calibrated against a card.
* Whether the failure was the activations (as the estimate says) or the 32-bit embeddings, and whether it really was checkpointing already on.
* The speed cost of checkpointing (about a third by experience), of batch 1, and of the 8-bit optimizer.
* Whether `expandable_segments` helps here.
* That an in-process retry after a real `CUDA error: out of memory` (AcceleratorError) recovers; the tests raise a fake `torch.OutOfMemoryError` on a CPU.
* That a checkpoint saved with one batch size resumes cleanly with the halved batch on the real model (it does on the tiny CPU model in the tests).

## The key of the teacher's rows (`rows_key`)

`cache_complete` also compares `rows_key`: a fingerprint of the training file (path, size, modification time), the amount of text (tokens), the teacher and how it is loaded. The worker stores it, with `teacher_load`, in the cache's `index.json`; it is not part of `distilljob.cache_matches` (the rows' own hash is). Changing the amount of text, the file or the teacher load therefore makes the teacher answer again instead of reusing answers for other rows. A cache made before the key existed has none and is judged as before.
When the teacher is a **local folder**, the key also counts its weight files (`files`: sorted name, size and modification time of `config.json` and every `*.safetensors`); the key of a Hub id is unchanged. A heal of a local teacher therefore sees one mismatch after the upgrade: `distill_cache` runs, finds the shared store complete by content, reports `reused` within minutes and rewrites the model's view with the new key.

## The shared teacher cache

The heavy part files of the teacher's answers live in `<project>/models/_teacher_cache/<id>/` (job keys `shared_cache: true`, `shared_dir`). The worker chooses the folder: `<id>` is a hash of the
teacher (folder and config), the tokenizer, `k`, the length, how the teacher is loaded, the general text **without its share** (the share is the budget), the seed, the model family and the training
file. Next to the parts the folder keeps `rowhashes.npy` (the sha1 of every answered row's token ids). A run whose rows start with the rows already answered (or are the start of them) uses that
folder and only asks the teacher for the rows that are missing; when the first rows differ it gets `<id>-2` ... and the first folder is never touched, because another model may still read it.

**Which weights (id version 2).** The id and the index also carry `teacher_weights` (`safetensors_io.weights_manifest`): `hub:<commit>` for a Hub snapshot, else a sha256 over `config.json` and, for every weight file, its name, size, the hash of its header and the hash of 16 blocks of 64 KiB spread over its data (about 11 MB read for a 27 B model, under a second on an SSD). File times are not used, so copying a folder keeps the saved answers. Limit: it catches a replaced checkpoint, not a single edited byte. `store_extends` and `cache_matches` compare it whenever both sides have it. Answers saved under the version 1 id (before the weights were recorded) are adopted once, with the note "These answers were made before the program checked the original's weight files; if you replaced them in place, delete the saved answers.", and from then on carry `teacher_weights`; a store whose index names other weights is not used. The Try results (`weights_sample` in their key) and the measurement of the neurons (`weights` in `scores.json`) use the same identity; files saved without the field are accepted.

The model's own `distill/cache/` keeps `index.json` (the parts its rows need, `store` = the shared folder, relative to it), `rows.npz` and nothing heavy, so the student and the program's skip
decision work as before. A cache an older version made in the model's own folder is still read; a partly made one is continued as before. The shared folder goes when no model points at it any more
(a model was deleted) and, for Auto-cut, at the end of the run (docs/AUTO_CUT.md). The disk check before a heal still counts the whole cache (it does not know what is already there).

## How fast the teacher answers

The teacher pass (`distilljob._teacher_answers`, `run_cache`) answers a window of 64 rows longest first, so a batch holds rows of similar length and wastes little on padding, and writes the rows to
the cache in their own order. Inside a row the top-k work is done chunk by chunk as before (the logits of one chunk bound the memory); the three result arrays of a row come back to the host
once, not once per chunk. The arrays are identical to the old ones (a test compares them). The log has a `cache_speed` line (tokens per second) about every minute and at the end, and the job
result has `tokens_per_s`. A Stop that comes while a window is being answered keeps the rows that were finished from the first one on without a gap; the others are answered again on Continue.
The speed-up on a 27B is not measured here.


## The speed knobs of Auto-cut (`docs/AUTO_CUT.md`, "Speed settings")

Auto-cut sets the heal's tokens per stage itself (`AutoSpec.stage_a_tokens_m`, `last_tokens_m`: 10 / 20 / 40 / 80 M; the recipe is 40 M for the first stage and 40 or 80 M for the last) and, with "same sequence length in
both stages", sends the shortest sequence length of the stages to every heal (`HealSpec.seq`), because the original's saved answers are shared only between heals on the same sequence length and way of loading the
original (`distilljob.shared_store_id`: `max_len` and `load` are in it). Then the second stage writes only the tokens it has more. A retry after a failed thinking check heals the healed model again on a budget the
heal takes (the stage's tokens plus 40 M, rounded up to 10 / 20 / 40 / 80: 40 gives 80): the student learns on all of them, only the original's answers are extended. Not changed by any speed setting: the LoRA size, the
batch and accumulation, gradient checkpointing, the top-k of the teacher's answers (it is in the key of the saved answers), the general-text share.

## Healing in rounds (experimental)

The Heal card has **Heal in rounds and check after each (stop early when it is good enough)**: the heal is trained in rounds of 1, 2, 5 (recommended), 10 or 20 M tokens, the token choice then
reads "At most N million tokens" (the cap), the first check comes after 5, 10 or 20 M, the check is Quick or Full, and the stop rule is "the first passed check" or "never" (record the checks, to compare
with a normal heal). A round with fewer than 8 estimated steps is not offered. The card shows the steps and the overhead of a round, the best and the worst time next to a normal heal's, and a live table
of the rounds. After the cap without a pass it offers **Heal 10 M more** (the same run goes on). The rules, the schedule, the costs and what is not verified: `docs/HEAL_ROUNDS.md`.

## What the rows cover (the coverage report)

Every teacher pass (`distill_cache`) and every online heal writes `coverage.json` next to the cache's `index.json`, puts it in the result (`coverage`), in the training `summary.json`, in `heal.json` (the Heal card shows it under "General text used") and in the Auto-cut report (`stages[].coverage`). It counts, for the rows the heal really learns from:

* rows by kind: your world's texts and conversations, the general text's texts and conversations;
* conversations whole (the end-of-turn token closes the row), cut at the sequence length (a family without an end token: a row as long as the sequence length), and left out (the prompt alone does not fit, or a general conversation longer than twice the sequence length);
* reasoning rows (a non-empty `<think>` block; the empty block of a template with thinking off is not one): whole, answer lost, cut inside the reasoning, left out. The left-out ones are known only by their ids, so a family or tokenizer without the `<think>` markers reports no reasoning rows at all (`known: false`) and only whole / cut for conversations;
* the share of the kept tokens that is question, reasoning, answer and plain text (the split of a conversation is found from the ids of the assistant turn's start, `<think>` and `</think>`, else from the completion mask of your own conversations);
* the lengths of all rows before cutting (<=512, 1k, 2k, 4k, 8k, 16k, >16k), the left-out rows included.

`notes` says something only when more than 10% of the reasoning rows lost their answer or were left out: below a sequence length of 8192 it advises choosing 8192 in Heal settings if the plan says it fits; at 8192 or more it says the traces are longer and that a longer bucket is not available yet. A reasoning share of the kept tokens below 75% is mentioned as Unsloth's guidance for keeping thinking, a guide and not a rule. The numbers describe the rows; they say nothing about how well the healed model will think.

## Keeping the output layer's logits (`head_checkpoint`, Expert)

The KL loss makes the output layer's logits `chunk` tokens at a time and, by default, makes them again in the backward pass (`torch.utils.checkpoint` around every chunk: `distilljob.chunked_kd_loss`). Job key `distill.head_checkpoint`: `"on"` (default, and what a job without the key does) or `"off"`: every chunk's logits and log-probabilities are kept for the backward pass. The loss and the gradients are identical (a test compares them in 32-bit); only memory and speed change. Memory: about `tokens x vocabulary x 10 bytes` more, 20 GB for 8,192 tokens of a 248,000-word vocabulary, so `healmem.heal_memory_gb` / `plan::heal_memory` add that term (the shared cases in `heal_cases.json` check both) and the page offers the box only when the estimate says it fits this card. Not in `heal_train_key` and not in any cache key: changing it keeps the saved answers and the learning done so far. When the card runs out of memory with it off, the first step of the safer ladder (worker `healmem.next_safer`, Rust `heal_next_safer`) turns it back on. The worker reports `head_checkpoint: false` in its `heal_settings` event only when it was asked for; a default job's events and summary are unchanged. How much faster it is has not been measured here (the "Find the fastest loss settings" test of the Heal card is for that).

The chunk size was already a setting (`distill.chunk`, the Heal card's "KL chunk" 64 to 512, halved by the out-of-memory ladder).

## The held-out loss (and why the "calibration" loss is optimistic)

The text-prediction error the Heal card has always shown ("calibration loss": `distill_eval.calib_loss`, summary `calib_loss_before / _after`) is measured on rows of **your own training file** (`train.jsonl`), which is also what the student learns from. It is therefore **not held out**: it reads better than the model will do on text it has not seen, and it would favour a higher LoRA rank in a comparison. It is kept as it was (the numbers of older heals stay comparable).

New: `distill_cache` takes `holdout: {"rows": 64}` (the program sends it when the general text is on). The rows are read from the general text **past the rows the heal learns from** (never a later window of a text a training row came from), rendered and cut the same way, saved as `holdout.npz` in the cache folder, and the original's own error on them is measured while it is loaded (`holdout.json`, `teacher_loss`). The training job's `eval.holdout` makes `distill_eval` carry an additive `holdout_loss` (before, during, after) and the summary `holdout_rows`, `holdout_loss_before`, `holdout_loss_after`, `teacher_holdout_loss`. The Heal card draws it as its own line and says what it is; the Auto-cut report has the three numbers per stage. Limits: it is general text, not your world's (all of your own rows are trained on, so there is none left to hold out); a cache that was complete before this existed has no held-out rows (the teacher pass is not run again for it); and a gain on these rows says nothing about thinking.


## Make the original's answers with vLLM (experimental)

Heal card, below "Heal in rounds": **Make the original's answers with vLLM** (only shown when Settings, vLLM engine, switch it on; spec key `vllm: {"teacher_cache": true}`, absent otherwise: a heal without it is byte-identical). The card shows which weights the engine will use and why (your original's own folder in 4 bits, or a confirmed pre-quantized copy), the time with the standard engine and with vLLM from **your own measured numbers** (start-up from the self-test, scoring speed from the teacher check; the report's estimates, labelled, where there are none), the warning that a heal in rounds starts the engine every round, and the verdict of **Check the vLLM teacher** (100 rows against the standard teacher; `different` refuses the option). `heal.json` records `teacher_engine` (engine, teacher quantization, start-up, speed, fallback). The cache key (`rows_key`) includes the engine and the quantization, so the two kinds of answers are never mixed. Preflight: engine found, card free, disk for the model, route, the saved check. See docs/VLLM.md.
