# The worker's contract (for nd-world)

nd-world embeds the training worker (`worker/ndforge_worker`) and starts it the way this program does: `python -m ndforge_worker <job.json>`, one JSON object per output line (see `protocol.py`).
What a consumer may rely on is written down in **`worker/ndforge_worker/contract.json`**:

```json
{"protocol": 1, "jobs": ["..."], "events": ["..."], "error_codes": ["..."]}
```

- `protocol` is the number the `hello` job also answers (`{"worker", "protocol", "packages"}`). It changes only when a line of the protocol changes its shape. Check it, not the package version.
- `jobs`: the values of `"job"` the worker accepts. `events`: the `"type"` of every line it may write (`log`, `progress`, `result`, `error` and the job-specific ones). `error_codes`: the `"code"` of an error line.
- **Names are only ever added.** A removed or renamed job, event or code breaks nd-world; `worker/tests/test_contract.py` reads the worker's code and fails when a listed name has disappeared, and when a new one is not listed yet.
- **Keys are only ever added, too.** A `progress` line may carry an additive `detail` (`{unit, done, total, rate, rate_unit, eta_s}`; without it the line is byte-identical to the older ones) and `reshape_try` takes an additive `want.search` (the automatic search; its `try_pass` events then carry `round`, `round_label`, `round_index`, `planned_total`, `budget_left_s` and `best`, and its analysis has a `search` key). No job, event or error code was added for them, so `contract.json` is unchanged.
- Consume the file, do not copy its words: read `contract.json` from the embedded worker and refuse a protocol number you do not know.

The Unsloth side: `tests/fixtures/unsloth/*.json` and `docs/UNSLOTH_API_CONTRACT.md` are meant to be identical in both repositories. `tests/fixtures/unsloth/SHA256SUMS` records the fixtures (of the text with LF line endings);
a test checks it, so a change to a fixture is always a conscious one. The workflow `contract-drift.yml` compares nd-world's copies with these (weekly, and by hand) when the secret `NDWORLD_READ_TOKEN` (a read-only token for
nd-world) is set; without it the job says "skipped" and passes. Nothing in nd-world is changed by any of this.

## The measured fit (added names)

New jobs `quantfit_estimate` (what the measured fit will do, from the header of a 16-bit GGUF only) and `quantfit` (measure, plan, optionally make the file and the plain sizes next to it; its result has `plan`, `report`, `rules`, `flags`, `files`, `measure_key`, `engine`, `notes`, `fits`), and the error codes `quantfit_failed` (a plain-language text) and `quantfit_stopped`; `llama_cpp_missing` is also used when a file is to be made and llama.cpp is not there. No new event. The export job also accepts the size id `mixed_measured` with the job key `mixed_measured` and answers with `mixed[<size>].measured`; the results of `quantcheck` gain the key `promotion` and the test text the kind `general`. Only additions: nothing was renamed or removed. See docs/MIXED_QUANTS.md ("Fit to a size (measured)").

## Healing in rounds (additive keys, experimental)

Healing in rounds (docs/HEAL_ROUNDS.md) trains a heal in short rounds and checks the thinking after each one. It needs no new job, event or error code, only new keys; `contract.json` is unchanged and a job without them behaves exactly as before.

- `distill_cache`: `upto_general_tokens` (answer the project rows and the general rows up to that many general tokens, then stop; the index stays `complete: false`) and `reuse_rows` (take the rows from `out/rows.npz` when the cache's own index says they are the rows of this run, without reading the general text again; needs `rows_key`). When the target rows are already answered the job returns before loading the teacher. Result keys `done_rows`, `general_rows_done`, `general_tokens_done`, `rows_before`.
- `distill`: `rounds` `{index, round_tokens, general_from, general_to, s_start, schedule?}` trains ONE round: the project rows and the general rows `general_from .. general_to - 1`, the optimizer and the scheduler continuing from the checkpoint of the previous round (HF's global step goes on from `s_start`). It writes `out/rounds/r<n>/adapter` and `out/rounds/r<n>/summary.json` besides `out/adapter` (the latest) and the result has `round {index, s_start, s_end, steps, tokens, calib_loss_after, complete}`. A partial teacher cache is accepted when it reaches `general_to`.
- `think_check`: `adapter` (a LoRA folder put on top of `model` without merging it; transformers backend only) and `double_quant` (a 4-bit base is loaded like the student is loaded for training). The cache key gets `adapter` (a hash of the adapter's content) and `double_quant` only when an adapter is given; without one the key is byte-identical to before.
