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
