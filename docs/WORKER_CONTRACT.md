# The worker's contract (for nd-world)

nd-world embeds the training worker (`worker/ndforge_worker`) and starts it the way this program does: `python -m ndforge_worker <job.json>`, one JSON object per output line (see `protocol.py`).
What a consumer may rely on is written down in **`worker/ndforge_worker/contract.json`**:

```json
{"protocol": 1, "jobs": ["..."], "events": ["..."], "error_codes": ["..."]}
```

- `protocol` is the number the `hello` job also answers (`{"worker", "protocol", "packages"}`). It changes only when a line of the protocol changes its shape. Check it, not the package version.
- `jobs`: the values of `"job"` the worker accepts. `events`: the `"type"` of every line it may write (`log`, `progress`, `result`, `error` and the job-specific ones). `error_codes`: the `"code"` of an error line.
- **Names are only ever added.** A removed or renamed job, event or code breaks nd-world; `worker/tests/test_contract.py` reads the worker's code and fails when a listed name has disappeared, and when a new one is not listed yet.
- Consume the file, do not copy its words: read `contract.json` from the embedded worker and refuse a protocol number you do not know.

The Unsloth side: `tests/fixtures/unsloth/*.json` and `docs/UNSLOTH_API_CONTRACT.md` are meant to be identical in both repositories. `tests/fixtures/unsloth/SHA256SUMS` records the fixtures (of the text with LF line endings);
a test checks it, so a change to a fixture is always a conscious one. The workflow `contract-drift.yml` compares nd-world's copies with these (weekly, and by hand) when the secret `NDWORLD_READ_TOKEN` (a read-only token for
nd-world) is set; without it the job says "skipped" and passes. Nothing in nd-world is changed by any of this.
