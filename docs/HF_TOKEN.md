# The Hugging Face token (app-wide)

One token, set once, for everything that downloads from huggingface.co: gated or very large models (for example Inkling-2-Mini), a model added by name, a cut or grow that fetches
its source model, the export of a listed model, the healing teacher and student, the thinking check, the importance matrix and the general text.

## Where to set it

**Training setup** page, card **Hugging Face token** (it shows *saved* or *not saved*). A password field, **Save token**, **Remove token**. A read token is enough: make one at
huggingface.co/settings/tokens; for a gated model also accept its terms on the model's page with the same account. The General-text chooser's token box is the same token and
shows the same state (both call the same two commands, `hf_token_set` `{token}` and `hf_token_clear`, both return the `general_state` with `has_token`).

A token is accepted only if it starts with `hf_` (`general::token_set`). The value is never shown again after saving.

## Where it is kept and how it travels

| Where | What |
|---|---|
| `secrets.json` in the settings folder (key `huggingface`, `general::TOKEN_SECRET`) | the only place it is stored (not encrypted; mode 0600 on Linux) |
| environment variable `HF_TOKEN` of the worker process | every job except `envmgr::jobs::NO_HUB_JOBS` (`hello`, `selftest`, `mixed_plan`, `quantfit_estimate`, `llama_offload_tune`, `imatrix_dat`: they never reach the Hub). `huggingface_hub`, `transformers` and `datasets` read `HF_TOKEN` by themselves, so `snapshot_download`, `hf_hub_download`, `from_pretrained`, `HfApi` and `load_dataset` in the worker all use it |
| `Authorization: Bearer ...` header | the program's own requests to the Hub: adding a model by name (`train::hub::lookup`) and the revision check of the Inkling cut (`reshape::moe_compat::fetch_repo_state`). Only to huggingface.co, or the address `NDFORGE_HF_URL` moves it to (tests) |
| vLLM Docker container | `docker run -e HF_TOKEN`: the NAME only, Docker takes the value from the worker's environment, so it is on no command line and in no log. For `docker_wsl` the worker adds `HF_TOKEN` to `WSLENV` (unverified on a real Windows + WSL machine). The `wsl` and `wslc` kinds and a vLLM server that is already running get nothing |

It is never in a job file (`job.json`), a job log, a `heal.json` or any other record, a result, the settings file, a problem report or a diagnostics bundle. Defence in depth:
the Rust side replaces every stored secret and every `hf_...` pattern in each log line, error text and report (`secrets::redact`, `unsloth::client::scrub`); the worker's
`protocol.scrub` replaces `hf_` + 20 letters or digits and the exact value of `HF_TOKEN` in every `log` and `error` line. Tests with a made-up token (`hf_FAKETOKEN123`):
`crates/ndforge-core/tests/hf_token.rs` (job file, events, error text, job logs, problem report, diagnostics bundle, secrets file), `worker/tests/test_hf_token.py`,
`ui/smoke/hf_token.test.mjs`.

## When Hugging Face says no (401 / 403)

The worker (`ndforge_worker/hf_access.py`) recognises a refusal of the Hub (`GatedRepoError`, a response of 401 or 403, "Cannot access gated repo", "Access to model ... is restricted",
"Invalid credentials in Authorization header") and answers with one plain message under the job's own error code: the model is gated or private, or the token is missing or not valid,
and where to fix it (the Hugging Face token card on the Training setup page). With a token saved the text says the token was not accepted (invalid, or no access to this model yet);
without one it says none is saved. `protocol.error` applies the same wording to any error text that is a raw Hub refusal, so no job can show a bare traceback line for it. The program's
own lookups (add a model, Inkling revision check) give the same advice (`general::hub_refused`). A 404 (a misspelled name) is not taken for a refusal. **No new error code**:
`contract.json` is unchanged.

## Not verified

Nothing here was run against the real Hugging Face (no token was available): the exception shapes are those of `huggingface_hub`, tested with stand-ins; the `docker -e HF_TOKEN`
path and the WSL forwarding were tested as command lines only.
