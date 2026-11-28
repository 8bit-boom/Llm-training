# Upgrade transformers (one click, with a way back)

Inkling-2-Mini needs transformers 5.24.0 or newer (`crates/ndforge-core/src/reshape/moe_compat.rs`, `MIN_TRANSFORMERS`). The program installs it for you into **its own environment** (`envmgr::env_dir()`), with the installer
the environment manager uses (`uv pip install --python <env python>`). No account, token or key is involved: it is an ordinary package install.

Where: the Slim page's software check (a button next to the transformers line), the failure message of the jobs that need it (`needs the training software updated: transformers ...`), and the card
**Python packages** on the Training setup page (installed version, needed version, Update, Go back, Check again).

## What happens (`envmgr/transformers_upgrade.rs`)
1. The dialog shows the plan (`transformers_upgrade_plan`: package, target `transformers>=5.24.0,<6`, the environment's Python, the installed version, what stays working, that the environment changes). Nothing runs until **Confirm**.
2. `transformers_upgrade_start(confirm)`: `confirm` is a required camelCase argument (Tauri ignores unknown keys, so a mis-spelled one would be a silent no-op; `ui/smoke/invoke_keys.test.mjs` checks it). Refused with a plain message while a job (this includes an Auto run), a queue run or waiting queue entries are active; checked again right before the first change.
3. `uv pip freeze` gives the PyTorch (`torch`, `torchvision`, `torchaudio`) and `bitsandbytes` pins, written to `<env>/transformers_upgrade_pins.txt` and passed as a constraint next to the setup's own `constraints.txt` and `worker/constraints.txt`.
4. `uv pip install --dry-run` shows what the resolver would change. If it lists any of the pinned packages (or cannot solve with the pins), nothing is installed and the result says why. Nothing else is upgraded (no `--upgrade`).
5. The version being replaced is written to `<app data>/transformers_upgrade.json` (`{package, previous, target, python, at, state, installed}`); then the real install runs (progress as `env-event`, tagged `task: "transformers"`; Cancel is `env_cancel`).
6. Checks: `import transformers` reports the new version (>= 5.24.0), the pins are unchanged, the worker's `hello` job starts and sees the same version. The Slim page then runs `moe_compat` again, and the setup page re-reads its checklist.
7. A failed install, a failed check or a Cancel after the install started puts the recorded version back (`uv pip install transformers==<previous>`, same constraints, not cancellable) and checks it; the result says plainly whether that worked, with the manual command when it did not.
8. **Go back to <version>** (`transformers_rollback_start(confirm, version)`) installs the recorded version the same way (preview, torch guard, checks). `version` must equal the recorded one, so a stale button cannot act.

Results are `{ok, stopped, text, previous, installed, changed: [{name, from, to}], planned, rolled_back, rollback_ok, aborted?}`; a refusal before anything is touched is an error text. No worker job, event or error code was added.

## Not verified
Tests use a stand-in Python and a stand-in `uv` (`crates/ndforge-core/tests/transformers_upgrade.rs`): success, failure with rollback, failed check, failed rollback, cancel, refusal while active, PyTorch change abort, Go back. A real upgrade with the real environment on a Windows PC, the real `uv --dry-run` output format on the lines ` + name==x` / ` - name==x` (the parser skips what it does not know; the PyTorch comparison after the install does not depend on it), and the GPU self-test afterwards were not run here.
