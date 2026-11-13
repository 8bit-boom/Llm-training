# Unsloth Desktop in ND Lore Forge

Unsloth Desktop is a native app (Windows, macOS, Linux) that runs Unsloth Studio, the training and chat interface, on your own computer. Lore Forge can send a saved
data version to it and follow the training, instead of training with its own worker. The calls both products use are in `docs/UNSLOTH_API_CONTRACT.md`.

## Owner decisions

1. Desktop is a **third "Where to train" option**, never the default. The worker on this computer stays the main trainer (it does two stages, full fine-tuning, continuing after a stop and exact memory figures).
2. The trained adapter is **copied back from Desktop's run folder only when Desktop runs on the same computer**. A Desktop on another computer ends with "the adapter stays there" in this version.
3. Experimental items (a GGUF made inside Unsloth, loading it for chat, an Ollama import) are **not built**; steps that need the real Desktop and a graphics card are marked below.
4. **No shared crate** with nd-world: Lore Forge has its own small client (`ureq`, already a dependency) and the contract document plus the JSON fixtures in `tests/fixtures/unsloth/`.

## What Unsloth Desktop is

Unsloth Desktop is the native app (Windows, macOS, Linux) of Unsloth Studio. It starts the same Studio backend and talks to it over HTTP, so the routes Lore Forge uses (OpenAI-shaped
`/v1/files` and `/v1/fine_tuning/jobs`, plus `/api/health`, `/api/train/hardware` and `/api/models/checkpoints`) exist in a Desktop of the same version. Things that matter here:

- It serves only **while the app is open**, and answers **on its own computer only** until its owner switches on *Settings, API, Remote & LAN, LAN access* (then plain HTTP on the network).
- API keys (`sk-unsloth-...`) are made in *Settings, API*, shown once, stored hashed there. Lore Forge cannot read an existing key and never looks at Unsloth's files: the owner pastes a key once.
- It updates itself, so the API under Lore Forge can change without notice: Lore Forge probes features, not version numbers.
- Unsloth keeps one training job at a time; a second one waits in a queue held in memory (lost when Unsloth restarts).
- Its default port is not documented for Desktop (Studio from a terminal uses 8888); "Find" tries 8888 and 8000 on this computer, and `NDFORGE_UNSLOTH_URL` adds an address first.

## What the integration does

1. **Training setup, card "Unsloth Desktop"**: address, Find, API key, Test (is it Unsloth, does the key work, Desktop or Studio, version, graphics card, does it have the training API), Save. The address goes to `settings.json`,
   the key to `secrets.json` under `unsloth:<address>` (the same file and permissions as the TrueNAS token). An address that is not this computer, not a private network and not https needs a tick.
2. **Step 3, Where to train, "Unsloth Desktop"**: the plan uses Unsloth's graphics card (`/api/train/hardware`). Press Train and Lore Forge: probes, sends the same chatml rows as "Export for Unsloth Studio"
   (`unsloth/chatml.rs`, guarded by golden files), starts the job (`unsloth/run.rs::job_body`, `load_in_4bit` always sent), follows status and events into the same charts and `metrics.jsonl`, shows the queue place,
   and on success copies the adapter and tokenizer files into `runs/<name>/adapter/` (atomic rename, free-space check) when Unsloth's run folder is on this computer. Test, Export and Retrain then work unchanged.
3. **Stop** cancels the job in Unsloth; that run cannot be continued. A lost connection (180 failed polls, 3 s apart) marks the run *interrupted*; **Continue** reattaches without doubling chart lines, fetches a result that
   finished meanwhile, or sends the data again when Unsloth lost a waiting job.
4. The state of a run in Unsloth: `state.json` has `backend: "unsloth"`, `unsloth_job`, `unsloth_events_seen`, `fine_tuned_model`, `adapter_remote`, `note`; `config.json` has `unsloth: {name, url}`; `unsloth.json` is the job object.
   No file of the run holds the key (tested), the problem report scrubs anything that looks like an Unsloth key.
5. Not available for Unsloth: full fine-tuning, two stages (world book first), a model folder on this computer (slimmed / healed / grown), continuing a stopped run. Each is refused with a message before anything is sent.

The worker (`worker/`) is not involved and was not changed; its job types and events stay as they are (nd-world installs it from this repository).

## Failure messages

The table is in [UNSLOTH_API_CONTRACT.md](UNSLOTH_API_CONTRACT.md) (not running, not Unsloth, LAN access off, key refused, too old, queued, queue lost, unknown parameter, out of memory, disk full, model folder not trainable).
The wording lives in `unsloth/client.rs::failure`, `unsloth/run.rs::explain_failure` and the run itself. Every text goes through `scrub`.

## Version and feature probing

`UnslothClient::probe` asks `/api/health` without a key (is it Unsloth, Desktop fields), with the key (a wrong key does not give 401 there: it gives the answer without `version`), `/v1/models` (401 on a bad key), then
`/v1/fine_tuning/jobs?limit=1`, `/api/models/checkpoints` and `/api/export/status` for the features. It is repeated at every run start. Version numbers are shown, never required. A new unknown job field is a 400
"Unknown parameter(s)": the stand-in server refuses unknown keys the same way, so a field added without checking fails the tests first.

## The contract the two products share

[UNSLOTH_API_CONTRACT.md](UNSLOTH_API_CONTRACT.md) and `tests/fixtures/unsloth/*.json` are meant to be identical in nd-world; both products' stand-in servers replay the fixtures. `job_body.json` is the body Lore Forge builds
(a test compares them); `job_fields.json` is the list of keys Unsloth accepts that both send. No shared crate (owner decision 4).

## Deviations from the plan

- `chatml_rows` returns `(train, eval, stats)` without a `Result` (nothing in it can fail; the "no examples" error stays in `export::studio`).
- The Train page names Unsloth with the pseudo server id `"unsloth"` and `TrainJob.unsloth` is a second optional target next to `remote` (the plan's smaller-diff variant).
- Test and Find are one command, `unsloth_check(url, key, find)`; the key typed in the form wins, an empty key means the saved one for that address.
- `job_body` takes the catalog model's `repo` (no separate `model_ref`); local model folders are refused for Unsloth (plan item LF5 is optional and unverified).
- When the adapter cannot be copied (Unsloth on another computer, or the disk is too full) the run ends **done** with `adapter_remote` and a note, not failed: the training itself worked and the adapter is safe in Unsloth.
- A failed or interrupted run in Unsloth counts as continuable (Continue reattaches or sends the data again); only a stopped (cancelled) one does not.
- The fixtures live in `tests/fixtures/unsloth/` at the repository root (as in the plan); the golden files of the Studio export are in `crates/ndforge-core/tests/fixtures/studio_golden/`.
- The events of a reattached run are cut at the newest event already shown (the plan paged until it found one; the cut avoids re-sending older lines).
- The Tauri crate passes `cargo check -p nd-lore-forge` (the window itself was never started, so the three commands and the page are only tested through the stand-in browser).
- `cargo test -p ndforge-core --test remote` has a timing test (`a_second_job_is_refused_while_one_runs_and_a_lost_client_can_reconnect`) that failed once while the whole suite ran in parallel and passed alone; it is older than this work.

## Not verified yet (needs a real Desktop and a GPU)

Everything above is tested against a stand-in server written from Unsloth's documentation and source (backend commit `1bbf9c3f`), never against a real Desktop. Specifically:

1. The Desktop version, the port it listens on, whether its data folder is `~/.unsloth/studio` (and so where `adapter_config.json` ends up), and whether Find finds it.
2. `/api/health` shows `desktop_owner`; with a key it shows `version`. The `/api/train/hardware` device keys (`name`, `total_memory_gb`, `bf16` in the fixture are guesses; the reader accepts a few spellings).
3. A key made in Settings, API works on `/v1/models`, `/v1/fine_tuning/jobs`, `/api/models/checkpoints`, `/api/export/status`; `/api/models/checkpoints` really gives host paths to an API-key caller.
4. A LoRA job with `load_in_4bit: false` trains in 16 bit; a QLoRA job picks the 4-bit weights; the adapter's `base_model_name_or_path` is something Lore Forge's Test page can load.
5. A run with a small model and the Quick preset: charts move, the adapter is copied, Step 4 Test runs on it, Step 5 makes a GGUF from it.
6. Stop gives *cancelled* and the right message; starting while Desktop chats gives the queue text; queue position wording ("position N") matches Unsloth's event text.
7. Desktop closed mid-run: *interrupted* after the poll budget; after reopening, can Continue reattach, and is the last checkpoint fetchable?
8. Auto-update between two runs: the probe still passes; record the new version in the tested-version table of the contract.
9. Out-of-memory and disk-full wording of real failures (`explain_failure` matches `out of memory`, `OOM`, `No space left`, `errno 28`).

## Steps of the plan not done

- 14 (the GPU checklist run), 15 (slim / healed / grown models as the Desktop model, optional, GPU), 16 (export through Unsloth, optional, experimental), 17 (adapter download from a Desktop on another computer, optional): owner decisions 2 and 3.
- The nd-world half is done in the other repository.

## Checklist for the first real run

Desktop open; key created; Training setup: Find, paste the key, Test (expect the card and "training API: yes"); Save; Step 3: Where to train = Unsloth Desktop, Quick preset, a small model; watch the charts;
let it finish; check `runs/<name>/adapter/` holds `adapter_config.json` and `adapter_model.safetensors`; Step 4 Test on that run; then repeat with Stop, with Desktop closed mid-run, and with Desktop busy chatting.
Write the Desktop version, port and anything that differed into the tested-version table of the contract document.
