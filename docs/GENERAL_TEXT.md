# General text for healing and for the importance matrix

Two features read **general text** from a dataset: healing by distillation ("Heal it", about 50 % of the healing tokens) and the importance matrix of the GGUF export
(about 20 % of the calibration text). Both use one reader (`worker/ndforge_worker/generaltext.py`) and one chooser (`ui/app.js`, `gcHtml`, shared by the Heal card and the
importance-matrix box). This page is the record of the owner's decisions.

## What changed and why

The old default `nvidia/Nemotron-Pretraining-SFT-v1` is **gated** on the Hub (`gated: manual`, licence "other"), so healing said "... is a gated dataset on the Hub ... Healing goes on
with your world's text only". Now:

1. The default is an **open** dataset: `mlabonne/open-perfectblend` (Apache-2.0, chat). The shares stay (healing about 50 %, matrix about 20 %).
2. The person can **choose** among presets or **add their own** dataset (a Hugging Face id, or a file on their computer), with a **Check it** button.
3. The failure messages say what is wrong and what to do.
4. The dataset is part of the **cache keys**.

## The presets

The table is data (`PRESETS` in `crates/ndforge-core/src/general.rs`): licence, a licence note, kind, gated, needs an acknowledgement. Checked on the Hub API on 2026-11-14
(gated or not, licence, columns from the dataset cards); **no rows were read** (see "Not verified yet").

| Dataset | Licence | Kind | Config / split | Note |
|---|---|---|---|---|
| `mlabonne/open-perfectblend` (default) | Apache-2.0 | chat, `conversations` ({from, value}) | - / `train` | about 1.42 M rows; also a `source` string |
| `HuggingFaceH4/ultrachat_200k` | MIT | chat, `messages` | - / `train_sft` | 207,865 rows |
| `allenai/tulu-3-sft-mixture` | ODC-By | chat, `messages` | - / `train` | about 939 k rows |
| `HuggingFaceFW/fineweb-edu` | ODC-By | text, `text` | `sample-10BT` / `train` | huge: only streamed, never fully downloaded |
| `wikimedia/wikipedia` | CC-BY-SA-3.0 / GFDL | text (id, url, title, `text`) | `20231101.en` / `train` | ShareAlike: the licence line says so |
| `faunix/Qwen3.8-27B-Distillation-40K` | Apache-2.0 | chat, `messages`, thinking inline (`<think>...</think>`, unverified) | - / `train` | 40,000 reasoning traces (code 40 %, math 27.5 %, science 17.5 %, logic 15 %); rows are LONG (about 5,750 tokens, about 64 % thinking): use a sequence length of 4096 or more for healing, longer rows are cut, which is fine for distillation. Good for healing a pruned Qwen3.x model; a weak fit for the matrix (it leans to code and maths) |
| `r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation` | other: noncommercial research | chat, `messages` with `reasoning_content` | `sft_balanced` / `train` | needs an acknowledgement (below); many other configs exist, only this one is a preset |
| `nvidia/Nemotron-Pretraining-SFT-v1` | other (NVIDIA) | chat (unverified) | `Nemotron-SFT-General` / `train` | **gated**: needs access approval and your token |

Row counts above come from the cards and are never used by the program.

### The acknowledgement (`needs_ack`)

`r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation` has the licence `other`: its LICENSE limits it to controlled, noncommercial research; the upstream sets mix CC BY-NC-SA 4.0,
CC BY-NC 3.0, CC BY-SA 3.0, CC BY 4.0, Apache-2.0, MIT and ODC-By; and the Alibaba Cloud Model Studio International Product Terms (section 4.48(d)(v)) restrict using the service's
outputs to train products or services that compete with Alibaba Cloud. The chooser shows the note in plain words ("Noncommercial research use; mixed licences; teacher-service
terms ... Read the dataset's LICENSE before using it for anything you publish or sell.") and a checkbox. **Until the box is ticked the Start / Make buttons are disabled, and the
core refuses the job** (`general::resolve(.., require_ack = true)`), so such a dataset is never used silently. The acknowledgement is stored once per dataset in
`general_datasets.json` (`acknowledged`). "Check it" does not need it (it reads 20 rows and uses nothing). Whether a use is allowed is the person's decision; the program does not judge it.
The dataset's README has inconsistent repository ids and counts (57,937 vs 51,389 rows); nothing here depends on a count.

Every preset shows its **licence line** in the chooser (and the dataset page link). A dataset you add yourself shows "the licence is yours to check"; the program also asks the Hub
API for the licence on "Check it" and shows it if it can (`general::hub_info`); if the Hub cannot be read nothing fails.

## What the worker reads (formats)

Per dataset the worker reads **a plain-text column** or **a chat column**; with no column named it finds one and says what it found (log line "General text (...): reading column
'conversations' as chat (from/value messages) ...", and "Check it" shows it too):

| Shape | Example | Used as |
|---|---|---|
| text | `"text": "..."` (or `content`, `body`, `document`, `raw_content`, `article`, `passage`; else the longest text of the row) | as it is |
| `messages` | `[{"role": "user", "content": "..."}, ...]` | chat |
| `conversations` | `[{"from": "human", "value": "..."}, ...]` (human/user -> user, gpt/assistant/bot/model -> assistant, system -> system) | chat |
| prompt + answer | `prompt` (or `instruction`, `question`, `input`) + `completion` (or `response`, `output`, `answer`) | a two-turn chat |

Order of detection without a named column: a chat column (`messages`, `conversations`, `conversation`, `chat`), a prompt + answer pair, a text column, the longest text. With a named
column the shape of the cell decides (a list of {role, content} is chat even when the column is called anything). A text column that holds a list of turns is joined as lines.

**Chat rows** are rendered with the MODEL's own chat template, exactly as the project's practice conversations are (`data.render_chat`: prompt through `apply_chat_template`, then the
answer and the family's end marker). A conversation is cut so that it ends with the assistant's answer and has a user turn before it; one that is not like that, or that the template
refuses (strict alternation ...), is skipped and counted. If nothing is left the message is "no usable column ... (the model's chat template refused the conversations)".

**Thinking.** Thinking that is already inline in the assistant text (`<think>...</think>`, as the faunix card suggests) is left alone. A separate `reasoning_content` field (r0b0tlab) is put
into `<think>\n...\n</think>\n\n` before the answer **once** (never double-wrapped: nothing is added when the text already has a `<think>`). Messages with `tool_calls`, a `tool_call_id` or a
`tool` role make the whole row **skipped** (the simpler, robust choice: no tool turns in general text); `trainable`, `tools`, `sampling_weight`, `domain` and the other fields are ignored.
The Qwen template of this program is non-thinking by default (`families.py`); the traces teach the original's way of thinking only where their rows contain it.

**Local files**, read the same way: `.jsonl`/`.ndjson` (one JSON per line), `.json` (a list, an object holding a list, or JSON lines), `.txt`/`.md` (blocks separated by blank lines are the
documents), `.csv`/`.tsv` (header row), `.parquet` (only when `pyarrow` is installed in the training software; otherwise a plain message). A file is read line by line where the format allows.

**Streaming and the sample.** A Hub dataset is read with `datasets.load_dataset(..., streaming=True)` and only the rows needed for the token budget are pulled; `fineweb-edu` is never downloaded.
The sample is **deterministic for the same seed**: the Hub stream is shuffled by `datasets` (seed 3407, buffer 2,000), a local file by a seeded window shuffle (`buffer_shuffle`).
(Without a `seed` in the source - old callers, tests - the dataset's own order is kept.)

## The chooser in the app

On the Heal card (tick "Also use general text") and in the importance-matrix box (tick "Better small sizes"), one list:

- the presets (name, licence, kind chat/text, a gated mark), the datasets you added, and **My own...**;
- under it: the one-line note, the **Licence** line (and the licence note), the acknowledgement box if the preset needs one, the gated warning;
- **My own...**: a Hugging Face dataset id (or address), config, split, column (optional) **or** a file path; **Add to my list** saves it; **Remove from my list** removes it;
- **Check it** reads about 20 rows (streaming, through the worker's `general_check` job) and answers: *Works* (with what was detected, a short sample, the licence from the Hub if known),
  *Needs access*, *Not found*, *No usable column*, *Offline*, *No rows*. It needs the training software (Training setup);
- **Hugging Face token** (for gated datasets): a field to save it, a button to remove it.

Where things are stored:

| What | Where |
|---|---|
| presets | in the program (`PRESETS`) |
| your own datasets, and the acknowledged licences | `general_datasets.json` in the settings folder, next to `custom_models.json`; a missing or damaged file counts as empty |
| the last choice | `general_choice.json` in the project folder, `{"heal": ..., "imatrix": ...}` (remembered per project and per place) |
| the Hugging Face token | `secrets.json` (key `huggingface`), like the other secrets (not encrypted, mode 0600 on Linux) |

### The token

The token is passed to the worker **only as the environment variable `HF_TOKEN`**, only for the jobs that read general text from the Hub (`general_check`, `distill_cache`, `distill`
with a general block, `export` with the matrix on; `envmgr::jobs::job_env`). It is never in `job.json`, in `heal.json`, in the page state (only `has_token`), in the log or in a problem
report: `scrub` (Rust, also used by the problem report) and `protocol.scrub` (worker, applied to every log and error line, and to the exact value of `HF_TOKEN`) replace `hf_` followed by
20 or more letters or digits with `[token]`. A token is accepted only if it starts with `hf_`. Removing it deletes it from `secrets.json`.

## Cache keys

- **Teacher cache** (`distill/cache/index.json`): `general_key` and `general_rows` are compared together with teacher, k, rows, `data_hash`, tokenizer and length
  (`distilljob.cache_matches`). `general_key` = `{used, id, file, config, split, column, share, seed}` (`generaltext.general_key`; Rust builds the identical value, `general::key`, and a test
  pins the literal). Another dataset, config, split, column, share or seed -> the cache is "other data": the worker discards the old parts and the app runs the cache job again
  (`cache_complete` compares the key too). An **older cache has no key and counts as other data** (also an old one without general text: the key is always stored). When the key changed
  after a stopped heal, the old training checkpoints (`distill/run`) are removed so they are not continued with other data. The `distill` job refuses a cache with another key.
- **Saved importance matrix** (`imatrix.json`): `settings_key` gains `general` (the same key, with the matrix's share 0.2 and seed); `find_saved` needs equality, so another dataset never reuses
  a matrix, and a matrix saved before this key counts as made from other data. The number of general texts follows from `chunks x ctx x share`, which are in the key as well.
- The heal job's *row count* is `rows`/`general_rows` in the cache index.

## Failure messages

All of them start with `The general text (<name>) could not be downloaded` (healing) or `... could not be read` (matrix), then the reason, then what happens (healing goes on with your
world's text only; the matrix is made from your project's text only). The dataset used or skipped is named in the log ("General text used: ..."), in `heal.json` (`general_used`, `note`,
`note_code`), in the result of the cache job (`stats`) and in the healed model's record (`general`).

| Cause | Message (the reason part) |
|---|---|
| gated, 401, 403 | "This dataset needs access approval on Hugging Face (<url>). Ask for access and add your Hugging Face token, or choose another dataset." (with a token already saved: "Your Hugging Face token was sent, but this dataset did not accept it: ask for access with the same account.") |
| not found | "Hugging Face has no dataset called '<id>' that this program can read. Check the spelling (owner/name); a private dataset needs your Hugging Face token." |
| unknown config or split | "Hugging Face does not know that config or split of this dataset (...)" |
| offline | "Hugging Face could not be reached; check the internet connection and try again (...)." |
| no usable column | "No column with usable text was found. Its columns: ... Name the column in "My own..."", or "The dataset has no column called 'x'." |
| no rows | "The dataset gave no rows. Check the config and the split." |
| a file | "The file ... was not found." / "Line 2 of the file is not valid JSON ..." / pyarrow missing |
| no `datasets` library | "The training software has no 'datasets' library ..." |

After a failure the note adds "Or use mlabonne/open-perfectblend instead (choose it in the list)" (unless that was the dataset), and on the Heal card a one-click
**Use mlabonne/open-perfectblend instead** button next to the note (it only changes the choice; the person presses Continue healing). The export page shows the sentence only (no button).

## Tests

- Worker (`worker/tests/test_generaltext.py`): local files in every format (jsonl, json, txt, csv; parquet with `importorskip`), text/messages/conversations/prompt+completion, the chat template
  with a stand-in tokenizer, inline think / `reasoning_content` / tools, an injected Hub loader (gated, not found, offline, no usable column, empty, bad config), the check job, the
  fallback notes, determinism for the same seed, the cache keys (`cache_matches`, `settings_key`/`find_saved`), the token (environment only, scrubbed from logs and notes).
- Core (`crates/ndforge-core/tests/general.rs`, plus heal tests in `tests/reshape.rs`): the preset table, the saved list (add, remove, missing file, damaged file), the acknowledgement,
  the choice per project, the token (kept, validated, never in the state), which jobs get `HF_TOKEN`, the scrubbing, the key literal, and heal runs with a fake worker (chosen dataset in the
  job, the token only in the environment, another dataset makes the teacher answer again, an acknowledgement is required).
- UI (`ui/smoke/general.test.mjs`): the list, a preset change, My own... (hub and file), Check it (success, each failure, a failing call), Add, Remove, the token, the acknowledgement,
  the importance-matrix box with the same chooser.

## Deviations from the brief

- Presets send **no column** (auto-detect) even where the card names one (`conversations`, `messages`, `text`); the card's column is only described in the note. Reason: the real column
  layouts could not be read here, and auto-detection prefers the chat columns, so a wrong note cannot break a run.
- The acknowledgement and the custom list share one file, `general_datasets.json` (`{"datasets": [...], "acknowledged": [...]}`), as the coordinator asked.
- The one-click "use the default instead" is a button on the Heal card only; the export page gets the sentence in its note.
- The old Expert field "General text (Hugging Face dataset)" is gone: the chooser is for everybody. Old callers that send `general_dataset` / `general_column` still work.
- `Nemotron` is listed as `chat`; its real layout is unverified.

## Not verified yet

Nothing here was run against the real Hub, a real model or a GPU (the datasets-server host is blocked in the build sandbox and the training libraries were not installed).

- The **real column layouts** and rows of every preset (the card says `conversations` for perfectblend, `messages` for ultrachat and tulu, `messages` with inline thinking for faunix,
  `messages` with `reasoning_content` for r0b0tlab, `text` for fineweb-edu and wikipedia); that `Nemotron-SFT-General` is a config of the gated set; that perfectblend rows pass the chat
  templates of Qwen3.5 and Gemma 4 (strict alternation: rows the template refuses are skipped, how many is unknown).
- That `datasets.load_dataset(..., streaming=True, token=...)` works with the version installed by the training setup (older versions call the argument `use_auth_token`), that
  `IterableDataset.shuffle(seed, buffer_size)` gives the same rows twice, and the **speed** of streaming (how long 10 M tokens of perfectblend take; first-row latency; fineweb-edu).
- The Hub's real error texts for gated, not found and offline (the classification looks at exception names and phrases such as "gated dataset", "ask for access", 401/403, "doesn't exist";
  the user's own report of the gated message is the only real sample) and that a saved token is accepted by a dataset the account has access to.
- The licence texts of the datasets (read the dataset pages), and the licence line the Hub API returns for "Check it".
- The thinking handling on a real Qwen3.x template (`<think>` before the answer); the interaction with the non-thinking default template.
- A heal or an export on the owner's PC with the new default; the memory and time of long rows (about 5,750 tokens) at sequence length 4096; `.parquet` with a real `pyarrow`.
- The Tauri commands (`general_*`, `hf_token_*`) and the window itself: `src-tauri` was not built here (disk), only `ndforge-core` and the page in a stand-in browser.
