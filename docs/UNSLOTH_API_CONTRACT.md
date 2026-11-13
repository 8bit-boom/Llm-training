# Unsloth API contract (shared by ND Lore Forge and nd-world)

Copied from ND Lore Forge (`docs/UNSLOTH_API_CONTRACT.md`); the same text and the same `tests/fixtures/unsloth/*.json` live in nd-world. **Keep both in sync.**
Each product has its own small client (Lore Forge: synchronous `ureq`, `crates/ndforge-core/src/unsloth/client.rs`; nd-world: asynchronous, in the gateway); this
file and the fixtures are what they share. Both products' stand-in servers replay the fixtures, so a drift shows up in tests.

Evidence marks: **[V]** read in Unsloth's documentation or source (backend commit `1bbf9c3f`, 2026-11-13), **[A]** assumption, **[GPU/Desktop]** needs a real Desktop
install and a graphics card to confirm. The fixture values are made up around the shapes read in the source: version numbers, ids and device fields are samples.

Unsloth Desktop is an app that starts the same Studio backend and talks to it over HTTP, so the calls below hold for a Desktop and for `unsloth studio` [V].

## Connection

- Base URL `B`, no trailing slash. Studio listens on port 8888 by default; the port Desktop uses is not documented [GPU/Desktop]. Lore Forge tries `http://127.0.0.1:8888`
  and `http://127.0.0.1:8000` (and the address in the environment variable `NDFORGE_UNSLOTH_URL` first) in its "Find" button.
- Desktop answers on this computer only until the owner switches on Settings -> API -> Remote & LAN -> LAN access (plain HTTP on the network) [V].
- Every call carries `Authorization: Bearer <sk-unsloth-...>` except the two probes without a key. Keys are made in Settings -> API, shown once, stored hashed [V].
- Timeouts: connect 5 s, read 30 s, uploads 300 s, export calls 3600 s.
- Never browser-side: both products call from server or native code, so CORS does not apply.

## Calls

| Purpose | Call | Expect |
|---|---|---|
| Is it Unsloth? (no key) | `GET B/api/health` | 200 `{"status":"healthy","service":"Unsloth UI Backend","studio_root_id",...,"desktop_protocol_version":1,"desktop_owner"?:{...},"chat_only":bool}` [V]. Fixture `health_unauth.json` |
| Cheap liveness (no key) | `GET B/api/liveness` | `{"status":"alive","service":"Unsloth UI Backend"}` [V] |
| Key valid, version | `GET B/api/health` with the key | adds `version`, `studio_version`, `server_url`, `cloudflare_url`, `secure`, `cpu_training`, `chat_only_reason` [V]. Fixture `health_auth.json`. A bad key does **not** give 401: it gives the unauthenticated body. "No `version` although a key was sent" = key refused |
| Key check that answers 401 | `GET B/v1/models` | 401 on a bad or revoked key [V] |
| Fine-tuning present | `GET B/v1/fine_tuning/jobs?limit=1` | 200 list; 404 = this Unsloth is too old [V] |
| Card | `GET B/api/train/hardware` | `{"devices":[...]}` [V route; the device keys in `hardware.json` are [A], read them tolerantly] |
| Upload | `POST B/v1/files` multipart, field `purpose=fine-tune`, field `file` (`train.jsonl`) | `{"id":"file-<32 hex>","bytes","filename","status":"processed",...}` [V]. Fixture `files.json` |
| Start | `POST B/v1/fine_tuning/jobs` | 200 job object, `status` `validating_files` or `queued` (id `ftjob_q<16 hex>`) [V]. Fixtures `job_queued.json`, `job_running.json` |
| Follow | `GET B/v1/fine_tuning/jobs/{id}` | `status`: `validating_files`, `queued`, `running`, `succeeded`, `cancelled`, `failed` [V]. Fixture `job_succeeded.json` |
| Events | `GET B/v1/fine_tuning/jobs/{id}/events?limit=100` | `{"data":[...],"has_more"}` **newest first**; `after=<id>` pages toward OLDER events; ids are stable (`ftevent_<run>_step<N>`); `type:"metrics"` events carry `{step,total_steps,train_loss,learning_rate,grad_norm,valid_loss,epoch}` [V]. Fixture `events.json` |
| Stop | `POST B/v1/fine_tuning/jobs/{id}/cancel` | 200 job `cancelled`; 400 "already ..."; 409 "not running" [V] |
| Pause / resume | `POST .../pause`, `.../resume` | **always 400** "Pausing and resuming fine-tuning jobs is not supported; use cancel." [V]. Never call them |
| Run folders | `GET B/api/models/checkpoints` | `{"outputs_dir","models":[{"name":"<run folder>","checkpoints":[{"display_name","path","loss"}],"base_model","peft_type","lora_rank","is_quantized"}]}`; the host paths are visible to API-key callers (Unsloth's own MCP code relies on it) [V]. Fixture `checkpoints.json` (`@OUTPUTS@` stands for the outputs folder) |
| Export (not used by v1) | `POST B/api/export/load-checkpoint {checkpoint_path,max_seq_length,op_id}`, then `POST B/api/export/export/{lora,merged,gguf} {save_directory,expected_checkpoint,op_id,quantization_method?}`, status `GET B/api/export/status` | one export at a time [V]. Lore Forge does not call these yet (optional step LF6) |

## The job body

```json
{"model": "<catalog repo, or an absolute local model folder>", "training_file": "file-...", "validation_file": "file-...",
 "hyperparameters": {"n_epochs": 2, "batch_size": 2, "learning_rate_multiplier": 1.0}, "method": {"type": "supervised"},
 "suffix": "<project>-<run name, [A-Za-z0-9_-], at most 40>", "seed": 3407,
 "training_type": "LoRA/QLoRA", "load_in_4bit": false, "max_seq_length": 4096, "gradient_accumulation_steps": 4, "warmup_steps": 10, "max_steps": 0, "save_steps": 100,
 "packing": false, "train_on_completions": true, "gradient_checkpointing": "unsloth", "lora_r": 16, "lora_alpha": 16, "lora_dropout": 0.0,
 "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
 "finetune_vision_layers": false, "finetune_language_layers": true, "finetune_attention_modules": true, "finetune_mlp_modules": true, "eval_steps": 0.1}
```

- Any `TrainingStartRequest` field may ride along as an extra top-level key; an unknown key is a 400 `Unknown parameter(s): a, b`. `model_name`, `local_datasets`,
  `local_eval_datasets`, `start_request_id`, `project_name` are managed and refused [V]. The key list both products send is `tests/fixtures/unsloth/job_fields.json`.
- `learning_rate = 2e-4 * learning_rate_multiplier`; `suffix` becomes `project_name`; a validation file sets `eval_steps` to 0.1 by default [V].
- **`load_in_4bit` defaults to `true` in Unsloth** [V]: always send it, `false` for LoRA, `true` for QLoRA.
- Full fine-tuning (`training_type: "Full Finetuning"`) has no adapter and Lore Forge's two-stage training (world book first) cannot be expressed: both are refused for Unsloth.
- A busy Unsloth queues the job in memory; a queued job does not survive an Unsloth restart [V]: its id then answers 404.

## Failure messages (each product words them in its own UI)

| Situation | Detect | Message |
|---|---|---|
| Not running / wrong port | connection refused or timeout | "Unsloth does not answer at <url>. Open Unsloth Desktop (it serves only while it is open) and check the address in Unsloth -> Settings -> API." |
| Not Unsloth | 200 but `service` is not "Unsloth UI Backend" | "Something else answers at <url> (not Unsloth)." |
| LAN access off | refused from a non-loopback address | "... Desktop alone listens on this computer only: in Unsloth open Settings -> API -> Remote & LAN and press Start on LAN access." |
| Key refused | 401/403 on `/v1/*`, or authed health without `version` | "Unsloth refused the API key. Create a new one in Unsloth (Settings -> API) and paste it here; Unsloth shows it only once." |
| Too old | 404 on `/v1/fine_tuning/jobs` | "This Unsloth has no training API for other programs. Update it, or use Export for Unsloth Studio." |
| Busy / queued | status `queued`, id `ftjob_q...` | "Unsloth is busy with another job; this one waits (position N)." |
| Queue lost | 404 on a `ftjob_q` id | "Unsloth restarted and forgot the waiting job. Start it again." |
| Unknown parameter | 400 `Unknown parameter(s): x` | "This Unsloth does not know the setting x. Update Unsloth, or report it." (the key name only is logged) |
| Out of memory | failed, message holds `out of memory` / `OOM` | "Unsloth ran out of graphics memory. Choose QLoRA, a shorter context or a smaller model, and close chat models in Unsloth." |
| Disk full | `No space left` / errno 28 | "The disk is full on <computer>. Free space and try again." |
| Model folder not trainable | 409 from start | Unsloth's message + "Train it on this computer instead." |

## Feature probing and versions

Probe features, not version numbers (Desktop updates itself): fine-tuning = `GET /v1/fine_tuning/jobs?limit=1` is 200; checkpoints = `GET /api/models/checkpoints` is 200
with `models`. Re-probe at every run start. The `/v1/*` routes are the OpenAI-shaped, stable ones; `/api/*` can move, so each `/api/*` use sits in one function.
The unknown-parameter 400 is the canary for a changed schema; the stand-in servers refuse unknown keys with Unsloth's text, so a field added without checking fails the tests first.

### Tested against

| Unsloth | Source | Lore Forge | nd-world |
|---|---|---|---|
| backend commit `1bbf9c3f` (read, not run) | source reading, 2026-11-13 | stand-in server only | stand-in server only |
| first real Desktop | - | not yet (GPU checklist in `docs/UNSLOTH_DESKTOP.md`) | not yet |

## Security

Keys are stored like other secrets (Lore Forge: `secrets.json` under `unsloth:<address>`; nd-world: its settings), never logged, never written into run folders, exports, reports or
error texts. Default addresses are loopback. Training data holds the GM's secrets: it is sent only to the Unsloth the owner configured. Unsloth's own files (auth database, desktop secret) are never read.
