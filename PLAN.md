# ND Lore Forge — plan

A desktop program for **Windows and Linux** with a GUI that turns an nd-world world (its lore, rules and mechanics) into a
training dataset and fine-tunes a small open model (Qwen3.5, Gemma 4 and similar) on it. The result is a small model made
for one world, which nd-world can then use for AI Chat, Talk to NPCs, Ask AI and the Chronicler.

Status: **plan only, nothing is implemented yet.** Working name: *ND Lore Forge* (`ndforge`).

---

## 1. Goals and non-goals

**Goals**
1. One GUI program, the same on Windows 10/11 and Linux (x86-64, NVIDIA GPU), that runs the whole loop:
   import world → build dataset → review → train → evaluate → export → use in nd-world.
2. Read nd-world data the way nd-world exports it today, so **nd-world does not need to change** for version 1.
3. Train Qwen3.5 and Gemma 4 small models (and other Hugging Face causal LMs) with LoRA / QLoRA on a single consumer or
   server GPU, including the 16 GB Tesla T10 / T4 and the V100 that nd-world's own docs target.
4. Keep secrets secret: a model meant for players never sees `[gmonly]` text or hidden entities during training.
5. Produce files nd-world can load straight away: GGUF + an Ollama `Modelfile`, or a model folder for Unsloth Studio.

**Non-goals (version 1)**
- Training from scratch or training large models (> ~14 B parameters).
- AMD / Intel / Apple GPUs (CPU-only works for data building and tiny smoke runs, not for real training).
- Multi-GPU / multi-machine training (single GPU only; remote training over SSH is a later phase, see §11).
- Image / audio fine-tuning — Qwen3.5 and Gemma 4 are multimodal, but only the text part is trained; vision towers stay frozen.

## 2. The key design decision: what "teaching lore" actually needs

Fine-tuning alone memorises facts poorly and makes a small model confidently invent details. The plan therefore uses
**three layers**, and the GUI makes them separate, visible steps:

| Layer | What it does | Data |
|---|---|---|
| **A. Continued pre-training (CPT, optional)** | Soaks the world's names, style, vocabulary and rules text into the model. | Raw "lore book" text rendered from every entity, rules, sessions. |
| **B. Supervised fine-tuning (SFT, main step)** | Teaches the model to *answer questions about* the world, explain mechanics, role-play NPCs, and say "that is not in the lore" instead of inventing. | Question/answer and multi-turn chat examples built from the lore (templated + teacher-generated + verified). |
| **C. Retrieval stays on** | nd-world already does RAG (`app/retrieval.py`, vault chunks, knowledge graph). The tuned model is trained *with* retrieved context in some examples, so it uses RAG well instead of replacing it. | A share of SFT examples include a "Context:" block the way nd-world builds its prompts. |

Result: a small model that knows the world's vocabulary and rules by heart, stays grounded when nd-world gives it
context, and admits gaps — which is what makes a 2–9 B model usable instead of the 26 B default.

## 3. Target models and hardware

| Family | Sizes worth training here | Notes |
|---|---|---|
| Qwen3.5 | ~0.8 B, 2 B, 4 B, 9 B | Strong multilingual and reasoning; has "thinking" mode — train with thinking off by default, option to keep it. |
| Gemma 4 | E2B, E4B (and 26B-A4B MoE only on ≥ 24–32 GB, experimental) | Matches nd-world's current default family (`gemma4:26b`), so the tuned E4B is a drop-in smaller sibling. |
| Others | Any HF causal LM with a chat template (Llama 3.x, Mistral, Phi, SmolLM …) | Via the generic backend. |

Exact model IDs are **not hard-coded**: a `models.toml` catalog lists the known-good repos (Unsloth's pre-quantised 4-bit
repos and the official ones), and the user can paste any Hugging Face ID. The catalog is checked against the hub at
implementation time because names change.

**Hardware presets** (auto-detected, user can override):

| VRAM | Default | Precision |
|---|---|---|
| 6–8 GB | Qwen3.5 2 B / Gemma 4 E2B, QLoRA 4-bit, seq 2048 | fp16 or bf16 |
| 12–16 GB (T4/T10, RTX 4060 Ti 16 GB, 4070 Ti S) | Qwen3.5 4 B / Gemma 4 E4B QLoRA; 9 B QLoRA at seq 2048 | **fp16 on Turing/Volta** (no bf16), bf16 on Ampere+ |
| 24 GB+ | 9 B LoRA 16-bit or QLoRA with long context | bf16 |

The hardware probe reads GPU name, compute capability, VRAM, driver/CUDA version and RAM, and picks precision automatically
(compute capability < 8.0 → fp16, SDPA attention instead of FlashAttention 2).

## 4. Technology choices

| Part | Choice | Why |
|---|---|---|
| Language | Python 3.12 | The whole training stack is Python. |
| GUI | **PySide6 (Qt 6)** + pyqtgraph for live charts | Native look on Windows and Linux, handles long background jobs, no browser needed, LGPL. |
| Training engine | **Unsloth** (primary) with a plain **Transformers + PEFT + TRL** backend as fallback | Unsloth supports Qwen3.5 and Gemma 4, cuts VRAM roughly in half, runs on Windows and Linux; the fallback covers models Unsloth does not support yet. |
| Datasets | `datasets` + JSONL files on disk | Easy to inspect, diff and version. |
| Teacher LLM (synthetic data) | Any **OpenAI-compatible endpoint**: the user's nd-world Unsloth Studio or Ollama (`gemma4:26b`), local llama.cpp server, or a cloud API | Reuses what the user already runs. |
| Export | llama.cpp `convert_hf_to_gguf.py` + `llama-quantize` (or Unsloth's `save_pretrained_gguf`), Ollama CLI | GGUF is what Ollama and Unsloth Studio load. |
| Environment | **uv**-managed virtual environment, created by the app on first run | Torch/CUDA wheels are big and GPU-specific; keeping them out of the installer keeps it small and lets the app pick the right CUDA build. |
| Packaging | PyInstaller (GUI shell) → Inno Setup installer (Windows), AppImage + tar.gz (Linux) | One-click install on both. |
| Secrets (HF token, API keys) | OS keyring (`keyring`: Windows Credential Manager / Secret Service) | Never in project files or logs. |

**Process model:** the GUI never imports torch. Each long job (dataset generation, training, evaluation, export) runs as a
**separate worker process** inside the managed environment and reports progress as JSON lines over a pipe. This keeps the
GUI responsive, lets "Stop" kill a job cleanly and always frees GPU memory, and means a CUDA crash cannot take the GUI down.
The same worker has a CLI (`ndforge train project.toml`) so a run can also be started headless on the server.

## 5. Where the data comes from (nd-world inputs)

All importers produce one canonical record per item (see §6). Supported sources, in priority order:

1. **World JSON export** (Tools → Export & Backup → World JSON): `{"world": {...}, "entities": [...]}` — entities with
   `kind, subtype, name, folder, tags, summary, body, visible_to_players, template, custom_fields_json`. Embedded
   `image_data` is ignored (dropped on import to save memory).
2. **Split export files**: `entities/{kind}.json`, `rules.json` (`rules_md`), `player-characters.json`, sheet templates.
3. **nd-world API with a token** (read-only): the same bearer token the MCP server uses (`/mcp` tools `list_worlds`,
   `search_entities`, `get_entity`, `get_rules`, `list_sessions`, `get_session`, `list_tables`, `list_quests`). Pulls
   sessions/recaps, facts, quests, random tables and relations that the JSON export does not include. Only read tools
   are ever called.
4. **A copy of `world.db`** (SQLite), opened strictly read-only (`mode=ro`), for the richest import: entities, aliases,
   `roleplay_personality`, `entity_relations`, `entity_links`, facts, game sessions, quests, random tables, rules text,
   AI instructions. The GUI warns to use a **backup copy, never the live database** (same rule as nd-world's CLAUDE.md).
5. **Loose files**: the repo's `lore/` JSON format (`{"entities": [...]}`), an Obsidian vault folder (markdown +
   frontmatter + `[[wikilinks]]`, as nd-world's vault sync reads it), plain `.md` / `.txt` / `.pdf` rulebooks
   (e.g. `docs/asterion_rules.md`, `app/core_rules.md`).

Each import is a **snapshot** with a content hash, so a dataset always says exactly which world data it came from and a
re-import shows a diff (added / changed / removed entities).

## 6. Data pipeline

```
sources ─► import ─► canonical records ─► audience filter ─► clean & chunk ─► dataset builders ─► review ─► versioned dataset
```

### 6.1 Canonical record
`{id, world, kind, subtype, name, aliases[], tags[], folder, summary, body_md, fields{}, relations[{type, target}],
visible_to_players, group_limited, source, updated_at}` plus separate record types for `rules_section`, `session_recap`,
`fact`, `quest`, `random_table`, `ai_instruction`, `pc_sheet`.

### 6.2 Audience filter (security-critical)
Every project chooses an **audience**: **GM model** (sees everything) or **Player model** (spoiler-safe). A model cannot
be trusted to keep a secret it was trained on, so for a Player model the filter removes, before anything else runs:
- entities / quests / facts with `visible_to_players = false`, or limited to player groups (`group_ids_json`);
- `[gmonly]…[/gmonly]` blocks; `[for=IDs]…[/for]` blocks (always dropped — the model is shared by all players);
  `[reveal=DATE]…` blocks whose date has not passed at build time; malformed tags → dropped (nd-world's "hide, never show");
- AI instructions not marked `applies_to_players`; GM-only fields of PC sheets.

The tag rules are a port of nd-world's `app/rendering.py::strip_gm_only`. Following nd-world's "Python is the oracle"
rule, the test suite includes **parity fixtures generated by running nd-world's own function** on tricky inputs
(nesting, unclosed tags, mixed case, dates) and the port must match byte for byte. After building, a **leak scan** checks
that no string from any removed block appears anywhere in the Player dataset; a hit blocks training.

`[spoiler]…[/spoiler]` is kept (players can click it open) but unwrapped to plain text.

### 6.3 Cleaning and chunking
Strip nd-world markup (`[color]`, `[mark]`, `[u]`, spoiler wrappers, autolinks, image embeds) to plain markdown; normalise
`[[wikilinks]]` to names; split long bodies by heading into ~300–800-token chunks that keep the entity name in each chunk;
deduplicate near-identical text (MinHash); keep accented names exactly (Anders de Vallière).

### 6.4 Dataset builders (each one can be switched on/off and weighted)
1. **Lore book (CPT text)** — every record rendered as a readable article ("# Name — kind / subtype\nAlso known as …\n
   Summary … Body … Related: …"), rules as their own chapters.
2. **Template Q&A (deterministic, no LLM needed)** — from structured fields: "Who is X?", "What is X also called?",
   "Where is X located?" (relations), "What does organisation Y do?", custom fields ("What is X's rank?"), random-table
   contents, rule section → "How does <mechanic> work?". Several phrasings per question.
3. **Teacher-generated Q&A (synthetic)** — for each chunk the teacher model writes N grounded questions + answers
   (factual, "explain", "compare", "what would happen if" for mechanics, multi-hop across related entities), multi-turn
   chats, worked rules examples (dice pools, combat turns for the world's ruleset — nd / worm / hunt / asterion / generic).
4. **NPC role-play** — for entities with `roleplay_personality`, in-character dialogues that follow that direction, with
   the system prompt format nd-world's Talk-to-NPC uses.
5. **Grounded-context examples** — questions answered with a nd-world-style retrieved context block, including examples
   where the context is incomplete and the right answer says so.
6. **"Not in the lore" negatives** — questions about invented names and events → a short refusal ("That isn't in this
   world's lore"). For Player models, also questions aimed at removed secrets → "Your character doesn't know that."
7. **General replay** — a small share (configurable, default ~10 %) of a general open instruction dataset so the model
   does not forget how to follow ordinary instructions.

**Verification pass:** every synthetic example is checked by the teacher (or a second model) as a judge: "Is every claim
in the answer supported by the source chunk?" Unsupported examples are dropped or sent to review. The generation is
resumable and cached per chunk hash, so re-running after a small world change only regenerates what changed.

**Split:** a held-out evaluation set is split **by fact/entity**, not by paraphrase, so evaluation measures recall, not
copy-paste. Default 90/10.

### 6.5 Review and versioning
The Dataset page shows every example with its source record, lets the user edit, approve, reject, filter (by kind,
builder, judge score) and search. "Freeze" writes `datasets/<version>/train.jsonl, eval.jsonl, manifest.json`
(source snapshot hash, audience, builder settings, counts, token totals). Training always points at a frozen version.

## 7. Training

- **Methods:** QLoRA (default), LoRA 16-bit, full fine-tune (only for ≤ 1 B models); optional two-stage **CPT → SFT**
  pipeline as one run.
- **Chat format:** taken from the model's own tokenizer chat template (Qwen3.5 / Gemma 4 differ); loss on assistant
  turns only; optional packing; Qwen thinking off unless chosen.
- **Presets:** *Smoke test* (50 steps, proves the setup), *Balanced*, *Max quality*; each computed from VRAM, model size
  and dataset size (rank, alpha, LR, batch / grad accumulation, epochs, max seq len). Every value stays editable in an
  "Advanced" panel; a VRAM estimate updates live before starting.
- **Run management:** each run gets `runs/<timestamp-name>/` with the full config, logs, checkpoints, adapter, metrics.
  Resume from checkpoint, stop, and "duplicate run with changes".
- **Live view:** loss / eval loss / learning rate / grad norm charts, tokens per second, VRAM, GPU temperature (useful for
  the passive T10), ETA, and periodic sample answers to fixed probe questions so you can *see* the model learn.
- **Safety nets:** OOM → suggest the next smaller preset and offer to retry; disk-space check before saving checkpoints.

## 8. Evaluation

Runs automatically at the end of training (and on demand) on the held-out set, base model vs tuned model:
- **Lore accuracy:** fuzzy match + LLM-judge score on held-out questions.
- **Hallucination rate:** share of "not in the lore" probes that get an invented answer.
- **Secret-leak test (Player models):** probes built from removed GM-only content; any reproduced secret = fail.
- **Rules / mechanics checks:** worked-example questions with known answers.
- **General sanity:** a few ordinary prompts, to catch a model that forgot how to talk.
- **Playground:** side-by-side chat with base and tuned model (optionally with a pasted context block), saved transcripts.
- **Run comparison** table and charts across runs.

## 9. Export and use in nd-world

1. Merge LoRA into the base weights (16-bit safetensors), or keep the adapter alone.
2. Convert to **GGUF** and quantise (Q4_K_M default; Q5_K_M, Q8_0, F16 options).
3. Write an Ollama `Modelfile` (chat template, stop tokens, recommended context and temperature), and optionally run
   `ollama create <world>-lore` locally or on a remote Ollama URL.
4. Optionally push to a **private** Hugging Face repo so Unsloth Studio's Model Hub can download it.
5. Write a model card: base model, world, audience (GM/Player), dataset version and hash, eval results, licence notes.
6. Show the exact next step for nd-world: set `OLLAMA_MODEL=<name>` (legacy backend) or `UNSLOTH_MODEL=<repo>` and
   `LLM_CONTEXT_TOKENS` (Unsloth Studio), and keep retrieval on.

Licence note shown in the export page: Gemma models are under Google's Gemma terms, Qwen under its own licence; the tuned
model inherits them.

## 10. GUI layout

Left sidebar, one page per step; a status bar shows the active project, GPU and the running job.

| Page | Contents |
|---|---|
| **Projects** | New / open / recent; a project = one world + one audience. Wizard for first project. |
| **Sources** | Add World JSON / split files / `world.db` copy / API URL + token / folders; import preview (counts per kind, diff vs last snapshot); audience switch with clear GM/Player explanation. |
| **Dataset** | Builder toggles and weights, teacher endpoint settings and a cost/time estimate, "Generate" with progress, review table and editor, leak-scan result, "Freeze version". |
| **Train** | Model picker (catalog + any HF ID, download progress), method, preset, advanced panel, VRAM estimate, Start/Stop/Resume, live charts and sample answers. |
| **Evaluate** | Metrics, base-vs-tuned table, playground chat, run comparison. |
| **Export** | Format/quant options, Ollama create / HF push, model card, "how to use it in nd-world" instructions. |
| **Settings** | Managed environment (install/repair/update, CUDA build), GPU info, storage paths (models cache, projects), tokens in keyring, theme (light/dark), language. |

## 11. Repository layout (this repo)

```
Llm-training/
  pyproject.toml            # ndforge package; GUI deps only (PySide6, pyqtgraph, pydantic, keyring, httpx)
  requirements/             # worker env: train-cuda124.txt, train-cuda128.txt, cpu.txt (pinned)
  src/ndforge/
    app.py                  # entry point
    gui/                    # pages/, widgets/, charts, job monitor
    core/                   # project.py (project.toml), paths, settings, jobs.py (worker process + JSON-line protocol)
    envmgr/                 # uv bootstrap, GPU/CUDA detection, torch wheel selection, health check
    ingest/                 # world_json.py, split_files.py, sqlite_copy.py, api_client.py, lore_folder.py, vault.py, documents.py
    privacy/                # markup.py (port of strip_gm_only), audience.py, leak_scan.py
    dataset/                # canonical.py, clean.py, chunk.py, builders/{lorebook,template_qa,synthetic,npc,grounded,negatives,replay}.py,
                            # verify.py, split.py, freeze.py
    teacher/                # OpenAI-compatible client, prompts, cache, rate limiting
    train/                  # worker.py, presets.py, vram_estimate.py, backends/{unsloth_backend,hf_backend}.py, callbacks.py
    evaluate/               # metrics.py, judge.py, probes.py, playground.py
    export/                 # merge.py, gguf.py, ollama.py, hf_push.py, model_card.py
    cli.py                  # headless: ndforge import|build|train|eval|export project.toml
  catalog/models.toml       # known-good model repos per family and size
  tests/                    # unit, parity fixtures from nd-world, tiny-model smoke training, pytest-qt GUI tests
  packaging/                # pyinstaller spec, windows/ (Inno Setup), linux/ (AppImage)
  docs/                     # user guide, dataset format, troubleshooting
  .github/workflows/        # CI: lint + tests on windows-latest and ubuntu-latest; build installers on tag
```

Project folder (user data, outside the repo):
```
MyWorld-Player/
  project.toml  sources/<snapshot-hash>/  records.jsonl  datasets/v1/  runs/<run>/  exports/<run>-<quant>.gguf
```

## 12. Milestones (in order; each ends with something usable)

| # | Milestone | Done when |
|---|---|---|
| **M0** | Repo skeleton, CI on Windows + Linux, PySide6 window, settings, project files | Installer-less dev run works on both OSes; CI green. |
| **M1** | Managed environment: uv bootstrap, GPU/CUDA probe, torch install, health check page | Fresh Windows and Linux machines reach "GPU ready" from the GUI. |
| **M2** | Importers (World JSON, split files, lore folder, markdown) + canonical records + import preview/diff | The repo's `lore/` sample and a real World JSON export import with correct counts. |
| **M3** | Audience filter + markup port + parity fixtures from nd-world + leak scan | Parity tests pass byte-for-byte; a seeded secret never reaches a Player dataset. |
| **M4** | Lore book + template Q&A builders, split, review table, freeze | A frozen dataset without any LLM. |
| **M5** | Worker process + training (Unsloth backend, QLoRA SFT) + live charts + stop/resume | *Smoke test* preset trains Qwen3.5 small / Gemma 4 E2B on a 16 GB card; GUI stays responsive; stop frees VRAM. |
| **M6** | Export: merge, GGUF, Ollama Modelfile + create | The tuned model answers in nd-world's AI Chat through Ollama. **First end-to-end version.** |
| **M7** | Teacher-generated data + verification + NPC + grounded + negatives + replay | Better eval scores than M4-only data on the same model. |
| **M8** | Evaluation suite, playground, run comparison | Base vs tuned report generated automatically. |
| **M9** | SQLite-copy and API importers (sessions, facts, quests, tables, relations, personalities) | Session recaps and NPC personalities appear in the dataset. |
| **M10** | CPT stage, HF fallback backend, HF push, model card | Two-stage run works; a non-Unsloth model trains. |
| **M11** | Packaging: Windows installer, Linux AppImage, user guide | Install from a release on clean Windows 11 and Ubuntu 24.04. |
| Later | Remote worker over SSH (train on the nd-world server's GPU from a laptop); incremental re-train when the world changes; optional nd-world "training bundle" export endpoint | — |

About the optional nd-world endpoint: if it is ever added, it goes through nd-world's own rules (Python handler first as
the oracle, Rust gateway port, parity + Rust-only tests, docs updated). Version 1 deliberately needs none of that.

## 13. Testing strategy

- **Unit:** importers on fixture exports, markup/audience filter (parity fixtures), chunking, template builders, split
  by entity, manifest hashing, preset/VRAM maths.
- **Leak tests:** synthetic worlds full of `[gmonly]`, `[for=…]`, `[reveal=…]`, hidden entities; assert zero leakage.
- **Smoke training in CI:** a tiny random model on CPU for a few steps through the real worker protocol (no GPU needed).
- **GPU tests (manual / self-hosted runner):** smoke preset per supported family on a 16 GB card.
- **GUI:** pytest-qt for page flows; the job monitor tested against a fake worker.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Small model still hallucinates lore | Grounded examples + negatives + keep RAG on; hallucination metric shown prominently. |
| GM secrets leak into a Player model | Filter first, leak scan blocks training, secret-leak eval, parity with nd-world's own stripping. |
| Windows training quirks (Triton, bitsandbytes, paths) | Unsloth's Windows support + pinned, tested wheel sets; fallback backend; health check page with fixes. |
| New model releases change formats | Catalog file, tokenizer chat templates instead of hard-coded formats, generic backend. |
| Old GPUs (T10/V100: no bf16, no FlashAttention 2) | Automatic fp16 + SDPA selection; presets tested on a T4-class card. |
| Teacher generation is slow on the home server | Caching per chunk, resumable jobs, can run overnight, cloud endpoint option. |
| Disk usage (models + checkpoints) | Shared model cache, checkpoint retention setting, space check before every save. |

## 15. Open questions for the owner

1. **Where will training run?** Your Windows/Linux desktop (which GPU and VRAM?) or the nd-world server with the T10 —
   this decides whether the remote-worker phase should move earlier.
2. **GM model, Player model, or both?** (Both = two projects from the same import.)
3. **Which worlds and rulesets first?** (e.g. the Asterion world / rules) — used for the first real test.
4. **Teacher model:** OK to use the server's Gemma 4 26B through Unsloth Studio/Ollama, or also a cloud API?
5. **Model size target:** fastest possible (2 B-class) or best quality that fits 16 GB (4–9 B)?
6. **Language(s)** of the lore and of the answers (English only, or also others)?
7. **Name** of the program (working name *ND Lore Forge*).
