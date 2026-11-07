# ND Lore Forge — plan (revision 3)

A desktop program for **Windows and Linux** with an easy, guided GUI that turns one nd-world world (its lore, rules and
mechanics) into training data and fine-tunes a small model on it — **Qwen3.5 2B / 4B / 9B** or **Gemma 4 E2B / E4B**.
The result is a small **GM model** made for that world, which nd-world can then use for AI Chat, Ask AI, Talk to NPCs and
the Chronicler.

Status: **plan only, nothing is implemented yet.** Working name: *ND Lore Forge* (`ndforge`).

### What changed in revision 2 (owner's answers, 2026-11-07)

| Question | Answer | Effect on the plan |
|---|---|---|
| Where to train | On the owner's PC; maybe on TrueNAS | PC is the default everywhere; TrueNAS becomes an optional "train on my server" milestone (§10). |
| GM or Player model | **GM model only, trained on all data** | Player/audience filtering and leak scans are removed. GM secrets are *kept* and labelled. A warning about player-facing AI is added (§8.3). |
| Which world | Chosen by the owner | The GUI lists the worlds nd-world offers and lets the owner pick one per project. |
| Teacher model | Local PC runs Gemma 4 and Qwen3.5; TrueNAS runs Gemma 4 | Teacher can be a model on the PC (Ollama / llama.cpp) or the TrueNAS Gemma 4 over the network (§6.4). |
| Models | Qwen3.5 2B, 4B, 9B; Gemma 4 E2B, E4B | Exactly these five in the model picker (§7.1). |
| Getting data | Either **connect nd-world to the program**, or a **special export file from nd-world** | New nd-world feature: the **Training Bundle** — one format, delivered either as a downloaded `.ndtrain` file or over a live connection (§5). | *(Superseded in revision 3: file only.)*
| GUI | Easy to understand, with detailed instructions | Guided step-by-step design with an instructions panel on every page, plain-language help for every setting (§4). |

### What changed in revision 3 (owner's answers, 2026-11-07)

| Topic | Answer | Effect on the plan |
|---|---|---|
| PC GPU | **RTX 5090 (32 GB)** | Default training PC. All five models fit with PEFT; 9B can use 16-bit LoRA; Full fine-tune is possible for the small models (§7). The 5090 is a new Blackwell card that needs a recent CUDA/PyTorch, so M4 verifies it first (§14). A bigger teacher model also fits on the PC (§6.4). |
| TrueNAS before teacher work | Yes | TrueNAS training moves to **M8**, before teacher-written examples (§12). |
| Name and `.ndtrain` extension | OK | Unchanged. |
| Fine-tuning versions | **Full, PEFT, SFT** | Training page offers these as clear choices (§7.2). |
| Getting data | **Export file only**, no linking to nd-world | The live "Connect" mode is **removed** from version 1 (§5). No token API, no Bearer auth, no extra nd-world security surface: only a download button. |

---

## 1. Goals and non-goals

**Goals**
1. One GUI program for Windows 10/11 and Linux (x86-64, NVIDIA GPU) that a non-expert can follow from start to finish:
   **Get world data → Make training data → Train → Test → Use in nd-world.**
2. Get the world in with **one file**: export a `.ndtrain` Training Bundle from nd-world and drop it into the program. No
   connection between the two programs is needed.
3. Train the five chosen models with **PEFT (LoRA/QLoRA), Full fine-tuning and SFT** (§7.2) on one GPU: the RTX 5090 PC (default) or the TrueNAS server (Tesla T10, 16 GB).
4. A **GM model** that knows everything in the world, including secrets, and knows which facts are secrets.
5. Produce files nd-world loads straight away: GGUF + Ollama `Modelfile`, or a model folder for Unsloth Studio.

**Non-goals (version 1)**
- Player-safe models (not needed; can be added later because the bundle keeps visibility flags).
- Training from scratch, models other than the five listed (an "advanced: other Hugging Face model" box may come later).
- AMD / Intel / Apple GPUs; multi-GPU training. CPU works for data preparation only.
- Image/audio training — Qwen3.5 and Gemma 4 are multimodal, but only the text part is trained.

## 2. How the model learns the world

Fine-tuning alone memorises facts poorly and makes a small model invent details, so training uses three layers. The GUI
explains each in plain words; the owner never has to know the terms.

| Layer | Plain-language name in the GUI | What it does |
|---|---|---|
| Continued pre-training (optional) | **"Read the world book"** | The model reads all lore as text, learning names, style, rules wording. |
| Supervised fine-tuning (main step) | **"Practise answering questions"** | Question/answer and chat examples: lore questions, rules explained, NPC role-play, and saying "that isn't in the lore" instead of inventing. |
| Retrieval stays on in nd-world | **"Use nd-world's lore search"** | Part of the practice examples include a context block like the one nd-world adds to its prompts, so the model uses nd-world's search results well instead of replacing them. |

---

## 3. Big picture

```
 nd-world (TrueNAS)                                   ND Lore Forge (RTX 5090 PC)
 ┌──────────────────────────────┐                     ┌───────────────────────────────────────────┐
 │ Export & Backup hub:         │                     │ 1 Import the .ndtrain file                │
 │  "Training bundle" button    │ ── world.ndtrain ─► │ 2 Make training data ◄── teacher model    │
 │  (the only nd-world change)  │    (copy the file)  │      (PC Ollama/llama.cpp or TrueNAS)     │
 └──────────────────────────────┘                     │ 3 Train  (PC GPU, or TrueNAS worker)      │
                                                      │ 4 Test                                    │
                                                      │ 5 Use in nd-world → GGUF / Ollama ────────┼──► back to nd-world
                                                      └───────────────────────────────────────────┘
```

## 4. The GUI: easy to understand, with detailed instructions

### 4.1 Design rules
1. **A guided path, not a pile of settings.** The main window is five numbered steps in a left sidebar with a progress
   tick for each. A step unlocks when the one before it is done; the next action is always one big highlighted button.
2. **An instructions panel on every page** (right side, can be collapsed), always with the same four headings:
   *What this step does* · *What you need before you start* · *What to do* (numbered, with small screenshots) ·
   *What happens next / how long it takes*.
3. **Every setting has a "?"** that opens a plain-language explanation, a recommended value, and what happens if it is
   set higher or lower. Example for "Epochs": *"How many times the model reads all practice examples. 2–3 is usual. More
   can make it repeat the examples word-for-word instead of understanding them."*
4. **Simple mode by default, Expert mode on a switch.** Simple mode shows only: world, model, "Quick test / Normal / Best
   quality", and Start. Every expert value still has a safe default computed for the PC's GPU.
5. **Check before you start.** Each step shows a checklist with green / yellow / red lights (GPU found, enough VRAM,
   enough disk space, teacher model reachable, nd-world reachable) and estimated **time** and **disk space** before the
   owner presses Start.
6. **Errors in plain language with a fix.** Never a raw stack trace first. E.g. *"The GPU ran out of memory. The 9B model
   needs more memory than this card has with these settings. [Use the recommended smaller settings] [Pick the 4B model]"*;
   the technical log is one click away ("Show details", "Copy for a bug report").
7. **Show, don't just tell.** During training, a few fixed probe questions about the chosen world are answered every
   few minutes, so the owner sees the answers improve. The loss chart has a short "what does this chart mean" note.
8. **Built-in help:** a first-run welcome tour (5 screens), a full user guide inside the app (Help → Guide, also in
   `docs/`), a glossary (LoRA, epoch, loss, quantisation, GGUF, VRAM …) linked from every page, and a troubleshooting page.
9. **Long jobs can be left alone:** a desktop notification when a job finishes or fails; jobs resume after a restart.
10. Light/dark theme following the system; text size setting; keyboard accessible. UI language: English first, all
    strings in translation files so other languages can be added.

### 4.2 Pages

| Page | What the owner sees |
|---|---|
| **Welcome / Projects** | "Create a project" (one project = one world + one model family), recent projects, the welcome tour. |
| **1 · Get world data** | One big card: **"Import a training file"** (drag and drop a `.ndtrain`; the file already contains the one world you exported, so there is nothing to pick) and a small **"Update from a newer file"** option for later. Instructions explain how to make the file in nd-world, with screenshots. After import: a summary ("312 entities, rules 48 pages, 27 session recaps, 140 facts …") and, on re-import, what changed. A third small option: "I only have an old World JSON export" (works, with less data). |
| **2 · Make training data** | Recipe with on/off switches and a "how much" slider per kind of example (§6.3), teacher model setup with Test button, time estimate, Generate with progress, then a review table (search, filter, edit, delete, "looks good") and **Save this version**. |
| **3 · Train** | Model picker with the five models as cards (size, VRAM needed, speed, "fits your GPU ✓/✗"), training type (PEFT / Full, with SFT and optional world-book stage; see 7.2), where to train (This PC / My TrueNAS server), preset, Start / Pause / Stop, live charts, probe answers, ETA, GPU temperature. |
| **4 · Test** | Score card (lore accuracy, made-up answers, rules questions, general chat), side-by-side chat: original model vs trained model, "ask your own question", save transcripts. Plain verdict: *"Better than the original on 86 % of lore questions."* |
| **5 · Use in nd-world** | Choose file size/quality (Q4 / Q5 / Q8), Create, then step-by-step instructions to load it in nd-world (Ollama or Unsloth Studio) with copy buttons for each command and setting; optional "Send to my TrueNAS Ollama" button. |
| **Settings** | GPU info, installed training environment (install / repair / update), folders, TrueNAS worker address and tokens (stored in the OS keyring), teacher model defaults, theme. |

---

## 5. Getting world data: the nd-world Training Bundle

### 5.1 File only
nd-world gets a **Training Bundle** download: Tools → 📦 Export & Backup → **"Training bundle (for ND Lore Forge)"** →
downloads `<world-slug>.ndtrain`. Copy it to the PC and drop it on page 1. It works offline and with nd-world on TrueNAS and
the PC anywhere else. To retrain after the world changed, export again and use "Update from a newer file": the program
shows what changed and rebuilds only the affected examples.

A live connection (token + address) is **deliberately not built**: it would need a new login method and a new API in
nd-world's Python app and Rust gateway for little gain. The bundle format is versioned, so a connection could be added later
on top of the same format without changing anything else. An existing **World JSON export** from any nd-world version can
also be imported (entities only, less data).

### 5.2 Bundle format (`.ndtrain`, format version 1)
A zip with UTF-8 JSON/JSONL files (a JSONL line = one record), so it streams for big worlds and is easy to inspect:

| File | Contents |
|---|---|
| `manifest.json` | `format: "nd-training-bundle"`, `version: 1`, created time, nd-world build (`ND_BUILD`), world id/name/slug/description, record counts, SHA-256 of every file. |
| `world.json` | Kinds (built-in + the world's custom kinds), entity templates (field ids → labels, so custom fields become readable), calendar config, ruleset names used by combat. |
| `entities.jsonl` | Every entity: kind, subtype, name, **aliases**, folder, tags, summary, body (markdown, **with `[gmonly]` and other tags kept raw**), custom fields, **roleplay_personality**, `visible_to_players`, `rag_priority`, updated_at. |
| `entity_notes.jsonl` | Notes attached to entities (with their visibility flag). |
| `relations.jsonl` | Typed relations (`located_in`, `member_of` …) and the untyped "related" links, by entity id + name. |
| `rules.md` + `rules_sheets.jsonl` | The world's effective rules text (the core rules fallback if the world has none) and rules text from sheet templates. |
| `sessions.jsonl` | Game sessions: number, date, title, recap, NPCs present, loot. |
| `facts.jsonl` | Session facts with their session and visibility. |
| `quests.jsonl` | Quests with status, category, body, linked entities, parent quest. |
| `tables.jsonl` | The world's random tables (name, category, description, entries + weights). |
| `ai_instructions.jsonl` | The GM's standing AI instructions (enabled ones), so the model can learn the house style. |
| `characters.jsonl` | Player characters: name, race, class, level, backstory, notes, equipment, feats. **No user accounts, emails or owner ids.** |

**Never included:** users, e-mails, passwords, tokens, 2FA, images/audio/video, maps and schematics geometry, dice
history, live/combat state, **character journals and goals** (private to the player even from the GM in nd-world) and
**GM→player private notes** (personal). These stay out even though the model is GM-only, because the bundle may be
copied to another machine.

### 5.3 Changes in nd-world (separate work in the nd-world repo, following its CLAUDE.md)
The only new surface is **one GM-only download** that uses the existing browser login. Python is the oracle, the Rust
gateway must answer byte-identically, and the Rust-only image must not lose the feature.

1. **Python first** — `app/training_bundle.py`: one streamed builder (reuse `app/streaming_export.py`, so a large world does
   not sit in memory) and the route `GET /worlds/{id}/export/training.ndtrain`, GM only. A button with a one-line explanation
   on the Export & Backup hub.
2. **Rust gateway port** — the same route in `rust/nd-gateway/src/export.rs` and the hub button in the Rust renderer. In
   `python_free()` mode it must not decline on accented names, rich-text notes, `null` where a list is expected, or thousands
   of rows.
3. **Tests** — a parity test in `tests/test_gateway_golden.py` (same world → same bytes from both servers) and a Rust-only test
   with `_python_free_gateway(seed)`; a player or logged-out request must be refused.
4. **Docs** — `docs/FORK_STATUS.md`, `docs/RUST_SWITCH_OFF.md`, `docs/API_REFERENCE.md`, README "Data & Backups".
5. Full suite before pushing (`bash scripts/gateway_suite.sh 4 all` + `pytest -n 4`).

Zip byte-identity needs a fixed file order, fixed timestamps and one agreed compression in both implementations; the parity
test also compares the unpacked files, so a mismatch points at the file that differs.

---

## 6. Making training data (page 2)

### 6.1 Import and normalise
The bundle (or World JSON) is read into one canonical record per item, saved in the project as a **snapshot** with a
content hash. Re-fetching shows a diff (added / changed / removed) and later steps only redo what changed.

### 6.2 Cleaning, GM secrets kept and labelled
- Because this is a GM model, `[gmonly]…[/gmonly]` content is **kept** but rewritten as a clear label, e.g.
  *"(GM secret: the duke is the cult leader)"*, so the model learns the fact *and* that players don't know it — useful when
  the GM asks "what don't my players know about X?". Same for `[for=…]` (→ "known only to <player character>") and
  `[reveal=DATE]` (→ "revealed on DATE"), and hidden entities (→ "hidden from players").
- Other nd-world markup (`[spoiler]`, `[color]`, `[mark]`, `[u]`, autolinks, image embeds) is unwrapped to plain markdown;
  `[[wikilinks]]` become names. The tag grammar follows nd-world's `app/rendering.py`; tests use fixtures produced by
  nd-world's own code so tags are recognised the same way (nesting, unclosed tags, mixed case).
- Long texts are split by heading into chunks that keep the entity name; near-duplicates are removed; accented names are
  kept exactly.

### 6.3 Kinds of practice examples (each with an on/off switch and a "how much" slider)
1. **World book** (for "Read the world book") — every record as a readable article; rules as chapters.
2. **Simple facts** (no AI needed) — from structured data: who/what is X, also known as, where is X (relations), members
   of an organisation, custom fields, random table contents, "how does <rule> work".
3. **Deeper questions** (teacher model) — per chunk, grounded questions and answers: explain, compare, connect two related
   entities, "what would happen if", multi-turn chats.
4. **Rules and mechanics practice** (teacher model) — worked examples for the world's rules (dice pools, combat turns for
   its ruleset: nd / worm / hunt / asterion / generic), checked against the rules text.
5. **NPC role-play** — in-character dialogues for entities with a `roleplay_personality`, using the same system prompt shape
   as nd-world's Talk to NPCs.
6. **Campaign history** — from session recaps, facts and quests: "what happened in session 12", "who did the party meet in
   …", quest status — the Chronicler's job.
7. **Using lore search** — questions answered with an nd-world-style context block, including cases where the context lacks
   the answer and the right reply says so.
8. **"Not in the lore"** — questions about made-up names and events → a short honest "that isn't in this world's lore".
9. **General chat** (default ~10 %) — a small open instruction dataset so the model does not forget ordinary conversation.
10. **House style** — the GM's AI instructions become the system prompt of a share of examples.

**Checking the teacher's work:** every teacher-written example is checked (by the teacher in "judge" mode): *is every
claim supported by the source text?* Unsupported examples are dropped or flagged for review. Generation is cached per
chunk, resumable, and can run overnight.

**Test questions are kept apart:** ~10 % of facts/entities are held out *by fact* (not by rewording) for page 4, so the
test measures what the model learned, not what it copied.

### 6.4 Teacher model options
Any OpenAI-compatible endpoint; the GUI offers presets with a Test button:
- **On this PC** — Ollama or a llama.cpp server running Gemma 4 or Qwen3.5 (a bigger size than the one being trained is
  better; the 5090's 32 GB can hold a quantised Gemma 4 26B-A4B or Qwen3.5 9B as teacher). The program generates first, then asks Ollama to unload the model (`keep_alive: 0`) before training, so they
  never fight for the GPU.
- **On TrueNAS** — the server's Gemma 4 (26B) through Unsloth Studio or Ollama, over the network. Can generate while the
  PC is free; warning shown not to run it on a session night (it shares the T10 with chat and transcription).
- **None** — only "World book" and "Simple facts" (no teacher needed); fine for a first try.

### 6.5 Review and save
The review table shows every example next to the text it came from; filter by kind, source, judge score; edit, delete,
mark good. **"Save this version"** writes `datasets/vN/train.jsonl`, `test.jsonl` and `manifest.json` (bundle hash,
recipe, counts, token totals). Training always uses a saved version, so every trained model can say exactly what it learned from.

---

## 7. Training (page 3)

### 7.1 The five models
Shown as cards with a "fits your GPU ✓/✗" mark for each training type. Figures are first estimates at 2048 tokens and are
**measured on the RTX 5090 in milestone M6** before the GUI treats them as facts. Gemma 4 "E" sizes are *effective* parameter
counts (the real weights are larger), which the estimate accounts for.

| Model | Good for | PEFT QLoRA (4-bit) | PEFT LoRA (16-bit) | Full fine-tune |
|---|---|---|---|---|
| Qwen3.5 2B | Fast tries | ~5–6 GB | ~8–10 GB | ✓ on 5090 (~20–28 GB) |
| Gemma 4 E2B | Same family as nd-world's chat model | ~6–8 GB | ~12 GB | ✓ probably on 5090; measured in M9 |
| Qwen3.5 4B | Good balance | ~8–9 GB | ~14–16 GB | borderline on 5090 (8-bit optimizer, short context); measured in M9 |
| Gemma 4 E4B | Good balance, Gemma family | ~9–11 GB | ~18–20 GB | ✗ too big for 32 GB (offered only on a bigger card) |
| Qwen3.5 9B | Best quality of the five | ~12–14 GB | ~24–28 GB (✓ on 5090, shorter context) | ✗ |

The exact Hugging Face repos live in `catalog/models.toml`, checked against the hub when implementation starts. Qwen3.5's
"thinking" mode is trained **off** by default, with an expert option to keep it.

### 7.2 Training types: Full, PEFT and SFT
Page 3 starts with a plain-language choice. **Full and PEFT say *which weights change*; SFT says *what the model is taught
from*** (question/answer examples). So SFT is not an alternative to the other two: it is the teaching step you run *with* either
one. The GUI shows it as two short questions:

**Question 1 — How much of the model should change?**

| Card | What it means in plain words | When to choose it |
|---|---|---|
| **PEFT** (LoRA / QLoRA) — *recommended* | Trains a small add-on layer and leaves the original model untouched. Needs far less memory, trains fast, and the result can be switched off or re-done cheaply. Sub-choice: *QLoRA* (smallest memory) or *LoRA 16-bit* (a bit better quality, more memory). | Almost always. The only type for the 9B model, Gemma 4 E4B and the TrueNAS server. |
| **Full fine-tune** | Changes every weight of the model. Strongest learning of a world's vocabulary, but needs much more memory, runs the risk of the model forgetting how to chat, and produces a full-size copy of the model. | Experiments on the small models (Qwen3.5 2B, maybe Gemma 4 E2B / Qwen3.5 4B) on the 5090. Shown only if the memory estimate says it fits; otherwise greyed out with the reason. |

**Question 2 — What should it learn from?** (stages; tick one or both, in this order)

| Stage | Name in the GUI | Data |
|---|---|---|
| Stage 1 (optional) | **Read the world book** (continued pre-training) | The raw lore text. |
| Stage 2 (main) | **Practice answering questions (SFT)** | The question/answer, chat, role-play and "not in the lore" examples from §6.3. |

Typical combinations: **PEFT + SFT** (default, *Normal* preset), **PEFT + world book → SFT** (*Best quality*), **Full + SFT**
(expert, small models). Each combination gets a one-sentence summary, a memory and time estimate, and a recommendation for
the detected GPU.

Not included: preference tuning (DPO/ORPO) and reinforcement learning. They can be added later without changing the data pipeline.

**Engine and settings**
- **Unsloth** for PEFT and Full where it supports the model, with plain Transformers + PEFT + TRL as the fallback backend.
- Chat format from each model's own tokenizer template; loss only on the assistant's answers.
- **Presets:** *Quick test* (~10–20 min, proves everything works), *Normal*, *Best quality*, computed from GPU, model, type and
  dataset size. Precision is chosen automatically: **bf16 on the RTX 5090**, fp16 on older cards like the T10 (Turing); full
  fine-tuning uses bf16 with an 8-bit optimizer and gradient checkpointing to save memory.
- Live: loss/test-loss chart, speed, VRAM, GPU temperature, ETA, probe answers; Pause / Stop / Resume; out-of-memory →
  plain explanation and one-click retry with smaller settings (shorter context, PEFT instead of Full).
- Each run: `runs/<name>/` with settings, logs, checkpoints, the result (adapter for PEFT, full weights for Full) and scores.

### 7.3 Where to train
- **This PC (default, RTX 5090)** — the training environment is installed by the program on first use (see §9). All types.
- **My TrueNAS server (optional, milestone M8)** — see §10. The T10 (16 GB, fp16) supports PEFT only.

---

## 8. Test and use (pages 4 and 5)

### 8.1 Test
Automatic after training, original vs trained model, on the held-out questions: **lore accuracy** (fuzzy match + judge),
**made-up answers** (share of "not in the lore" questions answered with inventions), **rules questions** with known
answers, **general chat** sanity, plus the side-by-side chat. Runs can be compared in a table.

### 8.2 Use in nd-world
1. Merge the trained adapter into the model; convert to **GGUF** and shrink it (Q4_K_M default, Q5_K_M, Q8_0).
2. Write an Ollama `Modelfile` (chat template, stop words, recommended context and temperature) and a model card
   (base model, world, dataset version, scores, licence: Gemma terms / Qwen licence).
3. Install it, with instructions and copy buttons for each path:
   - **Ollama on the PC:** `ollama create <world>-gm -f Modelfile` (button does it).
   - **Ollama on TrueNAS (legacy backend):** copy the GGUF + Modelfile to the server (button: upload over the network if
     the Ollama API is reachable), then set `OLLAMA_MODEL=<world>-gm` in nd-world.
   - **Unsloth Studio (current backend):** put the GGUF where Studio's Model Hub sees it (or a private Hugging Face repo),
     set `UNSLOTH_MODEL` and `LLM_CONTEXT_TOKENS` to the context the model was loaded with.
   - Keep nd-world's lore retrieval on.

### 8.3 Important warning shown on page 5
This model knows **all** GM secrets. If players in this world can use AI (the per-world "players can use Ask AI" option),
they would be talking to a model that knows the twists. The page tells the GM to either keep player AI off for that world
or keep player-facing AI on the original model. (A later nd-world option "separate model for player-facing AI" is listed as
a possible follow-up, not part of this plan.)

---

## 9. Technology and installation

| Part | Choice | Why |
|---|---|---|
| Language | Python 3.12 | The training stack is Python. |
| GUI | **PySide6 (Qt 6)** + pyqtgraph | Native on Windows and Linux, good for long background jobs, rich help panels (HTML), LGPL. |
| Training | Unsloth; fallback Transformers + PEFT + TRL | See §7.2. |
| Data | `datasets` + JSONL on disk | Easy to inspect and version. |
| Teacher | OpenAI-compatible HTTP client (httpx) | Works with Ollama, llama.cpp, Unsloth Studio. |
| Export | llama.cpp convert + quantise (or Unsloth's GGUF export), Ollama API/CLI | GGUF is what nd-world's backends load. |
| Environment | **uv**-managed environment created by the app | Torch/CUDA is large and GPU-specific; the installer stays small and the app picks the right CUDA build. |
| Packaging | PyInstaller → Inno Setup installer (Windows), AppImage + tar.gz (Linux) | One-click install. |
| Secrets | OS keyring (Windows Credential Manager / Secret Service) | Hugging Face and TrueNAS-worker tokens never in project files or logs. |

**Process model:** the GUI never loads torch. Every long job (import, data generation, training, testing, export) runs as a
**separate worker process** that reports progress as JSON lines; Stop always ends it and frees the GPU, and a crash cannot
take the window down. The same worker has a command line (`ndforge train project.toml`), which is also what the TrueNAS
worker runs.

**First run on a PC:** the welcome tour ends at "Set up training": the app checks the NVIDIA driver, picks the CUDA build,
installs the environment with a progress bar (several GB, one time), and runs a 30-second GPU self-test. If the driver is
too old it says which version is needed and links the NVIDIA download page.

## 10. Optional: training on TrueNAS

- A Docker image **`nd-lore-forge-worker`** (built from this repo, `Dockerfile.worker`) with the training environment and a
  small job API (HTTPS + token; start job, stream progress, download results). A ready `truenas-compose.yml` snippet with
  GPU passthrough is provided, in the same style as nd-world's own compose files.
- In the GUI: Settings → "My TrueNAS server" (address + token + Test), then page 3 → "Where to train: My TrueNAS server".
  The project's saved dataset is uploaded, progress and charts look exactly like local training, the result comes back
  to the PC (or is installed directly into the server's Ollama).
- GPU sharing on the T10: before starting, the worker asks Ollama / Unsloth Studio to unload their models and warns that AI
  chat, transcription and image generation will be slow or unavailable until training ends; precision fp16, PEFT only. All five
  models fit the 16 GB card with QLoRA (9B with the shorter context preset).

---

## 11. Repository layout (this repo)

```
Llm-training/
  pyproject.toml              # ndforge (GUI side: PySide6, pyqtgraph, pydantic, keyring, httpx)
  requirements/               # worker environment, pinned: cuda12x.txt, cpu.txt
  catalog/models.toml         # the five models: repos, chat template notes, VRAM measurements
  src/ndforge/
    app.py                    # entry point
    gui/                      # main window, steps/, instructions panel, help/glossary viewer, charts, job monitor
    help/                     # page instructions, setting explanations, glossary, guide (markdown + screenshots)
    core/                     # project.py (project.toml), settings, jobs.py (worker process + JSON-line protocol)
    envmgr/                   # uv bootstrap, GPU/driver/CUDA detection, self-test
    ingest/                   # bundle.py (.ndtrain), world_json.py (legacy), snapshot + diff
    dataset/                  # canonical.py, markup.py (GM-secret labelling), chunk.py, builders/*.py, verify.py, split.py, freeze.py
    teacher/                  # OpenAI-compatible client, prompts, cache, Ollama unload
    train/                    # worker.py, presets.py, vram_estimate.py, modes/{peft,full}.py, stages/{world_book,sft}.py, backends/{unsloth,hf}.py, probes.py
    evaluate/                 # metrics.py, judge.py, compare.py, playground.py
    export/                   # merge.py, gguf.py, ollama.py, model_card.py
    remote/                   # TrueNAS worker client + server (job API)
    cli.py                    # ndforge import|build|train|test|export
  docs/                       # user guide, bundle format spec (shared with nd-world), troubleshooting
  tests/                      # unit, bundle fixtures, markup fixtures from nd-world, tiny-model smoke run, pytest-qt
  packaging/                  # pyinstaller spec, windows/ (Inno Setup), linux/ (AppImage), Dockerfile.worker
  .github/workflows/          # lint + tests on windows-latest and ubuntu-latest; installers on tag
```

The **bundle format spec** (`docs/BUNDLE_FORMAT.md`) is written first and copied to nd-world's docs, so both repos build
against the same contract; sample bundles from the spec are test fixtures on both sides.

## 12. Milestones (in order; each ends with something usable)

| # | Milestone | Where | Done when |
|---|---|---|---|
| **M0** | Bundle format spec v1 + sample bundles | both | Spec agreed; fixtures committed to both repos. |
| **M1** | App skeleton: guided window, 5 steps, instructions panel, help/glossary, settings, projects; CI on Windows + Linux | Lore Forge | Clicking through all pages works on both OSes (steps empty). |
| **M2** | nd-world Training Bundle download (Python + Rust, parity + Rust-only tests, docs) | nd-world | Export hub downloads a `.ndtrain` from both images; suite green. |
| **M3** | Page 1: import `.ndtrain` and legacy World JSON, snapshot + diff, summary | Lore Forge | A real bundle of the chosen world imports with correct counts. |
| **M4** | Environment manager: driver check, uv install, GPU self-test, **RTX 5090 (Blackwell) verified** | Lore Forge | Fresh Windows and Linux PCs reach "GPU ready"; the 5090 trains a tiny model. |
| **M5** | Data: cleaning + GM-secret labels, World book + Simple facts, held-out split, review table, save version | Lore Forge | A saved dataset without any teacher model. |
| **M6** | **PEFT + SFT training on the PC**: worker, Unsloth QLoRA/LoRA, presets, live charts, probes, stop/resume; **VRAM measured for all five models** | Lore Forge | *Quick test* trains each of the five on the 5090; Stop frees VRAM. |
| **M7** | Export + install: GGUF, Modelfile, Ollama on PC, instructions for TrueNAS Ollama / Unsloth Studio, GM warning | Lore Forge | **First complete version:** the trained model answers in nd-world's AI Chat. |
| **M8** | **TrueNAS worker**: image, compose snippet, remote training from the GUI (PEFT) | Lore Forge | A run started on the PC trains on the T10 and the result comes back. |
| **M9** | **Full fine-tune** type + "Read the world book" stage; memory estimates measured; type-choice cards in the GUI | Lore Forge | Full + SFT runs on the small models that fit the 5090; unsuitable combinations are greyed out with reasons. |
| **M10** | Teacher-generated examples (deeper questions, rules practice, NPC role-play, history, lore search, not-in-lore, general, house style) + checking | Lore Forge | Better test scores than M5-only data on the same model. |
| **M11** | Test page: scores, side-by-side chat, run comparison (Full vs PEFT vs both) | Lore Forge | Original vs trained report after every run. |
| **M12** | Installers (Inno Setup, AppImage), in-app guide with screenshots, troubleshooting | Lore Forge | Install from a release on clean Windows 11 and Ubuntu 24.04 and follow the guide end to end. |

## 13. Testing

- **Lore Forge:** unit tests for import, markup labelling (fixtures produced by nd-world's own tag code), chunking, builders,
  held-out split, manifest hashing, presets and VRAM maths; a tiny-model CPU smoke run through the real worker protocol in
  CI; pytest-qt tests for the guided flow and plain-language error dialogs; manual GPU runs of the five models in each allowed training type.
- **nd-world:** parity (Python vs gateway) and Rust-only tests for the bundle download, access tests (player / logged-out refused), a big-world test
  (thousands of entities, accented names, `null` lists) to make sure the export streams and never declines.
- **End to end:** sample world → bundle → dataset → Quick test training → GGUF → Ollama → question answered.

## 14. Risks

| Risk | Mitigation |
|---|---|
| Small model still invents lore | Grounded and "not in the lore" examples, nd-world retrieval kept on, "made-up answers" score shown prominently. |
| Players reach the GM model through Ask AI | Warning and instructions on page 5 (§8.3). |
| Bundle leaves the server with all secrets | Text only, no accounts; the instructions say to treat `.ndtrain` files like backups; GM-only download. |
| Python and Rust bundles drift | One spec, shared fixtures, byte-parity test in nd-world's suite. |
| RTX 5090 is a new architecture (Blackwell, needs CUDA 12.8+ and a matching PyTorch) and some training libraries lag behind | M4 verifies a pinned working set on the 5090 first; HF-based fallback backend; clear driver/CUDA messages. |
| Full fine-tuning makes the model forget or overfit | Offered only on small models, low learning rate, mixed-in general examples, test page compares it with PEFT before it is exported. |
| Windows training quirks | Unsloth's Windows support, pinned tested wheels, fallback backend, self-test with fixes. |
| GPU contention (teacher vs training on the PC; chat/STT vs training on TrueNAS) | Generate first, unload, then train; unload + warning on TrueNAS. |
| Model releases change | `catalog/models.toml`, tokenizer chat templates, fallback backend. |
| Disk space | Shared model cache, checkpoint retention setting, space check before each save. |

## 15. Remaining questions (not blocking the start)

1. Is the CPU/RAM of the PC worth noting (for data preparation speed)? Not needed, the program detects it.
2. When a world changes often, is "export a new file → Update from a newer file" convenient enough, or should a one-click "retrain with the new export" button be added to M12? (Planned as a button on page 1 either way.)
