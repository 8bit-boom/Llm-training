# Heal a model: any model, not only the ones the Slim page made

Request (the owner): "On Heal mode section I should be able to choose model to heal."

Before this change the Heal card existed only inside the result card of a model made on the Slim or grow page, and the core required
`models/<name>/model/ndforge_reshape.json`: the student was always the made folder and the teacher always the recorded original. Now the page has a
**Heal a model** section that stands on its own, with two pickers, a Check, and then the same heal UI bound to the chosen pair. Healing is still distillation
(docs/RESHAPE.md, docs/HEAL_SETTINGS.md); only the choice of the two models changed.

## What the person sees

* **Model to heal** (student): *Made by this program* (the first is preselected), *My models* (entries of the model list that are folders or Hugging Face ids),
  *Earlier heals of other models*, *Models that come with the program* (allowed, with a warning: healing an unmodified model is pointless),
  *A folder on this computer...* (path field and Browse) and *A Hugging Face model...* (id field).
* **Original model** (teacher): for a made model the recorded original (default); for any other student the person must choose (model list, folder, Hub id).
  When the student's model card names a base model (`base_model:` in the README front matter, else `_name_or_path` of config.json when it is a Hub id) the
  list offers it as **a guess**, labelled as one; it is never chosen for the person.
* **Check** (explicit button; reads files only): a short list with green ticks, yellow warnings, red refusals and grey notes, then one sentence saying whether
  the pair can be used. Nothing is loaded, nothing is written, a Hugging Face model is not downloaded by it.
* Under the Check the existing Heal card: general text chooser, tokens, estimate, Expert settings, Start / Stop / Continue, "safer settings" after an
  out-of-memory error, the chart, and for a made student the original / thinner / healed comparison in its result card. Start stays disabled until the pair
  passed the Check **and** the old conditions hold (training-data version saved, training software installed, the memory estimate does not say "does not fit").
* A pair that is not "a made model with its recorded original" is healed from the section; that one stays in the result card of the made model (the same code
  `healCard`, the same state `rs.hl` / `rs.hlSpec` / `rs.hlRun`; the section shows a pointer instead of a second copy, so there is never more than one set of
  controls on the page).
* An earlier heal of a model from elsewhere appears in *Earlier heals of other models*; it can be selected again (Continue) or deleted with its files.
* The healed copy is registered in the model list as "<student> · healed", `derived_from` = the student's folder or Hugging Face id, with a note that it is a
  derivative of both models and the licences must be checked. The Check shows the licences it can read (model card front matter; the Hub API when it answers;
  never a failure when it does not).

## What the Check decides

| Line | Level |
|------|-------|
| a model that cannot be read (no config.json, no readable .safetensors, not a Hub name, Hub unreachable) | refused, with the reason |
| GGUF only, or `quantization_config` in config.json | refused ("full-precision .safetensors only", as on the Slim page) |
| family not supported (`reshape.check_supported`: Qwen3.5 / Qwen3 / Qwen2 / Llama / Mistral-style; refused families and custom code) | refused, with the reason |
| tokenizer files missing in one of the two | refused |
| the two tokenizers do not give the same token ids (see "Which tokenizer files are compared") | refused: "do not use the same tokenizer"; a size difference gives both vocabulary sizes, a probe difference shows the first differing text with both id lists |
| tokenizer files differ, but both give the same ids on the probe texts | warning: "The tokenizer files differ but both produce the same token ids on N test texts; healing can go on." |
| the two models use different chat templates | warning only, never a refusal (see below) |
| the two are the same model | refused |
| no teacher chosen | refused, "Choose the original model" |
| a Hub model that is not downloaded and not enough free disk | refused |
| teacher smaller than the student | warning |
| output layers of different sizes (padding) | warning |
| teacher in 4-bit (+2 GB) larger than the card | warning (the estimate below says whether healing fits) |
| download size, disk tight | warning |
| an unmodified model of the program as the student | warning |
| hidden sizes differ | grey note: fine, distillation uses output probabilities only |
| licences | grey note, always |

The worker job `heal_check` (`worker/ndforge_worker/healcheck.py`) reads the files; the comparison (`reshape::heal_source::verdict`) is in Rust. The facts of a
model are kept in the settings folder (`heal_facts.json`): a folder until its config.json changes, a Hugging Face model for ten minutes; the **Check** button
asks again every time.

## Which tokenizer files are compared

Distillation needs one thing of the two tokenizers: the same text gives the **same token ids** (the teacher's answers are per-position probabilities over the
shared vocabulary). The Check therefore no longer compares every tokenizer file. `worker/ndforge_worker/tokpair.py` decides a pair in three steps; the Check
(`heal_check`) and the two places in `distilljob.py` where the jobs compared the tokenizers (`run_cache`, and the online path of `run`) use the **same function**
(`tokpair.decide_pair`), so a pair the Check accepted is accepted by the job.

1. **Ids hash.** sha256 over only the files that decide ids: a canonical form of `tokenizer.json`, and `vocab.json`, `merges.txt`, `tokenizer.model`,
   `added_tokens.json` as they are. Equal => the same tokenizer (green tick, no probe).
   * In `tokenizer.json` **in**: `model` (type, vocab, merges, unk token, ...), `added_tokens` (content, id, special, normalized, lstrip, rstrip, single_word; their
     order does not matter), `normalizer`, `pre_tokenizer`, and the `post_processor` (the part that inserts special tokens: `TemplateProcessing`, Bert / Roberta
     processors and unknown ones are kept whole; `ByteLevel` keeps only its type, its offset flags do not change ids).
   * **Out**: `padding`, `truncation`, `decoder`, `version`, any other key, and the key order. A `tokenizer.json` that is not readable JSON is hashed as raw bytes.
   * **Out** (not hashed at all): `tokenizer_config.json` (chat template, padding side, `model_max_length`, clean-up flags, `name_or_path`, ...) and
     `special_tokens_map.json` (it only names special tokens).
   * The rule is **conservative on purpose**: any difference in the included parts (a cosmetic one such as a differently written normalizer too) is not accepted by
     the hash; it goes on to the probe, and is refused if the probe cannot show the ids are the same.
2. **Vocabulary sizes** (tokenizer files: base vocabulary plus added tokens, where the files say it): different => refused, with both sizes, whatever the probe
   would say. The `vocab_size` of `config.json` (output rows) differing is still only the padding warning.
3. **Probe** (only when the ids hashes differ). Both tokenizers are loaded with `transformers.AutoTokenizer` (import guarded) and encode the 40 texts of
   `tokpair.PROBES` with `add_special_tokens=False`: ASCII words and sentences, punctuation, digits and numbers, dates, code (Python, C, SQL, HTML, JSON), URLs,
   mixed case, accented Latin, Cyrillic, Greek, CJK, Hangul, Arabic, Hebrew, Devanagari, emoji (with skin tone, flag, variation selector), runs of spaces, leading /
   trailing space, tabs, `\n` / `\r\n` / empty lines, the chat special tokens of the families (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<s>[INST]`, `<think>`,
   `<start_of_turn>`, `<|begin_of_text|>`) and an empty-ish text (a single space). **All equal => accepted with the yellow warning.** Any difference, any load or
   encode error => refused, with the first differing text and both id lists (the first 16 ids). Without the `transformers` library the probe is skipped and the
   pair is refused with the sentence saying the test texts were not run (a refusal in doubt would block the feature, but a wrong acceptance corrupts the healing
   signal, so a probe that could not run does not accept).

`distilljob.tokenizer_hash` (every tokenizer file) is **unchanged**. It is still the key of the teacher cache (`index.json` keeps its `tokenizer_hash` field, keyed
by the **student's** tokenizer files) and of the same-student consistency checks, so old caches and old indexes stay valid.

### A different chat template is only a warning

The healing texts are rendered by the **student's** tokenizer: `distilljob.run_cache` and `run` build the rows with `AutoTokenizer.from_pretrained(student)` and
`data.build_rows` (the template is applied there, as in training). The rows are token ids; the teacher (cache job or online) is fed those ids and never renders
text with its own template. A different template therefore does not matter for the maths. The Check says so as a warning: "The two models use different chat
templates. That is fine for healing: the texts are written with the template of the model you heal, and the original then reads exactly the same token ids...". The
healed copy keeps the student's template. (The template of a model is read from `chat_template.jinja`, else `chat_template.json`, else `tokenizer_config.json`.)

The worker returns `ids_hash`, `template_hash` and `tokenizer_dir` with the facts of each model (`tokenizer_hash` stays, for the cache and old callers). When the
ids hashes differ and the sizes agree, the Rust side asks the worker for the probe once per pair (`heal_check` with `pair: {student, teacher}` folders; a verdict
that could not run is not kept, so the next Check tries again) and `verdict` shows it. The facts cache keys got a `v2|` prefix so facts kept by the old worker are
asked again. The Heal card is offered for a pair "accepted with warnings" (`ok` is false only when a line refuses).

## Files and code

* `models/<name>/heal_source.json` (only for a heal of a model the Slim page did not make, or a made model healed with another teacher): `student`, `teacher`
  (`{"kind": "made" | "folder" | "hub", ...}`), `created`, `student_name`, `teacher_name`, `student_dir` (where the Hugging Face cache put a Hub student),
  `check` (the sizes the Check found: parameters, layers, hidden size, vocabulary, family). Written at Start, nothing is written before.
* Heal output stays `models/<name>/healed/`; `heal.json`, `distill/cache/`, `distill/run/`, `heal_config.json`, `heal_metrics.jsonl` are as before. The student is
  read where it is (a folder) or in the Hugging Face cache (never copied); a Hub student is downloaded first (`heal_check` with `fetch: true`, which uses
  `reshapejob._resolve_source`) and the folder is remembered in `heal_source.json`. A Hub **teacher** is downloaded by the jobs themselves, as before.
* Core: `crates/ndforge-core/src/reshape/heal_source.rs` (`Source`, `HealSource`, `read_origin`, `student_folder`, `resolve_student`, `heal_check`, `verdict`,
  `heal_pair`, `heal_pair_estimate`, `heal_candidates`, `heal_prepare`, `fetch_student`, `heal_from_source`); `runs.rs` `heal`, `heal_options(_with)`,
  `heal_estimate(_with)`, `register_folder` use `read_origin` / `student_folder`. For a made model both answer from `model/` and `ndforge_reshape.json` exactly as
  before (the existing tests are unchanged and pass).
* Tauri (`src-tauri/src/lib.rs`): `heal_candidates`, `heal_check`, `heal_pair`, `heal_pair_estimate`, `heal_source_start`. The old `heal_options`, `heal_estimate`,
  `heal_start`, `heal_stop`, the job types (`distill_cache`, `distill`, `distill_merge`) and the event names are unchanged: the nd-world repository embeds this
  worker. The new job type is `heal_check`.
* Worker: `healcheck.py`; `distilljob.merge` writes a compatible `ndforge_reshape.json` when the student has none: `origin` = the teacher (given in the merge job
  as `origin`), `healed: {...}`, `layers`, `params`, `healed_from`, and a README that says "Made with ND Lore Forge (Heal a model)". The made-model merge is
  untouched. The distill jobs needed no change: they already take the student folder and the teacher repo from the job, and compute the calibration loss "before
  healing" themselves when the job has none (`slim_loss` null).
* UI: `healSection()`, `hsCheck`, `hsChanged`, `hsReloadPair` and `wireHealSection` in `ui/app.js`; `healCard(m, pair)` is the old card, now also used for a pair.

## Owner decisions (defaults of the brief, as built)

1. The section stands on its own and uses the existing heal UI (no second implementation).
2. The student can be a made model, a model-list entry, a folder or a Hub id; the Check shows what it is, whether it is on disk or would be downloaded (size, free
   disk), and refuses quantized and GGUF models.
3. The teacher defaults to the recorded original and must be chosen otherwise; a guess is offered and labelled as one; the two tokenizers must give the same token ids (not the same files).
4. Nothing is written until Start. The healed copy is a derivative: the section, the model list note and the help say so.

## Tests

* Worker, the pair decision: `worker/tests/test_tokpair.py` (the probe constant; the canonical ids hash ignoring padding / truncation / decoder / version / chat template
  and changing with a vocabulary, merge, added-token or normalizer change; settings-only folders accepted; size mismatch and a missing file refused; the probe with
  stand-in tokenizers: equal accepted with the yellow warning, the first difference shown, a load or encode error refused, transformers missing skipped; and,
  where `tokenizers` / `transformers` are installed, the tiny real tokenizers: a changed chat template accepted, cosmetic `tokenizer.json` changes accepted, a swapped
  vocabulary refused). The old test of a refused different tokenizer (`test_distill_torch.py`) stays as it was and passes.
* Worker without torch: `worker/tests/test_heal_check.py` (stand-in folders: facts, tokenizer hash and vocabulary, quantized, GGUF, unsupported family, multimodal
  config, the guess from `base_model` / `_name_or_path`, a bad name, the job through the real entry point).
* Worker with torch (CPU, tiny models): `worker/tests/test_heal_external_torch.py`: a thinned tiny model with `ndforge_reshape.json` removed, an explicit teacher,
  cache + distill + merge; the output folder is complete, loads, the metadata names the teacher; and the made-model merge still keeps its own metadata.
* Rust: `crates/ndforge-core/tests/reshape.rs`, section "healing any model": the verdict (identical tokenizer, same ids but other settings and a template warning, the probe verdicts, a pair accepted with a warning that starts (fake worker), different vocabulary / files / missing,
  quantized, GGUF, unsupported, teacher smaller, small card, download and disk), Hub names, `resolve_student` / `read_origin` (made, folder, Hub, bare folder),
  `heal_source.json` round trip and an unreadable file, candidates (made, custom folder / Hub, odd entries, earlier heals, empty), and (unix, fake worker) the
  whole start from a folder with a Hub teacher, the Check that is not asked twice, a pair that fails and leaves nothing, a Hub student downloaded first, another
  teacher for a made model.
* UI: `ui/smoke/reshape.test.mjs`, the last group: the section without a result card, the groups of both pickers, the made model preselected, the typed
  folder / id, the Check list, Start disabled / enabled, switching between a made model and another one, the guess, Browse, an error.

## Deviations

* **The Check's file reading is in the worker only.** The brief listed Rust tests "with temp-dir folders holding config/tokenizer files". The folders are read by
  the worker (`healcheck.py`), tested with stand-in folders there; the Rust verdict is tested from the facts the worker returns, and the Rust integration tests use a
  fake worker. A second reader in Rust would have duplicated the tokenizer hash (it must be the hash the jobs use).
* **A made model with its recorded original keeps today's Start** (in its result card, without a Check): the old card and its tests stay as they were. The Check is
  available for it (it then reads the recorded original too). Any other pair, and a made model with another teacher, cannot start without a passed Check.
* **The Heal controls are not shown twice.** The section hosts them for every pair except "made model with its recorded original", where it points to the card.
* The teacher of a heal is a folder or a Hub id, never a "made" model (a made model that is in the model list is a folder).
* The first version required the tokenizer files to be **identical** (chat templates in `tokenizer_config.json` counted), which refused models from the Hub that
  only differ in settings. Replaced by the ids hash and the probe above (owner decisions of the follow-up: 1. an ids hash over the id-deciding files, 2. a canonical
  `tokenizer.json`, 3. a probe of about 40 texts as a fallback, strict, 4. a template difference is a warning, 5. the cache key is unchanged and the jobs use the same
  pair decision, 6. the yellow messages reach the page through the existing check list).
* The brief said "compare vocabulary sizes too (config vocab_size / lm_head rows / tokenizer length)". The tokenizer length refuses; the config `vocab_size` /
  output rows differing stays a warning (padding), as the Check always did: real models pad the output layer.
* The brief's "added_tokens (content/id/special/normalized flags)" also includes `lstrip`, `rstrip` and `single_word`: they change how text is split around the token.
* A family the worker has no chat format for (anything but Qwen3.5) is formatted with the `generic` family (the tokenizer's own end token closes the answer).
* `src-tauri` was not compiled here (the brief says not to build it); the five new commands follow the shape of `reshape_try_plan` and `heal_start`.

## Not verified yet

* Healing a **real** downloaded pruned model (a real Qwen3 / Llama / Mistral student): the torch tests use a tiny Qwen3.5-architecture model; the `generic` chat
  format with a real non-Qwen3.5 tokenizer and `backends.load` of such a student have never run.
* The Hub side of `heal_check`: `get_safetensors_metadata`, the tokenizer files downloaded with `hf_hub_download`, the "already in the Hugging Face cache" test
  (`snapshot_download(local_files_only=True)`), the licence from the Hub API (`card_data`), gated or private models with an `HF_TOKEN`, and the download of a Hub
  student with progress. Only stand-in folders and a fake worker were run.
* **The tokenizer rule against real Hub models.** Only stand-in tokenizers and the tiny trained tokenizers of the tests were used: a real community-pruned model whose
  `tokenizer_config.json` differs from its original, tokenizers whose normalizers are written differently but behave the same (they go to the probe), and a
  `tokenizer.model` (SentencePiece) pair were not tried. Nothing says yet how often the probe, rather than the hash, decides on real models.
* **The probe set's coverage.** 40 texts cannot prove two tokenizers equal: two tokenizers that differ only on a rare token or a rare byte sequence would be accepted.
  The ids hash is the strict rule; the probe is the fallback for the case where the files differ.
* Settings in `tokenizer_config.json` that can change ids (`add_prefix_space`, `legacy`, `clean_up_tokenization_spaces` for decoding only) are not hashed: when the ids
  hash is equal the probe is not run, so such a setting alone is not noticed. For fast tokenizers (`tokenizer.json`) they do not change the ids.
* The `transformers` library of the app's training environment loading real tokenizers of the families other than the tiny ones in the Check (Hub tokenizers are
  loaded from the Hugging Face cache folder the Check downloaded the small files to).
* The UI in a real window (Browse, the look of the check list, the dark theme), and Windows paths in the folder field.
* The five Tauri commands (compiled by CI only).
* Memory and time estimates for students of families other than Qwen3.x (the sizes come from config.json; the formulas were measured on Qwen3.x only).
