# ND Lore Forge: "Slim or grow a model" (layer pruning + depth up-scaling) — implementation plan

Status: plan only. Nothing in `/home/user/Llm-training` was changed. Written 2026-11-13 against the checkout in `/home/user/Llm-training`,
Unsloth Studio sparse clone `.../scratchpad/src/unsloth` (commit 1bbf9c3, 2026-11-13), transformers 5.22.2 (wheel source read) and llama.cpp
`master` (converter + `src/models/qwen35.cpp` read on 2026-11-13).

Legend: **[V]** verified in source I read (file cited) · **[A]** assumption, must be checked (most need a GPU run) · **[GPU]** only a real GPU run can verify.

---

## 0. Verdict: can this be done inside Unsloth Studio instead?

**In Unsloth Studio: partly — only MoE expert pruning (REAP). No layer pruning, no depth up-scaling. For dense / hybrid Qwen3.5 27B and 9B: no.**

Evidence (all under `studio/backend/`):
- `core/export/reap.py` is the only pruning code. Docstring: *"Opt-in REAP expert pruning before an export (Core's `FastLanguageModel.reap_prune`)"*. It calls
  `FastLanguageModel.reap_prune(model, tokenizer, sparsity=, dataset=, num_calibration_samples=, save_directory=)` (line ~397). It removes **routed experts per MoE layer**;
  the report is `kept_experts` per layer (`write_reap_report`). Nothing touches the number of decoder layers.
- `models/export.py` `ReapOptions`: `sparsity` (0–1, default 0.25, "Fraction of the routed experts to remove in every MoE layer"), `num_calibration_samples` (8–4096, default 512),
  `dataset` (HF dataset id, default `mlabonne/open-perfectblend`). It is attached only to `ExportMergedModelRequest`, `ExportBaseModelRequest` and (line 363) the GGUF request
  (`core/export/worker.py` ~568: REAP for `merged`, `base`, `gguf` exports). So REAP happens **at export time**, not as a new base model you train further in Studio.
- Which models: `moe_info_from_config` returns info only for a routed MoE with `num_experts > top_k` (keys `num_experts / num_local_experts / n_routed_experts / moe_num_experts`).
  Refused: identity/zero experts, bidirectional/diffusion, model-parallel configs, pre-quantized checkpoints (except mxfp4 / bnb ids that Unsloth maps to 16-bit),
  "stateful" types `glm_moe_dsa, glm5_next(_text), hy_v4, zaya`, hash-routed `deepseek_v4`, grouped routers without `topk_group`, models Studio runs on transformers 4.x,
  checkpoints with an attached/trained MTP head (`mtp_head.safetensors`, `strategy == "mtp"` manifest), and 4-bit-loaded models (`apply_reap`). So: Qwen3.5 **MoE** variants
  (`qwen3_5_moe`) qualify **[A: which exact repos; Core's `reap_prune` itself is not in the sparse clone]**; Qwen3.5 2B/4B/9B/27B are dense → `moe_info_from_config` → `None` → no REAP option.
- Layer pruning / up-scaling: `grep -rniE "prune|upscal|depth.?up|passthrough|mergekit|num_hidden_layers|drop_layer|block.?influence|shortgpt"` over `studio/` finds only REAP,
  an "open-staging prune" file cleanup (`main.py:925`) and HTTP "passthrough" proxying. No `num_hidden_layers` / `layer_types` editing anywhere; `models/training.py` has no
  `layers_to_transform`-style option. (`models/export.py:50 offload_layers` is RAM offload during export, not pruning.)
- What Studio *can* do with our output: load a local HF folder for export/REAP probing (`routes/export.py` `get_moe_info` handles `is_local_path`); training a local folder
  **[A]**. So a slim/grown folder made by ND Lore Forge can be opened in Studio afterwards, and for a Qwen3.5 **MoE** base the two can be stacked (our layer cut, then Studio's REAP
  on export).

Consequence: we build layer pruning and depth up-scaling ourselves (worker + Rust + UI). Expert pruning is an optional later phase (§11 Phase E) that, when Unsloth is installed,
calls the same `FastLanguageModel.reap_prune` Studio uses.

---

## 1. Feasibility and risk summary

| Item | Fact / estimate |
|---|---|
| Qwen3.5 27B size | 64 decoder layers **[V llama.cpp `src/models/qwen35.cpp`: `case 64: LLM_TYPE_27B`]**, bf16 ≈ 54–56 GB on disk/in memory. Does **not** fit a 32 GB RTX 5090 at 16-bit. |
| 27B → ~21B | 64 → 48 layers = drop 4 blocks of 4 (each block = 3 GatedDeltaNet + 1 full attention). Per layer ≈ (27.8 − ~2.5 embed/lm_head − ~0.45 vision)/64 ≈ 0.39 B → −6.2 B ≈ 21.6 B **[A: exact numbers come from the inspect job, §4.3]**. |
| 9B → ~13B | 32 → 48 layers = add 4 blocks (16 layers, ≈ 0.22 B each) ≈ 13.2–13.5 B **[A same]**. |
| Scoring memory (prune) | One forward pass per calibration batch with hooks. 27B: 16-bit needs ~56 GB → on a 32 GB card use **4-bit NF4 (~17 GB)** for scoring (ranking is robust to 4-bit **[A]**), or `device_map="auto"` CPU offload (needs ≥ 64 GB RAM, slow). 9B and smaller: 16-bit. Grow needs **no** model load at all (pure tensor copying). |
| Calibration | Default 128 samples × 512 tokens ≈ 65k tokens from the project's saved dataset (`datasets/vN/train.jsonl`, half world-book text, half chat). Cost: 27B 4-bit on a 5090 ≈ 1–3 min **[GPU]**. Stability is checked by a split-half test (§4.2 `halves_agree`). |
| Disk | Prune 27B: source 56 GB (HF cache) + output ~43 GB. Grow 9B: 19 GB + ~27 GB. Job refuses when free space < output × 1.1. |
| Heal (prune) | A normal Step 3 run with the slim model as base. 21.6 B LoRA 16-bit ≈ 48 GB (`hub::estimate_vram` = 2.2·B + 1) → does not fit 32 GB → **QLoRA** (≈ 17 GB) on the 5090; Unsloth advises against 4-bit for Qwen3.5 (`catalog/models.toml` note), so quality after healing is uncertain **[GPU]**. |
| Train grown 13.4 B | LoRA 16-bit ≈ 30.5 GB (tight on 32 GB), QLoRA ≈ 11.4 GB. |
| Expected gain, grow | **Honest: little or none from the project data alone.** SOLAR/LLaMA-Pro gains came from billions of tokens of continued pre-training; a world's data is ~1–10 M tokens. With "identity" init the grown model is exactly the original until trained, so it cannot be worse at the start, but it costs +40 % memory/disk/time. The UI says this. |
| Expected loss, prune | Removing 25 % of depth without healing typically raises calibration loss clearly and hurts general chat; healing recovers part of it **[GPU, measured by our test page]**. Never present prune as free. |
| Hybrid pattern | Cut / duplicate only **whole periods** (4 layers for Qwen3.5). Then `layer_types` still equals the `full_attention_interval` rule, so old and new transformers/llama.cpp agree (§2). |
| Vision | Untouched: vision tensors (`model.visual.*`) are copied byte-for-byte; Qwen3.5 has no deepstack injection into text layers **[V: 0 matches for "deepstack" in transformers `modeling_qwen3_5.py`; llama.cpp writes `num_deepstack_layers 0` for Qwen3.5]**, so the official mmproj fits. |
| Gemma 4 | **Refused in v1**: per-layer input embeddings packed for all layers (`hidden_size_per_layer_input`, `vocab_size_per_layer_input`) and KV sharing of the last `num_kv_shared_layers` layers [V `tests/data/gemma4_config.json`, transformers `modeling_gemma4.py:1063`]. Needs tensor slicing; later. |

---

## 2. Verified facts the design depends on

| # | Fact | Source |
|---|---|---|
| F1 | Qwen3.5 text config: `num_hidden_layers`, `layer_types` (list of `"linear_attention"`/`"full_attention"`), `full_attention_interval` (4), `mtp_num_hidden_layers` (1), `mlp_only_layers` ([]); nested under `text_config` with `vision_config` beside it | `worker/tests/data/qwen3_5_config.json` (real 4B config); transformers `configuration_qwen3_5.py` |
| F2 | If `layer_types` is missing, transformers builds it as `"linear_attention" if (i+1) % interval else "full_attention"` | `configuration_qwen3_5.py` `__post_init__` |
| F3 | `len(layer_types) == num_hidden_layers` is enforced (validation) | transformers `configuration_utils.py` `layer_type_validation` / `validate_layer_type` |
| F4 | Each decoder layer is built from `config.layer_types[layer_idx]`; caches use `layer_idx`; forward iterates `self.layers[: config.num_hidden_layers]` and picks the mask by `config.layer_types[i]` → renumbering layers 0..N-1 with a matching `layer_types` list is all a reload needs | `modeling_qwen3_5.py` 848–856, 1215, 1278–1282 |
| F5 | Decoder layer = `h + mixer(norm(h))`, then `h + mlp(norm(h))`; mixer output goes through `linear_attn.out_proj` or `self_attn.o_proj`, MLP through `mlp.down_proj` → zeroing those three makes a layer an exact identity | `modeling_qwen3_5.py` 861–902 |
| F6 | Checkpoint key layout (VLM): `model.language_model.layers.N.*`, `model.language_model.embed_tokens`, `model.visual.*`, `lm_head.weight`, MTP head `mtp.*` (outside `model.`); `AutoModelForCausalLM` maps `qwen3_5` → `Qwen3_5ForCausalLM` (text only, ignores `^mtp.*` and `^model.visual.*`, strips `language_model` via `PrefixChange`) | `modeling_qwen3_5.py` 912, 1668, 1753–1764; `models/auto/modeling_auto.py:897`; `conversion_mapping.py:1156` |
| F7 | llama.cpp converter writes `qwen35.attention.recurrent_layers` from `layer_types` (errors if length ≠ `num_hidden_layers`) **and** `full_attention_interval`; runtime prefers the array, falls back to the interval rule | llama.cpp `conversion/qwen.py` (`Qwen3NextModel.set_gguf_parameters`), `src/models/qwen35.cpp` lines 17–23 |
| F8 | llama.cpp maps `mtp.layers.0.*` to layer index `num_hidden_layers` (nextn); if `mtp_num_hidden_layers == 0` and no MTP tensors are found, the converter asserts unless `--no-mtp` → **keep the MTP tensors and `mtp_num_hidden_layers` unchanged** | `conversion/qwen.py` `_QwenMtpMixin` |
| F9 | Unknown layer counts load in llama.cpp as `LLM_TYPE_UNKNOWN` (a label only) | `src/models/qwen35.cpp` 25–30 |
| F10 | `huggingface_hub.HfApi.get_safetensors_metadata(repo)` returns every tensor's dtype/shape without downloading weights | `huggingface_hub/hf_api.py:6669`, `utils/_safetensors.py` |
| F11 | The worker env has `transformers>=5, trl, peft, datasets, accelerate, bitsandbytes` (+ Unsloth optional) | `crates/ndforge-core/src/envmgr/setup.rs:32` |
| F12 | CI runs worker tests with CPU torch + transformers ≥ 5 + accelerate; this sandbox has only numpy + pytest (no torch, transformers, safetensors) | `.github/workflows/ci.yml:47-49`; `pip list` here |
| F13 | A local model folder is accepted as a base: `hub::lookup` → `lookup_local` → `describe_folder` (config, chat template, file sizes, README licence) → catalog entry saved in `custom_models.json`; params estimated from file size /2 with a warning | `crates/ndforge-core/src/train/hub.rs` |
| F14 | Export: for a local base, `vision.fetch_mmproj` finds no `-GGUF` repo (`gguf_repo_candidates` returns `[]` for a dir) and falls back to `convert_hf_to_gguf.py --mmproj` on the base folder | `worker/ndforge_worker/exportjob.py` `run()`, `vision.py` |
| F15 | Evaluate job: `model.repo` = original, `full_model` = whole trained model folder (loads original, answers, frees it, loads the other) — reusable for "original vs slim" | `worker/ndforge_worker/evaluatejob.py` |

Assumptions to check on the GPU: Unsloth `FastLanguageModel.from_pretrained` loads a 48-layer local Qwen3.5 folder **[A/GPU]**; Unsloth's bundled llama.cpp
(used by `save_pretrained_gguf`) handles it — safe either way because we keep the interval pattern (F7) **[A/GPU]**; Ollama's Qwen3.5 engine loads a 48-layer GGUF **[A/GPU]**;
4-bit scoring ranks blocks like 16-bit scoring **[A/GPU, check once on 9B: both modes, compare chosen span]**.

---

## 3. Architecture overview

```
UI page "🧬 Slim or grow a model" (ui/app.js)
   │ invoke reshape_state / reshape_inspect / reshape_start / reshape_test / reshape_remove ; events on "reshape-event"
src-tauri/src/lib.rs  (thin commands, spawn_task)
   │
crates/ndforge-core/src/reshape/   plan.rs (pure sizing/validation)  runs.rs (models/<name>/ registry, jobs, register as base, test)
   │   uses envmgr::jobs::run_job, train::hub::lookup_local, train::catalog::add_custom, evaluate::runs (refactored), export::runs (vision_repo)
   │
worker: python -m ndforge_worker job.json
   ├─ "reshape_inspect" → reshapejob.run_inspect  (no torch: safetensors headers / HF metadata)
   └─ "reshape"         → reshapejob.run          (download → [prune: calibrate+score with torch] → choose → write shards (no torch) → [check loss])
        pure helpers: reshape.py (config/plan math), safetensors_io.py (header read, chunked shard writer)
```

Project layout addition (doc comment in `project.rs` must list it):
```
models/<name>/              one reshaped model (made by "Slim or grow")
  state.json                status: running|done|failed|stopped, spec, created, error
  result.json               the worker's result (scores, chosen layers, params, losses)
  test.json                 original vs this model on the test questions (same format as runs/<run>/test.json)
  model/                    the HF folder (safetensors shards + index, config.json, tokenizer, processor files, ndforge_reshape.json)
  model.partial/            while writing; renamed to model/ at the end, removed on failure/cancel
```
`<name>` = `{YYYYmmdd-HHMMSS}-{source-slug}-{slim|grow}-{layers_after}L`, e.g. `20261113-141500-qwen3.5-27b-slim-48L`.

---

## 4. Worker (Python)

### 4.1 `worker/ndforge_worker/safetensors_io.py` (new, stdlib only, no numpy/torch)

```python
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}

@dataclass(frozen=True)
class TensorRef:
    file: str          # absolute path of the shard
    dtype: str
    shape: tuple
    start: int         # absolute byte offset in the file (8 + header_len + data_offsets[0])
    end: int
    @property
    def nbytes(self) -> int: ...
    @property
    def numel(self) -> int: ...

def read_header(path) -> tuple[dict, int]:
    """(header without "__metadata__", absolute offset of the data area). Raises ValueError("not a safetensors file: ...") on a bad length/JSON."""

def list_tensors(folder) -> dict[str, TensorRef]:
    """All tensors of a model folder: model.safetensors.index.json (weight_map) when present, else every *.safetensors file. Duplicate names -> ValueError."""

def tensors_from_hub_metadata(meta) -> dict[str, tuple[str, tuple]]:
    """{name: (dtype, shape)} from HfApi.get_safetensors_metadata(...).files_metadata (inspect without download)."""

@dataclass
class OutTensor:
    name: str
    dtype: str
    shape: tuple
    source: TensorRef | None    # None => write zeros
    # nbytes property

def write_shards(tensors: list[OutTensor], out_dir, shard_bytes: int = 5 * 2**30, chunk: int = 64 * 2**20,
                 on_progress=None, should_stop=None) -> dict:
    """Writes model-00001-of-0000N.safetensors (+ model.safetensors.index.json when N > 1, always with metadata.total_size)
    in the given order, copying bytes in `chunk` pieces (never holds a tensor in memory), zeros for source None.
    Header JSON is padded with spaces to a multiple of 8 bytes; "__metadata__": {"format": "pt"}.
    on_progress(done_bytes, total_bytes); should_stop() -> True raises Cancelled. Returns the index dict."""
```
Notes: a tensor larger than `shard_bytes` gets a shard of its own. Shard file names follow HF (`model-00001-of-00004.safetensors`); a single shard is `model.safetensors`.

### 4.2 `worker/ndforge_worker/reshape.py` (new, pure: stdlib only)

```python
SUPPORTED_TYPES = {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "llama", "mistral", "qwen2", "qwen3", "qwen3_moe"}
REFUSED = {"gemma4": "Gemma 4 stores per-layer inputs for all layers in one table and shares attention memory between its last layers; "
                     "cutting or copying layers is not supported for it yet."}
LAYER_PREFIXES = ("model.language_model.layers.", "model.layers.", "language_model.model.layers.")
LAYER_KEY = re.compile(r"^(model\.language_model\.layers\.|model\.layers\.|language_model\.model\.layers\.)(\d+)\.(.+)$")
ZERO_SUFFIXES = ("linear_attn.out_proj.weight", "self_attn.o_proj.weight", "self_attn.o_proj.bias", "mlp.down_proj.weight", "mlp.down_proj.bias")
INDEX_LIST_KEYS = ("mlp_only_layers",)          # lists of layer indices (remapped)
INT_INDEX_KEYS = ("max_window_layers",)         # an index boundary (clamped; refused if it would change meaning, see fix_config)

class ReshapeError(ValueError): """plain-language message for the user"""

def text_config(cfg: dict) -> dict                      # cfg["text_config"] if it is a dict, else cfg
def model_type(cfg: dict) -> str                       # top-level model_type (falls back to text_config's)
def check_supported(cfg: dict) -> None                 # raises ReshapeError (REFUSED text, unknown type, quantization_config present, auto_map, missing num_hidden_layers)
def layer_types_of(tc: dict) -> list[str]              # tc["layer_types"], else F2 rule when full_attention_interval given, else ["full_attention"] * L
def pattern_period(types: list[str]) -> int            # smallest p | L with types[i] == types[i % p]; 1 for all-equal; L if aperiodic
def split_key(name: str) -> tuple[str, int, str] | None   # (prefix, index, rest) for decoder-layer tensors; None for mtp.*, model.visual.*, embed, norm, lm_head
def group_params(tensors: dict[str, tuple[str, tuple]], n_layers: int) -> dict
    # {"layers": [numel per layer], "embed": int, "lm_head": int, "vision": int (names containing "visual." / "vision"), "mtp": int (startswith "mtp."),
    #  "other": int, "total": int, "bytes_per_param": float (from the dominant dtype)}; ReshapeError if any layer index >= n_layers or an unknown "*.layers.N." prefix appears
def params_after(groups: dict, src: list[int]) -> int  # total - sum(layers) + sum(layers[s] for s in src)
def block_count(n_layers: int, period: int) -> int     # n_layers // period (ReshapeError if not divisible)
def limits(n_blocks: int, mode: str, protect_first: int = 1, protect_last: int = 1) -> tuple[int, int]
    # (min, max) blocks to drop/add: prune max = min(n_blocks // 2, n_blocks - protect_first - protect_last); grow max = n_blocks // 2; min 1
def blocks_for_target(groups, period, target_params, mode, protect_first=1, protect_last=1) -> int   # nearest feasible count, clamped to limits()
def finalize_distance(sums: list[list[float]], counts: list[list[int]]) -> list[list[float]]   # D[a][b] = sums/counts (a<b), 0 on the diagonal, mirrored
def choose_drop(D, n: int, strategy: str = "contiguous", protect_first: int = 1, protect_last: int = 1) -> list[int]
    # D over block boundaries 0..B (B = number of blocks). "contiguous": a in [protect_first, B - protect_last - n] minimising D[a][a+n] (Gromov et al.);
    # "separate": the n blocks k in [protect_first, B - protect_last) with the smallest D[k][k+1] (ShortGPT Block Influence on blocks). Ties -> lowest index.
    # Returns sorted block indices; ReshapeError when n does not fit.
def halves_agree(D1, D2, n, strategy, protect_first, protect_last) -> bool   # same choice from both halves of the calibration set
def blocks_to_layers(blocks: list[int], period: int) -> list[int]
def prune_src(n_layers: int, drop: list[int]) -> list[int]      # kept old indices in order; ReshapeError if drop empty/out of range/would remove everything
def grow_src(n_layers: int, period: int, n_blocks: int, layout: str = "solar") -> tuple[list[int], list[bool]]
    # exact code below; second list: True for the SECOND occurrence of a duplicated source layer (zero-initialised in "identity" mode).
def check_pattern(types_before: list[str], src: list[int], period: int) -> None   # new types == period pattern; ReshapeError("... would break the attention pattern ...")
def fix_config(cfg: dict, src: list[int]) -> dict
    # deep copy; in text_config: num_hidden_layers = len(src); layer_types = [old[s] for s in src] (written even if it was absent? only if present before);
    # keep full_attention_interval (check_pattern guarantees consistency); INDEX_LIST_KEYS remapped to [j for j,s in enumerate(src) if s in old_set];
    # INT_INDEX_KEYS: if value >= old L -> set new L; elif value == 0 -> keep; else ReshapeError (sliding-window boundary inside the model).
    # Any OTHER list value in text_config whose length == old L -> remapped like layer_types and listed in a note (generic safety).
    # vision_config, mtp_num_hidden_layers, top-level keys untouched. Returns new dict.
def plan_tensors(tensors: dict[str, "TensorRef"], src: list[int], zero_mask: list[bool] | None) -> list["OutTensor"]
    # non-layer tensors first (in original order), then layers 0..N-1: for new j, every tensor of old src[j] renamed prefix+str(j)+rest;
    # source None when zero_mask[j] and rest in ZERO_SUFFIXES. ReshapeError if zero_mask[j] and the layer has no mixer-out AND no down_proj suffix.
def describe(groups, src_before_len, src, types) -> dict   # summary numbers for result / ndforge_reshape.json
```
`grow_src` — SOLAR depth up-scaling / mergekit passthrough (SOLAR 10.7B = layers 0–23 + 8–31 of a 32-layer model), in whole periods:
```python
def grow_src(L, period, n_blocks, layout="solar"):
    if layout != "solar": raise ReshapeError("Unknown layout")   # the only layout in v1 (a "repeat the middle span" layout gives the same list whenever spans are period-aligned)
    m = n_blocks * period                      # layers added
    if m <= 0 or m > L // 2: raise ReshapeError("Grow by at most half of the layers.")
    half = (L + m) // 2
    half -= half % period                      # whole periods
    src = list(range(0, half)) + list(range(half - m, L))   # the m layers half-m .. half-1 appear twice; len == L + m
    seen, zero = set(), []
    for s in src:
        zero.append(s in seen); seen.add(s)
    return src, zero
```
Expected (tests): L=32, P=4, n=4 → `list(range(24)) + list(range(8, 32))` (48 layers), `zero` True at positions 24..39 (the second copies of 8..23; with those zeroed the
model computes exactly layers 0..31 = the original). L=64, P=4, n=2 → `range(36) + range(28, 64)` (72). L=24, P=4, n=1 → half 14→12: `range(12) + range(8, 24)` (28).
`len(set(src)) == L` always (every original layer is kept).

### 4.3 `worker/ndforge_worker/reshapejob.py` (new)

```python
def run_inspect(cfg) -> int
    # cfg: {"job": "reshape_inspect", "source": {"repo": str}}
    # local dir: config.json + safetensors_io.list_tensors; hub id: hf_hub_download("config.json") + HfApi().get_safetensors_metadata(repo)
    # result: {"model_type", "supported": bool, "why": str|None, "layers": L, "layer_types": [...], "period": p, "blocks": L//p,
    #          "layer_params": [...], "embed_params", "lm_head_params", "vision_params", "mtp_params", "other_params", "total_params",
    #          "bytes_per_param", "has_vision": bool, "tied": bool, "source_bytes": int, "local": bool}
    # never raises for "unsupported": returns supported=false + why (so the UI can say why); errors only for unreadable/not found.

def run(cfg) -> int      # the "reshape" job, keys in §5.1
    protocol.reserve_stdout()
    1. src_dir = _resolve_source(cfg["source"]["repo"])            # local dir, or snapshot_download(allow_patterns=["*.safetensors","*.json","*.jinja","*.txt","*.model","*.tiktoken","README*","LICENSE*"]) with progress "download"
    2. config = json(src_dir/config.json); reshape.check_supported(config); tensors = safetensors_io.list_tensors(src_dir); groups = reshape.group_params(...)
    3. types = layer_types_of(tc); P = pattern_period(types); require P == tc.get("full_attention_interval", P) for qwen3_5*
    4. if mode == "prune":
         if cfg.get("drop_layers"): drop = sorted(cfg["drop_layers"])            # expert override, no torch needed
         else: D, D1, D2, loss_before = _score(src_dir, cfg, P)                    # torch
               blocks = choose_drop(D, cfg["blocks"], cfg["strategy"], pf, pl); drop = blocks_to_layers(blocks, P)
               emit("reshape_scores", distances=[D[k][k+1] for k], span_matrix=D, chosen=blocks, stable=halves_agree(...))
         src, zero = prune_src(L, drop), None
       else ("grow"): src, zero = grow_src(L, P, cfg["blocks"]); zero = zero if cfg["grow"]["init"] == "identity" else None
    5. check_pattern(types, src, P); new_cfg = fix_config(config, src)
    6. disk check: shutil.disk_usage(out.parent).free >= 1.1 * sum(out nbytes) else error "disk_full"
    7. out = Path(cfg["out"]) (".../model.partial"), clean; write_shards(plan_tensors(tensors, src, zero), out, on_progress=progress("write"), should_stop=STOP file exists)
    8. write config.json (new_cfg, indent 2), copy side files (§4.4), write ndforge_reshape.json (§5.2), README.md (licence front matter copied from source README if any)
    9. if mode == "prune" and cfg["check"] (default true when torch present): loss_after = _loss on the new folder (same load mode, same batches)  # "check"
       if mode == "grow" and init == "identity" and cfg["check"]: loss_after must equal loss_before within 1e-3 relative, else error "grow_check_failed" (a self-test of the surgery)
   10. rename model.partial -> model (done by Rust? NO: worker renames, Rust only cleans up on failure); result (§5.3)

def _score(src_dir, cfg, P) -> (D, D_half1, D_half2, loss_before)      # torch
    model = _load(src_dir, cfg["load"])   # "auto": 16-bit if est_bytes <= 0.85*free VRAM, else 4-bit NF4 (bitsandbytes) if est/3.6 fits, else device_map="auto" with max_memory {0: vram-3 GiB, "cpu": ram-8 GiB}
                                          # AutoModelForCausalLM (text only, F6); dtype bf16 (fp16 on pre-Ampere); model.eval(); config.use_cache=False
    layers = _find_layers(model, L)       # the nn.ModuleList of length L whose items' class name ends with "DecoderLayer"; ReshapeError otherwise
    batches = _calibration(tokenizer, cfg["calibration"], family)   # data.read_jsonl + data.encode_rows(tok, rows, family, max_len, seed); half text/half chat; samples*;
                                          # batch size 4 at 512 tokens (8 when 16-bit and small); attention_mask kept for token masking
    hooks: forward_pre_hook(with_kwargs=True) on layers[k*P] captures hidden_states input (float32 copy on the same device) for k=0..B-1;
           forward_hook on layers[L-1] captures its output as boundary B.
    per batch: H = [h_0..h_B] (each [b,t,d]); for a<b: cos = cosine_similarity(h_a, h_b, dim=-1); ang = arccos(clamp(cos,-1,1))/pi; masked sum over valid tokens
               -> sums[a][b] += ..., counts[a][b] += n_tokens; also into half-1 / half-2 accumulators by batch parity; loss from model(..., labels=input_ids masked by attention)
    free boundary tensors after each batch; torch.no_grad(); progress("score", i/n)
    return finalize_distance(...) x3, mean loss
def _loss(dir_or_model, batches) -> float       # mean token cross-entropy; used for before/after
def _find_layers(model, n) -> torch.nn.ModuleList
def _calibration(tokenizer, calib_cfg, family) -> list[dict]   # [{"input_ids": LongTensor[b,t], "attention_mask": ...}], deterministic (seed)
```
Memory: per batch with B+1 = 17 boundaries × 4×512×5120 × 4 bytes ≈ 0.7 GB for 27B; fine. If OOM in `_score` → retry once with batch size 1, then error `out_of_memory` (plain text from `__main__._explain`).

### 4.4 Side files copied from the source folder (verbatim, if present)
`tokenizer.json, tokenizer_config.json, special_tokens_map.json, vocab.json, merges.txt, added_tokens.json, tokenizer.model, chat_template.jinja, chat_template.json,
generation_config.json, preprocessor_config.json, processor_config.json, video_preprocessor_config.json, audio_preprocessor_config.json, LICENSE*`
(reuse `vision.PROCESSOR_FILES` + a `TOKENIZER_FILES` tuple in `reshapejob.py`). `model.safetensors.index.json` and shards are **written**, never copied.

### 4.5 Changes to existing worker files
- `__main__.py`: dispatch `"reshape_inspect"` (error code `inspect_failed`) and `"reshape"` (`reshape_failed`); catch `reshape.ReshapeError` → `protocol.error("reshape_refused", str(exc))` (no traceback text).
  Add to `_explain`: `"no space left"` → plain disk message.
- `evaluatejob.py`: new optional keys (backwards compatible):
  - `trained_base`: base folder the `adapter` was trained on when it is NOT `model.repo` (a run on a slim model, compared with the real original). If set: original side = `_load(model.repo)`; free; trained side = `PeftModel.from_pretrained(_load(trained_base), adapter)`.
  - `load_4bit`: `{"original": bool, "trained": bool}` → `_load(repo, four_bit)` uses `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=_dtype())`.
  - result gets `"loaded": {"original": "16-bit"|"4-bit", "trained": ...}`.
- `exportjob.py`: `cfg["base"]["vision_repo"]` (optional): used instead of `base.repo` for `vision.base_modalities` and `vision.fetch_mmproj`; the local `--mmproj` conversion fallback still runs on `base.repo` (vision tensors are identical, F14/§1).
- `backends.py`: `cfg["hyper"].get("extra_targets")` (list) appended to `QWEN_TARGETS` in `load_unsloth`; `cfg["hyper"].get("layers_to_transform")` (list of ints) passed to `get_peft_model` / `LoraConfig(layers_to_transform=..., layers_pattern="layers")` **[A: Unsloth's `get_peft_model` accepts `layers_to_transform` — verify; if not, the option is hf-backend only]**.
  Needed because a grown Qwen3.5 with identity init has zeroed `linear_attn.out_proj`, which `QWEN_TARGETS` does not train (it lists only `q,k,v,o,gate,up,down` — F: `families.py`); the hf path already uses `"all-linear"`.
- `tests/tinymodel.py`: `build_arch_dir(path, family, num_layers=4)` parameter (default unchanged), `layer_types` built with period 4 for qwen3_5 when `num_layers % 4 == 0`.

---

## 5. Data formats

### 5.1 Worker job `reshape` (written by Rust)
```json
{"job": "reshape",
 "source": {"repo": "unsloth/Qwen3.5-27B", "family": "qwen3_5"},
 "mode": "prune",                       // "prune" | "grow"
 "blocks": 4,                           // whole periods to drop / add
 "strategy": "contiguous",              // prune: "contiguous" | "separate"
 "protect": {"first": 1, "last": 1},    // blocks never dropped
 "drop_layers": null,                   // expert: explicit old layer indices (must be whole periods); skips scoring
 "grow": {"layout": "solar", "init": "identity"},   // init: "identity" | "copy"
 "calibration": {"train": "<project>/datasets/v3/train.jsonl", "samples": 128, "max_len": 512, "seed": 3407},
 "load": "auto",                        // "auto" | "16bit" | "4bit" | "offload"
 "check": true,                         // loss before/after (needs torch)
 "shard_gb": 5,
 "out": "<project>/models/<name>/model.partial",
 "final": "<project>/models/<name>/model"}
```
Events (protocol.py, unchanged format): `progress` with steps `download, inspect, calibrate, score, choose, write, check, done`; `log`; one extra event
`{"type": "reshape_scores", "boundary_distance": [D[k][k+1]...], "matrix": D, "chosen_blocks": [...], "period": 4, "stable": true}`; final `result`.

### 5.2 `model/ndforge_reshape.json` (inside the produced folder, self-describing)
```json
{"format": 1, "tool": "ND Lore Forge", "created": "2026-11-13T14:15:00Z",
 "origin": {"repo": "unsloth/Qwen3.5-27B", "family": "qwen3_5", "layers": 64, "params": 27800000000},
 "mode": "prune", "period": 4, "layers": 48, "params": 21600000000,
 "src_layers": [0, 1, ...],            // new layer j = old src_layers[j]
 "dropped": [36, 37, ..., 51], "duplicated": [], "zero_init": [],
 "strategy": "contiguous", "calibration": {"version": 3, "samples": 128, "max_len": 512, "tokens": 65536},
 "loss_before": 1.92, "loss_after": 2.35, "stable": true,
 "vision": "unchanged (copied from the original)", "mtp": "copied unchanged"}
```

### 5.3 Worker result (`result` event data, also `models/<name>/result.json`)
`{"model_dir", "mode", "layers_before", "layers_after", "src_layers", "dropped", "duplicated", "zero_init", "params_before", "params_after", "bytes_written",
"loss_before", "loss_after", "ppl_before", "ppl_after", "stable", "boundary_distance", "chosen_blocks", "load": "4bit", "seconds", "notes": [..]}`

### 5.4 `models/<name>/state.json` (Rust)
`{"name", "status": "running|done|failed|stopped", "created", "updated", "spec": {...ReshapeSpec}, "source_model_id", "model_id" (catalog id once registered), "error"}`

### 5.5 Catalog entry for the result (`custom_models.json`)
`ModelEntry` gains two optional fields (serde default, `skip_serializing_if = "Option::is_none"`):
```rust
/// the model this one was cut from / grown from (Hugging Face name or folder), for reshaped models
#[serde(default, skip_serializing_if = "Option::is_none")] pub derived_from: Option<String>,
/// one line for the model list, e.g. "slimmed: 64 → 48 layers"
#[serde(default, skip_serializing_if = "Option::is_none")] pub reshape: Option<String>,
```
id = `custom-local-<name>`, name = `"Qwen3.5 27B · slim 48 layers"`, repo = absolute path of `models/<name>/model`, family = source family,
params_b = exact (§6.3 hub change), good_for = "Made on the Slim or grow page from <origin>.", note = expected-quality text.

---

## 6. Rust core

### 6.1 `crates/ndforge-core/src/reshape/mod.rs` (new)
`//! Slim or grow a model ...` + `pub mod plan; pub mod runs;`. Add `pub mod reshape;` to `lib.rs`.

### 6.2 `reshape/plan.rs` (pure, no I/O)
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)] #[serde(rename_all = "lowercase")]
pub enum Mode { Prune, Grow }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inspect {            // = the worker's reshape_inspect result
    pub model_type: String, pub supported: bool, pub why: Option<String>, pub layers: u32, pub layer_types: Vec<String>, pub period: u32, pub blocks: u32,
    pub layer_params: Vec<u64>, pub embed_params: u64, pub lm_head_params: u64, pub vision_params: u64, pub mtp_params: u64, pub other_params: u64,
    pub total_params: u64, pub bytes_per_param: f64, pub has_vision: bool, pub local: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReshapeSpec {
    pub source_model_id: String, pub mode: Mode, pub blocks: u32,
    #[serde(default = "contiguous")] pub strategy: String,          // "contiguous" | "separate"
    #[serde(default = "one")] pub protect_first: u32, #[serde(default = "one")] pub protect_last: u32,
    #[serde(default = "identity")] pub init: String,                // "identity" | "copy"
    #[serde(default = "s128")] pub calib_samples: u32, #[serde(default = "s512")] pub calib_len: u32,
    #[serde(default = "auto")] pub load: String, #[serde(default)] pub drop_layers: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SizeOption { pub blocks: u32, pub layers: u32, pub params_b: f64, pub disk_gb: f64, pub vram_lora_gb: f64, pub vram_qlora_gb: f64,
                        pub fits_lora: Option<bool>, pub fits_qlora: Option<bool> }

pub fn limits(ins: &Inspect, mode: Mode, protect_first: u32, protect_last: u32) -> (u32, u32)     // same rule as reshape.limits (Python); tests keep both equal
pub fn params_after(ins: &Inspect, mode: Mode, blocks: u32) -> u64    // prune: subtract the mean block params (exact choice is unknown before scoring); grow: add the
                                                                      // params of the duplicated layers per grow_src ("solar" middle span) -> exact
pub fn options(ins: &Inspect, mode: Mode, vram_gb: Option<f64>) -> Vec<SizeOption>   // every feasible block count; vram via train::hub::estimate_vram
pub fn blocks_for_target(ins: &Inspect, mode: Mode, target_b: f64) -> u32
pub fn scoring_load(params_total: u64, bytes_per_param: f64, vram_gb: Option<f64>) -> &'static str   // "16bit" | "4bit" | "offload" (same thresholds as worker "auto")
pub fn validate(spec: &ReshapeSpec, ins: &Inspect) -> Result<()>     // supported, blocks in limits, drop_layers whole periods & not protected, init/strategy/load known
pub fn warnings(spec: &ReshapeSpec, ins: &Inspect, vram_gb: Option<f64>) -> Vec<String>   // >25 % removed, grow "expect little gain", heal needs QLoRA, disk
pub fn disk_needed_gb(ins: &Inspect, mode: Mode, blocks: u32) -> f64  // output bytes * 1.1 (+ source bytes when not local and not cached: unknown -> add)
```

### 6.3 `reshape/runs.rs`
```rust
pub fn models_dir(p: &Project) -> PathBuf                         // p.folder.join("models")
pub fn name_for(src: &ModelEntry, mode: Mode, layers_after: u32) -> String
pub fn inspect(m: &ModelEntry, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Inspect>   // runs reshape_inspect; caches in config_dir()/reshape_inspect.json by repo (+ mtime for local)
pub fn build_job(p: &Project, m: &ModelEntry, spec: &ReshapeSpec, ins: &Inspect, name: &str) -> Result<Value>
    // needs the newest saved dataset version (train::runs::latest_version) unless mode == Grow or drop_layers set; error "Save a training-data version in Step 2 first..."
pub fn execute(p: &Project, spec: &ReshapeSpec, stop: Arc<AtomicBool>, on_event: &mut dyn FnMut(Value)) -> Result<Value>
    // lookup model (catalog::find), env check (python_path), inspect, validate, write state running (+ train::runs::ActiveRun-like guard), run via train::runs::drive
    // (STOP file -> worker should_stop; grace 60 s then kill); on Ok: result.json, state done, register(); on Err: remove model.partial, state failed/stopped
pub fn register(p: &Project, name: &str) -> Result<String>        // hub::lookup_local(models/<name>/model) -> entry with derived_from/reshape/name/good_for/note -> catalog::add_custom; returns id; stores id in state.json
pub fn list(p: &Project) -> Vec<Value>                            // state.json + result summary + tested flag + model_id, newest first
pub fn remove(p: &Project, name: &str) -> Result<()>              // refuses if a run in runs/ uses this model's repo (config.json model.repo); removes catalog entry + folder
pub fn mark_interrupted(p: &Project) -> usize                     // like train::runs::mark_interrupted for models/
pub fn test(p: &Project, name: &str, judge: Option<&Teacher>, vram_gb: Option<f64>, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value>
    // evaluate::runs::execute_pair with original = origin repo, trained = Full(models/<name>/model); writes models/<name>/test.json
pub fn read_test(p: &Project, name: &str) -> Option<Value>
```
Name validation everywhere: reject `/ \ ..` (same as `read_test` in evaluate).

### 6.4 Changes to existing Rust files
- `train/catalog.rs`: two new optional fields (§5.5). `train/hub.rs` `evaluate()` literal: `derived_from: None, reshape: None`.
- `train/hub.rs` `describe_folder`: if `ndforge_reshape.json` exists, set `info["safetensors"]["total"] = params` (exact size, no "estimated" warning) and keep `cardData.license`
  from the copied README. `lookup_local` warning list: add "This model was made by Slim or grow from <origin>." when present.
- `evaluate/runs.rs`: refactor `answer_with_worker` into
  ```rust
  pub enum Side { Adapter { dir: PathBuf, base: Option<String> }, Full(PathBuf) }
  pub struct Pair { pub original: String, pub family: String, pub trained: Side, pub four_bit: (bool, bool) }
  fn answer_pair(pair: &Pair, qs: &[Question], which: &[&str], max_new: u32, kill, on_event) -> Result<Value>
  pub fn execute_pair(p: &Project, pair: &Pair, version: u32, label: &str, out_file: &Path, per_category: usize, judge_with: Option<&Teacher>, kill, on_event) -> Result<Value>
  ```
  `execute` / `ask` for runs keep their signatures and build a `Pair` from the run (`trained_base` = `None`). New `pub fn pair_for_run(p, run, against_origin: bool, vram_gb)`:
  when the run's base model is a reshaped one and `against_origin`, `original` = `derived_from`, `trained = Adapter { base: Some(slim dir) }` (or `Full`).
  `four_bit` per side = `params_b * 2 + 3 > vram_gb * 0.9` (params from catalog / reshape.json). Result JSON gains `"against": "base"|"origin"` and `"loaded"`.
  `compare()` also reads `models/*/test.json` (row `name` = `"model:<name>"`, method `"slim"`/`"grow"`, steps null).
- `export/runs.rs` `execute_many`: `job["base"]["vision_repo"] = derived_from` when the run's base repo has `ndforge_reshape.json` (read `origin.repo`).
  MODEL_CARD: add a line "Base model: <slim name>, made from <origin> by removing/adding layers (ND Lore Forge)". `CardInfo` gets `derived_from: Option<&str>`.
- `train/runs.rs` `build_job`: if the base model folder has `ndforge_reshape.json` with `zero_init` non-empty, add `hyper.extra_targets = ["out_proj"]` to the job JSON (not to `Hyper`
  struct defaults) and log it. Expert option "Train only the new layers" → `hyper.layers_to_transform = <new layer indices>` (from `zero_init`/`duplicated` positions).
- `train/plan.rs`: no change (memory from `params_b`).
- `project.rs`: doc comment lists `models/`; `Project::create` creates `models/` (open() of an older project tolerates its absence — `models_dir` is created lazily).
- `model.rs` (`AppModel`): `reshape_state() -> Value` (`{"models": catalog models with supported flag unknown until inspected, "made": reshape::runs::list, "dataset_version",
  "env_ready", "vram_gb", "free_disk_gb", "unsloth": bench::env_has_unsloth()}`), `reshape_options(model_id, mode) -> Value` (inspect + plan::options + warnings),
  call `reshape::runs::mark_interrupted` in `adopt()`.
- `help.rs`: add `("reshape", include_str!("../help/reshape.md"))`, array length 10.
- `diagnose.rs`: list `models/` folders and their status in the problem report (names + status only).

### 6.5 Tauri commands (`src-tauri/src/lib.rs`)
```rust
#[tauri::command] fn reshape_state(s: State<App>) -> Out<Value>
#[tauri::command] async fn reshape_options(s: State<App>, model: String, mode: String) -> Out<Value>     // spawn_blocking: inspect may hit the network
#[tauri::command] fn reshape_start(app, s, tasks, spec: Value) -> Out<()>      // spawn_task(app, &tasks, "reshape-event", ...) ; KeepAwake like train_start
#[tauri::command] fn reshape_stop(tasks: State<Tasks>)                         // sets the cancel flag used as `stop` (STOP file, then kill after grace)
#[tauri::command] fn reshape_test(app, s, tasks, name: String, judge: bool) -> Out<()>   // "reshape-event"
#[tauri::command] fn reshape_remove(s: State<App>, name: String) -> Out<Value>
```
+ `test_start` gets `against: Option<String>` ("base"|"origin"). Register all in `generate_handler!`. One job at a time via the existing `Tasks.busy`.

### 6.6 UI (`ui/app.js`, `ui/style.css`)
- `renderSteps()`: after the 5 steps add `{key: "reshape", label: "🧬 Slim or grow a model", unlocked: !!state.bundle}`. `TITLES.reshape`. `render()` / `go()` dispatch `renderReshape()` / `rsRefresh()`.
- State object `rs = {state, model, mode: "prune", options: null, blocks: null, spec: {...defaults}, running, text, progress, log: [], scores: null, result: null, error, ok}`.
- Page, top to bottom:
  1. Intro card (plain words): what slimming/growing does, "always test and usually train afterwards", honest expectations (grow: little gain from one world's data; prune: answers get worse until healed).
  2. **Start from**: `<select>` of catalog models (built-in + custom). On change → `reshape_options`. Shows: layers, pattern ("3 fast linear-attention layers + 1 full-attention layer, repeated 16×"), parameters, vision yes/no, or the refusal `why`.
  3. **Slim / Grow** radio.
  4. **Size slider** over `options[]` (one stop per block): label "48 layers · 21.6 B parameters · 43 GB on disk · training: LoRA ≈ 48 GB ✗, QLoRA ≈ 17 GB ✓ on this card (32 GB)". Prune > 25 % → yellow warning.
  5. Grow only: "Start as an exact copy of the original (recommended)" (`identity`) vs "Copy the layers (SOLAR style)" (`copy`).
  6. Prune only: "Measure on: data version vN (world book + answers), 128 samples" (`64/128/256`). Expert mode: strategy (one block of layers / separate blocks), protect first/last, load mode, explicit layer list.
  7. **Make the model** button (disabled without env / dataset version for prune / disk), progress bar + step text, log `<details>`, **Stop**.
  8. Result card: bar chart (inline SVG, same style as `lossChart`) of `boundary_distance` per block with the chosen blocks highlighted; "Removed layers 36–51 (4 blocks)", params before/after,
     "Text-prediction error on your data: 1.92 → 2.35 before healing" (or "unchanged: exact copy"), stability note when `stable == false` ("the measurement was not stable; try 256 samples").
     Buttons: **Test against the original** (`reshape_test`; result table re-uses the Step 4 `CATS` rows: Original vs Slim), **Train it on your world (Step 3)** (sets `tr.model = model_id`, `go("train")`),
     **Delete**.
  9. List of made models (name, status, size, tested headline).
- Step 4 page: for a run on a reshaped base show a select "Compare with: the slim model it was trained from | the original <origin>" → `against`.
- Step 3 model list shows `reshape` text under the name (from catalog entry).

### 6.7 Help text `crates/ndforge-core/help/reshape.md` (new)
Sections: What it does · Slim (how layers are chosen: the measurement on your own text; why whole blocks of 4 for Qwen3.5; vision stays) · Grow (exact-copy start; honest gain; memory) ·
After making it: test, then train ("heal") in Step 3; QLoRA note for big models · Files and disk space · Images/vision and GGUF (official vision file reused; layer count is unusual for
llama.cpp/Ollama: check with Test it) · Problems (out of memory → 4-bit scoring / fewer samples; disk; unsupported model).

---

## 7. How the result is used by the rest of the app

| Feature | How it works with a reshaped model |
|---|---|
| Base model (Step 3) | Registered via `hub::lookup_local` + `catalog::add_custom` (§6.3) → appears in the model list like a folder model; local-only (existing warning: cannot train on a server). |
| Training | Unchanged pipeline (`trainjob.py` / `backends.py` load `model.repo` = the folder). Grown identity models get `extra_targets ["out_proj"]`. |
| Benchmark | Unchanged (`train/bench.rs` uses the `ModelEntry`). |
| Test page | (a) `reshape_test`: original (origin repo) vs reshaped (`full_model`), `models/<name>/test.json`; (b) runs on the reshaped base: against the slim base (default) or the origin (`trained_base`). Big sides load in 4-bit (shown in the result). |
| Export / GGUF | `exportjob` with `base.repo` = slim folder: Unsloth `save_pretrained_gguf` or llama.cpp converter. Text GGUF: converter writes the per-layer recurrent flags from `layer_types` (F7) and we keep the interval pattern so older converters/runtimes agree. MTP tensors kept (F8). **Vision mmproj:** `vision_repo` = origin → the official `<origin>-GGUF` mmproj is copied (vision tower and projector byte-identical, output dim = unchanged text `hidden_size`); fallback converts `--mmproj` from the slim folder (same tensors). |
| Ollama | Single GGUF install as today **[GPU: Ollama's Qwen3.5 engine with 48 layers]**. |
| Unsloth Studio | The slim folder (and exports) can be opened in Studio; for a Qwen3.5 MoE base Studio's REAP can be applied on export afterwards. |

---

## 8. Error handling (plain-language messages; code → text)

| Where | Condition | Message / behaviour |
|---|---|---|
| inspect | unsupported type | `supported:false`, `why` shown on the page (Gemma 4 text from `REFUSED`; others: "Only Qwen3.5 and plain Llama/Qwen/Mistral-style models can be slimmed or grown so far.") |
| inspect | quantized / `auto_map` | same messages as `hub.rs` (`quantization_config`, trust_remote_code) |
| plan | blocks outside limits | "Slim at most half of the layers, and never the first or last block." / "Grow by at most half." |
| plan | prune without dataset version | "Save a training-data version in Step 2 first: the layers are measured on your own text." |
| worker | `drop_layers` not whole periods / protected | `reshape_refused`: "Layers 3 to 6 do not form whole blocks of 4 (3 fast + 1 full attention). Remove whole blocks so the model keeps its pattern." |
| worker | disk | `disk_full`: "There is not enough free space: <need> GB needed in <folder>, <free> GB free." (checked before writing) |
| worker | OOM in scoring | retry batch 1, then `_explain` OOM text + "choose 4-bit measuring (Expert mode) or fewer samples" |
| worker | no torch (prune scoring) | `reshape_refused`: "Measuring the layers needs the training software (Training setup)." Grow and explicit-layer prune work without torch. |
| worker | grow identity self-check fails | `grow_check_failed`: "The new model does not give the same answers as the original although it should; nothing was saved. Please make a problem report." (partial removed) |
| worker | download fails | existing `_explain` (401/gated/not found) |
| Rust | cancel/stop | STOP file → worker stops between tensors (`should_stop`) → `model.partial` removed, state `stopped`; kill after 60 s grace |
| Rust | program closed mid-job | `mark_interrupted` → state `failed` with "The program was closed while this model was being made. Start it again." + remove `model.partial` |
| Rust | remove while used | "Run <run> was trained on this model. Delete that run first, or keep the model." |
| export | mmproj fails | existing behaviour: note only, text GGUF still made |

---

## 9. Test plan

### 9.1 Worker, no torch (run here and in CI) — `worker/tests/test_reshape.py` (new)
Fixture `fake_model(tmp, layers=8, family="qwen3_5")`: writes `config.json` (from `tests/data/qwen3_5_config.json`, `text_config` shrunk: hidden 8, `num_hidden_layers` = layers,
`layer_types` period 4, plus a small `vision_config`), `tokenizer_config.json` with a chat template, `chat_template.jinja`, `preprocessor_config.json`, and source shards written
by a tiny independent writer in the test file (`struct.pack("<Q", len(header))` + JSON + bytes; NOT `safetensors_io`, so the module is checked against an independent encoder),
filled with deterministic bytes (`bytes((i * 7 + k) % 256 for k in range(n))`, i = tensor number) so copies are checkable; tensors: `model.language_model.embed_tokens.weight`, per layer `input_layernorm.weight`,
`linear_attn.out_proj.weight` / `self_attn.o_proj.weight` by type, `mlp.down_proj.weight`, `mlp.up_proj.weight`; `model.visual.blocks.0.attn.qkv.weight`, `mtp.fc.weight`,
`mtp.layers.0.mlp.down_proj.weight`, `lm_head.weight`. A `llama` variant without `layer_types`.
Tests:
1. `pattern_period`: real 4B config → 4; all-full → 1; `["a","b","a"]` → 3 (aperiodic → L).
2. `layer_types_of` without `layer_types` but `full_attention_interval=4` equals the real list.
3. `split_key`: VLM layer key → (prefix, 12, rest); `model.layers.3.x` ok; `mtp.layers.0.x`, `model.visual.blocks.3.x`, `lm_head.weight` → None; unknown `foo.layers.1.x` → `group_params` raises.
4. `group_params` / `params_after` numbers on the fake model.
5. `choose_drop` contiguous / separate on a hand-made D (known answers), protection respected, ties lowest index, too-large n raises.
6. `halves_agree` true/false cases.
7. `prune_src` / `grow_src`: the three expected lists in §4.2; m > L/2 raises; `len(set(src)) == L`; `prune_src` with an empty / out-of-range drop raises.
8. `check_pattern` raises for `drop=[0,1]` on Qwen3.5 types; passes for `[4,5,6,7]`.
9. `fix_config`: counts, remapped `layer_types`, `full_attention_interval` kept, `mlp_only_layers` remapped, `vision_config` deep-equal, `mtp_num_hidden_layers` unchanged, extra list
   of length L remapped + note; `max_window_layers` rules; Gemma 4 config → `ReshapeError` with the REFUSED text; `quantization_config` refused.
10. `safetensors_io`: write→read round trip (names, dtypes, shapes, byte-exact data, zeros), header length % 8 == 0, `__metadata__.format == "pt"`, sharding with tiny
    `shard_bytes` → N files + index with correct `weight_map` and `total_size`, `should_stop` → raises and leaves no complete index. If `import safetensors` works (CI has it via
    transformers): also `safetensors.safe_open(..., framework="numpy")` reads every tensor identically (skip otherwise).
11. `plan_tensors`: renamed layer keys, vision/mtp/embed untouched and first, zero sources only for `ZERO_SUFFIXES` of flagged layers.
12. Job end-to-end via subprocess (pattern of `test_worker.py`): `reshape_inspect` on the fake folder; `reshape` grow identity with `check: false` → `model/` exists,
    `model.partial` gone, `config.json` `num_hidden_layers == 12`, side files copied, `ndforge_reshape.json` correct, result JSON keys; prune with explicit `drop_layers=[4,5,6,7]`
    and `check:false`; prune needing scores without torch → `reshape_refused` code (test only when torch is missing); disk check: call `reshapejob.run(cfg)` in-process with
    `monkeypatch.setattr(shutil, "disk_usage", lambda p: (10, 9, 1))` and `capsys`-free parsing (patch `protocol._out` to an `io.StringIO`) → `disk_full` error, no `model/` folder.
13. `exportjob` `vision_repo`: call `vision.base_modalities` / `fetch_mmproj` stand-ins (existing tests pattern with `list_files`/`hub_download` fakes) — unit-test a small helper
    `exportjob._vision_source(cfg) -> str` instead of the whole export.

### 9.2 Worker, torch on CPU (CI; skipped here) — add to `worker/tests/test_reshape_torch.py` with `pytest.importorskip("torch")`, `("transformers")`
Using `tinymodel.build_arch_dir(path, "qwen3_5", num_layers=12)` and `build_llama_dir`:
1. **Prune equals skipping**: logits of the pruned folder (drop block 1) == logits of the original with `layers` ModuleList reduced to the kept layers and `config.layer_types`/`num_hidden_layers`
   patched (atol 1e-5) → proves key renaming + config.
2. **Identity grow is exact**: grown folder (n=1 block, init identity) logits == original logits (atol 1e-5); `check: true` path passes.
3. Copy grow loads and generates; `layer_types` length 16, pattern period 4.
4. Scoring path: `reshape` prune with calibration from `tinymodel.sample_rows` → result has `boundary_distance` of length 3, `chosen_blocks == [1]` (only middle allowed), loss numbers finite,
   folder loads with `AutoModelForCausalLM`.
5. A tiny LoRA run (existing `trainjob` test helper) on the pruned folder, then `evaluate` with `trained_base` = pruned folder, `model.repo` = original → both sides answered.
6. Grown identity model + `extra_targets ["out_proj"]` on the hf backend: adapter contains `out_proj` LoRA weights.
7. Export via llama.cpp (skip unless `find_llama_cpp()` finds it, like `test_gguf_is_made_with_llama_cpp...`): pruned Qwen3.5 tiny → GGUF made (converter accepts 8 layers).

### 9.3 Rust (`cargo test`) — `crates/ndforge-core/tests/reshape.rs` (new) + small additions
1. `plan::options` on a fake 64-layer Inspect (layer_params 0.39e9 each, others 2.95e9): 4 blocks → 48 layers, ≈ 21.7 B; `blocks_for_target(21.0)` → 4; limits; `scoring_load` (27.8 B on 32 GB → "4bit", 9.6 B → "16bit", no VRAM → "4bit").
2. `validate`: unsupported, too many blocks, protected drop, non-period drop_layers.
3. Python/Rust parity of `limits`: a JSON table of cases in `worker/tests/data/reshape_limits.json` read by both test suites.
4. `runs::execute` with the fake worker shell script (`tests/export.rs` pattern): first call answers `reshape_inspect`, second writes `model.partial/` → renames → result. Assert
   state done, `result.json`, catalog entry in `custom_models.json` with `derived_from`, `reshape`, exact `params_b` (from `ndforge_reshape.json`), `list()` shows it.
   Failure path: worker prints error → state failed, `model.partial` removed. Stop path: kill flag → state stopped.
5. `hub::lookup_local` on a folder with `ndforge_reshape.json` → no "estimated from file sizes" warning, params exact.
6. `evaluate`: `reshape::runs::test` with a fake worker that writes answers → `models/<name>/test.json`, `compare()` contains `model:<name>`; job JSON has `model.repo = origin`, `full_model`.
   Run on a reshaped base with `against = origin` → job has `trained_base`.
7. `export`: fake worker captures the job file; for a run whose base folder has `ndforge_reshape.json`, `base.vision_repo == origin`.
8. `train::runs::build_job` adds `extra_targets` for a zero-init base.
9. `help_markdown("reshape")` is not the fallback text; `Project::create` makes `models/`.
10. `remove` refuses while a run uses the model.

### 9.4 GPU-only checklist (RTX 5090; document results in `docs/RESHAPE.md`)
1. 9B: scoring in 16-bit and 4-bit → same chosen span? (validates 4-bit scoring). Time and peak VRAM.
2. 27B: prune 64→48 with 4-bit scoring: time, peak VRAM/RAM, disk; output loads in transformers and **Unsloth** `FastLanguageModel`.
3. Test page original vs slim (both 4-bit): lore/rules/general numbers. Then QLoRA heal (Step 3) and test against origin.
4. 9B grow 32→48 identity: `check` passes on GPU (loss equal), LoRA 16-bit fits?, test vs original after training.
5. Export: Unsloth GGUF and llama.cpp GGUF of the 48-layer model; `llama-server -m ... --mmproj <official>` answers an image question; Ollama install + "Test it".
6. Speed in tokens/s vs the original (bench page).

---

## 10. Docs to update
- `README.md`: new section "Slim or grow a model" (what, limits, honest expectations, verification state) and a Status table row ("built; needs a first run on the RTX 5090").
- `docs/RESHAPE.md` (new): method (angular distance on block boundaries, Gromov et al. 2024 "The Unreasonable Ineffectiveness of the Deeper Layers"; ShortGPT Block Influence;
  SOLAR depth up-scaling / mergekit passthrough; LLaMA Pro identity init), the verified-facts table (§2), file formats (§5), GPU checklist results, Unsloth Studio verdict (§0).
- `crates/ndforge-core/help/reshape.md` (new, §6.7); `help/test.md` (against slim/origin), `help/use.md` (vision file for reshaped models), `help/troubleshooting.md` (3 entries: OOM while measuring, disk, Ollama refuses an unusual layer count).
- `project.rs` doc comment (models/), `docs/TRAINING_SPEED.md` one paragraph: pruned models train/run faster, grown slower.
- `PLAN.md`: add milestone M13 "Slim or grow".

---

## 11. Ordered task list (each step is one commit; run `cargo test --workspace` and `cd worker && python -m pytest -q tests` after each)

**Phase A — pure worker helpers (no torch; fully testable here)**

1. **safetensors reader/writer.** Create `worker/ndforge_worker/safetensors_io.py` exactly per §4.1. Create `worker/tests/test_safetensors_io.py` with tests 9.1-10 (+ an independent
   minimal writer helper in the test file for the source shards). Accept: tests pass without numpy/torch; header 8-aligned; chunked copy (assert peak memory not needed—just use chunk=7 bytes in a test to exercise chunk boundaries).
2. **Config/layer math.** Create `worker/ndforge_worker/reshape.py` with every function of §4.2 except `plan_tensors` (use the final `grow_src` code block, `layout` only `"solar"`).
   Create `worker/tests/test_reshape.py` tests 9.1-1…9 and `worker/tests/data/reshape_limits.json` (≥ 8 cases: `{"blocks": 16, "mode": "prune", "pf": 1, "pl": 1, "min": 1, "max": 8}` …).
   Accept: all pass; Gemma 4 refused with the exact REFUSED text.
3. **Tensor plan.** Add `plan_tensors` + `describe` to `reshape.py` (imports `OutTensor`, `TensorRef` from `safetensors_io`). Tests 9.1-11. Accept: pass.
4. **Inspect job.** Create `worker/ndforge_worker/reshapejob.py` with `run_inspect` (local folders via `list_tensors`; Hub via `get_safetensors_metadata` + `hf_hub_download("config.json")`,
   imported lazily). Add `"reshape_inspect"` dispatch in `__main__.py`. Test: subprocess on the fake folder (9.1-12 first part). Accept: result keys exactly as §4.3.
5. **Reshape job without torch.** In `reshapejob.py` implement `run` steps 1–8 and 10 for grow and explicit-`drop_layers` prune (scoring path raises `ReshapeError("Measuring the layers needs…")`
   when `import torch` fails); `_resolve_source` (local dir or `snapshot_download` with the allow-list); side-file copy (§4.4); `ndforge_reshape.json` (§5.2); README licence;
   STOP-file `should_stop`; disk check; rename `.partial` → final. Dispatch `"reshape"` in `__main__.py` with `ReshapeError` → `reshape_refused`. Tests 9.1-12 rest.
   Accept: produced folder passes `list_tensors`, `config.json` has 12 layers for grow n=1 on 8 layers.

**Phase B — torch parts (CPU-testable in CI)**

6. **Tiny model layers.** `tests/tinymodel.py`: `build_arch_dir(..., num_layers=4)` (qwen3_5 types period 4 when divisible). Accept: existing tests unchanged (default 4).
7. **Scoring.** In `reshapejob.py` add `_load`, `_find_layers`, `_calibration`, `_score`, `_loss` (§4.3) and steps 4/9 of `run` (scores event, loss before/after, identity self-check).
   Create `worker/tests/test_reshape_torch.py` tests 9.2-1…4. Accept: CI green (skips here).
8. **evaluate/export/backends hooks.** `evaluatejob.py`: `trained_base`, `load_4bit`, `loaded` (§4.5). `exportjob.py`: `_vision_source(cfg)` helper + use of `base.vision_repo`.
   `backends.py`: `extra_targets`, `layers_to_transform` (hf path for sure; Unsloth path guarded by `try/except TypeError` → log "not supported by Unsloth here; training all layers").
   Tests 9.1-13, 9.2-5…7. Accept: old evaluate/export tests unchanged.

**Phase C — Rust core**

9. **Catalog fields.** `train/catalog.rs` add `derived_from`, `reshape` (§5.5); update `hub.rs` literal; `describe_folder` reads `ndforge_reshape.json` (§6.4). Tests 9.3-5. Accept: existing `tests/hub.rs` pass; old `custom_models.json` still loads.
10. **Plan module.** Create `reshape/mod.rs`, `reshape/plan.rs` (§6.2), `pub mod reshape;` in `lib.rs`. Tests 9.3-1…3 in `tests/reshape.rs`. Accept: pass; limits parity file read.
11. **Runs module.** Create `reshape/runs.rs` (§6.3) using `train::runs::drive` (make `write_json`/`read_json` local copies as other modules do). `project.rs` doc + `models/` in `create`.
    `model.rs`: `reshape_state`, `reshape_options`, `mark_interrupted` call in `adopt`. Tests 9.3-4, 9.3-9, 9.3-10. Accept: pass on unix (fake worker is `#[cfg(unix)]` like `tests/export.rs`).
12. **Evaluate refactor.** `evaluate/runs.rs` `Side`, `Pair`, `answer_pair`, `execute_pair`, `pair_for_run`, `compare()` incl. models (§6.4); `reshape::runs::test`/`read_test`. Tests 9.3-6 and all existing `tests/evaluate.rs`. Accept: existing JSON of `test.json` unchanged except new keys `against`, `loaded`.
13. **Export + training hooks.** `export/runs.rs` `vision_repo` + card line; `train/runs.rs` `extra_targets`/`layers_to_transform`. Tests 9.3-7, 9.3-8.

**Phase D — window, help, docs**

14. **Tauri commands.** §6.5 in `src-tauri/src/lib.rs` + `generate_handler!`; `test_start` `against`. Accept: `cargo build -p nd-lore-forge`.
15. **UI page.** `ui/app.js`: sidebar item, `renderReshape`, `wireReshape`, `rsRefresh`, `onReshapeEvent` (listen `"reshape-event"` in `init()`), distance bar chart, result card, made-models list;
    Step 3 shows `reshape` text; Step 4 "Compare with" select. `ui/style.css`: `.bars` chart styles only if needed. Accept: manual run with a fake worker (`NDFORGE_ENV_DIR` pointing to a
    script env, as the Rust tests do) shows options, progress, result; no console errors.
16. **Help + docs.** `help/reshape.md`, `help.rs` entry, edits to `help/test.md`, `help/use.md`, `help/troubleshooting.md`, `README.md`, `docs/RESHAPE.md`, `PLAN.md`, `docs/TRAINING_SPEED.md`. Accept: `help_markdown("reshape")` test passes.

**Phase E — GPU verification and optional extensions (owner's PC)**

17. Run the §9.4 checklist; record numbers in `docs/RESHAPE.md`; fix what breaks (likely spots: Unsloth loading of a 48-layer folder, Ollama).
18. *(optional)* **Distillation heal.** New job `distill_cache`: original model (4-bit) writes top-32 log-probs per token of the training rows to `models/<name>/distill/*.npy`; `trainjob.py`
    option `distill: {"cache": dir, "alpha": 0.5, "temperature": 1.0}` → custom `compute_loss` = α·CE + (1−α)·KL(top-k). Only after step 17 shows plain healing is insufficient.
19. *(optional)* **MoE expert pruning (REAP).** For `qwen3_5_moe` bases when Unsloth is installed: worker option `experts: {"sparsity": 0.25}` calling
    `FastLanguageModel.reap_prune(model, tokenizer, sparsity=, dataset=<datasets.Dataset of calibration texts>, num_calibration_samples=, save_directory=)` before the layer cut is written.
    **[A: the dataset format Core expects — read `unsloth/models/*reap*` in Unsloth core first; it is not in the sparse clone].** Reuse Studio's refusal list (§0) for the UI gate.
20. *(optional)* Gemma 4 support: slice `embed_tokens_per_layer` / `per_layer_model_projection` by layer and keep `num_kv_shared_layers` consistent — design separately.

---

## 12. Open questions for the owner
1. Is 4-bit scoring of the 27B acceptable, or should the default be CPU-offload 16-bit (needs ≥ 64 GB RAM, slower)?
2. Keep reshaped models inside the project (`models/`, default in this plan) or in a shared folder for all projects (saves disk when several projects use the same slim model)?
3. Grow: offer it at all in normal mode, or only in Expert mode given the expected small gain?
