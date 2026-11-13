"""The "Slim or grow a model" jobs.

    reshape_inspect   {"job": "reshape_inspect", "source": {"repo": "<hub id or folder>"}}  -> what the model is made of (no weights are read or downloaded)
    reshape           cut whole blocks of layers out of a model (prune) or copy a span of layers in (grow), writing a new model folder

Layers are moved as raw bytes between safetensors files (safetensors_io), so neither job needs a GPU or even torch, except for measuring which
layers to cut (prune without an explicit layer list)."""
import datetime
import json
import re
import shutil
import time
from pathlib import Path

from . import protocol, reshape, reshape_width, safetensors_io, slice_io, vision
from .reshape import ReshapeError

# copied byte for byte from the source folder when present (the weights and the index are written, never copied)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json", "tokenizer.model",
                   "chat_template.jinja", "chat_template.json", "generation_config.json")
SIDE_FILES = tuple(dict.fromkeys(TOKENIZER_FILES + vision.PROCESSOR_FILES))
DOWNLOAD_PATTERNS = ["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken", "README*", "LICENSE*"]


# ---------------------------------------------------------------------------------------------------------------- sources

def _hub_config(repo: str) -> dict:
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(repo, "config.json")).read_text(encoding="utf-8"))


def _hub_tensors(repo: str) -> dict:
    """{name: (dtype, shape)} from the headers on the Hub (nothing is downloaded)."""
    from huggingface_hub import HfApi
    return safetensors_io.tensors_from_hub_metadata(HfApi().get_safetensors_metadata(repo))


def check_period(cfg: dict, types: list) -> int:
    """The pattern period, and for Qwen3.5 a check that it is the model's own full_attention_interval (so the interval rule stays true after a cut)."""
    period = reshape.pattern_period(types)
    interval = reshape.text_config(cfg).get("full_attention_interval")
    if reshape.model_type(cfg).startswith("qwen3_5") and isinstance(interval, int) and interval > 0 and period != interval:
        raise ReshapeError(f"The layer types of this model do not follow its attention interval ({interval}); it cannot be reshaped safely.")
    return period


# ------------------------------------------------------------------------------------------------------------------ inspect

def _empty(mt: str, why: str, layers: int = 0, local: bool = False) -> dict:
    return {"model_type": mt, "supported": False, "why": why, "layers": layers, "layer_types": [], "period": 1, "blocks": 0, "layer_params": [],
            "embed_params": 0, "lm_head_params": 0, "vision_params": 0, "mtp_params": 0, "other_params": 0, "total_params": 0, "bytes_per_param": 2.0,
            "has_vision": False, "tied": False, "source_bytes": 0, "local": local,
            "width_dims": None, "width_supported": False, "width_why": why, "width_choices": None}


def _width_info(cfg: dict, divisor: int) -> dict:
    """What the page needs to offer "thinner layers": the sizes, whether this model allows it, and why not."""
    why = reshape_width.width_supported(cfg)
    if why:
        return {"width_dims": None, "width_supported": False, "width_why": why, "width_choices": None}
    dims = reshape_width.width_dims(cfg)
    return {"width_dims": dims, "width_supported": True, "width_why": None, "width_choices": reshape_width.width_choices(dims, divisor)}


def inspect_model(repo: str, divisor: int = reshape_width.FFN_DIVISOR) -> dict:
    local = Path(repo).is_dir()
    if local:
        f = Path(repo) / "config.json"
        if not f.is_file():
            raise ReshapeError(f"There is no config.json in {repo}.")
        cfg = json.loads(f.read_text(encoding="utf-8"))
    else:
        cfg = _hub_config(repo)
    mt = reshape.model_type(cfg)
    tc = reshape.text_config(cfg)
    n = tc.get("num_hidden_layers")
    n = n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else 0
    try:
        reshape.check_supported(cfg)
        if local:
            refs = safetensors_io.list_tensors(repo)
            tensors = {k: (r.dtype, r.shape) for k, r in refs.items()}
            source_bytes = sum(r.nbytes for r in refs.values())
        else:
            tensors = _hub_tensors(repo)
            source_bytes = sum(reshape_numel(s) * safetensors_io.DTYPE_BYTES.get(d, 2) for d, s in tensors.values())
        types = reshape.layer_types_of(tc)
        period = check_period(cfg, types)
        groups = reshape.group_params(tensors, n)
    except ReshapeError as exc:
        return _empty(mt, str(exc), n, local)
    return {"model_type": mt, "supported": True, "why": None, "layers": n, "layer_types": types, "period": period, "blocks": n // period,
            "layer_params": groups["layers"], "embed_params": groups["embed"], "lm_head_params": groups["lm_head"], "vision_params": groups["vision"],
            "mtp_params": groups["mtp"], "other_params": groups["other"], "total_params": groups["total"], "bytes_per_param": groups["bytes_per_param"],
            "has_vision": groups["vision"] > 0 or cfg.get("vision_config") is not None, "tied": groups["lm_head"] == 0,
            "source_bytes": source_bytes, "local": local, **_width_info(cfg, divisor)}


def reshape_numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def run_inspect(cfg: dict) -> int:
    protocol.reserve_stdout()
    protocol.progress("inspect", 0.0, "Reading the model's layout")
    out = inspect_model(cfg["source"]["repo"], int(cfg.get("ffn_divisor", reshape_width.FFN_DIVISOR)))
    protocol.progress("inspect", 1.0, "")
    protocol.result(out)
    return 0


def run_measure(cfg: dict) -> int:
    """job reshape_measure: measure a model once (layer distances and width scores in one pass) and keep the result in cfg["out_dir"]; a measurement that
    is already there for the same model, data and sizes is reused unless cfg["force"] is set."""
    protocol.reserve_stdout()
    src_dir = _resolve_source(cfg["source"]["repo"])
    config = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    reshape.check_supported(config)
    why = reshape_width.width_supported(config)
    if why:
        raise ReshapeError(why)
    tc = reshape.text_config(config)
    types = reshape.layer_types_of(tc)
    period = check_period(config, types)
    dims = reshape_width.width_dims(config)
    folder = Path(cfg["out_dir"])
    if cfg.get("force") and folder.is_dir():
        shutil.rmtree(folder)
    sc = _scores_for({**cfg, "scores_dir": str(folder)}, src_dir, period, config, dims, types, True)
    m = sc["meta"]
    stable = None
    try:
        nb = len(types) // period
        n = max(1, nb // 4)
        stable = reshape.halves_agree(m["matrix_h1"], m["matrix_h2"], n, "contiguous", 1, 1)
    except (KeyError, TypeError, ReshapeError):
        pass
    protocol.progress("done", 1.0, "")
    protocol.result({"key": folder.name, "tokens": m.get("tokens"), "load": m.get("load"), "loss": m.get("loss_before"), "stable_depth": stable, "dir": str(folder)})
    return 0


# ------------------------------------------------------------------------------------------------------------------- the job

def _resolve_source(repo: str) -> Path:
    """A local folder as it is, a Hub id downloaded (weights, configs, tokenizer; no GGUF or pictures) into the Hugging Face cache."""
    if Path(repo).is_dir():
        return Path(repo)
    from huggingface_hub import snapshot_download
    protocol.progress("download", 0.0, f"Downloading {repo} (this can take a while)")
    path = Path(snapshot_download(repo, allow_patterns=DOWNLOAD_PATTERNS))
    protocol.progress("download", 1.0, "")
    return path


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ReshapeError("Measuring the layers needs the training software (Training setup).") from None


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ------------------------------------------------------------------------------------------- measuring (needs torch)

BUILTIN_TEXTS = [
    "The harbour district was quiet until the bells rang; by dawn the smugglers had gone and only the gulls were left to argue over the nets.",
    "A guard captain keeps a ledger of every rumour. Most are false, some are dangerous, and a few are the reason the city still stands.",
    "Q: Who rules the northern marches? A: No single lord does. Three families share the road tolls and distrust each other equally.",
    "Rules: when a character is wounded twice in one scene, the next wound is serious. Healing takes a full night of rest or a skilled healer.",
]


def _dtype():
    import torch
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    return torch.float32


def _ram_bytes() -> int:
    import os
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def _choose_load(requested: str, est_bytes: int) -> str:
    """"16bit" | "4bit" | "offload" for the measuring run. auto: 16-bit if the weights fit the free graphics memory (with room for the measuring),
    else 4-bit NF4 if they fit that way, else the weights are spread over the card and the PC's memory."""
    import torch
    if requested in ("16bit", "4bit", "offload"):
        return requested
    if not torch.cuda.is_available():
        return "16bit"
    free = torch.cuda.mem_get_info()[0]
    if est_bytes <= 0.85 * free:
        return "16bit"
    if est_bytes / 3.6 <= 0.85 * free:
        return "4bit"
    return "offload"


def _load(folder, mode: str):
    """The text model of a folder (AutoModelForCausalLM: vision tower and the multi-token head are left out), ready for measuring."""
    import torch
    from transformers import AutoModelForCausalLM
    kw: dict = {}
    dtype = _dtype()
    if torch.cuda.is_available():
        if mode == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            kw["device_map"] = {"": 0}
        elif mode == "offload":
            vram = torch.cuda.mem_get_info()[0] / 2**30
            ram = _ram_bytes() / 2**30
            kw["device_map"] = "auto"
            kw["max_memory"] = {0: f"{max(1, int(vram - 3))}GiB", "cpu": f"{max(4, int(ram - 8))}GiB"}
        else:
            kw["device_map"] = {"": 0}
    try:
        model = AutoModelForCausalLM.from_pretrained(str(folder), dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(folder), torch_dtype=dtype, **kw)
    model.eval()
    model.config.use_cache = False
    return model


def _free() -> None:
    """Call after `del`-ing the model: collect garbage and give the graphics memory back."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _find_layers(model, n: int):
    """The nn.ModuleList of exactly n decoder layers."""
    import torch
    found = [(name, m) for name, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) == n and all(type(x).__name__.endswith("DecoderLayer") for x in m)]
    if not found:
        raise ReshapeError("The layers of this model could not be found in the loaded program model, so they cannot be measured.")
    found.sort(key=lambda nm: (not nm[0].endswith("layers"), len(nm[0])))
    return found[0][1]


def _calibration(tokenizer, calib_cfg: dict, fam_name: str) -> list:
    """Deterministic batches [{"input_ids": LongTensor[b,t], "attention_mask": LongTensor[b,t]}]: half world-book text, half practice conversations
    (whatever the training file has), right-padded. Without a training file a few built-in sentences are used (enough for the exact-copy self-check)."""
    import random
    import torch
    from . import data
    samples = max(2, int(calib_cfg.get("samples", 128)))
    max_len = int(calib_cfg.get("max_len", 512))
    seed = int(calib_cfg.get("seed", 3407))
    path = calib_cfg.get("train")
    rows = data.read_jsonl(path) if path and Path(path).is_file() else []
    if not rows:
        rows = [{"type": "text", "text": t} for t in BUILTIN_TEXTS] * max(1, samples // len(BUILTIN_TEXTS))
    rng = random.Random(seed)
    texts = [r for r in rows if r.get("type") == "text"]
    chats = [r for r in rows if r.get("type") != "text"]
    rng.shuffle(texts)
    rng.shuffle(chats)
    want_t = min(len(texts), samples // 2)
    want_c = min(len(chats), samples - want_t)
    want_t = min(len(texts), samples - want_c)
    enc, _ = data.encode_rows(tokenizer, texts[:want_t] + chats[:want_c], fam_name, max_len, seed)
    if not enc:
        raise ReshapeError("Nothing in the training data could be used for measuring. Save a training-data version in Step 2 first.")
    inner = data._inner(tokenizer)
    pad = inner.pad_token_id if inner.pad_token_id is not None else (inner.eos_token_id or 0)
    bs = max(1, min(int(calib_cfg.get("batch", 4)), len(enc) // 2))
    batches = []
    for i in range(0, len(enc), bs):
        part = [e["input_ids"] for e in enc[i:i + bs]]
        width = max(len(x) for x in part)
        ids = torch.full((len(part), width), pad, dtype=torch.long)
        mask = torch.zeros((len(part), width), dtype=torch.long)
        for k, x in enumerate(part):
            ids[k, :len(x)] = torch.tensor(x)
            mask[k, :len(x)] = 1
        batches.append({"input_ids": ids, "attention_mask": mask})
    return batches


def _split_single(batches: list) -> list:
    return [{"input_ids": b["input_ids"][i:i + 1], "attention_mask": b["attention_mask"][i:i + 1]} for b in batches for i in range(len(b["input_ids"]))]


def _token_losses(logits, ids, mask):
    """(sum of next-token cross-entropy, number of tokens) over the real (non-padding) positions of a batch."""
    import torch.nn.functional as F
    total, count = 0.0, 0
    ids, mask = ids.to(logits.device), mask.to(logits.device)
    for i in range(ids.shape[0]):
        n = int(mask[i].sum())
        if n < 2:
            continue
        total += float(F.cross_entropy(logits[i, :n - 1].float(), ids[i, 1:n], reduction="sum"))
        count += n - 1
    return total, count


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _measure(model, layers, period: int, batches: list, mode: str, on_batch=None):
    """One forward pass per batch with hooks on the block boundaries. Returns (D, D_half1, D_half2, mean loss, tokens).
    on_batch(index, input_ids, attention_mask), when given, is called before each forward pass (the width measuring uses it to learn the batch)."""
    import math
    import torch
    import torch.nn.functional as F
    nb = len(layers) // period
    dev = next(model.parameters()).device
    store = torch.device("cpu") if mode == "offload" else None
    held: dict = {}

    def pre(k):
        def hook(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            h = h.detach().float()
            held[k] = h.to(store) if store else h
        return hook

    def post(module, args, output):
        h = (output[0] if isinstance(output, tuple) else output).detach().float()
        held[nb] = h.to(store) if store else h

    handles = [layers[k * period].register_forward_pre_hook(pre(k), with_kwargs=True) for k in range(nb)]
    handles.append(layers[-1].register_forward_hook(post))
    size = nb + 1
    acc = [([[0.0] * size for _ in range(size)], [[0] * size for _ in range(size)]) for _ in range(3)]       # all, half 1, half 2
    loss_sum, loss_n, tokens = 0.0, 0, 0
    try:
        with torch.no_grad():
            for bi, b in enumerate(batches):
                held.clear()
                ids, mask = b["input_ids"].to(dev), b["attention_mask"].to(dev)
                if on_batch:
                    on_batch(bi, ids, mask)
                out = model(input_ids=ids, attention_mask=mask, use_cache=False)
                ls, ln = _token_losses(out.logits, ids, mask)
                loss_sum += ls
                loss_n += ln
                del out
                m = mask.to(held[0].device).float()
                tokens += int(m.sum())
                targets = (acc[0], acc[1 + bi % 2])
                for a in range(size):
                    for c in range(a + 1, size):
                        cos = F.cosine_similarity(held[a], held[c], dim=-1).clamp(-1.0, 1.0)
                        ang = torch.arccos(cos) / math.pi
                        s = float((ang * m).sum())
                        n = int(m.sum())
                        for sums, counts in targets:
                            sums[a][c] += s
                            counts[a][c] += n
                held.clear()
                protocol.progress("score", (bi + 1) / len(batches), f"Measuring the layers: {bi + 1} of {len(batches)} batches")
    finally:
        for h in handles:
            h.remove()
    all_, h1, h2 = (reshape.finalize_distance(*a) for a in acc)
    if not any(any(r) for r in acc[2][1]):          # only one batch: both halves are the same measurement
        h2 = h1
    return all_, h1, h2, (loss_sum / loss_n if loss_n else None), tokens


def _score(src_dir: Path, cfg: dict, period: int):
    """(D, D_half1, D_half2, loss_before, info) for the source model: D[a][b] = mean angular distance between the hidden states at block boundaries a and b
    over the calibration tokens. info = {"load", "tokens"}."""
    from transformers import AutoTokenizer
    config = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    n_layers = int(reshape.text_config(config)["num_hidden_layers"])
    est = sum(r.nbytes for r in safetensors_io.list_tensors(src_dir).values())
    mode = _choose_load(cfg.get("load", "auto"), est)
    protocol.progress("calibrate", 0.0, "Preparing the measuring text")
    tok = AutoTokenizer.from_pretrained(str(src_dir))
    batches = _calibration(tok, cfg.get("calibration") or {}, (cfg.get("source") or {}).get("family", "generic"))
    protocol.log(f"Measuring on {len(batches)} batches ({sum(int(b['attention_mask'].sum()) for b in batches):,} tokens), model loaded as {mode}.")
    protocol.progress("calibrate", 1.0, "")
    protocol.progress("score", 0.0, f"Loading the model ({mode})")
    model = _load(src_dir, mode)
    layers = _find_layers(model, n_layers)
    try:
        try:
            D, D1, D2, loss, tokens = _measure(model, layers, period, batches, mode)
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            protocol.log("Out of memory; trying again with one text at a time.")
            _free()
            D, D1, D2, loss, tokens = _measure(model, layers, period, _split_single(batches), mode)
    except RuntimeError as exc:
        if _is_oom(exc):
            raise ReshapeError("The graphics card ran out of memory while measuring the layers. Choose 4-bit measuring (Expert mode) or fewer samples.") from None
        raise
    finally:
        del layers
        del model
        _free()
    return D, D1, D2, loss, {"load": mode, "tokens": tokens}


# ---------------------------------------------------------------------------------------------- width scores (the cache)

SCORES_FORMAT = 1
SCORE_ARRAYS = ("mlp", "attn", "gdn")


def _read_scores(scores_dir, cfg: dict, n_layers: int, dims: dict, types: list):
    """The measurement cached in `scores_dir` (scores.json + scores.npz), or None when there is none or it does not belong to this model and data.
    Returns {"meta": scores.json, "mlp"/"attn"/"gdn": float32 arrays [layers, structures], and the same with "_h1" / "_h2" for the two halves}."""
    if not scores_dir:
        return None
    folder = Path(scores_dir)
    jf, nf = folder / "scores.json", folder / "scores.npz"
    if not (jf.is_file() and nf.is_file()):
        return None
    try:
        meta = json.loads(jf.read_text(encoding="utf-8"))
        np = slice_io.np()
        with np.load(nf) as z:
            arrays = {k: np.array(z[k], dtype=np.float32) for k in z.files}
    except (OSError, ValueError, KeyError) as exc:
        protocol.log(f"The earlier measurement could not be read ({exc}); measuring again.")
        return None
    cal = cfg.get("calibration") or {}
    if meta.get("format") != SCORES_FORMAT or meta.get("source") != (cfg.get("source") or {}).get("repo"):
        protocol.log("The earlier measurement belongs to another model; measuring again.")
        return None
    if cal.get("version") is not None and meta.get("data_version") not in (None, cal.get("version")):
        protocol.log(f"The earlier measurement used data version v{meta.get('data_version')}; measuring again on v{cal.get('version')}.")
        return None
    if (cal.get("samples") is not None and meta.get("samples") != cal.get("samples")) or (cal.get("max_len") is not None and meta.get("max_len") != cal.get("max_len")):
        protocol.log("The earlier measurement used another number of samples or length; measuring again.")
        return None
    n_full = sum(1 for t in types if t == "full_attention")
    want = {"mlp": (n_layers, dims["intermediate"]), "attn": (n_full, dims["heads"]), "gdn": (n_layers - n_full, dims["lin_v_heads"])}
    for name, shape in want.items():
        for suffix in ("", "_h1", "_h2"):
            a = arrays.get(name + suffix)
            if a is None or tuple(a.shape) != shape:
                protocol.log("The earlier measurement does not fit this model's layers; measuring again.")
                return None
    return {"meta": meta, **arrays}


def _depth_from_scores(sc: dict):
    """(D, D_half1, D_half2, loss_before, info) from a cached measurement, or None when it holds no layer distances."""
    m = sc["meta"]
    if not m.get("matrix"):
        return None
    return m["matrix"], m.get("matrix_h1") or m["matrix"], m.get("matrix_h2") or m["matrix"], m.get("loss_before"), {"load": m.get("load"), "tokens": m.get("tokens")}


def _scores_for(cfg: dict, src_dir: Path, period: int, config: dict, dims: dict, types: list, need_depth: bool):
    """The cached measurement when it covers what is needed; otherwise the model is measured (needs torch)."""
    sc = _read_scores(cfg.get("scores_dir"), cfg, int(reshape.text_config(config)["num_hidden_layers"]), dims, types)
    if sc is not None and (not need_depth or sc["meta"].get("matrix")):
        protocol.log("Using the earlier measurement of this model on your data.")
        return sc
    _require_torch()
    return _measure_and_save(cfg, src_dir, period, config, dims, types)


def _keep_sets(sc: dict, src: list, types: list, dims: dict, width: dict):
    """(keep, overlap): per kept old layer the structures that stay, and the mean Jaccard overlap of the choices made from the two halves of the
    measuring text (1.0 = the same choice from either half)."""
    keep: dict = {"mlp": {}, "heads": {}, "gdn": {}}
    ov: list = []
    ordinal = {"full_attention": [], "linear_attention": []}
    for i, t in enumerate(types):
        ordinal[t if t == "full_attention" else "linear_attention"].append(i)
    row = {i: k for t in ordinal.values() for k, i in enumerate(t)}
    for s in sorted(set(src)):
        k = row[s]
        if width["intermediate"] != dims["intermediate"]:
            keep["mlp"][s] = reshape_width.keep_top([float(x) for x in sc["mlp"][s]], width["intermediate"])
            ov.append(reshape_width.overlap(reshape_width.keep_top([float(x) for x in sc["mlp_h1"][s]], width["intermediate"]),
                                            reshape_width.keep_top([float(x) for x in sc["mlp_h2"][s]], width["intermediate"])))
        if types[s] == "full_attention" and width["heads"] != dims["heads"]:
            g = lambda a: reshape_width.keep_grouped([float(x) for x in a[k]], dims["kv_heads"], width["heads"])  # noqa: E731
            keep["heads"][s] = g(sc["attn"])
            ov.append(reshape_width.overlap(g(sc["attn_h1"]), g(sc["attn_h2"])))
        if types[s] == "linear_attention" and width["lin_v_heads"] != dims["lin_v_heads"]:
            g = lambda a: reshape_width.keep_grouped([float(x) for x in a[k]], dims["lin_k_heads"], width["lin_v_heads"])  # noqa: E731
            keep["gdn"][s] = g(sc["gdn"])
            ov.append(reshape_width.overlap(g(sc["gdn_h1"]), g(sc["gdn_h2"])))
    return keep, (sum(ov) / len(ov) if ov else 1.0)


def _mtp_keep(tensors: dict, dims: dict, width: dict):
    """What stays in the multi-token head (never run, so chosen by weight size), or None when there is none."""
    if not any(k.startswith("mtp.layers.") for k in tensors):
        return None
    scores = slice_io.weight_norm_scores(tensors, dims)
    keep: dict = {"mlp": {}, "heads": {}}
    for i, e in scores.items():
        if width["intermediate"] != dims["intermediate"]:
            keep["mlp"][i] = reshape_width.keep_top(e["mlp"], width["intermediate"])
        if width["heads"] != dims["heads"]:
            if e["heads"] is None:
                raise ReshapeError("The multi-token head of this model has no plain attention output; its attention heads cannot be thinned.")
            keep["heads"][i] = reshape_width.keep_grouped(e["heads"], dims["kv_heads"], width["heads"])
    return keep


def _width_event(sc: dict, keep: dict, src: list, overlap: float) -> dict:
    """a small summary for the page's chart: the mean MLP score of every layer, and the share of all scores that the kept neurons hold"""
    mean = [float(sum(float(x) for x in row) / max(1, len(row))) for row in sc["mlp"]]
    total = sum(float(x) for s in keep["mlp"] for x in sc["mlp"][s])
    kept = sum(float(sc["mlp"][s][i]) for s, idx in keep["mlp"].items() for i in idx)
    return {"layers": len(mean), "mlp_mean": mean, "kept_share": (kept / total) if total else None, "stable_overlap": overlap}


def _save_kept(path: Path, src: list, keep: dict, mtp_keep) -> None:
    """width_kept.npz: the kept indices of every layer of the new model (int32), so the surgery can be reproduced exactly"""
    np = slice_io.np()
    arrays: dict = {"src": np.array(src, dtype=np.int32)}
    for j, s in enumerate(src):
        for part in ("mlp", "heads", "gdn"):
            if s in keep[part]:
                arrays[f"{part}_{j}"] = np.array(keep[part][s], dtype=np.int32)
    for part, d in (mtp_keep or {}).items():
        for i, v in d.items():
            arrays[f"mtp_{part}_{i}"] = np.array(v, dtype=np.int32)
    np.savez(path, **arrays)


# ----------------------------------------------------------------------------------- measuring the width (needs torch)

def _measure_all(model, layers, period: int, batches: list, mode: str, types: list, dims: dict):
    """One pass over the batches that measures what the layer cut needs (block-boundary distances, as _measure) and what the width cut needs.
    Width importance (NVIDIA Model-Optimizer's aggregation): a hook on the INPUT of each output projection (mlp.down_proj, self_attn.o_proj,
    linear_attn.out_proj) takes |activation|, averages it over the real tokens of every text, squares it, and sums over the texts; the score of a channel is the
    square root of that sum. A head's score is the L2 norm of its channels' scores. Kept for all texts and separately for the two halves (by batch parity).
    Returns ((D, D1, D2, loss, tokens), scores) with scores = {"mlp": [L, I], "attn": [n_full, nh], "gdn": [n_lin, nv]} (+ "_h1", "_h2") as float32 numpy arrays."""
    import numpy as np
    import torch
    sizes = {"mlp": dims["intermediate"], "attn": dims["heads"] * dims["head_dim"], "gdn": dims["lin_v_heads"] * dims["lin_v_dim"]}
    state = {"mask": None, "batch": 0}
    acc: dict = {}            # (kind, ordinal, part) -> float32 [C], part 0 = all texts, 1 / 2 = the halves
    ordinals = {"mlp": {}, "attn": {}, "gdn": {}}
    handles = []

    def hook_for(kind, k):
        def hook(module, args):
            x = args[0] if args else None
            if x is None or x.shape[-1] != sizes[kind]:
                raise ReshapeError("The layers of this model do not have the sizes config.json promises, so they cannot be measured.")
            a = x.detach().float().abs()
            if a.dim() == 2:
                a = a[None]
            m = state["mask"].to(a.device).float()
            per = (a * m[..., None]).sum(1) / m.sum(1).clamp(min=1.0)[:, None]
            sq = (per ** 2).sum(0)
            for part in (0, 1 + state["batch"] % 2):
                key = (kind, k, part)
                acc[key] = sq.clone() if key not in acc else acc[key] + sq
        return hook

    for i, layer in enumerate(layers):
        full = types[i] == "full_attention"
        for kind, mod in (("mlp", getattr(getattr(layer, "mlp", None), "down_proj", None)),
                          ("attn" if full else "gdn", getattr(getattr(layer, "self_attn" if full else "linear_attn", None), "o_proj" if full else "out_proj", None))):
            if mod is None:
                raise ReshapeError(f"Layer {i} has no plain {'MLP' if kind == 'mlp' else 'attention'} output projection, so it cannot be measured.")
            k = i if kind == "mlp" else len(ordinals[kind])
            ordinals[kind][i] = k
            handles.append(mod.register_forward_pre_hook(hook_for(kind, k)))

    def on_batch(bi, ids, mask):
        state["mask"], state["batch"] = mask, bi

    try:
        depth = _measure(model, layers, period, batches, mode, on_batch)
    finally:
        for h in handles:
            h.remove()
    scores: dict = {}
    per_head = {"attn": dims["head_dim"], "gdn": dims["lin_v_dim"]}
    for kind in ("mlp", "attn", "gdn"):
        n = len(ordinals[kind])
        for part, suffix in ((0, ""), (1, "_h1"), (2, "_h2")):
            rows = []
            for k in range(n):
                v = acc.get((kind, k, part))
                if v is None and part == 2:
                    v = acc.get((kind, k, 1))            # a single batch: both halves are the same measurement
                if v is None:
                    raise ReshapeError("Nothing was measured; the measuring text is empty.")
                s = v.sqrt()
                if kind != "mlp":
                    s = s.reshape(-1, per_head[kind]).pow(2).sum(1).sqrt()
                rows.append(s.cpu().numpy().astype(np.float32))
            scores[kind + suffix] = np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
    return depth, scores


def _write_scores(folder: Path, meta: dict, scores: dict) -> None:
    """scores.npz first, scores.json last: a folder with a scores.json is complete"""
    import os
    np = slice_io.np()
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "scores.partial.npz"
    np.savez(tmp, **scores)
    os.replace(tmp, folder / "scores.npz")
    tmp_json = folder / "scores.partial.json"
    tmp_json.write_text(json.dumps(meta), encoding="utf-8")
    os.replace(tmp_json, folder / "scores.json")


def _measure_and_save(cfg: dict, src_dir: Path, period: int, config: dict, dims: dict, types: list) -> dict:
    """Measures the model on the project's training text (needs torch), keeps the result in cfg["scores_dir"] and returns it like _read_scores."""
    from transformers import AutoTokenizer
    calib = cfg.get("calibration") or {}
    train = calib.get("train")
    if not (train and Path(train).is_file()):
        raise ReshapeError("Thinning the layers is measured on your own text. Save a training-data version in Step 2 first.")
    n_layers = int(reshape.text_config(config)["num_hidden_layers"])
    est = sum(r.nbytes for r in safetensors_io.list_tensors(src_dir).values())
    mode = _choose_load(cfg.get("load", "auto"), est)
    protocol.progress("calibrate", 0.0, "Preparing the measuring text")
    tok = AutoTokenizer.from_pretrained(str(src_dir))
    batches = _calibration(tok, calib, (cfg.get("source") or {}).get("family", "generic"))
    n_tokens = sum(int(b["attention_mask"].sum()) for b in batches)
    protocol.log(f"Measuring on {len(batches)} batches ({n_tokens:,} tokens), model loaded as {mode}.")
    protocol.progress("calibrate", 1.0, "")
    protocol.progress("score", 0.0, f"Loading the model ({mode})")
    model = _load(src_dir, mode)
    layers = _find_layers(model, n_layers)
    try:
        try:
            (D, D1, D2, loss, tokens), scores = _measure_all(model, layers, period, batches, mode, types, dims)
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            protocol.log("Out of memory; trying again with one text at a time.")
            _free()
            (D, D1, D2, loss, tokens), scores = _measure_all(model, layers, period, _split_single(batches), mode, types, dims)
    except RuntimeError as exc:
        if _is_oom(exc):
            raise ReshapeError("The graphics card ran out of memory while measuring the layers. Choose 4-bit measuring (Expert mode) or fewer samples.") from None
        raise
    finally:
        del layers
        del model
        _free()
    meta = {"format": SCORES_FORMAT, "source": (cfg.get("source") or {}).get("repo"), "key": Path(cfg.get("scores_dir") or "").name or None,
            "data_version": calib.get("version"), "samples": calib.get("samples"), "max_len": calib.get("max_len"), "seed": calib.get("seed", 3407), "load": mode,
            "tokens": tokens, "period": period, "boundary_distance": [D[k][k + 1] for k in range(len(D) - 1)], "matrix": D, "matrix_h1": D1, "matrix_h2": D2,
            "loss_before": loss, "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if cfg.get("scores_dir"):
        _write_scores(Path(cfg["scores_dir"]), meta, scores)
    protocol.progress("score", 1.0, "")
    return {"meta": meta, **scores}


def _loss(folder, cfg: dict, mode: str):
    """(mean token cross-entropy of the model in `folder` on the calibration batches, load mode, tokens). Same batches as _score for the same cfg."""
    import torch
    from transformers import AutoTokenizer
    est = sum(r.nbytes for r in safetensors_io.list_tensors(folder).values())
    mode = _choose_load(mode, est)
    tok = AutoTokenizer.from_pretrained(str(folder))
    batches = _calibration(tok, cfg.get("calibration") or {}, (cfg.get("source") or {}).get("family", "generic"))
    model = _load(folder, mode)
    try:
        dev = next(model.parameters()).device
        total, count = 0.0, 0
        with torch.no_grad():
            for b in batches:
                ids, mask = b["input_ids"].to(dev), b["attention_mask"].to(dev)
                ls, ln = _token_losses(model(input_ids=ids, attention_mask=mask, use_cache=False).logits, ids, mask)
                total += ls
                count += ln
    finally:
        del model
        _free()
    return (total / count if count else None), mode, count


def _licence_front_matter(readme: Path) -> str:
    """The licence lines of a model card's front matter (license, license_name, license_link), or ''."""
    if not readme.is_file():
        return ""
    m = re.match(r"^---\r?\n(.*?)\r?\n---", readme.read_text(encoding="utf-8", errors="replace"), re.S)
    if not m:
        return ""
    return "".join(line + "\n" for line in m.group(1).splitlines() if re.match(r"^license(_name|_link)?\s*:", line))


def _copy_side_files(src_dir: Path, out: Path) -> list:
    copied = []
    names = [f for f in SIDE_FILES if (src_dir / f).is_file()] + sorted(p.name for p in src_dir.glob("LICENSE*") if p.is_file())
    for name in dict.fromkeys(names):
        shutil.copy2(src_dir / name, out / name)
        copied.append(name)
    return copied


def _ppl(loss):
    import math
    return None if loss is None else round(math.exp(min(loss, 50.0)), 3)


def _check(cfg: dict, mode: str, src_dir: Path, out: Path, zero, loss_before, load_used: str, notes: list):
    """Text-prediction error before and after (needs torch). A grown model that starts as an exact copy must predict exactly like the original: that is
    a self-test of the surgery. Returns (loss_before, loss_after, load mode); loss_after is the string "failed" when the self-test failed."""
    if not cfg.get("check", True) or (mode == "grow" and zero is None):
        return loss_before, None, load_used
    if not _have_torch():
        notes.append("The before/after check was skipped because the training software is not installed.")
        return loss_before, None, load_used
    try:
        protocol.progress("check", 0.0, "Checking the new model on your text")
        if loss_before is None:
            loss_before, load_used, _ = _loss(src_dir, cfg, load_used)
        protocol.progress("check", 0.5, "Checking the new model on your text")
        loss_after, _, _ = _loss(out, cfg, load_used)
        protocol.progress("check", 1.0, "")
    except Exception as exc:       # the model itself is fine; only the measurement failed
        protocol.log(f"The check failed: {type(exc).__name__}: {exc}")
        notes.append("The before/after check could not be run (see the log).")
        return loss_before, None, load_used
    if mode == "grow" and loss_before is not None and loss_after is not None and abs(loss_after - loss_before) > 1e-3 * max(abs(loss_before), 1e-9):
        protocol.log(f"Self-test failed: loss {loss_before} before, {loss_after} after.")
        return loss_before, "failed", load_used
    return loss_before, loss_after, load_used


def run(cfg: dict) -> int:
    protocol.reserve_stdout()
    started = time.monotonic()
    mode = cfg["mode"]
    if mode not in ("prune", "grow"):
        raise ReshapeError(f"Unknown mode {mode!r}.")
    out = Path(cfg["out"])
    final = Path(cfg["final"]) if cfg.get("final") else out.with_name("model")
    stop_file = out.parent / "STOP"
    if final.exists():
        raise ReshapeError(f"{final} already exists.")
    notes: list = []
    width_req = cfg.get("width")
    if width_req and mode != "prune":
        raise ReshapeError("Thinning the layers is only possible when slimming a model.")
    divisor = int(cfg.get("ffn_divisor", reshape_width.FFN_DIVISOR))

    src_dir = _resolve_source(cfg["source"]["repo"])
    protocol.progress("inspect", 0.0, "Reading the model's layout")
    config = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    reshape.check_supported(config)
    tc = reshape.text_config(config)
    n_layers = int(tc["num_hidden_layers"])
    tensors = safetensors_io.list_tensors(src_dir)
    groups = reshape.group_params({k: (r.dtype, r.shape) for k, r in tensors.items()}, n_layers)
    types = reshape.layer_types_of(tc)
    period = check_period(config, types)
    protect = cfg.get("protect") or {}
    pf, pl = int(protect.get("first", 1)), int(protect.get("last", 1))
    nb = reshape.block_count(n_layers, period)
    protocol.progress("inspect", 1.0, f"{n_layers} layers in blocks of {period}")
    width = dims = keep = mtp_keep = sc = None
    overlap = None
    if width_req:
        why = reshape_width.width_supported(config)
        if why:
            raise ReshapeError(why)
        dims = reshape_width.width_dims(config)
        width = reshape_width.check_width(dims, width_req, divisor)

    scores, loss_before, loss_after, stable, strategy, chosen = None, None, None, None, cfg.get("strategy", "contiguous"), []
    load_used, tokens = cfg.get("load", "auto"), None
    calibration = cfg.get("calibration") or {}
    zero = None
    if mode == "prune":
        blocks = int(cfg.get("blocks") or 0)
        if cfg.get("drop_layers"):
            chosen = reshape.check_whole_blocks([int(x) for x in cfg["drop_layers"]], period, n_layers, pf, pl)
            strategy = "explicit"
        elif width and blocks == 0:
            chosen, strategy = [], "none"
        else:
            lo, hi = reshape.limits(nb, "prune", pf, pl)
            if not lo <= blocks <= hi:
                raise ReshapeError("Slim at most half of the layers, and never the first or last block.")
            if width:
                sc = _scores_for(cfg, src_dir, period, config, dims, types, True)
                D, D1, D2, loss_before, info = _depth_from_scores(sc)
            else:
                _require_torch()
                D, D1, D2, loss_before, info = _score(src_dir, cfg, period)
            load_used, tokens = info["load"], info["tokens"]
            chosen = reshape.choose_drop(D, blocks, strategy, pf, pl)
            stable = reshape.halves_agree(D1, D2, blocks, strategy, pf, pl)
            scores = {"boundary_distance": [D[k][k + 1] for k in range(nb)], "matrix": D, "chosen_blocks": chosen, "period": period, "stable": stable}
            protocol.emit("reshape_scores", **scores)
        if chosen:
            drop = reshape.blocks_to_layers(chosen, period)
            src = reshape.prune_src(n_layers, drop)
            protocol.log(f"Removing layers {drop[0]} to {drop[-1]} ({len(chosen)} blocks of {period})." if len(chosen) == 1 or drop == list(range(drop[0], drop[-1] + 1))
                         else f"Removing blocks {chosen} (layers {drop}).")
        else:
            src = list(range(n_layers))
        if width:
            if sc is None:
                sc = _scores_for(cfg, src_dir, period, config, dims, types, False)
                load_used, tokens = sc["meta"].get("load", load_used), sc["meta"].get("tokens")
                loss_before = sc["meta"].get("loss_before")
            keep, overlap = _keep_sets(sc, src, types, dims, width)
            mtp_keep = _mtp_keep(tensors, dims, width) if groups["mtp"] else None
            protocol.emit("width_scores", **_width_event(sc, keep, src, overlap))
            protocol.log(f"Thinning every layer: MLP {dims['intermediate']:,} to {width['intermediate']:,}"
                         + (f", linear-attention value heads {dims['lin_v_heads']} to {width['lin_v_heads']}" if width["lin_v_heads"] != dims["lin_v_heads"] else "")
                         + (f", attention heads {dims['heads']} to {width['heads']}" if width["heads"] != dims["heads"] else "") + ".")
    else:
        grow = cfg.get("grow") or {}
        src, zero_all = reshape.grow_src(n_layers, period, int(cfg["blocks"]), grow.get("layout", "solar"))
        init = grow.get("init", "identity")
        if init not in ("identity", "copy"):
            raise ReshapeError(f"Unknown start {init!r}.")
        zero = zero_all if init == "identity" else None
        if zero is not None:
            notes.append("The new layers start switched off (their output weights are zero), so the grown model first answers exactly like the original.")
    reshape.check_pattern(types, src, period)
    notes_cfg: list = []
    new_cfg = reshape.fix_config(config, src, notes_cfg)
    notes += notes_cfg
    plan = reshape.plan_tensors(tensors, src, zero)
    if width:
        new_cfg = reshape_width.fix_config_width(new_cfg, dims, width)
        plan = reshape_width.plan_width(plan, src, types, dims, width, keep, mtp_keep)
        if groups["mtp"]:
            notes.append("The multi-token head was cut to fit but was not measured or healed; do not use it for speculative decoding." if mtp_keep
                         else "The multi-token head was left as it was.")
    new_groups = reshape.group_params({t.name: (t.dtype, t.shape) for t in plan}, len(src)) if width else None
    need = sum(t.nbytes for t in plan)
    free = shutil.disk_usage(out.parent)[2]
    if free < need * 1.1:
        protocol.error("disk_full", f"There is not enough free space: {need * 1.1 / 1e9:.1f} GB needed in {out.parent}, {free / 1e9:.1f} GB free.")
        return 1

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    last = [-1.0]

    def on_progress(done, total):
        if total and (done / total - last[0] >= 0.01 or done == total):
            last[0] = done / total
            protocol.progress("write", done / total, f"Writing the new model: {done / 1e9:.1f} of {total / 1e9:.1f} GB")

    try:
        safetensors_io.write_shards(plan, out, shard_bytes=int(float(cfg.get("shard_gb", 5)) * 2**30), on_progress=on_progress,
                                    should_stop=lambda: stop_file.exists())
        (out / "config.json").write_text(json.dumps(new_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        _copy_side_files(src_dir, out)
        loss_before, loss_after, load_used = _check(cfg, mode, src_dir, out, zero, loss_before, load_used, notes)
        if loss_after == "failed":
            shutil.rmtree(out, ignore_errors=True)
            protocol.error("grow_check_failed", "The new model does not give the same answers as the original although it should; nothing was saved. "
                                                "Please make a problem report.")
            return 1
        zero_init = [j for j, z in enumerate(zero) if z] if zero else []
        info = reshape.describe(groups, n_layers, src, types)
        if width:
            info["params_after"] = new_groups["total"]            # exact: counted from the shapes that were written
        wmeta = None
        if width:
            wmeta = {"intermediate_size": [dims["intermediate"], width["intermediate"]], "linear_num_value_heads": [dims["lin_v_heads"], width["lin_v_heads"]],
                     "num_attention_heads": [dims["heads"], width["heads"]], "kept": "width_kept.npz", "scores_key": Path(cfg.get("scores_dir") or "").name or None,
                     "stable_overlap": round(overlap, 4), "mtp": ("sliced by weight size, not measured" if mtp_keep else "copied unchanged") if groups["mtp"] else "none"}
            _save_kept(out / "width_kept.npz", src, keep, mtp_keep)
        shape_word = ("both" if chosen else "width") if width else ("depth" if mode == "prune" else None)
        meta = {"format": 2 if width else 1, "tool": "ND Lore Forge", "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "origin": {"repo": cfg["source"]["repo"], "family": cfg["source"].get("family"), "layers": n_layers, "params": groups["total"]},
                "mode": mode, "period": period, "layers": len(src), "params": info["params_after"], "src_layers": src, "dropped": info["dropped"],
                "duplicated": info["duplicated"], "zero_init": zero_init, "strategy": strategy if mode == "prune" else cfg.get("grow", {}).get("layout", "solar"),
                "calibration": ({"version": calibration.get("version"), "samples": calibration.get("samples"), "max_len": calibration.get("max_len"),
                                 "tokens": tokens} if (scores or sc) else None),
                "loss_before": loss_before, "loss_after": loss_after, "stable": stable,
                "vision": "unchanged (copied from the original)" if groups["vision"] else "none",
                "mtp": ("sliced by weight size, not measured" if mtp_keep else "copied unchanged") if width and groups["mtp"] else "copied unchanged" if groups["mtp"] else "none"}
        if width:
            meta["shape"] = shape_word
            meta["width"] = wmeta
        (out / "ndforge_reshape.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        label = f"{'Slimmed' if mode == 'prune' else 'Grown'} from {cfg['source']['repo']}: {n_layers} to {len(src)} layers."
        if width:
            label += f" Every layer made thinner (MLP {dims['intermediate']} to {width['intermediate']})."
        (out / "README.md").write_text(f"---\n{_licence_front_matter(src_dir / 'README.md')}---\n# {out.parent.name}\n\n{label}\n\n"
                                        "Made with ND Lore Forge (Slim or grow a model). See ndforge_reshape.json for exactly which layers were used.\n", encoding="utf-8")
        out.rename(final)
    except safetensors_io.Cancelled:
        shutil.rmtree(out, ignore_errors=True)
        protocol.result({"stopped": True, "model_dir": None})
        return 0
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    protocol.progress("done", 1.0, "")
    protocol.result({"model_dir": str(final), "mode": mode, "layers_before": n_layers, "layers_after": len(src), "src_layers": src, "dropped": info["dropped"],
                     "duplicated": info["duplicated"], "zero_init": zero_init, "params_before": groups["total"], "params_after": info["params_after"],
                     "bytes_written": need, "loss_before": loss_before, "loss_after": loss_after, "ppl_before": _ppl(loss_before), "ppl_after": _ppl(loss_after), "stable": stable,
                     "boundary_distance": (scores or {}).get("boundary_distance"), "chosen_blocks": chosen if mode == "prune" else None, "load": load_used,
                     "seconds": round(time.monotonic() - started, 1), "notes": notes, "stopped": False,
                     "shape": shape_word, "width": ({"before": {"intermediate_size": dims["intermediate"], "linear_num_value_heads": dims["lin_v_heads"],
                                                                "num_attention_heads": dims["heads"]},
                                                      "after": {"intermediate_size": width["intermediate"], "linear_num_value_heads": width["lin_v_heads"],
                                                                "num_attention_heads": width["heads"]},
                                                      "stable_overlap": wmeta["stable_overlap"], "scores_key": wmeta["scores_key"], "mtp": wmeta["mtp"]}
                                                     if width else None)})
    return 0
