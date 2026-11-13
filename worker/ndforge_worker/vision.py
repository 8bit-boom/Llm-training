"""Keeping a model's image (and audio) abilities through fine-tuning.

Training only changes the language layers, so the base model's vision part is exactly right for the fine-tuned model. In a GGUF the vision part is a
separate "mmproj" file that llama.cpp / Unsloth Studio load next to the text file. This module finds that file for the base model (the official GGUF
repo of the same model, else a conversion with llama.cpp when the base model is a local folder) and copies the image-processor files a merged model needs.
Nothing here may fail an export: every problem becomes a note."""
import json
import shutil
from pathlib import Path

# the files that tell a loader how to turn an image / video / audio clip into model input
PROCESSOR_FILES = ("preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "audio_preprocessor_config.json", "chat_template.jinja")
MMPROJ_PREFERENCE = ("mmproj-f16.gguf", "mmproj-bf16.gguf", "mmproj-f32.gguf")


def _read_config(repo: str, hub_download=None):
    p = Path(repo)
    if p.is_dir():
        f = p / "config.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    if hub_download is None:
        from huggingface_hub import hf_hub_download as hub_download
    return json.loads(Path(hub_download(repo, "config.json")).read_text(encoding="utf-8"))


def base_modalities(repo: str, hub_download=None) -> list:
    """["image", "audio"] etc. the base model understands besides text ([] for a text-only model or when it cannot be told)."""
    try:
        c = _read_config(repo, hub_download)
    except Exception:
        return []
    found = []
    archs = " ".join(c.get("architectures") or [])
    if c.get("vision_config") is not None or "ConditionalGeneration" in archs or "VL" in archs:
        found.append("image")
    if c.get("audio_config") is not None:
        found.append("audio")
    return found


def gguf_repo_candidates(repo: str) -> list:
    """Where the official GGUF files of a model live: <repo>-GGUF, and unsloth's repo of the same name."""
    if Path(repo).is_dir() or repo.endswith("-GGUF"):
        return [repo] if repo.endswith("-GGUF") else []
    name = repo.split("/")[-1]
    out = [repo + "-GGUF"]
    if not repo.startswith("unsloth/"):
        out.append(f"unsloth/{name}-GGUF")
    return out


def pick_mmproj(files) -> str | None:
    by_lower = {f.lower().split("/")[-1]: f for f in files}
    for want in MMPROJ_PREFERENCE:
        if want in by_lower:
            return by_lower[want]
    other = [f for f in files if "mmproj" in f.lower() and f.lower().endswith(".gguf")]
    return sorted(other)[0] if other else None


def fetch_mmproj(repo: str, out: Path, list_files=None, hub_download=None):
    """Copies the base model's mmproj into `out`. Returns (path or None, note)."""
    if list_files is None or hub_download is None:
        from huggingface_hub import hf_hub_download, list_repo_files
        list_files, hub_download = list_files or list_repo_files, hub_download or hf_hub_download
    problems = []
    for gr in gguf_repo_candidates(repo):
        try:
            name = pick_mmproj(list_files(gr))
            if not name:
                problems.append(f"{gr} has no mmproj file")
                continue
            dest = out / "mmproj-F16.gguf" if name.lower().endswith("f16.gguf") else out / name.split("/")[-1]
            shutil.copyfile(hub_download(gr, name), dest)
            return dest, f"Vision file taken from {gr} ({name})."
        except Exception as exc:
            problems.append(f"{gr}: {type(exc).__name__}")
    return None, "No vision file was found for this model (" + ("; ".join(problems) or "its base model is not on the Hugging Face Hub") + "). The GGUF understands text only."


def copy_processor_files(repo: str, target: Path, hub_download=None) -> list:
    """Puts the base model's image/video processor files next to a merged model (the converters and loaders need them). Returns what was copied."""
    copied = []
    for name in PROCESSOR_FILES:
        if (target / name).exists():
            continue
        try:
            p = Path(repo)
            if p.is_dir():
                src = p / name
                if not src.is_file():
                    continue
            else:
                if hub_download is None:
                    from huggingface_hub import hf_hub_download as hub_download
                src = Path(hub_download(repo, name))
            shutil.copyfile(src, target / name)
            copied.append(name)
        except Exception:
            continue   # the file does not exist for this model: fine
    return copied


def adapter_is_for_full_model(adapter_dir: Path) -> bool:
    """True when an adapter was trained on the whole image+text model (its layer names say "language_model"), so that it must be merged into that model."""
    f = Path(adapter_dir) / "adapter_model.safetensors"
    if not f.is_file():
        return False
    try:
        from safetensors import safe_open
        with safe_open(str(f), framework="pt") as s:
            return any("language_model" in k for k in s.keys())
    except Exception:
        return False


def with_base_processor(tokenizer, repo: str, auto_processor=None):
    """An adapter trained on the text part alone saves a bare tokenizer, but Unsloth's GGUF export builds the vision file from a *processor* (without
    one it stops with "KeyError: 'image_mean'"). Wrap the tokenizer in the base model's own processor, keeping the tokenizer and its chat template.
    Anything that goes wrong returns the tokenizer as it was (the export then simply has no vision file). Unsloth Studio does the same."""
    if tokenizer is None or hasattr(tokenizer, "tokenizer"):
        return tokenizer
    try:
        if auto_processor is None:
            from transformers import AutoProcessor as auto_processor
        processor = auto_processor.from_pretrained(repo)
    except Exception:
        return tokenizer
    if not hasattr(processor, "tokenizer"):
        return tokenizer
    processor.tokenizer = tokenizer
    if getattr(tokenizer, "chat_template", None):
        processor.chat_template = tokenizer.chat_template
    return processor
