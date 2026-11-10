"""The export job: a trained LoRA adapter -> a merged 16-bit model -> a GGUF file (and a Modelfile is written by the Rust program).

Config keys: adapter (folder) OR full_model (the whole model of a full fine-tune: nothing to merge), base {repo, family}, out (folder), quant ("q4_k_m", "q5_k_m", "q8_0", "f16"), backend (auto|unsloth|hf), gguf (bool),
keep_merged (bool, default false: the merged 16-bit copy is large and only needed to make the GGUF; with gguf false the merged model is always kept)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import backends, protocol
from .families import default_non_thinking

QUANTS = ("q4_k_m", "q5_k_m", "q8_0", "f16")


def find_llama_cpp():
    """(convert script, quantize program) of a llama.cpp the user has: LLAMA_CPP_DIR, a `llama.cpp` folder next to the program, or on PATH."""
    cands = [os.environ.get("LLAMA_CPP_DIR"), str(Path.home() / "llama.cpp"), str(Path(__file__).resolve().parents[2] / "llama.cpp")]
    exe = "llama-quantize.exe" if os.name == "nt" else "llama-quantize"
    for d in filter(None, cands):
        d = Path(d)
        conv = d / "convert_hf_to_gguf.py"
        quant = next((q for q in (d / exe, d / "build" / "bin" / exe, d / "build" / "bin" / "Release" / exe, d / exe) if q.is_file()), None)
        if conv.is_file() and quant:
            return conv, quant
    return None, None


def _run(cmd, label):
    protocol.log("$ " + " ".join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout.splitlines()[-40:]:
        protocol.log(line)
    if p.returncode != 0:
        raise RuntimeError(f"{label} failed (exit code {p.returncode}). See the log.")


def patch_template_dir(folder: Path, family: str):
    """Make the saved tokenizer's chat template answer WITHOUT thinking by default (what the model was trained on), for llama.cpp / Unsloth Studio."""
    if family != "qwen3_5":
        return
    f = folder / "chat_template.jinja"
    cfg = folder / "tokenizer_config.json"
    if f.exists():
        f.write_text(default_non_thinking(f.read_text(encoding="utf-8")), encoding="utf-8")
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        if isinstance(data.get("chat_template"), str):
            data["chat_template"] = default_non_thinking(data["chat_template"])
            cfg.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_hf(cfg, merged: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else (torch.float16 if torch.cuda.is_available() else torch.float32)
    protocol.progress("merge", 0.1, "Loading the base model")
    try:
        base = AutoModelForCausalLM.from_pretrained(cfg["base"]["repo"], dtype=dtype)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(cfg["base"]["repo"], torch_dtype=dtype)
    protocol.progress("merge", 0.4, "Adding the trained layers")
    model = PeftModel.from_pretrained(base, cfg["adapter"]).merge_and_unload()
    protocol.progress("merge", 0.7, "Saving the merged model")
    model.save_pretrained(str(merged), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(cfg["adapter"] if (Path(cfg["adapter"]) / "tokenizer_config.json").exists() else cfg["base"]["repo"])
    tok.save_pretrained(str(merged))
    patch_template_dir(merged, cfg["base"]["family"])


def gguf_with_llama_cpp(cfg, merged: Path, out: Path, quant: str) -> Path:
    conv, quantize = find_llama_cpp()
    if conv is None:
        raise RuntimeError("To make a GGUF file without Unsloth, llama.cpp is needed (convert_hf_to_gguf.py and llama-quantize). Install it from "
                           "https://github.com/ggml-org/llama.cpp and set LLAMA_CPP_DIR, or install Unsloth in the training setup. "
                           f"The merged model is saved in {merged}.")
    f16 = out / "model-f16.gguf"
    protocol.progress("gguf", 0.2, "Converting to GGUF")
    _run([sys.executable, conv, merged, "--outfile", f16, "--outtype", "f16"], "The GGUF conversion")
    final = out / f"model-{quant}.gguf"
    if quant == "f16":
        f16.rename(final)
    else:
        protocol.progress("gguf", 0.6, f"Shrinking to {quant.upper()}")
        _run([quantize, f16, final, quant.upper()], "The quantisation")
        f16.unlink(missing_ok=True)
    return final


def find_gguf(dirs, quant: str):
    """Unsloth's save_pretrained_gguf writes into its own folder; pick the file of the wanted quantisation."""
    files = [f for d in dirs if Path(d).exists() for f in Path(d).rglob("*.gguf")]
    want = [f for f in files if quant.lower() in f.name.lower()]
    pool = want or [f for f in files if "mmproj" not in f.name.lower()]
    pool = [f for f in pool if "mmproj" not in f.name.lower()]
    return max(pool, key=lambda f: f.stat().st_mtime) if pool else None


def run(cfg: dict) -> int:
    protocol.reserve_stdout()
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    quant = cfg.get("quant", "q4_k_m").lower()
    if quant not in QUANTS:
        raise ValueError(f"Unknown quantisation: {quant}")
    backend = backends.prepare(cfg.get("backend", "auto"))
    full_model = cfg.get("full_model")
    merged = Path(full_model) if full_model else out / "merged"
    gguf_path = None
    note = ""
    if backend == "unsloth" and cfg.get("gguf", True):
        from unsloth import FastLanguageModel, FastModel
        from .families import family
        loader = FastModel if family(cfg["base"]["family"])["loader"] == "FastModel" else FastLanguageModel
        protocol.progress("merge", 0.1, "Loading the trained model")
        model, tokenizer = loader.from_pretrained(model_name=full_model or cfg["adapter"], max_seq_length=cfg.get("max_seq", 2048), dtype=None, load_in_4bit=False, load_in_16bit=True)
        tmp = out / "unsloth_gguf"
        protocol.progress("gguf", 0.3, "Merging and converting to GGUF (Unsloth)")
        model.save_pretrained_gguf(str(tmp), tokenizer, quantization_method=quant)
        gguf_path = find_gguf([tmp, str(tmp) + "_gguf", out], quant)
        if gguf_path is None:
            raise RuntimeError("Unsloth finished but no GGUF file was found.")
        final = out / f"model-{quant}.gguf"
        shutil.move(str(gguf_path), str(final))
        gguf_path = final
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(str(tmp) + "_gguf", ignore_errors=True)
        if cfg.get("keep_merged") and not full_model:   # the GGUF path above leaves no merged model behind: save it too, from the loaded model
            protocol.progress("merge", 0.85, "Saving the merged 16-bit model")
            model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
            patch_template_dir(merged, cfg["base"]["family"])
    else:
        if full_model:
            protocol.progress("merge", 0.5, "Using the fully trained model")
            patch_template_dir(merged, cfg["base"]["family"])
        else:
            merge_hf(cfg, merged)
        if cfg.get("gguf", True):
            gguf_path = gguf_with_llama_cpp(cfg, merged, out, quant)
    if gguf_path and not cfg.get("keep_merged") and not full_model and merged.exists():
        shutil.rmtree(merged, ignore_errors=True)
    elif not gguf_path:
        note = "Merged 16-bit model only (no GGUF requested)."
    protocol.progress("done", 1.0, "Export ready")
    summary = {"gguf": str(gguf_path) if gguf_path else None, "merged_dir": str(merged) if merged.exists() else None, "full_model": bool(full_model), "quant": quant, "backend": backend,
               "size_mb": round(gguf_path.stat().st_size / 1048576, 1) if gguf_path else None, "note": note}
    (out / "export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    protocol.result(summary)
    return 0
