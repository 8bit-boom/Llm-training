"""The export job: a trained LoRA adapter -> a merged 16-bit model -> a GGUF file (and a Modelfile is written by the Rust program).

Config keys: adapter (folder) OR full_model (the whole model of a full fine-tune: nothing to merge), base {repo, family, vision_repo (optional: the model whose vision file is used; see _vision_source)}, out (folder), quant ("q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"), quants (optional list: several sizes from one conversion; result.ggufs maps each to its file), backend (auto|unsloth|hf), gguf (bool),
keep_merged (bool, default false: the merged 16-bit copy is large and only needed to make the GGUF; with gguf false the merged model is always kept),
imatrix (optional: the importance matrix for the sizes below q8_0, see imatrix.py; it always goes through llama.cpp. Result: `imatrix` with made / reused / applied / not_applied / notes / file)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import backends, imatrix, protocol, vision
from .families import default_non_thinking

QUANTS = ("q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16")


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


def _vision_source(cfg: dict) -> str:
    """Whose picture abilities (and official vision file) the exported model keeps. Normally the base model itself; for a base that was made by
    "Slim or grow a model" (vision part copied unchanged) it is the original model it was made from (`base.vision_repo`), which has the official GGUF repo."""
    return cfg["base"].get("vision_repo") or cfg["base"]["repo"]


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
    # an adapter trained on the whole image+text model (Unsloth does that) is merged into that whole model, so the merged model keeps its vision part;
    # any other adapter was trained on the text model alone
    loaders = [AutoModelForCausalLM]
    if vision.adapter_is_for_full_model(cfg["adapter"]):
        try:
            from transformers import AutoModelForImageTextToText
            loaders.insert(0, AutoModelForImageTextToText)
        except ImportError:
            pass
    protocol.progress("merge", 0.1, "Loading the base model")
    base, last = None, None
    for loader in loaders:
        try:
            try:
                base = loader.from_pretrained(cfg["base"]["repo"], dtype=dtype)
            except TypeError:
                base = loader.from_pretrained(cfg["base"]["repo"], torch_dtype=dtype)
            break
        except Exception as exc:   # this class cannot load the model: the next one may
            last = exc
            protocol.log(f"{loader.__name__} could not load the base model ({type(exc).__name__}); trying another way.")
    if base is None:
        raise last
    protocol.progress("merge", 0.4, "Adding the trained layers")
    model = PeftModel.from_pretrained(base, cfg["adapter"]).merge_and_unload()
    protocol.progress("merge", 0.7, "Saving the merged model")
    model.save_pretrained(str(merged), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(cfg["adapter"] if (Path(cfg["adapter"]) / "tokenizer_config.json").exists() else cfg["base"]["repo"])
    tok.save_pretrained(str(merged))
    copied = vision.copy_processor_files(cfg["base"]["repo"], merged)
    if copied:
        protocol.log("Image processor files copied from the base model: " + ", ".join(copied))
    patch_template_dir(merged, cfg["base"]["family"])


def gguf_with_llama_cpp(cfg, merged: Path, out: Path, quants, report: dict = None) -> dict:
    """Converts once to f16, then makes every wanted size from it. Returns {quant: file}. With cfg["imatrix"] and a `report` dict (see new_report) the
    importance matrix is made (or reused) after the conversion and given to every size below q8_0; the report says what happened. Raises imatrix.Stopped on Stop."""
    quants = [quants] if isinstance(quants, str) else list(quants)
    conv, quantize = find_llama_cpp()
    if conv is None:
        raise RuntimeError("To make a GGUF file without Unsloth, llama.cpp is needed (convert_hf_to_gguf.py and llama-quantize). Install it from "
                           "https://github.com/ggml-org/llama.cpp and set LLAMA_CPP_DIR, or install Unsloth in the training setup. "
                           f"The merged model is saved in {merged}.")
    f16 = out / "model-f16.gguf"
    protocol.progress("gguf", 0.2, "Converting to GGUF")
    _run([sys.executable, conv, merged, "--outfile", f16, "--outtype", "f16"], "The GGUF conversion")
    files = {}
    others = [q for q in quants if q != "f16"]
    im = cfg.get("imatrix") if report is not None else None
    try:
        matrix = imatrix.prepare(cfg, merged, out, f16, quantize, quants, report) if im else None
        for i, quant in enumerate(others):
            if im and imatrix.stop_requested(im):
                raise imatrix.Stopped()
            final = out / f"model-{quant}.gguf"
            protocol.progress("gguf", 0.5 + 0.4 * i / max(1, len(others)), f"Shrinking to {quant.upper()}")
            if matrix is not None and quant in imatrix.NO_IMATRIX:
                report["not_applied"][quant] = "high precision: the matrix would change almost nothing"
            if matrix is not None and quant not in imatrix.NO_IMATRIX:
                rc, _ = imatrix.run_streaming(imatrix.launcher(quantize) + ["--imatrix", matrix, f16, final, quant.upper()], im, 1, None)
                if rc == 0 and final.is_file():
                    report["applied"].append(quant)
                    files[quant] = final
                    continue
                msg = f"llama-quantize did not accept the importance matrix for {quant.upper()} (exit code {rc}); that size was made without it."
                report["notes"].append(msg)
                report["not_applied"][quant] = "llama-quantize refused the matrix"
                protocol.log(msg)
                final.unlink(missing_ok=True)
            _run(imatrix.launcher(quantize) + [f16, final, quant.upper()], "The quantisation")
            files[quant] = final
    except imatrix.Stopped:
        f16.unlink(missing_ok=True)
        raise
    if "f16" in quants:
        final = out / "model-f16.gguf"
        files["f16"] = final            # the converter's own output, kept as it is
    else:
        f16.unlink(missing_ok=True)
    return {q: files[q] for q in quants}


def new_report() -> dict:
    return {"requested": True, "made": False, "reused": False, "file": None, "applied": [], "not_applied": {}, "notes": [], "info": None}


def use_unsloth_for(backend: str, cfg: dict, quants: list, report) -> bool:
    """Does Unsloth's save_pretrained_gguf make the GGUF? Only with the unsloth backend, and not when an importance matrix is wanted: the installed Unsloth is not
    known to take one, so then llama.cpp (convert + llama-imatrix + llama-quantize) is used. Without llama-imatrix, or when no size below q8_0 is wanted,
    the export goes through Unsloth as before and `report["notes"]` says why there is no matrix."""
    if backend != "unsloth" or not cfg.get("gguf", True):
        return False
    if report is None:
        return True
    conv, quantize = find_llama_cpp()
    if not [q for q in quants if q not in imatrix.NO_IMATRIX]:
        report["notes"].append("The importance matrix helps the sizes below Q8_0; none of them was chosen, so it was not made.")
    elif conv is not None and imatrix.find_llama_imatrix(quantize) is not None:
        protocol.log("An importance matrix was asked for: the GGUF is made with llama.cpp (it applies the matrix) instead of Unsloth.")
        return False
    else:
        report["notes"].append("An importance matrix was asked for, but llama.cpp (with llama-imatrix) was not found, so the GGUF was made through Unsloth without it. "
                               "Install a current llama.cpp and set LLAMA_CPP_DIR to use the matrix.")
        protocol.log(report["notes"][-1])
    return True


def find_gguf(dirs, quant: str, strict: bool = False):
    """Unsloth's save_pretrained_gguf writes into its own folder; pick the file of the wanted quantisation (strict: only a file that names it)."""
    files = [f for d in dirs if Path(d).exists() for f in Path(d).rglob("*.gguf")]
    want = [f for f in files if quant.lower() in f.name.lower()]
    pool = want if strict else (want or [f for f in files if "mmproj" not in f.name.lower()])
    pool = [f for f in pool if "mmproj" not in f.name.lower()]
    return max(pool, key=lambda f: f.stat().st_mtime) if pool else None


def run(cfg: dict) -> int:
    protocol.reserve_stdout()
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    quants = [q.lower() for q in (cfg.get("quants") or [cfg.get("quant", "q4_k_m")])]
    for q in quants:
        if q not in QUANTS:
            raise ValueError(f"Unknown quantisation: {q}")
    quant = quants[0]
    backend = backends.prepare(cfg.get("backend", "auto"))
    full_model = cfg.get("full_model")
    merged = Path(full_model) if full_model else out / "merged"
    gguf_path = None
    ggufs = {}
    note = ""
    report = new_report() if (cfg.get("imatrix") and cfg["imatrix"].get("enabled", True) and cfg.get("gguf", True)) else None
    if report is None:
        cfg = {k: v for k, v in cfg.items() if k != "imatrix"}
    use_unsloth = use_unsloth_for(backend, cfg, quants, report)
    if use_unsloth:
        from unsloth import FastLanguageModel, FastModel
        from .families import family
        loader = FastModel if family(cfg["base"]["family"])["loader"] == "FastModel" else FastLanguageModel
        protocol.progress("merge", 0.1, "Loading the trained model")
        model, tokenizer = loader.from_pretrained(model_name=full_model or cfg["adapter"], max_seq_length=cfg.get("max_seq", 2048), dtype=None, load_in_4bit=False, load_in_16bit=True)
        tmp = out / "unsloth_gguf"
        # a text-only adapter comes back with a bare tokenizer; the vision file needs the base model's processor around it
        wrapped = vision.with_base_processor(tokenizer, cfg["base"]["repo"])
        if wrapped is not tokenizer:
            protocol.log("The base model's image processor was added so the vision file can be made.")
        tokenizer = wrapped
        protocol.progress("gguf", 0.3, "Merging and converting to GGUF (Unsloth)")
        model.save_pretrained_gguf(str(tmp), tokenizer, quantization_method=quants if len(quants) > 1 else quant)
        for q in quants:
            found = find_gguf([tmp, str(tmp) + "_gguf", out], q, strict=len(quants) > 1)
            if found is None:
                raise RuntimeError(f"Unsloth finished but no GGUF file for {q} was found.")
            final = out / f"model-{q}.gguf"
            if Path(found) != final:
                shutil.move(str(found), str(final))
            ggufs[q] = final
        gguf_path = ggufs[quant]
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
            ggufs = gguf_with_llama_cpp(cfg, merged, out, quants, report)
            gguf_path = ggufs[quant]
    mmproj, vnote, mods = None, "", []
    if gguf_path and cfg.get("vision", True):
        try:
            mods = vision.base_modalities(_vision_source(cfg))
            if mods and full_model:
                vnote = "Full fine-tuning changed the vision part of the model too, so the base model's vision file does not fit it. The GGUF understands text only."
            elif mods:
                protocol.progress("vision", 0.95, "Adding the vision file")
                mmproj, vnote = vision.fetch_mmproj(_vision_source(cfg), out)
                if mmproj is None and Path(cfg["base"]["repo"]).is_dir():
                    conv, _ = find_llama_cpp()
                    if conv:
                        target = out / "mmproj-F16.gguf"
                        _run([sys.executable, conv, cfg["base"]["repo"], "--mmproj", "--outfile", target, "--outtype", "f16"], "The vision conversion")
                        mmproj, vnote = target, "Vision file converted from the base model with llama.cpp."
        except Exception as exc:   # the vision file is a bonus: the export itself is done
            mmproj, vnote = None, f"The vision file could not be added ({type(exc).__name__}: {str(exc)[:200]}). The GGUF understands text only."
        if vnote:
            protocol.log(vnote)
    if gguf_path and not cfg.get("keep_merged") and not full_model and merged.exists():
        shutil.rmtree(merged, ignore_errors=True)
    elif not gguf_path:
        note = "Merged 16-bit model only (no GGUF requested)."
    protocol.progress("done", 1.0, "Export ready")
    summary = {"gguf": str(gguf_path) if gguf_path else None, "ggufs": {q: str(f) for q, f in ggufs.items()}, "mmproj": str(mmproj) if mmproj else None, "modalities": mods, "vision_note": vnote, "merged_dir": str(merged) if merged.exists() else None, "full_model": bool(full_model), "quant": quant, "backend": backend,
               "size_mb": round(gguf_path.stat().st_size / 1048576, 1) if gguf_path else None, "note": note,
               "imatrix": dict(report, note=" ".join(report["notes"])) if report is not None else None}
    (out / "export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    protocol.result(summary)
    return 0
