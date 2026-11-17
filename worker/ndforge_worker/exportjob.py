"""The export job: a trained LoRA adapter -> a merged 16-bit model -> a GGUF file (and a Modelfile is written by the Rust program).

Config keys: adapter (folder) OR full_model (the whole model of a full fine-tune: nothing to merge), base {repo, family, vision_repo (optional: the model whose vision file is used; see _vision_source)}, out (folder), quant ("q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"), quants (optional list: several sizes from one conversion; result.ggufs maps each to its file), backend (auto|unsloth|hf), gguf (bool),
mixed_fit (optional {"target_gb", "matrix": a matrix file}: the target of the size "mixed_fit", see mixedquant.py), keep_merged (bool, default false: the merged 16-bit copy is large and only needed to make the GGUF; with gguf false the merged model is always kept),
imatrix (optional: the importance matrix for the sizes below q8_0, see imatrix.py. It is measured with llama.cpp's llama-imatrix; the GGUF is then made by Unsloth (its
save_pretrained_gguf(imatrix_file=...)) when the installed build can apply a matrix, else the llama.cpp way. Result: `imatrix` with made / reused / applied / not_applied / notes / file / nextn / via;
and `via` ("unsloth" | "llama.cpp") for the GGUF step of every export)."""
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import backends, imatrix, llamacpp, mixedquant, protocol, vision
from .families import default_non_thinking

QUANTS = ("q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16") + mixedquant.MIXED_IDS      # the last ones are "Mixed precision (our own recipe)": llama.cpp only, see mixedquant.py


def find_llama_cpp():
    """(convert script, quantize program) of a llama.cpp the user has or the program installed: see llamacpp.py for the places and their order."""
    return llamacpp.find_llama_cpp()


class ExportLlamaMissing(RuntimeError, llamacpp.LlamaCppMissing):
    """The export needs llama.cpp and it is not there (error code llama_cpp_missing)."""


def _vision_source(cfg: dict) -> str:
    """Whose picture abilities (and official vision file) the exported model keeps. Normally the base model itself; for a base that was made by
    "Slim or grow a model" (vision part copied unchanged) it is the original model it was made from (`base.vision_repo`), which has the official GGUF repo."""
    return cfg["base"].get("vision_repo") or cfg["base"]["repo"]


_JOB = {"cfg": None}      # the export job being run: `_run` looks for its stop file here (the tools are started from many places)


def _stop_info(cfg) -> dict:
    """Where the Stop button's file is: the job key `stop_file`, else the one inside the importance-matrix block (older program versions)."""
    cfg = cfg or {}
    return {"stop_file": cfg.get("stop_file") or (cfg.get("imatrix") or {}).get("stop_file")}


def _run(cmd, label):
    """Runs a tool of llama.cpp (converter, quantize). With a stop file in the job (`stop_file`) the Stop button ends the tool at once: raises imatrix.Stopped."""
    rc, _tail = imatrix.run_streaming([str(c) for c in cmd], _stop_info(_JOB["cfg"]), 1, label=None, log_tail=True)
    if rc != 0:
        raise RuntimeError(f"{label} failed (exit code {rc}). See the log.")


THINKING_DEFAULTS = ("off", "keep")


def thinking_default_of(cfg: dict) -> str:
    """job key `thinking_default`: "off" (default; the exported template answers without thinking unless asked, what a model trained thinking-off expects) or "keep"
    (the model's own template stays as it is: it thinks unless the client switches thinking off, which is what a model healed with reasoning data expects)."""
    v = cfg.get("thinking_default", "off")
    return v if v in THINKING_DEFAULTS else "off"


def patch_template_dir(folder: Path, family: str, thinking_default: str = "off"):
    """Make the saved tokenizer's chat template answer WITHOUT thinking by default (what the model was trained on), for llama.cpp / Unsloth Studio.
    With thinking_default="keep" the template is left exactly as the model ships it."""
    if family != "qwen3_5" or thinking_default == "keep":
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
    patch_template_dir(merged, cfg["base"]["family"], thinking_default_of(cfg))


def _make_mixed(cfg, quantize, f16: Path, final: Path, quant: str, matrix, report, im, mixed_report: dict):
    """One mixed-precision size ("Mixed precision (our own recipe)", mixedquant.py): llama-quantize with the preset's base type, `--output-tensor-type`, `--token-embedding-type`
    and its ordered `--tensor-type REGEX=TYPE` rules, with the importance matrix when there is one (and once more without it when llama-quantize refuses it, as for the other sizes)."""
    fit_cfg = cfg.get("mixed_fit") or {}
    try:
        info = mixedquant.for_export(quant, f16, fit_cfg, fit_cfg.get("matrix") or (str(matrix) if matrix is not None else None))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"The 16-bit GGUF could not be read to plan the mixed-precision size ({exc}).")
    for n in info["notes"]:
        protocol.log(n)
    tail = info["flags"] + [f16, final, info["base"]]
    base = imatrix.launcher(quantize)
    used = False
    if matrix is not None:
        rc, _ = imatrix.run_streaming(base + ["--imatrix", matrix] + tail, im, 1, None)
        if rc == 0 and final.is_file():
            report["applied"].append(quant)
            used = True
        else:
            msg = f"llama-quantize did not accept the importance matrix for {info['label']} (exit code {rc}); that size was made without it."
            report["notes"].append(msg)
            report["not_applied"][quant] = "llama-quantize refused the matrix"
            protocol.log(msg)
            final.unlink(missing_ok=True)
    if not used:
        _run(base + tail, f"The quantisation ({info['label']})")
    mixed_report[quant] = {"id": info["id"], "requested": quant, "label": info["label"], "recipe": mixedquant.LABEL, "base": info["base"], "embd": info["embd"], "output": info["output"], "rules": info["rules"],
                           "estimate": {k: info["estimate"][k] for k in ("bytes", "gb", "bpw", "unmatched", "dead_rules")}, "notes": info["notes"], "fit": info.get("fit"), "matrix_used": used,
                           "actual_bytes": final.stat().st_size if final.is_file() else None}


def gguf_with_llama_cpp(cfg, merged: Path, out: Path, quants, report: dict = None, mixed_report: dict = None) -> dict:
    """Converts once to f16, then makes every wanted size from it. Returns {quant: file}. With cfg["imatrix"] and a `report` dict (see new_report) the
    importance matrix is made (or reused) after the conversion and given to every size below q8_0; the report says what happened. Raises imatrix.Stopped on Stop."""
    quants = [quants] if isinstance(quants, str) else list(quants)
    _JOB["cfg"] = cfg
    conv, quantize = find_llama_cpp()
    if conv is None:
        raise ExportLlamaMissing("To make a GGUF file without Unsloth, llama.cpp is needed (convert_hf_to_gguf.py and llama-quantize). Install it from "
                                 "https://github.com/ggml-org/llama.cpp and set LLAMA_CPP_DIR, or install Unsloth in the training setup. "
                                 f"{llamacpp.SEARCH_ORDER} {llamacpp.CAN_INSTALL} The merged model is saved in {merged}.")
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
            if quant in mixedquant.MIXED_IDS:
                _make_mixed(cfg, quantize, f16, final, quant, matrix, report, im, mixed_report if mixed_report is not None else {})
                files[quant] = final
                continue
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
    return {"requested": True, "made": False, "reused": False, "file": None, "applied": [], "not_applied": {}, "notes": [], "info": None, "nextn": False, "via": None}


def unsloth_can_apply_imatrix(save_fn, has_resolver=None) -> bool:
    """Can this Unsloth build APPLY an importance matrix, not merely swallow the keyword? The same test as Unsloth Studio's export code: `save_pretrained_gguf`
    has a parameter `imatrix_file` (positional-only does not count: it is passed by keyword), or takes **kwargs and unsloth_zoo has `llama_cpp.resolve_imatrix_file`
    (a `**kwargs` alone may be a function that filters them away; unsloth_zoo's CPU loader is such a function and is refused here, which is stricter than Studio). `has_resolver` (a function returning bool) replaces the import check, for the tests."""
    try:
        params = inspect.signature(save_fn).parameters
    except (TypeError, ValueError):
        return False
    p = params.get("imatrix_file")
    if p is not None and p.kind is not inspect.Parameter.POSITIONAL_ONLY:
        return True
    if not any(q.kind is inspect.Parameter.VAR_KEYWORD for q in params.values()):
        return False
    if str(getattr(save_fn, "__module__", "")).startswith("unsloth_zoo.cpu"):
        return False        # unsloth_zoo's CPU loader takes **kwargs and drops imatrix_file without a word (version 2026.11.4); Studio's test would call it capable
    if has_resolver is not None:
        return bool(has_resolver())
    try:
        from unsloth_zoo.llama_cpp import resolve_imatrix_file  # noqa: F401
    except Exception:       # noqa: BLE001 - any trouble importing means "cannot"
        return False
    return True


def installed_save_fn():
    """The function Unsloth puts on a model as `save_pretrained_gguf` (None when Unsloth cannot be imported). Looked at before the model is loaded, so that the matrix
    can be measured while no model sits in memory."""
    try:
        import unsloth.save as unsloth_save
        return unsloth_save.unsloth_save_pretrained_gguf
    except Exception:       # noqa: BLE001 - no GPU, not installed, a different build: then it cannot be asked
        return None


def plan_gguf(backend: str, cfg: dict, quants: list, report, can_apply=None):
    """(use_unsloth, matrix_via_unsloth): who makes the GGUF, and does Unsloth get a matrix measured here?
    Without a matrix asked for (report None): Unsloth with the unsloth back end, as before. With one, and sizes below Q8_0:
      - llama.cpp (convert script + llama-quantize + llama-imatrix) found and `can_apply()` (this Unsloth build applies a matrix): Unsloth makes the GGUF and is given the matrix;
      - llama.cpp found, the build cannot apply one: llama.cpp makes the GGUF (convert + llama-imatrix + llama-quantize), as before;
      - llama.cpp or llama-imatrix not found: Unsloth without a matrix, and a note says so.
    `report["via"]` and the notes say which; the log too. `can_apply` is called only when it matters (the unsloth package is imported late)."""
    if backend != "unsloth" or not cfg.get("gguf", True):
        return False, False
    if report is None:
        return True, False
    report["via"] = "unsloth"
    conv, quantize = find_llama_cpp()
    use_file = bool((cfg.get("imatrix") or {}).get("use_file"))      # a matrix from the library: nothing is measured, so llama-imatrix is not needed to apply it
    if not [q for q in quants if q not in imatrix.NO_IMATRIX]:
        report["notes"].append("The importance matrix helps the sizes below Q8_0; none of them was chosen, so it was not made.")
    elif not use_file and (conv is None or imatrix.find_llama_imatrix(quantize) is None):
        report["notes"].append("An importance matrix was asked for, but llama.cpp (with llama-imatrix) was not found, so the GGUF was made through Unsloth without it. "
                               "Install a current llama.cpp (the program can do it: Training setup, \"Install llama.cpp now\") or set LLAMA_CPP_DIR to use the matrix.")
        protocol.log(report["notes"][-1])
    elif can_apply is not None and can_apply():
        protocol.log("An importance matrix was asked for: llama.cpp measures it, Unsloth applies it (save_pretrained_gguf with imatrix_file).")
        return True, True
    elif use_file and conv is None:
        report["notes"].append("A saved importance matrix was chosen, but this Unsloth build cannot apply one and llama.cpp was not found, so the GGUF was made through Unsloth without it. "
                               "Install a current llama.cpp (the program can do it: Training setup, \"Install llama.cpp now\") or set LLAMA_CPP_DIR to use the matrix.")
        protocol.log(report["notes"][-1])
    else:
        report["via"] = "llama.cpp"
        protocol.log("An importance matrix was asked for, but this Unsloth build cannot apply one: the GGUF is made with llama.cpp (it applies the matrix) instead of Unsloth.")
        return False, False
    return True, False


def use_unsloth_for(backend: str, cfg: dict, quants: list, report, can_apply=None) -> bool:
    """Does Unsloth's save_pretrained_gguf make the GGUF? See plan_gguf."""
    return plan_gguf(backend, cfg, quants, report, can_apply)[0]


def _newer_than_weights(path: Path, cfg: dict) -> bool:
    folder = cfg.get("full_model") or cfg.get("adapter")
    files = list(Path(folder).glob("*.safetensors")) if folder and Path(folder).is_dir() else []
    return Path(path).is_file() and all(Path(path).stat().st_mtime >= f.stat().st_mtime for f in files)


def matrix_for_unsloth(cfg: dict, merged: Path, out: Path, quants: list, report: dict):
    """The importance matrix (made or reused, saved as out/imatrix.gguf) for an export that Unsloth finishes. It is measured with llama.cpp on a 16-bit GGUF: one
    already in the export folder (model-f16.gguf, newer than the weights), else made now (merge with Transformers + PEFT, then llama.cpp's converter; both are removed
    afterwards, Unsloth merges again by itself). Returns the matrix file or None; every problem is a note, never an error. Raises imatrix.Stopped on Stop."""
    conv, quantize = find_llama_cpp()
    f16 = out / "model-f16.gguf"
    scratch = out / "imatrix-f16.gguf"
    made_merged = []

    def make_f16():
        if _newer_than_weights(f16, cfg):
            protocol.log("The 16-bit GGUF already in the export folder is used to measure the matrix.")
            return f16
        if not cfg.get("full_model") and not merged.exists():
            merge_hf(cfg, merged)
            made_merged.append(merged)
            _release_memory()
        protocol.progress("gguf", 0.15, "Converting to a 16-bit GGUF to measure the matrix")
        _run([sys.executable, conv, merged, "--outfile", scratch, "--outtype", "f16"], "The GGUF conversion")
        return scratch

    try:
        return imatrix.prepare(cfg, merged, out, make_f16, quantize, quants, report)
    except imatrix.Stopped:
        raise
    except Exception as exc:    # noqa: BLE001 - merging or converting failed: Unsloth still makes the files, without the matrix
        msg = f"The importance matrix could not be made ({type(exc).__name__}: {str(exc)[:200]}), so the sizes were made without it."
        report["notes"].append(msg)
        protocol.log(msg)
        return None
    finally:
        scratch.unlink(missing_ok=True)
        (out / "imatrix.partial.gguf").unlink(missing_ok=True)
        for m in made_merged:           # Unsloth merges by itself (and saves the merged model again when it is kept)
            shutil.rmtree(m, ignore_errors=True)


def _release_memory():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:       # noqa: BLE001 - only a courtesy to the next loader
        pass


def find_gguf(dirs, quant: str, strict: bool = False):
    """Unsloth's save_pretrained_gguf writes into its own folder; pick the file of the wanted quantisation (strict: only a file that names it)."""
    files = [f for d in dirs if Path(d).exists() for f in Path(d).rglob("*.gguf")]
    want = [f for f in files if quant.lower() in f.name.lower()]
    pool = want if strict else (want or [f for f in files if "mmproj" not in f.name.lower()])
    pool = [f for f in pool if "mmproj" not in f.name.lower() and "imatrix" not in f.name.lower()]
    return max(pool, key=lambda f: f.stat().st_mtime) if pool else None


def run(cfg: dict) -> int:
    _JOB["cfg"] = cfg
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
    mixed_info, mixed_notes = {}, []
    if backend == "unsloth" and cfg.get("gguf", True) and any(q in mixedquant.MIXED_IDS for q in quants):
        # Unsloth's save_pretrained_gguf cannot give single tensors their own types, which a mixed-precision size is made of: the whole batch goes the llama.cpp way (a second pass
        # through Unsloth for the plain sizes would merge the model again)
        if find_llama_cpp()[0] is None:
            raise ExportLlamaMissing("The mixed-precision sizes need llama.cpp (convert_hf_to_gguf.py and llama-quantize): Unsloth cannot give single tensors their own types. Install it from "
                                     f"https://github.com/ggml-org/llama.cpp and set LLAMA_CPP_DIR, or choose only the plain sizes. {llamacpp.SEARCH_ORDER} {llamacpp.CAN_INSTALL}")
        mixed_notes.append("The mixed-precision sizes need llama-quantize's per-tensor types, which Unsloth's export cannot do, so this export was made with llama.cpp (convert, then llama-quantize)"
                           + (" for every size." if len(quants) > len([q for q in quants if q in mixedquant.MIXED_IDS]) else "."))
        protocol.log(mixed_notes[-1])
        use_unsloth, unsloth_matrix = False, False
        if report is not None:
            report["via"] = "llama.cpp"
    else:
        use_unsloth, unsloth_matrix = plan_gguf(backend, cfg, quants, report, lambda: unsloth_can_apply_imatrix(installed_save_fn()))
    matrix_file = None
    if use_unsloth and unsloth_matrix:
        matrix_file = matrix_for_unsloth(cfg, merged, out, quants, report)
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
        method = quants if len(quants) > 1 else quant
        if matrix_file is not None and not unsloth_can_apply_imatrix(model.save_pretrained_gguf):
            report["notes"].append("The loaded Unsloth could not take the importance matrix after all, so the GGUF was made through Unsloth without it.")
            protocol.log(report["notes"][-1])
            report.update(applied=[], nextn=False)
            matrix_file = None
        try:
            if matrix_file is None:
                model.save_pretrained_gguf(str(tmp), tokenizer, quantization_method=method)
            else:
                # one call for the whole batch: another call would merge the model again. The matrix does (almost) nothing for Q8_0 and the 16-bit files; llama-quantize ignores it there
                model.save_pretrained_gguf(str(tmp), tokenizer, quantization_method=method, imatrix_file=str(matrix_file))
        except Exception as exc:    # noqa: BLE001 - Unsloth refused the matrix: once more without it
            if matrix_file is None or "imatrix" not in str(exc).lower():
                raise
            msg = f"Unsloth did not accept the importance matrix ({type(exc).__name__}: {str(exc)[:200]}), so the GGUF was made through Unsloth without it."
            report["notes"].append(msg)
            protocol.log(msg)
            report.update(applied=[], nextn=False)
            matrix_file = None
            shutil.rmtree(tmp, ignore_errors=True)
            model.save_pretrained_gguf(str(tmp), tokenizer, quantization_method=method)
        if matrix_file is not None:
            for q in quants:
                if q in imatrix.NO_IMATRIX:
                    report["not_applied"][q] = "high precision: the matrix would change almost nothing"
                else:
                    report["applied"].append(q)
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
            patch_template_dir(merged, cfg["base"]["family"], thinking_default_of(cfg))
    else:
        if full_model:
            protocol.progress("merge", 0.5, "Using the fully trained model")
            patch_template_dir(merged, cfg["base"]["family"], thinking_default_of(cfg))
        else:
            merge_hf(cfg, merged)
        if cfg.get("gguf", True):
            ggufs = gguf_with_llama_cpp(cfg, merged, out, quants, **({} if report is None else {"report": report}), **({"mixed_report": mixed_info} if any(q in mixedquant.MIXED_IDS for q in quants) else {}))
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
    via = (report["via"] if report is not None and report.get("via") else "unsloth" if use_unsloth else "llama.cpp") if gguf_path else None
    if report is not None:
        report["via"] = via
    summary = {"gguf": str(gguf_path) if gguf_path else None, "via": via, "ggufs": {q: str(f) for q, f in ggufs.items()}, "mmproj": str(mmproj) if mmproj else None, "modalities": mods, "vision_note": vnote, "merged_dir": str(merged) if merged.exists() else None, "full_model": bool(full_model), "quant": quant, "backend": backend,
               "size_mb": round(gguf_path.stat().st_size / 1048576, 1) if gguf_path else None, "note": note,
               "imatrix": dict(report, note=" ".join(report["notes"])) if report is not None else None, "mixed": mixed_info or None, "mixed_notes": mixed_notes}
    (out / "export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    protocol.result(summary)
    return 0
