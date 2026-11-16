"""The matrix library's own jobs: make an importance matrix WITHOUT exporting a GGUF (docs/IMATRIX.md, "The matrix library"), and make a legacy .dat from one.

    imatrix_make   model source (a trained run | a model: folder, Hugging Face id | an existing GGUF file) + calibration (the project's saved data with general text |
                   a text file of the person's own) + options  ->  <root>/<id>/imatrix.gguf (+ imatrix.dat) + imatrix.json
    imatrix_dat    <root>/<id>/imatrix.gguf  ->  imatrix.dat (llama-imatrix --in-file ... --output-format dat)

The measuring itself is imatrix.make_matrix, the same function the GGUF export calls. Nothing here raises an untranslated error: every failure is a MatrixError with a
plain-language text (the worker's entry point turns it into the job's error), the Stop button (the STOP file) ends the job with imatrix.Stopped ("Cancelled.").

Job: {"job": "imatrix_make", "root": library folder, "id": folder name, "label": text, "model_name": short name, "stop_file": path, "chunks": 128, "ctx": 512,
"nextn": true, "ngl": 99, "dat": false, "statistics": true,
"source": {"kind": "run", "run": name, "adapter": folder | None, "full_model": folder | None, "base": {"repo", "family"}, "f16_candidates": [GGUF files that may be reused]}
        | {"kind": "model", "model": folder or Hugging Face id, "family": ..., "name": ...}
        | {"kind": "gguf", "file": path, "tokenizer": folder or Hugging Face id | None, "family": ...},
"calibration": {"kind": "data", "train": train.jsonl, "data_version": n, "general_share": 0.2, "general": {"id" | "file", ...} | None, "seed": 3407}
             | {"kind": "text", "file": path}}.
The Hugging Face token reaches the worker only as the environment variable HF_TOKEN."""
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from . import data, exportjob, imatrix, llamacpp, protocol

SOURCES = ("run", "model", "gguf")
DAT_NAME = "imatrix.dat"
NO_LLAMA = ("llama.cpp was not found, so no importance matrix can be made. Install it from https://github.com/ggml-org/llama.cpp (a build that has llama-imatrix"
            "{conv}), then tell this program where it is: set the environment variable LLAMA_CPP_DIR to its folder. " + llamacpp.SEARCH_ORDER + " " + llamacpp.CAN_INSTALL)


class MatrixError(Exception):
    """A failure with a plain-language text for the person."""


class MatrixLlamaMissing(MatrixError, llamacpp.LlamaCppMissing):
    """llama.cpp (or its converter) is not there: error code llama_cpp_missing, the page offers the installer."""


def plain(exc: BaseException) -> str:
    """A plain-language text for any exception that nobody planned for."""
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC or "no space left" in low:
        return "The disk is full. Free some space (the matrix library sits in the project folder) and try again."
    if "out of memory" in low:
        return "The computer or its graphics card ran out of memory. Use a smaller model, or a GGUF file instead of a model folder. (" + text[:160] + ")"
    if "no module named" in low:
        return "A training library is missing in the training software (it is needed to merge a trained run). Run the training setup again. (" + text[:160] + ")"
    return f"The matrix could not be made ({text[:200]})."


def free_bytes(path) -> int:
    """Free space on the disk that holds `path` (the nearest folder that exists)."""
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def folder_bytes(folder, pattern="*.safetensors") -> int:
    return sum(f.stat().st_size for f in Path(folder).glob(pattern) if f.is_file())


def check_space(where, need: int, what: str):
    free = free_bytes(where)
    if need and free < need * 1.05:
        raise MatrixError(f"There is not enough free disk space for {what}: about {need / 1e9:.1f} GB are needed and {free / 1e9:.1f} GB are free. Free some space and try again.")


# ------------------------------------------------------------------------------------------------------------------ llama.cpp

def find_tools(need_convert: bool):
    """(convert script or None, llama-imatrix). A source that is already a GGUF needs only llama-imatrix. One finder for everything: llamacpp.py."""
    conv, quantize = exportjob.find_llama_cpp()
    tool = imatrix.find_llama_imatrix(quantize) or llamacpp.find_tool("llama-imatrix")      # llama-imatrix alone is enough for a GGUF source
    if tool is None:
        raise MatrixLlamaMissing(NO_LLAMA.format(conv=" and convert_hf_to_gguf.py" if need_convert else ""))
    if need_convert and conv is None:
        raise MatrixLlamaMissing("llama.cpp was found, but not its convert_hf_to_gguf.py (needed to turn a model into a GGUF file). Use a full llama.cpp folder (a download of the source or a git "
                                 "clone, with a build that has llama-quantize) and set LLAMA_CPP_DIR to it; or choose a GGUF file as the model. " + llamacpp.CAN_INSTALL)
    return conv, tool


def convert_to_f16(conv, folder, out_file, im: dict, span):
    """The export's conversion (convert_hf_to_gguf.py --outtype f16), run so that Stop ends it at once."""
    protocol.progress("imatrix", span[0], "Converting the model to a 16-bit GGUF file")
    rc, _ = imatrix.run_streaming([sys.executable, conv, folder, "--outfile", out_file, "--outtype", "f16"], im, 1, None)
    if rc != 0 or not Path(out_file).is_file():
        Path(out_file).unlink(missing_ok=True)
        raise MatrixError(f"llama.cpp could not turn this model into a GGUF file (exit code {rc}; the log has its last lines). Is it a full-precision Hugging Face model folder "
                          "(safetensors files, the kind a trained model is saved as) of an architecture this llama.cpp knows? Update llama.cpp, or choose a GGUF file as the model.")


def fetch_model_folder(repo: str, im: dict) -> Path:
    """A Hugging Face model id as a folder: the local cache first, else downloaded (the token only through the environment). Raises MatrixError."""
    cached = imatrix.hf_cache_dirs(repo)
    if cached:
        return cached[0]
    protocol.progress("imatrix", 0.0, f"Downloading {repo} from Hugging Face")
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=repo, ignore_patterns=["*.gguf", "*.onnx", "*.msgpack", "*.h5", "*.pth", "original/*", "*.md"]))
    except Exception as exc:    # noqa: BLE001 - offline, gated, wrong name, the library missing: one plain answer
        low = f"{type(exc).__name__} {exc}".lower()
        why = ("needs access: ask for it on its Hugging Face page and add your Hugging Face token (in the general-text box)" if "401" in low or "gated" in low or "403" in low
               else "was not found: check the name (owner/model)" if "404" in low or "not found" in low or "repositorynotfound" in low
               else "could not be reached: check the internet connection")
        raise MatrixError(f"The model {repo} could not be downloaded from Hugging Face: it {why}. ({type(exc).__name__})")


# ------------------------------------------------------------------------------------------------------------------ the sources

def source_f16(job: dict, src: dict, work: Path, conv, im: dict, notes: list):
    """(16-bit GGUF to measure on, tokenizer sources, family, description of the source, weights fingerprint)."""
    kind = src.get("kind")
    span = (0.02, 0.2)
    if kind == "gguf":
        f = Path(src.get("file") or "")
        if not f.is_file():
            raise MatrixError(f"The GGUF file {f} was not found. Choose it again.")
        tok = [src["tokenizer"]] if src.get("tokenizer") else []
        st = f.stat()
        return f, tok, src.get("family") or "generic", f"the GGUF file {f.name}", f"{f.name}:{st.st_size}:{st.st_mtime_ns}"
    if kind == "model":
        model = str(src.get("model") or "")
        if not model:
            raise MatrixError("Choose a model.")
        folder = Path(model) if Path(model).is_dir() else None
        if folder is None:
            if Path(model).exists() or "/" not in model:
                raise MatrixError(f"The model {model} was not found. Choose a folder with a full-precision model, or a Hugging Face id like owner/name.")
            folder = fetch_model_folder(model, im)
        if not (folder / "config.json").is_file():
            raise MatrixError(f"The folder {folder} is not a Hugging Face model folder (it has no config.json). Choose the folder that holds config.json and the .safetensors files.")
        check_space(work, int(folder_bytes(folder) * 1.0), "the 16-bit GGUF file of this model")
        f16 = work / "model-f16.gguf"
        convert_to_f16(conv, folder, f16, im, span)
        return f16, [folder, model], src.get("family") or "generic", f"the model {src.get('name') or model}", imatrix.weights_fingerprint({"full_model": str(folder)})
    if kind == "run":
        base = src.get("base") or {}
        cfg = {"adapter": src.get("adapter"), "full_model": src.get("full_model"), "base": {"repo": base.get("repo", ""), "family": base.get("family", "generic")}}
        label = f"the trained run {src.get('run')}"
        weights = imatrix.weights_fingerprint(cfg)
        for c in src.get("f16_candidates") or []:
            if Path(c).is_file() and exportjob._newer_than_weights(Path(c), cfg):
                protocol.log(f"The 16-bit GGUF of an earlier export ({c}) is used: the run was not merged again.")
                notes.append("The 16-bit file of an earlier export was used, so the run was not merged again.")
                return Path(c), [base.get("repo")], cfg["base"]["family"], label, weights
        if cfg["full_model"]:
            merged = Path(cfg["full_model"])
        else:
            if not cfg["adapter"] or not Path(cfg["adapter"]).is_dir():
                raise MatrixError(f"The saved add-on of the run {src.get('run')} was not found, so it cannot be merged into its model.")
            merged = work / "merged"
            protocol.progress("imatrix", 0.01, "Merging the trained add-on into its model")
            check_space(work, 2 * 10 ** 9, "the merged model")
            exportjob.merge_hf(cfg, merged)
            exportjob._release_memory()
        check_space(work, int(folder_bytes(merged)), "the 16-bit GGUF file of this run")
        f16 = work / "model-f16.gguf"
        convert_to_f16(conv, merged, f16, im, span)
        return f16, [merged, base.get("repo")], cfg["base"]["family"], label, weights
    raise MatrixError("Choose where the model comes from: a trained run, a model, or a GGUF file.")


def own_text(path) -> str:
    f = Path(path or "")
    if not f.is_file():
        raise MatrixError(f"The text file {f} was not found. Choose it again.")
    try:
        head = f.read_bytes()[: 1 << 22]
    except OSError as exc:
        raise MatrixError(f"The text file {f.name} could not be read ({exc.strerror or type(exc).__name__}).")
    text = head.decode("utf-8", "replace")
    if not text.strip():
        raise MatrixError(f"The text file {f.name} is empty. Choose a file with the text the model should read (plain text, a few hundred KB or more).")
    return text


# ------------------------------------------------------------------------------------------------------------------ statistics and the legacy format

def parse_statistics(lines) -> dict | None:
    """The few lines the page shows about a matrix, read leniently from `llama-imatrix --show-statistics` (a table with a header; the exact text is not known for every
    build). {"lines": [...], "entries": n} or None when nothing sensible is found. Never raises."""
    try:
        lines = [l.rstrip("\r\n") for l in lines]
        entries = None
        for l in lines:
            m = re.search(r"\((\d+)\s+tensors?\)", l)
            if m:
                entries = int(m.group(1))
                break
        head_i = next((i for i, l in enumerate(lines) if "\t" in l and re.search(r"\bMax\b", l) and re.search(r"\bMin\b", l)), None)
        out = []
        if entries is not None:
            out.append(f"Tensors measured: {entries}")
        if head_i is not None:
            head = [c.strip() for c in lines[head_i].split("\t")]
            head = [c for c in head if c != ""]
            col = {name: i for i, name in enumerate(head)}
            mean_k = next((k for k in col if k in ("μ", "Mean", "mean")), None)
            rows = []
            for l in lines[head_i + 1:]:
                cells = [c.strip() for c in l.split("\t") if c.strip() != ""]
                if len(cells) < len(head):
                    continue
                try:
                    rows.append([cells[0]] + [float(c) if re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?|nan|inf|-inf", c) else None for c in cells[1:len(head)]])
                except ValueError:
                    continue
            vals = lambda k: [r[col[k]] for r in rows if k in col and col[k] < len(r) and isinstance(r[col[k]], float) and r[col[k]] == r[col[k]]]       # noqa: E731
            if entries is None and rows:
                out.append(f"Tensors measured: {len(rows)}")
                entries = len(rows)
            mx = vals("Max")
            if mx:
                out.append(f"Largest importance value: {max(mx):.4g}")
            if mean_k and vals(mean_k):
                v = vals(mean_k)
                out.append(f"Mean importance (average of the tensors' means): {sum(v) / len(v):.4g}")
            act = vals("% Active")
            if act:
                out.append(f"Share of weights that were active: {sum(act) / len(act):.1f} % on average")
        return {"lines": out, "entries": entries} if out else None
    except Exception:       # noqa: BLE001 - statistics are a nicety
        return None


def show_statistics(tool, f16, matrix, im: dict):
    """`llama-imatrix -m <model> --in-file <matrix> --show-statistics`, parsed. None when this build does not know it or prints nothing sensible. Stop still works."""
    try:
        rc, tail = imatrix.run_streaming(imatrix.launcher(tool) + ["-m", f16, "--in-file", matrix, "--show-statistics"], im, 1, None, tail_len=20000, log_tail=False)
    except imatrix.Stopped:
        raise
    except Exception:       # noqa: BLE001
        return None
    if rc != 0:
        protocol.log("This llama-imatrix has no --show-statistics (or could not read the matrix): no statistics.")
        return None
    return parse_statistics(tail)


def make_dat(tool, gguf: Path, dat: Path, im: dict, notes: list) -> bool:
    """The legacy .dat of a GGUF matrix, as the llama.cpp README documents: `llama-imatrix --in-file x.gguf --output-format dat -o y.dat`; if this build does not know the
    flag, once more with only the extension (`-o y.dat`: a name that does not end in .gguf is written in the legacy format). A note says when neither works."""
    base = imatrix.launcher(tool) + ["--in-file", gguf]
    for extra in (["--output-format", "dat"], []):
        tmp = dat.with_name("imatrix.partial.dat")
        tmp.unlink(missing_ok=True)
        rc, _ = imatrix.run_streaming(base + extra + ["-o", tmp], im, 1, None)
        if rc == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            os.replace(tmp, dat)
            return True
        tmp.unlink(missing_ok=True)
        protocol.log("The legacy .dat could not be made this way" + ("; trying with the extension only." if extra else "."))
    notes.append("The legacy .dat file could not be made: this llama-imatrix did not write it. The matrix itself (imatrix.gguf) is fine; use a newer llama.cpp to convert it.")
    return False


# ------------------------------------------------------------------------------------------------------------------ the jobs

def _size_mb(path) -> float:
    return round(Path(path).stat().st_size / 1048576, 2)


def make(job: dict) -> dict:
    """Makes a matrix into <root>/<id>/. Returns the result. Raises MatrixError (plain text) or imatrix.Stopped; the folder is removed again then."""
    from . import imatrixadapt          # (it imports this module: a late import)
    root, mid = Path(job["root"]), str(job["id"])
    if not mid or mid.startswith("_") or "/" in mid or "\\" in mid or ".." in mid:
        raise MatrixError("That is not a valid name for a matrix.")
    out = root / mid
    if out.exists():
        raise MatrixError(f"The folder {out} exists already. Try again in a minute.")
    src, cal = job.get("source") or {}, job.get("calibration") or {}
    chunks, ctx = int(job.get("chunks", 128)), int(job.get("ctx", imatrix.DEFAULT_CTX))
    if chunks not in imatrix.CHUNKS:
        raise MatrixError(f"Unknown amount of calibration text: {chunks} chunks. Choose 64, 128 or 256.")
    if src.get("kind") not in SOURCES:
        raise MatrixError("Choose where the model comes from: a trained run, a model, or a GGUF file.")
    if cal.get("kind") not in ("data", "text"):
        raise MatrixError("Choose the calibration text: your saved data, or a text file of your own.")
    im = {"stop_file": job.get("stop_file"), "general": cal.get("general"), "seed": int(cal.get("seed", 3407)), "data_version": cal.get("data_version")}
    notes: list = []
    # the tools and the inputs are checked before anything is made
    conv, tool = find_tools(need_convert=src["kind"] != "gguf")
    if src["kind"] == "gguf" and not Path(src.get("file") or "").is_file():
        raise MatrixError(f"The GGUF file {src.get('file')} was not found. Choose it again.")
    if cal["kind"] == "data":
        train = cal.get("train")
        if not train or not Path(train).is_file():
            raise MatrixError("The saved training data of this project (train.jsonl) was not found, so it cannot be used as calibration text. Make the training data again (Step 2), or choose a text file of your own.")
        if src["kind"] == "gguf" and not src.get("tokenizer"):
            raise MatrixError("A GGUF file has no chat template this program can read. To use your saved data, tell which model the file was made from (its folder or Hugging Face id: its chat template is "
                              "used to write the practice conversations), or choose a text file of your own as the calibration text.")
        own = None
    else:
        own = own_text(cal.get("file"))
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    work = out / "work"
    work.mkdir()
    try:
        protocol.progress("imatrix", 0.0, "Preparing")
        f16, tok_sources, family, described, weights = source_f16(job, src, work, conv, im, notes)
        if imatrix.stop_requested(im):
            raise imatrix.Stopped()
        head = imatrix.gguf_has_mtp(f16)
        want_nextn, nnote = False, ""
        if job.get("nextn", True):
            want_nextn, nnote = imatrix.nextn_plan(tool, head)
        if cal["kind"] == "data":
            try:
                tokenizer = imatrix.load_tokenizer([s for s in tok_sources if s])
            except Exception as exc:    # noqa: BLE001
                raise MatrixError("The chat template (the tokenizer) of the model could not be read, so the practice conversations cannot be written the way the model sees them. "
                                  f"({type(exc).__name__}: {str(exc)[:160]}) Choose a model folder or Hugging Face id that has its tokenizer, or a text file of your own as the calibration text.")
            try:
                rows = data.read_jsonl(cal["train"])
            except Exception as exc:    # noqa: BLE001
                raise MatrixError(f"The training data could not be read ({type(exc).__name__}: {str(exc)[:160]}).")
            share = float(cal.get("general_share", imatrix.DEFAULT_SHARE))
            try:
                text, stats, general, gnote = imatrix.calibration_text(tokenizer, rows, family, im, ctx, chunks, share, notes.append, 0.2)
            except imatrix.Stopped:
                raise
            except ValueError as exc:
                raise MatrixError(f"The calibration text could not be made: {exc}")
            calib = work / "calibration.txt"
            calib.write_text(text, encoding="utf-8")
            data_hash, parse_special = imatrix.file_sha256(cal["train"]), True
            what = f"your saved data (version {cal.get('data_version')}) as the model sees it" + (f" and general text ({(cal.get('general') or {}).get('file') or (cal.get('general') or {}).get('id') or 'the default dataset'})" if general else "")
        else:
            share, general, gnote = 0.0, [], ""
            calib = Path(cal["file"])
            stats = {"project_tokens": 0, "general_tokens": 0}
            data_hash = imatrix.file_sha256(calib)
            parse_special = bool(re.search(r"<\|[^|\n]{1,40}\|>", own)) or bool(re.search(r"<(start|end)_of_turn>|\[INST\]", own))
            what = f"the text file {calib.name} (used as it is)"
        partial = out / "imatrix.partial.gguf"
        rc, used_nextn, left_out = imatrix.make_matrix(tool, f16, calib, partial, im, chunks, ctx, want_nextn, nnote, int(job.get("ngl", 99)), parse_special, span=(0.2, 0.95))
        if left_out:
            notes.append(f"The MTP head was left out: {left_out}.")
        if rc != 0 or not partial.is_file():
            raise MatrixError(f"llama-imatrix failed (exit code {rc}; the log has its last lines). Common causes: the file is not a model llama-imatrix can read (try a 16-bit or Q8_0 GGUF), "
                              "not enough memory (a smaller model, or fewer layers on the graphics card), or an old llama.cpp. Update llama.cpp and try again.")
        target = out / "imatrix.gguf"
        os.replace(partial, target)
        key = {"model": job.get("model_name") or "", "weights": hashlib.sha256(weights.encode()).hexdigest() if src["kind"] == "gguf" else weights, "data_hash": data_hash,
               "chunks": chunks, "ctx": ctx, "general_share": share, "general": None, "nextn": want_nextn}
        base_repo = (src.get("base") or {}).get("repo") or (src.get("model") if src["kind"] == "model" else None)
        info = imatrix.matrix_info(key, {**im, "general": cal.get("general") if share > 0 else None}, base_repo, stats, general, gnote, used_nextn, want_nextn, left_out, tool)
        summary = imatrix.matrix_summary(target) or {}
        info.update(id=mid, label=str(job.get("label") or "").strip()[:80], source=described, source_kind=src["kind"], calibration=what, calibration_kind=cal["kind"],
                    run=src.get("run") if src["kind"] == "run" else None, size=target.stat().st_size, arch=summary.get("arch"), layers=summary.get("layers"), entries=summary.get("entries"),
                    via="llama.cpp", has_dat=False, notes=notes,
                    cut_model=imatrixadapt.cut_model_of(src.get("model") if src["kind"] == "model" else (src.get("base") or {}).get("repo") if src["kind"] == "run" else src.get("tokenizer")))
        (out / "imatrix.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        protocol.progress("imatrix", 0.96, "Reading the matrix")
        statistics = show_statistics(tool, f16, target, im) if job.get("statistics", True) else None
        if job.get("dat"):
            protocol.progress("imatrix", 0.98, "Making the legacy .dat file")
            info["has_dat"] = make_dat(tool, target, out / DAT_NAME, im, notes)
        info["statistics"] = statistics
        info["notes"] = notes
        (out / "imatrix.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        protocol.progress("imatrix", 1.0, "Importance matrix ready")
        return {"id": mid, "dir": str(out), "file": str(target), "dat": str(out / DAT_NAME) if info["has_dat"] else None, "size_mb": _size_mb(target), "info": info, "notes": notes,
                "statistics": statistics}
    except BaseException as exc:
        shutil.rmtree(out, ignore_errors=True)
        if isinstance(exc, (MatrixError, imatrix.Stopped, KeyboardInterrupt)):
            raise
        raise MatrixError(plain(exc))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def make_dat_job(job: dict) -> dict:
    """The legacy .dat of a saved matrix: job {"dir": library folder, "stop_file"}."""
    d = Path(job["dir"])
    gguf = d / "imatrix.gguf"
    if not gguf.is_file():
        raise MatrixError("The matrix file imatrix.gguf of this entry was not found.")
    _, tool = find_tools(need_convert=False)
    im = {"stop_file": job.get("stop_file")}
    notes: list = []
    try:
        ok = make_dat(tool, gguf, d / DAT_NAME, im, notes)
    except imatrix.Stopped:
        (d / "imatrix.partial.dat").unlink(missing_ok=True)
        raise
    except Exception as exc:    # noqa: BLE001
        raise MatrixError(plain(exc))
    if not ok:
        raise MatrixError(notes[-1] if notes else "The legacy .dat file could not be made.")
    try:
        info = json.loads((d / "imatrix.json").read_text(encoding="utf-8"))
        info["has_dat"] = True
        (d / "imatrix.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError):
        pass
    return {"dir": str(d), "dat": str(d / DAT_NAME), "size_mb": _size_mb(d / DAT_NAME)}


def run(job: dict) -> int:
    protocol.reserve_stdout()
    protocol.result(make(job))
    return 0


def run_dat(job: dict) -> int:
    protocol.reserve_stdout()
    protocol.result(make_dat_job(job))
    return 0
