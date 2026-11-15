"""Where llama.cpp is, in ONE place (docs/LLAMA_CPP_INSTALL.md). The export, the importance matrices, the matrix merge and Compare quants all go through here.

A llama.cpp folder D counts when it has `convert_hf_to_gguf.py` AND `llama-quantize(.exe)` in D, in D/build/bin or in D/build/bin/Release. The places, in order:
    1. LLAMA_CPP_DIR                 (explicit: it always wins)
    2. NDFORGE_LLAMA_CPP_DIR         (the program sets it to the folder it installed llama.cpp into)
    3. ~/llama.cpp
    4. a `llama.cpp` folder next to the program (the repository's parent)
`llama-imatrix` and `llama-perplexity` are looked for next to llama-quantize, in every one of those folders (and their build folders), then on the PATH. The converter script
and llama-quantize are never taken from the PATH."""
import os
import shutil
from pathlib import Path

MISSING_CODE = "llama_cpp_missing"

SEARCH_ORDER = ("This program looks in LLAMA_CPP_DIR, then in the folder where it installed llama.cpp itself, then in a folder called llama.cpp in your home folder, then in a folder "
                "called llama.cpp next to the program; llama-imatrix and llama-perplexity are also looked for on the PATH (the converter script is not).")
CAN_INSTALL = "The program can install llama.cpp for you: press \"Install llama.cpp now\" (Training setup page)."


class LlamaCppMissing(Exception):
    """llama.cpp (or a part of it) was not found. The worker reports it with the error code `llama_cpp_missing`, so the page can offer the installer."""
    code = MISSING_CODE


def exe(name: str) -> str:
    return name + ".exe" if os.name == "nt" else name


def candidates() -> list:
    """[(how, folder)] in the order above; `how` is what the Rust side shows (env_var, program_install, home_folder, next_to_program)."""
    out = []
    for how, value in (("env_var", os.environ.get("LLAMA_CPP_DIR")), ("program_install", os.environ.get("NDFORGE_LLAMA_CPP_DIR"))):
        if value:
            out.append((how, Path(value)))
    out.append(("home_folder", Path.home() / "llama.cpp"))
    out.append(("next_to_program", Path(__file__).resolve().parents[2] / "llama.cpp"))
    return out


def layout(d) -> tuple:
    """(convert script, quantize program) when D is a usable llama.cpp folder, else (None, None)."""
    d = Path(d)
    conv = d / "convert_hf_to_gguf.py"
    q = exe("llama-quantize")
    quant = next((p for p in (d / q, d / "build" / "bin" / q, d / "build" / "bin" / "Release" / q) if p.is_file()), None)
    return (conv, quant) if conv.is_file() and quant else (None, None)


def find_llama_cpp():
    """(convert script, quantize program) of the first usable folder, or (None, None)."""
    for _, d in candidates():
        conv, quant = layout(d)
        if conv:
            return conv, quant
    return None, None


def locate() -> dict:
    """What a worker would find, for tests and logs: {"folder", "how", "convert", "quantize"} or {}."""
    for how, d in candidates():
        conv, quant = layout(d)
        if conv:
            return {"folder": str(d), "how": how, "convert": str(conv), "quantize": str(quant)}
    return {}


def find_tool(name: str, quantize=None):
    """`llama-imatrix`, `llama-perplexity`, ...: next to llama-quantize (and its parent), in every candidate folder and its build folders, then on the PATH. None when not found."""
    e = exe(name)
    dirs = []
    if quantize:
        q = Path(quantize)
        dirs += [q.parent, q.parent.parent]
    for _, d in candidates():
        dirs += [d, d / "build" / "bin", d / "build" / "bin" / "Release"]
    for d in dirs:
        if (d / e).is_file():
            return d / e
    found = shutil.which(name)
    return Path(found) if found else None


def not_found_text(what: str, need_convert: bool = False) -> str:
    """The plain-language text for a missing llama.cpp. `what` is the thing that needs it ("an importance matrix can be made", ...)."""
    conv = " and convert_hf_to_gguf.py" if need_convert else ""
    return (f"llama.cpp was not found, so {what}. Install it from https://github.com/ggml-org/llama.cpp (a build that has llama-imatrix{conv}), then tell this program where it is: "
            f"set the environment variable LLAMA_CPP_DIR to its folder. {SEARCH_ORDER} {CAN_INSTALL}")
