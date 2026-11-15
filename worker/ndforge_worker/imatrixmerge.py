"""Merging importance matrices (docs/IMATRIX.md, "Merging matrices"): two or more matrices of the library, files, and/or Unsloth's published matrix for a base model, combined
by llama.cpp into one new library entry.

    llama-imatrix --in-file a.gguf --in-file b.gguf -o merged.gguf          (no model and no text needed: "No prompt provided; combining precomputed matrices only")

llama-imatrix ADDS the statistics (tensor `<name>.in_sum2` and `<name>.counts` of every input are summed), so a large matrix outweighs a small one; the worker says how much
each parent weighs. The published matrix describes the UNMODIFIED base model, not a fine-tune of it.

    imatrix_published   look up Unsloth's matrix for a base model on the Hub (the repo and the file name are derived the way Unsloth's own code does it); says its size
    imatrix_merge       resolve the inputs (downloading the published one into a cache), check them against each other from their GGUF headers (a mismatch is a warning, never
                        a refusal), run llama-imatrix, register <root>/<id>/imatrix.gguf + imatrix.json (source_kind "merged", the parents with their hashes)

Job: {"job": "imatrix_published", "base": "Qwen/Qwen3.8-27B", "cache": folder}
and  {"job": "imatrix_merge", "root": library folder, "id", "label", "stop_file", "cache": folder, "notes": [text], "inputs": [{"kind": "library", "id", "file", "label", "model", "chunks"} |
{"kind": "file", "file"} | {"kind": "published", "base": "owner/name"}]}.
The Hugging Face token reaches the worker only as the environment variable HF_TOKEN; HF_ENDPOINT (default https://huggingface.co) is read for mirrors and for the tests."""
import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import imatrix, protocol
from .imatrixjob import MatrixError, check_space, find_tools, free_bytes, plain  # noqa: F401  (free_bytes is replaced by the tests)

# the file names Unsloth's zoo looks for, in its order (unsloth_zoo.llama_cpp.IMATRIX_UPSTREAM_NAMES, version 2026.11.4); the GGUF one is named `.gguf_file` so that the Hub does
# not list it as a model, and is renamed to `.gguf` locally. unsloth.save looks for the first two only.
UPSTREAM_NAMES = ("imatrix_unsloth.dat", "imatrix_unsloth.gguf_file", "imatrix_unsloth.gguf")
ENDPOINT = "https://huggingface.co"
TIMEOUT = 30
MAX_PUBLISHED = 4 << 30                 # a published matrix is some hundred MB; refuse to fetch anything absurd


class HubError(MatrixError):
    """A Hub problem in plain words. `kind`: offline | not_found | no_file | gated | local | http."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or ENDPOINT).rstrip("/")


def token() -> str:
    return (os.environ.get("HF_TOKEN") or "").strip()


# ------------------------------------------------------------------------------------------------------------------ which repo? (as Unsloth's code derives it)

def published_repo(base: str):
    """The Hub repo that holds Unsloth's matrix for `base`, derived exactly as unsloth/save.py `_gguf_repo_candidates` does: a name that is a folder on this computer has none
    (None); a name already ending in `-GGUF` is kept; any other `owner/name` becomes `unsloth/<name>-GGUF` (only the last part of the name is used). Before that the code asks
    `get_model_name(name, load_in_4bit=False)`, which maps Unsloth's pre-quantised names back to the 16-bit one; of that only the well-known suffixes `-unsloth-bnb-4bit` and
    `-bnb-4bit` are undone here (the full table is Unsloth's and cannot be read without running it)."""
    name = str(base or "").strip()
    if not name or os.path.isdir(name):
        return None
    for suffix in ("-unsloth-bnb-4bit", "-bnb-4bit"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name if name.endswith("-GGUF") else f"unsloth/{name.split('/')[-1]}-GGUF"


# ------------------------------------------------------------------------------------------------------------------ the Hub (urllib only: nothing to install, nothing to import)

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def _open(url: str, tok: str, timeout: int = TIMEOUT, depth: int = 0):
    """GET with redirects followed by hand: the token goes only to the Hub itself, never to the host it redirects to (a download server)."""
    headers = {"User-Agent": "ndforge-worker"}
    if tok and urllib.parse.urlsplit(url).netloc == urllib.parse.urlsplit(endpoint()).netloc:
        headers["Authorization"] = "Bearer " + tok
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        return opener.open(urllib.request.Request(url, headers=headers), timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307, 308) and exc.headers.get("Location") and depth < 5:
            return _open(urllib.parse.urljoin(url, exc.headers["Location"]), tok, timeout, depth + 1)
        raise


def _hub_error(exc: BaseException, repo: str, what: str) -> HubError:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403):
            return HubError("gated", f"The Hub refused {what}: the repository {repo} needs you to accept its terms on its Hugging Face page and a Hugging Face token (the box under the general "
                                     f"text) to download it. (HTTP {exc.code})")
        if exc.code == 404:
            return HubError("not_found", f"The Hub has no repository {repo}.")
        return HubError("http", f"The Hub answered {what} with an error (HTTP {exc.code}). Try again later.")
    return HubError("offline", f"huggingface.co could not be reached ({type(exc).__name__}). Check the internet connection; only this download needs it, everything else in this "
                               "program works offline.")


def lookup(base: str, cache) -> dict:
    """Unsloth's published matrix for `base`: {"base", "repo", "file" (the name on the Hub), "local_name" (`.gguf_file` becomes `.gguf`), "size", "url", "page", "gated", "cached",
    "cache_file"}. Raises HubError (plain words) when the base is a local folder, the repo does not exist, holds none of the file names, needs access, or the Hub cannot be reached."""
    repo = published_repo(base)
    if repo is None:
        raise HubError("local", f"{base or 'This model'} is a folder on this computer (a model made by Slim or grow, or a healed one), so it has no published matrix on the Hub. "
                                "Choose the original model it was made from.")
    url = f"{endpoint()}/api/models/{urllib.parse.quote(repo, safe='/')}?blobs=true"
    try:
        with _open(url, token()) as r:
            info = json.loads(r.read(1 << 24).decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404 or (exc.code == 401 and not token()):
            raise HubError("not_found", f"There is no published matrix for {base}: the Hub has no repository {repo} (or it is private). The name is made from the model's name the way "
                                        "Unsloth's own code does it (unsloth/<name>-GGUF). Choose another base model, or measure a matrix yourself.")
        raise _hub_error(exc, repo, "the lookup")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise _hub_error(exc, repo, "the lookup")
    files = {s.get("rfilename"): s for s in info.get("siblings") or [] if isinstance(s, dict)}
    name = next((n for n in UPSTREAM_NAMES if n in files), None)
    if name is None:
        raise HubError("no_file", f"The repository {repo} exists but holds none of the files {', '.join(UPSTREAM_NAMES)}, so Unsloth publishes no matrix for {base}.")
    size = files[name].get("size")
    local_name = name[: -len(".gguf_file")] + ".gguf" if name.endswith(".gguf_file") else name
    cache_file = Path(cache) / repo.replace("/", "--") / local_name if cache else None
    cached = bool(cache_file and cache_file.is_file() and (not isinstance(size, int) or cache_file.stat().st_size == size))
    return {"base": base, "repo": repo, "file": name, "local_name": local_name, "size": size if isinstance(size, int) else None, "page": f"{endpoint()}/{repo}",
            "url": f"{endpoint()}/{urllib.parse.quote(repo, safe='/')}/resolve/main/{name}", "gated": bool(info.get("gated")), "cached": cached, "cache_file": str(cache_file) if cache_file else None}


def download(info: dict, im: dict, span=(0.05, 0.55)) -> Path:
    """Downloads the published matrix into its cache folder (kept for the next merge; nothing is downloaded when the cached file has the size the Hub states). The Stop file ends it
    at once. Returns the local file."""
    target = Path(info["cache_file"])
    if info["cached"]:
        protocol.log(f"The published matrix is already downloaded ({target}).")
        return target
    if info.get("size") and info["size"] > MAX_PUBLISHED:
        raise HubError("http", f"The published matrix is {info['size'] / 1e9:.1f} GB, far more than a matrix should be; it was not downloaded.")
    target.parent.mkdir(parents=True, exist_ok=True)
    check_space(target.parent, int(info.get("size") or 0), "the published matrix")
    part = target.with_name(target.name + ".part")
    tok = token()
    try:
        with _open(info["url"], tok) as r, open(part, "wb") as out:
            total = int(r.headers.get("Content-Length") or info.get("size") or 0)
            got, last = 0, 0.0
            protocol.progress("imatrix", span[0], f"Downloading Unsloth's published matrix ({info['repo']})")
            while True:
                if imatrix.stop_requested(im):
                    raise imatrix.Stopped()
                block = r.read(1 << 20)
                if not block:
                    break
                out.write(block)
                got += len(block)
                if time.time() - last > 0.5 and total:
                    last = time.time()
                    protocol.progress("imatrix", span[0] + (span[1] - span[0]) * min(got / total, 0.99), f"Downloading Unsloth's published matrix: {got / 1e6:.0f} of {total / 1e6:.0f} MB")
        if info.get("size") and part.stat().st_size != info["size"]:
            raise HubError("offline", "The download of the published matrix was cut off. Try again.")
        os.replace(part, target)
    except HubError:
        part.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, OSError) as exc:
        part.unlink(missing_ok=True)
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
            raise MatrixError("The disk is full while downloading the published matrix. Free some space and try again.")
        raise _hub_error(exc, info["repo"], "the download")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    protocol.log(f"Downloaded {info['repo']}/{info['file']} ({target.stat().st_size / 1e6:.1f} MB) into {target}.")
    return target


# ------------------------------------------------------------------------------------------------------------------ do the inputs fit each other?

def _tensor_set(path):
    """The weight names a matrix has data for (`blk.0.attn_k.weight` of `blk.0.attn_k.weight.in_sum2`), or None when it cannot be read (a legacy .dat)."""
    try:
        _, names = imatrix.read_gguf_header(path)
    except (OSError, ValueError):
        return None
    return {re.sub(r"\.(in_sum2|in_sum|counts)$", "", n) for n in names}


def compat(inputs: list) -> list:
    """Plain-language warnings from the headers of the resolved inputs (each {"label", "file"}): architecture, layers, and the tensors that only some of them have. Never a refusal:
    llama-imatrix decides (and its own message is passed on)."""
    out, infos = [], []
    for i in inputs:
        s = imatrix.matrix_summary(i["file"])
        if s is None:
            out.append(f"{i['label']} could not be read here (a legacy .dat file or an unusual file), so it could not be checked against the others. llama-imatrix will tell if they do not fit.")
        infos.append((i, s, _tensor_set(i["file"]) if s is not None else None))
    known = [x for x in infos if x[1] is not None]
    for (a, sa, ta), (b, sb, tb) in zip(known, known[1:]):
        if sa["arch"] and sb["arch"] and sa["arch"] != sb["arch"]:
            out.append(f"{a['label']} is for the architecture \"{sa['arch']}\" and {b['label']} for \"{sb['arch']}\". They will most likely not fit together.")
        if sa["layers"] is not None and sb["layers"] is not None and sa["layers"] != sb["layers"]:
            out.append(f"{a['label']} has data for {sa['layers']} layers and {b['label']} for {sb['layers']}. They may come from different models (a slimmed or grown model, or an MTP head in only one).")
        if ta is not None and tb is not None and ta != tb:
            only = sorted(ta ^ tb)
            out.append(f"{len(only)} tensors have data in only one of {a['label']} and {b['label']} (for example {only[0]}). Their statistics are merged where both have them; the rest is copied.")
    return out


def weights(inputs: list) -> tuple:
    """([{"label", "chunks", "share"}], note): how much each parent weighs. The statistics are ADDED, so a matrix made from more text weighs more. `share` is None when a chunk count is
    not known; the note is "" unless one parent carries more than three quarters of all text."""
    chunks = [i.get("chunks") for i in inputs]
    total = sum(c for c in chunks if isinstance(c, int))
    rows = [{"label": i["label"], "chunks": c if isinstance(c, int) else None, "share": round(c / total, 3) if isinstance(c, int) and all(isinstance(x, int) for x in chunks) and total else None}
            for i, c in zip(inputs, chunks)]
    note = ""
    if all(isinstance(c, int) for c in chunks) and total:
        big = max(zip(chunks, inputs), key=lambda t: t[0])
        if big[0] / total > 0.75:
            note = (f"{big[1]['label']} was measured on {big[0]} chunks of text and carries about {round(big[0] / total * 100)} % of the merged statistics: merging adds them, so the larger "
                    "matrix outweighs the smaller. To give your own matrix more say, make it with more chunks.")
    return rows, note


# ------------------------------------------------------------------------------------------------------------------ llama-imatrix's own words

def explain_failure(rc: int, tail) -> str:
    """llama-imatrix's error in plain words: the line that says what is wrong (`failed to load`, `inconsistent`, ...) is quoted."""
    lines = [l.strip() for l in tail if l.strip()]
    pick = next((l for l in reversed(lines) if re.search(r"inconsistent|mismatch|does not match|different|size", l, re.I) and re.search(r"count|expert|size|shape|tensor|matrices|imatrix", l, re.I)), None)
    if pick:
        return f"llama-imatrix refused to combine them: its message is \"{pick[:200]}\". The matrices were probably made from different models (another architecture, another size, a slimmed or grown model)."
    pick = next((l for l in reversed(lines) if re.search(r"failed to load|failed to read|unable to|cannot|can't|not a gguf|bad magic|error", l, re.I)), None)
    if pick:
        return f"llama-imatrix could not combine the matrices (exit code {rc}): \"{pick[:200]}\". If one of the files is not an importance matrix, take it out of the merge."
    return f"llama-imatrix failed (exit code {rc}; the log has its last lines). Update llama.cpp and try again, and check that every file is an importance matrix."


# ------------------------------------------------------------------------------------------------------------------ the jobs

def sha(path) -> str:
    return imatrix.file_sha256(path)


def merge(job: dict) -> dict:
    """Merges the inputs into <root>/<id>/. Returns the result. Raises MatrixError (plain text) or imatrix.Stopped; the folder is removed again then."""
    root, mid = Path(job["root"]), str(job["id"])
    if not mid or mid.startswith("_") or "/" in mid or "\\" in mid or ".." in mid:
        raise MatrixError("That is not a valid name for a matrix.")
    inputs_in = job.get("inputs") or []
    if len(inputs_in) < 2:
        raise MatrixError("Choose at least two matrices to merge (a matrix of the library, a file, or Unsloth's published matrix).")
    out = root / mid
    if out.exists():
        raise MatrixError(f"The folder {out} exists already. Try again in a minute.")
    im = {"stop_file": job.get("stop_file")}
    _, tool = find_tools(need_convert=False)
    notes = list(job.get("notes") or [])
    # --- resolve: the files, the published matrix is downloaded first (a plain message when that cannot be done)
    resolved = []
    for n, i in enumerate(inputs_in):
        kind = i.get("kind")
        if kind == "library":
            f = Path(i.get("file") or "")
            if not f.is_file():
                raise MatrixError(f"The saved matrix {i.get('label') or i.get('id')} has no file any more. Take it out of the merge.")
            resolved.append({"kind": "library", "id": i.get("id"), "file": f, "label": i.get("label") or i.get("id") or f.name, "model": i.get("model"), "chunks": i.get("chunks")})
        elif kind == "file":
            f = Path(i.get("file") or "")
            if not f.is_file():
                raise MatrixError(f"The matrix file {f} was not found. Choose it again.")
            if f.suffix.lower() not in (".gguf", ".dat"):
                raise MatrixError(f"{f.name} is not a matrix file: choose an imatrix.gguf or a legacy .dat file.")
            resolved.append({"kind": "file", "file": f, "label": f.name, "model": None, "chunks": None})
        elif kind == "published":
            protocol.progress("imatrix", 0.02, "Looking up Unsloth's published matrix")
            info = lookup(i.get("base") or "", job.get("cache"))
            f = download(info, im)
            resolved.append({"kind": "published", "file": f, "label": f"Unsloth's published matrix for {info['base']}", "model": None, "chunks": None, "hub": info})
        else:
            raise MatrixError("One of the things to merge is neither a saved matrix, a file nor the published matrix.")
        if imatrix.stop_requested(im):
            raise imatrix.Stopped()
    paths = [str(r["file"]) for r in resolved]
    if len({os.path.realpath(p) for p in paths}) < len(paths):
        raise MatrixError("The same matrix file is in the merge twice: it would count double. Take one out.")
    for r in resolved:
        s = imatrix.matrix_summary(r["file"])
        if s and not isinstance(r.get("chunks"), int) and isinstance(s.get("chunks"), int):
            r["chunks"] = s["chunks"]
    warnings = compat(resolved)
    rows, weight_note = weights(resolved)
    if any(r["kind"] == "published" for r in resolved):
        warnings.append("Unsloth's published matrix describes the UNMODIFIED base model, not your fine-tune: a reasonable start for a light LoRA, less so for a heavily changed model.")
    if weight_note:
        warnings.append(weight_note)
    for w in warnings:
        protocol.log("Warning: " + w)
    notes += ["Warning: " + w for w in warnings]
    root.mkdir(parents=True, exist_ok=True)
    check_space(root, int(sum(Path(p).stat().st_size for p in paths) * 1.1), "the merged matrix")
    out.mkdir()
    try:
        protocol.progress("imatrix", 0.6, "Combining the matrices")
        partial = out / "imatrix.partial.gguf"
        base = imatrix.launcher(tool) + [x for p in paths for x in ("--in-file", p)] + ["-o", partial]
        rc, tail = imatrix.run_streaming(base + ["--output-format", "gguf"], im, 1, "imatrix", (0.6, 0.95), what="Combining the matrices")
        if rc != 0 and any("output-format" in line for line in tail):          # an older llama-imatrix: its default format is the one asked for
            protocol.log("This llama-imatrix has no --output-format; trying without it.")
            partial.unlink(missing_ok=True)
            rc, tail = imatrix.run_streaming(base, im, 1, "imatrix", (0.6, 0.95), what="Combining the matrices")
        if rc != 0 or not partial.is_file() or partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
            raise MatrixError(explain_failure(rc, tail))
        target = out / "imatrix.gguf"
        os.replace(partial, target)
        summary = imatrix.matrix_summary(target) or {}
        parents = []
        for r in resolved:
            p = {"kind": r["kind"], "label": r["label"], "id": r.get("id"), "path": str(r["file"]), "sha256": sha(r["file"]), "size": Path(r["file"]).stat().st_size, "chunks": r.get("chunks")}
            if r["kind"] == "published":
                p.update(repo=r["hub"]["repo"], file=r["hub"]["file"], base=r["hub"]["base"], url=r["hub"]["url"])
            parents.append(p)
        models = {r.get("model") for r in resolved if r["kind"] == "library"}
        chunks = [r.get("chunks") for r in resolved]
        total_chunks = sum(chunks) if all(isinstance(c, int) for c in chunks) else None
        info = {"format": imatrix.FORMAT, "id": mid, "label": str(job.get("label") or "").strip()[:80], "model": models.pop() if len(models) == 1 and None not in models else "",
                "source": "a merge of " + ", ".join(r["label"] for r in resolved), "source_kind": "merged", "via": "llama.cpp", "parents": parents, "merge_weights": rows,
                "chunks": total_chunks, "ctx": None, "general_used": None, "nextn": None, "nextn_asked": False, "arch": summary.get("arch"), "layers": summary.get("layers"),
                "entries": summary.get("entries"), "size": target.stat().st_size, "has_dat": False, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "llama_cpp": imatrix.tool_version(tool), "base_repo": job.get("base_repo"), "calibration": "the text the parents were measured on", "calibration_kind": "merged",
                "notes": notes, "statistics": None}
        (out / "imatrix.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        protocol.progress("imatrix", 1.0, "Merged matrix ready")
        return {"id": mid, "dir": str(out), "file": str(target), "dat": None, "size_mb": round(target.stat().st_size / 1048576, 2), "info": info, "notes": notes, "warnings": warnings,
                "weights": rows, "statistics": None}
    except BaseException as exc:
        shutil.rmtree(out, ignore_errors=True)
        if isinstance(exc, (MatrixError, imatrix.Stopped, KeyboardInterrupt)):
            raise
        raise MatrixError(plain(exc))


def run_published(job: dict) -> int:
    protocol.reserve_stdout()
    protocol.result(lookup(job.get("base") or "", job.get("cache")))
    return 0


def run(job: dict) -> int:
    protocol.reserve_stdout()
    protocol.result(merge(job))
    return 0
