"""Merging importance matrices (imatrixmerge.py): the repo and file name of Unsloth's published matrix as Unsloth's code derives them, the Hub lookup and download against a
stand-in Hub (a local HTTP server: offline, not found, no file, gated, redirects, the token), the merge command line, the registration of a library entry, the warnings, every
failure message. A stand-in llama-imatrix (a small Python script, the same on Windows and Linux); no llama.cpp, no network, no torch."""
import hashlib
import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from test_imatrix import flag

from ndforge_worker import __main__ as entry
from ndforge_worker import exportjob, imatrix, imatrixjob, imatrixmerge, protocol

IMATRIX = '''import json, os, sys, time
a = sys.argv[1:]
if a == ["--version"]:
    print("version: 9999 (stub1234)")
    sys.exit(0)
open(os.environ["STUB_LOG"], "a").write(json.dumps({"tool": "imatrix", "args": a}) + "\\n")
mode = os.environ.get("MERGE_MODE", "")
ins = [a[i + 1] for i, x in enumerate(a) if x == "--in-file"]
if mode == "old" and "--output-format" in a:
    print("error: invalid argument: --output-format")
    sys.exit(1)
if mode == "mismatch":
    print("load_imatrix: inconsistent expert count for blk.3.ffn_up_exps.weight (4 vs 8)")
    sys.exit(1)
if mode == "unreadable":
    print("main : failed to load " + ins[0])
    sys.exit(1)
if mode == "silent":
    sys.exit(1)
if mode == "empty":
    open(a[a.index("-o") + 1], "wb").write(b"")
    sys.exit(0)
if mode == "hang":
    open(os.environ["STUB_STARTED"], "w").write("x")
    for _ in range(600):
        time.sleep(0.05)
    sys.exit(0)
open(a[a.index("-o") + 1], "wb").write(open(ins[0], "rb").read())
'''


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def matrix_file(path, layers=2, arch="qwen35", chunks=64, extra=()):
    """a tiny hand-built GGUF matrix header (version 3) with `layers` blocks"""
    names = [f"blk.{i}.attn_k.weight.{x}" for i in range(layers) for x in ("in_sum2", "counts")] + list(extra)
    kvs = [("general.type", 8, s("imatrix")), ("imatrix.chunk_count", 4, struct.pack("<I", chunks))]
    if arch:
        kvs.append(("general.architecture", 8, s(arch)))
    out = b"GGUF" + struct.pack("<IQQ", 3, len(names), len(kvs))
    for k, t, v in kvs:
        out += s(k) + struct.pack("<I", t) + v
    for n in names:
        out += s(n) + struct.pack("<I", 1) + struct.pack("<Q", 4) + struct.pack("<I", 0) + struct.pack("<Q", 0)
    Path(path).write_bytes(out)
    return str(path)


# ------------------------------------------------------------------------------------------------------------------ a stand-in Hub

class Hub:
    """A local HTTP server that answers like the Hub: /api/models/<repo>?blobs=true, /<repo>/resolve/main/<file> (optionally through a redirect to a second server, the way the
    real download goes through a CDN) and records the Authorization header of every request."""

    def __init__(self):
        self.repos, self.seen, self.status = {}, [], {}
        me = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body=b"", ctype="application/octet-stream", headers=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                me.seen.append((self.server.server_port, self.path, self.headers.get("Authorization")))
                path = self.path.split("?")[0]
                if path in me.status:
                    return self._send(me.status[path], b"{}")
                if path.startswith("/api/models/"):
                    repo = path[len("/api/models/"):]
                    r = me.repos.get(repo)
                    if r is None:
                        return self._send(404, b'{"error": "Repository not found"}', "application/json")
                    if r.get("token") and self.headers.get("Authorization") != "Bearer " + r["token"] and not r.get("list_open"):
                        return self._send(401, b"{}")
                    body = {"id": repo, "gated": r.get("gated", False), "siblings": [{"rfilename": n, "size": len(b)} for n, b in r["files"].items()] + [{"rfilename": "README.md", "size": 5}]}
                    return self._send(200, json.dumps(body).encode(), "application/json")
                for repo, r in me.repos.items():
                    pre = f"/{repo}/resolve/main/"
                    if path.startswith(pre):
                        name = path[len(pre):]
                        if r.get("token") and self.headers.get("Authorization") != "Bearer " + r["token"]:
                            return self._send(401, b"{}")
                        if name not in r["files"]:
                            return self._send(404)
                        if me.cdn is not None:
                            return self._send(302, headers={"Location": f"http://127.0.0.1:{me.cdn.port}/cdn/{repo}/{name}"})
                        return self._send(200, r["files"][name][: r.get("cut", None)] if r.get("cut") else r["files"][name])
                if path.startswith("/cdn/"):
                    repo, name = path[len("/cdn/"):].rsplit("/", 1)
                    return self._send(200, me.repos[repo]["files"][name])
                self._send(404)

        self.cdn = None
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_port
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def hub(monkeypatch):
    h = Hub()
    monkeypatch.setenv("HF_ENDPOINT", h.url)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    yield h
    h.close()
    if h.cdn:
        h.cdn.close()


@pytest.fixture
def mg(tmp_path, monkeypatch, hub):
    tool = tmp_path / "llama-imatrix.py"
    tool.write_text(IMATRIX, encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("STUB_LOG", str(log))
    monkeypatch.setenv("STUB_STARTED", str(tmp_path / "started"))
    monkeypatch.delenv("MERGE_MODE", raising=False)
    monkeypatch.setattr(imatrixmerge, "find_tools", lambda need_convert: (None, tool))
    monkeypatch.setattr(imatrixjob, "free_bytes", lambda p: 10 ** 12)
    events = []
    monkeypatch.setattr(protocol, "emit", lambda kind, **f: events.append({"type": kind, **f}))

    class W:
        pass

    w = W()
    w.root, w.hub, w.events, w.tool = tmp_path, hub, events, tool
    w.lib, w.cache = tmp_path / "imatrix", tmp_path / "imatrix_downloads"
    w.a = matrix_file(tmp_path / "a.gguf", chunks=64)
    w.b = matrix_file(tmp_path / "b.gguf", chunks=64)
    w.pub_bytes = matrix_file(tmp_path / "pub-src.gguf", chunks=1000)
    hub.repos["unsloth/Qwen3.8-27B-GGUF"] = {"files": {"imatrix_unsloth.gguf": Path(w.pub_bytes).read_bytes(), "README.md": b"hello"}}

    def calls():
        return [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []

    def job(**over):
        j = {"job": "imatrix_merge", "root": str(w.lib), "id": "20261115-1030-merged", "label": "my merge", "stop_file": str(tmp_path / "STOP"), "cache": str(w.cache), "notes": [],
             "inputs": [{"kind": "library", "id": "x1", "file": w.a, "label": "first", "model": "run1", "chunks": 64}, {"kind": "library", "id": "x2", "file": w.b, "label": "second", "model": "run1", "chunks": 64}]}
        j.update(over)
        return j

    w.calls, w.job = calls, job
    w.run = lambda **kw: imatrixmerge.merge(job(**kw))
    w.folder = w.lib / "20261115-1030-merged"
    return w


def err(mg_, exc=imatrixjob.MatrixError, **kw):
    with pytest.raises(exc) as e:
        mg_.run(**kw)
    return str(e.value)


# ------------------------------------------------------------------------------------------------------------------ the repo, as Unsloth's code derives it

@pytest.mark.parametrize("base,repo", [("Qwen/Qwen3.8-27B", "unsloth/Qwen3.8-27B-GGUF"), ("unsloth/Qwen3.8-27B", "unsloth/Qwen3.8-27B-GGUF"), ("Qwen3.8-27B", "unsloth/Qwen3.8-27B-GGUF"),
                                       ("unsloth/Qwen3.8-27B-unsloth-bnb-4bit", "unsloth/Qwen3.8-27B-GGUF"), ("unsloth/llama-3-8b-bnb-4bit", "unsloth/llama-3-8b-GGUF"),
                                       ("someone/Model-GGUF", "someone/Model-GGUF"), ("  Qwen/Qwen3.5-9B ", "unsloth/Qwen3.5-9B-GGUF")])
def test_the_repo_is_unsloth_slash_the_models_name_dash_gguf(base, repo):
    assert imatrixmerge.published_repo(base) == repo


def test_a_folder_or_an_empty_name_has_no_published_matrix(tmp_path):
    assert imatrixmerge.published_repo(str(tmp_path)) is None and imatrixmerge.published_repo("") is None and imatrixmerge.published_repo(None) is None


def test_the_file_names_are_those_of_unsloths_zoo_in_its_order():
    assert imatrixmerge.UPSTREAM_NAMES == ("imatrix_unsloth.dat", "imatrix_unsloth.gguf_file", "imatrix_unsloth.gguf")


# ------------------------------------------------------------------------------------------------------------------ the lookup and the download

def test_the_lookup_names_repo_file_size_and_source(mg):
    r = imatrixmerge.lookup("Qwen/Qwen3.8-27B", mg.cache)
    size = Path(mg.pub_bytes).stat().st_size
    assert (r["repo"], r["file"], r["local_name"], r["size"], r["gated"], r["cached"]) == ("unsloth/Qwen3.8-27B-GGUF", "imatrix_unsloth.gguf", "imatrix_unsloth.gguf", size, False, False)
    assert r["page"] == f"{mg.hub.url}/unsloth/Qwen3.8-27B-GGUF" and r["url"].endswith("/unsloth/Qwen3.8-27B-GGUF/resolve/main/imatrix_unsloth.gguf") and r["base"] == "Qwen/Qwen3.8-27B"
    assert [p for _, p, _ in mg.hub.seen] == ["/api/models/unsloth/Qwen3.8-27B-GGUF?blobs=true"], "only the listing, nothing downloaded"


def test_the_gguf_file_name_is_renamed_to_gguf_locally_and_the_dat_comes_first(mg):
    mg.hub.repos["unsloth/A-GGUF"] = {"files": {"imatrix_unsloth.gguf_file": b"GGUFx"}}
    r = imatrixmerge.lookup("org/A", mg.cache)
    assert (r["file"], r["local_name"]) == ("imatrix_unsloth.gguf_file", "imatrix_unsloth.gguf") and r["cache_file"].endswith("unsloth--A-GGUF" + __import__("os").sep + "imatrix_unsloth.gguf")
    mg.hub.repos["unsloth/B-GGUF"] = {"files": {"imatrix_unsloth.gguf": b"GGUFx", "imatrix_unsloth.dat": b"DAT"}}
    assert imatrixmerge.lookup("org/B", mg.cache)["file"] == "imatrix_unsloth.dat", "the order of Unsloth's own code"


def test_the_download_goes_to_the_cache_with_progress_and_is_not_repeated(mg):
    info = imatrixmerge.lookup("Qwen/Qwen3.8-27B", mg.cache)
    f = imatrixmerge.download(info, {"stop_file": None})
    assert f.read_bytes() == Path(mg.pub_bytes).read_bytes() and f.parent == mg.cache / "unsloth--Qwen3.8-27B-GGUF" and not list(f.parent.glob("*.part"))
    assert any(e["type"] == "progress" and "Downloading Unsloth's published matrix" in e["text"] for e in mg.events)
    n = len(mg.hub.seen)
    info = imatrixmerge.lookup("Qwen/Qwen3.8-27B", mg.cache)
    assert info["cached"] is True
    imatrixmerge.download(info, {"stop_file": None})
    assert [p for _, p, _ in mg.hub.seen[n:]] == ["/api/models/unsloth/Qwen3.8-27B-GGUF?blobs=true"], "the second time nothing is downloaded"
    f.write_bytes(b"short")
    assert imatrixmerge.lookup("Qwen/Qwen3.8-27B", mg.cache)["cached"] is False, "a cached file of another size is downloaded again"


def test_the_token_goes_only_to_the_hub_never_to_the_server_it_redirects_to(mg, monkeypatch):
    cdn = Hub()
    cdn.repos = mg.hub.repos
    mg.hub.cdn = cdn
    monkeypatch.setenv("HF_TOKEN", "hf_secrettoken1234567890abc")
    info = imatrixmerge.lookup("Qwen/Qwen3.8-27B", mg.cache)
    f = imatrixmerge.download(info, {"stop_file": None})
    assert f.is_file()
    hub_calls = [(p, a) for port, p, a in mg.hub.seen if port == mg.hub.port]
    assert all(a == "Bearer hf_secrettoken1234567890abc" for _, a in hub_calls) and len(hub_calls) == 2
    cdn_calls = [(p, a) for port, p, a in cdn.seen if port == cdn.port]
    assert cdn_calls and all(a is None for _, a in cdn_calls) and cdn_calls[0][0].startswith("/cdn/"), "no token for the download server"
    assert "hf_secrettoken" not in json.dumps(mg.events)


def test_a_cut_off_download_is_removed_and_said(mg):
    mg.hub.repos["unsloth/Qwen3.8-27B-GGUF"]["cut"] = 10        # the server says 10 bytes but the file is larger: the listing says otherwise
    info = imatrixmerge.lookup("Qwen/Qwen3.8-27B", mg.cache)
    with pytest.raises(imatrixmerge.HubError) as e:
        imatrixmerge.download(info, {"stop_file": None})
    assert e.value.kind == "offline" and "cut off" in str(e.value)
    assert not list(mg.cache.rglob("*.part")) and not list(mg.cache.rglob("imatrix_unsloth.gguf"))


def test_stop_ends_the_download_at_once_and_leaves_no_part_file(mg):
    info = imatrixmerge.lookup("Qwen/Qwen3.8-27B", mg.cache)
    (mg.root / "STOP").write_text("")
    with pytest.raises(imatrix.Stopped):
        imatrixmerge.download(info, {"stop_file": str(mg.root / "STOP")})
    assert not list(mg.cache.rglob("*.part")) and not list(mg.cache.rglob("imatrix_unsloth.gguf"))


def test_every_hub_problem_has_its_own_plain_message(mg, monkeypatch, tmp_path):
    def kind(base, **env):
        with pytest.raises(imatrixmerge.HubError) as e:
            imatrixmerge.lookup(base, mg.cache)
        return e.value.kind, str(e.value)
    k, m = kind("Qwen/Unknown-Model")
    assert k == "not_found" and "no published matrix for Qwen/Unknown-Model" in m and "unsloth/Unknown-Model-GGUF" in m and "the way Unsloth's own code does it" in m and "measure a matrix yourself" in m
    mg.hub.repos["unsloth/Empty-GGUF"] = {"files": {"model.gguf": b"x"}}
    k, m = kind("org/Empty")
    assert k == "no_file" and "holds none of the files imatrix_unsloth.dat, imatrix_unsloth.gguf_file, imatrix_unsloth.gguf" in m
    mg.hub.repos["unsloth/Gated-GGUF"] = {"files": {"imatrix_unsloth.gguf": b"x"}, "token": "hf_rightrighttoken1234567", "list_open": True}
    info = imatrixmerge.lookup("org/Gated", mg.cache)
    with pytest.raises(imatrixmerge.HubError) as e:
        imatrixmerge.download(info, {"stop_file": None})
    assert e.value.kind == "gated" and "accept its terms" in str(e.value) and "token" in str(e.value) and "HTTP 401" in str(e.value)
    k, m = kind(str(tmp_path))
    assert k == "local" and "folder on this computer" in m and "original model" in m
    mg.hub.status["/api/models/unsloth/Down-GGUF"] = 500
    k, m = kind("org/Down")
    assert k == "http" and "HTTP 500" in m
    mg.hub.close()
    k, m = kind("Qwen/Qwen3.8-27B")
    assert k == "offline" and "could not be reached" in m and "internet connection" in m and "everything else in this program works offline" in m


def test_a_private_repo_without_a_token_reads_as_not_found(mg):
    mg.hub.repos["unsloth/Private-GGUF"] = {"files": {"imatrix_unsloth.gguf": b"x"}, "token": "hf_abcdefabcdefabcdef123456"}
    with pytest.raises(imatrixmerge.HubError) as e:
        imatrixmerge.lookup("org/Private", mg.cache)
    assert e.value.kind == "not_found" and "or it is private" in str(e.value)


# ------------------------------------------------------------------------------------------------------------------ merging

def test_two_library_matrices_are_merged_with_the_exact_command_and_registered(mg):
    r = mg.run()
    (c,) = mg.calls()
    a = c["args"]
    assert a[:4] == ["--in-file", mg.a, "--in-file", mg.b] and flag(a, "-o").endswith("imatrix.partial.gguf") and flag(a, "--output-format") == "gguf"
    assert "-m" not in a and "-f" not in a and "--chunks" not in a, "no model and no text are needed to combine matrices"
    assert sorted(p.name for p in mg.folder.iterdir()) == ["imatrix.gguf", "imatrix.json"] and (mg.folder / "imatrix.gguf").read_bytes() == Path(mg.a).read_bytes()
    info = json.loads((mg.folder / "imatrix.json").read_text(encoding="utf-8"))
    assert (info["source_kind"], info["via"], info["label"], info["id"], info["model"], info["chunks"], info["has_dat"], info["format"]) == ("merged", "llama.cpp", "my merge", "20261115-1030-merged", "run1", 128, False, 1)
    assert info["source"] == "a merge of first, second" and info["llama_cpp"] == "version: 9999 (stub1234)" and info["arch"] == "qwen35" and info["layers"] == 2 and info["size"] == (mg.folder / "imatrix.gguf").stat().st_size
    p1, p2 = info["parents"]
    assert (p1["kind"], p1["id"], p1["path"], p1["label"], p1["chunks"]) == ("library", "x1", mg.a, "first", 64) and p1["sha256"] == hashlib.sha256(Path(mg.a).read_bytes()).hexdigest() and p2["sha256"] == hashlib.sha256(Path(mg.b).read_bytes()).hexdigest()
    assert [w["share"] for w in info["merge_weights"]] == [0.5, 0.5]
    assert r["id"] == "20261115-1030-merged" and Path(r["file"]) == mg.folder / "imatrix.gguf" and r["warnings"] == [] and r["weights"] == info["merge_weights"]


def test_progress_never_goes_back_and_ends_at_one(mg):
    mg.run()
    prog = [e for e in mg.events if e["type"] == "progress"]
    assert {e["step"] for e in prog} == {"imatrix"} and prog[-1]["value"] == 1.0 and prog[-1]["text"] == "Merged matrix ready"
    assert all(a["value"] <= b["value"] for a, b in zip(prog, prog[1:])), [e["value"] for e in prog]
    assert any(e["text"] == "Combining the matrices" for e in prog)


def test_the_published_matrix_is_downloaded_merged_and_recorded_with_its_source(mg):
    r = mg.run(inputs=[{"kind": "library", "id": "x1", "file": mg.a, "label": "mine", "model": "run1", "chunks": 64}, {"kind": "published", "base": "Qwen/Qwen3.8-27B"}])
    info = r["info"]
    pub = info["parents"][1]
    assert (pub["kind"], pub["repo"], pub["file"], pub["base"]) == ("published", "unsloth/Qwen3.8-27B-GGUF", "imatrix_unsloth.gguf", "Qwen/Qwen3.8-27B") and pub["url"].startswith(mg.hub.url) and pub["chunks"] == 1000
    assert pub["sha256"] == hashlib.sha256(Path(mg.pub_bytes).read_bytes()).hexdigest() and pub["label"] == "Unsloth's published matrix for Qwen/Qwen3.8-27B" and info["chunks"] == 1064
    assert "Warning: Unsloth's published matrix describes the UNMODIFIED base model, not your fine-tune" in " ".join(info["notes"])
    assert any("carries about 94 % of the merged statistics" in w and "make it with more chunks" in w for w in r["warnings"]), "the larger matrix outweighs the smaller: said plainly"
    assert [w["share"] for w in info["merge_weights"]] == [0.06, 0.94] and info["model"] == "run1"
    assert (mg.cache / "unsloth--Qwen3.8-27B-GGUF" / "imatrix_unsloth.gguf").is_file()
    a = mg.calls()[0]["args"]
    assert a[0] == "--in-file" and a[1] == mg.a and a[3].endswith("imatrix_unsloth.gguf")


def test_a_matrix_file_and_the_library_can_be_mixed_and_three_are_merged(mg):
    third = matrix_file(mg.root / "c.gguf", chunks=64)
    r = mg.run(inputs=mg.job()["inputs"] + [{"kind": "file", "file": third}])
    a = mg.calls()[0]["args"]
    assert a.count("--in-file") == 3 and a[5] == third
    assert [p["kind"] for p in r["info"]["parents"]] == ["library", "library", "file"] and r["info"]["parents"][2]["label"] == "c.gguf" and r["info"]["chunks"] == 128 + 64
    assert r["info"]["model"] == "run1"


def test_models_that_differ_leave_the_model_field_empty(mg):
    r = mg.run(inputs=[{"kind": "library", "file": mg.a, "label": "a", "model": "run1", "chunks": 64}, {"kind": "library", "file": mg.b, "label": "b", "model": "run2", "chunks": 64}])
    assert r["info"]["model"] == ""


def test_a_merged_entry_is_used_by_an_export_like_any_other_entry(mg, tmp_path):
    mg.run()
    lib_file = mg.folder / "imatrix.gguf"
    out = tmp_path / "export"
    out.mkdir()
    merged = tmp_path / "merged-model"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    report = exportjob.new_report()
    cfg = {"adapter": str(tmp_path / "adapter"), "base": {"repo": "org/base", "family": "qwen3_5"}, "imatrix": {}}
    got = imatrix.use_saved_file(cfg, merged, out, None, {"use_file": str(lib_file), "use_id": "20261115-1030-merged"}, report)
    assert got == out / "imatrix.gguf" and got.read_bytes() == lib_file.read_bytes() and report["reused"] is True and report["from_library"] == "20261115-1030-merged"
    assert json.loads((out / "imatrix.json").read_text(encoding="utf-8"))["source_kind"] == "merged"


# ------------------------------------------------------------------------------------------------------------------ compatibility: warnings, never refusals

def test_different_architectures_layers_and_tensor_sets_are_yellow_warnings_and_the_merge_still_runs(mg):
    other = matrix_file(mg.root / "o.gguf", layers=3, arch="llama", extra=["blk.2.ffn_up.weight.in_sum2"])
    r = mg.run(inputs=[{"kind": "file", "file": mg.a}, {"kind": "file", "file": other}])
    w = " ".join(r["warnings"])
    assert 'a.gguf is for the architecture "qwen35" and o.gguf for "llama". They will most likely not fit together' in w
    assert "a.gguf has data for 2 layers and o.gguf for 3" in w and "tensors have data in only one of a.gguf and o.gguf" in w
    assert all(n.startswith("Warning: ") for n in r["notes"] if "architecture" in n) and mg.folder.is_dir() and len(mg.calls()) == 1
    assert [n for n in r["info"]["notes"] if n.startswith("Warning: ")], "the warnings are kept with the entry"


def test_a_legacy_dat_file_cannot_be_checked_and_says_so(mg):
    dat = mg.root / "old.dat"
    dat.write_bytes(b"LEGACY")
    r = mg.run(inputs=[{"kind": "file", "file": mg.a}, {"kind": "file", "file": str(dat)}])
    assert any("old.dat could not be read here (a legacy .dat file" in w for w in r["warnings"]) and r["info"]["parents"][1]["chunks"] is None
    assert [w["share"] for w in r["info"]["merge_weights"]] == [None, None] and r["info"]["chunks"] is None


def test_matching_matrices_have_no_warnings(mg):
    assert imatrixmerge.compat([{"label": "a", "file": mg.a}, {"label": "b", "file": mg.b}]) == []
    rows, note = imatrixmerge.weights([{"label": "a", "chunks": 64}, {"label": "b", "chunks": 64}])
    assert note == "" and [r["share"] for r in rows] == [0.5, 0.5]
    rows, note = imatrixmerge.weights([{"label": "a", "chunks": 10}, {"label": "b", "chunks": None}])
    assert note == "" and [r["share"] for r in rows] == [None, None]


# ------------------------------------------------------------------------------------------------------------------ every failure

def test_fewer_than_two_things_to_merge_is_refused_plainly(mg):
    assert "at least two matrices" in err(mg, inputs=mg.job()["inputs"][:1])
    assert "at least two matrices" in err(mg, inputs=[])
    assert not mg.folder.exists() and mg.calls() == []


def test_missing_wrong_or_doubled_inputs_are_refused_before_llama_imatrix_runs(mg):
    ins = mg.job()["inputs"]
    assert "has no file any more" in err(mg, inputs=[ins[0], {**ins[1], "file": str(mg.root / "gone.gguf")}])
    assert "was not found" in err(mg, inputs=[ins[0], {"kind": "file", "file": str(mg.root / "gone.gguf")}])
    (mg.root / "x.txt").write_text("x")
    assert "is not a matrix file" in err(mg, inputs=[ins[0], {"kind": "file", "file": str(mg.root / "x.txt")}])
    assert "in the merge twice" in err(mg, inputs=[ins[0], {"kind": "file", "file": mg.a}])
    assert "neither a saved matrix" in err(mg, inputs=[ins[0], {"kind": "nonsense"}])
    assert "not a valid name" in err(mg, id="../x")
    assert mg.calls() == [] and not mg.folder.exists()


def test_an_existing_folder_is_not_overwritten(mg):
    mg.run()
    assert "exists already" in err(mg)


def test_a_published_matrix_that_cannot_be_had_ends_the_merge_with_the_hubs_words_and_leaves_nothing(mg):
    ins = [mg.job()["inputs"][0], {"kind": "published", "base": "Qwen/Unknown-Model"}]
    msg = err(mg, imatrixmerge.HubError, inputs=ins)
    assert "no published matrix for Qwen/Unknown-Model" in msg and not mg.folder.exists() and mg.calls() == []
    mg.hub.close()
    assert "could not be reached" in err(mg, imatrixmerge.HubError, inputs=[mg.job()["inputs"][0], {"kind": "published", "base": "Qwen/Qwen3.8-27B"}])
    assert "folder on this computer" in err(mg, imatrixmerge.HubError, inputs=[mg.job()["inputs"][0], {"kind": "published", "base": str(mg.root)}])


@pytest.mark.parametrize("mode,words", [("mismatch", "refused to combine them"), ("unreadable", "could not combine the matrices"), ("silent", "failed (exit code 1"), ("empty", "failed (exit code 0")])
def test_llama_imatrixs_own_error_is_passed_on_in_plain_words(mg, monkeypatch, mode, words):
    monkeypatch.setenv("MERGE_MODE", mode)
    msg = err(mg)
    assert words in msg
    if mode == "mismatch":
        assert 'inconsistent expert count for blk.3.ffn_up_exps.weight (4 vs 8)' in msg and "different models" in msg
    if mode == "unreadable":
        assert "failed to load" in msg and "not an importance matrix" in msg
    assert not mg.folder.exists(), "a failed merge leaves no half-made entry"


def test_an_older_llama_imatrix_without_output_format_is_tried_again_without_it(mg, monkeypatch):
    monkeypatch.setenv("MERGE_MODE", "old")
    mg.run()
    first, second = [c["args"] for c in mg.calls()]
    assert "--output-format" in first and "--output-format" not in second and mg.folder.is_dir()


def test_no_llama_cpp_says_how_to_install_it(mg, monkeypatch):
    monkeypatch.undo()          # the real find_tools, with nothing to find
    monkeypatch.setenv("LLAMA_CPP_DIR", str(mg.root / "nowhere"))
    monkeypatch.setenv("PATH", str(mg.root / "nowhere"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: mg.root / "nohome"))
    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: (None, None))
    monkeypatch.setattr(imatrix, "find_llama_imatrix", lambda q: None)
    msg = err(mg)
    assert "llama.cpp was not found" in msg and "LLAMA_CPP_DIR" in msg and not mg.folder.exists()


def test_stop_kills_llama_imatrix_and_removes_the_half_made_entry(mg, monkeypatch):
    monkeypatch.setenv("MERGE_MODE", "hang")

    def waiter():
        for _ in range(400):
            if (mg.root / "started").exists():
                break
            time.sleep(0.05)
        (mg.root / "STOP").write_text("")
    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    t0 = time.time()
    with pytest.raises(imatrix.Stopped):
        mg.run()
    t.join()
    assert time.time() - t0 < 15 and not mg.folder.exists()


def test_not_enough_disk_is_said_with_numbers(mg, monkeypatch):
    monkeypatch.setattr(imatrixjob, "free_bytes", lambda p: 10)
    assert "not enough free disk space for the merged matrix" in err(mg) and not mg.folder.exists()


# ------------------------------------------------------------------------------------------------------------------ the worker entry

def test_the_worker_entry_runs_a_lookup_and_a_merge_and_reports_failures_in_plain_words(mg, monkeypatch):
    jobfile = mg.root / "job.json"
    jobfile.write_text(json.dumps({"job": "imatrix_published", "base": "Qwen/Qwen3.8-27B", "cache": str(mg.cache)}), encoding="utf-8")
    assert entry.main(["x", str(jobfile)]) == 0
    res = [e for e in mg.events if e["type"] == "result"]
    assert res[0]["data"]["repo"] == "unsloth/Qwen3.8-27B-GGUF" and res[0]["data"]["size"] == Path(mg.pub_bytes).stat().st_size
    jobfile.write_text(json.dumps(mg.job()), encoding="utf-8")
    assert entry.main(["x", str(jobfile)]) == 0 and (mg.folder / "imatrix.gguf").is_file()
    jobfile.write_text(json.dumps({"job": "imatrix_published", "base": "Qwen/Unknown-Model", "cache": str(mg.cache)}), encoding="utf-8")
    assert entry.main(["x", str(jobfile)]) == 1
    assert [e for e in mg.events if e["type"] == "error"][0]["code"] == "imatrix_failed" and "no published matrix" in [e for e in mg.events if e["type"] == "error"][0]["message"]
    monkeypatch.setattr(imatrixmerge, "merge", lambda job: (_ for _ in ()).throw(imatrix.Stopped()))
    jobfile.write_text(json.dumps(mg.job()), encoding="utf-8")
    assert entry.main(["x", str(jobfile)]) == 1
    assert [e for e in mg.events if e["type"] == "error"][-1] == {"type": "error", "code": "imatrix_stopped", "message": "Cancelled."}
