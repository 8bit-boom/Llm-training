"""The vLLM teacher cache, worker side, with a fake container (no Docker, no vLLM, no torch): teacher choice, identity, staging, resume, stop, fallback, command templates."""
import contextlib
import hashlib
import io
import re
import subprocess
import json
import shutil
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from ndforge_worker import cachewrite, distilljob as dj, safetensors_io, vllm_cache as vc, vllmcache, vllmengine as ve  # noqa: E402
from test_cache_identity import META, cfg_for, make_model, rows_of, setup  # noqa: E402
from test_vllm_cache import FakeScorer, expected, make_rows  # noqa: E402

K = 4


# ------------------------------------------------------------------------------------------------------------------------------ the teacher choice

def sel(origin, **t):
    return vllmcache.select_teacher(t, origin, "/models/origin")


def test_the_default_pair_is_used_only_for_its_own_original():
    s = sel("Qwen/Qwen3.8-27B")
    assert s["route"] == "prequant" and s["model"] == "cyankiwi/Qwen3.8-27B-AWQ-INT4" and s["quant"] == "awq:cyankiwi/Qwen3.8-27B-AWQ-INT4@unpinned"
    assert sel("qwen/qwen3.8-27b/")["route"] == "prequant"          # spelling of the id does not matter
    for other in ("Someone/Qwen3.8-27B-Ultimate-Heretic", "Qwen/Qwen3.8-14B", "C:\\models\\heretic"):
        s = sel(other)
        assert s["route"] == "bnb" and s["quant"] == "bnb-nf4" and s["model"] == "/models/origin", other


def test_prequant_for_another_original_is_refused_not_swapped():
    with pytest.raises(vllmcache.VllmCacheError, match="no confirmed quantized copy"):
        sel("Someone/Finetune", route="prequant")
    with pytest.raises(vllmcache.VllmCacheError, match="not a confirmed quantized copy"):
        sel("Someone/Finetune", quantized_repo="cyankiwi/Qwen3.8-27B-AWQ-INT4")


def test_the_owner_can_confirm_a_quantized_copy_per_original():
    s = sel("Someone/Finetune", confirmed={"Someone/Finetune": "Someone/Finetune-AWQ"}, quantized_repo="Someone/Finetune-AWQ", quantized_rev="abc123")
    assert s["route"] == "prequant" and s["quant"] == "awq:Someone/Finetune-AWQ@abc123"
    with pytest.raises(vllmcache.VllmCacheError):                    # a confirmation for one original does not cover another
        sel("Other/Model", confirmed={"Someone/Finetune": "Someone/Finetune-AWQ"}, quantized_repo="Someone/Finetune-AWQ")
    s = sel("Qwen/Qwen3.8-27B", route="bnb")
    assert s["route"] == "bnb", "the owner can always choose the in-flight route"
    assert sel("My/Own", pairs={"My/Own": "My/Own-AWQ"})["route"] == "prequant"
    with pytest.raises(vllmcache.VllmCacheError, match="Unknown"):
        sel("x", route="fp8")


# ------------------------------------------------------------------------------------------------------------------------------ the identity

def job_for(tmp_path, **eng):
    return {"engine": {"kind": "vllm", "teacher": {}, "container": {"kind": "docker", "image": "vllm/vllm-openai:v0.34.0"}, **eng}}


def test_identity_has_engine_quant_and_origin(tmp_path):
    origin = make_model(tmp_path / "origin")
    i = vllmcache.identity(job_for(tmp_path), "Someone/Finetune", origin)
    assert i == {"engine": "vllm-0.34.0", "teacher_quant": "bnb-nf4", "teacher_origin": {"repo": "Someone/Finetune", "weights": safetensors_io.weights_manifest(origin)}}
    j = vllmcache.identity(job_for(tmp_path), "Qwen/Qwen3.8-27B", origin)
    assert j["teacher_quant"].startswith("awq:cyankiwi/")


def test_a_vllm_cache_never_shares_a_store_with_the_standard_teacher(tmp_path):
    teacher, student, cfg = setup(tmp_path)
    vid = vllmcache.identity(job_for(tmp_path), "org/T", teacher)
    plain = dj.shared_store_id(cfg, teacher, student, 8, 128, 3407)
    with_engine = dj.shared_store_id(cfg, teacher, student, 8, 128, 3407, 2, vid)
    other_quant = dj.shared_store_id(cfg, teacher, student, 8, 128, 3407, 2, {**vid, "teacher_quant": "awq:x/y@1"})
    other_engine = dj.shared_store_id(cfg, teacher, student, 8, 128, 3407, 2, {**vid, "engine": "vllm-0.35.0"})
    assert len({plain, with_engine, other_quant, other_engine}) == 4
    assert dj.shared_store_id(cfg, teacher, student, 8, 128, 3407, 2, dict(vid)) == with_engine


def test_the_standard_store_id_is_what_it_always_was(tmp_path):
    teacher, student, cfg = setup(tmp_path)
    data = cfg["data"]
    ident = {"v": 2, "teacher": str(Path(teacher).resolve()), "teacher_config": dj._file_sha256(Path(teacher) / "config.json"), "tokenizer": dj.tokenizer_hash(student), "k": 8, "max_len": 128,
             "load": "auto", "general": dj.generaltext.general_key(None, 0, 3407), "seed": 3407, "family": "qwen3_5", "train": dj._file_sha256(data["train"]),
             "teacher_weights": safetensors_io.weights_manifest(teacher)}
    ident["general"].pop("share", None)
    assert dj.shared_store_id(cfg, teacher, student, 8, 128, 3407) == hashlib.sha256(json.dumps(ident, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def test_cache_meta_without_an_engine_has_no_engine_keys(tmp_path):
    teacher, student, cfg = setup(tmp_path)
    stats = {"project_rows": 1, "general_rows": 0, "general_source": None}
    plain = dj.cache_meta(cfg, rows_of(1), stats, student, 8, 128, teacher)
    assert not {"engine", "teacher_quant", "teacher_origin"} & set(plain)
    vid = vllmcache.identity(job_for(tmp_path), "org/T", teacher)
    v = dj.cache_meta(cfg, rows_of(1), stats, student, 8, 128, teacher, vid)
    assert {k: v[k] for k in vid} == vid
    assert not dj.cache_matches({**v, "format": 1}, plain) and not dj.cache_matches({**plain, "format": 1}, v)
    assert dj.cache_matches({**v, "format": 1}, dict(v))


def fake(n, k=8):
    return (np.zeros((n, k), dtype=np.uint32), np.zeros((n, k), dtype=np.float16), np.zeros(n, dtype=np.float32))


def test_store_extends_and_choose_store_keep_the_two_apart(tmp_path):
    teacher, student, cfg = setup(tmp_path)
    rows = rows_of(4)
    base = tmp_path / "shared"
    plain_meta = {**META, "teacher_weights": safetensors_io.weights_manifest(teacher)}
    vid = vllmcache.identity(job_for(tmp_path), "org/T", teacher)
    vmeta = {**plain_meta, **vid}
    hashes = dj.row_hashes(rows)
    s1, _ = dj.choose_store(base, cfg, teacher, student, 8, 128, 3407, plain_meta, hashes)
    w = dj.SharedCacheWriter(s1, tmp_path / "view1", plain_meta, hashes, 8)
    for r in rows:
        w.add_row(*fake(4))
    w.close()
    s2, note = dj.choose_store(base, cfg, teacher, student, 8, 128, 3407, vmeta, hashes, vid)
    assert s2 != s1 and note is None
    assert not dj.store_extends(s1, vmeta, hashes), "even when forced into the same folder the index says whose answers they are"
    w2 = dj.SharedCacheWriter(s2, tmp_path / "view2", vmeta, hashes, 8)
    for r in rows:
        w2.add_row(*fake(4))
    w2.close()
    assert dj.read_cache_index(s2)["engine"] == "vllm-0.34.0" and "engine" not in dj.read_cache_index(s1)
    s3, _ = dj.choose_store(base, cfg, teacher, student, 8, 128, 3407, plain_meta, hashes)       # the standard teacher finds its own store again
    assert s3 == s1
    s4, _ = dj.choose_store(base, cfg, teacher, student, 8, 128, 3407, vmeta, hashes, vid)
    assert s4 == s2


# ----------------------------------------------------------------------------------------------------------------------- the command templates

def cfg_of(kind="docker", **kw):
    return ve.engine_config({"engine": {"kind": kind, **kw}})


def test_docker_templates_and_paths():
    box = vllmcache.Container(cfg_of("docker"), "C:\\models\\origin")
    cmd = box.start_command()
    assert cmd[:4] == ["docker", "run", "-d", "--gpus"] and "--entrypoint" in cmd and cmd[cmd.index("--entrypoint") + 1] == "sleep" and cmd[-2:] == ["vllm/vllm-openai:v0.34.0", "infinity"]
    assert "C:\\models\\origin:/models/base:ro" in cmd and box.model_arg == "/models/base"
    assert box.cp_in_command("C:\\w\\rows.npz", "/work/data/rows.npz") == ["docker", "cp", "C:\\w\\rows.npz", f"{box.name}:/work/data/rows.npz"]
    assert box.cp_out_command("/work/out/index.json", "C:\\t\\index.json") == ["docker", "cp", f"{box.name}:/work/out/index.json", "C:\\t\\index.json"]
    assert box.exec_command(["python3", "x.py"]) == ["docker", "exec", box.name, "python3", "x.py"]
    hub = vllmcache.Container(cfg_of("docker"), "Qwen/Qwen3.8-27B")
    assert "ndforge-vllm-hf:/root/.cache/huggingface:rw" in hub.start_command() and hub.model_arg == "Qwen/Qwen3.8-27B"


def test_docker_in_wsl_uses_linux_paths():
    box = vllmcache.Container(cfg_of("docker_wsl", distro="Ubuntu"), "C:\\models\\origin")
    assert box.start_command()[:6] == ["wsl.exe", "-d", "Ubuntu", "--", "docker", "run"] and "/mnt/c/models/origin:/models/base:ro" in box.start_command()
    assert box.cp_in_command("C:\\w\\rows.npz", "/work/data/rows.npz")[-2:] == ["/mnt/c/w/rows.npz", f"{box.name}:/work/data/rows.npz"]


def test_wslc_templates_are_guesses_and_editable():
    cfg = cfg_of("wslc")
    box = vllmcache.Container(cfg, "Qwen/Qwen3.8-27B")
    assert ve.WSLC_TEMPLATES["cp_in"] == "wslc cp {src} {name}:{dst}"
    run = box.start_command()
    assert run[:3] == ["wslc", "run", "-d"] and "sleep" in run and run[-2:] == [cfg["image"], "infinity"] and "ndforge-vllm-hf:/models" in run
    assert box.extra_llm == {"download_dir": "/models"}
    assert box.exec_command(["true"]) == ["wslc", "exec", box.name, "true"]
    cfg2 = cfg_of("wslc", wslc={"cp_in": "wslc container cp {src} {name}:{dst}", "exec": "wslc exec -- {name} {args}"})
    box2 = vllmcache.Container(cfg2, "Qwen/Qwen3.8-27B")
    assert box2.cp_in_command("a", "/b") == ["wslc", "container", "cp", "a", f"{box2.name}:/b"] and box2.exec_command(["x"])[:3] == ["wslc", "exec", "--"]
    cfg3 = cfg_of("docker", exec_templates={"exec": "docker exec -i {name} {args}"})
    assert vllmcache.Container(cfg3, "a/b").exec_command(["x"])[:3] == ["docker", "exec", "-i"]
    with pytest.raises(ve.EngineError):
        vllmcache.exec_templates({**cfg_of("docker"), "exec_templates": {"bogus": "x"}})
    with pytest.raises(vllmcache.VllmCacheError, match="url"):
        vllmcache.Container(cfg_of("url", url="http://x"), "m")


def test_the_selftest_captures_cp_and_exec_help():
    cmds = [" ".join(c) for c in ve.WSLC_HELP_COMMANDS]
    assert "wslc cp --help" in cmds and "wslc exec --help" in cmds


# ------------------------------------------------------------------------------------------------------------------------- a fake container

class FakeWorld:
    """Stands in for `docker` (and, with `exe="wslc"`, for `wslc`): keeps a folder per container, answers run/exec/cp/stop/rm, and runs vllm_cache.py in this process when the script is exec'ed.
    `bind`: "ok" (a `-v host:/work` mount shows the host folder as /work), "refuse" (the run command is refused) or "invisible" (accepted, but the container sees another folder).
    `cp`: "docker" (any destination), "dir_only" (a file destination that does not exist is ERROR_PATH_NOT_FOUND, `name:/work/` works), "none" (every cp fails), "drive_letter"
    (a source `C:\\...` is taken for `container:path`; `C:/...` works). `stdin_ok`: `exec -i ... cat > f` works. Existing tests keep the old behaviour: no bind, plain docker cp."""

    def __init__(self, tmp_path, monkeypatch, scorer=None, fail_script=None, folder="container", bind="refuse", cp="docker", stdin_ok=True, exe="docker"):
        self.root = tmp_path / folder
        self.root.mkdir()
        self.calls: list = []
        self.running: set = set()
        self.scorer = scorer or FakeScorer()
        self.fail_script = fail_script
        self.bind, self.cp_mode, self.stdin_ok, self.exe = bind, cp, stdin_ok, exe
        self.bound: dict = {}
        monkeypatch.setattr(ve, "_run", self.run)
        monkeypatch.setattr(ve, "_popen", self.popen)
        monkeypatch.setattr(vc, "build_scorer", lambda args: (self.scorer, []))
        monkeypatch.setattr(vllmcache, "SYNC_SECONDS", 0.0)
        monkeypatch.setattr(vllmcache.time, "sleep", lambda s: None)

    def local(self, p: str, name: str = "") -> Path:
        if p.startswith("/work") and name in self.bound:
            return Path(self.bound[name]) / p[len("/work"):].lstrip("/")
        return self.root / p.lstrip("/")

    def _res(self, code=0, out=b"", err=b""):
        return types.SimpleNamespace(returncode=code, stdout=out, stderr=err)

    def run(self, cmd, **kw):
        self.calls.append(list(cmd))
        c = list(cmd)
        if c[0] != self.exe:
            return self._res()
        if self.exe == "wslc" and c[1] == "list":
            return self._res()
        sub = c[1]
        if sub == "run":
            name = c[c.index("--name") + 1]
            mounts = [c[i + 1] for i, x in enumerate(c) if x == "-v" and c[i + 1].endswith(":/work")]
            if mounts:
                if self.bind == "refuse":
                    return self._res(125, err=b"invalid mount path: the work folder is not accepted")
                host = mounts[0][:-len(":/work")]
                if self.bind == "ok":
                    self.bound[name] = host
                    Path(host).mkdir(parents=True, exist_ok=True)
            self.running.add(name)
            self.root.mkdir(exist_ok=True)
        elif sub in ("stop", "rm", "remove"):
            self.running.discard(c[-1])
        elif sub == "inspect":
            return self._res(out=b'[{"Mounts": []}]')
        elif sub == "exec":
            return self._exec(c, kw)
        elif sub == "cp":
            return self._cp(c)
        return self._res()

    def _name(self, c):
        return next((x for x in c if x.startswith("ndforge-vllm-")), "")

    def _exec(self, c, kw):
        i = 2
        if c[i] == "-i":
            i += 1
        name = c[i]
        args = c[i + 1:]
        stdin = kw.get("stdin")
        if args[:1] == ["true"]:
            return self._res()
        if args[:1] == ["mkdir"]:
            for d in args[2:]:
                self.local(d, name).mkdir(parents=True, exist_ok=True)
        elif args[:1] == ["touch"]:
            self.local(args[1], name).write_text("1")
        elif args[:1] == ["ls"]:
            return self._res(out=b"total 0\n")
        elif args[:1] == ["mv"]:
            self.local(args[1], name).replace(self.local(args[2], name))
        elif args[:1] == ["cat"]:
            f = self.local(args[1], name)
            if not f.is_file():
                return self._res(1, err=b"cat: no such file")
            data = f.read_bytes()
            if kw.get("stdout") not in (None, subprocess.PIPE) and hasattr(kw["stdout"], "write"):
                kw["stdout"].write(data)
                return self._res()
            return self._res(out=data)
        elif args[:2] == ["sh", "-c"]:
            for part in args[2].split("&&"):
                part = part.strip()
                if part.startswith("mkdir -p "):
                    self.local(part.split()[-1], name).mkdir(parents=True, exist_ok=True)
                elif part.startswith("touch "):
                    self.local(part.split()[-1], name).write_text("1")
                elif part.startswith("cat > "):
                    if not (self.stdin_ok and "-i" in c[:4]) or stdin is None or stdin is subprocess.DEVNULL:
                        return self._res(1, err=b"stdin is not connected")
                    f = self.local(part.split("> ", 1)[1].strip(), name)
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_bytes(stdin.read())
                elif part.startswith("printf "):
                    m = re.match(r"printf %s '?(.*?)'? > (\S+)$", part)
                    f = self.local(m.group(2), name)
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(m.group(1))
        return self._res()

    def _cp(self, c):
        if "--help" in c:
            return self._res(out=b"Usage: cp SRC CONTAINER:DST")
        a, b = c[2], c[3]
        if self.cp_mode == "none":
            return self._res(1, err=b"Could not find the file in container. Error code: ERROR_PATH_NOT_FOUND")
        into = a.startswith("ndforge-vllm-") or (":" in a and a.split(":", 1)[0].startswith("ndforge-vllm-"))
        if not into:                                                        # host -> container
            if self.cp_mode == "drive_letter":
                if re.match(r"^[A-Za-z]:\\", a):
                    return self._res(1, err=b"no such container: C")
                if a.startswith("C:/fakehost/"):
                    a = "/" + a[len("C:/fakehost/"):]
            name, dst = b.split(":", 1)
            f = self.local(dst, name)
            if self.cp_mode == "dir_only":
                if dst.endswith("/"):
                    f = f / Path(a).name
                    f.parent.mkdir(parents=True, exist_ok=True)
                elif not f.exists():
                    return self._res(1, err=b"Could not find the file " + dst.encode() + b" in container. Error code: ERROR_PATH_NOT_FOUND")
            f.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(a, f)
            return self._res()
        name, srcp = a.split(":", 1)
        src = self.local(srcp, name)
        if not src.exists():
            return self._res(1, err=b"no such file")
        dst = Path(b)
        if b.endswith("/") or dst.is_dir():
            dst = dst / src.name
        shutil.copy(src, dst)
        return self._res()

    def popen(self, cmd, **kw):
        self.calls.append(list(cmd))
        name = cmd[2]
        args = [a.replace("/work", str(self.local("/work", name))) if a.startswith("/work") else a for a in cmd[5:]]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if self.fail_script:
                vc.emit("log", text="Loading weights took 12.5 s")
                vc.emit("error", code="vllm_cache_failed", message=self.fail_script)
                rc = 1
            else:
                rc = vc.main(args)
        lines = ["INFO 11-19 Loading weights took 12.5 seconds\n", "INFO Capturing CUDA graphs\n"] + [x + "\n" for x in buf.getvalue().splitlines()]
        proc = types.SimpleNamespace(stdout=iter(lines), returncode=rc, wait=lambda t=None: rc, poll=lambda: rc, terminate=lambda: None)
        return proc


def cache_job(tmp_path, **eng):
    return {"engine": {"kind": "vllm", "teacher": {}, "container": {"kind": "docker"}, **eng}, "part_tokens": 30}


def run_it(tmp_path, world, writer, rows_dir, first, last, stop_file=None, origin="Someone/Finetune", job=None):
    origin_dir = make_model(tmp_path / "origin")
    return vllmcache.run_chunk(job or cache_job(tmp_path), writer, rows_dir, {"teacher": "t", "k": K, "rows": 5, "data_hash": "h", "tokenizer_hash": "x", "max_len": 64, "general_key": {}, "general_rows": 0},
                               origin, origin_dir, first, last, K, 64, stop_file)


def new_writer(tmp_path, rows=5):
    meta = {"teacher": "t", "k": K, "rows": rows, "data_hash": "h", "tokenizer_hash": "x", "max_len": 64, "general_key": {}, "general_rows": 0}
    return cachewrite.CacheWriter(tmp_path / "cache", meta, 25)


def test_a_chunk_is_staged_scored_copied_out_and_the_container_removed(tmp_path, monkeypatch):
    rows = tmp_path / "rows"
    rows.mkdir()
    toks = make_rows(rows, [12, 30, 7, 25, 9])
    world = FakeWorld(tmp_path, monkeypatch)
    w = new_writer(tmp_path)
    r = run_it(tmp_path, world, w, rows, 0, 5)
    w.close()
    assert r["engine"] == "vllm-0.34.0" and r["teacher_quant"] == "bnb-nf4" and r["route"] == "bnb" and not r["stopped"] and r["rows"] == 5
    assert r["load_phases"] and any("Loading weights" in x["line"] for x in r["load_phases"]) and r["startup_seconds"] is not None
    # staged INTO the container (cp), nothing bind-mounted but the model folder
    cps = [c for c in world.calls if c[:2] == ["docker", "cp"]]
    assert any(c[-1].endswith(":/work/data/rows.npz") for c in cps) and any(c[-1].endswith(":/work/vllm_cache.py") for c in cps) and any(c[-1].endswith(":/work/cachewrite.py") for c in cps)
    runs = [c for c in world.calls if c[:2] == ["docker", "run"]]
    assert len(runs) == 2 and [x for x in runs[0] if x.startswith("-v")] == ["-v", "-v"], "first the host work folder is tried as /work, then the plain container"
    assert [x for x in runs[1] if x.startswith("-v")] == ["-v"] and not any(":/work" in x for x in runs[1]) and not any("rows" in x for x in runs[1])
    assert any(c[:3] == ["docker", "exec", c[2]] and c[3:5] == ["pip", "install"] and "vllm-bnb-plugin" in c[5] for c in world.calls), "the bnb plugin is installed for the in-flight route"
    assert not world.running, "the container is stopped and removed"
    assert [c[1] for c in world.calls if c[1] in ("stop", "rm")] == ["stop", "rm"] * 2, "the refused attempt is removed too"
    rd = dj.CacheReader(tmp_path / "cache")
    assert rd.done_rows == 5
    for i, t in enumerate(toks):
        ids, logp, resid = rd.get(i)
        e_ids, e_top, e_res = expected(FakeScorer.lp(t.tolist()))
        assert (ids == e_ids).all() and np.allclose(resid, e_res, atol=1e-6)


def test_prequant_route_downloads_into_the_volume_and_installs_nothing(tmp_path, monkeypatch):
    rows = tmp_path / "rows"
    rows.mkdir()
    make_rows(rows, [10, 10])
    world = FakeWorld(tmp_path, monkeypatch)
    w = new_writer(tmp_path, 2)
    r = run_it(tmp_path, world, w, rows, 0, 2, origin="Qwen/Qwen3.8-27B")
    assert r["route"] == "prequant" and r["teacher_quant"].startswith("awq:cyankiwi/")
    assert not any("pip" in c for c in world.calls)
    run = [c for c in world.calls if c[:2] == ["docker", "run"]][0]
    assert "ndforge-vllm-hf:/root/.cache/huggingface:rw" in run
    script = [c for c in world.calls if "vllm_cache.py" in " ".join(c) and c[1] == "exec"][0]
    assert script[script.index("--model") + 1] == "cyankiwi/Qwen3.8-27B-AWQ-INT4" and "--quantization" not in script


def test_resume_goes_on_after_the_rows_the_writer_has(tmp_path, monkeypatch):
    rows = tmp_path / "rows"
    rows.mkdir()
    toks = make_rows(rows, [12, 30, 7, 25, 9])
    world = FakeWorld(tmp_path, monkeypatch)
    w = new_writer(tmp_path)
    run_it(tmp_path, world, w, rows, 0, 2)                        # first session: rows 0-1 (a heal round, or a stopped job)
    w.flush()
    assert w.done_rows == 2
    world2 = FakeWorld(tmp_path, monkeypatch, folder="container2")
    w2 = new_writer(tmp_path)
    assert w2.done_rows == 2
    run_it(tmp_path, world2, w2, rows, w2.done_rows, 5)
    w2.close()
    script = [c for c in world2.calls if "vllm_cache.py" in " ".join(c) and c[1] == "exec"][0]
    assert script[script.index("--first-row") + 1] == "2" and script[script.index("--last-row") + 1] == "5"
    rd = dj.CacheReader(tmp_path / "cache")
    assert rd.done_rows == 5
    assert (rd.get(4)[0] == expected(FakeScorer.lp(toks[4].tolist()))[0]).all()
    assert len(world2.scorer.calls) and sum(len(c) for c in world2.scorer.calls) == 3


def test_the_stop_file_reaches_the_container_and_it_is_still_removed(tmp_path, monkeypatch):
    rows = tmp_path / "rows"
    rows.mkdir()
    make_rows(rows, [10, 10, 10])
    world = FakeWorld(tmp_path, monkeypatch)
    stop = tmp_path / "STOP"
    stop.write_text("1")
    w = new_writer(tmp_path, 3)
    run_it(tmp_path, world, w, rows, 0, 3, stop_file=stop)
    assert any(c[1] == "exec" and "touch" in c and c[-1] == "/work/STOP" for c in world.calls)
    assert not world.running


def test_a_failing_script_keeps_the_parts_and_still_removes_the_container(tmp_path, monkeypatch):
    rows = tmp_path / "rows"
    rows.mkdir()
    make_rows(rows, [10, 10])
    world = FakeWorld(tmp_path, monkeypatch, fail_script="CUDA out of memory")
    w = new_writer(tmp_path, 2)
    with pytest.raises(vllmcache.VllmCacheError, match="CUDA out of memory"):
        run_it(tmp_path, world, w, rows, 0, 2)
    assert not world.running and [c[1] for c in world.calls if c[1] in ("stop", "rm")] == ["stop", "rm"] * 2


def test_a_container_that_cannot_start_is_a_plain_error_and_is_cleaned(tmp_path, monkeypatch):
    rows = tmp_path / "rows"
    rows.mkdir()
    make_rows(rows, [10])
    world = FakeWorld(tmp_path, monkeypatch)
    orig = world.run

    def bad(cmd, **kw):
        if cmd[1] == "run":
            world.calls.append(list(cmd))
            return types.SimpleNamespace(returncode=125, stdout=b"", stderr=b"could not select device driver with capabilities: [[gpu]]")
        return orig(cmd, **kw)

    monkeypatch.setattr(ve, "_run", bad)
    with pytest.raises(vllmcache.VllmCacheError, match="device driver"):
        run_it(tmp_path, world, new_writer(tmp_path, 1), rows, 0, 1)
    assert [c[1] for c in world.calls if c[1] in ("stop", "rm")] == ["stop", "rm"] * 2, "the refused work-folder attempt and the plain one are both removed"


# ------------------------------------------------------------------------------------------- run_cache's decision: used, fallback, required, refused

class W:
    def __init__(self, done=0):
        self.done_rows, self.tokens = done, 0

    def flush(self):
        pass


def vjob(**eng):
    return {"engine": {"kind": "vllm", "teacher": {}, "container": {"kind": "docker"}, **eng}, "teacher": {"repo": "Someone/Finetune"}}


def call_teacher(job, monkeypatch, behaviour, tmp_path):
    result = {"notes": []}
    w = W()

    def chunk(cfg, writer, out, meta, origin, origin_dir, first, last, k, max_seq, stop_file, **kw):
        return behaviour(writer, last)

    monkeypatch.setattr(vllmcache, "run_chunk", chunk)
    return dj._vllm_teacher(job, w, [], tmp_path, {"engine": "vllm-0.34.0", "teacher_quant": "bnb-nf4"}, tmp_path, K, 64, 10, None, result), result


def test_a_good_run_is_done(tmp_path, monkeypatch):
    def ok(w, last):
        w.done_rows = last
        return {"tokens_per_s": 900.0, "startup_seconds": 500.0, "stopped": False, "notes": ["n"]}
    (outcome, note), res = call_teacher(vjob(), monkeypatch, ok, tmp_path)
    assert outcome == "done" and res["teacher_engine"]["tokens_per_s"] == 900.0 and res["teacher_engine"]["unchecked"] is True and res["tokens_per_s"] == 900.0


def test_a_failure_falls_back_with_a_plain_note(tmp_path, monkeypatch):
    def boom(w, last):
        raise vllmcache.VllmCacheError("CUDA out of memory")
    (outcome, note), res = call_teacher(vjob(), monkeypatch, boom, tmp_path)
    assert outcome == "fallback" and "CUDA out of memory" in note and "standard teacher" in note
    assert res["teacher_engine"]["fallback"] is True and res["teacher_engine"]["used"] is False


def test_an_incomplete_run_is_a_fallback_and_required_is_an_error(tmp_path, monkeypatch):
    def half(w, last):
        w.done_rows = 3
        return {"stopped": False}
    assert call_teacher(vjob(), monkeypatch, half, tmp_path)[0][0] == "fallback"
    with pytest.raises(dj.ReshapeError, match="required"):
        call_teacher(vjob(required=True), monkeypatch, half, tmp_path)


def test_a_stop_is_not_a_failure(tmp_path, monkeypatch):
    def stopped(w, last):
        w.done_rows = 2
        return {"stopped": True}
    assert call_teacher(vjob(), monkeypatch, stopped, tmp_path)[0][0] == "stopped"


def test_a_teacher_check_that_says_different_blocks_vllm(tmp_path, monkeypatch):
    called = []
    (outcome, note), res = call_teacher(vjob(teacher_check={"verdict": "different"}), monkeypatch, lambda w, last: called.append(1), tmp_path)
    assert outcome == "fallback" and not called and "different" in note
    with pytest.raises(dj.ReshapeError):
        call_teacher(vjob(teacher_check={"verdict": "different"}, required=True), monkeypatch, lambda w, last: called.append(1), tmp_path)


def test_wanted_is_off_by_default():
    assert not vllmcache.wanted({}) and not vllmcache.wanted({"engine": None}) and not vllmcache.wanted({"engine": {"kind": "docker"}})
    assert vllmcache.wanted({"engine": {"kind": "vllm"}})
