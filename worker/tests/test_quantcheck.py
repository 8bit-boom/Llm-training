"""Compare quants (quantcheck.py) with a stand-in llama-perplexity (a small Python script started through this interpreter: the same on Windows and Linux), hand-built
GGUF headers and a stand-in tokenizer. No torch, no llama.cpp, no network, no big files."""
import json
import struct
import threading
import time
from pathlib import Path

import pytest
from test_imatrix import StubTok, flag, general_file, project_rows, write_rows  # noqa: F401

from ndforge_worker import __main__ as entry
from ndforge_worker import exportjob, ggufinfo, imatrix, protocol, quantcheck

VOCAB = 10

PERPLEXITY = '''import json, os, struct, sys, time
a = sys.argv[1:]
if a == ["--version"]:
    print("version: 9999 (stubperp)")
    sys.exit(0)
log = os.environ["STUB_LOG"]
def flag(n):
    return a[a.index(n) + 1]
model = flag("-m")
mode = os.environ.get("PERP_MODE", "")
open(log, "a").write(json.dumps({"tool": "perplexity", "args": a}) + "\\n")
def hang():
    open(os.environ["STUB_STARTED"], "w").write("x")
    for _ in range(600):
        time.sleep(0.05)
    sys.exit(0)
if "--kl-divergence" not in a:
    if mode == "base_fail":
        print("error: unable to load model")
        sys.exit(1)
    if mode == "base_full":
        print("write error: No space left on device")
        sys.exit(1)
    if mode == "base_hang":
        hang()
    n, ctx = int(flag("--chunks")), int(flag("-c"))
    print("perplexity: tokenizing the input ..")
    for i in range(1, n + 1):
        print("[%d]6.1234," % i)
    if mode != "base_nofile":
        open(flag("--kl-divergence-base"), "wb").write(b"_logits_" + struct.pack("<iii", ctx, int(os.environ.get("PERP_VOCAB", "10")), n) + b"\\0" * 100)
    sys.exit(0)
if os.environ.get("PERP_HANG_ON") == os.path.basename(model):
    hang()
side = json.load(open(model + ".stub.json"))
fmt = side.get("format", "classic")
if fmt == "fail":
    print("llama_model_load: error loading model: out of memory")
    sys.exit(2)
n = int(flag("--chunks"))
if fmt == "nofigures":
    print("kl_divergence: nothing to show")
    sys.exit(0)
print("kl_divergence: computing over %d chunks, n_ctx=128, batch_size=2048, n_seq=4" % n)
print("chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Delta p RMS            Same top p")
for i in range(1, n + 1):
    print("%4d    %9.4f \\u00b1 %9.4f    %10.5f \\u00b1 %10.5f    %10.5f \\u00b1 %10.5f    %6.3f \\u00b1 %6.3f %%    %6.3f \\u00b1 %6.3f %%" % (i, 6.2, 0.3, 0.001, 0.002, side["mean"], 0.0005, 1.5, 0.1, side["top"], 0.3))
if fmt == "rowsonly":
    sys.exit(0)
if fmt == "table":
    print("| Metric | Value |")
    print("|---|---|")
    print("| Mean PPL(Q) | %.6f \\u00b1 0.037833 |" % side["ppl"])
    print("| Mean PPL(base) | 6.225194 \\u00b1 0.037771 |")
    print("| Mean KLD | %.8f \\u00b1 0.00000020 |" % side["mean"])
    print("| Maximum KLD | %.6f |" % side["max"])
    print("| 99.9%% KLD | %.6f |" % (side["max"] / 2))
    print("| Median KLD | 0.000013 |")
    print("| Same top p | %.3f \\u00b1 0.013 %% |" % side["top"])
    sys.exit(0)
print("")
print("====== Perplexity statistics ======")
print("Mean PPL(Q)                   : %10.6f \\u00b1 %10.6f" % (side["ppl"], 0.037833))
print("Mean PPL(base)                : %10.6f \\u00b1 %10.6f" % (6.225194, 0.037771))
print("Cor(ln(PPL(Q)), ln(PPL(base))): 99.99%")
print("Mean PPL(Q)/PPL(base)         : %10.6f \\u00b1 %10.6f" % (1.0004, 0.0001))
print("Mean PPL(Q)-PPL(base)         : %10.6f \\u00b1 %10.6f" % (0.0025, 0.0005))
print("")
print("====== KL divergence statistics ======")
print("Mean    KLD: %10.6f \\u00b1 %10.6f" % (side["mean"], 0.0001))
print("Maximum KLD: %10.6f" % side["max"])
print("99.9%%   KLD: %10.6f" % (side["max"] / 2))
print("99.0%%   KLD: %10.6f" % (side["max"] / 4))
print("95.0%%   KLD: %10.6f" % (side["max"] / 8))
print("90.0%%   KLD: %10.6f" % (side["max"] / 16))
print("Median  KLD: %10.6f" % (side["mean"] / 3))
print("Minimum KLD: %10.6f" % -0.000059)
print("")
print("====== Token probability statistics ======")
print("Mean    \\u0394p: -0.0745 \\u00b1 0.3952 %")
print("RMS \\u0394p    : %6.3f \\u00b1 %5.3f %%" % (1.5, 0.001))
print("Same top p: %6.3f \\u00b1 %5.3f %%" % (side["top"], 0.013))
'''


def gguf_file(path, vocab=VOCAB, arch="qwen35", ftype=1, tensors=(("token_embd.weight", [16, VOCAB], 1),), pad=0):
    """a tiny hand-built GGUF (version 3): a token list of `vocab` words, the architecture, the file type and tensor infos, then `pad` bytes of data"""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kvs = [s("general.architecture") + struct.pack("<I", 8) + s(arch)]
    if ftype is not None:
        kvs.append(s("general.file_type") + struct.pack("<II", 4, ftype))
    if vocab:
        kvs.append(s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, vocab) + b"".join(s(f"t{i}") for i in range(vocab)))
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs)
    for name, dims, typ in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", typ, 0)
    Path(path).write_bytes(out + b"\0" * pad)
    return str(path)


def stub(path, **side):
    Path(str(path) + ".stub.json").write_text(json.dumps({"mean": 0.01, "max": 0.5, "top": 97.5, "ppl": 6.23, **side}), encoding="utf-8")


@pytest.fixture
def qc(tmp_path, monkeypatch):
    tool = tmp_path / "llama-perplexity.py"
    tool.write_text(PERPLEXITY, encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("STUB_LOG", str(log))
    monkeypatch.setenv("STUB_STARTED", str(tmp_path / "started"))
    for v in ("PERP_MODE", "PERP_HANG_ON", "PERP_VOCAB"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(quantcheck, "find_perplexity", lambda: tool)
    monkeypatch.setattr(imatrix, "load_tokenizer", lambda sources: StubTok())
    monkeypatch.setattr(quantcheck, "free_bytes", lambda p: 10 ** 12)
    events = []
    monkeypatch.setattr(protocol, "emit", lambda kind, **f: events.append({"type": kind, **f}))

    class W:
        pass

    w = W()
    w.root, w.tool, w.log, w.events = tmp_path, tool, log, events
    w.lib = tmp_path / "quantcheck"
    w.ref = gguf_file(tmp_path / "ref-f16.gguf", pad=4000)
    w.q4 = gguf_file(tmp_path / "q4.gguf", ftype=15, tensors=(("token_embd.weight", [16, VOCAB], 12),), pad=1000)
    w.q5 = gguf_file(tmp_path / "q5.gguf", ftype=17, tensors=(("token_embd.weight", [16, VOCAB], 13),), pad=1500)
    stub(w.q4, mean=0.04, max=1.2, top=95.0)
    stub(w.q5, mean=0.01, max=0.4, top=98.0)
    w.own = tmp_path / "text.txt"
    w.own.write_text("The old kingdom of Ard stood between two rivers. " * 200, encoding="utf-8")
    w.train = write_rows(tmp_path / "train.jsonl", project_rows())
    w.test = write_rows(tmp_path / "test.jsonl", project_rows(n_chat=10, n_text=0))

    def calls():
        return [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []

    def job(**over):
        j = {"job": "quantcheck", "root": str(w.lib), "id": "20261115-1030-ref", "resume": False, "stop_file": str(tmp_path / "STOP"), "chunks": 4, "ctx": 128, "ngl": 99, "keep_logits": False,
             "reference": {"kind": "file", "file": w.ref},
             "quants": [{"path": w.q4, "label": "Q4_K_M", "group": "Q4_K_M", "matrix": False, "mixed": None}, {"path": w.q5, "label": "Q5_K_M", "group": "Q5_K_M", "matrix": False, "mixed": None}],
             "text": {"kind": "file", "file": str(w.own)}}
        j.update(over)
        return j

    w.calls, w.job = calls, job
    w.run = lambda **kw: quantcheck.make(job(**kw))
    w.folder = w.lib / "20261115-1030-ref"
    return w


def err(qc_, **kw):
    with pytest.raises(quantcheck.QuantCheckError) as e:
        qc_.run(**kw)
    return str(e.value)


# ------------------------------------------------------------------------------------------------------------------ the size of the base file

def test_the_base_file_size_follows_llama_cpps_format():
    # 8 bytes magic + 3 ints, the tokens of every chunk, and per chunk the second half of the context, each token 2 x (2 x ceil(vocab / 2) + 4) bytes
    assert quantcheck.kld_file_bytes(8, 2, 3) == 20 + 4 * 8 * 2 + 2 * 3 * 16
    assert quantcheck.kld_file_bytes(512, 16, 248320) == 20 + 4 * 512 * 16 + 16 * 255 * 2 * (2 * 124160 + 4)
    big = quantcheck.kld_file_bytes(512, 16, 248320)
    assert 2.0e9 < big < 2.1e9, "16 chunks of 512 for a vocabulary of 248,320 words: about 2 GB"
    assert quantcheck.kld_file_bytes(512, 64, 248320) > 8e9
    assert quantcheck.kld_file_bytes(512, 16, 248321) == quantcheck.kld_file_bytes(512, 16, 248322), "an odd vocabulary is rounded up to an even one"


def test_the_plan_defaults_to_16_chunks_and_shows_every_amount():
    p = quantcheck.plan(248320, 512, free=100 * 10 ** 9)
    assert p["default_chunks"] == 16 and [o["chunks"] for o in p["options"]] == [4, 8, 16, 32, 64, 128] and p["refusal"] is None and not p["vocab_assumed"]
    assert p["options"][2]["bytes"] == quantcheck.kld_file_bytes(512, 16, 248320) and p["options"][2]["gb"] == 2.03 and all(o["fits"] for o in p["options"][:5])
    assert all(o["fits"] for o in p["options"]) and p["options"][5]["gb"] == 16.21
    assert p["usable"] == 90 * 10 ** 9


def test_the_plan_shrinks_the_default_to_what_fits_and_refuses_when_nothing_does():
    p = quantcheck.plan(248320, 512, free=3 * 10 ** 9)      # 2.7 GB usable: 16 chunks (2.03 GB) fit, 32 (4 GB) do not
    assert p["default_chunks"] == 16 and [o["fits"] for o in p["options"]] == [True, True, True, False, False, False]
    p = quantcheck.plan(248320, 512, free=int(1.5e9))       # 1.35 GB usable: 8 chunks (1.01 GB)
    assert p["default_chunks"] == 8 and p["max_chunks"] == 10
    p = quantcheck.plan(248320, 512, free=int(0.2e9))
    assert p["default_chunks"] is None and "not enough free disk space" in p["refusal"] and "0.20 GB are free" in p["refusal"] and "248,320" in p["refusal"]
    p = quantcheck.plan(None, 512, free=None)
    assert p["vocab_assumed"] and p["vocab"] == 250_000 and p["usable"] is None and p["default_chunks"] == 16


def test_the_plan_job_reads_the_vocabulary_from_the_reference_header(qc):
    r = quantcheck.plan_job({"job": "quantcheck_plan", "reference": qc.ref, "ctx": 128, "root": str(qc.lib)})
    assert r["vocab"] == VOCAB and not r["vocab_assumed"] and r["ctx"] == 128 and r["free"] == 10 ** 12 and r["scored_per_chunk"] == 63
    r = quantcheck.plan_job({"job": "quantcheck_plan", "reference": str(qc.root / "nothing.gguf"), "ctx": 128, "root": str(qc.lib), "vocab": 1000})
    assert r["vocab"] == 1000 and "could not be read" in r["note"]


# ------------------------------------------------------------------------------------------------------------------ reading the output

CLASSIC = """
====== Perplexity statistics ======
Mean PPL(Q)                   :   6.227711 ±   0.037833
Mean PPL(base)                :   6.225194 ±   0.037771
Cor(ln(PPL(Q)), ln(PPL(base))):  99.99%
Mean ln(PPL(Q)/PPL(base))     :   0.000404 ±   0.000086
Mean PPL(Q)/PPL(base)         :   1.000404 ±   0.000086
Mean PPL(Q)-PPL(base)         :   0.002517 ±   0.000536

====== KL divergence statistics ======
Mean    KLD:   0.025150 ±   0.000200
Maximum KLD:   0.912206
99.9%   KLD:   0.300799
99.0%   KLD:   0.122222
95.0%   KLD:   0.055000
90.0%   KLD:   0.030000
Median  KLD:   0.000013
10.0%   KLD:  -0.000002
Minimum KLD:  -0.000059

====== Token probability statistics ======
Mean    Δp: -0.075 ± 0.395 %
Maximum Δp:  4.186%
RMS Δp    :  1.150 ± 0.001 %
Same top p: 97.739 ± 0.013 %
""".splitlines()


def test_the_classic_output_is_read_figure_by_figure():
    f = quantcheck.parse_kld(CLASSIC)
    assert f == {"ppl": 6.227711, "ppl_base": 6.225194, "mean_kld": 0.02515, "kld_unc": 0.0002, "max_kld": 0.912206, "p99_9": 0.300799, "p99": 0.122222, "p95": 0.055, "p90": 0.03,
                 "median_kld": 1.3e-05, "rms_dp": 1.15, "same_top": 97.739}


def test_the_table_of_the_readme_is_read_too_and_so_are_odd_spacings():
    table = ["| Metric                         |                    Value |", "|--------------------------------|--------------------------|",
             "| Mean PPL(Q)                    |      6.227711 ± 0.037833 |", "| Mean PPL(base)                 |      6.225194 ± 0.037771 |",
             "| Mean    KLD                    |  0.00002515 ± 0.00000020 |", "| Maximum KLD                    |                 0.012206 |",
             "| 99.9%   KLD                    |                 0.000799 |", "| Median  KLD                    |                 0.000013 |",
             "| Same top p                     |         99.739 ± 0.013 % |"]
    f = quantcheck.parse_kld(table)
    assert f["mean_kld"] == 2.515e-05 and f["kld_unc"] == 2e-07 and f["max_kld"] == 0.012206 and f["p99_9"] == 0.000799 and f["same_top"] == 99.739 and f["ppl"] == 6.227711
    f = quantcheck.parse_kld(["Mean KLD :0.5", "same top p=88.5 %", "KLD (mean) : 9"])
    assert f == {"mean_kld": 0.5, "same_top": 88.5}


def test_if_only_the_running_table_was_printed_its_last_row_is_used_and_marked_partial():
    rows = ["chunk  PPL  ln(PPL(Q)/PPL(base))  KL Divergence  Δp RMS  Same top p",
            "   1      6.2000 ±    0.3000       0.00100 ±    0.00200       0.01500 ±    0.00050    1.500 ±  0.100 %    97.000 ±  0.300 %",
            "   2      6.2100 ±    0.3000       0.00100 ±    0.00200       0.01700 ±    0.00050    1.600 ±  0.100 %    96.500 ±  0.300 %"]
    f = quantcheck.parse_kld(rows)
    assert f["mean_kld"] == 0.017 and f["same_top"] == 96.5 and f["rms_dp"] == 1.6 and f["ppl"] == 6.21 and f["partial"] is True


@pytest.mark.parametrize("junk", [[], [""], ["error: nothing here", "kl_divergence: 15.31 seconds per pass - ETA 0.50 minutes"], [None, 5, "Mean KLD: nan"], ["\x00\x01 ± ± :"]])
def test_output_that_is_not_figures_gives_nothing_and_never_raises(junk):
    assert "mean_kld" not in quantcheck.parse_kld(junk)


def test_the_markers_of_chat_templates_are_taken_out_of_the_text():
    assert quantcheck.strip_specials("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n") == "user\nHi\n\nassistant\nHello\n"
    assert quantcheck.strip_specials("<start_of_turn>user\nHi<end_of_turn>\n[INST]x[/INST]") == "user\nHi\n\nx\n"

    class Tok:
        all_special_tokens = ["<weird>"]
    assert quantcheck.strip_specials("a<weird>b", Tok()) == "a\nb\n"


# ------------------------------------------------------------------------------------------------------------------ a whole comparison

def test_a_comparison_makes_one_base_pass_and_one_pass_per_quant_with_the_exact_command_lines(qc):
    r = qc.run()
    base, c4, c5 = [c["args"] for c in qc.calls()]
    assert flag(base, "-m") == qc.ref and flag(base, "-f").endswith("test.txt") and flag(base, "--kl-divergence-base").endswith("base.kld") and "--kl-divergence" not in base
    assert flag(base, "-c") == "128" and flag(base, "--chunks") == "4" and flag(base, "-ngl") == "99"
    for a, q in ((c4, qc.q4), (c5, qc.q5)):
        assert flag(a, "-m") == q and flag(a, "--kl-divergence-base") == flag(base, "--kl-divergence-base") and "--kl-divergence" in a and "-f" not in a
        assert flag(a, "-c") == "128" and flag(a, "--chunks") == "4" and flag(a, "-ngl") == "99"
    assert r["state"] == "done" and [x["status"] for x in r["rows"]] == ["done", "done"]
    r4, r5 = r["rows"]
    assert (r4["mean_kld"], r4["max_kld"], r4["same_top"], r4["ppl"], r4["label"]) == (0.04, 1.2, 95.0, 6.23, "Q4_K_M")
    assert r4["size"] == Path(qc.q4).stat().st_size and r4["params"] == 160 and r4["bpw"] == round(r4["size"] * 8 / 160, 4)
    assert r["key"]["chunks"] == 4 and r["key"]["ctx"] == 128 and r["key"]["llama_cpp"] == "version: 9999 (stubperp)" and len(r["key"]["reference"]) > 10 and len(r["key"]["text"]) == 64
    assert r["reference"]["name"] == "ref-f16.gguf" and r["reference"]["vocab"] == VOCAB and r["settings"] == {"chunks": 4, "ctx": 128, "ngl": 99}
    saved = json.loads((qc.folder / "results.json").read_text(encoding="utf-8"))
    assert saved["rows"][0]["mean_kld"] == 0.04 and saved["state"] == "done" and saved["pareto"] == r["pareto"] == ["1", "2"]
    assert (qc.folder / "test.txt").is_file() and not (qc.folder / "base.kld").exists(), "the big logits file is deleted afterwards"
    assert r["base"]["deleted"] is True and sorted(p.name for p in qc.folder.iterdir()) == ["results.json", "test.txt"]


def test_progress_is_one_bar_over_all_passes_and_never_goes_back(qc):
    qc.run()
    prog = [e for e in qc.events if e["type"] == "progress"]
    assert {e["step"] for e in prog} == {"quantcheck"} and prog[-1]["value"] == 1.0 and prog[-1]["text"] == "Comparison ready"
    assert all(a["value"] <= b["value"] for a, b in zip(prog, prog[1:])), [e["value"] for e in prog]
    assert any(e["text"] == "Comparing Q4_K_M" for e in prog) and any(e["text"] == "Comparing Q5_K_M" for e in prog) and any(e["text"] == "Chunk 4 of 4" for e in prog)
    assert any(e["text"] == "The 16-bit model reads the test text" for e in prog)
    logs = [e["text"] for e in qc.events if e["type"] == "log"]
    assert any(t.startswith("$ ") and "--kl-divergence" in t for t in logs) and any("Mean    KLD" in t for t in logs)


@pytest.mark.parametrize("fmt", ["classic", "table"])
def test_both_output_formats_give_the_same_figures(qc, fmt):
    stub(qc.q4, mean=0.04, max=1.2, top=95.0, format=fmt)
    r = qc.run()
    assert (r["rows"][0]["mean_kld"], r["rows"][0]["same_top"], r["rows"][0]["max_kld"]) == (0.04, 95.0, 1.2)


def test_keep_logits_keeps_the_base_file(qc):
    r = qc.run(keep_logits=True)
    assert (qc.folder / "base.kld").is_file() and r["base"]["kept"] is True and r["base"]["bytes"] == (qc.folder / "base.kld").stat().st_size


def test_the_pareto_line_and_the_words_of_the_verdict(qc):
    rows = [{"id": "1", "label": "Q2_K", "size": 2_000_000_000, "mean_kld": 0.20, "status": "done", "same_top": 80.0}, {"id": "2", "label": "Q4_K_M", "size": 4_000_000_000, "mean_kld": 0.03, "status": "done"},
            {"id": "3", "label": "Q3_K_M (odd)", "size": 4_100_000_000, "mean_kld": 0.05, "status": "done"}, {"id": "4", "label": "Q8_0", "size": 8_000_000_000, "mean_kld": 0.001, "status": "done"},
            {"id": "5", "label": "broken", "size": 1, "status": "failed"}]
    assert quantcheck.pareto(rows) == ["1", "2", "4"]
    v = quantcheck.verdict(rows)
    assert "smallest file is Q2_K (2.00 GB" in v[0] and "closest to the 16-bit model is Q8_0 (8.00 GB" in v[0]
    assert v[1] == "On the line (nothing smaller is closer to the 16-bit model): Q2_K, Q4_K_M, Q8_0." and "Q3_K_M (odd)" in v[2] and v[2].startswith("Not worth it")
    assert any(s.startswith("At about 4.00 GB, Q4_K_M keeps the most of the 16-bit model") for s in v)
    assert "proxy" in v[-1] and "your own questions" in v[-1]
    assert quantcheck.verdict([]) == ["No quant was measured, so there is nothing to compare yet."]
    assert quantcheck.verdict([rows[1]])[0].startswith("Q4_K_M (4.00 GB) has a mean KL divergence of 0.0300")


def test_a_matrix_and_a_mixed_size_are_compared_in_words(qc):
    rows = [{"id": "1", "label": "Q4_K_M, no matrix", "group": "Q4_K_M", "matrix": False, "size": 4_000_000_000, "mean_kld": 0.040, "status": "done"},
            {"id": "2", "label": "Q4_K_M, with matrix", "group": "Q4_K_M", "matrix": True, "size": 4_000_000_000, "mean_kld": 0.025, "status": "done"},
            {"id": "3", "label": "Mixed 4-bit (balanced)", "group": "Mixed", "matrix": True, "mixed": "mixed_4", "size": 4_300_000_000, "mean_kld": 0.020, "status": "done"}]
    v = quantcheck.verdict(rows)
    assert "Q4_K_M with an importance matrix is 38 % closer to the 16-bit model than without one (mean KLD 0.0250 against 0.0400)" in " ".join(v)
    assert any(s.startswith("Mixed 4-bit (balanced) is 0.30 GB larger than Q4_K_M, with matrix and 20 % closer") for s in v)


def test_a_text_file_of_my_own_is_used_as_it_is_and_the_data_text_prefers_held_out_rows(qc):
    qc.run()
    assert (qc.folder / "test.txt").read_text(encoding="utf-8") == qc.own.read_text(encoding="utf-8")
    import shutil
    shutil.rmtree(qc.folder)
    r = qc.run(text={"kind": "data", "train": qc.train, "test": qc.test, "family": "qwen3_5", "tokenizer": ["x"], "seed": 1234})
    text = (qc.folder / "test.txt").read_text(encoding="utf-8")
    assert "fact0_0" in text and "<|im_start|>" not in text and "lore" not in text and "user\nWho is hero" in text, "chat template through, markers out, held-out rows only"
    assert "held-out" in r["text"]["source"] and r["notes"] == []
    shutil.rmtree(qc.folder)
    r = qc.run(text={"kind": "data", "train": qc.train, "test": str(qc.root / "no-test.jsonl"), "family": "qwen3_5", "tokenizer": ["x"]})
    assert "sample of the training data" in r["text"]["source"] and any("holds out no test questions" in n for n in r["notes"])
    first = (qc.folder / "test.txt").read_text(encoding="utf-8")
    shutil.rmtree(qc.folder)
    qc.run(text={"kind": "data", "train": qc.train, "test": None, "family": "qwen3_5", "tokenizer": ["x"]})
    assert (qc.folder / "test.txt").read_text(encoding="utf-8") == first, "the same data and seed give the same text"


def test_too_little_data_text_uses_fewer_chunks_and_says_so_or_refuses(qc):
    small = write_rows(qc.root / "small.jsonl", project_rows(n_chat=7, n_text=0, words=40))
    r = qc.run(chunks=8, text={"kind": "data", "train": qc.train, "test": small, "family": "qwen3_5", "tokenizer": ["x"]})
    got = r["settings"]["chunks"]
    assert 2 <= got < 8 and any("test text is short" in n and f"{got} chunks" in n for n in r["notes"])
    assert flag(qc.calls()[0]["args"], "--chunks") == str(got)
    import shutil
    shutil.rmtree(qc.folder)
    tiny = write_rows(qc.root / "tiny.jsonl", project_rows(n_chat=1, n_text=0, words=5))
    assert "too little test text" in err(qc, text={"kind": "data", "train": qc.train, "test": tiny, "family": "qwen3_5", "tokenizer": ["x"]})
    assert not qc.folder.exists()


# ------------------------------------------------------------------------------------------------------------------ the disk

def test_not_enough_free_disk_is_refused_before_anything_runs_with_the_numbers(qc, monkeypatch):
    monkeypatch.setattr(quantcheck, "free_bytes", lambda p: 100)
    msg = err(qc)
    assert "not enough free disk space" in msg and "free some space" in msg.lower()
    assert qc.calls() == [] and not qc.folder.exists()


def test_a_smaller_test_that_would_fit_is_named(qc, monkeypatch):
    need = quantcheck.kld_file_bytes(128, 4, VOCAB)
    monkeypatch.setattr(quantcheck, "free_bytes", lambda p: int(need * 0.9 / 4 * 3) )      # 4 chunks do not fit, 2 do
    msg = err(qc)
    assert "not enough free disk space for 4 chunks" in msg and "chunks would fit" in msg and "a tenth is kept in reserve" in msg
    assert qc.calls() == []


def test_the_disk_filling_up_during_the_base_pass_is_said_in_plain_words(qc, monkeypatch):
    monkeypatch.setenv("PERP_MODE", "base_full")
    msg = err(qc)
    assert "disk is full" in msg and not (qc.folder / "base.kld").exists()
    assert json.loads((qc.folder / "results.json").read_text(encoding="utf-8"))["state"] == "failed"


# ------------------------------------------------------------------------------------------------------------------ resume and Stop

def _stop_when_started(qc, delay=0.0):
    def waiter():
        for _ in range(400):
            if (qc.root / "started").exists():
                break
            time.sleep(0.05)
        time.sleep(delay)
        (qc.root / "STOP").write_text("")
    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    return t


def test_stop_kills_the_pass_keeps_the_finished_ones_and_a_resume_skips_them(qc, monkeypatch):
    monkeypatch.setenv("PERP_HANG_ON", "q5.gguf")
    t = _stop_when_started(qc)
    t0 = time.time()
    with pytest.raises(imatrix.Stopped):
        qc.run(keep_logits=False)
    t.join()
    assert time.time() - t0 < 20, "the tool (it would run about 30 s) was killed, not waited for"
    saved = json.loads((qc.folder / "results.json").read_text(encoding="utf-8"))
    assert saved["state"] == "stopped" and [r["status"] for r in saved["rows"]] == ["done", "pending"] and saved["rows"][0]["mean_kld"] == 0.04
    assert (qc.folder / "base.kld").is_file(), "the finished base pass is kept so that the rest can go on"
    n_before = len(qc.calls())
    monkeypatch.delenv("PERP_HANG_ON")
    (qc.root / "STOP").unlink()
    (qc.root / "started").unlink()
    r = qc.run(resume=True)
    new = qc.calls()[n_before:]
    assert len(new) == 1 and flag(new[0]["args"], "-m") == qc.q5 and "--kl-divergence" in new[0]["args"], "no second base pass, no second pass for the finished quant"
    assert [x["status"] for x in r["rows"]] == ["done", "done"] and r["rows"][0]["mean_kld"] == 0.04 and r["rows"][1]["mean_kld"] == 0.01 and r["state"] == "done"
    assert not (qc.folder / "base.kld").exists()


def test_stop_in_the_base_pass_leaves_no_half_written_logits(qc, monkeypatch):
    monkeypatch.setenv("PERP_MODE", "base_hang")
    t = _stop_when_started(qc)
    with pytest.raises(imatrix.Stopped):
        qc.run()
    t.join()
    assert not (qc.folder / "base.kld").exists()
    assert json.loads((qc.folder / "results.json").read_text(encoding="utf-8"))["state"] == "stopped"


def test_a_resume_with_other_settings_is_refused_while_its_logits_exist(qc, monkeypatch):
    monkeypatch.setenv("PERP_HANG_ON", "q5.gguf")
    t = _stop_when_started(qc)
    with pytest.raises(imatrix.Stopped):
        qc.run()
    t.join()
    monkeypatch.delenv("PERP_HANG_ON")
    (qc.root / "STOP").unlink()
    assert "other settings" in err(qc, resume=True, chunks=2)
    assert (qc.folder / "base.kld").is_file()
    # a new quant added to a finished comparison: only that one is measured
    q6 = gguf_file(qc.root / "q6.gguf", ftype=18, tensors=(("token_embd.weight", [16, VOCAB], 14),), pad=2000)
    stub(q6, mean=0.005, max=0.2, top=99.0)
    n = len(qc.calls())
    r = qc.run(resume=True, quants=[{"path": qc.q4, "label": "Q4_K_M"}, {"path": qc.q5, "label": "Q5_K_M"}, {"path": q6, "label": "Q6_K"}])
    assert [x["status"] for x in r["rows"]] == ["done"] * 3 and [flag(c["args"], "-m") for c in qc.calls()[n:]] == [qc.q5, q6]


def test_an_existing_folder_is_not_overwritten_without_resume(qc):
    qc.run()
    assert "exists already" in err(qc)


# ------------------------------------------------------------------------------------------------------------------ every failure

def test_no_llama_perplexity_says_how_to_install_and_where_it_looks(qc, monkeypatch):
    monkeypatch.setattr(quantcheck, "find_perplexity", lambda: None)
    msg = err(qc)
    assert "llama-perplexity was not found" in msg and "LLAMA_CPP_DIR" in msg and "https://github.com/ggml-org/llama.cpp" in msg
    assert not qc.folder.exists()


def test_find_perplexity_looks_next_to_quantize_in_llama_cpp_dir_and_on_the_path(tmp_path, monkeypatch):
    exe = "llama-perplexity.exe" if __import__("os").name == "nt" else "llama-perplexity"
    a, b = tmp_path / "a", tmp_path / "b" / "build" / "bin"
    a.mkdir()
    b.mkdir(parents=True)
    (a / exe).write_text("x")
    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: (a / "convert.py", a / "llama-quantize"))
    assert quantcheck.find_perplexity() == a / exe
    (a / exe).unlink()
    (b / exe).write_text("x")
    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: (None, None))
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path / "b"))
    assert quantcheck.find_perplexity() == b / exe
    (b / exe).unlink()
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path / "nowhere"))
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    assert quantcheck.find_perplexity() is None


def test_an_unreadable_or_missing_reference_is_a_plain_message(qc):
    (qc.root / "junk.gguf").write_bytes(b"not a gguf at all")
    assert "could not be read as a GGUF file" in err(qc, reference={"kind": "file", "file": str(qc.root / "junk.gguf")})
    assert "was not found" in err(qc, reference={"kind": "file", "file": str(qc.root / "gone.gguf")})
    assert "Choose the 16-bit reference" in err(qc, reference={})
    assert qc.calls() == [] and not qc.folder.exists()


def test_a_reference_that_is_already_quantised_is_a_note_not_a_refusal(qc):
    r = qc.run(reference={"kind": "file", "file": qc.q5})
    assert any("not a 16-bit file" in n for n in r["notes"]) and r["state"] == "done"


@pytest.mark.parametrize("quants,msg", [([], "at least one quantised GGUF"), ([{"path": ""}], "at least one quantised GGUF")])
def test_a_comparison_needs_a_quant(qc, quants, msg):
    assert msg in err(qc, quants=quants)


def test_a_quant_with_another_vocabulary_or_architecture_or_unreadable_fails_only_its_row(qc):
    other = gguf_file(qc.root / "other.gguf", vocab=20, pad=500)
    arch = gguf_file(qc.root / "arch.gguf", arch="llama", pad=500)
    (qc.root / "bad.gguf").write_bytes(b"GGUF nonsense")
    r = qc.run(quants=[{"path": other, "label": "Other"}, {"path": arch, "label": "Llama"}, {"path": str(qc.root / "bad.gguf"), "label": "Bad"}, {"path": qc.q5, "label": "Q5_K_M"}])
    st = [(x["label"], x["status"]) for x in r["rows"]]
    assert st == [("Other", "failed"), ("Llama", "failed"), ("Bad", "failed"), ("Q5_K_M", "done")]
    assert "vocabulary has 20 words" in r["rows"][0]["error"] and "not the same model" in r["rows"][1]["error"] and "could not be read as a GGUF" in r["rows"][2]["error"]
    assert r["state"] == "done" and (qc.folder / "base.kld").is_file() and any("kept" in n for n in r["notes"]), "a failed row keeps the base file for a retry"
    assert len([c for c in qc.calls() if "--kl-divergence" in c["args"]]) == 1, "no tool call for a file that cannot be compared"


def test_a_tool_that_prints_no_figures_or_fails_is_a_failed_row_with_a_reason(qc):
    stub(qc.q4, format="nofigures")
    stub(qc.q5, format="fail")
    r = qc.run()
    assert [x["status"] for x in r["rows"]] == ["failed", "failed"]
    assert "printed no KL divergence figures" in r["rows"][0]["error"] and "ran out of memory" in r["rows"][1]["error"]
    assert r["verdict"] == ["No quant was measured, so there is nothing to compare yet."]


def test_a_running_table_alone_is_still_a_measurement_marked_partial(qc):
    stub(qc.q4, format="rowsonly", mean=0.017)
    r = qc.run()
    assert r["rows"][0]["status"] == "done" and r["rows"][0]["partial"] is True and r["rows"][0]["mean_kld"] == 0.017


def test_a_base_pass_that_fails_or_writes_nothing_ends_the_comparison_plainly(qc, monkeypatch):
    monkeypatch.setenv("PERP_MODE", "base_fail")
    assert "could not load the file" in err(qc)
    assert not (qc.folder / "base.kld").exists() and json.loads((qc.folder / "results.json").read_text(encoding="utf-8"))["state"] == "failed"
    import shutil
    shutil.rmtree(qc.folder)
    monkeypatch.setenv("PERP_MODE", "base_nofile")
    assert "did not write the file of the 16-bit model's answers" in err(qc)


def test_a_text_file_that_is_missing_or_too_short_is_refused(qc):
    assert "was not found" in err(qc, text={"kind": "file", "file": str(qc.root / "gone.txt")})
    (qc.root / "short.txt").write_text("hi there")
    assert "too short" in err(qc, text={"kind": "file", "file": str(qc.root / "short.txt")})
    assert "Choose the test text" in err(qc, text={})
    assert not qc.folder.exists() and qc.calls() == []


def test_a_tokenizer_that_cannot_be_read_says_to_name_the_model(qc, monkeypatch):
    def boom(sources):
        raise OSError("no tokenizer files")
    monkeypatch.setattr(imatrix, "load_tokenizer", boom)
    assert "chat template" in err(qc, text={"kind": "data", "train": qc.train, "test": qc.test, "family": "qwen3_5", "tokenizer": ["x"]})


def test_a_data_file_with_no_rows_is_refused(qc):
    (qc.root / "empty.jsonl").write_text("")
    assert "could not be read, so there is no test text" in err(qc, text={"kind": "data", "train": str(qc.root / "empty.jsonl"), "test": None, "family": "x", "tokenizer": ["x"]})


def test_a_bad_amount_or_name_is_refused(qc):
    assert "Choose between" in err(qc, chunks=1)
    assert "not a valid name" in err(qc, id="../x")


# ------------------------------------------------------------------------------------------------------------------ making the reference

def test_a_16_bit_reference_is_made_with_the_converter_and_kept(qc, monkeypatch, tmp_path):
    from ndforge_worker import imatrixjob
    model = qc.root / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "w.safetensors").write_bytes(b"W")
    src = Path(qc.ref)
    conv = qc.root / "convert_hf_to_gguf.py"
    conv.write_text("import shutil, sys\na = sys.argv\nshutil.copy(%r, a[a.index('--outfile') + 1])\n" % str(src), encoding="utf-8")
    monkeypatch.setattr(imatrixjob, "find_tools", lambda need_convert: (conv, qc.root / "llama-imatrix.py"))
    r = qc.run(reference={"kind": "make", "source": {"kind": "model", "model": str(model), "family": "qwen3_5", "name": "m"}})
    assert (qc.folder / "reference-f16.gguf").is_file() and r["reference"]["name"] == "reference-f16.gguf" and any("16-bit reference was made" in n for n in r["notes"])
    assert not (qc.folder / "model-f16.gguf").exists() and not (qc.folder / "merged").exists()
    assert flag(qc.calls()[0]["args"], "-m") == str(qc.folder / "reference-f16.gguf")


def test_making_a_reference_without_llama_cpp_is_a_plain_message(qc, monkeypatch):
    from ndforge_worker import imatrixjob
    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: (None, None))
    monkeypatch.setattr(imatrix, "find_llama_imatrix", lambda q: None)
    monkeypatch.setenv("LLAMA_CPP_DIR", str(qc.root / "nowhere"))
    monkeypatch.setenv("PATH", str(qc.root / "nowhere"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: qc.root / "nohome"))
    assert "llama.cpp was not found" in err(qc, reference={"kind": "make", "source": {"kind": "model", "model": str(qc.root)}})
    assert not qc.folder.exists()


# ------------------------------------------------------------------------------------------------------------------ the GGUF reader and the worker entry

def test_the_gguf_reader_gives_the_vocabulary_the_shapes_the_types_and_the_precision(qc):
    d = ggufinfo.describe(qc.ref)
    assert d["vocab"] == VOCAB and d["arch"] == "qwen35" and d["params"] == 160 and d["full_precision"] is True and d["size"] == Path(qc.ref).stat().st_size
    assert ggufinfo.describe(qc.q4)["full_precision"] is False
    info = ggufinfo.read(qc.q4)
    assert info["tensors"] == [{"name": "token_embd.weight", "dims": [16, VOCAB], "type": 12, "offset": 0}] and info["data_start"] % 32 == 0
    assert ggufinfo.type_name(12) == "Q4_K" and ggufinfo.BPW["Q6_K"] == 6.5625 and ggufinfo.BPW["Q8_0"] == 8.5
    no_ft = gguf_file(qc.root / "nf.gguf", ftype=None, tensors=(("w", [4, 4], 1), ("v", [4, 4], 8)))
    assert ggufinfo.is_full_precision(ggufinfo.read(no_ft)) is False
    with pytest.raises(ValueError):
        ggufinfo.read(str(qc.own))


def test_the_worker_entry_runs_a_comparison_and_a_plan_and_reports_failures_in_plain_words(qc, monkeypatch):
    jobfile = qc.root / "job.json"
    jobfile.write_text(json.dumps(qc.job()), encoding="utf-8")
    assert entry.main(["x", str(jobfile)]) == 0
    res = [e for e in qc.events if e["type"] == "result"]
    assert res[0]["data"]["state"] == "done" and len(res[0]["data"]["rows"]) == 2
    jobfile.write_text(json.dumps({"job": "quantcheck_plan", "reference": qc.ref, "ctx": 128, "root": str(qc.lib)}), encoding="utf-8")
    assert entry.main(["x", str(jobfile)]) == 0 and [e for e in qc.events if e["type"] == "result"][1]["data"]["vocab"] == VOCAB
    for exc in (imatrix.Stopped(), quantcheck.QuantCheckError("plain words"), KeyError("x")):
        monkeypatch.setattr(quantcheck, "make", lambda job, e=exc: (_ for _ in ()).throw(e))
        jobfile.write_text(json.dumps(qc.job()), encoding="utf-8")
        assert entry.main(["x", str(jobfile)]) == 1
    errs = [e for e in qc.events if e["type"] == "error"]
    assert errs[0] == {"type": "error", "code": "quantcheck_stopped", "message": "Cancelled."} and errs[1] == {"type": "error", "code": "quantcheck_failed", "message": "plain words"}
    assert errs[2]["code"] == "quantcheck_failed"
