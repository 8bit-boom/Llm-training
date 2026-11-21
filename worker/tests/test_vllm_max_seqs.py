"""The in-container scoring script and the number of sequences: an explicit max_num_seqs, one retry on vLLM's own limit (the owner's Mamba cache blocks error), the sequence cap and the table of plain-language hints."""
import sys
import types

import pytest

from ndforge_worker import vllm_cache as vc, vllmcache, vllmengine as ve, vllmquant as vq

OWNER_ERROR = ("max_num_seqs (256) exceeds available Mamba cache blocks (171). Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed. "
               "Please lower max_num_seqs to at most 171 or increase gpu_memory_utilization.")


class Fake:
    """A fake `vllm` module: LLM() raises the owner's ValueError for max_num_seqs above `limit` and records every construction."""

    def __init__(self, limit=171, other_error=None):
        self.limit, self.other, self.calls = limit, other_error, []
        fake = self

        class LLM:
            def __init__(self, **kw):
                fake.calls.append(dict(kw))
                if fake.other:
                    raise fake.other
                seqs = kw.get("max_num_seqs", 256)                 # vLLM's own default
                if seqs > fake.limit:
                    raise ValueError(OWNER_ERROR.replace("256", str(seqs)).replace("171", str(fake.limit)))

        class SamplingParams:
            def __init__(self, **kw):
                self.kw = kw

        self.mod = types.ModuleType("vllm")
        self.mod.LLM, self.mod.SamplingParams, self.mod.__version__ = LLM, SamplingParams, "0.34.0"
        inputs = types.ModuleType("vllm.inputs")
        inputs.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
        self.inputs = inputs


@pytest.fixture
def logs(monkeypatch):
    out = []
    monkeypatch.setattr(vc, "log", out.append)
    monkeypatch.setattr(vc, "timing", lambda *a, **k: None)
    return out


def install(monkeypatch, fake):
    monkeypatch.setitem(sys.modules, "vllm", fake.mod)
    monkeypatch.setitem(sys.modules, "vllm.inputs", fake.inputs)


def args(*extra):
    return vc.parse_args(["--model", "/m", "--data", "/d", "--out", "/o", *extra])


def test_every_llm_gets_an_explicit_max_num_seqs_32_by_default(monkeypatch, logs):
    f = Fake()
    install(monkeypatch, f)
    vc.build_scorer(args())
    assert [c["max_num_seqs"] for c in f.calls] == [32], "vLLM's default of 256 is never left to chance"
    assert vc.llm_kwargs(args("--max-num-seqs", "12"))["max_num_seqs"] == 12
    assert "max_num_seqs" not in vc.llm_kwargs(args("--max-num-seqs", "0")), "0 = vLLM's own default, when somebody asks for it"


def test_the_owners_error_is_retried_once_with_the_number_vllm_names_and_a_log_line(monkeypatch, logs):
    f = Fake(limit=171)
    install(monkeypatch, f)
    scorer, notes = vc.build_scorer(args("--max-num-seqs", "256"))
    assert [c["max_num_seqs"] for c in f.calls] == [256, 128], "min(171, 256 // 2)"
    assert any("at most 171" in x and "max_num_seqs=128" in x for x in logs), logs
    assert any("vLLM needs a smaller max_num_seqs; the program retried with 128" in n for n in notes)
    f2 = Fake(limit=10)
    install(monkeypatch, f2)
    vc.build_scorer(args())
    assert [c["max_num_seqs"] for c in f2.calls] == [32, 10], "vLLM's number when it is lower than half: min(10, 16)"


def test_a_second_failure_is_a_plain_error_and_a_different_error_is_not_retried(monkeypatch, logs):
    f = Fake(limit=0)
    install(monkeypatch, f)
    with pytest.raises(vc.CacheScriptError, match="needs a smaller max_num_seqs; the program already retried once"):
        vc.build_scorer(args())
    assert len(f.calls) == 2, "one retry, not a loop"
    g = Fake(other_error=ValueError("something else is wrong with the model"))
    install(monkeypatch, g)
    with pytest.raises(ValueError, match="something else"):
        vc.build_scorer(args())
    assert len(g.calls) == 1
    h = Fake(other_error=RuntimeError("Engine core initialization failed. See root cause above."))
    install(monkeypatch, h)
    with pytest.raises(RuntimeError):
        vc.build_scorer(args())
    assert len(h.calls) == 1


def test_the_limit_is_read_from_both_spellings_and_never_from_other_errors():
    assert vc.seqs_limit(OWNER_ERROR) == 171
    assert vc.seqs_limit("max_num_seqs (64) exceeds available Mamba cache blocks (40).") == 40
    assert vc.seqs_limit("Please lower max_num_seqs to at most 7") == 7
    assert vc.seqs_limit("ValueError: Free memory on device is less than desired GPU memory utilization") is None
    assert vc.seqs_limit("the number (171) of blocks") is None
    assert vc.retry_seqs(171, 256) == 128 and vc.retry_seqs(10, 32) == 10 and vc.retry_seqs(5, None) == 5 and vc.retry_seqs(0, 1) == 1


def test_the_sequence_cap_comes_from_the_cache_that_is_left_and_is_logged_before_the_start():
    # the owner's numbers: a 4-bit weight size of 19.14 GB, 1.5 GB overhead, a 28.0 GB budget (29.8 GB free) = 7.4 GB of cache; a 4,096-token row about 0.48 GB
    n, text = vq.seq_cap(19.14, 31.8, 0.88, 29.8, 4096)
    assert n == 15 and "Rows scored at once: 15" in text and "7.4 GB" in text and "block limit of its own" in text
    assert vq.seq_cap(19.14, 31.8, 0.88, 29.8, 512)[0] == 32, "short rows: the default of 32 is the cap"
    assert vq.seq_cap(40, 31.8, 0.88, 29.8, 4096)[0] == 1, "never below one"
    sel = {"route": "bnb", "quant": "bnb-nf4"}
    a = vllmcache.script_args({"engine": {}}, sel, "/m", {}, 0, 10, 4096, 32, 15)
    assert a[a.index("--max-num-seqs") + 1] == "15"
    seqs = lambda a: a[a.index("--max-num-seqs") + 1]      # noqa: E731
    assert seqs(vllmcache.script_args({"engine": {}}, sel, "/m", {}, 0, 10, 4096, 32)) == "32"
    assert seqs(vllmcache.script_args({"engine": {"max_num_seqs": 8}}, sel, "/m", {}, 0, 10, 4096, 32, 15)) == "8", "the owner's own setting wins"


def test_the_hint_table_explains_the_known_startup_errors():
    h = ve.hints_for(OWNER_ERROR)
    assert any("smaller max_num_seqs" in x and "retries once" in x for x in h)
    assert any("Not enough free graphics memory" in x for x in ve.hints_for("ValueError: Free memory on device (15.1/31.8 GiB) on startup is less than desired GPU memory utilization (0.88, 28.0 GiB)."))
    assert any("sequence length does not fit" in x for x in ve.hints_for("ValueError: The model's max seq len (4104) is larger than the maximum number of tokens that can be stored in KV cache (2048)."))
    assert any("does not know this quantization" in x for x in ve.hints_for("ValueError: Unknown quantization method: bitsandbytes"))
    assert any("ran out of memory" in x for x in ve.hints_for("torch.OutOfMemoryError: CUDA out of memory."))
    assert ve.hints_for("INFO all fine") == []


def test_the_failure_text_shows_the_hint_under_the_root_cause_lines(monkeypatch):
    logged = []
    monkeypatch.setattr(vllmcache.protocol, "log", logged.append)
    box = vllmcache.Container.__new__(vllmcache.Container)
    box.started, box.kind, box.port, box.cfg = False, "docker", 1, {"kind": "docker"}
    text = box.evidence(["ValueError: " + OWNER_ERROR, "RuntimeError: Engine core initialization failed. See root cause above."])
    assert "Root cause:" in text and "What it probably means: vLLM needs a smaller max_num_seqs" in text
    assert any(x.startswith("Known cause: vLLM needs a smaller max_num_seqs") for x in logged)
