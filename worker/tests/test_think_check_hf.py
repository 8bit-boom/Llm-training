"""The transformers backend of the thinking check on a tiny random model with the real Qwen3.5 chat template (plumbing only: the answers of a random model mean nothing)."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tinymodel import build_arch_dir, build_tokenizer  # noqa: E402

from ndforge_worker import protocol, reshapejob  # noqa: E402
from ndforge_worker import thinkcheck as tc  # noqa: E402
from ndforge_worker import thinkcheck_hf as hf  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return build_arch_dir(tmp_path_factory.mktemp("tiny") / "m", "qwen3_5")


def reqs(n_think=3, n_plain=2, max_new=6):
    out = [{"id": f"t{i}", "messages": [{"role": "user", "content": f"What is {i} plus 2?"}], "thinking": True, "max_new": max_new} for i in range(n_think)]
    return out + [{"id": f"c{i}", "messages": [{"role": "user", "content": f"Spell cat {i}."}], "thinking": False, "max_new": 3} for i in range(n_plain)]


SETTINGS = {**tc.SAMPLING, "max_new": 6}


def run(tiny, requests, job=None, stop=lambda: False):
    got = []
    hf.generate(requests, SETTINGS, got.append, stop, {"model": str(tiny), "load": "16bit", **(job or {})})
    return got


def test_the_prompt_ends_with_the_thinking_block_the_request_asks_for():
    tok = build_tokenizer("qwen3_5")
    on, off = reqs(1, 1)
    assert hf.prompt_of(tok, on).endswith("<|im_start|>assistant\n<think>\n")
    assert hf.prompt_of(tok, off).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_stop_ids_and_the_cut_of_a_generated_row():
    tok = build_tokenizer("qwen3_5")
    stops = hf.stop_ids(tok, None)
    assert tok.convert_tokens_to_ids("<|im_end|>") in stops and tok.convert_tokens_to_ids("<|endoftext|>") in stops
    assert hf.split_generated([5, 6, stops[0], 9, 9], stops, 0) == ([5, 6], False)
    assert hf.split_generated([5, 6, 7, 999, 999], stops, 999) == ([5, 6, 7], True)
    assert hf.split_generated([], stops, 999) == ([], True)


def test_every_question_is_answered_once_with_the_numbers_the_scorer_needs(tiny):
    got = run(tiny, reqs())
    assert sorted(g["id"] for g in got) == ["c0", "c1", "t0", "t1", "t2"]
    for g in got:
        assert isinstance(g["text"], str) and g["gen_tokens"] == len(g["tokens"]) <= (6 if g["id"][0] == "t" else 3) and isinstance(g["truncated"], bool)
        assert (g["think_tokens"] is None) == (g["id"][0] == "c")
        if g["id"][0] == "t":
            assert 0 <= g["think_tokens"] <= g["gen_tokens"]


def test_one_at_a_time_and_in_batches_both_work(tiny):
    assert len(run(tiny, reqs(), {"batch": 1})) == 5 and len(run(tiny, reqs(), {"batch": 8})) == 5


def test_stop_ends_between_batches(tiny):
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 2
    got = run(tiny, reqs(), {"batch": 1}, stop=stop)
    assert 0 < len(got) < 5


def test_out_of_memory_halves_the_batch_and_goes_on(tiny, monkeypatch):
    sizes = []
    real = hf._generate_batch

    def fake(model, enc, settings, max_new, stops, pad_id, seed):
        sizes.append(enc["input_ids"].shape[0])
        if enc["input_ids"].shape[0] > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return real(model, enc, settings, max_new, stops, pad_id, seed)
    monkeypatch.setattr(hf, "_generate_batch", fake)
    got = run(tiny, reqs(4, 0), {"batch": 8})
    assert len(got) == 4 and sizes[0] == 4 and sizes[-1] == 2 and max(sizes[2:]) <= 2


def test_another_error_passes_through(tiny, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("something else")
    monkeypatch.setattr(hf, "_generate_batch", boom)
    with pytest.raises(RuntimeError, match="something else"):
        run(tiny, reqs(1, 0))


def test_the_whole_job_through_the_worker_entry(tiny, monkeypatch, tmp_path):
    import io
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    monkeypatch.setattr(tc, "MAX_NEW", {"full": 4, "quick": 4, "control": 3})
    assert tc.run({"job": "think_check", "model": str(tiny), "backend": "hf", "load": "16bit", "quick": True, "batch": 8, "cache_file": str(tmp_path / "o.json")}) == 0
    lines = [json.loads(x) for x in buf.getvalue().splitlines() if x.startswith("{")]
    res = lines[-1]["data"]
    assert res["main"]["n"] == 16 and res["control"]["n"] == 8 and res["cached"] is False and res["settings"]["max_new"] == 4
    assert sum(1 for x in lines if x["type"] == "think_item") == 24
    assert tc.run({"job": "think_check", "model": str(tiny), "backend": "hf", "load": "16bit", "quick": True, "cache_file": str(tmp_path / "o.json")}) == 0


# ------------------------------------------------------------------------------------------------------------ a model with a LoRA on top, not merged

def save_adapter(tiny, folder, b_scale):
    from peft import LoraConfig, get_peft_model
    from test_reshape_torch import load
    pm = get_peft_model(load(tiny), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
    if b_scale:
        gen = torch.Generator().manual_seed(5)
        for name, p in pm.named_parameters():
            if "lora_B" in name:
                p.data.copy_(torch.randn(p.shape, generator=gen) * b_scale)
    pm.save_pretrained(str(folder))
    return str(folder)


def tokens_of(got):
    return {g["id"]: g["tokens"] for g in got}


def test_an_adapter_with_zero_b_gives_the_bases_answers_and_a_trained_one_changes_them(tiny, tmp_path):
    base = tokens_of(run(tiny, reqs(), {"batch": 8}))
    zero = save_adapter(tiny, tmp_path / "zero", 0)
    assert tokens_of(run(tiny, reqs(), {"batch": 8, "adapter": zero})) == base          # PEFT starts with B = 0: the same model, the same seed, the same tokens
    big = save_adapter(tiny, tmp_path / "big", 1.0)
    assert tokens_of(run(tiny, reqs(), {"batch": 8, "adapter": big})) != base


def test_the_job_with_an_adapter_runs_through_the_worker_entry(tiny, monkeypatch, tmp_path):
    import io
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    monkeypatch.setattr(tc, "MAX_NEW", {"full": 4, "quick": 4, "control": 3})
    adapter = save_adapter(tiny, tmp_path / "a", 0.5)
    assert tc.run({"job": "think_check", "model": str(tiny), "backend": "hf", "load": "16bit", "quick": True, "adapter": adapter, "double_quant": True,
                   "cache_file": str(tmp_path / "round.json")}) == 0
    lines = [json.loads(x) for x in buf.getvalue().splitlines() if x.startswith("{")]
    res = lines[-1]["data"]
    assert res["main"]["n"] == 16 and res["cached"] is False
    saved = json.loads((tmp_path / "round.json").read_text())
    assert saved["key"]["adapter"].startswith("adapter:") and saved["key"]["double_quant"] is True
