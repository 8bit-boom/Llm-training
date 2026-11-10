"""Speed helpers: padding, batches of similar length, kernel notes, the checkpointing decision, and a tiny real run that reports tokens per second."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from ndforge_worker import speed  # noqa: E402

LENGTHS = [20, 25, 400, 18, 380, 22, 410, 30, 15, 390, 28, 24, 405, 19, 21, 395]


def test_padding_is_the_share_of_computed_tokens_that_is_not_real():
    assert speed.padding_fraction([10, 10, 10, 10], [[0, 1], [2, 3]]) == 0.0
    assert speed.padding_fraction([10, 30], [[0, 1]]) == pytest.approx(1 - 40 / 60)
    assert speed.padding_fraction([5], []) == 0.0


def test_bucketed_batches_cover_every_example_once_group_similar_lengths_and_change_every_epoch():
    order, batches = speed.bucket_order(LENGTHS, 4, seed=1, epoch=0)
    assert sorted(order) == list(range(len(LENGTHS))) and [i for b in batches for i in b] == order
    assert all(len(b) == 4 for b in batches)
    spread = lambda bs: sum(max(LENGTHS[i] for i in b) - min(LENGTHS[i] for i in b) for b in bs)
    rnd = speed._batches(list(range(len(LENGTHS))), 4)
    assert spread(batches) < spread(rnd)
    assert speed.bucket_order(LENGTHS, 4, 1, 0)[0] == order                       # the same seed and epoch: the same order
    assert speed.bucket_order(LENGTHS, 4, 1, 1)[0] != order                       # a new grouping each epoch
    assert sorted(speed.bucket_order([7, 8, 9], 4, 1, 0)[0]) == [0, 1, 2]         # fewer examples than a batch


def test_the_estimate_shows_the_saving_and_bucketing_is_used_only_when_it_pays():
    rnd, bucketed = speed.padding_estimate(LENGTHS, 4)
    assert rnd > 0.4 and bucketed < rnd / 2, (rnd, bucketed)   # one batch must mix short and long here (6 long of 16), so bucketing cannot reach zero
    assert speed.padding_estimate([100] * 12, 4) == (0.0, 0.0)
    assert speed.padding_estimate(LENGTHS, 1) == (0.0, 0.0)                       # a batch of one has no padding
    assert speed.should_bucket("auto", rnd, len(LENGTHS), 4) is True
    assert speed.should_bucket("auto", 0.03, len(LENGTHS), 4) is False            # little to gain
    assert speed.should_bucket("on", 0.0, len(LENGTHS), 4) is True and speed.should_bucket("off", 0.9, len(LENGTHS), 4) is False
    assert speed.should_bucket("auto", 0.9, 5, 4) is False and speed.should_bucket("auto", 0.9, 100, 1) is False


def test_the_sampler_gives_every_index_each_epoch_and_a_new_order_the_next():
    pytest.importorskip("torch")
    s = speed.make_sampler(LENGTHS, 4, seed=3)
    first, second = list(s), list(s)
    assert sorted(first) == sorted(second) == list(range(len(LENGTHS))) and first != second and len(s) == len(LENGTHS)
    s.set_epoch(0)
    assert list(s) == first and list(s) == first        # the trainer names the epoch: then the order follows it, not the call count


def test_tokens_are_counted_real_and_padded():
    torch = pytest.importorskip("torch")
    c = speed.TokenCounter()
    batch = lambda: {"input_ids": torch.zeros(2, 5, dtype=torch.long), "attention_mask": torch.tensor([[1] * 5, [1, 1, 1, 0, 0]])}
    out = c.wrap(lambda b: batch())([])
    c.wrap(lambda b: batch())([])
    assert out["input_ids"].shape == (2, 5) and (c.real, c.padded) == (16, 20)


def test_kernel_notes_name_what_is_missing(monkeypatch):
    import importlib.util as iu
    real = iu.find_spec
    monkeypatch.setattr(iu, "find_spec", lambda n, *a, **k: None if n in ("fla", "causal_conv1d") else real(n, *a, **k))
    n = speed.kernel_notes("qwen3_5", "hf", True, (12, 0))
    assert len(n) == 1 and "flash-linear-attention and causal-conv1d" in n[0] and "slower" in n[0] and "Unsloth" in n[0]
    monkeypatch.setattr(iu, "find_spec", lambda n, *a, **k: object())
    assert speed.kernel_notes("qwen3_5", "hf", True, (12, 0)) == []
    assert "Unsloth's own fast kernels" in speed.kernel_notes("qwen3_5", "unsloth", True, (12, 0))[0]
    assert speed.kernel_notes("gemma4", "hf", True, (12, 0)) == []
    old = speed.kernel_notes("gemma4", "hf", True, (7, 5))
    assert len(old) == 1 and "no bf16" in old[0]


def test_checkpointing_stays_on_unless_the_longest_batch_clearly_fits_without_it():
    gb = 1024 ** 3
    assert speed.checkpointing_choice(10 * gb, 32 * gb, False) is False           # plenty of room: switched off (faster)
    assert speed.checkpointing_choice(30 * gb, 32 * gb, False) is True            # too close to the limit
    assert speed.checkpointing_choice(None, 32 * gb, True) is True                # ran out of memory
    assert speed.checkpointing_choice(None, 32 * gb, False) is True and speed.checkpointing_choice(1, 0, False) is True


def test_a_real_tiny_run_reports_tokens_per_second_padding_and_the_batching(tmp_path):
    pytest.importorskip("trl")
    from test_train_cpu import run_job
    from tinymodel import write_jsonl
    rows = []
    for i in range(24):    # short answers and long pieces, as in a real world
        long = i % 3 == 0
        rows.append({"type": "chat", "id": f"c{i}", "builder": "identity", "source": f"entity:{i}", "messages": [
            {"role": "system", "content": "You are the assistant."}, {"role": "user", "content": f"Who is Anders number {i}?"},
            {"role": "assistant", "content": ("Anders is a Widow-Bound Hunter who hunts colossal beasts near Dockside. " * 12) if long else "A hunter."}]})
    train = tmp_path / "mixed.jsonl"
    write_jsonl(train, rows)
    results = {}
    for mode in ("on", "off"):
        p, ev, out = run_job(tmp_path / mode, "qwen3_5", "llama", dataset={"train": str(train)}, out=str(tmp_path / mode / "run"), hyper={"batch": 4, "bucket": mode, "max_steps": 6, "max_seq": 512})
        assert p.returncode == 0, p.stderr[-2000:]
        sp = [e for e in ev if e["type"] == "speed"]
        assert len(sp) == 1
        results[mode] = (sp[0], [e for e in ev if e["type"] == "metric"])
    on, off = results["on"][0], results["off"][0]
    assert on["engine"] == off["engine"] == "hf", "the standard libraries trained (no Unsloth needed)"
    assert on["bucketed"] is True and off["bucketed"] is False
    assert on["pad_random_pct"] > 30 and on["pad_bucketed_pct"] < on["pad_random_pct"] / 2 and on["max_len"] > on["mean_len"] > 0 and on["batch"] == 4
    for mode, (sp, metrics) in results.items():
        assert metrics and all(m["tok_s"] and m["tok_s"] > 0 for m in metrics), (mode, metrics[:2])
    assert max(m["pad_pct"] for m in results["on"][1]) < min(m["pad_pct"] for m in results["off"][1]) + 1, "bucketed batches carry less padding"
    # the loss still goes down with bucketed batches
    losses = [m["loss"] for m in results["on"][1]]
    assert losses[0] > losses[-1], losses


def test_the_checkpointing_switch_works_on_a_real_model_with_the_memory_probe_stubbed(tmp_path, monkeypatch):
    """The decision itself needs a CUDA card; this checks the plumbing: off when the probe says there is room, back on (with its hooks) when there is not."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from tinymodel import build_llama_dir, sample_rows
    from transformers import AutoModelForCausalLM
    from ndforge_worker import data, trainjob
    build_llama_dir(tmp_path / "m", "qwen3_5")
    tok = data._inner(__import__("tinymodel").build_tokenizer("qwen3_5"))
    rows = data.encode_rows(tok, sample_rows(6, 0), "qwen3_5", 128)[0]
    gb = 1024 ** 3

    def fresh():
        base = AutoModelForCausalLM.from_pretrained(str(tmp_path / "m"))
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        return get_peft_model(base, LoraConfig(r=4, lora_alpha=4, task_type="CAUSAL_LM", target_modules="all-linear"))

    class Props:
        total_memory = 32 * gb
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: Props())
    for probe, expect_on in (((8 * gb, False), False), ((None, True), True), ((31 * gb, False), True)):
        model = fresh()
        assert model.base_model.model.is_gradient_checkpointing
        monkeypatch.setattr(trainjob.speed, "probe_checkpointing", lambda m, b, bf16, probe=probe: (assert_off(m), probe)[1])
        assert_off = lambda m: None if not m.base_model.model.is_gradient_checkpointing else (_ for _ in ()).throw(AssertionError("must be off while probing"))
        use, peak = trainjob._decide_checkpointing(model, rows, 2, 0, False)
        assert use is expect_on and model.base_model.model.is_gradient_checkpointing is expect_on, (probe, use)
    # a probe that crashes keeps the safe setting
    model = fresh()
    monkeypatch.setattr(trainjob.speed, "probe_checkpointing", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    use, peak = trainjob._decide_checkpointing(model, rows, 2, 0, False)
    assert use is True and model.base_model.model.is_gradient_checkpointing


# ---------------------------------------------------------------------------------------------------------------- advanced options
FIELDS = {"torch_compile", "use_liger_kernel"}


def test_advanced_options_are_off_unless_asked_for_and_say_why_when_they_cannot_be_on(monkeypatch):
    off = {"compile": "off", "liger": "off", "padding_free": "off"}
    assert speed.advanced_options(off, "generic", "llama", "hf", True, (12, 0), FIELDS) == ({}, False, [])
    monkeypatch.setattr(speed, "_have", lambda m: False)
    ask = {"compile": "on", "liger": "on", "padding_free": "on"}
    kw, pf, notes = speed.advanced_options(ask, "generic", "llama", "hf", True, (12, 0), FIELDS)
    assert kw == {} and pf is False and len(notes) == 3
    assert "Triton" in notes[0] and "liger-kernel package is not installed" in notes[1] and "flash-attn package" in notes[2] and all("asked for but" in n for n in notes)
    # Unsloth manages these itself; no card: no compile
    assert all("plain-libraries path" in n for n in speed.advanced_options(ask, "generic", "llama", "unsloth", True, (12, 0), FIELDS)[2])
    assert "only with a graphics card" in speed.advanced_options({"compile": "on"}, "generic", "llama", "hf", False, None, FIELDS)[2][0]


def test_advanced_options_turn_on_when_everything_they_need_is_there(monkeypatch):
    import types
    monkeypatch.setattr(speed, "_have", lambda m: True)
    mod = types.ModuleType("liger_kernel.transformers.monkey_patch")
    mod.MODEL_TYPE_TO_APPLY_LIGER_FN = {"llama": object()}
    for name in ("liger_kernel", "liger_kernel.transformers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers.monkey_patch", mod)
    ask = {"compile": "on", "liger": "on", "padding_free": "on"}
    kw, pf, notes = speed.advanced_options(ask, "generic", "llama", "hf", True, (12, 0), FIELDS)
    assert kw == {"torch_compile": True, "use_liger_kernel": True} and pf is True and len(notes) == 3
    # a model type Liger has no kernels for, an old Transformers without the setting
    assert "no kernels for the model type \"qwen3_5\"" in speed.advanced_options({"liger": "on"}, "qwen3_5", "qwen3_5", "hf", True, (12, 0), FIELDS)[2][0]
    assert "no such setting" in speed.advanced_options({"liger": "on"}, "generic", "llama", "hf", True, (12, 0), set())[2][0]
    # padding-free: not on a card before the RTX 30 series, and not for the families whose attention layers have not been checked
    assert "RTX 30 series" in speed.advanced_options({"padding_free": "on"}, "generic", "llama", "hf", True, (7, 5), FIELDS)[2][0]
    for fam in ("qwen3_5", "gemma4"):
        kw, pf, notes = speed.advanced_options({"padding_free": "on"}, fam, "x", "hf", True, (12, 0), FIELDS)
        assert pf is False and "has not been checked" in notes[0]


def test_padding_free_batches_put_examples_end_to_end_with_restarting_positions_and_no_label_across_a_boundary():
    torch = pytest.importorskip("torch")
    rows = [{"input_ids": [10, 11, 12, 13], "completion_mask": [False, False, True, True]}, {"input_ids": [20, 21, 22], "completion_mask": [True, True, True]}]
    out = speed.flatten_collator(0)(rows)
    assert set(out) == {"input_ids", "labels", "position_ids"} and out["input_ids"].shape == (1, 7)
    assert out["input_ids"][0].tolist() == [10, 11, 12, 13, 20, 21, 22]
    assert out["position_ids"][0].tolist() == [0, 1, 2, 3, 0, 1, 2]
    # only the answer is learned, and the first token of an example is never predicted from the example before it
    assert out["labels"][0].tolist() == [-100, -100, 12, 13, -100, 21, 22]
    c = speed.TokenCounter()
    c.wrap(speed.flatten_collator(0))(rows)
    assert (c.real, c.padded) == (7, 7)


def test_asking_for_everything_on_a_cpu_machine_trains_normally_and_says_what_stayed_off(tmp_path):
    pytest.importorskip("trl")
    from test_train_cpu import run_job
    p, ev, out = run_job(tmp_path, "qwen3_5", "llama", hyper={"max_steps": 4, "compile": "on", "liger": "on", "padding_free": "on"})
    assert p.returncode == 0, p.stderr[-2000:]
    sp = [e for e in ev if e["type"] == "speed"][0]
    assert sp["options"] == [] and sp["padding_free"] is False
    assert len([n for n in sp["notes"] if "asked for but" in n]) == 3, sp["notes"]
    assert [e for e in ev if e["type"] == "metric"] and ev[-1]["type"] == "result"


def test_an_explicit_unsloth_choice_without_unsloth_says_what_to_do_instead(tmp_path):
    pytest.importorskip("trl")
    try:
        import unsloth  # noqa: F401
        pytest.skip("Unsloth is installed here")
    except Exception:
        pass
    from test_train_cpu import run_job
    p, ev, out = run_job(tmp_path, "qwen3_5", "llama", backend="unsloth", hyper={"max_steps": 2})
    assert p.returncode == 1 and ev[-1]["type"] == "error"
    msg = ev[-1]["message"]
    assert "Unsloth could not be loaded" in msg and "standard" in msg and "Training engine" in msg, msg
