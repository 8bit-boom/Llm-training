"""Healing and graphics memory without torch: the automatic settings, the activation estimate, out-of-memory detection, the safer-settings ladder, the retry
controller with a fake trainer, the allocator setting. The ladder / activation / defaults are checked against tests/data/heal_cases.json, which the Rust
tests read as well (crates/ndforge-core/tests/reshape.rs), so the page and the worker use the same rules."""
import json
from pathlib import Path

import pytest

from ndforge_worker import healmem

CASES = json.loads((Path(__file__).parent / "data" / "heal_cases.json").read_text())


def test_the_ladder_matches_the_shared_cases():
    for c in CASES["ladder"]:
        got = healmem.next_safer(c["from"])
        if c["to"] is None:
            assert got is None, c
        else:
            assert got is not None and got[0] == c["to"] and got[1] == c["label"], c


def test_the_ladder_keeps_the_sequences_per_update_and_ends():
    s = {"checkpointing": False, "batch": 4, "grad_accum": 8, "chunk": 256}
    seen = 0
    order = []
    while True:
        step = healmem.next_safer(s)
        if step is None:
            break
        s, label = step
        order.append(label.split(" (")[0])
        assert s["batch"] * s["grad_accum"] == 32
        seen += 1
        assert seen < 20
    assert order == ["gradient checkpointing", "a smaller batch", "a smaller batch", "a smaller KL chunk", "a smaller KL chunk"]
    assert s == {"checkpointing": True, "batch": 1, "grad_accum": 32, "chunk": 64}


def test_an_odd_batch_goes_to_one_and_keeps_the_product():
    s, _ = healmem.next_safer({"checkpointing": True, "batch": 3, "grad_accum": 11, "chunk": 256})
    assert (s["batch"], s["grad_accum"]) == (1, 33)


def test_the_32_bit_embedding_copy_is_the_first_step_when_it_is_on_and_only_then():
    base = {"checkpointing": False, "batch": 4, "grad_accum": 8, "chunk": 256}
    s, label = healmem.next_safer({**base, "upcast": True})
    assert label == "the 32-bit embedding copy turned off" and s == {**base, "upcast": False}          # nothing else changes
    assert healmem.next_safer(s)[1] == "gradient checkpointing"                                          # then the old ladder, from its first rung
    assert healmem.next_safer(base)[1] == "gradient checkpointing"                                       # no key = off: the old ladder as it was
    s = {**base, "upcast": True}
    labels = []
    while (step := healmem.next_safer(s)) is not None:
        s, label = step
        labels.append(label.split(" (")[0])
        assert s["batch"] * s["grad_accum"] == 32 and len(labels) < 20
    assert labels == ["the 32-bit embedding copy turned off", "gradient checkpointing", "a smaller batch", "a smaller batch", "a smaller KL chunk", "a smaller KL chunk"]
    assert "embeddings and output layer in 32-bit" in healmem.describe({**base, "upcast": True}) and "32-bit" not in healmem.describe(base)


def test_the_memory_estimate_matches_the_shared_cases_and_counts_the_32_bit_copy_only_when_it_is_on():
    for c in CASES["memory"]:
        got = healmem.heal_memory_gb(c["student_b"], c["layers"], c["hidden"], c["vocab"], c["seq"], c["batch"], c["checkpointing"], c["rank"], c["kl_chunk"], c["qlora"], c["upcast"])
        assert got == pytest.approx(c["gb"], rel=1e-3), c
    args = dict(student_b=16.0, layers=40, hidden=5120, vocab=248_320, seq=1024, batch=2, checkpointing=True, rank=32, kl_chunk=128)
    on, off = healmem.heal_memory_gb(qlora=True, upcast=True, **args), healmem.heal_memory_gb(qlora=True, upcast=False, **args)
    # both tables (2 x 248,320 x 5,120 values) are 2 bytes per value more when copied to 32-bit; the estimate adds 1 GB of overhead and then 10 %
    assert on - off == pytest.approx(2 * 248_320 * 5120 * 2 / 1e9 * 1.10, abs=0.02) and 5.0 < on - off < 6.0
    assert healmem.heal_memory_gb(qlora=False, upcast=True, **args) == healmem.heal_memory_gb(qlora=False, upcast=False, **args)    # 16-bit LoRA has no such copy


def test_the_activation_estimate_matches_the_shared_cases_and_checkpointing_lowers_it():
    for c in CASES["activation"]:
        assert healmem.activation_gb(c["layers"], c["hidden"], c["seq"], c["batch"], c["checkpointing"]) == pytest.approx(c["gb"], rel=1e-3)
    a = lambda **kw: healmem.activation_gb(**{"layers": 40, "hidden": 4096, "seq": 1024, "batch": 2, "checkpointing": False, **kw})  # noqa: E731
    assert a(checkpointing=True) < a() / 5
    assert a(seq=2048) == pytest.approx(2 * a()) and a(batch=4) == pytest.approx(2 * a()) and a(seq=512) < a()


def test_the_automatic_settings_follow_the_student_size():
    for c in CASES["auto"]:
        got = healmem.auto_defaults(c["student_b"])
        assert got == {k: v for k, v in c.items() if k != "student_b"}, c
    small, big = healmem.auto_defaults(8.0), healmem.auto_defaults(16.0)
    assert (small["max_seq"], small["rank"], big["max_seq"], big["rank"], big["chunk"]) == (2048, 64, 1024, 32, 128)
    for b in (1, 8, 16, 21, 30):
        d = healmem.auto_defaults(b)
        assert d["batch"] * d["grad_accum"] == healmem.EFFECTIVE_BATCH


def test_checkpointing_auto():
    assert healmem.checkpointing_auto(8.0, 0.1, 100.0) and healmem.checkpointing_auto(21.0, 0.1, 100.0)       # 8 B or more: always
    assert healmem.checkpointing_auto(1.0, 1.0, None)                                                        # card unknown: on
    assert healmem.checkpointing_auto(1.0, 7.0, 10.0) and not healmem.checkpointing_auto(1.0, 5.0, 10.0)     # more than 60 % of the free memory


class AcceleratorError(RuntimeError):
    pass


class OutOfMemoryError(RuntimeError):
    pass


def test_out_of_memory_is_found_by_type_or_text():
    assert healmem.is_oom(OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert healmem.is_oom(OutOfMemoryError("anything"))                                  # the class name is enough (torch.OutOfMemoryError)
    assert healmem.is_oom(AcceleratorError("CUDA error: out of memory\nCUDA kernel errors might be asynchronously reported"))
    assert healmem.is_oom(RuntimeError("CUDA error: Out Of Memory"))
    assert not healmem.is_oom(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert not healmem.is_oom(ValueError("shape mismatch")) and not healmem.is_oom(MemoryError("out of memory"))   # the computer's own memory is another matter


S0 = {"checkpointing": False, "batch": 4, "grad_accum": 8, "chunk": 256, "max_seq": 2048, "rank": 64, "method": "qlora"}


def test_the_retry_controller_goes_down_the_ladder_and_finishes():
    calls, notes, freed = [], [], []

    def attempt(s, n):
        calls.append((n, dict(s)))
        if n < 2:
            raise OutOfMemoryError("CUDA out of memory")
        return "trained"
    r, used, tried = healmem.run_with_oom_retry(attempt, S0, on_retry=lambda n, label, s: notes.append((n, label)), free=lambda: freed.append(1))
    assert r == "trained" and len(calls) == 3 and len(freed) == 2
    assert [c[1]["checkpointing"] for c in calls] == [False, True, True] and [c[1]["batch"] for c in calls] == [4, 4, 2] and calls[2][1]["grad_accum"] == 16
    assert tried == ["gradient checkpointing", "a smaller batch (2 per step, 16 steps added up)"] and [n for n, _ in notes] == [1, 2]
    assert used["batch"] * used["grad_accum"] == 32 and used["max_seq"] == 2048 and used["rank"] == 64      # nothing that changes the maths


def test_other_errors_are_not_retried_and_the_failure_after_the_ladder_is_plain():
    def bad(s, n):
        raise ValueError("boom")
    with pytest.raises(ValueError):
        healmem.run_with_oom_retry(bad, S0)
    seen = []

    def always(s, n):
        seen.append(n)
        raise AcceleratorError("CUDA error: out of memory")
    with pytest.raises(healmem.HealOutOfMemory) as e:
        healmem.run_with_oom_retry(always, S0)
    text = str(e.value)
    assert seen == [0, 1, 2, 3]                                                          # one try and MAX_RETRIES retries
    assert "step train" in text and "sequence length 2048" in text and "rank 64" in text and "QLORA" in text
    assert text.index("shorter sequence length") < text.index("lower LoRA rank") < text.index("smaller model")
    assert "gradient checkpointing" in text and "AcceleratorError" in text and "saved answers" in text
    assert "QLoRA (4-bit) instead" not in text                                           # already QLoRA: not offered again
    again = healmem.advice({**S0, "method": "lora"}, [])
    assert again.index("shorter sequence length") < again.index("QLoRA (4-bit) instead") < again.index("lower LoRA rank")


def test_a_failed_attempt_is_released_before_the_memory_is_freed():
    import gc
    import weakref

    class Big:
        pass
    refs = []

    def attempt(s, n):
        big = Big()
        refs.append(weakref.ref(big))
        if n == 0:
            raise OutOfMemoryError("out of memory")
        return 1
    alive = []
    healmem.run_with_oom_retry(attempt, S0, free=lambda: (gc.collect(), alive.append(refs[0]() is not None)))
    assert alive == [False]                                                              # the traceback no longer held the first attempt's objects


def test_the_allocator_setting_is_set_once_and_never_over_the_persons_own(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    import os
    assert healmem.set_alloc_conf() is True and os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert healmem.set_alloc_conf() is False                                             # already set
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    assert healmem.set_alloc_conf() is False and os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF")
    monkeypatch.setattr("sys.platform", "win32")
    assert healmem.set_alloc_conf() is False and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ   # not supported on Windows (torch only warns)


def test_a_memory_error_outside_the_retries_names_the_step_and_the_next_actions():
    from ndforge_worker.__main__ import _explain_heal
    cache = _explain_heal("distill_cache", AcceleratorError("CUDA error: out of memory"))
    assert "step cache" in cache and "4-bit" in cache and cache.index("1.") < cache.index("2.") < cache.index("3.")
    assert "step train" in _explain_heal("distill", OutOfMemoryError("CUDA out of memory")) and "QLoRA" in _explain_heal("distill", OutOfMemoryError("x"))
    assert _explain_heal("distill_merge", ValueError("boom")) == "ValueError: boom"
