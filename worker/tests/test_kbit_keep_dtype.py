"""QLoRA healing with the embeddings and the output layer kept in 16-bit (backends.prepare_kbit_keep_dtype, kbit_upcast off) against PEFT's own kbit
preparation, on a tiny bf16 model of the real architecture on a CPU. bitsandbytes and a graphics card are not needed: the 4-bit part is not what is tested."""
import copy
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from test_reshape_width_torch import perturb  # noqa: E402
from tinymodel import build_arch_dir  # noqa: E402

from ndforge_worker import backends  # noqa: E402
from ndforge_worker import distilljob as dj  # noqa: E402


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    d = build_arch_dir(tmp_path_factory.mktemp("kbit") / "base", "qwen3_5", num_layers=4)
    perturb(d)
    return d


def load(folder, dtype):
    """the tiny model with its own output layer (the tiny config ties the two tables; the real students do not)"""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(folder), dtype=dtype)
    m.config.tie_word_embeddings = False
    m.lm_head.weight = torch.nn.Parameter(m.lm_head.weight.detach().clone())
    assert m.get_input_embeddings().weight is not m.get_output_embeddings().weight
    return m


@pytest.fixture()
def bf16(folder):
    return load(folder, torch.bfloat16)


def nbytes(m):
    return sum(p.numel() * p.element_size() for p in m.parameters())


def test_the_light_preparation_freezes_everything_keeps_the_big_tables_in_16_bit_and_makes_the_small_parameters_32_bit(bf16):
    m = backends.prepare_kbit_keep_dtype(bf16)
    assert m is bf16 and not any(p.requires_grad for p in m.parameters())
    assert m.get_input_embeddings().weight.dtype == torch.bfloat16 and m.get_output_embeddings().weight.dtype == torch.bfloat16
    for name, p in m.named_parameters():
        assert p.dtype == (torch.float32 if p.ndim <= 1 else torch.bfloat16), name         # norm weights, the per-head decay and bias vectors: 32-bit; every matrix: as loaded
    small = [n for n, p in m.named_parameters() if p.ndim <= 1]
    assert any(n.endswith("A_log") for n in small) and any(n.endswith("dt_bias") for n in small) and any("norm" in n for n in small)
    assert m.is_gradient_checkpointing is True
    ids = torch.tensor([[1, 2, 3]])
    assert m.get_input_embeddings()(ids).requires_grad, "the input embedding hook makes the layers' inputs require grads, so that checkpointing reaches the LoRA layers"


def test_the_reported_saving_is_the_bytes_of_the_matrices_that_were_not_made_32_bit_and_peft_makes_everything_32_bit(bf16, monkeypatch):
    logs = []
    monkeypatch.setattr(backends.protocol, "log", logs.append)
    peft_model = copy.deepcopy(bf16)
    independent = sum(p.numel() for n, p in bf16.named_parameters() if p.ndim >= 2) * 2         # every matrix and table, 2 bytes less than in 32-bit
    tables = (bf16.get_input_embeddings().weight.numel() + bf16.get_output_embeddings().weight.numel()) * 2
    assert backends.kbit_cast_savings(bf16) == independent and independent > tables > 0
    light = backends.prepare_kbit_keep_dtype(bf16)
    assert len(logs) == 1 and "Keeping the embeddings and the output layer in 16-bit: saves about" in logs[0] and logs[0].endswith("GB (compared with turning every 16-bit weight into 32-bit).")
    assert f"{independent / 1e9:.4f}" in logs[0]
    # PEFT's own function, on a copy of the same model: everything is 32-bit, which costs exactly the reported bytes more
    from peft import prepare_model_for_kbit_training
    prepare_model_for_kbit_training(peft_model)
    assert {p.dtype for p in peft_model.parameters()} == {torch.float32}
    assert not any(p.requires_grad for p in peft_model.parameters())
    assert nbytes(peft_model) - nbytes(light) == independent


def test_the_saving_is_a_real_number_for_a_big_vocabulary_without_building_it():
    """the shapes of the failing run: two tables of 248,320 x 5,120 values (untied)"""
    values = 2 * 248_320 * 5_120
    assert values * 2 / 1e9 == pytest.approx(5.09, abs=0.01) and values * 4 / 1e9 == pytest.approx(10.17, abs=0.01)      # 16-bit and 32-bit copies of both tables
    assert 248_320 * 5_120 * 2 / 1e9 == pytest.approx(2.54, abs=0.01)                                                    # one table in 16-bit


def test_a_tied_table_is_counted_once(folder):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(folder), dtype=torch.bfloat16)
    assert m.get_input_embeddings().weight is m.get_output_embeddings().weight
    table = m.get_input_embeddings().weight.numel()
    others = sum(p.numel() for n, p in m.named_parameters() if p.ndim >= 2 and "embed_tokens" not in n)
    assert backends.kbit_cast_savings(m) == (table + others) * 2


def targets_and_inputs(folder):
    teacher = load(folder, torch.float32).eval()
    ids = torch.randint(0, teacher.config.vocab_size, (2, 40), generator=torch.Generator().manual_seed(5))
    att = torch.ones_like(ids)
    with torch.no_grad():
        h, _ = dj.forward_hidden(teacher, ids, att)
        tg = dj.topk_from_logits(h.reshape(-1, h.shape[-1]) @ teacher.lm_head.weight.t(), 16)
    return {"input_ids": ids, "attention_mask": att, "labels": ids}, tg


def student_of(folder, noise, seed=2):
    """a 16-bit student that differs from the teacher by `noise` (a fraction of each weight's spread): a thinned model that was not healed yet"""
    m = load(folder, torch.float32)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn(p.shape, generator=gen) * noise * p.std().clamp(min=1e-3))
    return m.to(torch.bfloat16)


def first_kd(model, inputs, tg):
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):      # as the Trainer runs it (bf16=True)
        return float(dj.student_loss(model, inputs, lambda i: tg, 1.0, 16)[1])


# Measured on this tiny model (noise 0.3 and 1.0 x spread, seeds 2 to 6; KL about 0.026 to 0.057): the first-step KL with 16-bit tables differs from the one with
# 32-bit tables (PEFT's cast) by 0.003 % to 0.14 %. The tolerance is 1 %: seven times the largest value measured, still far below what a real fault (a wrong dtype
# in the product, a missing transpose, tables that are not the same) causes. A KL near zero (an almost healed model) is dominated by the 16-bit rounding of the
# BODY, which both variants share, so it is not compared relatively here (the test asks for a KL above 0.01).
KD_TOLERANCE = 0.01


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_the_first_step_kl_with_16_bit_tables_equals_the_one_with_32_bit_tables_within_a_small_tolerance(folder, seed):
    inputs, tg = targets_and_inputs(folder)
    student = student_of(folder, 0.3, seed)
    light = backends.prepare_kbit_keep_dtype(copy.deepcopy(student))
    from peft import prepare_model_for_kbit_training
    old = prepare_model_for_kbit_training(copy.deepcopy(student))
    tables_only = backends.prepare_kbit_keep_dtype(copy.deepcopy(student))
    for t in (tables_only.get_input_embeddings(), tables_only.get_output_embeddings()):
        t.weight.data = t.weight.data.float()
    assert old.get_output_embeddings().weight.dtype == torch.float32 and light.get_output_embeddings().weight.dtype == torch.bfloat16
    a, b, c = first_kd(light, inputs, tg), first_kd(old, inputs, tg), first_kd(tables_only, inputs, tg)
    assert a > 0.01 and b > 0.01, (a, b)                                  # a real KL (the student is not healed), not a noise floor
    assert a == pytest.approx(b, rel=KD_TOLERANCE), (a, b)
    assert a == pytest.approx(c, rel=KD_TOLERANCE), (a, c)


def test_the_loss_path_runs_the_product_in_the_tables_type_and_the_rest_in_32_bit(folder, monkeypatch):
    inputs, tg = targets_and_inputs(folder)
    student = backends.prepare_kbit_keep_dtype(student_of(folder, 0.3))
    seen = []
    real = dj.topk_from_logits
    monkeypatch.setattr(dj, "topk_from_logits", lambda logits, k: (seen.append(logits.dtype), real(logits, k))[1])
    h, _ = dj.forward_hidden(student, inputs["input_ids"], inputs["attention_mask"])
    h = h.reshape(-1, h.shape[-1])
    assert h.dtype == torch.bfloat16
    ids, logp, resid = real((h.to(torch.bfloat16) @ student.lm_head.weight.t()), 16)
    assert logp.dtype == torch.float32 and resid.dtype == torch.float32 and ids.dtype == torch.int64 and torch.isfinite(logp).all() and (resid >= 0).all()
    kd, n, ce, _ = dj.chunked_kd_loss(h, student.lm_head.weight, tg, torch.ones(h.shape[0]), 16, inputs["labels"].reshape(-1), torch.ones(h.shape[0], dtype=torch.bool))
    assert kd.dtype == torch.float32 and ce.dtype == torch.float32 and torch.isfinite(kd) and torch.isfinite(ce)
    # a 32-bit hidden state against 16-bit tables (the teacher with a different load) no longer fails with a dtype error
    kd2, *_ = dj.chunked_kd_loss(h.float(), student.lm_head.weight, tg, torch.ones(h.shape[0]), 16)
    assert torch.isfinite(kd2)


def test_a_lora_on_the_light_prepared_model_gets_gradients_from_the_kl_loss_and_the_frozen_tables_get_none(folder):
    from peft import LoraConfig, get_peft_model
    inputs, tg = targets_and_inputs(folder)
    model = backends.prepare_kbit_keep_dtype(student_of(folder, 0.3))
    lora = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, target_modules="all-linear", task_type="CAUSAL_LM"))
    gen = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for n, p in lora.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn(p.shape, generator=gen).to(p.dtype) * 0.05)      # B starts at zero, which would leave A without a gradient on the first step
    lora.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, kd, _ = dj.student_loss(lora, inputs, lambda i: tg, 1.0, 16)
    loss.backward()
    trainable = [(n, p) for n, p in lora.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n, _ in trainable)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0 for n, p in trainable), [n for n, p in trainable if p.grad is None or not float(p.grad.abs().sum()) > 0]
    assert lora.get_output_embeddings().weight.grad is None and lora.get_input_embeddings().weight.grad is None
    assert lora.get_output_embeddings().weight.dtype == torch.bfloat16 and float(kd.detach()) > 0


# --- load_hf picks the preparation by kbit_upcast (no bitsandbytes, no graphics card: the loaders are replaced) ---------------------------------------------

@pytest.fixture()
def load_hf_stubs(folder, monkeypatch):
    import peft
    import transformers
    from tinymodel import build_tokenizer
    calls = {"peft_prep": 0, "light": 0, "bnb": None, "from_pretrained": []}
    proto = load(folder, torch.bfloat16)                  # made before the loader is replaced
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def fake_pretrained(repo, **kw):
        calls["from_pretrained"].append(kw)
        return copy.deepcopy(proto)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_pretrained))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **kw: build_tokenizer("qwen3_5")))

    class FakeBnb:
        def __init__(self, **kw):
            calls["bnb"] = kw
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", FakeBnb)
    real_peft = peft.prepare_model_for_kbit_training

    def spy_peft(model, *a, **kw):
        calls["peft_prep"] += 1
        return real_peft(model, *a, **kw)
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", spy_peft)
    real_light = backends.prepare_kbit_keep_dtype

    def spy_light(model):
        calls["light"] += 1
        return real_light(model)
    monkeypatch.setattr(backends, "prepare_kbit_keep_dtype", spy_light)
    monkeypatch.setattr(backends.protocol, "progress", lambda *a, **kw: None)
    monkeypatch.setattr(backends.protocol, "log", lambda *a, **kw: None)
    return calls


def qlora_cfg(**over):
    return {"method": "qlora", "model": {"repo": "x", "family": "qwen3_5"}, "hyper": {"rank": 4, "alpha": 4, "max_seq": 64, "seed": 1}, **over}


def test_load_hf_with_qlora_uses_pefts_preparation_by_default_and_when_kbit_upcast_is_on(load_hf_stubs):
    for cfg in (qlora_cfg(), qlora_cfg(kbit_upcast=True)):
        load_hf_stubs.update(peft_prep=0, light=0)
        model, _, _ = backends.load_hf(cfg)
        assert (load_hf_stubs["peft_prep"], load_hf_stubs["light"]) == (1, 0)
        assert load_hf_stubs["bnb"]["load_in_4bit"] is True
        assert model.get_base_model().get_output_embeddings().weight.dtype == torch.float32          # the old behaviour, unchanged


def test_load_hf_with_qlora_and_kbit_upcast_off_keeps_the_tables_in_16_bit(load_hf_stubs):
    model, _, _ = backends.load_hf(qlora_cfg(kbit_upcast=False))
    assert (load_hf_stubs["peft_prep"], load_hf_stubs["light"]) == (0, 1)
    base = model.get_base_model()
    assert base.get_output_embeddings().weight.dtype == torch.bfloat16 and base.get_input_embeddings().weight.dtype == torch.bfloat16
    assert base.is_gradient_checkpointing is True
    assert any(p.requires_grad and "lora_" in n for n, p in model.named_parameters())


def test_the_other_methods_do_not_look_at_the_setting(load_hf_stubs):
    for method in ("lora", "rs_lora"):
        backends.load_hf(qlora_cfg(method=method, kbit_upcast=False))
    assert (load_hf_stubs["peft_prep"], load_hf_stubs["light"], load_hf_stubs["bnb"]) == (0, 0, None)


# --- the distill job passes the setting on ------------------------------------------------------------------------------------------------------------

def test_the_distill_job_asks_for_the_light_preparation_unless_the_old_behaviour_is_chosen(tmp_path, folder, monkeypatch):
    """The whole job on the tiny model with the loader replaced: what `backends.load` is asked for, at the start and in the retry after the first ladder rung."""
    from test_distill_torch import captured_trainer, distill_cfg, make_cache, metrics, run_in_process, thin_student
    student = thin_student(tmp_path, folder)
    make_cache(tmp_path, student, folder, monkeypatch)
    asked = []
    real_load = backends.load

    def spy(lcfg, backend):
        asked.append(lcfg.get("kbit_upcast"))
        return real_load(lcfg, backend)
    monkeypatch.setattr(backends, "load", spy)
    got = captured_trainer(monkeypatch)
    cfg = distill_cfg(tmp_path, student, folder, "cache", hyper={"max_steps": 1})
    code, ev = run_in_process(monkeypatch, dj.run, cfg)
    assert code == 0 and asked == [False] and [e for e in ev if e["type"] == "heal_settings"][0]["settings"]["upcast"] is False and got
    cfg = distill_cfg(tmp_path, student, folder, "cache", hyper={"max_steps": 1}, out=str(tmp_path / "run-on"))
    cfg["method"] = "qlora"
    cfg["distill"]["kbit_upcast"] = True
    cfg["hyper"]["optimizer"] = "adamw"
    monkeypatch.setattr(backends, "load", lambda lcfg, backend: (asked.append(lcfg.get("kbit_upcast")), real_load({**lcfg, "method": "lora"}, backend))[1])      # no bitsandbytes here: the model itself is loaded as LoRA
    code, ev = run_in_process(monkeypatch, dj.run, cfg)
    assert code == 0 and asked[-1] is True and [e for e in ev if e["type"] == "heal_settings"][0]["settings"]["upcast"] is True
    assert json.dumps(metrics(ev)[0]["kd"])


def test_an_out_of_memory_error_with_the_32_bit_copy_on_first_turns_it_off_and_reloads_the_model_without_it(tmp_path, folder, monkeypatch):
    from test_distill_torch import distill_cfg, fail_at, make_cache, run_in_process, thin_student
    student = thin_student(tmp_path, folder)
    make_cache(tmp_path, student, folder, monkeypatch)
    asked = []
    real_load = backends.load
    monkeypatch.setattr(backends, "load", lambda lcfg, backend: (asked.append(lcfg.get("kbit_upcast")), real_load({**lcfg, "method": "lora"}, backend))[1])    # no bitsandbytes here
    fail_at(monkeypatch, {8})
    cfg = distill_cfg(tmp_path, student, folder, "cache", hyper={"save_steps": 5, "checkpointing": "on", "optimizer": "adamw"})
    cfg["method"] = "qlora"
    cfg["distill"]["kbit_upcast"] = True
    code, ev = run_in_process(monkeypatch, dj.run, cfg)
    r = ev[-1]["data"]
    assert code == 0 and r["steps"] == 20 and r["oom_retries"] == ["the 32-bit embedding copy turned off"], r
    assert asked == [True, False] and r["settings"]["upcast"] is False and r["settings"]["checkpointing"] is True       # the maths and the other settings did not move
    assert [e["settings"]["upcast"] for e in ev if e["type"] == "heal_settings"] == [True] and [e["settings"]["upcast"] for e in ev if e["type"] == "heal_retry"] == [False]
    assert any("Going on from step 5" in e.get("text", "") for e in ev if e["type"] == "log")
