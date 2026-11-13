"""Healing by distillation: the loss math. The numpy part needs numpy only; the torch part runs where torch is installed (CI)."""
import math

import pytest

np = pytest.importorskip("numpy")

from ndforge_worker import distilljob as dj  # noqa: E402


def logits_for(probs):
    return np.log(np.array([probs], dtype=np.float64))


def test_reference_kl_is_zero_for_equal_distributions_and_never_negative():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(5, 40))
    for k in (1, 4, 32, 40):
        assert dj.kd_topk_reference(t, t, k) == pytest.approx(0.0, abs=1e-12)
        s = t + rng.normal(size=t.shape)
        assert dj.kd_topk_reference(s, t, k) > 0
    assert dj.kd_topk_reference(t + 3.0, t, 5) == pytest.approx(0.0, abs=1e-12)         # a constant shift is no difference to a softmax


def test_reference_kl_with_the_whole_vocabulary_is_the_plain_kl():
    rng = np.random.default_rng(1)
    t, s = rng.normal(size=(3, 12)), rng.normal(size=(3, 12))
    pt = np.exp(t - np.log(np.exp(t).sum(-1, keepdims=True)))
    ps = np.exp(s - np.log(np.exp(s).sum(-1, keepdims=True)))
    plain = float((pt * np.log(pt / ps)).sum(-1).mean())
    assert dj.kd_topk_reference(s, t, 12) == pytest.approx(plain, rel=1e-9)


def test_reference_kl_residual_bucket_by_hand():
    # teacher 0.5 / 0.3 / 0.2, student 0.4 / 0.4 / 0.2, only the top-1 kept: the bucket holds 0.5 (teacher) and 0.6 (student)
    t, s = logits_for([0.5, 0.3, 0.2]), logits_for([0.4, 0.4, 0.2])
    expect = 0.5 * math.log(0.5 / 0.4) + 0.5 * math.log(0.5 / 0.6)
    assert dj.kd_topk_reference(s, t, 1) == pytest.approx(expect, rel=1e-9)
    # the cut distribution loses information: the top-1 KL is not larger than the full one (data processing inequality)
    assert dj.kd_topk_reference(s, t, 1) <= dj.kd_topk_reference(s, t, 3) + 1e-12


def test_a_more_wrong_student_has_a_larger_loss():
    t = logits_for([0.7, 0.2, 0.1])
    near, far = logits_for([0.6, 0.25, 0.15]), logits_for([0.1, 0.2, 0.7])
    assert dj.kd_topk_reference(near, t, 2) < dj.kd_topk_reference(far, t, 2)


# --------------------------------------------------------------------------------------------------------------- torch

def test_the_torch_loss_equals_the_numpy_twin_and_chunking_changes_nothing():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    n, V, C, k = 10, 50, 8, 6
    teacher = torch.randn(n, V, generator=g)
    h = torch.randn(n, C, generator=g, requires_grad=True)
    w = torch.randn(V, C, generator=g)
    mask = torch.tensor([1.0] * 8 + [0.0, 0.0])
    ids, lp, resid = dj.topk_from_logits(teacher, k)
    assert ids.shape == (n, k) and lp.shape == (n, k) and resid.shape == (n,) and (resid >= 0).all()
    assert torch.allclose(lp.exp().sum(-1) + resid, torch.ones(n), atol=1e-6)
    # the plain (full logits) loss against the independent numpy version
    student = h @ w.t()
    full, count = dj.kd_topk_loss(torch.log_softmax(student, -1), ids, lp, resid, mask)
    ref = dj.kd_topk_reference(student.detach().numpy()[:8], teacher.numpy()[:8], k)
    assert float(count) == 8 and float(full) / 8 == pytest.approx(ref, abs=1e-5)
    full.backward()
    g_full = h.grad.clone()
    h.grad = None
    # the same in chunks of 3 tokens: the same value and the same gradient
    kd, n_tok, ce, n_lab = dj.chunked_kd_loss(h, w, (ids, lp, resid), mask, chunk=3)
    assert float(n_tok) == 8 and float(kd) == pytest.approx(float(full), abs=1e-5) and float(ce) == 0.0
    kd.backward()
    assert torch.allclose(h.grad, g_full, atol=1e-5)
    # the student equal to the teacher: no loss
    same, _ = dj.kd_topk_loss(torch.log_softmax(teacher, -1), ids, lp, resid, mask)
    assert abs(float(same)) < 1e-5


def test_chunked_cross_entropy_term():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(1)
    n, V, C = 7, 30, 5
    h, w = torch.randn(n, C, generator=g), torch.randn(V, C, generator=g)
    labels = torch.randint(0, V, (n,), generator=g)
    lmask = torch.tensor([0, 0, 1, 1, 1, 0, 1])
    ids, lp, resid = dj.topk_from_logits(torch.randn(n, V, generator=g), 4)
    kd, n_tok, ce, n_lab = dj.chunked_kd_loss(h, w, (ids, lp, resid), torch.ones(n), chunk=2, labels=labels, label_mask=lmask)
    want = F.cross_entropy(h @ w.t(), labels, reduction="none")[lmask.bool()].sum()
    assert float(n_lab) == 4 and float(ce) == pytest.approx(float(want), abs=1e-4)


def test_the_hidden_states_through_the_final_norm_hook_reproduce_the_logits(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_reshape_torch import IDS, load
    from tinymodel import build_arch_dir
    model = load(build_arch_dir(tmp_path / "m", "qwen3_5", num_layers=4))
    mask = torch.ones_like(IDS)
    with torch.no_grad():
        full = model(input_ids=IDS, attention_mask=mask, use_cache=False).logits
        hidden, out = dj.forward_hidden(model, IDS, mask)
    assert hidden.shape == (IDS.shape[0], IDS.shape[1], model.config.hidden_size)
    assert out.logits.shape[1] == 1                                          # only the last position went through the output layer
    w = model.get_output_embeddings().weight
    assert torch.allclose(hidden @ w.t(), full, atol=1e-4)
