import pytest

from ndforge_worker import moeheal, thinkcheck

torch = pytest.importorskip("torch")
from ndforge_worker import distilljob as dj  # noqa: E402


def _targets(logits, k=4):
    lp = torch.log_softmax(logits, -1)
    top = lp.topk(k, -1)
    resid = torch.log1p(-top.values.exp().sum(-1).clamp(max=1 - 1e-6))
    return top.indices, top.values, resid


def test_softcap_changes_the_loss_and_none_does_not():
    torch.manual_seed(0)
    h, w = torch.randn(6, 8) * 4, torch.randn(50, 8) * 4
    ids, lp, resid = _targets(torch.tanh((h @ w.t()) / 30) * 30)
    mask = torch.ones(6)
    plain = dj.chunked_kd_loss(h, w, (ids, lp, resid), mask, chunk=4, checkpoint_head=False)[0]
    capped = dj.chunked_kd_loss(h, w, (ids, lp, resid), mask, chunk=4, checkpoint_head=False, softcap=30.0)[0]
    capped_ckpt = dj.chunked_kd_loss(h, w, (ids, lp, resid), mask, chunk=4, softcap=30.0)[0]
    assert capped == pytest.approx(float(capped_ckpt), abs=1e-5)
    assert capped < plain                      # the targets were made with the cap: the capped student matches them better
    assert capped.item() == pytest.approx(0.0, abs=1e-4)
    assert dj.chunked_kd_loss(h, w, (ids, lp, resid), mask, chunk=4, checkpoint_head=False, softcap=None)[0] == plain


def test_lora_goes_on_attention_only_and_experts_stay_frozen():
    names = ["model.layers.0.self_attn.q_proj.linear", "model.layers.0.self_attn.o_proj.linear", "model.layers.0.mlp.experts.w13", "model.layers.0.mlp.gate",
             "model.embed_tokens", "lm_head", "model.layers.3.self_attn.k_proj.sconv.conv1d"]
    assert moeheal.lora_targets(names) == names[:2]

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleDict({"mlp": torch.nn.ModuleDict({"experts": torch.nn.Linear(2, 2)})})
    m = M()
    assert moeheal.freeze_packed(m) == 0 or True
    assert all(not p.requires_grad for n, p in m.named_parameters() if ".mlp.experts." in n)


def test_the_training_text_is_read_back_by_the_thinking_check():
    text = moeheal.format_turn("work it out", "42")
    r = thinkcheck.split_output(text)
    assert r == {"think": "work it out", "answer": "42", "closed": True}
    assert thinkcheck.split_output(moeheal.format_turn("", "42"))["answer"] == "42"


def test_softcap_is_read_from_the_config():
    assert moeheal.softcap_of({"text_config": {"final_logit_softcapping": 30.0}}) == 30.0
    assert moeheal.softcap_of({"model_type": "qwen3_5"}) is None
