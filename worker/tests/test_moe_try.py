"""Try first for a mixture of experts: the questions, and that an imitated cut gives the loss of the cut that is really made (tiny stand-in model)."""
import types

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import inkfake  # noqa: E402
import test_moe_stats as T  # noqa: E402
from ndforge_worker import moe_layout as ml  # noqa: E402
from ndforge_worker import moetry  # noqa: E402
from ndforge_worker.reshape import ReshapeError  # noqa: E402


class LoopModel(T.Model):
    """the stand-in model, reading its number of passes from its config at every forward pass, as Inkling2Loop does"""

    def __init__(self, e=T.E, k=T.K):
        super().__init__(e, k)
        self.config = types.SimpleNamespace(loop_num_passes=T.PASSES)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            for _ in range(self.config.loop_num_passes if T.CORE[0] <= i < T.CORE[1] else 1):
                x = x + 0.5 * layer.mlp(F.layer_norm(x, (T.H,)))
        return x


class LM(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.emb, self.body, self.head = nn.Embedding(50, T.H), body, nn.Linear(T.H, 50)
        self.config = body.config

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return types.SimpleNamespace(logits=self.head(self.body(self.emb(input_ids))))


def cfg_for(experts=T.E, k=T.K):
    c = inkfake.tiny_config(experts, k)
    return c


def batches(n=4):
    g = torch.Generator().manual_seed(8)
    return [{"input_ids": torch.randint(0, 50, (2, 12), generator=g), "attention_mask": torch.ones(2, 12, dtype=torch.long)} for _ in range(n)]


@pytest.fixture()
def lm():
    torch.manual_seed(11)
    return LM(LoopModel()).eval()


def scores_of(lm):
    from ndforge_worker import moemeasure as mm
    layers = list(lm.body.layers)
    return mm.measure(lm, layers, batches(), T.E, T.CORE, T.PASSES)


def test_keys_round_trip_and_garbage_is_refused():
    assert moetry.parse_key(moetry.make_key(e=96, k=6, p=2, s="freq")) == {"e": 96, "k": 6, "p": 2, "s": "freq"}
    assert moetry.make_key() == "base" and moetry.parse_key("base") == {"e": None, "k": None, "p": None, "s": None}
    for bad in ("x=1", "e=", "e=abc"):
        with pytest.raises((ReshapeError, ValueError)):
            moetry.parse_key(bad)


def test_the_default_questions_never_offer_what_the_owner_ruled_out():
    qs = moetry.default_questions(128, 8, 3)
    assert qs[0] == "base" and qs[1:5] == ["e=112", "e=96", "e=80", "e=64"]
    assert "k=6" in qs and "k=4" in qs and "p=2" in qs and "p=1" not in qs                      # one pass only for a variable loop
    assert any("s=freq" in q for q in qs) and any("e=96|k=6" == q for q in qs)
    assert moetry.default_questions(128, 8, 3, varloop=True).count("p=1") == 1
    sp = moetry.default_questions(128, 8, 3, goal="speed")
    assert sp[1:4] == ["k=6", "k=4", "p=2"]
    # no expert count below 32 or above half removed
    assert all(moetry.parse_key(q)["e"] is None or 64 <= moetry.parse_key(q)["e"] <= 112 for q in qs)
    assert len(qs) == len(set(qs))


def test_every_answer_is_the_loss_of_the_cut_that_would_be_made(lm):
    layers = list(lm.body.layers)
    data = batches()
    cfg = cfg_for()
    base = moetry.ask(lm, layers, data, "base", cfg, None)
    # the experts per token: the same as setting it
    k1 = moetry.ask(lm, layers, data, "k=1", cfg, None)
    for layer in layers:
        layer.mlp.gate.top_k = 1
    real = moetry.loss_of(lm, data)[0]
    for layer in layers:
        layer.mlp.gate.top_k = T.K
    assert k1["loss"] == pytest.approx(real, abs=1e-6) and k1["loss"] != pytest.approx(base["loss"], abs=1e-6)
    # the passes: the same as a model whose config says so
    p1 = moetry.ask(lm, layers, data, "p=1", cfg, None)
    lm.config.loop_num_passes = 1
    real = moetry.loss_of(lm, data)[0]
    lm.config.loop_num_passes = T.PASSES
    assert p1["loss"] == pytest.approx(real, abs=1e-6)
    # the baseline did not change after all of that
    again = moetry.ask(lm, layers, data, "base", cfg, None)
    assert again["loss"] == pytest.approx(base["loss"], abs=1e-7)
    assert not any(l.mlp.gate._forward_hooks for l in layers)


def test_removing_experts_is_the_loss_of_the_physically_smaller_model(lm):
    """the exactness claim, end to end on a loss: imitation == slicing experts, router rows and bias"""
    layers = list(lm.body.layers)
    data = batches()
    scores = scores_of(lm)
    keep = ml.select_experts(scores, 4, "reap")
    imit = __import__("ndforge_worker.moe_stats", fromlist=["x"]).Imitation(layers, keep, T.E).attach()
    masked = moetry.loss_of(lm, data)[0]
    imit.detach()
    small = LoopModel(e=4)
    with torch.no_grad():
        for src, dst in zip(lm.body.layers, small.layers):
            ids = torch.tensor(keep[list(lm.body.layers).index(src)])
            dst.mlp.gate.weight.copy_(src.mlp.gate.weight[ids])
            dst.mlp.gate.e_score_correction_bias.copy_(src.mlp.gate.e_score_correction_bias[ids])
            dst.mlp.experts.gate_up_proj.copy_(src.mlp.experts.gate_up_proj[ids])
            dst.mlp.experts.down_proj.copy_(src.mlp.experts.down_proj[ids])
    twin = LM(small)
    twin.emb.load_state_dict(lm.emb.state_dict())
    twin.head.load_state_dict(lm.head.state_dict())
    twin.eval()
    physical = moetry.loss_of(twin, data)[0]
    assert masked == pytest.approx(physical, abs=1e-5)


def test_the_analysis_has_the_rise_the_band_and_the_speed_changes(lm):
    layers = list(lm.body.layers)
    data = {"passes": {"base": {"loss": 2.0, "loss_h1": 2.0, "loss_h2": 2.0}, "p=2": {"loss": 2.5, "loss_h1": 2.4, "loss_h2": 2.6, "coverage": None, "spread": None},
                       "k=1": {"loss": 2.01, "loss_h1": 2.0, "loss_h2": 2.02}}}
    rows = {r["key"]: r for r in moetry.rows_of(data, inkfake.real_config())["rows"]}
    assert rows["p=2"]["band"] == "large" and rows["p=2"]["rise"] == pytest.approx(0.25)
    assert rows["p=2"]["visits_ratio"] == pytest.approx(36 / 48) and rows["k=1"]["active_ratio"] == pytest.approx(1 / 8)
    assert rows["k=1"]["band"] == "small" and rows["base"]["rise"] == 0.0
    assert moetry.band(None) == "?" and moetry.band(0.05) == "moderate"
    with pytest.raises(ReshapeError, match="measure them first"):
        moetry.ask(lm, layers, batches(), "e=64", cfg_for(), None)
