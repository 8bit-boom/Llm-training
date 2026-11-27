"""Expert statistics, the masked router and the exactness of pruning, on a tiny stand-in with the same module convention as transformers' Inkling2Loop (torch on the CPU)."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ndforge_worker import moe_layout as ml  # noqa: E402
from ndforge_worker import moe_stats as ms  # noqa: E402

H, I, E, K, L = 8, 6, 6, 2, 4
CORE, PASSES = (1, 3), 3


class Router(nn.Module):
    def __init__(self, e, k):
        super().__init__()
        self.top_k, self.num_experts, self.hidden_dim = k, e, H
        self.weight = nn.Parameter(torch.randn(e, H))
        self.score_fn = torch.sigmoid
        self.routed_scaling_factor = 2.0
        self.e_score_correction_bias = nn.Buffer(torch.randn(e) * 0.1)

    def forward(self, x):
        flat = x.float().reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        idx = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        w = scores.gather(1, idx)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, w * self.routed_scaling_factor, idx


class Experts(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.num_experts = e
        self.gate_up_proj = nn.Parameter(torch.randn(e, 2 * I, H) * 0.5)
        self.down_proj = nn.Parameter(torch.randn(e, H, I) * 0.5)

    def _apply_gate(self, gu):
        g, u = gu.chunk(2, dim=-1)
        return F.silu(g) * u

    def forward(self, x, idx, w):
        out = torch.zeros_like(x, dtype=torch.float32)
        for e in range(self.num_experts):
            tok, slot = torch.where(idx == e)
            if tok.numel():
                y = F.linear(self._apply_gate(F.linear(x[tok], self.gate_up_proj[e])), self.down_proj[e])
                out.index_add_(0, tok, (y * w[tok, slot, None]).float())
        return out.to(x.dtype)


class Moe(nn.Module):
    def __init__(self, e, k):
        super().__init__()
        self.gate, self.experts = Router(e, k), Experts(e)

    def forward(self, x):
        b, t, h = x.shape
        flat = x.reshape(-1, h)
        _, w, idx = self.gate(flat)
        return self.experts(flat, idx, w).reshape(b, t, h)


class Layer(nn.Module):
    def __init__(self, e, k):
        super().__init__()
        self.mlp = Moe(e, k)


class Model(nn.Module):
    def __init__(self, e=E, k=K):
        super().__init__()
        self.layers = nn.ModuleList([Layer(e, k) for _ in range(L)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            for _ in range(PASSES if CORE[0] <= i < CORE[1] else 1):
                x = x + 0.5 * layer.mlp(F.layer_norm(x, (H,)))
        return x


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return Model().eval()


def data(n=3, t=11):
    g = torch.Generator().manual_seed(1)
    return [torch.randn(2, t, H, generator=g) for _ in range(n)]


def run(model, rec, batches):
    rec.attach()
    with torch.no_grad():
        for b, x in enumerate(batches):
            rec.new_batch(b % 2)
            model(x)
    rec.detach()
    return rec.result()


def test_counts_cover_every_token_visit_and_pass(model):
    batches = data()
    res = run(model, ms.Recorder(model.layers, E, CORE, PASSES), batches)
    tokens = sum(x.shape[0] * x.shape[1] for x in batches)
    for i in range(L):
        row = res["layers"][str(i)]
        times = PASSES if CORE[0] <= i < CORE[1] else 1
        assert sum(row["freq"]) == tokens * K * times                    # k experts for every token and visit
        assert len(row["per_pass_freq"]) == times and all(sum(p) == tokens * K for p in row["per_pass_freq"])
        assert sum(row["gate"]) == pytest.approx(tokens * times, rel=1e-5)         # the weights of one token add up to 1 (before the scaling factor)
        assert [a + b for a, b in zip(row["halves"]["a"]["freq"], row["halves"]["b"]["freq"])] == row["freq"]
    assert res["tokens"][0] == tokens and res["tokens"][1] == tokens * PASSES


def test_reap_is_the_mean_of_weight_times_output_size(model):
    x = data(1)[0]
    rec = ms.Recorder(model.layers[:1], E, (0, 0), 1)
    res = run(model, rec, [x])
    layer = model.layers[0].mlp
    flat = F.layer_norm(x, (H,)).reshape(-1, H)               # what the layer's MoE block is given
    with torch.no_grad():
        _, w, idx = layer.gate(flat)
    want_sum, want_n = [0.0] * E, [0] * E
    for t in range(flat.shape[0]):
        for s in range(K):
            e = int(idx[t, s])
            y = F.linear(layer.experts._apply_gate(F.linear(flat[t], layer.experts.gate_up_proj[e])), layer.experts.down_proj[e])
            want_sum[e] += float((w[t, s] / 2.0 * y.norm()).detach())
            want_n[e] += 1
    row = res["layers"]["0"]
    for e in range(E):
        assert row["sum_wn"][e] == pytest.approx(want_sum[e], rel=1e-4, abs=1e-6)
        assert row["reap"][e] == pytest.approx(want_sum[e] / want_n[e] if want_n[e] else 0.0, rel=1e-4, abs=1e-6)


def test_the_measurement_does_not_change_the_model(model):
    x = data(1)[0]
    with torch.no_grad():
        before = model(x)
    run(model, ms.Recorder(model.layers, E, CORE, PASSES), [x])
    with torch.no_grad():
        assert torch.equal(model(x), before)
    assert not model.layers[0].mlp.gate._forward_hooks and not model.layers[0].mlp.experts._forward_pre_hooks


def test_a_pruned_model_routes_exactly_like_the_original_with_the_removed_experts_switched_off(model):
    """the claim the whole cut rests on: sigmoid scores + normalisation over the chosen experts -> slicing experts, router rows and bias is the same as masking"""
    keep = {i: [0, 2, 3, 5] for i in range(L)}
    x = data(1)[0]
    imitation = ms.Imitation(model.layers, keep, E).attach()
    with torch.no_grad():
        masked = model(x)
    imitation.detach()
    small = Model(e=4, k=K).eval()
    with torch.no_grad():
        for src, dst in zip(model.layers, small.layers):
            ids = torch.tensor(keep[0])
            dst.mlp.gate.weight.copy_(src.mlp.gate.weight[ids])
            dst.mlp.gate.e_score_correction_bias.copy_(src.mlp.gate.e_score_correction_bias[ids])
            dst.mlp.experts.gate_up_proj.copy_(src.mlp.experts.gate_up_proj[ids])
            dst.mlp.experts.down_proj.copy_(src.mlp.experts.down_proj[ids])
        physical = small(x)
    assert torch.allclose(masked, physical, atol=1e-6), (masked - physical).abs().max()
    # and the masked loads are those of the physical model, only for kept experts, summing to one
    loads = imitation.loads()
    assert all(len(v) == 4 and sum(v) == pytest.approx(1.0) for v in loads.values())
    assert all(float(imitation.load[l][[1, 4]].sum()) == 0 for l in keep)


def test_keeping_everything_changes_nothing_and_a_smaller_top_k_is_imitated(model):
    x = data(1)[0]
    with torch.no_grad():
        base = model(x)
    all_keep = {i: list(range(E)) for i in range(L)}
    im = ms.Imitation(model.layers, all_keep, E).attach()
    with torch.no_grad():
        same = model(x)
    im.detach()
    assert torch.allclose(same, base, atol=1e-6)
    im = ms.Imitation(model.layers, all_keep, E, top_k=1).attach()
    with torch.no_grad():
        k1 = model(x)
    im.detach()
    assert not torch.allclose(k1, base, atol=1e-4)
    assert sum(float(v.sum()) for v in im.load.values()) == pytest.approx(x.shape[0] * x.shape[1] * (L + 2 * (CORE[1] - CORE[0])))


def test_scores_feed_the_selection(model):
    res = run(model, ms.Recorder(model.layers, E, CORE, PASSES), data())
    kept = ml.select_experts(res, 4, "reap")
    assert all(len(v) == 4 for v in kept.values())
    cov = ml.coverage(res, kept)
    assert all(0.5 < c <= 1.0 for c in cov.values())
