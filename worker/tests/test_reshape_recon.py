"""The maths of refitting the kept neurons after a width cut (numpy only, no torch): the least-squares solution, the held-out error from the statistics alone, the rows."""
import pytest

np = pytest.importorskip("numpy")

from ndforge_worker import reshape_recon as rr  # noqa: E402
from ndforge_worker import slice_io  # noqa: E402

N, H, I, K = 400, 6, 10, 6          # tokens, hidden size, neurons, kept neurons


def problem(seed=1, n=N, mix=0.0):
    """activations A [n, I], the original down_proj W0 [H, I]; the first K neurons are kept. With mix > 0 the removed neurons are partly built from the kept ones."""
    r = np.random.default_rng(seed)
    a = r.standard_normal((n, I))
    a[:, K:] = (1 - mix) * a[:, K:] + mix * (a[:, :K] @ r.standard_normal((K, I - K)))
    w0 = r.standard_normal((H, I))
    return a, w0


def stats_of(a, w0, k=K):
    s = rr.new_stats(k, w0.shape[0])
    return rr.add_stats(s, a[:, :k], a[:, k:], w0[:, k:])


def test_the_solution_equals_ridge_least_squares_on_the_augmented_system():
    a, w0 = problem()
    s = stats_of(a, w0)
    lam = rr.ridge_lambda(s["G"], 0.05)
    got = rr.solve_ridge(s["G"], s["C"], lam)
    err = a[:, K:] @ w0[:, K:].T                                         # E [N, H]
    aug_a = np.vstack([a[:, :K], np.sqrt(lam) * np.eye(K)])
    aug_b = np.vstack([err, np.zeros((K, H))])
    want = np.linalg.lstsq(aug_a, aug_b, rcond=None)[0]                  # D^T [K, H]
    assert np.allclose(got, want, atol=1e-8)
    assert np.allclose(rr.solve_delta(s, 0.05), want.T, atol=1e-8)
    assert rr.solve_delta(s, 0.05).shape == (H, K)


def test_the_solver_falls_back_to_least_squares_when_the_matrix_is_singular():
    g = np.zeros((3, 3))
    g[0, 0] = 1.0                                                         # rank one, lambda 0: the Cholesky factor does not exist
    c = np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
    x = rr.solve_ridge(g, c, 0.0)
    assert np.allclose(g @ x, c, atol=1e-9) and np.isfinite(x).all()


def test_the_held_out_error_from_the_statistics_equals_the_direct_computation():
    a, w0 = problem(2, n=600)
    fit, held = stats_of(a[:300], w0), stats_of(a[300:], w0)
    d = rr.solve_delta(fit)
    direct = float(((a[300:, :K] @ d.T - a[300:, K:] @ w0[:, K:].T) ** 2).sum())
    assert rr.heldout_error(d, held) == pytest.approx(direct, rel=1e-9)
    assert rr.heldout_error(np.zeros_like(d), held) == pytest.approx(held["e2"], rel=1e-12)
    assert rr.removed_share(np.zeros_like(d), held) == pytest.approx(0.0, abs=1e-12)
    # the refit is fitted on the first half: on its own rows it can only help, on the other half it helps less
    assert rr.removed_share(d, fit) >= rr.removed_share(d, held) - 1e-12 and rr.removed_share(d, fit) > 0


def test_a_removed_neuron_that_is_a_combination_of_the_kept_ones_is_given_back_almost_entirely():
    a, w0 = problem(3, mix=1.0)                                           # every removed neuron is exactly a linear combination of the kept ones
    s = stats_of(a, w0)
    d = rr.solve_delta(s, 1e-9)
    assert rr.removed_share(d, s) == pytest.approx(1.0, abs=1e-6)
    out = rr.fit_two_halves(stats_of(a[:300], w0), stats_of(a[300:], w0), 1e-9)
    assert out["share"] == pytest.approx(1.0, abs=1e-5) and out["share_h1_to_h2"] == pytest.approx(1.0, abs=1e-5)
    # nothing in common (independent neurons): the refit gives back about nothing on rows it has not seen
    b, w1 = problem(4, mix=0.0)
    half = rr.fit_two_halves(stats_of(b[:200], w1), stats_of(b[200:], w1), 0.01)
    assert abs(half["share"]) < 0.2


def test_a_huge_penalty_shrinks_the_change_to_zero_and_a_zero_penalty_is_plain_least_squares():
    a, w0 = problem(5)
    s = stats_of(a, w0)
    assert np.abs(rr.solve_delta(s, 1e12)).max() < 1e-9
    assert np.abs(rr.solve_delta(s, 1.0)).max() < np.abs(rr.solve_delta(s, 1e-6)).max()
    plain = rr.solve_delta(s, 0.0)
    err = a[:, K:] @ w0[:, K:].T
    assert np.allclose(plain.T, np.linalg.lstsq(a[:, :K], err, rcond=None)[0], atol=1e-8)


def test_statistics_add_up_batch_by_batch_and_by_halves():
    a, w0 = problem(6, n=300)
    whole = stats_of(a, w0)
    parts = rr.new_stats(K, H)
    for lo in range(0, 300, 70):
        rr.add_stats(parts, a[lo:lo + 70, :K], a[lo:lo + 70, K:], w0[:, K:])
    assert parts["n"] == 300 and np.allclose(parts["G"], whole["G"]) and np.allclose(parts["C"], whole["C"]) and parts["e2"] == pytest.approx(whole["e2"])
    merged = rr.merge_stats(stats_of(a[:100], w0), stats_of(a[100:], w0))
    assert np.allclose(merged["G"], whole["G"]) and merged["n"] == 300


def test_layer_summary_gives_the_median_and_the_worst():
    assert rr.layer_summary([0.5, None, 0.1, 0.9]) == {"median": 0.5, "worst": 0.1, "layers": 3}
    assert rr.layer_summary([0.2, 0.4])["median"] == pytest.approx(0.3)
    assert rr.layer_summary([None]) == {"median": None, "worst": None, "layers": 0}
    assert rr.removed_share(np.zeros((H, K)), rr.new_stats(K, H)) is None, "nothing was taken away: no share"


def test_the_tail_rows_are_disjoint_from_the_head_and_the_overlap_is_counted():
    assert rr.tail_rows(100, 40, 30) == list(range(70, 100))
    assert rr.tail_rows(100, 80, 50) == list(range(80, 100)), "never into the head: fewer rows than asked for"
    assert rr.tail_rows(100, 100, 10) == [] and rr.tail_rows(0, 0, 10) == []
    head = [[1, 2, 3], [4, 5], [6]]
    tail = [[7, 8], [4, 5], [9]]
    assert rr.overlap(tail, head) == 1 and rr.overlap([[7, 8]], head) == 0 and rr.overlap([], head) == 0
    idx = rr.tail_rows(50, 20, 10)
    assert not set(idx) & set(range(20))


def test_the_tag_follows_every_input_and_nothing_else():
    base = dict(weights="dir:a", width=13312, scoring="activation", skipped_blocks=[3, 2], variant="pruned", alpha=0.01, rows_key="r1")
    t = rr.recon_tag(**base)
    assert len(t) == 16 and t == rr.recon_tag(**{**base, "skipped_blocks": [2, 3, 3], "scoring": None})
    for key, other in (("weights", "dir:b"), ("width", 12288), ("scoring", "output"), ("skipped_blocks", [2]), ("variant", "original"), ("alpha", 0.02), ("rows_key", "r2")):
        assert rr.recon_tag(**{**base, key: other}) != t, key


def test_combine_bf16_adds_in_float32_and_rounds_like_every_other_bf16_value():
    r = np.random.default_rng(7)
    w0 = slice_io.bf16_to_f32(slice_io.f32_to_bf16(r.standard_normal((H, K)).astype(np.float32)))
    d = (r.standard_normal((H, K)) * 0.01).astype(np.float32)
    bits = rr.combine_bf16(w0, d)
    assert bits.dtype == np.uint16 and bits.shape == (H, K)
    assert np.array_equal(bits, slice_io.f32_to_bf16(w0 + d))
    assert np.allclose(slice_io.bf16_to_f32(bits), w0 + d, rtol=1e-2, atol=1e-3)
    assert np.array_equal(rr.combine_bf16(w0, np.zeros_like(d)), slice_io.f32_to_bf16(w0)), "a zero change keeps the original values"
