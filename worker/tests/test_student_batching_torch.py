"""Student micro-batching by a token budget (`hyper.student_batch_tokens`): a padded batch of rows equals the single rows, the budget 0 is today's way, the sampler is deterministic, a retry halves the budget.
Tiny random models with GatedDeltaNet layers on a CPU (the torch reference path of the layers). The fla / Triton path with padding is covered only by the owner's GPU run."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

from test_distill_torch import base, distill_cfg, make_cache, metrics, run_in_process, teacher, thin_student  # noqa: E402,F401
from test_reshape_torch import load  # noqa: E402

from ndforge_worker import distilljob as dj, healmem, speed  # noqa: E402

LENGTHS = [3, 11, 25, 40, 7, 18]


def row(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(5, 380, (n,), generator=g)


def targets_for(rows, k=8, vocab=400, seed=5):
    """Seeded teacher answers for each real position of each row, in row order (the shapes of a real cache)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for r in rows:
        n = len(r)
        ids = torch.randint(0, vocab, (n, k), generator=g)
        logp = torch.log_softmax(torch.randn(n, k, generator=g), dim=-1) + float(np.log(0.9))
        out.append((ids, logp, torch.full((n,), 0.1)))
    return out


def padded(rows, tg, pad=0):
    n = max(len(r) for r in rows)
    ids = torch.full((len(rows), n), pad, dtype=torch.long)
    att = torch.zeros((len(rows), n), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = r
        att[i, :len(r)] = 1
    return {"input_ids": ids, "attention_mask": att, "labels": ids.clone(), "t_ids": torch.cat([t[0] for t in tg]), "t_logp": torch.cat([t[1] for t in tg]), "t_resid": torch.cat([t[2] for t in tg])}


def fn(inp):
    return inp["t_ids"], inp["t_logp"], inp["t_resid"]


def grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def test_a_padded_batch_gives_the_same_loss_and_gradients_as_the_average_of_single_rows(base):
    model = load(base)
    model.train()
    rows = [row(n, i) for i, n in enumerate(LENGTHS)]
    tg = targets_for(rows)
    # one micro-batch of all six rows, very different lengths, right-padded
    batch = padded(rows, tg)
    kd, n_rows = dj.student_loss_rows(model, batch, fn, 16, True)
    (kd / len(rows)).backward()
    g_pad, loss_pad = grads(model), float(kd) / len(rows)
    model.zero_grad()
    # the same rows one at a time, each loss divided by the number of rows (what a step of today's single-row accumulation does)
    total = 0.0
    for r, t in zip(rows, tg):
        loss, _, _ = dj.student_loss(model, padded([r], [t]), fn, 1.0, 16, True)
        (loss / len(rows)).backward()
        total += float(loss) / len(rows)
    g_one = grads(model)
    assert int(round(float(n_rows))) == len(rows)
    assert loss_pad == pytest.approx(total, abs=1e-5)
    assert set(g_pad) == set(g_one) and g_pad
    worst = max(float((g_pad[k] - g_one[k]).abs().max()) for k in g_pad)
    assert worst < 1e-5, worst


def test_the_padding_never_reaches_the_real_positions_whatever_the_pad_token(base):
    # right padding is exact for a causal model, also for the GatedDeltaNet convolution and recurrence: a row alone and the same row inside a longer padded batch have the same hidden states
    model = load(base)
    model.eval()
    short, long_ = row(5, 1), row(37, 2)
    with torch.no_grad():
        alone, _ = dj.forward_hidden(model, short.unsqueeze(0), torch.ones(1, 5, dtype=torch.long))
        for pad in (0, 7, 399):
            b = padded([short, long_], targets_for([short, long_]), pad=pad)
            both, _ = dj.forward_hidden(model, b["input_ids"], b["attention_mask"])
            assert torch.allclose(both[0, :5], alone[0], atol=1e-5), pad


def test_budget_zero_is_one_row_per_micro_batch_in_the_given_order():
    assert speed.split_by_token_budget(LENGTHS, 0) == [[i] for i in range(len(LENGTHS))]
    rows = [{"input_ids": list(range(n)), "completion_mask": [True] * n, "row": i} for i, n in enumerate(LENGTHS)]
    collate = dj._group_collator(0, None, 0, 1)
    out = collate(rows)
    assert out["n_rows"] == len(rows) and [g["input_ids"].shape[0] for g in out["groups"]] == [1] * len(rows)
    assert sorted(g["input_ids"].shape[1] for g in out["groups"]) == sorted(LENGTHS)          # every row once; the order of the micro-batches is shuffled by the seed


def test_the_split_respects_the_budget_and_keeps_every_row_once():
    lengths = [900, 30, 30, 700, 5, 5, 5, 120, 4096, 60]
    parts = speed.split_by_token_budget(lengths, 1024)
    flat = sorted(i for p in parts for i in p)
    assert flat == list(range(len(lengths)))
    for p in parts:
        longest = max(lengths[i] for i in p)
        assert len(p) == 1 or len(p) * longest <= 1024, (p, longest)
    assert [8] in parts, "a row longer than the budget is a micro-batch of its own"
    assert len(parts) < len(lengths), "rows of similar length share a forward pass"
    assert speed.padded_share(lengths, parts) < 0.5


def test_the_micro_batch_order_is_deterministic_per_seed_and_update_and_differs_between_updates():
    a = speed.micro_batches(LENGTHS * 5, 64, 7, 3)
    assert a == speed.micro_batches(LENGTHS * 5, 64, 7, 3)
    assert a != speed.micro_batches(LENGTHS * 5, 64, 7, 4) and sorted(map(sorted, a)) == sorted(map(sorted, speed.micro_batches(LENGTHS * 5, 64, 7, 4)))


def test_a_retry_halves_the_token_budget_first_and_ends_with_one_row_per_step():
    s = {"checkpointing": False, "batch": 2, "grad_accum": 16, "chunk": 128, "batch_tokens": 4096}
    labels = []
    while True:
        nxt = healmem.next_safer(s)
        if nxt is None or "batch_tokens" not in nxt[0]:
            break
        s, label = nxt
        labels.append((s["batch_tokens"], label))
    assert [b for b, _ in labels] == [2048, 1024, 512] and all("token budget" in l for _, l in labels)
    s2, last = healmem.next_safer(s)
    assert "batch_tokens" not in s2 and "one row per forward pass" in last
    nxt = healmem.next_safer(s2)
    assert nxt and nxt[1] == "gradient checkpointing", "only after the budget is used up do the other steps follow"
    assert "tokens per forward pass" in healmem.describe({"batch": 1, "grad_accum": 32, "batch_tokens": 2048})


def test_a_trainer_run_with_a_token_budget_follows_the_run_without_it(tmp_path, teacher, monkeypatch):
    student = thin_student(tmp_path, teacher)
    make_cache(tmp_path, student, teacher, monkeypatch)
    results = {}
    for name, bt in (("plain", 0), ("budget", 256)):
        hyper = {"batch": 1, "grad_accum": 4, "max_steps": 6, "seed": 3407, "save_steps": 100}
        if bt:
            hyper["student_batch_tokens"] = bt
        cfg = distill_cfg(tmp_path, student, teacher, "cache", hyper=hyper, out=str(tmp_path / f"run-{name}"))
        cfg["eval"] = None
        cfg["distill"]["eval_every"] = 0
        code, ev = run_in_process(monkeypatch, dj.run, cfg)
        assert code == 0 and ev[-1]["type"] == "result", ev[-1]
        results[name] = ([e["kd"] for e in metrics(ev)], [e for e in ev if e["type"] == "heal_settings"][0]["settings"], ev[-1]["data"])
    plain, budget = results["plain"], results["budget"]
    assert "batch_tokens" not in plain[1] and budget[1]["batch_tokens"] == 256
    assert len(plain[0]) == len(budget[0]) == 6, "the same number of updates"
    assert budget[0][0] == pytest.approx(plain[0][0], rel=1e-3), "the first update's loss is the same"
    assert budget[0] == pytest.approx(plain[0], rel=5e-3), "and the run follows (the same rows, the same weights per row, the same updates)"
    assert budget[2]["steps"] == plain[2]["steps"] == 6 and budget[2]["tokens_seen"] == plain[2]["tokens_seen"]


def test_the_speed_report_measures_real_rows_and_every_token_budget_with_the_same_first_loss(tmp_path, teacher, monkeypatch):
    import json
    from test_distill_torch import bench_run
    student = thin_student(tmp_path, teacher)
    out = tmp_path / "speed"
    cfg = distill_cfg(tmp_path, student, teacher, "cache", out=str(tmp_path / "no-output"), hyper={"batch": 1, "grad_accum": 6, "max_steps": 1})
    cfg["distill"]["chunk"] = 32
    cfg["bench"] = {"steps": 2, "warmup": 1, "targets": "rows", "rows_tokens": 20000, "out_dir": str(out), "batch_tokens_sweep": [0, 64, 256], "sweep_updates": 1, "sweep_warmup": 1}
    code, ev, res = bench_run(monkeypatch, cfg)
    assert code == 0 and res, ev[-1]
    rep = json.loads((out / "heal_speed_report.json").read_text())
    assert rep["rows"]["n"] >= 6 and rep["rows"]["max"] >= rep["rows"]["p90"] >= rep["rows"]["p50"] > 0, rep["rows"]
    sw = rep["batch_sweep"]
    assert [r["batch_tokens"] for r in sw["results"]] == [0, 64, 256] and sw["per_update_rows"] == 6
    losses = [r["first_loss"] for r in sw["results"]]
    assert all(x is not None for x in losses) and losses[1] == pytest.approx(losses[0], rel=1e-3) and losses[2] == pytest.approx(losses[0], rel=1e-3), losses
    assert sw["results"][0]["micro_batches"] == 6 and sw["results"][2]["micro_batches"] < 6, "rows of similar length share a forward pass"
    assert all(r["speedup"] for r in sw["results"]) and "value" in sw["recommended"]
    md = (out / "heal_speed_report.md").read_text()
    assert "Tokens per forward pass" in md and "The rows measured" in md


def test_the_sweep_can_repeat_the_baseline_and_the_best_and_reports_the_spread(tmp_path, teacher, monkeypatch):
    import json
    from test_distill_torch import bench_run
    student = thin_student(tmp_path, teacher)
    out = tmp_path / "speed"
    cfg = distill_cfg(tmp_path, student, teacher, "cache", out=str(tmp_path / "no-output"), hyper={"batch": 1, "grad_accum": 4, "max_steps": 1})
    cfg["distill"]["chunk"] = 32
    cfg["bench"] = {"steps": 2, "warmup": 1, "targets": "rows", "rows_tokens": 20000, "out_dir": str(out), "batch_tokens_sweep": [0, 128, 512], "sweep_updates": 1, "sweep_warmup": 1, "sweep_repeat": 2, "worst_case": True}
    code, ev, res = bench_run(monkeypatch, cfg)
    assert code == 0 and res, ev[-1]
    sw = json.loads((out / "heal_speed_report.json").read_text())["batch_sweep"]
    base = next(r for r in sw["results"] if r["batch_tokens"] == 0)
    assert len(base["repeats"]) == 2 and "spread" in sw and sw["spread"] >= 0
    assert any(len(r.get("repeats", [])) == 2 for r in sw["results"] if r["batch_tokens"] != 0), "the fastest candidate is repeated too"
    assert sw["recommended"]["spread"] == sw["spread"]
