"""quantcheck.promotion (is the measured mixed file worth keeping over the plain file of about its size?) and the `general` test text."""
import pytest

from ndforge_worker import imatrix, quantcheck


def row(i, label, size_gb, kld, unc=0.0005, mixed=None):
    r = {"id": str(i), "label": label, "size": int(size_gb * 1e9), "status": "done", "mean_kld": kld, "mixed": mixed, "group": label, "matrix": True}
    if unc is not None:
        r["kld_unc"] = unc
    return r


def test_clearly_better_is_promoted_and_says_how_it_compares_with_the_next_bigger_file():
    rows = [row(0, "Q4_K_M", 10.6, 0.0400), row(1, "Q5_K_M", 12.4, 0.0150), row(2, "Mixed", 10.6, 0.0300, mixed="mixed_measured")]
    (p,) = quantcheck.promotion(rows)
    assert p["id"] == "2" and p["against"] == "0" and p["promoted"] is True and p["delta_pct"] == 25.0 and p["noise"] == pytest.approx(0.000707, abs=1e-6)
    assert "25.0 % closer" in p["reason"] and "worth keeping" in p["reason"]
    assert "Q5_K_M is 1.80 GB larger and 50 % closer" in p["reason"]


def test_a_difference_inside_the_noise_is_a_tie_and_is_not_promoted():
    rows = [row(0, "Q4_K_M", 10.6, 0.0400, 0.002), row(1, "Mixed", 10.6, 0.0390, mixed="mixed_measured", unc=0.002)]
    (p,) = quantcheck.promotion(rows)
    assert p["promoted"] is False and "tie within measurement noise" in p["reason"]


def test_worse_is_not_promoted_and_says_so():
    rows = [row(0, "Q4_K_M", 10.6, 0.0300), row(1, "Mixed", 10.6, 0.0400, mixed="mixed_measured")]
    (p,) = quantcheck.promotion(rows)
    assert p["promoted"] is False and p["delta_pct"] < 0 and "further from the 16-bit model" in p["reason"] and "plain file is better" in p["reason"]


def test_the_plain_file_is_the_largest_one_not_bigger_than_one_percent_above():
    rows = [row(0, "Q4_K_M", 10.0, 0.05), row(1, "Q5_K_M", 10.9, 0.03), row(2, "Mixed", 10.95, 0.025, mixed="mixed_measured"), row(3, "Q6_K", 13.0, 0.01)]
    (p,) = quantcheck.promotion(rows)
    assert p["against"] == "1" and p["promoted"] is True            # 10.9 <= 1.01 x 10.95


def test_no_plain_file_of_equal_or_smaller_size_compares_with_the_smallest_plain_file():
    rows = [row(0, "Q4_K_M", 10.6, 0.0400), row(1, "Mixed", 9.0, 0.0600, mixed="mixed_measured")]
    (p,) = quantcheck.promotion(rows)
    assert p["against"] == "0" and p["promoted"] is False and "further from" in p["reason"]


def test_a_missing_plus_minus_figure_counts_as_zero_noise_and_says_so():
    rows = [row(0, "Q4_K_M", 10.6, 0.0400, None), row(1, "Mixed", 10.6, 0.0399, mixed="mixed_measured", unc=None)]
    (p,) = quantcheck.promotion(rows)
    assert p["promoted"] is True and p["noise"] is None and "no plus-minus figure" in p["reason"]


def test_without_a_mixed_row_or_a_plain_row_there_is_nothing_to_say_and_unfinished_rows_are_skipped():
    assert quantcheck.promotion([row(0, "Q4_K_M", 10.6, 0.04)]) == []
    assert quantcheck.promotion([row(0, "Mixed", 10.6, 0.04, mixed="mixed_measured")]) == []
    unfinished = {**row(1, "Mixed", 10.6, 0.01, mixed="mixed_measured"), "status": "error"}
    assert quantcheck.promotion([row(0, "Q4_K_M", 10.6, 0.04), unfinished]) == []


def test_the_verdict_has_the_sentence_and_the_summary_the_key():
    rows = [row(0, "Q4_K_M", 10.6, 0.04), row(1, "Mixed", 10.6, 0.03, mixed="mixed_measured")]
    v = quantcheck.verdict(rows)
    assert any("worth keeping" in s for s in v) and v[-1].startswith("KL divergence is a proxy")
    res = quantcheck._summary({"rows": rows})
    assert res["promotion"][0]["promoted"] is True


def test_the_general_text_kind_builds_chunks_from_a_stubbed_source(monkeypatch):
    class Tok:
        pass
    monkeypatch.setattr(imatrix, "load_tokenizer", lambda sources: Tok())
    monkeypatch.setattr(imatrix, "count_tokens", lambda tok, text: len(text.split()))
    seen = {}

    def fake(source, tokenizer, budget, ctx, should_stop=None, fam_name=None, seed=3407, report=None):
        seen.update(source=source, budget=budget, ctx=ctx, seed=seed, fam=fam_name)
        return [" ".join(f"w{i}" for i in range(400)) for _ in range(3)], ""
    monkeypatch.setattr(imatrix, "fetch_general", fake)
    notes = []
    text, desc, used, chunks = quantcheck.build_text({"kind": "general", "source": {"id": "x/y"}, "tokenizer": ["m"], "seed": 1234, "family": "qwen3_5"}, 64, 8, notes)
    assert seen["seed"] == 1234 and seen["source"] == {"id": "x/y"} and seen["fam"] == "qwen3_5" and seen["budget"] == int(64 * 8 * quantcheck.SLACK) + 1
    assert used == 3 and chunks == 8 and "general text" in desc and text.count("w0") == 3
    assert any("overlap" in n for n in notes)
    notes.clear()
    quantcheck.build_text({"kind": "general", "tokenizer": ["m"]}, 512, 8, notes)
    assert seen["seed"] == quantcheck.SEED == 1234 and any("short" in n for n in notes)


def test_the_general_text_refuses_in_plain_words(monkeypatch):
    monkeypatch.setattr(imatrix, "load_tokenizer", lambda sources: object())
    monkeypatch.setattr(imatrix, "fetch_general", lambda *a, **k: ([], "The general text (x) had nothing to read."))
    with pytest.raises(quantcheck.QuantCheckError, match="nothing to read. Choose a text file"):
        quantcheck.build_text({"kind": "general", "tokenizer": ["m"]}, 64, 8, [])
    monkeypatch.setattr(imatrix, "fetch_general", lambda *a, **k: (["one two"], ""))
    monkeypatch.setattr(imatrix, "count_tokens", lambda tok, text: 2)
    with pytest.raises(quantcheck.QuantCheckError, match="too little general text"):
        quantcheck.build_text({"kind": "general", "tokenizer": ["m"]}, 64, 8, [])
    monkeypatch.setattr(imatrix, "load_tokenizer", lambda sources: (_ for _ in ()).throw(RuntimeError("no")))
    with pytest.raises(quantcheck.QuantCheckError, match="tokenizer"):
        quantcheck.build_text({"kind": "general"}, 64, 8, [])
    with pytest.raises(quantcheck.QuantCheckError, match="Choose the test text"):
        quantcheck.build_text({"kind": "weird"}, 64, 8, [])
