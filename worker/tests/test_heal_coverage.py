"""What the rows of a heal cover: complete / cut / left out conversations, reasoning rows, token shares, lengths (no torch, no tokenizer files)."""
import pytest

pytest.importorskip("numpy")

from ndforge_worker import data as data_mod  # noqa: E402
from ndforge_worker import distilljob as dj  # noqa: E402

START, ASSIST, NL, END, OPEN, CLOSE = 1, 2, 3, 4, 5, 6
MARK = {"assistant": [START, ASSIST, NL], "end": END, "think_open": OPEN, "think_close": CLOSE}
Q = [START, 7, NL, 8, 8, 8, END, NL]                 # a user turn, 8 tokens


def chat(trace=None, answer=4, closed=True, cut_at=None):
    """a general-style chat row: user turn, assistant marker, optional reasoning block, answer, end marker"""
    ids = list(Q) + [START, ASSIST, NL]
    if trace is not None:
        ids += [OPEN] + [9] * trace + [CLOSE]
    ids += [10] * answer
    if closed:
        ids += [END, NL]
    if cut_at is not None:
        ids = ids[:cut_at]
    return {"input_ids": ids, "completion_mask": [1] * len(ids), "kind": "general", "source": "general"}


def project_chat(answer=4, prompt=6, closed=True):
    ids = [START, 7, NL, 8, 8, 8][:prompt] + [10] * answer + ([END, NL] if closed else [])
    return {"input_ids": ids, "completion_mask": [0] * prompt + [1] * (len(ids) - prompt), "kind": "chat", "source": "project"}


def text(n):
    return {"input_ids": [11] * n, "completion_mask": [1] * n, "kind": "text", "source": "project"}


def test_the_row_kinds_and_the_complete_cut_counts():
    rows = [text(10), project_chat(), project_chat(closed=False), chat(trace=20), chat(trace=20, closed=False)]
    c = dj.coverage(rows, MARK, 4096, {}, general_chat=True)
    assert c["rows"] == {"project_text": 1, "project_chat": 2, "general_text": 0, "general_chat": 2}
    assert c["chat_rows"]["complete"] == 2 and c["chat_rows"]["cut"] == 2


def test_reasoning_rows_complete_cut_in_the_answer_and_cut_in_the_reasoning():
    whole = chat(trace=30, answer=5)
    cut_answer = chat(trace=30, answer=5, closed=False)               # the answer was cut: no end marker
    n_prompt = len(Q) + 3
    cut_reason = chat(trace=30, answer=5, closed=False, cut_at=n_prompt + 1 + 10)       # ends inside the reasoning: no </think>
    empty = chat(trace=1, answer=5)                                   # an empty reasoning block of a template with thinking off
    c = dj.coverage([whole, cut_answer, cut_reason, empty], MARK, 4096, {}, general_chat=True)
    r = c["reasoning_rows"]
    assert (r["complete"], r["cut_answer"], r["cut_reasoning"]) == (1, 1, 1), r
    assert r["known"] is True and r["total"] == 3
    assert c["chat_rows"]["complete"] == 2 and c["chat_rows"]["cut"] == 2


def test_token_shares_sum_to_one_and_split_the_row():
    row = chat(trace=30, answer=5)
    one = dj.classify_row(row, MARK, 4096, True)
    assert one["prompt"] == len(Q) + 3 and one["reasoning"] == 32 and one["answer"] == 5 + 2 and one["think"] == "complete"
    assert one["prompt"] + one["reasoning"] + one["answer"] == len(row["input_ids"])
    c = dj.coverage([row, text(40), project_chat()], MARK, 4096, {}, general_chat=True)
    assert sum(c["token_share"].values()) == pytest.approx(1.0, abs=1e-3)
    assert c["token_share"]["plain"] > 0 and c["token_share"]["reasoning"] > 0


def test_a_project_chat_uses_its_completion_mask_when_there_is_no_assistant_marker():
    row = project_chat(answer=6, prompt=4)
    c = dj.classify_row(row, {"assistant": None, "end": END, "think_open": None, "think_close": None}, 4096, True)
    assert (c["prompt"], c["answer"], c["reasoning"]) == (4, 6 + 2, 0)


def test_a_family_without_thinking_markers_reports_prompt_answer_and_complete_cut_only():
    markers = {"assistant": [START, ASSIST, NL], "end": None, "think_open": None, "think_close": None}
    short, long = chat(answer=4), chat(answer=100)
    c = dj.coverage([short, long], markers, 40, {}, general_chat=True)         # no end marker known: shorter than max_seq = complete
    assert c["chat_rows"] == {"complete": 1, "cut": 1, "skipped": 0, "skipped_by": {"project_chat": 0, "general_chat": 0}}
    assert c["reasoning_rows"]["known"] is False and c["reasoning_rows"]["total"] == 0
    assert c["token_share"]["reasoning"] == 0.0 and c["notes"] == []


def test_left_out_rows_are_counted_and_the_histogram_includes_them():
    buckets = {label: 0 for label in data_mod.LENGTH_LABELS}
    buckets["<=512"], buckets[">16k"] = 7, 2
    stats = {"skipped_too_long": 2, "general_skipped_long": 3, "length_buckets": buckets, "skipped_reasoning": 1, "general_skipped_reasoning": 2}
    c = dj.coverage([chat(trace=20)], MARK, 4096, stats, general_chat=True)
    assert c["chat_rows"]["skipped"] == 5 and c["chat_rows"]["skipped_by"] == {"project_chat": 2, "general_chat": 3}
    assert c["reasoning_rows"]["skipped"] == 3 and c["reasoning_rows"]["total"] == 4
    assert c["length_buckets"][">16k"] == 2 and c["length_buckets"]["<=512"] == 7
    stats["general_length_buckets"] = {"<=512": 1, "<=2k": 4}
    merged = dj.coverage([], MARK, 4096, stats)["length_buckets"]
    assert merged["<=512"] == 8 and merged["<=2k"] == 4 and list(merged) == [k for k in data_mod.LENGTH_LABELS if k in merged]


def lost_rows(lost, whole):
    rows = [chat(trace=30) for _ in range(whole)] + [chat(trace=30, closed=False) for _ in range(lost)]
    return rows


def test_the_note_appears_above_ten_percent_with_the_right_advice():
    rows = lost_rows(2, 8)                                                # 20% lost
    low = dj.coverage(rows, MARK, 4096, {}, general_chat=True)
    assert any("choose 8192 in Heal settings if the plan says it fits" in n for n in low["notes"]), low["notes"]
    high = dj.coverage(rows, MARK, 8192, {}, general_chat=True)
    assert any("longer than 8192" in n and "not available yet" in n for n in high["notes"]), high["notes"]
    assert not any("choose 8192" in n for n in high["notes"])
    few = dj.coverage(lost_rows(1, 19), MARK, 4096, {}, general_chat=True)     # 5%
    assert not any("8192" in n for n in few["notes"])


def test_the_reasoning_share_is_guidance_not_a_rule():
    rows = [chat(trace=3, answer=200) for _ in range(5)]                  # a little reasoning, a lot of answer
    c = dj.coverage(rows, MARK, 4096, {}, general_chat=True)
    assert any("75%" in n and "not a rule" in n for n in c["notes"]), c["notes"]


def test_length_buckets():
    assert [data_mod.length_bucket(n) for n in (1, 512, 513, 1024, 2049, 4096, 8193, 16384, 16385)] == ["<=512", "<=512", "<=1k", "<=1k", "<=4k", "<=4k", "<=16k", "<=16k", ">16k"]


class Inner:
    unk_token_id = 0
    eos_token_id = 99
    TOKENS = {"<|im_start|>": 1, "<|im_end|>": 4, "<think>": 5, "</think>": 6}

    def convert_tokens_to_ids(self, t):
        return self.TOKENS.get(t, 0)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1, 2, 3] if text.startswith("<|im_start|>assistant") else [7]}


def test_marker_ids_from_a_tokenizer_with_and_without_the_markers():
    m = dj.marker_ids(Inner(), "qwen3_5")
    assert m == {"assistant": [1, 2, 3], "end": 4, "think_open": 5, "think_close": 6}
    probe = dj.reasoning_probe(m)
    assert probe([1, 5, 9, 9, 9, 6, 7]) and probe([1, 5, 9]) and not probe([1, 5, 9, 6]) and not probe([1, 7])
    bare = Inner()
    bare.TOKENS = {}
    m2 = dj.marker_ids(bare, "generic")
    assert m2["assistant"] is None and m2["think_open"] is None and m2["end"] == 99
    assert dj.reasoning_probe(m2) is None


def test_encode_rows_counts_lengths_and_left_out_reasoning_rows():
    class Tok:
        bos_token = None

        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [ord(c) % 250 + 1 for c in text]}

        def apply_chat_template(self, *a, **k):          # never used: text rows only
            raise AssertionError

    rows = [{"type": "text", "text": "a" * 100}, {"type": "text", "text": "b" * 700}]
    enc, st = data_mod.encode_rows(Tok(), rows, "generic", 1000, 3407, probe=lambda ids: True)
    assert len(enc) == 2 and st["length_buckets"]["<=512"] == 1 and st["length_buckets"]["<=1k"] == 1
    assert st["skipped_reasoning"] == 0
