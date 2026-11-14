"""General text for healing and the importance matrix: formats, detection, chat rendering, the Hub stand-in, plain failure messages, keys, the token (no network)."""
import csv
import json
from pathlib import Path

import pytest

from ndforge_worker import data as data_mod
from ndforge_worker import generaltext as gt
from ndforge_worker import imatrix, protocol


class WordTok:
    """one token per word; ChatML template; special tokens count as words"""
    eos_token = "<|im_end|>"

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(text.split())))}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def render(msgs):
    return data_mod.render_chat(WordTok(), msgs, "generic")[1]


def jl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def read(source, **kw):
    rep = {}
    out = list(gt.texts({**source, "report": rep, **kw}))
    return out, rep


CHAT = [{"role": "user", "content": "What is two plus two?"}, {"role": "assistant", "content": "Four."}]
SHARE = [{"from": "system", "value": "Be brief."}, {"from": "human", "value": "Hi there"}, {"from": "gpt", "value": "Hello!"}]


def test_text_column_is_found_without_naming_it(tmp_path):
    f = jl(tmp_path / "a.jsonl", [{"id": 1, "text": "alpha beta gamma"}, {"id": 2, "text": "delta"}])
    out, rep = read({"file": f})
    assert out == ["alpha beta gamma", "delta"] and rep["detected"] == {"kind": "text", "column": "text"} and rep["rows_read"] == 2


@pytest.mark.parametrize("col,rows,expect", [
    ("messages", [{"messages": CHAT, "source": "x"}], "messages"),
    ("conversations", [{"conversations": SHARE, "source": "x"}], "conversations"),
])
def test_chat_columns_are_rendered_with_the_model_template(tmp_path, col, rows, expect):
    out, rep = read({"file": jl(tmp_path / "c.jsonl", rows), "render": render})
    assert rep["detected"]["kind"] == expect and rep["detected"]["column"] == col
    assert out[0].startswith("<|im_start|>") and out[0].rstrip().endswith("<|im_end|>")
    if expect == "conversations":                      # human/gpt/system mapped to user/assistant/system
        assert "<|im_start|>system\nBe brief.<|im_end|>" in out[0] and "<|im_start|>user\nHi there<|im_end|>" in out[0] and out[0].rstrip().endswith("Hello!<|im_end|>")


@pytest.mark.parametrize("ans", ["completion", "response", "output", "answer"])
def test_prompt_and_completion_pairs(tmp_path, ans):
    out, rep = read({"file": jl(tmp_path / "p.jsonl", [{"prompt": "Name a colour", ans: "Red"}]), "render": render})
    assert rep["detected"] == {"kind": "pair", "column": "prompt", "answer": ans}
    assert "<|im_start|>user\nName a colour<|im_end|>" in out[0] and "<|im_start|>assistant\nRed<|im_end|>" in out[0]


def test_chat_without_a_renderer_is_joined_as_plain_lines(tmp_path):
    out, _ = read({"file": jl(tmp_path / "c.jsonl", [{"messages": CHAT}])})
    assert out == ["What is two plus two?\nFour."]


def test_conversations_that_end_with_the_user_are_cut_and_unusable_ones_skipped(tmp_path):
    rows = [{"messages": CHAT + [{"role": "user", "content": "and?"}]}, {"messages": [{"role": "assistant", "content": "alone"}]}, {"messages": [{"role": "tool", "content": "x"}]}]
    out, rep = read({"file": jl(tmp_path / "c.jsonl", rows), "render": render, "column": "messages"})
    assert len(out) == 1 and rep["skipped"] == 2 and out[0].count("<|im_start|>") == 2


@pytest.mark.parametrize("ext", ["jsonl", "json", "txt", "csv"])
def test_every_local_format_gives_the_same_texts(tmp_path, ext):
    docs = ["first document about dragons", "second document about ships"]
    p = tmp_path / f"d.{ext}"
    if ext == "jsonl":
        jl(p, [{"text": d} for d in docs])
    elif ext == "json":
        p.write_text(json.dumps({"data": [{"text": d} for d in docs]}), encoding="utf-8")
    elif ext == "txt":
        p.write_text("\n\n".join(docs) + "\n", encoding="utf-8")
    else:
        with open(p, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "text"])
            for i, d in enumerate(docs):
                w.writerow([i, d])
    out, rep = read({"file": str(p)})
    assert out == docs and rep["detected"]["column"] == "text"


def test_json_list_and_json_lines_in_a_json_file(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps([{"messages": CHAT}]), encoding="utf-8")
    b = tmp_path / "b.json"
    b.write_text('{"text": "one line of text"}\n{"text": "two"}\n', encoding="utf-8")
    assert len(read({"file": str(a)})[0]) == 1 and read({"file": str(b)})[0] == ["one line of text", "two"]


def test_parquet_when_pyarrow_is_there(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.table({"text": ["parquet one", "parquet two"]}), str(tmp_path / "d.parquet"))
    assert read({"file": str(tmp_path / "d.parquet")})[0] == ["parquet one", "parquet two"]


def test_a_named_column_is_used_and_a_wrong_one_is_a_plain_error(tmp_path):
    f = jl(tmp_path / "a.jsonl", [{"body": "named column text", "text": "other"}])
    assert read({"file": f, "column": "body"})[0] == ["named column text"]
    with pytest.raises(gt.GeneralTextError) as e:
        read({"file": f, "column": "nope"})
    assert e.value.code == "no_column" and "'nope'" in e.value.plain and "body" in e.value.plain


def test_nothing_usable_is_no_usable_column(tmp_path):
    f = jl(tmp_path / "n.jsonl", [{"id": 1, "n": 2}, {"id": 3, "n": 4}])
    with pytest.raises(gt.GeneralTextError) as e:
        read({"file": f})
    assert e.value.code == "no_column"
    with pytest.raises(gt.GeneralTextError) as e2:
        read({"file": jl(tmp_path / "e.jsonl", [])})
    assert e2.value.code == "empty"


def test_a_missing_or_broken_file_has_plain_messages(tmp_path):
    with pytest.raises(gt.GeneralTextError) as e:
        read({"file": str(tmp_path / "gone.jsonl")})
    assert e.value.code == "not_found" and "gone.jsonl" in e.value.plain
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"text": "ok"}\n{not json\n', encoding="utf-8")
    with pytest.raises(gt.GeneralTextError) as e2:
        read({"file": str(bad)})
    assert "Line 2" in e2.value.plain


def test_same_seed_same_sample_other_seed_other_order(tmp_path):
    f = jl(tmp_path / "s.jsonl", [{"text": f"document number {i} here"} for i in range(60)])
    a, b = read({"file": f, "seed": 7})[0], read({"file": f, "seed": 7})[0]
    c = read({"file": f, "seed": 8})[0]
    plain = read({"file": f})[0]
    assert a == b and a != c and sorted(a) == sorted(plain) and a != plain
    assert list(gt.buffer_shuffle(range(10_000), 1, 100)) == list(gt.buffer_shuffle(range(10_000), 1, 100))
    assert sorted(gt.buffer_shuffle(range(500), 1, 100)) == list(range(500))


# ------------------------------------------------------------------------------------------------------------ the two reasoning presets

def test_inline_think_in_messages_is_left_alone(tmp_path):
    msgs = [{"role": "user", "content": "Solve it"}, {"role": "assistant", "content": "<think>\nhmm\n</think>\n\nThe answer is 4."}]
    out, _ = read({"file": jl(tmp_path / "r.jsonl", [{"messages": msgs, "domain": "math", "tokens_think": 3}]), "render": render})
    assert out[0].count("<think>") == 1 and "<think>\nhmm\n</think>\n\nThe answer is 4.<|im_end|>" in out[0]
    no_think = read({"file": jl(tmp_path / "r2.jsonl", [{"messages": CHAT}]), "render": render})[0]
    assert "<think>" not in no_think[0]


def test_reasoning_content_goes_inline_once_and_tools_are_ignored(tmp_path):
    rows = [
        {"messages": [{"role": "user", "content": "Q?", "trainable": False, "reasoning_content": None, "tool_calls": None},
                      {"role": "assistant", "content": "A.", "reasoning_content": "step one", "tool_calls": [], "trainable": True}], "tools": [], "sampling_weight": 1.0},
        {"messages": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "<think>x</think>A.", "reasoning_content": "already inline"}]},
        {"messages": [{"role": "user", "content": "Weather?"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "w"}}]},
                      {"role": "tool", "content": "sunny", "tool_call_id": "1", "name": "w"}, {"role": "assistant", "content": "Sunny."}]},
    ]
    out, rep = read({"file": jl(tmp_path / "m.jsonl", rows), "render": render})
    assert len(out) == 2 and rep["skipped"] == 1                                    # the tool-using row is skipped
    assert "<|im_start|>assistant\n<think>\nstep one\n</think>\n\nA.<|im_end|>" in out[0]
    assert out[1].count("<think>") == 1 and "already inline" not in out[1]


# ------------------------------------------------------------------------------------------------------------ the Hub stand-in and plain failures

def loader_of(rows=None, exc=None):
    def load(source, token):
        load.seen.append((dict(source), token))
        if exc:
            raise exc
        return iter(rows or [])
    load.seen = []
    return load


def test_a_hub_dataset_goes_through_the_injected_loader_with_the_token_from_the_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_" + "a" * 34)
    ld = loader_of([{"conversations": SHARE}])
    out, rep = read({"id": "org/set", "config": "c1", "split": "train", "loader": ld})
    assert len(out) == 1 and ld.seen[0][0]["config"] == "c1" and ld.seen[0][1] == "hf_" + "a" * 34 and rep["detected"]["kind"] == "conversations"
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    read({"id": "org/set", "loader": ld})
    assert ld.seen[1][1] is None


GATED = "Dataset 'nvidia/Nemotron-Pretraining-SFT-v1' is a gated dataset on the Hub. Visit the dataset page at https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1 to ask for access."


@pytest.mark.parametrize("exc,code,words", [
    (RuntimeError(GATED), "gated", ["needs access approval on Hugging Face (https://huggingface.co/datasets/org/set)", "add your Hugging Face token, or choose another dataset"]),
    (RuntimeError("401 Client Error: Unauthorized for url"), "gated", ["access approval"]),
    (type("DatasetNotFoundError", (Exception,), {})("Dataset 'org/set' doesn't exist on the Hub"), "not_found", ["has no dataset called 'org/set'"]),
    (FileNotFoundError("x"), "not_found", ["no dataset"]),
    (ConnectionError("Couldn't reach 'huggingface.co' (offline)"), "offline", ["could not be reached", "huggingface.co"]),
    (OSError("Network is unreachable"), "offline", ["internet connection"]),
    (ValueError("Unknown split \"dev\". Should be one of ['train']"), "bad_config", ["config or split"]),
    (ModuleNotFoundError("No module named 'datasets'"), "other", ["no 'datasets' library"]),
])
def test_explain_gives_each_failure_its_own_plain_message(exc, code, words):
    e = gt.explain(exc, {"id": "org/set"})
    assert e.code == code and all(w in e.plain for w in words), e.plain


def test_check_says_what_happened_for_every_verdict(tmp_path):
    ok = gt.check({"id": "org/set", "loader": loader_of([{"messages": CHAT}] * 30)})
    assert ok["verdict"] == "works" and ok["texts"] == 20 and ok["detected"]["kind"] == "messages" and ok["sample"][0] == "What is two plus two?\nFour."
    assert "chat template" in ok["message"]
    assert gt.check({"id": "o/s", "loader": loader_of(exc=RuntimeError(GATED))})["verdict"] == "needs_access"
    assert gt.check({"id": "o/s", "loader": loader_of(exc=RuntimeError("404 Not Found"))})["verdict"] == "not_found"
    assert gt.check({"id": "o/s", "loader": loader_of(exc=ConnectionError("offline"))})["verdict"] == "offline"
    assert gt.check({"id": "o/s", "loader": loader_of([{"a": 1}, {"a": 2}])})["verdict"] == "no_column"
    assert gt.check({"id": "o/s", "loader": loader_of([])})["verdict"] == "empty"
    f = jl(tmp_path / "t.jsonl", [{"text": "hello world again"}])
    assert gt.check({"file": f})["verdict"] == "works" and gt.check({"file": str(tmp_path / "x.txt")})["verdict"] == "not_found"


def test_the_check_job_answers_through_the_protocol(monkeypatch, tmp_path):
    out = []
    monkeypatch.setattr(protocol, "emit", lambda kind, **f: out.append((kind, f)))
    monkeypatch.setattr(gt, "hub_loader", lambda source, token: iter([{"text": "some general text"}]))
    assert gt.run_check({"source": {"id": "org/set"}, "rows": 5}) == 0
    assert out[-1][0] == "result" and out[-1][1]["data"]["verdict"] == "works"


# ------------------------------------------------------------------------------------------------------------ build_rows and the fallback message

def patch_project(monkeypatch, n=3, length=20):
    from ndforge_worker import distilljob as dj
    rows = [{"input_ids": list(range(1, length + 1)), "completion_mask": [1] * length, "kind": "chat"} for _ in range(n)]
    monkeypatch.setattr(data_mod, "load_training_rows", lambda tok, path, fam, max_len, seed: (rows, {"tokens": n * length}))
    return dj


def test_build_rows_renders_chat_with_the_model_template_and_names_what_it_read(tmp_path, monkeypatch):
    dj = patch_project(monkeypatch)
    f = jl(tmp_path / "g.jsonl", [{"conversations": [{"from": "human", "value": "word " * 20}, {"from": "gpt", "value": "reply " * 20}]} for _ in range(10)])
    rows, st = dj.build_rows(WordTok(), {"train": "x", "tokens": 100, "general": {"share": 0.5, "file": f}}, "generic", 64)
    assert st["general_error"] is None and st["general_rows"] > 0 and st["general_detected"]["kind"] == "conversations" and st["general_source"] == f
    assert dj.DEFAULT_GENERAL == "mlabonne/open-perfectblend"


@pytest.mark.parametrize("exc,code,words", [
    (RuntimeError(GATED), "gated", ["could not be downloaded", "needs access approval", "use mlabonne/open-perfectblend instead", "your world's text only"]),
    (RuntimeError("404 Not Found"), "not_found", ["could not be downloaded", "has no dataset"]),
    (ConnectionError("offline"), "offline", ["internet connection"]),
])
def test_build_rows_falls_back_with_a_plain_note_and_the_default_hint(monkeypatch, exc, code, words):
    dj = patch_project(monkeypatch)
    monkeypatch.setattr(gt, "hub_loader", loader_of(exc=exc))
    rows, st = dj.build_rows(WordTok(), {"train": "x", "tokens": 100, "general": {"share": 0.5, "id": "org/set"}}, "generic", 64)
    assert len(rows) == 3 and st["general_error_code"] == code and all(w in st["general_error"] for w in words), st["general_error"]
    _, st2 = dj.build_rows(WordTok(), {"train": "x", "tokens": 100, "general": {"share": 0.5, "id": dj.DEFAULT_GENERAL}}, "generic", 64)
    assert "instead" not in st2["general_error"]


def test_no_usable_column_is_its_own_note(monkeypatch):
    dj = patch_project(monkeypatch)
    monkeypatch.setattr(gt, "hub_loader", loader_of([{"a": 1}] * 3))
    _, st = dj.build_rows(WordTok(), {"train": "x", "tokens": 100, "general": {"share": 0.5, "id": "org/set"}}, "generic", 64)
    assert st["general_error_code"] == "no_column" and "No column with usable text" in st["general_error"]


# ------------------------------------------------------------------------------------------------------------ keys

def test_general_key_follows_every_part_of_the_dataset():
    base = {"id": "a/b", "config": "c", "split": "train", "column": None}
    k = gt.general_key(base, 0.5, 3407)
    assert k == {"used": True, "id": "a/b", "file": None, "config": "c", "split": "train", "column": None, "share": 0.5, "seed": 3407}
    for change in ({"id": "a/c"}, {"config": "d"}, {"split": "test"}, {"column": "text"}, {"file": "x.jsonl"}):
        assert gt.general_key({**base, **change}, 0.5, 3407) != k
    assert gt.general_key(base, 0.4, 3407) != k and gt.general_key(base, 0.5, 1) != k
    assert gt.general_key(None, 0.5, 1) == gt.general_key(base, 0, 1) == {"used": False}
    assert gt.general_key({}, 0.5, 1) == {"used": False}


def test_teacher_cache_of_another_dataset_or_an_old_cache_is_other_data():
    from ndforge_worker import distilljob as dj
    meta = {"teacher": "t", "k": 8, "rows": 10, "data_hash": "h", "tokenizer_hash": "x", "max_len": 64, "general_rows": 4,
            "general_key": gt.general_key({"id": "a/b"}, 0.5, 3407)}
    assert dj.cache_matches(dict(meta), meta)
    assert not dj.cache_matches({**meta, "general_key": gt.general_key({"id": "a/c"}, 0.5, 3407)}, meta)
    old = {k: v for k, v in meta.items() if k != "general_key"}
    assert not dj.cache_matches(old, meta)
    assert not dj.cache_matches({**meta, "general_rows": 5}, meta)


def test_a_saved_matrix_is_not_reused_for_another_dataset(tmp_path):
    cfg = {"adapter": str(tmp_path / "none")}
    im = {"model": "r", "chunks": 64, "ctx": 128, "general_share": 0.2, "general": {"id": "a/b"}}
    k1 = imatrix.settings_key(cfg, im, "h")
    k2 = imatrix.settings_key(cfg, {**im, "general": {"id": "a/c"}}, "h")
    k3 = imatrix.settings_key(cfg, {**im, "general": {"id": "a/b", "config": "x"}}, "h")
    assert k1["general"] != k2["general"] and k1["general"] != k3["general"] and k1 != k2
    d = tmp_path / "saved"
    d.mkdir()
    (d / "imatrix.gguf").write_bytes(b"x")
    (d / "imatrix.json").write_text(json.dumps({"format": imatrix.FORMAT, **k1, "general_used": True}), encoding="utf-8")
    assert imatrix.find_saved([d], k1, True)[0] == d and imatrix.find_saved([d], k2, True)[0] is None
    old = {k: v for k, v in k1.items() if k != "general"}
    (d / "imatrix.json").write_text(json.dumps({"format": imatrix.FORMAT, **old, "general_used": True}), encoding="utf-8")
    assert imatrix.find_saved([d], k1, True)[0] is None


# ------------------------------------------------------------------------------------------------------------ the token

def test_tokens_are_scrubbed_from_everything_the_worker_prints(monkeypatch):
    import io
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    tok = "hf_" + "Ab12" * 9
    monkeypatch.setenv("HF_TOKEN", tok)
    assert tok not in protocol.scrub(f"Bearer {tok} failed") and "[token]" in protocol.scrub(f"Bearer {tok} failed")
    monkeypatch.setenv("HF_TOKEN", "custom-secret-value")
    assert protocol.scrub("x custom-secret-value y") == "x [token] y"
    protocol.log(f"401 for {tok}")
    protocol.error("e", f"bad {tok}")
    err = buf.getvalue()
    assert tok not in err and "[token]" in err
    assert "hf_short" in protocol.scrub("hf_short")


def test_failure_notes_never_contain_the_token(monkeypatch):
    tok = "hf_" + "Zz9" * 12
    monkeypatch.setenv("HF_TOKEN", tok)
    note, code = gt.failure_note(RuntimeError(f"weird failure with {tok}"), {"id": "o/s"}, "downloaded", "tail.")
    assert tok not in note and code == "other"
    gated = gt.explain(RuntimeError(GATED), {"id": "o/s"})
    assert "token was sent" in gated.plain and tok not in gated.plain


def test_a_cache_is_judged_by_its_dataset_not_by_the_training_runs_seed():
    """The teacher cache holds its own rows: changing the training seed (Expert mode) must not make a good cache 'other data'; changing the dataset or the share must."""
    from ndforge_worker import generaltext as gt
    src = {"id": "acme/blend", "config": "default", "split": "train"}
    made = gt.general_key(src, 0.5, 3407)
    assert gt.same_general(made, src, 0.5)                                    # the same dataset: fine, whatever seed the run uses
    assert not gt.same_general(made, {**src, "id": "acme/other"}, 0.5)        # another dataset
    assert not gt.same_general(made, {**src, "split": "validation"}, 0.5)     # another split
    assert not gt.same_general(made, {**src, "column": "text"}, 0.5)          # another column
    assert not gt.same_general(made, src, 0.25)                               # another share
    assert not gt.same_general(made, None, 0.5)                               # general text switched off
    assert gt.same_general(gt.general_key(None, 0.0, 3407), None, 0.0)        # off then and off now
    assert not gt.same_general({}, src, 0.5)                                  # an old cache without a key
