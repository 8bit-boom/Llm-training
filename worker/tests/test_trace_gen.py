"""trace_gen: prompts from a file, the filters, the output format of a heal source, resume by prompt hash, Stop, the choice of the way and the fallback. Fake backends; no model, no GPU."""
import io
import json
import threading
import time

import pytest

import test_think_check_vllm as ttv
from ndforge_worker import generaltext, protocol, thinkcheck as tc, tracegen as tg
from ndforge_worker import vllmengine as ve


def prompts_file(tmp_path, n=6, name="p.jsonl", key="prompt"):
    f = tmp_path / name
    f.write_text("\n".join(json.dumps({key: f"Question number {i}: what is {i} plus {i}? Explain a little."}) for i in range(n)), encoding="utf-8")
    return f


def out(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    return lambda: [json.loads(x) for x in buf.getvalue().splitlines() if x.startswith("{")]


class Backend:
    """Answers every request with a good trace unless a rule says otherwise: rules[i] = text / (text, truncated)."""

    def __init__(self, rules=None, fail=None):
        self.rules, self.fail, self.calls = rules or {}, fail, []

    def generate(self, requests, settings, on_result, should_stop, job):
        self.calls.append([r["id"] for r in requests])
        if self.fail:
            raise tc.ThinkCheckError(self.fail)
        for i, r in enumerate(requests):
            if should_stop():
                return
            rule = self.rules.get(i)
            text, trunc = (rule if isinstance(rule, tuple) else (rule, False)) if rule is not None else (f"let me think about {i}\n</think>\n\nThe answer is {i * 2}.", False)
            on_result({"id": r["id"], "text": text, "gen_tokens": 10, "truncated": trunc})


def job(tmp_path, **kw):
    return {"job": "trace_gen", "model": "org/original", "backend": "hf", "source": {"file": str(prompts_file(tmp_path))}, "out_file": str(tmp_path / "out" / "answers.jsonl"), **kw}


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_the_prompts_are_the_user_turns_of_a_file_a_dataset_shape_or_a_chat(tmp_path):
    chat = tmp_path / "chat.jsonl"
    chat.write_text("\n".join(json.dumps(r) for r in [
        {"messages": [{"role": "user", "content": "First question that is long enough"}, {"role": "assistant", "content": "an answer"}]},
        {"messages": [{"role": "user", "content": "First question that is long enough"}, {"role": "assistant", "content": "again"}]},     # a duplicate
        {"messages": [{"from": "human", "value": "Second question, also long enough"}, {"from": "gpt", "value": "ok"}]}]))
    got = [p for p, _ in tg.prompts({"file": str(chat)})]
    assert got == ["First question that is long enough", "Second question, also long enough"]
    assert [p for p, _ in tg.prompts({"file": str(prompts_file(tmp_path))})][0].startswith("Question number")
    pair = tmp_path / "pair.jsonl"
    pair.write_text(json.dumps({"instruction": "Explain how a rainbow forms please", "output": "Light bends."}))
    assert [p for p, _ in tg.prompts({"file": str(pair)})] == ["Explain how a rainbow forms please"]


def test_good_answers_are_written_in_the_heal_source_format_and_the_filters_reject_the_rest(tmp_path, monkeypatch):
    be = Backend({1: "no end of thinking ... still thinking", 2: ("thought\n</think>\n\nanswer", True), 3: "thought\n</think>\n\n", 4: ("wait " * 800 + "\n</think>\n\nfine", False), 5: "gar����bled text\n</think>\n\nok " * 3})
    monkeypatch.setattr(tc, "load_backend", lambda n: be)
    o = out(monkeypatch)
    assert tg.run(job(tmp_path)) == 0
    res = o()[-1]["data"]
    assert res["attempted"] == 6 and res["kept"] == 1 and res["rejected"] == {"not_closed": 1, "cut_by_limit": 1, "empty_answer": 1, "loop": 1, "garbled": 1}
    kept = rows(tmp_path / "out" / "answers.jsonl")
    assert len(kept) == 1 and kept[0]["messages"][0] == {"role": "user", "content": "Question number 0: what is 0 plus 0? Explain a little."}
    assert kept[0]["messages"][1] == {"role": "assistant", "content": "<think>\nlet me think about 0\n</think>\n\nThe answer is 0."} and kept[0]["backend"] == "hf"
    # the file is a valid `file` source of a heal: the chat is read back as a chat
    item = generaltext.row_item(kept[0], generaltext.detect(kept[0]))
    assert item[0] == "chat" and item[1][-1]["content"].startswith("<think>\n")
    assert len(rows(tmp_path / "out" / "answers.jsonl.rejected.jsonl")) == 5


def test_a_second_run_skips_what_is_written_or_was_rejected_and_goes_on_with_new_prompts(tmp_path, monkeypatch):
    be = Backend({1: "unclosed"})
    monkeypatch.setattr(tc, "load_backend", lambda n: be)
    o = out(monkeypatch)
    assert tg.run(job(tmp_path, limit=3)) == 0
    assert o()[-1]["data"]["attempted"] == 3
    assert tg.run(job(tmp_path, limit=10)) == 0
    res = o()[-1]["data"]
    assert res["skipped_existing"] == 3 and res["attempted"] == 3, "the other three prompts"
    assert len({h for call in be.calls for h in call}) == 6, "no prompt was asked twice"
    assert tg.run(job(tmp_path, limit=10)) == 0
    assert o()[-1]["data"]["attempted"] == 0 and "Nothing new" in o()[-1]["data"]["note"]


def test_stop_keeps_what_was_written(tmp_path, monkeypatch):
    stop = tmp_path / "stop"

    class StopAfterTwo(Backend):
        def generate(self, requests, settings, on_result, should_stop, job):
            for i, r in enumerate(requests):
                if i == 2:
                    stop.write_text("x")
                if should_stop():
                    return
                on_result({"id": r["id"], "text": "t\n</think>\n\nok", "gen_tokens": 3, "truncated": False})

    monkeypatch.setattr(tc, "load_backend", lambda n: StopAfterTwo())
    o = out(monkeypatch)
    assert tg.run(job(tmp_path, stop_file=str(stop))) == 0
    res = o()[-1]["data"]
    assert res["stopped"] is True and res["kept"] == 2 and len(rows(tmp_path / "out" / "answers.jsonl")) == 2
    progress = [x for x in o() if x["type"] == "progress"]
    assert progress[-1]["detail"]["unit"] == "answers" and progress[-1]["detail"]["total"] == 6


def test_the_way_is_chosen_vllm_then_llama_then_hf_and_a_failure_goes_to_the_next(tmp_path, monkeypatch):
    j = {"model": "org/m", "engine": {"kind": "docker"}}
    assert tg.choose_chain(j) == ["vllm", "hf"] and tg.choose_chain({"model": "org/m"}) == ["hf"] and tg.choose_chain({**j, "backend": "hf"}) == ["hf"]
    assert tg.choose_chain({"model": "org/m", "engine": {"kind": "podman"}}) == ["hf"], "a bad engine setting is skipped, not fatal"
    bad, good = Backend(fail="the engine did not start"), Backend()
    monkeypatch.setattr(tc, "load_backend", lambda n: bad if n == "vllm" else good)
    o = out(monkeypatch)
    assert tg.run(job(tmp_path, backend="auto", engine={"kind": "docker"})) == 0
    res = o()[-1]["data"]
    assert res["backend"] == "hf" and res["kept"] == 6 and "vllm way failed" in res["notes"][0]
    with pytest.raises(tc.ThinkCheckError):                               # a named way does not fall back
        tg.run(job(tmp_path, out_file=str(tmp_path / "o2.jsonl"), backend="vllm"))


def test_the_vllm_way_end_to_end_with_a_fake_server_and_the_engine_is_stopped(tmp_path, monkeypatch):
    f = ttv.Fake()
    try:
        calls = {"run": []}
        monkeypatch.setattr(ve, "_run", lambda cmd, **kw: calls["run"].append(cmd))
        monkeypatch.setattr(ve, "_popen", lambda cmd, **kw: ttv.Proc())
        monkeypatch.setattr(ve, "free_port", lambda: f.port)
        suite = tc.load_suite()
        src = tmp_path / "s.jsonl"
        src.write_text("\n".join(json.dumps({"prompt": i["prompt"]}) for i in suite["items"][:5]))
        o = out(monkeypatch)
        assert tg.run({"job": "trace_gen", "model": "org/original", "backend": "vllm", "engine": {"kind": "docker"}, "source": {"file": str(src)}, "out_file": str(tmp_path / "v.jsonl"), "start_timeout": 10}) == 0
        res = o()[-1]["data"]
        assert res["backend"] == "vllm" and res["kept"] == 5 and calls["run"][0][:2] == ["docker", "stop"]
        assert {r["max_tokens"] for r in f.requests} == {6144} and all(r["chat_template_kwargs"]["enable_thinking"] for r in f.requests)
        assert rows(tmp_path / "v.jsonl")[0]["backend"] == "vllm"
    finally:
        f.close()


def test_missing_input_is_a_plain_error(tmp_path):
    with pytest.raises(tc.ThinkCheckError, match="needs a model"):
        tg.run({"job": "trace_gen", "model": "m"})
    with pytest.raises(tc.ThinkCheckError):
        tg.run({"job": "trace_gen", "model": "m", "source": {"file": str(tmp_path / "none.jsonl")}, "out_file": str(tmp_path / "o.jsonl")})
