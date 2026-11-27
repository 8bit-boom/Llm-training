"""The thinking check reads Inkling's segments and asks with its template arguments and sampling; the Qwen paths stay as they were."""
from ndforge_worker import families, thinkcheck, thinkcheck_llama

ANSWER = "<|message_model|><|content_text|>42<|end_message|><|content_model_end_sampling|>"
TRACE = "<|message_model|><|content_thinking|>let me see: 6 times 7<|end_message|>"


def test_a_closed_trace_and_an_answer_segment():
    r = thinkcheck.split_output(TRACE + ANSWER)
    assert r == {"think": "let me see: 6 times 7", "answer": "42", "closed": True}


def test_several_trace_segments_are_one_trace():
    r = thinkcheck.split_output("<|message_model|><|content_thinking|>first<|end_message|><|message_model|><|content_thinking|>second<|end_message|>" + ANSWER)
    assert r["think"] == "first\nsecond" and r["answer"] == "42" and r["closed"] is True


def test_a_trace_that_never_ends_is_not_closed_and_has_no_answer():
    r = thinkcheck.split_output("<|message_model|><|content_thinking|>and then I thought about it and thought about it")
    assert r["closed"] is False and r["answer"] == "" and r["think"].startswith("and then")
    r = thinkcheck.split_output(TRACE)
    assert r["closed"] is False and r["answer"] == ""


def test_an_answer_without_thinking_is_closed_by_itself():
    r = thinkcheck.split_output(ANSWER, started_in_think=False)
    assert r == {"think": "", "answer": "42", "closed": True}


def test_qwen_text_is_read_exactly_as_before():
    assert thinkcheck.split_output("work it out\n</think>\n\n42") == {"think": "work it out", "answer": "42", "closed": True}
    assert thinkcheck.split_output("still going", started_in_think=True) == {"think": "still going", "answer": "", "closed": False}


def test_the_item_is_scored_through_the_segments():
    item = {"id": "q1", "kind": "int", "answer": 42}
    got = thinkcheck.score_item(item, {"text": TRACE + ANSWER, "gen_tokens": 30})
    assert got["closed"] is True and got["think"] is True and got["got"] in ("42", 42)


def test_the_template_arguments_and_sampling_follow_the_family():
    assert families.think_kwargs("inkling2", True) == {"reasoning_effort": "high"} and families.think_kwargs("inkling2", False) == {"reasoning_effort": "none"}
    assert families.think_kwargs("qwen3_5", True) == {"enable_thinking": True} and families.think_kwargs("generic", False) == {"enable_thinking": False}
    s = thinkcheck.settings_of({"family": "inkling2"})
    assert (s["temperature"], s["top_p"], s["top_k"], s["seed"]) == (1.0, 1.0, 0, 3407)
    q = thinkcheck.settings_of({})
    assert (q["temperature"], q["top_p"], q["top_k"]) == (0.6, 0.95, 20)
    assert thinkcheck.settings_of({"family": "inkling2", "temperature": 0.3})["temperature"] == 0.3        # the job's own number wins
    req = {"messages": [], "max_new": 10, "thinking": True}
    assert thinkcheck_llama.body_of(req, s, "inkling2")["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert thinkcheck_llama.body_of({**req, "thinking": False}, s, "inkling2")["chat_template_kwargs"] == {"reasoning_effort": "none"}
    assert thinkcheck_llama.body_of(req, q)["chat_template_kwargs"] == {"enable_thinking": True}
