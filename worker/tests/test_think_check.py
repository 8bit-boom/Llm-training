"""The thinking check, the part that needs no torch: the suite, reading an output, scoring, loops and garbage, the summary, the gates (shared cases with Rust), the jobs."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ndforge_worker import thinkcheck as tc

DATA = Path(__file__).parent / "data"
WORKER = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------------------------------------------------ the suite

def test_the_suite_has_32_questions_in_four_topics_and_8_control_questions():
    s = tc.load_suite()
    assert s["version"] == tc.SUITE_VERSION == 1
    assert len(s["items"]) == 32 and len(s["control"]) == 8
    assert {t: sum(1 for i in s["items"] if i["topic"] == t) for t in ("arith", "logic", "code", "count")} == {"arith": 8, "logic": 8, "code": 8, "count": 8}
    ids = [i["id"] for i in s["items"] + s["control"]]
    assert len(set(ids)) == 40
    assert all(i["kind"] in ("number", "exact", "choice") and i["prompt"] and str(i["answer"]) for i in s["items"] + s["control"])
    assert all(i["kind"] != "choice" or i["answer"] in "ABCD" for i in s["items"])
    first16 = s["items"][:tc.QUICK_ITEMS]
    assert {t: sum(1 for i in first16 if i["topic"] == t) for t in ("arith", "logic", "code", "count")} == {"arith": 4, "logic": 4, "code": 4, "count": 4}


def test_the_code_questions_have_the_answer_the_program_prints():
    import contextlib
    import io
    import re
    for it in tc.load_suite()["items"]:
        if it["topic"] != "code":
            continue
        src = re.search(r"```python\n(.*?)\n```", it["prompt"], re.S).group(1)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(src, {})                                  # noqa: S102 - the suite's own snippets
        assert tc.is_correct(it, "Answer: " + buf.getvalue().strip()), it["id"]


def test_every_question_is_solved_by_its_own_answer_and_not_by_a_wrong_one():
    for it in tc.load_suite()["items"] + tc.load_suite()["control"]:
        assert tc.is_correct(it, f"Answer: {it['answer']}"), it["id"]
        assert not tc.is_correct(it, "Answer: zzz 99999"), it["id"]


def test_a_broken_suite_file_is_refused(tmp_path):
    f = tmp_path / "s.jsonl"
    f.write_text(json.dumps({"id": "a", "prompt": "p", "answer": "1", "kind": "number"}) + "\n")
    with pytest.raises(tc.ThinkCheckError, match="suite_version"):
        tc.load_suite(f)
    f.write_text(json.dumps({"suite_version": 1}) + "\n" + json.dumps({"id": "a", "prompt": "p", "answer": "1", "kind": "poem"}) + "\n")
    with pytest.raises(tc.ThinkCheckError, match="unknown kind"):
        tc.load_suite(f)


def test_quick_takes_the_first_16_and_the_settings_follow_the_job():
    suite = tc.load_suite()
    full = tc.settings_of({})
    assert (full["max_new"], full["temperature"], full["top_p"], full["top_k"], full["seed"], full["quick"]) == (3072, 0.6, 0.95, 20, 3407, False)
    quick = tc.settings_of({"quick": True})
    assert quick["max_new"] == 2048 and len(tc.questions(suite, quick)[0]) == 16 and len(tc.questions(suite, full)[0]) == 32
    assert tc.questions(suite, tc.settings_of({"control": False}))[1] == []
    assert tc.settings_of({"temperature": 0.3})["temperature"] == 0.3 and tc.settings_of({"max_new": 100})["max_new"] == 100


# ------------------------------------------------------------------------------------------------------------ reading an output

def test_split_output_with_thinking_on():
    s = tc.split_output("Let me see.\nSix sevens.\n</think>\n\nIt is 42.\nAnswer: 42")
    assert s == {"think": "Let me see.\nSix sevens.", "answer": "It is 42.\nAnswer: 42", "closed": True}
    assert tc.split_output("<think>\nhmm\n</think>\n\n7")["think"] == "hmm"                         # a template that did not open the block
    cut = tc.split_output("still thinking and thinking")
    assert cut == {"think": "still thinking and thinking", "answer": "", "closed": False}
    assert tc.split_output("</think>\n\nAnswer: 3") == {"think": "", "answer": "Answer: 3", "closed": True}   # an empty trace
    assert tc.split_output("") == {"think": "", "answer": "", "closed": False}


def test_split_output_with_thinking_off():
    assert tc.split_output("Answer: Paris", started_in_think=False) == {"think": "", "answer": "Answer: Paris", "closed": False}
    assert tc.split_output("<think>\n\n</think>\n\nAnswer: Paris", started_in_think=False)["answer"] == "Answer: Paris"


@pytest.mark.parametrize("text,kind,expect", [
    ("so\nAnswer: 42", "number", "42"), ("Answer: **1,250**.", "number", "1250"), ("answer is: 2.50", "number", "2.5"), ("the total is 7 apples", "number", "7"),
    ("Answer: -3", "number", "-3"), ("Answer: 12.0", "number", "12"), ("Answer: \\boxed{58}", "number", "58"), ("no digits here", "number", None),
    ("I think 3 or 4.\nAnswer: 4", "number", "4"), ("Answer: 0.0", "number", "0"),
    ("Answer: B", "choice", "B"), ("Answer: (c)", "choice", "C"), ("Answer: B) No A are C", "choice", "B"), ("**Answer:** D.", "choice", "D"),
    ("The right option is A.", "choice", "A"), ("nothing", "choice", None),
    ("Answer: **Ben**.", "exact", "ben"), ("Answer: `bANana`", "exact", "banana"), ("so it is Friday", "exact", "so it is friday"), ("Answer:  4   5 ", "exact", "4 5"),
    ("", "exact", None), ("  \n ", "number", None),
])
def test_extract_answer(text, kind, expect):
    assert tc.extract_answer(text, kind) == expect


def test_is_correct_uses_the_items_own_normal_form():
    item = {"id": "x", "kind": "exact", "answer": "bANana", "prompt": "p"}
    assert tc.is_correct(item, "Answer: Banana.") and not tc.is_correct(item, "Answer: banan")
    num = {"id": "n", "kind": "number", "answer": "2.5", "prompt": "p"}
    assert tc.is_correct(num, "Answer: 2.50") and tc.is_correct(num, "Answer: 5/2 = 2.5") and not tc.is_correct(num, "Answer: 25")


# ------------------------------------------------------------------------------------------------------------ loops and garbage

def test_a_repeated_run_of_32_tokens_is_a_loop():
    run = list(range(9000, 9032))
    assert tc.loop_detect(list(range(500)) + run * 4)
    assert not tc.loop_detect(list(range(500)) + run * 3)
    assert not tc.loop_detect(list(range(1200)))
    assert not tc.loop_detect([])
    assert not tc.loop_detect(run * 4 + list(range(2000, 3100)))              # the repeats are older than the last 1024 tokens


def test_text_that_compresses_to_almost_nothing_is_a_loop():
    assert tc.loop_detect([], "wait, " * 600)
    assert not tc.loop_detect([], "wait, " * 20)                                 # too short to judge by size
    import random
    r = random.Random(1)
    assert not tc.loop_detect([], " ".join(r.choice(["alpha", "beta", "gamma", "delta", "x", "y", "z"]) + str(r.randrange(1000)) for _ in range(500)))


def test_garbage_is_replacement_or_control_characters_or_another_script():
    assert not tc.garbage_detect("") and not tc.garbage_detect("Normal text with an accent: café, naïve.\nNext line\ttab.")
    assert tc.garbage_detect("ok " + "�" * 5)
    assert tc.garbage_detect("abc\x00\x01\x02def")
    assert not tc.garbage_detect("one stray � in a long long long long long long long long long long long long long long long long long long long long long long long long long long long long text")
    assert tc.garbage_detect("好的，让我们一步一步地想一想这个问题，然后给出最终的答案。" * 3)
    assert tc.garbage_detect("Давайте подумаем над этим вопросом шаг за шагом, потом ответим.")
    assert not tc.garbage_detect("The answer is 5 (the Chinese word 好 is a greeting). Then more English words follow to dilute the single foreign letter.")


# ------------------------------------------------------------------------------------------------------------ one answered question

ITEM = {"id": "a1", "topic": "arith", "kind": "number", "answer": "42", "prompt": "p"}


def test_a_good_answer_with_thinking():
    r = tc.score_item(ITEM, {"text": "six sevens\n</think>\n\nAnswer: 42", "gen_tokens": 12, "think_tokens": 3})
    assert (r["ok"], r["think"], r["closed"], r["think_tokens"], r["truncated"], r["loop"], r["garbage"], r["got"]) == (True, True, True, 3, False, False, False, "42")


def test_a_trace_that_never_ends_has_no_answer():
    r = tc.score_item(ITEM, {"text": "Answer: 42 but wait, maybe", "gen_tokens": 3072, "truncated": True})
    assert (r["ok"], r["think"], r["closed"], r["truncated"]) == (False, True, False, True)


def test_the_trace_length_is_estimated_when_the_backend_gives_none():
    assert tc.score_item(ITEM, {"text": "x" * 400 + "\n</think>\n\nAnswer: 42"})["think_tokens"] == 100


def test_a_control_answer_has_no_trace():
    r = tc.score_item(ITEM, {"text": "Answer: 42"}, thinking=False)
    assert (r["ok"], r["think"], r["closed"]) == (True, False, False)


def test_a_looping_answer_is_marked():
    text = "Wait, let me recompute it once more. " * 200 + "\n</think>\n\nAnswer: 42"
    assert tc.score_item(ITEM, {"text": text})["loop"] is True


# ------------------------------------------------------------------------------------------------------------ summary and gates

def result(**kw):
    base = {"id": "x", "topic": "arith", "kind": "number", "ok": True, "got": "1", "gold": "1", "think": True, "closed": True, "think_tokens": 100, "gen_tokens": 120,
            "truncated": False, "loop": False, "garbage": False}
    return {**base, **kw}


def test_summarize():
    rs = [result(think_tokens=t, ok=i % 2 == 0, closed=i != 3, loop=i == 0) for i, t in enumerate([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])]
    s = tc.summarize(rs)
    assert s["n"] == 10 and s["accuracy"] == 0.5 and s["closed_rate"] == 0.9 and s["loop_rate"] == 0.1 and s["think_rate"] == 1.0
    assert s["median_think_tokens"] == 500 and s["p90_think_tokens"] == 900
    assert tc.summarize([])["n"] == 0 and tc.summarize([])["median_think_tokens"] == 0
    assert tc.summarize([result(think=False, think_tokens=0)])["median_think_tokens"] == 0                      # no trace: no length


GATE_CASES = json.loads((DATA / "think_gate_cases.json").read_text())


def test_the_gate_names_are_the_shared_ones():
    assert list(tc.GATES) == GATE_CASES["gates"]


@pytest.mark.parametrize("case", GATE_CASES["cases"], ids=lambda c: c["name"])
def test_gates_follow_the_shared_cases(case):
    g = tc.gates(case["original"], case["candidate"])
    assert (g["verdict"], g["failed"], g["warned"]) == (case["expect"]["verdict"], case["expect"]["failed"], case["expect"]["warned"])
    assert [c["name"] for c in g["checks"]] == GATE_CASES["gates"]


# ------------------------------------------------------------------------------------------------------------ the jobs

def run_worker(tmp_path, job):
    f = tmp_path / "job.json"
    f.write_text(json.dumps(job), encoding="utf-8")
    p = subprocess.run([sys.executable, "-X", "utf8", "-m", "ndforge_worker", str(f)], cwd=WORKER, capture_output=True, text=True, timeout=120)
    return p, [json.loads(line) for line in p.stdout.splitlines() if line.startswith("{")]


def test_the_plan_job_needs_no_model(tmp_path):
    p, ev = run_worker(tmp_path, {"job": "think_check_plan", "backend": "llama"})
    assert p.returncode == 0, p.stderr
    d = ev[-1]["data"]
    assert (d["items"], d["control_items"], d["backend"], d["suite_version"]) == (32, 8, "llama", 1) and 40_000 < d["est_tokens"] < 60_000 and d["est_minutes"] > 0
    p, ev = run_worker(tmp_path, {"job": "think_check_plan", "quick": True})
    assert ev[-1]["data"]["items"] == 16 and ev[-1]["data"]["est_tokens"] < 30_000


def test_an_unknown_backend_is_a_plain_error(tmp_path):
    p, ev = run_worker(tmp_path, {"job": "think_check", "backend": "nope", "model": "x"})
    assert p.returncode == 1 and ev[-1]["type"] == "error" and ev[-1]["code"] == "think_check_failed" and "Unknown way" in ev[-1]["message"]


class FakeBackend:
    """answers every question: the right answer after a short trace, except the ones in `wrong`; stops after `stop_after` answers when asked to"""

    def __init__(self, wrong=(), stop_after=None):
        self.wrong, self.stop_after, self.asked = set(wrong), stop_after, []

    def generate(self, requests, settings, on_result, should_stop, job):
        self.asked = requests
        items = {i["id"]: i for i in tc.load_suite()["items"] + tc.load_suite()["control"]}
        for n, rq in enumerate(requests):
            if should_stop() or (self.stop_after is not None and n >= self.stop_after):
                return
            ans = "nope" if rq["id"] in self.wrong else items[rq["id"]]["answer"]
            text = f"Answer: {ans}" if not rq["thinking"] else f"thinking about {rq['id']}\n</think>\n\nAnswer: {ans}"
            on_result({"id": rq["id"], "text": text, "gen_tokens": 20, "truncated": False, "think_tokens": 5 if rq["thinking"] else None})


def captured(monkeypatch):
    import io

    from ndforge_worker import protocol
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    return lambda: [json.loads(x) for x in buf.getvalue().splitlines() if x.startswith("{")]


def test_the_job_scores_every_answer_and_emits_events(monkeypatch):
    lines_of = captured(monkeypatch)
    be = FakeBackend(wrong={"arith1", "logic2", "ctrl1"})
    monkeypatch.setattr(tc, "load_backend", lambda name: be)
    assert tc.run({"job": "think_check", "model": "m", "backend": "hf"}) == 0
    lines = lines_of()
    res = lines[-1]["data"]
    assert res["main"]["n"] == 32 and res["control"]["n"] == 8 and res["main"]["accuracy"] == round(30 / 32, 4) and res["control"]["accuracy"] == 0.875
    assert res["main"]["closed_rate"] == 1.0 and res["main"]["think_rate"] == 1.0 and res["control"]["think_rate"] == 0.0 and res["stopped"] is False
    items = [x for x in lines if x["type"] == "think_item"]
    assert len(items) == 40 and {"id", "ok", "closed", "think_tokens"} <= set(items[0]) and sum(1 for x in items if x["control"]) == 8
    assert any(x["type"] == "progress" and x["step"] == "think" for x in lines)
    assert [r["thinking"] for r in be.asked].count(True) == 32 and all(r["max_new"] == 3072 for r in be.asked if r["thinking"])
    assert all(r["max_new"] == 512 for r in be.asked if not r["thinking"])
    assert len(res["items"]) == 40 and res["settings"]["seed"] == 3407 and res["suite_version"] == 1


def test_quick_without_control_and_stop(monkeypatch, tmp_path):
    lines_of = captured(monkeypatch)
    be = FakeBackend(stop_after=5)
    monkeypatch.setattr(tc, "load_backend", lambda name: be)
    stop = tmp_path / "STOP"
    assert tc.run({"job": "think_check", "model": "m", "quick": True, "control": False, "stop_file": str(stop)}) == 0
    res = lines_of()[-1]["data"]
    assert len(be.asked) == 16 and all(r["max_new"] == 2048 for r in be.asked)
    assert res["main"]["n"] == 5 and res["control"]["n"] == 0 and res["stopped"] is True


# ------------------------------------------------------------------------------------------------------------ the saved numbers of a model

def test_a_finished_check_is_answered_from_the_cache_and_a_changed_setting_is_not(monkeypatch, tmp_path):
    lines_of = captured(monkeypatch)
    model = tmp_path / "orig"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"x" * 10)
    be = FakeBackend()
    monkeypatch.setattr(tc, "load_backend", lambda name: be)
    job = {"job": "think_check", "model": str(model), "backend": "hf", "cache_file": str(tmp_path / "c" / "orig.json")}
    assert tc.run(dict(job)) == 0
    first = lines_of()[-1]["data"]
    assert first["cached"] is False and (tmp_path / "c" / "orig.json").is_file() and not (tmp_path / "c" / "orig.json.partial").exists()
    asked = len(be.asked)
    be.asked = []
    assert tc.run(dict(job)) == 0
    again = lines_of()[-1]["data"]
    assert again["cached"] is True and be.asked == [] and again["main"] == first["main"] and asked == 40
    # another setting, another backend, another load, or other weights: asked again
    for change in ({"quick": True}, {"temperature": 0.7}, {"backend": "llama"}, {"load": "4bit"}, {"control": False}):
        be.asked = []
        tc.run({**job, **change})
        assert be.asked, change
    (model / "model.safetensors").write_bytes(b"x" * 11)
    be.asked = []
    tc.run(dict(job))
    assert be.asked


def test_a_stopped_check_is_not_saved(monkeypatch, tmp_path):
    captured(monkeypatch)
    cache = tmp_path / "c.json"
    monkeypatch.setattr(tc, "load_backend", lambda name: FakeBackend(stop_after=3))
    tc.run({"job": "think_check", "model": "org/x", "cache_file": str(cache)})
    assert not cache.exists()


def test_fingerprints(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF")
    assert tc.fingerprint(str(f)).startswith("file:m.gguf:4:") and tc.fingerprint("org/name") == "repo:org/name"
    d = tmp_path / "d"
    d.mkdir()
    assert tc.fingerprint(str(d)).startswith("dir:")


def test_the_check_always_samples_it_is_never_greedy():
    """Greedy decoding hid the failure of a real run (docs/REAL_RUN_17B.md): the thinking check asks sampled answers, with the settings Qwen recommends for thinking."""
    import ast
    from pathlib import Path

    from ndforge_worker import thinkcheck_hf
    tree = ast.parse(Path(thinkcheck_hf.__file__).read_text(encoding="utf-8"))
    gens = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "generate"]
    assert gens, "the check generates with model.generate"
    for call in gens:
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        assert isinstance(kw.get("do_sample"), ast.Constant) and kw["do_sample"].value is True, "do_sample=True"
        assert {"temperature", "top_p", "top_k"} <= set(kw)
