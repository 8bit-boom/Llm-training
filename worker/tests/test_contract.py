"""The worker's public surface: job types, event types, error codes. nd-world embeds this worker and the Rust program reads these words, so none may be removed or renamed.

`ndforge_worker/contract.json` lists them. The sets found in the code (by reading it with `ast`, nothing is imported or run) must equal the list: a name that disappeared is the
mistake this test is for; a new name has to be added to the list on purpose (new names are always allowed)."""
import ast
import json
from pathlib import Path

import pytest

from ndforge_worker import protocol

ROOT = Path(__file__).resolve().parents[1] / "ndforge_worker"
MANIFEST = json.loads((ROOT / "contract.json").read_text(encoding="utf-8"))
REMOVED = "nd-world embeds this worker: do not remove or rename {what}; add new names instead. Missing from the code: {names}"
ADDED = "{what} found in the code but not in contract.json: {names}. Add them to worker/ndforge_worker/contract.json (new names are fine)."


def strs(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    if isinstance(node, ast.IfExp):
        return strs(node.body) + strs(node.orelse)
    return []


def found_jobs():
    tree = ast.parse((ROOT / "__main__.py").read_text(encoding="utf-8"))
    out = set()
    for n in (m for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name in ("main", "_run") for m in ast.walk(f)):       # `main` reads the job; `_run` dispatches it
        if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and n.left.id == "kind":
            for c in n.comparators:
                out.update(strs(c))
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Name) and n.slice.id == "kind" and isinstance(n.value, ast.Dict):
            for k in n.value.keys:
                out.update(strs(k))
    return out


def found_events_and_codes():
    events, codes = {"log", "progress", "result", "error"}, set()
    for f in sorted(ROOT.glob("*.py")):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call):
                fn = n.func
                name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else ""
                if name == "emit" and n.args:
                    events.update(strs(n.args[0]))
                if name == "error" and n.args and isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "protocol":
                    codes.update(strs(n.args[0]))
                if name == "_code" and len(n.args) >= 2:
                    codes.update(strs(n.args[1]))
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("code", "MISSING_CODE"):
                codes.update(strs(n.value))
    codes.discard("")
    return events, codes


def check(what, found, listed):
    gone, new = sorted(set(listed) - set(found)), sorted(set(found) - set(listed))
    assert not gone, REMOVED.format(what=what, names=gone)
    assert not new, ADDED.format(what=what, names=new)


def test_the_manifest_has_the_protocol_number_of_the_worker():
    assert MANIFEST["protocol"] == protocol.PROTOCOL_VERSION
    assert set(MANIFEST) == {"protocol", "jobs", "events", "error_codes"}
    for k in ("jobs", "events", "error_codes"):
        assert MANIFEST[k] == sorted(set(MANIFEST[k])), f"{k} is kept sorted and without repeats"


def test_every_job_type_of_the_manifest_is_still_dispatched():
    check("job types", found_jobs(), MANIFEST["jobs"])


def test_every_event_type_of_the_manifest_is_still_emitted():
    events, _ = found_events_and_codes()
    check("event types", events, MANIFEST["events"])


def test_every_error_code_of_the_manifest_is_still_reported():
    _, codes = found_events_and_codes()
    check("error codes", codes, MANIFEST["error_codes"])


def test_the_message_for_a_removal_says_what_to_do():
    with pytest.raises(AssertionError, match="nd-world embeds this worker: do not remove or rename job types; add new names instead"):
        check("job types", {"train"}, ["train", "export"])
    with pytest.raises(AssertionError, match="Add them to worker/ndforge_worker/contract.json"):
        check("job types", {"train", "export"}, ["train"])
