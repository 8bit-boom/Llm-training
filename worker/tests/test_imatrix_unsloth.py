"""The importance matrix with Unsloth's own GGUF export: llama.cpp's llama-imatrix measures the matrix, Unsloth's save_pretrained_gguf(imatrix_file=...) applies it
when the installed build can. Stand-in llama.cpp tools (test_imatrix.py) and a stand-in Unsloth (a fake `unsloth` module); no torch, no llama.cpp, no network."""
import json
import sys
import types
from pathlib import Path

import pytest
from test_imatrix import world  # noqa: F401  (world is a fixture)

from ndforge_worker import exportjob, imatrix, protocol


# ------------------------------------------------------------------------------------------------------------------ can this build apply a matrix?

def with_param(self, save_directory, tokenizer=None, quantization_method="q8_0", imatrix_file=None):
    pass


def kwargs_only(self, save_directory, tokenizer=None, **kwargs):
    pass


def without(self, save_directory, tokenizer=None, quantization_method="q8_0"):
    pass


def positional_only(self, save_directory, imatrix_file=None, /, **kw):
    pass


def test_a_save_function_with_an_imatrix_file_parameter_can_apply_one():
    assert exportjob.unsloth_can_apply_imatrix(with_param) is True


def test_a_function_with_only_kwargs_needs_the_zoo_to_resolve_the_file():
    assert exportjob.unsloth_can_apply_imatrix(kwargs_only, has_resolver=lambda: True) is True
    assert exportjob.unsloth_can_apply_imatrix(kwargs_only, has_resolver=lambda: False) is False


def test_a_function_without_the_parameter_or_kwargs_cannot_and_the_zoo_is_not_asked():
    def boom():
        raise AssertionError("the resolver must not be asked")
    assert exportjob.unsloth_can_apply_imatrix(without, has_resolver=boom) is False


def test_a_positional_only_parameter_does_not_count_but_kwargs_still_may():
    def only(self, imatrix_file=None, /):
        pass
    assert exportjob.unsloth_can_apply_imatrix(only, has_resolver=lambda: True) is False
    assert exportjob.unsloth_can_apply_imatrix(positional_only, has_resolver=lambda: True) is True


def test_the_cpu_loader_of_the_zoo_swallows_the_keyword_so_it_cannot_apply():
    def cpu_save(self, save_directory, tokenizer=None, quantization_method="q8_0", **kwargs):
        pass
    cpu_save.__module__ = "unsloth_zoo.cpu.loader"
    assert exportjob.unsloth_can_apply_imatrix(cpu_save, has_resolver=lambda: True) is False
    assert exportjob.unsloth_can_apply_imatrix(kwargs_only, has_resolver=lambda: True) is True


def test_what_cannot_be_inspected_cannot_apply():
    assert exportjob.unsloth_can_apply_imatrix(None) is False
    assert exportjob.unsloth_can_apply_imatrix(len, has_resolver=lambda: True) in (True, False)      # a builtin without a signature must not raise


def test_the_zoo_import_check_is_made_when_no_resolver_is_given(monkeypatch):
    zoo = types.ModuleType("unsloth_zoo")
    cpp = types.ModuleType("unsloth_zoo.llama_cpp")
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.llama_cpp", cpp)
    assert exportjob.unsloth_can_apply_imatrix(kwargs_only) is False                  # the module has no resolve_imatrix_file
    cpp.resolve_imatrix_file = lambda *a, **k: None
    assert exportjob.unsloth_can_apply_imatrix(kwargs_only) is True
    monkeypatch.setitem(sys.modules, "unsloth_zoo.llama_cpp", None)                   # an import that fails
    assert exportjob.unsloth_can_apply_imatrix(kwargs_only) is False


def test_without_unsloth_installed_there_is_no_save_function(monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth", None)
    monkeypatch.setitem(sys.modules, "unsloth.save", None)
    assert exportjob.installed_save_fn() is None
    save = types.ModuleType("unsloth.save")
    save.unsloth_save_pretrained_gguf = with_param
    monkeypatch.setitem(sys.modules, "unsloth", types.ModuleType("unsloth"))
    monkeypatch.setitem(sys.modules, "unsloth.save", save)
    assert exportjob.installed_save_fn() is with_param


# ------------------------------------------------------------------------------------------------------------------ the decision

@pytest.mark.parametrize("can, found, expect", [
    (True, True, (True, True, "unsloth")),              # Unsloth applies the matrix that llama.cpp measured
    (False, True, (False, False, "llama.cpp")),         # today's behaviour: llama.cpp makes the GGUF
])
def test_who_makes_the_gguf_when_a_matrix_is_asked_for(world, can, found, expect):
    rep = exportjob.new_report()
    plan = exportjob.plan_gguf("unsloth", world.cfg(), ["q4_k_m", "q8_0"], rep, lambda: can)
    assert (*plan, rep["via"]) == expect and rep["notes"] == []


def test_without_llama_cpp_unsloth_makes_it_without_a_matrix_and_says_so(world, monkeypatch):
    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: (None, None))
    rep = exportjob.new_report()
    assert exportjob.plan_gguf("unsloth", world.cfg(), ["q4_k_m"], rep, lambda: True) == (True, False)
    assert "llama.cpp (with llama-imatrix) was not found" in rep["notes"][0] and "through Unsloth without it" in rep["notes"][0] and rep["via"] == "unsloth"
    monkeypatch.undo()


def test_without_llama_imatrix_it_is_the_same(world, monkeypatch):
    monkeypatch.setattr(imatrix, "find_llama_imatrix", lambda q: None)
    rep = exportjob.new_report()
    assert exportjob.plan_gguf("unsloth", world.cfg(), ["q4_k_m"], rep, lambda: True) == (True, False) and "through Unsloth without it" in rep["notes"][0]


def test_without_a_size_below_q8_unsloth_gets_no_matrix(world):
    rep = exportjob.new_report()
    assert exportjob.plan_gguf("unsloth", world.cfg(), ["q8_0", "f16"], rep, lambda: True) == (True, False) and "below Q8_0" in rep["notes"][0]


def test_no_matrix_asked_for_or_another_backend(world):
    assert exportjob.plan_gguf("unsloth", world.cfg(), ["q4_k_m"], None, lambda: True) == (True, False)
    assert exportjob.plan_gguf("hf", world.cfg(), ["q4_k_m"], exportjob.new_report(), lambda: True) == (False, False)
    assert exportjob.plan_gguf("unsloth", {**world.cfg(), "gguf": False}, ["q4_k_m"], exportjob.new_report(), lambda: True) == (False, False)


def test_the_build_is_asked_only_when_it_matters(world, monkeypatch):
    def boom():
        raise AssertionError("not needed")
    assert exportjob.plan_gguf("unsloth", world.cfg(), ["q4_k_m"], None, boom) == (True, False)
    assert exportjob.plan_gguf("hf", world.cfg(), ["q4_k_m"], exportjob.new_report(), boom) == (False, False)
    monkeypatch.setattr(imatrix, "find_llama_imatrix", lambda q: None)
    assert exportjob.plan_gguf("unsloth", world.cfg(), ["q4_k_m"], exportjob.new_report(), boom) == (True, False)


# ------------------------------------------------------------------------------------------------------------------ a whole export through a stand-in Unsloth

class FakeModel:
    """what FastLanguageModel gives back: records the calls of save_pretrained_gguf / save_pretrained_merged"""
    refuse = None
    calls = []

    def save_pretrained_gguf(self, save_directory, tokenizer=None, quantization_method="fast_quantized", imatrix_file=None):
        FakeModel.calls.append({"dir": str(save_directory), "method": quantization_method, "imatrix_file": imatrix_file})
        if imatrix_file is not None and FakeModel.refuse:
            raise FakeModel.refuse
        methods = quantization_method if isinstance(quantization_method, list) else [quantization_method]
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        for q in methods:
            (Path(save_directory) / f"model.{q.upper()}.gguf").write_bytes(f"UNSLOTH-{q}{'+imatrix' if imatrix_file else ''}".encode())

    def save_pretrained_merged(self, path, tokenizer, save_method="merged_16bit"):
        FakeModel.calls.append({"merged": str(path)})
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text("{}")


class NoParamModel(FakeModel):
    def save_pretrained_gguf(self, save_directory, tokenizer=None, quantization_method="fast_quantized"):
        return FakeModel.save_pretrained_gguf(self, save_directory, tokenizer, quantization_method)


@pytest.fixture
def unsloth_world(world, monkeypatch):
    """world + an Unsloth that applies a matrix, a merge that makes a model folder (with an MTP head when `head`)"""
    FakeModel.calls, FakeModel.refuse = [], None
    state = types.SimpleNamespace(model_cls=FakeModel, loaded=[], merges=[], head=False, save_fn=with_param)

    class Loader:
        @staticmethod
        def from_pretrained(**kw):
            state.loaded.append(kw)
            return state.model_cls(), object()

    mod = types.ModuleType("unsloth")
    mod.FastLanguageModel = mod.FastModel = Loader
    monkeypatch.setitem(sys.modules, "unsloth", mod)
    monkeypatch.setattr(exportjob, "installed_save_fn", lambda: state.save_fn)
    monkeypatch.setattr(exportjob.backends, "prepare", lambda b: "unsloth")
    monkeypatch.setattr(exportjob.vision, "with_base_processor", lambda tok, repo: tok)
    monkeypatch.setattr(protocol, "reserve_stdout", lambda: None)

    def merge(cfg, merged):
        state.merges.append(str(merged))
        merged.mkdir(parents=True, exist_ok=True)
        (merged / "config.json").write_text(json.dumps({"mtp_num_hidden_layers": 1} if state.head else {}))
    monkeypatch.setattr(exportjob, "merge_hf", merge)

    def job(out=None, quants=("q4_k_m", "q8_0"), **extra):
        out = out or world.root / "job"
        cfg = {**world.cfg(**extra.pop("im", {})), "out": str(out), "quants": list(quants), "quant": quants[0], "backend": "unsloth", "gguf": True, "vision": False, **extra}
        world.events.clear()
        assert exportjob.run(cfg) == 0
        return out, [e for e in world.events if e["type"] == "result"][0]["data"]

    state.job, state.world = job, world
    return state


def test_unsloth_gets_the_matrix_for_the_sizes_it_makes(unsloth_world):
    w = unsloth_world.world
    out, result = unsloth_world.job()
    gguf_calls = [c for c in FakeModel.calls if "method" in c]
    (call,) = gguf_calls                                                                    # one call for the whole batch: the model is not merged twice
    assert call["imatrix_file"] == str(out / "imatrix.gguf") and call["method"] == ["q4_k_m", "q8_0"]
    assert (out / "imatrix.gguf").read_bytes() == b"IMATRIX-FILE" and (out / "imatrix.json").is_file()
    assert (out / "model-q4_k_m.gguf").read_bytes() == b"UNSLOTH-q4_k_m+imatrix"
    assert w.calls("quantize") == []                                                        # llama-quantize is not used: Unsloth quantises
    (m,) = w.calls("imatrix")
    assert m["args"][m["args"].index("-m") + 1] == str(out / "imatrix-f16.gguf")           # measured on a 16-bit GGUF that llama.cpp's converter made ...
    assert not (out / "imatrix-f16.gguf").exists() and not (out / "merged").exists()        # ... and removed again, with the merged copy made for it
    assert unsloth_world.merges == [str(out / "merged")]
    im = result["imatrix"]
    assert (im["via"], im["made"], im["applied"], im["not_applied"].keys(), im["notes"]) == ("unsloth", True, ["q4_k_m"], {"q8_0"}, [])
    assert result["via"] == "unsloth" and json.loads((out / "export.json").read_text(encoding="utf-8"))["imatrix"]["via"] == "unsloth"


def test_the_matrix_is_measured_before_unsloth_loads_the_model(unsloth_world, monkeypatch):
    order = []
    real = imatrix.run_streaming
    monkeypatch.setattr(imatrix, "run_streaming", lambda *a, **k: (order.append("measure"), real(*a, **k))[1])
    unsloth_world.loaded = order
    orig = sys.modules["unsloth"].FastLanguageModel.from_pretrained
    sys.modules["unsloth"].FastLanguageModel.from_pretrained = staticmethod(lambda **kw: (order.append("load"), orig(**kw))[1])
    unsloth_world.job()
    assert order.index("measure") < order.index("load")


def test_a_saved_matrix_is_reused_without_converting_or_merging(unsloth_world):
    w = unsloth_world.world
    out1, _ = unsloth_world.job(out=w.root / "j1")
    n_merges = len(unsloth_world.merges)
    out2, result = unsloth_world.job(out=w.root / "j2", im={"reuse_dirs": [str(out1)]})
    assert result["imatrix"]["reused"] and not result["imatrix"]["made"] and len(unsloth_world.merges) == n_merges and len(w.calls("imatrix")) == 1
    assert [c for c in FakeModel.calls if "method" in c][-1]["imatrix_file"] == str(out2 / "imatrix.gguf")


def test_a_16_bit_gguf_already_in_the_export_folder_is_used(unsloth_world):
    w = unsloth_world.world
    out = w.root / "job"
    out.mkdir()
    (out / "model-f16.gguf").write_bytes(b"GGUF-f16")
    out, result = unsloth_world.job(out=out)
    (m,) = w.calls("imatrix")
    assert m["args"][m["args"].index("-m") + 1] == str(out / "model-f16.gguf") and unsloth_world.merges == [] and result["imatrix"]["made"]


def test_a_full_model_is_not_merged_or_removed(unsloth_world):
    w = unsloth_world.world
    full = w.root / "fullmodel"
    full.mkdir()
    (full / "config.json").write_text("{}")
    _, result = unsloth_world.job(full_model=str(full), adapter=None)
    assert unsloth_world.merges == [] and full.is_dir() and result["imatrix"]["made"] and result["imatrix"]["applied"] == ["q4_k_m"]


def test_keep_merged_behaves_as_before(unsloth_world):
    out, result = unsloth_world.job(keep_merged=True)
    assert [c["merged"] for c in FakeModel.calls if "merged" in c] == [str(out / "merged")]       # Unsloth saves it again, from the loaded model
    assert (out / "merged" / "config.json").is_file() and result["merged_dir"] == str(out / "merged")


def test_a_build_that_cannot_apply_a_matrix_goes_the_llama_cpp_way(unsloth_world):
    w = unsloth_world.world
    unsloth_world.save_fn = without
    out, result = unsloth_world.job()
    assert unsloth_world.loaded == [] and FakeModel.calls == []                           # Unsloth is not used at all
    assert result["via"] == "llama.cpp" and result["imatrix"]["via"] == "llama.cpp" and result["imatrix"]["applied"] == ["q4_k_m"]
    assert any("--imatrix" in c["args"] for c in w.calls("quantize"))


def test_without_llama_cpp_unsloth_exports_without_a_matrix(unsloth_world, monkeypatch):
    w = unsloth_world.world
    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: (None, None))
    out, result = unsloth_world.job()
    (call,) = [c for c in FakeModel.calls if "method" in c]
    assert call["imatrix_file"] is None and result["via"] == "unsloth" and result["imatrix"]["via"] == "unsloth"
    assert not result["imatrix"]["made"] and "llama.cpp (with llama-imatrix) was not found" in result["imatrix"]["note"] and w.calls() == []
    assert (out / "model-q4_k_m.gguf").read_bytes() == b"UNSLOTH-q4_k_m"


def test_a_failing_llama_imatrix_still_exports_through_unsloth_without_the_matrix(unsloth_world, monkeypatch):
    monkeypatch.setenv("IMATRIX_MODE", "fail")
    out, result = unsloth_world.job()
    (call,) = [c for c in FakeModel.calls if "method" in c]
    assert call["imatrix_file"] is None and "llama-imatrix failed" in result["imatrix"]["note"] and result["imatrix"]["applied"] == [] and result["via"] == "unsloth"


def test_a_merge_that_fails_is_a_note_not_an_error(unsloth_world, monkeypatch):
    def broken(cfg, merged):
        raise OSError("disk full")
    monkeypatch.setattr(exportjob, "merge_hf", broken)
    out, result = unsloth_world.job()
    (call,) = [c for c in FakeModel.calls if "method" in c]
    assert call["imatrix_file"] is None and "could not be made (OSError: disk full)" in result["imatrix"]["note"] and (out / "model-q4_k_m.gguf").is_file()


def test_unsloth_refusing_the_matrix_is_tried_again_without_it(unsloth_world):
    FakeModel.refuse = RuntimeError("Unsloth: your installed unsloth_zoo's quantize_gguf does not support imatrix.")
    out, result = unsloth_world.job()
    calls = [c for c in FakeModel.calls if "method" in c]
    assert [c["imatrix_file"] is not None for c in calls] == [True, False]
    assert "Unsloth did not accept the importance matrix" in result["imatrix"]["note"] and result["imatrix"]["applied"] == [] and result["imatrix"]["not_applied"] == {}
    assert (out / "model-q4_k_m.gguf").read_bytes() == b"UNSLOTH-q4_k_m"


def test_another_unsloth_error_is_not_swallowed(unsloth_world):
    FakeModel.refuse = RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        unsloth_world.job()


def test_a_loaded_model_that_cannot_take_the_matrix_after_all_is_a_note(unsloth_world):
    unsloth_world.model_cls = NoParamModel                  # the installed function looked fine, the loaded model's does not take it
    out, result = unsloth_world.job()
    (call,) = [c for c in FakeModel.calls if "method" in c]
    assert call["imatrix_file"] is None and "could not take the importance matrix after all" in result["imatrix"]["note"] and result["imatrix"]["applied"] == []


def test_only_one_size_below_q8_and_the_16_bit_file(unsloth_world):
    out, result = unsloth_world.job(quants=("q4_k_m", "f16"))
    assert result["imatrix"]["applied"] == ["q4_k_m"] and list(result["imatrix"]["not_applied"]) == ["f16"] and result["ggufs"].keys() == {"q4_k_m", "f16"}


def test_the_imatrix_file_in_the_folder_is_never_taken_for_a_gguf(tmp_path):
    (tmp_path / "imatrix.gguf").write_bytes(b"x")
    assert exportjob.find_gguf([tmp_path], "q4_k_m") is None
    (tmp_path / "model.Q4_K_M.gguf").write_bytes(b"x")
    assert exportjob.find_gguf([tmp_path], "q4_k_m").name == "model.Q4_K_M.gguf"


def test_the_mtp_head_reaches_llama_imatrix_through_the_unsloth_path_too(unsloth_world):
    w = unsloth_world.world
    unsloth_world.head = True
    out, result = unsloth_world.job(im={"nextn": True})
    (m,) = w.calls("imatrix")
    assert "--nextn" in m["args"] and result["imatrix"]["nextn"] is True
    assert json.loads((out / "imatrix.json").read_text(encoding="utf-8"))["nextn"] is True


def test_a_stop_while_measuring_ends_the_export_before_unsloth_loads(unsloth_world, monkeypatch):
    w = unsloth_world.world
    (w.root / "STOP").write_text("")
    with pytest.raises(imatrix.Stopped):
        unsloth_world.job()
    assert unsloth_world.loaded == [] and FakeModel.calls == []
    assert not (w.root / "job" / "imatrix-f16.gguf").exists() and not (w.root / "job" / "merged").exists()


def test_without_a_matrix_the_unsloth_export_is_as_before(unsloth_world):
    w = unsloth_world.world
    cfg = {k: v for k, v in w.cfg().items() if k != "imatrix"}
    out = w.root / "plain"
    assert exportjob.run({**cfg, "out": str(out), "quants": ["q4_k_m"], "quant": "q4_k_m", "backend": "unsloth", "gguf": True, "vision": False}) == 0
    (call,) = FakeModel.calls
    assert call["imatrix_file"] is None and call["method"] == "q4_k_m" and w.calls() == []
    result = [e for e in w.events if e["type"] == "result"][0]["data"]
    assert result["via"] == "unsloth" and result["imatrix"] is None


def test_the_llama_cpp_export_says_so_too(world, monkeypatch):
    monkeypatch.setattr(protocol, "reserve_stdout", lambda: None)
    monkeypatch.setattr(exportjob.backends, "prepare", lambda b: "hf")
    monkeypatch.setattr(exportjob, "merge_hf", lambda cfg, merged: merged.mkdir(parents=True, exist_ok=True))
    cfg = {**world.cfg(), "out": str(world.root / "o"), "quants": ["q4_k_m"], "quant": "q4_k_m", "backend": "hf", "gguf": True, "vision": False}
    assert exportjob.run(cfg) == 0
    result = [e for e in world.events if e["type"] == "result"][0]["data"]
    assert result["via"] == "llama.cpp" and result["imatrix"]["via"] == "llama.cpp"
