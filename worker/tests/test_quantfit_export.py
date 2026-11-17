"""The size `mixed_measured` in an export (exportjob._make_mixed + quantfit.for_export) with stand-in llama.cpp tools: the exact command line with the tensor-type file and with rules, the
matrix refusal, a target below the minimum, the fallback to the guessed fit, and the labels."""
import json
import re
from pathlib import Path

import numpy as np
import pytest
from test_imatrix import world  # noqa: F401  (a fixture mw needs)
from test_mixedquant import QUANTIZE_FLAGS, mw, quantize_calls  # noqa: F401  (mw is a fixture)
from test_quantfit_job import synthetic_model
from test_quantmeasure import make_matrix

from ndforge_worker import exportjob, ggufrows, llamacpp, mixedquant, quantfit, quantmeasure

# the stand-in of test_mixedquant plus the help text and `--tensor-type-file` of a recent llama-quantize
QUANTIZE_TYPEFILE = QUANTIZE_FLAGS.replace('''mat, rest, i = None, [], 0''', '''if "--help" in a:
    print("usage: llama-quantize [--imatrix] [--tensor-type-file]\\n  --tensor-type-file FILE: per-tensor types, one TENSOR=TYPE per line", file=sys.stderr)
    sys.exit(1)
mat, rest, i = None, [], 0''').replace('"--tensor-type"):', '"--tensor-type", "--tensor-type-file"):')


@pytest.fixture
def mm(mw, monkeypatch, tmp_path):
    """mw with a synthetic 16-bit model (real numbers, not zeros) as the converter's output, and a matrix of that model"""
    llamacpp._HELP_CACHE.clear()
    gguf = synthetic_model(tmp_path / "prebuilt2.gguf")
    monkeypatch.setenv("PREBUILT_GGUF", str(gguf))
    src = ggufrows.open_source(gguf)
    mw.matrix = make_matrix(tmp_path / "lib.gguf", src, np.random.default_rng(2))
    mw.src = src
    mw.target = quantfit.plain_bytes(src, "Q4_K_M") / 1e9

    def export(quants=("mixed_measured",), measured=None, with_matrix=True, **cfg_extra):
        out = mw.root / "outm"
        out.mkdir(exist_ok=True)
        cfg = mw.cfg()
        cfg["mixed_measured"] = {"target_gb": mw.target, "matrix": str(mw.matrix), "sample_rows": 16, "engine": "reference", "cache_dir": str(mw.root / "quantfit"), **(measured or {})}
        cfg.update(cfg_extra)
        if not with_matrix:
            cfg.pop("imatrix")
        info, rep = {}, exportjob.new_report() if with_matrix else None
        files = exportjob.gguf_with_llama_cpp(cfg, mw.merged, out, list(quants), rep, info)
        return files, rep, info, out
    mw.export_m = export
    return mw


def calls(w):
    return [c for c in quantize_calls(w) if c != ["--help"]]


def test_the_size_is_known_and_in_the_table_of_ids():
    assert mixedquant.MEASURED_ID == "mixed_measured" == quantfit.ID and mixedquant.MIXED_IDS[-1] == "mixed_measured" and "mixed_measured" in exportjob.QUANTS
    assert exportjob.QUANTS[:5] == ("q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16")


def test_the_command_line_with_rules_when_llama_quantize_has_no_type_file(mm):
    files, rep, info, out = mm.export_m()
    (call,) = calls(mm)
    r = info["mixed_measured"]
    assert r["id"] == "mixed_measured" and r["label"] == "Mixed: measured fit to a size" and r["recipe"] == mixedquant.LABEL
    assert call[:2] == ["--imatrix", str(out / "imatrix.gguf")] and call[2:6] == ["--output-tensor-type", "Q6_K", "--token-embedding-type", "Q4_K"]
    assert call[-3:-1] == [str(out / "model-f16.gguf"), str(out / "model-mixed_measured.gguf.partial")] and call[-1] == r["base"]
    rules = [call[i + 1] for i, x in enumerate(call) if x == "--tensor-type"]
    assert rules == r["rules"] and len(rules) > 3 and "--tensor-type-file" not in call and not (out / "mixed_measured.tensor-types.txt").exists()
    assert files["mixed_measured"].is_file() and rep["applied"] == ["mixed_measured"]
    m = r["measured"]
    assert m["engine"] == "reference" and m["sample_rows"] == 16 and m["fits"] is True and len(m["measure_key"]) == 16 and m["predicted"]["error"] > 0 and set(m["simulated"]) == {"plain_q4_k_m", "plain_q5_k_m"}
    assert r["estimate"]["unmatched"] == 0 and r["fit"]["fits"] and r["actual_bytes"] == files["mixed_measured"].stat().st_size
    assert r["estimate"]["bytes"] <= mm.target * 1e9 and r["estimate"]["bytes"] > 0.9 * mm.target * 1e9


def test_the_command_line_with_the_type_file_when_llama_quantize_has_one(mm):
    (mm.lcpp / "llama-quantize.py").write_text(QUANTIZE_TYPEFILE, encoding="utf-8")
    files, rep, info, out = mm.export_m()
    (call,) = calls(mm)
    tf = out / "mixed_measured.tensor-types.txt"
    assert call[call.index("--tensor-type-file") + 1] == str(tf) and "--tensor-type" not in call
    assert tf.read_text(encoding="utf-8").splitlines() == info["mixed_measured"]["rules"]
    assert info["mixed_measured"]["measured"]["tensor_type_file"] == str(tf)


def test_a_matrix_the_tool_refuses_makes_the_size_without_it_and_says_so(mm, monkeypatch):
    monkeypatch.setenv("QUANTIZE_REFUSES_IMATRIX", "1")
    files, rep, info, out = mm.export_m()
    first, second = calls(mm)
    assert "--imatrix" in first and "--imatrix" not in second and second[:2] == ["--output-tensor-type", "Q6_K"] and second[2:4] == first[first.index("--output-tensor-type") + 2:][:2]
    assert any("did not accept the importance matrix for Mixed: measured fit to a size" in n for n in rep["notes"]) and rep["not_applied"]["mixed_measured"] == "llama-quantize refused the matrix"
    assert files["mixed_measured"].is_file()


def test_a_target_below_the_minimum_makes_the_floor_plan_with_a_loud_note(mm):
    files, rep, info, out = mm.export_m(measured={"target_gb": 0.0001})
    r = info["mixed_measured"]
    assert files["mixed_measured"].is_file() and r["fit"]["fits"] is False and r["measured"]["fits"] is False
    assert any("smaller than the smallest this model can be" in n and "The smallest plan was made" in n for n in r["notes"])
    assert set(re.findall(r"=(Q\d_K)$", "\n".join(r["rules"]), re.M)) == {"Q3_K"}


def test_without_the_export_matrix_the_saved_one_decides_and_is_given_to_the_tool(mm):
    files, rep, info, out = mm.export_m(with_matrix=False)
    (call,) = calls(mm)
    assert call[:2] == ["--imatrix", str(mm.matrix)] and info["mixed_measured"]["id"] == "mixed_measured" and info["mixed_measured"]["matrix_used"] is True


def test_a_matrix_of_another_model_falls_back_to_the_guessed_fit_with_a_note(mm, tmp_path):
    bad = make_matrix(tmp_path / "bad.gguf", mm.src, np.random.default_rng(1), bad="blk.1.attn_k.weight")
    files, rep, info, out = mm.export_m(measured={"matrix": str(bad)})
    r = info["mixed_measured"]
    assert files["mixed_measured"].is_file() and r["id"] == "mixed_fit" and r["requested"] == "mixed_measured"
    assert r["notes"][0].startswith("The measured fit to a size could not be done") and "not of this model; adapt it first" in r["notes"][0] and "measured" not in r


def test_no_target_falls_back_with_a_note(mm):
    files, rep, info, out = mm.export_m(measured={"target_gb": 0})
    r = info["mixed_measured"]
    assert r["notes"][0].startswith("The measured fit to a size could not be done") and "Give the size you want" in r["notes"][0] and files["mixed_measured"].is_file()


def test_it_can_be_made_next_to_plain_sizes_and_nothing_says_dynamic_or_ud(mm):
    files, rep, info, out = mm.export_m(quants=("q4_k_m", "mixed_measured", "mixed_4"))
    assert sorted(p.name for p in out.glob("model-*.gguf")) == ["model-mixed_4.gguf", "model-mixed_measured.gguf", "model-q4_k_m.gguf"]
    text = json.dumps(info) + json.dumps(rep) + " ".join(p.name for p in out.iterdir()) + quantfit.LABEL + quantfit.DESCRIPTION
    assert not re.search(r"dynamic|UD-", text, re.I)
    assert info["mixed_measured"]["measured"]["measure_key"] and "measured" not in info["mixed_4"]


def test_a_second_export_reuses_the_measures(mm):
    mm.export_m()
    cache = list((mm.root / "quantfit").glob("*/measure.json"))
    assert len(cache) == 1
    stamp = cache[0].stat().st_mtime_ns
    mm.export_m()
    assert cache[0].stat().st_mtime_ns == stamp
