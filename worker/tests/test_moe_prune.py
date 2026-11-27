"""Expert pruning on the packed format, byte for byte, on a fake Inkling folder with 64 experts per layer."""
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import inkfake
import stfile
from ndforge_worker import moe_layout as ml
from ndforge_worker import reshapejob

WORKER = Path(__file__).resolve().parents[1]
E = 64


def scores(experts=E, layers=24):
    """a made-up measurement: every layer ranks its experts differently, no ties in reap"""
    rows = {}
    for l in range(layers):
        reap = [float((e * 7 + l * 3) % experts) + 1.0 for e in range(experts)]
        rows[str(l)] = {"reap": reap, "freq": [float((e * 5 + l) % experts) for e in range(experts)], "gate": [float(e + 1) for e in range(experts)]}
    return {"meta": {"experts": experts}, "layers": rows}


def run_job(tmp_path, job):
    f = tmp_path / "job.json"
    f.write_text(json.dumps(job), encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "ndforge_worker", str(f)], cwd=WORKER, capture_output=True, text=True, timeout=120)
    return p.returncode, [json.loads(line) for line in p.stdout.splitlines()]


@pytest.fixture()
def folder(tmp_path):
    src = tmp_path / "ink"
    made = inkfake.write_folder(src, inkfake.tiny_config(experts=E, top_k=2))
    sd = tmp_path / "scores" / "k1"
    sd.mkdir(parents=True)
    (sd / "moe_scores.json").write_text(json.dumps(scores()), encoding="utf-8")
    return src, made, sd


def job(tmp_path, src, sd, **exp):
    d = tmp_path / "models" / "m1"
    d.mkdir(parents=True, exist_ok=True)
    return {"job": "reshape", "source": {"repo": str(src), "family": "inkling2"}, "mode": "prune", "shard_gb": 5, "out": str(d / "model.partial"), "final": str(d / "model"),
            "experts": {"keep": 32, "scoring": "reap", "scores_dir": str(sd), **exp}}


def test_selection_is_deterministic_and_keeps_the_same_number_everywhere():
    s = scores()
    kept = ml.select_experts(s, 32, "reap")
    assert sorted(kept) == list(range(24)) and all(len(v) == 32 and v == sorted(v) for v in kept.values())
    row = s["layers"]["0"]["reap"]
    assert set(kept[0]) == set(sorted(range(E), key=lambda e: -row[e])[:32])
    assert ml.select_experts(s, 32, "reap") == kept
    # ties: the more used expert wins, then the lower id
    tied = {"layers": {"0": {"reap": [1.0] * 8, "freq": [0, 5, 5, 1, 0, 0, 0, 0], "gate": [1.0] * 8}}}
    assert ml.select_experts(tied, 2, "reap") == {0: [1, 2]}
    assert ml.select_experts(tied, 3, "reap") == {0: [1, 2, 3]}
    with pytest.raises(ml.MoeError):
        ml.select_experts(s, 100, "reap")
    with pytest.raises(ml.MoeError, match="Unknown"):
        ml.select_experts(s, 8, "magic")
    cov = ml.coverage(s, ml.select_experts(s, 32, "gate"))
    assert all(0.7 < c < 1.0 for c in cov.values())


def test_the_cut_is_a_byte_exact_slice(tmp_path, folder):
    src, made, sd = folder
    j = job(tmp_path, src, sd)
    code, ev = run_job(tmp_path, j)
    assert code == 0 and ev[-1]["type"] == "result", ev[-1]
    r = ev[-1]["data"]
    final = Path(j["final"])
    kept = ml.select_experts(scores(), 32, "reap")
    got = {k: v for k, v in stfile.read_folder(final).items() if k not in ("fuse_weight", "layers.0.dflare_norm.weight")}       # (the drafter is a file of its own)
    assert set(got) == set(made["tensors"])
    sliced = 0
    for name, (dtype, shape, data) in made["tensors"].items():
        layer, role, _ = ml.classify(name)
        g = got[name]
        assert g[0] == dtype
        if role in ("experts", "expert_companion", "router", "router_bias"):
            width = stfile.BYTES[dtype]
            assert g[1] == (32,) + tuple(shape[1:]), name
            assert g[2] == stfile.slice_reference(data, shape, width, 0, kept[layer]), name
            sliced += 1
        else:
            assert g[1] == tuple(shape) and g[2] == data, name
    assert sliced == 24 * 2 + 20 * 6 + 4 * 4
    # packed layers (nvfp4 0-19, mxfp8 20-23) were both cut without being decoded; the one-value denominator stayed whole
    assert got["model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator"][1] == (1,)
    assert got["model.llm.layers.21.mlp.experts.w13_weight"][0] == "F8_E4M3" and got["model.llm.layers.3.mlp.experts.w13_weight"][0] == "U8"
    # config, quantization config, drafter, vision
    cfg = json.loads((final / "config.json").read_text())
    assert cfg["text_config"]["n_routed_experts"] == 32 and cfg["text_config"]["num_experts_per_tok"] == 2 and cfg["mtp_config"]
    assert (final / "hf_quant_config.json").read_bytes() == (src / "hf_quant_config.json").read_bytes()
    assert (final / "mtp.safetensors").read_bytes() == (src / "mtp.safetensors").read_bytes()
    k = json.loads((final / "moe_kept.json").read_text())
    assert k["experts_before"] == E and k["kept"]["5"] == kept[5]
    meta = json.loads((final / "ndforge_reshape.json").read_text())
    assert meta["moe"]["experts"] == [64, 32] and meta["moe"]["scoring"] == "reap" and meta["moe"]["mtp"] == "kept, not healed" and meta["moe"]["rebalanced"] is False
    assert any("not rebalanced" in n for n in r["notes"])
    # the folder reads back as a 32-expert model whose size is the planner's size table row
    ins = reshapejob.inspect_model(str(final))
    assert ins["supported"] and ins["moe"]["experts"] == 32 and ins["total_params"] - ins["mtp_params"] == r["params_after"] < r["params_before"]
    before = reshapejob.inspect_model(str(src))["moe"]
    assert {x["experts"]: x for x in before["sizes"]}[32]["params"] == r["params_after"]
    assert sum(1 for p in final.glob("*.safetensors")) >= 2


def test_every_scoring_and_no_stray_files(tmp_path, folder):
    src, _, sd = folder
    for scoring in ("freq", "gate"):
        j = job(tmp_path, src, sd, scoring=scoring)
        code, ev = run_job(tmp_path, j)
        assert code == 0, ev[-1]
        meta = json.loads((Path(j["final"]) / "ndforge_reshape.json").read_text())
        assert meta["moe"]["scoring"] == scoring
        import shutil
        shutil.rmtree(Path(j["final"]).parent)


def test_pruning_and_a_top_k_cut_in_one_go_drops_nothing_else(tmp_path, folder):
    src, _, sd = folder
    j = {**job(tmp_path, src, sd), "top_k": 1}
    code, ev = run_job(tmp_path, j)
    assert code == 0
    cfg = json.loads((Path(j["final"]) / "config.json").read_text())
    assert cfg["text_config"]["n_routed_experts"] == 32 and cfg["text_config"]["num_experts_per_tok"] == 1
    assert (Path(j["final"]) / "mtp.safetensors").is_file()


def test_the_router_bias_is_nudged_only_when_the_measurement_has_the_pruned_loads(tmp_path, folder):
    src, made, sd = folder
    s = scores()
    kept = ml.select_experts(s, 32, "reap")
    s["pruned_load"] = {str(l): [1.0 if i == 0 else 0.0 for i in range(32)] for l in range(24)}         # expert 0 of every layer takes everything
    (sd / "moe_scores.json").write_text(json.dumps(s), encoding="utf-8")
    j = job(tmp_path, src, sd)
    code, ev = run_job(tmp_path, j)
    assert code == 0, ev[-1]
    got = stfile.read_folder(Path(j["final"]))
    name = "model.llm.layers.7.mlp.gate.bias"
    new = struct.unpack("<32f", got[name][2])
    old = struct.unpack(f"<{E}f", made["tensors"][name][2])
    base = [old[e] for e in kept[7]]
    assert new[0] < base[0] and all(n > b for n, b in zip(new[1:], base[1:]))         # the overloaded expert is made less likely, the idle ones more
    assert json.loads((Path(j["final"]) / "ndforge_reshape.json").read_text())["moe"]["rebalanced"] is True


def test_refusals(tmp_path, folder):
    src, _, sd = folder
    for kw, text in (({"keep": 40}, "multiple of 16"), ({"keep": 16}, "at least 32"), ({"keep": 64}, "smaller"), ({"scoring": "magic"}, "Unknown way")):
        j = job(tmp_path, src, sd, **kw)
        code, ev = run_job(tmp_path, j)
        assert code != 0 and text in ev[-1]["message"], (kw, ev[-1])
        assert not Path(j["final"]).exists() and not Path(j["out"]).exists()
    empty = tmp_path / "none"
    empty.mkdir()
    code, ev = run_job(tmp_path, job(tmp_path, src, empty))
    assert code != 0 and "measured" in ev[-1]["message"]


def test_rebalance_arithmetic():
    assert ml.rebalance_bias([0.0, 0.0], [0.9, 0.1], [0.5, 0.5], gamma=0.01, iterations=10) == pytest.approx([-0.1, 0.1])
    assert ml.rebalance_bias([0.0, 0.0], [0.5, 0.5], [0.5, 0.5]) == [0.0, 0.0]            # already even: left alone
