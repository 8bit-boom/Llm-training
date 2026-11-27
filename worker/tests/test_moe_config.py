"""Config surgery of Inkling-2-Mini cuts (golden) and the config-only cut job on a fake packed folder."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

import inkfake
import stfile
from ndforge_worker import moe_layout as ml

WORKER = Path(__file__).resolve().parents[1]


@pytest.fixture()
def cfg():
    return inkfake.real_config()


def tc(c):
    return c["text_config"]


def test_passes_change_one_field_and_drop_the_drafter(cfg):
    notes = []
    new = ml.fix_config_inkling(cfg, passes=2, notes=notes)
    assert tc(new)["loop_config"] == {**tc(cfg)["loop_config"], "num_passes": 2}
    assert "mtp_config" not in new and "mtp_config" in cfg
    assert any("speculative" in n for n in notes)
    # nothing else moved
    a, b = copy.deepcopy(cfg), copy.deepcopy(new)
    a.pop("mtp_config")
    tc(a)["loop_config"]["num_passes"] = 2
    assert a == b


def test_more_passes_are_allowed_as_a_question_to_try(cfg):
    assert tc(ml.fix_config_inkling(cfg, passes=4))["loop_config"]["num_passes"] == 4
    with pytest.raises(ml.MoeError, match="More than 8"):
        ml.fix_config_inkling(cfg, passes=9)


def test_one_pass_only_for_a_variable_loop(cfg):
    with pytest.raises(ml.MoeError, match="fixed loop"):
        ml.fix_config_inkling(cfg, passes=1)
    var = copy.deepcopy(cfg)
    tc(var)["loop_config"]["is_varloop"] = True
    assert tc(ml.fix_config_inkling(var, passes=1))["loop_config"]["num_passes"] == 1
    flat = copy.deepcopy(cfg)
    del tc(flat)["loop_config"]
    with pytest.raises(ml.MoeError, match="no loop"):
        ml.fix_config_inkling(flat, passes=2)


def test_top_k_keeps_the_drafter(cfg):
    new = ml.fix_config_inkling(cfg, top_k=6)
    assert tc(new)["num_experts_per_tok"] == 6 and new["mtp_config"] == cfg["mtp_config"]
    assert tc(ml.fix_config_inkling(cfg, top_k=10))["num_experts_per_tok"] == 10          # growing is a question to try as well
    for bad in (0, 33):
        with pytest.raises(ml.MoeError):
            ml.fix_config_inkling(cfg, top_k=bad)


def test_experts_follow_the_rules(cfg):
    new = ml.fix_config_inkling(cfg, experts=96)
    assert tc(new)["n_routed_experts"] == 96 and tc(new)["num_experts_per_tok"] == 8 and "mtp_config" in new
    for bad in (100, 16, 48):
        with pytest.raises(ml.MoeError):
            ml.fix_config_inkling(cfg, experts=bad)


def test_whole_blocks_renumber_the_loop(cfg):
    new = ml.fix_config_inkling(cfg, drop_blocks=[3])                  # the coda
    t = tc(new)
    assert t["num_hidden_layers"] == 18 and t["loop_config"]["core_start"] == 6 and t["loop_config"]["core_end"] == 18
    assert t["local_layer_ids"] == [i for i in tc(cfg)["local_layer_ids"] if i < 18] and "mtp_config" not in new
    new = ml.fix_config_inkling(cfg, drop_blocks=[1])                  # one core block: the core shrinks to one block
    t = tc(new)
    assert t["num_hidden_layers"] == 18 and (t["loop_config"]["core_start"], t["loop_config"]["core_end"]) == (6, 12)
    assert t["local_layer_ids"] == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16]
    assert ml.layer_kinds(new).count("full_attention") == 3
    for bad in ([0], [], [4], [1, 2, 3]):
        with pytest.raises(ml.MoeError):
            ml.fix_config_inkling(cfg, drop_blocks=bad)
    # two core blocks go together: no loop is left to run
    with pytest.raises(ml.MoeError, match="whole loop"):
        ml.fix_config_inkling(cfg, drop_blocks=[1, 2])


# ------------------------------------------------------------------------------------------------------------------ the job

def run_job(tmp_path, job):
    f = tmp_path / "job.json"
    f.write_text(json.dumps(job), encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "ndforge_worker", str(f)], cwd=WORKER, capture_output=True, text=True, timeout=120)
    return p.returncode, [json.loads(line) for line in p.stdout.splitlines()]


def job(tmp_path, src, **kw):
    d = tmp_path / "models" / "m1"
    d.mkdir(parents=True, exist_ok=True)
    return {"job": "reshape", "source": {"repo": str(src), "family": "inkling2"}, "mode": "prune", "out": str(d / "model.partial"), "final": str(d / "model"), "shard_gb": 5, **kw}


RESULT_KEYS = {"model_dir", "mode", "layers_before", "layers_after", "src_layers", "dropped", "duplicated", "zero_init", "params_before", "params_after", "params_breakdown",
               "bytes_written", "loss_before", "loss_after", "ppl_before", "ppl_after", "stable", "boundary_distance", "chosen_blocks", "load", "seconds",
               "notes", "stopped", "shape", "width"}


def test_a_pass_cut_copies_the_weights_and_removes_the_drafter(tmp_path):
    src = tmp_path / "ink"
    made = inkfake.write_folder(src)
    j = job(tmp_path, src, passes=2)
    code, ev = run_job(tmp_path, j)
    assert code == 0 and ev[-1]["type"] == "result", ev[-1]
    r = ev[-1]["data"]
    assert RESULT_KEYS <= set(r) and r["moe"]["passes"] == [3, 2] and r["moe"]["visits"] == [48, 36] and r["moe"]["mtp"] == "removed" and r["stopped"] is False
    final = Path(j["final"])
    assert final.is_dir() and not Path(j["out"]).exists()
    new = json.loads((final / "config.json").read_text())
    assert tc(new)["loop_config"]["num_passes"] == 2 and "mtp_config" not in new
    assert not (final / "mtp.safetensors").exists()
    q = json.loads((final / "hf_quant_config.json").read_text())
    assert "mtp" not in q and q["model"] == json.loads((src / "hf_quant_config.json").read_text())["model"]
    # every tensor is the original's bytes
    got = stfile.read_folder(final)
    want = {k: v for k, v in made["tensors"].items()}
    assert set(got) == set(want) and all(got[k][2] == want[k][2] for k in want)
    assert json.loads((final / "model.safetensors.index.json").read_text()) == json.loads((src / "model.safetensors.index.json").read_text())
    meta = json.loads((final / "ndforge_reshape.json").read_text())
    assert meta["shape"] == "moe" and meta["moe"]["weights"] == "copied unchanged" and meta["origin"]["layers"] == 24
    for name in ("tokenizer_config.json", "processor_config.json", "generation_config.json"):
        assert (final / name).read_bytes() == (src / name).read_bytes()
    # and it reads as an Inkling folder again, with 2 passes
    from ndforge_worker import reshapejob
    ins = reshapejob.inspect_model(str(final))
    assert ins["supported"] and ins["moe"]["passes"] == 2 and ins["moe"]["visits"] == 36 and ins["total_params"] == r["params_after"]


def test_a_top_k_cut_keeps_the_drafter_file(tmp_path):
    src = tmp_path / "ink"
    inkfake.write_folder(src)
    j = job(tmp_path, src, top_k=1)
    code, ev = run_job(tmp_path, j)
    r = ev[-1]["data"]
    assert code == 0 and r["moe"]["top_k"] == [2, 1] and r["moe"]["mtp"] == "kept, not healed"
    final = Path(j["final"])
    assert (final / "mtp.safetensors").read_bytes() == (src / "mtp.safetensors").read_bytes()
    assert "mtp" in json.loads((final / "hf_quant_config.json").read_text())
    assert json.loads((final / "config.json").read_text())["mtp_config"] == json.loads((src / "config.json").read_text())["mtp_config"]
    assert any("not healed" in n for n in r["notes"])


def test_refusals_are_plain_and_leave_nothing_behind(tmp_path):
    src = tmp_path / "ink"
    inkfake.write_folder(src)
    for kw, text in (({"passes": 1}, "fixed loop"), ({}, "Nothing to change"), ({"top_k": 99}, "Experts per token"), ({"passes": 2, "width": {"intermediate": 8}}, "does not apply"),
                     ({"mode": "grow", "passes": 4}, "Only cutting")):
        j = job(tmp_path, src, **kw)
        code, ev = run_job(tmp_path, j)
        assert code != 0 and ev[-1]["type"] == "error" and text in ev[-1]["message"], (kw, ev[-1])
        assert not Path(j["final"]).exists() and not Path(j["out"]).exists()
