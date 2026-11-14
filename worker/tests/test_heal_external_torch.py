"""Healing a model that the Slim page did not make (CPU, tiny models): a thinned folder WITHOUT ndforge_reshape.json, with the teacher named explicitly."""
import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")

from test_distill_torch import distill_cfg, make_cache, metrics, run_in_process, teacher, thin_student  # noqa: E402,F401
from test_reshape_torch import IDS, load  # noqa: E402

from ndforge_worker import distilljob as dj  # noqa: E402
from ndforge_worker import healcheck  # noqa: E402


def external_student(tmp_path, teacher):
    """the thinned tiny model as if it came from elsewhere: no ndforge_reshape.json, no width_kept.npz"""
    made = Path(thin_student(tmp_path, teacher))
    ext = tmp_path / "somebody-elses-pruned-model"
    shutil.copytree(made, ext)
    (ext / "ndforge_reshape.json").unlink(missing_ok=True)
    (ext / "width_kept.npz").unlink(missing_ok=True)
    assert not (ext / "ndforge_reshape.json").exists()
    return ext


def test_an_external_student_is_healed_through_cache_distill_and_merge(tmp_path, teacher, monkeypatch):
    student = external_student(tmp_path, teacher)
    # the check sees the pair: same tokenizer, a thinner student
    facts = healcheck.facts_of(str(student)), healcheck.facts_of(str(teacher))
    assert facts[0]["tokenizer_hash"] == facts[1]["tokenizer_hash"] and facts[0]["params"] < facts[1]["params"] and facts[0]["error"] is None
    make_cache(tmp_path, student, teacher, monkeypatch)
    cfg = distill_cfg(tmp_path, student, teacher, "cache")
    cfg["eval"].pop("teacher_loss")                                                # nothing recorded about "before healing": the job computes it fresh
    code, ev = run_in_process(monkeypatch, dj.run, cfg)
    assert code == 0 and ev[-1]["type"] == "result", ev[-1]
    r = ev[-1]["data"]
    kd = [e["kd"] for e in metrics(ev)]
    assert r["steps"] == 20 and sum(kd[-5:]) < sum(kd[:5]) and r["calib_loss_before"] is not None and r["teacher_calib_loss"] is None
    final = tmp_path / "healed"
    mcfg = {"job": "distill_merge", "base": str(student), "adapter": r["adapter_dir"], "out": str(tmp_path / "healed.partial"), "final": str(final), "summary": r,
            "origin": {"repo": str(teacher), "family": "qwen3_5", "params": 123}, "student_label": "somebody-elses-pruned-model", "stop_file": str(tmp_path / "STOP2")}
    code, ev = run_in_process(monkeypatch, dj.merge, mcfg)
    assert code == 0 and ev[-1]["data"]["model_dir"] == str(final), ev[-1]
    # a complete model folder: config, tokenizer files, weights, a README with the licence line, and the metadata the later steps read
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors", "README.md", "ndforge_reshape.json"):
        assert (final / name).is_file(), name
    meta = json.loads((final / "ndforge_reshape.json").read_text(encoding="utf-8"))
    assert meta["origin"]["repo"] == str(teacher) and meta["healed"]["from"] == "somebody-elses-pruned-model" and meta["healed"]["lora_rank"] == 8
    assert meta["healed"]["kd_last"] == r["kd_last"] and meta["layers"] == 4 and meta["params"] > 0 and meta["healed_from"] == "somebody-elses-pruned-model"
    assert not (final / "width_kept.npz").exists()
    healed = load(final)                                                           # it loads, and it differs from the student it started from
    plain = load(student)
    assert not torch.allclose(healed(input_ids=IDS, use_cache=False).logits, plain(input_ids=IDS, use_cache=False).logits)
    assert healcheck.facts_of(str(final))["error"] is None


def test_the_merge_of_a_made_model_still_keeps_its_own_metadata(tmp_path, teacher, monkeypatch):
    made = Path(thin_student(tmp_path, teacher))
    meta = {"format": 2, "origin": {"repo": "acme/Original", "family": "qwen3_5", "params": 5}, "mode": "prune", "layers": 4, "params": 77}
    (made / "ndforge_reshape.json").write_text(json.dumps(meta), encoding="utf-8")
    make_cache(tmp_path, made, teacher, monkeypatch)
    code, ev = run_in_process(monkeypatch, dj.run, distill_cfg(tmp_path, made, teacher, "cache"))
    assert code == 0, ev[-1]
    r = ev[-1]["data"]
    mcfg = {"job": "distill_merge", "base": str(made), "adapter": r["adapter_dir"], "out": str(tmp_path / "h.partial"), "final": str(tmp_path / "h"), "summary": r, "stop_file": str(tmp_path / "S")}
    code, ev = run_in_process(monkeypatch, dj.merge, mcfg)
    assert code == 0, ev[-1]
    out = json.loads((tmp_path / "h" / "ndforge_reshape.json").read_text(encoding="utf-8"))
    assert out["origin"]["repo"] == "acme/Original" and out["mode"] == "prune" and out["params"] == 77 and "healed_from" not in out and out["healed"]["lora_rank"] == 8
