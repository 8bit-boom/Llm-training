"""The Triton pin comes from the torch wheel's own requirement; the table is the fallback; the heal-speed report says whether the installed Triton matches."""
from ndforge_worker import healspeed, tritonpin


def req(v):
    return [f'triton=={v}; platform_system == "Linux" and platform_machine == "x86_64"', "pytorch-triton-rocm==9.9; extra == 'rocm'"]


def test_the_wheel_requirement_gives_the_pin_for_torch_2_11_2_14_and_2_15():
    assert tritonpin.pin_for("2.11.0+cu128", req("3.6.0")) == ("3.6", "wheel")
    assert tritonpin.pin_for("2.14.1", req("3.8.0")) == ("3.8", "wheel")
    assert tritonpin.pin_for("2.15.0", req("3.9.1")) == ("3.9", "wheel")


def test_without_a_requirement_the_table_then_unknown():
    assert tritonpin.pin_for("2.14.1+cu130", []) == ("3.8", "table")
    assert tritonpin.pin_for("2.15.0", []) == ("3.9", "table")
    assert tritonpin.pin_for("2.11.0+cu128", []) == (None, None)
    assert tritonpin.pin_for("2.11.0", ["pytorch-triton==3.0.0", "triton-windows==3.9.0"]) == (None, None)


def test_the_owners_case_is_called_a_mismatch_in_plain_words():
    s = tritonpin.status("2.11.0+cu128", "3.9.0", req("3.6.0"))
    assert s["matches"] is False and s["wants"] == "3.6"
    assert s["text"] == "Triton 3.9.0 does not match torch 2.11.0 (wants 3.6.x): press Install fast kernels to fix."
    ok = tritonpin.status("2.11.0+cu128", "3.6.2", req("3.6.0"))
    assert ok["matches"] is True and "matches" in ok["text"]
    assert tritonpin.status("2.11.0", None, req("3.6.0"))["text"].startswith("PyTorch 2.11.0: no Triton")
    assert tritonpin.status("2.11.0", "3.9.0", [])["matches"] is None


def test_the_heal_speed_report_prints_torch_triton_and_the_match(monkeypatch):
    monkeypatch.setattr(tritonpin, "installed_status", lambda: tritonpin.status("2.11.0+cu128", "3.9.0", req("3.6.0")))
    facts = healspeed.environment_facts()
    assert facts["triton_match"]["matches"] is False
    rep = {"environment": facts, "kernels": {}}
    assert any("does not match torch 2.11.0" in l for l in healspeed.verdict_lines(rep))
