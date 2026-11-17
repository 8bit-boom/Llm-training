"""Code that is shared with (ported from or into) other repositories is pinned by hash, so a drift is a loud test failure and not a silent difference."""
import hashlib
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "ndforge_worker"
KQUANTS_BODY_SHA256 = "5de063b81a36f52cdf8258e596aec8c9ba59774c9a621db1ca058be41b368d33"


def test_kquants_body_is_the_port():
    """kquants.py from `from __future__` on is quant-convert-gui's quant_gui/kquants.py at a6920274, byte for byte. (Recompute with
    `git show a6920274:quant_gui/kquants.py | sed -n '/^from __future__/,$p' | sha256sum` in that repository.)"""
    text = (WORKER / "kquants.py").read_bytes().decode("utf-8")
    body = text[text.index("\nfrom __future__ import annotations") + 1:]
    got = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert got == KQUANTS_BODY_SHA256, (
        "ndforge_worker/kquants.py no longer matches the port of quant-convert-gui/quant_gui/kquants.py (a6920274); it is a verbatim port, "
        "change it there (outside this repo) and update the port and this hash together")


def test_kquants_keeps_both_licence_notices():
    head = (WORKER / "kquants.py").read_text(encoding="utf-8").split("\nfrom __future__ import annotations")[0]
    assert "Copyright (c) 2026 8bit-boom" in head and "Copyright (c) 2023-2026 The ggml authors" in head
    assert head.count("Permission is hereby granted") == 2


def test_golden_fixture_is_the_one_ggml_made():
    p = Path(__file__).parent / "fixtures" / "kquants_golden.npz"
    assert p.stat().st_size < 40_000
    assert hashlib.sha256(p.read_bytes()).hexdigest() == "ad85273498165f6de953cf983705359995d7b63fa683daa07ea5c71fdf87df36"
