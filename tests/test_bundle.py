import json
import zipfile
from pathlib import Path

import pytest

from ndforge.ingest.bundle import BundleError, open_bundle, write_bundle
from ndforge.ingest.diff import diff_bundles
from sample_bundle import WORLD, build


def test_roundtrip_and_summary(tmp_path):
    b = open_bundle(build(tmp_path / "a.ndtrain"))
    assert b.world_name == "Ashfall Reach" and b.counts["entities.jsonl"] == 2
    assert "2 entities" in b.summary_lines() and "world rules text" in b.summary_lines()
    names = [r["name"] for r in b.records("entities.jsonl")]
    assert "Anders de Vallière" in names  # accents survive
    assert list(b.records("quests.jsonl")) == []  # optional member absent


def test_writer_is_deterministic(tmp_path):
    a = build(tmp_path / "a.ndtrain").read_bytes()
    b = build(tmp_path / "b.ndtrain").read_bytes()
    assert a == b
    with zipfile.ZipFile(tmp_path / "a.ndtrain") as z:
        assert z.namelist()[0] == "manifest.json"
        assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in z.infolist())


def test_committed_fixture_is_current():
    fixture = Path(__file__).parent / "fixtures" / "sample.ndtrain"
    if fixture.exists():
        assert open_bundle(fixture).world_name == "Ashfall Reach"


def test_rejects_not_a_zip(tmp_path):
    f = tmp_path / "x.ndtrain"
    f.write_text("hello")
    with pytest.raises(BundleError, match="not a zip"):
        open_bundle(f)


def test_rejects_wrong_format_and_newer_version(tmp_path):
    p = build(tmp_path / "a.ndtrain")
    for patch, text in (({"format": "other"}, "not a training bundle"), ({"version": 99}, "Update ND Lore Forge")):
        out = tmp_path / "b.ndtrain"
        with zipfile.ZipFile(p) as zin, zipfile.ZipFile(out, "w") as zout:
            for n in zin.namelist():
                data = zin.read(n)
                if n == "manifest.json":
                    data = json.dumps({**json.loads(data), **patch}).encode()
                zout.writestr(n, data)
        with pytest.raises(BundleError, match=text):
            open_bundle(out)


def test_detects_tampering_and_extra_files(tmp_path):
    p = build(tmp_path / "a.ndtrain")
    out = tmp_path / "t.ndtrain"
    with zipfile.ZipFile(p) as zin, zipfile.ZipFile(out, "w") as zout:
        for n in zin.namelist():
            zout.writestr(n, zin.read(n) + (b"x" if n == "facts.jsonl" else b""))
    with pytest.raises(BundleError, match="damaged"):
        open_bundle(out)
    out2 = tmp_path / "e.ndtrain"
    with zipfile.ZipFile(p) as zin, zipfile.ZipFile(out2, "w") as zout:
        for n in zin.namelist():
            zout.writestr(n, zin.read(n))
        zout.writestr("users.jsonl", b"{}\n")
    with pytest.raises(BundleError, match="not in its manifest"):
        open_bundle(out2)


def test_writer_rejects_unknown_member(tmp_path):
    with pytest.raises(ValueError):
        write_bundle(tmp_path / "x.ndtrain", world=WORLD, members={"users.jsonl": []})


def test_diff(tmp_path):
    d = diff_bundles(open_bundle(build(tmp_path / "1.ndtrain", 1)), open_bundle(build(tmp_path / "2.ndtrain", 2)))
    ent = d.kinds["entities.jsonl"]
    assert ent.added == ["The Greyfather Pack"] and ent.changed == ["Anders de Vallière"] and ent.removed == ["Dockside"]
    assert "1 new, 1 changed, 1 removed" in d.summary()
    same = diff_bundles(open_bundle(tmp_path / "1.ndtrain"), open_bundle(tmp_path / "1.ndtrain"))
    assert same.summary() == "Nothing changed."
