"""scripts/check_release_version.py: the release tag must be v<Cargo.toml version>, and no other file may carry a different version."""
import importlib.util
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_release_version", REPO / "scripts" / "check_release_version.py")
crv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crv)


def copy(tmp: Path) -> Path:
    for rel in ["Cargo.toml", "src-tauri/Cargo.toml", "src-tauri/tauri.conf.json", "crates/ndforge-core/Cargo.toml", "packaging/windows/ndforge.iss"]:
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp / rel)
    return tmp


def version() -> str:
    import re
    return re.search(r'\[workspace\.package\][^\[]*?^version\s*=\s*"([^"]+)"', (REPO / "Cargo.toml").read_text(), re.S | re.M).group(1)


def test_the_repository_agrees_with_itself_and_with_its_own_tag():
    assert crv.problems(None) == []
    assert crv.problems("v" + version()) == []


def test_a_tag_that_differs_is_refused():
    assert crv.problems("v" + version() + "1")
    assert crv.problems(version()), "the tag needs the v"
    assert crv.problems("v9.9.9")


def test_a_second_version_in_tauri_conf_or_the_installer_script_is_refused(tmp_path):
    root = copy(tmp_path)
    conf = root / "src-tauri/tauri.conf.json"
    conf.write_text(conf.read_text().replace('"productName"', '"version": "9.9.9", "productName"', 1))
    assert any("tauri.conf.json" in p for p in crv.problems(None, root))
    conf.write_text(conf.read_text().replace('"version": "9.9.9", ', '"version": "%s", ' % version(), 1))
    assert crv.problems(None, root) == [], "the same number is fine"
    iss = root / "packaging/windows/ndforge.iss"
    iss.write_text(iss.read_text().replace('#define AppVersion "%s"' % version(), '#define AppVersion "9.9.9"'))
    assert any("ndforge.iss" in p for p in crv.problems(None, root))


def test_a_crate_with_its_own_version_is_refused(tmp_path):
    root = copy(tmp_path)
    p = root / "crates/ndforge-core/Cargo.toml"
    p.write_text(p.read_text().replace("version.workspace = true", 'version = "1.0.0"'))
    assert any("ndforge-core" in x for x in crv.problems(None, root))
