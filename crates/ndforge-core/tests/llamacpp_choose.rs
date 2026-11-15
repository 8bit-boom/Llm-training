//! llama.cpp installer, pure parts: which release file fits which computer (many realistic asset lists), nvidia-smi parsing, the safe unpacking, the source subset,
//! the record, the search order. No network (the install flow with a stand-in server is in llamacpp_install.rs).
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use ndforge_core::envmgr::gpu::parse_nvidia_smi;
use ndforge_core::envmgr::llamacpp::archive::{link_stays_inside, parse_digest, safe_parts, unpack, EntryKind};
use ndforge_core::envmgr::llamacpp::choose::*;
use ndforge_core::envmgr::llamacpp::*;
use tempfile::tempdir;

mod common;
use common::fake_github::*;

static ENV: Mutex<()> = Mutex::new(());

fn assets(names: &[&str]) -> Vec<Asset> {
    names.iter().map(|n| Asset::new(n, 1_000_000)).collect()
}

/// A recent release's files (from memory of the project's naming; the real list is in docs/LLAMA_CPP_INSTALL.md under "Not verified yet").
fn new_style() -> Vec<Asset> {
    assets(&[
        "llama-b7000-bin-win-cpu-x64.zip", "llama-b7000-bin-win-cpu-arm64.zip", "llama-b7000-bin-win-cuda-12.4-x64.zip", "llama-b7000-bin-win-cuda-13.1-x64.zip", "llama-b7000-bin-win-vulkan-x64.zip",
        "llama-b7000-bin-win-hip-radeon-x64.zip", "llama-b7000-bin-win-sycl-x64.zip", "llama-b7000-bin-win-opencl-adreno-arm64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-13.1-x64.zip",
        "llama-b7000-bin-ubuntu-x64.tar.gz", "llama-b7000-bin-ubuntu-vulkan-x64.tar.gz", "llama-b7000-bin-ubuntu-arm64.tar.gz", "llama-b7000-bin-ubuntu-rocm-7.0-x64.tar.gz", "llama-b7000-bin-ubuntu-openvino-x64.tar.gz",
        "llama-b7000-bin-macos-arm64.tar.gz", "llama-b7000-bin-macos-x64.tar.gz", "llama-b7000-bin-macos-arm64-kleidiai.tar.gz", "llama-b7000-xcframework.zip", "llama-b7000-bin-win-cuda-12.4-x64-ci.zip",
    ])
}

fn with_128() -> Vec<Asset> {
    let mut v = new_style();
    v.extend(assets(&["llama-b7000-bin-win-cuda-12.8-x64.zip", "cudart-llama-bin-win-cuda-12.8-x64.zip"]));
    v
}

fn old_style() -> Vec<Asset> {
    assets(&[
        "llama-b4000-bin-win-avx-x64.zip", "llama-b4000-bin-win-avx2-x64.zip", "llama-b4000-bin-win-avx512-x64.zip", "llama-b4000-bin-win-noavx-x64.zip", "llama-b4000-bin-win-openblas-x64.zip",
        "llama-b4000-bin-win-cuda-cu11.7.1-x64.zip", "llama-b4000-bin-win-cuda-cu12.2.0-x64.zip", "cudart-llama-bin-win-cu11.7.1-x64.zip", "cudart-llama-bin-win-cu12.2.0-x64.zip",
        "llama-b4000-bin-win-vulkan-x64.zip", "llama-b4000-bin-win-kompute-x64.zip", "llama-b4000-bin-ubuntu-x64.zip", "llama-b4000-bin-macos-arm64.zip", "llama-b4000-bin-macos-x64.zip",
    ])
}

fn nv(name: &str, cuda: Option<(u32, u32)>, cc: Option<(u32, u32)>) -> Nvidia {
    Nvidia { name: name.into(), driver_cuda: cuda, compute_cap: cc, blackwell: cc.is_some_and(|c| c.0 >= 10) }
}

fn win(n: Option<Nvidia>, vulkan: bool) -> Machine {
    Machine { os: Os::Windows, arch: Arch::X64, nvidia: n, vulkan_likely: vulkan, tools: BuildTools { nvcc: false, cmake: false, cxx: false } }
}

fn linux(n: Option<Nvidia>, tools: bool) -> Machine {
    Machine { os: Os::Linux, arch: Arch::X64, nvidia: n, vulkan_likely: false, tools: BuildTools { nvcc: tools, cmake: tools, cxx: tools } }
}

fn names(c: &Choice) -> Vec<&str> {
    c.assets.iter().map(|a| a.name.as_str()).collect()
}

#[test]
fn blackwell_on_windows_needs_cuda_12_8_or_newer_and_gets_its_runtime_files() {
    let m = win(Some(nv("NVIDIA GeForce RTX 5090", Some((12, 9)), Some((12, 0)))), false);
    let c = choose(&with_128(), &m, Want::Auto).unwrap();
    assert_eq!((c.backend.as_str(), c.cuda_version.as_deref()), ("cuda", Some("12.8")));
    assert_eq!(names(&c), ["llama-b7000-bin-win-cuda-12.8-x64.zip", "cudart-llama-bin-win-cuda-12.8-x64.zip"]);
    // a driver that supports CUDA 13 takes the 13.1 build when there is one
    let m13 = win(Some(nv("NVIDIA GeForce RTX 5090", Some((13, 1)), Some((12, 0)))), false);
    assert_eq!(choose(&with_128(), &m13, Want::Auto).unwrap().cuda_version.as_deref(), Some("13.1"));
    // the driver's 12.9 is below the 13.1 build: the highest that fits is 12.8
    assert_eq!(choose(&with_128(), &win(Some(nv("RTX 5080", Some((12, 9)), Some((12, 0)))), false), Want::Auto).unwrap().cuda_version.as_deref(), Some("12.8"));
}

#[test]
fn blackwell_never_gets_a_cuda_build_older_than_12_8() {
    let m = win(Some(nv("NVIDIA GeForce RTX 5090", Some((12, 9)), Some((12, 0)))), false);
    let c = choose(&new_style(), &m, Want::Auto).unwrap(); // only 12.4 and 13.1 exist: 13.1 is above the driver, 12.4 is too old
    assert_eq!(c.backend, "vulkan");
    assert!(c.notes.iter().any(|n| n.contains("too old for an RTX 50") && n.contains("13.1")), "{:?}", c.notes);
    let e = choose(&new_style(), &m, Want::Cuda).unwrap_err().0;
    assert!(e.contains("No CUDA build fits") && e.contains("12.8"), "{e}");
}

#[test]
fn an_older_card_gets_the_highest_cuda_not_above_the_driver() {
    let m = win(Some(nv("NVIDIA GeForce RTX 3080", Some((12, 6)), Some((8, 6)))), false);
    let c = choose(&with_128(), &m, Want::Auto).unwrap();
    assert_eq!((c.backend.as_str(), c.cuda_version.as_deref()), ("cuda", Some("12.4")));
    let m = win(Some(nv("NVIDIA GeForce RTX 3080", Some((12, 9)), Some((8, 6)))), false);
    assert_eq!(choose(&with_128(), &m, Want::Auto).unwrap().cuda_version.as_deref(), Some("12.8"));
}

#[test]
fn a_driver_older_than_every_cuda_build_falls_back_to_vulkan_then_cpu() {
    let m = win(Some(nv("GeForce GTX 1060", Some((11, 4)), Some((6, 1)))), false);
    let c = choose(&new_style(), &m, Want::Auto).unwrap();
    assert_eq!(c.backend, "vulkan");
    assert!(c.notes.iter().any(|n| n.contains("newer driver")));
    let no_vulkan = assets(&["llama-b7000-bin-win-cpu-x64.zip", "llama-b7000-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]);
    assert_eq!(choose(&no_vulkan, &m, Want::Auto).unwrap().backend, "cpu");
    // an unreadable driver version: no CUDA guess
    let unknown = win(Some(nv("RTX 4090", None, Some((8, 9)))), false);
    let c = choose(&new_style(), &unknown, Want::Auto).unwrap();
    assert_ne!(c.backend, "cuda");
    assert!(c.notes.iter().any(|n| n.contains("could not be read")));
}

#[test]
fn a_cuda_build_without_runtime_files_is_skipped_for_the_next_lower_one() {
    let a = assets(&["llama-b7000-bin-win-cuda-12.4-x64.zip", "llama-b7000-bin-win-cuda-12.8-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip", "llama-b7000-bin-win-cpu-x64.zip"]);
    let c = choose(&a, &win(Some(nv("RTX 4090", Some((12, 9)), Some((8, 9)))), false), Want::Auto).unwrap();
    assert_eq!(c.cuda_version.as_deref(), Some("12.4"));
    let blackwell = choose(&a, &win(Some(nv("RTX 5090", Some((12, 9)), Some((12, 0)))), false), Want::Auto).unwrap();
    assert_eq!(blackwell.backend, "cpu"); // 12.8 has no runtime package, 12.4 is too old
}

#[test]
fn no_nvidia_card_gets_vulkan_for_amd_or_intel_and_the_cpu_build_otherwise() {
    let c = choose(&new_style(), &win(None, true), Want::Auto).unwrap();
    assert_eq!((c.backend.as_str(), names(&c)), ("vulkan", vec!["llama-b7000-bin-win-vulkan-x64.zip"]));
    let c = choose(&new_style(), &win(None, false), Want::Auto).unwrap();
    assert_eq!((c.backend.as_str(), names(&c)), ("cpu", vec!["llama-b7000-bin-win-cpu-x64.zip"]));
    // AMD but this release has no Vulkan file
    let c = choose(&assets(&["llama-b7000-bin-win-cpu-x64.zip"]), &win(None, true), Want::Auto).unwrap();
    assert_eq!(c.backend, "cpu");
}

#[test]
fn extra_files_are_never_chosen() {
    let a = assets(&["llama-b7000-bin-win-cuda-12.4-x64-ci.zip", "llama-b7000-bin-win-hip-radeon-x64.zip", "llama-b7000-bin-win-sycl-x64.zip", "llama-b7000-bin-win-openvino-x64.zip", "llama-b7000-bin-win-opencl-adreno-arm64.zip", "llama-b7000-xcframework.zip", "llama-b7000-bin-win-cuda-13.1-x64.zip"]);
    let e = choose(&a, &win(None, false), Want::Auto).unwrap_err().0;
    assert!(e.contains("no file for this computer") && e.contains("llama-b7000-xcframework.zip") && e.contains("LLAMA_CPP_DIR"), "{e}");
}

#[test]
fn older_names_work_too() {
    let m = win(Some(nv("RTX 3060", Some((12, 4)), Some((8, 6)))), false);
    let c = choose(&old_style(), &m, Want::Auto).unwrap();
    assert_eq!(names(&c), ["llama-b4000-bin-win-cuda-cu12.2.0-x64.zip", "cudart-llama-bin-win-cu12.2.0-x64.zip"]);
    assert_eq!(c.cuda_version.as_deref(), Some("12.2"));
    // the best plain build among the old CPU variants is avx2
    let c = choose(&old_style(), &win(None, false), Want::Auto).unwrap();
    assert_eq!(names(&c), ["llama-b4000-bin-win-avx2-x64.zip"]);
    assert_eq!(choose(&old_style(), &win(None, false), Want::Vulkan).unwrap().backend, "vulkan");
}

#[test]
fn linux_gets_the_cpu_or_vulkan_build_and_cuda_is_a_build_option() {
    let gpu = nv("RTX 4090", Some((12, 8)), Some((8, 9)));
    let c = choose(&new_style(), &linux(Some(gpu.clone()), true), Want::Auto).unwrap();
    assert_eq!((c.backend.as_str(), names(&c)), ("cpu", vec!["llama-b7000-bin-ubuntu-x64.tar.gz"]));
    assert!(c.reason.contains("build one with CUDA"));
    let mut amd = linux(None, false);
    amd.vulkan_likely = true;
    assert_eq!(choose(&new_style(), &amd, Want::Auto).unwrap().backend, "vulkan");
    // the option
    let b = choose(&new_style(), &linux(Some(gpu.clone()), true), Want::Build).unwrap();
    assert_eq!((b.backend.as_str(), b.assets.len()), ("build", 0));
    let e = choose(&new_style(), &linux(Some(gpu.clone()), false), Want::Build).unwrap_err().0;
    assert!(e.contains("nvcc") && e.contains("cmake") && e.contains("C++ compiler"), "{e}");
    assert!(choose(&new_style(), &win(Some(gpu), false), Want::Build).unwrap_err().0.contains("Linux only"));
    // the options list says what is available
    let o = options(&new_style(), &linux(None, false));
    let ids: Vec<(&str, bool)> = o.iter().map(|x| (x.id, x.available)).collect();
    assert_eq!(ids, [("cuda", false), ("vulkan", true), ("cpu", true), ("build", false)]);
}

#[test]
fn macos_gets_metal_on_apple_silicon_and_the_cpu_build_on_intel() {
    let mac = |arch| Machine::cpu_only(Os::Macos, arch);
    let c = choose(&new_style(), &mac(Arch::Arm64), Want::Auto).unwrap();
    assert_eq!((c.backend.as_str(), names(&c)), ("metal", vec!["llama-b7000-bin-macos-arm64.tar.gz"])); // not the kleidiai one
    assert_eq!(choose(&new_style(), &mac(Arch::X64), Want::Auto).unwrap().backend, "cpu");
    assert_eq!(choose(&old_style(), &mac(Arch::Arm64), Want::Auto).unwrap().assets[0].name, "llama-b4000-bin-macos-arm64.zip");
    assert!(choose(&new_style(), &mac(Arch::Arm64), Want::Cuda).unwrap().notes[0].contains("no CUDA or Vulkan"));
}

#[test]
fn a_zip_is_preferred_over_a_tar_gz_of_the_same_build() {
    let a = assets(&["llama-b7000-bin-ubuntu-x64.tar.gz", "llama-b7000-bin-ubuntu-x64.zip"]);
    assert_eq!(choose(&a, &linux(None, false), Want::Auto).unwrap().assets[0].name, "llama-b7000-bin-ubuntu-x64.zip");
}

#[test]
fn an_unknown_list_is_a_plain_error_never_a_panic() {
    for list in [vec![], vec!["README.md", "source.tar.gz", "llama-.zip", "---.zip", "b123.zip"], vec!["llama-b7000-bin-freebsd-x64.zip"], vec!["llama-b7000-bin-win-cuda-x64.zip"]] {
        for m in [win(None, false), linux(None, false), Machine::cpu_only(Os::Macos, Arch::Arm64), Machine::cpu_only(Os::Other, Arch::Other), win(Some(nv("RTX 5090", Some((12, 9)), Some((12, 0)))), true)] {
            for w in [Want::Auto, Want::Cuda, Want::Vulkan, Want::Cpu, Want::Build] {
                let _ = choose(&assets(&list), &m, w); // must not panic
            }
        }
    }
    let e = choose(&assets(&["README.md", "weird.zip"]), &win(None, false), Want::Auto).unwrap_err().0;
    assert!(e.contains("README.md") && e.contains("weird.zip") && e.contains("docs/LLAMA_CPP_INSTALL.md"), "{e}");
    assert!(choose(&[], &win(None, false), Want::Auto).unwrap_err().0.contains("lists no files"));
    assert!(choose(&new_style(), &Machine::cpu_only(Os::Other, Arch::Other), Want::Auto).unwrap_err().0.contains("no ready-made"));
}

#[test]
fn the_names_are_read_by_tokens() {
    let p = parse_asset("cudart-llama-bin-win-cuda-12.4-x64.zip");
    assert!(p.cudart && p.kind == Kind::Cuda(Some((12, 4))) && p.os == Some(Os::Windows));
    let p = parse_asset("llama-b6000-bin-win-cuda-cu12.2.0-x64.zip");
    assert_eq!(p.kind, Kind::Cuda(Some((12, 2))));
    assert_eq!(parse_asset("llama-b6000-bin-ubuntu-x64.tar.gz").kind, Kind::Cpu(8));
    assert!(matches!(parse_asset("llama-b6000-bin-ubuntu-rocm-7.0-x64.tar.gz").kind, Kind::Other(_)));
    assert!(parse_asset("llama-b6000-bin-win-cpu-x64-CI.zip").ci);
    assert_eq!(parse_asset("LLAMA-B6000-BIN-WIN-VULKAN-X64.ZIP").kind, Kind::Vulkan);
}

const SMI: &str = "Sat Nov 14 10:12:01 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |\n|-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n|   0  NVIDIA GeForce RTX 5090        WDDM  |   00000000:01:00.0  On |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n";

#[test]
fn nvidia_smi_header_gives_the_drivers_cuda_version() {
    assert_eq!(parse_cuda_version(SMI), Some((12, 9)));
    assert_eq!(parse_cuda_version("| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |"), Some((12, 2)));
    assert_eq!(parse_cuda_version("CUDA Version: 13"), Some((13, 0)));
    assert_eq!(parse_cuda_version("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."), None);
    assert_eq!(parse_cuda_version(""), None);
    assert_eq!(parse_cuda_version("CUDA Version: N/A"), None);
}

#[test]
fn the_cuda_version_can_also_come_from_the_driver_number() {
    assert_eq!(cuda_from_driver(570), Some((12, 8)));
    assert_eq!(cuda_from_driver(572), Some((12, 8)));
    assert_eq!(cuda_from_driver(580), Some((13, 0)));
    assert_eq!(cuda_from_driver(535), Some((12, 2)));
    assert_eq!(cuda_from_driver(470), None);
    let gpus = parse_nvidia_smi("NVIDIA GeForce RTX 5090, 32607, 570.133.07, 12.0\n");
    let n = nvidia_from(&gpus, "garbage").unwrap();
    assert_eq!((n.driver_cuda, n.blackwell, n.compute_cap), (Some((12, 8)), true, Some((12, 0))));
    let n = nvidia_from(&gpus, SMI).unwrap();
    assert_eq!(n.driver_cuda, Some((12, 9)));
    assert!(nvidia_from(&[], SMI).is_none());
    let n = nvidia_from(&parse_nvidia_smi("Tesla T10, 16384, 535.104.05, 7.5\nNVIDIA GeForce RTX 5090, 32607, 570.133.07, 12.0\n"), SMI).unwrap();
    assert!(n.blackwell, "any Blackwell card in the machine counts");
}

#[test]
fn amd_and_intel_cards_are_recognised_for_vulkan() {
    assert!(vulkan_vendor_in("AMD Radeon RX 7900 XTX\n"));
    assert!(vulkan_vendor_in("Intel(R) Arc(TM) A770 Graphics"));
    assert!(vulkan_vendor_in("Intel(R) UHD Graphics 770"));
    assert!(!vulkan_vendor_in("NVIDIA GeForce RTX 5090"));
    assert!(!vulkan_vendor_in(""));
}

#[test]
fn digests_and_names_are_checked() {
    let h = "a".repeat(64);
    assert_eq!(parse_digest(&format!("sha256:{}", h.to_uppercase())), Some(h.clone()));
    assert_eq!(parse_digest("sha256:abc"), None);
    assert_eq!(parse_digest(&format!("md5:{h}")), None);
    assert_eq!(safe_parts("a/b/c.txt"), Some(vec!["a".into(), "b".into(), "c.txt".into()]));
    assert_eq!(safe_parts("./a//b"), Some(vec!["a".into(), "b".into()]));
    for bad in ["../x", "a/../../x", "/etc/passwd", "C:/Windows/x", "c:\\x", "a\\..\\..\\x", "", "/", "a\0b"] {
        assert_eq!(safe_parts(bad), None, "{bad:?}");
    }
    assert!(link_stays_inside(&["bin".into(), "libx.so".into()], "libx.so.1"));
    assert!(link_stays_inside(&["a".into(), "b".into(), "l".into()], "../c"));
    assert!(!link_stays_inside(&["bin".into(), "l".into()], "../../etc"));
    assert!(!link_stays_inside(&["l".into()], "../x"));
    assert!(!link_stays_inside(&["bin".into(), "l".into()], "/etc/passwd"));
    assert!(!link_stays_inside(&["bin".into(), "l".into()], "C:\\x"));
}

fn all(_: &[String], _: &EntryKind) -> Option<Vec<String>> {
    unreachable!()
}

fn everything(parts: &[String], _: &EntryKind) -> Option<Vec<String>> {
    Some(parts.to_vec())
}

#[test]
fn unpacking_refuses_every_name_that_leaves_the_folder_and_writes_nothing() {
    let _ = all;
    for bad in ["../evil.txt", "a/../../evil.txt", "/abs/evil.txt", "C:/evil.txt", "a\\..\\..\\evil.txt"] {
        let d = tempdir().unwrap();
        let z = d.path().join("x.zip");
        std::fs::write(&z, zip_bytes(&[("ok.txt", b"1", None), (bad, b"2", None)])).unwrap();
        let to = d.path().join("out").join("inner");
        let e = unpack(&z, Archive::Zip, &to, &everything, &AtomicBool::new(false)).unwrap_err().0;
        assert!(e.contains("unsafe path"), "{bad}: {e}");
        assert!(!to.join("ok.txt").exists(), "the names are all checked before anything is written ({bad})");
        assert!(!d.path().join("out").join("evil.txt").exists() && !d.path().join("evil.txt").exists());
    }
}

#[test]
fn unpacking_a_tar_gz_refuses_unsafe_names_hard_links_and_reads_long_names() {
    let d = tempdir().unwrap();
    let z = d.path().join("x.tar.gz");
    std::fs::write(&z, tar_gz(vec![tar_entry("ok.txt", b'0', b"1", "", 0o644), tar_entry("../evil.txt", b'0', b"2", "", 0o644)])).unwrap();
    assert!(unpack(&z, Archive::TarGz, &d.path().join("o"), &everything, &AtomicBool::new(false)).unwrap_err().0.contains("unsafe path"));
    std::fs::write(&z, tar_gz(vec![tar_entry("l", b'1', b"", "ok.txt", 0o644)])).unwrap();
    assert!(unpack(&z, Archive::TarGz, &d.path().join("o2"), &everything, &AtomicBool::new(false)).unwrap_err().0.contains("hard link"));
    // a GNU long name
    let long = format!("llama-b7000/{}/file.txt", "d".repeat(120));
    let mut long_data = long.clone().into_bytes();
    long_data.push(0);
    std::fs::write(&z, tar_gz(vec![tar_entry("././@LongLink", b'L', &long_data, "", 0o644), tar_entry("short", b'0', b"hello", "", 0o644), tar_entry("llama-b7000/", b'5', b"", "", 0o755), tar_entry("llama-b7000/run", b'0', b"#!/bin/sh\n", "", 0o755)])).unwrap();
    let n = unpack(&z, Archive::TarGz, &d.path().join("o3"), &everything, &AtomicBool::new(false)).unwrap();
    assert_eq!(n.files, 2);
    assert_eq!(std::fs::read(d.path().join("o3").join(&long)).unwrap(), b"hello");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(d.path().join("o3/llama-b7000/run")).unwrap().permissions().mode() & 0o111, 0o111);
    }
    // an incomplete archive is an error, not a half result
    let bytes = std::fs::read(&z).unwrap();
    std::fs::write(&z, &bytes[..bytes.len() / 2]).unwrap();
    assert!(unpack(&z, Archive::TarGz, &d.path().join("o4"), &everything, &AtomicBool::new(false)).is_err());
}

#[cfg(unix)]
#[test]
fn links_outside_the_tree_are_refused_inside_ones_are_made_and_the_exec_bit_is_set() {
    use std::os::unix::fs::PermissionsExt;
    let d = tempdir().unwrap();
    let mk = |name: &str, entries: Vec<Vec<u8>>| {
        let z = d.path().join(name);
        std::fs::write(&z, tar_gz(entries)).unwrap();
        z
    };
    let bad = mk("bad.tar.gz", vec![tar_entry("p/libx.so.1", b'0', b"x", "", 0o644), tar_entry("p/libx.so", b'2', b"", "../../../etc/passwd", 0o777)]);
    assert!(unpack(&bad, Archive::TarGz, &d.path().join("o1"), &everything, &AtomicBool::new(false)).unwrap_err().0.contains("outside"));
    let abs = mk("abs.tar.gz", vec![tar_entry("p/l", b'2', b"", "/etc/passwd", 0o777)]);
    assert!(unpack(&abs, Archive::TarGz, &d.path().join("o2"), &everything, &AtomicBool::new(false)).is_err());
    let good = mk("good.tar.gz", vec![tar_entry("p/libx.so.1", b'0', b"x", "", 0o644), tar_entry("p/libx.so", b'2', b"", "libx.so.1", 0o777), tar_entry("p/tool", b'0', b"#!/bin/sh\n", "", 0o755), tar_entry("p/data", b'0', b"d", "", 0o666)]);
    unpack(&good, Archive::TarGz, &d.path().join("o3"), &everything, &AtomicBool::new(false)).unwrap();
    assert_eq!(std::fs::read(d.path().join("o3/p/libx.so")).unwrap(), b"x");
    assert!(std::fs::symlink_metadata(d.path().join("o3/p/libx.so")).unwrap().file_type().is_symlink());
    assert_eq!(std::fs::metadata(d.path().join("o3/p/tool")).unwrap().permissions().mode() & 0o777, 0o755);
    assert_eq!(std::fs::metadata(d.path().join("o3/p/data")).unwrap().permissions().mode() & 0o777, 0o644, "no other permission bits are taken from the archive");
    // a zip's symlink entry (mode 0o120777) pointing out is refused as well
    let z = d.path().join("l.zip");
    std::fs::write(&z, zip_bytes(&[("p/l", b"../../x", Some(0o120777))])).unwrap();
    assert!(unpack(&z, Archive::Zip, &d.path().join("o4"), &everything, &AtomicBool::new(false)).unwrap_err().0.contains("outside"));
    let zok = d.path().join("lok.zip");
    std::fs::write(&zok, zip_bytes(&[("p/a", b"1", Some(0o644)), ("p/l", b"a", Some(0o120777)), ("p/run", b"x", Some(0o755))])).unwrap();
    unpack(&zok, Archive::Zip, &d.path().join("o5"), &everything, &AtomicBool::new(false)).unwrap();
    assert!(std::fs::symlink_metadata(d.path().join("o5/p/l")).unwrap().file_type().is_symlink());
    assert_eq!(std::fs::metadata(d.path().join("o5/p/run")).unwrap().permissions().mode() & 0o111, 0o111);
}

#[test]
fn a_cancel_stops_the_unpacking() {
    let d = tempdir().unwrap();
    let z = d.path().join("x.zip");
    std::fs::write(&z, zip_bytes(&[("a", b"1", None), ("b", b"2", None)])).unwrap();
    let e = unpack(&z, Archive::Zip, &d.path().join("o"), &everything, &AtomicBool::new(true)).unwrap_err().0;
    assert_eq!(e, "Cancelled.");
}

pub fn source_zip() -> Vec<u8> {
    zip_bytes(&[
        ("llama.cpp-b6000/", b"", None), ("llama.cpp-b6000/convert_hf_to_gguf.py", b"import torch\nfrom conversion import ModelBase\n", None), ("llama.cpp-b6000/convert_hf_to_gguf_update.py", b"x", None),
        ("llama.cpp-b6000/conversion/", b"", None), ("llama.cpp-b6000/conversion/__init__.py", b"", None), ("llama.cpp-b6000/conversion/base.py", b"x", None),
        ("llama.cpp-b6000/gguf-py/", b"", None), ("llama.cpp-b6000/gguf-py/gguf/__init__.py", b"", None), ("llama.cpp-b6000/gguf-py/gguf/vocab.py", b"x", None), ("llama.cpp-b6000/gguf-py/gguf/__pycache__/x.pyc", b"x", None),
        ("llama.cpp-b6000/gguf-py/tests/test_x.py", b"x", None), ("llama.cpp-b6000/gguf-py/examples/e.py", b"x", None), ("llama.cpp-b6000/gguf-py/pyproject.toml", b"x", None), ("llama.cpp-b6000/gguf-py/README.md", b"x", None),
        ("llama.cpp-b6000/models/ggml-vocab-llama-bpe.gguf", &[7u8; 1000], None), ("llama.cpp-b6000/models/templates/llama-cpp-rwkv-world.jinja", b"{{ x }}", None), ("llama.cpp-b6000/models/templates/Qwen-Qwen3-0.6B.jinja", b"{{ y }}", None),
        ("llama.cpp-b6000/requirements.txt", b"numpy", None), ("llama.cpp-b6000/requirements/requirements-convert_hf_to_gguf.txt", b"torch", None), ("llama.cpp-b6000/LICENSE", b"MIT", None),
        ("llama.cpp-b6000/docs/build.md", b"x", None), ("llama.cpp-b6000/src/llama.cpp", b"x", None), ("llama.cpp-b6000/tests/test-x.cpp", b"x", None), ("llama.cpp-b6000/CMakeLists.txt", b"project(x)", None),
    ])
}

#[test]
fn only_what_the_converter_needs_is_taken_from_the_source() {
    let d = tempdir().unwrap();
    let z = d.path().join("s.zip");
    std::fs::write(&z, source_zip()).unwrap();
    let root = d.path().join("root");
    unpack(&z, Archive::Zip, &root, &source_subset, &AtomicBool::new(false)).unwrap();
    let has = |p: &str| root.join(p).exists();
    for yes in ["convert_hf_to_gguf.py", "convert_hf_to_gguf_update.py", "conversion/__init__.py", "conversion/base.py", "gguf-py/gguf/__init__.py", "gguf-py/gguf/vocab.py", "gguf-py/pyproject.toml", "gguf-py/README.md",
                "models/templates/llama-cpp-rwkv-world.jinja", "models/templates/Qwen-Qwen3-0.6B.jinja", "requirements.txt", "requirements/requirements-convert_hf_to_gguf.txt", "LICENSE"] {
        assert!(has(yes), "{yes} should be unpacked");
    }
    for no in ["docs", "src", "tests", "CMakeLists.txt", "gguf-py/tests", "gguf-py/examples", "gguf-py/gguf/__pycache__", "models/ggml-vocab-llama-bpe.gguf", "llama.cpp-b6000"] {
        assert!(!has(no), "{no} should not be unpacked");
    }
    verify_source(&root).unwrap();
    std::fs::remove_dir_all(root.join("conversion")).unwrap();
    assert!(verify_source(&root).unwrap_err().message.contains("conversion"));
    std::fs::remove_dir_all(root.join("gguf-py")).unwrap();
    assert!(verify_source(&root).unwrap_err().message.contains("gguf-py"));
    // an older, single-file converter needs no conversion folder
    std::fs::write(root.join("convert_hf_to_gguf.py"), "import torch\n").unwrap();
    std::fs::create_dir_all(root.join("gguf-py/gguf")).unwrap();
    std::fs::write(root.join("gguf-py/gguf/__init__.py"), "").unwrap();
    verify_source(&root).unwrap();
}

#[test]
fn the_installed_layout_is_what_the_workers_finder_accepts() {
    let d = tempdir().unwrap();
    let r = d.path().join("llama.cpp");
    assert!(layout_of(&r).is_none());
    std::fs::create_dir_all(r.join("build/bin")).unwrap();
    std::fs::write(r.join(exe("llama-quantize")), "").unwrap();
    assert!(layout_of(&r).is_none(), "the converter is needed too");
    std::fs::write(r.join("convert_hf_to_gguf.py"), "").unwrap();
    assert_eq!(layout_of(&r).unwrap().1, r.join(exe("llama-quantize")));
    std::fs::remove_file(r.join(exe("llama-quantize"))).unwrap();
    assert!(layout_of(&r).is_none());
    std::fs::write(r.join("build/bin").join(exe("llama-quantize")), "").unwrap();
    assert_eq!(layout_of(&r).unwrap().1, r.join("build/bin").join(exe("llama-quantize")));
    std::fs::remove_file(r.join("build/bin").join(exe("llama-quantize"))).unwrap();
    std::fs::create_dir_all(r.join("build/bin/Release")).unwrap();
    std::fs::write(r.join("build/bin/Release").join(exe("llama-quantize")), "").unwrap();
    assert!(layout_of(&r).is_some());
}

fn cfg_in(d: &std::path::Path) -> Config {
    Config { base_dir: d.join("data"), api_base: "http://127.0.0.1:1".into(), web_base: "http://127.0.0.1:1".into(), allow_local: true, tag_pin: None, python: None, home: Some(d.join("home")), worker_dir: d.join("app/worker") }
}

fn fake_install(dir: &std::path::Path, tag: &str) {
    std::fs::create_dir_all(dir.join("build/bin")).unwrap();
    std::fs::write(dir.join("convert_hf_to_gguf.py"), "").unwrap();
    for t in ["llama-quantize", "llama-imatrix", "llama-perplexity"] {
        std::fs::write(dir.join("build/bin").join(exe(t)), "").unwrap();
    }
    let info = InstallInfo { schema: SCHEMA, tag: tag.into(), backend: "cpu".into(), ..Default::default() };
    std::fs::write(dir.join("install.json"), serde_json::to_vec(&info).unwrap()).unwrap();
}

#[test]
fn install_json_round_trips_and_the_status_tells_absent_old_corrupt_and_fine() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("LLAMA_CPP_DIR");
    let d = tempdir().unwrap();
    let cfg = cfg_in(d.path());
    // absent
    let s = status(&cfg);
    assert!(s["installed"].is_null() && s["install_error"].is_null() && s["found"].is_null() && !s["layout_ok"].as_bool().unwrap() && !s["folder_without_record"].as_bool().unwrap());
    // fine
    let dir = cfg.install_dir();
    fake_install(&dir, "b6000");
    let mut info = read_info(&dir).unwrap().unwrap();
    info.sha256.insert("a.zip".into(), "ff".into());
    info.versions.insert("llama-imatrix".into(), "version: 6000 (abc)".into());
    std::fs::write(dir.join("install.json"), serde_json::to_vec_pretty(&info).unwrap()).unwrap();
    assert_eq!(read_info(&dir).unwrap().unwrap(), info);
    let s = status(&cfg);
    assert_eq!(s["installed"]["tag"], "b6000");
    assert_eq!((s["found"]["how"].as_str(), s["found"]["how_label"].as_str()), (Some("program_install"), Some("installed by this program")));
    assert_eq!(s["tools"]["imatrix"]["how"], "program_install");
    assert!(s["layout_ok"].as_bool().unwrap() && s["converter_found"].as_bool().unwrap());
    // old: a record from before the schema field, with fields missing: still read
    std::fs::write(dir.join("install.json"), r#"{"tag": "b3000"}"#).unwrap();
    let old = read_info(&dir).unwrap().unwrap();
    assert_eq!((old.tag.as_str(), old.schema, old.backend.as_str()), ("b3000", 0, ""));
    assert_eq!(status(&cfg)["installed"]["tag"], "b3000");
    // corrupt
    std::fs::write(dir.join("install.json"), "{ not json").unwrap();
    let s = status(&cfg);
    assert!(s["installed"].is_null() && s["install_error"].as_str().unwrap().contains("damaged"));
    // a folder with no record
    std::fs::remove_file(dir.join("install.json")).unwrap();
    assert!(status(&cfg)["folder_without_record"].as_bool().unwrap());
}

#[test]
fn the_search_order_is_env_var_then_the_installed_folder_then_home_then_next_to_the_program() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    let cfg = cfg_in(d.path());
    let (mine, home, next, theirs) = (cfg.install_dir(), d.path().join("home/llama.cpp"), d.path().join("app/llama.cpp"), d.path().join("theirs"));
    std::env::remove_var("LLAMA_CPP_DIR");
    assert!(find(&cfg).is_none());
    fake_install(&next, "b1");
    assert_eq!(find(&cfg).unwrap().how, How::NextToProgram);
    fake_install(&home, "b1");
    assert_eq!(find(&cfg).unwrap().how, How::HomeFolder);
    fake_install(&mine, "b1");
    assert_eq!(find(&cfg).unwrap().how, How::ProgramInstall);
    fake_install(&theirs, "b1");
    std::env::set_var("LLAMA_CPP_DIR", &theirs);
    let f = find(&cfg).unwrap();
    assert_eq!((f.how, f.folder.clone()), (How::EnvVar, theirs.clone()), "an explicit LLAMA_CPP_DIR always wins");
    assert_eq!(status(&cfg)["found"]["how_label"], "your LLAMA_CPP_DIR folder");
    // a LLAMA_CPP_DIR that is not a llama.cpp does not hide the installed one
    std::env::set_var("LLAMA_CPP_DIR", d.path().join("nowhere"));
    assert_eq!(find(&cfg).unwrap().how, How::ProgramInstall);
    std::env::remove_var("LLAMA_CPP_DIR");
    assert_eq!(worker_env(&cfg), vec![("NDFORGE_LLAMA_CPP_DIR".to_string(), mine.to_string_lossy().to_string())]);
    std::fs::remove_dir_all(&mine).unwrap();
    assert!(worker_env(&cfg).is_empty());
}

#[test]
fn a_tool_is_found_next_to_quantize_then_in_any_candidate_then_on_the_path() {
    let d = tempdir().unwrap();
    let a = d.path().join("a");
    let b = d.path().join("b");
    let pathdir = d.path().join("p");
    for x in [&a, &b, &pathdir] {
        std::fs::create_dir_all(x).unwrap();
    }
    fake_install(&a, "b1");
    let cands = vec![(How::HomeFolder, a.clone()), (How::EnvVar, b.clone())];
    let found = find_in(&cands);
    let t = find_tool(&cands, found.as_ref(), "llama-imatrix", &[pathdir.clone()]).unwrap();
    assert_eq!((t.0, t.1), (a.join("build/bin").join(exe("llama-imatrix")), How::HomeFolder));
    // only llama-perplexity in a folder that is no complete llama.cpp
    std::fs::write(b.join(exe("llama-perplexity")), "").unwrap();
    std::fs::remove_file(a.join("build/bin").join(exe("llama-perplexity"))).unwrap();
    assert_eq!(find_tool(&cands, found.as_ref(), "llama-perplexity", &[pathdir.clone()]).unwrap(), (b.join(exe("llama-perplexity")), How::EnvVar));
    std::fs::remove_file(b.join(exe("llama-perplexity"))).unwrap();
    std::fs::write(pathdir.join(exe("llama-perplexity")), "").unwrap();
    assert_eq!(find_tool(&cands, found.as_ref(), "llama-perplexity", &[pathdir.clone()]).unwrap().1, How::Path);
    assert!(find_tool(&cands, found.as_ref(), "llama-cli", &[pathdir]).is_none());
}

#[test]
fn the_update_hint_compares_tags() {
    assert!(update_available("b6000", "b6001"));
    assert!(update_available("b999", "b1000"), "numbers, not text");
    assert!(!update_available("b6000", "b6000"));
    assert!(!update_available("b6001", "b6000"), "never suggests going back");
    assert!(update_available("v1", "v2"));
    assert!(!update_available("", "b6000"));
}

#[test]
fn remove_deletes_only_what_the_program_installed() {
    let d = tempdir().unwrap();
    let cfg = cfg_in(d.path());
    remove(&cfg).unwrap(); // nothing there
    let dir = cfg.install_dir();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("precious.txt"), "mine").unwrap();
    let e = remove(&cfg).unwrap_err();
    assert_eq!(e.code, "not_ours");
    assert!(dir.join("precious.txt").exists());
    fake_install(&dir, "b1");
    remove(&cfg).unwrap();
    assert!(!dir.exists());
    assert!(d.path().join("data").exists() || !dir.exists());
}

#[test]
fn a_swap_keeps_the_old_install_when_the_new_one_cannot_be_put_in_place() {
    let d = tempdir().unwrap();
    let cfg = cfg_in(d.path());
    std::fs::create_dir_all(&cfg.base_dir).unwrap();
    fake_install(&cfg.install_dir(), "b1");
    let e = swap_into_place(&cfg, &d.path().join("does-not-exist")).unwrap_err();
    assert_eq!(e.code, "in_use");
    assert_eq!(read_info(&cfg.install_dir()).unwrap().unwrap().tag, "b1");
    assert!(layout_of(&cfg.install_dir()).is_some());
    let new = cfg.base_dir.join("new");
    fake_install(&new, "b2");
    swap_into_place(&cfg, &new).unwrap();
    assert_eq!(read_info(&cfg.install_dir()).unwrap().unwrap().tag, "b2");
    assert!(!new.exists() && !cfg.base_dir.join("llama.cpp.old").exists());
}

#[test]
fn starting_fine_is_judged_from_what_the_program_did() {
    let ok = |code, out: &str| started_fine(&RunOut { started: true, code, output: out.into(), timed_out: false });
    assert!(ok(Some(0), "usage: llama-imatrix [options]"));
    assert!(ok(Some(1), "usage: ./llama-quantize [options] model-f32.gguf [model-quant.gguf] type [nthreads]"), "llama-quantize --help exits 1");
    assert!(!ok(Some(127), "sh: llama-x: not found"));
    assert!(!ok(Some(-1073741515), "x"), "Windows: a DLL is missing");
    assert!(!ok(None, "x"), "killed by a signal");
    assert!(!ok(Some(1), "./llama-imatrix: error while loading shared libraries: libggml.so: cannot open shared object file"));
    assert!(!ok(Some(1), "CUDA driver version is insufficient for CUDA runtime version"));
    assert!(!ok(Some(0), ""));
    assert!(!started_fine(&RunOut { started: false, ..Default::default() }));
    assert!(!started_fine(&RunOut { started: true, code: Some(0), output: "x".into(), timed_out: true }));
}

#[test]
fn the_hidden_settings_only_accept_a_server_on_this_computer() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("NDFORGE_LLAMA_CPP_API_URL", "https://evil.example.com");
    std::env::set_var("NDFORGE_LLAMA_CPP_WEB_URL", "https://evil.example.com");
    std::env::set_var("NDFORGE_LLAMA_CPP_TAG", " b5000 ");
    let c = Config::from_env();
    assert!(!c.allow_local);
    assert_eq!((c.api_base.as_str(), c.web_base.as_str(), c.tag_pin.as_deref()), ("https://api.github.com", "https://github.com", Some("b5000")));
    std::env::set_var("NDFORGE_LLAMA_CPP_API_URL", "http://127.0.0.1:9");
    std::env::remove_var("NDFORGE_LLAMA_CPP_WEB_URL");
    let c = Config::from_env();
    assert!(c.allow_local && c.api_base == "http://127.0.0.1:9" && c.web_base == "https://github.com");
    for k in ["NDFORGE_LLAMA_CPP_API_URL", "NDFORGE_LLAMA_CPP_TAG"] {
        std::env::remove_var(k);
    }
}

#[test]
fn only_github_hosts_over_https_may_be_contacted() {
    use ndforge_core::envmgr::llamacpp::net::url_allowed;
    let d = tempdir().unwrap();
    let mut c = cfg_in(d.path());
    c.allow_local = false;
    for ok in ["https://github.com/ggml-org/llama.cpp/archive/refs/tags/b1.zip", "https://codeload.github.com/x", "https://objects.githubusercontent.com/x", "https://release-assets.githubusercontent.com/x?y=1", "https://GitHub.com/x"] {
        assert!(url_allowed(&c, ok, false), "{ok}");
    }
    for bad in ["http://github.com/x", "https://github.com.evil.org/x", "https://evil.org/github.com", "https://github.com@evil.org/x", "https://api.github.com/x", "ftp://github.com/x", "https://raw.githubusercontent.com/x", "http://127.0.0.1:5/x", "github.com/x"] {
        assert!(!url_allowed(&c, bad, false), "{bad}");
    }
    assert!(url_allowed(&c, "https://api.github.com/repos/x", true));
    c.allow_local = true;
    assert!(url_allowed(&c, "http://127.0.0.1:5/x", false));
    assert!(!url_allowed(&c, "http://127.0.0.1.evil.org/x", false));
}

#[test]
fn the_proxy_comes_from_the_environment_unless_no_proxy_excludes_the_host() {
    use ndforge_core::envmgr::llamacpp::net::proxy_for;
    let env = |pairs: &'static [(&'static str, &'static str)]| move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string());
    assert_eq!(proxy_for("github.com", &env(&[])), None);
    assert_eq!(proxy_for("github.com", &env(&[("HTTPS_PROXY", "http://proxy.corp:3128")])).as_deref(), Some("http://proxy.corp:3128"));
    assert_eq!(proxy_for("github.com", &env(&[("https_proxy", "http://p:1"), ("HTTP_PROXY", "http://q:2")])).as_deref(), Some("http://p:1"));
    assert_eq!(proxy_for("github.com", &env(&[("ALL_PROXY", "socks5://s:1080")])).as_deref(), Some("socks5://s:1080"));
    assert_eq!(proxy_for("github.com", &env(&[("HTTPS_PROXY", "  ")])), None);
    let no = &[("HTTPS_PROXY", "http://p:1"), ("NO_PROXY", "localhost, .github.com ,10.0.0.1")];
    assert_eq!(proxy_for("api.github.com", &env(no)), None);
    assert_eq!(proxy_for("github.com", &env(no)), None);
    assert_eq!(proxy_for("codeload.example.com", &env(no)).as_deref(), Some("http://p:1"));
    assert_eq!(proxy_for("anything", &env(&[("HTTPS_PROXY", "http://p:1"), ("no_proxy", "*")])), None);
}
