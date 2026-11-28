//! Installing the Python environment with `uv`: create it, install the CUDA build of PyTorch, keep it (a later install must not swap it for the
//! CPU-only build from the normal package index, which is what Windows gets there), install the worker package, and check that PyTorch really has CUDA.
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicBool;

use super::process::{run_lines, Stream};
use super::python_path;
use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct SetupPlan {
    pub env_dir: PathBuf,
    pub uv: PathBuf,
    pub worker_dir: PathBuf,
    /// "cu128" / "cu126": the PyTorch CUDA build (see gpu::cuda_wheel_for)
    pub wheel: String,
    pub python_version: String,
    /// try to install Unsloth (faster, less memory); false = the standard libraries only (Transformers + PEFT + TRL), which install everywhere
    pub unsloth: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Step { index: usize, total: usize, text: String },
    Log(String),
}

const TOTAL: usize = 5;

/// The libraries the plain Transformers + PEFT + TRL fallback needs (what Unsloth would otherwise bring along).
const FALLBACK_LIBS: [&str; 6] = ["transformers>=5.24.0,<6", "trl", "peft", "datasets", "accelerate", "bitsandbytes"];

/// The fast kernels of Qwen3.5's linear-attention layers, each tried on its own on both paths (with and without Unsloth): a failure never fails the install.
/// triton-windows: the Triton they need, which PyTorch does not ship for Windows; flash-linear-attention: the delta-rule kernels (the big speed-up);
/// causal-conv1d: the short convolution (a small part of the time; it only comes as source, so a ready-made package is rarely found).
pub fn kernel_extras() -> Vec<&'static str> {
    let mut v = vec!["flash-linear-attention", "causal-conv1d"];
    if cfg!(windows) {
        v.insert(0, "triton-windows");
    }
    v
}

/// The Triton that belongs to a PyTorch version (triton-windows must match the torch minor: an unpinned install pulls the build made for the NEXT torch). From the research report:
/// torch 2.14.x needs triton ~= 3.8 (`>=3.8,<3.9`), torch 2.15 `>=3.9,<3.10`. None for a torch this table does not know: the install then leaves Triton as it is and says so.
pub fn triton_pin(torch_version: &str) -> Option<&'static str> {
    let mut it = torch_version.split(|c: char| !c.is_ascii_digit()).filter(|x| !x.is_empty());
    match (it.next().and_then(|x| x.parse::<u32>().ok()), it.next().and_then(|x| x.parse::<u32>().ok())) {
        (Some(2), Some(14)) => Some(">=3.8,<3.9"),
        (Some(2), Some(15)) => Some(">=3.9,<3.10"),
        _ => None,
    }
}

/// The Triton a torch wheel asks for, read from the wheel's own requirement lines (`importlib.metadata.requires("torch")`, e.g. `triton==3.6.0; platform_system == "Linux"`):
/// the exact version, e.g. "3.6.0". Lines of other packages (`triton-windows`, `pytorch-triton`) do not count. The requirement is the truth for that very wheel.
pub fn triton_exact_from_requires(lines: &[String]) -> Option<String> {
    for l in lines {
        let Some(rest) = l.trim().strip_prefix("triton") else { continue };
        if rest.starts_with(|c: char| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) {
            continue;
        }
        let rest = rest.trim_start().trim_start_matches('(').trim_start();
        if let Some(v) = rest.strip_prefix("==") {
            let v: String = v.trim_start().chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
            if v.split('.').filter(|x| !x.is_empty()).count() >= 2 {
                return Some(v);
            }
        }
    }
    None
}

/// The first two numbers of a version ("3.9.0" -> (3, 9)).
fn major_minor(v: &str) -> Option<(u32, u32)> {
    let mut it = v.split(|c: char| !c.is_ascii_digit()).filter(|x| !x.is_empty());
    Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
}

fn pin_of_minor((a, b): (u32, u32)) -> String {
    format!(">={a}.{b},<{a}.{}", b + 1)
}

/// Where a Triton pin came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinSource {
    /// the torch wheel's own requirement
    Wheel,
    /// the table of `triton_pin`
    Table,
}

/// The `triton-windows` pin for a torch: from the wheel's requirement lines first, then the table, else None ("unknown"). The pin is `>=3.x,<3.(x+1)`.
pub fn triton_pin_for(torch_version: &str, requires: &[String]) -> Option<(String, PinSource)> {
    if let Some(m) = triton_exact_from_requires(requires).as_deref().and_then(major_minor) {
        return Some((pin_of_minor(m), PinSource::Wheel));
    }
    triton_pin(torch_version).map(|p| (p.to_string(), PinSource::Table))
}

/// The requirement text of one kernel package: triton-windows pinned from the torch version (None = not installed, with the reason in the log), flash-linear-attention `>=0.6`.
/// `requires` are the requirement lines of the installed torch wheel (empty when they could not be read): the table is only the fallback.
pub fn kernel_requirement_with(pkg: &str, torch_version: &str, requires: &[String]) -> std::result::Result<String, String> {
    match pkg {
        "triton-windows" => triton_pin_for(torch_version, requires).map(|(p, _)| format!("triton-windows{p}")).ok_or_else(|| format!("The Triton version for PyTorch {torch_version} is unknown (its wheel names none and it is not in the table of 2.14 and 2.15): triton-windows is left as it is, because an unpinned install can pull the build made for another torch.")),
        "flash-linear-attention" => Ok("flash-linear-attention>=0.6".to_string()),
        other => Ok(other.to_string()),
    }
}

/// `kernel_requirement_with` without the wheel's requirement lines (the table only).
pub fn kernel_requirement(pkg: &str, torch_version: &str) -> std::result::Result<String, String> {
    kernel_requirement_with(pkg, torch_version, &[])
}

/// Does the installed Triton belong to the installed torch?
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TritonStatus {
    pub torch: String,
    /// the installed `triton-windows` (else `triton`) version; None: none installed
    pub triton: Option<String>,
    /// the pin for this torch (`>=3.6,<3.7`), None when unknown
    pub pin: Option<String>,
    pub source: Option<PinSource>,
    /// None: unknown (no pin or no Triton to compare)
    pub matches: Option<bool>,
}

/// Compares the installed Triton with what the torch wheel asks for (pure: the facts come from `read_triton_status`).
pub fn triton_status(torch: &str, triton: Option<&str>, requires: &[String]) -> TritonStatus {
    let pin = triton_pin_for(torch, requires);
    let matches = match (&pin, triton) {
        (Some((p, _)), Some(t)) => {
            let want = p.strip_prefix(">=").and_then(|x| x.split(',').next()).and_then(major_minor);
            want.zip(major_minor(t)).map(|(w, h)| w == h)
        }
        _ => None,
    };
    TritonStatus { torch: torch.to_string(), triton: triton.map(String::from), pin: pin.as_ref().map(|(p, _)| p.clone()), source: pin.map(|(_, s)| s), matches }
}

impl TritonStatus {
    /// "3.x" of the pin (">=3.6,<3.7" -> "3.6").
    pub fn wants(&self) -> Option<String> {
        self.pin.as_deref().and_then(|p| p.strip_prefix(">=")).and_then(|x| x.split(',').next()).map(String::from)
    }

    /// One plain line for the post-install check, the report and the cards.
    pub fn line(&self) -> String {
        let torch = self.torch.split('+').next().unwrap_or(&self.torch);
        let wants = self.wants().map(|w| format!("wants {w}.x")).unwrap_or_else(|| "its Triton version is unknown".into());
        match (&self.triton, self.matches) {
            (Some(t), Some(true)) => format!("PyTorch {torch}, Triton {t}: matches ({wants})."),
            (Some(t), Some(false)) => format!("Triton {t} does not match torch {torch} ({wants}): press Install fast kernels to fix."),
            (Some(t), None) => format!("PyTorch {torch}, Triton {t}: whether they match is unknown ({wants})."),
            (None, _) => format!("PyTorch {torch}: no Triton is installed ({wants})."),
        }
    }
}

/// Prints one JSON line: the torch version, the installed triton-windows / triton version and the requirement lines of the torch wheel. Imports no torch (only the metadata).
pub const TRITON_FACTS: &str = "import json, importlib.metadata as m\nr = {'torch': None, 'triton': None, 'requires': []}\ntry:\n r['torch'] = m.version('torch')\n r['requires'] = [x for x in (m.requires('torch') or []) if x.lower().startswith('triton')]\nexcept Exception:\n pass\nfor n in ('triton-windows', 'triton'):\n try:\n  r['triton'] = m.version(n)\n  break\n except Exception:\n  pass\nprint(json.dumps(r))\n";

/// Asks the environment's Python (metadata only, about a second) for `TritonStatus`; None when there is no torch or the answer cannot be read.
pub fn read_triton_status(py: &str) -> Option<TritonStatus> {
    let mut cmd = Command::new(py);
    cmd.args(["-c", TRITON_FACTS]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let o = cmd.output().ok().filter(|o| o.status.success())?;
    parse_triton_facts(&String::from_utf8_lossy(&o.stdout))
}

/// Reads the line of `TRITON_FACTS`.
pub fn parse_triton_facts(out: &str) -> Option<TritonStatus> {
    let v: serde_json::Value = serde_json::from_str(out.lines().rev().find(|l| l.trim_start().starts_with('{'))?).ok()?;
    let torch = v["torch"].as_str()?;
    let requires: Vec<String> = v["requires"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
    Some(triton_status(torch, v["triton"].as_str(), &requires))
}

/// The `uv` arguments that install one kernel package: ready-made packages only (never a build from source), into this Python. The pins are added by the caller.
pub fn kernel_extra_args(pkg: &str, py: &str) -> Vec<String> {
    ["pip", "install", "--python", py, "--only-binary=:all:", pkg].iter().map(|s| s.to_string()).collect()
}

/// The check after the fast kernels are installed: `fla.ops.gated_delta_rule` imports AND a tiny forward + backward runs on the graphics card.
pub const KERNEL_CHECK: &str = "import sys\ntry:\n import torch\n from fla.ops.gated_delta_rule import chunk_gated_delta_rule as f\nexcept Exception as e:\n print('IMPORT', type(e).__name__, str(e)[:200]); sys.exit(4)\ntry:\n \
    assert torch.cuda.is_available(), 'no graphics card'\n b, t, h, d = 1, 128, 4, 64\n mk = lambda *s: torch.randn(*s, device='cuda', dtype=torch.bfloat16, requires_grad=True)\n q, k, v = mk(b, t, h, d), mk(b, t, h, d), mk(b, t, h, d)\n \
    g = (-torch.rand(b, t, h, device='cuda')).requires_grad_(True); beta = torch.rand(b, t, h, device='cuda', dtype=torch.bfloat16).requires_grad_(True)\n o = f(q, k, v, g, beta)\n o = o[0] if isinstance(o, tuple) else o\n \
    o.float().sum().backward(); torch.cuda.synchronize(); print('OK')\nexcept Exception as e:\n print('RUN', type(e).__name__, str(e)[:200]); sys.exit(5)\n";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelCheck {
    pub ok: bool,
    pub text: String,
}

/// Reads the answer of `KERNEL_CHECK` (exit code and output) as a plain sentence. Nothing is rolled back: a failed check only says what happens now.
pub fn parse_kernel_check(code: Option<i32>, out: &str) -> KernelCheck {
    let line = out.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string();
    match code {
        Some(0) if line.contains("OK") => KernelCheck { ok: true, text: "The fast kernels work: flash-linear-attention imported and a small forward and backward ran on the graphics card.".into() },
        Some(4) => KernelCheck { ok: false, text: format!("flash-linear-attention is installed but does not import ({line}). The heal keeps using the slow PyTorch kernels. Nothing was removed or changed back; the package can be removed with `uv pip uninstall` if it gets in the way.") },
        Some(5) => KernelCheck { ok: false, text: format!("flash-linear-attention imports but its kernel failed on this graphics card ({line}). The heal keeps using the slow PyTorch kernels unless the import works at run time. Nothing was removed or changed back.") },
        _ => KernelCheck { ok: false, text: format!("The check of the fast kernels could not be run ({line}). The environment is as the install left it.") },
    }
}

/// Runs `KERNEL_CHECK` in the environment's Python.
pub fn run_kernel_check(py: &str, on_event: &mut dyn FnMut(Event)) -> KernelCheck {
    let mut cmd = Command::new(py);
    cmd.args(["-c", KERNEL_CHECK]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let r = match cmd.output() {
        Ok(o) => parse_kernel_check(o.status.code(), &format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))),
        Err(e) => parse_kernel_check(None, &e.to_string()),
    };
    on_event(Event::Log(r.text.clone()));
    r
}

/// The PyTorch version of the environment ("2.14.1+cu130"), None when it cannot be read.
pub fn torch_version(py: &str) -> Option<String> {
    let mut cmd = Command::new(py);
    cmd.args(["-c", "import torch; print(torch.__version__)"]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let o = cmd.output().ok().filter(|o| o.status.success())?;
    Some(String::from_utf8_lossy(&o.stdout).trim().to_string()).filter(|v| !v.is_empty())
}

/// The torch wheel's own `triton...` requirement lines (empty when they cannot be read).
fn read_triton_requires(py: &str) -> Vec<String> {
    let mut cmd = Command::new(py);
    cmd.args(["-c", "import importlib.metadata as m\nfor x in (m.requires('torch') or []):\n if x.lower().startswith('triton'): print(x)"]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    cmd.output().ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()).unwrap_or_default()
}

/// What the log says when a kernel package could not be installed.
fn kernel_failure_note(pkg: &str) -> String {
    match pkg {
        "causal-conv1d" => "causal-conv1d has no ready-made package for this system; the short convolution uses the PyTorch version (a small part of the time).".into(),
        "flash-linear-attention" => "flash-linear-attention could not be installed: Qwen3.5's linear-attention layers will be several times slower when measuring, healing and checking.".into(),
        other => format!("{other} could not be installed; flash-linear-attention may not work without it."),
    }
}

fn step(on_event: &mut dyn FnMut(Event), index: usize, text: &str) {
    on_event(Event::Step { index, total: TOTAL, text: text.to_string() });
}

fn run_uv(plan: &SetupPlan, args: &[String], cancel: &AtomicBool, on_event: &mut dyn FnMut(Event)) -> Result<std::process::ExitStatus> {
    let mut cmd = Command::new(&plan.uv);
    cmd.args(args);
    run_lines(cmd, cancel, &mut |_: Stream, line: &str| on_event(Event::Log(line.to_string())))
}

fn fail(text: &str, code: Option<i32>) -> Error {
    Error::new(format!("The step \"{text}\" failed (exit code {}). The log above shows why; fix that and press the button again.", code.unwrap_or(-1)))
}

pub fn run_setup(plan: &SetupPlan, cancel: &AtomicBool, on_event: &mut dyn FnMut(Event)) -> Result<()> {
    let py = python_path(&plan.env_dir).to_string_lossy().to_string();
    let a = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
    std::fs::create_dir_all(plan.env_dir.parent().unwrap_or(&plan.env_dir))?;

    let t1 = "Creating the Python environment";
    step(on_event, 1, t1);
    let st = run_uv(plan, &a(&["venv", "--python", &plan.python_version, "--seed", &plan.env_dir.to_string_lossy()]), cancel, on_event)?;
    if !st.success() {
        return Err(fail(t1, st.code()));
    }

    let index = format!("https://download.pytorch.org/whl/{}", plan.wheel);
    let t2 = format!("Installing PyTorch for NVIDIA ({}) - several GB, please wait", plan.wheel);
    step(on_event, 2, &t2);
    let torch_libs = ["torch", "torchvision", "torchaudio"];
    let mut args = a(&["pip", "install", "--python", &py]);
    args.extend(torch_libs.iter().map(|s| s.to_string()));
    args.extend(a(&["--index-url", &index]));
    let st = run_uv(plan, &args, cancel, on_event)?;
    if !st.success() {
        return Err(fail(&t2, st.code()));
    }

    // Everything installed after this may use the normal package index too, but must keep exactly these builds: Unsloth and its friends ask for
    // torchvision and others whose own requirements would otherwise pull in a CPU-only torch from PyPI.
    let constraints = plan.env_dir.join("constraints.txt");
    let pinned = pin_torch(plan, &py, &torch_libs, &constraints);
    // the upper bounds the CI's worker tests proved (worker/constraints.txt); Unsloth is not pinned
    let tested = plan.worker_dir.join("constraints.txt");
    let with_pins = |mut v: Vec<String>| -> Vec<String> {
        if pinned {
            v.extend(a(&["--constraint", &constraints.to_string_lossy()]));
        }
        if tested.is_file() {
            v.extend(a(&["--constraint", &tested.to_string_lossy()]));
        }
        v.extend(a(&["--extra-index-url", &index, "--index-strategy", "unsafe-best-match"]));
        v
    };

    // (the kernel packages are installed after step 3, see `install_kernels`)
    // Unsloth (fast, small memory) is preferred when asked for; if it cannot be installed (it is picky about versions and about Windows) or was not wanted,
    // the standard libraries are installed instead and training still works, only slower and with more memory.
    let t3 = if plan.unsloth { "Installing Unsloth and the training libraries" } else { "Installing the training libraries (without Unsloth)" };
    step(on_event, 3, t3);
    let unsloth = if plan.unsloth {
        let st = run_uv(plan, &with_pins(a(&["pip", "install", "--python", &py, "unsloth"])), cancel, on_event)?;
        if !st.success() {
            on_event(Event::Log("Unsloth could not be installed. Installing the standard Transformers + PEFT + TRL libraries instead (slower, needs more graphics memory).".into()));
        }
        st.success()
    } else {
        on_event(Event::Log("Unsloth is skipped, as you chose. Installing the standard Transformers + PEFT + TRL libraries (slower than Unsloth, needs more graphics memory).".into()));
        false
    };
    let mut extras: Vec<String> = Vec::new();
    if !unsloth {
        let mut args = a(&["pip", "install", "--python", &py]);
        args.extend(FALLBACK_LIBS.iter().map(|s| s.to_string()));
        let st = run_uv(plan, &with_pins(args), cancel, on_event)?;
        if !st.success() {
            return Err(fail(t3, st.code()));
        }
    }
    extras.extend(install_kernels(plan, &py, &with_pins, cancel, on_event)?);

    let t4 = "Installing the ND Lore Forge worker";
    step(on_event, 4, t4);
    let st = run_uv(plan, &with_pins(a(&["pip", "install", "--python", &py, &plan.worker_dir.to_string_lossy()])), cancel, on_event)?;
    if !st.success() {
        return Err(fail(t4, st.code()));
    }

    let t5 = "Checking that PyTorch can use the graphics card";
    step(on_event, 5, t5);
    let mut verified = torch_has_cuda(&py, on_event);
    if verified == Some(false) {
        on_event(Event::Log("PyTorch in the environment is the CPU-only build (a package replaced it). Putting the NVIDIA build back.".into()));
        let mut args = a(&["pip", "install", "--python", &py, "--reinstall"]);
        args.extend(torch_libs.iter().map(|s| s.to_string()));
        args.extend(a(&["--index-url", &index]));
        let st = run_uv(plan, &args, cancel, on_event)?;
        if !st.success() {
            return Err(fail(t5, st.code()));
        }
        verified = torch_has_cuda(&py, on_event);
        if verified == Some(false) {
            return Err(Error::new(format!(
                "PyTorch in the training environment is still the CPU-only build after reinstalling it from {index}. Check that this computer can reach download.pytorch.org (a proxy or a firewall may block it), then press the install button again.")));
        }
    }

    std::fs::create_dir_all(&plan.env_dir)?;
    let extras_json = extras.iter().map(|e| format!("\"{e}\"")).collect::<Vec<_>>().join(", ");
    let marker = format!("{{\"wheel\": \"{}\", \"python\": \"{}\", \"unsloth\": {}, \"engine\": \"{}\", \"extras\": [{}], \"cuda_torch\": {}, \"requirements\": {}}}\n",
                         plan.wheel, plan.python_version, unsloth, if unsloth { "unsloth" } else { "standard" }, extras_json, verified.unwrap_or(false), super::health::ENV_REQUIREMENTS_VERSION);
    std::fs::write(plan.env_dir.join("ndforge_env.json"), marker)?;
    let _ = std::fs::remove_file(plan.env_dir.join("selftest.json")); // a new environment has to pass the test again
    Ok(())
}

/// Tries each kernel package (see `kernel_extras`), with the same pins as everything else so that PyTorch is never swapped. Returns the ones that were installed;
/// a failure is only logged in plain words. An error is returned only when the install was cancelled.
fn install_kernels(plan: &SetupPlan, py: &str, with_pins: &dyn Fn(Vec<String>) -> Vec<String>, cancel: &AtomicBool, on_event: &mut dyn FnMut(Event)) -> Result<Vec<String>> {
    let mut done = Vec::new();
    let tv = torch_version(py).unwrap_or_default();
    let requires = read_triton_requires(py);
    for pkg in kernel_extras() {
        let req = match kernel_requirement_with(pkg, &tv, &requires) {
            Ok(r) => r,
            Err(why) => {
                on_event(Event::Log(why));          // an unknown torch: nothing is guessed
                continue;
            }
        };
        let st = run_uv(plan, &with_pins(kernel_extra_args(&req, py)), cancel, on_event)?;
        if st.success() {
            done.push(pkg.to_string());
        } else {
            on_event(Event::Log(format!("{} The environment is as it was before this package.", kernel_failure_note(pkg))));
        }
    }
    if done.iter().any(|d| d == "flash-linear-attention") {
        run_kernel_check(py, on_event);          // says plainly whether the kernels work; nothing is rolled back
    }
    if let Some(t) = read_triton_status(py) {
        on_event(Event::Log(t.line()));          // torch, triton-windows and whether they match, after the install too
    }
    Ok(done)
}

/// The "Install the fast kernels" button: only the kernel packages, into the environment that exists. Updates `extras` in the marker. Returns what was installed.
pub fn run_kernels_only(plan: &SetupPlan, cancel: &AtomicBool, on_event: &mut dyn FnMut(Event)) -> Result<Vec<String>> {
    let py = python_path(&plan.env_dir).to_string_lossy().to_string();
    if !std::path::Path::new(&py).is_file() || !plan.env_dir.join("ndforge_env.json").is_file() {
        return Err(Error::new("The training software is not installed yet. Press \"Install the training software\" first."));
    }
    let index = format!("https://download.pytorch.org/whl/{}", plan.wheel);
    let constraints = plan.env_dir.join("constraints.txt");
    let tested = plan.worker_dir.join("constraints.txt");
    let with_pins = |mut v: Vec<String>| -> Vec<String> {
        for c in [&constraints, &tested] {
            if c.is_file() {
                v.extend(["--constraint".to_string(), c.to_string_lossy().to_string()]);
            }
        }
        v.extend(["--extra-index-url".to_string(), index.clone(), "--index-strategy".to_string(), "unsafe-best-match".to_string()]);
        v
    };
    let done = install_kernels(plan, &py, &with_pins, cancel, on_event)?;
    // the marker keeps what was installed (the other keys stay as they are)
    let marker = plan.env_dir.join("ndforge_env.json");
    if let Ok(mut v) = std::fs::read(&marker).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).map_err(|e| e.to_string())) {
        let mut all: Vec<String> = v["extras"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
        for d in &done {
            if !all.contains(d) {
                all.push(d.clone());
            }
        }
        v["extras"] = serde_json::json!(all);
        if let Ok(t) = serde_json::to_string_pretty(&v) {      // pretty: the checks that read the marker look for `"key": value` with the space
            let _ = std::fs::write(&marker, t + "\n");
        }
    }
    on_event(Event::Log(if done.is_empty() { "No fast kernel could be installed.".to_string() } else { format!("Installed: {}.", done.join(", ")) }));
    Ok(done)
}

/// Writes `constraints.txt` with the exact versions of the torch libraries that are installed now; false when they could not be read.
fn pin_torch(plan: &SetupPlan, py: &str, libs: &[&str], file: &std::path::Path) -> bool {
    let Ok(out) = Command::new(&plan.uv).args(["pip", "freeze", "--python", py]).output() else { return false };
    if !out.status.success() {
        return false;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let pins: Vec<&str> = text.lines().map(str::trim).filter(|l| libs.iter().any(|n| l.starts_with(&format!("{n}==")))).collect();
    if pins.is_empty() || std::fs::write(file, pins.join("\n") + "\n").is_err() {
        return false;
    }
    true
}

/// Does the environment's PyTorch have CUDA support (a build, not whether a card is present)? None when it could not be asked.
fn torch_has_cuda(py: &str, on_event: &mut dyn FnMut(Event)) -> Option<bool> {
    let script = "import sys, torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda); sys.exit(0 if torch.version.cuda else 3)";
    let mut cmd = Command::new(py);
    cmd.args(["-c", script]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // no console window
    }
    let out = cmd.output().ok()?;
    for l in String::from_utf8_lossy(&out.stdout).lines().chain(String::from_utf8_lossy(&out.stderr).lines().take(3)) {
        on_event(Event::Log(l.to_string()));
    }
    match out.status.code() {
        Some(0) => Some(true),
        Some(3) => Some(false),
        _ => None,
    }
}
