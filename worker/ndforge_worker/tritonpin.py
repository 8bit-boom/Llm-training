"""Which Triton a torch wheel asks for, and whether the installed one is it. The torch wheel's own requirement (`importlib.metadata.requires("torch")`, e.g.
`triton==3.6.0; platform_system == "Linux"`) is the truth for that wheel; `triton-windows` follows the same minor. The Rust side (`envmgr/setup.rs`) derives the same pin to install."""
import re

# the table of the Rust side, the fallback when the wheel names none
TABLE = {(2, 14): "3.8", (2, 15): "3.9"}
_REQ = re.compile(r"^triton(?![\w.\-])\s*\(?\s*==\s*([0-9]+(?:\.[0-9]+)+)")


def _minor(v):
    m = re.match(r"\s*(\d+)\.(\d+)", v or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def wheel_triton(requires):
    """The exact Triton the torch wheel asks for ("3.6.0"), or None."""
    for line in requires or []:
        m = _REQ.match(str(line).strip())
        if m:
            return m.group(1)
    return None


def pin_for(torch_version, requires):
    """(wanted minor "3.6", source "wheel" | "table") or (None, None) when unknown."""
    exact = _minor(wheel_triton(requires))
    if exact:
        return f"{exact[0]}.{exact[1]}", "wheel"
    t = _minor(torch_version)
    if t in TABLE:
        return TABLE[t], "table"
    return None, None


def status(torch_version, triton_version, requires) -> dict:
    """{torch, triton, wants, source, matches (True / False / None), text}: the line the report and the cards print."""
    wants, source = pin_for(torch_version, requires)
    have = _minor(triton_version)
    matches = None if not wants or not have else (f"{have[0]}.{have[1]}" == wants)
    torch = (torch_version or "unknown").split("+")[0]
    w = f"wants {wants}.x" if wants else "its Triton version is unknown"
    if not triton_version:
        text = f"PyTorch {torch}: no Triton is installed ({w})."
    elif matches is True:
        text = f"PyTorch {torch}, Triton {triton_version}: matches ({w})."
    elif matches is False:
        text = f"Triton {triton_version} does not match torch {torch} ({w}): press Install fast kernels to fix."
    else:
        text = f"PyTorch {torch}, Triton {triton_version}: whether they match is unknown ({w})."
    return {"torch": torch_version, "triton": triton_version, "wants": wants, "source": source, "matches": matches, "text": text}


def installed_status() -> dict:
    """`status` of this environment (metadata only; never raises)."""
    import importlib.metadata as md
    try:
        tv = md.version("torch")
        reqs = [r for r in (md.requires("torch") or []) if r.lower().startswith("triton")]
    except Exception:       # noqa: BLE001
        return status(None, None, [])
    triton = None
    for n in ("triton-windows", "triton"):
        try:
            triton = md.version(n)
            break
        except Exception:       # noqa: BLE001
            continue
    return status(tv, triton, reqs)
