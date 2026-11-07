"""GPU self-test: is PyTorch installed, does it see the GPU, can the GPU do maths, and can it train a tiny model?

Never imports torch at module load, so the worker still answers (with a plain error) on a machine where it is not installed yet."""
import platform
import time

from . import protocol


def _info_without_gpu(torch) -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda_available": False,
            "cuda_version": getattr(torch.version, "cuda", None)}


def run(job: dict) -> int:
    try:
        import torch
    except ImportError:
        protocol.error("torch_missing", "PyTorch is not installed in the training environment yet. Run the training setup first.")
        return 1
    info = _info_without_gpu(torch)
    protocol.progress("detect", 0.1, "Looking for the graphics card")
    if not torch.cuda.is_available():
        if job.get("require_gpu", True):
            protocol.error("no_cuda", "PyTorch cannot see an NVIDIA graphics card. Update the NVIDIA driver and run the setup again.")
            return 1
        protocol.result({**info, "train_ok": False, "note": "CPU only"})
        return 0
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    info.update({"cuda_available": True, "device_name": props.name, "compute_capability": f"{major}.{minor}",
                 "vram_mb": props.total_memory // (1024 * 1024), "bf16": bool(torch.cuda.is_bf16_supported()),
                 "arch_list": torch.cuda.get_arch_list()})
    # a card whose architecture this PyTorch build has no kernels for fails in a confusing way later, so say it here
    sm = f"sm_{major}{minor}"
    if not any(a.startswith(sm) or a.replace("sm_", "compute_") == sm.replace("sm_", "compute_") for a in info["arch_list"]) \
            and not any(a.startswith("compute_") for a in info["arch_list"]):
        protocol.error("arch_unsupported", f"This PyTorch build does not support the graphics card's architecture ({sm}). "
                       f"A newer CUDA build is needed (RTX 50 cards need CUDA 12.8 or newer).")
        return 1
    protocol.progress("matmul", 0.3, "Testing the graphics card")
    dev = torch.device("cuda:0")
    dtype = torch.bfloat16 if info["bf16"] else torch.float16
    try:
        a = torch.randn(1024, 1024, device=dev, dtype=dtype)
        b = torch.randn(1024, 1024, device=dev, dtype=dtype)
        c = a @ b
        torch.cuda.synchronize()
        info["matmul_ok"] = bool(torch.isfinite(c).all().item())
    except Exception as exc:  # a CUDA error here is the finding, not a crash
        protocol.error("matmul_failed", f"The graphics card could not do a simple calculation: {exc}")
        return 1
    protocol.progress("train", 0.5, "Training a tiny model")
    steps = int(job.get("steps", 200))
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)).to(dev)
    target_w = torch.randn(64, 1, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = last = None
    t0 = time.time()
    for i in range(steps):
        x = torch.randn(256, 64, device=dev)
        y = x @ target_w
        with torch.autocast("cuda", dtype=dtype):
            loss = torch.nn.functional.mse_loss(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss.item())
        if i == steps - 1:
            last = float(loss.item())
        if i % 20 == 0:
            protocol.progress("train", 0.5 + 0.45 * i / steps, f"step {i}/{steps}")
    torch.cuda.synchronize()
    info.update({"train_ok": bool(last is not None and last < first * 0.5), "loss_first": first, "loss_last": last,
                 "train_seconds": round(time.time() - t0, 2), "peak_vram_mb": torch.cuda.max_memory_allocated() // (1024 * 1024),
                 "precision": "bf16" if dtype == torch.bfloat16 else "fp16"})
    if not info["train_ok"]:
        protocol.error("train_failed", "The tiny training test did not learn anything. Something is wrong with the graphics card setup.")
        return 1
    protocol.progress("done", 1.0, "GPU ready")
    protocol.result(info)
    return 0
