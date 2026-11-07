"""Live reporting while training: metrics, sample answers to fixed questions, and a graceful stop."""
import os
import subprocess
import time

from transformers import TrainerCallback

from . import protocol

_temp_cache = {"t": 0.0, "v": None}


def gpu_temperature():
    """The graphics card's temperature (cached for 5 s); None when it cannot be read."""
    if time.time() - _temp_cache["t"] < 5:
        return _temp_cache["v"]
    _temp_cache["t"] = time.time()
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
        _temp_cache["v"] = int(out.stdout.strip().splitlines()[0]) if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        _temp_cache["v"] = None
    return _temp_cache["v"]


class LiveCallback(TrainerCallback):
    def __init__(self, stop_file: str, answer_fn=None, probes=None, probe_every: int = 0):
        self.stop_file, self.answer_fn, self.probes, self.probe_every = stop_file, answer_fn, probes or [], probe_every
        self.t0 = None
        self.start_step = 0
        self.stopped = False
        self.last_loss = None
        self.peak_vram_mb = 0

    # -- helpers
    def _vram(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.peak_vram_mb = max(self.peak_vram_mb, torch.cuda.max_memory_allocated() // (1024 * 1024))
        except Exception:
            pass
        return self.peak_vram_mb

    def _probe(self, step: int):
        if not (self.answer_fn and self.probes):
            return
        for q in self.probes:
            try:
                protocol.emit("sample", step=step, question=q["question"], answer=self.answer_fn(q))
            except Exception as exc:  # a probe must never stop the training
                protocol.log(f"Sample answer skipped: {type(exc).__name__}: {exc}")
                return

    # -- Trainer hooks
    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.time()
        self.start_step = state.global_step
        protocol.emit("start", total_steps=state.max_steps, epochs=args.num_train_epochs, resumed_from=self.start_step)
        self._probe(self.start_step)  # the answers before any learning, to compare with

    def on_log(self, args, state, control, logs=None, **kw):
        logs = logs or {}
        if "loss" not in logs:
            return
        self.last_loss = logs["loss"]
        done = max(1, state.global_step - self.start_step)
        per_step = (time.time() - self.t0) / done if self.t0 else 0
        protocol.emit("metric", step=state.global_step, total_steps=state.max_steps, loss=logs.get("loss"), lr=logs.get("learning_rate"),
                      grad_norm=logs.get("grad_norm"), epoch=logs.get("epoch"), vram_mb=self._vram(), gpu_temp=gpu_temperature(),
                      seconds_per_step=round(per_step, 3), eta_s=int(per_step * max(0, state.max_steps - state.global_step)))
        protocol.progress("train", state.global_step / max(1, state.max_steps), f"Step {state.global_step} of {state.max_steps}")

    def on_step_end(self, args, state, control, **kw):
        if os.path.exists(self.stop_file):
            self.stopped = True
            control.should_save = True
            control.should_training_stop = True
            return
        if self.probe_every and state.global_step % self.probe_every == 0:
            self._probe(state.global_step)

    def on_train_end(self, args, state, control, **kw):
        if not self.stopped:
            self._probe(state.global_step)
