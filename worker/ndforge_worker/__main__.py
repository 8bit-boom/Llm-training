"""Entry point: python -m ndforge_worker <job.json>. The job file is JSON with a "job" key; see protocol.py for the output."""
import json
import sys

from . import __version__, protocol


def _code(exc: Exception, default: str) -> str:
    """The error code of exceptions that carry one (llama_cpp_missing: the page then offers to install llama.cpp), else `default`."""
    return getattr(exc, "code", None) or default


# the libraries a problem report wants the versions of; read from the installed metadata, never imported (importing torch takes seconds and needs the card)
PACKAGES = ("torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "unsloth", "datasets")


def _package_versions() -> dict:
    from importlib import metadata
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


# the fast-kernel packages for Qwen3.5's linear-attention layers (`speed.gdn_kernels` says what a loaded model really uses); versions from the metadata, nothing imported
ACCELERATORS = ("flash-linear-attention", "causal-conv1d", "triton", "triton-windows")


def _accelerators() -> dict:
    from importlib import metadata
    out = {}
    for name in ACCELERATORS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _explain(exc: Exception) -> str:
    if isinstance(getattr(exc, "tail", None), list):     # a llama.cpp tool failed: its message already holds the last output lines (scrubbed); keep them whole, user name out
        return protocol.redact_user(protocol.scrub(str(exc)))[:4000]
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if "no space left" in low:
        return "The disk is full. Free some space (or choose another folder) and try again. (" + text[:200] + ")"
    if "out of memory" in low:
        return "The graphics card ran out of memory. Use a smaller sequence length, LoRA 4-bit (QLoRA), or a smaller model. (" + text[:200] + ")"
    if "no module named 'numpy'" in low:
        return "The training software is incomplete (numpy). Run Training setup again."
    if "no module named" in low:
        return "A training library is missing in the training software. Run the training setup again. (" + text[:200] + ")"
    if "401" in low or "gated" in low or "repository not found" in low:
        return "The model could not be downloaded from Hugging Face. Check the internet connection (and the model name). (" + text[:200] + ")"
    return text[:600]


def _explain_heal(kind: str, exc: Exception) -> str:
    """Healing jobs: an out-of-memory error that was not handled by the retries (the cache step, or while loading a model) names the step and the next actions."""
    from . import healmem
    text = f"{type(exc).__name__}: {exc}"
    if kind == "distill_cache" and healmem.is_oom(exc):
        return ("The graphics card ran out of memory during step cache (the original model answering your texts). What to change next, in this order: 1. load the original "
                "as 4-bit (Healing settings, Expert: original loaded as 4-bit); 2. a shorter sequence length; 3. a smaller model. Continue goes on where it stopped. (" + text[:200] + ")")
    if kind == "distill" and healmem.is_oom(exc):
        return ("The graphics card ran out of memory during step train, while the models were being loaded, before any training step. What to change next, in this order: "
                "1. QLoRA (4-bit) instead of LoRA 16-bit; 2. load the original as 4-bit; 3. a smaller model. (" + text[:200] + ")")
    return _explain(exc)


def main(argv: list) -> int:
    if len(argv) != 2:
        protocol.error("usage", "Usage: python -m ndforge_worker <job.json>")
        return 1
    try:
        with open(argv[1], encoding="utf-8") as fh:
            job = json.load(fh)
    except (OSError, ValueError) as exc:
        protocol.error("bad_job", f"The job file could not be read: {exc}")
        return 1
    kind = job.get("job")
    protocol.start_job()
    from . import power
    with power.keep_awake(kind, job):         # long jobs hold the computer awake; the short ones and the other systems are left alone
        return _run(kind, job)


def _run(kind, job: dict) -> int:
    if kind == "hello":
        protocol.result({"worker": __version__, "protocol": protocol.PROTOCOL_VERSION, "packages": _package_versions(), "accelerators": _accelerators()})
        return 0
    if kind == "selftest":
        from . import selftest
        return selftest.run(job)
    if kind == "train":
        from . import trainjob
        try:
            return trainjob.run(job)
        except Exception as exc:  # turn any failure into a plain error for the window (the traceback goes to the log)
            import traceback
            traceback.print_exc()
            protocol.error("train_failed", _explain(exc))
            return 1
    if kind == "export":
        from . import exportjob, imatrix
        try:
            return exportjob.run(job)
        except imatrix.Stopped:   # the Stop button during the importance matrix: the same word as a killed worker
            protocol.error("export_stopped", "Cancelled.")
            return 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            protocol.error(_code(exc, "export_failed"), str(exc) if _code(exc, "") else _explain(exc))
            return 1
    if kind in ("imatrix_make", "imatrix_dat", "imatrix_merge", "imatrix_published", "imatrix_adapt", "imatrix_adapt_preview"):
        from . import imatrix, imatrixadapt, imatrixjob, imatrixmerge
        try:
            return {"imatrix_make": imatrixjob.run, "imatrix_dat": imatrixjob.run_dat, "imatrix_merge": imatrixmerge.run, "imatrix_published": imatrixmerge.run_published,
                    "imatrix_adapt": imatrixadapt.run, "imatrix_adapt_preview": imatrixadapt.run_preview}[kind](job)
        except imatrix.Stopped:   # the Stop button (the STOP file): the same word as a killed worker
            protocol.error("imatrix_stopped", "Cancelled.")
            return 1
        except imatrixjob.MatrixError as exc:   # a plain-language failure: no traceback text
            protocol.error(_code(exc, "imatrix_failed"), str(exc))
            return 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            protocol.error("imatrix_failed", imatrixjob.plain(exc))
            return 1
    if kind in ("quantcheck", "quantcheck_plan"):
        from . import imatrix, quantcheck
        try:
            return {"quantcheck": quantcheck.run, "quantcheck_plan": quantcheck.run_plan}[kind](job)
        except imatrix.Stopped:   # the Stop button (the STOP file): the same word as a killed worker
            protocol.error("quantcheck_stopped", "Cancelled.")
            return 1
        except quantcheck.QuantCheckError as exc:   # a plain-language failure: no traceback text
            protocol.error(_code(exc, "quantcheck_failed"), str(exc))
            return 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            protocol.error("quantcheck_failed", _explain(exc))
            return 1
    if kind in ("quantfit", "quantfit_estimate"):
        from . import imatrix, quantfit, quantmeasure
        try:
            return {"quantfit": quantfit.run, "quantfit_estimate": quantfit.run_estimate}[kind](job)
        except (imatrix.Stopped, quantmeasure.Stopped):   # the Stop button (the STOP file): the same word as a killed worker
            protocol.error("quantfit_stopped", "Cancelled.")
            return 1
        except quantfit.QuantFitError as exc:   # a plain-language failure: no traceback text (llama.cpp missing carries its own code)
            protocol.error(_code(exc, "quantfit_failed"), str(exc))
            return 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            protocol.error(_code(exc, "quantfit_failed"), str(exc) if _code(exc, "") else _explain(exc))
            return 1
    if kind == "mixed_plan":
        from . import mixedquant
        try:
            return mixedquant.run_plan(job)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            protocol.error("mixed_failed", str(exc) if isinstance(exc, ValueError) else _explain(exc))
            return 1
    if kind == "evaluate":
        from . import evaluatejob
        try:
            return evaluatejob.run(job)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            protocol.error("evaluate_failed", _explain(exc))
            return 1
    if kind in ("reshape_inspect", "reshape", "reshape_measure", "reshape_try"):
        from . import reshape, reshapejob, reshapetry
        try:
            return {"reshape_inspect": reshapejob.run_inspect, "reshape": reshapejob.run, "reshape_measure": reshapejob.run_measure,
                    "reshape_try": reshapetry.run_try}[kind](job)
        except reshape.ReshapeError as exc:   # a plain-language refusal: no traceback text
            protocol.error("reshape_refused", str(exc))
            return 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            protocol.error({"reshape_inspect": "inspect_failed", "reshape": "reshape_failed", "reshape_measure": "measure_failed",
                                                   "reshape_try": "try_failed"}[kind], _explain(exc))
            return 1
    if kind in ("distill_cache", "distill", "distill_merge"):
        from . import distilljob, reshape
        try:
            return {"distill_cache": distilljob.run_cache, "distill": distilljob.run, "distill_merge": distilljob.merge}[kind](job)
        except reshape.ReshapeError as exc:   # a plain-language refusal: no traceback text
            from . import healmem
            protocol.error("distill_unstable" if str(exc) == distilljob.UNSTABLE else "distill_oom" if isinstance(exc, healmem.HealOutOfMemory) else "reshape_refused", str(exc))
            return 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            protocol.error({"distill_cache": "distill_failed", "distill": "distill_failed", "distill_merge": "merge_failed"}[kind], _explain_heal(kind, exc))
            return 1
    if kind == "heal_check":
        from . import healcheck
        try:
            return healcheck.run_check(job)
        except Exception as exc:
            protocol.error("heal_check_failed", _explain(exc))
            return 1
    if kind in ("think_check", "think_check_plan"):
        from . import thinkcheck
        try:
            return {"think_check": thinkcheck.run, "think_check_plan": thinkcheck.run_plan}[kind](job)
        except thinkcheck.ThinkCheckError as exc:   # a plain-language failure: no traceback text
            protocol.error(exc.code, str(exc))
            return 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            protocol.error(_code(exc, "think_check_failed"), str(exc) if _code(exc, "") else _explain(exc))   # llama_cpp_missing: the page offers the installer
            return 1
    if kind == "trace_gen":
        from . import thinkcheck, tracegen
        try:
            return tracegen.run(job)
        except thinkcheck.ThinkCheckError as exc:   # a plain-language failure: no traceback text
            protocol.error("trace_gen_failed", str(exc))
            return 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            protocol.error("trace_gen_failed", _explain(exc))
            return 1
    if kind == "vllm_container_probe":
        from . import vllmcache
        try:
            r = vllmcache.probe(job)
            protocol.result(r)
            return 0 if r["ok"] else 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            protocol.error("vllm_container_probe_failed", _explain(exc))
            return 1
    if kind == "vllm_selftest":
        from . import thinkcheck, vllmselftest
        try:
            return vllmselftest.run(job)
        except thinkcheck.ThinkCheckError as exc:   # a plain-language refusal: no traceback text
            protocol.error("vllm_selftest_failed", str(exc))
            return 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            protocol.error("vllm_selftest_failed", _explain(exc))
            return 1
    if kind == "vllm_teacher_check":
        from . import vllmteachercheck
        try:
            return vllmteachercheck.run(job)
        except vllmteachercheck.TeacherCheckError as exc:   # a plain-language failure: no traceback text
            protocol.error("vllm_teacher_check_failed", str(exc))
            return 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            protocol.error("vllm_teacher_check_failed", _explain(exc))
            return 1
    if kind == "general_check":
        from . import generaltext
        return generaltext.run_check(job)
    protocol.error("unknown_job", f"Unknown job type: {kind!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
