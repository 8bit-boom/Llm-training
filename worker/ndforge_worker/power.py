"""Keeps the computer from going to sleep while a long job runs (Windows only; elsewhere it does nothing).

`SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` tells Windows the system is in use. It is called on the main thread (the flag lives as long as the thread does or until it is reset)
and reset when the job ends. The display may still switch off; `NDFORGE_KEEP_DISPLAY=1` also keeps it on. Away mode is not used.
This is a best effort: whether Windows honours it on a computer with Modern Standby is not guaranteed (see docs/INSTALL.md), and every failure is swallowed and logged once.
Off when the environment has `NDFORGE_KEEP_AWAKE=0` (the program sets it when the setting is off) or the job has `"keep_awake": false`.
The power plan is never changed."""
import contextlib
import os
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

# jobs that finish in seconds: no reason to hold the computer awake
SHORT_JOBS = frozenset({"hello", "reshape_inspect", "general_check", "imatrix_published", "imatrix_adapt_preview", "heal_check", "selftest"})

_warned = [False]


def is_short(kind) -> bool:
    k = str(kind or "")
    return k in SHORT_JOBS or k.endswith("_plan")


def wanted(job: dict | None = None) -> bool:
    if os.environ.get("NDFORGE_KEEP_AWAKE") == "0":
        return False
    return not (isinstance(job, dict) and job.get("keep_awake") is False)


def _set(flags: int):
    import ctypes
    return ctypes.windll.kernel32.SetThreadExecutionState(flags & 0xFFFFFFFF)


def _note(message: str) -> None:
    if not _warned[0]:
        _warned[0] = True
        from . import protocol
        protocol.log(message)


@contextlib.contextmanager
def keep_awake(job_kind=None, job: dict | None = None):
    """Holds the system awake for the body of the `with` block."""
    active = False
    if sys.platform == "win32" and not is_short(job_kind) and wanted(job):
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        if os.environ.get("NDFORGE_KEEP_DISPLAY") == "1":
            flags |= ES_DISPLAY_REQUIRED
        try:
            active = bool(_set(flags))
            if active:
                from . import protocol
                protocol.log("Keeping this computer awake while the job runs.")
            else:
                _note("Windows did not accept the request to keep this computer awake; it may go to sleep during a long job.")
        except Exception as exc:     # noqa: BLE001 - a broken power API must never stop a job
            _note(f"Could not ask Windows to keep this computer awake ({type(exc).__name__}); it may go to sleep during a long job.")
    try:
        yield active
    finally:
        if active:
            try:
                _set(ES_CONTINUOUS)
            except Exception:        # noqa: BLE001
                pass
