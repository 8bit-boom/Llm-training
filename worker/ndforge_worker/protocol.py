"""The worker's output protocol: one JSON object per line on stdout.

    {"type": "log", "text": "..."}                         free text for the log view
    {"type": "progress", "step": "name", "value": 0.0-1.0, "text": "..."}
    {"type": "result", "data": {...}}                      the job's answer (exactly one, last)
    {"type": "error", "code": "short_code", "message": "plain-language text for the user"}

Exit status: 0 after a result, 1 after an error."""
import contextlib
import json
import os
import re
import sys
import time

# bumped when a line of the protocol changes shape (never for a new job, event or key: those are only added). The program reads it from the `hello` job.
PROTOCOL_VERSION = 1

# the protocol goes to the REAL stdout; libraries that print (the Trainer's log dicts ...) are moved to stderr by `reserve_stdout`
_out = sys.stdout


def reserve_stdout() -> None:
    """From now on `print` and library output go to stderr (shown in the log); only protocol lines use stdout."""
    sys.stdout = sys.stderr


_HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{20,}")


_SECRETS: list = []


def add_secret(value: str) -> None:
    """Registers a secret of this run (the vLLM engine's per-run API key): `scrub` takes it out of every log line, error and report from now on."""
    if value and len(value) >= 8 and value not in _SECRETS:
        _SECRETS.append(value)


def scrub(text):
    """Takes Hugging Face tokens (hf_...), the token in HF_TOKEN and registered secrets (`add_secret`) out of a text, so that they never reach a log, an error or a problem report."""
    if not isinstance(text, str):
        return text
    text = _HF_TOKEN.sub("[token]", text)
    for sec in _SECRETS:
        text = text.replace(sec, "[token]")
    tok = (os.environ.get("HF_TOKEN") or "").strip()
    return text.replace(tok, "[token]") if len(tok) >= 8 else text


_USER_DIR = re.compile(r"(?i)([A-Z]:[\\/]+Users[\\/]+)[^\\/\s\"']+|(/home/|/Users/)[^/\s\"']+")


def redact_user(text):
    """Takes the person's home folder and user name out of a text that leaves the program in a report (`C:\\Users\\<name>\\...` -> `C:\\Users\\<user>\\...`, the home folder -> `~`)."""
    if not isinstance(text, str):
        return text
    home = os.path.expanduser("~")
    if len(home) > 3 and home not in ("/", "/root"):
        text = text.replace(home, "~").replace(home.replace("\\", "\\\\"), "~")
    text = _USER_DIR.sub(lambda m: (m.group(1) or m.group(2)) + "<user>", text)
    name = (os.environ.get("USERNAME") or os.environ.get("USER") or "").strip()
    if len(name) >= 3 and name.lower() not in ("root", "user", "admin"):
        text = re.sub(re.escape(name), "<user>", text, flags=re.I)
    return text


def emit(kind: str, **fields) -> None:
    if kind in ("log", "error"):
        fields = {k: scrub(v) if k in ("text", "message") else v for k, v in fields.items()}
    _out.write(json.dumps({"type": kind, **fields}, ensure_ascii=False) + "\n")
    _out.flush()


def log(text: str) -> None:
    emit("log", text=text)


def progress(step: str, value: float, text: str = "", detail: dict | None = None) -> None:
    """A progress line. `detail` (optional, see `make_detail`) is an additive key: without it the line is byte-identical to the older ones."""
    if detail:
        emit("progress", step=step, value=round(float(value), 4), text=text, detail=detail)
    else:
        emit("progress", step=step, value=round(float(value), 4), text=text)


def make_detail(unit: str, done, total, started: float | None = None, now: float | None = None, rate_unit: str | None = None,
                done_at_start=0) -> dict:
    """What a progress line says in numbers: `{unit, done, total, rate, rate_unit, eta_s}`. The rate comes from the work done since `started`
    (a monotonic time; `done_at_start` is what was done before this run began), the estimate of the time left from the rate; both are left out until there is a rate."""
    d: dict = {"unit": unit, "done": round(float(done), 3), "total": round(float(total), 3)}
    if started is not None:
        spent = (now if now is not None else time.monotonic()) - started
        gained = float(done) - float(done_at_start)
        if spent > 0 and gained > 0:
            rate = gained / spent
            d["rate"] = round(rate, 3)
            if rate_unit:
                d["rate_unit"] = rate_unit
            d["eta_s"] = round(max(0.0, float(total) - float(done)) / rate, 1)
    return d


# keys the running job collects for its result (which kernels ran, the timings): `result` adds them to a dict answer unless the answer has the key already
EXTRAS: dict = {}


def add_extra(key: str, value) -> None:
    EXTRAS[key] = value


# where the time went: `timing(stage, seconds, **fields)` logs one `timing: {...}` line (an ordinary log line: no new event type) and keeps the record;
# `result` adds the records to a dict answer as `timings` (new key), with a last record `job` for the whole job.
_TIMINGS: list = []
_START = [None]


def start_job() -> None:
    """Called by the entry point when a job starts: forgets what an earlier job in this process collected."""
    _TIMINGS.clear()
    EXTRAS.clear()
    _START[0] = time.monotonic()


def timing(stage: str, seconds: float, **fields) -> dict:
    rec = {"stage": stage, "seconds": round(float(seconds), 2), **{k: v for k, v in fields.items() if v is not None}}
    _TIMINGS.append(rec)
    log("timing: " + json.dumps(rec, ensure_ascii=False, sort_keys=True))
    return rec


def timings() -> list:
    return list(_TIMINGS)


@contextlib.contextmanager
def timed(stage: str, **fields):
    """Times the body and records it; the body may add fields to the yielded dict (`t["tokens"] = n`)."""
    extra: dict = {}
    t0 = time.monotonic()
    try:
        yield extra
    finally:
        timing(stage, time.monotonic() - t0, **{**fields, **extra})


def result(data: dict) -> None:
    if isinstance(data, dict):
        data = {**data, **{k: v for k, v in EXTRAS.items() if k not in data}}
        if _TIMINGS and "timings" not in data:
            total = round(time.monotonic() - _START[0], 2) if _START[0] is not None else round(sum(t["seconds"] for t in _TIMINGS), 2)
            data["timings"] = _TIMINGS + [{"stage": "job", "seconds": total}]
    emit("result", data=data)


def error(code: str, message: str) -> None:
    emit("error", code=code, message=message)
