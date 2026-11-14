"""The worker's output protocol: one JSON object per line on stdout.

    {"type": "log", "text": "..."}                         free text for the log view
    {"type": "progress", "step": "name", "value": 0.0-1.0, "text": "..."}
    {"type": "result", "data": {...}}                      the job's answer (exactly one, last)
    {"type": "error", "code": "short_code", "message": "plain-language text for the user"}

Exit status: 0 after a result, 1 after an error."""
import json
import os
import re
import sys

# the protocol goes to the REAL stdout; libraries that print (the Trainer's log dicts ...) are moved to stderr by `reserve_stdout`
_out = sys.stdout


def reserve_stdout() -> None:
    """From now on `print` and library output go to stderr (shown in the log); only protocol lines use stdout."""
    sys.stdout = sys.stderr


_HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{20,}")


def scrub(text):
    """Takes Hugging Face tokens (hf_...) and the token in HF_TOKEN out of a text, so that they never reach a log, an error or a problem report."""
    if not isinstance(text, str):
        return text
    text = _HF_TOKEN.sub("[token]", text)
    tok = (os.environ.get("HF_TOKEN") or "").strip()
    return text.replace(tok, "[token]") if len(tok) >= 8 else text


def emit(kind: str, **fields) -> None:
    if kind in ("log", "error"):
        fields = {k: scrub(v) if k in ("text", "message") else v for k, v in fields.items()}
    _out.write(json.dumps({"type": kind, **fields}, ensure_ascii=False) + "\n")
    _out.flush()


def log(text: str) -> None:
    emit("log", text=text)


def progress(step: str, value: float, text: str = "") -> None:
    emit("progress", step=step, value=round(float(value), 4), text=text)


def result(data: dict) -> None:
    emit("result", data=data)


def error(code: str, message: str) -> None:
    emit("error", code=code, message=message)
