"""The worker's output protocol: one JSON object per line on stdout.

    {"type": "log", "text": "..."}                         free text for the log view
    {"type": "progress", "step": "name", "value": 0.0-1.0, "text": "..."}
    {"type": "result", "data": {...}}                      the job's answer (exactly one, last)
    {"type": "error", "code": "short_code", "message": "plain-language text for the user"}

Exit status: 0 after a result, 1 after an error."""
import json
import sys


def emit(kind: str, **fields) -> None:
    sys.stdout.write(json.dumps({"type": kind, **fields}, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(text: str) -> None:
    emit("log", text=text)


def progress(step: str, value: float, text: str = "") -> None:
    emit("progress", step=step, value=round(float(value), 4), text=text)


def result(data: dict) -> None:
    emit("result", data=data)


def error(code: str, message: str) -> None:
    emit("error", code=code, message=message)
