"""Entry point: python -m ndforge_worker <job.json>. The job file is JSON with a "job" key; see protocol.py for the output."""
import json
import sys

from . import __version__, protocol


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
    if kind == "hello":
        protocol.result({"worker": __version__})
        return 0
    if kind == "selftest":
        from . import selftest
        return selftest.run(job)
    protocol.error("unknown_job", f"Unknown job type: {kind!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
