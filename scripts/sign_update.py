#!/usr/bin/env python3
"""Signs installers for the in-app updater: writes <file>.sig next to each file (docs/UPDATES.md, "Format").

    UPDATE_SIGNING_KEY=<base64 seed> python3 scripts/sign_update.py FILE...          (what release.yml does)
    python3 scripts/sign_update.py --key-file update-signing-key.private.txt FILE...

The .sig file holds the BASE64 text of the 64-byte ed25519 signature of the message  b"ndforge-update-v1:" + the lowercase hex SHA-256 of the file.
The private key is read, never printed, and never written anywhere.
"""
import argparse
import base64
import hashlib
import os
import sys

PREFIX = b"ndforge-update-v1:"


def sha256_hex(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key-file", help="file with the private key (default: the environment variable UPDATE_SIGNING_KEY)")
    ap.add_argument("--outdir", help="write the .sig files here instead of next to the files")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError:
        print("This needs the cryptography package: pip install cryptography", file=sys.stderr)
        return 2
    text = open(a.key_file, encoding="utf-8").read() if a.key_file else os.environ.get("UPDATE_SIGNING_KEY", "")
    try:
        seed = base64.b64decode(text.strip(), validate=True)
        key = Ed25519PrivateKey.from_private_bytes(seed)
    except Exception:
        print("The signing key is missing or is not a base64 ed25519 seed (32 bytes). It is not shown here.", file=sys.stderr)
        return 1
    for f in a.files:
        sig = key.sign(PREFIX + sha256_hex(f).encode())
        out = os.path.join(a.outdir, os.path.basename(f) + ".sig") if a.outdir else f + ".sig"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as o:
            o.write(base64.b64encode(sig).decode() + "\n")
        print("signed", os.path.basename(f))
    return 0


if __name__ == "__main__":
    sys.exit(main())
