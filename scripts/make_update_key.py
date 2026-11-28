#!/usr/bin/env python3
"""Makes the ed25519 key pair that signs the installers (docs/UPDATES.md). Run it ON YOUR OWN PC, not in CI.

    pip install cryptography
    python3 scripts/make_update_key.py [--out update-signing-key.private.txt]

It prints the PUBLIC key (paste it into UPDATE_PUBLIC_KEY in crates/ndforge-core/src/update/key.rs) and writes the PRIVATE key to a file (base64 of the 32-byte seed, one line, readable by you only).
The private key is never printed. Put the file's content into the GitHub secret UPDATE_SIGNING_KEY, then keep the file somewhere safe and OUT of the repository. An existing file is never overwritten.
"""
import argparse
import base64
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="update-signing-key.private.txt", help="where the private key is written (default: ./update-signing-key.private.txt)")
    a = ap.parse_args()
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError:
        print("This needs the cryptography package: pip install cryptography", file=sys.stderr)
        return 2
    if os.path.exists(a.out):
        print(f"{a.out} already exists and was not touched. Pick another --out, or move the old key away on purpose.", file=sys.stderr)
        return 1
    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(base64.b64encode(seed).decode() + "\n")
    print("Public key (paste into UPDATE_PUBLIC_KEY in crates/ndforge-core/src/update/key.rs):")
    print(base64.b64encode(pub).decode())
    print(f"\nThe private key was written to {a.out} (not shown). Copy its content into the GitHub secret UPDATE_SIGNING_KEY, then store the file safely outside the repository.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
