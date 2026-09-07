#!/usr/bin/env python3
"""Run hello world in Novita, optionally verifying one mounted bucket file."""
import argparse
import hashlib
import os
from pathlib import Path

from novita_sandbox.code_interpreter import Sandbox

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="Relative file inside ./local to copy and verify")
    args = parser.parse_args()
    if not os.environ.get("NOVITA_API_KEY"):
        parser.error("Set NOVITA_API_KEY locally before running this command.")
    payload = None
    if args.file:
        mount = (ROOT / "local").resolve()
        if not os.path.ismount(mount):
            parser.error("Mount the bucket first with scripts/charts_bucket.py start")
        path = (mount / args.file).resolve()
        if not path.is_relative_to(mount) or not path.is_file():
            parser.error("--file must name a file inside the mounted bucket")
        with path.open("rb") as stream:
            payload = stream.read(10 * 1024 * 1024 + 1)
        if len(payload) > 10 * 1024 * 1024:
            parser.error("Smoke test files must be 10 MiB or smaller")

    sandbox = Sandbox.create(timeout=120)
    print("Created sandbox:", sandbox.sandbox_id)
    try:
        code = "print('hello world')"
        if payload is not None:
            sandbox.files.write("/tmp/charts-input", payload)
            digest = hashlib.sha256(payload).hexdigest()
            code += (
                "\nimport hashlib\n"
                "from pathlib import Path\n"
                "digest = hashlib.sha256(Path('/tmp/charts-input').read_bytes()).hexdigest()\n"
                f"assert digest == {digest!r}, 'File checksum mismatch'\n"
                "print('Bucket file SHA256 verified:', digest)"
            )
        execution = sandbox.run_code(code, timeout=30)
        for line in execution.logs.stdout:
            print(line, end="" if line.endswith("\n") else "\n")
        if execution.error:
            raise RuntimeError(f"Sandbox code failed: {execution.error.name}")
    finally:
        sandbox.kill()
        print("Sandbox closed.")


if __name__ == "__main__":
    main()
