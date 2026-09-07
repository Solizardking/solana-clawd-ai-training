#!/usr/bin/env python3
"""Run hello world in Novita, optionally verifying one mounted bucket file."""
import argparse
import hashlib
import os
from pathlib import Path

from novita_sandbox.code_interpreter import Sandbox
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="Relative file inside ./local to copy and verify")
    parser.add_argument("--env-file", type=Path, help="Private dotenv file with Novita credentials")
    parser.add_argument("--bundle", type=Path, help="Local research bundle to copy and verify")
    args = parser.parse_args()
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("Credential file does not exist")
        values = dotenv_values(args.env_file, interpolate=False)
        for key in ("NOVITA_API_KEY", "NOVITA_USER_ID"):
            if values.get(key):
                os.environ[key] = values[key]
    if not os.environ.get("NOVITA_API_KEY"):
        parser.error("Set NOVITA_API_KEY locally before running this command.")
    if args.file and args.bundle:
        parser.error("Choose --file or --bundle")
    payload = None
    path = None
    if args.file:
        mount = (ROOT / "local").resolve()
        if not os.path.ismount(mount):
            parser.error("Mount the bucket first with scripts/charts_bucket.py start")
        path = (mount / args.file).resolve()
        if not path.is_relative_to(mount) or not path.is_file():
            parser.error("--file must name a file inside the mounted bucket")
    if args.bundle:
        path = args.bundle.resolve()
        if not path.is_file():
            parser.error("Research bundle does not exist")
    if path is not None:
        with path.open("rb") as stream:
            payload = stream.read(10 * 1024 * 1024 + 1)
        if len(payload) > 10 * 1024 * 1024:
            parser.error("Smoke test files must be 10 MiB or smaller")

    sandbox = Sandbox.create(timeout=120)
    try:
        print("Created sandbox:", sandbox.sandbox_id)
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
