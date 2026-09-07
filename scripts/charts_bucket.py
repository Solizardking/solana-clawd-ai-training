#!/usr/bin/env python3
"""Mount the charts bucket using the same credentials as the Hugging Face CLI."""
import argparse
import os
from pathlib import Path
import subprocess

from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "ordlibrary/charts"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "start", "stop", "status"])
    args = parser.parse_args()
    mount = ROOT / "local"
    if args.action in {"status", "stop"}:
        command = ["hf-mount", args.action]
        if args.action == "stop":
            command.append(str(mount))
        return subprocess.call(command)
    token = get_token()
    if not token:
        parser.error("Log in first: .venv-connect/bin/hf auth login")
    api = HfApi(token=token)
    print("Authenticated account:", api.whoami()["name"], flush=True)
    api.bucket_info(BUCKET)
    print("Bucket accessible:", BUCKET, flush=True)
    if args.action == "check":
        return 0
    if os.path.ismount(mount):
        print("Mount already exists; use status to verify its bucket:", mount)
        return subprocess.call(["hf-mount", "status"])
    if mount.exists() and any(mount.iterdir()):
        parser.error(f"Refusing to hide existing files in {mount}")
    mount.mkdir(exist_ok=True)
    # hf-mount does not automatically use the Hub Python client's token cache.
    # Pass the resolved token through the environment, never the command line.
    env = os.environ.copy()
    env["HF_TOKEN"] = token
    return subprocess.call(
        ["hf-mount", "start", "bucket", BUCKET, str(mount)], env=env
    )


if __name__ == "__main__":
    raise SystemExit(main())
