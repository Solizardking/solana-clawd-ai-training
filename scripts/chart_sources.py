#!/usr/bin/env python3
"""Fetch a checksum-verified Solarchive sample and a bounded live tape snapshot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

from clawd_ws_client import fetch_health, recv_pump_frames

ROOT = Path(__file__).resolve().parents[1]
REPO = "solarchive/solarchive"
LIMIT = 10 * 1024 * 1024


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "clawd-chart-research/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError("Source exceeds 10 MiB sample limit")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/chart-research")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = json.loads(fetch(f"https://huggingface.co/api/datasets/{REPO}?expand=sha"))["sha"]
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}"
    index = json.loads(fetch(f"{base}/tokens/2020-10/index.json"))
    item = index["files"][0]
    if Path(item["name"]).name != item["name"] or item["size_bytes"] > LIMIT:
        raise ValueError("Invalid or oversized sample index entry")
    source = f"tokens/2020-10/{item['name']}"
    data = fetch(f"{base}/{source}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != item["sha256"] or len(data) != item["size_bytes"]:
        raise ValueError("Solarchive sample checksum or size mismatch")
    if data[:4] != b"PAR1" or data[-4:] != b"PAR1":
        raise ValueError("Solarchive sample is not Parquet")
    path = args.output / "solarchive-tokens-2020-10.parquet"
    path.write_bytes(data)
    report = {"captured_at": datetime.now(timezone.utc).isoformat(),
              "solarchive": {"repo_id": REPO, "revision": revision, "file": source,
                             "sha256": digest, "bytes": len(data),
                             "license": "CC-BY-4.0", "attribution": "Data from SolArchive.org"}}
    # Persist dataset verification even if the independent live feed is down.
    (args.output / "solarchive-source.json").write_text(json.dumps(report, indent=2) + "\n")
    health = fetch_health(timeout=15)
    if health.get("status") != "ok":
        raise RuntimeError("Clawd live feed health is not ok")
    frames = recv_pump_frames(timeout=15, max_frames=2)
    if not frames:
        raise RuntimeError("No live frames received")
    report["live"] = {"url": "wss://clawd-ws.fly.dev/ws", "health_status": health["status"], "frames": frames}
    (args.output / "sources.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(data)} Solarchive bytes and {len(frames)} live frames; wrote {args.output / 'sources.json'}")


if __name__ == "__main__":
    main()
