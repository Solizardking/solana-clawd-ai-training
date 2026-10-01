#!/usr/bin/env python3
"""Prepare or publish a citation-only update to the current Hub dataset card."""

import argparse
import base64
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
REPO = "solanaclawd/solana-clawd-realtime-research-instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="Publish README.md only; requires HF_TOKEN")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if args.push and not token:
        parser.error("Publishing requires HF_TOKEN in the environment")
    headers = {"User-Agent": "solana-clawd-citations/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def get(url):
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            return response.read()

    try:
        revision = json.loads(get(f"https://huggingface.co/api/datasets/{REPO}"))["sha"]
        card = get(f"https://huggingface.co/datasets/{REPO}/raw/{revision}/README.md").decode()
        citations = (ROOT / "data/realtime_research_citations.md").read_text().strip()
        marker = "## Research Citations\n"
        if marker in card:
            start = card.index(marker)
            end = card.find("\n## ", start + len(marker))
            card = card[:start] + citations + "\n" + (card[end:] if end >= 0 else "")
        else:
            card = card.rstrip() + "\n\n" + citations + "\n"
        output = ROOT / "local/realtime-research-hub-README.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(card, encoding="utf-8")
        print(f"Prepared citation-only card: {output}")
        if args.push:
            operations = [
                {"key": "header", "value": {"summary": "Cite Kamat RED-2400 and hour-aware risk management research", "parentCommit": revision}},
                {"key": "file", "value": {"path": "README.md", "content": base64.b64encode(card.encode()).decode(), "encoding": "base64"}},
            ]
            body = ("\n".join(json.dumps(item) for item in operations) + "\n").encode()
            request = Request(f"https://huggingface.co/api/datasets/{REPO}/commit/main", data=body,
                              headers={**headers, "Content-Type": "application/x-ndjson"}, method="POST")
            with urlopen(request, timeout=60) as response:
                result = json.load(response)
            print("Published:", result.get("commitUrl", result.get("commitOid")))
            published = get(f"https://huggingface.co/datasets/{REPO}/raw/{result['commitOid']}/README.md").decode()
            if published != card:
                raise ValueError("Published card did not match prepared content")
            print("Verified published card matches prepared content")
    except (HTTPError, URLError, OSError, ValueError, KeyError) as exc:
        print(f"Citation update failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
