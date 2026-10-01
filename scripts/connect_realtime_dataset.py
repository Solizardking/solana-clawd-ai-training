#!/usr/bin/env python3
"""Read a bounded preview from the public Hugging Face dataset viewer."""

import argparse
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATASET = "solanaclawd/solana-clawd-realtime-research-instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "eval", "test"), default="train")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Save returned records as JSONL")
    args = parser.parse_args()
    if args.offset < 0 or not 1 <= args.limit <= 100:
        parser.error("--offset must be nonnegative and --limit must be 1..100")
    query = urlencode(dict(dataset=DATASET, config="default", split=args.split,
                           offset=args.offset, length=args.limit))
    request = Request("https://datasets-server.huggingface.co/rows?" + query,
                      headers={"User-Agent": "solana-clawd-dataset-connector/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
        records = [item["row"] for item in payload["rows"]]
        if not records:
            raise ValueError("No rows returned; choose a smaller offset")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({"dataset": DATASET, "split": args.split,
                          "total_rows": payload["num_rows_total"],
                          "preview_rows": len(records),
                          "columns": list(records[0]),
                          "first_record_id": records[0].get("record_id"),
                          "first_message_roles": [m["role"] for m in records[0]["messages"]],
                          "output": str(args.output) if args.output else None}, indent=2))
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError) as exc:
        print(f"Dataset connection failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
