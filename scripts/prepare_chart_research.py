#!/usr/bin/env python3
"""Index the supplied chart research for the text-only Trading Factory model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

MODEL = "solanaclawd/solana-nvidia-trading-factory-8b"
REVISION = "b5e5cbe55ab5b9e3dc1da3588af3f268990d12ca"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=Path, default=Path.home() / "Downloads/arvix")
    parser.add_argument("--chartdete", type=Path, default=Path.home() / "Downloads/ChartDete-main")
    parser.add_argument("--yolo-paper", type=Path, default=Path.home() / "Downloads/YOLO_Object_Recognition_Algorithm_and_Buy-Sell_Decision_Model_Over_2D_Candlestick_Charts.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/chart-research/bundle.json")
    parser.add_argument("--weights", type=Path, default=Path.home() / "Downloads/solgpt---nl-trading-desk (5)/PUMP-MCP-main/ChartScanAI/weights/custom_yolov8.pt")
    args = parser.parse_args()
    if not shutil.which("pdftotext"):
        parser.error("pdftotext is required (Poppler)")
    if not args.papers.is_dir() or not args.yolo_paper.is_file():
        parser.error("Research PDF directory or YOLO paper is missing")
    if not (args.chartdete / "README.md").is_file():
        parser.error("ChartDete README is missing")
    if not args.weights.is_file():
        parser.error("Custom YOLO weights are missing; specify --weights")
    papers, seen = [], {}
    for path in sorted(args.papers.glob("*.pdf")) + [args.yolo_paper]:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in seen:
            seen[digest]["source_paths"].append(str(path))
            continue
        result = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, timeout=60, check=True)
        text = result.stdout.strip()
        entry = {"name": path.name, "source_paths": [str(path)], "sha256": digest,
                 "text": text, "text_extracted": bool(text)}
        papers.append(entry)
        seen[digest] = entry
    checkpoints = sorted(str(p.relative_to(args.chartdete)) for p in args.chartdete.rglob("*.pth"))
    with args.weights.open("rb") as stream:
        weights_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    bundle = {
        "schema_version": 1,
        "model": {"id": MODEL, "revision": REVISION, "input_type": "text",
                  "hosted_providers_at_setup": [], "inference_verified": False},
        "bucket": "hf://buckets/ordlibrary/charts",
        "custom_yolo": {"path": str(args.weights), "sha256": weights_sha,
                        "size_bytes": args.weights.stat().st_size, "inference_verified": False},
        "historical_data": {"repo_id": "solarchive/solarchive", "repo_type": "dataset",
                            "attribution": "Data from SolArchive.org", "license": "CC-BY-4.0",
                            "sample_connector": "scripts/chart_sources.py"},
        "live_data": {"health_url": "https://clawd-ws.fly.dev/health",
                      "ws_url": "wss://clawd-ws.fly.dev/ws",
                      "client": "scripts/clawd_ws_client.py"},
        "chartdete": {"source_path": str(args.chartdete), "readme": (args.chartdete / "README.md").read_text(),
                      "checkpoints": checkpoints, "inference_verified": False},
        "papers": papers,
        "workflow": "Chart detector -> structured detections plus OHLCV context -> text model research analysis",
        "execution_mode": "research_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2) + "\n")
    print(f"Prepared {len(papers)} unique PDFs; {len(checkpoints)} ChartDete checkpoints")
    print(f"Bundle: {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
