#!/usr/bin/env python3
"""Download selected charts assets to local output; never modify the bucket."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('files', nargs='+')
    args = p.parse_args()
    root = Path('outputs/chart-agent/assets')
    root.mkdir(parents=True, exist_ok=True)
    for name in args.files:
        if name.startswith('/') or '..' in Path(name).parts:
            p.error('Relative bucket paths only')
    try:
        HfApi().download_bucket_files('ordlibrary/charts', [(name, root / name) for name in args.files], raise_on_missing_files=True)
    except Exception as exc:
        print('Bucket download failed:', type(exc).__name__)
        raise SystemExit(1)
