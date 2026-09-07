#!/usr/bin/env python3
"""Download the pinned GGUF + projector and verify upstream SHA256 digests."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    config = json.loads((ROOT / 'configs/chart-agent-model.json').read_text())
    output = ROOT / 'outputs/chart-agent/models'
    for asset in config['files']:
        path = hf_hub_download(config['repo'], asset['name'], revision=config['revision'], local_dir=output)
        with open(path, 'rb') as file:
            digest = hashlib.file_digest(file, 'sha256').hexdigest()
        if digest != asset['sha256']:
            raise SystemExit('SHA256 mismatch: ' + asset['name'])
        print('Verified:', asset['name'], flush=True)
