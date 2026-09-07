#!/usr/bin/env python3
"""Stage a completed private adapter and its pinned base config for conversion."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

BASE = 'DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU'
BASE_REVISION = 'f3831969c184aa06fdea1bf1060f27a9c9b0eccd'

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-smoke', action='store_true')
    args = p.parse_args()
    api = HfApi()
    info = api.model_info(args.repo, revision=args.revision)
    args.output.mkdir(parents=True, exist_ok=True)
    def download(name):
        return Path(hf_hub_download(args.repo, name, revision=info.sha, local_dir=args.output))
    status = json.loads(download('run-status.json').read_text())
    if status['returncode'] != 0 or (status['smoke_only'] and not args.allow_smoke):
        raise SystemExit('Refusing unsuccessful or unapproved smoke adapter')
    config = json.loads(download('adapter/adapter_config.json').read_text())
    if config['base_model_name_or_path'] != BASE:
        raise SystemExit('Adapter base does not match the configured trainable counterpart')
    download('adapter/adapter_model.safetensors')
    hf_hub_download(BASE, 'config.json', revision=BASE_REVISION, local_dir=args.output / 'base-config')
    (args.output / 'provenance.json').write_text(json.dumps({'repo': args.repo, 'revision': info.sha,
        'base': BASE, 'base_revision': BASE_REVISION, 'smoke_only': status['smoke_only'],
        'status': 'staged_not_deployed'}, indent=2))
    print('Staged adapter at', args.output)
