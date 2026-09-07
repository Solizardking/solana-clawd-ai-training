#!/usr/bin/env python3
"""Verify persisted chart-training artifacts at a pinned Hub revision."""
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

root = Path(__file__).resolve().parents[1]
record = json.loads((root / 'outputs/hf-chart-job.json').read_text())
api = HfApi()
job = api.inspect_job(job_id=record['job_id'])
if job.status.stage != 'COMPLETED':
    raise SystemExit('Job is not completed: ' + job.status.stage)
repo = api.model_info(record['model_repo'], files_metadata=True)
files = {f.rfilename: f for f in repo.siblings}
required = ['run-status.json', 'metrics.json', 'data-manifest.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors']
for name in required:
    if name not in files or not files[name].size:
        raise SystemExit('Missing or empty artifact: ' + name)
out = root / 'outputs/chart-training-verification'
out.mkdir(parents=True, exist_ok=True)
artifacts = {}
for name in required:
    if name.endswith('.json'):
        path = hf_hub_download(record['model_repo'], name, revision=repo.sha, local_dir=out)
        artifacts[name] = json.loads(Path(path).read_text())
status = artifacts['run-status.json']
if status['returncode'] != 0 or status['data_revision'] != record['data_revision']:
    raise SystemExit('Persisted run status does not match successful job input')
report = {'job_id': job.id, 'job_stage': job.status.stage, 'model_repo': record['model_repo'],
          'model_revision': repo.sha, 'private': repo.private, 'smoke_only': status['smoke_only'],
          'adapter_bytes': files['adapter/adapter_model.safetensors'].size,
          'metrics': artifacts['metrics.json'], 'full_training_verified': not status['smoke_only']}
(out / 'verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
