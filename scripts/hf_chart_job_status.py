#!/usr/bin/env python3
"""Read the current chart training Job and its recent logs without exposing secrets."""
import json
from pathlib import Path
from huggingface_hub import HfApi

record = json.loads((Path(__file__).resolve().parents[1] / 'outputs/hf-chart-job.json').read_text())
api = HfApi()
job = api.inspect_job(job_id=record['job_id'])
print(record['job_url'])
print('Status:', job.status.stage, job.status.message or '')
for line in api.fetch_job_logs(job_id=job.id, tail=30):
    print(line)
