#!/usr/bin/env python3
"""Exercise the built Linux API container with local detector/research assets."""
import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
import httpx

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--image', required=True, type=Path)
p.add_argument('--tag', default='clawd-chart-api:local')
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
key = secrets.token_urlsafe(32)
env = {**os.environ, 'CHART_API_KEY': key}
name = 'clawd-chart-smoke-' + secrets.token_hex(4)
container = subprocess.check_output(['docker', 'run', '--rm', '-d', '--name', name,
    '-p', '127.0.0.1:18090:8090', '-e', 'CHART_API_KEY',
    '-v', str(root / 'outputs/chart-agent') + ':/app/outputs/chart-agent:ro', args.tag], env=env, text=True).strip()
try:
    with httpx.Client(base_url='http://127.0.0.1:18090', timeout=90) as client:
        for _ in range(50):
            try:
                response = client.get('/health', timeout=2)
                if response.status_code == 200: break
            except httpx.TransportError: pass
            time.sleep(1)
        else: raise RuntimeError('Container did not become healthy')
        assert client.get('/ready').status_code == 401
        headers = {'Authorization': 'Bearer ' + key}
        response = client.get('/ready', headers=headers)
        response.raise_for_status()
        ready = response.json()
        for field in ('detector_ready', 'pattern_detector_ready', 'research_ready'):
            assert ready[field], field
        response = client.post('/detect', headers=headers, json={'image_base64': base64.b64encode(args.image.read_bytes()).decode()})
        response.raise_for_status()
        detection = response.json()
        report = {'container_image': args.tag, 'image_id': subprocess.check_output(['docker', 'image', 'inspect', args.tag, '--format', '{{.Id}}'], text=True).strip(),
                  'authentication_checked': True, 'readiness': {k: v for k,v in ready.items() if k != 'tape'},
                  'detection': detection, 'cuda_model_container_tested': False}
        output = root / 'outputs/chart-agent/container-smoke.json'
        output.write_text(json.dumps(report, indent=2))
        print('API container authentication, detector loading, research and /detect passed.')
        print('Evidence:', output)
finally:
    subprocess.run(['docker', 'stop', '--time', '5', container], check=True, stdout=subprocess.DEVNULL)
