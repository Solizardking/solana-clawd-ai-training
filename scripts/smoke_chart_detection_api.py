#!/usr/bin/env python3
"""Verify both detectors through the authenticated localhost API."""
import argparse
import base64
import json
from pathlib import Path
import httpx

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path)
    args = p.parse_args()
    key = Path('outputs/chart-agent/api-key').read_text().strip()
    with httpx.Client(base_url='http://127.0.0.1:8090', timeout=30,
                      headers={'Authorization': 'Bearer ' + key}) as client:
        response = client.post('/detect', json={'image_base64': base64.b64encode(args.image.read_bytes()).decode()})
        response.raise_for_status()
        report = response.json()
    assert 'pattern_detector' in report and 'detector_classes' in report
    Path('outputs/chart-agent/detection-api-smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
