#!/usr/bin/env python3
"""Check authenticated local app and optionally run one complete analysis."""
import argparse
import base64
import json
from pathlib import Path
import httpx

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analyze', action='store_true')
    args = p.parse_args()
    key = Path('outputs/chart-agent/api-key').read_text().strip()
    with httpx.Client(base_url='http://127.0.0.1:8090', timeout=330, headers={'Authorization': 'Bearer ' + key}) as client:
        report = {}
        for endpoint in ('ready', 'tools', 'live'):
            response = client.get('/' + endpoint)
            response.raise_for_status()
            data = response.json()
            if endpoint == 'tools':
                data = {'catalog_count': data['catalog_count'], 'connected_count': sum(t['connected'] for t in data['tools'])}
            if endpoint == 'live':
                data['events'] = [{'received_at': e['received_at'], 'type': e['event'].get('type')} for e in data['events']]
            if endpoint == 'ready':
                data.pop('tape', None)
            report[endpoint] = data
        if args.analyze:
            image = Path('outputs/chart-agent/assets/images/syn_bar_0005.png')
            response = client.post('/analyze', json={'question': 'Read the chart: identify the category with the highest value and cite the visible numbers. Keep the answer under 100 words.',
                'image_base64': base64.b64encode(image.read_bytes()).decode(), 'max_tokens': 350})
            response.raise_for_status()
            report['analysis'] = response.json()
        Path('outputs/chart-agent/service-smoke.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
