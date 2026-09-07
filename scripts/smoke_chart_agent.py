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
    p.add_argument('--tools', action='store_true')
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
        response = client.post('/tokenize', json={'text': 'Solana Token-2022 mint So11111111111111111111111111111111111111112 decimals=9 PDA ALT'})
        response.raise_for_status()
        report['tokenizer'] = response.json()
        if args.analyze:
            image = Path('outputs/chart-agent/assets/images/syn_bar_0005.png')
            response = client.post('/analyze', json={'question': 'Read the chart: identify the category with the highest value and cite the visible numbers. Keep the answer under 100 words.',
                'image_base64': base64.b64encode(image.read_bytes()).decode(), 'max_tokens': 350})
            response.raise_for_status()
            report['analysis'] = response.json()
        if args.tools:
            response = client.post('/analyze', json={'question': 'Use get_token_candles to retrieve 1-minute OHLCV for Solana mint So11111111111111111111111111111111111111112. Report latest_closed_candle.open_time_utc exactly as supplied, closing price, whether the data is stale, and the source. Do not use research papers for current prices. Keep the answer under 100 words.', 'max_tokens': 400})
            response.raise_for_status()
            report['tool_analysis'] = response.json()
            calls = [c for c in report['tool_analysis']['tools_used'] if c['name'] == 'get_token_candles' and 'latest_closed_candle' in c['result']]
            report['tool_timestamp_exact'] = bool(calls) and calls[-1]['result']['latest_closed_candle']['open_time_utc'] in report['tool_analysis']['answer']
        Path('outputs/chart-agent/service-smoke.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if args.tools and not report['tool_timestamp_exact']:
            raise SystemExit('Model did not reproduce the tool UTC timestamp exactly; see saved report')
