#!/usr/bin/env python3
"""Small held-out vision smoke evaluation; not a profitability benchmark."""
import argparse
import base64
import json
import re
import time
from pathlib import Path
import httpx


def normalized(text):
    return re.sub(r'[^a-z0-9.+-]', '', text.casefold())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8091')
    p.add_argument('--samples', default='outputs/chart-agent/eval-samples.json')
    p.add_argument('--output', default='outputs/chart-agent/vision-evaluation.json')
    args = p.parse_args()
    results = []
    with httpx.Client(timeout=240) as client:
        for row in json.loads(Path(args.samples).read_text()):
            image = Path('outputs/chart-agent/assets') / row['image']
            question, expected = row['messages'][0]['content'], row['messages'][-1]['content']
            started = time.monotonic()
            response = client.post(args.url + '/v1/chat/completions', json={
                'model': 'clawd-chart-fable', 'temperature': 0, 'max_tokens': 100,
                'chat_template_kwargs': {'enable_thinking': False},
                'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': question + ' Reply with only the answer, including its units if applicable.'},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode()}}
                ]}]})
            response.raise_for_status()
            payload = response.json()
            answer = payload['choices'][0]['message'].get('content') or ''
            result = dict(image=row['image'], question=question, expected=expected, answer=answer,
                          normalized_exact_match=normalized(answer) == normalized(expected),
                          seconds=round(time.monotonic() - started, 2), usage=payload.get('usage'))
            results.append(result)
            Path(args.output).write_text(json.dumps(dict(results=results, note='Small local held-out smoke set; no trading-performance claim.'), indent=2))
            print(json.dumps(result), flush=True)
