#!/usr/bin/env python3
"""Resumable visual benchmark against the base or an explicitly selected LoRA."""
import argparse
import base64
import hashlib
import json
import re
import time
from pathlib import Path
import httpx


def answer_matches(expected, answer):
    # Exact label or scalar, with optional units. Never substring-match numbers.
    normalized = lambda s: re.sub(r'\s+', ' ', s.strip().casefold()).rstrip('.')
    expected, answer = normalized(expected), normalized(answer)
    if expected == answer:
        return True
    if re.fullmatch(r'-?\d+(?:\.\d+)?', expected):
        m = re.fullmatch(r'(-?\d+(?:\.\d+)?)\s*(?:units?)?', answer)
        return bool(m) and float(expected) == float(m[1])
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('outputs/chart-agent/visual-benchmark/manifest.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:8091')
    p.add_argument('--lora-id', type=int)
    args = p.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    cases = json.loads(manifest_bytes)['cases']
    result = {'manifest_sha256': digest, 'lora_id': args.lora_id, 'results': [], 'complete': False}
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result['manifest_sha256'] != digest or result['lora_id'] != args.lora_id:
            raise ValueError('Cannot resume results for a different benchmark or adapter')
    done = {r['id'] for r in result['results'] if 'error' not in r}
    result['results'] = [r for r in result['results'] if 'error' not in r]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=240) as client:
        for case in cases:
            if case['id'] in done: continue
            image = (args.manifest.parent / case['image']).read_bytes()
            if hashlib.sha256(image).hexdigest() != case['image_sha256']:
                raise ValueError('Benchmark image hash mismatch')
            request = {'model': 'clawd-chart-fable', 'temperature': 0, 'max_tokens': 60,
                       'chat_template_kwargs': {'enable_thinking': False},
                       'messages': [{'role': 'user', 'content': [
                           {'type': 'text', 'text': case['question'] + ' Reply only with the category name or number.'},
                           {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(image).decode()}}]}]}
            if args.lora_id is not None: request['lora'] = [{'id': args.lora_id, 'scale': 1.0}]
            start = time.monotonic()
            response = client.post(args.url + '/v1/chat/completions', json=request)
            response.raise_for_status()
            payload = response.json()
            answer = payload['choices'][0]['message'].get('content') or ''
            row = {'id': case['id'], 'chart_type': case['chart_type'], 'expected': case['answer'], 'answer': answer,
                   'correct': answer_matches(case['answer'], answer), 'seconds': time.monotonic() - start}
            result['results'].append(row)
            result['correct'] = sum(r['correct'] for r in result['results'])
            result['evaluated'] = len(result['results'])
            result['complete'] = result['evaluated'] == len(cases)
            args.output.write_text(json.dumps(result, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
