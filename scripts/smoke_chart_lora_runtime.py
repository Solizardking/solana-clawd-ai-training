#!/usr/bin/env python3
"""Verify per-request LoRA execution while default requests remain base-only."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx

with httpx.Client(base_url='http://127.0.0.1:8091', timeout=240) as client:
    response = client.get('/lora-adapters')
    response.raise_for_status()
    adapters = response.json()
    if len(adapters) != 1 or adapters[0]['scale'] != 0:
        raise SystemExit('Expected exactly one adapter loaded with default scale zero')
    body = {'model': 'clawd-chart-fable', 'temperature': 0, 'max_tokens': 40,
            'chat_template_kwargs': {'enable_thinking': False}, 'lora': [{'id': adapters[0]['id'], 'scale': 1}],
            'messages': [{'role': 'user', 'content': 'A chart has Cedar=40 units and Elm=15 units. What is the difference? Reply with the number only.'}]}
    observed_lora = False
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, '/v1/chat/completions', json=body)
        while not pending.done():
            slots_response = client.get('/slots')
            slots_response.raise_for_status()
            for slot in slots_response.json():
                if slot.get('is_processing') and slot.get('params', {}).get('lora') == body['lora']:
                    observed_lora = True
            time.sleep(0.1)
        response = pending.result()
    response.raise_for_status()
    payload = response.json()
    answer = payload['choices'][0]['message'].get('content') or ''
    after = client.get('/lora-adapters')
    after.raise_for_status()
    if after.json()[0]['scale'] != 0:
        raise SystemExit('Per-request adapter unexpectedly changed global default')
    report = {'adapter_id': adapters[0]['id'], 'request_scale': 1, 'default_scale_after_request': 0,
              'answer': answer, 'expected': '25', 'answer_correct': answer.strip() == '25',
              'runtime_request_succeeded': True, 'request_lora_observed_in_slot': observed_lora,
              'smoke_only': True, 'accuracy_improvement_verified': False}
    Path('outputs/chart-agent/smoke-adapter/runtime-smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not observed_lora:
        raise SystemExit('Did not observe the requested LoRA in the active inference slot')
    if not report['answer_correct']:
        raise SystemExit('LoRA runtime answered the arithmetic smoke incorrectly')
