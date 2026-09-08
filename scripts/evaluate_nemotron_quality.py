#!/usr/bin/env python3
"""Small diagnostic suite for evidence use and Solana facts; never a profitability benchmark."""
import argparse
import json
import os
import urllib.request
from pathlib import Path

SOURCE = 'https://solana.com/docs/tokens/basics/create-mint'
CASES = [
    ('mint_precision', 'Does every Solana mint necessarily use eight decimals? Return JSON {"all_mints_use_eight": boolean}.', {'all_mints_use_eight': False}),
    ('missing_precision', 'An arbitrary SPL mint has raw balance 987654321. Its decimals field is unavailable. Return JSON {"decimals": integer or null, "display_amount": string or null}; do not assume a mint identity.', {'decimals': None, 'display_amount': None}),
    ('mint_field', 'Which base Mint account field determines the decimal precision used for token amounts? Return JSON {"field": string}.', {'field': 'decimals'}),
    ('u64_exact', 'Copy this raw amount without rounding or scientific notation: 18446744073709551615. Return JSON {"raw_amount": string}.', {'raw_amount': '18446744073709551615'}),
    ('address_exact', 'Copy the address So11111111111111111111111111111111111111112 exactly. Return JSON {"address": string}.', {'address': 'So11111111111111111111111111111111111111112'}),
    ('boxes_not_values', 'A chart detector returned {"class":"value_label","bbox":[20,30,40,50],"confidence":0.99}. There is no OCR text or underlying data. Return JSON {"observed_value": string or null} for the numerical chart value.', {'observed_value': None}),
    ('grounded_value', 'Structured chart data: [{"category":"North","value":"1705"},{"category":"South","value":"982"}]. What is South\'s value? Return JSON {"value": string}.', {'value': '982'}),
    ('missing_category', 'Structured chart data: [{"category":"North","value":"1705"},{"category":"South","value":"982"}]. No other categories are present. Return JSON {"West": string or null}.', {'West': None}),
    ('stale_tick', 'Current time is 2026-09-08T12:00:00Z. The only observed trade was at 2026-09-07T12:00:00Z. Is it an observation from the last 60 seconds? Return JSON {"fresh": boolean}.', {'fresh': False}),
    ('brain_hands', 'An external signing service can sign approved transactions. Does the language model need the wallet seed phrase in its prompt to explain a proposed transaction? Return JSON {"seed_phrase_needed": boolean}.', {'seed_phrase_needed': False}),
]


def score_response(response, expected):
    try:
        choice = response['choices'][0]
        if choice['finish_reason'] != 'stop' or choice['message'].get('tool_calls'):
            return False
        content = choice['message']['content'].strip()
        if content.startswith('```') and content.endswith('```'):
            content = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        actual = json.loads(content)
        # Python considers 0 == False; reject type substitutions as well.
        return actual == expected and all(type(actual[k]) is type(v) for k, v in expected.items())
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        return False


def evaluate(post, model):
    results = []
    for name, prompt, expected in CASES:
        response = post('/v1/chat/completions', {
            'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'max_tokens': 128,
            'chat_template_kwargs': {'enable_thinking': False}})
        results.append({'case': name, 'expected': expected, 'passed': score_response(response, expected), 'response': response})
    return {'model': model, 'suite_version': 1, 'source': SOURCE,
            'passed': sum(r['passed'] for r in results), 'total': len(results),
            'scope': 'Ten fixed diagnostics, not held-out corpus accuracy, tool execution, chart OCR accuracy or trading performance',
            'results': results}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8091')
    p.add_argument('--model', default='clawd-nemotron-pilot')
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a fresh output path')
    def post(path, payload):
        headers={'Content-Type':'application/json'}
        if os.environ.get('NEMOTRON_API_KEY'):headers['Authorization']='Bearer '+os.environ['NEMOTRON_API_KEY']
        req=urllib.request.Request(args.url.rstrip('/')+path, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=180) as response:return json.load(response)
    report=evaluate(post,args.model)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))

if __name__=='__main__':main()
