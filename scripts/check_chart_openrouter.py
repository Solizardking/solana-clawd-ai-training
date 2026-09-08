#!/usr/bin/env python3
"""Validate a private OpenRouter profile without inference charges or secret output."""
import argparse
import json
from pathlib import Path
import sys
import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=Path('.env.openrouter'))
    args = parser.parse_args()
    allowed = {'SOLGPT_API_BASE', 'SOLGPT_API_KEY', 'SOLGPT_APP_TITLE', 'SOLGPT_HTTP_REFERER'}
    values = {}
    for line in args.env_file.read_text().splitlines():
        key, sep, value = line.partition('=')
        if sep and key.strip() in allowed:
            values[key.strip()] = value.strip().strip('\"\'')
    if values.get('SOLGPT_API_BASE', '').rstrip('/') != 'https://openrouter.ai/api/v1':
        raise SystemExit('Profile must target the official OpenRouter API')
    key = values.get('SOLGPT_API_KEY')
    if not key:
        raise SystemExit('Profile has no provider key')
    headers = {'Authorization': 'Bearer ' + key,
               'HTTP-Referer': values.get('SOLGPT_HTTP_REFERER', ''),
               'X-OpenRouter-Title': values.get('SOLGPT_APP_TITLE', '')}
    try:
        with httpx.Client(timeout=20, follow_redirects=False) as client:
            response = client.get('https://openrouter.ai/api/v1/key', headers=headers)
        print(json.dumps({'provider': 'openrouter', 'http_status': response.status_code,
                          'authenticated': response.status_code == 200,
                          'inference_requested': False}))
        return 0 if response.status_code == 200 else 1
    except httpx.HTTPError as exc:
        print(json.dumps({'provider': 'openrouter', 'error': type(exc).__name__}))
        return 1


if __name__ == '__main__':
    sys.exit(main())
