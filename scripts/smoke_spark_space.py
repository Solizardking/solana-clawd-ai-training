#!/usr/bin/env python3
"""Verify the private Space's real model, tokenizer, conversion tool and OCR path."""
import base64
import json
from pathlib import Path
import httpx
from huggingface_hub import get_token

ROOT=Path(__file__).resolve().parents[1]
keys=json.loads((ROOT/'outputs/spark-hosting-secrets.json').read_text())
headers={'Authorization':'Bearer '+get_token(),'X-Chart-API-Key':keys['CHART_API_KEY']}
url='https://ordlibrary-clawd-spark-chart-agent.hf.space'
with httpx.Client(base_url=url,headers=headers,timeout=300) as client:
    assert client.get('/ready',headers={'X-Chart-API-Key':''}).status_code==401
    r=client.get('/ready');r.raise_for_status();ready=r.json()
    assert ready['model_ready'] and ready['model_backend']=='spark'
    assert ready['detector_ready'] and ready['pattern_detector_ready'] and ready['research_ready']
    r=client.post('/tokenize',json={'text':'18446744073709551615'});r.raise_for_status();tokens=r.json();assert tokens['roundtrip_exact']
    r=client.post('/analyze',json={'question':'Use convert_token_amount to convert raw amount 123456789 with known mint decimals 6 into display units. Report the exact result.','max_tokens':256});r.raise_for_status();conversion=r.json()
    assert any(t['name']=='convert_token_amount' and t['result'].get('decimal_amount')=='123.456789' for t in conversion['tools_used']),conversion
    assert '123.456789' in conversion['answer']
    image=ROOT/'outputs/chart-agent/assets/images/syn_bar_0005.png'
    encoded=base64.b64encode(image.read_bytes()).decode()
    r=client.post('/analyze',json={'question':'Describe only the chart labels and values supported by OCR. State uncertainty. Do not invent a trading signal.','image_base64':encoded,'max_tokens':256});r.raise_for_status();chart=r.json()
    assert chart['evidence']['ocr']['available']
    report={'url':url,'ready':ready,'tokenizer':tokens,'conversion':conversion,'chart':chart,'authentication_checked':True}
    (ROOT/'outputs/spark-space-smoke.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'model':ready['model_name'],'tokenizer_exact':True,'conversion_tool_executed':True,'ocr_available':True,'live_tape_connected':ready['tape']['connected'],'live_tape_stale':ready['tape']['stale'],'chart_answer':chart['answer']},indent=2))
