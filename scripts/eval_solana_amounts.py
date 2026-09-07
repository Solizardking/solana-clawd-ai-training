#!/usr/bin/env python3
"""Independent exact-string Solana amount and address benchmark; no training writes."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import httpx


def cases():
    rows=[]
    for raw,decimals in [('9007199254740993',9),('18446744073709551615',6),('1',9),('123456789012345678',8)]:
        padded=raw.zfill(decimals+1)
        ui=padded[:-decimals]+'.'+padded[-decimals:]
        rows.append({'id':'raw-'+raw, 'question':f'A Solana token amount is the integer string {raw} and its mint decimals is {decimals}. Divide by 10^{decimals}. Return only the exact decimal amount with exactly {decimals} fractional digits, no commas or units.', 'expected':ui})
        rows.append({'id':'ui-'+raw, 'question':f'A Solana token amount is {ui} tokens. The mint decimals is {decimals}. Multiply by 10^{decimals}. Return only the exact raw integer amount, no commas or units.', 'expected':raw})
    for before,after in [(9007199254740993,9007199254740992),(18446744073709551615,18446744073709551600)]:
        rows.append({'id':'delta-'+str(before),'question':f'A raw token balance decreased from {before} to {after}. Return only the exact raw integer decrease, no commas or units.', 'expected':str(before-after)})
    for mint in ['So11111111111111111111111111111111111111112','8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump']:
        rows.append({'id':'mint-'+mint,'question':f'Return this Solana mint address verbatim, with no surrounding text: {mint}','expected':mint})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--url',default='http://127.0.0.1:8091')
    p.add_argument('--lora-id',type=int)
    args=p.parse_args()
    suite=cases();digest=hashlib.sha256(json.dumps(suite,sort_keys=True).encode()).hexdigest()
    with httpx.Client(timeout=240) as client:
        a=client.get(args.url+'/lora-adapters');a.raise_for_status();adapters=a.json()
        if args.lora_id is None and any(x['scale']!=0 for x in adapters):
            raise ValueError('Base benchmark requires all default adapter scales to be zero')
        selected=next((x for x in adapters if x['id']==args.lora_id),None)
        if args.lora_id is not None and selected is None: raise ValueError('Requested adapter is not loaded')
        identity={'id':selected['id'],'path':selected['path']} if selected else None
        report={'suite_sha256':digest,'adapter':identity,'results':[],'complete':False,'training_data':False}
        if args.output.exists():
            report=json.loads(args.output.read_text())
            if report['suite_sha256']!=digest or report['adapter']!=identity:
                raise ValueError('Cannot resume a different benchmark or adapter')
        done={r['id'] for r in report['results']}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        for case in suite:
            if case['id'] in done: continue
            t=client.post(args.url+'/tokenize',json={'content':case['question'],'add_special':False});t.raise_for_status()
            restored=client.post(args.url+'/detokenize',json={'tokens':t.json()['tokens']});restored.raise_for_status()
            request={'model':'clawd-chart-fable','temperature':0,'max_tokens':100,
                'chat_template_kwargs':{'enable_thinking':False},'messages':[{'role':'user','content':case['question']}]}
            if selected:request['lora']=[{'id':args.lora_id,'scale':1}]
            start=time.monotonic();r=client.post(args.url+'/v1/chat/completions',json=request);r.raise_for_status()
            answer=r.json()['choices'][0]['message'].get('content') or ''
            row={**case,'answer':answer,'correct':answer.strip()==case['expected'],
                'tokenizer_roundtrip_exact':restored.json()['content']==case['question'],'seconds':time.monotonic()-start}
            report['results'].append(row);report['correct']=sum(x['correct'] for x in report['results'])
            report['complete']=len(report['results'])==len(suite)
            args.output.write_text(json.dumps(report,indent=2))
            print(json.dumps(row),flush=True)

if __name__=='__main__':main()
