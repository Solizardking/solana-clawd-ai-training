#!/usr/bin/env python3
"""Measure pinned chat-template lengths without weights; not a NeMo loss-mask audit."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer',type=Path,default=Path('outputs/nemotron-bf16-tokenizer'))
    p.add_argument('--data',type=Path,default=Path('outputs/nemotron-chart-data'))
    p.add_argument('--output',type=Path,default=Path('outputs/nemotron-token-length-audit.json'))
    args=p.parse_args()
    if args.output.exists():
        p.error('Choose a fresh report path')
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    report={'splits':{},'weights_loaded':False,'nemo_loss_mask_verified':False,
            'template_mode':'default, no generation prompt; tools decoded from serialized list',
            'tokenizer_files_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest()
                for x in args.tokenizer.iterdir() if x.is_file()}}
    for split in ['train','validation','test']:
        lengths=[]
        with (args.data/f'{split}.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                tokens=tokenizer.apply_chat_template(row['messages'],tools=json.loads(row['tools']) or None,
                                                     tokenize=True,add_generation_prompt=False,return_dict=False)
                if not isinstance(tokens,list) or not all(isinstance(token,int) for token in tokens):
                    raise TypeError('Expected a flat token ID list, not a tokenizer result mapping')
                lengths.append(len(tokens))
        values=sorted(lengths)
        report['splits'][split]={'rows':len(values),'max_tokens':max(values),'total_tokens':sum(values),
            'p50':values[len(values)//2],'p95':values[int(len(values)*.95)],
            'over_4096':sum(x>4096 for x in values),'over_8192':sum(x>8192 for x in values),
            'over_16384':sum(x>16384 for x in values),
            'input_sha256':hashlib.sha256((args.data/f'{split}.jsonl').read_bytes()).hexdigest()}
        print(json.dumps({split:report['splits'][split]}),flush=True)
    strings=['18446744073709551615','123.456789','So11111111111111111111111111111111111111112']
    report['exact_roundtrips']={s:tokenizer.decode(tokenizer.encode(s,add_special_tokens=False))==s for s in strings}
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
