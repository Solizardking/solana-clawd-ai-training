#!/usr/bin/env python3
"""Pinned Spark 4B inference/quality test. No training data, uploads, or deployment."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi

ROOT=Path(__file__).resolve().parents[1]
MODEL='XHToken/Spark-X2.5-4B'
REVISION='5e10fcc0286756aebf7c41dc52c1e42d95c70281'
NAME='clawd-spark-4b-evaluation'


def payload():
    hashes={name:hashlib.sha256((ROOT/'outputs/spark-preflight'/name).read_bytes()).hexdigest() for name in ['configuration_spark.py','modeling_spark.py']}
    source=r'''
import subprocess,sys,time,json,hashlib,re
subprocess.check_call([sys.executable,'-m','pip','install','--quiet','transformers==4.57.1','accelerate==1.10.1','sentencepiece==0.2.1'],timeout=600)
import torch
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline
assert torch.cuda.is_available(),'GPU required for this test'
start=time.monotonic()
root=Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=['*.json','*.jinja','*.safetensors','*.py','*.model']))
for name,digest in CODE_HASHES.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,'Custom code differs from reviewed revision'
tokenizer=AutoTokenizer.from_pretrained(root,trust_remote_code=False)
model=AutoModelForCausalLM.from_pretrained(root,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='eager').eval()
pipe=pipeline('text-generation',model=model,tokenizer=tokenizer)
hello=pipe([{'role':'user','content':'Who are you?'}],max_new_tokens=96,do_sample=False,tokenizer_encode_kwargs={'enable_thinking':False})
print('SPARK_PIPELINE '+json.dumps({'result':hello,'startup_seconds':round(time.monotonic()-start,2),'gpu':torch.cuda.get_device_name(),'parameters':sum(p.numel() for p in model.parameters())}),flush=True)

def generate(messages,max_new_tokens=128,tools=None):
 prompt=tokenizer.apply_chat_template(messages,tools=tools,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to(model.device)
 assert inputs['input_ids'].shape[-1]<=4096,'Bounded smoke context exceeded'
 torch.cuda.synchronize();begin=time.monotonic()
 with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,logits_to_keep=1,pad_token_id=tokenizer.pad_token_id)
 torch.cuda.synchronize();elapsed=time.monotonic()-begin
 ids=out[0,inputs['input_ids'].shape[-1]:].tolist()
 return {'text':tokenizer.decode(ids,skip_special_tokens=True),'finish_reason':'stop' if ids and ids[-1]==tokenizer.eos_token_id else 'length','new_tokens':len(ids),'seconds':elapsed}

def post(path,payload):
 result=generate(payload['messages'],payload['max_tokens'])
 return {'model':'spark-4b','choices':[{'finish_reason':result['finish_reason'],'message':{'content':result['text']}}],'measurement':result}
namespace={'__name__':'quality'};exec(QUALITY_SOURCE,namespace)
report=namespace['evaluate'](post,'spark-4b')
print('SPARK_QUALITY '+json.dumps(report),flush=True)
roundtrips={s:tokenizer.decode(tokenizer.encode(s,add_special_tokens=False))==s for s in ['18446744073709551615','123.456789','So11111111111111111111111111111111111111112']}
tool={'type':'function','function':{'name':'convert_token_amount','description':'Convert an exact raw token amount using its known mint decimals.','parameters':{'type':'object','properties':{'amount':{'type':'string'},'decimals':{'type':'integer'}},'required':['amount','decimals']}}}
response=generate([{'role':'system','content':'Use the provided conversion tool for amount conversions. Call it exactly once. Do not calculate the result yourself.'},{'role':'user','content':'Call convert_token_amount for raw amount 123456789 with decimals 6.'}],256,[tool])
match=re.fullmatch(r'\s*<tool_call>convert_token_amount(.*?)</tool_call>\s*',response['text'],re.S)
args={};valid=False
if match:
 pairs=re.findall(r'<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>',match.group(1),re.S)
 remainder=re.sub(r'<arg_key>.*?</arg_key><arg_value>.*?</arg_value>','',match.group(1),flags=re.S).strip()
 if len(pairs)==2 and not remainder and len(dict(pairs))==2:
  args=dict(pairs)
  valid=args=={'amount':'123456789','decimals':'6'} and response['finish_reason']=='stop'
print('SPARK_TOOLS '+json.dumps({'response':response,'parsed_arguments':args,'passed':valid,'expected_exact_tool_result':'123.456789'}),flush=True)
print('SPARK_SUMMARY '+json.dumps({'quality_passed':report['passed'],'quality_total':report['total'],'tokenizer_roundtrips':roundtrips,'tool_call_passed':valid,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'training':False,'uploads':False}),flush=True)
'''
    for key,value in [('CODE_HASHES',hashes),('QUALITY_SOURCE',(ROOT/'scripts/evaluate_nemotron_quality.py').read_text()),('MODEL',MODEL),('REVISION',REVISION)]:
        source=source.replace(key,repr(value))
    compile(source,'spark-smoke','exec')
    return source


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    source=payload();r={'name':NAME,'model':MODEL,'revision':REVISION,'image':'pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime','flavor':'a10g-small','timeout':'30m','uploads':False,'training':False,'runner_sha256':hashlib.sha256(source.encode()).hexdigest()}
    if args.launch:
        api=HfApi();owner=api.whoami()['name']
        if owner!='ordlibrary':raise RuntimeError('Expected ordlibrary account')
        for job in api.list_jobs(namespace=owner):
            if getattr(job,'name',None)==NAME and job.status.stage not in {'COMPLETED','ERROR','CANCELED','DELETED'}:raise RuntimeError('Spark test already active')
        job=api.run_job(image=r['image'],command=['python','-u','-c',source],flavor=r['flavor'],timeout=r['timeout'],namespace=owner,name=NAME,env={'HF_HUB_DISABLE_PROGRESS_BARS':'1'})
        r.update(job_id=job.id,job_url=f'https://huggingface.co/jobs/{owner}/{job.id}')
        (ROOT/f'outputs/hf-spark-smoke-{job.id}.json').write_text(json.dumps(r,indent=2))
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
