#!/usr/bin/env python3
"""Bounded NVIDIA inference smoke; public model only, no training-data uploads."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi

MODEL = 'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4'
REVISION = 'cc84af2fe71647d87f4486c064f320e1e7535243'
RUNNER = r'''
import json,subprocess,time,urllib.request
cmd=['vllm','serve','--model','nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4',
 '--revision','cc84af2fe71647d87f4486c064f320e1e7535243','--served-model-name','clawd-nemotron',
 '--host','127.0.0.1','--port','8000','--moe-backend','humming','--linear-backend','humming',
 '--quantization','modelopt_fp4','--mamba-backend','flashinfer','--mamba-cache-mode','align',
 '--mamba-ssu-algorithm','simple','--max-model-len','4096','--max-num-seqs','2',
 '--reasoning-parser','nemotron_v3','--tool-call-parser','qwen3_coder','--enable-auto-tool-choice']
started=time.monotonic(); proc=subprocess.Popen(cmd)
def post(path,payload):
 req=urllib.request.Request('http://127.0.0.1:8000'+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as response:return json.load(response)
try:
 while time.monotonic()-started<2700:
  if proc.poll() is not None:raise RuntimeError('Model server exited during startup')
  try:
   with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5) as response:
    if response.status==200:break
  except Exception:time.sleep(5)
 else:raise TimeoutError('Model startup exceeded 45 minutes')
 print('NEMOTRON_READY '+json.dumps({'startup_seconds':round(time.monotonic()-started,2)}),flush=True)
 prompt='Explain Solana mint decimals in two sentences. Do not request signing keys.'
 begin=time.monotonic()
 answer=post('/v1/chat/completions',{'model':'clawd-nemotron','messages':[{'role':'user','content':prompt}],'max_tokens':256,'temperature':1.0,'top_p':0.95,'chat_template_kwargs':{'enable_thinking':False}})
 print('NEMOTRON_GENERATION '+json.dumps({'seconds':round(time.monotonic()-begin,2),'response':answer}),flush=True)
 text='18446744073709551615 So11111111111111111111111111111111111111112'
 encoded=post('/tokenize',{'model':'clawd-nemotron','prompt':text,'add_special_tokens':False})
 restored=post('/detokenize',{'model':'clawd-nemotron','tokens':encoded['tokens']})
 assert restored['prompt']==text,'Tokenizer roundtrip changed an amount or address'
 tool={'type':'function','function':{'name':'convert_token_amount','description':'Convert an exact base-unit integer to a decimal string.','parameters':{'type':'object','properties':{'amount':{'type':'string'},'decimals':{'type':'integer'}},'required':['amount','decimals']}}}
 results={}
 for choice in ['auto','required']:
  result=post('/v1/chat/completions',{'model':'clawd-nemotron','messages':[{'role':'system','content':'Call the provided conversion tool for any token amount conversion. Never calculate it yourself.'},{'role':'user','content':'Use convert_token_amount with amount 123456789 and decimals 6.'}],'tools':[tool],'tool_choice':choice,'temperature':0,'max_tokens':512,'chat_template_kwargs':{'enable_thinking':False}})
  print('NEMOTRON_TOOL_RESPONSE '+json.dumps({'tool_choice':choice,'response':result}),flush=True)
  calls=result['choices'][0]['message'].get('tool_calls') or []
  valid=False
  if len(calls)==1 and calls[0]['function']['name']=='convert_token_amount':
   valid=json.loads(calls[0]['function']['arguments'])=={'amount':'123456789','decimals':6}
  results[choice]=valid
 print('NEMOTRON_TOOL_CHECKS '+json.dumps(results),flush=True)
 assert results['auto'],'Production auto tool choice failed; inspect recorded response'
 print('NEMOTRON_SMOKE_SUCCESS '+json.dumps({'tokenizer_roundtrip':True,'tool_call_parsed':True,'required_tool_choice_passed':results['required'],'training_performed':False}),flush=True)
finally:
 proc.terminate()
 try:proc.wait(timeout=30)
 except subprocess.TimeoutExpired:proc.kill();proc.wait()
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch',action='store_true')
    args=p.parse_args()
    compile(RUNNER,'nemotron-smoke','exec')
    record={'model':MODEL,'revision':REVISION,'image':'vllm/vllm-openai:v0.27.1','flavor':'a100-large','timeout':'1h','training_performed':False,'uploads':False}
    if args.launch:
        api=HfApi(); owner=api.whoami()['name']
        for job in api.list_jobs(namespace=owner):
            if getattr(job,'name',None)=='clawd-nemotron-inference-smoke' and job.status.stage in ('RUNNING','PENDING','INITIALIZING'):
                raise RuntimeError('An active Nemotron smoke job already exists')
        job=api.run_job(image=record['image'],command=['python3','-u','-c',RUNNER],flavor=record['flavor'],timeout=record['timeout'],namespace=owner,name='clawd-nemotron-inference-smoke',env={'HF_HUB_DISABLE_PROGRESS_BARS':'1'})
        record.update(job_id=job.id,job_url=f'https://huggingface.co/jobs/{owner}/{job.id}')
        root=Path(__file__).resolve().parents[1]/'outputs'
        (root/f'hf-nemotron-smoke-{job.id}.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
