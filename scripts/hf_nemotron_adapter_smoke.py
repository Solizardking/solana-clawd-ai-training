#!/usr/bin/env python3
"""Bounded private adapter reload test; no uploads or persistent deployment."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from huggingface_hub import HfApi, get_token
from hf_nemotron_smoke_job import RUNNER as BASE_RUNNER

REPO = 'ordlibrary/clawd-nemotron-chart-lora-pilot'
REVISION = '7a763bf1035d5e5350b94e13362724f640f5c312'
NAME = 'clawd-nemotron-adapter-smoke'


def payload(revision=REVISION, checkpoint='checkpoints/epoch_0_step_9/model'):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Pin the exact 40-character adapter commit')
    if checkpoint.startswith('/') or '..' in checkpoint.split('/'):
        raise ValueError('Checkpoint must be a repository-relative directory')
    setup = f'''from huggingface_hub import snapshot_download
from pathlib import Path
root=snapshot_download({REPO!r},revision={revision!r},allow_patterns=[{(checkpoint+'/adapter_config.json')!r},{(checkpoint+'/adapter_model.safetensors')!r}])
adapter=str(Path(root)/{checkpoint!r})
'''
    runner = setup + BASE_RUNNER
    # The tested Humming MoE kernel rejects enable_lora; Marlin declares support.
    runner = runner.replace("'--moe-backend','humming'", "'--moe-backend','marlin'")
    runner = runner.replace("started=time.monotonic(); proc=subprocess.Popen(cmd)", "cmd += ['--enable-lora','--max-lora-rank','8','--lora-modules','clawd-nemotron-pilot='+adapter]\nstarted=time.monotonic(); proc=subprocess.Popen(cmd)")
    runner = runner.replace("'model':'clawd-nemotron'", "'model':'clawd-nemotron-pilot'")
    proof = '''
 with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=10) as response:models=json.load(response)
 assert any(m['id']=='clawd-nemotron-pilot' for m in models['data']), 'Adapter not registered'
 probes={}
 for model_name in ['clawd-nemotron','clawd-nemotron-pilot']:
  probes[model_name]=post('/v1/chat/completions',{'model':model_name,'messages':[{'role':'user','content':'Explain Solana mint decimals.'}],'temperature':0,'max_tokens':1,'logprobs':True,'top_logprobs':5,'chat_template_kwargs':{'enable_thinking':False}})
  assert probes[model_name]['model']==model_name
 base=probes['clawd-nemotron']['choices'][0]['logprobs']
 tuned=probes['clawd-nemotron-pilot']['choices'][0]['logprobs']
 assert base and tuned and base != tuned, 'No measured adapter effect on token probabilities'
 print('NEMOTRON_ADAPTER_EFFECT '+json.dumps({'repo':REPO,'revision':REVISION,'probes':probes}),flush=True)
'''.replace('REPO',repr(REPO)).replace('REVISION',repr(revision))
    runner = runner.replace(" prompt='Explain Solana mint decimals", proof + " prompt='Explain Solana mint decimals")
    quality_source = (Path(__file__).resolve().parent/'evaluate_nemotron_quality.py').read_text()
    quality = "\n suite={'__name__':'quality_suite'}\n exec("+repr(quality_source)+",suite)\n for candidate in ['clawd-nemotron','clawd-nemotron-pilot']:\n  print('NEMOTRON_QUALITY '+json.dumps(suite['evaluate'](post,candidate)),flush=True)\n"
    runner = runner.replace('\nfinally:', quality+'\nfinally:')
    compile(runner, 'adapter-smoke', 'exec')
    return runner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--revision',default=REVISION)
    parser.add_argument('--checkpoint',default='checkpoints/epoch_0_step_9/model')
    args=parser.parse_args()
    runner=payload(args.revision,args.checkpoint)
    record={'name':NAME,'adapter_repo':REPO,'adapter_revision':args.revision,'checkpoint':args.checkpoint,'quality_suite':1,'image':'vllm/vllm-openai:v0.27.1','flavor':'a100-large','timeout':'30m','uploads':False,'runner_sha256':hashlib.sha256(runner.encode()).hexdigest()}
    if args.launch:
        api=HfApi(); owner=api.whoami()['name']
        if owner!='ordlibrary':raise RuntimeError('Expected ordlibrary account')
        token=get_token()
        if not token:raise RuntimeError('HF authentication required')
        for job in api.list_jobs(namespace=owner):
            if getattr(job,'name',None)==NAME and job.status.stage not in {'COMPLETED','ERROR','CANCELED','DELETED'}:
                raise RuntimeError('An adapter smoke job is already active')
        job=api.run_job(image=record['image'],command=['python3','-u','-c',runner],flavor=record['flavor'],timeout=record['timeout'],namespace=owner,name=NAME,secrets={'HF_TOKEN':token},env={'HF_HUB_DISABLE_PROGRESS_BARS':'1'})
        record.update(job_id=job.id,job_url=f'https://huggingface.co/jobs/{owner}/{job.id}')
        (Path(__file__).resolve().parents[1]/'outputs'/f'hf-nemotron-adapter-smoke-{job.id}.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
