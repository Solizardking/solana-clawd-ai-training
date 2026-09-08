#!/usr/bin/env python3
"""One-step public-data LoRA compatibility check; no private data or artifact uploads."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml
from huggingface_hub import HfApi

ROOT=Path(__file__).resolve().parents[1]
REVISION='a9904d24bcc1d289a1950fa9d2b978c47cf903b9'
NAME='clawd-nemotron-lora-compatibility'


def payload():
    config=yaml.safe_load((ROOT/'deploy/nemotron/training/upstream-lora.yaml').read_text())
    config['model']['revision']=REVISION
    config['step_scheduler'].update(global_batch_size=1,local_batch_size=1,max_steps=1,
                                    ckpt_every_steps=1000,val_every_steps=1000)
    config['checkpoint']['enabled']=False
    config['lr_scheduler']['lr_warmup_steps']=0
    config['dataloader']['num_workers']=0
    for key,split in [('dataset','train'),('validation_dataset','validation')]:
        config[key]['path']=f'/tmp/nemotron-compat/{split}.jsonl'
        config[key]['seq_length']=16384
        config[key]['tokenizer']['revision']=REVISION
    rows=[{'messages':[{'role':'user','content':q,'reasoning_content':''},
                      {'role':'assistant','content':a,'reasoning_content':''}],'tools':'[]'}
          for q,a in [('What is a Solana mint decimal count?','It specifies the number of decimal places used to display base-unit token amounts.'),
                      ('Should an analysis model receive a signing key?','No. Keep transaction signing separate from analysis.'),
                      ('Is OCR a verified token amount?','No. OCR can misread digits and requires verification.'),
                      ('Does a chart detector prove profitability?','No. Detection labels alone do not establish trading performance.')]]
    runner='''import json,pathlib,subprocess,sys,time
root=pathlib.Path('/tmp/nemotron-compat');root.mkdir(exist_ok=True)
config=CONFIG
rows=ROWS
for split in ['train','validation']:
 (root/(split+'.jsonl')).write_text(''.join(json.dumps(row)+'\\n' for row in rows))
(root/'config.yaml').write_text(json.dumps(config))
print('NEMOTRON_LORA_COMPAT_START '+json.dumps({'steps':1,'rows':len(rows),'private_data':False,'checkpoint_retained':False}),flush=True)
subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],check=True)
started=time.monotonic()
result=subprocess.run(['automodel',str(root/'config.yaml'),'--nproc-per-node','1'],cwd='/opt/Automodel',timeout=1500)
print('NEMOTRON_LORA_COMPAT_RESULT '+json.dumps({'returncode':result.returncode,'seconds':round(time.monotonic()-started,2),'checkpoint_retained':False,'full_context_memory_test':False}),flush=True)
sys.exit(result.returncode)
'''.replace('CONFIG',repr(config)).replace('ROWS',repr(rows))
    compile(runner,'nemotron-lora-compat','exec')
    return runner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch',action='store_true')
    args=p.parse_args()
    runner=payload()
    report={'image':'nvcr.io/nvidia/nemo-automodel:26.08','flavor':'a100-large',
            'timeout':'30m','model_revision':REVISION,'steps':1,'private_data':False,
            'artifact_uploads':False,'checkpoint_retained':False,
            'runner_sha256':hashlib.sha256(runner.encode()).hexdigest()}
    if args.launch:
        api=HfApi();owner=api.whoami()['name']
        terminal={'COMPLETED','ERROR','CANCELED','DELETED'}
        for job in api.list_jobs(namespace=owner):
            if getattr(job,'name',None)==NAME and job.status.stage not in terminal:
                raise RuntimeError('An existing compatibility job is not terminal')
        job=api.run_job(image=report['image'],command=['python3','-u','-c',runner],
                       flavor=report['flavor'],timeout=report['timeout'],namespace=owner,name=NAME,
                       env={'HF_HUB_DISABLE_PROGRESS_BARS':'1','PYTHONUNBUFFERED':'1'})
        report.update(job_id=job.id,job_url=f'https://huggingface.co/jobs/{owner}/{job.id}')
        (ROOT/f'outputs/hf-nemotron-lora-compat-{job.id}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
