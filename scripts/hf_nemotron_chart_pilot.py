#!/usr/bin/env python3
"""Ten-step real-data pilot from the existing private HF package, with private adapter retention."""
import argparse
import hashlib
import json
from pathlib import Path
import yaml
from huggingface_hub import HfApi, get_token

ROOT=Path(__file__).resolve().parents[1]
NAME='clawd-nemotron-chart-pilot'
DATA_REPO='ordlibrary/clawd-chart-foundation-training'
DATA_REVISION='6456947cf89cbb5d1a4c0c120199a5038532bfc6'
ARCHIVE_SHA='3e389d917a4cecac5c9994dcfa68434c55cff7a21db04cb44803849a9e3f24df'
MODEL_REPO='ordlibrary/clawd-nemotron-chart-lora-pilot'


def payload(representative=False):
    config=yaml.safe_load((ROOT/'outputs/nemotron-lora-pilot-16k/pilot.yaml').read_text())
    config['step_scheduler'].update(global_batch_size=1,local_batch_size=1,max_steps=10,ckpt_every_steps=5,val_every_steps=1000)
    config['distributed']['activation_checkpointing']=True
    # Avoid materializing the full vocabulary logits and their fp32 loss copy.
    config['loss_fn']={'_target_':'nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy'}
    config['model']['output_hidden_states']=True
    config['checkpoint'].update(checkpoint_dir='/tmp/nemotron-pilot/output/checkpoints',enabled=True)
    config['lr_scheduler']['lr_warmup_steps']=2
    config['dataloader'].update(num_workers=0,shuffle=False)
    if representative:
        config['step_scheduler'].update(global_batch_size=8,max_steps=100,ckpt_every_steps=25,val_every_steps=1000)
        config['lr_scheduler']['lr_warmup_steps']=10
    for key,split in [('dataset','train'),('validation_dataset','validation')]:
        config[key]['path']=f'/tmp/nemotron-pilot/selected/{split}.jsonl'
    preparation=(ROOT/'scripts/prepare_nemotron_chart_data.py').read_text()
    expected=json.loads((ROOT/'outputs/nemotron-chart-data/manifest.json').read_text())['outputs_sha256']
    runner=r'''import hashlib,json,os,pathlib,signal,subprocess,sys,time,zipfile
from huggingface_hub import HfApi,hf_hub_download
from transformers import AutoTokenizer
root=pathlib.Path('/tmp/nemotron-pilot');root.mkdir(exist_ok=True)
raw=root/'raw';raw.mkdir(exist_ok=True)
archive=pathlib.Path(hf_hub_download(os.environ['DATA_REPO'],'chart-foundation-data.zip',repo_type='dataset',revision=os.environ['DATA_REVISION']))
with archive.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==os.environ['ARCHIVE_SHA']
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():
  if not (raw/name).resolve().is_relative_to(raw.resolve()):raise ValueError('Unsafe archive member')
 z.extractall(raw)
namespace={'__name__':'data_preparation'}
exec(PREPARATION,namespace)
ready=root/'prepared';namespace['prepare'](raw,ready)
for split in ['train','validation']:
 with (ready/(split+'.jsonl')).open('rb') as f:
  assert hashlib.file_digest(f,'sha256').hexdigest()==EXPECTED[split+'.jsonl']
print('NEMOTRON_REAL_PILOT_DATA_VERIFIED',flush=True)
config=CONFIG
out=root/'output';out.mkdir(exist_ok=True)
selected=root/'selected';selected.mkdir(exist_ok=True)
tokenizer=AutoTokenizer.from_pretrained(config['model']['pretrained_model_name_or_path'],revision=config['model']['revision'],trust_remote_code=False)
selection={}
for split,count in [('train',64),('validation',8)]:
 rows=[json.loads(line) for line in (ready/(split+'.jsonl')).read_text().splitlines()]
 ranked=[]
 for index,row in enumerate(rows):
  ids=tokenizer.apply_chat_template(row['messages'],tools=json.loads(row['tools']) or None,tokenize=True,add_generation_prompt=False,return_dict=False)
  assert isinstance(ids,list) and all(isinstance(x,int) for x in ids)
  ranked.append((len(ids),index))
 ranked.sort(reverse=True)
 # The longest example runs first; others span the observed length distribution.
 chosen=[ranked[round(i*(len(ranked)-1)/(count-1))] for i in range(count)]
 assert max(n for n,_ in chosen)<=config['dataset']['seq_length']
 (selected/(split+'.jsonl')).write_text(''.join(json.dumps(rows[index],ensure_ascii=False)+'\n' for _,index in chosen))
 selection[split]={'source_rows':len(rows),'selected_rows':count,'indices':[i for _,i in chosen],'token_lengths':[n for n,_ in chosen]}
(out/'selection.json').write_text(json.dumps(selection,indent=2))
(out/'pilot.yaml').write_text(json.dumps(config,indent=2))
print('NEMOTRON_REAL_PILOT_SELECTION '+json.dumps(selection),flush=True)
started=time.monotonic()
process=subprocess.Popen(['automodel',str(out/'pilot.yaml'),'--nproc-per-node','1'],cwd='/opt/Automodel',start_new_session=True)
try:
 returncode=process.wait(timeout=2700)
except subprocess.TimeoutExpired:
 os.killpg(process.pid,signal.SIGTERM)
 try:process.wait(timeout=30)
 except subprocess.TimeoutExpired:
  os.killpg(process.pid,signal.SIGKILL);process.wait()
 returncode=124
status={'returncode':returncode,'seconds':round(time.monotonic()-started,2),'full_training_completed':False,'visual_rows_used':0,'steps_requested':10,'data_revision':os.environ['DATA_REVISION']}
(out/'run-status.json').write_text(json.dumps(status,indent=2))
files=[p for p in out.rglob('*') if p.is_file()]
if sum(p.stat().st_size for p in files)>2_000_000_000:raise RuntimeError('Refusing unexpectedly large checkpoint upload')
api=HfApi()
if not api.repo_info(os.environ['MODEL_REPO']).private:raise RuntimeError('Output repository must remain private')
api.upload_folder(repo_id=os.environ['MODEL_REPO'],folder_path=str(out),commit_message='Retain bounded Nemotron chart pilot checkpoint and provenance')
print('NEMOTRON_REAL_PILOT_RESULT '+json.dumps(status),flush=True)
sys.exit(returncode)
'''
    if representative:
        runner=runner.replace("import hashlib,json,os,pathlib,signal,subprocess,sys,time,zipfile", "import hashlib,json,os,pathlib,random,signal,subprocess,sys,time,zipfile")
        runner=runner.replace("[('train',64),('validation',8)]", "[('train',800),('validation',32)]")
        runner=runner.replace("chosen=[ranked[round(i*(len(ranked)-1)/(count-1))] for i in range(count)]", "chosen=random.Random(1111).sample(ranked,count) if split=='train' else [ranked[round(i*(len(ranked)-1)/(count-1))] for i in range(count)]")
        runner=runner.replace("'steps_requested':10", "'steps_requested':100")
        runner=runner.replace("folder_path=str(out),commit_message=", "folder_path=str(out),path_in_repo='representative-100',commit_message=")
    runner=runner.replace('PREPARATION',repr(preparation)).replace('EXPECTED',repr(expected)).replace('CONFIG',repr(config))
    compile(runner,'nemotron-real-pilot','exec')
    return runner


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--launch',action='store_true')
    p.add_argument('--representative',action='store_true',help='100 steps on 800 seeded random rows; keep checkpoint under representative-100')
    args=p.parse_args()
    runner=payload(args.representative)
    report={'name':NAME,'image':'nvcr.io/nvidia/nemo-automodel:26.08','flavor':'a100-large','timeout':'1h',
            'steps':10,'dataset_repo':DATA_REPO,'dataset_revision':DATA_REVISION,'model_repo':MODEL_REPO,
            'private_output':True,'uploads':'pilot adapters, optimizer state, configuration, selection indices and status; no detector or raw dataset upload',
            'runner_sha256':hashlib.sha256(runner.encode()).hexdigest(),'full_training':False}
    report['loss']='FusedLinearCrossEntropy'
    report['model_output_hidden_states']=True
    if args.representative:
        report.update(steps=100,training_rows=800,validation_rows=32,output_prefix='representative-100')
    if args.launch:
        api=HfApi();owner=api.whoami()['name']
        if owner!='ordlibrary':raise RuntimeError('Expected the existing ordlibrary training account')
        terminal={'COMPLETED','ERROR','CANCELED','DELETED'}
        for job in api.list_jobs(namespace=owner):
            if getattr(job,'name',None)==NAME and job.status.stage not in terminal:raise RuntimeError('A pilot is already active')
        api.create_repo(MODEL_REPO,private=True,exist_ok=True)
        if not api.repo_info(MODEL_REPO).private:raise RuntimeError('Refusing public output repository')
        job=api.run_job(image=report['image'],command=['python3','-u','-c',runner],flavor=report['flavor'],timeout=report['timeout'],namespace=owner,name=NAME,
            env={'DATA_REPO':DATA_REPO,'DATA_REVISION':DATA_REVISION,'ARCHIVE_SHA':ARCHIVE_SHA,'MODEL_REPO':MODEL_REPO,'HF_HUB_DISABLE_PROGRESS_BARS':'1','PYTHONUNBUFFERED':'1'},secrets={'HF_TOKEN':get_token()})
        report.update(job_id=job.id,job_url=f'https://huggingface.co/jobs/{owner}/{job.id}')
        (ROOT/f'outputs/hf-nemotron-chart-pilot-{job.id}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
