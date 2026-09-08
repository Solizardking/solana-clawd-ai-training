#!/usr/bin/env python3
"""Stage a private training package and launch a bounded Hugging Face GPU Job."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, get_token
from huggingface_hub.utils import disable_progress_bars

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=('27b','4b'), default='27b')
    p.add_argument('--full', action='store_true', help='Run one full epoch per stage (default: smoke)')
    p.add_argument('--timeout', default='1h')
    args = p.parse_args()
    disable_progress_bars()
    api = HfApi()
    owner = api.whoami()['name']
    data_repo = f'{owner}/clawd-chart-foundation-training'
    model_repo = f'{owner}/clawd-chart-foundation-{args.profile}-' + ('lora' if args.full else 'smoke')
    archive = ROOT / 'outputs/chart-foundation-data.zip'
    digest = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
    api.create_repo(data_repo, repo_type='dataset', private=True, exist_ok=True)
    api.create_repo(model_repo, private=True, exist_ok=True)
    for repo, kind in ((data_repo, 'dataset'), (model_repo, 'model')):
        if not api.repo_info(repo, repo_type=kind).private:
            raise RuntimeError(f'Refusing to stage local training materials in public repository {repo}')
    commit = api.create_commit(repo_id=data_repo, repo_type='dataset', commit_message='Stage reproducible chart training inputs', operations=[
        CommitOperationAdd(path_in_repo='chart-foundation-data.zip', path_or_fileobj=str(archive)),
        CommitOperationAdd(path_in_repo='train_chart_foundation.py', path_or_fileobj=str(ROOT / 'scripts/train_chart_foundation.py')),
    ])
    runner = '''import hashlib, json, os, pathlib, subprocess, sys, zipfile
from huggingface_hub import HfApi, hf_hub_download
root = pathlib.Path('/tmp/chart-run'); root.mkdir(exist_ok=True)
data = root / 'data'; data.mkdir(exist_ok=True)
archive = pathlib.Path(hf_hub_download(os.environ['DATA_REPO'], 'chart-foundation-data.zip', repo_type='dataset', revision=os.environ['DATA_REVISION']))
with archive.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == os.environ['DATA_SHA256']
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        assert (data/name).resolve().is_relative_to(data.resolve())
    z.extractall(data)
script = hf_hub_download(os.environ['DATA_REPO'], 'train_chart_foundation.py', repo_type='dataset', revision=os.environ['DATA_REVISION'])
out = root / 'output'; out.mkdir(exist_ok=True)
(out/'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
cmd = [sys.executable, '-u', script, '--data', str(data), '--output', str(out), '--hub-repo', os.environ['MODEL_REPO'], '--profile', os.environ['MODEL_PROFILE']]
if os.environ['SMOKE'] == '1': cmd.append('--smoke')
result = subprocess.run(cmd)
(out/'run-status.json').write_text(json.dumps({'returncode': result.returncode, 'smoke_only': os.environ['SMOKE']=='1', 'data_revision':os.environ['DATA_REVISION']}))
HfApi().upload_folder(repo_id=os.environ['MODEL_REPO'], folder_path=str(out), commit_message='Save training job artifacts and status')
if result.returncode == 0:
    HfApi().upload_folder(repo_id=os.environ['MODEL_REPO'], folder_path=str(out/'adapter'), commit_message='Make completed adapter loadable from repository root')
sys.exit(result.returncode)
'''
    bootstrap = "import subprocess,sys; subprocess.run([sys.executable,'-m','pip','install','transformers==5.16.1','peft==0.20.0','accelerate==1.14.0','bitsandbytes>=0.46,<1','pillow==12.3.0','huggingface_hub==1.30.0'],check=True); exec(" + repr(runner) + ")"
    job = api.run_job(image='pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel', command=['python','-u','-c',bootstrap],
        flavor='a100-large', timeout=args.timeout, namespace=owner,
        name='clawd-chart-foundation-' + args.profile + '-' + ('full' if args.full else 'smoke'),
        env={'DATA_REPO':data_repo, 'DATA_REVISION':commit.oid, 'DATA_SHA256':digest, 'MODEL_REPO':model_repo,
             'MODEL_PROFILE':args.profile, 'SMOKE':'0' if args.full else '1','PYTHONUNBUFFERED':'1',
             'PIP_BREAK_SYSTEM_PACKAGES':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1'}, secrets={'HF_TOKEN':get_token()})
    record = {'job_id':job.id, 'job_url':f'https://huggingface.co/jobs/{owner}/{job.id}', 'data_repo':data_repo,
              'data_revision':commit.oid,'profile':args.profile,'model_repo':model_repo,'timeout':args.timeout,'smoke_only':not args.full}
    (ROOT / 'outputs/hf-chart-job.json').write_text(json.dumps(record,indent=2)+'\n')
    (ROOT / f'outputs/hf-chart-job-{job.id}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__ == '__main__':
    main()
