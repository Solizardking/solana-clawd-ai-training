#!/usr/bin/env python3
"""Stage or deploy a private HF chart Space. No raw training corpus is uploaded."""
import argparse
import json
import os
import secrets
import shutil
from pathlib import Path
from huggingface_hub import HfApi, get_token

ROOT=Path(__file__).resolve().parents[1]
REPO='ordlibrary/clawd-spark-chart-agent'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deploy',action='store_true')
    p.add_argument('--include-chartdete',action='store_true',help='Only after specific detector-upload authorization')
    args=p.parse_args()
    stage=ROOT/'outputs/spark-space-package';stage.mkdir(exist_ok=True)
    for directory in ['chart_agent','docs','outputs/chart-agent/assets/weights']:(stage/directory).mkdir(parents=True,exist_ok=True)
    for f in (ROOT/'chart_agent').iterdir():
        if f.suffix in ('.py','.html'):shutil.copy2(f,stage/'chart_agent'/f.name)
    shutil.copy2(ROOT/'deploy/chart-agent/spark/Dockerfile',stage/'Dockerfile')
    shutil.copy2(ROOT/'docs/solgpt-chart-tool-catalog.md',stage/'docs/solgpt-chart-tool-catalog.md')
    for name in ['assets/weights/best.onnx','assets/chart-pattern.onnx','research.sqlite','examples.sqlite']:
        target=stage/'outputs/chart-agent'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'outputs/chart-agent'/name,target)
    if args.include_chartdete:
        target=stage/'outputs/chart-agent/chartdete';target.mkdir(exist_ok=True)
        for name in ['chart-elements.onnx','report.json']:
            shutil.copy2(ROOT/'outputs/chart-agent/chartdete/training/full/evaluation'/name,target/name)
    (stage/'README.md').write_text('---\ntitle: Clawd Spark Chart Agent\nemoji: 🦞\ncolorFrom: green\ncolorTo: blue\nsdk: docker\napp_port: 7860\n---\nPrivate Spark chart research service. Authenticated API and model endpoints. Detector labels are observations, not trading instructions.\n')
    print('Staged private Space sources, existing detectors, and research/example indexes.')
    if not args.deploy:return
    api=HfApi();assert api.whoami()['name']=='ordlibrary'
    secret_path=ROOT/'outputs/spark-hosting-secrets.json'
    if secret_path.exists():keys=json.loads(secret_path.read_text())
    else:
        keys={k:secrets.token_urlsafe(36) for k in ['CHART_API_KEY','CHART_MODEL_API_KEY']}
        fd=os.open(secret_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:json.dump(keys,f)
    space_secrets=[{'key':k,'value':v} for k,v in dict(keys,HF_TOKEN=get_token()).items()]
    api.create_repo(REPO,repo_type='space',space_sdk='docker',private=True,exist_ok=True,space_secrets=space_secrets)
    assert api.repo_info(REPO,repo_type='space').private
    # Updating secrets also handles a resumed deployment into the existing Space.
    for secret in space_secrets:api.add_space_secret(REPO,**secret)
    api.add_space_variable(REPO,key='SPARK_CONTEXT_SIZE',value='8192')
    if args.include_chartdete:api.add_space_variable(REPO,key='CHART_ELEMENT_DETECTOR',value='/app/outputs/chart-agent/chartdete/chart-elements.onnx')
    commit=api.upload_folder(repo_id=REPO,repo_type='space',folder_path=stage,commit_message='Deploy authenticated Spark chart stack and Orin-compatible API')
    result={'space':REPO,'revision':commit.oid,'url':'https://ordlibrary-clawd-spark-chart-agent.hf.space','chartdete_included':args.include_chartdete,'gpu_activation':'pending verified adapter'}
    (ROOT/'outputs/spark-space-deployment.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
