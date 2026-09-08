#!/usr/bin/env python3
"""Bounded Spark LoRA pilot with assistant-only labels and post-training diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, get_token
import hf_spark_smoke
import hf_nemotron_chart_pilot as data

ROOT = Path(__file__).resolve().parents[1]
NAME = 'clawd-spark-4b-pilot'
REPO = 'ordlibrary/clawd-spark-4b-lora-pilot'


def corrections():
    rows = []
    subjects = ['SPL Token mint account', 'Solana token Mint struct', 'Token-2022 base mint state', 'Solana mint account']
    questions = [
        ('Name the field that stores the number of fractional digits in a {s}. Reply with JSON using key name.', {'name':'decimals'}),
        ('Which member of a {s} specifies base-ten display precision? Reply with JSON using key member.', {'member':'decimals'}),
        ('Is decimal_precision an actual base field of a {s}? Give JSON with keys valid and correct_field.', {'valid':False,'correct_field':'decimals'}),
        ('When implementing a token balance formatter, which member do I read from the {s}? Return JSON with key field.', {'field':'decimals'}),
        ('What exact field identifier sets token unit scaling in a {s}? Return JSON with key field.', {'field':'decimals'}),
        ('Identify the member controlling fractional places in the {s}. Return JSON with key field.', {'field':'decimals'}),
        ('Which field of a {s} determines the power of ten used for display amounts? Return JSON with key field.', {'field':'decimals'}),
        ('Explain the decimals field in a {s}.', 'The decimals field stores the number of base-ten digits to the right of the decimal point. Read it from the mint account; do not infer it from the token name.'),
    ]
    for subject in subjects:
        for question,answer in questions:
            rows.append({'messages':[{'role':'user','content':question.format(s=subject)}, {'role':'assistant','content':json.dumps(answer) if isinstance(answer,dict) else answer}], 'tools':'[]'})
    return rows


def payload():
    source = hf_spark_smoke.payload().replace("'sentencepiece==0.2.1'", "'sentencepiece==0.2.1','peft==0.17.1'")
    insertion = r'''
import os,random,zipfile,copy
from huggingface_hub import hf_hub_download,HfApi
from peft import LoraConfig,get_peft_model,PeftModel
ns={'__name__':'quality_before'};exec(QUALITY_TEXT,ns)
before=ns['evaluate'](post,'spark-base')
print('SPARK_BEFORE '+json.dumps(before),flush=True)
work=Path('/tmp/spark-pilot');work.mkdir(exist_ok=True)
raw=work/'raw';raw.mkdir(exist_ok=True)
archive=Path(hf_hub_download(DATA_REPO,'chart-foundation-data.zip',repo_type='dataset',revision=DATA_REVISION))
with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==ARCHIVE_SHA
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():assert (raw/name).resolve().is_relative_to(raw.resolve())
 z.extractall(raw)
prep={'__name__':'preparation'};exec(PREPARATION,prep)
ready=work/'prepared';prep['prepare'](raw,ready)
with (ready/'train.jsonl').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==TRAIN_SHA

def encode(row):
 messages=copy.deepcopy(row['messages'])
 if not messages or messages[-1]['role']!='assistant':return None
 for m in messages:
  for call in m.get('tool_calls') or []:
   a=call['function']['arguments']
   if isinstance(a,str):call['function']['arguments']=json.loads(a)
 tools=row.get('tools') or []
 if isinstance(tools,str):tools=json.loads(tools)
 prefix=tokenizer.apply_chat_template(messages[:-1],tools=tools or None,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 full=tokenizer.apply_chat_template(messages,tools=tools or None,tokenize=False,add_generation_prompt=False,enable_thinking=False)
 if not full.startswith(prefix):return None
 ids=tokenizer.encode(full,add_special_tokens=False)
 pids=tokenizer.encode(prefix,add_special_tokens=False)
 if ids[:len(pids)]!=pids or len(ids)>1024 or len(ids)<=len(pids):return None
 labels=[-100]*len(pids)+ids[len(pids):]
 assert labels[-1]==tokenizer.eos_token_id and any(x!=-100 for x in labels)
 return {'input_ids':torch.tensor([ids],device=model.device),'labels':torch.tensor([labels],device=model.device)}

random.seed(1111);torch.manual_seed(1111)
rows=[json.loads(s) for s in (ready/'train.jsonl').read_text().splitlines()]
order=list(range(len(rows)));random.shuffle(order)
selected=[];indices=[]
for i in order:
 if any(m.get('content')==case[1] for m in rows[i]['messages'] for case in ns['CASES']):continue
 item=encode(rows[i])
 if item is not None:
  selected.append(item);indices.append(i)
 if len(selected)==192:break
assert len(selected)==192
fixes=[encode(r) for r in CORRECTIONS]
assert all(x is not None for x in fixes)
selected+=fixes*2
random.shuffle(selected)
assert len(selected)==256
model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,target_modules=['q_k_v_proj','out_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
model.config.use_cache=False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
model.train();started=time.monotonic();losses=[]
for step in range(128):
 optimizer.zero_grad(set_to_none=True);step_loss=0.0
 for batch in selected[(step%64)*4:(step%64)*4+4]:
  loss=model(**batch,use_cache=False).loss
  assert torch.isfinite(loss),'Nonfinite training loss'
  (loss/4).backward();step_loss+=loss.item()/4
 torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
 optimizer.step();losses.append(step_loss)
 if step%8==0 or step==127:print('SPARK_TRAIN '+json.dumps({'step':step+1,'loss':step_loss,'seconds':time.monotonic()-started}),flush=True)
 if time.monotonic()-started>1800:raise RuntimeError('Training time budget exceeded')
out=work/'adapter';out.mkdir(exist_ok=True)
model.save_pretrained(out,safe_serialization=True)
(out/'README.md').write_text('---\nbase_model: XHToken/Spark-X2.5-4B\nlibrary_name: peft\nlicense: apache-2.0\n---\n# Clawd Spark corrective pilot\nBounded Solana training pilot; see training.json and evaluation.json for scope and results.\n')
adapter_config=json.loads((out/'adapter_config.json').read_text())
adapter_config['base_model_name_or_path']='XHToken/Spark-X2.5-4B'
adapter_config['revision']='5e10fcc0286756aebf7c41dc52c1e42d95c70281'
(out/'adapter_config.json').write_text(json.dumps(adapter_config,indent=2))
tokenizer.save_pretrained(out)
metadata={'base_model':MODEL_ID,'base_revision':MODEL_REV,'data_repo':DATA_REPO,'data_revision':DATA_REVISION,'data_sha256':ARCHIVE_SHA,'selection_indices':indices,'correction_source':'https://solana.com/docs/tokens/basics/create-mint','correction_unique_rows':len(CORRECTIONS),'correction_presentations':128,'epochs':2,'existing_rows':192,'steps':128,'max_tokens':1024,'assistant_only_final_turn':True,'seed':1111,'losses':losses,'training_seconds':time.monotonic()-started,'promotion':'pending_evaluation','visual_addon_rows':0}
(out/'training.json').write_text(json.dumps(metadata,indent=2))
# Reattach from saved bytes to verify the artifact, rather than only in-memory weights.
base=model.unload();del model,optimizer,selected,fixes;torch.cuda.empty_cache()
model=PeftModel.from_pretrained(base,out).eval()
model.gradient_checkpointing_disable();model.config.use_cache=True
print('SPARK_ADAPTER_RELOADED',flush=True)
'''
    values={'QUALITY_TEXT':(ROOT/'scripts/evaluate_nemotron_quality.py').read_text(), 'DATA_REPO':data.DATA_REPO, 'DATA_REVISION':data.DATA_REVISION,'ARCHIVE_SHA':data.ARCHIVE_SHA,'PREPARATION':(ROOT/'scripts/prepare_nemotron_chart_data.py').read_text(),'TRAIN_SHA':json.loads((ROOT/'outputs/nemotron-chart-data/manifest.json').read_text())['outputs_sha256']['train.jsonl'],'CORRECTIONS':corrections(),'MODEL_ID':hf_spark_smoke.MODEL,'MODEL_REV':hf_spark_smoke.REVISION}
    for k,v in values.items():insertion=insertion.replace(k,repr(v))
    source=source.replace("'training':False", "'training':True")
    source=source.replace("namespace={'__name__':'quality'}",insertion+"\nnamespace={'__name__':'quality'}")
    source += '\n' + r'''
heldout=[]
for prompt in ['What member of an SPL Mint stores the number of digits after the decimal point? Return JSON with key field.', 'Name the SPL mint property that a token display formatter uses for its divisor. Return JSON with key field.', 'In the on-chain Mint structure, which member holds base-10 fractional precision? Return JSON with key field.', 'A wallet SDK needs the Mint member controlling fractional token digits. Return JSON with key field.', 'For a Token-2022 mint, give the base-state property used to scale raw units. Return JSON with key field.', 'A developer wrote mint.decimal_precision. What real member should replace decimal_precision? Return JSON with key field.']:
 result=post('',{'messages':[{'role':'user','content':prompt}],'max_tokens':128})
 heldout.append({'prompt':prompt,'response':result,'passed':namespace['score_response'](result,{'field':'decimals'})})
artifact=out/'adapter_model.safetensors'
from safetensors.torch import load_file
weights=load_file(str(artifact));assert weights and all(torch.isfinite(w).all() for w in weights.values())
assert any(torch.count_nonzero(w)>0 for k,w in weights.items() if 'lora_B' in k)
result={'before':before,'after':report,'heldout_mint':heldout,'native_tool_request_passed':valid,'roundtrips':roundtrips,'adapter_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'promotion_gate_passed':report['passed']==report['total'] and valid and all(x['passed'] for x in heldout) and all(roundtrips.values()),'scope':'Targeted pilot; known regression suite is development evaluation, six paraphrases include three development checks and three new heldout checks; no production deployment'}
(out/'evaluation.json').write_text(json.dumps(result,indent=2))
api=HfApi()
assert api.repo_info(OUTPUT_REPO).private
commit=api.upload_folder(repo_id=OUTPUT_REPO,folder_path=str(out),commit_message='Save Spark 128-step Solana corrective pilot and reload evaluation')
print('SPARK_PILOT_RESULT '+json.dumps({'repo':OUTPUT_REPO,'revision':commit.oid,**result}),flush=True)
'''.replace('OUTPUT_REPO',repr(REPO))
    compile(source,'spark-pilot','exec')
    return source


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    source=payload()
    report={'name':NAME,'model':hf_spark_smoke.MODEL,'revision':hf_spark_smoke.REVISION,'flavor':'a10g-small','timeout':'1h','steps':128,'existing_rows':192,'corrective_presentations':128,'epochs':2,'private_output':REPO,'runner_sha256':hashlib.sha256(source.encode()).hexdigest()}
    if args.launch:
        api=HfApi();assert api.whoami()['name']=='ordlibrary'
        for job in api.list_jobs(namespace='ordlibrary'):
            if getattr(job,'name',None)==NAME and job.status.stage not in {'COMPLETED','ERROR','CANCELED','DELETED'}:raise RuntimeError('Pilot already active')
        api.create_repo(REPO,private=True,exist_ok=True);assert api.repo_info(REPO).private
        job=api.run_job(image='pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime',command=['python','-u','-c',source],flavor=report['flavor'],timeout=report['timeout'],namespace='ordlibrary',name=NAME,env={'HF_HUB_DISABLE_PROGRESS_BARS':'1'},secrets={'HF_TOKEN':get_token()})
        report.update(job_id=job.id,job_url=f'https://huggingface.co/jobs/ordlibrary/{job.id}')
        (ROOT/f'outputs/hf-spark-pilot-{job.id}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
