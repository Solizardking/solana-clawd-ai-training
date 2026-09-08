#!/usr/bin/env python3
"""CPU-only native NeMo ingestion audit using the existing private training archive."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, get_token
from hf_nemotron_chart_pilot import payload, DATA_REPO, DATA_REVISION, ARCHIVE_SHA, MODEL_REPO

NAME='clawd-nemotron-native-data-audit'


def runner():
    setup=payload().split('selection={}')[0]
    template=(Path(__file__).resolve().parents[1]/'deploy/nemotron/training/assistant-mask.jinja').read_text()
    setup+='\ntokenizer.chat_template='+repr(template)+'\n'
    audit=r'''
from nemo_automodel.components.datasets.llm.agent_chat import make_agent_chat_dataset
reports={}
for split in ['train','validation','test']:
 path=ready/(split+'.jsonl')
 rows=[json.loads(line) for line in path.read_text().splitlines()]
 ds=make_agent_chat_dataset(tokenizer=tokenizer,path=str(path),seq_length=16384,truncation=True,truncate_history=True,drop_history_reasoning_content=True)
 stats={'input_rows':len(rows),'native_rows':len(ds),'empty_supervision':0,'over_context':0,'max_tokens':0,'total_tokens':0,'supervised_tokens':0,'final_answer_text_missing':0,'missing_example_ids':[]}
 assert len(ds)==len(rows),'Native adapter changed dataset cardinality'
 for row_index,(row,item) in enumerate(zip(rows,ds)):
  labels=list(item['labels']);ids=list(item['input_ids'])
  assert len(labels)==len(ids)
  active=[int(x) for x in labels if x!=-100]
  # NeMo packages input_ids[:-1] and masked labels[1:] for next-token prediction.
  assert all(int(label)==int(token) for label,token in zip(labels[:-1],ids[1:]) if label!=-100),'Unexpected next-token label alignment'
  stats['empty_supervision']+=not bool(active)
  stats['over_context']+=len(ids)>16384
  stats['max_tokens']=max(stats['max_tokens'],len(ids))
  stats['total_tokens']+=len(ids);stats['supervised_tokens']+=len(active)
  target=next(m['content'] for m in reversed(row['messages']) if m['role']=='assistant')
  if target.strip() not in tokenizer.decode(active):
   stats['final_answer_text_missing']+=1
   if len(stats['missing_example_ids'])<20:stats['missing_example_ids'].append(row.get('id',row_index))
 reports[split]=stats
 print('NEMOTRON_NATIVE_SPLIT '+json.dumps({'split':split,**stats}),flush=True)
fixture={'tools':'[]','messages':[{'role':'system','content':'SYSTEM_SENTINEL_6ef9','reasoning_content':''},{'role':'user','content':'USER_SENTINEL_c453','reasoning_content':''},{'role':'assistant','content':'ANSWER_SENTINEL_a103','reasoning_content':''},{'role':'user','content':'SECOND_USER_SENTINEL_95ae','reasoning_content':''},{'role':'assistant','content':'FINAL_ANSWER_SENTINEL_192f','reasoning_content':''}]}
path=root/'mask-fixture.jsonl';path.write_text(json.dumps(fixture)+'\n')
item=make_agent_chat_dataset(tokenizer=tokenizer,path=str(path),seq_length=16384,truncation=True,truncate_history=True,drop_history_reasoning_content=True)[0]
active=[int(x) for x in item['labels'] if x!=-100];decoded=tokenizer.decode(active)
mask={'system_excluded':'SYSTEM_SENTINEL_6ef9' not in decoded,'users_excluded':all(s not in decoded for s in ['USER_SENTINEL_c453','SECOND_USER_SENTINEL_95ae']),'both_assistants_included':all(s in decoded for s in ['ANSWER_SENTINEL_a103','FINAL_ANSWER_SENTINEL_192f']),'eos_supervised':tokenizer.eos_token_id in active}
report={'data_revision':os.environ['DATA_REVISION'],'splits':reports,'mask_fixture':mask,'weights_loaded':False,'uploads':False}
print('NEMOTRON_NATIVE_AUDIT '+json.dumps(report),flush=True)
assert all(mask.values()),'Assistant-only mask fixture failed'
assert all(s['empty_supervision']==0 and s['over_context']==0 and s['final_answer_text_missing']==0 for s in reports.values()),'Inspect native formatting discrepancies'
'''
    result=setup+audit
    compile(result,'native-data-audit','exec')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--launch',action='store_true');args=p.parse_args()
    source=runner()
    record={'name':NAME,'image':'nvcr.io/nvidia/nemo-automodel:26.08','flavor':'cpu-basic','timeout':'1h','weights_loaded':False,'uploads':False,'runner_sha256':hashlib.sha256(source.encode()).hexdigest()}
    if args.launch:
        api=HfApi();owner=api.whoami()['name']
        if owner!='ordlibrary':raise RuntimeError('Expected ordlibrary account')
        for job in api.list_jobs(namespace=owner):
            if getattr(job,'name',None)==NAME and job.status.stage not in {'COMPLETED','ERROR','CANCELED','DELETED'}:raise RuntimeError('Audit already active')
        job=api.run_job(image=record['image'],command=['python3','-u','-c',source],flavor=record['flavor'],timeout=record['timeout'],namespace=owner,name=NAME,env={'DATA_REPO':DATA_REPO,'DATA_REVISION':DATA_REVISION,'ARCHIVE_SHA':ARCHIVE_SHA,'MODEL_REPO':MODEL_REPO,'HF_HUB_DISABLE_PROGRESS_BARS':'1','PYTHONUNBUFFERED':'1'},secrets={'HF_TOKEN':get_token()})
        record.update(job_id=job.id,job_url=f'https://huggingface.co/jobs/{owner}/{job.id}')
        (Path(__file__).resolve().parents[1]/'outputs'/f'hf-nemotron-native-audit-{job.id}.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
