#!/usr/bin/env python3
"""Bounded local training monitor; no uploads, new training jobs, or model activation."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from huggingface_hub import HfApi

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/chart-agent/chartdete/training'
STATE=ROOT/'outputs/chart-agent/training-watch.json'


def atomic_write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2))
    tmp.replace(path)


def tick(pid,record,auto_evaluate):
    state={'checked_at':datetime.now(timezone.utc).isoformat(),'watcher_pid':os.getpid()}
    process=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    active=process.returncode==0 and 'scripts/train_chartdete_detector.py' in process.stdout
    rows=list(csv.DictReader((RUN/'full/results.csv').read_text().splitlines()))
    completed=RUN/'full-result.json'
    valid_completed=False
    if completed.exists():
        result=json.loads(completed.read_text())
        valid_completed=result.get('training_finished') is True and result.get('smoke_only') is False
    state['detector']={'pid':pid,'process_active':active,'training_complete':valid_completed,
                       'last_completed_epoch':rows[-1] if rows else None}
    report=RUN/'full/evaluation/report.json'
    attempt=RUN/'full/evaluation/monitor-attempt.json'
    if valid_completed and not active and auto_evaluate and not report.exists() and not attempt.exists():
        attempt.parent.mkdir(parents=True,exist_ok=True)
        atomic_write(attempt,{'started_at':state['checked_at'],'status':'running'})
        state['detector']['evaluation']='running'
        atomic_write(STATE,state)
        with (attempt.parent/'monitor-console.log').open('w') as log:
            evaluation=subprocess.run([str(ROOT/'.venv-detector/bin/python'),'-u',str(ROOT/'scripts/evaluate_chartdete_detector.py'),'--device','mps'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        atomic_write(attempt,{'finished_at':datetime.now(timezone.utc).isoformat(),'returncode':evaluation.returncode,
                              'status':'completed' if evaluation.returncode==0 and report.exists() else 'failed_needs_review'})
    state['detector']['evaluation_report_available']=report.exists()
    if not active and not valid_completed:state['detector']['needs_recovery_review']=True
    try:
        job=HfApi().inspect_job(job_id=record['job_id'])
        state['llm']={'job_id':job.id,'stage':job.status.stage,'url':record['job_url']}
    except Exception as exc:
        state['llm']={'job_id':record['job_id'],'observation_error':type(exc).__name__,'terminal_state_not_inferred':True}
    atomic_write(STATE,state)
    print(json.dumps(state),flush=True)
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detector-pid',type=int,required=True)
    p.add_argument('--record',type=Path,default=ROOT/'outputs/hf-chart-full-job-6a9f3645259f8e97255ecdd8.json')
    p.add_argument('--interval',type=int,default=300)
    p.add_argument('--hours',type=float,default=12)
    p.add_argument('--once',action='store_true')
    p.add_argument('--evaluate-on-completion',action='store_true')
    args=p.parse_args()
    if args.interval<60 or not 0<args.hours<=24:p.error('Use interval >=60 seconds and duration <=24 hours')
    record=json.loads(args.record.read_text())
    with (STATE.parent/'training-watch.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another training monitor holds the lock')
        deadline=time.monotonic()+args.hours*3600
        while True:
            try:
                state=tick(args.detector_pid,record,args.evaluate_on_completion)
                terminal=state['llm'].get('stage') in {'COMPLETED','ERROR','CANCELED','TIMEOUT','DELETED'}
                if args.once or (not state['detector']['process_active'] and terminal):break
            except Exception as exc:
                # A partial CSV write or observation failure is not training failure.
                error={'checked_at':datetime.now(timezone.utc).isoformat(),'watcher_pid':os.getpid(),
                       'observation_error':type(exc).__name__,'terminal_state_not_inferred':True}
                atomic_write(STATE,error)
                print(json.dumps(error),flush=True)
                if args.once:raise
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            time.sleep(min(args.interval,remaining))

if __name__=='__main__':main()
