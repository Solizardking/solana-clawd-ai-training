#!/usr/bin/env python3
"""Run the existing Clawd agent through the local chart gateway; observer by default."""
import os
import runpy
import sys
from pathlib import Path

root=Path.home()/'clawd-chart'
config={}
for line in (Path.home()/'.config/clawd-chart/gateway.env').read_text().splitlines():
    key,value=line.split('=',1);config[key]=value
os.environ.update(CLAWD_API_KEY=config['CHART_API_KEY'],CLAWD_INFERENCE_URL='http://127.0.0.1:8092/model/v1',CLAWD_MODEL='clawd-spark',CLAWD_CHART_URL='http://127.0.0.1:8092')
sys.argv[0]=str(root/'nvidia/nemotron_ultra_agent.py')
if '--mode' not in sys.argv and not any(x.startswith('--mode=') for x in sys.argv):sys.argv+=['--mode','observer']
runpy.run_path(sys.argv[0],run_name='__main__')
