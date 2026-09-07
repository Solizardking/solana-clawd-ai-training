#!/usr/bin/env python3
"""Prepare a reviewable detector checkpoint backup locally; performs no uploads."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'outputs/chart-agent/chartdete/training'
metrics=(run/'full/results.csv').read_bytes()
rows=list(csv.DictReader(metrics.decode().splitlines()))
epoch=int(rows[-1]['epoch'])
checkpoint=run/f'full/weights/epoch{epoch-1}.pt'
with zipfile.ZipFile(checkpoint) as z:
    if z.testzip() is not None:raise ValueError('Incomplete checkpoint')
output=run/f'backup-package-epoch-{epoch}'
output.mkdir(exist_ok=True)
files={checkpoint.name:checkpoint,'source-manifest.json':run.parent/'prepared/manifest.json',
       'training-provenance.json':run/'full-provenance.json',
       'train_chartdete_detector.py':ROOT/'scripts/train_chartdete_detector.py'}
for name,source in files.items():shutil.copyfile(source,output/name)
(output/'results.csv').write_bytes(metrics)
(output/'README.md').write_text('# Clawd Chart Elements YOLO11n checkpoint backup\n\n'
    f'Completed epoch {epoch}; training is unfinished. Not approved for serving.\n'
    '18 chart-element classes from https://github.com/pengyu965/ChartDete.\n'
    'Validation metrics are not held-out test results or trading accuracy.\n'
    'The checkpoint contains optimizer and EMA state for recovery.\n')
manifest={'completed_epoch':epoch,'proposed_private_repository':'ordlibrary/clawd-chart-elements-yolo11n',
          'uploaded':False,'test_evaluated':False,'files':[]}
for path in sorted(output.iterdir()):
    if path.name=='package-manifest.json':continue
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    manifest['files'].append({'name':path.name,'bytes':path.stat().st_size,'sha256':digest})
manifest['total_payload_bytes']=sum(x['bytes'] for x in manifest['files'])
(output/'package-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
print('Local package:',output)
