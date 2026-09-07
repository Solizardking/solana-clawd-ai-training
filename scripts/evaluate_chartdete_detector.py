#!/usr/bin/env python3
"""Evaluate a completed ChartDete run once on the reserved test split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'outputs/chart-agent/yolo-config'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'outputs/chart-agent/chartdete/training')
    p.add_argument('--data',type=Path,default=ROOT/'outputs/chart-agent/chartdete/prepared/data.yaml')
    p.add_argument('--device',default='cpu')
    p.add_argument('--smoke',action='store_true',help='Validate only four smoke validation images; never touch test')
    args=p.parse_args()
    name='smoke' if args.smoke else 'full'
    result=json.loads((args.run/f'{name}-result.json').read_text())
    if not result['training_finished'] or result['smoke_only'] != args.smoke:
        raise ValueError('A matching completed training run is required')
    checkpoint=Path(result['best_checkpoint']).resolve()
    if checkpoint != (args.run/name/'weights/best.pt').resolve():
        raise ValueError('Unexpected checkpoint outside the selected run')
    provenance=json.loads((args.run/f'{name}-provenance.json').read_text())
    manifest=json.loads((args.data.parent/'manifest.json').read_text())
    if manifest != provenance['data_manifest']:
        raise ValueError('Dataset differs from training provenance')
    out=args.run/name/'evaluation'
    out.mkdir(parents=True,exist_ok=True)
    report_path=out/'report.json'
    if report_path.exists(): raise ValueError('Evaluation already recorded; preserve it rather than rerunning test')
    onnx_path=out/'chart-elements.onnx'
    subprocess.run([sys.executable,str(ROOT/'scripts/export_chart_pattern_detector.py'),
        '--checkpoint',str(checkpoint),'--output',str(onnx_path),
        '--description','ChartDete 18-class detector; see adjacent evaluation report before activation'],check=True)
    from ultralytics import YOLO
    model=YOLO(str(checkpoint))
    expected={int(k):v for k,v in json.loads(args.data.read_text())['names'].items()}
    if model.names != expected: raise ValueError('Checkpoint class mapping differs from dataset')
    metrics=model.val(data=str(args.run/'smoke-data.yaml' if args.smoke else args.data),
        split='val' if args.smoke else 'test', device=args.device, imgsz=640 if args.smoke else 960,
        batch=2,workers=0,plots=False,save_json=True,project=str(out),name='metrics',exist_ok=False)
    per_class=[]
    for i,c in enumerate(metrics.box.ap_class_index):
        precision,recall,ap50,ap=metrics.box.class_result(i)
        per_class.append({'class_id':int(c),'name':model.names[int(c)],'precision':float(precision),
            'recall':float(recall),'AP50':float(ap50),'AP50_95':float(ap)})
    report={'smoke_only':args.smoke,'split':'smoke_validation' if args.smoke else 'test',
        'checkpoint_sha256':hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),
        'dataset_archive_sha256':manifest['archive_sha256'],'metrics':metrics.results_dict,
        'per_class':per_class,'export_verification':json.loads(onnx_path.with_suffix('.verification.json').read_text()),
        'served':False,'activation_approved':False}
    report_path.write_text(json.dumps(report,indent=2))
    print('Evaluation saved:',report_path)

if __name__=='__main__': main()
