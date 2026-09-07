#!/usr/bin/env python3
"""Train an 18-class chart-element detector; reserve upstream test for final evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('outputs/chart-agent/chartdete/prepared/data.yaml'))
    p.add_argument('--output', type=Path, default=Path('outputs/chart-agent/chartdete/training'))
    p.add_argument('--device', default='mps')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; run with GPU access or specify another device')
    root = args.data.resolve().parent
    config = json.loads(args.data.read_text())
    if len(config['names']) != 18: raise ValueError('Expected 18 ChartDete categories')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = args.data.resolve()
    if args.smoke:
        if args.resume: raise ValueError('Smoke must not resume a full run')
        for split, count in [('train',8),('val',4)]:
            images=sorted((root/config[split]).glob('*.jpg'))[:count]
            if len(images)!=count: raise ValueError('Missing prepared images')
            listing=output/f'smoke-{split}.txt'
            listing.write_text('\n'.join(str(x) for x in images)+'\n')
            config[split]=str(listing)
        config.pop('test',None)
        data=output/'smoke-data.yaml'
        data.write_text(json.dumps(config,indent=2))
    checkpoint = args.resume or output/'yolo11n.pt'
    if not args.resume and not checkpoint.exists():
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(str(checkpoint))
    digest=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()
    name='smoke' if args.smoke else 'full'
    if not args.resume and (output/name).exists():
        raise ValueError('Existing run directory; inspect it and resume explicitly')
    (output/f'{name}-provenance.json').write_text(json.dumps({'initial_checkpoint_sha256':digest,
        'data_manifest':json.loads((root/'manifest.json').read_text()), 'smoke_only':args.smoke,
        'device':args.device,'epochs':1 if args.smoke else args.epochs},indent=2))
    model=YOLO(str(checkpoint))
    model.train(data=str(data), device=args.device, epochs=1 if args.smoke else args.epochs,
        batch=args.batch, imgsz=640 if args.smoke else 960, workers=0, project=str(output),name=name,
        resume=bool(args.resume), patience=8, seed=42, deterministic=True, save=True, save_period=1,
        plots=False, cache=False, amp=False, mosaic=0, mixup=0, flipud=0, fliplr=0,
        degrees=0, shear=0, perspective=0, translate=0.05, scale=0.15,
        hsv_h=0, hsv_s=0, hsv_v=0.15)
    result={'smoke_only':args.smoke,'training_finished':True,'best_checkpoint':str(model.trainer.best),
            'validation':model.trainer.validator.metrics.results_dict,'test_evaluated':False}
    (output/f'{name}-result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
