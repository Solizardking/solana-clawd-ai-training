#!/usr/bin/env python3
"""Extract chart evidence locally; candidates require answer-grounding review before training."""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chart_agent.detector import Detector, decode_image
from chart_agent.ocr import extract_text


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'outputs/nemotron-chart-data')
    p.add_argument('--images-root', type=Path, default=ROOT/'outputs/chart-foundation-data')
    p.add_argument('--detector', type=Path, default=ROOT/'outputs/chart-agent/chartdete/training/full/evaluation/chart-elements.onnx')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/nemotron-visual-evidence')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.output.exists() and not args.resume:
        p.error('Use a fresh output directory; previous evidence is preserved')
    source = args.data/'train-visual-pending.jsonl'
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    root = args.images_root.resolve()
    image_paths = sorted({path for row in rows for path in row['images']})
    for name in image_paths:
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Image missing or outside root: {name}')
    detector = Detector(args.detector)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output/'images').mkdir(exist_ok=args.resume)
    config = {'source_sha256':sha(raw), 'detector_sha256':sha(args.detector.read_bytes()),
              'ocr_script_sha256':sha((ROOT/'chart_agent/ocr.py').read_bytes()),
              'image_count':len(image_paths), 'row_count':len(rows),
              'training_ready':False, 'status':'extracting'}
    if args.resume:
        previous = json.loads((args.output/'manifest.json').read_text())
        for key in ['source_sha256','detector_sha256','ocr_script_sha256']:
            if previous[key] != config[key]:
                raise ValueError(f'Resume provenance changed: {key}')
    (args.output/'manifest.json').write_text(json.dumps(config,indent=2))
    def extract(name):
        blob = (root/name).read_bytes()
        target = 'images/'+sha(name.encode())+'.json'
        if args.resume and (args.output/target).exists():
            cached = json.loads((args.output/target).read_text())
            if cached['image'] != name or cached['image_sha256'] != sha(blob):
                raise ValueError('Cached image provenance differs')
            return name,target,cached
        evidence = {'image':name,'image_sha256':sha(blob),
                    'raw_image_visible_to_nemotron':False}
        try:
            image = decode_image(blob)
        except (OSError, ValueError) as exc:
            evidence.update(image_usable=False, error=type(exc).__name__,
                            ocr={'available':False,'words':[]},chart_elements=[])
        else:
            evidence.update(image_usable=True,size=list(image.size),
                            ocr=extract_text(image),chart_elements=detector.detect(image))
        (args.output/target).write_text(json.dumps(evidence,ensure_ascii=False))
        return name,target,evidence
    by_image = {}
    failed_ocr = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index,(name,target,evidence) in enumerate(pool.map(extract,image_paths),1):
            by_image[name] = {'path':target,'evidence':evidence}
            failed_ocr += not evidence['ocr']['available']
            if index % 50 == 0 or index == len(image_paths):
                print(json.dumps({'images_completed':index,'total':len(image_paths),'ocr_unavailable':failed_ocr}),flush=True)
    # Preserve original targets separately; OCR presence alone cannot prove an answer follows.
    with (args.output/'review-candidates.jsonl').open('w') as stream:
        for row in rows:
            candidate = {'id':row['id'],'group':row.get('group'),'split':row.get('split'),
                         'sources':row.get('sources'),'original_messages':row['messages'],
                         'observations':[by_image[name]['path'] for name in row['images']],
                         'training_ready':False,
                         'review_required':'Verify the target is supported by OCR and geometry; rewrite or exclude unsupported targets.'}
            stream.write(json.dumps(candidate,ensure_ascii=False)+'\n')
    config.update(status='extraction_complete',ocr_unavailable=failed_ocr,
                  unusable_images=[name for name,item in by_image.items()
                                   if item['evidence'].get('image_usable') is False],
                  candidates_sha256=sha((args.output/'review-candidates.jsonl').read_bytes()))
    (args.output/'manifest.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(config),flush=True)


if __name__ == '__main__':
    main()
