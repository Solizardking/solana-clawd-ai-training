#!/usr/bin/env python3
"""Create a new evidence version for the verified re-download; preserve frozen inputs."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from chart_agent.detector import Detector,decode_image
from chart_agent.ocr import extract_text


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=ROOT/'outputs/nemotron-visual-evidence-upscaled'
    out=ROOT/'outputs/nemotron-visual-evidence-repaired'
    replacement=ROOT/'outputs/chart-agent/source-recheck/hs_0016.png'
    verified=json.loads(replacement.with_name('verification.json').read_text())
    if not verified['valid_image'] or digest(replacement)!=verified['download_sha256']:
        raise ValueError('Re-downloaded image differs from verification')
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['status']!='extraction_complete':raise ValueError('Source extraction incomplete')
    detector_path=ROOT/'outputs/chart-agent/chartdete/training/full/evaluation/chart-elements.onnx'
    if digest(detector_path)!=manifest['detector_sha256']:raise ValueError('Detector changed')
    if digest(ROOT/'chart_agent/ocr.py')!=manifest['ocr_script_sha256']:raise ValueError('OCR changed')
    image=decode_image(replacement.read_bytes())
    name='images/hs_0016.png'
    evidence={'image':name,'image_sha256':digest(replacement),'image_usable':True,
              'size':list(image.size),'ocr':extract_text(image),'chart_elements':Detector(detector_path).detect(image),
              'raw_image_visible_to_nemotron':False,'source_override':verified}
    if not evidence['ocr']['available']:raise ValueError('Replacement OCR unavailable')
    shutil.copytree(source,out)  # Refuses existing output.
    target=out/'images'/(hashlib.sha256(name.encode()).hexdigest()+'.json')
    target.write_text(json.dumps(evidence,ensure_ascii=False))
    files=sorted((out/'images').glob('*.json'))
    observations=[json.loads(p.read_text()) for p in files]
    manifest.update(parent_evidence=str(source.relative_to(ROOT)),parent_manifest_sha256=digest(source/'manifest.json'),
                    source_image_overrides={name:verified},
                    unusable_images=[r['image'] for r in observations if r.get('image_usable') is False],
                    ocr_unavailable=sum(not r['ocr']['available'] for r in observations))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    receipts={str(p.relative_to(out)):digest(p) for p in files}
    (out/'observation-hashes.json').write_text(json.dumps(receipts,indent=2))
    rows=[json.loads(l) for l in (out/'review-candidates.jsonl').read_text().splitlines()]
    assert len(files)==1712 and len(rows)==3198
    assert all((out/ref).is_file() for r in rows for ref in r['observations'])
    report={'images':len(files),'candidates':len(rows),'all_references_resolve':True,
            'ocr_unavailable':manifest['ocr_unavailable'],'unusable_images':manifest['unusable_images'],
            'training_ready':False,'grounding_review_completed':False,
            'observation_receipts_sha256':digest(out/'observation-hashes.json')}
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
