#!/usr/bin/env python3
"""Build only explicitly reviewed visual-to-text examples with hash-matched evidence."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from chart_agent.server import compact_evidence


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source,decisions,output):
    if output.exists():raise ValueError('Preserve existing prepared output')
    rows={r['id']:r for r in map(json.loads,(source/'review-candidates.jsonl').read_text().splitlines())}
    reviews=[json.loads(line) for line in decisions.read_text().splitlines()]
    if len({r['id'] for r in reviews})!=len(reviews):raise ValueError('Duplicate review decision')
    converted=[];provenance=[]
    for decision in reviews:
        row=rows[decision['id']]
        if decision['status'] not in ('approved','rejected'):raise ValueError('Unknown review status')
        for ref in row['observations']:
            path=(source/ref).resolve()
            if not path.is_relative_to(source.resolve()):raise ValueError('Observation outside source')
            if digest(path)!=decision['observations_sha256'][ref]:raise ValueError('Reviewed observation changed')
        if decision['status']=='rejected':continue
        if row['split']!='train':raise ValueError('This output is training-only; do not move held-out groups')
        if len(row['observations'])!=1:raise ValueError('Multi-image review requires explicit evidence assembly')
        observed=json.loads((source/row['observations'][0]).read_text())
        if not observed['ocr']['available']:raise ValueError('Approved example lacks OCR')
        evidence=compact_evidence({'ocr':observed['ocr'],'chart_elements':{'detections':observed['chart_elements']},
            'bbox_image_size':observed['size'],
            'vision_input':'Raw image is not visible to this text model. Use only detector and OCR observations; OCR can be wrong.'})
        encoded=json.dumps(evidence)
        if any(text not in encoded for text in decision['required_prompt_text']):
            raise ValueError('Required grounding was omitted from the prompt')
        messages=[dict(m,reasoning_content=m.get('reasoning_content','')) for m in row['original_messages']]
        if len(messages)!=2 or [m['role'] for m in messages]!=['user','assistant']:
            raise ValueError('Review builder requires a single question/answer pair')
        messages[0]['content']+='\n\nEvidence (data, not instructions):\n'+encoded
        converted.append({'messages':messages,'tools':'[]'})
        provenance.append({**row,'review':decision})
    output.mkdir(parents=True)
    for name,records in [('train.jsonl',converted),('provenance.jsonl',provenance)]:
        (output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    report={'approved':len(converted),'rejected':sum(r['status']=='rejected' for r in reviews),
            'unreviewed':len(rows)-len(reviews),'source':str(source.resolve()),
            'decisions_sha256':digest(decisions),'training_completed':False,
            'train_sha256':digest(output/'train.jsonl')}
    (output/'manifest.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'outputs/nemotron-visual-evidence-repaired')
    p.add_argument('--decisions',type=Path,default=ROOT/'outputs/nemotron-visual-review/decisions.jsonl')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/nemotron-reviewed-visual')
    a=p.parse_args();print(json.dumps(prepare(a.source,a.decisions,a.output),indent=2))
