#!/usr/bin/env python3
"""Approve narrowly defined direct-value bar questions only when OCR geometry and target agree."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from chart_agent.server import compact_evidence

PATTERNS=[r'How much revenue did (.+) generate\?',r"What was (.+)'s revenue\?",r'Revenue for (.+)\?']


def check(messages,observation):
    if len(messages)!=2 or [m['role'] for m in messages]!=['user','assistant']:return None
    category=None
    for pattern in PATTERNS:
        match=re.fullmatch(pattern,messages[0]['content'])
        if match:category=match.group(1);break
    if not category:return None
    if not observation['image'].startswith('images/syn_bar_'):return None
    if not observation['ocr']['available']:return None
    words=observation['ocr']['words'];height=observation['size'][1]
    labels=[w for w in words if w['text']==category and w['confidence']>=.9 and w['bbox'][1]>=height*.8]
    if len(labels)!=1:return None
    label=labels[0];cx=(label['bbox'][0]+label['bbox'][2])/2
    aligned=[]
    for word in words:
        if not re.fullmatch(r'\$\d+',word['text']) or word['confidence']<.9:continue
        x,y,x2,y2=word['bbox'];center=(x+x2)/2
        tolerance=max(x2-x,label['bbox'][2]-label['bbox'][0])/2+5
        if y2<label['bbox'][1] and abs(center-cx)<=tolerance:aligned.append(word)
    if len(aligned)!=1:return None
    value=aligned[0]['text']
    if messages[1]['content']!=value:return None
    evidence=compact_evidence({'ocr':observation['ocr'],'chart_elements':{'detections':observation['chart_elements']},
        'bbox_image_size':observation['size'],
        'vision_input':'Raw image is not visible to this text model. Use only detector and OCR observations; OCR can be wrong.'})
    encoded=json.dumps(evidence)
    if category not in encoded or value not in encoded:return None
    return {'category':category,'value':value,'category_bbox':label['bbox'],'value_bbox':aligned[0]['bbox']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'outputs/nemotron-visual-evidence-repaired')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/nemotron-direct-bar-review.jsonl')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve existing review output')
    approved=[]
    for line in (a.source/'review-candidates.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['split']!='train' or len(row['observations'])!=1:continue
        path=a.source/row['observations'][0];obs=json.loads(path.read_text())
        result=check(row['original_messages'],obs)
        if result:
            approved.append({'id':row['id'],'status':'approved','required_prompt_text':[result['category'],result['value']],
                'note':'Unique bottom category and aligned above-category currency label, confidence >=0.9, exactly match the direct-question target.',
                'review_method':'Deterministic OCR/geometry consistency check; no human review or OCR accuracy guarantee',
                'grounding':result,'observations_sha256':{row['observations'][0]:hashlib.sha256(path.read_bytes()).hexdigest()}})
    a.output.write_text(''.join(json.dumps(r)+'\n' for r in approved))
    print(json.dumps({'approved_direct_bar_rows':len(approved),'other_rows_not_approved':3198-len(approved)}))


if __name__=='__main__':main()
