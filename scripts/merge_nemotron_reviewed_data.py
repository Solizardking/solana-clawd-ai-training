#!/usr/bin/env python3
"""Merge reviewed visual evidence into a fresh text dataset without changing split membership."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge(base,reviewed,decisions,output):
    if output.exists():raise ValueError('Preserve existing merged dataset')
    manifest=json.loads((base/'manifest.json').read_text())
    for name,digest in manifest['outputs_sha256'].items():
        if sha(base/name)!=digest:raise ValueError('Base preparation changed: '+name)
    review_manifest=json.loads((reviewed/'manifest.json').read_text())
    if sha(reviewed/'train.jsonl')!=review_manifest['train_sha256'] or sha(decisions)!=review_manifest['decisions_sha256']:
        raise ValueError('Reviewed data or decisions changed')
    additions=rows(reviewed/'train.jsonl');provenance=rows(reviewed/'provenance.jsonl')
    if len(additions)!=len(provenance):raise ValueError('Reviewed provenance count mismatch')
    decisions_by_id={r['id']:r for r in rows(decisions)}
    seen_ids=set();groups={}
    for split in ('train','validation','test'):
        for row in rows(base/f'{split}-provenance.jsonl'):
            seen_ids.add(row['id'])
            if row.get('group') is not None:groups[row['group']]=split
    added_ids=set()
    for row in provenance:
        if row['id'] in seen_ids or row['id'] in added_ids:raise ValueError('Duplicate supervised example ID')
        if row['split']!='train' or groups.get(row.get('group'),'train')!='train':raise ValueError('Held-out group leakage')
        if decisions_by_id[row['id']]['status']!='approved':raise ValueError('Unapproved visual example')
        added_ids.add(row['id'])
    pending=rows(base/'train-visual-pending.jsonl')
    if not added_ids <= {r['id'] for r in pending}:raise ValueError('Reviewed IDs absent from pending set')
    rejected_ids={key for key,value in decisions_by_id.items() if value['status']=='rejected'}
    shutil.copytree(base,output)
    with (output/'train.jsonl').open('a') as f:
        for row in additions:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (output/'train-provenance.jsonl').open('a') as f:
        for row in provenance:
            f.write(json.dumps({k:row.get(k) for k in ('id','group','split','sources')},ensure_ascii=False)+'\n')
    for filename,records in [('train-visual-pending.jsonl',[r for r in pending if r['id'] not in added_ids|rejected_ids]),
                             ('train-visual-rejected.jsonl',[r for r in pending if r['id'] in rejected_ids])]:
        (output/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    shutil.copy2(reviewed/'provenance.jsonl',output/'reviewed-visual-provenance.jsonl')
    manifest['counts']['train'].update(text_ready=manifest['counts']['train']['text_ready']+len(additions),
                                      visual_pending=len(pending)-len(added_ids)-len(rejected_ids),visual_rejected=len(rejected_ids))
    manifest.update(training_completed=False,visual_enrichment_completed=False,
                    reviewed_visual_rows_added=len(additions),parent_manifest_sha256=sha(base/'manifest.json'),
                    review_manifest_sha256=sha(reviewed/'manifest.json'),decisions_sha256=sha(decisions))
    manifest['outputs_sha256']={p.name:sha(p) for p in output.glob('*.jsonl')}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest['counts']


if __name__=='__main__':
    print(json.dumps(merge(ROOT/'outputs/nemotron-chart-data',ROOT/'outputs/nemotron-reviewed-visual-expanded',
         ROOT/'outputs/nemotron-visual-review/combined-decisions.jsonl',ROOT/'outputs/nemotron-chart-data-reviewed'),indent=2))
