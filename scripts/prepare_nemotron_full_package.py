#!/usr/bin/env python3
"""Prepare a small reproducible delta for full Nemotron training; never upload or launch."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(output):
    if output.exists():raise ValueError('Preserve existing package')
    base=ROOT/'outputs/nemotron-chart-data';merged=ROOT/'outputs/nemotron-chart-data-reviewed'
    bm=json.loads((base/'manifest.json').read_text());mm=json.loads((merged/'manifest.json').read_text())
    for directory,manifest in [(base,bm),(merged,mm)]:
        for split in ['train','validation','test']:
            if sha(directory/f'{split}.jsonl')!=manifest['outputs_sha256'][f'{split}.jsonl']:raise ValueError('Dataset hash changed')
    original=(base/'train.jsonl').read_bytes();all_rows=(merged/'train.jsonl').read_bytes()
    if not all_rows.startswith(original):raise ValueError('Base training prefix changed')
    additions=all_rows[len(original):]
    rows=[json.loads(line) for line in additions.splitlines()]
    if sha(merged/'reviewed-visual-provenance.jsonl')!=mm['outputs_sha256']['reviewed-visual-provenance.jsonl']:raise ValueError('Reviewed provenance changed')
    provenance=[json.loads(line) for line in (merged/'reviewed-visual-provenance.jsonl').read_text().splitlines()]
    if len(rows)!=9 or len(provenance)!=9 or len({r['id'] for r in provenance})!=9:raise ValueError('Expected nine unique reviewed additions')
    if any(bm['outputs_sha256'][f'{s}.jsonl']!=mm['outputs_sha256'][f'{s}.jsonl'] for s in ['validation','test']):raise ValueError('Holdouts changed')
    output.mkdir(parents=True)
    (output/'reviewed-visual.jsonl').write_bytes(additions)
    for source,name in [(merged/'reviewed-visual-provenance.jsonl','reviewed-visual-provenance.jsonl'),(ROOT/'deploy/nemotron/training/assistant-mask.jinja','assistant-mask.jinja')]:shutil.copy2(source,output/name)
    manifest={'base_dataset_repo':'ordlibrary/clawd-chart-foundation-training','base_revision':'6456947cf89cbb5d1a4c0c120199a5038532bfc6','base_archive':'chart-foundation-data.zip','base_archive_sha256':'3e389d917a4cecac5c9994dcfa68434c55cff7a21db04cb44803849a9e3f24df',
              'prepared_base_sha256':{f'{s}.jsonl':bm['outputs_sha256'][f'{s}.jsonl'] for s in ['train','validation','test']},
              'prepared_merged_sha256':{f'{s}.jsonl':mm['outputs_sha256'][f'{s}.jsonl'] for s in ['train','validation','test']},
              'files_sha256':{p.name:sha(p) for p in output.iterdir()},'reviewed_rows':len(rows),'training_rows':31642,'validation_rows':3257,'test_rows':3126,
              'native_mask_verification':'pending','full_training_completed':False,'uploads_performed':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/nemotron-full-package');args=p.parse_args()
    print(json.dumps(prepare(args.output),indent=2))
