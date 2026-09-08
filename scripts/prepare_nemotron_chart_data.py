#!/usr/bin/env python3
"""Prepare text-only Nemotron agent-chat data; retain visual rows for evidence enrichment."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def prepare(source, output):
    if output.exists():
        raise ValueError('Use a new output directory to preserve earlier preparation')
    rows = {}
    for split in ('train', 'validation', 'test'):
        rows[split] = [json.loads(line) for line in (source/f'{split}.jsonl').read_text().splitlines() if line.strip()]
    seen = {}
    for split, records in rows.items():
        for row in records:
            for field in ('id', 'group'):
                value = row.get(field)
                if value is None:
                    continue
                key = (field, value)
                if key in seen and seen[key] != split:
                    raise ValueError(f'Cross-split {field} leakage')
                seen[key] = split
    output.mkdir(parents=True)
    report = {'source': str(source.resolve()), 'counts': {}, 'inputs_sha256': {},
              'training_completed': False, 'visual_enrichment_completed': False,
              'documents_require_separate_training_stage': True}
    for split, records in rows.items():
        counts = {'text_ready': 0, 'visual_pending': 0}
        with (output/f'{split}.jsonl').open('w') as ready, (output/f'{split}-visual-pending.jsonl').open('w') as pending, (output/f'{split}-provenance.jsonl').open('w') as provenance:
            for row in records:
                if row.get('images'):
                    pending.write(json.dumps(row, ensure_ascii=False)+'\n')
                    counts['visual_pending'] += 1
                    continue
                messages = row['messages']
                if any(not isinstance(m.get('content'), str) for m in messages):
                    raise ValueError('Text rows must contain string content')
                if any(m.get('tool_calls') or m.get('function_call') for m in messages):
                    raise ValueError('Structured tool calls require reviewed Nemotron serialization')
                converted = {'messages': [dict(m, reasoning_content=m.get('reasoning_content', '')) for m in messages],
                             'tools': json.dumps(row.get('tools', []), ensure_ascii=False)}
                ready.write(json.dumps(converted, ensure_ascii=False)+'\n')
                provenance.write(json.dumps({k: row.get(k) for k in ('id','group','split','sources')}, ensure_ascii=False)+'\n')
                counts['text_ready'] += 1
        report['counts'][split] = counts
        with (source/f'{split}.jsonl').open('rb') as f:
            report['inputs_sha256'][split] = hashlib.file_digest(f,'sha256').hexdigest()
    for path in source.glob('documents-*.jsonl'):
        shutil.copy2(path, output/path.name)
    report['outputs_sha256'] = {}
    for path in output.glob('*.jsonl'):
        with path.open('rb') as f:
            report['outputs_sha256'][path.name] = hashlib.file_digest(f,'sha256').hexdigest()
    (output/'manifest.json').write_text(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/chart-foundation-data'))
    p.add_argument('--output',type=Path,default=Path('outputs/nemotron-chart-data'))
    args=p.parse_args()
    print(json.dumps(prepare(args.source,args.output),indent=2))
