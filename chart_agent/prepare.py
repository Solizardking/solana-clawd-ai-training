"""Prepare audited train/eval records and a local research index."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter
from .research import build_index


def split_for(group):
    # Group images / conversation families together to prevent QA leakage.
    n = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return 'test' if n < 5 else 'validation' if n < 10 else 'train'


def prepare(bucket, output, metadata=None):
    import pyarrow.parquet as pq
    bucket, output = Path(bucket), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts, rejected, sources = Counter(), Counter(), []
    seen = set()
    handles = {s: (output / (s + '.jsonl')).open('w') for s in ('train', 'validation', 'test')}

    def emit(row, group, lane):
        digest = hashlib.sha256(json.dumps(row['messages'], sort_keys=True).encode()).hexdigest()
        if digest in seen:
            rejected['duplicate'] += 1
            return
        seen.add(digest)
        split = split_for(group)
        handles[split].write(json.dumps(dict(row, split=split, lane=lane, group=group), ensure_ascii=False) + '\n')
        counts[f'{lane}/{split}'] += 1

    try:
        for path in sorted((bucket / 'data').glob('*.parquet')):
            table = pq.ParquetFile(path)
            source = dict(file=path.name, rows=table.metadata.num_rows, columns=table.schema_arrow.names)
            sources.append(source)
            if 'messages' not in table.schema_arrow.names:
                rejected['non_chart_conversation_schema'] += table.metadata.num_rows
                continue
            for batch in table.iter_batches(batch_size=128):
                for index, row in enumerate(batch.to_pylist()):
                    messages = row['messages']
                    if not isinstance(messages, list) or not messages or messages[-1].get('role') != 'assistant' or any(
                        m.get('role') not in {'user', 'assistant', 'system'} or not isinstance(m.get('content'), str) for m in messages
                    ):
                        rejected['invalid_conversation'] += 1
                        continue
                    trajectory = row.get('trajectory') or {}
                    # near_dup_of is the source family identifier when supplied.
                    group = trajectory.get('near_dup_of') or trajectory.get('id') or hashlib.sha256(messages[0]['content'].encode()).hexdigest()
                    emit(dict(messages=messages, source=f'ordlibrary/charts/data/{path.name}'), str(group), 'chart_reasoning')
        meta = Path(metadata) if metadata else bucket / 'metadata.csv'
        with meta.open(newline='') as file:
            for line, row in enumerate(csv.reader(file), 1):
                if row and row[0] in {'image', 'image_path', 'file_name'}:
                    continue
                if len(row) != 11 or not row[0].startswith('images/') or not row[3] or not row[4]:
                    rejected['invalid_image_qa_row'] += 1
                    continue
                image = (bucket / row[0]).resolve()
                if not image.is_relative_to(bucket.resolve()) or not image.is_file():
                    rejected['missing_or_unsafe_image'] += 1
                    continue
                emit(dict(messages=[{'role': 'user', 'content': row[3]}, {'role': 'assistant', 'content': row[4]}],
                          image=row[0], source=f'ordlibrary/charts/metadata.csv:{line}', original_split=row[9]), row[0], 'chart_image_qa')
    finally:
        for handle in handles.values():
            handle.close()
    report = dict(counts=dict(counts), rejected=dict(rejected), sources=sources,
                  image_root=str(bucket.resolve()), training_performed=False,
                  note='Use train for training/retrieval examples. Validation/test are held out. Licenses and near-duplicate families require review before publishing.')
    (output / 'manifest.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bucket', default='local')
    p.add_argument('--output', default='outputs/chart-agent/dataset')
    p.add_argument('--metadata')
    p.add_argument('--research', nargs='*', default=[])
    args = p.parse_args()
    report = prepare(args.bucket, args.output, args.metadata)
    if args.research:
        report['research'] = build_index(args.research, Path(args.output).parent / 'research.sqlite')
        (Path(args.output) / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
