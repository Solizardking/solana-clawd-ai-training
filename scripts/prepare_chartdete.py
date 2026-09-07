#!/usr/bin/env python3
"""Extract original ChartDete COCO splits and prepare 18-class YOLO labels."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import zipfile


def normalized_box(box, width, height):
    if width <= 0 or height <= 0 or len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError('Invalid bounding box or image dimensions')
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=Path('outputs/chart-agent/chartdete/pmc_2022.zip'))
    p.add_argument('--output', type=Path, default=Path('outputs/chart-agent/chartdete/prepared'))
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stats, seen, categories = {}, {}, None
    prefix = 'pmc_2022/pmc_coco/element_detection/'
    with zipfile.ZipFile(args.archive) as z:
        for split, source in [('train', 'train'), ('val', 'val'), ('test', 'split3_test')]:
            raw = z.read(prefix+source+'.json')
            data = json.loads(raw)
            cats = sorted(data['categories'], key=lambda x:x['id'])
            if categories is not None and cats != categories:
                raise ValueError('Category mapping changed across splits')
            categories = cats
            mapping = {c['id']:i for i,c in enumerate(cats)}
            (root/'annotations').mkdir(exist_ok=True)
            (root/'annotations'/f'{split}.json').write_bytes(raw)
            annotations = defaultdict(list)
            for ann in data['annotations']:
                if ann['category_id'] not in mapping: raise ValueError('Unknown category')
                annotations[ann['image_id']].append(ann)
            for sub in ('images','labels'):
                (root/sub/split).mkdir(parents=True, exist_ok=True)
            count = {'images':0, 'labels':0, 'skipped_crowd_or_degenerate':0, 'clipped_boxes':0}
            for im in data['images']:
                filename = im['file_name']
                if Path(filename).name != filename: raise ValueError('Unsafe image filename')
                content = z.read(prefix+source+'/'+filename)
                digest = hashlib.sha256(content).hexdigest()
                if digest in seen and seen[digest] != split:
                    raise ValueError('Exact image duplicate across upstream splits')
                seen[digest] = split
                (root/'images'/split/filename).write_bytes(content)
                labels = []
                for ann in annotations.pop(im['id'], []):
                    box = normalized_box(ann['bbox'], im['width'], im['height'])
                    if ann.get('iscrowd',0) or box is None:
                        count['skipped_crowd_or_degenerate'] += 1
                        continue
                    x,y,w,h = ann['bbox']
                    count['clipped_boxes'] += int(x<0 or y<0 or x+w>im['width'] or y+h>im['height'])
                    labels.append(str(mapping[ann['category_id']])+' '+' '.join(f'{v:.9f}' for v in box))
                (root/'labels'/split/Path(filename).with_suffix('.txt')).write_text('\n'.join(labels)+'\n')
                count['images'] += 1
                count['labels'] += len(labels)
            if annotations: raise ValueError('Annotations reference missing images')
            stats[split] = count
    config = {'path':str(root), 'train':'images/train', 'val':'images/val', 'test':'images/test',
              'names':{i:c['name'] for i,c in enumerate(categories)}}
    # JSON is valid YAML and preserves names without a second serializer dependency.
    (root/'data.yaml').write_text(json.dumps(config,indent=2)+'\n')
    with args.archive.open('rb') as stream: digest = hashlib.file_digest(stream,'sha256').hexdigest()
    report = {'archive_sha256':digest, 'splits':stats, 'classes':config['names'],
              'split_policy':'original ChartDete splits; exact image hashes checked across splits',
              'source':'https://github.com/pengyu965/ChartDete', 'model_trained':False}
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
