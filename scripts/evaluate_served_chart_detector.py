#!/usr/bin/env python3
"""Measure the served ONNX detector at its default confidence and NMS settings."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chart_agent.detector import Detector, decode_image


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def match(predictions, truth):
    """Confidence-ordered, one-to-one matching within one image and class."""
    remaining = set(range(len(truth)))
    tp = 0
    for prediction in sorted(predictions, key=lambda p: -p['confidence']):
        best = max(remaining, key=lambda i: iou(prediction['bbox_xyxy'], truth[i]), default=None)
        if best is not None and iou(prediction['bbox_xyxy'], truth[best]) >= 0.5:
            remaining.remove(best)
            tp += 1
    return tp, len(predictions)-tp, len(truth)-tp


def evaluate(model_path, data_path, split, image_list=None):
    config = json.loads(data_path.read_text())
    data_root = Path(config['path'])
    images = ([Path(p) for p in image_list.read_text().splitlines() if p]
              if image_list else sorted((data_root/config[split]).glob('*')))
    if not images:
        raise ValueError('No evaluation images')
    detector = Detector(model_path)
    expected = {int(k): v for k, v in config['names'].items()}
    if detector.names != expected:
        raise ValueError('Export class mapping differs from annotations')
    totals = {name: [0, 0, 0] for name in expected.values()}
    for path in images:
        # Resolve labels only within the selected prepared dataset.
        relative = path.resolve().relative_to((data_root/'images').resolve())
        labels = data_root/'labels'/relative.with_suffix('.txt')
        image = decode_image(path.read_bytes())
        w, h = image.size
        truth = {name: [] for name in expected.values()}
        for line in labels.read_text().splitlines():
            cls, cx, cy, bw, bh = map(float, line.split())
            if cls != int(cls) or int(cls) not in expected:
                raise ValueError('Invalid annotation class')
            truth[expected[int(cls)]].append([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h])
        predictions = detector.detect(image)
        for name in totals:
            counts = match([p for p in predictions if p['label'] == name], truth[name])
            totals[name] = [a+b for a, b in zip(totals[name], counts)]
    def metrics(counts):
        tp, fp, fn = counts
        return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else 0,
                    recall=tp/(tp+fn) if tp+fn else 0)
    return dict(runtime='chart_agent.detector.Detector', split=split,
                smoke_only=image_list is not None, images=len(images), confidence=0.35,
                iou_threshold=0.5, nms_iou=0.45, max_detections=100,
                input_size=[detector.height, detector.width],
                metric_scope='precision_recall_at_serving_threshold_not_average_precision',
                onnx_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                overall=metrics([sum(c[i] for c in totals.values()) for i in range(3)]),
                per_class={name: metrics(c) for name, c in totals.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--image-list', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Preserve the existing evaluation report')
    result = evaluate(args.model, args.data, args.split, args.image_list)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['overall']))
