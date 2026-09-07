#!/usr/bin/env python3
"""Run the converted chart-label detector on local example images."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chart_agent.detector import Detector, decode_image

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('images', nargs='+', type=Path)
    args = p.parse_args()
    model = Detector('outputs/chart-agent/assets/chart-pattern.onnx')
    results = []
    for path in args.images:
        data = path.read_bytes()
        image = decode_image(data)
        start = time.monotonic()
        detections = model.detect(image)
        results.append({'image': path.name, 'sha256': hashlib.sha256(data).hexdigest(),
                        'size': image.size, 'seconds': time.monotonic() - start, 'detections': detections})
    report = {'classes': model.names, 'results': results, 'heldout_accuracy_evaluated': False,
              'note': 'Example execution only; no ground-truth annotation set or profitability evaluation.'}
    Path('outputs/chart-agent/pattern-smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
