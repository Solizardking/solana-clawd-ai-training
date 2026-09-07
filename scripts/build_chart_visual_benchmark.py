#!/usr/bin/env python3
"""Generate independently seeded visual QA absent from the active training package."""
import hashlib
import json
import os
from pathlib import Path
import random
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/chart-agent/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out = ROOT / 'outputs/chart-agent/visual-benchmark'
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(202609071819)
    cases = []
    for number in range(12):
        kind = ['vertical_bar', 'horizontal_bar', 'line', 'grouped_bar'][number % 4]
        labels = ['Aster', 'Birch', 'Cedar', 'Dahlia', 'Elm']
        values = rng.sample(range(12, 95), 5)
        other = rng.sample(range(10, 90), 5)
        dark = number % 3 == 0
        with plt.style.context('dark_background' if dark else 'default'):
            fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
            ax.set_title('Independent chart reading test — ' + kind.replace('_', ' '))
            if kind == 'horizontal_bar':
                bars = ax.barh(labels, values, color='#47ad97')
                ax.bar_label(bars, padding=4)
                ax.set_xlabel('Units'); ax.set_xlim(0, 110)
            elif kind == 'line':
                ax.plot(labels, values, color='#738cdb', marker='o', linewidth=2)
                for x, v in enumerate(values): ax.annotate(str(v), (x, v), xytext=(0, 10), textcoords='offset points', ha='center')
                ax.set_ylabel('Units'); ax.set_ylim(0, 110)
            elif kind == 'grouped_bar':
                bars = ax.bar([x - .2 for x in range(5)], values, width=.38, label='Series One', color='#47ad97')
                bars2 = ax.bar([x + .2 for x in range(5)], other, width=.38, label='Series Two', color='#b88eca')
                ax.bar_label(bars, padding=3); ax.bar_label(bars2, padding=3)
                ax.set_xticks(range(5), labels); ax.set_ylabel('Units'); ax.set_ylim(0, 115); ax.legend(loc='upper left', ncol=2)
            else:
                bars = ax.bar(labels, values, color='#47ad97')
                ax.bar_label(bars, padding=3); ax.set_ylabel('Units'); ax.set_ylim(0, 110)
            ax.grid(axis='x' if kind == 'horizontal_bar' else 'y', alpha=.15)
            fig.tight_layout()
            image = out / f'{number:02d}-{kind}.png'
            fig.savefig(image); plt.close(fig)
        prefix = 'For Series One, ' if kind == 'grouped_bar' else ''
        qa = [(prefix + 'which category has the highest value?', labels[values.index(max(values))]),
              (prefix + 'what is the absolute difference between Aster and Elm?', str(abs(values[0] - values[4]))),
              (prefix + 'what is the value for Cedar?', str(values[2]))]
        for i, (question, answer) in enumerate(qa):
            cases.append({'id': f'{number:02d}-{i}', 'image': image.name, 'question': question, 'answer': answer,
                          'chart_type': kind, 'dark': dark, 'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()})
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'seed': 202609071819,
              'synthetic': True, 'note': 'Generated independently after full training began; never add these cases to training or retrieval.',
              'cases': cases}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    print('Created', len(cases), 'questions across 12 independent charts at', out)


if __name__ == '__main__':
    main()
