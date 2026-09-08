"""Bounded local OCR observations; recognized text is untrusted and may be inaccurate."""
import csv
import io
import math
import os
import shutil
import subprocess


def extract_text(image):
    executable = shutil.which('tesseract')
    if not executable:
        return {'available': False, 'error': 'tesseract_not_installed', 'words': []}
    width, height = image.size
    # Thin chart fonts lose digits at native resolution. Bound both upscaling
    # and downscaling, then map observations back to the original coordinates.
    from PIL import Image
    scale = min(2.0, 2400 / max(width, height))
    rendered = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                            Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    rendered.save(buffer, format='PNG')
    try:
        result = subprocess.run([executable, 'stdin', 'stdout', '-l', 'eng', '--psm', '11', 'tsv'],
            input=buffer.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=20, env=dict(os.environ, OMP_THREAD_LIMIT='2'), check=True)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        return {'available': False, 'error': type(exc).__name__, 'words': []}
    words = []
    sx, sy = width/rendered.width, height/rendered.height
    for row in csv.DictReader(io.StringIO(result.stdout.decode('utf-8', errors='replace')), delimiter='\t'):
        text = (row.get('text') or '').strip()
        if not text:
            continue
        try:
            confidence = float(row['conf'])
            x, y, w, h = (int(row[k]) for k in ('left','top','width','height'))
        except (ValueError, KeyError):
            continue
        if not math.isfinite(confidence) or confidence < 30 or w <= 0 or h <= 0:
            continue
        words.append({'text': text[:256], 'confidence': confidence/100,
                      'bbox': [round(x*sx,2),round(y*sy,2),round((x+w)*sx,2),round((y+h)*sy,2)]})
    return {'available': True, 'engine': 'tesseract', 'language': 'eng', 'bbox_image_size': [width,height],
            'words': words[:300], 'omitted_words': max(0,len(words)-300),
            'interpretation': 'OCR estimates, not verified values. Preserve uncertainty; do not infer mint identities or exact amounts from OCR alone.'}
