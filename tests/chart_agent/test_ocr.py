import json
import subprocess
from types import SimpleNamespace

from PIL import Image
from chart_agent import ocr


def test_ocr_coordinates_and_invalid_confidence(monkeypatch):
    monkeypatch.setattr(ocr.shutil, 'which', lambda name: '/fixture/tesseract')
    tsv = 'text\tconf\tleft\ttop\twidth\theight\nVolume\t95\t10\t20\t30\t40\nNoise\tnan\t1\t1\t1\t1\n'
    monkeypatch.setattr(ocr.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=tsv.encode()))
    result = ocr.extract_text(Image.new('RGB', (4800, 2400)))
    assert result['words'] == [{'text': 'Volume', 'confidence': .95, 'bbox': [20, 40, 80, 120]}]
    assert result['bbox_image_size'] == [4800, 2400]


def test_ocr_timeout_is_missing_evidence(monkeypatch):
    monkeypatch.setattr(ocr.shutil, 'which', lambda name: '/fixture/tesseract')
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('tesseract', 20)
    monkeypatch.setattr(ocr.subprocess, 'run', timeout)
    result = ocr.extract_text(Image.new('RGB', (10, 10)))
    assert result == {'available': False, 'error': 'TimeoutExpired', 'words': []}


def test_dense_evidence_retains_ocr_and_valid_json():
    from chart_agent.server import compact_evidence
    evidence = {
        'research': [{'text': 'x' * 15000}],
        'chart_elements': {'detections': [{'label': 'plot_area', 'bbox': [1, 2, 3, 4]}] * 100},
        'ocr': {'words': [{'text': 'Volume', 'confidence': .95, 'bbox': [1, 2, 3, 4]}] * 300},
        'vision_input': 'Raw image is not visible; OCR can be wrong.'}
    encoded = json.dumps(compact_evidence(evidence))
    restored = json.loads(encoded)
    assert len(encoded) <= 11000
    assert restored['ocr']['words'][0]['text'] == 'Volume'
    assert restored['ocr']['words_omitted_from_prompt'] > 0
    assert restored['chart_elements']['detections']
    assert evidence['ocr']['words'] == [{'text': 'Volume', 'confidence': .95, 'bbox': [1, 2, 3, 4]}] * 300


def test_upscaled_ocr_maps_back_to_original_coordinates(monkeypatch):
    monkeypatch.setattr(ocr.shutil, 'which', lambda name: '/fixture/tesseract')
    tsv = 'text\tconf\tleft\ttop\twidth\theight\n$12300\t95\t20\t40\t60\t80\n'
    monkeypatch.setattr(ocr.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=tsv.encode()))
    result = ocr.extract_text(Image.new('RGB', (1000, 500)))
    assert result['words'][0]['bbox'] == [10, 20, 40, 60]
    assert result['bbox_image_size'] == [1000, 500]
