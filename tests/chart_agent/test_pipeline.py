import asyncio
import base64
import csv
import io
import json
from pathlib import Path

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from chart_agent.detector import decode_image, nms
from chart_agent.prepare import prepare, split_for
from chart_agent.realtime import Tape
from chart_agent.research import build_index, search
from chart_agent.tools import SolGptBridge, catalog, validate_mint


def test_catalog_contract():
    tools = catalog()
    assert len(tools) == 72
    assert sum(t['core'] for t in tools.values()) == 37
    assert 'prepare_user_swap' in tools
    assert not {'execute_swap', 'sponge_bridge', 'place_order'} & tools.keys()


def test_address_validation():
    assert validate_mint('So11111111111111111111111111111111111111112')
    # Observed Nemotron pilot regression: one extra base58 digit in wrapped SOL.
    for invalid in ['SOL', '0' * 44, 'z' * 44, '../../secrets',
                    'So111111111111111111111111111111111111111112']:
        with pytest.raises(ValueError):
            validate_mint(invalid)


def test_openrouter_provider_key_is_not_used_for_mcp(monkeypatch):
    monkeypatch.delenv('SOLGPT_MCP_TOKEN', raising=False)
    monkeypatch.setenv('SOLGPT_API_KEY', 'sk-or-v1-fixture')
    assert SolGptBridge().key is None
    monkeypatch.setenv('SOLGPT_API_KEY', 'provider-fixture')
    monkeypatch.setenv('SOLGPT_API_BASE', 'https://openrouter.ai/api/v1')
    assert SolGptBridge().key is None
    monkeypatch.setenv('SOLGPT_MCP_TOKEN', 'mcp-fixture')
    assert SolGptBridge().key == 'mcp-fixture'


def test_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [30, 30, 40, 40]])
    assert nms(boxes, np.array([0.9, 0.8, 0.7])) == [0, 2]


def test_image_validation():
    buf = io.BytesIO()
    Image.new('RGB', (20, 10)).save(buf, format='PNG')
    assert decode_image(buf.getvalue()).size == (20, 10)
    with pytest.raises(OSError):
        decode_image(b'not an image')


def test_research_quotes_and_citations(tmp_path):
    source = tmp_path / 'research.md'
    source.write_text('Chart detection uses bounding boxes and axes. Solana mint decimals affect amounts.')
    db = tmp_path / 'r.sqlite'
    build_index([source], db)
    rows = search(db, '"chart" OR (axes)')
    assert len(rows) == 1
    assert rows[0]['page'] == 1
    assert rows[0]['source'].endswith('research.md')
    assert search(db, '!!!') == []


def test_dataset_excludes_helm_groups_images_and_keeps_distinct_images(tmp_path):
    bucket, out = tmp_path / 'bucket', tmp_path / 'out'
    (bucket / 'data').mkdir(parents=True)
    (bucket / 'images').mkdir()
    for name in ('a.png', 'b.png'):
        Image.new('RGB', (10, 10)).save(bucket / 'images' / name)
    pq.write_table(pa.Table.from_pylist([{'chart_name': 'helm', 'values': 'replicas: 1'}]), bucket / 'data/helm.parquet')
    row = {'messages': [{'role': 'user', 'content': 'chart question'}, {'role': 'assistant', 'content': 'answer'}], 'trajectory': {'id': 'family'}}
    pq.write_table(pa.Table.from_pylist([row, row]), bucket / 'data/chart.parquet')
    with (bucket / 'metadata.csv').open('w') as f:
        writer = csv.writer(f)
        for image, question in [('a.png', 'max?'), ('a.png', 'min?'), ('b.png', 'max?')]:
            writer.writerow(['images/' + image, question, 'synthetic', question, '10', 'bar', 'lookup', 'easy', 'true', 'train', 'notes'])
        writer.writerow(['truncated'])
    report = prepare(bucket, out)
    assert report['rejected'] == {'duplicate': 1, 'non_chart_conversation_schema': 1, 'invalid_image_qa_row': 1}
    rows = [json.loads(line) for p in out.glob('*.jsonl') for line in p.read_text().splitlines()]
    assert len(rows) == 4
    image_rows = [r for r in rows if r.get('image') == 'images/a.png']
    assert len({r['split'] for r in image_rows}) == 1
    assert split_for('same') == split_for('same')


def test_tape_disconnected_is_stale():
    tape = Tape()
    assert tape.snapshot()['stale']
    import time
    tape.last_received = time.time()
    assert tape.snapshot()['stale']
    tape.connected = True
    assert not tape.snapshot()['stale']
    for i in range(250):
        tape.events.append({'i': i})
    assert len(tape.events) == 200


def test_candles_reject_bad_time_and_detect_body_engulfing():
    from chart_agent.candles import candle_features
    candles = [[100, 12, 13, 9, 10, 1000], [160, 9, 14, 8, 13, 2000]]
    features = candle_features(candles)
    assert features['shapes'][-1]['shape'] == 'bullish_body_engulfing'
    assert features['sma20'] is None
    for invalid in [candles[::-1], [[100, 12, 11, 9, 10, 100], candles[1]], [[100, 12, 13, 9, float('nan'), 100], candles[1]]]:
        with pytest.raises(ValueError):
            candle_features(invalid)


def test_examples_never_index_test_split(tmp_path):
    from chart_agent.examples import index_examples, retrieve
    rows = [{'split': split, 'lane': 'chart_reasoning', 'source': 'fixture', 'group': split,
             'messages': [{'role': 'user', 'content': split + ' chart'}, {'role': 'assistant', 'content': split + ' answer'}]}
            for split in ['train', 'test', 'validation']]
    data = tmp_path / 'data.jsonl'
    data.write_text('\n'.join(json.dumps(r) for r in rows))
    db = tmp_path / 'examples.sqlite'
    assert index_examples(data, db) == 1
    assert retrieve(db, 'chart')[0]['answer'] == 'train answer'


def test_credentials_only_select_requested_keys(tmp_path, monkeypatch):
    from chart_agent.credentials import load_solgpt_env
    monkeypatch.delenv('SOLGPT_API_KEY', raising=False)
    monkeypatch.delenv('UNRELATED_SECRET', raising=False)
    monkeypatch.setenv('SOLGPT_MCP_URL', 'https://existing.example/mcp')
    path = tmp_path / '.env'
    path.write_text('SOLGPT_API_KEY="fixture"\nUNRELATED_SECRET=ignored\nSOLGPT_MCP_URL=https://override.example/mcp')
    load_solgpt_env(path)
    import os
    assert os.environ['SOLGPT_API_KEY'] == 'fixture'
    assert 'UNRELATED_SECRET' not in os.environ
    assert os.environ['SOLGPT_MCP_URL'] == 'https://existing.example/mcp'


def test_bridge_requires_real_connection(monkeypatch):
    monkeypatch.delenv('SOLGPT_MCP_TOKEN', raising=False)
    monkeypatch.delenv('SOLGPT_API_KEY', raising=False)
    bridge = SolGptBridge()
    with pytest.raises(ValueError, match='required'):
        asyncio.run(bridge.call('get_price', {}))
    with pytest.raises(ValueError, match='outside'):
        asyncio.run(bridge.call('execute_swap', {}))


def test_api_auth_and_invalid_input(monkeypatch):
    from chart_agent.server import app
    monkeypatch.setenv('CHART_API_KEY', 'test-only-secret')

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            assert (await client.get('/health')).status_code == 200
            assert (await client.get('/live')).status_code == 401
            assert (await client.post('/detect', json={'image_base64': 'invalid!'})).status_code == 401
            headers = {'Authorization': 'Bearer test-only-secret'}
            assert (await client.get('/live', headers=headers)).status_code == 200
            result = await client.post('/analyze', headers=headers, json={'question': 'chart', 'mint': 'bad'})
            assert result.status_code == 422
            result = await client.post('/detect', headers=headers, json={'image_base64': 'invalid!'})
            assert result.status_code == 422
            result = await client.post('/analyze', headers=headers, json={'question': 'chart', 'image_base64': 'invalid!'})
            assert result.status_code == 422
    asyncio.run(run())


def test_detector_results_keep_three_label_sets_separate(monkeypatch):
    import chart_agent.server as server
    class FakeDetector:
        def __init__(self, label): self.names = {0: label}
        def detect(self, image):
            return [{'label': self.names[0], 'confidence': 0.8, 'bbox_xyxy': [1, 2, 3, 4]}]
    monkeypatch.setattr(server, 'detector', FakeDetector('symbol_title'))
    monkeypatch.setattr(server, 'pattern_detector', FakeDetector('Buy'))
    monkeypatch.setattr(server, 'element_detector', FakeDetector('plot_area'))
    image = Image.new('RGB', (20, 10))
    result = asyncio.run(server.detection_evidence(image))
    assert result['detections'][0]['label'] == 'symbol_title'
    assert result['pattern_detector']['detections'][0]['label'] == 'Buy'
    assert result['chart_elements']['detections'][0]['label'] == 'plot_area'
    assert result['chart_elements']['bbox_image_size'] == [20, 10]
    monkeypatch.setattr(server, 'element_detector', None)
    monkeypatch.setattr(server, 'element_detector_error', 'not_configured')
    result = asyncio.run(server.detection_evidence(image))
    assert result['chart_elements_error'] == 'not_configured'
    assert 'chart_elements' not in result
