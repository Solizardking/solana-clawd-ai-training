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
    for invalid in ['SOL', '0' * 44, 'z' * 44, '../../secrets']:
        with pytest.raises(ValueError):
            validate_mint(invalid)


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
            headers = {'Authorization': 'Bearer test-only-secret'}
            assert (await client.get('/live', headers=headers)).status_code == 200
            result = await client.post('/analyze', headers=headers, json={'question': 'chart', 'mint': 'bad'})
            assert result.status_code == 422
            result = await client.post('/analyze', headers=headers, json={'question': 'chart', 'image_base64': 'invalid!'})
            assert result.status_code == 422
    asyncio.run(run())
