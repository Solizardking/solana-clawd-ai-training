import pytest
import asyncio
import base64
import io
import json
import httpx
from PIL import Image


@pytest.mark.parametrize('backend', ['nemotron', 'spark'])
def test_nemotron_receives_evidence_without_image_and_uses_private_backend_key(monkeypatch, backend):
    import chart_agent.server as server
    monkeypatch.setattr(server, 'model_backend', backend)
    monkeypatch.setattr(server, 'model_name', 'clawd-nemotron')
    monkeypatch.setenv('CHART_MODEL_API_KEY', 'model-only-fixture')
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={'choices': [{'message': {'content': 'Plot area detected; label text unavailable.'}}]})
    real_client = httpx.AsyncClient
    monkeypatch.setattr(server.httpx, 'AsyncClient', lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    async def detections(image):
        return {'chart_elements': {'detections': [{'class_name': 'plot_area', 'bbox': [1, 1, 8, 8]}]}}
    monkeypatch.setattr(server, 'detection_evidence', detections)
    monkeypatch.setattr(server, 'extract_text', lambda image: {'available': True, 'words': [{'text': 'Volume', 'confidence': 0.95}]})
    image = io.BytesIO()
    Image.new('RGB', (10, 10)).save(image, format='PNG')
    body = server.AnalyzeRequest(question='Read the chart', image_base64=base64.b64encode(image.getvalue()).decode(), use_research=False)
    result = asyncio.run(server.analyze_impl(body))
    payload = json.loads(seen[0].content)
    assert payload['model'] == 'clawd-nemotron'
    assert payload['tool_choice'] == 'auto'
    assert isinstance(payload['messages'][1]['content'], str)
    assert 'plot_area' in payload['messages'][1]['content']
    assert 'OCR can be wrong' in payload['messages'][1]['content']
    assert 'Volume' in payload['messages'][1]['content']
    assert 'image_url' not in json.dumps(payload)
    assert seen[0].headers['authorization'] == 'Bearer model-only-fixture'
    assert 'model-only-fixture' not in json.dumps(result)
    assert result['model'] == 'clawd-nemotron'


def test_nemotron_tokenizer_roundtrip_contract(monkeypatch):
    import chart_agent.server as server
    monkeypatch.setattr(server, 'model_backend', 'nemotron')
    monkeypatch.setattr(server, 'model_name', 'clawd-nemotron')
    def handler(request):
        data = json.loads(request.content)
        assert data['model'] == 'clawd-nemotron'
        if request.url.path == '/tokenize':
            assert data['prompt'] == '18446744073709551615'
            assert data['add_special_tokens'] is False
            return httpx.Response(200, json={'tokens': [1, 2], 'count': 2})
        assert data['tokens'] == [1, 2]
        return httpx.Response(200, json={'prompt': '18446744073709551615'})
    real_client = httpx.AsyncClient
    monkeypatch.setattr(server.httpx, 'AsyncClient', lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    result = asyncio.run(server.tokenize(server.TokenizeRequest(text='18446744073709551615')))
    assert result['roundtrip_exact'] is True


def test_truncated_tool_calls_never_execute(monkeypatch):
    import pytest
    from fastapi import HTTPException
    import chart_agent.server as server
    async def forbidden(*args):
        raise AssertionError('A truncated call must not execute')
    monkeypatch.setattr(server, 'run_tool', forbidden)
    def handler(request):
        return httpx.Response(200, json={'choices': [{'finish_reason': 'length', 'message': {
            'content': None, 'tool_calls': [{'id': 'fixture', 'type': 'function', 'function': {
                'name': 'convert_token_amount', 'arguments': '{"amount":"123","decimals":6}'}}]}}]})
    real_client = httpx.AsyncClient
    monkeypatch.setattr(server.httpx, 'AsyncClient', lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.analyze_impl(server.AnalyzeRequest(question='Convert 123', use_research=False)))
    assert exc.value.status_code == 502
