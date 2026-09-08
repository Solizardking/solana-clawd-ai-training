from fastapi.testclient import TestClient
import chart_agent.spark_server as service


def test_private_space_needs_model_key_even_with_hf_auth(monkeypatch):
    monkeypatch.setenv('CHART_MODEL_API_KEY','model-fixture')
    monkeypatch.setitem(service.state,'alias','clawd-spark')
    monkeypatch.setitem(service.state,'adapter','fixture-adapter')
    client=TestClient(service.app)
    assert client.get('/health',headers={'Authorization':'Bearer hf-fixture'}).status_code==401
    response=client.get('/health',headers={'Authorization':'Bearer hf-fixture','X-Chart-Model-Key':'model-fixture'})
    assert response.status_code==200
    assert response.json()['model']=='clawd-spark'


def test_chat_rejects_unknown_model_and_unsupported_modes(monkeypatch):
    monkeypatch.setenv('CHART_MODEL_API_KEY','model-fixture')
    monkeypatch.setitem(service.state,'alias','clawd-spark')
    client=TestClient(service.app)
    body={'model':'wrong-model','messages':[{'role':'user','content':'hello'}]}
    headers={'Authorization':'Bearer model-fixture'}
    assert client.post('/v1/chat/completions',headers=headers,json=body).status_code==404
    body.update(model='clawd-spark',stream=True)
    assert client.post('/v1/chat/completions',headers=headers,json=body).status_code==422
