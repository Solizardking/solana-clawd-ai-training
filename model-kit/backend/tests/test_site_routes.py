from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_json():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["protocol"] == "CAAP/1.0"


def test_fly_health_alias():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_api_index_json():
    response = client.get("/api")
    assert response.status_code == 200
    assert response.json()["service"] == "solana-ai-model-kit-api"


def test_site_root_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Solana AI Model Kit" in response.text


def test_register_page_html():
    response = client.get("/register")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "register" in response.text.lower()


def test_register_host_serves_register_page():
    response = client.get("/", headers={"host": "register.x402.wtf"})
    assert response.status_code == 200
    assert "register" in response.text.lower()


def test_static_app_js():
    response = client.get("/app.js")
    assert response.status_code == 200
    assert "fly.dev" in response.text
