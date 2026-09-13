from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_valuation_endpoint():
    response = client.post("/api/v1/valuation", json={"bedrooms": 3, "bathrooms": 2, "area": 1800, "zipcode": "91901", "asking_price": 500000})
    assert response.status_code == 200
    assert response.json()["estimated_value"] > 0


def test_invalid_property_input():
    response = client.post("/api/v1/valuation", json={"area": -1})
    assert response.status_code == 422


def test_comparables_endpoint():
    response = client.post("/api/v1/comparables/search", json={"bedrooms": 3, "bathrooms": 2, "area": 1800, "zipcode": "91901", "limit": 3})
    assert response.status_code == 200
    assert len(response.json()) <= 3
