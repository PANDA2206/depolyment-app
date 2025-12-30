from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_live():
    response = client.get("/health/live")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_recommendations_mock_strategy():
    payload = {
        "user_id": "test-user",
        "style_goals": ["evening"],
        "preferred_colors": ["black"],
        "budget": 500,
    }
    response = client.post("/api/v1/recommendations/", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "test-user"
    assert body["items"]
