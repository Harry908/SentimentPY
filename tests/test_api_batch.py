from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_sentiment_batch_continues_on_invalid_inputs() -> None:
    response = client.post(
        "/sentiment/batch",
        json={"texts": ["I love this", "1233323", "!@#$%^&*()", "This is bad"]},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["total"] == 4
    assert payload["succeeded"] == 2
    assert payload["failed"] == 2
    assert len(payload["results"]) == 4

    first = payload["results"][0]
    assert first["ok"] is True
    assert str(first["sentiment"]).lower() in {"positive", "neutral", "negative"}

    second = payload["results"][1]
    assert second["ok"] is False
    assert "alphabetic" in second["error"].lower()

    third = payload["results"][2]
    assert third["ok"] is False
    assert "alphabetic" in third["error"].lower()

    fourth = payload["results"][3]
    assert fourth["ok"] is True
    assert str(fourth["sentiment"]).lower() in {"positive", "neutral", "negative"}


def test_sentiment_endpoint_returns_structured_error() -> None:
    response = client.post("/sentiment", json={"text": "1233323"})

    assert response.status_code == 400
    payload = response.json()
    assert payload["ok"] is False
    assert "error" in payload
    assert "message" in payload["error"]
