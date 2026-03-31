import pytest
from fastapi.testclient import TestClient

from app import main
from app.sentiment import BatchSentimentResult, SentimentResult, predict_sentiment, validate_text_input


client = TestClient(main.app)


def test_validate_text_input_rejects_empty() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_text_input("   ")


def test_predict_sentiment_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        predict_sentiment(123)


@pytest.mark.parametrize("value", ["1233323", "!@#$%^&*()", "___"])
def test_validate_text_input_rejects_non_alphabetic_only(value: str) -> None:
    with pytest.raises(ValueError, match="alphabetic"):
        validate_text_input(value)


def test_validate_text_input_accepts_linguistic_text() -> None:
    assert validate_text_input(" Hi there! ") == "Hi there!"


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_sentiment_batch_continues_on_invalid_inputs(monkeypatch) -> None:
    def fake_batch_predict(texts):
        assert texts == ["I love this", "1233323", "!@#$%^&*()", "This is bad"]
        return [
            BatchSentimentResult(index=0, ok=True, input="I love this", sentiment="positive", confidence=0.9),
            BatchSentimentResult(
                index=1,
                ok=False,
                input="1233323",
                error="Input text must contain at least one alphabetic character.",
            ),
            BatchSentimentResult(
                index=2,
                ok=False,
                input="!@#$%^&*()",
                error="Input text must contain at least one alphabetic character.",
            ),
            BatchSentimentResult(index=3, ok=True, input="This is bad", sentiment="negative", confidence=0.88),
        ]

    monkeypatch.setattr(main, "batch_predict_sentiment", fake_batch_predict)

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
    assert first["sentiment"] == "positive"
    assert first["error"] is None

    second = payload["results"][1]
    assert second["ok"] is False
    assert "alphabetic" in second["error"].lower()
    assert second["sentiment"] is None

    third = payload["results"][2]
    assert third["ok"] is False
    assert "alphabetic" in third["error"].lower()
    assert third["sentiment"] is None

    fourth = payload["results"][3]
    assert fourth["ok"] is True
    assert fourth["sentiment"] == "negative"
    assert fourth["error"] is None


def test_sentiment_endpoint_success(monkeypatch) -> None:
    monkeypatch.setattr(main, "predict_sentiment", lambda text: SentimentResult(sentiment="neutral", confidence=0.77))

    response = client.post("/sentiment", json={"text": "All good"})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "neutral", "confidence": 0.77}


def test_sentiment_endpoint_returns_structured_error(monkeypatch) -> None:
    def fake_predict(_):
        raise ValueError("Input text must contain at least one alphabetic character.")

    monkeypatch.setattr(main, "predict_sentiment", fake_predict)

    response = client.post("/sentiment", json={"text": "1233323"})

    assert response.status_code == 400
    payload = response.json()
    assert payload["ok"] is False
    assert "error" in payload
    assert "message" in payload["error"]


def test_sentiment_endpoint_returns_internal_error(monkeypatch) -> None:
    def fake_predict(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "predict_sentiment", fake_predict)

    response = client.post("/sentiment", json={"text": "Hello"})
    assert response.status_code == 500
    assert response.json() == {
        "ok": False,
        "error": {"message": "Internal server error."},
    }
