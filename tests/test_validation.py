from app.sentiment import predict_sentiment, validate_text_input


def test_validate_text_input_rejects_empty() -> None:
    try:
        validate_text_input("   ")
        assert False, "Expected ValueError for empty input"
    except ValueError:
        assert True


def test_predict_sentiment_rejects_non_string() -> None:
    try:
        predict_sentiment(123)
        assert False, "Expected TypeError for non-string input"
    except TypeError:
        assert True
