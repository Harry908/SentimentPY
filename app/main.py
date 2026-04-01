from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StrictStr

from app.sentiment import batch_predict_sentiment, predict_sentiment

app = FastAPI(title="Sentiment API", version="1.0.0")


class SentimentRequest(BaseModel):
    """Single-item request payload."""

    text: StrictStr


class SentimentResponse(BaseModel):
    """Single-item successful response payload."""

    sentiment: str
    confidence: float


class BatchSentimentRequest(BaseModel):
    """Batch request payload."""

    texts: list[Any]


class BatchItemResult(BaseModel):
    """Per-item batch result with isolated success/error state."""

    index: int
    ok: bool
    input: str
    sentiment: str | None = None
    confidence: float | None = None
    error: str | None = None


class BatchSentimentResponse(BaseModel):
    """Batch response summary and ordered per-item results."""

    ok: bool
    total: int
    succeeded: int
    failed: int
    results: list[BatchItemResult]


def _error_response(message: str, status_code: int) -> JSONResponse:
    """Build a consistent error response shape for API clients."""

    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "error": {
                "message": message,
            },
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness probe endpoint."""

    return {"status": "ok"}


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(payload: SentimentRequest) -> SentimentResponse | JSONResponse:
    """Predict sentiment for a single input text."""

    try:
        result = predict_sentiment(payload.text)
        # Return only the simplified public fields.
        return SentimentResponse(**result.__dict__)
    except (TypeError, ValueError) as exc:
        return _error_response(str(exc), status_code=400)
    except Exception:
        return _error_response("Internal server error.", status_code=500)


@app.post("/sentiment/batch", response_model=BatchSentimentResponse)
def sentiment_batch(payload: BatchSentimentRequest) -> BatchSentimentResponse:
    """Predict sentiment for many items without aborting the full request."""

    # Sentiment layer handles per-item validation and failures.
    sentiment_results = batch_predict_sentiment(payload.texts)
    results = [BatchItemResult(**item.__dict__) for item in sentiment_results]
    succeeded = sum(1 for item in sentiment_results if item.ok)
    failed = len(sentiment_results) - succeeded
    return BatchSentimentResponse(
        ok=failed == 0,
        total=len(sentiment_results),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )
