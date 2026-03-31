from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StrictStr

from app.sentiment import predict_sentiment, validate_text_input

app = FastAPI(title="Sentiment API", version="1.0.0")


class SentimentRequest(BaseModel):
    text: StrictStr


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    raw_label: str
    raw_score: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(payload: SentimentRequest) -> SentimentResponse:
    try:
        validate_text_input(payload.text)
        result = predict_sentiment(payload.text)
        return SentimentResponse(**result.__dict__)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
