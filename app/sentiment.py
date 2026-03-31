from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from transformers import pipeline

import torch

@dataclass
class SentimentResult:
    sentiment: str
    confidence: float
    raw_label: str
    raw_score: float


def validate_text_input(text: Any) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input text cannot be empty.")

    return cleaned


@lru_cache(maxsize=1)
def get_classifier():
    # Use a native 3-class model (negative/neutral/positive).
    device = 0 if torch.cuda.is_available() else -1 # Use GPU if available, else CPU.
    return pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device
    )


def normalize_sentiment(raw_label: str, raw_score: float) -> str:
    del raw_score  # Not needed for direct 3-class label mapping.

    label = raw_label.strip().upper()
    label_map = {
        "NEGATIVE": "Negative",
        "NEUTRAL": "Neutral",
        "POSITIVE": "Positive",
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive",
    }
    return label_map.get(label, "Neutral")


def predict_sentiment(text: Any) -> SentimentResult:
    clean_text = validate_text_input(text)
    classifier = get_classifier()
    prediction = classifier(clean_text)[0]

    raw_label = str(prediction["label"])
    raw_score = float(prediction["score"])
    sentiment = normalize_sentiment(raw_label, raw_score)

    return SentimentResult(
        sentiment=sentiment,
        confidence=round(raw_score, 4),
        raw_label=raw_label,
        raw_score=round(raw_score, 4),
    )
