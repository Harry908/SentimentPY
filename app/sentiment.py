from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from transformers import pipeline

import torch

@dataclass
class SentimentResult:
    """Public sentiment response used by API and scripts."""

    sentiment: str
    confidence: float


@dataclass
class BatchSentimentResult:
    """Per-item batch outcome produced by the sentiment layer."""

    index: int
    ok: bool
    input: str
    sentiment: str | None = None
    confidence: float | None = None
    error: str | None = None


def validate_text_input(text: Any) -> str:
    """Validate incoming text and enforce a minimal linguistic signal."""

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Remove outer whitespace before checks.
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input text cannot be empty.")

    # Reject numeric/symbol-only payloads so callers get a clear validation error.
    if not any(char.isalpha() for char in cleaned):
        raise ValueError("Input text must contain at least one alphabetic character.")

    return cleaned


@lru_cache(maxsize=1)
def get_classifier():
    """Create and cache the Hugging Face sentiment classifier."""

    # Use a native 3-class model (negative/neutral/positive).
    # Choose GPU when available, otherwise fallback to CPU.
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device,
    )


def _prediction_to_result(prediction: dict[str, Any]) -> SentimentResult:
    """Convert model output into the simplified public result shape."""

    # Keep label as-is from the model and only standardize confidence precision.
    raw_label = str(prediction["label"])
    raw_score = float(prediction["score"])
    return SentimentResult(
        sentiment=raw_label,
        confidence=round(raw_score, 4),
    )


def _safe_input_repr(value: Any) -> str:
    """Create a safe printable representation for non-string inputs."""

    if isinstance(value, str):
        return value
    return repr(value)


def predict_sentiment(text: Any) -> SentimentResult:
    """Predict sentiment for a single input string."""
    # Validate once before model inference.
    clean_text = validate_text_input(text)
    classifier = get_classifier()
    prediction = classifier(clean_text)[0]
    return _prediction_to_result(prediction)


def batch_predict_sentiment(texts: list[Any]) -> list[BatchSentimentResult]:
    """Predict sentiment for many inputs with per-item validation."""

    results: list[BatchSentimentResult] = []
    items_by_index: dict[int, BatchSentimentResult] = {}
    valid_indices: list[int] = []
    valid_texts: list[str] = []

    # Validate inputs once here so API layer can stay thin.
    for idx, value in enumerate(texts):
        try:
            cleaned = validate_text_input(value)
            valid_indices.append(idx)
            valid_texts.append(cleaned)
            item = BatchSentimentResult(index=idx, ok=True, input=cleaned)
            results.append(item)
            items_by_index[idx] = item
        except (TypeError, ValueError) as exc:
            results.append(
                BatchSentimentResult(
                    index=idx,
                    ok=False,
                    input=_safe_input_repr(value),
                    error=str(exc),
                )
            )

    if valid_texts:
        try:
            classifier = get_classifier()
            predictions = classifier(valid_texts)
            if len(predictions) != len(valid_texts):
                raise RuntimeError("Prediction count does not match valid input count.")

            for idx, prediction in zip(valid_indices, predictions):
                item = items_by_index.get(idx)
                if item is None:
                    continue
                result = _prediction_to_result(prediction)
                item.sentiment = result.sentiment
                item.confidence = result.confidence
        except Exception:
            # Keep behavior simple: mark all previously-valid items as failed.
            for idx in valid_indices:
                item = items_by_index.get(idx)
                if item is None:
                    continue
                item.ok = False
                item.error = "Internal server error."

    return results
