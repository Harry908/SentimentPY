# SentimentPY

A small FastAPI project that predicts text sentiment (Positive, Negative, Neutral) using a Hugging Face model.

Model used: `cardiffnlp/twitter-roberta-base-sentiment-latest` (native 3-class sentiment output).

## Objective coverage

- Accepts text input and returns sentiment.
- Uses one NLP model/library: Hugging Face Transformers pipeline.
- Includes basic input checks: empty input and non-string input.
- Evaluates at least 12 sentences (4 positive, 4 negative, 4 neutral).
- Includes short analysis of 2 incorrect/uncertain predictions.

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

## Run API

```powershell
uvicorn app.main:app --reload
```

Open interactive docs at: `http://127.0.0.1:8000/docs`

### Example request

```json
POST /sentiment
{
  "text": "I love this product"
}
```

### Example response

```json
{
  "sentiment": "Positive",
  "confidence": 0.9998,
  "raw_label": "POSITIVE",
  "raw_score": 0.9998
}
```

## Run 12-sentence evaluation

```powershell
python -m scripts.run_evaluation
```

This prints results and saves output to `results/evaluation_output.txt`.

## Run validation tests

```powershell
pytest -q
```

