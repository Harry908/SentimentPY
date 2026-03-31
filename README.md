# SentimentPY

A simple FastAPI sentiment API using a Hugging Face model.

Model used: `cardiffnlp/twitter-roberta-base-sentiment-latest` (native 3-class sentiment output).

## Features

- Single prediction endpoint: `POST /sentiment`
- Batch prediction endpoint: `POST /sentiment/batch`
- Per-item validation and error isolation in batch mode
- Evaluation script that calls the API and writes a report
- Pytest suite in one file

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

Or use the helper script:

```bat
start_api.bat
```

`--reload` is for development only (auto-restarts on file changes).

Open interactive docs at: `http://127.0.0.1:8000/docs`

## API Summary

### `POST /sentiment`

Request:

```json
{
  "text": "I love this product"
}
```

Success response:

```json
{
  "sentiment": "positive",
  "confidence": 0.9998
}
```

Validation error response:

```json
{
  "ok": false,
  "error": {
    "message": "Input text must contain at least one alphabetic character."
  }
}
```

### `POST /sentiment/batch`

Request:

```json
{
  "texts": ["I love this", "1233323", "This is bad"]
}
```

Response (per-item success/error):

```json
{
  "ok": false,
  "total": 3,
  "succeeded": 2,
  "failed": 1,
  "results": [
    {"index": 0, "ok": true, "input": "I love this", "sentiment": "positive", "confidence": 0.99, "error": null},
    {"index": 1, "ok": false, "input": "1233323", "sentiment": null, "confidence": null, "error": "Input text must contain at least one alphabetic character."},
    {"index": 2, "ok": true, "input": "This is bad", "sentiment": "negative", "confidence": 0.93, "error": null}
  ]
}
```

## Run Evaluation

```powershell
python -m scripts.run_evaluation
```

This prints results and writes `results/evaluation_output.txt`.

## Run Tests

```powershell
python -m pytest -q
```

