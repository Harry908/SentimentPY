# SentimentPY

FastAPI sentiment API using the Hugging Face model `cardiffnlp/twitter-roberta-base-sentiment-latest`.

## What This App Does

- Single prediction endpoint: `POST /sentiment`
- Batch prediction endpoint: `POST /sentiment/batch`
- Per-item validation with error isolation in batch mode
- Evaluation script that calls the API and writes a report
- Pytest test suite

## Quick Start

1. Create and activate virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Start API.

```bat
start_api.bat
```

Alternative:

```powershell
uvicorn app.main:app --reload
```

Docs URLs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## API Endpoints

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

Response:

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

## Evaluation

Run:

```powershell
python -m scripts.run_evaluation
```

## Tests

```powershell
python -m pytest -q
```

## Data and outputs:

- Test data folder: `data`
- Test data file: `data/test_sentences.csv`
- Results folder: `results`
- Evaluation output file: `results/evaluation_output.txt`
- App screenshots:
  - `results/single_statement_analysis.png`
  - `results/batch_statements_analysis.png`
- Video Demo: https://drive.google.com/file/d/1cEvitSH2sIpxEmZypj919S8rD4SeL99d/view?usp=sharing
