from __future__ import annotations

import csv
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

DATA_FILE = Path("data/test_sentences.csv")
OUTPUT_FILE = Path("results/evaluation_output.txt")


def run() -> None:
    source_rows = []
    client = TestClient(app)

    with DATA_FILE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_rows.append(row)

    texts = [row["sentence"] for row in source_rows]
    batch_payload = client.post("/sentiment/batch", json={"texts": texts}).json()
    batch_results = batch_payload["results"]

    rows = []
    for row, item in zip(source_rows, batch_results):
        sentence = row["sentence"]
        expected = row["expected"]

        predicted = str(item["sentiment"]) if item["ok"] else "INPUT_ERROR"
        confidence = f"{float(item['confidence'] or 0.0):.4f}"

        passed = predicted.lower() == expected.lower()
        rows.append(
            {
                "sentence": sentence,
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
                "pass": "PASS" if passed else "FAIL",
            }
        )

    total = len(rows)
    passed_count = sum(1 for row in rows if row["pass"] == "PASS")

    lines = []
    lines.append("Sentiment Evaluation Results")
    lines.append("=" * 80)
    for idx, row in enumerate(rows, start=1):
        lines.append(f"{idx:02d}. {row['sentence']}")
        lines.append(
            f"    expected={row['expected']:<8} predicted={row['predicted']:<8} "
            f"confidence={row['confidence']} result={row['pass']}"
        )
    lines.append("=" * 80)
    lines.append(f"Total: {total}")
    lines.append(f"Passed: {passed_count}")
    lines.append(f"Failed: {total - passed_count}")

    report = "\n".join(lines)
    print(report)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report + "\n", encoding="utf-8")

if __name__ == "__main__":
    run()
