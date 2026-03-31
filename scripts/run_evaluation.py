from __future__ import annotations

import csv
from pathlib import Path

from app.sentiment import predict_sentiment

DATA_FILE = Path("data/test_sentences.csv")
OUTPUT_FILE = Path("results/evaluation_output.txt")


def run() -> None:
    rows = []
    with DATA_FILE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sentence = row["sentence"]
            expected = row["expected"]
            prediction = predict_sentiment(sentence)
            passed = prediction.sentiment.lower() == expected.lower()
            rows.append(
                {
                    "sentence": sentence,
                    "expected": expected,
                    "predicted": prediction.sentiment,
                    "confidence": f"{prediction.confidence:.4f}",
                    "raw_label": prediction.raw_label,
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
            f"confidence={row['confidence']} raw={row['raw_label']} result={row['pass']}"
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
