# Error/Uncertain Prediction Analysis

This project uses a native 3-class model: cardiffnlp/twitter-roberta-base-sentiment-latest. 
In the latest 12-sentence run, 11 predictions passed and 1 failed. Below are two useful analysis cases: one incorrect and one uncertain.

## Case 1
Sentence: "The phone is fine, not great but not bad either."

- Expected: Neutral
- Predicted: Positive
- Confidence: 0.6558
- Why it failed: The sentence contains balanced wording with mild contrast ("not great" and "not bad"), but the model leaned slightly toward positive sentiment.
- Mitigation: Fine-tune on domain-specific neutral/mixed examples, or apply a post-processing rule for mixed polarity phrases.

## Case 2
Sentence: "The device stopped working after two days."

- Expected: Negative
- Predicted: Negative
- Confidence: 0.7052
- Why uncertain: Prediction is correct, but confidence is lower than other negative examples, suggesting wording sensitivity around failure descriptions.
- Mitigation: Expand evaluation with more product-failure phrasings and monitor confidence distribution to define an uncertainty threshold.
