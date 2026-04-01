# Error Analysis

Evaluation: 21/23 passed. Two failures below.

## Case 1: Mixed Polarity
**Sentence:** "The phone is fine, not great but not bad either."
- Expected: Neutral | Predicted: Positive (0.6558)
- Issue: Balanced sentiment with negations, but model focused on the positive opener ("fine") and missed the neutral intent despite contradictory cues.

## Case 2: Conflicting Sentiment
**Sentence:** "The camera quality is excellent, but the battery life is terrible."
- Expected: Neutral | Predicted: Negative (0.8238)
- Issue: Clear mixed sentiment (praise + critique), but model weighted the strong negative keyword ("terrible") as the dominant signal, ignoring the positive balance.
