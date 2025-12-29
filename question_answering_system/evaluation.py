"""
Evaluation metrics for QA:
- Exact Match (EM)
- F1 Score
"""

def exact_match(pred, gold):
    return int(pred.strip().lower() == gold.strip().lower())

def f1_score(pred, gold):
    pred_words = pred.lower().split()
    gold_words = gold.lower().split()

    common = set(pred_words) & set(gold_words)
    if not common:
        return 0.0

    precision = len(common) / len(pred_words)
    recall = len(common) / len(gold_words)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
