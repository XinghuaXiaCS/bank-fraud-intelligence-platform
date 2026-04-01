from __future__ import annotations

from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score


def classification_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    preds = (scores >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }
