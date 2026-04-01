from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def build_logistic(C: float = 1.0, max_iter: int = 500) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced", solver="liblinear")
