from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier


def build_hgb(learning_rate: float = 0.07, max_depth: int = 6, max_iter: int = 180) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        random_state=42,
    )
