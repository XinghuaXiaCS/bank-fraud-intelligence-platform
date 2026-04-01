from __future__ import annotations

from sklearn.ensemble import IsolationForest


def build_isolation_forest(contamination: float = 0.04, n_estimators: int = 150) -> IsolationForest:
    return IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=42,
    )
