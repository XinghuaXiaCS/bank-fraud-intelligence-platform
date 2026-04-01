"""Real-time scoring: load model artifacts and score a feature matrix.

Used by both the score_alerts script and, in a production setting,
by a low-latency inference service receiving single transactions.
"""
from __future__ import annotations

import joblib
import pandas as pd


def load_artifacts(model_dir: str) -> dict:
    """Load all trained model artifacts from disk."""
    return {
        "logit": joblib.load(f"{model_dir}/logit.joblib"),
        "hgb": joblib.load(f"{model_dir}/hgb.joblib"),
        "iforest": joblib.load(f"{model_dir}/iforest.joblib"),
    }


def score_batch(df: pd.DataFrame, models: dict, feature_names: list[str]) -> pd.DataFrame:
    """Score a dataframe with all loaded models.

    Adds columns: logit_score, hgb_score, anomaly_score.
    """
    out = df.copy()
    X = out[feature_names]
    out["logit_score"] = models["logit"].predict_proba(X)[:, 1]
    out["hgb_score"] = models["hgb"].predict_proba(X)[:, 1]
    out["anomaly_score"] = -models["iforest"].score_samples(X)
    return out
