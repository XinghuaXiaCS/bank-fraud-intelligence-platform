"""Score fusion and investigation prioritisation.

Combines rule hits, supervised model scores, and anomaly scores into a
single fraud risk score, then derives an investigation priority score
that also accounts for transaction severity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def combine_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Fuse individual model scores into a blended fraud score.

    Weights are chosen to balance:
      - hgb_score: highest discriminative power on labelled data
      - logit_score: well-calibrated baseline probability
      - anomaly_score_norm: captures novel patterns outside labelled data
      - rule_score_norm: deterministic business rules
    """
    out = df.copy()
    out["fraud_score"] = (
        0.40 * out["hgb_score"]
        + 0.25 * out["logit_score"]
        + 0.20 * out["anomaly_score_norm"]
        + 0.15 * out["rule_score_norm"]
    ).clip(0, 1)

    # Investigation priority accounts for both fraud likelihood and exposure
    amount_severity = np.clip(out["amount"] / 3000.0, 0, 1)
    out["investigation_priority_score"] = (
        0.70 * out["fraud_score"] + 0.30 * amount_severity
    ).clip(0, 1)

    return out


def assign_actions(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Assign recommended actions and priority bands based on thresholds."""
    out = df.copy()
    out["recommended_action"] = np.select(
        [
            out["fraud_score"] >= thresholds["challenge_score"],
            out["fraud_score"] >= thresholds["review_score"],
        ],
        ["challenge", "review"],
        default="approve",
    )
    out["priority_band"] = np.select(
        [
            out["investigation_priority_score"] >= thresholds["high_priority_score"],
            out["investigation_priority_score"] >= thresholds["review_score"],
        ],
        ["high", "medium"],
        default="low",
    )
    return out
