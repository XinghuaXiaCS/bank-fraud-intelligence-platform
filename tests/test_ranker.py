"""Tests for score fusion and investigation prioritisation."""
import pandas as pd
import numpy as np
from src.models.ranker import combine_scores, assign_actions


def _make_scored_data():
    return pd.DataFrame({
        "hgb_score": [0.1, 0.5, 0.9],
        "logit_score": [0.2, 0.4, 0.8],
        "anomaly_score_norm": [0.1, 0.3, 0.7],
        "rule_score_norm": [0.0, 0.25, 1.0],
        "amount": [50, 500, 5000],
    })


def test_combine_scores_range():
    """fraud_score and investigation_priority_score should be in [0, 1]."""
    df = _make_scored_data()
    out = combine_scores(df)
    assert (out["fraud_score"] >= 0).all() and (out["fraud_score"] <= 1).all()
    assert (out["investigation_priority_score"] >= 0).all() and (out["investigation_priority_score"] <= 1).all()


def test_high_risk_gets_higher_score():
    """The highest-risk row should get the highest fraud score."""
    df = _make_scored_data()
    out = combine_scores(df)
    assert out["fraud_score"].iloc[2] > out["fraud_score"].iloc[0]


def test_assign_actions_produces_expected_columns():
    """assign_actions should create recommended_action and priority_band."""
    df = combine_scores(_make_scored_data())
    thresholds = {"review_score": 0.30, "challenge_score": 0.55, "high_priority_score": 0.60}
    out = assign_actions(df, thresholds)
    assert "recommended_action" in out.columns
    assert "priority_band" in out.columns


def test_assign_actions_low_score_approved():
    """Low-scoring transactions should be approved."""
    df = combine_scores(_make_scored_data())
    thresholds = {"review_score": 0.30, "challenge_score": 0.55, "high_priority_score": 0.60}
    out = assign_actions(df, thresholds)
    assert out.iloc[0]["recommended_action"] == "approve"
