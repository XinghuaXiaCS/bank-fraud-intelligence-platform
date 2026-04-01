"""Tests for feature engineering pipeline."""
import pandas as pd
import numpy as np
from src.feature_store import build_features, MODEL_FEATURES


def _make_sample_transactions():
    """Create a small sample transaction dataset for testing."""
    return pd.DataFrame({
        "account_id": [1, 1, 1, 2, 2],
        "timestamp": [
            "2025-01-01 10:00:00",
            "2025-01-01 10:15:00",  # 15 min after first
            "2025-01-01 10:30:00",  # 30 min after first
            "2025-01-02 14:00:00",
            "2025-01-02 14:05:00",
        ],
        "amount": [100, 200, 150, 80, 300],
        "device_seen_before": [1, 0, 1, 1, 0],
        "geo_seen_before": [1, 0, 1, 1, 1],
        "merchant_category": ["retail", "crypto", "retail", "grocery", "electronics"],
        "channel_risk": [0.1, 0.45, 0.1, 0.1, 0.45],
        "device_risk": [0.1, 0.8, 0.1, 0.1, 0.5],
        "num_linked_accounts": [1, 3, 1, 1, 2],
        "geo_distance_score": [0.1, 0.9, 0.1, 0.1, 0.5],
        "merchant_risk": [0.2, 0.8, 0.2, 0.1, 0.4],
        "failed_logins_24h": [0, 3, 0, 0, 1],
        "new_payee_recent": [0, 1, 0, 0, 0],
        "shared_counterparty_risk": [0.1, 0.7, 0.1, 0.1, 0.3],
        "device_id": [200000, 200001, 200000, 200002, 200003],
        "counterparty_account_id": [10001, 10002, 10003, 10004, 10005],
        "channel": ["card_present", "ecommerce", "card_present", "mobile_app", "ecommerce"],
        "geo_hash": ["AKL", "SYD", "AKL", "WLG", "MEL"],
        "label_fraud": [0, 1, 0, 0, 0],
        "label_ato": [0, 0, 0, 0, 0],
    })


def _make_sample_accounts():
    return pd.DataFrame({
        "account_id": [1, 2],
        "customer_id": [10, 20],
        "product_type": ["current", "savings"],
        "avg_monthly_turnover": [5000, 3000],
        "current_balance": [10000, 5000],
        "dormant_flag": [0, 0],
    })


def test_build_features_generates_expected_columns():
    """build_features should produce all MODEL_FEATURES columns."""
    tx = _make_sample_transactions()
    acc = _make_sample_accounts()
    out = build_features(tx, acc)
    for col in MODEL_FEATURES:
        assert col in out.columns, f"Missing feature: {col}"


def test_velocity_features_are_non_negative():
    """Velocity features should be zero or positive."""
    tx = _make_sample_transactions()
    acc = _make_sample_accounts()
    out = build_features(tx, acc)
    for col in ["txn_count_1h", "txn_count_24h", "txn_amount_sum_1h",
                "unique_merchants_1h", "unique_devices_24h", "unique_geos_24h"]:
        assert (out[col] >= 0).all(), f"{col} has negative values"


def test_velocity_features_are_time_aware():
    """Transactions within the same 1h window should accumulate counts."""
    tx = _make_sample_transactions()
    acc = _make_sample_accounts()
    out = build_features(tx, acc)
    # Account 1 has 3 transactions within 30 minutes
    acct1 = out[out["account_id"] == 1].sort_values("timestamp")
    # The third row should see at least 1 prior transaction in the 1h window
    assert acct1.iloc[2]["txn_count_1h"] >= 1


def test_log_amount_computed():
    """log_amount should be log1p of the amount."""
    tx = _make_sample_transactions()
    out = build_features(tx)
    assert "log_amount" in out.columns
    expected = np.log1p(out["amount"].iloc[0])
    assert abs(out["log_amount"].iloc[0] - expected) < 1e-6


def test_no_nan_in_model_features():
    """MODEL_FEATURES should not contain NaN after build_features."""
    tx = _make_sample_transactions()
    acc = _make_sample_accounts()
    out = build_features(tx, acc)
    for col in MODEL_FEATURES:
        assert out[col].notna().all(), f"{col} contains NaN values"
