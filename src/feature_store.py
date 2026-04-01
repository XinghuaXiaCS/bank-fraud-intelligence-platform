"""Feature engineering for the fraud intelligence platform.

Builds transaction-level features including:
  - basic amount and time features
  - time-window velocity features (1h, 24h)
  - behavioural deviation features
  - account/customer profile features
  - placeholder graph features (populated by graph_features module)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .preprocessing import add_time_features, safe_log1p


def _velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute real time-window velocity features per account.

    Uses pandas rolling on a DatetimeIndex to count events and sum amounts
    within actual 1-hour and 24-hour lookback windows.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

    # Per-account rolling aggregations on timestamp index
    results_1h = []
    results_24h = []
    amt_1h = []
    merchant_1h = []
    device_24h = []
    geo_24h = []

    # Encode categorical columns as integers for rolling uniqueness counts
    merch_codes = out["merchant_category"].astype("category").cat.codes.astype(float)
    dev_codes = out["device_id"].astype(float)
    geo_codes = out["geo_hash"].astype("category").cat.codes.astype(float)

    for _, grp in out.groupby("account_id"):
        idx = grp.index
        g = grp.set_index("timestamp").sort_index()

        # Transaction count and amount sum in trailing windows
        count_1h = g["amount"].rolling("1h", closed="left").count().fillna(0)
        count_24h = g["amount"].rolling("24h", closed="left").count().fillna(0)
        sum_1h = g["amount"].rolling("1h", closed="left").sum().fillna(0)

        # Unique merchants in trailing 1h (use encoded integers)
        g_merch = merch_codes.loc[idx].copy()
        g_merch.index = g.index
        unique_merch = g_merch.rolling("1h", closed="left").apply(
            lambda x: len(set(x)) if len(x) > 0 else 0, raw=False
        ).fillna(0)

        # Unique devices in trailing 24h
        g_dev = dev_codes.loc[idx].copy()
        g_dev.index = g.index
        unique_dev = g_dev.rolling("24h", closed="left").apply(
            lambda x: len(set(x)) if len(x) > 0 else 0, raw=False
        ).fillna(0)

        # Unique geos in trailing 24h
        g_geo = geo_codes.loc[idx].copy()
        g_geo.index = g.index
        unique_geo = g_geo.rolling("24h", closed="left").apply(
            lambda x: len(set(x)) if len(x) > 0 else 0, raw=False
        ).fillna(0)

        results_1h.append(count_1h.reset_index(drop=True))
        results_24h.append(count_24h.reset_index(drop=True))
        amt_1h.append(sum_1h.reset_index(drop=True))
        merchant_1h.append(unique_merch.reset_index(drop=True))
        device_24h.append(unique_dev.reset_index(drop=True))
        geo_24h.append(unique_geo.reset_index(drop=True))

    out["txn_count_1h"] = pd.concat(results_1h, ignore_index=True).values.astype(float)
    out["txn_count_24h"] = pd.concat(results_24h, ignore_index=True).values.astype(float)
    out["txn_amount_sum_1h"] = pd.concat(amt_1h, ignore_index=True).values.astype(float)
    out["unique_merchants_1h"] = pd.concat(merchant_1h, ignore_index=True).values.astype(float)
    out["unique_devices_24h"] = pd.concat(device_24h, ignore_index=True).values.astype(float)
    out["unique_geos_24h"] = pd.concat(geo_24h, ignore_index=True).values.astype(float)

    return out


def build_features(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame | None = None,
    customers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the full feature matrix from raw transaction data."""
    df = add_time_features(transactions)
    df = df.sort_values(["account_id", "timestamp"]).copy()

    # Merge account-level context
    if accounts is not None:
        merge_cols = ["account_id"]
        avail = [c for c in ["customer_id", "product_type", "avg_monthly_turnover",
                              "current_balance", "dormant_flag"] if c in accounts.columns]
        df = df.merge(accounts[merge_cols + avail], on="account_id", how="left")

    # Merge customer-level context
    if customers is not None and "customer_id" in df.columns:
        cust_cols = [c for c in ["customer_id", "customer_segment", "age_band",
                                  "historical_risk_band"] if c in customers.columns]
        df = df.merge(customers[cust_cols], on="customer_id", how="left")

    # --- Basic features ---
    df["log_amount"] = safe_log1p(df["amount"])
    df["is_new_device"] = (df["device_seen_before"] == 0).astype(int)
    df["is_new_geo"] = (df["geo_seen_before"] == 0).astype(int)
    turnover = df["avg_monthly_turnover"].fillna(5000) if "avg_monthly_turnover" in df.columns else 5000
    df["high_amount_vs_turnover"] = (df["amount"] / (turnover + 1)).clip(0, 5)

    # --- Velocity features (real time windows) ---
    df = _velocity_features(df)

    # --- Historical deviation features ---
    df["amount_rolling_mean"] = df.groupby("account_id")["amount"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    df["amount_rolling_std"] = df.groupby("account_id")["amount"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).std()
    ).fillna(0)
    df["amount_z"] = (
        (df["amount"] - df["amount_rolling_mean"].fillna(df["amount"].median()))
        / (df["amount_rolling_std"] + 1.0)
    )
    df["peer_amount_ratio"] = df["amount"] / (
        df.groupby("merchant_category")["amount"].transform("median") + 1.0
    )

    # --- Placeholder graph features (populated by graph_features module) ---
    for col in ["graph_degree", "graph_neighbor_fraud_ratio",
                "graph_pagerank", "graph_community_fraud_ratio"]:
        if col not in df.columns:
            df[col] = 0.0

    return df


# Features used by supervised models
MODEL_FEATURES = [
    "amount",
    "log_amount",
    "channel_risk",
    "device_risk",
    "is_new_device",
    "is_new_geo",
    "txn_count_1h",
    "txn_count_24h",
    "txn_amount_sum_1h",
    "unique_merchants_1h",
    "unique_devices_24h",
    "unique_geos_24h",
    "num_linked_accounts",
    "geo_distance_score",
    "high_amount_vs_turnover",
    "amount_z",
    "peer_amount_ratio",
    "hour",
    "dayofweek",
    "is_weekend",
    "merchant_risk",
    "failed_logins_24h",
    "new_payee_recent",
    "shared_counterparty_risk",
    "graph_degree",
    "graph_neighbor_fraud_ratio",
    "graph_pagerank",
    "graph_community_fraud_ratio",
]
