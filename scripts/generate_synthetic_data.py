"""Generate synthetic multi-table banking fraud data.

Tables produced:
  - customers
  - accounts
  - devices
  - login_events
  - transactions (train / test)
  - investigations
  - alerts
  - edges (entity graph)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import pandas as pd

from src.config import load_config


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------

def make_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "customer_segment": rng.choice(
            ["mass", "affluent", "sme"], size=n, p=[0.70, 0.20, 0.10]
        ),
        "kyc_level": rng.choice([1, 2, 3], size=n, p=[0.15, 0.65, 0.20]),
        "age_band": rng.choice(
            ["18_24", "25_34", "35_49", "50_plus"], size=n, p=[0.15, 0.35, 0.30, 0.20]
        ),
        "occupation_group": rng.choice(
            ["professional", "trade", "retail", "public_sector",
             "self_employed", "student", "retired"],
            size=n, p=[0.25, 0.15, 0.15, 0.15, 0.12, 0.10, 0.08],
        ),
        "residency_flag": rng.choice(["domestic", "overseas"], size=n, p=[0.92, 0.08]),
        "historical_risk_band": rng.choice(
            ["low", "medium", "high"], size=n, p=[0.82, 0.15, 0.03]
        ),
    })


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

def make_accounts(customers: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    cids = rng.choice(customers["customer_id"].values, size=n, replace=True)
    open_dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 5, size=n), unit="D"
    )
    return pd.DataFrame({
        "account_id": np.arange(10_000, 10_000 + n),
        "customer_id": cids,
        "product_type": rng.choice(
            ["current", "savings", "credit"], size=n, p=[0.60, 0.25, 0.15]
        ),
        "open_date": open_dates.strftime("%Y-%m-%d"),
        "current_balance": rng.gamma(2.5, 3000, size=n).round(2),
        "avg_monthly_turnover": rng.gamma(2.2, 1800, size=n).round(2),
        "dormant_flag": rng.choice([0, 1], size=n, p=[0.92, 0.08]),
    })


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------

def make_devices(n: int, rng: np.random.Generator) -> pd.DataFrame:
    ids = np.arange(200_000, 200_000 + n)
    return pd.DataFrame({
        "device_id": ids,
        "first_seen": (
            pd.to_datetime("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 365, size=n), unit="D")
        ).strftime("%Y-%m-%d"),
        "last_seen": (
            pd.to_datetime("2025-06-01")
            + pd.to_timedelta(rng.integers(0, 180, size=n), unit="D")
        ).strftime("%Y-%m-%d"),
        "num_linked_accounts": rng.choice(
            [1, 2, 3, 4, 5], size=n, p=[0.72, 0.17, 0.07, 0.03, 0.01]
        ),
        "risk_score_external": rng.beta(1.3, 5.5, size=n).round(4),
    })


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

MERCHANT_RISK = {
    "retail": 0.20, "travel": 0.35, "crypto": 0.80,
    "electronics": 0.40, "grocery": 0.10, "cash_transfer": 0.75,
}
CHANNEL_RISK = {
    "card_present": 0.10, "ecommerce": 0.45,
    "bank_transfer": 0.55, "mobile_app": 0.25,
}


def make_transactions(
    accounts: pd.DataFrame,
    devices: pd.DataFrame,
    n: int,
    start_date: str,
    fraud_rate: float,
    ato_rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    acct_ids = rng.choice(accounts["account_id"].values, size=n, replace=True)
    dev_ids = rng.choice(devices["device_id"].values, size=n, replace=True)
    dev_linked = devices.set_index("device_id")["num_linked_accounts"]
    dev_risk = devices.set_index("device_id")["risk_score_external"]

    mcats = rng.choice(list(MERCHANT_RISK.keys()), size=n,
                       p=[0.35, 0.08, 0.03, 0.12, 0.35, 0.07])
    chans = rng.choice(list(CHANNEL_RISK.keys()), size=n,
                       p=[0.28, 0.30, 0.18, 0.24])

    base_amt = rng.gamma(2.0, 80, size=n)
    risk_mult = np.array([MERCHANT_RISK[m] for m in mcats]) * rng.uniform(0.5, 2.5, n)
    amount = np.clip(base_amt + risk_mult * 500, 2, None)

    ts = pd.to_datetime(start_date) + pd.to_timedelta(
        rng.integers(0, 60 * 60 * 24 * 90, size=n), unit="s"
    )
    cp_ids = rng.choice(accounts["account_id"].values, size=n, replace=True)

    df = pd.DataFrame({
        "transaction_id": np.arange(1, n + 1),
        "account_id": acct_ids,
        "timestamp": ts,
        "amount": amount.round(2),
        "merchant_category": mcats,
        "merchant_risk": [MERCHANT_RISK[m] for m in mcats],
        "counterparty_account_id": cp_ids,
        "channel": chans,
        "channel_risk": [CHANNEL_RISK[c] for c in chans],
        "device_id": dev_ids,
        "device_risk": dev_risk.reindex(dev_ids).values,
        "geo_hash": rng.choice(["AKL", "WLG", "CHC", "DUD", "SYD", "MEL"],
                               size=n, p=[0.28, 0.20, 0.18, 0.08, 0.16, 0.10]),
        "geo_distance_score": rng.uniform(0, 1, n).round(4),
        "device_seen_before": rng.choice([0, 1], size=n, p=[0.10, 0.90]),
        "geo_seen_before": rng.choice([0, 1], size=n, p=[0.08, 0.92]),
        "failed_logins_24h": rng.poisson(0.3, n),
        "new_payee_recent": rng.choice([0, 1], size=n, p=[0.90, 0.10]),
        "shared_counterparty_risk": rng.beta(1.1, 6.0, n).round(4),
        "num_linked_accounts": dev_linked.reindex(dev_ids).values,
    })

    # Sort chronologically per account for velocity features downstream
    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
    df["transaction_id"] = np.arange(1, n + 1)

    # --- Fraud labels (calibrated to target rate) ---
    sig = (
        0.90 * (df["merchant_risk"] > 0.7).astype(float)
        + 1.00 * (df["channel_risk"] > 0.5).astype(float)
        + 1.10 * (df["device_seen_before"] == 0).astype(float)
        + 0.90 * (df["geo_seen_before"] == 0).astype(float)
        + 0.90 * (df["new_payee_recent"] == 1).astype(float)
        + 0.70 * (df["num_linked_accounts"] >= 3).astype(float)
        + 0.0008 * df["amount"]
        + 0.20 * df["failed_logins_24h"]
    )
    prob = 1.0 / (1.0 + np.exp(-(sig - 4.2)))
    raw = prob.mean()
    calibrated = np.clip(prob * (fraud_rate / max(raw, 1e-8)), 0, 0.85)
    df["label_fraud"] = rng.binomial(1, calibrated)

    # Boost known high-risk combos slightly
    hot = (df["merchant_category"].isin(["crypto", "cash_transfer"])) & (df["device_seen_before"] == 0)
    df.loc[hot & (rng.random(n) < 0.12), "label_fraud"] = 1

    # --- ATO labels (calibrated to target rate) ---
    ato_sig = (
        1.10 * (df["device_seen_before"] == 0).astype(float)
        + 1.00 * (df["geo_seen_before"] == 0).astype(float)
        + 0.40 * df["failed_logins_24h"]
        + 0.80 * (df["channel"] == "mobile_app").astype(float)
    )
    ato_prob = 1.0 / (1.0 + np.exp(-(ato_sig - 3.8)))
    raw_ato = ato_prob.mean()
    cal_ato = np.clip(ato_prob * (ato_rate / max(raw_ato, 1e-8)), 0, 0.80)
    df["label_ato"] = rng.binomial(1, cal_ato)

    return df


# ---------------------------------------------------------------------------
# Login events
# ---------------------------------------------------------------------------

def make_login_events(
    accounts: pd.DataFrame,
    devices: pd.DataFrame,
    n: int,
    start_date: str,
    ato_rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    acct_ids = rng.choice(accounts["account_id"].values, size=n, replace=True)
    dev_ids = rng.choice(devices["device_id"].values, size=n, replace=True)
    ts = pd.to_datetime(start_date) + pd.to_timedelta(
        rng.integers(0, 60 * 60 * 24 * 90, size=n), unit="s"
    )
    success = rng.choice([0, 1], size=n, p=[0.08, 0.92])
    pw_reset = rng.choice([0, 1], size=n, p=[0.94, 0.06])
    dev_seen = rng.choice([0, 1], size=n, p=[0.12, 0.88])

    sig = (
        1.20 * (dev_seen == 0).astype(float)
        + 0.80 * (pw_reset == 1).astype(float)
        + 0.60 * (success == 0).astype(float)
    )
    prob = 1.0 / (1.0 + np.exp(-(sig - 2.5)))
    raw = prob.mean()
    cal = np.clip(prob * (ato_rate / max(raw, 1e-8)), 0, 0.80)
    label_ato = rng.binomial(1, cal)

    return pd.DataFrame({
        "login_id": np.arange(1, n + 1),
        "account_id": acct_ids,
        "timestamp": ts,
        "success_flag": success,
        "device_id": dev_ids,
        "ip_address": [
            f"10.{rng.integers(0,256)}.{rng.integers(0,256)}.{rng.integers(0,256)}"
            for _ in range(n)
        ],
        "geo_hash": rng.choice(["AKL", "WLG", "CHC", "DUD", "SYD", "MEL"],
                               size=n, p=[0.28, 0.20, 0.18, 0.08, 0.16, 0.10]),
        "browser_fingerprint": [f"fp_{rng.integers(10000,99999)}" for _ in range(n)],
        "password_reset_recent": pw_reset,
        "device_seen_before": dev_seen,
        "label_ato": label_ato,
    })


# ---------------------------------------------------------------------------
# Investigations
# ---------------------------------------------------------------------------

def make_investigations(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    base = df[["transaction_id", "account_id", "amount", "label_fraud", "label_ato"]].copy()
    suspicion = 0.6 * base["label_fraud"] + 0.2 * base["label_ato"] + 0.2 * np.clip(base["amount"] / 2500, 0, 1)
    p = np.clip(suspicion + rate, 0, 0.95)
    mask = rng.random(len(base)) < p
    base = base[mask].copy()
    n = len(base)
    base["case_id"] = np.arange(50_000, 50_000 + n)
    base["review_outcome"] = np.where(
        base["label_fraud"] == 1, "substantiated",
        rng.choice(["not_substantiated", "monitor"], size=n, p=[0.70, 0.30]),
    )
    base["confirmed_loss"] = np.where(base["label_fraud"] == 1, (0.4 * base["amount"]).round(2), 0.0)
    base["recovery_amount"] = np.where(base["label_fraud"] == 1, (0.25 * base["amount"]).round(2), 0.0)
    base["investigator_team"] = rng.choice(["L1_auto", "L2_analyst", "L3_specialist"], size=n, p=[0.55, 0.35, 0.10])
    return base[["case_id", "transaction_id", "account_id", "review_outcome",
                  "confirmed_loss", "recovery_amount", "investigator_team"]]


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def make_alerts(txn: pd.DataFrame, inv: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    flagged = txn[
        (txn["label_fraud"] == 1) | (txn["label_ato"] == 1) | (rng.random(len(txn)) < 0.06)
    ].drop_duplicates("transaction_id").copy()
    n = len(flagged)

    trigger = np.where(
        flagged["label_fraud"].values == 1,
        rng.choice(["rule_engine", "ml_model", "anomaly_detector"], size=n, p=[0.30, 0.50, 0.20]),
        rng.choice(["rule_engine", "ml_model", "anomaly_detector"], size=n, p=[0.50, 0.30, 0.20]),
    )
    score = rng.beta(2.0, 3.0, n).round(4)
    score = np.where(flagged["label_fraud"].values == 1, np.clip(score + 0.3, 0, 1), score)

    priority = np.select([score >= 0.7, score >= 0.4], ["high", "medium"], default="low")
    inv_set = set(inv["transaction_id"].values)
    outcome = [
        "investigated" if tid in inv_set else rng.choice(["auto_closed", "pending"])
        for tid in flagged["transaction_id"].values
    ]
    return pd.DataFrame({
        "alert_id": np.arange(80_000, 80_000 + n),
        "entity_type": "transaction",
        "entity_id": flagged["transaction_id"].values,
        "event_time": flagged["timestamp"].values,
        "trigger_source": trigger,
        "model_score": score.round(4),
        "alert_priority": priority,
        "review_outcome": outcome,
        "recovery_amount": np.where(flagged["label_fraud"].values == 1, (0.2 * flagged["amount"].values).round(2), 0.0),
    })


# ---------------------------------------------------------------------------
# Edges (entity graph)
# ---------------------------------------------------------------------------

def make_edges(txn: pd.DataFrame, logins: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []

    for _, r in txn[["account_id", "device_id"]].drop_duplicates().iterrows():
        rows.append(("account", int(r["account_id"]), "device", int(r["device_id"]), "account_to_device"))

    cp = txn[["account_id", "counterparty_account_id"]].drop_duplicates()
    cp = cp.sample(n=min(5000, len(cp)), random_state=42)
    for _, r in cp.iterrows():
        rows.append(("account", int(r["account_id"]), "account", int(r["counterparty_account_id"]), "transaction_to_counterparty"))

    for _, r in logins[["account_id", "device_id"]].drop_duplicates().iterrows():
        rows.append(("account", int(r["account_id"]), "device", int(r["device_id"]), "login_to_device"))

    edges = pd.DataFrame(rows, columns=["src_entity_type", "src_entity_id", "dst_entity_type", "dst_entity_id", "edge_type"])
    edges = edges.drop_duplicates()
    edges["weight"] = rng.uniform(0.1, 1.0, len(edges)).round(4)
    return edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["random_seed"]
    rng = np.random.default_rng(seed)
    paths = cfg["paths"]
    synth = cfg["synthetic_data"]
    out_dir = Path(paths["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    customers = make_customers(synth["n_customers"], rng)
    accounts = make_accounts(customers, synth["n_accounts"], rng)
    devices = make_devices(synth.get("n_devices", 2500), rng)

    train = make_transactions(accounts, devices, synth["n_transactions_train"],
                              synth["start_date"], synth["fraud_rate"], synth["ato_rate"], rng)
    test = make_transactions(accounts, devices, synth["n_transactions_test"],
                             synth["test_start_date"], synth["fraud_rate"], synth["ato_rate"], rng)

    login_events = make_login_events(accounts, devices, synth.get("n_login_events", 8000),
                                     synth["start_date"], synth["ato_rate"], rng)

    investigations = make_investigations(train, synth["investigation_rate"], rng)
    alerts = make_alerts(train, investigations, rng)
    edges = make_edges(train, login_events, rng)

    customers.to_csv(paths["customers_dataset"], index=False)
    accounts.to_csv(paths["accounts_dataset"], index=False)
    devices.to_csv(paths["devices_dataset"], index=False)
    train.to_csv(paths["train_dataset"], index=False)
    test.to_csv(paths["test_dataset"], index=False)
    login_events.to_csv(paths["login_events_dataset"], index=False)
    investigations.to_csv(paths["investigations_dataset"], index=False)
    alerts.to_csv(paths["alerts_dataset"], index=False)
    edges.to_csv(paths["edges_dataset"], index=False)

    print(f"Saved synthetic datasets to {out_dir.resolve()}")
    print({
        "train_rows": len(train),
        "test_rows": len(test),
        "train_fraud_rate": round(float(train["label_fraud"].mean()), 4),
        "test_fraud_rate": round(float(test["label_fraud"].mean()), 4),
        "train_ato_rate": round(float(train["label_ato"].mean()), 4),
        "login_events": len(login_events),
        "alerts": len(alerts),
        "edges": len(edges),
    })


if __name__ == "__main__":
    main()
