"""Score the holdout test set and produce the alert queue.

Loads trained models, applies feature engineering, and outputs a
scored and prioritised alert file with reason codes.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.feature_store import build_features, MODEL_FEATURES
from src.models.graph_features import build_graph_features
from src.rules_engine import apply_rules, build_reason_codes
from src.scoring.realtime_scoring import load_artifacts, score_batch
from src.models.ranker import combine_scores, assign_actions


def minmax_norm(series: pd.Series) -> pd.Series:
    lo, hi = float(series.min()), float(series.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    reports_dir = Path(paths["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(paths["test_dataset"])
    accounts = pd.read_csv(paths["accounts_dataset"])
    customers = pd.read_csv(paths["customers_dataset"])

    df = build_features(test, accounts, customers)
    df = build_graph_features(df)
    df = apply_rules(df, cfg["rules"])

    models = load_artifacts(paths["model_dir"])
    df = score_batch(df, models, MODEL_FEATURES)
    df["anomaly_score_norm"] = minmax_norm(df["anomaly_score"])
    df["rule_score_norm"] = (df["rule_score"] / 4.0).clip(0, 1)

    # ATO scoring
    ato_path = Path(paths["model_dir"]) / "ato_logit.joblib"
    if ato_path.exists():
        ato_model = joblib.load(ato_path)
        df["ato_score"] = ato_model.predict_proba(df[MODEL_FEATURES])[:, 1]
    else:
        df["ato_score"] = 0.0

    df = combine_scores(df)
    df = assign_actions(df, cfg["thresholds"])
    df["reason_codes"] = df.apply(build_reason_codes, axis=1)

    keep = [
        "transaction_id", "account_id", "timestamp", "amount",
        "label_fraud", "label_ato",
        "logit_score", "hgb_score", "anomaly_score", "ato_score",
        "fraud_score", "investigation_priority_score",
        "recommended_action", "priority_band", "reason_codes",
    ]
    out = df.sort_values("investigation_priority_score", ascending=False)[keep]
    out.to_csv(reports_dir / "alerts_scored.csv", index=False)

    # Summary statistics
    print(f"Saved {len(out)} scored alerts to {reports_dir / 'alerts_scored.csv'}")
    print(f"  Actions: approve={int((out['recommended_action']=='approve').sum())} "
          f"review={int((out['recommended_action']=='review').sum())} "
          f"challenge={int((out['recommended_action']=='challenge').sum())}")
    print(f"  Priority: high={int((out['priority_band']=='high').sum())} "
          f"medium={int((out['priority_band']=='medium').sum())} "
          f"low={int((out['priority_band']=='low').sum())}")


if __name__ == "__main__":
    main()
