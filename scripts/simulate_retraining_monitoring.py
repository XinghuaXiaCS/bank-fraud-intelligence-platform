"""Simulate production drift monitoring.

Compares feature distributions, score distributions, and alert volumes
between training and holdout test data to detect data or concept drift.
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
import numpy as np
import pandas as pd

from src.config import load_config
from src.feature_store import build_features, MODEL_FEATURES
from src.models.graph_features import build_graph_features
from src.monitoring.drift import population_stability_index
from src.monitoring.performance_tracking import classification_metrics
from src.scoring.realtime_scoring import load_artifacts, score_batch
from src.models.ranker import combine_scores, assign_actions
from src.rules_engine import apply_rules
from src.utils.io import dump_json


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
    mon = cfg["monitoring"]

    train = pd.read_csv(paths["train_dataset"])
    test = pd.read_csv(paths["test_dataset"])
    accounts = pd.read_csv(paths["accounts_dataset"])
    customers = pd.read_csv(paths["customers_dataset"])

    train_f = build_graph_features(build_features(train, accounts, customers))
    test_f = build_graph_features(build_features(test, accounts, customers))

    # --- Feature drift ---
    feature_drift = {}
    for col in MODEL_FEATURES:
        psi = population_stability_index(train_f[col], test_f[col], bins=mon["psi_bins"])
        feature_drift[col] = {
            "psi": round(psi, 6),
            "drift_flag": psi >= mon["drift_alert_threshold"],
        }

    # --- Score drift ---
    models = load_artifacts(paths["model_dir"])
    train_r = apply_rules(train_f, cfg["rules"])
    test_r = apply_rules(test_f, cfg["rules"])

    train_scored = score_batch(train_r, models, MODEL_FEATURES)
    test_scored = score_batch(test_r, models, MODEL_FEATURES)

    for df_ in [train_scored, test_scored]:
        df_["anomaly_score_norm"] = minmax_norm(df_["anomaly_score"])
        df_["rule_score_norm"] = (df_["rule_score"] / 4.0).clip(0, 1)

    train_scored = combine_scores(train_scored)
    test_scored = combine_scores(test_scored)
    train_scored = assign_actions(train_scored, cfg["thresholds"])
    test_scored = assign_actions(test_scored, cfg["thresholds"])

    score_drift = {}
    for score_col in ["logit_score", "hgb_score", "anomaly_score", "fraud_score", "investigation_priority_score"]:
        psi = population_stability_index(train_scored[score_col], test_scored[score_col], bins=mon["psi_bins"])
        score_drift[score_col] = {
            "psi": round(psi, 6),
            "drift_flag": psi >= mon["drift_alert_threshold"],
        }

    # --- Alert volume drift ---
    train_actions = train_scored["recommended_action"].value_counts(normalize=True).to_dict()
    test_actions = test_scored["recommended_action"].value_counts(normalize=True).to_dict()
    alert_volume_drift = {
        "train_action_distribution": {k: round(v, 4) for k, v in train_actions.items()},
        "test_action_distribution": {k: round(v, 4) for k, v in test_actions.items()},
    }

    # --- Test performance ---
    y_test = test_scored["label_fraud"].astype(int)
    test_metrics = classification_metrics(y_test, test_scored["fraud_score"].values, cfg["thresholds"]["review_score"])

    # --- Compile report ---
    drift_report = {
        "feature_drift": feature_drift,
        "score_drift": score_drift,
        "alert_volume_drift": alert_volume_drift,
        "test_set_performance": test_metrics,
    }

    dump_json(drift_report, Path(paths["reports_dir"]) / "drift_report.json")
    print("Saved drift report.")
    print(f"  Test performance: {test_metrics}")
    flagged = [k for k, v in feature_drift.items() if v["drift_flag"]]
    if flagged:
        print(f"  Features with drift: {flagged}")
    else:
        print("  No feature drift detected.")


if __name__ == "__main__":
    main()
