"""Train the full fraud detection model stack.

Produces:
  - Logistic Regression baseline
  - HistGradientBoosting main model
  - Isolation Forest anomaly layer
  - ATO detection model (logistic)
  - Ensemble scoring and investigation prioritisation
  - Feature importance report
  - Model card and governance report
  - Fairness report by protected segments
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
from src.rules_engine import apply_rules
from src.models.logistic import build_logistic
from src.models.xgb_model import build_hgb
from src.models.anomaly import build_isolation_forest
from src.models.ranker import combine_scores, assign_actions
from src.monitoring.performance_tracking import classification_metrics
from src.explainability.shap_reports import export_feature_importance
from src.utils.io import dump_json


def minmax_norm(series: pd.Series) -> pd.Series:
    lo, hi = float(series.min()), float(series.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def run_fairness_check(
    df: pd.DataFrame, score_col: str, label_col: str,
    segments: list[str], threshold: float,
) -> dict:
    """Compute false-positive rate per segment for fairness analysis."""
    report: dict = {}
    preds = (df[score_col] >= threshold).astype(int)
    negatives = df[label_col] == 0

    for seg in segments:
        if seg not in df.columns:
            continue
        seg_report = {}
        for val, grp in df[negatives].groupby(seg):
            grp_preds = preds.loc[grp.index]
            fp_count = int(grp_preds.sum())
            total = len(grp)
            seg_report[str(val)] = {
                "false_positive_count": fp_count,
                "total_negatives": total,
                "false_positive_rate": round(fp_count / max(total, 1), 4),
            }
        report[seg] = seg_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    reports_dir = Path(paths["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(paths["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    train = pd.read_csv(paths["train_dataset"])
    accounts = pd.read_csv(paths["accounts_dataset"])
    customers = pd.read_csv(paths["customers_dataset"])

    # ---- Feature engineering ----
    df = build_features(train, accounts, customers)
    df = build_graph_features(df)
    df = apply_rules(df, cfg["rules"])

    X = df[MODEL_FEATURES].copy()
    y_fraud = df["label_fraud"].astype(int)
    y_ato = df["label_ato"].astype(int)

    # ---- Train fraud models ----
    logit = build_logistic(**cfg["models"]["logistic"])
    hgb = build_hgb(**cfg["models"]["hgb"])
    iforest = build_isolation_forest(**cfg["models"]["isolation_forest"])

    logit.fit(X, y_fraud)
    hgb.fit(X, y_fraud)
    iforest.fit(X)

    # ---- Train ATO model ----
    ato_logit = build_logistic(**cfg["models"]["logistic"])
    ato_logit.fit(X, y_ato)

    # ---- Score training set ----
    df["logit_score"] = logit.predict_proba(X)[:, 1]
    df["hgb_score"] = hgb.predict_proba(X)[:, 1]
    df["anomaly_score"] = -iforest.score_samples(X)
    df["anomaly_score_norm"] = minmax_norm(df["anomaly_score"])
    df["rule_score_norm"] = (df["rule_score"] / 4.0).clip(0, 1)
    df["ato_score"] = ato_logit.predict_proba(X)[:, 1]
    df = combine_scores(df)
    df = assign_actions(df, cfg["thresholds"])

    # ---- Evaluation ----
    baseline_metrics = classification_metrics(y_fraud, df["logit_score"].values, cfg["thresholds"]["review_score"])
    ensemble_metrics = classification_metrics(y_fraud, df["fraud_score"].values, cfg["thresholds"]["review_score"])
    ato_metrics = classification_metrics(y_ato, df["ato_score"].values, cfg["thresholds"]["review_score"])

    # ---- Save artifacts ----
    export_feature_importance(MODEL_FEATURES, hgb, reports_dir / "feature_importance.csv")
    dump_json(baseline_metrics, reports_dir / "baseline_metrics.json")
    dump_json(ensemble_metrics, reports_dir / "ensemble_metrics.json")
    dump_json(ato_metrics, reports_dir / "ato_metrics.json")

    joblib.dump(logit, model_dir / "logit.joblib")
    joblib.dump(hgb, model_dir / "hgb.joblib")
    joblib.dump(iforest, model_dir / "iforest.joblib")
    joblib.dump(ato_logit, model_dir / "ato_logit.joblib")

    # ---- Fairness report ----
    fairness_segments = cfg.get("fairness", {}).get("protected_segments", [])
    fairness_report = run_fairness_check(
        df, "fraud_score", "label_fraud", fairness_segments, cfg["thresholds"]["review_score"]
    )
    dump_json(fairness_report, reports_dir / "fairness_report.json")

    # ---- Model card ----
    model_card = f"""# Model Card

## System
Integrated Fraud Intelligence Platform

## Primary use
Transaction fraud detection, account takeover risk scoring, and investigation prioritisation.

## Models
- Logistic Regression baseline (fraud)
- HistGradientBoostingClassifier main model (fraud)
- IsolationForest anomaly layer
- Logistic Regression ATO model
- Rule engine with reason codes
- Blended risk score with investigation priority ranking

## Training data
Synthetic reproduction of production banking fraud data with:
- {len(train)} transactions
- Fraud rate: {y_fraud.mean():.4f}
- ATO rate: {y_ato.mean():.4f}
- Multi-table structure: customers, accounts, devices, transactions, login events, investigations

## Key metrics (training set)

### Fraud detection
- Baseline (Logistic) ROC-AUC: {baseline_metrics['roc_auc']:.4f}
- Ensemble ROC-AUC: {ensemble_metrics['roc_auc']:.4f}
- Ensemble PR-AUC: {ensemble_metrics['pr_auc']:.4f}
- Ensemble Precision @ threshold: {ensemble_metrics['precision']:.4f}
- Ensemble Recall @ threshold: {ensemble_metrics['recall']:.4f}
- Ensemble F1: {ensemble_metrics['f1']:.4f}

### Account takeover detection
- ATO ROC-AUC: {ato_metrics['roc_auc']:.4f}
- ATO PR-AUC: {ato_metrics['pr_auc']:.4f}

## Alert distribution (training set)
- Approved: {int((df['recommended_action'] == 'approve').sum())}
- Review: {int((df['recommended_action'] == 'review').sum())}
- Challenge: {int((df['recommended_action'] == 'challenge').sum())}
- High priority: {int((df['priority_band'] == 'high').sum())}
- Medium priority: {int((df['priority_band'] == 'medium').sum())}
- Low priority: {int((df['priority_band'] == 'low').sum())}

## Fairness
False-positive rate parity checked across: {', '.join(fairness_segments) if fairness_segments else 'none configured'}.
See fairness_report.json for segment-level breakdown.

## Limitations
- This open-source version uses synthetic data; distributions are modelled on but may not exactly match production fraud patterns.
- Graph layer uses engineered features rather than a full GNN.
- Decision thresholds are illustrative and should be calibrated to investigator capacity.
- ATO model trained on transaction-level labels; a login-level model would be more precise.

## Retraining
- Recommended cadence: monthly or after significant drift detection.
- Fallback: revert to logistic baseline if main model degrades.
"""
    (reports_dir / "model_card.md").write_text(model_card, encoding="utf-8")

    # ---- Governance report ----
    governance = """# Governance Report

## Data governance
- Original production data has been replaced with synthetic data for public publication.
- No personal identifiable information is used.

## Model governance
- Rules and model scores are preserved separately for auditability.
- Alert actions are threshold-based and fully reproducible.
- Feature importance is exported for model review.
- Reason codes are generated for every alert.

## Fairness and bias
- False-positive rate parity is monitored across age bands and customer segments.
- Detailed segment-level fairness metrics are in fairness_report.json.
- Before production use, disparate impact testing should be extended.

## Monitoring
- Drift monitoring uses Population Stability Index (PSI) per feature.
- Score and alert volume drift are tracked in drift_report.json.
- Recommended: alert-to-case conversion rate and investigator feedback loop.

## Explainability
- Feature importance from gradient boosting is exported.
- Rule engine produces human-readable reason codes for every alert.
- SHAP values can be computed on demand via src/explainability/shap_reports.py.

## Production readiness
- Before deployment: add challenger model testing, A/B evaluation, and retraining governance.
- Threshold calibration should be linked to investigator capacity constraints.
- Model card should be reviewed and signed off by model risk owner.
"""
    (reports_dir / "governance_report.md").write_text(governance, encoding="utf-8")

    # ---- Summary ----
    print("Training complete.")
    print(f"  Fraud rate: {y_fraud.mean():.4f}  |  ATO rate: {y_ato.mean():.4f}")
    print(f"  Baseline: {baseline_metrics}")
    print(f"  Ensemble: {ensemble_metrics}")
    print(f"  ATO:      {ato_metrics}")
    print(f"  Actions:  approve={int((df['recommended_action']=='approve').sum())} "
          f"review={int((df['recommended_action']=='review').sum())} "
          f"challenge={int((df['recommended_action']=='challenge').sum())}")


if __name__ == "__main__":
    main()
