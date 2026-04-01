# Model Card

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
- 25000 transactions
- Fraud rate: 0.0362
- ATO rate: 0.0148
- Multi-table structure: customers, accounts, devices, transactions, login events, investigations

## Key metrics (training set)

### Fraud detection
- Baseline (Logistic) ROC-AUC: 0.7142
- Ensemble ROC-AUC: 0.7362
- Ensemble PR-AUC: 0.1975
- Ensemble Precision @ threshold: 0.1061
- Ensemble Recall @ threshold: 0.4956
- Ensemble F1: 0.1747

### Account takeover detection
- ATO ROC-AUC: 0.6632
- ATO PR-AUC: 0.0356

## Alert distribution (training set)
- Approved: 20776
- Review: 4128
- Challenge: 96
- High priority: 336
- Medium priority: 2232
- Low priority: 22432

## Fairness
False-positive rate parity checked across: age_band, customer_segment.
See fairness_report.json for segment-level breakdown.

## Limitations
- This open-source version uses synthetic data; distributions are modelled on but may not exactly match production fraud patterns.
- Graph layer uses engineered features rather than a full GNN.
- Decision thresholds are illustrative and should be calibrated to investigator capacity.
- ATO model trained on transaction-level labels; a login-level model would be more precise.

## Retraining
- Recommended cadence: monthly or after significant drift detection.
- Fallback: revert to logistic baseline if main model degrades.
