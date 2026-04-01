# Integrated Fraud Intelligence Platform for Banking and Payment Abuse Detection

A production-grade fraud analytics system combining **rules, supervised machine learning, anomaly detection, graph/network analytics, investigation prioritisation, and governance-ready outputs** — reproduced from real-world banking fraud work with synthetic data for public publication.

This repository is a restructured and open-sourced reproduction of a fraud detection system I built during my time at Yusys Technologies, adapted with synthetic data for public publication. It is intentionally structured so the same detection framework can be transferred from **banking and payments** to **public-sector payment integrity and provider misuse** problems such as **fraud, waste, and abuse detection**.

---

## Executive summary

This platform implements an end-to-end fraud intelligence stack that can:

- detect suspicious events and entities across multiple fraud types
- combine deterministic rules with statistical and machine learning models
- surface emerging fraud patterns through anomaly detection
- incorporate linked-entity risk using graph-based features (degree, PageRank, community fraud ratio)
- score account takeover risk alongside transaction fraud
- rank alerts for investigation teams based on likelihood, severity, and actionability
- produce explainable, governance-ready outputs including fairness analysis
- transfer naturally from banking fraud to payment integrity, claim integrity, and provider abuse monitoring

In a real organisation, fraud detection does not stop at model training. It requires triage, monitoring, reason codes, governance, and business-ready outputs. This repository is built around that operating model.

---

## Business problem

The platform is designed around five practical use cases:

1. **Transaction fraud detection** — Detect suspicious payment events using transactional, behavioural, velocity, and contextual features.
2. **Account takeover risk detection** — Flag risky logins and account access events based on device novelty, geo change, velocity, and account behaviour.
3. **Account/customer anomaly detection** — Identify unusual behaviour that may indicate new fraud patterns not yet well represented in labelled data.
4. **Linked-entity and network risk detection** — Capture hidden risk across shared devices, counterparties, and connected entities using graph-derived features.
5. **Investigation prioritisation** — Rank alerts by fraud likelihood, severity, and actionability so scarce investigation capacity is allocated to the most valuable cases.

---

## Solution architecture

```text
Raw events / entities
    |
    v
Feature engineering (velocity, deviation, profile, graph)
    |
    +--> Rules engine ---------------------------------+
    |                                                  |
    +--> Logistic Regression baseline (fraud + ATO)    |
    |                                                  |
    +--> HistGradientBoosting main fraud model         |--> Score fusion --> Alerts --> Priority ranking --> Investigation queue
    |                                                  |
    +--> Isolation Forest anomaly layer                |
    |                                                  |
    +--> Graph features (degree, PageRank, community) -+
    |
    v
Explainability + reason codes + drift monitoring + fairness + governance
```

---

## Data architecture

The project uses a realistic multi-table structure:

| Table | Rows | Description |
|---|---|---|
| `customers` | 2,500 | Customer demographics, segments, KYC level, risk band |
| `accounts` | 3,200 | Account type, balance, turnover, dormancy flag |
| `devices` | 2,500 | Device history, linked accounts, external risk score |
| `transactions` (train) | 25,000 | Payment events with fraud and ATO labels |
| `transactions` (test) | 10,000 | Out-of-time holdout for evaluation |
| `login_events` | 8,000 | Login attempts with ATO labels |
| `investigations` | ~5,000 | Case outcomes, confirmed loss, recovery |
| `alerts` | ~2,700 | Alert triggers, priority, review outcomes |
| `edges` | ~38,000 | Entity graph: account↔device, account↔counterparty |

---

## Feature engineering

Features are organised into seven groups:

1. **Basic transaction features** — amount, log amount, merchant risk, channel risk
2. **Novelty features** — new device flag, new geo flag, geo distance score
3. **Velocity features** — real-time-window counts and sums (1h, 24h) including transaction count, amount sum, unique merchants, unique devices, unique geos
4. **Behavioural deviation** — amount z-score vs rolling history, peer amount ratio, amount vs turnover
5. **Account/customer profile** — linked account count, failed logins, new payee, shared counterparty risk
6. **Time features** — hour, day of week, weekend flag
7. **Graph features** — degree centrality, neighbour fraud ratio, PageRank, community fraud ratio

See [`docs/feature_dictionary.md`](docs/feature_dictionary.md) for the full feature dictionary.

---

## Methods used

### Production methods
- Deterministic fraud rules with reason codes
- Logistic Regression baseline (fraud and ATO)
- HistGradientBoostingClassifier main model
- Isolation Forest anomaly layer
- Time-window velocity feature engineering
- Graph/network feature engineering (degree, PageRank, community fraud ratio)
- Blended score fusion with investigation priority ranking
- PSI-based drift monitoring
- Fairness analysis by protected segments

### Advanced methods included or scaffolded
- Linked-entity risk features for fraud rings and synthetic identities
- SHAP-based explainability (available when shap package is installed)
- Optional GNN extension scaffold (`src/models/gnn.py`)
- Prioritisation logic that separates fraud probability from investigation priority

---

## Repository structure

```text
bank-fraud-intelligence-platform/
├── configs/
│   └── default.yaml
├── dashboards/
│   └── README.md
├── data/
│   └── synthetic/             # 9 CSV tables
├── docs/
│   ├── ACC_MIGRATION_GUIDE.md
│   ├── feature_dictionary.md
│   └── fraud_strategy.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_ensemble_and_ranking.ipynb
│   ├── 05_graph_features.ipynb
│   └── 06_monitoring_and_drift.ipynb
├── reports/
│   ├── alerts_scored.csv
│   ├── ato_metrics.json
│   ├── baseline_metrics.json
│   ├── drift_report.json
│   ├── ensemble_metrics.json
│   ├── fairness_report.json
│   ├── feature_importance.csv
│   ├── governance_report.md
│   └── model_card.md
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── score_alerts.py
│   ├── simulate_retraining_monitoring.py
│   └── train_pipeline.py
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── rules_engine.py
│   ├── feature_store.py
│   ├── explainability/
│   │   └── shap_reports.py
│   ├── models/
│   │   ├── logistic.py
│   │   ├── xgb_model.py
│   │   ├── anomaly.py
│   │   ├── graph_features.py
│   │   ├── gnn.py
│   │   └── ranker.py
│   ├── monitoring/
│   │   ├── drift.py
│   │   └── performance_tracking.py
│   ├── scoring/
│   │   ├── realtime_scoring.py
│   │   └── batch_scoring.py
│   └── utils/
│       └── io.py
└── tests/
    ├── test_drift.py
    ├── test_feature_store.py
    ├── test_graph_features.py
    ├── test_ranker.py
    └── test_rules_engine.py
```

---

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate synthetic fraud data

```bash
python scripts/generate_synthetic_data.py --config configs/default.yaml
```

### 3. Train the fraud stack

```bash
python scripts/train_pipeline.py --config configs/default.yaml
```

### 4. Score alerts on the holdout set

```bash
python scripts/score_alerts.py --config configs/default.yaml
```

### 5. Produce the drift report

```bash
python scripts/simulate_retraining_monitoring.py --config configs/default.yaml
```

### 6. Run tests

```bash
python -m pytest tests/ -v
```

---

## Outputs

After running the pipeline, the repository produces the following artifacts in `reports/`:

| Artifact | Description |
|---|---|
| `baseline_metrics.json` | Logistic Regression performance |
| `ensemble_metrics.json` | Blended fraud score performance |
| `ato_metrics.json` | Account takeover model performance |
| `drift_report.json` | Feature, score, and alert volume drift |
| `fairness_report.json` | False-positive rate by protected segments |
| `feature_importance.csv` | Feature importance from gradient boosting |
| `alerts_scored.csv` | Full scored and prioritised alert queue |
| `model_card.md` | Model documentation for governance |
| `governance_report.md` | Governance, fairness, and monitoring summary |

---

## Portability to ACC-style fraud, waste, and abuse analytics

This project is intentionally designed to be portable.

| Banking / payments | ACC-style analogue |
|---|---|
| customer / account | client / claim |
| transaction | payment / service event |
| merchant / counterparty | provider |
| payment abuse | compensation irregularity |
| merchant billing anomaly | provider billing anomaly |
| linked accounts / devices | linked clients / providers / services |
| investigation queue | fraud / integrity review queue |

See [`docs/ACC_MIGRATION_GUIDE.md`](docs/ACC_MIGRATION_GUIDE.md) for the full translation layer.

---


