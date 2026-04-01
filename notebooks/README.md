# Notebooks

Analysis notebooks that walk through each stage of the fraud intelligence platform.

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Exploratory data analysis across all data tables |
| `02_feature_engineering.ipynb` | Velocity features, behavioural deviation, and graph features |
| `03_baseline_models.ipynb` | Logistic Regression vs HistGradientBoosting vs Isolation Forest |
| `04_ensemble_and_ranking.ipynb` | Score fusion, investigation prioritisation, and alert actions |
| `05_graph_features.ipynb` | Entity graph analysis and linked-risk feature exploration |
| `06_monitoring_and_drift.ipynb` | Drift monitoring, fairness analysis, and governance review |

## Prerequisites

Run the full pipeline before opening the notebooks:

```bash
python scripts/generate_synthetic_data.py --config configs/default.yaml
python scripts/train_pipeline.py --config configs/default.yaml
python scripts/score_alerts.py --config configs/default.yaml
python scripts/simulate_retraining_monitoring.py --config configs/default.yaml
```
