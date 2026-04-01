# Governance Report

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
