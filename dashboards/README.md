# Dashboard Design

This directory is a placeholder for dashboard implementation. In a production deployment, the following views would be built using Streamlit, Plotly Dash, or an enterprise BI tool.

## Recommended views

### 1. Alert queue
- Sortable table of scored alerts with fraud score, priority band, and recommended action.
- Colour-coded rows: green (approve), amber (review), red (challenge).
- Drill-down to individual alert detail.

### 2. Alert detail
- Transaction metadata and context.
- Model score breakdown: rule score, logistic score, HGB score, anomaly score.
- Reason codes in plain English.
- Feature contribution chart (SHAP waterfall or feature importance bar).
- Entity graph neighbourhood view showing linked accounts and devices.

### 3. Investigation queue
- Cases sorted by investigation priority score.
- Queue assignment: L1 auto / L2 analyst / L3 specialist.
- Expected loss and recovery estimates.
- Case status tracking: open / under review / closed.

### 4. Performance monitoring
- Model metric trends over time: ROC-AUC, PR-AUC, precision, recall.
- Alert volume by action type and priority band.
- Feature drift heatmap (PSI by feature and time period).
- Score distribution shift comparison (train vs current).

### 5. Fairness dashboard
- False-positive rate comparison across age bands and customer segments.
- Alert rate parity by protected attributes.
- Trend monitoring for fairness metrics.

### 6. Network risk view
- Interactive entity graph showing accounts, devices, and counterparties.
- Nodes coloured by risk score; edges weighted by interaction frequency.
- Community detection highlighting suspicious clusters.
- Drill-down from suspicious network to individual cases.
