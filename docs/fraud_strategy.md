# Fraud Strategy Document

## Purpose

This document describes the fraud detection strategy implemented in the Integrated Fraud Intelligence Platform. It covers the fraud taxonomy, detection use cases, label design, business rules, and operational workflow.

---

## Fraud taxonomy

The platform addresses four categories of financial abuse:

1. **Transaction fraud**: Unauthorised or suspicious payments, including card-not-present fraud, velocity attacks, card testing, and mule account transfers.
2. **Account takeover (ATO)**: Unauthorised access to legitimate accounts through credential compromise, device hijacking, or social engineering.
3. **Behavioural anomaly**: Changes in account or customer behaviour that may indicate new or emerging fraud patterns not yet well-represented in labelled data.
4. **Network/ring fraud**: Coordinated abuse involving multiple linked accounts, devices, or counterparties — including synthetic identity fraud, mule networks, and collusive schemes.

---

## Detection use cases

### Use case 1: Transaction fraud detection

Detect suspicious payment events using transactional, behavioural, and contextual features. Key signals include high-risk merchant categories, new device/geo combinations, velocity spikes, and amount anomalies.

**Output**: Transaction fraud score, recommended action (approve / review / challenge).

### Use case 2: Account takeover detection

Flag risky logins and account access events. Key signals include new device, geo change, failed login spikes, recent password resets, and unusual channel usage.

**Output**: ATO risk score, step-up authentication recommendation.

### Use case 3: Account/customer anomaly detection

Identify behaviour that deviates from established baselines. Key signals include dormant account activation, transaction pattern shifts, and peer group deviation.

**Output**: Anomaly score (via Isolation Forest), integrated into ensemble scoring.

### Use case 4: Linked-entity network risk

Capture hidden risk from shared devices, counterparties, and connected entities. Key signals include graph degree, neighbour fraud ratio, PageRank centrality, and community fraud ratio.

**Output**: Graph risk features integrated into supervised models.

### Use case 5: Investigation prioritisation

Rank alerts for investigation teams based on fraud likelihood, transaction severity, and actionability.

**Output**: Investigation priority score, priority band (low / medium / high), queue routing.

---

## Label design

### Primary labels
- `label_fraud`: Confirmed transaction fraud (binary).
- `label_ato`: Confirmed account takeover event (binary).

### Operational labels (investigations table)
- `review_outcome`: substantiated / not_substantiated / monitor.
- `confirmed_loss`: Dollar value of confirmed fraudulent loss.
- `recovery_amount`: Dollar value recovered through intervention.
- `investigator_team`: L1_auto / L2_analyst / L3_specialist.

### Alert labels (alerts table)
- `alert_priority`: low / medium / high.
- `trigger_source`: rule_engine / ml_model / anomaly_detector.
- `review_outcome`: investigated / auto_closed / pending.

---

## Business rules

The rule engine applies deterministic thresholds before model scoring:

| Rule | Condition | Rationale |
|---|---|---|
| HIGH_AMOUNT_NEW_DEVICE | Amount ≥ $2,200 AND device not seen before | Large payment from unknown device |
| RAPID_ACTIVITY | ≥ 6 transactions in trailing 1 hour | Velocity attack pattern |
| DEVICE_FANOUT | ≥ 3 accounts linked to device | Shared device / mule indicator |
| GEO_NOVELTY | Transaction from new geographic location | Unusual location for account |

Rules contribute to the blended fraud score and generate human-readable reason codes.

---

## Model stack

| Layer | Method | Purpose |
|---|---|---|
| Layer 0 | Rule engine | Deterministic high-confidence rules |
| Layer 1a | Logistic Regression | Calibrated baseline probability |
| Layer 1b | HistGradientBoosting | Primary discriminative model |
| Layer 2 | Isolation Forest | Unsupervised anomaly detection |
| Layer 3 | Graph feature engineering | Linked-entity risk signals |
| Layer 4 | Score fusion + ranking | Blended fraud score and investigation priority |

---

## Operational workflow

```
Event stream → Feature engineering → Rule engine + Model scoring → Score fusion
    → Alert generation → Priority ranking → Investigation queue
    → Case outcome → Feedback into monitoring and retraining
```

---

## Thresholds and actions

| Score range | Action | Queue |
|---|---|---|
| fraud_score < 0.25 | Approve | No alert |
| 0.25 ≤ fraud_score < 0.45 | Review | L1 / L2 analyst queue |
| fraud_score ≥ 0.45 | Challenge / hold | L2 / L3 specialist queue |

Thresholds should be calibrated to investigator capacity and business risk appetite.
