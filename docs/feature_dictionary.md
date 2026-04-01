# Feature Dictionary

This document describes all features used by the supervised models in the fraud intelligence platform.

---

## Basic transaction features

| Feature | Type | Description |
|---|---|---|
| `amount` | float | Transaction amount in dollars |
| `log_amount` | float | log(1 + amount); reduces skewness |
| `channel_risk` | float | Risk score associated with the payment channel (0–1) |
| `device_risk` | float | External risk score for the device used (0–1) |
| `merchant_risk` | float | Risk level of the merchant category (0–1) |

## Novelty features

| Feature | Type | Description |
|---|---|---|
| `is_new_device` | int | 1 if the device has not been seen on this account before |
| `is_new_geo` | int | 1 if the geographic location is new for this account |
| `geo_distance_score` | float | Distance-based risk between current and historical locations (0–1) |

## Velocity features (time-window based)

All velocity features are computed using real trailing time windows per account.

| Feature | Type | Description |
|---|---|---|
| `txn_count_1h` | float | Number of transactions by this account in the trailing 1 hour |
| `txn_count_24h` | float | Number of transactions by this account in the trailing 24 hours |
| `txn_amount_sum_1h` | float | Total transaction amount by this account in the trailing 1 hour |
| `unique_merchants_1h` | float | Distinct merchant categories in the trailing 1 hour |
| `unique_devices_24h` | float | Distinct devices used by this account in the trailing 24 hours |
| `unique_geos_24h` | float | Distinct geographic locations in the trailing 24 hours |

## Behavioural deviation features

| Feature | Type | Description |
|---|---|---|
| `high_amount_vs_turnover` | float | Current amount relative to the account's average monthly turnover |
| `amount_z` | float | Z-score of current amount vs the account's trailing 10-transaction rolling average |
| `peer_amount_ratio` | float | Current amount relative to the median for the same merchant category |

## Account/customer profile features

| Feature | Type | Description |
|---|---|---|
| `num_linked_accounts` | int | Number of accounts linked to the same device |
| `failed_logins_24h` | int | Count of failed login attempts in the past 24 hours |
| `new_payee_recent` | int | 1 if a new payee was added recently |
| `shared_counterparty_risk` | float | Risk score derived from counterparty relationships (0–1) |

## Time features

| Feature | Type | Description |
|---|---|---|
| `hour` | int | Hour of the day (0–23) |
| `dayofweek` | int | Day of the week (0=Monday, 6=Sunday) |
| `is_weekend` | int | 1 if Saturday or Sunday |

## Graph/network features

All graph features are derived from an entity graph connecting accounts, devices, and counterparties.

| Feature | Type | Description |
|---|---|---|
| `graph_degree` | float | Number of edges connected to the account node in the entity graph |
| `graph_neighbor_fraud_ratio` | float | Proportion of neighbouring account nodes that are known fraud |
| `graph_pagerank` | float | PageRank centrality score of the account node |
| `graph_community_fraud_ratio` | float | Fraud rate within the account's connected component |

---

## Feature computation notes

- Velocity features use pandas DatetimeIndex rolling with `closed="left"` to prevent look-ahead leakage.
- Graph features are computed from training data only; test-time graph features use the training graph structure.
- All features are computed in `src/feature_store.py` and `src/models/graph_features.py`.
- The feature list used by models is defined in `src.feature_store.MODEL_FEATURES`.
