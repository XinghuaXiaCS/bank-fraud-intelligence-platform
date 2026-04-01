from __future__ import annotations

import pandas as pd


def apply_rules(df: pd.DataFrame, rules_cfg: dict) -> pd.DataFrame:
    out = df.copy()
    out["rule_high_amount_new_device"] = (
        (out["amount"] >= rules_cfg["high_amount_threshold"]) & (out["is_new_device"] == 1)
    ).astype(int)
    out["rule_rapid_activity"] = (out["txn_count_1h"] >= rules_cfg["rapid_txn_threshold_1h"]).astype(int)
    out["rule_device_fanout"] = (out["num_linked_accounts"] >= rules_cfg["linked_accounts_threshold"]).astype(int)
    out["rule_geo_novelty"] = (out["is_new_geo"] >= rules_cfg["geo_novelty_penalty"]).astype(int)
    out["rule_score"] = out[
        [
            "rule_high_amount_new_device",
            "rule_rapid_activity",
            "rule_device_fanout",
            "rule_geo_novelty",
        ]
    ].sum(axis=1)
    return out


def build_reason_codes(row: pd.Series) -> str:
    reasons = []
    if row.get("rule_high_amount_new_device", 0):
        reasons.append("HIGH_AMOUNT_NEW_DEVICE")
    if row.get("rule_rapid_activity", 0):
        reasons.append("RAPID_ACTIVITY")
    if row.get("rule_device_fanout", 0):
        reasons.append("DEVICE_FANOUT")
    if row.get("rule_geo_novelty", 0):
        reasons.append("GEO_NOVELTY")
    return "|".join(reasons) if reasons else "MODEL_ONLY"
