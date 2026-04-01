import pandas as pd
from src.rules_engine import apply_rules


def test_apply_rules_adds_rule_score():
    df = pd.DataFrame({
        "amount": [3000, 50],
        "is_new_device": [1, 0],
        "txn_count_1h": [8, 1],
        "num_linked_accounts": [5, 1],
        "is_new_geo": [1, 0],
    })
    cfg = {
        "high_amount_threshold": 2200,
        "rapid_txn_threshold_1h": 6,
        "linked_accounts_threshold": 3,
        "geo_novelty_penalty": 1,
    }
    out = apply_rules(df, cfg)
    assert "rule_score" in out.columns
    assert out.loc[0, "rule_score"] >= 3
