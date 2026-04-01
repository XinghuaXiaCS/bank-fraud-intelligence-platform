"""Tests for graph feature engineering."""
import pandas as pd
import numpy as np
from src.models.graph_features import build_graph_features


def _make_graph_test_data():
    """Small dataset with known graph structure."""
    return pd.DataFrame({
        "account_id": [1, 1, 2, 2, 3],
        "device_id": [100, 100, 100, 101, 102],  # accounts 1 and 2 share device 100
        "counterparty_account_id": [2, 3, 1, 3, 1],
        "label_fraud": [1, 1, 0, 0, 0],  # account 1 is fraudulent
    })


def test_graph_features_added():
    """build_graph_features should add all four graph feature columns."""
    df = _make_graph_test_data()
    out = build_graph_features(df)
    for col in ["graph_degree", "graph_neighbor_fraud_ratio", "graph_pagerank", "graph_community_fraud_ratio"]:
        assert col in out.columns, f"Missing: {col}"


def test_graph_degree_positive():
    """Accounts connected via edges should have positive degree."""
    df = _make_graph_test_data()
    out = build_graph_features(df)
    assert (out["graph_degree"] > 0).all()


def test_neighbor_fraud_ratio_range():
    """Neighbour fraud ratio should be between 0 and 1."""
    df = _make_graph_test_data()
    out = build_graph_features(df)
    assert (out["graph_neighbor_fraud_ratio"] >= 0).all()
    assert (out["graph_neighbor_fraud_ratio"] <= 1).all()


def test_pagerank_non_negative():
    """PageRank scores should be non-negative."""
    df = _make_graph_test_data()
    out = build_graph_features(df)
    assert (out["graph_pagerank"] >= 0).all()
