"""Graph-based linked-entity risk features.

Constructs a bipartite graph of accounts, devices, and counterparties,
then computes:
  - degree centrality
  - neighbour fraud ratio
  - PageRank score
  - community-level fraud ratio (connected components)
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add graph-derived features to the transaction dataframe."""
    required = ["account_id", "device_id", "counterparty_account_id", "label_fraud"]
    if not all(c in df.columns for c in required):
        for col in ["graph_degree", "graph_neighbor_fraud_ratio",
                     "graph_pagerank", "graph_community_fraud_ratio"]:
            if col not in df.columns:
                df[col] = 0.0
        return df

    # Build edges
    edges = []
    subset = df[required].dropna()
    for _, row in subset.iterrows():
        edges.append((f"acct:{int(row['account_id'])}", f"dev:{int(row['device_id'])}"))
        edges.append((f"acct:{int(row['account_id'])}", f"cp:{int(row['counterparty_account_id'])}"))

    graph = nx.Graph()
    graph.add_edges_from(edges)

    # --- Degree ---
    degree = dict(graph.degree())

    # --- Fraud set ---
    fraud_accounts = set(
        f"acct:{int(x)}" for x in df.loc[df["label_fraud"] == 1, "account_id"].unique()
    )

    # --- Neighbour fraud ratio ---
    neighbor_fraud_ratio = {}
    for node in graph.nodes:
        neighbours = list(graph.neighbors(node))
        if not neighbours:
            neighbor_fraud_ratio[node] = 0.0
            continue
        risky = sum(1 for n in neighbours if n in fraud_accounts)
        neighbor_fraud_ratio[node] = risky / len(neighbours)

    # --- PageRank ---
    try:
        pagerank = nx.pagerank(graph, max_iter=50, tol=1e-4)
    except Exception:
        pagerank = {n: 0.0 for n in graph.nodes}

    # --- Community fraud ratio (connected components) ---
    community_fraud = {}
    for comp in nx.connected_components(graph):
        accts_in_comp = [n for n in comp if n.startswith("acct:")]
        fraud_in_comp = sum(1 for a in accts_in_comp if a in fraud_accounts)
        ratio = fraud_in_comp / max(len(accts_in_comp), 1)
        for node in comp:
            community_fraud[node] = ratio

    # --- Map back to dataframe ---
    out = df.copy()
    acct_nodes = "acct:" + out["account_id"].astype(int).astype(str)
    out["graph_degree"] = acct_nodes.map(degree).fillna(0).astype(float)
    out["graph_neighbor_fraud_ratio"] = acct_nodes.map(neighbor_fraud_ratio).fillna(0).astype(float)
    out["graph_pagerank"] = acct_nodes.map(pagerank).fillna(0).astype(float)
    out["graph_community_fraud_ratio"] = acct_nodes.map(community_fraud).fillna(0).astype(float)

    return out
