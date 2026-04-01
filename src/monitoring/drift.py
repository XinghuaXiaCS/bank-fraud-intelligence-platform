from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    eps = 1e-6
    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(expected.quantile(quantiles).values)
    if len(cut_points) < 3:
        return 0.0
    e_bins = pd.cut(expected, bins=cut_points, include_lowest=True)
    a_bins = pd.cut(actual, bins=cut_points, include_lowest=True)
    e_dist = e_bins.value_counts(normalize=True).sort_index() + eps
    a_dist = a_bins.value_counts(normalize=True).sort_index() + eps
    common = e_dist.index.intersection(a_dist.index)
    psi = ((a_dist.loc[common] - e_dist.loc[common]) * np.log(a_dist.loc[common] / e_dist.loc[common])).sum()
    return float(psi)
