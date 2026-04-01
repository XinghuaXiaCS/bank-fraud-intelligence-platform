import pandas as pd
from src.monitoring.drift import population_stability_index


def test_population_stability_index_non_negative():
    a = pd.Series([1, 2, 3, 4, 5, 6])
    b = pd.Series([1, 2, 3, 10, 12, 15])
    assert population_stability_index(a, b) >= 0
