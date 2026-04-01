from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out["hour"] = out[timestamp_col].dt.hour
    out["dayofweek"] = out[timestamp_col].dt.dayofweek
    out["month"] = out[timestamp_col].dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    return out


def safe_log1p(series: pd.Series) -> pd.Series:
    return np.log1p(np.maximum(series.astype(float), 0.0))
