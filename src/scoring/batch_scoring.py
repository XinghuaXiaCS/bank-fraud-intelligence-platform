"""Batch scoring pipeline.

In a production setting this module would:
  - read a batch of new events from a data warehouse or feature store
  - apply the full feature pipeline
  - score with all models
  - write results back for alert generation

Here it wraps the realtime scoring components with batch-oriented
logging and chunked processing support.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .realtime_scoring import load_artifacts, score_batch

logger = logging.getLogger(__name__)


def score_batch_chunked(
    df: pd.DataFrame,
    models: dict,
    feature_names: list[str],
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """Score a large dataframe in chunks to manage memory.

    This is useful when the dataset is too large to fit in memory at once,
    which is common in daily batch runs over millions of transactions.
    """
    chunks = []
    n = len(df)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = df.iloc[start:end]
        scored = score_batch(chunk, models, feature_names)
        chunks.append(scored)
        logger.info("Scored chunk %d-%d of %d", start, end, n)

    return pd.concat(chunks, ignore_index=True)


def run_batch_pipeline(
    data_path: str | Path,
    model_dir: str | Path,
    output_path: str | Path,
    feature_names: list[str],
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """End-to-end batch scoring: load data, score, write results."""
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    logger.info("Loading models from %s", model_dir)
    models = load_artifacts(str(model_dir))

    logger.info("Scoring %d rows in chunks of %d", len(df), chunk_size)
    scored = score_batch_chunked(df, models, feature_names, chunk_size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    logger.info("Saved scored output to %s", output_path)

    return scored
