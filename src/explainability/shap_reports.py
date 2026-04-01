"""Explainability utilities for the fraud intelligence platform.

Provides two levels of feature explanation:
  1. Built-in feature importance from sklearn models (always available).
  2. SHAP-based explanation (available when the shap package is installed).

Usage:
    from src.explainability.shap_reports import export_feature_importance, explain_with_shap

    # Always works
    export_feature_importance(feature_names, model, "reports/feature_importance.csv")

    # Requires: pip install shap
    explain_with_shap(model, X_sample, feature_names, "reports/shap_summary.csv")
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def export_feature_importance(
    feature_names: list[str],
    model,
    output_path: str | Path,
) -> pd.DataFrame:
    """Export built-in feature importance from a trained model.

    Supports tree-based models (feature_importances_) and linear models (coef_).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        values = np.zeros(len(feature_names))

    df = (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
    )
    df.to_csv(path, index=False)
    return df


def explain_with_shap(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    output_path: str | Path,
    max_samples: int = 500,
) -> pd.DataFrame | None:
    """Compute SHAP values for a model and export mean absolute SHAP importance.

    Returns None if the shap package is not installed.

    Args:
        model: A trained sklearn-compatible classifier.
        X: Feature matrix (DataFrame with feature_names columns).
        feature_names: List of feature column names.
        output_path: Path to save the SHAP importance CSV.
        max_samples: Maximum number of samples to explain (for speed).
    """
    try:
        import shap
    except ImportError:
        print("shap package not installed; skipping SHAP explanation.")
        print("Install with: pip install shap")
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sample = X[feature_names].head(max_samples)

    # Use TreeExplainer for tree-based models, otherwise KernelExplainer
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        # For binary classification, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        background = shap.sample(sample, min(50, len(sample)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
    )
    df.to_csv(path, index=False)
    return df
