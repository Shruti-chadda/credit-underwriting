"""
explain.py  —  SHAP explanation utilities
"""

import shap
import numpy as np
import pandas as pd


def build_explainer(model, X_background: pd.DataFrame) -> shap.TreeExplainer:
    background = shap.sample(X_background, min(200, len(X_background)), random_state=42)
    return shap.TreeExplainer(model, background)


def get_top_reasons(
    explainer: shap.TreeExplainer,
    input_df: pd.DataFrame,
    top_n: int = 3
) -> list:
    """
    Returns top_n features ranked by |SHAP value| for class-1 (Good Credit).

    Handles all SHAP output shapes:
      - list of arrays  [class0_arr, class1_arr]   — older sklearn RF
      - 3-D array       (n_samples, n_features, n_classes)  — newer sklearn RF
      - 2-D array       (n_samples, n_features)              — single-output
    """
    raw = explainer.shap_values(input_df)
    arr = np.array(raw)

    if arr.ndim == 3:
        # shape (n_samples, n_features, n_classes) → class-1 for first sample
        vals = arr[0, :, 1]
    elif arr.ndim == 2 and arr.shape[0] == 2:
        # list-of-arrays stacked → [class0, class1], each (n_samples, n_features)
        vals = arr[1, 0, :]  if arr.shape[1] == 1 else arr[1][0]
    elif arr.ndim == 2:
        # (n_samples, n_features)
        vals = arr[0]
    else:
        # fallback: flatten and take first n_features values
        vals = arr.flatten()[:len(input_df.columns)]

    vals = np.array(vals, dtype=float).flatten()

    pairs = sorted(
        zip(input_df.columns, vals),
        key=lambda x: abs(float(x[1])),
        reverse=True
    )

    results = []
    for feature, sv in pairs[:top_n]:
        sv = float(sv)
        direction = "increases" if sv > 0 else "decreases"
        results.append({
            "feature":   feature,
            "shap_val":  round(sv, 4),
            "direction": direction,
            "reason":    f"{feature} {direction} credit risk"
        })

    return results