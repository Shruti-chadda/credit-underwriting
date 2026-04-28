"""
fairness.py
===========
Basic fairness / bias detection utilities.

Checks demographic parity and equalized odds across
the 'Sex' feature (male vs female).
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def demographic_parity_diff(y_pred: pd.Series, group: pd.Series) -> float:
    """
    Demographic parity difference:
    |P(ŷ=1 | group=A) - P(ŷ=1 | group=B)|

    Closer to 0 is fairer.
    """
    groups = group.unique()
    if len(groups) < 2:
        return 0.0
    rates = [y_pred[group == g].mean() for g in groups]
    return round(abs(rates[0] - rates[1]), 4)


def equalized_odds_diff(y_true: pd.Series,
                        y_pred: pd.Series,
                        group: pd.Series) -> dict:
    """
    Equalized odds: difference in TPR and FPR across groups.
    Returns dict with tpr_diff and fpr_diff.
    """
    groups = group.unique()
    if len(groups) < 2:
        return {"tpr_diff": 0.0, "fpr_diff": 0.0}

    def rates(mask):
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return tpr, fpr

    tprs, fprs = [], []
    for g in groups:
        mask = group == g
        tpr, fpr = rates(mask)
        tprs.append(tpr)
        fprs.append(fpr)

    return {
        "tpr_diff": round(abs(tprs[0] - tprs[1]), 4),
        "fpr_diff": round(abs(fprs[0] - fprs[1]), 4)
    }


def run_fairness_report(df: pd.DataFrame,
                        y_pred_col: str = "y_pred",
                        y_true_col: str = "Risk",
                        group_col: str = "Sex") -> None:
    """
    Print a fairness summary to stdout.

    df must contain columns: group_col, y_true_col, y_pred_col
    """
    y_true = df[y_true_col]
    y_pred = df[y_pred_col]
    group  = df[group_col]

    dp = demographic_parity_diff(y_pred, group)
    eo = equalized_odds_diff(y_true, y_pred, group)

    print("=" * 45)
    print("  FAIRNESS REPORT")
    print("=" * 45)
    print(f"  Group column      : {group_col}")
    print(f"  Groups            : {list(group.unique())}")
    print(f"  Demographic Parity Diff : {dp}")
    print(f"  Equalized Odds - TPR diff : {eo['tpr_diff']}")
    print(f"  Equalized Odds - FPR diff : {eo['fpr_diff']}")
    print("=" * 45)

    if dp > 0.10:
        print("⚠️  WARNING: Demographic parity gap > 10% — possible bias.")
    else:
        print("✅  Demographic parity looks acceptable.")

    if eo["tpr_diff"] > 0.10:
        print("⚠️  WARNING: TPR gap > 10% across groups.")
    if eo["fpr_diff"] > 0.10:
        print("⚠️  WARNING: FPR gap > 10% across groups.")
