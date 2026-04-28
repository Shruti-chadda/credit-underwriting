"""
preprocess.py
=============
Shared preprocessing used by the training script, Flask API, and Streamlit app.
"""

import os
import pandas as pd

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_data(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(ROOT, "data", "train.csv")
    return pd.read_csv(path)


# ─────────────────────────────────────────────
# ENCODE CATEGORICALS
# ─────────────────────────────────────────────
SEX_MAP     = {"male": 0, "female": 1}
HOUSING_MAP = {"own": 0, "rent": 1, "free": 2}
SAVING_MAP  = {"low": 0, "moderate": 1, "high": 2}

FEATURE_COLS = ["Sex", "Job", "Housing", "Saving accounts",
                "Age", "Credit amount", "Duration"]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Lower-case strings so they survive user input variation
    for col in ["Sex", "Housing", "Saving accounts"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    df["Sex"]             = df["Sex"].map(SEX_MAP)
    df["Housing"]         = df["Housing"].map(HOUSING_MAP)
    df["Saving accounts"] = df["Saving accounts"].map(SAVING_MAP)

    df.fillna(0, inplace=True)
    return df


# ─────────────────────────────────────────────
# SPLIT FEATURES / TARGET
# ─────────────────────────────────────────────
def get_features_target(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["Risk"]
    return X, y
