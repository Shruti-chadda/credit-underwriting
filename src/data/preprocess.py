"""
preprocess.py — shared preprocessing (Sex removed for fairness)
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

HOUSING_MAP = {"own": 0, "rent": 1, "free": 2}
SAVING_MAP  = {"low": 0, "moderate": 1, "high": 2}

# Sex intentionally excluded — fairness-aware model
FEATURE_COLS = ["Job", "Housing", "Saving accounts", "Age", "Credit amount", "Duration"]


def load_data(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(ROOT, "data", "train.csv")
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Sex", "Housing", "Saving accounts"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    if "Sex"             in df.columns: df["Sex"]             = df["Sex"].map({"male": 0, "female": 1})
    if "Housing"         in df.columns: df["Housing"]         = df["Housing"].map(HOUSING_MAP)
    if "Saving accounts" in df.columns: df["Saving accounts"] = df["Saving accounts"].map(SAVING_MAP)

    df.fillna(0, inplace=True)
    return df


def get_features_target(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["Risk"]
    return X, y