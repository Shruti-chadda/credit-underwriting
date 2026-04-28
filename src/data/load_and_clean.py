"""
load_and_clean.py
=================
Downloads (or generates) the German Credit dataset, cleans it,
runs exploratory data analysis, and saves:
  - data/train.csv
  - data/test.csv
  - eda_plots/*.png

Run from the project root:
    python src/data/load_and_clean.py
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no GUI required
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data")
PLOT_DIR = os.path.join(ROOT, "eda_plots")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD / GENERATE DATA
# ─────────────────────────────────────────────
def load_raw() -> pd.DataFrame:
    """
    Try to fetch the UCI German Credit dataset.
    Fall back to a synthetic version with the same schema
    if the network request fails.
    """
    UCI_URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/statlog/german/german.data"
    )

    # UCI column spec (25 cols, last = target 1=good 2=bad)
    UCI_COLS = [
        "status", "duration", "credit_history", "purpose",
        "credit_amount", "savings", "employment", "installment_rate",
        "personal_status", "other_debtors", "residence_since",
        "property", "age", "other_installments", "housing",
        "existing_credits", "job_uci", "dependents", "telephone",
        "foreign_worker", "target"
    ]

    try:
        print("⬇️  Attempting to download UCI German Credit dataset …")
        raw = urllib.request.urlopen(UCI_URL, timeout=8).read().decode()
        rows = [line.split() for line in raw.strip().splitlines()]
        df_raw = pd.DataFrame(rows, columns=UCI_COLS)

        # Map UCI codes → friendly values
        df = pd.DataFrame()
        df["Sex"]            = df_raw["personal_status"].apply(
            lambda x: "male" if x in ("A91", "A93", "A94") else "female"
        )
        df["Job"]            = df_raw["job_uci"].map(
            {"A171": 0, "A172": 1, "A173": 2, "A174": 3}
        ).fillna(2).astype(int)
        df["Housing"]        = df_raw["housing"].map(
            {"A151": "rent", "A152": "own", "A153": "free"}
        ).fillna("own")
        df["Saving accounts"] = df_raw["savings"].map(
            {"A61": "low", "A62": "low", "A63": "moderate",
             "A64": "high", "A65": "low"}
        ).fillna("low")
        df["Age"]            = df_raw["age"].astype(int)
        df["Credit amount"]  = df_raw["credit_amount"].astype(int)
        df["Duration"]       = df_raw["duration"].astype(int)
        df["Risk"]           = df_raw["target"].astype(int).map({1: 1, 2: 0})

        print(f"✅ Downloaded {len(df)} rows from UCI.")
        return df

    except Exception as e:
        print(f"⚠️  Download failed ({e}). Generating synthetic dataset …")
        return _generate_synthetic()


def _generate_synthetic(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sex      = rng.choice(["male", "female"], n, p=[0.69, 0.31])
    job      = rng.choice([0, 1, 2, 3],       n, p=[0.05, 0.20, 0.63, 0.12])
    housing  = rng.choice(["own", "rent", "free"], n, p=[0.71, 0.18, 0.11])
    saving   = rng.choice(["low", "moderate", "high"], n, p=[0.60, 0.25, 0.15])
    age      = np.clip(rng.normal(35, 11, n).astype(int), 19, 75)
    credit   = np.clip(np.exp(rng.normal(7.5, 0.7, n)).astype(int), 250, 20000)
    duration = rng.choice([6,12,18,24,36,48,60], n,
                           p=[0.05,0.20,0.15,0.25,0.20,0.10,0.05])

    logit = (
        - 0.3 * (sex == "female")
        + 0.2 * job
        - 0.4 * (housing == "rent")
        - 0.5 * (saving == "low")
        + 0.3 * (saving == "high")
        - 0.01 * (credit - 3000) / 1000
        - 0.02 * (duration - 20)
        + 0.015 * (age - 30)
        + rng.normal(0, 0.3, n)
    )
    risk = (1 / (1 + np.exp(-logit)) > 0.45).astype(int)

    return pd.DataFrame({
        "Sex": sex, "Job": job, "Housing": housing,
        "Saving accounts": saving, "Age": age,
        "Credit amount": credit, "Duration": duration, "Risk": risk
    })


# ─────────────────────────────────────────────
# 2. CLEAN DATA
# ─────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sex"]             = df["Sex"].str.lower().str.strip()
    df["Housing"]         = df["Housing"].str.lower().str.strip()
    df["Saving accounts"] = df["Saving accounts"].str.lower().str.strip()

    df["Saving accounts"] = df["Saving accounts"].fillna("low")
    df.dropna(subset=["Risk"], inplace=True)
    df["Risk"] = df["Risk"].astype(int)

    # Remove obvious outliers
    df = df[df["Age"].between(18, 100)]
    df = df[df["Credit amount"].between(100, 100_000)]
    df = df[df["Duration"].between(1, 120)]

    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# 3. EDA
# ─────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> None:
    print("\n📊 Running EDA …")
    print(df.describe())
    print("\nRisk distribution:\n", df["Risk"].value_counts())
    print("\nMissing values:\n", df.isnull().sum())

    palette = {0: "#E74C3C", 1: "#2ECC71"}

    # Plot 1 — Risk distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    df["Risk"].map({1: "Good", 0: "Bad"}).value_counts().plot(
        kind="bar", color=["#2ECC71", "#E74C3C"], ax=ax, edgecolor="black"
    )
    ax.set_title("Risk Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Risk Label"); ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "risk_distribution.png"), dpi=150)
    plt.close(fig)

    # Plot 2 — Age vs Risk
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="Risk", y="Age", data=df,
                hue="Risk", palette={0: "#E74C3C", 1: "#2ECC71"}, legend=False, ax=ax)
    ax.set_title("Age vs Risk", fontsize=14, fontweight="bold")
    ax.set_xticklabels(["Bad Credit", "Good Credit"])
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "age_vs_risk.png"), dpi=150)
    plt.close(fig)

    # Plot 3 — Credit amount distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["Credit amount"], kde=True, color="#3498DB", ax=ax)
    ax.set_title("Credit Amount Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "credit_distribution.png"), dpi=150)
    plt.close(fig)

    # Plot 4 — Saving accounts vs Risk (stacked bar)
    fig, ax = plt.subplots(figsize=(7, 4))
    ct = pd.crosstab(df["Saving accounts"], df["Risk"], normalize="index") * 100
    ct.columns = ["Bad", "Good"]
    ct.plot(kind="bar", stacked=True,
            color=["#E74C3C", "#2ECC71"], ax=ax, edgecolor="black")
    ax.set_title("Saving Accounts vs Risk (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Saving Accounts Level"); ax.set_ylabel("Percentage")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "saving_vs_risk.png"), dpi=150)
    plt.close(fig)

    # Plot 5 — Correlation heatmap (numeric)
    num_df = df.copy()
    num_df["Sex"]             = num_df["Sex"].map({"male": 0, "female": 1})
    num_df["Housing"]         = num_df["Housing"].map({"own": 0, "rent": 1, "free": 2})
    num_df["Saving accounts"] = num_df["Saving accounts"].map({"low": 0, "moderate": 1, "high": 2})
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"✅ EDA plots saved to {PLOT_DIR}/")


# ─────────────────────────────────────────────
# 4. SAVE TRAIN / TEST
# ─────────────────────────────────────────────
def save_splits(df: pd.DataFrame) -> None:
    train, test = train_test_split(df, test_size=0.2, random_state=42,
                                   stratify=df["Risk"])
    train.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(DATA_DIR, "test.csv"),  index=False)
    print(f"✅ Saved {len(train)} train rows → data/train.csv")
    print(f"✅ Saved {len(test)} test rows  → data/test.csv")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df_raw   = load_raw()
    df_clean = clean(df_raw)
    run_eda(df_clean)
    save_splits(df_clean)
    print("\n🎉 load_and_clean.py complete!")
