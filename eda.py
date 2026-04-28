"""
eda.py
======
Exploratory Data Analysis for the German Credit Risk dataset.
Generates and saves all plots to eda_plots/.

Run from the project root:
    python eda.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no GUI popup, saves to files
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
PLOT_DIR = os.path.join(ROOT, "eda_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.dpi"]     = 150
plt.rcParams["figure.figsize"] = (8, 4)

GOOD_COLOR = "#2ECC71"
BAD_COLOR  = "#E74C3C"
BLUE       = "#3498DB"
PURPLE     = "#9B59B6"
ORANGE     = "#F39C12"


def save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved → eda_plots/{name}")


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("\n📂 Loading dataset …")
df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print(f"   Shape   : {df.shape}")
print(f"   Columns : {list(df.columns)}")
print(f"\n{df.head()}\n")


# ─────────────────────────────────────────────
# 2. BASIC SUMMARY
# ─────────────────────────────────────────────
print("=" * 50)
print("  BASIC SUMMARY")
print("=" * 50)
print("\n--- Data Types ---")
print(df.dtypes)
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Descriptive Statistics ---")
print(df.describe())
print("\n--- Risk Distribution ---")
print(df["Risk"].value_counts().rename({1: "Good Credit", 0: "Bad Credit"}))


# ─────────────────────────────────────────────
# 3. PLOT 1 — Risk Distribution
# ─────────────────────────────────────────────
print("\n📊 Generating plots …")

risk_labeled = df["Risk"].map({1: "Good Credit", 0: "Bad Credit"})
counts = risk_labeled.value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar
axes[0].bar(counts.index, counts.values,
            color=[GOOD_COLOR, BAD_COLOR], edgecolor="black", width=0.5)
axes[0].set_title("Risk Distribution (Count)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

# Pie
axes[1].pie(counts.values, labels=counts.index,
            colors=[GOOD_COLOR, BAD_COLOR],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Risk Distribution (%)", fontsize=13, fontweight="bold")

plt.suptitle("Credit Risk Distribution", fontsize=15, fontweight="bold")
plt.tight_layout()
save(fig, "01_risk_distribution.png")


# ─────────────────────────────────────────────
# 4. PLOT 2 — Age Analysis
# ─────────────────────────────────────────────
good_age = df[df["Risk"] == 1]["Age"]
bad_age  = df[df["Risk"] == 0]["Age"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(df["Age"], bins=20, color=BLUE, edgecolor="black", alpha=0.8)
axes[0].set_title("Age Distribution", fontweight="bold")
axes[0].set_xlabel("Age"); axes[0].set_ylabel("Count")

axes[1].boxplot([bad_age, good_age], labels=["Bad Credit", "Good Credit"],
                patch_artist=True,
                boxprops=dict(facecolor=BLUE, alpha=0.6),
                medianprops=dict(color="white", linewidth=2))
axes[1].set_title("Age by Risk", fontweight="bold")
axes[1].set_ylabel("Age")

axes[2].hist(bad_age,  bins=20, alpha=0.6, color=BAD_COLOR,  label="Bad Credit",  density=True)
axes[2].hist(good_age, bins=20, alpha=0.6, color=GOOD_COLOR, label="Good Credit", density=True)
axes[2].set_title("Age Density by Risk", fontweight="bold")
axes[2].legend()

plt.suptitle("Age Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "02_age_analysis.png")

print(f"   Mean Age — Good: {good_age.mean():.1f}  |  Bad: {bad_age.mean():.1f}")


# ─────────────────────────────────────────────
# 5. PLOT 3 — Credit Amount Analysis
# ─────────────────────────────────────────────
good_c = df[df["Risk"] == 1]["Credit amount"]
bad_c  = df[df["Risk"] == 0]["Credit amount"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(df["Credit amount"], bins=30, color=PURPLE, edgecolor="black", alpha=0.8)
axes[0].set_title("Credit Amount Distribution", fontweight="bold")
axes[0].set_xlabel("Credit Amount")

axes[1].boxplot([bad_c, good_c], labels=["Bad Credit", "Good Credit"],
                patch_artist=True,
                boxprops=dict(facecolor=PURPLE, alpha=0.6),
                medianprops=dict(color="white", linewidth=2))
axes[1].set_title("Credit Amount by Risk", fontweight="bold")

axes[2].hist(bad_c,  bins=25, alpha=0.6, color=BAD_COLOR,  label="Bad Credit",  density=True)
axes[2].hist(good_c, bins=25, alpha=0.6, color=GOOD_COLOR, label="Good Credit", density=True)
axes[2].set_title("Credit Amount Density", fontweight="bold")
axes[2].legend()

plt.suptitle("Credit Amount Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "03_credit_amount_analysis.png")

print(f"   Mean Credit — Good: {good_c.mean():,.0f}  |  Bad: {bad_c.mean():,.0f}")


# ─────────────────────────────────────────────
# 6. PLOT 4 — Loan Duration Analysis
# ─────────────────────────────────────────────
good_d = df[df["Risk"] == 1]["Duration"]
bad_d  = df[df["Risk"] == 0]["Duration"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

dur_risk = df.groupby(["Duration", "Risk"]).size().unstack(fill_value=0)
dur_risk.columns = ["Bad Credit", "Good Credit"]
dur_risk.plot(kind="bar", ax=axes[0],
              color=[BAD_COLOR, GOOD_COLOR], edgecolor="black")
axes[0].set_title("Duration vs Risk (Count)", fontweight="bold")
axes[0].set_xlabel("Duration (months)")
axes[0].tick_params(axis="x", rotation=45)

axes[1].boxplot([bad_d, good_d], labels=["Bad Credit", "Good Credit"],
                patch_artist=True,
                boxprops=dict(facecolor=ORANGE, alpha=0.7),
                medianprops=dict(color="white", linewidth=2))
axes[1].set_title("Duration by Risk", fontweight="bold")
axes[1].set_ylabel("Duration (months)")

plt.suptitle("Loan Duration Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "04_duration_analysis.png")


# ─────────────────────────────────────────────
# 7. PLOT 5 — Categorical Features vs Risk (Count)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
cat_features = ["Sex", "Housing", "Saving accounts"]

for ax, feat in zip(axes, cat_features):
    ct = pd.crosstab(df[feat], df["Risk"])
    ct.columns = ["Bad Credit", "Good Credit"]
    ct.plot(kind="bar", ax=ax, color=[BAD_COLOR, GOOD_COLOR], edgecolor="black")
    ax.set_title(f"{feat} vs Risk", fontweight="bold")
    ax.set_xlabel(feat)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="upper right")

plt.suptitle("Categorical Features vs Risk (Count)", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "05_categorical_vs_risk_count.png")


# ─────────────────────────────────────────────
# 8. PLOT 6 — Risk Rate by Category (%)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, feat in zip(axes, cat_features):
    ct = pd.crosstab(df[feat], df["Risk"], normalize="index") * 100
    ct.columns = ["Bad %", "Good %"]
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=[BAD_COLOR, GOOD_COLOR], edgecolor="black")
    ax.set_title(f"{feat} — Risk Rate (%)", fontweight="bold")
    ax.set_ylabel("Percentage")
    ax.tick_params(axis="x", rotation=15)
    ax.axhline(50, color="white", linestyle="--", linewidth=1)

plt.suptitle("Risk Rate by Category (%)", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "06_categorical_risk_rate_pct.png")


# ─────────────────────────────────────────────
# 9. PLOT 7 — Job Level Analysis
# ─────────────────────────────────────────────
job_labels = {
    0: "Unskilled\n(non-res)", 1: "Unskilled\n(res)",
    2: "Skilled", 3: "Highly\nQualified"
}
df["Job Label"] = df["Job"].map(job_labels)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ct = pd.crosstab(df["Job Label"], df["Risk"])
ct.columns = ["Bad Credit", "Good Credit"]
ct.plot(kind="bar", ax=axes[0], color=[BAD_COLOR, GOOD_COLOR], edgecolor="black")
axes[0].set_title("Job Level vs Risk (Count)", fontweight="bold")
axes[0].tick_params(axis="x", rotation=10)

ct_pct = pd.crosstab(df["Job Label"], df["Risk"], normalize="index") * 100
ct_pct.columns = ["Bad %", "Good %"]
ct_pct.plot(kind="bar", stacked=True, ax=axes[1],
            color=[BAD_COLOR, GOOD_COLOR], edgecolor="black")
axes[1].set_title("Job Level vs Risk (%)", fontweight="bold")
axes[1].tick_params(axis="x", rotation=10)

plt.suptitle("Job Level Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "07_job_level_analysis.png")
df.drop(columns=["Job Label"], inplace=True)


# ─────────────────────────────────────────────
# 10. PLOT 8 — Saving Accounts vs Risk (stacked %)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ct = pd.crosstab(df["Saving accounts"], df["Risk"], normalize="index") * 100
ct.columns = ["Bad %", "Good %"]
ct = ct.reindex(["low", "moderate", "high"])
ct.plot(kind="bar", stacked=True, ax=ax,
        color=[BAD_COLOR, GOOD_COLOR], edgecolor="black")
ax.set_title("Saving Accounts vs Risk (%)", fontsize=13, fontweight="bold")
ax.set_xlabel("Saving Accounts Level"); ax.set_ylabel("Percentage")
ax.tick_params(axis="x", rotation=0)
ax.axhline(50, color="white", linestyle="--", linewidth=1)
plt.tight_layout()
save(fig, "08_saving_vs_risk.png")


# ─────────────────────────────────────────────
# 11. PLOT 9 — Correlation Heatmap
# ─────────────────────────────────────────────
num_df = df.copy()
num_df["Sex"]             = num_df["Sex"].map({"male": 0, "female": 1})
num_df["Housing"]         = num_df["Housing"].map({"own": 0, "rent": 1, "free": 2})
num_df["Saving accounts"] = num_df["Saving accounts"].map({"low": 0, "moderate": 1, "high": 2})

corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, ax=ax, linewidths=0.5,
            annot_kws={"size": 10})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "09_correlation_heatmap.png")


# ─────────────────────────────────────────────
# 12. PLOT 10 — Outlier Detection (Box plots)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
num_feats = ["Age", "Credit amount", "Duration"]

for ax, feat in zip(axes, num_feats):
    ax.boxplot(df[feat], patch_artist=True,
               boxprops=dict(facecolor=BLUE, alpha=0.6),
               medianprops=dict(color="white", linewidth=2))
    ax.set_title(f"{feat} — Outliers", fontweight="bold")
    ax.set_ylabel(feat)

    q1, q3 = df[feat].quantile([0.25, 0.75])
    iqr     = q3 - q1
    n_out   = len(df[(df[feat] < q1 - 1.5 * iqr) | (df[feat] > q3 + 1.5 * iqr)])
    ax.set_xlabel(f"Outliers detected: {n_out}")

plt.suptitle("Outlier Detection (IQR Method)", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "10_outlier_detection.png")


# ─────────────────────────────────────────────
# 13. PLOT 11 — Pairplot (numeric features)
# ─────────────────────────────────────────────
pair_df = df[["Age", "Credit amount", "Duration", "Risk"]].copy()
pair_df["Risk"] = pair_df["Risk"].map({1: "Good", 0: "Bad"})

g = sns.pairplot(pair_df, hue="Risk",
                 palette={"Good": GOOD_COLOR, "Bad": BAD_COLOR},
                 plot_kws={"alpha": 0.5}, diag_kind="kde")
g.fig.suptitle("Pairplot — Numeric Features by Risk", y=1.02,
               fontsize=13, fontweight="bold")
save(g.fig, "11_pairplot.png")


# ─────────────────────────────────────────────
# 14. PRINT KEY INSIGHTS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  KEY INSIGHTS")
print("=" * 55)

insights = [
    ("Risk Balance",     f"{(df['Risk']==1).mean()*100:.1f}% Good / {(df['Risk']==0).mean()*100:.1f}% Bad — balanced dataset"),
    ("Age",              f"Good mean: {good_age.mean():.1f} yrs  |  Bad mean: {bad_age.mean():.1f} yrs"),
    ("Credit Amount",    f"Good mean: {good_c.mean():,.0f}  |  Bad mean: {bad_c.mean():,.0f}"),
    ("Duration",         f"Good mean: {good_d.mean():.1f} mo  |  Bad mean: {bad_d.mean():.1f} mo"),
    ("Saving Accounts",  "'High' savers → safest group; 'Low' savers → highest risk"),
    ("Housing",          "Owners have better credit than renters"),
    ("Gender",           f"Male: {(df[df['Sex']=='male']['Risk'].mean()*100):.1f}% good  |  Female: {(df[df['Sex']=='female']['Risk'].mean()*100):.1f}% good"),
    ("Top Features",     "Saving Accounts, Duration, Credit Amount, Age"),
]

for feat, insight in insights:
    print(f"  {feat:<20}: {insight}")

print("=" * 55)
print(f"\n🎉 EDA complete! All 11 plots saved to → eda_plots/\n")