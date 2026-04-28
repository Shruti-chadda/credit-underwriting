"""
train_model.py
==============
Train a Random Forest classifier on data/train.csv,
evaluate on data/test.csv, and save models/model.pkl.

Run from the project root:
    python src/models/train_model.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow imports from project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.data.preprocess import load_data, preprocess_data, get_features_target

MODEL_PATH = os.path.join(ROOT, "models", "model.pkl")
PLOT_DIR   = os.path.join(ROOT, "eda_plots")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# LOAD & PREPARE
# ─────────────────────────────────────────────
print("Loading training data …")
df_train = load_data(os.path.join(ROOT, "data", "train.csv"))
df_train = preprocess_data(df_train)
X_train, y_train = get_features_target(df_train)

print("Loading test data …")
df_test = load_data(os.path.join(ROOT, "data", "test.csv"))
df_test = preprocess_data(df_test)
X_test, y_test = get_features_target(df_test)

print(f"Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")
print(f"Class balance (train): {dict(y_train.value_counts())}")


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("\n🤖 Training Random Forest …")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)


# ─────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring="roc_auc")
print(f"\n5-Fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


# ─────────────────────────────────────────────
# EVALUATE ON TEST SET
# ─────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"\nTest AUC : {roc_auc_score(y_test, y_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Bad", "Good"]))


# ─────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["Bad", "Good"]).plot(ax=ax, colorbar=False)
ax.set_title("Confusion Matrix (Test Set)", fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)
print("Confusion matrix saved → eda_plots/confusion_matrix.png")


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))
importances.plot(kind="barh", color="#3498DB", edgecolor="black", ax=ax)
ax.set_title("Feature Importances", fontsize=14, fontweight="bold")
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "feature_importance.png"), dpi=150)
plt.close(fig)
print("Feature importance plot saved → eda_plots/feature_importance.png")


# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"\n✅ Model saved → {MODEL_PATH}")
print("\n🎉 Training complete!")
