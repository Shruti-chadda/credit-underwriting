"""
src/api/app.py
==============
Flask REST API for the Credit Underwriting model.

Run from project root:
    python src/api/app.py

Endpoints:
    POST /predict       — predict risk for a single applicant
    GET  /health        — health check
    GET  /model/info    — model feature importances
"""

import os
import sys
import pickle

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Allow imports from root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.data.preprocess import load_data, preprocess_data, get_features_target
from src.models.explain import build_explainer, get_top_reasons

# ─────────────────────────────────────────────
# INIT APP
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# LOAD MODEL + BACKGROUND DATA
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "models", "model.pkl")

print("Loading model …")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Loading background data for SHAP …")
df_bg = preprocess_data(load_data())
X_bg, _ = get_features_target(df_bg)

explainer = build_explainer(model, X_bg)
FEATURE_COLS = list(X_bg.columns)
print("✅ API ready.")


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "RandomForest"})


# ─────────────────────────────────────────────
# MODEL INFO
# ─────────────────────────────────────────────
@app.route("/model/info", methods=["GET"])
def model_info():
    importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    return jsonify({"features": FEATURE_COLS, "importances": importances})


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    required = ["Sex", "Job", "Housing", "Saving accounts",
                "Age", "Credit amount", "Duration"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        input_df = pd.DataFrame([{
            "Sex":             data["Sex"],
            "Job":             int(data["Job"]),
            "Housing":         data["Housing"],
            "Saving accounts": data["Saving accounts"],
            "Age":             int(data["Age"]),
            "Credit amount":   float(data["Credit amount"]),
            "Duration":        int(data["Duration"]),
        }])

        processed = preprocess_data(input_df)
        X_input   = processed[FEATURE_COLS]

        prob       = float(model.predict_proba(X_input)[0][1])
        prediction = int(model.predict(X_input)[0])

        reasons = get_top_reasons(explainer, X_input, top_n=3)

        suggested_loan = int(200_000 * prob) if prediction == 1 else int(50_000 * prob)

        return jsonify({
            "risk_score":          round(prob * 100, 2),
            "decision":            "Good Credit" if prediction == 1 else "Bad Credit",
            "approved":            prediction == 1,
            "reasons":             [r["reason"] for r in reasons],
            "top_features":        reasons,
            "suggested_loan_amount": suggested_loan,
            "comparison":          (
                "Risk lower than average" if prob < 0.5
                else "Risk higher than average"
            )
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
