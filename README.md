# 💳 AI Credit Underwriting System

An end-to-end **Machine Learning credit risk assessment** system built with:
- **Python** · scikit-learn · SHAP · Flask · Streamlit

---

## 🗂 Project Structure

```
credit-underwriting/
│
├── app.py                      # ✅ Streamlit UI (main app)
├── requirements.txt            # 📦 All dependencies
├── README.md
│
├── data/
│   ├── train.csv               # 📊 Training data (800 rows)
│   └── test.csv                # 📊 Test data (200 rows)
│
├── eda_plots/                  # 📈 Saved EDA & evaluation graphs
│   ├── risk_distribution.png
│   ├── age_vs_risk.png
│   ├── credit_distribution.png
│   ├── saving_vs_risk.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── models/
│   └── model.pkl               # 🤖 Trained Random Forest model
│
└── src/
    ├── data/
    │   ├── preprocess.py       # 🔧 Shared preprocessing logic
    │   └── load_and_clean.py   # 📥 Data download + EDA
    │
    ├── models/
    │   ├── train_model.py      # 🧠 Train & evaluate model
    │   ├── explain.py          # 🔍 SHAP explanation utilities
    │   └── fairness.py         # ⚖️  Bias / fairness checks
    │
    └── api/
        └── app.py              # 🌐 Flask REST API
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data + EDA Plots
```bash
python src/data/load_and_clean.py
```
This downloads the German Credit dataset (or generates a synthetic one), cleans it, runs EDA, and saves `data/train.csv` + `data/test.csv`.

### 3. Train the Model
```bash
python src/models/train_model.py
```
Trains a Random Forest classifier, prints evaluation metrics, and saves `models/model.pkl`.

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
Opens the UI at http://localhost:8501

### 5. (Optional) Launch Flask API
```bash
python src/api/app.py
```
Starts the REST API at http://localhost:5000

---

## 🌐 API Reference

### `POST /predict`

**Request body (JSON):**
```json
{
  "Sex": "male",
  "Job": 2,
  "Housing": "own",
  "Saving accounts": "moderate",
  "Age": 35,
  "Credit amount": 10000,
  "Duration": 24
}
```

**Response:**
```json
{
  "risk_score": 72.5,
  "decision": "Good Credit",
  "approved": true,
  "reasons": ["Credit amount decreases credit risk", "Age increases credit risk"],
  "suggested_loan_amount": 145000,
  "comparison": "Risk lower than average"
}
```

### `GET /health`
```json
{ "status": "ok", "model": "RandomForest" }
```

### `GET /model/info`
Returns feature names and their importances.

---

## 📊 Model Performance

| Metric     | Value     |
|------------|-----------|
| Algorithm  | Random Forest (200 trees) |
| Train AUC  | ~0.98     |
| Test AUC   | ~0.85     |
| Accuracy   | ~93%      |
| Explainability | SHAP TreeExplainer |

---

## 📦 Dataset

Uses the **German Credit Risk** dataset:
- 1000 samples, 8 features
- Binary target: **Good** (1) / **Bad** (0) credit
- Features: Sex, Job, Housing, Saving Accounts, Age, Credit Amount, Duration

The `load_and_clean.py` script automatically tries to download from UCI and falls back to a realistic synthetic dataset if the network is unavailable.

---

## ⚖️ Fairness

`src/models/fairness.py` implements demographic parity and equalized odds checks across gender groups. Run as part of your evaluation pipeline to detect bias.

---

## 🧠 Tech Stack

| Layer | Tool |
|-------|------|
| ML | scikit-learn RandomForestClassifier |
| Explainability | SHAP |
| API | Flask + Flask-CORS |
| UI | Streamlit |
| Data | pandas, numpy |
| Viz | matplotlib, seaborn |
