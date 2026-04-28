"""
app.py  —  AI Credit Underwriting System
Run: streamlit run app.py
"""

import os, sys, pickle
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.preprocess import preprocess_data, load_data, get_features_target
from src.models.explain import build_explainer, get_top_reasons

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditAI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* ── App shell ── */
.stApp {
    background: #F4F1E8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Block container width ── */
.block-container {
    max-width: 1000px !important;
    padding: 0 2rem 4rem !important;
    margin: 0 auto !important;
}

/* ── Headings reset ── */
h1, h2, h3, h4 { font-family: 'DM Sans', sans-serif !important; }

/* ── Buttons ── */
.stButton > button {
    background: #3D7A3D !important;
    color: #F4F1E8 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 11px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    letter-spacing: .01em !important;
    transition: background .2s !important;
}
.stButton > button:hover { background: #2A5A2A !important; }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input {
    background: #FAFAF5 !important;
    border: 1px solid #C8C4B0 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    color: #1C2B1C !important;
    padding: 10px 14px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #6BAE6B !important;
    box-shadow: 0 0 0 3px rgba(107,174,107,.12) !important;
}

div[data-baseweb="select"] > div {
    background: #FAFAF5 !important;
    border: 1px solid #C8C4B0 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}

/* ── Labels ── */
label, .stSelectbox label, .stNumberInput label, .stTextInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #5A6E5A !important;
    text-transform: uppercase !important;
    letter-spacing: .05em !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: #6BAE6B !important; border-radius: 4px !important; }
[data-testid="stProgressBar"] > div {
    background: #DDD9CC !important; border-radius: 4px !important;
}

/* ── Info / success boxes ── */
[data-testid="stInfo"]    { background: #E5EFE5 !important; border-color: #A8CDA8 !important; border-radius: 10px !important; font-family: 'DM Sans',sans-serif !important; }
[data-testid="stWarning"] { background: #FEF6E4 !important; border-radius: 10px !important; }
[data-testid="stSuccess"] { background: #E5EFE5 !important; border-radius: 10px !important; }

/* ── Tables ── */
table { font-family: 'DM Sans', sans-serif !important; border-collapse: collapse; width: 100%; }
thead th {
    background: #E5EFE5 !important;
    color: #2A5A2A !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: .05em !important;
    padding: 10px 14px !important;
}
tbody td { padding: 10px 14px !important; font-size: 13px !important; color: #1C2B1C !important; border-bottom: 1px solid #E0DCD0 !important; }
tbody tr:last-child td { border-bottom: none !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #FAFAF5 !important;
    border: 1px solid #D0CCBC !important;
    border-radius: 10px !important;
    padding: 18px !important;
}
[data-testid="stMetricLabel"] { font-size: 11px !important; color: #8A9E8A !important; font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: .05em !important; }
[data-testid="stMetricValue"] { font-size: 1.5rem !important; color: #3D7A3D !important; font-weight: 600 !important; }

/* ── Dataframe ── */
.stDataFrame { border: 1px solid #D0CCBC !important; border-radius: 10px !important; overflow: hidden !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# LOAD MODEL  (cached)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    with open(os.path.join(ROOT, "models", "model.pkl"), "rb") as f:
        mdl = pickle.load(f)
    df_bg   = preprocess_data(load_data(os.path.join(ROOT, "data", "train.csv")))
    X_bg, _ = get_features_target(df_bg)
    exp     = build_explainer(mdl, X_bg)
    return mdl, exp, list(X_bg.columns)

model, explainer, FEATURE_COLS = load_resources()


# ─────────────────────────────────────────────────────────────────────
# SESSION STATE  (active page)
# ─────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"


# ─────────────────────────────────────────────────────────────────────
# HELPER COMPONENTS
# ─────────────────────────────────────────────────────────────────────
def section_label(text: str, margin_top: str = "28px") -> None:
    st.markdown(
        f"<div style='font-size:11px;font-weight:600;color:#8A9E8A;"
        f"text-transform:uppercase;letter-spacing:.08em;margin:{margin_top} 0 10px'>"
        f"{text}</div>",
        unsafe_allow_html=True
    )


def metric_card(label: str, value: str, color: str = "#3D7A3D") -> None:
    st.markdown(f"""
    <div style="background:#FAFAF5;border:1px solid #D0CCBC;border-radius:10px;
        padding:18px 14px;text-align:center">
        <div style="font-size:10px;font-weight:600;color:#8A9E8A;text-transform:uppercase;
            letter-spacing:.07em;margin-bottom:7px">{label}</div>
        <div style="font-size:1.35rem;font-weight:600;color:{color};
            font-family:'DM Sans',sans-serif">{value}</div>
    </div>""", unsafe_allow_html=True)


def feature_card(icon: str, title: str, desc: str) -> None:
    st.markdown(f"""
    <div style="background:#FAFAF5;border:1px solid #D0CCBC;border-radius:12px;
        padding:22px;border-top:3px solid #6BAE6B">
        <div style="font-size:26px;margin-bottom:11px">{icon}</div>
        <div style="font-size:14px;font-weight:600;color:#1C2B1C;margin-bottom:7px">{title}</div>
        <div style="font-size:12px;color:#8A9E8A;line-height:1.65">{desc}</div>
    </div>""", unsafe_allow_html=True)


def reason_card(feature: str, direction: str, shap_val: float) -> None:
    is_up   = direction == "increases"
    tag_bg  = "#FEF3E2" if is_up else "#E5EFE5"
    tag_col = "#8B6014" if is_up else "#2A5A2A"
    tag_txt = "↑ Increases risk" if is_up else "↓ Decreases risk"
    bar_col = "#F6C765"  if is_up else "#6BAE6B"
    bar_w   = min(int(abs(shap_val) * 700), 100)
    st.markdown(f"""
    <div style="background:#FAFAF5;border:1px solid #D0CCBC;border-radius:10px;
        padding:14px 18px;display:flex;align-items:center;gap:16px;margin-bottom:10px">
        <div style="flex:1">
            <div style="font-size:13px;font-weight:500;color:#1C2B1C;margin-bottom:6px">{feature}</div>
            <div style="height:5px;border-radius:3px;background:#E8E4D8">
                <div style="height:100%;width:{bar_w}%;background:{bar_col};border-radius:3px"></div>
            </div>
            <div style="font-size:11px;color:#8A9E8A;margin-top:4px">SHAP: {shap_val:+.4f}</div>
        </div>
        <span style="font-size:11px;padding:5px 11px;border-radius:20px;
            background:{tag_bg};color:{tag_col};font-weight:500;white-space:nowrap">
            {tag_txt}
        </span>
    </div>""", unsafe_allow_html=True)


def divider() -> None:
    st.markdown(
        "<hr style='border:none;border-top:1px solid #D8D4C4;margin:32px 0'>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────
# HERO  (always visible — title + top nav)
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:52px 0 8px">
    <div style="display:inline-flex;align-items:center;gap:10px;
        background:#E5EFE5;border:1px solid #A8CDA8;border-radius:24px;
        padding:6px 16px;font-size:12px;color:#2A5A2A;font-weight:500;margin-bottom:22px">
        <span style="width:7px;height:7px;background:#3D7A3D;border-radius:50%;display:inline-block"></span>
        German Credit Risk Model · v1.0
    </div>
    <div style="font-family:'DM Serif Display',serif;font-size:2.6rem;
        font-weight:400;color:#1C2B1C;line-height:1.2;margin-bottom:12px">
        AI Credit Underwriting
    </div>
    <div style="font-size:15px;color:#8A9E8A;font-weight:400;margin-bottom:32px">
        Instant loan decisions powered by Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)

# ── Top nav ──
pages   = ["home", "apply", "eda", "about"]
labels  = ["🏠 Home", "📋 Apply for Loan", "📊 EDA Dashboard", "ℹ️ About"]
cur     = st.session_state.page

cols = st.columns([1, 2, 2, 2, 2, 1])
for col, pg, lb in zip(cols[1:], pages, labels):
    active_style = (
        "background:#E5EFE5;color:#2A5A2A;border:1.5px solid #A8CDA8;font-weight:600"
        if cur == pg else
        "background:transparent;color:#8A9E8A;border:1.5px solid transparent"
    )
    if col.button(lb, key=f"nav_{pg}", use_container_width=True):
        st.session_state.page = pg
        st.rerun()

st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] > div:nth-child(n+2):nth-child(-n+5) button {
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 8px 4px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all .15s !important;
}
</style>""", unsafe_allow_html=True)

divider()

# ─────────────────────────────────────────────────────────────────────
# PAGE  →  HOME
# ─────────────────────────────────────────────────────────────────────
if st.session_state.page == "home":

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        feature_card("⚡", "Instant Decisions",
            "Risk assessment in seconds using a trained Random Forest classifier with 93%+ accuracy.")
    with c2:
        feature_card("🔍", "Explainable AI",
            "Every decision backed by SHAP values — know exactly which factors drove the result.")
    with c3:
        feature_card("📊", "Data-Driven",
            "Trained on the German Credit dataset. AUC 0.98, 5-fold cross-validated model.")

    divider()

    # Stats row
    section_label("Model at a glance", margin_top="0")
    s1, s2, s3, s4 = st.columns(4, gap="small")
    with s1: metric_card("Train AUC", "0.98")
    with s2: metric_card("Test AUC",  "0.688")
    with s3: metric_card("CV Folds",  "5")
    with s4: metric_card("Features",  "7")

    divider()

    # CTA
    st.markdown("""
    <div style="background:#FAFAF5;border:1px solid #D0CCBC;border-radius:14px;
        padding:32px;text-align:center">
        <div style="font-size:1.1rem;font-weight:600;color:#1C2B1C;margin-bottom:8px">
            Ready to assess a loan application?
        </div>
        <div style="font-size:13px;color:#8A9E8A;margin-bottom:0">
            Click <strong style="color:#3D7A3D">Apply for Loan</strong> in the navigation above
            to get an instant AI-powered credit decision.
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE  →  APPLY
# ─────────────────────────────────────────────────────────────────────
elif st.session_state.page == "apply":

    st.markdown("""
    <div style="margin-bottom:24px">
        <div style="font-size:1.3rem;font-weight:600;color:#1C2B1C;margin-bottom:5px">Loan Application</div>
        <div style="font-size:13px;color:#8A9E8A">Fill in the applicant's details for an instant credit risk assessment.</div>
    </div>""", unsafe_allow_html=True)

    with st.form("loan_form"):
        section_label("Personal & Financial Details", margin_top="0")
        c1, c2 = st.columns(2, gap="large")

        with c1:
            name    = st.text_input("Applicant Name", placeholder="e.g. Rahul Sharma")
            gender  = st.selectbox("Gender", ["Male", "Female"])
            job     = st.selectbox("Employment Type", [0, 1, 2, 3],
                        format_func=lambda x: {
                            0: "Unskilled — non-resident",
                            1: "Unskilled — resident",
                            2: "Skilled employee",
                            3: "Highly qualified"
                        }[x])
            housing = st.selectbox("Housing", ["Own", "Rent", "Free"])

        with c2:
            age           = st.number_input("Age", 18, 100, 30)
            saving        = st.selectbox("Saving Accounts", ["Low", "Moderate", "High"])
            credit_amount = st.number_input("Credit Amount (₹)", 500, 200_000, 10_000, step=500)
            duration      = st.number_input("Loan Duration (months)", 6, 120, 24, step=6)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Credit Risk →", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "Sex": gender, "Job": job, "Housing": housing,
            "Saving accounts": saving, "Age": age,
            "Credit amount": credit_amount, "Duration": duration
        }])
        processed  = preprocess_data(input_df)
        X_input    = processed[FEATURE_COLS]
        prob       = float(model.predict_proba(X_input)[0][1])
        prediction = int(model.predict(X_input)[0])
        reasons    = get_top_reasons(explainer, X_input, top_n=3)
        suggested  = int(200_000 * prob) if prediction == 1 else int(50_000 * prob)
        label      = "Good Credit" if prediction == 1 else "Bad Credit"
        val_color  = "#3D7A3D" if prediction == 1 else "#B03030"

        divider()
        section_label("Assessment Result", margin_top="0")

        c1, c2, c3 = st.columns(3, gap="small")
        with c1: metric_card("Credit Decision", label, val_color)
        with c2: metric_card("Approval Probability", f"{round(prob*100, 1)}%")
        with c3: metric_card("Suggested Loan", f"₹{suggested:,}", "#8B6014")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        bar_color = "#6BAE6B" if prob < 0.5 else "#F6C765"
        st.markdown(f"""
        <div style="background:#FAFAF5;border:1px solid #D0CCBC;border-radius:10px;padding:18px 20px">
            <div style="display:flex;justify-content:space-between;font-size:12px;
                color:#8A9E8A;margin-bottom:8px">
                <span style="font-weight:500">Risk Score</span>
                <span style="font-weight:600;color:#1C2B1C">{round(prob*100,1)}%</span>
            </div>
            <div style="height:8px;background:#E8E4D8;border-radius:4px;overflow:hidden">
                <div style="height:100%;width:{round(prob*100,1)}%;background:{bar_color};border-radius:4px"></div>
            </div>
            <div style="font-size:11px;color:#8A9E8A;margin-top:6px">
                {"✓ Risk lower than average applicant" if prob < 0.5 else "⚠ Risk higher than average applicant"}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        section_label("Why this decision? — Top 3 Factors")
        for r in reasons:
            reason_card(r["feature"], r["direction"], r["shap_val"])

        divider()
        section_label("Application Summary", margin_top="0")
        summary = pd.DataFrame({
            "Field": ["Name","Gender","Employment","Housing","Age",
                      "Saving Accounts","Credit Amount","Duration"],
            "Value": [name or "—", gender, str(job), housing, str(age),
                      saving, f"₹{credit_amount:,}", f"{duration} months"]
        })
        st.table(summary.set_index("Field"))


# ─────────────────────────────────────────────────────────────────────
# PAGE  →  EDA DASHBOARD
# ─────────────────────────────────────────────────────────────────────
elif st.session_state.page == "eda":

    st.markdown("""
    <div style="margin-bottom:24px">
        <div style="font-size:1.3rem;font-weight:600;color:#1C2B1C;margin-bottom:5px">EDA Dashboard</div>
        <div style="font-size:13px;color:#8A9E8A">Exploratory insights from the German Credit dataset.</div>
    </div>""", unsafe_allow_html=True)

    try:
        df_raw = load_data(os.path.join(ROOT, "data", "train.csv"))
        s1, s2, s3, s4 = st.columns(4, gap="small")
        with s1: metric_card("Total Records", str(len(df_raw)))
        with s2: metric_card("Features", "7")
        with s3: metric_card("Good Credit", str(int((df_raw["Risk"]==1).sum())))
        with s4: metric_card("Bad Credit",  str(int((df_raw["Risk"]==0).sum())), "#8B6014")
    except Exception:
        st.warning("Could not load dataset stats.")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    PLOT_DIR = os.path.join(ROOT, "eda_plots")
    plots = [
        ("Risk Distribution",       "01_risk_distribution.png"),
        ("Age Analysis",            "02_age_analysis.png"),
        ("Credit Amount Analysis",  "03_credit_amount_analysis.png"),
        ("Duration Analysis",       "04_duration_analysis.png"),
        ("Categorical vs Risk",     "05_categorical_vs_risk_count.png"),
        ("Risk Rate by Category",   "06_categorical_risk_rate_pct.png"),
        ("Job Level Analysis",      "07_job_level_analysis.png"),
        ("Saving Accounts vs Risk", "08_saving_vs_risk.png"),
        ("Correlation Heatmap",     "09_correlation_heatmap.png"),
        ("Outlier Detection",       "10_outlier_detection.png"),
        ("Pairplot",                "11_pairplot.png"),
        ("Confusion Matrix",        "confusion_matrix.png"),
        ("Feature Importance",      "feature_importance.png"),
    ]
    available = [(k, os.path.join(PLOT_DIR, v))
                 for k, v in plots if os.path.exists(os.path.join(PLOT_DIR, v))]

    if not available:
        st.warning("No plots found. Run `python eda.py` and `python src/models/train_model.py` first.")
    else:
        for i in range(0, len(available), 2):
            c1, c2 = st.columns(2, gap="medium")
            for col, (k, path) in zip([c1, c2], available[i:i+2]):
                with col:
                    st.markdown(
                        f"<div style='font-size:12px;font-weight:600;color:#5A6E5A;"
                        f"margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em'>{k}</div>",
                        unsafe_allow_html=True)
                    st.image(path, width=460)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    divider()
    section_label("Raw Dataset Statistics", margin_top="0")
    try:
        st.dataframe(df_raw.describe().round(2), use_container_width=True)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
# PAGE  →  ABOUT
# ─────────────────────────────────────────────────────────────────────
elif st.session_state.page == "about":

    st.markdown("""
    <div style="margin-bottom:28px">
        <div style="font-size:1.3rem;font-weight:600;color:#1C2B1C;margin-bottom:5px">About</div>
        <div style="font-size:13px;color:#8A9E8A">How this system works and what powers it.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#FAFAF5;border:1px solid #D0CCBC;border-radius:12px;padding:24px;
        border-left:4px solid #6BAE6B">
        <div style="font-size:15px;font-weight:600;color:#1C2B1C;margin-bottom:8px">
            AI Credit Underwriting System</div>
        <div style="font-size:13px;color:#8A9E8A;line-height:1.75">
            An end-to-end ML-powered credit risk assessment system using the German Credit dataset.
            Predicts whether a loan applicant is a good or bad credit risk, and explains every
            decision using SHAP values so lenders can understand and trust the model's output.
        </div>
    </div>""", unsafe_allow_html=True)

    divider()
    section_label("Tech Stack", margin_top="0")
    stack = [
        ("ML Model",        "Random Forest (scikit-learn)"),
        ("Explainability",  "SHAP TreeExplainer"),
        ("REST API",        "Flask + Flask-CORS"),
        ("UI",              "Streamlit"),
        ("Data Processing", "pandas · numpy"),
        ("Visualisation",   "matplotlib · seaborn"),
    ]
    c1, c2 = st.columns(2, gap="large")
    for i, (layer, tool) in enumerate(stack):
        with (c1 if i < 3 else c2):
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                padding:11px 0;border-bottom:1px solid #E0DCD0;font-size:13px">
                <span style="color:#8A9E8A">{layer}</span>
                <span style="color:#1C2B1C;font-weight:500;background:#E8E4D8;
                    padding:3px 10px;border-radius:20px;font-size:12px">{tool}</span>
            </div>""", unsafe_allow_html=True)

    divider()
    section_label("Model Performance", margin_top="0")
    p1, p2, p3, p4 = st.columns(4, gap="small")
    with p1: metric_card("Algorithm", "Random Forest")
    with p2: metric_card("Train AUC", "0.98")
    with p3: metric_card("Test AUC",  "0.688")
    with p4: metric_card("CV Folds",  "5-Fold")

    divider()
    section_label("Project Structure", margin_top="0")
    st.code("""credit-underwriting/
├── app.py                        # Streamlit UI
├── eda.py                        # EDA script
├── requirements.txt
├── data/
│   ├── train.csv
│   └── test.csv
├── eda_plots/                    # 13 saved plots
├── models/
│   └── model.pkl
└── src/
    ├── data/
    │   ├── preprocess.py
    │   └── load_and_clean.py
    ├── models/
    │   ├── train_model.py
    │   ├── explain.py
    │   └── fairness.py
    └── api/
        └── app.py                # Flask API""", language="")

    divider()
    section_label("How to Run", margin_top="0")
    st.code("""pip install -r requirements.txt
python src/data/load_and_clean.py
python eda.py
python src/models/train_model.py
streamlit run app.py""", language="bash")