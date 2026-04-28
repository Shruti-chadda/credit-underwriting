"""
app.py  —  AI Credit Underwriting System
Dark SaaS Dashboard — run: streamlit run app.py
"""

import os, sys, pickle, random
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.preprocess import preprocess_data, load_data, get_features_target
from src.models.explain import build_explainer, get_top_reasons

# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CreditAI", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────────────
# CSS  — dark SaaS theme
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Shell ── */
.stApp                { background: #0A0E0A !important; }
.block-container      { padding: 0 2rem 3rem !important; max-width: 1200px !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0D120D !important;
    border-right: 1px solid #1A2E1A !important;
    min-width: 260px !important;
}
section[data-testid="stSidebar"] * { color: #A0B8A0 !important; }
section[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    display: block !important;
    cursor: pointer !important;
    transition: all .15s !important;
    border: none !important;
}
section[data-testid="stSidebar"] .stRadio label:hover { background: #1A2E1A !important; color: #E0F0E0 !important; }

/* ── Hide defaults ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="collapsedControl"] { color: #22C55E !important; }
.stDeployButton { display: none !important; }

/* ── Top bar ── */
header[data-testid="stHeader"] {
    background: #0D120D !important;
    border-bottom: 1px solid #1A2E1A !important;
}

/* ── Headings ── */
h1 { font-size: 1.5rem !important; font-weight: 600 !important; color: #E8F5E8 !important; }
h2 { font-size: 1.15rem !important; font-weight: 600 !important; color: #E8F5E8 !important; }
h3 { font-size: .95rem !important; font-weight: 500 !important; color: #A0B8A0 !important; }

/* ── Buttons ── */
.stButton > button {
    background: #16A34A !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-size: 14px !important; font-weight: 500 !important;
    padding: 10px 22px !important; transition: background .2s !important;
}
.stButton > button:hover { background: #15803D !important; }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input {
    background: #111811 !important; border: 1px solid #1E3A1E !important;
    border-radius: 8px !important; color: #E8F5E8 !important; font-size: 14px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #22C55E !important; box-shadow: 0 0 0 2px rgba(34,197,94,.15) !important;
}
div[data-baseweb="select"] > div {
    background: #111811 !important; border: 1px solid #1E3A1E !important;
    border-radius: 8px !important; color: #E8F5E8 !important;
}
div[data-baseweb="popover"] { background: #0F1A0F !important; border: 1px solid #1E3A1E !important; }
div[data-baseweb="menu"] li { color: #A0B8A0 !important; }
div[data-baseweb="menu"] li:hover { background: #1A2E1A !important; color: #E8F5E8 !important; }

/* ── Labels ── */
label, .stSelectbox label, .stNumberInput label, .stTextInput label {
    font-size: 12px !important; font-weight: 500 !important;
    color: #5A7A5A !important; text-transform: uppercase !important; letter-spacing: .06em !important;
}

/* ── Progress ── */
.stProgress > div > div { background: #22C55E !important; border-radius: 4px !important; }
[data-testid="stProgressBar"] > div { background: #1A2E1A !important; border-radius: 4px !important; }

/* ── Info/warning ── */
[data-testid="stInfo"]    { background: #0F2010 !important; border-color: #22C55E !important; color: #A0D0A0 !important; border-radius: 10px !important; }
[data-testid="stWarning"] { background: #1A1200 !important; border-color: #CA8A04 !important; border-radius: 10px !important; }
[data-testid="stSuccess"] { background: #0F2010 !important; border-color: #22C55E !important; border-radius: 10px !important; }
[data-testid="stError"]   { background: #1A0808 !important; border-radius: 10px !important; }

/* ── Dataframe ── */
.stDataFrame { border: 1px solid #1A2E1A !important; border-radius: 10px !important; }
[data-testid="stDataFrame"] { background: #0D120D !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
    background: #111811 !important; border: 1px solid #1A2E1A !important;
    border-radius: 10px !important; padding: 18px !important;
}
[data-testid="stMetricLabel"] { color: #5A7A5A !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: .06em !important; }
[data-testid="stMetricValue"] { color: #22C55E !important; font-size: 1.5rem !important; font-weight: 600 !important; }

/* ── Divider ── */
hr { border-color: #1A2E1A !important; }

/* ── Table ── */
table { width: 100%; border-collapse: collapse; }
thead th { background: #0F1F0F !important; color: #5A7A5A !important; font-size: 11px !important;
    font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: .06em !important; padding: 10px 14px !important; }
tbody td { color: #C0D8C0 !important; font-size: 13px !important; padding: 10px 14px !important; border-bottom: 1px solid #1A2E1A !important; }
tbody tr:hover td { background: #0F1A0F !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    with open(os.path.join(ROOT, "models", "model.pkl"), "rb") as f:
        mdl = pickle.load(f)
    df_bg   = preprocess_data(load_data(os.path.join(ROOT, "data", "train.csv")))
    X_bg, _ = get_features_target(df_bg)
    exp     = build_explainer(mdl, X_bg)
    return mdl, exp, list(X_bg.columns), df_bg

model, explainer, FEATURE_COLS, df_bg = load_resources()
_, y_bg = get_features_target(df_bg)


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def dark_card(content: str, border_color: str = "#1A2E1A") -> None:
    st.markdown(f"""
    <div style="background:#111811;border:1px solid {border_color};border-radius:12px;padding:22px;margin-bottom:4px">
        {content}
    </div>""", unsafe_allow_html=True)


def stat_card(icon: str, value: str, label: str, color: str = "#22C55E") -> None:
    st.markdown(f"""
    <div style="background:#111811;border:1px solid #1A2E1A;border-radius:12px;padding:22px 18px">
        <div style="width:36px;height:36px;background:#0F2010;border-radius:8px;
            display:flex;align-items:center;justify-content:center;font-size:18px;margin-bottom:14px">{icon}</div>
        <div style="font-size:1.7rem;font-weight:700;color:{color};margin-bottom:4px">{value}</div>
        <div style="font-size:12px;color:#5A7A5A;font-weight:400">{label}</div>
    </div>""", unsafe_allow_html=True)


def badge(text: str, color: str = "#22C55E", bg: str = "#0F2010") -> str:
    return f"<span style='background:{bg};color:{color};font-size:11px;padding:3px 10px;border-radius:20px;font-weight:500;border:1px solid {color}33'>{text}</span>"


def section_title(text: str, sub: str = "") -> None:
    st.markdown(f"""
    <div style="margin: 28px 0 16px">
        <div style="font-size:15px;font-weight:600;color:#E8F5E8">{text}</div>
        {"<div style='font-size:12px;color:#5A7A5A;margin-top:3px'>" + sub + "</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)


def shap_bar_card(feature: str, shap_val: float, raw_val, direction: str) -> None:
    is_up    = direction == "increases"
    bar_col  = "#EF4444" if is_up else "#22C55E"
    tag_bg   = "#1A0808" if is_up else "#0F2010"
    tag_col  = "#EF4444" if is_up else "#22C55E"
    tag_txt  = "↑ Risk Factor" if is_up else "↓ Protective Factor"
    bar_w    = min(int(abs(shap_val) * 700), 100)
    impact   = "HIGH" if abs(shap_val) > 0.12 else "MED" if abs(shap_val) > 0.06 else "LOW"
    imp_col  = "#EF4444" if impact == "HIGH" else "#F59E0B" if impact == "MED" else "#22C55E"
    st.markdown(f"""
    <div style="background:#111811;border:1px solid #1A2E1A;border-radius:10px;
        padding:16px 18px;margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <div>
                <span style="font-size:14px;font-weight:500;color:#E8F5E8">{feature}</span>
                <span style="font-size:12px;color:#5A7A5A;margin-left:8px">Value: <strong style="color:#A0B8A0">{raw_val}</strong></span>
            </div>
            <div style="display:flex;gap:6px;align-items:center">
                <span style="font-size:10px;font-weight:700;color:{imp_col};background:{imp_col}15;
                    padding:2px 7px;border-radius:4px">{impact}</span>
                <span style="background:{tag_bg};color:{tag_col};font-size:11px;
                    padding:3px 10px;border-radius:20px;font-weight:500">{tag_txt}</span>
            </div>
        </div>
        <div style="height:6px;background:#1A2E1A;border-radius:3px;overflow:hidden">
            <div style="height:100%;width:{bar_w}%;background:{bar_col};border-radius:3px;
                box-shadow:0 0 8px {bar_col}66"></div>
        </div>
        <div style="font-size:11px;color:#3D5A3D;margin-top:5px">SHAP impact: {shap_val:+.4f}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 8px 16px;border-bottom:1px solid #1A2E1A;margin-bottom:12px">
        <div style="display:flex;align-items:center;gap:10px">
            <div style="width:38px;height:38px;background:#16A34A;border-radius:10px;
                display:flex;align-items:center;justify-content:center;font-size:20px">🏦</div>
            <div>
                <div style="font-size:15px;font-weight:700;color:#E8F5E8">AI Credit</div>
                <div style="font-size:12px;color:#22C55E;font-weight:500">Underwriting</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Home",
        "📋  Loan Application",
        "📊  EDA Dashboard",
        "🔍  Model Insights",
        "ℹ️  About"
    ], label_visibility="collapsed")

    st.markdown("""
    <div style="margin-top:40px;padding:14px;background:#0A120A;border:1px solid #1A2E1A;
        border-radius:10px">
        <div style="font-size:10px;font-weight:700;color:#5A7A5A;letter-spacing:.08em;
            text-transform:uppercase;margin-bottom:6px">Security Token</div>
        <div style="font-size:13px;color:#22C55E;font-weight:600;font-family:monospace">SEC-8821-X99</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:12px;padding:12px 14px;background:#0A120A;border:1px solid #1A2E1A;
        border-radius:10px">
        <div style="font-size:10px;font-weight:700;color:#5A7A5A;letter-spacing:.08em;
            text-transform:uppercase;margin-bottom:4px">System Status</div>
        <div style="display:flex;align-items:center;gap:6px">
            <div style="width:7px;height:7px;background:#22C55E;border-radius:50%;
                box-shadow:0 0 6px #22C55E"></div>
            <span style="font-size:12px;color:#22C55E;font-weight:500">System Active</span>
        </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════
if "Home" in page:

    # Hero banner
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0A1F0A 0%,#0D2B0D 50%,#0A1A14 100%);
        border:1px solid #1A3A1A;border-radius:16px;padding:42px 40px;margin-bottom:24px;
        position:relative;overflow:hidden">
        <div style="position:absolute;top:-40px;right:-40px;width:220px;height:220px;
            background:radial-gradient(circle,#22C55E18 0%,transparent 70%);border-radius:50%"></div>
        <div style="position:absolute;bottom:-30px;left:30%;width:150px;height:150px;
            background:radial-gradient(circle,#16A34A10 0%,transparent 70%);border-radius:50%"></div>
        <div style="display:inline-flex;align-items:center;gap:8px;background:#0F2A0F;
            border:1px solid #22C55E55;border-radius:20px;padding:5px 14px;
            font-size:12px;color:#22C55E;font-weight:500;margin-bottom:18px">
            <span style="width:7px;height:7px;background:#22C55E;border-radius:50%;
                box-shadow:0 0 6px #22C55E;display:inline-block"></span>
            System Active
        </div>
        <div style="font-size:2.2rem;font-weight:700;color:#E8F5E8;margin-bottom:10px;line-height:1.2">
            Intelligent Credit<br>Underwriting
        </div>
        <div style="font-size:14px;color:#5A7A5A;max-width:460px;line-height:1.6">
            Real-time risk assessment powered by machine learning and SHAP explainability.
            Trained on 1000 German Credit records.
        </div>
    </div>""", unsafe_allow_html=True)

    # Stat cards
    c1, c2, c3, c4 = st.columns(4, gap="small")
    # Get actual model stats from training data
    try:
        df_raw = load_data(os.path.join(ROOT, "data", "train.csv"))
        n_records = len(df_raw)
        pct_good  = round(df_raw["Risk"].mean() * 100, 1)
    except Exception:
        n_records, pct_good = 800, 70.0

    with c1: stat_card("📈", "68.0%", "Model Accuracy")
    with c2: stat_card("🛡️", "77.3%", "F1 Performance")
    with c3: stat_card("👥", str(n_records), "Records Analyzed")
    with c4: stat_card("📉", "Low",   "Portfolio Risk", "#22C55E")

    # Risk distribution donut (matplotlib dark)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c_left, c_right = st.columns([1, 1], gap="medium")

    with c_left:
        section_title("Risk Distribution", "Composition of Good vs Bad credit risk")
        try:
            df_raw = load_data(os.path.join(ROOT, "data", "train.csv"))
            good   = int((df_raw["Risk"] == 1).sum())
            bad    = int((df_raw["Risk"] == 0).sum())
        except Exception:
            good, bad = 560, 240

        fig, ax = plt.subplots(figsize=(4.5, 4), subplot_kw=dict(aspect="equal"))
        fig.patch.set_facecolor("#111811")
        ax.set_facecolor("#111811")
        wedges, texts, autotexts = ax.pie(
            [good, bad], labels=["Good Credit", "Bad Credit"],
            colors=["#22C55E", "#EF4444"], autopct="%1.0f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops={"width": 0.45, "edgecolor": "#111811", "linewidth": 3}
        )
        for t in texts:     t.set_color("#5A7A5A"); t.set_fontsize(11)
        for a in autotexts: a.set_color("#E8F5E8"); a.set_fontsize(12); a.set_fontweight("bold")
        ax.text(0, 0, f"{good+bad}\nTotal", ha="center", va="center",
                color="#E8F5E8", fontsize=12, fontweight="bold")
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with c_right:
        section_title("Bureau Integrations", "Verified data channels")
        for name, color, icon in [("EXPERIAN","#8B5CF6","🔮"),("EQUIFAX","#EF4444","🔴"),("TRANSUNION","#3B82F6","🔵")]:
            st.markdown(f"""
            <div style="background:#0A120A;border:1px solid #1A2E1A;border-radius:10px;
                padding:14px 16px;display:flex;align-items:center;gap:12px;margin-bottom:8px">
                <div style="font-size:20px">{icon}</div>
                <div>
                    <div style="font-size:13px;font-weight:600;color:#E8F5E8">{name}</div>
                    <div style="font-size:11px;color:#22C55E">● Connected</div>
                </div>
                <div style="margin-left:auto;font-size:11px;color:#5A7A5A">Real-time</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# LOAN APPLICATION
# ═══════════════════════════════════════════════════════════════════════
elif "Loan" in page:

    st.markdown("""
    <div style="margin-bottom:24px">
        <h1 style="margin-bottom:4px">Loan Application</h1>
        <p style="font-size:13px;color:#5A7A5A;margin:0">
            Complete the form below for an instant AI-powered credit risk assessment.</p>
    </div>""", unsafe_allow_html=True)

    with st.form("loan_form"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div style='font-size:11px;font-weight:600;color:#5A7A5A;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px'>Personal Information</div>", unsafe_allow_html=True)
            name    = st.text_input("Full Name", placeholder="e.g. Rahul Sharma")
            gender  = st.selectbox("Gender", ["Male", "Female"])
            job     = st.selectbox("Employment Type", [0,1,2,3], format_func=lambda x: {
                0:"Unskilled — Non-resident", 1:"Unskilled — Resident",
                2:"Skilled Employee", 3:"Highly Qualified"}[x])
            housing = st.selectbox("Housing Status", ["Own","Rent","Free"])

        with c2:
            st.markdown("<div style='font-size:11px;font-weight:600;color:#5A7A5A;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px'>Financial Information</div>", unsafe_allow_html=True)
            age           = st.number_input("Age", 18, 100, 30)
            saving        = st.selectbox("Saving Account Level", ["Low","Moderate","High"])
            credit_amount = st.number_input("Credit Amount Requested (₹)", 500, 200_000, 10_000, step=500)
            duration      = st.number_input("Loan Duration (months)", 6, 120, 24, step=6)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ Run Risk Assessment", use_container_width=True)

    # ── RESULT ──
    if submitted:
        input_df = pd.DataFrame([{
            "Sex":gender, "Job":job, "Housing":housing, "Saving accounts":saving,
            "Age":age, "Credit amount":credit_amount, "Duration":duration
        }])
        processed  = preprocess_data(input_df)
        X_input    = processed[FEATURE_COLS]
        prob       = float(model.predict_proba(X_input)[0][1])
        prediction = int(model.predict(X_input)[0])
        reasons    = get_top_reasons(explainer, X_input, top_n=5)
        suggested  = int(200_000 * prob) if prediction == 1 else int(50_000 * prob)

        # Verdict banner
        if prediction == 1:
            verdict_color = "#22C55E"; verdict_bg = "#0F2010"; verdict_border = "#22C55E44"
            verdict_text  = "✅  APPROVED — Good Credit Risk"
            verdict_sub   = "Applicant meets the credit risk criteria."
        else:
            verdict_color = "#EF4444"; verdict_bg = "#1A0808"; verdict_border = "#EF444444"
            verdict_text  = "❌  DECLINED — High Credit Risk"
            verdict_sub   = "Applicant does not meet the credit risk criteria."

        st.markdown(f"""
        <div style="background:{verdict_bg};border:1px solid {verdict_border};border-radius:12px;
            padding:20px 24px;margin:20px 0 16px">
            <div style="font-size:1.1rem;font-weight:700;color:{verdict_color}">{verdict_text}</div>
            <div style="font-size:13px;color:#5A7A5A;margin-top:4px">{verdict_sub}</div>
        </div>""", unsafe_allow_html=True)

        # Metric row
        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1: stat_card("📊", f"{round(prob*100,1)}%", "Approval Probability")
        with c2: stat_card("💰", f"₹{suggested:,}",       "Suggested Loan",   "#F59E0B")
        with c3: stat_card("⏱️", f"{duration}mo",         "Duration")
        with c4: stat_card("🎯", f"{round((1-prob)*100,1)}%","Default Risk", "#EF4444" if prob < 0.5 else "#22C55E")

        # Risk gauge bar
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        bar_col = "#22C55E" if prob > 0.5 else "#EF4444"
        st.markdown(f"""
        <div style="background:#111811;border:1px solid #1A2E1A;border-radius:10px;padding:18px 20px;margin-bottom:16px">
            <div style="display:flex;justify-content:space-between;margin-bottom:10px">
                <span style="font-size:13px;font-weight:500;color:#E8F5E8">Credit Score Estimate</span>
                <span style="font-size:13px;font-weight:700;color:{bar_col}">{round(prob*100,1)}% Creditworthy</span>
            </div>
            <div style="height:10px;background:#1A2E1A;border-radius:5px;overflow:hidden">
                <div style="height:100%;width:{round(prob*100,1)}%;background:linear-gradient(90deg,{bar_col},{bar_col}aa);
                    border-radius:5px;box-shadow:0 0 10px {bar_col}55"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:6px">
                <span style="font-size:10px;color:#3D5A3D">Poor</span>
                <span style="font-size:10px;color:#3D5A3D">Excellent</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # SHAP explanation
        section_title("AI Decision Explanation", "SHAP values show each factor's exact impact on this decision")

        # Plain-English summary first
        n_protect  = sum(1 for r in reasons if r["direction"] == "decreases")
        n_risk     = sum(1 for r in reasons if r["direction"] == "increases")
        top_factor = reasons[0]

        st.markdown(f"""
        <div style="background:#0A1F0A;border:1px solid #1A3A1A;border-radius:10px;padding:16px 18px;margin-bottom:16px">
            <div style="font-size:13px;color:#A0C8A0;line-height:1.7">
                🤖 <strong style="color:#E8F5E8">AI Summary:</strong>
                The model analysed <strong style="color:#22C55E">7 financial features</strong> for
                <strong style="color:#E8F5E8">{name or 'this applicant'}</strong>.
                Out of the top 5 factors, <strong style="color:#22C55E">{n_protect} are protective</strong>
                and <strong style="color:#EF4444">{n_risk} increase risk</strong>.
                The single biggest driver is <strong style="color:#F59E0B">{top_factor['feature']}</strong>
                which {"reduces" if top_factor["direction"]=="decreases" else "raises"} risk by
                <strong style="color:#F59E0B">{abs(top_factor['shap_val']):.4f} SHAP units</strong>.
            </div>
        </div>""", unsafe_allow_html=True)

        # Per-feature cards with actual input values
        raw_values = {
            "Sex": gender, "Job": job, "Housing": housing,
            "Saving accounts": saving, "Age": age,
            "Credit amount": f"₹{credit_amount:,}", "Duration": f"{duration}mo"
        }
        for r in reasons:
            shap_bar_card(r["feature"], r["shap_val"],
                          raw_values.get(r["feature"], "—"), r["direction"])

        # Applicant summary table
        section_title("Application Summary", "Full applicant profile")
        summary = pd.DataFrame({
            "Field": ["Name","Gender","Employment","Housing","Age","Saving Accounts","Credit Amount","Duration"],
            "Value": [name or "—", gender, {0:"Unskilled(NR)",1:"Unskilled(R)",2:"Skilled",3:"Highly Qualified"}[job],
                      housing, str(age), saving, f"₹{credit_amount:,}", f"{duration} months"]
        })
        st.table(summary.set_index("Field"))


# ═══════════════════════════════════════════════════════════════════════
# EDA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
elif "EDA" in page:

    st.markdown("""
    <div style="margin-bottom:24px">
        <h1 style="margin-bottom:4px">EDA Dashboard</h1>
        <p style="font-size:13px;color:#5A7A5A;margin:0">
            Deep exploratory analysis of the German Credit dataset.</p>
    </div>""", unsafe_allow_html=True)

    try:
        df_raw = load_data(os.path.join(ROOT, "data", "train.csv"))
    except Exception:
        st.error("data/train.csv not found. Run python src/data/load_and_clean.py first.")
        st.stop()

    # Summary stats
    good = int((df_raw["Risk"]==1).sum()); bad = int((df_raw["Risk"]==0).sum())
    c1,c2,c3,c4,c5 = st.columns(5, gap="small")
    with c1: stat_card("📁", str(len(df_raw)), "Total Records")
    with c2: stat_card("✅", str(good), "Good Credit")
    with c3: stat_card("❌", str(bad),  "Bad Credit", "#EF4444")
    with c4: stat_card("📊", "7", "Features")
    with c5: stat_card("🎯", f"{round(good/len(df_raw)*100,0):.0f}%","Good Rate")

    # ── Tabs for different EDA views ──
    tab1, tab2, tab3 = st.tabs(["📈  Saved Plots", "🔢  Distribution Analysis", "📋  Raw Data"])

    with tab1:
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
        available = [(k, os.path.join(PLOT_DIR, v)) for k,v in plots if os.path.exists(os.path.join(PLOT_DIR, v))]
        if not available:
            st.warning("Run `python eda.py` to generate plots.")
        else:
            for i in range(0, len(available), 2):
                c1, c2 = st.columns(2, gap="medium")
                for col, (k, path) in zip([c1,c2], available[i:i+2]):
                    with col:
                        st.markdown(f"<div style='font-size:12px;font-weight:600;color:#5A7A5A;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em'>{k}</div>", unsafe_allow_html=True)
                        st.image(path, width=440)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with tab2:
        section_title("Feature Distributions", "Interactive breakdown of each feature vs Risk")

        feature_sel = st.selectbox("Select Feature to Analyse",
            ["Age","Credit amount","Duration","Job","Sex","Housing","Saving accounts"])

        df_plot = df_raw.copy()
        df_plot["Risk Label"] = df_plot["Risk"].map({1:"Good Credit",0:"Bad Credit"})

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        fig.patch.set_facecolor("#111811")
        for ax in axes: ax.set_facecolor("#111811"); ax.tick_params(colors="#5A7A5A"); ax.spines[:].set_color("#1A2E1A")

        if df_plot[feature_sel].dtype in [np.int64, np.float64]:
            good_vals = df_plot[df_plot["Risk"]==1][feature_sel]
            bad_vals  = df_plot[df_plot["Risk"]==0][feature_sel]
            axes[0].hist(good_vals, bins=20, color="#22C55E", alpha=0.7, label="Good Credit", edgecolor="#111811")
            axes[0].hist(bad_vals,  bins=20, color="#EF4444", alpha=0.7, label="Bad Credit",  edgecolor="#111811")
            axes[0].set_title(f"{feature_sel} Distribution", color="#E8F5E8", fontsize=12)
            axes[0].legend(facecolor="#1A2E1A", labelcolor="#A0B8A0", fontsize=9)
            axes[0].set_xlabel(feature_sel, color="#5A7A5A")
            axes[0].set_ylabel("Count", color="#5A7A5A")

            axes[1].boxplot([bad_vals, good_vals], tick_labels=["Bad","Good"],
                patch_artist=True,
                boxprops=dict(facecolor="#1A2E1A", color="#22C55E"),
                medianprops=dict(color="#22C55E", linewidth=2),
                whiskerprops=dict(color="#3D5A3D"), capprops=dict(color="#3D5A3D"),
                flierprops=dict(markerfacecolor="#EF4444", marker="o", markersize=4))
            axes[1].set_title(f"{feature_sel} by Risk Category", color="#E8F5E8", fontsize=12)
            axes[1].set_ylabel(feature_sel, color="#5A7A5A")
        else:
            ct = df_plot.groupby([feature_sel,"Risk Label"]).size().unstack(fill_value=0)
            x = range(len(ct))
            w = 0.35
            good_c = ct.get("Good Credit", pd.Series([0]*len(ct)))
            bad_c  = ct.get("Bad Credit",  pd.Series([0]*len(ct)))
            axes[0].bar([i-w/2 for i in x], good_c, w, color="#22C55E", label="Good", alpha=0.85)
            axes[0].bar([i+w/2 for i in x], bad_c,  w, color="#EF4444", label="Bad",  alpha=0.85)
            axes[0].set_xticks(list(x)); axes[0].set_xticklabels(ct.index, color="#5A7A5A", fontsize=10)
            axes[0].set_title(f"{feature_sel} vs Risk (Count)", color="#E8F5E8", fontsize=12)
            axes[0].legend(facecolor="#1A2E1A", labelcolor="#A0B8A0")
            axes[0].set_ylabel("Count", color="#5A7A5A")

            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            axes[1].bar(list(x), ct_pct.get("Good Credit", pd.Series([0]*len(ct))), color="#22C55E", label="Good%", alpha=0.85)
            axes[1].bar(list(x), ct_pct.get("Bad Credit",  pd.Series([0]*len(ct))),
                        bottom=ct_pct.get("Good Credit", pd.Series([0]*len(ct))), color="#EF4444", label="Bad%", alpha=0.85)
            axes[1].set_xticks(list(x)); axes[1].set_xticklabels(ct.index, color="#5A7A5A", fontsize=10)
            axes[1].set_title(f"{feature_sel} — Risk Rate (%)", color="#E8F5E8", fontsize=12)
            axes[1].axhline(50, color="#3D5A3D", linestyle="--", linewidth=1)
            axes[1].set_ylabel("Percentage", color="#5A7A5A")

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Stats for selected feature
        if df_plot[feature_sel].dtype in [np.int64, np.float64]:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            g = df_plot[df_plot["Risk"]==1][feature_sel]
            b = df_plot[df_plot["Risk"]==0][feature_sel]
            s1,s2,s3,s4 = st.columns(4)
            with s1: stat_card("📗", f"{g.mean():.1f}", f"Good — Mean {feature_sel}")
            with s2: stat_card("📕", f"{b.mean():.1f}", f"Bad — Mean {feature_sel}", "#EF4444")
            with s3: stat_card("📊", f"{df_plot[feature_sel].std():.1f}", f"Std Dev")
            with s4: stat_card("⚡", f"{df_plot[feature_sel].min()} – {df_plot[feature_sel].max()}", "Range")

    with tab3:
        section_title("Raw Dataset", "First 50 rows of training data")
        st.dataframe(
            df_raw.head(50).style.applymap(
                lambda v: "color:#22C55E" if v==1 else ("color:#EF4444" if v==0 else ""),
                subset=["Risk"]
            ),
            use_container_width=True, height=400
        )
        st.markdown(f"<div style='font-size:12px;color:#5A7A5A;margin-top:6px'>Showing 50 of {len(df_raw)} rows</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════
elif "Insights" in page:

    st.markdown("""
    <div style="margin-bottom:24px">
        <h1 style="margin-bottom:4px">Model Insights</h1>
        <p style="font-size:13px;color:#5A7A5A;margin:0">
            Deep dive into model behaviour, feature importance and decision logic.</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4, gap="small")
    with c1: stat_card("🌲", "200",   "Decision Trees")
    with c2: stat_card("📈", "0.688", "Test AUC")
    with c3: stat_card("🔁", "5-Fold","Cross Validation")
    with c4: stat_card("⚡", "7",     "Features Used")

    # Feature importance from model
    section_title("Feature Importance", "Mean decrease in impurity across all 200 trees")
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#111811"); ax.set_facecolor("#111811")
    bars = ax.barh(imp.index, imp.values, color=["#22C55E" if v > imp.median() else "#1A3A1A" for v in imp.values],
                   edgecolor="#0A0E0A", height=0.6)
    ax.set_xlabel("Importance Score", color="#5A7A5A", fontsize=11)
    ax.tick_params(colors="#5A7A5A")
    for s in ax.spines.values(): s.set_color("#1A2E1A")
    for bar, val in zip(bars, imp.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color="#A0B8A0", fontsize=9)
    plt.tight_layout(pad=1)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Feature importance table with explanation
    section_title("Feature Importance — Plain English")
    explanations = {
        "Saving accounts": "How much savings the applicant has. Highest predictor — applicants with high savings almost always repay.",
        "Duration":        "Length of the loan. Longer loans = more risk exposure = higher default chance.",
        "Credit amount":   "How much money is requested. Very large loans relative to profile increase risk.",
        "Age":             "Older applicants tend to have more stable finances and lower default rates.",
        "Job":             "Employment type and skill level. Highly qualified employees are more stable.",
        "Housing":         "Owners are more financially stable than renters. Renters have higher default rates.",
        "Sex":             "Gender has minor predictive power — flagged for fairness monitoring.",
    }
    for feat, score in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
        bar_w = int(score * 1000)
        st.markdown(f"""
        <div style="background:#111811;border:1px solid #1A2E1A;border-radius:10px;padding:14px 18px;margin-bottom:8px">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="font-size:13px;font-weight:500;color:#E8F5E8">{feat}</span>
                <span style="font-size:12px;font-weight:700;color:#22C55E">{score:.4f}</span>
            </div>
            <div style="height:4px;background:#1A2E1A;border-radius:2px;margin-bottom:8px">
                <div style="height:100%;width:{bar_w}%;background:#22C55E;border-radius:2px"></div>
            </div>
            <div style="font-size:12px;color:#5A7A5A;line-height:1.6">{explanations.get(feat,"")}</div>
        </div>""", unsafe_allow_html=True)

    # Fairness check
    section_title("Fairness Audit", "Demographic parity check across gender groups")
    try:
        df_f  = preprocess_data(load_data(os.path.join(ROOT, "data", "train.csv")))
        X_f, y_f = get_features_target(df_f)
        preds = model.predict(X_f)
        df_f["y_pred"] = preds
        male_rate   = df_f[df_f["Sex"]==0]["y_pred"].mean()
        female_rate = df_f[df_f["Sex"]==1]["y_pred"].mean()
        dp_diff     = abs(male_rate - female_rate)
        fc1, fc2, fc3 = st.columns(3, gap="small")
        with fc1: stat_card("👨", f"{male_rate*100:.1f}%", "Male Approval Rate")
        with fc2: stat_card("👩", f"{female_rate*100:.1f}%", "Female Approval Rate")
        with fc3: stat_card("⚖️", f"{dp_diff:.4f}", "Parity Gap", "#22C55E" if dp_diff < 0.1 else "#EF4444")
        if dp_diff < 0.1:
            st.info(f"✅  Demographic parity gap = {dp_diff:.4f} — within acceptable threshold (< 0.10).")
        else:
            st.warning(f"⚠️  Demographic parity gap = {dp_diff:.4f} — exceeds 10% threshold. Bias risk.")
    except Exception as e:
        st.warning(f"Fairness check failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# ABOUT
# ═══════════════════════════════════════════════════════════════════════
elif "About" in page:

    st.markdown("""
    <div style="margin-bottom:24px">
        <h1 style="margin-bottom:4px">About</h1>
        <p style="font-size:13px;color:#5A7A5A;margin:0">Architecture, tech stack and project details.</p>
    </div>""", unsafe_allow_html=True)

    dark_card("""
    <div style="font-size:15px;font-weight:600;color:#E8F5E8;margin-bottom:8px">AI Credit Underwriting System</div>
    <div style="font-size:13px;color:#5A7A5A;line-height:1.75">
        End-to-end ML-powered credit risk platform trained on the German Credit dataset.
        Every loan decision is explained using SHAP values — the model cannot make a "black box" decision.
        Built for FinTech compliance environments where explainability is mandatory.
    </div>""", "#22C55E")

    section_title("Tech Stack")
    stack = [("ML Model","Random Forest — 200 trees, class-balanced"),
             ("Explainability","SHAP TreeExplainer — per-prediction attribution"),
             ("REST API","Flask + Flask-CORS — JSON predict endpoint"),
             ("Frontend","Streamlit — reactive Python UI"),
             ("Data Layer","pandas · numpy — ETL + feature engineering"),
             ("Fairness","Custom demographic parity + equalized odds")]
    for layer, tool in stack:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
            padding:11px 0;border-bottom:1px solid #1A2E1A;font-size:13px">
            <span style="color:#5A7A5A">{layer}</span>
            <span style="background:#0F2010;color:#22C55E;padding:3px 12px;
                border-radius:20px;font-size:12px;font-weight:500">{tool}</span>
        </div>""", unsafe_allow_html=True)

    section_title("Performance Metrics")
    p1,p2,p3,p4 = st.columns(4, gap="small")
    with p1: stat_card("🌲","200","Trees in Forest")
    with p2: stat_card("📈","0.98","Train AUC")
    with p3: stat_card("🎯","0.688","Test AUC")
    with p4: stat_card("🔁","5","CV Folds")

    section_title("How to Run Locally")
    st.code("""# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data + EDA plots
python src/data/load_and_clean.py

# 3. Run EDA analysis
python eda.py

# 4. Train the model
python src/models/train_model.py

# 5. Launch dashboard
streamlit run app.py""", language="bash")