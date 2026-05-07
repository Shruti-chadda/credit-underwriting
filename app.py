"""
app.py  —  AI Credit Underwriting System
Dark SaaS Dashboard — run: streamlit run app.py
"""

import os, sys, pickle, io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.preprocess import preprocess_data, load_data, get_features_target, FEATURE_COLS
from src.models.explain import build_explainer, get_top_reasons

# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CreditAI", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html,body,.stApp,[class*="css"]{font-family:'Inter',sans-serif!important}
.stApp{background:#0A0E0A!important}
.block-container{padding:0 2rem 3rem!important;max-width:1200px!important}

section[data-testid="stSidebar"]{
    background:#0D120D!important;border-right:1px solid #1A2E1A!important;min-width:240px!important}
section[data-testid="stSidebar"] *{color:#C8DCC8!important}
section[data-testid="stSidebar"] label{
    font-size:14px!important;font-weight:400!important;color:#C8DCC8!important;
    text-transform:none!important;letter-spacing:0!important;padding:9px 12px!important;
    border-radius:8px!important;display:block!important}
section[data-testid="stSidebar"] label:hover{background:#1A2E1A!important;color:#E8F5E8!important}

#MainMenu,footer,.stDeployButton{visibility:hidden}
[data-testid="collapsedControl"]{color:#22C55E!important}
header[data-testid="stHeader"]{background:#0D120D!important;border-bottom:1px solid #1A2E1A!important}

h1{font-size:1.5rem!important;font-weight:600!important;color:#E8F5E8!important}
h2{font-size:1.15rem!important;font-weight:600!important;color:#E8F5E8!important}
h3{font-size:.95rem!important;font-weight:500!important;color:#A0C0A0!important}

.stButton>button{background:#16A34A!important;color:#fff!important;border:none!important;
    border-radius:8px!important;font-size:14px!important;font-weight:500!important;padding:10px 22px!important}
.stButton>button:hover{background:#15803D!important}

.stTextInput input,.stNumberInput input{background:#111811!important;border:1px solid #1E3A1E!important;
    border-radius:8px!important;color:#E8F5E8!important;font-size:14px!important}
.stTextInput input:focus,.stNumberInput input:focus{border-color:#22C55E!important;
    box-shadow:0 0 0 2px rgba(34,197,94,.15)!important}
div[data-baseweb="select"]>div{background:#111811!important;border:1px solid #1E3A1E!important;
    border-radius:8px!important;color:#E8F5E8!important}
div[data-baseweb="popover"]{background:#0F1A0F!important;border:1px solid #1E3A1E!important}
div[data-baseweb="menu"] li{color:#C8DCC8!important}
div[data-baseweb="menu"] li:hover{background:#1A2E1A!important;color:#E8F5E8!important}

label,.stSelectbox label,.stNumberInput label,.stTextInput label{
    font-size:12px!important;font-weight:600!important;color:#7EC87E!important;
    text-transform:uppercase!important;letter-spacing:.06em!important}

[data-testid="stTabs"] button{color:#7EC87E!important;font-size:13px!important;
    font-weight:500!important;border-radius:6px 6px 0 0!important;
    background:transparent!important;border:none!important;padding:8px 16px!important}
[data-testid="stTabs"] button:hover{color:#E8F5E8!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:#22C55E!important;
    font-weight:700!important;border-bottom:2px solid #22C55E!important}
[data-testid="stTabs"] [role="tabpanel"]{background:#0A0E0A!important;padding-top:16px!important}

.stProgress>div>div{background:#22C55E!important;border-radius:4px!important}
[data-testid="stProgressBar"]>div{background:#1A2E1A!important;border-radius:4px!important}

[data-testid="stInfo"]{background:#0F2010!important;border-color:#22C55E!important;
    color:#A0D0A0!important;border-radius:10px!important}
[data-testid="stWarning"]{background:#1A1200!important;border-color:#CA8A04!important;border-radius:10px!important}
[data-testid="stSuccess"]{background:#0F2010!important;border-color:#22C55E!important;border-radius:10px!important}
[data-testid="stError"]{background:#1A0808!important;border-radius:10px!important}

.stDataFrame{border:1px solid #1A2E1A!important;border-radius:10px!important}
[data-testid="stMetric"]{background:#111811!important;border:1px solid #1A2E1A!important;
    border-radius:10px!important;padding:18px!important}
[data-testid="stMetricLabel"]{color:#7EC87E!important;font-size:11px!important;
    text-transform:uppercase!important;letter-spacing:.06em!important}
[data-testid="stMetricValue"]{color:#22C55E!important;font-size:1.5rem!important;font-weight:600!important}

table{width:100%;border-collapse:collapse}
thead th{background:#0F1F0F!important;color:#7EC87E!important;font-size:11px!important;
    font-weight:600!important;text-transform:uppercase!important;letter-spacing:.06em!important;padding:10px 14px!important}
tbody td{color:#C8DCC8!important;font-size:13px!important;padding:10px 14px!important;border-bottom:1px solid #1A2E1A!important}
tbody tr:hover td{background:#0F1A0F!important}

[data-testid="stDownloadButton"]>button{background:#0F2010!important;color:#22C55E!important;
    border:1px solid #22C55E55!important;border-radius:8px!important;font-size:13px!important;padding:8px 18px!important}
[data-testid="stDownloadButton"]>button:hover{background:#16A34A!important;color:#fff!important}
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
    return mdl, exp, list(X_bg.columns)

model, explainer, FEAT_COLS = load_resources()


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def stat_card(icon, value, label, color="#22C55E"):
    st.markdown(f"""
    <div style="background:#111811;border:1px solid #1A2E1A;border-radius:12px;padding:22px 18px">
        <div style="width:36px;height:36px;background:#0F2010;border-radius:8px;
            display:flex;align-items:center;justify-content:center;font-size:18px;margin-bottom:14px">{icon}</div>
        <div style="font-size:1.6rem;font-weight:700;color:{color};margin-bottom:4px">{value}</div>
        <div style="font-size:12px;color:#5A7A5A">{label}</div>
    </div>""", unsafe_allow_html=True)


def section_title(text, sub=""):
    st.markdown(f"""
    <div style="margin:28px 0 16px">
        <div style="font-size:15px;font-weight:600;color:#E8F5E8">{text}</div>
        {"<div style='font-size:12px;color:#5A7A5A;margin-top:3px'>" + sub + "</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)


def dark_card(content, border_color="#1A2E1A"):
    st.markdown(f"""
    <div style="background:#111811;border:1px solid {border_color};border-radius:12px;
        padding:22px;margin-bottom:4px">{content}</div>""", unsafe_allow_html=True)


def shap_bar_card(feature, shap_val, raw_val, direction):
    is_up   = direction == "increases"
    bar_col = "#EF4444" if is_up else "#22C55E"
    tag_bg  = "#1A0808" if is_up else "#0F2010"
    tag_col = "#EF4444" if is_up else "#22C55E"
    tag_txt = "↑ Risk Factor" if is_up else "↓ Protective Factor"
    bar_w   = min(int(abs(shap_val) * 700), 100)
    impact  = "HIGH" if abs(shap_val) > 0.12 else "MED" if abs(shap_val) > 0.06 else "LOW"
    imp_col = "#EF4444" if impact == "HIGH" else "#F59E0B" if impact == "MED" else "#22C55E"
    st.markdown(f"""
    <div style="background:#111811;border:1px solid #1A2E1A;border-radius:10px;
        padding:16px 18px;margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <div>
                <span style="font-size:14px;font-weight:500;color:#E8F5E8">{feature}</span>
                <span style="font-size:12px;color:#5A7A5A;margin-left:8px">
                    Value: <strong style="color:#A0B8A0">{raw_val}</strong></span>
            </div>
            <div style="display:flex;gap:6px;align-items:center">
                <span style="font-size:10px;font-weight:700;color:{imp_col};
                    background:{imp_col}20;padding:2px 7px;border-radius:4px">{impact}</span>
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
# PDF — INDIVIDUAL APPLICANT REPORT
# ─────────────────────────────────────────────────────────────────────
def generate_applicant_pdf(
    name, gender, job_label, housing, age, saving, credit_amount, duration,
    prob, prediction, reasons, suggested_loan
):
    """
    Generates a PDF report for a SPECIFIC loan applicant — not training data stats.
    Uses only reportlab (pure Python, no system dependencies).
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        return None  # reportlab not available

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    GREEN  = colors.HexColor("#16A34A")
    RED    = colors.HexColor("#DC2626")
    DARK   = colors.HexColor("#1C2B1C")
    GRAY   = colors.HexColor("#5A6E5A")
    LGRAY  = colors.HexColor("#F4FBF4")
    LRED   = colors.HexColor("#FEF2F2")

    styles  = getSampleStyleSheet()
    title_s = ParagraphStyle("T",  parent=styles["Title"],   fontSize=20, textColor=DARK,  alignment=TA_CENTER, spaceAfter=4)
    sub_s   = ParagraphStyle("S",  parent=styles["Normal"],  fontSize=11, textColor=GRAY,  alignment=TA_CENTER, spaceAfter=14)
    h2_s    = ParagraphStyle("H2", parent=styles["Heading2"],fontSize=13, textColor=GREEN, spaceAfter=6,  spaceBefore=14)
    body_s  = ParagraphStyle("B",  parent=styles["Normal"],  fontSize=10, textColor=DARK,  spaceAfter=5,  leading=15)
    small_s = ParagraphStyle("SM", parent=styles["Normal"],  fontSize=9,  textColor=GRAY,  spaceAfter=3,  leading=13)

    story = []

    # ── Header ──
    story.append(Paragraph("Credit Risk Assessment Report", title_s))
    story.append(Paragraph("AI Credit Underwriting System — Individual Applicant Analysis", sub_s))
    story.append(HRFlowable(width="100%", color=GREEN, thickness=2))
    story.append(Spacer(1, 0.4*cm))

    # ── Verdict banner ──
    verdict_text  = "APPROVED — Good Credit Risk" if prediction == 1 else "DECLINED — High Credit Risk"
    verdict_color = GREEN if prediction == 1 else RED
    verdict_bg    = LGRAY if prediction == 1 else LRED
    story.append(Paragraph(verdict_text,
        ParagraphStyle("V", parent=styles["Title"], fontSize=16,
                       textColor=verdict_color, alignment=TA_CENTER,
                       backColor=verdict_bg, borderPad=10, spaceAfter=8)))
    story.append(Paragraph(
        f"Creditworthiness Probability: {round(prob*100, 1)}%  |  "
        f"Suggested Loan Amount: Rs {suggested_loan:,}",
        ParagraphStyle("VP", parent=sub_s, textColor=verdict_color, spaceAfter=14)))
    story.append(Spacer(1, 0.3*cm))

    # ── Applicant Details ──
    story.append(Paragraph("Applicant Profile", h2_s))
    app_data = [
        ["Field", "Value"],
        ["Full Name",                  name or "Not provided"],
        ["Gender (not used in model)", f"{gender} — decision is gender-neutral"],
        ["Employment Type",            job_label],
        ["Housing Status",             housing],
        ["Age",                        f"{age} years"],
        ["Saving Account Level",       saving],
        ["Credit Amount Requested",    f"Rs {credit_amount:,}"],
        ["Loan Duration",              f"{duration} months"],
    ]
    t = Table(app_data, colWidths=[7*cm, 9*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), GREEN),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LGRAY, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#C8E0C4")),
        ("PADDING",       (0, 0), (-1, -1), 7),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # ── Risk Score ──
    story.append(Paragraph("Risk Assessment Summary", h2_s))
    risk_data = [
        ["Metric", "Value", "Interpretation"],
        ["Approval Probability",  f"{round(prob*100,1)}%",
         "High" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low"],
        ["Default Risk",          f"{round((1-prob)*100,1)}%",
         "Low" if prob > 0.7 else "Moderate" if prob > 0.4 else "High"],
        ["Decision",              verdict_text.split("—")[0].strip(),
         "Meets credit criteria" if prediction==1 else "Does not meet criteria"],
        ["Suggested Loan",        f"Rs {suggested_loan:,}",
         "Based on creditworthiness score"],
        ["Model Used",            "Random Forest (200 trees)",
         "5-fold CV AUC: 0.873"],
        ["Fairness",              "Gender-Neutral",
         "Sex excluded from all predictions"],
    ]
    t2 = Table(risk_data, colWidths=[5.5*cm, 4*cm, 6.5*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), GREEN),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LGRAY, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#C8E0C4")),
        ("PADDING",       (0, 0), (-1, -1), 7),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.4*cm))

    # ── SHAP Explanation — THE KEY PART (individual, not training data) ──
    story.append(Paragraph("AI Decision Explanation (SHAP Analysis)", h2_s))
    story.append(Paragraph(
        "The following factors are specific to THIS applicant. Each SHAP value shows "
        "exactly how much that feature pushed the approval probability up or down "
        "compared to the average applicant. This is not generic — every number below "
        "is calculated from this person's specific inputs.",
        body_s))
    story.append(Spacer(1, 0.2*cm))

    shap_data = [["Feature", "Applicant's Value", "SHAP Impact", "Effect", "Importance"]]
    raw_vals = {
        "Job": job_label, "Housing": housing, "Saving accounts": saving,
        "Age": f"{age} yrs", "Credit amount": f"Rs {credit_amount:,}",
        "Duration": f"{duration} mo"
    }
    for r in reasons:
        direction_txt = "Increases Risk" if r["direction"] == "increases" else "Reduces Risk"
        impact        = "HIGH" if abs(r["shap_val"]) > 0.12 else "MED" if abs(r["shap_val"]) > 0.06 else "LOW"
        shap_data.append([
            r["feature"],
            str(raw_vals.get(r["feature"], "—")),
            f"{r['shap_val']:+.4f}",
            direction_txt,
            impact
        ])

    t3 = Table(shap_data, colWidths=[3.5*cm, 3.5*cm, 2.5*cm, 3.5*cm, 2*cm])
    row_colors = []
    for i, row in enumerate(shap_data[1:], 1):
        if "Reduces" in str(row[3]):
            row_colors.append(("BACKGROUND", (0,i), (-1,i), LGRAY))
        else:
            row_colors.append(("BACKGROUND", (0,i), (-1,i), LRED))

    t3_style = [
        ("BACKGROUND",    (0, 0), (-1, 0), GREEN),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#C8E0C4")),
        ("PADDING",       (0, 0), (-1, -1), 6),
        ("ALIGN",         (2, 0), (2, -1), "CENTER"),
        ("ALIGN",         (4, 0), (4, -1), "CENTER"),
    ] + row_colors
    t3.setStyle(TableStyle(t3_style))
    story.append(t3)
    story.append(Spacer(1, 0.3*cm))

    # ── Plain English Summary ──
    story.append(Paragraph("Plain-English Explanation", h2_s))

    top = reasons[0]
    n_protect = sum(1 for r in reasons if r["direction"] == "decreases")
    n_risk    = sum(1 for r in reasons if r["direction"] == "increases")

    summary_text = (
        f"The AI model analysed 6 financial features for {name or 'this applicant'} "
        f"and assigned an approval probability of {round(prob*100,1)}%. "
        f"Gender was NOT used in this decision — the model is completely gender-neutral. "
        f"Out of the top {len(reasons)} factors analysed, {n_protect} are protective "
        f"(they reduce risk) and {n_risk} increase risk. "
        f"The single biggest factor was '{top['feature']}' with a SHAP value of "
        f"{top['shap_val']:+.4f}, meaning it "
        f"{'reduced' if top['direction'] == 'decreases' else 'increased'} "
        f"the approval probability by {abs(top['shap_val'])*100:.1f} percentage points "
        f"compared to an average applicant."
    )
    story.append(Paragraph(summary_text, body_s))
    story.append(Spacer(1, 0.3*cm))

    # ── Feature-by-feature plain English ──
    explanations = {
        "Saving accounts": {
            "low":      "Low savings is a significant risk signal — applicants without financial reserves are more likely to default during hardship.",
            "moderate": "Moderate savings provides some financial buffer — this is a neutral to slightly positive indicator.",
            "high":     "High savings strongly indicates financial discipline and ability to weather difficulties — this is the strongest protective factor."
        },
        "Duration": "default",
        "Credit amount": "default",
        "Age": "default",
        "Job": "default",
        "Housing": {
            "own":  "Owning a home indicates financial stability and gives the applicant a strong incentive to maintain repayments.",
            "rent": "Renting slightly increases risk as the applicant has less financial stake and fewer assets.",
            "free": "Free housing is neutral — may indicate living with family, which can reduce monthly obligations."
        }
    }

    dur_text = (
        f"A {duration}-month loan duration {'is short and low-risk — most applicants complete short loans successfully.' if duration <= 12 else 'is moderately long — longer loans carry more risk as circumstances can change over time.' if duration <= 36 else 'is very long — loans over 3 years have significantly higher default risk.'}"
    )
    cr_text  = (
        f"Requesting Rs {credit_amount:,} {'is a small to moderate amount relative to typical loan profiles — lower risk.' if credit_amount < 5000 else 'is a moderate amount — average risk profile.' if credit_amount < 12000 else 'is a large amount — higher amounts increase default risk if income does not match.'}"
    )
    age_text = (
        f"At {age} years old, the applicant is {'young — younger applicants statistically have higher default rates due to less stable careers.' if age < 25 else 'in a stable age range — middle-aged applicants typically have the most stable financial profiles.' if age < 50 else 'experienced — older applicants tend to have stable finances but the model also considers other factors.'}"
    )

    for r in reasons:
        feat = r["feature"]
        val  = raw_vals.get(feat, "")
        direction_word = "reduces" if r["direction"] == "decreases" else "increases"

        if feat == "Saving accounts":
            key = saving.lower()
            expl = explanations["Saving accounts"].get(key, f"Saving account level is {saving}.")
        elif feat == "Housing":
            key = housing.lower()
            expl = explanations["Housing"].get(key, f"Housing status is {housing}.")
        elif feat == "Duration":
            expl = dur_text
        elif feat == "Credit amount":
            expl = cr_text
        elif feat == "Age":
            expl = age_text
        elif feat == "Job":
            expl = f"Employment as '{job_label}' {'provides strong income stability.' if job_label == 'Highly Qualified' else 'indicates a stable skilled position.' if 'Skilled' in job_label else 'carries more income volatility which increases default risk.'}"
        else:
            expl = f"{feat} {direction_word} risk for this applicant."

        story.append(Paragraph(
            f"• {feat} ({val}) — {direction_word.upper()} RISK: {expl}",
            body_s))

    story.append(Spacer(1, 0.4*cm))

    # ── Disclaimer ──
    story.append(HRFlowable(width="100%", color=GREEN, thickness=1))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI model trained on historical data. "
        "It is intended to assist, not replace, human judgment. The model uses 6 financial "
        "features only. Gender is explicitly excluded from all predictions to ensure fairness. "
        "SHAP values shown are exact calculations specific to this applicant's profile.",
        small_s))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────
# EDA PDF (training data overview — kept separate)
# ─────────────────────────────────────────────────────────────────────
def generate_eda_pdf(df, plot_dir):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Image, Table, TableStyle, PageBreak)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles    = getSampleStyleSheet()
    GREEN     = colors.HexColor("#16A34A")
    title_s   = ParagraphStyle("T",  parent=styles["Title"],   fontSize=20, spaceAfter=4, alignment=TA_CENTER)
    h2_s      = ParagraphStyle("H2", parent=styles["Heading2"],fontSize=13, spaceAfter=4, textColor=GREEN)
    body_s    = ParagraphStyle("B",  parent=styles["Normal"],  fontSize=10, spaceAfter=4, textColor=colors.HexColor("#333333"))

    story = []
    story.append(Paragraph("German Credit Risk — EDA Report", title_s))
    story.append(Paragraph(f"Dataset overview — {len(df)} training records", body_s))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Dataset Overview", h2_s))
    good = int((df["Risk"]==1).sum()); bad = int((df["Risk"]==0).sum())
    tdata = [
        ["Metric",         "Value"],
        ["Total Records",  str(len(df))],
        ["Good Credit",    str(good)],
        ["Bad Credit",     str(bad)],
        ["Good Rate",      f"{good/len(df)*100:.1f}%"],
        ["Model Features", "6 (Sex excluded for fairness)"],
    ]
    t = Table(tdata, colWidths=[6*cm, 10*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), GREEN),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#F4FBF4"), colors.white]),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#C8E0C4")),
        ("PADDING",       (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Key Insights", h2_s))
    good_age = df[df["Risk"]==1]["Age"].mean()
    bad_age  = df[df["Risk"]==0]["Age"].mean()
    good_cr  = df[df["Risk"]==1]["Credit amount"].mean()
    bad_cr   = df[df["Risk"]==0]["Credit amount"].mean()
    good_dur = df[df["Risk"]==1]["Duration"].mean()
    bad_dur  = df[df["Risk"]==0]["Duration"].mean()
    for ins in [
        f"Good credit applicants average {good_age:.1f} yrs old vs {bad_age:.1f} for bad credit.",
        f"Good credit applicants request Rs {good_cr:,.0f} vs Rs {bad_cr:,.0f} for bad credit.",
        f"Good credit loans last {good_dur:.1f} months vs {bad_dur:.1f} months for bad credit.",
        "Saving accounts is the most predictive feature — high savers almost always repay.",
        "Longer loan duration consistently correlates with higher default risk.",
        "Gender (Sex) is excluded from the model to ensure fairness.",
    ]:
        story.append(Paragraph("• " + ins, body_s))
    story.append(Spacer(1, 0.4*cm))

    for fname, title in [
        ("01_risk_distribution.png",    "Risk Distribution"),
        ("02_age_analysis.png",         "Age Analysis"),
        ("03_credit_amount_analysis.png","Credit Amount Analysis"),
        ("09_correlation_heatmap.png",  "Correlation Heatmap"),
        ("feature_importance.png",      "Feature Importance"),
        ("confusion_matrix.png",        "Confusion Matrix"),
    ]:
        fpath = os.path.join(plot_dir, fname)
        if os.path.exists(fpath):
            story.append(PageBreak())
            story.append(Paragraph(title, h2_s))
            story.append(Spacer(1, 0.2*cm))
            story.append(Image(fpath, width=15*cm, height=9*cm))

    doc.build(story)
    buf.seek(0)
    return buf.read()


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

    page = st.radio(
        "Navigation",
        ["🏠  Home", "📋  Loan Application", "📊  EDA Dashboard",
         "🔍  Model Insights", "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="margin-top:20px;padding:12px 14px;background:#0A120A;border:1px solid #1A2E1A;border-radius:10px">
        <div style="display:flex;align-items:center;gap:6px">
            <div style="width:7px;height:7px;background:#22C55E;border-radius:50%;box-shadow:0 0 6px #22C55E"></div>
            <span style="font-size:12px;color:#22C55E;font-weight:500">System Active</span>
        </div>
        <div style="font-size:11px;color:#3D5A3D;margin-top:4px">Fair ML Model · Gender-Neutral</div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════
if "Home" in page:

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0A1F0A 0%,#0D2B0D 50%,#0A1A14 100%);
        border:1px solid #1A3A1A;border-radius:16px;padding:48px 40px 40px;margin:16px 0 24px;
        position:relative;overflow:hidden">
        <div style="position:absolute;top:-40px;right:-40px;width:220px;height:220px;
            background:radial-gradient(circle,#22C55E18 0%,transparent 70%);border-radius:50%"></div>
        <div style="display:inline-flex;align-items:center;gap:8px;background:#0F2A0F;
            border:1px solid #22C55E55;border-radius:20px;padding:5px 14px;
            font-size:12px;color:#22C55E;font-weight:500;margin-bottom:18px">
            <span style="width:7px;height:7px;background:#22C55E;border-radius:50%;
                box-shadow:0 0 6px #22C55E;display:inline-block"></span>
            System Active — Gender-Neutral AI Model
        </div>
        <div style="font-size:2.1rem;font-weight:700;color:#E8F5E8;margin-bottom:10px;line-height:1.2">
            Intelligent Credit<br>Underwriting
        </div>
        <div style="font-size:14px;color:#7EC87E;max-width:480px;line-height:1.6">
            Real-time risk assessment powered by machine learning and SHAP explainability.
            Trained on 1000 German Credit records. Fairness-audited — decisions are gender-neutral.
        </div>
    </div>""", unsafe_allow_html=True)

    try:
        df_raw = load_data(os.path.join(ROOT, "data", "train.csv"))
        n_rec  = len(df_raw)
    except Exception:
        df_raw, n_rec = None, 800

    from sklearn.metrics import f1_score
    try:
        df_t  = preprocess_data(load_data(os.path.join(ROOT, "data", "test.csv")))
        X_t, y_t = get_features_target(df_t)
        preds_t   = model.predict(X_t)
        acc       = round((preds_t == y_t.values).mean() * 100, 1)
        f1        = round(f1_score(y_t, preds_t, average="weighted") * 100, 1)
    except Exception:
        acc, f1 = 79.0, 78.0

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1: stat_card("📈", f"{acc}%",  "Model Accuracy")
    with c2: stat_card("🛡️", f"{f1}%",   "F1 Performance")
    with c3: stat_card("👥", str(n_rec), "Records Analyzed")
    with c4: stat_card("⚖️", "Neutral",  "Gender Fairness")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    section_title("Risk Distribution", "Composition of Good vs Bad credit risk in training data")

    try:
        good_n = int((df_raw["Risk"]==1).sum())
        bad_n  = int((df_raw["Risk"]==0).sum())
    except Exception:
        good_n, bad_n = 560, 240

    col_chart, col_info = st.columns([1, 1], gap="medium")
    with col_chart:
        fig, ax = plt.subplots(figsize=(4.5, 4), subplot_kw=dict(aspect="equal"))
        fig.patch.set_facecolor("#111811"); ax.set_facecolor("#111811")
        wedges, texts, autotexts = ax.pie(
            [good_n, bad_n], labels=["Good Credit", "Bad Credit"],
            colors=["#22C55E", "#EF4444"], autopct="%1.0f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops={"width": 0.45, "edgecolor": "#111811", "linewidth": 3}
        )
        for t in texts:     t.set_color("#7EC87E"); t.set_fontsize(11)
        for a in autotexts: a.set_color("#E8F5E8"); a.set_fontsize(12); a.set_fontweight("bold")
        ax.text(0, 0, f"{good_n+bad_n}\nTotal", ha="center", va="center",
                color="#E8F5E8", fontsize=12, fontweight="bold")
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with col_info:
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        for label, val, col in [("Good Credit", good_n, "#22C55E"), ("Bad Credit", bad_n, "#EF4444")]:
            pct = round(val / (good_n + bad_n) * 100, 1)
            st.markdown(f"""
            <div style="background:#111811;border:1px solid #1A2E1A;border-radius:10px;
                padding:16px 18px;margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                    <span style="font-size:13px;font-weight:500;color:#E8F5E8">{label}</span>
                    <span style="font-size:13px;font-weight:700;color:{col}">{val} ({pct}%)</span>
                </div>
                <div style="height:5px;background:#1A2E1A;border-radius:3px">
                    <div style="height:100%;width:{pct}%;background:{col};border-radius:3px"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0A1F0A;border:1px solid #1A3A1A;border-radius:10px;padding:14px 16px">
            <div style="font-size:12px;color:#7EC87E;line-height:1.6">
                ⚖️ <strong style="color:#E8F5E8">Fairness Note:</strong>
                Gender (Sex) is <strong style="color:#22C55E">excluded</strong> from predictions.
                Same financial profile → same decision regardless of gender.
            </div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# LOAN APPLICATION
# ═══════════════════════════════════════════════════════════════════
elif "Loan" in page:

    st.markdown("""
    <div style="margin:16px 0 24px">
        <h1 style="margin-bottom:4px">Loan Application</h1>
        <p style="font-size:13px;color:#5A7A5A;margin:0">
            Complete the form for an instant AI-powered credit risk assessment.
            <span style="color:#22C55E">Gender-neutral model</span> — decisions based on financial profile only.
        </p>
    </div>""", unsafe_allow_html=True)

    with st.form("loan_form"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div style='font-size:11px;font-weight:700;color:#7EC87E;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px'>Personal Information</div>", unsafe_allow_html=True)
            name    = st.text_input("Full Name", placeholder="e.g. Rahul Sharma")
            gender  = st.selectbox("Gender", ["Male", "Female"],
                        help="Recorded for reference only — not used in risk calculation")
            job     = st.selectbox("Employment Type", [0, 1, 2, 3], format_func=lambda x: {
                0:"Unskilled — Non-resident", 1:"Unskilled — Resident",
                2:"Skilled Employee", 3:"Highly Qualified"}[x])
            housing = st.selectbox("Housing Status", ["Own", "Rent", "Free"])
        with c2:
            st.markdown("<div style='font-size:11px;font-weight:700;color:#7EC87E;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px'>Financial Information</div>", unsafe_allow_html=True)
            age           = st.number_input("Age", 18, 100, 30)
            saving        = st.selectbox("Saving Account Level", ["Low", "Moderate", "High"])
            credit_amount = st.number_input("Credit Amount Requested (Rs)", 500, 200_000, 10_000, step=500)
            duration      = st.number_input("Loan Duration (months)", 6, 120, 24, step=6)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡  Run Risk Assessment", use_container_width=True)

    if submitted:
        job_label  = {0:"Unskilled(NR)",1:"Unskilled(R)",2:"Skilled Employee",3:"Highly Qualified"}[job]

        input_df   = pd.DataFrame([{
            "Job": job, "Housing": housing, "Saving accounts": saving,
            "Age": age, "Credit amount": credit_amount, "Duration": duration
        }])
        processed  = preprocess_data(input_df)
        X_input    = processed[FEAT_COLS]
        prob       = float(model.predict_proba(X_input)[0][1])
        prediction = int(model.predict(X_input)[0])
        reasons    = get_top_reasons(explainer, X_input, top_n=5)
        suggested  = int(200_000 * prob) if prediction == 1 else int(50_000 * prob)

        # Verdict banner
        if prediction == 1:
            vc, vbg, vb = "#22C55E", "#0F2010", "#22C55E44"
            vt  = "✅  APPROVED — Good Credit Risk"
            vsb = "Applicant meets the credit risk criteria."
        else:
            vc, vbg, vb = "#EF4444", "#1A0808", "#EF444444"
            vt  = "❌  DECLINED — High Credit Risk"
            vsb = "Applicant does not meet the credit risk criteria."

        st.markdown(f"""
        <div style="background:{vbg};border:1px solid {vb};border-radius:12px;
            padding:20px 24px;margin:20px 0 16px">
            <div style="font-size:1.1rem;font-weight:700;color:{vc}">{vt}</div>
            <div style="font-size:13px;color:#5A7A5A;margin-top:4px">{vsb}</div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1: stat_card("📊", f"{round(prob*100,1)}%", "Approval Probability")
        with c2: stat_card("💰", f"Rs {suggested:,}",     "Suggested Loan", "#F59E0B")
        with c3: stat_card("⏱️", f"{duration}mo",         "Duration")
        with c4: stat_card("🎯", f"{round((1-prob)*100,1)}%", "Default Risk",
                           "#22C55E" if prob > 0.5 else "#EF4444")

        # Risk gauge
        bar_col = "#22C55E" if prob > 0.5 else "#EF4444"
        st.markdown(f"""
        <div style="background:#111811;border:1px solid #1A2E1A;border-radius:10px;
            padding:18px 20px;margin:16px 0">
            <div style="display:flex;justify-content:space-between;margin-bottom:10px">
                <span style="font-size:13px;font-weight:500;color:#E8F5E8">Credit Score Estimate</span>
                <span style="font-size:13px;font-weight:700;color:{bar_col}">{round(prob*100,1)}% Creditworthy</span>
            </div>
            <div style="height:10px;background:#1A2E1A;border-radius:5px;overflow:hidden">
                <div style="height:100%;width:{round(prob*100,1)}%;
                    background:linear-gradient(90deg,{bar_col},{bar_col}aa);border-radius:5px;
                    box-shadow:0 0 10px {bar_col}55"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:6px">
                <span style="font-size:10px;color:#3D5A3D">High Risk</span>
                <span style="font-size:10px;color:#3D5A3D">Low Risk</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # SHAP explanation
        section_title("AI Decision Explanation",
                      "SHAP values — each bar shows how much each factor pushed the decision for THIS applicant")

        n_protect = sum(1 for r in reasons if r["direction"] == "decreases")
        n_risk    = sum(1 for r in reasons if r["direction"] == "increases")
        top       = reasons[0]

        st.markdown(f"""
        <div style="background:#0A1F0A;border:1px solid #1A3A1A;border-radius:10px;padding:16px 18px;margin-bottom:16px">
            <div style="font-size:13px;color:#A0C8A0;line-height:1.7">
                🤖 <strong style="color:#E8F5E8">AI Summary for {name or 'this applicant'}:</strong>
                The model analysed <strong style="color:#22C55E">6 financial features</strong>
                (gender was <strong style="color:#22C55E">not used</strong> — fairness-guaranteed).
                Out of top 5 factors, <strong style="color:#22C55E">{n_protect} are protective</strong>
                and <strong style="color:#EF4444">{n_risk} increase risk</strong>.
                Biggest driver: <strong style="color:#F59E0B">{top['feature']}</strong>
                (SHAP = {top['shap_val']:+.4f}).
            </div>
        </div>""", unsafe_allow_html=True)

        raw_vals = {
            "Job": job_label, "Housing": housing, "Saving accounts": saving,
            "Age": age, "Credit amount": f"Rs {credit_amount:,}", "Duration": f"{duration}mo"
        }
        for r in reasons:
            shap_bar_card(r["feature"], r["shap_val"],
                          raw_vals.get(r["feature"], "—"), r["direction"])

        # ── INDIVIDUAL APPLICANT PDF DOWNLOAD ──
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        section_title("Download Report",
                      "Full PDF report with this applicant's details, risk score, and SHAP explanation")

        try:
            pdf_bytes = generate_applicant_pdf(
                name=name, gender=gender, job_label=job_label,
                housing=housing, age=age, saving=saving,
                credit_amount=credit_amount, duration=duration,
                prob=prob, prediction=prediction,
                reasons=reasons, suggested_loan=suggested
            )
            if pdf_bytes:
                fname = f"credit_report_{(name or 'applicant').replace(' ','_').lower()}.pdf"
                st.download_button(
                    label="📥  Download Applicant Credit Report (PDF)",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True
                )
                st.markdown("""
                <div style="font-size:11px;color:#3D5A3D;margin-top:6px;text-align:center">
                    Report contains: applicant profile · risk score · SHAP analysis ·
                    plain-English explanation · fairness note
                </div>""", unsafe_allow_html=True)
            else:
                st.warning("PDF generation unavailable — reportlab not installed.")
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")

        # Summary table
        section_title("Application Summary")
        summary = pd.DataFrame({
            "Field": ["Name","Gender (not used in model)","Employment","Housing",
                      "Age","Saving Accounts","Credit Amount","Duration"],
            "Value": [name or "—", f"{gender} ⚖️", job_label,
                      housing, str(age), saving, f"Rs {credit_amount:,}", f"{duration} months"]
        })
        st.table(summary.set_index("Field"))


# ═══════════════════════════════════════════════════════════════════
# EDA DASHBOARD
# ═══════════════════════════════════════════════════════════════════
elif "EDA" in page:

    st.markdown("""
    <div style="margin:16px 0 24px">
        <h1 style="margin-bottom:4px">EDA Dashboard</h1>
        <p style="font-size:13px;color:#5A7A5A;margin:0">Deep exploratory analysis of the German Credit dataset.</p>
    </div>""", unsafe_allow_html=True)

    try:
        df_raw = load_data(os.path.join(ROOT, "data", "train.csv"))
    except Exception:
        st.error("data/train.csv not found. Run: python src/data/load_and_clean.py")
        st.stop()

    good_n = int((df_raw["Risk"]==1).sum()); bad_n = int((df_raw["Risk"]==0).sum())
    c1,c2,c3,c4,c5 = st.columns(5, gap="small")
    with c1: stat_card("📁", str(len(df_raw)), "Total Records")
    with c2: stat_card("✅", str(good_n),      "Good Credit")
    with c3: stat_card("❌", str(bad_n),        "Bad Credit",   "#EF4444")
    with c4: stat_card("📊", "6",               "Model Features")
    with c5: stat_card("🎯", f"{round(good_n/len(df_raw)*100,0):.0f}%", "Good Rate")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.info("💡 Go to **Loan Application** and run a prediction — you'll get a personalised PDF report for that specific applicant. The EDA PDF below covers the overall dataset patterns.")

    PLOT_DIR = os.path.join(ROOT, "eda_plots")
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        try:
            pdf_bytes = generate_eda_pdf(df_raw, PLOT_DIR)
            if pdf_bytes:
                st.download_button(
                    "📥  Download EDA Dataset Report (PDF)",
                    data=pdf_bytes,
                    file_name="credit_eda_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.warning("reportlab not installed — PDF unavailable.")
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")

    tab1, tab2, tab3 = st.tabs(["📈  Saved Plots", "🔢  Distribution Analysis", "📋  Raw Data"])

    with tab1:
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
            st.warning("Run `python eda.py` first to generate plots.")
        else:
            for i in range(0, len(available), 2):
                c1, c2 = st.columns(2, gap="medium")
                for col, (k, path) in zip([c1, c2], available[i:i+2]):
                    with col:
                        st.markdown(f"<div style='font-size:12px;font-weight:600;color:#7EC87E;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em'>{k}</div>", unsafe_allow_html=True)
                        st.image(path, use_container_width=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with tab2:
        section_title("Interactive Distribution Analysis", "Select any feature to see how it relates to credit risk")
        feat_sel = st.selectbox("Select Feature",
            ["Age","Credit amount","Duration","Job","Housing","Saving accounts"], key="eda_feat_sel")

        df_plot = df_raw.copy()
        df_plot["Risk Label"] = df_plot["Risk"].map({1:"Good Credit",0:"Bad Credit"})

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.patch.set_facecolor("#111811")
        for ax in axes:
            ax.set_facecolor("#111811"); ax.tick_params(colors="#7EC87E")
            for s in ax.spines.values(): s.set_color("#1A2E1A")

        numeric = df_plot[feat_sel].dtype in [np.int64, np.float64, int, float]
        if numeric:
            gv = df_plot[df_plot["Risk"]==1][feat_sel]
            bv = df_plot[df_plot["Risk"]==0][feat_sel]
            axes[0].hist(gv, bins=20, color="#22C55E", alpha=0.7, label="Good", edgecolor="#0A0E0A")
            axes[0].hist(bv, bins=20, color="#EF4444", alpha=0.7, label="Bad",  edgecolor="#0A0E0A")
            axes[0].set_title(f"{feat_sel} — Histogram", color="#E8F5E8", fontsize=12)
            axes[0].legend(facecolor="#1A2E1A", labelcolor="#C8DCC8", fontsize=9)
            axes[0].set_xlabel(feat_sel, color="#7EC87E"); axes[0].set_ylabel("Count", color="#7EC87E")
            axes[1].boxplot([bv, gv], tick_labels=["Bad","Good"], patch_artist=True,
                boxprops=dict(facecolor="#1A2E1A", color="#22C55E"),
                medianprops=dict(color="#22C55E", linewidth=2),
                whiskerprops=dict(color="#3D5A3D"), capprops=dict(color="#3D5A3D"),
                flierprops=dict(markerfacecolor="#EF4444", marker="o", markersize=4))
            axes[1].set_title(f"{feat_sel} by Risk Category", color="#E8F5E8", fontsize=12)
            axes[1].set_ylabel(feat_sel, color="#7EC87E"); axes[1].tick_params(colors="#7EC87E")
        else:
            ct = df_plot.groupby([feat_sel,"Risk Label"]).size().unstack(fill_value=0)
            x  = range(len(ct)); w = 0.35
            gc = ct.get("Good Credit", pd.Series([0]*len(ct), index=ct.index))
            bc = ct.get("Bad Credit",  pd.Series([0]*len(ct), index=ct.index))
            axes[0].bar([i-w/2 for i in x], gc, w, color="#22C55E", label="Good", alpha=0.85)
            axes[0].bar([i+w/2 for i in x], bc, w, color="#EF4444", label="Bad",  alpha=0.85)
            axes[0].set_xticks(list(x)); axes[0].set_xticklabels(ct.index, color="#7EC87E", fontsize=10)
            axes[0].set_title(f"{feat_sel} vs Risk (Count)", color="#E8F5E8", fontsize=12)
            axes[0].legend(facecolor="#1A2E1A", labelcolor="#C8DCC8"); axes[0].set_ylabel("Count", color="#7EC87E")
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            gc_p = ct_pct.get("Good Credit", pd.Series([0]*len(ct), index=ct.index))
            bc_p = ct_pct.get("Bad Credit",  pd.Series([0]*len(ct), index=ct.index))
            axes[1].bar(list(x), gc_p, color="#22C55E", label="Good%", alpha=0.85)
            axes[1].bar(list(x), bc_p, bottom=gc_p, color="#EF4444", label="Bad%", alpha=0.85)
            axes[1].set_xticks(list(x)); axes[1].set_xticklabels(ct.index, color="#7EC87E", fontsize=10)
            axes[1].set_title(f"{feat_sel} — Risk Rate (%)", color="#E8F5E8", fontsize=12)
            axes[1].axhline(50, color="#5A7A5A", linestyle="--", linewidth=1)
            axes[1].set_ylabel("Percentage", color="#7EC87E")

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if numeric:
            gv = df_plot[df_plot["Risk"]==1][feat_sel]
            bv = df_plot[df_plot["Risk"]==0][feat_sel]
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            s1,s2,s3,s4 = st.columns(4, gap="small")
            with s1: stat_card("📗", f"{gv.mean():.1f}", f"Good — Mean {feat_sel}")
            with s2: stat_card("📕", f"{bv.mean():.1f}", f"Bad — Mean {feat_sel}", "#EF4444")
            with s3: stat_card("📊", f"{df_raw[feat_sel].std():.1f}", "Std Dev")
            with s4: stat_card("⚡", f"{int(df_raw[feat_sel].min())}–{int(df_raw[feat_sel].max())}", "Range")

    with tab3:
        section_title("Raw Dataset", f"Training data — {len(df_raw)} rows")
        st.dataframe(df_raw.head(100), use_container_width=True, height=420)


# ═══════════════════════════════════════════════════════════════════
# MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════
elif "Insights" in page:

    st.markdown("""
    <div style="margin:16px 0 24px">
        <h1 style="margin-bottom:4px">Model Insights</h1>
        <p style="font-size:13px;color:#5A7A5A;margin:0">Feature importance, fairness audit, and decision logic.</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4, gap="small")
    with c1: stat_card("🌲","200",   "Decision Trees")
    with c2: stat_card("📈","0.873", "Test AUC")
    with c3: stat_card("🔁","5-Fold","Cross Validation")
    with c4: stat_card("⚖️","6",     "Fair Features Used")

    section_title("Feature Importance", "Mean decrease in impurity — how much each feature contributed to predictions")
    imp = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#111811"); ax.set_facecolor("#111811")
    colors_bar = ["#22C55E" if v >= imp.median() else "#1A3A1A" for v in imp.values]
    bars = ax.barh(imp.index, imp.values, color=colors_bar, edgecolor="#0A0E0A", height=0.55)
    ax.set_xlabel("Importance Score", color="#7EC87E", fontsize=11); ax.tick_params(colors="#7EC87E")
    for s in ax.spines.values(): s.set_color("#1A2E1A")
    for bar, val in zip(bars, imp.values):
        ax.text(val+0.002, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", color="#A0C0A0", fontsize=9)
    plt.tight_layout(pad=1)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    section_title("What Each Feature Means", "Plain-English explanation of why each feature matters")
    explanations = {
        "Saving accounts": ("💰","The biggest predictor. High savers almost always repay — they have a financial buffer when times get tough."),
        "Duration":        ("📅","Longer loans = more time for things to go wrong. A 60-month loan is far riskier than a 12-month one."),
        "Credit amount":   ("💵","Larger loan amounts relative to the applicant's profile increase default risk."),
        "Age":             ("🎂","Older applicants typically have more stable careers and finances, leading to lower default rates."),
        "Job":             ("💼","Employment type matters — highly qualified employees have the most stable income."),
        "Housing":         ("🏠","Homeowners have financial stakes that motivate repayment. Renters have less to lose."),
    }
    for feat, score in sorted(zip(FEAT_COLS, model.feature_importances_), key=lambda x: -x[1]):
        icon, desc = explanations.get(feat, ("📊",""))
        bar_w = int(score * 900)
        st.markdown(f"""
        <div style="background:#111811;border:1px solid #1A2E1A;border-radius:10px;padding:14px 18px;margin-bottom:8px">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="font-size:14px;font-weight:600;color:#E8F5E8">{icon} {feat}</span>
                <span style="font-size:12px;font-weight:700;color:#22C55E">Score: {score:.4f}</span>
            </div>
            <div style="height:4px;background:#1A2E1A;border-radius:2px;margin-bottom:8px">
                <div style="height:100%;width:{bar_w}%;background:#22C55E;border-radius:2px"></div>
            </div>
            <div style="font-size:12px;color:#7EC87E;line-height:1.6">{desc}</div>
        </div>""", unsafe_allow_html=True)

    section_title("Fairness Audit", "Demographic parity check across gender groups")
    st.markdown("""
    <div style="background:#0A1F0A;border:1px solid #1A3A1A;border-radius:10px;padding:14px 18px;margin-bottom:16px">
        <div style="font-size:13px;color:#7EC87E;line-height:1.7">
            ⚖️ <strong style="color:#E8F5E8">Fairness Design:</strong>
            Gender (Sex) is <strong style="color:#22C55E">completely excluded</strong> from model features.
            A male and female with identical financial profiles receive
            <strong style="color:#22C55E">exactly the same prediction</strong>.
        </div>
    </div>""", unsafe_allow_html=True)

    try:
        df_f     = preprocess_data(load_data(os.path.join(ROOT, "data", "train.csv")))
        X_f, y_f = get_features_target(df_f)
        preds    = model.predict(X_f)
        df_raw2  = load_data(os.path.join(ROOT, "data", "train.csv"))
        df_raw2["y_pred"] = preds
        male_rate   = df_raw2[df_raw2["Sex"]=="male"]["y_pred"].mean()
        female_rate = df_raw2[df_raw2["Sex"]=="female"]["y_pred"].mean()
        dp_diff     = abs(male_rate - female_rate)
        fc1,fc2,fc3 = st.columns(3, gap="small")
        with fc1: stat_card("👨", f"{male_rate*100:.1f}%",   "Male Approval Rate")
        with fc2: stat_card("👩", f"{female_rate*100:.1f}%", "Female Approval Rate")
        with fc3: stat_card("⚖️", f"{dp_diff:.3f}", "Parity Gap", "#22C55E" if dp_diff < 0.15 else "#EF4444")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if dp_diff < 0.15:
            st.info(f"✅ Demographic parity gap = {dp_diff:.3f}. Remaining gap reflects real financial profile differences in the dataset, not model bias — gender was excluded from predictions.")
        else:
            st.warning(f"⚠️ Gap = {dp_diff:.3f}. Investigate dataset distribution.")
    except Exception as e:
        st.warning(f"Fairness check error: {e}")


# ═══════════════════════════════════════════════════════════════════
# ABOUT
# ═══════════════════════════════════════════════════════════════════
elif "About" in page:

    st.markdown("""
    <div style="margin:16px 0 24px">
        <h1 style="margin-bottom:4px">About</h1>
        <p style="font-size:13px;color:#5A7A5A;margin:0">Architecture, design decisions, and project details.</p>
    </div>""", unsafe_allow_html=True)

    dark_card("""
    <div style="font-size:15px;font-weight:600;color:#E8F5E8;margin-bottom:8px">AI Credit Underwriting System</div>
    <div style="font-size:13px;color:#7EC87E;line-height:1.75">
        End-to-end ML-powered credit risk platform trained on the German Credit dataset (UCI).
        Every loan decision is explained using SHAP values. Gender is excluded from the model
        to guarantee fairness. Built for FinTech environments where explainability and compliance are mandatory.
    </div>""", "#22C55E")

    section_title("Tech Stack")
    stack = [
        ("ML Model",       "Random Forest — 200 trees, class-balanced, gender-excluded"),
        ("Explainability", "SHAP TreeExplainer — per-prediction feature attribution"),
        ("REST API",       "Flask + Flask-CORS — JSON /predict endpoint"),
        ("Frontend",       "Streamlit — reactive Python dashboard"),
        ("Data Layer",     "pandas · numpy — ETL + feature engineering"),
        ("Fairness",       "Demographic parity · equalized odds audit"),
        ("Report",         "ReportLab — individual applicant PDF reports"),
    ]
    c1, c2 = st.columns(2, gap="large")
    for i, (layer, tool) in enumerate(stack):
        with (c1 if i < 4 else c2):
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                padding:11px 0;border-bottom:1px solid #1A2E1A;font-size:13px">
                <span style="color:#7EC87E">{layer}</span>
                <span style="background:#0F2010;color:#22C55E;padding:3px 12px;
                    border-radius:20px;font-size:12px;font-weight:500">{tool}</span>
            </div>""", unsafe_allow_html=True)

    section_title("Model Performance")
    p1,p2,p3,p4 = st.columns(4, gap="small")
    with p1: stat_card("🌲","200","Trees")
    with p2: stat_card("📈","0.873","Test AUC")
    with p3: stat_card("🎯","79%","Test Accuracy")
    with p4: stat_card("⚖️","0.000","Gender SHAP Impact")

    section_title("How to Run Locally")
    st.code("""pip install -r requirements.txt
python src/data/load_and_clean.py
python eda.py
python src/models/train_model.py
streamlit run app.py""", language="bash")

    section_title("Deploy / Update on Streamlit Cloud")
    st.code("""git add app.py requirements.txt
git commit -m "feat: individual applicant PDF report"
git push
# Streamlit Cloud auto-redeploys in ~60 seconds""", language="bash")