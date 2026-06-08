import os
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================================
# Page configuration
# ==========================================================
st.set_page_config(
    page_title="AI Churn Early Warning Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic_saas_churn_dataset_v2.csv"
MODEL_PATH = BASE_DIR / "best_churn_model_v2.pkl"


# ==========================================================
# Styling
# ==========================================================
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --navy: #0B102F;
            --navy-2: #111827;
            --blue: #2563EB;
            --blue-soft: #DBEAFE;
            --teal: #0F766E;
            --teal-soft: #CCFBF1;
            --purple: #6D28D9;
            --purple-soft: #EDE9FE;
            --green: #16A34A;
            --green-soft: #DCFCE7;
            --orange: #F59E0B;
            --orange-soft: #FEF3C7;
            --red: #DC2626;
            --red-soft: #FEE2E2;
            --bg: #F8FAFC;
            --card: #FFFFFF;
            --border: #E2E8F0;
            --muted: #64748B;
        }

        .stApp {
            background: linear-gradient(180deg, #F8FAFC 0%, #EEF6FF 100%);
            color: var(--navy);
        }

        h1, h2, h3 {
            color: var(--navy);
            letter-spacing: -0.03em;
        }

        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--border);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--navy);
        }

        .main .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1440px;
        }

        .hero-card {
            background:
                radial-gradient(circle at 88% 18%, rgba(20, 184, 166, 0.28), transparent 30%),
                linear-gradient(135deg, #0B102F 0%, #123C69 58%, #0F766E 100%);
            color: white;
            padding: 34px 38px;
            border-radius: 28px;
            margin-bottom: 26px;
            box-shadow: 0 22px 55px rgba(15, 23, 42, 0.22);
            border: 1px solid rgba(255,255,255,0.12);
        }

        .hero-pill {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            background: rgba(255,255,255,0.13);
            border: 1px solid rgba(255,255,255,0.22);
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 18px;
            letter-spacing: 0.02em;
        }

        .hero-title {
            font-size: 46px;
            font-weight: 850;
            line-height: 1.04;
            margin-bottom: 12px;
            letter-spacing: -0.045em;
        }

        .hero-subtitle {
            font-size: 18px;
            opacity: 0.90;
            max-width: 920px;
            line-height: 1.45;
        }

        .hero-foot {
            margin-top: 22px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .hero-chip {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.18);
            font-size: 12px;
            font-weight: 650;
            color: rgba(255,255,255,0.92);
        }

        .section-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 30px 0 14px 0;
        }

        .section-dot {
            width: 13px;
            height: 13px;
            background: linear-gradient(135deg, var(--blue), var(--teal));
            border-radius: 999px;
            box-shadow: 0 0 0 6px rgba(37, 99, 235, 0.09);
        }

        .section-title {
            font-size: 25px;
            font-weight: 850;
            color: var(--navy);
            letter-spacing: -0.03em;
        }

        .section-subtitle {
            color: var(--muted);
            font-size: 14px;
            margin-top: -4px;
            margin-bottom: 16px;
        }

        .kpi-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 21px 21px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            min-height: 142px;
            position: relative;
            overflow: hidden;
        }

        .kpi-card:after {
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            width: 92px;
            height: 92px;
            background: radial-gradient(circle, rgba(15,118,110,0.12), transparent 66%);
        }

        .kpi-label {
            color: var(--muted);
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.055em;
            margin-bottom: 8px;
        }

        .kpi-value {
            font-size: 32px;
            font-weight: 850;
            color: var(--navy);
            margin-bottom: 6px;
            letter-spacing: -0.03em;
        }

        .kpi-help {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.35;
            max-width: 260px;
        }

        .health-score-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            margin-bottom: 16px;
        }

        .health-score-label {
            font-size: 12px;
            font-weight: 850;
            letter-spacing: 0.055em;
            text-transform: uppercase;
            color: var(--blue);
            margin-bottom: 10px;
        }

        .health-score-value {
            font-size: 43px;
            font-weight: 850;
            color: var(--navy);
            line-height: 1;
            margin-bottom: 10px;
            letter-spacing: -0.04em;
        }

        .health-score-status {
            display: inline-block;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .health-score-status.healthy { background: var(--green-soft); color: #166534; }
        .health-score-status.watchlist { background: var(--orange-soft); color: #92400E; }
        .health-score-status.atrisk { background: var(--red-soft); color: #991B1B; }

        .health-score-body {
            color: #475569;
            font-size: 14px;
            line-height: 1.55;
        }

        .priority-box, .insight-card, .detail-card, .signal-card, .method-card, .action-panel {
            background: white;
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.055);
        }

        .priority-box {
            min-height: 255px;
            margin-bottom: 10px;
        }

        .priority-label, .insight-label, .method-label {
            display: inline-block;
            font-size: 11px;
            font-weight: 850;
            letter-spacing: 0.065em;
            text-transform: uppercase;
            color: var(--blue);
            margin-bottom: 10px;
        }

        .priority-title {
            font-size: 22px;
            font-weight: 850;
            color: var(--navy);
            margin-bottom: 6px;
            letter-spacing: -0.03em;
        }

        .priority-sub {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 14px;
        }

        .priority-metric {
            font-size: 13px;
            color: #475569;
            margin-bottom: 6px;
        }

        .priority-metric strong { color: var(--navy); }

        .priority-highlight {
            margin-top: 14px;
            padding: 11px 12px;
            border-radius: 14px;
            background: #F8FAFC;
            color: #334155;
            font-size: 13px;
            line-height: 1.42;
            border: 1px solid #E5E7EB;
        }

        .risk-badge {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 850;
            border: 1px solid transparent;
        }

        .risk-high { background: var(--red-soft); color: #991B1B; border-color: #FECACA; }
        .risk-medium { background: var(--orange-soft); color: #92400E; border-color: #FDE68A; }
        .risk-low { background: var(--green-soft); color: #166534; border-color: #BBF7D0; }

        .priority-pill {
            display: inline-block;
            padding: 7px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 850;
            color: white;
            background: var(--blue);
            margin-right: 6px;
        }

        .priority-p1 { background: #991B1B; }
        .priority-p2 { background: #B45309; }
        .priority-p3 { background: #1D4ED8; }
        .priority-monitor { background: #475569; }

        .insight-title {
            color: var(--navy);
            font-weight: 850;
            margin-bottom: 6px;
            font-size: 17px;
            line-height: 1.25;
        }

        .insight-value {
            font-size: 31px;
            font-weight: 850;
            color: var(--navy);
            margin-bottom: 8px;
            letter-spacing: -0.03em;
        }

        .insight-body {
            color: #475569;
            font-size: 14px;
            line-height: 1.5;
        }

        .insight-summary-box {
            margin-top: 14px;
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-left: 5px solid var(--teal);
            border-radius: 18px;
            padding: 17px 19px;
            color: #334155;
            font-size: 14px;
            line-height: 1.6;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
        }

        .account-header-card {
            background:
                radial-gradient(circle at 92% 20%, rgba(37,99,235,0.10), transparent 24%),
                white;
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.055);
            margin-bottom: 16px;
        }

        .account-header-title {
            font-size: 30px;
            font-weight: 850;
            color: var(--navy);
            margin-bottom: 7px;
            letter-spacing: -0.04em;
        }

        .account-header-sub {
            color: var(--muted);
            font-size: 15px;
            margin-bottom: 13px;
        }

        .account-chip {
            display: inline-block;
            background: #EEF2FF;
            color: #4338CA;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 12px;
            font-weight: 750;
            margin-right: 8px;
            margin-bottom: 8px;
        }

        .detail-card { margin-bottom: 14px; }
        .detail-card.risk-high { border-left: 6px solid var(--red); background: linear-gradient(180deg, #FFF8F8 0%, #FFFFFF 100%); }
        .detail-card.risk-medium { border-left: 6px solid var(--orange); background: linear-gradient(180deg, #FFFBEB 0%, #FFFFFF 100%); }
        .detail-card.risk-low { border-left: 6px solid var(--green); background: linear-gradient(180deg, #F6FFFA 0%, #FFFFFF 100%); }

        .detail-card-title {
            font-size: 16px;
            font-weight: 850;
            margin-bottom: 8px;
            color: var(--navy);
        }

        .detail-card-body {
            font-size: 14px;
            line-height: 1.62;
            color: #374151;
        }

        .signal-card { min-height: 132px; }

        .signal-label {
            font-size: 12px;
            color: var(--muted);
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.045em;
            margin-bottom: 6px;
        }

        .signal-value {
            font-size: 28px;
            font-weight: 850;
            color: var(--navy);
            margin-bottom: 8px;
            letter-spacing: -0.03em;
        }

        .signal-note {
            font-size: 13px;
            color: #475569;
            line-height: 1.4;
        }

        .action-panel {
            background: linear-gradient(135deg, #EFF6FF 0%, #F8FAFC 100%);
            border-color: #DBEAFE;
            margin-bottom: 14px;
        }

        .action-panel-title {
            font-size: 16px;
            font-weight: 850;
            color: #1D4ED8;
            margin-bottom: 10px;
        }

        .method-card { min-height: 210px; }
        .method-title { font-size: 19px; font-weight: 850; color: var(--navy); margin-bottom: 8px; }
        .method-body { color: #475569; font-size: 14px; line-height: 1.55; }
        .formula-box {
            background: #F8FAFC;
            border: 1px solid var(--border);
            padding: 14px;
            border-radius: 14px;
            font-weight: 850;
            color: var(--navy);
            margin: 12px 0;
            font-size: 14px;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid var(--border);
            padding: 18px;
            border-radius: 18px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 9px 16px;
            background: #FFFFFF;
            border: 1px solid var(--border);
            color: #334155;
        }

        .stTabs [aria-selected="true"] {
            background: var(--navy) !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-pill">AI Churn Copilot V2 · Retention Intelligence Workspace</div>
            <div class="hero-title">Predict churn risk before it becomes churn.</div>
            <div class="hero-subtitle">
                A proactive customer health dashboard that combines usage, renewal context, support experience,
                sentiment, product feedback and bug friction into explainable account-level churn probabilities.
            </div>
            <div class="hero-foot">
                <span class="hero-chip">Portfolio monitoring</span>
                <span class="hero-chip">Account-level explanations</span>
                <span class="hero-chip">Product feedback signals</span>
                <span class="hero-chip">Bug resolution signals</span>
                <span class="hero-chip">Next-best-action guidance</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-dot"></div>
            <div class="section-title">{html.escape(title)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value">{html.escape(str(value))}</div>
            <div class="kpi-help">{html.escape(help_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level: str) -> str:
    css_class = "risk-low"
    if str(level).lower() == "high":
        css_class = "risk-high"
    elif str(level).lower() == "medium":
        css_class = "risk-medium"
    return f'<span class="risk-badge {css_class}">{html.escape(str(level))} risk</span>'


def priority_pill(priority: str) -> str:
    mapping = {
        "P1": "priority-p1",
        "P2": "priority-p2",
        "P3": "priority-p3",
        "Monitor": "priority-monitor",
    }
    css_class = mapping.get(str(priority), "priority-monitor")
    return f'<span class="priority-pill {css_class}">{html.escape(str(priority))}</span>'


# ==========================================================
# Data and model loading
# ==========================================================
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: Path):
    model_package = joblib.load(path)
    if isinstance(model_package, dict):
        return model_package["model"], model_package["columns"]
    return model_package, None


def prepare_features(df: pd.DataFrame, model_columns=None) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ["account_id", "churn_label"]]
    X = df[feature_cols].copy()
    X = pd.get_dummies(X, drop_first=True)
    if model_columns is not None:
        X = X.reindex(columns=model_columns, fill_value=0)
    return X


def safe_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return preds.astype(float)


# ==========================================================
# Business logic
# ==========================================================
def risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def safe_row_get(row: pd.Series, key: str, default=0):
    if key in row.index and pd.notna(row[key]):
        return row[key]
    return default


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def fmt_num(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def fmt_currency(value: float) -> str:
    if pd.isna(value):
        return "$0"
    return f"${value:,.0f}"


def priority_label(row: pd.Series) -> str:
    if row["risk_level"] == "High" and row["mrr"] >= 2000:
        return "P1"
    if row["risk_level"] == "High":
        return "P2"
    if row["risk_level"] == "Medium" and row["renewal_due_days"] < 30:
        return "P2"
    if row["risk_level"] == "Medium":
        return "P3"
    return "Monitor"


def dominant_risk_theme(row: pd.Series) -> str:
    if row["renewal_due_days"] < 30 and row["renewal_intent"] in ["neutral", "negative"]:
        return "Commercial risk"
    if row["data_accuracy_bug_flag"] == 1 or row["stuck_bugs_count"] >= 1:
        return "Product reliability risk"
    if row["critical_feedback_count_90d"] >= 1 or row["roadmap_alignment_score"] < 40:
        return "Product gap risk"
    if row["tool_activity_score"] < 35 and row["feature_adoption_score"] < 40:
        return "Adoption risk"
    if row["reopened_tickets"] >= 2 or row["csat_support"] < 3.2:
        return "Support risk"
    if row["sentiment_csm"] < -0.2:
        return "Relationship risk"
    return "General health risk"


def build_risk_chips(row: pd.Series) -> list[str]:
    chips = []
    if row["tool_activity_score"] < 35:
        chips.append("Low activity")
    if row["feature_adoption_score"] < 40:
        chips.append("Weak adoption")
    if row["usage_change_vs_prev_quarter"] < -20:
        chips.append("Usage decline")
    if row["csat_support"] < 3.2:
        chips.append("Low support CSAT")
    if row["reopened_tickets"] >= 2:
        chips.append("Support friction")
    if row["sentiment_csm"] < -0.2:
        chips.append("Negative sentiment")
    if row["renewal_due_days"] < 30 and row["renewal_intent"] in ["neutral", "negative"]:
        chips.append("Renewal risk")
    if row["critical_feedback_count_90d"] >= 1:
        chips.append("Critical feedback")
    if row["roadmap_alignment_score"] < 40:
        chips.append("Roadmap mismatch")
    if row["stuck_bugs_count"] >= 1:
        chips.append("Stuck bug")
    if row["data_accuracy_bug_flag"] == 1:
        chips.append("Data trust issue")
    return chips[:7]


def build_explanation(row: pd.Series) -> str:
    risk = row["risk_level"]
    signals = []

    if row["tool_activity_score"] < 35:
        signals.append("product engagement is weak")
    elif row["tool_activity_score"] < 55:
        signals.append("product usage is below the ideal level")

    if row["feature_adoption_score"] < 40:
        signals.append("adoption of key capabilities is limited")
    elif row["feature_adoption_score"] < 60:
        signals.append("feature adoption is only partial")

    if row["usage_change_vs_prev_quarter"] < -20:
        signals.append("usage has declined sharply compared with the previous quarter")
    elif row["usage_change_vs_prev_quarter"] < -5:
        signals.append("usage is trending down compared with the previous quarter")

    if row["csat_support"] < 3.2:
        signals.append("support satisfaction is below target")

    if row["reopened_tickets"] >= 2:
        signals.append("there is repeated support friction")

    if row["sentiment_csm"] < -0.2:
        signals.append("recent Customer Success interactions show negative sentiment")

    if row["renewal_due_days"] < 30 and row["renewal_intent"] == "negative":
        signals.append("renewal risk is high in the short term")
    elif row["renewal_due_days"] < 30 and row["renewal_intent"] == "neutral":
        signals.append("renewal confidence remains uncertain")

    if row["critical_feedback_count_90d"] >= 1:
        signals.append("the customer has raised critical product feedback")

    if row["unresolved_feedback_count"] >= 3:
        signals.append("several product feedback items remain unresolved")

    if row["avg_unresolved_feedback_age_days"] > 60:
        signals.append("unresolved feedback has been open for a long time")

    if row["roadmap_alignment_score"] < 40:
        signals.append("customer needs show low alignment with the current roadmap")

    if row["feedback_sentiment_score"] < -0.2:
        signals.append("feedback-related sentiment is negative")

    if row["competitor_mentioned_in_feedback"] == 1:
        signals.append("a competitor has been mentioned in product feedback")

    if row["feedback_linked_to_renewal"] == 1:
        signals.append("product feedback is linked to renewal conversations")

    if row["critical_bugs_reported_90d"] >= 1:
        signals.append("critical bugs have been reported")

    if row["stuck_bugs_count"] >= 1:
        signals.append("there are stuck unresolved bugs")

    if row["oldest_open_bug_days"] > 45:
        signals.append("the oldest open bug has been unresolved for too long")

    if row["data_accuracy_bug_flag"] == 1:
        signals.append("there is a data accuracy or trust-related bug")

    if row["bug_linked_to_renewal_flag"] == 1:
        signals.append("a bug is linked to renewal conversations")

    if not signals:
        if risk == "Low":
            return "This account currently appears healthy overall. No major churn signals are visible at this stage, so regular monitoring should be enough."
        if risk == "Medium":
            return "This account shows some moderate attention points. A preventive review would help confirm whether risk is increasing."
        return "This account is classified as high risk and should be reviewed closely, even if the visible signals are distributed across several areas."

    if len(signals) == 1:
        signal_text = signals[0]
    elif len(signals) == 2:
        signal_text = signals[0] + " and " + signals[1]
    else:
        signal_text = ", ".join(signals[:3])

    if risk == "Low":
        return f"This account currently looks relatively stable, although {signal_text}. These signals do not yet suggest critical churn risk, but they are worth monitoring."
    if risk == "Medium":
        return f"This account shows early signs of deterioration because {signal_text}. The situation is not yet critical, but a proactive intervention would be advisable."

    ending = ""
    if row["mrr"] >= 5000:
        ending = " Given the revenue weight of this account, the risk deserves immediate attention."
    elif row["renewal_due_days"] < 30:
        ending = " The timing is especially sensitive because renewal is approaching."
    return f"This account is being prioritised because {signal_text}. It should be treated as a high-risk case for intervention.{ending}"


def build_recommendation(row: pd.Series, churn_probability: float) -> str:
    account_id = html.escape(str(row.get("account_id", "This account")))

    risk_band = "low"
    if churn_probability >= 0.70:
        risk_band = "high"
    elif churn_probability >= 0.40:
        risk_band = "medium"

    negative_signals = []
    positive_signals = []

    if row["csat_support"] < 3.2:
        negative_signals.append("low support CSAT is weakening the customer experience")
    elif row["csat_support"] >= 4.2:
        positive_signals.append("support satisfaction remains solid")

    if row["sentiment_csm"] < -0.2:
        negative_signals.append("CSM sentiment suggests relationship friction")
    elif row["sentiment_csm"] > 0.25:
        positive_signals.append("CSM sentiment suggests a constructive relationship")

    if row["login_count_30d"] < 5:
        negative_signals.append("login activity is very limited")
    elif row["login_count_30d"] >= 18:
        positive_signals.append("login activity indicates recurring usage")

    if row["feature_adoption_score"] < 40:
        negative_signals.append("feature adoption appears shallow")
    elif row["feature_adoption_score"] >= 70:
        positive_signals.append("feature adoption is relatively strong")

    if row["tickets_last_quarter"] >= 8:
        negative_signals.append("support demand is elevated and may indicate operational friction")

    if row["critical_feedback_count_90d"] >= 1 or row["roadmap_alignment_score"] < 40:
        negative_signals.append("unmet product needs or roadmap mismatch may be increasing risk")

    if row["stuck_bugs_count"] >= 1 or row["data_accuracy_bug_flag"] == 1:
        negative_signals.append("product reliability or data trust issues require attention")

    if row["feedback_linked_to_renewal"] == 1 or row["bug_linked_to_renewal_flag"] == 1:
        negative_signals.append("product friction is connected to renewal conversations")

    impact_note = ""
    if row["mrr"] >= 5000:
        impact_note = " Given the revenue impact of this account, proactive ownership is especially important."
    elif row["mrr"] >= 1500:
        impact_note = " This account also carries meaningful commercial value and deserves close attention."

    if risk_band == "high":
        opening = f"<strong>High-risk retention priority.</strong> {account_id} shows a material likelihood of churn and should be treated as an active retention case."
        action = "Recommended next step: schedule proactive outreach with a clear recovery plan, align on the main risk drivers, and define one or two concrete actions the customer can feel within the next cycle."
    elif risk_band == "medium":
        opening = f"<strong>Medium-risk account to stabilize.</strong> {account_id} is not yet critical, but several indicators suggest the relationship may weaken if no action is taken."
        action = "Recommended next step: run a structured check-in, validate whether the customer is realizing value, and intervene early on adoption, support, product feedback or bug-related blockers."
    else:
        opening = f"<strong>Healthy account with expansion potential.</strong> {account_id} appears relatively stable at this stage, with no immediate signs of severe churn risk."
        action = "Recommended next step: maintain a steady engagement rhythm, reinforce business value, and look for opportunities to deepen adoption or identify expansion signals."

    signals_text = ""
    if negative_signals:
        signals_text += " Key risk signals: " + "; ".join(negative_signals).capitalize() + "."
    if positive_signals:
        signals_text += " Positive signals: " + "; ".join(positive_signals).capitalize() + "."
    if not negative_signals and risk_band in ["high", "medium"]:
        signals_text += " The account risk is elevated even though the visible operational signals are mixed, so a qualitative review with the CSM is recommended."
    if not positive_signals and risk_band == "low":
        signals_text += " Even if the account is currently low risk, it would still benefit from stronger evidence of value realization and long-term stickiness."

    return f"{opening} {action}{signals_text}{impact_note}"


def top_signal_table(row: pd.Series) -> pd.DataFrame:
    signals = [
        ("Product activity score", row["tool_activity_score"], "Usage"),
        ("Feature adoption score", row["feature_adoption_score"], "Usage"),
        ("Usage change vs previous quarter", row["usage_change_vs_prev_quarter"], "Usage"),
        ("Days since last login", row["days_since_last_login"], "Usage"),
        ("Support CSAT", row["csat_support"], "Support"),
        ("Reopened tickets", row["reopened_tickets"], "Support"),
        ("CSM sentiment", row["sentiment_csm"], "Sentiment"),
        ("Unresolved feedback", row["unresolved_feedback_count"], "Feedback"),
        ("Roadmap alignment score", row["roadmap_alignment_score"], "Feedback"),
        ("Stuck bugs", row["stuck_bugs_count"], "Bugs"),
        ("Oldest open bug days", row["oldest_open_bug_days"], "Bugs"),
        ("Data accuracy bug flag", row["data_accuracy_bug_flag"], "Bugs"),
    ]
    out = pd.DataFrame(signals, columns=["Signal", "Value", "Family"])
    out["Value"] = out["Value"].map(lambda x: f"{x:.2f}" if isinstance(x, float) else str(x))
    return out


def portfolio_driver_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Driver", "Share of accounts"])
    signals = {
        "Low product activity": (df["tool_activity_score"] < 35).mean(),
        "Low feature adoption": (df["feature_adoption_score"] < 40).mean(),
        "Negative CSM sentiment": (df["sentiment_csm"] < -0.2).mean(),
        "Low support CSAT": (df["csat_support"] < 3.2).mean(),
        "Reopened tickets": (df["reopened_tickets"] >= 2).mean(),
        "Renewal risk": ((df["renewal_due_days"] < 30) & (df["renewal_intent"].isin(["neutral", "negative"]))).mean(),
        "Critical feedback": (df["critical_feedback_count_90d"] >= 1).mean(),
        "Roadmap mismatch": (df["roadmap_alignment_score"] < 40).mean(),
        "Stuck bugs": (df["stuck_bugs_count"] >= 1).mean(),
        "Data accuracy bugs": (df["data_accuracy_bug_flag"] == 1).mean(),
        "No workaround available": (df["workaround_available_flag"] == 0).mean(),
    }
    out = pd.DataFrame({
        "Driver": list(signals.keys()),
        "Share of accounts": [round(v * 100, 1) for v in signals.values()],
    }).sort_values("Share of accounts", ascending=False).reset_index(drop=True)
    return out


def compute_portfolio_health_score(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0

    avg_churn = df["churn_probability"].mean()
    high_risk_share = (df["risk_level"] == "High").mean()
    activity = df["tool_activity_score"].mean() / 100
    adoption = df["feature_adoption_score"].mean() / 100
    csat = df["csat_support"].mean() / 5
    sentiment = (df["sentiment_csm"].mean() + 1) / 2
    roadmap = df["roadmap_alignment_score"].mean() / 100
    bug_pressure = min(1.0, df["stuck_bugs_count"].mean() / 2)
    feedback_pressure = min(1.0, df["unresolved_feedback_count"].mean() / 5)

    score = (
        (1 - avg_churn) * 0.30
        + (1 - high_risk_share) * 0.18
        + activity * 0.12
        + adoption * 0.12
        + csat * 0.10
        + sentiment * 0.06
        + roadmap * 0.06
        + (1 - bug_pressure) * 0.03
        + (1 - feedback_pressure) * 0.03
    ) * 100

    return round(float(score), 1)


def portfolio_health_status(score: float) -> tuple[str, str]:
    if score >= 75:
        return "Healthy", "healthy"
    if score >= 55:
        return "Watchlist", "watchlist"
    return "At Risk", "atrisk"


# ==========================================================
# App start
# ==========================================================
inject_custom_css()
render_hero()

missing = []
if not os.path.exists(DATA_PATH):
    missing.append(str(DATA_PATH))
if not os.path.exists(MODEL_PATH):
    missing.append(str(MODEL_PATH))

if missing:
    st.error(
        "Missing required file(s): "
        + ", ".join(missing)
        + ". Put the V2 dataset CSV and the trained V2 model PKL in the same folder as this app."
    )
    st.stop()

# Load and score data
df = load_data(DATA_PATH)
model, model_columns = load_model(MODEL_PATH)
X = prepare_features(df, model_columns)
df["churn_probability"] = safe_predict_proba(model, X)
df["risk_level"] = df["churn_probability"].apply(risk_level)
df["priority"] = df.apply(priority_label, axis=1)
df = df.sort_values("churn_probability", ascending=False).reset_index(drop=True)

# Sidebar filters
st.sidebar.markdown("## Filters")
st.sidebar.caption("Narrow the portfolio view by customer profile and risk level.")
segments = ["All"] + sorted(df["segment"].dropna().unique().tolist())
plans = ["All"] + sorted(df["plan_tier"].dropna().unique().tolist())
risk_levels = ["All", "Low", "Medium", "High"]
priorities = ["All", "P1", "P2", "P3", "Monitor"]

segment_filter = st.sidebar.selectbox("Segment", segments)
plan_filter = st.sidebar.selectbox("Plan tier", plans)
risk_filter = st.sidebar.selectbox("Risk level", risk_levels)
priority_filter = st.sidebar.selectbox("Priority", priorities)
min_mrr = st.sidebar.slider("Minimum MRR", 0, int(max(df["mrr"].max(), 1)), 0, step=250)

filtered = df.copy()
if segment_filter != "All":
    filtered = filtered[filtered["segment"] == segment_filter]
if plan_filter != "All":
    filtered = filtered[filtered["plan_tier"] == plan_filter]
if risk_filter != "All":
    filtered = filtered[filtered["risk_level"] == risk_filter]
if priority_filter != "All":
    filtered = filtered[filtered["priority"] == priority_filter]
filtered = filtered[filtered["mrr"] >= min_mrr]
filtered = filtered.sort_values("churn_probability", ascending=False).reset_index(drop=True)

if filtered.empty:
    st.warning("No accounts match the current filters. Adjust the filters in the sidebar to continue.")
    st.stop()

retention_rate = ((filtered["renewed_last_quarter"] == 1) | (filtered["subscription_status"] == "active")).mean()
portfolio_health_score = compute_portfolio_health_score(filtered)
portfolio_health_label, portfolio_health_class = portfolio_health_status(portfolio_health_score)
mrr_at_risk = filtered.loc[filtered["risk_level"] == "High", "mrr"].sum()

# Executive snapshot
section_header(
    "Executive portfolio snapshot",
    "A quick view of portfolio risk, commercial exposure, and retention health.",
)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    metric_card("Accounts", f"{len(filtered):,}", "Accounts currently included in the selected view.")
with col2:
    metric_card("Average churn probability", fmt_pct(filtered["churn_probability"].mean()), "Average predicted churn risk across the selected portfolio.")
with col3:
    metric_card("High-risk accounts", f"{int((filtered['risk_level'] == 'High').sum()):,}", "Accounts currently classified as high churn risk.")
with col4:
    metric_card("Retention snapshot", fmt_pct(retention_rate), "Share of accounts active or recently renewed.")
with col5:
    metric_card("MRR at risk", fmt_currency(mrr_at_risk), "Monthly recurring revenue linked to high-risk accounts.")

st.markdown(
    f"""
    <div class="health-score-card">
        <div class="health-score-label">Portfolio health</div>
        <div class="health-score-value">{portfolio_health_score} / 100</div>
        <div class="health-score-status {portfolio_health_class}">{portfolio_health_label}</div>
        <div class="health-score-body">
            This score summarises the current portfolio view using average churn probability, share of high-risk accounts,
            product activity, feature adoption, support satisfaction, CSM sentiment, roadmap alignment, unresolved feedback
            and bug pressure.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# V2 signals snapshot
section_header(
    "Product friction snapshot",
    "New V2 indicators focused on unmet product needs and product reliability issues.",
)

v1, v2, v3, v4, v5 = st.columns(5)
with v1:
    metric_card("Critical feedback", f"{int((filtered['critical_feedback_count_90d'] >= 1).sum()):,}", "Accounts with at least one critical feedback item.")
with v2:
    metric_card("Roadmap mismatch", f"{int((filtered['roadmap_alignment_score'] < 40).sum()):,}", "Accounts whose requests have low roadmap alignment.")
with v3:
    metric_card("Stuck bugs", f"{int((filtered['stuck_bugs_count'] >= 1).sum()):,}", "Accounts with bugs stuck beyond threshold or SLA.")
with v4:
    metric_card("Data accuracy bugs", f"{int((filtered['data_accuracy_bug_flag'] == 1).sum()):,}", "Accounts affected by data trust or integrity issues.")
with v5:
    metric_card("Avg. bug resolution", f"{filtered['avg_bug_resolution_days'].mean():.1f} days", "Average time required to resolve reported bugs.")

# Top 3 priority cards
section_header(
    "Top 3 priority accounts",
    "Accounts that currently deserve the fastest review based on predicted churn probability and commercial relevance.",
)

top3 = filtered.head(3)
pcols = st.columns(3)
for i, (_, row) in enumerate(top3.iterrows()):
    with pcols[i]:
        card_html = (
            f'<div class="priority-box">'
            f'{priority_pill(row["priority"])} {risk_badge(row["risk_level"])}'
            f'<div class="priority-title" style="margin-top:12px;">{html.escape(str(row["account_id"]))}</div>'
            f'<div class="priority-sub">{html.escape(str(row["segment"]))} · {html.escape(str(row["plan_tier"]))} · {html.escape(str(row["subscription_type"]))}</div>'
            f'<div class="priority-metric"><strong>Churn probability:</strong> {fmt_pct(row["churn_probability"])}</div>'
            f'<div class="priority-metric"><strong>MRR:</strong> {fmt_currency(row["mrr"])}</div>'
            f'<div class="priority-metric"><strong>Renewal intent:</strong> {html.escape(str(row["renewal_intent"]))}</div>'
            f'<div class="priority-metric"><strong>Renewal due:</strong> {int(row["renewal_due_days"])} days</div>'
            f'<div class="priority-highlight">{html.escape(build_explanation(row))}</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Dashboard",
    "Account Table",
    "Account Detail",
    "Methodology",
])

with tab1:
    section_header("Risk distribution", "How risk is distributed across the current portfolio view.")
    c1, c2 = st.columns(2)

    with c1:
        risk_counts = filtered["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig, ax = plt.subplots(figsize=(6.4, 4))
        bars = ax.bar(risk_counts.index, risk_counts.values, color=["#16A34A", "#F59E0B", "#DC2626"])
        ax.set_title("Accounts by risk level", fontsize=12, pad=12, color="#0B102F")
        ax.set_ylabel("Accounts", fontsize=10, color="#475569")
        upper = max(risk_counts.values) * 1.20 if max(risk_counts.values) > 0 else 1
        ax.set_ylim(0, upper)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(axis="x", labelsize=10, colors="#475569", length=0)
        ax.tick_params(axis="y", labelsize=9, colors="#64748B")
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, risk_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + upper * 0.025, f"{int(value)}", ha="center", va="bottom", fontsize=9, color="#334155")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        churn_by_segment = filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6.4, 4))
        bars = ax.barh(churn_by_segment.index, churn_by_segment.values, color="#2563EB")
        ax.set_title("Average churn probability by segment", fontsize=12, pad=12, color="#0B102F")
        ax.set_xlabel("Churn probability", fontsize=10, color="#475569")
        upper = max(churn_by_segment.values) * 1.20 if len(churn_by_segment.values) > 0 else 1
        ax.set_xlim(0, upper)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(axis="x", labelsize=9, colors="#64748B")
        ax.tick_params(axis="y", labelsize=10, colors="#475569", length=0)
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, churn_by_segment.values):
            ax.text(value + upper * 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", ha="left", fontsize=9, color="#334155")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    section_header("Top risk drivers in portfolio", "Most common risk patterns in the current filtered view.")
    drivers = portfolio_driver_summary(filtered)

    top_driver = drivers.iloc[0]["Driver"] if len(drivers) > 0 else "No driver available"
    top_share = drivers.iloc[0]["Share of accounts"] if len(drivers) > 0 else 0
    second_driver = drivers.iloc[1]["Driver"] if len(drivers) > 1 else "No second driver"
    second_share = drivers.iloc[1]["Share of accounts"] if len(drivers) > 1 else 0
    renewal_risk_count = int(((filtered["renewal_due_days"] < 30) & (filtered["renewal_intent"].isin(["neutral", "negative"]))).sum())
    highest_segment = filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=False).index[0]
    highest_segment_risk = filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=False).iloc[0]

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-label">Primary risk driver</div>
                <div class="insight-title">{html.escape(str(top_driver))}</div>
                <div class="insight-value">{top_share}%</div>
                <div class="insight-body">This is the most common risk pattern in the current portfolio view and should be addressed first at scale.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ic2:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-label">Highest-risk segment</div>
                <div class="insight-title">{html.escape(str(highest_segment))}</div>
                <div class="insight-value">{highest_segment_risk:.1%}</div>
                <div class="insight-body">This segment currently shows the highest average churn probability and deserves closer portfolio monitoring.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ic3:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-label">Renewal pressure</div>
                <div class="insight-title">Accounts needing follow-up</div>
                <div class="insight-value">{renewal_risk_count}</div>
                <div class="insight-body">These accounts are approaching renewal with uncertain or negative intent and may require immediate action.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    summary_text = (
        f"In this portfolio view, the main structural weakness is <strong>{html.escape(str(top_driver))}</strong>, affecting <strong>{top_share}%</strong> of accounts. "
        f"The second most common signal is <strong>{html.escape(str(second_driver))}</strong> at <strong>{second_share}%</strong>. "
        f"The most exposed segment is <strong>{html.escape(str(highest_segment))}</strong>, while <strong>{renewal_risk_count}</strong> accounts may require short-term renewal follow-up."
    )
    st.markdown(f'<div class="insight-summary-box">{summary_text}</div>', unsafe_allow_html=True)

    st.markdown("#### Driver detail")
    st.dataframe(drivers, use_container_width=True, hide_index=True)

with tab2:
    section_header("Account Table", "A ranked operational view of accounts, ordered from highest to lowest churn probability.")

    priority_counts = filtered["priority"].value_counts()
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.metric("P1 accounts", int(priority_counts.get("P1", 0)))
    with t2:
        st.metric("P2 accounts", int(priority_counts.get("P2", 0)))
    with t3:
        st.metric("P3 accounts", int(priority_counts.get("P3", 0)))
    with t4:
        st.metric("Monitor", int(priority_counts.get("Monitor", 0)))

    st.markdown("#### Accounts requiring attention")
    attention_df = filtered[filtered["priority"].isin(["P1", "P2"])].copy()
    if attention_df.empty:
        st.info("No high-priority accounts in the current filtered view.")
    else:
        attention_preview = attention_df[[
            "account_id", "priority", "segment", "plan_tier", "mrr",
            "churn_probability", "risk_level", "renewal_intent", "renewal_due_days",
            "critical_feedback_count_90d", "stuck_bugs_count",
        ]].head(12).copy()
        attention_preview["mrr"] = attention_preview["mrr"].map(fmt_currency)
        attention_preview["churn_probability"] = attention_preview["churn_probability"].map(fmt_pct)
        st.dataframe(attention_preview, use_container_width=True, hide_index=True)

    st.markdown("#### Full account view")
    table_cols = [
        "priority", "account_id", "segment", "plan_tier", "subscription_type", "mrr",
        "churn_probability", "risk_level", "renewal_intent", "renewal_due_days",
        "tool_activity_score", "feature_adoption_score", "csat_support", "sentiment_csm",
        "unresolved_feedback_count", "roadmap_alignment_score", "stuck_bugs_count", "data_accuracy_bug_flag",
    ]
    display_df = filtered[table_cols].copy()
    priority_order = {"P1": 1, "P2": 2, "P3": 3, "Monitor": 4}
    display_df["priority_sort"] = display_df["priority"].map(priority_order)
    display_df = display_df.sort_values(by=["priority_sort", "churn_probability"], ascending=[True, False]).drop(columns=["priority_sort"])
    display_df["mrr"] = display_df["mrr"].map(fmt_currency)
    display_df["churn_probability"] = display_df["churn_probability"].map(fmt_pct)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab3:
    section_header("Account Detail", "Drill down into one account to understand its risk, drivers, and recommended action.")

    selected_id = st.selectbox("Select account", filtered["account_id"].tolist())
    row = filtered.loc[filtered["account_id"] == selected_id].iloc[0]

    chips = build_risk_chips(row)
    chips_html = "".join([f'<span class="account-chip">{html.escape(chip)}</span>' for chip in chips])
    dominant_theme = dominant_risk_theme(row)
    chips_html += f'<span class="account-chip">{html.escape(dominant_theme)}</span>'

    st.markdown(
        f"""
        <div class="account-header-card">
            <div>{priority_pill(row['priority'])} {risk_badge(row['risk_level'])}</div>
            <div class="account-header-title" style="margin-top:12px;">{html.escape(str(row['account_id']))}</div>
            <div class="account-header-sub">{html.escape(str(row['segment']))} · {html.escape(str(row['plan_tier']))} · {html.escape(str(row['subscription_type']))} · {html.escape(str(row['subscription_status']))}</div>
            {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Churn probability", fmt_pct(row["churn_probability"]), "Predicted probability that this account churns.")
    with c2:
        metric_card("Risk level", row["risk_level"], "Based on probability thresholds: High ≥ 70%, Medium ≥ 40%.")
    with c3:
        metric_card("Renewal due", f"{int(row['renewal_due_days'])} days", "Time until next renewal moment.")
    with c4:
        metric_card("MRR", fmt_currency(row["mrr"]), "Monthly recurring revenue for this account.")

    risk_class = "risk-high" if row["risk_level"] == "High" else "risk-medium" if row["risk_level"] == "Medium" else "risk-low"
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown(
            f"""
            <div class="detail-card {risk_class}">
                <div class="detail-card-title">Why this account has this risk level</div>
                <div class="detail-card-body">{html.escape(build_explanation(row))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="action-panel">
                <div class="action-panel-title">AI Recommendation</div>
                <div class="detail-card-body">{build_recommendation(row, float(row['churn_probability']))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_header("Core health signals", "Key account-level indicators used by the model and by the explanation layer.")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="signal-card"><div class="signal-label">Product activity</div><div class="signal-value">{row["tool_activity_score"]:.0f}</div><div class="signal-note">Recent usage intensity across the product.</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="signal-card"><div class="signal-label">Feature adoption</div><div class="signal-value">{row["feature_adoption_score"]:.0f}</div><div class="signal-note">How deeply the account uses relevant capabilities.</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="signal-card"><div class="signal-label">Support CSAT</div><div class="signal-value">{row["csat_support"]:.1f}</div><div class="signal-note">Customer satisfaction with support interactions.</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="signal-card"><div class="signal-label">CSM sentiment</div><div class="signal-value">{row["sentiment_csm"]:.2f}</div><div class="signal-note">Sentiment estimate from Customer Success interactions.</div></div>', unsafe_allow_html=True)

    section_header("Product feedback & bug resolution", "V2 signals that capture unmet product needs and product reliability issues.")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(f'<div class="signal-card"><div class="signal-label">Unresolved feedback</div><div class="signal-value">{int(row["unresolved_feedback_count"])}</div><div class="signal-note">Open product feedback items linked to this account.</div></div>', unsafe_allow_html=True)
    with f2:
        st.markdown(f'<div class="signal-card"><div class="signal-label">Roadmap alignment</div><div class="signal-value">{row["roadmap_alignment_score"]:.0f}</div><div class="signal-note">How well customer needs align with roadmap direction.</div></div>', unsafe_allow_html=True)
    with f3:
        st.markdown(f'<div class="signal-card"><div class="signal-label">Stuck bugs</div><div class="signal-value">{int(row["stuck_bugs_count"])}</div><div class="signal-note">Bugs open beyond threshold or marked as stuck.</div></div>', unsafe_allow_html=True)
    with f4:
        data_bug = "Yes" if int(row["data_accuracy_bug_flag"]) == 1 else "No"
        st.markdown(f'<div class="signal-card"><div class="signal-label">Data accuracy bug</div><div class="signal-value">{data_bug}</div><div class="signal-note">Whether a bug affects data trust or integrity.</div></div>', unsafe_allow_html=True)

    st.markdown("#### Supporting account signals")
    st.dataframe(top_signal_table(row), use_container_width=True, hide_index=True)

    st.markdown("#### Raw account data")
    st.caption("Detailed account-level fields used in the prototype. Values are converted to text to avoid display issues with mixed data types.")
    raw_df = pd.DataFrame({"Field": row.index.astype(str), "Value": [str(v) for v in row.values]})
    st.dataframe(raw_df, use_container_width=True, hide_index=True)

with tab4:
    section_header("Methodology", "A non-technical explanation of how the Copilot calculates risk.")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            """
            <div class="method-card">
                <div class="method-label">Core formula</div>
                <div class="method-title">Single-account churn probability</div>
                <div class="formula-box">P(churn = 1 | account signals)</div>
                <div class="method-body">
                    For each account, the model estimates the probability that the account belongs to the churn class,
                    given its customer profile, usage, renewal, support, sentiment, feedback and bug signals.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="method-card">
                <div class="method-label">Portfolio KPI</div>
                <div class="method-title">Average churn probability</div>
                <div class="formula-box">Σ account churn probabilities / number of accounts</div>
                <div class="method-body">
                    The portfolio KPI is the average of all account-level probabilities in the selected view. It is a risk exposure metric, not a direct count of customers expected to churn.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="method-card">
                <div class="method-label">Risk thresholds</div>
                <div class="method-title">Probability to risk level</div>
                <div class="formula-box">High ≥ 70% · Medium 40–69.9% · Low &lt; 40%</div>
                <div class="method-body">
                    These thresholds help Customer Success teams prioritize work. They can be adjusted later based on real company data, calibration and operational capacity.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Variables included")
    variable_groups = {
        "Account & customer profile": ["segment", "subscription_type", "plan_tier", "tenure_months", "mrr"],
        "Subscription & renewal context": ["bundles_contracted", "extra_users", "renewal_due_days", "subscription_status", "renewed_last_quarter", "renewal_intent", "in_grace_period"],
        "Product usage & adoption": ["login_count_30d", "tool_activity_score", "usage_change_vs_prev_quarter", "feature_adoption_score", "key_reports_used_30d", "days_since_last_login"],
        "Customer Success engagement": ["emails_with_csm_30d", "csm_meetings_last_quarter", "strategic_review_done", "success_plan_active"],
        "Support experience": ["tickets_last_quarter", "reopened_tickets", "avg_resolution_days", "csat_support"],
        "Sentiment": ["sentiment_csm", "sentiment_support"],
        "Product Feedback & Roadmap Fit": ["feedback_count_90d", "feature_requests_count_90d", "data_recalculation_requests_90d", "critical_feedback_count_90d", "unresolved_feedback_count", "avg_unresolved_feedback_age_days", "roadmap_alignment_score", "feedback_sentiment_score", "competitor_mentioned_in_feedback", "feedback_linked_to_renewal"],
        "Product Reliability & Bug Resolution": ["bugs_reported_90d", "critical_bugs_reported_90d", "open_bugs_count", "stuck_bugs_count", "avg_bug_resolution_days", "oldest_open_bug_days", "bug_reopen_count", "data_accuracy_bug_flag", "workaround_available_flag", "bug_sentiment_score", "bug_linked_to_renewal_flag"],
    }

    for group, variables in variable_groups.items():
        with st.expander(group):
            st.write(", ".join([f"`{v}`" for v in variables]))

    st.info(
        "The model does not use account_id as a predictive signal. The churn_label is used during training and validation only, not for live scoring."
    )
