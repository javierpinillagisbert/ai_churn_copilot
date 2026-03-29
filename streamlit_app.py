import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Churn Early Warning Copilot", layout="wide")
st.caption("Interactive churn intelligence workspace for Customer Success teams.")

DATA_PATH = "synthetic_saas_churn_dataset.csv"
MODEL_PATH = "best_churn_model.pkl"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def safe_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return preds.astype(float)

def risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"

def risk_badge(level: str) -> str:
    colors = {
        "High": "#fee2e2",
        "Medium": "#fef3c7",
        "Low": "#dcfce7",
    }
    text_colors = {
        "High": "#b91c1c",
        "Medium": "#b45309",
        "Low": "#166534",
    }
    return f'''
    <div style="
        display:inline-block;
        padding:6px 12px;
        border-radius:999px;
        background:{colors.get(level, '#e5e7eb')};
        color:{text_colors.get(level, '#111827')};
        font-weight:700;
        font-size:14px;">
        {level} risk
    </div>
    '''

def build_explanation(row: pd.Series) -> str:
    reasons = []

    if row["tool_activity_score"] < 35:
        reasons.append("product activity has dropped to a low level")
    elif row["tool_activity_score"] < 55:
        reasons.append("product activity is below the healthy midpoint")

    if row["feature_adoption_score"] < 40:
        reasons.append("adoption of key features is weak")
    elif row["feature_adoption_score"] < 60:
        reasons.append("feature adoption is still only partial")

    if row["usage_change_vs_prev_quarter"] < -20:
        reasons.append("usage has fallen sharply versus the previous quarter")
    elif row["usage_change_vs_prev_quarter"] < -5:
        reasons.append("usage is trending down compared with the previous quarter")

    if row["days_since_last_login"] > 20:
        reasons.append("the account has been inactive for a relatively long period")

    if row["csat_support"] < 3.2:
        reasons.append("support satisfaction is below the expected level")

    if row["reopened_tickets"] >= 2:
        reasons.append("multiple support issues were reopened")

    if row["sentiment_csm"] < -0.2:
        reasons.append("recent interactions with the CSM show negative sentiment")

    if row["renewal_intent"] == "negative":
        reasons.append("renewal intent is currently negative")
    elif row["renewal_intent"] == "neutral":
        reasons.append("renewal intent is still uncertain")

    if row["renewal_due_days"] < 30 and row["renewed_last_quarter"] == 0:
        reasons.append("renewal is approaching without a confirmed renewal signal")

    if not reasons:
        return (
            "This account currently looks stable. Product usage is holding up, the relationship appears healthy, "
            "and there are no major short-term churn signals."
        )

    intro = "This account appears at risk because "
    if len(reasons) == 1:
        return intro + reasons[0] + "."
    if len(reasons) == 2:
        return intro + reasons[0] + " and " + reasons[1] + "."
    if len(reasons) == 3:
        return intro + ", ".join(reasons[:2]) + ", and " + reasons[2] + "."
    return intro + ", ".join(reasons[:3]) + ", and " + reasons[3] + "."

def build_recommendation(row: pd.Series, prob: float) -> str:
    actions = []

    if row["tool_activity_score"] < 35 or row["feature_adoption_score"] < 40:
        actions.append("schedule an adoption-focused session to recover usage of underused features")

    if row["reopened_tickets"] >= 2 or row["csat_support"] < 3.2:
        actions.append("review unresolved support friction jointly with the Support team before the next customer touchpoint")

    if row["sentiment_csm"] < -0.2:
        actions.append("prepare a proactive outreach from the CSM with a clear recovery narrative and next steps")

    if row["renewal_due_days"] < 30 and row["renewal_intent"] in ["neutral", "negative"]:
        actions.append("prioritise a renewal-risk follow-up in the next 48 hours")

    if row["strategic_review_done"] == 0 and row["segment"] != "SMB":
        actions.append("book a strategic business review to reconnect product usage with business value")

    if row["success_plan_active"] == 0 and prob >= 0.40:
        actions.append("create a short-term success plan with concrete milestones and ownership")

    if not actions:
        return "Maintain normal follow-up, keep monitoring the account, and reassess if engagement starts to soften."

    return "Recommended next action: " + actions[0] + "."

def top_signal_table(row: pd.Series) -> pd.DataFrame:
    signals = [
        ("Tool activity score", float(row["tool_activity_score"])),
        ("Feature adoption score", float(row["feature_adoption_score"])),
        ("Usage change vs previous quarter", float(row["usage_change_vs_prev_quarter"])),
        ("Days since last login", float(row["days_since_last_login"])),
        ("CSAT support", float(row["csat_support"])),
        ("Reopened tickets", float(row["reopened_tickets"])),
        ("Sentiment with CSM", float(row["sentiment_csm"])),
    ]
    return pd.DataFrame(signals, columns=["Signal", "Value"])

def build_risk_chips(row: pd.Series):
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

    return chips[:5]

def portfolio_driver_summary(df: pd.DataFrame) -> pd.DataFrame:
    signals = {
        "Low product activity": (df["tool_activity_score"] < 35).mean(),
        "Low feature adoption": (df["feature_adoption_score"] < 40).mean(),
        "Negative CSM sentiment": (df["sentiment_csm"] < -0.2).mean(),
        "Low support CSAT": (df["csat_support"] < 3.2).mean(),
        "Reopened tickets": (df["reopened_tickets"] >= 2).mean(),
        "Renewal risk": ((df["renewal_due_days"] < 30) & (df["renewal_intent"].isin(["neutral", "negative"]))).mean(),
    }
    out = pd.DataFrame({
        "Driver": list(signals.keys()),
        "Share of accounts": [round(v * 100, 1) for v in signals.values()]
    }).sort_values("Share of accounts", ascending=False)
    return out

def compute_portfolio_health_score(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0

    avg_churn = df["churn_probability"].mean()
    high_risk_share = (df["risk_level"] == "High").mean()
    activity = df["tool_activity_score"].mean() / 100
    adoption = df["feature_adoption_score"].mean() / 100
    csat = df["csat_support"].mean() / 5
    sentiment = (df["sentiment_csm"].mean() + 1) / 2  # convierte de -1..1 a 0..1

    score = (
        (1 - avg_churn) * 0.35 +
        (1 - high_risk_share) * 0.20 +
        activity * 0.15 +
        adoption * 0.15 +
        csat * 0.10 +
        sentiment * 0.05
    ) * 100

    return round(score, 1)

def portfolio_health_status(score: float):
    if score >= 75:
        return "Healthy", "healthy"
    if score >= 55:
        return "Watchlist", "watchlist"
    return "At Risk", "atrisk"

st.markdown("""
<style>
.priority-box {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    min-height: 210px;
}
.priority-label {
    display: inline-block;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 10px;
}
.priority-title {
    font-size: 20px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 8px;
}
.priority-sub {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 16px;
}
.priority-metric {
    font-size: 13px;
    color: #475569;
    margin-bottom: 6px;
}
.priority-metric strong {
    color: #111827;
}
.priority-highlight {
    margin-top: 14px;
    padding: 10px 12px;
    border-radius: 12px;
    background: #f8fafc;
    color: #334155;
    font-size: 13px;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.main-hero {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    border: 1px solid #e5e7eb;
    border-radius: 22px;
    padding: 28px 30px 22px 30px;
    margin-bottom: 20px;
}
.hero-eyebrow {
    color: #6366f1;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.hero-title {
    font-size: 42px;
    font-weight: 800;
    line-height: 1.05;
    color: #111827;
    margin-bottom: 12px;
}
.hero-subtitle {
    font-size: 17px;
    color: #475569;
    max-width: 900px;
}
.kpi-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.kpi-label {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 800;
    color: #111827;
}
.section-title {
    font-size: 28px;
    font-weight: 800;
    color: #1f2937;
    margin-bottom: 4px;
}
.section-subtitle {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 16px;
}
.insight-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    min-height: 170px;
}
.insight-label {
    display: inline-block;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 10px;
}
.insight-title {
    font-size: 18px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 8px;
    line-height: 1.2;
}
.insight-value {
    font-size: 30px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 10px;
}
.insight-body {
    color: #475569;
    font-size: 14px;
    line-height: 1.5;
}
.insight-summary-box {
    margin-top: 14px;
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 16px 18px;
    color: #334155;
    font-size: 14px;
    line-height: 1.6;
}
.account-header-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    margin-bottom: 16px;
}
.account-header-title {
    font-size: 28px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 8px;
}
.account-header-sub {
    color: #64748b;
    font-size: 15px;
    margin-bottom: 12px;
}
.account-chip {
    display: inline-block;
    background: #eef2ff;
    color: #4338ca;
    border-radius: 999px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 700;
    margin-right: 8px;
    margin-bottom: 8px;
}
.detail-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    margin-bottom: 14px;
}
.detail-card-title {
    font-size: 16px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 10px;
}
.detail-card-body {
    color: #475569;
    font-size: 14px;
    line-height: 1.6;
}
.signal-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 16px;
    text-align: left;
    min-height: 120px;
}
.signal-label {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 6px;
}
.signal-value {
    font-size: 26px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 8px;
}
.signal-note {
    font-size: 13px;
    color: #475569;
    line-height: 1.4;
}
.action-panel {
    background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);
    border: 1px solid #dbeafe;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 14px;
}
.action-panel-title {
    font-size: 16px;
    font-weight: 800;
    color: #1d4ed8;
    margin-bottom: 10px;
}
.raw-data-note {
    color: #64748b;
    font-size: 13px;
    margin-bottom: 10px;
}
.health-score-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e5e7eb;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    margin-bottom: 18px;
}
.health-score-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 10px;
}
.health-score-value {
    font-size: 42px;
    font-weight: 800;
    color: #111827;
    line-height: 1;
    margin-bottom: 10px;
}
.health-score-status {
    display: inline-block;
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 12px;
}
.health-score-status.healthy {
    background: #dcfce7;
    color: #166534;
}
.health-score-status.watchlist {
    background: #fef3c7;
    color: #b45309;
}
.health-score-status.atrisk {
    background: #fee2e2;
    color: #b91c1c;
}
.health-score-body {
    color: #475569;
    font-size: 14px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-hero">
    <div class="hero-eyebrow">Retention Intelligence</div>
    <div class="hero-title">AI Churn Early Warning Copilot</div>
    <div class="hero-subtitle">
        A decision-support workspace for Customer Success teams to identify at-risk accounts,
        understand the main churn drivers, and prioritise the next best action with confidence.
    </div>
</div>
""", unsafe_allow_html=True)

missing = []
if not os.path.exists(DATA_PATH):
    missing.append(DATA_PATH)
if not os.path.exists(MODEL_PATH):
    missing.append(MODEL_PATH)

if missing:
    st.error(
        "Missing required file(s): " + ", ".join(missing) +
        ". Put the dataset CSV and the trained model PKL in the same folder as this app."
    )
    st.stop()

df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

feature_cols = [c for c in df.columns if c not in ["account_id", "churn_label"]]
X = df[feature_cols].copy()
df["churn_probability"] = safe_predict_proba(model, X)
df["risk_level"] = df["churn_probability"].apply(risk_level)
df = df.sort_values("churn_probability", ascending=False).reset_index(drop=True)

st.sidebar.header("Filters")
segments = ["All"] + sorted(df["segment"].dropna().unique().tolist())
plans = ["All"] + sorted(df["plan_tier"].dropna().unique().tolist())
risk_levels = ["All", "Low", "Medium", "High"]

segment_filter = st.sidebar.selectbox("Segment", segments)
plan_filter = st.sidebar.selectbox("Plan tier", plans)
risk_filter = st.sidebar.selectbox("Risk level", risk_levels)

filtered = df.copy()
if segment_filter != "All":
    filtered = filtered[filtered["segment"] == segment_filter]
if plan_filter != "All":
    filtered = filtered[filtered["plan_tier"] == plan_filter]
if risk_filter != "All":
    filtered = filtered[filtered["risk_level"] == risk_filter]

filtered = filtered.sort_values("churn_probability", ascending=False).reset_index(drop=True)

retention_rate = ((filtered["renewed_last_quarter"] == 1) | (filtered["subscription_status"] == "active")).mean()
portfolio_health_score = compute_portfolio_health_score(filtered)
portfolio_health_label, portfolio_health_class = portfolio_health_status(portfolio_health_score)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Accounts</div>
        <div class="kpi-value">{len(filtered)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Average churn probability</div>
        <div class="kpi-value">{filtered['churn_probability'].mean():.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">High-risk accounts</div>
        <div class="kpi-value">{int((filtered["risk_level"] == "High").sum())}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Retention snapshot</div>
        <div class="kpi-value">{retention_rate:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f'''
<div class="health-score-card">
    <div class="health-score-label">Portfolio health</div>
    <div class="health-score-value">{portfolio_health_score} / 100</div>
    <div class="health-score-status {portfolio_health_class}">{portfolio_health_label}</div>
    <div class="health-score-body">
        This score summarises the overall health of the current portfolio view using average churn probability,
        share of high-risk accounts, product activity, feature adoption, support satisfaction, and CSM sentiment.
    </div>
</div>
''', unsafe_allow_html=True)

st.divider()

st.markdown('<div class="section-title">Top 3 priority accounts</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Accounts that currently deserve the fastest review based on predicted churn probability.</div>',
    unsafe_allow_html=True
)

top3 = filtered.head(3)

if top3.empty:
    st.info("No accounts available with the selected filters.")
else:
    pcols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with pcols[i]:
            highlight_text = build_explanation(row)

            card_html = (
                f'<div class="priority-box">'
                f'<div class="priority-label">Priority account</div>'
                f'<div class="priority-title">{row["account_id"]}</div>'
                f'<div class="priority-sub">{row["segment"]} · {row["plan_tier"]} · {row["subscription_type"]}</div>'
                f'<div class="priority-metric"><strong>Churn probability:</strong> {row["churn_probability"]:.1%}</div>'
                f'<div class="priority-metric"><strong>Renewal intent:</strong> {row["renewal_intent"]}</div>'
                f'<div class="priority-metric"><strong>Renewal due:</strong> {int(row["renewal_due_days"])} days</div>'
                f'<div class="priority-highlight">{highlight_text}</div>'
                f'</div>'
            )

            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown(risk_badge(row["risk_level"]), unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["Portfolio Dashboard", "Account Table", "Account Detail"])

with tab1:
    st.subheader("Portfolio Dashboard")
    c1, c2 = st.columns(2)

    with c1:
        risk_counts = filtered["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        st.bar_chart(risk_counts)

    with c2:
        churn_by_segment = filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=False)
        st.bar_chart(churn_by_segment)

    st.markdown('<div class="section-title">Top risk drivers in portfolio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">A quick reading of the most common risk patterns in the current portfolio view.</div>',
    unsafe_allow_html=True
)

drivers = portfolio_driver_summary(filtered)

top_driver = drivers.iloc[0]["Driver"] if len(drivers) > 0 else "No driver available"
top_share = drivers.iloc[0]["Share of accounts"] if len(drivers) > 0 else 0

second_driver = drivers.iloc[1]["Driver"] if len(drivers) > 1 else "No second driver"
second_share = drivers.iloc[1]["Share of accounts"] if len(drivers) > 1 else 0

renewal_risk_count = int(((filtered["renewal_due_days"] < 30) & (filtered["renewal_intent"].isin(["neutral", "negative"]))).sum())

highest_segment = (
    filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=False).index[0]
    if len(filtered) > 0 else "N/A"
)

ic1, ic2, ic3 = st.columns(3)

with ic1:
    st.markdown(f'''
    <div class="insight-card">
        <div class="insight-label">Primary risk driver</div>
        <div class="insight-title">{top_driver}</div>
        <div class="insight-value">{top_share}%</div>
        <div class="insight-body">
            This is the most common risk pattern in the current portfolio view and should be addressed first at scale.
        </div>
    </div>
    ''', unsafe_allow_html=True)

with ic2:
    st.markdown(f'''
    <div class="insight-card">
        <div class="insight-label">Highest-risk segment</div>
        <div class="insight-title">{highest_segment}</div>
        <div class="insight-value">{filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=False).iloc[0]:.1%}</div>
        <div class="insight-body">
            This segment currently shows the highest average churn probability and deserves closer portfolio monitoring.
        </div>
    </div>
    ''', unsafe_allow_html=True)

with ic3:
    st.markdown(f'''
    <div class="insight-card">
        <div class="insight-label">Renewal pressure</div>
        <div class="insight-title">Accounts needing follow-up</div>
        <div class="insight-value">{renewal_risk_count}</div>
        <div class="insight-body">
            These accounts are approaching renewal with uncertain or negative intent and may require immediate action.
        </div>
    </div>
    ''', unsafe_allow_html=True)

summary_text = (
    f"In this portfolio view, the main structural weakness is **{top_driver}**, affecting **{top_share}%** of accounts. "
    f"The second most common signal is **{second_driver}** at **{second_share}%**. "
    f"The most exposed segment is **{highest_segment}**, while **{renewal_risk_count}** accounts may require short-term renewal follow-up."
)

st.markdown(
    f'<div class="insight-summary-box">{summary_text}</div>',
    unsafe_allow_html=True
)

st.markdown("### Quick observations")
obs = []
if len(filtered) > 0:
        highest_segment = filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=False).index[0]
        obs.append(f"- Highest average risk segment in the current view: **{highest_segment}**.")
        if (filtered["risk_level"] == "High").sum() > 0:
            obs.append(f"- There are **{int((filtered['risk_level'] == 'High').sum())}** accounts requiring priority review.")
        if filtered["tool_activity_score"].mean() < 50:
            obs.append("- Average product activity is below the healthy midpoint, suggesting adoption risk in the portfolio.")
        if filtered["sentiment_csm"].mean() < 0:
            obs.append("- The average CSM sentiment is below neutral, which may indicate relationship friction across the portfolio.")
st.write("\n".join(obs) if obs else "No observations available.")

with tab2:
    st.subheader("Account Table")
    table_cols = [
        "account_id", "segment", "plan_tier", "subscription_type",
        "churn_probability", "risk_level", "renewal_intent", "renewal_due_days",
        "tool_activity_score", "feature_adoption_score", "csat_support", "sentiment_csm"
    ]
    display_df = filtered[table_cols].copy()
    display_df["churn_probability"] = display_df["churn_probability"].map(lambda x: f"{x:.1%}")
    st.caption("Accounts are ordered from highest to lowest churn probability.")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Account Detail")

    if filtered.empty:
        st.info("No accounts available with the selected filters.")
    else:
        selected_id = st.selectbox("Select account", filtered["account_id"].tolist())
        row = filtered.loc[filtered["account_id"] == selected_id].iloc[0]

        # Header
        chips = build_risk_chips(row)
        chips_html = "".join([f'<span class="account-chip">{chip}</span>' for chip in chips])

        st.markdown(f'''
        <div class="account-header-card">
            <div class="account-header-title">{row["account_id"]}</div>
            <div class="account-header-sub">{row["segment"]} · {row["plan_tier"]} · {row["subscription_type"]} · {row["subscription_status"]}</div>
            {chips_html}
        </div>
        ''', unsafe_allow_html=True)

        # Top metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Churn probability", f'{row["churn_probability"]:.1%}')
        with c2:
            st.markdown(risk_badge(row["risk_level"]), unsafe_allow_html=True)
        with c3:
            st.metric("Renewal due in", f'{int(row["renewal_due_days"])} days')

        # Explanation and action
        left, right = st.columns([1.1, 1])

        with left:
            st.markdown(f'''
            <div class="detail-card">
                <div class="detail-card-title">Why this account is at risk</div>
                <div class="detail-card-body">{build_explanation(row)}</div>
            </div>
            ''', unsafe_allow_html=True)

        with right:
            st.markdown(f'''
            <div class="action-panel">
                <div class="action-panel-title">Next best action</div>
                <div class="detail-card-body">{build_recommendation(row, float(row["churn_probability"]))}</div>
            </div>
            ''', unsafe_allow_html=True)

        # Health signals
        st.markdown('<div class="section-title">Health signals</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Core indicators used to understand account health and prioritise intervention.</div>',
            unsafe_allow_html=True
        )

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.markdown(f'''
            <div class="signal-card">
                <div class="signal-label">Product activity</div>
                <div class="signal-value">{row["tool_activity_score"]:.0f}</div>
                <div class="signal-note">Measures recent usage intensity across the product.</div>
            </div>
            ''', unsafe_allow_html=True)

        with s2:
            st.markdown(f'''
            <div class="signal-card">
                <div class="signal-label">Feature adoption</div>
                <div class="signal-value">{row["feature_adoption_score"]:.0f}</div>
                <div class="signal-note">Reflects how deeply the account uses relevant capabilities.</div>
            </div>
            ''', unsafe_allow_html=True)

        with s3:
            st.markdown(f'''
            <div class="signal-card">
                <div class="signal-label">Support CSAT</div>
                <div class="signal-value">{row["csat_support"]:.1f}</div>
                <div class="signal-note">Customer satisfaction with support interactions.</div>
            </div>
            ''', unsafe_allow_html=True)

        with s4:
            st.markdown(f'''
            <div class="signal-card">
                <div class="signal-label">CSM sentiment</div>
                <div class="signal-value">{row["sentiment_csm"]:.2f}</div>
                <div class="signal-note">Sentiment estimate based on interactions with Customer Success.</div>
            </div>
            ''', unsafe_allow_html=True)

        # Additional signal table
        st.markdown("### Supporting account signals")
        st.dataframe(top_signal_table(row), use_container_width=True, hide_index=True)

        # Raw data
        st.markdown("### Raw account data")
        st.caption("Detailed account-level fields used in the prototype.")
        raw_df = pd.DataFrame({
            "Field": row.index,
            "Value": row.values
        })
        st.dataframe(raw_df, use_container_width=True, hide_index=True)
