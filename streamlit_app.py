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

st.markdown("""
<style>
.priority-box {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 12px;
}
.priority-title {
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 6px;
}
.priority-sub {
    color: #475569;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("AI Churn Early Warning Copilot for SaaS Customer Health")
st.caption("MVP in Streamlit using a synthetic SaaS dataset and a saved churn model.")

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

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accounts", len(filtered))
col2.metric("Avg churn probability", f"{filtered['churn_probability'].mean():.1%}")
col3.metric("High-risk accounts", int((filtered["risk_level"] == "High").sum()))
retention_rate = ((filtered["renewed_last_quarter"] == 1) | (filtered["subscription_status"] == "active")).mean()
col4.metric("Retention snapshot", f"{retention_rate:.1%}")

st.divider()

st.subheader("Top 3 priority accounts")
top3 = filtered.head(3)

if top3.empty:
    st.info("No accounts available with the selected filters.")
else:
    pcols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with pcols[i]:
            st.markdown(f'''
            <div class="priority-box">
                <div class="priority-title">{row['account_id']}</div>
                <div class="priority-sub">{row['segment']} · {row['plan_tier']} · {row['subscription_type']}</div>
                <br>
                <div><strong>Churn probability:</strong> {row['churn_probability']:.1%}</div>
                <div><strong>Renewal intent:</strong> {row['renewal_intent']}</div>
                <div><strong>Renewal due:</strong> {int(row['renewal_due_days'])} days</div>
            </div>
            ''', unsafe_allow_html=True)
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

    st.markdown("### Top risk drivers in portfolio")
    drivers = portfolio_driver_summary(filtered)
    d1, d2 = st.columns([1.2, 1])
    with d1:
        st.dataframe(drivers, use_container_width=True, hide_index=True)
    with d2:
        top_driver = drivers.iloc[0]["Driver"]
        top_share = drivers.iloc[0]["Share of accounts"]
        summary_text = f"The most common risk driver in the current portfolio view is **{top_driver}**, affecting **{top_share}%** of accounts."
        if len(drivers) > 1:
            second_driver = drivers.iloc[1]["Driver"]
            second_share = drivers.iloc[1]["Share of accounts"]
            summary_text += f" The second most common signal is **{second_driver}** at **{second_share}%**."
        st.info(summary_text)

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

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn probability", f"{row['churn_probability']:.1%}")
        c2.markdown(risk_badge(row["risk_level"]), unsafe_allow_html=True)
        c3.metric("Renewal due in", f"{int(row['renewal_due_days'])} days")

        st.markdown("### Account context")
        context_cols = st.columns(4)
        context_cols[0].write(f"**Segment:** {row['segment']}")
        context_cols[1].write(f"**Plan:** {row['plan_tier']}")
        context_cols[2].write(f"**Subscription:** {row['subscription_type']}")
        context_cols[3].write(f"**Status:** {row['subscription_status']}")

        st.markdown("### Why this account is at risk")
        st.info(build_explanation(row))

        st.markdown("### Recommended action")
        st.success(build_recommendation(row, float(row["churn_probability"])))

        st.markdown("### Key signals")
        st.dataframe(top_signal_table(row), use_container_width=True, hide_index=True)

        st.markdown("### Raw account data")
        raw_df = pd.DataFrame({
            "Field": row.index,
            "Value": row.values
        })
        st.dataframe(raw_df, use_container_width=True, hide_index=True)
