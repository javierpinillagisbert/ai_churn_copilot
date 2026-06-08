
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Churn Early Warning Copilot", layout="wide")

DATA_PATH = "synthetic_saas_churn_dataset.csv"
MODEL_PATH = "best_churn_model.pkl"

# -----------------------------
# Helpers
# -----------------------------
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

def build_explanation(row: pd.Series) -> str:
    reasons = []

    if row["tool_activity_score"] < 35:
        reasons.append("low product activity")
    elif row["tool_activity_score"] < 55:
        reasons.append("moderate product activity")

    if row["feature_adoption_score"] < 40:
        reasons.append("low feature adoption")
    elif row["feature_adoption_score"] < 60:
        reasons.append("partial feature adoption")

    if row["usage_change_vs_prev_quarter"] < -20:
        reasons.append("strong decline in usage vs previous quarter")
    elif row["usage_change_vs_prev_quarter"] < -5:
        reasons.append("recent decline in usage")

    if row["days_since_last_login"] > 20:
        reasons.append("long inactivity period")

    if row["csat_support"] < 3.2:
        reasons.append("low support satisfaction")

    if row["reopened_tickets"] >= 2:
        reasons.append("reopened support tickets")

    if row["sentiment_csm"] < -0.2:
        reasons.append("negative sentiment in CSM interactions")

    if row["renewal_intent"] == "negative":
        reasons.append("negative renewal intent")
    elif row["renewal_intent"] == "neutral":
        reasons.append("uncertain renewal intent")

    if row["renewal_due_days"] < 30 and row["renewed_last_quarter"] == 0:
        reasons.append("renewal approaching without confirmed renewal")

    if not reasons:
        return "The account shows stable health signals with no major short-term churn indicators."

    top_reasons = reasons[:4]
    return "Main risk signals detected: " + ", ".join(top_reasons) + "."

def build_recommendation(row: pd.Series, prob: float) -> str:
    actions = []

    if row["tool_activity_score"] < 35 or row["feature_adoption_score"] < 40:
        actions.append("schedule an adoption review focused on underused features")

    if row["reopened_tickets"] >= 2 or row["csat_support"] < 3.2:
        actions.append("coordinate with Support to review unresolved friction points")

    if row["sentiment_csm"] < -0.2:
        actions.append("prepare a proactive CSM outreach with a clear recovery plan")

    if row["renewal_due_days"] < 30 and row["renewal_intent"] in ["neutral", "negative"]:
        actions.append("prioritize renewal-risk follow-up within the next 48 hours")

    if row["strategic_review_done"] == 0 and row["segment"] != "SMB":
        actions.append("book a strategic business review to reinforce value realization")

    if row["success_plan_active"] == 0 and prob >= 0.40:
        actions.append("create a short-term success plan with milestones")

    if not actions:
        return "Maintain regular follow-up and monitor account health through the normal lifecycle."

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

# -----------------------------
# Main app
# -----------------------------
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

# Sidebar filters
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

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accounts", len(filtered))
col2.metric("Avg churn probability", f"{filtered['churn_probability'].mean():.1%}")
col3.metric("High-risk accounts", int((filtered["risk_level"] == "High").sum()))
retention_rate = ((filtered["renewed_last_quarter"] == 1) | (filtered["subscription_status"] == "active")).mean()
col4.metric("Retention snapshot", f"{retention_rate:.1%}")

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

    st.markdown("**Quick observations**")
    obs = []
    if len(filtered) > 0:
        highest_segment = filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=False).index[0]
        obs.append(f"- Highest average risk segment in the current view: **{highest_segment}**.")
        if (filtered["risk_level"] == "High").sum() > 0:
            obs.append(f"- There are **{int((filtered['risk_level'] == 'High').sum())}** accounts requiring priority review.")
        if filtered["tool_activity_score"].mean() < 50:
            obs.append("- Average activity is below the healthy midpoint, suggesting adoption risk in the filtered portfolio.")
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
        c2.metric("Risk level", row["risk_level"])
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
