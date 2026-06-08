
import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Semrush Retention Intelligence Copilot V3",
    page_icon="📈",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic_semrush_churn_dataset_v3.csv"
MODEL_PATH = BASE_DIR / "best_churn_model_semrush_v3.pkl"

EXCLUDED_COLUMNS = ["account_id", "account_name", "churn_label", "synthetic_true_churn_probability"]


def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --navy: #071427;
            --blue: #2563EB;
            --teal: #0F766E;
            --purple: #6D28D9;
            --green: #16A34A;
            --amber: #D97706;
            --red: #DC2626;
            --slate: #475569;
            --muted: #64748B;
            --border: #E2E8F0;
            --bg: #F8FAFC;
            --card: #FFFFFF;
        }
        .stApp { background: linear-gradient(180deg, #F8FAFC 0%, #EEF6FF 100%); color: var(--navy); }
        h1, h2, h3 { color: var(--navy); letter-spacing: -0.03em; }
        section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--border); }
        .hero {
            background: radial-gradient(circle at 100% 0%, rgba(15,118,110,.30), transparent 26%),
                        linear-gradient(135deg, #071427 0%, #123C69 58%, #0F766E 100%);
            border-radius: 28px; padding: 34px 38px; color: white;
            box-shadow: 0 18px 50px rgba(7, 20, 39, .22); margin-bottom: 26px;
        }
        .hero-eyebrow { display:inline-block; font-size: 12px; text-transform: uppercase; letter-spacing:.10em; font-weight:800; padding: 7px 12px; border-radius:999px; background: rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.22); margin-bottom:16px; }
        .hero-title { font-size: 44px; font-weight: 900; line-height:1.04; margin-bottom: 10px; }
        .hero-sub { font-size: 17px; line-height:1.55; opacity:.90; max-width: 1020px; }
        .mini-pill { display:inline-block; padding:5px 9px; border-radius:999px; background:#EFF6FF; color:#1D4ED8; font-weight:800; font-size:12px; margin:3px; }
        .section-head { margin: 30px 0 14px; }
        .section-title { font-size: 24px; font-weight: 900; color: var(--navy); display:flex; align-items:center; gap:10px; }
        .section-title::before { content:""; display:inline-block; width:12px; height:12px; border-radius:999px; background:linear-gradient(135deg,var(--blue),var(--teal)); }
        .section-sub { color: var(--muted); font-size: 14px; margin-top: 3px; }
        .metric-card, .panel-card, .playbook-card, .signal-card {
            background:white; border:1px solid var(--border); border-radius:22px;
            box-shadow:0 10px 28px rgba(15,23,42,.055); padding:20px;
        }
        .metric-label { color: var(--muted); font-size: 12px; font-weight: 800; text-transform:uppercase; letter-spacing:.06em; margin-bottom:8px; }
        .metric-value { color: var(--navy); font-size: 31px; font-weight: 900; line-height:1; margin-bottom: 8px; }
        .metric-note { color: var(--muted); font-size:13px; line-height:1.35; }
        .risk-badge { display:inline-block; border-radius:999px; padding:7px 12px; font-size:13px; font-weight:900; }
        .risk-high { background:#FEE2E2; color:#991B1B; border:1px solid #FECACA; }
        .risk-medium { background:#FEF3C7; color:#92400E; border:1px solid #FDE68A; }
        .risk-low { background:#DCFCE7; color:#166534; border:1px solid #BBF7D0; }
        .driver-chip { display:inline-block; background:#F1F5F9; color:#334155; padding:6px 10px; margin:3px 4px 3px 0; border-radius:999px; font-size:12px; font-weight:800; }
        .cta-box { background:linear-gradient(135deg,#ECFEFF,#F8FAFC); border:1px solid #BAE6FD; border-radius:22px; padding:20px; box-shadow:0 10px 28px rgba(15,23,42,.05); }
        .cta-title { color:#0E7490; font-weight:900; font-size:18px; margin-bottom:8px; }
        .cta-body { color:#334155; font-size:14px; line-height:1.55; }
        .playbook-grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:14px; }
        .playbook-label { color: var(--muted); font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.07em; margin-bottom:8px; }
        .playbook-title { color: var(--navy); font-size:16px; font-weight:900; margin-bottom:6px; }
        .playbook-body { color:#475569; font-size:13px; line-height:1.45; }
        .bar-wrap { background:#E2E8F0; border-radius:999px; height:12px; overflow:hidden; margin:6px 0 12px; }
        .bar-fill { height:12px; border-radius:999px; background:linear-gradient(90deg,#0EA5E9,#0F766E); }
        .bar-fill.warn { background:linear-gradient(90deg,#F59E0B,#DC2626); }
        .small-muted { color: var(--muted); font-size:12px; }
        .method-card { background:#FFFFFF; border:1px solid var(--border); border-radius:18px; padding:18px; margin-bottom:12px; }
        div[data-testid="stDataFrame"] { border-radius:18px; overflow:hidden; border:1px solid var(--border); }

        .story-grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:14px; margin: 10px 0 18px; }
        .story-card { background:white; border:1px solid var(--border); border-radius:22px; padding:19px; box-shadow:0 10px 28px rgba(15,23,42,.055); min-height:145px; }
        .story-kicker { color:var(--muted); font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.07em; margin-bottom:8px; }
        .story-headline { color:var(--navy); font-size:20px; font-weight:900; line-height:1.18; margin-bottom:8px; }
        .story-body { color:#475569; font-size:13px; line-height:1.45; }
        .story-bar-list { background:white; border:1px solid var(--border); border-radius:22px; padding:20px; box-shadow:0 10px 28px rgba(15,23,42,.055); margin-bottom:20px; }
        .story-row { display:grid; grid-template-columns: 230px 1fr 76px; gap:16px; align-items:center; padding:10px 0; border-bottom:1px solid #EEF2F7; }
        .story-row:last-child { border-bottom:none; }
        .story-name { color:#0F172A; font-size:14px; font-weight:850; }
        .story-value { color:#334155; font-size:14px; font-weight:900; text-align:right; }
        .story-track { height:12px; border-radius:999px; background:#E2E8F0; overflow:hidden; }
        .story-fill { height:12px; border-radius:999px; background:linear-gradient(90deg,#2563EB,#0F766E); }
        .story-fill.hot { background:linear-gradient(90deg,#F59E0B,#DC2626); }
        .story-fill.medium { background:linear-gradient(90deg,#38BDF8,#2563EB); }
        .story-note { background:linear-gradient(135deg,#ECFEFF,#F8FAFC); border:1px solid #BAE6FD; border-radius:20px; padding:16px 18px; color:#334155; font-size:14px; line-height:1.55; margin-top:12px; }
        .story-callout { background:#F8FAFC; border:1px solid var(--border); border-radius:18px; padding:14px 16px; color:#475569; font-size:13px; line-height:1.45; margin-top:10px; }

        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=""):
    st.markdown(f'<div class="section-head"><div class="section-title">{html.escape(title)}</div><div class="section-sub">{html.escape(subtitle)}</div></div>', unsafe_allow_html=True)


def metric_card(label, value, note=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{html.escape(str(label))}</div><div class="metric-value">{html.escape(str(value))}</div><div class="metric-note">{html.escape(str(note))}</div></div>', unsafe_allow_html=True)


def risk_level(prob):
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def risk_badge(level):
    css = "risk-high" if level == "High" else "risk-medium" if level == "Medium" else "risk-low"
    return f'<span class="risk-badge {css}">{html.escape(level)} risk</span>'


def render_static_bar_chart(series, title, y_label="", as_percent=False):
    """Render a static Matplotlib bar chart so the dashboard does not zoom/resize on scroll."""
    clean = series.dropna()
    labels = [str(x) for x in clean.index]
    values = clean.values.astype(float)

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(labels, values, color="#2563EB", width=0.58)

    ax.set_title(title, fontsize=11, fontweight="bold", color="#071427", pad=14)
    ax.set_ylabel(y_label, fontsize=9, color="#475569")
    ax.tick_params(axis="x", labelsize=9, colors="#475569", rotation=0)
    ax.tick_params(axis="y", labelsize=8, colors="#64748B")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    if len(values) > 0:
        ymax = max(values) * 1.25 if max(values) > 0 else 1
        ax.set_ylim(0, ymax)
        for bar, value in zip(bars, values):
            label = f"{value:.1%}" if as_percent else f"{int(value):,}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#334155",
                fontweight="bold",
            )

    if as_percent:
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True, clear_figure=True)


def render_static_horizontal_bar_chart(series, title, x_label="", as_percent=False):
    """Render a static Matplotlib horizontal bar chart so the dashboard remains fixed while scrolling."""
    clean = series.dropna().sort_values(ascending=True)
    labels = [str(x) for x in clean.index]
    values = clean.values.astype(float)

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.barh(labels, values, color="#2563EB", height=0.56)

    ax.set_title(title, fontsize=11, fontweight="bold", color="#071427", pad=14)
    ax.set_xlabel(x_label, fontsize=9, color="#475569")
    ax.tick_params(axis="x", labelsize=8, colors="#64748B")
    ax.tick_params(axis="y", labelsize=9, colors="#475569")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    if len(values) > 0:
        xmax = max(values) * 1.25 if max(values) > 0 else 1
        ax.set_xlim(0, xmax)
        for bar, value in zip(bars, values):
            label = f"{value:.1%}" if as_percent else f"{int(value):,}"
            ax.text(
                value + xmax * 0.018,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=8,
                color="#334155",
                fontweight="bold",
            )

    if as_percent:
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True, clear_figure=True)


def driver_story_copy(family_name: str, score: float) -> str:
    """Plain-English explanations for portfolio-level risk families."""
    name = str(family_name)
    if name == "Renewal & commercial":
        return "Commercial pressure is the strongest theme: renewal timing, pricing objections, grace period or weak intent are creating the clearest retention risk."
    if name == "Feedback & roadmap":
        return "Customers are signalling unmet product needs. This is a Product/CS alignment topic, not only a support follow-up."
    if name == "Sentiment":
        return "Tone and relationship signals are weakening. CSMs should validate expectations and look for frustration before it becomes renewal risk."
    if name == "Usage & adoption":
        return "Customers are not using enough of the product to clearly realise value. Adoption recovery should be prioritised."
    if name == "Support experience":
        return "Support friction is visible: reopened issues, slow resolution or lower CSAT may be eroding confidence."
    if name == "Bugs & reliability":
        return "Reliability risk exists when bugs, stuck issues or data-trust problems reduce confidence in the platform."
    if name == "Limit friction":
        return "Some customers are close to package limits. This can be healthy usage, but it needs expectation management or upgrade fit review."
    return "This signal family contributes to portfolio risk and should be reviewed together with account-level context."


def plan_family_story_copy(plan_family: str, probability: float) -> str:
    """CSM-friendly explanation for average churn probability by package family."""
    name = str(plan_family)
    if probability >= 0.50:
        level = "high"
    elif probability >= 0.40:
        level = "elevated"
    else:
        level = "contained"

    if "Local" in name:
        action = "Review location activation, frozen locations, listings sync, review workflows and local rank tracking adoption."
    elif "Traffic" in name:
        action = "Validate whether market and competitor insights are being used in recurring business reviews and reporting workflows."
    elif "AI Visibility" in name:
        action = "Run an AI Visibility workflow session: prompts, competitors, AI Visibility Score and recurring AI search review."
    elif "SEO" in name:
        action = "Review Site Audit, Position Tracking, crawl limits, keyword tracking usage, My Reports and advanced value workflows."
    elif "Content" in name:
        action = "Check whether content workflows are connected to measurable outputs: briefs, pages, rankings and reporting."
    elif "Semrush One" in name:
        action = "Use broader suite adoption to reinforce value across SEO, AI, Traffic and reporting workflows."
    else:
        action = "Review whether the package is aligned with the customer’s main business goals and active workflows."

    return f"Risk is {level} for this package family. Recommended CSM focus: {action}"


def render_story_summary_cards(title_1, value_1, body_1, title_2, value_2, body_2, title_3, value_3, body_3):
    st.markdown(
        f"""
        <div class="story-grid">
            <div class="story-card">
                <div class="story-kicker">Primary theme</div>
                <div class="story-headline">{html.escape(str(title_1))}</div>
                <div class="metric-value" style="font-size:28px;">{html.escape(str(value_1))}</div>
                <div class="story-body">{html.escape(str(body_1))}</div>
            </div>
            <div class="story-card">
                <div class="story-kicker">Secondary signal</div>
                <div class="story-headline">{html.escape(str(title_2))}</div>
                <div class="metric-value" style="font-size:28px;">{html.escape(str(value_2))}</div>
                <div class="story-body">{html.escape(str(body_2))}</div>
            </div>
            <div class="story-card">
                <div class="story-kicker">Lowest pressure</div>
                <div class="story-headline">{html.escape(str(title_3))}</div>
                <div class="metric-value" style="font-size:28px;">{html.escape(str(value_3))}</div>
                <div class="story-body">{html.escape(str(body_3))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ranked_story_bars(df, label_col, value_col, value_formatter, max_value=None, hot_threshold=None):
    view = df[[label_col, value_col]].copy().dropna().sort_values(value_col, ascending=False)
    if view.empty:
        st.info("No data available for this section.")
        return

    if max_value is None:
        raw_max = float(view[value_col].max())
        max_value = raw_max if raw_max > 0 else 1.0

    rows = []
    for _, r in view.iterrows():
        label = html.escape(str(r[label_col]))
        val = float(r[value_col])
        width = max(2, min(100, val / max_value * 100))
        css = "hot" if hot_threshold is not None and val >= hot_threshold else "medium" if width >= 60 else ""
        formatted_value = html.escape(str(value_formatter(val)))

        # Keep this HTML as one compact string.
        # If it is indented in a triple-quoted string, Streamlit/Markdown may render it as code.
        rows.append(
            f'<div class="story-row">'
            f'<div class="story-name">{label}</div>'
            f'<div class="story-track"><div class="story-fill {css}" style="width:{width:.1f}%;"></div></div>'
            f'<div class="story-value">{formatted_value}</div>'
            f'</div>'
        )

    html_block = f'<div class="story-bar-list">{"".join(rows)}</div>'
    st.markdown(html_block, unsafe_allow_html=True)


def render_driver_story_section(avg_scores: pd.DataFrame):
    view = avg_scores.copy().sort_values("Average score", ascending=False).reset_index(drop=True)
    if view.empty:
        st.info("No risk driver data available.")
        return
    top = view.iloc[0]
    second = view.iloc[1] if len(view) > 1 else view.iloc[0]
    lowest = view.iloc[-1]

    render_story_summary_cards(
        top["Risk family"], f"{top['Average score']:.1f}/100", driver_story_copy(top["Risk family"], top["Average score"]),
        second["Risk family"], f"{second['Average score']:.1f}/100", driver_story_copy(second["Risk family"], second["Average score"]),
        lowest["Risk family"], f"{lowest['Average score']:.1f}/100", "This is currently the least visible source of risk in the selected portfolio view. Keep monitoring, but prioritise stronger signals first.",
    )
    render_ranked_story_bars(
        view,
        "Risk family",
        "Average score",
        lambda v: f"{v:.1f}/100",
        max_value=max(35.0, float(view["Average score"].max())),
        hot_threshold=25,
    )
    st.markdown(
        f"""
        <div class="story-note">
            <strong>How to read this:</strong> these are not separate churn probabilities. They are explainability scores that show
            where risk is building. In this view, the strongest portfolio story is <strong>{html.escape(str(top['Risk family']))}</strong>.
            CSMs should start by reviewing accounts where this theme appears together with high churn probability or near-term renewal.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_commercial_exposure_story_section(by_family: pd.Series):
    view = by_family.sort_values(ascending=False).reset_index()
    view.columns = ["Plan family", "Average churn probability"]
    if view.empty:
        st.info("No commercial exposure data available.")
        return
    top = view.iloc[0]
    second = view.iloc[1] if len(view) > 1 else view.iloc[0]
    lowest = view.iloc[-1]

    render_story_summary_cards(
        top["Plan family"], f"{top['Average churn probability']:.1%}", plan_family_story_copy(top["Plan family"], top["Average churn probability"]),
        second["Plan family"], f"{second['Average churn probability']:.1%}", plan_family_story_copy(second["Plan family"], second["Average churn probability"]),
        lowest["Plan family"], f"{lowest['Average churn probability']:.1%}", "This package family currently shows the lowest average predicted churn probability in the selected view. Use it as a benchmark for healthy adoption patterns.",
    )
    render_ranked_story_bars(
        view,
        "Plan family",
        "Average churn probability",
        lambda v: f"{v:.1%}",
        max_value=max(0.60, float(view["Average churn probability"].max())),
        hot_threshold=0.50,
    )
    st.markdown(
        f"""
        <div class="story-note">
            <strong>CSM takeaway:</strong> package-level risk should not be read as “this product is bad”. It tells us where to inspect the customer journey first.
            For <strong>{html.escape(str(top['Plan family']))}</strong>, the next step is to connect the package-specific usage pattern with account-level drivers:
            adoption, limits, feedback, bugs, sentiment and renewal timing.
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    package = joblib.load(path)
    if isinstance(package, dict):
        return package["model"], package["columns"], package
    return package, None, {}


def prepare_features(df, model_columns=None):
    X = df.drop(columns=[c for c in EXCLUDED_COLUMNS if c in df.columns], errors="ignore").copy()
    X = pd.get_dummies(X, drop_first=True)
    if model_columns is not None:
        X = X.reindex(columns=model_columns, fill_value=0)
    return X


def predict_for_df(model, model_columns, df):
    X = prepare_features(df, model_columns)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def yesno(v):
    return "Yes" if int(v) == 1 else "No"


def family_risk_scores(row):
    scores = {}
    scores["Usage & adoption"] = np.clip(
        0.32 * max(0, 65 - row.get("tool_activity_score", 0)) +
        0.28 * max(0, 60 - row.get("feature_adoption_score", 0)) +
        0.25 * max(0, -row.get("usage_change_vs_prev_quarter", 0)) +
        0.20 * max(0, row.get("days_since_last_login", 0) - 14), 0, 100)
    scores["Renewal & commercial"] = np.clip(
        30 * (row.get("renewal_due_days", 999) < 45) +
        30 * (row.get("renewal_intent", "positive") == "negative") +
        16 * (row.get("renewal_intent", "positive") == "neutral") +
        22 * row.get("pricing_objection_flag", 0) +
        26 * row.get("downgrade_risk_flag", 0) +
        20 * row.get("in_grace_period", 0), 0, 100)
    scores["Support experience"] = np.clip(
        9 * row.get("reopened_tickets", 0) +
        3 * row.get("avg_resolution_days", 0) +
        25 * max(0, 3.5 - row.get("csat_support", 5)), 0, 100)
    scores["Feedback & roadmap"] = np.clip(
        10 * row.get("critical_feedback_count_90d", 0) +
        7 * row.get("unresolved_feedback_count", 0) +
        0.35 * row.get("avg_unresolved_feedback_age_days", 0) +
        0.55 * max(0, 55 - row.get("roadmap_alignment_score", 100)) +
        24 * row.get("feedback_linked_to_renewal", 0) +
        20 * row.get("competitor_mentioned_in_feedback", 0), 0, 100)
    scores["Bugs & reliability"] = np.clip(
        14 * row.get("critical_bugs_reported_90d", 0) +
        18 * row.get("stuck_bugs_count", 0) +
        0.45 * row.get("oldest_open_bug_days", 0) +
        28 * row.get("data_accuracy_bug_flag", 0) +
        24 * row.get("bug_linked_to_renewal_flag", 0) -
        10 * row.get("workaround_available_flag", 0), 0, 100)
    scores["Sentiment"] = np.clip(
        40 * max(0, -row.get("sentiment_csm", 0)) +
        35 * max(0, -row.get("sentiment_support", 0)) +
        30 * max(0, -row.get("feedback_sentiment_score", 0)) +
        25 * max(0, -row.get("bug_sentiment_score", 0)), 0, 100)
    scores["Limit friction"] = np.clip(
        0.75 * max(0, row.get("crawl_budget_usage_pct", 0) - 80) +
        0.75 * max(0, row.get("keyword_tracking_usage_pct", 0) - 80) +
        0.80 * max(0, row.get("ai_visibility_prompt_usage_pct", 0) - 80) +
        18 * row.get("limit_upgrade_recommended_flag", 0), 0, 100)
    return {k: round(float(v), 1) for k, v in scores.items()}


def top_drivers(row, limit=6):
    drivers = []
    if row.get("tool_activity_score", 100) < 40: drivers.append("Low product activity")
    if row.get("feature_adoption_score", 100) < 45: drivers.append("Weak feature adoption")
    if row.get("usage_change_vs_prev_quarter", 0) < -15: drivers.append("Usage decline")
    if row.get("crawl_budget_usage_pct", 0) > 85: drivers.append("Crawl budget saturation")
    if row.get("keyword_tracking_usage_pct", 0) > 85: drivers.append("Keyword tracking limit pressure")
    if row.get("ai_visibility_active", 0) and row.get("ai_visibility_prompts_tracked", 0) < 10: drivers.append("AI Visibility under-adoption")
    if row.get("roadmap_alignment_score", 100) < 40: drivers.append("Roadmap mismatch")
    if row.get("feedback_linked_to_renewal", 0): drivers.append("Feedback linked to renewal")
    if row.get("critical_feedback_count_90d", 0) > 0: drivers.append("Critical unresolved product feedback")
    if row.get("data_accuracy_bug_flag", 0): drivers.append("Data accuracy trust issue")
    if row.get("stuck_bugs_count", 0) > 0: drivers.append("Stuck bugs")
    if row.get("workaround_available_flag", 1) == 0: drivers.append("No workaround available")
    if row.get("renewal_due_days", 999) < 45 and row.get("renewal_intent", "positive") != "positive": drivers.append("Near-term renewal risk")
    if row.get("sentiment_csm", 0) < -0.2: drivers.append("Negative CSM sentiment")
    return drivers[:limit]


def semrush_csm_cta(row):
    actions = []
    owners = []

    if row.get("data_accuracy_bug_flag", 0) and row.get("stuck_bugs_count", 0) >= 1:
        actions.append("Escalate as a data-trust risk, not a regular bug. Confirm data impact, request Product/Support ETA, document workaround and set a customer update cadence.")
        owners += ["Support Lead", "Product Owner", "CSM"]
    if row.get("crawl_budget_usage_pct", 0) > 85 and row.get("site_audit_projects_active", 0) > 0:
        actions.append("Run a Site Audit limit-fit review: confirm crawl scope, audit frequency, pages-per-audit setup, excluded pages and whether the current plan limits fit website size.")
        owners += ["CSM", "Technical Support", "Account Owner"]
    if row.get("keyword_tracking_usage_pct", 0) > 85:
        actions.append("Review Position Tracking setup: keyword allocation, inactive campaigns, devices/locations, and whether additional tracking capacity or a higher tier is justified.")
        owners += ["CSM", "SEO Specialist", "Account Owner"]
    if row.get("ai_visibility_active", 0) and row.get("ai_visibility_prompts_tracked", 0) < 10:
        actions.append("Book an AI Visibility workflow session. Help the customer define priority prompts, select competitors, review AI Visibility Score and schedule a recurring AI search review.")
        owners += ["CSM", "AI Visibility Specialist"]
    if row.get("local_toolkit_active", 0) and (row.get("frozen_locations_count", 0) > 0 or row.get("duplicate_suppression_issues", 0) > 2):
        actions.append("Run a Local Toolkit cleanup session: review frozen locations, duplicate suppression, listing sync status, review workflows and location ownership.")
        owners += ["CSM", "Local Specialist", "Support"]
    if row.get("feedback_linked_to_renewal", 0) and row.get("roadmap_alignment_score", 100) < 45:
        actions.append("Treat the product gap as a renewal dependency. Align with Product on roadmap status, prepare expectation management and capture workaround options before the renewal conversation.")
        owners += ["CSM", "Product Owner", "Account Owner"]
    if row.get("plan_tier", "") in ["Guru", "Business", "Pro+", "Advanced"] and row.get("tool_activity_score", 100) < 45:
        actions.append("Run an adoption recovery session focused on paid-but-underused workflows: Site Audit, Position Tracking, competitor analysis, My Reports, AI Visibility or API/Share of Voice depending on entitlement.")
        owners += ["CSM", "Account Owner"]
    if row.get("api_access_available", 0) and not row.get("api_access_used", 0):
        actions.append("For this advanced-plan account, validate whether API access could automate reporting or reduce manual work. Offer an API/value discovery conversation.")
        owners += ["CSM", "Solutions Consultant"]
    if row.get("renewal_due_days", 999) < 30 and row.get("renewal_intent", "positive") != "positive":
        actions.append("Trigger renewal-risk motion: confirm decision criteria, agree recovery actions, involve commercial owner and set a next-step date before renewal.")
        owners += ["CSM", "Account Owner", "Renewals"]
    if not actions:
        actions.append("Maintain proactive value reinforcement. Review adoption goals, confirm active workflows, and identify one expansion or enablement opportunity for the next customer touchpoint.")
        owners += ["CSM"]

    # de-duplicate owners while preserving order
    seen = set(); owners_unique=[]
    for o in owners:
        if o not in seen:
            seen.add(o); owners_unique.append(o)

    urgency = "High" if row["churn_probability"] >= 0.70 or row.get("renewal_due_days", 999) < 30 else "Medium" if row["churn_probability"] >= 0.40 else "Low"
    return actions[:3], owners_unique[:5], urgency


def playbook_html(row):
    actions, owners, urgency = semrush_csm_cta(row)
    driver_text = ", ".join(top_drivers(row, 4)) or "No major risk concentration"
    cards = [
        ("Risk reason", driver_text, "What is creating the strongest visible risk for this account."),
        ("Business impact", f"MRR ${row.get('mrr',0):,.0f} · Renewal in {int(row.get('renewal_due_days',0))} days", "Use this to decide whether to escalate or monitor."),
        ("Recommended CSM action", actions[0], "Primary operational next step."),
        ("Owners to involve", ", ".join(owners), "Recommended cross-functional support."),
        ("Urgency", urgency, "Based on churn probability, renewal timing and severity of risk signals."),
    ]
    html_cards = []
    for label, title, body in cards:
        html_cards.append(f'<div class="playbook-card"><div class="playbook-label">{html.escape(label)}</div><div class="playbook-title">{html.escape(str(title))}</div><div class="playbook-body">{html.escape(str(body))}</div></div>')
    if len(actions) > 1:
        html_cards.append(f'<div class="playbook-card"><div class="playbook-label">Secondary action</div><div class="playbook-title">{html.escape(actions[1])}</div><div class="playbook-body">Use this if the primary action is already in motion.</div></div>')
    return '<div class="playbook-grid">' + ''.join(html_cards) + '</div>'


def bar(label, value):
    warn = "warn" if value >= 65 else ""
    return f'<div class="small-muted"><strong>{html.escape(label)}</strong> · {value:.1f}/100</div><div class="bar-wrap"><div class="bar-fill {warn}" style="width:{min(100, max(0, value)):.1f}%"></div></div>'


def simulate_scenarios(row, model, model_columns):
    base = pd.DataFrame([row.drop(labels=[c for c in ["churn_probability", "risk_level", "priority"] if c in row.index])])
    current = float(row["churn_probability"])
    scenarios = []

    def add(name, description, changes):
        simulated = base.copy()
        for k, v in changes.items():
            if k in simulated.columns:
                simulated.loc[simulated.index[0], k] = v(simulated.loc[simulated.index[0], k]) if callable(v) else v
        new_prob = float(predict_for_df(model, model_columns, simulated)[0])
        scenarios.append({
            "Scenario": name,
            "What changes": description,
            "Current probability": current,
            "Simulated probability": new_prob,
            "Change": new_prob - current,
            "New risk level": risk_level(new_prob),
        })

    add("Improve Site Audit & usage", "Increase activity/adoption and reduce inactivity", {
        "tool_activity_score": lambda x: min(88, max(x, 72)),
        "feature_adoption_score": lambda x: min(85, max(x, 70)),
        "days_since_last_login": lambda x: min(x, 5),
        "site_audit_last_run_days": lambda x: min(x, 7),
    })
    add("Resolve limit friction", "Reduce crawl/keyword/prompt saturation and mark upgrade-fit action", {
        "crawl_budget_usage_pct": lambda x: min(x, 72),
        "keyword_tracking_usage_pct": lambda x: min(x, 72),
        "ai_visibility_prompt_usage_pct": lambda x: min(x, 72),
        "limit_upgrade_recommended_flag": 0,
    })
    add("Strengthen renewal signal", "Move renewal intent to positive and remove grace/pricing/downgrade risk", {
        "renewal_intent": "positive",
        "in_grace_period": 0,
        "pricing_objection_flag": 0,
        "downgrade_risk_flag": 0,
    })
    add("Resolve product feedback", "Close critical feedback and improve roadmap alignment/sentiment", {
        "critical_feedback_count_90d": 0,
        "unresolved_feedback_count": lambda x: max(0, min(x, 1)),
        "avg_unresolved_feedback_age_days": lambda x: min(x, 14),
        "roadmap_alignment_score": lambda x: max(x, 75),
        "feedback_sentiment_score": lambda x: max(x, 0.35),
        "feedback_linked_to_renewal": 0,
    })
    add("Resolve bug reliability issues", "Remove stuck/data-trust bugs and set workaround available", {
        "critical_bugs_reported_90d": 0,
        "stuck_bugs_count": 0,
        "open_bugs_count": lambda x: max(0, min(x, 1)),
        "oldest_open_bug_days": lambda x: min(x, 7),
        "data_accuracy_bug_flag": 0,
        "workaround_available_flag": 1,
        "bug_linked_to_renewal_flag": 0,
        "bug_sentiment_score": lambda x: max(x, 0.35),
    })
    add("Best-case recovery package", "Combine adoption, renewal, feedback and bug recovery", {
        "tool_activity_score": lambda x: min(90, max(x, 78)),
        "feature_adoption_score": lambda x: min(90, max(x, 76)),
        "days_since_last_login": lambda x: min(x, 4),
        "renewal_intent": "positive",
        "in_grace_period": 0,
        "pricing_objection_flag": 0,
        "unresolved_feedback_count": 0,
        "critical_feedback_count_90d": 0,
        "roadmap_alignment_score": lambda x: max(x, 80),
        "stuck_bugs_count": 0,
        "data_accuracy_bug_flag": 0,
        "workaround_available_flag": 1,
        "bug_linked_to_renewal_flag": 0,
    })
    out = pd.DataFrame(scenarios)
    out["Current probability"] = out["Current probability"].map(lambda x: f"{x:.1%}")
    out["Simulated probability"] = out["Simulated probability"].map(lambda x: f"{x:.1%}")
    out["Change"] = out["Change"].map(lambda x: f"{x*100:+.1f} pp")
    return out


# Load data and model
inject_css()

st.markdown(
    """
    <div class="hero">
      <div class="hero-eyebrow">Semrush-like synthetic demo · CSM Retention Workspace</div>
      <div class="hero-title">Semrush Retention Intelligence Copilot V3</div>
      <div class="hero-sub">
        A simulated Customer Success decision-support workspace that connects Semrush-like product adoption,
        toolkit limits, renewal context, support friction, product feedback and bug reliability into clear CSM next actions.
        This demo uses synthetic data inspired by public Semrush product information, not internal customer data.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

missing = [str(p.name) for p in [DATA_PATH, MODEL_PATH] if not p.exists()]
if missing:
    st.error("Missing required file(s): " + ", ".join(missing) + ". Place them in the same folder as this app.")
    st.stop()

raw = load_data(DATA_PATH)
model, model_columns, model_package = load_model(MODEL_PATH)
df = raw.copy()
df["churn_probability"] = predict_for_df(model, model_columns, df)
df["risk_level"] = df["churn_probability"].apply(risk_level)
df["priority"] = np.select(
    [
        (df["risk_level"] == "High") & (df["mrr"] >= 1500),
        (df["risk_level"] == "High"),
        (df["risk_level"] == "Medium") & (df["renewal_due_days"] < 60),
        (df["risk_level"] == "Medium"),
    ],
    ["P1", "P2", "P2", "P3"],
    default="Monitor",
)
df = df.sort_values("churn_probability", ascending=False).reset_index(drop=True)

# Sidebar
st.sidebar.header("Filters")
segments = ["All"] + sorted(df["segment"].dropna().unique().tolist())
plan_families = ["All"] + sorted(df["plan_family"].dropna().unique().tolist())
plans = ["All"] + sorted(df["plan_tier"].dropna().unique().tolist())
risks = ["All", "High", "Medium", "Low"]
regions = ["All"] + sorted(df["region"].dropna().unique().tolist())

segment_filter = st.sidebar.selectbox("Segment", segments)
family_filter = st.sidebar.selectbox("Plan family", plan_families)
plan_filter = st.sidebar.selectbox("Plan tier", plans)
risk_filter = st.sidebar.selectbox("Risk level", risks)
region_filter = st.sidebar.selectbox("Region", regions)
only_renewal = st.sidebar.checkbox("Renewal due < 60 days")
only_limit = st.sidebar.checkbox("Limit pressure detected")
only_bugs = st.sidebar.checkbox("Bug/data trust issues")

filtered = df.copy()
if segment_filter != "All": filtered = filtered[filtered["segment"] == segment_filter]
if family_filter != "All": filtered = filtered[filtered["plan_family"] == family_filter]
if plan_filter != "All": filtered = filtered[filtered["plan_tier"] == plan_filter]
if risk_filter != "All": filtered = filtered[filtered["risk_level"] == risk_filter]
if region_filter != "All": filtered = filtered[filtered["region"] == region_filter]
if only_renewal: filtered = filtered[filtered["renewal_due_days"] < 60]
if only_limit: filtered = filtered[(filtered["crawl_budget_usage_pct"] > 85) | (filtered["keyword_tracking_usage_pct"] > 85) | (filtered["ai_visibility_prompt_usage_pct"] > 85)]
if only_bugs: filtered = filtered[(filtered["stuck_bugs_count"] > 0) | (filtered["data_accuracy_bug_flag"] == 1)]
filtered = filtered.sort_values("churn_probability", ascending=False).reset_index(drop=True)

if filtered.empty:
    st.warning("No accounts match the selected filters.")
    st.stop()

# Top metrics
avg_churn = filtered["churn_probability"].mean()
high_count = int((filtered["risk_level"] == "High").sum())
mrr_at_risk = filtered.loc[filtered["risk_level"] == "High", "mrr"].sum()
renewal_pressure = int(((filtered["renewal_due_days"] < 60) & (filtered["renewal_intent"] != "positive")).sum())
limit_pressure = int(((filtered["crawl_budget_usage_pct"] > 85) | (filtered["keyword_tracking_usage_pct"] > 85) | (filtered["ai_visibility_prompt_usage_pct"] > 85)).sum())

m1,m2,m3,m4,m5 = st.columns(5)
with m1: metric_card("Accounts", f"{len(filtered):,}", "Accounts in current view")
with m2: metric_card("Avg churn probability", f"{avg_churn:.1%}", "Average predicted risk")
with m3: metric_card("High-risk accounts", f"{high_count:,}", "High >= 70%")
with m4: metric_card("MRR at risk", f"${mrr_at_risk:,.0f}", "High-risk MRR exposure")
with m5: metric_card("Renewal pressure", f"{renewal_pressure:,}", "Due < 60 days and not positive")

# Priority cards
section_header("Priority accounts for CSM review", "Accounts where the model detects the highest operational retention risk.")
top3 = filtered.head(3)
pcols = st.columns(3)
for i, (_, r) in enumerate(top3.iterrows()):
    with pcols[i]:
        actions, owners, urgency = semrush_csm_cta(r)
        chips = ''.join([f'<span class="driver-chip">{html.escape(x)}</span>' for x in top_drivers(r, 3)])
        st.markdown(
            f"""
            <div class="panel-card">
              <div class="metric-label">{html.escape(r['priority'])} · {html.escape(r['plan_family'])}</div>
              <div style="font-size:20px;font-weight:900;color:#071427;margin-bottom:3px;">{html.escape(r['account_name'])}</div>
              <div class="small-muted">{html.escape(r['account_id'])} · {html.escape(r['segment'])} · {html.escape(r['plan_tier'])}</div>
              <div style="margin:12px 0;">{risk_badge(r['risk_level'])}</div>
              <div style="font-size:27px;font-weight:900;color:#071427;">{r['churn_probability']:.1%}</div>
              <div class="small-muted">Renewal in {int(r['renewal_due_days'])} days · MRR ${r['mrr']:,.0f}</div>
              <div style="margin-top:10px;">{chips}</div>
              <div class="cta-box" style="margin-top:12px;padding:14px;border-radius:16px;"><div class="cta-title" style="font-size:14px;">CSM next action</div><div class="cta-body">{html.escape(actions[0])}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Tabs
executive, adoption, table_tab, detail_tab, playbook_tab, methodology = st.tabs([
    "Executive Dashboard", "Semrush Tool Adoption", "Account Table", "Account Detail", "CSM Playbook", "Methodology"
])

with executive:
    section_header("Portfolio risk overview", "Risk concentration by family and segment.")
    risk_counts = filtered["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)
    by_segment = filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=False)
    by_family = filtered.groupby("plan_family")["churn_probability"].mean().sort_values(ascending=False)
    d1,d2 = st.columns(2)
    with d1:
        render_static_bar_chart(
            risk_counts,
            title="Accounts by risk level",
            y_label="Accounts",
            as_percent=False,
        )
    with d2:
        render_static_bar_chart(
            by_segment,
            title="Average churn probability by segment",
            y_label="Churn probability",
            as_percent=True,
        )

    section_header("Where is risk building?", "A storytelling view of the strongest risk themes in the current portfolio.")
    family_scores = pd.DataFrame([family_risk_scores(r) for _, r in filtered.iterrows()])
    avg_scores = family_scores.mean().sort_values(ascending=False).reset_index()
    avg_scores.columns = ["Risk family", "Average score"]
    render_driver_story_section(avg_scores)

    section_header("Which package families need attention?", "A CSM-friendly view of average predicted churn by Semrush-like package family.")
    render_commercial_exposure_story_section(by_family)

with adoption:
    section_header("Semrush-like tool adoption signals", "Concrete usage and limit patterns that CSMs can act on.")
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Limit pressure", f"{limit_pressure}", "Accounts using >85% of a key limit")
    with c2: metric_card("AI under-adoption", f"{int(((filtered['ai_visibility_active']==1) & (filtered['ai_visibility_prompts_tracked']<10)).sum())}", "AI Visibility active but <10 prompts")
    with c3: metric_card("Data trust bugs", f"{int((filtered['data_accuracy_bug_flag']==1).sum())}", "Accounts with data accuracy bug flag")
    with c4: metric_card("Roadmap mismatch", f"{int((filtered['roadmap_alignment_score']<40).sum())}", "Feedback misaligned with roadmap")

    adoption_cols = [
        "account_id","account_name","segment","plan_family","plan_tier","churn_probability","risk_level",
        "tool_activity_score","feature_adoption_score","crawl_budget_usage_pct","keyword_tracking_usage_pct",
        "ai_visibility_prompt_usage_pct","local_locations_active","roadmap_alignment_score","stuck_bugs_count","data_accuracy_bug_flag"
    ]
    show = filtered[adoption_cols].copy().head(200)
    show["churn_probability"] = show["churn_probability"].map(lambda x: f"{x:.1%}")
    st.dataframe(show, use_container_width=True, hide_index=True)

with table_tab:
    section_header("Operational account table", "Ranked by predicted churn probability and enriched with CSM priority.")
    cols = [
        "priority","account_id","account_name","segment","region","plan_family","plan_tier","mrr","churn_probability","risk_level","renewal_due_days","renewal_intent",
        "tool_activity_score","feature_adoption_score","crawl_budget_usage_pct","keyword_tracking_usage_pct","roadmap_alignment_score","stuck_bugs_count","data_accuracy_bug_flag"
    ]
    table = filtered[cols].copy()
    table["mrr"] = table["mrr"].map(lambda x: f"${x:,.0f}")
    table["churn_probability"] = table["churn_probability"].map(lambda x: f"{x:.1%}")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("Download current account view as CSV", table.to_csv(index=False), "semrush_copilot_account_view.csv", "text/csv")

with detail_tab:
    section_header("Account Detail", "CSM-facing explanation, playbook and simulator for a selected account.")
    selected = st.selectbox("Select account", filtered["account_id"].tolist(), format_func=lambda x: f"{x} · {filtered.loc[filtered['account_id']==x, 'account_name'].iloc[0]}")
    row = filtered.loc[filtered["account_id"] == selected].iloc[0]
    drivers = top_drivers(row)
    chips = ''.join([f'<span class="driver-chip">{html.escape(d)}</span>' for d in drivers])
    st.markdown(
        f"""
        <div class="panel-card">
          <div class="metric-label">{html.escape(row['account_id'])} · {html.escape(row['plan_family'])} · {html.escape(row['plan_tier'])}</div>
          <div style="font-size:28px;font-weight:900;color:#071427;">{html.escape(row['account_name'])}</div>
          <div class="small-muted">{html.escape(row['segment'])} · {html.escape(row['industry'])} · {html.escape(row['region'])} · MRR ${row['mrr']:,.0f}</div>
          <div style="margin:14px 0;">{risk_badge(row['risk_level'])} <span style="font-size:34px;font-weight:900;margin-left:12px;color:#071427;">{row['churn_probability']:.1%}</span></div>
          <div>{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("Risk driver scores", "What kind of risk is driving this account: adoption, renewal, support, feedback, bugs or limits.")
    scores = family_risk_scores(row)
    cols = st.columns(2)
    score_items = list(scores.items())
    for idx, (label, value) in enumerate(score_items):
        with cols[idx % 2]:
            st.markdown(bar(label, value), unsafe_allow_html=True)

    section_header("Structured CSM action plan", "Direct instructions for the CSM and cross-functional owners.")
    st.markdown(playbook_html(row), unsafe_allow_html=True)

    section_header("Semrush-specific signal snapshot", "The most relevant product usage and friction fields for this account.")
    snapshot = pd.DataFrame([
        ["Plan family", row["plan_family"]],
        ["Plan tier", row["plan_tier"]],
        ["Crawl budget usage", f"{row['crawl_budget_usage_pct']:.1f}%"],
        ["Keyword tracking usage", f"{row['keyword_tracking_usage_pct']:.1f}%"],
        ["AI prompts tracked", f"{int(row['ai_visibility_prompts_tracked'])} / {int(row['ai_visibility_prompt_limit'])}"],
        ["Site Audit last run", f"{int(row['site_audit_last_run_days'])} days ago"],
        ["Site Audit health", f"{row['site_audit_health_score_avg']:.1f}"],
        ["Position Tracking campaigns", int(row['position_tracking_campaigns_active'])],
        ["Local active locations", f"{int(row['local_locations_active'])} / {int(row['local_locations_target'])}"],
        ["Roadmap alignment", f"{row['roadmap_alignment_score']:.1f}/100"],
        ["Open / stuck bugs", f"{int(row['open_bugs_count'])} / {int(row['stuck_bugs_count'])}"],
        ["Data accuracy bug", yesno(row['data_accuracy_bug_flag'])],
        ["Workaround available", yesno(row['workaround_available_flag'])],
    ], columns=["Signal", "Value"])
    snapshot["Value"] = snapshot["Value"].astype(str)
    st.dataframe(snapshot, use_container_width=True, hide_index=True)

    section_header("Scenario Simulator", "Estimate how risk might change if the team resolves specific blockers.")
    scenarios = simulate_scenarios(row, model, model_columns)
    st.dataframe(scenarios, use_container_width=True, hide_index=True)
    st.caption("Scenario results are model-based simulations on synthetic data. They are decision-support estimates, not guaranteed outcomes.")

with playbook_tab:
    section_header("CSM Playbook", "Rules that translate model signals into direct customer-success actions.")
    rules = pd.DataFrame([
        ["Crawl budget saturation", "crawl_budget_usage_pct > 85", "Run a Site Audit limit-fit review: crawl scope, audit frequency, excluded pages and plan limit fit."],
        ["Keyword tracking limit pressure", "keyword_tracking_usage_pct > 85", "Review Position Tracking keyword allocation, inactive campaigns, devices/locations and upgrade fit."],
        ["AI Visibility under-adoption", "ai_visibility_active = 1 and prompts tracked < 10", "Book an AI Visibility workflow session: define prompts, competitors and recurring AI search review."],
        ["Product gap linked to renewal", "feedback_linked_to_renewal = 1 and roadmap_alignment_score < 45", "Treat as renewal dependency. Align Product roadmap status and workaround before renewal conversation."],
        ["Data trust bug", "data_accuracy_bug_flag = 1 and stuck_bugs_count >= 1", "Escalate as a trust-risk issue. Confirm impact, ETA, workaround and update cadence."],
        ["Advanced plan under-adoption", "Business / Pro+ / Advanced and activity < 45", "Run adoption recovery focused on paid-but-underused workflows and executive reporting."],
        ["Active but frustrated", "High activity and negative sentiment", "Do not treat as disengagement; run value-and-friction review to remove blockers."],
    ], columns=["Risk pattern", "Condition", "CSM instruction"])
    st.dataframe(rules, use_container_width=True, hide_index=True)

with methodology:
    section_header("Methodology", "How the V3 Copilot calculates predictions and why this is Semrush-like.")
    st.markdown(
        """
        <div class="method-card"><strong>What is predicted?</strong><br>
        For each synthetic account, the model estimates <code>P(churn = 1 | account signals)</code>. The portfolio KPI is the average of those account-level probabilities.</div>
        <div class="method-card"><strong>What goes into the model?</strong><br>
        Account profile, plan family, toolkit adoption, usage limits, Site Audit and Position Tracking usage, AI Visibility adoption, Local Toolkit activity, support friction, feedback, bugs, sentiment and renewal context.</div>
        <div class="method-card"><strong>What is not used live?</strong><br>
        <code>account_id</code>, <code>account_name</code>, <code>churn_label</code>, and <code>synthetic_true_churn_probability</code> are excluded from predictive features. The churn label is used only during training.</div>
        <div class="method-card"><strong>Important disclaimer</strong><br>
        This is synthetic demo data inspired by public Semrush product information. It is not internal Semrush customer data and should be retrained on authorized historical company data before production use.</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("Model package:", model_package.get("model_type", "Unknown"), "· Training columns:", len(model_columns or []))
