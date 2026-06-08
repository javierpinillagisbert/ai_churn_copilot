
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


        .routing-grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:14px; margin: 10px 0 18px; }
        .routing-card { background:white; border:1px solid var(--border); border-radius:22px; padding:20px; box-shadow:0 10px 28px rgba(15,23,42,.055); min-height:210px; }
        .routing-card.support { border-top:5px solid #2563EB; }
        .routing-card.sales { border-top:5px solid #7C3AED; }
        .routing-card.product { border-top:5px solid #0F766E; }
        .routing-card.csm { border-top:5px solid #64748B; }
        .routing-team { color:var(--navy); font-size:18px; font-weight:900; margin-bottom:8px; }
        .routing-kicker { color:var(--muted); font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.07em; margin:10px 0 5px; }
        .routing-body { color:#475569; font-size:13px; line-height:1.45; }
        .urgency-pill { display:inline-block; border-radius:999px; padding:5px 9px; font-size:11px; font-weight:900; margin-bottom:10px; }
        .urgency-high { background:#FEE2E2; color:#991B1B; border:1px solid #FECACA; }
        .urgency-medium { background:#FEF3C7; color:#92400E; border:1px solid #FDE68A; }
        .urgency-low { background:#DCFCE7; color:#166534; border:1px solid #BBF7D0; }
        .handoff-box { background:linear-gradient(135deg,#F8FAFC,#EFF6FF); border:1px solid #BFDBFE; border-radius:20px; padding:16px 18px; color:#334155; font-size:14px; line-height:1.55; margin:12px 0 18px; }
        .queue-grid { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:14px; margin: 10px 0 22px; }
        .queue-card { background:white; border:1px solid var(--border); border-radius:22px; padding:19px; box-shadow:0 10px 28px rgba(15,23,42,.055); }
        .queue-label { color:var(--muted); font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.07em; margin-bottom:8px; }
        .queue-value { color:var(--navy); font-size:32px; font-weight:900; line-height:1; margin-bottom:8px; }
        .queue-note { color:#475569; font-size:13px; line-height:1.4; }
        .routing-note { background:#F8FAFC; border:1px solid var(--border); border-radius:18px; padding:14px 16px; color:#475569; font-size:13px; line-height:1.45; margin-top:10px; }



        /* UX briefing cards: Scenario Simulator + Signal Briefing */
        .ux-card { background:white; border:1px solid var(--border); border-radius:22px; padding:20px; box-shadow:0 10px 28px rgba(15,23,42,.055); }
        .ux-grid-3 { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:14px; margin:12px 0 18px; }
        .ux-grid-2 { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:14px; margin:12px 0 18px; }
        .ux-kicker { color:var(--muted); font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.07em; margin-bottom:8px; }
        .ux-title { color:var(--navy); font-size:19px; font-weight:900; line-height:1.18; margin-bottom:8px; }
        .ux-body { color:#475569; font-size:13px; line-height:1.48; }
        .ux-big { color:var(--navy); font-size:34px; font-weight:950; line-height:1; margin:8px 0; }
        .ux-impact { display:inline-block; padding:6px 10px; border-radius:999px; font-size:12px; font-weight:900; margin:4px 5px 4px 0; }
        .ux-impact.high { background:#DCFCE7; color:#166534; border:1px solid #BBF7D0; }
        .ux-impact.medium { background:#FEF3C7; color:#92400E; border:1px solid #FDE68A; }
        .ux-impact.low { background:#F1F5F9; color:#475569; border:1px solid #E2E8F0; }
        .ux-owner { display:inline-block; padding:5px 9px; border-radius:999px; background:#EFF6FF; color:#1D4ED8; font-size:11px; font-weight:900; margin:3px 4px 0 0; }
        .ux-owner.support { background:#DBEAFE; color:#1D4ED8; }
        .ux-owner.sales { background:#EDE9FE; color:#6D28D9; }
        .ux-owner.product { background:#CCFBF1; color:#0F766E; }
        .ux-owner.csm { background:#F1F5F9; color:#334155; }
        .scenario-hero { background:radial-gradient(circle at 100% 0%, rgba(15,118,110,.18), transparent 26%), linear-gradient(135deg,#FFFFFF,#F0FDFA); border:1px solid #99F6E4; border-radius:24px; padding:22px; box-shadow:0 12px 30px rgba(15,23,42,.06); margin:12px 0 16px; }
        .scenario-metrics { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:12px; margin-top:14px; }
        .scenario-metric { background:white; border:1px solid #DDEBF7; border-radius:16px; padding:13px; }
        .scenario-label { color:var(--muted); font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px; }
        .scenario-value { color:var(--navy); font-size:22px; font-weight:950; }
        .scenario-card { background:white; border:1px solid var(--border); border-radius:22px; padding:18px; box-shadow:0 10px 28px rgba(15,23,42,.055); min-height:245px; }
        .scenario-card.best { border-top:5px solid #0F766E; }
        .scenario-card.medium { border-top:5px solid #D97706; }
        .scenario-card.low { border-top:5px solid #64748B; }
        .signal-section { background:white; border:1px solid var(--border); border-radius:24px; padding:20px; box-shadow:0 10px 28px rgba(15,23,42,.055); margin:14px 0 18px; }
        .signal-section-title { color:var(--navy); font-size:20px; font-weight:950; margin-bottom:4px; }
        .signal-section-sub { color:var(--muted); font-size:13px; margin-bottom:12px; }
        .signal-brief-grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:12px; }
        .signal-brief-card { border:1px solid #E2E8F0; border-radius:18px; padding:15px; background:#FFFFFF; min-height:205px; position:relative; }
        .signal-brief-card.good { border-top:5px solid #16A34A; }
        .signal-brief-card.watch { border-top:5px solid #D97706; }
        .signal-brief-card.risk { border-top:5px solid #DC2626; }
        .signal-name { color:var(--navy); font-size:15px; font-weight:950; margin-bottom:7px; display:flex; align-items:center; gap:7px; }
        .signal-value-big { color:var(--navy); font-size:25px; font-weight:950; margin-bottom:8px; }
        .signal-status { display:inline-block; padding:5px 9px; border-radius:999px; font-size:11px; font-weight:900; margin-bottom:9px; }
        .signal-status.good { background:#DCFCE7; color:#166534; border:1px solid #BBF7D0; }
        .signal-status.watch { background:#FEF3C7; color:#92400E; border:1px solid #FDE68A; }
        .signal-status.risk { background:#FEE2E2; color:#991B1B; border:1px solid #FECACA; }
        .signal-progress { background:#E2E8F0; border-radius:999px; height:9px; overflow:hidden; margin:8px 0 10px; }
        .signal-progress-fill { height:9px; border-radius:999px; background:linear-gradient(90deg,#38BDF8,#2563EB); }
        .signal-progress-fill.good { background:linear-gradient(90deg,#86EFAC,#16A34A); }
        .signal-progress-fill.watch { background:linear-gradient(90deg,#FBBF24,#D97706); }
        .signal-progress-fill.risk { background:linear-gradient(90deg,#F87171,#DC2626); }
        .tooltip-wrap { position:relative; display:inline-flex; align-items:center; justify-content:center; width:18px; height:18px; border-radius:999px; background:#EFF6FF; color:#1D4ED8; font-size:12px; font-weight:950; cursor:help; }
        .tooltip-wrap .tooltip-box { visibility:hidden; opacity:0; position:absolute; z-index:9999; left:0; top:25px; min-width:285px; max-width:340px; background:#071427; color:white; border-radius:14px; padding:12px 13px; box-shadow:0 18px 45px rgba(7,20,39,.25); transition:opacity .15s ease; font-size:12px; line-height:1.45; font-weight:500; }
        .tooltip-wrap:hover .tooltip-box { visibility:visible; opacity:1; }
        .tooltip-box strong { color:#A7F3D0; }
        .signal-cta { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:14px; padding:10px 11px; color:#334155; font-size:12px; line-height:1.4; margin-top:10px; }
        .briefing-note { background:linear-gradient(135deg,#ECFEFF,#F8FAFC); border:1px solid #BAE6FD; border-radius:20px; padding:16px 18px; color:#334155; font-size:14px; line-height:1.55; margin:12px 0 18px; }

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



def _is_flag_on(row, key):
    try:
        return int(row.get(key, 0)) == 1
    except Exception:
        return False


def _pct(row, key, default=0):
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def build_department_routing(row):
    """Return cross-functional teams that should be involved for this account.

    The goal is CSM usability: explain who should help, why they are needed,
    and what the handoff should ask for.
    """
    routing = []

    support_reasons = []
    support_actions = []

    if _is_flag_on(row, "data_accuracy_bug_flag"):
        support_reasons.append("data accuracy discrepancy")
        support_actions.append("validate the reported discrepancy, confirm expected behaviour vs defect, and prepare a customer-facing explanation")

    if _is_flag_on(row, "site_audit_bug_flag"):
        support_reasons.append("Site Audit issue")
        support_actions.append("review crawl settings, crawl limits, excluded pages, audit frequency and expected behaviour")

    if _is_flag_on(row, "position_tracking_bug_flag"):
        support_reasons.append("Position Tracking issue")
        support_actions.append("check campaign setup, keywords, location, device, search engine and tracking configuration")

    if _is_flag_on(row, "ai_visibility_bug_flag"):
        support_reasons.append("AI Visibility issue")
        support_actions.append("review prompt setup, tracked competitors, prompt limits and reporting interpretation")

    if _is_flag_on(row, "local_listing_sync_bug_flag"):
        support_reasons.append("Local listing sync issue")
        support_actions.append("validate Listing Management sync status, frozen locations, duplicate suppression and location-level setup")

    if _pct(row, "crawl_budget_usage_pct") > 85:
        support_reasons.append("crawl budget limit friction")
        support_actions.append("clarify monthly/per-audit crawl limits and help the customer verify current usage")

    if _pct(row, "keyword_tracking_usage_pct") > 85:
        support_reasons.append("keyword tracking limit friction")
        support_actions.append("clarify keyword tracking limits, campaign allocation and user/team allocation where relevant")

    if _pct(row, "ai_visibility_prompt_usage_pct") > 85:
        support_reasons.append("AI Visibility prompt limit friction")
        support_actions.append("clarify prompt tracking limits, current prompt usage and whether the customer needs a package review")

    if _pct(row, "tool_limit_feedback_count") >= 1:
        support_reasons.append("tool limit misunderstanding or friction")
        support_actions.append("check whether the customer is facing a product limit, user allocation issue or configuration issue")

    if _pct(row, "technical_tickets_last_quarter") >= 4:
        support_reasons.append("high technical support volume")
        support_actions.append("review repeated technical topics and provide a clear navigation or troubleshooting summary")

    hard_block = _pct(row, "stuck_bugs_count") >= 1 and _is_flag_on(row, "workaround_available_flag") is False
    if hard_block:
        support_reasons.append("hard block without workaround")
        support_actions.append("treat as blocked workflow, confirm severity, document customer impact and define update cadence")

    if support_reasons:
        routing.append({
            "team": "Customer Support",
            "type": "support",
            "why": ", ".join(support_reasons[:4]),
            "action": "; ".join(support_actions[:3]),
            "urgency": "High" if hard_block or _is_flag_on(row, "data_accuracy_bug_flag") else "Medium",
        })

    ae_reasons = []
    ae_actions = []

    if _is_flag_on(row, "pricing_objection_flag"):
        ae_reasons.append("pricing objection")
        ae_actions.append("review pricing concern, commercial options and value narrative")

    if _is_flag_on(row, "limit_upgrade_recommended_flag"):
        ae_reasons.append("plan or limit upgrade may be needed")
        ae_actions.append("assess whether current package limits fit actual usage needs")

    if _is_flag_on(row, "downgrade_risk_flag"):
        ae_reasons.append("downgrade risk")
        ae_actions.append("prepare downgrade prevention or right-sizing conversation")

    if _is_flag_on(row, "expansion_opportunity_flag") and row.get("risk_level", "") != "High":
        ae_reasons.append("possible expansion opportunity")
        ae_actions.append("review whether additional toolkit, limits or seats would unlock more value")

    if _pct(row, "renewal_due_days", 999) < 45 and str(row.get("renewal_intent", "positive")) in ["neutral", "negative"]:
        ae_reasons.append("commercially sensitive renewal")
        ae_actions.append("confirm contract terms, renewal options, package scope, purchased add-ons and commercial commitments before customer follow-up")

    if _pct(row, "billing_tickets_last_quarter") >= 2:
        ae_reasons.append("billing or contract questions")
        ae_actions.append("validate contract details, invoice-related questions and purchased package scope")

    if ae_reasons:
        routing.append({
            "team": "Account Executive / Sales",
            "type": "sales",
            "why": ", ".join(ae_reasons[:4]),
            "action": "; ".join(ae_actions[:3]),
            "urgency": "High" if _pct(row, "renewal_due_days", 999) < 30 and str(row.get("renewal_intent", "positive")) != "positive" else "Medium",
        })

    product_reasons = []
    product_actions = []

    if _pct(row, "critical_feedback_count_90d") >= 1 and _pct(row, "roadmap_alignment_score", 100) < 45:
        product_reasons.append("critical product feedback with low roadmap alignment")
        product_actions.append("confirm roadmap status, alternative workflow or workaround")

    if _is_flag_on(row, "data_accuracy_bug_flag") and _pct(row, "stuck_bugs_count") >= 1:
        product_reasons.append("stuck data accuracy bug")
        product_actions.append("confirm investigation status, customer impact and expected update timeline")

    if _is_flag_on(row, "feedback_linked_to_renewal"):
        product_reasons.append("product feedback linked to renewal")
        product_actions.append("provide roadmap or workaround context before renewal conversation")

    if _pct(row, "oldest_open_bug_days") > 45 and _pct(row, "open_bugs_count") > 0:
        product_reasons.append("old unresolved bug")
        product_actions.append("review bug status and decide whether an escalation path or workaround is needed")

    if product_reasons:
        routing.append({
            "team": "Product / Engineering",
            "type": "product",
            "why": ", ".join(product_reasons[:4]),
            "action": "; ".join(product_actions[:3]),
            "urgency": "High" if _is_flag_on(row, "feedback_linked_to_renewal") or _is_flag_on(row, "data_accuracy_bug_flag") else "Medium",
        })

    if not routing:
        routing.append({
            "team": "CSM only",
            "type": "csm",
            "why": "no cross-functional dependency detected",
            "action": "continue proactive monitoring, value reinforcement and adoption coaching",
            "urgency": "Low",
        })

    return routing


def department_routing_html(row):
    routing = build_department_routing(row)
    cards = []
    for item in routing:
        urgency = str(item.get("urgency", "Medium"))
        urgency_class = "urgency-high" if urgency == "High" else "urgency-medium" if urgency == "Medium" else "urgency-low"
        card_type = html.escape(str(item.get("type", "csm")))
        cards.append(
            f'<div class="routing-card {card_type}">'
            f'<div class="urgency-pill {urgency_class}">{html.escape(urgency)} urgency</div>'
            f'<div class="routing-team">{html.escape(str(item.get("team", "")))}</div>'
            f'<div class="routing-kicker">Why involve them</div>'
            f'<div class="routing-body">{html.escape(str(item.get("why", "")))}</div>'
            f'<div class="routing-kicker">CSM handoff ask</div>'
            f'<div class="routing-body">{html.escape(str(item.get("action", "")))}</div>'
            f'</div>'
        )
    return '<div class="routing-grid">' + ''.join(cards) + '</div>'


def build_handoff_message(row):
    routing = build_department_routing(row)
    account = row.get("account_name", row.get("account_id", "this account"))
    risk = f"{row.get('churn_probability', 0):.1%}" if "churn_probability" in row else "elevated"
    drivers = ", ".join(top_drivers(row, 5)) or "general churn risk"
    lines = [
        f"Account: {account} ({row.get('account_id', 'N/A')})",
        f"Context: predicted churn risk is {risk}. Main visible drivers: {drivers}.",
        "",
        "Requested support:",
    ]
    for item in routing:
        if item.get("team") == "CSM only":
            continue
        lines.append(f"- {item['team']}: {item['action']}")
    if len(lines) == 4:
        lines.append("- CSM: continue proactive monitoring and value reinforcement; no cross-functional dependency detected yet.")
    lines.append("")
    lines.append("Goal: give the customer a clear next step, reduce friction before the next touchpoint, and protect renewal confidence.")
    return "\n".join(lines)


def route_filter_matches(row, routing_filter):
    if routing_filter == "All":
        return True
    teams = {item.get("team") for item in build_department_routing(row)}
    if routing_filter == "Needs Customer Support":
        return "Customer Support" in teams
    if routing_filter == "Needs Account Executive / Sales":
        return "Account Executive / Sales" in teams
    if routing_filter == "Needs Product / Engineering":
        return "Product / Engineering" in teams
    if routing_filter == "CSM only":
        return teams == {"CSM only"}
    return True


def routing_summary_counts(df):
    counts = {
        "Customer Support": 0,
        "Account Executive / Sales": 0,
        "Product / Engineering": 0,
        "CSM only": 0,
    }
    for _, r in df.iterrows():
        teams = {item.get("team") for item in build_department_routing(r)}
        for k in counts:
            if k in teams:
                counts[k] += 1
    return counts


def routing_summary_html(df):
    counts = routing_summary_counts(df)
    total = max(len(df), 1)
    def card(label, value, note):
        pct = value / total
        return (
            f'<div class="queue-card">'
            f'<div class="queue-label">{html.escape(label)}</div>'
            f'<div class="queue-value">{value}</div>'
            f'<div class="queue-note">{pct:.1%} of current portfolio · {html.escape(note)}</div>'
            f'</div>'
        )
    return '<div class="queue-grid">' + ''.join([
        card("Needs Support", counts["Customer Support"], "tool issues, limits, navigation or hard blocks"),
        card("Needs AE / Sales", counts["Account Executive / Sales"], "pricing, package fit or contract questions"),
        card("Needs Product", counts["Product / Engineering"], "roadmap, stuck bugs or product gaps"),
        card("CSM only", counts["CSM only"], "no cross-functional dependency detected"),
    ]) + '</div>'


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



def _impact_level(delta_pp: float) -> str:
    improvement = abs(float(delta_pp))
    if improvement >= 15:
        return "high"
    if improvement >= 6:
        return "medium"
    return "low"


def _owner_class(owner: str) -> str:
    o = owner.lower()
    if "support" in o:
        return "support"
    if "sales" in o or "account executive" in o or "ae" in o:
        return "sales"
    if "product" in o or "engineering" in o:
        return "product"
    return "csm"


def _tooltip(title: str, what: str, problem: str, action: str, owner: str) -> str:
    content = (
        f"<strong>{html.escape(title)}</strong><br>"
        f"<strong>What it means:</strong> {html.escape(what)}<br>"
        f"<strong>When it is a problem:</strong> {html.escape(problem)}<br>"
        f"<strong>Suggested action:</strong> {html.escape(action)}<br>"
        f"<strong>Who may help:</strong> {html.escape(owner)}"
    )
    return f'<span class="tooltip-wrap">? <span class="tooltip-box">{content}</span></span>'


def _status_from_threshold(value, risk_threshold, watch_threshold=None, inverse=False):
    """Return visual status for numeric indicators.

    inverse=False: higher is worse. inverse=True: lower is worse.
    """
    try:
        v = float(value)
    except Exception:
        return "watch", "Review"
    if watch_threshold is None:
        watch_threshold = risk_threshold * 0.75
    if inverse:
        if v <= risk_threshold:
            return "risk", "Risk"
        if v <= watch_threshold:
            return "watch", "Watch"
        return "good", "Healthy"
    if v >= risk_threshold:
        return "risk", "Risk"
    if v >= watch_threshold:
        return "watch", "Watch"
    return "good", "Healthy"


def _value_pct(row, key):
    try:
        return max(0.0, min(100.0, float(row.get(key, 0))))
    except Exception:
        return 0.0


def _render_signal_card(title, value, status_class, status_label, progress_pct, what, problem, action, owner, note=""):
    owner_html = "".join([f'<span class="ux-owner {_owner_class(x)}">{html.escape(x)}</span>' for x in owner.split(" + ")])
    tooltip = _tooltip(title, what, problem, action, owner)
    progress = max(0, min(100, float(progress_pct)))
    body_note = html.escape(note) if note else html.escape(what)
    return (
        f'<div class="signal-brief-card {status_class}">'
        f'<div class="signal-name">{html.escape(title)} {tooltip}</div>'
        f'<div class="signal-status {status_class}">{html.escape(status_label)}</div>'
        f'<div class="signal-value-big">{html.escape(str(value))}</div>'
        f'<div class="signal-progress"><div class="signal-progress-fill {status_class}" style="width:{progress:.1f}%;"></div></div>'
        f'<div class="ux-body">{body_note}</div>'
        f'<div class="signal-cta"><strong>Suggested action:</strong> {html.escape(action)}<br>{owner_html}</div>'
        f'</div>'
    )


def render_signal_briefing(row):
    """Human-readable Semrush signal briefing for non-technical CSM users."""
    usage_cards = []

    # Site Audit activity: lower last-run days is better; 999 means not active.
    last_run = float(row.get("site_audit_last_run_days", 999))
    if last_run > 60:
        st_cls, st_label, progress = "risk", "Inactive", 15
    elif last_run > 21:
        st_cls, st_label, progress = "watch", "Needs review", 55
    else:
        st_cls, st_label, progress = "good", "Active", 90
    usage_cards.append(_render_signal_card(
        "Site Audit activity",
        f"{int(last_run)} days ago" if last_run < 900 else "Not recently used",
        st_cls, st_label, progress,
        "Shows when the customer last ran Site Audit. Regular audits help the customer see technical SEO value.",
        "If Site Audit has not run recently, the customer may not be getting ongoing technical SEO insights.",
        "Ask whether Site Audit is part of their workflow. If not, run a setup/value session around audit scope, frequency and reporting.",
        "CSM + Customer Support",
        "Checks whether technical SEO monitoring is actually part of the customer workflow."
    ))

    pt_campaigns = int(row.get("position_tracking_campaigns_active", 0))
    if pt_campaigns == 0:
        st_cls, st_label, progress = "risk", "Not active", 10
    elif pt_campaigns <= 1:
        st_cls, st_label, progress = "watch", "Limited", 50
    else:
        st_cls, st_label, progress = "good", "Active", 85
    usage_cards.append(_render_signal_card(
        "Position Tracking setup",
        f"{pt_campaigns} campaigns",
        st_cls, st_label, progress,
        "Shows whether the customer is tracking rankings through active campaigns.",
        "No or very few campaigns can mean the customer is not monitoring the keyword outcomes they care about.",
        "Review campaign setup: location, device, search engine, competitors and keyword list. Confirm it matches their business goals.",
        "CSM + Customer Support",
        "Shows whether ranking monitoring is configured and connected to customer goals."
    ))

    ai_active = int(row.get("ai_visibility_active", 0)) == 1
    ai_prompts = int(row.get("ai_visibility_prompts_tracked", 0))
    ai_limit = max(1, int(row.get("ai_visibility_prompt_limit", 0)))
    ai_pct = ai_prompts / ai_limit * 100 if ai_active else 0
    if not ai_active:
        st_cls, st_label, progress = "good", "Not purchased", 0
        value = "Not active"
        action = "No action unless AI Visibility is part of the customer goals. If relevant, discuss fit with the commercial owner."
        owner = "CSM + Account Executive / Sales"
    elif ai_prompts < 10:
        st_cls, st_label, progress = "risk", "Underused", max(8, ai_pct)
        value = f"{ai_prompts}/{ai_limit} prompts"
        action = "Book an AI Visibility workflow session: define priority prompts, select competitors and set a recurring AI search review."
        owner = "CSM"
    else:
        st_cls, st_label, progress = "good", "In use", min(100, ai_pct)
        value = f"{ai_prompts}/{ai_limit} prompts"
        action = "Use current prompt tracking as a value proof point in the next business review."
        owner = "CSM"
    usage_cards.append(_render_signal_card(
        "AI Visibility adoption",
        value,
        st_cls, st_label, progress,
        "Shows whether the customer is using prompt tracking and AI search visibility workflows.",
        "If AI Visibility is active but prompt tracking is underused, the customer may not see enough value from the toolkit.",
        action,
        owner,
        "Checks whether AI Visibility is becoming a real workflow or only a purchased feature."
    ))

    local_active = int(row.get("local_locations_active", 0))
    local_target = max(0, int(row.get("local_locations_target", 0)))
    if local_target == 0:
        st_cls, st_label, progress = "good", "Not applicable", 0
        value = "Not a local setup"
    else:
        local_pct = local_active / max(1, local_target) * 100
        if local_pct < 50:
            st_cls, st_label, progress = "risk", "Low activation", local_pct
        elif local_pct < 80:
            st_cls, st_label, progress = "watch", "Partial", local_pct
        else:
            st_cls, st_label, progress = "good", "Healthy", local_pct
        value = f"{local_active}/{local_target} locations"
    usage_cards.append(_render_signal_card(
        "Local locations activation",
        value,
        st_cls, st_label, progress,
        "Shows whether purchased/target local locations are active and being managed.",
        "Low activation can mean the customer is paying for local workflows that are not fully live.",
        "Review inactive or frozen locations, confirm Listing Management setup and activate Review Management workflows.",
        "CSM + Customer Support",
        "Highlights whether local value is actually activated, not just contracted."
    ))

    limit_cards = []
    crawl_pct = _value_pct(row, "crawl_budget_usage_pct")
    cls, label = _status_from_threshold(crawl_pct, 85, 65)
    limit_cards.append(_render_signal_card(
        "Crawl budget usage",
        f"{crawl_pct:.1f}%",
        cls, label, crawl_pct,
        "Shows how much of the account's available Site Audit crawl capacity is being used.",
        "High usage can be healthy adoption, but it may also create limit frustration if the customer cannot audit enough pages.",
        "Review Site Audit scope, excluded pages, audit frequency and whether the current plan limit fits website size.",
        "Customer Support + Account Executive / Sales",
        "Useful to distinguish healthy usage from plan-limit friction."
    ))

    kw_pct = _value_pct(row, "keyword_tracking_usage_pct")
    cls, label = _status_from_threshold(kw_pct, 85, 65)
    limit_cards.append(_render_signal_card(
        "Keyword tracking usage",
        f"{kw_pct:.1f}%",
        cls, label, kw_pct,
        "Shows how much of the account's Position Tracking keyword capacity is being used.",
        "High usage may indicate that the customer is constrained by limits or needs help prioritising campaigns/keywords.",
        "Review keyword allocation, inactive campaigns, locations/devices and whether the current tier fits tracking needs.",
        "Customer Support + Account Executive / Sales",
        "Helps detect when the customer is blocked by tracking capacity rather than lack of interest."
    ))

    prompt_pct = _value_pct(row, "ai_visibility_prompt_usage_pct")
    cls, label = _status_from_threshold(prompt_pct, 85, 65)
    if not ai_active:
        cls, label, prompt_pct = "good", "Not applicable", 0
    limit_cards.append(_render_signal_card(
        "AI prompt limit usage",
        f"{prompt_pct:.1f}%" if ai_active else "Not active",
        cls, label, prompt_pct,
        "Shows how much of the AI Visibility prompt tracking allowance is being used.",
        "Near-limit usage can create confusion if the customer expects to track more prompts than contracted.",
        "Clarify prompt limits, current usage and whether the customer needs a workflow redesign or package review.",
        "Customer Support + Account Executive / Sales",
        "Important for explaining AI Visibility limits in plain customer terms."
    ))

    api_pct = _value_pct(row, "api_units_usage_pct") if "api_units_usage_pct" in row.index else 0
    api_available = int(row.get("api_access_available", 0)) == 1
    api_used = int(row.get("api_access_used", 0)) == 1
    if not api_available:
        cls, label, progress, value = "good", "Not included", 0, "Not included"
        api_action = "No action unless API is part of the customer requirements. If needed, involve Sales to review package fit."
        api_owner = "CSM + Account Executive / Sales"
    elif not api_used:
        cls, label, progress, value = "watch", "Available but unused", 35, "Not used"
        api_action = "Explore whether API access could automate reporting or reduce manual work for this account."
        api_owner = "CSM + Solutions Consultant"
    else:
        cls, label = _status_from_threshold(api_pct, 85, 65)
        progress, value = api_pct, f"{api_pct:.1f}%"
        api_action = "If usage is high, help the customer understand API unit consumption and involve Sales if additional capacity is needed."
        api_owner = "Customer Support + Account Executive / Sales"
    limit_cards.append(_render_signal_card(
        "API access / units",
        value,
        cls, label, progress,
        "Shows whether API access is available and whether usage may be creating capacity pressure.",
        "Unused API access can be missed value; saturated API usage can become a limit or pricing conversation.",
        api_action,
        api_owner,
        "Useful for advanced accounts where automation and reporting workflows matter."
    ))

    trust_cards = []
    roadmap = float(row.get("roadmap_alignment_score", 100))
    cls, label = _status_from_threshold(roadmap, 40, 60, inverse=True)
    trust_cards.append(_render_signal_card(
        "Product fit / roadmap alignment",
        f"{roadmap:.1f}/100",
        cls, label, roadmap,
        "Shows whether the customer's needs appear aligned with the product direction or roadmap.",
        "Low alignment means the customer may feel the product is not evolving in the direction they need.",
        "Clarify workarounds, manage expectations and involve Product if the feedback is renewal-critical.",
        "CSM + Product / Engineering",
        "Shows whether churn risk is coming from product gap, not just service experience."
    ))

    feedback = int(row.get("unresolved_feedback_count", 0))
    critical_feedback = int(row.get("critical_feedback_count_90d", 0))
    if critical_feedback > 0 or feedback >= 3:
        cls, label, progress = "risk", "Open friction", min(100, 25 + feedback * 12 + critical_feedback * 20)
    elif feedback > 0:
        cls, label, progress = "watch", "Some friction", min(100, 25 + feedback * 12)
    else:
        cls, label, progress = "good", "Clear", 5
    trust_cards.append(_render_signal_card(
        "Product feedback pressure",
        f"{feedback} unresolved · {critical_feedback} critical",
        cls, label, progress,
        "Shows whether the customer has unresolved requests, feature gaps or recalculation needs.",
        "Repeated or critical feedback can become renewal risk if there is no workaround or roadmap clarity.",
        "Separate support issue from product gap. Confirm workaround, roadmap status and renewal dependency.",
        "CSM + Product / Engineering",
        "Keeps product gaps visible without overloading the CSM with raw feedback records."
    ))

    open_bugs = int(row.get("open_bugs_count", 0))
    stuck_bugs = int(row.get("stuck_bugs_count", 0))
    data_bug = int(row.get("data_accuracy_bug_flag", 0)) == 1
    if data_bug or stuck_bugs > 0:
        cls, label, progress = "risk", "Trust risk", min(100, 45 + stuck_bugs * 25 + (25 if data_bug else 0))
    elif open_bugs > 0:
        cls, label, progress = "watch", "Needs follow-up", min(100, 25 + open_bugs * 12)
    else:
        cls, label, progress = "good", "No visible blocker", 5
    trust_cards.append(_render_signal_card(
        "Bugs / data trust",
        f"{open_bugs} open · {stuck_bugs} stuck",
        cls, label, progress,
        "Shows whether open or stuck bugs are affecting product reliability or data confidence.",
        "Data accuracy issues can damage trust faster than normal support friction, especially near renewal.",
        "Escalate as a trust-risk issue. Confirm impact, ETA, workaround and customer update cadence.",
        "Customer Support + Product / Engineering",
        "Highlights whether the customer can trust the product enough to stay."
    ))

    workaround = int(row.get("workaround_available_flag", 1)) == 1
    if workaround:
        cls, label, progress, value = "good", "Available", 85, "Yes"
    elif stuck_bugs > 0 or data_bug:
        cls, label, progress, value = "risk", "Missing", 15, "No"
    else:
        cls, label, progress, value = "watch", "Check", 50, "Unknown/No"
    trust_cards.append(_render_signal_card(
        "Workaround availability",
        value,
        cls, label, progress,
        "Shows whether there is a practical temporary path while a bug or product gap is unresolved.",
        "No workaround makes a support or product issue much more urgent for the customer.",
        "Ask Support/Product to document the workaround or confirm that none exists, then set clear customer expectations.",
        "Customer Support + Product / Engineering",
        "Turns a technical issue into a clear expectation-management action."
    ))

    def section(title, sub, cards):
        st.markdown(
            f'<div class="signal-section"><div class="signal-section-title">{html.escape(title)}</div>'
            f'<div class="signal-section-sub">{html.escape(sub)}</div>'
            f'<div class="signal-brief-grid">{"".join(cards)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="briefing-note"><strong>How to read this section:</strong> these cards translate Semrush-like fields into plain CSM language. Hover over the <strong>?</strong> icon on each card to understand the metric, why it matters, what the problem may be, and which team can help.</div>',
        unsafe_allow_html=True,
    )
    section("Usage & adoption", "Is the customer actively using the workflows they pay for?", usage_cards)
    section("Limits & friction", "Is the customer blocked by package limits, allocations or configuration expectations?", limit_cards)
    section("Product fit & trust", "Does the customer believe the product fits their needs and can be trusted?", trust_cards)


def simulate_scenarios_raw(row, model, model_columns):
    base = pd.DataFrame([row.drop(labels=[c for c in ["churn_probability", "risk_level", "priority"] if c in row.index])])
    current = float(row["churn_probability"])
    scenarios = []

    def add(name, lever, description, changes, cta, owners, tooltip_title, tooltip_what, tooltip_problem):
        simulated = base.copy()
        for k, v in changes.items():
            if k in simulated.columns:
                simulated.loc[simulated.index[0], k] = v(simulated.loc[simulated.index[0], k]) if callable(v) else v
        new_prob = float(predict_for_df(model, model_columns, simulated)[0])
        delta = new_prob - current
        scenarios.append({
            "Scenario": name,
            "Lever": lever,
            "What changes": description,
            "Current probability": current,
            "Simulated probability": new_prob,
            "Change": delta,
            "New risk level": risk_level(new_prob),
            "CTA": cta,
            "Owners": owners,
            "Tooltip title": tooltip_title,
            "Tooltip what": tooltip_what,
            "Tooltip problem": tooltip_problem,
        })

    add(
        "Improve Site Audit & usage",
        "Adoption recovery",
        "Increase activity/adoption and reduce inactivity",
        {
            "tool_activity_score": lambda x: min(88, max(x, 72)),
            "feature_adoption_score": lambda x: min(85, max(x, 70)),
            "days_since_last_login": lambda x: min(x, 5),
            "site_audit_last_run_days": lambda x: min(x, 7),
        },
        "Run a workflow recovery session focused on Site Audit, Position Tracking and reporting habits.",
        "CSM + Customer Support",
        "Improve usage and adoption",
        "Tests how risk might change if the customer starts using the core workflows more consistently.",
        "If this has the highest impact, churn risk is likely driven by low value realization rather than only bugs or pricing."
    )
    add(
        "Resolve limit friction",
        "Limits and package fit",
        "Reduce crawl/keyword/prompt saturation and mark upgrade-fit action",
        {
            "crawl_budget_usage_pct": lambda x: min(x, 72),
            "keyword_tracking_usage_pct": lambda x: min(x, 72),
            "ai_visibility_prompt_usage_pct": lambda x: min(x, 72),
            "limit_upgrade_recommended_flag": 0,
        },
        "Clarify limits, allocation and package fit. Involve Sales if the issue is actually capacity, not configuration.",
        "Customer Support + Account Executive / Sales",
        "Resolve limit friction",
        "Tests the impact of reducing saturation around crawl, keyword tracking or AI prompt limits.",
        "If this matters, the customer may be active but frustrated by plan limits or allocation expectations."
    )
    add(
        "Strengthen renewal signal",
        "Commercial confidence",
        "Move renewal intent to positive and remove grace/pricing/downgrade risk",
        {
            "renewal_intent": "positive",
            "in_grace_period": 0,
            "pricing_objection_flag": 0,
            "downgrade_risk_flag": 0,
        },
        "Align with the commercial owner before the next customer conversation and clarify decision criteria.",
        "CSM + Account Executive / Sales",
        "Strengthen renewal signal",
        "Tests how risk might change if the commercial conversation becomes positive.",
        "If this has high impact, urgency is tied to renewal timing, pricing, contract fit or downgrade risk."
    )
    add(
        "Resolve product feedback",
        "Product fit",
        "Close critical feedback and improve roadmap alignment/sentiment",
        {
            "critical_feedback_count_90d": 0,
            "unresolved_feedback_count": lambda x: max(0, min(x, 1)),
            "avg_unresolved_feedback_age_days": lambda x: min(x, 14),
            "roadmap_alignment_score": lambda x: max(x, 75),
            "feedback_sentiment_score": lambda x: max(x, 0.35),
            "feedback_linked_to_renewal": 0,
        },
        "Confirm workaround, roadmap status and expectation-management message before renewal.",
        "CSM + Product / Engineering",
        "Resolve product feedback",
        "Tests the impact of reducing unresolved product gaps and improving roadmap fit.",
        "If this is important, the risk is probably about unmet product needs, not just usage or support quality."
    )
    add(
        "Resolve bug reliability issues",
        "Trust recovery",
        "Remove stuck/data-trust bugs and set workaround available",
        {
            "critical_bugs_reported_90d": 0,
            "stuck_bugs_count": 0,
            "open_bugs_count": lambda x: max(0, min(x, 1)),
            "oldest_open_bug_days": lambda x: min(x, 7),
            "data_accuracy_bug_flag": 0,
            "workaround_available_flag": 1,
            "bug_linked_to_renewal_flag": 0,
            "bug_sentiment_score": lambda x: max(x, 0.35),
        },
        "Escalate as trust-risk: confirm impact, workaround, ETA and update cadence with Support/Product.",
        "Customer Support + Product / Engineering",
        "Resolve reliability issues",
        "Tests how risk might change if stuck bugs and data-trust problems are solved.",
        "If this has high impact, the customer may trust the platform less because of reliability or data accuracy concerns."
    )
    add(
        "Best-case recovery package",
        "Combined action plan",
        "Combine adoption, renewal, feedback and bug recovery",
        {
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
        },
        "Create a coordinated recovery plan with CSM, Support, Sales and Product focused on value, renewal and blockers.",
        "CSM + Support + Sales + Product",
        "Best-case recovery package",
        "Shows the potential impact of acting across multiple risk families at once.",
        "This is not a promise. It helps the CSM explain which combination of actions might create the strongest recovery path."
    )
    return pd.DataFrame(scenarios)


def render_scenario_story(row, model, model_columns):
    raw_scenarios = simulate_scenarios_raw(row, model, model_columns)
    if raw_scenarios.empty:
        st.info("No scenario data available for this account.")
        return

    raw_scenarios["Improvement pp"] = (raw_scenarios["Current probability"] - raw_scenarios["Simulated probability"]) * 100
    best = raw_scenarios.sort_values("Improvement pp", ascending=False).iloc[0]
    current = float(best["Current probability"])
    after = float(best["Simulated probability"])
    improvement = float(best["Improvement pp"])
    impact_class = _impact_level(improvement)

    owners_html = "".join([f'<span class="ux-owner {_owner_class(x.strip())}">{html.escape(x.strip())}</span>' for x in str(best["Owners"]).split("+")])
    st.markdown(
        f'<div class="scenario-hero">'
        f'<div class="ux-kicker">Best lever to reduce risk</div>'
        f'<div class="ux-title">{html.escape(str(best["Scenario"]))}</div>'
        f'<div class="ux-body">{html.escape(str(best["Tooltip problem"]))}</div>'
        f'<div class="scenario-metrics">'
        f'<div class="scenario-metric"><div class="scenario-label">Current risk</div><div class="scenario-value">{current:.1%}</div></div>'
        f'<div class="scenario-metric"><div class="scenario-label">Potential risk</div><div class="scenario-value">{after:.1%}</div></div>'
        f'<div class="scenario-metric"><div class="scenario-label">Potential impact</div><div class="scenario-value">-{improvement:.1f} pp</div></div>'
        f'<div class="scenario-metric"><div class="scenario-label">New level</div><div class="scenario-value">{html.escape(str(best["New risk level"]))}</div></div>'
        f'</div>'
        f'<div class="signal-cta" style="margin-top:14px;"><strong>Suggested CSM action:</strong> {html.escape(str(best["CTA"]))}<br>{owners_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cards = []
    for _, r in raw_scenarios.sort_values("Improvement pp", ascending=False).iterrows():
        imp = max(0, float(r["Improvement pp"]))
        cls = _impact_level(imp)
        owners = "".join([f'<span class="ux-owner {_owner_class(x.strip())}">{html.escape(x.strip())}</span>' for x in str(r["Owners"]).split("+")])
        tooltip = _tooltip(
            str(r["Tooltip title"]),
            str(r["Tooltip what"]),
            str(r["Tooltip problem"]),
            str(r["CTA"]),
            str(r["Owners"]),
        )
        cards.append(
            f'<div class="scenario-card {"best" if cls == "high" else cls}">'
            f'<div class="ux-kicker">{html.escape(str(r["Lever"]))}</div>'
            f'<div class="ux-title">{html.escape(str(r["Scenario"]))} {tooltip}</div>'
            f'<span class="ux-impact {cls}">{"High" if cls == "high" else "Medium" if cls == "medium" else "Low"} potential impact</span>'
            f'<div class="ux-big">{float(r["Simulated probability"]):.1%}</div>'
            f'<div class="ux-body"><strong>Change:</strong> {float(r["Change"])*100:+.1f} pp from current {float(r["Current probability"]):.1%}. New level: <strong>{html.escape(str(r["New risk level"]))}</strong>.</div>'
            f'<div class="signal-cta"><strong>CTA:</strong> {html.escape(str(r["CTA"]))}<br>{owners}</div>'
            f'</div>'
        )
    st.markdown('<div class="ux-grid-3">' + ''.join(cards) + '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="briefing-note"><strong>How to use this:</strong> scenarios are not guarantees. They help the CSM compare which lever may reduce risk fastest and which departments should be involved before the next customer touchpoint.</div>',
        unsafe_allow_html=True,
    )


def simulate_scenarios(row, model, model_columns):
    """Formatted version kept for compatibility and export/debugging."""
    out = simulate_scenarios_raw(row, model, model_columns).copy()
    out["Current probability"] = out["Current probability"].map(lambda x: f"{x:.1%}")
    out["Simulated probability"] = out["Simulated probability"].map(lambda x: f"{x:.1%}")
    out["Change"] = out["Change"].map(lambda x: f"{x*100:+.1f} pp")
    return out[["Scenario", "What changes", "Current probability", "Simulated probability", "Change", "New risk level", "CTA", "Owners"]]


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
routing_options = ["All", "Needs Customer Support", "Needs Account Executive / Sales", "Needs Product / Engineering", "CSM only"]
routing_filter = st.sidebar.selectbox("Action routing", routing_options)

filtered = df.copy()
if segment_filter != "All": filtered = filtered[filtered["segment"] == segment_filter]
if family_filter != "All": filtered = filtered[filtered["plan_family"] == family_filter]
if plan_filter != "All": filtered = filtered[filtered["plan_tier"] == plan_filter]
if risk_filter != "All": filtered = filtered[filtered["risk_level"] == risk_filter]
if region_filter != "All": filtered = filtered[filtered["region"] == region_filter]
if only_renewal: filtered = filtered[filtered["renewal_due_days"] < 60]
if only_limit: filtered = filtered[(filtered["crawl_budget_usage_pct"] > 85) | (filtered["keyword_tracking_usage_pct"] > 85) | (filtered["ai_visibility_prompt_usage_pct"] > 85)]
if only_bugs: filtered = filtered[(filtered["stuck_bugs_count"] > 0) | (filtered["data_accuracy_bug_flag"] == 1)]
if routing_filter != "All":
    filtered = filtered[filtered.apply(lambda r: route_filter_matches(r, routing_filter), axis=1)]
filtered = filtered.sort_values("churn_probability", ascending=False).reset_index(drop=True)

if filtered.empty:
    st.warning("No accounts match the selected filters.")
    st.stop()


# =========================
# V4 CSM Mission Control UX
# =========================

avg_churn = filtered["churn_probability"].mean()
high_count = int((filtered["risk_level"] == "High").sum())
medium_count = int((filtered["risk_level"] == "Medium").sum())
low_count = int((filtered["risk_level"] == "Low").sum())
mrr_at_risk = filtered.loc[filtered["risk_level"] == "High", "mrr"].sum()
renewal_pressure = int(((filtered["renewal_due_days"] < 60) & (filtered["renewal_intent"] != "positive")).sum())
renewal_pressure_30 = int(((filtered["renewal_due_days"] < 30) & (filtered["renewal_intent"] != "positive")).sum())
limit_pressure = int(((filtered["crawl_budget_usage_pct"] > 85) | (filtered["keyword_tracking_usage_pct"] > 85) | (filtered["ai_visibility_prompt_usage_pct"] > 85)).sum())
low_usage_count = int(((filtered["tool_activity_score"] < 45) | (filtered["feature_adoption_score"] < 45)).sum())
product_friction_count = int(((filtered["stuck_bugs_count"] > 0) | (filtered["data_accuracy_bug_flag"] == 1) | (filtered["unresolved_feedback_count"] >= 3) | (filtered["roadmap_alignment_score"] < 40)).sum())
support_needed_count = int(filtered.apply(lambda r: route_filter_matches(r, "Needs Customer Support"), axis=1).sum())
sales_needed_count = int(filtered.apply(lambda r: route_filter_matches(r, "Needs Account Executive / Sales"), axis=1).sum())
product_needed_count = int(filtered.apply(lambda r: route_filter_matches(r, "Needs Product / Engineering"), axis=1).sum())
csm_only_count = int(filtered.apply(lambda r: route_filter_matches(r, "CSM only"), axis=1).sum())

family_scores_df = pd.DataFrame([family_risk_scores(r) for _, r in filtered.iterrows()])
avg_family_scores = family_scores_df.mean().sort_values(ascending=False) if not family_scores_df.empty else pd.Series(dtype=float)
primary_theme = avg_family_scores.index[0] if len(avg_family_scores) else "No theme detected"
secondary_theme = avg_family_scores.index[1] if len(avg_family_scores) > 1 else "No secondary theme"
primary_theme_score = float(avg_family_scores.iloc[0]) if len(avg_family_scores) else 0.0
highest_segment = filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=False).index[0] if len(filtered) else "N/A"
highest_plan_family = filtered.groupby("plan_family")["churn_probability"].mean().sort_values(ascending=False).index[0] if len(filtered) else "N/A"


def _fmt_pct(x):
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "N/A"


def _fmt_money(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "N/A"


def urgency_label(row):
    if row.get("risk_level") == "High" and row.get("renewal_due_days", 999) < 30:
        return "Act today"
    if row.get("risk_level") == "High":
        return "Act this week"
    if row.get("risk_level") == "Medium" and row.get("renewal_due_days", 999) < 60:
        return "Review this week"
    if row.get("risk_level") == "Medium":
        return "Monitor closely"
    return "Monitor"


def account_story_summary(row):
    drivers = top_drivers(row, 3)
    if not drivers:
        return "No major visible risk driver stands out. Continue regular value reinforcement."
    if len(drivers) == 1:
        driver_text = drivers[0]
    elif len(drivers) == 2:
        driver_text = f"{drivers[0]} and {drivers[1]}"
    else:
        driver_text = f"{drivers[0]}, {drivers[1]} and {drivers[2]}"
    return f"This account needs attention because the strongest visible signals are {driver_text}."


def snapshot_card_html(kicker, title, value, body, cta, tone="blue"):
    tone_color = {"blue":"#2563EB", "teal":"#0F766E", "red":"#DC2626", "amber":"#D97706", "purple":"#6D28D9"}.get(tone, "#2563EB")
    return (
        f'<div class="mission-card" style="border-top:5px solid {tone_color};">'
        f'<div class="mission-kicker">{html.escape(str(kicker))}</div>'
        f'<div class="mission-title">{html.escape(str(title))}</div>'
        f'<div class="mission-value">{html.escape(str(value))}</div>'
        f'<div class="mission-body">{html.escape(str(body))}</div>'
        f'<div class="mission-cta">{html.escape(str(cta))}</div>'
        f'</div>'
    )


def mission_guide_html():
    steps = [
        ("1", "Check Mission Control", "Start with the portfolio story and the biggest risk theme."),
        ("2", "Open Priority Queue", "Review P1 and P2 accounts before scanning tables."),
        ("3", "Read Account Briefing", "Understand the account story, owner, action and evidence."),
        ("4", "Use Simulator", "Compare which action could reduce risk fastest."),
        ("5", "Send handoff", "Involve Support, Sales or Product only when the signal requires it."),
    ]
    cards = "".join([
        f'<div class="guide-step"><div class="guide-number">{n}</div><div class="guide-title">{html.escape(t)}</div><div class="guide-body">{html.escape(b)}</div></div>'
        for n,t,b in steps
    ])
    return f'<div class="guide-grid">{cards}</div>'


def daily_brief_html():
    if high_count > 0:
        opening = f"Today, start with {high_count} high-risk accounts."
    elif medium_count > 0:
        opening = f"No high-risk accounts in this view, but {medium_count} medium-risk accounts deserve monitoring."
    else:
        opening = "This portfolio view currently looks stable."
    body = (
        f"{opening} The clearest portfolio theme is {primary_theme} ({primary_theme_score:.1f}/100), "
        f"with {secondary_theme} as the secondary signal. Commercial exposure is {_fmt_money(mrr_at_risk)} in high-risk MRR. "
        f"CSMs should first review P1/P2 accounts with renewal pressure, product friction, or cross-functional blockers."
    )
    return (
        f'<div class="daily-brief"><div class="daily-kicker">Today\'s CSM brief</div>'
        f'<div class="daily-title">Start with the accounts where risk is actionable.</div>'
        f'<div class="daily-body">{html.escape(body)}</div></div>'
    )


def mission_snapshot_grid_html():
    cards = [
        snapshot_card_html("Portfolio risk", "How healthy is my book?", f"{avg_churn:.1%}", f"Average churn probability across {len(filtered):,} selected accounts. Main risk theme: {primary_theme}.", "Open Priority Queue", "blue"),
        snapshot_card_html("Revenue & renewal", "Where is commercial risk?", _fmt_money(mrr_at_risk), f"{renewal_pressure_30} accounts renew in <30 days with weak or non-positive intent.", "Open renewal-risk details", "red"),
        snapshot_card_html("Adoption gap", "Who pays but underuses?", f"{low_usage_count}", "Accounts with low product activity or shallow feature adoption.", "Open adoption recovery list", "teal"),
        snapshot_card_html("Product friction", "Where is trust breaking?", f"{product_friction_count}", "Accounts with stuck bugs, data accuracy flags, unresolved feedback or roadmap mismatch.", "Open product friction cases", "amber"),
        snapshot_card_html("Team dependencies", "Who needs to help?", f"{support_needed_count + sales_needed_count + product_needed_count}", "Accounts where Support, Sales or Product may need to help the CSM unblock risk.", "Review cross-functional routing", "purple"),
    ]
    return f'<div class="mission-grid">{"".join(cards)}</div>'


def render_account_card(row, compact=False):
    actions, owners, urgency = semrush_csm_cta(row)
    routing = build_department_routing(row)
    owner_line = " + ".join([x.get("team", "") for x in routing[:3]]) if routing else "CSM"
    drivers = top_drivers(row, 3)
    chips = "".join([f'<span class="driver-chip">{html.escape(str(d))}</span>' for d in drivers])
    urgency_text = urgency_label(row)
    risk_css = "risk-high" if row["risk_level"] == "High" else "risk-medium" if row["risk_level"] == "Medium" else "risk-low"
    height = "min-height:300px;" if not compact else "min-height:245px;"
    return (
        f'<div class="account-queue-card" style="{height}">'
        f'<div class="account-queue-top"><span class="priority-pill">{html.escape(str(row["priority"]))}</span><span class="urgency-mini {risk_css}">{html.escape(urgency_text)}</span></div>'
        f'<div class="account-queue-title">{html.escape(str(row["account_name"]))}</div>'
        f'<div class="account-queue-sub">{html.escape(str(row["account_id"]))} · {html.escape(str(row["segment"]))} · {html.escape(str(row["plan_family"]))} · {html.escape(str(row["plan_tier"]))}</div>'
        f'<div class="account-risk-line"><span>{risk_badge(row["risk_level"])}</span><span class="account-risk-value">{row["churn_probability"]:.1%}</span></div>'
        f'<div class="account-queue-sub">MRR {_fmt_money(row["mrr"])} · Renewal in {int(row["renewal_due_days"])} days · Intent: {html.escape(str(row["renewal_intent"]))}</div>'
        f'<div style="margin:10px 0 8px;">{chips}</div>'
        f'<div class="account-action"><strong>CSM action:</strong> {html.escape(str(actions[0] if actions else "Review account health and confirm next best action."))}</div>'
        f'<div class="account-owner"><strong>Teams:</strong> {html.escape(owner_line)}</div>'
        f'</div>'
    )


def render_priority_cards(dataframe, max_cards=6):
    if dataframe.empty:
        st.info("No accounts in this queue for the current filters.")
        return
    cards = "".join([render_account_card(r, compact=True) for _, r in dataframe.head(max_cards).iterrows()])
    st.markdown(f'<div class="account-card-grid">{cards}</div>', unsafe_allow_html=True)


def three_question_html(row):
    usage_score = (float(row.get("tool_activity_score", 0)) + float(row.get("feature_adoption_score", 0))) / 2
    usage_status = "Yes" if usage_score >= 65 else "Partially" if usage_score >= 40 else "Not enough"
    usage_body = "The customer shows healthy product activity." if usage_score >= 65 else "Usage or adoption needs review before value is clear." if usage_score >= 40 else "The customer may not be realizing enough product value."

    fit_score = (float(row.get("roadmap_alignment_score", 50)) + max(0, 100 - float(row.get("unresolved_feedback_count", 0)) * 15)) / 2
    fit_status = "Mostly" if fit_score >= 65 else "Not fully" if fit_score >= 40 else "No clear fit"
    fit_body = "Current needs appear mostly aligned." if fit_score >= 65 else "Product feedback or roadmap gaps may need expectation management." if fit_score >= 40 else "Unmet product needs may be a major churn reason."

    trust_score = 100
    trust_score -= 25 if int(row.get("data_accuracy_bug_flag", 0)) == 1 else 0
    trust_score -= min(40, int(row.get("stuck_bugs_count", 0)) * 20)
    trust_score -= 20 if float(row.get("csat_support", 5)) < 3.2 else 0
    trust_status = "Yes" if trust_score >= 70 else "At risk" if trust_score >= 40 else "Weak"
    trust_body = "No major trust blocker is visible." if trust_score >= 70 else "Support, bug or data-trust signals need attention." if trust_score >= 40 else "Trust may be materially damaged by unresolved friction."

    items = [
        ("Is the customer using the product?", usage_status, usage_body, usage_score),
        ("Does the product meet their needs?", fit_status, fit_body, fit_score),
        ("Can they trust the product enough to stay?", trust_status, trust_body, trust_score),
    ]
    cards = "".join([
        f'<div class="question-card"><div class="question-title">{html.escape(q)}</div><div class="question-answer">{html.escape(ans)}</div><div class="question-body">{html.escape(body)}</div><div class="bar-wrap"><div class="bar-fill" style="width:{max(0,min(100,score)):.1f}%;"></div></div></div>'
        for q, ans, body, score in items
    ])
    return f'<div class="question-grid">{cards}</div>'


def render_selected_account_briefing(row, include_scenario=False):
    drivers = top_drivers(row)
    chips = ''.join([f'<span class="driver-chip">{html.escape(str(d))}</span>' for d in drivers])
    st.markdown(
        f'<div class="account-brief-hero">'
        f'<div class="metric-label">{html.escape(str(row["account_id"]))} · {html.escape(str(row["plan_family"]))} · {html.escape(str(row["plan_tier"]))}</div>'
        f'<div class="account-brief-title">{html.escape(str(row["account_name"]))}</div>'
        f'<div class="account-brief-sub">{html.escape(str(row["segment"]))} · {html.escape(str(row["industry"]))} · {html.escape(str(row["region"]))} · MRR {_fmt_money(row["mrr"])}</div>'
        f'<div class="account-risk-line"><span>{risk_badge(row["risk_level"])}</span><span class="account-risk-value">{row["churn_probability"]:.1%}</span></div>'
        f'<div class="account-story">{html.escape(account_story_summary(row))}</div>'
        f'<div>{chips}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    section_header("Three-question briefing", "A simple way for any CSM or stakeholder to read the account story.")
    st.markdown(three_question_html(row), unsafe_allow_html=True)

    section_header("Risk family breakdown", "What kind of risk is driving this account: adoption, renewal, support, feedback, bugs or limits.")
    scores = family_risk_scores(row)
    cols = st.columns(2)
    for idx, (label, value) in enumerate(scores.items()):
        with cols[idx % 2]:
            st.markdown(bar(label, value), unsafe_allow_html=True)

    section_header("Structured CSM action plan", "Direct instructions for the CSM and cross-functional owners.")
    st.markdown(playbook_html(row), unsafe_allow_html=True)

    section_header("Who needs to be involved?", "Cross-functional routing for blockers that the CSM should not solve alone.")
    st.markdown(department_routing_html(row), unsafe_allow_html=True)
    with st.expander("Suggested handoff message", expanded=False):
        st.code(build_handoff_message(row), language="text")

    section_header("Account signal briefing", "Plain-English evidence grouped by Usage, Limits, and Product trust.")
    render_signal_briefing(row)

    if include_scenario:
        section_header("Scenario Simulator", "What could reduce the risk fastest, and which team should help?")
        render_scenario_story(row, model, model_columns)
        with st.expander("View scenario calculation table", expanded=False):
            scenarios = simulate_scenarios(row, model, model_columns)
            st.dataframe(scenarios, use_container_width=True, hide_index=True)
            st.caption("Scenario results are model-based simulations on synthetic data. They are decision-support estimates, not guaranteed outcomes.")

    with st.expander("Supporting account data", expanded=False):
        key_cols = [
            "account_id","account_name","segment","region","plan_family","plan_tier","mrr","churn_probability","risk_level",
            "renewal_due_days","renewal_intent","tool_activity_score","feature_adoption_score","crawl_budget_usage_pct",
            "keyword_tracking_usage_pct","ai_visibility_prompt_usage_pct","roadmap_alignment_score","unresolved_feedback_count",
            "stuck_bugs_count","data_accuracy_bug_flag","csat_support","sentiment_csm"
        ]
        available_cols = [c for c in key_cols if c in row.index]
        evidence = pd.DataFrame({"Field": available_cols, "Value": [str(row[c]) for c in available_cols]})
        st.dataframe(evidence, use_container_width=True, hide_index=True)

    with st.expander("Raw account data", expanded=False):
        raw_df = pd.DataFrame({"Field": row.index, "Value": [str(v) for v in row.values]})
        st.dataframe(raw_df, use_container_width=True, hide_index=True)


# =========================
# Account 360° immersive briefing layer (V8)
# =========================

def _status_from_score(score: float, inverse: bool = False) -> tuple[str, str]:
    score = float(score)
    if inverse:
        if score >= 70:
            return "Healthy", "low"
        if score >= 40:
            return "Needs review", "medium"
        return "At risk", "high"
    if score >= 70:
        return "High pressure", "high"
    if score >= 40:
        return "Needs review", "medium"
    return "Controlled", "low"


def account_360_core_html(row: pd.Series) -> str:
    actions, owners, urgency = semrush_csm_cta(row)
    drivers = top_drivers(row, 5)
    driver_chips = "".join([f'<span class="driver-chip">{html.escape(str(d))}</span>' for d in drivers])
    routing = build_department_routing(row)
    teams = " + ".join([r.get("team", "") for r in routing[:3]]) if routing else "CSM"
    main_theme = dominant_account_theme(row)
    return f'''
    <div class="account360-shell">
        <div class="account360-core-card">
            <div class="account360-eyebrow">Account 360° cockpit</div>
            <div class="account360-title">{html.escape(str(row["account_name"]))}</div>
            <div class="account360-sub">{html.escape(str(row["account_id"]))} · {html.escape(str(row["segment"]))} · {html.escape(str(row["plan_family"]))} · {html.escape(str(row["plan_tier"]))}</div>
            <div class="account360-risk-row"><span>{risk_badge(row["risk_level"])}</span><span class="account360-prob">{float(row["churn_probability"]):.1%}</span></div>
            <div class="account360-story"><strong>Main account story:</strong> {html.escape(account_story_summary(row))}</div>
            <div>{driver_chips}</div>
        </div>
        <div class="account360-priority-card">
            <div class="account360-kicker">First action to take</div>
            <div class="account360-action-title">{html.escape(actions[0] if actions else "Review account health and confirm the next best action.")}</div>
            <div class="account360-action-meta">Main theme: <strong>{html.escape(main_theme)}</strong> · Urgency: <strong>{html.escape(urgency)}</strong></div>
            <div class="account360-action-meta">Teams likely needed: <strong>{html.escape(teams)}</strong></div>
        </div>
        <div class="account360-commercial-card">
            <div class="account360-kicker">Commercial context</div>
            <div class="account360-metric-row"><span>MRR</span><strong>{_fmt_money(row.get("mrr", 0))}</strong></div>
            <div class="account360-metric-row"><span>Renewal due</span><strong>{int(row.get("renewal_due_days", 0))} days</strong></div>
            <div class="account360-metric-row"><span>Renewal intent</span><strong>{html.escape(str(row.get("renewal_intent", "unknown")))}</strong></div>
            <div class="account360-metric-row"><span>Priority</span><strong>{html.escape(str(row.get("priority", "Monitor")))}</strong></div>
        </div>
    </div>
    '''


def account_360_radar_html(row: pd.Series) -> str:
    scores = family_risk_scores(row)
    ordered = [
        ("Usage & adoption", "Is the customer actively using the product?", "CSM"),
        ("Renewal & commercial", "Is the commercial timing sensitive?", "CSM + AE"),
        ("Support experience", "Is service friction weakening trust?", "Customer Support"),
        ("Feedback & roadmap", "Are unmet needs creating product-fit risk?", "CSM + Product"),
        ("Bugs & reliability", "Can the customer trust the product output?", "Support + Product"),
        ("Limit friction", "Is the customer hitting plan or tool limits?", "Support + AE"),
    ]
    positions = [
        ("50%", "5%", "#0F766E"),
        ("86%", "28%", "#DC2626"),
        ("78%", "72%", "#2563EB"),
        ("50%", "93%", "#6D28D9"),
        ("18%", "72%", "#D97706"),
        ("14%", "28%", "#0891B2"),
    ]
    nodes = []
    details = []
    for (label, meaning, owner), (left, top, color) in zip(ordered, positions):
        value = float(scores.get(label, 0))
        status, css = _status_from_score(value)
        nodes.append(
            f'<div class="account360-node {css}" style="left:{left}; top:{top}; --node-color:{color};" title="{html.escape(meaning)} Suggested owner: {html.escape(owner)}.">'
            f'<span class="account360-node-label">{html.escape(label)}</span>'
            f'<span class="account360-node-value">{value:.0f}</span>'
            f'</div>'
        )
        details.append(
            f'<div class="account360-radar-row">'
            f'<div><strong>{html.escape(label)}</strong><br><span>{html.escape(meaning)}</span></div>'
            f'<div class="account360-status {css}">{html.escape(status)}</div>'
            f'<div class="account360-mini-bar"><div style="width:{max(0,min(100,value)):.1f}%; background:{color};"></div></div>'
            f'<div class="account360-owner">{html.escape(owner)}</div>'
            f'</div>'
        )
    center = float(row.get("churn_probability", 0)) * 100
    return f'''
    <div class="account360-radar-shell">
        <div class="account360-radar-stage">
            <div class="account360-ring ring-a"></div>
            <div class="account360-ring ring-b"></div>
            <div class="account360-ring ring-c"></div>
            <div class="account360-center">
                <div class="core-label">Account risk</div>
                <div class="core-value">{center:.0f}%</div>
                <div class="core-note">{html.escape(str(row.get("risk_level", "")))}</div>
            </div>
            {''.join(nodes)}
        </div>
        <div class="account360-radar-details">
            <div class="account360-kicker">Risk dimensions</div>
            <div class="account360-details-title">Where pressure is coming from</div>
            {''.join(details)}
        </div>
    </div>
    '''


def account_360_questions_html(row: pd.Series) -> str:
    usage_score = (float(row.get("tool_activity_score", 0)) + float(row.get("feature_adoption_score", 0))) / 2
    usage_status = "Healthy" if usage_score >= 65 else "Partial" if usage_score >= 40 else "Weak"
    usage_cta = "Run an adoption recovery session around core workflows." if usage_score < 50 else "Reinforce value and identify the next adoption milestone."

    fit_score = (float(row.get("roadmap_alignment_score", 50)) + max(0, 100 - float(row.get("unresolved_feedback_count", 0)) * 15)) / 2
    fit_status = "Aligned" if fit_score >= 65 else "Needs expectation management" if fit_score >= 40 else "Product-fit risk"
    fit_cta = "Clarify roadmap fit, workaround options and renewal dependency." if fit_score < 55 else "Keep validating product needs and future roadmap alignment."

    trust_score = 100
    trust_score -= 25 if int(row.get("data_accuracy_bug_flag", 0)) == 1 else 0
    trust_score -= min(40, int(row.get("stuck_bugs_count", 0)) * 20)
    trust_score -= 20 if float(row.get("csat_support", 5)) < 3.2 else 0
    trust_status = "Stable" if trust_score >= 70 else "At risk" if trust_score >= 40 else "Weak"
    trust_cta = "Escalate data-trust or bug issues with Support/Product and set an update cadence." if trust_score < 55 else "Maintain proactive support follow-up and value reinforcement."

    items = [
        ("Is the customer using the product?", usage_status, usage_score, usage_cta, "Usage & adoption"),
        ("Does the product meet their needs?", fit_status, fit_score, fit_cta, "Product fit"),
        ("Can they trust the product enough to stay?", trust_status, trust_score, trust_cta, "Trust & reliability"),
    ]
    cards = []
    for question, status, score, cta, lens in items:
        _, css = _status_from_score(score, inverse=True)
        cards.append(
            f'<div class="account360-lens-card {css}">'
            f'<div class="account360-kicker">{html.escape(lens)}</div>'
            f'<div class="account360-lens-question">{html.escape(question)}</div>'
            f'<div class="account360-lens-status">{html.escape(status)}</div>'
            f'<div class="account360-mini-bar"><div style="width:{max(0,min(100,score)):.1f}%;"></div></div>'
            f'<div class="account360-lens-cta"><strong>Suggested move:</strong> {html.escape(cta)}</div>'
            f'</div>'
        )
    return f'<div class="account360-lens-grid">{"".join(cards)}</div>'


def action_360_panel_html(row: pd.Series) -> str:
    actions, owners, urgency = semrush_csm_cta(row)
    routing = build_department_routing(row)
    route_cards = []
    for item in routing[:3]:
        team = item.get("team", "CSM")
        team_css = "support" if "Support" in team else "sales" if "Sales" in team or "Executive" in team else "product" if "Product" in team else "csm"
        route_cards.append(
            f'<div class="account360-route-card {team_css}">'
            f'<div class="account360-route-team">{html.escape(team)}</div>'
            f'<div class="account360-route-label">Why needed</div><div class="account360-route-body">{html.escape(str(item.get("why", "")))}</div>'
            f'<div class="account360-route-label">Ask them to</div><div class="account360-route-body">{html.escape(str(item.get("action", "")))}</div>'
            f'</div>'
        )
    if not route_cards:
        route_cards.append('<div class="account360-route-card csm"><div class="account360-route-team">CSM only</div><div class="account360-route-body">No cross-functional blocker detected.</div></div>')
    return f'''
    <div class="account360-action-shell">
        <div class="account360-action-main">
            <div class="account360-kicker">Action 360°</div>
            <div class="account360-action-head">Recommended next move</div>
            <div class="account360-action-text">{html.escape(actions[0] if actions else "Review account health and validate next best action.")}</div>
            <div class="account360-action-meta">Urgency: <strong>{html.escape(urgency)}</strong> · Owners: <strong>{html.escape(", ".join(owners))}</strong></div>
        </div>
        <div class="account360-route-grid">{''.join(route_cards)}</div>
    </div>
    '''


def render_selected_account_briefing(row, include_scenario=False):
    st.markdown(account_360_core_html(row), unsafe_allow_html=True)

    section_header("Account 360° risk map", "A visual cockpit of what is happening in this account and which dimension needs action first.")
    st.markdown(account_360_radar_html(row), unsafe_allow_html=True)

    section_header("Three-question CSM lens", "A non-technical way to explain the account situation before a customer call.")
    st.markdown(account_360_questions_html(row), unsafe_allow_html=True)

    section_header("Action 360°", "Recommended action, cross-functional owners and the fastest path to unblock risk.")
    st.markdown(action_360_panel_html(row), unsafe_allow_html=True)
    with st.expander("Suggested handoff message", expanded=False):
        st.code(build_handoff_message(row), language="text")

    section_header("Scenario Simulator", "Which intervention could reduce churn probability fastest for this account?")
    render_scenario_story(row, model, model_columns)
    with st.expander("View scenario calculation table", expanded=False):
        scenarios = simulate_scenarios(row, model, model_columns)
        st.dataframe(scenarios, use_container_width=True, hide_index=True)
        st.caption("Scenario results are model-based simulations on synthetic data. They are decision-support estimates, not guaranteed outcomes.")

    section_header("Account signal briefing", "Plain-English evidence grouped by Usage, Limits, and Product trust. Hover the help badges and open expanders for more context.")
    render_signal_briefing(row)

    section_header("Structured CSM action plan", "The same information as an operational playbook, kept after the visual summary for detail review.")
    st.markdown(playbook_html(row), unsafe_allow_html=True)

    section_header("Supporting evidence", "Detailed metrics remain available, but are intentionally hidden until the CSM needs them.")
    with st.expander("Key account evidence", expanded=False):
        key_cols = [
            "account_id","account_name","segment","region","plan_family","plan_tier","mrr","churn_probability","risk_level",
            "renewal_due_days","renewal_intent","tool_activity_score","feature_adoption_score","crawl_budget_usage_pct",
            "keyword_tracking_usage_pct","ai_visibility_prompt_usage_pct","roadmap_alignment_score","unresolved_feedback_count",
            "stuck_bugs_count","data_accuracy_bug_flag","csat_support","sentiment_csm"
        ]
        available_cols = [c for c in key_cols if c in row.index]
        evidence = pd.DataFrame({"Field": available_cols, "Value": [str(row[c]) for c in available_cols]})
        st.dataframe(evidence, use_container_width=True, hide_index=True)

    with st.expander("Raw account data", expanded=False):
        raw_df = pd.DataFrame({"Field": row.index, "Value": [str(v) for v in row.values]})
        st.dataframe(raw_df, use_container_width=True, hide_index=True)


# Additional CSS for guided V4 UX
st.markdown(
    '<style>'
    '.guide-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:12px 0 22px;}'
    '.guide-step{background:white;border:1px solid var(--border);border-radius:20px;padding:16px;box-shadow:0 8px 22px rgba(15,23,42,.045);}'
    '.guide-number{width:28px;height:28px;border-radius:999px;background:linear-gradient(135deg,#2563EB,#0F766E);color:white;display:flex;align-items:center;justify-content:center;font-weight:950;margin-bottom:10px;}'
    '.guide-title{color:var(--navy);font-size:14px;font-weight:950;margin-bottom:6px;}'
    '.guide-body{color:#64748B;font-size:12px;line-height:1.35;}'
    '.daily-brief{background:radial-gradient(circle at 100% 0%, rgba(15,118,110,.20), transparent 28%),linear-gradient(135deg,#FFFFFF,#F0FDFA);border:1px solid #99F6E4;border-radius:24px;padding:22px;box-shadow:0 12px 30px rgba(15,23,42,.06);margin:10px 0 18px;}'
    '.daily-kicker{color:#0F766E;font-size:12px;font-weight:950;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;}'
    '.daily-title{color:var(--navy);font-size:24px;font-weight:950;margin-bottom:8px;}'
    '.daily-body{color:#334155;font-size:15px;line-height:1.55;}'
    '.mission-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin:12px 0 22px;}'
    '.mission-card{background:white;border:1px solid var(--border);border-radius:22px;padding:18px;box-shadow:0 10px 28px rgba(15,23,42,.055);min-height:235px;}'
    '.mission-kicker{color:var(--muted);font-size:11px;font-weight:950;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;}'
    '.mission-title{color:var(--navy);font-size:16px;font-weight:950;line-height:1.2;margin-bottom:10px;}'
    '.mission-value{color:var(--navy);font-size:31px;font-weight:950;line-height:1;margin-bottom:10px;}'
    '.mission-body{color:#475569;font-size:13px;line-height:1.4;margin-bottom:12px;}'
    '.mission-cta{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:14px;padding:10px;color:#334155;font-size:12px;font-weight:850;}'
    '.account-card-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:12px 0 22px;}'
    '.account-queue-card{background:white;border:1px solid var(--border);border-radius:22px;padding:19px;box-shadow:0 10px 28px rgba(15,23,42,.055);}'
    '.account-queue-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}'
    '.priority-pill{display:inline-block;border-radius:999px;padding:6px 10px;background:#EEF2FF;color:#4338CA;font-size:12px;font-weight:950;}'
    '.urgency-mini{display:inline-block;border-radius:999px;padding:5px 8px;font-size:11px;font-weight:950;}'
    '.account-queue-title{color:var(--navy);font-size:19px;font-weight:950;line-height:1.2;margin-bottom:5px;}'
    '.account-queue-sub{color:#64748B;font-size:12px;line-height:1.35;margin-bottom:6px;}'
    '.account-risk-line{display:flex;align-items:center;gap:10px;margin:12px 0;}'
    '.account-risk-value{color:var(--navy);font-size:28px;font-weight:950;}'
    '.account-action{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:14px;padding:11px;color:#334155;font-size:12px;line-height:1.4;margin-top:10px;}'
    '.account-owner{color:#475569;font-size:12px;margin-top:8px;}'
    '.account-brief-hero{background:white;border:1px solid var(--border);border-radius:24px;padding:24px;box-shadow:0 12px 30px rgba(15,23,42,.06);margin:12px 0 20px;}'
    '.account-brief-title{color:var(--navy);font-size:32px;font-weight:950;line-height:1.1;margin-bottom:7px;}'
    '.account-brief-sub{color:#64748B;font-size:14px;margin-bottom:12px;}'
    '.account-story{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:16px;padding:13px;color:#334155;font-size:14px;line-height:1.45;margin:12px 0;}'
    '.question-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:12px 0 20px;}'
    '.question-card{background:white;border:1px solid var(--border);border-radius:22px;padding:19px;box-shadow:0 10px 28px rgba(15,23,42,.055);min-height:175px;}'
    '.question-title{color:#475569;font-size:12px;font-weight:950;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;}'
    '.question-answer{color:var(--navy);font-size:24px;font-weight:950;margin-bottom:8px;}'
    '.question-body{color:#475569;font-size:13px;line-height:1.4;margin-bottom:8px;}'
    '.return-panel{background:linear-gradient(135deg,#FFFFFF,#EFF6FF);border:1px solid #BFDBFE;border-radius:22px;padding:17px 20px;box-shadow:0 10px 28px rgba(15,23,42,.055);margin:8px 0 12px;}'
    '.return-eyebrow{color:#2563EB;font-size:11px;font-weight:950;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;}'
    '.return-title{color:var(--navy);font-size:20px;font-weight:950;line-height:1.2;margin-bottom:4px;}'
    '.return-body{color:#475569;font-size:13px;line-height:1.45;}'
    '.mission-card{transition:transform .15s ease, box-shadow .15s ease, border-color .15s ease;}'
    '.mission-card:hover{transform:translateY(-3px);box-shadow:0 18px 42px rgba(15,23,42,.10);border-color:#BFDBFE;}'
    'div[data-testid="stButton"] > button{background:linear-gradient(135deg,#071427 0%,#2563EB 58%,#0F766E 100%) !important;color:#FFFFFF !important;border:0 !important;border-radius:16px !important;min-height:46px !important;padding:0.65rem 1rem !important;font-weight:950 !important;box-shadow:0 10px 24px rgba(37,99,235,.22) !important;transition:transform .12s ease, box-shadow .12s ease, filter .12s ease !important;}'
    'div[data-testid="stButton"] > button:hover{transform:translateY(-2px) !important;box-shadow:0 14px 30px rgba(37,99,235,.30) !important;filter:brightness(1.05) !important;}'
    'div[data-testid="stButton"] > button:active{transform:translateY(0px) !important;box-shadow:0 8px 18px rgba(37,99,235,.20) !important;}'
    'div[data-testid="stButton"] > button p{color:#FFFFFF !important;font-weight:950 !important;}'
    '@media (max-width:1100px){.mission-grid,.guide-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.account-card-grid,.question-grid{grid-template-columns:1fr;}}'
    '</style>',
    unsafe_allow_html=True,
)


# =========================
# Guided navigation with real CTAs
# =========================

VIEW_OPTIONS = [
    "Mission Control",
    "Priority Queue",
    "Portfolio Insights",
    "Account Briefing",
    "Scenario Simulator",
    "CSM Playbook",
    "Methodology & Data Mapping",
]

QUEUE_FILTER_OPTIONS = [
    "All accounts",
    "P1/P2 accounts",
    "Renewal risk",
    "Adoption gap",
    "Product friction",
    "Cross-functional routing",
    "Needs Customer Support",
    "Needs Account Executive / Sales",
    "Needs Product / Engineering",
]

if "active_view" not in st.session_state:
    st.session_state["active_view"] = "Mission Control"
if "queue_filter" not in st.session_state:
    st.session_state["queue_filter"] = "All accounts"


def go_to(view: str, queue_filter: str | None = None):
    st.session_state["active_view"] = view
    if queue_filter is not None:
        st.session_state["queue_filter"] = queue_filter
    st.rerun()


def apply_queue_filter(dataframe: pd.DataFrame, mode: str) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    if mode == "P1/P2 accounts":
        return dataframe[dataframe["priority"].isin(["P1", "P2"])]

    if mode == "Renewal risk":
        return dataframe[
            (dataframe["renewal_due_days"] < 60)
            & (dataframe["renewal_intent"] != "positive")
        ]

    if mode == "Adoption gap":
        return dataframe[
            (dataframe["tool_activity_score"] < 45)
            | (dataframe["feature_adoption_score"] < 45)
            | (dataframe["site_audit_last_run_days"] > 60)
            | (dataframe["position_tracking_campaigns_active"] == 0)
            | ((dataframe["ai_visibility_active"] == 1) & (dataframe["ai_visibility_prompts_tracked"] < 10))
        ]

    if mode == "Product friction":
        return dataframe[
            (dataframe["stuck_bugs_count"] > 0)
            | (dataframe["data_accuracy_bug_flag"] == 1)
            | (dataframe["unresolved_feedback_count"] >= 3)
            | (dataframe["roadmap_alignment_score"] < 40)
            | (dataframe["workaround_available_flag"] == 0)
        ]

    if mode == "Cross-functional routing":
        return dataframe[
            dataframe.apply(lambda r: not route_filter_matches(r, "CSM only"), axis=1)
        ]

    if mode in ["Needs Customer Support", "Needs Account Executive / Sales", "Needs Product / Engineering"]:
        return dataframe[dataframe.apply(lambda r: route_filter_matches(r, mode), axis=1)]

    return dataframe


def queue_context(mode: str) -> tuple[str, str]:
    contexts = {
        "All accounts": (
            "All accounts in the current portfolio view",
            "Use this when you want the complete book after applying sidebar filters. Start with P1/P2 before moving to Monitor accounts.",
        ),
        "P1/P2 accounts": (
            "Priority queue · P1/P2 accounts",
            "Start here. These are the accounts with the clearest risk, revenue impact or renewal sensitivity.",
        ),
        "Renewal risk": (
            "Renewal-risk queue",
            "Accounts where commercial timing, renewal intent, pricing, downgrade or grace-period signals need CSM + AE attention.",
        ),
        "Adoption gap": (
            "Adoption recovery queue",
            "Accounts that may be paying for value they are not fully realizing yet: weak usage, shallow feature adoption or inactive workflows.",
        ),
        "Product friction": (
            "Product friction queue",
            "Accounts where trust may be breaking because of stuck bugs, data accuracy flags, unresolved feedback or roadmap mismatch.",
        ),
        "Cross-functional routing": (
            "Cross-functional dependency queue",
            "Accounts where the CSM likely needs Customer Support, Account Executive / Sales, or Product / Engineering to unblock risk.",
        ),
        "Needs Customer Support": (
            "Customer Support handoff queue",
            "Accounts that need technical clarification, tool troubleshooting, limit explanation, navigation guidance or hard-block support.",
        ),
        "Needs Account Executive / Sales": (
            "Account Executive / Sales handoff queue",
            "Accounts where pricing, package fit, contract scope, renewal terms or commercial options should be reviewed.",
        ),
        "Needs Product / Engineering": (
            "Product / Engineering handoff queue",
            "Accounts where risk depends on roadmap fit, product gaps, stuck bugs or data-trust issues that CSM/Support cannot solve alone.",
        ),
    }
    return contexts.get(mode, contexts["All accounts"])


def render_return_control(current_view: str, current_queue_filter: str, current_queue_title: str):
    """Small navigation helper shown after a Mission Control CTA has moved the user into a drill-down view."""
    if current_view == "Mission Control":
        return

    if current_view == "Priority Queue":
        title = current_queue_title
        body = (
            "You are now in a filtered work queue opened from Mission Control. "
            "Review the accounts here, then return to the main panel when you want to choose another snapshot."
        )
    else:
        title = current_view
        body = "You are reviewing a detailed workspace. Return to Mission Control to restart the guided CSM flow."

    st.markdown(
        f'<div class="return-panel">'
        f'<div class="return-eyebrow">Current workspace</div>'
        f'<div class="return-title">{html.escape(str(title))}</div>'
        f'<div class="return-body">{html.escape(body)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    back_col, hint_col = st.columns([1.15, 4])
    with back_col:
        if st.button("← Back to Mission Control", key=f"back_to_mission_{current_view}_{current_queue_filter}", use_container_width=True):
            go_to("Mission Control", "All accounts")
    with hint_col:
        if current_view == "Priority Queue":
            st.caption(f"Active queue focus: {current_queue_filter}. Use the sidebar or the Mission Control cards to change the work queue.")
        else:
            st.caption("Use this detail view for evidence, simulation, or handoff preparation.")



def portfolio_360_html():
    """Immersive but readable portfolio 360 visual for Mission Control."""
    # Normalize values for the radial bars. These are visual intensities, not new model outputs.
    renewal_intensity = min(100, max(0, primary_theme_score))
    adoption_intensity = min(100, max(0, low_usage_count / max(len(filtered), 1) * 100))
    friction_intensity = min(100, max(0, product_friction_count / max(len(filtered), 1) * 100))
    dependency_intensity = min(100, max(0, (support_needed_count + sales_needed_count + product_needed_count) / max(len(filtered), 1) * 100))
    revenue_intensity = min(100, max(0, mrr_at_risk / max(filtered["mrr"].sum(), 1) * 100))

    def node(label, value, left, top, color, queue):
        return (
            f'<button class="orbit-node" style="left:{left}; top:{top}; --node-color:{color};" '
            f'title="Open {html.escape(queue)} from the Mission Control buttons below.">' 
            f'<span class="orbit-dot"></span><span class="orbit-label">{html.escape(label)}</span>'
            f'<span class="orbit-value">{html.escape(value)}</span></button>'
        )

    nodes = "".join([
        node("Renewal", f"{renewal_pressure_30}", "64%", "13%", "#DC2626", "Renewal risk"),
        node("Adoption", f"{low_usage_count}", "76%", "55%", "#0F766E", "Adoption gap"),
        node("Friction", f"{product_friction_count}", "41%", "76%", "#D97706", "Product friction"),
        node("Teams", f"{support_needed_count + sales_needed_count + product_needed_count}", "15%", "53%", "#6D28D9", "Cross-functional routing"),
        node("MRR", _fmt_money(mrr_at_risk), "24%", "14%", "#2563EB", "P1/P2 accounts"),
    ])

    return f"""
    <div class="portfolio-360-shell">
        <div class="portfolio-360-copy">
            <div class="immersive-eyebrow">Portfolio 360°</div>
            <div class="immersive-title">A visual command hub for what needs action first.</div>
            <div class="immersive-body">
                The center shows the overall portfolio risk. The surrounding orbit highlights where the pressure is coming from:
                commercial timing, adoption gaps, product friction, revenue exposure and cross-functional blockers.
            </div>
            <div class="immersive-takeaway"><strong>Recommended starting point:</strong> review P1/P2 accounts where high risk overlaps with renewal pressure, product friction or another department dependency.</div>
        </div>
        <div class="portfolio-360-stage">
            <div class="orbit-ring ring-1"></div>
            <div class="orbit-ring ring-2"></div>
            <div class="orbit-ring ring-3"></div>
            <div class="risk-core">
                <div class="core-label">Average risk</div>
                <div class="core-value">{avg_churn:.1%}</div>
                <div class="core-note">{len(filtered):,} accounts</div>
            </div>
            {nodes}
        </div>
        <div class="portfolio-360-legend">
            <div class="legend-row"><span class="legend-chip red"></span><span>Renewal pressure</span><strong>{renewal_intensity:.0f}/100</strong></div>
            <div class="legend-row"><span class="legend-chip teal"></span><span>Adoption gap</span><strong>{adoption_intensity:.0f}/100</strong></div>
            <div class="legend-row"><span class="legend-chip amber"></span><span>Product friction</span><strong>{friction_intensity:.0f}/100</strong></div>
            <div class="legend-row"><span class="legend-chip purple"></span><span>Team dependencies</span><strong>{dependency_intensity:.0f}/100</strong></div>
            <div class="legend-row"><span class="legend-chip blue"></span><span>MRR at risk share</span><strong>{revenue_intensity:.0f}/100</strong></div>
        </div>
    </div>
    """


def immersive_priority_landscape_html(dataframe):
    """Priority lanes with a 3D-lite look. Keeps full account cards available below."""
    lanes = []
    lane_data = [
        ("P1", "Act today", "Front lane: immediate attention", "#DC2626"),
        ("P2", "Review this week", "Second lane: active follow-up", "#D97706"),
        ("P3", "Stabilize", "Third lane: monitor and prevent escalation", "#2563EB"),
        ("Monitor", "Regular cadence", "Back lane: keep value visible", "#0F766E"),
    ]
    for priority, label, note, color in lane_data:
        subset = dataframe[dataframe["priority"] == priority]
        top = subset.sort_values("churn_probability", ascending=False).head(1)
        if top.empty:
            headline = "No accounts"
            details = "Nothing in this lane for the selected view."
            risk = "—"
        else:
            r = top.iloc[0]
            headline = str(r.get("account_name", r.get("account_id", "Account")))
            details = f'{r.get("plan_family", "")} · renewal in {int(r.get("renewal_due_days", 0))} days · {_fmt_money(r.get("mrr", 0))} MRR'
            risk = f'{float(r.get("churn_probability", 0)):.1%}'
        lanes.append(
            f'<div class="priority-3d-lane" style="--lane-color:{color};">'
            f'<div class="lane-top"><span>{html.escape(priority)}</span><strong>{len(subset)}</strong></div>'
            f'<div class="lane-label">{html.escape(label)}</div>'
            f'<div class="lane-risk">{html.escape(risk)}</div>'
            f'<div class="lane-account">{html.escape(headline)}</div>'
            f'<div class="lane-details">{html.escape(details)}</div>'
            f'<div class="lane-note">{html.escape(note)}</div>'
            f'</div>'
        )
    return f'<div class="priority-3d-board">{"".join(lanes)}</div>'


def render_clickable_mission_cards():
    cards = [
        {
            "kicker": "Portfolio risk",
            "title": "How healthy is my book?",
            "value": f"{avg_churn:.1%}",
            "body": f"Average churn probability across {len(filtered):,} selected accounts. Main risk theme: {primary_theme}.",
            "button": "🚀 Open Priority Queue",
            "tone": "blue",
            "queue": "P1/P2 accounts",
            "icon": "◎",
        },
        {
            "kicker": "Revenue & renewal",
            "title": "Where is commercial risk?",
            "value": _fmt_money(mrr_at_risk),
            "body": f"{renewal_pressure_30} accounts renew in <30 days with weak or non-positive intent.",
            "button": "💼 Open renewal-risk details",
            "tone": "red",
            "queue": "Renewal risk",
            "icon": "$",
        },
        {
            "kicker": "Adoption gap",
            "title": "Who pays but underuses?",
            "value": f"{low_usage_count}",
            "body": "Accounts with low product activity, shallow feature adoption or inactive Semrush-like workflows.",
            "button": "📈 Open adoption recovery list",
            "tone": "teal",
            "queue": "Adoption gap",
            "icon": "↗",
        },
        {
            "kicker": "Product friction",
            "title": "Where is trust breaking?",
            "value": f"{product_friction_count}",
            "body": "Accounts with stuck bugs, data accuracy flags, unresolved feedback or roadmap mismatch.",
            "button": "🛠️ Open product friction cases",
            "tone": "amber",
            "queue": "Product friction",
            "icon": "⚙",
        },
        {
            "kicker": "Team dependencies",
            "title": "Who needs to help?",
            "value": f"{support_needed_count + sales_needed_count + product_needed_count}",
            "body": "Accounts where Support, Sales or Product may need to help the CSM unblock risk.",
            "button": "🤝 Review cross-functional routing",
            "tone": "purple",
            "queue": "Cross-functional routing",
            "icon": "◌",
        },
    ]

    tone_color = {"blue": "#2563EB", "teal": "#0F766E", "red": "#DC2626", "amber": "#D97706", "purple": "#6D28D9"}
    cols = st.columns(5)
    for i, card in enumerate(cards):
        color = tone_color.get(card["tone"], "#2563EB")
        with cols[i]:
            st.markdown(
                f'<div class="mission-card immersive-card" style="--accent:{color}; border-top:5px solid {color};">'
                f'<div class="card-depth-glow"></div>'
                f'<div class="mission-icon-orb">{html.escape(card["icon"])}</div>'
                f'<div class="mission-kicker">{html.escape(card["kicker"])}</div>'
                f'<div class="mission-title">{html.escape(card["title"])}</div>'
                f'<div class="mission-value">{html.escape(card["value"])}</div>'
                f'<div class="mission-body">{html.escape(card["body"])}</div>'
                f'<div class="card-direction">Focus queue → {html.escape(card["queue"])}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(card["button"], key=f"mission_cta_{i}", use_container_width=True):
                go_to("Priority Queue", card["queue"])



# Immersive V7 corporate 3D-lite visual layer
st.markdown(
    """
    <style>
    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.18);
        box-shadow: 0 28px 70px rgba(7,20,39,.28);
    }
    .hero:after {
        content:"";
        position:absolute;
        right:-90px;
        top:-90px;
        width:310px;
        height:310px;
        border-radius:999px;
        background: radial-gradient(circle, rgba(167,243,208,.30), rgba(37,99,235,.08) 45%, transparent 70%);
        filter: blur(.2px);
    }
    .mission-card.immersive-card {
        position:relative;
        overflow:hidden;
        transform-style: preserve-3d;
        box-shadow: 0 20px 45px rgba(15,23,42,.10), inset 0 1px 0 rgba(255,255,255,.8);
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        min-height: 270px;
    }
    .mission-card.immersive-card:hover {
        transform: perspective(900px) rotateX(2deg) rotateY(-2deg) translateY(-6px);
        box-shadow: 0 30px 70px rgba(15,23,42,.17), 0 0 0 1px rgba(37,99,235,.08);
    }
    .card-depth-glow {
        position:absolute;
        right:-55px;
        top:-55px;
        width:145px;
        height:145px;
        border-radius:999px;
        background: radial-gradient(circle, color-mix(in srgb, var(--accent), white 18%), transparent 62%);
        opacity:.25;
    }
    .mission-icon-orb {
        width:42px;
        height:42px;
        border-radius:16px;
        display:flex;
        align-items:center;
        justify-content:center;
        color:white;
        font-size:20px;
        font-weight:950;
        background: linear-gradient(135deg, var(--accent), #071427);
        box-shadow: 0 12px 25px color-mix(in srgb, var(--accent), transparent 70%);
        margin-bottom:12px;
    }
    .card-direction {
        margin-top:12px;
        background:linear-gradient(135deg,#F8FAFC,#EFF6FF);
        border:1px solid #DBEAFE;
        color:#334155;
        padding:9px 10px;
        border-radius:14px;
        font-size:12px;
        font-weight:850;
    }
    .portfolio-360-shell {
        display:grid;
        grid-template-columns: 1.05fr 1.25fr .88fr;
        gap:18px;
        align-items:center;
        background: radial-gradient(circle at 50% 40%, rgba(37,99,235,.10), transparent 34%), linear-gradient(135deg, #FFFFFF, #F8FAFC 48%, #ECFEFF);
        border:1px solid #BFE7F6;
        border-radius:30px;
        padding:24px;
        box-shadow:0 26px 65px rgba(15,23,42,.10);
        margin: 14px 0 22px;
        overflow:hidden;
        position:relative;
    }
    .portfolio-360-shell:before {
        content:"";
        position:absolute;
        inset:-2px;
        background:linear-gradient(120deg, rgba(37,99,235,.10), transparent, rgba(15,118,110,.12));
        pointer-events:none;
    }
    .portfolio-360-copy, .portfolio-360-stage, .portfolio-360-legend { position:relative; z-index:1; }
    .immersive-eyebrow { color:#0F766E; font-size:12px; font-weight:950; text-transform:uppercase; letter-spacing:.10em; margin-bottom:8px; }
    .immersive-title { color:#071427; font-size:26px; font-weight:950; line-height:1.08; margin-bottom:10px; }
    .immersive-body { color:#475569; font-size:14px; line-height:1.55; margin-bottom:12px; }
    .immersive-takeaway { background:#FFFFFF; border:1px solid #DDEBF7; border-radius:18px; padding:13px 14px; color:#334155; font-size:13px; line-height:1.45; }
    .portfolio-360-stage {
        height:330px;
        min-width:360px;
        perspective: 1000px;
        transform-style: preserve-3d;
    }
    .orbit-ring {
        position:absolute;
        left:50%;
        top:50%;
        transform:translate(-50%, -50%) rotateX(63deg) rotateZ(-12deg);
        border:1px solid rgba(37,99,235,.23);
        border-radius:50%;
        box-shadow: 0 0 38px rgba(37,99,235,.08);
    }
    .ring-1 { width:145px; height:145px; }
    .ring-2 { width:230px; height:230px; border-color:rgba(15,118,110,.25); }
    .ring-3 { width:320px; height:320px; border-color:rgba(109,40,217,.14); }
    .risk-core {
        position:absolute;
        left:50%;
        top:50%;
        transform:translate(-50%, -50%);
        width:145px;
        height:145px;
        border-radius:50%;
        background: radial-gradient(circle at 35% 30%, #FFFFFF, #EFF6FF 52%, #DBEAFE);
        border:1px solid #BFDBFE;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        box-shadow: 0 22px 45px rgba(37,99,235,.18);
    }
    .core-label { color:#64748B; font-size:11px; font-weight:950; text-transform:uppercase; letter-spacing:.07em; }
    .core-value { color:#071427; font-size:31px; font-weight:950; line-height:1; margin:6px 0; }
    .core-note { color:#64748B; font-size:12px; font-weight:800; }
    .orbit-node {
        position:absolute;
        border:0;
        background:white;
        border-radius:18px;
        padding:10px 12px;
        min-width:116px;
        text-align:left;
        box-shadow: 0 14px 34px rgba(15,23,42,.13);
        border:1px solid #E2E8F0;
        transform:translate(-50%, -50%);
    }
    .orbit-dot { display:inline-block; width:10px; height:10px; border-radius:50%; background:var(--node-color); margin-right:6px; box-shadow:0 0 0 5px color-mix(in srgb, var(--node-color), transparent 82%); }
    .orbit-label { color:#334155; font-size:12px; font-weight:900; }
    .orbit-value { display:block; margin-top:5px; color:#071427; font-size:18px; font-weight:950; }
    .portfolio-360-legend { background:white; border:1px solid #E2E8F0; border-radius:22px; padding:16px; box-shadow:0 14px 34px rgba(15,23,42,.06); }
    .legend-row { display:grid; grid-template-columns:18px 1fr auto; gap:8px; align-items:center; padding:10px 0; border-bottom:1px solid #EEF2F7; color:#475569; font-size:13px; }
    .legend-row:last-child { border-bottom:0; }
    .legend-row strong { color:#071427; }
    .legend-chip { width:12px; height:12px; border-radius:999px; display:inline-block; }
    .legend-chip.red { background:#DC2626; } .legend-chip.teal { background:#0F766E; } .legend-chip.amber { background:#D97706; } .legend-chip.purple { background:#6D28D9; } .legend-chip.blue { background:#2563EB; }
    .priority-3d-board {
        display:grid;
        grid-template-columns: repeat(4, minmax(0,1fr));
        gap:16px;
        margin: 14px 0 22px;
        perspective: 1200px;
    }
    .priority-3d-lane {
        position:relative;
        background: linear-gradient(145deg,#FFFFFF,#F8FAFC);
        border:1px solid #E2E8F0;
        border-top:5px solid var(--lane-color);
        border-radius:24px;
        padding:18px;
        min-height:245px;
        box-shadow:0 22px 52px rgba(15,23,42,.11);
        transform: rotateX(1.5deg);
        overflow:hidden;
    }
    .priority-3d-lane:after {
        content:"";
        position:absolute;
        width:120px;
        height:120px;
        right:-45px;
        top:-45px;
        border-radius:999px;
        background: radial-gradient(circle, color-mix(in srgb, var(--lane-color), white 12%), transparent 68%);
        opacity:.20;
    }
    .lane-top { display:flex; justify-content:space-between; align-items:center; color:#64748B; font-size:12px; font-weight:950; text-transform:uppercase; letter-spacing:.08em; margin-bottom:12px; }
    .lane-top strong { color:var(--lane-color); font-size:24px; }
    .lane-label { color:#071427; font-size:20px; font-weight:950; margin-bottom:12px; }
    .lane-risk { color:#071427; font-size:34px; font-weight:950; line-height:1; margin-bottom:10px; }
    .lane-account { color:#334155; font-size:14px; font-weight:900; line-height:1.25; margin-bottom:6px; }
    .lane-details, .lane-note { color:#64748B; font-size:12px; line-height:1.38; }
    .lane-note { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:14px; padding:9px 10px; margin-top:10px; }
    .account-queue-card, .question-card, .scenario-card, .routing-card, .signal-brief-card {
        transition: transform .16s ease, box-shadow .16s ease;
    }
    .account-queue-card:hover, .question-card:hover, .scenario-card:hover, .routing-card:hover, .signal-brief-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 22px 52px rgba(15,23,42,.12);
    }
    @media (max-width:1100px){
        .portfolio-360-shell{grid-template-columns:1fr;}
        .portfolio-360-stage{min-width:0;}
        .priority-3d-board{grid-template-columns:1fr 1fr;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# V8 Account 360° immersive CSS layer
st.markdown(
    """
    <style>
    .account360-shell { display:grid; grid-template-columns: 1.25fr 1.05fr .85fr; gap:16px; margin: 14px 0 22px; perspective:1200px; }
    .account360-core-card, .account360-priority-card, .account360-commercial-card {
        background: radial-gradient(circle at 100% 0%, rgba(15,118,110,.12), transparent 28%), linear-gradient(145deg,#FFFFFF,#F8FAFC);
        border:1px solid #E2E8F0; border-radius:28px; padding:24px; box-shadow:0 24px 58px rgba(15,23,42,.10); position:relative; overflow:hidden;
    }
    .account360-core-card { transform: perspective(900px) rotateX(.8deg); }
    .account360-priority-card { border-top:5px solid #2563EB; }
    .account360-commercial-card { border-top:5px solid #0F766E; }
    .account360-eyebrow, .account360-kicker { color:#0F766E; font-size:12px; font-weight:950; text-transform:uppercase; letter-spacing:.09em; margin-bottom:8px; }
    .account360-title { color:#071427; font-size:34px; font-weight:950; line-height:1.08; margin-bottom:6px; }
    .account360-sub { color:#64748B; font-size:14px; line-height:1.4; margin-bottom:12px; }
    .account360-risk-row { display:flex; align-items:center; gap:12px; margin: 10px 0 12px; }
    .account360-prob { color:#071427; font-size:36px; font-weight:950; line-height:1; }
    .account360-story { background:#FFFFFF; border:1px solid #DDEBF7; border-radius:18px; padding:14px; color:#334155; font-size:14px; line-height:1.5; margin:12px 0; }
    .account360-action-title { color:#071427; font-size:20px; font-weight:950; line-height:1.25; margin-bottom:12px; }
    .account360-action-meta { color:#475569; font-size:13px; line-height:1.45; margin-top:8px; }
    .account360-metric-row { display:flex; justify-content:space-between; align-items:center; padding:11px 0; border-bottom:1px solid #EEF2F7; color:#64748B; font-size:13px; }
    .account360-metric-row:last-child { border-bottom:0; }
    .account360-metric-row strong { color:#071427; font-size:15px; }
    .account360-radar-shell { display:grid; grid-template-columns: 1.05fr 1fr; gap:18px; align-items:center; background:linear-gradient(135deg,#FFFFFF,#F8FAFC 50%,#ECFEFF); border:1px solid #BFE7F6; border-radius:30px; padding:24px; box-shadow:0 26px 65px rgba(15,23,42,.10); margin: 14px 0 22px; }
    .account360-radar-stage { height:390px; min-width:390px; position:relative; perspective:1000px; }
    .account360-ring { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%) rotateX(62deg) rotateZ(-9deg); border-radius:50%; border:1px solid rgba(37,99,235,.24); box-shadow:0 0 38px rgba(37,99,235,.08); }
    .account360-ring.ring-a { width:150px; height:150px; }
    .account360-ring.ring-b { width:250px; height:250px; border-color:rgba(15,118,110,.25); }
    .account360-ring.ring-c { width:350px; height:350px; border-color:rgba(109,40,217,.18); }
    .account360-center { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); width:150px; height:150px; border-radius:50%; background:radial-gradient(circle at 35% 28%,#FFFFFF,#EFF6FF 55%,#DBEAFE); border:1px solid #BFDBFE; display:flex; flex-direction:column; align-items:center; justify-content:center; box-shadow:0 22px 45px rgba(37,99,235,.18); }
    .account360-node { position:absolute; transform:translate(-50%,-50%); background:white; border:1px solid #E2E8F0; border-top:4px solid var(--node-color); border-radius:18px; min-width:126px; padding:10px 11px; box-shadow:0 16px 34px rgba(15,23,42,.13); }
    .account360-node.high { box-shadow:0 16px 38px rgba(220,38,38,.18); } .account360-node.medium { box-shadow:0 16px 38px rgba(217,119,6,.16); }
    .account360-node-label { display:block; color:#334155; font-size:11px; font-weight:950; line-height:1.2; }
    .account360-node-value { display:block; color:#071427; font-size:22px; font-weight:950; margin-top:5px; }
    .account360-radar-details { background:white; border:1px solid #E2E8F0; border-radius:24px; padding:20px; box-shadow:0 14px 34px rgba(15,23,42,.06); }
    .account360-details-title { color:#071427; font-size:22px; font-weight:950; margin-bottom:12px; }
    .account360-radar-row { display:grid; grid-template-columns: 1.6fr .9fr 1fr .9fr; gap:10px; align-items:center; padding:11px 0; border-bottom:1px solid #EEF2F7; color:#475569; font-size:12px; }
    .account360-radar-row:last-child { border-bottom:0; }
    .account360-radar-row strong { color:#071427; font-size:13px; }
    .account360-status { display:inline-block; text-align:center; border-radius:999px; padding:6px 8px; font-size:11px; font-weight:950; }
    .account360-status.high { background:#FEE2E2; color:#991B1B; } .account360-status.medium { background:#FEF3C7; color:#92400E; } .account360-status.low { background:#DCFCE7; color:#166534; }
    .account360-mini-bar { height:11px; border-radius:999px; background:#E2E8F0; overflow:hidden; }
    .account360-mini-bar div { height:11px; border-radius:999px; background:linear-gradient(90deg,#2563EB,#0F766E); }
    .account360-owner { color:#64748B; font-size:11px; font-weight:850; }
    .account360-lens-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin:12px 0 22px; }
    .account360-lens-card { background:white; border:1px solid #E2E8F0; border-radius:24px; padding:20px; box-shadow:0 18px 42px rgba(15,23,42,.08); border-top:5px solid #0F766E; min-height:225px; transition:transform .16s ease, box-shadow .16s ease; }
    .account360-lens-card:hover { transform:translateY(-5px); box-shadow:0 28px 62px rgba(15,23,42,.13); }
    .account360-lens-card.high { border-top-color:#DC2626; } .account360-lens-card.medium { border-top-color:#D97706; } .account360-lens-card.low { border-top-color:#0F766E; }
    .account360-lens-question { color:#475569; font-size:13px; font-weight:950; text-transform:uppercase; letter-spacing:.05em; line-height:1.25; margin-bottom:10px; }
    .account360-lens-status { color:#071427; font-size:25px; font-weight:950; margin-bottom:10px; }
    .account360-lens-cta { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:16px; padding:12px; color:#334155; font-size:13px; line-height:1.45; margin-top:12px; }
    .account360-action-shell { display:grid; grid-template-columns: .9fr 1.5fr; gap:16px; margin:12px 0 22px; }
    .account360-action-main { background:radial-gradient(circle at 100% 0%,rgba(37,99,235,.16),transparent 30%),linear-gradient(145deg,#FFFFFF,#EFF6FF); border:1px solid #BFDBFE; border-radius:26px; padding:22px; box-shadow:0 18px 42px rgba(15,23,42,.08); }
    .account360-action-head { color:#071427; font-size:26px; font-weight:950; line-height:1.12; margin-bottom:12px; }
    .account360-action-text { color:#334155; font-size:14px; line-height:1.55; background:white; border:1px solid #DBEAFE; border-radius:18px; padding:14px; }
    .account360-route-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
    .account360-route-card { background:white; border:1px solid #E2E8F0; border-radius:22px; padding:18px; box-shadow:0 14px 34px rgba(15,23,42,.07); border-top:5px solid #64748B; }
    .account360-route-card.support { border-top-color:#2563EB; } .account360-route-card.sales { border-top-color:#7C3AED; } .account360-route-card.product { border-top-color:#0F766E; }
    .account360-route-team { color:#071427; font-size:17px; font-weight:950; margin-bottom:9px; }
    .account360-route-label { color:#64748B; font-size:10px; font-weight:950; text-transform:uppercase; letter-spacing:.08em; margin:9px 0 4px; }
    .account360-route-body { color:#475569; font-size:12px; line-height:1.42; }
    .account360-search-panel { background:linear-gradient(135deg,#FFFFFF,#F0FDFA); border:1px solid #99F6E4; border-radius:20px; padding:15px; margin:10px 0 16px; box-shadow:0 10px 26px rgba(15,23,42,.06); }
    .account360-search-title { color:#0F766E; font-size:12px; font-weight:950; text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }
    .account360-search-body { color:#475569; font-size:12px; line-height:1.35; }
    @media (max-width:1100px){ .account360-shell, .account360-radar-shell, .account360-action-shell { grid-template-columns:1fr; } .account360-lens-grid, .account360-route-grid { grid-template-columns:1fr; } .account360-radar-stage { min-width:0; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar navigation should follow CTA clicks.
st.sidebar.divider()
st.sidebar.header("Workspace")
current_view = st.session_state.get("active_view", "Mission Control")
if current_view not in VIEW_OPTIONS:
    current_view = "Mission Control"
selected_view = st.sidebar.radio(
    "Go to",
    VIEW_OPTIONS,
    index=VIEW_OPTIONS.index(current_view),
)
st.session_state["active_view"] = selected_view

current_queue = st.session_state.get("queue_filter", "All accounts")
if current_queue not in QUEUE_FILTER_OPTIONS:
    current_queue = "All accounts"
selected_queue_filter = st.sidebar.selectbox(
    "Priority Queue focus",
    QUEUE_FILTER_OPTIONS,
    index=QUEUE_FILTER_OPTIONS.index(current_queue),
)
st.session_state["queue_filter"] = selected_queue_filter

st.sidebar.divider()
st.sidebar.markdown(
    '<div class="account360-search-panel"><div class="account360-search-title">Account 360° search</div><div class="account360-search-body">Jump directly to one customer account and open its 360° briefing.</div></div>',
    unsafe_allow_html=True,
)
if not filtered.empty:
    account_options = filtered["account_id"].tolist()
    current_selected = st.session_state.get("selected_account_id", account_options[0])
    selected_index = account_options.index(current_selected) if current_selected in account_options else 0
    selected_jump = st.sidebar.selectbox(
        "Search / select account",
        account_options,
        index=selected_index,
        format_func=lambda x: f"{x} · {filtered.loc[filtered['account_id']==x, 'account_name'].iloc[0]}",
        key="global_account_search",
    )
    if st.sidebar.button("Open Account 360°", key="open_account_360_sidebar", use_container_width=True):
        st.session_state["selected_account_id"] = selected_jump
        st.session_state["active_view"] = "Account Briefing"
        st.rerun()

active_view = st.session_state["active_view"]
active_queue_filter = st.session_state["queue_filter"]
queue_filtered = apply_queue_filter(filtered, active_queue_filter).sort_values("churn_probability", ascending=False).reset_index(drop=True)
queue_title, queue_description = queue_context(active_queue_filter)

# Show a clear way back to the main panel whenever the user drills down from a Mission Control CTA.
render_return_control(active_view, active_queue_filter, queue_title)


if active_view == "Mission Control":
    section_header("Mission Control", "Start here. Each snapshot is now a real entry point into a filtered CSM work queue.")
    st.markdown(mission_guide_html(), unsafe_allow_html=True)
    st.markdown(daily_brief_html(), unsafe_allow_html=True)
    st.markdown(portfolio_360_html(), unsafe_allow_html=True)
    render_clickable_mission_cards()

    st.markdown(
        '<div class="briefing-note"><strong>How to use the cards:</strong> click any snapshot CTA to open the Priority Queue with the right filter already applied. The goal is to move from portfolio signal → focused queue → account briefing → handoff or recovery action.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("View details behind Portfolio Risk", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            render_static_bar_chart(filtered["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int), "Accounts by risk level", "Accounts", False)
        with c2:
            by_segment = filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=False)
            render_static_bar_chart(by_segment, "Average churn probability by segment", "Churn probability", True)

    with st.expander("View details behind Revenue & Renewal Exposure", expanded=False):
        renewal_cols = ["priority", "account_id", "account_name", "segment", "plan_family", "mrr", "churn_probability", "renewal_due_days", "renewal_intent", "pricing_objection_flag", "downgrade_risk_flag"]
        renewal_view = filtered[(filtered["renewal_due_days"] < 60) | (filtered["risk_level"] == "High")][renewal_cols].copy().head(50)
        renewal_view["mrr"] = renewal_view["mrr"].map(lambda x: f"${x:,.0f}")
        renewal_view["churn_probability"] = renewal_view["churn_probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(renewal_view, use_container_width=True, hide_index=True)

    with st.expander("View details behind Adoption Gap", expanded=False):
        adoption_view = filtered[(filtered["tool_activity_score"] < 45) | (filtered["feature_adoption_score"] < 45)][["account_id", "account_name", "plan_family", "tool_activity_score", "feature_adoption_score", "site_audit_last_run_days", "position_tracking_campaigns_active", "ai_visibility_prompts_tracked", "churn_probability"]].head(50).copy()
        adoption_view["churn_probability"] = adoption_view["churn_probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(adoption_view, use_container_width=True, hide_index=True)

    with st.expander("View details behind Product Friction & Trust Risk", expanded=False):
        friction_view = filtered[(filtered["stuck_bugs_count"] > 0) | (filtered["data_accuracy_bug_flag"] == 1) | (filtered["unresolved_feedback_count"] >= 3) | (filtered["roadmap_alignment_score"] < 40)][["account_id", "account_name", "plan_family", "stuck_bugs_count", "data_accuracy_bug_flag", "unresolved_feedback_count", "roadmap_alignment_score", "workaround_available_flag", "churn_probability"]].head(50).copy()
        friction_view["churn_probability"] = friction_view["churn_probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(friction_view, use_container_width=True, hide_index=True)

    with st.expander("View cross-functional dependency details", expanded=False):
        st.markdown(routing_summary_html(filtered), unsafe_allow_html=True)


elif active_view == "Priority Queue":
    section_header("CSM Priority Queue", "A guided work queue. Use the focus selector or Mission Control CTAs to choose the right account set.")
    st.markdown(
        f'<div class="daily-brief"><div class="daily-kicker">Current queue focus</div><div class="daily-title">{html.escape(queue_title)}</div><div class="daily-body">{html.escape(queue_description)}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(immersive_priority_landscape_html(queue_filtered), unsafe_allow_html=True)

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        metric_card("Accounts in this queue", len(queue_filtered), "After sidebar filters and queue focus")
    with q2:
        metric_card("Average churn probability", _fmt_pct(queue_filtered["churn_probability"].mean()) if len(queue_filtered) else "N/A", "Mean predicted risk in this queue")
    with q3:
        metric_card("MRR in queue", _fmt_money(queue_filtered["mrr"].sum()) if len(queue_filtered) else "$0", "Commercial exposure represented here")
    with q4:
        metric_card("High-risk accounts", int((queue_filtered["risk_level"] == "High").sum()) if len(queue_filtered) else 0, "Accounts already over the high-risk threshold")

    st.markdown('<div class="briefing-note"><strong>How to use this queue:</strong> start with P1/P2 cards. If this queue came from a Mission Control CTA, it is already filtered for the relevant risk theme. Open Account Briefing for the selected account when you need evidence, scenario simulation or handoff details.</div>', unsafe_allow_html=True)

    if queue_filtered.empty:
        st.info("No accounts match this queue focus with the current sidebar filters.")
    else:
        for priority, title, expanded in [("P1", "P1 — Act first", True), ("P2", "P2 — Review this week", True), ("P3", "P3 — Monitor closely", False), ("Monitor", "Monitor — Regular cadence", False)]:
            subset = queue_filtered[queue_filtered["priority"] == priority]
            if not subset.empty:
                with st.expander(title, expanded=expanded):
                    render_priority_cards(subset, max_cards=9)

        with st.expander("Full operational account table for this queue", expanded=False):
            cols = ["priority", "account_id", "account_name", "segment", "region", "plan_family", "plan_tier", "mrr", "churn_probability", "risk_level", "renewal_due_days", "renewal_intent", "tool_activity_score", "feature_adoption_score", "crawl_budget_usage_pct", "keyword_tracking_usage_pct", "roadmap_alignment_score", "stuck_bugs_count", "data_accuracy_bug_flag"]
            table = queue_filtered[cols].copy()
            table["mrr"] = table["mrr"].map(lambda x: f"${x:,.0f}")
            table["churn_probability"] = table["churn_probability"].map(lambda x: f"{x:.1%}")
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.download_button("Download this queue as CSV", table.to_csv(index=False), f"semrush_copilot_{active_queue_filter.lower().replace(' ', '_').replace('/', '_')}.csv", "text/csv")


elif active_view == "Portfolio Insights":
    section_header("Portfolio Insights", "A guided view of where risk is building before looking at account-level evidence.")
    c1, c2 = st.columns(2)
    with c1:
        render_static_bar_chart(filtered["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int), "Accounts by risk level", "Accounts", False)
    with c2:
        by_segment = filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=False)
        render_static_bar_chart(by_segment, "Average churn probability by segment", "Churn probability", True)

    section_header("Where is risk building?", "Risk family score explained as a CSM story, not a raw table.")
    avg_scores = avg_family_scores.reset_index()
    avg_scores.columns = ["Risk family", "Average score"]
    render_driver_story_section(avg_scores)

    section_header("Which package families need attention?", "A CSM-friendly view of risk concentration by Semrush-like package family.")
    by_family = filtered.groupby("plan_family")["churn_probability"].mean().sort_values(ascending=False)
    render_commercial_exposure_story_section(by_family)

    section_header("Cross-functional action routing", "Which teams are needed to unblock retention risk in the current portfolio view.")
    st.markdown(routing_summary_html(filtered), unsafe_allow_html=True)


elif active_view == "Account Briefing":
    section_header("Account 360° Briefing", "Search any customer, understand the full account story, then move from risk evidence to action.")
    briefing_source = queue_filtered if active_queue_filter != "All accounts" and not queue_filtered.empty else filtered
    if briefing_source.empty:
        st.info("No accounts available for the current filters.")
    else:
        account_options = briefing_source["account_id"].tolist()
        default_selected = st.session_state.get("selected_account_id", account_options[0])
        default_index = account_options.index(default_selected) if default_selected in account_options else 0
        selected = st.selectbox(
            "Search / select account",
            account_options,
            index=default_index,
            format_func=lambda x: f"{x} · {briefing_source.loc[briefing_source['account_id']==x, 'account_name'].iloc[0]}",
            key="briefing_account",
        )
        st.session_state["selected_account_id"] = selected
        row = briefing_source.loc[briefing_source["account_id"] == selected].iloc[0]
        st.caption(f"Briefing source: {active_queue_filter}. Change Priority Queue focus in the sidebar if you want a different account set.")
        b1, b2, b3 = st.columns([1, 1, 3])
        with b1:
            if st.button("Open simulator for this account", key="briefing_to_sim", use_container_width=True):
                st.session_state["selected_account_id"] = selected
                go_to("Scenario Simulator", active_queue_filter)
        with b2:
            if st.button("Back to queue", key="briefing_back_queue", use_container_width=True):
                go_to("Priority Queue", active_queue_filter)
        render_selected_account_briefing(row, include_scenario=False)


elif active_view == "Scenario Simulator":
    section_header("Scenario Simulator", "Compare which action could reduce churn probability fastest before deciding the recovery plan.")
    sim_source = queue_filtered if active_queue_filter != "All accounts" and not queue_filtered.empty else filtered
    if sim_source.empty:
        st.info("No accounts available for simulation with the current filters.")
    else:
        sim_options = sim_source["account_id"].tolist()
        default_sim = st.session_state.get("selected_account_id", sim_options[0])
        default_sim_index = sim_options.index(default_sim) if default_sim in sim_options else 0
        selected_sim = st.selectbox(
            "Search / select account for simulation",
            sim_options,
            index=default_sim_index,
            format_func=lambda x: f"{x} · {sim_source.loc[sim_source['account_id']==x, 'account_name'].iloc[0]}",
            key="simulator_account",
        )
        st.session_state["selected_account_id"] = selected_sim
        sim_row = sim_source.loc[sim_source["account_id"] == selected_sim].iloc[0]
        st.caption(f"Simulation source: {active_queue_filter}. Change Priority Queue focus in the sidebar if you want to simulate a different queue.")
        st.markdown(account_360_core_html(sim_row), unsafe_allow_html=True)
        render_scenario_story(sim_row, model, model_columns)
        with st.expander("View scenario calculation table", expanded=False):
            scenarios = simulate_scenarios(sim_row, model, model_columns)
            st.dataframe(scenarios, use_container_width=True, hide_index=True)
            st.caption("Scenario results are model-based simulations on synthetic data. They are decision-support estimates, not guaranteed outcomes.")


elif active_view == "CSM Playbook":
    section_header("CSM Playbook", "Rules that translate model signals into direct customer-success actions.")
    st.markdown('<div class="briefing-note"><strong>How to use this:</strong> the playbook is not another dashboard. It explains why the Copilot recommends involving CSM, Support, Sales or Product for specific account patterns.</div>', unsafe_allow_html=True)

    section_header("Cross-functional routing rules", "When the CSM should involve Customer Support, Account Executive / Sales, or Product / Engineering.")
    routing_rules = pd.DataFrame([
        ["Customer Support", "Tool discrepancy or bug", "data_accuracy_bug_flag, site_audit_bug_flag, position_tracking_bug_flag, ai_visibility_bug_flag, local_listing_sync_bug_flag", "Validate issue, confirm expected behaviour vs defect, document workaround and customer-facing explanation."],
        ["Customer Support", "Limit misunderstanding", "crawl_budget_usage_pct > 85, keyword_tracking_usage_pct > 85, ai_visibility_prompt_usage_pct > 85, tool_limit_feedback_count >= 1", "Clarify account limits, user/team allocation, current usage and navigation path to verify limits."],
        ["Customer Support", "Navigation / troubleshooting help", "technical_tickets_last_quarter >= 4 or repeated tool-specific issues", "Provide precise navigation explanation for Site Audit, Position Tracking, backlinks, prompt tracking or listing setup."],
        ["Customer Support", "Hard block", "stuck_bugs_count >= 1 and workaround_available_flag = 0", "Treat as blocked workflow, confirm severity, document impact and define update cadence."],
        ["Account Executive / Sales", "Pricing or package fit", "pricing_objection_flag, downgrade_risk_flag, limit_upgrade_recommended_flag", "Review commercial options, upgrade/downgrade fit, package scope and value narrative."],
        ["Account Executive / Sales", "Contract questions", "renewal_due_days < 45 and weak renewal intent, billing tickets, purchased add-ons uncertainty", "Confirm contract terms, purchased packages, renewal options and commercial commitments before customer follow-up."],
        ["Product / Engineering", "Roadmap or product gap", "critical_feedback_count_90d >= 1 and roadmap_alignment_score < 45", "Confirm roadmap status, workaround or alternative workflow before renewal conversation."],
        ["Product / Engineering", "Stuck data-trust issue", "data_accuracy_bug_flag = 1 and stuck_bugs_count >= 1", "Confirm investigation status, customer impact, ETA and next update path."],
    ], columns=["Team", "Trigger", "Signals", "CSM handoff ask"])
    st.dataframe(routing_rules, use_container_width=True, hide_index=True)

    section_header("Semrush-specific CSM action rules", "Rules that translate model signals into direct customer-success actions.")
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


elif active_view == "Methodology & Data Mapping":
    section_header("Methodology & Data Mapping", "Keep the details available, but separate from the daily CSM workflow.")
    st.markdown(
        '<div class="method-card"><strong>What is predicted?</strong><br>For each synthetic account, the model estimates <code>P(churn = 1 | account signals)</code>. The portfolio KPI is the average of those account-level probabilities.</div>'
        '<div class="method-card"><strong>What goes into the model?</strong><br>Account profile, plan family, toolkit adoption, usage limits, Site Audit and Position Tracking usage, AI Visibility adoption, Local Toolkit activity, support friction, feedback, bugs, sentiment and renewal context.</div>'
        '<div class="method-card"><strong>What is not used live?</strong><br><code>account_id</code>, <code>account_name</code>, <code>churn_label</code>, and <code>synthetic_true_churn_probability</code> are excluded from predictive features. The churn label is used only during training.</div>'
        '<div class="method-card"><strong>Cross-functional routing</strong><br>The routing layer is rule-based. It does not change the churn probability; it translates visible risk patterns into CSM handoff guidance for Support, Account Executive / Sales, or Product / Engineering.</div>'
        '<div class="method-card"><strong>Internal navigation</strong><br>Mission Control cards route the CSM into the correct Priority Queue filter. This keeps the first view simple while preserving all detailed account evidence in drill-down sections.</div>'
        '<div class="method-card"><strong>Important disclaimer</strong><br>This is synthetic demo data inspired by public Semrush product information. It is not internal Semrush customer data and should be retrained on authorized historical company data before production use.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Suggested company data mapping", expanded=False):
        mapping = pd.DataFrame([
            ["login_count_30d / tool_activity_score", "Product analytics", "User activity, sessions, workflow events"],
            ["crawl_budget_usage_pct", "Product usage / Site Audit", "Pages crawled, monthly limits, audit settings"],
            ["keyword_tracking_usage_pct", "Position Tracking / subscription limits", "Tracked keywords, locations, devices, keyword allowance"],
            ["ai_visibility_prompt_usage_pct", "AI Visibility Toolkit", "Prompts tracked, prompt limit, prompt research usage"],
            ["tickets_last_quarter / csat_support", "Customer Support platform", "Support volume, reopenings, resolution time, CSAT"],
            ["feedback_count_90d / roadmap_alignment_score", "Feedback / Product Ops", "Feature requests, unresolved feedback, roadmap status"],
            ["stuck_bugs_count / data_accuracy_bug_flag", "Bug tracker / Support escalation", "Open bugs, severity, data trust issues, workaround"],
            ["renewal_due_days / renewal_intent", "CRM / CS platform", "Renewal date, forecast, risk notes, opportunity context"],
        ], columns=["Copilot signal", "Likely source", "What to map"])
        st.dataframe(mapping, use_container_width=True, hide_index=True)
    st.write("Model package:", model_package.get("model_type", "Unknown"), "· Training columns:", len(model_columns or []))
