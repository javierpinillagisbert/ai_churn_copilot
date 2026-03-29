print("Empieza la ejecución del script")

import numpy as np
import pandas as pd

# ----------------------------
# Configuración general
# ----------------------------
np.random.seed(42)
N = 2500  # número de cuentas

# ----------------------------
# Funciones auxiliares
# ----------------------------
def clip_array(arr, low, high):
    return np.clip(arr, low, high)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ----------------------------
# Variables base
# ----------------------------
segments = np.random.choice(
    ["SMB", "Mid-Market", "Enterprise"],
    size=N,
    p=[0.60, 0.30, 0.10]
)

subscription_type = np.where(
    np.random.rand(N) < 0.65, "annual", "monthly"
)

plan_tier = []
for seg in segments:
    if seg == "SMB":
        plan_tier.append(np.random.choice(["Pro", "Guru"], p=[0.65, 0.35]))
    elif seg == "Mid-Market":
        plan_tier.append(np.random.choice(["Guru", "Business"], p=[0.50, 0.50]))
    else:
        plan_tier.append(np.random.choice(["Business", "Enterprise"], p=[0.35, 0.65]))
plan_tier = np.array(plan_tier)

tenure_months = []
for seg in segments:
    if seg == "SMB":
        tenure_months.append(np.random.randint(1, 37))
    elif seg == "Mid-Market":
        tenure_months.append(np.random.randint(3, 49))
    else:
        tenure_months.append(np.random.randint(6, 61))
tenure_months = np.array(tenure_months)

# MRR por plan
mrr = []
for seg, plan in zip(segments, plan_tier):
    if plan == "Pro":
        base = np.random.normal(150, 40)
    elif plan == "Guru":
        base = np.random.normal(400, 120)
    elif plan == "Business":
        base = np.random.normal(1500, 500)
    else:
        base = np.random.normal(7000, 2500)

    if seg == "Enterprise":
        base *= np.random.uniform(1.0, 1.4)
    mrr.append(max(50, base))
mrr = np.round(mrr, 2)

bundles_contracted = []
for plan in plan_tier:
    if plan == "Pro":
        bundles_contracted.append(np.random.randint(1, 3))
    elif plan == "Guru":
        bundles_contracted.append(np.random.randint(1, 4))
    elif plan == "Business":
        bundles_contracted.append(np.random.randint(2, 5))
    else:
        bundles_contracted.append(np.random.randint(3, 7))
bundles_contracted = np.array(bundles_contracted)

extra_users = []
for seg in segments:
    if seg == "SMB":
        extra_users.append(np.random.poisson(1))
    elif seg == "Mid-Market":
        extra_users.append(np.random.poisson(4))
    else:
        extra_users.append(np.random.poisson(12))
extra_users = clip_array(np.array(extra_users), 0, 50)

renewal_due_days = np.random.randint(0, 181, size=N)

# ----------------------------
# Engagement y adopción
# ----------------------------
login_base = []
for seg, plan in zip(segments, plan_tier):
    if seg == "SMB":
        val = np.random.normal(18, 8)
    elif seg == "Mid-Market":
        val = np.random.normal(35, 12)
    else:
        val = np.random.normal(55, 18)

    if plan in ["Business", "Enterprise"]:
        val += np.random.normal(5, 3)
    login_base.append(val)

login_count_30d = clip_array(np.round(login_base), 0, 120).astype(int)

days_since_last_login = clip_array(
    np.round(np.random.normal(12, 10, N) - (login_count_30d / 8)),
    0,
    60
).astype(int)

key_reports_used_30d = clip_array(
    np.round(login_count_30d * np.random.uniform(0.15, 0.45, size=N) + np.random.normal(0, 2, size=N)),
    0,
    40
).astype(int)

tool_activity_score = (
    0.45 * (login_count_30d / 120) * 100
    + 0.30 * (key_reports_used_30d / 40) * 100
    + 0.15 * (bundles_contracted / 6) * 100
    + 0.10 * (extra_users / 50) * 100
)
tool_activity_score += np.random.normal(0, 8, size=N)
tool_activity_score = clip_array(np.round(tool_activity_score, 1), 0, 100)

usage_change_vs_prev_quarter = np.random.normal(0, 18, size=N)
usage_change_vs_prev_quarter += np.where(tool_activity_score > 70, 8, 0)
usage_change_vs_prev_quarter += np.where(tool_activity_score < 35, -15, 0)
usage_change_vs_prev_quarter = clip_array(np.round(usage_change_vs_prev_quarter, 1), -80, 50)

feature_adoption_score = (
    0.40 * tool_activity_score
    + 0.20 * (bundles_contracted / 6) * 100
    + 0.15 * (extra_users / 50) * 100
    + 0.10 * np.minimum(tenure_months / 24, 1.0) * 100
    + np.random.normal(0, 10, size=N)
)
feature_adoption_score = clip_array(np.round(feature_adoption_score, 1), 0, 100)

# ----------------------------
# Relación con CSM
# ----------------------------
csm_meetings_last_quarter = []
for seg in segments:
    if seg == "SMB":
        csm_meetings_last_quarter.append(np.random.poisson(0.5))
    elif seg == "Mid-Market":
        csm_meetings_last_quarter.append(np.random.poisson(1.5))
    else:
        csm_meetings_last_quarter.append(np.random.poisson(3.0))
csm_meetings_last_quarter = clip_array(np.array(csm_meetings_last_quarter), 0, 6)

emails_with_csm_30d = (
    csm_meetings_last_quarter * np.random.randint(1, 5, size=N)
    + np.random.poisson(2, size=N)
)
emails_with_csm_30d = clip_array(emails_with_csm_30d, 0, 25)

strategic_review_done = np.where(
    (segments != "SMB") & (np.random.rand(N) < 0.65), 1, 0
)
strategic_review_done = np.where(
    (segments == "SMB") & (np.random.rand(N) < 0.20), 1, strategic_review_done
)

success_plan_active = np.where(
    (tool_activity_score < 55) | (segments == "Enterprise"),
    np.where(np.random.rand(N) < 0.55, 1, 0),
    np.where(np.random.rand(N) < 0.25, 1, 0)
)

# ----------------------------
# Soporte y experiencia
# ----------------------------
tickets_last_quarter = np.random.poisson(2.5, size=N)
tickets_last_quarter += np.where(tool_activity_score < 35, 2, 0)
tickets_last_quarter = clip_array(tickets_last_quarter, 0, 20)

reopened_tickets = np.random.binomial(
    n=np.maximum(tickets_last_quarter, 1),
    p=clip_array(0.08 + (tickets_last_quarter / 60), 0.05, 0.45)
)
reopened_tickets = clip_array(reopened_tickets, 0, 8)

avg_resolution_days = (
    np.random.normal(3.5, 1.5, size=N)
    + reopened_tickets * 0.9
    + np.where(tickets_last_quarter > 8, 1.2, 0)
)
avg_resolution_days = clip_array(np.round(avg_resolution_days, 1), 0.5, 20)

csat_support = (
    4.5
    - (avg_resolution_days * 0.12)
    - (reopened_tickets * 0.18)
    + np.random.normal(0, 0.35, size=N)
)
csat_support = clip_array(np.round(csat_support, 1), 1.0, 5.0)

# ----------------------------
# Sentiment
# ----------------------------
sentiment_support = (
    0.55 * ((csat_support - 3) / 2)
    - 0.12 * reopened_tickets
    - 0.03 * avg_resolution_days
    + np.random.normal(0, 0.18, size=N)
)
sentiment_support = clip_array(np.round(sentiment_support, 2), -1.0, 1.0)

sentiment_csm = (
    0.30 * ((feature_adoption_score - 50) / 50)
    + 0.20 * ((tool_activity_score - 50) / 50)
    + 0.15 * strategic_review_done
    + 0.10 * success_plan_active
    + 0.10 * np.where(csm_meetings_last_quarter > 0, 1, -0.2)
    + np.random.normal(0, 0.2, size=N)
)
sentiment_csm = clip_array(np.round(sentiment_csm, 2), -1.0, 1.0)

# ----------------------------
# Renovación e intención
# ----------------------------
renewal_intent = []
for i in range(N):
    score = 0
    score += 1 if tool_activity_score[i] > 65 else 0
    score += 1 if feature_adoption_score[i] > 60 else 0
    score += 1 if csat_support[i] > 4 else 0
    score += 1 if sentiment_csm[i] > 0.2 else 0
    score -= 1 if usage_change_vs_prev_quarter[i] < -20 else 0
    score -= 1 if reopened_tickets[i] >= 2 else 0
    score -= 1 if csat_support[i] < 3 else 0

    if score >= 2:
        renewal_intent.append(np.random.choice(["positive", "neutral"], p=[0.8, 0.2]))
    elif score <= -1:
        renewal_intent.append(np.random.choice(["negative", "neutral"], p=[0.75, 0.25]))
    else:
        renewal_intent.append(np.random.choice(["positive", "neutral", "negative"], p=[0.25, 0.5, 0.25]))
renewal_intent = np.array(renewal_intent)

renewed_last_quarter = np.where(
    (renewal_due_days < 30) & (renewal_intent == "positive") & (np.random.rand(N) < 0.75),
    1,
    0
)

in_grace_period = np.where(
    (renewal_due_days < 20) & (renewed_last_quarter == 0) & (np.random.rand(N) < 0.20),
    1,
    0
)

subscription_status = []
for i in range(N):
    if renewed_last_quarter[i] == 1:
        subscription_status.append("active")
    elif in_grace_period[i] == 1:
        subscription_status.append("renewal_pending")
    else:
        # provisional; se refinará tras calcular churn
        subscription_status.append("active")
subscription_status = np.array(subscription_status)

# ----------------------------
# Riesgo de churn y etiqueta final
# ----------------------------
risk_score = (
    0.9 * np.where(subscription_type == "monthly", 1, 0)
    + 1.5 * np.where(renewal_intent == "negative", 1, 0)
    + 0.5 * np.where(renewal_intent == "neutral", 1, 0)
    + 1.2 * np.where(renewed_last_quarter == 0, 1, -0.6)
    + 1.0 * np.where(in_grace_period == 0, 0.2, -0.4)
    + 1.3 * np.where(tool_activity_score < 35, 1, 0)
    + 1.0 * np.where(feature_adoption_score < 40, 1, 0)
    + 1.0 * np.where(usage_change_vs_prev_quarter < -20, 1, 0)
    + 0.8 * np.where(days_since_last_login > 20, 1, 0)
    + 0.8 * np.where(csat_support < 3.2, 1, 0)
    + 0.9 * np.where(sentiment_csm < -0.2, 1, 0)
    + 0.6 * np.where(sentiment_support < -0.2, 1, 0)
    + 0.8 * np.where(reopened_tickets >= 2, 1, 0)
    - 0.7 * strategic_review_done
    - 0.6 * success_plan_active
    - 0.3 * np.where(tenure_months > 18, 1, 0)
    - 0.3 * np.where(bundles_contracted >= 3, 1, 0)
)

# Ajuste de base para llevar churn a un rango realista
risk_score = risk_score - 2.7

churn_probability = sigmoid(risk_score)
churn_label = np.random.binomial(1, churn_probability, size=N)

# Refinar subscription_status para reflejar churn realista
subscription_status = np.where(
    churn_label == 1,
    np.where((renewed_last_quarter == 0) & (in_grace_period == 0), "churned", "renewal_pending"),
    np.where(renewed_last_quarter == 1, "active", subscription_status)
)

# Regla final para evitar falsos churns excesivos
churn_label = np.where(
    (subscription_status == "churned") &
    (renewed_last_quarter == 0) &
    (renewal_intent == "negative") &
    (in_grace_period == 0),
    1,
    0
)

# ----------------------------
# DataFrame final
# ----------------------------
df = pd.DataFrame({
    "account_id": [f"ACC_{i:05d}" for i in range(1, N + 1)],
    "segment": segments,
    "subscription_type": subscription_type,
    "plan_tier": plan_tier,
    "tenure_months": tenure_months,
    "mrr": mrr,
    "bundles_contracted": bundles_contracted,
    "extra_users": extra_users,
    "renewal_due_days": renewal_due_days,
    "subscription_status": subscription_status,
    "renewed_last_quarter": renewed_last_quarter,
    "renewal_intent": renewal_intent,
    "in_grace_period": in_grace_period,
    "login_count_30d": login_count_30d,
    "tool_activity_score": tool_activity_score,
    "usage_change_vs_prev_quarter": usage_change_vs_prev_quarter,
    "feature_adoption_score": feature_adoption_score,
    "key_reports_used_30d": key_reports_used_30d,
    "days_since_last_login": days_since_last_login,
    "emails_with_csm_30d": emails_with_csm_30d,
    "csm_meetings_last_quarter": csm_meetings_last_quarter,
    "strategic_review_done": strategic_review_done,
    "success_plan_active": success_plan_active,
    "tickets_last_quarter": tickets_last_quarter,
    "reopened_tickets": reopened_tickets,
    "avg_resolution_days": avg_resolution_days,
    "csat_support": csat_support,
    "sentiment_csm": sentiment_csm,
    "sentiment_support": sentiment_support,
    "churn_label": churn_label
})

# ----------------------------
# Métricas agregadas de negocio
# ----------------------------
total_accounts = len(df)
churn_rate = df["churn_label"].mean()
retention_rate = ((df["renewed_last_quarter"] == 1) | (df["subscription_status"] == "active")).mean()

print("Número de cuentas:", total_accounts)
print("Churn rate simulado:", round(churn_rate, 4))
print("Retention rate simulada:", round(retention_rate, 4))
print("\nDistribución por segmento:")
print(df["segment"].value_counts(normalize=True).round(3))
print("\nDistribución churn:")
print(df["churn_label"].value_counts(normalize=True).round(3))

# Exportar a CSV
df.to_csv("synthetic_saas_churn_dataset.csv", index=False)
print("\nArchivo exportado como: synthetic_saas_churn_dataset.csv")

print("Termina la ejecución del script")