"""
Generate synthetic CSM ownership fields for the Semrush-like churn dataset.

This script enriches synthetic_semrush_churn_dataset_v3.csv with:
- CSM owner
- CSM region/timezone/focus area
- CSM manager
- book tier
- portfolio size

The CSM fields are intended for workflow filtering and book-of-business UX,
not as predictive model inputs.
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_DATA = BASE_DIR / "synthetic_semrush_churn_dataset_v3.csv"
OUTPUT_DATA = BASE_DIR / "synthetic_semrush_churn_dataset_v10_csm.csv"
OUTPUT_ROSTER = BASE_DIR / "csm_roster_semrush_v10.csv"

CSM_ROSTER = [
    ("Emma Rodriguez","EMEA","CET","SEO Toolkit","Senior CSM","Marta Klein","Strategic"),
    ("Daniel Meyer","EMEA","CET","AI Visibility","Senior CSM","Marta Klein","Strategic"),
    ("Sofia Novak","EMEA","CET","Agencies","CSM","Marta Klein","Growth"),
    ("Lucas Martin","EMEA","CET","Local Toolkit","CSM","Marta Klein","Growth"),
    ("Nina Petrovic","EMEA","CET","Traffic & Market Toolkit","Senior CSM","Marta Klein","Strategic"),
    ("Oliver Bennett","EMEA","GMT","SEO Toolkit","CSM","Marta Klein","Growth"),
    ("Aisha Khan","EMEA","GMT","Content Toolkit","CSM","Marta Klein","Scale"),
    ("Clara Dubois","EMEA","CET","Enterprise","Senior CSM","Marta Klein","Strategic"),
    ("Marco Rossi","EMEA","CET","Local Toolkit","CSM","Marta Klein","Growth"),
    ("Lea Schneider","EMEA","CET","AI Visibility","CSM","Marta Klein","Growth"),
    ("Noah Wilson","Americas","EST","SEO Toolkit","Senior CSM","James Carter","Strategic"),
    ("Ava Thompson","Americas","EST","Agencies","CSM","James Carter","Growth"),
    ("Ethan Brooks","Americas","CST","Local Toolkit","CSM","James Carter","Growth"),
    ("Mia Johnson","Americas","PST","AI Visibility","Senior CSM","James Carter","Strategic"),
    ("Liam Anderson","Americas","EST","Traffic & Market Toolkit","CSM","James Carter","Growth"),
    ("Isabella Garcia","Americas","CST","Content Toolkit","CSM","James Carter","Scale"),
    ("Benjamin Lee","Americas","PST","Enterprise","Senior CSM","James Carter","Strategic"),
    ("Harper Davis","Americas","EST","SEO Toolkit","CSM","James Carter","Growth"),
    ("Elena Morales","Americas","CST","Local Toolkit","CSM","James Carter","Growth"),
    ("Samuel Price","Americas","PST","AI Visibility","CSM","James Carter","Growth"),
    ("Yuki Tanaka","APAC","JST","SEO Toolkit","Senior CSM","Priya Nair","Strategic"),
    ("Mei Chen","APAC","SGT","AI Visibility","Senior CSM","Priya Nair","Strategic"),
    ("Arjun Mehta","APAC","SGT","Traffic & Market Toolkit","CSM","Priya Nair","Growth"),
    ("Hana Kim","APAC","KST","Local Toolkit","CSM","Priya Nair","Growth"),
    ("Olivia Nguyen","APAC","AEST","Content Toolkit","CSM","Priya Nair","Scale"),
    ("Kenji Sato","APAC","JST","Enterprise","Senior CSM","Priya Nair","Strategic"),
    ("Rina Patel","APAC","SGT","Agencies","CSM","Priya Nair","Growth"),
    ("Alex Wong","APAC","SGT","SEO Toolkit","CSM","Priya Nair","Growth"),
    ("Grace Lim","APAC","SGT","AI Visibility","CSM","Priya Nair","Growth"),
    ("Tom Walker","APAC","AEST","Local Toolkit","CSM","Priya Nair","Growth"),
    ("Julia Stein","EMEA","CET","Semrush One","Principal CSM","Marta Klein","Strategic"),
    ("Victor Hugo","Americas","EST","Semrush One","Principal CSM","James Carter","Strategic"),
    ("Priya Shah","APAC","SGT","Semrush One","Principal CSM","Priya Nair","Strategic"),
    ("Amelia Hart","EMEA","GMT","Scale Motion","Scale CSM","Marta Klein","Scale"),
    ("Carlos Rivera","Americas","CST","Scale Motion","Scale CSM","James Carter","Scale"),
]


def norm_region(region: str) -> str:
    r = str(region)
    if r in ["EMEA", "Europe", "Middle East", "Africa"] or "EMEA" in r:
        return "EMEA"
    if r in ["Americas", "North America", "LATAM", "USA", "Canada"] or "America" in r:
        return "Americas"
    if r in ["APAC", "Asia Pacific", "Asia", "Oceania"] or "APAC" in r:
        return "APAC"
    return r if r in ["EMEA", "Americas", "APAC"] else "EMEA"


def account_focus(row: pd.Series) -> str:
    if row.get("plan_family") == "Semrush One":
        return "Semrush One"
    if int(row.get("ai_visibility_active", 0)) == 1 or row.get("plan_family") == "AI Visibility Toolkit":
        return "AI Visibility"
    if int(row.get("local_toolkit_active", 0)) == 1 or row.get("plan_family") == "Local Toolkit":
        return "Local Toolkit"
    if row.get("plan_family") == "Traffic & Market Toolkit":
        return "Traffic & Market Toolkit"
    if row.get("plan_family") == "Content Toolkit":
        return "Content Toolkit"
    if row.get("customer_type") == "Agency" or int(row.get("agency_flag", 0)) == 1:
        return "Agencies"
    if row.get("segment") == "Enterprise":
        return "Enterprise"
    return "SEO Toolkit"


def main():
    df = pd.read_csv(INPUT_DATA)
    roster = pd.DataFrame(
        CSM_ROSTER,
        columns=["csm_name", "csm_region", "csm_timezone", "csm_focus_area", "csm_seniority", "csm_manager", "book_tier"],
    )
    roster["target_book_size"] = 100
    roster.to_csv(OUTPUT_ROSTER, index=False)

    loads = {name: 0 for name in roster["csm_name"]}
    assignments = {}
    capacity = 105

    # Enterprise/high-value first, but with a strong balancing cap around 100 accounts per CSM.
    for idx in df.sort_values(["segment", "mrr"], ascending=[True, False]).index:
        row = df.loc[idx]
        region = norm_region(row.get("region"))
        focus = account_focus(row)

        candidates = roster[(roster["csm_region"].eq(region)) & (roster["csm_name"].map(loads) < capacity)].copy()
        if candidates.empty:
            candidates = roster[roster["csm_name"].map(loads) < capacity].copy()
        if candidates.empty:
            candidates = roster.copy()

        scored = []
        for _, csm in candidates.iterrows():
            score = 0
            if csm.csm_region == region:
                score += 5
            if csm.csm_focus_area == focus:
                score += 7
            if focus == "SEO Toolkit" and csm.csm_focus_area in ["SEO Toolkit", "Semrush One"]:
                score += 3
            if focus == "AI Visibility" and csm.csm_focus_area in ["AI Visibility", "Semrush One"]:
                score += 4
            if focus == "Local Toolkit" and csm.csm_focus_area == "Local Toolkit":
                score += 4
            if row.get("segment") == "Enterprise" and csm.book_tier == "Strategic":
                score += 3
            if row.get("segment") == "SMB" and csm.book_tier in ["Scale", "Growth"]:
                score += 2
            if row.get("customer_type") == "Agency" and csm.csm_focus_area == "Agencies":
                score += 5
            score -= loads[csm.csm_name] / 2.5
            scored.append((score, -loads[csm.csm_name], csm.csm_name))

        scored.sort(reverse=True)
        chosen = scored[0][2]
        loads[chosen] += 1
        assignments[idx] = chosen

    out = df.copy()
    out["csm_name"] = out.index.map(assignments)
    out = out.merge(roster, on="csm_name", how="left")
    out["csm_portfolio_size"] = out.groupby("csm_name")["account_id"].transform("count").astype(int)
    out["csm_assigned_account_rank"] = out.groupby("csm_name")["synthetic_true_churn_probability"].rank(method="first", ascending=False).astype(int)

    front = [
        "account_id", "account_name", "csm_name", "csm_region", "csm_timezone", "csm_focus_area",
        "csm_seniority", "csm_manager", "book_tier", "csm_portfolio_size", "csm_assigned_account_rank",
    ]
    out = out[front + [c for c in out.columns if c not in front]]
    out.to_csv(OUTPUT_DATA, index=False)

    print(f"Saved {OUTPUT_DATA.name}: {out.shape[0]} rows, {out.shape[1]} columns")
    print("CSM book size summary:")
    print(out.groupby("csm_name")["account_id"].count().describe())


if __name__ == "__main__":
    main()
