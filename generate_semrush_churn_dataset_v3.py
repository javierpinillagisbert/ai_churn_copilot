
"""
Generate a Semrush-like synthetic churn dataset for the AI Churn Copilot V3.

This is synthetic demo data inspired by publicly documented Semrush toolkits,
limits and workflows. It does not contain any internal Semrush customer data.
"""

from pathlib import Path
import numpy as np
import pandas as pd

RNG_SEED = 42

INDUSTRIES = [
    "Retail & Ecommerce", "SaaS & Technology", "Agency", "Healthcare", "Travel",
    "Education", "Finance", "Real Estate", "Automotive", "Media & Publishing",
]
REGIONS = ["North America", "EMEA", "LATAM", "APAC"]
SEGMENTS = ["SMB", "Mid-Market", "Enterprise", "Agency"]
SUB_TYPES = ["monthly", "annual", "custom contract"]
RENEWAL_INTENTS = ["positive", "neutral", "negative"]
SUB_STATUSES = ["active", "renewal_pending", "grace_period", "downgrade_discussion"]

PLAN_OPTIONS = [
    ("SEO Toolkit", "Pro"),
    ("SEO Toolkit", "Guru"),
    ("SEO Toolkit", "Business"),
    ("Semrush One", "Starter"),
    ("Semrush One", "Pro+"),
    ("Semrush One", "Advanced"),
    ("AI Visibility Toolkit", "Standalone"),
    ("Local Toolkit", "Base"),
    ("Local Toolkit", "Pro"),
    ("Content Toolkit", "Standard"),
    ("Traffic & Market Toolkit", "Standard"),
]

PLAN_LIMITS = {
    ("SEO Toolkit", "Pro"): {"websites": 5, "keywords": 500, "crawl": 100000, "results": 10000, "prompts": 0, "api": 0, "sov": 0, "historical": 0},
    ("SEO Toolkit", "Guru"): {"websites": 15, "keywords": 1500, "crawl": 300000, "results": 30000, "prompts": 0, "api": 0, "sov": 0, "historical": 1},
    ("SEO Toolkit", "Business"): {"websites": 40, "keywords": 5000, "crawl": 1000000, "results": 50000, "prompts": 0, "api": 1, "sov": 1, "historical": 1},
    ("Semrush One", "Starter"): {"websites": 5, "keywords": 500, "crawl": 100000, "results": 10000, "prompts": 50, "api": 0, "sov": 0, "historical": 0},
    ("Semrush One", "Pro+"): {"websites": 15, "keywords": 1500, "crawl": 300000, "results": 30000, "prompts": 100, "api": 0, "sov": 0, "historical": 1},
    ("Semrush One", "Advanced"): {"websites": 40, "keywords": 5000, "crawl": 1000000, "results": 50000, "prompts": 200, "api": 1, "sov": 1, "historical": 1},
    ("AI Visibility Toolkit", "Standalone"): {"websites": 0, "keywords": 0, "crawl": 0, "results": 0, "prompts": 25, "api": 0, "sov": 0, "historical": 0},
    ("Local Toolkit", "Base"): {"websites": 0, "keywords": 0, "crawl": 0, "results": 0, "prompts": 0, "api": 0, "sov": 0, "historical": 0},
    ("Local Toolkit", "Pro"): {"websites": 0, "keywords": 0, "crawl": 0, "results": 0, "prompts": 0, "api": 1, "sov": 0, "historical": 1},
    ("Content Toolkit", "Standard"): {"websites": 5, "keywords": 500, "crawl": 100000, "results": 10000, "prompts": 0, "api": 0, "sov": 0, "historical": 0},
    ("Traffic & Market Toolkit", "Standard"): {"websites": 0, "keywords": 0, "crawl": 0, "results": 50000, "prompts": 0, "api": 0, "sov": 0, "historical": 1},
}


def pct(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return min(100.0, round((numerator / denominator) * 100, 1))


def clipped_normal(rng, mean, sd, low, high):
    return float(np.clip(rng.normal(mean, sd), low, high))


def choose_plan(rng, segment):
    if segment == "Enterprise":
        probs = np.array([0.04, 0.10, 0.22, 0.03, 0.10, 0.20, 0.10, 0.02, 0.08, 0.04, 0.07])
    elif segment == "Mid-Market":
        probs = np.array([0.10, 0.22, 0.18, 0.08, 0.13, 0.09, 0.06, 0.03, 0.05, 0.03, 0.03])
    elif segment == "Agency":
        probs = np.array([0.08, 0.18, 0.28, 0.04, 0.08, 0.12, 0.05, 0.02, 0.05, 0.03, 0.07])
    else:
        probs = np.array([0.25, 0.18, 0.05, 0.16, 0.08, 0.03, 0.08, 0.06, 0.04, 0.05, 0.02])
    probs = probs / probs.sum()
    return PLAN_OPTIONS[rng.choice(len(PLAN_OPTIONS), p=probs)]


def generate_dataset(n_accounts=3500, output_path="synthetic_semrush_churn_dataset_v3.csv", seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    account_prefixes = ["Bright", "Northstar", "Summit", "BlueRiver", "Atlas", "GreenPeak", "Nova", "MarketWave", "Signal", "Orbit", "Cedar", "Silverline", "Vertex", "GrowthLab", "BrandEdge"]
    account_suffixes = ["Media", "Digital", "Group", "Labs", "Studios", "SEO", "Partners", "Analytics", "Global", "Marketing", "Health", "Commerce", "Solutions"]

    for i in range(n_accounts):
        segment = rng.choice(SEGMENTS, p=[0.40, 0.29, 0.16, 0.15])
        plan_family, plan_tier = choose_plan(rng, segment)
        limits = PLAN_LIMITS[(plan_family, plan_tier)]
        industry = rng.choice(INDUSTRIES)
        region = rng.choice(REGIONS, p=[0.36, 0.38, 0.10, 0.16])
        customer_type = "Agency" if segment == "Agency" else rng.choice(["Brand", "In-house team", "Consultant"], p=[0.62, 0.30, 0.08])
        agency_flag = int(customer_type == "Agency")
        multi_location_flag = int((industry in ["Retail & Ecommerce", "Healthcare", "Real Estate", "Automotive"] and rng.random() < 0.42) or plan_family == "Local Toolkit")
        subscription_type = rng.choice(SUB_TYPES, p=[0.42, 0.50, 0.08]) if segment != "Enterprise" else rng.choice(SUB_TYPES, p=[0.10, 0.45, 0.45])
        tenure_months = int(np.clip(rng.gamma(3.2, 9.5), 1, 96))

        base_mrr = {
            "SMB": rng.normal(220, 85), "Mid-Market": rng.normal(720, 260), "Enterprise": rng.normal(3500, 1350), "Agency": rng.normal(1100, 500)
        }[segment]
        tier_multiplier = {"Pro": 1.0, "Guru": 1.75, "Business": 3.6, "Starter": 1.1, "Pro+": 2.0, "Advanced": 4.1, "Standalone": 0.8, "Base": 0.7, "Standard": 1.2}.get(plan_tier, 1.0)
        mrr = max(79, round(base_mrr * tier_multiplier + rng.normal(0, 70), 0))
        arr = round(mrr * 12, 0)

        number_of_users = int(np.clip(rng.poisson({"SMB": 2.5, "Mid-Market": 5.5, "Enterprise": 14, "Agency": 8}.get(segment, 3)) + 1, 1, 80))
        active_user_ratio = np.clip(rng.normal(0.68, 0.22), 0.05, 1)
        active_users_30d = int(max(0, min(number_of_users, round(number_of_users * active_user_ratio))))

        seo_active = int(plan_family in ["SEO Toolkit", "Semrush One", "Content Toolkit"] or rng.random() < 0.25)
        traffic_active = int(plan_family == "Traffic & Market Toolkit" or (segment in ["Mid-Market", "Enterprise", "Agency"] and rng.random() < 0.25))
        ai_active = int(plan_family in ["AI Visibility Toolkit", "Semrush One"] or rng.random() < 0.18)
        local_active = int(plan_family == "Local Toolkit" or (multi_location_flag and rng.random() < 0.30))
        content_active = int(plan_family in ["Content Toolkit", "Semrush One"] or rng.random() < 0.22)
        app_center_usage_flag = int(rng.random() < (0.10 + 0.10 * (segment in ["Enterprise", "Agency"])))

        websites_limit = limits["websites"]
        keywords_limit = limits["keywords"]
        crawl_limit = limits["crawl"]
        results_limit = limits["results"]
        prompt_limit = limits["prompts"]

        if seo_active and websites_limit > 0:
            websites_monitored = int(np.clip(rng.poisson(max(1, websites_limit * rng.uniform(0.20, 0.70))), 0, websites_limit + 5))
            keywords_tracked = int(np.clip(rng.normal(keywords_limit * rng.uniform(0.28, 0.82), keywords_limit * 0.15), 0, keywords_limit * 1.10))
            pages_crawled_month = int(np.clip(rng.normal(crawl_limit * rng.uniform(0.20, 0.88), crawl_limit * 0.12), 0, crawl_limit * 1.10))
            results_per_report_used = int(np.clip(rng.normal(results_limit * rng.uniform(0.12, 0.60), results_limit * 0.10), 0, results_limit))
            keyword_metric_updates_limit = int(keywords_limit * rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20]))
            keyword_metric_updates_used = int(np.clip(rng.normal(keyword_metric_updates_limit * rng.uniform(0.18, 0.74), keyword_metric_updates_limit * 0.12), 0, keyword_metric_updates_limit * 1.05))
        else:
            websites_monitored = 0; keywords_tracked = 0; pages_crawled_month = 0; results_per_report_used = 0
            keyword_metric_updates_limit = 0; keyword_metric_updates_used = 0

        websites_usage_pct = pct(websites_monitored, websites_limit)
        keyword_tracking_usage_pct = pct(keywords_tracked, keywords_limit)
        crawl_budget_usage_pct = pct(pages_crawled_month, crawl_limit)
        results_per_report_usage_pct = pct(results_per_report_used, results_limit)
        keyword_metric_updates_usage_pct = pct(keyword_metric_updates_used, keyword_metric_updates_limit)

        api_access_available = limits["api"]
        api_access_used = int(api_access_available and rng.random() < (0.50 if segment in ["Enterprise", "Agency"] else 0.25))
        historical_data_available = limits["historical"]
        historical_data_used = int(historical_data_available and rng.random() < 0.58)
        share_of_voice_available = limits["sov"]
        share_of_voice_used = int(share_of_voice_available and rng.random() < 0.42)

        site_audit_projects_active = int(np.clip(rng.poisson(1 + websites_monitored * 0.35), 0, max(1, websites_monitored + 2))) if seo_active else 0
        site_audit_last_run_days = int(np.clip(rng.gamma(2.4, 7.0), 0, 120)) if site_audit_projects_active else 999
        site_audit_health_score_avg = round(clipped_normal(rng, 78 - 18 * (site_audit_last_run_days > 45), 13, 20, 100), 1) if site_audit_projects_active else 0
        site_audit_blocking_issues_count = int(np.clip(rng.poisson(max(0.2, (100 - site_audit_health_score_avg)/12)), 0, 20)) if site_audit_projects_active else 0
        position_tracking_campaigns_active = int(np.clip(rng.poisson(1 + websites_monitored * 0.25), 0, max(1, websites_monitored + 1))) if seo_active else 0
        position_tracking_keywords_active = int(keywords_tracked * rng.uniform(0.65, 1.0)) if position_tracking_campaigns_active else 0
        position_tracking_multitargeting_used = int(position_tracking_campaigns_active > 1 and rng.random() < (0.55 if plan_tier in ["Guru", "Business", "Pro+", "Advanced"] else 0.15))
        keyword_magic_queries_30d = int(np.clip(rng.poisson(18 if seo_active else 3), 0, 300))
        backlink_reports_used_30d = int(np.clip(rng.poisson(8 if seo_active else 2), 0, 120))
        competitor_gap_reports_used_30d = int(np.clip(rng.poisson(6 if seo_active else 1), 0, 100))
        my_reports_used_30d = int(np.clip(rng.poisson(5 if segment in ["Agency", "Enterprise"] else 2), 0, 80))
        lookerstudio_integration_used = int(rng.random() < (0.28 if segment in ["Enterprise", "Agency"] else 0.09))

        # AI Visibility usage
        if ai_active:
            ai_visibility_prompt_limit = prompt_limit if prompt_limit > 0 else 25
            ai_visibility_prompts_tracked = int(np.clip(rng.normal(ai_visibility_prompt_limit * rng.uniform(0.18, 0.72), ai_visibility_prompt_limit * 0.15), 0, ai_visibility_prompt_limit))
            ai_visibility_daily_query_limit = 300
            ai_visibility_daily_queries_used = int(np.clip(rng.normal(ai_visibility_daily_query_limit * rng.uniform(0.12, 0.75), 45), 0, ai_visibility_daily_query_limit))
            prompt_research_query_limit = 1000
            prompt_research_queries_used = int(np.clip(rng.normal(prompt_research_query_limit * rng.uniform(0.05, 0.55), 110), 0, prompt_research_query_limit))
            ai_visibility_exports_limit = 10
            ai_visibility_exports_used = int(np.clip(rng.poisson(2.5), 0, 10))
            ai_search_checks_limit = 100
            ai_search_checks_used = int(np.clip(rng.normal(ai_search_checks_limit * rng.uniform(0.05, 0.50), 12), 0, ai_search_checks_limit))
            brand_performance_reports_used = int(np.clip(rng.poisson(3.5), 0, 50))
            competitor_ai_visibility_reports_used = int(np.clip(rng.poisson(2.2), 0, 50))
            ai_visibility_score_tracked = round(clipped_normal(rng, 52, 18, 0, 100), 1)
        else:
            ai_visibility_prompt_limit = 0; ai_visibility_prompts_tracked = 0; ai_visibility_daily_query_limit = 0; ai_visibility_daily_queries_used = 0
            prompt_research_query_limit = 0; prompt_research_queries_used = 0; ai_visibility_exports_limit = 0; ai_visibility_exports_used = 0
            ai_search_checks_limit = 0; ai_search_checks_used = 0; brand_performance_reports_used = 0; competitor_ai_visibility_reports_used = 0; ai_visibility_score_tracked = 0
        ai_visibility_prompt_usage_pct = pct(ai_visibility_prompts_tracked, ai_visibility_prompt_limit)
        ai_visibility_query_usage_pct = pct(ai_visibility_daily_queries_used, ai_visibility_daily_query_limit)
        ai_visibility_exports_usage_pct = pct(ai_visibility_exports_used, ai_visibility_exports_limit)

        # Local usage
        if local_active:
            local_locations_target = int(np.clip(rng.poisson(8 if multi_location_flag else 2) + 1, 1, 350))
            local_locations_active = int(np.clip(rng.normal(local_locations_target * rng.uniform(0.45, 0.95), max(1, local_locations_target * 0.12)), 0, local_locations_target))
            listing_management_active = int(rng.random() < 0.85)
            review_management_active = int(rng.random() < 0.72)
            review_generation_used = int(review_management_active and rng.random() < 0.48)
            local_rank_tracking_active = int(rng.random() < 0.66)
            duplicate_suppression_issues = int(np.clip(rng.poisson(local_locations_active * 0.08), 0, 20))
            frozen_locations_count = int(np.clip(rng.poisson(local_locations_target * rng.uniform(0.00, 0.08)), 0, local_locations_target))
            local_reviews_collected_30d = int(np.clip(rng.poisson(max(1, local_locations_active * rng.uniform(0.7, 4.5))), 0, 5000))
            average_local_rating = round(clipped_normal(rng, 4.15, 0.45, 1.0, 5.0), 2)
        else:
            local_locations_target = 0; local_locations_active = 0; listing_management_active = 0; review_management_active = 0; review_generation_used = 0
            local_rank_tracking_active = 0; duplicate_suppression_issues = 0; frozen_locations_count = 0; local_reviews_collected_30d = 0; average_local_rating = 0

        # Generic engagement scores
        login_count_30d = int(np.clip(rng.poisson(8 + active_users_30d * 3.4), 0, 350))
        days_since_last_login = int(np.clip(rng.exponential(8 + 14 * (active_users_30d == 0)), 0, 120))
        tool_activity_score = round(np.clip(
            20 + 0.13 * min(login_count_30d, 250) + 0.20 * min(websites_usage_pct, 100) + 0.16 * min(keyword_tracking_usage_pct, 100)
            + 0.18 * min(ai_visibility_prompt_usage_pct, 100) + 0.08 * min(local_locations_active * 6, 100) - 0.35 * days_since_last_login + rng.normal(0, 10), 0, 100
        ), 1)
        feature_adoption_score = round(np.clip(
            12 + 10 * seo_active + 9 * traffic_active + 10 * ai_active + 8 * local_active + 7 * content_active
            + 7 * api_access_used + 6 * share_of_voice_used + 5 * lookerstudio_integration_used + 3 * my_reports_used_30d
            + rng.normal(0, 12), 0, 100
        ), 1)
        usage_change_vs_prev_quarter = round(clipped_normal(rng, 2 + (tool_activity_score - 50) / 5, 18, -80, 90), 1)
        key_reports_used_30d = int(np.clip(keyword_magic_queries_30d/10 + backlink_reports_used_30d + competitor_gap_reports_used_30d + my_reports_used_30d + rng.poisson(2), 0, 250))

        # Renewal and commercial context
        renewal_due_days = int(np.clip(rng.integers(5, 365), 5, 365))
        pricing_objection_flag = int(rng.random() < (0.12 + 0.10*(mrr > 2000) + 0.10*(tool_activity_score < 35)))
        limit_upgrade_recommended_flag = int((crawl_budget_usage_pct > 85) or (keyword_tracking_usage_pct > 90) or (ai_visibility_prompt_usage_pct > 85))
        expansion_opportunity_flag = int((tool_activity_score > 70 and feature_adoption_score > 65 and (keyword_tracking_usage_pct > 75 or ai_visibility_prompt_usage_pct > 75 or local_locations_active > 5)) and rng.random() < 0.45)
        downgrade_risk_flag = int(tool_activity_score < 35 and pricing_objection_flag and rng.random() < 0.65)

        # Support/bugs
        base_ticket_lambda = 1.0 + 2.2*(crawl_budget_usage_pct > 85) + 1.8*(keyword_tracking_usage_pct > 90) + 1.8*(site_audit_blocking_issues_count > 5) + 1.5*(local_active and duplicate_suppression_issues > 2)
        tickets_last_quarter = int(np.clip(rng.poisson(base_ticket_lambda), 0, 60))
        technical_tickets_last_quarter = int(np.clip(rng.binomial(tickets_last_quarter, 0.62), 0, tickets_last_quarter))
        billing_tickets_last_quarter = int(tickets_last_quarter - technical_tickets_last_quarter)
        reopened_tickets = int(np.clip(rng.poisson(max(0.05, tickets_last_quarter * 0.12)), 0, tickets_last_quarter))
        avg_resolution_days = round(clipped_normal(rng, 2.8 + 1.2*reopened_tickets + 2.5*(technical_tickets_last_quarter > 4), 2.3, 0.2, 45), 1)
        csat_support = round(np.clip(4.5 - 0.20*reopened_tickets - 0.08*avg_resolution_days - 0.35*(technical_tickets_last_quarter > 6) + rng.normal(0, 0.45), 1, 5), 2)

        bugs_reported_90d = int(np.clip(rng.poisson(0.3 + 0.18*technical_tickets_last_quarter), 0, 30))
        critical_bugs_reported_90d = int(np.clip(rng.binomial(bugs_reported_90d, 0.18), 0, bugs_reported_90d))
        open_bugs_count = int(np.clip(rng.binomial(bugs_reported_90d, 0.35), 0, bugs_reported_90d))
        stuck_bugs_count = int(np.clip(rng.binomial(open_bugs_count, 0.25), 0, open_bugs_count))
        data_accuracy_bug_flag = int(rng.random() < (0.05 + 0.10*(bugs_reported_90d > 2) + 0.12*(site_audit_blocking_issues_count > 8)))
        site_audit_bug_flag = int(seo_active and rng.random() < (0.06 + 0.08*(site_audit_blocking_issues_count > 8)))
        position_tracking_bug_flag = int(position_tracking_campaigns_active > 0 and rng.random() < (0.05 + 0.06*(keyword_tracking_usage_pct > 90)))
        ai_visibility_bug_flag = int(ai_active and rng.random() < (0.05 + 0.08*(ai_visibility_prompt_usage_pct > 90)))
        local_listing_sync_bug_flag = int(local_active and rng.random() < (0.05 + 0.10*(duplicate_suppression_issues > 2)))
        workaround_available_flag = int(rng.random() < max(0.15, 0.78 - 0.22*data_accuracy_bug_flag - 0.12*stuck_bugs_count))
        oldest_open_bug_days = int(np.clip(rng.gamma(2.2, 18) * (1 if open_bugs_count else 0), 0, 220))
        bug_linked_to_renewal_flag = int(open_bugs_count > 0 and renewal_due_days < 75 and rng.random() < (0.10 + 0.20*data_accuracy_bug_flag + 0.10*stuck_bugs_count))
        bug_sentiment_score = round(np.clip(rng.normal(0.2, 0.38) - 0.30*critical_bugs_reported_90d - 0.25*stuck_bugs_count - 0.25*data_accuracy_bug_flag + 0.15*workaround_available_flag, -1, 1), 2)

        # Feedback and roadmap fit
        feedback_count_90d = int(np.clip(rng.poisson(0.8 + 0.02*tool_activity_score + 0.05*number_of_users + 0.7*limit_upgrade_recommended_flag), 0, 35))
        feature_requests_count_90d = int(np.clip(rng.binomial(feedback_count_90d, 0.42), 0, feedback_count_90d))
        data_recalculation_requests_90d = int(np.clip(rng.binomial(feedback_count_90d, 0.20 + 0.15*data_accuracy_bug_flag), 0, feedback_count_90d))
        tool_limit_feedback_count = int(np.clip(rng.binomial(feedback_count_90d, 0.18 + 0.25*limit_upgrade_recommended_flag), 0, feedback_count_90d))
        database_coverage_feedback_count = int(np.clip(rng.binomial(feedback_count_90d, 0.12 + 0.06*(region == "APAC")), 0, feedback_count_90d))
        ai_visibility_feedback_count = int(np.clip(rng.binomial(feedback_count_90d, 0.22 if ai_active else 0.03), 0, feedback_count_90d))
        local_toolkit_feedback_count = int(np.clip(rng.binomial(feedback_count_90d, 0.22 if local_active else 0.03), 0, feedback_count_90d))
        critical_feedback_count_90d = int(np.clip(rng.binomial(feedback_count_90d, 0.12 + 0.12*(renewal_due_days < 75)), 0, feedback_count_90d))
        unresolved_feedback_count = int(np.clip(rng.binomial(feedback_count_90d, 0.34 + 0.16*(critical_feedback_count_90d > 0)), 0, feedback_count_90d))
        avg_unresolved_feedback_age_days = round(np.clip(rng.gamma(2.2, 18) * (1 if unresolved_feedback_count else 0), 0, 220), 1)
        roadmap_alignment_score = round(np.clip(rng.normal(67, 18) - 7*critical_feedback_count_90d - 0.12*avg_unresolved_feedback_age_days - 10*tool_limit_feedback_count, 0, 100), 1)
        feedback_sentiment_score = round(np.clip(rng.normal(0.18, 0.35) - 0.25*critical_feedback_count_90d - 0.18*unresolved_feedback_count - 0.18*(roadmap_alignment_score < 35), -1, 1), 2)
        competitor_mentioned_in_feedback = int(feedback_count_90d > 0 and rng.random() < (0.05 + 0.08*(feedback_sentiment_score < -0.25) + 0.05*pricing_objection_flag))
        feedback_linked_to_renewal = int(feedback_count_90d > 0 and renewal_due_days < 100 and rng.random() < (0.08 + 0.13*(critical_feedback_count_90d > 0) + 0.10*(roadmap_alignment_score < 40)))

        sentiment_csm = round(np.clip(rng.normal(0.25, 0.42) - 0.23*pricing_objection_flag - 0.20*downgrade_risk_flag - 0.18*feedback_linked_to_renewal - 0.16*bug_linked_to_renewal_flag + 0.10*expansion_opportunity_flag, -1, 1), 2)
        sentiment_support = round(np.clip(rng.normal(0.20, 0.40) - 0.15*reopened_tickets - 0.11*avg_resolution_days - 0.20*critical_bugs_reported_90d, -1, 1), 2)
        emails_with_csm_30d = int(np.clip(rng.poisson(2.5 + 2.0*(segment in ["Enterprise", "Agency"]) + 2.0*(renewal_due_days < 60)), 0, 60))
        csm_meetings_last_quarter = int(np.clip(rng.poisson(1.0 + 1.0*(segment in ["Enterprise", "Agency"]) + 1.5*(renewal_due_days < 60)), 0, 20))
        strategic_review_done = int(rng.random() < (0.28 + 0.25*(segment in ["Enterprise", "Agency"]) + 0.15*(csm_meetings_last_quarter > 2)))
        success_plan_active = int(rng.random() < (0.35 + 0.22*(segment in ["Enterprise", "Agency"]) + 0.16*strategic_review_done))

        renewal_base = 0.55 + 0.006*(tool_activity_score-50) + 0.004*(feature_adoption_score-50) + 0.20*success_plan_active + 0.12*strategic_review_done - 0.17*pricing_objection_flag - 0.16*downgrade_risk_flag - 0.15*feedback_linked_to_renewal - 0.18*bug_linked_to_renewal_flag
        if renewal_due_days > 180:
            renewal_base += 0.06
        renewal_base = np.clip(renewal_base, 0.05, 0.95)
        if renewal_base > 0.67:
            renewal_intent = "positive"
        elif renewal_base > 0.40:
            renewal_intent = "neutral"
        else:
            renewal_intent = "negative"
        if rng.random() < 0.07:
            renewal_intent = rng.choice(RENEWAL_INTENTS)
        renewed_last_quarter = int(rng.random() < (0.35 + 0.25*(renewal_intent == "positive") - 0.15*(renewal_intent == "negative")))
        in_grace_period = int(rng.random() < (0.02 + 0.08*(pricing_objection_flag) + 0.06*(renewal_intent == "negative")))
        subscription_status = "active"
        if in_grace_period:
            subscription_status = "grace_period"
        elif renewal_due_days < 45:
            subscription_status = "renewal_pending"
        elif downgrade_risk_flag:
            subscription_status = "downgrade_discussion"

        # Logistic churn probability for synthetic outcome generation
        risk_score = -3.25
        risk_score += 0.030 * max(0, 55 - tool_activity_score)
        risk_score += 0.024 * max(0, 55 - feature_adoption_score)
        risk_score += 0.013 * max(0, -usage_change_vs_prev_quarter)
        risk_score += 0.018 * max(0, days_since_last_login - 18)
        risk_score += 0.18 * (renewal_due_days < 45)
        risk_score += 0.68 * (renewal_intent == "negative") + 0.28 * (renewal_intent == "neutral")
        risk_score += 0.55 * in_grace_period + 0.32 * pricing_objection_flag + 0.45 * downgrade_risk_flag
        risk_score += 0.18 * reopened_tickets + 0.06 * avg_resolution_days + 0.42 * max(0, 3.5 - csat_support)
        risk_score += 0.35 * max(0, -sentiment_csm) + 0.25 * max(0, -sentiment_support)
        risk_score += 0.12 * unresolved_feedback_count + 0.26 * critical_feedback_count_90d + 0.008 * avg_unresolved_feedback_age_days
        risk_score += 0.014 * max(0, 45 - roadmap_alignment_score)
        risk_score += 0.42 * competitor_mentioned_in_feedback + 0.48 * feedback_linked_to_renewal
        risk_score += 0.22 * critical_bugs_reported_90d + 0.32 * stuck_bugs_count + 0.009 * oldest_open_bug_days
        risk_score += 0.55 * data_accuracy_bug_flag + 0.48 * bug_linked_to_renewal_flag - 0.22 * workaround_available_flag
        risk_score += 0.20 * (crawl_budget_usage_pct > 90) + 0.18 * (keyword_tracking_usage_pct > 90) + 0.15 * (ai_visibility_prompt_usage_pct > 90)
        risk_score -= 0.25 * success_plan_active + 0.20 * strategic_review_done + 0.18 * expansion_opportunity_flag
        risk_score += rng.normal(0, 0.35)
        churn_probability_true = 1 / (1 + np.exp(-risk_score))
        churn_label = int(rng.random() < churn_probability_true)

        account_name = f"{rng.choice(account_prefixes)} {rng.choice(account_suffixes)}"
        row = {
            "account_id": f"SEM-{i+1:05d}",
            "account_name": account_name,
            "segment": segment,
            "industry": industry,
            "region": region,
            "customer_type": customer_type,
            "agency_flag": agency_flag,
            "multi_location_business_flag": multi_location_flag,
            "tenure_months": tenure_months,
            "mrr": float(mrr),
            "arr": float(arr),
            "plan_family": plan_family,
            "plan_tier": plan_tier,
            "subscription_type": subscription_type,
            "number_of_users": number_of_users,
            "active_users_30d": active_users_30d,
            "seo_toolkit_active": seo_active,
            "traffic_market_toolkit_active": traffic_active,
            "ai_visibility_active": ai_active,
            "local_toolkit_active": local_active,
            "content_toolkit_active": content_active,
            "app_center_usage_flag": app_center_usage_flag,
            "api_access_available": api_access_available,
            "api_access_used": api_access_used,
            "historical_data_available": historical_data_available,
            "historical_data_used": historical_data_used,
            "share_of_voice_available": share_of_voice_available,
            "share_of_voice_used": share_of_voice_used,
            "websites_monitored": websites_monitored,
            "websites_limit": websites_limit,
            "websites_usage_pct": websites_usage_pct,
            "keywords_tracked": keywords_tracked,
            "keywords_limit": keywords_limit,
            "keyword_tracking_usage_pct": keyword_tracking_usage_pct,
            "pages_crawled_month": pages_crawled_month,
            "pages_crawl_limit": crawl_limit,
            "crawl_budget_usage_pct": crawl_budget_usage_pct,
            "results_per_report_used": results_per_report_used,
            "results_per_report_limit": results_limit,
            "results_per_report_usage_pct": results_per_report_usage_pct,
            "keyword_metric_updates_used": keyword_metric_updates_used,
            "keyword_metric_updates_limit": keyword_metric_updates_limit,
            "keyword_metric_updates_usage_pct": keyword_metric_updates_usage_pct,
            "site_audit_projects_active": site_audit_projects_active,
            "site_audit_last_run_days": site_audit_last_run_days,
            "site_audit_health_score_avg": site_audit_health_score_avg,
            "site_audit_blocking_issues_count": site_audit_blocking_issues_count,
            "position_tracking_campaigns_active": position_tracking_campaigns_active,
            "position_tracking_keywords_active": position_tracking_keywords_active,
            "position_tracking_multitargeting_used": position_tracking_multitargeting_used,
            "keyword_magic_queries_30d": keyword_magic_queries_30d,
            "backlink_reports_used_30d": backlink_reports_used_30d,
            "competitor_gap_reports_used_30d": competitor_gap_reports_used_30d,
            "my_reports_used_30d": my_reports_used_30d,
            "lookerstudio_integration_used": lookerstudio_integration_used,
            "ai_visibility_score_tracked": ai_visibility_score_tracked,
            "ai_visibility_prompts_tracked": ai_visibility_prompts_tracked,
            "ai_visibility_prompt_limit": ai_visibility_prompt_limit,
            "ai_visibility_prompt_usage_pct": ai_visibility_prompt_usage_pct,
            "ai_visibility_daily_queries_used": ai_visibility_daily_queries_used,
            "ai_visibility_daily_query_limit": ai_visibility_daily_query_limit,
            "ai_visibility_query_usage_pct": ai_visibility_query_usage_pct,
            "prompt_research_queries_used": prompt_research_queries_used,
            "prompt_research_query_limit": prompt_research_query_limit,
            "ai_visibility_exports_used": ai_visibility_exports_used,
            "ai_visibility_exports_limit": ai_visibility_exports_limit,
            "ai_visibility_exports_usage_pct": ai_visibility_exports_usage_pct,
            "ai_search_checks_used": ai_search_checks_used,
            "ai_search_checks_limit": ai_search_checks_limit,
            "brand_performance_reports_used": brand_performance_reports_used,
            "competitor_ai_visibility_reports_used": competitor_ai_visibility_reports_used,
            "local_locations_active": local_locations_active,
            "local_locations_target": local_locations_target,
            "listing_management_active": listing_management_active,
            "review_management_active": review_management_active,
            "review_generation_used": review_generation_used,
            "local_rank_tracking_active": local_rank_tracking_active,
            "duplicate_suppression_issues": duplicate_suppression_issues,
            "frozen_locations_count": frozen_locations_count,
            "local_reviews_collected_30d": local_reviews_collected_30d,
            "average_local_rating": average_local_rating,
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
            "renewal_due_days": renewal_due_days,
            "renewal_intent": renewal_intent,
            "renewed_last_quarter": renewed_last_quarter,
            "subscription_status": subscription_status,
            "in_grace_period": in_grace_period,
            "expansion_opportunity_flag": expansion_opportunity_flag,
            "downgrade_risk_flag": downgrade_risk_flag,
            "pricing_objection_flag": pricing_objection_flag,
            "limit_upgrade_recommended_flag": limit_upgrade_recommended_flag,
            "tickets_last_quarter": tickets_last_quarter,
            "technical_tickets_last_quarter": technical_tickets_last_quarter,
            "billing_tickets_last_quarter": billing_tickets_last_quarter,
            "reopened_tickets": reopened_tickets,
            "avg_resolution_days": avg_resolution_days,
            "csat_support": csat_support,
            "sentiment_csm": sentiment_csm,
            "sentiment_support": sentiment_support,
            "bugs_reported_90d": bugs_reported_90d,
            "critical_bugs_reported_90d": critical_bugs_reported_90d,
            "open_bugs_count": open_bugs_count,
            "stuck_bugs_count": stuck_bugs_count,
            "avg_bug_resolution_days": round(avg_resolution_days + rng.uniform(0, 8) * (bugs_reported_90d > 0), 1),
            "oldest_open_bug_days": oldest_open_bug_days,
            "bug_reopen_count": reopened_tickets if bugs_reported_90d > 0 else 0,
            "data_accuracy_bug_flag": data_accuracy_bug_flag,
            "site_audit_bug_flag": site_audit_bug_flag,
            "position_tracking_bug_flag": position_tracking_bug_flag,
            "ai_visibility_bug_flag": ai_visibility_bug_flag,
            "local_listing_sync_bug_flag": local_listing_sync_bug_flag,
            "workaround_available_flag": workaround_available_flag,
            "bug_sentiment_score": bug_sentiment_score,
            "bug_linked_to_renewal_flag": bug_linked_to_renewal_flag,
            "feedback_count_90d": feedback_count_90d,
            "feature_requests_count_90d": feature_requests_count_90d,
            "data_recalculation_requests_90d": data_recalculation_requests_90d,
            "tool_limit_feedback_count": tool_limit_feedback_count,
            "database_coverage_feedback_count": database_coverage_feedback_count,
            "ai_visibility_feedback_count": ai_visibility_feedback_count,
            "local_toolkit_feedback_count": local_toolkit_feedback_count,
            "critical_feedback_count_90d": critical_feedback_count_90d,
            "unresolved_feedback_count": unresolved_feedback_count,
            "avg_unresolved_feedback_age_days": avg_unresolved_feedback_age_days,
            "roadmap_alignment_score": roadmap_alignment_score,
            "feedback_sentiment_score": feedback_sentiment_score,
            "competitor_mentioned_in_feedback": competitor_mentioned_in_feedback,
            "feedback_linked_to_renewal": feedback_linked_to_renewal,
            "synthetic_true_churn_probability": round(float(churn_probability_true), 4),
            "churn_label": churn_label,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "synthetic_semrush_churn_dataset_v3.csv"
    df = generate_dataset(output_path=out)
    print(f"Saved {len(df):,} rows and {df.shape[1]} columns to {out}")
    print(f"Synthetic churn rate: {df['churn_label'].mean():.1%}")
